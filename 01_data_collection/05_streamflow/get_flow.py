"""
Gets flow values from NWM S3 bucket, creates site-specific NetCDF files for stramflow.

Adapted from original code by Josh Cunningham (GitHub: @JoshCu)
Written by Quinn Lee and Sonam Lama (GitHub: @quinnylee, @slama0077)
"""

from typing import Optional
import asyncio
import os
import logging
import json
import warnings
from s3fs import S3FileSystem
from s3fs.core import _error_wrapper, version_id_kw
import s3fs
import xarray as xr
from dask.distributed import Client, LocalCluster
from colorama import init

class S3ParallelFileSystem(S3FileSystem):
    """S3FileSystem subclass that supports parallel downloads"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _cat_file(
        self,
        path: str,
        version_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> bytes:
        bucket, key, vers = self.split_path(path)
        version_kw = version_id_kw(version_id or vers)

        # If start/end specified, use single range request
        if start is not None or end is not None:
            head = {"Range": await self._process_limits(path, start, end)}
            return await self._download_chunk(bucket, key, head, version_kw)

        # For large files, use parallel downloads
        try:
            obj_size = (
                await self._call_s3(
                    "head_object", Bucket=bucket, Key=key, **version_kw, **self.req_kw
                )
            )["ContentLength"]
        except Exception:
            # Fall back to single request if HEAD fails
            return await self._download_chunk(bucket, key, {}, version_kw)

        CHUNK_SIZE = 5 * 1024 * 1024  # 1MB chunks
        if obj_size <= CHUNK_SIZE:
            return await self._download_chunk(bucket, key, {}, version_kw)

        # Calculate chunks for parallel download
        chunks = []
        for start in range(0, obj_size, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE - 1, obj_size - 1)
            range_header = f"bytes={start}-{end}"
            chunks.append({"Range": range_header})

        # Download chunks in parallel
        async def download_all_chunks():
            tasks = [
                self._download_chunk(bucket, key, chunk_head, version_kw) for chunk_head in chunks
            ]
            chunks_data = await asyncio.gather(*tasks)
            return b"".join(chunks_data)

        return await _error_wrapper(download_all_chunks, retries=self.retries)

    async def _download_chunk(self, bucket: str, key: str, head: dict, version_kw: dict) -> bytes:
        """Helper function to download a single chunk"""

        async def _call_and_read():
            resp = await self._call_s3(
                "get_object",
                Bucket=bucket,
                Key=key,
                **version_kw,
                **head,
                **self.req_kw,
            )
            try:
                return await resp["Body"].read()
            finally:
                resp["Body"].close()

        return await _error_wrapper(_call_and_read, retries=self.retries)

def read_key_value_file(filepath):
    """Reads all keys and values into one list"""
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    key_value_list = [(key, value) for key, value in data.items()]
    return key_value_list

def download_nwm_output(start_time, end_time, feature_ids) -> xr.Dataset:
    """Load zarr datasets from S3 within the specified time range."""
    # if a LocalCluster is not already running, start one
    try:
        client = Client.current()
    except ValueError:
        cluster = LocalCluster()
        client = Client(cluster)

    logging.debug("Creating s3fs object")
    store = s3fs.S3Map(
        "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
        s3=S3ParallelFileSystem(anon=True),
        # s3=S3FileSystem(anon=True),
    )

    logging.debug("Opening zarr store")
    dataset = xr.open_zarr(store, consolidated=True)

    # select the feature_id
    logging.debug("Selecting feature_id")
    dataset = dataset.sel(time=slice(start_time, end_time), feature_id=feature_ids)

    # drop everything except coordinates feature_id, gage_id, time and variables streamflow
    dataset = dataset[["streamflow"]]
    dataset = dataset.rename({"feature_id": "catchment_id"})
    logging.debug("Computing dataset")
    logging.debug("Dataset: %s", dataset)

    return dataset

def main():
    """Generates the site-specific streamflow NetCDF files"""

    warnings.filterwarnings("ignore", message="No data was returned by the request.")

    # Initialize colorama
    init(autoreset=True)

    logging.basicConfig(level=logging.INFO)

    with open('../02_get_upstream_basins/output/camels_upstream_dict.json',
              'r', 
              encoding="utf-8") as file:
        features_dict = json.load(file)

    for key in list(features_dict.keys()):
        if f"{key}-streamflow.nc" in os.listdir('./uncorrected/'):
            continue

        catchments = [int(key)] + features_dict[key]
        dataset = dataset = download_nwm_output(
            start_time= '1979-10-01',
            end_time= '2023-02-01',
            feature_ids=catchments
            )
        dataset = dataset.drop_vars(['elevation', 'latitude', 'longitude', 'order', 'gage_id'])
        for var in dataset.data_vars:
            dataset[var] = dataset[var].astype("float32")

        try:
            client = Client.current()
        except ValueError:
            cluster = LocalCluster()
            client = Client(cluster)

        dataset.to_netcdf(f"./uncorrected/{key}-streamflow.nc")

if __name__== "__main__":
    main()