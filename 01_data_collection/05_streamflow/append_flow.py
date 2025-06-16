'''
Appends flow values from a Zarr store to site-specific NetCDF files.

Adapted from original code by Josh Cunningham (GitHub: @JoshCu)
Written by Sonam Lama and Quinn Lee (GitHub: @slama0077, @quinnylee)
'''
import s3fs
from s3fs import S3FileSystem
from typing import Optional
from s3fs.core import _error_wrapper, version_id_kw
import asyncio
from dask.distributed import Client, LocalCluster
import logging
import xarray as xr
import os
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from functools import partial
import json
import numpy as np
from tqdm import tqdm
import shutil
import tempfile
from concurrent.futures import as_completed

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
        except Exception as e:
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

def download_nwm_output(start_time, end_time, feature_ids) -> xr.Dataset:
    """Load zarr datasets from S3 within the specified time range."""
    # if a LocalCluster is not already running, start one
    try:
        client = Client.current()
    except ValueError:
        cluster = LocalCluster(dashboard_address=":8787")
        client = Client(cluster)

    logging.debug("Creating s3fs object")
    store = s3fs.S3Map(
        f"s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
        s3=S3ParallelFileSystem(anon=True),
        # s3=S3FileSystem(anon=True),
    )

    logging.debug("Opening zarr store")
    dataset = xr.open_zarr(store, consolidated=True, chunks=None)

    # select the feature_id
    # Ensure feature_ids is a list or array
    logging.debug("Selecting feature_id")
    dataset = dataset.sel(time=slice(start_time, end_time), feature_id=feature_ids)

    # drop everything except coordinates feature_id, gage_id, time and variables streamflow
    dataset = dataset.rename({"feature_id": "catchment_id"})
    dataset = dataset["streamflow"].transpose("catchment_id", "time")
    
    print(dataset)
    logging.debug("Computing dataset")
    logging.debug("Dataset: %s", dataset)

    return dataset

def update_netcdf(basin_file: str, zarr_store: xr.Dataset) -> None:
    '''
    This function opens a pair's NetCDF file, extracts the corresponding
    streamflow time series from the Zarr store, appends the streamflow data,
    and saves the updated NetCDF file.
    '''
    try:
        # Extract site IDs from filename
        # Open the pair's NetCDF file
        ds_nc = xr.open_dataset(f"../03_forcing_gen/outputcamels/{basin_file}/{basin_file}-aggregated.nc", engine="netcdf4", chunks={})
        logging.info(f"Opened NetCDF file for {basin_file}")
        # print("opened nc file")
        # There is a time mismatch between the zarr chrtout file and the netcdf
        # forcings I generated. The zarr file cuts off at 2023-02-01, whereas
        # the netcdf forcings go to 2024-09-30. I don't know why, probably just
        # based on how recently our sources updated.
        subset_ds_nc = ds_nc.isel(time=slice(0, 379897))
                                  
        combined = xr.merge([subset_ds_nc, zarr_store])
        logging.info(f"Merged datasets for {basin_file}")
        logging.info(combined)
        # print("appended data")
        # Save updated file
        # Note to Sonam: we can change this location to {pair_file} and 
        # overwrite the original pair-specific NetCDFs if we are confident in 
        # this process.
        combined.to_netcdf(f"./corrected/{basin_file}.nc", mode="w", engine="netcdf4")
        
        logging.info(f"Saved updated NetCDF file for {basin_file}")

    except Exception as e:
        print(f"Error processing {basin_file}: {e}")

def main():
    logging.basicConfig(level=logging.INFO)

    # # Get list of NetCDF files
    netcdf_files = glob(
        "../03_forcing_gen/outputcamels/*/*-aggregated.nc")
    
    # get all feature ids from json dictionary 
    with open("../02_get_upstream_basins/output/camels_upstream_dict.json", "r") as f:
        feature_id_dict = json.load(f)

    for file in netcdf_files:
        catid = os.path.basename(file).split('-')[0]
        logging.info(f"Processing file: {file} with catchment ID: {catid}")

        ds = xr.open_dataset(file, engine="netcdf4")

        catchment_ids = ds['ids'].values
        catchment_ids = [int(catchment_id) for catchment_id in catchment_ids]

        logging.info("Downloading dataset...")
        zarr_store = download_nwm_output("1979-10-01", "2023-02-01", catchment_ids)
        logging.info(zarr_store)
        # zarr_path = "./camels_sf.zarr"

        logging.info("Saving dataset...")
        update_netcdf(catid, zarr_store)
    # first = True
    # for chunk in tqdm(catchment_chunks, desc="writing chunks", unit="chunk"):
    #     chunked_data = zarr_store.sel(feature_id=chunk)
    #     if first:
    #         # First chunk: create the store
    #         chunked_data.to_zarr(zarr_path, mode="w")
    #         first = False
    #     else:
    #         # Append to store along feature_id
    #         chunked_data.to_zarr(zarr_path, mode="a", append_dim="feature_id")


    # print(feature_ids)
    # # Open Zarr store



    # # # Save Zarr store to local disk
    # logging.info("Saving dataset to local disk...")
    # zarr_store.to_zarr("./camels_sf.zarr", mode="w")

    
    
    # # # Create a partial function that always includes zarr_store
    # update_netcdf_partial = partial(update_netcdf, zarr_store=zarr_store)

    # # # Note to Sonam: we can change the number of workers if we upgrade out of 
    # # # the small instance
    # with ProcessPoolExecutor(max_workers=16) as executor:
    #     print("executing process")
    #     executor.map(update_netcdf_partial, netcdf_files)

if __name__ == "__main__":
    main()
