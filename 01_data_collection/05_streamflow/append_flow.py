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
from dask.distributed import Client, LocalCluster, progress
import logging
import xarray as xr
import os
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from functools import partial

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
        cluster = LocalCluster()
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
    logging.debug("Selecting feature_id")
    dataset = dataset.sel(time=slice(start_time, end_time), feature_id=feature_ids)

    # drop everything except coordinates feature_id, gage_id, time and variables streamflow
    dataset = dataset["streamflow"]
    # print(dataset)
    logging.debug("Computing dataset")
    logging.debug("Dataset: %s", dataset)

    return dataset

def update_netcdf(pair_file: str, zarr_store: xr.Dataset) -> None:
    '''
    This function opens a pair's NetCDF file, extracts the corresponding
    streamflow time series from the Zarr store, appends the streamflow data,
    and saves the updated NetCDF file.
    '''
    try:
        # Extract site IDs from filename
        pair_id = os.path.splitext(os.path.basename(pair_file))[0]
        head_id = int(pair_id.split('-')[0])
        tail_id = int(pair_id.split('-')[1])
        # print("got ids")
        # Open the pair's NetCDF file
        ds_nc = xr.open_dataset(pair_file)
        # print("opened nc file")
        # There is a time mismatch between the zarr chrtout file and the netcdf
        # forcings I generated. The zarr file cuts off at 2023-02-01, whereas
        # the netcdf forcings go to 2024-09-30. I don't know why, probably just
        # based on how recently our sources updated.
        subset_ds_nc = ds_nc.isel(time=slice(0, 379897)) 
        # print("subset nc")
        # Ensure site ID exists in Zarr store
        if head_id not in zarr_store['feature_id'].values:
            print(f"Skipping {head_id}, not found in Zarr store.")
            return
        if tail_id not in zarr_store['feature_id'].values:
            print(f"Skipping {tail_id}, not found in Zarr store.")
            return
        
        # Extract the corresponding streamflow data and match it to NetCDF 
        # # format
        head_streamflow_data = zarr_store['streamflow'].sel(feature_id=head_id)
        head_streamflow_data = head_streamflow_data.interp_like(subset_ds_nc)
        # print("extracted head data")
        tail_streamflow_data = zarr_store['streamflow'].sel(feature_id=tail_id)
        tail_streamflow_data = tail_streamflow_data.interp_like(subset_ds_nc)
        # print("extracted tail data")
        # head_streamflow_array = zarr_store.sel(feature_id=head_id)
        # print(head_streamflow_array[:10].values)

        # Append to NetCDF file
        subset_ds_nc["streamflow"] = xr.DataArray(
            head_streamflow_data, dims=["time"])
        subset_ds_nc["streamflow_d"] = xr.DataArray(
            tail_streamflow_data, dims=["time"])
        # print("appended data")
        # Save updated file
        # Note to Sonam: we can change this location to {pair_file} and 
        # overwrite the original pair-specific NetCDFs if we are confident in 
        # this process.
        subset_ds_nc.to_netcdf(f"/media/volume/NeuralHydrology/Test_Quinn_Data/completed_forcings/{pair_id}.nc", mode="w") 
        print(f"Updated {pair_file}")

    except Exception as e:
        print(f"Error processing {pair_file}: {e}")

def main():

    # Open Zarr store
    zarr_store = xr.open_zarr("/home/exouser/nwm_network_lstm_test/data_collection_preprocess/camels_sf.zarr")

    # Get list of NetCDF files
    # Note to Sonam: I picked a folder with a small number of files for testing.
    # If we want to do all the forcings, our glob will look like
    # # "/media/volume/NeuralHydrology/Test_Quinn_Data/forcings/*/*/*.nc"
    netcdf_files = glob(
        "/media/volume/NeuralHydrology/Test_Quinn_Data/uncorrected_forcing_files/*/*.nc")
    
    # Create a partial function that always includes zarr_store
    update_netcdf_partial = partial(update_netcdf, zarr_store=zarr_store)

    # Note to Sonam: we can change the number of workers if we upgrade out of 
    # the small instance
    with ProcessPoolExecutor(max_workers=25) as executor:
        print("executing process")
        executor.map(update_netcdf_partial, netcdf_files)

if __name__ == "__main__":
    main()
