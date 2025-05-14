"""Load and process zarr datasets from S3.
This module provides functions to load zarr datasets from S3, clip them to a specified
geographical bounding box, and cache the results in a netCDF file.
It also includes functions to validate the time range of the datasets and
compute the store for the datasets.

Adapted from the original code by Josh Cunningham (GitHub: @JoshCu)
https://github.com/CIROH-UA/NGIAB_data_preprocess

Adapted by Quinn Lee (GitHub @quinnylee)
"""

import logging
import os
from pathlib import Path
from typing import Tuple
import sys
import geopandas as gpd
import numpy as np
import s3fs
import xarray as xr
from dask.distributed import Client, LocalCluster, progress
import rich

sys.path.append("./modules")
from s3fs_utils import S3ParallelFileSystem

logger = logging.getLogger(__name__)

def load_zarr_datasets(forcing_vars: list[str] = None) -> xr.Dataset:
    """Load zarr datasets from S3 within the specified time range."""
    # if a LocalCluster is not already running, start one
    if not forcing_vars:
        forcing_vars = ["lwdown", "precip", "psfc", "q2d", "swdown", "t2d", "u2d", "v2d"]
    try:
        client = Client.current()
    except ValueError:
        cluster = LocalCluster()
        client = Client(cluster)
    s3_urls = [
        f"s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/forcing/{var}.zarr"
        for var in forcing_vars
    ]
    # default cache is readahead which is detrimental to performance in this case
    fs = S3ParallelFileSystem(anon=True, default_cache_type="none")  # default_block_size
    s3_stores = [s3fs.S3Map(url, s3=fs) for url in s3_urls]
    # the cache option here just holds accessed data in memory to prevent s3 being queried
    # multiple times
    # most of the data is read once and written to disk but some of the coordinate data is
    # read multiple times
    dataset = xr.open_mfdataset(s3_stores, parallel=True, engine="zarr", cache=True)
    return dataset

def load_aorc_zarr_datasets(start_year: int = 1979, end_year: int = 2024) -> xr.Dataset:
    """Load the aorc zarr dataset from S3."""
    try:
        client = Client.current()
    except ValueError:
        cluster = LocalCluster()
        client = Client(cluster)
    info = f"Loading AORC zarr datasets from {start_year} to {end_year}"
    logger.info(info)
    estimated_time_s = ((end_year - start_year) * 2.5) + 3.5
    # from testing, it's about 2.1s per year + 3.5s overhead
    logger.info("This should take roughly %s seconds", estimated_time_s)
    fs = S3ParallelFileSystem(anon=True, default_cache_type="none")
    s3_url = "s3://noaa-nws-aorc-v1-1-1km/"
    urls = [f"{s3_url}{i}.zarr" for i in range(start_year, end_year)]
    filestores = [s3fs.S3Map(url, s3=fs) for url in urls]
    timer = rich.progress.Progress()
    timer.start()
    dataset = xr.open_mfdataset(filestores, parallel=True, engine="zarr", cache=True)
    # add dataset.crs and dataset.crs.esri_pe_string
    dataset = dataset.assign_coords(crs=1)
    dataset.crs.attrs = {"esri_pe_string": "+proj=longlat +datum=WGS84 +no_defs"}

    # rename latitude and longitude to x and y
    dataset = dataset.rename({"latitude": "y", "longitude": "x"})

    timer.stop()
    return dataset


def validate_time_range(dataset: xr.Dataset, start_time: str, end_time: str) -> Tuple[str, str]:
    '''
    Ensure that all selected times are in the passed dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with a time coordinate.
    start_time : str
        Desired start time in YYYY/MM/DD HH:MM:SS format.
    end_time : str
        Desired end time in YYYY/MM/DD HH:MM:SS format.

    Returns
    -------
    str
        start_time, or if not available, earliest available timestep in dataset.
    str
        end_time, or if not available, latest available timestep in dataset.
    '''
    end_time_in_dataset = dataset.time.isel(time=-1).values
    start_time_in_dataset = dataset.time.isel(time=0).values
    if np.datetime64(start_time) < start_time_in_dataset:
        warning1 = f"provided start {start_time} is before the start of the dataset "
        warning2 = f"{start_time_in_dataset}, selecting from {start_time_in_dataset}"
        warning = warning1 + warning2
        logger.warning(warning)
        start_time = start_time_in_dataset
    if np.datetime64(end_time) > end_time_in_dataset:
        warning1 = f"provided end {end_time} is after the end of the dataset "
        warning2 = "{end_time_in_dataset}, selecting until {end_time_in_dataset}"
        warning = warning1 + warning2
        logger.warning(warning)
        end_time = end_time_in_dataset
    return start_time, end_time


def clip_dataset_to_bounds(
    dataset: xr.Dataset, bounds: Tuple[float, float, float, float], start_time: str, end_time: str
) -> xr.Dataset:
    """
    Clip the dataset to specified geographical bounds.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to be clipped.
    bounds : tuple[float, float, float, float]
        Corners of bounding box. bounds[0] is x_min, bounds[1] is y_min, 
        bounds[2] is x_max, bounds[3] is y_max.
    start_time : str
        Desired start time in YYYY/MM/DD HH:MM:SS format.
    end_time : str
        Desired end time in YYYY/MM/DD HH:MM:SS format.
    
    Returns
    -------
    xr.Dataset
        Clipped dataset.
    """
    # check time range here in case just this function is imported and not the whole module
    start_time, end_time = validate_time_range(dataset, start_time, end_time)
    dataset = dataset.sel(
        x=slice(bounds[0], bounds[2]+0.01), # buffer added to deal with weird skinny geometries
        y=slice(bounds[1], bounds[3]+0.01),
        time=slice(start_time, end_time),
    )
    logger.debug(slice(bounds[0], bounds[2]+0.01))
    logger.debug(slice(bounds[1], bounds[3]+0.01))
    logger.info("Selected time range and clipped to bounds")
    return dataset


def compute_store(stores: xr.Dataset, cached_nc_path: Path) -> xr.Dataset:
    """Compute the store and save it to a cached netCDF file."""
    logger.info("Downloading and caching forcing data, this may take a while")

    # sort of terrible work around for half downloaded files
    temp_path = cached_nc_path.with_suffix(".downloading.nc")
    if os.path.exists(temp_path):
        os.remove(temp_path)

    ## Cast every single variable to float32 to save space to save a lot of memory issues later
    ## easier to do it now in this slow download step than later in the steps without dask
    for var in stores.data_vars:
        if var != "crs":
            stores[var] = stores[var].astype("float32")

    client = Client.current()
    future = client.compute(stores.to_netcdf(temp_path, compute=False))
    # Display progress bar
    progress(future)
    future.result()

    os.rename(temp_path, cached_nc_path)

    data = xr.open_mfdataset(cached_nc_path, parallel=True, engine="h5netcdf")
    return data


def get_forcing_data(
    cached_nc_path: Path,
    start_time: str,
    end_time: str,
    gdf: gpd.GeoDataFrame,
    forcing_vars: list[str] = None,
) -> xr.Dataset:
    """Get forcing data from zarr datasets, clip to bounds and cache to netCDF file."""
    merged_data = None

    if os.path.exists(cached_nc_path):
        logger.info("Found cached nc file")
        # open the cached file and check that the time range is correct
        cached_data = xr.open_mfdataset(
            cached_nc_path, parallel=True, engine="h5netcdf"
        )
        range_in_cache = cached_data.time[0].values <= np.datetime64(
            start_time
        ) and cached_data.time[-1].values >= np.datetime64(end_time)

        gdf = gdf.to_crs(cached_data.crs.esri_pe_string)

        if not range_in_cache:
            # only do this if the time range is not in the cache as it is slow
            # this catches cases where a user entered 2030 as the end on the first run and
            # the cache only goes to 2023
            # it will prevent the cache from being deleted and reloaded every time
            lazy_store = load_zarr_datasets()
            start_time, end_time = validate_time_range(lazy_store, start_time, end_time)

        if forcing_vars:
            # check if the forcing vars are all in the cached data
            # the zarr file names dont exactly match the forcing vars within them
            cached_vars = cached_data.data_vars.keys()
            cached_vars = [var.lower() for var in cached_vars if var != "crs"]
            # replace rainrate with precip
            cached_vars = [var.replace("rainrate", "precip") for var in cached_vars]
            missing_vars = set(forcing_vars) - set(cached_vars)
            if len(missing_vars) > 0:
                logger.info("Missing forcing vars in cache: %s", missing_vars)
                range_in_cache = False

        if range_in_cache:
            logger.info("Time range is within cached data")
            logger.debug("Opened cached nc file: [%s]", cached_nc_path)
            merged_data = clip_dataset_to_bounds(
                cached_data, gdf.total_bounds, start_time, end_time
            )
            logger.debug("Clipped stores")
        else:
            logger.info("Time range is incorrect")
            os.remove(cached_nc_path)
            logger.debug("Removed cached nc file")

    if merged_data is None:
        logger.info("Loading zarr stores")
        # create new event loop
        # lazy_store = load_zarr_datasets(forcing_vars)
        start_year = int(start_time.split("-")[0])
        end_year = int(end_time.split("-")[0]) + 1
        lazy_store = load_aorc_zarr_datasets(start_year, end_year)
        gdf = gdf.to_crs(lazy_store.crs.esri_pe_string)  # for retro
        logger.debug("Got zarr stores")
        clipped_store = clip_dataset_to_bounds(lazy_store, gdf.total_bounds, start_time, end_time)
        logger.info("Clipped forcing data to bounds")
        logger.debug(lazy_store.head())
        merged_data = compute_store(clipped_store, cached_nc_path)
        logger.info("Forcing data loaded and cached")
        # close the event loop

    # merged_data = merged_data.drop_vars(["crs"])
    return merged_data
