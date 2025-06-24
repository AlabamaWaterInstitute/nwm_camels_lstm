"""
Forcings Generation Module

This script calculates zonal statistics (mean value of raster cells within a polygon)
for a set of catchment geometries over a time-series of meteorological forcings.

This updated version uses the xarray.apply_ufunc pattern for parallel computation.
This approach leverages Dask's native scheduler to handle memory management,
parallelism, and task distribution, eliminating the need for manual chunking,
shared memory, and multiprocessing pools, which resolves the 'KilledWorker' errors.
"""

# Make sure you have this import at the top of your file
from dask.distributed import Client, LocalCluster

import logging
import time
import warnings
from pathlib import Path
from typing import Tuple
import multiprocessing
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# --- Basic Setup ---
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="'DataFrame.swapaxes' is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="'GeoDataFrame.swapaxes' is deprecated", category=FutureWarning)


# ==============================================================================
# SECTION 1: CORE HELPER FUNCTIONS (Largely Unchanged)
# These functions perform the fundamental calculations and data preparations.
# ==============================================================================

def weighted_sum_of_cells(flat_raster: np.ndarray, cell_ids: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """
    Calculates the weighted mean for a set of cells over all timesteps in a chunk.
    This function remains the core of the zonal statistic calculation.
    """
    result = np.sum(flat_raster[:, cell_ids] * factors, axis=1)
    sum_of_weights = np.sum(factors)
    if sum_of_weights > 0:
        result /= sum_of_weights
    return result


def get_cell_weights(raster: xr.Dataset, gdf: gpd.GeoDataFrame, wkt: str) -> pd.DataFrame:
    """
    Uses exactextract to find the intersection of raster cells and a single polygon,
    calculating the coverage fraction for each cell.
    """
    # Use a small buffer to handle skinny polygons
    xmin, xmax = raster.x.min().item(), raster.x.max().item() + 0.001
    ymin, ymax = raster.y.min().item(), raster.y.max().item() + 0.01
    
    data_vars = list(raster.data_vars)
    if not data_vars:
        raise ValueError("Input raster for get_cell_weights has no data variables.")
        
    rastersource = NumPyRasterSource(
        raster[data_vars[0]], srs_wkt=wkt, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    )
    output = exact_extract(
        rastersource,
        gdf,
        ["cell_id", "coverage"],
        include_cols=["ID"],
        output="pandas",
    )
    return output.set_index("ID")


def get_cell_weights_parallel(gdf: gpd.GeoDataFrame, input_forcings: xr.Dataset, num_partitions: int) -> pd.DataFrame:
    """
    Efficiently pre-calculates the cell weights for all catchments in parallel.
    This is a critical pre-processing step.
    """
    logger.info("Pre-calculating cell weights for all catchments...")
    gdf_chunks = np.array_split(gdf, num_partitions)
    wkt = gdf.crs.to_wkt()
    one_timestep = input_forcings.isel(time=0).compute()

    with multiprocessing.Pool(processes=num_partitions) as pool:
        args = [(one_timestep, gdf_chunk, wkt) for gdf_chunk in gdf_chunks]
        catchments_list = pool.starmap(get_cell_weights, args)
        
    logger.info("Cell weights calculation complete.")
    return pd.concat(catchments_list)

# --- Dataset Utility Functions (Unchanged) ---

def add_apcp_surface_to_dataset(dataset: xr.Dataset) -> xr.Dataset:
    dataset["APCP_surface"] = dataset["precip_rate"] * 3600
    dataset["APCP_surface"].attrs["units"] = "mm h^-1"
    dataset["APCP_surface"].attrs["source_note"] = "This is just the precip_rate variable converted to mm/h by multiplying by 3600"
    return dataset

def add_precip_rate_to_dataset(dataset: xr.Dataset) -> xr.Dataset:
    dataset["precip_rate"] = dataset["APCP_surface"] / 3600
    dataset["precip_rate"].attrs["units"] = "mm s^-1"
    dataset["precip_rate"].attrs["source_note"] = "This is just the APCP_surface variable converted to mm/s by dividing by 3600"
    return dataset


# ==============================================================================
# SECTION 2: NEW DASK-NATIVE COMPUTATION LOGIC
# This section contains the new, refactored functions for parallel processing.
# ==============================================================================

def calculate_zonal_stats_for_chunk(raster_chunk: np.ndarray, **kwargs) -> np.ndarray:
    """
    Core computation function designed to work on a single chunk of raster data.
    This function is what Dask will run in parallel on each chunk. It knows nothing
    about shared memory or multiprocessing; it just performs a calculation.

    Args:
        raster_chunk: A NumPy array for one chunk, with shape (time, y, x).
        catchment_weights: A DataFrame with cell_id and coverage for ALL catchments.

    Returns:
        A NumPy array with results, shaped (num_catchments, num_timesteps).
    """

     # Unpack the list of catchment information from the keyword arguments
    # Get a handle to the Dask client from within the worker process
    # The list is passed directly, no need to gather.
    catchment_info_list = kwargs['catchment_info']
    
    num_timesteps, _, _ = raster_chunk.shape
    flat_raster = raster_chunk.reshape(num_timesteps, -1)

    results_list = []
    for info in catchment_info_list:
        cell_ids = info['cell_ids']
        factors = info['factors']
        
        mean_values = weighted_sum_of_cells(flat_raster, cell_ids, factors)
        results_list.append(mean_values)

    return np.stack(results_list, axis=0)

def _write_final_netcdf(final_ds: xr.Dataset, forcings_dir: Path, variables_map: dict, units: dict):
    """
    Helper function to process and write the final computed dataset to a NetCDF file.
    """
    # Rename variables to match the desired output format
    rename_dict = {key: value for key, value in variables_map.items() if key in final_ds}
    final_ds = final_ds.rename_vars(rename_dict)

    # Add units and handle precipitation variable conversions
    for var in final_ds.data_vars:
        if var in units:
            final_ds[var].attrs["units"] = units.get(var, "unknown")
    
    if "APCP_surface" in final_ds.data_vars:
        final_ds = add_precip_rate_to_dataset(final_ds)
    elif "precip_rate" in final_ds.data_vars:
        final_ds = add_apcp_surface_to_dataset(final_ds)



    # Ensure data type is float32 for efficiency
    for var in final_ds.data_vars:
        final_ds[var] = final_ds[var].astype(np.float32)

    logger.info("Saving final forcings to disk...")
    output_path = forcings_dir / "forcings.nc"
    final_ds.to_netcdf(output_path, engine="netcdf4")
    logger.info(f"Forcing generation complete! Output saved to: {output_path}")
    

# ==============================================================================
# SECTION 3: REFACTORED MAIN FUNCTION
# This is the new entry point, replacing the old, complex compute_zonal_stats.
# ==============================================================================

def compute_zonal_stats(gdf: gpd.GeoDataFrame, merged_data: xr.Dataset, forcings_dir: Path):
    """
    Final, simplified version. Relies on small chunking and float32 dtype for 
    memory management, removing all complex client/scatter/gather logic.
    """
    timer_start = time.time()

    #  try:
    #     client = Client.current()
    #     logger.info("Using existing Dask client.")
    # except ValueError:
    #     # Force Dask to use only ONE worker. This gives that worker
    #     # the maximum possible memory to avoid crashing.
    #     # Adjust '28GB' to a safe value for your system (e.g., 80% of total RAM).
    #     cluster = LocalCluster(n_workers=2, memory_limit='28GB')
    #     client = Client(cluster)
    #     logger.info("Started new local Dask client with 1 worker to maximize memory safety.") 

    # Explicitly create and manage the Dask cluster
    with LocalCluster(n_workers=2, memory_limit='28GB') as cluster:
        with Client(cluster) as client:
            logger.info(f"Started Dask client with {len(client.scheduler_info()['workers'])} workers.")

    variables_map = {
        "LWDOWN": "DLWRF_surface", "PSFC": "PRES_surface", "Q2D": "SPFH_2maboveground",
        "RAINRATE": "precip_rate", "SWDOWN": "DSWRF_surface", "T2D": "TMP_2maboveground",
        "U2D": "UGRD_10maboveground", "V2D": "VGRD_10maboveground",
        "APCP_surface": "APCP_surface"
    }
    units = {var: merged_data[var].attrs.get("units", "unknown") for var in merged_data.data_vars}

    num_partitions = max(1, multiprocessing.cpu_count() - 1)
    catchments_with_weights = get_cell_weights_parallel(gdf, merged_data, num_partitions)
    catchment_ids = catchments_with_weights.index.unique().sort_values()
    catchment_ids.name = 'catchment'

    logger.info("Preparing catchment data...")
    catchment_info_list = []
    for c_id in catchment_ids:
        catchment_data = catchments_with_weights.loc[c_id]
        catchment_info_list.append({
            'id': c_id,
            'cell_ids': catchment_data["cell_id"],
            'factors': catchment_data["coverage"]
        })

    all_variable_results = []
    
    progress = Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                        "[progress.percentage]{task.percentage:>3.0f}%", TimeElapsedColumn())
    
    data_vars_to_process = [v for v in variables_map.keys() if v in merged_data.data_vars]

    with progress:
        task_id = progress.add_task("[cyan]Processing variables...", total=len(data_vars_to_process))
        for var_name in data_vars_to_process:
            progress.update(task_id, description=f"[cyan]Processing [bold]{var_name}[/bold]...")
            data_array = merged_data[var_name]

            gufunc_result = xr.apply_ufunc(
                calculate_zonal_stats_for_chunk,
                data_array,
                # Pass the actual list directly. Dask will warn, but it should work now.
                kwargs={'catchment_info': catchment_info_list},
                input_core_dims=[['time', 'y', 'x']],
                output_core_dims=[['catchment', 'time']],
                exclude_dims=set(('y', 'x')),
                dask="parallelized",
                output_dtypes=[data_array.dtype],
                dask_gufunc_kwargs={
                    'output_sizes': {'catchment': len(catchment_ids)},
                    'allow_rechunk': True
                }
            )

            zonal_stats_da = xr.DataArray(
                data=gufunc_result.data,
                dims=['catchment', 'time'],
                coords={'catchment': catchment_ids, 'time': data_array.time.values}
            )
            
            all_variable_results.append(zonal_stats_da.rename(var_name))
            progress.advance(task_id)

    logger.info("All variables have been prepared. Merging results...")
    final_lazy_ds = xr.merge(all_variable_results)

    logger.info("Triggering Dask computation. This may take some time...")
    computed_ds = final_lazy_ds.compute()
    logger.info("Computation complete.")

    _write_final_netcdf(computed_ds, forcings_dir, variables_map, units)
    logger.info(f"Total processing time: {time.time() - timer_start:.2f} seconds")
