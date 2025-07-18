"""
Completes preprocessing step 2 (see issue #2 )
https://github.com/AlabamaWaterInstitute/nwm_camels_lstm/issues/2

Usage:
python final_preprocessing.py 
-d /path/to/json/of/basin/pairs 
-c /path/to/dir/of/aggregated.nc/files/ 
-s /path/to/streamflow/dir/ 
-o /path/to/output/dir/

Optional flag: -D for debug logging

Written by Quinn Lee (GitHub: @quinnylee) and Pratiksha Chaudhari (GitHub: @pratikshac15)
Inspired by code by Josh Cunningham (GitHub: @joshcu)
"""

import argparse
import logging
import time
import json
from datetime import timedelta
from pathlib import Path
from collections import defaultdict
import os
import xarray as xr
from dask.distributed import Client, LocalCluster
import shutil

def process_basin_pair_from_cache(d_basin, u_basin, master_upstreams_dict, cache_dir, output_dir, streamflow_dir):
    """
    Processes a single (d, u) pair using pre-computed aggregated forcing files.
    Averages meteorological forcing values and appends streamflow.
    """
    try:
        logging.info("Processing pair from cache: Downstream=%s, Upstream=%s", d_basin, u_basin)

        # load cached file
        cached_file_path = cache_dir / f"{d_basin}-aggregated.nc"
        temp_file_path = output_dir / f"{d_basin}_{u_basin}_temp/"

        if os.path.exists(temp_file_path):
            shutil.rmtree(temp_file_path)
        os.makedirs(temp_file_path)
            
        logging.info("Loading cached file: %s", cached_file_path)
        ds_aggregated = xr.open_dataset(cached_file_path)
        ds_aggregated = ds_aggregated.isel(time=slice(379897)) # streamflow data is smalelr than forcings

        catchment_dim_name = 'catchment_id'
        vars = ["DLWRF_surface", "PRES_surface", "SPFH_2maboveground", "precip_rate", "DSWRF_surface",
                "TMP_2maboveground", "UGRD_10maboveground", "VGRD_10maboveground", "APCP_surface"]
        
        try:
            client = Client.current()
        except ValueError:
            cluster = LocalCluster()
            client = Client(cluster)

        # Calculate forcing means for DOWNSTREAM 'd'
        for var in vars:
            var_forcings_d = ds_aggregated[var]
            var_d_average = var_forcings_d.mean(dim=catchment_dim_name, keep_attrs=True)
            var_d_average = var_d_average.rename(f"{var}_d")
            var_d_average.to_netcdf(temp_file_path / f"{var}_d.nc")
            var_forcings_d.close()
            var_d_average.close()

        logging.info("Calculated averages for 'd' (%s)", d_basin)

        # Calculate forcing means for UPSTREAM 'u'
        upstreams_of_u = master_upstreams_dict.get(u_basin, [])
        basins_for_u_avg_needed = [u_basin] + upstreams_of_u
     
        # Here is the real janky bit.
        # Catchment_id is just an index counting up from 0, not the actual NHD ID.
        # We need to find the indices of the basins we want to average.
        available_basins_in_file = ds_aggregated["ids"].values
        basins_to_actually_select = [b for b in basins_for_u_avg_needed 
                                     if str(b) in available_basins_in_file]
        catchment_id_indices = []
        for i, available_basin in enumerate(available_basins_in_file):
            if int(available_basin) in basins_to_actually_select:
                catchment_id_indices.append(i)

        if not basins_to_actually_select:
            raise ValueError(f"None of the required upstreams for {u_basin} were found in the cached file for {d_basin}.")
              
        # Select using the validated list of available basins, then average.
        ds_aggregated = xr.open_dataset(cached_file_path)
        ds_aggregated = ds_aggregated.isel(time=slice(379897))
        forcing_u = ds_aggregated.sel({catchment_dim_name: catchment_id_indices})
        ds_aggregated.close()

        for var in vars:
            var_forcings_u = forcing_u[var]
            var_u_average = var_forcings_u.mean(dim=catchment_dim_name, keep_attrs=True)
            var_u_average = var_u_average.rename(f"{var}_u")
            var_u_average.to_netcdf(temp_file_path / f"{var}_u.nc")
            var_forcings_u.close()
            var_u_average.close()

        forcing_u.close()
        logging.info("Calculated average for 'u' (%s) using %d available sub-basins.", u_basin, len(basins_to_actually_select))

        # append streamflow data
        sf_filepath = streamflow_dir / f"{d_basin}-streamflow.nc"
        streamflow = xr.open_dataset(sf_filepath)

        streamflow_d = streamflow['streamflow'].sel(catchment_id=d_basin)
        streamflow_d = streamflow_d.rename('streamflow_d').drop_vars("catchment_id")
        streamflow_d.to_netcdf(temp_file_path / "streamflow_d.nc")
        streamflow_d.close()

        streamflow_u = streamflow['streamflow'].sel(catchment_id=u_basin)
        streamflow_u = streamflow_u.rename('streamflow_u').drop_vars("catchment_id")
        streamflow_u.to_netcdf(temp_file_path/ "streamflow_u.nc")

        streamflow_u.close()
        streamflow.close()
        
        logging.info("Appended streamflows.")
        # Combine into the final dataset 

        results = [xr.open_dataset(file, chunks="auto") for file in temp_file_path.glob("*.nc")]
        final_ds = xr.merge(results)

        logging.info("Saving to disk")

        # Save the final preprocessed file
        output_filename = output_dir / f"{d_basin}_{u_basin}.nc"
        final_ds.to_netcdf(output_filename, engine="netcdf4")

        # close everything and clean up
        _ = [result.close() for result in results]
        shutil.rmtree(temp_file_path)
       
        return f"Successfully processed pair ({d_basin}, {u_basin})"

    except Exception as e:
        logging.error("ERROR processing pair (%s, %s): %s", d_basin, u_basin, e, exc_info=True)
        # Make sure the file is closed even if an error occurs
        if 'ds_aggregated' in locals() and ds_aggregated:
            ds_aggregated.close()
        return f"ERROR processing pair ({d_basin}, {u_basin}): {e}"
  
def replace_downstreams(data, downstream_col, terminal_code):
    '''If a node is above a terminal node, set the downstream id to the negative of the current node.'''
    ds0_mask = data[downstream_col] == terminal_code
    new_data = data.copy()
    new_data.loc[ds0_mask, downstream_col] = ds0_mask.index[ds0_mask]

    # Also set negative any nodes in downstream col not in data.index
    new_data.loc[~data[downstream_col].isin(data.index), downstream_col] *= -1
    return new_data

def reverse_network(network_connections):
    '''
    This function was sourced from the NOAA-OWP t-route codebase
    https://github.com/NOAA-OWP/t-route

    Reverse network connections graph
    
    Arguments:
    ----------
    network_connections (dict, int: [int]): downstream network connections
    
    Returns:
    --------
    rg (dict, int: [int]): upstream network connections
    
    '''
    rg = defaultdict(list)
    for src, dst in network_connections.items():
        rg[src] # a linter may tell you this is not used. but it is used to initialize the key
        for n in dst:
            rg[n].append(src)
    rg.default_factory = None
    return rg

def extract_connections(rows, target_col, terminal_codes=None):
    '''
    This function was sourced from the NOAA-OWP t-route codebase
    https://github.com/NOAA-OWP/t-route
    Extract connection network from dataframe.

    Arguments:
    ----------
    rows (DataFrame): Dataframe indexed by key_col.
    key_col    (str): Source of each edge
    target_col (str): Target of edge

    Returns:
    --------
    network (dict, int: [int]): {segment id: [list of downstream adjacent segment ids]}
    
    '''
    if terminal_codes is not None:
        terminal_codes = set(terminal_codes)
    else:
        terminal_codes = {0}

    network = {}
    for src, dst in rows[target_col].items():
        if src not in network:
            network[src] = []

        if dst not in terminal_codes:
            network[src].append(dst)
    return network

def get_upstreams(basin, upstreams, rconn):
    '''Recursively get upstream basins.'''
    direct_upstreams = rconn[basin]
    for direct_upstream in direct_upstreams:
        if direct_upstream not in upstreams:
            upstreams.append(direct_upstream)
            get_upstreams(direct_upstream, upstreams, rconn)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Combine pre-computed forcings for d/u basin pairs."
        )
    parser.add_argument("-d",
                        "--dictionary", 
                        type=Path,
                        help="Path to JSON file of (d, u) basin pairs.",
                        required=True)
    parser.add_argument("-c",
                        "--cache_dir", 
                        type=Path,
                        help="Path to the directory with original -aggregated.nc files.",
                        required=True)
    parser.add_argument("-s",
                        "--streamflow",
                        type=Path,
                        help="Path to directory of streamflow data",
                        required=True)
    parser.add_argument("-o",
                        "--output_dir", 
                        type=Path,
                        help="Path to the final output directory.",
                        required=True)
    parser.add_argument("-D", "--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()

def main():
    """Main function to process (d, u) basin pairs."""
    logging.basicConfig(
        level=logging.DEBUG if parse_arguments().debug else logging.INFO)
    args = parse_arguments()
   
    # so we only compute for basins that have files locally
    cached_sites = os.listdir(args.cache_dir)
    for i in range(len(cached_sites)):
        cached_sites[i] = cached_sites[i].replace("-aggregated.nc", "")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load the (d, u) basin pairs
    with open(args.dictionary, 'r', encoding='utf-8') as f:
        n_step_upstreams = json.load(f)

    subset_upstreams = {}

    # get upstreams for upstream. horrendous notation but that's all i have for today
    all_upstreams = set()
    for downstream, upstreams in n_step_upstreams.items():
        if downstream not in cached_sites:
            continue
        # Simplify to only the first upstream for each downstream
        subset_upstreams[downstream] = upstreams[0] 
        all_upstreams.add(upstreams[0])
  
    # Set up routelink
    routelink_ds = xr.open_dataset("../RouteLink_CONUS.nc")
    # Subset the dataset to only the columns we want
    subslice = [
        "link",
        "to",
        "gages",
    ]
    routelink_df = routelink_ds[subslice].to_dataframe().astype({"link": int, "to": int,})
    routelink_df = routelink_df.set_index("link")
    # Reorganize RouteLink file
    routelink_df = routelink_df.sort_index()
    routelink_df = replace_downstreams(routelink_df, "to", 0)
    # Extract topology from RouteLink file
    connections = extract_connections(routelink_df, "to")
    rconn = reverse_network(connections)
    # recursively get upstreams
    upstream_dict = {}
    for basin in all_upstreams:
        upstreams = []
        get_upstreams(basin, upstreams, rconn)
        upstream_dict[basin] = upstreams

    tasks = []
    for d_basin_str, u_basin in subset_upstreams.items():
        d_basin = int(d_basin_str)
        output_name = f"{d_basin}_{u_basin}.nc"
        if os.path.exists(os.path.join(args.output_dir, output_name)):
            continue
        tasks.append((d_basin, u_basin))
          
    if not tasks:
        logging.warning("No basin pairs to process. Exiting.")
        exit()
     
    total_tasks = len(tasks)
    logging.info("Found %d basin pairs to process from cache. Starting...", total_tasks)

    results = []
    start_time_total = time.time()

    for i, (d_basin, u_basin) in enumerate(tasks):
        logging.info("--- Processing task %d of %d ---", i+1, total_tasks)
        result = process_basin_pair_from_cache(d_basin, u_basin, upstream_dict, args.cache_dir, args.output_dir, args.streamflow)
        results.append(result)
       
        elapsed_time = time.time() - start_time_total
        avg_time_per_task = elapsed_time / (i + 1)
        tasks_remaining = total_tasks - (i + 1)
        estimated_time_remaining = avg_time_per_task * tasks_remaining
        eta_str = str(timedelta(seconds=int(estimated_time_remaining)))
        logging.info(f"Progress: {i+1}/{total_tasks} ({((i+1)/total_tasks)*100:.2f}%) complete. ETA: {eta_str}")

    logging.info("\n--- Processing Complete ---")
    success_count = 0
    for res in results:
        if "Successfully" in res:
            success_count += 1
            logging.info(res)
        else:
            logging.error(res)
           
    total_elapsed_time = time.time() - start_time_total
    total_time_str = str(timedelta(seconds=int(total_elapsed_time)))
    logging.info("\nTotal pairs processed: %d", total_tasks)
    logging.info("Successful: %d", success_count)
    logging.info("Errors: %d", len(tasks) - success_count)
    logging.info("Total execution time: %s", total_time_str)
    logging.info("Preprocessed files saved in '%s' directory.", args.output_dir)

if __name__ == '__main__':
    main()
    