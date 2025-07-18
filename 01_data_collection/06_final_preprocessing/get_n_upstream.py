"""
What This Code Does:

The script is the complete and functional code for the first part of plan: identifying the upstream/downstream basin pairs.

It:

1. Loads the river network topology.

2. Takes n as an input.

3. Identifies every upstream basin u that is exactly n reaches away from a downstream CAMELS basin d.

4. Saves this list of (d, u) pairs to a JSON file for later use.

Written by Pratiksha Chaudhari (GitHub: @pratikshac15)
"""



import xarray as xr
from collections import defaultdict
import json
import pandas as pd
import os

def get_n_step_upstreams(start_basin, n, reversed_connections):
    """Finds all basins exactly n steps upstream of a starting basin."""
    if n < 1:
        return []
    
    current_reaches = [start_basin]
    for _ in range(n):
        next_level_reaches = []
        for reach in current_reaches:
            upstreams = reversed_connections.get(reach, [])
            next_level_reaches.extend(upstreams)
        current_reaches = list(set(next_level_reaches))
        if not current_reaches:
            return []
            
    return current_reaches

def main():
    """
    Main function to load data, process basins, and save results.
    """
    print("Starting script...")

    # --- Configuration ---
    # Set the desired number of upstream reaches
    n = 1
    
    # Define file paths. Assumes RouteLink and camels_basins are in the parent directory.
    script_dir = os.path.dirname(__file__)
    routelink_path = os.path.join(script_dir, '../RouteLink_CONUS.nc')
    basins_path = os.path.join(script_dir, '../02_get_upstream_basins/alabama_basins.txt')
    
    # --- 1. Load Data and Build Network ---
    print(f"Loading RouteLink file from: {routelink_path}")
    if not os.path.exists(routelink_path):
        print(f"ERROR: RouteLink_CONUS.nc not found at {routelink_path}")
        return

    routelink_ds = xr.open_dataset(routelink_path)
    routelink_df = routelink_ds[['link', 'to']].to_dataframe()
    routelink_df['link'] = routelink_df['link'].astype(int)
    routelink_df['to'] = routelink_df['to'].astype(int)
    routelink_df = routelink_df.set_index('link')
    
    connections = {}
    for src, row in routelink_df.iterrows():
        dst = row['to']
        connections[src] = [dst] if dst != 0 else []
        
    rconn = defaultdict(list)
    for src, dst_list in connections.items():
        for dst in dst_list:
            rconn[dst].append(src)
            
    print("Network topology built successfully.")

    # --- 2. Process CAMELS Basins ---
    print(f"Loading CAMELS basins from: {basins_path}")
    if not os.path.exists(basins_path):
        print(f"ERROR: alabama_basins.txt not found at {basins_path}")
        return
        
    with open(basins_path, 'r') as f:
        camels_basins = [int(basin) for basin in f.read().splitlines()]

    upstream_dict = {}
    print(f"Processing {len(camels_basins)} basins to find upstreams at n={n}...")
    for basin in camels_basins:
        n_step_upstreams = get_n_step_upstreams(basin, n, rconn)
        if n_step_upstreams:
            upstream_dict[basin] = n_step_upstreams

    # --- 3. Save Results ---
    output_filename = f'alabama_n{n}_upstream_dict.json'
    with open(output_filename, 'w') as f:
        json.dump(upstream_dict, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"Results saved to: {os.path.abspath(output_filename)}")

if __name__ == '__main__':
    main()