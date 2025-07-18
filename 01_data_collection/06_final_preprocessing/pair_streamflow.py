"""
Creates paired streamflow files for downstream and upstream basin pairs.
This is intended as a data preparation step for hydrological models,
creating the "target" data (what the model tries to predict).

This script reads pre-generated NetCDF files, each containing streamflow
for a major basin and all of its sub-basins. It then extracts the specific
time series for a given downstream/upstream (d/u) pair and saves them
together in a new file.

Usage:
python pair_streamflow.py -d /path/to/pairs.json -s /path/to/streamflow/files -o /path/to/output/

Optional flag: -D for debug logging


"""

import argparse
import logging
import json
import time
from pathlib import Path
import os
import xarray as xr

def process_streamflow_pair(d_basin: int, u_basin: int, streamflow_dir: Path, output_dir: Path):
    """
    Selects the d/u streamflow time series from a pre-downloaded NetCDF file.

    For a given downstream basin ('d_basin'), this function loads the corresponding
    master streamflow file. It then extracts the specific time series for both
    the 'd_basin' outlet and a specified upstream basin ('u_basin') outlet,
    saving them as a new, paired NetCDF file.

    Args:
        d_basin (int): The feature ID of the downstream basin.
        u_basin (int): The feature ID of the upstream basin.
        streamflow_dir (Path): The directory containing the master streamflow files
                               (e.g., '{d_basin}-streamflow.nc').
        output_dir (Path): The directory where the paired NetCDF will be saved.

    Returns:
        str: A status message indicating success or failure.
    """
    try:
        logging.info("Processing streamflow pair: Downstream=%s, Upstream=%s", d_basin, u_basin)
        ds = None  # Initialize to ensure it's defined for the finally block

        # 1. Load the corresponding streamflow file for the downstream basin.
        # This file must contain streamflow for 'd_basin' and all its upstreams.
        streamflow_file = streamflow_dir / f"{d_basin}-streamflow.nc"
        if not streamflow_file.exists():
            raise FileNotFoundError(f"Input file not found: {streamflow_file}")
        
        ds = xr.open_dataset(streamflow_file)

        # 2. Select the time series for the downstream basin outlet
        flow_d = ds['streamflow'].sel(catchment_id=d_basin)
        flow_d = flow_d.rename('streamflow_d')  # Rename for clarity and merging

        # 3. Select the time series for the upstream basin outlet
        flow_u = ds['streamflow'].sel(catchment_id=u_basin)
        flow_u = flow_u.rename('streamflow_u')  # Rename for clarity and merging

        # 4. Merge them into a final dataset
        final_ds = xr.merge([flow_d, flow_u])
        final_ds.attrs['comment'] = f"Paired streamflow for downstream basin {d_basin} and upstream basin {u_basin}."
        final_ds.attrs['creation_date'] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        # 5. Save the final paired file
        output_filename = output_dir / f"{d_basin}_{u_basin}_streamflow.nc"
        final_ds.to_netcdf(output_filename)

        return f"Successfully processed streamflow pair ({d_basin}, {u_basin})"

    except Exception as e:
        logging.error("ERROR processing streamflow pair (%s, %s): %s", d_basin, u_basin, e, exc_info=True)
        return f"ERROR processing streamflow pair ({d_basin}, {u_basin}): {e}"
    
    finally:
        # Ensure files are closed even if errors occur
        if ds is not None:
            ds.close()
        if 'final_ds' in locals() and final_ds is not None:
            final_ds.close()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create paired streamflow files for downstream/upstream basin pairs."
    )
    parser.add_argument(
        "-d", "--dictionary",
        type=Path,
        required=True,
        help="Path to the JSON file of (d, u) basin pairs."
    )
    parser.add_argument(
        "-s", "--streamflow_dir",
        type=Path,
        required=True,
        help="Path to the input directory with '-streamflow.nc' files."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        required=True,
        help="Path to the directory for final paired output files."
    )
    parser.add_argument(
        "-D", "--debug",
        action="store_true",
        help="Enable debug logging."
    )
    return parser.parse_args()


def main():
    """Main function to drive the script."""
    args = parse_arguments()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load the (d, u) basin pairs from the JSON file
    try:
        with open(args.dictionary, 'r', encoding='utf-8') as f:
            basin_pairs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.critical("Could not read or parse the basin pairs dictionary: %s", e)
        return

    # Prepare a list of tasks to execute
    tasks = []
    for d_basin_str, u_basins in basin_pairs.items():
        # The forcing script logic used only the first upstream. We'll do the same for consistency.
        if not u_basins:
            continue
        u_basin = u_basins[0]
        d_basin = int(d_basin_str)

        # Check if the output file already exists
        output_file = args.output_dir / f"{d_basin}_{u_basin}_streamflow.nc"
        if output_file.exists():
            logging.debug("Output file %s already exists. Skipping.", output_file.name)
            continue
        
        tasks.append((d_basin, u_basin))

    if not tasks:
        logging.warning("No new basin pairs to process. All output files may already exist. Exiting.")
        return

    # Process all tasks
    total_tasks = len(tasks)
    logging.info("Found %d new basin pairs to process. Starting...", total_tasks)
    start_time = time.time()
    
    success_count = 0
    results = []

    for i, (d_basin, u_basin) in enumerate(tasks):
        logging.info("--- Processing task %d of %d ---", i + 1, total_tasks)
        result = process_streamflow_pair(d_basin, u_basin, args.streamflow_dir, args.output_dir)
        results.append(result)
        if "Successfully" in result:
            success_count += 1

    # --- Final Summary ---
    logging.info("\n--- Processing Complete ---")
    
    # Log detailed results
    for res in results:
        if "Successfully" in res:
            logging.info(res)
        else:
            logging.error(res)

    total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    logging.info("\nTotal pairs processed: %d", total_tasks)
    logging.info("Successful: %d", success_count)
    logging.info("Errors: %d", total_tasks - success_count)
    logging.info("Total execution time: %s", total_time_str)
    logging.info("Paired streamflow files saved in '%s'", args.output_dir)


if __name__ == '__main__':
    main()
