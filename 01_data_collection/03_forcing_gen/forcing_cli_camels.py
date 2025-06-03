"""Generate forcing data for CAMELS basins and all the upstream basins.

Heavily inspired by the original code by Josh Cunningham (GitHub: @JoshCu)
https://github.com/CIROH-UA/NGIAB_data_preprocess

Hydrofabric files sourced from:
https://water.noaa.gov/resources/downloads/nwm/NWM_channel_hydrofabric.tar.gz
Download, unzip, use GeoPandas to read the hydrofabric file's nwm_reaches_conus layer
and nwm_catchments_conus layer, and save as parquet files.

Written by Quinn Lee (GitHub: @quinnylee)
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
import shutil
import json
import geopandas as gpd
import xarray as xr
import os
# from memory_profiler import profile

from modules.custom_logging import setup_logging
from modules.forcings import compute_zonal_stats
from modules.zarr_utils import get_forcing_data

# Constants
DATE_FORMAT = "%Y-%m-%d"  # used for datetime parsing
DATE_FORMAT_HINT = "YYYY-MM-DD"  # printed in help message

def head_geom_selection(headwater: str,
                       gdb: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    '''
    Select headwater row from GeoDataFrame containing all basins in study area.

    Parameters
    ----------
    headwater : str
        NWM 3.0 reach ID of headwater basin
    gdb : gpd.GeoDataFrame
        GeoDataFrame that contains geometry information about all basins in
        study area.

    Returns
    -------
    head_gdf : gpd.GeoDataFrame
        The row in gdb that corresponds to the headwater basin.
    '''
    head_geom = gdb[gdb['ID'] == int(headwater)]['geometry'].values[0]

    return head_geom

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Subsetting hydrofabrics, forcing generation, and realization creation"
    )

    parser.add_argument(
        "--hf",
        "--hydrofabric",
        type=Path,
        help="path to hydrofabric gpkg",
    )

    parser.add_argument(
        "-d",
        "--dictionary",
        type=Path,
        help="path to txt file of basins and upstreams in dict form",
        required=True
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        help="path to the output dir, e.g. /path/to/forcings",
        required=True,
    )

    parser.add_argument(
        "--start_date",
        "--start",
        type=lambda s: datetime.strptime(s, DATE_FORMAT),
        help=f"Start date for forcings/realization (format {DATE_FORMAT_HINT})",
        required=True,
    )
    parser.add_argument(
        "--end_date",
        "--end",
        type=lambda s: datetime.strptime(s, DATE_FORMAT),
        help=f"End date for forcings/realization (format {DATE_FORMAT_HINT})",
        required=True,
    )
    parser.add_argument(
        "-D",
        "--debug",
        action="store_true",
        help="enable debug logging",
    )

    return parser.parse_args()

def process_catchment(geometries_dict, output_dir, camels_basin, merged_data):
    """Process a catchment by creating a directory and computing zonal stats."""
    catchment_dir = Path(f"{output_dir}/{camels_basin}/")
    if not catchment_dir.exists():
        catchment_dir.mkdir()
        logging.debug("Created directory: %s", catchment_dir)

    output_file = Path(f"{output_dir}/{camels_basin}/{camels_basin}-aggregated.nc")
    if not output_file.exists():
        # working directory for in progress files
        forcing_working_dir = Path(f"{output_dir}/{camels_basin}/{camels_basin}-working-dir/")
        if not forcing_working_dir.exists():
            forcing_working_dir.mkdir(parents=True, exist_ok=True)
            logging.debug("Created working directory: %s", forcing_working_dir)

        temp_dir = forcing_working_dir / "temp"
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True, exist_ok=True)
        geometries_dict = geometries_dict.to_crs(merged_data.crs.esri_pe_string)
        print(geometries_dict.dtypes)
        compute_zonal_stats(geometries_dict, merged_data, forcing_working_dir)
        logging.debug("Computed zonal stats for %s", camels_basin)

        shutil.copy(forcing_working_dir / "forcings.nc", output_file)
        logging.info("Created forcings file: %s", output_file)
        # remove the working directory
        shutil.rmtree(forcing_working_dir)
        # print(f"Removing working directory: {cached_nc_path}")
        # cached_nc_path.unlink()

# @profile
def main() -> None:
    """Main function to generate forcing data for CAMELS basins."""
    time.sleep(0.01)
    args = parse_arguments()

    setup_logging(args.debug)

    logging.debug("debug works")

    # check if hydrofabric exists
    if not Path("../hfv3_conuscats.parquet").exists():
        cat_gdb = gpd.read_file(args.hydrofabric, layer="nwm_catchments_conus")
        cat_gdb.to_parquet("../hfv3_conuscats.parquet", index=False)
    else:
        cat_gdb = gpd.read_parquet("../hfv3_conuscats.parquet")

    # load basin and upstream dictionary
    with open(args.dictionary, 'r', encoding="utf-8") as f:
        camels_basins = json.load(f)

    start_time = args.start_date.strftime("%Y-%m-%d %H:%M")
    end_time = args.end_date.strftime("%Y-%m-%d %H:%M")

    for k, v in camels_basins.items():
        camels_basin = int(k)
        camels_upstreams = []

        # append upstream basins to list
        if int(k) not in cat_gdb['ID'].values:
            logging.debug("%s not in gdb", k)
            continue
        for upstream in v:
            if upstream not in cat_gdb['ID'].values:
                logging.debug("%s not in gdb", upstream)
                continue
            else:
                camels_upstreams.append(upstream)

        # add downstream basin to list of basins to compute
        to_compute = [camels_basin] + camels_upstreams
        geometries = []

        # get geometries for all basins in to_compute
        for basin in to_compute:
            geom = head_geom_selection(basin, cat_gdb)
            geometries.append(geom)

        # get union of all geometries
        geometries_dict = {'ID': to_compute, 'geometry': geometries}
        gdf_id = gpd.GeoDataFrame(geometries_dict, crs="EPSG:4326")
        gdf_id.head()
        total_geometries = gpd.GeoSeries(geometries).union_all()
        total_gdf = gpd.GeoDataFrame(crs="EPSG:4326", geometry=[total_geometries])

        # get raw gridded data for entire upstream region of basin
        cached_nc_path = Path(f"raw_output/{k}-raw-gridded-data.nc")
        aggregated_nc_path = Path(f"outputcamels/{k}/{k}-aggregated.nc")
        if not aggregated_nc_path.exists():
            if not cached_nc_path.exists():
                logging.debug("cached nc path: %s", cached_nc_path)
                merged_data = get_forcing_data(cached_nc_path,
                                            start_time,
                                            end_time,
                                            total_gdf)
            else:
                merged_data = xr.open_dataset(cached_nc_path)
                # process the catchment and all its upstreams

            process_catchment(gdf_id, args.output_dir, camels_basin, merged_data)

        
        if cached_nc_path.exists():
            # remove the cached nc file
            cached_nc_path.unlink()
            logging.debug("Removing cached nc file: %s", cached_nc_path)

if __name__ == "__main__":
    main()
