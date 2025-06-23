# NWM/CAMELS-based Streamflow Reconstruction using LSTM

This repository contains tools used to preprocess and run long-short term memory (LSTM) models to reconstruct streamflow in ungaged locations upstream of Catchment Attributes and MEteorology for Large-sample Studies (CAMELS) basins. Using a known gage value (the CAMELS USGS gaged streamflow) as an input into an LSTM, as well as other meteorological and geographical inputs sourced from the National Water Model (NWM) 3.0 retrospective dataset, we use the LSTM outputs to reconstruct streamflow at upstream ungaged locations.

The contents of this repository are chronologically ordered as follows:

- Data Collection
  - Link USGS/NWM
    - Converts USGS gages into their corresponding NWM reach IDs. Use `link_usgs_to_nwm.ipynb`.
  - Get Upstream Basins
    - Collects list of every NWM reach and basin upstream of a CAMELS basin. Use `get_upstream_basins.ipynb`.
  - Forcing Generation
    - Aggregates gridded time-series meteorological data for each NWM basin in our study area. Use `forcing_cli_camels.py`.
  - Attribute Generation
    - Collects static basin attributes for each NWM basin in our study area. Use `camels_attributes.ipynb`.
  - Streamflow
    - Adds streamflow values NWM retrospective to streamflow-specific basin datasets. Use `get_flow.py`. 
- Custom NeuralHydrology (NH) Classes
  - Custom-defined NH models and dataset classes, as well as training methods. `basetrainer.py` and `earlystopper.py` belong in `neuralhydrology/neuralhydrology/training`. `config.py` belongs in `neuralhydrology/neuralhydrology/utils`. `modifiedcudalstm.py` belongs in `neuralhydrology/neuralhydrology/modelzoo`. `nwm3retro.py` belongs in `neuralhydrology/neuralhydrology/datasetzoo`.
- Model Configurations
  - Example model configurations used in NH.

For specific usage instructions for each notebook or script, please view the header docstring or markdown cell.

## Data Outputs

NetCDF data outputs (forcings and streamflow) are stored at `s3://camels-nwm-reanalysis`. **Note: This bucket is incomplete**

## Installation
We recommend using two separate virtual environments, one for data collection and one for the LSTM. For data collection, a list of dependencies is listed in `01_dependencies.txt`. For the LSTM portion (sections 2 and 3), please follow the official [NeuralHydrology instructions](https://neuralhydrology.readthedocs.io/en/latest/index.html).

## Credits
Our code is heavily inspired by work done by Josh Cunningham (@JoshCu) and James Halgren (@jameshalgren). This code was authored by Quinn Lee (@quinnylee) and Sonam Lama (@slama0077).
