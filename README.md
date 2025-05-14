# NWM/CAMELS-based Streamflow Reconstruction using LSTM

This repository contains tools used to preprocess and run long-short term memory (LSTM) models to reconstruct streamflow in ungaged locations upstream of Catchment Attributes and MEteorology for Large-sample Studies (CAMELS) basins. Using a known gage value (the CAMELS USGS gaged streamflow) as an input into an LSTM, as well as other meteorological and geographical inputs sourced from the National Water Model (NWM) 3.0 retrospective dataset, we use the LSTM outputs to reconstruct streamflow at upstream ungaged locations.

The contents of this repository are as follows:

- Data Collection
  - Link USGS/NWM
    - Converts USGS gages into their corresponding NWM reach IDs.
  - Get Upstream Basins
    - Collects list of every NWM reach and basin upstream of a CAMELS basin.
  - Forcing Generation
    - Aggregates gridded time-series meteorological data for each NWM basin in our study area.
  - Attribute Generation
    - Collects static basin attributes for each NWM basin in our study area.
  - Streamflow
    - Appends streamflow values from NWM retrospective to basin datasets.
- Custom NeuralHydrology (NH) Classes
  - Custom-defined NH models and dataset classes, as well as training methods.
- Model Configurations
  - Example model configurations used in NH.

## Installation
We recommend using two separate virtual environments. For data collection, a list of dependencies is listed in `01_dependencies.txt`. For the LSTM portion (sections 2 and 3), please follow the official [NeuralHydrology instructions](https://neuralhydrology.readthedocs.io/en/latest/index.html).

## Credits
Our code is heavily inspired by work done by Josh Cunningham (@JoshCu) and James Halgren (@jameshalgren). This code was authored by Quinn Lee (@quinnylee) and Sonam Lama (@slama0077).