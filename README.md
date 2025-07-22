# AIMIP CMORisation Pipeline

This repository provides a modular pipeline for preparing and CMORising reanalysis or model output data to meet AIMIP (AI Model Intercomparison Project) specifications. The pipeline performs multiple preprocessing steps and produces CMOR-compliant NetCDF files ready for submission.

## Features

- ✨ Compute **daily and monthly means** from raw input data
- ✨ Rename variables, adjust units, filter **pressure levels**, and handle AIMIP-specific conventions
- ✨ Split concatenated files into per-variable files
- ✨ Regrid variables from native resolution (`gn`) to target grid (`gr`)
- ✨ CMORise data by using **template NetCDF files**, preserving metadata and structure

All steps are defined in individual Python modules and executed sequentially through a single `pipeline.py` entry point.

> ℹ️ **Note:** Parts of the CMORisation logic are based on code from [Nikolay Koldunov's AIMIP repository](https://github.com/koldunovn/aimip/tree/main), especially `cmor_utils.py` and `native_to_1degree.py`. All credits go to him.

---

## Getting Started

See [`USAGE.md`](docs/USAGE.md) for a short step-by-step quickstart guide on running the pipeline.

---

## Configuration

The pipeline is fully controlled via a `config.yaml` file. It defines:

- Input and output directories  
- Which years to process
- Variable renaming and unit conversion rules  
- Pressure levels to retain  
- Metadata overrides for CMOR output  
- Paths to external scripts and templates  
- Logging options and intermediate output control  

---

## Dependencies

Make sure to install the required Python packages into a clean environment:

```bash
python3 -m venv aimip_env
source aimip_env/bin/activate
pip install -r requirements.txt
```

---

## Pipeline Steps

| Step                | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| `CalcMeansStep`     | Computes daily (limited to 1978–1979) and monthly means from ERA5 files |
| `RenameVarsStep`    | Renames variables and dims, converts units, filters pressure levels     |
| `SplitVarsStep`     | Separates combined datasets into per-variable files                     |
| `RunRegriddingStep` | Applies regridding script and moves output to correct subfolders        |
| `CmoriseStep`       | Replaces data in CMOR templates and applies metadata overrides          |

Each step checks whether its outputs already exist and skips processing unless forced by change detection.

---

## Known Bottlenecks

With the newly integrated option of selecting whether to cmorise on native or 1x1 degree grid comes a bottleneck that has to be treated manually (for the moment).
If you set either ```use_native``` or ```use_regridded``` to ```true``` for your first run, then switch to the other one and run it **for a different year**, you will face an issue in your third run as now your ```data/4_cmorisation/[var]``` ```gn``` and ```gr``` folders contain different time spans. It is recommended to just run ```rm -r data/4_cmorisation/*``` and do another clean run from scratch.

---

## Attribution

This pipeline incorporates ideas and adapted code from:

* [Nikolay Koldunov’s AIMIP repository](https://github.com/koldunovn/aimip/tree/main)

Please cite his work where appropriate.

---

## License

Licensed under the Apache License 2.0.
