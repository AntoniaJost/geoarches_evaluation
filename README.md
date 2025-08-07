# GEOARCHES EVALUATION PIPELINE

This repository consists of two parts: Evaluation and Cmorisation.

## Evaluation

tbd by Robert

## AIMIP CMORisation Pipeline

This folder of the repository provides a modular pipeline for preparing and CMORising reanalysis or model output data to meet AIMIP (AI Model Intercomparison Project) specifications. The pipeline performs multiple preprocessing steps and produces CMOR-compliant NetCDF files ready for submission.

### Features

- ✨ Compute **daily and monthly means** from raw input data
- ✨ Rename variables, adjust units, filter **pressure levels**, and handle AIMIP-specific conventions
- ✨ Split concatenated files into per-variable files
- ✨ CMORise data by using **template NetCDF files**, preserving metadata and structure

All steps are defined in individual Python modules and executed sequentially through a single `pipeline.py` entry point.

---

### Getting Started

See [`USAGE.md`](docs/USAGE.md) for a short step-by-step quickstart guide on running the pipeline.

---

### Configuration

The pipeline is fully controlled via a `config.yaml` file. It defines:

- Input and output directories  
- Which years to process
- Variable renaming and unit conversion rules  
- Pressure levels to retain  
- Metadata overrides for CMOR output  
- Paths to external scripts and templates  
- Logging options and intermediate output control  

---

### Dependencies

Make sure to install the required Python packages into a clean environment:

```bash
python3 -m venv aimip_env
source aimip_env/bin/activate
pip install -r requirements.txt
```

---

### Pipeline Steps

| Step                | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| `RenameVarsStep`    | Renames variables and dims, converts units, filters pressure levels     |
| `SplitVarsStep`     | Separates combined datasets into per-variable files                     |
| `CmoriseStep`       | Replaces data in CMOR templates and applies metadata overrides          |

Each step checks whether its outputs already exist and skips processing unless forced by change detection.

---

## License

Licensed under the Apache License 2.0.
