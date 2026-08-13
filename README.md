# GEOARCHES EVALUATION PIPELINE

This repository consists of two parts: Evaluation and Cmorisation.

## Evaluation

Evaluating the arches models consists of two steps.
1. Rollout ArchesWeather or ArchesWeatherGen for a predefined time span to obtain climate projections. The rollout module will produce outputs that are named following the AIMIP standard. The rollout is currently started by running `rollout.py`. This is subject to change — see the issues for further information.
2. CMORise the rollout output (see [AIMIP CMORisation Pipeline](#aimip-cmorisation-pipeline) below) and evaluate it by loading the corresponding files and computing predefined metrics.

This part of the repository serves as a drop-in replacement for tools like PCMDI. The code relies only on python libraries for visualisation and evaluation, and evaluates only **data available in cmorised format**.

### Features

- ✨ Spatial **maps** — means, biases, anomalies, trends, pressure–latitude and Hovmöller cross sections
- ✨ Climate **indices** — SOI, NAM/SAM, NAO, monsoon indices
- ✨ **Timeseries, spectra, distributions and extreme-value** diagnostics
- ✨ Fully **Hydra/YAML-driven** configuration for models and metrics — no code changes needed to add a model or enable a metric
- ✨ Optional **quantitative baseline** scoring (RMSE, pattern correlation, bias, …) against a reference model

---

### Prerequisites

You need to symlink a couple of folders just as you do for [geoarches](https://geoarches.readthedocs.io/en/latest/getting_started/installation/#downloading-archesweather-and-archesweathergen), and provide ERA5 as input data. The following connections are needed for sure:

* a ```data/era5_240/full``` folder with 6 hourly era5 input netcdf files (contact Robert for the full path of the data on Levante).
* an ```evalstore``` folder containing the model input as netcdf files (again, contact Robert for full path if required). Structure has to be ```evalstore/{model_name}/{period}/{type}```, e.g. ```.../evalstore/archesweather-m-seed0-gc-sst_sic-weight_01/2000-01-01T12:00_2040-12-31T12:00/daily/...nc```
* a ```wandblogs``` folder

---

### Configuration

The evaluation is started via `eval.py` inside the `evaluation/` directory. The whole code uses Hydra and relies on YAML files located in the `configs` directory, which contains a top-level `config.yaml` and two subdirectories, `climdata` and `eval`. `climdata` has a top-level config `data.yaml` merging all `.yaml` files in its `models` subdirectory. In the same manner, `eval/metrics.yaml` merges all the files in the `eval/metrics` subdirectory.

**Model configs** — Each model config contains items `path_to_daily_data` and `path_to_monthly_data`. If one of these is not available, the item is set to `null`. For each model, a `model_label` item (used in legends) is specified as well as a `color` item (in valid matplotlib format), used e.g. in line plots. Further, for each model a `period` item can be specified to select a data range from the xarray datasets, e.g. `period: ["1979-01-01", "1980-01-01"]`. It is possible (and for many plots necessary) to define a reference model by setting `is_reference: True` in the corresponding model config file — all other models will be evaluated against this reference. Example files are given in `configs/climdata/models`.

**Metric configs** — Each metric is specified by a corresponding `.yaml` file, collected in `configs/eval/metrics.yaml`. For example, `annual_cycle.yaml`:
```yaml
AnnualCycle:
  _target_: metrics.module.SeasonalCycles
  variables:
    - {"name": "tos", "pressure_level": null}
    - {"name": "tas", "pressure_level": null}
    - {"name": "uas", "pressure_level": null}
    - {"name": "hus", "pressure_level": 70000}
  mean_groups:
    - "time.year"
    - "time.month"
  baseline_mean_groups:
    - "time.month"
  detrend: False
  linear_trend: True
  compute_anomalies: False
  baseline_period: ["1981-01-01", "2010-12-31"]
  plotter_kwargs:
    output_path: "timeseries/annual_cycle"
    figsize: [12, 8]
    linewidth: 1.0
```
The file starts with the name of the climate metric/characteristic, here `AnnualCycle`. As usual with Hydra, a `_target_` item is used to instantiate the corresponding class — this is subject to change and will, in a later release, be replaced by an automatic tool to make the user independent of code knowledge. After that, the variables of interest are defined as a list of dictionaries, where each entry contains the CMOR variable name and the pressure level (`null` for surface variables and a number in Pa for level variables). Further specifications depend on the variable of choice. The `plotter_kwargs` define the output directory for the metric plots and other characteristics like linewidth and figsize.

---

### Available Metrics

The following metric classes are available and can be referenced in metric yaml files via `_target_`:

| Class | yaml file | Description |
| --- | --- | --- |
| `metrics.module.XYMaps` | `seasonal_means.yaml` | Spatial mean maps per season or annual |
| `metrics.module.XYBiasMaps` | `bias_maps.yaml` | Spatial bias maps relative to the reference model |
| `metrics.module.XYAnomalyMaps` | — | Spatial anomaly maps |
| `metrics.module.XYTrendMaps` | `seasonal_trend.yaml` | Spatial linear-trend maps (regression slope per decade) |
| `metrics.module.LatTimeMap` | `lat_time.yaml` | Latitude–time Hovmöller diagrams |
| `metrics.module.TimeLongitudeMap` | `time_lon.yaml` | Longitude–time Hovmöller diagrams |
| `metrics.module.PressureLatMap` | `pressure_lat.yaml` | Zonal-mean pressure–latitude cross sections |
| `metrics.module.PressureLatBiasMap` | `pressure_lat_bias.yaml` | Pressure–latitude bias cross sections relative to the reference model |
| `metrics.module.LatitudeProfile` | `latitudinal.yaml` | Zonal-mean latitude line profiles |
| `metrics.module.VariableCorrelationMaps` | `variable_correlations.yaml` | Point-wise correlation maps between pairs of variables |
| `metrics.module.SeasonalCycles` | `annual_cycle.yaml` | Annual and seasonal cycle timeseries |
| `metrics.module.SouthernOscillationIndex` | `soi.yaml` | Southern Oscillation Index |
| `metrics.module.NorthernAnnularMode` | `nam.yaml` | Northern Annular Mode (NAM/AO) |
| `metrics.module.SouthernAnnularMode` | `sam.yaml` | Southern Annular Mode (SAM) |
| `metrics.module.NorthernAtlanticOscillationIndex` | `nao.yaml` | North Atlantic Oscillation (NAO) |
| `metrics.module.MonsoonIndices` | `webster_yang.yaml` | Monsoon indices (Webster–Yang and related) |
| `metrics.module.RadialSpectrum` | `spectrum.yaml` | Radial power spectrum |
| `metrics.module.Histogram` | `hist.yaml` | Variable distribution histograms |
| `metrics.module.ReturnPeriods` | `return_periods.yaml` | Block-maxima extreme-value (GEV) return periods |
| `metrics.module.QuantitativeBaseline` | `quantitative_baseline.yaml` | Aggregate quantitative skill scores from upstream metric outputs (disabled by default) |

---

### Running the Evaluation

#### Locally

After setting up the model and metric configs (see sections above), run the evaluation from inside the `evaluation/` directory:

```bash
python eval.py
```

Hydra will automatically pick up `configs/config.yaml`. Output plots are written under the path specified by `output_path` in `config.yaml`.

To override any config value on the command line, use Hydra syntax:

```bash
python eval.py output_path=./evalstore/results/my_run
```

To evaluate only a subset of the enabled metrics, pass a list of metric keys as `target_metrics`:

```bash
python eval.py 'target_metrics=[NorthernAtlanticOscillationIndex,NorthernAnnularMode]'
```

The metric keys must match the top-level names defined in the metric yaml files (e.g. `NorthernAtlanticOscillationIndex` in `nao.yaml`).

#### On a SLURM Cluster

Adjust the SLURM directives at the top of `eval.sh` (account, partition, time limit, …) and submit:

```bash
sbatch eval.sh
```

Standard output and error are written to `logs/<job-name>.<job-id>.out` and `.err` respectively.

---

### Step-by-Step Setup

1. **Add a model config** — Create a new yaml file in `configs/climdata/models/`. At minimum the file must contain:
   ```yaml
   my_model:
     path_to_daily_data: "/path/to/cmor/day"
     path_to_monthly_data: "/path/to/cmor/Amon"  # set to null if unavailable
     model_label: "My Model"
     color: [0.2, 0.4, 0.8]
     period: ["1979-01-01", "2014-12-31"]
   ```
   To use this model as the reference against which all others are evaluated, add `is_reference: True`.

2. **Enable the model** — Add the file to the `defaults` list in `configs/climdata/data.yaml`:
   ```yaml
   defaults:
     - _self_
     - models/my_model.yaml
     - models/era5_1x1.yaml
   ```

3. **Enable metrics** — Add the desired metric yaml files to the `defaults` list in `configs/eval/metrics.yaml`:
   ```yaml
   defaults:
     - _self_
     - metrics/annual_cycle.yaml
     - metrics/bias_maps.yaml
   ```

4. **Set the output path** — Edit `output_path` in `configs/config.yaml`:
   ```yaml
   output_path: "./evalstore/results/my_experiment"
   ```

5. **Run** — Execute `python eval.py` locally or submit `eval.sh` to SLURM.

---

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

See [`USAGE.md`](cmorisation/docs/USAGE.md) for a short step-by-step quickstart guide on running the pipeline.

---

### Configuration

The pipeline is fully controlled via a `config.yaml` and a `run_pipeline.sh` file. It defines:

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
