# Climate Evaluation Tools for Geoarches

## General
This part of the repository serves as a drop-in replacement for tools like PCMDI.  The code relies only on python libraries for visualisation and evaluation. The code evaluates only **data available in cmorised format.**

The evaluation code is started via eval.py. The whole code uses hydra and relies 
on yaml files. The .yaml files are located in the config directory and contains an upper level config "config.yaml" and two directories "climdata" and "eval". "climadata" has a top-level config "data.yaml" merging all ".yaml" files in the subdirectory "models". In the same manner, "metrics.yaml" merges all the files in the metrics subdirectory of eval. 

## Model Configs
Each model config contains items ```path_to_daily_data``` and ```path_to_monthly_data```. If one of these is not available, the items are 
set to ```null```. For each model, a ```model_label``` item (used in legends)
is specified as well as a ```color``` item (in valid matplotlib format). The color item is used in e.g. line plots. Further, for each model a ```period``` item can be specified to select a data range from the xarray datasets, e.g. ```period: ["1979-01-01", "1980-01-01"]```.  
It is possible (and for many plots necessary) to define a reference model by  setting ```is_reference:  True``` in the corresponding model config file. All other 
data will be evaluted against this reference model.

Example files are given in the subdirectories. The user has to change the items. 

## Metrics and Plotting
Each metric is specified by a corresponding ".yaml" file. These fieles are collected
in the "configs/eval/metrics.yaml" file. 
Let us inspect "annual_cycle.yaml". 
```
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
The file starts with the name of the climate metric/ characteristic, here "AnnualCycle". As usual with hydra, a ```_target_``` item is used to instantiate the corresponding class. This is subject to change and will, in a later release, be replaced by an automatic tool to make the user independent of code knowledge. After that, the variables of interest are defined as a list of dictionaries, where each list entry containes the cmor variable name and the pressure level (```null``` for surface variables and a number in dPa for level variables).  Further specifications depend on the variable of choice. The plotter_kwargs define the output directory for the metric plots and the user can define other characteristics like linewidth and figsize. 

## Available Metrics

The following metric classes are available and can be referenced in metric yaml files via ```_target_```:

| Class | yaml file | Description |
|---|---|---|
| ```metrics.module.XYMaps``` | ```seasonal_means.yaml``` | Spatial mean maps per season or annual |
| ```metrics.module.XYBiasMaps``` | ```bias_maps.yaml``` | Spatial bias maps relative to the reference model |
| ```metrics.module.XYAnomalyMaps``` | — | Spatial anomaly maps |
| ```metrics.module.LatTimeMap``` | ```lat_time.yaml``` | Latitude–time Hovmöller diagrams |
| ```metrics.module.SeasonalCycles``` | ```annual_cycle.yaml``` | Annual and seasonal cycle timeseries |
| ```metrics.module.SouthernOscillationIndex``` | ```soi.yaml``` | Southern Oscillation Index |
| ```metrics.module.NorthernAnnularMode``` | ```nam.yaml``` | Northern Annular Mode (NAM/AO) |
| ```metrics.module.SouthernAnnularMode``` | — | Southern Annular Mode (SAM) |
| ```metrics.module.NorthernAtlanticOscillationIndex``` | ```nao.yaml``` | North Atlantic Oscillation (NAO) |
| ```metrics.module.MonsoonIndices``` | — | Monsoon indices (Webster–Yang and related) |
| ```metrics.module.RadialSpectrum``` | ```spectrum.yaml``` | Radial power spectrum |
| ```metrics.module.ZonalSpectrum``` | — | Zonal power spectrum |
| ```metrics.module.SmallScalesEnergy``` | — | Small-scale energy fraction |
| ```metrics.module.Histogram``` | ```hist.yaml``` | Variable distribution histograms |
| ```metrics.module.TropicalCycloneFrequency``` | ```tropical_cyclones.yaml``` | Tropical cyclone track frequency |

## Running the Evaluation

### Locally

After setting up the model and metric configs (see sections above), run the evaluation from inside the ```evaluation/``` directory:

```bash
python eval.py
```

Hydra will automatically pick up ```configs/config.yaml```.  Output plots are written under the path specified by ```output_path``` in ```config.yaml```.

To override any config value on the command line, use Hydra syntax:

```bash
python eval.py output_path=./evalstore/results/my_run
```

To evaluate only a subset of the enabled metrics, pass a list of metric keys as ```target_metrics```:

```bash
python eval.py 'target_metrics=[NorthernAtlanticOscillationIndex,NorthernAnnularMode]'
```

The metric keys must match the top-level names defined in the metric yaml files (e.g. ```NorthernAtlanticOscillationIndex``` in ```nao.yaml```).

### On a SLURM Cluster

Adjust the SLURM directives at the top of ```eval.sh``` (account, partition, time limit, …) and submit:

```bash
sbatch eval.sh
```

Standard output and error are written to ```logs/<job-name>.<job-id>.out``` and ```.err``` respectively.

## Step-by-Step Setup

1. **Add a model config** — Create a new yaml file in ```configs/climdata/models/```.  At minimum the file must contain:
   ```yaml
   my_model:
     path_to_daily_data: "/path/to/cmor/day"
     path_to_monthly_data: "/path/to/cmor/Amon"  # set to null if unavailable
     model_label: "My Model"
     color: [0.2, 0.4, 0.8]
     period: ["1979-01-01", "2014-12-31"]
   ```
   To use this model as the reference against which all others are evaluated, add ```is_reference: True```.

2. **Enable the model** — Add the file to the ```defaults``` list in ```configs/climdata/data.yaml```:
   ```yaml
   defaults:
     - _self_
     - models/my_model.yaml
     - models/era5_1x1.yaml
   ```

3. **Enable metrics** — Uncomment or add the desired metric yaml files in ```configs/eval/metrics.yaml```:
   ```yaml
   defaults:
     - _self_
     - metrics/annual_cycle.yaml
     - metrics/bias_maps.yaml
   ```

4. **Set the output path** — Edit ```output_path``` in ```configs/config.yaml```:
   ```yaml
   output_path: "./evalstore/results/my_experiment"
   ```

5. **Run** — Execute ```python eval.py``` locally or submit ```eval.sh``` to SLURM.



