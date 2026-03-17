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






