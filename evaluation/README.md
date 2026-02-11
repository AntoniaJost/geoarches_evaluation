# Climate Evaluation Tools for Geoarches

This part of the repository serves as a drop-in replacement for tools like PCMDI. 
The aim is to allow the user to use python libraries only for visualisation and
evaluation. 

The evaluation code is started via eval.py. The whole code uses hydra and thus 
allows to flexibly work with yaml files. 

In the config directory, there is a config.yaml. This file is the main config
and combines all information about the data, found in configs/models/data.yaml
and all information about the desired metrics, found in configs/eval/metrics.yaml

In configs/models there is a dedicated data directory. For each model / ground truth 
a separate .yaml file is placed in this directory. The data.yaml file collects the 
desired files and combines them into a single file. 

Similar for all desired metrics, the metrics.yaml file collect all desired metrics.

When running eval.py, the GeoClimate class is instantiated. Within this class, 
so called CMORDataContainers are created. Each CMORDataContainer contains the
data of a model, together with the labels, linewidths, colors and other attributes
connected to the model data. Further, the GeoClimate class contains instances
of all metrics / evaluation targets. 

Within eval.py, GeoClimate.eval() is called and all the quantitaive and 
qualitative measures of climate are evaluated.

