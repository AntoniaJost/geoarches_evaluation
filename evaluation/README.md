# Climate Evaluation Tools for Geoarches

This part of the repo allows to produce long rollouts for climate projects / free runing simulations. The goal is to enable evaluation with respect to common climate metrics and to produce outputs following the **AIMIP naming conventions**. These outputs can then be **cmorized wit the second part of the repository**. 

Preferably, the user of this code uses a directory structure adhering to the geoarches suggestions, i.e. **having directories named modelstore and evalstore**. 

To start a simulation, **rollout.py** and **configs/rollout.yaml** are needed. **rollout.py** works similar to the **main_hydra.py** file of geoarches. You can submit a job by using an sbatch script like the one given in this repository, i.e. rollout.sh. 

The config file should be self-explaining. 

Evaluation routines will be provided updated asap. 

For questions contact me via **robert.brunstein@ovgu.de**