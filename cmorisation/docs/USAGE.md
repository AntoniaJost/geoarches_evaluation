# CMORisation Pipeline – Quick Usage Guide

This pipeline processes reanalysis or model output data into AIMIP-compliant CMORised NetCDF files.

## 1. Set up Python environment

Create an environment and activate it:

```bash
python3 -m venv aimip
source ~/your_path/aimip/bin/activate
pip install -r requirements.txt
```

## 2. Configure the pipeline

#### 2a. Edit `config.yaml`:

* Adjust all params in the `cmorise.global_attributes` part according to your likings.

#### 2b. Edit `run_pipeline.sh`:

* Adjust the sbatch configs as you need.
* Set the right path to your environment.
* Adjust ONLY:
  * ```MODEL_TAG```
  * ```NAME```
  * ```MEMBER```  
  * ```ENSEBLME```
  * ```TAG```
  * ```INPUT_DIR```
  * ```TIMESPAN```
  * ```TIMESPANS_DAILY```
  * ```AIMIP```
  * ```RUN_DIR```
  * ```REPO_DIR```
  * ```LOG_DIR```

Unless you want to run a specific / own configuration, there shouldn't be any need for touching any more code than this. 

## 3. Run the pipeline

Interactive run:

```bash
python src/pipeline.py --config config.yaml
```

Batch run with SLURM, but make sure that you are <span style="color:red; font-weight:bold;">inside the cmorisation folder!</span>

```bash
sbatch run_pipeline.sh
```

## 4. Output

Final files are written to:

```
data/4_cmorisation/{daily,monthly}/[var]/[var]_..._gn_YYYYMM.nc
```

Intermediate steps are cleaned if `delete_intermediate_outputs: true` is set in the `config.yaml`.
