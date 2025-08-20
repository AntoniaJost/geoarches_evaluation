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

* Add the path where your cloned respository lies (`repo_dir`).
* Define the path where your data will be stored (```log_dir```). 
* Adjust all params in the `cmorise.global_attributes` part according to your likings.
* You can also specify variable/unit mappings and pressure levels, but if you want to be CF-compliant, I'd recommend to leave them as they are.
* ℹ️ The param `unit_mapping.time_slice_daily` limits your daily runs. It will not process (daily) data beyond the end date that you put here.

The 2 minimum things you have to adjust here are:

```yaml
general:
  repo_dir: "path/to/repository/"
  log_dir: "where/you/want/your/results/to/be"
```

#### 2b. Edit `run_pipeline.sh`:

* Set the right path to your environment.
* Adjust the sbatch configs as you need.
* Adjust
  * ```MODEL_TAG```
  * ```ENSEBLME```
  * ```TAG```
  * ```INPUT_DIR```
  * ```TIMESPAN```


## 3. Run the pipeline

Interactive run:

```bash
python src/pipeline.py --config config.yaml
```

Batch run with SLURM, but make sure that you are <span style="color:red; font-weight:bold;">inside the cmorisation folder!!</span>:

```bash
sbatch run_pipeline.sh
```

## 4. Output

Final files are written to:

```
data/4_cmorisation/{daily,monthly}/[var]/[var]_..._gn_YYYYMM.nc
```

Intermediate steps are cleaned if `delete_intermediate_outputs: true` is set.
