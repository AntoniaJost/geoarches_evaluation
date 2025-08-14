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

* Define the path where your data will be stored (```work_dir```; should ideally be ```.../geoarches_evaluation/cmorisation/data```).
* Similarly, adjust ```means_changed_path``` to ```.../geoarches_evaluation/cmorisation/data/1_means/means_changed.json```.
* You can also specify variable/unit mappings and pressure levels, but I'd recommend to leave them as they are.

The 2 minimum things you have to adjust here are:

```yaml
general:
  work_dir: "path_to/repository/data"
  means_changed_path: "same_path_to/repository/data/1_means/means_changed.json"
```

#### 2b. Edit `run_pipeline.sh`:

* Set the right path to your environment.
* Adjust
  * ```TAG```
  * ```INPUT_DIR```
  * ```BASE_OUT```
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
