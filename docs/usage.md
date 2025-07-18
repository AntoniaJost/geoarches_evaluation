# CMORisation Pipeline – Quick Usage Guide

This pipeline processes reanalysis or model output data into AIMIP-compliant CMORised NetCDF files.

## 1. Set up Python environment

Activate existing environment:

```bash
source ~/your_path/aimip/bin/activate
````

Or create a new one:

```bash
python3 -m venv aimip
source aimip/bin/activate
pip install -r requirements.txt
```

## 2. Configure the pipeline

Edit `config.yaml`:

* Define input and output paths
* Set `years` as a list or string range (`["1979", "1980"]` or `"1979:1982"`)
* Specify variable/unit mappings, pressure levels, and template options

The 4 minimum things you have to adjust are:

```yaml
general:
  work_dir: "path_to/repository/data"
  means_changed_path: "same_path_to/repository/data/1_means/means_changed.json"
calc_means:
  source_dir: "link/to/your/input/data"
cmorise:
  cmor_template_base_dir: "path/to/templates"
```

The templates can be downloaded by running, in a separate notebook:

```python
import xarray as xr
import s3fs

fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': 'https://s3.eu-dkrz-1.dkrz.cloud'},anon=True)
fs.get('ai-mip/MPI-M/', './MPI-M-local/', recursive=True)
```
For further information, please check out [Nikolay Koldunov's instructions:](https://github.com/koldunovn/aimip/blob/main/data_read/data_reading_examples.ipynb)

## 3. Run the pipeline

Interactive run:

```bash
python src/pipeline.py --config config.yaml
```

Batch run with SLURM:

```bash
sbatch run_pipeline.sh
```

Example `run_pipeline.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=cmor_pipeline
#SBATCH --output=logs/pipeline.log
#SBATCH --error=logs/pipeline.log
#SBATCH --time=04:00:00
#SBATCH --partition=standard

module load python/3.11.5
source /work/bk1450/a270220/aimip/bin/activate

mkdir -p logs
python src/pipeline.py --config config.yaml
```

## 4. Output

Final files are written to:

```
data/4_cmorisation/{daily,monthly}/[var]/[var]_..._gr_YYYYMM.nc
```

Intermediate steps are cleaned if `delete_intermediate_outputs: true` is set.
