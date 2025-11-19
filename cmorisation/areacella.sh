#!/bin/bash
#SBATCH --job-name=areacella
#SBATCH --output=logs/areacella_%x_%j.out
#SBATCH --error=logs/areacella_%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=compute
#SBATCH --account=bk1450

set -euo pipefail

BASE="/work/bk1450/a270220/cmorised_awm/archesgen-m-gc-sst_sic-weight_01_member01"
OUTDIR="${BASE}/areacella"

DATASET="ArchesWeatherGen"
EXP="aimip"
ENSEMBLE="r1i1p1f1"
GRID="gn"

mkdir -p "${OUTDIR}/Amon/${GRID}"
mkdir -p "${OUTDIR}/day/${GRID}"

# --- Amon ---
AMON_FILE=$(find ${BASE}/4_cmorisation/Amon -name "*_${DATASET}_${EXP}_${ENSEMBLE}_${GRID}_*.nc" | head -n 1)
if [[ -n "$AMON_FILE" ]]; then
  echo "Creating areacella (Amon) from $AMON_FILE"
  cdo gridarea "$AMON_FILE" areacella_tmp.nc
  cdo chname,cell_area,areacella areacella_tmp.nc \
    "${OUTDIR}/Amon/${GRID}/areacella_fx_${DATASET}_${EXP}_${ENSEMBLE}_${GRID}.nc"
  rm -f areacella_tmp.nc
fi

# --- day ---
DAY_FILE=$(find ${BASE}/4_cmorisation/day -name "*_${DATASET}_${EXP}_${ENSEMBLE}_${GRID}_*.nc" | head -n 1)
if [[ -n "$DAY_FILE" ]]; then
  echo "Creating areacella (day) from $DAY_FILE"
  cdo gridarea "$DAY_FILE" areacella_tmp.nc
  cdo chname,cell_area,areacella areacella_tmp.nc \
    "${OUTDIR}/day/${GRID}/areacella_fx_${DATASET}_${EXP}_${ENSEMBLE}_${GRID}.nc"
  rm -f areacella_tmp.nc
fi

echo "Done: areacella files written under ${OUTDIR}/Amon/${GRID} and ${OUTDIR}/day/${GRID}"
