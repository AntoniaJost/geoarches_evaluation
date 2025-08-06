#!/bin/bash
SRC="/home/a/a270220/projects/cmorisation_v2/data/4_cmorisation"
DST="/work/bk1450/a270220/cmorised_awm"

for scale in daily monthly; do
    find "$SRC/$scale" -type f -name "*.nc" -exec bash -c '
        src="$1"
        SRC_BASE="$2"
        DST_BASE="$3"

        rel_path="${src#${SRC_BASE}/}"
        dest_file="${DST_BASE}/${rel_path}"

        mkdir -p "$(dirname "$dest_file")"
        cp -n "$src" "$dest_file"
    ' bash {} "$SRC" "$DST" \;
done
