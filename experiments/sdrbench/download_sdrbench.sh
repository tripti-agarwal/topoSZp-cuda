#!/usr/bin/env bash
# Download SDRBench datasets (https://sdrbench.github.io) and write a manifest
# of the fields the experiments use.
#
#   ./download_sdrbench.sh [DATA_DIR] [DATASET ...]
#
#   DATA_DIR  where the data go (default: ./sdrbench-data)
#   DATASET   any of: CESM-ATM-2D CESM-ATM-3D Hurricane NYX SCALE-LETKF
#             (default: all five)
#
# Use data you already have instead of downloading it: set <NAME>_SRC to a
# folder containing the raw files, e.g.
#   CESM_ATM_2D_SRC=/lcrc/project/ECP-EZ/tripti/SDRBENCH-CESM-ATM-cleared-1800x3600/original_data \
#   CESM_ATM_3D_SRC=/lcrc/project/ECP-EZ/public/compression/cesmbig-26x1800x3600 \
#   ./download_sdrbench.sh /path/to/data
# (variables: CESM_ATM_2D_SRC CESM_ATM_3D_SRC HURRICANE_SRC NYX_SRC SCALE_LETKF_SRC).
# The files are linked, not copied.
#
# Archives are streamed into tar, so no extra space is needed for the .tar.gz.
# Each dataset is verified: only files whose size matches the dataset's
# dimensions (float32) are used. Result: DATA_DIR/manifest.csv with
#   dataset,field,path,d1,d2,d3      (d1 slowest dimension; d3 empty for 2D)
# which run_experiments.py reads.
#
# If a download URL has moved, give it explicitly:
#   URL_NYX=https://.../SDRBENCH-EXASKY-NYX-512x512x512.tar.gz ./download_sdrbench.sh
# (variables: URL_CESM_ATM_2D URL_CESM_ATM_3D URL_HURRICANE URL_NYX URL_SCALE_LETKF).

set -u
DATA=${1:-./sdrbench-data}; shift || true
WANT=("$@"); [ ${#WANT[@]} -gt 0 ] || WANT=(CESM-ATM-2D CESM-ATM-3D Hurricane NYX SCALE-LETKF)
mkdir -p "$DATA"; DATA=$(cd "$DATA" && pwd)
LOG=$DATA/download.log
say() { echo "$*" | tee -a "$LOG"; }

# SDRBench Globus mirrors (the first is the one the repo's earlier script used).
H1=https://g-d0cd3f.fd635.8443.data.globus.org/raw-data
H2=https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data

# name | dims (slowest first) | env prefix | archive paths (tried in order on both mirrors)
DATASETS=(
  "CESM-ATM-2D|1800x3600|CESM_ATM_2D|CESM-ATM/SDRBENCH-CESM-ATM-cleared-1800x3600.tar.gz CESM-ATM/SDRBENCH-CESM-ATM-1800x3600.tar.gz"
  "CESM-ATM-3D|26x1800x3600|CESM_ATM_3D|CESM-ATM/SDRBENCH-CESM-ATM-26x1800x3600.tar.gz"
  "Hurricane|100x500x500|HURRICANE|Hurricane-ISABEL/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz"
  "NYX|512x512x512|NYX|EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz"
  "SCALE-LETKF|98x1200x1200|SCALE_LETKF|SCALE_LETKF/SDRBENCH-SCALE-98x1200x1200.tar.gz SCALE_LETKF/SDRBENCH-SCALE_98x1200x1200.tar.gz"
)

fetch() {  # fetch <dest dir> <url>: stream the archive into tar
  local dst=$1 url=$2
  if command -v curl >/dev/null; then
    curl -fL --retry 3 --retry-delay 5 -sS "$url" | tar -xz -C "$dst"
  else
    wget -q -O - "$url" | tar -xz -C "$dst"
  fi
  local rc=("${PIPESTATUS[@]}")
  [ "${rc[0]}" = 0 ] && [ "${rc[1]}" = 0 ]
}

say "SDRBench download into $DATA  ($(date))"
for spec in "${DATASETS[@]}"; do
  IFS='|' read -r name dims env paths <<< "$spec"
  [[ " ${WANT[*]} " == *" $name "* ]] || continue
  dir=$DATA/$name; mkdir -p "$dir/raw"
  src_var=${env}_SRC; url_var=URL_${env}
  if [ -n "${!src_var:-}" ]; then
    say "== $name: linking files from ${!src_var}"
    find "${!src_var}" -maxdepth 2 -type f \( -name '*.f32' -o -name '*.dat' -o -name '*.bin' \) -exec ln -sf {} "$dir/raw/" \;
  elif [ -n "$(find "$dir/raw" -type f -o -type l 2>/dev/null | head -1)" ]; then
    say "== $name: already present in $dir/raw, not downloading"
  else
    urls=()
    [ -n "${!url_var:-}" ] && urls+=("${!url_var}")
    for p in $paths; do urls+=("$H1/$p" "$H2/$p"); done
    ok=0
    for u in "${urls[@]}"; do
      say "== $name: downloading $u"
      if fetch "$dir/raw" "$u"; then ok=1; break; fi
      say "   failed, trying the next URL"
      rm -rf "$dir/raw"; mkdir -p "$dir/raw"
    done
    if [ $ok = 0 ]; then
      say "!! $name: every URL failed. Download the archive from https://sdrbench.github.io and set $url_var=<url>,"
      say "   or extract it yourself and set $src_var=<folder>."
      continue
    fi
  fi
  # verify sizes and add to the manifest
  IFS=x read -r -a D <<< "$dims"; n=1; for d in "${D[@]}"; do n=$((n * d)); done; want=$((n * 4))
  good=0; bad=0
  : > "$dir/fields.csv"
  while IFS= read -r f; do
    sz=$(stat -L -c %s "$f" 2>/dev/null || echo 0)
    if [ "$sz" = "$want" ]; then
      fld=$(basename "$f"); fld=${fld%.f32}; fld=${fld%.dat}; fld=${fld%.bin}
      echo "$name,$fld,$(readlink -f "$f"),${D[0]},${D[1]},${D[2]:-}" >> "$dir/fields.csv"; good=$((good + 1))
    else
      bad=$((bad + 1))
    fi
  done < <(find -L "$dir/raw" -type f \( -name '*.f32' -o -name '*.dat' -o -name '*.bin' \) | sort)
  say "   $name: $good fields of $dims float32$( [ $bad -gt 0 ] && echo ", $bad other files ignored (size does not match)")"
done

{ echo "dataset,field,path,d1,d2,d3"; cat "$DATA"/*/fields.csv 2>/dev/null; } > "$DATA/manifest.csv"
say "Manifest: $DATA/manifest.csv ($(($(wc -l < "$DATA/manifest.csv") - 1)) fields)"
say "Disk use: $(du -shL "$DATA" 2>/dev/null | cut -f1)"
