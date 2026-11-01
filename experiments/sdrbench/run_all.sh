#!/usr/bin/env bash
# Run every TopoSZp+ experiment on the SDRBench data and make the tables.
#
#   DATA=/path/to/sdrbench-data ./run_all.sh [RESULTS_DIR] [extra run_experiments.py options]
#
# Steps (each is skipped if already done):
#   1. setup.sh               build TopoSZp+ (OpenMP + CUDA), checkers, SZ3 / SZ2 / ZFP
#   2. download_sdrbench.sh   only if $DATA/manifest.csv does not exist
#   3. run_experiments.py     all experiments; resumes if RESULTS_DIR already has results
#   4. summarize.py           summary.md, tables.tex, plots/
#
# Environment:
#   DATA      data folder (default: ./sdrbench-data next to this script)
#   EXPS      experiments to run (default: main,scaling,baselines,bitrate,pl,vis)
#   WORK      scratch folder for compressed / decompressed files (default: RESULTS_DIR/work);
#             a node-local disk is faster for the 3D fields
#   *_SRC     use data you already have (see download_sdrbench.sh), e.g. CESM_ATM_3D_SRC=/lcrc/...
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${DATA:-$HERE/sdrbench-data}
OUT=${1:-$HERE/results/$(date +%Y%m%d_%H%M%S)}; shift || true
EXPS=${EXPS:-main,scaling,baselines,bitrate,pl,vis}

[ -x "$HERE/bin/toposzp_comp" ] || "$HERE/setup.sh" "$HERE/bin" || { echo "setup failed: see $HERE/bin/setup.log"; exit 1; }
[ -f "$DATA/manifest.csv" ] || "$HERE/download_sdrbench.sh" "$DATA" || exit 1
[ "$(wc -l < "$DATA/manifest.csv")" -gt 1 ] || { echo "no fields in $DATA/manifest.csv"; exit 1; }

python3 "$HERE/run_experiments.py" --data "$DATA" --out "$OUT" --bin "$HERE/bin" --exp "$EXPS" \
        ${WORK:+--work "$WORK"} "$@"
python3 "$HERE/summarize.py" "$OUT"
