#!/bin/bash
#
# TopoSZp-CUDA Experiment Script
# Recreates experiments from arxiv.org/abs/2602.17552 with the updated code
#
# Original paper tested:
#   - 5 CESM datasets (2D): ATM(1800x3600), CLIMATE(768x1152), ICE(384x320), LAND(192x288), OCEAN(384x320)
#   - Error bounds: 1e-3, 1e-4, 1e-5
#   - Metrics: compression ratio, throughput, FN/FP/FT, preservation rate
#
# Updated code adds:
#   - 3D data support (CESM-ATM-26: 26x1800x3600)
#   - CUDA GPU acceleration
#   - Saddle stencil + RBF restoration
#   - PL (TTK-style) topology mode
#
# Usage:
#   bash run_experiments.sh [quick|full]
#     quick = 3 datasets × 3 error bounds (fast, ~5 min)
#     full  = all datasets × 3 error bounds (~30 min)

set -e

MODE=${1:-quick}
BUILD_DIR="$(cd "$(dirname "$0")/../build" && pwd)"
RESULTS_DIR="$(dirname "$0")/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$RESULTS_DIR/$TIMESTAMP"
mkdir -p "$OUTDIR"

# Paths
DATA_2D="/lcrc/project/ECP-EZ/tripti/SDRBENCH-CESM-ATM-cleared-1800x3600/original_data"
DATA_3D="/lcrc/project/ECP-EZ/tripti/SDRBENCH-CESM-ATM-26x1800x3600/original_data"

# Error bounds (same as paper)
ERROR_BOUNDS="1e-3 1e-4 1e-5"

# Block size
BLOCK=64

# 2D datasets (1800 x 3600 = 6,480,000 elements)
ROWS_2D=1800
COLS_2D=3600

# 3D datasets (26 x 1800 x 3600 = 168,480,000 elements)
D1_3D=26
D2_3D=1800
D3_3D=3600

if [ "$MODE" = "quick" ]; then
    DATASETS_2D="CLDHGH CLDLOW FREQSH"
    DATASETS_3D="CLOUD OMEGA T"
else
    # Full: all 2D fields available
    DATASETS_2D=$(ls "$DATA_2D"/*.dat 2>/dev/null | xargs -I{} basename {} .dat | sed 's/_1_1800_3600//' | sort)
    DATASETS_3D=$(ls "$DATA_3D"/*.f32 2>/dev/null | xargs -I{} basename {} .f32 | sed 's/_1_26_1800_3600//' | sort)
fi

# CSV output files
CSV_2D="$OUTDIR/results_2d.csv"
CSV_3D="$OUTDIR/results_3d.csv"
CSV_SUMMARY="$OUTDIR/summary.csv"
LOG="$OUTDIR/experiment.log"

echo "TopoSZp-CUDA Experiment — $TIMESTAMP" | tee "$LOG"
echo "Mode: $MODE" | tee -a "$LOG"
echo "Build: $BUILD_DIR" | tee -a "$LOG"
echo "Error bounds: $ERROR_BOUNDS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# GPU info
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$LOG"
fi
echo "" | tee -a "$LOG"

###############################################################################
# 2D Experiments
###############################################################################
echo "Dataset,ErrorBound,CompRatio,CompTime_ms,DecompTime_ms,MaxError,Orig_CPs,Preserved_CPs,PreservRate,FN,FP,FT,Maxima_Orig,Maxima_Pres,Minima_Orig,Minima_Pres,Saddle_Orig,Saddle_Pres" > "$CSV_2D"

echo "========== 2D EXPERIMENTS (${ROWS_2D}x${COLS_2D}) ==========" | tee -a "$LOG"
for ds in $DATASETS_2D; do
    FILE="$DATA_2D/${ds}_1_${ROWS_2D}_${COLS_2D}.dat"
    if [ ! -f "$FILE" ]; then
        echo "SKIP: $FILE not found" | tee -a "$LOG"
        continue
    fi

    for eb in $ERROR_BOUNDS; do
        echo -n "  $ds eb=$eb ... " | tee -a "$LOG"

        OUT=$("$BUILD_DIR/test_cuda_topology" "$FILE" $ROWS_2D $COLS_2D $eb $BLOCK 2>&1)

        # Parse output
        comp_ratio=$(echo "$OUT" | grep -oP 'ratio: \K[0-9.]+' | head -1)
        comp_time=$(echo "$OUT" | grep -oP 'comp_s=\K[0-9.]+')
        decomp_time=$(echo "$OUT" | grep -oP 'decomp_s=\K[0-9.]+')
        max_err=$(echo "$OUT" | grep 'Max pointwise error' | grep -oP ': \K[0-9.e+-]+' | head -1)
        orig_cps=$(echo "$OUT" | grep 'Original CPs:' | grep -oP 'CPs: \K[0-9]+')
        pres_cps=$(echo "$OUT" | grep 'Preserved (same type)' | grep -oP ': \K[0-9]+')
        pres_rate=$(echo "$OUT" | grep 'Preserved (same type)' | grep -oP '\(\K[0-9.]+')
        fn=$(echo "$OUT" | grep 'Lost (became regular)' | grep -oP ': \K[0-9]+')
        fp=$(echo "$OUT" | grep 'New CPs' | grep -oP ': \K[0-9]+')
        ft=$(echo "$OUT" | grep 'Type changed' | grep -oP ': \K[0-9]+')

        # Per-type counts from "Found X critical points" section
        max_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Maxima:' | grep -oP 'Maxima: \K[0-9]+' | head -1)
        min_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Minima:' | grep -oP 'Minima: \K[0-9]+' | head -1)
        sad_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Saddles:' | grep -oP 'Saddles: \K[0-9]+' | head -1)
        max_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Maxima:' | grep -oP 'Maxima: \K[0-9]+' | head -1)
        min_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Minima:' | grep -oP 'Minima: \K[0-9]+' | head -1)
        sad_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Saddles:' | grep -oP 'Saddles: \K[0-9]+' | head -1)

        # Convert seconds to ms
        comp_ms=$(echo "$comp_time" | awk '{printf "%.2f", $1*1000}')
        decomp_ms=$(echo "$decomp_time" | awk '{printf "%.2f", $1*1000}')

        echo "$ds,$eb,$comp_ratio,$comp_ms,$decomp_ms,$max_err,$orig_cps,$pres_cps,$pres_rate,$fn,$fp,$ft,$max_orig,$max_pres,$min_orig,$min_pres,$sad_orig,$sad_pres" >> "$CSV_2D"

        echo "ratio=${comp_ratio}x  preserved=${pres_rate}%  FN=$fn FP=$fp FT=$ft  comp=${comp_ms}ms" | tee -a "$LOG"

        # Save full output
        echo "$OUT" > "$OUTDIR/${ds}_2d_eb${eb}.log"
    done
done

###############################################################################
# 3D Experiments
###############################################################################
echo "Dataset,ErrorBound,CompRatio,CompTime_ms,DecompTime_ms,MaxError,Orig_CPs,Preserved_CPs,PreservRate,FN,FP,FT,Maxima_Orig,Maxima_Pres,Minima_Orig,Minima_Pres,Saddle_Orig,Saddle_Pres" > "$CSV_3D"

echo "" | tee -a "$LOG"
echo "========== 3D EXPERIMENTS (${D1_3D}x${D2_3D}x${D3_3D}) ==========" | tee -a "$LOG"
for ds in $DATASETS_3D; do
    FILE="$DATA_3D/${ds}_1_${D1_3D}_${D2_3D}_${D3_3D}.f32"
    if [ ! -f "$FILE" ]; then
        echo "SKIP: $FILE not found" | tee -a "$LOG"
        continue
    fi

    for eb in $ERROR_BOUNDS; do
        echo -n "  $ds eb=$eb ... " | tee -a "$LOG"

        OUT=$("$BUILD_DIR/test_cuda_topology_3d" "$FILE" $D1_3D $D2_3D $D3_3D $eb $BLOCK 2>&1)

        # Parse output (3D format slightly different)
        comp_ratio=$(echo "$OUT" | grep -oP 'ratio: \K[0-9.]+' | head -1)
        comp_time=$(echo "$OUT" | grep -oP 'comp_s=\K[0-9.]+')
        decomp_time=$(echo "$OUT" | grep -oP 'decomp_s=\K[0-9.]+')
        max_err=$(echo "$OUT" | grep 'Max error' | grep -oP ': \K[0-9.e+-]+' | head -1)
        orig_cps=$(echo "$OUT" | grep 'Original:' | grep -oP 'Original: \K[0-9]+')
        pres_cps=$(echo "$OUT" | grep 'Preserved:' | grep -oP 'Preserved: \K[0-9]+' | head -1)
        pres_rate=$(echo "$OUT" | grep 'Preserved:' | grep -oP '\(\K[0-9.]+' | head -1)
        fn=$(echo "$OUT" | grep 'Lost:' | grep -oP 'Lost: \K[0-9]+')
        fp=$(echo "$OUT" | grep 'New:' | grep -oP 'New: \K[0-9]+')
        ft=$(echo "$OUT" | grep 'Changed:' | grep -oP 'Changed: \K[0-9]+')

        max_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Maxima:' | grep -oP 'Maxima: \K[0-9]+' | head -1)
        min_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Minima:' | grep -oP 'Minima: \K[0-9]+' | head -1)
        sad_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Saddles:' | grep -oP 'Saddles: \K[0-9]+' | head -1)
        max_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Maxima:' | grep -oP 'Maxima: \K[0-9]+' | head -1)
        min_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Minima:' | grep -oP 'Minima: \K[0-9]+' | head -1)
        sad_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Saddles:' | grep -oP 'Saddles: \K[0-9]+' | head -1)

        comp_ms=$(echo "$comp_time" | awk '{printf "%.2f", $1*1000}')
        decomp_ms=$(echo "$decomp_time" | awk '{printf "%.2f", $1*1000}')

        echo "$ds,$eb,$comp_ratio,$comp_ms,$decomp_ms,$max_err,$orig_cps,$pres_cps,$pres_rate,$fn,$fp,$ft,$max_orig,$max_pres,$min_orig,$min_pres,$sad_orig,$sad_pres" >> "$CSV_3D"

        echo "ratio=${comp_ratio}x  preserved=${pres_rate}%  FN=$fn FP=$fp FT=$ft  comp=${comp_ms}ms" | tee -a "$LOG"

        echo "$OUT" > "$OUTDIR/${ds}_3d_eb${eb}.log"
    done
done

###############################################################################
# Summary
###############################################################################
echo "" | tee -a "$LOG"
echo "========== RESULTS SUMMARY ==========" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "--- 2D Results ---" | tee -a "$LOG"
column -t -s',' "$CSV_2D" 2>/dev/null | head -20 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- 3D Results ---" | tee -a "$LOG"
column -t -s',' "$CSV_3D" 2>/dev/null | head -20 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Results saved to: $OUTDIR" | tee -a "$LOG"
echo "  2D CSV: $CSV_2D" | tee -a "$LOG"
echo "  3D CSV: $CSV_3D" | tee -a "$LOG"
echo "  Full log: $LOG" | tee -a "$LOG"
echo "  Per-run logs: $OUTDIR/*.log" | tee -a "$LOG"
