#!/bin/bash
#
# TopoSZp-CUDA Experiment Script
# Recreates experiments from arxiv.org/abs/2602.17552 with the updated code
#
# Datasets:
#   2D: CESM-ATM (1800x3600) — 60 fields
#   3D: CESM-ATM-26 (26x1800x3600) — 28 fields (tripti + public)
#   3D: Miranda density (3072x3072x3072) — chunked processing
#
# Error bounds: 1e-3, 1e-4, 1e-5
# Metrics: compression ratio, throughput, FN/FP/FT, preservation rate
#
# Usage:
#   bash run_experiments.sh [quick|full|miranda]
#     quick   = 3 datasets × 3 error bounds per category (~5 min)
#     full    = all datasets × 3 error bounds (~60 min)
#     miranda = Miranda 3D chunked processing (~2 hrs)

set -e

MODE=${1:-quick}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/../build" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$RESULTS_DIR/$TIMESTAMP"
mkdir -p "$OUTDIR"

# ---- Data paths ----
DATA_2D="/lcrc/project/ECP-EZ/tripti/SDRBENCH-CESM-ATM-cleared-1800x3600/original_data"

# 3D CESM paths (try public first, fall back to tripti's copy)
DATA_3D_PUBLIC="/lcrc/project/ECP-EZ/public/compression/cesmbig-26x1800x3600"
DATA_3D_TRIPTI="/lcrc/project/ECP-EZ/tripti/SDRBENCH-CESM-ATM-26x1800x3600/original_data"

# Miranda 3D
DATA_MIRANDA="/lcrc/project/ECP-EZ/public/compression/SDRBENCH-Miranda-3072x3072x3072"

# ---- Parameters ----
ERROR_BOUNDS="1e-3 1e-4 1e-5"
BLOCK=64

# ---- CSV header ----
CSV_HEADER="Source,Dataset,Dims,ErrorBound,CompRatio,CompTime_ms,DecompTime_ms,MaxError,Orig_CPs,Preserved_CPs,PreservRate,FN,FP,FT,Maxima_Orig,Maxima_Pres,Minima_Orig,Minima_Pres,Saddle_Orig,Saddle_Pres"

CSV_2D="$OUTDIR/results_2d.csv"
CSV_3D="$OUTDIR/results_3d.csv"
CSV_MIRANDA="$OUTDIR/results_miranda.csv"
CSV_AVG="$OUTDIR/averages.csv"
LOG="$OUTDIR/experiment.log"

echo "TopoSZp-CUDA Experiment — $TIMESTAMP" | tee "$LOG"
echo "Mode: $MODE" | tee -a "$LOG"
echo "Build: $BUILD_DIR" | tee -a "$LOG"
echo "Error bounds: $ERROR_BOUNDS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# GPU info
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | tee -a "$LOG"
fi
echo "" | tee -a "$LOG"

###############################################################################
# Helper: parse test_cuda_topology output
###############################################################################
parse_2d_output() {
    local OUT="$1"
    comp_ratio=$(echo "$OUT" | sed -n 's/.*ratio: \([0-9.]*\).*/\1/p' | head -1)
    comp_time=$(echo "$OUT" | grep 'TIMING' | sed -n 's/.*TIMING comp_s=\([0-9.]*\).*/\1/p' | head -1)
    decomp_time=$(echo "$OUT" | grep 'TIMING' | sed -n 's/.*decomp_s=\([0-9.]*\).*/\1/p' | head -1)
    max_err=$(echo "$OUT" | grep 'Max pointwise error' | sed -n 's/.*: \([0-9.e+-]*\).*/\1/p' | head -1)
    orig_cps=$(echo "$OUT" | grep 'Original CPs:' | sed -n 's/.*CPs: \([0-9]*\).*/\1/p')
    pres_cps=$(echo "$OUT" | grep 'Preserved (same type)' | sed -n 's/.*: \([0-9]*\).*/\1/p')
    pres_rate=$(echo "$OUT" | grep 'Preserved (same type)' | sed -n 's/.*(\([0-9.]*\).*/\1/p')
    fn=$(echo "$OUT" | grep 'Lost (became regular)' | sed -n 's/.*: \([0-9]*\).*/\1/p')
    fp=$(echo "$OUT" | grep 'New CPs' | sed -n 's/.*: \([0-9]*\).*/\1/p')
    ft=$(echo "$OUT" | grep 'Type changed' | sed -n 's/.*: \([0-9]*\).*/\1/p')
    max_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Maxima:' | sed -n 's/.*Maxima: \([0-9]*\).*/\1/p' | head -1)
    min_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Minima:' | sed -n 's/.*Minima: \([0-9]*\).*/\1/p' | head -1)
    sad_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Saddles:' | sed -n 's/.*Saddles: \([0-9]*\).*/\1/p' | head -1)
    max_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Maxima:' | sed -n 's/.*Maxima: \([0-9]*\).*/\1/p' | head -1)
    min_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Minima:' | sed -n 's/.*Minima: \([0-9]*\).*/\1/p' | head -1)
    sad_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Saddles:' | sed -n 's/.*Saddles: \([0-9]*\).*/\1/p' | head -1)
    comp_ms=$(echo "$comp_time" | awk '{printf "%.2f", $1*1000}')
    decomp_ms=$(echo "$decomp_time" | awk '{printf "%.2f", $1*1000}')
}

###############################################################################
# Helper: parse test_cuda_topology_3d output
###############################################################################
parse_3d_output() {
    local OUT="$1"
    comp_ratio=$(echo "$OUT" | sed -n 's/.*ratio: \([0-9.]*\).*/\1/p' | head -1)
    comp_time=$(echo "$OUT" | grep 'TIMING' | sed -n 's/.*TIMING comp_s=\([0-9.]*\).*/\1/p' | head -1)
    decomp_time=$(echo "$OUT" | grep 'TIMING' | sed -n 's/.*decomp_s=\([0-9.]*\).*/\1/p' | head -1)
    max_err=$(echo "$OUT" | grep 'Max error' | sed -n 's/.*: \([0-9.e+-]*\).*/\1/p' | head -1)
    orig_cps=$(echo "$OUT" | grep 'Original:' | sed -n 's/.*Original: \([0-9]*\).*/\1/p')
    pres_cps=$(echo "$OUT" | grep 'Preserved:' | sed -n 's/.*Preserved: \([0-9]*\).*/\1/p' | head -1)
    pres_rate=$(echo "$OUT" | grep 'Preserved:' | sed -n 's/.*(\([0-9.]*\).*/\1/p' | head -1)
    fn=$(echo "$OUT" | grep 'Lost:' | sed -n 's/.*Lost: \([0-9]*\).*/\1/p')
    fp=$(echo "$OUT" | grep 'New:' | sed -n 's/.*New: \([0-9]*\).*/\1/p')
    ft=$(echo "$OUT" | grep 'Changed:' | sed -n 's/.*Changed: \([0-9]*\).*/\1/p')
    max_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Maxima:' | sed -n 's/.*Maxima: \([0-9]*\).*/\1/p' | head -1)
    min_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Minima:' | sed -n 's/.*Minima: \([0-9]*\).*/\1/p' | head -1)
    sad_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Saddles:' | sed -n 's/.*Saddles: \([0-9]*\).*/\1/p' | head -1)
    max_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Maxima:' | sed -n 's/.*Maxima: \([0-9]*\).*/\1/p' | head -1)
    min_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Minima:' | sed -n 's/.*Minima: \([0-9]*\).*/\1/p' | head -1)
    sad_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Saddles:' | sed -n 's/.*Saddles: \([0-9]*\).*/\1/p' | head -1)
    comp_ms=$(echo "$comp_time" | awk '{printf "%.2f", $1*1000}')
    decomp_ms=$(echo "$decomp_time" | awk '{printf "%.2f", $1*1000}')
}

###############################################################################
# 2D Experiments: CESM-ATM 1800x3600
###############################################################################
run_2d() {
    local ROWS=1800 COLS=3600
    echo "$CSV_HEADER" > "$CSV_2D"

    if [ "$MODE" = "quick" ]; then
        DATASETS_2D="CLDHGH CLDLOW FREQSH"
    else
        DATASETS_2D=$(ls "$DATA_2D"/*.dat 2>/dev/null | xargs -I{} basename {} .dat | sed 's/_1_1800_3600//' | sort)
    fi

    echo "========== 2D: CESM-ATM (${ROWS}x${COLS}) ==========" | tee -a "$LOG"
    for ds in $DATASETS_2D; do
        FILE="$DATA_2D/${ds}_1_${ROWS}_${COLS}.dat"
        [ ! -f "$FILE" ] && { echo "  SKIP: $ds (not found)" | tee -a "$LOG"; continue; }

        for eb in $ERROR_BOUNDS; do
            echo -n "  $ds eb=$eb ... " | tee -a "$LOG"
            OUT=$("$BUILD_DIR/test_cuda_topology" "$FILE" $ROWS $COLS $eb $BLOCK 2>&1) || true
            parse_2d_output "$OUT"
            echo "CESM-ATM-2D,$ds,${ROWS}x${COLS},$eb,$comp_ratio,$comp_ms,$decomp_ms,$max_err,$orig_cps,$pres_cps,$pres_rate,$fn,$fp,$ft,$max_orig,$max_pres,$min_orig,$min_pres,$sad_orig,$sad_pres" >> "$CSV_2D"
            echo "ratio=${comp_ratio}x  pres=${pres_rate}%  FN=$fn FP=$fp FT=$ft  t=${comp_ms}ms" | tee -a "$LOG"
            echo "$OUT" > "$OUTDIR/${ds}_2d_eb${eb}.log"
        done
    done
}

###############################################################################
# 3D Experiments: CESM-ATM 26x1800x3600
###############################################################################
run_3d_cesm() {
    local D1=26 D2=1800 D3=3600
    echo "$CSV_HEADER" > "$CSV_3D"

    # Determine data dir and extension
    local DATA_DIR EXT
    if [ -d "$DATA_3D_PUBLIC" ]; then
        DATA_DIR="$DATA_3D_PUBLIC"
    else
        DATA_DIR="$DATA_3D_TRIPTI"
    fi

    if [ "$MODE" = "quick" ]; then
        DATASETS_3D="CLOUD OMEGA T"
    else
        # Collect all unique field names from both .dat and .f32 files
        DATASETS_3D=$(ls "$DATA_DIR"/*_1_26_1800_3600.dat "$DATA_DIR"/*_1_26_1800_3600.f32 \
                       "$DATA_3D_TRIPTI"/*_1_26_1800_3600.dat "$DATA_3D_TRIPTI"/*_1_26_1800_3600.f32 \
                       2>/dev/null | xargs -I{} basename {} | sed 's/_1_26_1800_3600\.\(dat\|f32\)//' | sort -u)
    fi

    echo "" | tee -a "$LOG"
    echo "========== 3D: CESM-ATM (${D1}x${D2}x${D3}) ==========" | tee -a "$LOG"
    for ds in $DATASETS_3D; do
        # Try multiple paths and extensions
        FILE=""
        for dir in "$DATA_DIR" "$DATA_3D_TRIPTI"; do
            for ext in dat f32; do
                candidate="$dir/${ds}_1_${D1}_${D2}_${D3}.${ext}"
                [ -f "$candidate" ] && FILE="$candidate" && break 2
            done
        done

        # Some fields have d1=27 (CMFMC, CMFMCDZM)
        if [ -z "$FILE" ]; then
            for dir in "$DATA_DIR" "$DATA_3D_TRIPTI"; do
                for ext in dat f32; do
                    candidate="$dir/${ds}_1_27_${D2}_${D3}.${ext}"
                    if [ -f "$candidate" ]; then
                        FILE="$candidate"; D1_ACT=27; break 2
                    fi
                done
            done
        fi

        [ -z "$FILE" ] && { echo "  SKIP: $ds (not found)" | tee -a "$LOG"; continue; }

        local D1_USE=${D1_ACT:-$D1}

        for eb in $ERROR_BOUNDS; do
            echo -n "  $ds eb=$eb ... " | tee -a "$LOG"
            OUT=$("$BUILD_DIR/test_cuda_topology_3d" "$FILE" $D1_USE $D2 $D3 $eb $BLOCK 2>&1) || true
            parse_3d_output "$OUT"
            echo "CESM-ATM-3D,$ds,${D1_USE}x${D2}x${D3},$eb,$comp_ratio,$comp_ms,$decomp_ms,$max_err,$orig_cps,$pres_cps,$pres_rate,$fn,$fp,$ft,$max_orig,$max_pres,$min_orig,$min_pres,$sad_orig,$sad_pres" >> "$CSV_3D"
            echo "ratio=${comp_ratio}x  pres=${pres_rate}%  FN=$fn FP=$fp FT=$ft  t=${comp_ms}ms" | tee -a "$LOG"
            echo "$OUT" > "$OUTDIR/${ds}_3d_eb${eb}.log"
        done
        unset D1_ACT
    done
}

###############################################################################
# 3D Experiments: Miranda 3072x3072x3072 (chunked)
#   Split into slabs of CHUNK_D1 × 3072 × 3072
#   A100 40GB can handle ~64 × 3072 × 3072 float32 = ~2.3 GB comfortably
###############################################################################
run_miranda() {
    local FULL_D=3072
    local CHUNK_D1=64
    local D2=3072 D3=3072
    local TEMP_DIR="$OUTDIR/miranda_chunks"
    mkdir -p "$TEMP_DIR"

    echo "$CSV_HEADER" > "$CSV_MIRANDA"

    if [ "$MODE" = "quick" ]; then
        MIRANDA_FILES="density-00"
    else
        MIRANDA_FILES="density-00 density-10 density-20 density-30 density-40 density-50"
    fi

    echo "" | tee -a "$LOG"
    echo "========== 3D: Miranda (${FULL_D}x${FULL_D}x${FULL_D}, chunked ${CHUNK_D1}x${D2}x${D3}) ==========" | tee -a "$LOG"

    for ds in $MIRANDA_FILES; do
        FILE="$DATA_MIRANDA/${ds}.f32"
        [ ! -f "$FILE" ] && { echo "  SKIP: $ds (not found)" | tee -a "$LOG"; continue; }

        local FILESIZE=$(stat -c%s "$FILE" 2>/dev/null || stat -f%z "$FILE")
        local EXPECTED=$((FULL_D * FULL_D * FULL_D * 4))
        if [ "$FILESIZE" -ne "$EXPECTED" ]; then
            echo "  SKIP: $ds (size $FILESIZE != expected $EXPECTED)" | tee -a "$LOG"
            continue
        fi

        local NUM_CHUNKS=$((FULL_D / CHUNK_D1))
        local CHUNK_BYTES=$((CHUNK_D1 * D2 * D3 * 4))

        for eb in $ERROR_BOUNDS; do
            echo -n "  $ds eb=$eb (${NUM_CHUNKS} chunks) ... " | tee -a "$LOG"

            # Aggregate metrics across chunks
            local total_orig=0 total_pres=0 total_fn=0 total_fp=0 total_ft=0
            local total_comp_ms=0 total_decomp_ms=0
            local total_comp_bytes=0 total_orig_bytes=0
            local max_error=0
            local chunks_done=0

            for c in $(seq 0 $((NUM_CHUNKS - 1))); do
                local OFFSET=$((c * CHUNK_BYTES))
                local CHUNK_FILE="$TEMP_DIR/${ds}_chunk${c}.f32"

                # Extract chunk with dd
                dd if="$FILE" of="$CHUNK_FILE" bs=$CHUNK_BYTES skip=$c count=1 2>/dev/null

                OUT=$("$BUILD_DIR/test_cuda_topology_3d" "$CHUNK_FILE" $CHUNK_D1 $D2 $D3 $eb $BLOCK 2>&1) || true
                parse_3d_output "$OUT"

                total_orig=$((total_orig + ${orig_cps:-0}))
                total_pres=$((total_pres + ${pres_cps:-0}))
                total_fn=$((total_fn + ${fn:-0}))
                total_fp=$((total_fp + ${fp:-0}))
                total_ft=$((total_ft + ${ft:-0}))
                total_comp_ms=$(echo "$total_comp_ms + ${comp_ms:-0}" | bc)
                total_decomp_ms=$(echo "$total_decomp_ms + ${decomp_ms:-0}" | bc)
                chunks_done=$((chunks_done + 1))

                # Clean up chunk
                rm -f "$CHUNK_FILE"
            done

            # Compute aggregates
            if [ "$total_orig" -gt 0 ]; then
                agg_rate=$(echo "scale=2; $total_pres * 100 / $total_orig" | bc)
            else
                agg_rate="N/A"
            fi
            agg_comp_ms=$(echo "scale=2; $total_comp_ms" | bc)
            agg_decomp_ms=$(echo "scale=2; $total_decomp_ms" | bc)

            echo "Miranda,$ds,${FULL_D}x${FULL_D}x${FULL_D},$eb,N/A,$agg_comp_ms,$agg_decomp_ms,N/A,$total_orig,$total_pres,$agg_rate,$total_fn,$total_fp,$total_ft,,,,,," >> "$CSV_MIRANDA"

            echo "pres=${agg_rate}%  FN=$total_fn FP=$total_fp FT=$total_ft  t=${agg_comp_ms}ms (${chunks_done} chunks)" | tee -a "$LOG"
        done
    done
    rmdir "$TEMP_DIR" 2>/dev/null || true
}

###############################################################################
# Compute averages per error bound
###############################################################################
compute_averages() {
    echo "" | tee -a "$LOG"
    echo "========== AVERAGES PER ERROR BOUND ==========" | tee -a "$LOG"

    echo "Type,ErrorBound,Avg_CompRatio,Avg_CompTime_ms,Avg_DecompTime_ms,Avg_PreservRate,Avg_FN,Avg_FP,Avg_FT,Num_Datasets" > "$CSV_AVG"

    for csv_file in "$CSV_2D" "$CSV_3D" "$CSV_MIRANDA"; do
        [ ! -f "$csv_file" ] && continue
        local label=$(basename "$csv_file" .csv | sed 's/results_//')

        for eb in $ERROR_BOUNDS; do
            local lines=$(grep ",$eb," "$csv_file" 2>/dev/null | grep -v "^Source")
            local count=$(echo "$lines" | grep -c . 2>/dev/null || echo 0)
            [ "$count" -eq 0 ] && continue

            local avg_ratio=$(echo "$lines" | awk -F',' '{s+=$5; n++} END {if(n>0) printf "%.2f",s/n; else print "N/A"}')
            local avg_comp=$(echo "$lines" | awk -F',' '{s+=$6; n++} END {if(n>0) printf "%.2f",s/n; else print "N/A"}')
            local avg_decomp=$(echo "$lines" | awk -F',' '{s+=$7; n++} END {if(n>0) printf "%.2f",s/n; else print "N/A"}')
            local avg_pres=$(echo "$lines" | awk -F',' '{s+=$11; n++} END {if(n>0) printf "%.2f",s/n; else print "N/A"}')
            local avg_fn=$(echo "$lines" | awk -F',' '{s+=$12; n++} END {if(n>0) printf "%.1f",s/n; else print "N/A"}')
            local avg_fp=$(echo "$lines" | awk -F',' '{s+=$13; n++} END {if(n>0) printf "%.1f",s/n; else print "N/A"}')
            local avg_ft=$(echo "$lines" | awk -F',' '{s+=$14; n++} END {if(n>0) printf "%.1f",s/n; else print "N/A"}')

            echo "$label,$eb,$avg_ratio,$avg_comp,$avg_decomp,$avg_pres,$avg_fn,$avg_fp,$avg_ft,$count" >> "$CSV_AVG"
            echo "  $label  eb=$eb  ratio=${avg_ratio}x  pres=${avg_pres}%  FN=${avg_fn}  FP=${avg_fp}  FT=${avg_ft}  (n=$count)" | tee -a "$LOG"
        done
    done
}

###############################################################################
# Summary
###############################################################################
print_summary() {
    echo "" | tee -a "$LOG"
    echo "==================== FINAL SUMMARY ====================" | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    echo "--- Averages ---" | tee -a "$LOG"
    column -t -s',' "$CSV_AVG" 2>/dev/null | tee -a "$LOG"

    echo "" | tee -a "$LOG"
    echo "--- 2D Results (first 10) ---" | tee -a "$LOG"
    head -11 "$CSV_2D" | column -t -s',' 2>/dev/null | tee -a "$LOG"

    echo "" | tee -a "$LOG"
    echo "--- 3D Results (first 10) ---" | tee -a "$LOG"
    head -11 "$CSV_3D" | column -t -s',' 2>/dev/null | tee -a "$LOG"

    if [ -f "$CSV_MIRANDA" ] && [ $(wc -l < "$CSV_MIRANDA") -gt 1 ]; then
        echo "" | tee -a "$LOG"
        echo "--- Miranda Results ---" | tee -a "$LOG"
        column -t -s',' "$CSV_MIRANDA" 2>/dev/null | tee -a "$LOG"
    fi

    echo "" | tee -a "$LOG"
    echo "Results saved to: $OUTDIR" | tee -a "$LOG"
    echo "  2D CSV:      $CSV_2D" | tee -a "$LOG"
    echo "  3D CSV:      $CSV_3D" | tee -a "$LOG"
    [ -f "$CSV_MIRANDA" ] && echo "  Miranda CSV: $CSV_MIRANDA" | tee -a "$LOG"
    echo "  Averages:    $CSV_AVG" | tee -a "$LOG"
    echo "  Full log:    $LOG" | tee -a "$LOG"
}

###############################################################################
# Main
###############################################################################
case "$MODE" in
    quick|full)
        run_2d
        run_3d_cesm
        compute_averages
        print_summary
        ;;
    miranda)
        run_miranda
        compute_averages
        print_summary
        ;;
    all)
        run_2d
        run_3d_cesm
        run_miranda
        compute_averages
        print_summary
        ;;
    *)
        echo "Usage: $0 [quick|full|miranda|all]"
        exit 1
        ;;
esac
