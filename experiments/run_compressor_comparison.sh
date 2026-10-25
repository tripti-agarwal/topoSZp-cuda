#!/bin/bash
#
# Experiment 2: Comparison against other compressors (Table 3, Figure 10)
# Compresses with SZ3, ZFP, Tthresh, SZ; decompresses; counts FN/FP/FT
# using the compare_critical_points tool
#
# Usage: bash run_compressor_comparison.sh [quick|full]
#   quick = ATM dataset only, 3 error bounds
#   full  = ATM, CLIMATE, ICE, LAND, OCEAN × 3 error bounds
#
# Prerequisites:
#   - Install compressors: bash install_compressors.sh
#   - Build TopoSZp: cmake/make in build/
#   - compare_critical_points built in build/

set -e

MODE=${1:-quick}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$RESULTS_DIR/comparison_$TIMESTAMP"
TMPDIR="$OUTDIR/tmp"
mkdir -p "$OUTDIR" "$TMPDIR"

# Compressor binaries (adjust paths as needed)
COMP_DIR="${COMPRESSOR_DIR:-/lcrc/project/ECP-EZ/tripti/compressors}"
SZ3_BIN=$(which sz3 2>/dev/null || echo "$COMP_DIR/bin/sz3")
ZFP_BIN=$(which zfp 2>/dev/null || echo "$COMP_DIR/bin/zfp")
TTHRESH_BIN=$(which tthresh 2>/dev/null || echo "$COMP_DIR/bin/tthresh")
SZ_BIN=$(which sz 2>/dev/null || echo "$COMP_DIR/bin/sz")
COMPARE_BIN="$BUILD_DIR/compare_critical_points"
TOPOSZP_BIN="$BUILD_DIR/test_cuda_topology"

ERROR_BOUNDS="1e-3 1e-4 1e-5"

# Datasets (Table 3 in paper)
declare -A DATASETS
DATASETS[ATM]="/lcrc/project/ECP-EZ/tripti/SDRBENCH-CESM-ATM-cleared-1800x3600/original_data|1800|3600"

if [ "$MODE" = "full" ]; then
    # Add more datasets if the data exists
    # CLIMATE, ICE, LAND, OCEAN paths - adjust to actual locations
    for ds_dir in /lcrc/project/ECP-EZ/tripti/SDRBENCH-CESM-*; do
        [ -d "$ds_dir/original_data" ] || continue
        name=$(basename "$ds_dir" | sed 's/SDRBENCH-CESM-//' | sed 's/-cleared//' | sed 's/-.*$//')
        # Try to detect dimensions from filename
        sample=$(ls "$ds_dir/original_data"/*.dat 2>/dev/null | head -1)
        if [ -n "$sample" ]; then
            dims=$(basename "$sample" | sed -n 's/.*_\([0-9]*_[0-9]*\)\.dat/\1/p')
            if [ -n "$dims" ]; then
                rows=$(echo "$dims" | cut -d_ -f1)
                cols=$(echo "$dims" | cut -d_ -f2)
                DATASETS[$name]="$ds_dir/original_data|$rows|$cols"
            fi
        fi
    done
fi

CSV="$OUTDIR/comparison.csv"
CSV_AVG="$OUTDIR/comparison_averages.csv"
LOG="$OUTDIR/comparison.log"

echo "Compressor Comparison Experiment — $TIMESTAMP" | tee "$LOG"
echo "Mode: $MODE" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Check which compressors are available
echo "Available compressors:" | tee -a "$LOG"
for name_bin in "SZ3:$SZ3_BIN" "ZFP:$ZFP_BIN" "Tthresh:$TTHRESH_BIN" "SZ:$SZ_BIN" "TopoSZp+:$TOPOSZP_BIN" "compare:$COMPARE_BIN"; do
    name=${name_bin%%:*}
    bin=${name_bin#*:}
    if [ -f "$bin" ] && [ -x "$bin" ]; then
        echo "  ✓ $name: $bin" | tee -a "$LOG"
    else
        echo "  ✗ $name: NOT FOUND at $bin" | tee -a "$LOG"
    fi
done
echo "" | tee -a "$LOG"

echo "Dataset,Compressor,ErrorBound,Avg_FN,Avg_FP,Avg_FT,Num_Fields,CompRatio,CompTime_s" > "$CSV"

###############################################################################
# Helper: compress with a given compressor, decompress, count FN/FP/FT
###############################################################################
run_compressor() {
    local compressor="$1" input="$2" rows="$3" cols="$4" eb="$5" output_prefix="$6"
    local nbele=$((rows * cols))
    local comp_file="${output_prefix}.compressed"
    local decomp_file="${output_prefix}.decompressed"
    local comp_time="N/A"
    local comp_size="N/A"
    local fn=0 fp=0 ft=0

    case "$compressor" in
        SZ3)
            if [ -f "$SZ3_BIN" ]; then
                local t0=$(date +%s%N)
                "$SZ3_BIN" -f -i "$input" -z "$comp_file" -2 $cols $rows -M ABS $eb 2>/dev/null || return 1
                local t1=$(date +%s%N)
                comp_time=$(echo "scale=6; ($t1 - $t0) / 1000000000" | bc)
                comp_size=$(stat -c%s "$comp_file" 2>/dev/null || stat -f%z "$comp_file")
                "$SZ3_BIN" -f -z "$comp_file" -o "$decomp_file" -2 $cols $rows 2>/dev/null || return 1
            else
                return 1
            fi
            ;;
        ZFP)
            if [ -f "$ZFP_BIN" ]; then
                local t0=$(date +%s%N)
                "$ZFP_BIN" -t float -2 $cols $rows -a $eb -i "$input" -z "$comp_file" 2>/dev/null || return 1
                local t1=$(date +%s%N)
                comp_time=$(echo "scale=6; ($t1 - $t0) / 1000000000" | bc)
                comp_size=$(stat -c%s "$comp_file" 2>/dev/null || stat -f%z "$comp_file")
                "$ZFP_BIN" -t float -2 $cols $rows -z "$comp_file" -o "$decomp_file" 2>/dev/null || return 1
            else
                return 1
            fi
            ;;
        Tthresh)
            if [ -f "$TTHRESH_BIN" ]; then
                local t0=$(date +%s%N)
                "$TTHRESH_BIN" "$input" -c "$comp_file" -t float -e $eb -s "$rows $cols" 2>/dev/null || return 1
                local t1=$(date +%s%N)
                comp_time=$(echo "scale=6; ($t1 - $t0) / 1000000000" | bc)
                comp_size=$(stat -c%s "$comp_file" 2>/dev/null || stat -f%z "$comp_file")
                "$TTHRESH_BIN" "$comp_file" -d "$decomp_file" 2>/dev/null || return 1
            else
                return 1
            fi
            ;;
        SZ)
            if [ -f "$SZ_BIN" ]; then
                local t0=$(date +%s%N)
                "$SZ_BIN" -z -f -i "$input" -M ABS -A $eb -2 $rows $cols 2>/dev/null || return 1
                local t1=$(date +%s%N)
                comp_time=$(echo "scale=6; ($t1 - $t0) / 1000000000" | bc)
                comp_file="${input}.sz"
                comp_size=$(stat -c%s "$comp_file" 2>/dev/null || stat -f%z "$comp_file")
                "$SZ_BIN" -x -f -s "$comp_file" -2 $rows $cols 2>/dev/null || return 1
                decomp_file="${comp_file}.out"
            else
                return 1
            fi
            ;;
    esac

    # Count FN/FP/FT using compare_critical_points
    # Output format: "FN FP FT Total DecompCount" (5 space-separated integers)
    if [ -f "$decomp_file" ] && [ -f "$COMPARE_BIN" ]; then
        local cmp_out=$("$COMPARE_BIN" "$input" "$decomp_file" $rows $cols $eb 2>/dev/null) || true
        fn=$(echo "$cmp_out" | tail -1 | awk '{print $1}')
        fp=$(echo "$cmp_out" | tail -1 | awk '{print $2}')
        ft=$(echo "$cmp_out" | tail -1 | awk '{print $3}')
        # Validate they're numbers
        case "$fn" in ''|*[!0-9]*) fn=0 ;; esac
        case "$fp" in ''|*[!0-9]*) fp=0 ;; esac
        case "$ft" in ''|*[!0-9]*) ft=0 ;; esac
    fi

    # Compute compression ratio
    local orig_size=$((nbele * 4))
    local ratio="N/A"
    if [ "$comp_size" != "N/A" ] && [ "$comp_size" -gt 0 ] 2>/dev/null; then
        ratio=$(echo "scale=2; $orig_size / $comp_size" | bc)
    fi

    # Clean up temp files
    rm -f "$comp_file" "$decomp_file" "${input}.sz" "${input}.sz.out" 2>/dev/null

    echo "${fn:-0}|${fp:-0}|${ft:-0}|${ratio}|${comp_time}"
}

###############################################################################
# Main loop
###############################################################################
COMPRESSORS="SZ3 ZFP Tthresh SZ"

for ds_name in "${!DATASETS[@]}"; do
    IFS='|' read -r data_dir rows cols <<< "${DATASETS[$ds_name]}"

    echo "========== $ds_name (${rows}x${cols}) ==========" | tee -a "$LOG"

    for eb in $ERROR_BOUNDS; do
        # Get list of data files
        files=$(ls "$data_dir"/*.dat 2>/dev/null | head -100)
        num_fields=0

        for compressor in $COMPRESSORS; do
            total_fn=0; total_fp=0; total_ft=0; field_count=0

            for f in $files; do
                field_name=$(basename "$f" .dat | sed "s/_1_${rows}_${cols}//")
                prefix="$TMPDIR/${ds_name}_${field_name}_${compressor}_${eb}"

                result=$(run_compressor "$compressor" "$f" "$rows" "$cols" "$eb" "$prefix" 2>/dev/null) || continue

                IFS='|' read -r fn fp ft ratio comp_time <<< "$result"
                total_fn=$((total_fn + ${fn:-0}))
                total_fp=$((total_fp + ${fp:-0}))
                total_ft=$((total_ft + ${ft:-0}))
                field_count=$((field_count + 1))
            done

            if [ "$field_count" -gt 0 ]; then
                avg_fn=$(echo "scale=2; $total_fn / $field_count" | bc)
                avg_fp=$(echo "scale=2; $total_fp / $field_count" | bc)
                avg_ft=$(echo "scale=2; $total_ft / $field_count" | bc)
            else
                avg_fn="N/A"; avg_fp="N/A"; avg_ft="N/A"
            fi

            echo "$ds_name,$compressor,$eb,$avg_fn,$avg_fp,$avg_ft,$field_count,$ratio,$comp_time" >> "$CSV"
            echo "  $compressor eb=$eb: FN=$avg_fn FP=$avg_fp FT=$avg_ft (n=$field_count)" | tee -a "$LOG"
        done

        # Also run TopoSZp+
        if [ -f "$TOPOSZP_BIN" ]; then
            total_fn=0; total_fp=0; total_ft=0; field_count=0
            for f in $files; do
                OUT=$("$TOPOSZP_BIN" "$f" $rows $cols $eb 64 2>&1) || continue
                fn=$(echo "$OUT" | grep 'Lost (became regular)' | sed -n 's/.*: \([0-9]*\).*/\1/p' | head -1)
                fp=$(echo "$OUT" | grep 'New CPs' | sed -n 's/.*: \([0-9]*\).*/\1/p' | head -1)
                ft=$(echo "$OUT" | grep 'Type changed' | sed -n 's/.*: \([0-9]*\).*/\1/p' | head -1)
                total_fn=$((total_fn + ${fn:-0}))
                total_fp=$((total_fp + ${fp:-0}))
                total_ft=$((total_ft + ${ft:-0}))
                field_count=$((field_count + 1))
            done
            if [ "$field_count" -gt 0 ]; then
                avg_fn=$(echo "scale=2; $total_fn / $field_count" | bc)
                avg_fp=$(echo "scale=2; $total_fp / $field_count" | bc)
                avg_ft=$(echo "scale=2; $total_ft / $field_count" | bc)
            else
                avg_fn="N/A"; avg_fp="N/A"; avg_ft="N/A"
            fi
            echo "$ds_name,TopoSZp+,$eb,$avg_fn,$avg_fp,$avg_ft,$field_count,," >> "$CSV"
            echo "  TopoSZp+ eb=$eb: FN=$avg_fn FP=$avg_fp FT=$avg_ft (n=$field_count)" | tee -a "$LOG"
        fi
    done
done

echo "" | tee -a "$LOG"
echo "Results: $CSV" | tee -a "$LOG"
echo "Done." | tee -a "$LOG"
