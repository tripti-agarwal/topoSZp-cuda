#!/bin/bash
#
# Experiment 3: Bit Rate vs False Cases (Figure 10 in the paper)
# Sweeps error bounds from 1e-2 to 1e-7 for each compressor
# to generate bit-rate vs FN/FP/FT data for plotting
#
# Usage: bash run_bitrate_sweep.sh [quick|full]

set -e

MODE=${1:-quick}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$RESULTS_DIR/bitrate_$TIMESTAMP"
TMPDIR="$OUTDIR/tmp"
mkdir -p "$OUTDIR" "$TMPDIR"

COMP_DIR="${COMPRESSOR_DIR:-/lcrc/project/ECP-EZ/tripti/compressors}"
SZ3_BIN=$(which sz3 2>/dev/null || echo "$COMP_DIR/bin/sz3")
ZFP_BIN=$(which zfp 2>/dev/null || echo "$COMP_DIR/bin/zfp")
TTHRESH_BIN=$(which tthresh 2>/dev/null || echo "$COMP_DIR/bin/tthresh")
SZ_BIN=$(which sz 2>/dev/null || echo "$COMP_DIR/bin/sz")
TOPOSZP_BIN="$BUILD_DIR/test_cuda_topology"
COMPARE_BIN="$BUILD_DIR/compare_critical_points"

# Error bounds for sweep (gives range of bit rates)
ERROR_BOUNDS="1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6"
if [ "$MODE" = "quick" ]; then
    ERROR_BOUNDS="1e-2 1e-3 1e-4 1e-5 1e-6"
fi

# ATM dataset (matching paper Figure 10)
DATA_DIR="/lcrc/project/ECP-EZ/tripti/SDRBENCH-CESM-ATM-cleared-1800x3600/original_data"
ROWS=1800
COLS=3600
NBELE=$((ROWS * COLS))

CSV="$OUTDIR/bitrate_sweep.csv"
LOG="$OUTDIR/bitrate.log"

echo "Bit Rate Sweep Experiment — $TIMESTAMP" | tee "$LOG"
echo "" | tee -a "$LOG"

echo "Compressor,ErrorBound,Avg_BitRate,Avg_FN,Avg_FP,Avg_FT,Avg_Total,Num_Fields" > "$CSV"

if [ "$MODE" = "quick" ]; then
    FILES=$(ls "$DATA_DIR"/{CLDHGH,CLDLOW,CLDMED,AEROD_v,FREQSH}_1_1800_3600.dat 2>/dev/null)
else
    FILES=$(ls "$DATA_DIR"/*.dat 2>/dev/null)
fi

for eb in $ERROR_BOUNDS; do
    echo "=== Error bound: $eb ===" | tee -a "$LOG"

    # --- TopoSZp+ ---
    if [ -f "$TOPOSZP_BIN" ]; then
        total_fn=0; total_fp=0; total_ft=0; total_bits=0; count=0
        for f in $FILES; do
            OUT=$("$TOPOSZP_BIN" "$f" $ROWS $COLS $eb 64 2>&1) || continue
            ratio=$(echo "$OUT" | sed -n 's/.*ratio: \([0-9.]*\).*/\1/p' | head -1)
            fn=$(echo "$OUT" | grep 'Lost (became regular)' | sed -n 's/.*: \([0-9]*\).*/\1/p' | head -1)
            fp=$(echo "$OUT" | grep 'New CPs' | sed -n 's/.*: \([0-9]*\).*/\1/p' | head -1)
            ft=$(echo "$OUT" | grep 'Type changed' | sed -n 's/.*: \([0-9]*\).*/\1/p' | head -1)
            if [ -n "$ratio" ] && [ "$ratio" != "0" ]; then
                bits=$(echo "scale=4; 32 / $ratio" | bc)
                total_bits=$(echo "$total_bits + $bits" | bc)
            fi
            total_fn=$((total_fn + ${fn:-0}))
            total_fp=$((total_fp + ${fp:-0}))
            total_ft=$((total_ft + ${ft:-0}))
            count=$((count + 1))
        done
        if [ "$count" -gt 0 ]; then
            avg_br=$(echo "scale=2; $total_bits / $count" | bc)
            avg_fn=$(echo "scale=2; $total_fn / $count" | bc)
            avg_fp=$(echo "scale=2; $total_fp / $count" | bc)
            avg_ft=$(echo "scale=2; $total_ft / $count" | bc)
            total_false=$(echo "scale=2; ($total_fn + $total_fp + $total_ft) / $count" | bc)
            echo "TopoSZp+,$eb,$avg_br,$avg_fn,$avg_fp,$avg_ft,$total_false,$count" >> "$CSV"
            echo "  TopoSZp+: br=$avg_br FN=$avg_fn FP=$avg_fp FT=$avg_ft" | tee -a "$LOG"
        fi
    fi

    # --- Other compressors ---
    for compressor in SZ3 SZ; do
        case "$compressor" in
            SZ3)    BIN="$SZ3_BIN" ;;
            SZ)     BIN="$SZ_BIN" ;;
        esac
        [ ! -f "$BIN" ] && { echo "  $compressor: SKIPPED (not installed)" | tee -a "$LOG"; continue; }

        total_fn=0; total_fp=0; total_ft=0; total_bits=0; count=0
        for f in $FILES; do
            field=$(basename "$f" .dat)
            comp_file="$TMPDIR/${field}_${compressor}_${eb}.compressed"
            decomp_file="$TMPDIR/${field}_${compressor}_${eb}.decompressed"

            # Compress and decompress
            case "$compressor" in
                SZ3)
                    "$SZ3_BIN" -f -i "$f" -z "$comp_file" -2 $COLS $ROWS -M ABS $eb 2>/dev/null || continue
                    "$SZ3_BIN" -f -z "$comp_file" -o "$decomp_file" -2 $COLS $ROWS 2>/dev/null || continue
                    ;;
                SZ)
                    "$SZ_BIN" -z -f -i "$f" -M ABS -A $eb -2 $ROWS $COLS 2>/dev/null || continue
                    comp_file="${f}.sz"
                    "$SZ_BIN" -x -f -s "$comp_file" -2 $ROWS $COLS 2>/dev/null || continue
                    decomp_file="${comp_file}.out"
                    ;;
            esac

            # Compute bit rate
            comp_size=$(stat -c%s "$comp_file" 2>/dev/null || stat -f%z "$comp_file" 2>/dev/null || echo 0)
            if [ "$comp_size" -gt 0 ] 2>/dev/null; then
                bits=$(echo "scale=4; $comp_size * 8 / $NBELE" | bc)
                total_bits=$(echo "$total_bits + $bits" | bc)
            fi

            # Count FN/FP/FT (output: "FN FP FT Total DecompCount")
            if [ -f "$decomp_file" ] && [ -f "$COMPARE_BIN" ]; then
                cmp_out=$("$COMPARE_BIN" "$f" "$decomp_file" $ROWS $COLS $eb 2>&1) || true
                last_line=$(echo "$cmp_out" | grep -E '^[0-9]+ [0-9]+ [0-9]+' | tail -1)
                fn=$(echo "$last_line" | awk '{print $1}')
                fp=$(echo "$last_line" | awk '{print $2}')
                ft=$(echo "$last_line" | awk '{print $3}')
                case "$fn" in ''|*[!0-9]*) fn=0 ;; esac
                case "$fp" in ''|*[!0-9]*) fp=0 ;; esac
                case "$ft" in ''|*[!0-9]*) ft=0 ;; esac
                total_fn=$((total_fn + ${fn:-0}))
                total_fp=$((total_fp + ${fp:-0}))
                total_ft=$((total_ft + ${ft:-0}))
            fi

            count=$((count + 1))
            rm -f "$comp_file" "$decomp_file" "${f}.sz" "${f}.sz.out" 2>/dev/null
        done

        if [ "$count" -gt 0 ]; then
            avg_br=$(echo "scale=2; $total_bits / $count" | bc)
            avg_fn=$(echo "scale=2; $total_fn / $count" | bc)
            avg_fp=$(echo "scale=2; $total_fp / $count" | bc)
            avg_ft=$(echo "scale=2; $total_ft / $count" | bc)
            total_false=$(echo "scale=2; ($total_fn + $total_fp + $total_ft) / $count" | bc)
            echo "$compressor,$eb,$avg_br,$avg_fn,$avg_fp,$avg_ft,$total_false,$count" >> "$CSV"
            echo "  $compressor: br=$avg_br FN=$avg_fn FP=$avg_fp FT=$avg_ft" | tee -a "$LOG"
        fi
    done
done

rm -rf "$TMPDIR"
echo "" | tee -a "$LOG"
echo "Results: $CSV" | tee -a "$LOG"
