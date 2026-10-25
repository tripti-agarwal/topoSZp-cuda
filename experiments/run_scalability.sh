#!/bin/bash
#
# Experiment 1: OpenMP Scalability (Table 2 in the paper)
# Runs TopoSZp+ compression with 1,2,4,8,16,32,36 threads
# on all 2D CESM datasets with eb=1e-3
#
# Usage: bash run_scalability.sh [quick|full]
#   quick = 5 representative fields
#   full  = all fields + all datasets

set -e

MODE=${1:-quick}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$RESULTS_DIR/scalability_$TIMESTAMP"
mkdir -p "$OUTDIR"

# Thread counts to test
THREADS="1 2 4 8 16 32 36"
EB="1e-3"
BLOCK=64

# Datasets
DATA_ATM="/lcrc/project/ECP-EZ/tripti/SDRBENCH-CESM-ATM-cleared-1800x3600/original_data"
DATA_3D="/lcrc/project/ECP-EZ/public/compression/cesmbig-26x1800x3600"

CSV="$OUTDIR/scalability.csv"
LOG="$OUTDIR/scalability.log"

echo "TopoSZp+ Scalability Experiment — $TIMESTAMP" | tee "$LOG"
echo "Threads: $THREADS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "Dataset,Field,Dims,Threads,CompTime_s,DecompTime_s,MaxError,PreservRate,FN,FP,FT" > "$CSV"

# CPU topology compress/decompress binary
TOPO_BIN="$BUILD_DIR/testfloat_compress_fastmode1"
DECOMP_BIN="$BUILD_DIR/testfloat_decompress_fastmode1"

# If the CPU binaries don't exist, use the CUDA test (which also runs OpenMP)
if [ ! -f "$TOPO_BIN" ]; then
    echo "CPU topology binary not found, using CUDA test program (includes OpenMP comparison)" | tee -a "$LOG"
    TOPO_BIN="$BUILD_DIR/test_cuda_topology"
    USE_CUDA_BIN=1
fi

###############################################################################
# 2D: ATM (1800x3600)
###############################################################################
if [ "$MODE" = "quick" ]; then
    FIELDS_2D="AEROD_v CLDHGH CLDLOW CLDMED FLDSC"
else
    FIELDS_2D=$(ls "$DATA_ATM"/*.dat 2>/dev/null | xargs -I{} basename {} .dat | sed 's/_1_1800_3600//' | sort)
fi

echo "========== 2D: ATM (1800x3600), eb=$EB ==========" | tee -a "$LOG"
for field in $FIELDS_2D; do
    FILE="$DATA_ATM/${field}_1_1800_3600.dat"
    [ ! -f "$FILE" ] && continue

    for t in $THREADS; do
        echo -n "  $field threads=$t ... " | tee -a "$LOG"
        export OMP_NUM_THREADS=$t

        if [ "${USE_CUDA_BIN:-0}" = "1" ]; then
            # Parse OpenMP timing from CUDA test output
            OUT=$("$TOPO_BIN" "$FILE" 1800 3600 $EB $BLOCK 2>&1) || true
            # Get OpenMP compression time
            comp_s=$(echo "$OUT" | grep -oP 'OpenMP.*Compressed.*in \K[0-9.]+' | head -1)
            decomp_s=$(echo "$OUT" | grep -oP 'OpenMP.*Decompressed in \K[0-9.]+' | head -1)
            max_err=$(echo "$OUT" | grep -oP 'Max pointwise error: \K[0-9.e+-]+' | head -1)
            pres_rate=$(echo "$OUT" | grep 'Preserved (same type)' | grep -oP '\(\K[0-9.]+' | head -1)
            fn=$(echo "$OUT" | grep 'Lost (became regular)' | grep -oP ': \K[0-9]+' | head -1)
            fp=$(echo "$OUT" | grep 'New CPs' | grep -oP ': \K[0-9]+' | head -1)
            ft=$(echo "$OUT" | grep 'Type changed' | grep -oP ': \K[0-9]+' | head -1)
            # Fall back to TIMING line
            if [ -z "$comp_s" ]; then
                comp_s=$(echo "$OUT" | grep -oP 'comp_s=\K[0-9.]+')
            fi
            if [ -z "$decomp_s" ]; then
                decomp_s=$(echo "$OUT" | grep -oP 'decomp_s=\K[0-9.]+')
            fi
        else
            OUT=$("$TOPO_BIN" "$FILE" 1800 3600 $EB $BLOCK 2>&1) || true
            comp_s=$(echo "$OUT" | grep -oP 'Compression.*in \K[0-9.]+')
            decomp_s=$(echo "$OUT" | grep -oP 'Decompression.*in \K[0-9.]+')
            max_err=$(echo "$OUT" | grep -oP 'Max error: \K[0-9.e+-]+')
            pres_rate="N/A"
            fn="N/A"; fp="N/A"; ft="N/A"
        fi

        echo "ATM-2D,$field,1800x3600,$t,${comp_s:-N/A},${decomp_s:-N/A},${max_err:-N/A},${pres_rate:-N/A},${fn:-N/A},${fp:-N/A},${ft:-N/A}" >> "$CSV"
        echo "comp=${comp_s:-N/A}s  decomp=${decomp_s:-N/A}s" | tee -a "$LOG"
        echo "$OUT" > "$OUTDIR/${field}_t${t}.log"
    done
done

###############################################################################
# 3D: ATM-26 (26x1800x3600) — only a few fields for scalability
###############################################################################
if [ "$MODE" = "full" ]; then
    FIELDS_3D="CLOUD T OMEGA"
    CUDA_3D="$BUILD_DIR/test_cuda_topology_3d"

    echo "" | tee -a "$LOG"
    echo "========== 3D: ATM-26 (26x1800x3600), eb=$EB ==========" | tee -a "$LOG"
    for field in $FIELDS_3D; do
        FILE=""
        for ext in dat f32; do
            [ -f "$DATA_3D/${field}_1_26_1800_3600.$ext" ] && FILE="$DATA_3D/${field}_1_26_1800_3600.$ext" && break
        done
        [ -z "$FILE" ] && continue

        for t in $THREADS; do
            echo -n "  $field threads=$t ... " | tee -a "$LOG"
            export OMP_NUM_THREADS=$t
            OUT=$("$CUDA_3D" "$FILE" 26 1800 3600 $EB $BLOCK 2>&1) || true
            comp_s=$(echo "$OUT" | grep -oP 'comp_s=\K[0-9.]+')
            decomp_s=$(echo "$OUT" | grep -oP 'decomp_s=\K[0-9.]+')
            echo "ATM-3D,$field,26x1800x3600,$t,${comp_s:-N/A},${decomp_s:-N/A},,,,," >> "$CSV"
            echo "comp=${comp_s:-N/A}s" | tee -a "$LOG"
        done
    done
fi

###############################################################################
# Summary
###############################################################################
echo "" | tee -a "$LOG"
echo "========== SCALABILITY SUMMARY ==========" | tee -a "$LOG"
column -t -s',' "$CSV" 2>/dev/null | head -40 | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Results: $CSV" | tee -a "$LOG"
