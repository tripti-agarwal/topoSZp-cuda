#!/bin/bash
#
# Experiment 4: GPU vs OpenMP Timing Comparison (Table 4 in the paper)
# Runs TopoSZp+ on 2D and 3D fields with both CUDA and OpenMP backends
# with absolute and relative error bounds
#
# Usage: bash run_gpu_vs_cpu.sh [quick|full]

set -e

MODE=${1:-quick}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$RESULTS_DIR/gpu_vs_cpu_$TIMESTAMP"
mkdir -p "$OUTDIR"

TOPO_2D="$BUILD_DIR/test_cuda_topology"
TOPO_3D="$BUILD_DIR/test_cuda_topology_3d"

# Data paths
DATA_2D="/lcrc/project/ECP-EZ/tripti/SDRBENCH-CESM-ATM-cleared-1800x3600/original_data"
DATA_3D="/lcrc/project/ECP-EZ/public/compression/cesmbig-26x1800x3600"
DATA_3D_ALT="/lcrc/project/ECP-EZ/tripti/SDRBENCH-CESM-ATM-26x1800x3600/original_data"

BLOCK=64
OMP_THREADS=32

CSV="$OUTDIR/gpu_vs_cpu.csv"
LOG="$OUTDIR/gpu_vs_cpu.log"

echo "GPU vs CPU Timing Experiment — $TIMESTAMP" | tee "$LOG"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a "$LOG"
fi
echo "OMP_NUM_THREADS=$OMP_THREADS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "Field,Dims,Bound,BoundType,CUDA_Total_Comp_s,CUDA_Total_Decomp_s,CUDA_Kernel_Comp_s,CUDA_Kernel_Decomp_s,CUDA_PostProc_s,OMP_Comp_s,OMP_Decomp_s,FN_After,FP,FT,MaxErr_eps" > "$CSV"

export OMP_NUM_THREADS=$OMP_THREADS

###############################################################################
# Parse output from test programs
###############################################################################
parse_results() {
    local OUT="$1"

    # CUDA total timing from "TIMING comp_s=X.XXX decomp_s=X.XXX"
    cuda_comp=$(echo "$OUT" | grep 'TIMING' | sed -n 's/.*TIMING comp_s=\([0-9.]*\).*/\1/p' | head -1)
    cuda_decomp=$(echo "$OUT" | grep 'TIMING' | sed -n 's/.*decomp_s=\([0-9.]*\).*/\1/p' | head -1)

    # CUDA kernel-only times (before OpenMP section)
    local cuda_section
    cuda_section=$(echo "$OUT" | sed -n '1,/OpenMP PIPELINE/p')
    # "Compressed: XXXXX bytes (ratio: X.XXx) in XX.XX ms"
    local kernel_comp_ms kernel_decomp_ms postproc_ms
    kernel_comp_ms=$(echo "$cuda_section" | grep 'Compressed:' | sed -n 's/.*in \([0-9.]*\) ms/\1/p' | head -1)
    kernel_decomp_ms=$(echo "$cuda_section" | grep 'Decompressed in' | sed -n 's/.*in \([0-9.]*\) ms/\1/p' | head -1)
    postproc_ms=$(echo "$cuda_section" | grep 'Post-processing done' | sed -n 's/.*in \([0-9.]*\) ms/\1/p' | head -1)
    # Convert ms to seconds
    if [ -n "$kernel_comp_ms" ]; then
        cuda_kernel_comp=$(echo "scale=6; $kernel_comp_ms / 1000" | bc)
    else
        cuda_kernel_comp=""
    fi
    if [ -n "$kernel_decomp_ms" ]; then
        cuda_kernel_decomp=$(echo "scale=6; $kernel_decomp_ms / 1000" | bc)
    else
        cuda_kernel_decomp=""
    fi
    if [ -n "$postproc_ms" ]; then
        cuda_postproc=$(echo "scale=6; $postproc_ms / 1000" | bc)
    else
        cuda_postproc=""
    fi

    # OpenMP timing from section after "OpenMP PIPELINE COMPARISON"
    # Compress: "Compressed: XXXXX bytes (ratio: X.XXx) in XX.XX ms"
    # Decompress: "Decompressed in XX.XX ms"
    local omp_section
    omp_section=$(echo "$OUT" | sed -n '/OpenMP PIPELINE/,/SIDE-BY-SIDE/p')
    local omp_comp_ms omp_decomp_ms
    omp_comp_ms=$(echo "$omp_section" | grep 'Compressed:' | sed -n 's/.*in \([0-9.]*\) ms/\1/p' | head -1)
    omp_decomp_ms=$(echo "$omp_section" | grep 'Decompressed in' | sed -n 's/.*in \([0-9.]*\) ms/\1/p' | head -1)
    # Convert ms to seconds
    if [ -n "$omp_comp_ms" ]; then
        omp_comp=$(echo "scale=6; $omp_comp_ms / 1000" | bc)
    else
        omp_comp=""
    fi
    if [ -n "$omp_decomp_ms" ]; then
        omp_decomp=$(echo "scale=6; $omp_decomp_ms / 1000" | bc)
    else
        omp_decomp=""
    fi

    # Final preservation
    fn_after=$(echo "$OUT" | grep 'Lost (became regular)' | sed -n 's/.*: \([0-9]*\).*/\1/p' | head -1)
    if [ -z "$fn_after" ]; then
        fn_after=$(echo "$OUT" | grep 'Lost:' | sed -n 's/.*Lost: \([0-9]*\).*/\1/p' | head -1)
    fi
    fp=$(echo "$OUT" | grep 'New CPs' | sed -n 's/.*: \([0-9]*\).*/\1/p' | head -1)
    if [ -z "$fp" ]; then
        fp=$(echo "$OUT" | grep 'New:' | sed -n 's/.*New: \([0-9]*\).*/\1/p' | head -1)
    fi
    ft=$(echo "$OUT" | grep 'Type changed' | sed -n 's/.*: \([0-9]*\).*/\1/p' | head -1)
    if [ -z "$ft" ]; then
        ft=$(echo "$OUT" | grep 'Changed:' | sed -n 's/.*Changed: \([0-9]*\).*/\1/p' | head -1)
    fi

    # Max error
    max_err_eps=$(echo "$OUT" | grep 'Max pointwise error\|Max error' | sed -n 's/.*: \([0-9.e+-]*\).*/\1/p' | head -1)
}

###############################################################################
# 2D fields
###############################################################################
if [ "$MODE" = "quick" ]; then
    FIELDS_2D="CLDHGH"
else
    FIELDS_2D="CLDHGH CLDLOW AEROD_v FREQSH CLDMED"
fi

echo "========== 2D Fields ==========" | tee -a "$LOG"
for field in $FIELDS_2D; do
    FILE="$DATA_2D/${field}_1_1800_3600.dat"
    [ ! -f "$FILE" ] && continue

    # Absolute error bounds
    for eb in 1e-3; do
        echo -n "  $field abs $eb ... " | tee -a "$LOG"
        OUT=$("$TOPO_2D" "$FILE" 1800 3600 $eb $BLOCK 2>&1) || true
        parse_results "$OUT"
        echo "$field,1800x3600,$eb,ABS,$cuda_comp,$cuda_decomp,$cuda_kernel_comp,$cuda_kernel_decomp,$cuda_postproc,$omp_comp,$omp_decomp,$fn_after,${fp:-0},${ft:-0},$max_err_eps" >> "$CSV"
        echo "total=${cuda_comp}s kernel_c=${cuda_kernel_comp}s kernel_d=${cuda_kernel_decomp}s post=${cuda_postproc}s OMP=${omp_comp}s" | tee -a "$LOG"
        echo "$OUT" > "$OUTDIR/${field}_2d_abs_${eb}.log"
    done

    # Relative error bounds
    for rel_eb in rel:1e-3 rel:1e-4; do
        echo -n "  $field $rel_eb ... " | tee -a "$LOG"
        OUT=$("$TOPO_2D" "$FILE" 1800 3600 "$rel_eb" $BLOCK 2>&1) || true
        parse_results "$OUT"
        echo "$field,1800x3600,$rel_eb,REL,$cuda_comp,$cuda_decomp,$cuda_kernel_comp,$cuda_kernel_decomp,$cuda_postproc,$omp_comp,$omp_decomp,$fn_after,${fp:-0},${ft:-0},$max_err_eps" >> "$CSV"
        echo "total=${cuda_comp}s kernel_c=${cuda_kernel_comp}s kernel_d=${cuda_kernel_decomp}s post=${cuda_postproc}s OMP=${omp_comp}s" | tee -a "$LOG"
        echo "$OUT" > "$OUTDIR/${field}_2d_${rel_eb//:/_}.log"
    done
done

###############################################################################
# 3D fields
###############################################################################
if [ "$MODE" = "quick" ]; then
    FIELDS_3D="CLDICE"
else
    FIELDS_3D="CLDICE CLOUD T OMEGA RELHUM"
fi

echo "" | tee -a "$LOG"
echo "========== 3D Fields ==========" | tee -a "$LOG"
for field in $FIELDS_3D; do
    FILE=""
    for dir in "$DATA_3D" "$DATA_3D_ALT"; do
        for ext in dat f32; do
            [ -f "$dir/${field}_1_26_1800_3600.$ext" ] && FILE="$dir/${field}_1_26_1800_3600.$ext" && break 2
        done
    done
    [ -z "$FILE" ] && continue

    # Absolute error bound
    for eb in 1e-7; do
        echo -n "  $field abs $eb ... " | tee -a "$LOG"
        OUT=$("$TOPO_3D" "$FILE" 26 1800 3600 $eb $BLOCK 2>&1) || true
        parse_results "$OUT"
        echo "$field,26x1800x3600,$eb,ABS,$cuda_comp,$cuda_decomp,$cuda_kernel_comp,$cuda_kernel_decomp,$cuda_postproc,$omp_comp,$omp_decomp,$fn_after,${fp:-0},${ft:-0},$max_err_eps" >> "$CSV"
        echo "CUDA=${cuda_comp}s kernel_c=${cuda_kernel_comp}s kernel_d=${cuda_kernel_decomp}s post=${cuda_postproc}s" | tee -a "$LOG"
        echo "$OUT" > "$OUTDIR/${field}_3d_abs_${eb}.log"
    done

    # Relative error bounds
    for rel_eb in rel:1e-3 rel:1e-4; do
        echo -n "  $field $rel_eb ... " | tee -a "$LOG"
        OUT=$("$TOPO_3D" "$FILE" 26 1800 3600 "$rel_eb" $BLOCK 2>&1) || true
        parse_results "$OUT"
        echo "$field,26x1800x3600,$rel_eb,REL,$cuda_comp,$cuda_decomp,$cuda_kernel_comp,$cuda_kernel_decomp,$cuda_postproc,$omp_comp,$omp_decomp,$fn_after,${fp:-0},${ft:-0},$max_err_eps" >> "$CSV"
        echo "total=${cuda_comp}s kernel_c=${cuda_kernel_comp}s kernel_d=${cuda_kernel_decomp}s post=${cuda_postproc}s OMP=${omp_comp}s" | tee -a "$LOG"
        echo "$OUT" > "$OUTDIR/${field}_3d_${rel_eb//:/_}.log"
    done
done

echo "" | tee -a "$LOG"
echo "Results: $CSV" | tee -a "$LOG"
column -t -s',' "$CSV" 2>/dev/null | tee -a "$LOG"
