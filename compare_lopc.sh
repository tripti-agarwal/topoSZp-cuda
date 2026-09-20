#!/bin/bash
# Compare TopoSZp CUDA vs LOPC on the same dataset
# Usage: ./compare_lopc.sh <data_file> <rows> <cols> <error_bound> <block_size>
#
# Example:
#   ./compare_lopc.sh data/CLDHGH_1_1800_3600.dat 1800 3600 1e-3 64

set -e

DATA=$1
ROWS=$2
COLS=$3
EB=$4
BS=$5
NBELE=$((ROWS * COLS))

if [ -z "$BS" ]; then
    echo "Usage: $0 <data_file> <rows> <cols> <error_bound> <block_size>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOPC_DIR="${SCRIPT_DIR}/../LOPC"
BUILD_DIR="${SCRIPT_DIR}/build"

# Check LOPC is available
if [ ! -d "$LOPC_DIR" ]; then
    echo "LOPC not found at $LOPC_DIR"
    echo "Clone it: git clone https://github.com/burtscher/LOPC.git $LOPC_DIR"
    exit 1
fi

# Build LOPC if needed (GPU version for A100 = sm_80)
if [ ! -f "$LOPC_DIR/topo_compress_gpu_f32" ]; then
    echo "=== Building LOPC ==="
    cd "$LOPC_DIR"
    sed -i 's/NV_SM := 70/NV_SM := 80/' makefile
    make gpu 2>&1
    cd "$SCRIPT_DIR"
    echo ""
fi

echo "============================================================"
echo "  TopoSZp CUDA vs LOPC Comparison"
echo "============================================================"
echo "Data: $DATA"
echo "Grid: ${ROWS} × ${COLS} = ${NBELE} elements"
echo "Error bound: $EB    Block size: $BS"
echo ""

# ---- Run TopoSZp CUDA ----
echo "============ TopoSZp CUDA ============"
${BUILD_DIR}/test_cuda_topology "$DATA" $ROWS $COLS $EB $BS 2>&1 | grep -E "Compressed:|Preserved:|Error bound:|Compression ratio:|Persistence|> |All saddles|Local order|All adjacent|Both CPs|extremum|Maxima:|Minima:|Saddles:|Status:"
echo ""

# ---- Run LOPC GPU ----
echo "============ LOPC GPU ============"
LOPC_COMPRESSED="/tmp/lopc_compressed.bin"
LOPC_DECOMPRESSED="/tmp/lopc_decompressed.f32"

# Compress
echo "--- LOPC Compression ---"
${LOPC_DIR}/topo_compress_gpu_f32 "$DATA" "$LOPC_COMPRESSED" $EB $ROWS $COLS
echo ""

# Decompress
echo "--- LOPC Decompression ---"
${LOPC_DIR}/topo_decompress_gpu_f32 "$LOPC_COMPRESSED" "$LOPC_DECOMPRESSED" $EB $ROWS $COLS
echo ""

# Compare: find CPs in decompressed data and check preservation
echo "--- LOPC Analysis ---"
LOPC_SIZE=$(stat -c%s "$LOPC_COMPRESSED" 2>/dev/null || stat -f%z "$LOPC_COMPRESSED")
ORIG_SIZE=$((NBELE * 4))
RATIO=$(echo "scale=2; $ORIG_SIZE / $LOPC_SIZE" | bc)
echo "Compressed size: $LOPC_SIZE bytes (ratio: ${RATIO}x)"

# Use TopoSZp test to analyze LOPC decompressed data
# Compare critical points between original and LOPC decompressed
echo ""
echo "--- Analyzing LOPC decompressed data with TopoSZp CP finder ---"
# We need a small program to compare — use the existing compare_critical_points if available
if [ -f "${BUILD_DIR}/compare_critical_points" ]; then
    ${BUILD_DIR}/compare_critical_points "$DATA" "$LOPC_DECOMPRESSED" $ROWS $COLS $EB
fi

# Compute max error
python3 -c "
import struct, sys
with open('$DATA', 'rb') as f:
    orig = struct.unpack('${NBELE}f', f.read(${NBELE}*4))
with open('$LOPC_DECOMPRESSED', 'rb') as f:
    decomp = struct.unpack('${NBELE}f', f.read(${NBELE}*4))
max_err = max(abs(a-b) for a,b in zip(orig, decomp))
violations_1eb = sum(1 for a,b in zip(orig, decomp) if abs(a-b) > float('$EB')*1.01)
violations_2eb = sum(1 for a,b in zip(orig, decomp) if abs(a-b) > float('$EB')*2.0)

# Count CPs in original and decompressed
def find_cps(data, rows, cols):
    maxima = minima = saddles = 0
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            c = data[i*cols+j]
            u,d,l,r = data[(i-1)*cols+j], data[(i+1)*cols+j], data[i*cols+j-1], data[i*cols+j+1]
            if c>u and c>d and c>l and c>r: maxima += 1
            elif c<u and c<d and c<l and c<r: minima += 1
            else:
                xh = c>u and c>d; xl = c<u and c<d
                yh = c>l and c>r; yl = c<l and c<r
                if (xh and yl) or (xl and yh): saddles += 1
    return maxima, minima, saddles

om, omi, os = find_cps(orig, $ROWS, $COLS)
dm, dmi, ds = find_cps(decomp, $ROWS, $COLS)

# Check preservation
orig_map = {}
for i in range(1, $ROWS-1):
    for j in range(1, $COLS-1):
        c = orig[i*$COLS+j]
        u,d,l,r = orig[(i-1)*$COLS+j], orig[(i+1)*$COLS+j], orig[i*$COLS+j-1], orig[i*$COLS+j+1]
        t = 0
        if c>u and c>d and c>l and c>r: t = 1
        elif c<u and c<d and c<l and c<r: t = 2
        else:
            xh = c>u and c>d; xl = c<u and c<d
            yh = c>l and c>r; yl = c<l and c<r
            if (xh and yl) or (xl and yh): t = 3
        if t: orig_map[i*$COLS+j] = t

preserved = lost_max = lost_min = lost_sad = 0
for pos, t in orig_map.items():
    i, j = pos // $COLS, pos % $COLS
    c = decomp[i*$COLS+j]
    u,d,l,r = decomp[(i-1)*$COLS+j], decomp[(i+1)*$COLS+j], decomp[i*$COLS+j-1], decomp[i*$COLS+j+1]
    dt = 0
    if c>u and c>d and c>l and c>r: dt = 1
    elif c<u and c<d and c<l and c<r: dt = 2
    else:
        xh = c>u and c>d; xl = c<u and c<d
        yh = c>l and c>r; yl = c<l and c<r
        if (xh and yl) or (xl and yh): dt = 3
    if dt == t: preserved += 1
    elif t==1: lost_max += 1
    elif t==2: lost_min += 1
    else: lost_sad += 1

total_orig = len(orig_map)
print(f'Max error: {max_err:.6e}')
print(f'Within 1×eb: {\"YES\" if violations_1eb==0 else \"NO\"} ({violations_1eb} over)')
print(f'Within 2×eb: {\"YES\" if violations_2eb==0 else \"NO\"} ({violations_2eb} over)')
print(f'Original CPs: {total_orig} (max:{om} min:{omi} sad:{os})')
print(f'Decomp CPs:   {dm+dmi+ds} (max:{dm} min:{dmi} sad:{ds})')
print(f'Preserved: {preserved}/{total_orig} ({100.0*preserved/total_orig:.2f}%)')
print(f'  Lost max: {lost_max}  min: {lost_min}  sad: {lost_sad}')

# Local order check
all_p, all_v = 0, 0
for i in range(1, $ROWS-1):
    for j in range(1, $COLS-1):
        for di,dj in [(0,1),(1,0)]:
            ni, nj = i+di, j+dj
            if ni >= $ROWS-1 or nj >= $COLS-1: continue
            oa, ob = orig[i*$COLS+j], orig[ni*$COLS+nj]
            if oa == ob: continue
            da, db = decomp[i*$COLS+j], decomp[ni*$COLS+nj]
            oo = -1 if oa < ob else 1
            dd = -1 if da < db else (1 if da > db else 0)
            all_p += 1
            if oo != dd: all_v += 1
print(f'Local order: {all_p-all_v}/{all_p} ({100.0*(all_p-all_v)/all_p:.4f}%) preserved')
" 2>&1

# Cleanup
rm -f "$LOPC_COMPRESSED" "$LOPC_DECOMPRESSED"

echo ""
echo "============ COMPARISON COMPLETE ============"
