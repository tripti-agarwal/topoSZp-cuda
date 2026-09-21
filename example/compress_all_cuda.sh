#!/bin/bash

# Script to compress all .dat/.f32 files using TopoSZp CUDA with topology preservation
# and run critical point comparison.
#
# Supports 2D and 3D datasets.
#
# Usage:
#   2D: ./compress_all_cuda.sh <input_dir> <output_dir> <rows> <cols> <error_bound> <block_size>
#   3D: ./compress_all_cuda.sh <input_dir> <output_dir> <d1> <d2> <d3> <error_bound> <block_size>
#
# Error bound: use absolute (e.g., 1e-3) or relative (e.g., rel:1e-4)
#
# Examples:
#   ./compress_all_cuda.sh ./data ./output_1e-3 1800 3600 1e-3 64
#   ./compress_all_cuda.sh ./data ./output_1e-3 26 1800 3600 1e-3 64

set -e

# Detect 2D vs 3D from argument count
if [ $# -eq 6 ]; then
    MODE="2D"
    INPUT_DIR="$1"; OUTPUT_DIR="$2"
    ROWS="$3"; COLS="$4"
    ERROR_BOUND="$5"; BLOCK_SIZE="$6"
    NBELE=$((ROWS * COLS))
    DIM_STR="${ROWS}×${COLS}"
elif [ $# -eq 7 ]; then
    MODE="3D"
    INPUT_DIR="$1"; OUTPUT_DIR="$2"
    D1="$3"; D2="$4"; D3="$5"
    ERROR_BOUND="$6"; BLOCK_SIZE="$7"
    NBELE=$((D1 * D2 * D3))
    DIM_STR="${D1}×${D2}×${D3}"
else
    echo "Usage (2D): $0 <input_dir> <output_dir> <rows> <cols> <error_bound> <block_size>"
    echo "Usage (3D): $0 <input_dir> <output_dir> <d1> <d2> <d3> <error_bound> <block_size>"
    echo ""
    echo "Error bound: 1e-3 for absolute, rel:1e-4 for relative"
    exit 1
fi

# Validate
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"; exit 1
fi
mkdir -p "$OUTPUT_DIR"

# Find build directory (look for test_cuda_topology)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR=""
for d in "$SCRIPT_DIR/../build" "$SCRIPT_DIR/../../build" "$(pwd)"; do
    if [ -f "$d/test_cuda_topology" ] || [ -f "$d/test_cuda_topology_3d" ]; then
        BUILD_DIR="$d"; break
    fi
done
if [ -z "$BUILD_DIR" ]; then
    echo "Error: Cannot find build directory with test executables"
    echo "Run from the build directory or ensure ../build/ contains the executables"
    exit 1
fi

# Find data files
mapfile -t data_files < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -name "*.dat" -o -name "*.f32" \) ! -name "*.SZp*" ! -name "*.out" | sort)

if [ ${#data_files[@]} -eq 0 ]; then
    echo "Error: No .dat or .f32 files found in '$INPUT_DIR'"; exit 1
fi

total_files=${#data_files[@]}

echo "============================================================"
echo "  TopoSZp CUDA Batch Processing ($MODE)"
echo "============================================================"
echo "Input:       $INPUT_DIR"
echo "Output:      $OUTPUT_DIR"
echo "Dimensions:  $DIM_STR ($NBELE elements)"
echo "Error bound: $ERROR_BOUND"
echo "Block size:  $BLOCK_SIZE"
echo "Files:       $total_files"
echo ""

# CSV output
CSV_FILE="${OUTPUT_DIR}/results_cuda.csv"
echo "filename,compression_ratio,max_error,preserved_pct,maxima_preserved,minima_preserved,saddles_preserved,saddles_total,local_order_pct,compression_ms,decompression_ms" > "$CSV_FILE"

processed=0; failed=0

for i in "${!data_files[@]}"; do
    data_file="${data_files[$i]}"
    filename=$(basename "$data_file")
    echo "[$((i+1))/$total_files] Processing: $filename"

    # Run the topology test and capture output
    if [ "$MODE" = "2D" ]; then
        output=$("$BUILD_DIR/test_cuda_topology" "$data_file" "$ROWS" "$COLS" "$ERROR_BOUND" "$BLOCK_SIZE" 2>&1) || true
    else
        output=$("$BUILD_DIR/test_cuda_topology_3d" "$data_file" "$D1" "$D2" "$D3" "$ERROR_BOUND" "$BLOCK_SIZE" 2>&1) || true
    fi

    # Parse results
    ratio=$(echo "$output" | grep -oP 'ratio: \K[0-9.]+' | head -1)
    max_err=$(echo "$output" | grep -oP 'Max pointwise error: \K[0-9.e+-]+' | head -1)
    if [ -z "$max_err" ]; then
        max_err=$(echo "$output" | grep -oP 'Max error: \K[0-9.e+-]+' | head -1)
    fi
    preserved=$(echo "$output" | grep -oP 'Preserved:.*\(\K[0-9.]+' | head -1)
    comp_time=$(echo "$output" | grep -oP 'Compressed.*in \K[0-9.]+' | head -1)
    decomp_time=$(echo "$output" | grep -oP 'Decompressed in \K[0-9.]+' | head -1)

    # Parse CP counts
    max_kept=$(echo "$output" | grep "Maxima:" | tail -1 | grep -oP '\K[0-9]+(?= /)')
    min_kept=$(echo "$output" | grep "Minima:" | tail -1 | grep -oP '\K[0-9]+(?= /)')
    sad_kept=$(echo "$output" | grep "Saddles:" | tail -1 | grep -oP '\K[0-9]+(?= /)')
    sad_total=$(echo "$output" | grep "Saddles:" | tail -1 | grep -oP '/ \K[0-9]+')
    local_order=$(echo "$output" | grep "All adjacent" | grep -oP '[0-9.]+(?=%)')

    if [ -n "$ratio" ]; then
        echo "  ✓ Ratio: ${ratio}x  Preserved: ${preserved}%  Max err: ${max_err}"
        echo "$filename,$ratio,$max_err,$preserved,$max_kept,$min_kept,$sad_kept,$sad_total,$local_order,$comp_time,$decomp_time" >> "$CSV_FILE"
        processed=$((processed + 1))
    else
        echo "  ✗ Failed"
        echo "$output" | tail -5
        failed=$((failed + 1))
    fi
    echo ""
done

echo "=== Summary ==="
echo "Processed: $processed / $total_files"
echo "Failed:    $failed"
echo "Results:   $CSV_FILE"
