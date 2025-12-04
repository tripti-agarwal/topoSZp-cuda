#!/bin/bash

# Script to compress and decompress all .dat files in a directory
# Usage: ./process_all_data.sh [data_directory] [block_size] [error_bound]
# Example: ./process_all_data.sh SDRBENCH-CESM-ATM-cleared-1800x3600 64 1E-3

# Don't exit on error - continue processing all files
set +e

# Default values
DATA_DIR="${1:-SDRBENCH-CESM-ATM-cleared-1800x3600}"
BLOCK_SIZE="${2:-64}"
ERROR_BOUND="${3:-1E-3}"

# Check if executables exist
if [ ! -f "./testfloat_compress_fastmode1" ]; then
    echo "Error: testfloat_compress_fastmode1 not found in current directory"
    exit 1
fi

if [ ! -f "./testfloat_decompress_fastmode1" ]; then
    echo "Error: testfloat_decompress_fastmode1 not found in current directory"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist"
    exit 1
fi

echo "=========================================="
echo "Processing all .dat files in: $DATA_DIR"
echo "Block size: $BLOCK_SIZE"
echo "Error bound: $ERROR_BOUND"
echo "=========================================="
echo ""

# Function to extract dimensions from filename
# Supports formats like: NAME_ID_ROWS_COLS.dat or NAME_ID_ROWSxCOLS.dat
extract_dimensions() {
    local filename="$1"
    local basename=$(basename "$filename" .dat)
    
    # Try pattern: _ROWS_COLS or _ROWSxCOLS at the end
    if [[ $basename =~ _([0-9]+)[x_]([0-9]+)$ ]]; then
        ROWS="${BASH_REMATCH[1]}"
        COLS="${BASH_REMATCH[2]}"
        return 0
    fi
    
    # If no pattern matches, return failure
    return 1
}

# Process each .dat file
processed=0
failed=0
skipped=0

# Count total files first
total_files=$(find "$DATA_DIR" -maxdepth 1 -name "*.dat" -type f | wc -l)
echo "Found $total_files .dat files to process"
echo ""

for dat_file in "$DATA_DIR"/*.dat; do
    # Check if file exists (handles case where no .dat files found)
    [ -f "$dat_file" ] || continue
    
    filename=$(basename "$dat_file")
    current=$((processed + failed + skipped + 1))
    echo "[$current/$total_files] Processing: $filename"
    
    # Extract dimensions from filename
    if ! extract_dimensions "$dat_file"; then
        echo "  Warning: Could not extract dimensions from filename"
        echo "  Expected format: NAME_ID_ROWS_COLS.dat or NAME_ID_ROWSxCOLS.dat"
        echo "  Skipping: $filename"
        ((skipped++))
        echo ""
        continue
    fi
    
    # Calculate number of elements
    nbEle=$((ROWS * COLS))
    
    echo "  Dimensions: ${ROWS}x${COLS}"
    echo "  Elements: $nbEle"
    
    # Compress
    echo "  Compressing..."
    if ./testfloat_compress_fastmode1 "$dat_file" "$BLOCK_SIZE" "$ERROR_BOUND" "$ROWS" "$COLS" 2>&1; then
        echo "  ✓ Compression successful"
    else
        echo "  ✗ Compression failed (exit code: $?)"
        ((failed++))
        echo ""
        continue
    fi
    
    # Check if compressed file exists
    compressed_file="${dat_file}.SZp"
    if [ ! -f "$compressed_file" ]; then
        echo "  ✗ Compressed file not found: $compressed_file"
        ((failed++))
        echo ""
        continue
    fi
    
    # Decompress
    echo "  Decompressing..."
    if ./testfloat_decompress_fastmode1 "$compressed_file" "$nbEle" "$BLOCK_SIZE" "$ERROR_BOUND" "$ROWS" "$COLS" 2>&1; then
        echo "  ✓ Decompression successful"
    else
        echo "  ✗ Decompression failed (exit code: $?)"
        ((failed++))
        echo ""
        continue
    fi
    
    # Check if decompressed file exists
    decompressed_file="${compressed_file}.out"
    if [ ! -f "$decompressed_file" ]; then
        echo "  ✗ Decompressed file not found: $decompressed_file"
        ((failed++))
        echo ""
        continue
    fi
    
    echo "  ✓ Complete: $filename"
    ((processed++))
    echo ""
done

# Summary
echo "=========================================="
echo "Summary:"
echo "  Processed: $processed"
echo "  Failed: $failed"
echo "  Skipped: $skipped"
echo "=========================================="

if [ $failed -gt 0 ]; then
    exit 1
fi

