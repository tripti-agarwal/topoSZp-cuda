#!/bin/bash

# Script to compress and decompress all .dat files in a directory
# Usage: ./process_all_data.sh <data_directory> <block_size> <error_bound> <output_directory> <rows> <cols>
# Example: ./process_all_data.sh SDRBENCH-CESM-ATM-cleared-1800x3600 64 1E-3 ./output 1800 3600

# Don't exit on error - continue processing all files
set +e

# Check if all required parameters are provided
if [ $# -lt 6 ]; then
    echo "Error: Missing required parameters"
    echo "Usage: $0 <data_directory> <block_size> <error_bound> <output_directory> <rows> <cols>"
    echo "Example: $0 SDRBENCH-CESM-ATM-cleared-1800x3600 64 1E-3 ./output 1800 3600"
    exit 1
fi

# Required parameters (no defaults)
DATA_DIR="$1"
BLOCK_SIZE="$2"
ERROR_BOUND="$3"
OUTPUT_DIR="$4"
ROWS="$5"
COLS="$6"

# Validate dimensions are positive integers
if ! [[ "$ROWS" =~ ^[0-9]+$ ]] || [ "$ROWS" -le 0 ]; then
    echo "Error: ROWS must be a positive integer, got: $ROWS"
    exit 1
fi

if ! [[ "$COLS" =~ ^[0-9]+$ ]] || [ "$COLS" -le 0 ]; then
    echo "Error: COLS must be a positive integer, got: $COLS"
    exit 1
fi

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

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create output directory '$OUTPUT_DIR'"
        exit 1
    fi
fi

echo "=========================================="
echo "Processing all .dat files in: $DATA_DIR"
echo "Block size: $BLOCK_SIZE"
echo "Error bound: $ERROR_BOUND"
echo "Output directory: $OUTPUT_DIR"
echo "Dimensions: ${ROWS}x${COLS}"
echo "=========================================="
echo ""

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
    
    # Calculate number of elements using provided dimensions
    nbEle=$((ROWS * COLS))
    
    # Validate file size
    file_size=$(stat -f%z "$dat_file" 2>/dev/null || stat -c%s "$dat_file" 2>/dev/null)
    expected_size_float32=$((nbEle * 4))  # 4 bytes per float32
    expected_size_float64=$((nbEle * 8))   # 8 bytes per float64
    
    echo "  Dimensions: ${ROWS}x${COLS}"
    echo "  Elements: $nbEle"
    if [ -n "$file_size" ]; then
        echo "  File size: $file_size bytes"
        if [ "$file_size" -eq "$expected_size_float32" ]; then
            echo "  ✓ File size matches float32 format (${expected_size_float32} bytes)"
        elif [ "$file_size" -eq "$expected_size_float64" ]; then
            echo "  ✗ ERROR: File size matches float64 format, but compression expects float32!"
            echo "    Expected (float32): $expected_size_float32 bytes"
            echo "    Actual (float64): $file_size bytes"
            echo "    Skipping: $filename"
            ((skipped++))
            echo ""
            continue
        else
            echo "  ✗ ERROR: File size doesn't match expected dimensions!"
            echo "    Expected (float32): $expected_size_float32 bytes (${ROWS}x${COLS} * 4 bytes)"
            echo "    Actual: $file_size bytes"
            # Try to suggest correct dimensions
            possible_rows=$((file_size / 4 / COLS))
            possible_cols=$((file_size / 4 / ROWS))
            if [ $((possible_rows * COLS * 4)) -eq "$file_size" ] && [ "$possible_rows" -gt 0 ]; then
                echo "    Suggested ROWS: $possible_rows (keeping COLS=$COLS)"
            fi
            if [ $((ROWS * possible_cols * 4)) -eq "$file_size" ] && [ "$possible_cols" -gt 0 ]; then
                echo "    Suggested COLS: $possible_cols (keeping ROWS=$ROWS)"
            fi
            echo "    This file will cause a segmentation fault - skipping"
            ((skipped++))
            echo ""
            continue
        fi
    else
        echo "  ⚠ Warning: Could not determine file size - proceeding anyway"
    fi
    
    # Compress with detailed error capture
    echo "  Compressing..."
    compress_output=$(./testfloat_compress_fastmode1 "$dat_file" "$BLOCK_SIZE" "$ERROR_BOUND" "$ROWS" "$COLS" 2>&1)
    compress_exit_code=$?
    
    if [ $compress_exit_code -eq 0 ]; then
        echo "  ✓ Compression successful"
    else
        echo "  ✗ Compression failed (exit code: $compress_exit_code)"
        if [ $compress_exit_code -eq 139 ]; then
            echo "    ⚠ Segmentation fault detected!"
            echo "    Possible causes:"
            echo "      1. Dimensions mismatch (ROWS=$ROWS, COLS=$COLS)"
            echo "      2. File format mismatch (expected float32)"
            echo "      3. Memory allocation issue"
            echo "      4. Block size incompatible with dimensions"
        fi
        if [ -n "$compress_output" ]; then
            echo "    Error output:"
            echo "$compress_output" | sed 's/^/      /'
        fi
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
    
    # Copy output files to destination directory
    echo "  Copying output files to $OUTPUT_DIR..."
    output_base=$(basename "$dat_file")
    
    # Copy compressed file
    if cp "$compressed_file" "$OUTPUT_DIR/${output_base}.SZp" 2>/dev/null; then
        echo "  ✓ Copied compressed file"
    else
        echo "  ✗ Failed to copy compressed file"
    fi
    
    # Copy decompressed file
    if cp "$decompressed_file" "$OUTPUT_DIR/${output_base}.SZp.out" 2>/dev/null; then
        echo "  ✓ Copied decompressed file"
    else
        echo "  ✗ Failed to copy decompressed file"
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

