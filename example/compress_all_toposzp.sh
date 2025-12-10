#!/bin/bash

# Script to compress all .dat files in a directory using TopoSZp
# Uses two separate commands: one for compression, one for decompression
# Usage: ./compress_all_toposzp.sh <input_directory> <output_directory> <rows> <cols> <error_bound> <block_size>
# Example: ./compress_all_toposzp.sh SDRBENCH-CESM-ATM-cleared-1800x3600 ./toposzp_1E-3 1800 3600 1E-3 64

set -e  # Exit on error

# Check arguments
if [ $# -lt 6 ]; then
    echo "Usage: $0 <input_directory> <output_directory> <rows> <cols> <error_bound> <block_size>"
    echo ""
    echo "Arguments:"
    echo "  input_directory  : Directory containing .dat files"
    echo "  output_directory : Directory where .SZp and .SZp.out files will be saved"
    echo "  rows              : Number of rows in the data"
    echo "  cols              : Number of columns in the data"
    echo "  error_bound       : Error bound for compression (e.g., 0.001 or 1E-3)"
    echo "  block_size        : Block size for compression (e.g., 64)"
    echo ""
    echo "Example:"
    echo "  $0 SDRBENCH-CESM-ATM-cleared-1800x3600 ./toposzp_1E-3 1800 3600 1E-3 64"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
ROWS="$3"
COLS="$4"
ERROR_BOUND="$5"
BLOCK_SIZE="$6"

# Validate arguments
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
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

if ! [[ "$ROWS" =~ ^[0-9]+$ ]] || [ "$ROWS" -le 0 ]; then
    echo "Error: rows must be a positive integer"
    exit 1
fi

if ! [[ "$COLS" =~ ^[0-9]+$ ]] || [ "$COLS" -le 0 ]; then
    echo "Error: cols must be a positive integer"
    exit 1
fi

if ! [[ "$BLOCK_SIZE" =~ ^[0-9]+$ ]] || [ "$BLOCK_SIZE" -le 0 ]; then
    echo "Error: block_size must be a positive integer"
    exit 1
fi

# Check if executables exist
if [ ! -f "./testfloat_compress_fastmode1" ]; then
    echo "Error: testfloat_compress_fastmode1 executable not found in current directory"
    echo "Please ensure testfloat_compress_fastmode1 is compiled and available in the current directory"
    exit 1
fi

if [ ! -f "./testfloat_decompress_fastmode1" ]; then
    echo "Error: testfloat_decompress_fastmode1 executable not found in current directory"
    echo "Please ensure testfloat_decompress_fastmode1 is compiled and available in the current directory"
    exit 1
fi

# Find all .dat files (excluding .SZp and .SZp.out files)
mapfile -t dat_files < <(find "$INPUT_DIR" -maxdepth 1 -type f -name "*.dat" ! -name "*.SZp*" | sort)

if [ ${#dat_files[@]} -eq 0 ]; then
    echo "Error: No .dat files found in '$INPUT_DIR'"
    exit 1
fi

# Calculate number of elements
NUM_ELEMENTS=$((ROWS * COLS))

echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Dimensions: ${ROWS}x${COLS} (${NUM_ELEMENTS} elements)"
echo "Error bound: $ERROR_BOUND"
echo "Block size: $BLOCK_SIZE"
echo ""

total_files=${#dat_files[@]}
echo "Found $total_files .dat file(s) to process"
echo ""

# Statistics
processed_count=0
failed_count=0
skipped_count=0

# Convert to absolute paths to avoid path resolution issues
INPUT_DIR=$(cd "$INPUT_DIR" && pwd)
OUTPUT_DIR=$(cd "$OUTPUT_DIR" 2>/dev/null && pwd || (mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd))

# Process each .dat file
for i in "${!dat_files[@]}"; do
    dat_file="${dat_files[$i]}"
    # Convert to absolute path
    dat_file=$(cd "$(dirname "$dat_file")" && pwd)/$(basename "$dat_file")
    filename=$(basename "$dat_file")
    
    # Output file paths in the output directory
    compressed_file="${OUTPUT_DIR}/${filename}.SZp"
    decompressed_file="${OUTPUT_DIR}/${filename}.SZp.out"
    
    # Skip if both output files already exist
    if [ -f "$compressed_file" ] && [ -f "$decompressed_file" ]; then
        echo "[$((i+1))/$total_files] Skipping: $filename (output files already exist)"
        skipped_count=$((skipped_count + 1))
        continue
    fi
    
    echo "[$((i+1))/$total_files] Processing: $filename"
    
    # Temporary files in input directory (will be moved to output directory)
    temp_compressed="${dat_file}.SZp"
    temp_decompressed="${temp_compressed}.out"
    
    # Step 1: Compression
    # testfloat_compress_fastmode1 <input> <block_size> <error_bound> <rows> <cols>
    # Output: <input>.SZp (created in same directory as input)
    if ./testfloat_compress_fastmode1 "$dat_file" "$BLOCK_SIZE" "$ERROR_BOUND" "$ROWS" "$COLS" 2>&1 | grep -v "Failed to open input file" | grep -v "cannot be read"; then
        # Check exit status separately since we're filtering output
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "  ✗ Compression failed"
            failed_count=$((failed_count + 1))
            echo ""
            continue
        fi
        
        if [ ! -f "$temp_compressed" ]; then
            echo "  ✗ Compression failed: compressed file not created"
            failed_count=$((failed_count + 1))
            echo ""
            continue
        fi
        
        # Move compressed file to output directory
        mv "$temp_compressed" "$compressed_file" 2>/dev/null || cp "$temp_compressed" "$compressed_file"
        
        # Create a symlink to the original file in the output directory so decompression can find it for comparison
        # The decompression code tries to read the original file by removing .SZp extension
        original_link="${OUTPUT_DIR}/${filename}"
        if [ ! -f "$original_link" ]; then
            ln -sf "$dat_file" "$original_link" 2>/dev/null || cp "$dat_file" "$original_link" 2>/dev/null || true
        fi
        
        # Step 2: Decompression
        # testfloat_decompress_fastmode1 <compressed> <num_elements> <block_size> <error_bound> <rows> <cols>
        # Output: <compressed>.out (created in same directory as compressed)
        # Use absolute path for compressed file
        compressed_file_abs=$(cd "$(dirname "$compressed_file")" && pwd)/$(basename "$compressed_file")
        # Filter out the "Failed to open input file" error which occurs when it can't find the original for comparison
        # This is non-fatal - decompression still succeeds
        if ./testfloat_decompress_fastmode1 "$compressed_file_abs" "$NUM_ELEMENTS" "$BLOCK_SIZE" "$ERROR_BOUND" "$ROWS" "$COLS" 2>&1 | grep -v "^Failed to open input file" | grep -v "^Error:.*cannot be read!"; then
            # Check exit status separately
            if [ ${PIPESTATUS[0]} -ne 0 ]; then
                echo "  ✗ Decompression failed"
                failed_count=$((failed_count + 1))
                echo ""
                continue
            fi
            # Check if decompressed file was created
            if [ -f "$decompressed_file" ]; then
                compressed_size=$(stat -f%z "$compressed_file" 2>/dev/null || stat -c%s "$compressed_file" 2>/dev/null)
                input_size=$(stat -f%z "$dat_file" 2>/dev/null || stat -c%s "$dat_file" 2>/dev/null)
                if [ -n "$compressed_size" ] && [ -n "$input_size" ] && [ "$input_size" -gt 0 ]; then
                    compression_ratio=$(echo "scale=2; $input_size / $compressed_size" | bc 2>/dev/null || echo "N/A")
                    echo "  ✓ Compression and decompression successful (ratio: ${compression_ratio}x)"
                else
                    echo "  ✓ Compression and decompression successful"
                fi
                processed_count=$((processed_count + 1))
            else
                echo "  ✗ Decompression failed: output file not created"
                failed_count=$((failed_count + 1))
            fi
        else
            echo "  ✗ Decompression failed (exit code: $?)"
            failed_count=$((failed_count + 1))
        fi
    else
        echo "  ✗ Compression failed (exit code: $?)"
        failed_count=$((failed_count + 1))
    fi
    echo ""
done

# Print summary
echo "=== Summary ==="
echo "Total files processed: $processed_count"
echo "Total files skipped: $skipped_count"
echo "Total files failed: $failed_count"
echo ""

if [ $failed_count -gt 0 ]; then
    exit 1
fi

