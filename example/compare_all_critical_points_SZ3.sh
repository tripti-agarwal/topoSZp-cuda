#!/bin/bash

# Script to compare critical points between original .dat files and decompressed .dat.SZ3.out files
# Usage: ./compare_all_data.sh <data_directory> <rows> <cols> <error_bound>

set -e  # Exit on error

# Check arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <data_directory> <rows> <cols> <error_bound>"
    echo ""
    echo "Arguments:"
    echo "  data_directory  : Directory containing .dat and .dat.SZ3.out files"
    echo "  rows            : Number of rows in the data"
    echo "  cols            : Number of columns in the data"
    echo "  error_bound     : Error bound for classification (e.g., 1e-3)"
    echo ""
    echo "Example:"
    echo "  $0 SDRBENCH-CESM-ATM-cleared-1800x3600 1800 3600 1e-3"
    exit 1
fi

DATA_DIR="$1"
ROWS="$2"
COLS="$3"
ERROR_BOUND="$4"

# Validate arguments
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory '$DATA_DIR' does not exist"
    exit 1
fi

if ! [[ "$ROWS" =~ ^[0-9]+$ ]] || [ "$ROWS" -le 0 ]; then
    echo "Error: rows must be a positive integer"
    exit 1
fi

if ! [[ "$COLS" =~ ^[0-9]+$ ]] || [ "$COLS" -le 0 ]; then
    echo "Error: cols must be a positive integer"
    exit 1
fi

# Check if compare_critical_points executable exists
if [ ! -f "./compare_critical_points" ]; then
    echo "Error: compare_critical_points executable not found in current directory"
    echo "Please compile it first:"
    echo "  g++ -o compare_critical_points compare_critical_points.cc -lm"
    exit 1
fi

# Find all .dat files (excluding .dat.SZ3.out files)
mapfile -t dat_files < <(find "$DATA_DIR" -maxdepth 1 -type f -name "*.dat" ! -name "*.SZ3.out" | sort)

if [ ${#dat_files[@]} -eq 0 ]; then
    echo "Error: No .dat files found in '$DATA_DIR'"
    exit 1
fi

total_files=${#dat_files[@]}
echo "Found $total_files .dat file(s) to process"
echo ""

# Statistics
total_false_cases=0
processed_count=0
failed_count=0

# Process each .dat file
for i in "${!dat_files[@]}"; do
    dat_file="${dat_files[$i]}"
    filename=$(basename "$dat_file")
    
    # Expected decompressed file name
    decompressed_file="${dat_file}.SZ3.out"
    
    # Check if decompressed file exists
    if [ ! -f "$decompressed_file" ]; then
        echo "[$((i+1))/$total_files] Skipping: $filename (decompressed file not found: $(basename "$decompressed_file"))"
        failed_count=$((failed_count + 1))
        continue
    fi
    
    echo "[$((i+1))/$total_files] Processing: $filename"
    
    # Run comparison
    output=$(./compare_critical_points "$dat_file" "$decompressed_file" "$ROWS" "$COLS" "$ERROR_BOUND" 2>&1)
    
    # Parse output: should be "false_negatives false_positives false_types grand_total decomp_count"
    if read -r false_neg false_pos false_types grand_total decomp_count <<< "$output"; then
        # Check if all values are numbers
        if [[ "$false_neg" =~ ^[0-9]+$ ]] && [[ "$false_pos" =~ ^[0-9]+$ ]] && [[ "$false_types" =~ ^[0-9]+$ ]] && [[ "$grand_total" =~ ^[0-9]+$ ]]; then
            echo "  False Negatives: $false_neg"
            echo "  False Positives: $false_pos"
            echo "  False Types: $false_types"
            echo "  Grand Total: $grand_total"
            total_false_cases=$((total_false_cases + grand_total))
            processed_count=$((processed_count + 1))
        else
            echo "  ✗ Comparison failed (invalid output format)"
            echo "  Error output: $output"
            failed_count=$((failed_count + 1))
        fi
    else
        echo "  ✗ Comparison failed"
        echo "  Error output: $output"
        failed_count=$((failed_count + 1))
    fi
    echo ""
done

# Print summary
echo "=== Summary ==="
echo "Total files processed: $processed_count"
echo "Total files failed: $failed_count"
echo "Total false cases across all files: $total_false_cases"
if [ $processed_count -gt 0 ]; then
    average_false=$(echo "scale=2; $total_false_cases / $processed_count" | bc)
    echo "Average false cases per file: $average_false"
fi


