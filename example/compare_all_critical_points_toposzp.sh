#!/bin/bash

# Script to compare critical points between original .dat files and decompressed .SZp.out files
# Usage: ./compare_all_critical_points_toposzp.sh <original_directory> <decompressed_directory> <rows> <cols> <error_bound> [output_csv]
# Example: ./compare_all_critical_points_toposzp.sh ../../SDRBENCH-CESM-ATM-cleared-1800x3600 ../../SDRBENCH-CESM-ATM-cleared-1800x3600/topoSZp_1E-3 1800 3600 1E-3

set -e  # Exit on error

# Check arguments
if [ $# -lt 5 ]; then
    echo "Usage: $0 <original_directory> <decompressed_directory> <rows> <cols> <error_bound> [output_csv]"
    echo ""
    echo "Arguments:"
    echo "  original_directory    : Directory containing original .dat files"
    echo "  decompressed_directory: Directory containing .SZp.out files"
    echo "  rows                  : Number of rows in the data"
    echo "  cols                  : Number of columns in the data"
    echo "  error_bound           : Error bound for classification (e.g., 1e-3 or 1E-3)"
    echo "  output_csv            : (Optional) CSV file to save results (default: critical_points_toposzp.csv)"
    echo ""
    echo "Example:"
    echo "  $0 ../../SDRBENCH-CESM-ATM-cleared-1800x3600 ../../SDRBENCH-CESM-ATM-cleared-1800x3600/topoSZp_1E-3 1800 3600 1E-3"
    exit 1
fi

ORIG_DIR="$1"
DECOMP_DIR="$2"
ROWS="$3"
COLS="$4"
ERROR_BOUND="$5"
OUTPUT_CSV="${6:-critical_points_toposzp.csv}"

# Validate arguments
if [ ! -d "$ORIG_DIR" ]; then
    echo "Error: Original directory '$ORIG_DIR' does not exist"
    exit 1
fi

if [ ! -d "$DECOMP_DIR" ]; then
    echo "Error: Decompressed directory '$DECOMP_DIR' does not exist"
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

# Find all .dat files in original directory (excluding .SZp.out files)
mapfile -t dat_files < <(find "$ORIG_DIR" -maxdepth 1 -type f -name "*.dat" ! -name "*.SZp*" | sort)

if [ ${#dat_files[@]} -eq 0 ]; then
    echo "Error: No .dat files found in '$ORIG_DIR'"
    exit 1
fi

total_files=${#dat_files[@]}
echo "Original directory: $ORIG_DIR"
echo "Decompressed directory: $DECOMP_DIR"
echo "Dimensions: ${ROWS}x${COLS}"
echo "Error bound: $ERROR_BOUND"
echo ""
echo "Found $total_files .dat file(s) to process"
echo ""

# Create CSV header if file doesn't exist
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "filename,false_negatives,false_positives,false_types,grand_total_false,decompressed_critical_points" > "$OUTPUT_CSV"
fi

# Statistics
processed_count=0
failed_count=0
skipped_count=0

# Process each .dat file
for i in "${!dat_files[@]}"; do
    dat_file="${dat_files[$i]}"
    filename=$(basename "$dat_file")
    
    # Expected decompressed file path
    decompressed_file="${DECOMP_DIR}/${filename}.SZp.out"
    
    # Skip if decompressed file doesn't exist
    if [ ! -f "$decompressed_file" ]; then
        echo "[$((i+1))/$total_files] Skipping: $filename (decompressed file not found: $decompressed_file)"
        skipped_count=$((skipped_count + 1))
        continue
    fi
    
    echo "[$((i+1))/$total_files] Processing: $filename"
    
    # Run compare_critical_points
    # Output format: false_negatives false_positives false_types grand_total_false decompressed_critical_points
    result=$(./compare_critical_points "$dat_file" "$decompressed_file" "$ROWS" "$COLS" "$ERROR_BOUND" 2>&1)
    
    if [ $? -eq 0 ]; then
        # Parse output (format: FN FP FT GT DECOMP_COUNT)
        read -r fn fp ft gt decomp_count <<< "$result"
        
        # Append to CSV
        echo "$filename,$fn,$fp,$ft,$gt,$decomp_count" >> "$OUTPUT_CSV"
        
        echo "  ✓ False Negatives: $fn, False Positives: $fp, False Types: $ft, Total: $gt, Decompressed CPs: $decomp_count"
        processed_count=$((processed_count + 1))
    else
        echo "  ✗ Failed to compare critical points"
        echo "  Error: $result"
        failed_count=$((failed_count + 1))
    fi
    echo ""
done

# Print summary
echo "=== Summary ==="
echo "Total files processed: $processed_count"
echo "Total files skipped: $skipped_count"
echo "Total files failed: $failed_count"
echo "Results saved to: $OUTPUT_CSV"
echo ""

if [ $failed_count -gt 0 ]; then
    exit 1
fi

