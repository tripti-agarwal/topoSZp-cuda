#!/bin/bash

# Script to compare critical points between original .dat files and decompressed files
# Supports .dat.SZ3.out, .dat.zfp.out, .dat.tthresh.out, and .dat.fpzip.out decompressed files
# Usage: ./compare_all_data.sh <data_directory> <rows> <cols> <error_bound> [decompressed_suffix]

set -e  # Exit on error

# Check arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <data_directory> <rows> <cols> <error_bound> [decompressed_suffix]"
    echo ""
    echo "Arguments:"
    echo "  data_directory    : Directory containing .dat and decompressed files"
    echo "  rows              : Number of rows in the data"
    echo "  cols              : Number of columns in the data"
    echo "  error_bound       : Error bound for classification (e.g., 1e-3)"
    echo "  decompressed_suffix: Optional suffix for decompressed files (default: .SZ3.out)"
    echo "                       Supported: .SZ3.out, .zfp.out, .tthresh.out, .fpzip.out"
    echo ""
    echo "Examples:"
    echo "  $0 SDRBENCH-CESM-ATM-cleared-1800x3600 1800 3600 1e-3"
    echo "  $0 ./ZFP_1e-3 384 320 1e-3 .zfp.out"
    echo "  $0 ./TTHRESH_1e-3 1800 3600 1e-3 .tthresh.out"
    echo "  $0 ./fpzip_1E-3 384 320 1e-3 .fpzip.out"
    exit 1
fi

DATA_DIR="$1"
ROWS="$2"
COLS="$3"
ERROR_BOUND="$4"
DECOMPRESSED_SUFFIX="${5}"  # Will be auto-detected if not specified

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

# Auto-detect decompressed suffix if not provided
if [ -z "$DECOMPRESSED_SUFFIX" ]; then
    # Try to find the first .dat file and check what decompressed files exist
    first_dat=$(find "$DATA_DIR" -maxdepth 1 -type f -name "*.dat" ! -name "*.SZ3.out" ! -name "*.zfp*" ! -name "*.tthresh*" ! -name "*.fpzip*" | head -1)
    if [ -n "$first_dat" ]; then
        base_name="${first_dat%.dat}"
        # Check for common suffixes
        if [ -f "${base_name}.dat.SZ3.out" ]; then
            DECOMPRESSED_SUFFIX=".SZ3.out"
            echo "Auto-detected decompressed suffix: .SZ3.out"
        elif [ -f "${base_name}.dat.zfp.out" ]; then
            DECOMPRESSED_SUFFIX=".zfp.out"
            echo "Auto-detected decompressed suffix: .zfp.out"
        elif [ -f "${base_name}.dat.tthresh.out" ]; then
            DECOMPRESSED_SUFFIX=".tthresh.out"
            echo "Auto-detected decompressed suffix: .tthresh.out"
        elif [ -f "${base_name}.dat.fpzip.out" ]; then
            DECOMPRESSED_SUFFIX=".fpzip.out"
            echo "Auto-detected decompressed suffix: .fpzip.out"
        elif [ -f "${base_name}.dat.SZp.out" ]; then
            DECOMPRESSED_SUFFIX=".SZp.out"
            echo "Auto-detected decompressed suffix: .SZp.out"
        else
            # Default to .SZ3.out if nothing found
            DECOMPRESSED_SUFFIX=".SZ3.out"
            echo "Warning: Could not auto-detect decompressed suffix, defaulting to .SZ3.out"
            echo "  If this is incorrect, please specify the suffix as the 5th argument"
        fi
    else
        # No .dat files found, use default
        DECOMPRESSED_SUFFIX=".SZ3.out"
    fi
    echo ""
fi

# Find all .dat files (excluding decompressed files)
# Exclude files that match the decompressed suffix pattern
if [[ "$DECOMPRESSED_SUFFIX" == ".zfp.out" ]]; then
    mapfile -t dat_files < <(find "$DATA_DIR" -maxdepth 1 -type f -name "*.dat" ! -name "*.zfp*" | sort)
elif [[ "$DECOMPRESSED_SUFFIX" == ".SZ3.out" ]]; then
    mapfile -t dat_files < <(find "$DATA_DIR" -maxdepth 1 -type f -name "*.dat" ! -name "*.SZ3.out" | sort)
elif [[ "$DECOMPRESSED_SUFFIX" == ".tthresh.out" ]]; then
    mapfile -t dat_files < <(find "$DATA_DIR" -maxdepth 1 -type f -name "*.dat" ! -name "*.tthresh*" | sort)
elif [[ "$DECOMPRESSED_SUFFIX" == ".fpzip.out" ]]; then
    mapfile -t dat_files < <(find "$DATA_DIR" -maxdepth 1 -type f -name "*.dat" ! -name "*.fpzip*" | sort)
else
    # Generic exclusion: remove files ending with the suffix
    suffix_pattern="*${DECOMPRESSED_SUFFIX}"
    mapfile -t dat_files < <(find "$DATA_DIR" -maxdepth 1 -type f -name "*.dat" ! -name "$suffix_pattern" | sort)
fi

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
    decompressed_file="${dat_file}${DECOMPRESSED_SUFFIX}"
    
    # Check if decompressed file exists
    if [ ! -f "$decompressed_file" ]; then
        echo "[$((i+1))/$total_files] Skipping: $filename (decompressed file not found: $(basename "$decompressed_file"))"
        failed_count=$((failed_count + 1))
        continue
    fi
    
    echo "[$((i+1))/$total_files] Processing: $filename"
    
    # Run comparison
    output=$(./compare_critical_points "$dat_file" "$decompressed_file" "$ROWS" "$COLS" "$ERROR_BOUND" 2>&1)
    exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "  ✗ Comparison failed (exit code: $exit_code)"
        echo "  Error output: $output"
        failed_count=$((failed_count + 1))
        echo ""
        continue
    fi
    
    # Parse output: should be "false_negatives false_positives false_types grand_total total_decompressed_critical_points"
    # Extract just the last line which contains the numbers
    result_line=$(echo "$output" | tail -1 | tr -d '\r')
    
    # Try to parse 5 values first (new format), then fall back to 4 values (old format)
    if read -r false_neg false_pos false_types grand_total total_decomp <<< "$result_line"; then
        # Check if we got 5 values and they're all numbers
        if [[ "$false_neg" =~ ^[0-9]+$ ]] && [[ "$false_pos" =~ ^[0-9]+$ ]] && [[ "$false_types" =~ ^[0-9]+$ ]] && [[ "$grand_total" =~ ^[0-9]+$ ]] && [[ "$total_decomp" =~ ^[0-9]+$ ]]; then
            echo "  False Negatives: $false_neg"
            echo "  False Positives: $false_pos"
            echo "  False Types: $false_types"
            echo "  Grand Total: $grand_total"
            echo "  Total Critical Points (Decompressed): $total_decomp"
            total_false_cases=$((total_false_cases + grand_total))
            processed_count=$((processed_count + 1))
        # Try parsing with 4 values (old format, in case executable wasn't recompiled)
        elif read -r false_neg false_pos false_types grand_total <<< "$result_line"; then
            if [[ "$false_neg" =~ ^[0-9]+$ ]] && [[ "$false_pos" =~ ^[0-9]+$ ]] && [[ "$false_types" =~ ^[0-9]+$ ]] && [[ "$grand_total" =~ ^[0-9]+$ ]]; then
                echo "  False Negatives: $false_neg"
                echo "  False Positives: $false_pos"
                echo "  False Types: $false_types"
                echo "  Grand Total: $grand_total"
                echo "  ⚠️  Note: Total Critical Points not available (recompile compare_critical_points for full output)"
                total_false_cases=$((total_false_cases + grand_total))
                processed_count=$((processed_count + 1))
            else
                echo "  ✗ Comparison failed (invalid output format)"
                echo "  Expected format: 4 or 5 numbers separated by spaces"
                echo "  Raw output: $output"
                echo "  Last line: '$result_line'"
                echo "  Parsed values: false_neg='$false_neg' false_pos='$false_pos' false_types='$false_types' grand_total='$grand_total' total_decomp='$total_decomp'"
                failed_count=$((failed_count + 1))
            fi
        else
            echo "  ✗ Comparison failed (could not parse output)"
            echo "  Raw output: $output"
            echo "  Last line: '$result_line'"
            failed_count=$((failed_count + 1))
        fi
    else
        echo "  ✗ Comparison failed (could not parse output)"
        echo "  Raw output: $output"
        echo "  Last line: '$result_line'"
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

# If all files were skipped, provide helpful message
if [ $processed_count -eq 0 ] && [ $failed_count -gt 0 ]; then
    echo ""
    echo "⚠️  Warning: No files were processed!"
    echo "   The script was looking for decompressed files with suffix: ${DECOMPRESSED_SUFFIX}"
    echo ""
    echo "   Checking what files exist in the directory..."
    echo ""
    echo "   .dat files found:"
    ls -1 "$DATA_DIR"/*.dat 2>/dev/null | head -5 | sed 's/^/      /' || echo "      (none found)"
    echo ""
    echo "   Files with suffix ${DECOMPRESSED_SUFFIX}:"
    ls -1 "$DATA_DIR"/*${DECOMPRESSED_SUFFIX} 2>/dev/null | head -5 | sed 's/^/      /' || echo "      (none found)"
    echo ""
    echo "   All .out files found:"
    ls -1 "$DATA_DIR"/*.out 2>/dev/null | head -10 | sed 's/^/      /' || echo "      (none found)"
    echo ""
    echo "   To fix this:"
    echo "   1. Make sure the decompressed files have the extension: ${DECOMPRESSED_SUFFIX}"
    echo "   2. Or specify the correct suffix as the 5th argument:"
    echo "      $0 $DATA_DIR $ROWS $COLS $ERROR_BOUND .zfp.out"
    echo "      $0 $DATA_DIR $ROWS $COLS $ERROR_BOUND .tthresh.out"
    echo "      $0 $DATA_DIR $ROWS $COLS $ERROR_BOUND .SZ3.out"
    echo "      $0 $DATA_DIR $ROWS $COLS $ERROR_BOUND .fpzip.out"
    echo "   3. Or verify the compression script created the files correctly"
fi


