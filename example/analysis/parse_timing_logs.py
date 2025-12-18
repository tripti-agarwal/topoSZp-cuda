#!/usr/bin/env python3
"""
Parse timing logs from topoSZp folders and generate average compression/decompression time tables.
"""

import os
import re
import glob
from collections import defaultdict
import csv

def parse_log_file(log_path):
    """Parse a log file and extract compression and decompression times."""
    comp_times = []
    decomp_times = []
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            
        # Pattern for compression time: "compression time=0.005757, total fastmode1 topology"
        # Use negative lookbehind to ensure we don't match "decompression time="
        comp_pattern = r'(?<!de)compression time=([\d.]+)'
        decomp_pattern = r'decompression time=([\d.]+)'
        
        comp_matches = re.findall(comp_pattern, content)
        decomp_matches = re.findall(decomp_pattern, content)
        
        comp_times = [float(t) for t in comp_matches]
        decomp_times = [float(t) for t in decomp_matches]
        
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
    
    return comp_times, decomp_times

def find_all_topoSZp_logs(base_dir, error_bounds=['1e-3', '1e-4']):
    """Find all topoSZp log files in the analysis directory for specified error bounds."""
    log_files = {}
    
    # Find all datasets (CESM-OCEAN, CESM-LAND, etc.)
    for dataset_dir in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
        
        # Find all topoSZp_* folders
        topoSZp_pattern = os.path.join(dataset_path, 'topoSZp_*')
        topoSZp_folders = glob.glob(topoSZp_pattern)
        
        for folder_path in topoSZp_folders:
            folder_name = os.path.basename(folder_path)
            
            # Look for log files with specified error bounds - collect ALL available
            found_any = False
            for eb in error_bounds:
                log_file = os.path.join(folder_path, f'{eb}.log')
                if os.path.exists(log_file):
                    # Include error bound in key to distinguish them
                    key = (dataset_dir, folder_name, eb)
                    log_files[key] = log_file
                    found_any = True
            
            # Fallback: use any .log file if specified error bounds not found
            if not found_any:
                log_pattern = os.path.join(folder_path, '*.log')
                log_files_found = glob.glob(log_pattern)
                if log_files_found:
                    # Prefer 1e-3.log if available
                    log_file = None
                    for lf in log_files_found:
                        if '1e-3.log' in lf:
                            log_file = lf
                            break
                    if not log_file:
                        log_file = log_files_found[0]
                    if log_file:
                        # Try to extract error bound from filename
                        eb = '1e-3'  # default
                        for e in error_bounds:
                            if e in log_file:
                                eb = e
                                break
                        key = (dataset_dir, folder_name, eb)
                        log_files[key] = log_file
    
    return log_files

def calculate_statistics(times):
    """Calculate average, min, max for a list of times."""
    if not times:
        return 0.0, 0.0, 0.0, 0
    return sum(times) / len(times), min(times), max(times), len(times)

def generate_table(log_files, output_file):
    """Generate a table of average compression and decompression times."""
    results = []
    
    # Group by dataset, folder name, and error bound
    for (dataset, folder, error_bound), log_path in sorted(log_files.items()):
        comp_times, decomp_times = parse_log_file(log_path)
        
        comp_avg, comp_min, comp_max, comp_count = calculate_statistics(comp_times)
        decomp_avg, decomp_min, decomp_max, decomp_count = calculate_statistics(decomp_times)
        
        results.append({
            'dataset': dataset,
            'folder': folder,
            'error_bound': error_bound,
            'comp_avg': comp_avg,
            'comp_min': comp_min,
            'comp_max': comp_max,
            'comp_count': comp_count,
            'decomp_avg': decomp_avg,
            'decomp_min': decomp_min,
            'decomp_max': decomp_max,
            'decomp_count': decomp_count,
            'log_file': log_path
        })
    
    # Write CSV file
    with open(output_file + '.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'dataset', 'folder', 'error_bound', 'comp_avg', 'comp_min', 'comp_max', 'comp_count',
            'decomp_avg', 'decomp_min', 'decomp_max', 'decomp_count'
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'dataset': r['dataset'],
                'folder': r['folder'],
                'error_bound': r.get('error_bound', '1e-3'),
                'comp_avg': f"{r['comp_avg']:.6f}",
                'comp_min': f"{r['comp_min']:.6f}",
                'comp_max': f"{r['comp_max']:.6f}",
                'comp_count': r['comp_count'],
                'decomp_avg': f"{r['decomp_avg']:.6f}",
                'decomp_min': f"{r['decomp_min']:.6f}",
                'decomp_max': f"{r['decomp_max']:.6f}",
                'decomp_count': r['decomp_count']
            })
    
    # Write formatted text table
    with open(output_file + '.txt', 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("Average Compression and Decompression Times\n")
        f.write("=" * 120 + "\n\n")
        
        # Group by dataset
        datasets = sorted(set(r['dataset'] for r in results))
        
        for dataset in datasets:
            f.write(f"\n{dataset}:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Folder':<25} {'Error Bound':<12} {'Comp Avg (s)':<15} {'Comp Min':<12} {'Comp Max':<12} {'Decomp Avg (s)':<15} {'Decomp Min':<12} {'Decomp Max':<12} {'Files':<8}\n")
            f.write("-" * 120 + "\n")
            
            dataset_results = [r for r in results if r['dataset'] == dataset]
            for r in sorted(dataset_results, key=lambda x: (x['error_bound'], x['folder'])):
                f.write(f"{r['folder']:<25} "
                       f"{r.get('error_bound', '1e-3'):<12} "
                       f"{r['comp_avg']:<15.6f} "
                       f"{r['comp_min']:<12.6f} "
                       f"{r['comp_max']:<12.6f} "
                       f"{r['decomp_avg']:<15.6f} "
                       f"{r['decomp_min']:<12.6f} "
                       f"{r['decomp_max']:<12.6f} "
                       f"{r['comp_count']:<8}\n")
        
        # Summary table (averages across all datasets for each folder type)
        f.write("\n" + "=" * 120 + "\n")
        f.write("Summary: Average Times Across All Datasets (by folder type)\n")
        f.write("=" * 120 + "\n\n")
        
        # Group by folder name (e.g., topoSZp_1_thread, topoSZp_2_thread)
        folder_types = defaultdict(list)
        for r in results:
            folder_types[r['folder']].append(r)
        
        f.write(f"{'Folder Type':<25} {'Avg Comp (s)':<15} {'Avg Decomp (s)':<15} {'Datasets':<10}\n")
        f.write("-" * 120 + "\n")
        
        for folder_type in sorted(folder_types.keys()):
            folder_results = folder_types[folder_type]
            avg_comp = sum(r['comp_avg'] for r in folder_results) / len(folder_results)
            avg_decomp = sum(r['decomp_avg'] for r in folder_results) / len(folder_results)
            num_datasets = len(folder_results)
            
            f.write(f"{folder_type:<25} {avg_comp:<15.6f} {avg_decomp:<15.6f} {num_datasets:<10}\n")
    
    print(f"Generated tables:")
    print(f"  - {output_file}.csv")
    print(f"  - {output_file}.txt")
    
    return results

def generate_thread_comparison_table(log_files, output_file):
    """Generate a table with datasets as rows and thread counts as columns, grouped by error bound."""
    # Extract thread-specific data
    thread_folders = ['topoSZp_1_thread', 'topoSZp_2_thread', 'topoSZp_4_thread', 
                      'topoSZp_8_thread', 'topoSZp_16_thread', 'topoSZp_18_thread',
                      'topoSZp_20_thread', 'topoSZp_24_thread', 'topoSZp_26_thread', 
                      'topoSZp_28_thread', 'topoSZp_30_thread',
                      'topoSZp_32_thread', 'topoSZp_36_thread']
    thread_numbers = [1, 2, 4, 8, 16, 18, 20, 24, 26, 28, 30, 32, 36]

    # Organize data by error bound, dataset, and thread
    error_bound_data = defaultdict(lambda: defaultdict(dict))
    
    for (dataset, folder, error_bound), log_path in log_files.items():
        # Only process thread-specific folders
        if folder in thread_folders:
            comp_times, decomp_times = parse_log_file(log_path)
            if comp_times and decomp_times:
                comp_avg = sum(comp_times) / len(comp_times)
                decomp_avg = sum(decomp_times) / len(decomp_times)
                error_bound_data[error_bound][dataset][folder] = (comp_avg, decomp_avg)
    
    # Generate tables for each error bound
    for error_bound in sorted(error_bound_data.keys()):
        dataset_data = error_bound_data[error_bound]
        # Get all datasets for this error bound
        datasets = sorted([d for d in dataset_data.keys() if any(f in dataset_data[d] for f in thread_folders)])
    
        # Generate CSV file for this error bound
        csv_filename = f"{output_file}_thread_comparison_{error_bound}.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header row
            header = ['Dataset'] + [f'Thread {t} (Comp/Decomp)' for t in thread_numbers]
            writer.writerow(header)
            
            # Data rows
            for dataset in datasets:
                row = [dataset]
                for folder in thread_folders:
                    if folder in dataset_data[dataset]:
                        comp_avg, decomp_avg = dataset_data[dataset][folder]
                        row.append(f"{comp_avg:.6f}/{decomp_avg:.6f}")
                    else:
                        row.append("N/A")
                writer.writerow(row)
        
        # Generate formatted text table for this error bound
        txt_filename = f"{output_file}_thread_comparison_{error_bound}.txt"
        with open(txt_filename, 'w') as f:
            f.write("=" * 150 + "\n")
            f.write(f"Average Compression and Decompression Times by Dataset and Thread Count (Error Bound: {error_bound})\n")
            f.write("=" * 150 + "\n\n")
            
            # Header
            f.write(f"{'Dataset':<20}")
            for t in thread_numbers:
                f.write(f"  Thread {t:<2}")
            f.write("\n")
            f.write(f"{'':<20}")
            for t in thread_numbers:
                f.write(f"  {'Comp/Decomp':<18}")
            f.write("\n")
            f.write("-" * 150 + "\n")
            
            # Data rows
            for dataset in datasets:
                f.write(f"{dataset:<20}")
                for folder in thread_folders:
                    if folder in dataset_data[dataset]:
                        comp_avg, decomp_avg = dataset_data[dataset][folder]
                        f.write(f"  {comp_avg:>7.6f}/{decomp_avg:>7.6f}")
                    else:
                        f.write(f"  {'N/A':>17}")
                f.write("\n")
            
            # Summary row (averages across all datasets for each thread count)
            f.write("\n" + "-" * 150 + "\n")
            f.write(f"{'Average (All Datasets)':<20}")
            for folder in thread_folders:
                thread_results = []
                for dataset in datasets:
                    if folder in dataset_data[dataset]:
                        thread_results.append(dataset_data[dataset][folder])
                if thread_results:
                    avg_comp = sum(c for c, d in thread_results) / len(thread_results)
                    avg_decomp = sum(d for c, d in thread_results) / len(thread_results)
                    f.write(f"  {avg_comp:>7.6f}/{avg_decomp:>7.6f}")
                else:
                    f.write(f"  {'N/A':>17}")
            f.write("\n")
    
    # Also generate a combined CSV with all error bounds
    combined_csv = output_file + '_thread_comparison.csv'
    with open(combined_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header row
        header = ['Dataset', 'Error Bound'] + [f'Thread {t} (Comp/Decomp)' for t in thread_numbers]
        writer.writerow(header)
        
        # Data rows for all error bounds
        for error_bound in sorted(error_bound_data.keys()):
            dataset_data = error_bound_data[error_bound]
            datasets = sorted([d for d in dataset_data.keys() if any(f in dataset_data[d] for f in thread_folders)])
            for dataset in datasets:
                row = [dataset, error_bound]
                for folder in thread_folders:
                    if folder in dataset_data[dataset]:
                        comp_avg, decomp_avg = dataset_data[dataset][folder]
                        row.append(f"{comp_avg:.6f}/{decomp_avg:.6f}")
                    else:
                        row.append("N/A")
                writer.writerow(row)
    
    print(f"Generated thread comparison tables:")
    for error_bound in sorted(error_bound_data.keys()):
        print(f"  - {output_file}_thread_comparison_{error_bound}.csv")
        print(f"  - {output_file}_thread_comparison_{error_bound}.txt")
    print(f"  - {combined_csv} (combined)")

def main():
    base_dir = os.getcwd()
    
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return
    
    print("Scanning for topoSZp log files (1e-3 and 1e-4)...")
    log_files = find_all_topoSZp_logs(base_dir, error_bounds=['1e-3', '1e-4'])
    
    print(f"Found {len(log_files)} topoSZp folders with log files")
    
    output_file = os.path.join(base_dir, 'topoSZp_timing_table')
    results = generate_table(log_files, output_file)
    
    print(f"\nProcessed {len(results)} log files")
    print(f"Results saved to: {output_file}.csv and {output_file}.txt")
    
    # Generate thread comparison table
    print("\nGenerating thread comparison table...")
    generate_thread_comparison_table(log_files, output_file)

if __name__ == '__main__':
    main()


