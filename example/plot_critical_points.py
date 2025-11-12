#!/usr/bin/env python3
"""
Matplotlib-based visualization of critical points data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import sys
import os

def load_float_data(filename, rows, cols):
    """Load float data from binary file"""
    try:
        data = np.fromfile(filename, dtype=np.float32)
        if len(data) != rows * cols:
            print(f"Warning: Expected {rows * cols} elements, got {len(data)}")
        data = data[:rows * cols].reshape(rows, cols)
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def load_critical_points(filename):
    """Load critical points from text file"""
    try:
        cps = []
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 5:
                    x, y, type_val, quantized_bin, sort_position = map(int, parts[:5])
                    cps.append({
                        'x': x, 'y': y, 'type': type_val, 
                        'quantized_bin': quantized_bin, 'sort_position': sort_position
                    })
        return cps
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

def create_custom_colormap():
    """Create a blue-to-green colormap for data visualization"""
    colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#008000']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('blue_green', colors, N=n_bins)
    return cmap

def plot_data_with_critical_points(data, cps, title, filename, rows, cols):
    """Plot data with critical points overlay"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create custom colormap
    cmap = create_custom_colormap()
    
    # Plot data as background
    im = ax.imshow(data, cmap=cmap, origin='lower', aspect='equal')
    
    # Separate critical points by type
    maxima = [cp for cp in cps if cp['type'] == 1]
    minima = [cp for cp in cps if cp['type'] == 2]
    saddles = [cp for cp in cps if cp['type'] == 3]
    
    # Plot critical points with different markers and colors
    if maxima:
        max_x = [cp['x'] for cp in maxima]
        max_y = [cp['y'] for cp in maxima]
        ax.scatter(max_x, max_y, c='red', marker='o', s=20, alpha=0.8, 
                  label=f'Maxima ({len(maxima)})', edgecolors='darkred', linewidth=0.5)
    
    if minima:
        min_x = [cp['x'] for cp in minima]
        min_y = [cp['y'] for cp in minima]
        ax.scatter(min_x, min_y, c='darkblue', marker='o', s=20, alpha=0.8,
                  label=f'Minima ({len(minima)})', edgecolors='navy', linewidth=0.5)
    
    if saddles:
        sad_x = [cp['x'] for cp in saddles]
        sad_y = [cp['y'] for cp in saddles]
        ax.scatter(sad_x, sad_y, c='black', marker='x', s=30, alpha=0.9,
                  label=f'Saddles ({len(saddles)})', linewidth=1.5)
    
    # Customize plot
    ax.set_title(f'{title}\nData Range: [{data.min():.6f}, {data.max():.6f}]', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Data Value', fontsize=12)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Set axis limits
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    
    # Add grid for better visibility
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_critical_points_comparison(orig_data, orig_cps, decomp_data, decomp_cps, 
                                  rows, cols, filename):
    """Create comparison plot showing original and decompressed data side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    cmap = create_custom_colormap()
    
    # Plot 1: Original data with critical points
    im1 = ax1.imshow(orig_data, cmap=cmap, origin='lower', aspect='equal')
    orig_maxima = [cp for cp in orig_cps if cp['type'] == 1]
    orig_minima = [cp for cp in orig_cps if cp['type'] == 2]
    orig_saddles = [cp for cp in orig_cps if cp['type'] == 3]
    
    if orig_maxima:
        ax1.scatter([cp['x'] for cp in orig_maxima], [cp['y'] for cp in orig_maxima],
                   c='red', marker='o', s=20, alpha=0.9, edgecolors='darkred', linewidth=0.8,
                   label=f'Maxima ({len(orig_maxima)})')
    if orig_minima:
        ax1.scatter([cp['x'] for cp in orig_minima], [cp['y'] for cp in orig_minima],
                   c='green', marker='^', s=20, alpha=0.8, edgecolors='darkgreen', linewidth=0.5,
                   label=f'Minima ({len(orig_minima)})')
    if orig_saddles:
        # White X marks for saddles - thinner and lighter
        ax1.scatter([cp['x'] for cp in orig_saddles], [cp['y'] for cp in orig_saddles],
                   c='white', marker='x', s=25, alpha=0.7, linewidth=1.5,
                   label=f'Saddles ({len(orig_saddles)})')
    
    ax1.set_title(f'Original Data\n{len(orig_cps)} critical points', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Data Value', fontsize=11)
    
    # Plot 2: Decompressed data with critical points
    im2 = ax2.imshow(decomp_data, cmap=cmap, origin='lower', aspect='equal')
    decomp_maxima = [cp for cp in decomp_cps if cp['type'] == 1]
    decomp_minima = [cp for cp in decomp_cps if cp['type'] == 2]
    decomp_saddles = [cp for cp in decomp_cps if cp['type'] == 3]
    
    if decomp_maxima:
        ax2.scatter([cp['x'] for cp in decomp_maxima], [cp['y'] for cp in decomp_maxima],
                   c='red', marker='o', s=20, alpha=0.9, edgecolors='darkred', linewidth=0.8,
                   label=f'Maxima ({len(decomp_maxima)})')
    if decomp_minima:
        ax2.scatter([cp['x'] for cp in decomp_minima], [cp['y'] for cp in decomp_minima],
                   c='green', marker='^', s=20, alpha=0.8, edgecolors='darkgreen', linewidth=0.5,
                   label=f'Minima ({len(decomp_minima)})')
    if decomp_saddles:
        # White X marks for saddles - thinner and lighter
        ax2.scatter([cp['x'] for cp in decomp_saddles], [cp['y'] for cp in decomp_saddles],
                   c='white', marker='x', s=25, alpha=0.7, linewidth=1.5,
                   label=f'Saddles ({len(decomp_saddles)})')
    
    ax2.set_title(f'Decompressed Data\n{len(decomp_cps)} critical points', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Coordinate', fontsize=12)
    ax2.set_ylabel('Y Coordinate', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Data Value', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_statistics(orig_cps, decomp_cps, filename):
    """Plot statistical comparison of critical points"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Count by type
    orig_counts = {'Maxima': 0, 'Minima': 0, 'Saddles': 0}
    decomp_counts = {'Maxima': 0, 'Minima': 0, 'Saddles': 0}
    
    for cp in orig_cps:
        if cp['type'] == 1:
            orig_counts['Maxima'] += 1
        elif cp['type'] == 2:
            orig_counts['Minima'] += 1
        elif cp['type'] == 3:
            orig_counts['Saddles'] += 1
    
    for cp in decomp_cps:
        if cp['type'] == 1:
            decomp_counts['Maxima'] += 1
        elif cp['type'] == 2:
            decomp_counts['Minima'] += 1
        elif cp['type'] == 3:
            decomp_counts['Saddles'] += 1
    
    # Plot 1: Type distribution comparison
    types = list(orig_counts.keys())
    orig_values = list(orig_counts.values())
    decomp_values = list(decomp_counts.values())
    
    x = np.arange(len(types))
    width = 0.35
    
    ax1.bar(x - width/2, orig_values, width, label='Original', color='skyblue', alpha=0.8)
    ax1.bar(x + width/2, decomp_values, width, label='Decompressed', color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Critical Point Type')
    ax1.set_ylabel('Count')
    ax1.set_title('Critical Points by Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(types)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (orig, decomp) in enumerate(zip(orig_values, decomp_values)):
        ax1.text(i - width/2, orig + 10, str(orig), ha='center', va='bottom')
        ax1.text(i + width/2, decomp + 10, str(decomp), ha='center', va='bottom')
    
    # Plot 2: Preservation rate
    preservation_rates = []
    for i, type_name in enumerate(types):
        if orig_values[i] > 0:
            rate = decomp_values[i] / orig_values[i] * 100
        else:
            rate = 100
        preservation_rates.append(rate)
    
    bars = ax2.bar(types, preservation_rates, color=['green', 'green', 'orange'], alpha=0.7)
    ax2.set_ylabel('Preservation Rate (%)')
    ax2.set_title('Critical Points Preservation Rate')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, rate in zip(bars, preservation_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # Plot 3: Spatial distribution (first 50x50 region)
    region_size = 50
    orig_region = np.zeros((region_size, region_size), dtype=int)
    decomp_region = np.zeros((region_size, region_size), dtype=int)
    
    for cp in orig_cps:
        if 0 <= cp['x'] < region_size and 0 <= cp['y'] < region_size:
            orig_region[cp['y'], cp['x']] = cp['type']
    
    for cp in decomp_cps:
        if 0 <= cp['x'] < region_size and 0 <= cp['y'] < region_size:
            decomp_region[cp['y'], cp['x']] = cp['type']
    
    im3 = ax3.imshow(orig_region, cmap='viridis', origin='lower', aspect='equal')
    ax3.set_title(f'Original Critical Points\n(First {region_size}x{region_size} region)')
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='Type')
    
    im4 = ax4.imshow(decomp_region, cmap='viridis', origin='lower', aspect='equal')
    ax4.set_title(f'Decompressed Critical Points\n(First {region_size}x{region_size} region)')
    ax4.set_xlabel('X Coordinate')
    ax4.set_ylabel('Y Coordinate')
    plt.colorbar(im4, ax=ax4, shrink=0.8, label='Type')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def find_critical_points_szp(data, rows, cols, err_bound=1e-3):
    """Use SZp library to find critical points (same as C++ code)"""
    import subprocess
    import tempfile
    import os
    
    # Write data to temporary file
    with tempfile.NamedTemporaryFile(suffix='.f32', delete=False) as tmp_file:
        data.astype(np.float32).tofile(tmp_file.name)
        tmp_filename = tmp_file.name
    
    try:
        # Call the C++ test program to get critical points
        result = subprocess.run([
            './test_critical_points_match', tmp_filename, tmp_filename, 
            str(err_bound), str(rows), str(cols)
        ], capture_output=True, text=True, encoding='utf-8', errors='replace',
        cwd='/u/tagarwal1/TopologySZp/SZp/example')
        
        # Parse the output to extract critical point counts
        lines = result.stdout.split('\n')
        orig_count = 0
        decomp_count = 0
        
        for line in lines:
            if 'Found' in line and 'critical points in original data' in line:
                orig_count = int(line.split()[1])
            elif 'Found' in line and 'critical points in decompressed data' in line:
                decomp_count = int(line.split()[1])
        
        # For now, return empty lists since we can't easily extract the actual coordinates
        # This is a limitation - we'd need to modify the C++ code to output coordinates
        print(f"Note: SZp found {orig_count} original and {decomp_count} decompressed critical points")
        print("Using simplified detection for visualization (coordinates not available from SZp output)")
        
        # Fall back to simple detection but with better filtering
        return find_critical_points_simple_filtered(data, rows, cols, err_bound)
        
    finally:
        # Clean up temporary file
        os.unlink(tmp_filename)

def find_critical_points_simple_filtered(data, rows, cols, err_bound=1e-3):
    """
    Critical points detection matching the C++ implementation exactly.
    Uses 4-connected neighborhood (up, down, left, right) as in szp_find_critical_points.
    """
    cps = []
    
    # Use 4-connected neighborhood to match C++ code exactly
    # C++ code checks: up, down, left, right (not diagonal neighbors)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = data[i, j]
            up = data[i - 1, j]
            down = data[i + 1, j]
            left = data[i, j - 1]
            right = data[i, j + 1]
            
            # Local maximum: center > all 4 neighbors (matching C++ code)
            if center > up and center > down and center > left and center > right:
                # Compute quantized_bin using same formula as C++ code
                inver_bound = 1.0 / err_bound
                quantized_bin = int((center + err_bound) * inver_bound)
                cps.append({'x': j, 'y': i, 'type': 1, 'quantized_bin': quantized_bin, 'sort_position': 0})
            
            # Local minimum: center < all 4 neighbors (matching C++ code)
            elif center < up and center < down and center < left and center < right:
                # Compute quantized_bin using same formula as C++ code
                inver_bound = 1.0 / err_bound
                quantized_bin = int((center + err_bound) * inver_bound)
                cps.append({'x': j, 'y': i, 'type': 2, 'quantized_bin': quantized_bin, 'sort_position': 0})
            
            # Saddle point: matches C++ logic exactly
            # (center < up && center < down && center > left && center > right) ||
            # (center > up && center > down && center < left && center < right)
            elif ((center < up and center < down and center > left and center > right) or
                  (center > up and center > down and center < left and center < right)):
                # Compute quantized_bin using same formula as C++ code
                inver_bound = 1.0 / err_bound
                quantized_bin = int((center + err_bound) * inver_bound)
                cps.append({'x': j, 'y': i, 'type': 3, 'quantized_bin': quantized_bin, 'sort_position': 0})
    
    return cps

def main():
    if len(sys.argv) != 5:
        print("Usage: python3 plot_critical_points.py <orig_data.bin> <decomp_data.bin> <rows> <cols>")
        print("Example: python3 plot_critical_points.py CLOUDf48.bin_slice_44.f32 CLOUDf48.bin_slice_44.f32.SZp.out 500 500")
        sys.exit(1)
    
    orig_data_file = sys.argv[1]
    decomp_data_file = sys.argv[2]
    rows = int(sys.argv[3])
    cols = int(sys.argv[4])
    
    print("=== Matplotlib Critical Points Visualization ===")
    print(f"Loading data: {rows}x{cols}")
    
    # Load data
    orig_data = load_float_data(orig_data_file, rows, cols)
    decomp_data = load_float_data(decomp_data_file, rows, cols)
    
    if orig_data is None or decomp_data is None:
        print("Error: Could not load data files")
        sys.exit(1)
    
    # Try to use C++ program if available, otherwise use Python-based detection
    import subprocess
    import os
    
    cpp_program = './test_critical_points_match'
    use_cpp = os.path.exists(cpp_program)
    
    if use_cpp:
        print("Running C++ critical point detection...")
        result = subprocess.run([
            cpp_program, orig_data_file, decomp_data_file, 
            '1E-3', str(rows), str(cols)
        ], capture_output=True, text=True, encoding='utf-8', errors='replace', 
        cwd='/u/tagarwal1/TopologySZp/SZp/example')
        
        if result.returncode == 0:
            # Load critical points from files created by C++ program
            print("Loading critical points from C++ output...")
            orig_cps = load_critical_points("temp_original_critical_points.txt")
            decomp_cps = load_critical_points("temp_decompressed_critical_points.txt")
            
            if len(orig_cps) == 0 or len(decomp_cps) == 0:
                print("Warning: C++ program didn't create critical point files, using Python detection")
                use_cpp = False
        else:
            print("Warning: C++ program failed, using Python-based detection")
            print(result.stderr)
            use_cpp = False
    
    if not use_cpp:
        # Use Python-based critical point detection
        print("Using Python-based critical point detection...")
        err_bound = 1e-3
        orig_cps = find_critical_points_simple_filtered(orig_data, rows, cols, err_bound)
        decomp_cps = find_critical_points_simple_filtered(decomp_data, rows, cols, err_bound)
    
    print(f"Found {len(orig_cps)} original critical points")
    print(f"Found {len(decomp_cps)} decompressed critical points")
    
    # Create comparison plot only
    print("\nGenerating comparison plot...")
    
    # Comparison plot
    plot_critical_points_comparison(orig_data, orig_cps, decomp_data, decomp_cps,
                                  rows, cols, "critical_points_comparison_matplotlib.png")
    
    print("\n=== Comparison plot generated successfully! ===")
    print("Generated file:")
    print("- critical_points_comparison_matplotlib.png")

if __name__ == "__main__":
    main()
