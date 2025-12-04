#!/usr/bin/env python3
"""
Convert .dat file (binary float data) to VTK ImageData (.vti) format
"""

import numpy as np
import vtk
from vtk.util import numpy_support
import argparse
import sys
import os

def load_dat_file(filename, nx, ny, nz=1, dtype=np.float32):
    """Load binary data from .dat file"""
    try:
        data = np.fromfile(filename, dtype=dtype)
        expected_size = nx * ny * nz
        if len(data) != expected_size:
            print(f"Warning: Expected {expected_size} elements, got {len(data)}")
            if len(data) < expected_size:
                raise ValueError(f"File too small: got {len(data)}, expected {expected_size}")
            data = data[:expected_size]
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def create_vtk_image(data, nx, ny, nz, scalar_name="data", spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    Create VTK ImageData from numpy array.
    
    The input data is a flat array in row-major (C) order.
    For 2D data, it's typically stored as (ny, nx) = (rows, cols) in row-major order.
    
    VTK ImageData expects:
    - Dimensions: (nx, ny, nz) where nx is width, ny is height
    - Data in Fortran order where x varies fastest
    
    Strategy:
    1. Reshape to (ny, nx) to match the logical layout (rows, cols)
    2. Transpose to (nx, ny) to match VTK's dimension order
    3. Reshape to (nx, ny, nz) for 3D structure
    4. Ravel in Fortran order so x varies fastest
    """
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin)
    
    if nz == 1:
        # 2D case: data is stored as (ny, nx) in row-major order
        # Reshape to (ny, nx) first to get the correct logical layout
        volume_2d = data.reshape((ny, nx), order="C")
        # Transpose to (nx, ny) to match VTK's dimension order
        volume_2d_T = volume_2d.T.copy()  # .copy() ensures contiguous array
        # Reshape to 3D
        volume = volume_2d_T.reshape((nx, ny, nz), order="C")
        # Ravel in Fortran order: x varies fastest (correct for VTK)
        vtk_data = volume.ravel(order="F")
    else:
        # 3D case: assume data is stored as (nz, ny, nx) in row-major order
        # Reshape to (nz, ny, nx) first
        volume_3d = data.reshape((nz, ny, nx), order="C")
        # Transpose to (nx, ny, nz) for VTK
        volume = np.transpose(volume_3d, (2, 1, 0)).copy()
        # Ravel in Fortran order
        vtk_data = volume.ravel(order="F")
    
    # Verify data size
    expected_size = nx * ny * nz
    if len(vtk_data) != expected_size:
        raise ValueError(f"Data size mismatch: expected {expected_size}, got {len(vtk_data)}")
    
    vtk_array = numpy_support.numpy_to_vtk(
        num_array=vtk_data,
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )
    vtk_array.SetName(scalar_name)
    image.GetPointData().SetScalars(vtk_array)
    
    return image

def main():
    parser = argparse.ArgumentParser(
        description="Convert .dat file (binary float) to VTK ImageData (.vti) format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2D data
  python3 dat_to_vti.py input.dat --nx 3600 --ny 1800 -o output.vti
  
  # 3D data
  python3 dat_to_vti.py input.dat --nx 100 --ny 100 --nz 50 -o output.vti
  
  # With custom spacing and origin
  python3 dat_to_vti.py input.dat --nx 3600 --ny 1800 --spacing 0.1 0.1 1.0 --origin 0 0 0 -o output.vti
        """
    )
    
    parser.add_argument('input', help='Input .dat file (binary float data)')
    parser.add_argument('--nx', type=int, required=True, help='Number of points in X direction')
    parser.add_argument('--ny', type=int, required=True, help='Number of points in Y direction')
    parser.add_argument('--nz', type=int, default=1, help='Number of points in Z direction (default: 1)')
    parser.add_argument('-o', '--output', help='Output .vti file (default: input.vti)')
    parser.add_argument('--scalar-name', default='data', help='Name of scalar array (default: data)')
    parser.add_argument('--spacing', nargs=3, type=float, default=[1.0, 1.0, 1.0],
                       help='Spacing in X, Y, Z directions (default: 1.0 1.0 1.0)')
    parser.add_argument('--origin', nargs=3, type=float, default=[0.0, 0.0, 0.0],
                       help='Origin coordinates (default: 0.0 0.0 0.0)')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32',
                       help='Data type (default: float32)')
    
    args = parser.parse_args()
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        base = os.path.splitext(args.input)[0]
        output_file = f"{base}.vti"
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    print(f"Converting {args.input} to {output_file}")
    print(f"Dimensions: {args.nx} x {args.ny} x {args.nz}")
    print(f"Expected data size: {args.nx * args.ny * args.nz:,} points")
    
    # Load data
    dtype_map = {'float32': np.float32, 'float64': np.float64}
    data = load_dat_file(args.input, args.nx, args.ny, args.nz, dtype_map[args.dtype])
    
    if data is None:
        print("Error: Failed to load data")
        sys.exit(1)
    
    print(f"Loaded {len(data):,} data points")
    print(f"Data range: [{data.min():.6f}, {data.max():.6f}]")
    
    # Create VTK image
    try:
        image = create_vtk_image(
            data, args.nx, args.ny, args.nz,
            scalar_name=args.scalar_name,
            spacing=tuple(args.spacing),
            origin=tuple(args.origin)
        )
    except Exception as e:
        print(f"Error creating VTK image: {e}")
        sys.exit(1)
    
    # Write to file
    print(f"\nWriting to {output_file}...")
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(image)
    writer.SetCompressorTypeToZLib()
    writer.SetDataModeToBinary()
    writer.Write()
    
    print(f"Successfully created {output_file}")
    print(f"VTK Image dimensions: {image.GetDimensions()}")
    print(f"VTK Image spacing: {image.GetSpacing()}")
    print(f"VTK Image origin: {image.GetOrigin()}")

if __name__ == "__main__":
    main()

