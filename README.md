# TopoSZp-CUDA

**Topology-preserving lossy compression for scientific data with GPU acceleration.**

TopoSZp-CUDA compresses floating-point scientific datasets while preserving
topological features — critical points (maxima, minima, saddles) and their
relative ordering. It supports both 2D and 3D structured grids and provides
CUDA-accelerated compression, decompression, and topology analysis alongside an
OpenMP-based CPU reference implementation.

* **Developers:** Tripti Agarwal (topology kernels, CUDA port, 3D extensions),
  Sheng Di (compression utilities)
* **Contact:** tripti.agarwal@utah.edu, tripti.coer@gmail.com
* **License:** MIT

---

## Features

- **Error-bounded lossy compression** — reconstructed values stay within a
  user-specified absolute or relative error bound.
- **Critical point preservation** — maxima, minima, and saddle points in the
  original data are preserved in the decompressed output.
- **2D and 3D data** — 4-connected neighborhoods for 2D grids,
  6-connected neighborhoods for 3D grids.
- **Saddle stencil restoration** — post-decompression pipeline restores saddle
  topology using axis-pattern detection and Refined Bit-Flip (RBF) with a
  3×3×3 Gaussian kernel (3D).
- **CUDA GPU acceleration** — parallel critical point finding, bin-based
  sorting, and block-level compression/decompression on NVIDIA GPUs.
- **OpenMP CPU path** — full CPU implementation for comparison and
  environments without a GPU.
- **LOPC comparison tools** — built-in benchmarks against the LOPC GPU
  compressor for critical point preservation and local order.

---

## Algorithm Overview

The topology-preserving compression pipeline works as follows:

1. **Quantize** — map floating-point values to integers using the error bound.
2. **Find critical points** — classify each grid point as maximum, minimum,
   saddle, or regular using its immediate neighbors (4-connected in 2D,
   6-connected in 3D).
3. **Sort critical points** — within each quantization bin, sort critical points
   by their original floating-point value to preserve relative ordering.
4. **Compress with topology metadata** — pack quantized data with 2-bit
   critical-point type tags and sort-position data into a compact byte stream.
5. **Decompress** — reconstruct quantized values and extract the embedded type
   and sort-position arrays.
6. **Post-process** — apply stencils to enforce extrema, then restore saddle
   topology using axis-pattern detection and RBF.

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| CMake | ≥ 3.20 | Required for building with CUDA |
| CUDA Toolkit | ≥ 11.0 | For GPU-accelerated code |
| C++17 compiler | GCC ≥ 7, Clang ≥ 5 | CUDA sources require C++17 |
| OpenMP | (optional) | Required for CPU reference and test programs |
| Thrust | (bundled with CUDA) | Used for parallel scan in compression |
| zstd | (optional) | Linked if found via pkg-config |

### Supported GPU Architectures

The default CMake build targets these CUDA architectures:

| Architecture | GPUs |
|---|---|
| `sm_60` / `sm_61` | Pascal (P100, GTX 1080) |
| `sm_70` | Volta (V100) |
| `sm_75` | Turing (T4, RTX 2080) |
| `sm_80` / `sm_86` | Ampere (A100, RTX 3090) |
| `sm_89` | Ada Lovelace (RTX 4090, L40) |
| `sm_90` | Hopper (H100) |

Override with `-DCMAKE_CUDA_ARCHITECTURES="80;90"` to target specific GPUs.

---

## Building

### CMake (recommended — required for CUDA)

```bash
mkdir build && cd build
cmake .. -DENABLE_CUDA=ON
make -j$(nproc)
```

To target specific GPU architectures:

```bash
cmake .. -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;90"
```

To install:

```bash
cmake .. -DENABLE_CUDA=ON -DCMAKE_INSTALL_PREFIX=/path/to/install
make -j$(nproc)
make install
```

### Autotools (CPU-only, no CUDA support)

```bash
./autogen.sh
./configure --prefix=$(pwd)/install --enable-openmp
make -j$(nproc)
make install
```

Set the number of OpenMP threads:

```bash
export OMP_NUM_THREADS=16
```

> **Note:** The Autotools build does not compile CUDA code. Use CMake for GPU
> support.

---

## Usage

All programs read raw binary floating-point files (32-bit float by default).

### 2D Topology-Preserving Compression

```bash
./test_cuda_topology <input_file> <rows> <cols> <errBound> <blockSize>
```

**Example** (CESM-ATM 2D, 1800 × 3600 grid, absolute error bound 1e-3):

```bash
./test_cuda_topology CLDHGH_1_1800_3600.dat 1800 3600 1e-3 64
```

### 3D Topology-Preserving Compression

```bash
./test_cuda_topology_3d <input_file> <d1> <d2> <d3> <errBound> <blockSize>
```

**Example** (3D volume, 100 × 500 × 500 grid, relative error bound 1e-4):

```bash
./test_cuda_topology_3d volume_100_500_500.dat 100 500 500 rel:1e-4 64
```

### Error Bound Modes

- **Absolute:** `1e-3` — maximum pointwise error is 0.001.
- **Relative:** `rel:1e-4` — error bound is computed as a fraction of the data
  value range (`(max - min) * rel`).

### Compression / Decompression (without topology)

```bash
./test_cuda_compress <input_file> <nbEle> <errBound> <blockSize>
./test_cuda_decompress <input_file> <nbEle> <absErrBound> <blockSize> [rows cols]
```

### LOPC Comparison

```bash
./compare_with_lopc <orig_file> <lopc_decomp> <rows> <cols> <eb> <blockSize> <lopc_comp_size>
./compare_with_lopc_3d <orig_file> <lopc_decomp> <d1> <d2> <d3> <eb> <blockSize> <lopc_comp_size>
```

---

## Downstream CMake Integration

After installing, use `find_package` in your CMake project:

```cmake
find_package(SZp REQUIRED)
target_link_libraries(myapp SZp::SZp_cuda)  # GPU library
target_link_libraries(myapp SZp::SZp)       # CPU (OpenMP) library
```

---

## Repository Structure

```
├── CMakeLists.txt              # CMake build (primary, CUDA + CPU)
├── configure.ac / Makefile.am  # Autotools build (CPU-only)
├── SZp/
│   ├── include/
│   │   ├── szp.h                       # Main API header
│   │   ├── szp_cuda_compress.cuh       # CUDA compression API
│   │   ├── szp_cuda_decompress.cuh     # CUDA decompression API
│   │   ├── szp_cuda_topology.cuh       # 2D topology (CUDA)
│   │   ├── szp_cuda_topology_3d.cuh    # 3D topology (CUDA)
│   │   ├── szp_cuda_common.cuh         # Shared CUDA device utilities
│   │   ├── szp_topology_3d.h           # 3D topology (CPU)
│   │   └── ...                         # Compression utilities
│   └── src/
│       ├── szp_cuda_compress.cu        # CUDA compression kernels
│       ├── szp_cuda_decompress.cu      # CUDA decompression kernels
│       ├── szp_cuda_topology.cu        # 2D critical point kernels
│       ├── szp_cuda_topology_3d.cu     # 3D critical point kernels
│       ├── szp_topology_3d.cc          # 3D topology (CPU/OpenMP)
│       └── ...                         # CPU compression sources
└── example/
    ├── test_cuda_topology.cu           # 2D topology end-to-end test
    ├── test_cuda_topology_3d.cu        # 3D topology end-to-end test
    ├── test_cuda_compress.cu           # Compression validation test
    ├── test_cuda_decompress.cu         # Decompression validation test
    ├── compare_with_lopc.cu            # 2D LOPC comparison
    ├── compare_with_lopc_3d.cu         # 3D LOPC comparison
    └── analysis/                       # Benchmark logs and scripts
```

---

## Citation

If you use TopoSZp-CUDA in your research, please cite the relevant publication.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for
details.
