# TopoSZp 

* Developers: Tripti Agarwal (topology part including kernels and utility), Sheng Di (utility)
* Email: tripti.agarwal@utah.edu, tripti.coer@gmail.com


## Installation
Configure and build the SZp:
```bash
# Decompress the downloaded files
# Change to the SZp directory and set up the installation directory:
cd SZp
mkdir install

# Run the configuration script:
./configure --prefix=$(pwd)/install --enable-openmp

# Compile the SZp using multiple threads:
make -j

# Install the compiled SZp:
make install

```

## Run SZp
```bash
export OMP_NUM_THREADS=$NUMTHREADS
```

## Topology-preserving compression: what the code guarantees

Build with CMake (`-DENABLE_CUDA=ON` adds the GPU library and tests) or with
the autotools steps above. Examples:

```bash
# 2D (rows cols) and 3D (d1 d2 d3, slowest dimension first)
./testfloat_compress_fastmode1   data.f32 64 1e-3 1800 3600
./testfloat_decompress_fastmode1 data.f32.SZp 6480000 64 1e-3 1800 3600
./testfloat_compress_fastmode1   CLDICE_1_26_1800_3600.f32 64 1e-7 26 1800 3600
./testfloat_decompress_fastmode1 CLDICE_1_26_1800_3600.f32.SZp 168480000 64 1e-7 26 1800 3600

# topology report (kept / FN / FT / FP per type, before and after correction)
g++ -O2 -fopenmp -ISZp/include example/topo_check.cc -o topo_check
./topo_check data.f32 data.f32.SZp.out 1e-3 1800 3600          # add d3 for 3D, --f64 for double data
```

* **Error bound.** `eb` is absolute. Plain quantization keeps every value within
  `eb` for float32 and float64 data (`SZp/include/szp_quant.h`); topology
  correction moves a value by less than `eb` more, so the output is within `2 eb`.
  The compressor refuses NaN/Inf and bounds whose bin indices do not fit 32 bits
  (`max|v| / eb >= 2^31`), and warns when the float32/float64 spacing at `max|v|`
  is larger than `eb` (topology correction cannot separate points there).
* **No false positives, no false types.** Quantization is monotone, and every
  correction is checked against the stored critical-point types of the point and
  its neighbours.
* **Extrema.** Every maximum and minimum is kept, and extrema that share a
  quantization bin keep the order of their original values (when float spacing
  allows it).
* **Saddles** are restored where possible (stencil search, then RBF smoothing),
  but are not guaranteed.
* **Definitions.** Critical points use strict comparisons with the face
  neighbours (4 in 2D, 6 in 3D). A saddle has at least one axis where the point
  is above both neighbours and one where it is below both; in 3D this counts
  many points as saddles (on CESM CLDICE about 89% of critical points).
* **Same result everywhere.** Output does not depend on the number of OpenMP
  threads, and the CPU (`szp_topo_restore_float_2d/3d`) and GPU
  (`szp_cuda_topo_restore_float_2d/3d`) restorations run the same algorithm.
* **Stream format.** Slot 0 of each per-thread offset table stores
  `"SZPT" | thread count`. Streams from older versions (slot 0 = 0) are still
  decoded, using thread-count detection.
* `diagnose.sh` builds everything and runs the thread-count, sanitizer,
  CPU/GPU and topology checks on a 2D and a 3D file.
