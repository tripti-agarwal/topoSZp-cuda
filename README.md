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

# PL mode: store and restore critical points with the standard PL definition
./testfloat_compress_fastmode1   --pl CLDICE_1_26_1800_3600.f32 64 1e-7 26 1800 3600
./testfloat_decompress_fastmode1 CLDICE_1_26_1800_3600.f32.SZp 168480000 64 1e-7 26 1800 3600   # mode read from the file
./test_cuda_topology_3d --pl CLDICE_1_26_1800_3600.f32 26 1800 3600 1e-7 64                     # GPU, same result

# saddles by the standard PL definition (TTK-style), independent of TopoSZp's own rule
g++ -O2 -fopenmp -ISZp/include example/pl_topo_check.cc -o pl_topo_check
./pl_topo_check data.f32 data.f32.SZp.out 1e-3 1800 3600       # add d3 for 3D, --f64, --no-pers; --selftest
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
* **PL mode (`--pl`, `szp_pl_topology.h`).** The compressor classifies points
  with the standard PL definition (as in TTK: Freudenthal triangulation,
  6/14 neighbours, link components, ties broken by index) and stores those
  types; 3D saddle kinds (1-, 2-, degenerate) are stored in 2 bits per saddle
  after the extrema ranks (file footer v2 with a mode flag; axis-mode files are
  unchanged). The decoder restores extrema with the ranked step over the PL
  neighbours, then fixes every point whose PL type differs from the stored one
  (lost saddles, false positives from quantization ties, wrong types) with an
  exhaustive single-point search: one value per gap between the sorted
  neighbour values, nearest first, kept only if no neighbour that had its
  stored type loses it. Rounds repeat until nothing changes (at most 8). Error
  stays within 2 eb, extrema are all kept, CPU and GPU
  (`szp_topo_restore_float_pl`, `szp_cuda_topo_restore_float_pl`) give the
  same output for any thread count. False positives/types are reduced, not
  guaranteed to be zero (the restoration statistics report them).
* **Checking with the standard definition.** `pl_topo_check` classifies points
  as in TTK (Freudenthal triangulation, 6/14 neighbours, connected pieces of the
  lower and upper link, ties broken by index) and reports, for raw SZp and the
  TopoSZp output: kept/FN/FT/FP per PL type (min, max, saddle; 1-, 2- and
  degenerate saddles in 3D), saddles kept at the same point and within one cell,
  saddles kept by persistence (join/split trees), and how PL saddles relate to
  the axis-rule types TopoSZp stores. TopoSZp's no-FP/FT guarantee is for its
  own rule, so PL false positives can appear.
* **Same result everywhere.** Output does not depend on the number of OpenMP
  threads, and the CPU (`szp_topo_restore_float_2d/3d`) and GPU
  (`szp_cuda_topo_restore_float_2d/3d`) restorations run the same algorithm.
* **Stream format.** Slot 0 of each per-thread offset table stores
  `"SZPT" | thread count`. Streams from older versions (slot 0 = 0) are still
  decoded, using thread-count detection.
* `bench.sh` runs the method on CPU (OpenMP) and GPU, 2D and 3D, for absolute
  and relative error bounds, and writes one table (`bench_<time>/results.md`,
  `results.csv`): compression and decompression time, maxima/minima/saddles kept,
  FP, FN and FT before (plain SZp) and after (TopoSZp). Example:
  `ABS2=1e-3 ABS3=1e-7 REL="1e-3 1e-4" ./bench.sh data/CLDHGH_1_1800_3600.dat 1800 3600 data/CLDICE_1_26_1800_3600.f32 26 1800 3600`
* `diagnose.sh` builds everything and runs the thread-count, sanitizer,
  CPU/GPU and topology checks on a 2D and a 3D file.
