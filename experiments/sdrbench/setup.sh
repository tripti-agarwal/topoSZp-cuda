#!/usr/bin/env bash
# Build everything the experiments need into one folder.
#
#   ./setup.sh [BIN_DIR]            (default: experiments/sdrbench/bin)
#   NO_GPU=1       skip the CUDA programs
#   NO_BASELINES=1 skip SZ3 / SZ2 / ZFP
#   TTHRESH=1      also build TTHRESH (not pointwise error-bounded; off by default)
#
# Run from anywhere inside the repository, on the GPU node (so nvcc and the
# GPU architecture are detected). On Swing: module load nvhpc (and cmake).
#
# Built:
#   toposzp_comp, toposzp_decomp   TopoSZp+ OpenMP compressor / decompressor
#   plain_szp                      plain SZp (same quantizer, no topology)
#   topo_check                     axis-rule critical points: FN/FP/FT before/after, error
#   pl_topo_check                  piecewise-linear (TTK) critical points, persistence
#   fstats                         value range / max |value| of a float32 file
#   test_cuda_topology[_3d]        TopoSZp+ CUDA pipeline (if a GPU is present)
#   sz3, sz (SZ2), zfp [, tthresh] baseline compressors

set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
BIN=${1:-$HERE/bin}; mkdir -p "$BIN"; BIN=$(cd "$BIN" && pwd)
BLD=$BIN/.build; mkdir -p "$BLD"
LOG=$BIN/setup.log; : > "$LOG"
NCPU=$(nproc)
CXX=${CXX_HOST:-g++}
say() { echo "$*" | tee -a "$LOG"; }
fail=0
run() { "$@" >> "$LOG" 2>&1 || { say "   FAILED: $*  (see $LOG)"; fail=1; return 1; }; }

cd "$REPO"
say "Repository: $REPO  ($(git rev-parse --short HEAD 2>/dev/null) on $(git branch --show-current 2>/dev/null))"
say "Compiler: $($CXX --version | head -1)"

SRC="SZp/src/szp.cc SZp/src/szp_CompressionToolkit.cc SZp/src/szp_TypeManager.cc SZp/src/szp_double.cc
     SZp/src/szp_float.cc SZp/src/szp_helper.cc SZp/src/szp_rw.cc SZp/src/szpd_double.cc SZp/src/szpd_float.cc
     SZp/src/szp_topology_3d.cc SZp/src/szp_topo_restore.cc"
FL="-O3 -fopenmp -ISZp/include -w"

say "== TopoSZp+ CPU tools"
run $CXX $FL $SRC example/testfloat_compress_fastmode1_topology.cc   -o "$BIN/toposzp_comp"
run $CXX $FL $SRC example/testfloat_decompress_fastmode1_topology.cc -o "$BIN/toposzp_decomp"
run $CXX $FL $SRC "$HERE/tools/plain_szp.cc"                           -o "$BIN/plain_szp"
run $CXX -O3 -fopenmp -ISZp/include example/topo_check.cc            -o "$BIN/topo_check"
run $CXX -O3 -fopenmp -ISZp/include example/pl_topo_check.cc         -o "$BIN/pl_topo_check"
run $CXX -O3 "$HERE/tools/fstats.cc"                                  -o "$BIN/fstats"
"$BIN/pl_topo_check" --selftest >> "$LOG" 2>&1 && say "   pl_topo_check selftest passed" || { say "   pl_topo_check selftest FAILED"; fail=1; }

if [ "${NO_GPU:-0}" != 1 ] && command -v nvcc >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
  [[ "$ARCH" =~ ^[0-9]+$ ]] || ARCH=80
  say "== CUDA programs (sm_$ARCH, $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))"
  # GNU host compilers: a loaded module may set CXX to another compiler.
  run cmake -S . -B "$BLD/cuda" -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=${CC_HOST:-gcc} -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CUDA_HOST_COMPILER=$CXX \
      -DCMAKE_CUDA_ARCHITECTURES="$ARCH" &&
  run cmake --build "$BLD/cuda" -j"$NCPU" --target test_cuda_topology test_cuda_topology_3d &&
  { cp "$BLD/cuda"/test_cuda_topology "$BLD/cuda"/test_cuda_topology_3d "$BIN/" 2>/dev/null ||
    find "$BLD/cuda" -name 'test_cuda_topology*' -type f -executable -exec cp {} "$BIN/" \; ; }
  [ -x "$BIN/test_cuda_topology" ] && say "   OK" || { say "   CUDA build FAILED (see $LOG)"; fail=1; }
else
  say "== CUDA programs: skipped (NO_GPU=1, or no nvcc / no visible GPU)"
fi

if [ "${NO_BASELINES:-0}" != 1 ]; then
  say "== Baseline compressors (sources in $BLD/src)"
  mkdir -p "$BLD/src"
  getsrc() {  # getsrc <dir> <git url> <ref>
    [ -d "$BLD/src/$1" ] || run git clone -q "$2" "$BLD/src/$1" || return 1
    if [ -n "$3" ]; then (cd "$BLD/src/$1" && git fetch -q --tags && git checkout -q "$3") >> "$LOG" 2>&1 || say "   $1: ref $3 not found, using $(cd "$BLD/src/$1" && git describe --tags --always)"; fi
    say "   $1 at $(cd "$BLD/src/$1" && git describe --tags --always)"
  }
  cmk() {  # cmk <dir> [cmake args]: configure, build, install into $BLD/inst
    local d=$1; shift
    run cmake -S "$BLD/src/$d" -B "$BLD/src/$d/build" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$BLD/inst" -DCMAKE_C_COMPILER=${CC_HOST:-gcc} -DCMAKE_CXX_COMPILER=$CXX "$@" &&
    run cmake --build "$BLD/src/$d/build" -j"$NCPU" &&
    run cmake --install "$BLD/src/$d/build"
  }
  if [ ! -x "$BIN/sz3" ]; then
    getsrc SZ3 https://github.com/szcompressor/SZ3.git "${SZ3_REF:-}" && cmk SZ3 &&
    cp "$BLD/inst/bin/sz3" "$BIN/" 2>/dev/null
  fi
  if [ ! -x "$BIN/sz" ]; then
    # SZ2 (deprecated upstream in 2025, moved from szcompressor/SZ to szcompressor/sz2)
    getsrc sz2 https://github.com/szcompressor/sz2.git "${SZ_REF:-}" && cmk sz2 -DBUILD_SZ_EXAMPLES=ON &&
    { cp "$BLD/inst/bin/sz" "$BIN/" 2>/dev/null || find "$BLD/src/sz2/build" -name sz -type f -executable -exec cp {} "$BIN/" \; ; }
  fi
  if [ ! -x "$BIN/zfp" ]; then
    getsrc zfp https://github.com/LLNL/zfp.git "${ZFP_REF:-}" && cmk zfp -DBUILD_UTILITIES=ON -DBUILD_TESTING=OFF &&
    { cp "$BLD/inst/bin/zfp" "$BIN/" 2>/dev/null || find "$BLD/src/zfp/build" -name zfp -type f -executable -exec cp {} "$BIN/" \; ; }
  fi
  if [ "${TTHRESH:-0}" = 1 ] && [ ! -x "$BIN/tthresh" ]; then
    getsrc tthresh https://github.com/rballester/tthresh.git "" && cmk tthresh &&
    find "$BLD/src/tthresh/build" -name tthresh -type f -executable -exec cp {} "$BIN/" \;
  fi
  for c in sz3 sz zfp tthresh; do [ -x "$BIN/$c" ] && say "   $c: OK"; done
  # SZ2 and SZ3 binaries need their shared libraries at run time
  echo "export LD_LIBRARY_PATH=$BLD/inst/lib:$BLD/inst/lib64:\${LD_LIBRARY_PATH:-}" > "$BIN/env.sh"
fi

say ""
say "Tools in $BIN:"; ls "$BIN" | grep -v '^\.' | sed 's/^/   /' | tee -a "$LOG"
[ $fail = 0 ] && say "Setup finished." || say "Setup finished with errors (see $LOG)."
exit $fail
