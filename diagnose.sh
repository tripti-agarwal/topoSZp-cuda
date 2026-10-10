#!/usr/bin/env bash
# TopoSZp diagnostic run: confirms (or rules out) the suspected bugs before fixing.
#
# Run from the repo root on the GPU machine:
#   ./diagnose.sh <file2d.f32> <rows> <cols>  <file3d.f32> <d1> <d2> <d3>  [absEB=1e-3] [blockSize=64]
#
# Example (SDRBench):
#   ./diagnose.sh CLDHGH_1_1800_3600.f32 1800 3600  Uf48.bin.f32 100 500 500  1e-3 64
#
# Env knobs:  THREADS="1 2 4 8 16"  REPS=3  SKIP_GPU=1  SKIP_SANITIZER=1  TIMEOUT=900
#
# Output: diag_<timestamp>/  with summary.txt (send me this) and one log per run.

set -u
if [ $# -lt 7 ]; then sed -n 2,12p "$0"; exit 1; fi

F2=$(realpath "$1"); R=$2; C=$3
F3=$(realpath "$4"); D1=$5; D2=$6; D3=$7
EB=${8:-1e-3}; BS=${9:-64}
NCPU=$(nproc)
THREADS=${THREADS:-"1 2 4 8 16 32"}
REPS=${REPS:-3}
TIMEOUT=${TIMEOUT:-900}
REPO=$(pwd)
OUT=$REPO/diag_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT/bin" "$OUT/logs" "$OUT/work"
S=$OUT/summary.txt
say() { echo "$*" | tee -a "$S"; }
twoEB=$(python3 -c "print(2*float('$EB'))")

# ---------------------------------------------------------------- 0. environment
say "=== 0. Environment ==="
say "date: $(date)   host: $(hostname)   cores: $NCPU"
say "git:  $(git rev-parse --short HEAD 2>/dev/null) $(git status --porcelain | wc -l) modified files"
say "gcc:  $(g++ --version | head -1)"
command -v nvcc >/dev/null && say "nvcc: $(nvcc --version | tail -1)"
command -v nvidia-smi >/dev/null && say "gpu:  $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
say "2D: $F2 ${R}x${C}   3D: $F3 ${D1}x${D2}x${D3}   eb=$EB  2eb=$twoEB  bs=$BS"
for f in "$F2:$((R*C))" "$F3:$((D1*D2*D3))"; do
  p=${f%:*}; n=${f#*:}; sz=$(stat -c %s "$p")
  [ "$sz" -eq $((n*4)) ] || say "WARNING: $p is $sz bytes, expected $((n*4)) (float32)"
done

# ---------------------------------------------------------------- 1. build
say ""; say "=== 1. Build ==="
SRC="SZp/src/szp.cc SZp/src/szp_CompressionToolkit.cc SZp/src/szp_TypeManager.cc SZp/src/szp_double.cc
     SZp/src/szp_float.cc SZp/src/szp_helper.cc SZp/src/szp_rw.cc SZp/src/szpd_double.cc SZp/src/szpd_float.cc"
CE=example/testfloat_compress_fastmode1_topology.cc
DE=example/testfloat_decompress_fastmode1_topology.cc
B=$OUT/bin
g++ -O2 -fopenmp -ISZp/include -w $SRC $CE -o $B/comp            >"$OUT/logs/build.log" 2>&1 &&
g++ -O2 -fopenmp -ISZp/include -w $SRC $DE -o $B/decomp          >>"$OUT/logs/build.log" 2>&1 &&
say "CPU build OK" || { say "CPU build FAILED (see logs/build.log)"; exit 1; }
if [ "${SKIP_SANITIZER:-0}" != 1 ]; then
  g++ -O1 -g -fopenmp -fsanitize=address -fno-omit-frame-pointer -ISZp/include -w $SRC $CE -o $B/comp_asan   >>"$OUT/logs/build.log" 2>&1
  g++ -O1 -g -fopenmp -fsanitize=address -fno-omit-frame-pointer -ISZp/include -w $SRC $DE -o $B/decomp_asan >>"$OUT/logs/build.log" 2>&1
  say "ASan build: $([ -x $B/comp_asan ] && echo OK || echo FAILED)"
fi
GPU=0
if [ "${SKIP_GPU:-0}" != 1 ] && command -v nvcc >/dev/null; then
  CC_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
  cmake -S . -B "$OUT/build_cuda" -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CC_ARCH:-80}" >"$OUT/logs/cmake.log" 2>&1 &&
  cmake --build "$OUT/build_cuda" -j"$NCPU" >>"$OUT/logs/cmake.log" 2>&1 && GPU=1
  say "CUDA build (sm_$CC_ARCH): $([ $GPU = 1 ] && echo OK || echo 'FAILED (see logs/cmake.log)')"
fi

# ---------------------------------------------------------------- 2. CPU thread sweep
# Confirms bugs 1-3: crashes, non-determinism, and >2eb error appearing only when T>1.
say ""; say "=== 2. CPU thread sweep (2D, compress+decompress with same T; also decompress with T=1) ==="
printf "%-4s %-4s %-6s %-6s %-14s %-9s %-6s %-14s %-9s %s\n" T rep c_rc d_rc maxerr err/eb ok d1_maxerr d1/eb szp_md5 | tee -a "$S"
run_cpu() { # $1=T $2=rep
  local T=$1 r=$2 w=$OUT/work/cpu_T${T}_r${r}.f32 L=$OUT/logs/cpu_T${T}_r${r}
  ln -sf "$F2" "$w"
  OMP_NUM_THREADS=$T timeout $TIMEOUT $B/comp "$w" $BS $EB $R $C > $L.comp.log 2>&1; local crc=$?
  local drc=NA e=NA d1=NA md5=NA
  if [ $crc = 0 ] && [ -f "$w.SZp" ]; then
    md5=$(md5sum "$w.SZp" | cut -c1-8)
    OMP_NUM_THREADS=$T timeout $TIMEOUT $B/decomp "$w.SZp" $((R*C)) $BS $EB $R $C > $L.dec.log 2>&1; drc=$?
    e=$(grep -m1 "Max absolute error" $L.dec.log | awk '{print $NF}')
    OMP_NUM_THREADS=1 timeout $TIMEOUT $B/decomp "$w.SZp" $((R*C)) $BS $EB $R $C > $L.dec1.log 2>&1
    d1=$(grep -m1 "Max absolute error" $L.dec1.log | awk '{print $NF}')
    rm -f "$w.SZp.out"
  fi
  local ratio=$(python3 -c "print('%.2f'%(float('$e')/float('$EB')))" 2>/dev/null || echo NA)
  local r1=$(python3 -c "print('%.2f'%(float('$d1')/float('$EB')))" 2>/dev/null || echo NA)
  local ok=$(python3 -c "print('PASS' if float('$e')<=2*float('$EB')*1.0001 else 'FAIL')" 2>/dev/null || echo CRASH)
  printf "%-4s %-4s %-6s %-6s %-14s %-9s %-6s %-14s %-9s %s\n" $T $r $crc $drc $e $ratio $ok $d1 $r1 $md5 | tee -a "$S"
}
for T in $THREADS; do
  [ "$T" -gt "$NCPU" ] && continue
  for r in $(seq 1 $REPS); do run_cpu $T $r; done
done
say "Read: c_rc=139 -> segfault (bug 1). Different md5 across reps at same T -> non-deterministic (bugs 1/2/5)."
say "      FAIL only for T>1 -> parallel bugs. d1/eb differs from err/eb -> decoder thread-count issue (bug 3)."

# ---------------------------------------------------------------- 3. AddressSanitizer
if [ -x "$B/comp_asan" ]; then
  say ""; say "=== 3. AddressSanitizer (compress+decompress, T=8) ==="
  w=$OUT/work/asan.f32; ln -sf "$F2" "$w"
  for k in 1 2 3; do
    OMP_NUM_THREADS=8 ASAN_OPTIONS=halt_on_error=1 timeout $TIMEOUT $B/comp_asan "$w" $BS $EB $R $C > $OUT/logs/asan_comp_$k.log 2>&1
    hit=$(grep -m1 -o "ERROR: AddressSanitizer: [a-z-]*" $OUT/logs/asan_comp_$k.log)
    loc=$(grep -m1 -o "szp_float.cc:[0-9]*" $OUT/logs/asan_comp_$k.log)
    say "compress try $k: ${hit:-clean} ${loc}"
  done
  if [ -f "$w.SZp" ]; then
    OMP_NUM_THREADS=8 ASAN_OPTIONS=halt_on_error=1 timeout $TIMEOUT $B/decomp_asan "$w.SZp" $((R*C)) $BS $EB $R $C > $OUT/logs/asan_dec.log 2>&1
    hit=$(grep -m1 -o "ERROR: AddressSanitizer: [a-z-]*" $OUT/logs/asan_dec.log)
    say "decompress: ${hit:-clean} $(grep -m1 -o '[a-z_]*\.cc:[0-9]*' $OUT/logs/asan_dec.log)"
  fi
fi

# ---------------------------------------------------------------- 4. GPU tests
if [ $GPU = 1 ]; then
  G=$OUT/build_cuda
  say ""; say "=== 4. GPU tests ==="
  grab() { grep -E "MATCH|DIFFER|Raw max error|Preserved|Lost:|Max error:|Status:|ratio|Match:|RBF restored|mismatch|PASS|FAIL|ERROR" "$1" | sed 's/^/    /' | tee -a "$S"; }
  for k in 1 2; do
    say "-- test_cuda_topology_3d run $k (3D)"
    OMP_NUM_THREADS=$NCPU timeout $TIMEOUT $G/test_cuda_topology_3d "$F3" $D1 $D2 $D3 $EB $BS > $OUT/logs/gpu3d_$k.log 2>&1
    say "    exit=$?"; grab $OUT/logs/gpu3d_$k.log
  done
  say "-- test_cuda_topology (2D)"
  OMP_NUM_THREADS=$NCPU timeout $TIMEOUT $G/test_cuda_topology "$F2" $R $C $EB $BS > $OUT/logs/gpu2d.log 2>&1
  say "    exit=$?"; grab $OUT/logs/gpu2d.log
  for T in 1 4 $NCPU; do
    say "-- CPU->GPU interop, test_cuda_decompress, OMP T=$T"
    OMP_NUM_THREADS=$T timeout $TIMEOUT $G/test_cuda_decompress "$F2" $((R*C)) $EB $BS $R $C > $OUT/logs/interop_cpu2gpu_T$T.log 2>&1
    say "    exit=$?"; grab $OUT/logs/interop_cpu2gpu_T$T.log
    say "-- GPU->CPU interop, test_cuda_compress, OMP T=$T"
    OMP_NUM_THREADS=$T timeout $TIMEOUT $G/test_cuda_compress "$F2" $((R*C)) $EB $BS > $OUT/logs/interop_gpu2cpu_T$T.log 2>&1
    say "    exit=$?"; grab $OUT/logs/interop_gpu2cpu_T$T.log
  done
  if [ "${SKIP_SANITIZER:-0}" != 1 ] && command -v compute-sanitizer >/dev/null; then
    say "-- compute-sanitizer memcheck on test_cuda_topology_3d (slow)"
    timeout $((TIMEOUT*4)) compute-sanitizer --tool memcheck --print-limit 20 \
      $G/test_cuda_topology_3d "$F3" $D1 $D2 $D3 $EB $BS > $OUT/logs/memcheck3d.log 2>&1
    say "    $(grep -m1 'ERROR SUMMARY' $OUT/logs/memcheck3d.log)"
    say "-- compute-sanitizer racecheck"
    timeout $((TIMEOUT*4)) compute-sanitizer --tool racecheck --print-limit 20 \
      $G/test_cuda_topology_3d "$F3" $D1 $D2 $D3 $EB $BS > $OUT/logs/racecheck3d.log 2>&1
    say "    $(grep -m1 -E 'RACECHECK SUMMARY|ERROR SUMMARY' $OUT/logs/racecheck3d.log)"
  fi
fi

rm -f $OUT/work/*.SZp
say ""; say "Done. Send me: $S  (and the logs/ folder if anything FAILs or crashes)"
