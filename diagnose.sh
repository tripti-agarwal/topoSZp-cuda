#!/usr/bin/env bash
# TopoSZp diagnostic run: confirms (or rules out) the suspected bugs before fixing.
#
# Run from the repo root, inside a GPU job (not on a login node):
#   ./diagnose.sh <file2d> <rows> <cols>  <file3d> <d1> <d2> <d3>  [eb=1e-3] [blockSize=64] [eb3=eb]
#
# eb is an absolute error bound for the 2D file; eb3 (optional) is the absolute
# bound for the 3D file. rel:X (X times the file's value range) also works.
# CLDICE values are below 1e-4, so give it its own eb3, e.g. 1e-7.
#
# Example (CESM-ATM from SDRBench):
#   ./diagnose.sh data/CLDHGH_1_1800_3600.dat 1800 3600  data/CLDICE_1_26_1800_3600.f32 26 1800 3600  1e-3 64 1e-7
#
# Topology (FN/FT/FP per type, before and after correction) is reported by
# example/topo_check for the CPU output at every thread count and for the GPU output.
#
# Env knobs:  THREADS="1 2 4 8 16 32"  REPS=3  TIMEOUT=900  SKIP_GPU=1  SKIP_SANITIZER=1
#
# Output: diag_<timestamp>/summary.txt (send this) and diag_<timestamp>/logs/ (one log per run).

set -u
if [ $# -lt 7 ]; then sed -n 2,16p "$0"; exit 1; fi

F2=$(realpath "$1"); R=$2; C=$3
F3=$(realpath "$4"); D1=$5; D2=$6; D3=$7
EB=${8:-1e-3}; BS=${9:-64}; EB3IN=${10:-$EB}
NCPU=$(nproc)
THREADS=${THREADS:-"1 2 4 8 16 32"}
REPS=${REPS:-3}
TIMEOUT=${TIMEOUT:-900}
T3=$((TIMEOUT * 3))                       # 3D runs get a longer limit
REPO=$(pwd)
OUT=$REPO/diag_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT/bin" "$OUT/logs" "$OUT/work"
S=$OUT/summary.txt
B=$OUT/bin
say() { echo "$*" | tee -a "$S"; }
rcnote() { case $1 in 0) echo "0";; 124) echo "124 (timeout)";; 134) echo "134 (abort)";;
                      136) echo "136 (FP exception)";; 139) echo "139 (segfault)";; *) echo "$1";; esac; }

# ---------------------------------------------------------------- 0. environment
say "=== 0. Environment ==="
say "date: $(date)   host: $(hostname)   cores: $NCPU"
say "git:  $(git rev-parse --short HEAD 2>/dev/null) on $(git branch --show-current 2>/dev/null)"
say "gcc:  $(g++ --version | head -1)"
say "cmake: $(cmake --version 2>/dev/null | head -1)"
command -v nvcc >/dev/null && say "nvcc: $(nvcc --version | tail -1)"
GPUVIS=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  GPUVIS=1; say "gpu:  $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
else
  say "gpu:  none visible (on a login node? GPU tests will be skipped; run inside a GPU job)"
fi

# ---------------------------------------------------------------- 1. build
say ""; say "=== 1. Build ==="
SRC="SZp/src/szp.cc SZp/src/szp_CompressionToolkit.cc SZp/src/szp_TypeManager.cc SZp/src/szp_double.cc
     SZp/src/szp_float.cc SZp/src/szp_helper.cc SZp/src/szp_rw.cc SZp/src/szpd_double.cc SZp/src/szpd_float.cc"
CE=example/testfloat_compress_fastmode1_topology.cc
DE=example/testfloat_decompress_fastmode1_topology.cc
cat > "$B/fieldstats.cc" <<'EOF'
// fieldstats <float32 file> [rel]  ->  n min max range zero% nan abs_eb
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
  std::vector<float> b(1 << 20); size_t n = 0, z = 0, nan = 0, k;
  float mn = INFINITY, mx = -INFINITY;
  while ((k = fread(b.data(), 4, b.size(), f)) > 0) {
    for (size_t i = 0; i < k; i++) { float v = b[i];
      if (std::isnan(v)) { nan++; continue; } if (v == 0.0f) z++;
      if (v < mn) mn = v; if (v > mx) mx = v; }
    n += k; }
  fclose(f);
  float range = mx - mn; if (!(range > 0)) range = 1.1920929e-07f;   // same as szp_rel_to_abs_float
  float rel = argc > 2 ? strtof(argv[2], 0) : 0;
  printf("%zu %.9g %.9g %.9g %.2f %zu %.9g\n", n, mn, mx, range, n ? 100.0 * z / n : 0.0, nan, rel > 0 ? rel * range : 0.0f);
}
EOF
g++ -O2 "$B/fieldstats.cc" -o "$B/fieldstats"                     >"$OUT/logs/build.log" 2>&1 &&
g++ -O2 -fopenmp -ISZp/include example/topo_check.cc -o "$B/topo_check"                   >>"$OUT/logs/build.log" 2>&1 &&
g++ -O2 -fopenmp -ISZp/include -w $SRC $CE -o "$B/comp"            >>"$OUT/logs/build.log" 2>&1 &&
g++ -O2 -fopenmp -ISZp/include -w $SRC $DE -o "$B/decomp"          >>"$OUT/logs/build.log" 2>&1 &&
say "CPU build OK" || { say "CPU build FAILED (see logs/build.log)"; exit 1; }
if [ "${SKIP_SANITIZER:-0}" != 1 ]; then
  g++ -O1 -g -fopenmp -fsanitize=address -fno-omit-frame-pointer -ISZp/include -w $SRC $CE -o "$B/comp_asan"   >>"$OUT/logs/build.log" 2>&1
  g++ -O1 -g -fopenmp -fsanitize=address -fno-omit-frame-pointer -ISZp/include -w $SRC $DE -o "$B/decomp_asan" >>"$OUT/logs/build.log" 2>&1
  say "ASan build: $([ -x "$B/comp_asan" ] && echo OK || echo FAILED)"
fi
GPU=0
if [ "${SKIP_GPU:-0}" != 1 ] && command -v nvcc >/dev/null; then
  CC_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
  [[ "$CC_ARCH" =~ ^[0-9]+$ ]] || CC_ARCH=80
  # Use the same GNU compilers as the CPU build (a loaded module may set CXX to
  # another compiler, e.g. an old Intel icpc that lacks C++17/OpenMP for CMake).
  HOSTCC=${HOSTCC:-gcc}; HOSTCXX=${HOSTCXX:-g++}
  say "CUDA host compilers: $(command -v $HOSTCC) / $(command -v $HOSTCXX)"
  cmake -S . -B "$OUT/build_cuda" -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$HOSTCC" -DCMAKE_CXX_COMPILER="$HOSTCXX" \
        -DCMAKE_CUDA_HOST_COMPILER="$HOSTCXX" \
        -DCMAKE_CUDA_ARCHITECTURES="$CC_ARCH" >"$OUT/logs/cmake.log" 2>&1 &&
  cmake --build "$OUT/build_cuda" -j"$NCPU" --target test_cuda_topology test_cuda_topology_3d \
        test_cuda_compress test_cuda_decompress >>"$OUT/logs/cmake.log" 2>&1 && GPU=1
  say "CUDA build (sm_$CC_ARCH): $([ $GPU = 1 ] && echo OK || echo 'FAILED (see logs/cmake.log)')"
  [ $GPU = 1 ] || grep -m8 -i -A3 "error" "$OUT/logs/cmake.log" | sed 's/^/    /' | tee -a "$S"
fi

# ---------------------------------------------------------------- 1b. inputs and error bounds
say ""; say "=== 1b. Inputs ==="
relv() { case "$1" in rel:*) echo "${1#rel:}";; *) echo 0;; esac; }
read -r N2 MIN2 MAX2 RNG2 Z2 NAN2 REL2 < <("$B/fieldstats" "$F2" "$(relv "$EB")")
read -r N3 MIN3 MAX3 RNG3 Z3 NAN3 REL3 < <("$B/fieldstats" "$F3" "$(relv "$EB3IN")")
if [ "$(relv "$EB")" = 0 ]; then ABS2=$EB; else ABS2=$REL2; fi
if [ "$(relv "$EB3IN")" = 0 ]; then ABS3=$EB3IN; else ABS3=$REL3; fi
say "2D: $F2  ${R}x${C}  min=$MIN2 max=$MAX2 zeros=${Z2}% NaN=$NAN2  -> abs eb=$ABS2"
say "3D: $F3  ${D1}x${D2}x${D3}  min=$MIN3 max=$MAX3 zeros=${Z3}% NaN=$NAN3  -> abs eb=$ABS3"
say "eb (2D)=$EB  eb (3D)=$EB3IN  blockSize=$BS"
[ "$N2" -eq $((R*C)) ]      || say "WARNING: 2D file has $N2 floats, expected $((R*C))"
[ "$N3" -eq $((D1*D2*D3)) ] || say "WARNING: 3D file has $N3 floats, expected $((D1*D2*D3))"
awk -v e="$ABS2" -v r="$RNG2" 'BEGIN{ if (e >= r) exit 1 }' || say "WARNING: 2D eb >= value range; use rel:"
awk -v e="$ABS3" -v r="$RNG3" 'BEGIN{ if (e >= r) exit 1 }' || say "WARNING: 3D eb >= value range; use rel:"

# ---------------------------------------------------------------- 2. CPU thread sweep
# Confirms bugs 1-3: crashes, non-determinism, and >2eb error appearing only when T>1.
say ""; say "=== 2. CPU thread sweep on 2D (compress+decompress with T threads; d1 = same file decoded with 1 thread) ==="
printf "%-4s %-4s %-5s %-5s %-14s %-8s %-6s %-14s %-8s %s\n" T rep c_rc d_rc maxerr err/eb ok d1_maxerr d1/eb szp_md5 | tee -a "$S"
ratio() { awk -v e="$1" -v b="$ABS2" 'BEGIN{ if (e == "NA" || e == "") print "NA"; else printf "%.2f", e / b }'; }
verdict() { awk -v e="$1" -v b="$ABS2" 'BEGIN{ if (e == "NA" || e == "") print "CRASH"; else print (e <= 2 * b * 1.0001) ? "PASS" : "FAIL" }'; }
run_cpu() { # $1=T $2=rep
  local T=$1 r=$2 w=$OUT/work/cpu_T${1}_r${2}.f32 L=$OUT/logs/cpu_T${1}_r${2}
  ln -sf "$F2" "$w"
  OMP_NUM_THREADS=$T timeout "$TIMEOUT" "$B/comp" "$w" "$BS" "$ABS2" "$R" "$C" > "$L.comp.log" 2>&1; local crc=$?
  local drc=NA e=NA d1=NA md5=NA
  if [ $crc = 0 ] && [ -f "$w.SZp" ]; then
    md5=$(md5sum "$w.SZp" | cut -c1-8)
    OMP_NUM_THREADS=$T timeout "$TIMEOUT" "$B/decomp" "$w.SZp" $((R*C)) "$BS" "$ABS2" "$R" "$C" > "$L.dec.log" 2>&1; drc=$?
    e=$(grep -m1 "Max absolute error" "$L.dec.log" | awk '{print $NF}'); e=${e:-NA}
    OMP_NUM_THREADS=1 timeout "$TIMEOUT" "$B/decomp" "$w.SZp" $((R*C)) "$BS" "$ABS2" "$R" "$C" > "$L.dec1.log" 2>&1
    d1=$(grep -m1 "Max absolute error" "$L.dec1.log" | awk '{print $NF}'); d1=${d1:-NA}
    if [ "$r" = 1 ]; then mv -f "$w.SZp.out" "$OUT/work/cpu_T${T}.out"; else rm -f "$w.SZp.out"; fi
  fi
  printf "%-4s %-4s %-5s %-5s %-14s %-8s %-6s %-14s %-8s %s\n" "$T" "$r" "$crc" "$drc" "$e" "$(ratio "$e")" \
         "$(verdict "$e")" "$d1" "$(ratio "$d1")" "$md5" | tee -a "$S"
}
for T in $THREADS; do
  [ "$T" -gt "$NCPU" ] && continue
  for r in $(seq 1 "$REPS"); do run_cpu "$T" "$r"; done
done
say ""; say "--- Topology of the CPU output (2D, abs eb $ABS2) ---"
first=""
for T in $THREADS; do
  o=$OUT/work/cpu_T${T}.out; [ -f "$o" ] || continue
  "$B/topo_check" "$F2" "$o" "$ABS2" "$R" "$C" > "$OUT/logs/topo_cpu_T$T.log" 2>&1
  if [ -z "$first" ]; then first=$T; sed 's/^/    /' "$OUT/logs/topo_cpu_T$T.log" | grep -v '^    TOPO' | tee -a "$S"; fi
  say "    T=$T: $(grep ^TOPO "$OUT/logs/topo_cpu_T$T.log")  out_md5=$(md5sum < "$o" | cut -c1-8)"
done
say "Read: the decompressed output (out_md5) should be identical for every T."
say ""
say "Read: rc 139 = segfault (bug 1), 124 = timeout. Different md5 across reps at the same T = non-deterministic output (bugs 1/2/5)."
say "      FAIL only when T>1 = parallel bugs. d1/eb different from err/eb = decoder thread-count problem (bug 3)."

# ---------------------------------------------------------------- 3. AddressSanitizer
if [ -x "$B/comp_asan" ]; then
  say ""; say "=== 3. AddressSanitizer on 2D (T=8) ==="
  w=$OUT/work/asan.f32; ln -sf "$F2" "$w"
  for k in 1 2 3; do
    OMP_NUM_THREADS=8 ASAN_OPTIONS=halt_on_error=1 timeout "$TIMEOUT" "$B/comp_asan" "$w" "$BS" "$ABS2" "$R" "$C" > "$OUT/logs/asan_comp_$k.log" 2>&1
    hit=$(grep -m1 -o "ERROR: AddressSanitizer: [a-z-]*" "$OUT/logs/asan_comp_$k.log")
    loc=$(grep -m1 -o "szp[a-z_]*\.cc:[0-9]*" "$OUT/logs/asan_comp_$k.log")
    say "compress try $k: ${hit:-clean} ${loc}"
  done
  if [ -f "$w.SZp" ]; then
    OMP_NUM_THREADS=8 ASAN_OPTIONS=halt_on_error=1 timeout "$TIMEOUT" "$B/decomp_asan" "$w.SZp" $((R*C)) "$BS" "$ABS2" "$R" "$C" > "$OUT/logs/asan_dec.log" 2>&1
    hit=$(grep -m1 -o "ERROR: AddressSanitizer: [a-z-]*" "$OUT/logs/asan_dec.log")
    say "decompress: ${hit:-clean} $(grep -m1 -o '[a-z_]*\.cc:[0-9]*' "$OUT/logs/asan_dec.log")"
  fi
fi

# ---------------------------------------------------------------- 4. GPU tests
if [ $GPU = 1 ] && [ $GPUVIS = 1 ]; then
  G=$OUT/build_cuda
  say ""; say "=== 4. GPU tests ==="
  KEYS="Found|MATCH|DIFFER|Raw max error|Raw CP|Compressed:|Preserved|Lost:|Non-zero|Max error:|Status:|RBF restored|mismatch|PASS|FAIL|ERROR|Error"
  grab() { grep -E "$KEYS" "$1" | sed 's/^/    /' | tee -a "$S"; }
  norm() { grep -E "$KEYS" "$1" | sed -E 's/ in [0-9.]+ ms//g; s/[0-9.]+ ms//g'; }
  for k in 1 2; do
    say "-- test_cuda_topology_3d, run $k (full 3D volume)"
    TOPOSZP_DUMP=$OUT/work/gpu3d_$k.out OMP_NUM_THREADS=$NCPU timeout "$T3" "$G/test_cuda_topology_3d" "$F3" "$D1" "$D2" "$D3" "$ABS3" "$BS" > "$OUT/logs/gpu3d_$k.log" 2>&1
    say "    exit=$(rcnote $?)"; grab "$OUT/logs/gpu3d_$k.log"
  done
  if [ -f "$OUT/work/gpu3d_1.out" ]; then
    say "    --- Topology of the GPU 3D output (abs eb $ABS3) ---"
    "$B/topo_check" "$F3" "$OUT/work/gpu3d_1.out" "$ABS3" "$D1" "$D2" "$D3" > "$OUT/logs/topo_gpu3d.log" 2>&1
    sed 's/^/    /' "$OUT/logs/topo_gpu3d.log" | tee -a "$S"
    cmp -s "$OUT/work/gpu3d_1.out" "$OUT/work/gpu3d_2.out" && say "    3D outputs of runs 1 and 2 are identical" \
      || say "    3D outputs of runs 1 and 2 DIFFER"
  fi
  if diff -q <(norm "$OUT/logs/gpu3d_1.log") <(norm "$OUT/logs/gpu3d_2.log") >/dev/null; then
    say "    3D runs 1 and 2 agree"; else say "    3D runs 1 and 2 DIFFER (non-deterministic GPU path)"; fi
  say "-- test_cuda_topology (2D)"
  TOPOSZP_DUMP=$OUT/work/gpu2d.out OMP_NUM_THREADS=$NCPU timeout "$TIMEOUT" "$G/test_cuda_topology" "$F2" "$R" "$C" "$ABS2" "$BS" > "$OUT/logs/gpu2d.log" 2>&1
  say "    exit=$(rcnote $?)"; grab "$OUT/logs/gpu2d.log"
  if [ -f "$OUT/work/gpu2d.out" ]; then
    say "    --- Topology of the GPU 2D output (abs eb $ABS2) ---"
    "$B/topo_check" "$F2" "$OUT/work/gpu2d.out" "$ABS2" "$R" "$C" > "$OUT/logs/topo_gpu2d.log" 2>&1
    sed 's/^/    /' "$OUT/logs/topo_gpu2d.log" | tee -a "$S"
    f1=$OUT/work/cpu_T1.out
    [ -f "$f1" ] && { cmp -s "$f1" "$OUT/work/gpu2d.out" && say "    GPU 2D output is identical to the CPU output" \
      || say "    GPU 2D output differs from the CPU output (compare the two topology tables)"; }
  fi
  for T in 1 4 "$NCPU"; do
    say "-- CPU compress (T=$T) -> GPU decompress: test_cuda_decompress"
    OMP_NUM_THREADS=$T timeout "$TIMEOUT" "$G/test_cuda_decompress" "$F2" $((R*C)) "$ABS2" "$BS" "$R" "$C" > "$OUT/logs/interop_cpu2gpu_T$T.log" 2>&1
    say "    exit=$(rcnote $?)"; grab "$OUT/logs/interop_cpu2gpu_T$T.log"
    say "-- GPU compress -> CPU decompress (T=$T): test_cuda_compress"
    OMP_NUM_THREADS=$T timeout "$TIMEOUT" "$G/test_cuda_compress" "$F2" $((R*C)) "$ABS2" "$BS" > "$OUT/logs/interop_gpu2cpu_T$T.log" 2>&1
    say "    exit=$(rcnote $?)"; grab "$OUT/logs/interop_gpu2cpu_T$T.log"
  done
  if [ "${SKIP_SANITIZER:-0}" != 1 ] && command -v compute-sanitizer >/dev/null; then
    # The sanitizer is 10-100x slower, so check a 3-level slab from the middle of the volume.
    CROP=$F3; CD1=$D1
    if [ "$D1" -gt 3 ]; then
      CROP=$OUT/work/crop3d.f32; CD1=3
      dd if="$F3" of="$CROP" bs=$((D2*D3*4)) skip=$((D1/2 - 1)) count=3 status=none
    fi
    for tool in memcheck racecheck; do
      say "-- compute-sanitizer $tool on 3D slab ${CD1}x${D2}x${D3} (slow)"
      timeout $((TIMEOUT*4)) compute-sanitizer --tool $tool --print-limit 20 \
        "$G/test_cuda_topology_3d" "$CROP" "$CD1" "$D2" "$D3" "$ABS3" "$BS" > "$OUT/logs/${tool}3d.log" 2>&1
      say "    exit=$(rcnote $?)  $(grep -m1 -E 'ERROR SUMMARY|RACECHECK SUMMARY' "$OUT/logs/${tool}3d.log")"
    done
  fi
elif [ "${SKIP_GPU:-0}" != 1 ]; then
  say ""; say "=== 4. GPU tests skipped (CUDA build OK: $GPU, GPU visible: $GPUVIS) ==="
fi

rm -f "$OUT"/work/*.SZp "$OUT"/work/*.out "$OUT"/work/crop3d.f32
say ""; say "Done. Send me: $S  (and the logs/ folder if anything FAILs or crashes)"
