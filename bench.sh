#!/usr/bin/env bash
# TopoSZp benchmark: compression time, decompression time, and critical points
# (maxima, minima, saddles kept; FP, FN, FT) before and after topology
# correction, for absolute and relative error bounds, 2D and 3D, CPU (OpenMP)
# and GPU (CUDA).
#
# Run from the repo root, inside a GPU job:
#   ./bench.sh <file2d> <rows> <cols>  <file3d> <d1> <d2> <d3>
# Example (CESM-ATM from SDRBench):
#   ./bench.sh data/CLDHGH_1_1800_3600.dat 1800 3600  data/CLDICE_1_26_1800_3600.f32 26 1800 3600
#
# Env knobs (defaults):
#   ABS2="1e-3"     absolute bounds for the 2D file (space-separated list)
#   ABS3="1e-7"     absolute bounds for the 3D file
#   REL="1e-3 1e-4" relative bounds (fraction of each file's value range), both files
#   THREADS="<nproc>" CPU thread counts      REPS=1 runs per case (times: minimum)
#   BS=64 block size   SKIP_GPU=1 CPU only   SKIP_CPU=1 GPU only   TIMEOUT=1800 seconds per run
#
# Definitions (paper's method: 4 neighbours in 2D, 6 in 3D, strict comparisons;
# saddle = above both neighbours along one axis and below both along another):
#   before = plain SZp quantization (no topology correction), after = TopoSZp output.
#   kept = same type at the same point; FN = critical point that became regular;
#   FT = critical point with another type; FP = new critical point.
#   comp time = critical points + ranks + compression; decomp time = decoding +
#   topology restoration (CPU: timers of the example programs; GPU: test_cuda_topology*).
#
# Output: bench_<timestamp>/results.md (table), results.csv, logs/.

set -u
if [ $# -lt 7 ]; then sed -n 2,27p "$0"; exit 1; fi
F2=$(realpath "$1"); R=$2; C=$3
F3=$(realpath "$4"); D1=$5; D2=$6; D3=$7
NCPU=$(nproc)
ABS2=${ABS2:-1e-3}; ABS3=${ABS3:-1e-7}; REL=${REL:-"1e-3 1e-4"}
THREADS=${THREADS:-$NCPU}; REPS=${REPS:-1}; BS=${BS:-64}; TIMEOUT=${TIMEOUT:-1800}
REPO=$(pwd)
OUT=$REPO/bench_$(date +%Y%m%d_%H%M%S); B=$OUT/bin; L=$OUT/logs; W=$OUT/work
mkdir -p "$B" "$L" "$W"
MD=$OUT/results.md; CSV=$OUT/results.csv
say() { echo "$*" | tee -a "$OUT/run.log"; }

say "TopoSZp benchmark  $(date)  host $(hostname)  cores $NCPU  git $(git rev-parse --short HEAD 2>/dev/null) ($(git branch --show-current 2>/dev/null))"

# ---------------------------------------------------------------- build
SRC="SZp/src/szp.cc SZp/src/szp_CompressionToolkit.cc SZp/src/szp_TypeManager.cc SZp/src/szp_double.cc
     SZp/src/szp_float.cc SZp/src/szp_helper.cc SZp/src/szp_rw.cc SZp/src/szpd_double.cc SZp/src/szpd_float.cc
     SZp/src/szp_topology_3d.cc SZp/src/szp_topo_restore.cc"
cat > "$B/range.cc" <<'EOF'
// range <float32 file>  ->  value range (max - min) as float, as szp_rel_to_abs_float
#include <cstdio>
#include <cmath>
#include <vector>
int main(int, char **argv) {
  FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
  std::vector<float> b(1 << 20); size_t k; float mn = INFINITY, mx = -INFINITY;
  while ((k = fread(b.data(), 4, b.size(), f)) > 0)
    for (size_t i = 0; i < k; i++) if (b[i] == b[i]) { if (b[i] < mn) mn = b[i]; if (b[i] > mx) mx = b[i]; }
  fclose(f);
  float r = mx - mn; if (!(r > 0)) r = 1.1920929e-07f;
  printf("%.9g\n", r);
}
EOF
CPU=0
if [ "${SKIP_CPU:-0}" != 1 ]; then
  g++ -O2 "$B/range.cc" -o "$B/range" > "$L/build.log" 2>&1 &&
  g++ -O2 -fopenmp -ISZp/include example/topo_check.cc -o "$B/topo_check" >> "$L/build.log" 2>&1 &&
  g++ -O2 -fopenmp -ISZp/include -w $SRC example/testfloat_compress_fastmode1_topology.cc -o "$B/comp" >> "$L/build.log" 2>&1 &&
  g++ -O2 -fopenmp -ISZp/include -w $SRC example/testfloat_decompress_fastmode1_topology.cc -o "$B/decomp" >> "$L/build.log" 2>&1 &&
  CPU=1
else
  g++ -O2 "$B/range.cc" -o "$B/range" > "$L/build.log" 2>&1 &&
  g++ -O2 -fopenmp -ISZp/include example/topo_check.cc -o "$B/topo_check" >> "$L/build.log" 2>&1
fi
[ -x "$B/topo_check" ] || { say "build FAILED (see $L/build.log)"; exit 1; }
say "CPU build: $([ $CPU = 1 ] && echo OK || echo skipped)"
GPU=0
if [ "${SKIP_GPU:-0}" != 1 ] && command -v nvcc >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
  [[ "$ARCH" =~ ^[0-9]+$ ]] || ARCH=80
  # GNU host compilers (a loaded module may set CXX to an old Intel compiler).
  HOSTCC=${HOSTCC:-gcc}; HOSTCXX=${HOSTCXX:-g++}
  cmake -S . -B "$OUT/build_cuda" -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$HOSTCC" -DCMAKE_CXX_COMPILER="$HOSTCXX" -DCMAKE_CUDA_HOST_COMPILER="$HOSTCXX" \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCH" > "$L/cmake.log" 2>&1 &&
  cmake --build "$OUT/build_cuda" -j"$NCPU" --target test_cuda_topology test_cuda_topology_3d >> "$L/cmake.log" 2>&1 && GPU=1
  say "GPU build (sm_$ARCH, $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)): $([ $GPU = 1 ] && echo OK || echo "FAILED (see $L/cmake.log)")"
else
  say "GPU: skipped (no nvcc / no visible GPU / SKIP_GPU=1)"
fi

# ---------------------------------------------------------------- cases
RNG2=$("$B/range" "$F2"); RNG3=$("$B/range" "$F3")
CASES=()   # "dim|mode|eb as given|absolute eb"
for e in $ABS2; do CASES+=("2|abs|$e|$e"); done
for e in $REL;  do CASES+=("2|rel|$e|$(awk -v r="$e" -v g="$RNG2" 'BEGIN{printf "%.9g", r*g}')"); done
for e in $ABS3; do CASES+=("3|abs|$e|$e"); done
for e in $REL;  do CASES+=("3|rel|$e|$(awk -v r="$e" -v g="$RNG3" 'BEGIN{printf "%.9g", r*g}')"); done
say "2D: $F2 (${R}x${C}, value range $RNG2)"
say "3D: $F3 (${D1}x${D2}x${D3}, value range $RNG3)"

echo "data,dims,platform,eb_mode,eb,abs_eb,comp_s,decomp_s,max_orig,max_before,max_after,min_orig,min_before,min_after,sad_orig,sad_before,sad_after,FP_before,FP_after,FN_before,FN_after,FT_before,FT_after,err_eb_before,err_eb_after" > "$CSV"
WARN=()

minf() { awk -v a="$1" -v b="$2" 'BEGIN{ if (a == "") print b; else if (b == "" || a + 0 <= b + 0) print a; else print b }'; }

# record <label> <dim> <platform> <mode> <eb> <abs> <comp> <decomp> <topo_check log>
record() {
  local lab=$1 dim=$2 plat=$3 mode=$4 eb=$5 abs=$6 tc=$7 td=$8 log=$9
  local b a
  b=$(grep '^TOPOROW before' "$log"); a=$(grep '^TOPOROW after' "$log")
  if [ -z "$b" ] || [ -z "$a" ]; then WARN+=("$lab $plat $mode $eb: no topology result (see $log)"); return; fi
  read -r _ _ mo mb no nb so sb fpb fnb ftb eeb <<< "$b"
  read -r _ _ _ ma _ na _ sa fpa fna fta eea <<< "$a"
  echo "$lab,$( [ $dim = 2 ] && echo "${R}x${C}" || echo "${D1}x${D2}x${D3}"),$plat,$mode,$eb,$abs,$tc,$td,$mo,$mb,$ma,$no,$nb,$na,$so,$sb,$sa,$fpb,$fpa,$fnb,$fna,$ftb,$fta,$eeb,$eea" >> "$CSV"
  awk -v e="$eea" 'BEGIN{exit !(e > 2.0)}' && WARN+=("$lab $plat $mode $eb: max error $eea eb > 2 eb")
  awk -v e="$eeb" 'BEGIN{exit !(e > 1.0)}' && WARN+=("$lab $plat $mode $eb: plain quantization error $eeb eb > 1 eb")
  say "  $lab $plat $mode $eb: comp ${tc}s decomp ${td}s  max $mb->$ma/$mo  min $nb->$na/$no  sad $sb->$sa/$so  FP $fpb->$fpa  FN $fnb->$fna  FT $ftb->$fta"
}

for cs in "${CASES[@]}"; do
  IFS='|' read -r dim mode eb abs <<< "$cs"
  if [ "$dim" = 2 ]; then F=$F2; DIMS="$R $C"; N=$((R*C)); lab=$(basename "$F2")
  else F=$F3; DIMS="$D1 $D2 $D3"; N=$((D1*D2*D3)); lab=$(basename "$F3"); fi
  tag=${dim}d_${mode}_${eb}

  if [ $CPU = 1 ]; then
    for T in $THREADS; do
      tc=""; td=""; w=$W/cpu_${tag}_T$T.f32; ln -sf "$F" "$w"
      for r in $(seq 1 "$REPS"); do
        OMP_NUM_THREADS=$T timeout "$TIMEOUT" "$B/comp" "$w" "$BS" "$abs" $DIMS > "$L/cpu_${tag}_T${T}_c$r.log" 2>&1 || { WARN+=("$lab CPU T=$T $mode $eb: compress rc $?"); break; }
        OMP_NUM_THREADS=$T timeout "$TIMEOUT" "$B/decomp" "$w.SZp" "$N" "$BS" "$abs" $DIMS > "$L/cpu_${tag}_T${T}_d$r.log" 2>&1 || { WARN+=("$lab CPU T=$T $mode $eb: decompress rc $?"); break; }
        tc=$(minf "$tc" "$(grep -m1 -o 'compression time=[0-9.e+-]*' "$L/cpu_${tag}_T${T}_c$r.log" | cut -d= -f2)")
        td=$(minf "$td" "$(grep -m1 -o 'decompression time=[0-9.e+-]*' "$L/cpu_${tag}_T${T}_d$r.log" | cut -d= -f2)")
      done
      if [ -f "$w.SZp.out" ]; then
        OMP_NUM_THREADS=$NCPU "$B/topo_check" "$F" "$w.SZp.out" "$abs" $DIMS > "$L/topo_cpu_${tag}_T$T.log" 2>&1
        record "$lab" "$dim" "CPU-${T}t" "$mode" "$eb" "$abs" "$tc" "$td" "$L/topo_cpu_${tag}_T$T.log"
      fi
      rm -f "$w.SZp" "$w.SZp.out"
    done
  fi

  if [ $GPU = 1 ]; then
    exe=$OUT/build_cuda/test_cuda_topology; [ "$dim" = 3 ] && exe=$OUT/build_cuda/test_cuda_topology_3d
    tc=""; td=""; o=$W/gpu_${tag}.out
    for r in $(seq 1 "$REPS"); do
      TOPOSZP_DUMP=$o OMP_NUM_THREADS=$NCPU timeout "$TIMEOUT" "$exe" "$F" $DIMS "$abs" "$BS" > "$L/gpu_${tag}_$r.log" 2>&1
      rc=$?; [ $rc = 0 ] || WARN+=("$lab GPU $mode $eb: test exit $rc (see $L/gpu_${tag}_$r.log)")
      t=$(grep -m1 '^TIMING' "$L/gpu_${tag}_$r.log")
      tc=$(minf "$tc" "$(echo "$t" | grep -o 'comp_s=[0-9.e+-]*' | cut -d= -f2)")
      td=$(minf "$td" "$(echo "$t" | grep -o 'decomp_s=[0-9.e+-]*' | cut -d= -f2)")
    done
    if [ -f "$o" ]; then
      OMP_NUM_THREADS=$NCPU "$B/topo_check" "$F" "$o" "$abs" $DIMS > "$L/topo_gpu_${tag}.log" 2>&1
      record "$lab" "$dim" "GPU" "$mode" "$eb" "$abs" "$tc" "$td" "$L/topo_gpu_${tag}.log"
      rm -f "$o"
    fi
  fi
done

# ---------------------------------------------------------------- table
{
  echo "# TopoSZp benchmark"
  echo
  echo "$(date), $(hostname), $NCPU cores$( [ $GPU = 1 ] && echo ", $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"), git $(git rev-parse --short HEAD 2>/dev/null), block size $BS, times = minimum of $REPS run(s)."
  echo
  echo "Before = plain SZp quantization (no topology correction); after = TopoSZp output. Kept counts are \"before → after / original\"."
  echo "Relative bounds are the fraction of the file's value range (2D range $RNG2, 3D range $RNG3)."
  echo
  echo "| Data | Platform | Error bound | Abs. bound | Comp. (s) | Decomp. (s) | Maxima kept | Minima kept | Saddles kept | FP | FN | FT |"
  echo "|---|---|---|---|---:|---:|---|---|---|---|---|---|"
  tail -n +2 "$CSV" | while IFS=, read -r d dims p m e a tc td mo mb ma no nb na so sb sa fpb fpa fnb fna ftb fta eeb eea; do
    echo "| $d ($dims) | $p | $m $e | $a | $tc | $td | $mb → $ma / $mo | $nb → $na / $no | $sb → $sa / $so | $fpb → $fpa | $fnb → $fna | $ftb → $fta |"
  done
  echo
  if [ ${#WARN[@]} -gt 0 ]; then echo "Warnings:"; for x in "${WARN[@]}"; do echo "- $x"; done
  else echo "All runs finished; plain quantization within 1 eb and TopoSZp output within 2 eb in every case."; fi
} > "$MD"
echo; cat "$MD"
echo; echo "Files: $MD  $CSV  $L/"
