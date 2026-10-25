#!/bin/bash
#
# TopoSZp-CUDA Full SDRBench Experiment Script
#
# Runs topology-preserving compression on ALL SDRBench datasets:
#   2D: CESM-ATM (1800x3600) — 60 fields
#   3D: CESM-ATM (26x1800x3600) — 28 fields
#   3D: Hurricane ISABEL (100x500x500) — 13 fields
#   3D: NYX (512x512x512) — 6 fields
#   3D: SCALE-LETKF (98x1200x1200) — 13 fields
#   3D: Miranda (256x384x384) — 7 fields (float64)
#
# Error bounds: 1e-3, 1e-4, 1e-5
# Outputs: CSV per dataset group + averages table
#
# Usage:
#   bash run_sdrbench.sh [quick|full]
#     quick = 2-3 fields per dataset × 3 bounds
#     full  = all fields × 3 bounds

set -e

MODE=${1:-quick}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/../build" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="$RESULTS_DIR/$TIMESTAMP"
mkdir -p "$OUTDIR"

# ---- Data paths ----
DATA_2D_CESM="/lcrc/project/ECP-EZ/tripti/SDRBENCH-CESM-ATM-cleared-1800x3600/original_data"
DATA_3D_CESM="/lcrc/project/ECP-EZ/public/compression/cesmbig-26x1800x3600"
DATA_3D_CESM_ALT="/lcrc/project/ECP-EZ/tripti/SDRBENCH-CESM-ATM-26x1800x3600/original_data"
DATA_HURRICANE="/lcrc/project/ECP-EZ/tripti/SDRBench-data/Hurricane-ISABEL"
DATA_NYX="/lcrc/project/ECP-EZ/tripti/SDRBench-data/NYX"
DATA_SCALE="/lcrc/project/ECP-EZ/tripti/SDRBench-data/SCALE-LETKF"
DATA_MIRANDA="/lcrc/project/ECP-EZ/tripti/SDRBench-data/Miranda"

ERROR_BOUNDS="1e-3 1e-4 1e-5"
BLOCK=64

CSV_HEADER="Source,Dataset,Dims,ErrorBound,CompRatio,CompTime_ms,DecompTime_ms,MaxError,Orig_CPs,Preserved_CPs,PreservRate,FN,FP,FT,Maxima_Orig,Maxima_Pres,Minima_Orig,Minima_Pres,Saddle_Orig,Saddle_Pres"
CSV_ALL="$OUTDIR/results_all.csv"
CSV_AVG="$OUTDIR/averages.csv"
LOG="$OUTDIR/experiment.log"

echo "TopoSZp-CUDA SDRBench Experiments — $TIMESTAMP" | tee "$LOG"
echo "Mode: $MODE" | tee -a "$LOG"
echo "" | tee -a "$LOG"

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | tee -a "$LOG"
fi
echo "" | tee -a "$LOG"

echo "$CSV_HEADER" > "$CSV_ALL"

###############################################################################
# Helpers
###############################################################################
parse_2d() {
    local OUT="$1"
    comp_ratio=$(echo "$OUT" | sed -n 's/.*ratio: \([0-9.]*\).*/\1/p' | head -1)
    comp_time=$(echo "$OUT" | sed -n 's/.*comp_s=\([0-9.]*\).*/\1/p' | head -1)
    decomp_time=$(echo "$OUT" | sed -n 's/.*decomp_s=\([0-9.]*\).*/\1/p' | head -1)
    max_err=$(echo "$OUT" | grep 'Max pointwise error' | sed -n 's/.*: \([0-9.e+-]*\).*/\1/p' | head -1)
    orig_cps=$(echo "$OUT" | grep 'Original CPs:' | sed -n 's/.*CPs: \([0-9]*\).*/\1/p')
    pres_cps=$(echo "$OUT" | grep 'Preserved (same type)' | sed -n 's/.*: \([0-9]*\).*/\1/p')
    pres_rate=$(echo "$OUT" | grep 'Preserved (same type)' | sed -n 's/.*(\([0-9.]*\).*/\1/p')
    fn=$(echo "$OUT" | grep 'Lost (became regular)' | sed -n 's/.*: \([0-9]*\).*/\1/p')
    fp=$(echo "$OUT" | grep 'New CPs' | sed -n 's/.*: \([0-9]*\).*/\1/p')
    ft=$(echo "$OUT" | grep 'Type changed' | sed -n 's/.*: \([0-9]*\).*/\1/p')
    max_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Maxima:' | sed -n 's/.*Maxima: \([0-9]*\).*/\1/p' | head -1)
    min_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Minima:' | sed -n 's/.*Minima: \([0-9]*\).*/\1/p' | head -1)
    sad_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Saddles:' | sed -n 's/.*Saddles: \([0-9]*\).*/\1/p' | head -1)
    max_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Maxima:' | sed -n 's/.*Maxima: \([0-9]*\).*/\1/p' | head -1)
    min_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Minima:' | sed -n 's/.*Minima: \([0-9]*\).*/\1/p' | head -1)
    sad_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Saddles:' | sed -n 's/.*Saddles: \([0-9]*\).*/\1/p' | head -1)
    comp_ms=$(echo "$comp_time" | awk '{printf "%.2f", $1*1000}')
    decomp_ms=$(echo "$decomp_time" | awk '{printf "%.2f", $1*1000}')
}

parse_3d() {
    local OUT="$1"
    comp_ratio=$(echo "$OUT" | sed -n 's/.*ratio: \([0-9.]*\).*/\1/p' | head -1)
    comp_time=$(echo "$OUT" | sed -n 's/.*comp_s=\([0-9.]*\).*/\1/p' | head -1)
    decomp_time=$(echo "$OUT" | sed -n 's/.*decomp_s=\([0-9.]*\).*/\1/p' | head -1)
    max_err=$(echo "$OUT" | grep 'Max error' | sed -n 's/.*: \([0-9.e+-]*\).*/\1/p' | head -1)
    orig_cps=$(echo "$OUT" | grep 'Original:' | sed -n 's/.*Original: \([0-9]*\).*/\1/p')
    pres_cps=$(echo "$OUT" | grep 'Preserved:' | sed -n 's/.*Preserved: \([0-9]*\).*/\1/p' | head -1)
    pres_rate=$(echo "$OUT" | grep 'Preserved:' | sed -n 's/.*(\([0-9.]*\).*/\1/p' | head -1)
    fn=$(echo "$OUT" | grep 'Lost:' | sed -n 's/.*Lost: \([0-9]*\).*/\1/p')
    fp=$(echo "$OUT" | grep 'New:' | sed -n 's/.*New: \([0-9]*\).*/\1/p')
    ft=$(echo "$OUT" | grep 'Changed:' | sed -n 's/.*Changed: \([0-9]*\).*/\1/p')
    max_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Maxima:' | sed -n 's/.*Maxima: \([0-9]*\).*/\1/p' | head -1)
    min_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Minima:' | sed -n 's/.*Minima: \([0-9]*\).*/\1/p' | head -1)
    sad_orig=$(echo "$OUT" | grep -A1 'Step 1' | grep 'Saddles:' | sed -n 's/.*Saddles: \([0-9]*\).*/\1/p' | head -1)
    max_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Maxima:' | sed -n 's/.*Maxima: \([0-9]*\).*/\1/p' | head -1)
    min_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Minima:' | sed -n 's/.*Minima: \([0-9]*\).*/\1/p' | head -1)
    sad_pres=$(echo "$OUT" | grep -A2 'Step 5' | grep 'Saddles:' | sed -n 's/.*Saddles: \([0-9]*\).*/\1/p' | head -1)
    comp_ms=$(echo "$comp_time" | awk '{printf "%.2f", $1*1000}')
    decomp_ms=$(echo "$decomp_time" | awk '{printf "%.2f", $1*1000}')
}

# run_dataset SOURCE BINARY FILE DIMS... EB BLOCK
run_2d_dataset() {
    local source="$1" ds="$2" file="$3" rows="$4" cols="$5"
    for eb in $ERROR_BOUNDS; do
        echo -n "  $ds eb=$eb ... " | tee -a "$LOG"
        OUT=$("$BUILD_DIR/test_cuda_topology" "$file" $rows $cols $eb $BLOCK 2>&1) || true
        parse_2d "$OUT"
        echo "$source,$ds,${rows}x${cols},$eb,$comp_ratio,$comp_ms,$decomp_ms,$max_err,$orig_cps,$pres_cps,$pres_rate,$fn,$fp,$ft,$max_orig,$max_pres,$min_orig,$min_pres,$sad_orig,$sad_pres" >> "$CSV_ALL"
        echo "ratio=${comp_ratio}x  pres=${pres_rate}%  FN=$fn FP=$fp FT=$ft  t=${comp_ms}ms" | tee -a "$LOG"
        echo "$OUT" > "$OUTDIR/${source}_${ds}_eb${eb}.log"
    done
}

run_3d_dataset() {
    local source="$1" ds="$2" file="$3" d1="$4" d2="$5" d3="$6"
    for eb in $ERROR_BOUNDS; do
        echo -n "  $ds eb=$eb ... " | tee -a "$LOG"
        OUT=$("$BUILD_DIR/test_cuda_topology_3d" "$file" $d1 $d2 $d3 $eb $BLOCK 2>&1) || true
        parse_3d "$OUT"
        echo "$source,$ds,${d1}x${d2}x${d3},$eb,$comp_ratio,$comp_ms,$decomp_ms,$max_err,$orig_cps,$pres_cps,$pres_rate,$fn,$fp,$ft,$max_orig,$max_pres,$min_orig,$min_pres,$sad_orig,$sad_pres" >> "$CSV_ALL"
        echo "ratio=${comp_ratio}x  pres=${pres_rate}%  FN=$fn FP=$fp FT=$ft  t=${comp_ms}ms" | tee -a "$LOG"
        echo "$OUT" > "$OUTDIR/${source}_${ds}_3d_eb${eb}.log"
    done
}

###############################################################################
# 1. CESM-ATM 2D (1800x3600)
###############################################################################
if [ -d "$DATA_2D_CESM" ]; then
    echo "========== CESM-ATM 2D (1800x3600) ==========" | tee -a "$LOG"
    if [ "$MODE" = "quick" ]; then
        FILES_2D="CLDHGH CLDLOW FREQSH"
    else
        FILES_2D=$(ls "$DATA_2D_CESM"/*.dat 2>/dev/null | xargs -I{} basename {} .dat | sed 's/_1_1800_3600//' | sort)
    fi
    for ds in $FILES_2D; do
        f="$DATA_2D_CESM/${ds}_1_1800_3600.dat"
        [ -f "$f" ] && run_2d_dataset "CESM-ATM-2D" "$ds" "$f" 1800 3600
    done
fi

###############################################################################
# 2. CESM-ATM 3D (26x1800x3600)
###############################################################################
echo "" | tee -a "$LOG"
echo "========== CESM-ATM 3D (26x1800x3600) ==========" | tee -a "$LOG"
if [ "$MODE" = "quick" ]; then
    FILES_3D_CESM="CLOUD OMEGA T"
else
    FILES_3D_CESM=""
    for dir in "$DATA_3D_CESM" "$DATA_3D_CESM_ALT"; do
        [ ! -d "$dir" ] && continue
        for f in "$dir"/*_1_26_1800_3600.dat "$dir"/*_1_26_1800_3600.f32; do
            [ ! -f "$f" ] && continue
            name=$(basename "$f" | sed 's/_1_26_1800_3600\.\(dat\|f32\)//')
            FILES_3D_CESM="$FILES_3D_CESM $name"
        done
    done
    FILES_3D_CESM=$(echo "$FILES_3D_CESM" | tr ' ' '\n' | sort -u | tr '\n' ' ')
fi
for ds in $FILES_3D_CESM; do
    f=""
    for dir in "$DATA_3D_CESM" "$DATA_3D_CESM_ALT"; do
        for ext in dat f32; do
            [ -f "$dir/${ds}_1_26_1800_3600.$ext" ] && f="$dir/${ds}_1_26_1800_3600.$ext" && break 2
        done
    done
    # Handle 27-level fields (CMFMC, CMFMCDZM)
    if [ -z "$f" ]; then
        for dir in "$DATA_3D_CESM" "$DATA_3D_CESM_ALT"; do
            for ext in dat f32; do
                [ -f "$dir/${ds}_1_27_1800_3600.$ext" ] && f="$dir/${ds}_1_27_1800_3600.$ext" && break 2
            done
        done
        [ -n "$f" ] && { run_3d_dataset "CESM-ATM-3D" "$ds" "$f" 27 1800 3600; continue; }
    fi
    [ -n "$f" ] && run_3d_dataset "CESM-ATM-3D" "$ds" "$f" 26 1800 3600
done

###############################################################################
# 3. Hurricane ISABEL (100x500x500)
###############################################################################
if [ -d "$DATA_HURRICANE" ]; then
    echo "" | tee -a "$LOG"
    echo "========== Hurricane ISABEL (100x500x500) ==========" | tee -a "$LOG"
    if [ "$MODE" = "quick" ]; then
        FILES_HURR="CLOUDf48 PRECIPf48 Pf48"
    else
        FILES_HURR=$(ls "$DATA_HURRICANE"/*.bin "$DATA_HURRICANE"/*.f32 2>/dev/null | xargs -I{} basename {} | sed 's/\.\(bin\|f32\)$//' | sort)
    fi
    for ds in $FILES_HURR; do
        f=""
        for ext in bin f32; do
            [ -f "$DATA_HURRICANE/${ds}.${ext}" ] && f="$DATA_HURRICANE/${ds}.${ext}" && break
        done
        [ -n "$f" ] && run_3d_dataset "Hurricane" "$ds" "$f" 100 500 500
    done
else
    echo "SKIP: Hurricane ISABEL not found at $DATA_HURRICANE" | tee -a "$LOG"
    echo "  Run: bash experiments/download_datasets.sh" | tee -a "$LOG"
fi

###############################################################################
# 4. NYX (512x512x512)
###############################################################################
if [ -d "$DATA_NYX" ]; then
    echo "" | tee -a "$LOG"
    echo "========== NYX (512x512x512) ==========" | tee -a "$LOG"
    if [ "$MODE" = "quick" ]; then
        FILES_NYX="baryon_density dark_matter_density temperature"
    else
        FILES_NYX=$(ls "$DATA_NYX"/*.f32 "$DATA_NYX"/*.dat 2>/dev/null | xargs -I{} basename {} | sed 's/\.\(f32\|dat\)$//' | sort)
    fi
    for ds in $FILES_NYX; do
        f=""
        for ext in f32 dat; do
            [ -f "$DATA_NYX/${ds}.${ext}" ] && f="$DATA_NYX/${ds}.${ext}" && break
        done
        [ -n "$f" ] && run_3d_dataset "NYX" "$ds" "$f" 512 512 512
    done
else
    echo "SKIP: NYX not found at $DATA_NYX" | tee -a "$LOG"
    echo "  Run: bash experiments/download_datasets.sh" | tee -a "$LOG"
fi

###############################################################################
# 5. SCALE-LETKF (98x1200x1200)
###############################################################################
if [ -d "$DATA_SCALE" ]; then
    echo "" | tee -a "$LOG"
    echo "========== SCALE-LETKF (98x1200x1200) ==========" | tee -a "$LOG"
    if [ "$MODE" = "quick" ]; then
        FILES_SCALE="PRES QV U"
    else
        FILES_SCALE=$(ls "$DATA_SCALE"/*.f32 "$DATA_SCALE"/*.dat 2>/dev/null | xargs -I{} basename {} | sed 's/\.\(f32\|dat\)$//' | sort)
    fi
    for ds in $FILES_SCALE; do
        f=""
        for ext in f32 dat; do
            [ -f "$DATA_SCALE/${ds}.${ext}" ] && f="$DATA_SCALE/${ds}.${ext}" && break
        done
        [ -n "$f" ] && run_3d_dataset "SCALE" "$ds" "$f" 98 1200 1200
    done
else
    echo "SKIP: SCALE-LETKF not found at $DATA_SCALE" | tee -a "$LOG"
    echo "  Run: bash experiments/download_datasets.sh" | tee -a "$LOG"
fi

###############################################################################
# 6. Miranda (256x384x384) — float64, needs special handling
###############################################################################
if [ -d "$DATA_MIRANDA" ]; then
    echo "" | tee -a "$LOG"
    echo "========== Miranda (256x384x384, float64) ==========" | tee -a "$LOG"
    echo "  NOTE: Miranda is float64. Testing if supported..." | tee -a "$LOG"
    if [ "$MODE" = "quick" ]; then
        FILES_MIRANDA="density pressure"
    else
        FILES_MIRANDA=$(ls "$DATA_MIRANDA"/*.f64 "$DATA_MIRANDA"/*.f32 2>/dev/null | xargs -I{} basename {} | sed 's/\.\(f64\|f32\)$//' | sort)
    fi
    for ds in $FILES_MIRANDA; do
        f=""
        for ext in f32 f64; do
            [ -f "$DATA_MIRANDA/${ds}.${ext}" ] && f="$DATA_MIRANDA/${ds}.${ext}" && break
        done
        [ -n "$f" ] && run_3d_dataset "Miranda" "$ds" "$f" 256 384 384
    done
else
    echo "SKIP: Miranda not found at $DATA_MIRANDA" | tee -a "$LOG"
    echo "  Run: bash experiments/download_datasets.sh" | tee -a "$LOG"
fi

###############################################################################
# Compute averages per source × error bound
###############################################################################
echo "" | tee -a "$LOG"
echo "========== AVERAGES PER DATASET × ERROR BOUND ==========" | tee -a "$LOG"

echo "Source,ErrorBound,Avg_CompRatio,Avg_CompTime_ms,Avg_DecompTime_ms,Avg_PreservRate,Avg_FN,Avg_FP,Avg_FT,Num_Fields" > "$CSV_AVG"

# Get unique sources
SOURCES=$(awk -F',' 'NR>1 {print $1}' "$CSV_ALL" | sort -u)

for src in $SOURCES; do
    for eb in $ERROR_BOUNDS; do
        lines=$(grep "^${src}," "$CSV_ALL" | grep ",$eb,")
        count=$(echo "$lines" | grep -c . 2>/dev/null || echo 0)
        [ "$count" -eq 0 ] && continue

        avg_ratio=$(echo "$lines" | awk -F',' '{s+=$5; n++} END {if(n>0) printf "%.2f",s/n; else print "N/A"}')
        avg_comp=$(echo "$lines" | awk -F',' '{s+=$6; n++} END {if(n>0) printf "%.2f",s/n; else print "N/A"}')
        avg_decomp=$(echo "$lines" | awk -F',' '{s+=$7; n++} END {if(n>0) printf "%.2f",s/n; else print "N/A"}')
        avg_pres=$(echo "$lines" | awk -F',' '{s+=$11; n++} END {if(n>0) printf "%.2f",s/n; else print "N/A"}')
        avg_fn=$(echo "$lines" | awk -F',' '{s+=$12; n++} END {if(n>0) printf "%.1f",s/n; else print "N/A"}')
        avg_fp=$(echo "$lines" | awk -F',' '{s+=$13; n++} END {if(n>0) printf "%.1f",s/n; else print "N/A"}')
        avg_ft=$(echo "$lines" | awk -F',' '{s+=$14; n++} END {if(n>0) printf "%.1f",s/n; else print "N/A"}')

        echo "$src,$eb,$avg_ratio,$avg_comp,$avg_decomp,$avg_pres,$avg_fn,$avg_fp,$avg_ft,$count" >> "$CSV_AVG"
        printf "  %-18s eb=%-6s  ratio=%5sx  pres=%7s%%  FN=%6s  FP=%4s  FT=%4s  (n=%d)\n" \
            "$src" "$eb" "$avg_ratio" "$avg_pres" "$avg_fn" "$avg_fp" "$avg_ft" "$count" | tee -a "$LOG"
    done
done

###############################################################################
# Grand averages per error bound (across all sources)
###############################################################################
echo "" | tee -a "$LOG"
echo "--- Grand Averages (all datasets) ---" | tee -a "$LOG"
for eb in $ERROR_BOUNDS; do
    lines=$(grep ",$eb," "$CSV_ALL" | grep -v "^Source")
    count=$(echo "$lines" | grep -c . 2>/dev/null || echo 0)
    [ "$count" -eq 0 ] && continue

    avg_ratio=$(echo "$lines" | awk -F',' '{s+=$5; n++} END {if(n>0) printf "%.2f",s/n}')
    avg_pres=$(echo "$lines" | awk -F',' '{s+=$11; n++} END {if(n>0) printf "%.2f",s/n}')
    avg_fn=$(echo "$lines" | awk -F',' '{s+=$12; n++} END {if(n>0) printf "%.1f",s/n}')
    avg_fp=$(echo "$lines" | awk -F',' '{s+=$13; n++} END {if(n>0) printf "%.1f",s/n}')
    avg_ft=$(echo "$lines" | awk -F',' '{s+=$14; n++} END {if(n>0) printf "%.1f",s/n}')

    echo "ALL,$eb,$avg_ratio,,,${avg_pres},$avg_fn,$avg_fp,$avg_ft,$count" >> "$CSV_AVG"
    printf "  %-18s eb=%-6s  ratio=%5sx  pres=%7s%%  FN=%6s  FP=%4s  FT=%4s  (n=%d)\n" \
        "ALL" "$eb" "$avg_ratio" "$avg_pres" "$avg_fn" "$avg_fp" "$avg_ft" "$count" | tee -a "$LOG"
done

###############################################################################
# Print summary
###############################################################################
echo "" | tee -a "$LOG"
echo "==================== RESULTS ====================" | tee -a "$LOG"
echo "All results: $CSV_ALL" | tee -a "$LOG"
echo "Averages:    $CSV_AVG" | tee -a "$LOG"
echo "Full log:    $LOG" | tee -a "$LOG"
echo "Per-run:     $OUTDIR/*.log" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "--- Averages Table ---" | tee -a "$LOG"
column -t -s',' "$CSV_AVG" 2>/dev/null | tee -a "$LOG"
