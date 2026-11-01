#!/usr/bin/env bash
# Submit the experiments as separate PBS jobs (Swing) instead of one long job.
#
#   DATA=/path/to/sdrbench-data ./submit_separate.sh RESULTS_DIR
#
# Jobs (all write into RESULTS_DIR; each has its own CSV, log and work folder):
#   main-<dataset>       exp_main.py      one job per dataset (OpenMP + CUDA)
#   baselines-<dataset>  exp_baselines.py one job per dataset
#   scaling-<dataset>    exp_scaling.py   one job per dataset
#   bitrate              exp_bitrate.py
#   pl                   exp_pl.py
#   vis                  exp_vis.py
#   summarize            summarize.py, after all the others have ended (afterany)
# If the tools are not built yet, a setup job runs first and the others wait for it.
#
# Run the same command again at any time: jobs that are queued or running, and
# jobs that finished (their jobs/<name>.out ends with "done."), are skipped; the
# rest is submitted. A job that hit its walltime is submitted again and resumes
# (finished runs are skipped). summarize.py merges the per-job CSVs.
#
# Per-user job limit ("qsub: would exceed ... per-user limit"): the jobs that did
# not fit are listed; run the command again when some have finished, or add
# WAIT=1 to keep trying every WAIT_MIN minutes (run it in screen / tmux or with
# nohup on the login node). The summarize job is submitted once all others are.
#
# Environment (all optional except DATA):
#   DATA       data folder with manifest.csv (download_sdrbench.sh)
#   EXPS       experiments to submit (default: main,baselines,scaling,bitrate,pl,vis)
#   DATASETS   datasets to submit for the per-dataset experiments (default: all in the manifest)
#   ACCOUNT    PBS account (default: SDR)          QUEUE    PBS queue (default: gpu)
#   SELECT     resources (default: select=1:ngpus=1); SELECT_<EXP> overrides, e.g.
#              SELECT_SCALING=select=1:ngpus=8 for all cores of a Swing node (a 1-GPU job gets a share)
#   PLACE      e.g. excl, to get a whole node for the timed jobs (main, scaling); default: none
#   WALLTIME   default for every job (default: 12:00:00); WALLTIME_<EXP> overrides, e.g. WALLTIME_PL=02:00:00
#   OPTS       options added to every experiment, e.g. "--reps 3 --threads 32"
#   <EXP>_OPTS options for one experiment, e.g. MAIN_OPTS="--bounds rel:1e-3,rel:1e-4"
#   DRYRUN=1   write the job scripts and print the qsub commands without submitting
#   WAIT=1     retry jobs that hit the per-user job limit every WAIT_MIN (default 10) minutes
#   FORCE=1    submit even jobs that finished before
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?usage: DATA=<data folder> $0 RESULTS_DIR}
: "${DATA:?set DATA=<data folder with manifest.csv>}"
DATA=$(cd "$DATA" && pwd)
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
[ -f "$DATA/manifest.csv" ] || { echo "no $DATA/manifest.csv: run download_sdrbench.sh first"; exit 1; }
EXPS=${EXPS:-main,baselines,scaling,bitrate,pl,vis}
ACCOUNT=${ACCOUNT:-SDR}; QUEUE=${QUEUE:-gpu}; SELECT=${SELECT:-select=1:ngpus=1}; PLACE=${PLACE:-}
WALLTIME=${WALLTIME:-12:00:00}; OPTS=${OPTS:-}; DRYRUN=${DRYRUN:-0}
WAIT=${WAIT:-0}; WAIT_MIN=${WAIT_MIN:-10}; FORCE=${FORCE:-0}; ME=${USER:-$(whoami)}
BIN=$HERE/bin
JOBS=$OUT/jobs; mkdir -p "$JOBS"

DATASETS=${DATASETS:-$(tail -n +2 "$DATA/manifest.csv" | cut -d, -f1 | awk '!seen[$0]++' | paste -sd, -)}
[ -n "$DATASETS" ] || { echo "no datasets in $DATA/manifest.csv"; exit 1; }

qsub_() {   # qsub_ <job script> [qsub options]: prints the job id
    if [ "$DRYRUN" = 1 ]; then
        echo "qsub ${*:2} $1" >&2; echo "dry.$(basename "$1" .sh)"
    else
        qsub "${@:2}" "$1"
    fi
}

write_job() {   # write_job <name> <walltime> <place: excl or empty> <select> <command...>
    local name=$1 wt=$2 place=$3 sel=$4; shift 4
    local f=$JOBS/$name.sh
    {
        echo '#!/bin/bash'
        echo "#PBS -N tsz-$name"
        echo "#PBS -q $QUEUE"
        echo "#PBS -A $ACCOUNT"
        echo "#PBS -l $sel"
        echo "#PBS -l walltime=$wt"
        [ -n "$place" ] && echo "#PBS -l place=$place"
        echo '#PBS -j oe'
        echo "#PBS -o $JOBS/$name.out"
        echo 'module load gcc/11.4.0 2>/dev/null || true'
        echo 'module load nvhpc 2>/dev/null || true'
        echo 'module load cmake 2>/dev/null || true'
        echo "cd $HERE"
        echo 'WORKARG=; [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR" ] && WORKARG="--work $TMPDIR/toposzp-work"   # node-local scratch'
        echo 'set -x'
        printf '%q ' "$@"; echo '$WORKARG'
    } > "$f"
    echo "$f"
}

walltime_for() {   # WALLTIME_<EXP> or WALLTIME
    local v="WALLTIME_$(echo "$1" | tr a-z A-Z)"
    echo "${!v:-$WALLTIME}"
}
select_for() {   # SELECT_<EXP> or SELECT
    local v="SELECT_$(echo "$1" | tr a-z A-Z)"
    echo "${!v:-$SELECT}"
}
opts_for() {
    local v="$(echo "$1" | tr a-z A-Z)_OPTS"
    echo "$OPTS ${!v:-}"
}

# ---- setup job if the tools are not built
DEP=()
if [ ! -x "$BIN/toposzp_comp" ]; then
    f=$JOBS/setup.sh
    printf '#!/bin/bash\n#PBS -N tsz-setup\n#PBS -q %s\n#PBS -A %s\n#PBS -l %s\n#PBS -l walltime=01:00:00\n#PBS -j oe\n#PBS -o %s\nmodule load gcc/11.4.0 2>/dev/null || true\nmodule load nvhpc 2>/dev/null || true\nmodule load cmake 2>/dev/null || true\n%q %q\n' \
        "$QUEUE" "$ACCOUNT" "$SELECT" "$JOBS/setup.out" "$HERE/setup.sh" "$BIN" > "$f"
    sid=$(qsub_ "$f"); echo "setup      $sid"
    DEP=(-W "depend=afterok:$sid")
fi

queued_id() {   # id of a queued / running / held job named tsz-<tag>, if any
    [ "$DRYRUN" = 1 ] && return 0
    command -v qselect >/dev/null || return 0
    qselect -u "$ME" -N "tsz-$1" -s QRHWT 2>/dev/null | head -1
}

IDS=()        # jobs the summary waits for
PENDING=()    # "exp|tag|place|datasets" not submitted (job limit)
LIMIT=0
submit() {   # submit <exp> <tag> <place> [dataset]
    local exp=$1 tag=$2 place=$3 ds=${4:-}
    local f id err
    id=$(queued_id "$tag")
    if [ -n "$id" ]; then
        IDS+=("$id"); printf '%-28s %s (already queued)\n' "$tag" "$id"; return 0
    fi
    if [ "$FORCE" != 1 ] && [ -f "$JOBS/$tag.out" ] && grep -q '^\[.*\] done\. Results in' "$JOBS/$tag.out"; then
        printf '%-28s finished before (FORCE=1 to run again)\n' "$tag"; return 0
    fi
    if [ "$LIMIT" = 1 ]; then PENDING+=("$exp|$tag|$place|$ds"); return 0; fi
    # shellcheck disable=SC2046
    f=$(write_job "$tag" "$(walltime_for "$exp")" "$place" "$(select_for "$exp")" python3 "$HERE/exp_$exp.py" --data "$DATA" --out "$OUT" \
        --bin "$BIN" --tag "$tag" ${ds:+--datasets "$ds"} $(opts_for "$exp"))
    if [ "$DRYRUN" = 1 ]; then
        id=$(qsub_ "$f" "${DEP[@]}")
    elif ! id=$(qsub "${DEP[@]}" "$f" 2>"$JOBS/.qsub.err"); then
        err=$(cat "$JOBS/.qsub.err")
        if echo "$err" | grep -qi 'limit'; then
            LIMIT=1; PENDING+=("$exp|$tag|$place|$ds"); return 0
        fi
        echo "qsub failed for $tag: $err" >&2; exit 1
    fi
    IDS+=("$id"); printf '%-28s %s\n' "$tag" "$id"
}

submit_all() {
    LIMIT=0; PENDING=()
    IFS=, read -r -a DS <<< "$DATASETS"
    for e in ${EXPS//,/ }; do
        case $e in
            main)      for d in "${DS[@]}"; do submit main      "main-$d"      "$PLACE" "$d"; done ;;
            scaling)   for d in "${DS[@]}"; do submit scaling   "scaling-$d"   "$PLACE" "$d"; done ;;
            baselines) for d in "${DS[@]}"; do submit baselines "baselines-$d" ""       "$d"; done ;;
            bitrate)   submit bitrate bitrate "" ;;
            pl)        submit pl pl "" ;;
            vis)       submit vis vis "" ;;
            *) echo "unknown experiment $e"; exit 1 ;;
        esac
    done
}

submit_all
while [ ${#PENDING[@]} -gt 0 ] && [ "$WAIT" = 1 ]; do
    echo "$(date +%H:%M) per-user job limit: ${#PENDING[@]} jobs waiting; next try in $WAIT_MIN min"
    sleep $((WAIT_MIN * 60))
    IDS=(); submit_all
done

if [ ${#PENDING[@]} -gt 0 ]; then
    echo
    echo "Not submitted (per-user job limit): $(printf '%s\n' "${PENDING[@]}" | cut -d'|' -f2 | paste -sd' ' -)"
    echo "Run the same command again when some jobs have finished (queued, running and finished jobs"
    echo "are skipped), or add WAIT=1 to keep trying. Then the summarize job is submitted."
    echo "watch with: qstat -u $ME"
    exit 0
fi

# ---- summary after all of them (afterany: also when some failed or hit the walltime)
f=$JOBS/summarize.sh
printf '#!/bin/bash\n#PBS -N tsz-summarize\n#PBS -q %s\n#PBS -A %s\n#PBS -l %s\n#PBS -l walltime=00:30:00\n#PBS -j oe\n#PBS -o %s\npython3 %q %q\n' \
    "$QUEUE" "$ACCOUNT" "$SELECT" "$JOBS/summarize.out" "$HERE/summarize.py" "$OUT" > "$f"
if [ ${#IDS[@]} -gt 0 ]; then
    sid=$(qsub_ "$f" -W "depend=afterany:$(IFS=:; echo "${IDS[*]}")") || {
        echo "summarize job not submitted (job limit?): run  python3 $HERE/summarize.py $OUT  when the jobs are done"; exit 0; }
else
    sid=$(qsub_ "$f") || { echo "run  python3 $HERE/summarize.py $OUT"; exit 0; }
fi
printf '%-28s %s\n' summarize "$sid"
echo "job scripts and outputs: $JOBS;  watch with: qstat -u $ME"
