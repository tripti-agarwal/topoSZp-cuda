#!/usr/bin/env bash
# Submit the experiments as PBS jobs (Swing).
#
#   DATA=/path/to/sdrbench-data ./submit_separate.sh RESULTS_DIR
#
# PLAN=compact (default): 7 jobs + summary, all 1 GPU, each sized to finish in
# about 2-3 h (so they fit a 4 h backfill walltime and the 9-jobs-per-user limit):
#   main-CESM-ATM-3D            exp_main       CESM-ATM-3D
#   main-rest                   exp_main       all other datasets
#   baselines-CESM-ATM-3D-abs   exp_baselines  CESM-ATM-3D, absolute bounds
#   baselines-CESM-ATM-3D-rel   exp_baselines  CESM-ATM-3D, relative bounds
#   baselines-2D-Hurricane      exp_baselines  CESM-ATM-2D, Hurricane
#   baselines-rest              exp_baselines  all other datasets (NYX, SCALE-LETKF)
#   other                       exp_scaling, exp_bitrate, exp_pl, exp_vis (one after another)
#   summarize                   summarize.py, after all the others have ended (afterany)
# PLAN=separate: one job per experiment, main / baselines / scaling one per dataset.
# If the tools are not built yet, a setup job runs first and the others wait for it.
#
# Every experiment writes its own CSV / log / work folder (--tag), so jobs can run
# at the same time; summarize.py merges the CSVs.
#
# Run the same command again at any time: jobs that are queued or running, and
# jobs that finished (every command of the job printed "done."), are skipped;
# the rest is submitted. A job that hit its walltime is submitted again and
# resumes (finished runs are skipped).
#
# Per-user job limit ("qsub: would exceed ... per-user limit"): the jobs that did
# not fit are listed; run the command again when some have finished, or add
# WAIT=1 to keep trying every WAIT_MIN minutes (in screen / tmux, or with nohup,
# on the login node). The summarize job is submitted once all others are.
#
# Environment (all optional except DATA):
#   DATA       data folder with manifest.csv (download_sdrbench.sh)
#   PLAN       compact (default) or separate
#   EXPS       experiments to submit (default: main,baselines,scaling,bitrate,pl,vis)
#   DATASETS   datasets (default: all in the manifest)
#   ACCOUNT    PBS account (default: SDR)          QUEUE    PBS queue (default: gpu)
#   SELECT     resources (default: select=1:ngpus=1); SELECT_<JOB> overrides for one job,
#              e.g. SELECT_OTHER (compact) or SELECT_SCALING_NYX (separate)
#   PLACE      e.g. excl, for the timed jobs (main, scaling); default: none
#   WALLTIME   default for every job (default: 04:00:00); WALLTIME_<JOB> overrides
#   OPTS       options added to every experiment, e.g. "--reps 3"
#   <EXP>_OPTS options for one experiment, e.g. MAIN_OPTS="--bounds rel:1e-3,rel:1e-4"
#   BASELINE_BOUNDS  bounds of the baselines jobs (default: abs:1e-3,abs:1e-4,abs:1e-5,rel:1e-3,rel:1e-4,rel:1e-5)
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
PLAN=${PLAN:-compact}
EXPS=${EXPS:-main,baselines,scaling,bitrate,pl,vis}
ACCOUNT=${ACCOUNT:-SDR}; QUEUE=${QUEUE:-gpu}; SELECT=${SELECT:-select=1:ngpus=1}; PLACE=${PLACE:-}
WALLTIME=${WALLTIME:-04:00:00}; OPTS=${OPTS:-}; DRYRUN=${DRYRUN:-0}
WAIT=${WAIT:-0}; WAIT_MIN=${WAIT_MIN:-10}; FORCE=${FORCE:-0}; ME=${USER:-$(whoami)}
BASELINE_BOUNDS=${BASELINE_BOUNDS:-abs:1e-3,abs:1e-4,abs:1e-5,rel:1e-3,rel:1e-4,rel:1e-5}
BIN=$HERE/bin
JOBS=$OUT/jobs; mkdir -p "$JOBS"

DATASETS=${DATASETS:-$(tail -n +2 "$DATA/manifest.csv" | cut -d, -f1 | awk '!seen[$0]++' | paste -sd, -)}
[ -n "$DATASETS" ] || { echo "no datasets in $DATA/manifest.csv"; exit 1; }
IFS=, read -r -a DS <<< "$DATASETS"
has_exp() { [[ ",$EXPS," == *",$1,"* ]]; }
var_or() { local v; v=$(echo "$1" | tr 'a-z-' 'A-Z_'); echo "${!v:-$2}"; }   # var_or NAME default
ds_except() { local out=() d x skip; for d in "${DS[@]}"; do skip=0; for x in "$@"; do [ "$d" = "$x" ] && skip=1; done
              [ $skip = 0 ] && out+=("$d"); done; (IFS=,; echo "${out[*]}"); }
ds_only() { local out=() d x; for d in "${DS[@]}"; do for x in "$@"; do [ "$d" = "$x" ] && out+=("$d"); done; done; (IFS=,; echo "${out[*]}"); }
filter_bounds() { echo "$BASELINE_BOUNDS" | tr , '\n' | grep "^$1:" | paste -sd, - || true; }

# ---- job list: JOB_NAMES[i]; JOB_CMDS[i] = "exp|tag|options;exp|tag|options"; JOB_TIMED[i] (PLACE applies)
JOB_NAMES=(); JOB_CMDS=(); JOB_TIMED=()
add_job() {   # add_job <name> <timed 0/1> <exp|tag|options>...
    local name=$1 timed=$2; shift 2
    [ $# -gt 0 ] || return 0
    local IFS=';'; JOB_NAMES+=("$name"); JOB_TIMED+=("$timed"); JOB_CMDS+=("$*")
}
if [ "$PLAN" = compact ]; then
    HEAVY=CESM-ATM-3D
    h=$(ds_only $HEAVY); rest=$(ds_except $HEAVY)
    light=$(ds_only CESM-ATM-2D Hurricane); other=$(ds_except $HEAVY CESM-ATM-2D Hurricane)
    if has_exp main; then
        [ -n "$h" ]    && add_job "main-$HEAVY" 1 "main|main-$HEAVY|--datasets $h"
        [ -n "$rest" ] && add_job main-rest 1 "main|main-rest|--datasets $rest"
    fi
    if has_exp baselines; then
        ab=$(filter_bounds abs); rb=$(filter_bounds rel)
        if [ -n "$h" ]; then
            [ -n "$ab" ] && add_job "baselines-$HEAVY-abs" 0 "baselines|baselines-$HEAVY-abs|--datasets $h --baseline-bounds $ab"
            [ -n "$rb" ] && add_job "baselines-$HEAVY-rel" 0 "baselines|baselines-$HEAVY-rel|--datasets $h --baseline-bounds $rb"
        fi
        [ -n "$light" ] && add_job baselines-2D-Hurricane 0 "baselines|baselines-2D-Hurricane|--datasets $light --baseline-bounds $BASELINE_BOUNDS"
        [ -n "$other" ] && add_job baselines-rest 0 "baselines|baselines-rest|--datasets $other --baseline-bounds $BASELINE_BOUNDS"
    fi
    o=()
    has_exp scaling && o+=("scaling|other-scaling|--datasets $DATASETS")
    has_exp bitrate && o+=("bitrate|other-bitrate|")
    has_exp pl      && o+=("pl|other-pl|--datasets $DATASETS")
    has_exp vis     && o+=("vis|other-vis|")
    [ ${#o[@]} -gt 0 ] && add_job other 1 "${o[@]}"
else
    for e in ${EXPS//,/ }; do
        case $e in
            main|scaling) for d in "${DS[@]}"; do add_job "$e-$d" 1 "$e|$e-$d|--datasets $d"; done ;;
            baselines)    for d in "${DS[@]}"; do add_job "$e-$d" 0 "$e|$e-$d|--datasets $d --baseline-bounds $BASELINE_BOUNDS"; done ;;
            bitrate|pl|vis) add_job "$e" 0 "$e|$e|" ;;
            *) echo "unknown experiment $e"; exit 1 ;;
        esac
    done
fi
[ ${#JOB_NAMES[@]} -gt 0 ] || { echo "nothing to submit"; exit 0; }

# ---- job scripts
header() {   # header <name> <walltime> <select> <place>
    echo '#!/bin/bash'
    echo "#PBS -N tsz-$1"
    echo "#PBS -q $QUEUE"
    echo "#PBS -A $ACCOUNT"
    echo "#PBS -l $3"
    echo "#PBS -l walltime=$2"
    if [ -n "$4" ]; then echo "#PBS -l place=$4"; fi
    echo '#PBS -j oe'
    echo "#PBS -o $JOBS/$1.out"
    echo 'module load gcc/11.4.0 2>/dev/null || true'
    echo 'module load nvhpc 2>/dev/null || true'
    echo 'module load cmake 2>/dev/null || true'
}
write_job() {   # write_job <index>: prints the script path
    local i=$1 name=${JOB_NAMES[$1]} cmd exp tag opts eo place=""
    local f=$JOBS/$name.sh
    [ "${JOB_TIMED[$i]}" = 1 ] && place=$PLACE
    {
        header "$name" "$(var_or "WALLTIME_$name" "$WALLTIME")" "$(var_or "SELECT_$name" "$SELECT")" "$place"
        echo "cd $HERE"
        echo 'WORKARG=; [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR" ] && WORKARG="--work $TMPDIR/toposzp-work"   # node-local scratch'
        echo 'set -x'
        IFS=';' read -r -a cmds <<< "${JOB_CMDS[$i]}"
        for cmd in "${cmds[@]}"; do
            IFS='|' read -r exp tag opts <<< "$cmd"
            eo=$(var_or "${exp}_OPTS" "")
            # shellcheck disable=SC2086
            printf '%q ' python3 "$HERE/exp_$exp.py" --data "$DATA" --out "$OUT" --bin "$BIN" --tag "$tag" $opts $OPTS $eo
            echo '$WORKARG'
        done
    } > "$f"
    echo "$f"
}
n_cmds() { local IFS=';'; set -- ${JOB_CMDS[$1]}; echo $#; }

qsub_() {   # qsub_ <job script> [qsub options]: prints the job id
    if [ "$DRYRUN" = 1 ]; then echo "qsub ${*:2} $1" >&2; echo "dry.$(basename "$1" .sh)"; else qsub "${@:2}" "$1"; fi
}

# ---- setup job if the tools are not built
DEP=()
if [ ! -x "$BIN/toposzp_comp" ]; then
    f=$JOBS/setup.sh
    { header setup 01:00:00 "$SELECT" ""; printf '%q %q\n' "$HERE/setup.sh" "$BIN"; } > "$f"
    sid=$(qsub_ "$f"); printf '%-28s %s\n' setup "$sid"
    DEP=(-W "depend=afterok:$sid")
fi

queued_id() {   # id of a queued / running / held job named tsz-<name>, if any
    [ "$DRYRUN" = 1 ] && return 0
    command -v qselect >/dev/null || return 0
    qselect -u "$ME" -N "tsz-$1" -s QRHWT 2>/dev/null | head -1
}

IDS=(); PENDING=(); LIMIT=0
submit() {   # submit <index>
    local i=$1 name=${JOB_NAMES[$1]} f id err
    id=$(queued_id "$name")
    if [ -n "$id" ]; then IDS+=("$id"); printf '%-28s %s (already queued)\n' "$name" "$id"; return 0; fi
    if [ "$FORCE" != 1 ] && [ -f "$JOBS/$name.out" ] &&
       [ "$(grep -c '^\[.*\] done\. Results in' "$JOBS/$name.out" || true)" -ge "$(n_cmds "$i")" ]; then
        printf '%-28s finished before (FORCE=1 to run again)\n' "$name"; return 0
    fi
    if [ "$LIMIT" = 1 ]; then PENDING+=("$name"); return 0; fi
    f=$(write_job "$i")
    if [ "$DRYRUN" = 1 ]; then
        id=$(qsub_ "$f" "${DEP[@]}")
    elif ! id=$(qsub "${DEP[@]}" "$f" 2>"$JOBS/.qsub.err"); then
        err=$(cat "$JOBS/.qsub.err")
        if echo "$err" | grep -qi 'limit'; then LIMIT=1; PENDING+=("$name"); return 0; fi
        echo "qsub failed for $name: $err" >&2; exit 1
    fi
    IDS+=("$id"); printf '%-28s %s\n' "$name" "$id"
}
submit_all() { LIMIT=0; PENDING=(); IDS=(); local i; for i in "${!JOB_NAMES[@]}"; do submit "$i"; done; }

submit_all
while [ ${#PENDING[@]} -gt 0 ] && [ "$WAIT" = 1 ]; do
    echo "$(date +%H:%M) per-user job limit: ${#PENDING[@]} jobs waiting; next try in $WAIT_MIN min"
    sleep $((WAIT_MIN * 60))
    submit_all
done
if [ ${#PENDING[@]} -gt 0 ]; then
    echo
    echo "Not submitted (per-user job limit): ${PENDING[*]}"
    echo "Run the same command again when some jobs have finished (queued, running and finished jobs"
    echo "are skipped), or add WAIT=1 to keep trying. Then the summarize job is submitted."
    echo "watch with: qstat -u $ME"
    exit 0
fi

# ---- summary after all of them (afterany: also when some failed or hit the walltime)
old_sid=$(queued_id summarize)   # replace a summary job from an earlier call (its dependencies are stale)
[ -n "$old_sid" ] && qdel "$old_sid" 2>/dev/null || true
f=$JOBS/summarize.sh
{ header summarize 00:30:00 "$SELECT" ""; printf 'python3 %q %q\n' "$HERE/summarize.py" "$OUT"; } > "$f"
if [ ${#IDS[@]} -gt 0 ]; then
    sid=$(qsub_ "$f" -W "depend=afterany:$(IFS=:; echo "${IDS[*]}")") ||
        { echo "summarize job not submitted: run  python3 $HERE/summarize.py $OUT  when the jobs are done"; exit 0; }
else
    sid=$(qsub_ "$f") || { echo "run  python3 $HERE/summarize.py $OUT"; exit 0; }
fi
printf '%-28s %s\n' summarize "$sid"
echo "job scripts and outputs: $JOBS;  watch with: qstat -u $ME"
