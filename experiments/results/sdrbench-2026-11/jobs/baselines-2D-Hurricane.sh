#!/bin/bash
#PBS -N tsz-baselines-2D-Hurricane
#PBS -q backfill
#PBS -A SDR
#PBS -l select=1:ngpus=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /lcrc/project/ECP-EZ/tripti/toposzp-results/jobs/baselines-2D-Hurricane.out
module load gcc/11.4.0 2>/dev/null || true
module load nvhpc 2>/dev/null || true
module load cmake 2>/dev/null || true
cd /lcrc/project/ECP-EZ/tripti/topoSZp-cuda/experiments/sdrbench
WORKARG=; [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR" ] && WORKARG="--work $TMPDIR/toposzp-work"   # node-local scratch
set -x
python3 /lcrc/project/ECP-EZ/tripti/topoSZp-cuda/experiments/sdrbench/exp_baselines.py --data /lcrc/project/ECP-EZ/tripti/sdrbench-data --out /lcrc/project/ECP-EZ/tripti/toposzp-results --bin /lcrc/project/ECP-EZ/tripti/topoSZp-cuda/experiments/sdrbench/bin --tag baselines-2D-Hurricane --datasets CESM-ATM-2D\,Hurricane --baseline-bounds abs:1e-3\,abs:1e-4\,abs:1e-5\,rel:1e-3\,rel:1e-4\,rel:1e-5 $WORKARG
