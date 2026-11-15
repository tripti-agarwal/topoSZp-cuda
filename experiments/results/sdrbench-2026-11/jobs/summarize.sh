#!/bin/bash
#PBS -N tsz-summarize
#PBS -q backfill
#PBS -A SDR
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /lcrc/project/ECP-EZ/tripti/toposzp-results/jobs/summarize.out
module load gcc/11.4.0 2>/dev/null || true
module load nvhpc 2>/dev/null || true
module load cmake 2>/dev/null || true
python3 /lcrc/project/ECP-EZ/tripti/topoSZp-cuda/experiments/sdrbench/summarize.py /lcrc/project/ECP-EZ/tripti/toposzp-results
