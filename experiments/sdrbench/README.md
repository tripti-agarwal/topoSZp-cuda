# TopoSZp+ experiments on SDRBench

Scripts to download the SDRBench datasets and run every experiment of the paper
with the current TopoSZp+ code (OpenMP and CUDA), from scratch, on one GPU node.

| Script | What it does |
|---|---|
| `download_sdrbench.sh` | Downloads CESM-ATM 2D (1800×3600), CESM-ATM 3D (26×1800×3600), Hurricane-ISABEL (100×500×500), NYX (512³) and SCALE-LETKF (98×1200×1200), checks every file's size, writes `manifest.csv`. Can link data you already have instead. |
| `setup.sh` | Builds the TopoSZp+ OpenMP compressor/decompressor, the CUDA programs, `topo_check`, `pl_topo_check`, a plain-SZp tool, and the baselines SZ3, SZ2 and ZFP (TTHRESH with `TTHRESH=1`). |
| `exp_main.py`, `exp_scaling.py`, `exp_baselines.py`, `exp_bitrate.py`, `exp_pl.py`, `exp_vis.py` | One script per experiment (table below); each runs alone, writes its own CSV and resumes. |
| `common.py` | Code shared by the experiment scripts (compressor calls, topology check, tables, options). |
| `run_experiments.py` | Runs several experiments one after another in one process (`--exp main,pl`). |
| `submit_separate.sh` | Submits every experiment as its own PBS job, the heavy ones (`main`, `baselines`, `scaling`) one job per dataset, plus a summary job at the end. |
| `summarize.py` | Makes `summary.md`, `tables.tex` and `plots/` from the CSVs (merges the per-job CSVs). |
| `run_all.sh`, `run_all.pbs` | Everything with one command / as one PBS job on Swing. |

## Quick start (Swing)

```bash
cd /lcrc/project/ECP-EZ/tripti/topoSZp-cuda          # the repository
git fetch && git checkout claude/sdrbench-experiments

# 1. data (≈35 GB). Link CESM if you already have it, download the rest:
CESM_ATM_2D_SRC=/lcrc/project/ECP-EZ/tripti/SDRBENCH-CESM-ATM-cleared-1800x3600/original_data \
CESM_ATM_3D_SRC=/lcrc/project/ECP-EZ/public/compression/cesmbig-26x1800x3600 \
experiments/sdrbench/download_sdrbench.sh /lcrc/project/ECP-EZ/tripti/sdrbench-data

# 2a. every experiment as its own job, main/baselines/scaling one job per dataset (recommended)
DATA=/lcrc/project/ECP-EZ/tripti/sdrbench-data \
experiments/sdrbench/submit_separate.sh /lcrc/project/ECP-EZ/tripti/toposzp-results

# 2b. or everything in one long job (resubmit the same line if it hits the walltime; it resumes)
qsub -v DATA=/lcrc/project/ECP-EZ/tripti/sdrbench-data,OUT=/lcrc/project/ECP-EZ/tripti/toposzp-results \
     experiments/sdrbench/run_all.pbs
```

## Batch jobs (submit_separate.sh)

`submit_separate.sh RESULTS` writes the job scripts into `RESULTS/jobs/` and submits them. The default
`PLAN=compact` is 7 jobs plus a summary job, all on 1 GPU, each sized to finish in about 2–3 h, so they fit
a 4 h backfill walltime and the 9-jobs-per-user limit on Swing:

| Job | Runs |
|---|---|
| `main-CESM-ATM-3D` | `exp_main.py` on CESM-ATM-3D |
| `main-rest` | `exp_main.py` on all other datasets |
| `baselines-CESM-ATM-3D-abs` / `-rel` | `exp_baselines.py` on CESM-ATM-3D, absolute / relative bounds |
| `baselines-2D-Hurricane` | `exp_baselines.py` on CESM-ATM-2D and Hurricane |
| `baselines-rest` | `exp_baselines.py` on NYX and SCALE-LETKF |
| `other` | `exp_scaling.py`, `exp_bitrate.py`, `exp_pl.py`, `exp_vis.py`, one after another |
| `summarize` | `summarize.py`, after all others have ended |

`PLAN=separate` submits one job per experiment instead (main, baselines and scaling one per dataset; 17
jobs). If the tools are not built, a `setup` job runs first. Each experiment writes `<experiment>.<tag>.csv`,
`run.<tag>.log`, `logs/<tag>/` and its own work folder, so jobs can run at the same time; `summarize.py`
merges the CSVs.

```bash
# on Swing with the allocation used up
QUEUE=backfill DATA=$DATA experiments/sdrbench/submit_separate.sh $OUT
# DRYRUN=1 only writes the job scripts and prints the qsub lines
DRYRUN=1 QUEUE=backfill DATA=$DATA experiments/sdrbench/submit_separate.sh $OUT
```

Run the same command again at any time: jobs that are queued, running or finished are skipped, and
the rest is submitted. A job that hit its walltime is submitted again and resumes (finished runs are
skipped, also those finished by an earlier single-job run). If qsub reports the per-user job limit, the
jobs that did not fit are listed: run the command again later, or add `WAIT=1` to keep trying every
10 minutes (inside `screen`/`tmux`, or with `nohup`, on the login node). `main` and `scaling` measure time: with
`PLACE=excl` they get a whole node, so other jobs on the node do not disturb the timings.

Any experiment also runs by hand, e.g. on an interactive GPU node:

```bash
python3 experiments/sdrbench/exp_main.py --data $DATA --out $OUT --datasets CESM-ATM-3D --tag main-CESM-ATM-3D
python3 experiments/sdrbench/exp_baselines.py --help     # options of one experiment
python3 experiments/sdrbench/summarize.py $OUT
```
Use a different `--tag` for jobs that run at the same time in the same `RESULTS`.

Or interactively on a GPU node (`qsub -I -l select=1:ngpus=1 -l walltime=04:00:00 -q gpu -A SDR`,
then `module load gcc/11.4.0 nvhpc cmake`; gcc 13, the default, is too new for nvcc):

```bash
experiments/sdrbench/setup.sh                       # once
DATA=/lcrc/project/ECP-EZ/tripti/sdrbench-data experiments/sdrbench/run_all.sh /lcrc/project/ECP-EZ/tripti/toposzp-results
```

A short trial first (2 fields per dataset, one bound, few minutes), all experiments in one process:

```bash
python3 experiments/sdrbench/run_experiments.py --data $DATA --out trial --bin experiments/sdrbench/bin \
        --fields-per-dataset 2 --bounds rel:1e-3 --baseline-bounds rel:1e-3 --bitrate-bounds abs:1e-2,abs:1e-3 \
        --scaling-fields 1 --pl-fields 1 --reps 1
python3 experiments/sdrbench/summarize.py trial
```

## Tables and numbers for the paper

```bash
python3 experiments/sdrbench/paper_tables.py $OUT            # writes $OUT/paper/
```

writes ready-to-`\input` tabulars (`datasets`, `preservation`, `time`, `scaling`, `false_cases`, `pl`), the
bit-rate plots, and `numbers.tex` with `\resdef{key}{value}` for every number quoted in the text. Copy
`$OUT/paper/` into the paper as `results/` and recompile; missing values print as `--` / `??`, so the paper
compiles with partial results. Options: `--bounds` (default `rel:1e-3,rel:1e-4,rel:1e-5`), `--time-bound`,
`--pl-bound`.

## Experiments and the paper

| Script (`--exp`) | Paper | Output |
|---|---|---|
| `exp_main.py` (`main`) | Dataset table (error bound ε_topo), topology preservation table (plain SZp → TopoSZp+), 3D/GPU section, CPU vs GPU timing | `main.csv` |
| `exp_scaling.py` (`scaling`) | Parallel scalability table (threads 1…cores) | `scaling.csv` |
| `exp_baselines.py` (`baselines`) | Average false cases table (FN/FP/FT per dataset, bound, compressor) | `baselines.csv` |
| `exp_bitrate.py` (`bitrate`) | Bit rate vs FN/FP/FT/total figure | `bitrate.csv`, `plots/bitrate_*.pdf` |
| `exp_pl.py` (`pl`) | Relation to TTK's piecewise-linear critical points (reviewer question) | `pl.csv` |
| `exp_vis.py` (`vis`) | Visualization figure (original / SZp / TopoSZp+ of CLDHGH) | `vis/*.f32`, `*.vti` |

Defaults (change with options, `--help` lists all):
- bounds `abs:1e-3,abs:1e-4,abs:1e-5,rel:1e-3,rel:1e-4,rel:1e-5` (rel = fraction of each field's value range);
- OpenMP threads = all cores; 3 timed repetitions (minimum reported), GPU program once (`--gpu-reps`);
- scaling: 5 fields per dataset at `rel:1e-3`, threads 1, 2, 4, … up to the core count;
- bit-rate sweep: CESM-ATM 2D, 9 absolute bounds from 1e-2 to 1e-6;
- PL check: 3 fields per dataset at `rel:1e-3`.

## What is measured, exactly

- Critical points: the paper's definition (4 face neighbors in 2D, 6 in 3D, strict comparisons), counted by
  `topo_check` for every compressor. FN = critical point that became regular, FT = critical point with another
  type, FP = new critical point at a regular point; each point is counted once.
- "Before" = plain SZp quantization (same quantizer as TopoSZp+, no restoration), "after" = TopoSZp+ output.
- TopoSZp+ CPU times: compression = critical points + ranks + encoding; decompression = decoding + topology
  restoration. GPU times are the `TIMING` line of the CUDA programs (same steps). Plain SZp times are reported
  separately, so the overhead of topology preservation can be stated.
- Compression ratio of TopoSZp+ is the size of the full stream (data, labels, ranks, footer).
- `cpu_gpu_identical` in `main.csv`: the OpenMP and CUDA outputs are compared byte by byte.
- A bound is skipped (status `format_limit`) when max|value| / bound ≥ 2³¹ (the 32-bit bin index of the format);
  such runs are listed but never averaged. Use the relative bounds for those fields (e.g. NYX).
- Baselines get the same absolute bound; SZ3/SZ2/ZFP are given the dimensions fastest-varying first.
  TTHRESH's `-e` is a relative (Frobenius) error target, not a pointwise bound, so it is off by default.

## Runtime and disk

Roughly, on one A100 node with 32 cores: `main` ≈ 5 h (dominated by the 33 CESM 3D fields × 6 bounds),
`baselines` ≈ 4–5 h, `scaling` ≈ 3 h, `bitrate` ≈ 1 h, `pl` ≈ 0.5 h. With `submit_separate.sh` the longest
single job is the CESM 3D part of `main` or `baselines`; every run resumes. The data take ≈ 35 GB; the work folder needs ≈ 3 GB at a time.
