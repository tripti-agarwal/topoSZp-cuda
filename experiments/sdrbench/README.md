# TopoSZp+ experiments on SDRBench

Scripts to download the SDRBench datasets and run every experiment of the paper
with the current TopoSZp+ code (OpenMP and CUDA), from scratch, on one GPU node.

| Script | What it does |
|---|---|
| `download_sdrbench.sh` | Downloads CESM-ATM 2D (1800×3600), CESM-ATM 3D (26×1800×3600), Hurricane-ISABEL (100×500×500), NYX (512³) and SCALE-LETKF (98×1200×1200), checks every file's size, writes `manifest.csv`. Can link data you already have instead. |
| `setup.sh` | Builds the TopoSZp+ OpenMP compressor/decompressor, the CUDA programs, `topo_check`, `pl_topo_check`, a plain-SZp tool, and the baselines SZ3, SZ2 and ZFP (TTHRESH with `TTHRESH=1`). |
| `run_experiments.py` | Runs the experiments; one CSV per experiment, resumable. |
| `summarize.py` | Makes `summary.md`, `tables.tex` and `plots/` from the CSVs. |
| `run_all.sh`, `run_all.pbs` | Everything with one command / as a PBS job on Swing. |

## Quick start (Swing)

```bash
cd /lcrc/project/ECP-EZ/tripti/topoSZp-cuda          # the repository
git fetch && git checkout claude/sdrbench-experiments

# 1. data (≈35 GB). Link CESM if you already have it, download the rest:
CESM_ATM_2D_SRC=/lcrc/project/ECP-EZ/tripti/SDRBENCH-CESM-ATM-cleared-1800x3600/original_data \
CESM_ATM_3D_SRC=/lcrc/project/ECP-EZ/public/compression/cesmbig-26x1800x3600 \
experiments/sdrbench/download_sdrbench.sh /lcrc/project/ECP-EZ/tripti/sdrbench-data

# 2. everything, as a batch job (resubmit the same line if it hits the walltime; it resumes)
qsub -v DATA=/lcrc/project/ECP-EZ/tripti/sdrbench-data,OUT=/lcrc/project/ECP-EZ/tripti/toposzp-results \
     experiments/sdrbench/run_all.pbs
```

Or interactively on a GPU node (`qsub -I -l select=1:ngpus=1 -l walltime=04:00:00 -q gpu -A SDR`,
then `module load nvhpc`):

```bash
experiments/sdrbench/setup.sh                       # once
DATA=/lcrc/project/ECP-EZ/tripti/sdrbench-data experiments/sdrbench/run_all.sh /lcrc/project/ECP-EZ/tripti/toposzp-results
```

A short trial first (2 fields per dataset, one bound, few minutes):

```bash
python3 experiments/sdrbench/run_experiments.py --data $DATA --out trial --bin experiments/sdrbench/bin \
        --fields-per-dataset 2 --bounds rel:1e-3 --baseline-bounds rel:1e-3 --bitrate-bounds abs:1e-2,abs:1e-3 \
        --scaling-fields 1 --pl-fields 1 --reps 1
python3 experiments/sdrbench/summarize.py trial
```

## Experiments and the paper

| `--exp` | Paper | Output |
|---|---|---|
| `main` | Dataset table (error bound ε_topo), topology preservation table (plain SZp → TopoSZp+), 3D/GPU section, CPU vs GPU timing | `main.csv` |
| `scaling` | Parallel scalability table (threads 1…cores) | `scaling.csv` |
| `baselines` | Average false cases table (FN/FP/FT per dataset, bound, compressor) | `baselines.csv` |
| `bitrate` | Bit rate vs FN/FP/FT/total figure | `bitrate.csv`, `plots/bitrate_*.pdf` |
| `pl` | Relation to TTK's piecewise-linear critical points (reviewer question) | `pl.csv` |
| `vis` | Visualization figure (original / SZp / TopoSZp+ of CLDHGH) | `vis/*.f32`, `*.vti` |

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
`baselines` ≈ 4–5 h, `scaling` ≈ 3 h, `bitrate` ≈ 1 h, `pl` ≈ 0.5 h. Run them as separate jobs with `EXPS=...`
if the walltime is short; every run resumes. The data take ≈ 35 GB; the work folder needs ≈ 3 GB at a time.
