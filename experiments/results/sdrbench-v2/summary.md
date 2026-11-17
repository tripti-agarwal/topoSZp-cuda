# TopoSZp+ experiment summary

```
date: 2026-11-16T20:44:55.632617
host: gpu1
cpu: AMD EPYC 7742 64-Core Processor
CPUs usable by this job: 32 logical, 16 physical cores (node: 256 logical)
OMP_PROC_BIND=close OMP_PLACES=cores
sockets / NUMA: Socket(s): 2;NUMA node(s): 8;
memory: 1007 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 580.178.04
nvcc: Build cuda_12.6.r12.6/compiler.34431801_0
gcc: g++ (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0
repo: 323a0ae (claude/sdrbench-experiments)
settings: threads=16 block=64 reps=3 gpu_reps=1
```

## Datasets

| Dataset | Fields | Dimensions | Size | Mean max error / eps (abs 1e-3) | Largest max error / eps |
|---|---|---|---|---|---|
| CESM-ATM-3D | 33 | 26 × 1800 × 3600 | 22.24 GB | 1.437 | 1.999 |
| CESM-ATM-2D | 60 | 1800 × 3600 | 1.56 GB | 1.467 | 1.761 |
| Hurricane | 20 | 100 × 500 × 500 | 2.00 GB | 1.727 | 1.998 |
| NYX | 6 | 512 × 512 × 512 | 3.22 GB | 1.996 | 1.996 |
| SCALE-LETKF | 12 | 98 × 1200 × 1200 | 6.77 GB | 1.528 | 1.998 |

Max error of TopoSZp+ in units of the error bound eps (the guarantee is < 2).

## Topology preservation: plain SZp (before) -> TopoSZp+ (after)

| Dataset | Bound | Fields | Ratio TopoSZp+ | Ratio SZp | All CPs kept % | Extrema kept % | Saddles kept % | FN per field | FP | FT | Max err / eps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CESM-ATM-3D | abs 1e-3 | 33 | 7.80× | 26.92× | 34.8 -> 54.2 | 33.6 -> 100.0 | 34.7 -> 47.9 | 2,571,921 -> 1,947,595 | 0 | 0 | 1.999 |
| CESM-ATM-3D | abs 1e-4 | 33 | 7.15× | 24.11× | 42.7 -> 58.5 | 41.4 -> 100.0 | 42.6 -> 52.7 | 2,357,160 -> 1,796,920 | 0 | 0 | 1.998 |
| CESM-ATM-3D | abs 1e-5 | 32 (+1 skipped) | 6.54× | 19.83× | 46.8 -> 62.5 | 47.3 -> 100.0 | 46.7 -> 57.2 | 2,242,170 -> 1,620,442 | 0 | 0 | 1.999 |
| CESM-ATM-3D | rel 1e-3 | 33 | 8.17× | 17.92× | 8.3 -> 64.2 | 9.3 -> 100.0 | 8.2 -> 59.8 | 2,996,532 -> 1,316,835 | 0 | 0 | 1.999 |
| CESM-ATM-3D | rel 1e-4 | 33 | 6.46× | 11.48× | 29.3 -> 79.4 | 30.7 -> 100.0 | 29.2 -> 76.8 | 2,170,580 -> 842,289 | 0 | 0 | 1.999 |
| CESM-ATM-3D | rel 1e-5 | 33 | 4.89× | 7.34× | 54.3 -> 87.1 | 56.1 -> 100.0 | 54.2 -> 85.4 | 1,382,777 -> 550,013 | 0 | 0 | 1.998 |
| CESM-ATM-2D | abs 1e-3 | 60 | 5.04× | 10.39× | 61.1 -> 95.2 | 62.9 -> 100.0 | 52.9 -> 75.4 | 17,335 -> 1,484 | 0 | 0 | 1.761 |
| CESM-ATM-2D | abs 1e-4 | 60 | 4.04× | 6.97× | 74.7 -> 96.8 | 75.9 -> 100.0 | 69.7 -> 84.0 | 8,146 -> 625 | 0 | 0 | 1.758 |
| CESM-ATM-2D | abs 1e-5 | 60 | 3.24× | 4.84× | 83.7 -> 98.1 | 84.5 -> 100.0 | 80.4 -> 90.6 | 3,420 -> 235 | 0 | 0 | 1.758 |
| CESM-ATM-2D | rel 1e-3 | 60 | 6.44× | 11.56× | 19.3 -> 91.3 | 22.2 -> 100.0 | 7.3 -> 54.7 | 39,375 -> 3,013 | 0 | 0 | 1.763 |
| CESM-ATM-2D | rel 1e-4 | 60 | 4.89× | 7.51× | 55.6 -> 97.2 | 59.1 -> 100.0 | 40.9 -> 84.8 | 15,449 -> 467 | 0 | 0 | 1.755 |
| CESM-ATM-2D | rel 1e-5 | 60 | 3.66× | 4.98× | 81.9 -> 99.3 | 83.8 -> 100.0 | 73.8 -> 96.6 | 3,810 -> 70 | 0 | 0 | 1.754 |
| Hurricane | abs 1e-3 | 20 | 6.29× | 20.87× | 56.6 -> 71.5 | 57.8 -> 100.0 | 56.3 -> 61.3 | 259,152 -> 171,984 | 0 | 0 | 1.998 |
| Hurricane | abs 1e-4 | 20 | 5.56× | 15.45× | 60.7 -> 75.2 | 62.2 -> 100.0 | 59.9 -> 66.4 | 234,228 -> 164,018 | 0 | 0 | 1.992 |
| Hurricane | abs 1e-5 | 20 | 4.76× | 10.48× | 65.1 -> 79.6 | 67.2 -> 100.0 | 64.1 -> 72.7 | 224,111 -> 153,611 | 0 | 0 | 1.993 |
| Hurricane | rel 1e-3 | 20 | 5.98× | 11.31× | 33.2 -> 78.4 | 44.2 -> 100.0 | 30.5 -> 71.6 | 430,066 -> 169,177 | 0 | 0 | 1.998 |
| Hurricane | rel 1e-4 | 20 | 4.78× | 7.90× | 59.8 -> 82.9 | 67.9 -> 100.0 | 58.2 -> 77.3 | 286,556 -> 145,121 | 0 | 0 | 1.997 |
| Hurricane | rel 1e-5 | 20 | 3.95× | 5.94× | 73.4 -> 85.0 | 76.2 -> 100.0 | 72.7 -> 79.8 | 213,932 -> 134,647 | 0 | 0 | 1.997 |
| NYX | abs 1e-3 | 2 (+4 skipped) | 2.20× | 2.74× | 81.2 -> 99.9 | 84.1 -> 100.0 | 80.8 -> 99.9 | 473,300 -> 1,346 | 0 | 0 | 1.996 |
| NYX | abs 1e-4 | 2 (+4 skipped) | 1.80× | 2.13× | 97.2 -> 100.0 | 97.4 -> 100.0 | 97.2 -> 100.0 | 64,179 -> 38 | 0 | 0 | 1.991 |
| NYX | abs 1e-5 | 1 (+5 skipped) | 1.37× | 1.61× | 100.0 -> 100.0 | 100.0 -> 100.0 | 100.0 -> 100.0 | 6,029 -> 1 | 0 | 0 | 1.928 |
| NYX | rel 1e-3 | 6 | 6.70× | 19.92× | 1.3 -> 54.9 | 6.3 -> 100.0 | 0.4 -> 43.7 | 8,060,859 -> 2,775,151 | 0 | 0 | 1.988 |
| NYX | rel 1e-4 | 6 | 5.19× | 12.12× | 12.9 -> 74.6 | 26.8 -> 100.0 | 10.3 -> 67.5 | 7,814,166 -> 2,218,003 | 0 | 0 | 1.998 |
| NYX | rel 1e-5 | 6 | 3.64× | 5.63× | 51.4 -> 86.3 | 60.7 -> 100.0 | 48.8 -> 83.9 | 4,776,423 -> 281,122 | 0 | 0 | 1.997 |
| SCALE-LETKF | abs 1e-3 | 12 | 7.17× | 24.72× | 41.4 -> 53.6 | 33.2 -> 100.0 | 41.5 -> 50.7 | 1,240,405 -> 976,284 | 0 | 0 | 1.998 |
| SCALE-LETKF | abs 1e-4 | 12 | 6.64× | 21.84× | 48.6 -> 57.3 | 43.0 -> 100.0 | 48.7 -> 54.6 | 1,082,865 -> 909,432 | 0 | 0 | 1.995 |
| SCALE-LETKF | abs 1e-5 | 11 (+1 skipped) | 6.29× | 17.35× | 46.8 -> 58.8 | 47.0 -> 100.0 | 46.8 -> 56.2 | 1,123,336 -> 887,804 | 0 | 0 | 1.998 |
| SCALE-LETKF | rel 1e-3 | 12 | 7.86× | 18.02× | 3.6 -> 59.5 | 4.5 -> 100.0 | 3.6 -> 57.0 | 1,948,128 -> 893,340 | 0 | 0 | 1.997 |
| SCALE-LETKF | rel 1e-4 | 12 | 6.17× | 11.75× | 22.9 -> 70.3 | 22.1 -> 100.0 | 23.0 -> 68.4 | 1,541,726 -> 654,103 | 0 | 0 | 1.998 |
| SCALE-LETKF | rel 1e-5 | 12 | 4.82× | 7.85× | 49.9 -> 75.4 | 47.1 -> 100.0 | 50.1 -> 73.8 | 1,010,954 -> 547,365 | 0 | 0 | 1.998 |

Percentages and FN are means over fields; FP and FT are totals over fields. "skipped": max|value|/eps exceeds the 32-bit bin index of the format (use a relative bound for these fields).

## Compression and decompression time: OpenMP (16 threads) and CUDA

| Dataset | Bound | Fields | CPU comp s | GB/s | CPU decomp s | GB/s | GPU comp s | GB/s | GPU decomp s | GB/s | CPU = GPU output |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CESM-ATM-3D | abs 1e-3 | 33 | 0.6669 | 1.01 | 2.1371 | 0.32 | 0.2754 | 2.45 | 0.4676 | 1.44 | 33/33 |
| CESM-ATM-3D | abs 1e-4 | 33 | 0.6955 | 0.97 | 2.1149 | 0.32 | 0.3355 | 2.01 | 0.4779 | 1.41 | 33/33 |
| CESM-ATM-3D | abs 1e-5 | 32 | 0.7316 | 0.92 | 2.0813 | 0.32 | 0.3254 | 2.07 | 0.4867 | 1.38 | 32/32 |
| CESM-ATM-3D | rel 1e-3 | 33 | 0.6593 | 1.02 | 2.2540 | 0.30 | 0.2617 | 2.58 | 0.5058 | 1.33 | 33/33 |
| CESM-ATM-3D | rel 1e-4 | 33 | 0.6927 | 0.97 | 1.9690 | 0.34 | 0.2671 | 2.52 | 0.4836 | 1.39 | 33/33 |
| CESM-ATM-3D | rel 1e-5 | 33 | 0.7133 | 0.94 | 1.8143 | 0.37 | 0.2852 | 2.36 | 0.4436 | 1.52 | 33/33 |
| CESM-ATM-2D | abs 1e-3 | 60 | 0.1273 | 0.20 | 0.1600 | 0.16 | 0.0181 | 1.44 | 0.0317 | 0.82 | 60/60 |
| CESM-ATM-2D | abs 1e-4 | 60 | 0.1324 | 0.20 | 0.1616 | 0.16 | 0.0197 | 1.31 | 0.0316 | 0.82 | 60/60 |
| CESM-ATM-2D | abs 1e-5 | 60 | 0.1343 | 0.19 | 0.1647 | 0.16 | 0.0205 | 1.27 | 0.0309 | 0.84 | 60/60 |
| CESM-ATM-2D | rel 1e-3 | 60 | 0.1283 | 0.20 | 0.1612 | 0.16 | 0.0151 | 1.71 | 0.0339 | 0.76 | 60/60 |
| CESM-ATM-2D | rel 1e-4 | 60 | 0.1291 | 0.20 | 0.1621 | 0.16 | 0.0162 | 1.60 | 0.0320 | 0.81 | 60/60 |
| CESM-ATM-2D | rel 1e-5 | 60 | 0.1292 | 0.20 | 0.1620 | 0.16 | 0.0189 | 1.37 | 0.0309 | 0.84 | 60/60 |
| Hurricane | abs 1e-3 | 20 | 0.2233 | 0.45 | 0.4042 | 0.25 | 0.0545 | 1.83 | 0.1119 | 0.89 | 20/20 |
| Hurricane | abs 1e-4 | 20 | 0.2321 | 0.43 | 0.4011 | 0.25 | 0.0588 | 1.70 | 0.1137 | 0.88 | 20/20 |
| Hurricane | abs 1e-5 | 20 | 0.2324 | 0.43 | 0.4072 | 0.25 | 0.0642 | 1.56 | 0.1112 | 0.90 | 20/20 |
| Hurricane | rel 1e-3 | 20 | 0.2203 | 0.45 | 0.4346 | 0.23 | 0.0494 | 2.02 | 0.1696 | 0.59 | 20/20 |
| Hurricane | rel 1e-4 | 20 | 0.2262 | 0.44 | 0.4058 | 0.25 | 0.0560 | 1.78 | 0.1351 | 0.74 | 20/20 |
| Hurricane | rel 1e-5 | 20 | 0.2311 | 0.43 | 0.3903 | 0.26 | 0.0590 | 1.70 | 0.1075 | 0.93 | 20/20 |
| NYX | abs 1e-3 | 2 | 1.9661 | 0.27 | 4.0110 | 0.13 | 0.3721 | 1.44 | 0.5014 | 1.07 | 2/2 |
| NYX | abs 1e-4 | 2 | 2.0674 | 0.26 | 3.9402 | 0.14 | 0.3819 | 1.41 | 0.4638 | 1.16 | 2/2 |
| NYX | rel 1e-3 | 6 | 0.9855 | 0.54 | 4.2228 | 0.13 | 0.2019 | 2.66 | 0.7712 | 0.70 | 6/6 |
| NYX | rel 1e-4 | 6 | 1.0060 | 0.53 | 4.1417 | 0.13 | 0.2451 | 2.19 | 0.7502 | 0.72 | 6/6 |
| NYX | rel 1e-5 | 6 | 1.0784 | 0.50 | 2.6402 | 0.20 | 0.2622 | 2.05 | 0.4870 | 1.10 | 6/6 |
| NYX | abs 1e-5 | 1 | 3.4038 | 0.16 | 6.5519 | 0.08 | 0.5082 | 1.06 | 0.4697 | 1.14 | 1/1 |
| SCALE-LETKF | abs 1e-3 | 12 | 0.6218 | 0.91 | 1.6331 | 0.35 | 0.2300 | 2.45 | 0.3906 | 1.45 | 12/12 |
| SCALE-LETKF | abs 1e-4 | 12 | 0.6610 | 0.85 | 1.5787 | 0.36 | 0.2567 | 2.20 | 0.3801 | 1.48 | 12/12 |
| SCALE-LETKF | rel 1e-3 | 12 | 0.5940 | 0.95 | 1.8596 | 0.30 | 0.1872 | 3.02 | 0.4219 | 1.34 | 12/12 |
| SCALE-LETKF | rel 1e-4 | 12 | 0.6137 | 0.92 | 1.6881 | 0.33 | 0.2063 | 2.74 | 0.4320 | 1.31 | 12/12 |
| SCALE-LETKF | rel 1e-5 | 12 | 0.6514 | 0.87 | 1.5264 | 0.37 | 0.2792 | 2.02 | 0.3951 | 1.43 | 12/12 |
| SCALE-LETKF | abs 1e-5 | 11 | 0.6839 | 0.83 | 1.6295 | 0.35 | 0.2755 | 2.05 | 0.4109 | 1.37 | 11/11 |

Means per field. Compression = critical points + ranks + encoding; decompression = decoding + topology restoration. GPU times are those reported by the CUDA test programs.

## OpenMP scalability: mean compression time per field (s), bound rel 1e-3

| Dataset | Fields | 1 t | 2 t | 4 t | 8 t | 16 t | Best speedup | Decomp speedup 16 t | Identical output |
|---|---|---|---|---|---|---|---|---|---|
| CESM-ATM-2D | 5 | 0.2705 | 0.1891 | 0.1379 | 0.1176 | 0.1231 | 2.3× (8 t) | 1.9× | yes |
| CESM-ATM-3D | 5 | 4.8426 | 2.9226 | 1.6790 | 1.0090 | 0.6764 | 7.2× (16 t) | 6.2× | yes |
| Hurricane | 5 | 0.9409 | 0.5701 | 0.3614 | 0.2472 | 0.2003 | 4.7× (16 t) | 5.1× | yes |
| NYX | 5 | 6.4755 | 3.6213 | 2.0756 | 1.3252 | 0.9377 | 6.9× (16 t) | 8.9× | yes |
| SCALE-LETKF | 5 | 3.7409 | 2.1428 | 1.2176 | 0.7363 | 0.4895 | 7.6× (16 t) | 6.1× | yes |

Full pipeline (critical points, ranks, encoding). Identical output = the decompressed field is the same for every thread count.

## Compressor comparison

| Dataset | Bound | Compressor | Fields | Bits/value | Ratio | FN | FP | FT | Total | Max err / eps |
|---|---|---|---|---|---|---|---|---|---|---|
| CESM-ATM-2D | abs 1e-3 | TopoSZp+ | 60 | 8.38 | 5.04× | 1,484 | 0 | 0 | 1,484 | 1.761 |
| CESM-ATM-2D | abs 1e-3 | SZp | 60 | 6.34 | 10.39× | 17,335 | 0 | 0 | 17,335 | 1.000 |
| CESM-ATM-2D | abs 1e-3 | SZ3 | 60 | 3.49 | 885.55× | 14,257 | 20,562 | 458 | 35,277 | 1.000 |
| CESM-ATM-2D | abs 1e-3 | SZ2 | 60 | 3.68 | 243.54× | 15,027 | 63,243 | 904 | 79,174 | 1.000 |
| CESM-ATM-2D | abs 1e-3 | ZFP | 60 | 7.40 | 10.52× | 7,644 | 12,060 | 198 | 19,902 | 0.465 |
| CESM-ATM-2D | abs 1e-3 | LOPC | 60 | 6.92 | 5.99× | 8,850 | 23 | 0 | 8,874 | 0.975 |
| CESM-ATM-2D | abs 1e-3 | LOPC (GPU) | 60 | 6.92 | 5.99× | 8,850 | 23 | 0 | 8,874 | 0.975 |
| CESM-ATM-2D | abs 1e-4 | TopoSZp+ | 60 | 10.73 | 4.04× | 625 | 0 | 0 | 625 | 1.758 |
| CESM-ATM-2D | abs 1e-4 | SZp | 60 | 8.70 | 6.97× | 8,146 | 0 | 0 | 8,146 | 1.000 |
| CESM-ATM-2D | abs 1e-4 | SZ3 | 60 | 5.62 | 177.59× | 6,632 | 9,958 | 153 | 16,743 | 1.000 |
| CESM-ATM-2D | abs 1e-4 | SZ2 | 60 | 5.91 | 55.89× | 6,847 | 41,554 | 376 | 48,777 | 1.000 |
| CESM-ATM-2D | abs 1e-4 | ZFP | 60 | 10.41 | 6.20× | 2,605 | 7,259 | 72 | 9,936 | 0.305 |
| CESM-ATM-2D | abs 1e-4 | LOPC | 60 | 9.23 | 4.82× | 3,504 | 14 | 0 | 3,518 | 0.606 |
| CESM-ATM-2D | abs 1e-4 | LOPC (GPU) | 60 | 9.23 | 4.82× | 3,504 | 14 | 0 | 3,518 | 0.606 |
| CESM-ATM-2D | abs 1e-5 | TopoSZp+ | 60 | 13.27 | 3.24× | 235 | 0 | 0 | 235 | 1.758 |
| CESM-ATM-2D | abs 1e-5 | SZp | 60 | 11.25 | 4.84× | 3,420 | 0 | 0 | 3,420 | 1.000 |
| CESM-ATM-2D | abs 1e-5 | SZ3 | 60 | 8.63 | 37.68× | 2,647 | 6,127 | 88 | 8,861 | 1.000 |
| CESM-ATM-2D | abs 1e-5 | SZ2 | 60 | 9.20 | 18.19× | 2,762 | 23,064 | 155 | 25,980 | 1.000 |
| CESM-ATM-2D | abs 1e-5 | ZFP | 60 | 12.79 | 4.49× | 1,111 | 5,787 | 34 | 6,932 | 0.382 |
| CESM-ATM-2D | abs 1e-5 | LOPC | 60 | 10.70 | 4.20× | 1,568 | 6 | 0 | 1,575 | 0.763 |
| CESM-ATM-2D | abs 1e-5 | LOPC (GPU) | 60 | 10.70 | 4.20× | 1,568 | 6 | 0 | 1,575 | 0.763 |
| CESM-ATM-2D | rel 1e-3 | TopoSZp+ | 60 | 5.17 | 6.44× | 3,013 | 0 | 0 | 3,013 | 1.763 |
| CESM-ATM-2D | rel 1e-3 | SZp | 60 | 3.10 | 11.56× | 39,375 | 0 | 0 | 39,375 | 1.000 |
| CESM-ATM-2D | rel 1e-3 | SZ3 | 60 | 0.55 | 273.33× | 31,672 | 59,385 | 1,262 | 92,319 | 1.000 |
| CESM-ATM-2D | rel 1e-3 | SZ2 | 60 | 0.86 | 87.74× | 33,011 | 220,196 | 2,439 | 255,645 | 1.000 |
| CESM-ATM-2D | rel 1e-3 | ZFP | 60 | 4.14 | 9.55× | 10,959 | 23,169 | 468 | 34,596 | 0.404 |
| CESM-ATM-2D | rel 1e-3 | LOPC | 60 | 4.41 | 8.08× | 17,330 | 24 | 0 | 17,354 | 0.491 |
| CESM-ATM-2D | rel 1e-3 | LOPC (GPU) | 60 | 4.41 | 8.08× | 17,330 | 24 | 0 | 17,354 | 0.491 |
| CESM-ATM-2D | rel 1e-4 | TopoSZp+ | 60 | 7.01 | 4.89× | 467 | 0 | 0 | 467 | 1.755 |
| CESM-ATM-2D | rel 1e-4 | SZp | 60 | 4.96 | 7.51× | 15,449 | 0 | 0 | 15,449 | 1.000 |
| CESM-ATM-2D | rel 1e-4 | SZ3 | 60 | 1.92 | 56.99× | 11,126 | 23,178 | 446 | 34,750 | 1.000 |
| CESM-ATM-2D | rel 1e-4 | SZ2 | 60 | 2.26 | 23.47× | 11,501 | 85,580 | 1,134 | 98,215 | 1.000 |
| CESM-ATM-2D | rel 1e-4 | ZFP | 60 | 6.37 | 6.24× | 2,697 | 8,757 | 106 | 11,561 | 0.426 |
| CESM-ATM-2D | rel 1e-4 | LOPC | 60 | 5.21 | 6.91× | 6,425 | 23 | 0 | 6,448 | 0.967 |
| CESM-ATM-2D | rel 1e-4 | LOPC (GPU) | 60 | 5.21 | 6.91× | 6,425 | 23 | 0 | 6,448 | 0.967 |
| CESM-ATM-2D | rel 1e-5 | TopoSZp+ | 60 | 9.48 | 3.66× | 70 | 0 | 0 | 70 | 1.754 |
| CESM-ATM-2D | rel 1e-5 | SZp | 60 | 7.45 | 4.98× | 3,810 | 0 | 0 | 3,810 | 1.000 |
| CESM-ATM-2D | rel 1e-5 | SZ3 | 60 | 4.12 | 15.51× | 2,699 | 6,154 | 73 | 8,926 | 1.000 |
| CESM-ATM-2D | rel 1e-5 | SZ2 | 60 | 4.40 | 10.72× | 2,837 | 21,852 | 207 | 24,896 | 1.000 |
| CESM-ATM-2D | rel 1e-5 | ZFP | 60 | 9.07 | 4.36× | 559 | 4,694 | 19 | 5,272 | 0.452 |
| CESM-ATM-2D | rel 1e-5 | LOPC | 60 | 7.31 | 5.12× | 1,526 | 20 | 0 | 1,546 | 0.992 |
| CESM-ATM-2D | rel 1e-5 | LOPC (GPU) | 60 | 7.31 | 5.12× | 1,526 | 20 | 0 | 1,546 | 0.992 |
| Hurricane | abs 1e-3 | TopoSZp+ | 20 | 8.33 | 6.29× | 171,984 | 0 | 0 | 171,984 | 1.998 |
| Hurricane | abs 1e-3 | SZp | 20 | 6.25 | 20.87× | 259,152 | 0 | 0 | 259,152 | 1.000 |
| Hurricane | abs 1e-3 | SZ3 | 20 | 3.33 | 11329.80× | 250,084 | 94,689 | 130 | 344,903 | 1.000 |
| Hurricane | abs 1e-3 | SZ2 | 20 | 3.07 | 228460.76× | 234,525 | 1,109,496 | 4,750 | 1,348,772 | 1.000 |
| Hurricane | abs 1e-3 | ZFP | 20 | 5.71 | 178.42× | 228,458 | 198,626 | 479 | 427,562 | 0.319 |
| Hurricane | abs 1e-3 | LOPC | 20 | 7.13 | 10.59× | 126,986 | 3 | 0 | 126,989 | 0.855 |
| Hurricane | abs 1e-3 | LOPC (GPU) | 20 | 7.13 | 10.59× | 126,986 | 3 | 0 | 126,989 | 0.855 |
| Hurricane | abs 1e-4 | TopoSZp+ | 20 | 10.07 | 5.56× | 164,018 | 0 | 0 | 164,018 | 1.992 |
| Hurricane | abs 1e-4 | SZp | 20 | 8.00 | 15.45× | 234,228 | 0 | 0 | 234,228 | 1.000 |
| Hurricane | abs 1e-4 | SZ3 | 20 | 4.79 | 1273.13× | 230,720 | 40,295 | 446 | 271,462 | 1.000 |
| Hurricane | abs 1e-4 | SZ2 | 20 | 4.76 | 414.45× | 211,797 | 1,212,914 | 5,508 | 1,430,218 | 1.000 |
| Hurricane | abs 1e-4 | ZFP | 20 | 7.76 | 52.26× | 212,665 | 163,890 | 1,316 | 377,872 | 1.221 |
| Hurricane | abs 1e-4 | LOPC | 20 | 8.94 | 9.18× | 112,605 | 2 | 0 | 112,606 | 0.534 |
| Hurricane | abs 1e-4 | LOPC (GPU) | 20 | 8.94 | 9.18× | 112,605 | 2 | 0 | 112,606 | 0.534 |
| Hurricane | abs 1e-5 | TopoSZp+ | 20 | 11.92 | 4.76× | 153,611 | 0 | 0 | 153,611 | 1.993 |
| Hurricane | abs 1e-5 | SZp | 20 | 9.86 | 10.48× | 224,111 | 0 | 0 | 224,111 | 1.000 |
| Hurricane | abs 1e-5 | SZ3 | 20 | 6.56 | 153.17× | 218,743 | 69,299 | 962 | 289,004 | 1.000 |
| Hurricane | abs 1e-5 | SZ2 | 20 | 6.54 | 86.89× | 202,023 | 1,092,171 | 5,490 | 1,299,685 | 1.000 |
| Hurricane | abs 1e-5 | ZFP | 20 | 9.30 | 25.75× | 201,024 | 124,086 | 2,303 | 327,413 | 12.207 |
| Hurricane | abs 1e-5 | LOPC | 20 | 10.34 | 8.06× | 107,028 | 1 | 0 | 107,029 | 0.668 |
| Hurricane | abs 1e-5 | LOPC (GPU) | 20 | 10.34 | 8.06× | 107,028 | 1 | 0 | 107,029 | 0.668 |
| Hurricane | rel 1e-3 | TopoSZp+ | 20 | 6.01 | 5.98× | 169,177 | 0 | 0 | 169,177 | 1.998 |
| Hurricane | rel 1e-3 | SZp | 20 | 3.92 | 11.31× | 430,066 | 0 | 0 | 430,066 | 1.000 |
| Hurricane | rel 1e-3 | SZ3 | 20 | 0.96 | 95.39× | 382,194 | 309,555 | 4,046 | 695,795 | 1.000 |
| Hurricane | rel 1e-3 | SZ2 | 20 | 1.28 | 66.28× | 364,769 | 1,243,749 | 12,457 | 1,620,975 | 1.000 |
| Hurricane | rel 1e-3 | ZFP | 20 | 3.41 | 20.74× | 228,942 | 352,688 | 3,305 | 584,936 | 0.350 |
| Hurricane | rel 1e-3 | LOPC | 20 | 5.35 | 10.04× | 195,285 | 7 | 0 | 195,292 | 0.471 |
| Hurricane | rel 1e-3 | LOPC (GPU) | 20 | 5.35 | 10.04× | 195,285 | 7 | 0 | 195,292 | 0.471 |
| Hurricane | rel 1e-4 | TopoSZp+ | 20 | 7.86 | 4.78× | 145,121 | 0 | 0 | 145,121 | 1.997 |
| Hurricane | rel 1e-4 | SZp | 20 | 5.78 | 7.90× | 286,556 | 0 | 0 | 286,556 | 1.000 |
| Hurricane | rel 1e-4 | SZ3 | 20 | 2.33 | 36.90× | 259,588 | 196,008 | 1,395 | 456,992 | 1.000 |
| Hurricane | rel 1e-4 | SZ2 | 20 | 2.33 | 31.54× | 248,558 | 983,811 | 4,812 | 1,237,182 | 1.000 |
| Hurricane | rel 1e-4 | ZFP | 20 | 5.30 | 12.88× | 189,220 | 334,338 | 4,167 | 527,725 | 0.279 |
| Hurricane | rel 1e-4 | LOPC | 20 | 6.40 | 7.83× | 127,620 | 5 | 0 | 127,624 | 0.752 |
| Hurricane | rel 1e-4 | LOPC (GPU) | 20 | 6.40 | 7.83× | 127,620 | 5 | 0 | 127,624 | 0.752 |
| Hurricane | rel 1e-5 | TopoSZp+ | 20 | 9.85 | 3.95× | 134,647 | 0 | 0 | 134,647 | 1.997 |
| Hurricane | rel 1e-5 | SZp | 20 | 7.78 | 5.94× | 213,932 | 0 | 0 | 213,932 | 1.000 |
| Hurricane | rel 1e-5 | SZ3 | 20 | 4.05 | 21.51× | 204,248 | 122,262 | 1,050 | 327,560 | 1.000 |
| Hurricane | rel 1e-5 | SZ2 | 20 | 3.88 | 18.58× | 196,415 | 738,599 | 3,118 | 938,132 | 1.000 |
| Hurricane | rel 1e-5 | ZFP | 20 | 7.20 | 9.13× | 165,426 | 318,832 | 4,470 | 488,728 | 0.274 |
| Hurricane | rel 1e-5 | LOPC | 20 | 7.96 | 6.32× | 99,607 | 2 | 0 | 99,609 | 0.941 |
| Hurricane | rel 1e-5 | LOPC (GPU) | 20 | 7.96 | 6.32× | 99,607 | 2 | 0 | 99,609 | 0.941 |
| CESM-ATM-3D | abs 1e-3 | TopoSZp+ | 1 | 2.69 | 11.89× | 4,129,783 | 0 | 0 | 4,129,783 | 0.520 |
| CESM-ATM-3D | abs 1e-3 | SZp | 1 | 0.62 | 51.20× | 4,651,732 | 0 | 0 | 4,651,732 | 0.521 |
| CESM-ATM-3D | abs 1e-3 | SZ3 | 1 | 0.00 | 40154.92× | 4,639,426 | 351,147 | 1,528 | 4,992,101 | 0.934 |
| CESM-ATM-3D | abs 1e-3 | SZ2 | 1 | 0.00 | 15316363.64× | 4,651,732 | 0 | 0 | 4,651,732 | 0.521 |
| CESM-ATM-3D | abs 1e-3 | ZFP | 1 | 0.15 | 210.09× | 4,651,196 | 5,039 | 32 | 4,656,267 | 0.080 |
| CESM-ATM-3D | abs 1e-3 | LOPC | 1 | 6.33 | 5.06× | 2,465,976 | 49,772 | 0 | 2,515,748 | 0.488 |
| CESM-ATM-3D | abs 1e-3 | LOPC (GPU) | 1 | 6.33 | 5.06× | 2,465,976 | 49,772 | 0 | 2,515,748 | 0.488 |
| CESM-ATM-3D | abs 1e-4 | TopoSZp+ | 1 | 2.71 | 11.80× | 4,099,144 | 0 | 0 | 4,099,144 | 1.957 |
| CESM-ATM-3D | abs 1e-4 | SZp | 1 | 0.65 | 49.58× | 4,651,638 | 0 | 0 | 4,651,638 | 1.000 |
| CESM-ATM-3D | abs 1e-4 | SZ3 | 1 | 0.00 | 14035.32× | 4,650,816 | 15,561 | 143 | 4,666,520 | 1.000 |
| CESM-ATM-3D | abs 1e-4 | SZ2 | 1 | 0.01 | 3548.44× | 4,548,421 | 2,387,647 | 16,125 | 6,952,193 | 1.000 |
| CESM-ATM-3D | abs 1e-4 | ZFP | 1 | 0.55 | 58.41× | 4,408,463 | 1,240,459 | 16,254 | 5,665,176 | 0.116 |
| CESM-ATM-3D | abs 1e-4 | LOPC | 1 | 6.19 | 5.17× | 2,447,558 | 49,771 | 0 | 2,497,329 | 0.305 |
| CESM-ATM-3D | abs 1e-4 | LOPC (GPU) | 1 | 6.19 | 5.17× | 2,447,558 | 49,771 | 0 | 2,497,329 | 0.305 |
| CESM-ATM-3D | abs 1e-5 | TopoSZp+ | 1 | 2.75 | 11.65× | 2,403,563 | 0 | 0 | 2,403,563 | 1.870 |
| CESM-ATM-3D | abs 1e-5 | SZp | 1 | 0.71 | 45.36× | 2,945,994 | 0 | 0 | 2,945,994 | 1.000 |
| CESM-ATM-3D | abs 1e-5 | SZ3 | 1 | 0.00 | 8225.46× | 2,942,056 | 74,537 | 555 | 3,017,148 | 1.000 |
| CESM-ATM-3D | abs 1e-5 | SZ2 | 1 | 0.01 | 2283.21× | 2,903,666 | 1,699,107 | 4,688 | 4,607,461 | 1.000 |
| CESM-ATM-3D | abs 1e-5 | ZFP | 1 | 0.63 | 50.46× | 2,748,271 | 1,148,165 | 11,395 | 3,907,831 | 0.142 |
| CESM-ATM-3D | abs 1e-5 | LOPC | 1 | 7.27 | 4.40× | 1,597,008 | 1,914 | 0 | 1,598,922 | 0.382 |
| CESM-ATM-3D | abs 1e-5 | LOPC (GPU) | 1 | 7.27 | 4.40× | 1,597,008 | 1,914 | 0 | 1,598,922 | 0.382 |
| CESM-ATM-3D | rel 1e-3 | TopoSZp+ | 10 | 3.83 | 8.47× | 1,954,314 | 0 | 0 | 1,954,314 | 1.999 |
| CESM-ATM-3D | rel 1e-3 | SZp | 10 | 1.75 | 19.59× | 4,123,792 | 0 | 0 | 4,123,792 | 1.000 |
| CESM-ATM-3D | rel 1e-3 | SZ3 | 10 | 0.47 | 92.18× | 3,727,320 | 2,277,252 | 32,053 | 6,036,625 | 1.000 |
| CESM-ATM-3D | rel 1e-3 | SZ2 | 10 | 0.73 | 55.50× | 3,539,717 | 8,474,341 | 106,594 | 12,120,652 | 1.000 |
| CESM-ATM-3D | rel 1e-3 | ZFP | 10 | 3.20 | 11.08× | 2,127,731 | 3,816,240 | 50,242 | 5,994,213 | 0.283 |
| CESM-ATM-3D | rel 1e-3 | LOPC | 10 | 4.66 | 7.42× | 1,862,541 | 6,844 | 0 | 1,869,385 | 0.913 |
| CESM-ATM-3D | rel 1e-3 | LOPC (GPU) | 10 | 4.66 | 7.42× | 1,862,541 | 6,844 | 0 | 1,869,385 | 0.913 |
| CESM-ATM-3D | rel 1e-4 | TopoSZp+ | 10 | 4.93 | 6.68× | 1,330,346 | 0 | 0 | 1,330,346 | 1.999 |
| CESM-ATM-3D | rel 1e-4 | SZp | 10 | 2.85 | 12.22× | 2,965,299 | 0 | 0 | 2,965,299 | 1.000 |
| CESM-ATM-3D | rel 1e-4 | SZ3 | 10 | 1.46 | 27.61× | 2,593,700 | 2,153,416 | 25,200 | 4,772,316 | 1.000 |
| CESM-ATM-3D | rel 1e-4 | SZ2 | 10 | 1.62 | 23.64× | 2,533,182 | 6,374,193 | 68,839 | 8,976,213 | 1.000 |
| CESM-ATM-3D | rel 1e-4 | ZFP | 10 | 5.32 | 6.41× | 1,324,373 | 4,571,040 | 51,888 | 5,947,300 | 0.276 |
| CESM-ATM-3D | rel 1e-4 | LOPC | 10 | 4.99 | 6.81× | 1,277,694 | 6,649 | 0 | 1,284,343 | 0.948 |
| CESM-ATM-3D | rel 1e-4 | LOPC (GPU) | 10 | 4.99 | 6.81× | 1,277,694 | 6,649 | 0 | 1,284,343 | 0.948 |
| CESM-ATM-3D | rel 1e-5 | TopoSZp+ | 10 | 6.42 | 5.14× | 784,397 | 0 | 0 | 784,397 | 1.998 |
| CESM-ATM-3D | rel 1e-5 | SZp | 10 | 4.35 | 7.86× | 1,958,318 | 0 | 0 | 1,958,318 | 1.000 |
| CESM-ATM-3D | rel 1e-5 | SZ3 | 10 | 2.99 | 12.50× | 1,686,098 | 2,052,499 | 21,158 | 3,759,755 | 1.000 |
| CESM-ATM-3D | rel 1e-5 | SZ2 | 10 | 3.11 | 11.57× | 1,674,434 | 7,000,044 | 46,930 | 8,721,407 | 1.000 |
| CESM-ATM-3D | rel 1e-5 | ZFP | 10 | 7.50 | 4.43× | 854,698 | 4,822,573 | 42,145 | 5,719,416 | 0.260 |
| CESM-ATM-3D | rel 1e-5 | LOPC | 10 | 5.83 | 5.75× | 870,781 | 6,440 | 0 | 877,222 | 0.842 |
| CESM-ATM-3D | rel 1e-5 | LOPC (GPU) | 10 | 5.83 | 5.75× | 870,781 | 6,440 | 0 | 877,222 | 0.842 |
| NYX | abs 1e-3 | TopoSZp+ | 2 | 15.05 | 2.20× | 1,346 | 0 | 0 | 1,346 | 1.996 |
| NYX | abs 1e-3 | SZp | 2 | 11.87 | 2.74× | 473,300 | 0 | 0 | 473,300 | 1.000 |
| NYX | abs 1e-3 | SZ3 | 2 | 7.41 | 4.76× | 303,861 | 978,622 | 3,460 | 1,285,943 | 1.000 |
| NYX | abs 1e-3 | SZ2 | 2 | 7.76 | 4.63× | 333,401 | 466,000 | 2,858 | 802,258 | 1.000 |
| NYX | abs 1e-3 | ZFP | 2 | 12.03 | 2.78× | 28,406 | 841,309 | 172 | 869,886 | 2.075 |
| NYX | abs 1e-3 | LOPC | 2 | 10.71 | 3.04× | 212,667 | 2 | 0 | 212,670 | 0.915 |
| NYX | abs 1e-3 | LOPC (GPU) | 2 | 10.71 | 3.04× | 212,667 | 2 | 0 | 212,670 | 0.915 |
| NYX | abs 1e-4 | TopoSZp+ | 2 | 18.11 | 1.80× | 38 | 0 | 0 | 38 | 1.991 |
| NYX | abs 1e-4 | SZp | 2 | 15.14 | 2.13× | 64,179 | 0 | 0 | 64,179 | 1.000 |
| NYX | abs 1e-4 | SZ3 | 2 | 10.78 | 3.11× | 40,658 | 678,859 | 187 | 719,704 | 1.000 |
| NYX | abs 1e-4 | SZ2 | 2 | 11.18 | 3.04× | 41,952 | 815,226 | 218 | 857,396 | 1.000 |
| NYX | abs 1e-4 | ZFP | 2 | 16.03 | 2.05× | 1,931 | 811,890 | 12 | 813,834 | 20.752 |
| NYX | abs 1e-4 | LOPC | 2 | 14.52 | 2.23× | 19,038 | 0 | 0 | 19,038 | 0.534 |
| NYX | abs 1e-4 | LOPC (GPU) | 2 | 14.52 | 2.23× | 19,038 | 0 | 0 | 19,038 | 0.534 |
| NYX | abs 1e-5 | TopoSZp+ | 1 | 23.38 | 1.37× | 1 | 0 | 0 | 1 | 1.928 |
| NYX | abs 1e-5 | SZp | 1 | 19.91 | 1.61× | 6,029 | 0 | 0 | 6,029 | 1.000 |
| NYX | abs 1e-5 | SZ3 | 1 | 17.43 | 1.84× | 3,531 | 1,271,839 | 35 | 1,275,405 | 1.000 |
| NYX | abs 1e-5 | SZ2 | 1 | 18.53 | 1.73× | 3,451 | 1,564,165 | 47 | 1,567,663 | 1.000 |
| NYX | abs 1e-5 | ZFP | 1 | 21.55 | 1.49× | 216 | 1,613,281 | 3 | 1,613,500 | 24.414 |
| NYX | abs 1e-5 | LOPC | 1 | 19.17 | 1.67× | 2,017 | 0 | 0 | 2,017 | 0.572 |
| NYX | abs 1e-5 | LOPC (GPU) | 1 | 19.17 | 1.67× | 2,017 | 0 | 0 | 2,017 | 0.572 |
| NYX | rel 1e-3 | TopoSZp+ | 6 | 5.24 | 6.70× | 2,775,151 | 0 | 0 | 2,775,151 | 1.988 |
| NYX | rel 1e-3 | SZp | 6 | 2.64 | 19.92× | 8,060,859 | 0 | 0 | 8,060,859 | 1.000 |
| NYX | rel 1e-3 | SZ3 | 6 | 0.16 | 2267.99× | 8,014,641 | 720,470 | 11,360 | 8,746,472 | 1.000 |
| NYX | rel 1e-3 | SZ2 | 6 | 0.40 | 713.72× | 7,617,553 | 5,159,408 | 169,692 | 12,946,654 | 1.000 |
| NYX | rel 1e-3 | ZFP | 6 | 2.45 | 36.63× | 5,714,826 | 3,173,769 | 482,832 | 9,371,427 | 0.262 |
| NYX | rel 1e-3 | LOPC | 6 | 6.46 | 5.07× | 2,865,669 | 9 | 0 | 2,865,678 | 0.453 |
| NYX | rel 1e-3 | LOPC (GPU) | 6 | 6.46 | 5.07× | 2,865,669 | 9 | 0 | 2,865,678 | 0.453 |
| NYX | rel 1e-4 | TopoSZp+ | 6 | 7.36 | 5.19× | 2,218,003 | 0 | 0 | 2,218,003 | 1.998 |
| NYX | rel 1e-4 | SZp | 6 | 4.76 | 12.12× | 7,814,166 | 0 | 0 | 7,814,166 | 1.000 |
| NYX | rel 1e-4 | SZ3 | 6 | 1.14 | 484.35× | 7,606,593 | 1,110,740 | 44,584 | 8,761,917 | 1.000 |
| NYX | rel 1e-4 | SZ2 | 6 | 1.75 | 121.88× | 7,524,591 | 2,760,124 | 92,279 | 10,376,994 | 1.000 |
| NYX | rel 1e-4 | ZFP | 6 | 5.17 | 16.94× | 2,009,831 | 2,368,614 | 163,792 | 4,542,237 | 0.289 |
| NYX | rel 1e-4 | LOPC | 6 | 7.00 | 4.62× | 2,389,794 | 11 | 0 | 2,389,805 | 0.498 |
| NYX | rel 1e-4 | LOPC (GPU) | 6 | 7.00 | 4.62× | 2,389,794 | 11 | 0 | 2,389,805 | 0.498 |
| NYX | rel 1e-5 | TopoSZp+ | 6 | 10.16 | 3.64× | 281,122 | 0 | 0 | 281,122 | 1.997 |
| NYX | rel 1e-5 | SZp | 6 | 7.60 | 5.63× | 4,776,423 | 0 | 0 | 4,776,423 | 1.000 |
| NYX | rel 1e-5 | SZ3 | 6 | 3.46 | 85.78× | 3,735,039 | 2,617,239 | 232,102 | 6,584,380 | 1.000 |
| NYX | rel 1e-5 | SZ2 | 6 | 3.93 | 31.25× | 4,264,831 | 2,551,022 | 138,628 | 6,954,480 | 1.000 |
| NYX | rel 1e-5 | ZFP | 6 | 8.10 | 7.99× | 541,641 | 952,521 | 12,739 | 1,506,901 | 0.265 |
| NYX | rel 1e-5 | LOPC | 6 | 8.68 | 3.76× | 1,478,812 | 4 | 0 | 1,478,816 | 0.860 |
| NYX | rel 1e-5 | LOPC (GPU) | 6 | 8.68 | 3.76× | 1,478,812 | 4 | 0 | 1,478,816 | 0.860 |
| SCALE-LETKF | abs 1e-3 | TopoSZp+ | 2 | 10.48 | 6.91× | 928,916 | 0 | 0 | 928,916 | 1.057 |
| SCALE-LETKF | abs 1e-3 | SZp | 2 | 8.47 | 26.42× | 1,003,844 | 0 | 0 | 1,003,844 | 1.000 |
| SCALE-LETKF | abs 1e-3 | SZ3 | 2 | 5.43 | 27794.68× | 1,003,744 | 12,312 | 35 | 1,016,091 | 1.000 |
| SCALE-LETKF | abs 1e-3 | SZ2 | 2 | 4.16 | 3804.67× | 886,799 | 8,187,148 | 14,222 | 9,088,169 | 0.998 |
| SCALE-LETKF | abs 1e-3 | ZFP | 2 | 7.48 | 265.47× | 996,717 | 36,391 | 706 | 1,033,814 | 3.906 |
| SCALE-LETKF | abs 1e-3 | LOPC | 2 | 8.59 | 6.55× | 538,434 | 562 | 0 | 538,996 | 0.732 |
| SCALE-LETKF | abs 1e-3 | LOPC (GPU) | 2 | 8.59 | 6.55× | 538,434 | 562 | 0 | 538,996 | 0.732 |
| SCALE-LETKF | abs 1e-4 | TopoSZp+ | 2 | 12.16 | 6.58× | 907,144 | 0 | 0 | 907,144 | 1.916 |
| SCALE-LETKF | abs 1e-4 | SZp | 2 | 10.15 | 23.09× | 1,002,497 | 0 | 0 | 1,002,497 | 1.000 |
| SCALE-LETKF | abs 1e-4 | SZ3 | 2 | 6.63 | 3801.91× | 1,000,489 | 28,032 | 362 | 1,028,883 | 1.000 |
| SCALE-LETKF | abs 1e-4 | SZ2 | 2 | 4.74 | 722.45× | 912,385 | 7,608,617 | 11,244 | 8,532,246 | 1.000 |
| SCALE-LETKF | abs 1e-4 | ZFP | 2 | 7.70 | 72.56× | 926,010 | 296,202 | 6,939 | 1,229,150 | 39.062 |
| SCALE-LETKF | abs 1e-4 | LOPC | 2 | 8.74 | 6.55× | 532,912 | 552 | 0 | 533,464 | 0.305 |
| SCALE-LETKF | abs 1e-4 | LOPC (GPU) | 2 | 8.74 | 6.55× | 532,912 | 552 | 0 | 533,464 | 0.305 |
| SCALE-LETKF | abs 1e-5 | TopoSZp+ | 1 | 3.02 | 10.60× | 1,609,167 | 0 | 0 | 1,609,167 | 1.995 |
| SCALE-LETKF | abs 1e-5 | SZp | 1 | 1.00 | 32.07× | 1,985,528 | 0 | 0 | 1,985,528 | 1.000 |
| SCALE-LETKF | abs 1e-5 | SZ3 | 1 | 0.05 | 582.99× | 1,950,339 | 267,723 | 4,935 | 2,222,997 | 1.000 |
| SCALE-LETKF | abs 1e-5 | SZ2 | 1 | 0.12 | 269.15× | 1,834,931 | 13,204,525 | 21,780 | 15,061,236 | 1.000 |
| SCALE-LETKF | abs 1e-5 | ZFP | 1 | 0.55 | 58.51× | 1,594,814 | 1,316,157 | 32,655 | 2,943,626 | 0.208 |
| SCALE-LETKF | abs 1e-5 | LOPC | 1 | 2.89 | 11.07× | 1,027,667 | 1,105 | 0 | 1,028,772 | 0.382 |
| SCALE-LETKF | abs 1e-5 | LOPC (GPU) | 1 | 2.89 | 11.07× | 1,027,667 | 1,105 | 0 | 1,028,772 | 0.382 |
| SCALE-LETKF | rel 1e-3 | TopoSZp+ | 2 | 3.84 | 8.53× | 723,204 | 0 | 0 | 723,204 | 1.990 |
| SCALE-LETKF | rel 1e-3 | SZp | 2 | 1.83 | 19.60× | 1,344,764 | 0 | 0 | 1,344,764 | 1.000 |
| SCALE-LETKF | rel 1e-3 | SZ3 | 2 | 0.09 | 520.35× | 1,274,640 | 778,780 | 10,876 | 2,064,296 | 1.000 |
| SCALE-LETKF | rel 1e-3 | SZ2 | 2 | 0.23 | 141.10× | 1,259,614 | 6,706,706 | 11,300 | 7,977,620 | 1.000 |
| SCALE-LETKF | rel 1e-3 | ZFP | 2 | 1.64 | 24.67× | 927,979 | 1,051,850 | 22,704 | 2,002,532 | 0.234 |
| SCALE-LETKF | rel 1e-3 | LOPC | 2 | 4.25 | 8.39× | 731,364 | 1,446 | 0 | 732,811 | 0.322 |
| SCALE-LETKF | rel 1e-3 | LOPC (GPU) | 2 | 4.25 | 8.39× | 731,364 | 1,446 | 0 | 732,811 | 0.322 |
| SCALE-LETKF | rel 1e-4 | TopoSZp+ | 2 | 5.06 | 6.66× | 538,469 | 0 | 0 | 538,469 | 1.996 |
| SCALE-LETKF | rel 1e-4 | SZp | 2 | 3.05 | 12.24× | 1,195,504 | 0 | 0 | 1,195,504 | 1.000 |
| SCALE-LETKF | rel 1e-4 | SZ3 | 2 | 0.48 | 66.83× | 1,052,176 | 1,402,372 | 19,280 | 2,473,828 | 1.000 |
| SCALE-LETKF | rel 1e-4 | SZ2 | 2 | 0.99 | 38.46× | 1,082,826 | 6,652,962 | 9,384 | 7,745,172 | 1.000 |
| SCALE-LETKF | rel 1e-4 | ZFP | 2 | 2.87 | 13.99× | 666,983 | 1,342,837 | 29,812 | 2,039,632 | 0.222 |
| SCALE-LETKF | rel 1e-4 | LOPC | 2 | 4.42 | 8.04× | 613,442 | 1,514 | 0 | 614,957 | 0.402 |
| SCALE-LETKF | rel 1e-4 | LOPC (GPU) | 2 | 4.42 | 8.04× | 613,442 | 1,514 | 0 | 614,957 | 0.402 |
| SCALE-LETKF | rel 1e-5 | TopoSZp+ | 2 | 6.76 | 5.17× | 414,587 | 0 | 0 | 414,587 | 1.998 |
| SCALE-LETKF | rel 1e-5 | SZp | 2 | 4.75 | 8.14× | 921,936 | 0 | 0 | 921,936 | 1.000 |
| SCALE-LETKF | rel 1e-5 | SZ3 | 2 | 1.43 | 22.94× | 809,630 | 647,979 | 8,746 | 1,466,355 | 1.000 |
| SCALE-LETKF | rel 1e-5 | SZ2 | 2 | 1.53 | 21.76× | 812,946 | 6,365,636 | 6,486 | 7,185,068 | 1.000 |
| SCALE-LETKF | rel 1e-5 | ZFP | 2 | 4.96 | 8.73× | 442,470 | 1,435,248 | 31,661 | 1,909,378 | 0.299 |
| SCALE-LETKF | rel 1e-5 | LOPC | 2 | 5.49 | 6.83× | 438,740 | 1,564 | 0 | 440,303 | 0.988 |
| SCALE-LETKF | rel 1e-5 | LOPC (GPU) | 2 | 5.49 | 6.83× | 438,740 | 1,564 | 0 | 440,303 | 0.988 |

Means over fields; FN/FP/FT counted with the same checker (topo_check) for every compressor. *TTHRESH -e is a relative (Frobenius) target, not a pointwise bound.  34 runs did not finish (see baselines.csv, status column).  Left out because not every compressor finished them: CESM-ATM-3D/CLDICE_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/CLOUD_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/CMFDQR_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/CMFDQ_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/CMFDT_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/CONCLD_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/DCQ_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/DTCOND_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/DTV_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/FICE_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/GCLDLWP_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/ICIMR_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/ICLDIWP_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/ICLDTWP_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/ICWMR_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/OMEGAT_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/OMEGA_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/QC_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/QRL_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/QRS_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/Q_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/RELHUM_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/T_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/UU_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/U_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/VD01_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/VQ_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/VT_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/VU_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/VV_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/V_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/Z3_1_26_1800_3600 abs 1e-3, CESM-ATM-3D/CLDICE_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/CLOUD_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/CMFDQR_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/CMFDQ_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/CMFDT_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/CONCLD_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/DCQ_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/DTCOND_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/DTV_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/FICE_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/GCLDLWP_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/ICIMR_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/ICLDIWP_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/ICLDTWP_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/ICWMR_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/OMEGAT_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/OMEGA_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/QC_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/QRL_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/QRS_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/Q_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/RELHUM_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/T_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/UU_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/U_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/VD01_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/VQ_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/VT_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/VU_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/VV_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/V_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/Z3_1_26_1800_3600 abs 1e-4, CESM-ATM-3D/CLDLIQ_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/CLOUD_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/CMFDQR_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/CMFDQ_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/CMFDT_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/CONCLD_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/DCQ_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/DTCOND_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/DTV_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/FICE_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/GCLDLWP_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/ICIMR_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/ICLDIWP_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/ICLDTWP_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/ICWMR_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/OMEGAT_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/OMEGA_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/QC_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/QRL_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/QRS_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/Q_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/RELHUM_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/T_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/UU_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/U_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/VD01_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/VQ_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/VT_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/VU_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/VV_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/V_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/Z3_1_26_1800_3600 abs 1e-5, CESM-ATM-3D/FICE_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/GCLDLWP_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/ICIMR_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/ICLDIWP_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/ICLDTWP_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/ICWMR_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/OMEGAT_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/OMEGA_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/QC_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/QRL_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/QRS_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/Q_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/RELHUM_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/T_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/UU_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/U_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/VD01_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/VQ_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/VT_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/VU_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/VV_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/V_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/Z3_1_26_1800_3600 rel 1e-3, CESM-ATM-3D/FICE_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/GCLDLWP_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/ICIMR_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/ICLDIWP_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/ICLDTWP_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/ICWMR_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/OMEGAT_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/OMEGA_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/QC_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/QRL_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/QRS_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/Q_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/RELHUM_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/T_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/UU_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/U_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/VD01_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/VQ_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/VT_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/VU_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/VV_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/V_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/Z3_1_26_1800_3600 rel 1e-4, CESM-ATM-3D/FICE_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/GCLDLWP_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/ICIMR_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/ICLDIWP_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/ICLDTWP_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/ICWMR_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/OMEGAT_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/OMEGA_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/QC_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/QRL_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/QRS_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/Q_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/RELHUM_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/T_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/UU_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/U_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/VD01_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/VQ_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/VT_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/VU_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/VV_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/V_1_26_1800_3600 rel 1e-5, CESM-ATM-3D/Z3_1_26_1800_3600 rel 1e-5, NYX/temperature abs 1e-3, NYX/velocity_x abs 1e-3, NYX/velocity_y abs 1e-3, NYX/velocity_z abs 1e-3, NYX/temperature abs 1e-4, NYX/velocity_x abs 1e-4, NYX/velocity_y abs 1e-4, NYX/velocity_z abs 1e-4, NYX/baryon_density abs 1e-5, NYX/temperature abs 1e-5, NYX/velocity_x abs 1e-5, NYX/velocity_y abs 1e-5, NYX/velocity_z abs 1e-5, SCALE-LETKF/QG-98x1200x1200 abs 1e-3, SCALE-LETKF/QI-98x1200x1200 abs 1e-3, SCALE-LETKF/QR-98x1200x1200 abs 1e-3, SCALE-LETKF/QS-98x1200x1200 abs 1e-3, SCALE-LETKF/QV-98x1200x1200 abs 1e-3, SCALE-LETKF/RH-98x1200x1200 abs 1e-3, SCALE-LETKF/T-98x1200x1200 abs 1e-3, SCALE-LETKF/U-98x1200x1200 abs 1e-3, SCALE-LETKF/V-98x1200x1200 abs 1e-3, SCALE-LETKF/W-98x1200x1200 abs 1e-3, SCALE-LETKF/QG-98x1200x1200 abs 1e-4, SCALE-LETKF/QI-98x1200x1200 abs 1e-4, SCALE-LETKF/QR-98x1200x1200 abs 1e-4, SCALE-LETKF/QS-98x1200x1200 abs 1e-4, SCALE-LETKF/QV-98x1200x1200 abs 1e-4, SCALE-LETKF/RH-98x1200x1200 abs 1e-4, SCALE-LETKF/T-98x1200x1200 abs 1e-4, SCALE-LETKF/U-98x1200x1200 abs 1e-4, SCALE-LETKF/V-98x1200x1200 abs 1e-4, SCALE-LETKF/W-98x1200x1200 abs 1e-4, SCALE-LETKF/PRES-98x1200x1200 abs 1e-5, SCALE-LETKF/QG-98x1200x1200 abs 1e-5, SCALE-LETKF/QI-98x1200x1200 abs 1e-5, SCALE-LETKF/QR-98x1200x1200 abs 1e-5, SCALE-LETKF/QS-98x1200x1200 abs 1e-5, SCALE-LETKF/QV-98x1200x1200 abs 1e-5, SCALE-LETKF/RH-98x1200x1200 abs 1e-5, SCALE-LETKF/T-98x1200x1200 abs 1e-5, SCALE-LETKF/U-98x1200x1200 abs 1e-5, SCALE-LETKF/V-98x1200x1200 abs 1e-5, SCALE-LETKF/W-98x1200x1200 abs 1e-5, SCALE-LETKF/QG-98x1200x1200 rel 1e-3, SCALE-LETKF/QI-98x1200x1200 rel 1e-3, SCALE-LETKF/QR-98x1200x1200 rel 1e-3, SCALE-LETKF/QS-98x1200x1200 rel 1e-3, SCALE-LETKF/QV-98x1200x1200 rel 1e-3, SCALE-LETKF/RH-98x1200x1200 rel 1e-3, SCALE-LETKF/T-98x1200x1200 rel 1e-3, SCALE-LETKF/U-98x1200x1200 rel 1e-3, SCALE-LETKF/V-98x1200x1200 rel 1e-3, SCALE-LETKF/W-98x1200x1200 rel 1e-3, SCALE-LETKF/QG-98x1200x1200 rel 1e-4, SCALE-LETKF/QI-98x1200x1200 rel 1e-4, SCALE-LETKF/QR-98x1200x1200 rel 1e-4, SCALE-LETKF/QS-98x1200x1200 rel 1e-4, SCALE-LETKF/QV-98x1200x1200 rel 1e-4, SCALE-LETKF/RH-98x1200x1200 rel 1e-4, SCALE-LETKF/T-98x1200x1200 rel 1e-4, SCALE-LETKF/U-98x1200x1200 rel 1e-4, SCALE-LETKF/V-98x1200x1200 rel 1e-4, SCALE-LETKF/W-98x1200x1200 rel 1e-4, SCALE-LETKF/QG-98x1200x1200 rel 1e-5, SCALE-LETKF/QI-98x1200x1200 rel 1e-5, SCALE-LETKF/QR-98x1200x1200 rel 1e-5, SCALE-LETKF/QS-98x1200x1200 rel 1e-5, SCALE-LETKF/QV-98x1200x1200 rel 1e-5, SCALE-LETKF/RH-98x1200x1200 rel 1e-5, SCALE-LETKF/T-98x1200x1200 rel 1e-5, SCALE-LETKF/U-98x1200x1200 rel 1e-5, SCALE-LETKF/V-98x1200x1200 rel 1e-5, SCALE-LETKF/W-98x1200x1200 rel 1e-5.

## Bit-rate sweep

| Dataset | Bound | Compressor | Fields | Bits/value | Ratio | FN | FP | FT | Total | Max err / eps |
|---|---|---|---|---|---|---|---|---|---|---|
| CESM-ATM-2D | abs 1e-2 | TopoSZp+ | 60 | 6.34 | 6.27× | 3,608 | 0 | 0 | 3,608 | 1.744 |
| CESM-ATM-2D | abs 1e-2 | SZp | 60 | 4.27 | 15.23× | 27,927 | 0 | 0 | 27,927 | 1.000 |
| CESM-ATM-2D | abs 1e-2 | SZ3 | 60 | 1.70 | 2473.96× | 25,388 | 18,392 | 291 | 44,071 | 1.000 |
| CESM-ATM-2D | abs 1e-2 | SZ2 | 60 | 1.95 | 1148.30× | 25,317 | 76,230 | 763 | 102,310 | 1.000 |
| CESM-ATM-2D | abs 1e-2 | ZFP | 60 | 5.35 | 18.11× | 15,201 | 18,358 | 420 | 33,979 | 0.362 |
| CESM-ATM-2D | abs 5e-3 | TopoSZp+ | 60 | 6.91 | 5.87× | 2,902 | 0 | 0 | 2,902 | 1.758 |
| CESM-ATM-2D | abs 5e-3 | SZp | 60 | 4.85 | 13.59× | 24,852 | 0 | 0 | 24,852 | 1.000 |
| CESM-ATM-2D | abs 5e-3 | SZ3 | 60 | 2.16 | 1858.82× | 21,901 | 21,242 | 416 | 43,559 | 1.000 |
| CESM-ATM-2D | abs 5e-3 | SZ2 | 60 | 2.43 | 714.06× | 21,995 | 80,382 | 802 | 103,179 | 1.000 |
| CESM-ATM-2D | abs 5e-3 | ZFP | 60 | 6.01 | 14.70× | 12,174 | 15,424 | 333 | 27,931 | 0.353 |
| CESM-ATM-2D | abs 1e-3 | TopoSZp+ | 60 | 8.38 | 5.04× | 1,484 | 0 | 0 | 1,484 | 1.761 |
| CESM-ATM-2D | abs 1e-3 | SZp | 60 | 6.34 | 10.39× | 17,335 | 0 | 0 | 17,335 | 1.000 |
| CESM-ATM-2D | abs 1e-3 | SZ3 | 60 | 3.49 | 885.55× | 14,257 | 20,562 | 458 | 35,277 | 1.000 |
| CESM-ATM-2D | abs 1e-3 | SZ2 | 60 | 3.68 | 243.54× | 15,027 | 63,243 | 904 | 79,174 | 1.000 |
| CESM-ATM-2D | abs 1e-3 | ZFP | 60 | 7.40 | 10.52× | 7,644 | 12,060 | 198 | 19,902 | 0.465 |
| CESM-ATM-2D | abs 5e-4 | TopoSZp+ | 60 | 9.06 | 4.72× | 1,171 | 0 | 0 | 1,171 | 1.748 |
| CESM-ATM-2D | abs 5e-4 | SZp | 60 | 7.02 | 9.22× | 14,025 | 0 | 0 | 14,025 | 1.000 |
| CESM-ATM-2D | abs 5e-4 | SZ3 | 60 | 4.10 | 576.72× | 11,600 | 14,442 | 275 | 26,317 | 1.000 |
| CESM-ATM-2D | abs 5e-4 | SZ2 | 60 | 4.28 | 152.13× | 11,942 | 55,268 | 713 | 67,922 | 1.000 |
| CESM-ATM-2D | abs 5e-4 | ZFP | 60 | 8.13 | 9.06× | 5,934 | 10,744 | 170 | 16,848 | 0.448 |
| CESM-ATM-2D | abs 1e-4 | TopoSZp+ | 60 | 10.73 | 4.04× | 625 | 0 | 0 | 625 | 1.758 |
| CESM-ATM-2D | abs 1e-4 | SZp | 60 | 8.70 | 6.97× | 8,146 | 0 | 0 | 8,146 | 1.000 |
| CESM-ATM-2D | abs 1e-4 | SZ3 | 60 | 5.62 | 177.59× | 6,632 | 9,958 | 153 | 16,743 | 1.000 |
| CESM-ATM-2D | abs 1e-4 | SZ2 | 60 | 5.91 | 55.89× | 6,847 | 41,554 | 376 | 48,777 | 1.000 |
| CESM-ATM-2D | abs 1e-4 | ZFP | 60 | 10.41 | 6.20× | 2,605 | 7,259 | 72 | 9,936 | 0.305 |
| CESM-ATM-2D | abs 5e-5 | TopoSZp+ | 60 | 11.48 | 3.77× | 448 | 0 | 0 | 448 | 1.761 |
| CESM-ATM-2D | abs 5e-5 | SZp | 60 | 9.45 | 6.21× | 6,381 | 0 | 0 | 6,381 | 1.000 |
| CESM-ATM-2D | abs 5e-5 | SZ3 | 60 | 6.74 | 103.13× | 5,059 | 8,632 | 155 | 13,846 | 1.000 |
| CESM-ATM-2D | abs 5e-5 | SZ2 | 60 | 7.07 | 38.64× | 5,346 | 31,048 | 290 | 36,684 | 1.000 |
| CESM-ATM-2D | abs 5e-5 | ZFP | 60 | 11.20 | 5.54× | 1,983 | 6,743 | 53 | 8,778 | 0.305 |
| CESM-ATM-2D | abs 1e-5 | TopoSZp+ | 60 | 13.27 | 3.24× | 235 | 0 | 0 | 235 | 1.758 |
| CESM-ATM-2D | abs 1e-5 | SZp | 60 | 11.25 | 4.84× | 3,420 | 0 | 0 | 3,420 | 1.000 |
| CESM-ATM-2D | abs 1e-5 | SZ3 | 60 | 8.63 | 37.68× | 2,647 | 6,127 | 88 | 8,861 | 1.000 |
| CESM-ATM-2D | abs 1e-5 | SZ2 | 60 | 9.20 | 18.19× | 2,762 | 23,064 | 155 | 25,980 | 1.000 |
| CESM-ATM-2D | abs 1e-5 | ZFP | 60 | 12.79 | 4.49× | 1,111 | 5,787 | 34 | 6,932 | 0.382 |
| CESM-ATM-2D | abs 5e-6 | TopoSZp+ | 60 | 14.06 | 3.04× | 178 | 0 | 0 | 178 | 1.630 |
| CESM-ATM-2D | abs 5e-6 | SZp | 60 | 12.04 | 4.37× | 2,583 | 0 | 0 | 2,583 | 1.000 |
| CESM-ATM-2D | abs 5e-6 | SZ3 | 60 | 9.04 | 25.18× | 2,026 | 5,466 | 56 | 7,548 | 1.000 |
| CESM-ATM-2D | abs 5e-6 | SZ2 | 60 | 9.56 | 13.83× | 2,080 | 21,255 | 116 | 23,451 | 1.000 |
| CESM-ATM-2D | abs 5e-6 | ZFP | 60 | 13.39 | 4.10× | 813 | 5,327 | 27 | 6,167 | 0.763 |
| CESM-ATM-2D | abs 1e-6 | TopoSZp+ | 60 | 15.94 | 2.62× | 77 | 0 | 0 | 77 | 1.907 |
| CESM-ATM-2D | abs 1e-6 | SZp | 60 | 13.93 | 3.52× | 1,349 | 0 | 0 | 1,349 | 1.000 |
| CESM-ATM-2D | abs 1e-6 | SZ3 | 60 | 10.53 | 11.15× | 1,047 | 4,291 | 24 | 5,362 | 1.000 |
| CESM-ATM-2D | abs 1e-6 | SZ2 | 60 | 11.17 | 7.90× | 1,037 | 15,921 | 71 | 17,028 | 1.000 |
| CESM-ATM-2D | abs 1e-6 | ZFP | 60 | 14.25 | 3.50× | 415 | 4,583 | 15 | 5,012 | 3.815 |

Means over fields; FN/FP/FT counted with the same checker (topo_check) for every compressor. *TTHRESH -e is a relative (Frobenius) target, not a pointwise bound.

## Cross-check with the piecewise-linear (TTK) definition

| Dataset | Bound | Fields | PL extrema | PL extrema kept % | PL saddles | PL saddles kept % (SZp -> TopoSZp+) | PL saddles with persistence >= 2 eps kept % | PL FP (SZp -> TopoSZp+) | PL FT (SZp -> TopoSZp+) | Axis saddles | Axis saddles that are PL saddles % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CESM-ATM-2D | rel 1e-3 | 3 | 284,114 | 99.9 | 283,414 | 34.6 -> 55.3 | 59.1 | 233,767 -> 228,220 | 5,868 -> 1,091 | 78,673 | 100.0 |
| CESM-ATM-3D | rel 1e-3 | 3 | 1,270,678 | 98.0 | 4,334,924 | 22.4 -> 29.9 | 61.4 | 1,275,671 -> 1,694,799 | 50,786 -> 57,596 | 11,137,568 | 12.6 |
| Hurricane | rel 1e-3 | 3 | 264,198 | 96.0 | 880,157 | 54.6 -> 57.0 | 76.5 | 58,907 -> 161,003 | 1,922 -> 9,185 | 975,638 | 18.3 |
| NYX | rel 1e-3 | 3 | 10,593,232 | 98.4 | 34,882,527 | 3.5 -> 12.0 | 39.8 | 1,121,926 -> 6,963,524 | 19,835 -> 1,424,052 | 27,381,854 | 21.3 |
| SCALE-LETKF | rel 1e-3 | 3 | 250,920 | 82.6 | 857,684 | 2.4 -> 5.8 | 60.5 | 211,963 -> 340,243 | 2,586 -> 7,647 | 4,695,631 | 6.6 |

Freudenthal triangulation (6 neighbors in 2D, 14 in 3D), ties broken by index, as TTK. TopoSZp+ guarantees refer to the axis rule; this table shows how the result looks under the PL definition.

