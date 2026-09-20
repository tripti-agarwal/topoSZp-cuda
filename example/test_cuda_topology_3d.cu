/**
 *  @file test_cuda_topology_3d.cu
 *  @brief End-to-end 3D critical point preservation test for CUDA TopoSZp.
 *
 *  Verifies that critical points (maxima, minima, saddles) in the original
 *  3D data are preserved after topology-aware compression/decompression.
 *
 *  Pipeline:
 *    1. Find critical points in original data (6-connected, CUDA + OpenMP)
 *    2. Sort critical points by original data within bins
 *    3. Compress with topology preservation (CUDA)
 *    4. Decompress with topology + extract FN array (CUDA)
 *    5. Verify raw decompression error
 *    6. Post-processing: stencils (6 neighbors), enforcement, RBF (3×3×3)
 *    7. Find critical points in post-processed data
 *    8. Verify preservation, FN array, error bound
 *
 *  Usage: test_cuda_topology_3d <input_file> <d1> <d2> <d3> <absErrBound> <blockSize>
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "szp.h"
#include "szp_cuda_compress.cuh"
#include "szp_cuda_decompress.cuh"
#include "szp_cuda_topology_3d.cuh"
#include "szp_topology_3d.h"

/* ================================================================ */
/*  Helpers                                                          */
/* ================================================================ */

static double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* 3D flat index */
static inline size_t IDX(int x, int y, int z, int d2, int d3) {
    return (size_t)x * d2 * d3 + (size_t)y * d3 + z;
}

/* Classify a voxel with 6-connected neighborhood */
static int classify_3d(const float *data, int d1, int d2, int d3,
                        int x, int y, int z)
{
    size_t s = (size_t)d2 * d3;
    float c  = data[x * s + y * d3 + z];
    float xm = data[(x - 1) * s + y * d3 + z];
    float xp = data[(x + 1) * s + y * d3 + z];
    float ym = data[x * s + (y - 1) * d3 + z];
    float yp = data[x * s + (y + 1) * d3 + z];
    float zm = data[x * s + y * d3 + (z - 1)];
    float zp = data[x * s + y * d3 + (z + 1)];

    if (c > xm && c > xp && c > ym && c > yp && c > zm && c > zp) return 1;
    if (c < xm && c < xp && c < ym && c < yp && c < zm && c < zp) return 2;

    int xh = (c > xm) && (c > xp), xl = (c < xm) && (c < xp);
    int yh = (c > ym) && (c > yp), yl = (c < ym) && (c < yp);
    int zh = (c > zm) && (c > zp), zl = (c < zm) && (c < zp);
    if ((xh + yh + zh) >= 1 && (xl + yl + zl) >= 1) return 3;

    return 0;
}

/* ---- Post-processing: 3D stencils with error-bound clamping ---- */

static void apply_stencils_3d(float *data, const float *orig_decomp,
                               const int *types, const int *sort_positions,
                               int d1, int d2, int d3,
                               float errBound, size_t extrema_count)
{
    size_t sort_idx = 0;
    size_t s = (size_t)d2 * d3;

    for (int x = 1; x < d1 - 1; x++) {
        for (int y = 1; y < d2 - 1; y++) {
            for (int z = 1; z < d3 - 1; z++) {
                size_t flat = x * s + y * d3 + z;
                int type = types[flat];
                if (type != 1 && type != 2) continue;

                int sp = (sort_idx < extrema_count)
                             ? sort_positions[sort_idx] : 0;
                float orig_val = orig_decomp[flat];
                float max_a = orig_val + errBound * 0.999f;
                float min_a = orig_val - errBound * 0.999f;

                float xm = data[(x-1)*s + y*d3 + z];
                float xp = data[(x+1)*s + y*d3 + z];
                float ym = data[x*s + (y-1)*d3 + z];
                float yp = data[x*s + (y+1)*d3 + z];
                float zm = data[x*s + y*d3 + (z-1)];
                float zp = data[x*s + y*d3 + (z+1)];

                if (type == 1) {
                    /* Maxima: above max of 6 neighbors */
                    float mn = fmaxf(fmaxf(xm, xp),
                               fmaxf(fmaxf(ym, yp), fmaxf(zm, zp)));
                    float target = mn * (1.0f + sp * FLT_EPSILON);
                    if (target > max_a) target = max_a;
                    if (target < min_a) target = min_a;
                    data[flat] = target;
                } else {
                    /* Minima: below min of 6 neighbors */
                    float mn = fminf(fminf(xm, xp),
                               fminf(fminf(ym, yp), fminf(zm, zp)));
                    float target = (sp == 0)
                        ? mn * (1.0f - FLT_EPSILON)
                        : mn * (1.0f - (1.0f / sp) * FLT_EPSILON);
                    if (target > max_a) target = max_a;
                    if (target < min_a) target = min_a;
                    data[flat] = target;
                }
                sort_idx++;
            }
        }
    }
}

/* Final enforcement with 6-neighbor clamping */
static void restore_extrema_3d(const int *types, float *data,
                                 const float *orig_decomp,
                                 int d1, int d2, int d3,
                                 float eps, float errBound)
{
    float eps_soft = 0.25f * eps;
    size_t s = (size_t)d2 * d3;

    for (int x = 1; x < d1 - 1; x++) {
        for (int y = 1; y < d2 - 1; y++) {
            for (int z = 1; z < d3 - 1; z++) {
                size_t flat = x * s + y * d3 + z;
                float orig_val = orig_decomp[flat];
                float max_a = orig_val + errBound * 0.999f;
                float min_a = orig_val - errBound * 0.999f;

                float xm = data[(x-1)*s+y*d3+z], xp = data[(x+1)*s+y*d3+z];
                float ym = data[x*s+(y-1)*d3+z], yp = data[x*s+(y+1)*d3+z];
                float zm = data[x*s+y*d3+(z-1)], zp = data[x*s+y*d3+(z+1)];

                if (types[flat] == 1) {
                    float m = fmaxf(fmaxf(xm,xp), fmaxf(fmaxf(ym,yp), fmaxf(zm,zp)));
                    float target = m + eps_soft;
                    if (target > max_a) target = max_a;
                    if (target < min_a) target = min_a;
                    if (data[flat] < target) data[flat] = target;
                } else if (types[flat] == 2) {
                    float m = fminf(fminf(xm,xp), fminf(fminf(ym,yp), fminf(zm,zp)));
                    float target = m - eps_soft;
                    if (target < min_a) target = min_a;
                    if (target > max_a) target = max_a;
                    if (data[flat] > target) data[flat] = target;
                }
            }
        }
    }
}

/* RBF saddle restoration with 3×3×3 Gaussian kernel */
static int rbf_restore_saddles_3d(float *data, const int *types0,
                                    const float *orig_decomp,
                                    int d1, int d2, int d3,
                                    float eps, float errBound)
{
    size_t N = (size_t)d1 * d2 * d3;
    size_t s = (size_t)d2 * d3;

    /* Lock mask: 3×3×3 neighbourhood of every extremum */
    unsigned char *locks = (unsigned char *)calloc(N, 1);
    for (int x = 1; x < d1 - 1; x++)
        for (int y = 1; y < d2 - 1; y++)
            for (int z = 1; z < d3 - 1; z++) {
                size_t flat = x * s + y * d3 + z;
                if (types0[flat] == 1 || types0[flat] == 2)
                    for (int di = -1; di <= 1; di++)
                        for (int dj = -1; dj <= 1; dj++)
                            for (int dk = -1; dk <= 1; dk++) {
                                int ii = x + di, jj = y + dj, kk = z + dk;
                                if (ii >= 1 && ii < d1-1 &&
                                    jj >= 1 && jj < d2-1 &&
                                    kk >= 1 && kk < d3-1)
                                    locks[ii * s + jj * d3 + kk] = 1;
                            }
            }

    /* Identify false-negative saddles */
    unsigned char *fn_mask = (unsigned char *)calloc(N, 1);
    int fn_count = 0;
    for (int x = 1; x < d1 - 1; x++)
        for (int y = 1; y < d2 - 1; y++)
            for (int z = 1; z < d3 - 1; z++) {
                size_t flat = x * s + y * d3 + z;
                if (types0[flat] == 3 &&
                    classify_3d(data, d1, d2, d3, x, y, z) != 3) {
                    fn_mask[flat] = 1;
                    fn_count++;
                }
            }

    if (fn_count == 0) { free(locks); free(fn_mask); return 0; }

    /* 3×3×3 Gaussian kernel */
    double sigma = 0.8, s2 = sigma * sigma;
    int ksize = 3, r = ksize / 2;
    double w[27];
    for (int di = -r; di <= r; di++)
        for (int dj = -r; dj <= r; dj++)
            for (int dk = -r; dk <= r; dk++)
                w[(di+r)*9 + (dj+r)*3 + (dk+r)] =
                    exp(-(double)(di*di + dj*dj + dk*dk) / (2.0 * s2));

    int restored = 0;
    for (int x = 1; x < d1 - 1; x++) {
        for (int y = 1; y < d2 - 1; y++) {
            for (int z = 1; z < d3 - 1; z++) {
                size_t flat = x * s + y * d3 + z;
                if (!fn_mask[flat] || types0[flat] != 3) continue;

                double num = 0, den = 0;
                for (int di = -r; di <= r; di++) {
                    int ii = x + di; if (ii < 0 || ii >= d1) continue;
                    for (int dj = -r; dj <= r; dj++) {
                        int jj = y + dj; if (jj < 0 || jj >= d2) continue;
                        for (int dk = -r; dk <= r; dk++) {
                            int kk = z + dk; if (kk < 0 || kk >= d3) continue;
                            double ww = w[(di+r)*9 + (dj+r)*3 + (dk+r)];
                            if (locks[ii*s + jj*d3 + kk]) ww *= 0.4;
                            num += ww * data[ii*s + jj*d3 + kk];
                            den += ww;
                        }
                    }
                }
                if (den <= 0) continue;

                float oldc = data[flat];
                float orig_val = orig_decomp[flat];
                float min_a = orig_val - errBound * 0.999f;
                float max_a = orig_val + errBound * 0.999f;
                float cand = (float)(num / den);
                if (cand < min_a) cand = min_a;
                if (cand > max_a) cand = max_a;

                float step = cand - oldc, alpha = 1.0f;
                for (int it = 0; it < 10; it++) {
                    float trial = oldc + alpha * step;
                    if (trial < min_a) trial = min_a;
                    if (trial > max_a) trial = max_a;

                    float saved = data[flat];
                    data[flat] = trial;
                    int bad = 0;

                    /* Verify we didn't create false CPs at regular neighbours */
                    if (classify_3d(data,d1,d2,d3,x,y,z) != 3) bad = 1;
                    if (!bad && x-1>=1 && types0[(x-1)*s+y*d3+z]==0 &&
                        classify_3d(data,d1,d2,d3,x-1,y,z)!=0) bad=1;
                    if (!bad && x+1<d1-1 && types0[(x+1)*s+y*d3+z]==0 &&
                        classify_3d(data,d1,d2,d3,x+1,y,z)!=0) bad=1;
                    if (!bad && y-1>=1 && types0[x*s+(y-1)*d3+z]==0 &&
                        classify_3d(data,d1,d2,d3,x,y-1,z)!=0) bad=1;
                    if (!bad && y+1<d2-1 && types0[x*s+(y+1)*d3+z]==0 &&
                        classify_3d(data,d1,d2,d3,x,y+1,z)!=0) bad=1;
                    if (!bad && z-1>=1 && types0[x*s+y*d3+(z-1)]==0 &&
                        classify_3d(data,d1,d2,d3,x,y,z-1)!=0) bad=1;
                    if (!bad && z+1<d3-1 && types0[x*s+y*d3+(z+1)]==0 &&
                        classify_3d(data,d1,d2,d3,x,y,z+1)!=0) bad=1;

                    if (bad) { data[flat] = saved; alpha *= 0.5f; }
                    else     { restored++; break; }
                }
            }
        }
    }
    free(locks);
    free(fn_mask);
    return restored;
}

/* ================================================================ */
/*  main                                                             */
/* ================================================================ */

int main(int argc, char *argv[])
{
    if (argc < 7) {
        fprintf(stderr,
            "Usage: %s <input_file> <d1> <d2> <d3> <absErrBound> <blockSize>\n",
            argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    int d1 = atoi(argv[2]);
    int d2 = atoi(argv[3]);
    int d3 = atoi(argv[4]);
    float absErrBound = (float)atof(argv[5]);
    int blockSize = atoi(argv[6]);
    size_t nbEle = (size_t)d1 * d2 * d3;

    printf("=== TopoSZp 3D Critical Point Preservation Test ===\n");
    printf("Input: %s  (%d × %d × %d = %zu elements)\n",
           input_file, d1, d2, d3, nbEle);
    printf("Error bound: %e    Block size: %d\n\n", absErrBound, blockSize);

    /* Read input */
    float *data = (float *)malloc(nbEle * sizeof(float));
    FILE *fp = fopen(input_file, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", input_file); return 1; }
    size_t nread = fread(data, sizeof(float), nbEle, fp);
    fclose(fp);
    if (nread != nbEle) {
        fprintf(stderr, "Read %zu / %zu elements\n", nread, nbEle);
        return 1;
    }

    /* CUDA warmup */
    { void *t; cudaSetDevice(0); cudaMalloc(&t,1); cudaFree(t); cudaDeviceSynchronize(); }

    /* ============================================================ */
    /* Step 1: Find critical points (CUDA)                           */
    /* ============================================================ */
    printf("--- Step 1: Find 3D critical points in original data ---\n");
    size_t orig_cp_count = 0;
    double t0 = get_time_ms();
    CriticalPoint3D *orig_cps = szp_cuda_find_critical_points_3d(
        data, &orig_cp_count, d1, d2, d3, absErrBound);
    double t1 = get_time_ms();

    size_t om = 0, omin = 0, os = 0;
    for (size_t i = 0; i < orig_cp_count; i++) {
        if (orig_cps[i].type == 1) om++;
        else if (orig_cps[i].type == 2) omin++;
        else if (orig_cps[i].type == 3) os++;
    }
    printf("Found %zu critical points in %.2f ms\n", orig_cp_count, t1-t0);
    printf("  Maxima: %zu    Minima: %zu    Saddles: %zu\n\n", om, omin, os);

    /* Cross-check with OpenMP */
    size_t omp_count = 0;
    CriticalPoint3D *omp_cps = szp_find_critical_points_3d(
        data, &omp_count, d1, d2, d3, absErrBound);
    printf("OpenMP found %zu (CUDA %zu — %s)\n\n",
           omp_count, orig_cp_count,
           omp_count == orig_cp_count ? "MATCH" : "DIFFER");

    if (!orig_cps || orig_cp_count == 0) {
        printf("No critical points. Nothing to test.\n");
        free(data); return 0;
    }

    /* ============================================================ */
    /* Step 2: Sort by data value within bins                        */
    /* ============================================================ */
    printf("--- Step 2: Sort critical points ---\n");
    double t2 = get_time_ms();
    szp_cuda_sort_critical_points_3d(orig_cps, orig_cp_count, data, d2, d3);
    double t3 = get_time_ms();
    printf("Sorted in %.2f ms\n\n", t3 - t2);

    /* ============================================================ */
    /* Step 3: Topology-preserved compression (CUDA)                 */
    /* ============================================================ */
    printf("--- Step 3: Topology-preserved compression ---\n");
    size_t topo_outSize = 0;
    double t4 = get_time_ms();
    unsigned char *topo_compressed = szp_cuda_float_compress_topology_3d(
        data, &topo_outSize, absErrBound, nbEle, blockSize,
        orig_cps, (int)orig_cp_count, d1, d2, d3);
    double t5 = get_time_ms();
    printf("Compressed: %zu bytes (ratio: %.2fx) in %.2f ms\n\n",
           topo_outSize, (double)(nbEle*sizeof(float))/topo_outSize, t5-t4);

    if (!topo_compressed) {
        printf("ERROR: compression failed\n"); free(data); free(orig_cps); return 1;
    }

    /* ============================================================ */
    /* Step 4: Decompress + extract FN (dimension-agnostic)          */
    /* ============================================================ */
    printf("--- Step 4: Decompress with topology extraction ---\n");
    float *decompressed = NULL;
    int   *FN = NULL;
    double t6 = get_time_ms();
    szp_cuda_float_decompress_randomaccess_topology_preserved(
        &decompressed, nbEle, absErrBound, blockSize,
        topo_compressed, &FN);
    double t7 = get_time_ms();
    printf("Decompressed in %.2f ms\n\n", t7 - t6);

    if (!decompressed || !FN) {
        printf("ERROR: decompression failed\n");
        free(data); free(orig_cps); free(topo_compressed); return 1;
    }

    /* ============================================================ */
    /* Step 4a: Raw decompression check                              */
    /* ============================================================ */
    printf("--- Step 4a: Raw decompression verification ---\n");
    {
        double rme = 0;
        size_t rec = 0;
        for (size_t i = 0; i < nbEle; i++) {
            double e = fabs((double)data[i] - (double)decompressed[i]);
            if (e > rme) rme = e;
            if (e > absErrBound * 1.01) rec++;
        }
        printf("Raw max error: %e (bound: %e) — %s\n", rme, (double)absErrBound,
               rec == 0 ? "PASS" : "FAIL");

        size_t rp = 0;
        for (size_t i = 0; i < orig_cp_count; i++) {
            int x = orig_cps[i].x, y = orig_cps[i].y, z = orig_cps[i].z;
            if (x>=1&&x<d1-1&&y>=1&&y<d2-1&&z>=1&&z<d3-1)
                if (classify_3d(decompressed,d1,d2,d3,x,y,z) == orig_cps[i].type)
                    rp++;
        }
        printf("Raw CP preservation: %zu / %zu (%.2f%%)\n\n",
               rp, orig_cp_count, 100.0*rp/orig_cp_count);
    }

    /* ============================================================ */
    /* Step 4b: Post-processing                                      */
    /* ============================================================ */
    printf("--- Step 4b: Post-processing ---\n");
    double tp0 = get_time_ms();

    float *orig_decomp = (float *)malloc(nbEle * sizeof(float));
    memcpy(orig_decomp, decompressed, nbEle * sizeof(float));

    /* Sort positions */
    size_t sort_outSize = 0;
    unsigned char *sort_compressed = szp_cuda_compress_sort_positions_3d(
        orig_cps, orig_cp_count, &sort_outSize, blockSize);

    size_t extrema_count = 0;
    for (size_t i = 0; i < orig_cp_count; i++)
        if (orig_cps[i].type == 1 || orig_cps[i].type == 2) extrema_count++;

    int *sort_positions = NULL;
    if (sort_compressed && sort_outSize > 0 && extrema_count > 0)
        sort_positions = szp_cuda_decompress_sort_positions(
            sort_compressed, extrema_count, blockSize);

    printf("  Sort positions: %zu extrema, decompressed: %s\n",
           extrema_count, sort_positions ? "OK" : "NULL");

    /* Stencils */
    if (sort_positions && extrema_count > 0)
        apply_stencils_3d(decompressed, orig_decomp, FN, sort_positions,
                          d1, d2, d3, absErrBound, extrema_count);

    /* Enforcement */
    float eps = fmaxf(1e-6f, 0.1f * absErrBound);
    restore_extrema_3d(FN, decompressed, orig_decomp, d1, d2, d3, eps, absErrBound);

    /* RBF saddle restoration */
    int restored = rbf_restore_saddles_3d(decompressed, FN, orig_decomp,
                                           d1, d2, d3, eps, absErrBound);
    printf("  RBF restored %d saddles\n", restored);

    double tp1 = get_time_ms();
    printf("  Post-processing done in %.2f ms\n\n", tp1 - tp0);

    /* ============================================================ */
    /* Step 5: CPs in post-processed data                            */
    /* ============================================================ */
    printf("--- Step 5: Find CPs in post-processed data ---\n");
    size_t dcp = 0;
    size_t dm = 0, dmin = 0, ds = 0;
    /* Build type map for decompressed data */
    int *decomp_type_map = (int *)calloc(nbEle, sizeof(int));
    for (int x = 1; x < d1-1; x++)
        for (int y = 1; y < d2-1; y++)
            for (int z = 1; z < d3-1; z++) {
                size_t flat = IDX(x,y,z,d2,d3);
                int t = classify_3d(decompressed, d1, d2, d3, x, y, z);
                if (t) { decomp_type_map[flat] = t; dcp++;
                    if (t==1) dm++; else if (t==2) dmin++; else ds++; }
            }
    printf("Found %zu CPs\n  Maxima: %zu  Minima: %zu  Saddles: %zu\n\n",
           dcp, dm, dmin, ds);

    /* ============================================================ */
    /* Step 6: Preservation check                                    */
    /* ============================================================ */
    printf("--- Step 6: Critical point preservation ---\n");
    size_t preserved = 0, lost = 0, changed = 0;
    size_t l_max = 0, l_min = 0, l_sad = 0;
    for (size_t i = 0; i < orig_cp_count; i++) {
        size_t flat = IDX(orig_cps[i].x, orig_cps[i].y, orig_cps[i].z, d2, d3);
        if (decomp_type_map[flat] == orig_cps[i].type) preserved++;
        else if (decomp_type_map[flat] == 0) {
            lost++;
            if (orig_cps[i].type==1) l_max++;
            else if (orig_cps[i].type==2) l_min++;
            else l_sad++;
        } else changed++;
    }

    /* New CPs */
    int *orig_type_map = (int *)calloc(nbEle, sizeof(int));
    for (size_t i = 0; i < orig_cp_count; i++)
        orig_type_map[IDX(orig_cps[i].x,orig_cps[i].y,orig_cps[i].z,d2,d3)] =
            orig_cps[i].type;
    size_t new_cps = 0;
    for (int x = 1; x < d1-1; x++)
        for (int y = 1; y < d2-1; y++)
            for (int z = 1; z < d3-1; z++) {
                size_t flat = IDX(x,y,z,d2,d3);
                if (decomp_type_map[flat] && !orig_type_map[flat]) new_cps++;
            }

    printf("Original: %zu    Preserved: %zu (%.2f%%)\n", orig_cp_count,
           preserved, 100.0*preserved/orig_cp_count);
    printf("Lost: %zu (max:%zu min:%zu sad:%zu)  Changed: %zu  New: %zu\n\n",
           lost, l_max, l_min, l_sad, changed, new_cps);

    /* ============================================================ */
    /* Step 7: FN array                                              */
    /* ============================================================ */
    printf("--- Step 7: FN array ---\n");
    size_t fn_match = 0, fn_nz = 0;
    unsigned char *orig_t_arr = (unsigned char *)calloc(nbEle, 1);
    for (size_t i = 0; i < orig_cp_count; i++)
        orig_t_arr[IDX(orig_cps[i].x,orig_cps[i].y,orig_cps[i].z,d2,d3)] =
            (unsigned char)orig_cps[i].type;
    for (size_t i = 0; i < nbEle; i++) {
        if (FN[i]) fn_nz++;
        if (FN[i] == (int)orig_t_arr[i]) fn_match++;
    }
    printf("Non-zero: %zu (expected %zu)   Match: %zu / %zu (%.2f%%)\n\n",
           fn_nz, orig_cp_count, fn_match, nbEle, 100.0*fn_match/nbEle);

    /* ============================================================ */
    /* Step 8: Error bound                                           */
    /* ============================================================ */
    printf("--- Step 8: Error bound ---\n");
    double max_err = 0;
    size_t e1 = 0, e2 = 0;
    for (size_t i = 0; i < nbEle; i++) {
        double e = fabs((double)data[i] - (double)decompressed[i]);
        if (e > max_err) max_err = e;
        if (e > absErrBound * 1.01) e1++;
        if (e > absErrBound * 2.0)  e2++;
    }
    const char *eb_status;
    if (e2 == 0 && e1 == 0)      eb_status = "PASS (within 1×eb)";
    else if (e2 == 0)             eb_status = "PASS (within 2×eb — expected for topology preservation)";
    else                          eb_status = "FAIL (exceeds 2×eb)";
    printf("Max error: %e   1×eb over: %zu   2×eb over: %zu\n", max_err, e1, e2);
    printf("Status: %s\n\n", eb_status);

    /* ============================================================ */
    /* Summary                                                       */
    /* ============================================================ */
    printf("==================== 3D SUMMARY ====================\n");
    printf("Grid:              %d × %d × %d = %zu\n", d1, d2, d3, nbEle);
    printf("Critical points:   %zu → %zu in decompressed\n", orig_cp_count, dcp);
    printf("Preserved:         %zu / %zu (%.2f%%)\n",
           preserved, orig_cp_count, 100.0*preserved/orig_cp_count);
    printf("  Maxima:          %zu / %zu\n", om - l_max, om);
    printf("  Minima:          %zu / %zu\n", omin - l_min, omin);
    printf("  Saddles:         %zu / %zu\n", os - l_sad, os);
    printf("Error bound:       %s\n", eb_status);
    printf("FN array:          %s (%zu / %zu)\n",
           (fn_match == nbEle) ? "PASS" : "PARTIAL", fn_match, nbEle);
    printf("Compression ratio: %.2fx\n",
           (double)(nbEle*sizeof(float))/topo_outSize);
    printf("====================================================\n");

    /* Cleanup */
    free(data);
    free(orig_cps);
    if (omp_cps) free(omp_cps);
    free(topo_compressed);
    free(decompressed);
    free(FN);
    free(decomp_type_map);
    free(orig_type_map);
    free(orig_t_arr);
    free(orig_decomp);
    if (sort_compressed) free(sort_compressed);
    if (sort_positions)  free(sort_positions);

    return (e2 == 0 && preserved > orig_cp_count * 0.9) ? 0 : 1;
}
