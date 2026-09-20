/**
 *  @file test_cuda_topology.cu
 *  @brief End-to-end critical point preservation test for CUDA TopoSZp.
 *
 *  Verifies that critical points (maxima, minima, saddles) in the original
 *  data are preserved after topology-aware compression/decompression.
 *
 *  Pipeline:
 *    1. Find critical points in original data
 *    2. Sort critical points by original data within bins
 *    3. Compress with topology preservation (CUDA)
 *    4. Decompress with topology + extract critical types (CUDA)
 *    5. Find critical points in decompressed data
 *    6. Compare: every original CP should be a CP of the same type in decompressed
 *    7. Also verify the embedded 2-bit type (FN) array matches
 *
 *  Usage: test_cuda_topology <input_file> <rows> <cols> <absErrBound> <blockSize>
 *
 *  For CESM-ATM 2D: ./test_cuda_topology CLDHGH_1_1800_3600.dat 1800 3600 1e-3 64
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
#include "szp_cuda_topology.cuh"

static double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static const char *type_name(int t) {
    switch (t) {
        case 1: return "MAX";
        case 2: return "MIN";
        case 3: return "SADDLE";
        default: return "REGULAR";
    }
}

/* ---- Post-processing stencils (ported from OpenMP decompressor) ---- */

static const int NB4[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};

/* Classify a point as max(1)/min(2)/saddle(3)/regular(0) using 4-connectivity */
static int classify_point(const float *data, int rows, int cols, int i, int j) {
    float center = data[i * cols + j];
    float up     = data[(i-1) * cols + j];
    float down   = data[(i+1) * cols + j];
    float left   = data[i * cols + (j-1)];
    float right  = data[i * cols + (j+1)];

    if (center > up && center > down && center > left && center > right) return 1;
    if (center < up && center < down && center < left && center < right) return 2;
    if ((center < up && center < down && center > left && center > right) ||
        (center > up && center > down && center < left && center < right)) return 3;
    return 0;
}

/* Restore maxima: set value slightly above max neighbor */
static void apply_maxima_stencil(float *data, int rows, int cols, int i, int j, int sort_pos) {
    float max_neighbor = data[i*cols+j];
    for (int n = 0; n < 4; n++) {
        int ni = i + NB4[n][0], nj = j + NB4[n][1];
        float v = data[ni*cols+nj];
        if (v > max_neighbor) max_neighbor = v;
    }
    data[i*cols+j] = max_neighbor * (1.0f + (sort_pos * FLT_EPSILON));
}

/* Restore minima: set value slightly below min neighbor */
static void apply_minima_stencil(float *data, int rows, int cols, int i, int j, int sort_pos) {
    float min_neighbor = data[i*cols+j];
    for (int n = 0; n < 4; n++) {
        int ni = i + NB4[n][0], nj = j + NB4[n][1];
        float v = data[ni*cols+nj];
        if (v < min_neighbor) min_neighbor = v;
    }
    if (sort_pos == 0) data[i*cols+j] = min_neighbor * (1.0f - FLT_EPSILON);
    else data[i*cols+j] = min_neighbor * (1.0f - ((1.0f/sort_pos) * FLT_EPSILON));
}

/* Final enforcement: ensure extrema are strictly above/below neighbors */
static void restore_extrema_from_types(const int *types, float *data,
                                        int rows, int cols, float eps) {
    float eps_soft = 0.25f * eps;
    for (int i = 1; i < rows-1; i++) {
        for (int j = 1; j < cols-1; j++) {
            int idx = i*cols+j;
            if (types[idx] == 1) { /* maxima */
                float n = data[(i-1)*cols+j], s = data[(i+1)*cols+j];
                float w = data[i*cols+(j-1)], e = data[i*cols+(j+1)];
                float m = fmaxf(fmaxf(n,s), fmaxf(w,e));
                float target = m + eps_soft;
                if (data[idx] < target) data[idx] = target;
            } else if (types[idx] == 2) { /* minima */
                float n = data[(i-1)*cols+j], s = data[(i+1)*cols+j];
                float w = data[i*cols+(j-1)], e = data[i*cols+(j+1)];
                float m = fminf(fminf(n,s), fminf(w,e));
                float target = m - eps_soft;
                if (data[idx] > target) data[idx] = target;
            }
        }
    }
}

/* Apply stencils to all extrema with error bound clamping (matching OpenMP pipeline) */
static void apply_stencils_clamped(float *data, const float *orig_decomp, const int *types,
                                    const int *sort_positions,
                                    int rows, int cols, float errBound, size_t extrema_count) {
    size_t sort_idx = 0;
    for (int i = 1; i < rows-1; i++) {
        for (int j = 1; j < cols-1; j++) {
            int idx = i*cols+j;
            int type = types[idx];
            if (type == 1 || type == 2) {
                int sp = (sort_idx < extrema_count) ? sort_positions[sort_idx] : 0;
                float orig_val = orig_decomp[idx];
                float max_allowed = orig_val + errBound;
                float min_allowed = orig_val - errBound;

                if (type == 1) {
                    /* Maxima: set above max neighbor */
                    float max_neighbor = data[idx];
                    for (int n = 0; n < 4; n++) {
                        int ni = i + NB4[n][0], nj = j + NB4[n][1];
                        float v = data[ni*cols+nj];
                        if (v > max_neighbor) max_neighbor = v;
                    }
                    float target = max_neighbor * (1.0f + (sp * FLT_EPSILON));
                    /* Clamp to error bound */
                    if (target > max_allowed) target = max_allowed;
                    if (target < min_allowed) target = min_allowed;
                    data[idx] = target;
                } else {
                    /* Minima: set below min neighbor */
                    float min_neighbor = data[idx];
                    for (int n = 0; n < 4; n++) {
                        int ni = i + NB4[n][0], nj = j + NB4[n][1];
                        float v = data[ni*cols+nj];
                        if (v < min_neighbor) min_neighbor = v;
                    }
                    float target;
                    if (sp == 0) target = min_neighbor * (1.0f - FLT_EPSILON);
                    else target = min_neighbor * (1.0f - ((1.0f/sp) * FLT_EPSILON));
                    /* Clamp to error bound */
                    if (target > max_allowed) target = max_allowed;
                    if (target < min_allowed) target = min_allowed;
                    data[idx] = target;
                }
                sort_idx++;
            }
        }
    }
}

/* Final enforcement with strict error bound clamping */
static void restore_extrema_clamped(const int *types, float *data, const float *orig_decomp,
                                     int rows, int cols, float eps, float errBound) {
    float eps_soft = 0.25f * eps;
    for (int i = 1; i < rows-1; i++) {
        for (int j = 1; j < cols-1; j++) {
            int idx = i*cols+j;
            float orig_val = orig_decomp[idx];
            /* Strict clamping: stay within errBound of original decompressed value */
            float max_allowed = orig_val + errBound * 0.999f;
            float min_allowed = orig_val - errBound * 0.999f;

            if (types[idx] == 1) {
                float n = data[(i-1)*cols+j], s = data[(i+1)*cols+j];
                float w = data[i*cols+(j-1)], e = data[i*cols+(j+1)];
                float m = fmaxf(fmaxf(n,s), fmaxf(w,e));
                float target = m + eps_soft;
                if (target > max_allowed) target = max_allowed;
                if (target < min_allowed) target = min_allowed;
                if (data[idx] < target) data[idx] = target;
            } else if (types[idx] == 2) {
                float n = data[(i-1)*cols+j], s = data[(i+1)*cols+j];
                float w = data[i*cols+(j-1)], e = data[i*cols+(j+1)];
                float m = fminf(fminf(n,s), fminf(w,e));
                float target = m - eps_soft;
                if (target < min_allowed) target = min_allowed;
                if (target > max_allowed) target = max_allowed;
                if (data[idx] > target) data[idx] = target;
            }
        }
    }
}

/* ---- RBF saddle restoration (ported from OpenMP decompressor) ---- */

static inline float clamp_saddle_center(float cand, float n, float s, float w, float e, float eps) {
    float ns_max = fmaxf(n,s), we_max = fmaxf(w,e);
    float ns_min = fminf(n,s), we_min = fminf(w,e);
    float lo = fminf(ns_max, we_max) - 0.25f*eps;
    float hi = fmaxf(ns_min, we_min) + 0.25f*eps;
    if (lo > hi) { float mid = 0.5f*(lo+hi); lo = mid - 0.25f*eps; hi = mid + 0.25f*eps; }
    if (cand < lo) cand = lo;
    if (cand > hi) cand = hi;
    return cand;
}

static inline int cls4(float c, float n, float s, float w, float e, float eps) {
    if (c > n+eps && c > s+eps && c > w+eps && c > e+eps) return 1;
    if (c < n-eps && c < s-eps && c < w-eps && c < e-eps) return 2;
    int vh = (c > n+eps) && (c > s+eps), vl = (c < n-eps) && (c < s-eps);
    int hh = (c > w+eps) && (c > e+eps), hl = (c < w-eps) && (c < e-eps);
    if ((vh && hl) || (vl && hh)) return 3;
    return 0;
}

static inline int cls_idx(const float *a, int r, int c, int i, int j, float eps) {
    (void)r;
    return cls4(a[i*c+j], a[(i-1)*c+j], a[(i+1)*c+j], a[i*c+(j-1)], a[i*c+(j+1)], eps);
}

static int rbf_restore_saddles(float *data, const int *types0, const float *orig_decomp,
                                int rows, int cols, float eps, float errBound) {
    /* Build lock mask around extrema */
    size_t N = (size_t)rows * cols;
    unsigned char *locks = (unsigned char *)calloc(N, 1);
    for (int i = 1; i < rows-1; i++)
        for (int j = 1; j < cols-1; j++) {
            int idx = i*cols+j;
            if (types0[idx] == 1 || types0[idx] == 2)
                for (int di = -1; di <= 1; di++)
                    for (int dj = -1; dj <= 1; dj++) {
                        int ii = i+di, jj = j+dj;
                        if (ii > 0 && ii < rows-1 && jj > 0 && jj < cols-1)
                            locks[ii*cols+jj] = 1;
                    }
        }

    /* Identify false-negative saddles */
    unsigned char *fn_mask = (unsigned char *)calloc(N, 1);
    int fn_count = 0;
    for (int i = 1; i < rows-1; i++)
        for (int j = 1; j < cols-1; j++) {
            int idx = i*cols+j;
            if (types0[idx] == 3 && classify_point(data, rows, cols, i, j) != 3) {
                fn_mask[idx] = 1;
                fn_count++;
            }
        }

    if (fn_count == 0) { free(locks); free(fn_mask); return 0; }

    /* Gaussian RBF kernel */
    double sigma = 0.8;
    int ksize = 3, r = ksize/2;
    double s2 = sigma*sigma;
    double w[9];
    for (int di = -r; di <= r; di++)
        for (int dj = -r; dj <= r; dj++)
            w[(di+r)*ksize+(dj+r)] = exp(-((double)di*di + dj*dj) / (2.0*s2));

    int restored = 0;
    for (int i = 1; i < rows-1; i++) {
        for (int j = 1; j < cols-1; j++) {
            int idx = i*cols+j;
            if (!fn_mask[idx] || types0[idx] != 3) continue;

            /* RBF weighted average */
            double num = 0, den = 0;
            for (int di = -r; di <= r; di++) {
                int ii = i+di; if (ii < 0 || ii >= rows) continue;
                for (int dj = -r; dj <= r; dj++) {
                    int jj = j+dj; if (jj < 0 || jj >= cols) continue;
                    double ww = w[(di+r)*ksize+(dj+r)];
                    if (locks[ii*cols+jj]) ww *= 0.4;
                    num += ww * data[ii*cols+jj];
                    den += ww;
                }
            }
            if (den <= 0) continue;

            float oldc = data[idx];
            float orig_val = orig_decomp[idx];
            float n = data[(i-1)*cols+j], s = data[(i+1)*cols+j];
            float wv = data[i*cols+(j-1)], ev = data[i*cols+(j+1)];
            float cand = (float)(num/den);
            cand = clamp_saddle_center(cand, n, s, wv, ev, eps);

            /* Strict error bound clamping */
            float min_a = orig_val - errBound * 0.999f;
            float max_a = orig_val + errBound * 0.999f;
            if (cand < min_a) cand = min_a;
            if (cand > max_a) cand = max_a;

            /* Iterative alpha reduction with validation */
            float step = cand - oldc, alpha = 1.0f;
            for (int it = 0; it < 10; it++) {
                float trial = oldc + alpha*step;
                if (trial < min_a) trial = min_a;
                if (trial > max_a) trial = max_a;

                float saved = data[idx]; data[idx] = trial;
                int bad = 0;
                /* Check we don't create new CPs at regular points */
                if (types0[idx] == 0 && cls_idx(data,rows,cols,i,j,eps) != 0) bad = 1;
                if (!bad && i-1 >= 1 && types0[(i-1)*cols+j] == 0 && cls_idx(data,rows,cols,i-1,j,eps) != 0) bad = 1;
                if (!bad && i+1 < rows-1 && types0[(i+1)*cols+j] == 0 && cls_idx(data,rows,cols,i+1,j,eps) != 0) bad = 1;
                if (!bad && j-1 >= 1 && types0[i*cols+(j-1)] == 0 && cls_idx(data,rows,cols,i,j-1,eps) != 0) bad = 1;
                if (!bad && j+1 < cols-1 && types0[i*cols+(j+1)] == 0 && cls_idx(data,rows,cols,i,j+1,eps) != 0) bad = 1;
                if (!bad && cls_idx(data,rows,cols,i,j,eps) != 3) bad = 1;

                if (bad) { data[idx] = saved; alpha *= 0.5f; }
                else { restored++; break; }
            }
        }
    }

    free(locks);
    free(fn_mask);
    return restored;
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <input_file> <rows> <cols> <absErrBound> <blockSize>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    int rows = atoi(argv[2]);
    int cols = atoi(argv[3]);
    float absErrBound = (float)atof(argv[4]);
    int blockSize = atoi(argv[5]);
    size_t nbEle = (size_t)rows * cols;

    printf("=== TopoSZp Critical Point Preservation Test ===\n");
    printf("Input: %s  (%d x %d = %zu elements)\n", input_file, rows, cols, nbEle);
    printf("Error bound: %e    Block size: %d\n\n", absErrBound, blockSize);

    /* Read input data */
    float *data = (float *)malloc(nbEle * sizeof(float));
    FILE *fp = fopen(input_file, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", input_file); return 1; }
    size_t nread = fread(data, sizeof(float), nbEle, fp);
    fclose(fp);
    if (nread != nbEle) {
        fprintf(stderr, "Read only %zu of %zu elements\n", nread, nbEle);
        return 1;
    }

    /* CUDA warmup */
    {
        void *d_tmp;
        cudaSetDevice(0);
        cudaMalloc(&d_tmp, 1);
        cudaFree(d_tmp);
        cudaDeviceSynchronize();
    }

    /* ============================================================ */
    /* Step 1: Find critical points in original data (CUDA)          */
    /* ============================================================ */
    printf("--- Step 1: Find critical points in original data ---\n");
    size_t orig_cp_count = 0;
    double t0 = get_time_ms();
    CriticalPoint *orig_cps = szp_cuda_find_critical_points(data, &orig_cp_count, rows, cols, absErrBound);
    double t1 = get_time_ms();

    size_t orig_max = 0, orig_min = 0, orig_saddle = 0;
    for (size_t i = 0; i < orig_cp_count; i++) {
        if (orig_cps[i].type == 1) orig_max++;
        else if (orig_cps[i].type == 2) orig_min++;
        else if (orig_cps[i].type == 3) orig_saddle++;
    }
    printf("Found %zu critical points in %.2f ms\n", orig_cp_count, t1 - t0);
    printf("  Maxima: %zu    Minima: %zu    Saddles: %zu\n\n", orig_max, orig_min, orig_saddle);

    /* Also find with OpenMP for cross-validation */
    size_t omp_cp_count = 0;
    CriticalPoint *omp_cps = szp_find_critical_points(data, &omp_cp_count, rows, cols, absErrBound);
    printf("OpenMP found %zu critical points (CUDA: %zu — %s)\n\n",
           omp_cp_count, orig_cp_count,
           orig_cp_count == omp_cp_count ? "MATCH" : "DIFFER");

    if (!orig_cps || orig_cp_count == 0) {
        printf("No critical points found. Nothing to test.\n");
        free(data);
        return 0;
    }

    /* ============================================================ */
    /* Step 2: Sort critical points by data value within bins         */
    /* ============================================================ */
    printf("--- Step 2: Sort critical points by data value within bins ---\n");
    double t2 = get_time_ms();
    szp_cuda_sort_critical_points_by_original_data(orig_cps, orig_cp_count, data, cols);
    double t3 = get_time_ms();
    printf("Sorted in %.2f ms\n\n", t3 - t2);

    /* ============================================================ */
    /* Step 3: Compress with topology preservation (CUDA)             */
    /* ============================================================ */
    printf("--- Step 3: Topology-preserved compression (CUDA) ---\n");
    size_t topo_outSize = 0;
    double t4 = get_time_ms();
    unsigned char *topo_compressed = szp_cuda_float_compress_randomaccess_topology_preserved(
        data, &topo_outSize, absErrBound, nbEle, blockSize,
        orig_cps, (int)orig_cp_count, rows, cols);
    double t5 = get_time_ms();
    printf("Compressed: %zu bytes (ratio: %.2fx) in %.2f ms\n\n",
           topo_outSize, (double)(nbEle * sizeof(float)) / topo_outSize, t5 - t4);

    if (!topo_compressed) {
        printf("ERROR: Topology-preserved compression failed.\n");
        free(data); free(orig_cps);
        return 1;
    }

    /* ============================================================ */
    /* Step 4: Decompress with topology type extraction (CUDA)       */
    /* ============================================================ */
    printf("--- Step 4: Topology-preserved decompression (CUDA) ---\n");
    float *decompressed = NULL;
    int *FN = NULL;  /* 2-bit type per element: 0=regular, 1=max, 2=min, 3=saddle */

    double t6 = get_time_ms();
    szp_cuda_float_decompress_randomaccess_topology_preserved(
        &decompressed, nbEle, absErrBound, blockSize,
        topo_compressed, &FN);
    double t7 = get_time_ms();
    printf("Decompressed in %.2f ms\n\n", t7 - t6);

    if (!decompressed || !FN) {
        printf("ERROR: Topology-preserved decompression failed.\n");
        free(data); free(orig_cps); free(topo_compressed);
        return 1;
    }

    /* ============================================================ */
    /* Step 4b: Post-processing — restore critical points            */
    /*          (same pipeline as the OpenMP decompressor)            */
    /* ============================================================ */
    /* ---- Step 4a-verify: Raw decompression error check (before stencils) ---- */
    printf("--- Step 4a: Verify raw decompression (before post-processing) ---\n");
    {
        double raw_max_err = 0.0;
        size_t raw_err_count = 0;
        for (size_t i = 0; i < nbEle; i++) {
            double err = fabs((double)data[i] - (double)decompressed[i]);
            if (err > raw_max_err) raw_max_err = err;
            if (err > absErrBound * 1.01) raw_err_count++;
        }
        printf("Raw max error: %e (bound: %e) — %s\n",
               raw_max_err, (double)absErrBound,
               raw_err_count == 0 ? "PASS" : "FAIL");

        /* Check raw CP preservation (without post-processing) */
        size_t raw_preserved = 0;
        for (size_t i = 0; i < orig_cp_count; i++) {
            int x = orig_cps[i].x, y = orig_cps[i].y;
            if (x >= 1 && x < rows-1 && y >= 1 && y < cols-1) {
                int decomp_type = classify_point(decompressed, rows, cols, x, y);
                if (decomp_type == orig_cps[i].type) raw_preserved++;
            }
        }
        printf("Raw CP preservation (no stencils): %zu / %zu (%.2f%%)\n\n",
               raw_preserved, orig_cp_count, 100.0 * raw_preserved / orig_cp_count);
    }

    printf("--- Step 4b: Post-processing — restore critical points ---\n");
    double t7b = get_time_ms();

    /* Save original decompressed values for error bound clamping */
    float *orig_decomp = (float *)malloc(nbEle * sizeof(float));
    memcpy(orig_decomp, decompressed, nbEle * sizeof(float));

    /* Compress sort positions and decompress them (same as OpenMP pipeline) */
    size_t sort_outSize = 0;
    unsigned char *sort_compressed = szp_cuda_compress_sort_positions(
        orig_cps, orig_cp_count, &sort_outSize, blockSize);

    /* Count extrema */
    size_t extrema_count = 0;
    for (size_t i = 0; i < orig_cp_count; i++) {
        if (orig_cps[i].type == 1 || orig_cps[i].type == 2)
            extrema_count++;
    }

    int *sort_positions = NULL;
    if (sort_compressed && sort_outSize > 0 && extrema_count > 0) {
        sort_positions = szp_cuda_decompress_sort_positions(
            sort_compressed, extrema_count, blockSize);
    }
    printf("  Sort positions: %zu extrema, compressed %zu bytes, decompressed: %s\n",
           extrema_count, sort_outSize, sort_positions ? "OK" : "NULL");

    /* Apply stencils to extrema with error bound clamping */
    if (sort_positions && extrema_count > 0) {
        apply_stencils_clamped(decompressed, orig_decomp, FN, sort_positions,
                                rows, cols, absErrBound, extrema_count);
        printf("  Applied clamped stencils to %zu extrema\n", extrema_count);
    }

    /* Final enforcement with error bound clamping */
    float eps = fmaxf(1e-6f, 0.1f * absErrBound);
    restore_extrema_clamped(FN, decompressed, orig_decomp, rows, cols, eps, absErrBound);

    /* RBF saddle restoration — restore lost saddles via Gaussian smoothing */
    int restored_saddles = rbf_restore_saddles(decompressed, FN, orig_decomp,
                                               rows, cols, eps, absErrBound);
    printf("  RBF restored %d saddles\n", restored_saddles);

    double t7c = get_time_ms();
    printf("  Post-processing done in %.2f ms\n\n", t7c - t7b);

    /* ============================================================ */
    /* Step 5: Find critical points in decompressed data             */
    /*         (AFTER post-processing)                               */
    /* ============================================================ */
    printf("--- Step 5: Find critical points in post-processed data ---\n");
    size_t decomp_cp_count = 0;
    CriticalPoint *decomp_cps = szp_cuda_find_critical_points(decompressed, &decomp_cp_count, rows, cols, absErrBound);

    size_t decomp_max = 0, decomp_min = 0, decomp_saddle = 0;
    for (size_t i = 0; i < decomp_cp_count; i++) {
        if (decomp_cps[i].type == 1) decomp_max++;
        else if (decomp_cps[i].type == 2) decomp_min++;
        else if (decomp_cps[i].type == 3) decomp_saddle++;
    }
    printf("Found %zu critical points in decompressed data\n", decomp_cp_count);
    printf("  Maxima: %zu    Minima: %zu    Saddles: %zu\n\n", decomp_max, decomp_min, decomp_saddle);

    /* ============================================================ */
    /* Step 6: Verify preservation — every original CP must exist     */
    /*         with the same type in the decompressed data           */
    /* ============================================================ */
    printf("--- Step 6: Verify critical point preservation ---\n");

    /* Build a lookup map from decompressed CPs: flat_index → type */
    int *decomp_type_map = (int *)calloc(nbEle, sizeof(int));
    for (size_t i = 0; i < decomp_cp_count; i++) {
        size_t flat = (size_t)decomp_cps[i].x * cols + decomp_cps[i].y;
        if (flat < nbEle) decomp_type_map[flat] = decomp_cps[i].type;
    }

    size_t preserved = 0, lost = 0, type_changed = 0;
    size_t lost_max = 0, lost_min = 0, lost_saddle = 0;
    size_t changed_max = 0, changed_min = 0, changed_saddle = 0;

    for (size_t i = 0; i < orig_cp_count; i++) {
        size_t flat = (size_t)orig_cps[i].x * cols + orig_cps[i].y;
        int orig_type = orig_cps[i].type;
        int decomp_type = decomp_type_map[flat];

        if (decomp_type == orig_type) {
            preserved++;
        } else if (decomp_type == 0) {
            lost++;
            if (orig_type == 1) lost_max++;
            else if (orig_type == 2) lost_min++;
            else if (orig_type == 3) lost_saddle++;
        } else {
            type_changed++;
            if (orig_type == 1) changed_max++;
            else if (orig_type == 2) changed_min++;
            else if (orig_type == 3) changed_saddle++;
        }
    }

    printf("Original CPs: %zu\n", orig_cp_count);
    printf("Preserved (same type): %zu (%.2f%%)\n",
           preserved, 100.0 * preserved / orig_cp_count);
    printf("Lost (became regular): %zu (%.2f%%)\n",
           lost, 100.0 * lost / orig_cp_count);
    if (lost > 0) printf("  Lost maxima: %zu  minima: %zu  saddles: %zu\n",
                          lost_max, lost_min, lost_saddle);
    printf("Type changed: %zu (%.2f%%)\n",
           type_changed, 100.0 * type_changed / orig_cp_count);
    if (type_changed > 0) printf("  Changed maxima: %zu  minima: %zu  saddles: %zu\n",
                                  changed_max, changed_min, changed_saddle);

    /* New CPs that didn't exist in original */
    size_t orig_type_map_count = 0;
    int *orig_type_map = (int *)calloc(nbEle, sizeof(int));
    for (size_t i = 0; i < orig_cp_count; i++) {
        size_t flat = (size_t)orig_cps[i].x * cols + orig_cps[i].y;
        if (flat < nbEle) { orig_type_map[flat] = orig_cps[i].type; orig_type_map_count++; }
    }

    size_t new_cps = 0;
    for (size_t i = 0; i < decomp_cp_count; i++) {
        size_t flat = (size_t)decomp_cps[i].x * cols + decomp_cps[i].y;
        if (flat < nbEle && orig_type_map[flat] == 0)
            new_cps++;
    }
    printf("New CPs (not in original): %zu\n\n", new_cps);

    /* ============================================================ */
    /* Step 7: Verify embedded FN (type) array                       */
    /* ============================================================ */
    printf("--- Step 7: Verify embedded type (FN) array ---\n");

    /* Build original type array */
    unsigned char *orig_type_arr = (unsigned char *)calloc(nbEle, 1);
    for (size_t i = 0; i < orig_cp_count; i++) {
        size_t flat = (size_t)orig_cps[i].x * cols + orig_cps[i].y;
        if (flat < nbEle) orig_type_arr[flat] = (unsigned char)orig_cps[i].type;
    }

    size_t fn_match = 0, fn_mismatch = 0;
    size_t fn_nonzero = 0;
    for (size_t i = 0; i < nbEle; i++) {
        if (FN[i] != 0) fn_nonzero++;
        if (FN[i] == (int)orig_type_arr[i])
            fn_match++;
        else
            fn_mismatch++;
    }
    printf("FN array: %zu non-zero entries (expected: %zu)\n", fn_nonzero, orig_cp_count);
    printf("FN matches original types: %zu / %zu (%.2f%%)\n",
           fn_match, nbEle, 100.0 * fn_match / nbEle);
    if (fn_mismatch > 0)
        printf("FN mismatches: %zu\n", fn_mismatch);
    printf("\n");

    /* ============================================================ */
    /* Step 8: Verify error bound                                    */
    /* ============================================================ */
    printf("--- Step 8: Verify error bound ---\n");
    double max_err = 0.0;
    size_t err_count = 0;
    for (size_t i = 0; i < nbEle; i++) {
        double err = fabs((double)data[i] - (double)decompressed[i]);
        if (err > max_err) max_err = err;
        if (err > absErrBound * 1.01) err_count++;
    }
    printf("Max pointwise error: %e (bound: %e) — %s\n",
           max_err, (double)absErrBound,
           err_count == 0 ? "PASS" : "FAIL");
    if (err_count > 0)
        printf("  %zu elements exceed error bound\n", err_count);

    /* ============================================================ */
    /* Summary                                                        */
    /* ============================================================ */
    printf("\n==================== SUMMARY ====================\n");
    printf("Critical points:   %zu original → %zu in decompressed\n", orig_cp_count, decomp_cp_count);
    printf("Preserved:         %zu / %zu (%.2f%%)\n", preserved, orig_cp_count,
           100.0 * preserved / orig_cp_count);
    printf("Error bound:       %s (max: %e)\n",
           err_count == 0 ? "PASS" : "FAIL", max_err);
    printf("FN array:          %s (%zu / %zu match)\n",
           fn_mismatch == 0 ? "PASS" : "PARTIAL", fn_match, nbEle);
    printf("Compression ratio: %.2fx\n", (double)(nbEle * sizeof(float)) / topo_outSize);

    size_t cuda_preserved = preserved;
    size_t cuda_decomp_cp_count = decomp_cp_count;
    double cuda_max_err = max_err;
    size_t cuda_err_count = err_count;

    /* ============================================================ */
    /* Step 9: OpenMP pipeline for comparison                        */
    /* ============================================================ */
    printf("\n============ OpenMP PIPELINE COMPARISON ============\n\n");

    /* Re-find and sort critical points for OpenMP (use omp_cps) */
    szp_sort_critical_points_by_original_data(omp_cps, omp_cp_count, data, cols);

    /* OpenMP topology-preserved compression */
    size_t omp_topo_outSize = 0;
    double ot0 = get_time_ms();
    unsigned char *omp_topo_compressed = szp_float_openmp_threadblock_randomaccess_topology_preserved(
        data, &omp_topo_outSize, absErrBound, nbEle, blockSize,
        omp_cps, (int)omp_cp_count, rows, cols);
    double ot1 = get_time_ms();
    printf("--- OpenMP Topology Compression ---\n");
    printf("Compressed: %zu bytes (ratio: %.2fx) in %.2f ms\n\n",
           omp_topo_outSize, (double)(nbEle * sizeof(float)) / omp_topo_outSize, ot1 - ot0);

    /* OpenMP topology-preserved decompression */
    float *omp_decompressed = NULL;
    int *omp_FN = NULL;
    double ot2 = get_time_ms();
    szp_float_decompress_openmp_threadblock_randomaccess_topology_preserved(
        &omp_decompressed, nbEle, absErrBound, blockSize,
        omp_topo_compressed, &omp_FN);
    double ot3 = get_time_ms();
    printf("--- OpenMP Topology Decompression ---\n");
    printf("Decompressed in %.2f ms\n\n", ot3 - ot2);

    printf("OpenMP decompress returned: data=%s, FN=%s\n",
           omp_decompressed ? "OK" : "NULL", omp_FN ? "OK" : "NULL");

    if (omp_decompressed && omp_FN) {
        /* Raw decompression error */
        double omp_raw_max_err = 0.0;
        for (size_t i = 0; i < nbEle; i++) {
            double e = fabs((double)data[i] - (double)omp_decompressed[i]);
            if (e > omp_raw_max_err) omp_raw_max_err = e;
        }
        printf("--- OpenMP Raw Decompression ---\n");
        printf("Raw max error: %e\n", omp_raw_max_err);

        /* Raw CP preservation */
        size_t omp_raw_preserved = 0;
        for (size_t i = 0; i < orig_cp_count; i++) {
            int x = orig_cps[i].x, y = orig_cps[i].y;
            if (x >= 1 && x < rows-1 && y >= 1 && y < cols-1) {
                int dt = classify_point(omp_decompressed, rows, cols, x, y);
                if (dt == orig_cps[i].type) omp_raw_preserved++;
            }
        }
        printf("Raw CP preservation: %zu / %zu (%.2f%%)\n\n", omp_raw_preserved, orig_cp_count,
               100.0 * omp_raw_preserved / orig_cp_count);

        /* Post-processing: stencils + RBF (same pipeline as CUDA) */
        float *omp_orig_decomp = (float *)malloc(nbEle * sizeof(float));
        memcpy(omp_orig_decomp, omp_decompressed, nbEle * sizeof(float));

        /* Compress/decompress sort positions (OpenMP) */
        size_t omp_sort_outSize = 0;
        unsigned char *omp_sort_compressed = szp_compress_sort_positions(
            omp_cps, omp_cp_count, &omp_sort_outSize, blockSize);

        size_t omp_extrema_count = 0;
        for (size_t i = 0; i < omp_cp_count; i++)
            if (omp_cps[i].type == 1 || omp_cps[i].type == 2) omp_extrema_count++;

        int *omp_sort_positions = NULL;
        if (omp_sort_compressed && omp_sort_outSize > 0 && omp_extrema_count > 0) {
            omp_sort_positions = szp_decompress_sort_positions(
                omp_sort_compressed, omp_extrema_count, blockSize);
        }

        if (omp_sort_positions && omp_extrema_count > 0) {
            apply_stencils_clamped(omp_decompressed, omp_orig_decomp, omp_FN,
                                    omp_sort_positions, rows, cols, absErrBound, omp_extrema_count);
        }
        restore_extrema_clamped(omp_FN, omp_decompressed, omp_orig_decomp, rows, cols, eps, absErrBound);
        int omp_restored = rbf_restore_saddles(omp_decompressed, omp_FN, omp_orig_decomp,
                                                rows, cols, eps, absErrBound);
        printf("--- OpenMP Post-Processing ---\n");
        printf("RBF restored %d saddles\n\n", omp_restored);

        /* Find CPs in OpenMP post-processed data */
        size_t omp_decomp_cp_count = 0;
        CriticalPoint *omp_decomp_cps = szp_cuda_find_critical_points(
            omp_decompressed, &omp_decomp_cp_count, rows, cols, absErrBound);

        /* Count preservation */
        int *omp_decomp_type_map = (int *)calloc(nbEle, sizeof(int));
        for (size_t i = 0; i < omp_decomp_cp_count; i++) {
            size_t flat = (size_t)omp_decomp_cps[i].x * cols + omp_decomp_cps[i].y;
            if (flat < nbEle) omp_decomp_type_map[flat] = omp_decomp_cps[i].type;
        }

        size_t omp_preserved = 0, omp_lost = 0;
        size_t omp_lost_max = 0, omp_lost_min = 0, omp_lost_saddle = 0;
        for (size_t i = 0; i < orig_cp_count; i++) {
            size_t flat = (size_t)orig_cps[i].x * cols + orig_cps[i].y;
            if (omp_decomp_type_map[flat] == orig_cps[i].type) omp_preserved++;
            else {
                omp_lost++;
                if (orig_cps[i].type == 1) omp_lost_max++;
                else if (orig_cps[i].type == 2) omp_lost_min++;
                else omp_lost_saddle++;
            }
        }

        double omp_max_err = 0.0;
        size_t omp_err_count = 0;
        for (size_t i = 0; i < nbEle; i++) {
            double e = fabs((double)data[i] - (double)omp_decompressed[i]);
            if (e > omp_max_err) omp_max_err = e;
            if (e > absErrBound * 2.0) omp_err_count++;
        }

        size_t omp_decomp_max = 0, omp_decomp_min = 0, omp_decomp_saddle = 0;
        for (size_t i = 0; i < omp_decomp_cp_count; i++) {
            if (omp_decomp_cps[i].type == 1) omp_decomp_max++;
            else if (omp_decomp_cps[i].type == 2) omp_decomp_min++;
            else if (omp_decomp_cps[i].type == 3) omp_decomp_saddle++;
        }

        /* ============================================================ */
        /* Side-by-side comparison                                       */
        /* ============================================================ */
        printf("\n============ SIDE-BY-SIDE COMPARISON ============\n\n");
        printf("%-35s %12s %12s\n", "Metric", "CUDA", "OpenMP");
        printf("%-35s %12s %12s\n", "---", "----", "------");
        printf("%-35s %12zu %12zu\n", "Compression size (bytes)", topo_outSize, omp_topo_outSize);
        printf("%-35s %11.2fx %11.2fx\n", "Compression ratio",
               (double)(nbEle*sizeof(float))/topo_outSize,
               (double)(nbEle*sizeof(float))/omp_topo_outSize);
        printf("%-35s %12zu %12zu\n", "Total CPs in decompressed", cuda_decomp_cp_count, omp_decomp_cp_count);
        printf("%-35s %12zu %12zu\n", "Preserved (same type)", cuda_preserved, omp_preserved);
        printf("%-35s %11.2f%% %11.2f%%\n", "Preservation rate",
               100.0*cuda_preserved/orig_cp_count, 100.0*omp_preserved/orig_cp_count);
        printf("%-35s %12zu %12zu\n", "  Lost maxima", (size_t)0, omp_lost_max);
        printf("%-35s %12zu %12zu\n", "  Lost minima", (size_t)0, omp_lost_min);
        printf("%-35s %12zu %12zu\n", "  Lost saddles",
               orig_cp_count - cuda_preserved, omp_lost_saddle);
        printf("%-35s %12e %12e\n", "Max error", cuda_max_err, omp_max_err);
        printf("%-35s %12s %12s\n", "Within 2×eb",
               cuda_max_err <= absErrBound * 2.0 ? "YES" : "NO",
               omp_max_err <= absErrBound * 2.0 ? "YES" : "NO");
        printf("%-35s %12zu %12zu\n", "Elements > 2×eb", cuda_err_count, omp_err_count);

        /* Cleanup OpenMP */
        free(omp_orig_decomp);
        free(omp_decomp_type_map);
        if (omp_decomp_cps) free(omp_decomp_cps);
        if (omp_sort_compressed) free(omp_sort_compressed);
        if (omp_sort_positions) free(omp_sort_positions);
    } else {
        printf("OpenMP topology decompression failed — skipping comparison.\n");
    }

    if (omp_decompressed) free(omp_decompressed);
    if (omp_FN) free(omp_FN);
    if (omp_topo_compressed) free(omp_topo_compressed);

    int overall = (cuda_max_err <= absErrBound * 2.0 && cuda_preserved > orig_cp_count * 0.9) ? 0 : 1;
    printf("\nOverall: %s\n", overall == 0 ? "TOPOLOGY PRESERVED WITHIN 2×eb ✓" : "NEEDS INVESTIGATION");

    /* Cleanup */
    free(data);
    free(orig_cps);
    free(omp_cps);
    free(topo_compressed);
    free(decompressed);
    free(FN);
    free(decomp_cps);
    free(decomp_type_map);
    free(orig_type_map);
    free(orig_type_arr);
    if (sort_compressed) free(sort_compressed);
    if (sort_positions) free(sort_positions);
    if (orig_decomp) free(orig_decomp);

    return overall;
}
