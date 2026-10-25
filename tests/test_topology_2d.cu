/**
 *  @file test_topology_2d.cu
 *  @brief Automated test: 2D topology preservation via CUDA pipeline.
 *
 *  Generates a 2D synthetic dataset with analytically known critical points
 *  (sin(x)*cos(y)), runs the full topology pipeline (find CPs → sort →
 *  compress → decompress), and asserts:
 *    - critical points are found
 *    - error bound is satisfied
 *    - high CP preservation rate
 *
 *  Returns 0 on pass, 1 on failure.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

#include "szp.h"
#include "szp_cuda_compress.cuh"
#include "szp_cuda_decompress.cuh"
#include "szp_cuda_topology.cuh"

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

/* Generate 2D data with known critical points: f(x,y) = sin(ax)*cos(by) */
static void generate_2d_topo_data(float *data, int rows, int cols) {
    float ax = 2.0f * (float)M_PI / (cols - 1) * 3.0f;  /* 3 full periods in x */
    float by = 2.0f * (float)M_PI / (rows - 1) * 3.0f;  /* 3 full periods in y */
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i * cols + j] = sinf(ax * j) * cosf(by * i) * 100.0f;
}

int main() {
    printf("=== test_topology_2d ===\n\n");

    /* CUDA warmup */
    {
        void *d_tmp;
        if (cudaSetDevice(0) != cudaSuccess) {
            printf("SKIP: no CUDA device available\n");
            return 0;
        }
        cudaMalloc(&d_tmp, 1);
        cudaFree(d_tmp);
        cudaDeviceSynchronize();
    }

    int failures = 0;
    int rows = 200, cols = 360;
    size_t nbEle = (size_t)rows * cols;
    float absErrBound = 1e-3f;
    int blockSize = 64;

    float *data = (float *)malloc(nbEle * sizeof(float));
    generate_2d_topo_data(data, rows, cols);

    printf("Grid: %d x %d = %zu elements, eb=%e, blockSize=%d\n\n",
           rows, cols, nbEle, absErrBound, blockSize);

    /* ---- Step 1: Find critical points ---- */
    printf("Step 1: Find critical points...\n");
    size_t cp_count = 0;
    CriticalPoint *cps = szp_cuda_find_critical_points(data, &cp_count,
                                                        rows, cols, absErrBound);
    if (!cps || cp_count == 0) {
        printf("  FAIL: no critical points found\n");
        free(data);
        return 1;
    }

    size_t nmax = 0, nmin = 0, nsaddle = 0;
    for (size_t i = 0; i < cp_count; i++) {
        if (cps[i].type == 1) nmax++;
        else if (cps[i].type == 2) nmin++;
        else if (cps[i].type == 3) nsaddle++;
    }
    printf("  Found %zu CPs (max=%zu min=%zu saddle=%zu)\n", cp_count, nmax, nmin, nsaddle);

    if (nmax == 0 || nmin == 0) {
        printf("  FAIL: expected maxima and minima in sin*cos data\n");
        failures++;
    } else {
        printf("  PASS: found maxima and minima\n");
    }

    /* ---- Step 2: Sort critical points ---- */
    printf("\nStep 2: Sort critical points...\n");
    szp_cuda_sort_critical_points_by_original_data(cps, cp_count, data, cols);
    printf("  Done.\n");

    /* ---- Step 3: Topology-preserved compression ---- */
    printf("\nStep 3: Topology-preserved compression...\n");
    size_t outSize = 0;
    unsigned char *compressed = szp_cuda_float_compress_randomaccess_topology_preserved(
        data, &outSize, absErrBound, nbEle, blockSize,
        cps, (int)cp_count, rows, cols);

    if (!compressed || outSize == 0) {
        printf("  FAIL: compression returned NULL or size 0\n");
        free(data); free(cps);
        return 1;
    }
    printf("  Compressed: %zu bytes (ratio: %.2fx)\n",
           outSize, (double)(nbEle * sizeof(float)) / outSize);

    /* ---- Step 4: Topology-preserved decompression ---- */
    printf("\nStep 4: Decompress with topology...\n");
    float *decompressed = NULL;
    int *FN = NULL;
    szp_cuda_float_decompress_randomaccess_topology_preserved(
        &decompressed, nbEle, absErrBound, blockSize, compressed, &FN);

    if (!decompressed || !FN) {
        printf("  FAIL: decompression returned NULL\n");
        free(data); free(cps); free(compressed);
        return 1;
    }
    printf("  Decompressed OK.\n");

    /* ---- Step 5: Verify error bound ---- */
    printf("\nStep 5: Verify error bound...\n");
    double max_err = 0.0;
    size_t err_count = 0;
    for (size_t i = 0; i < nbEle; i++) {
        double err = fabs((double)data[i] - (double)decompressed[i]);
        if (err > max_err) max_err = err;
        if (err > absErrBound * 1.01) err_count++;
    }
    if (err_count > 0) {
        printf("  FAIL: %zu elements exceed error bound (max_err=%e)\n", err_count, max_err);
        failures++;
    } else {
        printf("  PASS: max_err=%e within bound %e\n", max_err, (double)absErrBound);
    }

    /* ---- Step 6: Verify FN array ---- */
    printf("\nStep 6: Verify FN array...\n");
    {
        size_t fn_nonzero = 0;
        for (size_t i = 0; i < nbEle; i++)
            if (FN[i] != 0) fn_nonzero++;

        if (fn_nonzero != cp_count) {
            printf("  WARN: FN has %zu non-zero entries, expected %zu\n", fn_nonzero, cp_count);
        } else {
            printf("  PASS: FN has %zu non-zero entries matching CP count\n", fn_nonzero);
        }
    }

    /* ---- Step 7: Verify CP preservation ---- */
    printf("\nStep 7: Verify CP preservation (raw, before post-processing)...\n");
    {
        size_t preserved = 0;
        for (size_t i = 0; i < cp_count; i++) {
            int x = cps[i].x, y = cps[i].y;
            if (x >= 1 && x < rows-1 && y >= 1 && y < cols-1) {
                int dt = classify_point(decompressed, rows, cols, x, y);
                if (dt == cps[i].type) preserved++;
            }
        }
        double rate = 100.0 * preserved / cp_count;
        printf("  Preserved: %zu / %zu (%.2f%%)\n", preserved, cp_count, rate);

        /* Extrema should be well-preserved even before post-processing */
        size_t extrema_total = nmax + nmin;
        size_t extrema_preserved = 0;
        for (size_t i = 0; i < cp_count; i++) {
            if (cps[i].type != 1 && cps[i].type != 2) continue;
            int x = cps[i].x, y = cps[i].y;
            if (x >= 1 && x < rows-1 && y >= 1 && y < cols-1) {
                int dt = classify_point(decompressed, rows, cols, x, y);
                if (dt == cps[i].type) extrema_preserved++;
            }
        }
        double extrema_rate = extrema_total > 0
            ? 100.0 * extrema_preserved / extrema_total : 100.0;
        printf("  Extrema preserved: %zu / %zu (%.2f%%)\n",
               extrema_preserved, extrema_total, extrema_rate);

        /* Overall preservation should be above 50% even raw */
        if (rate < 50.0) {
            printf("  FAIL: preservation rate below 50%%\n");
            failures++;
        } else {
            printf("  PASS: preservation rate above threshold\n");
        }
    }

    /* ---- Cleanup ---- */
    free(data);
    free(cps);
    free(compressed);
    free(decompressed);
    free(FN);

    printf("\n=== %s (%d failure%s) ===\n",
           failures == 0 ? "ALL PASSED" : "FAILURES",
           failures, failures == 1 ? "" : "s");
    return failures > 0 ? 1 : 0;
}
