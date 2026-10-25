/**
 *  @file test_topology_3d.cu
 *  @brief Automated test: 3D topology preservation via CUDA pipeline.
 *
 *  Generates a 3D synthetic dataset with known critical points using
 *  f(x,y,z) = sin(ax)*cos(by)*sin(cz), runs the full 3D topology pipeline
 *  (find CPs → sort → compress → decompress), and asserts:
 *    - critical points are found (maxima, minima, saddles)
 *    - error bound is satisfied
 *    - reasonable CP preservation rate
 *
 *  Returns 0 on pass, 1 on failure.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#include "szp.h"
#include "szp_cuda_compress.cuh"
#include "szp_cuda_decompress.cuh"
#include "szp_cuda_topology_3d.cuh"

static inline size_t IDX(int x, int y, int z, int d2, int d3) {
    return (size_t)x * d2 * d3 + y * d3 + z;
}

/* Classify a 3D point using 6-connected neighborhood */
static int classify_3d(const float *data, int d1, int d2, int d3,
                        int x, int y, int z) {
    size_t s = (size_t)d2 * d3;
    float c = data[IDX(x,y,z,d2,d3)];
    float xm = data[IDX(x-1,y,z,d2,d3)], xp = data[IDX(x+1,y,z,d2,d3)];
    float ym = data[IDX(x,y-1,z,d2,d3)], yp = data[IDX(x,y+1,z,d2,d3)];
    float zm = data[IDX(x,y,z-1,d2,d3)], zp = data[IDX(x,y,z+1,d2,d3)];
    (void)s;

    /* Check max/min */
    if (c > xm && c > xp && c > ym && c > yp && c > zm && c > zp) return 1;
    if (c < xm && c < xp && c < ym && c < yp && c < zm && c < zp) return 2;

    /* Saddle: high on some axes, low on others */
    int high_axes = 0, low_axes = 0;
    if (c > xm && c > xp) high_axes++;
    if (c < xm && c < xp) low_axes++;
    if (c > ym && c > yp) high_axes++;
    if (c < ym && c < yp) low_axes++;
    if (c > zm && c > zp) high_axes++;
    if (c < zm && c < zp) low_axes++;
    if (high_axes >= 1 && low_axes >= 1) return 3;

    return 0;
}

/* Generate 3D data with known critical points */
static void generate_3d_topo_data(float *data, int d1, int d2, int d3) {
    float ax = 2.0f * (float)M_PI / (d3 - 1) * 2.0f;
    float by = 2.0f * (float)M_PI / (d2 - 1) * 2.0f;
    float cz = 2.0f * (float)M_PI / (d1 - 1) * 2.0f;
    for (int x = 0; x < d1; x++)
        for (int y = 0; y < d2; y++)
            for (int z = 0; z < d3; z++)
                data[IDX(x,y,z,d2,d3)] = sinf(cz * x) * cosf(by * y) * sinf(ax * z) * 100.0f;
}

int main() {
    printf("=== test_topology_3d ===\n\n");

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
    int d1 = 40, d2 = 50, d3 = 60;
    size_t nbEle = (size_t)d1 * d2 * d3;
    float absErrBound = 1e-3f;
    int blockSize = 64;

    float *data = (float *)malloc(nbEle * sizeof(float));
    generate_3d_topo_data(data, d1, d2, d3);

    printf("Grid: %d x %d x %d = %zu elements, eb=%e, blockSize=%d\n\n",
           d1, d2, d3, nbEle, absErrBound, blockSize);

    /* ---- Step 1: Find 3D critical points ---- */
    printf("Step 1: Find 3D critical points...\n");
    size_t cp_count = 0;
    CriticalPoint3D *cps = szp_cuda_find_critical_points_3d(data, &cp_count,
                                                              d1, d2, d3, absErrBound);
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
        printf("  FAIL: expected maxima and minima in 3D sin*cos*sin data\n");
        failures++;
    } else {
        printf("  PASS: found maxima and minima\n");
    }

    if (nsaddle == 0) {
        printf("  WARN: no saddles found (may be expected for small grids)\n");
    } else {
        printf("  Found %zu saddles\n", nsaddle);
    }

    /* ---- Step 2: Sort critical points ---- */
    printf("\nStep 2: Sort 3D critical points...\n");
    szp_cuda_sort_critical_points_3d(cps, cp_count, data, d2, d3);
    printf("  Done.\n");

    /* ---- Step 3: 3D topology-preserved compression ---- */
    printf("\nStep 3: 3D topology-preserved compression...\n");
    size_t outSize = 0;
    unsigned char *compressed = szp_cuda_float_compress_topology_3d(
        data, &outSize, absErrBound, nbEle, blockSize,
        cps, (int)cp_count, d1, d2, d3);

    if (!compressed || outSize == 0) {
        printf("  FAIL: compression returned NULL or size 0\n");
        free(data); free(cps);
        return 1;
    }
    printf("  Compressed: %zu bytes (ratio: %.2fx)\n",
           outSize, (double)(nbEle * sizeof(float)) / outSize);

    /* ---- Step 4: Decompress with topology ---- */
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
            printf("  PASS: FN count matches CP count (%zu)\n", fn_nonzero);
        }
    }

    /* ---- Step 7: Verify 3D CP preservation ---- */
    printf("\nStep 7: Verify 3D CP preservation (raw)...\n");
    {
        size_t preserved = 0;
        for (size_t i = 0; i < cp_count; i++) {
            int x = cps[i].x, y = cps[i].y, z = cps[i].z;
            if (x >= 1 && x < d1-1 && y >= 1 && y < d2-1 && z >= 1 && z < d3-1) {
                int dt = classify_3d(decompressed, d1, d2, d3, x, y, z);
                if (dt == cps[i].type) preserved++;
            }
        }
        double rate = 100.0 * preserved / cp_count;
        printf("  Preserved: %zu / %zu (%.2f%%)\n", preserved, cp_count, rate);

        /* Extrema preservation */
        size_t extrema_total = nmax + nmin;
        size_t extrema_preserved = 0;
        for (size_t i = 0; i < cp_count; i++) {
            if (cps[i].type != 1 && cps[i].type != 2) continue;
            int x = cps[i].x, y = cps[i].y, z = cps[i].z;
            if (x >= 1 && x < d1-1 && y >= 1 && y < d2-1 && z >= 1 && z < d3-1) {
                int dt = classify_3d(decompressed, d1, d2, d3, x, y, z);
                if (dt == cps[i].type) extrema_preserved++;
            }
        }
        double extrema_rate = extrema_total > 0
            ? 100.0 * extrema_preserved / extrema_total : 100.0;
        printf("  Extrema preserved: %zu / %zu (%.2f%%)\n",
               extrema_preserved, extrema_total, extrema_rate);

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
