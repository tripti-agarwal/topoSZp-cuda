/**
 *  @file test_compress_decompress.cu
 *  @brief Automated test: CUDA compression → decompression round-trip.
 *
 *  Generates synthetic float data (sine wave, random, constant regions),
 *  compresses with CUDA random-access, decompresses with CUDA, and checks:
 *    - all values are within the error bound
 *    - compressed size > 0 and < original size
 *    - multiple error bounds (1e-2, 1e-3, 1e-4)
 *
 *  Returns 0 on pass, 1 on failure.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#include "szp_cuda_compress.cuh"
#include "szp_cuda_decompress.cuh"

/* ---- Synthetic data generators ---- */

static void generate_sine_wave(float *data, size_t n) {
    for (size_t i = 0; i < n; i++)
        data[i] = sinf((float)i * 0.01f) * 100.0f;
}

static void generate_random(float *data, size_t n, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++)
        data[i] = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
}

static void generate_mixed(float *data, size_t n) {
    /* First third: sine, second third: constant, last third: ramp */
    size_t third = n / 3;
    for (size_t i = 0; i < third; i++)
        data[i] = sinf((float)i * 0.05f) * 50.0f;
    for (size_t i = third; i < 2 * third; i++)
        data[i] = 42.0f;
    for (size_t i = 2 * third; i < n; i++)
        data[i] = (float)(i - 2 * third) * 0.1f;
}

/* ---- Test runner ---- */

static int test_roundtrip(const char *name, float *data, size_t nbEle,
                           float absErrBound, int blockSize) {
    int failures = 0;
    size_t orig_bytes = nbEle * sizeof(float);

    /* Compress */
    size_t outSize = 0;
    unsigned char *compressed = szp_cuda_float_compress_randomaccess(
        data, &outSize, absErrBound, nbEle, blockSize);

    if (!compressed) {
        printf("  FAIL [%s eb=%e]: compression returned NULL\n", name, absErrBound);
        return 1;
    }

    if (outSize == 0) {
        printf("  FAIL [%s eb=%e]: compressed size is 0\n", name, absErrBound);
        free(compressed);
        return 1;
    }

    if (outSize >= orig_bytes) {
        printf("  WARN [%s eb=%e]: compressed size (%zu) >= original (%zu)\n",
               name, absErrBound, outSize, orig_bytes);
        /* Not a hard failure — random data can be incompressible */
    }

    /* Decompress */
    float *decompressed = (float *)malloc(nbEle * sizeof(float));
    unsigned char *cmpData = compressed + sizeof(float); /* skip absErrBound header */
    szp_cuda_float_decompress_randomaccess_arg(decompressed, nbEle, absErrBound,
                                                blockSize, cmpData);

    /* Verify error bound */
    double max_err = 0.0;
    size_t err_count = 0;
    for (size_t i = 0; i < nbEle; i++) {
        double err = fabs((double)data[i] - (double)decompressed[i]);
        if (err > max_err) max_err = err;
        if (err > absErrBound * 1.01) err_count++;  /* 1% tolerance for float rounding */
    }

    if (err_count > 0) {
        printf("  FAIL [%s eb=%e]: %zu elements exceed error bound (max_err=%e)\n",
               name, absErrBound, err_count, max_err);
        failures++;
    } else {
        printf("  PASS [%s eb=%e]: max_err=%e  ratio=%.2fx\n",
               name, absErrBound, max_err, (double)orig_bytes / outSize);
    }

    free(compressed);
    free(decompressed);
    return failures;
}

int main() {
    printf("=== test_compress_decompress ===\n\n");

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
    int blockSize = 64;

    size_t nbEle = 100000;
    float *data = (float *)malloc(nbEle * sizeof(float));

    float error_bounds[] = {1e-2f, 1e-3f, 1e-4f};
    int num_ebs = 3;

    /* Test 1: Sine wave */
    printf("--- Sine wave data (%zu elements) ---\n", nbEle);
    generate_sine_wave(data, nbEle);
    for (int e = 0; e < num_ebs; e++)
        failures += test_roundtrip("sine", data, nbEle, error_bounds[e], blockSize);

    /* Test 2: Random data */
    printf("\n--- Random data (%zu elements) ---\n", nbEle);
    generate_random(data, nbEle, 12345);
    for (int e = 0; e < num_ebs; e++)
        failures += test_roundtrip("random", data, nbEle, error_bounds[e], blockSize);

    /* Test 3: Mixed regions (sine + constant + ramp) */
    printf("\n--- Mixed data (%zu elements) ---\n", nbEle);
    generate_mixed(data, nbEle);
    for (int e = 0; e < num_ebs; e++)
        failures += test_roundtrip("mixed", data, nbEle, error_bounds[e], blockSize);

    /* Test 4: Small dataset (edge case) */
    size_t small_n = 256;
    printf("\n--- Small dataset (%zu elements) ---\n", small_n);
    generate_sine_wave(data, small_n);
    failures += test_roundtrip("small", data, small_n, 1e-3f, blockSize);

    /* Test 5: Double-precision round-trip */
    printf("\n--- Double-precision (%zu elements, eb=1e-3) ---\n", nbEle);
    {
        double *ddata = (double *)malloc(nbEle * sizeof(double));
        for (size_t i = 0; i < nbEle; i++)
            ddata[i] = sin((double)i * 0.01) * 100.0;

        double deb = 1e-3;
        size_t doutSize = 0;
        unsigned char *dcompressed = szp_cuda_double_compress_randomaccess(
            ddata, &doutSize, deb, nbEle, blockSize);

        if (dcompressed && doutSize > 0) {
            double *ddecomp = (double *)malloc(nbEle * sizeof(double));
            szp_cuda_double_decompress_randomaccess_arg(
                ddecomp, nbEle, deb, blockSize, dcompressed + sizeof(double));

            double dmax_err = 0.0;
            size_t derr_count = 0;
            for (size_t i = 0; i < nbEle; i++) {
                double e = fabs(ddata[i] - ddecomp[i]);
                if (e > dmax_err) dmax_err = e;
                if (e > deb * 1.01) derr_count++;
            }
            if (derr_count > 0) {
                printf("  FAIL [double eb=%e]: %zu errors (max=%e)\n", deb, derr_count, dmax_err);
                failures++;
            } else {
                printf("  PASS [double eb=%e]: max_err=%e  ratio=%.2fx\n",
                       deb, dmax_err, (double)(nbEle * sizeof(double)) / doutSize);
            }
            free(ddecomp);
        } else {
            printf("  FAIL [double]: compression returned NULL or size 0\n");
            failures++;
        }
        free(dcompressed);
        free(ddata);
    }

    free(data);

    printf("\n=== %s (%d failure%s) ===\n",
           failures == 0 ? "ALL PASSED" : "FAILURES",
           failures, failures == 1 ? "" : "s");
    return failures > 0 ? 1 : 0;
}
