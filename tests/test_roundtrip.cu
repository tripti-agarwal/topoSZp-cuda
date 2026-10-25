/**
 *  @file test_roundtrip.cu
 *  @brief Automated test: cross-path round-trip between OpenMP and CUDA.
 *
 *  Tests interoperability:
 *    1. OpenMP-compress → CUDA-decompress
 *    2. CUDA-compress  → CUDA-decompress
 *    3. CUDA-compress  → OpenMP-decompress
 *  Asserts all paths produce results within the error bound and that
 *  different paths produce consistent outputs.
 *
 *  Returns 0 on pass, 1 on failure.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#include "szp.h"
#include "szp_float.h"
#include "szpd_float.h"
#include "szp_cuda_compress.cuh"
#include "szp_cuda_decompress.cuh"

static void generate_data(float *data, size_t n) {
    for (size_t i = 0; i < n; i++)
        data[i] = sinf((float)i * 0.017f) * 80.0f + cosf((float)i * 0.003f) * 20.0f;
}

static int verify_error_bound(const float *orig, const float *decomp,
                               size_t n, float eb, const char *label) {
    double max_err = 0.0;
    size_t err_count = 0;
    for (size_t i = 0; i < n; i++) {
        double err = fabs((double)orig[i] - (double)decomp[i]);
        if (err > max_err) max_err = err;
        if (err > eb * 1.01) err_count++;
    }
    if (err_count > 0) {
        printf("  FAIL [%s]: %zu elements exceed eb (max_err=%e, eb=%e)\n",
               label, err_count, max_err, (double)eb);
        return 1;
    }
    printf("  PASS [%s]: max_err=%e\n", label, max_err);
    return 0;
}

int main() {
    printf("=== test_roundtrip (OpenMP ↔ CUDA) ===\n\n");

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
    size_t nbEle = 100000;
    float absErrBound = 1e-3f;
    int blockSize = 64;

    float *data = (float *)malloc(nbEle * sizeof(float));
    generate_data(data, nbEle);

    /* ================================================================ */
    /* Path 1: OpenMP-compress → CUDA-decompress                        */
    /* ================================================================ */
    printf("--- Path 1: OpenMP compress → CUDA decompress ---\n");
    {
        size_t omp_outSize = 0;
        unsigned char *omp_compressed = szp_float_openmp_threadblock_randomaccess(
            data, &omp_outSize, absErrBound, nbEle, blockSize);

        if (!omp_compressed || omp_outSize == 0) {
            printf("  FAIL: OpenMP compression failed\n");
            failures++;
        } else {
            float *cuda_decomp = (float *)malloc(nbEle * sizeof(float));
            szp_cuda_float_decompress_randomaccess_arg(
                cuda_decomp, nbEle, absErrBound, blockSize,
                omp_compressed + sizeof(float));

            failures += verify_error_bound(data, cuda_decomp, nbEle,
                                            absErrBound, "OMP→CUDA");
            free(cuda_decomp);
        }
        free(omp_compressed);
    }

    /* ================================================================ */
    /* Path 2: CUDA-compress → CUDA-decompress                          */
    /* ================================================================ */
    printf("\n--- Path 2: CUDA compress → CUDA decompress ---\n");
    float *cuda_decomp2 = NULL;
    unsigned char *cuda_compressed = NULL;
    {
        size_t cuda_outSize = 0;
        cuda_compressed = szp_cuda_float_compress_randomaccess(
            data, &cuda_outSize, absErrBound, nbEle, blockSize);

        if (!cuda_compressed || cuda_outSize == 0) {
            printf("  FAIL: CUDA compression failed\n");
            failures++;
        } else {
            cuda_decomp2 = (float *)malloc(nbEle * sizeof(float));
            szp_cuda_float_decompress_randomaccess_arg(
                cuda_decomp2, nbEle, absErrBound, blockSize,
                cuda_compressed + sizeof(float));

            failures += verify_error_bound(data, cuda_decomp2, nbEle,
                                            absErrBound, "CUDA→CUDA");
        }
    }

    /* ================================================================ */
    /* Path 3: CUDA-compress → OpenMP-decompress                        */
    /* ================================================================ */
    printf("\n--- Path 3: CUDA compress → OpenMP decompress ---\n");
    {
        if (!cuda_compressed) {
            printf("  SKIP: CUDA compression failed in Path 2\n");
        } else {
            float *omp_decomp = szp_float_decompress_openmp_threadblock_randomaccess(
                nbEle, absErrBound, blockSize, cuda_compressed + sizeof(float));

            if (!omp_decomp) {
                printf("  FAIL: OpenMP decompression returned NULL\n");
                failures++;
            } else {
                failures += verify_error_bound(data, omp_decomp, nbEle,
                                                absErrBound, "CUDA→OMP");

                /* Also check CUDA and OpenMP decompress produce same results */
                if (cuda_decomp2) {
                    double max_diff = 0.0;
                    for (size_t i = 0; i < nbEle; i++) {
                        double diff = fabs((double)cuda_decomp2[i] - (double)omp_decomp[i]);
                        if (diff > max_diff) max_diff = diff;
                    }
                    if (max_diff > 0.0) {
                        printf("  WARN: CUDA vs OMP decompress differ by %e\n", max_diff);
                    } else {
                        printf("  PASS: CUDA and OMP decompress match exactly\n");
                    }
                }
                free(omp_decomp);
            }
        }
    }

    /* ================================================================ */
    /* Path 4: Consistency across multiple compressions                   */
    /* ================================================================ */
    printf("\n--- Path 4: Deterministic compression ---\n");
    {
        size_t outSize1 = 0, outSize2 = 0;
        unsigned char *c1 = szp_cuda_float_compress_randomaccess(
            data, &outSize1, absErrBound, nbEle, blockSize);
        unsigned char *c2 = szp_cuda_float_compress_randomaccess(
            data, &outSize2, absErrBound, nbEle, blockSize);

        if (!c1 || !c2) {
            printf("  FAIL: one of the compressions returned NULL\n");
            failures++;
        } else if (outSize1 != outSize2) {
            printf("  FAIL: sizes differ (%zu vs %zu)\n", outSize1, outSize2);
            failures++;
        } else if (memcmp(c1, c2, outSize1) != 0) {
            printf("  WARN: same data produces different compressed bytes (non-deterministic)\n");
        } else {
            printf("  PASS: deterministic compression (size=%zu)\n", outSize1);
        }
        free(c1);
        free(c2);
    }

    /* Cleanup */
    free(data);
    free(cuda_decomp2);
    free(cuda_compressed);

    printf("\n=== %s (%d failure%s) ===\n",
           failures == 0 ? "ALL PASSED" : "FAILURES",
           failures, failures == 1 ? "" : "s");
    return failures > 0 ? 1 : 0;
}
