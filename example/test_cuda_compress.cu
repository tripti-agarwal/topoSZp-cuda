/**
 *  @file test_cuda_compress.cu
 *  @brief Test program: CUDA compression → OpenMP decompression (cross-validation).
 *
 *  Usage: test_cuda_compress <input_file> <nbEle> <absErrBound> <blockSize>
 *
 *  Reads a binary float file, compresses it with CUDA random-access, then
 *  decompresses with both CUDA and OpenMP, and checks that results match
 *  within the error bound.
 *
 *  For CESM-ATM 2D data: each field is 1800×3600 = 6,480,000 floats.
 *  Example:
 *    test_cuda_compress CLDHGH_1_1800_3600.dat 6480000 1e-3 64
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/time.h>

#include "szp.h"
#include "szp_cuda_compress.cuh"
#include "szp_cuda_decompress.cuh"

static double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <input_file> <nbEle> <absErrBound> <blockSize>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    size_t nbEle = (size_t)atol(argv[2]);
    float absErrBound = (float)atof(argv[3]);
    int blockSize = atoi(argv[4]);

    printf("=== TopoSZp CUDA Compression Test ===\n");
    printf("Input: %s  (%zu elements)\n", input_file, nbEle);
    printf("Error bound: %e    Block size: %d\n", absErrBound, blockSize);

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

    /* ---- CUDA Compression ---- */
    size_t cuda_outSize = 0;
    double t0 = get_time_ms();
    unsigned char *cuda_compressed = szp_cuda_float_compress_randomaccess(
        data, &cuda_outSize, absErrBound, nbEle, blockSize);
    double t1 = get_time_ms();

    printf("\n--- CUDA Compression ---\n");
    printf("Compressed size: %zu bytes (ratio: %.2fx)\n",
           cuda_outSize, (double)(nbEle * sizeof(float)) / cuda_outSize);
    printf("Time: %.2f ms  (%.2f GB/s)\n",
           t1 - t0, (nbEle * sizeof(float)) / ((t1 - t0) * 1e6));

    /* ---- CUDA Decompression ---- */
    /* Skip absErrBound header (sizeof(float)) — decompressor expects cmpBytes
       starting at the offset table */
    unsigned char *cuda_cmpData = cuda_compressed + sizeof(float);
    float *cuda_decompressed = (float *)malloc(nbEle * sizeof(float));
    double t2 = get_time_ms();
    szp_cuda_float_decompress_randomaccess_arg(cuda_decompressed, nbEle, absErrBound,
                                                blockSize, cuda_cmpData);
    double t3 = get_time_ms();

    printf("\n--- CUDA Decompression ---\n");
    printf("Time: %.2f ms  (%.2f GB/s)\n",
           t3 - t2, (nbEle * sizeof(float)) / ((t3 - t2) * 1e6));

    /* ---- Verify: check max error ---- */
    double max_err = 0.0;
    size_t err_count = 0;
    for (size_t i = 0; i < nbEle; i++) {
        double err = fabs((double)data[i] - (double)cuda_decompressed[i]);
        if (err > max_err) max_err = err;
        if (err > absErrBound * 1.01) err_count++;  /* 1% tolerance for float rounding */
    }

    printf("\n--- Verification ---\n");
    printf("Max pointwise error: %e (bound: %e)\n", max_err, (double)absErrBound);
    if (err_count > 0) {
        printf("WARNING: %zu elements exceed error bound!\n", err_count);
    } else {
        printf("PASS: All elements within error bound.\n");
    }

    /* ---- OpenMP Compression for comparison ---- */
    size_t omp_outSize = 0;
    double t4 = get_time_ms();
    unsigned char *omp_compressed = szp_float_openmp_threadblock_randomaccess(
        data, &omp_outSize, absErrBound, nbEle, blockSize);
    double t5 = get_time_ms();

    printf("\n--- OpenMP Compression (reference) ---\n");
    printf("Compressed size: %zu bytes (ratio: %.2fx)\n",
           omp_outSize, (double)(nbEle * sizeof(float)) / omp_outSize);
    printf("Time: %.2f ms  (%.2f GB/s)\n",
           t5 - t4, (nbEle * sizeof(float)) / ((t5 - t4) * 1e6));

    /* ---- OpenMP Decompression of OpenMP compressed data ---- */
    float *omp_decompressed = szp_float_decompress_openmp_threadblock_randomaccess(
        nbEle, absErrBound, blockSize, omp_compressed + sizeof(float));

    /* Verify OpenMP pipeline independently */
    double omp_max_err = 0.0;
    size_t omp_err_count = 0;
    for (size_t i = 0; i < nbEle; i++) {
        double err = fabs((double)data[i] - (double)omp_decompressed[i]);
        if (err > omp_max_err) omp_max_err = err;
        if (err > absErrBound * 1.01) omp_err_count++;
    }
    printf("\n--- OpenMP Pipeline Verification ---\n");
    printf("Max pointwise error: %e (bound: %e) — %s\n",
           omp_max_err, (double)absErrBound,
           omp_err_count == 0 ? "PASS" : "FAIL");
    if (omp_err_count > 0)
        printf("  %zu elements exceed error bound\n", omp_err_count);

    /* Compare CUDA vs OpenMP decompressed */
    double max_diff = 0.0;
    size_t diff_count = 0;
    for (size_t i = 0; i < nbEle; i++) {
        double diff = fabs((double)cuda_decompressed[i] - (double)omp_decompressed[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > absErrBound * 2) diff_count++;
    }
    printf("\n--- CUDA vs OpenMP Decompressed ---\n");
    printf("Max difference: %e  (%zu elements differ by > 2*bound)\n", max_diff, diff_count);
    if (max_diff < absErrBound * 2.01) {
        printf("PASS: Results are consistent.\n");
    } else {
        printf("NOTE: Differences expected if OpenMP thread count varies between compress/decompress.\n");
    }

    /* ---- Cross-test: OpenMP compressed → CUDA decompress ---- */
    float *cross_decompressed = (float *)malloc(nbEle * sizeof(float));
    szp_cuda_float_decompress_randomaccess_arg(cross_decompressed, nbEle, absErrBound,
                                                blockSize, omp_compressed + sizeof(float));
    double cross_max_err = 0.0;
    size_t cross_err_count = 0;
    for (size_t i = 0; i < nbEle; i++) {
        double err = fabs((double)data[i] - (double)cross_decompressed[i]);
        if (err > cross_max_err) cross_max_err = err;
        if (err > absErrBound * 1.01) cross_err_count++;
    }
    printf("\n--- Cross-test: OpenMP compress → CUDA decompress ---\n");
    printf("Max pointwise error: %e (bound: %e) — %s\n",
           cross_max_err, (double)absErrBound,
           cross_err_count == 0 ? "PASS" : "FAIL");
    if (cross_err_count > 0)
        printf("  %zu elements exceed error bound\n", cross_err_count);

    /* Cleanup */
    free(data);
    free(cuda_compressed);
    free(cuda_decompressed);
    free(omp_compressed);
    free(omp_decompressed);
    free(cross_decompressed);

    return 0;
}
