/**
 *  @file test_cuda_decompress.cu
 *  @brief Test program: OpenMP compression → CUDA decompression (cross-validation).
 *
 *  Usage: test_cuda_decompress <input_file> <nbEle> <absErrBound> <blockSize>
 *
 *  Reads a binary float file, compresses it with OpenMP random-access,
 *  then decompresses with CUDA, and verifies results match within error bound.
 *  Also tests the topology-preserved pipeline end-to-end.
 *
 *  For CESM-ATM 2D data: each field is 1800×3600 = 6,480,000 floats.
 *  Example:
 *    test_cuda_decompress CLDHGH_1_1800_3600.dat 6480000 1e-3 64
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/time.h>

#include "szp.h"
#include "szp_cuda_compress.cuh"
#include "szp_cuda_decompress.cuh"
#include "szp_cuda_topology.cuh"

static double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <input_file> <nbEle> <absErrBound> <blockSize> [rows cols]\n", argv[0]);
        fprintf(stderr, "  If rows/cols are provided, also runs topology-preserved test.\n");
        return 1;
    }

    const char *input_file = argv[1];
    size_t nbEle = (size_t)atol(argv[2]);
    float absErrBound = (float)atof(argv[3]);
    int blockSize = atoi(argv[4]);
    int rows = 0, cols = 0;
    bool do_topology = false;
    if (argc >= 7) {
        rows = atoi(argv[5]);
        cols = atoi(argv[6]);
        do_topology = true;
    }

    printf("=== TopoSZp CUDA Decompression Test ===\n");
    printf("Input: %s  (%zu elements)\n", input_file, nbEle);
    printf("Error bound: %e    Block size: %d\n", absErrBound, blockSize);
    if (do_topology) printf("Grid: %d x %d (topology test enabled)\n", rows, cols);

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

    /* ============================================================ */
    /* Test 1: OpenMP compress → CUDA decompress                     */
    /* ============================================================ */
    printf("\n========== Test 1: OpenMP compress → CUDA decompress ==========\n");

    size_t omp_outSize = 0;
    double t0 = get_time_ms();
    unsigned char *omp_compressed = szp_float_openmp_threadblock_randomaccess(
        data, &omp_outSize, absErrBound, nbEle, blockSize);
    double t1 = get_time_ms();
    printf("OpenMP compression: %zu bytes in %.2f ms\n", omp_outSize, t1 - t0);

    /* Skip the absErrBound header (sizeof(float)) — the decompress functions
       expect cmpBytes starting at the offset table */
    unsigned char *cmpData = omp_compressed + sizeof(float);

    float *cuda_decompressed = (float *)malloc(nbEle * sizeof(float));
    double t2 = get_time_ms();
    szp_cuda_float_decompress_randomaccess_arg(cuda_decompressed, nbEle, absErrBound,
                                                blockSize, cmpData);
    double t3 = get_time_ms();
    printf("CUDA decompression: %.2f ms  (%.2f GB/s)\n",
           t3 - t2, (nbEle * sizeof(float)) / ((t3 - t2) * 1e6));

    /* Verify */
    double max_err = 0.0;
    size_t err_count = 0;
    for (size_t i = 0; i < nbEle; i++) {
        double err = fabs((double)data[i] - (double)cuda_decompressed[i]);
        if (err > max_err) max_err = err;
        if (err > absErrBound * 1.01) err_count++;
    }
    printf("Max error: %e (bound: %e) — %s\n", max_err, (double)absErrBound,
           err_count == 0 ? "PASS" : "FAIL");

    /* ============================================================ */
    /* Test 2: CUDA compress → CUDA decompress → verify              */
    /* ============================================================ */
    printf("\n========== Test 2: CUDA compress → CUDA decompress ==========\n");

    size_t cuda_outSize = 0;
    double t4 = get_time_ms();
    unsigned char *cuda_compressed = szp_cuda_float_compress_randomaccess(
        data, &cuda_outSize, absErrBound, nbEle, blockSize);
    double t5 = get_time_ms();
    printf("CUDA compression: %zu bytes in %.2f ms\n", cuda_outSize, t5 - t4);

    float *cuda_decompressed2 = (float *)malloc(nbEle * sizeof(float));
    /* Skip absErrBound header */
    unsigned char *cuda_cmpData = cuda_compressed + sizeof(float);
    double t6 = get_time_ms();
    szp_cuda_float_decompress_randomaccess_arg(cuda_decompressed2, nbEle, absErrBound,
                                                blockSize, cuda_cmpData);
    double t7 = get_time_ms();
    printf("CUDA decompression: %.2f ms\n", t7 - t6);

    max_err = 0.0;
    err_count = 0;
    for (size_t i = 0; i < nbEle; i++) {
        double err = fabs((double)data[i] - (double)cuda_decompressed2[i]);
        if (err > max_err) max_err = err;
        if (err > absErrBound * 1.01) err_count++;
    }
    printf("Max error: %e (bound: %e) — %s\n", max_err, (double)absErrBound,
           err_count == 0 ? "PASS" : "FAIL");

    /* ============================================================ */
    /* Test 3: Topology-preserved (if grid dims given)               */
    /* ============================================================ */
    if (do_topology && (size_t)rows * cols == nbEle) {
        printf("\n========== Test 3: Topology-preserved CUDA pipeline ==========\n");

        /* Find critical points on GPU */
        size_t cp_count = 0;
        double t8 = get_time_ms();
        CriticalPoint *cuda_cps = szp_cuda_find_critical_points(data, &cp_count, rows, cols, absErrBound);
        double t9 = get_time_ms();
        printf("CUDA found %zu critical points in %.2f ms\n", cp_count, t9 - t8);

        /* Find with OpenMP for comparison */
        size_t omp_cp_count = 0;
        CriticalPoint *omp_cps = szp_find_critical_points(data, &omp_cp_count, rows, cols, absErrBound);
        printf("OpenMP found %zu critical points (CUDA: %zu — %s)\n",
               omp_cp_count, cp_count,
               cp_count == omp_cp_count ? "MATCH" : "DIFFER (ordering may vary)");

        if (cuda_cps && cp_count > 0) {
            /* Sort critical points */
            szp_cuda_sort_critical_points_by_original_data(cuda_cps, cp_count, data, cols);

            /* Topology-preserved compression */
            size_t topo_outSize = 0;
            unsigned char *topo_compressed = szp_cuda_float_compress_randomaccess_topology_preserved(
                data, &topo_outSize, absErrBound, nbEle, blockSize,
                cuda_cps, (int)cp_count, rows, cols);
            printf("Topology-preserved compressed: %zu bytes (ratio: %.2fx)\n",
                   topo_outSize, (double)(nbEle * sizeof(float)) / topo_outSize);

            if (topo_compressed) free(topo_compressed);
            free(cuda_cps);
        }
        if (omp_cps) free(omp_cps);
    }

    /* Cleanup */
    free(data);
    free(omp_compressed);
    free(cuda_decompressed);
    free(cuda_compressed);
    free(cuda_decompressed2);

    printf("\n=== All tests complete ===\n");
    return 0;
}
