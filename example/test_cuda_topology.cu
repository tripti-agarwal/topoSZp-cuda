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

/* Apply stencils to all extrema using sort positions */
static void apply_stencils(float *data, const int *types, const int *sort_positions,
                            int rows, int cols, size_t extrema_count) {
    /* Build mapping from grid position to sort_position index.
       Extrema are enumerated in row-major order. */
    size_t sort_idx = 0;
    for (int i = 1; i < rows-1; i++) {
        for (int j = 1; j < cols-1; j++) {
            int idx = i*cols+j;
            int type = types[idx];
            if (type == 1 || type == 2) {
                int sp = (sort_idx < extrema_count) ? sort_positions[sort_idx] : 0;
                if (type == 1) apply_maxima_stencil(data, rows, cols, i, j, sp);
                else           apply_minima_stencil(data, rows, cols, i, j, sp);
                sort_idx++;
            }
        }
    }
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
    printf("--- Step 4b: Post-processing — restore critical points ---\n");
    double t7b = get_time_ms();

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
        /* The compressed sort positions have offset table at front */
        sort_positions = szp_cuda_decompress_sort_positions(
            sort_compressed + sizeof(size_t), /* skip first offset entry */
            extrema_count, blockSize);
    }

    /* Apply stencils to extrema using sort positions */
    if (sort_positions && extrema_count > 0) {
        apply_stencils(decompressed, FN, sort_positions, rows, cols, extrema_count);
        printf("  Applied stencils to %zu extrema\n", extrema_count);
    }

    /* Final enforcement: ensure extrema are strictly above/below neighbors */
    float eps = fmaxf(1e-6f, 0.1f * absErrBound);
    restore_extrema_from_types(FN, decompressed, rows, cols, eps);

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

    int overall = (err_count == 0 && preserved == orig_cp_count) ? 0 : 1;
    printf("\nOverall: %s\n", overall == 0 ? "ALL CRITICAL POINTS PRESERVED ✓" :
           (preserved > orig_cp_count * 0.99) ? "MOSTLY PRESERVED (>99%)" : "SOME CRITICAL POINTS LOST");

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

    return overall;
}
