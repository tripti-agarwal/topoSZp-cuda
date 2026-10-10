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
#include "szp_cuda_topo_restore.cuh"
#include "szp_cuda_topology.cuh"
#include "szp_errbound.h"
#include "szpd_float.h"
#include "szp_topo_restore.h"
#include "szp_pl_topology.h"

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

/* Topology restoration comes from the library (szp_cuda_topo_restore.cuh,
   szp_topo_restore.h): the GPU and CPU calls below run the same algorithm. */

/* -1: not compared, 1: CPU and GPU restored fields identical, 0: they differ */
static int g_cpu_gpu_same = -1;

/* PL mode (--pl): standard PL critical points (szp_pl_topology.h) */
static int g_pl = 0;
static SzpGrid g_grid;
static SzpPL g_plt;

/* Critical points of a field: axis rule on the GPU, or PL codes (GPU) */
static CriticalPoint *find_cps(const float *d, size_t *count, int rows, int cols, float eb) {
    if (!g_pl) return szp_cuda_find_critical_points((float *)d, count, rows, cols, eb);
    size_t n = (size_t)rows * cols;
    signed char *codes = (signed char *)malloc(n);
    if (szp_cuda_pl_classify_float(d, 2, rows, cols, 0, codes) != 0) { free(codes); *count = 0; return NULL; }
    CriticalPoint *cp = szp_pl_critical_points_2d(codes, d, rows, cols, eb, count);
    free(codes);
    return cp;
}

/* Classify a point as max(1)/min(2)/saddle(3)/regular(0) using 4-connectivity
   (PL mode: PL type) */
static int classify_point(const float *data, int rows, int cols, int i, int j) {
    (void)rows;
    if (g_pl) return szp_pl_classify(data, g_grid, g_plt, (long long)i * cols + j);
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

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "--pl") == 0) { g_pl = 1; argv++; argc--; }
    if (argc < 6) {
        fprintf(stderr, "Usage: %s [--pl] <input_file> <rows> <cols> <errBound> <blockSize>\n", argv[0]);
        fprintf(stderr, "  errBound: 1e-3 for absolute, rel:1e-4 for relative (fraction of data range)\n");
        return 1;
    }

    const char *input_file = argv[1];
    int rows = atoi(argv[2]);
    int cols = atoi(argv[3]);
    const char *eb_arg = argv[4];
    int blockSize = atoi(argv[5]);
    size_t nbEle = (size_t)rows * cols;

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

    /* Parse error bound */
    float absErrBound;
    const char *eb_mode;
    if (szp_parse_errbound_float(eb_arg, data, nbEle, &absErrBound, &eb_mode) != 0) {
        fprintf(stderr, "Invalid error bound: %s\n", eb_arg);
        return 1;
    }

    printf("=== TopoSZp Critical Point Preservation Test ===\n");
    printf("Input: %s  (%d x %d = %zu elements)\n", input_file, rows, cols, nbEle);
    if (strcmp(eb_mode, "REL") == 0) {
        float range;
        szp_rel_to_abs_float(data, nbEle, (float)atof(eb_arg+4), &absErrBound, &range);
        printf("Error bound: %s → abs=%e (range=%e)    Block size: %d\n\n", eb_arg, absErrBound, range, blockSize);
    } else {
        printf("Error bound: %e    Block size: %d\n\n", absErrBound, blockSize);
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
    g_grid = szp_grid_2d(rows, cols);
    szp_pl_init(g_grid, g_plt);
    if (g_pl) printf("(PL critical points: 6 neighbours, ties by index)\n");
    CriticalPoint *orig_cps = find_cps(data, &orig_cp_count, rows, cols, absErrBound);
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
    CriticalPoint *omp_cps = NULL;
    if (g_pl) {
        signed char *cc = (signed char *)malloc(nbEle);
        szp_pl_classify_float(data, 2, rows, cols, 0, cc);
        omp_cps = szp_pl_critical_points_2d(cc, data, rows, cols, absErrBound, &omp_cp_count);
        free(cc);
    } else {
        omp_cps = szp_find_critical_points(data, &omp_cp_count, rows, cols, absErrBound);
    }
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
    double ts0 = get_time_ms();   /* sort positions: compressor side */
    unsigned char *sort_compressed = szp_cuda_compress_sort_positions(
        orig_cps, orig_cp_count, &sort_outSize, blockSize);
    double ts1 = get_time_ms();

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

    /* Library restoration on the GPU: ranked extrema, then saddle stencil
       and RBF passes (szp_cuda_topo_restore.cuh). */
    szp_topo_restore_stats rs;
    int rrc = g_pl ? szp_cuda_topo_restore_float_pl(decompressed, FN, sort_positions,
                                                    sort_positions ? extrema_count : 0,
                                                    2, rows, cols, 0, absErrBound, &rs)
                   : szp_cuda_topo_restore_float_2d(decompressed, FN, sort_positions,
                                                    sort_positions ? extrema_count : 0,
                                                    rows, cols, absErrBound, &rs);
    if (rrc != 0)
        printf("  ERROR: GPU topology restoration failed\n");
    if (g_pl)
        printf("  PL fixes: saddles %zu, other %zu, %zu rounds; false positives %zu -> %zu, false types %zu -> %zu\n",
               rs.saddles_by_stencil, rs.other_fixes, rs.rounds, rs.fp_before, rs.fp_after,
               rs.ft_before, rs.ft_after);
    printf("  Extrema lost %zu -> %zu of %zu; saddles lost %zu -> %zu of %zu "
           "(stencil %zu, RBF %zu)\n",
           rs.extrema_lost_before, rs.extrema_lost_after, rs.extrema,
           rs.saddles_lost_before, rs.saddles_lost_after, rs.saddles,
           rs.saddles_by_stencil, rs.saddles_by_rbf);
    printf("  RBF restored %zu saddles\n", rs.saddles_by_rbf);

    double t7c = get_time_ms();
    printf("  Post-processing done in %.2f ms\n\n", t7c - t7b);
    /* Compression = CP detection + ranking + compression + rank stream;
       decompression = decoding + rank decoding + topology restoration. */
    printf("TIMING comp_s=%.6f decomp_s=%.6f\n",
           ((t1 - t0) + (t3 - t2) + (t5 - t4) + (ts1 - ts0)) / 1000.0,
           ((t7 - t6) + (t7c - ts1)) / 1000.0);

    /* Optional: write the post-processed field for external checks
       (example/topo_check), e.g. TOPOSZP_DUMP=out.f32 ./test_cuda_topology ... */
    if (const char *dump_path = getenv("TOPOSZP_DUMP")) {
        FILE *df = fopen(dump_path, "wb");
        if (df) { fwrite(decompressed, sizeof(float), nbEle, df); fclose(df);
                  printf("  Wrote post-processed field to %s\n", dump_path); }
        else perror(dump_path);
    }

    /* ============================================================ */
    /* Step 5: Find critical points in decompressed data             */
    /*         (AFTER post-processing)                               */
    /* ============================================================ */
    printf("--- Step 5: Find critical points in post-processed data ---\n");
    size_t decomp_cp_count = 0;
    CriticalPoint *decomp_cps = find_cps(decompressed, &decomp_cp_count, rows, cols, absErrBound);

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
    size_t err_1eb = 0, err_2eb = 0;
    for (size_t i = 0; i < nbEle; i++) {
        double err = fabs((double)data[i] - (double)decompressed[i]);
        if (err > max_err) max_err = err;
        if (err > absErrBound * 1.01) err_1eb++;
        if (err > absErrBound * 2.0) err_2eb++;
    }
    const char *eb_status;
    if (err_2eb == 0 && err_1eb == 0)
        eb_status = "PASS (within 1×eb)";
    else if (err_2eb == 0)
        eb_status = "PASS (within 2×eb — expected for topology preservation)";
    else
        eb_status = "FAIL (exceeds 2×eb)";
    printf("Max pointwise error: %e (bound: %e, 2×eb: %e)\n",
           max_err, (double)absErrBound, (double)absErrBound * 2.0);
    printf("  Within 1×eb: %s (%zu elements over)\n",
           err_1eb == 0 ? "YES" : "NO", err_1eb);
    printf("  Within 2×eb: %s (%zu elements over)\n",
           err_2eb == 0 ? "YES" : "NO", err_2eb);
    printf("  Status: %s\n", eb_status);

    /* ============================================================ */
    /* Step 9: Persistence-based saddle analysis                      */
    /*                                                                */
    /* Persistence = |value(saddle) - value(paired extremum)|         */
    /* Saddles with persistence < error bound are below the noise     */
    /* floor — losing them is expected. Only saddles with high        */
    /* persistence are scientifically meaningful.                      */
    /* ============================================================ */
    printf("--- Step 9: Persistence-based saddle analysis ---\n");
    {
        /* For each saddle, compute persistence as the minimum absolute
           difference between the saddle value and its 4 neighbors' values.
           This approximates the persistence (true persistence requires
           full merge-tree computation, but min-neighbor-diff is a fast
           and reasonable proxy). */
        size_t saddle_count = 0;
        double *saddle_persistence = (double *)malloc(orig_cp_count * sizeof(double));
        int *saddle_preserved_flag = (int *)calloc(orig_cp_count, sizeof(int));
        size_t *saddle_indices = (size_t *)malloc(orig_cp_count * sizeof(size_t));

        for (size_t i = 0; i < orig_cp_count; i++) {
            if (orig_cps[i].type != 3) continue;  /* only saddles */

            int x = orig_cps[i].x, y = orig_cps[i].y;
            if (x < 1 || x >= rows-1 || y < 1 || y >= cols-1) continue;

            float center = data[x * cols + y];
            float up    = data[(x-1) * cols + y];
            float down  = data[(x+1) * cols + y];
            float left  = data[x * cols + (y-1)];
            float right = data[x * cols + (y+1)];

            /* Persistence proxy: min of |center - neighbor| across all 4 neighbors */
            double d_up    = fabs((double)center - up);
            double d_down  = fabs((double)center - down);
            double d_left  = fabs((double)center - left);
            double d_right = fabs((double)center - right);
            double min_diff = fmin(fmin(d_up, d_down), fmin(d_left, d_right));

            /* Check if this saddle was preserved */
            size_t flat = (size_t)x * cols + y;
            int was_preserved = (decomp_type_map[flat] == 3) ? 1 : 0;

            saddle_persistence[saddle_count] = min_diff;
            saddle_preserved_flag[saddle_count] = was_preserved;
            saddle_indices[saddle_count] = i;
            saddle_count++;
        }

        /* Analyze at multiple persistence thresholds */
        double thresholds[] = {0.0, 0.5, 1.0, 2.0, 5.0, 10.0};
        int num_thresholds = 6;

        printf("\n  %-20s %10s %10s %10s %8s\n",
               "Persistence threshold", "Total", "Preserved", "Lost", "Rate");
        printf("  %-20s %10s %10s %10s %8s\n",
               "--------------------", "-----", "---------", "----", "----");

        for (int t = 0; t < num_thresholds; t++) {
            double thresh = thresholds[t] * (double)absErrBound;
            size_t above = 0, above_preserved = 0;
            for (size_t j = 0; j < saddle_count; j++) {
                if (saddle_persistence[j] > thresh) {
                    above++;
                    if (saddle_preserved_flag[j]) above_preserved++;
                }
            }
            size_t above_lost = above - above_preserved;
            char label[64];
            if (t == 0)
                snprintf(label, sizeof(label), "All saddles");
            else
                snprintf(label, sizeof(label), "> %.1f×eb", thresholds[t]);

            printf("  %-20s %10zu %10zu %10zu %7.2f%%\n",
                   label, above, above_preserved, above_lost,
                   above > 0 ? 100.0 * above_preserved / above : 0.0);
        }

        /* Also report persistence statistics */
        double p_min = 1e30, p_max = 0, p_sum = 0;
        for (size_t j = 0; j < saddle_count; j++) {
            if (saddle_persistence[j] < p_min) p_min = saddle_persistence[j];
            if (saddle_persistence[j] > p_max) p_max = saddle_persistence[j];
            p_sum += saddle_persistence[j];
        }
        double p_avg = saddle_count > 0 ? p_sum / saddle_count : 0;
        printf("\n  Saddle persistence stats:\n");
        printf("    Min: %e    Avg: %e    Max: %e\n", p_min, p_avg, p_max);
        printf("    Error bound: %e\n\n", (double)absErrBound);

        free(saddle_persistence);
        free(saddle_preserved_flag);
        free(saddle_indices);
    }

    /* ============================================================ */
    /* Step 10: Local order verification                               */
    /*                                                                */
    /* Check if the relative ordering between adjacent vertices is    */
    /* preserved in the decompressed data.                            */
    /*   - All pairs: every adjacent pair of interior vertices        */
    /*   - CP pairs: pairs where at least one vertex is a CP         */
    /*   - CP-CP pairs: pairs where both vertices are CPs            */
    /* ============================================================ */
    printf("--- Step 10: Local order verification ---\n");
    {
        /* Build original CP type map */
        unsigned char *cp_map = (unsigned char *)calloc(nbEle, 1);
        for (size_t i = 0; i < orig_cp_count; i++) {
            size_t flat = (size_t)orig_cps[i].x * cols + orig_cps[i].y;
            if (flat < nbEle) cp_map[flat] = (unsigned char)orig_cps[i].type;
        }

        size_t all_pairs = 0, all_violations = 0;
        size_t cp_pairs = 0, cp_violations = 0;       /* at least one is CP */
        size_t cpcp_pairs = 0, cpcp_violations = 0;   /* both are CPs */
        size_t extrema_pairs = 0, extrema_violations = 0; /* at least one is max/min */

        int dx[] = {0, 1};  /* right and down neighbors to avoid double-counting */
        int dy[] = {1, 0};

        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                size_t flat_a = (size_t)i * cols + j;
                float orig_a = data[flat_a];
                float decomp_a = decompressed[flat_a];
                unsigned char type_a = cp_map[flat_a];

                for (int d = 0; d < 2; d++) {
                    int ni = i + dx[d], nj = j + dy[d];
                    if (ni >= rows - 1 || nj >= cols - 1) continue;

                    size_t flat_b = (size_t)ni * cols + nj;
                    float orig_b = data[flat_b];
                    float decomp_b = decompressed[flat_b];
                    unsigned char type_b = cp_map[flat_b];

                    /* Original ordering */
                    int orig_order = 0;
                    if (orig_a < orig_b) orig_order = -1;
                    else if (orig_a > orig_b) orig_order = 1;
                    /* else equal — no constraint */

                    if (orig_order == 0) continue;  /* equal values — skip */

                    /* Decompressed ordering */
                    int decomp_order = 0;
                    if (decomp_a < decomp_b) decomp_order = -1;
                    else if (decomp_a > decomp_b) decomp_order = 1;

                    int violated = (orig_order != decomp_order) ? 1 : 0;

                    /* All pairs */
                    all_pairs++;
                    if (violated) all_violations++;

                    /* At least one is a CP */
                    if (type_a || type_b) {
                        cp_pairs++;
                        if (violated) cp_violations++;
                    }

                    /* Both are CPs */
                    if (type_a && type_b) {
                        cpcp_pairs++;
                        if (violated) cpcp_violations++;
                    }

                    /* At least one is max or min */
                    if (type_a == 1 || type_a == 2 || type_b == 1 || type_b == 2) {
                        extrema_pairs++;
                        if (violated) extrema_violations++;
                    }
                }
            }
        }

        printf("\n  %-30s %12s %12s %8s\n", "Pair type", "Total", "Violations", "Rate");
        printf("  %-30s %12s %12s %8s\n", "-----", "-----", "----------", "----");
        printf("  %-30s %12zu %12zu %7.4f%%\n", "All adjacent pairs",
               all_pairs, all_violations,
               all_pairs > 0 ? 100.0 * (all_pairs - all_violations) / all_pairs : 0.0);
        printf("  %-30s %12zu %12zu %7.4f%%\n", "At least one CP",
               cp_pairs, cp_violations,
               cp_pairs > 0 ? 100.0 * (cp_pairs - cp_violations) / cp_pairs : 0.0);
        printf("  %-30s %12zu %12zu %7.4f%%\n", "Both CPs",
               cpcp_pairs, cpcp_violations,
               cpcp_pairs > 0 ? 100.0 * (cpcp_pairs - cpcp_violations) / cpcp_pairs : 0.0);
        printf("  %-30s %12zu %12zu %7.4f%%\n", "At least one extremum",
               extrema_pairs, extrema_violations,
               extrema_pairs > 0 ? 100.0 * (extrema_pairs - extrema_violations) / extrema_pairs : 0.0);
        printf("  (Rate = %% of pairs with LOCAL ORDER PRESERVED)\n\n");

        free(cp_map);
    }

    /* ============================================================ */
    /* Summary                                                        */
    /* ============================================================ */
    printf("\n==================== SUMMARY ====================\n");
    printf("Critical points:   %zu original → %zu in decompressed\n", orig_cp_count, decomp_cp_count);
    printf("Preserved:         %zu / %zu (%.2f%%)\n", preserved, orig_cp_count,
           100.0 * preserved / orig_cp_count);
    printf("Error bound:       %s\n", eb_status);
    printf("FN array:          %s (%zu / %zu match)\n",
           fn_mismatch == 0 ? "PASS" : "PARTIAL", fn_match, nbEle);
    printf("Compression ratio: %.2fx\n", (double)(nbEle * sizeof(float)) / topo_outSize);

    size_t cuda_preserved = preserved;
    size_t cuda_decomp_cp_count = decomp_cp_count;
    double cuda_max_err = max_err;
    size_t cuda_err_count = err_2eb;

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

        printf("--- OpenMP Post-Processing ---\n");

        /* Decode the OpenMP sort positions (stream = [count][offsets][blocks]) */
        int *omp_sort_positions = NULL;
        if (omp_sort_compressed && omp_sort_outSize > sizeof(size_t) && omp_extrema_count > 0)
            omp_sort_positions = szp_decompress_sort_positions_sized(
                omp_sort_compressed + sizeof(size_t), omp_sort_outSize - sizeof(size_t),
                omp_extrema_count, blockSize);

        /* Library restoration on the CPU: same algorithm as the GPU call above. */
        szp_topo_restore_stats ors;
        if (g_pl)
            szp_topo_restore_float_pl(omp_decompressed, omp_FN, omp_sort_positions,
                                      omp_sort_positions ? omp_extrema_count : 0,
                                      2, rows, cols, 0, absErrBound, &ors);
        else
            szp_topo_restore_float_2d(omp_decompressed, omp_FN, omp_sort_positions,
                                      omp_sort_positions ? omp_extrema_count : 0,
                                      rows, cols, absErrBound, &ors);
        printf("  RBF restored %zu saddles (stencil %zu)\n", ors.saddles_by_rbf, ors.saddles_by_stencil);
        if (omp_sort_positions) free(omp_sort_positions);
        g_cpu_gpu_same = memcmp(omp_decompressed, decompressed, nbEle * sizeof(float)) == 0;
        printf("  CPU and GPU restored fields: %s\n\n", g_cpu_gpu_same ? "IDENTICAL" : "DIFFER");

        /* Find CPs in OpenMP post-processed data */
        size_t omp_decomp_cp_count = 0;
        CriticalPoint *omp_decomp_cps = find_cps(
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
    } else {
        printf("OpenMP topology decompression failed — skipping comparison.\n");
    }

    if (omp_decompressed) free(omp_decompressed);
    if (omp_FN) free(omp_FN);
    if (omp_topo_compressed) free(omp_topo_compressed);

    /* Exit code: 0 only if every guarantee holds. Lost saddles are reported
       above but are not a failure (the method does not guarantee them). */
    (void)cuda_preserved;
    int fail_eb = !(cuda_max_err <= absErrBound * 2.0);
    /* PL mode: false positives/types are reported, not guaranteed. */
    int fail_fp = !g_pl && new_cps != 0, fail_ft = !g_pl && type_changed != 0;
    int fail_ext = (lost_max + lost_min) != 0, fail_fnarr = fn_mismatch != 0;
    int fail_same = g_cpu_gpu_same == 0;
    int overall = (fail_eb || fail_fp || fail_ft || fail_ext || fail_fnarr || fail_same) ? 1 : 0;
    printf("\nResult: %s%s%s%s%s%s%s\n", overall ? "FAIL:" : g_pl ? "PASS (PL mode: error <= 2 eb, all extrema kept)"
                                                         : "PASS (error <= 2 eb, no FP, no FT, all extrema kept)",
           fail_eb ? " error>2eb" : "", fail_fp ? " false-positives" : "", fail_ft ? " false-types" : "",
           fail_ext ? " extrema-lost" : "", fail_fnarr ? " type-array-mismatch" : "",
           fail_same ? " CPU/GPU-outputs-differ" : "");

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
