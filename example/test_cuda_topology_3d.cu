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
 *    6. Post-processing: saddle stencil + RBF (3×3×3) for lost saddles, then
 *       ranked extrema restoration (szp_topo_restore.h). Saddle changes are
 *       rejected if they would change a neighbour's type.
 *    7. Find critical points in post-processed data
 *    8. Verify preservation, FN array, error bound
 *
 *  Usage: test_cuda_topology_3d [--pl] <input_file> <d1> <d2> <d3> <absErrBound> <blockSize>
 *  --pl: standard PL critical points (szp_pl_topology.h, 14 neighbours) instead
 *        of the 6-neighbour axis rule; the counts below then use the PL types
 *        (saddle kinds compared in the "PL check" line).
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
#include "szp_errbound.h"
#include "szp_cuda_topology_3d.cuh"
#include "szp_cuda_topo_restore.cuh"
#include "szp_topology_3d.h"
#include "szp_topo_restore.h"
#include "szp_pl_topology.h"

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

/* PL mode (--pl) */
static int g_pl = 0;
static SzpGrid g_grid;
static SzpPL g_plt;

/* Classify a voxel with 6-connected neighborhood (PL mode: PL type, saddle
   kinds folded into 3) */
static int classify_3d(const float *data, int d1, int d2, int d3,
                        int x, int y, int z)
{
    if (g_pl) {
        int c = szp_pl_classify(data, g_grid, g_plt, (long long)IDX(x, y, z, d2, d3));
        return c > 3 ? 3 : c;
    }
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

/* Topology restoration comes from the library (szp_cuda_topo_restore.cuh,
   szp_topo_restore.h), the same algorithm as the CPU path. */

/* ================================================================ */
/*  main                                                             */
/* ================================================================ */

int main(int argc, char *argv[])
{
    if (argc > 1 && strcmp(argv[1], "--pl") == 0) { g_pl = 1; argv++; argc--; }
    if (argc < 7) {
        fprintf(stderr,
            "Usage: %s [--pl] <input_file> <d1> <d2> <d3> <errBound> <blockSize>\n",
            argv[0]);
        fprintf(stderr, "  errBound: 1e-3 for absolute, rel:1e-4 for relative (fraction of data range)\n");
        return 1;
    }

    const char *input_file = argv[1];
    int d1 = atoi(argv[2]);
    int d2 = atoi(argv[3]);
    int d3 = atoi(argv[4]);
    const char *eb_arg = argv[5];
    int blockSize = atoi(argv[6]);
    size_t nbEle = (size_t)d1 * d2 * d3;

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

    /* Parse error bound */
    float absErrBound;
    const char *eb_mode;
    if (szp_parse_errbound_float(eb_arg, data, nbEle, &absErrBound, &eb_mode) != 0) {
        fprintf(stderr, "Invalid error bound: %s\n", eb_arg);
        return 1;
    }

    g_grid = szp_grid_3d(d1, d2, d3);
    szp_pl_init(g_grid, g_plt);
    printf("=== TopoSZp 3D Critical Point Preservation Test%s ===\n",
           g_pl ? " (PL critical points)" : "");
    printf("Input: %s  (%d × %d × %d = %zu elements)\n",
           input_file, d1, d2, d3, nbEle);
    if (strcmp(eb_mode, "REL") == 0) {
        float range;
        szp_rel_to_abs_float(data, nbEle, (float)atof(eb_arg+4), &absErrBound, &range);
        printf("Error bound: %s → abs=%e (range=%e)    Block size: %d\n\n",
               eb_arg, absErrBound, range, blockSize);
    } else {
        printf("Error bound: %e    Block size: %d\n\n", absErrBound, blockSize);
    }

    /* CUDA warmup */
    { void *t; cudaSetDevice(0); cudaMalloc(&t,1); cudaFree(t); cudaDeviceSynchronize(); }

    /* ============================================================ */
    /* Step 1: Find critical points (CUDA)                           */
    /* ============================================================ */
    printf("--- Step 1: Find 3D critical points in original data ---\n");
    size_t orig_cp_count = 0;
    signed char *pl_codes = NULL;
    double t0 = get_time_ms();
    CriticalPoint3D *orig_cps = NULL;
    if (g_pl) {
        pl_codes = (signed char *)malloc(nbEle);
        if (szp_cuda_pl_classify_float(data, 3, d1, d2, d3, pl_codes) != 0) {
            printf("ERROR: GPU PL classification failed\n"); return 1;
        }
        orig_cps = szp_pl_critical_points_3d(pl_codes, data, d1, d2, d3, absErrBound, &orig_cp_count);
    } else {
        orig_cps = szp_cuda_find_critical_points_3d(data, &orig_cp_count, d1, d2, d3, absErrBound);
    }
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
    CriticalPoint3D *omp_cps = NULL;
    if (g_pl) {
        signed char *cpu_codes = (signed char *)malloc(nbEle);
        szp_pl_classify_float(data, 3, d1, d2, d3, cpu_codes);
        int same = memcmp(cpu_codes, pl_codes, nbEle) == 0;
        size_t k1 = 0, k2 = 0, kx = 0;
        for (size_t i = 0; i < nbEle; i++) {
            omp_count += cpu_codes[i] != 0;
            k1 += pl_codes[i] == SZP_PL_SAD1; k2 += pl_codes[i] == SZP_PL_SAD2; kx += pl_codes[i] == SZP_PL_SADX;
        }
        free(cpu_codes);
        printf("  1-saddles: %zu  2-saddles: %zu  degenerate: %zu\n", k1, k2, kx);
        printf("OpenMP found %zu (CUDA %zu — %s)\n\n", omp_count, orig_cp_count,
               same ? "MATCH" : "DIFFER");
    } else {
        omp_cps = szp_find_critical_points_3d(data, &omp_count, d1, d2, d3, absErrBound);
        printf("OpenMP found %zu (CUDA %zu — %s)\n\n",
               omp_count, orig_cp_count,
               omp_count == orig_cp_count ? "MATCH" : "DIFFER");
    }

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

    /* Library restoration on the GPU: ranked extrema, then saddle stencil
       and RBF passes (szp_cuda_topo_restore.cuh). */
    szp_topo_restore_stats rs;
    if (g_pl) {
        /* Stored codes as the decoder sees them: 2-bit types from the stream
           plus the packed saddle kinds. */
        int *codes = (int *)malloc(nbEle * sizeof(int));
        memcpy(codes, FN, nbEle * sizeof(int));
        size_t kinds_size = 0;
        unsigned char *kinds = szp_pl_pack_saddle_kinds(pl_codes, nbEle, &kinds_size);
        if (szp_pl_apply_saddle_kinds(codes, nbEle, kinds, kinds_size) != 0)
            printf("  ERROR: saddle kinds do not match the stored types\n");
        printf("  Saddle kinds: %zu bytes\n", kinds_size);
        if (szp_cuda_topo_restore_float_pl(decompressed, codes, sort_positions,
                                           sort_positions ? extrema_count : 0,
                                           3, d1, d2, d3, absErrBound, &rs) != 0)
            printf("  ERROR: GPU topology restoration failed\n");
        printf("  Extrema lost %zu -> %zu of %zu; saddles lost %zu -> %zu of %zu\n",
               rs.extrema_lost_before, rs.extrema_lost_after, rs.extrema,
               rs.saddles_lost_before, rs.saddles_lost_after, rs.saddles);
        printf("  PL fixes: saddles %zu, other %zu, %zu rounds; false positives %zu -> %zu, false types %zu -> %zu\n",
               rs.saddles_by_stencil, rs.other_fixes, rs.rounds, rs.fp_before, rs.fp_after,
               rs.ft_before, rs.ft_after);
        /* Compressor side of the guarantee: the same restoration on the CPU
           (must equal the GPU result), then the exact values it stores. */
        float *sim = (float *)malloc(nbEle * sizeof(float));
        memcpy(sim, orig_decomp, nbEle * sizeof(float));
        szp_topo_restore_stats cs;
        szp_topo_restore_float_pl(sim, codes, sort_positions, sort_positions ? extrema_count : 0,
                                  3, d1, d2, d3, absErrBound, &cs);
        int same = memcmp(sim, decompressed, nbEle * sizeof(float)) == 0;
        printf("  CPU and GPU restored fields: %s\n", same ? "IDENTICAL" : "DIFFER");
        size_t exc_size = 0, exc_count = 0;
        unsigned char *exc = szp_pl_exceptions_float(data, sim, codes, 3, d1, d2, d3, &exc_size, &exc_count);
        long long applied = szp_pl_apply_exceptions_float(decompressed, nbEle, exc, exc_size);
        printf("  PL guarantee: %zu points stored exactly (%zu bytes, %.2f%% of points), applied %lld; "
               "ratio with them %.2fx\n", exc_count, exc_size, 100.0 * exc_count / nbEle, applied,
               (double)(nbEle * sizeof(float)) / (topo_outSize + sort_outSize + kinds_size + exc_size));
        if (memcmp(sim, decompressed, nbEle * sizeof(float)) != 0)
            printf("  ERROR: GPU output with exceptions differs from the compressor's simulation\n");
        free(sim);
        free(exc);
        free(codes);
        free(kinds);
    } else {
        if (szp_cuda_topo_restore_float_3d(decompressed, FN, sort_positions,
                                           sort_positions ? extrema_count : 0,
                                           d1, d2, d3, absErrBound, &rs) != 0)
            printf("  ERROR: GPU topology restoration failed\n");
        printf("  Extrema lost %zu -> %zu of %zu; saddles lost %zu -> %zu of %zu\n",
               rs.extrema_lost_before, rs.extrema_lost_after, rs.extrema,
               rs.saddles_lost_before, rs.saddles_lost_after, rs.saddles);
        printf("  Saddle stencil restored %zu saddles; RBF restored %zu saddles\n",
               rs.saddles_by_stencil, rs.saddles_by_rbf);
    }

    double tp1 = get_time_ms();
    printf("  Post-processing done in %.2f ms\n\n", tp1 - tp0);

    /* Optional: write the post-processed field for external checks
       (example/topo_check), e.g. TOPOSZP_DUMP=out.f32 ./test_cuda_topology_3d ... */
    if (const char *dump_path = getenv("TOPOSZP_DUMP")) {
        FILE *df = fopen(dump_path, "wb");
        if (df) { fwrite(decompressed, sizeof(float), nbEle, df); fclose(df);
                  printf("  Wrote post-processed field to %s\n", dump_path); }
        else perror(dump_path);
    }

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
    size_t changed_to[4][4] = {{0}};
    for (size_t i = 0; i < orig_cp_count; i++) {
        size_t flat = IDX(orig_cps[i].x, orig_cps[i].y, orig_cps[i].z, d2, d3);
        if (decomp_type_map[flat] == orig_cps[i].type) preserved++;
        else if (decomp_type_map[flat] == 0) {
            lost++;
            if (orig_cps[i].type==1) l_max++;
            else if (orig_cps[i].type==2) l_min++;
            else l_sad++;
        } else {
            changed++;
            changed_to[orig_cps[i].type & 3][decomp_type_map[flat] & 3]++;
        }
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
    printf("Lost: %zu (max:%zu min:%zu sad:%zu)  Changed: %zu  New: %zu\n",
           lost, l_max, l_min, l_sad, changed, new_cps);
    if (changed) {
        static const char *tn[4] = {"reg", "max", "min", "sad"};
        printf("Changed types:");
        for (int a = 1; a < 4; a++)
            for (int b = 1; b < 4; b++)
                if (changed_to[a][b]) printf("  %s->%s: %zu", tn[a], tn[b], changed_to[a][b]);
        printf("\n");
    }
    printf("\n");

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
    /* ---- Persistence-based saddle analysis ---- */
    printf("--- Persistence-based saddle analysis ---\n");
    {
        size_t s = (size_t)d2 * d3;
        size_t saddle_count = 0;
        double *saddle_pers = (double *)malloc(orig_cp_count * sizeof(double));
        int *saddle_kept = (int *)calloc(orig_cp_count, sizeof(int));

        /* Use existing decomp_type_map (already built in Step 5) */
        int *dtype_map = decomp_type_map;

        for (size_t i = 0; i < orig_cp_count; i++) {
            if (orig_cps[i].type != 3) continue;
            int x = orig_cps[i].x, y = orig_cps[i].y, z = orig_cps[i].z;
            if (x<1||x>=d1-1||y<1||y>=d2-1||z<1||z>=d3-1) continue;

            size_t flat = (size_t)x*s + y*d3 + z;
            float c = data[flat];
            float nb[6] = {
                data[(x-1)*s+y*d3+z], data[(x+1)*s+y*d3+z],
                data[x*s+(y-1)*d3+z], data[x*s+(y+1)*d3+z],
                data[x*s+y*d3+(z-1)], data[x*s+y*d3+(z+1)]
            };
            double min_diff = 1e30;
            for (int n = 0; n < 6; n++) {
                double d = fabs((double)c - nb[n]);
                if (d < min_diff) min_diff = d;
            }

            saddle_pers[saddle_count] = min_diff;
            saddle_kept[saddle_count] = (dtype_map[flat] == 3) ? 1 : 0;
            saddle_count++;
        }

        double thresholds[] = {0.0, 0.5, 1.0, 2.0, 5.0, 10.0};
        int nt = 6;
        printf("\n  %-20s %10s %10s %10s %8s\n",
               "Persistence threshold", "Total", "Preserved", "Lost", "Rate");
        printf("  %-20s %10s %10s %10s %8s\n",
               "--------------------", "-----", "---------", "----", "----");
        for (int t = 0; t < nt; t++) {
            double thresh = thresholds[t] * (double)absErrBound;
            size_t above = 0, above_kept = 0;
            for (size_t j = 0; j < saddle_count; j++) {
                if (saddle_pers[j] > thresh) {
                    above++;
                    if (saddle_kept[j]) above_kept++;
                }
            }
            char label[64];
            if (t == 0) snprintf(label, sizeof(label), "All saddles");
            else snprintf(label, sizeof(label), "> %.1f×eb", thresholds[t]);
            printf("  %-20s %10zu %10zu %10zu %7.2f%%\n",
                   label, above, above_kept, above - above_kept,
                   above > 0 ? 100.0*above_kept/above : 0.0);
        }

        double p_min=1e30, p_max=0, p_sum=0;
        for (size_t j = 0; j < saddle_count; j++) {
            if (saddle_pers[j] < p_min) p_min = saddle_pers[j];
            if (saddle_pers[j] > p_max) p_max = saddle_pers[j];
            p_sum += saddle_pers[j];
        }
        printf("\n  Saddle persistence: min=%e  avg=%e  max=%e\n",
               p_min, saddle_count>0 ? p_sum/saddle_count : 0.0, p_max);
        printf("  Error bound: %e\n\n", (double)absErrBound);

        free(saddle_pers); free(saddle_kept);
        /* dtype_map is alias for decomp_type_map — freed later */
    }

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

    /* PL mode: strict comparison of the PL codes (saddle kinds included). */
    size_t pl_fp = 0, pl_ft = 0, pl_sad = 0, pl_sad_kept = 0;
    if (g_pl) {
        signed char *after = (signed char *)malloc(nbEle);
        szp_pl_classify_float(decompressed, 3, d1, d2, d3, after);
        for (size_t i = 0; i < nbEle; i++) {
            int o = pl_codes[i], a = after[i];
            if (o >= 3) { pl_sad++; pl_sad_kept += a == o; }
            if (a != o && a != 0) { if (o == 0) pl_fp++; else pl_ft++; }
        }
        free(after);
        printf("PL check: saddles kept %zu / %zu (%.2f%%, kinds compared), false positives %zu, false types %zu\n",
               pl_sad_kept, pl_sad, pl_sad ? 100.0 * pl_sad_kept / pl_sad : 100.0, pl_fp, pl_ft);
        free(pl_codes);
    }

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

    /* Exit code: 0 only if every guarantee holds. Lost saddles are reported
       above but are not a failure (the method does not guarantee them). In
       PL mode FP/FT are checked with the PL codes (saddle kinds included). */
    int fail_eb = e2 != 0, fail_fp = g_pl ? pl_fp != 0 : new_cps != 0, fail_ft = g_pl ? pl_ft != 0 : changed != 0;
    int fail_ext = (l_max + l_min) != 0, fail_fnarr = fn_match != nbEle;
    int overall = (fail_eb || fail_fp || fail_ft || fail_ext || fail_fnarr) ? 1 : 0;
    printf("Result: %s%s%s%s%s%s\n", overall ? "FAIL:" : g_pl ? "PASS (PL mode: error <= 2 eb, no FP, no FT, all extrema kept)"
                                                       : "PASS (error <= 2 eb, no FP, no FT, all extrema kept)",
           fail_eb ? " error>2eb" : "", fail_fp ? " false-positives" : "", fail_ft ? " false-types" : "",
           fail_ext ? " extrema-lost" : "", fail_fnarr ? " type-array-mismatch" : "");
    return overall;
}
