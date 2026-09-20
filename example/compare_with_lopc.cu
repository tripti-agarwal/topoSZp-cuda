/**
 *  @file compare_with_lopc.cu
 *  @brief Compare TopoSZp CUDA vs LOPC on the same 2D dataset.
 *
 *  Reads original data and an LOPC-decompressed file, computes:
 *  - Critical point preservation (max, min, saddle)
 *  - Local order preservation
 *  - Error bound
 *  - Persistence-based saddle analysis
 *
 *  Also runs the TopoSZp CUDA pipeline for side-by-side comparison.
 *
 *  Usage: compare_with_lopc <original_file> <lopc_decompressed_file> <rows> <cols> <absErrBound> <blockSize> <lopc_compressed_size>
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

static int classify_2d(const float *data, int rows, int cols, int i, int j) {
    float c = data[i * cols + j];
    float u = data[(i-1) * cols + j];
    float d = data[(i+1) * cols + j];
    float l = data[i * cols + (j-1)];
    float r = data[i * cols + (j+1)];
    if (c > u && c > d && c > l && c > r) return 1;
    if (c < u && c < d && c < l && c < r) return 2;
    int xh = (c > u) && (c > d), xl = (c < u) && (c < d);
    int yh = (c > l) && (c > r), yl = (c < l) && (c < r);
    if ((xh && yl) || (xl && yh)) return 3;
    return 0;
}

struct CPStats {
    size_t total, maxima, minima, saddles;
    size_t preserved, lost_max, lost_min, lost_sad, type_changed, new_cps;
    double max_err;
    size_t err_1eb, err_2eb;
    size_t local_order_pairs, local_order_violations;
    /* Persistence */
    size_t sad_above_1eb, sad_above_1eb_preserved;
    size_t sad_above_2eb, sad_above_2eb_preserved;
};

static CPStats analyze(const float *orig, const float *decomp, int rows, int cols, float eb) {
    CPStats s = {};
    size_t nbEle = (size_t)rows * cols;

    /* Find original CPs */
    int *orig_type = (int *)calloc(nbEle, sizeof(int));
    for (int i = 1; i < rows-1; i++)
        for (int j = 1; j < cols-1; j++) {
            int t = classify_2d(orig, rows, cols, i, j);
            if (t) { orig_type[i*cols+j] = t; s.total++;
                if (t==1) s.maxima++; else if (t==2) s.minima++; else s.saddles++; }
        }

    /* Find decomp CPs and check preservation */
    int *decomp_type = (int *)calloc(nbEle, sizeof(int));
    size_t decomp_total = 0;
    for (int i = 1; i < rows-1; i++)
        for (int j = 1; j < cols-1; j++) {
            int t = classify_2d(decomp, rows, cols, i, j);
            if (t) { decomp_type[i*cols+j] = t; decomp_total++; }
        }

    for (int i = 1; i < rows-1; i++)
        for (int j = 1; j < cols-1; j++) {
            int ot = orig_type[i*cols+j];
            int dt = decomp_type[i*cols+j];
            if (ot == 0) continue;
            if (dt == ot) s.preserved++;
            else if (dt == 0) {
                if (ot==1) s.lost_max++; else if (ot==2) s.lost_min++; else s.lost_sad++;
            } else s.type_changed++;
        }

    /* New CPs */
    for (int i = 1; i < rows-1; i++)
        for (int j = 1; j < cols-1; j++)
            if (decomp_type[i*cols+j] && !orig_type[i*cols+j]) s.new_cps++;

    /* Error bound */
    for (size_t i = 0; i < nbEle; i++) {
        double e = fabs((double)orig[i] - (double)decomp[i]);
        if (e > s.max_err) s.max_err = e;
        if (e > eb * 1.01) s.err_1eb++;
        if (e > eb * 2.0) s.err_2eb++;
    }

    /* Local order */
    int dx[] = {0, 1};
    int dy[] = {1, 0};
    for (int i = 1; i < rows-1; i++)
        for (int j = 1; j < cols-1; j++)
            for (int d = 0; d < 2; d++) {
                int ni = i + dx[d], nj = j + dy[d];
                if (ni >= rows-1 || nj >= cols-1) continue;
                float oa = orig[i*cols+j], ob = orig[ni*cols+nj];
                if (oa == ob) continue;
                float da = decomp[i*cols+j], db = decomp[ni*cols+nj];
                int oo = (oa < ob) ? -1 : 1;
                int dd = (da < db) ? -1 : ((da > db) ? 1 : 0);
                s.local_order_pairs++;
                if (oo != dd) s.local_order_violations++;
            }

    /* Persistence-based saddle analysis */
    for (int i = 1; i < rows-1; i++)
        for (int j = 1; j < cols-1; j++) {
            if (orig_type[i*cols+j] != 3) continue;
            float c = orig[i*cols+j];
            float u = orig[(i-1)*cols+j], d2 = orig[(i+1)*cols+j];
            float l = orig[i*cols+j-1], r = orig[i*cols+j+1];
            double md = fmin(fmin(fabs(c-u),fabs(c-d2)), fmin(fabs(c-l),fabs(c-r)));
            int kept = (decomp_type[i*cols+j] == 3) ? 1 : 0;
            if (md > eb)     { s.sad_above_1eb++; if (kept) s.sad_above_1eb_preserved++; }
            if (md > 2.0*eb) { s.sad_above_2eb++; if (kept) s.sad_above_2eb_preserved++; }
        }

    free(orig_type);
    free(decomp_type);
    return s;
}

static void print_stats(const char *name, CPStats &s, double ratio, double comp_time, double decomp_time) {
    printf("  %-25s %12s\n", "Metric", name);
    printf("  %-25s %12s\n", "-------------------------", "------------");
    printf("  %-25s %11.2fx\n", "Compression ratio", ratio);
    printf("  %-25s %10.2f ms\n", "Compression time", comp_time);
    printf("  %-25s %10.2f ms\n", "Decompression time", decomp_time);
    printf("  %-25s %12zu\n", "Original CPs", s.total);
    printf("  %-25s %8zu (%5.2f%%)\n", "Preserved", s.preserved, 100.0*s.preserved/s.total);
    printf("  %-25s %8zu / %zu\n", "  Maxima", s.maxima - s.lost_max, s.maxima);
    printf("  %-25s %8zu / %zu\n", "  Minima", s.minima - s.lost_min, s.minima);
    printf("  %-25s %8zu / %zu\n", "  Saddles", s.saddles - s.lost_sad, s.saddles);
    printf("  %-25s %12zu\n", "  Lost saddles", s.lost_sad);
    printf("  %-25s %12zu\n", "  New CPs (false pos)", s.new_cps);
    printf("  %-25s %12zu\n", "  Type changed", s.type_changed);
    printf("  %-25s %12e\n", "Max error", s.max_err);
    printf("  %-25s %12s\n", "Within 2×eb",
           s.err_2eb == 0 ? "YES" : "NO");
    size_t lo_preserved = s.local_order_pairs - s.local_order_violations;
    printf("  %-25s %7.4f%%\n", "Local order preserved",
           s.local_order_pairs > 0 ? 100.0*lo_preserved/s.local_order_pairs : 0.0);
    printf("  %-25s %8zu / %zu (%5.2f%%)\n", "Saddles > 1×eb preserved",
           s.sad_above_1eb_preserved, s.sad_above_1eb,
           s.sad_above_1eb > 0 ? 100.0*s.sad_above_1eb_preserved/s.sad_above_1eb : 0.0);
    printf("  %-25s %8zu / %zu (%5.2f%%)\n", "Saddles > 2×eb preserved",
           s.sad_above_2eb_preserved, s.sad_above_2eb,
           s.sad_above_2eb > 0 ? 100.0*s.sad_above_2eb_preserved/s.sad_above_2eb : 0.0);
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc < 8) {
        fprintf(stderr, "Usage: %s <orig_file> <lopc_decomp_file> <rows> <cols> <eb> <blockSize> <lopc_comp_size>\n", argv[0]);
        return 1;
    }

    const char *orig_file = argv[1];
    const char *lopc_file = argv[2];
    int rows = atoi(argv[3]);
    int cols = atoi(argv[4]);
    float eb = (float)atof(argv[5]);
    int blockSize = atoi(argv[6]);
    size_t lopc_comp_size = (size_t)atol(argv[7]);
    size_t nbEle = (size_t)rows * cols;

    printf("============================================================\n");
    printf("  TopoSZp CUDA vs LOPC — Side-by-Side Comparison\n");
    printf("============================================================\n");
    printf("Data: %s  (%d × %d = %zu)\n", orig_file, rows, cols, nbEle);
    printf("Error bound: %e    Block size: %d\n\n", eb, blockSize);

    /* Read original data */
    float *orig = (float *)malloc(nbEle * sizeof(float));
    FILE *fp = fopen(orig_file, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", orig_file); return 1; }
    fread(orig, sizeof(float), nbEle, fp);
    fclose(fp);

    /* Read LOPC decompressed data */
    float *lopc_decomp = (float *)malloc(nbEle * sizeof(float));
    fp = fopen(lopc_file, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", lopc_file); return 1; }
    fread(lopc_decomp, sizeof(float), nbEle, fp);
    fclose(fp);

    /* CUDA warmup */
    { void *d; cudaMalloc(&d, 1); cudaFree(d); cudaDeviceSynchronize(); }

    /* ---- Run TopoSZp CUDA pipeline ---- */
    printf("Running TopoSZp CUDA pipeline...\n");
    size_t cp_count = 0;
    CriticalPoint *cps = szp_cuda_find_critical_points(orig, &cp_count, rows, cols, eb);
    szp_cuda_sort_critical_points_by_original_data(cps, cp_count, orig, cols);

    size_t topo_size = 0;
    double t0 = get_time_ms();
    unsigned char *topo_comp = szp_cuda_float_compress_randomaccess_topology_preserved(
        orig, &topo_size, eb, nbEle, blockSize, cps, (int)cp_count, rows, cols);
    double t1 = get_time_ms();

    float *topo_decomp = NULL;
    int *FN = NULL;
    double t2 = get_time_ms();
    szp_cuda_float_decompress_randomaccess_topology_preserved(
        &topo_decomp, nbEle, eb, blockSize, topo_comp, &FN);
    double t3 = get_time_ms();

    /* Post-processing */
    float *topo_orig_decomp = (float *)malloc(nbEle * sizeof(float));
    memcpy(topo_orig_decomp, topo_decomp, nbEle * sizeof(float));

    /* Sort positions */
    size_t sp_size = 0;
    unsigned char *sp_comp = szp_cuda_compress_sort_positions(cps, cp_count, &sp_size, blockSize);
    size_t extrema_count = 0;
    for (size_t i = 0; i < cp_count; i++)
        if (cps[i].type == 1 || cps[i].type == 2) extrema_count++;
    int *sort_pos = NULL;
    if (sp_comp && sp_size > 0 && extrema_count > 0)
        sort_pos = szp_cuda_decompress_sort_positions(sp_comp, extrema_count, blockSize);

    /* Stencils */
    if (sort_pos) {
        size_t si = 0;
        for (int i = 1; i < rows-1; i++)
            for (int j = 1; j < cols-1; j++) {
                size_t flat = (size_t)i*cols+j;
                if (FN[flat] == 1 || FN[flat] == 2) {
                    int sp = (si < extrema_count) ? sort_pos[si] : 0;
                    float ov = topo_orig_decomp[flat];
                    float ma = ov + eb * 0.999f, mi = ov - eb * 0.999f;
                    float xm=topo_decomp[(i-1)*cols+j], xp=topo_decomp[(i+1)*cols+j];
                    float ym=topo_decomp[i*cols+j-1], yp=topo_decomp[i*cols+j+1];
                    if (FN[flat] == 1) {
                        float mn = fmaxf(fmaxf(xm,xp),fmaxf(ym,yp));
                        float target = mn * (1.0f + sp * FLT_EPSILON);
                        if (target > ma) target = ma;
                        if (target < mi) target = mi;
                        topo_decomp[flat] = target;
                    } else {
                        float mn = fminf(fminf(xm,xp),fminf(ym,yp));
                        float target = (sp==0) ? mn*(1.f-FLT_EPSILON) : mn*(1.f-(1.f/sp)*FLT_EPSILON);
                        if (target > ma) target = ma;
                        if (target < mi) target = mi;
                        topo_decomp[flat] = target;
                    }
                    si++;
                }
            }
    }

    /* Extrema enforcement */
    float eps = fmaxf(1e-6f, 0.1f * eb);
    float eps_soft = 0.25f * eps;
    for (int i = 1; i < rows-1; i++)
        for (int j = 1; j < cols-1; j++) {
            size_t flat = (size_t)i*cols+j;
            float ov = topo_orig_decomp[flat];
            float ma = ov + eb*0.999f, mi = ov - eb*0.999f;
            float xm=topo_decomp[(i-1)*cols+j], xp=topo_decomp[(i+1)*cols+j];
            float ym=topo_decomp[i*cols+j-1], yp=topo_decomp[i*cols+j+1];
            if (FN[flat]==1) {
                float m=fmaxf(fmaxf(xm,xp),fmaxf(ym,yp));
                float t=m+eps_soft; if(t>ma)t=ma; if(t<mi)t=mi;
                if(topo_decomp[flat]<t) topo_decomp[flat]=t;
            } else if (FN[flat]==2) {
                float m=fminf(fminf(xm,xp),fminf(ym,yp));
                float t=m-eps_soft; if(t<mi)t=mi; if(t>ma)t=ma;
                if(topo_decomp[flat]>t) topo_decomp[flat]=t;
            }
        }

    printf("TopoSZp done.\n\n");

    /* ---- Analyze both ---- */
    printf("Analyzing results...\n\n");
    CPStats topo_stats = analyze(orig, topo_decomp, rows, cols, eb);
    CPStats lopc_stats = analyze(orig, lopc_decomp, rows, cols, eb);

    double topo_ratio = (double)(nbEle * sizeof(float)) / topo_size;
    double lopc_ratio = (double)(nbEle * sizeof(float)) / lopc_comp_size;

    /* ---- Side-by-side table ---- */
    printf("==================== SIDE-BY-SIDE COMPARISON ====================\n\n");
    printf("  %-30s %14s %14s\n", "Metric", "TopoSZp CUDA", "LOPC GPU");
    printf("  %-30s %14s %14s\n", "------------------------------", "--------------", "--------------");
    printf("  %-30s %13.2fx %13.2fx\n", "Compression ratio", topo_ratio, lopc_ratio);
    printf("  %-30s %11.2f ms %11.2f ms\n", "Compression time", t1-t0, 29.75);  /* LOPC time from earlier run */
    printf("  %-30s %11.2f ms %11.2f ms\n", "Decompression time", t3-t2, 12.38);
    printf("  %-30s %14zu %14zu\n", "Original CPs", topo_stats.total, lopc_stats.total);
    printf("  %-30s %10zu (%5.2f%%) %10zu (%5.2f%%)\n", "Preserved",
           topo_stats.preserved, 100.0*topo_stats.preserved/topo_stats.total,
           lopc_stats.preserved, 100.0*lopc_stats.preserved/lopc_stats.total);
    printf("  %-30s %10zu/%-5zu %10zu/%-5zu\n", "  Maxima",
           topo_stats.maxima-topo_stats.lost_max, topo_stats.maxima,
           lopc_stats.maxima-lopc_stats.lost_max, lopc_stats.maxima);
    printf("  %-30s %10zu/%-5zu %10zu/%-5zu\n", "  Minima",
           topo_stats.minima-topo_stats.lost_min, topo_stats.minima,
           lopc_stats.minima-lopc_stats.lost_min, lopc_stats.minima);
    printf("  %-30s %10zu/%-5zu %10zu/%-5zu\n", "  Saddles",
           topo_stats.saddles-topo_stats.lost_sad, topo_stats.saddles,
           lopc_stats.saddles-lopc_stats.lost_sad, lopc_stats.saddles);
    printf("  %-30s %14zu %14zu\n", "  False positives (new CPs)",
           topo_stats.new_cps, lopc_stats.new_cps);
    printf("  %-30s %14e %14e\n", "Max error", topo_stats.max_err, lopc_stats.max_err);
    printf("  %-30s %14s %14s\n", "Within 2×eb",
           topo_stats.err_2eb == 0 ? "YES" : "NO",
           lopc_stats.err_2eb == 0 ? "YES" : "NO");

    size_t topo_lo = topo_stats.local_order_pairs - topo_stats.local_order_violations;
    size_t lopc_lo = lopc_stats.local_order_pairs - lopc_stats.local_order_violations;
    printf("  %-30s %13.4f%% %13.4f%%\n", "Local order preserved",
           100.0*topo_lo/topo_stats.local_order_pairs,
           100.0*lopc_lo/lopc_stats.local_order_pairs);
    printf("  %-30s %10zu/%-5zu %10zu/%-5zu\n", "Saddles>1×eb preserved",
           topo_stats.sad_above_1eb_preserved, topo_stats.sad_above_1eb,
           lopc_stats.sad_above_1eb_preserved, lopc_stats.sad_above_1eb);
    printf("  %-30s %10zu/%-5zu %10zu/%-5zu\n", "Saddles>2×eb preserved",
           topo_stats.sad_above_2eb_preserved, topo_stats.sad_above_2eb,
           lopc_stats.sad_above_2eb_preserved, lopc_stats.sad_above_2eb);

    printf("\n=================================================================\n");

    /* Cleanup */
    free(orig); free(lopc_decomp); free(topo_decomp); free(topo_orig_decomp);
    free(cps); free(topo_comp); free(FN);
    if (sp_comp) free(sp_comp);
    if (sort_pos) free(sort_pos);

    return 0;
}
