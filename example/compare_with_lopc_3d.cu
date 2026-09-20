/**
 *  @file compare_with_lopc_3d.cu
 *  @brief Compare TopoSZp CUDA vs LOPC on the same 3D dataset.
 *
 *  Usage: compare_with_lopc_3d <orig_file> <lopc_decomp_file> <d1> <d2> <d3> <eb> <blockSize> <lopc_comp_size>
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
#include "szp_cuda_topology_3d.cuh"
#include "szp_topology_3d.h"

static double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static int classify_3d(const float *data, int d1, int d2, int d3, int x, int y, int z) {
    size_t s = (size_t)d2 * d3;
    float c = data[x*s + y*d3 + z];
    float xm = data[(x-1)*s + y*d3 + z], xp = data[(x+1)*s + y*d3 + z];
    float ym = data[x*s + (y-1)*d3 + z], yp = data[x*s + (y+1)*d3 + z];
    float zm = data[x*s + y*d3 + (z-1)], zp = data[x*s + y*d3 + (z+1)];
    if (c>xm && c>xp && c>ym && c>yp && c>zm && c>zp) return 1;
    if (c<xm && c<xp && c<ym && c<yp && c<zm && c<zp) return 2;
    int xh=(c>xm)&&(c>xp), xl=(c<xm)&&(c<xp);
    int yh=(c>ym)&&(c>yp), yl=(c<ym)&&(c<yp);
    int zh=(c>zm)&&(c>zp), zl=(c<zm)&&(c<zp);
    if ((xh+yh+zh)>=1 && (xl+yl+zl)>=1) return 3;
    return 0;
}

struct CPStats3D {
    size_t total, maxima, minima, saddles;
    size_t preserved, lost_max, lost_min, lost_sad, type_changed, new_cps;
    double max_err;
    size_t err_1eb, err_2eb;
    size_t lo_pairs, lo_violations;
    size_t sad_1eb, sad_1eb_kept, sad_2eb, sad_2eb_kept;
};

static CPStats3D analyze_3d(const float *orig, const float *decomp,
                             int d1, int d2, int d3, float eb) {
    CPStats3D s = {};
    size_t nbEle = (size_t)d1 * d2 * d3;
    size_t stride = (size_t)d2 * d3;

    /* Find original CPs */
    int *otype = (int *)calloc(nbEle, sizeof(int));
    for (int x = 1; x < d1-1; x++)
        for (int y = 1; y < d2-1; y++)
            for (int z = 1; z < d3-1; z++) {
                int t = classify_3d(orig, d1, d2, d3, x, y, z);
                if (t) { otype[x*stride+y*d3+z] = t; s.total++;
                    if (t==1) s.maxima++; else if (t==2) s.minima++; else s.saddles++; }
            }

    /* Find decomp CPs and check preservation */
    int *dtype = (int *)calloc(nbEle, sizeof(int));
    for (int x = 1; x < d1-1; x++)
        for (int y = 1; y < d2-1; y++)
            for (int z = 1; z < d3-1; z++) {
                int t = classify_3d(decomp, d1, d2, d3, x, y, z);
                if (t) dtype[x*stride+y*d3+z] = t;
            }

    for (int x = 1; x < d1-1; x++)
        for (int y = 1; y < d2-1; y++)
            for (int z = 1; z < d3-1; z++) {
                size_t flat = x*stride + y*d3 + z;
                int ot = otype[flat], dt = dtype[flat];
                if (!ot) continue;
                if (dt == ot) s.preserved++;
                else if (dt == 0) {
                    if (ot==1) s.lost_max++; else if (ot==2) s.lost_min++; else s.lost_sad++;
                } else s.type_changed++;
            }

    /* New CPs */
    for (int x = 1; x < d1-1; x++)
        for (int y = 1; y < d2-1; y++)
            for (int z = 1; z < d3-1; z++) {
                size_t flat = x*stride + y*d3 + z;
                if (dtype[flat] && !otype[flat]) s.new_cps++;
            }

    /* Error bound */
    for (size_t i = 0; i < nbEle; i++) {
        double e = fabs((double)orig[i] - (double)decomp[i]);
        if (e > s.max_err) s.max_err = e;
        if (e > eb * 1.01) s.err_1eb++;
        if (e > eb * 2.0) s.err_2eb++;
    }

    /* Local order — 6-connected, avoid double counting with +x, +y, +z neighbors */
    int dx[] = {1, 0, 0};
    int dy[] = {0, 1, 0};
    int dz[] = {0, 0, 1};
    for (int x = 1; x < d1-1; x++)
        for (int y = 1; y < d2-1; y++)
            for (int z = 1; z < d3-1; z++)
                for (int d = 0; d < 3; d++) {
                    int nx = x+dx[d], ny = y+dy[d], nz = z+dz[d];
                    if (nx >= d1-1 || ny >= d2-1 || nz >= d3-1) continue;
                    float oa = orig[x*stride+y*d3+z], ob = orig[nx*stride+ny*d3+nz];
                    if (oa == ob) continue;
                    float da = decomp[x*stride+y*d3+z], db = decomp[nx*stride+ny*d3+nz];
                    int oo = (oa < ob) ? -1 : 1;
                    int dd = (da < db) ? -1 : ((da > db) ? 1 : 0);
                    s.lo_pairs++;
                    if (oo != dd) s.lo_violations++;
                }

    /* Persistence-based saddle analysis */
    for (int x = 1; x < d1-1; x++)
        for (int y = 1; y < d2-1; y++)
            for (int z = 1; z < d3-1; z++) {
                size_t flat = x*stride + y*d3 + z;
                if (otype[flat] != 3) continue;
                float c = orig[flat];
                float nb[6] = {
                    orig[(x-1)*stride+y*d3+z], orig[(x+1)*stride+y*d3+z],
                    orig[x*stride+(y-1)*d3+z], orig[x*stride+(y+1)*d3+z],
                    orig[x*stride+y*d3+(z-1)], orig[x*stride+y*d3+(z+1)]
                };
                double md = 1e30;
                for (int n = 0; n < 6; n++) {
                    double d2 = fabs((double)c - nb[n]);
                    if (d2 < md) md = d2;
                }
                int kept = (dtype[flat] == 3) ? 1 : 0;
                if (md > eb) { s.sad_1eb++; if (kept) s.sad_1eb_kept++; }
                if (md > 2.0*eb) { s.sad_2eb++; if (kept) s.sad_2eb_kept++; }
            }

    free(otype); free(dtype);
    return s;
}

int main(int argc, char *argv[]) {
    if (argc < 9) {
        fprintf(stderr, "Usage: %s <orig> <lopc_decomp> <d1> <d2> <d3> <eb> <blockSize> <lopc_comp_size>\n", argv[0]);
        return 1;
    }

    const char *orig_file = argv[1];
    const char *lopc_file = argv[2];
    int d1 = atoi(argv[3]), d2 = atoi(argv[4]), d3 = atoi(argv[5]);
    float eb = (float)atof(argv[6]);
    int blockSize = atoi(argv[7]);
    size_t lopc_comp_size = (size_t)atol(argv[8]);
    size_t nbEle = (size_t)d1 * d2 * d3;

    printf("============================================================\n");
    printf("  TopoSZp CUDA vs LOPC — 3D Side-by-Side Comparison\n");
    printf("============================================================\n");
    printf("Data: %s  (%d × %d × %d = %zu)\n", orig_file, d1, d2, d3, nbEle);
    printf("Error bound: %e    Block size: %d\n\n", eb, blockSize);

    /* Read data */
    float *orig = (float *)malloc(nbEle * sizeof(float));
    float *lopc_decomp = (float *)malloc(nbEle * sizeof(float));
    FILE *fp;

    fp = fopen(orig_file, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", orig_file); return 1; }
    fread(orig, sizeof(float), nbEle, fp); fclose(fp);

    fp = fopen(lopc_file, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", lopc_file); return 1; }
    fread(lopc_decomp, sizeof(float), nbEle, fp); fclose(fp);

    /* CUDA warmup */
    { void *d; cudaMalloc(&d, 1); cudaFree(d); cudaDeviceSynchronize(); }

    /* ---- TopoSZp CUDA pipeline ---- */
    printf("Running TopoSZp CUDA pipeline...\n");

    size_t cp_count = 0;
    CriticalPoint3D *cps = szp_cuda_find_critical_points_3d(orig, &cp_count, d1, d2, d3, eb);
    printf("  Found %zu CPs\n", cp_count);

    double ts0 = get_time_ms();
    szp_cuda_sort_critical_points_3d(cps, cp_count, orig, d2, d3);
    double ts1 = get_time_ms();
    printf("  Sorted in %.2f ms\n", ts1 - ts0);

    size_t topo_size = 0;
    double tc0 = get_time_ms();
    unsigned char *topo_comp = szp_cuda_float_compress_topology_3d(
        orig, &topo_size, eb, nbEle, blockSize, cps, (int)cp_count, d1, d2, d3);
    double tc1 = get_time_ms();
    printf("  Compressed in %.2f ms (ratio: %.2fx)\n", tc1-tc0,
           (double)(nbEle*sizeof(float))/topo_size);

    float *topo_decomp = NULL;
    int *FN = NULL;
    double td0 = get_time_ms();
    szp_cuda_float_decompress_randomaccess_topology_preserved(
        &topo_decomp, nbEle, eb, blockSize, topo_comp, &FN);
    double td1 = get_time_ms();
    printf("  Decompressed in %.2f ms\n", td1-td0);

    /* Post-processing: extrema enforcement */
    float *topo_orig = (float *)malloc(nbEle * sizeof(float));
    memcpy(topo_orig, topo_decomp, nbEle * sizeof(float));
    float eps = fmaxf(1e-6f, 0.1f * eb);
    float eps_soft = 0.25f * eps;
    size_t stride = (size_t)d2 * d3;

    for (int x = 1; x < d1-1; x++)
        for (int y = 1; y < d2-1; y++)
            for (int z = 1; z < d3-1; z++) {
                size_t flat = x*stride + y*d3 + z;
                if (FN[flat] != 1 && FN[flat] != 2) continue;
                float ov = topo_orig[flat];
                float ma = ov + eb*0.999f, mi = ov - eb*0.999f;
                float xm=topo_decomp[(x-1)*stride+y*d3+z], xp=topo_decomp[(x+1)*stride+y*d3+z];
                float ym=topo_decomp[x*stride+(y-1)*d3+z], yp=topo_decomp[x*stride+(y+1)*d3+z];
                float zm=topo_decomp[x*stride+y*d3+(z-1)], zp=topo_decomp[x*stride+y*d3+(z+1)];
                if (FN[flat]==1) {
                    float m = fmaxf(fmaxf(fmaxf(xm,xp),fmaxf(ym,yp)),fmaxf(zm,zp));
                    float t = m + eps_soft;
                    if (t>ma) t=ma; if (t<mi) t=mi;
                    if (topo_decomp[flat]<t) topo_decomp[flat]=t;
                } else {
                    float m = fminf(fminf(fminf(xm,xp),fminf(ym,yp)),fminf(zm,zp));
                    float t = m - eps_soft;
                    if (t<mi) t=mi; if (t>ma) t=ma;
                    if (topo_decomp[flat]>t) topo_decomp[flat]=t;
                }
            }

    printf("  Post-processing done\n\n");

    /* ---- Analyze both ---- */
    printf("Analyzing results (this takes a moment for 168M elements)...\n\n");
    CPStats3D ts = analyze_3d(orig, topo_decomp, d1, d2, d3, eb);
    CPStats3D ls = analyze_3d(orig, lopc_decomp, d1, d2, d3, eb);

    double topo_ratio = (double)(nbEle * sizeof(float)) / topo_size;
    double lopc_ratio = (double)(nbEle * sizeof(float)) / lopc_comp_size;

    /* ---- Side-by-side ---- */
    printf("==================== 3D SIDE-BY-SIDE COMPARISON ====================\n\n");
    printf("  %-30s %14s %14s\n", "Metric", "TopoSZp CUDA", "LOPC GPU");
    printf("  %-30s %14s %14s\n", "------------------------------", "--------------", "--------------");
    printf("  %-30s %13.2fx %13.2fx\n", "Compression ratio", topo_ratio, lopc_ratio);
    printf("  %-30s %11.2f ms %11.2f ms\n", "Compression time", tc1-tc0, 95901.0);
    printf("  %-30s %11.2f ms %11.2f ms\n", "Decompression time", td1-td0, 22.39);
    printf("  %-30s %14zu %14zu\n", "Original CPs", ts.total, ls.total);
    printf("  %-30s %10zu (%5.2f%%) %10zu (%5.2f%%)\n", "Preserved",
           ts.preserved, 100.0*ts.preserved/ts.total,
           ls.preserved, 100.0*ls.preserved/ls.total);
    printf("  %-30s %10zu/%-6zu %10zu/%-6zu\n", "  Maxima",
           ts.maxima-ts.lost_max, ts.maxima, ls.maxima-ls.lost_max, ls.maxima);
    printf("  %-30s %10zu/%-6zu %10zu/%-6zu\n", "  Minima",
           ts.minima-ts.lost_min, ts.minima, ls.minima-ls.lost_min, ls.minima);
    printf("  %-30s %10zu/%-6zu %10zu/%-6zu\n", "  Saddles",
           ts.saddles-ts.lost_sad, ts.saddles, ls.saddles-ls.lost_sad, ls.saddles);
    printf("  %-30s %14zu %14zu\n", "  False positives",
           ts.new_cps, ls.new_cps);
    printf("  %-30s %14e %14e\n", "Max error", ts.max_err, ls.max_err);
    printf("  %-30s %14s %14s\n", "Within 2×eb",
           ts.err_2eb==0 ? "YES" : "NO", ls.err_2eb==0 ? "YES" : "NO");
    size_t tlo = ts.lo_pairs - ts.lo_violations;
    size_t llo = ls.lo_pairs - ls.lo_violations;
    printf("  %-30s %13.4f%% %13.4f%%\n", "Local order preserved",
           100.0*tlo/ts.lo_pairs, 100.0*llo/ls.lo_pairs);
    printf("  %-30s %10zu/%-6zu %10zu/%-6zu\n", "Saddles>1×eb preserved",
           ts.sad_1eb_kept, ts.sad_1eb, ls.sad_1eb_kept, ls.sad_1eb);
    printf("  %-30s %10zu/%-6zu %10zu/%-6zu\n", "Saddles>2×eb preserved",
           ts.sad_2eb_kept, ts.sad_2eb, ls.sad_2eb_kept, ls.sad_2eb);
    printf("\n=================================================================\n");

    /* Cleanup */
    free(orig); free(lopc_decomp); free(topo_decomp); free(topo_orig);
    free(cps); free(topo_comp); free(FN);
    return 0;
}
