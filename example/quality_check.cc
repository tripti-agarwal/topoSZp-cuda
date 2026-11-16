/**
 *  @file quality_check.cc
 *  @brief PSNR and tie-broken critical points of a reconstruction (any compressor).
 *
 *  quality_check <original> <decompressed> <d1> <d2> [d3]      (float32, slowest dimension first)
 *
 *  Prints one line:
 *    QUALITY psnr=<dB> nrmse=<..> maxerr=<..> range=<..> tb_orig=<n> tb_kept=<n> tb_FN=<n> tb_FP=<n> tb_FT=<n>
 *
 *  PSNR = 20 log10(range / RMSE), range of the original field.
 *
 *  tb_*: critical points by the axis rule of TopoSZp+ (face neighbours,
 *  interior points), but with ties broken by the grid index, as in simulation
 *  of simplicity: of two equal values, the one with the larger index counts as
 *  larger. Under this convention no two neighbours are equal, so a compressor
 *  that maps strictly ordered neighbours to equal values in index order (as
 *  LOPC does) keeps its critical points, while under strict comparisons
 *  (topo_check) a tied extremum or saddle is lost. FN / FP / FT as in
 *  topo_check: critical -> regular, regular -> critical, critical -> other type.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<float> load(const char *path, size_t n)
{
    std::vector<float> v(n);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    size_t got = fread(v.data(), sizeof(float), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "%s: read %zu of %zu values\n", path, got, n); exit(1); }
    return v;
}

/* 0 regular, 1 maximum, 2 minimum, 3 saddle; ties broken by index */
static inline int classify_tb(const float *a, long long k, const long long *stride, int nd)
{
    int hi_axes = 0, lo_axes = 0, all_hi = 1, all_lo = 1;
    for (int ax = 0; ax < nd; ax++) {
        int hi = 0, lo = 0;
        for (int s = -1; s <= 1; s += 2) {
            const long long n = k + s * stride[ax];
            const bool greater = a[k] > a[n] || (a[k] == a[n] && k > n);
            if (greater) hi++; else lo++;
        }
        if (hi == 2) hi_axes++; else all_hi = 0;
        if (lo == 2) lo_axes++; else all_lo = 0;
    }
    if (all_hi) return 1;
    if (all_lo) return 2;
    return (hi_axes >= 1 && lo_axes >= 1) ? 3 : 0;
}

int main(int argc, char **argv)
{
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "usage: %s <original> <decompressed> <d1> <d2> [d3]\n", argv[0]);
        return 1;
    }
    const long long d1 = atoll(argv[3]), d2 = atoll(argv[4]), d3 = argc == 6 ? atoll(argv[5]) : 1;
    const int nd = argc == 6 ? 3 : 2;
    if (d1 < 3 || d2 < 3 || (nd == 3 && d3 < 3)) { fprintf(stderr, "bad dimensions\n"); return 1; }
    const size_t n = (size_t)d1 * d2 * d3;
    std::vector<float> a = load(argv[1], n), b = load(argv[2], n);

    double lo = a[0], hi = a[0], se = 0, maxerr = 0;
    #pragma omp parallel for reduction(min:lo) reduction(max:hi, maxerr) reduction(+:se)
    for (long long k = 0; k < (long long)n; k++) {
        const double e = fabs((double)a[k] - (double)b[k]);
        se += e * e;
        if (e > maxerr) maxerr = e;
        if (a[k] < lo) lo = a[k];
        if (a[k] > hi) hi = a[k];
    }
    const double range = hi - lo, rmse = sqrt(se / (double)n);
    const double psnr = (rmse > 0 && range > 0) ? 20.0 * log10(range / rmse) : INFINITY;

    const long long stride3[3] = {d2 * d3, d3, 1}, stride2[2] = {d2, 1};
    const long long *stride = nd == 3 ? stride3 : stride2;
    long long orig = 0, kept = 0, fn = 0, fp = 0, ft = 0;
    #pragma omp parallel for collapse(2) reduction(+:orig, kept, fn, fp, ft)
    for (long long x = 1; x < d1 - 1; x++)
        for (long long y = 1; y < d2 - 1; y++)
            for (long long z = (nd == 3 ? 1 : 0); z < (nd == 3 ? d3 - 1 : 1); z++) {
                const long long k = (x * d2 + y) * d3 + z;
                const int t0 = classify_tb(a.data(), k, stride, nd), t1 = classify_tb(b.data(), k, stride, nd);
                if (t0) { orig++; if (t1 == t0) kept++; else if (t1 == 0) fn++; else ft++; }
                else if (t1) fp++;
            }
    printf("QUALITY psnr=%.4f nrmse=%.6g maxerr=%.9g range=%.9g tb_orig=%lld tb_kept=%lld tb_FN=%lld tb_FP=%lld tb_FT=%lld\n",
           psnr, range > 0 ? rmse / range : 0.0, maxerr, range, orig, kept, fn, fp, ft);
    return 0;
}
