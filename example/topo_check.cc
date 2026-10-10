/**
 * topo_check: topology and error report for TopoSZp output.
 *
 *   topo_check <original.f32> <decompressed.f32> <absErrBound> <d1> <d2> [d3]
 *
 * 2D uses the 4-neighbour rule of szp_find_critical_points, 3D the
 * 6-neighbour rule of szp_find_critical_points_3d (saddle = at least one
 * axis where the point is above both neighbours and one where it is below).
 *
 * "raw" is plain SZp quantization of the original, computed here exactly as
 * the decoder does it ((float)q * eb with q = (int)(v / eb)), i.e. the field
 * before any topology correction. "final" is the decompressed file given.
 *
 * Per type it reports: kept (same type, same place), FN (lost: point is
 * regular afterwards), FT (point is critical afterwards but with another
 * type) and FP (new critical point at an originally regular point).
 * The last line, starting with TOPO, is a one-line summary for scripts.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

static std::vector<float> read_f32(const char *path, size_t n)
{
    std::vector<float> v(n);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    size_t got = fread(v.data(), sizeof(float), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "%s: read %zu of %zu floats\n", path, got, n); exit(1); }
    return v;
}

struct Grid { size_t d1, d2, d3; bool is3d; };

static int classify(const std::vector<float> &a, const Grid &g, size_t x, size_t y, size_t z)
{
    if (!g.is3d) {
        size_t C = g.d2, k = x * C + y;
        float c = a[k], n = a[k - C], s = a[k + C], w = a[k - 1], e = a[k + 1];
        if (c > n && c > s && c > w && c > e) return 1;
        if (c < n && c < s && c < w && c < e) return 2;
        if ((c < n && c < s && c > w && c > e) || (c > n && c > s && c < w && c < e)) return 3;
        return 0;
    }
    size_t P = g.d2 * g.d3, k = x * P + y * g.d3 + z;
    float c = a[k];
    float xm = a[k - P], xp = a[k + P], ym = a[k - g.d3], yp = a[k + g.d3], zm = a[k - 1], zp = a[k + 1];
    if (c > xm && c > xp && c > ym && c > yp && c > zm && c > zp) return 1;
    if (c < xm && c < xp && c < ym && c < yp && c < zm && c < zp) return 2;
    int hi = (c > xm && c > xp) + (c > ym && c > yp) + (c > zm && c > zp);
    int lo = (c < xm && c < xp) + (c < ym && c < yp) + (c < zm && c < zp);
    return (hi >= 1 && lo >= 1) ? 3 : 0;
}

struct Stats { long orig[4] = {0}, kept[4] = {0}, fn[4] = {0}, ft[4] = {0}, fp[4] = {0}; };

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "usage: %s <original.f32> <decompressed.f32> <absErrBound> <d1> <d2> [d3]\n", argv[0]);
        return 1;
    }
    float eb = (float)atof(argv[3]);
    Grid g;
    g.d1 = strtoull(argv[4], 0, 10); g.d2 = strtoull(argv[5], 0, 10);
    g.is3d = argc > 6; g.d3 = g.is3d ? strtoull(argv[6], 0, 10) : 1;
    size_t N = g.d1 * g.d2 * g.d3;
    if (!(eb > 0) || g.d1 < 3 || g.d2 < 3 || (g.is3d && g.d3 < 3)) { fprintf(stderr, "bad arguments\n"); return 1; }

    std::vector<float> a = read_f32(argv[1], N), fin = read_f32(argv[2], N), raw(N);
    const double inv = 1.0 / (double)eb;
    double err_raw = 0, err_fin = 0;
    for (size_t k = 0; k < N; k++) {
        raw[k] = (float)(int)((double)a[k] * inv) * eb;
        err_raw = std::max(err_raw, fabs((double)a[k] - (double)raw[k]));
        err_fin = std::max(err_fin, fabs((double)a[k] - (double)fin[k]));
    }

    Stats sr, sf;
    struct Ext { int bin; float o, v; };
    std::vector<Ext> mx, mn;
    size_t zlo = g.is3d ? 1 : 0, zhi = g.is3d ? g.d3 - 1 : 1;
    for (size_t x = 1; x + 1 < g.d1; x++)
        for (size_t y = 1; y + 1 < g.d2; y++)
            for (size_t z = zlo; z < zhi; z++) {
                int t0 = classify(a, g, x, y, z);
                int tr = classify(raw, g, x, y, z);
                int tf = classify(fin, g, x, y, z);
                for (int s = 0; s < 2; s++) {
                    Stats &st = s ? sf : sr;
                    int t1 = s ? tf : tr;
                    if (t0) {
                        if (s == 0) { sr.orig[t0]++; sf.orig[t0]++; }
                        if (t1 == t0) st.kept[t0]++;
                        else if (t1 == 0) st.fn[t0]++;
                        else st.ft[t0]++;
                    } else if (t1) st.fp[t1]++;
                }
                if (t0 == 1 || t0 == 2) {
                    size_t k = g.is3d ? (x * g.d2 + y) * g.d3 + z : x * g.d2 + y;
                    Ext e = { (int)((double)a[k] * inv), a[k], fin[k] };
                    (t0 == 1 ? mx : mn).push_back(e);
                }
            }

    // Relative order of extrema of one type that share a quantization bin:
    // a smaller original value must stay strictly smaller in the output.
    auto order_violations = [](std::vector<Ext> &v) {
        std::sort(v.begin(), v.end(), [](const Ext &p, const Ext &q) { return p.bin != q.bin ? p.bin < q.bin : p.o < q.o; });
        long bad = 0;
        for (size_t k = 1; k < v.size(); k++)
            if (v[k].bin == v[k - 1].bin && v[k].o > v[k - 1].o && !(v[k].v > v[k - 1].v)) bad++;
        return bad;
    };
    long ov_max = order_violations(mx), ov_min = order_violations(mn);

    for (int t = 1; t <= 3; t++)
        for (Stats *st : {&sr, &sf}) {
            st->orig[0] += st->orig[t]; st->kept[0] += st->kept[t];
            st->fn[0] += st->fn[t]; st->ft[0] += st->ft[t]; st->fp[0] += st->fp[t];
        }

    const char *name[4] = {"total", "maxima", "minima", "saddles"};
    printf("%s grid %zux%zu%s, abs eb %.6g\n", g.is3d ? "3D" : "2D", g.d1, g.d2,
           g.is3d ? (std::string("x") + std::to_string(g.d3)).c_str() : "", eb);
    float amax = 0;
    for (size_t k = 0; k < N; k++) amax = std::max(amax, std::fabs(a[k]));
    float ulp = std::nextafter(amax, INFINITY) - amax;
    if (ulp > eb)
        printf("WARNING: float32 spacing at |value| %.6g is %.3g, larger than eb; the bound cannot be met in float32 "
               "(use a larger eb)\n", amax, ulp);
    printf("max error: raw %.6g (%.3f eb), final %.6g (%.3f eb)  %s\n", err_raw, err_raw / eb, err_fin, err_fin / eb,
           err_fin <= 2.0 * eb ? "PASS (<= 2 eb)" : "FAIL (> 2 eb)");
    printf("%-8s %10s | %-38s | %-38s\n", "", "original", "raw SZp (before topology correction)", "final TopoSZp output");
    printf("%-8s %10s | %9s %8s %8s %8s %3s| %9s %8s %8s %8s %3s\n", "type", "count",
           "kept", "FN", "FT", "FP", "", "kept", "FN", "FT", "FP", "");
    for (int t : {1, 2, 3, 0}) {
        auto pct = [](long k, long o) { return o ? 100.0 * k / o : 100.0; };
        printf("%-8s %10ld | %9ld %8ld %8ld %8ld %3.0f%%| %9ld %8ld %8ld %8ld %3.0f%%\n", name[t], sr.orig[t],
               sr.kept[t], sr.fn[t], sr.ft[t], sr.fp[t], pct(sr.kept[t], sr.orig[t]),
               sf.kept[t], sf.fn[t], sf.ft[t], sf.fp[t], pct(sf.kept[t], sf.orig[t]));
    }
    printf("extrema order violations within a bin (final): maxima %ld, minima %ld\n", ov_max, ov_min);
    printf("TOPO err/eb=%.3f orig=%ld kept=%ld FN=%ld(max %ld,min %ld,sad %ld) FT=%ld FP=%ld ordviol=%ld raw_kept=%ld\n",
           err_fin / eb, sf.orig[0], sf.kept[0], sf.fn[0], sf.fn[1], sf.fn[2], sf.fn[3], sf.ft[0], sf.fp[0],
           ov_max + ov_min, sr.kept[0]);
    return 0;
}
