/**
 * pl_topo_check: saddle preservation measured with the standard
 * piecewise-linear (PL) definition of critical points, independent of the
 * axis rule TopoSZp uses internally.
 *
 *   pl_topo_check [--f64] [--no-pers] <original> <decompressed> <absErrBound> <d1> <d2> [d3]
 *   pl_topo_check --selftest
 *
 * Files are raw float32 by default, float64 with --f64 (both files).
 * Build: g++ -O2 -fopenmp -ISZp/include example/pl_topo_check.cc -o pl_topo_check
 * Layout: index = (x*d2 + y)*d3 + z, x along d1 (slowest), as in topo_check.
 *
 * Definition (as in TTK's ScalarFieldCriticalPoints):
 *  - The grid is split into triangles (2D) or tetrahedra (3D) with the
 *    Freudenthal (Kuhn) triangulation: every grid point has 6 (2D) or 14 (3D)
 *    neighbours, the axis neighbours plus the diagonals along (+,+[,+]).
 *  - Ties are broken by index (simulation of simplicity): a < b iff
 *    f(a) < f(b), or f(a) == f(b) and index(a) < index(b). Flat areas of the
 *    quantized field are then ordered by index, as TTK does.
 *  - For each point, the lower link (neighbours below it) and upper link
 *    (neighbours above it) are split into connected pieces along the
 *    triangulation. Lower link empty: minimum. Upper link empty: maximum.
 *    One lower and one upper piece: regular. Otherwise a saddle:
 *    2D: saddle (more than two pieces each = monkey saddle, also counted).
 *    3D: >=2 lower pieces and 1 upper piece: 1-saddle (index-1 saddle);
 *        1 lower piece and >=2 upper pieces: 2-saddle (index-2 saddle);
 *        >=2 of both: degenerate saddle.
 *  Only interior points are compared (same points as topo_check).
 *
 * Persistence: a sweep from low to high values merges the lower sets with
 * union-find (join tree); each saddle where two pieces merge is paired with
 * the younger minimum, persistence = f(saddle) - f(minimum). The sweep from
 * high to low pairs saddles with maxima (split tree). A saddle's persistence
 * is the largest of its pairs. Saddles in no such pair (3D saddles that
 * create loops or cavities, paired with another saddle) are listed as
 * "unpaired". Persistence is from the original field; low-persistence
 * saddles are small bumps comparable to the error bound.
 *
 * Reported for "raw" (plain SZp quantization, before topology correction)
 * and "final" (the decompressed file given):
 *  - per PL type: kept (same type, same point), FN (now regular), FT (now
 *    critical with another type), FP (new critical point);
 *  - original saddles kept, by persistence threshold;
 *  - how PL saddles relate to TopoSZp's own axis-rule types (what TopoSZp
 *    stores and restores), and preservation for each group.
 * The last line, starting with PLTOPO, is a one-line summary for scripts.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <string>
#include <random>
#include "szp_quant.h"
#if defined(_OPENMP) && defined(__GLIBCXX__)
#include <parallel/algorithm>
#define PL_SORT __gnu_parallel::sort
#else
#define PL_SORT std::sort
#endif

typedef uint32_t idx_t;   /* grids up to 2^32 - 2 points */

/* ------------------------------------------------------------------ grid */
struct Grid {
    size_t d1, d2, d3;
    int nd;
    size_t n() const { return d1 * d2 * d3; }
};

/* Neighbour offsets, link edges and link triangles of the Freudenthal
   triangulation. Two grid points are joined by an edge when their offset has
   all components in {0,1} or all in {0,-1}; every set of pairwise joined
   points is a simplex. */
struct Stencil {
    int nb = 0;
    int d[14][3];
    std::vector<std::pair<int, int>> edges;
    std::vector<std::array<int, 3>> tris;
};

static bool is_edge(const int *a, const int *b)
{
    int e[3] = {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    bool nz = e[0] || e[1] || e[2];
    bool pos = e[0] >= 0 && e[1] >= 0 && e[2] >= 0 && e[0] <= 1 && e[1] <= 1 && e[2] <= 1;
    bool neg = e[0] <= 0 && e[1] <= 0 && e[2] <= 0 && e[0] >= -1 && e[1] >= -1 && e[2] >= -1;
    return nz && (pos || neg);
}

static Stencil make_stencil(int nd)
{
    Stencil s;
    int zero[3] = {0, 0, 0};
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
            for (int z = (nd == 3 ? -1 : 0); z <= (nd == 3 ? 1 : 0); z++) {
                int o[3] = {x, y, z};
                if (is_edge(o, zero)) { memcpy(s.d[s.nb], o, sizeof(o)); s.nb++; }
            }
    for (int a = 0; a < s.nb; a++)
        for (int b = a + 1; b < s.nb; b++)
            if (is_edge(s.d[a], s.d[b])) s.edges.push_back({a, b});
    if (nd == 3)
        for (int a = 0; a < s.nb; a++)
            for (int b = a + 1; b < s.nb; b++)
                for (int c = b + 1; c < s.nb; c++)
                    if (is_edge(s.d[a], s.d[b]) && is_edge(s.d[a], s.d[c]) && is_edge(s.d[b], s.d[c]))
                        s.tris.push_back({a, b, c});
    /* Link of an interior point: a hexagon (2D) or a sphere with 14 vertices,
       36 edges and 24 triangles (3D). */
    bool ok = nd == 2 ? (s.nb == 6 && s.edges.size() == 6)
                      : (s.nb == 14 && s.edges.size() == 36 && s.tris.size() == 24);
    if (!ok) { fprintf(stderr, "pl_topo_check: internal error in the triangulation tables\n"); exit(2); }
    return s;
}

/* PL types. Codes 1 and 2 match topo_check (1 = maximum, 2 = minimum). */
enum { PL_REG = 0, PL_MAX = 1, PL_MIN = 2, PL_SAD1 = 3, PL_SAD2 = 4, PL_SADX = 5, PL_NT = 6 };
static bool pl_is_saddle(int t) { return t >= PL_SAD1; }

struct LinkInfo { int lower_cc, upper_cc, chi_lower; };

/* Lower/upper link pieces of point p at (x,y,z); neighbours outside the grid
   are left out (link of a boundary point). */
template <typename T>
static LinkInfo link_info(const T *f, const Grid &g, const Stencil &s, size_t x, size_t y, size_t z)
{
    const size_t p = (x * g.d2 + y) * g.d3 + z;
    const T fp = f[p];
    signed char side[14];   /* -1 lower, +1 upper, 0 outside */
    int nlow = 0;
    for (int a = 0; a < s.nb; a++) {
        long X = (long)x + s.d[a][0], Y = (long)y + s.d[a][1], Z = (long)z + s.d[a][2];
        if (X < 0 || Y < 0 || Z < 0 || X >= (long)g.d1 || Y >= (long)g.d2 || Z >= (long)g.d3) { side[a] = 0; continue; }
        size_t q = ((size_t)X * g.d2 + (size_t)Y) * g.d3 + (size_t)Z;
        bool lower = f[q] < fp || (f[q] == fp && q < p);
        side[a] = lower ? -1 : 1;
        nlow += lower;
    }
    int par[14];
    for (int a = 0; a < s.nb; a++) par[a] = a;
    auto find = [&](int a) { while (par[a] != a) a = par[a] = par[par[a]]; return a; };
    int nedge_low = 0;
    for (auto &e : s.edges) {
        if (side[e.first] == 0 || side[e.first] != side[e.second]) continue;
        if (side[e.first] < 0) nedge_low++;
        int r1 = find(e.first), r2 = find(e.second);
        if (r1 != r2) par[r1] = r2;
    }
    int ntri_low = 0;
    for (auto &t : s.tris)
        if (side[t[0]] < 0 && side[t[1]] < 0 && side[t[2]] < 0) ntri_low++;
    LinkInfo li = {0, 0, nlow - nedge_low + ntri_low};
    for (int a = 0; a < s.nb; a++)
        if (side[a] && find(a) == a) (side[a] < 0 ? li.lower_cc : li.upper_cc)++;
    return li;
}

static int pl_type(const LinkInfo &li, int nd)
{
    if (li.lower_cc == 0) return PL_MIN;
    if (li.upper_cc == 0) return PL_MAX;
    if (li.lower_cc == 1 && li.upper_cc == 1) return PL_REG;
    if (nd == 2) return PL_SAD1;
    if (li.upper_cc == 1) return PL_SAD1;
    if (li.lower_cc == 1) return PL_SAD2;
    return PL_SADX;
}

/* TopoSZp's own rule (topo_check / szp_find_critical_points[_3d]):
   1 max, 2 min, 3 saddle, 0 regular. */
template <typename T>
static int axis_type(const T *a, const Grid &g, size_t x, size_t y, size_t z)
{
    if (g.nd == 2) {
        size_t C = g.d2, k = x * C + y;
        T c = a[k], n = a[k - C], s = a[k + C], w = a[k - 1], e = a[k + 1];
        if (c > n && c > s && c > w && c > e) return 1;
        if (c < n && c < s && c < w && c < e) return 2;
        if ((c < n && c < s && c > w && c > e) || (c > n && c > s && c < w && c < e)) return 3;
        return 0;
    }
    size_t P = g.d2 * g.d3, k = x * P + y * g.d3 + z;
    T c = a[k];
    T xm = a[k - P], xp = a[k + P], ym = a[k - g.d3], yp = a[k + g.d3], zm = a[k - 1], zp = a[k + 1];
    if (c > xm && c > xp && c > ym && c > yp && c > zm && c > zp) return 1;
    if (c < xm && c < xp && c < ym && c < yp && c < zm && c < zp) return 2;
    int hi = (c > xm && c > xp) + (c > ym && c > yp) + (c > zm && c > zp);
    int lo = (c < xm && c < xp) + (c < ym && c < yp) + (c < zm && c < zp);
    return (hi >= 1 && lo >= 1) ? 3 : 0;
}

/* PL type of every interior point (boundary points get PL_REG). */
template <typename T>
static std::vector<signed char> classify_all(const T *f, const Grid &g, const Stencil &s)
{
    std::vector<signed char> t(g.n(), PL_REG);
    const size_t zlo = g.nd == 3 ? 1 : 0, zhi = g.nd == 3 ? g.d3 - 1 : 1;
    #pragma omp parallel for schedule(dynamic, 1) collapse(2)
    for (size_t x = 1; x < g.d1 - 1; x++)
        for (size_t y = 1; y < g.d2 - 1; y++)
            for (size_t z = zlo; z < zhi; z++)
                t[(x * g.d2 + y) * g.d3 + z] = (signed char)pl_type(link_info(f, g, s, x, y, z), g.nd);
    return t;
}

/* ---------------------------------------------------------- persistence */
/* Persistence of each listed saddle (sorted indices) from the join and split
   trees; -1 = in no extremum-saddle pair. */
template <typename T>
static std::vector<double> saddle_persistence(const T *f, const Grid &g, const Stencil &s,
                                              const std::vector<idx_t> &saddles)
{
    const size_t N = g.n();
    std::vector<double> pers(saddles.size(), -1.0);
    std::vector<idx_t> order(N);
    std::iota(order.begin(), order.end(), (idx_t)0);
    PL_SORT(order.begin(), order.end(), [f](idx_t a, idx_t b) { return f[a] < f[b] || (f[a] == f[b] && a < b); });

    const idx_t NONE = (idx_t)-1;
    std::vector<idx_t> par(N);
    long off[14];
    for (int a = 0; a < s.nb; a++) off[a] = ((long)s.d[a][0] * (long)g.d2 + s.d[a][1]) * (long)g.d3 + s.d[a][2];

    for (int up = 1; up >= 0; up--) {
        /* up = 1: join tree (low to high, roots are minima);
           up = 0: split tree (high to low, roots are maxima).
           A component's root is always its oldest extremum. */
        std::fill(par.begin(), par.end(), NONE);
        auto find = [&](idx_t a) { while (par[a] != a) a = par[a] = par[par[a]]; return a; };
        auto older = [&](idx_t a, idx_t b) {   /* is a born before b in this sweep */
            bool lo = f[a] < f[b] || (f[a] == f[b] && a < b);
            return up ? lo : !lo;
        };
        for (size_t t = 0; t < N; t++) {
            idx_t v = up ? order[t] : order[N - 1 - t];
            size_t z = v % g.d3, y = (v / g.d3) % g.d2, x = v / (g.d3 * g.d2);
            idx_t roots[14];
            int nr = 0;
            for (int a = 0; a < s.nb; a++) {
                long X = (long)x + s.d[a][0], Y = (long)y + s.d[a][1], Z = (long)z + s.d[a][2];
                if (X < 0 || Y < 0 || Z < 0 || X >= (long)g.d1 || Y >= (long)g.d2 || Z >= (long)g.d3) continue;
                idx_t q = (idx_t)((long)v + off[a]);
                if (par[q] == NONE) continue;
                idx_t r = find(q);
                bool seen = false;
                for (int i = 0; i < nr; i++) seen |= roots[i] == r;
                if (!seen) roots[nr++] = r;
            }
            if (nr == 0) { par[v] = v; continue; }
            idx_t keep = roots[0];
            for (int i = 1; i < nr; i++) if (older(roots[i], keep)) keep = roots[i];
            par[v] = keep;
            if (nr < 2) continue;
            double p = 0;
            for (int i = 0; i < nr; i++) {
                if (roots[i] == keep) continue;
                par[roots[i]] = keep;
                p = std::max(p, fabs((double)f[v] - (double)f[roots[i]]));
            }
            auto it = std::lower_bound(saddles.begin(), saddles.end(), v);
            if (it != saddles.end() && *it == v) {
                double &dst = pers[it - saddles.begin()];
                dst = std::max(dst, p);
            }
        }
    }
    return pers;
}

/* ------------------------------------------------------------------- I/O */
template <typename T>
static std::vector<T> read_raw(const char *path, size_t n)
{
    std::vector<T> v(n);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    size_t got = fread(v.data(), sizeof(T), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "%s: read %zu of %zu values\n", path, got, n); exit(1); }
    return v;
}

/* ------------------------------------------------------------------ main */
template <typename T>
static int run(const char *orig_path, const char *dec_path, double eb, const Grid &g, bool want_pers)
{
    const size_t N = g.n();
    const char *tname = sizeof(T) == sizeof(float) ? "float32" : "float64";
    const Stencil s = make_stencil(g.nd);
    std::vector<T> a = read_raw<T>(orig_path, N), fin = read_raw<T>(dec_path, N), raw(N);
    szp_check_errbound<T>(a.data(), N, eb, "pl_topo_check");
    const double inv = 1.0 / eb;
    #pragma omp parallel for
    for (long long k = 0; k < (long long)N; k++)
        raw[k] = szp_dequantize<T>(szp_quantize<T>(a[k], eb, inv), eb);

    std::vector<signed char> t0 = classify_all(a.data(), g, s);
    std::vector<signed char> tr = classify_all(raw.data(), g, s);
    std::vector<signed char> tf = classify_all(fin.data(), g, s);
    std::vector<T>().swap(raw);

    /* Per-type kept / FN / FT / FP, and cross-table with the axis rule. */
    long orig[PL_NT] = {0}, kept[2][PL_NT] = {{0}}, fn[2][PL_NT] = {{0}}, ft[2][PL_NT] = {{0}}, fp[2][PL_NT] = {{0}};
    long anysad[2] = {0, 0};          /* PL saddle still some kind of saddle */
    long cross[PL_NT][4] = {{0}};     /* [PL type][axis type] in the original */
    long by_axis[4][3] = {{0}};       /* PL saddles: [axis type][count, raw kept, final kept] */
    long axis_sad_total = 0;
    std::vector<idx_t> sad;           /* original PL saddles */
    std::vector<unsigned char> sflag; /* per saddle: 1 raw kept, 2 final kept, 4 raw near, 8 final near */
    /* "near": a critical point of the same type within one cell (3x3[x3] box) */
    auto near = [&](const std::vector<signed char> &t, size_t x, size_t y, size_t z, int type) {
        for (size_t X = x - 1; X <= x + 1; X++)
            for (size_t Y = y - 1; Y <= y + 1; Y++)
                for (size_t Z = (g.nd == 3 ? z - 1 : z); Z <= (g.nd == 3 ? z + 1 : z); Z++)
                    if (t[(X * g.d2 + Y) * g.d3 + Z] == type) return true;
        return false;
    };
    const size_t zlo = g.nd == 3 ? 1 : 0, zhi = g.nd == 3 ? g.d3 - 1 : 1;
    for (size_t x = 1; x + 1 < g.d1; x++)
        for (size_t y = 1; y + 1 < g.d2; y++)
            for (size_t z = zlo; z < zhi; z++) {
                size_t k = (x * g.d2 + y) * g.d3 + z;
                int o = t0[k], at = axis_type(a.data(), g, x, y, z);
                cross[o][at]++;
                if (at == 3) axis_sad_total++;
                for (int w = 0; w < 2; w++) {
                    int n1 = w ? tf[k] : tr[k];
                    if (o) {
                        if (n1 == o) kept[w][o]++;
                        else if (n1 == PL_REG) fn[w][o]++;
                        else ft[w][o]++;
                        if (pl_is_saddle(o) && pl_is_saddle(n1)) anysad[w]++;
                    } else if (n1) fp[w][n1]++;
                }
                if (o) orig[o]++;
                if (pl_is_saddle(o)) {
                    sad.push_back((idx_t)k);
                    sflag.push_back((unsigned char)((tr[k] == o) | (tf[k] == o) << 1 |
                                                    near(tr, x, y, z, o) << 2 | near(tf, x, y, z, o) << 3));
                    by_axis[at][0]++;
                    by_axis[at][1] += tr[k] == o;
                    by_axis[at][2] += tf[k] == o;
                }
            }

    const char *pln2[PL_NT] = {"regular", "maxima", "minima", "saddles", "-", "-"};
    const char *pln3[PL_NT] = {"regular", "maxima", "minima", "1-saddles", "2-saddles", "degen-sad"};
    const char *const *pln = g.nd == 3 ? pln3 : pln2;
    auto pct = [](long k, long o) { return o ? 100.0 * k / o : 100.0; };

    std::string dims = std::to_string(g.d1) + "x" + std::to_string(g.d2) + (g.nd == 3 ? "x" + std::to_string(g.d3) : "");
    printf("PL critical points (Freudenthal triangulation, %d neighbours, ties by index), %s grid %s, %s, abs eb %.6g\n",
           s.nb, g.nd == 3 ? "3D" : "2D", dims.c_str(), tname, eb);
    printf("%-10s %10s | %-38s | %-38s\n", "", "original", "raw SZp (before topology correction)", "final TopoSZp output");
    printf("%-10s %10s | %9s %8s %8s %8s %3s| %9s %8s %8s %8s %3s\n", "type", "count",
           "kept", "FN", "FT", "FP", "", "kept", "FN", "FT", "FP", "");
    long tot[5][2] = {{0}};
    int last = g.nd == 3 ? PL_SADX : PL_SAD1;
    for (int t = 1; t <= last; t++) {
        printf("%-10s %10ld | %9ld %8ld %8ld %8ld %3.0f%%| %9ld %8ld %8ld %8ld %3.0f%%\n", pln[t], orig[t],
               kept[0][t], fn[0][t], ft[0][t], fp[0][t], pct(kept[0][t], orig[t]),
               kept[1][t], fn[1][t], ft[1][t], fp[1][t], pct(kept[1][t], orig[t]));
        for (int w = 0; w < 2; w++) {
            tot[0][w] += orig[t]; tot[1][w] += kept[w][t]; tot[2][w] += fn[w][t];
            tot[3][w] += ft[w][t]; tot[4][w] += fp[w][t];
        }
    }
    printf("%-10s %10ld | %9ld %8ld %8ld %8ld %3.0f%%| %9ld %8ld %8ld %8ld %3.0f%%\n", "total", tot[0][0],
           tot[1][0], tot[2][0], tot[3][0], tot[4][0], pct(tot[1][0], tot[0][0]),
           tot[1][1], tot[2][1], tot[3][1], tot[4][1], pct(tot[1][1], tot[0][1]));
    long nsad = (long)sad.size(), ksad[2] = {0, 0};
    for (int t = PL_SAD1; t <= last; t++) { ksad[0] += kept[0][t]; ksad[1] += kept[1][t]; }
    long nr_near = 0, nf_near = 0;
    for (unsigned char fl : sflag) { nr_near += (fl >> 2) & 1; nf_near += (fl >> 3) & 1; }
    printf("PL saddles kept: raw %ld/%ld (%.1f%%), final %ld/%ld (%.1f%%); still a saddle of any kind: raw %.1f%%, final %.1f%%\n",
           ksad[0], nsad, pct(ksad[0], nsad), ksad[1], nsad, pct(ksad[1], nsad), pct(anysad[0], nsad), pct(anysad[1], nsad));
    printf("PL saddles kept within one cell (same type at the point or a neighbour, i.e. moved by one cell at most): "
           "raw %.1f%%, final %.1f%%\n", pct(nr_near, nsad), pct(nf_near, nsad));

    /* Relation to TopoSZp's own (axis-rule) types. */
    printf("\nPL type vs TopoSZp axis-rule type, original field (axis rule = what TopoSZp stores and restores)\n");
    printf("%-10s %10s %10s %10s %10s\n", "PL \\ axis", "saddle", "maximum", "minimum", "regular");
    for (int t = 0; t <= last; t++)
        printf("%-10s %10ld %10ld %10ld %10ld\n", pln[t], cross[t][3], cross[t][1], cross[t][2], cross[t][0]);
    long axis_sad_pl_sad = 0;
    for (int t = PL_SAD1; t <= last; t++) axis_sad_pl_sad += cross[t][3];
    printf("TopoSZp saddles that are PL saddles: %ld of %ld (%.1f%%); PL saddles TopoSZp also stores as saddles: %.1f%%\n",
           axis_sad_pl_sad, axis_sad_total, pct(axis_sad_pl_sad, axis_sad_total), pct(axis_sad_pl_sad, nsad));
    printf("PL saddles kept, by TopoSZp axis type at the point:\n");
    printf("  %-22s %10s %12s %12s\n", "axis type", "PL saddles", "raw kept", "final kept");
    const char *an[4] = {"regular (not stored)", "maximum", "minimum", "saddle (restored)"};
    for (int at : {3, 0, 1, 2})
        if (by_axis[at][0])
            printf("  %-22s %10ld %11.1f%% %11.1f%%\n", an[at], by_axis[at][0],
                   pct(by_axis[at][1], by_axis[at][0]), pct(by_axis[at][2], by_axis[at][0]));

    /* Preservation by persistence. */
    double kept_sig = -1;
    if (want_pers && nsad) {
        std::vector<double> pers = saddle_persistence(a.data(), g, s, sad);
        const double th[] = {0, 1, 2, 5, 10, 100, 1000};
        printf("\nOriginal PL saddles by persistence (join/split trees of the original field)\n");
        printf("  kept = same type at the same point; +-1 = same type within one cell\n");
        printf("  %-18s %10s %10s %10s %10s %10s\n", "persistence", "saddles", "raw kept", "raw +-1", "final kept", "final +-1");
        auto row = [&](const char *lab, auto sel) {
            long c = 0, k[4] = {0, 0, 0, 0};
            for (size_t i = 0; i < sad.size(); i++)
                if (sel(pers[i])) { c++; for (int b = 0; b < 4; b++) k[b] += (sflag[i] >> b) & 1; }
            printf("  %-18s %10ld %9.1f%% %9.1f%% %9.1f%% %9.1f%%\n", lab, c,
                   pct(k[0], c), pct(k[2], c), pct(k[1], c), pct(k[3], c));
            return c ? pct(k[1], c) : -1.0;
        };
        row("paired (any)", [](double p) { return p >= 0; });
        for (double m : th) {
            if (m == 0) continue;
            char lab[32];
            snprintf(lab, sizeof(lab), ">= %g eb", m);
            double r = row(lab, [&](double p) { return p >= 0 && p >= m * eb; });
            if (m == 2) kept_sig = r;
        }
        row("unpaired", [](double p) { return p < 0; });
    }

    printf("PLTOPO saddles=%ld kept_raw=%.1f%% kept_final=%.1f%% kept_final_1cell=%.1f%% kept_final_pers>=2eb=%.1f%% axis_saddles=%ld axis_saddles_that_are_PL=%.1f%% "
           "extrema=%ld extrema_kept_final=%.1f%% FP_final=%ld FT_final=%ld\n",
           nsad, pct(ksad[0], nsad), pct(ksad[1], nsad), pct(nf_near, nsad), kept_sig, axis_sad_total, pct(axis_sad_pl_sad, axis_sad_total),
           orig[PL_MAX] + orig[PL_MIN], pct(kept[1][PL_MAX] + kept[1][PL_MIN], orig[PL_MAX] + orig[PL_MIN]),
           tot[4][1], tot[3][1]);
    return 0;
}

/* ------------------------------------------------------------- self-test */
/* Checks the triangulation and classification on small random fields with
   many ties:
   (1) sum over all points of 1 - chi(lower link) = chi(rectangle/box) = 1;
   (2) at interior points, the piece counts agree with chi(lower link)
       (2D: chi = lower pieces unless maximum; 3D: chi = lower - upper + 1);
   (3) a non-degenerate function with known critical points. */
static int selftest()
{
    std::mt19937 rng(12345);
    int fails = 0;
    for (int nd = 2; nd <= 3; nd++) {
        Stencil s = make_stencil(nd);
        for (int trial = 0; trial < 200; trial++) {
            Grid g = {(size_t)(3 + rng() % 6), (size_t)(3 + rng() % 6), nd == 3 ? (size_t)(3 + rng() % 6) : 1, nd};
            std::vector<float> f(g.n());
            int levels = 1 + rng() % 5;
            for (auto &v : f) v = (float)(rng() % levels);
            long sum = 0;
            for (size_t x = 0; x < g.d1; x++)
                for (size_t y = 0; y < g.d2; y++)
                    for (size_t z = 0; z < g.d3; z++) {
                        LinkInfo li = link_info(f.data(), g, s, x, y, z);
                        sum += 1 - li.chi_lower;
                        bool inner = x > 0 && y > 0 && x + 1 < g.d1 && y + 1 < g.d2 && (nd == 2 || (z > 0 && z + 1 < g.d3));
                        if (!inner) continue;
                        int expect;
                        if (li.upper_cc == 0) expect = nd == 2 ? 0 : 2;
                        else if (li.lower_cc == 0) expect = 0;
                        else expect = nd == 2 ? li.lower_cc : li.lower_cc - li.upper_cc + 1;
                        if (li.chi_lower != expect) fails++;
                    }
            if (sum != 1) fails++;
        }
        /* f = (x-c)^2 + (y-c)^2 - (z-c)^2 style saddle in the centre */
        Grid g = {9, 9, nd == 3 ? (size_t)9 : 1, nd};
        std::vector<float> f(g.n());
        for (size_t x = 0; x < 9; x++)
            for (size_t y = 0; y < 9; y++)
                for (size_t z = 0; z < g.d3; z++) {
                    double X = x - 4.0 + 0.1, Y = y - 4.0 + 0.2, Z = z - 4.0 + 0.3;
                    f[(x * 9 + y) * g.d3 + z] = nd == 2 ? (float)(X * X - Y * Y) : (float)(X * X + Y * Y - Z * Z);
                }
        auto t = classify_all(f.data(), g, s);
        long cnt[PL_NT] = {0};
        for (auto v : t) cnt[(int)v]++;
        /* 2D x^2-y^2: one saddle. 3D x^2+y^2-z^2: one 1-saddle (lower link =
           two caps around the z axis, upper link = one ring). */
        bool ok = nd == 2 ? (cnt[PL_SAD1] == 1 && cnt[PL_MAX] + cnt[PL_MIN] == 0)
                          : (cnt[PL_SAD1] == 1 && cnt[PL_SAD2] + cnt[PL_SADX] + cnt[PL_MAX] + cnt[PL_MIN] == 0);
        if (!ok) { fails++; fprintf(stderr, "selftest %dD quadratic saddle: max %ld min %ld s1 %ld s2 %ld sx %ld\n",
                                    nd, cnt[PL_MAX], cnt[PL_MIN], cnt[PL_SAD1], cnt[PL_SAD2], cnt[PL_SADX]); }
        /* Persistence on a 1D-like double well: two minima separated by a
           barrier of height 3 -> the saddle on the barrier has persistence 3. */
        Grid h = {3, 9, nd == 3 ? (size_t)3 : 1, nd};
        std::vector<float> w(h.n());
        const float prof[9] = {9, 0, 5, 6, 7, 4, 2, 5, 9};   /* minima 0 and 2, barrier top 7 */
        for (size_t x = 0; x < h.d1; x++)
            for (size_t y = 0; y < h.d2; y++)
                for (size_t z = 0; z < h.d3; z++)
                    w[(x * h.d2 + y) * h.d3 + z] = prof[y] + 0.01f * (float)(x + z);
        /* the merge happens at the cheapest crossing: y = 4, x = z = 0 */
        std::vector<idx_t> one = {(idx_t)((0 * h.d2 + 4) * h.d3 + 0)};
        auto p = saddle_persistence(w.data(), h, s, one);
        if (!(fabs(p[0] - 5.0) < 1e-4)) { fails++; fprintf(stderr, "selftest %dD persistence: got %g, want 5\n", nd, p[0]); }
    }
    printf("pl_topo_check selftest: %s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "--selftest") == 0) return selftest();
    bool f64 = false, pers = true;
    while (argc > 1 && argv[1][0] == '-' && argv[1][1] == '-') {
        if (strcmp(argv[1], "--f64") == 0) f64 = true;
        else if (strcmp(argv[1], "--no-pers") == 0) pers = false;
        else { fprintf(stderr, "unknown option %s\n", argv[1]); return 1; }
        argv++; argc--;
    }
    if (argc < 6) {
        fprintf(stderr, "usage: %s [--f64] [--no-pers] <original> <decompressed> <absErrBound> <d1> <d2> [d3]\n"
                        "       %s --selftest\n", argv[0], argv[0]);
        return 1;
    }
    double eb = atof(argv[3]);
    if (!f64) eb = (double)(float)eb;   /* the float APIs take the bound as float */
    Grid g;
    g.d1 = strtoull(argv[4], 0, 10); g.d2 = strtoull(argv[5], 0, 10);
    g.nd = argc > 6 ? 3 : 2; g.d3 = g.nd == 3 ? strtoull(argv[6], 0, 10) : 1;
    if (!(eb > 0) || g.d1 < 3 || g.d2 < 3 || (g.nd == 3 && g.d3 < 3)) { fprintf(stderr, "bad arguments\n"); return 1; }
    if (g.n() >= (size_t)UINT32_MAX) { fprintf(stderr, "grid too large for this tool (max 2^32-2 points)\n"); return 1; }
    return f64 ? run<double>(argv[1], argv[2], eb, g, pers) : run<float>(argv[1], argv[2], eb, g, pers);
}
