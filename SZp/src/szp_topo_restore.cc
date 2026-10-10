/**
 *  @file szp_topo_restore.cc
 *  @brief CPU (OpenMP) topology restoration; see szp_topo_restore.h.
 */
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "szp_topo_restore.h"
#include "szp_pl_topology.h"
#include "szp_quant.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/* One saddle pass (stencil or RBF) over all colour classes. */
static size_t saddle_pass_cpu(float *data, const int *types, const float *orig,
                              const unsigned char *locks, const SzpGrid &g, float eb,
                              const double *w, int use_rbf)
{
    std::vector<long long> list;
    std::vector<size_t> start;
    szp_saddle_candidates(data, types, g, list, start);
    const int ncol = g.nd == 3 ? 27 : 9;
    size_t restored = 0;
    for (int c = 0; c < ncol; c++) {
        const long long lo = (long long)start[c], hi = (long long)start[c + 1];
        size_t got = 0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(+:got)
#endif
        for (long long i = lo; i < hi; i++) {
            long long k = list[(size_t)i];
            got += use_rbf ? szp_saddle_rbf_point(data, types, locks, orig, g, k, eb, w)
                           : szp_saddle_stencil_point(data, types, orig, g, k, eb);
        }
        restored += got;
    }
    return restored;
}

static void restore_cpu(float *data, const int *types, const int *sort_positions,
                        size_t extrema_count, const SzpGrid &g, float eb,
                        szp_topo_restore_stats *stats)
{
    szp_topo_restore_stats st;
    memset(&st, 0, sizeof(st));
    const size_t N = (size_t)g.n[0] * g.n[1] * g.n[2];
    std::vector<float> orig(data, data + N);

    szp_count_lost(data, types, g, &st.extrema, &st.extrema_lost_before, &st.saddles, NULL);

    /* 1. extrema */
    szp_restore_extrema_ranked(data, orig.data(), types, sort_positions, extrema_count, g, eb);
    szp_count_lost(data, types, g, NULL, NULL, NULL, &st.saddles_lost_before);

    /* 2-3. saddles: stencil then RBF, twice */
    std::vector<unsigned char> locks;
    szp_extremum_locks(types, g, locks);
    double w[27];
    szp_rbf_weights(g.nd, w);
    for (int round = 0; round < 2; round++) {
        st.saddles_by_stencil += saddle_pass_cpu(data, types, orig.data(), locks.data(), g, eb, w, 0);
        st.saddles_by_rbf += saddle_pass_cpu(data, types, orig.data(), locks.data(), g, eb, w, 1);
    }
    szp_count_lost(data, types, g, NULL, &st.extrema_lost_after, NULL, &st.saddles_lost_after);
    if (stats) *stats = st;
}

extern "C" void szp_topo_restore_float_2d(float *data, const int *types, const int *sort_positions,
                                          size_t extrema_count, int rows, int cols, float errBound,
                                          szp_topo_restore_stats *stats)
{
    if (!data || !types || rows < 3 || cols < 3 || !(errBound > 0)) return;
    restore_cpu(data, types, sort_positions, extrema_count, szp_grid_2d(rows, cols), errBound, stats);
}

extern "C" void szp_topo_restore_float_3d(float *data, const int *types, const int *sort_positions,
                                          size_t extrema_count, int d1, int d2, int d3, float errBound,
                                          szp_topo_restore_stats *stats)
{
    if (!data || !types || d1 < 3 || d2 < 3 || d3 < 3 || !(errBound > 0)) return;
    restore_cpu(data, types, sort_positions, extrema_count, szp_grid_3d(d1, d2, d3), errBound, stats);
}

/* ===================================================================== */
/*  PL mode (szp_pl_topology.h)                                          */
/* ===================================================================== */

extern "C" void szp_pl_classify_float(const float *data, int nd, int d1, int d2, int d3, signed char *codes)
{
    const SzpGrid g = nd == 3 ? szp_grid_3d(d1, d2, d3) : szp_grid_2d(d1, d2);
    const long long N = (long long)g.n[0] * g.n[1] * g.n[2];
    SzpPL p;
    szp_pl_init(g, p);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long long k = 0; k < N; k++)
        codes[k] = szp_is_interior(g, k) ? (signed char)szp_pl_classify(data, g, p, k) : 0;
}

extern "C" unsigned char *szp_pl_pack_saddle_kinds(const signed char *codes, size_t n, size_t *outSize)
{
    size_t cnt = 0;
    for (size_t k = 0; k < n; k++) cnt += codes[k] >= 3;
    const size_t bytes = sizeof(size_t) + (2 * cnt + 7) / 8;
    unsigned char *out = (unsigned char *)calloc(bytes, 1);
    if (!out) { *outSize = 0; return NULL; }
    memcpy(out, &cnt, sizeof(size_t));
    unsigned char *b = out + sizeof(size_t);
    size_t j = 0;
    for (size_t k = 0; k < n; k++)
        if (codes[k] >= 3) {
            unsigned int kind = (unsigned int)(codes[k] - 3) & 3u;   /* 0 1-saddle, 1 2-saddle, 2 degenerate */
            b[j >> 2] |= (unsigned char)(kind << (6 - 2 * (j & 3)));
            j++;
        }
    *outSize = bytes;
    return out;
}

extern "C" int szp_pl_apply_saddle_kinds(int *types, size_t n, const unsigned char *buf, size_t size)
{
    if (!buf || size < sizeof(size_t)) return -1;
    size_t cnt = 0, have = 0;
    memcpy(&cnt, buf, sizeof(size_t));
    for (size_t k = 0; k < n; k++) have += types[k] == 3;
    if (have != cnt || size < sizeof(size_t) + (2 * cnt + 7) / 8) return -1;
    const unsigned char *b = buf + sizeof(size_t);
    size_t j = 0;
    for (size_t k = 0; k < n; k++)
        if (types[k] == 3) {
            unsigned int kind = (b[j >> 2] >> (6 - 2 * (j & 3))) & 3u;
            types[k] = 3 + (int)(kind > 2 ? 2 : kind);
            j++;
        }
    return 0;
}

extern "C" CriticalPoint *szp_pl_critical_points_2d(const signed char *codes, const float *data,
                                                    int rows, int cols, float errBound, size_t *count)
{
    const size_t N = (size_t)rows * cols;
    const double eb = (double)errBound, inv = 1.0 / eb;
    size_t n = 0;
    for (size_t k = 0; k < N; k++) n += codes[k] != 0;
    *count = n;
    if (!n) return NULL;
    CriticalPoint *cp = (CriticalPoint *)malloc(n * sizeof(CriticalPoint));
    if (!cp) { *count = 0; return NULL; }
    size_t j = 0;
    for (size_t k = 0; k < N; k++)
        if (codes[k]) {
            CriticalPoint c;
            c.x = (int)(k / (size_t)cols); c.y = (int)(k % (size_t)cols);
            c.type = codes[k] > 3 ? 3 : codes[k];
            c.quantized_bin = szp_quantize<float>(data[k], eb, inv);
            c.sort_position = 0;
            cp[j++] = c;
        }
    return cp;
}

extern "C" CriticalPoint3D *szp_pl_critical_points_3d(const signed char *codes, const float *data,
                                                      int d1, int d2, int d3, float errBound, size_t *count)
{
    const size_t P = (size_t)d2 * d3, N = (size_t)d1 * P;
    const double eb = (double)errBound, inv = 1.0 / eb;
    size_t n = 0;
    for (size_t k = 0; k < N; k++) n += codes[k] != 0;
    *count = n;
    if (!n) return NULL;
    CriticalPoint3D *cp = (CriticalPoint3D *)malloc(n * sizeof(CriticalPoint3D));
    if (!cp) { *count = 0; return NULL; }
    size_t j = 0;
    for (size_t k = 0; k < N; k++)
        if (codes[k]) {
            CriticalPoint3D c;
            c.x = (int)(k / P); c.y = (int)((k / (size_t)d3) % (size_t)d2); c.z = (int)(k % (size_t)d3);
            c.type = codes[k] > 3 ? 3 : codes[k];
            c.quantized_bin = szp_quantize<float>(data[k], eb, inv);
            c.sort_position = 0;
            cp[j++] = c;
        }
    return cp;
}

/* One PL fix pass over all colour classes; returns fixes of stored saddles
   and of other points. */
static void pl_fix_pass_cpu(float *data, const int *codes, const float *orig, const SzpGrid &g,
                            float eb, size_t *sad_fixed, size_t *other_fixed)
{
    std::vector<long long> list;
    std::vector<size_t> start;
    szp_pl_candidates(data, codes, g, list, start);
    SzpPL p;
    szp_pl_init(g, p);
    const int ncol = g.nd == 3 ? 27 : 9;
    size_t sf = 0, of = 0;
    for (int c = 0; c < ncol; c++) {
        const long long lo = (long long)start[c], hi = (long long)start[c + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 256) reduction(+:sf, of)
#endif
        for (long long i = lo; i < hi; i++) {
            long long k = list[(size_t)i];
            if (szp_pl_fix_point(data, codes, orig, g, p, k, eb)) {
                if (codes[k] >= 3) sf++; else of++;
            }
        }
    }
    *sad_fixed = sf;
    *other_fixed = of;
}

extern "C" void szp_topo_restore_float_pl(float *data, const int *codes, const int *sort_positions,
                                          size_t extrema_count, int nd, int d1, int d2, int d3,
                                          float errBound, szp_topo_restore_stats *stats)
{
    if (!data || !codes || d1 < 3 || d2 < 3 || (nd == 3 && d3 < 3) || !(errBound > 0)) return;
    const SzpGrid g = nd == 3 ? szp_grid_3d(d1, d2, d3) : szp_grid_2d(d1, d2);
    const size_t N = (size_t)g.n[0] * g.n[1] * g.n[2];
    szp_topo_restore_stats st;
    memset(&st, 0, sizeof(st));
    std::vector<float> orig(data, data + N);

    SzpPLCounts c0 = szp_pl_count(data, codes, g);
    st.extrema = c0.ext; st.extrema_lost_before = c0.ext_lost; st.saddles = c0.sad;
    st.fp_before = c0.fp; st.ft_before = c0.ft;

    szp_pl_restore_extrema(data, orig.data(), codes, sort_positions, extrema_count, g, errBound);
    st.saddles_lost_before = szp_pl_count(data, codes, g).sad_lost;

    for (int round = 0; round < SZP_PL_ROUNDS; round++) {
        size_t sf = 0, of = 0;
        pl_fix_pass_cpu(data, codes, orig.data(), g, errBound, &sf, &of);
        st.saddles_by_stencil += sf;
        st.other_fixes += of;
        st.rounds = (size_t)round + 1;
        if (sf + of == 0) break;
    }
    SzpPLCounts c1 = szp_pl_count(data, codes, g);
    st.extrema_lost_after = c1.ext_lost; st.saddles_lost_after = c1.sad_lost;
    st.fp_after = c1.fp; st.ft_after = c1.ft;
    if (stats) *stats = st;
}

/* ---------------- exceptions (szp_pl_topology.h, step 3) -------------- */

static inline int pl_violates(int stored, int now)
{
    if (now == stored) return 0;
    if (now != 0) return 1;                  /* false positive or false type */
    return stored == SZP_PL_MAX || stored == SZP_PL_MIN;   /* lost extremum */
}

extern "C" void szp_pl_violations(const float *field, const int *codes, int nd, int d1, int d2, int d3,
                                  size_t *fp, size_t *ft, size_t *ext_lost, size_t *sad_lost)
{
    const SzpGrid g = nd == 3 ? szp_grid_3d(d1, d2, d3) : szp_grid_2d(d1, d2);
    SzpPLCounts c = szp_pl_count(field, codes, g);
    if (fp) *fp = c.fp;
    if (ft) *ft = c.ft;
    if (ext_lost) *ext_lost = c.ext_lost;
    if (sad_lost) *sad_lost = c.sad_lost;
}

extern "C" unsigned char *szp_pl_exceptions_float(const float *orig, float *field, const int *codes,
                                                  int nd, int d1, int d2, int d3,
                                                  size_t *outSize, size_t *count)
{
    const SzpGrid g = nd == 3 ? szp_grid_3d(d1, d2, d3) : szp_grid_2d(d1, d2);
    const long long N = (long long)g.n[0] * g.n[1] * g.n[2];
    SzpPL p;
    szp_pl_init(g, p);

    /* Initial violations (parallel scan, raster order). */
    std::vector<unsigned char> flag((size_t)N, 0);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long long k = 0; k < N; k++)
        flag[(size_t)k] = szp_is_interior(g, k) && pl_violates(codes[k], szp_pl_classify(field, g, p, k));
    std::vector<long long> viol;
    for (long long k = 0; k < N; k++)
        if (flag[(size_t)k]) viol.push_back(k);

    /* mark: 1 = value stored exactly. flag is reused to dedupe re-checks. */
    std::vector<unsigned char> mark((size_t)N, 0);
    std::fill(flag.begin(), flag.end(), 0);
    std::vector<long long> fresh, check;
    while (!viol.empty()) {
        fresh.clear();
        for (long long k : viol) {
            if (!mark[(size_t)k]) { mark[(size_t)k] = 1; fresh.push_back(k); continue; }
            /* The point itself is exact and still wrong: store the neighbours
               whose order relative to it differs from the original (ties by
               index). With every relation as in the original, the point has
               its original type; such a neighbour is never exact already. */
            for (int a = 0; a < p.nb; a++) {
                long long q = k + p.off[a];
                if (mark[(size_t)q]) continue;
                bool lo_now = field[q] < field[k] || (field[q] == field[k] && q < k);
                bool lo_orig = orig[q] < orig[k] || (orig[q] == orig[k] && q < k);
                if (lo_now != lo_orig) { mark[(size_t)q] = 1; fresh.push_back(q); }
            }
        }
        for (long long k : fresh) field[k] = orig[k];
        /* Only the changed points and their neighbours can change type. */
        check.clear();
        for (long long k : fresh) {
            if (!flag[(size_t)k]) { flag[(size_t)k] = 1; check.push_back(k); }
            for (int a = 0; a < p.nb; a++) {
                long long q = k + p.off[a];
                if (q < 0 || q >= N) continue;
                if (!flag[(size_t)q]) { flag[(size_t)q] = 1; check.push_back(q); }
            }
        }
        std::sort(check.begin(), check.end());
        std::vector<unsigned char> bad(check.size(), 0);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long i = 0; i < (long long)check.size(); i++) {
            long long k = check[(size_t)i];
            bad[(size_t)i] = szp_is_interior(g, k) && pl_violates(codes[k], szp_pl_classify(field, g, p, k));
        }
        viol.clear();
        for (size_t i = 0; i < check.size(); i++) {
            flag[(size_t)check[i]] = 0;
            if (bad[i]) viol.push_back(check[i]);
        }
    }

    /* Pack: [count][LEB128 deltas of the indices][float values]. */
    size_t m = 0;
    for (long long k = 0; k < N; k++) m += mark[(size_t)k];
    std::vector<unsigned char> out(sizeof(size_t));
    memcpy(out.data(), &m, sizeof(size_t));
    long long prev = -1;
    for (long long k = 0; k < N; k++)
        if (mark[(size_t)k]) {
            unsigned long long d = (unsigned long long)(k - prev - 1);
            prev = k;
            do {
                unsigned char b = (unsigned char)(d & 0x7f);
                d >>= 7;
                out.push_back(d ? (unsigned char)(b | 0x80) : b);
            } while (d);
        }
    const size_t vals_at = out.size();
    out.resize(vals_at + m * sizeof(float));
    size_t j = 0;
    for (long long k = 0; k < N; k++)
        if (mark[(size_t)k]) memcpy(out.data() + vals_at + sizeof(float) * j++, &orig[k], sizeof(float));

    unsigned char *buf = (unsigned char *)malloc(out.size());
    if (!buf) { *outSize = 0; if (count) *count = 0; return NULL; }
    memcpy(buf, out.data(), out.size());
    *outSize = out.size();
    if (count) *count = m;
    return buf;
}

extern "C" long long szp_pl_apply_exceptions_float(float *field, size_t n, const unsigned char *buf, size_t size)
{
    if (!buf || size < sizeof(size_t)) return -1;
    size_t m = 0;
    memcpy(&m, buf, sizeof(size_t));
    if (m > n) return -1;
    /* Decode the indices first (the values follow them). */
    std::vector<size_t> idx(m);
    size_t pos = sizeof(size_t);
    long long prev = -1;
    for (size_t i = 0; i < m; i++) {
        unsigned long long d = 0;
        int shift = 0;
        for (;;) {
            if (pos >= size || shift > 63) return -1;
            unsigned char b = buf[pos++];
            d |= (unsigned long long)(b & 0x7f) << shift;
            shift += 7;
            if (!(b & 0x80)) break;
        }
        long long k = prev + 1 + (long long)d;
        if (k < 0 || (size_t)k >= n) return -1;
        idx[i] = (size_t)k;
        prev = k;
    }
    if (size - pos != m * sizeof(float)) return -1;
    for (size_t i = 0; i < m; i++) memcpy(&field[idx[i]], buf + pos + sizeof(float) * i, sizeof(float));
    return (long long)m;
}
