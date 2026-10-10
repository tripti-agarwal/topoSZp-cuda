/**
 *  @file szp_topo_restore.h
 *  @brief Topology restoration after decompression, shared by the OpenMP
 *         (CPU) and CUDA (GPU) paths, 2D and 3D.
 *
 *  Inputs: the decompressed field, the stored type of every point
 *  (0 regular, 1 maximum, 2 minimum, 3 saddle) and the stored rank of every
 *  extremum (its order by original value among critical points that share a
 *  quantization bin, listed in raster order of the interior).
 *
 *  Steps, in this order:
 *   1. Extrema (host, OpenMP). For every stored maximum/minimum with
 *      dequantized value d and rank r:
 *          maxima:  D(p) = d + (r + 1) * eta
 *          minima:  D(p) = d - (R + 1 - r) * eta
 *          eta = 0.5 * eb / (R + 1),  R = largest rank among same-type
 *                                         extrema that share the bin
 *      so extrema become strict and keep the order of their original values;
 *      if a neighbour is still in the way the value is pushed just past it.
 *      An extremum that is alone in its bin and already strict is untouched.
 *      This step only creates orderings that hold in the original data, so it
 *      cannot create false positives or false types.
 *   2. Saddle stencil, 3. Saddle RBF (CPU or GPU). For every stored saddle
 *      that is not a saddle yet: the stencil tries value windows that make
 *      it a saddle; the RBF step tries a Gaussian-weighted average of the
 *      neighbourhood (extremum neighbourhoods down-weighted), backing off
 *      towards the current value. A change is kept only if the point becomes
 *      a saddle and no neighbour gets a critical type other than its stored
 *      one, loses its stored type, or stops being below a stored maximum /
 *      above a stored minimum next to it. Both passes run twice.
 *
 *  Every changed value stays within eb of its dequantized value, so with the
 *  base SZp bound |original - dequantized| <= eb the error is at most 2 eb.
 *
 *  Determinism: the saddle passes visit points in 3^dim colour classes
 *  ((coordinate - 1) mod 3 per axis). Two points of one class are at least 3
 *  apart, farther than anything one update reads (distance 2), so a class is
 *  processed fully in parallel and the result is the same for any thread
 *  count and identical on CPU and GPU (all floating-point operations are
 *  explicitly rounded on the device, see szp_quant.h).
 *
 *  Critical points use strict comparisons with the face neighbours
 *  (4 in 2D, 6 in 3D); a saddle has at least one axis where the point is
 *  above both neighbours and one where it is below both.
 */
#ifndef _SZP_TOPO_RESTORE_H
#define _SZP_TOPO_RESTORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Counts reported by the restoration (all optional to read). */
typedef struct {
    size_t extrema;              /* stored maxima + minima */
    size_t extrema_lost_before;  /* stored extrema that were not extrema after dequantization */
    size_t extrema_lost_after;   /* ... after restoration */
    size_t saddles;              /* stored saddles */
    size_t saddles_lost_before;  /* stored saddles that were not saddles after the extrema step */
    size_t saddles_by_stencil;   /* restored by the stencil passes */
    size_t saddles_by_rbf;       /* restored by the RBF passes */
    size_t saddles_lost_after;   /* still not saddles at the end */
    /* PL mode only (szp_pl_topology.h); 0 in axis mode, where the
       restoration guarantees no false positives and no false types. */
    size_t fp_before, fp_after;  /* points stored regular that are critical */
    size_t ft_before, ft_after;  /* critical points with another code than stored */
    size_t other_fixes;          /* points other than saddles fixed (FP/FT/extrema) */
    size_t rounds;               /* fix rounds run */
} szp_topo_restore_stats;

/**
 * CPU (OpenMP) restoration of a decompressed 2D field (rows x cols, row-major).
 * types: stored type per point; sort_positions: one rank per stored extremum
 * in raster order (may be NULL: all ranks 0); errBound: the absolute bound
 * used for compression. stats may be NULL.
 */
void szp_topo_restore_float_2d(float *data, const int *types, const int *sort_positions,
                               size_t extrema_count, int rows, int cols, float errBound,
                               szp_topo_restore_stats *stats);

/** Same for a 3D field d1 x d2 x d3 (flat index x*d2*d3 + y*d3 + z). */
void szp_topo_restore_float_3d(float *data, const int *types, const int *sort_positions,
                               size_t extrema_count, int d1, int d2, int d3, float errBound,
                               szp_topo_restore_stats *stats);

#ifdef __cplusplus
}
#endif

/* ===================================================================== */
/*  C++ implementation shared by szp_topo_restore.cc and the CUDA path    */
/* ===================================================================== */
#ifdef __cplusplus

#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include "szp_quant.h"

/* ---------------- grid and point classification (host + device) ------- */

struct SzpGrid {
    int nd;          /* 2 or 3 */
    int n[3];        /* extent per axis (n[2] = 1 in 2D) */
    long long s[3];  /* flat-index stride per axis */
};

static inline SzpGrid szp_grid_2d(int rows, int cols)
{
    SzpGrid g; g.nd = 2; g.n[0] = rows; g.n[1] = cols; g.n[2] = 1;
    g.s[0] = cols; g.s[1] = 1; g.s[2] = 0; return g;
}
static inline SzpGrid szp_grid_3d(int d1, int d2, int d3)
{
    SzpGrid g; g.nd = 3; g.n[0] = d1; g.n[1] = d2; g.n[2] = d3;
    g.s[0] = (long long)d2 * d3; g.s[1] = d3; g.s[2] = 1; return g;
}

/** Interior points (all neighbours exist) are the only ones classified. */
static SZP_QHD int szp_is_interior(const SzpGrid &g, long long k)
{
    for (int a = 0; a < g.nd; a++) {
        long long c = (k / g.s[a]) % g.n[a];
        if (c < 1 || c > g.n[a] - 2) return 0;
    }
    return 1;
}

/** 0 regular, 1 maximum, 2 minimum, 3 saddle (strict comparisons). */
static SZP_QHD int szp_classify(const float *v, const SzpGrid &g, long long k)
{
    const float c = v[k];
    int above_all = 1, below_all = 1, hi_axes = 0, lo_axes = 0;
    for (int a = 0; a < g.nd; a++) {
        float m = v[k - g.s[a]], p = v[k + g.s[a]];
        int gm = c > m, gp = c > p, lm = c < m, lp = c < p;
        above_all &= gm & gp;
        below_all &= lm & lp;
        hi_axes += gm & gp;
        lo_axes += lm & lp;
    }
    if (above_all) return 1;
    if (below_all) return 2;
    return (hi_axes >= 1 && lo_axes >= 1) ? 3 : 0;
}

/* ---------------- saddle update rules (host + device) ----------------- */

/** Types of the 2*nd face neighbours (-1 for boundary points). */
static SZP_QHD void szp_neighbour_types(const float *v, const SzpGrid &g, long long k, int out[6])
{
    for (int a = 0; a < g.nd; a++)
        for (int sgn = 0; sgn < 2; sgn++) {
            long long nk = sgn ? k + g.s[a] : k - g.s[a];
            out[2 * a + sgn] = szp_is_interior(g, nk) ? szp_classify(v, g, nk) : -1;
        }
}

/**
 * With the trial value written at k: 1 if k is a saddle and no neighbour's
 * type changed in a way the stored types forbid.
 */
static SZP_QHD int szp_saddle_change_ok(const float *v, const int *types, const SzpGrid &g,
                                        long long k, const int before[6])
{
    if (szp_classify(v, g, k) != 3) return 0;
    const float c = v[k];
    for (int a = 0; a < g.nd; a++)
        for (int sgn = 0; sgn < 2; sgn++) {
            long long nk = sgn ? k + g.s[a] : k - g.s[a];
            int t0 = types[nk];
            if (t0 == 1 && !(c < v[nk])) return 0;   /* stay below a stored maximum */
            if (t0 == 2 && !(c > v[nk])) return 0;   /* stay above a stored minimum */
            if (before[2 * a + sgn] < 0) continue;   /* boundary: never classified */
            int after = szp_classify(v, g, nk);
            if (after != 0 && after != t0) return 0;                      /* FP or FT */
            if (t0 != 0 && before[2 * a + sgn] == t0 && after != t0) return 0;  /* lost */
        }
    return 1;
}

/** Try value t at k; keep it if allowed. Returns 1 if kept. */
static SZP_QHD int szp_try_value(float *v, const int *types, const SzpGrid &g, long long k,
                                 float t, const int before[6])
{
    const float saved = v[k];
    v[k] = t;
    if (szp_saddle_change_ok(v, types, g, k, before)) return 1;
    v[k] = saved;
    return 0;
}

/**
 * Stencil: for every split of the axes into "high" (the point must be above
 * both neighbours) and "low" (below both), the allowed values form an open
 * window (hi_thr, lo_thr). Candidates inside the window and within the bound
 * are tried in a fixed order. Returns 1 if the point became a saddle.
 */
static SZP_QHD int szp_saddle_stencil_point(float *v, const int *types, const float *orig,
                                            const SzpGrid &g, long long k, float eb)
{
    if (types[k] != 3 || szp_classify(v, g, k) == 3) return 0;
    int before[6];
    szp_neighbour_types(v, g, k, before);
    const float lo_b = szp_fadd(orig[k], -szp_fmul(eb, 0.999f));
    const float hi_b = szp_fadd(orig[k], szp_fmul(eb, 0.999f));
    const float fr[5] = {0.5f, 0.25f, 0.75f, 0.1f, 0.9f};
    const int npat = (1 << g.nd) - 1;
    for (int pat = 1; pat < npat; pat++) {          /* skip all-low (0) and all-high */
        float hi_thr = -INFINITY, lo_thr = INFINITY;
        for (int a = 0; a < g.nd; a++) {
            float m = v[k - g.s[a]], p = v[k + g.s[a]];
            if ((pat >> a) & 1) hi_thr = fmaxf(hi_thr, fmaxf(m, p));
            else                lo_thr = fminf(lo_thr, fminf(m, p));
        }
        /* allowed values: above hi_thr, below lo_thr, within [lo_b, hi_b] */
        float wlo = fmaxf(hi_thr, lo_b), whi = fminf(lo_thr, hi_b);
        if (!(wlo < whi)) continue;
        const float span = szp_fadd(whi, -wlo);
        for (int q = 0; q < 5; q++) {
            float t = szp_fadd(wlo, szp_fmul(span, fr[q]));
            if (!(t > hi_thr && t < lo_thr && t >= lo_b && t <= hi_b)) continue;
            if (szp_try_value(v, types, g, k, t, before)) return 1;
        }
    }
    return 0;
}

/** RBF weights: (2r+1)^nd Gaussian weights with r = 1, sigma = 0.8. */
static inline void szp_rbf_weights(int nd, double w[27])
{
    const double s2 = 0.8 * 0.8;
    int idx = 0;
    for (int a = -1; a <= 1; a++)
        for (int b = -1; b <= 1; b++)
            for (int c = (nd == 3 ? -1 : 0); c <= (nd == 3 ? 1 : 0); c++)
                w[idx++] = exp(-(double)(a * a + b * b + c * c) / (2.0 * s2));
}

/**
 * RBF: weighted average of the 3^nd neighbourhood (points next to a stored
 * extremum count 0.4x), clamped to the bound, then halved towards the
 * current value up to 10 times until the change is allowed.
 */
static SZP_QHD int szp_saddle_rbf_point(float *v, const int *types, const unsigned char *locks,
                                        const float *orig, const SzpGrid &g, long long k,
                                        float eb, const double *w)
{
    if (types[k] != 3 || szp_classify(v, g, k) == 3) return 0;
    double num = 0.0, den = 0.0;
    int idx = 0;
    const long long s2 = g.nd == 3 ? g.s[2] : 0;
    for (int a = -1; a <= 1; a++)
        for (int b = -1; b <= 1; b++)
            for (int c = (g.nd == 3 ? -1 : 0); c <= (g.nd == 3 ? 1 : 0); c++) {
                long long nk = k + a * g.s[0] + b * g.s[1] + c * s2;
                double ww = w[idx++];
                if (locks[nk]) ww = szp_qmul(ww, 0.4);
                num = szp_dadd(num, szp_qmul(ww, (double)v[nk]));
                den = szp_dadd(den, ww);
            }
    if (!(den > 0.0)) return 0;
    int before[6];
    szp_neighbour_types(v, g, k, before);
    const float lo_b = szp_fadd(orig[k], -szp_fmul(eb, 0.999f));
    const float hi_b = szp_fadd(orig[k], szp_fmul(eb, 0.999f));
    float cand = (float)(num / den);
    cand = fminf(fmaxf(cand, lo_b), hi_b);
    const float old = v[k], step = szp_fadd(cand, -old);
    float alpha = 1.0f;
    for (int it = 0; it < 10; it++) {
        float t = szp_fadd(old, szp_fmul(alpha, step));
        t = fminf(fmaxf(t, lo_b), hi_b);
        if (szp_try_value(v, types, g, k, t, before)) return 1;
        alpha = szp_fmul(alpha, 0.5f);
    }
    return 0;
}

/* ---------------- host helpers (shared by the CPU and CUDA drivers) ---- */

/** Colour class of an interior point: ((x-1) mod 3, (y-1) mod 3, (z-1) mod 3). */
static inline int szp_colour(const SzpGrid &g, long long k)
{
    int c = 0;
    for (int a = 0; a < g.nd; a++) c = c * 3 + (int)(((k / g.s[a]) % g.n[a] - 1) % 3);
    return c;
}

/**
 * Stored saddles that are not saddles now, grouped by colour (counting sort,
 * raster order inside each colour). start has 3^nd + 1 entries.
 */
static inline void szp_saddle_candidates(const float *v, const int *types, const SzpGrid &g,
                                         std::vector<long long> &list, std::vector<size_t> &start)
{
    const int ncol = g.nd == 3 ? 27 : 9;
    const long long N = (long long)g.n[0] * g.n[1] * g.n[2];
    std::vector<unsigned char> flag((size_t)N, 0);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long long k = 0; k < N; k++)
        flag[(size_t)k] = types[k] == 3 && szp_is_interior(g, k) && szp_classify(v, g, k) != 3;
    std::vector<long long> all;
    for (long long k = 0; k < N; k++)
        if (flag[(size_t)k]) all.push_back(k);
    start.assign(ncol + 1, 0);
    for (long long k : all) start[szp_colour(g, k) + 1]++;
    for (int c = 0; c < ncol; c++) start[c + 1] += start[c];
    list.resize(all.size());
    std::vector<size_t> pos(start.begin(), start.end() - 1);
    for (long long k : all) list[pos[szp_colour(g, k)]++] = k;
}

/** Points next to a stored extremum (its 3^nd neighbourhood). */
static inline void szp_extremum_locks(const int *types, const SzpGrid &g, std::vector<unsigned char> &locks)
{
    const long long N = (long long)g.n[0] * g.n[1] * g.n[2];
    locks.assign((size_t)N, 0);
    const long long s2 = g.nd == 3 ? g.s[2] : 0;
    /* Gather form (each point looks at its own neighbourhood): no write races. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long long k = 0; k < N; k++) {
        if (!szp_is_interior(g, k)) continue;
        unsigned char hit = 0;
        for (int a = -1; a <= 1 && !hit; a++)
            for (int b = -1; b <= 1 && !hit; b++)
                for (int c = (g.nd == 3 ? -1 : 0); c <= (g.nd == 3 ? 1 : 0); c++) {
                    long long nk = k + a * g.s[0] + b * g.s[1] + c * s2;
                    if ((types[nk] == 1 || types[nk] == 2) && szp_is_interior(g, nk)) { hit = 1; break; }
                }
        locks[(size_t)k] = hit;
    }
}

/* ---------------- extrema step (host, OpenMP) ------------------------- */

struct SzpExtremum {
    size_t idx;    /* flat index */
    float  d;      /* decompressed value before any correction */
    int    type;   /* 1 = maximum, 2 = minimum */
    int    rank;   /* stored sort position (0-based, ties share) */
};

static inline void
szp_restore_extrema_ranked_core(float *data, std::vector<SzpExtremum> &ext,
                                const ptrdiff_t *nb_off, int n_nb, double errBound)
{
    const size_t n = ext.size();
    if (n == 0 || errBound <= 0) return;

    /* Group extrema of the same type that share a decompressed value: sort
       compact (type, value bits, index) keys and scan the runs. Only
       equality of values matters here, so the raw bits are used as the key
       (with -0.0 folded into +0.0). */
    struct GroupKey { unsigned long long key; size_t k; };
    std::vector<GroupKey> keys(n);
    for (size_t k = 0; k < n; k++) {
        float dv = (ext[k].d == 0.0f) ? 0.0f : ext[k].d;
        unsigned int bits;
        memcpy(&bits, &dv, sizeof(bits));
        keys[k].key = ((unsigned long long)(unsigned int)ext[k].type << 32) | bits;
        keys[k].k = k;
    }
    std::sort(keys.begin(), keys.end(), [](const GroupKey &a, const GroupKey &b) {
        return a.key != b.key ? a.key < b.key : a.k < b.k;
    });
    std::vector<int> gmax(n), gsize(n);
    for (size_t s = 0; s < n;) {
        size_t e = s;
        int R = 0;
        while (e < n && keys[e].key == keys[s].key) {
            R = std::max(R, ext[keys[e].k].rank);
            e++;
        }
        for (size_t q = s; q < e; q++) {
            gmax[keys[q].k] = R;
            gsize[keys[q].k] = (int)(e - s);
        }
        s = e;
    }

    std::vector<float> tgt(n);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long long kk = 0; kk < (long long)n; kk++) {
        const size_t k = (size_t)kk;
        const SzpExtremum &e = ext[k];
        const double d = (double)e.d;
        const float c = data[e.idx];

        float m = data[(ptrdiff_t)e.idx + nb_off[0]];
        for (int q = 1; q < n_nb; q++) {
            float v = data[(ptrdiff_t)e.idx + nb_off[q]];
            m = (e.type == 1) ? fmaxf(m, v) : fminf(m, v);
        }

        /* Largest float <= d + eb and smallest float >= d - eb. */
        float hi = (float)(d + errBound);
        if ((double)hi > d + errBound) hi = nextafterf(hi, -INFINITY);
        float lo = (float)(d - errBound);
        if ((double)lo < d - errBound) lo = nextafterf(lo, INFINITY);

        const double eta = 0.5 * errBound / ((double)gmax[k] + 1.0);
        float t = c;
        if (e.type == 1) {
            if (!(gsize[k] == 1 && c > m)) {
                t = (float)(d + ((double)e.rank + 1.0) * eta);
                if (!(t > m)) t = nextafterf(m, INFINITY);
                t = fminf(fmaxf(t, lo), hi);
            }
        } else {
            if (!(gsize[k] == 1 && c < m)) {
                t = (float)(d - ((double)gmax[k] + 1.0 - (double)e.rank) * eta);
                if (!(t < m)) t = nextafterf(m, -INFINITY);
                t = fminf(fmaxf(t, lo), hi);
            }
        }
        tgt[k] = t;
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long long kk = 0; kk < (long long)n; kk++)
        data[ext[(size_t)kk].idx] = tgt[(size_t)kk];
}

/**
 * Extrema step on any grid. orig: field right after dequantization;
 * sort_positions: one rank per stored extremum in raster order.
 */
static inline void szp_restore_extrema_ranked(float *data, const float *orig, const int *types,
                                              const int *sort_positions, size_t extrema_count,
                                              const SzpGrid &g, float errBound)
{
    const long long N = (long long)g.n[0] * g.n[1] * g.n[2];
    std::vector<SzpExtremum> ext;
    ext.reserve(extrema_count ? extrema_count : 1024);
    size_t r = 0;
    for (long long k = 0; k < N; k++) {
        int t = types[k];
        if ((t != 1 && t != 2) || !szp_is_interior(g, k)) continue;
        int rank = (sort_positions && r < extrema_count) ? sort_positions[r] : 0;
        r++;
        ext.push_back({(size_t)k, orig[k], t, rank < 0 ? 0 : rank});
    }
    ptrdiff_t off[6];
    for (int a = 0; a < g.nd; a++) { off[2 * a] = -(ptrdiff_t)g.s[a]; off[2 * a + 1] = (ptrdiff_t)g.s[a]; }
    szp_restore_extrema_ranked_core(data, ext, off, 2 * g.nd, (double)errBound);
}

/* Kept for existing callers. */
static inline void
szp_restore_extrema_ranked_2d(float *data, const float *orig_decomp, const int *types,
                              const int *sort_positions, size_t extrema_count,
                              int rows, int cols, float errBound)
{
    if (!data || !orig_decomp || !types || rows < 3 || cols < 3) return;
    szp_restore_extrema_ranked(data, orig_decomp, types, sort_positions, extrema_count,
                               szp_grid_2d(rows, cols), errBound);
}
static inline void
szp_restore_extrema_ranked_3d(float *data, const float *orig_decomp, const int *types,
                              const int *sort_positions, size_t extrema_count,
                              int d1, int d2, int d3, float errBound)
{
    if (!data || !orig_decomp || !types || d1 < 3 || d2 < 3 || d3 < 3) return;
    szp_restore_extrema_ranked(data, orig_decomp, types, sort_positions, extrema_count,
                               szp_grid_3d(d1, d2, d3), errBound);
}

/** Count stored points of each kind that do not have their stored type. */
static inline void szp_count_lost(const float *v, const int *types, const SzpGrid &g,
                                  size_t *ext_total, size_t *ext_lost, size_t *sad_total, size_t *sad_lost)
{
    const long long N = (long long)g.n[0] * g.n[1] * g.n[2];
    size_t et = 0, el = 0, st = 0, sl = 0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:et,el,st,sl) schedule(static)
#endif
    for (long long k = 0; k < N; k++) {
        int t = types[k];
        if (t == 0 || !szp_is_interior(g, k)) continue;
        int now = szp_classify(v, g, k);
        if (t == 3) { st++; sl += (now != 3); }
        else        { et++; el += (now != t); }
    }
    if (ext_total) *ext_total = et;
    if (ext_lost) *ext_lost = el;
    if (sad_total) *sad_total = st;
    if (sad_lost) *sad_lost = sl;
}

#endif /* __cplusplus */
#endif /* _SZP_TOPO_RESTORE_H */
