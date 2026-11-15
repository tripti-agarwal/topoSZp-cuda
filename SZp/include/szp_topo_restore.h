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
 *   1. Extrema (CPU: OpenMP; GPU: on the device). For every stored maximum/minimum with
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
 *      above a stored minimum next to it.
 *   4. Saddle neighbourhood move. Most saddles that are still lost cannot be
 *      restored by changing the saddle point alone: quantization tied a
 *      neighbour it must be above with a neighbour it must be below. For
 *      every such saddle and every assignment of its axes to "high" (above
 *      both neighbours), "low" (below both) or "free" (at least one high and
 *      one low), a value t for the point is chosen and the high neighbours
 *      that are not below t are moved just below it, the low neighbours that
 *      are not above t just above it; only points stored as regular or as
 *      saddles are moved, each within eb of its dequantized value. The change
 *      is kept only if the point becomes a saddle and no point within
 *      distance 2 gets a critical type other than its stored one or loses
 *      its stored type. This pass runs up to twice.
 *
 *  Every changed value stays within eb of its dequantized value, so with the
 *  base SZp bound |original - dequantized| <= eb the error is at most 2 eb.
 *
 *  Determinism: the saddle passes visit points in 3^dim colour classes
 *  ((coordinate - 1) mod 3 per axis). Two points of one class are at least 3
 *  apart, farther than anything one update reads (distance 2), so a class is
 *  processed fully in parallel and the result is the same for any thread
 *  count and identical on CPU and GPU (all floating-point operations are
 *  explicitly rounded on the device, see szp_quant.h). The neighbourhood
 *  move writes up to distance 1 and reads up to distance 3, so it uses 5^dim
 *  colour classes ((coordinate - 1) mod 5 per axis).
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
    size_t saddles_by_neighbours;/* restored by the neighbourhood-move passes */
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
#ifdef _OPENMP
#include <omp.h>
#endif
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

/** Appends to chk the interior points at Manhattan distance `ring` (1 or 2)
    from k (coordinates co) and whether each is a stored critical point that
    is present in v; returns the new count. */
static SZP_QHD int szp_nbr_ring(const float *v, const int *types, const SzpGrid &g, long long k,
                                const int *co, int ring, long long *chk, int *chk_keep, int nchk)
{
    const int nd = g.nd;
    for (int a = 0; a < nd; a++)
        for (int b = (ring == 1 ? a : 0); b <= a; b++)
            for (int sa = -1; sa <= 1; sa += 2)
                for (int sb = -1; sb <= (ring == 2 && b < a ? 1 : -1); sb += 2) {
                    /* ring 1: sa e_a; ring 2: 2 sa e_a (b == a) or sa e_a + sb e_b (b < a) */
                    int d[3] = {0, 0, 0};
                    if (ring == 1) d[a] = sa;
                    else if (b == a) d[a] = 2 * sa;
                    else { d[a] = sa; d[b] = sb; }
                    long long q = k;
                    int inside = 1;
                    for (int x = 0; x < nd; x++) {
                        const int c = co[x] + d[x];
                        inside &= c >= 1 && c <= g.n[x] - 2;     /* interior only */
                        q += d[x] * g.s[x];
                    }
                    if (!inside) continue;
                    chk[nchk] = q;
                    chk_keep[nchk++] = types[q] != 0 && szp_classify(v, g, q) == types[q];
                }
    return nchk;
}

/**
 * Neighbourhood move (step 4): make stored saddle k a saddle by choosing its
 * value and moving face neighbours that are in the way. Returns 1 if kept.
 * Reads values up to distance 3 from k, writes k and its face neighbours.
 */
static SZP_QHD int szp_saddle_nbr_point(float *v, const int *types, const float *orig,
                                        const SzpGrid &g, long long k, float eb)
{
    if (types[k] != 3 || szp_classify(v, g, k) == 3) return 0;
    const int nd = g.nd;
    int co[3] = {0, 0, 0};
    for (int a = 0; a < nd; a++) co[a] = (int)((k / g.s[a]) % g.n[a]);

    /* Points whose type can change: Manhattan distance <= 2 from k. The face
       neighbours (ring 1) are listed now; most failed trials fail there, so
       ring 2 is listed only when a trial gets that far (nchk1 = end of ring 1).
       chk_keep: the point is a stored critical point that is present now. */
    long long chk[25];
    int chk_keep[25], nchk = 0, nchk1, ring2 = 0;
    nchk = szp_nbr_ring(v, types, g, k, co, 1, chk, chk_keep, 0);
    nchk1 = nchk;

    const float W = szp_fmul(eb, 0.999f), delta = szp_fmul(eb, 0.015625f);
    const float p_lo = szp_fadd(orig[k], -W), p_hi = szp_fadd(orig[k], W);
    const int npat = nd == 3 ? 27 : 9;
    for (int pc = 0; pc < npat; pc++) {
        int role[3] = {0, 0, 0}, nh = 0, nl = 0;     /* per axis: 0 free, 1 high, 2 low */
        { int x = pc; for (int a = 0; a < nd; a++) { role[a] = x % 3; x /= 3; nh += role[a] == 1; nl += role[a] == 2; } }
        if (!nh || !nl) continue;

        /* values t reachable for k: above what every high neighbour can go
           below, below what every low neighbour can go above */
        float tlo = p_lo, thi = p_hi;
        for (int a = 0; a < nd; a++) {
            if (role[a] == 0) continue;
            for (int sgn = 0; sgn < 2; sgn++) {
                long long n = sgn ? k + g.s[a] : k - g.s[a];
                int movable = types[n] == 0 || types[n] == 3;
                if (role[a] == 1) tlo = fmaxf(tlo, movable ? fminf(v[n], szp_fadd(orig[n], -W)) : v[n]);
                else              thi = fminf(thi, movable ? fmaxf(v[n], szp_fadd(orig[n], W)) : v[n]);
            }
        }
        if (!(tlo < thi)) continue;
        const float span = szp_fadd(thi, -tlo);
        float cand[3];
        cand[0] = fminf(fmaxf(v[k], szp_fadd(tlo, szp_fmul(0.25f, span))), szp_fadd(tlo, szp_fmul(0.75f, span)));
        cand[1] = szp_fadd(tlo, szp_fmul(0.5f, span));
        cand[2] = orig[k];
        for (int ci = 0; ci < 3; ci++) {
            const float t = cand[ci];
            if (!(t > tlo && t < thi)) continue;
            if ((ci >= 1 && t == cand[0]) || (ci == 2 && t == cand[1])) continue;   /* tried already */
            long long idx[7];
            float saved[7];
            int ns = 0, feasible = 1;
            idx[ns] = k; saved[ns++] = v[k]; v[k] = t;
            for (int a = 0; a < nd; a++) {
                if (role[a] == 0) continue;
                for (int sgn = 0; sgn < 2; sgn++) {
                    long long n = sgn ? k + g.s[a] : k - g.s[a];
                    const int movable = types[n] == 0 || types[n] == 3;
                    float nv;
                    if (role[a] == 1) {
                        if (v[n] < t) continue;
                        const float lim = szp_fadd(orig[n], -W);
                        nv = fmaxf(szp_fadd(t, -delta), lim);
                        if (!(nv < t)) nv = nextafterf(t, -INFINITY);
                        if (!movable || nv < lim) feasible = 0;
                    } else {
                        if (v[n] > t) continue;
                        const float lim = szp_fadd(orig[n], W);
                        nv = fminf(szp_fadd(t, delta), lim);
                        if (!(nv > t)) nv = nextafterf(t, INFINITY);
                        if (!movable || nv > lim) feasible = 0;
                    }
                    idx[ns] = n; saved[ns++] = v[n]; v[n] = nv;
                }
            }
            int ok = feasible && szp_classify(v, g, k) == 3;
            for (int i = 0; ok && i < nchk; i++) {
                const int t0 = types[chk[i]], after = szp_classify(v, g, chk[i]);
                if (after != 0 && after != t0) ok = 0;                      /* FP or FT */
                else if (chk_keep[i] && after != t0) ok = 0;                /* lost */
                if (ok && i == nchk1 - 1 && !ring2) {
                    /* ring 2 describes the field without this trial: undo it, list, redo */
                    for (int j = 0; j < ns; j++) { const float x = v[idx[j]]; v[idx[j]] = saved[j]; saved[j] = x; }
                    nchk = szp_nbr_ring(v, types, g, k, co, 2, chk, chk_keep, nchk);
                    for (int j = 0; j < ns; j++) { const float x = v[idx[j]]; v[idx[j]] = saved[j]; saved[j] = x; }
                    ring2 = 1;
                }
            }
            if (ok) return 1;
            for (int i = ns - 1; i >= 0; i--) v[idx[i]] = saved[i];
        }
    }
    return 0;
}

/* ---------------- host helpers (shared by the CPU and CUDA drivers) ---- */

/** Colour class of an interior point: ((x-1) mod 3, (y-1) mod 3, (z-1) mod 3). */
static SZP_QHD int szp_colour_mod(const SzpGrid &g, long long k, int mod)
{
    int c = 0;
    for (int a = 0; a < g.nd; a++) c = c * mod + (int)(((k / g.s[a]) % g.n[a] - 1) % mod);
    return c;
}
static SZP_QHD int szp_colour(const SzpGrid &g, long long k) { return szp_colour_mod(g, k, 3); }
/** Number of colour classes for a modulus (3 for the single-point passes, 5 for the neighbourhood move). */
static SZP_QHD int szp_ncolours(const SzpGrid &g, int mod) { return g.nd == 3 ? mod * mod * mod : mod * mod; }

/**
 * Stored saddles that are not saddles now, grouped by colour (counting sort,
 * raster order inside each colour). start has mod^nd + 1 entries.
 */
static inline void szp_saddle_candidates(const float *v, const int *types, const SzpGrid &g,
                                         std::vector<long long> &list, std::vector<size_t> &start,
                                         int mod = 3)
{
    const int ncol = szp_ncolours(g, mod);
    const long long N = (long long)g.n[0] * g.n[1] * g.n[2];
    /* Per-thread lists over contiguous index ranges; inside a colour the
       points keep index order (thread 0's first), whatever the thread count. */
#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif
    std::vector<std::vector<long long> > part((size_t)nt);
    std::vector<std::vector<unsigned char> > pcol((size_t)nt);
    std::vector<size_t> cnt((size_t)nt * ncol, 0);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static, 1) num_threads(nt)
#endif
    for (int t = 0; t < nt; t++) {
        const long long lo = N * t / nt, hi = N * (t + 1) / nt;
        std::vector<long long> &mine = part[(size_t)t];
        std::vector<unsigned char> &col = pcol[(size_t)t];
        size_t *c = &cnt[(size_t)t * ncol];
        for (long long k = lo; k < hi; k++)
            if (types[k] == 3 && szp_is_interior(g, k) && szp_classify(v, g, k) != 3) {
                const int cl = szp_colour_mod(g, k, mod);
                mine.push_back(k); col.push_back((unsigned char)cl); c[cl]++;
            }
    }
    start.assign(ncol + 1, 0);
    for (int t = 0; t < nt; t++)
        for (int c = 0; c < ncol; c++) start[c + 1] += cnt[(size_t)t * ncol + c];
    for (int c = 0; c < ncol; c++) start[c + 1] += start[c];
    list.resize(start[ncol]);
    /* cnt becomes the write offset of thread t in colour c */
    for (int c = 0; c < ncol; c++) {
        size_t off = start[c];
        for (int t = 0; t < nt; t++) { size_t n = cnt[(size_t)t * ncol + c]; cnt[(size_t)t * ncol + c] = off; off += n; }
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static, 1) num_threads(nt)
#endif
    for (int t = 0; t < nt; t++) {
        size_t *pos = &cnt[(size_t)t * ncol];
        const std::vector<long long> &mine = part[(size_t)t];
        const std::vector<unsigned char> &col = pcol[(size_t)t];
        for (size_t i = 0; i < mine.size(); i++) list[pos[col[i]]++] = mine[i];
    }
}

/** Whether the neighbourhood move runs a second time: the first pass
    restored `got` of `candidates` (same rule on host and device). */
#define SZP_NBR_REPEAT_DIV 8
static inline int szp_nbr_repeat(unsigned long long got, unsigned long long candidates)
{
    return got > 0 && got * SZP_NBR_REPEAT_DIV >= candidates;
}

/** 1 if interior point k has a stored extremum in its 3^nd neighbourhood. */
static SZP_QHD unsigned char szp_extremum_lock_at(const int *types, const SzpGrid &g, long long k)
{
    if (!szp_is_interior(g, k)) return 0;
    const long long s2 = g.nd == 3 ? g.s[2] : 0;
    for (int a = -1; a <= 1; a++)
        for (int b = -1; b <= 1; b++)
            for (int c = (g.nd == 3 ? -1 : 0); c <= (g.nd == 3 ? 1 : 0); c++) {
                long long nk = k + a * g.s[0] + b * g.s[1] + c * s2;
                if ((types[nk] == 1 || types[nk] == 2) && szp_is_interior(g, nk)) return 1;
            }
    return 0;
}

/** Marks the interior points of the 3^nd neighbourhood of stored extremum k
    (locks must be zero before). Several extrema may write the same byte; all
    write 1. */
static SZP_QHD void szp_extremum_lock_from(const int *types, const SzpGrid &g, long long k, unsigned char *locks)
{
    if ((types[k] != 1 && types[k] != 2) || !szp_is_interior(g, k)) return;
    const long long s2 = g.nd == 3 ? g.s[2] : 0;
    for (int a = -1; a <= 1; a++)
        for (int b = -1; b <= 1; b++)
            for (int c = (g.nd == 3 ? -1 : 0); c <= (g.nd == 3 ? 1 : 0); c++) {
                long long nk = k + a * g.s[0] + b * g.s[1] + c * s2;
                if (!szp_is_interior(g, nk)) continue;
#if defined(_OPENMP) && !defined(__CUDACC__)
                #pragma omp atomic write
#endif
                locks[nk] = 1;
            }
}

/** Points next to a stored extremum (its 3^nd neighbourhood). n_extrema:
    number of stored extrema, only used to choose the cheaper of two forms
    with the same result (each extremum marks its neighbourhood, or each
    point looks for an extremum). */
static inline void szp_extremum_locks(const int *types, const SzpGrid &g, std::vector<unsigned char> &locks,
                                      size_t n_extrema = 0)
{
    const long long N = (long long)g.n[0] * g.n[1] * g.n[2];
    locks.assign((size_t)N, 0);
    unsigned char *L = locks.data();
    const bool gather = n_extrema > (size_t)N / 16;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long long k = 0; k < N; k++) {
        if (gather) L[k] = szp_extremum_lock_at(types, g, k);
        else        szp_extremum_lock_from(types, g, k, L);
    }
}

/* ---------------- extrema step (host, OpenMP) ------------------------- */

struct SzpExtremum {
    size_t idx;    /* flat index */
    float  d;      /* decompressed value before any correction */
    int    type;   /* 1 = maximum, 2 = minimum */
    int    rank;   /* stored sort position (0-based, ties share) */
};

/**
 * New value of one extremum (see step 1 above). data: current field (read
 * only); dval: its decompressed value; gmax / gsize: largest rank and size of
 * its group (same type, same decompressed value). Every operation is rounded
 * separately (szp_quant.h), so host and device give the same float.
 */
static SZP_QHD float szp_extremum_target(const float *data, ptrdiff_t idx, float dval, int type, int rank,
                                         int gmax, int gsize, const ptrdiff_t *nb_off, int n_nb,
                                         double errBound)
{
    const double d = (double)dval;
    const float c = data[idx];

    float m = data[idx + nb_off[0]];
    for (int q = 1; q < n_nb; q++) {
        float v = data[idx + nb_off[q]];
        m = (type == 1) ? fmaxf(m, v) : fminf(m, v);
    }

    /* Largest float <= d + eb and smallest float >= d - eb. */
    const double dhi = szp_dadd(d, errBound), dlo = szp_dadd(d, -errBound);
    float hi = (float)dhi;
    if ((double)hi > dhi) hi = nextafterf(hi, -INFINITY);
    float lo = (float)dlo;
    if ((double)lo < dlo) lo = nextafterf(lo, INFINITY);

    const double eta = szp_qmul(0.5, errBound) / szp_dadd((double)gmax, 1.0);
    float t = c;
    if (type == 1) {
        if (!(gsize == 1 && c > m)) {
            t = (float)szp_dadd(d, szp_qmul(szp_dadd((double)rank, 1.0), eta));
            if (!(t > m)) t = nextafterf(m, INFINITY);
            t = fminf(fmaxf(t, lo), hi);
        }
    } else {
        if (!(gsize == 1 && c < m)) {
            t = (float)szp_dadd(d, -szp_qmul(szp_dadd(szp_dadd((double)gmax, 1.0), -(double)rank), eta));
            if (!(t < m)) t = nextafterf(m, -INFINITY);
            t = fminf(fmaxf(t, lo), hi);
        }
    }
    return t;
}

/** Group key of an extremum: type and the bits of its decompressed value (-0 as +0). */
static SZP_QHD unsigned long long szp_extremum_key(int type, float d)
{
    float dv = (d == 0.0f) ? 0.0f : d;
    unsigned int bits;
#if defined(__CUDA_ARCH__)
    bits = __float_as_uint(dv);
#else
    memcpy(&bits, &dv, sizeof(bits));
#endif
    return ((unsigned long long)(unsigned int)type << 32) | bits;
}

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
        keys[k].key = szp_extremum_key(ext[k].type, ext[k].d);
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
        tgt[k] = szp_extremum_target(data, (ptrdiff_t)e.idx, e.d, e.type, e.rank, gmax[k], gsize[k],
                                     nb_off, n_nb, errBound);
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
