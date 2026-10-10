/**
 *  @file szp_pl_topology.h
 *  @brief Critical points with the standard piecewise-linear (PL) definition
 *         and the matching topology restoration, shared by the CPU (OpenMP)
 *         and GPU (CUDA) paths, 2D and 3D. Selected with the "PL" mode of
 *         TopoSZp; the default "axis" mode is in szp_topo_restore.h.
 *
 *  Definition (as in TTK's ScalarFieldCriticalPoints and example/pl_topo_check):
 *   - Freudenthal triangulation of the grid: each point has 6 (2D) or 14 (3D)
 *     neighbours, the axis neighbours plus the diagonals along (+,+[,+]) and
 *     (-,-[,-]); two neighbours are joined in the link when their offsets
 *     differ by a vector with all components in {0,1} or all in {0,-1}.
 *   - Ties are broken by flat index: q is below k iff v[q] < v[k], or
 *     v[q] == v[k] and q < k.
 *   - Lower link empty: minimum; upper link empty: maximum; one lower and one
 *     upper connected piece: regular; otherwise a saddle. In 3D: >= 2 lower
 *     pieces and 1 upper piece = 1-saddle, 1 lower and >= 2 upper = 2-saddle,
 *     >= 2 of both = degenerate saddle. Only interior points are classified.
 *  Codes: 0 regular, 1 maximum, 2 minimum, 3 saddle (2D) / 1-saddle (3D),
 *         4 2-saddle, 5 degenerate saddle. The compressed stream keeps the
 *         2-bit type (codes >= 3 stored as 3); the 3D saddle kind is stored
 *         separately (szp_pl_pack_saddle_kinds).
 *
 *  Restoration (szp_topo_restore_float_pl, szp_cuda_topo_restore_float_pl):
 *   1. Extrema: the ranked step of szp_topo_restore.h, over the 6/14 PL
 *      neighbours, so stored extrema become strict PL extrema in the order of
 *      their original values.
 *   2. Fix passes: every interior point whose PL code differs from its stored
 *      code (lost saddles, false critical points from quantization ties, wrong
 *      types) gets an exhaustive single-point search. The code of a point
 *      depends only on which gap between its sorted neighbour values its
 *      value falls in, so one value per gap (within the bound) covers every
 *      possible outcome; values are tried nearest first. A value is kept when
 *      the point gets its stored code and no neighbour that had its stored
 *      code loses it. Passes repeat (colour classes as in szp_topo_restore.h)
 *      until nothing changes or 8 rounds.
 *   3. Exceptions (guarantee): the decoder is deterministic (same result on
 *      CPU and GPU, any thread count), so the compressor runs the same
 *      decoding and restoration and finds every point that would end up with
 *      a false positive, a false type, or a lost extremum. It stores the
 *      original value of such a point; if the point is still wrong, it also
 *      stores the neighbours whose order relative to it differs from the
 *      original (a point whose relations to all its neighbours are as in the
 *      original has exactly its original type). This
 *      repeats until no point is wrong, and the decoder applies the stored
 *      values last (szp_pl_apply_exceptions_float).
 *  Result: no false positives, no false types, all extrema kept; saddles may
 *  be lost (FN). Every value stays within 2 eb (exceptions are exact).
 */
#ifndef _SZP_PL_TOPOLOGY_H
#define _SZP_PL_TOPOLOGY_H

#include <stddef.h>
#include "szp_topo_restore.h"
#include "szp_float.h"        /* CriticalPoint */
#include "szp_topology_3d.h"  /* CriticalPoint3D */

#define SZP_PL_ROUNDS 8   /* maximum fix rounds */

/* File container of the example programs. v1 (axis mode, unchanged):
     [topology][sort positions][topology size][sort size]
   v3 (written in PL mode):
     [topology][sort positions][saddle kinds][exceptions][topology size]
     [sort size][kinds size][exceptions size][flags][magic]
                                         (sizes, flags and magic are size_t)
   flags bit 0 = PL critical points. (v2 = v3 without the exceptions chunk
   and its size; still read.) */
#define SZP_CONTAINER_MAGIC_V2 0x325654504F545A53ULL   /* "SZTOPTV2" */
#define SZP_CONTAINER_MAGIC_V3 0x335654504F545A53ULL   /* "SZTOPTV3" */
#define SZP_CONTAINER_FLAG_PL  1ULL

enum { SZP_PL_REG = 0, SZP_PL_MAX = 1, SZP_PL_MIN = 2, SZP_PL_SAD1 = 3, SZP_PL_SAD2 = 4, SZP_PL_SADX = 5 };

#ifdef __cplusplus
extern "C" {
#endif

/** PL code of every point (boundary points get 0). nd = 2 (d3 ignored) or 3. OpenMP. */
void szp_pl_classify_float(const float *data, int nd, int d1, int d2, int d3, signed char *codes);

/**
 * Pack the kind of every stored saddle (codes >= 3, raster order) in 2 bits:
 * [count (size_t)][bits]. Returns a malloc'ed buffer, *outSize its size.
 */
unsigned char *szp_pl_pack_saddle_kinds(const signed char *codes, size_t n, size_t *outSize);

/**
 * Turn the 2-bit types of a decoded stream (3 = saddle) into PL codes using
 * the packed saddle kinds. Returns 0, or -1 if the counts do not match
 * (types are then left with 3 for every saddle).
 */
int szp_pl_apply_saddle_kinds(int *types, size_t n, const unsigned char *buf, size_t size);

/**
 * Critical-point lists (raster order, interior points) for the TopoSZp
 * compressors from PL codes: type = code clamped to 3, quantized_bin from
 * szp_quantize, sort_position 0 (rank them with the usual sort functions).
 * Return malloc'ed arrays (NULL if none), *count their length.
 */
CriticalPoint *szp_pl_critical_points_2d(const signed char *codes, const float *data,
                                         int rows, int cols, float errBound, size_t *count);
CriticalPoint3D *szp_pl_critical_points_3d(const signed char *codes, const float *data,
                                           int d1, int d2, int d3, float errBound, size_t *count);

/**
 * Compressor side of the guarantee. orig: original data; field: the decoder's
 * output before exceptions (szp_topo_restore_float_pl on the decoded stream),
 * updated in place to the final output; codes: stored PL codes (0..5).
 * Returns the packed exception list (malloc'ed; [count (size_t)][LEB128 index
 * deltas][float values]), *outSize its size, *count the number of points.
 */
unsigned char *szp_pl_exceptions_float(const float *orig, float *field, const int *codes,
                                       int nd, int d1, int d2, int d3,
                                       size_t *outSize, size_t *count);

/** Decoder side: write the stored values. Returns the number of points, or -1 if malformed. */
long long szp_pl_apply_exceptions_float(float *field, size_t n, const unsigned char *buf, size_t size);

/**
 * Points of field whose PL code violates the guarantee against the stored
 * codes: false positives, false types, lost extrema (lost saddles are not
 * counted). Each pointer may be NULL.
 */
void szp_pl_violations(const float *field, const int *codes, int nd, int d1, int d2, int d3,
                       size_t *fp, size_t *ft, size_t *ext_lost, size_t *sad_lost);

/**
 * CPU restoration in PL mode. codes: stored PL code per point (0..5);
 * sort_positions: ranks of the stored extrema in raster order (may be NULL).
 * nd = 2: grid d1 x d2 (d3 ignored); nd = 3: d1 x d2 x d3.
 */
void szp_topo_restore_float_pl(float *data, const int *codes, const int *sort_positions,
                               size_t extrema_count, int nd, int d1, int d2, int d3,
                               float errBound, szp_topo_restore_stats *stats);

#ifdef __cplusplus
}
#endif

/* ===================================================================== */
#ifdef __cplusplus
#include <math.h>
#include <vector>

/** Neighbour offsets and link adjacency of the Freudenthal triangulation. */
struct SzpPL {
    int nb;                 /* 6 (2D) or 14 (3D) */
    long long off[14];      /* flat-index offsets */
    signed char d[14][3];   /* offset vectors */
    unsigned short adj[14]; /* link neighbours of neighbour a (bit mask) */
};

static SZP_QHD void szp_pl_init(const SzpGrid &g, SzpPL &p)
{
    const int nm = g.nd == 3 ? 7 : 3;
    int n = 0;
    for (int sgn = 0; sgn < 2; sgn++)
        for (int m = 1; m <= nm; m++) {
            long long o = 0;
            for (int a = 0; a < 3; a++) {
                int c = (a < g.nd && ((m >> a) & 1)) ? (sgn ? -1 : 1) : 0;
                p.d[n][a] = (signed char)c;
                if (a < g.nd) o += (long long)c * g.s[a];
            }
            p.off[n++] = o;
        }
    p.nb = n;
    for (int a = 0; a < n; a++) {
        unsigned short m = 0;
        for (int b = 0; b < n; b++) {
            if (a == b) continue;
            int e0 = p.d[a][0] - p.d[b][0], e1 = p.d[a][1] - p.d[b][1], e2 = p.d[a][2] - p.d[b][2];
            int pos = e0 >= 0 && e1 >= 0 && e2 >= 0 && e0 <= 1 && e1 <= 1 && e2 <= 1;
            int neg = e0 <= 0 && e1 <= 0 && e2 <= 0 && e0 >= -1 && e1 >= -1 && e2 >= -1;
            if (pos || neg) m |= (unsigned short)(1u << b);
        }
        p.adj[a] = m;
    }
}

static SZP_QHD int szp_pl_ctz(unsigned int x)
{
#if defined(__CUDA_ARCH__)
    return __ffs((int)x) - 1;
#else
    return __builtin_ctz(x);
#endif
}

/** Number of connected pieces of the link vertices in mask. */
static SZP_QHD int szp_pl_pieces(const SzpPL &p, unsigned int mask)
{
    int n = 0;
    while (mask) {
        unsigned int seen = 1u << szp_pl_ctz(mask), front = seen;
        while (front) {
            int a = szp_pl_ctz(front);
            front &= front - 1;
            unsigned int nx = (unsigned int)p.adj[a] & mask & ~seen;
            seen |= nx;
            front |= nx;
        }
        mask &= ~seen;
        n++;
    }
    return n;
}

/** PL code of interior point k. */
static SZP_QHD int szp_pl_classify(const float *v, const SzpGrid &g, const SzpPL &p, long long k)
{
    const float c = v[k];
    unsigned int lo = 0, up = 0;
    for (int a = 0; a < p.nb; a++) {
        long long q = k + p.off[a];
        float w = v[q];
        if (w < c || (w == c && q < k)) lo |= 1u << a;
        else up |= 1u << a;
    }
    if (!lo) return SZP_PL_MIN;
    if (!up) return SZP_PL_MAX;
    int cl = szp_pl_pieces(p, lo), cu = szp_pl_pieces(p, up);
    if (cl == 1 && cu == 1) return SZP_PL_REG;
    if (g.nd == 2 || cu == 1) return SZP_PL_SAD1;
    if (cl == 1) return SZP_PL_SAD2;
    return SZP_PL_SADX;
}

/**
 * Exhaustive single-point fix of interior point k towards codes[k]
 * (see the file comment). orig = field after dequantization. Returns 1 if
 * the point now has its stored code (value changed), 0 otherwise.
 */
static SZP_QHD int szp_pl_fix_point(float *v, const int *codes, const float *orig, const SzpGrid &g,
                                    const SzpPL &p, long long k, float eb)
{
    const int target = codes[k];
    if (szp_pl_classify(v, g, p, k) == target) return 0;

    /* Neighbours that have their stored code now and must keep it. */
    unsigned int keep = 0;
    for (int a = 0; a < p.nb; a++) {
        long long q = k + p.off[a];
        if (szp_is_interior(g, q) && szp_pl_classify(v, g, p, q) == codes[q]) keep |= 1u << a;
    }

    const float lo_b = szp_fadd(orig[k], -szp_fmul(eb, 0.999f));
    const float hi_b = szp_fadd(orig[k], szp_fmul(eb, 0.999f));

    /* Sorted neighbour values; one candidate per gap inside [lo_b, hi_b]. */
    float nv[14];
    for (int a = 0; a < p.nb; a++) nv[a] = v[k + p.off[a]];
    for (int a = 1; a < p.nb; a++) {
        float x = nv[a];
        int b = a - 1;
        while (b >= 0 && nv[b] > x) { nv[b + 1] = nv[b]; b--; }
        nv[b + 1] = x;
    }
    /* Candidates: one value inside every gap between distinct neighbour
       values, plus every neighbour value itself (a tie is resolved by index,
       which can split equal neighbours into below and above). Sorted by
       distance from the current value, then by value. */
    float cand[30];
    float dist[30];
    int nc = 0;
    const float cur = v[k];
    for (int i = 0; i <= p.nb; i++) {
        for (int pass = 0; pass < 2; pass++) {
            float t;
            if (pass == 0) {                       /* inside the gap below nv[i] */
                if (i > 0 && i < p.nb && nv[i - 1] == nv[i]) continue;
                float wlo = i == 0 ? lo_b : fmaxf(nv[i - 1], lo_b);
                float whi = i == p.nb ? hi_b : fminf(nv[i], hi_b);
                if (!(wlo <= whi)) continue;
                if (i == 0)         t = whi < hi_b ? szp_fadd(wlo, szp_fmul(szp_fadd(whi, -wlo), 0.5f)) : wlo;
                else if (i == p.nb) t = wlo > lo_b ? szp_fadd(wlo, szp_fmul(szp_fadd(whi, -wlo), 0.5f)) : whi;
                else                t = szp_fadd(wlo, szp_fmul(szp_fadd(whi, -wlo), 0.5f));
            } else {                               /* equal to nv[i] */
                if (i == p.nb || (i > 0 && nv[i - 1] == nv[i])) continue;
                t = nv[i];
            }
            if (!(t >= lo_b && t <= hi_b) || t == cur) continue;
            float d = fabsf(szp_fadd(t, -cur));
            int dup = 0;
            for (int j = 0; j < nc; j++) dup |= cand[j] == t;
            if (dup) continue;
            int j = nc++;
            while (j > 0 && (dist[j - 1] > d || (dist[j - 1] == d && cand[j - 1] > t))) {
                cand[j] = cand[j - 1]; dist[j] = dist[j - 1]; j--;
            }
            cand[j] = t; dist[j] = d;
        }
    }

    for (int c = 0; c < nc; c++) {
        v[k] = cand[c];
        if (szp_pl_classify(v, g, p, k) != target) continue;
        int ok = 1;
        for (int a = 0; a < p.nb && ok; a++)
            if ((keep >> a) & 1) {
                long long q = k + p.off[a];
                if (szp_pl_classify(v, g, p, q) != codes[q]) ok = 0;
            }
        if (ok) return 1;
    }
    v[k] = cur;
    return 0;
}

/* ---------------- host helpers shared by the CPU and CUDA drivers ------ */

/** Interior points whose PL code differs from the stored one, by colour. */
static inline void szp_pl_candidates(const float *v, const int *codes, const SzpGrid &g,
                                     std::vector<long long> &list, std::vector<size_t> &start)
{
    const int ncol = g.nd == 3 ? 27 : 9;
    const long long N = (long long)g.n[0] * g.n[1] * g.n[2];
    SzpPL p;
    szp_pl_init(g, p);
    std::vector<unsigned char> flag((size_t)N, 0);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long long k = 0; k < N; k++)
        flag[(size_t)k] = szp_is_interior(g, k) && szp_pl_classify(v, g, p, k) != codes[k];
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

/** Mismatch counts against the stored codes. */
struct SzpPLCounts { size_t ext, ext_lost, sad, sad_lost, fp, ft; };

static inline SzpPLCounts szp_pl_count(const float *v, const int *codes, const SzpGrid &g)
{
    const long long N = (long long)g.n[0] * g.n[1] * g.n[2];
    SzpPL p;
    szp_pl_init(g, p);
    size_t ext = 0, el = 0, sad = 0, sl = 0, fp = 0, ft = 0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:ext,el,sad,sl,fp,ft) schedule(static)
#endif
    for (long long k = 0; k < N; k++) {
        if (!szp_is_interior(g, k)) continue;
        int t = codes[k], now = szp_pl_classify(v, g, p, k);
        if (t == 1 || t == 2) { ext++; el += now != t; }
        else if (t >= 3)      { sad++; sl += now != t; }
        if (now != t && now != 0) { if (t == 0) fp++; else ft++; }
    }
    SzpPLCounts c = {ext, el, sad, sl, fp, ft};
    return c;
}

/** Extrema step of szp_topo_restore.h over the PL neighbours. */
static inline void szp_pl_restore_extrema(float *data, const float *orig, const int *codes,
                                          const int *sort_positions, size_t extrema_count,
                                          const SzpGrid &g, float errBound)
{
    const long long N = (long long)g.n[0] * g.n[1] * g.n[2];
    std::vector<SzpExtremum> ext;
    size_t r = 0;
    for (long long k = 0; k < N; k++) {
        int t = codes[k];
        if ((t != 1 && t != 2) || !szp_is_interior(g, k)) continue;
        int rank = (sort_positions && r < extrema_count) ? sort_positions[r] : 0;
        r++;
        ext.push_back({(size_t)k, orig[k], t, rank < 0 ? 0 : rank});
    }
    SzpPL p;
    szp_pl_init(g, p);
    ptrdiff_t off[14];
    for (int a = 0; a < p.nb; a++) off[a] = (ptrdiff_t)p.off[a];
    szp_restore_extrema_ranked_core(data, ext, off, p.nb, (double)errBound);
}

#endif /* __cplusplus */
#endif /* _SZP_PL_TOPOLOGY_H */
