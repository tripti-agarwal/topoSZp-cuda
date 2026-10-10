/**
 *  @file szp_topo_restore.h
 *  @brief Restore local extrema after decompression (C++ only, header-only).
 *
 *  Used by the 2D OpenMP decompressor example and the 3D CUDA test so both
 *  apply the same rule.
 *
 *  For every point stored as a maximum or minimum, with decompressed value d
 *  (its quantization bin) and stored rank r (its order by original value
 *  among critical points in that bin):
 *
 *    maxima:  D(p) = d + (r + 1) * eta
 *    minima:  D(p) = d - (R + 1 - r) * eta
 *    eta = 0.5 * eb / (R + 1),  R = largest rank among same-type extrema
 *                                   that share the bin
 *
 *  so maxima end up strictly above d and minima strictly below, and extrema
 *  of the same type in the same bin keep the order of their original values.
 *  If a neighbour is still in the way (e.g. a saddle moved by the RBF step),
 *  the value is pushed just past that neighbour. Every result is clamped to
 *  |D(p) - d| <= eb, so with the base SZp bound |original - d| <= eb the
 *  point error stays within 2 * eb.
 *
 *  An extremum that is alone in its bin (no other extremum of the same type
 *  shares the value) and is already strict is left untouched.
 *
 *  All targets are computed from the field as it is on entry and written
 *  afterwards, so the result does not depend on the thread count.
 */

#ifndef _SZP_TOPO_RESTORE_H
#define _SZP_TOPO_RESTORE_H

#ifdef __cplusplus

#include <stddef.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>

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
 * 2D, 4-neighbour. types: stored type of every point (0..3); sort_positions:
 * one rank per stored extremum, in raster order of the interior; orig_decomp:
 * the field right after dequantization.
 */
static inline void
szp_restore_extrema_ranked_2d(float *data, const float *orig_decomp, const int *types,
                              const int *sort_positions, size_t extrema_count,
                              int rows, int cols, float errBound)
{
    if (!data || !orig_decomp || !types || rows < 3 || cols < 3) return;
    std::vector<SzpExtremum> ext;
    ext.reserve(extrema_count ? extrema_count : 1024);
    size_t k = 0;
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            size_t idx = (size_t)i * (size_t)cols + (size_t)j;
            int t = types[idx];
            if (t != 1 && t != 2) continue;
            int r = (sort_positions && k < extrema_count) ? sort_positions[k] : 0;
            k++;
            ext.push_back({idx, orig_decomp[idx], t, r < 0 ? 0 : r});
        }
    }
    const ptrdiff_t off[4] = {-(ptrdiff_t)cols, (ptrdiff_t)cols, -1, 1};
    szp_restore_extrema_ranked_core(data, ext, off, 4, (double)errBound);
}

/** 3D, 6-neighbour; flat index x*d2*d3 + y*d3 + z. */
static inline void
szp_restore_extrema_ranked_3d(float *data, const float *orig_decomp, const int *types,
                              const int *sort_positions, size_t extrema_count,
                              int d1, int d2, int d3, float errBound)
{
    if (!data || !orig_decomp || !types || d1 < 3 || d2 < 3 || d3 < 3) return;
    const size_t plane = (size_t)d2 * (size_t)d3;
    std::vector<SzpExtremum> ext;
    ext.reserve(extrema_count ? extrema_count : 1024);
    size_t k = 0;
    for (int x = 1; x < d1 - 1; x++) {
        for (int y = 1; y < d2 - 1; y++) {
            for (int z = 1; z < d3 - 1; z++) {
                size_t idx = (size_t)x * plane + (size_t)y * (size_t)d3 + (size_t)z;
                int t = types[idx];
                if (t != 1 && t != 2) continue;
                int r = (sort_positions && k < extrema_count) ? sort_positions[k] : 0;
                k++;
                ext.push_back({idx, orig_decomp[idx], t, r < 0 ? 0 : r});
            }
        }
    }
    const ptrdiff_t off[6] = {-(ptrdiff_t)plane, (ptrdiff_t)plane,
                              -(ptrdiff_t)d3, (ptrdiff_t)d3, -1, 1};
    szp_restore_extrema_ranked_core(data, ext, off, 6, (double)errBound);
}

#endif /* __cplusplus */
#endif /* _SZP_TOPO_RESTORE_H */
