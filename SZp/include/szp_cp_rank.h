/**
 *  @file szp_cp_rank.h
 *  @brief Shared helpers for critical-point lists (C++ only, used by the
 *         OpenMP 2D and 3D topology code).
 *
 *  - szp_assign_sort_positions(): rank critical points by original value
 *    inside each quantization bin. Ranks start at 0 and equal values share a
 *    rank. The result depends only on the data, not on the thread count or on
 *    the order of the input list.
 */

#ifndef _SZP_CP_RANK_H
#define _SZP_CP_RANK_H

#ifdef __cplusplus

#include <stddef.h>
#include <algorithm>
#include <vector>
#if defined(_OPENMP) && defined(__GLIBCXX__)
#include <parallel/algorithm>
#endif

template <class CP, class FlatIndexFn>
static void szp_assign_sort_positions(CP *cps, size_t n, const float *data, FlatIndexFn flat_of)
{
    if (!cps || n == 0 || !data) return;

    /* Compact keys sorted by (bin, value, index). The index makes every key
       unique, so the order (and the ranks) never depend on the sort
       algorithm or the thread count. */
    struct RankKey { int bin; float val; size_t idx; };
    std::vector<RankKey> keys(n);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long long i = 0; i < (long long)n; i++) {
        keys[i].bin = cps[i].quantized_bin;
        keys[i].val = data[flat_of(cps[i])];
        keys[i].idx = (size_t)i;
    }
    auto less = [](const RankKey &a, const RankKey &b) {
        if (a.bin != b.bin) return a.bin < b.bin;
        if (a.val != b.val) return a.val < b.val;
        return a.idx < b.idx;
    };
#if defined(_OPENMP) && defined(__GLIBCXX__)
    __gnu_parallel::sort(keys.begin(), keys.end(), less);
#else
    std::sort(keys.begin(), keys.end(), less);
#endif

    int pos = 0;
    for (size_t k = 0; k < n; k++) {
        if (k > 0) {
            if (keys[k].bin != keys[k - 1].bin) pos = 0;
            else if (keys[k].val != keys[k - 1].val) pos++;
        }
        cps[keys[k].idx].sort_position = pos;
    }
}

#endif /* __cplusplus */
#endif /* _SZP_CP_RANK_H */
