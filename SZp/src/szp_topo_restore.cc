/**
 *  @file szp_topo_restore.cc
 *  @brief CPU (OpenMP) topology restoration; see szp_topo_restore.h.
 */
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "szp_topo_restore.h"
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
