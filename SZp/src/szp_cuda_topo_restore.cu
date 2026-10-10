/**
 *  @file szp_cuda_topo_restore.cu
 *  @brief GPU topology restoration; see szp_cuda_topo_restore.cuh and
 *         szp_topo_restore.h for the algorithm.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include "szp_cuda_topo_restore.cuh"
#include "szp_topo_restore.h"
#include "szp_pl_topology.h"

#define SZP_CUDA_TRY(call)                                                      \
    do {                                                                        \
        cudaError_t e_ = (call);                                                \
        if (e_ != cudaSuccess) {                                                \
            fprintf(stderr, "szp_cuda_topo_restore: %s failed: %s\n", #call,    \
                    cudaGetErrorString(e_));                                    \
            goto fail;                                                          \
        }                                                                       \
    } while (0)

/* One thread per candidate of the current colour class. Points of one class
   are at least 3 apart, so the updates do not interact. */
__global__ void szp_saddle_kernel(float *v, const int *types, const float *orig,
                                  const unsigned char *locks, SzpGrid g, float eb,
                                  const double *w, const long long *list,
                                  size_t lo, size_t hi, int use_rbf,
                                  unsigned long long *restored)
{
    size_t i = lo + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hi) return;
    long long k = list[i];
    int ok = use_rbf ? szp_saddle_rbf_point(v, types, locks, orig, g, k, eb, w)
                     : szp_saddle_stencil_point(v, types, orig, g, k, eb);
    if (ok) atomicAdd(restored, 1ULL);
}

static int restore_gpu(float *data, const int *types, const int *sort_positions,
                       size_t extrema_count, const SzpGrid &g, float eb,
                       szp_topo_restore_stats *stats)
{
    szp_topo_restore_stats st;
    memset(&st, 0, sizeof(st));
    const size_t N = (size_t)g.n[0] * g.n[1] * g.n[2];
    const int ncol = g.nd == 3 ? 27 : 9;
    std::vector<float> orig(data, data + N);
    std::vector<unsigned char> locks;
    std::vector<long long> list;
    std::vector<size_t> start;
    double w[27];
    float *d_v = NULL, *d_orig = NULL;
    int *d_types = NULL;
    unsigned char *d_locks = NULL;
    double *d_w = NULL;
    long long *d_list = NULL;
    unsigned long long *d_count = NULL;
    size_t list_cap = 0;
    int rc = -1;

    szp_count_lost(data, types, g, &st.extrema, &st.extrema_lost_before, &st.saddles, NULL);

    /* 1. extrema (host) */
    szp_restore_extrema_ranked(data, orig.data(), types, sort_positions, extrema_count, g, eb);
    szp_count_lost(data, types, g, NULL, NULL, NULL, &st.saddles_lost_before);

    /* 2-3. saddles (GPU) */
    szp_extremum_locks(types, g, locks);
    szp_rbf_weights(g.nd, w);
    SZP_CUDA_TRY(cudaMalloc((void **)&d_v, N * sizeof(float)));
    SZP_CUDA_TRY(cudaMalloc((void **)&d_orig, N * sizeof(float)));
    SZP_CUDA_TRY(cudaMalloc((void **)&d_types, N * sizeof(int)));
    SZP_CUDA_TRY(cudaMalloc((void **)&d_locks, N));
    SZP_CUDA_TRY(cudaMalloc((void **)&d_w, sizeof(w)));
    SZP_CUDA_TRY(cudaMalloc((void **)&d_count, sizeof(unsigned long long)));
    SZP_CUDA_TRY(cudaMemcpy(d_v, data, N * sizeof(float), cudaMemcpyHostToDevice));
    SZP_CUDA_TRY(cudaMemcpy(d_orig, orig.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    SZP_CUDA_TRY(cudaMemcpy(d_types, types, N * sizeof(int), cudaMemcpyHostToDevice));
    SZP_CUDA_TRY(cudaMemcpy(d_locks, locks.data(), N, cudaMemcpyHostToDevice));
    SZP_CUDA_TRY(cudaMemcpy(d_w, w, sizeof(w), cudaMemcpyHostToDevice));

    for (int round = 0; round < 2; round++) {
        for (int use_rbf = 0; use_rbf < 2; use_rbf++) {
            /* Candidates from the current field (same order as the CPU). */
            if (round > 0 || use_rbf > 0)
                SZP_CUDA_TRY(cudaMemcpy(data, d_v, N * sizeof(float), cudaMemcpyDeviceToHost));
            szp_saddle_candidates(data, types, g, list, start);
            if (list.empty()) continue;
            if (list.size() > list_cap) {
                if (d_list) cudaFree(d_list);
                d_list = NULL;
                list_cap = list.size();
                SZP_CUDA_TRY(cudaMalloc((void **)&d_list, list_cap * sizeof(long long)));
            }
            SZP_CUDA_TRY(cudaMemcpy(d_list, list.data(), list.size() * sizeof(long long),
                                    cudaMemcpyHostToDevice));
            SZP_CUDA_TRY(cudaMemset(d_count, 0, sizeof(unsigned long long)));
            for (int c = 0; c < ncol; c++) {
                size_t lo = start[c], hi = start[c + 1];
                if (hi <= lo) continue;
                const unsigned int tpb = 128;
                unsigned int nblk = (unsigned int)((hi - lo + tpb - 1) / tpb);
                szp_saddle_kernel<<<nblk, tpb>>>(d_v, d_types, d_orig, d_locks, g, eb, d_w,
                                                 d_list, lo, hi, use_rbf, d_count);
                SZP_CUDA_TRY(cudaGetLastError());
            }
            unsigned long long got = 0;
            SZP_CUDA_TRY(cudaMemcpy(&got, d_count, sizeof(got), cudaMemcpyDeviceToHost));
            if (use_rbf) st.saddles_by_rbf += (size_t)got;
            else         st.saddles_by_stencil += (size_t)got;
        }
    }
    SZP_CUDA_TRY(cudaMemcpy(data, d_v, N * sizeof(float), cudaMemcpyDeviceToHost));
    szp_count_lost(data, types, g, NULL, &st.extrema_lost_after, NULL, &st.saddles_lost_after);
    if (stats) *stats = st;
    rc = 0;

fail:
    cudaFree(d_v); cudaFree(d_orig); cudaFree(d_types); cudaFree(d_locks);
    cudaFree(d_w); cudaFree(d_list); cudaFree(d_count);
    return rc;
}

extern "C" int szp_cuda_topo_restore_float_2d(float *data, const int *types, const int *sort_positions,
                                              size_t extrema_count, int rows, int cols, float errBound,
                                              szp_topo_restore_stats *stats)
{
    if (!data || !types || rows < 3 || cols < 3 || !(errBound > 0)) return -1;
    return restore_gpu(data, types, sort_positions, extrema_count, szp_grid_2d(rows, cols), errBound, stats);
}

extern "C" int szp_cuda_topo_restore_float_3d(float *data, const int *types, const int *sort_positions,
                                              size_t extrema_count, int d1, int d2, int d3, float errBound,
                                              szp_topo_restore_stats *stats)
{
    if (!data || !types || d1 < 3 || d2 < 3 || d3 < 3 || !(errBound > 0)) return -1;
    return restore_gpu(data, types, sort_positions, extrema_count, szp_grid_3d(d1, d2, d3), errBound, stats);
}

/* ===================================================================== */
/*  PL mode (szp_pl_topology.h)                                          */
/* ===================================================================== */

__global__ void szp_pl_classify_kernel(const float *v, SzpGrid g, signed char *codes, long long N)
{
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    SzpPL p;
    szp_pl_init(g, p);
    codes[k] = szp_is_interior(g, k) ? (signed char)szp_pl_classify(v, g, p, k) : 0;
}

/* One thread per candidate of the current colour class (points >= 3 apart). */
__global__ void szp_pl_fix_kernel(float *v, const int *codes, const float *orig, SzpGrid g, float eb,
                                  const long long *list, size_t lo, size_t hi,
                                  unsigned long long *fixed /* [0] saddles, [1] other */)
{
    size_t i = lo + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hi) return;
    SzpPL p;
    szp_pl_init(g, p);
    long long k = list[i];
    if (szp_pl_fix_point(v, codes, orig, g, p, k, eb))
        atomicAdd(&fixed[codes[k] >= 3 ? 0 : 1], 1ULL);
}

extern "C" int szp_cuda_pl_classify_float(const float *data, int nd, int d1, int d2, int d3, signed char *codes)
{
    const SzpGrid g = nd == 3 ? szp_grid_3d(d1, d2, d3) : szp_grid_2d(d1, d2);
    const long long N = (long long)g.n[0] * g.n[1] * g.n[2];
    float *d_v = NULL;
    signed char *d_c = NULL;
    int rc = -1;
    SZP_CUDA_TRY(cudaMalloc((void **)&d_v, (size_t)N * sizeof(float)));
    SZP_CUDA_TRY(cudaMalloc((void **)&d_c, (size_t)N));
    SZP_CUDA_TRY(cudaMemcpy(d_v, data, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    {
        const unsigned int tpb = 256;
        unsigned int nblk = (unsigned int)((N + tpb - 1) / tpb);
        szp_pl_classify_kernel<<<nblk, tpb>>>(d_v, g, d_c, N);
        SZP_CUDA_TRY(cudaGetLastError());
    }
    SZP_CUDA_TRY(cudaMemcpy(codes, d_c, (size_t)N, cudaMemcpyDeviceToHost));
    rc = 0;
fail:
    cudaFree(d_v); cudaFree(d_c);
    return rc;
}

extern "C" int szp_cuda_topo_restore_float_pl(float *data, const int *codes, const int *sort_positions,
                                              size_t extrema_count, int nd, int d1, int d2, int d3,
                                              float errBound, szp_topo_restore_stats *stats)
{
    if (!data || !codes || d1 < 3 || d2 < 3 || (nd == 3 && d3 < 3) || !(errBound > 0)) return -1;
    const SzpGrid g = nd == 3 ? szp_grid_3d(d1, d2, d3) : szp_grid_2d(d1, d2);
    const size_t N = (size_t)g.n[0] * g.n[1] * g.n[2];
    const int ncol = g.nd == 3 ? 27 : 9;
    szp_topo_restore_stats st;
    memset(&st, 0, sizeof(st));
    std::vector<float> orig(data, data + N);
    std::vector<long long> list;
    std::vector<size_t> start;
    float *d_v = NULL, *d_orig = NULL;
    int *d_codes = NULL;
    long long *d_list = NULL;
    unsigned long long *d_count = NULL;
    size_t list_cap = 0;
    int rc = -1;

    SzpPLCounts c0 = szp_pl_count(data, codes, g);
    st.extrema = c0.ext; st.extrema_lost_before = c0.ext_lost; st.saddles = c0.sad;
    st.fp_before = c0.fp; st.ft_before = c0.ft;

    /* 1. extrema (host) */
    szp_pl_restore_extrema(data, orig.data(), codes, sort_positions, extrema_count, g, errBound);
    st.saddles_lost_before = szp_pl_count(data, codes, g).sad_lost;

    /* 2. fix passes (GPU) */
    SZP_CUDA_TRY(cudaMalloc((void **)&d_v, N * sizeof(float)));
    SZP_CUDA_TRY(cudaMalloc((void **)&d_orig, N * sizeof(float)));
    SZP_CUDA_TRY(cudaMalloc((void **)&d_codes, N * sizeof(int)));
    SZP_CUDA_TRY(cudaMalloc((void **)&d_count, 2 * sizeof(unsigned long long)));
    SZP_CUDA_TRY(cudaMemcpy(d_v, data, N * sizeof(float), cudaMemcpyHostToDevice));
    SZP_CUDA_TRY(cudaMemcpy(d_orig, orig.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    SZP_CUDA_TRY(cudaMemcpy(d_codes, codes, N * sizeof(int), cudaMemcpyHostToDevice));

    for (int round = 0; round < SZP_PL_ROUNDS; round++) {
        if (round > 0)
            SZP_CUDA_TRY(cudaMemcpy(data, d_v, N * sizeof(float), cudaMemcpyDeviceToHost));
        szp_pl_candidates(data, codes, g, list, start);   /* same order as the CPU */
        st.rounds = (size_t)round + 1;
        if (list.empty()) break;
        if (list.size() > list_cap) {
            if (d_list) cudaFree(d_list);
            d_list = NULL;
            list_cap = list.size();
            SZP_CUDA_TRY(cudaMalloc((void **)&d_list, list_cap * sizeof(long long)));
        }
        SZP_CUDA_TRY(cudaMemcpy(d_list, list.data(), list.size() * sizeof(long long), cudaMemcpyHostToDevice));
        SZP_CUDA_TRY(cudaMemset(d_count, 0, 2 * sizeof(unsigned long long)));
        for (int c = 0; c < ncol; c++) {
            size_t lo = start[c], hi = start[c + 1];
            if (hi <= lo) continue;
            const unsigned int tpb = 128;
            unsigned int nblk = (unsigned int)((hi - lo + tpb - 1) / tpb);
            szp_pl_fix_kernel<<<nblk, tpb>>>(d_v, d_codes, d_orig, g, errBound, d_list, lo, hi, d_count);
            SZP_CUDA_TRY(cudaGetLastError());
        }
        unsigned long long got[2] = {0, 0};
        SZP_CUDA_TRY(cudaMemcpy(got, d_count, sizeof(got), cudaMemcpyDeviceToHost));
        st.saddles_by_stencil += (size_t)got[0];
        st.other_fixes += (size_t)got[1];
        if (got[0] + got[1] == 0) break;
    }
    SZP_CUDA_TRY(cudaMemcpy(data, d_v, N * sizeof(float), cudaMemcpyDeviceToHost));
    {
        SzpPLCounts c1 = szp_pl_count(data, codes, g);
        st.extrema_lost_after = c1.ext_lost; st.saddles_lost_after = c1.sad_lost;
        st.fp_after = c1.fp; st.ft_after = c1.ft;
    }
    if (stats) *stats = st;
    rc = 0;

fail:
    cudaFree(d_v); cudaFree(d_orig); cudaFree(d_codes); cudaFree(d_list); cudaFree(d_count);
    return rc;
}
