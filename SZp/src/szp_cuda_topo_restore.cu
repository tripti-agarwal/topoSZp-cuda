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
