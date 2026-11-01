/**
 *  @file szp_cuda_topo_restore.cu
 *  @brief GPU topology restoration; see szp_cuda_topo_restore.cuh and
 *         szp_topo_restore.h for the algorithm.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
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

/* Previous driver: extrema step, counts and candidate lists on the host,
   saddle passes on the GPU. Kept for comparison (SZP_CUDA_RESTORE=hybrid). */
static int restore_gpu_hybrid(float *data, const int *types, const int *sort_positions,
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

/* ===================================================================== */
/*  Device-resident restoration: every step on the GPU                    */
/* ===================================================================== */

#define SZP_TPB 256   /* threads per block (multiple of 32) */

static unsigned int szp_nblk(long long n) { return (unsigned int)((n + SZP_TPB - 1) / SZP_TPB); }

/* Stored points that do not have their stored type: cnt[0] extrema,
   cnt[1] extrema lost, cnt[2] saddles, cnt[3] saddles lost (szp_count_lost). */
__global__ void szp_count_lost_kernel(const float *v, const int *types, SzpGrid g, long long N,
                                      unsigned long long *cnt)
{
    unsigned long long et = 0, el = 0, st = 0, sl = 0;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < N; k += stride) {
        int t = types[k];
        if (t == 0 || !szp_is_interior(g, k)) continue;
        int now = szp_classify(v, g, k);
        if (t == 3) { st++; sl += (now != 3); }
        else        { et++; el += (now != t); }
    }
    if (et) atomicAdd(&cnt[0], et);
    if (el) atomicAdd(&cnt[1], el);
    if (st) atomicAdd(&cnt[2], st);
    if (sl) atomicAdd(&cnt[3], sl);
}

static SZP_QHD int szp_is_stored_extremum(const int *types, const SzpGrid &g, long long k)
{
    int t = types[k];
    return (t == 1 || t == 2) && szp_is_interior(g, k);
}

/* Extrema in raster order (the order of the stored ranks): per-block counts,
   then a scatter with the in-block prefix (warp ballots). */
__global__ void szp_ext_count_kernel(const int *types, SzpGrid g, long long N, unsigned long long *bcount)
{
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int f = k < N && szp_is_stored_extremum(types, g, k);
    int c = __syncthreads_count(f);
    if (threadIdx.x == 0) bcount[blockIdx.x] = (unsigned long long)c;
}

__global__ void szp_ext_scatter_kernel(const int *types, SzpGrid g, long long N,
                                       const unsigned long long *boff, long long *ext_idx)
{
    __shared__ unsigned int wsum[SZP_TPB / 32];
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int f = k < N && szp_is_stored_extremum(types, g, k);
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int b = __ballot_sync(0xffffffffu, f);
    const unsigned int pre = __popc(b & ((1u << lane) - 1u));
    if (lane == 0) wsum[warp] = __popc(b);
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int acc = 0;
        for (unsigned int w = 0; w < blockDim.x / 32; w++) { unsigned int t = wsum[w]; wsum[w] = acc; acc += t; }
    }
    __syncthreads();
    if (f) ext_idx[boff[blockIdx.x] + wsum[warp] + pre] = k;
}

static __device__ __forceinline__ unsigned long long szp_mix64(unsigned long long x)
{
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}

#define SZP_EMPTY_KEY 0xffffffffffffffffULL   /* never a key: types are <= 2 */

/* Groups of extrema with the same type and decompressed value (a hash table
   instead of the CPU's sort): largest rank and size per group. Both are
   independent of the insertion order. */
__global__ void szp_ext_group_kernel(const long long *ext_idx, long long n, const float *orig, const int *types,
                                     const int *sort_positions, unsigned long long extrema_count,
                                     unsigned long long *tkey, int *tmax, int *tcnt, unsigned long long mask,
                                     int *rank, unsigned long long *slot)
{
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const long long idx = ext_idx[r];
    const unsigned long long key = szp_extremum_key(types[idx], orig[idx]);
    int rk = (sort_positions && (unsigned long long)r < extrema_count) ? sort_positions[r] : 0;
    if (rk < 0) rk = 0;
    unsigned long long h = szp_mix64(key) & mask;
    for (;;) {
        unsigned long long prev = atomicCAS(&tkey[h], SZP_EMPTY_KEY, key);
        if (prev == SZP_EMPTY_KEY || prev == key) break;
        h = (h + 1) & mask;
    }
    atomicMax(&tmax[h], rk);
    atomicAdd(&tcnt[h], 1);
    rank[r] = rk;
    slot[r] = h;
}

__global__ void szp_ext_target_kernel(const float *v, const float *orig, const int *types, const long long *ext_idx,
                                      long long n, const int *rank, const unsigned long long *slot,
                                      const int *tmax, const int *tcnt, SzpGrid g, double eb, float *tgt)
{
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    ptrdiff_t off[6];
    for (int a = 0; a < g.nd; a++) { off[2 * a] = -(ptrdiff_t)g.s[a]; off[2 * a + 1] = (ptrdiff_t)g.s[a]; }
    const long long idx = ext_idx[r];
    const unsigned long long h = slot[r];
    tgt[r] = szp_extremum_target(v, (ptrdiff_t)idx, orig[idx], types[idx], rank[r], tmax[h], tcnt[h],
                                 off, 2 * g.nd, eb);
}

__global__ void szp_ext_write_kernel(float *v, const long long *ext_idx, long long n, const float *tgt)
{
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n) v[ext_idx[r]] = tgt[r];
}

__global__ void szp_locks_kernel(const int *types, SzpGrid g, long long N, unsigned char *locks)
{
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) locks[k] = szp_extremum_lock_at(types, g, k);
}

/* Saddle candidates (stored saddles that are not saddles now) by colour:
   mark[k] = colour or 255, per-colour counts, then a scatter. The order
   inside a colour does not matter: points of one colour do not interact. */
__global__ void szp_cand_mark_kernel(const float *v, const int *types, SzpGrid g, long long N,
                                     unsigned char *mark, unsigned long long *ccount)
{
    __shared__ unsigned int h[27];
    if (threadIdx.x < 27) h[threadIdx.x] = 0;
    __syncthreads();
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) {
        unsigned char m = 255;
        if (types[k] == 3 && szp_is_interior(g, k) && szp_classify(v, g, k) != 3) {
            int c = szp_colour(g, k);
            m = (unsigned char)c;
            atomicAdd(&h[c], 1u);
        }
        mark[k] = m;
    }
    __syncthreads();
    if (threadIdx.x < 27 && h[threadIdx.x]) atomicAdd(&ccount[threadIdx.x], (unsigned long long)h[threadIdx.x]);
}

__global__ void szp_cand_scatter_kernel(const unsigned char *mark, long long N, unsigned long long *cursor,
                                        long long *list)
{
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N || mark[k] == 255) return;
    unsigned long long pos = atomicAdd(&cursor[mark[k]], 1ULL);
    list[pos] = k;
}

static int restore_gpu_device(float *data, const int *types, const int *sort_positions,
                              size_t extrema_count, const SzpGrid &g, float eb,
                              szp_topo_restore_stats *stats)
{
    szp_topo_restore_stats st;
    memset(&st, 0, sizeof(st));
    const long long N = (long long)g.n[0] * g.n[1] * g.n[2];
    const int ncol = g.nd == 3 ? 27 : 9;
    const unsigned int nb = szp_nblk(N);
    const unsigned int count_blocks = nb < 4096u ? nb : 4096u;
    double w[27];
    float *d_v = NULL, *d_orig = NULL, *d_tgt = NULL;
    int *d_types = NULL, *d_sp = NULL, *d_rank = NULL, *d_tmax = NULL, *d_tcnt = NULL;
    unsigned char *d_locks = NULL, *d_mark = NULL;
    double *d_w = NULL;
    long long *d_ext = NULL, *d_list = NULL;
    unsigned long long *d_cnt = NULL, *d_bcount = NULL, *d_tkey = NULL, *d_slot = NULL, *d_cc = NULL;
    std::vector<unsigned long long> boff;
    unsigned long long cnt[4];
    long long n_ext = 0;
    size_t list_cap = 0;
    int rc = -1;

    SZP_CUDA_TRY(cudaMalloc((void **)&d_v, (size_t)N * sizeof(float)));
    SZP_CUDA_TRY(cudaMalloc((void **)&d_orig, (size_t)N * sizeof(float)));
    SZP_CUDA_TRY(cudaMalloc((void **)&d_types, (size_t)N * sizeof(int)));
    SZP_CUDA_TRY(cudaMalloc((void **)&d_cnt, 4 * sizeof(unsigned long long)));
    SZP_CUDA_TRY(cudaMemcpy(d_v, data, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    SZP_CUDA_TRY(cudaMemcpy(d_orig, d_v, (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice));
    SZP_CUDA_TRY(cudaMemcpy(d_types, types, (size_t)N * sizeof(int), cudaMemcpyHostToDevice));

    /* counts before */
    SZP_CUDA_TRY(cudaMemset(d_cnt, 0, 4 * sizeof(unsigned long long)));
    szp_count_lost_kernel<<<count_blocks, SZP_TPB>>>(d_v, d_types, g, N, d_cnt);
    SZP_CUDA_TRY(cudaGetLastError());
    SZP_CUDA_TRY(cudaMemcpy(cnt, d_cnt, sizeof(cnt), cudaMemcpyDeviceToHost));
    st.extrema = (size_t)cnt[0]; st.extrema_lost_before = (size_t)cnt[1]; st.saddles = (size_t)cnt[2];

    /* 1. extrema */
    SZP_CUDA_TRY(cudaMalloc((void **)&d_bcount, (size_t)nb * sizeof(unsigned long long)));
    szp_ext_count_kernel<<<nb, SZP_TPB>>>(d_types, g, N, d_bcount);
    SZP_CUDA_TRY(cudaGetLastError());
    boff.resize(nb);
    SZP_CUDA_TRY(cudaMemcpy(boff.data(), d_bcount, (size_t)nb * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    for (unsigned int b = 0; b < nb; b++) { unsigned long long c = boff[b]; boff[b] = (unsigned long long)n_ext; n_ext += (long long)c; }
    if (n_ext > 0) {
        unsigned long long tsize = 1024;
        while (tsize < 2ULL * (unsigned long long)n_ext) tsize <<= 1;
        const unsigned int ne = szp_nblk(n_ext);
        SZP_CUDA_TRY(cudaMemcpy(d_bcount, boff.data(), (size_t)nb * sizeof(unsigned long long), cudaMemcpyHostToDevice));
        SZP_CUDA_TRY(cudaMalloc((void **)&d_ext, (size_t)n_ext * sizeof(long long)));
        szp_ext_scatter_kernel<<<nb, SZP_TPB>>>(d_types, g, N, d_bcount, d_ext);
        SZP_CUDA_TRY(cudaGetLastError());
        if (sort_positions && extrema_count > 0) {
            SZP_CUDA_TRY(cudaMalloc((void **)&d_sp, extrema_count * sizeof(int)));
            SZP_CUDA_TRY(cudaMemcpy(d_sp, sort_positions, extrema_count * sizeof(int), cudaMemcpyHostToDevice));
        }
        SZP_CUDA_TRY(cudaMalloc((void **)&d_tkey, tsize * sizeof(unsigned long long)));
        SZP_CUDA_TRY(cudaMalloc((void **)&d_tmax, tsize * sizeof(int)));
        SZP_CUDA_TRY(cudaMalloc((void **)&d_tcnt, tsize * sizeof(int)));
        SZP_CUDA_TRY(cudaMalloc((void **)&d_rank, (size_t)n_ext * sizeof(int)));
        SZP_CUDA_TRY(cudaMalloc((void **)&d_slot, (size_t)n_ext * sizeof(unsigned long long)));
        SZP_CUDA_TRY(cudaMalloc((void **)&d_tgt, (size_t)n_ext * sizeof(float)));
        SZP_CUDA_TRY(cudaMemset(d_tkey, 0xff, tsize * sizeof(unsigned long long)));
        SZP_CUDA_TRY(cudaMemset(d_tmax, 0, tsize * sizeof(int)));
        SZP_CUDA_TRY(cudaMemset(d_tcnt, 0, tsize * sizeof(int)));
        szp_ext_group_kernel<<<ne, SZP_TPB>>>(d_ext, n_ext, d_orig, d_types, d_sp,
                                              (unsigned long long)(d_sp ? extrema_count : 0),
                                              d_tkey, d_tmax, d_tcnt, tsize - 1, d_rank, d_slot);
        SZP_CUDA_TRY(cudaGetLastError());
        szp_ext_target_kernel<<<ne, SZP_TPB>>>(d_v, d_orig, d_types, d_ext, n_ext, d_rank, d_slot,
                                               d_tmax, d_tcnt, g, (double)eb, d_tgt);
        SZP_CUDA_TRY(cudaGetLastError());
        szp_ext_write_kernel<<<ne, SZP_TPB>>>(d_v, d_ext, n_ext, d_tgt);
        SZP_CUDA_TRY(cudaGetLastError());
        /* free the extrema buffers before the saddle passes */
        cudaFree(d_tkey); cudaFree(d_tmax); cudaFree(d_tcnt); cudaFree(d_rank); cudaFree(d_slot);
        cudaFree(d_tgt); cudaFree(d_ext); cudaFree(d_sp);
        d_tkey = NULL; d_tmax = NULL; d_tcnt = NULL; d_rank = NULL; d_slot = NULL; d_tgt = NULL; d_ext = NULL; d_sp = NULL;
    }
    cudaFree(d_bcount); d_bcount = NULL;

    SZP_CUDA_TRY(cudaMemset(d_cnt, 0, 4 * sizeof(unsigned long long)));
    szp_count_lost_kernel<<<count_blocks, SZP_TPB>>>(d_v, d_types, g, N, d_cnt);
    SZP_CUDA_TRY(cudaGetLastError());
    SZP_CUDA_TRY(cudaMemcpy(cnt, d_cnt, sizeof(cnt), cudaMemcpyDeviceToHost));
    st.saddles_lost_before = (size_t)cnt[3];

    /* 2-3. saddles: stencil then RBF, twice */
    SZP_CUDA_TRY(cudaMalloc((void **)&d_locks, (size_t)N));
    SZP_CUDA_TRY(cudaMalloc((void **)&d_mark, (size_t)N));
    SZP_CUDA_TRY(cudaMalloc((void **)&d_cc, 2 * 27 * sizeof(unsigned long long)));
    SZP_CUDA_TRY(cudaMalloc((void **)&d_w, sizeof(w)));
    szp_rbf_weights(g.nd, w);
    SZP_CUDA_TRY(cudaMemcpy(d_w, w, sizeof(w), cudaMemcpyHostToDevice));
    szp_locks_kernel<<<nb, SZP_TPB>>>(d_types, g, N, d_locks);
    SZP_CUDA_TRY(cudaGetLastError());

    for (int round = 0; round < 2; round++) {
        for (int use_rbf = 0; use_rbf < 2; use_rbf++) {
            unsigned long long cc[27], start[28];
            SZP_CUDA_TRY(cudaMemset(d_cc, 0, 27 * sizeof(unsigned long long)));
            szp_cand_mark_kernel<<<nb, SZP_TPB>>>(d_v, d_types, g, N, d_mark, d_cc);
            SZP_CUDA_TRY(cudaGetLastError());
            SZP_CUDA_TRY(cudaMemcpy(cc, d_cc, sizeof(cc), cudaMemcpyDeviceToHost));
            start[0] = 0;
            for (int c = 0; c < 27; c++) start[c + 1] = start[c] + cc[c];
            const unsigned long long total = start[27];
            if (total == 0) continue;
            if (total > list_cap) {
                cudaFree(d_list); d_list = NULL;
                list_cap = (size_t)total;
                SZP_CUDA_TRY(cudaMalloc((void **)&d_list, list_cap * sizeof(long long)));
            }
            SZP_CUDA_TRY(cudaMemcpy(d_cc + 27, start, 27 * sizeof(unsigned long long), cudaMemcpyHostToDevice));
            szp_cand_scatter_kernel<<<nb, SZP_TPB>>>(d_mark, N, d_cc + 27, d_list);
            SZP_CUDA_TRY(cudaGetLastError());
            SZP_CUDA_TRY(cudaMemset(d_cnt, 0, sizeof(unsigned long long)));
            for (int c = 0; c < ncol; c++) {
                size_t lo = (size_t)start[c], hi = (size_t)start[c + 1];
                if (hi <= lo) continue;
                const unsigned int tpb = 128;
                unsigned int nblk = (unsigned int)((hi - lo + tpb - 1) / tpb);
                szp_saddle_kernel<<<nblk, tpb>>>(d_v, d_types, d_orig, d_locks, g, eb, d_w,
                                                 d_list, lo, hi, use_rbf, d_cnt);
                SZP_CUDA_TRY(cudaGetLastError());
            }
            unsigned long long got = 0;
            SZP_CUDA_TRY(cudaMemcpy(&got, d_cnt, sizeof(got), cudaMemcpyDeviceToHost));
            if (use_rbf) st.saddles_by_rbf += (size_t)got;
            else         st.saddles_by_stencil += (size_t)got;
        }
    }

    SZP_CUDA_TRY(cudaMemset(d_cnt, 0, 4 * sizeof(unsigned long long)));
    szp_count_lost_kernel<<<count_blocks, SZP_TPB>>>(d_v, d_types, g, N, d_cnt);
    SZP_CUDA_TRY(cudaGetLastError());
    SZP_CUDA_TRY(cudaMemcpy(cnt, d_cnt, sizeof(cnt), cudaMemcpyDeviceToHost));
    st.extrema_lost_after = (size_t)cnt[1]; st.saddles_lost_after = (size_t)cnt[3];

    SZP_CUDA_TRY(cudaMemcpy(data, d_v, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    if (stats) *stats = st;
    rc = 0;

fail:
    cudaFree(d_v); cudaFree(d_orig); cudaFree(d_tgt); cudaFree(d_types); cudaFree(d_sp); cudaFree(d_rank);
    cudaFree(d_tmax); cudaFree(d_tcnt); cudaFree(d_locks); cudaFree(d_mark); cudaFree(d_w); cudaFree(d_ext);
    cudaFree(d_list); cudaFree(d_cnt); cudaFree(d_bcount); cudaFree(d_tkey); cudaFree(d_slot); cudaFree(d_cc);
    return rc;
}

/* SZP_CUDA_RESTORE=hybrid selects the previous driver (for comparisons). */
static int restore_gpu(float *data, const int *types, const int *sort_positions,
                       size_t extrema_count, const SzpGrid &g, float eb,
                       szp_topo_restore_stats *stats)
{
    const char *mode = getenv("SZP_CUDA_RESTORE");
    if (mode && strcmp(mode, "hybrid") == 0)
        return restore_gpu_hybrid(data, types, sort_positions, extrema_count, g, eb, stats);
    return restore_gpu_device(data, types, sort_positions, extrema_count, g, eb, stats);
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
