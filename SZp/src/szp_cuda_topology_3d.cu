/**
 *  @file szp_cuda_topology_3d.cu
 *  @brief CUDA implementation of 3D topology-preserving compression functions.
 *
 *  Extension of the 2D topology code (szp_cuda_topology.cu) to 3D grids
 *  using 6-connected (face-adjacent) neighborhood: ±x, ±y, ±z.
 *
 *  Critical point classification (6-connected):
 *    max:    center > all 6 face-neighbors
 *    min:    center < all 6 face-neighbors
 *    saddle: high along some axes and low along others
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "szp_cuda_topology_3d.cuh"
#include "szp_defines.h"

/* ------------------------------------------------------------------ */
/*  Helper: CUDA error checking                                        */
/* ------------------------------------------------------------------ */
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)
#endif

/* ================================================================== */
/*  1.  3D CRITICAL POINT FINDING                                      */
/* ================================================================== */

/**
 * Kernel: one thread per interior voxel.
 * Interior = x in [1, d1-2], y in [1, d2-2], z in [1, d3-2].
 * Uses 6-connected (face-adjacent) neighborhood.
 */
__global__ void find_critical_points_3d_kernel(
    const float *data, int d1, int d2, int d3,
    double inver_bound, float absErrBound,
    CriticalPoint3D *results, unsigned int *d_count)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int d1m2 = (unsigned int)(d1 - 2);
    unsigned int d2m2 = (unsigned int)(d2 - 2);
    unsigned int d3m2 = (unsigned int)(d3 - 2);
    unsigned int total_interior = d1m2 * d2m2 * d3m2;
    if (idx >= total_interior) return;

    /* Convert flat interior index to 3D coordinates */
    unsigned int plane = d2m2 * d3m2;
    int x = 1 + (int)(idx / plane);
    unsigned int rem = idx % plane;
    int y = 1 + (int)(rem / d3m2);
    int z = 1 + (int)(rem % d3m2);

    /* Flat index into data array */
    int sd2d3 = d2 * d3;
    size_t flat = (size_t)x * sd2d3 + (size_t)y * d3 + z;

    float center = data[flat];
    float xm = data[flat - sd2d3];       /* x-1 */
    float xp = data[flat + sd2d3];       /* x+1 */
    float ym = data[flat - d3];          /* y-1 */
    float yp = data[flat + d3];          /* y+1 */
    float zm = data[flat - 1];           /* z-1 */
    float zp = data[flat + 1];           /* z+1 */

    int type = 0;
    if (center > xm && center > xp &&
        center > ym && center > yp &&
        center > zm && center > zp) {
        type = 1;   /* local maximum */
    } else if (center < xm && center < xp &&
               center < ym && center < yp &&
               center < zm && center < zp) {
        type = 2;   /* local minimum */
    } else {
        /* Saddle detection: check axis-aligned directional behaviour.
           A saddle exists when the center is a local extremum along
           some axes and the opposite along other axes. */
        int x_high = (center > xm) && (center > xp);
        int x_low  = (center < xm) && (center < xp);
        int y_high = (center > ym) && (center > yp);
        int y_low  = (center < ym) && (center < yp);
        int z_high = (center > zm) && (center > zp);
        int z_low  = (center < zm) && (center < zp);

        int high_axes = x_high + y_high + z_high;
        int low_axes  = x_low  + y_low  + z_low;

        if (high_axes >= 1 && low_axes >= 1) {
            type = 3;   /* saddle */
        }
    }

    if (type != 0) {
        int quantized_bin = (int)((center + absErrBound) * inver_bound);
        unsigned int pos = atomicAdd(d_count, 1u);
        results[pos].x = x;
        results[pos].y = y;
        results[pos].z = z;
        results[pos].type = type;
        results[pos].quantized_bin = quantized_bin;
        results[pos].sort_position = 0;
    }
}

extern "C"
CriticalPoint3D *szp_cuda_find_critical_points_3d(
    float *data, size_t *outCount,
    int d1, int d2, int d3, float absErrBound)
{
    if (!data || d1 <= 2 || d2 <= 2 || d3 <= 2) {
        *outCount = 0;
        return NULL;
    }

    size_t grid_size = (size_t)d1 * d2 * d3;
    unsigned int total_interior = (unsigned int)(d1 - 2)
                                * (unsigned int)(d2 - 2)
                                * (unsigned int)(d3 - 2);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, grid_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, data, grid_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    CriticalPoint3D *d_results;
    CUDA_CHECK(cudaMalloc(&d_results, total_interior * sizeof(CriticalPoint3D)));

    unsigned int *d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned int)));

    double inver_bound = 1.0 / absErrBound;

    int tpb = 256;
    int blocks = ((int)total_interior + tpb - 1) / tpb;
    find_critical_points_3d_kernel<<<blocks, tpb>>>(
        d_data, d1, d2, d3, inver_bound, absErrBound,
        d_results, d_count);
    CUDA_CHECK(cudaGetLastError());

    unsigned int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    CriticalPoint3D *results = NULL;
    if (h_count > 0) {
        results = (CriticalPoint3D *)malloc(h_count * sizeof(CriticalPoint3D));
        CUDA_CHECK(cudaMemcpy(results, d_results,
                              h_count * sizeof(CriticalPoint3D),
                              cudaMemcpyDeviceToHost));
    }

    *outCount = (size_t)h_count;

    cudaFree(d_data);
    cudaFree(d_results);
    cudaFree(d_count);

    return results;
}

/* ================================================================== */
/*  2.  SORT 3D CRITICAL POINTS BY ORIGINAL DATA WITHIN BINS           */
/* ================================================================== */

struct BinValueKey3D {
    int    bin;
    float  value;
    size_t original_index;
};

struct BinValueCmp3D {
    __host__ __device__
    bool operator()(const BinValueKey3D &a, const BinValueKey3D &b) const {
        if (a.bin != b.bin) return a.bin < b.bin;
        return a.value < b.value;
    }
};

__global__ void build_sort_keys_3d_kernel(
    const CriticalPoint3D *cp, size_t n,
    const float *data, int d2, int d3,
    BinValueKey3D *keys)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    size_t flat = (size_t)cp[idx].x * d2 * d3
                + (size_t)cp[idx].y * d3
                + cp[idx].z;
    keys[idx].bin            = cp[idx].quantized_bin;
    keys[idx].value          = data[flat];
    keys[idx].original_index = idx;
}

__global__ void assign_sort_positions_3d_kernel(
    CriticalPoint3D *cp, const BinValueKey3D *sorted_keys, size_t n)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int pos = 0;
    if (idx > 0 && sorted_keys[idx].bin == sorted_keys[idx - 1].bin) {
        if (sorted_keys[idx].value != sorted_keys[idx - 1].value) {
            /* Count distinct values before this one in the bin */
            size_t k = idx;
            while (k > 0 && sorted_keys[k - 1].bin == sorted_keys[idx].bin) {
                if (sorted_keys[k].value != sorted_keys[k - 1].value)
                    pos++;
                k--;
            }
        } else {
            /* Same value — find first element with this value in this bin */
            size_t k = idx - 1;
            while (k > 0 && sorted_keys[k - 1].bin == sorted_keys[idx].bin &&
                   sorted_keys[k - 1].value == sorted_keys[idx].value) {
                k--;
            }
            pos = 0;
            size_t m = k;
            while (m > 0 && sorted_keys[m - 1].bin == sorted_keys[idx].bin) {
                if (sorted_keys[m].value != sorted_keys[m - 1].value)
                    pos++;
                m--;
            }
        }
    }

    cp[sorted_keys[idx].original_index].sort_position = pos;
}

extern "C"
void szp_cuda_sort_critical_points_3d(
    CriticalPoint3D *critical_points, size_t critical_count,
    float *data, int d2, int d3)
{
    if (!critical_points || critical_count == 0 || !data) return;

    CriticalPoint3D *d_cp;
    float *d_data;
    BinValueKey3D *d_keys;

    CUDA_CHECK(cudaMalloc(&d_cp, critical_count * sizeof(CriticalPoint3D)));

    /* Find max flat index to determine how much data to copy */
    size_t max_flat = 0;
    for (size_t i = 0; i < critical_count; i++) {
        size_t flat = (size_t)critical_points[i].x * d2 * d3
                    + (size_t)critical_points[i].y * d3
                    + critical_points[i].z;
        if (flat > max_flat) max_flat = flat;
    }

    CUDA_CHECK(cudaMalloc(&d_data, (max_flat + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, data, (max_flat + 1) * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cp, critical_points,
                          critical_count * sizeof(CriticalPoint3D),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_keys, critical_count * sizeof(BinValueKey3D)));

    int tpb = 256;
    int blocks = ((int)critical_count + tpb - 1) / tpb;

    build_sort_keys_3d_kernel<<<blocks, tpb>>>(
        d_cp, critical_count, d_data, d2, d3, d_keys);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<BinValueKey3D> keys_ptr(d_keys);
    thrust::sort(keys_ptr, keys_ptr + critical_count, BinValueCmp3D());

    assign_sort_positions_3d_kernel<<<blocks, tpb>>>(
        d_cp, d_keys, critical_count);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(critical_points, d_cp,
                          critical_count * sizeof(CriticalPoint3D),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_cp);
    cudaFree(d_data);
    cudaFree(d_keys);
}

/* ================================================================== */
/*  3.  COMPRESS SORT POSITIONS (3D)                                   */
/* ================================================================== */

/*
 * The sort-position compression kernels are identical to the 2D versions
 * because they operate on a flat int array of sort_position values.
 * Only the extraction (which CPs are extrema) differs, using CriticalPoint3D.
 * We re-declare the device kernels here (static linkage) to avoid ODR issues.
 */

__global__ static void compress_sort_positions_sizing_3d(
    const int *sort_positions, size_t nbEle, int blockSize,
    size_t *block_sizes)
{
    size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_blocks = (nbEle + blockSize - 1) / blockSize;
    if (bid >= num_blocks) return;

    size_t bs = bid * blockSize;
    size_t cur = (bs + blockSize > nbEle) ? (nbEle - bs) : (size_t)blockSize;
    size_t out = sizeof(int);

    if (cur > 1) {
        int prior = sort_positions[bs];
        unsigned int mx = 0;
        for (size_t j = 1; j < cur; j++) {
            int d = sort_positions[bs + j] - prior;
            prior = sort_positions[bs + j];
            unsigned int ad = (d < 0) ? (unsigned int)(-d) : (unsigned int)d;
            if (ad > mx) mx = ad;
        }
        out += 1;
        if (mx > 0) {
            unsigned int bc = (unsigned int)floorf(log2f((float)mx)) + 1u;
            unsigned int n = (unsigned int)(cur - 1);
            unsigned int sb = (n + 7) / 8;
            unsigned int bcc = bc / 8, rb = bc % 8;
            unsigned int mb = bcc * n;
            if (rb > 0) mb += (rb * n + 7) / 8;
            out += sb + mb;
        }
    }
    block_sizes[bid] = out;
}

__global__ static void compress_sort_positions_packing_3d(
    const int *sort_positions, size_t nbEle, int blockSize,
    const size_t *block_offsets, unsigned char *output)
{
    size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_blocks = (nbEle + blockSize - 1) / blockSize;
    if (bid >= num_blocks) return;

    size_t bs = bid * blockSize;
    size_t cur = (bs + blockSize > nbEle) ? (nbEle - bs) : (size_t)blockSize;
    unsigned char *ptr = output + block_offsets[bid];

    int fv = sort_positions[bs];
    memcpy(ptr, &fv, sizeof(int));
    ptr += sizeof(int);
    if (cur <= 1) return;

    unsigned int n = (unsigned int)(cur - 1);
    unsigned char lsigns[256];
    unsigned int  lmags[256];
    unsigned int mx = 0;
    int prior = fv;
    for (unsigned int j = 0; j < n; j++) {
        int d = sort_positions[bs + j + 1] - prior;
        prior = sort_positions[bs + j + 1];
        if (d < 0) { lsigns[j] = 1; lmags[j] = (unsigned int)(-d); }
        else       { lsigns[j] = 0; lmags[j] = (unsigned int)d; }
        if (lmags[j] > mx) mx = lmags[j];
    }
    if (mx == 0) { *ptr++ = 0; return; }

    unsigned int bc = (unsigned int)floorf(log2f((float)mx)) + 1u;
    *ptr++ = (unsigned char)bc;

    /* sign bits MSB-first */
    unsigned int sb = (n + 7) / 8;
    for (unsigned int b = 0; b < sb; b++) {
        unsigned char t = 0;
        for (unsigned int k = 0; k < 8 && b * 8 + k < n; k++)
            t |= (lsigns[b * 8 + k] << (7 - k));
        *ptr++ = t;
    }

    /* magnitude bits — Jiajun layout */
    unsigned int bcc = bc / 8, rb = bc % 8;
    if (bcc > 0) {
        for (unsigned int i = 0; i < n; i++) {
            unsigned int v = lmags[i] >> rb;
            for (unsigned int j = 0; j < bcc; j++) {
                ptr[i * bcc + j] = (unsigned char)(v & 0xFF);
                v >>= 8;
            }
        }
        ptr += bcc * n;
    }
    if (rb > 0) {
        unsigned int trb = (rb * n + 7) / 8;
        for (unsigned int b = 0; b < trb; b++) ptr[b] = 0;
        unsigned int mask = (1u << rb) - 1;
        for (unsigned int i = 0; i < n; i++) {
            unsigned int v = lmags[i] & mask;
            unsigned int bstart = i * rb;
            for (unsigned int b = 0; b < rb; b++) {
                unsigned int gb = bstart + b;
                unsigned int bi = gb / 8;
                unsigned int bp = 7 - (gb % 8);
                unsigned int sbit = rb - 1 - b;
                if ((v >> sbit) & 1) ptr[bi] |= (1u << bp);
            }
        }
        ptr += trb;
    }
}

extern "C"
unsigned char *szp_cuda_compress_sort_positions_3d(
    CriticalPoint3D *critical_points, size_t critical_count,
    size_t *outSize, int blockSize)
{
    if (!critical_points || critical_count == 0) {
        *outSize = 0;
        return NULL;
    }

    /* Extract sort_position values for extrema (type 1 or 2) */
    size_t extrema_count = 0;
    for (size_t i = 0; i < critical_count; i++)
        if (critical_points[i].type == 1 || critical_points[i].type == 2)
            extrema_count++;
    if (extrema_count == 0) { *outSize = 0; return NULL; }

    int *sp = (int *)malloc(extrema_count * sizeof(int));
    size_t ei = 0;
    for (size_t i = 0; i < critical_count; i++)
        if (critical_points[i].type == 1 || critical_points[i].type == 2)
            sp[ei++] = critical_points[i].sort_position;

    int *d_sp;
    CUDA_CHECK(cudaMalloc(&d_sp, extrema_count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sp, sp, extrema_count * sizeof(int),
                          cudaMemcpyHostToDevice));

    size_t nb = (extrema_count + blockSize - 1) / blockSize;
    size_t *d_bs, *d_bo;
    CUDA_CHECK(cudaMalloc(&d_bs, nb * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_bo, nb * sizeof(size_t)));

    int tpb = 256;
    int grid = ((int)nb + tpb - 1) / tpb;
    compress_sort_positions_sizing_3d<<<grid, tpb>>>(d_sp, extrema_count,
                                                      blockSize, d_bs);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<size_t> sp_ptr(d_bs);
    thrust::device_ptr<size_t> op_ptr(d_bo);
    thrust::exclusive_scan(sp_ptr, sp_ptr + nb, op_ptr);

    size_t lsz, loff;
    CUDA_CHECK(cudaMemcpy(&lsz, d_bs + nb - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&loff, d_bo + nb - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    size_t total = loff + lsz;

    size_t hdr = sizeof(size_t);
    *outSize = hdr + total;
    unsigned char *out = (unsigned char *)malloc(*outSize);
    size_t zero = 0;
    memcpy(out, &zero, sizeof(size_t));

    unsigned char *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, total));
    compress_sort_positions_packing_3d<<<grid, tpb>>>(d_sp, extrema_count,
                                                       blockSize, d_bo, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out + hdr, d_out, total, cudaMemcpyDeviceToHost));

    cudaFree(d_sp); cudaFree(d_bs); cudaFree(d_bo); cudaFree(d_out);
    free(sp);
    return out;
}

/* ================================================================== */
/*  4.  TOPOLOGY-PRESERVED 3D COMPRESSION                             */
/* ================================================================== */

/* Mark 3D critical point types into a flat array */
__global__ void mark_critical_types_3d_kernel(
    const CriticalPoint3D *cp, int critical_count,
    int d1, int d2, int d3,
    unsigned char *critical_type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= critical_count) return;

    int x = cp[idx].x, y = cp[idx].y, z = cp[idx].z;
    if (x < 0 || x >= d1 || y < 0 || y >= d2 || z < 0 || z >= d3) return;

    size_t flat = (size_t)x * d2 * d3 + (size_t)y * d3 + z;
    unsigned char t = (unsigned char)cp[idx].type;
    if (t >= 1 && t <= 3) {
        critical_type[flat] = t;
    }
}

/*
 * The sizing and packing kernels for topology-preserved compression are
 * identical to the 2D versions — they work on flat 1D data with a per-element
 * critical_type array.  Re-declare with static linkage to avoid ODR conflicts.
 */

__global__ static void compress_topo_sizing_3d(
    const float *data, size_t nbEle, unsigned int blockSize, double inver_bound,
    size_t *block_sizes)
{
    size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_blocks = (nbEle + blockSize - 1) / blockSize;
    if (bid >= num_blocks) return;

    size_t bs = bid * blockSize;
    size_t cur = (bs + blockSize > nbEle) ? (nbEle - bs) : (size_t)blockSize;
    size_t out = sizeof(int);

    if (cur > 1) {
        int prior = (int)((double)data[bs] * inver_bound);
        unsigned int mx = 0;
        for (size_t j = 1; j < cur; j++) {
            int c = (int)((double)data[bs + j] * inver_bound);
            int d = c - prior; prior = c;
            unsigned int ad = (d < 0) ? (unsigned int)(-d) : (unsigned int)d;
            if (ad > mx) mx = ad;
        }
        out += 1;
        if (mx > 0) {
            unsigned int bc = (unsigned int)floorf(log2f((float)mx)) + 1u;
            unsigned int n = (unsigned int)(cur - 1);
            unsigned int sb = (n + 7) / 8;
            unsigned int bcc = bc / 8, rb = bc % 8;
            unsigned int mb = bcc * n;
            if (rb > 0) mb += (rb * n + 7) / 8;
            out += sb + mb;
        }
    }

    unsigned int tb = (2 * (unsigned int)cur + 7) / 8;
    out += tb;
    block_sizes[bid] = out;
}

__global__ static void compress_topo_packing_3d(
    const float *data, size_t nbEle, unsigned int blockSize, double inver_bound,
    const unsigned char *critical_type,
    const size_t *block_offsets, unsigned char *output)
{
    size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_blocks = (nbEle + blockSize - 1) / blockSize;
    if (bid >= num_blocks) return;

    size_t bs = bid * blockSize;
    size_t cur = (bs + blockSize > nbEle) ? (nbEle - bs) : (size_t)blockSize;
    unsigned char *ptr = output + block_offsets[bid];

    int prior = (int)((double)data[bs] * inver_bound);
    memcpy(ptr, &prior, sizeof(int));
    ptr += sizeof(int);

    if (cur > 1) {
        unsigned int n = (unsigned int)(cur - 1);
        unsigned char lsigns[256];
        unsigned int  lmags[256];
        unsigned int mx = 0;

        for (unsigned int j = 0; j < n; j++) {
            int c = (int)((double)data[bs + j + 1] * inver_bound);
            int d = c - prior; prior = c;
            if (d < 0) { lsigns[j] = 1; lmags[j] = (unsigned int)(-d); }
            else       { lsigns[j] = 0; lmags[j] = (unsigned int)d; }
            if (lmags[j] > mx) mx = lmags[j];
        }

        if (mx == 0) {
            *ptr++ = 0;
        } else {
            unsigned int bc = (unsigned int)floorf(log2f((float)mx)) + 1u;
            *ptr++ = (unsigned char)bc;

            unsigned int sb = (n + 7) / 8;
            for (unsigned int b = 0; b < sb; b++) {
                unsigned char t = 0;
                for (unsigned int k = 0; k < 8 && b * 8 + k < n; k++)
                    t |= (lsigns[b * 8 + k] << (7 - k));
                *ptr++ = t;
            }

            unsigned int bcc = bc / 8, rb = bc % 8;
            if (bcc > 0) {
                for (unsigned int i = 0; i < n; i++) {
                    unsigned int v = lmags[i] >> rb;
                    for (unsigned int j = 0; j < bcc; j++) {
                        ptr[i * bcc + j] = (unsigned char)(v & 0xFF);
                        v >>= 8;
                    }
                }
                ptr += bcc * n;
            }
            if (rb > 0) {
                unsigned int trb = (rb * n + 7) / 8;
                for (unsigned int b = 0; b < trb; b++) ptr[b] = 0;
                unsigned int mask = (1u << rb) - 1;
                for (unsigned int i = 0; i < n; i++) {
                    unsigned int v = lmags[i] & mask;
                    unsigned int bstart = i * rb;
                    for (unsigned int b = 0; b < rb; b++) {
                        unsigned int gb = bstart + b;
                        unsigned int bi = gb / 8;
                        unsigned int bp = 7 - (gb % 8);
                        unsigned int sbit = rb - 1 - b;
                        if ((v >> sbit) & 1) ptr[bi] |= (1u << bp);
                    }
                }
                ptr += trb;
            }
        }
    }

    /* Pack 2-bit critical type data (MSB-first, 4 per byte) */
    unsigned int tb = (2 * (unsigned int)cur + 7) / 8;
    for (unsigned int b = 0; b < tb; b++) ptr[b] = 0;
    for (unsigned int j = 0; j < (unsigned int)cur; j++) {
        unsigned char t = critical_type[bs + j];
        unsigned int bi = j / 4;
        unsigned int sh = 6 - 2 * (j % 4);
        ptr[bi] |= (t << sh);
    }
}

extern "C"
unsigned char *szp_cuda_float_compress_topology_3d(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize,
    CriticalPoint3D *critical_points, int critical_count,
    int d1, int d2, int d3)
{
    if (absErrBound <= 0.0 || !oriData || nbEle == 0) {
        *outSize = 0;
        return NULL;
    }

    double inver_bound = 1.0 / absErrBound;
    unsigned int bsz = (unsigned int)blockSize;
    size_t num_blocks = (nbEle + bsz - 1) / bsz;

    /* Device allocations */
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, nbEle * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, oriData, nbEle * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* Build critical type array */
    unsigned char *d_ct;
    CUDA_CHECK(cudaMalloc(&d_ct, nbEle));
    CUDA_CHECK(cudaMemset(d_ct, 0, nbEle));

    if (critical_points && critical_count > 0) {
        CriticalPoint3D *d_cp;
        CUDA_CHECK(cudaMalloc(&d_cp, critical_count * sizeof(CriticalPoint3D)));
        CUDA_CHECK(cudaMemcpy(d_cp, critical_points,
                              critical_count * sizeof(CriticalPoint3D),
                              cudaMemcpyHostToDevice));

        int tpb = 256;
        int grid = (critical_count + tpb - 1) / tpb;
        mark_critical_types_3d_kernel<<<grid, tpb>>>(
            d_cp, critical_count, d1, d2, d3, d_ct);
        CUDA_CHECK(cudaGetLastError());
        cudaFree(d_cp);
    }

    /* Sizing pass */
    size_t *d_bs, *d_bo;
    CUDA_CHECK(cudaMalloc(&d_bs, num_blocks * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_bo, num_blocks * sizeof(size_t)));

    int tpb = 256;
    int grid = ((int)num_blocks + tpb - 1) / tpb;
    compress_topo_sizing_3d<<<grid, tpb>>>(
        d_data, nbEle, bsz, inver_bound, d_bs);
    CUDA_CHECK(cudaGetLastError());

    /* Prefix sum */
    thrust::device_ptr<size_t> sp(d_bs);
    thrust::device_ptr<size_t> op(d_bo);
    thrust::exclusive_scan(sp, sp + num_blocks, op);

    size_t lsz, loff;
    CUDA_CHECK(cudaMemcpy(&lsz, d_bs + num_blocks - 1,
                          sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&loff, d_bo + num_blocks - 1,
                          sizeof(size_t), cudaMemcpyDeviceToHost));
    size_t total = loff + lsz;

    size_t hdr = sizeof(size_t);
    *outSize = hdr + total;
    unsigned char *output = (unsigned char *)malloc(*outSize);
    size_t zero = 0;
    memcpy(output, &zero, sizeof(size_t));

    /* Packing pass */
    unsigned char *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, total));
    compress_topo_packing_3d<<<grid, tpb>>>(
        d_data, nbEle, bsz, inver_bound, d_ct, d_bo, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(output + hdr, d_out, total, cudaMemcpyDeviceToHost));

    cudaFree(d_data);
    cudaFree(d_ct);
    cudaFree(d_bs);
    cudaFree(d_bo);
    cudaFree(d_out);

    return output;
}
