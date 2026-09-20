/**
 *  @file szp_cuda_topology.cu
 *  @brief CUDA implementation of topology-preserving compression functions.
 *
 *  Converted from the OpenMP implementation in szp_float.cc.
 *  Covers: critical point finding, bin-based sorting, sort position compression,
 *  and topology-preserved random-access compression.
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
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/count.h>
#include <thrust/copy.h>

#include "szp_cuda_topology.cuh"
#include "szp_cuda_common.cuh"
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
/*  1.  CRITICAL POINT FINDING                                         */
/* ================================================================== */

/**
 * Kernel: one thread per interior grid cell.
 * Writes to a global counter via atomicAdd for stream compaction.
 */
__global__ void find_critical_points_kernel(
    const float *data, int rows, int cols, double inver_bound, float absErrBound,
    CriticalPoint *results, unsigned int *d_count)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_interior = (unsigned int)(rows - 2) * (unsigned int)(cols - 2);
    if (idx >= total_interior) return;

    int i = 1 + (int)(idx / (unsigned int)(cols - 2));
    int j = 1 + (int)(idx % (unsigned int)(cols - 2));

    float center = data[i * cols + j];
    float up     = data[(i - 1) * cols + j];
    float down   = data[(i + 1) * cols + j];
    float left   = data[i * cols + (j - 1)];
    float right  = data[i * cols + (j + 1)];

    int type = 0;
    if (center > up && center > down && center > left && center > right) {
        type = 1;  // local maximum
    } else if (center < up && center < down && center < left && center < right) {
        type = 2;  // local minimum
    } else if ((center < up && center < down && center > left && center > right) ||
               (center > up && center > down && center < left && center < right)) {
        type = 3;  // saddle
    }

    if (type != 0) {
        int quantized_bin = (int)((center + absErrBound) * inver_bound);
        unsigned int pos = atomicAdd(d_count, 1u);
        results[pos].x = i;
        results[pos].y = j;
        results[pos].type = type;
        results[pos].quantized_bin = quantized_bin;
        results[pos].sort_position = 0;
    }
}

extern "C"
CriticalPoint *szp_cuda_find_critical_points(float *data, size_t *outCount,
                                              int rows, int cols, float absErrBound)
{
    if (!data || rows <= 2 || cols <= 2) {
        *outCount = 0;
        return NULL;
    }

    size_t grid_size = (size_t)rows * cols;
    unsigned int total_interior = (unsigned int)(rows - 2) * (unsigned int)(cols - 2);

    /* Allocate device memory */
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, grid_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, data, grid_size * sizeof(float), cudaMemcpyHostToDevice));

    /* Allocate device output (worst-case: every interior cell is a critical point) */
    CriticalPoint *d_results;
    CUDA_CHECK(cudaMalloc(&d_results, total_interior * sizeof(CriticalPoint)));

    unsigned int *d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned int)));

    double inver_bound = 1.0 / absErrBound;

    int threadsPerBlock = 256;
    int blocks = ((int)total_interior + threadsPerBlock - 1) / threadsPerBlock;
    find_critical_points_kernel<<<blocks, threadsPerBlock>>>(
        d_data, rows, cols, inver_bound, absErrBound, d_results, d_count);
    CUDA_CHECK(cudaGetLastError());

    /* Copy count back */
    unsigned int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    CriticalPoint *results = NULL;
    if (h_count > 0) {
        results = (CriticalPoint *)malloc(h_count * sizeof(CriticalPoint));
        CUDA_CHECK(cudaMemcpy(results, d_results, h_count * sizeof(CriticalPoint), cudaMemcpyDeviceToHost));
    }

    *outCount = (size_t)h_count;

    cudaFree(d_data);
    cudaFree(d_results);
    cudaFree(d_count);

    return results;
}

/* ================================================================== */
/*  2.  SORT CRITICAL POINTS BY ORIGINAL DATA WITHIN BINS              */
/* ================================================================== */

/* Device-side key for sort: pack (bin, value) so that Thrust sort groups by bin
   and sorts by value within each bin. */

struct BinValueKey {
    int    bin;
    float  value;
    size_t original_index;
};

struct BinValueCmp {
    __host__ __device__ bool operator()(const BinValueKey &a, const BinValueKey &b) const {
        if (a.bin != b.bin) return a.bin < b.bin;
        return a.value < b.value;
    }
};

__global__ void build_sort_keys_kernel(
    const CriticalPoint *cp, size_t n, const float *data, int cols,
    BinValueKey *keys)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    keys[idx].bin            = cp[idx].quantized_bin;
    keys[idx].value          = data[cp[idx].x * cols + cp[idx].y];
    keys[idx].original_index = idx;
}

/* O(n) parallel boundary marking — replaces the O(n²) backward walk */
__global__ void mark_boundaries_kernel(
    const BinValueKey *sorted_keys, int *boundaries, size_t n)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (idx == 0) {
        boundaries[idx] = 0;
    } else {
        int same_bin   = (sorted_keys[idx].bin == sorted_keys[idx - 1].bin);
        int same_value = (sorted_keys[idx].value == sorted_keys[idx - 1].value);
        if (!same_bin)
            boundaries[idx] = 0;    /* new bin → reset to 0 */
        else if (!same_value)
            boundaries[idx] = 1;    /* new value in same bin → increment */
        else
            boundaries[idx] = 0;    /* same bin, same value → same position */
    }
}

/* O(n) sequential scan to compute cumulative positions within each bin */
__global__ void compute_sort_positions_kernel(
    CriticalPoint *cp, const BinValueKey *sorted_keys,
    const int *boundaries, size_t n)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int pos = 0;
    for (size_t i = 0; i < n; i++) {
        if (i > 0 && sorted_keys[i].bin != sorted_keys[i - 1].bin)
            pos = 0;
        else
            pos += boundaries[i];
        cp[sorted_keys[i].original_index].sort_position = pos;
    }
}

extern "C"
void szp_cuda_sort_critical_points_by_original_data(CriticalPoint *critical_points,
                                                     size_t critical_count,
                                                     float *data, int cols)
{
    if (!critical_points || critical_count == 0 || !data) return;

    /* Find max flat index to determine data copy size */
    size_t max_flat = 0;
    for (size_t i = 0; i < critical_count; i++) {
        size_t flat = (size_t)critical_points[i].x * cols + critical_points[i].y;
        if (flat > max_flat) max_flat = flat;
    }
    size_t data_bytes = (max_flat + 1) * sizeof(float);

    /* Single device allocation */
    size_t cp_bytes   = critical_count * sizeof(CriticalPoint);
    size_t keys_bytes = critical_count * sizeof(BinValueKey);
    size_t bnd_bytes  = critical_count * sizeof(int);
    size_t total_pool = cp_bytes + data_bytes + keys_bytes + bnd_bytes;

    unsigned char *d_pool = NULL;
    CUDA_CHECK(cudaMalloc(&d_pool, total_pool));

    CriticalPoint *d_cp = (CriticalPoint *)d_pool;
    float *d_data        = (float *)(d_pool + cp_bytes);
    BinValueKey *d_keys  = (BinValueKey *)((unsigned char *)d_data + data_bytes);
    int *d_boundaries    = (int *)((unsigned char *)d_keys + keys_bytes);

    CUDA_CHECK(cudaMemcpy(d_cp, critical_points, cp_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data, data, data_bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocks = ((int)critical_count + threadsPerBlock - 1) / threadsPerBlock;

    /* Build keys */
    build_sort_keys_kernel<<<blocks, threadsPerBlock>>>(d_cp, critical_count, d_data, cols, d_keys);
    CUDA_CHECK(cudaGetLastError());

    /* Sort by (bin, value) using Thrust */
    thrust::device_ptr<BinValueKey> keys_ptr(d_keys);
    thrust::sort(keys_ptr, keys_ptr + critical_count, BinValueCmp());

    /* O(n) position assignment: mark boundaries → sequential scan */
    mark_boundaries_kernel<<<blocks, threadsPerBlock>>>(d_keys, d_boundaries, critical_count);
    CUDA_CHECK(cudaGetLastError());

    compute_sort_positions_kernel<<<1, 1>>>(d_cp, d_keys, d_boundaries, critical_count);
    CUDA_CHECK(cudaGetLastError());

    /* Copy back */
    CUDA_CHECK(cudaMemcpy(critical_points, d_cp, cp_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_pool);
}

/* ================================================================== */
/*  3.  COMPRESS SORT POSITIONS                                        */
/* ================================================================== */

/* Each compression block for sort positions uses the same layout as the
   float random-access blocks:
     [first_value: int32] [bit_count: uint8] [sign_bits] [magnitude_bits]
   Uses nChunks=1 in the header.  */

__global__ void compress_sort_positions_sizing_kernel(
    const int *sort_positions, size_t nbEle, int blockSize,
    size_t *block_sizes)
{
    size_t block_id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_blocks = (nbEle + blockSize - 1) / blockSize;
    if (block_id >= num_blocks) return;

    size_t block_start = block_id * blockSize;
    size_t current_block_size = (block_start + blockSize > nbEle) ? (nbEle - block_start) : (size_t)blockSize;

    size_t out_bytes = sizeof(int);  /* first_value */

    if (current_block_size > 1) {
        /* Compute deltas and find max */
        int prior = sort_positions[block_start];
        unsigned int max_val = 0;
        for (size_t j = 1; j < current_block_size; j++) {
            int diff = sort_positions[block_start + j] - prior;
            prior = sort_positions[block_start + j];
            unsigned int abs_diff = (diff < 0) ? (unsigned int)(-diff) : (unsigned int)diff;
            if (abs_diff > max_val) max_val = abs_diff;
        }

        out_bytes += 1;  /* bit_count byte */
        if (max_val > 0) {
            unsigned int bit_count = (unsigned int)floorf(log2f((float)max_val)) + 1u;
            unsigned int sign_bytes = ((unsigned int)(current_block_size - 1) + 7) / 8;
            unsigned int byte_count_per = bit_count / 8;
            unsigned int remainder_bit = bit_count % 8;
            unsigned int n = (unsigned int)(current_block_size - 1);
            unsigned int mag_bytes = byte_count_per * n;
            if (remainder_bit > 0) {
                mag_bytes += (remainder_bit * n + 7) / 8;
            }
            out_bytes += sign_bytes + mag_bytes;
        }
    }

    block_sizes[block_id] = out_bytes;
}

__global__ void compress_sort_positions_packing_kernel(
    const int *sort_positions, size_t nbEle, int blockSize,
    const size_t *block_offsets, unsigned char *output)
{
    size_t block_id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_blocks = (nbEle + blockSize - 1) / blockSize;
    if (block_id >= num_blocks) return;

    size_t block_start = block_id * blockSize;
    size_t current_block_size = (block_start + blockSize > nbEle) ? (nbEle - block_start) : (size_t)blockSize;

    unsigned char *ptr = output + block_offsets[block_id];

    /* Write first value */
    int first_val = sort_positions[block_start];
    memcpy(ptr, &first_val, sizeof(int));
    ptr += sizeof(int);

    if (current_block_size <= 1) return;

    /* Compute deltas */
    unsigned int n = (unsigned int)(current_block_size - 1);
    unsigned char local_signs[256];
    unsigned int  local_mags[256];
    unsigned int  max_val = 0;

    int prior = first_val;
    for (unsigned int j = 0; j < n; j++) {
        int diff = sort_positions[block_start + j + 1] - prior;
        prior = sort_positions[block_start + j + 1];
        if (diff < 0) {
            local_signs[j] = 1;
            local_mags[j]  = (unsigned int)(-diff);
        } else {
            local_signs[j] = 0;
            local_mags[j]  = (unsigned int)diff;
        }
        if (local_mags[j] > max_val) max_val = local_mags[j];
    }

    if (max_val == 0) {
        *ptr++ = 0;
        return;
    }

    unsigned int bit_count = (unsigned int)floorf(log2f((float)max_val)) + 1u;
    *ptr++ = (unsigned char)bit_count;

    /* Pack sign bits (MSB-first) */
    unsigned int sign_bytes = (n + 7) / 8;
    for (unsigned int b = 0; b < sign_bytes; b++) {
        unsigned char tmp = 0;
        for (unsigned int k = 0; k < 8 && b * 8 + k < n; k++) {
            tmp |= (local_signs[b * 8 + k] << (7 - k));
        }
        *ptr++ = tmp;
    }

    /* Pack magnitude bits using Jiajun layout */
    unsigned int byte_count = bit_count / 8;
    unsigned int remainder_bit = bit_count % 8;

    /* Full bytes first (little-endian per element) */
    if (byte_count > 0) {
        for (unsigned int i = 0; i < n; i++) {
            unsigned int val = local_mags[i] >> remainder_bit;
            for (unsigned int j = 0; j < byte_count; j++) {
                ptr[i * byte_count + j] = (unsigned char)(val & 0xFF);
                val >>= 8;
            }
        }
        ptr += byte_count * n;
    }

    /* Remainder bits (MSB-first generic packing) */
    if (remainder_bit > 0) {
        unsigned int total_rem_bytes = (remainder_bit * n + 7) / 8;
        /* Zero output */
        for (unsigned int b = 0; b < total_rem_bytes; b++) ptr[b] = 0;

        unsigned int mask = (1u << remainder_bit) - 1;
        for (unsigned int i = 0; i < n; i++) {
            unsigned int val = local_mags[i] & mask;
            unsigned int bit_start = i * remainder_bit;
            for (unsigned int b = 0; b < remainder_bit; b++) {
                unsigned int global_bit = bit_start + b;
                unsigned int byte_idx = global_bit / 8;
                unsigned int bit_in_byte = 7 - (global_bit % 8);
                unsigned int src_bit = remainder_bit - 1 - b;
                if ((val >> src_bit) & 1) {
                    ptr[byte_idx] |= (1u << bit_in_byte);
                }
            }
        }
        ptr += total_rem_bytes;
    }
}

extern "C"
unsigned char *szp_cuda_compress_sort_positions(CriticalPoint *critical_points,
                                                 size_t critical_count,
                                                 size_t *outSize, int blockSize)
{
    if (!critical_points || critical_count == 0) {
        *outSize = 0;
        return NULL;
    }

    /* Extract sort_position values for extrema (type 1 or 2) */
    size_t extrema_count = 0;
    for (size_t i = 0; i < critical_count; i++) {
        if (critical_points[i].type == 1 || critical_points[i].type == 2)
            extrema_count++;
    }
    if (extrema_count == 0) { *outSize = 0; return NULL; }

    int *sort_positions = (int *)malloc(extrema_count * sizeof(int));
    size_t ei = 0;
    for (size_t i = 0; i < critical_count; i++) {
        if (critical_points[i].type == 1 || critical_points[i].type == 2)
            sort_positions[ei++] = critical_points[i].sort_position;
    }

    /* Copy to device */
    int *d_sort_positions;
    CUDA_CHECK(cudaMalloc(&d_sort_positions, extrema_count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sort_positions, sort_positions, extrema_count * sizeof(int), cudaMemcpyHostToDevice));

    size_t num_blocks = (extrema_count + blockSize - 1) / blockSize;

    /* Sizing pass */
    size_t *d_block_sizes, *d_block_offsets;
    CUDA_CHECK(cudaMalloc(&d_block_sizes,   num_blocks * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, num_blocks * sizeof(size_t)));

    int tpb = 256;
    int grid = ((int)num_blocks + tpb - 1) / tpb;
    compress_sort_positions_sizing_kernel<<<grid, tpb>>>(
        d_sort_positions, extrema_count, blockSize, d_block_sizes);
    CUDA_CHECK(cudaGetLastError());

    /* Prefix sum */
    thrust::device_ptr<size_t> sizes_ptr(d_block_sizes);
    thrust::device_ptr<size_t> offsets_ptr(d_block_offsets);
    thrust::exclusive_scan(sizes_ptr, sizes_ptr + num_blocks, offsets_ptr);

    /* Total size */
    size_t last_size, last_offset;
    CUDA_CHECK(cudaMemcpy(&last_size,   d_block_sizes   + num_blocks - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_offset, d_block_offsets + num_blocks - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    size_t totalCompressed = last_offset + last_size;

    /* Header: nChunks=1, one offset entry (0) */
    size_t nChunks = 1;
    size_t headerSize = nChunks * sizeof(size_t);
    *outSize = headerSize + totalCompressed;

    unsigned char *output = (unsigned char *)malloc(*outSize);
    size_t zero_off = 0;
    memcpy(output, &zero_off, sizeof(size_t));

    /* Packing pass */
    unsigned char *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, totalCompressed));
    compress_sort_positions_packing_kernel<<<grid, tpb>>>(
        d_sort_positions, extrema_count, blockSize, d_block_offsets, d_output);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(output + headerSize, d_output, totalCompressed, cudaMemcpyDeviceToHost));

    /* Cleanup */
    cudaFree(d_sort_positions);
    cudaFree(d_block_sizes);
    cudaFree(d_block_offsets);
    cudaFree(d_output);
    free(sort_positions);

    return output;
}

/* ================================================================== */
/*  4.  TOPOLOGY-PRESERVED RANDOM-ACCESS COMPRESSION                   */
/* ================================================================== */

/* Mark critical point types into a flat array */
__global__ void mark_critical_types_kernel(
    const CriticalPoint *cp, int critical_count, int rows, int cols,
    unsigned char *critical_type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= critical_count) return;

    int x = cp[idx].x, y = cp[idx].y;
    if (x < 0 || x >= rows || y < 0 || y >= cols) return;

    size_t flat = (size_t)x * cols + y;
    unsigned char t = (unsigned char)cp[idx].type;
    if (t >= 1 && t <= 3) {
        critical_type[flat] = t;
    }
}

/* Sizing kernel: compute output bytes per compression block.
   Same as random-access, plus 2-bit type bytes. */
__global__ void compress_topo_sizing_kernel(
    const float *data, size_t nbEle, unsigned int blockSize, double inver_bound,
    size_t *block_sizes)
{
    size_t block_id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_blocks = (nbEle + blockSize - 1) / blockSize;
    if (block_id >= num_blocks) return;

    size_t block_start = block_id * blockSize;
    size_t current_block_size = (block_start + blockSize > nbEle) ? (nbEle - block_start) : blockSize;

    size_t out_bytes = sizeof(int);  /* first_value */

    unsigned int max_val = 0;
    if (current_block_size > 1) {
        int prior = (int)((double)data[block_start] * inver_bound);
        for (size_t j = 1; j < current_block_size; j++) {
            int current = (int)((double)data[block_start + j] * inver_bound);
            int diff = current - prior;
            prior = current;
            unsigned int abs_diff = (diff < 0) ? (unsigned int)(-diff) : (unsigned int)diff;
            if (abs_diff > max_val) max_val = abs_diff;
        }

        out_bytes += 1;  /* bit_count */
        if (max_val > 0) {
            unsigned int bit_count = (unsigned int)floorf(log2f((float)max_val)) + 1u;
            unsigned int n = (unsigned int)(current_block_size - 1);
            unsigned int sign_bytes = (n + 7) / 8;
            unsigned int bc = bit_count / 8;
            unsigned int rb = bit_count % 8;
            unsigned int mag_bytes = bc * n;
            if (rb > 0) mag_bytes += (rb * n + 7) / 8;
            out_bytes += sign_bytes + mag_bytes;
        }
    }

    /* 2-bit type data */
    unsigned int type_bytes = (2 * (unsigned int)current_block_size + 7) / 8;
    out_bytes += type_bytes;

    block_sizes[block_id] = out_bytes;
}

/* Packing kernel for topology-preserved compression */
__global__ void compress_topo_packing_kernel(
    const float *data, size_t nbEle, unsigned int blockSize, double inver_bound,
    const unsigned char *critical_type,
    const size_t *block_offsets, unsigned char *output)
{
    size_t block_id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_blocks = (nbEle + blockSize - 1) / blockSize;
    if (block_id >= num_blocks) return;

    size_t block_start = block_id * blockSize;
    size_t current_block_size = (block_start + blockSize > nbEle) ? (nbEle - block_start) : blockSize;

    unsigned char *ptr = output + block_offsets[block_id];

    /* First value */
    int prior = (int)((double)data[block_start] * inver_bound);
    memcpy(ptr, &prior, sizeof(int));
    ptr += sizeof(int);

    if (current_block_size > 1) {
        unsigned int n = (unsigned int)(current_block_size - 1);
        unsigned char local_signs[256];
        unsigned int  local_mags[256];
        unsigned int  max_val = 0;

        for (unsigned int j = 0; j < n; j++) {
            int current = (int)((double)data[block_start + j + 1] * inver_bound);
            int diff = current - prior;
            prior = current;
            if (diff < 0) {
                local_signs[j] = 1;
                local_mags[j]  = (unsigned int)(-diff);
            } else {
                local_signs[j] = 0;
                local_mags[j]  = (unsigned int)diff;
            }
            if (local_mags[j] > max_val) max_val = local_mags[j];
        }

        if (max_val == 0) {
            *ptr++ = 0;
        } else {
            unsigned int bit_count = (unsigned int)floorf(log2f((float)max_val)) + 1u;
            *ptr++ = (unsigned char)bit_count;

            /* Pack sign bits */
            unsigned int sign_bytes = (n + 7) / 8;
            for (unsigned int b = 0; b < sign_bytes; b++) {
                unsigned char tmp = 0;
                for (unsigned int k = 0; k < 8 && b * 8 + k < n; k++) {
                    tmp |= (local_signs[b * 8 + k] << (7 - k));
                }
                *ptr++ = tmp;
            }

            /* Pack magnitude bits */
            unsigned int byte_count = bit_count / 8;
            unsigned int remainder_bit = bit_count % 8;

            if (byte_count > 0) {
                for (unsigned int i = 0; i < n; i++) {
                    unsigned int val = local_mags[i] >> remainder_bit;
                    for (unsigned int j = 0; j < byte_count; j++) {
                        ptr[i * byte_count + j] = (unsigned char)(val & 0xFF);
                        val >>= 8;
                    }
                }
                ptr += byte_count * n;
            }

            if (remainder_bit > 0) {
                unsigned int total_rem_bytes = (remainder_bit * n + 7) / 8;
                for (unsigned int b = 0; b < total_rem_bytes; b++) ptr[b] = 0;

                unsigned int mask = (1u << remainder_bit) - 1;
                for (unsigned int i = 0; i < n; i++) {
                    unsigned int val = local_mags[i] & mask;
                    unsigned int bit_start = i * remainder_bit;
                    for (unsigned int b = 0; b < remainder_bit; b++) {
                        unsigned int global_bit = bit_start + b;
                        unsigned int byte_idx = global_bit / 8;
                        unsigned int bit_in_byte = 7 - (global_bit % 8);
                        unsigned int src_bit = remainder_bit - 1 - b;
                        if ((val >> src_bit) & 1) {
                            ptr[byte_idx] |= (1u << bit_in_byte);
                        }
                    }
                }
                ptr += total_rem_bytes;
            }
        }
    }

    /* Pack 2-bit critical type data for this block (MSB-first, 4 per byte) */
    unsigned int type_bytes = (2 * (unsigned int)current_block_size + 7) / 8;
    for (unsigned int b = 0; b < type_bytes; b++) ptr[b] = 0;

    for (unsigned int j = 0; j < (unsigned int)current_block_size; j++) {
        unsigned char t = critical_type[block_start + j];
        unsigned int byte_idx = j / 4;
        unsigned int shift = 6 - 2 * (j % 4);
        ptr[byte_idx] |= (t << shift);
    }
}

extern "C"
unsigned char *szp_cuda_float_compress_randomaccess_topology_preserved(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize,
    CriticalPoint *critical_points, int critical_count,
    int rows, int cols)
{
    if (absErrBound <= 0.0 || !oriData || nbEle == 0) {
        *outSize = 0;
        return NULL;
    }

    double inver_bound = 1.0 / absErrBound;
    unsigned int block_size = (unsigned int)blockSize;
    size_t num_blocks = (nbEle + block_size - 1) / block_size;

    /* Device allocations */
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, nbEle * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, oriData, nbEle * sizeof(float), cudaMemcpyHostToDevice));

    /* Build critical type array on device */
    unsigned char *d_critical_type;
    CUDA_CHECK(cudaMalloc(&d_critical_type, nbEle));
    CUDA_CHECK(cudaMemset(d_critical_type, 0, nbEle));

    if (critical_points && critical_count > 0) {
        CriticalPoint *d_cp;
        CUDA_CHECK(cudaMalloc(&d_cp, critical_count * sizeof(CriticalPoint)));
        CUDA_CHECK(cudaMemcpy(d_cp, critical_points, critical_count * sizeof(CriticalPoint), cudaMemcpyHostToDevice));

        int tpb = 256;
        int grid = (critical_count + tpb - 1) / tpb;
        mark_critical_types_kernel<<<grid, tpb>>>(d_cp, critical_count, rows, cols, d_critical_type);
        CUDA_CHECK(cudaGetLastError());
        cudaFree(d_cp);
    }

    /* Sizing pass */
    size_t *d_block_sizes, *d_block_offsets;
    CUDA_CHECK(cudaMalloc(&d_block_sizes,   num_blocks * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, num_blocks * sizeof(size_t)));

    int tpb = 256;
    int grid = ((int)num_blocks + tpb - 1) / tpb;
    compress_topo_sizing_kernel<<<grid, tpb>>>(d_data, nbEle, block_size, inver_bound, d_block_sizes);
    CUDA_CHECK(cudaGetLastError());

    /* Prefix sum */
    thrust::device_ptr<size_t> sizes_ptr(d_block_sizes);
    thrust::device_ptr<size_t> offsets_ptr(d_block_offsets);
    thrust::exclusive_scan(sizes_ptr, sizes_ptr + num_blocks, offsets_ptr);

    /* Total size */
    size_t last_size, last_offset;
    CUDA_CHECK(cudaMemcpy(&last_size,   d_block_sizes   + num_blocks - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_offset, d_block_offsets + num_blocks - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    size_t totalCompressed = last_offset + last_size;

    /* Header: nChunks=1, one offset (0) */
    size_t nChunks = 1;
    size_t headerSize = nChunks * sizeof(size_t);
    *outSize = headerSize + totalCompressed;

    unsigned char *output = (unsigned char *)malloc(*outSize);
    size_t zero_off = 0;
    memcpy(output, &zero_off, sizeof(size_t));

    /* Packing pass */
    unsigned char *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, totalCompressed));
    compress_topo_packing_kernel<<<grid, tpb>>>(
        d_data, nbEle, block_size, inver_bound, d_critical_type, d_block_offsets, d_output);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(output + headerSize, d_output, totalCompressed, cudaMemcpyDeviceToHost));

    /* Cleanup */
    cudaFree(d_data);
    cudaFree(d_critical_type);
    cudaFree(d_block_sizes);
    cudaFree(d_block_offsets);
    cudaFree(d_output);

    return output;
}
