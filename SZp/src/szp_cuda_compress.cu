/**
 *  @file szp_cuda_compress.cu
 *  @brief CUDA implementations of all SZp compression functions.
 *
 *  Byte-compatible with the OpenMP format produced by szp_float.cc / szp_double.cc.
 *  Uses nChunks = 1 in the offset table — decompress with CUDA or OMP_NUM_THREADS=1.
 *
 *  Two-pass approach for random-access compression:
 *    Pass 1  – sizing kernel: compute per-block compressed sizes
 *    Scan    – thrust::exclusive_scan on sizes → offsets
 *    Pass 2  – packing kernel: write packed output at computed offsets
 *
 *  For threadblock (non-random-access) compression the entire chunk is processed
 *  sequentially by a single CUDA thread (same serial dependency as the OpenMP version).
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "szp_cuda_compress.cuh"
#include "szp_defines.h"

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Helpers                                                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
        }                                                                      \
    } while (0)

/* Maximum compression-block size we support in local arrays.                 */
#define SZP_CUDA_MAX_BLOCK 1024

/* ─── Device: MSB-first 1-bit sign packing (matches convertIntArray2ByteArray_fast_1b_args) ── */
__device__ static unsigned int
pack_signs_1b(const unsigned char *signs, unsigned int n, unsigned char *out)
{
    unsigned int byteLen = (n + 7) / 8;
    for (unsigned int i = 0; i < byteLen; i++) {
        unsigned char tmp = 0;
        for (unsigned int j = 0; j < 8 && i * 8 + j < n; j++)
            tmp |= (signs[i * 8 + j] & 1) << (7 - j);
        out[i] = tmp;
    }
    return byteLen;
}

/* ─── Device: generic Nb-bit MSB-first packing ──────────────────────────── */
/*  Matches the Jiajun_convertUInt2Byte_fast_Nb_args family.                  */
__device__ static unsigned int
pack_Nb_bits(const unsigned int *values, unsigned int n,
             unsigned char *out, unsigned int bits)
{
    unsigned int total_bits  = n * bits;
    unsigned int total_bytes = (total_bits + 7) / 8;
    for (unsigned int i = 0; i < total_bytes; i++) out[i] = 0;

    for (unsigned int k = 0; k < n; k++) {
        unsigned int val = values[k] & ((1u << bits) - 1);
        unsigned int bit_start = k * bits;
        for (unsigned int b = 0; b < bits; b++) {
            unsigned int gbit     = bit_start + b;
            unsigned int byte_idx = gbit / 8;
            unsigned int bit_pos  = 7 - (gbit % 8);          /* MSB first */
            unsigned int src_bit  = bits - 1 - b;            /* MSB of val first */
            if ((val >> src_bit) & 1)
                out[byte_idx] |= (1u << bit_pos);
        }
    }
    return total_bytes;
}

/* ─── Device: Jiajun_save_fixed_length_bits equivalent ───────────────────── */
/*  Layout: [full-byte part per element, LE] [remainder-bit part, Nb-packed]  */
__device__ static unsigned int
save_fixed_length_bits(unsigned int *values, unsigned int n,
                       unsigned char *out, unsigned int bit_count)
{
    unsigned int byte_count    = bit_count / 8;
    unsigned int remainder_bit = bit_count % 8;
    unsigned int byte_offset   = byte_count * n;
    unsigned int byteLength;
    if (remainder_bit == 0)
        byteLength = byte_offset;
    else
        byteLength = byte_count * n + (remainder_bit * n + 7) / 8;

    /* --- full-byte portion (little-endian per element, upper bits) --- */
    if (byte_count > 0) {
        unsigned int idx = 0;
        for (unsigned int i = 0; i < n; i++) {
            unsigned int val = values[i] >> remainder_bit;
            for (unsigned int j = 0; j < byte_count; j++) {
                out[idx++] = (unsigned char)(val & 0xFF);
                val >>= 8;
            }
        }
    }

    /* --- remainder-bit portion (lower bits, Nb-packed MSB-first) --- */
    if (remainder_bit > 0) {
        /* Mask to lower remainder_bit bits — done in-place (mirrors CPU code
           which mutates the array before dispatching to Nb packer).  The values
           array lives in thread-local memory so this is safe.  */
        if (byte_count > 0) {
            unsigned int mask = (1u << remainder_bit) - 1;
            for (unsigned int i = 0; i < n; i++)
                values[i] &= mask;
        }
        pack_Nb_bits(values, n, out + byte_offset, remainder_bit);
    }

    return byteLength;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Random-access compression kernels (templated for float / double)          */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* ─── Pass 1: compute per-block compressed size ─────────────────────────── */
template <typename T>
__global__ void
ra_compress_sizing_kernel(const T *data, size_t nbEle, int blockSize,
                          double inver_bound, size_t *blockSizes)
{
    size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t numBlocks = (nbEle + blockSize - 1) / blockSize;
    if (bid >= numBlocks) return;

    size_t block_start = bid * blockSize;
    size_t current_block_size = (block_start + blockSize > nbEle)
                                    ? (nbEle - block_start)
                                    : (size_t)blockSize;

    /* first_value (4 bytes for float, also 4 for int representation) */
    size_t sz = sizeof(int);

    if (current_block_size <= 1) {
        blockSizes[bid] = sz;
        return;
    }

    unsigned int n = (unsigned int)(current_block_size - 1);

    /* Quantize + delta + find max */
    int prior = (int)(data[block_start] * inver_bound);
    unsigned int maxv = 0;
    for (unsigned int j = 0; j < n; j++) {
        int cur = (int)(data[block_start + j + 1] * inver_bound);
        int diff = cur - prior;
        prior = cur;
        unsigned int ad = (diff < 0) ? (unsigned int)(-diff) : (unsigned int)diff;
        if (ad > maxv) maxv = ad;
    }

    if (maxv == 0) {
        sz += 1;                     /* bit_count = 0 byte */
    } else {
        unsigned int bit_count = (unsigned int)floorf(log2f((float)maxv)) + 1;
        unsigned int signBytes = (n + 7) / 8;
        unsigned int byte_count    = bit_count / 8;
        unsigned int remainder_bit = bit_count % 8;
        unsigned int magBytes;
        if (remainder_bit == 0)
            magBytes = byte_count * n;
        else
            magBytes = byte_count * n + (remainder_bit * n + 7) / 8;
        sz += 1 + signBytes + magBytes;
    }
    blockSizes[bid] = sz;
}

/* ─── Pass 2: quantize, delta-encode, pack ──────────────────────────────── */
template <typename T>
__global__ void
ra_compress_packing_kernel(const T *data, size_t nbEle, int blockSize,
                           double inver_bound, const size_t *blockOffsets,
                           unsigned char *output)
{
    size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t numBlocks = (nbEle + blockSize - 1) / blockSize;
    if (bid >= numBlocks) return;

    size_t block_start = bid * blockSize;
    size_t current_block_size = (block_start + blockSize > nbEle)
                                    ? (nbEle - block_start)
                                    : (size_t)blockSize;

    unsigned char *bp = output + blockOffsets[bid];

    /* Write quantized first value */
    int prior = (int)(data[block_start] * inver_bound);
    memcpy(bp, &prior, sizeof(int));
    bp += sizeof(int);

    if (current_block_size <= 1) return;

    unsigned int n = (unsigned int)(current_block_size - 1);
    unsigned char local_signs[SZP_CUDA_MAX_BLOCK];
    unsigned int  local_predict[SZP_CUDA_MAX_BLOCK];
    unsigned int maxv = 0;

    for (unsigned int j = 0; j < n; j++) {
        int cur = (int)(data[block_start + j + 1] * inver_bound);
        int diff = cur - prior;
        prior = cur;
        if (diff == 0) {
            local_signs[j]   = 0;
            local_predict[j] = 0;
        } else if (diff < 0) {
            local_signs[j]   = 1;
            local_predict[j] = (unsigned int)(-diff);
        } else {
            local_signs[j]   = 0;
            local_predict[j] = (unsigned int)diff;
        }
        if (local_predict[j] > maxv) maxv = local_predict[j];
    }

    if (maxv == 0) {
        *bp++ = 0;
    } else {
        unsigned int bit_count = (unsigned int)floorf(log2f((float)maxv)) + 1;
        *bp++ = (unsigned char)bit_count;

        unsigned int slen = pack_signs_1b(local_signs, n, bp);
        bp += slen;

        unsigned int mlen = save_fixed_length_bits(local_predict, n, bp, bit_count);
        bp += mlen;
        (void)mlen;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Threadblock (non-random-access) compression kernel                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* Each CUDA thread processes one "chunk" (same as one OMP thread).           *
 * nChunks is a host-chosen parallelism level; we use 1 for simplicity.       *
 * Two-pass: sizing then packing, same idea as random-access.                 */

template <typename T>
__global__ void
tb_compress_sizing_kernel(const T *data, size_t nbEle, int blockSize,
                          double inver_bound, unsigned int nChunks,
                          size_t *chunkSizes)
{
    unsigned int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= nChunks) return;

    size_t chunkEle = nbEle / nChunks;
    size_t lo = cid * chunkEle;
    size_t hi = (cid == nChunks - 1) ? nbEle : (cid + 1) * chunkEle;
    if (lo >= hi) { chunkSizes[cid] = 0; return; }

    size_t sz = 0;

    /* First element stored raw */
    int prior = (int)(data[lo] * inver_bound);
    sz += sizeof(int);

    for (size_t i = lo + 1; i < hi; i += blockSize) {
        size_t bsz = ((i + blockSize) > hi) ? (hi - i) : (size_t)blockSize;
        unsigned int maxv = 0;
        int p = prior;
        for (size_t j = 0; j < bsz; j++) {
            int c = (int)(data[i + j] * inver_bound);
            int d = c - p; p = c;
            unsigned int ad = (d < 0) ? (unsigned int)(-d) : (unsigned int)d;
            if (ad > maxv) maxv = ad;
        }
        prior = p;

        if (maxv == 0) {
            sz += 1;
        } else {
            unsigned int bc = (unsigned int)floorf(log2f((float)maxv)) + 1;
            unsigned int n = (unsigned int)bsz;
            unsigned int signBytes = (n + 7) / 8;
            unsigned int byte_count    = bc / 8;
            unsigned int remainder_bit = bc % 8;
            unsigned int magBytes;
            if (remainder_bit == 0)
                magBytes = byte_count * n;
            else
                magBytes = byte_count * n + (remainder_bit * n + 7) / 8;
            sz += 1 + signBytes + magBytes;
        }
    }
    chunkSizes[cid] = sz;
}

template <typename T>
__global__ void
tb_compress_packing_kernel(const T *data, size_t nbEle, int blockSize,
                           double inver_bound, unsigned int nChunks,
                           const size_t *chunkOffsets, unsigned char *output)
{
    unsigned int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= nChunks) return;

    size_t chunkEle = nbEle / nChunks;
    size_t lo = cid * chunkEle;
    size_t hi = (cid == nChunks - 1) ? nbEle : (cid + 1) * chunkEle;
    if (lo >= hi) return;

    unsigned char *bp = output + chunkOffsets[cid];

    int prior = (int)(data[lo] * inver_bound);
    memcpy(bp, &prior, sizeof(int));
    bp += sizeof(int);

    unsigned char local_signs[SZP_CUDA_MAX_BLOCK];
    unsigned int  local_predict[SZP_CUDA_MAX_BLOCK];

    for (size_t i = lo + 1; i < hi; i += blockSize) {
        size_t bsz = ((i + blockSize) > hi) ? (hi - i) : (size_t)blockSize;
        unsigned int n = (unsigned int)bsz;
        unsigned int maxv = 0;

        for (unsigned int j = 0; j < n; j++) {
            int cur = (int)(data[i + j] * inver_bound);
            int diff = cur - prior;
            prior = cur;
            if (diff == 0) {
                local_signs[j]   = 0;
                local_predict[j] = 0;
            } else if (diff < 0) {
                local_signs[j]   = 1;
                local_predict[j] = (unsigned int)(-diff);
            } else {
                local_signs[j]   = 0;
                local_predict[j] = (unsigned int)diff;
            }
            if (local_predict[j] > maxv) maxv = local_predict[j];
        }

        if (maxv == 0) {
            *bp++ = 0;
        } else {
            unsigned int bit_count = (unsigned int)floorf(log2f((float)maxv)) + 1;
            *bp++ = (unsigned char)bit_count;

            unsigned int slen = pack_signs_1b(local_signs, n, bp);
            bp += slen;

            unsigned int mlen = save_fixed_length_bits(local_predict, n, bp, bit_count);
            bp += mlen;
            (void)mlen;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Quantization helper kernels                                               */
/* ═══════════════════════════════════════════════════════════════════════════ */

template <typename T>
__global__ void
quantize_kernel(const T *data, int *qArr, size_t nbEle, double inver_bound)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nbEle)
        qArr[idx] = (int)(data[idx] * inver_bound);
}

__global__ void
diff_kernel(const int *qArr, int *diffArr, size_t nbEle)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
        diffArr[0] = qArr[0];
    else if (idx < nbEle)
        diffArr[idx] = qArr[idx] - qArr[idx - 1];
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Host-side random-access compression (templated)                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

template <typename T>
static unsigned char *
cuda_compress_randomaccess_impl(T *oriData, size_t *outSize, T absErrBound,
                                size_t nbEle, int blockSize)
{
    if (nbEle == 0 || absErrBound <= 0) { *outSize = 0; return NULL; }

    double inver_bound = 1.0 / (double)absErrBound;
    size_t numBlocks = (nbEle + blockSize - 1) / blockSize;

    /* --- device allocations --- */
    T *d_data = NULL;
    size_t *d_blockSizes = NULL, *d_blockOffsets = NULL;
    unsigned char *d_output = NULL;

    CUDA_CHECK(cudaMalloc(&d_data, nbEle * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, oriData, nbEle * sizeof(T), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_blockSizes,   numBlocks * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_blockOffsets,  numBlocks * sizeof(size_t)));

    /* --- Pass 1: sizing --- */
    int tpb = 256;
    int gpb = (int)((numBlocks + tpb - 1) / tpb);
    ra_compress_sizing_kernel<T><<<gpb, tpb>>>(d_data, nbEle, blockSize,
                                                inver_bound, d_blockSizes);
    CUDA_CHECK(cudaGetLastError());

    /* --- prefix sum --- */
    thrust::device_ptr<size_t> sz_ptr(d_blockSizes);
    thrust::device_ptr<size_t> of_ptr(d_blockOffsets);
    thrust::exclusive_scan(sz_ptr, sz_ptr + numBlocks, of_ptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* --- total compressed body size --- */
    size_t lastSz = 0, lastOff = 0;
    CUDA_CHECK(cudaMemcpy(&lastSz,  d_blockSizes   + numBlocks - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&lastOff, d_blockOffsets  + numBlocks - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    size_t totalBody = lastOff + lastSz;

    /* --- allocate device output & run Pass 2 --- */
    CUDA_CHECK(cudaMalloc(&d_output, totalBody));
    ra_compress_packing_kernel<T><<<gpb, tpb>>>(d_data, nbEle, blockSize,
                                                 inver_bound, d_blockOffsets,
                                                 d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* --- assemble host output ─── */
    /* Format: [absErrBound : sizeof(T)] [offset0 = 0 : sizeof(size_t)] [body] */
    size_t nChunks   = 1;
    size_t headerSz  = sizeof(T) + nChunks * sizeof(size_t);
    *outSize = headerSz + totalBody;
    unsigned char *output = (unsigned char *)malloc(*outSize);
    if (!output) { *outSize = 0; goto cleanup; }

    /* Write absErrBound as raw bytes (same endianness as memcpy, matching the  *
     * OpenMP code on little-endian + floatToBytes which stores big-endian.     *
     * We use memcpy here for simplicity; call floatToBytes on the host side    *
     * if cross-endian is needed).                                              */
    memcpy(output, &absErrBound, sizeof(T));

    /* single offset = 0 */
    { size_t zero = 0; memcpy(output + sizeof(T), &zero, sizeof(size_t)); }

    /* copy body from device */
    CUDA_CHECK(cudaMemcpy(output + headerSz, d_output, totalBody, cudaMemcpyDeviceToHost));

cleanup:
    if (d_data)         cudaFree(d_data);
    if (d_blockSizes)   cudaFree(d_blockSizes);
    if (d_blockOffsets) cudaFree(d_blockOffsets);
    if (d_output)       cudaFree(d_output);
    return output;
}

/* Pre-allocated variant */
template <typename T>
static void
cuda_compress_randomaccess_arg_impl(unsigned char *output, T *oriData,
                                    size_t *outSize, T absErrBound,
                                    size_t nbEle, int blockSize)
{
    if (!output || nbEle == 0 || absErrBound <= 0) { *outSize = 0; return; }

    double inver_bound = 1.0 / (double)absErrBound;
    size_t numBlocks = (nbEle + blockSize - 1) / blockSize;

    T *d_data = NULL;
    size_t *d_blockSizes = NULL, *d_blockOffsets = NULL;
    unsigned char *d_output = NULL;

    CUDA_CHECK(cudaMalloc(&d_data, nbEle * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, oriData, nbEle * sizeof(T), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_blockSizes,  numBlocks * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_blockOffsets, numBlocks * sizeof(size_t)));

    int tpb = 256;
    int gpb = (int)((numBlocks + tpb - 1) / tpb);
    ra_compress_sizing_kernel<T><<<gpb, tpb>>>(d_data, nbEle, blockSize,
                                                inver_bound, d_blockSizes);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<size_t> sz_ptr(d_blockSizes);
    thrust::device_ptr<size_t> of_ptr(d_blockOffsets);
    thrust::exclusive_scan(sz_ptr, sz_ptr + numBlocks, of_ptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t lastSz = 0, lastOff = 0;
    CUDA_CHECK(cudaMemcpy(&lastSz,  d_blockSizes  + numBlocks - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&lastOff, d_blockOffsets + numBlocks - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    size_t totalBody = lastOff + lastSz;

    CUDA_CHECK(cudaMalloc(&d_output, totalBody));
    ra_compress_packing_kernel<T><<<gpb, tpb>>>(d_data, nbEle, blockSize,
                                                 inver_bound, d_blockOffsets,
                                                 d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t nChunks  = 1;
    size_t headerSz = sizeof(T) + nChunks * sizeof(size_t);
    *outSize = headerSz + totalBody;

    memcpy(output, &absErrBound, sizeof(T));
    { size_t zero = 0; memcpy(output + sizeof(T), &zero, sizeof(size_t)); }
    CUDA_CHECK(cudaMemcpy(output + headerSz, d_output, totalBody, cudaMemcpyDeviceToHost));

    cudaFree(d_data); cudaFree(d_blockSizes);
    cudaFree(d_blockOffsets); cudaFree(d_output);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Host-side threadblock compression (templated)                             */
/* ═══════════════════════════════════════════════════════════════════════════ */

template <typename T>
static unsigned char *
cuda_compress_threadblock_impl(T *oriData, size_t *outSize, T absErrBound,
                               size_t nbEle, int blockSize)
{
    if (nbEle == 0 || absErrBound <= 0) { *outSize = 0; return NULL; }

    double inver_bound = 1.0 / (double)absErrBound;
    unsigned int nChunks = 1;     /* single chunk — serial within */

    T *d_data = NULL;
    size_t *d_chunkSizes = NULL, *d_chunkOffsets = NULL;
    unsigned char *d_output = NULL;

    CUDA_CHECK(cudaMalloc(&d_data, nbEle * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, oriData, nbEle * sizeof(T), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_chunkSizes,   nChunks * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_chunkOffsets,  nChunks * sizeof(size_t)));

    tb_compress_sizing_kernel<T><<<1, 1>>>(d_data, nbEle, blockSize,
                                            inver_bound, nChunks, d_chunkSizes);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<size_t> sz_ptr(d_chunkSizes);
    thrust::device_ptr<size_t> of_ptr(d_chunkOffsets);
    thrust::exclusive_scan(sz_ptr, sz_ptr + nChunks, of_ptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t totalBody = 0;
    CUDA_CHECK(cudaMemcpy(&totalBody, d_chunkSizes, sizeof(size_t), cudaMemcpyDeviceToHost));

    size_t headerSz = sizeof(T) + nChunks * sizeof(size_t);
    *outSize = headerSz + totalBody;
    unsigned char *output = (unsigned char *)malloc(*outSize);
    if (!output) { *outSize = 0; goto cleanup; }

    CUDA_CHECK(cudaMalloc(&d_output, totalBody));
    tb_compress_packing_kernel<T><<<1, 1>>>(d_data, nbEle, blockSize,
                                             inver_bound, nChunks,
                                             d_chunkOffsets, d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    memcpy(output, &absErrBound, sizeof(T));
    { size_t zero = 0; memcpy(output + sizeof(T), &zero, sizeof(size_t)); }
    CUDA_CHECK(cudaMemcpy(output + headerSz, d_output, totalBody, cudaMemcpyDeviceToHost));

cleanup:
    if (d_data)         cudaFree(d_data);
    if (d_chunkSizes)   cudaFree(d_chunkSizes);
    if (d_chunkOffsets) cudaFree(d_chunkOffsets);
    if (d_output)       cudaFree(d_output);
    return output;
}

template <typename T>
static void
cuda_compress_threadblock_arg_impl(unsigned char *output, T *oriData,
                                   size_t *outSize, T absErrBound,
                                   size_t nbEle, int blockSize)
{
    if (!output || nbEle == 0 || absErrBound <= 0) { *outSize = 0; return; }

    double inver_bound = 1.0 / (double)absErrBound;
    unsigned int nChunks = 1;

    T *d_data = NULL;
    size_t *d_chunkSizes = NULL, *d_chunkOffsets = NULL;
    unsigned char *d_output = NULL;

    CUDA_CHECK(cudaMalloc(&d_data, nbEle * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, oriData, nbEle * sizeof(T), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_chunkSizes,  nChunks * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_chunkOffsets, nChunks * sizeof(size_t)));

    tb_compress_sizing_kernel<T><<<1, 1>>>(d_data, nbEle, blockSize,
                                            inver_bound, nChunks, d_chunkSizes);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<size_t> sz_ptr(d_chunkSizes);
    thrust::device_ptr<size_t> of_ptr(d_chunkOffsets);
    thrust::exclusive_scan(sz_ptr, sz_ptr + nChunks, of_ptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t totalBody = 0;
    CUDA_CHECK(cudaMemcpy(&totalBody, d_chunkSizes, sizeof(size_t), cudaMemcpyDeviceToHost));

    size_t headerSz = sizeof(T) + nChunks * sizeof(size_t);
    *outSize = headerSz + totalBody;

    memcpy(output, &absErrBound, sizeof(T));
    { size_t zero = 0; memcpy(output + sizeof(T), &zero, sizeof(size_t)); }

    CUDA_CHECK(cudaMalloc(&d_output, totalBody));
    tb_compress_packing_kernel<T><<<1, 1>>>(d_data, nbEle, blockSize,
                                             inver_bound, nChunks,
                                             d_chunkOffsets, d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output + headerSz, d_output, totalBody, cudaMemcpyDeviceToHost));

    cudaFree(d_data); cudaFree(d_chunkSizes);
    cudaFree(d_chunkOffsets); cudaFree(d_output);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Quantization helper implementations                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

static int *
cuda_direct_predict_quantization_impl(float *oriData, size_t * /*outSize*/,
                                      float absErrBound, size_t nbEle,
                                      int /*blockSize*/)
{
    if (nbEle == 0) return NULL;
    double inver_bound = 1.0 / (double)absErrBound;

    float *d_data = NULL;
    int   *d_qArr = NULL, *d_diffArr = NULL;

    CUDA_CHECK(cudaMalloc(&d_data, nbEle * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, oriData, nbEle * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_qArr,    nbEle * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_diffArr, nbEle * sizeof(int)));

    int tpb = 256;
    int gpb = (int)((nbEle + tpb - 1) / tpb);

    quantize_kernel<float><<<gpb, tpb>>>(d_data, d_qArr, nbEle, inver_bound);
    CUDA_CHECK(cudaGetLastError());

    diff_kernel<<<gpb, tpb>>>(d_qArr, d_diffArr, nbEle);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int *result = (int *)malloc(nbEle * sizeof(int));
    CUDA_CHECK(cudaMemcpy(result, d_diffArr, nbEle * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_data); cudaFree(d_qArr); cudaFree(d_diffArr);
    return result;
}

/* Threadblock predict quantization — same as direct for CUDA, but chunked   */
static int *
cuda_threadblock_predict_quantization_impl(float *oriData, size_t *outSize,
                                           float absErrBound, size_t nbEle,
                                           int blockSize)
{
    /* For CUDA the result is identical to direct predict quantization:        *
     * both produce element-wise (quantized[i] - quantized[i-1]).             */
    return cuda_direct_predict_quantization_impl(oriData, outSize,
                                                  absErrBound, nbEle, blockSize);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  C-linkage wrappers (instantiate templates for float and double)            */
/* ═══════════════════════════════════════════════════════════════════════════ */

extern "C" {

/* ─── Random-access float ─── */
unsigned char *szp_cuda_float_compress_randomaccess(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize)
{
    return cuda_compress_randomaccess_impl<float>(oriData, outSize, absErrBound,
                                                   nbEle, blockSize);
}

void szp_cuda_float_compress_randomaccess_arg(
    unsigned char *output, float *oriData, size_t *outSize,
    float absErrBound, size_t nbEle, int blockSize)
{
    cuda_compress_randomaccess_arg_impl<float>(output, oriData, outSize,
                                                absErrBound, nbEle, blockSize);
}

/* ─── Random-access double ─── */
unsigned char *szp_cuda_double_compress_randomaccess(
    double *oriData, size_t *outSize, double absErrBound,
    size_t nbEle, int blockSize)
{
    return cuda_compress_randomaccess_impl<double>(oriData, outSize, absErrBound,
                                                    nbEle, blockSize);
}

void szp_cuda_double_compress_randomaccess_arg(
    unsigned char *output, double *oriData, size_t *outSize,
    double absErrBound, size_t nbEle, int blockSize)
{
    cuda_compress_randomaccess_arg_impl<double>(output, oriData, outSize,
                                                 absErrBound, nbEle, blockSize);
}

/* ─── Threadblock float ─── */
unsigned char *szp_cuda_float_compress_threadblock(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize)
{
    return cuda_compress_threadblock_impl<float>(oriData, outSize, absErrBound,
                                                  nbEle, blockSize);
}

void szp_cuda_float_compress_threadblock_arg(
    unsigned char *output, float *oriData, size_t *outSize,
    float absErrBound, size_t nbEle, int blockSize)
{
    cuda_compress_threadblock_arg_impl<float>(output, oriData, outSize,
                                               absErrBound, nbEle, blockSize);
}

/* ─── Threadblock double ─── */
unsigned char *szp_cuda_double_compress_threadblock(
    double *oriData, size_t *outSize, double absErrBound,
    size_t nbEle, int blockSize)
{
    return cuda_compress_threadblock_impl<double>(oriData, outSize, absErrBound,
                                                   nbEle, blockSize);
}

void szp_cuda_double_compress_threadblock_arg(
    unsigned char *output, double *oriData, size_t *outSize,
    double absErrBound, size_t nbEle, int blockSize)
{
    cuda_compress_threadblock_arg_impl<double>(output, oriData, outSize,
                                                absErrBound, nbEle, blockSize);
}

/* ─── Quantization helpers ─── */
int *szp_cuda_float_direct_predict_quantization(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize)
{
    return cuda_direct_predict_quantization_impl(oriData, outSize,
                                                  absErrBound, nbEle, blockSize);
}

int *szp_cuda_float_threadblock_predict_quantization(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize)
{
    return cuda_threadblock_predict_quantization_impl(oriData, outSize,
                                                      absErrBound, nbEle, blockSize);
}

} /* extern "C" */
