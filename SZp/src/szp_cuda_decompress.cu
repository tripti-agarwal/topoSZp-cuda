/**
 *  @file szp_cuda_decompress.cu
 *  @brief CUDA decompression kernels for TopoSZp.
 *
 *  Byte-compatible with the OpenMP compressed format.  Data compressed
 *  by the OpenMP compressor can be decompressed here and vice-versa.
 *
 *  Compressed layout produced by the OpenMP compressor:
 *
 *    Random-access mode
 *    ------------------
 *    [offsets[0..N-1] : size_t[N]]   per-OMP-thread byte offsets
 *    [compressed data ...]           rcp = cmpBytes + N*sizeof(size_t)
 *
 *    Each OMP thread's data is a sequence of independently-decodable
 *    compression blocks:
 *      [first_value : int32]         raw quantized anchor
 *      [bit_count   : uint8]         0 → all diffs zero; else #bits per mag
 *      [sign_bits   : packed 1-bit]  (block_size-1+7)/8 bytes, MSB-first
 *      [magnitudes  : packed N-bit]  Jiajun fixed-length layout
 *
 *    Threadblock (non-random-access) mode
 *    ------------------------------------
 *    Same header.  Each OMP thread's data starts with ONE int32 anchor
 *    for its entire chunk; subsequent blocks carry only
 *      [bit_count] [sign_bits] [magnitudes]
 *    and the prefix-sum chains across blocks.
 *
 *    Topology-preserved mode  (random-access + per-block type data)
 *    ---------------------------------------------------------------
 *    Same as random-access, but after each block's sign+magnitude:
 *      [type_bits : packed 2-bit, (2*block_size+7)/8 bytes]
 *    encoding critical-point type (0=regular,1=max,2=min,3=saddle).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "szp_cuda_decompress.cuh"
#include "szp_defines.h"

/* ================================================================== */
/*  CUDA helpers                                                       */
/* ================================================================== */

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_e));         \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/* ================================================================== */
/*  Device-side bit unpacking (serial, one CUDA thread per block)      */
/* ================================================================== */

/**
 * Unpack 1-bit sign array from MSB-first packed bytes.
 * Matches convertByteArray2IntArray_fast_1b_args on CPU.
 */
__device__ static void
device_unpack_1b(const unsigned char *packed, unsigned char *out,
                 unsigned int n)
{
    for (unsigned int i = 0; i < n; i++) {
        unsigned int byte_idx = i >> 3;           /* i / 8 */
        unsigned int bit_idx  = 7 - (i & 7);      /* 7 - i%8 */
        out[i] = (packed[byte_idx] >> bit_idx) & 1;
    }
}

/**
 * Unpack 2-bit type array from MSB-first packed bytes.
 * Matches convertByteArray2IntArray_fast_2b on CPU.
 */
__device__ static void
device_unpack_2b(const unsigned char *packed, int *out,
                 unsigned int n)
{
    for (unsigned int i = 0; i < n; i++) {
        unsigned int byte_idx = i >> 2;            /* i / 4 */
        unsigned int slot     = 3 - (i & 3);       /* 3 - i%4 */
        out[i] = (packed[byte_idx] >> (slot * 2)) & 0x03;
    }
}

/**
 * Extract one N-bit value from the Jiajun fixed-length packed layout.
 *
 * Layout: first  byte_count*totalElements  bytes hold the full-byte
 * portions (little-endian per element), followed by the remainder-bit
 * portions packed with the Jiajun_convertUInt2Byte_fast_Xb functions
 * (MSB-first within each byte, sequential element order).
 *
 * This function mirrors Jiajun_extract_fixed_length_bits but computes
 * a single element given its index.
 */
__device__ static unsigned int
device_extract_fixed_bits(const unsigned char *data,
                          unsigned int idx,
                          unsigned int totalElements,
                          unsigned int bit_count)
{
    unsigned int byte_count    = bit_count >> 3;     /* bit_count / 8 */
    unsigned int remainder_bit = bit_count & 7;      /* bit_count % 8 */
    unsigned int byte_offset   = byte_count * totalElements;

    unsigned int value = 0;

    /* ---- remainder-bit portion (stored after full-byte section) ---- */
    if (remainder_bit > 0) {
        /*
         * The Jiajun packing stores remainder_bit bits per element
         * sequentially, MSB-first, starting at data[byte_offset].
         * Element idx starts at global bit position idx*remainder_bit.
         */
        unsigned int global_bit = idx * remainder_bit;
        unsigned int start_byte = byte_offset + (global_bit >> 3);
        unsigned int start_bit  = global_bit & 7;

        /* Gather up to 3 bytes to cover the range safely. */
        unsigned int raw = 0;
        raw |= ((unsigned int)data[start_byte]) << 16;
        if (start_bit + remainder_bit > 8)
            raw |= ((unsigned int)data[start_byte + 1]) << 8;
        if (start_bit + remainder_bit > 16)
            raw |= ((unsigned int)data[start_byte + 2]);

        unsigned int shift = 24 - start_bit - remainder_bit;
        unsigned int mask  = (1u << remainder_bit) - 1u;
        value = (raw >> shift) & mask;
    }

    /* ---- full-byte portion (stored first, little-endian per elem) --- */
    if (byte_count > 0) {
        unsigned int full = 0;
        const unsigned char *base = data + idx * byte_count;
        for (unsigned int j = 0; j < byte_count; j++)
            full |= ((unsigned int)base[j]) << (8 * j);
        full <<= remainder_bit;
        value |= full;
    }

    return value;
}

/**
 * Compute the byte length of a Jiajun fixed-length packed array.
 * Mirrors the byteLength computation in Jiajun_extract_fixed_length_bits.
 */
__device__ __host__ static unsigned int
fixed_bits_byte_length(unsigned int totalElements, unsigned int bit_count)
{
    unsigned int byte_count    = bit_count >> 3;
    unsigned int remainder_bit = bit_count & 7;
    unsigned int byte_offset   = byte_count * totalElements;
    if (remainder_bit == 0)
        return byte_offset;
    return byte_offset + (remainder_bit * totalElements - 1) / 8 + 1;
}

/* Same function but host-only for offset scanning */
static unsigned int
host_fixed_bits_byte_length(unsigned int totalElements, unsigned int bit_count)
{
    unsigned int byte_count    = bit_count / 8;
    unsigned int remainder_bit = bit_count % 8;
    unsigned int byte_offset   = byte_count * totalElements;
    if (remainder_bit == 0)
        return byte_offset;
    return byte_offset + (remainder_bit * totalElements - 1) / 8 + 1;
}

/* ================================================================== */
/*  Host-side block-offset scanner                                     */
/*                                                                     */
/*  The OMP compressor records per-OMP-thread offsets.  We need to     */
/*  walk through each thread's compressed stream to find individual    */
/*  compression-block byte offsets.  This is O(num_blocks) on CPU     */
/*  and very fast (just arithmetic, no decompression).                 */
/* ================================================================== */

/**
 * Scan random-access compressed data to build per-compression-block
 * byte offset table.
 *
 * @param cmpBytes        full compressed buffer (starts with offset table)
 * @param nbEle           total number of elements
 * @param blockSize       compression block size
 * @param[out] numBlocks  total number of compression blocks
 * @return  host-allocated array of byte offsets (relative to rcp)
 *          for each compression block, caller must free()
 */
static size_t *
scan_randomaccess_block_offsets(const unsigned char *cmpBytes,
                                size_t nbEle, int blockSize,
                                size_t *numBlocks,
                                unsigned int *outNbThreads)
{
    /*
     * Determine nbThreads from the offset table.
     * The compressor writes nbThreads size_t offsets at the beginning.
     * We need to figure out nbThreads.  The offsets are monotonically
     * increasing, and offsets[0] is always 0.  We can infer nbThreads
     * by looking at the pattern.
     *
     * Strategy: try nbThreads = 1,2,4,8,16,32,... and check if the
     * implied rcp + offsets[0] makes sense.
     *
     * Simpler: we know the number of compression blocks, and from the
     * offsets we can reconstruct.  But the cleanest way: we just need
     * to know where rcp starts.  offsets[0] should be 0 (first thread
     * starts at rcp+0).  The number of offset entries equals nbThreads.
     *
     * We can try powers of 2 and verify: cast cmpBytes as size_t*,
     * check if entry 0 == 0.  If so, that's one valid nbThreads.
     * Then check if entry 1 could be a valid byte offset (< total size).
     *
     * Best approach: try nbThreads from 1 up, validate by checking
     * that all offsets are monotonically non-decreasing and < total
     * data size.  Use smallest valid nbThreads where we can parse
     * all blocks.  In practice, just try common values.
     */

    /* Try to detect nbThreads: offsets[0] should be 0. */
    const size_t *offs = (const size_t *)cmpBytes;

    /* Binary search for nbThreads: we know offsets[0]==0 always.
     * Walk up checking when offs[k] starts looking like data.
     * The first offset entry that is NOT 0 (for k>0) or that
     * exceeds a reasonable bound tells us where the offset table ends.
     *
     * Most robust: try nbThreads = 1..128 and verify:
     *   - offs[0] == 0
     *   - all offs[i] < total compressed size
     *   - we can walk through all blocks without going out of bounds
     */
    unsigned int nbThreads = 0;

    /* Estimate total compressed data size (upper bound) */
    size_t maxCmpSize = sizeof(float) * nbEle + sizeof(float);

    for (unsigned int try_nt = 1; try_nt <= 256; try_nt++) {
        if (offs[0] != 0) break;  /* offsets[0] must be 0 */

        /* Check all offsets are within bounds */
        int valid = 1;
        for (unsigned int k = 1; k < try_nt; k++) {
            if (offs[k] > maxCmpSize) { valid = 0; break; }
            if (offs[k] < offs[k-1]) { valid = 0; break; }
        }
        if (!valid) break;

        /* Verify we can parse at least the first block after the header */
        size_t hdr_size = try_nt * sizeof(size_t);
        const unsigned char *rcp = cmpBytes + hdr_size;

        /* Try to parse first block: 4 bytes (int32) + 1 byte (bit_count) */
        if (hdr_size + 5 <= maxCmpSize) {
            nbThreads = try_nt;
            /* Keep going to find the largest valid nbThreads */
        }
    }
    /* Fallback: if nothing worked, assume 1 thread */
    if (nbThreads == 0) nbThreads = 1;

    *outNbThreads = nbThreads;

    size_t hdr_size = nbThreads * sizeof(size_t);
    const unsigned char *rcp = cmpBytes + hdr_size;

    /* Count total compression blocks */
    size_t total_blocks = (nbEle + blockSize - 1) / blockSize;
    *numBlocks = total_blocks;

    size_t *block_offsets = (size_t *)malloc(total_blocks * sizeof(size_t));
    if (!block_offsets) return NULL;

    /* Figure out each OMP thread's element range */
    size_t threadblocksize = nbEle / nbThreads;

    /* Walk through each OMP thread's compressed data */
    size_t global_block_idx = 0;
    for (unsigned int tid = 0; tid < nbThreads; tid++) {
        size_t lo = tid * threadblocksize;
        size_t hi = (tid == nbThreads - 1) ? nbEle : (tid + 1) * threadblocksize;

        const unsigned char *ptr = rcp + offs[tid];
        size_t base_offset = offs[tid];  /* relative to rcp */

        /* In random-access mode, each block starts with int32 anchor */
        for (size_t i = lo; i < hi; i += blockSize) {
            if (global_block_idx >= total_blocks) break;

            block_offsets[global_block_idx] = (size_t)(ptr - rcp);
            global_block_idx++;

            size_t current_block_size = ((i + blockSize) > hi)
                                        ? (hi - i) : (size_t)blockSize;

            /* Skip: int32 anchor */
            ptr += sizeof(int);

            if (current_block_size > 1) {
                /* Skip: bit_count */
                unsigned int bit_count = ptr[0];
                ptr++;

                if (bit_count == 0) {
                    /* no sign/magnitude data */
                } else {
                    /* sign bytes */
                    unsigned int sign_bytes = (unsigned int)((current_block_size - 2) / 8 + 1);
                    ptr += sign_bytes;

                    /* magnitude bytes */
                    unsigned int mag_bytes = host_fixed_bits_byte_length(
                        (unsigned int)(current_block_size - 1), bit_count);
                    ptr += mag_bytes;
                }
            }
        }
    }

    return block_offsets;
}

/**
 * Scan topology-preserved random-access compressed data.
 * Same as scan_randomaccess_block_offsets but accounts for the extra
 * 2-bit type data appended after each block's sign+magnitude.
 */
static size_t *
scan_randomaccess_topo_block_offsets(const unsigned char *cmpBytes,
                                     size_t nbEle, int blockSize,
                                     size_t *numBlocks,
                                     unsigned int *outNbThreads)
{
    const size_t *offs = (const size_t *)cmpBytes;

    unsigned int nbThreads = 0;
    size_t maxCmpSize = 8ull * nbEle + 1024;

    for (unsigned int try_nt = 1; try_nt <= 256; try_nt++) {
        if (offs[0] != 0) break;
        int valid = 1;
        for (unsigned int k = 1; k < try_nt; k++) {
            if (offs[k] > maxCmpSize || offs[k] < offs[k-1]) {
                valid = 0; break;
            }
        }
        if (!valid) break;
        size_t hdr_size = try_nt * sizeof(size_t);
        if (hdr_size + 5 <= maxCmpSize)
            nbThreads = try_nt;
    }
    if (nbThreads == 0) nbThreads = 1;
    *outNbThreads = nbThreads;

    size_t hdr_size = nbThreads * sizeof(size_t);
    const unsigned char *rcp = cmpBytes + hdr_size;

    size_t total_blocks = (nbEle + blockSize - 1) / blockSize;
    *numBlocks = total_blocks;

    size_t *block_offsets = (size_t *)malloc(total_blocks * sizeof(size_t));
    if (!block_offsets) return NULL;

    size_t threadblocksize = nbEle / nbThreads;
    size_t global_block_idx = 0;

    for (unsigned int tid = 0; tid < nbThreads; tid++) {
        /* Use block-based distribution matching the compressor */
        size_t num_blocks_total = (nbEle + blockSize - 1) / blockSize;
        size_t blocks_per_thread = (num_blocks_total + nbThreads - 1) / nbThreads;
        size_t start_block = tid * blocks_per_thread;
        size_t end_block   = (tid + 1) * blocks_per_thread;
        if (end_block > num_blocks_total) end_block = num_blocks_total;

        const unsigned char *ptr = rcp + offs[tid];

        for (size_t bidx = start_block; bidx < end_block; bidx++) {
            size_t i = bidx * blockSize;
            if (i >= nbEle) break;

            if (global_block_idx < total_blocks)
                block_offsets[global_block_idx] = (size_t)(ptr - rcp);
            global_block_idx++;

            size_t current_block_size = ((i + blockSize) > nbEle)
                                        ? (nbEle - i) : (size_t)blockSize;

            /* int32 anchor */
            ptr += sizeof(int);

            unsigned int actual_new = (current_block_size > 1) ? (unsigned int)(current_block_size - 1) : 0;

            /* bit_count */
            unsigned int bit_count = ptr[0];
            ptr++;

            if (bit_count != 0 && actual_new > 0) {
                unsigned int sign_bytes = (actual_new + 7) / 8;
                ptr += sign_bytes;
                unsigned int mag_bytes = host_fixed_bits_byte_length(actual_new, bit_count);
                ptr += mag_bytes;
            }

            /* 2-bit type data */
            unsigned int type_bytes = (unsigned int)((2 * current_block_size + 7) / 8);
            ptr += type_bytes;
        }
    }

    return block_offsets;
}

/**
 * Scan threadblock (non-random-access) compressed data.
 * Each OMP thread's chunk starts with ONE int32 anchor, then blocks
 * carry only [bit_count][signs][magnitudes] with prefix-sum chaining.
 *
 * Returns per-OMP-thread information needed for the kernel.
 */
static void
scan_threadblock_offsets(const unsigned char *cmpBytes,
                         size_t nbEle, int blockSize,
                         unsigned int *outNbThreads)
{
    const size_t *offs = (const size_t *)cmpBytes;
    unsigned int nbThreads = 0;
    size_t maxCmpSize = sizeof(float) * nbEle + sizeof(float);

    for (unsigned int try_nt = 1; try_nt <= 256; try_nt++) {
        if (offs[0] != 0) break;
        int valid = 1;
        for (unsigned int k = 1; k < try_nt; k++) {
            if (offs[k] > maxCmpSize || offs[k] < offs[k-1]) {
                valid = 0; break;
            }
        }
        if (!valid) break;
        size_t hdr_size = try_nt * sizeof(size_t);
        if (hdr_size + 5 <= maxCmpSize)
            nbThreads = try_nt;
    }
    if (nbThreads == 0) nbThreads = 1;
    *outNbThreads = nbThreads;
}

/* ================================================================== */
/*  CUDA kernels                                                       */
/* ================================================================== */

/**
 * Random-access decompression kernel.
 * One CUDA thread per compression block.
 * Each thread independently decodes its block.
 */
template <typename T>
__global__ void
kernel_decompress_randomaccess(T            *newData,
                               const unsigned char *rcp,
                               const size_t *block_offsets,
                               size_t        nbEle,
                               T             absErrBound,
                               int           compBlockSize,
                               size_t        numBlocks)
{
    size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= numBlocks) return;

    size_t elem_start = bid * compBlockSize;
    size_t current_block_size = ((elem_start + compBlockSize) > nbEle)
                                ? (nbEle - elem_start) : (size_t)compBlockSize;
    if (current_block_size == 0) return;

    const unsigned char *ptr = rcp + block_offsets[bid];
    T *dst = newData + elem_start;

    /* Read anchor */
    int prior;
    memcpy(&prior, ptr, sizeof(int));
    ptr += sizeof(int);

    T ori_prior = (T)prior * absErrBound;
    dst[0] = ori_prior;

    if (current_block_size <= 1) return;

    unsigned int actual_n = (unsigned int)(current_block_size - 1);

    /* Read bit_count */
    unsigned int bit_count = ptr[0];
    ptr++;

    if (bit_count == 0) {
        /* All values same as anchor */
        for (unsigned int j = 0; j < actual_n; j++)
            dst[1 + j] = ori_prior;
        return;
    }

    /* Unpack sign bits */
    unsigned int sign_bytes = (actual_n + 7) / 8;
    const unsigned char *sign_data = ptr;
    ptr += sign_bytes;

    /* Unpack magnitude bits */
    const unsigned char *mag_data = ptr;

    /* Sequential prefix-sum reconstruction */
    int current;
    for (unsigned int j = 0; j < actual_n; j++) {
        /* Extract sign */
        unsigned int sb = j >> 3;
        unsigned int si = 7 - (j & 7);
        int sign = (sign_data[sb] >> si) & 1;

        /* Extract magnitude */
        unsigned int mag = device_extract_fixed_bits(mag_data, j, actual_n,
                                                     bit_count);

        int diff = sign ? -(int)mag : (int)mag;
        current = prior + diff;
        dst[1 + j] = (T)current * absErrBound;
        prior = current;
    }
}

/**
 * Threadblock (non-random-access) decompression kernel.
 * One CUDA thread per OMP-thread chunk.
 * Each thread sequentially decodes its chunk (serial prefix-sum).
 */
template <typename T>
__global__ void
kernel_decompress_threadblock(T            *newData,
                              const unsigned char *rcp,
                              const size_t *offsets,
                              size_t        nbEle,
                              T             absErrBound,
                              int           compBlockSize,
                              unsigned int  nbOMPThreads)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nbOMPThreads) return;

    size_t threadblocksize = nbEle / nbOMPThreads;
    size_t lo = tid * threadblocksize;
    size_t hi = (tid == nbOMPThreads - 1) ? nbEle : (tid + 1) * threadblocksize;
    if (lo >= hi) return;

    T *dst = newData + lo;
    const unsigned char *ptr = rcp + offsets[tid];

    int prior, current, diff;

    /* First element: raw anchor for this chunk */
    memcpy(&prior, ptr, sizeof(int));
    ptr += sizeof(int);

    T ori_prior = (T)prior * absErrBound;
    *dst++ = ori_prior;

    int block_size = compBlockSize;

    for (size_t i = lo + 1; i < hi; i += block_size) {
        size_t cur_bs = ((i + block_size) > hi) ? (hi - i) : (size_t)block_size;
        if (cur_bs == 0) continue;

        unsigned int bit_count = ptr[0];
        ptr++;

        if (bit_count == 0) {
            ori_prior = (T)prior * absErrBound;
            for (size_t j = 0; j < cur_bs; j++)
                *dst++ = ori_prior;
        } else {
            unsigned int sign_bytes = (unsigned int)((cur_bs - 1) / 8 + 1);
            const unsigned char *sign_data = ptr;
            ptr += sign_bytes;

            unsigned int mag_byte_len = fixed_bits_byte_length(
                (unsigned int)cur_bs, bit_count);
            const unsigned char *mag_data = ptr;
            ptr += mag_byte_len;

            for (unsigned int j = 0; j < (unsigned int)cur_bs; j++) {
                unsigned int sb = j >> 3;
                unsigned int si = 7 - (j & 7);
                int sign = (sign_data[sb] >> si) & 1;

                unsigned int mag = device_extract_fixed_bits(
                    mag_data, j, (unsigned int)cur_bs, bit_count);

                diff = sign ? -(int)mag : (int)mag;
                current = prior + diff;
                *dst++ = (T)current * absErrBound;
                prior = current;
            }
        }
    }
}

/**
 * Topology-preserved random-access decompression kernel.
 * One CUDA thread per compression block.
 * Writes both decompressed data and critical-point type array.
 */
__global__ void
kernel_decompress_randomaccess_topo(float        *newData,
                                    int          *fnData,
                                    const unsigned char *rcp,
                                    const size_t *block_offsets,
                                    size_t        nbEle,
                                    float         absErrBound,
                                    int           compBlockSize,
                                    size_t        numBlocks)
{
    size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= numBlocks) return;

    size_t elem_start = bid * compBlockSize;
    size_t current_block_size = ((elem_start + compBlockSize) > nbEle)
                                ? (nbEle - elem_start) : (size_t)compBlockSize;
    if (current_block_size == 0) return;

    const unsigned char *ptr = rcp + block_offsets[bid];
    float *dst = newData + elem_start;

    /* Read anchor */
    int prior;
    memcpy(&prior, ptr, sizeof(int));
    ptr += sizeof(int);

    float scale = absErrBound;
    float ori = (float)prior * scale;
    dst[0] = ori;

    unsigned int actual_n = (current_block_size > 1)
                            ? (unsigned int)(current_block_size - 1) : 0;

    /* Read bit_count */
    unsigned int bit_count = ptr[0];
    ptr++;

    if (bit_count == 0) {
        for (unsigned int j = 0; j < actual_n; j++)
            dst[1 + j] = ori;
    } else if (actual_n > 0) {
        unsigned int sign_bytes = (actual_n + 7) / 8;
        const unsigned char *sign_data = ptr;
        ptr += sign_bytes;

        unsigned int mag_byte_len = fixed_bits_byte_length(actual_n, bit_count);
        const unsigned char *mag_data = ptr;
        ptr += mag_byte_len;

        int current;
        for (unsigned int j = 0; j < actual_n; j++) {
            unsigned int sb = j >> 3;
            unsigned int si = 7 - (j & 7);
            int sign = (sign_data[sb] >> si) & 1;

            unsigned int mag = device_extract_fixed_bits(mag_data, j,
                                                         actual_n, bit_count);
            int diff = sign ? -(int)mag : (int)mag;
            current = prior + diff;
            dst[1 + j] = (float)current * scale;
            prior = current;
        }
    }

    /* Advance ptr past sign+magnitude to reach type data.
     * If bit_count != 0 and actual_n > 0, ptr is already past magnitude.
     * If bit_count == 0, ptr is right after the bit_count byte.
     * We need to get to the type data regardless. */
    if (bit_count == 0) {
        /* ptr is already at type data */
    } else if (actual_n > 0) {
        /* ptr was already advanced past magnitude above - but we used
         * local mag_data pointer.  Recompute ptr position. */
        const unsigned char *base = rcp + block_offsets[bid] + sizeof(int) + 1;
        if (bit_count != 0 && actual_n > 0) {
            unsigned int sign_bytes = (actual_n + 7) / 8;
            unsigned int mag_byte_len = fixed_bits_byte_length(actual_n, bit_count);
            base += sign_bytes + mag_byte_len;
        }
        ptr = base;
    }

    /* Unpack 2-bit type data */
    unsigned int type_bytes = (2 * (unsigned int)current_block_size + 7) / 8;
    device_unpack_2b(ptr, fnData + elem_start, (unsigned int)current_block_size);
}

/**
 * Sort-position decompression kernel.
 * Same as random-access float decompression but writes int output
 * and does not multiply by absErrBound.
 */
__global__ void
kernel_decompress_sort_positions(int          *newData,
                                 const unsigned char *rcp,
                                 const size_t *block_offsets,
                                 size_t        nbEle,
                                 int           compBlockSize,
                                 size_t        numBlocks)
{
    size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= numBlocks) return;

    size_t elem_start = bid * compBlockSize;
    size_t current_block_size = ((elem_start + compBlockSize) > nbEle)
                                ? (nbEle - elem_start) : (size_t)compBlockSize;
    if (current_block_size == 0) return;

    const unsigned char *ptr = rcp + block_offsets[bid];
    int *dst = newData + elem_start;

    /* Read anchor */
    int prior;
    memcpy(&prior, ptr, sizeof(int));
    ptr += sizeof(int);
    dst[0] = prior;

    if (current_block_size <= 1) return;

    unsigned int actual_n = (unsigned int)(current_block_size - 1);
    unsigned int bit_count = ptr[0];
    ptr++;

    if (bit_count == 0) {
        for (unsigned int j = 0; j < actual_n; j++)
            dst[1 + j] = prior;
        return;
    }

    unsigned int sign_bytes = (actual_n + 7) / 8;
    const unsigned char *sign_data = ptr;
    ptr += sign_bytes;

    const unsigned char *mag_data = ptr;

    int current;
    for (unsigned int j = 0; j < actual_n; j++) {
        unsigned int sb = j >> 3;
        unsigned int si = 7 - (j & 7);
        int sign = (sign_data[sb] >> si) & 1;

        unsigned int mag = device_extract_fixed_bits(mag_data, j, actual_n,
                                                     bit_count);
        int diff = sign ? -(int)mag : (int)mag;
        current = prior + diff;
        dst[1 + j] = current;
        prior = current;
    }
}

/* ================================================================== */
/*  Host wrapper: random-access decompression (templated)              */
/* ================================================================== */

template <typename T>
static void
cuda_decompress_randomaccess_impl(T *hostOut, size_t nbEle,
                                   T absErrBound, int blockSize,
                                   unsigned char *cmpBytes)
{
    if (!cmpBytes || nbEle == 0) return;

    /* 1. Scan compressed data on CPU to build per-block offsets */
    size_t numBlocks = 0;
    unsigned int nbOMPThreads = 0;
    size_t *h_block_offsets = scan_randomaccess_block_offsets(
        cmpBytes, nbEle, blockSize, &numBlocks, &nbOMPThreads);
    if (!h_block_offsets) return;

    /* 2. Compute total compressed size */
    const size_t *offs = (const size_t *)cmpBytes;
    size_t hdr_size = nbOMPThreads * sizeof(size_t);
    /* The total compressed data size after header */
    /* We need to figure out the end.  The last OMP thread's data ends
     * at the end of the compressed buffer.  We can compute this by
     * summing up all block sizes, or just use the maximum offset + data. */
    /* Conservative: use sizeof(T)*nbEle as upper bound */
    size_t cmpTotalSize = hdr_size;
    /* Walk through last thread to find end */
    {
        size_t lastOff = offs[nbOMPThreads - 1];
        /* Walk blocks of last thread to find end */
        size_t threadblocksize = nbEle / nbOMPThreads;
        size_t lo = (nbOMPThreads - 1) * threadblocksize;
        size_t hi = nbEle;
        const unsigned char *rcp = cmpBytes + hdr_size;
        const unsigned char *ptr = rcp + lastOff;

        for (size_t i = lo; i < hi; i += blockSize) {
            size_t cur_bs = ((i + blockSize) > hi) ? (hi - i) : (size_t)blockSize;
            ptr += sizeof(int);
            if (cur_bs > 1) {
                unsigned int bc = ptr[0]; ptr++;
                if (bc != 0) {
                    unsigned int sb = (unsigned int)((cur_bs - 2) / 8 + 1);
                    ptr += sb;
                    ptr += host_fixed_bits_byte_length((unsigned int)(cur_bs - 1), bc);
                }
            }
        }
        cmpTotalSize = (size_t)(ptr - cmpBytes);
    }

    /* 3. Allocate device memory */
    unsigned char *d_cmpBytes;
    T *d_newData;
    size_t *d_block_offsets;

    CUDA_CHECK(cudaMalloc(&d_cmpBytes, cmpTotalSize));
    CUDA_CHECK(cudaMalloc(&d_newData, nbEle * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, numBlocks * sizeof(size_t)));

    /* 4. Copy data to device */
    CUDA_CHECK(cudaMemcpy(d_cmpBytes, cmpBytes, cmpTotalSize,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_offsets, h_block_offsets,
                          numBlocks * sizeof(size_t),
                          cudaMemcpyHostToDevice));

    /* 5. Launch kernel */
    unsigned char *d_rcp = d_cmpBytes + hdr_size;
    int cudaBlkSize = 256;
    int numCudaBlocks = (int)((numBlocks + cudaBlkSize - 1) / cudaBlkSize);

    kernel_decompress_randomaccess<T><<<numCudaBlocks, cudaBlkSize>>>(
        d_newData, d_rcp, d_block_offsets, nbEle, absErrBound,
        blockSize, numBlocks);
    CUDA_CHECK(cudaGetLastError());

    /* 6. Copy result back */
    CUDA_CHECK(cudaMemcpy(hostOut, d_newData, nbEle * sizeof(T),
                          cudaMemcpyDeviceToHost));

    /* 7. Cleanup */
    cudaFree(d_cmpBytes);
    cudaFree(d_newData);
    cudaFree(d_block_offsets);
    free(h_block_offsets);
}

/* ================================================================== */
/*  Host wrapper: threadblock decompression (templated)                */
/* ================================================================== */

template <typename T>
static void
cuda_decompress_threadblock_impl(T *hostOut, size_t nbEle,
                                  T absErrBound, int blockSize,
                                  unsigned char *cmpBytes)
{
    if (!cmpBytes || nbEle == 0) return;

    unsigned int nbOMPThreads = 0;
    scan_threadblock_offsets(cmpBytes, nbEle, blockSize, &nbOMPThreads);

    size_t hdr_size = nbOMPThreads * sizeof(size_t);
    const size_t *offs = (const size_t *)cmpBytes;

    /* Compute total compressed data size (same conservative scan) */
    size_t cmpTotalSize = hdr_size;
    {
        size_t threadblocksize = nbEle / nbOMPThreads;
        for (unsigned int tid = 0; tid < nbOMPThreads; tid++) {
            size_t lo = tid * threadblocksize;
            size_t hi = (tid == nbOMPThreads - 1) ? nbEle : (tid + 1) * threadblocksize;
            const unsigned char *rcp = cmpBytes + hdr_size;
            const unsigned char *ptr = rcp + offs[tid];

            if (lo < hi) {
                ptr += sizeof(int); /* anchor */
                for (size_t i = lo + 1; i < hi; i += blockSize) {
                    size_t cur_bs = ((i + blockSize) > hi) ? (hi - i) : (size_t)blockSize;
                    unsigned int bc = ptr[0]; ptr++;
                    if (bc != 0) {
                        unsigned int sb = (unsigned int)((cur_bs - 1) / 8 + 1);
                        ptr += sb;
                        ptr += host_fixed_bits_byte_length((unsigned int)cur_bs, bc);
                    }
                }
            }
            size_t end_pos = (size_t)(ptr - cmpBytes);
            if (end_pos > cmpTotalSize) cmpTotalSize = end_pos;
        }
    }

    /* Allocate device memory */
    unsigned char *d_cmpBytes;
    T *d_newData;
    size_t *d_offsets;

    CUDA_CHECK(cudaMalloc(&d_cmpBytes, cmpTotalSize));
    CUDA_CHECK(cudaMalloc(&d_newData, nbEle * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_offsets, nbOMPThreads * sizeof(size_t)));

    /* Copy to device */
    CUDA_CHECK(cudaMemcpy(d_cmpBytes, cmpBytes, cmpTotalSize,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, offs, nbOMPThreads * sizeof(size_t),
                          cudaMemcpyHostToDevice));

    /* Launch kernel: one CUDA thread per OMP thread */
    unsigned char *d_rcp = d_cmpBytes + hdr_size;
    int cudaBlkSize = 256;
    int numCudaBlocks = (int)((nbOMPThreads + cudaBlkSize - 1) / cudaBlkSize);

    kernel_decompress_threadblock<T><<<numCudaBlocks, cudaBlkSize>>>(
        d_newData, d_rcp, d_offsets, nbEle, absErrBound,
        blockSize, nbOMPThreads);
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back */
    CUDA_CHECK(cudaMemcpy(hostOut, d_newData, nbEle * sizeof(T),
                          cudaMemcpyDeviceToHost));

    /* Cleanup */
    cudaFree(d_cmpBytes);
    cudaFree(d_newData);
    cudaFree(d_offsets);
}

/* ================================================================== */
/*  Public C-linkage API                                               */
/* ================================================================== */

extern "C" {

/* ---- Random-access float ---- */

float *szp_cuda_float_decompress_randomaccess(
    size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes)
{
    float *newData = (float *)malloc(nbEle * sizeof(float));
    if (!newData) return NULL;
    cuda_decompress_randomaccess_impl<float>(newData, nbEle, absErrBound,
                                              blockSize, cmpBytes);
    return newData;
}

void szp_cuda_float_decompress_randomaccess_arg(
    float *newData, size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes)
{
    cuda_decompress_randomaccess_impl<float>(newData, nbEle, absErrBound,
                                              blockSize, cmpBytes);
}

/* ---- Random-access double ---- */

double *szp_cuda_double_decompress_randomaccess(
    size_t nbEle, double absErrBound, int blockSize,
    unsigned char *cmpBytes)
{
    double *newData = (double *)malloc(nbEle * sizeof(double));
    if (!newData) return NULL;
    cuda_decompress_randomaccess_impl<double>(newData, nbEle, absErrBound,
                                               blockSize, cmpBytes);
    return newData;
}

void szp_cuda_double_decompress_randomaccess_arg(
    double *newData, size_t nbEle, double absErrBound, int blockSize,
    unsigned char *cmpBytes)
{
    cuda_decompress_randomaccess_impl<double>(newData, nbEle, absErrBound,
                                               blockSize, cmpBytes);
}

/* ---- Threadblock float ---- */

float *szp_cuda_float_decompress_threadblock(
    size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes)
{
    float *newData = (float *)malloc(nbEle * sizeof(float));
    if (!newData) return NULL;
    cuda_decompress_threadblock_impl<float>(newData, nbEle, absErrBound,
                                             blockSize, cmpBytes);
    return newData;
}

void szp_cuda_float_decompress_threadblock_arg(
    float *newData, size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes)
{
    cuda_decompress_threadblock_impl<float>(newData, nbEle, absErrBound,
                                             blockSize, cmpBytes);
}

/* ---- Threadblock double ---- */

double *szp_cuda_double_decompress_threadblock(
    size_t nbEle, double absErrBound, int blockSize,
    unsigned char *cmpBytes)
{
    double *newData = (double *)malloc(nbEle * sizeof(double));
    if (!newData) return NULL;
    cuda_decompress_threadblock_impl<double>(newData, nbEle, absErrBound,
                                              blockSize, cmpBytes);
    return newData;
}

void szp_cuda_double_decompress_threadblock_arg(
    double *newData, size_t nbEle, double absErrBound, int blockSize,
    unsigned char *cmpBytes)
{
    cuda_decompress_threadblock_impl<double>(newData, nbEle, absErrBound,
                                              blockSize, cmpBytes);
}

/* ---- Topology-preserved float ---- */

void szp_cuda_float_decompress_randomaccess_topology_preserved(
    float **newData, size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes, int **FN)
{
    if (!cmpBytes || nbEle == 0) {
        *newData = NULL;
        *FN = NULL;
        return;
    }

    *newData = (float *)malloc(nbEle * sizeof(float));
    *FN      = (int *)calloc(nbEle, sizeof(int));
    if (!*newData || !*FN) {
        free(*newData); free(*FN);
        *newData = NULL; *FN = NULL;
        return;
    }

    /* Scan block offsets */
    size_t numBlocks = 0;
    unsigned int nbOMPThreads = 0;
    size_t *h_block_offsets = scan_randomaccess_topo_block_offsets(
        cmpBytes, nbEle, blockSize, &numBlocks, &nbOMPThreads);
    if (!h_block_offsets) return;

    size_t hdr_size = nbOMPThreads * sizeof(size_t);

    /* Compute total compressed size by walking to end of last block */
    size_t cmpTotalSize = hdr_size;
    if (numBlocks > 0) {
        const unsigned char *rcp = cmpBytes + hdr_size;
        const unsigned char *ptr = rcp + h_block_offsets[numBlocks - 1];
        size_t last_start = (numBlocks - 1) * blockSize;
        size_t last_bs = ((last_start + blockSize) > nbEle)
                         ? (nbEle - last_start) : (size_t)blockSize;
        /* Skip anchor */
        ptr += sizeof(int);
        unsigned int actual_n = (last_bs > 1) ? (unsigned int)(last_bs - 1) : 0;
        unsigned int bc = ptr[0]; ptr++;
        if (bc != 0 && actual_n > 0) {
            ptr += (actual_n + 7) / 8;
            ptr += host_fixed_bits_byte_length(actual_n, bc);
        }
        ptr += (2 * (unsigned int)last_bs + 7) / 8;
        cmpTotalSize = (size_t)(ptr - cmpBytes);
    }

    /* Allocate device memory */
    unsigned char *d_cmpBytes;
    float *d_newData;
    int *d_fnData;
    size_t *d_block_offsets;

    CUDA_CHECK(cudaMalloc(&d_cmpBytes, cmpTotalSize));
    CUDA_CHECK(cudaMalloc(&d_newData, nbEle * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fnData, nbEle * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, numBlocks * sizeof(size_t)));

    CUDA_CHECK(cudaMemset(d_fnData, 0, nbEle * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_cmpBytes, cmpBytes, cmpTotalSize,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_offsets, h_block_offsets,
                          numBlocks * sizeof(size_t),
                          cudaMemcpyHostToDevice));

    unsigned char *d_rcp = d_cmpBytes + hdr_size;
    int cudaBlkSize = 256;
    int numCudaBlocks = (int)((numBlocks + cudaBlkSize - 1) / cudaBlkSize);

    kernel_decompress_randomaccess_topo<<<numCudaBlocks, cudaBlkSize>>>(
        d_newData, d_fnData, d_rcp, d_block_offsets, nbEle,
        absErrBound, blockSize, numBlocks);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(*newData, d_newData, nbEle * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(*FN, d_fnData, nbEle * sizeof(int),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_cmpBytes);
    cudaFree(d_newData);
    cudaFree(d_fnData);
    cudaFree(d_block_offsets);
    free(h_block_offsets);
}

/* ---- Sort-position decompression ---- */

int *szp_cuda_decompress_sort_positions(
    unsigned char *cmpBytes, size_t critical_count, int blockSize)
{
    if (!cmpBytes || critical_count == 0) return NULL;

    int *newData = (int *)malloc(critical_count * sizeof(int));
    if (!newData) return NULL;

    /* Scan block offsets — same format as random-access */
    size_t numBlocks = 0;
    unsigned int nbOMPThreads = 0;
    size_t *h_block_offsets = scan_randomaccess_block_offsets(
        cmpBytes, critical_count, blockSize, &numBlocks, &nbOMPThreads);
    if (!h_block_offsets) { free(newData); return NULL; }

    size_t hdr_size = nbOMPThreads * sizeof(size_t);

    /* Compute total compressed size */
    size_t cmpTotalSize = hdr_size;
    if (numBlocks > 0) {
        const unsigned char *rcp = cmpBytes + hdr_size;
        const unsigned char *ptr = rcp + h_block_offsets[numBlocks - 1];
        size_t last_start = (numBlocks - 1) * blockSize;
        size_t last_bs = ((last_start + blockSize) > critical_count)
                         ? (critical_count - last_start) : (size_t)blockSize;
        ptr += sizeof(int);
        if (last_bs > 1) {
            unsigned int bc = ptr[0]; ptr++;
            if (bc != 0) {
                unsigned int sb = (unsigned int)((last_bs - 2) / 8 + 1);
                ptr += sb;
                ptr += host_fixed_bits_byte_length((unsigned int)(last_bs - 1), bc);
            }
        }
        cmpTotalSize = (size_t)(ptr - cmpBytes);
    }

    /* Device memory */
    unsigned char *d_cmpBytes;
    int *d_newData;
    size_t *d_block_offsets;

    CUDA_CHECK(cudaMalloc(&d_cmpBytes, cmpTotalSize));
    CUDA_CHECK(cudaMalloc(&d_newData, critical_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, numBlocks * sizeof(size_t)));

    CUDA_CHECK(cudaMemcpy(d_cmpBytes, cmpBytes, cmpTotalSize,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_offsets, h_block_offsets,
                          numBlocks * sizeof(size_t),
                          cudaMemcpyHostToDevice));

    unsigned char *d_rcp = d_cmpBytes + hdr_size;
    int cudaBlkSize = 256;
    int numCudaBlocks = (int)((numBlocks + cudaBlkSize - 1) / cudaBlkSize);

    kernel_decompress_sort_positions<<<numCudaBlocks, cudaBlkSize>>>(
        d_newData, d_rcp, d_block_offsets, critical_count,
        blockSize, numBlocks);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(newData, d_newData, critical_count * sizeof(int),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_cmpBytes);
    cudaFree(d_newData);
    cudaFree(d_block_offsets);
    free(h_block_offsets);

    return newData;
}

} /* extern "C" */
