/**
 *  @file szp_cuda_common.cuh
 *  @brief CUDA device utilities for topoSZp — bit-packing, reductions, prefix sums.
 *
 *  All bit-packing layouts are byte-compatible with the CPU (OpenMP) implementation
 *  in szp_TypeManager.cc so that data compressed on the CPU can be decompressed on
 *  the GPU and vice-versa.
 */

#pragma once
#ifndef SZP_CUDA_COMMON_CUH
#define SZP_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

/* ================================================================
 *  Error-checking macro
 * ================================================================ */

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

namespace szp_cuda {

/* ================================================================
 *  Union types (mirrors szp_defines.h)
 * ================================================================ */

union lfloat {
    float         value;
    unsigned int  ivalue;
    unsigned char byte[4];
};

union ldouble {
    double         value;
    unsigned long  lvalue;
    unsigned char  byte[8];
};

/* ================================================================
 *  Byte / float / double conversion  (native endian, no swap)
 *
 *  The CPU code stores the first quantized value in each block with
 *  a raw memcpy(&dst, &int_val, 4) — native endian.  We match that.
 * ================================================================ */

__device__ __forceinline__
void device_floatToBytes(unsigned char *b, float num)
{
    lfloat buf;
    buf.value = num;
    b[0] = buf.byte[0];
    b[1] = buf.byte[1];
    b[2] = buf.byte[2];
    b[3] = buf.byte[3];
}

__device__ __forceinline__
float device_bytesToFloat(const unsigned char *b)
{
    lfloat buf;
    buf.byte[0] = b[0];
    buf.byte[1] = b[1];
    buf.byte[2] = b[2];
    buf.byte[3] = b[3];
    return buf.value;
}

__device__ __forceinline__
void device_doubleToBytes(unsigned char *b, double num)
{
    ldouble buf;
    buf.value = num;
    #pragma unroll
    for (int i = 0; i < 8; i++) b[i] = buf.byte[i];
}

__device__ __forceinline__
double device_bytesToDouble(const unsigned char *b)
{
    ldouble buf;
    #pragma unroll
    for (int i = 0; i < 8; i++) buf.byte[i] = b[i];
    return buf.value;
}

/* Big-endian float encode used for the absErrBound header byte. */
__device__ __forceinline__
void device_floatToBytes_bigEndian(unsigned char *b, float num)
{
    lfloat buf;
    buf.value = num;
    b[0] = buf.byte[3];
    b[1] = buf.byte[2];
    b[2] = buf.byte[1];
    b[3] = buf.byte[0];
}

__device__ __forceinline__
float device_bytesToFloat_bigEndian(const unsigned char *b)
{
    lfloat buf;
    buf.byte[3] = b[0];
    buf.byte[2] = b[1];
    buf.byte[1] = b[2];
    buf.byte[0] = b[3];
    return buf.value;
}

__device__ __forceinline__
void device_doubleToBytes_bigEndian(unsigned char *b, double num)
{
    ldouble buf;
    buf.value = num;
    #pragma unroll
    for (int i = 0; i < 8; i++) b[i] = buf.byte[7 - i];
}

__device__ __forceinline__
double device_bytesToDouble_bigEndian(const unsigned char *b)
{
    ldouble buf;
    #pragma unroll
    for (int i = 0; i < 8; i++) buf.byte[7 - i] = b[i];
    return buf.value;
}

/* ================================================================
 *  Size helpers
 * ================================================================ */

/// Number of bytes needed to pack `n` 1-bit values.
__device__ __host__ __forceinline__
unsigned int device_compute_sign_byte_length(unsigned int n)
{
    return (n + 7u) / 8u;
}

/// Total bytes produced by Jiajun_save_fixed_length_bits for `n` elements
/// at `bit_count` bits each.
__device__ __host__ __forceinline__
unsigned int device_compute_fixed_bits_byte_length(unsigned int n,
                                                   unsigned int bit_count)
{
    if (n == 0 || bit_count == 0) return 0;
    unsigned int byte_count    = bit_count / 8;
    unsigned int remainder_bit = bit_count % 8;
    unsigned int full_part     = byte_count * n;
    if (remainder_bit == 0) return full_part;
    unsigned int remainder_part = (remainder_bit * n + 7u) / 8u;
    return full_part + remainder_part;
}

/* ================================================================
 *  1-bit packing / unpacking  (MSB-first, matches CPU)
 *
 *  Byte layout produced by convertIntArray2ByteArray_fast_1b_args:
 *    element 0  →  bit 7 of byte 0
 *    element 1  →  bit 6 of byte 0
 *    ...
 *    element 7  →  bit 0 of byte 0
 *    element 8  →  bit 7 of byte 1
 *    ...
 * ================================================================ */

/// Extract the 1-bit value for element `idx` from packed byte array.
__device__ __forceinline__
unsigned char device_extract_1bit(const unsigned char *byteArray,
                                  unsigned int          idx)
{
    unsigned int byteIdx   = idx / 8u;
    unsigned int bitOffset = 7u - (idx % 8u);          // MSB-first
    return (byteArray[byteIdx] >> bitOffset) & 0x01u;
}

/// Atomically set the 1-bit value for element `idx`.
/// `val` must be 0 or 1.
__device__ __forceinline__
void device_pack_1bit(unsigned char val,
                      unsigned int  idx,
                      unsigned char *byteArray)
{
    unsigned int byteIdx   = idx / 8u;
    unsigned int bitOffset = 7u - (idx % 8u);
    // Use atomicOr for thread-safe packing when multiple threads share a byte.
    if (val) {
        atomicOr((unsigned int *)(byteArray + (byteIdx & ~3u)),
                 ((unsigned int)1u) << (8u * (byteIdx & 3u) + bitOffset));
    }
}

/// Pack 1-bit when the caller guarantees no two threads write the same byte
/// (e.g. one thread packs the full byte).  Faster, no atomic.
__device__ __forceinline__
void device_pack_1bit_exclusive(unsigned char val,
                                unsigned int  idx,
                                unsigned char *byteArray)
{
    unsigned int byteIdx   = idx / 8u;
    unsigned int bitOffset = 7u - (idx % 8u);
    if (val) byteArray[byteIdx] |= (1u << bitOffset);
}

/* ================================================================
 *  2-bit packing / unpacking  (MSB-first, matches CPU)
 *
 *  Byte layout produced by convertIntArray2ByteArray_fast_2b:
 *    element 0  →  bits 7-6 of byte 0
 *    element 1  →  bits 5-4 of byte 0
 *    element 2  →  bits 3-2 of byte 0
 *    element 3  →  bits 1-0 of byte 0
 *    element 4  →  bits 7-6 of byte 1
 *    ...
 * ================================================================ */

__device__ __forceinline__
unsigned char device_extract_2bit(const unsigned char *byteArray,
                                  unsigned int          idx)
{
    unsigned int byteIdx = idx / 4u;
    unsigned int shift   = 6u - 2u * (idx % 4u);        // 6, 4, 2, 0
    return (byteArray[byteIdx] >> shift) & 0x03u;
}

__device__ __forceinline__
void device_pack_2bit(unsigned char val,
                      unsigned int  idx,
                      unsigned char *byteArray)
{
    unsigned int byteIdx = idx / 4u;
    unsigned int shift   = 6u - 2u * (idx % 4u);
    atomicOr((unsigned int *)(byteArray + (byteIdx & ~3u)),
             ((unsigned int)(val & 0x03u)) << (8u * (byteIdx & 3u) + shift));
}

__device__ __forceinline__
void device_pack_2bit_exclusive(unsigned char val,
                                unsigned int  idx,
                                unsigned char *byteArray)
{
    unsigned int byteIdx = idx / 4u;
    unsigned int shift   = 6u - 2u * (idx % 4u);
    byteArray[byteIdx] |= (val & 0x03u) << shift;
}

/* ================================================================
 *  Generic N-bit packing / unpacking  (Jiajun layout)
 *
 *  The CPU's Jiajun_save_fixed_length_bits splits each value into:
 *    full_bytes  = value >> remainder_bit   (byte_count bytes, LE)
 *    remainder   = value & ((1<<remainder_bit)-1)
 *
 *  Storage layout (consecutive in memory):
 *    [full-byte section]  byte_count * totalElements bytes, LE per element
 *    [remainder section]  remainder_bit-wide values packed MSB-first
 *                         (exactly the Nb-bit conversion layout)
 *
 *  Nb-bit MSB-first packing: element n at width W starts at bit
 *  position n*W counted from MSB of the first byte in the remainder
 *  section.  So:
 *    byteIndex  = n*W / 8
 *    bitInByte  = n*W % 8          (0 = MSB position)
 *    The W-bit value sits at bits [7-bitInByte .. 7-bitInByte-W+1].
 *    It may span two bytes when bitInByte + W > 8.
 * ================================================================ */

/// Extract the N-bit value for element `idx` (random-access, no serial walk).
///
/// @param byteArray      Pointer to the start of the packed section
///                        (right after the bit_count byte in the block).
///                        This is the sign section only when called for signs;
///                        for magnitudes it points to the magnitude section.
/// @param idx            Element index within this block (0-based).
/// @param totalElements  Total elements in this section (block_size - 1 for
///                        delta-encoded blocks).
/// @param bit_count      Number of bits per element (1..31).
__device__ __forceinline__
unsigned int device_extract_fixed_length_bits(const unsigned char *byteArray,
                                              unsigned int idx,
                                              unsigned int totalElements,
                                              unsigned int bit_count)
{
    unsigned int byte_count    = bit_count / 8u;
    unsigned int remainder_bit = bit_count % 8u;

    /* ---------- full-byte portion (little-endian per element) ---------- */
    unsigned int full_value = 0;
    if (byte_count > 0) {
        unsigned int base = idx * byte_count;
        #pragma unroll 4
        for (unsigned int j = 0; j < byte_count; j++) {
            full_value |= ((unsigned int)byteArray[base + j]) << (8u * j);
        }
        full_value <<= remainder_bit;
    }

    /* ---------- remainder portion (Nb-bit MSB-first) ---------- */
    unsigned int rem_value = 0;
    if (remainder_bit > 0) {
        unsigned int rem_section_start = byte_count * totalElements;
        unsigned int bit_position = idx * remainder_bit;     // from MSB
        unsigned int rem_byte  = rem_section_start + bit_position / 8u;
        unsigned int rem_shift = bit_position % 8u;          // bits from MSB

        // Read 2 bytes covering the field (safe: the packed section is
        // always followed by either more block data or end-of-buffer padding).
        unsigned int raw = ((unsigned int)byteArray[rem_byte]) << 8u;
        raw |= (unsigned int)byteArray[rem_byte + 1u];

        unsigned int shift = 16u - rem_shift - remainder_bit;
        unsigned int mask  = (1u << remainder_bit) - 1u;
        rem_value = (raw >> shift) & mask;
    }

    return full_value | rem_value;
}

/// Pack a single N-bit value at element `idx` (random-access).
///
/// IMPORTANT: the caller must zero-initialise the output buffer before
/// packing, because we use OR to accumulate bits.
__device__ __forceinline__
void device_save_fixed_length_bits(unsigned int  value,
                                   unsigned int  idx,
                                   unsigned int  totalElements,
                                   unsigned int  bit_count,
                                   unsigned char *byteArray)
{
    unsigned int byte_count    = bit_count / 8u;
    unsigned int remainder_bit = bit_count % 8u;

    /* ---------- full-byte portion (little-endian) ---------- */
    if (byte_count > 0) {
        unsigned int full = value >> remainder_bit;
        unsigned int base = idx * byte_count;
        #pragma unroll 4
        for (unsigned int j = 0; j < byte_count; j++) {
            byteArray[base + j] = (unsigned char)(full & 0xFFu);
            full >>= 8u;
        }
    }

    /* ---------- remainder portion (Nb-bit MSB-first) ---------- */
    if (remainder_bit > 0) {
        unsigned int rem = value & ((1u << remainder_bit) - 1u);
        unsigned int rem_section_start = byte_count * totalElements;
        unsigned int bit_position = idx * remainder_bit;
        unsigned int rem_byte  = rem_section_start + bit_position / 8u;
        unsigned int rem_shift = bit_position % 8u;

        // Place `remainder_bit` bits starting at `rem_shift` from MSB in a
        // 16-bit window, then OR into the two bytes.
        unsigned int placed = rem << (16u - rem_shift - remainder_bit);
        atomicOr((unsigned int *)(byteArray + (rem_byte & ~3u)),
                 ((placed >> 8u) & 0xFFu) << (8u * (rem_byte & 3u)));
        if (rem_shift + remainder_bit > 8u) {
            unsigned int next = rem_byte + 1u;
            atomicOr((unsigned int *)(byteArray + (next & ~3u)),
                     (placed & 0xFFu) << (8u * (next & 3u)));
        }
    }
}

/// Non-atomic version for when the caller guarantees exclusive byte access
/// (e.g. single-thread-per-block serialised packing).
__device__ __forceinline__
void device_save_fixed_length_bits_exclusive(unsigned int  value,
                                             unsigned int  idx,
                                             unsigned int  totalElements,
                                             unsigned int  bit_count,
                                             unsigned char *byteArray)
{
    unsigned int byte_count    = bit_count / 8u;
    unsigned int remainder_bit = bit_count % 8u;

    if (byte_count > 0) {
        unsigned int full = value >> remainder_bit;
        unsigned int base = idx * byte_count;
        #pragma unroll 4
        for (unsigned int j = 0; j < byte_count; j++) {
            byteArray[base + j] = (unsigned char)(full & 0xFFu);
            full >>= 8u;
        }
    }

    if (remainder_bit > 0) {
        unsigned int rem = value & ((1u << remainder_bit) - 1u);
        unsigned int rem_section_start = byte_count * totalElements;
        unsigned int bit_position = idx * remainder_bit;
        unsigned int rem_byte  = rem_section_start + bit_position / 8u;
        unsigned int rem_shift = bit_position % 8u;

        unsigned int placed = rem << (16u - rem_shift - remainder_bit);
        byteArray[rem_byte] |= (unsigned char)((placed >> 8u) & 0xFFu);
        if (rem_shift + remainder_bit > 8u) {
            byteArray[rem_byte + 1u] |= (unsigned char)(placed & 0xFFu);
        }
    }
}

/* ================================================================
 *  Serial bit-pack / unpack helpers (used when a single thread
 *  processes a whole block — mirrors the CPU serial code exactly)
 * ================================================================ */

/// Serial 1-bit pack of `n` unsigned-char values into byteArray.
/// Returns number of bytes written. Matches convertIntArray2ByteArray_fast_1b_args.
__device__ __forceinline__
unsigned int device_serial_pack_1b(const unsigned char *intArray,
                                   unsigned int n,
                                   unsigned char *result)
{
    unsigned int byteLen = (n + 7u) / 8u;
    unsigned int idx = 0;
    for (unsigned int i = 0; i < byteLen; i++) {
        unsigned char tmp = 0;
        for (unsigned int j = 0; j < 8u && idx < n; j++, idx++) {
            tmp |= (intArray[idx] & 0x01u) << (7u - j);
        }
        result[i] = tmp;
    }
    return byteLen;
}

/// Serial 1-bit unpack. Matches convertByteArray2IntArray_fast_1b_args.
__device__ __forceinline__
void device_serial_unpack_1b(unsigned int n,
                             const unsigned char *byteArray,
                             unsigned char *intArray)
{
    unsigned int idx = 0;
    unsigned int byteLen = (n + 7u) / 8u;
    for (unsigned int i = 0; i < byteLen && idx < n; i++) {
        unsigned char tmp = byteArray[i];
        for (int j = 7; j >= 0 && idx < n; j--, idx++) {
            intArray[idx] = (tmp >> j) & 0x01u;
        }
    }
}

/// Serial 2-bit pack. Matches convertIntArray2ByteArray_fast_2b_args.
__device__ __forceinline__
unsigned int device_serial_pack_2b(const unsigned char *intArray,
                                   unsigned int n,
                                   unsigned char *result)
{
    unsigned int byteLen = (n * 2u + 7u) / 8u;
    unsigned int idx = 0;
    for (unsigned int i = 0; i < byteLen; i++) {
        unsigned char tmp = 0;
        for (unsigned int j = 0; j < 4u && idx < n; j++, idx++) {
            tmp |= (intArray[idx] & 0x03u) << (6u - 2u * j);
        }
        result[i] = tmp;
    }
    return byteLen;
}

/// Serial 2-bit unpack. Matches convertByteArray2IntArray_fast_2b (the _args variant).
__device__ __forceinline__
void device_serial_unpack_2b(unsigned int n,
                             const unsigned char *byteArray,
                             unsigned char *intArray)
{
    unsigned int idx = 0;
    unsigned int byteLen = (n * 2u + 7u) / 8u;
    for (unsigned int i = 0; i < byteLen && idx < n; i++) {
        unsigned char tmp = byteArray[i];
        for (unsigned int j = 0; j < 4u && idx < n; j++, idx++) {
            intArray[idx] = (tmp >> (6u - 2u * j)) & 0x03u;
        }
    }
}

/// Serial fixed-length-bit pack.  Matches Jiajun_save_fixed_length_bits.
/// Returns number of bytes written.
__device__ __forceinline__
unsigned int device_serial_save_fixed_length_bits(unsigned int *values,
                                                  unsigned int  n,
                                                  unsigned char *result,
                                                  unsigned int  bit_count)
{
    unsigned int byte_count    = bit_count / 8u;
    unsigned int remainder_bit = bit_count % 8u;
    unsigned int byteLength    = device_compute_fixed_bits_byte_length(n, bit_count);

    // --- full-byte section (LE per element) ---
    if (byte_count > 0) {
        unsigned int pos = 0;
        for (unsigned int elem = 0; elem < n; elem++) {
            unsigned int tmp = values[elem] >> remainder_bit;
            for (unsigned int j = 0; j < byte_count; j++) {
                result[pos++] = (unsigned char)(tmp & 0xFFu);
                tmp >>= 8u;
            }
        }
    }

    // --- remainder section (Nb-bit MSB-first) ---
    if (remainder_bit > 0) {
        unsigned int rem_start = byte_count * n;
        // Prepare masked remainder values
        unsigned int mask = (1u << remainder_bit) - 1u;

        // Zero the remainder section
        unsigned int remBytes = byteLength - rem_start;
        for (unsigned int i = 0; i < remBytes; i++) result[rem_start + i] = 0;

        for (unsigned int elem = 0; elem < n; elem++) {
            unsigned int rem = (byte_count > 0)
                ? (values[elem] & mask)
                : values[elem];
            unsigned int bit_pos = elem * remainder_bit;
            unsigned int bIdx = rem_start + bit_pos / 8u;
            unsigned int bOff = bit_pos % 8u;

            unsigned int placed = rem << (16u - bOff - remainder_bit);
            result[bIdx] |= (unsigned char)((placed >> 8u) & 0xFFu);
            if (bOff + remainder_bit > 8u) {
                result[bIdx + 1u] |= (unsigned char)(placed & 0xFFu);
            }
        }
    }

    return byteLength;
}

/// Serial fixed-length-bit extract.  Matches Jiajun_extract_fixed_length_bits.
/// Returns number of bytes consumed.
__device__ __forceinline__
unsigned int device_serial_extract_fixed_length_bits(const unsigned char *packed,
                                                     unsigned int n,
                                                     unsigned int *values,
                                                     unsigned int bit_count)
{
    unsigned int byte_count    = bit_count / 8u;
    unsigned int remainder_bit = bit_count % 8u;
    unsigned int byteLength    = device_compute_fixed_bits_byte_length(n, bit_count);

    // --- remainder portion first (CPU does this first too) ---
    if (remainder_bit > 0) {
        unsigned int rem_start = byte_count * n;
        for (unsigned int elem = 0; elem < n; elem++) {
            unsigned int bit_pos = elem * remainder_bit;
            unsigned int bIdx = rem_start + bit_pos / 8u;
            unsigned int bOff = bit_pos % 8u;

            unsigned int raw = ((unsigned int)packed[bIdx]) << 8u;
            if (bIdx + 1u < byteLength)
                raw |= (unsigned int)packed[bIdx + 1u];

            unsigned int shift = 16u - bOff - remainder_bit;
            unsigned int mask  = (1u << remainder_bit) - 1u;
            values[elem] = (raw >> shift) & mask;
        }
    } else {
        // No remainder — zero-init so the OR below is clean
        for (unsigned int i = 0; i < n; i++) values[i] = 0;
    }

    // --- full-byte portion (LE per element), combined via OR ---
    if (byte_count > 0) {
        unsigned int pos = 0;
        for (unsigned int elem = 0; elem < n; elem++) {
            unsigned int tmp = 0;
            for (unsigned int j = 0; j < byte_count; j++) {
                tmp |= ((unsigned int)packed[pos++]) << (8u * j);
            }
            tmp <<= remainder_bit;
            values[elem] |= tmp;
        }
    }

    return byteLength;
}

/* ================================================================
 *  Warp-level inclusive prefix sum (for delta reconstruction)
 * ================================================================ */

__device__ __forceinline__
int warp_prefix_sum_inclusive(int val)
{
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if ((threadIdx.x & 31) >= offset) val += n;
    }
    return val;
}

/// Warp-level inclusive prefix sum, limited to the first `width` lanes.
__device__ __forceinline__
int warp_prefix_sum_inclusive_w(int val, int width)
{
    for (int offset = 1; offset < width; offset <<= 1) {
        int n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if ((threadIdx.x & 31) >= offset) val += n;
    }
    return val;
}

/* ================================================================
 *  Block-level sequential prefix sum in shared memory
 *
 *  For blocks larger than a warp (blockSize > 32), thread 0
 *  performs a sequential scan.  This is used for delta
 *  reconstruction where each element depends on the previous.
 * ================================================================ */

__device__ __forceinline__
void block_prefix_sum_sequential(int *shared_data, unsigned int n)
{
    for (unsigned int i = 1; i < n; i++) {
        shared_data[i] += shared_data[i - 1];
    }
}

/* ================================================================
 *  Shared-memory block-level max reduction
 * ================================================================ */

/// Find the maximum unsigned int across all participating threads.
/// `shared_mem` must have at least `blockDim.x` entries.
/// Returns the result in every thread (broadcast).
__device__ __forceinline__
unsigned int block_reduce_max(unsigned int val,
                              volatile unsigned int *shared_mem,
                              unsigned int tid,
                              unsigned int blockSize)
{
    shared_mem[tid] = val;
    __syncthreads();

    // Tree reduction
    for (unsigned int s = blockSize / 2u; s > 0; s >>= 1u) {
        if (tid < s) {
            if (shared_mem[tid + s] > shared_mem[tid])
                shared_mem[tid] = shared_mem[tid + s];
        }
        __syncthreads();
    }

    return shared_mem[0];  // broadcast to all threads
}

/// Warp-level max reduction (no shared memory).
__device__ __forceinline__
unsigned int warp_reduce_max(unsigned int val)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        unsigned int other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (other > val) val = other;
    }
    return val;  // result valid in lane 0
}

/* ================================================================
 *  Block-level sequential compression of a single block
 *
 *  This mirrors the inner loop of the CPU random-access compressor.
 *  A single thread processes one compression block, writing the
 *  packed output to `out`.  Returns the number of bytes written.
 *
 *  Template parameter T is float or double.
 * ================================================================ */

template <typename T>
__device__ __forceinline__
unsigned int device_compress_block(const T      *data,
                                   unsigned int  block_start,
                                   unsigned int  block_len,   // elements in this block
                                   double        inver_bound,
                                   unsigned char *out,
                                   unsigned char *tmp_sign,    // scratch [block_len-1]
                                   unsigned int  *tmp_mag)     // scratch [block_len-1]
{
    unsigned int bytes_written = 0;

    // First element — raw quantised value
    int prior = (int)((double)data[block_start] * inver_bound);
    // Store as native-endian int (matches CPU memcpy)
    *((int *)out) = prior;
    out += sizeof(int);
    bytes_written += sizeof(int);

    if (block_len <= 1) return bytes_written;

    unsigned int n = block_len - 1;  // number of deltas
    unsigned int max_mag = 0;

    for (unsigned int j = 0; j < n; j++) {
        int current = (int)((double)data[block_start + j + 1] * inver_bound);
        int diff    = current - prior;
        prior       = current;
        if (diff < 0) {
            tmp_sign[j] = 1;
            tmp_mag[j]  = (unsigned int)(-diff);
        } else {
            tmp_sign[j] = 0;
            tmp_mag[j]  = (unsigned int)diff;
        }
        if (tmp_mag[j] > max_mag) max_mag = tmp_mag[j];
    }

    if (max_mag == 0) {
        out[0] = 0;
        bytes_written += 1;
    } else {
        unsigned int bit_count = (unsigned int)floorf(log2f((float)max_mag)) + 1u;
        out[0] = (unsigned char)bit_count;
        out++;
        bytes_written++;

        unsigned int signLen = device_serial_pack_1b(tmp_sign, n, out);
        out += signLen;
        bytes_written += signLen;

        unsigned int magLen = device_serial_save_fixed_length_bits(tmp_mag, n, out, bit_count);
        bytes_written += magLen;
    }

    return bytes_written;
}

/// Same as above but also packs 2-bit critical-type information.
template <typename T>
__device__ __forceinline__
unsigned int device_compress_block_topology(const T             *data,
                                            unsigned int         block_start,
                                            unsigned int         block_len,
                                            double               inver_bound,
                                            const unsigned char *critical_type,
                                            unsigned int         nbEle,
                                            unsigned char       *out,
                                            unsigned char       *tmp_sign,
                                            unsigned int        *tmp_mag,
                                            unsigned char       *tmp_type)
{
    unsigned int bytes_written = 0;

    // First element — raw quantised value
    int prior = (int)((double)data[block_start] * inver_bound);
    *((int *)out) = prior;
    out += sizeof(int);
    bytes_written += sizeof(int);

    unsigned int n = (block_len > 1) ? (block_len - 1) : 0;
    unsigned int max_mag = 0;

    for (unsigned int j = 0; j < n; j++) {
        int current = (int)((double)data[block_start + j + 1] * inver_bound);
        int diff    = current - prior;
        prior       = current;
        if (diff < 0) {
            tmp_sign[j] = 1;
            tmp_mag[j]  = (unsigned int)(-diff);
        } else {
            tmp_sign[j] = 0;
            tmp_mag[j]  = (unsigned int)diff;
        }
        if (tmp_mag[j] > max_mag) max_mag = tmp_mag[j];
    }

    if (max_mag == 0) {
        out[0] = 0;
        out++;
        bytes_written++;
    } else {
        unsigned int bit_count = (unsigned int)floorf(log2f((float)max_mag)) + 1u;
        out[0] = (unsigned char)bit_count;
        out++;
        bytes_written++;

        if (n > 0) {
            unsigned int signLen = device_serial_pack_1b(tmp_sign, n, out);
            out += signLen;
            bytes_written += signLen;

            unsigned int magLen = device_serial_save_fixed_length_bits(tmp_mag, n, out, bit_count);
            out += magLen;
            bytes_written += magLen;
        }
    }

    // 2-bit type information for every element in the block
    for (unsigned int j = 0; j < block_len; j++) {
        unsigned int idx = block_start + j;
        tmp_type[j] = (idx < nbEle) ? critical_type[idx] : 0;
    }
    unsigned int typeLen = device_serial_pack_2b(tmp_type, block_len, out);
    bytes_written += typeLen;

    return bytes_written;
}

/* ================================================================
 *  Block-level sequential decompression of a single block
 *
 *  Mirrors the CPU random-access decompressor inner loop.
 *  A single thread reads from `in` and writes to `out_data`.
 *  Returns the number of compressed bytes consumed.
 * ================================================================ */

template <typename T>
__device__ __forceinline__
unsigned int device_decompress_block(const unsigned char *in,
                                     T                   *out_data,
                                     unsigned int         block_len,
                                     T                    scale,   // absErrBound
                                     unsigned char       *tmp_sign,
                                     unsigned int        *tmp_mag)
{
    unsigned int bytes_read = 0;

    // First element — raw quantised value
    int prior;
    prior = *((const int *)in);
    in += sizeof(int);
    bytes_read += sizeof(int);

    out_data[0] = (T)prior * scale;

    if (block_len <= 1) return bytes_read;

    unsigned int n = block_len - 1;
    unsigned int bit_count = in[0];
    in++;
    bytes_read++;

    if (bit_count == 0) {
        // All deltas zero — fill with the same value
        T val = out_data[0];
        for (unsigned int j = 0; j < n; j++) {
            out_data[1 + j] = val;
        }
    } else {
        unsigned int signLen = (n + 7u) / 8u;
        device_serial_unpack_1b(n, in, tmp_sign);
        in += signLen;
        bytes_read += signLen;

        unsigned int magLen = device_serial_extract_fixed_length_bits(in, n, tmp_mag, bit_count);
        in += magLen;
        bytes_read += magLen;

        for (unsigned int j = 0; j < n; j++) {
            int diff = tmp_sign[j] ? -(int)tmp_mag[j] : (int)tmp_mag[j];
            int current = prior + diff;
            prior = current;
            out_data[1 + j] = (T)current * scale;
        }
    }

    return bytes_read;
}

/// Decompress with 2-bit topology type extraction.
template <typename T>
__device__ __forceinline__
unsigned int device_decompress_block_topology(const unsigned char *in,
                                              T                   *out_data,
                                              int                 *out_type,
                                              unsigned int         block_start,
                                              unsigned int         block_len,
                                              T                    scale,
                                              unsigned char       *tmp_sign,
                                              unsigned int        *tmp_mag,
                                              unsigned char       *tmp_type)
{
    unsigned int bytes_read = 0;

    int prior = *((const int *)in);
    in += sizeof(int);
    bytes_read += sizeof(int);

    out_data[0] = (T)prior * scale;

    unsigned int n = (block_len > 1) ? (block_len - 1) : 0;

    unsigned int bit_count = in[0];
    in++;
    bytes_read++;

    if (bit_count == 0) {
        T val = out_data[0];
        for (unsigned int j = 0; j < n; j++) {
            out_data[1 + j] = val;
        }
    } else {
        if (n > 0) {
            unsigned int signLen = (n + 7u) / 8u;
            device_serial_unpack_1b(n, in, tmp_sign);
            in += signLen;
            bytes_read += signLen;

            unsigned int magLen = device_serial_extract_fixed_length_bits(in, n, tmp_mag, bit_count);
            in += magLen;
            bytes_read += magLen;

            for (unsigned int j = 0; j < n; j++) {
                int diff = tmp_sign[j] ? -(int)tmp_mag[j] : (int)tmp_mag[j];
                int current = prior + diff;
                prior = current;
                out_data[1 + j] = (T)current * scale;
            }
        }
    }

    // Extract 2-bit type information
    unsigned int typeLen = (2u * block_len + 7u) / 8u;
    device_serial_unpack_2b(block_len, in, tmp_type);
    bytes_read += typeLen;

    for (unsigned int j = 0; j < block_len; j++) {
        out_type[block_start + j] = (int)tmp_type[j];
    }

    return bytes_read;
}

/* ================================================================
 *  Integer block decompress (for sort_positions)
 * ================================================================ */

__device__ __forceinline__
unsigned int device_decompress_block_int(const unsigned char *in,
                                         int                 *out_data,
                                         unsigned int         block_len,
                                         unsigned char       *tmp_sign,
                                         unsigned int        *tmp_mag)
{
    unsigned int bytes_read = 0;

    int prior = *((const int *)in);
    in += sizeof(int);
    bytes_read += sizeof(int);

    out_data[0] = prior;

    if (block_len <= 1) return bytes_read;

    unsigned int n = block_len - 1;
    unsigned int bit_count = in[0];
    in++;
    bytes_read++;

    if (bit_count == 0) {
        for (unsigned int j = 0; j < n; j++) {
            out_data[1 + j] = prior;
        }
    } else {
        unsigned int signLen = (n + 7u) / 8u;
        device_serial_unpack_1b(n, in, tmp_sign);
        in += signLen;
        bytes_read += signLen;

        unsigned int magLen = device_serial_extract_fixed_length_bits(in, n, tmp_mag, bit_count);
        in += magLen;
        bytes_read += magLen;

        for (unsigned int j = 0; j < n; j++) {
            int diff = tmp_sign[j] ? -(int)tmp_mag[j] : (int)tmp_mag[j];
            int current = prior + diff;
            prior = current;
            out_data[1 + j] = current;
        }
    }

    return bytes_read;
}

}  // namespace szp_cuda

#endif  // SZP_CUDA_COMMON_CUH
