/**
 *  @file szp_detect_threads.h
 *  @brief Auto-detect nbThreads from compressed data offset table.
 *
 *  The SZp compressed format stores per-thread byte offsets at the beginning
 *  of the compressed buffer, but does NOT store how many threads were used.
 *  This helper detects nbThreads by trying candidates and validating that
 *  all blocks can be walked without going out of bounds.
 */

#ifndef _SZP_DETECT_THREADS_H
#define _SZP_DETECT_THREADS_H

#include <stddef.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute the byte length of fixed-length bit-packed data.
 * Matches Jiajun_save_fixed_length_bits output size.
 */
static inline unsigned int
szp_fixed_bits_byte_length(unsigned int totalElements, unsigned int bit_count)
{
    unsigned int byte_count    = bit_count / 8;
    unsigned int remainder_bit = bit_count % 8;
    unsigned int byte_offset   = byte_count * totalElements;
    if (remainder_bit == 0)
        return byte_offset;
    return byte_offset + (remainder_bit * totalElements - 1) / 8 + 1;
}

/**
 * Validate a candidate nbThreads for random-access compressed data.
 * Walks ALL blocks for all threads and checks no pointer exceeds bufEnd.
 *
 * @param cmpBytes   compressed buffer starting at offset table
 * @param nbEle      total number of elements
 * @param blockSize  compression block size
 * @param try_nt     candidate nbThreads to validate
 * @param bufLimit   upper bound on valid buffer size (bytes from cmpBytes)
 * @return 1 if valid, 0 if not
 */
static inline int
szp_validate_nbThreads_randomaccess(const unsigned char *cmpBytes,
                                     size_t nbEle, int blockSize,
                                     unsigned int try_nt,
                                     size_t bufLimit)
{
    const size_t *offs = (const size_t *)cmpBytes;
    size_t hdr_size = (size_t)try_nt * sizeof(size_t);
    const unsigned char *rcp = cmpBytes + hdr_size;
    const unsigned char *bufEnd = cmpBytes + bufLimit;

    /* Use block-based distribution (matches compressor) */
    size_t num_blocks = ((size_t)nbEle + blockSize - 1) / blockSize;
    size_t blocks_per_thread = (num_blocks + try_nt - 1) / try_nt;

    for (unsigned int tid = 0; tid < try_nt; tid++) {
        size_t start_block = tid * blocks_per_thread;
        size_t end_block   = (tid + 1) * blocks_per_thread;
        if (end_block > num_blocks) end_block = num_blocks;

        const unsigned char *ptr = rcp + offs[tid];
        if (ptr >= bufEnd || ptr < rcp) return 0;

        for (size_t bidx = start_block; bidx < end_block; bidx++) {
            size_t i = bidx * blockSize;
            if (i >= nbEle) break;
            size_t cur_bs = ((i + blockSize) > nbEle) ? (nbEle - i) : (size_t)blockSize;

            if (ptr + sizeof(int) > bufEnd) return 0;
            ptr += sizeof(int); /* anchor */

            if (cur_bs > 1) {
                if (ptr >= bufEnd) return 0;
                unsigned int bc = ptr[0]; ptr++;
                if (bc > 32) return 0; /* bit_count sanity */
                if (bc != 0) {
                    unsigned int n = (unsigned int)(cur_bs - 1);
                    unsigned int sb = (n + 7) / 8;
                    unsigned int mb = szp_fixed_bits_byte_length(n, bc);
                    ptr += sb + mb;
                    if (ptr > bufEnd) return 0;
                }
            }
        }
    }
    return 1;
}

/**
 * Validate a candidate nbThreads for threadblock (non-random-access) data.
 * Each thread's chunk starts with ONE int32 anchor, then blocks carry
 * only [bit_count][signs][magnitudes] with prefix-sum chaining.
 */
static inline int
szp_validate_nbThreads_threadblock(const unsigned char *cmpBytes,
                                    size_t nbEle, int blockSize,
                                    unsigned int try_nt,
                                    size_t bufLimit)
{
    const size_t *offs = (const size_t *)cmpBytes;
    size_t hdr_size = (size_t)try_nt * sizeof(size_t);
    const unsigned char *rcp = cmpBytes + hdr_size;
    const unsigned char *bufEnd = cmpBytes + bufLimit;

    size_t threadblocksize = nbEle / try_nt;

    for (unsigned int tid = 0; tid < try_nt; tid++) {
        size_t lo = tid * threadblocksize;
        size_t hi = (tid == try_nt - 1) ? nbEle : (tid + 1) * threadblocksize;

        const unsigned char *ptr = rcp + offs[tid];
        if (ptr >= bufEnd || ptr < rcp) return 0;

        if (lo < hi) {
            /* First element: raw anchor */
            if (ptr + sizeof(int) > bufEnd) return 0;
            ptr += sizeof(int);

            for (size_t i = lo + 1; i < hi; i += blockSize) {
                size_t cur_bs = ((i + blockSize) > hi) ? (hi - i) : (size_t)blockSize;
                if (ptr >= bufEnd) return 0;
                unsigned int bc = ptr[0]; ptr++;
                if (bc > 32) return 0;
                if (bc != 0) {
                    unsigned int n = (unsigned int)cur_bs;
                    unsigned int sb = (n + 7) / 8;
                    unsigned int mb = szp_fixed_bits_byte_length(n, bc);
                    ptr += sb + mb;
                    if (ptr > bufEnd) return 0;
                }
            }
        }
    }
    return 1;
}

/**
 * Auto-detect nbThreads from random-access compressed data.
 * Tries candidates 1..128 and returns the first that passes full validation.
 *
 * @param cmpBytes   compressed buffer starting at offset table
 * @param nbEle      total number of elements
 * @param blockSize  compression block size
 * @return detected nbThreads (>= 1)
 */
static inline unsigned int
szp_detect_nbThreads_randomaccess(const unsigned char *cmpBytes,
                                   size_t nbEle, int blockSize)
{
    const size_t *offs = (const size_t *)cmpBytes;
    size_t maxCmpSize = sizeof(float) * nbEle + sizeof(float);

    if (offs[0] != 0) return 1;

    for (unsigned int try_nt = 1; try_nt <= 128; try_nt++) {
        /* Basic offset validity */
        int valid = 1;
        for (unsigned int k = 1; k < try_nt; k++) {
            if (offs[k] > maxCmpSize || offs[k] < offs[k-1]) {
                valid = 0; break;
            }
        }
        if (!valid) break;

        /* Full walk validation */
        if (szp_validate_nbThreads_randomaccess(cmpBytes, nbEle, blockSize,
                                                 try_nt, maxCmpSize)) {
            return try_nt;
        }
    }
    return 1;
}

/**
 * Auto-detect nbThreads from threadblock (non-random-access) compressed data.
 */
static inline unsigned int
szp_detect_nbThreads_threadblock(const unsigned char *cmpBytes,
                                  size_t nbEle, int blockSize)
{
    const size_t *offs = (const size_t *)cmpBytes;
    size_t maxCmpSize = sizeof(float) * nbEle + sizeof(float);

    if (offs[0] != 0) return 1;

    for (unsigned int try_nt = 1; try_nt <= 128; try_nt++) {
        int valid = 1;
        for (unsigned int k = 1; k < try_nt; k++) {
            if (offs[k] > maxCmpSize || offs[k] < offs[k-1]) {
                valid = 0; break;
            }
        }
        if (!valid) break;

        if (szp_validate_nbThreads_threadblock(cmpBytes, nbEle, blockSize,
                                                try_nt, maxCmpSize)) {
            return try_nt;
        }
    }
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* _SZP_DETECT_THREADS_H */
