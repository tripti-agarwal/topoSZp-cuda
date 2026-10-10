/**
 *  @file szp_detect_threads.h
 *  @brief Auto-detect nbThreads from the per-thread offset table of an SZp
 *         stream, and the block-to-thread layout shared by all encoders.
 *
 *  The SZp compressed format stores one byte offset per compressing thread at
 *  the start of the buffer but does NOT store how many threads were used.
 *  A candidate thread count T is accepted only if, walking the blocks that
 *  each thread t < T owns, the walk ends exactly where offset t+1 says the
 *  next thread's data starts (and, when the stream size is known, the last
 *  thread ends exactly at the end of the stream). Byte-level coincidences
 *  that fooled the old "does not run past the buffer" test (common on
 *  fields with many zeros) do not pass this check.
 *
 *  When the stream size is known, use the *_sized variants: they return the
 *  first T whose walk ends exactly at the end of the stream. Without the size,
 *  the largest T that passes all internal boundary checks is returned
 *  (T = 1 has no internal boundaries and always passes, so it is only the
 *  answer when nothing larger fits).
 *
 *  Streams written by this version store the thread count: slot 0 of the
 *  offset table (always offset 0) holds SZP_STREAM_TAG | nbThreads. Decoders
 *  read the count from there and fall back to the detection above only for
 *  older streams, whose slot 0 is 0. szp_read_offset() returns 0 for slot 0
 *  in both cases.
 *
 *  This header is plain C and is shared by the OpenMP and CUDA decoders.
 */

#ifndef _SZP_DETECT_THREADS_H
#define _SZP_DETECT_THREADS_H

#include <stddef.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SZP_MAX_DETECT_THREADS
#define SZP_MAX_DETECT_THREADS 1024
#endif

/* "SZPT" in the high 32 bits of slot 0; the thread count in the low 32 bits. */
#define SZP_STREAM_TAG 0x535A505400000000ULL

/** Write the thread count into slot 0 of an offset table (encoders). */
static inline void
szp_stream_write_tag(unsigned char *cmpBytes, unsigned int nbThreads)
{
    unsigned long long v = SZP_STREAM_TAG | (unsigned long long)nbThreads;
    size_t s = (size_t)v;
    memcpy(cmpBytes, &s, sizeof(size_t));
}

/** Thread count stored in the stream, or 0 for an older untagged stream. */
static inline unsigned int
szp_stream_tagged_threads(const unsigned char *cmpBytes)
{
    size_t s;
    memcpy(&s, cmpBytes, sizeof(size_t));
    unsigned long long v = (unsigned long long)s;
    if ((v & 0xFFFFFFFF00000000ULL) != SZP_STREAM_TAG) return 0;
    return (unsigned int)(v & 0xFFFFFFFFULL);
}

/**
 * Read entry k of the offset table (slot 0 is always offset 0, whether or
 * not it carries the tag). The table is not necessarily 8-byte aligned (e.g.
 * the sort-position stream follows the topology stream at an arbitrary byte
 * offset), so it is read with memcpy.
 */
static inline size_t
szp_read_offset(const unsigned char *cmpBytes, size_t k)
{
    if (k == 0) return 0;
    size_t v;
    memcpy(&v, cmpBytes + k * sizeof(size_t), sizeof(size_t));
    return v;
}

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
 * Blocks [*start, *end) owned by thread tid out of nt, exactly as the
 * random-access encoders (plain, topology, sort positions) assign them:
 * ceil(num_blocks / nt) consecutive blocks per thread, except that with
 * 16 or more threads and fewer than 8 blocks per thread only
 * ceil(num_blocks / 8) threads get work and the rest write nothing.
 */
static inline void
szp_thread_block_range(size_t num_blocks, unsigned int nt, unsigned int tid,
                       size_t *start, size_t *end)
{
    size_t s, e;
    if (nt >= 16 && num_blocks < (size_t)nt * 8) {
        size_t opt = (num_blocks + 7) / 8;
        if (opt < 1) opt = 1;
        size_t bpt = (num_blocks + opt - 1) / opt;
        size_t eff = (size_t)tid % opt;
        s = eff * bpt;
        e = (eff + 1) * bpt;
        if (e > num_blocks) e = num_blocks;
        if ((size_t)tid >= opt) s = e;          /* no work for this thread */
    } else {
        size_t bpt = (num_blocks + nt - 1) / nt;
        s = (size_t)tid * bpt;
        e = s + bpt;
        if (e > num_blocks) e = num_blocks;
    }
    if (s > e) s = e;
    *start = s;
    *end = e;
}

/**
 * Walk blocks [start_block, end_block) of a random-access stream starting at
 * ptr. Every block is [int32 anchor][u8 bit_count] then, if bit_count != 0,
 * the sign bytes and fixed-length magnitudes of the remaining elements; the
 * topology format appends 2-bit types for every element of the block.
 * Returns the position after the last block, or NULL if the data is not a
 * valid block sequence or runs past bufEnd.
 */
static inline const unsigned char *
szp_walk_randomaccess_blocks(const unsigned char *ptr, const unsigned char *bufEnd,
                             size_t start_block, size_t end_block,
                             size_t nbEle, int blockSize, int with_types)
{
    for (size_t bidx = start_block; bidx < end_block; bidx++) {
        size_t i = bidx * (size_t)blockSize;
        if (i >= nbEle) break;
        size_t cur_bs = ((i + blockSize) > nbEle) ? (nbEle - i) : (size_t)blockSize;

        if ((size_t)(bufEnd - ptr) < sizeof(int) + 1) return NULL;
        ptr += sizeof(int);                      /* anchor */
        unsigned int bc = *ptr++;                /* always written by the encoders */
        if (bc > 32) return NULL;
        if (bc != 0) {
            if (cur_bs < 2) return NULL;         /* encoders write 0 for one-element blocks */
            unsigned int n = (unsigned int)(cur_bs - 1);
            size_t len = (size_t)(n + 7) / 8 + szp_fixed_bits_byte_length(n, bc);
            if ((size_t)(bufEnd - ptr) < len) return NULL;
            ptr += len;
        }
        if (with_types) {
            size_t tb = (2 * cur_bs + 7) / 8;
            if ((size_t)(bufEnd - ptr) < tb) return NULL;
            ptr += tb;
        }
    }
    return ptr;
}

/**
 * Check candidate try_nt for a random-access stream (plain, topology or sort
 * positions). bufLimit is the stream size if exact != 0, otherwise an upper
 * bound on it.
 */
static inline int
szp_validate_nbThreads_ra_generic(const unsigned char *cmpBytes, size_t bufLimit, int exact,
                                  size_t nbEle, int blockSize, unsigned int try_nt,
                                  int with_types)
{
    size_t hdr_size = (size_t)try_nt * sizeof(size_t);
    if (hdr_size > bufLimit) return 0;
    const unsigned char *rcp = cmpBytes + hdr_size;
    const unsigned char *bufEnd = cmpBytes + bufLimit;
    size_t num_blocks = (nbEle + (size_t)blockSize - 1) / (size_t)blockSize;

    for (unsigned int tid = 0; tid < try_nt; tid++) {
        size_t seg_off = szp_read_offset(cmpBytes, tid);
        if (seg_off > bufLimit - hdr_size) return 0;
        size_t sb, eb;
        szp_thread_block_range(num_blocks, try_nt, tid, &sb, &eb);
        const unsigned char *end = szp_walk_randomaccess_blocks(rcp + seg_off, bufEnd, sb, eb,
                                                                nbEle, blockSize, with_types);
        if (!end) return 0;
        if (tid + 1 < try_nt) {
            if (end != rcp + szp_read_offset(cmpBytes, tid + 1)) return 0;   /* must meet the next thread's data exactly */
        } else if (exact && end != bufEnd) {
            return 0;
        }
    }
    return 1;
}

/**
 * Check candidate try_nt for a thread-block (non-random-access) stream:
 * each thread owns nbEle/T consecutive elements (the last one also gets the
 * remainder), stored as one int32 anchor followed by blocks of
 * [u8 bit_count][signs][magnitudes].
 */
static inline int
szp_validate_nbThreads_tb_generic(const unsigned char *cmpBytes, size_t bufLimit, int exact,
                                  size_t nbEle, int blockSize, unsigned int try_nt)
{
    size_t hdr_size = (size_t)try_nt * sizeof(size_t);
    if (hdr_size > bufLimit) return 0;
    const unsigned char *rcp = cmpBytes + hdr_size;
    const unsigned char *bufEnd = cmpBytes + bufLimit;
    size_t threadblocksize = nbEle / try_nt;

    for (unsigned int tid = 0; tid < try_nt; tid++) {
        size_t seg_off = szp_read_offset(cmpBytes, tid);
        if (seg_off > bufLimit - hdr_size) return 0;
        size_t lo = tid * threadblocksize;
        size_t hi = (tid == try_nt - 1) ? nbEle : (tid + 1) * threadblocksize;
        const unsigned char *ptr = rcp + seg_off;

        if (lo < hi) {
            if ((size_t)(bufEnd - ptr) < sizeof(int)) return 0;
            ptr += sizeof(int);                  /* first element: raw anchor */
            for (size_t i = lo + 1; i < hi; i += blockSize) {
                size_t cur_bs = ((i + blockSize) > hi) ? (hi - i) : (size_t)blockSize;
                if (ptr >= bufEnd) return 0;
                unsigned int bc = *ptr++;
                if (bc > 32) return 0;
                if (bc != 0) {
                    unsigned int n = (unsigned int)cur_bs;
                    size_t len = (size_t)(n + 7) / 8 + szp_fixed_bits_byte_length(n, bc);
                    if ((size_t)(bufEnd - ptr) < len) return 0;
                    ptr += len;
                }
            }
        }
        if (tid + 1 < try_nt) {
            if (ptr != rcp + szp_read_offset(cmpBytes, tid + 1)) return 0;
        } else if (exact && ptr != bufEnd) {
            return 0;
        }
    }
    return 1;
}

/* kind: 0 = random access, 1 = random access + 2-bit types, 2 = thread block */
static inline unsigned int
szp_detect_nbThreads_generic(const unsigned char *cmpBytes, size_t cmpSize,
                             size_t nbEle, int blockSize, int kind)
{
    if (!cmpBytes || nbEle == 0 || blockSize <= 0) return 1;
    {
        unsigned int tagged = szp_stream_tagged_threads(cmpBytes);
        if (tagged >= 1) return tagged;
    }
    int exact = (cmpSize != 0);
    size_t limit = cmpSize;
    if (!exact) {
        /* Loose upper bound on the stream size: worst case ~4.2 bytes per
           element for the data, 0.25 for 2-bit types, plus the offset table. */
        limit = 5 * nbEle + (kind == 1 ? nbEle / 4 + 1 : 0) + 1024
              + (size_t)SZP_MAX_DETECT_THREADS * sizeof(size_t);
    }
    {
        size_t slot0;
        memcpy(&slot0, cmpBytes, sizeof(size_t));
        if (limit < sizeof(size_t) || slot0 != 0) return 1;
    }

    unsigned int best = 0;
    for (unsigned int try_nt = 1; try_nt <= SZP_MAX_DETECT_THREADS; try_nt++) {
        size_t hdr = (size_t)try_nt * sizeof(size_t);
        if (hdr > limit) break;
        /* Offsets must be non-decreasing and point inside the stream. */
        if (try_nt >= 2) {
            size_t o = szp_read_offset(cmpBytes, try_nt - 1);
            if (o < szp_read_offset(cmpBytes, try_nt - 2) || o > limit - hdr) break;
        }
        int ok = (kind == 2)
            ? szp_validate_nbThreads_tb_generic(cmpBytes, limit, exact, nbEle, blockSize, try_nt)
            : szp_validate_nbThreads_ra_generic(cmpBytes, limit, exact, nbEle, blockSize, try_nt, kind == 1);
        if (ok) {
            best = try_nt;
            if (exact) break;
        }
    }
    return best ? best : 1;
}

/* ---- Entry points (unknown stream size) ---- */

static inline unsigned int
szp_detect_nbThreads_randomaccess(const unsigned char *cmpBytes, size_t nbEle, int blockSize)
{
    return szp_detect_nbThreads_generic(cmpBytes, 0, nbEle, blockSize, 0);
}

static inline unsigned int
szp_detect_nbThreads_randomaccess_topo(const unsigned char *cmpBytes, size_t nbEle, int blockSize)
{
    return szp_detect_nbThreads_generic(cmpBytes, 0, nbEle, blockSize, 1);
}

static inline unsigned int
szp_detect_nbThreads_threadblock(const unsigned char *cmpBytes, size_t nbEle, int blockSize)
{
    return szp_detect_nbThreads_generic(cmpBytes, 0, nbEle, blockSize, 2);
}

/* ---- Entry points (stream size known, preferred) ---- */

static inline unsigned int
szp_detect_nbThreads_randomaccess_sized(const unsigned char *cmpBytes, size_t cmpSize,
                                        size_t nbEle, int blockSize)
{
    return szp_detect_nbThreads_generic(cmpBytes, cmpSize, nbEle, blockSize, 0);
}

static inline unsigned int
szp_detect_nbThreads_randomaccess_topo_sized(const unsigned char *cmpBytes, size_t cmpSize,
                                             size_t nbEle, int blockSize)
{
    return szp_detect_nbThreads_generic(cmpBytes, cmpSize, nbEle, blockSize, 1);
}

static inline unsigned int
szp_detect_nbThreads_threadblock_sized(const unsigned char *cmpBytes, size_t cmpSize,
                                       size_t nbEle, int blockSize)
{
    return szp_detect_nbThreads_generic(cmpBytes, cmpSize, nbEle, blockSize, 2);
}

#ifdef __cplusplus
}
#endif

#endif /* _SZP_DETECT_THREADS_H */
