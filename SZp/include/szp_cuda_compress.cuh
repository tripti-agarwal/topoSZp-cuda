/**
 *  @file szp_cuda_compress.cuh
 *  @brief CUDA compression kernels for topoSZp — drop-in replacements for the OpenMP
 *         compression functions in szp_float.cc / szp_double.cc.
 *
 *  Output is byte-compatible with the OpenMP format (nChunks = 1).
 *  CUDA-compressed data can be decompressed by either the CUDA decompressor
 *  or the OpenMP decompressor with OMP_NUM_THREADS=1.
 */

#ifndef SZP_CUDA_COMPRESS_CUH
#define SZP_CUDA_COMPRESS_CUH

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/* ────────────────────────── Random-access compression ────────────────────── */

unsigned char *szp_cuda_float_compress_randomaccess(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize);

void szp_cuda_float_compress_randomaccess_arg(
    unsigned char *output, float *oriData, size_t *outSize,
    float absErrBound, size_t nbEle, int blockSize);

unsigned char *szp_cuda_double_compress_randomaccess(
    double *oriData, size_t *outSize, double absErrBound,
    size_t nbEle, int blockSize);

void szp_cuda_double_compress_randomaccess_arg(
    unsigned char *output, double *oriData, size_t *outSize,
    double absErrBound, size_t nbEle, int blockSize);

/* ─────────────────── Threadblock (non-random-access) compression ─────────── */

unsigned char *szp_cuda_float_compress_threadblock(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize);

void szp_cuda_float_compress_threadblock_arg(
    unsigned char *output, float *oriData, size_t *outSize,
    float absErrBound, size_t nbEle, int blockSize);

unsigned char *szp_cuda_double_compress_threadblock(
    double *oriData, size_t *outSize, double absErrBound,
    size_t nbEle, int blockSize);

void szp_cuda_double_compress_threadblock_arg(
    unsigned char *output, double *oriData, size_t *outSize,
    double absErrBound, size_t nbEle, int blockSize);

/* ──────────────────────── Quantization helpers ───────────────────────────── */

int *szp_cuda_float_direct_predict_quantization(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize);

int *szp_cuda_float_threadblock_predict_quantization(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize);

#ifdef __cplusplus
}
#endif

#endif /* SZP_CUDA_COMPRESS_CUH */
