/**
 *  @file szp_cuda_decompress.cuh
 *  @brief CUDA decompression kernels for TopoSZp.
 *
 *  Provides GPU-accelerated decompression that is byte-compatible with
 *  the OpenMP-compressed format (same compressed data can be decompressed
 *  on either CPU or GPU).
 */

#ifndef _SZP_CUDA_DECOMPRESS_CUH
#define _SZP_CUDA_DECOMPRESS_CUH

#include <stddef.h>

#include "szp_topo_restore.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Random-access decompression (each compression block independent)  */
/* ------------------------------------------------------------------ */

float *szp_cuda_float_decompress_randomaccess(
    size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes);

void szp_cuda_float_decompress_randomaccess_arg(
    float *newData, size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes);

double *szp_cuda_double_decompress_randomaccess(
    size_t nbEle, double absErrBound, int blockSize,
    unsigned char *cmpBytes);

void szp_cuda_double_decompress_randomaccess_arg(
    double *newData, size_t nbEle, double absErrBound, int blockSize,
    unsigned char *cmpBytes);

/* ------------------------------------------------------------------ */
/*  Threadblock decompression (serial dependency within chunks)       */
/* ------------------------------------------------------------------ */

float *szp_cuda_float_decompress_threadblock(
    size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes);

void szp_cuda_float_decompress_threadblock_arg(
    float *newData, size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes);

double *szp_cuda_double_decompress_threadblock(
    size_t nbEle, double absErrBound, int blockSize,
    unsigned char *cmpBytes);

void szp_cuda_double_decompress_threadblock_arg(
    double *newData, size_t nbEle, double absErrBound, int blockSize,
    unsigned char *cmpBytes);

/* ------------------------------------------------------------------ */
/*  Topology-preserved decompression (float only)                     */
/* ------------------------------------------------------------------ */

void szp_cuda_float_decompress_randomaccess_topology_preserved(
    float **newData, size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes, int **FN);

/* ------------------------------------------------------------------ */
/*  Sort-position decompression                                       */
/* ------------------------------------------------------------------ */

int *szp_cuda_decompress_sort_positions(
    unsigned char *cmpBytes, size_t critical_count, int blockSize);

/* ------------------------------------------------------------------ */
/*  Device-resident variants                                          */
/* ------------------------------------------------------------------ */

/* Decoded field and labels as device arrays (cudaFree them); 0 on success. */
int szp_cuda_float_decompress_topology_device(
    size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes, float **d_dataOut, int **d_typesOut);

/* Decoded ranks as a device array (cudaFree it); NULL on failure. */
int *szp_cuda_decompress_sort_positions_device(
    unsigned char *cmpBytes, size_t critical_count, int blockSize);

/* Decoding, rank decoding and topology restoration without leaving the
   device; only the final field is copied to the host (*newData, malloc'd).
   Same result as the separate calls. sortBytes may be NULL; d3 = 0 for a 2D
   field d1 x d2. Returns 0 on success. */
int szp_cuda_float_decompress_topology_restored(
    float **newData, size_t nbEle, float absErrBound, int blockSize,
    unsigned char *topoBytes, unsigned char *sortBytes, size_t extrema_count,
    int d1, int d2, int d3, szp_topo_restore_stats *stats);

#ifdef __cplusplus
}
#endif

#endif /* _SZP_CUDA_DECOMPRESS_CUH */
