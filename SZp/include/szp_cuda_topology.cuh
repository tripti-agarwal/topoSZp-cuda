/**
 *  @file szp_cuda_topology.cuh
 *  @brief CUDA kernel declarations for topology-preserving compression functions.
 *
 *  Provides GPU-accelerated critical point finding, sorting, and topology-preserved
 *  compression/decompression. Converted from the OpenMP implementation in szp_float.cc.
 */

#ifndef _SZP_CUDA_TOPOLOGY_CUH
#define _SZP_CUDA_TOPOLOGY_CUH

#include <cstddef>
#include "szp_float.h"  // CriticalPoint struct

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Find critical points (local max, min, saddle) in a 2D grid using CUDA.
 * One CUDA thread per interior grid cell.
 */
CriticalPoint *szp_cuda_find_critical_points(float *data, size_t *outCount,
                                              int rows, int cols, float absErrBound);

/**
 * Sort critical points by original data values within each quantized bin.
 * Uses Thrust for histogram, prefix-sum, and per-bin sorting.
 */
void szp_cuda_sort_critical_points_by_original_data(CriticalPoint *critical_points,
                                                     size_t critical_count,
                                                     float *data, int cols);

/**
 * Compress sort_position values from critical points (extrema only).
 * Two-pass: sizing kernel → prefix-sum → packing kernel.
 */
unsigned char *szp_cuda_compress_sort_positions(CriticalPoint *critical_points,
                                                 size_t critical_count,
                                                 size_t *outSize, int blockSize);

/**
 * Random-access compression with topology (critical point type) metadata.
 * Each block also encodes 2-bit type information per element.
 */
unsigned char *szp_cuda_float_compress_randomaccess_topology_preserved(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize,
    CriticalPoint *critical_points, int critical_count,
    int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif /* _SZP_CUDA_TOPOLOGY_CUH */
