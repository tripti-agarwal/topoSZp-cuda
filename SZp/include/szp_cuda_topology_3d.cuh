/**
 *  @file szp_cuda_topology_3d.cuh
 *  @brief CUDA kernel declarations for 3D topology-preserving compression.
 */

#ifndef _SZP_CUDA_TOPOLOGY_3D_CUH
#define _SZP_CUDA_TOPOLOGY_3D_CUH

#include <stddef.h>
#include "szp_topology_3d.h"

#ifdef __cplusplus
extern "C" {
#endif

CriticalPoint3D *szp_cuda_find_critical_points_3d(float *data, size_t *outCount,
                                                    int d1, int d2, int d3,
                                                    float absErrBound);

void szp_cuda_sort_critical_points_3d(CriticalPoint3D *critical_points,
                                       size_t critical_count,
                                       float *data, int d2, int d3);

unsigned char *szp_cuda_compress_sort_positions_3d(CriticalPoint3D *critical_points,
                                                     size_t critical_count,
                                                     size_t *outSize, int blockSize);

unsigned char *szp_cuda_float_compress_topology_3d(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize,
    CriticalPoint3D *critical_points, int critical_count,
    int d1, int d2, int d3);

#ifdef __cplusplus
}
#endif

#endif /* _SZP_CUDA_TOPOLOGY_3D_CUH */
