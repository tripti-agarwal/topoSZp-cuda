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

/**
 * Detection, ranking, encoding and the rank stream in one call; the field is
 * uploaded once and stays on the device. topoOut / sortOut (malloc'd, free
 * them) are byte-identical to the streams of the separate calls above.
 * sortOut is NULL when the field has no extrema. Returns 0 on success.
 */
int szp_cuda_float_compress_topology_ranked_3d(
    float *oriData, size_t nbEle, float absErrBound, int blockSize,
    int d1, int d2, int d3,
    unsigned char **topoOut, size_t *topoSize,
    unsigned char **sortOut, size_t *sortSize,
    size_t *criticalCount, size_t *extremaCount);

unsigned char *szp_cuda_double_compress_topology_3d(
    double *oriData, size_t *outSize, double absErrBound,
    size_t nbEle, int blockSize,
    CriticalPoint3D *critical_points, int critical_count,
    int d1, int d2, int d3);

#ifdef __cplusplus
}
#endif

#endif /* _SZP_CUDA_TOPOLOGY_3D_CUH */
