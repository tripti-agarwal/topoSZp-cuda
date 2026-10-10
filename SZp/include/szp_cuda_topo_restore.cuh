/**
 *  @file szp_cuda_topo_restore.cuh
 *  @brief GPU topology restoration (same algorithm and results as the CPU
 *         version in szp_topo_restore.h).
 *
 *  Arrays are host arrays; data is updated in place. The extrema step runs
 *  on the host (it is a sort over the extrema only); the saddle stencil and
 *  RBF passes run on the GPU, one kernel launch per colour class.
 *  Returns 0 on success, -1 on a CUDA error (data is then left as after
 *  the extrema step).
 */
#ifndef _SZP_CUDA_TOPO_RESTORE_CUH
#define _SZP_CUDA_TOPO_RESTORE_CUH

#include <stddef.h>
#include "szp_topo_restore.h"

#ifdef __cplusplus
extern "C" {
#endif

int szp_cuda_topo_restore_float_2d(float *data, const int *types, const int *sort_positions,
                                   size_t extrema_count, int rows, int cols, float errBound,
                                   szp_topo_restore_stats *stats);

int szp_cuda_topo_restore_float_3d(float *data, const int *types, const int *sort_positions,
                                   size_t extrema_count, int d1, int d2, int d3, float errBound,
                                   szp_topo_restore_stats *stats);

#ifdef __cplusplus
}
#endif

#endif /* _SZP_CUDA_TOPO_RESTORE_CUH */
