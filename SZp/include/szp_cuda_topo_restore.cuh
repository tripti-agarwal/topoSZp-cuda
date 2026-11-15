/**
 *  @file szp_cuda_topo_restore.cuh
 *  @brief GPU topology restoration (same algorithm and results as the CPU
 *         version in szp_topo_restore.h).
 *
 *  Arrays are host arrays; data is updated in place. The field is copied to
 *  the GPU once and back once; every step runs on the device: counts, the
 *  extrema step (ranks in raster order by a block-wise compaction, groups of
 *  equal decompressed value by a hash table), the extremum locks, the saddle
 *  candidate lists, and the stencil / RBF passes (one kernel launch per
 *  colour class), and the neighbourhood-move passes.
 *  PL mode (szp_cuda_topo_restore_float_pl) still runs its extrema step and
 *  candidate lists on the host.
 *  Returns 0 on success, -1 on a CUDA error (data is then unchanged).
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

/* The same restoration for a field that is already on the device: d_data
   (restored in place), d_types and d_sort_positions (may be NULL) are device
   arrays; the restored field is also copied to host_out. d3 = 0 for a 2D
   field d1 x d2. */
int szp_cuda_topo_restore_float_device(float *host_out, float *d_data, int *d_types,
                                       int *d_sort_positions, size_t extrema_count,
                                       int d1, int d2, int d3, float errBound,
                                       szp_topo_restore_stats *stats);

/* PL mode (szp_pl_topology.h): same algorithm and results as
   szp_topo_restore_float_pl / szp_pl_classify_float. */
int szp_cuda_topo_restore_float_pl(float *data, const int *codes, const int *sort_positions,
                                   size_t extrema_count, int nd, int d1, int d2, int d3,
                                   float errBound, szp_topo_restore_stats *stats);
int szp_cuda_pl_classify_float(const float *data, int nd, int d1, int d2, int d3, signed char *codes);

#ifdef __cplusplus
}
#endif

#endif /* _SZP_CUDA_TOPO_RESTORE_CUH */
