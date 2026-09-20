/**
 *  @file szp_topology_3d.h
 *  @brief 3D critical point detection and topology-preserving compression.
 *
 *  Extends TopoSZp from 2D (4-connected) to 3D (6-connected) grids.
 *  Critical points: local maxima, minima, and saddle points in 3D scalar fields.
 */

#ifndef _SZP_TOPOLOGY_3D_H
#define _SZP_TOPOLOGY_3D_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 3D critical point with grid coordinates, type, quantized bin, and sort position.
 * Type encoding: 0=regular, 1=max, 2=min, 3=saddle (fits in 2 bits, same as 2D)
 */
typedef struct {
    int x, y, z;
    int type;            /* 1=max, 2=min, 3=saddle, 0=regular */
    int quantized_bin;
    int sort_position;
} CriticalPoint3D;

/* ---- OpenMP functions ---- */

/**
 * Find critical points in a 3D grid using 6-connected neighborhood.
 * One thread per interior voxel (x in [1,d1-2], y in [1,d2-2], z in [1,d3-2]).
 */
CriticalPoint3D *szp_find_critical_points_3d(float *data, size_t *outCount,
                                              int d1, int d2, int d3,
                                              float absErrBound);

/**
 * Sort critical points by original data value within each quantized bin.
 */
void szp_sort_critical_points_3d(CriticalPoint3D *critical_points,
                                  size_t critical_count,
                                  float *data, int d2, int d3);

/**
 * Compress sort_position values from 3D critical points (extrema only).
 */
unsigned char *szp_compress_sort_positions_3d(CriticalPoint3D *critical_points,
                                               size_t critical_count,
                                               size_t *outSize, int blockSize);

/**
 * Topology-preserved compression for 3D data.
 * Same compressed format as 2D but critical_type array uses 3D flat indices.
 */
unsigned char *szp_float_compress_topology_3d(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize,
    CriticalPoint3D *critical_points, int critical_count,
    int d1, int d2, int d3);

#ifdef __cplusplus
}
#endif

#endif /* _SZP_TOPOLOGY_3D_H */
