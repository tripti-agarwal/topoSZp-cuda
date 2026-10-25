/**
 *  @file szp_cuda_topology_3d.cu
 *  @brief CUDA implementation of 3D topology-preserving compression functions.
 *
 *  Extension of the 2D topology code (szp_cuda_topology.cu) to 3D grids
 *  using 6-connected (face-adjacent) neighborhood: ±x, ±y, ±z.
 *
 *  Critical point classification (6-connected):
 *    max:    center > all 6 face-neighbors
 *    min:    center < all 6 face-neighbors
 *    saddle: high along some axes and low along others
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>

#include "szp_cuda_topology_3d.cuh"
#include "szp_cuda_common.cuh"  // CUDA_CHECK, CUDA_CHECK_KERNEL
#include "szp_defines.h"

/* ================================================================== */
/*  1.  3D CRITICAL POINT FINDING (shared-memory tiled)                */
/* ================================================================== */

/* Tile dimensions for shared-memory tiling.
 * Each thread block processes a TILE_X × TILE_Y × TILE_Z tile of
 * interior voxels and loads a (TILE_X+2) × (TILE_Y+2) × (TILE_Z+2)
 * halo region into shared memory.  Neighbor reads come from shared
 * memory instead of far-stride global reads (especially ±d2*d3).
 *
 * Shared memory per block = (TX+2)*(TY+2)*(TZ+2) * sizeof(float)
 *   8×8×4  → 10×10×6 × 4 = 2400 B,  256 threads  — high occupancy
 *   8×8×8  → 10×10×10× 4 = 4000 B,  512 threads
 * Both well under the 48 KB limit.
 */
#define CP3D_TILE_X 8
#define CP3D_TILE_Y 8
#define CP3D_TILE_Z 4
#define CP3D_HALO_X (CP3D_TILE_X + 2)
#define CP3D_HALO_Y (CP3D_TILE_Y + 2)
#define CP3D_HALO_Z (CP3D_TILE_Z + 2)

/**
 * Kernel: shared-memory tiled 3D critical point finder.
 *
 * Launched with a 3D grid where each block covers one tile of interior
 * voxels.  Each block loads its tile + 1-voxel halo into shared memory,
 * then every thread classifies its voxel via fast shared-memory reads
 * for the 6-connected neighbors.
 *
 * This eliminates the far-stride global reads (±d2*d3) that cause
 * severe cache thrashing in a flat-index kernel.
 */
__global__ void find_critical_points_3d_kernel(
    const float *data, int d1, int d2, int d3,
    double inver_bound, float absErrBound,
    CriticalPoint3D *results, unsigned int *d_count,
    size_t /* total_interior — unused, kept for API compat */)
{
    /* Shared memory tile including 1-voxel halo on each side */
    __shared__ float tile[CP3D_HALO_X][CP3D_HALO_Y][CP3D_HALO_Z];

    /* Global coordinates of this tile's interior origin.
     * Interior voxels span x in [1, d1-2], y in [1, d2-2], z in [1, d3-2]. */
    int tile_x0 = 1 + (int)blockIdx.x * CP3D_TILE_X;
    int tile_y0 = 1 + (int)blockIdx.y * CP3D_TILE_Y;
    int tile_z0 = 1 + (int)blockIdx.z * CP3D_TILE_Z;

    /* Local thread indices within the tile */
    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;
    int tz = (int)threadIdx.z;

    /* --- Load the halo region into shared memory ---
     * Halo spans [tile_x0-1 .. tile_x0+TILE_X] in each dim.
     * We have TILE_X*TILE_Y*TILE_Z threads; each may load multiple elements. */
    int halo_total = CP3D_HALO_X * CP3D_HALO_Y * CP3D_HALO_Z;
    int tid_flat = tx * (CP3D_TILE_Y * CP3D_TILE_Z)
                 + ty * CP3D_TILE_Z + tz;
    int threads_per_block = CP3D_TILE_X * CP3D_TILE_Y * CP3D_TILE_Z;

    for (int h = tid_flat; h < halo_total; h += threads_per_block) {
        int hx = h / (CP3D_HALO_Y * CP3D_HALO_Z);
        int hr = h % (CP3D_HALO_Y * CP3D_HALO_Z);
        int hy = hr / CP3D_HALO_Z;
        int hz = hr % CP3D_HALO_Z;

        /* Global coordinate for this halo element */
        int gx = tile_x0 - 1 + hx;
        int gy = tile_y0 - 1 + hy;
        int gz = tile_z0 - 1 + hz;

        /* Clamp to valid data range [0, d-1] for boundary tiles */
        if (gx >= 0 && gx < d1 && gy >= 0 && gy < d2 && gz >= 0 && gz < d3) {
            tile[hx][hy][hz] = data[(size_t)gx * d2 * d3
                                  + (size_t)gy * d3 + gz];
        } else {
            tile[hx][hy][hz] = 0.0f;
        }
    }

    __syncthreads();

    /* Global coordinate of this thread's interior voxel */
    int x = tile_x0 + tx;
    int y = tile_y0 + ty;
    int z = tile_z0 + tz;

    /* Check this thread maps to a valid interior voxel */
    if (x > d1 - 2 || y > d2 - 2 || z > d3 - 2) return;

    /* Shared-memory indices: interior voxel sits at (tx+1, ty+1, tz+1). */
    int sx = tx + 1;
    int sy = ty + 1;
    int sz = tz + 1;

    float center = tile[sx][sy][sz];
    float xm = tile[sx - 1][sy][sz];      /* x-1 */
    float xp = tile[sx + 1][sy][sz];      /* x+1 */
    float ym = tile[sx][sy - 1][sz];      /* y-1 */
    float yp = tile[sx][sy + 1][sz];      /* y+1 */
    float zm = tile[sx][sy][sz - 1];      /* z-1 */
    float zp = tile[sx][sy][sz + 1];      /* z+1 */

    int type = 0;
    if (center > xm && center > xp &&
        center > ym && center > yp &&
        center > zm && center > zp) {
        type = 1;   /* local maximum */
    } else if (center < xm && center < xp &&
               center < ym && center < yp &&
               center < zm && center < zp) {
        type = 2;   /* local minimum */
    } else {
        /* Saddle detection: check axis-aligned directional behaviour.
           A saddle exists when the center is a local extremum along
           some axes and the opposite along other axes. */
        int x_high = (center > xm) && (center > xp);
        int x_low  = (center < xm) && (center < xp);
        int y_high = (center > ym) && (center > yp);
        int y_low  = (center < ym) && (center < yp);
        int z_high = (center > zm) && (center > zp);
        int z_low  = (center < zm) && (center < zp);

        int high_axes = x_high + y_high + z_high;
        int low_axes  = x_low  + y_low  + z_low;

        if (high_axes >= 1 && low_axes >= 1) {
            type = 3;   /* saddle */
        }
    }

    if (type != 0) {
        int quantized_bin = (int)((center + absErrBound) * inver_bound);
        unsigned int pos = atomicAdd(d_count, 1u);
        results[pos].x = x;
        results[pos].y = y;
        results[pos].z = z;
        results[pos].type = type;
        results[pos].quantized_bin = quantized_bin;
        results[pos].sort_position = 0;
    }
}

extern "C"
CriticalPoint3D *szp_cuda_find_critical_points_3d(
    float *data, size_t *outCount,
    int d1, int d2, int d3, float absErrBound)
{
    if (!data || d1 <= 2 || d2 <= 2 || d3 <= 2) {
        *outCount = 0;
        return NULL;
    }

    szp_cuda_validate_dims(d1, d2, d3, "szp_cuda_find_critical_points_3d");

    size_t grid_size = (size_t)d1 * d2 * d3;
    size_t total_interior = (size_t)(d1 - 2)
                          * (size_t)(d2 - 2)
                          * (size_t)(d3 - 2);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, grid_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, data, grid_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    CriticalPoint3D *d_results;
    CUDA_CHECK(cudaMalloc(&d_results, total_interior * sizeof(CriticalPoint3D)));

    unsigned int *d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned int)));

    double inver_bound = 1.0 / absErrBound;

    /* 3D grid of thread blocks, each covering one tile of interior voxels.
     * Block dim = (TILE_X, TILE_Y, TILE_Z) = (8, 8, 4) = 256 threads.
     * Grid dim covers all interior tiles, rounding up. */
    int interior_x = d1 - 2;
    int interior_y = d2 - 2;
    int interior_z = d3 - 2;
    dim3 blockDim3(CP3D_TILE_X, CP3D_TILE_Y, CP3D_TILE_Z);
    dim3 gridDim3((interior_x + CP3D_TILE_X - 1) / CP3D_TILE_X,
                  (interior_y + CP3D_TILE_Y - 1) / CP3D_TILE_Y,
                  (interior_z + CP3D_TILE_Z - 1) / CP3D_TILE_Z);

    find_critical_points_3d_kernel<<<gridDim3, blockDim3>>>(
        d_data, d1, d2, d3, inver_bound, absErrBound,
        d_results, d_count, total_interior);
    CUDA_CHECK(cudaGetLastError());

    unsigned int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    CriticalPoint3D *results = NULL;
    if (h_count > 0) {
        results = (CriticalPoint3D *)malloc(h_count * sizeof(CriticalPoint3D));
        CUDA_CHECK(cudaMemcpy(results, d_results,
                              h_count * sizeof(CriticalPoint3D),
                              cudaMemcpyDeviceToHost));
    }

    *outCount = (size_t)h_count;

    cudaFree(d_data);
    cudaFree(d_results);
    cudaFree(d_count);

    return results;
}

/* ================================================================== */
/*  2.  SORT 3D CRITICAL POINTS BY ORIGINAL DATA WITHIN BINS           */
/* ================================================================== */

struct BinValueKey3D {
    int    bin;
    float  value;
    size_t original_index;
};

struct BinValueCmp3D {
    __host__ __device__
    bool operator()(const BinValueKey3D &a, const BinValueKey3D &b) const {
        if (a.bin != b.bin) return a.bin < b.bin;
        return a.value < b.value;
    }
};

__global__ void build_sort_keys_3d_kernel(
    const CriticalPoint3D *cp, size_t n,
    const float *data, int d2, int d3,
    BinValueKey3D *keys)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    size_t flat = (size_t)cp[idx].x * d2 * d3
                + (size_t)cp[idx].y * d3
                + cp[idx].z;
    keys[idx].bin            = cp[idx].quantized_bin;
    keys[idx].value          = data[flat];
    keys[idx].original_index = idx;
}

/**
 * Step 1: Mark boundaries — write 1 where (bin, value) changes from predecessor.
 * Element 0 always gets 0 (start of first bin). Elements within the same bin
 * with the same value get 0 (no boundary). Elements with a different bin OR
 * different value from predecessor get 1.
 */
__global__ void mark_boundaries_3d_kernel(
    const BinValueKey3D *sorted_keys, int *boundaries, size_t n)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (idx == 0) {
        boundaries[idx] = 0;
    } else {
        int same_bin   = (sorted_keys[idx].bin == sorted_keys[idx - 1].bin);
        int same_value = (sorted_keys[idx].value == sorted_keys[idx - 1].value);
        if (!same_bin) {
            boundaries[idx] = 0;  /* start of new bin → position 0 */
        } else if (!same_value) {
            boundaries[idx] = 1;  /* new value within same bin → increment */
        } else {
            boundaries[idx] = 0;  /* same bin, same value → same position */
        }
    }
}

/**
 * Step 3: Segmented prefix sum + parallel scatter.
 *
 * boundaries[] is 0/1: 1 where (bin,value) differs from predecessor, else 0.
 * An exclusive_scan_by_key over boundaries[], keyed by bin, produces the
 * within-bin rank (number of distinct values before this element in its bin).
 * A parallel kernel then scatters the rank to cp[original_index].sort_position.
 */
__global__ void scatter_sort_positions_3d_kernel(
    CriticalPoint3D *cp, const BinValueKey3D *sorted_keys,
    const int *positions, size_t n)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    cp[sorted_keys[idx].original_index].sort_position = positions[idx];
}

extern "C"
void szp_cuda_sort_critical_points_3d(
    CriticalPoint3D *critical_points, size_t critical_count,
    float *data, int d2, int d3)
{
    if (!critical_points || critical_count == 0 || !data) return;

    /* Find max flat index on host to determine data copy size.
       Use OpenMP for large counts. */
    size_t max_flat = 0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(max:max_flat)
    #endif
    for (size_t i = 0; i < critical_count; i++) {
        size_t flat = (size_t)critical_points[i].x * d2 * d3
                    + (size_t)critical_points[i].y * d3
                    + critical_points[i].z;
        if (flat > max_flat) max_flat = flat;
    }
    size_t data_bytes = (max_flat + 1) * sizeof(float);

    /* Separate allocations for proper alignment */
    CriticalPoint3D *d_cp = NULL;
    float *d_data = NULL;
    BinValueKey3D *d_keys = NULL;
    int *d_boundaries = NULL;

    CUDA_CHECK(cudaMalloc(&d_cp, critical_count * sizeof(CriticalPoint3D)));
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_keys, critical_count * sizeof(BinValueKey3D)));
    CUDA_CHECK(cudaMalloc(&d_boundaries, critical_count * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_cp, critical_points, critical_count * sizeof(CriticalPoint3D), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data, data, data_bytes, cudaMemcpyHostToDevice));

    int tpb = 256;
    int grid = ((int)critical_count + tpb - 1) / tpb;

    /* Build sort keys */
    build_sort_keys_3d_kernel<<<grid, tpb>>>(
        d_cp, critical_count, d_data, d2, d3, d_keys);
    CUDA_CHECK(cudaGetLastError());
    thrust::device_ptr<BinValueKey3D> keys_ptr(d_keys);
    thrust::sort(keys_ptr, keys_ptr + critical_count, BinValueCmp3D());

    /* Mark boundaries (O(n) parallel): boundaries[i] = 1 where value changes */
    mark_boundaries_3d_kernel<<<grid, tpb>>>(d_keys, d_boundaries, critical_count);
    CUDA_CHECK(cudaGetLastError());

    /* Segmented exclusive prefix sum keyed by bin:
       produces within-bin rank for each element.  O(n) parallel via Thrust. */
    {
        /* Build bin-key array for the scan: key[i] = sorted_keys[i].bin */
        int *d_bin_keys = NULL;
        int *d_positions = NULL;
        CUDA_CHECK(cudaMalloc(&d_bin_keys, critical_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_positions, critical_count * sizeof(int)));

        /* Extract bins in parallel */
        thrust::device_ptr<BinValueKey3D> kp(d_keys);
        thrust::device_ptr<int> bk(d_bin_keys);
        thrust::transform(kp, kp + critical_count, bk,
            [] __device__ (const BinValueKey3D &k) { return k.bin; });

        /* Segmented exclusive scan: resets to 0 at every new bin */
        thrust::device_ptr<int> bp(d_boundaries);
        thrust::device_ptr<int> pp(d_positions);
        thrust::exclusive_scan_by_key(bk, bk + critical_count, bp, pp);
        CUDA_CHECK(cudaGetLastError());

        /* Scatter positions back to cp[] in parallel */
        scatter_sort_positions_3d_kernel<<<grid, tpb>>>(
            d_cp, d_keys, d_positions, critical_count);
        CUDA_CHECK(cudaGetLastError());

        cudaFree(d_bin_keys);
        cudaFree(d_positions);
    }

    /* Copy results back */
    CUDA_CHECK(cudaMemcpy(critical_points, d_cp, critical_count * sizeof(CriticalPoint3D), cudaMemcpyDeviceToHost));

    cudaFree(d_cp);
    cudaFree(d_data);
    cudaFree(d_keys);
    cudaFree(d_boundaries);
}

/* ================================================================== */
/*  3.  COMPRESS SORT POSITIONS (3D)                                   */
/* ================================================================== */

/*
 * The sort-position compression kernels are identical to the 2D versions
 * because they operate on a flat int array of sort_position values.
 * Only the extraction (which CPs are extrema) differs, using CriticalPoint3D.
 * We re-declare the device kernels here (static linkage) to avoid ODR issues.
 */

__global__ static void compress_sort_positions_sizing_3d(
    const int *sort_positions, size_t nbEle, int blockSize,
    size_t *block_sizes)
{
    size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_blocks = (nbEle + blockSize - 1) / blockSize;
    if (bid >= num_blocks) return;

    size_t bs = bid * blockSize;
    size_t cur = (bs + blockSize > nbEle) ? (nbEle - bs) : (size_t)blockSize;
    size_t out = sizeof(int);

    if (cur > 1) {
        int prior = sort_positions[bs];
        unsigned int mx = 0;
        for (size_t j = 1; j < cur; j++) {
            int d = sort_positions[bs + j] - prior;
            prior = sort_positions[bs + j];
            unsigned int ad = (d < 0) ? (unsigned int)(-d) : (unsigned int)d;
            if (ad > mx) mx = ad;
        }
        out += 1;
        if (mx > 0) {
            unsigned int bc = (unsigned int)floorf(log2f((float)mx)) + 1u;
            unsigned int n = (unsigned int)(cur - 1);
            unsigned int sb = (n + 7) / 8;
            unsigned int bcc = bc / 8, rb = bc % 8;
            unsigned int mb = bcc * n;
            if (rb > 0) mb += (rb * n + 7) / 8;
            out += sb + mb;
        }
    }
    block_sizes[bid] = out;
}

__global__ static void compress_sort_positions_packing_3d(
    const int *sort_positions, size_t nbEle, int blockSize,
    const size_t *block_offsets, unsigned char *output)
{
    size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_blocks = (nbEle + blockSize - 1) / blockSize;
    if (bid >= num_blocks) return;

    size_t bs = bid * blockSize;
    size_t cur = (bs + blockSize > nbEle) ? (nbEle - bs) : (size_t)blockSize;
    unsigned char *ptr = output + block_offsets[bid];

    int fv = sort_positions[bs];
    memcpy(ptr, &fv, sizeof(int));
    ptr += sizeof(int);
    if (cur <= 1) return;

    unsigned int n = (unsigned int)(cur - 1);
    unsigned char lsigns[256];
    unsigned int  lmags[256];
    unsigned int mx = 0;
    int prior = fv;
    for (unsigned int j = 0; j < n; j++) {
        int d = sort_positions[bs + j + 1] - prior;
        prior = sort_positions[bs + j + 1];
        if (d < 0) { lsigns[j] = 1; lmags[j] = (unsigned int)(-d); }
        else       { lsigns[j] = 0; lmags[j] = (unsigned int)d; }
        if (lmags[j] > mx) mx = lmags[j];
    }
    if (mx == 0) { *ptr++ = 0; return; }

    unsigned int bc = (unsigned int)floorf(log2f((float)mx)) + 1u;
    *ptr++ = (unsigned char)bc;

    /* sign bits MSB-first */
    unsigned int sb = (n + 7) / 8;
    for (unsigned int b = 0; b < sb; b++) {
        unsigned char t = 0;
        for (unsigned int k = 0; k < 8 && b * 8 + k < n; k++)
            t |= (lsigns[b * 8 + k] << (7 - k));
        *ptr++ = t;
    }

    /* magnitude bits — Jiajun layout */
    unsigned int bcc = bc / 8, rb = bc % 8;
    if (bcc > 0) {
        for (unsigned int i = 0; i < n; i++) {
            unsigned int v = lmags[i] >> rb;
            for (unsigned int j = 0; j < bcc; j++) {
                ptr[i * bcc + j] = (unsigned char)(v & 0xFF);
                v >>= 8;
            }
        }
        ptr += bcc * n;
    }
    if (rb > 0) {
        unsigned int trb = (rb * n + 7) / 8;
        for (unsigned int b = 0; b < trb; b++) ptr[b] = 0;
        unsigned int mask = (1u << rb) - 1;
        for (unsigned int i = 0; i < n; i++) {
            unsigned int v = lmags[i] & mask;
            unsigned int bstart = i * rb;
            for (unsigned int b = 0; b < rb; b++) {
                unsigned int gb = bstart + b;
                unsigned int bi = gb / 8;
                unsigned int bp = 7 - (gb % 8);
                unsigned int sbit = rb - 1 - b;
                if ((v >> sbit) & 1) ptr[bi] |= (1u << bp);
            }
        }
        ptr += trb;
    }
}

extern "C"
unsigned char *szp_cuda_compress_sort_positions_3d(
    CriticalPoint3D *critical_points, size_t critical_count,
    size_t *outSize, int blockSize)
{
    if (!critical_points || critical_count == 0) {
        *outSize = 0;
        return NULL;
    }

    szp_cuda_validate_blocksize(blockSize, SZP_CUDA_TOPO_MAX_BLOCK,
                                "szp_cuda_compress_sort_positions_3d");

    /* Extract sort_position values for extrema (type 1 or 2) */
    size_t extrema_count = 0;
    for (size_t i = 0; i < critical_count; i++)
        if (critical_points[i].type == 1 || critical_points[i].type == 2)
            extrema_count++;
    if (extrema_count == 0) { *outSize = 0; return NULL; }

    int *sp = (int *)malloc(extrema_count * sizeof(int));
    size_t ei = 0;
    for (size_t i = 0; i < critical_count; i++)
        if (critical_points[i].type == 1 || critical_points[i].type == 2)
            sp[ei++] = critical_points[i].sort_position;

    int *d_sp;
    CUDA_CHECK(cudaMalloc(&d_sp, extrema_count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sp, sp, extrema_count * sizeof(int),
                          cudaMemcpyHostToDevice));

    size_t nb = (extrema_count + blockSize - 1) / blockSize;
    size_t *d_bs, *d_bo;
    CUDA_CHECK(cudaMalloc(&d_bs, nb * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_bo, nb * sizeof(size_t)));

    int tpb = 256;
    int grid = ((int)nb + tpb - 1) / tpb;
    compress_sort_positions_sizing_3d<<<grid, tpb>>>(d_sp, extrema_count,
                                                      blockSize, d_bs);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<size_t> sp_ptr(d_bs);
    thrust::device_ptr<size_t> op_ptr(d_bo);
    thrust::exclusive_scan(sp_ptr, sp_ptr + nb, op_ptr);

    size_t lsz, loff;
    CUDA_CHECK(cudaMemcpy(&lsz, d_bs + nb - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&loff, d_bo + nb - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    size_t total = loff + lsz;

    size_t hdr = sizeof(size_t);
    *outSize = hdr + total;
    unsigned char *out = (unsigned char *)malloc(*outSize);
    size_t zero = 0;
    memcpy(out, &zero, sizeof(size_t));

    unsigned char *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, total));
    compress_sort_positions_packing_3d<<<grid, tpb>>>(d_sp, extrema_count,
                                                       blockSize, d_bo, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out + hdr, d_out, total, cudaMemcpyDeviceToHost));

    cudaFree(d_sp); cudaFree(d_bs); cudaFree(d_bo); cudaFree(d_out);
    free(sp);
    return out;
}

/* ================================================================== */
/*  4.  TOPOLOGY-PRESERVED 3D COMPRESSION                             */
/* ================================================================== */

/* Mark 3D critical point types into a flat array */
__global__ void mark_critical_types_3d_kernel(
    const CriticalPoint3D *cp, int critical_count,
    int d1, int d2, int d3,
    unsigned char *critical_type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= critical_count) return;

    int x = cp[idx].x, y = cp[idx].y, z = cp[idx].z;
    if (x < 0 || x >= d1 || y < 0 || y >= d2 || z < 0 || z >= d3) return;

    size_t flat = (size_t)x * d2 * d3 + (size_t)y * d3 + z;
    unsigned char t = (unsigned char)cp[idx].type;
    if (t >= 1 && t <= 3) {
        critical_type[flat] = t;
    }
}

/* ================================================================== */
/*  SADDLE-AWARE QUANTIZATION                                          */
/*                                                                     */
/*  After standard quantization, saddle points may lose their saddle   */
/*  condition because neighboring quantized values change relative     */
/*  ordering.  This kernel detects broken saddles and adjusts the      */
/*  center quantized value to restore the saddle condition.            */
/*  Priority: maxima/minima are never modified.                        */
/* ================================================================== */

/**
 * Quantize all data to integers: q[i] = (int)((double)data[i] * inver_bound)
 * Uses double precision arithmetic to avoid float rounding errors.
 */
__global__ static void quantize_3d_kernel_float(
    const float *data, int *qdata, size_t nbEle, double inver_bound)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nbEle) qdata[idx] = (int)((double)data[idx] * inver_bound);
}

__global__ static void quantize_3d_kernel_double(
    const double *data, int *qdata, size_t nbEle, double inver_bound)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nbEle) qdata[idx] = (int)(data[idx] * inver_bound);
}

/**
 * Pre-quantization perturbation: adjust FLOAT data at saddle neighbors
 * to ensure the saddle condition survives quantization.
 *
 * For each saddle, identify which axes are HIGH (center > both neighbors)
 * and which are LOW (center < both neighbors) in the ORIGINAL data.
 * Then perturb NON-CRITICAL neighbors by ±0.5*eb:
 *   - High-axis neighbors: decrease by 0.5*eb (so they quantize lower)
 *   - Low-axis neighbors:  increase by 0.5*eb (so they quantize higher)
 *
 * This creates a quantization-level gap that preserves the saddle pattern.
 * Only modifies neighbors that are REGULAR points (type 0).
 * Max added error per element: 0.5*eb (well within the 2*eb budget).
 */
__global__ static void perturb_saddle_neighbors_3d_kernel(
    float *data, const unsigned char *critical_type,
    int d1, int d2, int d3, float absErrBound,
    size_t total_interior)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_interior) return;

    size_t d2m2 = (size_t)(d2-2);
    size_t d3m2 = (size_t)(d3-2);
    size_t plane = d2m2 * d3m2;
    int x = 1 + (int)(idx / plane);
    size_t rr = idx % plane;
    int y = 1 + (int)(rr / d3m2);
    int z = 1 + (int)(rr % d3m2);

    size_t flat = (size_t)x * d2 * d3 + (size_t)y * d3 + z;
    if (critical_type[flat] != 3) return;  /* only process saddles */

    float center = data[flat];
    float half_eb = 0.5f * absErrBound;

    /* 6 neighbors: indices and values */
    size_t nb[6];
    nb[0] = (size_t)(x-1)*d2*d3 + y*d3 + z;  /* x- */
    nb[1] = (size_t)(x+1)*d2*d3 + y*d3 + z;  /* x+ */
    nb[2] = (size_t)x*d2*d3 + (y-1)*d3 + z;  /* y- */
    nb[3] = (size_t)x*d2*d3 + (y+1)*d3 + z;  /* y+ */
    nb[4] = (size_t)x*d2*d3 + y*d3 + (z-1);  /* z- */
    nb[5] = (size_t)x*d2*d3 + y*d3 + (z+1);  /* z+ */

    /* Determine which axes are high/low in original data */
    /* Axis 0 (x): neighbors nb[0], nb[1] */
    /* Axis 1 (y): neighbors nb[2], nb[3] */
    /* Axis 2 (z): neighbors nb[4], nb[5] */
    for (int axis = 0; axis < 3; axis++) {
        size_t n0 = nb[axis*2];
        size_t n1 = nb[axis*2+1];
        float v0 = data[n0];
        float v1 = data[n1];

        int is_high = (center > v0) && (center > v1);  /* center above both */
        int is_low  = (center < v0) && (center < v1);  /* center below both */

        if (is_high) {
            /* Decrease high-axis neighbors (if they're regular points) */
            if (critical_type[n0] == 0)
                atomicAdd(&data[n0], -half_eb);
            if (critical_type[n1] == 0)
                atomicAdd(&data[n1], -half_eb);
        } else if (is_low) {
            /* Increase low-axis neighbors (if they're regular points) */
            if (critical_type[n0] == 0)
                atomicAdd(&data[n0], half_eb);
            if (critical_type[n1] == 0)
                atomicAdd(&data[n1], half_eb);
        }
    }
}

/**
 * Fix saddle quantized values to preserve saddle condition.
 * For each saddle point, check if the 6-connected neighborhood
 * still has the saddle property in quantized space. If not,
 * adjust the center quantized value.
 *
 * Saddle condition: high along some axes AND low along others.
 * We find a valid quantized value that satisfies this.
 */
/**
 * Fix saddle quantized values using double-buffering to avoid races.
 * Reads neighbor values from qdata_src (previous pass snapshot),
 * writes corrections to qdata_dst.  The caller swaps/copies between passes.
 */
__global__ static void fix_saddles_3d_kernel(
    const int *qdata_src, int *qdata_dst,
    const unsigned char *critical_type,
    int d1, int d2, int d3, size_t nbEle,
    size_t total_interior)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_interior) return;

    size_t d2m2 = (size_t)(d2-2);
    size_t d3m2 = (size_t)(d3-2);
    size_t plane = d2m2 * d3m2;
    int x = 1 + (int)(idx / plane);
    size_t rr = idx % plane;
    int y = 1 + (int)(rr / d3m2);
    int z = 1 + (int)(rr % d3m2);

    size_t flat = (size_t)x * d2 * d3 + (size_t)y * d3 + z;
    if (critical_type[flat] != 3) return;  /* only fix saddles */

    /* Read all values from the source (snapshot) buffer */
    size_t sd2d3 = (size_t)d2 * d3;
    int c  = qdata_src[flat];
    int xm = qdata_src[flat - sd2d3];
    int xp = qdata_src[flat + sd2d3];
    int ym = qdata_src[flat - d3];
    int yp = qdata_src[flat + d3];
    int zm = qdata_src[flat - 1];
    int zp = qdata_src[flat + 1];

    /* Check if saddle condition already holds */
    int x_high = (c > xm) && (c > xp);
    int x_low  = (c < xm) && (c < xp);
    int y_high = (c > ym) && (c > yp);
    int y_low  = (c < ym) && (c < yp);
    int z_high = (c > zm) && (c > zp);
    int z_low  = (c < zm) && (c < zp);
    int high_axes = x_high + y_high + z_high;
    int low_axes  = x_low  + y_low  + z_low;

    if (high_axes >= 1 && low_axes >= 1) return;  /* saddle is preserved */

    /* Saddle is broken — try to find a valid quantized value.
     * Strategy: for each axis, compute the range where center would be
     * high or low. Find a value that makes at least one axis high and
     * one axis low. Try values within ±1 of current quantized value first. */

    /* Axis bounds: to be high along axis A, need c > max(Am, Ap)
     *              to be low  along axis A, need c < min(Am, Ap) */
    int x_max = (xm > xp) ? xm : xp;  /* need c > x_max to be x-high */
    int x_min = (xm < xp) ? xm : xp;  /* need c < x_min to be x-low  */
    int y_max = (ym > yp) ? ym : yp;
    int y_min = (ym < yp) ? ym : yp;
    int z_max = (zm > zp) ? zm : zp;
    int z_min = (zm < zp) ? zm : zp;

    /* Try candidates c-1, c, c+1 (within ±1 quantization step = within error bound) */
    for (int delta = 0; delta <= 1; delta++) {
        for (int sign = -1; sign <= 1; sign += 2) {
            int trial = c + sign * delta;
            if (delta == 0 && sign == 1) continue;  /* skip duplicate c+0 */

            int th = 0, tl = 0;
            if (trial > x_max) th++; if (trial < x_min) tl++;
            if (trial > y_max) th++; if (trial < y_min) tl++;
            if (trial > z_max) th++; if (trial < z_min) tl++;

            if (th >= 1 && tl >= 1) {
                /* Check: don't break neighboring maxima/minima */
                int ok = 1;
                /* If a neighbor is a maximum, it must stay > center */
                if (critical_type[flat - sd2d3] == 1 && xm <= trial) ok = 0;
                if (critical_type[flat + sd2d3] == 1 && xp <= trial) ok = 0;
                if (critical_type[flat - d3]    == 1 && ym <= trial) ok = 0;
                if (critical_type[flat + d3]    == 1 && yp <= trial) ok = 0;
                if (critical_type[flat - 1]     == 1 && zm <= trial) ok = 0;
                if (critical_type[flat + 1]     == 1 && zp <= trial) ok = 0;
                /* If a neighbor is a minimum, it must stay < center */
                if (critical_type[flat - sd2d3] == 2 && xm >= trial) ok = 0;
                if (critical_type[flat + sd2d3] == 2 && xp >= trial) ok = 0;
                if (critical_type[flat - d3]    == 2 && ym >= trial) ok = 0;
                if (critical_type[flat + d3]    == 2 && yp >= trial) ok = 0;
                if (critical_type[flat - 1]     == 2 && zm >= trial) ok = 0;
                if (critical_type[flat + 1]     == 2 && zp >= trial) ok = 0;

                if (ok) { qdata_dst[flat] = trial; return; }
            }
        }
    }
    /* ±1 didn't work — don't try ±2 to keep total error within 2×eb */
}

/**
 * Compression kernels that work on PRE-QUANTIZED data (int array)
 * instead of floating-point data.
 */
__global__ static void compress_topo_sizing_3d_preq(
    const int *qdata, size_t nbEle, unsigned int blockSize,
    size_t *block_sizes)
{
    size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_blocks = (nbEle + blockSize - 1) / blockSize;
    if (bid >= num_blocks) return;

    size_t bs = bid * blockSize;
    size_t cur = (bs + blockSize > nbEle) ? (nbEle - bs) : (size_t)blockSize;
    size_t out = sizeof(int);

    if (cur > 1) {
        int prior = qdata[bs];
        unsigned int mx = 0;
        for (size_t j = 1; j < cur; j++) {
            int c = qdata[bs + j];
            int d = c - prior; prior = c;
            unsigned int ad = (d < 0) ? (unsigned int)(-d) : (unsigned int)d;
            if (ad > mx) mx = ad;
        }
        out += 1;
        if (mx > 0) {
            unsigned int bc = (unsigned int)floorf(log2f((float)mx)) + 1u;
            unsigned int n = (unsigned int)(cur - 1);
            unsigned int sb = (n + 7) / 8;
            unsigned int bcc = bc / 8, rb = bc % 8;
            unsigned int mb = bcc * n;
            if (rb > 0) mb += (rb * n + 7) / 8;
            out += sb + mb;
        }
    }

    unsigned int tb = (2 * (unsigned int)cur + 7) / 8;
    out += tb;
    block_sizes[bid] = out;
}

__global__ static void compress_topo_packing_3d_preq(
    const int *qdata, size_t nbEle, unsigned int blockSize,
    const unsigned char *critical_type,
    const size_t *block_offsets, unsigned char *output)
{
    size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_blocks = (nbEle + blockSize - 1) / blockSize;
    if (bid >= num_blocks) return;

    size_t bs = bid * blockSize;
    size_t cur = (bs + blockSize > nbEle) ? (nbEle - bs) : (size_t)blockSize;
    unsigned char *ptr = output + block_offsets[bid];

    int prior = qdata[bs];
    memcpy(ptr, &prior, sizeof(int));
    ptr += sizeof(int);

    if (cur > 1) {
        unsigned int n = (unsigned int)(cur - 1);
        unsigned char lsigns[256];
        unsigned int  lmags[256];
        unsigned int mx = 0;

        for (unsigned int j = 0; j < n; j++) {
            int c = qdata[bs + j + 1];
            int d = c - prior; prior = c;
            if (d < 0) { lsigns[j] = 1; lmags[j] = (unsigned int)(-d); }
            else       { lsigns[j] = 0; lmags[j] = (unsigned int)d; }
            if (lmags[j] > mx) mx = lmags[j];
        }

        if (mx == 0) {
            *ptr++ = 0;
        } else {
            unsigned int bc = (unsigned int)floorf(log2f((float)mx)) + 1u;
            *ptr++ = (unsigned char)bc;

            unsigned int sb = (n + 7) / 8;
            for (unsigned int b = 0; b < sb; b++) {
                unsigned char t = 0;
                for (unsigned int k = 0; k < 8 && b * 8 + k < n; k++)
                    t |= (lsigns[b * 8 + k] << (7 - k));
                *ptr++ = t;
            }

            unsigned int bcc = bc / 8, rb = bc % 8;
            if (bcc > 0) {
                for (unsigned int i = 0; i < n; i++) {
                    unsigned int v = lmags[i] >> rb;
                    for (unsigned int j = 0; j < bcc; j++) {
                        ptr[i * bcc + j] = (unsigned char)(v & 0xFF);
                        v >>= 8;
                    }
                }
                ptr += bcc * n;
            }
            if (rb > 0) {
                unsigned int trb = (rb * n + 7) / 8;
                for (unsigned int b = 0; b < trb; b++) ptr[b] = 0;
                unsigned int mask = (1u << rb) - 1;
                for (unsigned int i = 0; i < n; i++) {
                    unsigned int v = lmags[i] & mask;
                    unsigned int bstart = i * rb;
                    for (unsigned int b = 0; b < rb; b++) {
                        unsigned int gb = bstart + b;
                        unsigned int bi = gb / 8;
                        unsigned int bp = 7 - (gb % 8);
                        unsigned int sbit = rb - 1 - b;
                        if ((v >> sbit) & 1) ptr[bi] |= (1u << bp);
                    }
                }
                ptr += trb;
            }
        }
    }

    /* Pack 2-bit critical type data (MSB-first, 4 per byte) */
    unsigned int tb = (2 * (unsigned int)cur + 7) / 8;
    for (unsigned int b = 0; b < tb; b++) ptr[b] = 0;
    for (unsigned int j = 0; j < (unsigned int)cur; j++) {
        unsigned char t = critical_type[bs + j];
        unsigned int bi = j / 4;
        unsigned int sh = 6 - 2 * (j % 4);
        ptr[bi] |= (t << sh);
    }
}

extern "C"
unsigned char *szp_cuda_float_compress_topology_3d(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize,
    CriticalPoint3D *critical_points, int critical_count,
    int d1, int d2, int d3)
{
    if (absErrBound <= 0.0 || !oriData || nbEle == 0) {
        *outSize = 0;
        return NULL;
    }

    szp_cuda_validate_blocksize(blockSize, SZP_CUDA_TOPO_MAX_BLOCK,
                                "szp_cuda_float_compress_topology_3d");
    szp_cuda_validate_dims(d1, d2, d3,
                           "szp_cuda_float_compress_topology_3d");

    double inver_bound = 1.0 / absErrBound;
    unsigned int bsz = (unsigned int)blockSize;
    size_t num_blocks = (nbEle + bsz - 1) / bsz;

    /* Device allocations */
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, nbEle * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, oriData, nbEle * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* Build critical type array */
    unsigned char *d_ct;
    CUDA_CHECK(cudaMalloc(&d_ct, nbEle));
    CUDA_CHECK(cudaMemset(d_ct, 0, nbEle));

    if (critical_points && critical_count > 0) {
        CriticalPoint3D *d_cp;
        CUDA_CHECK(cudaMalloc(&d_cp, critical_count * sizeof(CriticalPoint3D)));
        CUDA_CHECK(cudaMemcpy(d_cp, critical_points,
                              critical_count * sizeof(CriticalPoint3D),
                              cudaMemcpyHostToDevice));

        int tpb = 256;
        int grid = (critical_count + tpb - 1) / tpb;
        mark_critical_types_3d_kernel<<<grid, tpb>>>(
            d_cp, critical_count, d1, d2, d3, d_ct);
        CUDA_CHECK(cudaGetLastError());
        cudaFree(d_cp);
    }

    /* ---- Standard quantization (no saddle adjustment) ---- */
    int *d_qdata;
    CUDA_CHECK(cudaMalloc(&d_qdata, nbEle * sizeof(int)));
    {
        int qtpb = 256;
        int qgrid = ((int)nbEle + qtpb - 1) / qtpb;
        quantize_3d_kernel_float<<<qgrid, qtpb>>>(d_data, d_qdata, nbEle, inver_bound);
        CUDA_CHECK(cudaGetLastError());
    }

    cudaFree(d_data);

    /* ---- Compress pre-quantized data ---- */
    size_t *d_bs, *d_bo;
    CUDA_CHECK(cudaMalloc(&d_bs, num_blocks * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_bo, num_blocks * sizeof(size_t)));

    int tpb = 256;
    int grid = ((int)num_blocks + tpb - 1) / tpb;
    compress_topo_sizing_3d_preq<<<grid, tpb>>>(d_qdata, nbEle, bsz, d_bs);
    CUDA_CHECK(cudaGetLastError());

    /* Prefix sum */
    thrust::device_ptr<size_t> sp(d_bs);
    thrust::device_ptr<size_t> op(d_bo);
    thrust::exclusive_scan(sp, sp + num_blocks, op);

    size_t lsz, loff;
    CUDA_CHECK(cudaMemcpy(&lsz, d_bs + num_blocks - 1,
                          sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&loff, d_bo + num_blocks - 1,
                          sizeof(size_t), cudaMemcpyDeviceToHost));
    size_t total = loff + lsz;

    size_t hdr = sizeof(size_t);
    *outSize = hdr + total;
    unsigned char *output = (unsigned char *)malloc(*outSize);
    size_t zero = 0;
    memcpy(output, &zero, sizeof(size_t));

    /* Packing pass */
    unsigned char *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, total));
    compress_topo_packing_3d_preq<<<grid, tpb>>>(
        d_qdata, nbEle, bsz, d_ct, d_bo, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(output + hdr, d_out, total, cudaMemcpyDeviceToHost));

    cudaFree(d_qdata);
    cudaFree(d_ct);
    cudaFree(d_bs);
    cudaFree(d_bo);
    cudaFree(d_out);

    return output;
}

/* ================================================================== */
/*  DOUBLE-PRECISION 3D TOPOLOGY COMPRESSION                           */
/* ================================================================== */

extern "C"
unsigned char *szp_cuda_double_compress_topology_3d(
    double *oriData, size_t *outSize, double absErrBound,
    size_t nbEle, int blockSize,
    CriticalPoint3D *critical_points, int critical_count,
    int d1, int d2, int d3)
{
    if (absErrBound <= 0.0 || !oriData || nbEle == 0) {
        *outSize = 0;
        return NULL;
    }

    szp_cuda_validate_blocksize(blockSize, SZP_CUDA_TOPO_MAX_BLOCK,
                                "szp_cuda_double_compress_topology_3d");
    szp_cuda_validate_dims(d1, d2, d3,
                           "szp_cuda_double_compress_topology_3d");

    double inver_bound = 1.0 / absErrBound;
    unsigned int bsz = (unsigned int)blockSize;
    size_t num_blocks = (nbEle + bsz - 1) / bsz;

    /* Device allocations */
    double *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, nbEle * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_data, oriData, nbEle * sizeof(double),
                          cudaMemcpyHostToDevice));

    /* Build critical type array */
    unsigned char *d_ct;
    CUDA_CHECK(cudaMalloc(&d_ct, nbEle));
    CUDA_CHECK(cudaMemset(d_ct, 0, nbEle));

    if (critical_points && critical_count > 0) {
        CriticalPoint3D *d_cp;
        CUDA_CHECK(cudaMalloc(&d_cp, critical_count * sizeof(CriticalPoint3D)));
        CUDA_CHECK(cudaMemcpy(d_cp, critical_points,
                              critical_count * sizeof(CriticalPoint3D),
                              cudaMemcpyHostToDevice));

        int mtpb = 256;
        int mgrid = (critical_count + mtpb - 1) / mtpb;
        mark_critical_types_3d_kernel<<<mgrid, mtpb>>>(
            d_cp, critical_count, d1, d2, d3, d_ct);
        CUDA_CHECK(cudaGetLastError());
        cudaFree(d_cp);
    }

    /* Saddle-aware quantization with double precision */
    int *d_qdata;
    CUDA_CHECK(cudaMalloc(&d_qdata, nbEle * sizeof(int)));
    {
        int qtpb = 256;
        int qgrid = ((int)nbEle + qtpb - 1) / qtpb;
        quantize_3d_kernel_double<<<qgrid, qtpb>>>(d_data, d_qdata, nbEle, inver_bound);
        CUDA_CHECK(cudaGetLastError());
    }

    /* Fix saddles using double-buffering to avoid read-write races.
     * Each pass reads from d_qdata_src and writes corrections to d_qdata.
     * After each pass, copy d_qdata back to d_qdata_src for the next pass. */
    {
        size_t total_interior = (size_t)(d1-2) * (size_t)(d2-2) * (size_t)(d3-2);
        int stpb = 256;
        int sgrid = (int)((total_interior + stpb - 1) / stpb);

        int *d_qdata_src;
        CUDA_CHECK(cudaMalloc(&d_qdata_src, nbEle * sizeof(int)));

        for (int pass = 0; pass < 3; pass++) {
            /* Snapshot current state into the read buffer */
            CUDA_CHECK(cudaMemcpy(d_qdata_src, d_qdata,
                                  nbEle * sizeof(int),
                                  cudaMemcpyDeviceToDevice));
            fix_saddles_3d_kernel<<<sgrid, stpb>>>(
                d_qdata_src, d_qdata, d_ct, d1, d2, d3, nbEle,
                total_interior);
            CUDA_CHECK(cudaGetLastError());
        }

        cudaFree(d_qdata_src);
    }

    cudaFree(d_data);

    /* Compress pre-quantized data (same kernels as float — they work on int) */
    size_t *d_bs, *d_bo;
    CUDA_CHECK(cudaMalloc(&d_bs, num_blocks * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_bo, num_blocks * sizeof(size_t)));

    int tpb = 256;
    int grid = ((int)num_blocks + tpb - 1) / tpb;
    compress_topo_sizing_3d_preq<<<grid, tpb>>>(d_qdata, nbEle, bsz, d_bs);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<size_t> sp(d_bs);
    thrust::device_ptr<size_t> op(d_bo);
    thrust::exclusive_scan(sp, sp + num_blocks, op);

    size_t lsz, loff;
    CUDA_CHECK(cudaMemcpy(&lsz, d_bs + num_blocks - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&loff, d_bo + num_blocks - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    size_t total = loff + lsz;

    size_t hdr = sizeof(size_t);
    *outSize = hdr + total;
    unsigned char *output = (unsigned char *)malloc(*outSize);
    size_t zero = 0;
    memcpy(output, &zero, sizeof(size_t));

    unsigned char *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, total));
    compress_topo_packing_3d_preq<<<grid, tpb>>>(
        d_qdata, nbEle, bsz, d_ct, d_bo, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(output + hdr, d_out, total, cudaMemcpyDeviceToHost));

    cudaFree(d_qdata);
    cudaFree(d_ct);
    cudaFree(d_bs);
    cudaFree(d_bo);
    cudaFree(d_out);

    return output;
}
