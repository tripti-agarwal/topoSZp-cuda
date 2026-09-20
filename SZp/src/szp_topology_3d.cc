/**
 *  @file szp_topology_3d.cc
 *  @brief OpenMP implementation of 3D topology-preserving compression functions.
 *
 *  Extends TopoSZp from 2D (4-connected) to 3D (6-connected) grids.
 *  Uses face-adjacent neighborhood (±x, ±y, ±z) for critical point detection.
 *  Flat index: x * d2 * d3 + y * d3 + z
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
#include "szp_topology_3d.h"
#include "szp_TypeManager.h"
#include "szp_CompressionToolkit.h"
#include "szp_defines.h"
#ifdef _OPENMP
#include "omp.h"
#endif

using namespace szp;

/* ================================================================== */
/*  1.  CRITICAL POINT FINDING (3D, 6-connected)                       */
/* ================================================================== */

CriticalPoint3D *szp_find_critical_points_3d(float *data, size_t *outCount,
                                              int d1, int d2, int d3,
                                              float absErrBound)
{
#ifdef _OPENMP
    if (!data || d1 <= 2 || d2 <= 2 || d3 <= 2) {
        *outCount = 0;
        return NULL;
    }

    size_t total_interior = (size_t)(d1 - 2) * (size_t)(d2 - 2) * (size_t)(d3 - 2);
    CriticalPoint3D *results = (CriticalPoint3D *)malloc(total_interior * sizeof(CriticalPoint3D));
    if (!results) { *outCount = 0; return NULL; }

    size_t count = 0;
    double inver_bound = 1.0 / absErrBound;

    #pragma omp parallel
    {
        int nbThreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        size_t chunk = total_interior / nbThreads;
        size_t start = tid * chunk;
        size_t end = (tid == nbThreads - 1) ? total_interior : (tid + 1) * chunk;

        size_t local_count = 0;
        CriticalPoint3D *local_results = (CriticalPoint3D *)malloc(
            (end - start) * sizeof(CriticalPoint3D));

        size_t d2m2 = (size_t)(d2 - 2);
        size_t d3m2 = (size_t)(d3 - 2);
        size_t plane = d2m2 * d3m2;

        for (size_t idx = start; idx < end; idx++) {
            int x = 1 + (int)(idx / plane);
            size_t rem = idx % plane;
            int y = 1 + (int)(rem / d3m2);
            int z = 1 + (int)(rem % d3m2);

            size_t flat = (size_t)x * d2 * d3 + (size_t)y * d3 + z;
            float center = data[flat];
            float xm = data[((size_t)(x-1)) * d2 * d3 + (size_t)y * d3 + z];
            float xp = data[((size_t)(x+1)) * d2 * d3 + (size_t)y * d3 + z];
            float ym = data[(size_t)x * d2 * d3 + ((size_t)(y-1)) * d3 + z];
            float yp = data[(size_t)x * d2 * d3 + ((size_t)(y+1)) * d3 + z];
            float zm = data[(size_t)x * d2 * d3 + (size_t)y * d3 + (z-1)];
            float zp = data[(size_t)x * d2 * d3 + (size_t)y * d3 + (z+1)];

            int type = 0;
            if (center > xm && center > xp && center > ym &&
                center > yp && center > zm && center > zp) {
                type = 1;  /* local maximum */
            } else if (center < xm && center < xp && center < ym &&
                       center < yp && center < zm && center < zp) {
                type = 2;  /* local minimum */
            } else {
                /* Saddle: high along some axes, low along others */
                int x_high = (center > xm) && (center > xp);
                int x_low  = (center < xm) && (center < xp);
                int y_high = (center > ym) && (center > yp);
                int y_low  = (center < ym) && (center < yp);
                int z_high = (center > zm) && (center > zp);
                int z_low  = (center < zm) && (center < zp);
                int high_axes = x_high + y_high + z_high;
                int low_axes  = x_low  + y_low  + z_low;
                if (high_axes >= 1 && low_axes >= 1) {
                    type = 3;  /* saddle */
                }
            }

            if (type != 0) {
                int qbin = (int)((center + absErrBound) * inver_bound);
                local_results[local_count].x = x;
                local_results[local_count].y = y;
                local_results[local_count].z = z;
                local_results[local_count].type = type;
                local_results[local_count].quantized_bin = qbin;
                local_results[local_count].sort_position = 0;
                local_count++;
            }
        }

        size_t my_offset;
        #pragma omp atomic capture
        { my_offset = count; count += local_count; }

        memcpy(results + my_offset, local_results, local_count * sizeof(CriticalPoint3D));
        free(local_results);
    }

    *outCount = count;
    if (count == 0) { free(results); return NULL; }
    results = (CriticalPoint3D *)realloc(results, count * sizeof(CriticalPoint3D));
    return results;
#else
    printf("Error! OpenMP not supported!\n");
    *outCount = 0;
    return NULL;
#endif
}

/* ================================================================== */
/*  2.  SORT CRITICAL POINTS BY DATA VALUE WITHIN BINS                 */
/* ================================================================== */

void szp_sort_critical_points_3d(CriticalPoint3D *critical_points,
                                  size_t critical_count,
                                  float *data, int d2, int d3)
{
#ifdef _OPENMP
    if (!critical_points || critical_count == 0 || !data) return;

    /* Find min/max bin */
    int min_bin = critical_points[0].quantized_bin;
    int max_bin = critical_points[0].quantized_bin;
    #pragma omp parallel for reduction(min:min_bin) reduction(max:max_bin)
    for (size_t i = 1; i < critical_count; i++) {
        int bin = critical_points[i].quantized_bin;
        if (bin < min_bin) min_bin = bin;
        if (bin > max_bin) max_bin = bin;
    }

    int bin_range = max_bin - min_bin + 1;
    bool *bin_exists = (bool *)calloc(bin_range, sizeof(bool));
    int *unique_bins = (int *)malloc(critical_count * sizeof(int));
    int num_unique_bins = 0;

    /* Collect unique bins */
    for (size_t i = 0; i < critical_count; i++) {
        int hash = critical_points[i].quantized_bin - min_bin;
        if (!bin_exists[hash]) {
            bin_exists[hash] = true;
            unique_bins[num_unique_bins++] = critical_points[i].quantized_bin;
        }
    }

    /* Build per-bin index */
    size_t *bin_counts = (size_t *)calloc(bin_range, sizeof(size_t));
    size_t *bin_offsets = (size_t *)malloc(bin_range * sizeof(size_t));
    size_t *bin_to_indices = (size_t *)malloc(critical_count * sizeof(size_t));

    for (size_t i = 0; i < critical_count; i++)
        bin_counts[critical_points[i].quantized_bin - min_bin]++;

    bin_offsets[0] = 0;
    for (int b = 1; b < bin_range; b++)
        bin_offsets[b] = bin_offsets[b-1] + bin_counts[b-1];

    memset(bin_counts, 0, bin_range * sizeof(size_t));
    for (size_t i = 0; i < critical_count; i++) {
        int hash = critical_points[i].quantized_bin - min_bin;
        bin_to_indices[bin_offsets[hash] + bin_counts[hash]] = i;
        bin_counts[hash]++;
    }

    /* Sort each bin by data value, assign sort positions */
    #pragma omp parallel for schedule(static)
    for (int bi = 0; bi < num_unique_bins; bi++) {
        int hash = unique_bins[bi] - min_bin;
        size_t bcount = bin_counts[hash];
        size_t boff = bin_offsets[hash];

        size_t *local_idx = (size_t *)malloc(bcount * sizeof(size_t));
        memcpy(local_idx, &bin_to_indices[boff], bcount * sizeof(size_t));

        /* Insertion sort by data value (3D flat index) */
        for (size_t ii = 1; ii < bcount; ii++) {
            size_t key = local_idx[ii];
            size_t flat_key = (size_t)critical_points[key].x * d2 * d3 +
                              (size_t)critical_points[key].y * d3 +
                              critical_points[key].z;
            float key_val = data[flat_key];
            size_t jj = ii;
            while (jj > 0) {
                size_t prev = local_idx[jj - 1];
                size_t flat_prev = (size_t)critical_points[prev].x * d2 * d3 +
                                   (size_t)critical_points[prev].y * d3 +
                                   critical_points[prev].z;
                if (data[flat_prev] > key_val) {
                    local_idx[jj] = local_idx[jj - 1];
                    jj--;
                } else break;
            }
            local_idx[jj] = key;
        }

        /* Assign sort positions (ties share the same position) */
        int pos = 0;
        size_t first = local_idx[0];
        size_t flat0 = (size_t)critical_points[first].x * d2 * d3 +
                        (size_t)critical_points[first].y * d3 +
                        critical_points[first].z;
        float prev_val = data[flat0];
        critical_points[first].sort_position = 0;

        for (size_t ii = 1; ii < bcount; ii++) {
            size_t ci = local_idx[ii];
            size_t flat_ci = (size_t)critical_points[ci].x * d2 * d3 +
                              (size_t)critical_points[ci].y * d3 +
                              critical_points[ci].z;
            float cur_val = data[flat_ci];
            if (cur_val != prev_val) pos++;
            critical_points[ci].sort_position = pos;
            prev_val = cur_val;
        }

        free(local_idx);
    }

    free(bin_exists);
    free(unique_bins);
    free(bin_counts);
    free(bin_offsets);
    free(bin_to_indices);
#else
    (void)critical_points; (void)critical_count; (void)data; (void)d2; (void)d3;
#endif
}

/* ================================================================== */
/*  3.  COMPRESS SORT POSITIONS (3D — same algorithm as 2D)            */
/* ================================================================== */

unsigned char *szp_compress_sort_positions_3d(CriticalPoint3D *critical_points,
                                               size_t critical_count,
                                               size_t *outSize, int blockSize)
{
#ifdef _OPENMP
    if (!critical_points || critical_count == 0) {
        *outSize = 0;
        return NULL;
    }

    /* Count extrema (types 1 and 2) */
    size_t extrema_count = 0;
    for (size_t i = 0; i < critical_count; i++) {
        if (critical_points[i].type == 1 || critical_points[i].type == 2)
            extrema_count++;
    }
    if (extrema_count == 0) { *outSize = 0; return NULL; }

    /* Extract sort_position values for extrema */
    int *sort_positions = (int *)malloc(extrema_count * sizeof(int));
    if (!sort_positions) { *outSize = 0; return NULL; }

    size_t ei = 0;
    for (size_t i = 0; i < critical_count; i++) {
        if (critical_points[i].type == 1 || critical_points[i].type == 2)
            sort_positions[ei++] = critical_points[i].sort_position;
    }

    /* Allocate output buffer */
    size_t maxBufSize = sizeof(size_t) + (extrema_count * (sizeof(int) + 32)) + 1024;
    unsigned char *output = (unsigned char *)malloc(maxBufSize);
    if (!output) { free(sort_positions); *outSize = 0; return NULL; }

    /* Write extrema_count as header */
    memcpy(output, &extrema_count, sizeof(size_t));
    unsigned char *outputBytes = output + sizeof(size_t);

    size_t maxPreservedBufferSize_perthread = 0;
    unsigned char *real_outputBytes;
    struct PaddedSize {
        size_t size;
        char padding[64 - sizeof(size_t)];
    };
    struct PaddedSize *outSize_perthread_arr;
    size_t *offsets_perthread_arr;

    *outSize = sizeof(size_t);

    unsigned int nbThreads = 0;
    unsigned int block_size = blockSize;
    if (block_size == 0) { free(sort_positions); free(output); *outSize = 0; return NULL; }

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            if (nbThreads == 0) nbThreads = 1;
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            *outSize += nbThreads * sizeof(size_t);
            outSize_perthread_arr = (struct PaddedSize *)malloc(nbThreads * sizeof(struct PaddedSize));
            offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));

            size_t elements_per_thread = (extrema_count + nbThreads - 1) / nbThreads;
            size_t blocks_pt = (elements_per_thread + block_size - 1) / block_size;
            size_t worst_per_block = sizeof(int) + 1;
            if (block_size > 1) {
                worst_per_block += ((block_size - 1) + 7) / 8;
                worst_per_block += ((block_size - 1) * 32 + 7) / 8;
            }
            maxPreservedBufferSize_perthread = blocks_pt * worst_per_block + 1024;
            if (maxPreservedBufferSize_perthread < 1024) maxPreservedBufferSize_perthread = 1024;
        }

        unsigned char *outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
        size_t outSize_perthread = 0;

        int tid = omp_get_thread_num();
        size_t num_blocks = (extrema_count + block_size - 1) / block_size;
        size_t blocks_per_thread = (num_blocks + nbThreads - 1) / nbThreads;
        size_t start_block = tid * blocks_per_thread;
        size_t end_block = (tid + 1) * blocks_per_thread;
        if (end_block > num_blocks) end_block = num_blocks;

        int prior = 0, current = 0, diff = 0;
        unsigned int max_val = 0, bit_count = 0;
        unsigned char *block_pointer = outputBytes_perthread;

        size_t temp_arr_size = (block_size > 1) ? (block_size - 1) : 1;
        unsigned char *temp_sign_arr = (unsigned char *)malloc(temp_arr_size * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc(temp_arr_size * sizeof(unsigned int));
        unsigned int signbytelength = 0, savedbitsbytelength = 0;

        if (outputBytes_perthread && temp_sign_arr && temp_predict_arr) {
            for (size_t block_idx = start_block; block_idx < end_block; block_idx++) {
                size_t i = block_idx * block_size;
                if (i >= extrema_count) break;
                size_t current_block_size = (i + block_size > extrema_count) ?
                                             (extrema_count - i) : (size_t)block_size;
                if (current_block_size == 0) continue;

                max_val = 0;
                prior = sort_positions[i];
                memcpy(block_pointer, &prior, sizeof(int));
                block_pointer += sizeof(int);
                outSize_perthread += sizeof(int);

                if (current_block_size > 1) {
                    for (size_t j = 0; j < current_block_size - 1; j++) {
                        current = sort_positions[i + j + 1];
                        diff = current - prior;
                        prior = current;
                        if (diff < 0) {
                            temp_sign_arr[j] = 1;
                            temp_predict_arr[j] = (unsigned int)(-diff);
                        } else {
                            temp_sign_arr[j] = 0;
                            temp_predict_arr[j] = (unsigned int)diff;
                        }
                        if (temp_predict_arr[j] > max_val) max_val = temp_predict_arr[j];
                    }
                }

                if (max_val == 0) {
                    *block_pointer++ = 0;
                    outSize_perthread++;
                } else {
                    bit_count = (unsigned int)(log2f((float)max_val)) + 1;
                    *block_pointer++ = (unsigned char)bit_count;
                    outSize_perthread++;
                    signbytelength = convertIntArray2ByteArray_fast_1b_args(
                        temp_sign_arr, current_block_size - 1, block_pointer);
                    block_pointer += signbytelength;
                    outSize_perthread += signbytelength;
                    savedbitsbytelength = Jiajun_save_fixed_length_bits(
                        temp_predict_arr, current_block_size - 1, block_pointer, bit_count);
                    block_pointer += savedbitsbytelength;
                    outSize_perthread += savedbitsbytelength;
                }
            }
        }

        outSize_perthread_arr[tid].size = outSize_perthread;
#pragma omp barrier
#pragma omp single
        {
            offsets_perthread_arr[0] = 0;
            for (unsigned int k = 1; k < nbThreads; k++)
                offsets_perthread_arr[k] = offsets_perthread_arr[k-1] + outSize_perthread_arr[k-1].size;
            *outSize += offsets_perthread_arr[nbThreads-1] + outSize_perthread_arr[nbThreads-1].size;
            memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));
        }
#pragma omp barrier
        if (outputBytes_perthread && real_outputBytes)
            memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);

        if (outputBytes_perthread) free(outputBytes_perthread);
        if (temp_sign_arr) free(temp_sign_arr);
        if (temp_predict_arr) free(temp_predict_arr);
#pragma omp single
        {
            free(outSize_perthread_arr);
            free(offsets_perthread_arr);
        }
    }

    free(sort_positions);
    return output;
#else
    printf("Error! OpenMP not supported!\n");
    *outSize = 0;
    return NULL;
#endif
}

/* ================================================================== */
/*  4.  TOPOLOGY-PRESERVED COMPRESSION (3D)                            */
/*      Same compressed format as 2D — just uses 3D flat indices.      */
/* ================================================================== */

unsigned char *szp_float_compress_topology_3d(
    float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize,
    CriticalPoint3D *critical_points, int critical_count,
    int d1, int d2, int d3)
{
#ifdef _OPENMP
    if (absErrBound <= 0.0 || !oriData || nbEle == 0) {
        *outSize = 0;
        return NULL;
    }

    float *op = oriData;
    size_t maxPreservedBufferSize = 8ull * nbEle + 1024ull;
    size_t maxPreservedBufferSize_perthread = 0;
    unsigned char *outputBytes = (unsigned char *)malloc(maxPreservedBufferSize);
    if (!outputBytes) { *outSize = 0; return NULL; }
    unsigned char *real_outputBytes = NULL;
    struct PaddedSize {
        size_t size;
        char padding[64 - sizeof(size_t)];
    };
    struct PaddedSize *outSize_perthread_arr = NULL;
    size_t *offsets_perthread_arr = NULL;

    *outSize = 0;

    /* Build critical_type array using 3D flat indices */
    unsigned char *critical_type = (unsigned char *)calloc(nbEle, 1);
    if (!critical_type) { free(outputBytes); *outSize = 0; return NULL; }

    #pragma omp parallel for
    for (int i = 0; i < critical_count; i++) {
        int x = critical_points[i].x, y = critical_points[i].y, z = critical_points[i].z;
        if (x < 0 || x >= d1 || y < 0 || y >= d2 || z < 0 || z >= d3) continue;
        size_t flat = (size_t)x * d2 * d3 + (size_t)y * d3 + z;
        unsigned char t = (unsigned char)critical_points[i].type;
        if (t >= 1 && t <= 3) {
            critical_type[flat] = t;
        }
    }

    unsigned int nbThreads = 0;
    double inver_bound = 1.0 / absErrBound;
    unsigned int block_size = (unsigned int)blockSize;
    if (block_size == 0) { free(outputBytes); free(critical_type); *outSize = 0; return NULL; }
    unsigned int new_block_size = (block_size > 1) ? (block_size - 1) : 1;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            if (nbThreads == 0) nbThreads = 1;
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            *outSize += nbThreads * sizeof(size_t);
            outSize_perthread_arr = (struct PaddedSize *)malloc(nbThreads * sizeof(struct PaddedSize));
            offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));

            size_t header_size = nbThreads * sizeof(size_t);
            if (maxPreservedBufferSize > header_size && nbThreads > 0) {
                maxPreservedBufferSize_perthread = (maxPreservedBufferSize - header_size) / nbThreads;
            } else {
                maxPreservedBufferSize_perthread = 1024;
            }
            if (maxPreservedBufferSize_perthread < 1024) maxPreservedBufferSize_perthread = 1024;
        }

        size_t i = 0, j = 0;
        unsigned char *outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
        size_t outSize_perthread = 0;

        int tid = omp_get_thread_num();
        size_t num_blocks = (nbEle + block_size - 1) / block_size;
        size_t blocks_per_thread = (num_blocks + nbThreads - 1) / nbThreads;
        size_t start_block = tid * blocks_per_thread;
        size_t end_block = (tid + 1) * blocks_per_thread;
        if (end_block > num_blocks) end_block = num_blocks;

        int prior = 0, current = 0, diff = 0;
        unsigned int maxv = 0, bit_count = 0;
        unsigned char *block_pointer = outputBytes_perthread;

        unsigned char *temp_sign_arr = (unsigned char *)malloc(new_block_size * sizeof(unsigned char));
        unsigned char *temp_type_arr = (unsigned char *)malloc(block_size * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc(new_block_size * sizeof(unsigned int));
        unsigned int signbytelength = 0, savedbitsbytelength = 0, typebytelength = 0;

        if (outputBytes_perthread && temp_sign_arr && temp_type_arr && temp_predict_arr) {
            for (size_t block_idx = start_block; block_idx < end_block; block_idx++) {
                i = block_idx * block_size;
                if (i >= nbEle) break;
                size_t current_block_size = (i + block_size > nbEle) ? (nbEle - i) : block_size;
                if (current_block_size == 0) continue;

                maxv = 0;
                prior = (int)((double)op[i] * inver_bound);
                memcpy(block_pointer, &prior, sizeof(int));
                block_pointer += sizeof(int);
                outSize_perthread += sizeof(int);

                if (current_block_size > 1) {
                    unsigned int actual_new = current_block_size - 1;
                    for (j = 0; j < actual_new; j++) {
                        current = (int)((double)op[i + j + 1] * inver_bound);
                        diff = current - prior;
                        prior = current;
                        if (diff == 0) { temp_sign_arr[j] = 0; temp_predict_arr[j] = 0; }
                        else if (diff > 0) {
                            temp_sign_arr[j] = 0;
                            temp_predict_arr[j] = (unsigned int)diff;
                            if ((unsigned int)diff > maxv) maxv = (unsigned int)diff;
                        } else {
                            temp_sign_arr[j] = 1;
                            unsigned int ad = (unsigned int)(-diff);
                            temp_predict_arr[j] = ad;
                            if (ad > maxv) maxv = ad;
                        }
                    }
                }

                /* Collect critical types for this block */
                for (j = 0; j < current_block_size; j++) {
                    size_t idx = i + j;
                    temp_type_arr[j] = (idx < nbEle) ? critical_type[idx] : 0;
                }

                if (maxv == 0) {
                    *block_pointer++ = 0;
                    outSize_perthread++;
                } else {
                    bit_count = (unsigned int)floorf(log2f((float)maxv)) + 1u;
                    *block_pointer++ = (unsigned char)bit_count;
                    outSize_perthread++;
                    unsigned int actual_new = (current_block_size > 1) ? (current_block_size - 1) : 0;
                    if (actual_new > 0) {
                        signbytelength = convertIntArray2ByteArray_fast_1b_args(
                            temp_sign_arr, actual_new, block_pointer);
                        block_pointer += signbytelength;
                        outSize_perthread += signbytelength;
                        savedbitsbytelength = Jiajun_save_fixed_length_bits(
                            temp_predict_arr, actual_new, block_pointer, bit_count);
                        block_pointer += savedbitsbytelength;
                        outSize_perthread += savedbitsbytelength;
                    }
                }

                /* Pack 2-bit type data */
                unsigned char *type_output = NULL;
                typebytelength = convertIntArray2ByteArray_fast_2b(
                    temp_type_arr, current_block_size, &type_output);
                memcpy(block_pointer, type_output, typebytelength);
                free(type_output);
                block_pointer += typebytelength;
                outSize_perthread += typebytelength;
            }
        }

        if (outSize_perthread_arr) outSize_perthread_arr[tid].size = outSize_perthread;

#pragma omp barrier
#pragma omp single
        {
            if (outSize_perthread_arr && offsets_perthread_arr) {
                offsets_perthread_arr[0] = 0;
                for (i = 1; i < nbThreads; i++)
                    offsets_perthread_arr[i] = offsets_perthread_arr[i-1] + outSize_perthread_arr[i-1].size;
                *outSize += offsets_perthread_arr[nbThreads-1] + outSize_perthread_arr[nbThreads-1].size;
                memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t));
            }
        }
#pragma omp barrier
        if (outSize_perthread_arr && outputBytes_perthread && real_outputBytes && offsets_perthread_arr)
            memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);
#pragma omp barrier

        if (outputBytes_perthread) free(outputBytes_perthread);
        if (temp_sign_arr) free(temp_sign_arr);
        if (temp_type_arr) free(temp_type_arr);
        if (temp_predict_arr) free(temp_predict_arr);
#pragma omp single
        {
            if (outSize_perthread_arr) free(outSize_perthread_arr);
            if (offsets_perthread_arr) free(offsets_perthread_arr);
        }
    }

    free(critical_type);

    if (*outSize > 0) {
        unsigned char *tmp = (unsigned char *)realloc(outputBytes, *outSize);
        if (tmp) outputBytes = tmp;
    }

    return outputBytes;
#else
    *outSize = 0;
    return NULL;
#endif
}
