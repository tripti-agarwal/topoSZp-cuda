/**
 *  @file szp_float.h
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>, Sheng Di <sdi1@anl.gov>
 *  @date Oct, 2023
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <limits.h>
#include "szp.h"
#include "szp_float.h"
#include <assert.h>
#include <math.h>
#include "szp_TypeManager.h"
#include "szp_quant.h"
#include "szp_detect_threads.h"
#include "szp_CompressionToolkit.h"
#include "szp_cp_rank.h"

#ifdef _OPENMP
#include "omp.h"
#endif
#include <stdlib.h>
#ifdef _POSIX_C_SOURCE
#include <malloc.h>  // For posix_memalign
#endif

using namespace szp;

CriticalPoint *szp_find_critical_points(float *data, size_t *outCount, int rows, int cols, float absErrBound) {
    #ifdef _OPENMP
    if (!data || rows <= 2 || cols <= 2) {
        *outCount = 0;
        return NULL;
    }

    const size_t interior = (size_t)(rows - 2) * (size_t)(cols - 2);
    CriticalPoint *results = (CriticalPoint *)malloc(interior * sizeof(CriticalPoint));
    if (!results) {
        *outCount = 0;
        return NULL;
    }

    /* Same bin formula as the compressor's quantization, so points that share
       a bin here also share a decompressed value. */
    const double inver_bound = 1.0 / absErrBound;
    int nbThreads = 1;
    size_t *prefix = NULL;   /* prefix[t] = first output slot of thread t */
    int failed = 0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            nbThreads = omp_get_num_threads();
            prefix = (size_t *)calloc((size_t)nbThreads + 1, sizeof(size_t));
            if (!prefix) failed = 1;
        }

        /* Each thread scans one contiguous range of the interior in raster
           order; the ranges are concatenated in thread order below, so the
           output is in raster order for any thread count. */
        const int tid = omp_get_thread_num();
        const size_t chunk = interior / (size_t)nbThreads;
        const size_t start_idx = (size_t)tid * chunk;
        const size_t end_idx = (tid == nbThreads - 1) ? interior : start_idx + chunk;

        size_t local_count = 0;
        CriticalPoint *local_results = NULL;
        if (!failed && end_idx > start_idx) {
            local_results = (CriticalPoint *)malloc((end_idx - start_idx) * sizeof(CriticalPoint));
            if (!local_results) {
                #pragma omp atomic write
                failed = 1;
            }
        }

        if (local_results) {
            for (size_t idx = start_idx; idx < end_idx; idx++) {
                int i = 1 + (int)(idx / (size_t)(cols - 2));
                int j = 1 + (int)(idx % (size_t)(cols - 2));

                float center = data[(size_t)i * cols + j];
                float up = data[(size_t)(i-1) * cols + j];
                float down = data[(size_t)(i+1) * cols + j];
                float left = data[(size_t)i * cols + (j-1)];
                float right = data[(size_t)i * cols + (j+1)];

                int type = 0;
                if (center > up && center > down && center > left && center > right) {
                    type = 1;   /* local maximum */
                } else if (center < up && center < down && center < left && center < right) {
                    type = 2;   /* local minimum */
                } else if ((center < up && center < down && center > left && center > right) ||
                           (center > up && center > down && center < left && center < right)) {
                    type = 3;   /* saddle */
                }
                if (type) {
                    int quantized_bin = szp_quantize<float>(center, (double)absErrBound, inver_bound);
                    local_results[local_count++] = (CriticalPoint){i, j, type, quantized_bin, 0};
                }
            }
        }

        if (prefix) prefix[tid + 1] = local_count;
        #pragma omp barrier
        #pragma omp single
        {
            if (prefix)
                for (int t = 0; t < nbThreads; t++) prefix[t + 1] += prefix[t];
        }
        /* implicit barrier after single */

        if (!failed && local_results)
            memcpy(results + prefix[tid], local_results, local_count * sizeof(CriticalPoint));
        free(local_results);
    }

    size_t count = (!failed && prefix) ? prefix[nbThreads] : 0;
    free(prefix);
    *outCount = count;

    if (count == 0) {
        free(results);
        return NULL;
    }

    CriticalPoint *shrunk = (CriticalPoint *)realloc(results, count * sizeof(CriticalPoint));
    return shrunk ? shrunk : results;
    #else
        printf("Error! OpenMP not supported!\n");
        *outCount = 0;
        return NULL;
    #endif
}

/**
 * Rank critical points by their original value inside each quantization bin.
 * Ranks start at 0 within each bin and equal values share a rank, e.g. CP1 in
 * bin 1 with value 0.01 gets 0, CP2 in bin 1 with value 0.02 gets 1, CP3 in
 * bin 2 gets 0. The ranks depend only on the data, not on the thread count or
 * on the order of the list.
 *
 * @param critical_points Array of critical points to rank
 * @param critical_count Number of critical points
 * @param data Original data array (32-bit float values)
 * @param cols Number of columns in the data grid
 */
void szp_sort_critical_points_by_original_data(CriticalPoint *critical_points, size_t critical_count,
                                             float *data, int cols) {
    const size_t c = (size_t)cols;
    szp_assign_sort_positions(critical_points, critical_count, data,
        [c](const CriticalPoint &p) { return (size_t)p.x * c + (size_t)p.y; });
}

/**
 * Compress sort_position values from critical points using integer compression with blocking and prediction.
 * Optimized for integer data with OpenMP threading, blocking, and prediction-based compression.
 * 
 * @param critical_points Array of critical points with sort_position values
 * @param critical_count Number of critical points
 * @param outSize Output size in bytes
 * @param blockSize Block size for compression
 * @return Compressed data as unsigned char array
 */
unsigned char *
szp_compress_sort_positions(CriticalPoint *critical_points, size_t critical_count, size_t *outSize, int blockSize) {
#ifdef _OPENMP
    if (!critical_points || critical_count == 0) {
        *outSize = 0;
        return NULL;
    }
    
    // Count maxima and minima only (types 1 and 2)
    size_t extrema_count = 0;
    for (size_t i = 0; i < critical_count; i++) {
        if (critical_points[i].type == 1 || critical_points[i].type == 2) {
            extrema_count++;
        }
    }
    
    if (extrema_count == 0) {
        *outSize = 0;
        return NULL;
    }
    
    // Extract sort_position values for extrema only
    int *sort_positions = (int *)malloc(extrema_count * sizeof(int));
    if (!sort_positions) {
        *outSize = 0;
        return NULL;
    }
    
    // Copy sort_position values for extrema only, in list order. The list is
    // in raster order, which is the order the decoder assigns ranks in.
    // A single pass is cheap and keeps the order independent of thread count.
    {
        size_t k = 0;
        for (size_t i = 0; i < critical_count; i++) {
            if (critical_points[i].type == 1 || critical_points[i].type == 2) {
                sort_positions[k++] = critical_points[i].sort_position;
            }
        }
    }

    // Allocate output buffer with reasonable size estimate
    // Each sort position is an int, worst case we need: header + offsets + compressed data
    // Estimate: header (sizeof(size_t)) + thread offsets (will be calculated) + 
    // worst case: each element needs sizeof(int) + overhead
    size_t maxPreservedBufferSize = sizeof(size_t) + // header for extrema_count
                                    (size_t)omp_get_max_threads() * sizeof(size_t) + // per-thread offset table
                                    (extrema_count * (sizeof(int) + 32)) + // worst case compressed size
                                    1024; // safety margin
    // Check for overflow
    if (maxPreservedBufferSize < sizeof(size_t) || maxPreservedBufferSize < extrema_count) {
        // Overflow detected, use a safer calculation
        const size_t max_safe = (SIZE_MAX / 2 > 1024) ? (SIZE_MAX / 2) : 1024;
        maxPreservedBufferSize = max_safe;
    }
    unsigned char *output = (unsigned char *)malloc(maxPreservedBufferSize);
    if (!output) {
        free(sort_positions);
        *outSize = 0;
        return NULL;
    }
    
    // Write extrema_count as header
    memcpy(output, &extrema_count, sizeof(size_t));
    unsigned char *outputBytes = output + sizeof(size_t);
    
    size_t maxPreservedBufferSize_perthread = 0;
    unsigned char *real_outputBytes; 
    // Use padded structure to avoid false sharing (64-byte cache line alignment)
    struct PaddedSize {
        size_t size;
        char padding[64 - sizeof(size_t)];  // Pad to 64-byte cache line boundary
    };
    struct PaddedSize *outSize_perthread_arr;
    size_t *offsets_perthread_arr;
    
    (*outSize) = sizeof(size_t); // Start with header size
    
    unsigned int nbThreads = 0;
    unsigned int threadblocksize = 0;
    unsigned int block_size = blockSize;
    
    // Validate block_size to prevent underflow/overflow
    if (block_size == 0) {
        free(sort_positions);
        free(output);
        *outSize = 0;
        return NULL;
    }

    // Adaptive thread limiting for better scaling with high thread counts
    size_t num_blocks_estimate = (extrema_count + block_size - 1) / block_size;
    int optimal_threads = 0;  // Will be set in parallel region

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            if (nbThreads == 0) nbThreads = 1; // Safety check
            
            // For high thread counts with small problems, limit effective parallelism
            if (nbThreads >= 16 && num_blocks_estimate < nbThreads * 8) {
                optimal_threads = (num_blocks_estimate + 7) / 8;
                if (optimal_threads < 1) optimal_threads = 1;
            } else {
                optimal_threads = nbThreads;
            }
            
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            (*outSize) += nbThreads * sizeof(size_t); 
            // Allocate padded structures to avoid false sharing
            outSize_perthread_arr = (struct PaddedSize *)malloc(nbThreads * sizeof(struct PaddedSize));
            offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));
            if (!outSize_perthread_arr || !offsets_perthread_arr) {
                // Set error flag - threads will see NULL and skip processing
                if (outSize_perthread_arr) free(outSize_perthread_arr);
                if (offsets_perthread_arr) free(offsets_perthread_arr);
                outSize_perthread_arr = NULL;
                offsets_perthread_arr = NULL;
            }

            // Conservative buffer size estimate: account for worst-case compression
            // Each block needs: initial int (4 bytes) + bit_count (1 byte) + 
            // sign array (ceil((block_size-1)/8)) + saved bits (ceil((block_size-1)*bit_count/8))
            // Use a safety factor of 2x to account for variable compression ratios
            size_t elements_per_thread = (extrema_count + nbThreads - 1) / nbThreads;
            size_t blocks_per_thread = (elements_per_thread > 0 && block_size > 0) ? 
                                       (elements_per_thread + block_size - 1) / block_size : 1;
            // Worst case: each block needs ~4 + 1 + (block_size-1)/8 + (block_size-1)*32/8 bytes
            // Ensure block_size > 0 to prevent underflow
            size_t worst_case_per_block = sizeof(int) + 1;
            if (block_size > 1) {
                worst_case_per_block += ((block_size - 1) + 7) / 8;
                worst_case_per_block += ((block_size - 1) * 32 + 7) / 8;
            }
            // Check for integer overflow in multiplication
            size_t base_size = blocks_per_thread * worst_case_per_block;
            if (base_size < blocks_per_thread || base_size < worst_case_per_block) {
                // Integer overflow detected, use a safe maximum
                maxPreservedBufferSize_perthread = extrema_count * (sizeof(int) + 32) + 1024;
            } else {
                maxPreservedBufferSize_perthread = base_size + 1024; // Add safety margin
            }
            // Ensure minimum buffer size
            if (maxPreservedBufferSize_perthread < 1024) {
                maxPreservedBufferSize_perthread = 1024;
            }
            threadblocksize = (extrema_count > 0) ? extrema_count / nbThreads : 0;
        }
        size_t i = 0;
        size_t j = 0;
        unsigned char *outputBytes_perthread = NULL;
        // Use cache-aligned allocation for better performance
        if (maxPreservedBufferSize_perthread > 0) {
            #if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
            if (posix_memalign((void**)&outputBytes_perthread, 64, maxPreservedBufferSize_perthread) != 0) {
                outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
            }
            #else
            outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
            #endif
        }
        size_t outSize_perthread = 0;
        
        int tid = omp_get_thread_num();
        
        // Use block-based distribution for better cache locality and load balancing
        size_t num_blocks = (extrema_count + block_size - 1) / block_size;
        size_t blocks_per_thread;
        size_t start_block, end_block;
        
        if (nbThreads >= 16 && num_blocks < nbThreads * 8) {
            // High thread count, small problem: use larger chunks per thread
            blocks_per_thread = (num_blocks + optimal_threads - 1) / optimal_threads;
            size_t effective_tid = tid % optimal_threads;
            start_block = effective_tid * blocks_per_thread;
            end_block = (effective_tid + 1) * blocks_per_thread;
            if (end_block > num_blocks) end_block = num_blocks;
            // Skip threads beyond optimal_threads to reduce memory bandwidth contention
            if (tid >= optimal_threads) {
                start_block = end_block;  // No work for this thread
            }
        } else {
            // Normal case: static distribution
            blocks_per_thread = (num_blocks + nbThreads - 1) / nbThreads;
            start_block = tid * blocks_per_thread;
            end_block = (tid + 1) * blocks_per_thread;
            if (end_block > num_blocks) end_block = num_blocks;
        }

        int prior = 0;
        int current = 0;
        int diff = 0;
        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *block_pointer = outputBytes_perthread;
        
        // Allocate temp arrays with size check and cache alignment
        size_t temp_arr_size = (block_size > 1) ? (block_size - 1) : 1;
        unsigned char *temp_sign_arr = NULL;
        unsigned int *temp_predict_arr = NULL;
        
        // Only process if main buffer was allocated successfully
        if (outputBytes_perthread) {
            // Use cache-aligned allocation for temp arrays too
            #if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
            if (posix_memalign((void**)&temp_sign_arr, 64, temp_arr_size * sizeof(unsigned char)) != 0) {
                temp_sign_arr = (unsigned char *)malloc(temp_arr_size * sizeof(unsigned char));
            }
            if (posix_memalign((void**)&temp_predict_arr, 64, temp_arr_size * sizeof(unsigned int)) != 0) {
                temp_predict_arr = (unsigned int *)malloc(temp_arr_size * sizeof(unsigned int));
            }
            #else
            temp_sign_arr = (unsigned char *)malloc(temp_arr_size * sizeof(unsigned char));
            temp_predict_arr = (unsigned int *)malloc(temp_arr_size * sizeof(unsigned int));
            #endif
        }
        
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
        
        // Only process if all allocations succeeded
        if (outputBytes_perthread && temp_sign_arr && temp_predict_arr) {
            // Process blocks assigned to this thread
            for (size_t block_idx = start_block; block_idx < end_block; block_idx++)
            {
                i = block_idx * block_size;
                size_t current_block_size = (i + block_size > extrema_count) ? (extrema_count - i) : block_size;
                if (current_block_size == 0 || i >= extrema_count) continue;

                max = 0;
                prior = sort_positions[i];
                memcpy(block_pointer, &prior, sizeof(int));
                block_pointer += sizeof(int);
                outSize_perthread += sizeof(int);

                if (current_block_size > 1)
                {
                    // No "omp simd" here: prior carries a dependence between
                    // iterations, which simd is not allowed to assume away.
                    for (j = 0; j < current_block_size - 1; j++)
                    {
                        current = sort_positions[i + j + 1];
                        diff = current - prior;
                        prior = current;
                        if (diff == 0)
                        {
                            temp_sign_arr[j] = 0;
                            temp_predict_arr[j] = 0;
                        }
                        else
                        {
                            if (diff < 0)
                            {
                                temp_sign_arr[j] = 1;
                                temp_predict_arr[j] = -diff;
                            }
                            else
                            {
                                temp_sign_arr[j] = 0;
                                temp_predict_arr[j] = diff;
                            }
                            if (max < temp_predict_arr[j])
                                max = temp_predict_arr[j];
                        }
                    }
                }

            if (max == 0) 
            {
                block_pointer[0] = 0;
                block_pointer++;
                outSize_perthread++;
            }
            else
            {
                bit_count = (int)(log2f((float)max)) + 1;
                block_pointer[0] = bit_count;
                outSize_perthread++;
                block_pointer++;
                signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, current_block_size - 1, block_pointer); 
                block_pointer += signbytelength;
                outSize_perthread += signbytelength;
                savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, current_block_size - 1, block_pointer, bit_count);
                block_pointer += savedbitsbytelength;
                outSize_perthread += savedbitsbytelength;
            }
        }
        } // End of if (outputBytes_perthread && temp_sign_arr && temp_predict_arr)

        if (outSize_perthread_arr) outSize_perthread_arr[tid].size = outSize_perthread;
#pragma omp barrier

        // Sequential prefix sum for offsets (small overhead, done once)
        #pragma omp single
        if (outSize_perthread_arr && offsets_perthread_arr)
        {
            offsets_perthread_arr[0] = 0;
            for (size_t k = 1; k < nbThreads; k++)
            {
                offsets_perthread_arr[k] = offsets_perthread_arr[k - 1] + outSize_perthread_arr[k - 1].size;
            }
            (*outSize) += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1].size;
            memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t)); szp_stream_write_tag(outputBytes, nbThreads);
        }
#pragma omp barrier
        
        // Copy thread-local results (no barrier needed after this)
        if (outputBytes_perthread && real_outputBytes && offsets_perthread_arr) {
            memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);
        }
        
        if (outputBytes_perthread) free(outputBytes_perthread);
        if (temp_sign_arr) free(temp_sign_arr);
        if (temp_predict_arr) free(temp_predict_arr);
        // Every thread must finish its memcpy (which reads offsets_perthread_arr)
        // before one thread frees the shared arrays. "omp single" has no
        // barrier on entry, so add one.
#pragma omp barrier
#pragma omp single
        {
            if (outSize_perthread_arr) free(outSize_perthread_arr);
            if (offsets_perthread_arr) free(offsets_perthread_arr);
        }
    }
    
    free(sort_positions);
    return output;
    
#else
    // Fallback to sequential version if OpenMP not available
    if (!critical_points || critical_count == 0) {
        *outSize = 0;
        return NULL;
    }
    
    // Count maxima and minima only (types 1 and 2)
    size_t extrema_count = 0;
    for (size_t i = 0; i < critical_count; i++) {
        if (critical_points[i].type == 1 || critical_points[i].type == 2) {
            extrema_count++;
        }
    }
    
    if (extrema_count == 0) {
        *outSize = 0;
        return NULL;
    }
    
    // Extract sort_position values for extrema only
    int *sort_positions = (int *)malloc(extrema_count * sizeof(int));
    if (!sort_positions) {
        *outSize = 0;
        return NULL;
    }
    
    // Copy sort_position values for extrema only
    size_t extrema_idx = 0;
    for (size_t i = 0; i < critical_count; i++) {
        if (critical_points[i].type == 1 || critical_points[i].type == 2) {
            sort_positions[extrema_idx] = critical_points[i].sort_position;
            extrema_idx++;
        }
    }
    
    // Allocate output buffer
    size_t maxPreservedBufferSize = sizeof(int) * extrema_count + sizeof(size_t);
    unsigned char *output = (unsigned char *)malloc(maxPreservedBufferSize);
    if (!output) {
        free(sort_positions);
        *outSize = 0;
        return NULL;
    }
    
    // Write extrema_count as header
    memcpy(output, &extrema_count, sizeof(size_t));
    unsigned char *outputBytes = output + sizeof(size_t);
    
    size_t outSize_perthread = 0;
    unsigned char *block_pointer = outputBytes;
    
    unsigned char *temp_sign_arr = (unsigned char *)malloc((blockSize - 1) * sizeof(unsigned char));
    unsigned int *temp_predict_arr = (unsigned int *)malloc((blockSize - 1) * sizeof(unsigned int));
    unsigned int signbytelength = 0; 
    unsigned int savedbitsbytelength = 0;
    
    for (size_t i = 0; i < extrema_count; i = i + blockSize)
    {
        size_t current_block_size = (i + blockSize > extrema_count) ? (extrema_count - i) : blockSize;
        if (current_block_size == 0) continue;

        unsigned int max = 0;
        int prior = sort_positions[i];
        memcpy(block_pointer, &prior, sizeof(int));
        block_pointer += sizeof(int);
        outSize_perthread += sizeof(int);

        if (current_block_size > 1)
        {
            for (size_t j = 0; j < current_block_size - 1; j++)
            {
                int current = sort_positions[i + j + 1];
                int diff = current - prior;
                prior = current;
                if (diff == 0)
                {
                    temp_sign_arr[j] = 0;
                    temp_predict_arr[j] = 0;
                }
                else
                {
                    if (diff < 0)
                    {
                        temp_sign_arr[j] = 1;
                        temp_predict_arr[j] = -diff;
                    }
                    else
                    {
                        temp_sign_arr[j] = 0;
                        temp_predict_arr[j] = diff;
                    }
                    if (max < temp_predict_arr[j])
                        max = temp_predict_arr[j];
                }
            }
        }

        if (max == 0) 
        {
            block_pointer[0] = 0;
            block_pointer++;
            outSize_perthread++;
        }
        else
        {
            unsigned int bit_count = (int)(log2f((float)max)) + 1;
            block_pointer[0] = bit_count;
            outSize_perthread++;
            block_pointer++;
            signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, current_block_size - 1, block_pointer); 
            block_pointer += signbytelength;
            outSize_perthread += signbytelength;
            savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, current_block_size - 1, block_pointer, bit_count);
            block_pointer += savedbitsbytelength;
            outSize_perthread += savedbitsbytelength;
        }
    }
    
    *outSize = sizeof(size_t) + outSize_perthread;
    
    free(sort_positions);
    free(temp_sign_arr);
    free(temp_predict_arr);
    return output;
#endif
}

unsigned char *
szp_float_openmp_threadblock_randomaccess_topology_preserved(float *oriData, size_t *outSize, float absErrBound,
                                                             size_t nbEle, int blockSize,
                                                             CriticalPoint *critical_points, int critical_count,
                                                             int rows, int cols)
{
#ifdef _OPENMP
    if (szp_check_errbound<float>(oriData, nbEle, (double)absErrBound, "TopoSZp compress (2D)") < 0) {
        *outSize = 0;
        return NULL;
    }
    if (absErrBound <= 0.0) return NULL;

    float *op = oriData;

    // Calculate buffer size with overflow protection
    // Check for potential overflow: 8 * nbEle + 1024
    size_t maxPreservedBufferSize = 0;
    const size_t max_safe_nbEle = (SIZE_MAX - 1024) / 8;
    if (nbEle > 0 && nbEle <= max_safe_nbEle) {
        maxPreservedBufferSize = 8ull * nbEle + 1024ull +
                                 (size_t)omp_get_max_threads() * sizeof(size_t); // offset table
    } else if (nbEle > max_safe_nbEle) {
        // Fallback for very large arrays - use a reasonable maximum
        maxPreservedBufferSize = (SIZE_MAX / 2 > 1024) ? (SIZE_MAX / 2) : 1024;
    } else {
        // nbEle is 0 or invalid
        maxPreservedBufferSize = 1024;
    }
    // Ensure minimum size
    if (maxPreservedBufferSize < 1024) {
        maxPreservedBufferSize = 1024;
    }
    size_t maxPreservedBufferSize_perthread = 0;
    unsigned char *outputBytes = (unsigned char *)malloc(maxPreservedBufferSize);
    if (!outputBytes) return NULL;
    unsigned char *real_outputBytes = NULL;
    // Use padded structure to avoid false sharing (64-byte cache line alignment)
    struct PaddedSize {
        size_t size;
        char padding[64 - sizeof(size_t)];  // Pad to 64-byte cache line boundary
    };
    struct PaddedSize *outSize_perthread_arr = NULL;
    size_t *offsets_perthread_arr = NULL;

    *outSize = 0;

    unsigned char *critical_type = (unsigned char *)calloc(nbEle, 1);
    if (!critical_type) { free(outputBytes); return NULL; }

    // Parallelize critical point marking for better scaling
    size_t unique_marked = 0;
    #pragma omp parallel
    {
        size_t local_unique = 0;
        #pragma omp for nowait
        for (int i = 0; i < critical_count; i++) {
            int x = critical_points[i].x, y = critical_points[i].y;
            if (x < 0 || x >= rows || y < 0 || y >= cols) continue;
            size_t flat = (size_t)x * (size_t)cols + (size_t)y;
            unsigned char t = (unsigned char)critical_points[i].type;
            if (t >= 1 && t <= 3) {
                if (critical_type[flat] == 0) {
                    #pragma omp atomic
                    local_unique++;
                }
                critical_type[flat] = t;
            }
        }
        #pragma omp atomic
        unique_marked += local_unique;
    }

    unsigned int nbThreads = 0;
    double inver_bound = 1.0 / absErrBound;
    unsigned int threadblocksize = 0;
    unsigned int remainder = 0;
    unsigned int block_size = (unsigned int)blockSize;
    if (block_size == 0) { free(outputBytes); free(critical_type); return NULL; }
    unsigned int new_block_size = (block_size > 1) ? (block_size - 1) : 1;
    unsigned int num_full_block_in_tb = 0;
    unsigned int num_remainder_in_tb = 0;

    size_t g_type_total = 0, g_t1 = 0, g_t2 = 0, g_t3 = 0;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            if (nbThreads == 0) nbThreads = 1; // Safety check
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            *outSize += nbThreads * sizeof(size_t);
            // Allocate padded structures to avoid false sharing
            outSize_perthread_arr = (struct PaddedSize *)malloc(nbThreads * sizeof(struct PaddedSize));
            offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));
            if (!outSize_perthread_arr || !offsets_perthread_arr) {
                if (outSize_perthread_arr) free(outSize_perthread_arr);
                if (offsets_perthread_arr) free(offsets_perthread_arr);
                outSize_perthread_arr = NULL;
                offsets_perthread_arr = NULL;
            }

            // Calculate per-thread buffer size with safety checks
            size_t header_size = nbThreads * sizeof(size_t);
            // Ensure we have enough space for header
            if (maxPreservedBufferSize <= header_size) {
                // Not enough space, use minimum
                maxPreservedBufferSize_perthread = 1024;
            } else if (nbThreads > 0) {
                size_t available_size = maxPreservedBufferSize - header_size;
                // Check for integer overflow in division
                if (available_size >= nbThreads) {
                    maxPreservedBufferSize_perthread = available_size / nbThreads;
                } else {
                    maxPreservedBufferSize_perthread = 1024; // Fallback
                }
            } else {
                maxPreservedBufferSize_perthread = 1024; // Minimum safe size
            }
            // Ensure minimum buffer size - critical check
            if (maxPreservedBufferSize_perthread == 0 || maxPreservedBufferSize_perthread < 1024) {
                maxPreservedBufferSize_perthread = 1024;
            }
            // Additional safety: ensure it's not unreasonably large (prevent overflow)
            const size_t max_safe_perthread = SIZE_MAX / 4;
            if (maxPreservedBufferSize_perthread > max_safe_perthread) {
                maxPreservedBufferSize_perthread = max_safe_perthread;
            }
            threadblocksize = (unsigned int)(nbEle / nbThreads);
            remainder = (unsigned int)(nbEle % nbThreads);
            num_full_block_in_tb = (threadblocksize) / block_size;
            num_remainder_in_tb = (threadblocksize) % block_size;
        }

        size_t i = 0, j = 0;
        unsigned char *outputBytes_perthread = NULL;
        // Use cache-aligned allocation for better performance
        if (maxPreservedBufferSize_perthread > 0 && maxPreservedBufferSize_perthread < SIZE_MAX) {
            #if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
            if (posix_memalign((void**)&outputBytes_perthread, 64, maxPreservedBufferSize_perthread) != 0) {
                outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
            }
            #else
            outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
            #endif
        } else {
            // Fallback to safe minimum if size is invalid
            outputBytes_perthread = (unsigned char *)malloc(1024);
        }
        size_t outSize_perthread = 0;

        int tid = omp_get_thread_num();
        
        // Use block-based distribution for better cache locality and load balancing
        size_t num_blocks = (nbEle + block_size - 1) / block_size;
        size_t blocks_per_thread;
        size_t start_block, end_block;
        
        // Adaptive thread limiting for high thread counts
        int optimal_threads = nbThreads;
        if (nbThreads >= 16 && num_blocks < nbThreads * 8) {
            optimal_threads = (num_blocks + 7) / 8;
            if (optimal_threads < 1) optimal_threads = 1;
        }
        
        if (nbThreads >= 16 && num_blocks < nbThreads * 8) {
            // High thread count, small problem: use larger chunks per thread
            blocks_per_thread = (num_blocks + optimal_threads - 1) / optimal_threads;
            size_t effective_tid = tid % optimal_threads;
            start_block = effective_tid * blocks_per_thread;
            end_block = (effective_tid + 1) * blocks_per_thread;
            if (end_block > num_blocks) end_block = num_blocks;
            // Skip threads beyond optimal_threads
            if (tid >= optimal_threads) {
                start_block = end_block;
            }
        } else {
            // Normal case: static block distribution
            blocks_per_thread = (num_blocks + nbThreads - 1) / nbThreads;
            start_block = tid * blocks_per_thread;
            end_block = (tid + 1) * blocks_per_thread;
            if (end_block > num_blocks) end_block = num_blocks;
        }
        
        size_t lo = start_block * block_size;
        size_t hi = end_block * block_size;
        if (hi > nbEle) hi = nbEle;

        int prior = 0, current = 0, diff = 0;
        unsigned int maxv = 0, bit_count = 0;
        unsigned char *block_pointer = outputBytes_perthread;

        // Allocate temp arrays with size validation
        size_t temp_sign_size = new_block_size * sizeof(unsigned char);
        size_t temp_type_size = block_size * sizeof(unsigned char);
        size_t temp_predict_size = new_block_size * sizeof(unsigned int);
        
        unsigned char *temp_sign_arr = NULL;
        unsigned char *temp_type_arr = NULL;
        unsigned int *temp_predict_arr = NULL;
        
        if (temp_sign_size > 0) temp_sign_arr = (unsigned char *)malloc(temp_sign_size);
        if (temp_type_size > 0) temp_type_arr = (unsigned char *)malloc(temp_type_size);
        if (temp_predict_size > 0) temp_predict_arr = (unsigned int *)malloc(temp_predict_size);
        unsigned int signbytelength = 0, savedbitsbytelength = 0, typebytelength = 0;

        size_t l_total = 0, l1 = 0, l2 = 0, l3 = 0;

        if (outputBytes_perthread && temp_sign_arr && temp_type_arr && temp_predict_arr) {
            // Process blocks assigned to this thread using block-based distribution
            for (size_t block_idx = start_block; block_idx < end_block; block_idx++) {
                i = block_idx * block_size;
                if (i >= nbEle) break;
                
                size_t current_block_size = (i + block_size > nbEle) ? (nbEle - i) : block_size;
                if (current_block_size == 0) continue;
                
                maxv = 0;
                prior = szp_quantize<float>(op[i], (double)absErrBound, inver_bound);
                memcpy(block_pointer, &prior, sizeof(int));
                block_pointer += sizeof(int);
                outSize_perthread += sizeof(int);

                if (current_block_size > 1) {
                    unsigned int actual_new_block_size = current_block_size - 1;
                    // No "omp simd": prior carries a dependence between iterations.
                    for (j = 0; j < actual_new_block_size; j++) {
                        current = szp_quantize<float>(op[i + j + 1], (double)absErrBound, inver_bound);
                        diff = current - prior;
                        prior = current;
                        if (diff == 0) { temp_sign_arr[j] = 0; temp_predict_arr[j] = 0; }
                        else if (diff > 0) { temp_sign_arr[j] = 0; if ((unsigned)diff > maxv) maxv = (unsigned)diff; temp_predict_arr[j] = (unsigned)diff; }
                        else { temp_sign_arr[j] = 1; unsigned int ad = (unsigned)(-diff); if (ad > maxv) maxv = ad; temp_predict_arr[j] = ad; }
                    }
                }

                for (j = 0; j < current_block_size; j++) {
                    size_t idx = i + j;
                    unsigned char tt = (idx < nbEle) ? critical_type[idx] : 0;
                    // Only store type information (1,2,3), not quantized_bin
                    temp_type_arr[j] = tt;
                    if (tt) { l_total++; if (tt==1) l1++; else if (tt==2) l2++; else l3++; }
                }

                if (maxv == 0) {
                    *block_pointer++ = 0;
                    outSize_perthread++;
                } else {
                    bit_count = (unsigned int)floorf(log2f((float)maxv)) + 1u;
                    *block_pointer++ = (unsigned char)bit_count;
                    outSize_perthread++;
                    unsigned int actual_new_block_size = (current_block_size > 1) ? (current_block_size - 1) : 0;
                    if (actual_new_block_size > 0) {
                        signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, actual_new_block_size, block_pointer);
                        block_pointer += signbytelength; outSize_perthread += signbytelength;
                        savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, actual_new_block_size, block_pointer, bit_count);
                        block_pointer += savedbitsbytelength; outSize_perthread += savedbitsbytelength;
                    }
                }

                // 2-bit types (1,2,3; 0 = regular), 4 per byte, first element
                // in the high bits: same bytes as convertIntArray2ByteArray_fast_2b,
                // packed in place instead of allocating a buffer per block.
                typebytelength = (unsigned int)((2 * current_block_size + 7) / 8);
                memset(block_pointer, 0, typebytelength);
                for (j = 0; j < current_block_size; j++)
                    block_pointer[j >> 2] |= (unsigned char)((temp_type_arr[j] & 0x3) << (6 - 2 * (j & 3)));
                block_pointer += typebytelength; outSize_perthread += typebytelength;
            }
        }

        if (outSize_perthread_arr) outSize_perthread_arr[tid].size = outSize_perthread;

#pragma omp atomic
        g_type_total += l_total;
#pragma omp atomic
        g_t1 += l1;
#pragma omp atomic
        g_t2 += l2;
#pragma omp atomic
        g_t3 += l3;

#pragma omp barrier

#pragma omp single
        {
            if (outSize_perthread_arr && offsets_perthread_arr) {
                offsets_perthread_arr[0] = 0;
                for (i = 1; i < nbThreads; i++)
                    offsets_perthread_arr[i] = offsets_perthread_arr[i - 1] + outSize_perthread_arr[i - 1].size;

                *outSize += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1].size;
                memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t)); szp_stream_write_tag(outputBytes, nbThreads);
            }
        }
#pragma omp barrier
        if (outSize_perthread_arr && outputBytes_perthread && real_outputBytes && offsets_perthread_arr) {
            memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);
        }
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
    return NULL;
#endif
}


int *
szp_float_openmp_direct_predict_quantization(float *oriData, size_t *outSize, float absErrBound,
                                             size_t nbEle, int blockSize)
{
#ifdef _OPENMP

    float *op = oriData;

    size_t i = 0;


    int *quti_arr = (int *)malloc(nbEle * sizeof(int));
    int *diff_arr = (int *)malloc(nbEle * sizeof(int));
    (*outSize) = 0;


    int nbThreads = 1;
    double inver_bound = 1;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            
            inver_bound = 1.0 / absErrBound;
            
        }

     
#pragma omp for schedule(static)
        for (i = 0; i < nbEle; i++)
        {
            quti_arr[i] = (op[i] + absErrBound) * inver_bound;
        }

#pragma omp single
        {
            diff_arr[0] = quti_arr[0];
        }

#pragma omp for schedule(static)
        for (i = 1; i < nbEle; i++)
        {
            diff_arr[i] = quti_arr[i] - quti_arr[i - 1];
        }
    }

    free(quti_arr);

    return diff_arr;
#else
    return NULL;
#endif
}

int *
szp_float_openmp_threadblock_predict_quantization(float *oriData, size_t *outSize, float absErrBound,
                                                  size_t nbEle, int blockSize)
{
#ifdef _OPENMP
    
    float *op = oriData;

  
    int *diff_arr = (int *)malloc(nbEle * sizeof(int));
    (*outSize) = 0;
    

    int nbThreads = 1;
    double inver_bound = 1;
    int threadblocksize = 1;
    int remainder = 1;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            
            inver_bound = 1.0 / absErrBound;
            threadblocksize = nbEle / nbThreads;
            remainder = nbEle % nbThreads;
            
        }
        size_t i = 0;
    
        int tid = omp_get_thread_num();
        int lo = tid * threadblocksize;
        int hi = (tid + 1) * threadblocksize;
        int prior = 0;
        int current = 0;
        prior = (op[lo] + absErrBound) * inver_bound;
        diff_arr[lo] = prior;
        for (i = lo + 1; i < hi; i++)
        {
            current = (op[i] + absErrBound) * inver_bound;
            diff_arr[i] = current - prior;
            prior = current;
        }
#pragma omp single
        {
            if (remainder != 0)
            {
                size_t remainder_lo = nbEle - remainder;
                prior = (op[remainder_lo] + absErrBound) * inver_bound;
                diff_arr[remainder_lo] = prior;
                for (i = nbEle - remainder + 1; i < nbEle; i++)
                {
                    current = (op[i] + absErrBound) * inver_bound;
                    diff_arr[i] = current - prior;
                    prior = current;
                }
            }
          
        }
    }

    return diff_arr;
#else
    return NULL;
#endif
}

unsigned char *
szp_float_openmp_threadblock(float *oriData, size_t *outSize, float absErrBound,
                             size_t nbEle, int blockSize)
{
#ifdef _OPENMP
    
    float *op = oriData;

    size_t maxPreservedBufferSize = sizeof(float) * nbEle + sizeof(float);
    size_t maxPreservedBufferSize_perthread = 0;

    unsigned char *real_outputBytes; 
    // Use padded structure to avoid false sharing (64-byte cache line alignment)
    struct PaddedSize {
        size_t size;
        char padding[64 - sizeof(size_t)];  // Pad to 64-byte cache line boundary
    };
    struct PaddedSize *outSize_perthread_arr;
    size_t *offsets_perthread_arr;

    unsigned char *output = (unsigned char *)malloc(maxPreservedBufferSize);
    floatToBytes(output, absErrBound); 
    unsigned char *outputBytes = output + sizeof(float); // skip the first buffer for absErrBound
   
    (*outSize) = 0;
  

    unsigned int nbThreads = 0;
    double inver_bound = 0;
    unsigned int threadblocksize = 0;
    unsigned int block_size = blockSize;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            (*outSize) += nbThreads * sizeof(size_t); 
            // Allocate padded structures to avoid false sharing
            outSize_perthread_arr = (struct PaddedSize *)malloc(nbThreads * sizeof(struct PaddedSize));
            offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));

            inver_bound = 1.0 / absErrBound;
            threadblocksize = nbEle / nbThreads;
        }
        size_t i = 0;
        size_t j = 0;
        maxPreservedBufferSize_perthread = (sizeof(float) * nbEle + nbThreads - 1) / nbThreads;
        unsigned char *outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
        size_t outSize_perthread = 0;
        
        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (tid == nbThreads - 1) {
            hi = nbEle; // Ensure the last thread processes all remaining elements
        }

        int prior = 0;
        int current = 0;
        int diff = 0;
        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *block_pointer = outputBytes_perthread;
        
        if (lo < hi) { // Ensure thread has data to process
            prior = szp_quantize<float>(op[lo], (double)absErrBound, inver_bound);
            memcpy(block_pointer, &prior, sizeof(int));
            block_pointer += sizeof(unsigned int);
            outSize_perthread += sizeof(unsigned int);
        }
        
        unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
       
        for (i = lo + 1; i < hi; i = i + block_size)
        {
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            max = 0;
            for (j = 0; j < current_block_size; j++)
            {
                current = szp_quantize<float>(op[i + j], (double)absErrBound, inver_bound);
                diff = current - prior;
                prior = current;
                if (diff == 0)
                {
                    temp_sign_arr[j] = 0;
                    temp_predict_arr[j] = 0;
                }
                else
                {
                    if (diff < 0)
                    {
                        temp_sign_arr[j] = 1;
                        temp_predict_arr[j] = -diff;
                    }
                    else
                    {
                        temp_sign_arr[j] = 0;
                        temp_predict_arr[j] = diff;
                    }
                    if (max < temp_predict_arr[j])
                        max = temp_predict_arr[j];
                }
            }

            if (max == 0) 
            {
                block_pointer[0] = 0;
                block_pointer++;
                outSize_perthread++;
            }
            else
            {
                bit_count = (int)(log2f(max)) + 1;
                block_pointer[0] = bit_count;
                
                outSize_perthread++;
                block_pointer++;
                signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, current_block_size, block_pointer); 
                block_pointer += signbytelength;
                outSize_perthread += signbytelength;
                
                savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, current_block_size, block_pointer, bit_count);
                
                block_pointer += savedbitsbytelength;
                outSize_perthread += savedbitsbytelength;
            }
        }

        outSize_perthread_arr[tid].size = outSize_perthread;
#pragma omp barrier

#pragma omp single
        {
            offsets_perthread_arr[0] = 0;
            for (i = 1; i < nbThreads; i++)
            {
                offsets_perthread_arr[i] = offsets_perthread_arr[i - 1] + outSize_perthread_arr[i - 1].size;
            }
            (*outSize) += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1].size;
            memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t)); szp_stream_write_tag(outputBytes, nbThreads);
        }
#pragma omp barrier
        memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);
        
        free(outputBytes_perthread);
        free(temp_sign_arr);
        free(temp_predict_arr);
#pragma omp barrier
#pragma omp single
        {
            free(outSize_perthread_arr);
            free(offsets_perthread_arr);
        }
    }
    (*outSize) += sizeof(float);
    return output;
#else
    printf("Error! OpenMP not supported!\n");
    return NULL;
#endif
}

/**
 * output: the first 4 bytes are used to store absErrorBound, then followed by compressed data bytes.
 * */
void szp_float_openmp_threadblock_arg(unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
                                      size_t nbEle, int blockSize)
{
#ifdef _OPENMP
    
    float *op = oriData;

    
    size_t maxPreservedBufferSize = sizeof(float) + sizeof(float) * nbEle; 
    size_t maxPreservedBufferSize_perthread = 0;
    
    unsigned char *real_outputBytes; 
    // Use padded structure to avoid false sharing (64-byte cache line alignment)
    struct PaddedSize {
        size_t size;
        char padding[64 - sizeof(size_t)];  // Pad to 64-byte cache line boundary
    };
    struct PaddedSize *outSize_perthread_arr;
    size_t *offsets_perthread_arr;
    
	unsigned char* outputBytes = output + sizeof(float);
	floatToBytes(output, absErrBound);

    (*outSize) = 0;


    unsigned int nbThreads = 0;
    double inver_bound = 0;
    unsigned int threadblocksize = 0;
    unsigned int block_size = blockSize;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            (*outSize) += nbThreads * sizeof(size_t); 
            // Allocate padded structures to avoid false sharing
            outSize_perthread_arr = (struct PaddedSize *)malloc(nbThreads * sizeof(struct PaddedSize));
            offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));

            inver_bound = 1.0 / absErrBound;
            threadblocksize = nbEle / nbThreads;
        }
        size_t i = 0;
        size_t j = 0;
        maxPreservedBufferSize_perthread = (sizeof(float) * nbEle + nbThreads - 1) / nbThreads;
        unsigned char *outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
        size_t outSize_perthread = 0;
        
        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (tid == nbThreads - 1) {
            hi = nbEle; // Ensure the last thread processes all remaining elements
        }

        int prior = 0;
        int current = 0;
        int diff = 0;
        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *block_pointer = outputBytes_perthread;
        
        if (lo < hi) { // Ensure thread has data to process
            prior = szp_quantize<float>(op[lo], (double)absErrBound, inver_bound);
            memcpy(block_pointer, &prior, sizeof(int));
            block_pointer += sizeof(unsigned int);
            outSize_perthread += sizeof(unsigned int);
        } // if nbThreads > nbEle, threadblocksize=0, causing no data to be written
        
        unsigned char *temp_sign_arr = (unsigned char *)malloc(block_size * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc(block_size * sizeof(unsigned int));
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
        
        for (i = lo + 1; i < hi; i = i + block_size)
        {
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            max = 0;
            for (j = 0; j < current_block_size; j++)
            {
                current = szp_quantize<float>(op[i + j], (double)absErrBound, inver_bound);
                diff = current - prior;
                prior = current;
                if (diff == 0)
                {
                    temp_sign_arr[j] = 0;
                    temp_predict_arr[j] = 0;
                }
                else
                {
                    if (diff < 0)
                    {
                        temp_sign_arr[j] = 1;
                        temp_predict_arr[j] = -diff;
                    }
                    else
                    {
                        temp_sign_arr[j] = 0;
                        temp_predict_arr[j] = diff;
                    }
                    if (max < temp_predict_arr[j])
                        max = temp_predict_arr[j];
                }
            }

            if (max == 0) 
            {
                block_pointer[0] = 0;
                block_pointer++;
                outSize_perthread++;
            }
            else
            {
                bit_count = (int)(log2f(max)) + 1;
                block_pointer[0] = bit_count;
                
                outSize_perthread++;
                block_pointer++;
                signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, current_block_size, block_pointer); 
                block_pointer += signbytelength;
                outSize_perthread += signbytelength;
                
                savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, current_block_size, block_pointer, bit_count);
                
                block_pointer += savedbitsbytelength;
                outSize_perthread += savedbitsbytelength;
            }
        }

        outSize_perthread_arr[tid].size = outSize_perthread;
#pragma omp barrier

#pragma omp single
        {
            offsets_perthread_arr[0] = 0;
            for (i = 1; i < nbThreads; i++)
            {
                offsets_perthread_arr[i] = offsets_perthread_arr[i - 1] + outSize_perthread_arr[i - 1].size;
                
            }
            (*outSize) += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1].size;
            memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t)); szp_stream_write_tag(outputBytes, nbThreads);
            
        }
#pragma omp barrier
        memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);
        
        free(outputBytes_perthread);
        free(temp_sign_arr);
        free(temp_predict_arr);
#pragma omp barrier
#pragma omp single
        {
            
            free(outSize_perthread_arr);
            free(offsets_perthread_arr);
        }

       
    }
    
    (*outSize) += sizeof(float);

    
#else
    printf("Error! OpenMP not supported!\n");
#endif
}

void szp_float_single_thread_arg(unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
                                 size_t nbEle, int blockSize)
{

    float *op = oriData;

    unsigned char* outputBytes = output + sizeof(float);
    floatToBytes(output, absErrBound);
    
    unsigned char *real_outputBytes; 
    size_t *outSize_perthread_arr;
    size_t *offsets_perthread_arr;

    (*outSize) = 0;

    double inver_bound = 1.0 / absErrBound;
    unsigned int block_size = blockSize;

    int nbThreads = 1;
    real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
    (*outSize) += nbThreads * sizeof(size_t); 
    outSize_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));
    offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));

    size_t maxPreservedBufferSize_perthread = sizeof(float) * nbEle;
    unsigned char *outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
    size_t outSize_perthread = 0;
    
    int tid = 0;
    size_t lo = 0;
    size_t hi = nbEle;

    int prior = 0;
    int current = 0;
    int diff = 0;
    unsigned int max = 0;
    unsigned int bit_count = 0;
    unsigned char *block_pointer = outputBytes_perthread;
    
    if (lo < hi) {
        prior = szp_quantize<float>(op[lo], (double)absErrBound, inver_bound);
        memcpy(block_pointer, &prior, sizeof(int));
        block_pointer += sizeof(unsigned int);
        outSize_perthread += sizeof(unsigned int);
    }
    
    unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned int signbytelength = 0; 
    unsigned int savedbitsbytelength = 0;
    
    for (size_t i = lo + 1; i < hi; i = i + block_size)
    {
        size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
        if (current_block_size == 0) continue;

        max = 0;
        for (size_t j = 0; j < current_block_size; j++)
        {
            current = szp_quantize<float>(op[i + j], (double)absErrBound, inver_bound);
            diff = current - prior;
            prior = current;
            if (diff == 0)
            {
                temp_sign_arr[j] = 0;
                temp_predict_arr[j] = 0;
            }
            else
            {
                if (diff < 0)
                {
                    temp_sign_arr[j] = 1;
                    temp_predict_arr[j] = -diff;
                }
                else
                {
                    temp_sign_arr[j] = 0;
                    temp_predict_arr[j] = diff;
                }
                if (max < temp_predict_arr[j])
                    max = temp_predict_arr[j];
            }
        }
        if (max == 0) 
        {
            block_pointer[0] = 0;
            block_pointer++;
            outSize_perthread++;
        }
        else
        {
            bit_count = (int)(log2f(max)) + 1;
            block_pointer[0] = bit_count;
            
            outSize_perthread++;
            block_pointer++;
            signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, current_block_size, block_pointer); 
            block_pointer += signbytelength;
            outSize_perthread += signbytelength;
            
            savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, current_block_size, block_pointer, bit_count);
            
            block_pointer += savedbitsbytelength;
            outSize_perthread += savedbitsbytelength;
        }
    }

    outSize_perthread_arr[tid] = outSize_perthread;
    offsets_perthread_arr[0] = 0;

    (*outSize) += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1];
    memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t)); szp_stream_write_tag(outputBytes, nbThreads);
    
    memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);
    
    free(outputBytes_perthread);
    free(temp_sign_arr);
    free(temp_predict_arr);
    
    free(outSize_perthread_arr);
    free(offsets_perthread_arr);
    
    (*outSize) += sizeof(float);
}

size_t szp_float_single_thread_arg_record(unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
                                       size_t nbEle, int blockSize)
{

    size_t total_memaccess = 0;
    float *op = oriData;
    
    unsigned char* outputBytes = output + sizeof(float);
    floatToBytes(output, absErrBound);
    
    unsigned char *real_outputBytes; 
    size_t *outSize_perthread_arr;
    size_t *offsets_perthread_arr;

    (*outSize) = 0;
    total_memaccess += sizeof(size_t);

    double inver_bound = 1.0 / absErrBound;
    unsigned int block_size = blockSize;

    int nbThreads = 1;
    real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
    (*outSize) += nbThreads * sizeof(size_t); 
   
    outSize_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));
    offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));

    size_t maxPreservedBufferSize_perthread = sizeof(float) * nbEle;
    unsigned char *outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
    size_t outSize_perthread = 0;
    
    int tid = 0;
    size_t lo = 0;
    size_t hi = nbEle;

    int prior = 0;
    int current = 0;
    int diff = 0;
    unsigned int max = 0;
    unsigned int bit_count = 0;
    unsigned char *block_pointer = outputBytes_perthread;

    if (lo < hi) {
        prior = szp_quantize<float>(op[lo], (double)absErrBound, inver_bound);
        total_memaccess += sizeof(float);
        memcpy(block_pointer, &prior, sizeof(int));
        total_memaccess += sizeof(int) * 2; // read and write
        block_pointer += sizeof(unsigned int);
        outSize_perthread += sizeof(unsigned int);
    }
    
    unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned int signbytelength = 0; 
    unsigned int savedbitsbytelength = 0;
    
    for (size_t i = lo + 1; i < hi; i = i + block_size)
    {
        size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
        if (current_block_size == 0) continue;

        max = 0;
        for (size_t j = 0; j < current_block_size; j++)
        {
            current = szp_quantize<float>(op[i + j], (double)absErrBound, inver_bound);
            total_memaccess += sizeof(float);
            diff = current - prior;
            prior = current;
            if (diff == 0)
            {
                temp_sign_arr[j] = 0;
                temp_predict_arr[j] = 0;
            }
            else
            {
                if (diff < 0)
                {
                    temp_sign_arr[j] = 1;
                    temp_predict_arr[j] = -diff;
                }
                else
                {
                    temp_sign_arr[j] = 0;
                    temp_predict_arr[j] = diff;
                }
                if (max < temp_predict_arr[j])
                    max = temp_predict_arr[j];
            }
            total_memaccess += sizeof(unsigned char) + sizeof(unsigned int); // for temp arrays
        }
        if (max == 0) 
        {
            block_pointer[0] = 0;
            total_memaccess += sizeof(unsigned char);
            block_pointer++;
            outSize_perthread++;
        }
        else
        {
            bit_count = (int)(log2f(max)) + 1;
            block_pointer[0] = bit_count;
            total_memaccess += sizeof(unsigned char);
            
            outSize_perthread++;
            block_pointer++;
            signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, current_block_size, block_pointer); 
            total_memaccess += sizeof(unsigned char) * current_block_size; // read temp_sign_arr
            block_pointer += signbytelength;
            total_memaccess += sizeof(unsigned char) * signbytelength; // write to block_pointer
            outSize_perthread += signbytelength;
            
            savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, current_block_size, block_pointer, bit_count);
            total_memaccess += sizeof(unsigned int) * current_block_size; // read temp_predict_arr
            
            block_pointer += savedbitsbytelength;
            total_memaccess += sizeof(unsigned char) * savedbitsbytelength; // write to block_pointer
            outSize_perthread += savedbitsbytelength;
        }
    }

    outSize_perthread_arr[tid] = outSize_perthread;
    total_memaccess += sizeof(size_t);

    offsets_perthread_arr[0] = 0;
    total_memaccess += sizeof(size_t);

    (*outSize) += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1];
    total_memaccess += (sizeof(size_t) * 3);
    memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t)); szp_stream_write_tag(outputBytes, nbThreads);
    total_memaccess += (sizeof(unsigned char) * nbThreads * sizeof(size_t)) * 2; // read and write
    
    memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);
    total_memaccess += (sizeof(unsigned char) * outSize_perthread) * 2; // read and write
    
    free(outputBytes_perthread);
    free(temp_sign_arr);
    free(temp_predict_arr);
    
    free(outSize_perthread_arr);
    free(offsets_perthread_arr);
    
    (*outSize) += sizeof(float);
    return total_memaccess;
}

unsigned char *
szp_float_openmp_threadblock_randomaccess(float *oriData, size_t *outSize, float absErrBound,
                                          size_t nbEle, int blockSize)
{
#ifdef _OPENMP
    if (szp_check_errbound<float>(oriData, nbEle, (double)absErrBound, "SZp compress (float)") < 0) {
        *outSize = 0;
        return NULL;
    }
    
    float *op = oriData;

    size_t maxPreservedBufferSize = sizeof(float) * nbEle + sizeof(float);
    unsigned char *output = (unsigned char *)malloc(maxPreservedBufferSize);
    floatToBytes(output, absErrBound);
    
    unsigned char* outputBytes = output + sizeof(float);
    
    size_t maxPreservedBufferSize_perthread = 0;
    unsigned char *real_outputBytes; 
    // Use padded structure to avoid false sharing (64-byte cache line alignment)
    struct PaddedSize {
        size_t size;
        char padding[64 - sizeof(size_t)];  // Pad to 64-byte cache line boundary
    };
    struct PaddedSize *outSize_perthread_arr;
    size_t *offsets_perthread_arr;
    
    (*outSize) = 0;
    
    unsigned int nbThreads = 0;
    double inver_bound = 0;
    unsigned int threadblocksize = 0;
    unsigned int block_size = blockSize;
    
    // Adaptive thread count: for small problems, reduce thread count to avoid overhead
    size_t num_blocks = (nbEle + block_size - 1) / block_size;
    int optimal_threads = nbThreads;
    
    // Set thread affinity for better NUMA performance (especially important for 32 threads)
    // This should be set before the parallel region via environment variables:
    // export OMP_PROC_BIND=close
    // export OMP_PLACES=cores
    // But we can also hint the scheduler here

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            // For high thread counts (>=16), reduce effective parallelism if problem is too small
            // This prevents memory bandwidth saturation and cache thrashing
            if (nbThreads >= 16 && num_blocks < nbThreads * 8) {
                // Problem too small for this many threads - will process in larger chunks
                // This reduces scheduling overhead and improves cache locality
                optimal_threads = (num_blocks + 7) / 8;  // Aim for ~8 blocks per thread minimum
                if (optimal_threads < 1) optimal_threads = 1;
                // Note: We can't change nbThreads here, but we'll use larger chunk sizes
            } else {
                optimal_threads = nbThreads;
            }
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            (*outSize) += nbThreads * sizeof(size_t); 
            // Allocate padded structures to avoid false sharing
            outSize_perthread_arr = (struct PaddedSize *)malloc(nbThreads * sizeof(struct PaddedSize));
            offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));

            maxPreservedBufferSize_perthread = (sizeof(float) * nbEle + nbThreads - 1) / nbThreads;
            inver_bound = 1.0 / absErrBound;
            threadblocksize = nbEle / nbThreads;
        }
        size_t j = 0;
        
        // Use aligned allocation for better cache performance (improves scaling)
        unsigned char *outputBytes_perthread = NULL;
        // Try aligned allocation first (better for cache performance with many threads)
        #if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
        if (posix_memalign((void**)&outputBytes_perthread, 64, maxPreservedBufferSize_perthread) != 0) {
            outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
        }
        #elif defined(__APPLE__) || defined(_WIN32)
        // macOS and Windows use different alignment functions
        outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
        #else
        // Try posix_memalign if available (Linux)
        if (posix_memalign((void**)&outputBytes_perthread, 64, maxPreservedBufferSize_perthread) != 0) {
            outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
        }
        #endif
        size_t outSize_perthread = 0;
        
        int tid = omp_get_thread_num();
        
        int prior = 0;
        int current = 0;
        int diff = 0;
        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *block_pointer = outputBytes_perthread;
        
        unsigned char *temp_sign_arr = (unsigned char *)malloc((block_size - 1) * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc((block_size - 1) * sizeof(unsigned int));
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
        
        // For better scaling: use work-stealing approach when thread count is high
        // For high thread counts with small problems, use larger chunks to reduce overhead
        // For normal cases, use static distribution for better cache locality
        size_t blocks_per_thread;
        size_t start_block, end_block;
        
        if (nbThreads >= 16 && num_blocks < nbThreads * 8) {
            // High thread count, small problem: use larger chunks per thread
            // This reduces scheduling overhead and improves cache efficiency
            blocks_per_thread = (num_blocks + optimal_threads - 1) / optimal_threads;
            size_t effective_tid = tid % optimal_threads;  // Wrap threads to reduce active count
            start_block = effective_tid * blocks_per_thread;
            end_block = (effective_tid + 1) * blocks_per_thread;
            if (end_block > num_blocks) end_block = num_blocks;
            // Skip threads beyond optimal_threads to reduce memory bandwidth contention
            if (tid >= optimal_threads) {
                start_block = end_block;  // No work for this thread
            }
        } else {
            // Normal case: static distribution
            blocks_per_thread = (num_blocks + nbThreads - 1) / nbThreads;
            start_block = tid * blocks_per_thread;
            end_block = (tid + 1) * blocks_per_thread;
            if (end_block > num_blocks) end_block = num_blocks;
        }
        
        // Process blocks assigned to this thread
        for (size_t block_idx = start_block; block_idx < end_block; block_idx++)
        {
            size_t block_start = block_idx * block_size;
            size_t current_block_size = (block_start + block_size > nbEle) ? (nbEle - block_start) : block_size;
            if (current_block_size == 0) continue;

            max = 0;
            prior = szp_quantize<float>(op[block_start], (double)absErrBound, inver_bound);
            memcpy(block_pointer, &prior, sizeof(int));
            block_pointer += sizeof(unsigned int);
            outSize_perthread += sizeof(unsigned int);

            if (current_block_size > 1)
            {
                // Vectorization hint for inner loop - helps compiler optimize
                // No "omp simd": prior carries a dependence between iterations.
                for (j = 0; j < current_block_size - 1; j++)
                {
                    current = szp_quantize<float>(op[block_start + j + 1], (double)absErrBound, inver_bound);
                    diff = current - prior;
                    prior = current;
                    if (diff == 0)
                    {
                        temp_sign_arr[j] = 0;
                        temp_predict_arr[j] = 0;
                    }
                    else
                    {
                        if (diff < 0)
                        {
                            temp_sign_arr[j] = 1;
                            temp_predict_arr[j] = -diff;
                        }
                        else
                        {
                            temp_sign_arr[j] = 0;
                            temp_predict_arr[j] = diff;
                        }
                        if (max < temp_predict_arr[j])
                            max = temp_predict_arr[j];
                    }
                }
            }

            if (max == 0) 
            {
                block_pointer[0] = 0;
                block_pointer++;
                outSize_perthread++;
            }
            else
            {
                bit_count = (int)(log2f(max)) + 1;
                block_pointer[0] = bit_count;
                outSize_perthread++;
                block_pointer++;
                signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, current_block_size - 1, block_pointer); 
                block_pointer += signbytelength;
                outSize_perthread += signbytelength;
                savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, current_block_size - 1, block_pointer, bit_count);
                block_pointer += savedbitsbytelength;
                outSize_perthread += savedbitsbytelength;
            }
        }  // End of block loop
        
        // Store size with padding to avoid false sharing
        // Note: outSize_perthread is accumulated across all blocks processed by this thread
        outSize_perthread_arr[tid].size = outSize_perthread;
#pragma omp barrier

        // Calculate offsets using prefix sum (sequential but fast for small thread counts)
        // This is still more efficient than the original due to reduced barrier overhead
#pragma omp single
        {
            offsets_perthread_arr[0] = 0;
            for (size_t i = 1; i < nbThreads; i++)
            {
                offsets_perthread_arr[i] = offsets_perthread_arr[i - 1] + outSize_perthread_arr[i - 1].size;
            }
            (*outSize) += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1].size;
            memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t)); szp_stream_write_tag(outputBytes, nbThreads);
        }
#pragma omp barrier
        memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);
#pragma omp barrier
        
        free(outputBytes_perthread);
        free(temp_sign_arr);
        free(temp_predict_arr);
#pragma omp single
        {
            free(outSize_perthread_arr);
            free(offsets_perthread_arr);
        }
    }
    
    (*outSize) += sizeof(float);
    return output;
    
#else
    printf("Error! OpenMP not supported!\n");
    return NULL;
#endif
}

void
szp_float_openmp_threadblock_randomaccess_arg(unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
                                          size_t nbEle, int blockSize)
{
#ifdef _OPENMP
    
    float *op = oriData;

    unsigned char* outputBytes = output + sizeof(float);
    floatToBytes(output, absErrBound);
    
    size_t maxPreservedBufferSize = sizeof(float) + sizeof(float) * nbEle; 
    size_t maxPreservedBufferSize_perthread = 0;
    unsigned char *real_outputBytes; 
    // Use padded structure to avoid false sharing (64-byte cache line alignment)
    struct PaddedSize {
        size_t size;
        char padding[64 - sizeof(size_t)];  // Pad to 64-byte cache line boundary
    };
    struct PaddedSize *outSize_perthread_arr;
    size_t *offsets_perthread_arr;
    
    (*outSize) = 0;
    

    unsigned int nbThreads = 0;
    double inver_bound = 0;
    unsigned int threadblocksize = 0;
    unsigned int block_size = blockSize;

#pragma omp parallel
    {
#pragma omp single
        {
            nbThreads = omp_get_num_threads();
            real_outputBytes = outputBytes + nbThreads * sizeof(size_t);
            (*outSize) += nbThreads * sizeof(size_t); 
            // Allocate padded structures to avoid false sharing
            outSize_perthread_arr = (struct PaddedSize *)malloc(nbThreads * sizeof(struct PaddedSize));
            offsets_perthread_arr = (size_t *)malloc(nbThreads * sizeof(size_t));

            maxPreservedBufferSize_perthread = (sizeof(float) * nbEle + nbThreads - 1) / nbThreads;
            inver_bound = 1.0 / absErrBound;
            threadblocksize = nbEle / nbThreads;
        }
        size_t i = 0;
        size_t j = 0;
        unsigned char *outputBytes_perthread = (unsigned char *)malloc(maxPreservedBufferSize_perthread);
        size_t outSize_perthread = 0;
        
        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (tid == nbThreads - 1) {
            hi = nbEle; // Ensure the last thread processes all remaining elements
        }

        int prior = 0;
        int current = 0;
        int diff = 0;
        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *block_pointer = outputBytes_perthread;
        
        unsigned char *temp_sign_arr = (unsigned char *)malloc((block_size - 1) * sizeof(unsigned char));
        
        unsigned int *temp_predict_arr = (unsigned int *)malloc((block_size - 1) * sizeof(unsigned int));
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
        
        for (i = lo; i < hi; i = i + block_size)
        {
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            max = 0;
            prior = szp_quantize<float>(op[i], (double)absErrBound, inver_bound);
            memcpy(block_pointer, &prior, sizeof(int));
            block_pointer += sizeof(unsigned int);
            outSize_perthread += sizeof(unsigned int);

            if (current_block_size > 1)
            {
                for (j = 0; j < current_block_size - 1; j++)
                {
                    current = szp_quantize<float>(op[i + j + 1], (double)absErrBound, inver_bound);
                    diff = current - prior;
                    prior = current;
                    if (diff == 0)
                    {
                        temp_sign_arr[j] = 0;
                        temp_predict_arr[j] = 0;
                    }
                    else
                    {
                        if (diff < 0)
                        {
                            temp_sign_arr[j] = 1;
                            temp_predict_arr[j] = -diff;
                        }
                        else
                        {
                            temp_sign_arr[j] = 0;
                            temp_predict_arr[j] = diff;
                        }
                        if (max < temp_predict_arr[j])
                            max = temp_predict_arr[j];
                    }
                }
            }

            if (max == 0) 
            {
                block_pointer[0] = 0;
                block_pointer++;
                outSize_perthread++;
            }
            else
            {
                bit_count = (int)(log2f(max)) + 1;
                block_pointer[0] = bit_count;
                outSize_perthread++;
                block_pointer++;
                signbytelength = convertIntArray2ByteArray_fast_1b_args(temp_sign_arr, current_block_size - 1, block_pointer); 
                block_pointer += signbytelength;
                outSize_perthread += signbytelength;
                savedbitsbytelength = Jiajun_save_fixed_length_bits(temp_predict_arr, current_block_size - 1, block_pointer, bit_count);
                block_pointer += savedbitsbytelength;
                outSize_perthread += savedbitsbytelength;
            }
        }

        outSize_perthread_arr[tid].size = outSize_perthread;
#pragma omp barrier

#pragma omp single
        {
            offsets_perthread_arr[0] = 0;
            for (i = 1; i < nbThreads; i++)
            {
                offsets_perthread_arr[i] = offsets_perthread_arr[i - 1] + outSize_perthread_arr[i - 1].size;
                
            }
            (*outSize) += offsets_perthread_arr[nbThreads - 1] + outSize_perthread_arr[nbThreads - 1].size;
            memcpy(outputBytes, offsets_perthread_arr, nbThreads * sizeof(size_t)); szp_stream_write_tag(outputBytes, nbThreads);
            
        }
#pragma omp barrier
        memcpy(real_outputBytes + offsets_perthread_arr[tid], outputBytes_perthread, outSize_perthread);
#pragma omp barrier
        
        free(outputBytes_perthread);
        free(temp_sign_arr);
        free(temp_predict_arr);
#pragma omp single
        {
            
            free(outSize_perthread_arr);
            free(offsets_perthread_arr);
        }
        
    }
    
    (*outSize) += sizeof(float);
    
#else
    printf("Error! OpenMP not supported!\n");
#endif
}
