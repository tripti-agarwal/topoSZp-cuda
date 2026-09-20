/**
 *  @file szpd_float.h
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>, Sheng Di <sdi1@anl.gov>
 *  @date Oct, 2023
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include "szpd_float.h"
#include <assert.h>
#include <math.h>
#include "szp_TypeManager.h"
#include "szp_CompressionToolkit.h"

#ifdef _OPENMP
#include "omp.h"
#endif
#ifdef _POSIX_C_SOURCE
#include <malloc.h>  // For posix_memalign
#endif

using namespace szp;

#include "szp_detect_threads.h"

float *szp_float_decompress_openmp_threadblock(size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
#ifdef _OPENMP
    float *newData = (float *)malloc(sizeof(float) * nbEle);
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 0;

    size_t threadblocksize = 0;
    int block_size = blockSize;

    nbThreads = szp_detect_nbThreads_threadblock(cmpBytes, nbEle, blockSize);
    rcp = cmpBytes + nbThreads * sizeof(size_t);
    threadblocksize = nbEle / nbThreads;

#pragma omp parallel num_threads(nbThreads)
    {
        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (tid == nbThreads - 1) {
            hi = nbEle; // Ensure the last thread processes all remaining elements
        }

        float *newData_perthread = newData + lo;
        size_t i = 0;
        size_t j = 0;

        int prior = 0;
        int current = 0;
        int diff = 0;

        unsigned int bit_count = 0;
        unsigned char *block_pointer = rcp + offsets[tid];

        float ori_prior = 0.0;
        float ori_current = 0.0;

        if (lo < hi) { // Ensure thread has data to process
            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);

            ori_prior = (float)prior * absErrBound;
            memcpy(newData_perthread, &ori_prior, sizeof(float)); 
            newData_perthread += 1;
        }
        
        unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
        unsigned int savedbitsbytelength = 0;
        
        for (i = lo + 1; i < hi; i = i + block_size)
        {
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            bit_count = block_pointer[0];
            block_pointer++;
            
            if (bit_count == 0)
            {
                ori_prior = (float)prior * absErrBound;
                
                for (j = 0; j < current_block_size; j++)
                {
                    memcpy(newData_perthread, &ori_prior, sizeof(float));
                    newData_perthread++;
                }
            }
            else
            {
                convertByteArray2IntArray_fast_1b_args(current_block_size, block_pointer, (current_block_size - 1) / 8 + 1, temp_sign_arr);
                block_pointer += ((current_block_size - 1) / 8 + 1);

                savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size, temp_predict_arr, bit_count);
                block_pointer += savedbitsbytelength;
                for (j = 0; j < current_block_size; j++)
                {
                    if (temp_sign_arr[j] == 0)
                    {
                        diff = temp_predict_arr[j];
                    }
                    else
                    {
                        diff = 0 - temp_predict_arr[j];
                    }
                    current = prior + diff;
                    ori_current = (float)current * absErrBound;
                    prior = current;
                    memcpy(newData_perthread, &ori_current, sizeof(float));
                    newData_perthread++;
                }
            }
        }
        free(temp_predict_arr);
        free(temp_sign_arr);
    }
    return newData;

#else
    printf("Error! OpenMP not supported!\n");
    return NULL; 
#endif
}

void szp_float_decompress_single_thread_arg(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 1;
    
    rcp = cmpBytes + nbThreads * sizeof(size_t);
    size_t block_size = blockSize;

    size_t lo = 0;
    size_t hi = nbEle;
    float *newData_perthread = newData;
    size_t i = 0;
    size_t j = 0;

    int prior = 0;
    int current = 0;
    int diff = 0;

    unsigned int bit_count = 0;
    unsigned char *block_pointer = rcp + offsets[0];

    float ori_prior = 0.0;
    float ori_current = 0.0;

    if (lo < hi) { // Ensure there is at least one element to decompress
        memcpy(&prior, block_pointer, sizeof(int));
        block_pointer += sizeof(unsigned int);

        ori_prior = (float)prior * absErrBound;
        memcpy(newData_perthread, &ori_prior, sizeof(float)); 
        newData_perthread += 1;
    }
    
    unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned int savedbitsbytelength = 0;
    
    for (i = lo + 1; i < hi; i = i + block_size)
    {
        size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
        if (current_block_size == 0) continue;

        bit_count = block_pointer[0];
        block_pointer++;
        
        if (bit_count == 0)
        {
            ori_prior = (float)prior * absErrBound;
            
            for (j = 0; j < current_block_size; j++)
            {
                memcpy(newData_perthread, &ori_prior, sizeof(float));
                newData_perthread++;
            }
        }
        else
        {
            convertByteArray2IntArray_fast_1b_args(current_block_size, block_pointer, (current_block_size - 1) / 8 + 1, temp_sign_arr);
            block_pointer += ((current_block_size - 1) / 8 + 1);

            savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size, temp_predict_arr, bit_count);
            block_pointer += savedbitsbytelength;
            for (j = 0; j < current_block_size; j++)
            {
                if (temp_sign_arr[j] == 0)
                {
                    diff = temp_predict_arr[j];
                }
                else
                {
                    diff = 0 - temp_predict_arr[j];
                }
                current = prior + diff;
                ori_current = (float)current * absErrBound;
                prior = current;
                memcpy(newData_perthread, &ori_current, sizeof(float));
                newData_perthread++;
            }
        }
    }
    
    free(temp_predict_arr);
    free(temp_sign_arr);
}

size_t szp_float_decompress_single_thread_arg_record(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
    size_t total_memaccess = 0;
    
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 1;
    
    rcp = cmpBytes + nbThreads * sizeof(size_t);
    size_t block_size = blockSize;

    size_t lo = 0;
    size_t hi = nbEle;
    float *newData_perthread = newData;
    size_t i = 0;
    size_t j = 0;

    int prior = 0;
    int current = 0;
    int diff = 0;

    unsigned int bit_count = 0;
    unsigned char *block_pointer = rcp + offsets[0]; 
    total_memaccess += sizeof(size_t); // Reading from offsets array

    float ori_prior = 0.0;
    float ori_current = 0.0;

    if (lo < hi) { // Ensure there is at least one element to decompress
        memcpy(&prior, block_pointer, sizeof(int));
        total_memaccess += sizeof(int) * 2; // read from block_pointer and write to prior
        block_pointer += sizeof(unsigned int);

        ori_prior = (float)prior * absErrBound;
        memcpy(newData_perthread, &ori_prior, sizeof(float)); 
        total_memaccess += sizeof(float) * 2; // read from ori_prior and write to newData_perthread
        newData_perthread += 1;
    }
    
    unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned int savedbitsbytelength = 0;

    // Unified loop for all remaining data blocks
    for (i = lo + 1; i < hi; i = i + block_size)
    {
        size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
        if (current_block_size == 0) continue;

        bit_count = block_pointer[0];
        total_memaccess += sizeof(unsigned char); // Reading bit_count
        block_pointer++;
        
        if (bit_count == 0)
        {
            ori_prior = (float)prior * absErrBound;
            for (j = 0; j < current_block_size; j++)
            {
                memcpy(newData_perthread, &ori_prior, sizeof(float));
                total_memaccess += sizeof(float) * 2; // read from ori_prior and write to newData_perthread
                newData_perthread++;
            }
        }
        else
        {
            size_t sign_byte_len = (current_block_size - 1) / 8 + 1;
            convertByteArray2IntArray_fast_1b_args(current_block_size, block_pointer, sign_byte_len, temp_sign_arr);
            total_memaccess += sizeof(unsigned char) * sign_byte_len;         // Reading from block_pointer
            total_memaccess += sizeof(unsigned char) * current_block_size;    // Writing to temp_sign_arr
            block_pointer += sign_byte_len;

            savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size, temp_predict_arr, bit_count);
            total_memaccess += sizeof(unsigned char) * savedbitsbytelength;   // Reading from block_pointer
            total_memaccess += sizeof(unsigned int) * current_block_size;     // Writing to temp_predict_arr
            block_pointer += savedbitsbytelength;

            for (j = 0; j < current_block_size; j++)
            {
                if (temp_sign_arr[j] == 0)
                {
                    diff = temp_predict_arr[j];
                }
                else
                {
                    diff = 0 - temp_predict_arr[j];
                }
                total_memaccess += sizeof(unsigned char); // Reading from temp_sign_arr
                total_memaccess += sizeof(unsigned int);  // Reading from temp_predict_arr

                current = prior + diff;
                ori_current = (float)current * absErrBound;
                prior = current;
                memcpy(newData_perthread, &ori_current, sizeof(float));
                total_memaccess += sizeof(float) * 2;     // read from ori_current and write to newData_perthread
                newData_perthread++;
            }
        }
    }
    
    free(temp_predict_arr);
    free(temp_sign_arr);
    return total_memaccess;
}

void szp_float_decompress_openmp_threadblock_arg(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
#ifdef _OPENMP

    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 0;

    size_t threadblocksize = 0;
    size_t block_size = blockSize;

    nbThreads = szp_detect_nbThreads_threadblock(cmpBytes, nbEle, blockSize);
    rcp = cmpBytes + nbThreads * sizeof(size_t);
    threadblocksize = nbEle / nbThreads;

#pragma omp parallel num_threads(nbThreads)
    {
        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (tid == nbThreads - 1) {
            hi = nbEle; // Ensure the last thread processes all remaining elements
        }

        float *newData_perthread = newData + lo;
        size_t i = 0;
        size_t j = 0;

        int prior = 0;
        int current = 0;
        int diff = 0;

        unsigned int bit_count = 0;
        unsigned char *block_pointer = rcp + offsets[tid];

        float ori_prior = 0.0;
        float ori_current = 0.0;

        if (lo < hi) { // Ensure thread has data to process
            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);

            ori_prior = (float)prior * absErrBound;
            memcpy(newData_perthread, &ori_prior, sizeof(float)); 
            newData_perthread += 1;
        }
        
        unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
        unsigned int savedbitsbytelength = 0;
        
        for (i = lo + 1; i < hi; i = i + block_size)
        {
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            bit_count = block_pointer[0];
            block_pointer++;
            
            if (bit_count == 0)
            {
                ori_prior = (float)prior * absErrBound;
                
                for (j = 0; j < current_block_size; j++)
                {
                    memcpy(newData_perthread, &ori_prior, sizeof(float));
                    newData_perthread++;
                }
            }
            else
            {
                convertByteArray2IntArray_fast_1b_args(current_block_size, block_pointer, (current_block_size - 1) / 8 + 1, temp_sign_arr);
                block_pointer += ((current_block_size - 1) / 8 + 1);

                savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size, temp_predict_arr, bit_count);
                block_pointer += savedbitsbytelength;
                for (j = 0; j < current_block_size; j++)
                {
                    if (temp_sign_arr[j] == 0)
                    {
                        diff = temp_predict_arr[j];
                    }
                    else
                    {
                        diff = 0 - temp_predict_arr[j];
                    }
                    current = prior + diff;
                    ori_current = (float)current * absErrBound;
                    prior = current;
                    memcpy(newData_perthread, &ori_current, sizeof(float));
                    newData_perthread++;
                }
            }
        }
        free(temp_predict_arr);
        free(temp_sign_arr);
    }

#else
    printf("Error! OpenMP not supported!\n");
#endif
}

float *szp_float_decompress_openmp_threadblock_randomaccess(size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
#ifdef _OPENMP
    float *newData = (float *)malloc(sizeof(float) * nbEle);
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 0;

    size_t threadblocksize = 0;
    int block_size = blockSize;

    nbThreads = szp_detect_nbThreads_randomaccess(cmpBytes, nbEle, blockSize);
    rcp = cmpBytes + nbThreads * sizeof(size_t);
    threadblocksize = nbEle / nbThreads;

#pragma omp parallel num_threads(nbThreads)
    {
        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (tid == nbThreads - 1) {
            hi = nbEle;
        }
        float *newData_perthread = newData + lo;
        size_t i = 0;
        size_t j = 0;

        int prior = 0;
        int current = 0;
        int diff = 0;

        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *outputBytes_perthread = rcp + offsets[tid]; 
        unsigned char *block_pointer = outputBytes_perthread;

        float ori_prior = 0.0;
        float ori_current = 0.0;

        unsigned char *temp_sign_arr = (unsigned char *)malloc((block_size-1) * sizeof(unsigned char)); // 1 direct value and (block_size - 1) diff. values
        
        unsigned int *temp_predict_arr = (unsigned int *)malloc((block_size-1) * sizeof(unsigned int));
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
        
        for (i = lo; i < hi; i = i + block_size)
        {
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);
            ori_prior = (float)prior * absErrBound;
            memcpy(newData_perthread, &ori_prior, sizeof(float)); 
            newData_perthread ++;

            if (current_block_size > 1)
            {
                bit_count = block_pointer[0];
                block_pointer++;

                if (bit_count == 0)
                {
                    for (j = 0; j < current_block_size - 1; j++)
                    {
                        memcpy(newData_perthread, &ori_prior, sizeof(float));
                        newData_perthread++;
                    }
                }
                else
                {
                    convertByteArray2IntArray_fast_1b_args(current_block_size - 1, block_pointer, (current_block_size - 2) / 8 + 1, temp_sign_arr);
                    block_pointer += ((current_block_size - 2) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size - 1, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < current_block_size - 1; j++)
                    {
                        if (temp_sign_arr[j] == 0)
                        {
                            diff = temp_predict_arr[j];
                        }
                        else
                        {
                            diff = 0 - temp_predict_arr[j];
                        }
                        current = prior + diff;
                        ori_current = (float)current * absErrBound;
                        prior = current;
                        memcpy(newData_perthread, &ori_current, sizeof(float));
                        newData_perthread++;
                    }
                }
            }
        }
        free(temp_sign_arr);
        free(temp_predict_arr);
    }
    return newData;

#else
    printf("Error! OpenMP not supported!\n");
#endif
}


void szp_float_decompress_openmp_threadblock_randomaccess_arg(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes)
{
#ifdef _OPENMP
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 0;

    size_t threadblocksize = 0;
    int block_size = blockSize;

    nbThreads = szp_detect_nbThreads_randomaccess(cmpBytes, nbEle, blockSize);
    rcp = cmpBytes + nbThreads * sizeof(size_t);
    threadblocksize = nbEle / nbThreads;

#pragma omp parallel num_threads(nbThreads)
    {
        int tid = omp_get_thread_num();
        size_t lo = tid * threadblocksize;
        size_t hi = (tid + 1) * threadblocksize;
        if (tid == nbThreads - 1) {
            hi = nbEle;
        }
        float *newData_perthread = newData + lo;
        size_t i = 0;
        size_t j = 0;

        int prior = 0;
        int current = 0;
        int diff = 0;

        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *outputBytes_perthread = rcp + offsets[tid]; 
        unsigned char *block_pointer = outputBytes_perthread;

        float ori_prior = 0.0;
        float ori_current = 0.0;

        
        unsigned char *temp_sign_arr = (unsigned char *)malloc((block_size-1) * sizeof(unsigned char)); // 1 direct value and block_size - 1 diff. values
        
        unsigned int *temp_predict_arr = (unsigned int *)malloc((block_size-1) * sizeof(unsigned int));
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
        
        for (i = lo; i < hi; i = i + block_size)
        {
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);
            ori_prior = (float)prior * absErrBound;
            memcpy(newData_perthread, &ori_prior, sizeof(float)); 
            newData_perthread ++;

            if (current_block_size > 1)
            {
                bit_count = block_pointer[0];
                block_pointer++;

                if (bit_count == 0)
                {
                    for (j = 0; j < current_block_size - 1; j++)
                    {
                        memcpy(newData_perthread, &ori_prior, sizeof(float));
                        newData_perthread++;
                    }
                }
                else
                {
                    convertByteArray2IntArray_fast_1b_args(current_block_size - 1, block_pointer, (current_block_size - 2) / 8 + 1, temp_sign_arr);
                    block_pointer += ((current_block_size - 2) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size - 1, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < current_block_size - 1; j++)
                    {
                        if (temp_sign_arr[j] == 0)
                        {
                            diff = temp_predict_arr[j];
                        }
                        else
                        {
                            diff = 0 - temp_predict_arr[j];
                        }
                        current = prior + diff;
                        ori_current = (float)current * absErrBound;
                        prior = current;
                        memcpy(newData_perthread, &ori_current, sizeof(float));
                        newData_perthread++;
                    }
                }
            }
        }
        free(temp_sign_arr);
        free(temp_predict_arr);
    }

#else
    printf("Error! OpenMP not supported!\n");
#endif
}

void szp_float_decompress_openmp_threadblock_randomaccess_topology_preserved(
    float **newData, size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes, int **FN)
{
#ifdef _OPENMP
    *newData = (float *)malloc(sizeof(float) * nbEle);
    *FN      = (int   *)calloc(nbEle, sizeof(int));

    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 0;

    const float scale = 1 * absErrBound;
    const unsigned int block_size     = (unsigned int)blockSize;
    const unsigned int new_block_size = block_size - 1;

    unsigned int threadblocksize = 0;
    unsigned int remainder       = 0;
    unsigned int num_remainder_in_tb = 0;

    // Adaptive thread limiting for better scaling with high thread counts
    size_t num_blocks_estimate = (nbEle + block_size - 1) / block_size;
    int optimal_threads = 0;

    nbThreads = szp_detect_nbThreads_randomaccess(cmpBytes, nbEle, blockSize);
    if (nbThreads == 0) nbThreads = 1;

    // For high thread counts with small problems, limit effective parallelism
    if (nbThreads >= 16 && num_blocks_estimate < nbThreads * 8) {
        optimal_threads = (num_blocks_estimate + 7) / 8;
        if (optimal_threads < 1) optimal_threads = 1;
    } else {
        optimal_threads = nbThreads;
    }

    rcp              = cmpBytes + nbThreads * sizeof(size_t);
    threadblocksize  = (unsigned int)(nbEle / nbThreads);
    remainder        = (unsigned int)(nbEle % nbThreads);
    num_remainder_in_tb = threadblocksize % block_size;

#pragma omp parallel num_threads(nbThreads)
    {
        const int tid = omp_get_thread_num();
        
        // Use block-based distribution for better cache locality and load balancing
        size_t num_blocks = (nbEle + block_size - 1) / block_size;
        size_t blocks_per_thread;
        size_t start_block, end_block;
        
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

        float *dst = *newData + lo;
        unsigned char *block_pointer = rcp + offsets[tid];

        int prior = 0, current = 0, diff = 0;
        unsigned int bit_count = 0;

        // Use cache-aligned allocation for temp arrays
        unsigned char *temp_sign_arr = NULL;
        unsigned int  *temp_predict_arr = NULL;
        unsigned char *temp_type_arr = NULL;
        
        #if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
        if (posix_memalign((void**)&temp_sign_arr, 64, new_block_size) != 0) {
            temp_sign_arr = (unsigned char *)malloc(new_block_size);
        }
        if (posix_memalign((void**)&temp_predict_arr, 64, new_block_size * sizeof(unsigned int)) != 0) {
            temp_predict_arr = (unsigned int *)malloc(new_block_size * sizeof(unsigned int));
        }
        if (posix_memalign((void**)&temp_type_arr, 64, block_size) != 0) {
            temp_type_arr = (unsigned char *)malloc(block_size);
        }
        #else
        temp_sign_arr = (unsigned char *)malloc(new_block_size);
        temp_predict_arr = (unsigned int *)malloc(new_block_size * sizeof(unsigned int));
        temp_type_arr = (unsigned char *)malloc(block_size);
        #endif

        // Process blocks assigned to this thread using block-based distribution
        for (size_t block_idx = start_block; block_idx < end_block; block_idx++) {
            size_t i = block_idx * block_size;
            if (i >= nbEle) break;
            
            size_t current_block_size = (i + block_size > nbEle) ? (nbEle - i) : block_size;
            if (current_block_size == 0) continue;
            
            unsigned int actual_new_block_size = (current_block_size > 1) ? (current_block_size - 1) : 0;
                memcpy(&prior, block_pointer, sizeof(int));
                block_pointer += sizeof(int);
                float ori = (float)prior * scale;
                *dst++ = ori;

            bit_count = *block_pointer++;
            if (bit_count == 0) {
                // Vectorized memset-like operation
                #pragma omp simd
                for (unsigned int j = 0; j < actual_new_block_size; j++) {
                    dst[j] = ori;
                }
                dst += actual_new_block_size;
            } else {
                const unsigned int signbytes = (actual_new_block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(actual_new_block_size, block_pointer, signbytes, temp_sign_arr);
                block_pointer += signbytes;

                const unsigned int savedbitsbytelength =
                    Jiajun_extract_fixed_length_bits(block_pointer, actual_new_block_size, temp_predict_arr, bit_count);
                block_pointer += savedbitsbytelength;

                // Add vectorization hint for inner loop
                #pragma omp simd
                for (unsigned int j = 0; j < actual_new_block_size; j++) {
                    diff    = temp_sign_arr[j] ? -(int)temp_predict_arr[j] : (int)temp_predict_arr[j];
                    current = prior + diff;
                    prior   = current;
                    dst[j]  = (float)current * scale;
                }
                dst += actual_new_block_size;
            }

            const unsigned int typebytelength = (2 * current_block_size + 7) / 8;
            memset(temp_type_arr, 0, current_block_size);
            // Safety check: ensure we don't read past the buffer
            unsigned int safe_typebytelength = typebytelength;
            if (typebytelength > 64) safe_typebytelength = 64; // Cap at reasonable max
            convertByteArray2IntArray_fast_2b(current_block_size, block_pointer, safe_typebytelength, &temp_type_arr);
            // Vectorized copy
            #pragma omp simd
            for (unsigned int j = 0; j < current_block_size; j++) {
                (*FN)[i + j] = temp_type_arr[j];
            }
            block_pointer += typebytelength;
        }

        free(temp_sign_arr);
        free(temp_predict_arr);
        free(temp_type_arr);
    }
#else
    (void)newData; (void)nbEle; (void)absErrBound; (void)blockSize; (void)cmpBytes; (void)FN;
#endif
}

/**
 * Decompress sort_position values from compressed data.
 * Based on szp_float_decompress_openmp_threadblock_randomaccess but adapted for integers.
 * 
 * @param cmpBytes Compressed data
 * @param critical_count Number of critical points
 * @param blockSize Block size used for compression
 * @return Decompressed sort positions array
 */
int *szp_decompress_sort_positions(unsigned char *cmpBytes, size_t critical_count, int blockSize) {
#ifdef _OPENMP
    if (!cmpBytes || critical_count == 0) {
        return NULL;
    }
    
    int *newData = (int *)malloc(sizeof(int) * critical_count);
    if (!newData) {
        return NULL;
    }
    
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 0;

    size_t threadblocksize = 0;
    int block_size = blockSize;

    // Adaptive thread limiting for better scaling with high thread counts
    size_t num_blocks_estimate = (critical_count + block_size - 1) / block_size;
    int optimal_threads = 0;

    nbThreads = szp_detect_nbThreads_randomaccess(cmpBytes, critical_count, blockSize);
    if (nbThreads == 0) nbThreads = 1;

    // For high thread counts with small problems, limit effective parallelism
    if (nbThreads >= 16 && num_blocks_estimate < nbThreads * 8) {
        optimal_threads = (num_blocks_estimate + 7) / 8;
        if (optimal_threads < 1) optimal_threads = 1;
    } else {
        optimal_threads = nbThreads;
    }

    rcp = cmpBytes + nbThreads * sizeof(size_t);
    threadblocksize = critical_count / nbThreads;

#pragma omp parallel num_threads(nbThreads)
    {
        int tid = omp_get_thread_num();
        
        // Use block-based distribution for better cache locality and load balancing
        size_t num_blocks = (critical_count + block_size - 1) / block_size;
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
            // Normal case: static block distribution
            blocks_per_thread = (num_blocks + nbThreads - 1) / nbThreads;
            start_block = tid * blocks_per_thread;
            end_block = (tid + 1) * blocks_per_thread;
            if (end_block > num_blocks) end_block = num_blocks;
        }
        
        size_t lo = start_block * block_size;
        size_t hi = end_block * block_size;
        if (hi > critical_count) hi = critical_count;
        
        int *newData_perthread = newData + lo;
        size_t i = 0;
        size_t j = 0;

        int prior = 0;
        int current = 0;
        int diff = 0;

        unsigned int max = 0;
        unsigned int bit_count = 0;
        unsigned char *outputBytes_perthread = rcp + offsets[tid]; 
        unsigned char *block_pointer = outputBytes_perthread;

        // Use cache-aligned allocation for temp arrays
        unsigned char *temp_sign_arr = NULL;
        unsigned int *temp_predict_arr = NULL;
        
        size_t temp_arr_size = (block_size > 1) ? (block_size - 1) : 1;
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
        
        unsigned int signbytelength = 0; 
        unsigned int savedbitsbytelength = 0;
        
        // Process blocks assigned to this thread
        for (size_t block_idx = start_block; block_idx < end_block; block_idx++)
        {
            i = block_idx * block_size;
            if (i >= critical_count) break;
            
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);
            memcpy(newData_perthread, &prior, sizeof(int)); 
            newData_perthread ++;

            if (current_block_size > 1)
            {
                bit_count = block_pointer[0];
                block_pointer++;

                if (bit_count == 0)
                {
                    for (j = 0; j < current_block_size - 1; j++)
                    {
                        memcpy(newData_perthread, &prior, sizeof(int));
                        newData_perthread++;
                    }
                }
                else
                {
                    convertByteArray2IntArray_fast_1b_args(current_block_size - 1, block_pointer, (current_block_size - 2) / 8 + 1, temp_sign_arr);
                    block_pointer += ((current_block_size - 2) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size - 1, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    // Add vectorization hint for inner loop
                    #pragma omp simd
                    for (j = 0; j < current_block_size - 1; j++)
                    {
                        diff = temp_sign_arr[j] ? -(int)temp_predict_arr[j] : (int)temp_predict_arr[j];
                        current = prior + diff;
                        prior = current;
                        newData_perthread[j] = current;
                    }
                    newData_perthread += current_block_size - 1;
                }
            }
        }
        free(temp_sign_arr);
        free(temp_predict_arr);
    }
    
    return newData;

#else
    // Fallback to sequential version if OpenMP not available
    if (!cmpBytes || critical_count == 0) {
        return NULL;
    }
    
    int *newData = (int *)malloc(sizeof(int) * critical_count);
    if (!newData) {
        return NULL;
    }
    
    unsigned char *block_pointer = cmpBytes;
    
    unsigned char *temp_sign_arr = (unsigned char *)malloc((blockSize - 1) * sizeof(unsigned char));
    unsigned int *temp_predict_arr = (unsigned int *)malloc((blockSize - 1) * sizeof(unsigned int));
    unsigned int savedbitsbytelength = 0;
    
    for (size_t i = 0; i < critical_count; i = i + blockSize)
    {
        size_t current_block_size = (i + blockSize > critical_count) ? (critical_count - i) : blockSize;
        if (current_block_size == 0) continue;

        unsigned int bit_count = block_pointer[0];
        block_pointer++;
        
        int prior = 0;
        memcpy(&prior, block_pointer, sizeof(int));
        block_pointer += sizeof(int);
        memcpy(newData + i, &prior, sizeof(int));
        
        if (bit_count == 0)
        {
            for (size_t j = 1; j < current_block_size; j++)
            {
                memcpy(newData + i + j, &prior, sizeof(int));
            }
        }
        else
        {
            convertByteArray2IntArray_fast_1b_args(current_block_size - 1, block_pointer, (current_block_size - 2) / 8 + 1, temp_sign_arr);
            block_pointer += ((current_block_size - 2) / 8 + 1);

            savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size - 1, temp_predict_arr, bit_count);
            block_pointer += savedbitsbytelength;
            for (size_t j = 0; j < current_block_size - 1; j++)
            {
                int diff;
                if (temp_sign_arr[j] == 0)
                {
                    diff = temp_predict_arr[j];
                }
                else
                {
                    diff = 0 - temp_predict_arr[j];
                }
                int current = prior + diff;
                prior = current;
                memcpy(newData + i + j + 1, &current, sizeof(int));
            }
        }
    }
    
    free(temp_sign_arr);
    free(temp_predict_arr);
    return newData;
#endif
}
