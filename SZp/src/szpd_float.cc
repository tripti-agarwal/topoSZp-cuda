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
#include "szp_quant.h"
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
    if (!newData) { fprintf(stderr, "szpd_float: malloc failed\n"); exit(EXIT_FAILURE); }
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
        unsigned char *block_pointer = rcp + szp_read_offset((const unsigned char *)offsets, tid);

        float ori_prior = 0.0;
        float ori_current = 0.0;

        if (lo < hi) { // Ensure thread has data to process
            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);

            ori_prior = szp_dequantize<float>(prior, (double)absErrBound);
            memcpy(newData_perthread, &ori_prior, sizeof(float)); 
            newData_perthread += 1;
        }
        
        unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
        if (!temp_sign_arr || !temp_predict_arr) { fprintf(stderr, "szpd_float: malloc failed\n"); exit(EXIT_FAILURE); }
        unsigned int savedbitsbytelength = 0;
        
        for (i = lo + 1; i < hi; i = i + block_size)
        {
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            bit_count = block_pointer[0];
            block_pointer++;
            
            if (bit_count == 0)
            {
                ori_prior = szp_dequantize<float>(prior, (double)absErrBound);
                
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
                    ori_current = szp_dequantize<float>(current, (double)absErrBound);
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
    unsigned char *block_pointer = rcp + szp_read_offset((const unsigned char *)offsets, 0);

    float ori_prior = 0.0;
    float ori_current = 0.0;

    if (lo < hi) { // Ensure there is at least one element to decompress
        memcpy(&prior, block_pointer, sizeof(int));
        block_pointer += sizeof(unsigned int);

        ori_prior = szp_dequantize<float>(prior, (double)absErrBound);
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
            ori_prior = szp_dequantize<float>(prior, (double)absErrBound);
            
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
                ori_current = szp_dequantize<float>(current, (double)absErrBound);
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
    unsigned char *block_pointer = rcp + szp_read_offset((const unsigned char *)offsets, 0); 
    total_memaccess += sizeof(size_t); // Reading from offsets array

    float ori_prior = 0.0;
    float ori_current = 0.0;

    if (lo < hi) { // Ensure there is at least one element to decompress
        memcpy(&prior, block_pointer, sizeof(int));
        total_memaccess += sizeof(int) * 2; // read from block_pointer and write to prior
        block_pointer += sizeof(unsigned int);

        ori_prior = szp_dequantize<float>(prior, (double)absErrBound);
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
            ori_prior = szp_dequantize<float>(prior, (double)absErrBound);
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
                ori_current = szp_dequantize<float>(current, (double)absErrBound);
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
        unsigned char *block_pointer = rcp + szp_read_offset((const unsigned char *)offsets, tid);

        float ori_prior = 0.0;
        float ori_current = 0.0;

        if (lo < hi) { // Ensure thread has data to process
            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);

            ori_prior = szp_dequantize<float>(prior, (double)absErrBound);
            memcpy(newData_perthread, &ori_prior, sizeof(float)); 
            newData_perthread += 1;
        }
        
        unsigned char *temp_sign_arr = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
        if (!temp_sign_arr || !temp_predict_arr) { fprintf(stderr, "szpd_float: malloc failed\n"); exit(EXIT_FAILURE); }
        unsigned int savedbitsbytelength = 0;
        
        for (i = lo + 1; i < hi; i = i + block_size)
        {
            size_t current_block_size = (i + block_size > hi) ? (hi - i) : block_size;
            if (current_block_size == 0) continue;

            bit_count = block_pointer[0];
            block_pointer++;
            
            if (bit_count == 0)
            {
                ori_prior = szp_dequantize<float>(prior, (double)absErrBound);
                
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
                    ori_current = szp_dequantize<float>(current, (double)absErrBound);
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
    if (!newData) { fprintf(stderr, "szpd_float: malloc failed\n"); exit(EXIT_FAILURE); }
    size_t *offsets = (size_t *)cmpBytes;
    unsigned char *rcp;
    unsigned int nbThreads = 0;

    size_t threadblocksize = 0;
    int block_size = blockSize;

    nbThreads = szp_detect_nbThreads_randomaccess(cmpBytes, nbEle, blockSize);
    rcp = cmpBytes + nbThreads * sizeof(size_t);

    /* Use block-based distribution matching the compressor */
    size_t num_blocks = (nbEle + block_size - 1) / block_size;

#pragma omp parallel num_threads(nbThreads)
    {
        int tid = omp_get_thread_num();
        size_t blocks_per_thread = (num_blocks + nbThreads - 1) / nbThreads;
        size_t start_block = tid * blocks_per_thread;
        size_t end_block = (tid + 1) * blocks_per_thread;
        if (end_block > num_blocks) end_block = num_blocks;

        size_t j = 0;

        int prior = 0;
        int current = 0;
        int diff = 0;

        unsigned int bit_count = 0;
        unsigned char *block_pointer = rcp + szp_read_offset((const unsigned char *)offsets, tid);

        float ori_prior = 0.0;
        float ori_current = 0.0;

        unsigned char *temp_sign_arr = (unsigned char *)malloc((block_size-1) * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc((block_size-1) * sizeof(unsigned int));
        if (!temp_sign_arr || !temp_predict_arr) { fprintf(stderr, "szpd_float: malloc failed\n"); exit(EXIT_FAILURE); }
        unsigned int savedbitsbytelength = 0;

        for (size_t block_idx = start_block; block_idx < end_block; block_idx++)
        {
            size_t i = block_idx * block_size;
            if (i >= nbEle) break;
            size_t current_block_size = (i + block_size > nbEle) ? (nbEle - i) : (size_t)block_size;
            if (current_block_size == 0) continue;

            float *dst = newData + i;

            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);
            ori_prior = szp_dequantize<float>(prior, (double)absErrBound);
            dst[0] = ori_prior;

            if (current_block_size > 1)
            {
                bit_count = block_pointer[0];
                block_pointer++;

                if (bit_count == 0)
                {
                    for (j = 0; j < current_block_size - 1; j++)
                        dst[1 + j] = ori_prior;
                }
                else
                {
                    convertByteArray2IntArray_fast_1b_args(current_block_size - 1, block_pointer, (current_block_size - 2) / 8 + 1, temp_sign_arr);
                    block_pointer += ((current_block_size - 2) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size - 1, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < current_block_size - 1; j++)
                    {
                        diff = (temp_sign_arr[j] == 0) ? (int)temp_predict_arr[j] : -(int)temp_predict_arr[j];
                        current = prior + diff;
                        ori_current = szp_dequantize<float>(current, (double)absErrBound);
                        prior = current;
                        dst[1 + j] = ori_current;
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

    /* Use block-based distribution matching the compressor */
    size_t num_blocks = (nbEle + block_size - 1) / block_size;

#pragma omp parallel num_threads(nbThreads)
    {
        int tid = omp_get_thread_num();
        size_t blocks_per_thread = (num_blocks + nbThreads - 1) / nbThreads;
        size_t start_block = tid * blocks_per_thread;
        size_t end_block = (tid + 1) * blocks_per_thread;
        if (end_block > num_blocks) end_block = num_blocks;

        size_t j = 0;
        int prior = 0;
        int current = 0;
        int diff = 0;
        unsigned int bit_count = 0;
        unsigned char *block_pointer = rcp + szp_read_offset((const unsigned char *)offsets, tid);
        float ori_prior = 0.0;
        float ori_current = 0.0;

        unsigned char *temp_sign_arr = (unsigned char *)malloc((block_size-1) * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc((block_size-1) * sizeof(unsigned int));
        if (!temp_sign_arr || !temp_predict_arr) { fprintf(stderr, "szpd_float: malloc failed\n"); exit(EXIT_FAILURE); }
        unsigned int savedbitsbytelength = 0;

        for (size_t block_idx = start_block; block_idx < end_block; block_idx++)
        {
            size_t i = block_idx * block_size;
            if (i >= nbEle) break;
            size_t current_block_size = (i + block_size > nbEle) ? (nbEle - i) : (size_t)block_size;
            if (current_block_size == 0) continue;

            float *dst = newData + i;

            memcpy(&prior, block_pointer, sizeof(int));
            block_pointer += sizeof(unsigned int);
            ori_prior = szp_dequantize<float>(prior, (double)absErrBound);
            dst[0] = ori_prior;

            if (current_block_size > 1)
            {
                bit_count = block_pointer[0];
                block_pointer++;

                if (bit_count == 0)
                {
                    for (j = 0; j < current_block_size - 1; j++)
                        dst[1 + j] = ori_prior;
                }
                else
                {
                    convertByteArray2IntArray_fast_1b_args(current_block_size - 1, block_pointer, (current_block_size - 2) / 8 + 1, temp_sign_arr);
                    block_pointer += ((current_block_size - 2) / 8 + 1);

                    savedbitsbytelength = Jiajun_extract_fixed_length_bits(block_pointer, current_block_size - 1, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    for (j = 0; j < current_block_size - 1; j++)
                    {
                        diff = (temp_sign_arr[j] == 0) ? (int)temp_predict_arr[j] : -(int)temp_predict_arr[j];
                        current = prior + diff;
                        ori_current = szp_dequantize<float>(current, (double)absErrBound);
                        prior = current;
                        dst[1 + j] = ori_current;
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

void szp_float_decompress_openmp_threadblock_randomaccess_topology_preserved_sized(
    float **newData, size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes, size_t cmpSize, int **FN)
{
#ifdef _OPENMP
    *newData = (float *)malloc(sizeof(float) * nbEle);
    *FN      = (int   *)calloc(nbEle, sizeof(int));
    if (!*newData || !*FN || !cmpBytes || blockSize <= 0) {
        free(*newData); free(*FN);
        *newData = NULL; *FN = NULL;
        return;
    }

    const float scale = 1 * absErrBound;
    const unsigned int block_size     = (unsigned int)blockSize;
    const unsigned int new_block_size = (block_size > 1) ? block_size - 1 : 1;
    const size_t num_blocks = (nbEle + block_size - 1) / block_size;

    /* cmpSize == 0 means "unknown"; detection is then less strict. */
    unsigned int nbThreads = szp_detect_nbThreads_randomaccess_topo_sized(cmpBytes, cmpSize, nbEle, blockSize);
    if (nbThreads == 0) nbThreads = 1;
    const unsigned char *rcp = cmpBytes + (size_t)nbThreads * sizeof(size_t);
    float *out = *newData;
    int *fn = *FN;

    /* One loop iteration per compressing thread's segment. The decoder's own
       thread count does not have to match the compressor's. */
#pragma omp parallel
    {
        unsigned char *temp_sign_arr = (unsigned char *)malloc(new_block_size);
        unsigned int  *temp_predict_arr = (unsigned int *)malloc(new_block_size * sizeof(unsigned int));
        if (!temp_sign_arr || !temp_predict_arr) { fprintf(stderr, "szpd_float: malloc failed\n"); exit(EXIT_FAILURE); }

        #pragma omp for schedule(static)
        for (long long seg = 0; seg < (long long)nbThreads; seg++) {
            size_t start_block, end_block;
            szp_thread_block_range(num_blocks, nbThreads, (unsigned int)seg, &start_block, &end_block);
            const unsigned char *block_pointer = rcp + szp_read_offset(cmpBytes, (size_t)seg);

            for (size_t block_idx = start_block; block_idx < end_block; block_idx++) {
                size_t i = block_idx * block_size;
                if (i >= nbEle) break;
                size_t current_block_size = (i + block_size > nbEle) ? (nbEle - i) : block_size;
                unsigned int actual_new_block_size = (current_block_size > 1) ? (unsigned int)(current_block_size - 1) : 0;
                float *dst = out + i;

                int prior = 0;
                memcpy(&prior, block_pointer, sizeof(int));
                block_pointer += sizeof(int);
                float ori = szp_dequantize<float>(prior, (double)scale);
                *dst++ = ori;

                unsigned int bit_count = *block_pointer++;
                if (bit_count == 0) {
                    for (unsigned int j = 0; j < actual_new_block_size; j++) dst[j] = ori;
                } else {
                    const unsigned int signbytes = (actual_new_block_size + 7) / 8;
                    convertByteArray2IntArray_fast_1b_args(actual_new_block_size, (unsigned char *)block_pointer, signbytes, temp_sign_arr);
                    block_pointer += signbytes;
                    const unsigned int savedbitsbytelength =
                        Jiajun_extract_fixed_length_bits((unsigned char *)block_pointer, actual_new_block_size, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    /* Prefix sum: no "omp simd", prior carries a dependence. */
                    for (unsigned int j = 0; j < actual_new_block_size; j++) {
                        int diff = temp_sign_arr[j] ? -(int)temp_predict_arr[j] : (int)temp_predict_arr[j];
                        prior += diff;
                        dst[j] = szp_dequantize<float>(prior, (double)scale);
                    }
                }

                /* 2-bit types, 4 per byte, first element in the high bits
                   (same layout as convertIntArray2ByteArray_fast_2b). Unpacked
                   in place instead of allocating a buffer for every block. */
                const unsigned int typebytelength = (unsigned int)((2 * current_block_size + 7) / 8);
                for (size_t j = 0; j < current_block_size; j++)
                    fn[i + j] = (block_pointer[j >> 2] >> (6 - 2 * (j & 3))) & 0x3;
                block_pointer += typebytelength;
            }
        }

        free(temp_sign_arr);
        free(temp_predict_arr);
    }
#else
    (void)newData; (void)nbEle; (void)absErrBound; (void)blockSize; (void)cmpBytes; (void)cmpSize; (void)FN;
#endif
}

void szp_float_decompress_openmp_threadblock_randomaccess_topology_preserved(
    float **newData, size_t nbEle, float absErrBound, int blockSize,
    unsigned char *cmpBytes, int **FN)
{
    szp_float_decompress_openmp_threadblock_randomaccess_topology_preserved_sized(
        newData, nbEle, absErrBound, blockSize, cmpBytes, 0, FN);
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
int *szp_decompress_sort_positions_sized(unsigned char *cmpBytes, size_t cmpSize, size_t critical_count, int blockSize) {
#ifdef _OPENMP
    if (!cmpBytes || critical_count == 0 || blockSize <= 0) {
        return NULL;
    }

    int *newData = (int *)malloc(sizeof(int) * critical_count);
    if (!newData) {
        return NULL;
    }

    const size_t block_size = (size_t)blockSize;
    const size_t num_blocks = (critical_count + block_size - 1) / block_size;
    const size_t temp_arr_size = (block_size > 1) ? (block_size - 1) : 1;

    unsigned int nbThreads = szp_detect_nbThreads_randomaccess_sized(cmpBytes, cmpSize, critical_count, blockSize);
    if (nbThreads == 0) nbThreads = 1;
    const unsigned char *rcp = cmpBytes + (size_t)nbThreads * sizeof(size_t);

#pragma omp parallel
    {
        unsigned char *temp_sign_arr = (unsigned char *)malloc(temp_arr_size * sizeof(unsigned char));
        unsigned int *temp_predict_arr = (unsigned int *)malloc(temp_arr_size * sizeof(unsigned int));
        if (!temp_sign_arr || !temp_predict_arr) { fprintf(stderr, "szpd_float: malloc failed\n"); exit(EXIT_FAILURE); }

        #pragma omp for schedule(static)
        for (long long seg = 0; seg < (long long)nbThreads; seg++) {
            size_t start_block, end_block;
            szp_thread_block_range(num_blocks, nbThreads, (unsigned int)seg, &start_block, &end_block);
            const unsigned char *block_pointer = rcp + szp_read_offset(cmpBytes, (size_t)seg);

            for (size_t block_idx = start_block; block_idx < end_block; block_idx++) {
                size_t i = block_idx * block_size;
                if (i >= critical_count) break;
                size_t current_block_size = (i + block_size > critical_count) ? (critical_count - i) : block_size;
                int *dst = newData + i;

                int prior = 0;
                memcpy(&prior, block_pointer, sizeof(int));
                block_pointer += sizeof(int);
                *dst++ = prior;

                /* The encoder always writes the bit-count byte. */
                unsigned int bit_count = *block_pointer++;
                if (current_block_size < 2) continue;
                if (bit_count == 0) {
                    for (size_t j = 0; j < current_block_size - 1; j++) dst[j] = prior;
                } else {
                    const size_t n = current_block_size - 1;
                    const size_t signbytes = (n + 7) / 8;
                    convertByteArray2IntArray_fast_1b_args(n, (unsigned char *)block_pointer, signbytes, temp_sign_arr);
                    block_pointer += signbytes;
                    unsigned int savedbitsbytelength =
                        Jiajun_extract_fixed_length_bits((unsigned char *)block_pointer, (unsigned int)n, temp_predict_arr, bit_count);
                    block_pointer += savedbitsbytelength;
                    /* Prefix sum: no "omp simd", prior carries a dependence. */
                    for (size_t j = 0; j < n; j++) {
                        int diff = temp_sign_arr[j] ? -(int)temp_predict_arr[j] : (int)temp_predict_arr[j];
                        prior += diff;
                        dst[j] = prior;
                    }
                }
            }
        }

        free(temp_sign_arr);
        free(temp_predict_arr);
    }

    return newData;
#else
    (void)cmpBytes; (void)cmpSize; (void)critical_count; (void)blockSize;
    return NULL;
#endif
}

int *szp_decompress_sort_positions(unsigned char *cmpBytes, size_t critical_count, int blockSize) {
    return szp_decompress_sort_positions_sized(cmpBytes, 0, critical_count, blockSize);
}
