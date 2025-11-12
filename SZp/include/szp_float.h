/**
 *  @file szp_Float.h
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Oct, 2023
 */

#ifndef _szp_Float_H
#define _szp_Float_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include "szp_defines.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int x, y;
    int type;  // 1 for max, 2 for min, 3 for saddle //0 for regular we don't need to save those
    int quantized_bin;
    int sort_position;  // Position within the bin (1, 2, 3, etc.)
} CriticalPoint;

CriticalPoint *szp_find_critical_points(float *data, size_t *outCount, int rows, int cols, float absErrBound);

void szp_sort_critical_points_by_original_data(CriticalPoint *critical_points, size_t critical_count, 
                                             float *data, int cols);

unsigned char *szp_compress_sort_positions(CriticalPoint *critical_points, size_t critical_count, size_t *outSize, int blockSize);

unsigned char *szp_float_openmp_threadblock_randomaccess_topology_preserved(float *oriData, size_t *outSize, float absErrBound,
    size_t nbEle, int blockSize, CriticalPoint *critical_points, int critical_count, int rows, int cols);

 
int *
szp_float_openmp_direct_predict_quantization(float *oriData, size_t *outSize, float absErrBound,
                                             size_t nbEle, int blockSize);

int *
szp_float_openmp_threadblock_predict_quantization(float *oriData, size_t *outSize, float absErrBound,
                                                  size_t nbEle, int blockSize);

unsigned char *
szp_float_openmp_threadblock(float *oriData, size_t *outSize, float absErrBound,
                             size_t nbEle, int blockSize);

void szp_float_openmp_threadblock_arg(unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
                                      size_t nbEle, int blockSize);

void szp_float_single_thread_arg(unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
                                 size_t nbEle, int blockSize);

size_t szp_float_single_thread_arg_record(unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
                                       size_t nbEle, int blockSize);

unsigned char *
szp_float_openmp_threadblock_randomaccess(float *oriData, size_t *outSize, float absErrBound,
                                          size_t nbEle, int blockSize);

void
szp_float_openmp_threadblock_randomaccess_arg(unsigned char *output, float *oriData, size_t *outSize, float absErrBound,
                                          size_t nbEle, int blockSize);

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _szp_Float_H  ----- */
