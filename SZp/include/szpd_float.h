/**
 *  @file szpd_Float.h
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Oct, 2023
 */

#ifndef _szpd_Float_H
#define _szpd_Float_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include "szp_defines.h"

#ifdef __cplusplus
extern "C" {
#endif

void szp_float_decompress_openmp_threadblock_randomaccess_topology_preserved(float **newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes, int **FN);

/* Same as above, with the exact size of the topology stream in bytes
   (0 = unknown). Knowing the size makes thread-count detection exact. */
void szp_float_decompress_openmp_threadblock_randomaccess_topology_preserved_sized(float **newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes, size_t cmpSize, int **FN);

int *szp_decompress_sort_positions(unsigned char *cmpBytes, size_t critical_count, int blockSize);

/* cmpBytes points just past the extrema-count header; cmpSize is the number
   of bytes from there to the end of the sort-position stream (0 = unknown). */
int *szp_decompress_sort_positions_sized(unsigned char *cmpBytes, size_t cmpSize, size_t critical_count, int blockSize);

float *szp_float_decompress_openmp_threadblock(size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);

void szp_float_decompress_openmp_threadblock_arg(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);

void szp_float_decompress_single_thread_arg(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);

size_t szp_float_decompress_single_thread_arg_record(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);

float *szp_float_decompress_openmp_threadblock_randomaccess(size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);

void szp_float_decompress_openmp_threadblock_randomaccess_arg(float *newData, size_t nbEle, float absErrBound, int blockSize, unsigned char *cmpBytes);

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _szpd_Float_H  ----- */
