/**
 *  @file szp_Float.h
 *  @author Jiajun Huang, Sheng Di
 *  @date Oct, 2023
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "szp.h"
#ifdef _OPENMP
#include "omp.h"
#endif
struct timeval startTime;
struct timeval endTime;   /* Start and end times */
struct timeval costStart; /*only used for recording the cost*/
double totalCost = 0;

void cost_start()
{
  totalCost = 0;
  gettimeofday(&costStart, NULL);
}

void cost_end()
{
  double elapsed;
  struct timeval costEnd;
  gettimeofday(&costEnd, NULL);
  elapsed = ((costEnd.tv_sec * 1000000 + costEnd.tv_usec) - (costStart.tv_sec * 1000000 + costStart.tv_usec)) / 1000000.0;
  totalCost += elapsed;
}

int main(int argc, char *argv[])
{
  char oriFilePath[640], outputFilePath[645];
  if (argc < 6)
  {
    printf("Usage: testfloat_compress_fastmode1 [srcFilePath] [block size] [err bound] [rows] [cols]\n");
    printf("Example: testfloat_compress_fastmode1 testfloat_8_8_128.dat 64 1E-3 500 500\n");
    exit(0);
  }

  sprintf(oriFilePath, "%s", argv[1]);
  int blockSize = atoi(argv[2]);
  float errBound = atof(argv[3]);
  int rows = atoi(argv[4]);
  int cols = atoi(argv[5]);
  sprintf(outputFilePath, "%s.SZp", oriFilePath);

  int status = 0;
  size_t nbEle;
  float *data = szp_readFloatData(oriFilePath, &nbEle, &status);
  if (status != SZ_SCES)
  {
    printf("Error: data file %s cannot be read!\n", oriFilePath);
    exit(0);
  }

  
  size_t critical_count;
  size_t outSize;
  size_t sort_positions_size;
  cost_start();
  CriticalPoint *critical_points = szp_find_critical_points(data, &critical_count, rows, cols, errBound);
  if (!critical_points || critical_count == 0) {
    printf("Error: No critical points found!\n");
    free(data);
    exit(1);
  }
  //critical points have a quantized bin value. for each critical point falling in the same bin, we sort them by their original data values
  szp_sort_critical_points_by_original_data(critical_points, critical_count, data, cols);

  unsigned char *bytes_sort_positions = szp_compress_sort_positions(critical_points, critical_count, &sort_positions_size, blockSize);
  if (!bytes_sort_positions || sort_positions_size == 0) {
    printf("Error: Failed to compress sort positions or size is zero!\n");
    free(critical_points);
    free(data);
    if (bytes_sort_positions) free(bytes_sort_positions);
    exit(1);
  }
  
  unsigned char *bytes_topology = szp_float_openmp_threadblock_randomaccess_topology_preserved(
    data, &outSize, errBound, nbEle, blockSize, critical_points, critical_count, rows, cols);
  if (!bytes_topology || outSize == 0) {
    printf("Error: Failed to compress topology or size is zero!\n");
    free(critical_points);
    free(data);
    free(bytes_sort_positions);
    if (bytes_topology) free(bytes_topology);
    exit(1);
  }
  
  // Append sort positions to topology bytes + footer with sizes
  // Footer format: [topology_size (size_t)] [sort_positions_size (size_t)]
  size_t footer_size = 2 * sizeof(size_t);
  size_t total_size = outSize + sort_positions_size + footer_size;
  if (total_size < outSize || total_size < sort_positions_size) {
    printf("Error: Integer overflow in size calculation!\n");
    free(critical_points);
    free(data);
    free(bytes_topology);
    free(bytes_sort_positions);
    exit(1);
  }
  
  unsigned char *combined_bytes = (unsigned char *)malloc(total_size);
  unsigned char *original_bytes_topology = NULL;
  if (combined_bytes) {
    memcpy(combined_bytes, bytes_topology, outSize);
    memcpy(combined_bytes + outSize, bytes_sort_positions, sort_positions_size);
    // Footer: write topology_size and sort_positions_size at the very end
    memcpy(combined_bytes + outSize + sort_positions_size, &outSize, sizeof(size_t));
    memcpy(combined_bytes + outSize + sort_positions_size + sizeof(size_t), &sort_positions_size, sizeof(size_t));
    
    // Free original topology bytes before reassigning pointer
    original_bytes_topology = bytes_topology;
    bytes_topology = combined_bytes;
    outSize = total_size;
    free(original_bytes_topology);
  } else {
    printf("Error: Failed to allocate combined bytes buffer (size=%zu)!\n", total_size);
    free(critical_points);
    free(data);
    free(bytes_topology);
    free(bytes_sort_positions);
    exit(1);
  }
  
  cost_end();
  //print the quantized bins
  
  printf("\ntimecost=%f, total fastmode1 topology\n", totalCost);
  printf("compression size = %zu, CR = %f\n", outSize, 1.0f * nbEle * sizeof(float) / outSize);
  
  szp_writeByteData(bytes_topology, outSize, outputFilePath, &status);
  
  if (status != SZ_SCES)
  {
    printf("Error: data file %s cannot be written!\n", outputFilePath);
    exit(0);
  }
  printf("done\n");
  
  free(critical_points);
  free(data);
  if (bytes_topology) {
    free(bytes_topology);  // This points to combined_bytes (if allocation succeeded) or original bytes_topology
  }
  if (bytes_sort_positions) {
    free(bytes_sort_positions);
  }

  return 0;
}
