/**
 *  @file szp_Float.h
 *  @author Jiajun Huang, Sheng Di
 *  @date Oct, 2023
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "szp.h"
#include "szp_topology_3d.h"
#include "szp_pl_topology.h"
#include <string.h>
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
  /* --pl: critical points with the standard PL definition (szp_pl_topology.h)
     instead of the axis rule; the decoder reads the mode from the file. */
  int pl = 0;
  if (argc > 1 && strcmp(argv[1], "--pl") == 0) { pl = 1; argv++; argc--; }
  if (argc < 6)
  {
    printf("Usage: testfloat_compress_fastmode1 [--pl] [srcFilePath] [block size] [err bound] [rows] [cols] [depth]\n");
    printf("  --pl: standard PL (TTK-style) critical points instead of the axis rule\n");
    printf("  2D: rows cols; 3D: d1 d2 d3 (slowest dimension first, flat index x*d2*d3 + y*d3 + z)\n");
    printf("Example: testfloat_compress_fastmode1 testfloat_8_8_128.dat 64 1E-3 500 500\n");
    printf("Example: testfloat_compress_fastmode1 CLDICE_1_26_1800_3600.f32 64 1E-7 26 1800 3600\n");
    exit(0);
  }

  sprintf(oriFilePath, "%s", argv[1]);
  int blockSize = atoi(argv[2]);
  float errBound = atof(argv[3]);
  int rows = atoi(argv[4]);
  int cols = atoi(argv[5]);
  int depth = (argc > 6) ? atoi(argv[6]) : 0;   /* > 0: 3D field rows x cols x depth */
  sprintf(outputFilePath, "%s.SZp", oriFilePath);

  int status = 0;
  size_t nbEle;
  float *data = szp_readFloatData(oriFilePath, &nbEle, &status);
  if (status != SZ_SCES)
  {
    printf("Error: data file %s cannot be read!\n", oriFilePath);
    exit(0);
  }

  // Validate dimensions match file size
  size_t expected_nbEle = (size_t)rows * (size_t)cols * (size_t)(depth > 0 ? depth : 1);
  if (nbEle != expected_nbEle) {
    printf("Error: Dimension mismatch!\n");
    printf("  File contains: %zu elements\n", nbEle);
    printf("  Expected from dimensions (%d x %d x %d): %zu elements\n", rows, cols, depth > 0 ? depth : 1, expected_nbEle);
    printf("  This will cause a segmentation fault. Please check your dimensions.\n");
    free(data);
    exit(1);
  }

  
  size_t critical_count;
  size_t outSize;
  size_t sort_positions_size;
  cost_start();
  void *critical_points = NULL;
  unsigned char *bytes_sort_positions = NULL, *bytes_topology = NULL, *bytes_kinds = NULL;
  size_t kinds_size = 0;
  signed char *pl_codes = NULL;
  if (pl) {
    pl_codes = (signed char *)malloc(nbEle);
    if (!pl_codes) { printf("Error: out of memory\n"); exit(1); }
    szp_pl_classify_float(data, depth > 0 ? 3 : 2, rows, cols, depth, pl_codes);
    if (depth > 0) bytes_kinds = szp_pl_pack_saddle_kinds(pl_codes, nbEle, &kinds_size);
  }
  if (depth > 0) {
    CriticalPoint3D *cps = pl ? szp_pl_critical_points_3d(pl_codes, data, rows, cols, depth, errBound, &critical_count)
                              : szp_find_critical_points_3d(data, &critical_count, rows, cols, depth, errBound);
    critical_points = cps;
    if (cps && critical_count > 0) {
      // rank critical points by original value within each quantization bin
      szp_sort_critical_points_3d(cps, critical_count, data, cols, depth);
      bytes_sort_positions = szp_compress_sort_positions_3d(cps, critical_count, &sort_positions_size, blockSize);
      bytes_topology = szp_float_compress_topology_3d(data, &outSize, errBound, nbEle, blockSize,
                                                      cps, (int)critical_count, rows, cols, depth);
    }
  } else {
    CriticalPoint *cps = pl ? szp_pl_critical_points_2d(pl_codes, data, rows, cols, errBound, &critical_count)
                            : szp_find_critical_points(data, &critical_count, rows, cols, errBound);
    critical_points = cps;
    if (cps && critical_count > 0) {
      // rank critical points by original value within each quantization bin
      szp_sort_critical_points_by_original_data(cps, critical_count, data, cols);
      bytes_sort_positions = szp_compress_sort_positions(cps, critical_count, &sort_positions_size, blockSize);
      bytes_topology = szp_float_openmp_threadblock_randomaccess_topology_preserved(
        data, &outSize, errBound, nbEle, blockSize, cps, critical_count, rows, cols);
    }
  }
  if (!critical_points || critical_count == 0) {
    printf("Error: No critical points found!\n");
    free(data);
    exit(1);
  }
  if (!bytes_sort_positions || sort_positions_size == 0) {
    printf("Error: Failed to compress sort positions or size is zero!\n");
    free(critical_points);
    free(data);
    if (bytes_sort_positions) free(bytes_sort_positions);
    if (bytes_topology) free(bytes_topology);
    exit(1);
  }
  if (!bytes_topology || outSize == 0) {
    printf("Error: Failed to compress topology or size is zero!\n");
    free(critical_points);
    free(data);
    free(bytes_sort_positions);
    if (bytes_topology) free(bytes_topology);
    exit(1);
  }

  // Append sort positions to topology bytes + footer with sizes
  // Footer format: [topology_size (size_t)] [sort_positions_size (size_t)];
  // in PL mode the v2 container of szp_pl_topology.h (adds saddle kinds,
  // flags and a magic number).
  size_t footer_size = pl ? 5 * sizeof(size_t) : 2 * sizeof(size_t);
  size_t total_size = outSize + sort_positions_size + kinds_size + footer_size;
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
    unsigned char *foot = combined_bytes + outSize + sort_positions_size;
    if (pl) {
      if (kinds_size) memcpy(foot, bytes_kinds, kinds_size);
      foot += kinds_size;
      size_t f[5] = {outSize, sort_positions_size, kinds_size, (size_t)SZP_CONTAINER_FLAG_PL,
                     (size_t)SZP_CONTAINER_MAGIC_V2};
      memcpy(foot, f, sizeof(f));
    } else {
      // Footer: write topology_size and sort_positions_size at the very end
      memcpy(foot, &outSize, sizeof(size_t));
      memcpy(foot + sizeof(size_t), &sort_positions_size, sizeof(size_t));
    }
    
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
  
  printf("\ncompression time=%f, total fastmode1 topology%s\n", totalCost, pl ? " (PL critical points)" : "");
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
  free(pl_codes);
  free(bytes_kinds);
  if (bytes_topology) {
    free(bytes_topology);  // This points to combined_bytes (if allocation succeeded) or original bytes_topology
  }
  if (bytes_sort_positions) {
    free(bytes_sort_positions);
  }

  return 0;
}
