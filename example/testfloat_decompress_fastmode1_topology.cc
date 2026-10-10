/**
 *  @file testfloat_decompress_fastmode1.c
 *  @author Sheng Di
 *  @date April, 2015
 *  @brief This is an example of using Decompression interface.
 *  (C) 2015 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include "szp.h"
#include "szpd_float.h"
#include "szp_topo_restore.h"
#include "szp_pl_topology.h"
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
    size_t nbEle, totalNbEle;
    char zipFilePath[640], outputFilePath[645];
    if (argc < 7)
    {
        printf("Usage: testfloat_decompress_fastmode1 [srcFilePath] [nbEle] [block size] [err bound] [rows] [cols] [depth]\n");
        printf("  2D: rows cols; 3D: d1 d2 d3 (same order as for compression)\n");
        printf("Example: testfloat_decompress_fastmode1 testfloat_8_8_128.dat.SZp 250000 64 1E-3 500 500\n");
        exit(0);
    }

    sprintf(zipFilePath, "%s", argv[1]);
    nbEle = strtoimax(argv[2], NULL, 10);
    sprintf(outputFilePath, "%s.out", zipFilePath);
    int blockSize = atoi(argv[3]);
    float errBound = atof(argv[4]);
    int rows = atoi(argv[5]);
    int cols = atoi(argv[6]);
    int depth = (argc > 7) ? atoi(argv[7]) : 0;   /* > 0: 3D field rows x cols x depth */
    if ((size_t)rows * (size_t)cols * (size_t)(depth > 0 ? depth : 1) != nbEle) {
        printf("Error: nbEle %zu does not match the dimensions\n", nbEle);
        exit(1);
    }

    size_t byteLength;
    int status;
    unsigned char *bytes = szp_readByteData(zipFilePath, &byteLength, &status);
    if (status != SZ_SCES)
    {
        printf("Error: %s cannot be read!\n", zipFilePath);
        exit(0);
    }
    float *data = NULL;
    int *critical_points_types = NULL;
    int *sort_positions = NULL;
    size_t extrema_count = 0;
    cost_start();

    // Footer: [topology_size (size_t)] [sort_positions_size (size_t)]. The
    // compressor writes [topology][sort positions][footer], so the sizes
    // must add up to the file size exactly; otherwise the whole file is
    // treated as the topology stream (old format).
    // In PL mode the compressor writes the v2 container (szp_pl_topology.h):
    // [topology][sort positions][saddle kinds][sizes x3][flags][magic].
    size_t topology_size = byteLength, sort_positions_size = 0, kinds_size = 0;
    int pl = 0;
    const size_t footer_size = 2 * sizeof(size_t);
    if (byteLength >= 5 * sizeof(size_t)) {
        size_t f[5];
        memcpy(f, bytes + byteLength - sizeof(f), sizeof(f));
        if (f[4] == (size_t)SZP_CONTAINER_MAGIC_V2 && f[0] > 0 &&
            f[0] + f[1] + f[2] + sizeof(f) == byteLength) {
            topology_size = f[0]; sort_positions_size = f[1]; kinds_size = f[2];
            pl = (f[3] & SZP_CONTAINER_FLAG_PL) != 0;
        }
    }
    if (topology_size == byteLength && byteLength >= footer_size) {
        size_t ts = 0, ss = 0;
        memcpy(&ts, bytes + byteLength - footer_size, sizeof(size_t));
        memcpy(&ss, bytes + byteLength - sizeof(size_t), sizeof(size_t));
        if (ts > 0 && ts < byteLength && ts + ss + footer_size == byteLength) {
            topology_size = ts;
            sort_positions_size = ss;
        }
    }

    // The decoder may read a little past the last block; give it zeroed padding.
    const size_t padding_size = 10240;
    unsigned char *padded_topology = (unsigned char *)malloc(topology_size + padding_size);
    if (!padded_topology) {
        printf("Error: Failed to allocate padded topology buffer (size=%zu)!\n", topology_size + padding_size);
        free(bytes);
        exit(1);
    }
    memcpy(padded_topology, bytes, topology_size);
    memset(padded_topology + topology_size, 0, padding_size);

    // Passing the exact stream size (when the footer gave one) makes the
    // thread-count detection exact.
    szp_float_decompress_openmp_threadblock_randomaccess_topology_preserved_sized(
        &data, nbEle, errBound, blockSize, padded_topology,
        (sort_positions_size > 0) ? topology_size : 0, &critical_points_types);
    free(padded_topology);
    if (!data || !critical_points_types) {
        printf("Error: Decompression failed\n");
        free(bytes);
        exit(1);
    }

    // Ranks of the stored extrema: [extrema_count (size_t)] [per-thread offsets] [payload]
    if (sort_positions_size >= sizeof(size_t)) {
        unsigned char *sort_start = bytes + topology_size;
        memcpy(&extrema_count, sort_start, sizeof(size_t));
        sort_positions = szp_decompress_sort_positions_sized(
            sort_start + sizeof(size_t), sort_positions_size - sizeof(size_t),
            extrema_count, blockSize);
        if (!sort_positions) {
            printf("Warning: sort positions decompression failed; extrema are restored without ranks\n");
            extrema_count = 0;
        }
    }

    // Topology restoration (library, szp_topo_restore.h): extrema with their
    // ranks, then saddles. Every value stays within errBound of its
    // dequantized value.
    szp_topo_restore_stats rs;
    if (pl) {
        // Stored types are 2-bit (3 = saddle); the 3D saddle kinds follow
        // the sort positions.
        if (depth > 0 &&
            szp_pl_apply_saddle_kinds(critical_points_types, nbEle,
                                      bytes + topology_size + sort_positions_size, kinds_size) != 0)
            printf("Warning: saddle kinds missing or inconsistent; 3D saddles are restored as 1-saddles\n");
        szp_topo_restore_float_pl(data, critical_points_types, sort_positions, extrema_count,
                                  depth > 0 ? 3 : 2, rows, cols, depth, errBound, &rs);
    } else if (depth > 0)
        szp_topo_restore_float_3d(data, critical_points_types, sort_positions, extrema_count,
                                  rows, cols, depth, errBound, &rs);
    else
        szp_topo_restore_float_2d(data, critical_points_types, sort_positions, extrema_count,
                                  rows, cols, errBound, &rs);

    cost_end();
    free(bytes);
    printf("decompression time=%f\n", totalCost);
    if (pl)
        printf("topology restoration (PL): extrema lost %zu -> %zu of %zu; saddles lost %zu -> %zu of %zu; "
               "false positives %zu -> %zu, false types %zu -> %zu (saddles fixed %zu, other fixes %zu, %zu rounds)\n",
               rs.extrema_lost_before, rs.extrema_lost_after, rs.extrema,
               rs.saddles_lost_before, rs.saddles_lost_after, rs.saddles,
               rs.fp_before, rs.fp_after, rs.ft_before, rs.ft_after,
               rs.saddles_by_stencil, rs.other_fixes, rs.rounds);
    else
        printf("topology restoration: extrema lost %zu -> %zu of %zu; saddles lost %zu -> %zu of %zu "
               "(stencil %zu, RBF %zu)\n",
               rs.extrema_lost_before, rs.extrema_lost_after, rs.extrema,
               rs.saddles_lost_before, rs.saddles_lost_after, rs.saddles,
               rs.saddles_by_stencil, rs.saddles_by_rbf);
    szp_writeFloatData_inBytes(data, nbEle, outputFilePath, &status);
    if (status != SZ_SCES)
    {
        printf("Error: %s cannot be written!\n", outputFilePath);
        exit(0);
    }
    printf("done\n");

    char oriFilePath[645];
    strcpy(oriFilePath, zipFilePath);
    oriFilePath[strlen(zipFilePath) - 4] = '\0';
    float *ori_data = szp_readFloatData(oriFilePath, &totalNbEle, &status);
    if (status != SZ_SCES)
    {
        printf("Error: %s cannot be read!\n", oriFilePath);
        exit(0);
    }

    size_t i = 0;
    float Max = 0, Min = 0, diffMax = 0;
    Max = ori_data[0];
    Min = ori_data[0];
    diffMax = fabs(data[0] - ori_data[0]);
    double sum1 = 0, sum2 = 0;
    for (i = 0; i < nbEle; i++)
    {
        sum1 += ori_data[i];
        sum2 += data[i];
    }
    double mean1 = sum1 / nbEle;
    double mean2 = sum2 / nbEle;

    double sum3 = 0, sum4 = 0;
    double sum = 0, prodSum = 0, relerr = 0;

    double maxpw_relerr = 0;
    for (i = 0; i < nbEle; i++)
    {
        if (Max < ori_data[i])
            Max = ori_data[i];
        if (Min > ori_data[i])
            Min = ori_data[i];

        float err = fabs(data[i] - ori_data[i]);
        if (ori_data[i] != 0)
        {
            if (fabs(ori_data[i]) > 1)
                relerr = err / ori_data[i];
            else
                relerr = err;
            if (maxpw_relerr < relerr)
                maxpw_relerr = relerr;
        }

        /*if(err > 1600000)
        {
            printf("i=%zu, ori=%f, dec=%f, diff=%f\n", i, ori_data[i], data[i], err);
            exit(0);
        }*/
        if (diffMax < err)
            diffMax = err;
        prodSum += (ori_data[i] - mean1) * (data[i] - mean2);
        sum3 += (ori_data[i] - mean1) * (ori_data[i] - mean1);
        sum4 += (data[i] - mean2) * (data[i] - mean2);
        sum += err * err;
    }
    double std1 = sqrt(sum3 / nbEle);
    double std2 = sqrt(sum4 / nbEle);
    double ee = prodSum / nbEle;
    double acEff = ee / std1 / std2;

    double mse = sum / nbEle;
    double range = Max - Min;
    double psnr = 20 * log10(range) - 10 * log10(mse);
    double nrmse = sqrt(mse) / range;

    double compressionRatio = 1.0 * nbEle * sizeof(float) / byteLength;

    printf("Min=%.20G, Max=%.20G, range=%.20G\n", Min, Max, range);
    printf("Max absolute error = %.10f\n", diffMax);
    printf("Max relative error = %f\n", diffMax / (Max - Min));
    printf("Max pw relative error = %f\n", maxpw_relerr);
    printf("PSNR = %f, NRMSE= %.20G\n", psnr, nrmse);
    printf("acEff=%f\n", acEff);
    printf("compressionRatio = %f\n", compressionRatio);


    

    free(data);
    free(critical_points_types);
    free(sort_positions);
    free(ori_data);
    return 0;
}
