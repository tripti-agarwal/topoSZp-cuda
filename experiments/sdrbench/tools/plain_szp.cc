// plain_szp <float32 input> <abs error bound> <block size> [output file]
//
// Plain SZp (no topology metadata, no restoration) with the same quantizer
// as TopoSZp+ (szp_quant.h). Used as the "SZp" baseline: compressed size,
// compression and decompression time, and optionally the decompressed field
// (e.g. for the visualization figure).
// Prints "PLAINSZP bytes=<compressed bytes> comp_s=<s> decomp_s=<s> max_err=<abs>".
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include "szp.h"
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: plain_szp <float32 input> <abs eb> <block size> [output]\n"); return 2; }
    size_t n = 0; int status = 0;
    float *data = szp_readFloatData(argv[1], &n, &status);
    if (status != 0 || !data || n == 0) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    float eb = (float)atof(argv[2]);
    int bs = atoi(argv[3]);
    size_t outSize = 0;
    double t0 = omp_get_wtime();
    unsigned char *cmp = szp_float_openmp_threadblock_randomaccess(data, &outSize, eb, n, bs);
    double t1 = omp_get_wtime();
    if (!cmp || outSize == 0) { fprintf(stderr, "compression failed\n"); return 1; }
    // The compressor writes the error bound (4 bytes) in front of the stream; the
    // decompressor takes the stream after it (as example/test_cuda_decompress.cu).
    float *dec = szp_float_decompress_openmp_threadblock_randomaccess(n, eb, bs, cmp + sizeof(float));
    double t2 = omp_get_wtime();
    if (!dec) { fprintf(stderr, "decompression failed\n"); return 1; }
    double me = 0;
    for (size_t i = 0; i < n; i++) { double e = std::fabs((double)data[i] - dec[i]); if (e > me) me = e; }
    if (me > eb) { fprintf(stderr, "error: max error %g exceeds the bound %g\n", me, (double)eb); return 3; }
    if (argc > 4) { szp_writeFloatData_inBytes(dec, n, argv[4], &status); if (status) { fprintf(stderr, "cannot write %s\n", argv[4]); return 1; } }
    printf("PLAINSZP bytes=%zu comp_s=%.6f decomp_s=%.6f max_err=%.9g\n", outSize + sizeof(float), t1 - t0, t2 - t1, me);
    free(cmp); free(dec); free(data);
    return 0;
}
