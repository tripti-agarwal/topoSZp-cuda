/**
 *  @file szp_quant.h
 *  @brief The one quantization rule shared by every SZp/TopoSZp compressor
 *         and decompressor (CPU and CUDA, float and double data).
 *
 *  Bin index of value v with absolute error bound eb:
 *      q = trunc(v / eb), then moved by one bin if the reconstruction of q
 *          would be more than eb away from v (this only happens at bin edges
 *          because of rounding).
 *  Reconstruction of bin q for data type T:
 *      float  data: (float)(q * eb)   (product computed in double, rounded once)
 *      double data: q * eb            (double)
 *
 *  Because the compressor checks the decoder's exact reconstruction, every
 *  decompressed value satisfies |v - v'| <= eb for both data types. The rule
 *  is monotone (v1 <= v2 gives q1 <= q2), which the topology guarantees
 *  (no false positives, no false types) rely on.
 *
 *  Limits that depend on the data type (checked by szp_check_errbound):
 *   - the format stores q as a 32-bit int, so max|v| / eb must stay below 2^31;
 *   - topology correction moves values by less than eb, so it cannot
 *     separate points when the spacing of representable values of T at
 *     max|v| (float or double spacing) is larger than eb.
 *
 *  Device code uses __dmul_rn so no fused multiply-add changes the result:
 *  the CPU and GPU paths produce identical bins and values.
 */
#ifndef _SZP_QUANT_H
#define _SZP_QUANT_H

#ifdef __cplusplus

#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define SZP_QHD __host__ __device__ __forceinline__
#else
#define SZP_QHD inline
#endif

static SZP_QHD double szp_qmul(double a, double b)
{
#if defined(__CUDA_ARCH__)
    return __dmul_rn(a, b);
#else
    return a * b;
#endif
}

/* Arithmetic that rounds every operation on both host and device, so the
   compiler cannot fuse a multiply and an add on the GPU (which would change
   results compared with the CPU). */
static SZP_QHD double szp_dadd(double a, double b)
{
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a, b);
#else
    return a + b;
#endif
}
static SZP_QHD float szp_fmul(float a, float b)
{
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
static SZP_QHD float szp_fadd(float a, float b)
{
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}

/** Reconstruction of bin q (eb is the absolute bound, as double). */
template <typename T>
SZP_QHD T szp_dequantize(int q, double eb)
{
    return (T)szp_qmul((double)q, eb);
}

/** Bin index of v; inv = 1.0 / eb, both as double. */
template <typename T>
SZP_QHD int szp_quantize(T v, double eb, double inv)
{
    int q = (int)szp_qmul((double)v, inv);
    double e = (double)v - (double)szp_dequantize<T>(q, eb);
    if (e > eb) q += 1;
    else if (e < -eb) q -= 1;
    return q;
}

/**
 * Check an absolute error bound against the data before compressing.
 * Returns 0 if fine, 1 if usable with a warning (printed), -1 if the bound
 * cannot be used (printed). `who` names the caller in the message.
 */
template <typename T>
static int szp_check_errbound(const T *data, size_t n, double eb, const char *who)
{
    const char *tname = sizeof(T) == sizeof(float) ? "float32" : "float64";
    if (!(eb > 0) || !isfinite(eb)) {
        fprintf(stderr, "%s: error bound %g must be a positive number\n", who, eb);
        return -1;
    }
    T amax = 0;
    size_t bad = 0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(max:amax) reduction(+:bad)
#endif
    for (long long i = 0; i < (long long)n; i++) {
        T a = data[i] < 0 ? -data[i] : data[i];
        if (!(a == a) || a > (T)DBL_MAX) { bad++; continue; }   /* NaN or Inf */
        if (a > amax) amax = a;
    }
    if (bad) {
        fprintf(stderr, "%s: %zu values are NaN or Inf; they cannot be compressed with an error bound\n", who, bad);
        return -1;
    }
    if ((double)amax / eb >= 2147483520.0) {
        fprintf(stderr, "%s: max |value| %g / error bound %g exceeds the 32-bit bin range of the format "
                        "(use eb >= %g for this %s data)\n", who, (double)amax, eb,
                (double)amax / 2147483520.0, tname);
        return -1;
    }
    T next = (T)nextafter((double)amax, INFINITY);
    if (sizeof(T) == sizeof(float)) next = (T)nextafterf((float)amax, INFINITY);
    double spacing = (double)next - (double)amax;
    if (spacing > eb) {
        fprintf(stderr, "%s: warning: %s spacing at max |value| %g is %g, larger than the error bound %g; "
                        "values are kept within the bound but topology correction cannot separate points "
                        "(use eb >= %g)\n", who, tname, (double)amax, spacing, eb, spacing);
        return 1;
    }
    return 0;
}

#endif /* __cplusplus */
#endif /* _SZP_QUANT_H */
