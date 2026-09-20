/**
 *  @file szp_errbound.h
 *  @brief Error bound mode utilities: convert relative error bound to absolute.
 *
 *  Supports two modes:
 *    ABS: user provides absolute error bound directly
 *    REL: user provides relative error bound (fraction of data range)
 *         absErrBound = relErrBound × (max_value - min_value)
 */

#ifndef _SZP_ERRBOUND_H
#define _SZP_ERRBOUND_H

#include <stddef.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute absolute error bound from relative error bound.
 * absErrBound = relErrBound × (max - min)
 *
 * @param data       input data array
 * @param nbEle      number of elements
 * @param relBound   relative error bound (e.g., 1e-4 = 0.01% of range)
 * @param[out] absOut  computed absolute error bound
 * @param[out] rangeOut  data range (max - min), can be NULL
 * @return 0 on success, -1 on error
 */
static inline int szp_rel_to_abs_float(const float *data, size_t nbEle,
                                        float relBound,
                                        float *absOut, float *rangeOut)
{
    if (!data || nbEle == 0 || relBound <= 0) return -1;

    float mn = data[0], mx = data[0];
    for (size_t i = 1; i < nbEle; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    float range = mx - mn;
    if (range <= 0) range = FLT_EPSILON;

    *absOut = relBound * range;
    if (rangeOut) *rangeOut = range;
    return 0;
}

static inline int szp_rel_to_abs_double(const double *data, size_t nbEle,
                                         double relBound,
                                         double *absOut, double *rangeOut)
{
    if (!data || nbEle == 0 || relBound <= 0) return -1;

    double mn = data[0], mx = data[0];
    for (size_t i = 1; i < nbEle; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    double range = mx - mn;
    if (range <= 0) range = 1e-15;

    *absOut = relBound * range;
    if (rangeOut) *rangeOut = range;
    return 0;
}

/**
 * Parse error bound argument: "1e-3" for absolute, "rel:1e-4" for relative.
 * For relative, requires data to compute the range.
 *
 * @param arg        command-line argument string
 * @param data       input data (needed for relative mode)
 * @param nbEle      number of elements
 * @param[out] absOut  computed absolute error bound
 * @param[out] modeOut "ABS" or "REL" string pointer
 * @return 0 on success
 */
static inline int szp_parse_errbound_float(const char *arg,
                                            const float *data, size_t nbEle,
                                            float *absOut, const char **modeOut)
{
    if (!arg) return -1;

    if (strncmp(arg, "rel:", 4) == 0 || strncmp(arg, "REL:", 4) == 0) {
        float rel = (float)atof(arg + 4);
        float range;
        if (szp_rel_to_abs_float(data, nbEle, rel, absOut, &range) != 0)
            return -1;
        if (modeOut) *modeOut = "REL";
        return 0;
    } else {
        *absOut = (float)atof(arg);
        if (modeOut) *modeOut = "ABS";
        return 0;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* _SZP_ERRBOUND_H */
