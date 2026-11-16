/**
 *  @file szp_timing.h
 *  @brief Optional step timing for experiments.
 *
 *  With SZP_TIMING=1 in the environment, the topology restoration (CPU and
 *  GPU) and the one-call GPU compression / decompression print one line per
 *  call to stderr:
 *      SZP_TIMING <what> <step>=<seconds> ...
 *  On the GPU this adds a device synchronization after every step, so the
 *  total can be slightly longer than without timing. Off by default.
 */
#ifndef SZP_TIMING_H
#define SZP_TIMING_H

#include <chrono>
#include <cstdlib>

static inline int szp_timing_on()
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("SZP_TIMING");
        on = (e && *e && *e != '0') ? 1 : 0;
    }
    return on;
}

static inline double szp_now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#endif
