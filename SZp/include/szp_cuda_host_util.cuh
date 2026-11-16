/**
 *  @file szp_cuda_host_util.cuh
 *  @brief Two host-side costs of the one-call GPU paths, moved off a single core.
 *
 *  1. Range check on the device. The format check of szp_check_errbound
 *     (max |value| / eb below 2^31, no NaN / Inf) scans the whole field. In
 *     the CUDA library that scan ran on one host thread and took longer than
 *     all compression kernels together; here chunks of the uploaded field are
 *     scanned by the GPU and only the per-chunk results come back.
 *  2. First touch of a result buffer. Copying a device array into freshly
 *     allocated host memory pays one page fault per page inside the copy;
 *     touching the pages first with several threads removes that from it.
 */
#ifndef SZP_CUDA_HOST_UTIL_CUH
#define SZP_CUDA_HOST_UTIL_CUH

#include <cuda_runtime.h>
#include <float.h>
#include <stdlib.h>
#include <thread>
#include <vector>
#ifdef __linux__
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23   /* Linux 5.14 */
#endif
#endif
#include "szp_quant.h"

#define SZP_RANGE_CHUNK 256

/* max |v| and the number of NaN / Inf values of each chunk of SZP_RANGE_CHUNK values */
__global__ static void szp_absmax_kernel(const float *data, size_t n, float *cmax, unsigned int *cbad)
{
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t lo = c * SZP_RANGE_CHUNK;
    if (lo >= n) return;
    size_t hi = lo + SZP_RANGE_CHUNK < n ? lo + SZP_RANGE_CHUNK : n;
    float m = 0.0f;
    unsigned int bad = 0;
    for (size_t i = lo; i < hi; i++) {
        float a = data[i] < 0 ? -data[i] : data[i];
        if (!(a == a) || a > FLT_MAX) { bad++; continue; }   /* NaN or Inf */
        if (a > m) m = a;
    }
    cmax[c] = m;
    cbad[c] = bad;
}

/** szp_check_errbound<float> for a field that is already on the device:
    same messages and return value (< 0: cannot be compressed). */
static int szp_cuda_check_errbound_device(const float *d_data, size_t n, double eb, const char *who)
{
    const size_t nchunk = (n + SZP_RANGE_CHUNK - 1) / SZP_RANGE_CHUNK;
    float *d_max = NULL;
    unsigned int *d_bad = NULL;
    if (cudaMalloc((void **)&d_max, nchunk * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_bad, nchunk * sizeof(unsigned int)) != cudaSuccess) {
        cudaFree(d_max); cudaFree(d_bad);
        fprintf(stderr, "%s: out of device memory in the range check\n", who);
        return -1;
    }
    const int tpb = 256;
    szp_absmax_kernel<<<(unsigned int)((nchunk + tpb - 1) / tpb), tpb>>>(d_data, n, d_max, d_bad);
    std::vector<float> h_max(nchunk);
    std::vector<unsigned int> h_bad(nchunk);
    cudaError_t e1 = cudaMemcpy(h_max.data(), d_max, nchunk * sizeof(float), cudaMemcpyDeviceToHost);
    cudaError_t e2 = cudaMemcpy(h_bad.data(), d_bad, nchunk * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_max); cudaFree(d_bad);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        fprintf(stderr, "%s: range check failed on the device\n", who);
        return -1;
    }
    float amax = 0.0f;
    size_t bad = 0;
    for (size_t c = 0; c < nchunk; c++) {
        if (h_max[c] > amax) amax = h_max[c];
        bad += h_bad[c];
    }
    return szp_check_errbound_limits<float>(amax, bad, eb, who);
}

/** Writes one byte per page of a freshly allocated buffer with several
    threads, so that a later copy into it does not fault page by page. */
static inline void szp_touch_pages(void *buf, size_t bytes)
{
    if (!buf || bytes < ((size_t)32 << 20)) return;
    unsigned nt = std::thread::hardware_concurrency();
#ifdef __linux__
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) nt = (unsigned)CPU_COUNT(&set);
#endif
    if (nt > 32) nt = 32;
    if (nt < 1) nt = 1;
    volatile char *p = (volatile char *)buf;
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; t++)
        th.emplace_back([=] {
            const size_t lo = bytes / nt * t, hi = t + 1 == nt ? bytes : bytes / nt * (t + 1);
            for (size_t i = lo; i < hi; i += 4096) p[i] = 0;
        });
    for (auto &x : th) x.join();
}

/**
 * Maps the pages of a freshly allocated result buffer in the background while
 * the GPU works, so the final copy into it does not wait for page faults.
 * With MADV_POPULATE_WRITE (Linux >= 5.14) the pages are mapped without
 * changing their contents, which is safe even if the copy has already begun;
 * join() must be called before the buffer is handed to the caller. Without
 * it, the buffer is touched before start() returns (szp_touch_pages).
 */
struct SzpPretouch {
    std::vector<std::thread> th;
    void start(void *buf, size_t bytes)
    {
        if (!buf || bytes < ((size_t)32 << 20)) return;
#ifdef __linux__
        const size_t pg = (size_t)sysconf(_SC_PAGESIZE);
        char *lo = (char *)(((size_t)buf + pg - 1) / pg * pg);
        char *hi = (char *)(((size_t)buf + bytes) / pg * pg);
        if (hi > lo && madvise(lo, pg, MADV_POPULATE_WRITE) == 0) {
            unsigned nt = std::thread::hardware_concurrency();
            cpu_set_t set;
            if (sched_getaffinity(0, sizeof(set), &set) == 0) nt = (unsigned)CPU_COUNT(&set);
            if (nt > 16) nt = 16;
            if (nt < 1) nt = 1;
            const size_t npages = (size_t)(hi - lo) / pg;
            for (unsigned t = 0; t < nt; t++) {
                char *a = lo + npages / nt * t * pg;
                char *b = t + 1 == nt ? hi : lo + npages / nt * (t + 1) * pg;
                th.emplace_back([a, b] { if (b > a) (void)madvise(a, (size_t)(b - a), MADV_POPULATE_WRITE); });
            }
            return;
        }
#endif
        szp_touch_pages(buf, bytes);
    }
    void join() { for (auto &x : th) x.join(); th.clear(); }
    ~SzpPretouch() { join(); }
};

#endif
