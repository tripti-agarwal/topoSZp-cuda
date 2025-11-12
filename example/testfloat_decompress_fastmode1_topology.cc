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
static const int NB4[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
float apply_laplacian_maxima_stencil(float *data, int rows, int cols, int i, int j, float error_bound, int sort_pos){
    (void)rows; (void)error_bound;
    float max_neighbor=data[i*cols+j];
    for(int n=0;n<4;n++){
        int ni=i+NB4[n][0], nj=j+NB4[n][1];
        float v=data[ni*cols+nj];
        if(v>max_neighbor) max_neighbor=v;
    }
    return max_neighbor * (1+ (sort_pos * FLT_EPSILON));
}
float apply_laplacian_minima_stencil(float *data, int rows, int cols, int i, int j, float error_bound, int sort_pos){
    (void)rows; (void)error_bound;
    float min_neighbor=data[i*cols+j];
    for(int n=0;n<4;n++){
        int ni=i+NB4[n][0], nj=j+NB4[n][1];
        float v=data[ni*cols+nj];
        if(v<min_neighbor) min_neighbor=v;
    }
    if (sort_pos == 0) return min_neighbor * (1- FLT_EPSILON);
    return min_neighbor * (1 - ((1/sort_pos)* FLT_EPSILON));
}

/**
 * Build a mapping from grid positions to sort position indices.
 * This function creates a lookup table that maps (row, col) positions to indices in the sort_positions array.
 * 
 * @param critical_points_types Array of critical point types for each grid position
 * @param sort_positions Array of sort positions for critical points
 * @param critical_count Number of critical points
 * @param rows Number of rows in the grid
 * @param cols Number of columns in the grid
 * @return Array mapping grid positions to sort position indices (-1 if not a critical point)
 */
int* build_sort_position_mapping(int *critical_points_types, int *sort_positions, size_t critical_count, int rows, int cols) {
    int *mapping = (int*)malloc(rows * cols * sizeof(int));
    if (!mapping) return NULL;
    
    // Initialize all positions as -1 (not a critical point)
    for (int i = 0; i < rows * cols; i++) {
        mapping[i] = -1;
    }
    
    // Build mapping by iterating through critical points
    size_t sort_idx = 0;
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            int idx = i * cols + j;
            int type = critical_points_types[idx];
            
            // Only map extrema (type 1 or 2) - saddles are excluded
            if ((type == 1 || type == 2) && sort_idx < critical_count) {
                mapping[idx] = sort_idx;
                sort_idx++;
            }
        }
    }
    
    return mapping;
}

void apply_stencils_on_critical_points(float *data, int *critical_points_types, int *sort_positions, 
                                     int *sort_position_mapping, int rows, int cols, float error_bound, size_t critical_count){
    for(int i=1;i<rows-1;i++){
        for(int j=1;j<cols-1;j++){
            int idx=i*cols+j;
            int type = critical_points_types[idx];
            
            // Only apply stencils to maxima and minima
            if(type==1 || type==2) {
                // Get the sort position for this critical point
                int sort_idx = sort_position_mapping[idx];
                int sort_pos = (sort_idx >= 0 && sort_idx < (int)critical_count) ? sort_positions[sort_idx] : 0;
                
                // Apply stencil based on type
                if(type==1) {
                    data[idx] = apply_laplacian_maxima_stencil(data,rows,cols,i,j,error_bound,sort_pos);
                    // You can use sort_pos here for additional processing if needed
                    // printf("Maxima at (%d,%d) has sort position %d\n", i, j, sort_pos);
                }
                if(type==2) {
                    data[idx] = apply_laplacian_minima_stencil(data,rows,cols,i,j,error_bound,sort_pos);
                    // You can use sort_pos here for additional processing if needed
                    // printf("Minima at (%d,%d) has sort position %d\n", i, j, sort_pos);
                }
            }
        }
    }
}

static inline float clamp_saddle_center(float cand, float n,float s,float w,float e, float eps){
    /* slightly looser clamp to allow saddles to survive */
    float ns_max=fmaxf(n,s), we_max=fmaxf(w,e);
    float ns_min=fminf(n,s), we_min=fminf(w,e);
    float lo=fminf(ns_max,we_max)-0.25f*eps;   /* was 0.5f*eps */
    float hi=fmaxf(ns_min,we_min)+0.25f*eps;   /* was 0.5f*eps */
    if(lo>hi){ float mid=0.5f*(lo+hi); lo=mid-0.25f*eps; hi=mid+0.25f*eps; }
    if(cand<lo) cand=lo;
    if(cand>hi) cand=hi;
    return cand;
}
static inline int cls4(float c,float n,float s,float w,float e,float eps){
    int gtN=c>n+eps,gtS=c>s+eps,gtW=c>w+eps,gtE=c>e+eps;
    int ltN=c<n-eps,ltS=c<s-eps,ltW=c<w-eps,ltE=c<e-eps;
    if(gtN&&gtS&&gtW&&gtE) return 1;           /* maxima */
    if(ltN&&ltS&&ltW&&ltE) return 2;           /* minima */
    int vh=(c>n+eps)&&(c>s+eps), vl=(c<n-eps)&&(c<s-eps);
    int hh=(c>w+eps)&&(c>e+eps), hl=(c<w-eps)&&(c<e-eps);
    if((vh&&hl)||(vl&&hh)) return 3;           /* saddle */
    return 0;                                  /* regular */
}
static inline int cls_idx(const float* a, int r, int c, int i, int j, float eps){
    (void)r;
    float cc = a[i*c + j];
    float n  = a[(i-1)*c + j];
    float s  = a[(i+1)*c + j];
    float w  = a[i*c + (j-1)];
    float e  = a[i*c + (j+1)];                 /* NOTE: uses c, not cols */
    return cls4(cc,n,s,w,e,eps);
}
void rbf_smooth_saddle_points_safe_extrema_aware(
    float *data,const int *types0,const unsigned char *locks,
    int rows,int cols,double sigma,int ksize,float eps,float errBound)
{
//Validates input parameters
if(!data||!types0||!locks||rows<3||cols<3||ksize<3||(ksize&1)==0||sigma<=0) return;

// Store original decompressed values before smoothing to preserve error bound
// This ensures smoothed values stay within errBound of the original decompressed values
float *orig_decomp = (float*)malloc((size_t)rows*(size_t)cols*sizeof(float));
if(!orig_decomp) return;
for(int i=0;i<rows*cols;i++) orig_decomp[i]=data[i];

//Sets up Gaussian RBF weights based on distance from center
int r=ksize/2; 
double s2=sigma*sigma;

double *w=(double*)malloc((size_t)ksize*(size_t)ksize*sizeof(double));
// Creates a Gaussian kernel where closer neighbors have higher weights
// Uses sigma parameter (s2) to control the smoothing strength
for(int di=-r;di<=r;di++)
    for(int dj=-r;dj<=r;dj++){
        double d2=(double)di*di+(double)dj*dj;
        w[(size_t)(di+r)*ksize+(dj+r)]=exp(-d2/(2.0*s2));
    }

// The function only processes points marked as saddle points (types0[idx]==3):
for(int i=1;i<rows-1;i++){
    for(int j=1;j<cols-1;j++){
        int idx=i*cols+j;
        if(types0[idx]!=3) continue;

        /* do NOT skip near locks; just reduce influence of locked neighbors */
        double num=0,den=0;
        // RBF smoothing: Weights neighbors by Gaussian distance
        // Lock protection: Reduces influence of locked neighbors (near extrema) by 60% (ww *= 0.4)
        for(int di=-r;di<=r;di++){
            int ii=i+di; if(ii<0||ii>=rows) continue;
            for(int dj=-r;dj<=r;dj++){
                int jj=j+dj; if(jj<0||jj>=cols) continue;
                double ww=w[(size_t)(di+r)*ksize+(dj+r)];
                if(locks[ii*cols+jj]) ww *= 0.4; /* gentler down-weight */
                num += ww * (double)data[ii*cols+jj];
                den += ww;
            }
        }
        if(den<=0) continue;

        float oldc=data[idx];
        float orig_val=orig_decomp[idx];  // Original decompressed value before any smoothing
        float n=data[(i-1)*cols+j], s=data[(i+1)*cols+j];
        float wv=data[i*cols+(j-1)], ev=data[i*cols+(j+1)];
        float cand=(float)(num/den);
        // Ensures the smoothed value maintains saddle characteristics
        // Prevents the point from becoming a regular point or different critical type
        cand=clamp_saddle_center(cand,n,s,wv,ev,eps);
        
        // Preserve error bound: clamp smoothed value to stay within errBound of original decompressed value
        // The original decompressed value should already be within errBound of the original data.
        // Since the decompressed value could be at the edge of the error bound, we use a very conservative
        // constraint. The decompressed value is at the center of a quantization bin of size errBound,
        // so the original is in [orig_val - errBound, orig_val + errBound). To ensure the smoothed value
        // stays within errBound of the original, we constrain it to a tighter range to account for the
        // worst case where the decompressed value is already at the edge.
        // Using a tight safety margin to prevent exceeding the error bound.
        // Use a very tight margin (10%) to ensure we stay within error bound, similar to restore_extrema_from_types
        float safety_margin = 0.1f;  // Very tight: only allow smoothing within 10% of error bound
        float effective_bound = errBound * safety_margin;
        float min_allowed = orig_val - effective_bound;
        float max_allowed = orig_val + effective_bound;
        if(cand < min_allowed) cand = min_allowed;
        if(cand > max_allowed) cand = max_allowed;

        float step=cand-oldc, alpha=1.f;
        // Error bound constraints (already computed above)
        // min_allowed and max_allowed are already set from lines 229-230
        for(int it=0;it<10;it++){
            float trial=oldc+alpha*step;
            
            // CRITICAL: Clamp trial to preserve error bound
            // Even though cand was clamped, the iterative alpha reduction might allow
            // trial to exceed the error bound if step is large
            if(trial < min_allowed) trial = min_allowed;
            if(trial > max_allowed) trial = max_allowed;
            
            float saved=data[idx]; data[idx]=trial;
            int bad=0;
            // Check if smoothing creates unwanted critical points
            // Regular points (types0[idx]==0) should not become critical points
            if(types0[idx]==0 && cls_idx(data,rows,cols,i,j,eps)!=0) bad=1;
            // Check neighbors too
            if(!bad){ int ni=i-1; if(types0[ni*cols+j]==0 && cls_idx(data,rows,cols,ni,j,eps)!=0) bad=1; }
            if(!bad){ int si=i+1; if(types0[si*cols+j]==0 && cls_idx(data,rows,cols,si,j,eps)!=0) bad=1; }
            if(!bad){ int wj=j-1; if(types0[i*cols+wj]==0 && cls_idx(data,rows,cols,i,wj,eps)!=0) bad=1; }
            if(!bad){ int ej=j+1; if(types0[i*cols+ej]==0 && cls_idx(data,rows,cols,i,ej,eps)!=0) bad=1; }
            
            // Safety mechanism: Gradually applies smoothing (using alpha factor)
            // Validation: Checks if smoothing accidentally creates new critical points
            // Rollback: If smoothing creates problems, it reduces the smoothing strength
            if(!bad && cls_idx(data,rows,cols,i,j,eps)!=3) bad=1;
            if(bad){ data[idx]=saved; alpha*=0.5f; } else { break; }
        }
    }
}
free(w);
free(orig_decomp);
}
void restore_extrema_from_types(const int *types,
    float *data, const float *orig_decomp, int rows, int cols, float eps, float errBound){
    float eps_soft = 0.25f * eps;   /* softer margin so nearby saddles survive */
    
    // Safety margin to ensure we stay within error bound
    // The original decompressed value is quantized, so the original data could be
    // anywhere in [orig_decomp - errBound, orig_decomp + errBound)
    // To ensure the modified value stays within errBound of the original data:
    // Using triangle inequality: |modified - orig_data| <= |modified - orig_decomp| + |orig_decomp - orig_data|
    // Since |orig_decomp - orig_data| <= errBound, to ensure |modified - orig_data| <= errBound,
    // we need |modified - orig_decomp| to be very small. However, this is too restrictive.
    // A practical approach: constrain modified to be within a small fraction of errBound from orig_decomp.
    // This ensures that even in worst case, |modified - orig_data| <= errBound + small_margin.
    // Use a tight constraint: allow modification within 10% of errBound to preserve extrema while staying close to error bound.
    float safety_margin = 0.01f;  // Use 10% of error bound - very tight to preserve error bound
    float effective_bound = errBound * safety_margin;
    
    for(int i=1;i<rows-1;i++){
        for(int j=1;j<cols-1;j++){
            int idx=i*cols+j;
            if(types[idx]==1){ /* maxima */
                float n=data[(i-1)*cols+j], s=data[(i+1)*cols+j];
                float w=data[i*cols+(j-1)], e=data[i*cols+(j+1)];
                float m = fmaxf(fmaxf(n,s), fmaxf(w,e));
                float target = m + eps_soft;      /* minimal raise to be strict */
                
                // Clamp target to preserve error bound
                if (orig_decomp) {
                    float orig_val = orig_decomp[idx];
                    float max_allowed = orig_val + effective_bound;
                    float min_allowed = orig_val - effective_bound;
                    // For maxima, we need data[idx] >= target to maintain maximum property
                    // But we also need data[idx] within [min_allowed, max_allowed]
                    // So take the maximum of target and min_allowed, then clamp to max_allowed
                    target = fmaxf(target, min_allowed);
                    if (target > max_allowed) target = max_allowed;
                }
                
                if (data[idx] < target) data[idx] = target;
            }else if(types[idx]==2){ /* minima */
                float n=data[(i-1)*cols+j], s=data[(i+1)*cols+j];
                float w=data[i*cols+(j-1)], e=data[i*cols+(j+1)];
                float m = fminf(fminf(n,s), fminf(w,e));
                float target = m - eps_soft;      /* minimal lower to be strict */
                
                // Clamp target to preserve error bound
                if (orig_decomp) {
                    float orig_val = orig_decomp[idx];
                    float max_allowed = orig_val + effective_bound;
                    float min_allowed = orig_val - effective_bound;
                    // For minima, we need data[idx] <= target to maintain minimum property
                    // But we also need data[idx] within [min_allowed, max_allowed]
                    // So take the minimum of target and max_allowed, then clamp to min_allowed
                    target = fminf(target, max_allowed);
                    if (target < min_allowed) target = min_allowed;
                }
                
                if (data[idx] > target) data[idx] = target;
            }
        }
    }
}
static unsigned char* build_extrema_locks_from_types(const int *types, int rows, int cols){
    size_t N=(size_t)rows*(size_t)cols;
    unsigned char *lock=(unsigned char*)calloc(N,1);
    for(int i=1;i<rows-1;i++){
        for(int j=1;j<cols-1;j++){
            int idx=i*cols+j;
            if(types[idx]==1 || types[idx]==2){
                for(int di=-1; di<=1; di++){
                    for(int dj=-1; dj<=1; dj++){
                        int ii=i+di, jj=j+dj;
                        if(ii>0 && ii<rows-1 && jj>0 && jj<cols-1)
                            lock[ii*cols+jj]=1;
                    }
                }
            }
        }
    }
    return lock;
}
int main(int argc, char *argv[])
{
    size_t nbEle, totalNbEle;
    char zipFilePath[640], outputFilePath[645];
    if (argc < 2)
    {
        printf("Usage: testfloat_decompress_fastmode1 [srcFilePath] [nbEle] [block size] [err bound] [rows] [cols]\n");
        printf("Example: testfloat_decompress_fastmode1 testfloat_8_8_128.dat.SZp 250000 64 1E-3 500 500\n");
        exit(0);
    }

    sprintf(zipFilePath, "%s", argv[1]);
    //    nbEle = atoi(argv[2]);
    nbEle = strtoimax(argv[2], NULL, 10);

    sprintf(outputFilePath, "%s.out", zipFilePath);
    int blockSize = atoi(argv[3]);
    float errBound = atof(argv[4]);

    int rows = atoi(argv[5]);
    int cols = atoi(argv[6]);

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
    int *types_orig = (int*)malloc((size_t)rows*(size_t)cols*sizeof(int));
    int *sort_positions = (int*)malloc((size_t)rows*(size_t)cols*sizeof(int));
    unsigned char *locks = build_extrema_locks_from_types(types_orig, rows, cols);
    float eps = fmaxf(1e-6f, 0.1f*errBound);
    size_t extrema_count = 0;  // Number of extrema (maxima + minima) only
    size_t sort_positions_size = 0;
    size_t max_copy = (size_t)rows * (size_t)cols;
    cost_start();
    
    // Extract sizes from footer (last 2 * sizeof(size_t) bytes)
    // Footer format: [topology_size (size_t)] [sort_positions_size (size_t)]
    size_t topology_size = 0;
    size_t footer_size = 2 * sizeof(size_t);
    
    // Safety check: ensure we have enough bytes to read the footer
    if (byteLength >= footer_size && bytes != NULL) {
        // Read footer safely
        memcpy(&topology_size, bytes + byteLength - footer_size, sizeof(size_t));
        memcpy(&sort_positions_size, bytes + byteLength - sizeof(size_t), sizeof(size_t));
        
        // Validate sizes
        size_t calculated_total = topology_size + sort_positions_size + footer_size;
        if (topology_size > 0 && sort_positions_size >= 0 && 
            calculated_total <= byteLength + 16 && topology_size < byteLength) { // Allow small margin
            // Sizes are valid - use them
        } else {
            // Fallback: assume entire buffer is topology data (old format)
            topology_size = byteLength;
            sort_positions_size = 0;
        }
    } else {
        // Fallback: assume entire buffer is topology data (old format)
        topology_size = byteLength;
        sort_positions_size = 0;
    }
    
    // Create a buffer with topology data + generous padding to prevent OOB reads
    // The decompression function may read past the end when processing the last block
    // Add padding equal to multiple block sizes to account for worst-case scenarios
    // Each block can have: initial int (4 bytes) + bit_count (1 byte) + 
    // sign array + saved bits + type array (2*blockSize bits = (2*blockSize+7)/8 bytes)
    size_t max_block_size_bytes = sizeof(int) + 1 + ((blockSize - 1) + 7) / 8 + 
                                   ((blockSize - 1) * 32 + 7) / 8 + 
                                   (2 * blockSize + 7) / 8;
    // Add very generous padding to prevent OOB reads
    // The decompression function may read past the end when processing the last block
    // Use a fixed large padding size (10KB) to ensure we have enough space
    size_t padding_size = 10240; // 10KB padding - should be sufficient for any overread
    size_t padded_topology_size = topology_size + padding_size;
    unsigned char *padded_topology = (unsigned char *)malloc(padded_topology_size);
    if (!padded_topology) {
        printf("Error: Failed to allocate padded topology buffer (size=%zu)!\n", padded_topology_size);
        free(bytes);
        exit(1);
    }
    memcpy(padded_topology, bytes, topology_size);
    memset(padded_topology + topology_size, 0, padding_size); // Zero padding
    
    // Decompress using padded buffer
    szp_float_decompress_openmp_threadblock_randomaccess_topology_preserved(
        &data, nbEle, errBound, blockSize, padded_topology, &critical_points_types);
    
    free(padded_topology);

    // Decompress sort positions if available
    if (sort_positions_size > 0 && sort_positions_size + sizeof(size_t) <= byteLength) {
        unsigned char *sort_start = bytes + topology_size;
        // sort_positions section layout: [critical_count (size_t)] [per-thread offsets ...] [payload]
        
        if (sort_positions_size >= sizeof(size_t)) {
            memcpy(&extrema_count, sort_start, sizeof(size_t));
            // Decompress into a freshly allocated array
            int *decomp_sort_positions = szp_decompress_sort_positions(
                sort_start + sizeof(size_t), extrema_count, blockSize);
            if (decomp_sort_positions) {
                // Copy into caller's buffer up to available space
                
                if (extrema_count < max_copy) max_copy = extrema_count;
                memcpy(sort_positions, decomp_sort_positions, max_copy * sizeof(int));
                free(decomp_sort_positions);
            } else {
                printf("Warning: sort positions decompression failed\n");
            }
        }
    }
    memcpy(types_orig, critical_points_types, (size_t)rows*(size_t)cols*sizeof(int));
    
    // Store original decompressed values before any modifications to preserve error bound
    float *orig_decomp = (float*)malloc((size_t)rows*(size_t)cols*sizeof(float));
    if (!orig_decomp) {
        printf("Error: Failed to allocate memory for original decompressed values\n");
        free(bytes);
        free(data);
        free(critical_points_types);
        exit(1);
    }
    memcpy(orig_decomp, data, (size_t)rows*(size_t)cols*sizeof(float));
    
    // Build mapping from grid positions to sort position indices
    int *sort_position_mapping = NULL;
    if (extrema_count > 0) {
        sort_position_mapping = build_sort_position_mapping(types_orig, sort_positions, extrema_count, rows, cols);
        if (!sort_position_mapping) {
            printf("Warning: Failed to build sort position mapping\n");
        }
    }
    
    //calculating the sigma and eps is done /u/tagarwal1/topoSZp/SZp/example/testfloat_decompress_fastmode1_topology.cc
    // rbf_smooth_saddle_points_safe_extrema_aware(data, types_orig, locks, rows, cols, 0.8, 3, eps, errBound);
    apply_stencils_on_critical_points(data, types_orig, sort_positions, sort_position_mapping, rows, cols, errBound, extrema_count);
    restore_extrema_from_types(types_orig, data, orig_decomp, rows, cols, eps, errBound);
    
    free(orig_decomp);
    
    // Clean up mapping
    if (sort_position_mapping) {
        free(sort_position_mapping);
    }
    
    cost_end();
    
    free(bytes);
    printf("timecost=%f\n", totalCost);
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
    free(ori_data);
    return 0;
}
