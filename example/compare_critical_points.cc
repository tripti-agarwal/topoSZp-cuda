/**
 *  @file compare_critical_points.cc
 *  @brief Compare critical points between original and decompressed data
 *  Shows false negatives (in original but not in decompressed) and
 *  false positives (in decompressed but not in original)
 * 
 *  Works with any decompressed binary float files:
 *    - .dat.SZp.out (SZp decompressed)
 *    - .dat.SZ3.out (SZ3 decompressed)
 *    - .dat.zfp.out (ZFP decompressed)
 *    - .dat.tthresh.out (tthresh decompressed)
 *    - .dat.fpzip.out (FPZIP decompressed)
 *    - .dat.sz.out (SZ decompressed)
 * 
 *  Usage: ./compare_critical_points <original_file> <decompressed_file> <rows> <cols> <error_bound>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>

// Critical point types
#define TYPE_REGULAR 0
#define TYPE_MAXIMA 1
#define TYPE_MINIMA 2
#define TYPE_SADDLE 3

/**
 * Classify a point as critical point type using 4-neighbor stencil
 * Returns: 0=regular, 1=maxima, 2=minima, 3=saddle
 */
static inline int classify_critical_point(const float* data, int rows, int cols, int i, int j, float eps) {
    if (i < 1 || i >= rows - 1 || j < 1 || j >= cols - 1) {
        return TYPE_REGULAR;  // Boundary points are not critical
    }
    
    float center = data[i * cols + j];
    float north = data[(i-1) * cols + j];
    float south = data[(i+1) * cols + j];
    float west = data[i * cols + (j-1)];
    float east = data[i * cols + (j+1)];
    
    // Check for maxima: center > all neighbors
    bool gt_n = center > north;
    bool gt_s = center > south;
    bool gt_w = center > west;
    bool gt_e = center > east;
    if (gt_n && gt_s && gt_w && gt_e) {
        return TYPE_MAXIMA;
    }
    
    // Check for minima: center < all neighbors
    bool lt_n = center < north;
    bool lt_s = center < south;
    bool lt_w = center < west;
    bool lt_e = center < east;
    if (lt_n && lt_s && lt_w && lt_e) {
        return TYPE_MINIMA;
    }
    
    // Check for saddle: (high in vertical, low in horizontal) OR (low in vertical, high in horizontal)
    bool vh = (center > north) && (center > south);  // vertical high
    bool vl = (center < north) && (center < south); // vertical low
    bool hh = (center > west) && (center > east);    // horizontal high
    bool hl = (center < west) && (center < east);   // horizontal low
    
    if ((vh && hl) || (vl && hh)) {
        return TYPE_SADDLE;
    }
    
    return TYPE_REGULAR;
}

/**
 * Find all critical points in the data
 * Returns array of critical point information: [row, col, type] for each point
 */
typedef struct {
    int row;
    int col;
    int type;
    float value;
} CriticalPointInfo;

CriticalPointInfo* find_critical_points(const float* data, int rows, int cols, float eps, 
                                    size_t* out_count) {
    if (!data || rows < 3 || cols < 3) {
        *out_count = 0;
        return NULL;
    }
    
    // Allocate maximum possible (all interior points)
    CriticalPointInfo* results = (CriticalPointInfo*)malloc((rows - 2) * (cols - 2) * sizeof(CriticalPointInfo));
    if (!results) {
        *out_count = 0;
        return NULL;
    }
    
    size_t count = 0;
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            int type = classify_critical_point(data, rows, cols, i, j, eps);
            if (type != TYPE_REGULAR) {
                results[count].row = i;
                results[count].col = j;
                results[count].type = type;
                results[count].value = data[i * cols + j];
                count++;
            }
        }
    }
    
    // Resize to actual count
    if (count > 0) {
        results = (CriticalPointInfo*)realloc(results, count * sizeof(CriticalPointInfo));
    } else {
        free(results);
        results = NULL;
    }
    
    *out_count = count;
    return results;
}

/**
 * Create a hash map for quick lookup of critical points
 * Uses a simple hash: row * cols + col
 */
typedef struct {
    int* keys;      // row * cols + col
    int* types;     // critical point type
    float* values;  // data value
    size_t size;
    size_t capacity;
} CriticalPointMap;

CriticalPointMap* create_critical_point_map(CriticalPointInfo* points, size_t count, int cols) {
    CriticalPointMap* map = (CriticalPointMap*)malloc(sizeof(CriticalPointMap));
    if (!map) return NULL;
    
    map->capacity = count * 2;  // 2x for hash table efficiency
    if (map->capacity < 100) map->capacity = 100;
    
    map->keys = (int*)calloc(map->capacity, sizeof(int));
    map->types = (int*)calloc(map->capacity, sizeof(int));
    map->values = (float*)calloc(map->capacity, sizeof(float));
    map->size = 0;
    
    if (!map->keys || !map->types || !map->values) {
        free(map->keys);
        free(map->types);
        free(map->values);
        free(map);
        return NULL;
    }
    
    // Initialize keys to -1 (empty marker)
    for (size_t i = 0; i < map->capacity; i++) {
        map->keys[i] = -1;
    }
    
    // Insert points into hash map using linear probing
    for (size_t i = 0; i < count; i++) {
        int key = points[i].row * cols + points[i].col;
        size_t idx = (size_t)key % map->capacity;
        
        // Linear probing for collision resolution
        while (map->keys[idx] != -1 && map->keys[idx] != key) {
            idx = (idx + 1) % map->capacity;
        }
        
        if (map->keys[idx] == -1) {
            map->keys[idx] = key;
            map->types[idx] = points[i].type;
            map->values[idx] = points[i].value;
            map->size++;
        }
    }
    
    return map;
}

bool find_in_map(CriticalPointMap* map, int row, int col, int cols, int* out_type, float* out_value) {
    if (!map) return false;
    
    int key = row * cols + col;
    size_t idx = (size_t)key % map->capacity;
    size_t start_idx = idx;
    
    // Linear probing search
    do {
        if (map->keys[idx] == key) {
            if (out_type) *out_type = map->types[idx];
            if (out_value) *out_value = map->values[idx];
            return true;
        }
        if (map->keys[idx] == -1) {
            return false;  // Empty slot, not found
        }
        idx = (idx + 1) % map->capacity;
    } while (idx != start_idx);
    
    return false;
}

void free_critical_point_map(CriticalPointMap* map) {
    if (map) {
        free(map->keys);
        free(map->types);
        free(map->values);
        free(map);
    }
}

/**
 * Read float data from a binary file
 * Returns allocated array of floats, or NULL on error
 * Sets nbEle to the number of elements read
 */
float* read_float_data(const char* filename, size_t* nbEle) {
    if (!filename || !nbEle) {
        return NULL;
    }
    
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        return NULL;
    }
    
    // Get file size
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    if (file_size <= 0 || file_size % sizeof(float) != 0) {
        fclose(fp);
        return NULL;
    }
    
    *nbEle = (size_t)(file_size / sizeof(float));
    
    // Allocate memory
    float* data = (float*)malloc(*nbEle * sizeof(float));
    if (!data) {
        fclose(fp);
        return NULL;
    }
    
    // Read data
    size_t read_count = fread(data, sizeof(float), *nbEle, fp);
    fclose(fp);
    
    if (read_count != *nbEle) {
        free(data);
        return NULL;
    }
    
    return data;
}

void print_usage(const char* prog_name) {
    printf("Usage: %s <original_file> <decompressed_file> <rows> <cols> <error_bound>\n", prog_name);
    printf("\n");
    printf("Arguments:\n");
    printf("  original_file      : Path to original .dat file\n");
    printf("  decompressed_file  : Path to decompressed file (.SZp.out, .SZ3.out, .zfp.out, .tthresh.out, .fpzip.out, .sz.out, etc.)\n");
    printf("  rows               : Number of rows in the data\n");
    printf("  cols               : Number of columns in the data\n");
    printf("  error_bound        : Error bound used for classification (e.g., 1e-3)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s original.dat decompressed.dat.SZ3.out 1800 3600 1e-3\n", prog_name);
    printf("  %s original.dat decompressed.dat.zfp.out 384 320 1e-3\n", prog_name);
    printf("  %s original.dat decompressed.dat.tthresh.out 1800 3600 1e-3\n", prog_name);
    printf("  %s original.dat decompressed.dat.fpzip.out 384 320 1e-3\n", prog_name);
    printf("  %s original.dat decompressed.dat.sz.out 384 320 1e-3\n", prog_name);
}

int main(int argc, char** argv) {
    if (argc != 6) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char* original_file = argv[1];
    const char* decompressed_file = argv[2];
    int rows = atoi(argv[3]);
    int cols = atoi(argv[4]);
    float error_bound = (float)atof(argv[5]);
    
    if (rows < 3 || cols < 3) {
        printf("Error: rows and cols must be at least 3\n");
        return 1;
    }
    
    if (error_bound <= 0) {
        printf("Error: error_bound must be positive\n");
        return 1;
    }
    
    // Read original data
    size_t nbEle_orig = 0;
    float* orig_data = read_float_data(original_file, &nbEle_orig);
    if (!orig_data) {
        printf("Error: Failed to read original file: %s\n", original_file);
        return 1;
    }
    if ((size_t)(rows * cols) != nbEle_orig) {
        printf("Error: Original data size mismatch. Expected %d, got %zu\n", rows * cols, nbEle_orig);
        free(orig_data);
        return 1;
    }
    
    // Read decompressed data
    size_t nbEle_decomp = 0;
    float* decomp_data = read_float_data(decompressed_file, &nbEle_decomp);
    if (!decomp_data) {
        printf("Error: Failed to read decompressed file: %s\n", decompressed_file);
        free(orig_data);
        return 1;
    }
    if ((size_t)(rows * cols) != nbEle_decomp) {
        printf("Error: Decompressed data size mismatch. Expected %d, got %zu\n", rows * cols, nbEle_decomp);
        free(orig_data);
        free(decomp_data);
        return 1;
    }
    
    // Find critical points in original data
    size_t orig_count = 0;
    CriticalPointInfo* orig_points = find_critical_points(orig_data, rows, cols, error_bound, &orig_count);
    
    // Find critical points in decompressed data
    size_t decomp_count = 0;
    CriticalPointInfo* decomp_points = find_critical_points(decomp_data, rows, cols, error_bound, &decomp_count);
    
    // Create hash maps for both original and decompressed critical points
    CriticalPointMap* orig_map = create_critical_point_map(orig_points, orig_count, cols);
    CriticalPointMap* decomp_map = create_critical_point_map(decomp_points, decomp_count, cols);
    
    // Statistics
    int false_negatives[4] = {0, 0, 0, 0};  // [total, maxima, minima, saddles]
    int false_positives[4] = {0, 0, 0, 0};
    int true_positives[4] = {0, 0, 0, 0};
    int false_types[4][4] = {{0}};  // [orig_type][decomp_type] - false type mismatches
    
    // Check decompressed points against original
    for (size_t i = 0; i < decomp_count; i++) {
        int row = decomp_points[i].row;
        int col = decomp_points[i].col;
        int type = decomp_points[i].type;
        
        int orig_type;
        float orig_value;
        if (find_in_map(orig_map, row, col, cols, &orig_type, &orig_value)) {
            if (orig_type == type) {
                // True positive: same type at same location
                true_positives[0]++;
                true_positives[type]++;
            } else {
                // False type: same location but different type
                false_types[orig_type][type]++;
                // Also counts as false negative for original type
                false_negatives[0]++;
                false_negatives[orig_type]++;
                // And false positive for decompressed type
                false_positives[0]++;
                false_positives[type]++;
            }
        } else {
            // False positive: in decompressed but not in original
            false_positives[0]++;
            false_positives[type]++;
        }
    }
    
    // Check original points against decompressed (find false negatives)
    for (size_t i = 0; i < orig_count; i++) {
        int row = orig_points[i].row;
        int col = orig_points[i].col;
        int type = orig_points[i].type;
        
        int decomp_type;
        float decomp_value;
        if (!find_in_map(decomp_map, row, col, cols, &decomp_type, &decomp_value)) {
            // False negative: in original but not in decompressed
            // (Only count if not already counted as type mismatch above)
            bool already_counted = false;
            for (size_t j = 0; j < decomp_count; j++) {
                if (decomp_points[j].row == row && decomp_points[j].col == col) {
                    already_counted = true;
                    break;
                }
            }
            if (!already_counted) {
                false_negatives[0]++;
                false_negatives[type]++;
            }
        }
    }
    
    // Calculate total false types
    int total_false_types = 0;
    for (int i = 1; i <= 3; i++) {
        for (int j = 1; j <= 3; j++) {
            if (i != j) {
                total_false_types += false_types[i][j];
            }
        }
    }
    
    // Print false negatives, false positives, false types, grand total, and total decompressed critical points
    int grand_total_false = false_negatives[0] + false_positives[0] + total_false_types;
    printf("%d %d %d %d %zu\n", false_negatives[0], false_positives[0], total_false_types, grand_total_false, decomp_count);
    
    // Cleanup
    free(orig_data);
    free(decomp_data);
    free(orig_points);
    free(decomp_points);
    free_critical_point_map(orig_map);
    free_critical_point_map(decomp_map);
    
    return 0;
}

