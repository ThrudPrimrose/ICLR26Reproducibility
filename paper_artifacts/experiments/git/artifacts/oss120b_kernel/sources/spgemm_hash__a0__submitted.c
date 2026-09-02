#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/*
 * Boolean sparse matrix-matrix multiplication C = A * B (OR-AND semiring) where A is MxK and B is KxN.
 * CSR format for both matrices. Arguments:
 *   A_indices: column indices of A (size nnz(A))
 *   A_indptr: row pointer for A (size M+1)
 *   B_indices: column indices of B (size nnz(B))
 *   B_indptr: row pointer for B (size K+1)
 *   N: number of columns of B (and of C)
 *   M: number of rows of A (and of C)
 *   C_indices: pre-allocated array of capacity nnz(C) (filled with -1 by the harness)
 *   C_indptr: array of size M+1, will be filled with CSR row pointers for C.
 */

#define HASH_SCALE 107
#define FIRST_TABLE 32
#define NBINS 8
#define MAX_TABLE 4096

static inline int64_t _select_bin(int64_t size) {
    if (size == 0) return -1;
    if (size > MAX_TABLE) return -1;
    int64_t low = 0;
    int64_t high = FIRST_TABLE;
    for (int b = 0; b < NBINS; ++b) {
        if (size > low && size <= high) {
            return b;
        }
        low = high;
        high <<= 1; // multiply by 2
    }
    return -1;
}

static inline int64_t _table_size(int64_t bin) {
    int64_t ts = FIRST_TABLE;
    for (int64_t i = 0; i < bin; ++i) {
        ts <<= 1;
    }
    return ts;
}

static void _bitonic_sort(int64_t *key, int64_t n) {
    for (int64_t size = 2; size < n; size <<= 1) {
        for (int64_t stride = size >> 1; stride > 0; stride >>= 1) {
            for (int64_t idx = 0; idx < n / 2; ++idx) {
                int ascending = 1;
                if ((idx / (size >> 1)) % 2 == 1) ascending = 0;
                int64_t pos = 2 * idx - (idx % stride);
                int64_t left = key[pos];
                int64_t right = key[pos + stride];
                int greater = (left > right) ? 1 : 0;
                if (greater == ascending) {
                    key[pos] = right;
                    key[pos + stride] = left;
                }
            }
        }
    }
    for (int64_t stride = n / 2; stride > 0; stride >>= 1) {
        for (int64_t idx = 0; idx < n / 2; ++idx) {
            int64_t pos = 2 * idx - (idx % stride);
            int64_t left = key[pos];
            int64_t right = key[pos + stride];
            if (left > right) {
                key[pos] = right;
                key[pos + stride] = left;
            }
        }
    }
}

void spgemm_hash(const int64_t *restrict A_indices,
                 const int64_t *restrict A_indptr,
                 const int64_t *restrict B_indices,
                 const int64_t *restrict B_indptr,
                 int64_t N,
                 int64_t M,
                 int64_t *restrict C_indices,
                 int64_t *restrict C_indptr) {
    const int64_t empty = N; // sentinel greater than any column index
    // Allocate temporary arrays
    int64_t *prod = (int64_t *)malloc((size_t)M * sizeof(int64_t));
    int64_t *row_bin = (int64_t *)malloc((size_t)M * sizeof(int64_t));
    int64_t *row_nnz = (int64_t *)malloc((size_t)M * sizeof(int64_t));
    if (row_nnz) memset(row_nnz, 0, (size_t)M * sizeof(int64_t));
    int64_t *bin_size = (int64_t *)calloc(NBINS, sizeof(int64_t));
    int64_t *bin_offset = (int64_t *)malloc(NBINS * sizeof(int64_t));
    int64_t *rows_in_bins = (int64_t *)malloc((size_t)M * sizeof(int64_t));
    if (!prod || !row_bin || !row_nnz || !bin_offset || !rows_in_bins) {
        // allocation failure: exit early (the harness will treat as incorrect)
        return;
    }
    // Phase 1: row analysis (product bounds)
    for (int64_t i = 0; i < M; ++i) {
        int64_t products = 0;
        for (int64_t j = A_indptr[i]; j < A_indptr[i + 1]; ++j) {
            int64_t a_col = A_indices[j];
            products += (B_indptr[a_col + 1] - B_indptr[a_col]);
        }
        if (products > N) products = N;
        prod[i] = products;
    }
    // Phase 2: binning rows by product estimate
    for (int b = 0; b < NBINS; ++b) bin_size[b] = 0;
    for (int64_t i = 0; i < M; ++i) {
        int64_t chosen = _select_bin(prod[i]);
        row_bin[i] = chosen;
        if (chosen >= 0) {
            bin_size[chosen]++;
        }
    }
    int64_t running = 0;
    for (int b = 0; b < NBINS; ++b) {
        bin_offset[b] = running;
        running += bin_size[b];
        bin_size[b] = 0; // reset for scatter
    }
    for (int64_t i = 0; i < M; ++i) rows_in_bins[i] = -1;
    for (int64_t i = 0; i < M; ++i) {
        int64_t chosen = row_bin[i];
        if (chosen >= 0) {
            rows_in_bins[bin_offset[chosen] + bin_size[chosen]] = i;
            bin_size[chosen]++;
        }
    }
    // Phase 3: symbolic phase – count distinct columns per row (parallelized)
#pragma omp parallel for schedule(dynamic) shared(row_bin, rows_in_bins, A_indptr, A_indices, B_indptr, B_indices, row_nnz, empty)
for (int64_t r = 0; r < M; ++r) {
    int64_t row = rows_in_bins[r];
    if (row >= 0) {
        int64_t ts = _table_size(row_bin[row]);
        int64_t *table = (int64_t *)malloc((size_t)MAX_TABLE * sizeof(int64_t));
        if (!table) continue; // out of memory - skip
        for (int64_t t = 0; t < ts; ++t) table[t] = empty;
        int64_t distinct = 0;
        for (int64_t j = A_indptr[row]; j < A_indptr[row + 1]; ++j) {
            int64_t a_col = A_indices[j];
            for (int64_t k = B_indptr[a_col]; k < B_indptr[a_col + 1]; ++k) {
                int64_t b_col = B_indices[k];
                int64_t slot = (b_col * HASH_SCALE) % ts;
                while (1) {
                    int64_t held = table[slot];
                    if (held == b_col) {
                        break;
                    } else if (held == empty) {
                        table[slot] = b_col;
                        distinct++;
                        break;
                    } else {
                        slot++;
                        if (slot == ts) slot = 0;
                    }
                }
            }
        }
        row_nnz[row] = distinct;
        free(table);
    }
}
    // Phase 4: exclusive scan to produce C_indptr
    int64_t cum = 0;
    for (int64_t i = 0; i < M; ++i) {
        C_indptr[i] = cum;
        cum += row_nnz[i];
    }
    C_indptr[M] = cum;
    // Phase 5a: binning rows by exact nnz
    for (int b = 0; b < NBINS; ++b) bin_size[b] = 0;
    for (int64_t i = 0; i < M; ++i) {
        int64_t chosen = _select_bin(row_nnz[i]);
        row_bin[i] = chosen;
        if (chosen >= 0) bin_size[chosen]++;
    }
    running = 0;
    for (int b = 0; b < NBINS; ++b) {
        bin_offset[b] = running;
        running += bin_size[b];
        bin_size[b] = 0;
    }
    for (int64_t i = 0; i < M; ++i) rows_in_bins[i] = -1;
    for (int64_t i = 0; i < M; ++i) {
        int64_t chosen = row_bin[i];
        if (chosen >= 0) {
            rows_in_bins[bin_offset[chosen] + bin_size[chosen]] = i;
            bin_size[chosen]++;
        }
    }
    // Phase 5b: numeric phase – fill hash table, sort, write indices (parallelized)
#pragma omp parallel for schedule(dynamic) shared(row_bin, rows_in_bins, A_indptr, A_indices, B_indptr, B_indices, C_indptr, C_indices, empty)
for (int64_t r = 0; r < M; ++r) {
    int64_t row = rows_in_bins[r];
    if (row >= 0) {
        int64_t ts = _table_size(row_bin[row]);
        int64_t *table = (int64_t *)malloc((size_t)MAX_TABLE * sizeof(int64_t));
        if (!table) continue; // out of memory, skip
        for (int64_t t = 0; t < ts; ++t) table[t] = empty;
        for (int64_t j = A_indptr[row]; j < A_indptr[row + 1]; ++j) {
            int64_t a_col = A_indices[j];
            for (int64_t k = B_indptr[a_col]; k < B_indptr[a_col + 1]; ++k) {
                int64_t b_col = B_indices[k];
                int64_t slot = (b_col * HASH_SCALE) % ts;
                while (1) {
                    int64_t held = table[slot];
                    if (held == b_col) {
                        break;
                    } else if (held == empty) {
                        table[slot] = b_col;
                        break;
                    } else {
                        slot++;
                        if (slot == ts) slot = 0;
                    }
                }
            }
        }
        _bitonic_sort(table, ts);
        int64_t base = C_indptr[row];
        int64_t count = C_indptr[row + 1] - base;
        for (int64_t t = 0; t < count; ++t) {
            C_indices[base + t] = table[t];
        }
        free(table);
    }
}
    // free temporaries
    free(prod);
    free(row_bin);
    free(row_nnz);
    free(bin_offset);
    free(rows_in_bins);
    free(bin_size);
}

