/*
 * C implementation of the spgemm_hash benchmark kernel.
 *
 * Computes the boolean sparse matrix-matrix product C = A * B over the
 * (OR, AND) semiring where A (M×K) and B (K×N) are stored in CSR format.
 * Only column indices are stored (boolean values are implicitly true).
 * The implementation follows the reference algorithm from spgemm_hash_numpy.py
 * and mirrors the five-phase GPU algorithm: row analysis, binning, symbolic
 * counting, exclusive scan, re‑binning, numeric insertion, bitonic sort and
 * final copy.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

/* Constants matching the nsparse reference */
#define HASH_SCALE 107      /* multiplicative hash constant */
#define FIRST_TABLE 32      /* smallest bin table size */
#define NBINS 8             /* number of bins (0,32] … (2048,4096] */
#define MAX_TABLE 4096      /* largest hash table size */

/* Helper: select a bin for a given size. Returns -1 for size==0 or >4096. */
static inline int select_bin(int64_t size) {
    int chosen = -1;
    int64_t low = 0;
    int64_t high = FIRST_TABLE;
    for (int b = 0; b < NBINS; ++b) {
        if (size > low && size <= high) {
            chosen = b;
        }
        low = high;
        high = high * 2;
    }
    return chosen;
}

/* Helper: table size (power‑of‑two) for a given bin index. */
static inline int table_size(int bin) {
    int ts = FIRST_TABLE;
    for (int i = 0; i < bin; ++i) {
        ts <<= 1;   /* multiply by 2 */
    }
    return ts;
}

/* Bitonic sort – exactly the algorithm from the reference (ascending). */
static void bitonic_sort(int64_t *key, int n) {
    int size = 2;
    while (size < n) {
        int stride = size / 2;
        while (stride > 0) {
            for (int idx = 0; idx < n / 2; ++idx) {
                int ascending = 1;
                if ((idx / (size / 2)) % 2 == 1) {
                    ascending = 0;
                }
                int pos = 2 * idx - (idx % stride);
                int64_t left = key[pos];
                int64_t right = key[pos + stride];
                int greater = left > right;
                if (greater == ascending) {
                    key[pos] = right;
                    key[pos + stride] = left;
                }
            }
            stride = stride / 2;
        }
        size = size * 2;
    }
    int stride = n / 2;
    while (stride > 0) {
        for (int idx = 0; idx < n / 2; ++idx) {
            int pos = 2 * idx - (idx % stride);
            int64_t left = key[pos];
            int64_t right = key[pos + stride];
            if (left > right) {
                key[pos] = right;
                key[pos + stride] = left;
            }
        }
        stride = stride / 2;
    }
}

static int cmp_int64(const void *a, const void *b) {
    int64_t aa = *(const int64_t *)a;
    int64_t bb = *(const int64_t *)b;
    return (aa > bb) - (aa < bb);
}

/* The kernel – matches the signature used by the harness. */
void spgemm_hash(const int64_t *restrict A_indices,
                 const int64_t *restrict A_indptr,
                 const int64_t *restrict B_indices,
                 const int64_t *restrict B_indptr,
                 int64_t N, int64_t M,
                 int64_t *restrict C_indices,
                 int64_t *restrict C_indptr) {
    const int64_t empty = N;                 /* sentinel > any column index */

    /* Allocate auxiliary arrays */
    int64_t *prod = (int64_t *)malloc((size_t)M * sizeof(int64_t));
    int *row_bin = (int *)malloc((size_t)M * sizeof(int));
    int64_t *row_nnz = (int64_t *)malloc((size_t)M * sizeof(int64_t));
    int64_t *rows_in_bins = (int64_t *)malloc((size_t)M * sizeof(int64_t));
    int64_t bin_size[NBINS];
    int64_t bin_offset[NBINS];

    /* ------------------------------------------------------------------
     * Phase 1 – row analysis (upper bound on distinct columns per row).
     * ------------------------------------------------------------------ */
    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < M; ++i) {
        int64_t products = 0;
        for (int64_t j = A_indptr[i]; j < A_indptr[i + 1]; ++j) {
            int64_t a_col = A_indices[j];
            products += B_indptr[a_col + 1] - B_indptr[a_col];
        }
        if (products > N) products = N;
        prod[i] = products;
    }

    /* ------------------------------------------------------------------
     * Phase 2 – bin rows by the estimated bound.
     * ------------------------------------------------------------------ */
    for (int b = 0; b < NBINS; ++b) bin_size[b] = 0;
    for (int64_t i = 0; i < M; ++i) {
        int chosen = select_bin(prod[i]);
        row_bin[i] = chosen;
        if (chosen >= 0) {
            bin_size[chosen]++;
        }
    }
    int64_t running = 0;
    for (int b = 0; b < NBINS; ++b) {
        bin_offset[b] = running;
        running += bin_size[b];
        bin_size[b] = 0;                      /* reuse for scattering */
    }
    for (int64_t r = 0; r < M; ++r) rows_in_bins[r] = -1;
    for (int64_t i = 0; i < M; ++i) {
        int chosen = row_bin[i];
        if (chosen >= 0) {
            rows_in_bins[bin_offset[chosen] + bin_size[chosen]] = i;
            bin_size[chosen]++;
        }
    }

    /* ------------------------------------------------------------------
     * Phase 3 – symbolic: count distinct columns per row using a hash set.
     * ------------------------------------------------------------------ */
    #pragma omp parallel for schedule(dynamic)
    for (int64_t r = 0; r < M; ++r) {
        int64_t row = rows_in_bins[r];
        if (row >= 0) {
            int bin = row_bin[row];
            int ts = table_size(bin);
            int64_t table[MAX_TABLE];
            for (int t = 0; t < ts; ++t) table[t] = empty;
            int64_t distinct = 0;
            for (int64_t j = A_indptr[row]; j < A_indptr[row + 1]; ++j) {
                int64_t a_col = A_indices[j];
                for (int64_t k = B_indptr[a_col]; k < B_indptr[a_col + 1]; ++k) {
                    int64_t b_col = B_indices[k];
                    int64_t slot = (b_col * HASH_SCALE) % ts;
                    while (1) {
                        int64_t held = table[slot];
                        if (held == b_col) {
                            break;                 /* already present */
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
        } else {
            /* empty row – row_nnz stays zero (allocated with malloc) */
        }
    }

    /* ------------------------------------------------------------------
     * Phase 4 – exclusive scan of row counts → C_indptr.
     * ------------------------------------------------------------------ */
    int64_t cum = 0;
    for (int64_t i = 0; i < M; ++i) {
        C_indptr[i] = cum;
        cum += row_nnz[i];
    }
    C_indptr[M] = cum;

    /* ------------------------------------------------------------------
     * Phase 5a – re‑bin rows by the exact nnz.
     * ------------------------------------------------------------------ */
    for (int b = 0; b < NBINS; ++b) bin_size[b] = 0;
    for (int64_t i = 0; i < M; ++i) {
        int chosen = select_bin(row_nnz[i]);
        row_bin[i] = chosen;
        if (chosen >= 0) bin_size[chosen]++;
    }
    running = 0;
    for (int b = 0; b < NBINS; ++b) {
        bin_offset[b] = running;
        running += bin_size[b];
        bin_size[b] = 0;
    }
    for (int64_t r = 0; r < M; ++r) rows_in_bins[r] = -1;
    for (int64_t i = 0; i < M; ++i) {
        int chosen = row_bin[i];
        if (chosen >= 0) {
            rows_in_bins[bin_offset[chosen] + bin_size[chosen]] = i;
            bin_size[chosen]++;
        }
    }

    /* ------------------------------------------------------------------
     * Phase 5b – numeric: hash again, extract distinct columns, sort, and copy into C_indices.
     * ------------------------------------------------------------------ */
    #pragma omp parallel for schedule(dynamic)
    for (int64_t r = 0; r < M; ++r) {
        int64_t row = rows_in_bins[r];
        if (row >= 0) {
            int bin = row_bin[row];
            int ts = table_size(bin);
            int64_t table[MAX_TABLE];
            for (int t = 0; t < ts; ++t) table[t] = empty;
            for (int64_t j = A_indptr[row]; j < A_indptr[row + 1]; ++j) {
                int64_t a_col = A_indices[j];
                for (int64_t k = B_indptr[a_col]; k < B_indptr[a_col + 1]; ++k) {
                    int64_t b_col = B_indices[k];
                    int64_t slot = (b_col * HASH_SCALE) % ts;
                    while (1) {
                        int64_t held = table[slot];
                        if (held == b_col) {
                            break;                 /* duplicate – ignore */
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
            /* Extract distinct columns into a compact array */
            int64_t count = C_indptr[row + 1] - C_indptr[row];
            int64_t out[count]; // VLA for the distinct columns
            int64_t idx = 0;
            for (int i = 0; i < ts && idx < count; ++i) {
                int64_t val = table[i];
                if (val != empty) {
                    out[idx++] = val;
                }
            }
            /* Sort the distinct columns */
            qsort(out, (size_t)count, sizeof(int64_t), cmp_int64);
            int64_t base = C_indptr[row];
            for (int64_t t = 0; t < count; ++t) {
                C_indices[base + t] = out[t];
            }
        }
    }

    /* Cleanup */
    free(prod);
    free(row_bin);
    free(row_nnz);
    free(rows_in_bins);
}

