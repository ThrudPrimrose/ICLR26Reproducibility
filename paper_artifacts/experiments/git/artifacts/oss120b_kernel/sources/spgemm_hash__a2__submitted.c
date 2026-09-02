/*
 * C implementation of the ``spgemm_hash`` benchmark kernel.
 *
 * The reference algorithm is described in ``spgemm_hash_numpy.py``. It computes the
 * boolean matrix product C = A * B over the (OR, AND) semiring, where A and B are
 * CSR matrices containing only column indices (no value array). The output matrix C
 * is also CSR. The kernel follows the five-phase algorithm used by the CUDA
 * implementation, but runs entirely on the CPU.
 */
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

/* -------------------------------------------------------------------------- */
/* Configuration constants – must match the Python reference. */
static const int64_t HASH_SCALE = 107;   /* multiplicative hash constant */
static const int64_t FIRST_TABLE = 32;   /* smallest bin table size */
static const int64_t NBINS = 8;          /* number of bins (0..7) */
static const int64_t MAX_TABLE = 4096;   /* largest bin table size */

/* -------------------------------------------------------------------------- */
/* Helper: map a row size to a bin index. Returns -1 for size == 0 or >MAX_TABLE. */
static inline int64_t select_bin(int64_t size) {
    if (size <= 0 || size > MAX_TABLE) {
        return -1;
    }
    int64_t low = 0;
    int64_t high = FIRST_TABLE;
    int64_t chosen = -1;
    for (int64_t b = 0; b < NBINS; ++b) {
        if (size > low && size <= high) {
            chosen = b;
        }
        low = high;
        high = high * 2;
    }
    return chosen;
}

/* Helper: given a bin index return the hash‑table size (power of two). */
static inline int64_t table_size(int64_t bin) {
    if (bin < 0) {
        return 0;
    }
    int64_t sz = FIRST_TABLE;
    for (int64_t i = 0; i < bin; ++i) {
        sz <<= 1; /* multiply by 2 */
    }
    return sz;
}

/* -------------------------------------------------------------------------- */
/* Bitonic sort – exactly the network used in the reference implementation. */
static void bitonic_sort(int64_t *key, int64_t n) {
    int64_t size = 2;
    while (size < n) {
        int64_t stride = size / 2;
        while (stride > 0) {
            for (int64_t idx = 0; idx < n / 2; ++idx) {
                int64_t ascending = 1;
                if ((idx / (size / 2)) % 2 == 1) {
                    ascending = 0;
                }
                int64_t pos = 2 * idx - (idx % stride);
                int64_t left = key[pos];
                int64_t right = key[pos + stride];
                int64_t greater = (left > right) ? 1 : 0;
                if (greater == ascending) {
                    key[pos] = right;
                    key[pos + stride] = left;
                }
            }
            stride = stride / 2;
        }
        size = size * 2;
    }
    int64_t stride = n / 2;
    while (stride > 0) {
        for (int64_t idx = 0; idx < n / 2; ++idx) {
            int64_t pos = 2 * idx - (idx % stride);
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

/* -------------------------------------------------------------------------- */
/* Main kernel – mirrors ``spgemm_hash_numpy.spgemm_hash``. */
void spgemm_hash(const int64_t *restrict A_indices,
                 const int64_t *restrict A_indptr,
                 const int64_t *restrict B_indices,
                 const int64_t *restrict B_indptr,
                 const int64_t N,   /* number of columns of B (and C) */
                 const int64_t M,   /* number of rows of A (and C) */
                 int64_t *restrict C_indices,
                 int64_t *restrict C_indptr) {
    /* ---------------------------------------------------------------------- */
    /*  Allocate temporary workspace.  All arrays are sized by ``M`` or ``NBINS``. */
    int64_t *prod = (int64_t *)calloc((size_t)M, sizeof(int64_t));
    int64_t *row_bin = (int64_t *)calloc((size_t)M, sizeof(int64_t));
    int64_t *row_nnz = (int64_t *)calloc((size_t)M, sizeof(int64_t));
    int64_t *bin_size = (int64_t *)calloc((size_t)NBINS, sizeof(int64_t));
    int64_t *bin_offset = (int64_t *)calloc((size_t)NBINS, sizeof(int64_t));
    int64_t *rows_in_bins = (int64_t *)malloc((size_t)M * sizeof(int64_t));
    if (!prod || !row_bin || !row_nnz || !bin_size || !bin_offset || !rows_in_bins) {
        /* allocation failure – nothing the benchmark can recover from */
        abort();
    }
    const int64_t empty = N; /* sentinel value for an empty hash slot */
    int64_t *table = (int64_t *)malloc((size_t)MAX_TABLE * sizeof(int64_t));
    if (!table) abort();

    /* ---------------------------------------------------------------------- */
    /* Phase 1: row analysis – compute an upper bound on distinct columns per row. */
    for (int64_t i = 0; i < M; ++i) {
        int64_t products = 0;
        for (int64_t j = A_indptr[i]; j < A_indptr[i + 1]; ++j) {
            int64_t a_col = A_indices[j];
            int64_t b_start = B_indptr[a_col];
            int64_t b_end = B_indptr[a_col + 1];
            products += (b_end - b_start);
        }
        if (products > N) {
            products = N;
        }
        prod[i] = products;
    }

    /* ---------------------------------------------------------------------- */
    /* Phase 2: bin rows by the estimated product size. */
    // histogram
    for (int64_t i = 0; i < M; ++i) {
        int64_t chosen = select_bin(prod[i]);
        row_bin[i] = chosen;
        if (chosen >= 0) {
            bin_size[chosen]++;
        }
    }
    // exclusive scan of bin sizes to obtain offsets
    int64_t running = 0;
    for (int64_t b = 0; b < NBINS; ++b) {
        bin_offset[b] = running;
        running += bin_size[b];
        bin_size[b] = 0; // will be reused as a counter during scatter
    }
    // scatter rows into bins (preserving original order)
    for (int64_t i = 0; i < M; ++i) {
        rows_in_bins[i] = -1; // initialise – rows with no bin stay -1
    }
    for (int64_t i = 0; i < M; ++i) {
        int64_t chosen = row_bin[i];
        if (chosen >= 0) {
            int64_t pos = bin_offset[chosen] + bin_size[chosen];
            rows_in_bins[pos] = i;
            bin_size[chosen]++;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* Phase 3: symbolic – count distinct columns per row using a private hash table. */
    // Re‑use bin_size as a scratch buffer (reset to zero for each row later).
    for (int64_t r = 0; r < M; ++r) {
        int64_t row = rows_in_bins[r];
        if (row < 0) {
            continue; // empty bin slot
        }
        int64_t ts = table_size(row_bin[row]);
        for (int64_t t = 0; t < ts; ++t) {
            table[t] = empty;
        }
        int64_t distinct = 0;
        for (int64_t j = A_indptr[row]; j < A_indptr[row + 1]; ++j) {
            int64_t a_col = A_indices[j];
            for (int64_t k = B_indptr[a_col]; k < B_indptr[a_col + 1]; ++k) {
                int64_t b_col = B_indices[k];
                int64_t slot = (b_col * HASH_SCALE) % ts;
                while (true) {
                    int64_t held = table[slot];
                    if (held == b_col) {
                        break; // already present
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
    }

    /* ---------------------------------------------------------------------- */
    /* Phase 4: exclusive scan of row_nnz to produce C_indptr. */
    int64_t sum = 0;
    for (int64_t i = 0; i < M; ++i) {
        C_indptr[i] = sum;
        sum += row_nnz[i];
    }
    C_indptr[M] = sum;

    /* ---------------------------------------------------------------------- */
    /* Phase 5a: re‑bin rows by their exact nnz values. */
    // reset bin counters
    for (int64_t b = 0; b < NBINS; ++b) {
        bin_size[b] = 0;
    }
    // compute new bins
    for (int64_t i = 0; i < M; ++i) {
        int64_t chosen = select_bin(row_nnz[i]);
        row_bin[i] = chosen;
        if (chosen >= 0) {
            bin_size[chosen]++;
        }
    }
    // exclusive scan for offsets
    running = 0;
    for (int64_t b = 0; b < NBINS; ++b) {
        bin_offset[b] = running;
        running += bin_size[b];
        bin_size[b] = 0; // reuse as counter for scatter
    }
    // scatter rows again
    for (int64_t i = 0; i < M; ++i) {
        rows_in_bins[i] = -1;
    }
    for (int64_t i = 0; i < M; ++i) {
        int64_t chosen = row_bin[i];
        if (chosen >= 0) {
            int64_t pos = bin_offset[chosen] + bin_size[chosen];
            rows_in_bins[pos] = i;
            bin_size[chosen]++;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* Phase 5b: numeric – repeat hash insertion, sort, and write result. */
    for (int64_t r = 0; r < M; ++r) {
        int64_t row = rows_in_bins[r];
        if (row < 0) {
            continue;
        }
        int64_t ts = table_size(row_bin[row]);
        for (int64_t t = 0; t < ts; ++t) {
            table[t] = empty;
        }
        // insert columns again
        for (int64_t j = A_indptr[row]; j < A_indptr[row + 1]; ++j) {
            int64_t a_col = A_indices[j];
            for (int64_t k = B_indptr[a_col]; k < B_indptr[a_col + 1]; ++k) {
                int64_t b_col = B_indices[k];
                int64_t slot = (b_col * HASH_SCALE) % ts;
                while (true) {
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
        // sort the table – only the first ``ts`` entries contain data or empty sentinel
        bitonic_sort(table, ts);
        // compact the sorted table into C_indices
        int64_t base = C_indptr[row];
        int64_t count = C_indptr[row + 1] - base;
        for (int64_t t = 0; t < count; ++t) {
            C_indices[base + t] = table[t];
        }
    }

    /* ---------------------------------------------------------------------- */
    /* Clean up */
    free(prod);
    free(row_bin);
    free(row_nnz);
    free(bin_size);
    free(bin_offset);
    free(rows_in_bins);
    free(table);
}

/* -------------------------------------------------------------------------- */
/* End of spgemm_hash implementation */
