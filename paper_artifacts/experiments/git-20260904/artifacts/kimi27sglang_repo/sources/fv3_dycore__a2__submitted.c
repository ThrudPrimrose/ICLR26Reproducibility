/* Optimized hand-written implementation of FV3 finite-volume transport (fv_tp_2d)
 * following the NumPy reference layout: arrays are (nx, ny, nk) with k fastest.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define NH 3

#define IDX3(a, i, j, k, ny, nk) ((a)[((i) * (ny) + (j)) * (nk) + (k)])

static inline double *alloc_or_malloc(uint8_t **ws_ptr, size_t *ws_size, size_t need, void **malloc_ptr) {
    if (*ws_ptr != NULL && *ws_size >= need) {
        double *p = (double *)(*ws_ptr);
        *ws_ptr += need;
        *ws_size -= need;
        return p;
    }
    *malloc_ptr = malloc(need);
    return (double *)(*malloc_ptr);
}

static void copy_corners_y(double *restrict q, int64_t nx, int64_t ny, int64_t nk) {
    #pragma omp parallel for
    for (int64_t k = 0; k < nk; k += 8) {
        int64_t kend = k + 8 <= nk ? k + 8 : nk;
        for (int64_t kk = k; kk < kend; ++kk) {
            IDX3(q,0,0,kk,ny,nk) = IDX3(q,5,0,kk,ny,nk);
            IDX3(q,1,0,kk,ny,nk) = IDX3(q,5,1,kk,ny,nk);
            IDX3(q,2,0,kk,ny,nk) = IDX3(q,5,2,kk,ny,nk);
            IDX3(q,0,1,kk,ny,nk) = IDX3(q,4,0,kk,ny,nk);
            IDX3(q,1,1,kk,ny,nk) = IDX3(q,4,1,kk,ny,nk);
            IDX3(q,2,1,kk,ny,nk) = IDX3(q,4,2,kk,ny,nk);
            IDX3(q,0,2,kk,ny,nk) = IDX3(q,3,0,kk,ny,nk);
            IDX3(q,1,2,kk,ny,nk) = IDX3(q,3,1,kk,ny,nk);
            IDX3(q,2,2,kk,ny,nk) = IDX3(q,3,2,kk,ny,nk);
            IDX3(q,0,ny-4,kk,ny,nk) = IDX3(q,2,ny-7,kk,ny,nk);
            IDX3(q,0,ny-3,kk,ny,nk) = IDX3(q,1,ny-7,kk,ny,nk);
            IDX3(q,0,ny-2,kk,ny,nk) = IDX3(q,0,ny-7,kk,ny,nk);
            IDX3(q,1,ny-4,kk,ny,nk) = IDX3(q,2,ny-6,kk,ny,nk);
            IDX3(q,1,ny-3,kk,ny,nk) = IDX3(q,1,ny-6,kk,ny,nk);
            IDX3(q,1,ny-2,kk,ny,nk) = IDX3(q,0,ny-6,kk,ny,nk);
            IDX3(q,2,ny-4,kk,ny,nk) = IDX3(q,2,ny-5,kk,ny,nk);
            IDX3(q,2,ny-3,kk,ny,nk) = IDX3(q,1,ny-5,kk,ny,nk);
            IDX3(q,2,ny-2,kk,ny,nk) = IDX3(q,0,ny-5,kk,ny,nk);
            IDX3(q,nx-4,0,kk,ny,nk) = IDX3(q,nx-2,3,kk,ny,nk);
            IDX3(q,nx-4,1,kk,ny,nk) = IDX3(q,nx-3,3,kk,ny,nk);
            IDX3(q,nx-4,2,kk,ny,nk) = IDX3(q,nx-4,3,kk,ny,nk);
            IDX3(q,nx-3,0,kk,ny,nk) = IDX3(q,nx-2,4,kk,ny,nk);
            IDX3(q,nx-3,1,kk,ny,nk) = IDX3(q,nx-3,4,kk,ny,nk);
            IDX3(q,nx-3,2,kk,ny,nk) = IDX3(q,nx-4,4,kk,ny,nk);
            IDX3(q,nx-2,0,kk,ny,nk) = IDX3(q,nx-2,5,kk,ny,nk);
            IDX3(q,nx-2,1,kk,ny,nk) = IDX3(q,nx-3,5,kk,ny,nk);
            IDX3(q,nx-2,2,kk,ny,nk) = IDX3(q,nx-4,5,kk,ny,nk);
            IDX3(q,nx-4,ny-2,kk,ny,nk) = IDX3(q,nx-2,ny-5,kk,ny,nk);
            IDX3(q,nx-4,ny-3,kk,ny,nk) = IDX3(q,nx-3,ny-5,kk,ny,nk);
            IDX3(q,nx-4,ny-4,kk,ny,nk) = IDX3(q,nx-4,ny-5,kk,ny,nk);
            IDX3(q,nx-3,ny-2,kk,ny,nk) = IDX3(q,nx-2,ny-6,kk,ny,nk);
            IDX3(q,nx-3,ny-3,kk,ny,nk) = IDX3(q,nx-3,ny-6,kk,ny,nk);
            IDX3(q,nx-3,ny-4,kk,ny,nk) = IDX3(q,nx-4,ny-6,kk,ny,nk);
            IDX3(q,nx-2,ny-2,kk,ny,nk) = IDX3(q,nx-2,ny-7,kk,ny,nk);
            IDX3(q,nx-2,ny-3,kk,ny,nk) = IDX3(q,nx-3,ny-7,kk,ny,nk);
            IDX3(q,nx-2,ny-4,kk,ny,nk) = IDX3(q,nx-4,ny-7,kk,ny,nk);
        }
    }
}

static void copy_corners_x(double *restrict q, int64_t nx, int64_t ny, int64_t nk) {
    #pragma omp parallel for
    for (int64_t k = 0; k < nk; k += 8) {
        int64_t kend = k + 8 <= nk ? k + 8 : nk;
        for (int64_t kk = k; kk < kend; ++kk) {
            IDX3(q,0,0,kk,ny,nk) = IDX3(q,0,5,kk,ny,nk);
            IDX3(q,0,1,kk,ny,nk) = IDX3(q,1,5,kk,ny,nk);
            IDX3(q,0,2,kk,ny,nk) = IDX3(q,2,5,kk,ny,nk);
            IDX3(q,1,0,kk,ny,nk) = IDX3(q,0,4,kk,ny,nk);
            IDX3(q,1,1,kk,ny,nk) = IDX3(q,1,4,kk,ny,nk);
            IDX3(q,1,2,kk,ny,nk) = IDX3(q,2,4,kk,ny,nk);
            IDX3(q,2,0,kk,ny,nk) = IDX3(q,0,3,kk,ny,nk);
            IDX3(q,2,1,kk,ny,nk) = IDX3(q,1,3,kk,ny,nk);
            IDX3(q,2,2,kk,ny,nk) = IDX3(q,2,3,kk,ny,nk);
            IDX3(q,0,ny-4,kk,ny,nk) = IDX3(q,2,ny-7,kk,ny,nk);
            IDX3(q,0,ny-3,kk,ny,nk) = IDX3(q,1,ny-7,kk,ny,nk);
            IDX3(q,0,ny-2,kk,ny,nk) = IDX3(q,0,ny-7,kk,ny,nk);
            IDX3(q,1,ny-4,kk,ny,nk) = IDX3(q,2,ny-6,kk,ny,nk);
            IDX3(q,1,ny-3,kk,ny,nk) = IDX3(q,1,ny-6,kk,ny,nk);
            IDX3(q,1,ny-2,kk,ny,nk) = IDX3(q,0,ny-6,kk,ny,nk);
            IDX3(q,2,ny-4,kk,ny,nk) = IDX3(q,2,ny-5,kk,ny,nk);
            IDX3(q,2,ny-3,kk,ny,nk) = IDX3(q,1,ny-5,kk,ny,nk);
            IDX3(q,2,ny-2,kk,ny,nk) = IDX3(q,0,ny-5,kk,ny,nk);
            IDX3(q,nx-4,0,kk,ny,nk) = IDX3(q,nx-2,3,kk,ny,nk);
            IDX3(q,nx-3,0,kk,ny,nk) = IDX3(q,nx-2,4,kk,ny,nk);
            IDX3(q,nx-2,0,kk,ny,nk) = IDX3(q,nx-2,5,kk,ny,nk);
            IDX3(q,nx-4,1,kk,ny,nk) = IDX3(q,nx-3,3,kk,ny,nk);
            IDX3(q,nx-3,1,kk,ny,nk) = IDX3(q,nx-3,4,kk,ny,nk);
            IDX3(q,nx-2,1,kk,ny,nk) = IDX3(q,nx-3,5,kk,ny,nk);
            IDX3(q,nx-4,2,kk,ny,nk) = IDX3(q,nx-4,3,kk,ny,nk);
            IDX3(q,nx-3,2,kk,ny,nk) = IDX3(q,nx-4,4,kk,ny,nk);
            IDX3(q,nx-2,2,kk,ny,nk) = IDX3(q,nx-4,5,kk,ny,nk);
            IDX3(q,nx-4,ny-2,kk,ny,nk) = IDX3(q,nx-2,ny-5,kk,ny,nk);
            IDX3(q,nx-3,ny-2,kk,ny,nk) = IDX3(q,nx-2,ny-6,kk,ny,nk);
            IDX3(q,nx-2,ny-2,kk,ny,nk) = IDX3(q,nx-2,ny-7,kk,ny,nk);
            IDX3(q,nx-4,ny-3,kk,ny,nk) = IDX3(q,nx-3,ny-5,kk,ny,nk);
            IDX3(q,nx-3,ny-3,kk,ny,nk) = IDX3(q,nx-3,ny-6,kk,ny,nk);
            IDX3(q,nx-2,ny-3,kk,ny,nk) = IDX3(q,nx-3,ny-7,kk,ny,nk);
            IDX3(q,nx-4,ny-4,kk,ny,nk) = IDX3(q,nx-4,ny-5,kk,ny,nk);
            IDX3(q,nx-3,ny-4,kk,ny,nk) = IDX3(q,nx-4,ny-6,kk,ny,nk);
            IDX3(q,nx-2,ny-4,kk,ny,nk) = IDX3(q,nx-4,ny-7,kk,ny,nk);
        }
    }
}
