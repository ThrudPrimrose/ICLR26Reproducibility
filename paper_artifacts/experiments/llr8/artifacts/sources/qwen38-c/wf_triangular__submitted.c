#include <stdint.h>
#include <omp.h>

/* Bit-exact tiled wavefront for the north+west triangular wavefront
   a[i,j] = (a[i,j] + a[i-1,j]) + a[i,j-1]   (j>=i, i>=1).
   Anti-diagonal (i+j) is the parallel front; we block it into T x T tiles
   so the team re-syncs only once per tile anti-diagonal (~2*N/T barriers).
   Per-cell association matches the reference exactly -> bit-identical
   (including the overflow / +/-inf / nan pattern in the deep region). */
#define WF_TILE 1024

static void tile_block(double *a, int64_t N, int64_t r, int64_t c, int64_t T)
{
    int64_t i0 = r*T; if (i0 < 1) i0 = 1;
    int64_t i1 = (r+1)*T; if (i1 > N) i1 = N;
    int64_t cj = c*T;
    int64_t j1 = (c+1)*T; if (j1 > N) j1 = N;
    for (int64_t i = i0; i < i1; i++) {
        int64_t j0 = cj; if (j0 < i) j0 = i;
        if (j0 >= j1) continue;
        const double *rowm1 = a + (i-1)*N;
        double *row = a + i*N;
        for (int64_t j = j0; j < j1; j++) {
            double t = row[j] + rowm1[j];
            row[j] = t + row[j-1];
        }
    }
}

static void wf_serial(double *a, int64_t N)
{
    for (int64_t i = 1; i < N; i++) {
        const double *rowm1 = a + (i-1)*N;
        double *row = a + i*N;
        for (int64_t j = i; j < N; j++) {
            double t = row[j] + rowm1[j];
            row[j] = t + row[j-1];
        }
    }
}

void wf_triangular_fp64(double *a, int64_t N, uint8_t *ws, int64_t wsbytes)
{
    (void)ws; (void)wsbytes;
    if (N < 2) return;
    if (N <= 2048) { wf_serial(a, N); return; }
    int64_t T = WF_TILE;
    int64_t nb = (N + T - 1) / T;
    #pragma omp parallel
    {
        for (int64_t d = 0; d <= 2*(nb-1); d++) {
            int64_t rmin = d - (nb-1); if (rmin < 0) rmin = 0;
            int64_t rmax = nb - 1; if (rmax > d/2) rmax = d/2;
            #pragma omp for schedule(static)
            for (int64_t r = rmin; r <= rmax; r++)
                tile_block(a, N, r, d - r, T);
        }
    }
}
