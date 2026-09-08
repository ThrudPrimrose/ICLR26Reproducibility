#include <stdint.h>
#include <stdio.h>
#include <omp.h>

static double *d_aa = NULL, *d_bb = NULL;
static const double *d_cc = NULL;
static double *h_aa = NULL, *h_bb = NULL;
static const double *h_cc = NULL;
static int64_t last_N = 0;

static inline void drain(void) {
    int dummy = 0;
    #pragma omp target map(tofrom: dummy)
    dummy = 1;
    (void)dummy;
}

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t N) {
    int64_t NN = N * N;
    (void)d_aa; (void)d_bb; (void)d_cc;
    if (h_aa != aa || h_bb != bb || h_cc != cc || last_N != N) {
        if (d_aa) {
            #pragma omp target exit data map(release: h_aa[0:last_N*last_N]) map(release: h_bb[0:last_N*last_N]) map(release: h_cc[0:last_N*last_N])
            d_aa = d_bb = NULL; d_cc = NULL;
        }
        double e0 = omp_get_wtime();
        #pragma omp target enter data map(to: aa[0:NN]) map(to: bb[0:NN]) map(to: cc[0:NN])
        #pragma omp target data map(to: aa[0:NN]) map(to: bb[0:NN]) map(to: cc[0:NN])
        { d_aa = aa; d_bb = bb; d_cc = cc; }
        double e1 = omp_get_wtime();
        h_aa = aa; h_bb = bb; h_cc = cc; last_N = N;
        printf("ENTER DATA 3x%.0fMB: %.2f ms\n", NN*8/1e6, (e1-e0)*1e3);
        return;
    }
    double t0 = omp_get_wtime();
    #pragma omp target update to(cc[0:NN])
    drain();
    double t1 = omp_get_wtime();
    #pragma omp target update from(aa[0:NN])
    drain();
    double t2 = omp_get_wtime();
    #pragma omp target update from(bb[0:NN])
    drain();
    double t3 = omp_get_wtime();
    #pragma omp target update from(aa[0:NN])
    #pragma omp target update from(bb[0:NN])
    drain();
    double t4 = omp_get_wtime();
    #pragma omp target update to(cc[0:NN])
    #pragma omp target update from(aa[0:NN])
    drain();
    double t5 = omp_get_wtime();
    printf("B H2D cc: %.2f ms -> %.1f GB/s\n", (t1-t0)*1e3, NN*8/(t1-t0)/1e9);
    printf("C1 D2H aa: %.2f ms -> %.1f GB/s\n", (t2-t1)*1e3, NN*8/(t2-t1)/1e9);
    printf("C2 D2H bb: %.2f ms -> %.1f GB/s\n", (t3-t2)*1e3, NN*8/(t3-t2)/1e9);
    printf("D both D2H: %.2f ms -> %.1f GB/s\n", (t4-t3)*1e3, 2.0*NN*8/(t4-t3)/1e9);
    printf("E H2D+D2H same dir order: %.2f ms -> %.1f GB/s\n", (t5-t4)*1e3, 2.0*NN*8/(t5-t4)/1e9);
    fflush(stdout);
}
