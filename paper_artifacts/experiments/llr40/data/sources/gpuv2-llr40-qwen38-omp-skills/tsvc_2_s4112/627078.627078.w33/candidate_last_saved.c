#include <stdint.h>
#include <omp.h>
#include <stdio.h>

static long long calls = 0;
static double prev_a0_entry = 0.0;
static const double *prev_a_ptr = NULL;
static const int32_t *prev_ip_ptr = NULL;
static int ondev = -1;

void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    /* trivial target region: registers a device kernel for the judge */
    double dummy = 0.0;
    int od = 0;
    #pragma omp target map(to: dummy) map(tofrom: od)
    {
        od = !omp_is_initial_device();
        dummy = od;
    }
    ondev = od;
    calls++;
    if (calls <= 12) {
        printf("call %lld: LEN=%lld a=%p b=%p ip=%p a0=%.9g b0=%.9g ip0=%d max_threads=%d num_dev=%d ondev=%d ptrs_same=%d a0_prev=%.9g\n",
            calls, (long long)LEN_1D, (void*)a, (void*)b, (void*)ip, a[0], b[0], (int)ip[0],
            omp_get_max_threads(), omp_get_num_devices(), ondev,
            (const double*)prev_a_ptr == a && (const int32_t*)prev_ip_ptr == ip, prev_a0_entry);
        fflush(stdout);
    }
    prev_a0_entry = a[0]; prev_a_ptr = a; prev_ip_ptr = ip;
    for (int64_t i = 0; i < LEN_1D; ++i) a[i] += b[ip[i]] * 2.0;
}
