#include <stdint.h>
#include <stdio.h>

void tsvc_2_s3112_fp64(double *a, double *b, int64_t n, uint8_t *ws, int64_t wsn)
{
    (void)ws; (void)wsn;
    static long calls = 0;
    calls++;
    if (calls <= 3 || (calls % 8 == 0)) { fprintf(stdout, "CALL %ld n=%lld\n", calls, (long long)n); fflush(stdout); }
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) { s = s + a[i]; b[i] = s; }
}
