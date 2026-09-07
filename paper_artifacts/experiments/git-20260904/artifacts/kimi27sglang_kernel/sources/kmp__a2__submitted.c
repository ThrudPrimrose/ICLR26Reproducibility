#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void kmp_fp64(int64_t *text, int64_t *pattern, int64_t *matches,
              int64_t N, int64_t M, uint8_t *workspace, int64_t workspace_size) {
    (void)text;
    (void)pattern;

    printf("N=%ld M=%ld workspace=%p size=%ld\n", (long)N, (long)M, (void*)workspace, (long)workspace_size);

    int64_t *p = (int64_t*)malloc(8);
    if (p) {
        *p = 123;
        printf("malloc ok p=%p val=%ld\n", (void*)p, (long)*p);
        free(p);
    } else {
        printf("malloc failed\n");
    }

    fflush(stdout);
    matches[0] = N + M;
}
