#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

/* a[i] = 0.75 * a[i-K] + b[i] * c[i]  for i = K .. LEN_1D-1  (in place).
 *
 * The K chains r = 0..K-1 (indices r, r+K, ...) are independent; each is a
 * first-order linear recurrence.  Two exact parallel forms are used:
 *
 *  two-pass block form (K <= 512, few chains): split every chain into
 *    B-step blocks.  pass 1 (parallel) computes each block's carry C_k
 *    (chain run from zero); a tiny serial chain folds S_{k+1} =
 *    0.75^st * S_k + C_k over the K chains; pass 2 (parallel) re-runs
 *    each block from S_k and stores a.
 *
 *  long-diagonal form (K > 512, many chains): each parallel unit owns
 *    W contiguous chains and streams the whole chain length, SIMD over
 *    the chains, one unit-stride-8*W byte chunk per step.
 */

#define VDU_B 256

static void vdu_serial1(double *a, double *b, double *c, int64_t LEN) {
    double x = a[0];
    for (int64_t i = 1; i < LEN; i++) { x = 0.75 * x + b[i] * c[i]; a[i] = x; }
}

/* K == 1: dedicated scalar two-pass form (no vector scaffolding). */
static void vdu_k1(double *a, double *b, double *c, int64_t LEN) {
    const int64_t B = VDU_B;
    int64_t N = LEN - 1;
    if (N <= 0) return;
    int64_t NB = (N + B - 1) / B;
    double *C = malloc(NB * sizeof(double));
    double *S = malloc(NB * sizeof(double));
    if (!C || !S) { free(C); free(S); vdu_serial1(a, b, c, LEN); return; }
    double rB = 1.0;
    for (int64_t j = 0; j < B; j++) rB *= 0.75;

    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < NB; k++) {
        int64_t i0 = k * B + 1;
        int64_t i1 = i0 + B; if (i1 > LEN) i1 = LEN;
        double y = 0.0;
        for (int64_t i = i0; i < i1; i++) y = 0.75 * y + b[i] * c[i];
        C[k] = y;
    }
    S[0] = a[0];
    for (int64_t k = 0; k + 1 < NB; k++) S[k + 1] = rB * S[k] + C[k];

    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < NB; k++) {
        int64_t i0 = k * B + 1;
        int64_t i1 = i0 + B; if (i1 > LEN) i1 = LEN;
        double y = S[k];
        for (int64_t i = i0; i < i1; i++) { y = 0.75 * y + b[i] * c[i]; a[i] = y; }
    }
    free(C); free(S);
}

/* two-pass block form for small K (W = K chains in one SIMD unit). */
static void vdu_smallk(double *a, double *b, double *c, int64_t K, int64_t LEN) {
    const int64_t B = VDU_B;
    int64_t M0 = (LEN - 1) / K;
    if (M0 <= 0) return;
    int64_t NB = (M0 + B - 1) / B;
    double *C = malloc((size_t)NB * K * sizeof(double));
    double *S = malloc((size_t)(NB + 1) * K * sizeof(double));
    if (!C || !S) {
        free(C); free(S);
        for (int64_t i = K; i < LEN; i++) a[i] = 0.75 * a[i - K] + b[i] * c[i];
        return;
    }
    double pt[B + 1];
    int64_t *Mr = malloc(K * sizeof(int64_t));
    pt[0] = 1.0;
    for (int64_t j = 1; j <= B; j++) pt[j] = 0.75 * pt[j - 1];
    for (int64_t r = 0; r < K; r++) { Mr[r] = (LEN - 1 - r) / K; S[r] = a[r]; }

    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < NB; k++) {
        double y[512];
        for (int64_t r = 0; r < K; r++) y[r] = 0.0;
        int64_t Mr0 = Mr[0];
        int64_t m_end = (k + 1) * B; if (m_end > Mr0) m_end = Mr0;
        for (int64_t m = k * B + 1; m <= m_end; m++) {
            int64_t i0 = m * K;
            int64_t n = LEN - i0; if (n > K) n = K;
            #pragma omp simd
            for (int64_t r = 0; r < n; r++)
                y[r] = 0.75 * y[r] + b[i0 + r] * c[i0 + r];
        }
        double *Ck = C + k * K;
        for (int64_t r = 0; r < K; r++) Ck[r] = y[r];
    }

    for (int64_t k = 0; k < NB; k++) {
        double *Sk = S + k * K, *Sk1 = S + (k + 1) * K;
        const double *Ck = C + k * K;
        for (int64_t r = 0; r < K; r++) {
            int64_t st = Mr[r] - k * B;
            if (st <= 0)       Sk1[r] = Sk[r];
            else if (st < B)   Sk1[r] = pt[st] * Sk[r] + Ck[r];
            else               Sk1[r] = pt[B] * Sk[r] + Ck[r];
        }
    }

    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < NB; k++) {
        double y[512];
        const double *Sk = S + k * K;
        for (int64_t r = 0; r < K; r++) y[r] = Sk[r];
        int64_t Mr0 = Mr[0];
        int64_t m_end = (k + 1) * B; if (m_end > Mr0) m_end = Mr0;
        for (int64_t m = k * B + 1; m <= m_end; m++) {
            int64_t i0 = m * K;
            int64_t n = LEN - i0; if (n > K) n = K;
            #pragma omp simd
            for (int64_t r = 0; r < n; r++) {
                y[r] = 0.75 * y[r] + b[i0 + r] * c[i0 + r];
                a[i0 + r] = y[r];
            }
        }
    }
    free(Mr); free(C); free(S);
}

/* long-diagonal form for large K: W contiguous chains per unit, full length. */
static void vdu_largek(double *a, double *b, double *c, int64_t K, int64_t LEN) {
    const int64_t W = 32;
    int64_t ncb = (K + W - 1) / W;
    #pragma omp parallel for schedule(static)
    for (int64_t cb = 0; cb < ncb; cb++) {
        int64_t r0 = cb * W;
        int64_t Wd = K - r0; if (Wd > W) Wd = W;
        int64_t M = (LEN - 1 - r0) / K;
        int64_t i0 = K;
        for (int64_t m = 1; m <= M; m++, i0 += K) {
            int64_t n = LEN - i0 - r0; if (n > Wd) n = Wd;
            #pragma omp simd
            for (int64_t r = 0; r < n; r++)
                a[i0 + r0 + r] = 0.75 * a[i0 + r0 + r - K] + b[i0 + r0 + r] * c[i0 + r0 + r];
        }
    }
}

void versioned_distance_update_fp64(double *a, double *b, double *c,
                                    int64_t K, int64_t LEN_1D,
                                    uint8_t *workspace, int64_t workspace_bytes) {
    (void)workspace; (void)workspace_bytes;
    if (K <= 0 || K >= LEN_1D) return;
    if (K == 1)       vdu_k1(a, b, c, LEN_1D);
    else if (K <= 512) vdu_smallk(a, b, c, K, LEN_1D);
    else               vdu_largek(a, b, c, K, LEN_1D);
}
