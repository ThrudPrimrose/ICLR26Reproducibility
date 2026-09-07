/* dfa: run a DFA over a symbol stream, tally state visits.
 *
 * The state recurrence is strictly serial, so we use the block-map trick:
 *   phase 1: split the stream into blocks; each thread traces all NS
 *            possible start states through one block (NS independent
 *            chains, SIMD gathers) -> block end-state maps.
 *   phase 2: resolve the true start state of each block (M map lookups).
 *   phase 3: re-run each block from its true start state (parallel,
 *            several blocks per thread interleaved for ILP), tallying
 *            per-thread partial histograms, merged at the end.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

static void dfa_seq(const int64_t *__restrict trans, const int64_t *__restrict symbols,
                    int64_t *__restrict counts, int64_t N, int64_t NS, int64_t NA) {
    int64_t state = 0;
    for (int64_t i = 0; i < N; i++) {
        state = trans[state * NA + symbols[i]];
        counts[state] += 1;
    }
    (void)NS;
}

/* Phase 1, one block: trace every start state through symbols[b0,b1).
 * Lanes are 32-bit (NS < 2^31). Writes NS results to map32. */
static void block_map(const int32_t *__restrict tab32, const int64_t *__restrict symbols,
                      int64_t b0, int64_t b1, int64_t NS, int64_t NA, int32_t *__restrict map32) {
    if (NS <= 16) {
        __mmask16 m = (__mmask16)((1u << NS) - 1);
        __m256i lane = (NS >= 12) ? _mm256_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)
                                  : (NS >= 8)  ? _mm256_set_epi32(0,0,0,0,7,6,5,4,3,2,1,0,0,0,0,0)
                                  : (NS >= 4)  ? _mm256_set_epi32(0,0,0,0,0,0,0,0,3,2,1,0,0,0,0,0)
                                  : _mm256_set_epi32(0,0,0,0,0,0,0,0,(NS==3)?2:0,(NS==2)?1:0,(NS==1)?0:0,0,0,0,0,0);
        int na = (int)NA;
        for (int64_t i = b0; i < b1; i++) {
            __m256i idx = _mm256_add_epi32(_mm256_mullo_epi32(lane, _mm256_set1_epi32(na)),
                                           _mm256_set1_epi32((int)symbols[i]));
            lane = _mm256_maskz_epi32(m, _mm256_i32gather_epi32(tab32, idx, 4));
        }
        _mm256_storeu_si256((__m256i *)map32, lane);
        return;
    }
    __mmask32 m = (__mmask32)((1ull << NS) - 1);
    __m512i lane[2];
    int nl = (int)((NS + 31) / 32);
    lane[0] = _mm512_set_epi32(31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,
                               15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    if (nl == 2) lane[1] = _mm512_set1_epi32(0);
    int na = (int)NA;
    for (int64_t i = b0; i < b1; i++) {
        __m512i s = _mm512_set1_epi32((int)symbols[i]);
        for (int k = 0; k < nl; k++) {
            __m512i idx = _mm512_add_epi32(_mm512_mullo_epi32(lane[k], _mm512_set1_epi32(na)), s);
            lane[k] = _mm512_maskz_epi32(m, _mm512_i32gather_epi32(tab32, idx, 4));
        }
    }
    _mm512_storeu_si512(map32, lane[0]);
    if (nl == 2) _mm512_storeu_si512(map32 + 32, lane[1]);
}

static void dfa_map(const int64_t *__restrict trans, const int64_t *__restrict symbols,
                    int64_t *__restrict counts, int64_t N, int64_t NS, int64_t NA) {
    int P = omp_get_max_threads();
    int64_t M = (int64_t)P * 8;
    int64_t B = (N + M - 1) / M;
    if (B < 1024) { B = 1024; }
    int64_t Mact = (N + B - 1) / B;

    /* int32 copy of the table so 32-bit gathers work */
    int32_t *tab32 = malloc((size_t)NS * (size_t)NA * 4);
    int32_t *maps  = malloc((size_t)Mact * (size_t)NS * 4);
    int64_t *starts = malloc((size_t)Mact * 8);
    int64_t *parts  = malloc((size_t)P * (size_t)NS * 8);
    if (!tab32 || !maps || !starts || !parts) {
        free(tab32); free(maps); free(starts); free(parts);
        dfa_seq(trans, symbols, counts, N, NS, NA); return;
    }
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < NS * NA; i++) tab32[i] = (int32_t)trans[i];

    /* phase 1: per-block maps */
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < Mact; j++) {
        int64_t b0 = j * B, b1 = b0 + B; if (b1 > N) b1 = N;
        block_map(tab32, symbols, b0, b1, NS, NA, maps + j * NS);
    }
    /* phase 2: resolve block start states */
    int64_t st = 0;
    for (int64_t j = 0; j < Mact; j++) { starts[j] = st; st = maps[j * NS + st]; }
    /* phase 3: re-run blocks; K blocks per thread interleaved */
    {
        int K = 4;
        if (Mact < K) K = (int)Mact;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            int64_t *my = parts + (int64_t)tid * NS;
            for (int64_t s = 0; s < NS; s++) my[s] = 0;
            /* static contiguous chunk of blocks */
            int64_t c0 = (int64_t)tid * (Mact / nt) + (tid < (int)(Mact % nt) ? tid : Mact % nt);
            int64_t c1 = c0 + Mact / nt + (tid < (int)(Mact % nt) ? 1 : 0);
            if (c1 < Mact) c1 = Mact;
            int64_t k;
            for (k = 0; k + K <= c1; k += K) {
                int64_t stt[K]; int64_t p0[K];
                for (int q = 0; q < K; q++) {
                    stt[q] = starts[k + q];
                    p0[q]  = (k + q) * B;
                }
                int64_t base = p0[0], len = B;
                if (p0[K-1] + B > N) len = N - p0[0];
                for (int64_t i = 0; i < len; i++) {
                    int64_t a[K];
                    for (int q = 0; q < K; q++) {
                        int64_t sym = symbols[base + q * B + i];
                        a[q] = tab32[stt[q] * NA + sym];
                        my[a[q]] += 1;
                    }
                    for (int q = 0; q < K; q++) stt[q] = a[q];
                }
            }
            for (; k < c1; k++) {
                int64_t b0 = k * B, b1 = b0 + B; if (b1 > N) b1 = N;
                int64_t s2 = starts[k];
                for (int64_t i = b0; i < b1; i++) {
                    s2 = tab32[s2 * NA + symbols[i]];
                    my[s2] += 1;
                }
            }
        }
    }
    /* merge */
    for (int64_t s = 0; s < NS; s++) counts[s] = 0;
    for (int t = 0; t < P; t++)
        for (int64_t s = 0; s < NS; s++) counts[s] += parts[(int64_t)t * NS + s];

    free(tab32); free(maps); free(starts); free(parts);
}

void dfa_int64(int64_t *restrict trans, int64_t *restrict symbols,
               int64_t *restrict counts, int64_t N, int64_t NS, int64_t NA) {
    if (N <= 0 || NS <= 0) return;
    int P = omp_get_max_threads();
    if (P >= 2 && NS <= 64 && NA <= (1 << 30) && NS * NA <= (1 << 20) &&
        (double)N * (double)NS <= (double)1 << 29 && N >= (1 << 15)) {
        dfa_map(trans, symbols, counts, N, NS, NA);
    } else {
        dfa_seq(trans, symbols, counts, N, NS, NA);
    }
}
