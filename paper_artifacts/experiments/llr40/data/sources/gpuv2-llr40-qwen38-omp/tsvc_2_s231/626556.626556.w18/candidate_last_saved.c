#include <stdint.h>
#include <omp.h>

/* tsvc_2 s231:  aa[j*L+i] = aa[j-1][i] + bb[j][i]  (j = 1..L-1), per column i.
 *
 * Result of column i:  aa_new[j] = a0 + (bb_1 + ... + bb_j),
 * with a0 = aa[i] (row 0, never written) and bb_k = bb[k*L+i].
 *
 * Big L: run on the GPU (MI300A: device and host share the HBM, so a map
 * round trip is an HBM-to-HBM copy, not PCIe).  One 256-lane team per
 * column; lane t owns rows [t*C, t*C+C).  Each lane sums its chunk, the
 * team scans the 256 chunk sums in shared memory, and each lane reapplies
 * its chunk with the prefix offset.  Memory traffic is fully coalesced
 * row-streams; the only per-lane chains are C adds.
 *
 * Small L: stay on the host, where the region launch would dominate.
 * The register-form scan (a = aa[i]; a += bb[j*L+i]; aa[j*L+i] = a) adds
 * in exactly the reference order (bit-identical) and never reads aa
 * except row 0. */

#define S231_NT 256

static void s231_host_cols(double *restrict aa, const double *restrict bb,
                           const int64_t L, int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
        double a = aa[i];
        for (int64_t j = 1; j < L; ++j) {
            a += bb[j * L + i];
            aa[j * L + i] = a;
        }
    }
}

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb,
                      const int64_t LEN_2D) {
    const int64_t L = LEN_2D;
    if (L <= 1) return;

    if (L < 512) {
        if (L <= 128) {
            s231_host_cols(aa, bb, L, 0, L);
        } else {
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < L; ++i) {
                double a = aa[i];
                for (int64_t j = 1; j < L; ++j) {
                    a += bb[j * L + i];
                    aa[j * L + i] = a;
                }
            }
        }
        return;
    }

    const int64_t n = L * L;
    const int64_t C = (L + S231_NT - 1) / S231_NT;

    #pragma omp target data map(to: bb[0:n]) map(tofrom: aa[0:n])
    {
        double smem[S231_NT];
        #pragma omp target teams
        {
            #pragma omp distribute shared(smem)
            for (int64_t i = 0; i < L; ++i) {
            const int tid = omp_get_thread_num();
            const int64_t j0 = (int64_t)tid * C;
            int64_t j1 = j0 + C;
            if (j1 > L) j1 = L;
            const double a0 = aa[i];
            const double b0 = bb[i];

            double s = 0.0;
            for (int64_t j = j0; j < j1; ++j)
                s += bb[j * L + i];
            smem[tid] = s;
            #pragma omp barrier
            for (int d = 1; d < S231_NT; d <<= 1) {
                if (tid >= d)
                    smem[tid] += smem[tid - d];
                #pragma omp barrier
            }
            const double off = smem[tid] - s - b0 + a0;

            double r = 0.0;
            for (int64_t j = j0; j < j1; ++j) {
                r += bb[j * L + i];
                aa[j * L + i] = r + off;
            }
        }
    }
}
