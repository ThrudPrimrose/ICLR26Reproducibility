/* TSVC s315 argmax_with_index: running maximum carrying value + index (first occurrence).
 *
 *  - Four interleaved AVX-512 chains of 8 lanes (256B stride each): independent
 *    value/index state per chain -> deep memory-level and instruction parallelism,
 *    one full 4x8 reduction at the end. Per chain: first occurrence of that chain's
 *    max, so the global first-occurrence argmax is the min position among the
 *    chains attaining the global max (and the scalar tail, which comes last).
 *  - Index update skipped when the per-chain compare mask is clear.
 *  - Strict > throughout: exact IEEE `a[i] > x` semantics (NaN, signed zero).
 *  - Large inputs: OpenMP. NUMA handling: once per process, probe each node's
 *    read speed for this array (the harness allocates it on one node) and pin the
 *    whole worker team to the best node -> all reads local.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <immintrin.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>
#include <chrono>

static inline void argmax_range(const double *__restrict a, int64_t n, double &outv, int64_t &outi) {
    if (n < 8) {
        double x = a[0];
        int64_t idx = 0;
        for (int64_t i = 1; i < n; ++i) {
            if (a[i] > x) {
                x = a[i];
                idx = i;
            }
        }
        outv = x;
        outi = idx;
        return;
    }
    if (n < 32) {
        const __m512i OFF = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512d vmax = _mm512_loadu_pd(a);
        __m512i idx = OFF;
        const int64_t nv = n & ~(int64_t)7;
        for (int64_t p = 8; p < nv; p += 8) {
            __m512d v = _mm512_loadu_pd(a + p);
            __mmask8 m = _mm512_cmp_pd_mask(v, vmax, _CMP_GT_OQ);
            vmax = _mm512_mask_mov_pd(vmax, m, v);
            if (m)
                idx = _mm512_mask_mov_epi64(idx, m, _mm512_add_epi64(OFF, _mm512_set1_epi64(p)));
        }
        double v[8];
        int64_t i8[8];
        _mm512_storeu_pd(v, vmax);
        _mm512_storeu_epi64(i8, idx);
        double best = v[0];
        int64_t bidx = i8[0];
        for (int k = 1; k < 8; ++k) {
            if (v[k] > best) {
                best = v[k];
                bidx = i8[k];
            } else if (v[k] == best && i8[k] < bidx) {
                bidx = i8[k];
            }
        }
        if (nv < n) {
            double tv = a[nv];
            int64_t ti = nv;
            for (int64_t i = nv + 1; i < n; ++i) {
                if (a[i] > tv) {
                    tv = a[i];
                    ti = i;
                }
            }
            if (tv > best) {
                best = tv;
                bidx = ti;
            } else if (tv == best && ti < bidx) {
                bidx = ti;
            }
        }
        outv = best;
        outi = bidx;
        return;
    }
    const __m512i OFF0 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    const __m512i OFF1 = _mm512_setr_epi64(8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i OFF2 = _mm512_setr_epi64(16, 17, 18, 19, 20, 21, 22, 23);
    const __m512i OFF3 = _mm512_setr_epi64(24, 25, 26, 27, 28, 29, 30, 31);
    __m512d v0 = _mm512_loadu_pd(a);
    __m512d v1 = _mm512_loadu_pd(a + 8);
    __m512d v2 = _mm512_loadu_pd(a + 16);
    __m512d v3 = _mm512_loadu_pd(a + 24);
    __m512i i0 = OFF0, i1 = OFF1, i2 = OFF2, i3 = OFF3;
    const int64_t nq = n & ~(int64_t)31;
    for (int64_t p = 32; p < nq; p += 32) {
        __m512d x0 = _mm512_loadu_pd(a + p);
        __m512d x1 = _mm512_loadu_pd(a + p + 8);
        __m512d x2 = _mm512_loadu_pd(a + p + 16);
        __m512d x3 = _mm512_loadu_pd(a + p + 24);
        __mmask8 m0 = _mm512_cmp_pd_mask(x0, v0, _CMP_GT_OQ);
        __mmask8 m1 = _mm512_cmp_pd_mask(x1, v1, _CMP_GT_OQ);
        __mmask8 m2 = _mm512_cmp_pd_mask(x2, v2, _CMP_GT_OQ);
        __mmask8 m3 = _mm512_cmp_pd_mask(x3, v3, _CMP_GT_OQ);
        v0 = _mm512_mask_mov_pd(v0, m0, x0);
        v1 = _mm512_mask_mov_pd(v1, m1, x1);
        v2 = _mm512_mask_mov_pd(v2, m2, x2);
        v3 = _mm512_mask_mov_pd(v3, m3, x3);
        if (m0 | m1 | m2 | m3) {
            __m512i bp = _mm512_set1_epi64(p);
            if (m0) i0 = _mm512_mask_mov_epi64(i0, m0, _mm512_add_epi64(OFF0, bp));
            if (m1) i1 = _mm512_mask_mov_epi64(i1, m1, _mm512_add_epi64(OFF1, bp));
            if (m2) i2 = _mm512_mask_mov_epi64(i2, m2, _mm512_add_epi64(OFF2, bp));
            if (m3) i3 = _mm512_mask_mov_epi64(i3, m3, _mm512_add_epi64(OFF3, bp));
        }
    }
    double vv[4][8];
    int64_t ii[4][8];
    _mm512_storeu_pd(vv[0], v0);
    _mm512_storeu_pd(vv[1], v1);
    _mm512_storeu_pd(vv[2], v2);
    _mm512_storeu_pd(vv[3], v3);
    _mm512_storeu_epi64(ii[0], i0);
    _mm512_storeu_epi64(ii[1], i1);
    _mm512_storeu_epi64(ii[2], i2);
    _mm512_storeu_epi64(ii[3], i3);
    double best = vv[0][0];
    int64_t bidx = ii[0][0];
    for (int c = 0; c < 4; ++c) {
        for (int k = 0; k < 8; ++k) {
            if (vv[c][k] > best) {
                best = vv[c][k];
                bidx = ii[c][k];
            } else if (vv[c][k] == best && ii[c][k] < bidx) {
                bidx = ii[c][k];
            }
        }
    }
    if (nq < n) {
        double tv = a[nq];
        int64_t ti = nq;
        for (int64_t i = nq + 1; i < n; ++i) {
            if (a[i] > tv) {
                tv = a[i];
                ti = i;
            }
        }
        if (tv > best) {
            best = tv;
            bidx = ti;
        } else if (tv == best && ti < bidx) {
            bidx = ti;
        }
    }
    outv = best;
    outi = bidx;
}

// ---------------- NUMA: find which node the array lives on, pin the team there ----------------
namespace {
struct NumaState {
    int done;      // 0 not probed, 1 probed
    int ncpus;     // allowed CPUs on best node (0 = keep default)
    cpu_set_t mask;
};
NumaState g_numa = {0, 0, {}};

void parse_cpulist(const char *s, cpu_set_t &set) {
    CPU_ZERO(&set);
    const char *p = s;
    for (;;) {
        while (*p && (*p < '0' || *p > '9') && *p != '-')
            if (*p == '\n' || *p == '\r')
                break;
            else
                ++p;
        if (*p < '0' || *p > '9')
            break;
        int lo = 0;
        while (*p >= '0' && *p <= '9')
            lo = lo * 10 + (*p++ - '0');
        int hi = lo;
        if (*p == '-') {
            ++p;
            while (*p >= '0' && *p <= '9')
                hi = hi * 10 + (*p++ - '0');
        }
        for (int c = lo; c <= hi && c < CPU_SETSIZE; ++c)
            CPU_SET(c, &set);
    }
}

// Once per process: read a small strided slice of `a` while pinned to each node and
// keep the node that reads fastest; pinning the main thread there makes spawned
// OpenMP workers inherit it, so the whole team reads locally.
void probe_numa(const double *a, int64_t n, int cap_threads) {
    if (g_numa.done)
        return;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) {
        g_numa.done = 1; // unsupported: run unpinned
        return;
    }
    int best_ncpus = 0;
    cpu_set_t best_mask;
    CPU_ZERO(&best_mask);
    double best_t = 1e30;
    for (int node = 0; node < 32; ++node) {
        char path[128];
        snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", node);
        FILE *f = fopen(path, "r");
        if (!f)
            break;
        char line[1024];
        if (!fgets(line, sizeof line, f)) {
            fclose(f);
            break;
        }
        fclose(f);
        cpu_set_t node_set;
        parse_cpulist(line, node_set);
        cpu_set_t inter;
        CPU_ZERO(&inter);
        for (int c = 0; c < CPU_SETSIZE; ++c)
            if (CPU_ISSET(c, &node_set) && CPU_ISSET(c, &allowed))
                CPU_SET(c, &inter);
        int cnt = 0;
        for (int c = 0; c < CPU_SETSIZE; ++c)
            if (CPU_ISSET(c, &inter))
                ++cnt;
        if (cnt == 0)
            continue;
        cpu_set_t old;
        if (sched_getaffinity(0, sizeof old, &old) != 0) {
            g_numa.done = 1;
            return;
        }
        if (sched_setaffinity(0, sizeof inter, &inter) != 0)
            continue;
        const double *__restrict src = a + (n > 131072 ? n / 2 : 0);
        volatile double sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 16384; ++j)
                sink += src[k * 16384 + j];
        auto t1 = std::chrono::steady_clock::now();
        double t = std::chrono::duration<double>(t1 - t0).count();
        sched_setaffinity(0, sizeof old, &old);
        if (t < best_t) {
            best_t = t;
            best_ncpus = cnt;
            best_mask = inter;
        }
        (void)sink;
    }
    g_numa.done = 1;
    // Pin the whole team to the data's node only when that node offers at least
    // the full thread set (concentration wins); otherwise stay distributed.
    if (best_ncpus >= cap_threads) {
        g_numa.ncpus = cap_threads;
        g_numa.mask = best_mask;
        sched_setaffinity(0, sizeof g_numa.mask, &g_numa.mask);
    }
}
} // namespace

extern "C" void argmax_with_index_fp64(const double *__restrict a, int64_t *__restrict out_index,
                                       double *__restrict out_value, const int64_t LEN_1D) {
    double gval;
    int64_t gidx;
    int nt = omp_get_max_threads();
    if (nt > 128)
        nt = 128;
    if (nt < 2 || LEN_1D < 8388608) {
        argmax_range(a, LEN_1D, gval, gidx);
    } else {
        probe_numa(a, LEN_1D, nt);
        int pnt = (g_numa.ncpus > 0) ? g_numa.ncpus : nt;
        int64_t first = (LEN_1D + pnt - 1) / pnt;
        argmax_range(a, first, gval, gidx);
#pragma omp parallel num_threads(pnt - 1)
        {
            int64_t tid = omp_get_thread_num() + 1;
            int64_t lo = tid * first;
            if (lo < LEN_1D) {
                int64_t hi = lo + first;
                if (hi > LEN_1D)
                    hi = LEN_1D;
                double tv;
                int64_t ti;
                argmax_range(a + lo, hi - lo, tv, ti);
                ti += lo;
#pragma omp critical
                {
                    if (tv > gval) {
                        gval = tv;
                        gidx = ti;
                    } else if (tv == gval && ti < gidx) {
                        gidx = ti;
                    }
                }
            }
        }
    }
    *out_value = gval;
    *out_index = gidx;
}
