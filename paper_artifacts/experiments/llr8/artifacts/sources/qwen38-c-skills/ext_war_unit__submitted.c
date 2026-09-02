#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <stdlib.h>

#define MAXT 256
#define WU_THRESH 262144

static pthread_t pool[MAXT];
static int pool_nt = 0;
static sem_t gate1;
static sem_t gate2;
static volatile int p1done = 0;
static volatile int p2done = 0;
static int64_t sp_lo[MAXT], sp_hi[MAXT];
static double *g_a;
static const double *g_b;
static double g_bound[MAXT * 8];
static int g_nt;
static int nt_cache = -1;

static void *worker(void *arg)
{
    int t = (int)(intptr_t)arg;
    for (;;) {
        sem_wait(&gate1);
        if (sp_hi[t] > sp_lo[t])
            g_bound[t * 8] = g_a[sp_hi[t]];
        __sync_synchronize();
        if (__sync_add_and_fetch(&p1done, 1) == g_nt) {
            __sync_synchronize();
            for (int i = 0; i < g_nt; i++) sem_post(&gate2);
        }
        sem_wait(&gate2);
        {
            int64_t lo = sp_lo[t];
            int64_t hi = sp_hi[t];
            double *a = g_a;
            const double *b = g_b;
            int64_t i;
            for (i = lo; i + 1 < hi; i++) a[i] = a[i + 1] + b[i];
            if (i < hi) a[i] = g_bound[t * 8] + b[i];
        }
        __sync_synchronize();
        __sync_add_and_fetch(&p2done, 1);
    }
    return (void *)0;
}

static void ensure_pool(int nt)
{
    if (pool_nt == nt) return;
    for (int i = 0; i < pool_nt; i++) pthread_join(pool[i], NULL);
    pool_nt = nt;
    if (nt <= 1) return;
    sem_init(&gate1, 0, 0);
    sem_init(&gate2, 0, 0);
    for (int i = 0; i < nt; i++) {
        if (pthread_create(&pool[i], NULL, worker, (void *)(intptr_t)i) != 0) {
            for (int j = 0; j < i; j++) pthread_join(pool[j], NULL);
            pool_nt = 0;
            return;
        }
    }
}

static int want_threads(void)
{
    if (nt_cache > 0) return nt_cache;
    int n = -1;
    const char *e = getenv("OMP_NUM_THREADS");
    if (e && *e) n = atoi(e);
    if (n < 2) n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 2) n = 2;
    if (n > MAXT) n = MAXT;
    nt_cache = n;
    return n;
}

void ext_war_unit_fp64(double *a, double *b, int64_t N, uint8_t *ws, int64_t ws_bytes)
{
    (void)ws; (void)ws_bytes;
    const int64_t n1 = N - 1;
    if (n1 <= 0) return;

    int nt = want_threads();
    if (N >= WU_THRESH && nt >= 2) {
        ensure_pool(nt);
        if (pool_nt >= 2) {
            g_nt = pool_nt;
            int64_t base = n1 / g_nt;
            int64_t rem = n1 % g_nt;
            int64_t off = 0;
            for (int t = 0; t < g_nt; t++) {
                sp_lo[t] = off;
                int64_t s = base + (t < rem ? 1 : 0);
                off += s;
                sp_hi[t] = off;
            }
            g_a = a;
            g_b = b;
            __sync_synchronize();
            p1done = 0;
            p2done = 0;
            for (int i = 0; i < g_nt; i++) sem_post(&gate1);
            while (__sync_add_and_fetch(&p2done, 0) < g_nt)
                __builtin_ia32_pause();
            return;
        }
    }
    for (int64_t i = 0; i < n1; i++) a[i] = a[i + 1] + b[i];
}
