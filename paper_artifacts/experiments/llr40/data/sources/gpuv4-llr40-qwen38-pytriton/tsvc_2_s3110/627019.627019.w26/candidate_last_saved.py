import os, sys, time
import numpy as np
import concurrent.futures as cf

def _bw_threads(a, K, fn, ms=250):
    n = a.size
    base, rem = divmod(n, K)
    bounds = [i * base + min(i, rem) for i in range(K + 1)]
    chs = [(bounds[i], bounds[i+1]) for i in range(K)]
    def one(c):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < ms / 1000:
            fn(a[c[0]:c[1]])
        return 1
    with cf.ThreadPoolExecutor(K) as ex:
        t0 = time.perf_counter()
        list(ex.map(one, chs))
        dt = time.perf_counter() - t0
    return a.nbytes / dt / 1e9

def s3110(aa, bb, LEN_2D):
    out = []
    p = out.append
    a = aa.reshape(-1)
    p("nbytes: %d  n: %d" % (aa.nbytes, a.size))
    t0 = time.perf_counter(); np.sum(a); t1 = time.perf_counter()
    p("single sum : %.1f ms  %.0f GB/s" % ((t1-t0)*1e3, aa.nbytes/(t1-t0)/1e9))
    t0 = time.perf_counter(); np.max(a); t1 = time.perf_counter()
    p("single max : %.1f ms  %.0f GB/s" % ((t1-t0)*1e3, aa.nbytes/(t1-t0)/1e9))
    t0 = time.perf_counter(); a.sum(dtype=np.float64); t1 = time.perf_counter()
    p("single a.sum: %.1f ms  %.0f GB/s" % ((t1-t0)*1e3, aa.nbytes/(t1-t0)/1e9))
    K = len(os.sched_getaffinity(0))
    for K2 in (12, 24, 48):
        bw = _bw_threads(a, K2, np.sum, 250)
        p("sum  K=%-3d aggregate %.0f GB/s" % (K2, bw))
    for K2 in (24,):
        bw = _bw_threads(a, K2, np.max, 250)
        p("max  K=%-3d aggregate %.0f GB/s" % (K2, bw))
    p("numactl: " + os.popen("numactl --hardware 2>/dev/null").read()[:1200])
    p("cpuset: " + os.popen("cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null").read().strip())
    # correct result
    idx = int(a.argmax())
    bb[0, 0] = a[idx] + (idx // LEN_2D) + (idx % LEN_2D)
    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()
