import os, sys, time

def s235(a, b, c, aa, bb, LEN_2D):
    L = int(LEN_2D)
    lines = []
    try:
        aff = os.sched_getaffinity(0)
        lines.append(f"affinity count: {len(aff)} min {min(aff)} max {max(aff)}")
    except Exception as e:
        lines.append(f"affinity: {type(e).__name__}")
    for f in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"):
        try:
            lines.append(f"{f}: {open(f).read().strip()}")
        except Exception:
            pass
    lines.append(f"cpu_count: {os.cpu_count()}")
    try:
        import numba
        lines.append(f"numba: {numba.__version__}")
        lines.append(f"numba default threads: {numba.config.NUMBA_NUM_THREADS}")
        from numba.np.ufunc import parallel  # noqa
    except Exception as e:
        lines.append(f"numba: {type(e).__name__} {e}")
    try:
        import numpy
        lines.append(f"numpy: {numpy.__version__}")
    except Exception as e:
        lines.append(f"numpy: {e}")
    lines.append(f"python: {sys.version.split()[0]}")
    # L3 size
    try:
        lines.append(f"l3: {open('/sys/devices/system/cpu/cpu0/cache/index3/size').read().strip()}")
    except Exception:
        pass
    # quick 2-thread streaming read bandwidth test
    try:
        x = a
        t0 = time.perf_counter()
        s = 0.0
        for k in range(20):
            s += x.sum()
        t1 = time.perf_counter()
        lines.append(f"numpy sum 1x{8*L*8/1e6:.0f}MB: {(t1-t0)*1e3:.2f} ms -> {8*L*8*20/((t1-t0)*1e9):.1f} GB/s (1 thread)")
    except Exception as e:
        lines.append(f"bwtest: {e}")
    print("\n".join(lines), flush=True)
    # do the real work (serial reference) - L small? do it fully, it's fine
    for i in range(L):
        a[i] = a[i] + b[i] * c[i]
        for j in range(1, L):
            aa[j, i] = aa[j - 1, i] + bb[j, i] * a[i]
    sys.stdout.flush()
