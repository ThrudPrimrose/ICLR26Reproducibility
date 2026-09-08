import os, sys
def vpvts(a, b, LEN_1D, S):
    aff = sorted(os.sched_getaffinity(0))
    try:
        siblings = {}
        for c in aff:
            try:
                sibs = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
                siblings[c] = sibs
            except OSError:
                siblings[c] = "?"
    except Exception as e:
        siblings = str(e)
    import ctypes
    mem = "n/a"
    try:
        mem = open("/proc/meminfo").readline().strip()
    except Exception:
        pass
    print("cpu_count:", os.cpu_count(), "affinity_n:", len(aff), "aff:", aff[:24], ("..." if len(aff)>24 else ""))
    print("env OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"), "NUMBA_NUM_THREADS:", os.environ.get("NUMBA_NUM_THREADS"))
    print("siblings sample:", dict(list(siblings.items())[:4]) if isinstance(siblings, dict) else siblings)
    print(mem)
    try:
        import numba
        from numba import config as nbc
        print("numba threads config:", nbc.NUMBA_NUM_THREADS)
        print("numba threading layer:", nbc.THREADING_LAYER)
    except Exception as e:
        print("numba err", e)
    try:
        import torch
        print("torch omp threads:", torch.get_num_threads())
        print("torch available:", torch.cuda.is_available(), torch.version.hip)
    except Exception as e:
        print("torch err", e)
    sys.stdout.flush()
    a[:LEN_1D] += b[:LEN_1D] * S
    return None
