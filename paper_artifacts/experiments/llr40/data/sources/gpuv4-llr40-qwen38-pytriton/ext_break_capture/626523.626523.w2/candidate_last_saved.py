import numpy as np
import time
import threading
import os

def _chunk_scan(a, K, lo, hi, fwd, CH, stop):
    i = lo
    if fwd:
        while i < hi:
            if stop.is_set():
                return None
            j = min(i + CH, hi)
            if a[i:j].max() > K:
                return i + int(np.argmax(a[i:j] > K))
            i = j
    else:
        i = hi
        while i > lo:
            if stop.is_set():
                return None
            j = max(i - CH, lo)
            if a[j:i].max() > K:
                return j + (i - j) - 1 - int(np.argmax((a[j:i] > K)[::-1]))
            i = j
    return None

def grid_test(a, K, n, L, R, nt, CH):
    stop = threading.Event()
    res = []
    def worker(lo, hi, fwd):
        r = _chunk_scan(a, K, lo, hi, fwd, CH, stop)
        if r is not None:
            res.append(r)
            stop.set()
    band = R - L
    ths = []
    for s in range(nt):
        lo = L + band * s // nt
        hi = L + band * (s + 1) // nt
        fwd = (s % 2 == 0)
        if fwd:
            ths.append(threading.Thread(target=worker, args=(lo, hi, True)))
        else:
            ths.append(threading.Thread(target=worker, args=(lo, hi + 1, False)))
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    stop.wait(timeout=60)
    for t in ths:
        t.join(timeout=60)
    dt = time.perf_counter() - t0
    c = res[0] if res else -1
    return dt, c

def ext_break_capture(a, out_index, out_value, LEN_1D, K=1.0):
    n = a.shape[0]
    K = float(K)
    lines = []
    c_true = 0
    found_any = False
    i = 0
    while i < n:
        j = min(i + (1 << 20), n)
        m = a[i:j] > K
        if m.any():
            c_true = i + int(np.argmax(m))
            found_any = True
            break
        i = j
    lines.append(f"n={n} K={K} crossing={c_true} frac={c_true/n:.4f} nproc={os.cpu_count()}")
    for (bf_lo, bf_hi) in ((0.39, 0.71),):
        L = int(n * bf_lo); R = int(n * bf_hi)
        for nt in (1, 2, 4, 8, 12):
            for CH in (1 << 20, 1 << 25):
                dt, c = grid_test(a, K, n, L, R, nt, CH)
                ok = "OK " if c == c_true else "BAD"
                lines.append(f"band[{bf_lo},{bf_hi}] nt={nt:2d} CH={CH>>20:2d}MB: {dt*1e3:8.3f} ms {ok} (frac={c/n if c>=0 else -1:.4f})")
    out_index[0] = c_true
    out_value[0] = a[c_true] if found_any else -1.0
    print("\n".join(lines), flush=True)
    return None
