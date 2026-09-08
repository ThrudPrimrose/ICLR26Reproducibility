import os, sys, time
import numpy as np
from numba import njit, prange
import numba


@njit(parallel=True)
def _kern(aa, bb, cc, a, b, c, d, M):
    for idx in prange(M):
        aa[idx] = aa[idx] + bb[idx] * cc[idx]
    for i in range(a.shape[0]):
        a[i] = b[i] + c[i] * d[i]


BENCH_CODE = r'''
import os, sys, time
os.environ["NUMBA_ENABLE_AVX"] = sys.argv[1]
import numpy as np
from numba import njit, prange
@njit(parallel=True)
def k(aa, bb, cc, a, b, c, d, M):
    for idx in prange(M):
        aa[idx] = aa[idx] + bb[idx] * cc[idx]
    for i in range(a.shape[0]):
        a[i] = b[i] + c[i] * d[i]
s = 4096
x = np.random.rand(s*s); y = np.random.rand(s*s); z = np.random.rand(s*s)
v1 = np.random.rand(s); v2 = np.random.rand(s); v3 = np.random.rand(s); v4 = np.random.rand(s)
k(x, y, z, v1, v2, v3, v4, s*s)
t0 = time.perf_counter()
for _ in range(5):
    k(x, y, z, v1, v2, v3, v4, s*s)
dt = (time.perf_counter() - t0) / 5
print(f"[bench avx=%s] s={s} {dt*1e3:.2f} ms -> {32*s*s/dt/1e9:.1f} GB/s(4pass) threads={os.environ.get('NUMBA_NUM_THREADS','?')}", flush=True)
'''



def _probe_import():
    lines = []
    aff = len(os.sched_getaffinity(0))
    lines.append(f'[probe] cpu_count={os.cpu_count()} affinity={aff} '
                 f'numba_threads={numba.config.NUMBA_NUM_THREADS} '
                 f'OMP={os.environ.get("OMP_NUM_THREADS")} TBB={os.environ.get("TBB_NUM_THREADS")} '
                 f'AVXenv={os.environ.get("NUMBA_ENABLE_AVX")}')
    try:
        s = 4096
        x = np.random.rand(s * s); y = np.random.rand(s * s); z = np.random.rand(s * s)
        v1 = np.random.rand(s); v2 = np.random.rand(s); v3 = np.random.rand(s); v4 = np.random.rand(s)
        t0 = time.perf_counter()
        _kern(x, y, z, v1, v2, v3, v4, s * s)
        lines.append(f'[probe] warmup compile+run: {time.perf_counter() - t0:.2f}s')
        t0 = time.perf_counter()
        for _ in range(5):
            _kern(x, y, z, v1, v2, v3, v4, s * s)
        dt = (time.perf_counter() - t0) / 5
        lines.append(f'[probe mainproc avx={os.environ.get("NUMBA_ENABLE_AVX")}] '
                     f'{dt*1e3:.2f} ms -> {32*s*s/dt/1e9:.1f} GB/s(4pass)')
        del x, y, z
    except Exception as e:
        import traceback
        lines.append('[probe] mainproc bench failed: ' + repr(e))
    import tempfile
    bench = os.path.join(tempfile.gettempdir(), '_avxbench_tsvc.py')
    with open(bench, 'w') as fh:
        fh.write(BENCH_CODE)
    import subprocess
    for avx in ('0', '1', '2'):
        try:
            env = dict(os.environ)
            env['NUMBA_ENABLE_AVX'] = avx
            out = subprocess.run([sys.executable, bench, avx], env=env, capture_output=True,
                                 text=True, timeout=240)
            for ln in out.stdout.strip().splitlines():
                lines.append('[probe] ' + ln)
            if out.returncode != 0:
                lines.append(f'[probe] avx={avx} rc={out.returncode} '
                             f'stderr={out.stderr[-400:]!r}')
        except Exception as e:
            lines.append(f'[probe] subprocess avx={avx} failed: {e!r}')
    print('\n'.join(lines), flush=True)


_state = {'printed': False}


def s2275(a, b, c, d, aa, bb, cc, LEN_2D):
    n = int(LEN_2D)
    if n <= 0:
        return None
    _cc = (aa.flags.c_contiguous and bb.flags.c_contiguous and cc.flags.c_contiguous
            and a.flags.c_contiguous and b.flags.c_contiguous
            and c.flags.c_contiguous and d.flags.c_contiguous)
    if not _state['printed']:
        _state['printed'] = True
        t0 = time.perf_counter()
        if _cc:
            _kern(aa.ravel(), bb.ravel(), cc.ravel(), a, b, c, d, n * n)
        else:
            t = np.ascontiguousarray(aa)
            t += np.ascontiguousarray(bb) * np.ascontiguousarray(cc)
            aa[...] = t
            a[...] = np.asarray(b) + np.asarray(c) * np.asarray(d)
        dt = time.perf_counter() - t0
        print(f'[probe] call: n={n} dt={dt*1e3:.3f} ms -> {32*n*n/dt/1e9:.1f} GB/s(4pass) C={_cc}', flush=True)
        return None
    if _cc:
        _kern(aa.ravel(), bb.ravel(), cc.ravel(), a, b, c, d, n * n)
    else:
        t = np.ascontiguousarray(aa)
        t += np.ascontiguousarray(bb) * np.ascontiguousarray(cc)
        aa[...] = t
        a[...] = np.asarray(b) + np.asarray(c) * np.asarray(d)
    return None


_probe_import()
