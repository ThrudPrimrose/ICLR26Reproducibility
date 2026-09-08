import os, time, subprocess, sys, json
import numpy as np

CHILD = r'''
import os, time, ctypes
import numpy as np
_P = ctypes.POINTER(ctypes.c_double)
lib = ctypes.CDLL("/shared/agent-36/libvtvtv_kern.so")
f = lib.vtvtv_kern
f.argtypes = [_P, _P, _P, _P, ctypes.c_longlong, ctypes.c_int]
nt = int(os.environ["CFG_NT"])
try:
    os.sched_setaffinity(0, {int(x) for x in os.environ.get("CFG_AFF", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23").split(",")})
except Exception as e:
    print("afffail", e); 
n = 48_000_000
a = np.random.rand(n); b = np.random.rand(n); c = np.random.rand(n)
pa = a.ctypes.data_as(_P); pb = b.ctypes.data_as(_P); pc = c.ctypes.data_as(_P)
f(pa, pa, pb, pc, n, nt)  # warm
ts = []
for _ in range(3):
    t0 = time.perf_counter(); f(pa, pa, pb, pc, n, nt); ts.append(time.perf_counter() - t0)
print("CFG nt=%s best=%.2fms %dGB/s" % (os.environ["CFG_NT"], min(ts)*1e3, int(32*n/min(ts)/1e9)))
'''

def run_cfg(nt, bind=None, aff=None, reps=3):
    env = dict(os.environ)
    env["CFG_NT"] = str(nt)
    if bind:
        env["OMP_PROC_BIND"] = bind
        env["OMP_PLACES"] = "cores"
    if aff is not None:
        env["CFG_AFF"] = ",".join(str(c) for c in aff)
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True,
                       text=True, timeout=300, env=env)
    return p.stdout.strip() + ("  [ERR " + p.stderr.strip()[:200] + "]" if p.returncode != 0 else "")


def vtvtv(a, b, c, LEN_1D):
    a *= b
    a *= c
    lines = []
    aff24 = list(range(24))
    aff48 = aff24 + list(range(96, 120))
    cfgs = [
        (24, None, aff24), (24, "close", aff24), (24, "spread", aff24),
        (24, "close", aff48), (24, "spread", aff48),
        (48, None, aff48), (48, "close", aff48), (48, "spread", aff48),
        (32, None, aff48), (16, None, aff24),
    ]
    for nt, bind, aff in cfgs:
        for _ in range(2):
            lines.append(run_cfg(nt, bind, aff))
    print("\n".join(lines), flush=True)
    return None
