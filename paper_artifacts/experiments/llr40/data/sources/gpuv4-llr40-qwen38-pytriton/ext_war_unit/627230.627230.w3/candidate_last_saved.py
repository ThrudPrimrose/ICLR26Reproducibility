import numpy as np
import ctypes

libc = ctypes.CDLL("libc.so.6", use_errno=True)
_syscall = libc.syscall
_syscall.restype = ctypes.c_long
SYS_MOVE_PAGES = 279
SYS_MIGRATE_PAGES = 256

def _nodes_of(ptr, size, stride=16384):
    n = size // stride
    addrs = (ctypes.c_void_p * n)(*[ptr + i * stride for i in range(n)])
    stat = (ctypes.c_int * n)()
    r = _syscall(SYS_MOVE_PAGES, 0, ctypes.c_size_t(n), ctypes.byref(addrs), 0,
                 ctypes.byref(stat), 0, 0)
    if r < 0:
        return "move_pages errno=%d" % ctypes.get_errno()
    cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for v in stat:
        if v == -1:
            cnt[8] += 1
        else:
            cnt[v] += 1
    return " ".join("n%d=%d" % (i, cnt[i]) for i in range(9))

def ext_war_unit(a, b, LEN_1D):
    import sys
    n = a.shape[0]; sz = n * a.itemsize
    ap = a.__array_interface__['data'][0]
    bp = b.__array_interface__['data'][0]
    sys.stdout.write("a: %s\nb: %s\n" % (_nodes_of(ap, sz), _nodes_of(bp, sz)))
    sys.stdout.flush()
    a[:-1] = a[1:] + b[:-1]
