import os, sys, time
import numpy as np

def s119(aa, bb, LEN_2D):
    try:
        aff = os.sched_getaffinity(0)
        cpu_max = open('/sys/fs/cgroup/cpu.max').read().strip()
    except Exception as e:
        aff, cpu_max = ('err', str(e)), 'n/a'
    gpu = 'n/a'
    try:
        import torch
        gpu = '%s avail=%s' % (torch.cuda.get_device_name(0), torch.cuda.is_available())
        free, total = torch.cuda.mem_get_info()
        gpu += ' freemem=%.1fG/%.1fG' % (free/1e9, total/1e9)
    except Exception as e:
        gpu = 'torch err: %s' % e
    print('N=%d ncpu=%s affinity=%d cgroup=%s' % (LEN_2D, os.cpu_count(), len(aff), cpu_max))
    print('GPU: %s' % gpu)
    sys.stdout.flush()
    t0 = time.perf_counter()
    for i in range(1, LEN_2D):
        np.add(aa[i - 1, :-1], bb[i, 1:], out=aa[i, 1:])
    dt = time.perf_counter() - t0
    print('numpy rowloop: %.1f ms' % (dt * 1e3))
    sys.stdout.flush()
    return None
