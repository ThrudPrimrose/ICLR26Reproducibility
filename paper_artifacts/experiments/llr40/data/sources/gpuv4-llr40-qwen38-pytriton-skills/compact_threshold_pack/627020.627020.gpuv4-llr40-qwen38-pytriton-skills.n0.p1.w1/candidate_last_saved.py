import os, sys
import numpy as np

def compact_threshold_pack(src, weight, packed, out_count, LEN_1D):
    print("PROBE affinity n=", len(os.sched_getaffinity(0)), sorted(os.sched_getaffinity(0))[:40])
    print("PROBE cpu_count=", os.cpu_count())
    try:
        nodes = set()
        for cpu in os.sched_getaffinity(0):
            try:
                with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id") as f:
                    nodes.add(int(f.read()))
                with open(f"/sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list") as f:
                    print("PROBE L3 cpu", cpu, f.read().strip()[:60])
                    break
            except Exception as e:
                print("PROBE err", cpu, e)
    except Exception as e:
        print("PROBE err", e)
    sys.stdout.flush()
    idx = np.flatnonzero(src > 0)
    n = idx.size
    if n:
        np.multiply(src[idx], weight[idx], out=packed[:n])
    out_count[0] = n
