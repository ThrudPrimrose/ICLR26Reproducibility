import os, sys
import numpy as np

out = []
out.append("CPUS=%s cpu_count=%s sched_getaffinity=%s" % (os.cpu_count(), os.cpu_count(), sorted(os.sched_getaffinity(0))[:80]))
try:
    out.append("NODES=%s" % sorted(os.listdir("/sys/devices/system/node"))[:10])
except Exception as e:
    out.append("NODES=ERR %s" % e)
try:
    flags = ""
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            out.append(line.strip()); break
    for line in open("/proc/cpuinfo"):
        if line.startswith("flags"):
            flags = line.split(":",1)[1]
            break
    out.append("HAS_avx512f=%s HAS_avx2=%s" % ("avx512f" in flags, "avx2" in flags))
except Exception as e:
    out.append("CPUINFO=ERR %s" % e)
out.append("GCC=%s" % os.popen("which gcc && gcc --version | head -1").read().strip()[:120])
out.append("MEM=%s" % os.popen("grep -E 'MemTotal|MemAvailable' /proc/meminfo").read().strip())
out.append("OMP_ENV=%s" % {k:v for k,v in os.environ.items() if 'OMP' in k or 'OPENBLAS' in k or 'NUMA' in k})
out.append("PY=%s" % sys.version.split()[0])
out.append("NPMEM=%s" % getattr(np, "__cpu_features__", None) if False else "np_ok")
try:
    out.append("CPFEAT=%s" % str(np.__cpu_features__.enabled))
except Exception as e:
    out.append("CPFEAT=ERR %s" % e)

sys.stdout.write("\n".join(out) + "\n")
sys.stdout.flush()

def vtvtv(a, b, c, LEN_1D):
    n = LEN_1D
    a = a[:n]
    np.multiply(a, b[:n], out=a)
    np.multiply(a, c[:n], out=a)
