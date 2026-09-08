import sys
def _f(a, b, ip, LEN_1D):
    print("SIZES: len(a)=%d len(b)=%d len(ip)=%d LEN_1D=%r (%s)" % (len(a), len(b), len(ip), LEN_1D, type(LEN_1D).__name__))
    print("DTYPES: a=%s b=%s ip=%s" % (a.dtype, b.dtype, ip.dtype))
    print("STRIDES: %s %s %s" % (a.strides, b.strides, ip.strides))
    print("CONTIG: %s %s %s" % (a.flags.c_contiguous, b.flags.c_contiguous, ip.flags.c_contiguous))
    sys.stdout.flush()
    for i in range(LEN_1D):
        a[i] = b[ip[i]]
def tsvc_2_vag_fp64(a, b, ip, LEN_1D): _f(a, b, ip, LEN_1D)
def tsvc_2_vag(a, b, ip, LEN_1D): _f(a, b, ip, LEN_1D)
def vag(a, b, ip, LEN_1D): _f(a, b, ip, LEN_1D)
