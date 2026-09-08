import sys

def s318(a, result, inc, LEN_1D):
    try:
        import numpy as np
        print("PROBE a.shape", a.shape, "a.dtype", a.dtype, "len(a)", len(a),
              "inc", inc, "LEN_1D", LEN_1D,
              "need", (LEN_1D-1)*inc+1,
              "a.flags", a.flags['C_CONTIGUOUS'])
        import numpy as np
        s = a[0:(LEN_1D-1)*inc+1:inc]
        v = np.abs(s)
        idx = int(v.argmax())
        result[0] = float(v[idx]) + float(idx)
        print("PROBE answer", result[0])
        sys.stdout.flush()
    except Exception as e:
        import traceback; traceback.print_exc(); sys.stdout.flush()
        raise
    return None
