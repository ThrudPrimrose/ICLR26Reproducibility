"""Emit the NumpyToX reference for one kernel in each language and print the loop nest.

The question is narrow: does the generated Fortran index 2D arrays in the reversed order the
ABI contract promises, or does it transliterate the numpy order the way the agents did?
"""
import sys

sys.path.insert(0, "/capstor/scratch/cscs/ybudanaz/x86_64/optarena")
sys.path.insert(0, "/capstor/scratch/cscs/ybudanaz/x86_64/optarena/hpcagent_bench/numpy_translators/src")

from hpcagent_bench.harness.agent import emit_reference_source

for kernel in sys.argv[1:]:
    for lang in ("c", "fortran"):
        print(f"##### {kernel} / {lang}")
        try:
            src = emit_reference_source(kernel, lang)
        except Exception as exc:  # noqa: BLE001 - probe reports whatever the emitter raises
            print(f"  EMIT FAILED: {type(exc).__name__}: {exc}")
            continue
        for line in src.splitlines():
            low = line.lower()
            if ("aa" in low or "bb" in low or "do " in low or "for (" in low
                    or "intent" in low or "omp" in low or "restrict" in low):
                print("   ", line.rstrip())
        print()
