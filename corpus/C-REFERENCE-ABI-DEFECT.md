# The C references never carried the ABI the judge binds

**Every C result produced before 2026-08-26 measures this defect, not the model.** Fortran results
are unaffected. Fixed in HPCAgent-Bench `cd9b3345` (405 files).

## What the agent was shown, and what it had to produce

The judge binds `void <native_base>_fp64(<params>)` and loads it from a standalone shared object.
The reference the agent was shown for 208 of 298 C kernels was verbatim TSVC:

```c
real_t s115(struct args_t *func_args) {
  initialise_arrays(__func__);
  gettimeofday(&func_args->t1, NULL);
  for (int j = 0; j < LEN_2D; j++)
    for (int i = j + 1; i < LEN_2D; i++)
      a[i] -= aa[j][i] * a[j];              /* globals */
  dummy(a, b, c, d, e, aa, bb, cc, 0.);
  return calc_checksum(__func__);
}
```

Wrong name, wrong signature, reads TSVC globals, calls TSVC helpers, times itself. An agent that
followed it emitted a library that could not load. Fortran, regenerated earlier, was already right:

```fortran
subroutine tsvc_2_s115_fp64(a, aa, LEN_2D) bind(C, name="tsvc_2_s115_fp64")
```

## Why it went unnoticed

`emit_io.py:46` treats a reference with no `hpcagent_bench-autogen` marker as a hand-written
OVERRIDE and never regenerates it. The TSVC adaptations carried no marker, so the generator skipped
exactly the broken files -- silently, and only for C/C++.

## The measurement it produced

oss120b, llr8 focus40, one agent per kernel:

| lane | kernels submitted | judge calls | dominant failure |
| --- | --- | --- | --- |
| Fortran | **35/40** | 166 (51 submit) | -- |
| C | **13/40** | 500 (14 submit) | 374 `incorrect` |

Of the 374 C `incorrect` verdicts: **208 were load-time `undefined symbol`** on TSVC globals
(`a`, `aa`, `d`, `LEN_1D`, `initialise_arrays`, `dummy`, `calc_checksum`) and 27 were
`exports no symbol`. 27 of the 40 focus40 C kernels had a broken reference -- the same 27 that
failed. The 13-vs-35 gap is the defect, not a capability difference.

The verdict was recorded as `incorrect` ("your code computed the wrong answer") when the code never
ran, and the agent's feedback was a 20-line cffi traceback. Agents retried the same kernels 15-16
times each (`s3110` x16, `s115` x15, `s119` x15) against a wall clock.

## Fix and verification

`regen_native_refs.py` regenerates through `numpyto_c` + `emit_bridge.bench_info_tempfile` and
installs the canonical header plus the autogen marker, so the file is regenerable from now on.
Excluded: `*_pluto_reference.c` (Pluto SCoP inputs, no manifest) and `tests/` fixtures. One emitter
refusal, `cegterg` (`NotImplementedError: __cb3.dtype`), outside focus40.

`verify_native_refs.py` grades each reference through `api.verify` -- the judge's own scoring path
against the numpy oracle. A bespoke driver was deliberately avoided: it could agree with the
emitter while both drift from what the judge binds.

| gate | result |
| --- | --- |
| focus40 C via `api.verify` | **40/40** correct, 0 wrong, 0 error |
| focus40 Fortran via `api.verify` | **40/40** correct, 0 wrong, 0 error |
| `test_abi_corpus_agreement` + param-order + never-folded | **11 passed** |
| `canonical <stem>_reference.<ext>` pre-commit hook | passed |

Both languages now derive the symbol from the same `naming.entry_symbol`, so C, C++, Fortran and
the binding agree by construction rather than by coincidence.

## Consequence for the paper

Discard every C arm before 2026-08-26 (llr8 `608446`, `608447`, `608987`, and the llr4/llr6 C arms
that ran against the same corpus). Fortran arms stand. Any C-vs-Fortran comparison drawn from them
is a comparison of two reference qualities.
