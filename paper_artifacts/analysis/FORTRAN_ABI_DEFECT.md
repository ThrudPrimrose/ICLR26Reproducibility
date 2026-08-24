# Fortran's failure rate is one ABI defect, not model capability

Collected 2026-08-24 from jobs 605696 / 605697 (qwen30b, llr-focus40, +/- skills)
against 605458 / 605460 (same kernels, C).

## The number that started this

| arm | not-ok calls | kernels reached | kernels with ZERO ok call |
|---|---|---|---|
| qwen30b / Fortran, no skills | 150 / 358 = **41.9%** | 26 / 40 | **14** |
| qwen30b / Fortran, skills | 162 / 407 = **39.8%** | 28 / 40 | **11** |
| qwen30b / C, no skills | 157 / 1355 = 11.6% | 38 / 40 | **0** |
| oss120b / C, no skills | 86 / 737 = 11.7% | 38 / 40 | **0** |

This is not an evenly-raised error rate. It is a set of kernels that fail
**100% of the time, every attempt, in both arms**, while passing reliably in C.

## The 11 kernels dead in BOTH Fortran arms

All 11 pass in both C arms, most of them near-always.

| kernel | Fortran ok/n (2 arms) | C qwen ok/n | C oss ok/n | judge detail |
|---|---|---|---|---|
| argmax_with_index | 0/9, 0/9 | 61/62 | 7/8 | out_index: integer mismatch |
| fuse_move_ifs | 0/8, 0/8 | 39/40 | 11/11 | a: numeric mismatch |
| tsvc_2_s115 | 0/8, 0/7 | 34/45 | 16/19 | a: NaN position mismatch |
| tsvc_2_s1232 | 0/9, 0/9 | 27/30 | 8/9 | aa: numeric mismatch |
| tsvc_2_s2233 | 0/8, 0/13 | 26/33 | 14/15 | aa: numeric mismatch |
| tsvc_2_s231 | 0/10, 0/7 | 28/32 | 6/6 | aa: numeric mismatch |
| tsvc_2_s233 | 0/5, 0/14 | 20/31 | 9/9 | aa: numeric mismatch |
| tsvc_2_s235 | 0/6, 0/10 | 13/27 | 11/11 | aa: numeric mismatch |
| tsvc_2_s275 | 0/6, 0/8 | 18/21 | 11/12 | aa: numeric mismatch |
| tsvc_2_s3110 | 0/11, 0/7 | 28/29 | 16/16 | bb: numeric mismatch |
| wf_triangular | 0/9, 0/7 | 19/36 | 12/27 | a: Inf position mismatch |

**10 of 11 touch a 2D array.** The four that do not name `aa`/`bb` in the detail
still read or write one: `s115` reads `aa[j,i]` into 1D `a`; `wf_triangular`'s `a`
IS 2D; `fuse_move_ifs` has `a`, `b`, `src` all 2D. Only `argmax_with_index` is a
different bug (0-based vs 1-based index in `out_index`).

## Root cause

The task spec hands the agent two artifacts with OPPOSITE memory conventions and
no explicit mapping between them:

    reference_numpy:  aa[j, i] = aa[j-1, i] + bb[j, i]       # numpy: row-major
    signature:        real(c_double) :: aa(LEN_2D, LEN_2D)   # Fortran: column-major

Same `bind(C)` buffer. numpy's `aa[j,i]` is linear offset `j*LEN_2D + i`; Fortran's
`aa(p,q)` is `(q-1)*LEN_2D + (p-1)`. To touch the same element the Fortran code must
write `aa(i+1, j+1)` -- indices REVERSED.

`abi_contract.md` Sec. 7 documents the mechanism:

> Arrays line up without copies because the emitter declares them with reversed
> extents, e.g. `A(NK, NI)`, so Fortran column-major access matches the row-major
> C buffer.

That signal is carried ENTIRELY by the extent order. For a square array --
`aa(LEN_2D, LEN_2D)` -- it carries no information at all. Nothing in the spec the
agent sees states the reversal.

What every agent wrote, vs what the emitter generates for the same kernel:

    aa(j, i)     = aa(j-1, i)      + bb(j, i)        ! agent: transliterated, TRANSPOSED
    aa((i)+1, (j)+1) = aa((i)+1, ((j-1))+1) + bb((i)+1, (j)+1)   ! emitter: CORRECT

The agent computes the transpose. Because `aa` is square the shapes match, so
nothing crashes -- it just returns wrong numbers forever. The judge reports a bare
`aa: numeric mismatch` that never mentions layout, so the repair loop has nothing
to converge on and burns every round.

## The skills packet already states the rule, and it did not help

`hpcagent_bench/skills/lang-fortran/SKILL.md` says, verbatim:

> **Column-major: first index fastest, so it belongs innermost.** An element the
> reference writes as `a[i][j]` is `a(j + 1, i + 1)` here -- subscripts reversed,
> starting at 1.

Verified delivered: the 605697 prompt is 36,733 bytes and contains
`subscripts reversed, starting at 1`; the 605696 prompt is 11,609 bytes and does not.
Both arms then failed the SAME 11 kernels, 100% of attempts.

So the correct rule, in the correct words, delivered verbatim in the prompt, changed
nothing. Two plausible reasons, not separated by this data:

1. It is filed under "Writing fast Fortran" -- framed as a PERFORMANCE note ("belongs
   innermost"), not a correctness gate that invalidates the answer.
2. It competes with ~25k bytes of other packet text at the moment of transliteration.

The implication is that more skill text is not the fix. A correct baseline in the
agent's hands is mechanical and unmissable; a rule it must remember to apply is not.

## The correct baseline exists and is thrown away

`harness/agent.py:emit_reference_source(kernel, language)` runs NumpyToX and returns
a correct per-language reference -- verified correct for `tsvc_2_s231` fortran above.
`StubAgent` submits exactly this source, so it is already the CI baseline.

`materialize_shared.sh` does not ship it. Its own comment:

> Per-language task sources are emitted on demand into a temp dir
> (harness/agent.py:emit_reference_source) and exist nowhere in the repo, so a
> kernel's copyable material is its numpy reference plus any vendored baseline.

The `<kernel>_reference.c` that IS shipped is the vendored TSVC original, labelled in
its own header "Not the scoring oracle". TSVC is a C suite, so there is no Fortran
counterpart -- the Fortran track ships no baseline in its own language at all.

## Survivorship: the defect flatters the arm it breaks

`submissions` only ever contains correctness-passing rows (verified: kernels with a
submission == kernels with >=1 ok call, 26 == 26). Best-per-kernel is therefore
already "best passing submission". It cannot rescue a kernel with zero passes -- it
drops it from the denominator instead.

| arm | median over SUBMITTED kernels | median counting the 14/11 dead as 1.0x |
|---|---|---|
| 605696 Fortran no skills | 1.099 | **1.000** |
| 605697 Fortran skills | 1.496 | **1.000** |

Every Fortran speedup figure published so far is computed over the survivors.

## Recommendations, in order of expected effect

1. **Ship the emitted per-language reference into `shared/tasks/`** (C and Fortran).
   Mechanical, needs no agent behaviour change, and directly shows the index order.
   Gate it: every shipped baseline must pass the stub-agent correctness run first.
2. **Report a dead kernel as 1.0x, not as absent.** Coverage and speedup should not
   be silently conditioned on success.
3. **Make the judge's mismatch detail name the suspicion.** A first-failure hint of
   "output matches the transpose of the reference" converts an unrecoverable loop
   into a one-round fix. Cheap to test: compare against the transposed reference on
   mismatch and say so.
4. **Sharpen the skill page** -- move layout from a perf note to a correctness gate,
   and add inclusive `do` bounds. Do this, but do not expect it to carry the fix
   alone; the evidence above is that it already did not.
