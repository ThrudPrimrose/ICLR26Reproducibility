# llr6v10 skills ablation: null across every model and language

Collected 2026-08-24 from the judge databases. Skills revision v10, `llr-focus40` (40 kernels).
Interactive version: `figures/skills_ablation.html`.

## Arms

| job | model | language | skills | state |
|---|---|---|---|---|
| 605458 | qwen30b | C | no | COMPLETED |
| 605459 | qwen30b | C | yes | COMPLETED |
| 605460 | oss120b | C | no | COMPLETED |
| 605461 | oss120b | C | yes | COMPLETED |
| 605696 | qwen30b | Fortran | no | COMPLETED |
| 605697 | qwen30b | Fortran | yes | COMPLETED |
| 605700 | Kimi-K2.7 (SGLang) | C | no | partial, context only |

## Result: no effect, anywhere

Paired per-kernel on the best passing submission. Exact two-sided sign test, ties excluded.

| ablation | paired n | skills better | worse | tied | p |
|---|---|---|---|---|---|
| qwen30b / C | 38 | 12 | 12 | 14 | **1.000** |
| oss120b / C | 38 | 16 | 14 | 8 | **0.856** |
| qwen30b / Fortran | 26 | 6 | 9 | 11 | **0.607** |
| **pooled** | **102** | **34** | **35** | **33** | **1.000** |

Consistent with ablation-2 and the llr6v8 C result. The packet is not a treatment that moves
speedup on this benchmark.

## Paired, never aggregate

The two legs of an ablation do not cover the same kernel set, so an unpaired median compares
different populations. Fortran is the worked example:

| statistic | no skills | skills | reads as |
|---|---|---|---|
| unpaired median over each leg's own kernels | 1.099 | 1.496 | skills win |
| paired on the 26 kernels both legs reached | 6 better | 9 worse | skills behind |

Only the paired figure is reportable.

## Per-arm quality

| arm | kernels /40 | submissions | ok % | incorrect % | median speedup |
|---|---|---|---|---|---|
| kimi27-sglang / C | 22 (partial) | 211 | 94.3 | 1.8 | 6.349 |
| oss120b / C | 38 | 150 | 88.3 | 4.7 | 5.263 |
| oss120b / C +skills | 38 | 127 | 85.7 | 7.7 | 5.719 |
| qwen30b / C | 38 | 366 | 88.4 | 8.0 | 2.502 |
| qwen30b / C +skills | 38 | 349 | 85.4 | 11.3 | 3.018 |
| qwen30b / Fortran | 26 | 74 | 58.1 | 37.4 | 1.099 |
| qwen30b / Fortran +skills | 28 | 80 | 60.2 | 36.4 | 1.496 |

Two effects dominate the skills variable:

- **Language.** Fortran loses 37% of judged calls to wrong answers against 2-11% for C, and
  reaches 26-28 of 40 kernels against 38. Root cause is a single ABI defect, not model
  capability -- see `FORTRAN_ABI_DEFECT.md`.
- **Model.** oss120b submits 2.4x less often than qwen30b on C and lands at twice the median
  speedup. Attempt volume is not yield.

## Measurement caveats that govern these numbers

1. **Speedup is quantized.** Every value lies on a reciprocal grid of two-decimal numbers
   (33.333, 25.0, 20.0, 16.667, 11.111). Above ~5x one grid step is a 25-33% jump with no
   intermediate value. Share of each arm's kernels in that region: qwen30b/C 36.8%,
   +skills 44.7%, oss120b/C 55.3% both legs, Fortran 30.8% / 39.3%, kimi 59.1%.
   Every **mean** speedup is therefore unusable; medians for oss120b and kimi fall inside the
   coarse region. The paired tests survive because they are dominated by kernels near parity,
   where the grid is fine.
2. **`speedup` disagrees with the stored timings.** Raw `baseline_ns / native_ns` of 35.27,
   31.24 and 22.92 are recorded as 33.333, 25.0 and 20.0. The two columns do not come from the
   same measurement; which is authoritative is unresolved.
3. **Dead kernels leave the denominator.** `submissions` holds only correctness-passing rows
   (verified: kernels with a submission == kernels with >=1 ok call). A kernel that never passes
   is absent, not zero. Counting the Fortran dead kernels as 1.0x moves both medians to 1.000.
4. **kimi27 is not an ablation arm.** Partial (22/40 at collection) and served through SGLang
   rather than vLLM, so it changes model and serving stack at once. Context only.

## Reproduce

```bash
python3.11 extract_arms.py     # judge DBs  -> arms.json      (sqlite3 only, no numpy)
python3.11 analyze_arms.py     # arms.json  -> the tables above
python3.11 build_payload.py    # arms.json  -> payload.json   (feeds the HTML figure)
```
