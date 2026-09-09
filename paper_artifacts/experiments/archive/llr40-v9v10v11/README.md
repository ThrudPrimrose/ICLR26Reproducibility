# Archived: llr40 campaigns v9, v10, v11 (and the v11w* waves)

Frozen 2026-09-09. These campaigns are CLOSED and must not be pooled with anything that
comes after them. They are kept because the numbers are real and reproducible from the CSVs
here -- not because the treatment they measured still exists.

## Why they are closed

The skills treatment changed shape, not just content, so a later arm is not a later sample of
the same experiment:

1. **The packet was inlined into every task text.** ~18.4k chars (~4.6k tokens) of skill pages
   were pasted into the prompt and re-sent on every turn. From 2026-09-09 the pages are files on
   disk and the prompt carries only a ~1.7k-char trigger block. The treatment is now "may open a
   page", where it used to be "cannot avoid the page".

2. **The packet sat between the "Task:" header and the task.** Measured on the v11-era prompt:
   292 lines separated the label from the assignment it labels, and the last thing an agent read
   before acting was the tail of the openmp page. The trigger block is now last.

3. **No arm ever carried the closing skill reminder.** `skill_reminder()` landed in
   agent_driver.py at 2026-09-09 10:03; every prompt in these campaigns was written before it.

4. **The Fortran pages shipped worked code for graded kernels.** `openmp-fortran` carried a
   complete two-pass argmax-with-index and a complete chunked prefix sum, which are the answers
   to `argmax_with_index` and `scan_affine_decay`. Cut on 2026-09-09; `maxloc` now leads (it is a
   language intrinsic and returns the 1-based index the judge wants) and the scan pragma is kept
   as a spelling.

## What the archived data says about skills

Null to negative under every unit computed from `llr40_observations.csv`:

| unit | C | Fortran |
|---|---|---|
| per-kernel, max across agents (the campaign scoring rule) | 0.940 | 1.029 |
| paired (model, language, kernel), n=227 | -12.3% | -8.2%, sign p=1.0 |

Win/loss is 15/24 against skills in both languages. On the four kernels whose answers appeared
as code in the pages, skills scored 0.61x (C) and 0.65x (Fortran) against 0.98/1.01 on every
other kernel -- agents pasted the recipe instead of reading their own nest.

An earlier note recorded Fortran +11.8% at sign p=0.018. That does not reproduce from the CSV
here under either unit above. Treat it as unverified rather than as a finding.

## Token accounting

`tokens` in these CSVs is the BILLED per-turn sum, not `effective`. The run trees were deleted
during a cleanup on 2026-09-09 and `effective` can only be computed from `claude.log`, so it is
UNRECOVERABLE for these campaigns. Campaigns from 2026-09-09 on carry the full breakdown inline
in each `tokens.json`. Do not compare a billed number here against an effective number later.
