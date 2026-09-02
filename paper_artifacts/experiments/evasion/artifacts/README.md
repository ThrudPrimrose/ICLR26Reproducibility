# evasion artifacts

The anti-cheat sweep: every submitted source in the judge databases, read for the signatures of an
answer that games the harness rather than optimising the loop. It audits the other experiments, so
it has no arms of its own.

`../data/evasion_candidates.csv` is the candidate table -- one row per flagged submission, with the
signature that flagged it. It is a CANDIDATE list: the detectors are syntactic and deliberately
over-trigger, and `../../../agent_evasion.md` and `../../../anti_cheat.md` are where the adjudicated
cases and the harness changes that followed them are written up.

```bash
python3 ../collect_evasion.py --run-root <RUN_ROOT> --out ../data/evasion_candidates.csv
```

The sweep prints scanned, parsed and skipped counts. A zero must be distinguishable from a crash.
