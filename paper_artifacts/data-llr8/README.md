# data-llr8 -- PARTIAL

Collected 2026-08-25 18:40 CEST from `$SCRATCH/hpcagent-bench-runs`, while four of the five
arms were still running. These CSVs are a checkpoint, not a result: re-run

    python3 collect.py --run-root $SCRATCH/hpcagent-bench-runs --campaign llr8 --out data-llr8

once every arm has landed, and the numbers below will move.

| job | arm | state at collection |
|---|---|---|
| 608447 | llr8-oss120b-c | COMPLETED (01:05:44) |
| 608446 | llr8-qwen30b-c | RUNNING |
| 608448 | llr8-kimi27sglang-c-a | RUNNING |
| 608449 | llr8-kimi27sglang-c-b | RUNNING |
| 608987 | llr8-oss120b-c-skills | RUNNING |
| 608988 | llr8-qwen30b-c-skills | PENDING (chained on 608446) -- no shards, skipped |

A kimi arm draws a 10-kernel BATCH of the focus40 tag, so its success denominator is 10 and not
40; `collect.py` carries that per arm. Reading a batch against the whole tag turns 10 solved of
10 drawn into 10 of 40, which is a 75% failure that never happened.
