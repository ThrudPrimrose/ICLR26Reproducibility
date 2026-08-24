# Next qwen30b / oss120b run: the inference server is not the bottleneck

Measured from the vLLM throughput lines of 605458/605459 (qwen30b) and 605460/605461
(oss120b), focus40, C. Both arms: `INFERENCE_NODES=1`, `AGENTS_PER_NODE=120`, TP=4, no PP.

## Utilisation

| arm | agents | mean Running | inference util | mean Waiting | mean agg tok/s | peak agg |
|---|---|---|---|---|---|---|
| qwen30b | 120 | 38.5 | **32%** | 0.02 | 127 | 1101 |
| qwen30b +skills | 120 | 45.6 | 38% | 0.47 | 124 | 846 |
| oss120b | 120 | 12.9 | **11%** | 0.11 | 671 | 3906 |
| oss120b +skills | 120 | 15.8 | 13% | 1.22 | 572 | 4287 |

`Waiting` is ~0 in three of four arms: **nothing is queued at the engine**. Zero-generation
samples are 0% except the oss skills arm (3.8%). Aggregate throughput rises monotonically with
concurrency in every arm. There is no stall and no saturation -- the engine is STARVED.

The agents exist; they are simply not generating. 68-89% of agent-seconds are spent somewhere
other than the LLM.

## Therefore: do NOT reach for the kimi fixes

- **SGLang will not help.** The 4.3x SGLang win was measured on kimi at PP=4, against a vLLM
  scheduler that stalled above concurrency 1 with 42-43% zero-generation samples. These arms
  are single-node TP=4, show 0% zero-generation, ~0 queue, and monotonic scaling. There is no
  stall here to fix.
- **Do not add inference nodes.** One is already 3-9x oversized for the load it receives.
- **Do not raise `AGENTS_PER_NODE`.** 120 are already configured and only 13-46 are ever in
  flight.

## The compounding win: batching is free speed

Per-agent throughput RISES with concurrency, because the engine batches:

| qwen30b | Running=30 | Running=120 |
|---|---|---|
| aggregate | 69 tok/s | 846 tok/s |
| per agent | **2.3 tok/s** | **7.05 tok/s** |

| oss120b | Running=1 | Running=67 |
|---|---|---|
| aggregate | 181 tok/s | 2767 tok/s |
| per agent | 181 tok/s | 41 tok/s |

qwen30b currently runs at ~2.3 tok/s per agent -- below the >=8 tok/s per agent the sizing
rule asks for -- purely because only a third of its agents are generating at once. Unblocking
agents makes every agent ~3x faster at no hardware cost. oss120b sits at concurrency 1 for the
plurality of its samples while the engine can sustain 21x that aggregate.

## So the lever is the judge, and it has three compounding parts

### 1. Run 4 judge ranks per node, not 1

A node is 4 sockets x 24 cores = 96 cores (192 threads). `GRADE_CPUS` resolves to
`detect_cores_per_socket()` = 24, and `role_srun` launches the judge with
`--ntasks-per-node=1 --cpus-per-task=24`. **One rank per node uses a quarter of the node.**
Confirmed against the run dirs: 12 judge DBs for `JUDGE_NODES=12`, 16 for 16 -- one rank each.

Four ranks per node, one per socket, is ~4x judge throughput on the same allocation. Socket
pinning keeps each timed child alone on its own 24 cores, so timing fidelity is preserved --
which is exactly why the child must stay one-at-a-time WITHIN a rank.

### 2. Keep timed runs serial; parallelise only what is not timed

The timed child takes every core the judge owns (`OMP_NUM_THREADS=GRADE_CPUS`). Two timed
children on one socket would contaminate each other's measurement, so a timed run must be
exclusive. That is a correctness constraint on the numbers, not just a memory one.

Untimed work has no such constraint: input generation, the numpy/C reference, the determinism
comparison and the fresh-seed correctness check can overlap, bounded by memory. Split the
grade into "timed, exclusive" and "untimed, parallel" rather than serialising the whole thing.

### 3. Sequence the verify legs so 4 ranks per node fits in memory

`scoring.independent_verify` holds eight full array sets at once (see
`SCORE_SUBMIT_DESIGN.md`). At the focus40 XL footprint of 3.88 GiB that is ~31 GiB per grade.

| judge ranks/node | today (8 sets) | sequenced (4 sets) |
|---|---|---|
| 1 | 31 GiB | 15 GiB |
| 4 | 124 GiB | **62 GiB** |

Sequencing is what makes the 4x rank increase safe on a 500 GiB node. Do it first.

Note the tension to respect: parallelising hidden variants within a rank multiplies the peak
by the number in flight. Parallelise them only under a byte budget, and never the timed leg.

## Judge sizing: how many judge nodes does an N-agent run need?

Measured demand and supply, focus40:

| arm | ranks | grades/h/rank | grades/h/agent | median arrival gap | mean gap |
|---|---|---|---|---|---|
| qwen30b | 12 | 28.8 | 2.88 | 35.5 s | 125 s |
| qwen30b +skills | 12 | 28.6 | 2.86 | 43.1 s | 126 s |
| oss120b | 16 | 41.2 | 5.49 | 25.4 s | 87 s |
| oss120b +skills | 16 | 28.7 | 3.82 | 37.8 s | 125 s |

An agent demands **3-5.5 grades/hour**. A rank currently delivers 28-41/hour.

Median arrival gap is 3-4x below the mean, so arrivals are BURSTY: a rank is idle much of the
time and then takes a cluster. Combined with inference util of 11-38%, neither tier is
throughput-saturated -- the system is LATENCY-bound. Each agent iterates
write -> grade -> read -> write, and it is the round-trip that limits it, not any tier's
capacity.

That has a direct consequence for sizing: **do not size the judge on mean throughput.** At
30% mean utilisation with bursty arrivals, queueing delay is what agents feel, and a fleet
sized to the mean makes per-iteration latency worse -- which lowers inference utilisation
further, because agents spend even longer not generating.

Sizing rule, with headroom for burst:

    judge_ranks  >=  agents x grades_per_agent_per_hour / rank_capacity_per_hour
                 ~=  agents x 5 / 100          (a saturated rank is ~100/h, not the 28-41 seen)
    judge_nodes  =  ceil(judge_ranks / 4)      once --ntasks-per-node=4 lands

| agents | grades/h demanded | ranks (mean) | ranks (2x burst headroom) | NODES at 4 ranks/node |
|---|---|---|---|---|
| 40 | 200 | 2 | 4 | **1** |
| 120 | 600 | 6 | 12 | **3** |

So a 40-agent run fits comfortably in **1 judge node** once a node hosts 4 ranks -- 2 only if
the burst headroom proves insufficient in practice. Today's 12-16 NODES for 120 agents becomes
3-4. That is a 4x reduction in judge allocation for the same service, or the same allocation
carrying 4x the agents.

Caveat on `rank_capacity_per_hour`: 100/h is inferred, not measured at saturation. These arms
never saturated a rank, so the true ceiling is unknown. Measure it directly before trusting
the table -- run one rank against a deliberately oversubscribed agent count and find where
queueing delay takes off.

### Rank imbalance is free capacity

Calls per rank are uneven: busiest/idlest is 1.6x, 1.6x, 1.9x and 2.5x across the four arms.
The busiest rank does roughly twice the work of the idlest, so part of the fleet is idle while
another part queues. Balancing the routing recovers that without adding a node.

### The launcher change

`role_srun` currently starts the judge with `--ntasks-per-node=1 --cpus-per-task=24`, using one
socket of four. The fix is:

    --ntasks-per-node=4 --cpus-per-task=24

One rank per socket, 96 of 96 cores used, each timed child alone on its own 24 cores so timing
fidelity is unchanged. Sequence the verify legs FIRST (`SCORE_SUBMIT_DESIGN.md`): four ranks at
today's 8-set peak is 124 GiB/node, and at the sequenced 4-set peak is 62 GiB.

## Order of work for the next submission

1. Sequence the verify legs and defer `redata` / `np_re` behind real builders. Halves grade peak.
2. Raise judge ranks to 4 per node with socket pinning. ~4x judge throughput.
3. Fix the `wf_diff_skew` / `wf_triangular` XL OOM (85 lost submissions) -- and repair
   `wf_north_west`'s non-monotonic ladder through `apply_sizes.py`, not by hand.
4. Re-measure utilisation. If mean Running rises toward 120, per-agent throughput should rise
   with it (2.3 -> ~7 tok/s for qwen30b) with no inference change at all.

Only after that does an inference-side change deserve attention, and the measurement to take
then is mean Running, not tokens/s.
