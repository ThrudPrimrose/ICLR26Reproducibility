# Serving Kimi K2.7 on MI300A -- what 152 smoke jobs settled

Everything below was measured on beverin (MI300A, gfx942, 4 dies/node, 192 cores, 513 GB).
Job range 594564-604942. The smoke tree itself is deleted; this is what it established.

## The headline

**SGLang, not vLLM, and one agent per replica rather than many.** vLLM never got Kimi past
~7 tok/s aggregate under campaign concurrency. SGLang reached 39 tok/s at concurrency 4 --
2.1x vLLM's best -- and stopped producing zero-token responses.

## Measured throughput (SGLang, TP=4, the surviving config)

Short prompts, no context load:

| concurrency | aggregate tok/s | per-agent tok/s |
| --- | --- | --- |
| 1 | 18.48 | 18.48 |
| 2 | 35.51 | 17.76 |
| 4 | **39.26** | 9.81 |
| 6 | 28.14 | 4.69 |

Aggregate PEAKS AT 4 and then falls. Concurrency 6 is worse than 2.

Under realistic context, the peak moves but the shape holds:

| ctx tokens | conc | aggregate tok/s | per-agent tok/s |
| --- | --- | --- | --- |
| 10,000 | 1 | 13.74 | 13.74 |
| 10,000 | 4 | 38.09 | 9.52 |
| 10,000 | 6 | 46.75 | 7.79 |
| 12,508 | 8 | 62.13 | 7.77 |
| 12,508 | 12 | **79.15** | 6.60 |
| 12,508 | 16 | 66.60 | 4.16 |
| 40,096 | 1 | 9.01 | 9.01 |
| 40,096 | 4 | 18.62 | 4.66 |
| 40,096 | 6 | 20.37 | 3.39 |

Two things fall out. Aggregate tops out near **concurrency 12** and regresses by 16. And
context is the dominant cost: at 40k context the engine delivers less than half its 10k rate.
An agentic campaign runs 85:1 prompt:generation, so this curve -- not the short-prompt one --
is the one that governs.

**Sizing rule that follows:** >= 8 tok/s per agent is the floor for a campaign to finish.
That puts a replica at ~12 agents, and scaling is achieved by MORE REPLICAS (more 6-node
jobs), never by raising agents per replica.

## Failures that cost whole runs, and what each actually was

| Symptom | Real cause | Fix |
| --- | --- | --- |
| 1 step / 147 s, degrading with load | inference step got **2 CPUs of 192** -- `--cpus-per-task` was passed to the judge role only | give every role the node; `--ntasks-per-node=1` + full CPUs |
| Grades unreproducible, races "passed" | judge graded on **2 CPUs of 192** | `GRADE_CPUS=24`, one judge per socket |
| Dies ~1 min after the FIRST request | async-scheduler broadcast on `pp.device_group`: 4-rank vs 2-rank **NCCL bootstrap collision** (`1024 vs 512` = nranks*256) | `--no-async-scheduling` |
| 20x slowdown + RCCL hangs | `VLLM_DISABLE_PYNCCL=1` | never set it |
| Silent wrong sums, cross-node | CXI/RCCL corruption, intermittent ~2/3 | `NCCL_NET_GDR_LEVEL=0` (unset != off) |
| 42% of engine time in 30 s stalls | **PP=4**; PP=1 arms lose 0% | prefer TP within a node, replicas across |
| Turn-1 400s logged as SUCCESS | SGLang needs **BOTH** `--reasoning-parser kimi_k2` AND `--tool-call-parser kimi_k2` | set both; no `--enable-auto-tool-choice` |
| Agent runs 36 min, invents a `Submit` tool, exits rc=0 | **MCP failed at init** -> no optarena tools, and nothing checks | semaphore + relaunch; connect budget is 5 s |
| ~96 tok/s numbers across all smokes | **tuned MoE config never loaded** | all those numbers are void |
| Every run served TRITON_MLA | AITER never enabled | note: the AITER master switch BREAKS MLA prefill on gfx942 (`fmha_v3_varlen_fwd invalid arg`) |
| "Throughput regression" at gate 4 | divided by wall clock **including ~170 s ramp** | measure steady state |

## Configuration that survived

```
INFERENCE_MODE=pp, INFERENCE_NODES=4, TP=4
--kv-cache-dtype fp8_e4m3 --page-size 64 --context-length 262144
--mem-fraction-static 0.42 --attention-backend triton
--reasoning-parser kimi_k2 --tool-call-parser kimi_k2
AGENTS_PER_NODE=12
```

`--mem-fraction-static` is the trap: on an APU it sizes KV against **node-wide RAM per rank**,
so the fraction runs BACKWARDS -- lower it to fit, do not raise it. 0.42 is the working value.

**256k context is free.** `max_running_requests` measured 2175 at that context, and concurrency
12 never touched the ceiling. The real context ceiling over 62k turns was **66,966 tokens**, so
the 262144 setting has never once been the binding constraint.

## Dead ends -- do not re-run these

- **CXI knob search: null.** The baseline scored both 4/5 and 5/5 on the same config, so the
  harness could not resolve a difference. Only `FI_MR_CACHE_MAX_COUNT=0` was proven harmful.
- **ofi plugin cannot be bind-mounted into the SGLang image** -- glibc 2.39 vs 2.35. The image
  already carries libfabric + libcxi, and CXI does not fix prefill anyway.
- **TP=8 / PP=1**: dead, KV goes 30 GiB negative; TP=8 also host-OOMs.
- **vLLM V1 + sibling + async**: dead.
- MI300A allocates **96 of 128 GiB** to HIP, and vLLM's fraction multiplies TOTAL -- set 0.70.
