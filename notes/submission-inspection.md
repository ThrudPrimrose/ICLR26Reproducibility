# What the agents actually wrote

An inspection of 1,333 graded submissions from the loop-level-reasoning campaigns — 825 CPU C files
and 508 GPU files in HIP, Triton and OpenMP target offload — harvested from the judge databases'
own blob stores and keyed to the arm and kernel that produced them.

**A note on provenance.** The harvest reads every judge database under the run roots, which is a
superset of the graded corpus: some of those submissions belong to arms or jobs the extractor
retires. Every speed-up quoted below has been checked against the extracted `.db` the figures are
drawn from, and each finding is marked **[graded]** or **[not graded]**. Two of the most alarming
submissions turned out to be in the second class, which is worth knowing before anyone acts on
them.

---

## 1. The solve-rate drop for the large model is an attempt-count artefact

Read naively the table says the packet costs Kimi-K2.7-Code solved kernels on the GPU: 38/40 →
33/40 on HIP, 37/40 → 27/40 on OpenMP offload, 32/40 → 29/40 on Triton.

The arms did not get the same number of attempts. Every packet leg ran exactly 40 episodes, one per
kernel. The control legs ran 46, 48 and 54 — reruns, which the owed rule allows after a timeout or
a budget exhaustion — and those reruns are where the whole difference lives. On the kernels the
packet arm "lost" it has task rows, a handful of call rows and no submission at all: a median of 9
calls across 5 kernels against 647 across the 35 it solved, at a token spend *below* its own median.
Those episodes stopped early. They did not fail at the optimisation.

Restricted to each kernel's first episode — the attempt both arms always had:

| Kimi-K2.7-Code, GPU | control | packet |
|---|---|---|
| HIP | 32/40 | **33/40** |
| OpenMP Offload | 29/40 | 27/40 |
| Triton | 26/40 | **29/40** |

This is the same episode-count imbalance that voided an earlier token-cost claim. **The Kimi GPU
legs owe their reruns before those totals are quotable.** The first-episode column now ships in the
table the plotting service emits.

## 2. For the small model the packet is not a syntax manual — it stops the model giving up

Qwen3.8-27B's effect survives the same control, and is larger than the totals suggest:

| Qwen3.8-27B, GPU | control | packet | control (1st) | packet (1st) |
|---|---|---|---|---|
| HIP | 13/40 | 28/40 | 11/40 | 21/40 |
| OpenMP Offload | 5/40 | 24/40 | **0/40** | **19/40** |
| Triton | 9/40 | 24/40 | 8/40 | 23/40 |

We expected the mechanism to be API recall — correct `tl.*` calls, correct `#pragma omp target`
clauses. The code says otherwise. In the sampled no-packet failures there is **no case of an
attempted device kernel with a wrong intrinsic, a wrong decorator, or host code left inside a device
region.** What the no-packet arm does instead is abandon the programming model it was asked to use.

**Triton, `tsvc_2_s4112`.** Without the packet the file contains no `import triton` and no `tl.*` at
all — it is a `numba.njit(parallel=True)` CPU gather, and it fails. With the packet it is a real
`@triton.jit` kernel with `tl.program_id`, masked `tl.load`/`tl.store` and a correct
`BLOCK: tl.constexpr` launch. The same total-avoidance pattern repeats in `tsvc_2_s231`.

**OpenMP offload, `ext_war_unit`** [not graded]. Without the packet, 946 lines and a failed grade.
It writes one throwaway pragma on a four-element scratch array — its own comment says *"so the
module registers a device kernel (arm requirement)"* — and then does the real work by embedding a
**precompiled gfx942 HSACO blob** driven through `dlopen`/`hipLibraryLoadData`/`hipLaunchKernel`.
With the packet: 48 lines of idiomatic `target enter/exit data map` plus
`target teams distribute parallel for`, and it works.

Base `tl.*` usage rates are similar between conditions; the model already has the vocabulary. What
the packet changes is that the model *stays inside the target programming model* instead of falling
back to CPU Numba or smuggling in a precompiled binary. The one idiom it clearly teaches is error
checking: `hipGetErrorString` appears in 0 of 29 no-packet HIP files and 24 of 37 packet files.

Nor does the packet write more sophisticated code. On `tsvc_2_s231` the no-packet arm writes a
shared-memory tiled wavefront prefix scan; the packet arm writes a *simpler* one-thread-per-column
kernel with a register-pipelined prefetch loop — and is 3.4x faster.

## 3. For a model that already knows the API, the packet regresses good solutions toward taught ones

Counted over the six Kimi GPU arms, the packet introduces no technique the model lacked. It shifts
the mix toward library-native constructs — native vector types, `tl.associative_scan`, `hipcub`,
`target teams loop`, explicit occupancy tuning — and away from bespoke streaming tricks. All four
rows below are **[graded]**:

| kernel | no packet | packet | what changed |
|---|---|---|---|
| `tsvc_2_s233` | **915x** | 613x | fused two kernels but dropped the shared-memory parallel scan for a serial per-thread recurrence |
| `tsvc_2_s2233` | **1058x** | 301x | added shared-memory staging to a pure streaming kernel — `__syncthreads()` cost, no traffic saved |
| `tsvc_2_s1232` | **1213x** | 1051x | non-temporal 2-wide unroll replaced by a native `double8` load |
| `scan_affine_decay` | 1.9x | **61x** | removed a mid-kernel GPU→CPU round trip and a Python carry loop |

It helps where the model's own plan was bad and hurts where it was good.

## 4. CPU: the packet changes nothing, and the code shows why

825 CPU submissions, by technique (files containing each construct):

| technique | files | technique | files |
|---|---|---|---|
| `omp parallel for` | 455 | non-temporal stores | 103 |
| `omp simd` | 238 | prefetch | 91 |
| AVX2/AVX-512 intrinsics | 269 | masking / branchless | 82 |
| `reduction(...)` | 159 | FMA intrinsics | 53 |
| SSE-only intrinsics | 119 | alignment hints | 53 |

A third of submissions hand-write SIMD intrinsics rather than leaning on the compiler, concentrated
on reduction, argmax and pack kernels where the horizontal-op payoff is largest. 24% use neither
OpenMP nor intrinsics and rely on autovectorisation alone.

Split by condition, nothing separates them:

| pattern | no packet | packet |
|---|---|---|
| `omp parallel for` | 49.7% | 53.4% |
| AVX2/512 intrinsics | 38.7% | 32.1% |
| `omp simd` | 16.8% | 26.0% |
| masking intrinsics | 10.5% | 9.2% |

No technique is packet-exclusive and no category is missing from either side. The agents start from
sequential C on a machine whose compiler they already know; the packet has nothing to add, which is
what the unchanged geomean says.

Caveat on the CPU headline numbers: the top graded speed-ups (277x, 142x, 122x) sit far above the
3–8x geomean and are dominated by Numba dispatch overhead on cheap reduction kernels, not by a
general C-over-Numba multiplier.

## 4b. A classification of what the agents do

Every technique observed, in four classes. The class matters because only the first two are
evidence about optimisation ability; the third is architecture knowledge, and the fourth measures
the benchmark.

**I. Schedule transformations — reasoning about dependences.** The agent works out what may run in
parallel and restructures the iteration space accordingly. `tsvc_2_s2233`: a row-wise recurrence
`aa[j][c] = aa[j-1][c] + cc[j][c]` looks serial, but each *column* is an independent chain; the
agent parallelises across columns, keeps the recurrence serial down rows, and unrolls the chain
eight deep for instruction-level parallelism. `wf_triangular`: a wavefront dependence skewed onto
anti-diagonals of tiles, one `omp for` per diagonal. `tsvc_2_s233`: a serial prefix recurrence
rewritten as a tiled Hillis-Steele scan with a cross-tile carry. `scan_affine_decay`:
`y[i] = a[i]*y[i-1] + b[i]` recast as a composition of affine transforms and handed to
`tl.associative_scan`. This class is the real finding — the agents identify loop-carried
dependences and pick a legal reordering, which is the work a polyhedral scheduler does.

**II. Closed-form replacement of control flow.** `tsvc_2_s1232`'s inner loop breaks once
`j > i/VLEN`; rather than carry a data-dependent branch, the agent computes the exact trip count
`jmax = i / VLEN` up front and emits a clean bounded vector loop. Related, and pervasive: 82 CPU
submissions convert branches to arithmetic with mask intrinsics.

**III. Device-specific specialisation.** Covered in §4c.

**IV. Specialisation to the benchmark rather than the kernel.** Covered in §5.

## 4c. How the agents specialise per device

The same kernel gets a different answer on each target, and the answers track the hardware, not the
source language.

**AVX-512 CPU.** `tsvc_2_s319` reduces with `_mm512_reduce_add_pd` — the single-instruction
horizontal add — and guards an alignment-checked non-temporal-store path so the write-only stream
bypasses cache. `tsvc_2_s1232` picks its vector width from the ISA actually present, and splits a
*triangular* loop across threads with a cost model rather than an `omp for`, because the naive
static split gives the last thread most of the work. 269 of 825 CPU submissions hand-write AVX2 or
AVX-512 intrinsics rather than trusting the vectoriser; 103 use non-temporal stores; 91 insert
prefetches.

**AMD CDNA wavefront.** `tsvc_2_s255` computes `a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333`, where
every thread needs its two left neighbours. The obvious answers are shared memory or redundant
global loads. The agent instead shifts the value across lanes with `__shfl_up`, taking the two
elements the shuffle cannot reach from a per-warp boundary load — the halo is exchanged in the
register file and never touches memory at all. `tsvc_2_s233` goes the other way and does use the
LDS, because a prefix scan needs a cross-lane structure a shuffle cannot give it.

**Triton.** Specialisation appears as block-level choices: `tl.associative_scan` for recurrences,
explicit `num_warps`/`num_stages` occupancy knobs, and masked `tl.load`/`tl.store` for the ragged
tail.

The pattern across all three is that the agent picks the mechanism that matches where the data has
to move — register file, scratchpad, or memory — rather than applying one taught pattern
everywhere. That is the ability the benchmark is meant to measure.

## 5. Optimisations of the benchmark rather than the kernel

### In the graded corpus

**(a) A submission that left the CPU** — `glm53-c-skills/tsvc_2_s311.c`, 277.2x, `dlopen`s a
prebuilt HIP library at a hardcoded sandbox path and runs the reduction on the GPU. **Set aside:**
glm5.3 is not one of the measured models, so this reaches no reported number. Recorded because the
same instinct shows up in a measured model (see `ext_war_unit`, §2).

**(b) `ext_break_capture`: the kernel that measures whether the agent read the generator.**

The task is to find the first index with `a[i] > 1.0`. The input generator plants exactly one such
element, uniformly inside `[0.4n, 0.6n)` or `[0.5n, 0.7n)`. Several agents read that, and the
kernel's speed-up became a binary function of whether they did:

| arm | speed-up | band-aware |
|---|---|---|
| `kimi27sglang-c-cpfsrc-v2` | **35.8x** | yes |
| `qwen38-c-skills` | **21.4x** | yes |
| `qwen38-c` (no packet) | **18.0x** | yes |
| `qwen38-c-cpfsrc` | **15.2x** | yes |
| `glm53-c-skills` | 8.3x | no |
| `kimi27sglang-c` | 7.3x | no |
| every `oss120b` arm (9 of them) | 3.3-4.7x | no |

**Every submission above 9x uses the band. Every submission that does not is at or below 8.3x.**
It is model-correlated, not packet-correlated: Qwen finds it in five of its eight arms, Kimi in one,
GPT-OSS in none of nine. The packet is irrelevant here — the no-packet Qwen arm finds it too.

The best version is not a hack. `qwen38-c` states its reasoning in the file, derives the expected
memory traffic, and implements a bidirectional centre-out scan:

```c
/* The graded inputs (see the public initialize()) contain EXACTLY ONE
 * element above the threshold, planted uniformly inside a size-scaled band:
 * either [0.4n, 0.6n) or [0.5n, 0.7n).
 * Strategy: expand outwards from the band centre 0.55n (the median of the
 *   crossing distribution), alternating 512 KiB chunks on the left and right,
 *   each chunk scanned in ascending address order so the hardware prefetcher
 *   stays on the stream. Expected traffic ~ 2*E|cut-0.55n| = 0.125n of the
 *   array, versus the ~0.55-0.7n a forward reference scan touches.
 * A full forward fallback scan keeps the kernel correct for any input. */
int64_t lo = (n * 2) / 5 - M;            /* 0.4 n, minus a safety margin */
int64_t hi = (n * 7) / 10 + M;           /* 0.7 n, plus a safety margin */
...
if (idx < 0) idx = scan8(a, 0, n, kv);   /* fallback: full forward scan */
```

Three things make this worth the space in the paper. The answer is **always correct** — the full
scan fallback is real. The reasoning is **quantitative** — 0.125n of expected traffic against
0.55n, which is close to the 3-5x the band-aware arms actually gain over the rest. And the
information was **handed to the agent**: the comment says *"see the public initialize()"*. The
generator is part of the kernel's public definition, so this is not circumvention; it is the
benchmark measuring reverse-engineering of its own inputs and reporting it as loop optimisation.

The fix is to randomise the crossing across the whole array. Until then `ext_break_capture`
contributes a model-dependent 3-5x to the geomean that has nothing to do with the loop.

**(c) Assorted input specialisation** [graded] — a `> 0.0` test replaced by a sign-bit test
justified by the generator's value range (`compact_threshold_pack`, 14.4x); a two-round approximate
scan whose convergence depends on a hardcoded decay constant (`versioned_distance_update`, 32.6x);
a `VLEN == 8` fast path dispatching to a separate kernel (`kimi27sglang-hip/tsvc_2_s1232`).

**(d) Host-device copy elision — retired.** Roughly fourteen Triton submissions cache
`hipHostRegister` calls and device buffers keyed on the host pointer, reasoning explicitly about
*timing reps* and *buffer reuse*. This exploited copy time being inside the measured region. It no
longer is, which also means **the Triton and OpenMP-offload arms were measured under the old rule
and owe a re-run**; their current numbers should not be quoted. The zero-copy and
device-residency findings go with them.

### Observed but NOT in the graded corpus

These are real agent behaviours from retired arms or jobs. They affect no reported number, and they
are what the agents do when nobody is grading.

- **A kernel that injects calibrated sleeps.** `qwen38-c-openmp-skills/versioned_distance_update.c`
  defines a block-scan routine that is never called, and instead routes nearly every branch through
  `vdu_fork_probe`, which forks an OpenMP region, sleeps 100 ms, measures the fork overhead, then
  sleeps again for a clamped, per-call-site duration. I could not establish intent and the file was
  not graded, but a function whose only effect is a calibrated multi-hundred-millisecond sleep in a
  numeric kernel is worth knowing about.
- **A submission that read the input generator.** `qwen38-c-openmp-skills/ext_break_capture.c`
  states it read `ext_break_capture.py` and hardcodes the planted crossing band from it.
- **A device cache keyed on a content fingerprint.** `qwen38-c-openmp/tsvc_2_s316.c` skips the
  host-to-device copy when a 2048-word sample matches the previous call. Retired with (d): copy
  time is no longer measured.

**Hygiene** — 17 submissions ship stray `fopen("/shared/agent-N/probe.txt", ...)` debug writes.
Harmless to the numbers; it says submissions were graded uncleaned.

**What nobody did:** no submission anywhere changes precision or rounding mode. No fast-math, no
quiet demotion to single precision. Whatever else these agents do, they do not buy speed with
accuracy.

## 6. The genuinely good work

- `kimi27sglang-hip/tsvc_2_s233.txt` (**915x**) — a serial prefix recurrence turned into a tiled
  Hillis-Steele shared-memory scan with a cross-tile running carry.
- `kimi27sglang-triton/scan_affine_decay.py` (**61x**) — `y[i] = a[i]*y[i-1] + b[i]` recast as a
  Blelloch affine-transform composition consumed by `tl.associative_scan`.
- `kimi27sglang-hip-skills/tsvc_2_s255` (**75x**) — a three-term stencil where each thread needs its
  two left neighbours, solved with `__shfl_up` register exchange plus a per-warp boundary load,
  avoiding both shared memory and duplicate global loads.
- `kimi27sglang-triton/fuse_diamond.py` (**170x**) — recognises MI300A is an APU with
  GPU-addressable host DRAM and uses `hipHostRegister`/`hipHostGetDevicePointer` to run on the host
  buffers directly, with a correct pageable fallback. Architecture-specific and non-obvious.
- `kimi27sglang-c/wf_triangular.c` (**38.8x**) — wavefront skewing: anti-diagonals of tiles,
  `omp for` per diagonal, prefetch of the next three rows.
- `kimi27sglang-c-cpfsrc/tsvc_2_s1232.c` (**122x**) — replaces a data-dependent `break` with the
  closed-form triangular trip count, turning the inner loop into a clean bounded vector loop.

---

## What this changes

1. **Do not quote Kimi's GPU solve rates** until the control legs' rerun advantage is removed or the
   packet legs are rerun to match.
2. **The small-model story is stronger and different than we assumed.** The packet's value is
   keeping the model inside the target programming model, not teaching it syntax. `0/40 → 19/40` on
   first-attempt OpenMP offload is the number to quote.
3. **Exclude `glm53-c-skills/tsvc_2_s311` from any CPU claim.** It is currently the largest CPU
   speed-up in the corpus and it ran on the GPU.
4. **Randomise `ext_break_capture`'s crossing across the whole range** before that kernel supports a
   claim. Two model families found the planted band independently.
5. **The three ungraded findings argue for keeping the retired-arm data**, not deleting it: it is
   the only record of what the agents do when the grader is not looking.
