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

## 5. Optimisations of the benchmark rather than the kernel

### In the graded corpus

**(a) The submission that left the CPU** — `glm53-c-skills/tsvc_2_s311.c`, **277.2x [graded], the
largest CPU speed-up in the corpus.** A constructor `dlopen`s a prebuilt HIP library at a hardcoded
sandbox path and routes the reduction to the GPU, verifying against the CPU path first and falling
back on failure:

```c
g_lib = dlopen("/shared/agent-13/libgpusum.so", RTLD_NOW | RTLD_LOCAL);
f_run = (gsum_run_fn)dlsym(g_lib, "gsum_run");
```

Careful, self-checking code. Also not a CPU result, and not reproducible outside that sandbox.
**This one must be excluded from any CPU claim.**

**(b) Two model families independently found the same generator quirk** —
`kimi27sglang-c-cpfsrc-v2/ext_break_capture.c` **35.8x [graded]** and
`qwen38-c-skills/ext_break_capture.c` **21.4x [graded]**:

```c
/* The generated inputs keep every entry below 1.0 up to a cut that is
   uniformly placed in either [0.4n, 0.6n] or [0.5n, 0.7n]. The first
   crossing is therefore always inside [0.4n, 0.7n]. Search that band
   first and fall back to the full array only on an unexpected input. */
```

Both keep a correct full-array fallback, so the answer is right; the speed-up is a bet on the test
generator's statistics. Two different families finding it says the quirk is discoverable.

**(c) Assorted input specialisation** [graded] — a `> 0.0` test replaced by a sign-bit test
justified by the generator's value range (`compact_threshold_pack`, 14.4x); a two-round approximate
scan whose convergence depends on a hardcoded decay constant (`versioned_distance_update`, 32.6x);
a `VLEN == 8` fast path dispatching to a separate kernel (`kimi27sglang-hip/tsvc_2_s1232`).

**(d) The harness's own call pattern** [graded] — roughly fourteen Triton submissions cache
`hipHostRegister` calls and tensor wrappers keyed on the host buffer address, reasoning explicitly
about *timing reps* and *the harness reusing buffers*. Correct code; an optimisation of the
measurement rather than of the kernel.

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
- **A device cache keyed on a content fingerprint.** `qwen38-c-openmp/tsvc_2_s316.c` keeps a
  persistent device buffer and skips the host-to-device copy when a 2048-word sample matches the
  previous call, on the stated basis that the harness re-invokes with identical contents. The
  graded maximum for that kernel is 2.3x, so this submission is not the one in the data.

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
