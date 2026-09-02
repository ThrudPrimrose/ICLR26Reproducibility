# What the harness does about cheating

Companion to [`agent_evasion.md`](agent_evasion.md). That document is what agents DID; this one is
what the harness DOES about it, mechanism by mechanism, and what each mechanism provably does not
catch. Neither repeats the other: a case study belongs there, a file:line belongs here.

Every line number is against `/capstor/scratch/cscs/ybudanaz/x86_64/optarena` at
`69965f42` + working tree of 2026-09-01. `hpcagent_bench/harness/scoring.py` is being rewritten
concurrently (determinism gate); its numbers were re-read at md5 `a18440cc856ceaefc5ec6696647a3447`
and will drift. Everything else is stable.

## 1. Threat model

The agent is served a NumPy reference and asked for a faster native kernel with the same outputs.
Honest is not "did not lie". It is: **the submitted kernel computes the reference's function on
every input in the kernel's domain, and the recorded time is the time to compute it.**

| moved | dishonest | why the gate can still pass |
|---|---|---|
| the function | yes | the two agree on the inputs the generator draws |
| the domain | yes | the generator never draws the others |
| the timing | yes | the work happens once, outside the timed reps |
| the arithmetic (FP reassociation) | **no -- permitted** | tolerance, not bit-equality, is the contract |
| the oracle or the manifest | yes | grading compares against what was tampered with |

### The FP line: reassociation is allowed, by whatever route

An optimizer that lifts a serial recurrence to a parallel scan reassociates the sum. That is a
legitimate implementation of an associative operator and it is the transformation under study, so
it cannot also be the thing that fails a submission. **The policy is that the grading tolerance
decides whether a submission is safe, not a rule about which constructs its source may contain.**
A build that reassociates -- or that goes further and enables `fast-math` -- and still matches the
oracle inside tolerance is a valid submission.

The harness does not put `-ffast-math` on the build line (`hpcagent_bench/flags.py:60` keeps it
off; `hpcagent_bench/harness/sandbox.py:159` refuses it as a build token). A submission can reach
the same semantics from source anyway. Measured on this image, gcc 16.1 / llvm 22.1.7, a float64
sum reduction at the harness CPU baseline
(`-O3 -march=native -fopenmp -fno-math-errno -fno-trapping-math -fno-signed-zeros
-ffp-contract=fast -fstrict-aliasing -fPIC`), counting packed `vaddpd` against scalar `vaddsd` in
`objdump -d`:

| source construct | vaddpd | vaddsd | reassociates | reach |
|---|---|---|---|---|
| plain C, gcc | 0 | 7 | no | -- |
| `#pragma GCC optimize("fast-math","Ofast")` | 3 | 3 | yes | full fast-math |
| `__attribute__((optimize("fast-math")))` | 3 | 3 | yes | full fast-math |
| plain C, clang | 0 | 9 | no | -- |
| `#pragma clang fp reassociate(on)` | 10 | 3 | yes | reassociation only |
| `#pragma omp simd reduction(+:t)` | 1 | 6 | yes | reassociation only, one loop |

All six rows reproduced on this toolchain. One correction to the recipe: `#pragma clang fp` is
refused mid-block (`can only appear at file scope or at the start of a compound statement`); at a
compound-statement start it gives the 10/3 above. The gcc row's first packed instruction is
literally `vaddpd (%rax),%ymm0,%ymm0`. The clang `omp simd` row is 8/7, not 1/6 -- 1/6 is gcc.

`fast-math` reaches slightly wider than reassociation: it also assumes no NaN/Inf
(`-ffinite-math-only`), permits `a/b -> a*(1/b)` (`-freciprocal-math`), and in the general case
flushes denormals. Two of those are moot here and were checked rather than assumed: the per-function
attribute route produced no `set_fast_math` constructor and no `crtfastmath.o` (`nm` count 0), and
neither did `-ffast-math` on a shared-library link line, so denormal flushing is unreachable in this
harness; and on fp64 gcc kept `vdivpd` under the attribute rather than substituting a reciprocal.
What is left is reassociation plus the no-NaN/Inf assumption. A kernel whose inputs can produce an
Inf can therefore come out categorically wrong rather than slightly rounded -- and that is what the
correctness gate is for, which is the whole point of grading on tolerance instead of on constructs.
The harness's own `_FP_RELAX` (`flags.py:60`) is deliberately narrower than fast-math.

**Consequence a reader should not have to derive: "no fast-math flag" is a build-configuration
rule, not a semantic guarantee.** FP results are not bit-reproducible across submissions by
construction, and the grading tolerance -- not the flag policy -- is what bounds correctness. The
contract is in
[`hpcagent_bench/docs/numerical_validation.md`](../../optarena/hpcagent_bench/docs/numerical_validation.md):
elementwise `|a-e| <= atol + rtol*|e|` (fp64 band 1e-9 / 1e-11) **or** the LAPACK normwise ratio
`max|a-e| / (eps * log2(n) * ||e||_inf) <= 30`.

Nothing in the harness scans submitted source text. Grepping `hpcagent_bench/` and `containers/`
for `forbidden`, `banned`, `disallow`, `pragma`, `fast-math`, `Ofast` finds only build-token
policy (`sandbox.py:159`), manifest validation (`spec.py:1576`), and the agent-side advisory
`containers/agent/tools/syntax_check.py`, which runs inside the agent's own container. That is now
a deliberate absence, not a gap.

The campaign says the same thing from the data side: `agent_evasion.md`'s `fastmath` detector
fired on 8 submissions and confirmed **0** (every hit a comment, four of them saying the code
AVOIDS reassociation to stay bit-exact), and `compiler_pragma` fired on 6 and confirmed **0**
(`fp-contract=off` x3, `no-tree-vectorize`, `unroll-loops`, `O3`). 14 textual hits, zero actual
uses of a fast-math construct.

### Determinism, one level down

Same concern, different layer: a result that depends on iteration order is unreproducible the way
a fast-math build is uncomparable. Upstream dace `5f9a3eb0d` ("use ordered sets in gpu transforms")
and `4a4bb0a3c` ("use `dict` instead of `set` when adding a NestedSDFG") are that bug -- Python set
order leaking into generated code, so the kernel changes with `PYTHONHASHSEED`. Use `OrderedSet`,
never a plain `set`, anywhere order is observed.

## 2. Precautions in place

| # | mechanism | file:line | what it catches | what it provably does not |
|---|---|---|---|---|
| 1 | two grading seeds | `harness/hidden_tests/seeds.py:35,38,41,47` | `/score` grades seed 1, `/submit` seed 2 (`harness/service.py:634`, `scoring.py:719`), so a kernel fitted to the `/score` feedback loop fails the recorded grade | the seeds are literally `1` and `2`; only `.dockerignore:12` + `scripts/check_no_hidden_in_image.py` keep them out of the agent image. An agent that never calls `/score` is unaffected |
| 2 | held-out fresh-seed reverify | `scoring.py:352` `independent_verify`, seed `:376`, public `:399`, check `:463`; gate `service.py:385` (`record.harden`) | a full fresh `Sandbox` rebuild + clean re-run on different VALUES before any `submissions` row is written | same SIZE, different values only. The "fresh" seed is `secret_seed_first()` -- the seed the agent can already probe through `/score`. Skipped entirely when `record.harden=false` |
| 3 | dual-oracle check | `scoring.py:282` `_dual_oracle_check`, called `:453`, `:1709`; config `config.yaml:324` | grades the submission against the compiled C reference as well as NumPy | **best-effort, never a hard fail**: returns `(True, False)` when the C reference did not build, and the caller swallows the `RuntimeError`. Both oracles come from the same manifest, so a wrong manifest is invisible to both |
| 4 | held-out cases + `overfit` | `harness/hidden_tests/__init__.py:53` `hidden_cases`; verdict `scoring.py:1019`; label `harness/recording.py:845`, `harness/runner.py:131` | 5 correctness-only cases = shape rung x `hidden_variant` x config subset. Public-correct + hidden-failing is recorded `status="overfit"` in `attempts`, never as a submission | 5 cases is the entire breadth. `cap_rung` (`hidden_tests/__init__.py:44`) clamps every rung to <= the timed preset, so a kernel that only breaks ABOVE the graded size is never probed |
| 5 | `hidden_variant` rotation | `support/distributions/hidden.py:46` (5 fixed variants), threaded `frameworks/benchmark.py:116`, consumed `scoring.py:754` | rotates the value distribution of every float array per held-out case, so a dropped sign branch or overflow guard shows up | **skips every manifest with `init.func_name`** -- `benchmark.py:131` says so in the source. Never rotates integer/index arrays, scalars, or shapes |
| 6 | `_grade` over every output | `harness/grading.py:58`, loop `:62`; comparator `frameworks/utilities.py:219` | iterates **`spec.output_args`** -- every declared output -- through `compare_arrays` with rtol/atol and ANDs the verdicts. Shape-checked, complex-aware, exact for integer dtypes, matching NaN positions and Inf signs | reads the output list FROM the manifest, so removing an entry from `output_args` removes an array from grading. That is exactly the tamper `oracle_integrity.py` exists for -- and it is never run (row 7) |
| 7 | oracle-integrity digest | `hpcagent_bench/oracle_integrity.py`, `ORACLE_PATTERNS` `:42`, `digest` `:57`, `verify` `:108`, `OracleTampered` `:97` | sha256 over every `*_numpy.py` and `*.yaml` under `paths.BENCHMARKS`; content, not mtime; raises on modified, removed OR added paths | **zero production call sites.** `grep -rn "oracle_integrity\|OracleTampered"` over the whole repo hits only `tests/test_oracle_integrity.py` and the module's own CLI. A mismatch does nothing today. Pattern coverage is also short: see gaps |
| 8 | `suspect` flag | threshold `scoring.py:336`, default `config.yaml:325` = `1000.0`, predicate `:345`, set `:382`; recorded `recording.py:828` | marks a submission whose speed-up is `> 1000.0` or non-finite | **a flag, not a rejection** -- the row still counts. `recording.py:828` copies `verify.suspect` ONLY, so with `record.harden=false` the column is structurally zero. Cannot fire below the top grid step (see gaps) |
| 9 | determinism gate | `scoring.py:253` `_determinism_check`, `:243` `_reproduces`, `:291` `_verify_triad`; test `tests/test_determinism_gate.py` | two clean runs of ONE build must agree AND run 1 must grade correct -- catches uninitialised memory, UB, races that passed once by luck | mid-rewrite: the signature is now `n_accum: int` with a LAPACK normwise band, not `bitwise: bool`, so `tests/test_determinism_gate.py:52` and `:104` no longer match the code. The new band admits a few-ulp float-atomic GPU reduction that the `lang-cuda`/`lang-hip` skill pages say is rejected |
| 10 | entry-point / ABI contract | `harness/native_call.py:494`; python analogue `:828` | the symbol is fetched BY NAME through cffi; a missing one raises a message naming the exact required signature and telling a C++ author to use `extern "C"` | only checks the symbol EXISTS. `ffi.cdef` declares the type; nothing verifies the `.so`'s real arity or argument types, so a right-name/wrong-signature kernel is called anyway. Nothing inspects extra exported symbols or `.init_array` constructors |
| 11 | build-line control | `sandbox.py:143` `_COMPILE_PREFIXES=("-I","-D")`, `:144` `_LINK_PREFIXES=("-l","-L")`, `:159` `_NEVER_ALLOWED`, `:194` `split_build`; commands built `languages.py:1554` | the measured build line comes entirely from `envs/compilers.yaml` -> `flags.py`. An agent's `build` list is filtered to `-I/-D` and `-l/-L`; `-O3`, `-march=native`, a response file `@file` and every other token are dropped. `_safe_link` (`:180`) refuses `-l:file` and any `-l` containing `/` | dropped SILENTLY -- neither the agent nor the record learns a flag was ignored. `-D` is an unrestricted-content channel. No Makefile or build file is ever consulted (`run_build_commands`, `languages.py:1640`, runs argv with no shell), and the judge's environment is a different container from the agent's, so those two routes are closed by construction rather than by a check |
| 12 | library request path | table `envs/libraries.yaml`; `languages.py:1010` filter constants, `:1052` `library_tokens`, `:1117` trial link; live path `sandbox.py:345` + `:73-118` | pkg-config's answer is filtered to `-I` on compile and `-L`/`-l` on link -- explicitly because `openblas.pc` really does emit `-fopenmp`, which would let a library request switch OpenMP on for a whole TU. rpath is derived from pkg-config, never accepted from a submission | `library_build_flags` has **no production consumer** (`grep`: only `tests/test_library_requests.py`), so that filtering is not on the `/submit` path. The live shared-mount route lets an agent link a `.so` it built itself with any flags it likes; nothing inspects that object |
| 13 | replay-cache followups | `native_call.py:560` (followups run after the last timed rep, same warmed child); producer `scoring.py:934`; test `tests/test_replay_cache_detection.py` | the memoization exploit -- cache rep 1, let warmup absorb the one honest call, let min-of-k take the fastest replay -- is reproduced against the shipping harness in that test. Defence: the 5 held-out cases run as followups INSIDE the same warmed child, so the hot cache replays the public answer onto unseen inputs and grading fails it | only detects a cache keyed on "have I run before". A cache keyed on an input hash computes honestly per distinct input and evades this entirely. Only fires where held-out cases exist, i.e. `/submit`, never `/score` |
| 14 | workspace zeroing | `native_call.py:536` | the harness-supplied scratch pair is zeroed per rep, untimed -- the one channel a kernel could memoize through and have the replay timed | only the harness's workspace. A kernel's own `static`/`.bss`/`malloc` storage is untouched (that is row 13's job), as is `/tmp` and the shared mount |
| 15 | grading-secret env scrub | `native_call.py:883` `GRADING_SECRET_ENV_PREFIXES`, `:886` `scrub_grading_secrets`, called `:955` | the measurement child deletes every `HPCAGENT_BENCH_SEEDS_*` variable before `dlopen`, so a kernel cannot `getenv` the seed it is graded on and regenerate the answers | prefix-scoped. `HPCAGENT_BENCH_SHARED_DIR`, `PYTHONPATH` and the rest survive into the process running agent code, and nothing stops the kernel reading `/proc` |
| 16 | timing discipline | `harness/timing.py:95` warmup, `:105` 50 ranked reps, `:139` `sampled_reps`, `:159` min-of-k, `:270` `validate_repeat`; route split `scoring.py:1011` | warmup is applied identically to submission and every baseline, so the ratio stays fair; `/score` is 5 reps and unrecorded, `/submit` keeps the ranked 50 and fails loudly rather than crediting an underpowered distributional test | warmup discard is what ENABLES row 13's exploit. `min_of_k` rewards one lucky sample |
| 17 | preset pinning | `service.py:596` | `/submit` ignores a body-supplied `preset` and grades the run's configured size, so a recorded row cannot measure a size the client chose | `/score` and `/profile` still honour an arbitrary preset -- neither is recorded |
| 18 | path containment | `sandbox.py:50` `resolve_shared`, called `service.py:265`, `:312`; basename contract `service.py:248` | the judge compiles and `dlopen`s what these name, so `resolve()` follows symlinks BEFORE the containment test and `source_file` must be exactly `<kernel>.<ext>` | contains WHERE a file comes from, never what is in it |
| 19 | run identity from env | `containers/agent/tools/http_json.py:184`, merged after the body `:263` | `run_id` / `optimizer` come from the launcher environment and are merged after the payload, so no tool payload can overwrite them | the agent owns its own container's env; this stops payload spoofing only. Token spend is self-reported (`:220`) |
| 20 | rank validation | `service.py:98` `421`, `:108` `rank_error`, enforced `:539` | a stale `JUDGE_URL` otherwise lands on a wrong-but-live judge that grades plausibly -- a wrong measurement wearing a right label | infrastructure correctness, not adversarial: the rank comes from the agent's own env |
| 21 | hidden-tests firewall | `.dockerignore:12`, `scripts/check_no_hidden_in_image.py:210`, `tests/test_firewall.py` | carries the entire guarantee behind row 1, since the seed VALUES protect nothing | a static check on Dockerfile + repo. A bind-mount of the repo into the agent container defeats it |
| 22 | harvest keeps source, not score | `service.py:405` `HarvestLedger`, `:318` `harvest_unsubmitted` | a run that dies without submitting has its last correct `/score` SOURCE re-graded through the ordinary `/submit` path -- promoting the `/score` NUMBER would put a min-of-5 measurement on a different seed into the same column | bounded by `record.harvest_budget_s`; what the budget cuts is logged, not recorded |
| 23 | `/profile` earns nothing | `service.py:671`, `:674` | diagnostic only -- nothing graded, recorded, or compared to a baseline | `tool: "none"` (`service.py:727`) builds and runs the agent's own instrumented source and returns its stdout: an unmetered execution channel on the judge |

## 3. Known gaps, with evidence

Counts below are my own sweep of the working tree on 2026-09-01; the tree is live and shared, so
they drift from earlier figures by one or two files. Method and totals are given so a zero is
distinguishable from a crash.

| gap | measurement | consequence |
|---|---|---|
| `ORACLE_PATTERNS = ("*_numpy.py", "*.yaml")` (`oracle_integrity.py:42`) misses the generators | swept `hpcagent_bench/benchmarks/`: **3149** `.py` scanned, 656 skipped (`__pycache__` + `*_numpy.py`), **2493** parsed, **164** define a top-level `initialize()` and are outside the digest | 164 files that decide what data a kernel is graded on are unhashed. `gesummv/gesummv.py` is one of them, and `oracle_integrity.py:10` cites that file as the real incident the module was written for |
| the digest is never called | `grep -rn "oracle_integrity\|OracleTampered"` over the repo: hits only `tests/test_oracle_integrity.py` and the module's own `__main__` | even the 2493-file half that IS covered is not checked at run start or before a grade. The "VOID the run" behaviour in `oracle_integrity.py:19` is aspirational |
| legacy `init.func_name` gets no distribution rotation | parsed **659** manifests, 0 skipped; **659** carry an `init:` block, **171 (26%)** set `init.func_name` | 171 kernels get seed rotation and the shape ladder but see the same value distribution in every held-out case, so a value-distribution overfit on them is undetectable. `frameworks/benchmark.py:131` states this |
| `suspect` cannot fire below the top grid step | credited speed-ups come off a `(1.01)**k` grid; `1.01**694 = 997.78`, `1.01**695 = 1007.75`, and the flag is `speedup > 1000.0` (`config.yaml:325`, `scoring.py:345`) | the flag has exactly one reachable rung. A kernel that deletes 99.9% of the work and stays correct on all 5 variants scores 999x and reads `clean` |
| the `suspect` column is verify-only | `recording.py:828`: `suspect = 1 if (verify is not None and verify.suspect) else 0` | with `record.harden=false` (row 2) `verify is None`, so the column is structurally zero for every row in the run -- not "no suspects found", but "not asked" |
| the `/shared/lib` route bypasses the flag matrix | `sandbox.py:76` reads the DIRECTORY rather than a declared list; `_safe_link` (`:180`) passes a bare `-lfoo`; `sandbox.py:326` is a bare `shutil.copy2` for a library delivery | an agent can compile its hot loop into `/shared/lib/libfoo.so` with any flags it likes and submit a shim. Under the permissive FP policy this no longer matters for fast-math, but it also carries `-march`, LTO, PGO and anything else, and nothing inspects the object |
| dual-oracle fails open | `scoring.py:286` returns `(True, False)` when the C reference did not build; `:455` swallows the exception | on any kernel whose C emit or build fails, the second oracle silently passes and row 3 is not in effect |
| the determinism test contradicts the code it covers | `scoring.py:253` takes `n_accum: int`; `tests/test_determinism_gate.py:104` reads `inspect.signature(fn).parameters["bitwise"]` | two tests fail against the current tree. Reported, not fixed -- `scoring.py` is owned by another agent this session |

## 4. What is explicitly permitted

Not enforced, and deliberately so. Listing it matters because a reader of a surprising result
needs to know which of these could be behind it.

| construct | status | note |
|---|---|---|
| `#pragma GCC optimize(...)`, `__attribute__((optimize(...)))` | permitted | full fast-math reach; table in Sec. 1 |
| `#pragma clang fp reassociate/contract/fast`, `#pragma float_control` | permitted | reassociation only, in the measured case |
| `#pragma STDC FP_CONTRACT ON` | permitted, and a no-op | the baseline already pins `-ffp-contract=fast` (`flags.py:120`, `_FP_CONTRACT`) |
| `#pragma omp simd`, `omp ... reduction(...)`, `omp declare reduction` | permitted | the scoped, portable way to say the same thing |
| `#pragma GCC ivdep`, `#pragma unroll`, `#pragma clang loop`, alignment hints, `__restrict__` | permitted | measured no-ops on the reduction: 0 packed / 7 scalar, same as baseline |
| macro- and `_Pragma`-obfuscated spellings of any of the above | permitted | they were checked because a detector would have had to see through them: `gcc -E` expands `#define FAST __attribute__((optimize("fast-math")))` and `_Pragma("GCC optimize(\"Ofast\")")` into their canonical form, so any future scan must run on the PREPROCESSED unit, never the raw file |
| `-D` in the submission's `build` list | permitted, unrestricted content | `sandbox.py:143`; a `-Dname=__attribute__((optimize(...)))` is a live route to the same place |
| requesting a compiler FLAG through `build` | **refused** | `sandbox.py:194` `split_build` drops it silently. This is a comparability rule about the build line, unchanged |
| a Makefile or build file the submission writes | **ignored** | never read: `languages.py:1554` constructs the argv, `languages.py:1640` runs it with no shell |
| a response file (`@file`) | **ignored** | not on the `-I/-D/-l/-L` allow-list, so dropped by `split_build` |
| judge environment manipulation | **not reachable** | the agent runs in a separate container; `native_call.py:886` additionally scrubs `HPCAGENT_BENCH_SEEDS_*` from the measurement child |

## 5. Limitations

- **Source recovery ceiling: 37%.** 1727 of 4677 accepted submissions have neither stored graded
  text nor a workspace file (`agent_evasion.md`). Nothing in either document says anything about
  them.
- **Textual detection tops out below the bug rate.** On the 60 `ext_break_*` submissions read end
  to end against the oracle, the nine detectors fired on 21: 20 true, 1 false, 6 missed --
  precision 0.95, recall 0.77. **All six misses were semantic** (`reduction(min:x)` with `x = i`
  reads as a correct first-crossing search), which no textual detector can see.
- **The precaution inventory is a code reading, not a red-team result.** Each "does not catch"
  column is derived from the mechanism, not demonstrated by an exploit -- except rows 13 and 8,
  where `tests/test_replay_cache_detection.py` and the `1.01**695` arithmetic are constructive.
- **`scoring.py` moved while this was written.** Its line numbers are pinned to one md5 (top of
  this file) and the determinism-gate row describes a rewrite in progress.
- **The FP table is one toolchain, one kernel, one node.** gcc 16.1 / llvm 22.1.7 on zen3, a
  float64 sum reduction. It shows that these constructs reach fast-math; it is not a survey of how
  much they buy.
