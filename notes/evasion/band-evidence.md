# `ext_break_capture`: which setups read the input generator

Evidence for the void list. Each setup is decided on its SOURCE: a literal band bound, or a scan
launched over a fixed sub-range. The speed-up is recorded but is not the test, because a speed-up
cut fails in both directions on these data (see the bottom of this file).

Reproduce with `harvest_sources.py` (pairs each submission to the source blob at or before its own
timestamp) and `pair_check.py` (lists every submission/blob pair for one kernel).

## Band-aware: 10 of 47 setups carrying the kernel

### `gpu-llr-focus40-qwen38-c-openmp-skills` -- 1007.75x

blob `63/6389df2431e0f97c371a71c6ffdeabc6b17c2b01ccd51675020891bd39866fff.txt`

```c
L32: const int64_t start = 2 * n / 5;                 /* floor(0.40 n) */
```
```c
L33: int64_t end = 7 * n / 10;                         /* floor(0.70 n) */
```

### `gpu-llr-focus40-qwen38-hip` -- 290.93x

blob `6f/6f34f465b4a02f0880ee51f8b280f3618fe0f67eee382efbd63ee24a1becf44a.txt`

```c
L155: int64_t B = (n >= 1024) ? (n * 4) / 10 - 1024 : 0;
```

### `gpu-llr-focus40-qwen38-hip-perf-playbook-amd-clean` -- 275.86x

blob `79/79ff5651e6abdd770005c324f3f38a2ee2cad82cc5c4d318db2a1fb9eb131a01.txt`

```c
L155: const int64_t lo = (N > 0) ? (int64_t)(0.40 * (double)N) : 0;
```
```c
L156: const int64_t hi = (N > 0) ? (int64_t)(0.70 * (double)N) : 0;
```

### `cpf-llr-focus40-kimi27sglang-c-cpfsrc-v2-clean` -- 35.82x

blob `44/442c8171505ee8de531c7dd9cd1b97b450d6a91ec8133b5f276aec04ada0c22e.txt`

```c
L135: int64_t start = (n * 4) / 10;
```

### `gpu-llr-focus40-qwen38-triton-skills-clean` -- 24.59x

blob `14/14fd02c710d41c1e6c51d97b701b8cc970128b264ccb922d1802f22f73513663.txt`

```c
L150: p1s = int(N * 0.50) - 1
```
```c
L153: p1e = int(N * 0.60) + 1
```

### `cpf-llr-focus40-qwen38-c-skills-clean` -- 21.37x

blob `c7/c70789f7a1a41e8a50890032afa0e035f625516bb0de1d1fceda90d1173c4466.txt`

```c
L17: * size-scaled index inside [0.4*n, 0.7*n) (the generator fills every other
```
```c
L68: const int64_t lo = (2 * n) / 5;   /* floor(0.4 n) */
```

### `cpf-llr-focus40-qwen38-c` -- 18.04x

blob `64/643208264774ee86c1f56af9d1e278b2233d51078247d5c893884f86a5bc5be6.txt`

```c
L69: int64_t lo = (n * 2) / 5 - M;              /* 0.4 n  (band lo minus margin) */
```
```c
L70: int64_t hi = (n * 7) / 10 + M;             /* 0.7 n  (band hi plus margin) */
```

### `cpf-llr-focus40-qwen38-c-cpfsrc-clean` -- 15.17x

blob `ae/ae340679ade23643728b07e6d7f5e1d1318cd9f9960190a00a22e4e2012c7c93.txt`

```c
L90: int64_t lo = (n * 2) / 5;
```
```c
L91: int64_t hi = (n * 7) / 10 + ((n * 7) % 10 != 0);
```

### `cpf-llr-focus40-qwen38-c-clean` -- 14.87x

blob `44/44189f5a9d156ec304c1b4d2af6461cf6ef10d7a91e11158c408232378719933.txt`

```c
L42: const int64_t blo = (n * 2) / 5;
```
```c
L43: const int64_t bhi = (n * 7) / 10;
```

### `gpu-llr-focus40-qwen38-triton` -- 5.93x

blob `c5/c5f8ecb4e460a367a71c9cdd0d430c91fe0d8707d90bddd38a6450f25f238032.txt`

```c
L5: # K, uniformly in the band [0.4*N, 0.7*N). So the first (== only) crossing lies
```

## Cleared

The other 37 setups carry no band constant and launch no restricted scan. Two were checked by hand
because they compute a fixed fraction with no decimal literal: `qwen38-hip-clean` (`NW / 2`) and
`qwen38-hip-skills-clean` (`warp >> 1`). Both are shuffle-reduction strides over a warp, not
sub-ranges of the array. All 47 resolved on source; none is undecided.

## Why the decision is not the speed-up

| setup | speed-up | verdict |
|---|---|---|
| `gpu-llr-focus40-kimi27sglang-hip-clean` | 177.62x | clean |
| `gpu-llr-focus40-qwen38-triton` | 5.93x | band-aware |

A cut anywhere between these two keeps a band-aware setup and voids an honest one. On the GPU the
kernel is bandwidth-bound, so an honest answer reaches a ratio large enough to cover what the band
saves, and the separation a threshold needs is absent.

## Per model, against how many setups carry the kernel

| model | setups carrying it | band-aware |
|---|---|---|
| GPT-OSS-120B | 21 | 0 |
| Qwen3.8-27B | 15 | 9 |
| Kimi-K2.7-Code | 10 | 1 |
| GLM-5.3 | 1 | 0 |

The model with the most exposure has none of the hits. Packet independence is the firmer half and
needs no rate: the best version in the corpus is the NO-PACKET `cpf-llr-focus40-qwen38-c` at
18.04x, against 14.87-21.37x for the packet-bearing setups of the same model and language.

## Colliding speed-up values: NOT bounded, and it reaches credited rows

An earlier version of this file said the value `1007.7545761573364` appeared on 3 rows of one arm
and was the only such repeat. Both halves were wrong; the scan behind them missed most of the
corpus. Corrected below.

**The sentinel.** `1007.7545761573364` appears on **10 rows**, 5 arms, 2 models, 4 kernels
(`tsvc_2_s316`, `tsvc_2_s1232`, `versioned_distance_update`, `ext_break_capture`) and 2 experiments,
with `native_ns` from 14,590 to 131,041 and `baseline_ns` from 78M to 479M. No two agree. The value
cannot be a measurement of any of them. Every one of the 10 carries `suspect = 1`, so these are
credited 1x and reach nothing.

**The wider problem, which is not flagged.** Across 5,899 graded rows in 1,320 judge databases there
are **101 speed-up values shared by more than one (arm, kernel)** at full 17-digit precision,
carrying **329 rows, of which 326 are credited** (`suspect = 0`). The collisions cross models,
languages and kernels: `3.6821371900248834` sits on `git-scicomp-qwen38-kernel/dfa`,
`llrblind-oss120b-c/versioned_distance_update` and `llrblind-oss120b-fortran/tsvc_2_s152`. No
colliding group shares `(native_ns, baseline_ns)`, so each row has its own timings and only the
derived speed-up collides.

Exposure is not uniform, and it falls on exactly two experiment families:

| family | graded rows | on a colliding value | share |
|---|---|---|---|
| `llrblind` | 888 | 255 | **28.7%** |
| `git-scicomp` | 390 | 50 | **12.8%** |
| `gpu-llr-focus40` | 2114 | 21 | 1.0% |
| `cpf-llr-focus40` | 1775 | 0 | 0.0% |

`cpf-llr-focus40` is clean, so the CPU language-packet panel does not depend on this. `llrblind`
and `git-scicomp` are the no-score-tool and repository-formulation experiments. Their reported
numbers should not be quoted until the cause is found. Root cause is not established here; it is
with the session auditing the scoring path, alongside the finding that 69% of submission rows sit
in a `(run_id, kernel)` group with more than one row.

Reproduce: `collide_check.py` in this directory.
