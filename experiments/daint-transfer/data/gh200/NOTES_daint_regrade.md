# LLR40 regrade on Daint (GH200): notes for the results

Run by Lars Hulsbergen on daint.alps (account g34), 2026-09-24/25. The package is your second zip (`llr40-daint-regrade.zip`, 20:50). The harness is unchanged (main e5699044b). Only the setup around it changed, as described below.

## Status

| backend | graded | skipped | solved (s_bar not NULL) | state |
|---|---:|---:|---:|---|
| fortran | 72 | 0 | 67 | done |
| hip | 328 | 109 | 304 | done |
| triton | 361 (+3 rows to exclude) | 7 | 337 | done |
| c | 564 | 225 | 552 | done (debug partition, see below) |

Every task row is stamped `score_rule = s-mw4x5-v2` and `timing_reduction = mw4x5-final-v2`, with the Numba baseline re-timed on Daint. Rows are unique on `db` + `run_id` + `benchmark`. Some `run_id`s appear twice in the pack, with different judge DBs and sources.

## Files

- `results/<backend>/rank-<r>/regrade-cells-0.db`: per-cell and per-task grades for fortran, hip and triton (16 ranks).
- `results_debug/c/rank-<r>/regrade-cells-0.db`: the C grades (40 ranks, one kernel per rank). These are the C results. 8 C rows graded earlier on `normal` in the 16-rank layout were superseded and are not included.
- `results/<backend>/rank-<r>/{worklist,skipped}.jsonl`: what that rank graded and skipped at run time.
- `final_worklists/<backend>/{worklist,skipped}.jsonl`: the worklist rebuilt with the final rules below. **This is the authoritative skipped list.** Each row gives `db`, `run_id`, `benchmark` and the reason.
- `results/exclude_rows.jsonl`: DB rows graded at run time that the final rules skip. There are 3 Triton rows (AMD inline asm), which the running job graded before the rule existed. Drop them.
- `pack/added_from_first_zip.tsv`: the 174 items from your first zip that the second one lacks: 160 Triton (host-resident arms, not `-device`), 7 HIP, 7 C. None duplicates a source already in the new pack. They are graded and tagged here so you can keep them or drop them.
- `changes/`: every file changed from your package, with the originals as `*.orig`.

## Changes to the package

1. **HIP is built on HIP's CUDA backend, not with hipify.** `hipify-perl` only translates CUDA→HIP, so on HIP sources it did nothing and nvcc failed on `hip/hip_runtime.h`. HIP is now compiled with the ROCm/hip + ROCm/hipother `rocm-7.2.4` headers (`hip-nvidia/include`, `-D__HIP_PLATFORM_NVIDIA__`) via `NVCC_APPEND_FLAGS` in `regrade_rank.sh`. Report these rows as **"HIP via CUDA backend"**, not "via hipify".
2. **`bin/nvcc` wrapper** (first on PATH), which works around two nvcc/hipcc toolchain differences:
   - The harness links GPU libraries with a raw `-Wl,-rpath,/shared/lib`, which nvcc rejects. The wrapper rewrites `-Wl,<x>` to `-Xlinker=<x>`.
   - nvcc compiles a `.cpp` host unit as plain C++, while hipcc compiles it as HIP. A `.cpp` that includes HIP is built with `-x cu`.
3. **Portability rules in `scripts/cscs/daint_worklist.py`** (skipped as "not portable", never graded as wrong):
   - **C:** items that fail a `gcc -fsyntax-only` probe on aarch64 with an x86-only error (intrinsics headers, `_mm*`, `__m256`, `__builtin_ia32_*`, `__builtin_cpu_supports`, target attributes): 225. Items with x86 code behind a guard that builds on ARM are graded: 33.
   - **HIP:**
     - builds only with the AMD-only overloads of `hip_nv_compat.h` (mask-less `__shfl_*`, `atomicCAS(double)`, clang's relaxed constexpr): 37
     - built on MI300A (recorded speed-up > 0) but does not build on the CUDA backend at all (rocPRIM/hipCUB, `__builtin_nontemporal_*`, AMD inline asm, more than 48 KB static shared memory, hipcc-only implicit includes): 70
     - AMD-only code: 2

     The shim itself is never used for grading. An item that never built on MI300A either is graded, and its build failure is its grade.
   - **Triton:** `inline_asm_elementwise` with AMDGPU `=v` constraints is added to the existing AMD-only list (3 more, 7 in total).
   - **Changed, per your answers:** comments are stripped before the AMD-only match (12 HIP items only said "wavefront" in a comment), and `__HIP_PLATFORM_AMD__` no longer counts as AMD-only.
   - `skipped.jsonl` rows now carry `db` and `run_id`.
4. **First zip only:** its `srun --cpu-bind=cores` without `--cpus-per-task` ran every C grade on one core. Your rank layout in the second zip fixes this, and all rows here use 72 cores per rank.

## C on the debug partition

The `normal` partition was full (earliest start about 13:25, with 4.5 h runtime), so C ran on `debug`: the same GH200 nodes, OS and features, capped at 30 min per job. The run was 3 chained jobs (`regrade_debug.sbatch`, `run_debug_chain.sh`) on 10 nodes = 40 ranks, one kernel per rank, 08:24-09:54 on 2026-09-25. The harness resumes past graded rows, and a job cut at 30 min loses only the item in progress, which is regraded from scratch. Each rank still owns one GH200 module, and baseline and submission are timed on the same module, so the partition does not enter the ratio. As a check, the 8 C items graded on both partitions agree within ±2 % (6 of 8) and 7 % / 10 % (2 of 8), which is run-to-run noise at n = 5.

## Open points

1. The 70 HIP items that built on MI300A but not on the CUDA backend are skipped. I extended your rule for AMD-only code to them. Please confirm.
2. 14 of those 70 only need **hipCUB**, which has a CUDA backend (wraps CUB). Install it and grade them? A follow-up run resumes and grades only these.
3. Keep or drop the 174 first-zip items (`pack/added_from_first_zip.tsv`)?
4. One HIP item (`tsvc_2_s255`, kimi27sglang-hip-skills, p22) puts host and kernels in one file. The harness refuses it as `cuda` ("needs 'device_source'"), so it has no score. This comes from the harness contract, not the code.
5. No `hpcagent_bench.db` judge record was written by `regrade cells`. Only `regrade-cells-0.db` exists per rank.
