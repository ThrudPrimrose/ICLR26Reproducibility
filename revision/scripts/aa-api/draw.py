"""A/A figure over four kernels through the HPCAgent-Bench plotting API (kernel_runs_figure).

Builds the `runs` frame that population.designed_runs + kernel_runs.run_costs produce from two old sources:
temperature3.db (results DB, user_version 6) and the repeat3-ml gelu CSV. Run:
    PYTHONPATH=$BENCH:$BENCH/agent python3 draw.py
"""

import math
import os
import pathlib
import sqlite3

import numpy as np
import pandas as pd

from hpcagent_bench.stats import cost, population
from hpcagent_bench.stats.figures import efficacy, kernel_runs, per_kernel

DATA = pathlib.Path(os.environ.get("AA_DATA", pathlib.Path(__file__).resolve().parents[2] / "data"))
OUT = pathlib.Path(os.environ.get("AA_OUT", "aa_four_kernels_api.pdf"))
KERNELS = ["scan_affine_decay", "kmp", "addusxx_g", "gelu"]
CONDITIONS = ("t0", "")  # same order as --conditions t0,default
#: Wider than the paper's square one-column 3.33 in: at 3.33 in the eight kernel names overlap.
WIDTH_IN = 4.3
#: The paper wants the four kernels wide and short: draw the text-wide (non-square) layout.
WIDE = False
#: gelu's CSV carries only the effective token total (no fresh/cached/output components), so every kernel
#: is drawn at the effective card; temperature3 tokens are effective_tokens + crashed_effective_tokens.
COST_MODEL = "effective"
COLUMNS = ["setup", "kernel", "run_root", "job", "episode_id", population.SLOT_COLUMN, "speedup", "tokens"]


def temperature3_runs() -> pd.DataFrame:
    """Per (setup, kernel, slot) the latest rerun; its best correct final/regrade speedup on its own kernel."""
    with sqlite3.connect(DATA / "temperature3.db") as db:
        episodes = pd.read_sql(
            "SELECT id, setup, job, slot, kernel, final_attempt_start_ms, "
            "effective_tokens + crashed_effective_tokens AS tokens FROM episodes "
            "WHERE setup IN ('temperature3-qwen38-c', 'temperature3-qwen38-c-t0', "
            "'temperature3-oss120b-c', 'temperature3-oss120b-c-t0')",
            db,
        )
        best = pd.read_sql(
            "SELECT g.episode_id AS id, MAX(g.speedup) AS speedup FROM grades g JOIN episodes e ON e.id = g.episode_id "
            "WHERE g.kind IN ('final', 'regrade') AND g.correct = 1 AND g.kernel = e.kernel GROUP BY g.episode_id",
            db,
        )
    episodes = episodes.loc[episodes["kernel"].isin(KERNELS)]
    latest = episodes.sort_values(["final_attempt_start_ms", "id"]).drop_duplicates(
        ["setup", "kernel", "slot"], keep="last"
    )
    runs = latest.merge(best, on="id", how="left")
    return runs.assign(run_root="temperature3", job=runs["job"].fillna(0).astype(int), episode_id=runs["id"].astype(str))


def gelu_runs() -> pd.DataFrame:
    """Per (setup, slot) of the gelu repeat: the best correct submission/attempt speedup over the slot's
    episodes (relaunches got new ids under the same .s<slot> suffix), tokens from the slot's episode row."""
    names = ["setup", "kernel", "episode_id", "row_kind", "status", "correct", "speedup", "tokens", "result", "final"]
    rows = pd.read_csv(DATA / "repeat3-ml-obs-1130.csv", header=None, names=names)
    setups = [f"rev-repeat3-ml-feedback-{model}-hip{suffix}" for model in ("qwen38", "oss120b") for suffix in ("", "-t0")]
    rows = rows.loc[rows["setup"].isin(setups) & (rows["kernel"] == "gelu")]
    rows = rows.assign(slot=rows["episode_id"].str.extract(r"\.s(\d+)$")[0].astype(int))
    answers = rows.loc[rows["row_kind"].isin(["submission", "attempt"]) & (rows["correct"] == 1)]
    best = answers.groupby(["setup", "slot"], as_index=False)["speedup"].max()
    # every slot any row names; tokens only from an episode row (NaN where the slot has none)
    slots = rows.groupby(["setup", "slot"], as_index=False).agg(episode_id=("episode_id", "last"))
    spent = rows.loc[(rows["row_kind"] == "episode") & rows["tokens"].notna()]
    tokens = spent.groupby(["setup", "slot"], as_index=False)["tokens"].max()
    runs = slots.merge(tokens, on=["setup", "slot"], how="left").merge(best, on=["setup", "slot"], how="left")
    return runs.assign(kernel="gelu", run_root="repeat3-ml", job=0)


def main() -> None:
    runs = pd.concat([temperature3_runs().loc[:, COLUMNS], gelu_runs().loc[:, COLUMNS]], ignore_index=True)
    solved = runs["speedup"].notna() & (runs["speedup"] > 0)
    runs[population.RUN_STATE_COLUMN] = [
        population.RunState.SOLVED if ok else population.RunState.UNSOLVED for ok in solved
    ]
    runs["speedup"] = runs["speedup"].where(solved, population.NOT_DELIVERED)
    runs = runs.sort_values(["setup", "kernel", population.SLOT_COLUMN, "run_root", "job"], ignore_index=True)
    card = cost.resolve(COST_MODEL, None)
    # Short kernel names so four kernels fit the side-by-side width without overlapping ticks.
    short = {"scan_affine_decay": "scan", "kmp": "kmp", "addusxx_g": "addusxx", "gelu": "gelu"}
    runs["kernel"] = runs["kernel"].map(short)
    if WIDE:
        kernel_runs.SQUARE_GROUPS = 1  # the bench's own switch between its square and text-wide layouts
    fig = kernel_runs.kernel_runs_figure(
        runs, [short[k] for k in KERNELS], "", WIDTH_IN, False, efficacy.cost_axis_label(card), kernel_runs.Mode.BOX, CONDITIONS
    )
    per_kernel.save(fig, OUT, print_size=True)
    states = runs[population.RUN_STATE_COLUMN].map(lambda state: state.text)
    drawn = runs.assign(condition=runs["setup"].map(kernel_runs.condition_of), **{population.RUN_STATE_COLUMN: states})
    drawn.to_csv(OUT.with_suffix(".csv"), index=False)
    for (kernel, setup), cell in drawn.groupby(["kernel", "setup"]):
        hits = cell.loc[cell[population.RUN_STATE_COLUMN] == population.RunState.SOLVED.text, "speedup"]
        sd = float(np.std(np.log(hits), ddof=1)) if len(hits) > 1 else math.nan
        print(f"{kernel:18s} {setup:40s} solved {len(hits)}/{len(cell)} sd_ln {sd:.3f} "
              f"range {hits.min():.2f}-{hits.max():.2f}")


if __name__ == "__main__":
    main()
