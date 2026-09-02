"""Consumers of the git pooling rule: what a kernel row means, and what it must never do.

The three rules ``aggregate_git.py`` states in prose are checked here against the COLLECTED table,
not against a hand-built fixture. A fixture would prove the function pools; only the real table
proves the experiment's own data still satisfies the invariants the figures assume -- that every
leg has a full roster, that a speed-up on an incorrect submission never reaches a figure, and that
the pairing the plotter narrows to is not empty.

Usage:  python3 -m pytest test_git_aggregate.py
"""
from __future__ import annotations

import collections
import csv
import functools
import pathlib

import pytest

import aggregate_git

DATA = pathlib.Path(__file__).resolve().parent / "data"


@functools.lru_cache(maxsize=None, typed=True)
def cells() -> tuple[dict[str, str], ...]:
    path = DATA / "git_experiment_all.csv"
    if not path.exists():
        pytest.skip(f"{path.name} is not in this clone; run collect_git.py on the cluster")
    with path.open() as handle:
        return tuple(csv.DictReader(handle))


@functools.lru_cache(maxsize=None, typed=True)
def table() -> tuple[dict[str, str], ...]:
    return tuple(aggregate_git.rows(list(cells())))


def legs() -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = collections.defaultdict(list)
    for row in table():
        out[(row["model"], row["skills"])].append(row)
    return out


def test_every_leg_carries_the_whole_roster() -> None:
    """A kernel an arm never reached is a row saying so, never a shorter denominator."""
    for key, rows in legs().items():
        assert len(rows) == aggregate_git.TAG_SIZE, f"{key} has {len(rows)} kernels"
        assert len({r["benchmark"] for r in rows}) == aggregate_git.TAG_SIZE, f"{key} repeats a kernel"


def test_four_legs_two_models_two_framings() -> None:
    assert set(legs()) == {(m, leg) for m in ("oss120b", "qwen38") for leg in ("0", "1")}


def test_a_speedup_never_comes_from_an_incorrect_submission() -> None:
    """The one rule that would silently corrupt every figure if it broke."""
    graded = {(aggregate_git.MODEL_OF[c["model"]], aggregate_git.LEG_OF_FRAMING[c["framing"]], c["kernel"])
              for c in cells() if c["correct"] == "1" and c["speedup"]}
    for row in table():
        key = (row["model"], row["skills"], row["benchmark"])
        assert bool(row["last_speedup"]) == (key in graded), f"{key} speed-up disagrees with its verdict"
        assert row["solved"] == ("1" if key in graded else "0"), f"{key} solved disagrees with its verdict"


def test_last_is_the_highest_numbered_graded_attempt() -> None:
    """Not the best one. The rule has to be blind to value or it is cherry-picking."""
    for row in table():
        if not row["last_speedup"]:
            continue
        graded = sorted((int(c["attempt"]), float(c["speedup"])) for c in cells()
                        if aggregate_git.MODEL_OF[c["model"]] == row["model"]
                        and aggregate_git.LEG_OF_FRAMING[c["framing"]] == row["skills"]
                        and c["kernel"] == row["benchmark"] and c["correct"] == "1" and c["speedup"])
        assert float(row["last_speedup"]) == pytest.approx(graded[-1][1])
        assert float(row["best_speedup"]) == pytest.approx(max(v for _, v in graded))


def test_tokens_count_every_attempt_not_only_the_graded_ones() -> None:
    """The token figure asks what the arm SPENT, so an attempt that ran out of turns still counts."""
    for row in table():
        spent = sum(int(c["tokens"] or 0) for c in cells()
                    if aggregate_git.MODEL_OF[c["model"]] == row["model"]
                    and aggregate_git.LEG_OF_FRAMING[c["framing"]] == row["skills"]
                    and c["kernel"] == row["benchmark"])
        assert int(row["tokens"]) == spent


def test_each_model_has_a_non_empty_paired_set() -> None:
    """The per-kernel figure narrows to kernels BOTH legs timed; an empty set draws nothing."""
    solved = {key: {r["benchmark"] for r in rows if r["last_speedup"]} for key, rows in legs().items()}
    for model in ("oss120b", "qwen38"):
        assert solved[(model, "0")] & solved[(model, "1")], f"{model} has no kernel both framings solved"


def test_pooling_is_deterministic() -> None:
    assert aggregate_git.rows(list(cells())) == aggregate_git.rows(list(cells()))
