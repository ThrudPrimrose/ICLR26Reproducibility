"""Consumers of the llr9 construction rule: which kernel comes from which campaign, and why.

llr9 is the experiment over the ``llr-focus40`` tag as the benchmark repository holds it: llr8's
waves for the kernels still in the tag, llr40v9 for the six that were re-measured, and nothing for
the six the re-cut removed. That rule is a claim about the DATA, not about the code, so these read
the collected tables -- a refreshed kernel that survived the filter, or an inherited kernel that
quietly went missing, is exactly the failure a passing unit test on the filter function would not
catch.

THE ROSTER IS CHECKED AGAINST THE TAG, not against a number. A hardcoded 40 passes for a table
holding the wrong forty kernels, which is the failure that matters here: the tag was re-cut once
already and the whole point of llr9 is to follow it. The tag lives in the benchmark repository,
which a clone of this artifact does not have, so that check skips when the repository is not beside
it and the weaker size check stands in.

Usage:  python3 -m pytest test_llr9_filter.py
"""
from __future__ import annotations

import csv
import functools
import pathlib

import pytest

import aggregate_llr9
import collect_llr9
from benchlib import kernels

DATA = pathlib.Path(__file__).resolve().parent / "data"

#: Where the benchmark manifests sit when the two repositories are checked out side by side. The
#: tag is read from the manifests rather than copied here: a copy is a second roster to keep in step
#: with the first, and it would go stale exactly when the tag moves again.
BENCHMARKS = (pathlib.Path(__file__).resolve().parents[4] / "optarena" / "hpcagent_bench" / "benchmarks" /
              "loop_level_reasoning")

TAG = "llr-focus40"


@functools.lru_cache(typed=True)
def tagged() -> frozenset[str]:
    """Kernels carrying the ``llr-focus40`` tag in the benchmark repository beside this one."""
    return frozenset(m.parent.name for m in sorted(BENCHMARKS.glob("*/*.yaml"))
                     if m.stem == m.parent.name and TAG in m.read_text(errors="replace"))


def rows(wave: pathlib.Path, name: str) -> list[dict[str, str]]:
    with (wave / name).open() as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(name="waves", scope="module")
def waves_fixture() -> list[pathlib.Path]:
    found = kernels.wave_dirs(DATA) if DATA.is_dir() else []
    if not found:
        pytest.skip(f"no collected waves under {DATA}; run collect_llr9.py")
    return found


def test_the_inherited_waves_carry_no_dropped_kernel(waves: list[pathlib.Path]) -> None:
    dropped = collect_llr9.REFRESHED | collect_llr9.DUPLICATE | collect_llr9.REPLACED
    for wave in waves:
        if wave.name == "v9":
            continue
        seen = {row["benchmark"] for row in rows(wave, "calls.csv")}
        assert not seen & dropped, wave.name


def test_the_fresh_wave_carries_only_refreshed_kernels(waves: list[pathlib.Path]) -> None:
    fresh = [w for w in waves if w.name == "v9"]
    if not fresh:
        pytest.skip("the llr40v9 wave has not been collected")
    assert {row["benchmark"] for row in rows(fresh[0], "calls.csv")} <= collect_llr9.REFRESHED


def test_the_duplicate_kernel_is_dropped_and_not_folded(waves: list[pathlib.Path]) -> None:
    # tsvc_2_s13110 is byte-identical to tsvc_2_s3110 in the benchmark repository. Folding the two
    # would report one agent's two independent draws as one attempt at one kernel; s3110 keeps
    # exactly its own rows.
    table = list(csv.DictReader((DATA / "kernels.csv").open())) if (DATA / "kernels.csv").is_file() else []
    if not table:
        pytest.skip("run aggregate_llr9.py")
    assert not [r for r in table if r["benchmark"] in collect_llr9.DUPLICATE]
    assert [r for r in table if r["benchmark"] == "tsvc_2_s3110"]


@functools.lru_cache(typed=True)
def drawn() -> frozenset[str]:
    """The kernels the pooled table holds. Cached: three tests read it and it is a file scan."""
    path = DATA / "kernels.csv"
    if not path.is_file():
        pytest.skip("run aggregate_llr9.py")
    with path.open() as handle:
        return frozenset(row["benchmark"] for row in csv.DictReader(handle))


def test_the_table_holds_the_tag_less_what_cannot_be_scored() -> None:
    # The size check that works from a clone: the tag counts tsvc_2_s2233 and the table cannot, so
    # the table is one short of TAG_SIZE and never anything else.
    assert drawn() == drawn() - aggregate_llr9.EXCLUDED
    assert len(drawn()) == aggregate_llr9.TAG_SIZE - len(aggregate_llr9.EXCLUDED)


def test_the_roster_is_the_tag_the_benchmark_repository_holds() -> None:
    if not BENCHMARKS.is_dir():
        pytest.skip(f"no benchmark repository at {BENCHMARKS}")
    tag = tagged()
    assert len(tag) == aggregate_llr9.TAG_SIZE
    assert drawn() == tag - aggregate_llr9.EXCLUDED


def test_every_dropped_kernel_is_one_the_tag_no_longer_carries() -> None:
    if not BENCHMARKS.is_dir():
        pytest.skip(f"no benchmark repository at {BENCHMARKS}")
    dropped = collect_llr9.DUPLICATE | collect_llr9.REPLACED
    assert not dropped & tagged()
    # ...and the refreshed six ARE still in it: they were re-measured, not retired.
    assert collect_llr9.REFRESHED.issubset(tagged())
