"""Consumers of the llr9 construction rule: which kernel comes from which campaign, and why.

llr9 is llr8's waves for every kernel they still speak for, plus llr40v9 for the six they do not.
That rule is a claim about the DATA, not about the code, so these read the collected tables: a
refreshed kernel that survived the filter, or an inherited kernel that quietly went missing, is
exactly the failure a passing unit test on the filter function would not catch.

Usage:  python3 -m pytest test_llr9_filter.py
"""
from __future__ import annotations

import csv
import pathlib

import pytest

import aggregate_llr9
import collect_llr9
from benchlib import kernels

DATA = pathlib.Path(__file__).resolve().parent / "data"


def rows(wave: pathlib.Path, name: str) -> list[dict[str, str]]:
    with (wave / name).open() as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(name="waves", scope="module")
def waves_fixture() -> list[pathlib.Path]:
    found = kernels.wave_dirs(DATA) if DATA.is_dir() else []
    if not found:
        pytest.skip(f"no collected waves under {DATA}; run collect_llr9.py")
    return found


def test_the_inherited_waves_carry_no_refreshed_kernel(waves: list[pathlib.Path]) -> None:
    for wave in waves:
        if wave.name == "v9":
            continue
        seen = {row["benchmark"] for row in rows(wave, "calls.csv")}
        assert not seen & (collect_llr9.REFRESHED | collect_llr9.DUPLICATE), wave.name


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


def test_the_tag_size_is_the_kernel_set_the_table_holds() -> None:
    path = DATA / "kernels.csv"
    if not path.is_file():
        pytest.skip("run aggregate_llr9.py")
    with path.open() as handle:
        drawn = {row["benchmark"] for row in csv.DictReader(handle)}
    assert len(drawn) == aggregate_llr9.TAG_SIZE
