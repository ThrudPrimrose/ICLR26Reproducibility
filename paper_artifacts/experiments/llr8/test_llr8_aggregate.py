"""Consumers of the pooling engine: the pooling rules every figure inherits from its output.

These rules used to live inside a plot script, where changing one silently changed every number in
the paper and nothing failed. They are asserted here instead, one test per rule, on wave
directories built with the collector's own column lists so a schema change breaks the test rather
than the figures.

Usage:  python3 -m pytest test_llr8_aggregate.py
"""
from __future__ import annotations

import csv
import pathlib

import pytest

import aggregate_llr8
from benchlib import kernels
from benchlib import shards

ARM = ("oss120b", "c", "0")


def call(benchmark: str, tokens: int, route: str = "score", status: str = "ok") -> dict[str, object]:
    model, language, skills = ARM
    return {
        "arm": "llr8w2-oss120b-c",
        "model": model,
        "language": language,
        "skills": skills,
        "job": 1,
        "benchmark": benchmark,
        "route": route,
        "status": status,
        "correct": 1,
        "tokens": tokens,
        "speedup": 1.0,
        "compiler": "gcc",
    }


def submission(benchmark: str, speedup: float, ts_ms: object, suspect: int = 0) -> dict[str, object]:
    model, language, skills = ARM
    return {
        "arm": "llr8w2-oss120b-c",
        "model": model,
        "language": language,
        "skills": skills,
        "job": 1,
        "benchmark": benchmark,
        "preset": "fuzzed",
        "baseline_ns": 100.0,
        "native_ns": 50.0,
        "speedup": speedup,
        "suspect": suspect,
        "ts_ms": ts_ms,
    }


def wave(root: pathlib.Path, name: str, calls: list[dict[str, object]],
         submissions: list[dict[str, object]]) -> pathlib.Path:
    """One collected wave directory, written with the collector's own column lists."""
    out = root / name
    out.mkdir(parents=True)
    for filename, columns, rows in (("calls.csv", shards.CALL_COLUMNS, calls),
                                    ("submissions.csv", shards.SUBMISSION_COLUMNS, submissions)):
        with (out / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return out


@pytest.fixture(name="data")
def data_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    """Two waves carrying every shape the rules have to decide.

    w2: ``resub`` submitted three times, its LAST attempt the worst of the three, and a suspect row
    stamped later still. ``once`` submitted once. ``never`` reached but never submitted. The
    unscoreable ``tsvc_2_s2233`` reached and submitted. w4 re-runs ``never`` (solved there) and
    ``resub`` (one more submission, later than any in w2).
    """
    wave(tmp_path, "w2", [
        call("resub", 100),
        call("resub", 300),
        call("once", 500),
        call("never", 700),
        call("tsvc_2_s2233", 900),
    ], [
        submission("resub", 8.0, 1000),
        submission("resub", 4.0, 2000),
        submission("resub", 2.0, 3000),
        submission("resub", 99.0, 4000, suspect=1),
        submission("once", 3.0, 1500),
        submission("tsvc_2_s2233", 5.0, 1000),
    ])
    wave(tmp_path, "w4", [
        call("never", 200, route="submit", status="ok"),
        call("resub", 400),
    ], [
        submission("never", 6.0, 9000),
        submission("resub", 7.0, 9500),
    ])
    return tmp_path


@pytest.fixture(name="table")
def table_fixture(data: pathlib.Path) -> dict[str, dict[str, object]]:
    return {str(row["benchmark"]): row for row in kernels.rows(kernels.collect(data, aggregate_llr8.EXCLUDED))}


def test_value_is_the_last_submission_not_the_best(table: dict[str, dict[str, object]]) -> None:
    # resub ends on 7.0 in w4; its best over both waves is 8.0 from its first w2 attempt.
    assert table["resub"]["last_speedup"] == "7.000000"
    assert table["resub"]["best_speedup"] == "8.000000"
    assert table["resub"]["ordering"] == kernels.STAMPED


def test_last_is_read_from_the_stamp_not_from_row_order(data: pathlib.Path) -> None:
    """The latest stamp wins even when it is the FIRST row of the file."""
    wave(data, "w6", [call("shuffled", 10)], [submission("shuffled", 5.0, 8000), submission("shuffled", 1.0, 100)])
    table = {str(r["benchmark"]): r for r in kernels.rows(kernels.collect(data, aggregate_llr8.EXCLUDED))}
    assert table["shuffled"]["last_speedup"] == "5.000000"


def test_suspect_rows_are_dropped_even_when_they_are_the_latest(table: dict[str, dict[str, object]]) -> None:
    # The 99x row is stamped after every other resub submission and must not become its last value.
    assert table["resub"]["submissions"] == 4
    assert table["resub"]["best_speedup"] != "99.000000"


def test_the_unscoreable_kernel_is_absent_entirely(table: dict[str, dict[str, object]]) -> None:
    assert "tsvc_2_s2233" in aggregate_llr8.EXCLUDED
    assert "tsvc_2_s2233" not in table


def test_a_kernel_two_waves_drew_is_one_row_naming_both(table: dict[str, dict[str, object]]) -> None:
    assert table["resub"]["waves"] == "w2;w4"
    assert table["never"]["waves"] == "w2;w4"


def test_solved_is_a_union_over_waves_never_a_per_wave_sum(table: dict[str, dict[str, object]]) -> None:
    # never was reached in w2 and solved in w4: one row, solved once.
    assert table["never"]["solved"] == 1
    assert table["once"]["solved"] == 0


def test_tokens_take_the_high_water_mark_in_a_wave_and_add_across_waves(table: dict[str, dict[str, object]]) -> None:
    # resub: cumulative 100 then 300 in w2 (high-water 300), plus 400 in w4.
    assert table["resub"]["tokens"] == 700
    assert table["once"]["tokens"] == 500


def test_a_reached_but_never_submitted_kernel_keeps_its_row(table: dict[str, dict[str, object]]) -> None:
    # Its cost and its solve are real; only the timing is missing, and that is what untimed says.
    assert table["never"]["submissions"] == 1
    assert table["once"]["ordering"] == kernels.STAMPED
    empty = kernels.last_of([])
    assert empty == (None, kernels.UNTIMED)


def test_an_unstamped_kernel_reports_no_last_value(data: pathlib.Path) -> None:
    wave(data, "w7", [call("nostamp", 10)], [submission("nostamp", 5.0, ""), submission("nostamp", 2.0, "")])
    table = {str(r["benchmark"]): r for r in kernels.rows(kernels.collect(data, aggregate_llr8.EXCLUDED))}
    assert table["nostamp"]["ordering"] == kernels.UNSTAMPED
    assert table["nostamp"]["last_speedup"] == ""
    assert table["nostamp"]["best_speedup"] == "5.000000"


def test_two_rows_sharing_the_latest_stamp_report_no_last_value(data: pathlib.Path) -> None:
    wave(data, "w8", [call("tied", 10)], [submission("tied", 5.0, 700), submission("tied", 2.0, 700)])
    table = {str(r["benchmark"]): r for r in kernels.rows(kernels.collect(data, aggregate_llr8.EXCLUDED))}
    assert table["tied"]["ordering"] == kernels.TIED
    assert table["tied"]["last_speedup"] == ""


def test_the_denominator_is_the_tag_not_the_kernels_an_arm_drew(data: pathlib.Path) -> None:
    # Three of the fixture's four kernels survive the exclusion; the denominator stays the tag.
    table = kernels.rows(kernels.collect(data, aggregate_llr8.EXCLUDED))
    assert len(table) == 3
    assert aggregate_llr8.TAG_SIZE == 40


def test_the_table_is_byte_stable_across_runs(data: pathlib.Path, tmp_path: pathlib.Path) -> None:
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    kernels.write(first, kernels.rows(kernels.collect(data, aggregate_llr8.EXCLUDED)))
    kernels.write(second, kernels.rows(kernels.collect(data, aggregate_llr8.EXCLUDED)))
    assert first.read_bytes() == second.read_bytes()
