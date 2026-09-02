"""Consumers of the before-and-after figure: the pairing rule and the direction the grey lines encode.

Two properties decide whether this figure tells the truth. PAIRING: a kernel only one leg produced
must leave the comparison, in both directions, or an arm that never reached a kernel reads as an arm
that reached it and did nothing. DIRECTION: "worse" is metric-dependent -- a smaller speed-up is
worse, a larger token bill is worse -- and getting it backwards on one row would grey exactly the
wrong half of the figure while still looking plausible.

Usage:  python3 -m pytest test_llr8_before_after.py
"""
from __future__ import annotations

import pathlib

import pytest

from benchlib import dumbbell
from benchlib import beforeafter as before_after


def kernel(benchmark: str, last: str, tokens: int) -> dict[str, str]:
    return {"benchmark": benchmark, "last_speedup": last, "best_speedup": last, "tokens": str(tokens)}


@pytest.fixture(name="cell")
def cell_fixture() -> dumbbell.Cell:
    """One cell whose four kernels carry every case: up, down, inside the dead band, one-sided.

    ``rise`` doubles, ``fall`` halves, ``flat`` moves half a percent, and ``only_off`` was reached
    by the base leg alone.
    """
    off = {
        "rise": kernel("rise", "2.0", 1_000_000),
        "fall": kernel("fall", "4.0", 4_000_000),
        "flat": kernel("flat", "3.0", 3_000_000),
        "only_off": kernel("only_off", "9.0", 9_000_000),
    }
    on = {
        "rise": kernel("rise", "4.0", 2_000_000),
        "fall": kernel("fall", "2.0", 2_000_000),
        "flat": kernel("flat", "3.015", 3_000_000),
    }
    return dumbbell.Cell("oss120b", "c", {"0": off, "1": on})


def test_a_kernel_only_one_leg_reached_is_not_a_pair(cell: dumbbell.Cell) -> None:
    pairs = before_after.paired(cell, before_after.SPEEDUP)
    assert sorted(pairs) == ["fall", "flat", "rise"]


def test_the_pair_is_before_then_after(cell: dumbbell.Cell) -> None:
    assert before_after.paired(cell, before_after.SPEEDUP)["rise"] == (2.0, 4.0)


def test_worse_on_speedup_means_slower(cell: dumbbell.Cell) -> None:
    up, down, flat = before_after.tally(before_after.paired(cell, before_after.SPEEDUP), before_after.SPEEDUP)
    assert (up, down, flat) == (1, 1, 1)
    assert before_after.worse(4.0, 2.0, before_after.SPEEDUP)
    assert not before_after.worse(2.0, 4.0, before_after.SPEEDUP)


def test_worse_on_tokens_means_dearer_not_smaller(cell: dumbbell.Cell) -> None:
    # The direction is inverted for a cost: rise pays twice as much with the packet, so it is the
    # one drawn grey here, and fall -- which halves its bill -- is the improvement.
    up, down, flat = before_after.tally(before_after.paired(cell, before_after.TOKENS), before_after.TOKENS)
    assert (up, down, flat) == (1, 1, 1)
    assert before_after.worse(1.0, 2.0, before_after.TOKENS)
    assert not before_after.worse(2.0, 1.0, before_after.TOKENS)


def test_a_change_inside_the_dead_band_is_neither_direction(cell: dumbbell.Cell) -> None:
    pairs = before_after.paired(cell, before_after.SPEEDUP)
    assert not before_after.worse(*pairs["flat"], before_after.SPEEDUP)
    assert not before_after.worse(*reversed(pairs["flat"]), before_after.SPEEDUP)


def test_a_kernel_with_no_last_submission_leaves_the_speedup_panel(cell: dumbbell.Cell) -> None:
    # An unstamped or never-submitted kernel carries an empty last_speedup and no defensible pair,
    # but it keeps its token row, so the two panels of a column can honestly hold different n.
    cell.legs["1"]["rise"]["last_speedup"] = ""
    assert sorted(before_after.paired(cell, before_after.SPEEDUP)) == ["fall", "flat"]
    assert sorted(before_after.paired(cell, before_after.TOKENS)) == ["fall", "flat", "rise"]


def test_a_cell_missing_a_leg_pairs_nothing(cell: dumbbell.Cell) -> None:
    lonely = dumbbell.Cell("oss120b", "c", {"0": cell.legs["0"]})
    assert before_after.paired(lonely, before_after.SPEEDUP) == {}


def test_render_writes_both_formats(cell: dumbbell.Cell, tmp_path: pathlib.Path) -> None:
    before_after.style.apply()
    out = before_after.render([cell], tmp_path / "before_after")
    assert out.with_suffix(".pdf").is_file() and out.with_suffix(".png").is_file()
