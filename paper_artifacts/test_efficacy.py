"""The properties Q is chosen for, checked against the real llr9 table.

Every assertion is a reason the metric is defined the way it is, not a restatement of a number:
identity at no effect, exact antisymmetry under swapping the arms, the ceiling cancelling on a
shared roster, completion counted once rather than twice, and the tail behaviour that made ln
rather than a plain relative delta the published scale.

Usage:  python3 -m pytest test_efficacy.py
"""

from __future__ import annotations

import math
import pathlib

import pytest

import efficacy

DATA = pathlib.Path(__file__).resolve().parent / "experiments/llr9/data/kernels.csv"


def legs():
    if not DATA.exists():
        pytest.skip("llr9 kernels.csv is not in this clone; run collect_llr9.py on the cluster")
    return efficacy.load(DATA)


def cell(solved=1, score=2.0, tokens=100):
    return (solved, score, tokens)


def test_no_effect_is_zero():
    same = {"k1": {"0": cell(), "1": cell()}, "k2": {"0": cell(1, 4.0), "1": cell(1, 4.0)}}
    assert efficacy.efficacy(same)[0] == pytest.approx(0.0)


def test_swapping_the_arms_negates_q():
    """An intervention cannot be worth more applied than removed. Fails for a plain (rho - 1)."""
    fwd = {"k1": {"0": cell(1, 2.0, 200), "1": cell(1, 8.0, 50)}}
    rev = {k: {"0": v["1"], "1": v["0"]} for k, v in fwd.items()}
    assert efficacy.efficacy(fwd)[0] == pytest.approx(-efficacy.efficacy(rev)[0])


def test_ceiling_cancels_on_a_shared_roster():
    """C_i only ever corrects a DIFFERING solve set; it must not move a paired comparison."""
    cells = legs()[("qwen38", "c")]
    paired = efficacy.paired(legs(), ("qwen38", "c"))
    assert all(v["0"][0] == v["1"][0] for v in paired.values()), "leg is no longer fully paired"
    ceil = efficacy.ceilings(legs())
    assert efficacy.gains(paired)[1] == pytest.approx(efficacy.gains(paired, ceil)[1])
    assert cells


def test_score_axis_ignores_unsolved_so_completion_is_not_double_counted():
    """An unsolved task scores 1.0 by convention; if it reached G, rho_c would be counted twice."""
    with_miss = {"k1": {"0": cell(), "1": cell()}, "k2": {"0": cell(), "1": cell(0, 1.0)}}
    without = {"k1": {"0": cell(), "1": cell()}}
    assert efficacy.gains(with_miss)[1] == pytest.approx(efficacy.gains(without)[1])
    assert efficacy.gains(with_miss)[0] < 1.0, "the miss must still show up on the completion axis"


def test_completion_ratio_survives_a_zero_and_a_clean_sweep():
    none = {"k1": {"0": cell(0, 1.0), "1": cell(0, 1.0)}}
    swept = {"k1": {"0": cell(), "1": cell()}}
    assert math.isfinite(efficacy.gains(none)[0]) and math.isfinite(efficacy.gains(swept)[0])


def test_ln_damps_an_outlier_that_a_relative_delta_would_let_win():
    """One 10x gain against nine 10x losses is a net loss; sum(rho - 1) calls it a win."""
    ratios = [10.0] + [0.1] * 9
    assert sum(math.log(r) for r in ratios) < 0
    assert sum(r - 1 for r in ratios) > 0


def test_g_is_symmetric_where_the_relative_delta_is_not():
    assert efficacy.g(2.0) == pytest.approx(-efficacy.g(0.5))
    assert efficacy.g(1.0) == pytest.approx(0.0)
    assert pytest.approx(-(0.5 - 1)) != (2.0 - 1)


def test_published_suite_effect_is_indistinguishable_from_none():
    """The llr9 finding: five of six legs have a bootstrap interval covering zero."""
    ls = legs()
    ceil = efficacy.ceilings(ls)
    covering = 0
    for key in ls:
        paired = efficacy.paired(ls, key)
        if not paired:
            continue
        lo, hi = efficacy.bootstrap(paired, n=400, ceil=ceil)
        covering += lo <= 0.0 <= hi
    assert covering >= 5
