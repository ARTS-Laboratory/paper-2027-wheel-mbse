"""The gene box, and the two setters that move it.

`MIN_WALL_MM` and `CY_BOUND_MM` are consumed by the `GENE_SPACE` literal at IMPORT time,
so both are settable only through a function that also re-snapshots the three module
arrays derived from it.  Everything here is solve-free and checked at full precision.

THESE TESTS MUTATE MODULE STATE, which is why `restore_box` is autouse rather than
opt-in.  `tests/test_stage3.py` takes its bounds in a MODULE-scoped fixture, so a floor
left behind by a test here would not merely leak — it would be baked into a fixture that
never recomputes, and the failure would surface somewhere else entirely.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import study_stage3 as so3  # noqa: E402
import wheel_fea as W       # noqa: E402
import wheel_genome as wg   # noqa: E402
import wheel_stage3 as S3   # noqa: E402


def so3_genes():
    return np.array(so3.load_genes(), dtype=float)

THICKNESS = (8, 9, 10, 11)          # t0, t1, t2, t3
CENTERLINE = (0, 1, 2, 3, 4, 5, 6, 7)
FILLETS = (12, 13)


@pytest.fixture(autouse=True)
def restore_box():
    """Put the box back exactly as imported, however the test left it."""
    saved = [(g["low"], g["high"]) for g in W.GENE_SPACE]
    min_wall, cy_bound = W.MIN_WALL_MM, W.CY_BOUND_MM
    yield
    for g, (lo, hi) in zip(W.GENE_SPACE, saved):
        g["low"], g["high"] = lo, hi
    W.MIN_WALL_MM, W.CY_BOUND_MM = min_wall, cy_bound
    W._refresh_gene_arrays()


# ---------------------------------------------------------------------------
# WHAT THE FLOOR MOVES, AND WHAT IT MUST NOT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("floor", [1.6, 1.8, 2.0, 2.2])
def test_the_floor_moves_the_four_thickness_lows_and_nothing_else(floor):
    before = [(g["low"], g["high"]) for g in W.GENE_SPACE]
    W.set_min_wall(floor)

    assert W.MIN_WALL_MM == floor
    for idx in THICKNESS:
        assert W.GENE_SPACE[idx]["low"] == floor
        assert W.GENE_SPACE[idx]["high"] == before[idx][1], "a ceiling moved"
    for idx in CENTERLINE + FILLETS:
        assert (W.GENE_SPACE[idx]["low"], W.GENE_SPACE[idx]["high"]) == before[idx], \
            f"gene {idx} moved and only t0..t3 should have"


@pytest.mark.parametrize("floor", [1.6, 2.2])
def test_the_floor_rebuilds_the_snapshot_arrays(floor):
    """The failure mode `_refresh_gene_arrays` exists to prevent: `GENE_SPACE` edited but
    `_GENE_LOW`/`_GENE_HIGH`/`_GENE_RANGE` stale, so `adaptive_gaussian_mutation` keeps
    clipping offspring to the bound it was told to drop and the GA silently ignores the
    setting.  A dict-only setter passes every other test in this file.
    """
    W.set_min_wall(floor)
    assert np.all(W._GENE_LOW[list(THICKNESS)] == floor)
    np.testing.assert_array_equal(
        W._GENE_RANGE, W._GENE_HIGH - W._GENE_LOW)
    np.testing.assert_array_equal(
        W._GENE_LOW, np.array([g["low"] for g in W.GENE_SPACE]))


def test_the_cy_bound_setter_still_rebuilds_them_after_the_refactor():
    """`set_cy_bound` grew the shared tail; it had this property before and must keep it."""
    W.set_cy_bound(45.0)
    assert W.CY_BOUND_MM == 45.0
    np.testing.assert_array_equal(W._GENE_LOW[[1, 3, 5, 7]], np.full(4, -45.0))
    np.testing.assert_array_equal(W._GENE_HIGH[[1, 3, 5, 7]], np.full(4, 45.0))
    np.testing.assert_array_equal(W._GENE_RANGE, W._GENE_HIGH - W._GENE_LOW)


# ---------------------------------------------------------------------------
# WHAT IT REFUSES
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, -1.0, 6.0, 8.0])
def test_a_floor_that_empties_or_inverts_the_box_is_refused(bad):
    """`t3`'s ceiling is the tightest at 6.0, so a floor there or above leaves that gene
    with nowhere to live.  Refused at the setter rather than discovered as a NaN 90
    minutes into a descent."""
    # Read the floor BEFORE the refused call rather than hardcoding it: what this asserts
    # is "unchanged", and a literal turns that into "still 2.0", which is a different
    # claim that stopped being true when PLAN.md §13 moved the default to 1.2.
    before = W.MIN_WALL_MM
    with pytest.raises(ValueError, match="min wall"):
        W.set_min_wall(bad)
    assert W.MIN_WALL_MM == before, "a refused floor must not half-apply"


# ---------------------------------------------------------------------------
# WHAT THE DESCENT DOES WITH IT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("floor", [1.6, 1.8, 2.0, 2.2])
def test_normalize_denormalize_round_trips_against_the_moved_floor(floor):
    W.set_min_wall(floor)
    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    genes = 0.5 * (low + high)
    # 1 ulp, not 0: the round trip is `(x-lo)/(hi-lo)` then `lo + z*(hi-lo)`, which is not
    # bit-exact in floating point and is not required to be.
    np.testing.assert_allclose(
        wg.denormalize(wg.normalize(genes, low, high), low, high), genes, rtol=1e-15)


@pytest.mark.parametrize("floor", [1.6, 2.2])
def test_the_normalized_gradient_follows_the_moved_floor(floor):
    """`objective(normalized=True)` scales by `(high - low)`, and moving the floor moves
    that factor on t0..t3 only.  A FINITE DIFFERENCE rather than the chain-rule identity
    `test_objective.py` already checks: that one would agree with itself even if the box
    the gradient was scaled by were the stale one.

    T1 only, so this costs no mesh and no solve.  It needs a genome where `hub_overlap` is
    VIOLATED, because within T1 that barrier is the only live `d/dt0` there is — `mass` is
    integrated over the mesh and so lives in a later tier, and `fillet_cap` is slack
    whenever `R_hub` is under its cap.  A flat barrier makes the assertion `0 == 0` and
    proves nothing, which is what the guard below exists to catch.

    IT READS `best_solution_ga_beam.json`, NOT `best_solution.json`, FOR THAT REASON.  This
    used to take whatever genome shipped, on the strength of the shipped one violating
    `hub_overlap` by +0.323 mm — an incidental property of one design, not a fact about the
    box.  §13's promotion made the shipped genome feasible on `hub_overlap` and this test
    went vacuous; the guard caught it, which is the guard working.  The GA/beam reference
    is pinned under its own name and never moves (see `tests/test_golden.py`), so pointing
    at it decouples a test ABOUT THE BOX from the question of which design ships.
    """
    import wheel_objective as WO

    W.set_min_wall(floor)
    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    genes = np.array(so3.load_genes("best_solution_ga_beam.json"), dtype=float)
    genes[list(THICKNESS)] = np.clip(genes[list(THICKNESS)],
                                     low[list(THICKNESS)], high[list(THICKNESS)])
    z = wg.normalize(genes, low, high)

    _, gz, _ = WO.objective(z, "smoke", tiers=("t1",), normalized=True)
    assert abs(gz[8]) > 1e-6, "t0's barrier gradient is flat; the test would be vacuous"

    h = 1e-6
    zp, zm = z.copy(), z.copy()
    zp[8] += h
    zm[8] -= h
    vp, _, _ = WO.objective(zp, "smoke", tiers=("t1",), normalized=True)
    vm, _, _ = WO.objective(zm, "smoke", tiers=("t1",), normalized=True)
    np.testing.assert_allclose((vp - vm) / (2 * h), gz[8], rtol=1e-5)


def test_a_start_below_a_raised_floor_is_projected_up_onto_it():
    """The 2.2 mm arm of the sweep starts from a genome whose four thicknesses are 2.0 —
    BELOW its own floor — so `normalize` puts them at a negative `z` and `descend`'s
    `z = project(z0)` is the only thing standing between that and a descent that begins
    outside its own box.  Asserted rather than discovered mid-run.
    """
    low0, high0, _ = wg.bounds_arrays(W.GENE_SPACE)
    genes = 0.5 * (low0 + high0)
    genes[list(THICKNESS)] = 2.0                 # the elite-10 answer's thicknesses

    W.set_min_wall(2.2)
    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)

    z_raw = wg.normalize(genes, low, high)
    assert np.all(z_raw[list(THICKNESS)] < 0.0), "the start must be outside the new box"

    lifted = wg.denormalize(S3.project(z_raw), low, high)
    np.testing.assert_allclose(lifted[list(THICKNESS)], 2.2, rtol=0, atol=1e-12)
    # and the genes that were already inside are untouched by the projection
    np.testing.assert_allclose(lifted[list(CENTERLINE)], genes[list(CENTERLINE)],
                               rtol=0, atol=1e-12)
