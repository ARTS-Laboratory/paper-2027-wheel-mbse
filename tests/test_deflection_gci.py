"""
Pins for `studies/study_deflection_gci.py` — the Richardson/GCI arithmetic and, above
all, WHICH MESH THE CELL SIZE IS MEASURED ON.

This file exists because that study shipped with two independent defects in code that
has no physics in it at all, and neither was caught by running it:

  1. `observed_order` used Roache's grid indices backwards.  It returned p = 1.2568 for a
     synthetic p = 1.3, and NO constant-ratio case can detect it — the swap is invisible
     when r21 == r32.  Only a non-constant ladder exposes it, so one is pinned here.

  2. `H_DEFS` drew its element and node counts from `wheel_mesh` while `WO.objective`
     solved on `wheel_wheel`.  Both modules export configs named smoke/coarse/medium/fine
     and they are DIFFERENT MESHES, so every h was the size of a cell in a mesh nobody
     solved on.  The reported refinement ratios were 1.826/1.789 against a true
     1.616/1.593, which inflated `p` by 25%.

Defect 2 is the one worth a permanent test, because it is silent in a way defect 1 is
not: nothing raises, nothing looks wrong, the ladder is monotone and the GCI barely
moves.  The extrapolated value and the GCI are ALMOST INVARIANT to a rescaling of h —
from three points, p and r enter Richardson only through r^p, which the measured phi
very nearly fix on their own — so the one number that actually goes wrong is `p`, and
`p` is the number a convergence study exists to produce.  A study can therefore be
wrong about its own subject while every headline it reports stays right.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import wheel_genome as wg          # noqa: E402
import wheel_mesh as wm            # noqa: E402
import wheel_wheel as ww           # noqa: E402
import study_deflection_gci as gci  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def genes():
    import json
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


# ---------------------------------------------------------------------------
# THE CELL SIZE IS MEASURED ON THE MESH THAT WAS SOLVED
# ---------------------------------------------------------------------------

def test_mesh_counts_come_from_the_wheel_that_was_actually_solved(genes):
    """`mesh_counts` must agree with an assembled `WheelMesh`, element for element.

    Not with `wheel_wheel.get_config(...).n_elements` — that is a formula, and a formula
    is what this study already got wrong once.  `n_nodes` in particular cannot be a
    formula: the seven sector blocks share seams and the merged node count is far below
    the sum of the grids (at `fine`, 132276 against the blocks' raw total).
    """
    for name in gci.LADDER:
        counts = gci.mesh_counts(name, genes)
        mesh = ww.build_wheel(genes, name)
        assert counts["n_elements"] == mesh.n_elements
        assert counts["n_nodes"] == mesh.n_nodes


def test_h_is_not_the_spoke_block_ladder(genes):
    """THE REGRESSION.  `wheel_mesh` and `wheel_wheel` share config NAMES, not meshes.

    Asserted as a strict inequality on every rung rather than as "h equals x", so it
    keeps biting if either ladder is re-tuned: any future edit that reaches for
    `wheel_mesh.get_config(name).n_elements` here fails, whatever the numbers become.
    """
    for name in gci.LADDER:
        spoke_block = wm.get_config(name)
        counts = gci.mesh_counts(name, genes)
        assert counts["n_elements"] > spoke_block.n_elements, (
            f"{name}: the full wheel must have more elements than one spoke block; "
            f"got {counts['n_elements']} vs {spoke_block.n_elements}")
        assert counts["n_span"] == ww.get_config(name).n_span != spoke_block.n_span


def test_the_refinement_ratios_are_the_ones_the_docstring_claims(genes):
    """The module docstring quotes 1.6162 and 1.5934, and a reader checks `p` against
    those.  If the ladder is re-tuned they must be updated together, so pin them."""
    h = {c: gci._h(gci.mesh_counts(c, genes)) for c in gci.EXTRAPOLATE_FROM}
    coarse, medium, fine = (h[c] for c in gci.EXTRAPOLATE_FROM)
    assert coarse / medium == pytest.approx(1.6162, abs=5e-4)
    assert medium / fine == pytest.approx(1.5934, abs=5e-4)


# ---------------------------------------------------------------------------
# THE OBSERVED ORDER
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p_true", [0.5, 1.0, 1.3, 2.0])
def test_observed_order_recovers_a_synthetic_constant_ratio_exactly(p_true):
    """phi_k = phi_exact - C h_k^p on a constant-ratio ladder, recovered to 1e-9.

    These pass under the index swap too — that is the point of the next test — but they
    are what pins the arithmetic itself.
    """
    r, exact, C = 2.0, 2.0, 0.3
    h = [r ** 2, r ** 1, 1.0]
    phi = [exact - C * x ** p_true for x in h]
    p, ok = gci.observed_order(phi, h)
    assert ok and p == pytest.approx(p_true, abs=1e-9)


@pytest.mark.parametrize("p_true", [0.5, 1.3, 2.0])
def test_observed_order_recovers_a_synthetic_NON_constant_ratio(p_true):
    """THE CASE THAT CAUGHT THE INDEX SWAP, and the only kind that can.

    r21 != r32, so using the coarse pair as the denominator no longer cancels.  The
    first draft returned 1.2568 here for p_true = 1.3; nothing else in the suite moved.
    """
    h = [1.0 / 4704 ** 0.5, 1.0 / 12288 ** 0.5, 1.0 / 31200 ** 0.5]  # the real ladder
    exact, C = 2.0, 0.3
    phi = [exact - C * x ** p_true for x in h]
    p, ok = gci.observed_order(phi, h)
    assert ok and p == pytest.approx(p_true, abs=1e-6)


def test_non_monotone_rungs_extrapolate_nothing():
    """An oscillatory triple is a real outcome and must return None, not a number.

    Three points always admit SOME fitted order; refusing when the sign flips is what
    stops a p from being reported for rungs that are not on one trend.
    """
    h = [4.0, 2.0, 1.0]
    p, ok = gci.observed_order([1.0, 2.0, 1.5], h)
    assert p is None and not ok
    assert gci.richardson([1.0, 2.0, 1.5], h)["extrapolated_mm"] is None


# ---------------------------------------------------------------------------
# WHY THE HEADLINE NUMBERS BARELY MOVED, STATED AS A PROPERTY
# ---------------------------------------------------------------------------

def test_rescaling_h_moves_p_but_essentially_not_the_extrapolation():
    """Multiply every h by a constant: `p` is unchanged, since only ratios enter.

    Raise every h to a POWER — which is what swapping to a differently-refined ladder
    does — and `p` scales inversely while `r32 ** p`, and therefore the extrapolated
    value and the GCI, stay put.  This is the property that let a 25% error in `p`
    survive alongside a correct 2.75% GCI, and it is why `p` needs its own pin.
    """
    phi = [1.97608, 1.99742, 2.01274]           # the measured SVK rungs
    h = [1.0 / 4704 ** 0.5, 1.0 / 12288 ** 0.5, 1.0 / 31200 ** 0.5]

    base = gci.richardson(phi, h)
    scaled = gci.richardson(phi, [10.0 * x for x in h])
    assert scaled["observed_order_p"] == pytest.approx(base["observed_order_p"], rel=1e-12)

    powered = gci.richardson(phi, [x ** 1.25 for x in h])
    assert powered["observed_order_p"] == pytest.approx(
        base["observed_order_p"] / 1.25, rel=2e-3)
    assert powered["extrapolated_mm"] == pytest.approx(
        base["extrapolated_mm"], rel=2e-4)
    assert powered["gci_fine_pct"] == pytest.approx(base["gci_fine_pct"], rel=5e-3)


def test_the_recorded_report_still_says_the_gate_is_undecidable():
    """PLAN §29's call rests on GCI(fine) >> the ±0.3% band.  Read it off the artifact.

    Deliberately NOT a pin on p, the extrapolated value or the GCI to more figures than
    the conclusion needs — the study is re-runnable and those move. What must not move
    silently is the comparison the plan retired a gate on.
    """
    import json
    path = os.path.join(REPO, "studies", "study_deflection_gci.json")
    with open(path) as fh:
        rep = json.load(fh)
    svk = rep["refinement"]["svk"]
    assert svk["monotone"]
    assert svk["gci_fine_pct"] > rep["gate_pct"], (
        f"GCI {svk['gci_fine_pct']:.3f}% is no longer wider than the "
        f"±{rep['gate_pct']}% band — PLAN §29's call needs re-deriving")
    assert not svk["gate_decidable"] and not svk["gate_decidable_under_all_h"]
    # And the counts in the artifact are the wheel's, not the spoke block's.
    for row in rep["rows"]:
        assert row["n_elements"] == ww.get_config(row["config"]).n_elements


def test_observed_p_is_not_the_williams_exponent():
    """PLAN §29 CLAIMED p = 0.502 matched Williams' lambda = 0.503 for the 322 deg
    junction wedge.  That agreement was an artifact of the wrong h and is now retired.

    Pinned as an inequality so the retraction cannot quietly un-retract: lambda for every
    wedge this design family produces is 0.50-0.51, and the measured p is not it.
    """
    import json
    path = os.path.join(REPO, "studies", "study_deflection_gci.json")
    with open(path) as fh:
        rep = json.load(fh)
    p = rep["refinement"]["svk"]["observed_order_p"]

    # Williams' mode-I eigenvalue for a traction-free re-entrant wedge of angle omega:
    # sin(lambda*omega) + lambda*sin(omega) = 0, smallest root in (0, 1).
    def williams(omega_deg):
        w = math.radians(omega_deg)
        f = lambda L: math.sin(L * w) + L * math.sin(w)
        lo, hi = 1e-6, 0.999
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    assert williams(360.0) == pytest.approx(0.5, abs=1e-6)     # a crack
    assert williams(270.0) == pytest.approx(0.5445, abs=1e-3)  # the textbook case
    lam = williams(322.0)                                       # the shipped hub wedge
    assert lam == pytest.approx(0.503, abs=2e-3)

    assert p > lam + 0.05, (
        f"observed p = {p:.4f} is back within 0.05 of Williams lambda = {lam:.4f}; "
        f"PLAN §29's retraction of that agreement needs revisiting")
