"""
Pins for `studies/study_corner_singularity.py` — PLAN §30.

That the peak stress diverges is NOT this file's finding and is already pinned, on a
different statistic, by
`tests/test_wheel_fea.py::test_peak_stress_diverges_but_the_field_converges` — read that
one first. What is pinned here is the part that was missing: the RATE of the divergence,
which of the four re-entrant corners produces it, and the wedge angle each corner
actually has. Those are what a filleted FEA mesh would have to move, and what the
retracted exponent match in §29 was reaching for (see `tests/test_deflection_gci.py`).

The pins are deliberately asymmetric. The DIVERGENCE is pinned hard, because a slope of
zero and a slope of -0.5 are qualitatively different claims and nothing should be able to
flip that quietly. The value of lambda is pinned loosely, because three or four rungs
starting from a 960-element mesh do not determine an exponent to two decimals and
pretending otherwise is how §29 went wrong in the first place.
"""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import wheel_genome as wg            # noqa: E402
import wheel_wheel as ww             # noqa: E402
import study_corner_singularity as cs  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


@pytest.fixture(scope="module")
def report():
    with open(os.path.join(REPO, "studies", "study_corner_singularity.json")) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# WILLIAMS
# ---------------------------------------------------------------------------

def test_williams_reproduces_the_two_cases_with_known_answers():
    """A crack is omega = 360 deg and lambda is exactly 1/2; omega = 270 deg is the
    textbook 0.5445. Without these the eigenvalue solver is an unchecked root-find."""
    assert cs.williams_lambda(360.0) == pytest.approx(0.5, abs=1e-6)
    assert cs.williams_lambda(270.0) == pytest.approx(0.5445, abs=1e-3)
    assert cs.williams_lambda(180.0) == pytest.approx(1.0, abs=1e-3)  # a flat edge


def test_williams_is_monotone_and_singular_only_when_re_entrant():
    """lambda < 1 — i.e. a singular stress — exactly when the wedge is re-entrant.

    This is what makes "re-entrant" the right test in `wheel_step_export`'s fillet
    classifier, and it is worth pinning next to the fillet work it justifies.
    """
    lams = [cs.williams_lambda(w) for w in (190, 240, 270, 300, 322, 359)]
    assert all(a > b for a, b in zip(lams, lams[1:])), lams
    assert all(0.5 <= L < 1.0 for L in lams)


# ---------------------------------------------------------------------------
# THE WEDGE IS MEASURED ON THE MESH, NOT QUOTED FROM THE MANIFEST
# ---------------------------------------------------------------------------

def test_the_wedge_is_measured_on_the_mesh_that_has_the_corner(genes):
    """`measured_wedge_deg` sums incident element interior angles at the corner node.

    Checked against the cases where the answer is known by construction rather than
    against a stored number: a node in the middle of the material sums to 360 deg, and
    the junction corners land in (180, 360) — re-entrant, which is the whole premise.

    The export manifest's 322 deg describes the exported SOLID, which is FILLETED at
    these corners. Reading the prediction off that body is exactly the wrong-body error
    §29 made with the cell size, so this never consults the manifest.
    """
    cfg = ww.get_config("coarse")
    mesh = ww.build_wheel(genes, cfg)

    # An interior node: take one well inside the rim band, away from every boundary.
    r = np.linalg.norm(mesh.coords, axis=1)
    inner = mesh.coords[np.argsort(np.abs(r - 49.4))[0]]
    got = cs.measured_wedge_deg(mesh, inner, cfg.order)
    assert got["wedge_deg"] == pytest.approx(360.0, abs=1.0)

    for name, p in cs.corner_points(genes, cfg).items():
        w = cs.measured_wedge_deg(mesh, p, cfg.order)
        assert w["node_gap_mm"] < 1e-9, f"{name}: corner is not a mesh node"
        assert 180.0 < w["wedge_deg"] < 360.0, f"{name}: {w['wedge_deg']:.2f} deg"


def test_all_four_junction_corners_are_re_entrant(report):
    for name, d in report["williams"].items():
        assert d["re_entrant"], f"{name} wedge {d['wedge_deg']:.2f} deg"
        assert 0.50 <= d["lambda"] < 0.53, f"{name} lambda {d['lambda']:.4f}"


# ---------------------------------------------------------------------------
# THE HEADLINE
# ---------------------------------------------------------------------------

def test_the_peak_stress_does_not_converge(report):
    """THE FINDING. A non-singular peak converges and gives d log(peak)/d log(h) ~ 0.

    Every corner here, and the global maximum with them, grows monotonically as the mesh
    refines. Pinned as a strict monotone increase plus a slope well away from zero,
    because that pair is the qualitative claim; the exponent is pinned separately and
    much more loosely.
    """
    for name, d in report["divergence"].items():
        peaks = d["peak_mpa"]
        assert all(b > a for a, b in zip(peaks, peaks[1:])), f"{name}: {peaks}"
        assert d["slope_finest3"] < -0.15, (
            f"{name}: d log(peak)/d log(h) = {d['slope_finest3']:+.4f} is close enough "
            f"to zero that the peak may now be converging — PLAN §30 needs re-deriving")
        assert d["diverges"]


def test_the_global_maximum_stress_is_at_a_junction_corner(report):
    """The divergent corner is not an out-of-the-way detail: it IS the peak stress of
    the whole wheel, at every rung. That is what connects §30 to the stress constraint."""
    for rung in report["rungs"]:
        best = max(c["peak_vm_mpa"] for c in rung["corners"].values())
        assert best == pytest.approx(rung["global_max_vm_mpa"], rel=1e-9)


def test_the_measured_exponent_is_consistent_with_williams_but_only_loosely(report):
    """lambda from the divergence rate against lambda from the wedge angle.

    abs=0.15, and the tolerance is the point. Williams puts every corner at 0.503-0.514;
    the divergence rates give 0.38-0.56 on a ladder whose coarsest rung is 960 elements.
    That is agreement on the MECHANISM and not on the number, and §29 was wrong precisely
    because it read a three-decimal match as confirmation.
    """
    for name, d in report["divergence"].items():
        if name not in report["williams"]:
            continue
        lam_w = report["williams"][name]["lambda"]
        assert d["lambda_from_slope"] == pytest.approx(lam_w, abs=0.15), (
            f"{name}: divergence gives {d['lambda_from_slope']:.4f}, Williams "
            f"{lam_w:.4f}")


def test_the_p_norm_the_optimizer_uses_diverges_far_more_slowly(report):
    """§30's qualifier, pinned so it cannot be dropped when the headline is quoted.

    The objective does not see the raw peak — it sees a Gauss-weighted p-norm, whose
    drift up the ladder is an order of magnitude slower. The stress CONSTRAINT is
    therefore not the raw singularity, and a reader who takes "the peak diverges" as
    "the constraint is meaningless" has over-read it.
    """
    gci = os.path.join(REPO, "studies", "study_deflection_gci.json")
    with open(gci) as fh:
        rows = json.load(fh)["rows"]
    lh = np.log([r["h"] for r in rows])
    util = np.log([r["svk"]["stress_utilisation"] for r in rows])
    slope = float(np.polyfit(lh, util, 1)[0])
    peak = report["divergence"]["global_max_vm"]["slope_all"]
    assert -0.10 < slope < 0.0, f"p-norm utilisation slope {slope:+.4f}"
    assert abs(slope) < abs(peak) / 5.0, (
        f"the p-norm ({slope:+.4f}) is no longer far flatter than the raw peak "
        f"({peak:+.4f}) — §30's qualifier needs re-deriving")
