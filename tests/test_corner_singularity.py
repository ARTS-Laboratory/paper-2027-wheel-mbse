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
    #
    # IT MUST BE A Q9 VERTEX.  `measured_wedge_deg` skips midside nodes by design — a
    # midside cannot be a corner of the geometry — and returns 0.0 rather than 360.0 for
    # one, so a selection that can land on a midside is testing the tie-break of
    # `argsort`, not the wedge.  It used to: a whole ring of nodes sits at exactly
    # r = 49.500000, `abs(r - 49.4)` ties across all of them, and the 2026-08-18 uncap
    # default flip reordered the seam merge's node ownership enough to move the winner
    # from a vertex to a midside.  Nothing about the mesh, its quality or the wedge
    # changed — only which of several equally-near nodes came first.  Restrict the
    # candidates to vertices so the test asserts what its docstring says.
    verts = np.unique(mesh.conn[:, :4])
    r = np.linalg.norm(mesh.coords[verts], axis=1)
    inner = mesh.coords[verts[np.argsort(np.abs(r - 49.4))[0]]]
    got = cs.measured_wedge_deg(mesh, inner, cfg.order)
    assert got["n_incident_corner_elements"] == 4, got
    assert got["wedge_deg"] == pytest.approx(360.0, abs=1.0)

    for name, p in cs.corner_points(genes, cfg).items():
        w = cs.measured_wedge_deg(mesh, p, cfg.order)
        assert w["node_gap_mm"] < 1e-9, f"{name}: corner is not a mesh node"
        assert 180.0 < w["wedge_deg"] < 360.0, f"{name}: {w['wedge_deg']:.2f} deg"


def test_all_four_junction_corners_are_re_entrant(report):
    """All four are re-entrant and singular — but the two families are not one band.

    UPDATED 2026-08-22 (PLAN §45), and the reason is the finding rather than a tolerance.
    This test used to read `0.50 <= lambda < 0.53` for all four, which was a capped-mesh
    window: §38 flipped `UNCAP_DEFAULT` on 2026-08-18 and the two `P_c` corners stopped
    being near-crack-like, moving from 296.75/307.94 deg of wedge to 268.08/271.02 and
    from lambda 0.5144/0.5079 to 0.5477/0.5429. The committed report was not refreshed at
    the flip, so the old window kept passing against a mesh the tree no longer builds.

    Pinned per family, because they are different claims:

      `P_t` is THE PART'S CORNER (agreeing with the exported flank crossing to 0.073 um,
      FILLET_PLAN PART 2) and its wedge is set by the spoke geometry, not by a mesh
      option. It is stable across every uncap setting measured and is pinned tightly.

      `P_c` is a MESH ARTEFACT and its wedge is a function of `uncap`. It is pinned only
      as "re-entrant and singular", with a band wide enough to hold both the capped and
      the uncapped values, because a number that moves when a model option moves must not
      be asserted as though it were a property of the wheel.
    """
    for name, d in report["williams"].items():
        assert d["re_entrant"], f"{name} wedge {d['wedge_deg']:.2f} deg"
        assert 0.50 <= d["lambda"] < 1.0, f"{name} lambda {d['lambda']:.4f}"
        if name.endswith("P_t"):
            assert d["lambda"] == pytest.approx(0.5031, abs=0.002), (
                f"{name} lambda {d['lambda']:.4f} — this is the PART's corner and its "
                f"wedge should not move with a mesh option")
        else:
            assert 0.50 <= d["lambda"] < 0.56, f"{name} lambda {d['lambda']:.4f}"


def test_the_committed_report_describes_the_mesh_the_tree_BUILDS_TODAY(genes, report):
    """The artifact must describe the DEFAULT mesh, not the mesh of the day it was run.

    WRITTEN BECAUSE IT DID NOT. `study_corner_singularity.json` was committed 2026-08-16
    against a capped mesh; §38 flipped `UNCAP_DEFAULT` on 2026-08-18 and this driver takes
    that default (`build_wheel(genes, cfg)`, bare). For four days the committed record of
    the arc's Step 0 baseline was two corners wrong by 28.7 and 36.9 deg of wedge, and
    every peak stress in it too high by ~28% — and nothing was red, because every test
    read the same stale file. See PLAN §45.

    `make studies` would not have caught it either: this driver is not one of the nine.
    So the check is here, and it is a re-measurement rather than a stored number — rebuild
    the finest rung and ask whether the report still describes it. Geometry only, no
    field solved, which is what makes it cheap enough to be a test.
    """
    finest = ww.get_config(report["ladder"][-1])
    mesh = ww.build_wheel(genes, finest)
    fine_rung = report["rungs"][-1]
    assert mesh.n_elements == fine_rung["n_elements"]
    assert mesh.n_nodes == fine_rung["n_nodes"]
    for name, p in cs.corner_points(genes, finest).items():
        got = cs.measured_wedge_deg(mesh, p, finest.order)
        assert got["wedge_deg"] == pytest.approx(
            report["williams"][name]["wedge_deg"], abs=0.05), (
            f"{name}: the mesh built today has a {got['wedge_deg']:.2f} deg wedge and "
            f"the committed report says {report['williams'][name]['wedge_deg']:.2f} — "
            f"re-run `make corner` and read what moved before trusting either")


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

    THE TWO SIDES CAME FROM DIFFERENT MESH VINTAGES BETWEEN 2026-08-18 AND 2026-08-23,
    AND NO LONGER DO. §45 refreshed `study_corner_singularity.json` against the uncapped
    default and recorded here, rather than quietly relying on it, that
    `study_deflection_gci.json` was still the 2026-08-14 CAPPED ladder and 95 minutes to
    re-run. §49 re-ran it. Both sides are now post-flip, and the margin the inequality
    rests on widened rather than narrowed — the p-norm slope moved −0.0441 → −0.0262
    against a peak slope of −0.4695, so 10.6x became 17.9x. The assertion is unchanged;
    what changed is that it no longer has to survive a mismatch to hold.
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
