"""
Pins for `studies/study_fillet_fold.py` and for the `fillet=` construction it measures —
FILLET_PLAN.md STEP 1 RECORD PART 6, PLAN.md §44.

WHY THIS FILE EXISTS AT ALL.  `wheel_wheel.sector_blocks(..., fillet=)` has been the
fillet arc's measuring instrument since 2026-08-17 and had NO test.  Its central number —
the largest radius it survives — was recorded twice, by two criteria neither of which was
written down, and the two rows disagreed by 10-20x with no script of either surviving to
tell them apart.  That is the failure this file is written against: the apparatus a
decision rests on has to be re-runnable, and what it measures has to be named.

WHAT IS PINNED, AND WHAT IS DELIBERATELY NOT.  The RECONCILIATION is pinned hard — both
recorded tables must keep reproducing, because if a change to `_filleted_spoke` moves
either one, the change is to the instrument and everything measured with it is in
question.  The window EDGES are pinned to the construction detail that sets them (the arc
cell count, the mid-side node fraction) rather than to their millimetre values alone, so
a legitimate re-design of the node allocation reads as a moved mechanism rather than as
an unexplained number.

THE FILLET PATH IS OPT-IN AND MUST STAY INERT.  `fillet=None` is the default everywhere
and the first test is the control that says the default mesh is untouched by any of this.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import _gate_guard                     # noqa: E402
import wheel_fem as fem                # noqa: E402
import wheel_genome as wg              # noqa: E402
import wheel_wheel as ww               # noqa: E402
import study_fillet_fold as ff         # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


@pytest.fixture(scope="module")
def report():
    with open(os.path.join(REPO, "studies", "study_fillet_fold.json")) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# THE CONTROL: THE DEFAULT MESH DOES NOT KNOW THIS PARAMETER EXISTS
# ---------------------------------------------------------------------------

def test_zero_radius_reproduces_the_unfilleted_spoke(genes):
    """`fillet=(0, 0)` goes through the whole Coons rebuild and must land back on
    `fillet=None`'s block.

    Not a tautology: the two take different code paths, and their agreement is an
    independent numerical check that `sample` is affine in eta — that the unfilleted
    spoke already IS the Coons patch of its own boundary curves.  Every fillet number in
    the study is a difference from this baseline, so a drift here would move all of them.
    PART 3 measured 2.842e-14 mm; the tolerance below is loose enough to be about
    construction rather than about the last bit of a sum order.
    """
    for cfg in ("coarse", "medium"):
        a = np.asarray(ww.sector_blocks(genes, cfg, fillet=None)["spoke"], float)
        b = np.asarray(ww.sector_blocks(genes, cfg, fillet=(0.0, 0.0),
                                     fillet_blocking="spoke")["spoke"], float)
        assert np.abs(a - b).max() < 1e-12, cfg


def test_the_default_wheel_is_clean_under_every_criterion(genes):
    """The unfilleted mesh must pass all three criteria, or none of them means anything.

    In particular `gauss` — `det J` at the points the assembly integrates — is positive
    everywhere on the shipped default.  That is what makes a negative one on a filleted
    mesh a finding rather than a property of the checker.
    """
    for cfg in ("coarse", "medium"):
        spoke = np.asarray(ww.sector_blocks(genes, cfg, fillet=None)["spoke"], float)
        assert ff.cell_verdict(spoke)["mixed_sign_cells"] == 0
        assert ff.gauss_verdict(spoke)["non_positive_elements"] == 0
        mesh = ww.build_wheel(genes, cfg)
        assert ff.mesh_gauss_verdict(mesh)["non_positive_elements"] == 0


# ---------------------------------------------------------------------------
# THE RECONCILIATION — BOTH RECORDED TABLES, FROM ONE SWEEP
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("junction", ["hub", "rim"])
def test_the_two_contested_criteria_both_reproduce_at_coarse(genes, junction):
    """PART 3 recorded 0.20 mm; PART 5 recorded 4.00 (hub) / 3.00 (rim).  Both are right.

    Measured here on PART 5's own grid so "largest surviving" means what it meant there:
    the largest grid point with no fold at it or below it.  `coarse` only, because the
    claim under test is that the CRITERIA differ, and that does not need a second config
    to demonstrate — `medium` is carried by the committed report.
    """
    rows = ff.sweep_one(genes, "coarse", junction, ff.LEGACY_GRID)
    summ = ff.summarize(rows, ff.LEGACY_GRID)
    assert summ["block_cells"]["largest_surviving_mm"] == 0.20
    assert summ["build_wheel"]["first_fold_mm"] == (4.00 if junction == "hub" else 3.00)


def test_the_criteria_are_ordered_by_strictness_not_by_disagreement(genes):
    """Anything `block_cells` calls folded, `build_wheel` may still accept — never the
    reverse on this construction.

    The direction is the whole reconciliation: PART 5's row is not a second opinion about
    the same quantity, it is a weaker instrument reading later.  Pinned as an ordering so
    a future change that made `build_wheel` the stricter of the two would fail here
    rather than quietly invert the argument.
    """
    rows = [r for r in ff.sweep_one(genes, "coarse", "hub", ff.LEGACY_GRID)
            if "folds" in r]
    assert any(r["folds"]["block_cells"] for r in rows), "nothing folded — grid too small"
    for r in rows:
        if r["folds"]["build_wheel"]:
            assert r["folds"]["mesh_gauss"], r["radius_mm"]


# ---------------------------------------------------------------------------
# WHAT THE GUARD DOES NOT SEE.  THIS IS THE NEW FACT, AND IT IS ABOUT build_wheel
# ---------------------------------------------------------------------------

def test_build_wheel_accepts_a_mesh_with_inverted_gauss_points(genes):
    """`_orient_elements` is a shoelace over each element's FOUR CORNERS.

    Every config here is `order=2`, so one Q9 element spans 2x2 cells and its five mid
    nodes take no part in that sum.  A fold inside an element is therefore invisible to
    the check that exists to catch folds, and `build_wheel` returns a mesh the assembly
    would integrate with a negative Jacobian — which contributes NEGATIVE stiffness, the
    exact failure `_orient_elements`' own docstring says it is there to prevent.

    Pinned at one radius rather than as a sweep so the failure message names a case.  If
    the guard is ever strengthened this test is the one that should be updated, WITH the
    new measurement — the same rule FILLET_PLAN.md sets for
    `test_peak_stress_diverges_but_the_field_converges`.
    """
    mesh = ww.build_wheel(genes, "coarse", fillet=(0.25, 0.0),      # does NOT raise
                          fillet_blocking="spoke")
    bad = ff.mesh_gauss_verdict(mesh)
    assert bad["non_positive_elements"] > 0
    assert bad["min_det_j"] < 0.0
    assert set(bad["by_block"]) == {"spoke"}


def test_the_shipped_radii_are_far_outside_anything_usable(genes):
    """`R_hub = 0.664`, `R_rim = 3.000` — both far outside the usable window, which
    tops out at 0.24 mm at `coarse` and 0.11 at `medium`.

    The arc's standing conclusion, pinned as a fact about the construction rather than
    quoted from a plan file: this instrument cannot mesh the fillet that ships, at either
    junction, and no refinement helps because the window CLOSES under refinement.
    """
    assert genes[12] > 0.24 and genes[13] > 0.24
    with pytest.raises(ValueError, match="non-positive area"):
        ww.build_wheel(genes, "medium", fillet=True, fillet_blocking="spoke")


# ---------------------------------------------------------------------------
# BOTH EDGES OF THE USABLE WINDOW BELONG TO THE NODE ALLOCATION
# ---------------------------------------------------------------------------

def test_the_window_closes_when_the_arc_claims_a_second_cell(genes):
    """0.24 mm is usable at `coarse` and 0.25 is not, and the difference is one node.

    `k0 = clip(round((s_A - s0) / ds), 1, cap)` steps from 1 to 2 between them.  Nothing
    geometric happens there — the notch, the tangent length and the end cross-section all
    move by about a percent across that step — which is the evidence that the limit
    belongs to the construction and not to the fillet.
    """
    rows = {r["radius_mm"]: r for r in ff.sweep_one(genes, "coarse", "hub",
                                                    (0.24, 0.25))}
    assert rows[0.24]["arc_cells"] == 1 and rows[0.25]["arc_cells"] == 2
    assert not rows[0.24]["folds"]["mesh_gauss"]
    assert rows[0.25]["folds"]["mesh_gauss"]
    ratio = rows[0.25]["end_cross_section_ratio"] / rows[0.24]["end_cross_section_ratio"]
    assert ratio == pytest.approx(1.0, abs=0.02), "the geometry barely moved"


def test_the_window_opens_when_the_mid_side_node_reaches_the_middle(genes):
    """Below the window the same clamp's LOWER bound bites, and a quadratic edge folds.

    With the tangent point nearer than one station, `k0` is held at 1 and the first Q9
    element's mid-side node is dragged toward its own end.  A quadratic edge is singular
    at its end once that fraction leaves (0.25, 0.75); measured, the window opens as it
    climbs back through ~0.4.  Pinned as the correlation rather than as a threshold,
    because the exact crossing depends on the other direction's distortion too.
    """
    rows = [r for r in ff.sweep_one(genes, "coarse", "hub", ff.FINE_GRID)
            if "folds" in r and r["arc_cells"] == 1]
    folded = [r for r in rows if r["folds"]["mesh_gauss"]]
    clean = [r for r in rows if not r["folds"]["mesh_gauss"]]
    assert folded and clean
    assert max(r["mid_frac_fillet_flank"] for r in folded) < 0.40
    assert min(r["mid_frac_fillet_flank"] for r in clean) > 0.39
    # every radius below the window folds — there is no usable fillet however small
    assert min(r["radius_mm"] for r in rows) in [r["radius_mm"] for r in folded]


# ---------------------------------------------------------------------------
# THE COMMITTED REPORT
# ---------------------------------------------------------------------------

def test_the_committed_report_passes_its_own_self_checks(report):
    """The artifact in the tree must be a passing run, not a saved failure."""
    assert report["pass"] is True
    assert report["reconciliation"]["pass"] is True
    assert all(c["pass"] for c in report["controls"].values())


def test_the_committed_report_still_describes_this_construction(genes, report):
    """Re-measure one cell and compare with what was committed.

    A stored-number test would pass forever after a change to `_filleted_spoke`; this
    re-runs the sweep and asks whether the artifact still describes the code.  One cell,
    at `coarse`, because the point is to catch drift rather than to re-run the study.
    """
    fresh = ff.sweep_one(genes, "coarse", "hub", (0.20, 0.25, float(genes[12])))
    stored = {r["radius_mm"]: r for r in report["sweep"]["coarse:hub"]}
    for row in fresh:
        was = stored[row["radius_mm"]]
        assert row["folds"] == was["folds"], row["radius_mm"]
        assert row["arc_cells"] == was["arc_cells"], row["radius_mm"]
        assert row["end_cross_section_mm"] == pytest.approx(
            was["end_cross_section_mm"], rel=1e-9)


def test_fold_margin_cannot_see_the_fillet_genes(genes):
    """The constraint that covers the guard's blind spot on the DEFAULT path is blind to
    the parameter that opens it on the fillet path.

    `fold_margin` is built from the centreline and the thickness — genes 0-11.  `R_hub`
    and `R_rim` are 12 and 13.  Checked by moving them across their whole box and finding
    the margin unchanged to the bit, which is the exact form of the claim: this is not a
    weak constraint on the fillet, it is not a constraint on the fillet at all.
    """
    from study_mesh_quality import fold_margin
    cfg = ww.get_config("coarse")
    base = fold_margin(genes, cfg)
    for R_hub, R_rim in ((0.4, 0.5), (4.0, 3.0)):
        moved = np.array(genes, float)
        moved[12], moved[13] = R_hub, R_rim
        assert fold_margin(moved, cfg) == base


def test_the_committed_default_path_section_reports_the_blind_spot(report):
    """The guard's blind spot is reachable without a fillet, and is mostly — not
    entirely — covered by `fold_margin`.

    Pinned as an ordering and a direction rather than as counts: it is a sampled section
    with a fixed seed, so the counts are reproducible but they are not thresholds anyone
    should be defending.
    """
    d = report["default_path_blindness"]
    assert d["built_non_positive_gauss"] > 0.05 * d["built"]
    assert d["built_non_positive_gauss_and_min_sj_ok"] > 0
    assert d["meshable_non_positive_gauss"] < 0.05 * d["built_non_positive_gauss"]


def test_the_report_names_which_criterion_each_recorded_table_was(report):
    """The reconciliation is the deliverable, so the artifact has to carry it in a form
    that survives without the prose around it."""
    rc = report["reconciliation"]
    assert "block_cells" in rc["part3_criterion"]
    assert "build_wheel" in rc["part5_criterion"]
    for key in ("coarse:hub", "coarse:rim", "medium:hub", "medium:rim"):
        assert rc["part3_largest_surviving"][key]["agrees"]
        assert rc["part5_first_fold"][key]["agrees"]
        assert report["mechanism"][key]["upper_edge_is_arc_cell_step"]
        assert report["mechanism"][key]["part3_criterion_matches_upper_edge"]


# ---------------------------------------------------------------------------
# THE GATE GUARD, ON A DRIVER THAT IS NOT ONE OF THE NINE
# ---------------------------------------------------------------------------
#
# `tests/test_study_gate_guard.py` covers the nine drivers on the `make studies` recipe
# and says so in its prose; this one is a `make fillet` driver like `corner` and
# `junction`, so its guard is pinned here instead of widening that file's claim.  The
# protective pattern is §43's and is copied deliberately: the wrapper stops execution
# AND `HERE` is redirected, so a driver that stops calling the guard cannot overwrite the
# committed artifact while proving it.


class _GuardPassed(Exception):
    """The real guard was called and did not refuse."""


@pytest.fixture
def guard_stops_here(monkeypatch, tmp_path):
    real = _gate_guard.refuse_degraded_out
    calls = []

    def wrapper(ap, args, committed, degraded):
        calls.append(committed)
        real(ap, args, committed, degraded)
        raise _GuardPassed

    monkeypatch.setattr(_gate_guard, "refuse_degraded_out", wrapper)
    monkeypatch.setattr(ff, "HERE", str(tmp_path))
    return calls


def test_the_make_fillet_invocation_is_not_refused(monkeypatch, guard_stops_here):
    """`make fillet`'s own argv must reach the work.

    The assertion protecting the recipe from its own guard: a condition written against
    the wrong default would take the target down at the END of a run, since make sees
    only an exit status.
    """
    monkeypatch.setattr(sys, "argv",
                        ["study_fillet_fold.py", "--out", "study_fillet_fold.json"])
    with pytest.raises(_GuardPassed):
        ff.main()
    assert guard_stops_here == ["study_fillet_fold.json"]


@pytest.mark.parametrize("argv", [
    ["--coarse-grid"],
    ["--configs", "coarse"],
    ["--junctions", "hub"],
    ["--genome", "stage3_knee_best_medium.json"],
])
def test_a_degraded_run_may_not_be_filed_as_the_artifact(monkeypatch, guard_stops_here,
                                                         argv):
    """Each of these would produce a report that reads like the gate and is not one — a
    sweep that cannot resolve the window, or one taken on half the cells."""
    monkeypatch.setattr(sys, "argv", ["study_fillet_fold.py", *argv])
    with pytest.raises(SystemExit) as excinfo:
        ff.main()
    assert excinfo.value.code != 0


def test_an_explicit_out_lets_a_degraded_run_through(monkeypatch, guard_stops_here):
    """The refusal is about the NAME.  Redirected, a partial sweep is a normal thing to
    want — it is how the window was resolved in the first place."""
    monkeypatch.setattr(sys, "argv", ["study_fillet_fold.py", "--coarse-grid",
                                      "--out", "study_fillet_fold_probe.json"])
    with pytest.raises(_GuardPassed):
        ff.main()
