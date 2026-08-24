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

import ast
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
import study_fillet_block as fb       # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


@pytest.fixture(scope="module")
def report():
    with open(os.path.join(REPO, "studies", "study_corner_singularity.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def fillet_report():
    """FILLET_PLAN.md Step 2's ladder — `make corner-fillet`, the same driver with
    `--fillet genome`."""
    path = os.path.join(REPO, "studies", "study_corner_singularity_fillet.json")
    with open(path) as fh:
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


# ---------------------------------------------------------------------------
# STEP 2 — THE SAME LADDER ON A FILLETED MESH
# ---------------------------------------------------------------------------
#
# Everything below reads `studies/study_corner_singularity_fillet.json` or rebuilds
# geometry; none of it solves a field, which is what keeps the set cheap enough to be
# tests rather than a study.
#
# THE SCOPE THESE PIN IS PART 10's AND IT HAS NOT MOVED. `fillet=` is a measurement
# instrument for ONE genome — 6 of 16 feasible genomes refuse it at their own radii — so
# nothing here asserts anything about the optimizer, and `test_nothing_wires_the_fillet_
# into_the_objective` says so in a form that can go red.


def test_the_probe_points_the_fillet_adds_are_on_the_geometry_they_claim(genes):
    """`A`, `arc` and `B` on one circle of the gene's radius; `B` and `N` on the ring.

    Checked against the construction rather than against stored coordinates, because the
    whole value of `fillet_points` is that it reads what `build_wheel` MESHED. A version
    that read the wrong grid index would still return four plausible points near the
    junction, and only geometry catches that.
    """
    cfg = ww.get_config("coarse")
    arcs = cs.fillet_arcs(genes, cfg, True)
    pts = cs.fillet_points(genes, cfg, True)
    for label, R_gene in (("hub", float(genes[12])), ("rim", float(genes[13]))):
        arc = arcs[label]
        assert arc["fit_residual_mm"] < 1e-9, (
            f"{label}: the arc's nodes are not on a circle to {arc['fit_residual_mm']:.2e} "
            f"mm — `fillet_arcs` is fitting the wrong grid")
        assert arc["radius"] == pytest.approx(R_gene, rel=1e-9), (
            f"{label}: fitted radius {arc['radius']:.6f} against gene {R_gene:.6f}")
        for name in ("A", "arc", "B"):
            d = float(np.linalg.norm(pts[f"{label}:{name}"] - arc["centre"]))
            assert d == pytest.approx(arc["radius"], abs=1e-9), f"{label}:{name} off arc"

        ring_r = (ww.HUB_RADIUS_MM if label == "hub"
                  else ww.rim_inner_radius(ww.HUB_RIM_SPAN_MM))
        for name in ("B", "N"):
            r = float(np.linalg.norm(pts[f"{label}:{name}"]))
            assert r == pytest.approx(ring_r, abs=1e-9), (
                f"{label}:{name} is at r = {r:.6f}, not on the ring circle {ring_r}")


def test_the_reference_corners_do_not_move_when_the_mesh_is_filleted(genes):
    """`corner_points` is the arc's BEFORE/AFTER anchor and must be blind to `fillet`.

    THE TRAP IS SPECIFIC AND IT IS SILENT. The filleted `<ring>_junction` patch has the
    same shape and the same index convention as the unfilleted one, and its `[-1, 0]` node
    really is still `P_c`. Its `[0, 0]` is not `P_t` — it is `N`, 0.47 mm away at the hub
    and 4.34 mm at the rim — so a `corner_points` that read the filleted blocking would
    keep the label, move the place, and report that `rim:P_t` stopped diverging on the
    strength of a measurement taken somewhere else.
    """
    cfg = ww.get_config("coarse")
    plain = cs.corner_points(genes, cfg)
    with_fillet = cs.probe_points(genes, cfg, fillet=True)
    for name, p in plain.items():
        assert np.allclose(with_fillet[name], p, atol=0.0), name

    filleted_blocks = ww.sector_blocks(genes, cfg, fillet=True)
    for ring, label in (("hub_junction", "hub"), ("rim_junction", "rim")):
        N = np.asarray(filleted_blocks[ring][0, 0], dtype=float)
        assert np.linalg.norm(N - plain[f"{label}:P_t"]) > 0.4, (
            f"{label}: `N` and `P_t` have come together, so this test no longer "
            f"distinguishes the two blockings — re-derive it before trusting it")
        assert np.allclose(filleted_blocks[ring][-1, 0], plain[f"{label}:P_c"])


def test_the_wedge_search_cannot_land_on_a_midside_node(genes):
    """A midside contributes a straight 180 deg and is skipped, so a nearest-node search
    that can select one returns 0.00 deg — a number shaped like a measurement.

    It never fired while every probe was an exact vertex, which the four unfilleted
    corners are. On a filleted mesh `P_t` is interior and not a node at all, and
    `rim:P_t` at `coarse` reported 0.00 deg against `hub:P_t`'s correct 360.00 purely on
    which of two equally-near nodes came first.
    """
    cfg = ww.get_config("coarse")
    mesh = ww.build_wheel(genes, cfg, fillet=True)
    for name, p in cs.corner_points(genes, cfg).items():
        got = cs.measured_wedge_deg(mesh, p, cfg.order)
        assert got["n_incident_corner_elements"] > 0, (
            f"{name}: the wedge search landed on a midside node and returned "
            f"{got['wedge_deg']:.2f} deg")
        if name.endswith("P_t"):
            assert got["wedge_deg"] == pytest.approx(360.0, abs=1.0), (
                f"{name}: {got['wedge_deg']:.2f} deg — the fillet should have made this "
                f"an interior point")


def test_the_filleted_report_describes_the_mesh_the_tree_BUILDS_TODAY(genes,
                                                                     fillet_report):
    """The filleted artifact's twin of the check §45 had to write for the unfilleted one.

    Same failure mode, same cost: rebuild the finest rung and ask whether the committed
    report still describes it. `study_corner_singularity_fillet.json` is not on `make
    studies` either, and `fillet=` reaches deeper into `wheel_wheel` than `uncap` did.
    """
    finest = ww.get_config(fillet_report["ladder"][-1])
    mesh = ww.build_wheel(genes, finest, fillet=True)
    fine_rung = fillet_report["rungs"][-1]
    assert mesh.n_elements == fine_rung["n_elements"]
    assert mesh.n_nodes == fine_rung["n_nodes"]
    for name, p in cs.probe_points(genes, finest, True).items():
        got = cs.measured_wedge_deg(mesh, p, finest.order)
        assert got["wedge_deg"] == pytest.approx(
            fillet_report["williams"][name]["wedge_deg"], abs=0.05), (
            f"{name}: the filleted mesh built today has a {got['wedge_deg']:.2f} deg "
            f"wedge and the committed report says "
            f"{fillet_report['williams'][name]['wedge_deg']:.2f} — re-run "
            f"`make corner-fillet` and read what moved before trusting either")


def test_the_two_reports_are_the_same_instrument(report, fillet_report):
    """One driver, one flag. A before/after measured by two scripts is two instruments,
    and this arc was already bitten by that once — PART 6 found two recorded fold tables
    disagreeing by 20x with neither criterion written down and no script of either
    surviving."""
    for key in ("genome", "ladder", "n_spokes", "probe_radius_mm"):
        assert report[key] == fillet_report[key], key
    assert report["fillet"] is None
    assert fillet_report["fillet"] is True


def test_the_fillet_removes_the_corner_it_reaches_and_not_the_other_one(fillet_report):
    """THE STEP 2 FINDING, both halves, and the second half is the one that limits it.

    `P_t` is the PART's corner and the one this fillet is tangent to. On the filleted mesh
    it is not a corner at all — it is a point in the material's interior, 360 deg, four
    incident elements — and its peak stops diverging.

    `P_c` is the END CAP's corner. The fillet does not reach it, it is still re-entrant to
    within 0.2 deg of its unfilleted wedge, and it still diverges. That is not a
    disappointment to be worked around: FILLET_PLAN's PART 8 closed the chain that
    predicts it, and this is the prediction arriving.
    """
    w = fillet_report["williams"]
    d = fillet_report["divergence"]
    for label in ("hub", "rim"):
        pt, pc = f"{label}:P_t", f"{label}:P_c"

        assert w[pt]["kind"] == "interior", (
            f"{pt}: {w[pt]['wedge_deg']:.2f} deg, kind {w[pt]['kind']} — the fillet was "
            f"supposed to take this corner out of the boundary")
        assert "lambda" not in w[pt], (
            f"{pt}: Williams' eigenvalue is quoted for an INTERIOR point, where it is "
            f"not defined — a 360 deg interior node would print a crack's 0.5000")
        assert not d[pt]["diverges"], (
            f"{pt}: d log(peak)/d log(h) = {d[pt]['slope_finest3']:+.4f} on the FILLETED "
            f"mesh — the fillet did not remove the singularity")

        assert w[pc]["kind"] == "re_entrant", f"{pc}: {w[pc]['kind']}"
        assert d[pc]["diverges"], (
            f"{pc}: the end-cap corner has stopped diverging on the filleted mesh. That "
            f"would be a bigger result than Step 2's and it needs its own measurement — "
            f"do not update this test without one")


def test_the_wedge_of_the_corner_the_fillet_does_not_reach_is_unchanged(report,
                                                                       fillet_report):
    """`P_c`'s wedge is a property of the END CAP, so re-blocking the sector around a
    fillet must not move it. Pinned because it is what licenses comparing the two `P_c`
    columns at all: if the wedge moved, the exponent moved, and the two ladders would be
    measuring different corners under one name."""
    for label in ("hub", "rim"):
        pc = f"{label}:P_c"
        a = report["williams"][pc]["wedge_deg"]
        b = fillet_report["williams"][pc]["wedge_deg"]
        assert b == pytest.approx(a, abs=0.5), (
            f"{pc}: {a:.2f} deg unfilleted against {b:.2f} filleted")


def test_the_fillet_surface_is_smooth_and_introduces_no_new_corner(fillet_report):
    """A fillet is tangent to both legs, so nothing on it may be re-entrant.

    The construction could have made one anyway: `A` and `B` are block corners as well as
    tangent points, and `N` is where four blocks meet. A re-entrant wedge at any of them
    would be a singularity the fillet INTRODUCED, which would be worth far more attention
    than the one it removed.

    `smooth` is a 25 deg band, and the width is the measurement rather than a slack
    tolerance: a node on a discretised curved boundary sums to 180 plus the arc's turn
    across one node, which is a mesh number that goes to zero under refinement. The finest
    rung's arc nodes come in at 183-189 deg.
    """
    w = fillet_report["williams"]
    for label in ("hub", "rim"):
        for name in ("A", "arc", "B"):
            k = w[f"{label}:{name}"]
            assert k["kind"] == "smooth", (
                f"{label}:{name} measures {k['wedge_deg']:.2f} deg ({k['kind']}) — the "
                f"fillet has introduced a corner where it should be tangent")
        n = w[f"{label}:N"]
        assert n["kind"] == "interior", f"{label}:N is {n['wedge_deg']:.2f} deg"


def test_the_fillet_surface_peak_settles_where_the_sharp_corner_never_did(report,
                                                                         fillet_report):
    """THE HEADLINE, AS A COMPARISON RATHER THAN AS A THRESHOLD.

    "Does the peak converge" is answered here by the ratio of successive differences up
    the ladder, not by the log-log slope. The slope was written for a ladder on which
    every probe was singular; a BOUNDED quantity still approaching its limit has a nonzero
    slope over three rungs, and two of the fillet's probes land there — `hub:B` fits
    -0.2331 while its differences run 1.76, 0.96, 0.22 MPa. The differences are the
    sharper instrument, and on the unfilleted ladder they are unambiguous: every one of
    the four corners holds its ratio at 0.999 or above.
    """
    for label in ("hub", "rim"):
        sharp = report["divergence"][f"{label}:P_t"]
        assert sharp["increment_ratio_finest"] > 0.9, (
            f"{label}:P_t unfilleted has stopped diverging on the successive-difference "
            f"test ({sharp['increment_ratio_finest']:+.3f}) — the whole before/after "
            f"rests on it")

        surf = fillet_report["divergence"][f"{label}:surface"]
        assert surf["settling"], (
            f"{label}:surface differences {surf['increments_mpa']} give ratios "
            f"{surf['increment_ratios']} — the fillet's own surface peak is not settling")
        assert surf["increment_ratio_finest"] < cs.SETTLING_RATIO
        assert surf["tail_fraction"] < 0.03, (
            f"{label}:surface still moves {surf['tail_fraction']:.2%} on the last rung")
        assert surf["peak_mpa"][-1] < sharp["peak_mpa"][-1], (
            f"{label}: the fillet surface's peak {surf['peak_mpa'][-1]:.2f} MPa is not "
            f"below the sharp corner's {sharp['peak_mpa'][-1]:.2f} MPa at `fine`")


def test_step_2s_HEADLINE_is_not_delivered_and_the_reason_is_measured(fillet_report):
    """FILLET_PLAN Step 2 says the claim to test is that "the peak stress stops
    diverging — this is the one that unlocks quoting a max". IT DOES NOT, and this test
    exists so that nobody reads the surface result above as though it had.

    The wheel's global maximum is on `rim:P_c`, the END CAP's artefact corner, which the
    fillet does not reach — located here by `argmax` over the whole Gauss field rather
    than asserted, exactly as PART 4 and PART 7 did by hand. `smoke` is excluded and the
    exclusion is the finding's own edge: on a 1152-element mesh the peak is out on the hub
    fillet instead, which is what a mesh too coarse to resolve a singularity looks like.
    """
    for rung in fillet_report["rungs"]:
        if rung["config"] == fillet_report["ladder"][0]:
            continue
        g = rung["global_peak"]
        assert g["nearest_probe"] == "rim:P_c", (
            f"{rung['config']}: the wheel's peak has moved to {g['nearest_probe']} "
            f"({g['distance_mm'] * 1000.0:.1f} um) — if the fillet has taken the peak off "
            f"the end-cap corner then Step 2's headline IS delivered, and that needs "
            f"its own section rather than a tolerance change here")
        assert g["distance_mm"] < 0.1

    assert fillet_report["divergence"]["global_max_vm"]["diverges"]
    assert fillet_report["divergence"]["global_max_vm"]["increment_ratio_finest"] > 0.9


def test_the_filleted_blocking_solves_the_SAME_WHEEL(fillet_report):
    """THE CONTROL. Without it the filleted ladder's 38% shift in `axle_drop_mm` is
    equally well explained by the fillet's stiffness and by a different model.

    The filleted blocking takes an explicit radius pair, so drive it toward zero: the mesh
    must reproduce the unfilleted wheel. It cannot be driven TO zero — `sector_blocks`
    refuses a zero radius, because the re-cut moves four blocks and "no fillet here" is a
    different blocking rather than this one at `R = 0` — so this is a limit, not an
    identity, and the smallest rung's residual is asserted rather than hidden.
    """
    c = fillet_report["continuity"]
    rows = [r for r in c["rows"] if r["built"]]
    assert len(rows) >= 8

    near_zero = min(rows, key=lambda r: r["R_mm"])
    assert abs(near_zero["rel_to_unfilleted"]) < 0.02, (
        f"at R = {near_zero['R_mm']} mm the filleted blocking gives "
        f"{near_zero['axle_drop_mm']:.6f} mm against the unfilleted "
        f"{c['unfilleted_axle_drop_mm']:.6f} — the two are not the same wheel and no "
        f"comparison in Step 2 is safe")

    # Monotone above the floor. The very smallest radius is excluded on a stated ground:
    # its boundary layer is thinner than `coarse` resolves and the drop reads HIGH, which
    # is a discretisation statement about the control and not about the wheel.
    body = sorted((r for r in rows if r["R_mm"] >= 0.05), key=lambda r: r["R_mm"])
    drops = [r["axle_drop_mm"] for r in body]
    assert all(b < a for a, b in zip(drops, drops[1:])), (
        f"the fillet's stiffening is not monotone in R: {drops}")

    assert c["shipped_rel_to_unfilleted"] < -0.30, (
        f"the genome's own fillet moves the axle drop by only "
        f"{c['shipped_rel_to_unfilleted']:+.2%}")


def test_the_deflection_converges_on_the_filleted_mesh_and_not_on_the_sharp_one(
        report, fillet_report):
    """FILLET_PLAN's reason 2 for this arc, measured: "the ±0.3% absolute deflection band
    could not be adjudicated", retired by §29 to a relative clause.

    The unfilleted axle drop is still climbing at `fine` — the corner's singular field
    pollutes a global functional, which is the mechanism §29 spent 95 minutes failing to
    identify. The filleted one is flat from `coarse` up. Both spreads are computed over
    the same three rungs of the same ladder.
    """
    sharp = report["deflection"]["interp"]
    rounded = fillet_report["deflection"]["interp"]

    # RESTATED AFTER PART 18.  This used to compare the SPREAD of `axle_drop_mm` and
    # demand the filleted one be under 0.3%.  That reading snaps to the nearest node and
    # on the filleted mesh its ladder is not even monotone -- increments +0.001359 then
    # -0.001236 -- so the 0.141% it reported was noise that happened to be small.  The
    # claim the arc actually needs is about what is STILL TO COME after the finest rung,
    # and on the interpolated reading both ladders are monotone and both settle, so the
    # remaining tail is well defined for each.
    for name, dd in (("unfilleted", sharp), ("filleted", rounded)):
        assert dd["monotone"], (name, dd["values_mm"])
        assert dd["settling"], (name, dd["increment_ratio"])
        assert dd["remaining_tail_pct"] is not None, name

    assert rounded["remaining_tail_pct"] < 0.3, (
        f"the filleted deflection's remaining tail is "
        f"{rounded['remaining_tail_pct']:.3f}% — it no longer sits inside the +-0.3% "
        "band this arc exists partly to earn back")
    assert sharp["remaining_tail_pct"] > 0.3, (
        f"the unfilleted deflection's tail is {sharp['remaining_tail_pct']:.3f}% — if it "
        "is now inside the band too, the fillet's convergence argument has lost its "
        "contrast and that is a finding")
    assert sharp["remaining_tail_pct"] > 5.0 * rounded["remaining_tail_pct"]

    # and the node reading is kept in the artifact precisely so this stays checkable:
    # on the filleted mesh it is NOT monotone, which is why its spread meant nothing
    assert fillet_report["deflection"]["node"]["monotone"] is False


def test_nothing_wires_the_fillet_into_the_objective():
    """PART 10's scope, as a check rather than a note. `fillet=` is a measurement
    instrument for ONE genome — 6 of 16 feasible genomes refuse it at their own radii and
    6 of the 10 that build sit under `MIN_SJ_TARGET` — so it must not reach the optimizer
    on the strength of Step 2. Step 2 measured corners; it did not measure genomes."""
    # PARSED, NOT GREPPED, and the two failures a grep gives here are both real. A pattern
    # loose enough to catch `fillet=True` also catches `wheel_objective`'s local
    # `fillet = jnp.sum(...)` — the fillet-MARGIN barrier, which has nothing to do with the
    # mesh — and `wheel_fea`'s `R_hub_fillet=`, which goes to the EXPORTER, where the
    # fillets have always been geometry. What may not appear is the keyword argument, and
    # `ast` knows the difference between one of those and an assignment.
    for mod in ("wheel_objective.py", "wheel_stage3.py", "wheel_fea.py",
                "wheel_adjoint.py", "wheel_pool.py"):
        with open(os.path.join(REPO, "src", mod)) as fh:
            tree = ast.parse(fh.read(), filename=mod)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg not in ("fillet", "fillet_blocking"):
                    continue
                assert isinstance(kw.value, ast.Constant) and kw.value.value is None, (
                    f"{mod}:{node.lineno} passes {kw.arg}= to a call — `fillet=` is a "
                    f"measurement instrument for one genome and must not reach the "
                    f"optimizer (PLAN §48, §52)")


def test_parse_fillet_refuses_what_it_cannot_mean():
    assert cs.parse_fillet("none") is None
    assert cs.parse_fillet(None) is None
    assert cs.parse_fillet("genome") is True
    assert cs.parse_fillet("0.5,2.0") == (0.5, 2.0)
    for bad in ("0.5", "0.5,1.0,2.0", "big"):
        with pytest.raises(Exception):
            cs.parse_fillet(bad)


def test_the_mesh_fillets_the_CORNERS_AND_RADII_THE_EXPORTER_ACTUALLY_BUILT(genes,
                                                                              report):
    """The filleted mesh has to be a model of the shipped part, not of the request.

    `wheel_step_export.kt_report`'s own docstring records the failure this guards: OCC
    quietly delivers a fraction of a radius it cannot fit, and the rim once shipped twelve
    of its twenty-four corners SQUARE while the manifest reported `kt_error_pct = +0.0%`.
    A mesh that rounds a corner the part has sharp would understate its stress and its
    compliance, and §52 quotes a 38% deflection change off exactly this construction.

    Two halves, and the second is the one that is not circular. The radius the mesh builds
    must equal the radius the exporter BUILT (not the one the genome asked for); and the
    manifest's `worst_wedge_deg` must be `P_t`'s, which is how we know the two bodies round
    the same corner family — the part does not have `P_c` at all, and that is why the mesh's
    last artefact corner is an artefact.
    """
    man_path = os.path.join(REPO, "export", "wheel_step_manifest.json")
    if not os.path.exists(man_path):
        pytest.skip("no export/wheel_step_manifest.json in this tree")
    with open(man_path) as fh:
        detail = {d["junction"]: d for d in json.load(fh)["fillets"]["detail"]}

    cfg = ww.get_config("coarse")
    arcs = cs.fillet_arcs(genes, cfg, True)
    # The FINEST rung's wedges, off the committed unfilleted report. A `coarse` wedge is
    # 1.9 deg off the `fine` one — the wedge is summed from incident element angles and
    # the elements are what refine — and the manifest's own value is rounded to whole
    # degrees, so a comparison run at `coarse` is comparing two roundings.
    unfilleted_wedges = {k: v["wedge_deg"] for k, v in report["williams"].items()}

    for label in ("hub", "rim"):
        d = detail[label]
        assert d["r_built_mm"] == pytest.approx(arcs[label]["radius"], rel=1e-9), (
            f"{label}: the mesh models a {arcs[label]['radius']:.4f} mm fillet and the "
            f"exporter BUILT {d['r_built_mm']:.4f} mm on {d['n_edges_filleted']} of "
            f"{d['n_edges_found']} edges — the mesh is a model of the request, not of "
            f"the part")
        assert d["n_edges_filleted"] == d["n_edges_found"], (
            f"{label}: {d['n_edges_found'] - d['n_edges_filleted']} corners shipped square")

        assert d["worst_wedge_deg"] == pytest.approx(
            unfilleted_wedges[f"{label}:P_t"], abs=2.0), (
            f"{label}: the manifest fillets a {d['worst_wedge_deg']:.1f} deg corner and "
            f"the mesh's P_t is {unfilleted_wedges[f'{label}:P_t']:.2f} deg — the two "
            f"bodies are rounding different corners")
        assert abs(d["worst_wedge_deg"] - unfilleted_wedges[f"{label}:P_c"]) > 20.0, (
            f"{label}: P_t and P_c have come within 20 deg of each other, so the wedge no "
            f"longer identifies which family the exporter filleted")


def test_the_arc_distance_is_a_distance_to_the_ARC_and_not_to_its_circle(genes):
    """`arc_peak`'s region is a tube round the fillet, and `_distance_to_arc` is what
    makes it one. Checked against four cases whose answers are known by construction
    rather than against stored numbers.

    The failure this rules out is the one that would not look like a failure: dropping the
    sweep test makes the region a tube round the whole CIRCLE, which at the rim is a 3 mm
    ring reaching well into the spoke and the rim band. The peak inside it would be a
    perfectly plausible number, larger than the fillet's, and mesh-dependent in a way the
    fillet's is not.
    """
    cfg = ww.get_config("coarse")
    for label, arc in cs.fillet_arcs(genes, cfg, True).items():
        C, R, a0, a1 = arc["centre"], arc["radius"], arc["a0"], arc["a1"]
        t = np.linspace(a0, a1, 501)
        on = C + R * np.column_stack([np.cos(t), np.sin(t)])
        assert cs._distance_to_arc(on, arc).max() < 1e-12, f"{label}: on-arc"

        off = C + (R + 0.1) * np.column_stack([np.cos(t), np.sin(t)])
        assert cs._distance_to_arc(off, arc) == pytest.approx(0.1, abs=1e-12), label

        # Past the far endpoint, the answer is the chord back to it — NOT zero, which is
        # what a circle-distance would give for a point still on the circle.
        step = np.radians(np.linspace(1.0, 20.0, 20))
        ext = a1 + math.copysign(1.0, a1 - a0) * step
        past = C + R * np.column_stack([np.cos(ext), np.sin(ext)])
        assert cs._distance_to_arc(past, arc) == pytest.approx(
            2.0 * R * np.sin(step / 2.0), abs=1e-12), f"{label}: past the endpoint"

        far = np.array([[-40.0, 0.0]])
        assert cs._distance_to_arc(far, arc)[0] == pytest.approx(
            min(float(np.linalg.norm(far[0] - arc["A"])),
                float(np.linalg.norm(far[0] - arc["B"]))), abs=1e-12), label


# ---------------------------------------------------------------------------
# THE TWO-OBJECTIVE PROFILE QUESTION (FILLET_PLAN PART 16)
# ---------------------------------------------------------------------------

def test_a_genome_robust_layer_profile_holds_the_deflection_band(fillet_report):
    """PART 13's surviving reason, attacked directly — and it turns out to be beatable.

    PART 13 declined the genome-robust profile on two grounds; PART 14 falsified one and
    PART 16 showed the argmax is not stale, so everything rested on this: at
    `GENOME_ROBUST_ENTRY/END` the shipped genome's filleted axle-drop spread over
    `coarse..fine` more than triples and leaves the +-0.3% band.  That was ONE alternative
    pair.  Priced across the whole candidate set — every cell clearing `MIN_SJ_TARGET` on
    the clamped, fold-clean box while refusing none of it — several hold the band on BOTH
    readings of the deflection, and the pair that SHIPS holds only one of them.

    Asserted so that each half fails differently.  If the admissible set empties, the
    promotion PART 17 sequences is off.  If the shipped pair starts holding both, the
    argument that it is not the best-converged profile available has gone away.
    """
    pr = fillet_report["profiles"]
    ship = tuple(pr["shipped_pair"])
    cand = {tuple(p) for p in pr["candidates"]}
    rows = {(r["entry"], r["end"]): r for r in pr["rows"]}
    assert len(cand) >= 20, len(cand)
    assert cand <= set(rows), "a candidate was never priced"

    both = sorted(p for p in cand
                  if rows[p]["inside_band"] and rows[p]["inside_band_patch"])
    assert both, ("no candidate holds both bands — the two-objective profile PART 16 and "
                  "PART 17 record would not exist, which is a finding")
    assert (fb.TWO_OBJECTIVE_ENTRY, fb.TWO_OBJECTIVE_END) in both

    # the shipped pair holds the single-node band and NOT the patch one; that asymmetry
    # is PART 17's re-pricing of PART 12 and it is the reason the promotion is not taken
    # on these statistics
    assert rows[ship]["inside_band"] is True
    assert rows[ship]["inside_band_patch"] is False, (
        "the shipped pair now holds the patch-mean band too — PART 17's re-pricing of "
        "PART 12's 0.141% no longer applies and the records should say so")
    for p in both:
        assert rows[p]["patch_spread_pct"] < rows[ship]["patch_spread_pct"]

    robust = tuple(pr["genome_robust_pair"])
    assert rows[robust]["inside_band"] is False


def test_the_band_is_separating_the_CONTACT_PATCH_and_not_the_fillet(fillet_report):
    """The finding that stops PART 17 from promoting on its own numbers.

    The 1-node spread jumps fourfold between adjacent cells with coarse and medium
    agreeing to 0.0002 mm, and it is not mesh quality, aspect ratio or topology — every
    mesh has the same element count.  It is how many nodes land in the contact patch at
    the finest rung, which the layer profile moves because the fillet re-cuts the rim
    blocks.  Every cell holding both bands reaches the same patch count; every cell
    failing either reaches fewer.

    If that stops being true, the band has started measuring the fillet again and the
    promotion can be settled without `make gci` — so this fails rather than passing.
    """
    pr = fillet_report["profiles"]
    cand = {tuple(p) for p in pr["candidates"]}
    ok, bad = set(), set()
    for r in pr["rows"]:
        if r["spread_pct"] is None or (r["entry"], r["end"]) not in cand:
            continue
        (ok if r["inside_band"] and r["inside_band_patch"] else bad).add(
            r["n_nodes_in_patch"][-1])
    assert ok and bad, (sorted(ok), sorted(bad))
    assert not (ok & bad), (
        f"patch counts {sorted(ok & bad)} appear on both sides — the band is no longer "
        "separated by the contact patch alone, which is a finding")
    assert min(ok) > max(bad), (sorted(ok), sorted(bad))

    # and the patch count really does move up the ladder, or the claim is about nothing
    for r in pr["rows"]:
        if r["patch_count_moves"] is not None:
            assert r["patch_count_moves"], r


def test_the_layer_profile_sweep_moved_the_mesh_it_claims_to_have_moved(fillet_report):
    """A sweep of nine profiles that silently priced one mesh nine times would pass every
    assertion above by reporting identical spreads.  The axle drops must actually differ.
    """
    pr = fillet_report["profiles"]
    drops = [tuple(r["axle_drop_mm"]) for r in pr["rows"] if r["axle_drop_mm"]]
    assert len(set(drops)) == len(drops), "two candidate profiles produced the same ladder"
