"""
Pins for `studies/study_fillet_block.py` — FILLET_PLAN.md STEP 1 RECORD PART 9.

WHY THIS FILE EXISTS.  `make filletblock` retires both of the routes that have stood at
the top of PLAN.md's ranked list for nine arcs, and promotes a third that nobody had
written down.  A conclusion that large has to be re-derivable by whoever doubts it, and
the three claims it rests on are structural rather than numerical:

  1. the region PART 3 named has a ZERO-degree corner at `B`, exactly, at every radius;
  2. the angle that kills route 2 is carried by three BOUNDARY nodes, so no scheme that
     generates an interior can move it;
  3. a block whose corners are OFF both tangent points meshes, and does so across the
     whole gene box.

EVERY TEST HERE RE-MEASURES.  PART 7's lesson is that a committed artifact whose driver
takes a bare default rots silently, because every test that reads it reads the same stale
file — so these build the geometry fresh and compare, and only two of them read
`study_fillet_block.json` at all: the freshness guard, which exists to catch exactly that,
and the self-check pin.

RADII ARE CHOSEN OFF THE COMMITTED GRID on purpose.  A claim that holds at the ten radii
the driver happens to sweep and nowhere else is not the claim being made.
"""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import wheel_genome as wg              # noqa: E402
import wheel_objective as wo           # noqa: E402
import wheel_wheel as ww               # noqa: E402
import study_fillet_block as fb        # noqa: E402
import study_fillet_fold as ff         # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Deliberately not on `study_fillet_block.RADII`, and spanning the gene box: `R_hub` runs
# 0.4-4.0 and `R_rim` 0.5-3.0.
OFF_GRID_RADII = (0.07, 0.33, 0.91, 2.37)


@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


@pytest.fixture(scope="module")
def report():
    with open(os.path.join(REPO, "studies", "study_fillet_block.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def junction_report():
    with open(os.path.join(REPO, "studies",
                           "study_junction_agreement.json")) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# THE CONTROL
# ---------------------------------------------------------------------------

def test_the_driver_never_asks_for_a_filleted_wheel(genes):
    """Nothing in this arc may move the default mesh, and the cheapest way to be sure is
    that the trimmed spoke at zero radius IS the default spoke — bit for bit, not to a
    tolerance.

    It is a real check rather than a tautology: `trimmed_spoke` re-derives the station
    range from `junction_stations` and re-samples, so a change to either would show up
    here as a nonzero difference while `sector_blocks` itself stayed green.
    """
    for cfg in ("coarse", "medium"):
        base = np.asarray(ww.sector_blocks(genes, cfg, fillet=None)["spoke"], float)
        trim = fb.trimmed_spoke(genes, cfg, 0.0, 0.0)
        assert np.abs(base - trim).max() == 0.0, cfg


# ---------------------------------------------------------------------------
# CLAIM 1 — THE REGION PART 3 NAMED IS NOT A BLOCK
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("junction", ("hub", "rim"))
@pytest.mark.parametrize("R", OFF_GRID_RADII)
def test_the_fillet_region_has_a_zero_angle_at_B(genes, junction, R):
    """The corner at `B` is exactly zero, and that is geometry rather than tolerance.

    Both curves meeting at `B` are CIRCLES — the ring circle and the fillet arc — and
    `_fillet_tangency` puts the arc's centre radially above `B` by exactly `R`, so the
    tangency is not solved for, it is constructed.  The measured angle is therefore at
    the level of `acos` near 1, which is why the bound is 1e-6 deg and not 1e-3.
    """
    g = fb.junction_geometry(genes, "coarse", junction, R)
    assert g["tangency"], (junction, R)
    ang = fb.region_angles(g)
    assert ang["at_B_deg"] < 1.0e-6, ang
    assert ang["is_cusp_at_B"]


@pytest.mark.parametrize("junction", ("hub", "rim"))
def test_the_corner_at_A_is_a_cusp_too_and_it_is_the_flank_s_CURVATURE(genes, junction):
    """`A` is under a degree, and the residue is the spline flank's own turn.

    If it were slack in the tangency solve it would grow with the solve's difficulty,
    i.e. with `R`.  It does not: across a 30x span of radius it stays inside a
    two-tenths-of-a-degree band, which is what a curvature term looks like and what a
    convergence residue does not.  This is the test that says "cusp" is a statement about
    the geometry and not about `_fillet_tangency`'s bisection.
    """
    got = []
    for R in OFF_GRID_RADII:
        g = fb.junction_geometry(genes, "coarse", junction, R)
        got.append(fb.region_angles(g)["at_A_deg"])
    assert max(got) < 1.0, got
    assert max(got) - min(got) < 0.25, got


@pytest.mark.parametrize("junction", ("hub", "rim"))
def test_no_quad_block_can_use_this_region(genes, junction):
    """Stated as the thing that actually blocks route 1: two of the three corners are
    unusable, and a tri-block does not help.

    A tri-block SUBDIVIDES a region's corners — its three quads inherit the region's
    three vertices, one each.  So the smallest corner any decomposition of `A - P_t - B`
    can offer is the region's own smallest corner, and that is `B`'s zero.  The assertion
    is on the SUM: 38 + 0.6 + 0 is not a triangle anybody meshes, and the two small ones
    are the two nobody had measured.
    """
    g = fb.junction_geometry(genes, "coarse", junction,
                             float(genes[12] if junction == "hub" else genes[13]))
    a = fb.region_angles(g)
    assert a["at_P_t_deg"] > 30.0
    assert min(a["at_A_deg"], a["at_B_deg"]) < 1.0
    assert sum((a["at_A_deg"], a["at_B_deg"], a["at_P_t_deg"])) < 45.0


# ---------------------------------------------------------------------------
# CLAIM 2 — ROUTE 2 CANNOT REACH THE ANGLE THAT FAILS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("junction", ("hub", "rim"))
def test_the_failing_angle_survives_an_elliptic_interior_solve(genes, junction):
    """Route 2 is "a generated spoke block".  Every such scheme holds the boundary and
    moves the interior; all three nodes carrying this angle are boundary nodes.

    Pinned as an EQUALITY, not a tolerance, because the claim is not "smoothing barely
    helps" — it is that the quantity is out of a smoother's reach by construction.  2000
    Winslow sweeps is far past where the block stops moving.
    """
    R = float(genes[12] if junction == "hub" else genes[13])
    row = fb.moved_corner(genes, "coarse", junction, R)
    assert row["winslow_max_boundary_shift_mm"] == 0.0
    assert row["angle_after_winslow_deg"] == row["angle_deg"]
    assert row["angle_is_a_boundary_quantity"]


def test_PART_3s_collapsed_corner_reproduces_at_coarse(genes):
    """PART 3 recorded 3.601 deg (hub) and 8.524 (rim) with end cross-sections of 2.759
    and 8.596 mm, on the CAPPED mesh of 2026-08-17.

    They reproduce on the uncapped default because none of the four is a `P_c` quantity:
    the fillet is at `P_t`, whose geometry §38's flip left alone to 0.01 deg (PART 7).
    That is worth pinning rather than assuming — it is the one place in this arc where a
    pre-flip number may be quoted forward, and the reason is specific to `P_t`.
    """
    want = {"hub": (3.601, 2.759), "rim": (8.524, 8.596)}
    for junction, (ang, xs) in want.items():
        R = float(genes[12] if junction == "hub" else genes[13])
        row = fb.moved_corner(genes, "coarse", junction, R)
        assert abs(row["angle_deg"] - ang) < 5e-3, (junction, row["angle_deg"])
        assert abs(row["end_cross_section_mm"] - xs) < 5e-3, junction


def test_the_node_angle_moves_with_the_config_but_the_curve_angle_does_not(genes):
    """The 3.601 deg above is a SAMPLED angle — corner to its two neighbours — so it
    changes when the neighbours do.  At `medium` the same corner reads 10.5.

    The quantity that decides whether a construction is viable is the angle between the
    two boundary CURVES, and that one is a property of the geometry: it agrees between
    the two configs to under a thousandth of a degree.  Both are reported by the driver so
    nobody has to work out which of the two a quoted number was.
    """
    for junction in ("hub", "rim"):
        R = float(genes[12] if junction == "hub" else genes[13])
        c = fb.moved_corner(genes, "coarse", junction, R)
        m = fb.moved_corner(genes, "medium", junction, R)
        assert abs(m["angle_deg"] - c["angle_deg"]) > 1.0, junction
        assert abs(m["tangent_angle_deg"] - c["tangent_angle_deg"]) < 1e-3, junction


# ---------------------------------------------------------------------------
# THE SPOKE WAS NEVER THE BLOCKER
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("R", OFF_GRID_RADII)
def test_the_spoke_ended_at_the_tangent_station_is_clean(genes, R):
    """PART 3's headline — "the spoke block is ruled, and a fillet 1.3-5.8x the wall
    cannot be absorbed by ruling" — is about where the fillet was PUT, not about ruling.

    Move the arc off the spoke's flank edge and end the spoke at `s_A`, and the same
    ruled block is clean, at radii the shipped construction folds at by 10x.  Checked
    under criterion C, `det J` at the Gauss points, because that is the criterion §44
    settled on.
    """
    for cfg in ("coarse", "medium"):
        grid = fb.trimmed_spoke(genes, cfg, R, float(genes[13]))
        assert grid is not None, (cfg, R)
        q = fb.block_quality(grid)
        assert q["mixed_sign_cells"] == 0, (cfg, R, q)
        assert q["non_positive_gauss_elements"] == 0, (cfg, R, q)


def test_the_shipped_construction_folds_where_the_trimmed_spoke_does_not(genes):
    """The contrast stated as one assertion, at the radii that ship.

    `make fillet`'s usable window for the shipped `fillet=` path is 0.12-0.24 mm at
    `coarse`; the genome carries 0.6636 and 3.0.  So the shipped path must fold here and
    the trimmed spoke must not — if either half ever stops being true, the re-diagnosis
    in PART 9 is what needs re-reading.
    """
    R_hub, R_rim = float(genes[12]), float(genes[13])
    shipped = np.asarray(ww.sector_blocks(genes, "coarse",
                                          fillet=(R_hub, R_rim))["spoke"], float)
    assert ff.gauss_verdict(shipped)["non_positive_elements"] > 0
    trimmed = fb.trimmed_spoke(genes, "coarse", R_hub, R_rim)
    assert ff.gauss_verdict(trimmed)["non_positive_elements"] == 0


# ---------------------------------------------------------------------------
# CLAIM 3 — THE BLOCK THAT DOES WORK
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("junction", ("hub", "rim"))
@pytest.mark.parametrize("R", OFF_GRID_RADII)
def test_the_boundary_layer_block_meshes_off_grid_too(genes, junction, R):
    """The one that works, at radii the committed sweep does not contain, and against the
    optimizer's own floor rather than against zero.

    `wheel_objective.MIN_SJ_TARGET` is 0.2 with a barrier weight of 3000 — the number
    §38 measured the faithful rim's junction block collapsing under (0.0072).  This block
    clears it by more than 4x everywhere tested, which is the difference between "does
    not fold" and "is a block the solver would be happy with".
    """
    cfgo = ww.get_config("coarse")
    g = fb.junction_geometry(genes, "coarse", junction, R)
    assert g["tangency"]
    grid = fb.candidate_boundary_layer(g, cfgo.nn(cfgo.n_weld), cfgo.nn(cfgo.n_thick))
    q = fb.block_quality(grid)
    assert q["valid"], (junction, R, q)
    assert q["min_scaled_jacobian"] > wo.MIN_SJ_TARGET, (junction, R, q)


@pytest.mark.parametrize("junction", ("hub", "rim"))
def test_the_two_candidates_that_keep_a_corner_on_a_tangency_get_WORSE_when_refined(
        genes, junction):
    """A construction whose fold shrinks under refinement is a resolution problem.  One
    whose fold GROWS has the wrong region, and that is the distinction this pins.

    Both failing candidates keep a block edge running along a pre-fillet surface through
    a tangent point, so both inherit the cusp; refining them puts more cells inside a
    region that pinches to zero width, and the count rises.  The working candidate is
    checked in the same loop for the opposite behaviour.
    """
    cfgo = ww.get_config("coarse")
    n_th, n_weld = cfgo.nn(cfgo.n_thick), cfgo.nn(cfgo.n_weld)
    R = float(genes[12] if junction == "hub" else genes[13])
    g = fb.junction_geometry(genes, "coarse", junction, R)
    for name in ("grown_junction", "pre_fillet_surfaces"):
        counts = []
        for m in (1, 2, 4):
            grid = fb.CANDIDATE_FN[name](g, (n_weld - 1) * m + 1, n_th)
            counts.append(fb.block_quality(grid)["mixed_sign_cells"])
        assert counts[0] > 0, (name, counts)
        assert counts[-1] > counts[0], (name, counts)
    for m in (1, 2, 4):
        grid = fb.candidate_boundary_layer(g, (n_weld - 1) * m + 1, n_th)
        assert fb.block_quality(grid)["valid"], (m, junction)


@pytest.mark.parametrize("junction", ("hub", "rim"))
def test_an_elliptic_interior_does_not_rescue_either_failing_candidate(genes, junction):
    """Route 2's technique, offered to route 1's candidates.  It is the fair test: if a
    generated interior fixed these, the answer would be "build both", not "neither".

    It does not, and it cannot, for the same reason it cannot help route 2 — the fold is
    forced by a boundary that pinches, and a smoother holds the boundary.
    """
    cfgo = ww.get_config("coarse")
    n_th, n_weld = cfgo.nn(cfgo.n_thick), cfgo.nn(cfgo.n_weld)
    R = float(genes[12] if junction == "hub" else genes[13])
    g = fb.junction_geometry(genes, "coarse", junction, R)
    for name in ("grown_junction", "pre_fillet_surfaces"):
        grid = fb.CANDIDATE_FN[name](g, n_weld, n_th)
        assert not fb.block_quality(fb.winslow(grid, 500))["valid"], name


@pytest.mark.parametrize("junction", ("hub", "rim"))
def test_offsetting_the_inner_edge_the_OTHER_way_folds(genes, junction):
    """The mutation that proves the docstring's claim about which way the offset goes.

    `candidate_boundary_layer` offsets `j1` ALONG `arc - C`, i.e. away from the fillet's
    centre, so the inner edge is a concentric arc of radius `R + w` and cannot cusp.
    Offset toward the centre instead and the inner edge is radius `R - w`; once `w`
    passes `R` it turns itself inside out.  Re-derived here rather than asserted in
    prose, because it is the one design choice in that block that is not forced by a
    seam.
    """
    cfgo = ww.get_config("coarse")
    n_th, n_weld = cfgo.nn(cfgo.n_thick), cfgo.nn(cfgo.n_weld)
    R = float(genes[12] if junction == "hub" else genes[13])
    g = fb.junction_geometry(genes, "coarse", junction, R)
    i0 = fb._cross_section(g, g["s_A"], n_th, g["A"])
    wall = float(np.linalg.norm(i0[-1] - g["A"]))
    j0 = np.asarray(ww._arc_between(g["C"], g["A"], g["B"], n_weld), float)
    nrm = j0 - np.asarray(g["C"], float)[None, :]
    nrm = nrm / np.linalg.norm(nrm, axis=1)[:, None]
    w = np.linspace(wall, fb.cut_depth_mm(g, wall), n_weld)[:, None]
    j1 = j0 - w * nrm                                  # toward the centre: the mutation
    j1[0] = i0[-1]
    i1 = ww._lerp_points(g["B"], j1[-1], n_th, np)
    assert not fb.block_quality(ww.coons_patch(j0, j1, i0, i1, xp=np))["valid"]


# ---------------------------------------------------------------------------
# THE PRICE, AND THE CROSS-CHECK AGAINST `make junction`
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("junction", ("hub", "rim"))
def test_the_working_block_cuts_ACROSS_the_ring_circle(genes, junction):
    """This is the block's price and it must not be quietly droppable.

    Its inner edge ends INSIDE the collar (hub) or the band (rim), which is exactly why
    it does not degenerate at `B` — and exactly why the ring block has to be notched.  If
    a later change makes `cut_depth_mm` return zero the block goes back to having a
    corner on a tangency, so the depth is pinned as strictly positive and as a real
    fraction of what the ring has to give.
    """
    cfgo = ww.get_config("coarse")
    n_th = cfgo.nn(cfgo.n_thick)
    R = float(genes[12] if junction == "hub" else genes[13])
    g = fb.junction_geometry(genes, "coarse", junction, R)
    wall = float(np.linalg.norm(fb._cross_section(g, g["s_A"], n_th, g["A"])[-1]
                                - g["A"]))
    d = fb.cut_depth_mm(g, wall)
    assert 0.0 < d < g["depth_available_mm"]
    grid = fb.candidate_boundary_layer(g, cfgo.nn(cfgo.n_weld), n_th)
    past = (np.linalg.norm(grid[-1, -1]) - g["ring_r"]) * g["void_sign"]
    assert past < 0.0, past          # the cut's far end is on the ring block's side


@pytest.mark.parametrize("junction", ("hub", "rim"))
def test_make_junction_s_void_is_a_ONE_NODE_CHORD_and_it_reproduces(
        genes, junction, junction_report):
    """`make junction`'s `void_deg` at `P_t` — the number PART 8's re-pricing rests on —
    is the angle to the spoke block's SECOND flank node, not to the flank's tangent.

    Reproduced here to the digit from the committed junction artifact, and reported next
    to the tangent so the 0.8 deg (hub) / 0.6 (rim) gap is on the record rather than
    surfacing later as a disagreement between two files in the same arc.  No verdict in
    PART 8 moves on it: both `P_t` rows clear by 5-20x, and the `P_c` rows are unaffected
    because under `uncap` that corner's leg is a straight continuation whose chord and
    tangent are the same direction.
    """
    row = next(c for c in junction_report["rings"][junction]["corners"]
               if c["name"] == "P_t" and c["source"] == "mesh (uncap=False)")
    g = fb.junction_geometry(genes, "coarse", junction,
                             float(genes[12] if junction == "hub" else genes[13]))
    a = fb.region_angles(g)
    assert abs(a["at_P_t_chord_deg"] - row["void_deg"]) < 1e-9, (
        a["at_P_t_chord_deg"], row["void_deg"])
    assert 0.4 < a["chord_minus_tangent_deg"] < 1.0, a["chord_minus_tangent_deg"]


# ---------------------------------------------------------------------------
# THE ARTIFACT
# ---------------------------------------------------------------------------

def test_the_committed_report_passes_its_own_self_checks(report):
    """`make filletblock` exits nonzero on these three, so a committed artifact that
    fails one is an artifact written by a run that should not have been filed."""
    assert report["self_checks"]["pass"], report["self_checks"]


def test_the_committed_report_describes_the_geometry_the_tree_BUILDS_TODAY(genes,
                                                                          report):
    """PART 7's guard, applied here from the start rather than after the fact.

    `study_corner_singularity.json` went four days out of date because its driver took a
    bare `UNCAP_DEFAULT` and every test that read it read the same stale file.  This
    rebuilds the cheapest slice — the region angles at the shipped hub radius — and
    compares it against the committed numbers, so a change to `sector_blocks`,
    `global_sampler` or the uncap default goes red HERE instead of being quoted forward.
    """
    R = float(genes[12])
    g = fb.junction_geometry(genes, "coarse", "hub", R)
    fresh = fb.region_angles(g)
    row = next(r for r in report["region"]["coarse"]["hub"]
               if abs(r["radius_mm"] - R) < 1e-9)
    for key in ("at_A_deg", "at_P_t_deg", "at_B_deg", "leg_flank_mm",
                "leg_ring_chord_mm", "at_P_t_chord_deg"):
        assert abs(fresh[key] - row[key]) < 1e-9, (key, fresh[key], row[key])


def test_the_report_records_that_neither_of_PART_3s_routes_survived(report):
    """The finding itself, pinned so that a future edit cannot soften it by accident.

    Both failing candidates must still be failing and the working one must still be
    working, at the shipped radii, at both junctions, at both configs — that is the whole
    of PART 9's verdict and it is four booleans.
    """
    for cfg in report["configs"]:
        for junction in report["junctions"]:
            ship = report["shipped_radii_mm"][junction]
            row = next(r for r in report["candidates"][cfg][junction]
                       if abs(r["radius_mm"] - ship) < 1e-9)
            for name in ("grown_junction", "pre_fillet_surfaces"):
                assert not any(q["valid"] for q in row["blocks"][name].values()
                               if q.get("built")), (cfg, junction, name)
            assert all(q["valid"] for q in row["blocks"]["boundary_layer"].values()
                       if q.get("built")), (cfg, junction)


def test_a_degraded_run_may_not_be_filed_as_the_committed_artifact():
    """§43's guard reaches this driver too — the tenth.

    Its degradation axes are its configs, its junctions and its genome: a run at one
    config or on a probe genome answers a smaller question than the committed artifact
    does, and every field would still be present.
    """
    import argparse
    ap = argparse.ArgumentParser()
    args = argparse.Namespace(out="study_fillet_block.json")
    with pytest.raises(SystemExit):
        fb._gate_guard.refuse_degraded_out(
            ap, args, "study_fillet_block.json", [(True, "a degraded probe run")])
