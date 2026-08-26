"""
Pins for `studies/study_fillet_block.py` — FILLET_PLAN.md STEP 1 RECORD PARTS 9 and 10.

WHY THIS FILE EXISTS.  `make filletblock` retires both of the routes that have stood at
the top of PLAN.md's ranked list for nine arcs, and promotes a third that nobody had
written down.  A conclusion that large has to be re-derivable by whoever doubts it, and
the three claims it rests on are structural rather than numerical:

  1. the region PART 3 named has a ZERO-degree corner at `B`, exactly, at every radius;
  2. the angle that kills route 2 is carried by three BOUNDARY nodes, so no scheme that
     generates an interior can move it;
  3. a block whose corners are OFF both tangent points meshes, and does so across the
     whole gene box.

STEP 1a ADDS A FOURTH, AND IT IS THE ONE THAT MATTERS FOR STEP 1b: the eleven blocks and
fourteen seams of the WHOLE filleted sector close, every seam whole-edge.  A block that
meshes is not a mesh — the fillet block's inner edge crosses the ring circle, so it has
two partners unless it is split, and the ring blocks it lands in have to close as quads
with node counts that agree.  Those tests build the sector fresh and check both halves of
"closes": the counts, and the coordinates.

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
    shipped = np.asarray(ww.sector_blocks(genes, "coarse", fillet=(R_hub, R_rim),
                                          fillet_blocking="spoke")["spoke"], float)
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
# THE WHOLE SECTOR: EVERY BLOCK, EVERY SEAM  (STEP 1a)
# ---------------------------------------------------------------------------
#
# A block that meshes is not a mesh.  These pin the BLOCKING — that the eleven blocks
# close into a sector whose every seam is whole-edge — because that, and not the single
# block PART 9 measured, is what STEP 1b would be wiring into `build_wheel`.

SECTOR_CFGS = ("coarse", "medium")


@pytest.mark.parametrize("cfg", SECTOR_CFGS)
def test_every_seam_of_the_filleted_sector_is_WHOLE_EDGE(genes, cfg):
    """No side of any block may appear in the seam table twice.

    This is the one structural rule the arc committed to before building anything, and
    it is the rule a partial-edge seam breaks: `_seam_table`'s docstring calls whole-edge
    single ownership "the whole safety net", and a mechanism that lets one edge have two
    partners can mis-pair silently while every Jacobian stays positive.  Asserted on the
    TABLE rather than on the geometry, because the table is what STEP 1b copies.
    """
    orientation = ww.flank_orientation(genes, ww.get_config(cfg),
                                       span_mm=ww.HUB_RIM_SPAN_MM)
    _, info = fb.filleted_sector(genes, cfg, float(genes[12]), float(genes[13]))
    dirn = info["dirn"]
    seen = {}
    for a, sa, b, sb, dk, _rev in ww._seam_table_filleted(orientation, dirn):
        for block, side, shift in ((a, sa, 0), (b, sb, dk)):
            key = (block, side)
            assert key not in seen, (
                f"{block}.{side} is claimed by two seams — that is a partial-edge seam "
                f"in all but name: {seen[key]} and {(a, sa, b, sb, dk)}")
            seen[key] = (a, sa, b, sb, dk)
        assert shift in (-1, 0, 1)


@pytest.mark.parametrize("cfg", SECTOR_CFGS)
def test_the_filleted_sector_closes_at_the_shipped_radii(genes, cfg):
    """Built fresh, not read: every seam's node counts agree and its nodes coincide.

    The tolerance is `SEAM_TOL_MM` (1e-9 mm) and the measurement comes in at 1e-14, so
    this is not a tolerance being tuned to pass — it is round-off against a bound four
    orders wider.
    """
    v = fb.sector_verdict(genes, cfg, float(genes[12]), float(genes[13]))
    assert v["built"], v.get("why")
    assert v["n_blocks"] == 11 and v["n_seams"] == 14
    for s in v["seams"]:
        assert s["counts_agree"], (s["a"], s["side_a"], s["b"], s["side_b"],
                                   s["n_a"], s["n_b"])
        assert s["closes"], (s["a"], s["side_a"], s["b"], s["side_b"], s["max_gap_mm"])
    assert v["max_seam_gap_mm"] < 1e-12


@pytest.mark.parametrize("cfg", SECTOR_CFGS)
def test_every_block_of_the_filleted_sector_INTEGRATES(genes, cfg):
    """§44's criterion, applied to all eleven: `det J` at the Gauss points, not "it built".

    And against `MIN_SJ_TARGET`, because a mesh the optimizer's barrier would reject is
    not a mesh this arc can hand to STEP 2.
    """
    v = fb.sector_verdict(genes, cfg, float(genes[12]), float(genes[13]))
    assert v["built"], v.get("why")
    for name, q in v["blocks"].items():
        assert q["non_positive_gauss_elements"] == 0, (name, q)
        assert q["mixed_sign_cells"] == 0, (name, q)
        assert q["min_scaled_jacobian"] > wo.MIN_SJ_TARGET, (name, q)


@pytest.mark.parametrize("R_hub,R_rim", [(0.40, 0.50), (0.91, 2.37), (3.00, 3.00)])
def test_the_sector_closes_OFF_the_committed_grid_too(genes, R_hub, R_rim):
    """Including at the gene box's own two floors, which is where the blocking is worst.

    `R_hub = 0.4` is the floor, and it is the cell that ruled out the alternative inner
    edge: an offset carried to the ring's full depth folds the weld block there, because
    an offset of `w >> R` is a spiral of radius `R + w` about the arc's centre.
    """
    v = fb.sector_verdict(genes, "coarse", R_hub, R_rim)
    assert v["built"], v.get("why")
    assert v["all_blocks_valid"] and v["seams_close"], v["worst_block"]
    assert v["min_scaled_jacobian"] > wo.MIN_SJ_TARGET


@pytest.mark.parametrize("junction", ("hub", "rim"))
def test_the_cut_at_B_reaches_the_rings_FAR_boundary_exactly(genes, junction):
    """Not a depth that was tuned: the far boundary, to round-off.

    It has to be exact, because that point is a corner of BOTH ring blocks and of the
    fillet block, and a cut that lands 1e-6 mm short of the bore leaves a sliver that no
    seam declares.
    """
    R = float(genes[12] if junction == "hub" else genes[13])
    c = fb.sector_curves(genes, "coarse", junction, R)
    assert c["built"], c.get("why")
    r_far = fb.far_boundary_radius(junction)
    assert abs(float(np.linalg.norm(c["L"])) - r_far) < 1e-12
    # and the dive is radial: L, N and the origin are colinear
    cross = float(c["N"][0] * c["L"][1] - c["N"][1] * c["L"][0])
    assert abs(cross) < 1e-9, cross


@pytest.mark.parametrize("junction", ("hub", "rim"))
def test_the_SHALLOW_cut_lands_tangent_which_is_why_it_cannot_close(genes, junction):
    """PART 9's own block, re-measured for the reason it cannot be the sector's.

    Its inner edge is a concentric offset of an arc TANGENT to the ring circle, so it is
    tangent to every circle concentric with that one — and the block cornered between the
    two is a sliver.  Pinned as a bound rather than a value: whatever the width profile
    does, the residual has to stay small enough that the sliver is unusable, and at the
    rim it is under `MIN_SJ_TARGET` outright.
    """
    row = fb.landing_angles(genes, "coarse", (junction,))[junction]
    assert row["landing_angle_deg"] < 15.0, row
    if junction == "rim":
        assert row["sliver_scaled_jacobian"] < wo.MIN_SJ_TARGET, row


def test_the_SECTOR_bounds_the_hub_radius_before_the_BLOCK_does(genes):
    """The limit moved, and it is worth knowing which limit it is.

    PART 9 measured the boundary-layer block clean out to 4.00 mm and it still is.  What
    runs out first is the ring's FREE block: past ~3.13 mm the fillet's tangent point has
    swept past the next sector's corner, and there is no free ring left to block.  A
    geometric statement about the sector, not a mesh-quality one — so both halves are
    asserted here, in one test, so neither can be quoted without the other.
    """
    lim = fb.sector_fit_limit(genes, "coarse", "hub")
    assert lim["limited"]
    assert 3.0 < lim["radius_mm"] < 3.3, lim
    # the BLOCK alone, at a radius the sector cannot take
    g = fb.junction_geometry(genes, "coarse", "hub", 4.0)
    n_th = ww.get_config("coarse").nn(ww.get_config("coarse").n_thick)
    q = fb.block_quality(fb.candidate_boundary_layer(g, 2 * n_th - 1, n_th))
    assert q["valid"] and q["min_scaled_jacobian"] > wo.MIN_SJ_TARGET, q
    assert fb.sector_verdict(genes, "coarse", 4.0, 3.0)["built"] is False


@pytest.mark.parametrize("cfg", SECTOR_CFGS)
def test_the_ring_blocks_radial_count_is_FORCED_to_n_thick(genes, cfg):
    """The node-count coupling, measured on the blocks rather than described.

    The cut at `B` carries `n_thick` nodes and it is the ring free block's left edge, so
    that block's radial count is `n_thick`; its right edge is the next sector's weld
    block's left edge, so the weld's is too.  `n_collar_r` and `n_rim_r` are therefore
    NOT what the filleted ring uses, and at `coarse` they are 7 against `n_thick`'s 9 —
    which is exactly why `fillet=None` needs its own path rather than a flag.
    """
    cfgo = ww.get_config(cfg)
    n_th = cfgo.nn(cfgo.n_thick)
    blocks, info = fb.filleted_sector(genes, cfg, float(genes[12]), float(genes[13]))
    assert blocks is not None, info["why"]
    for name in ("hub_ring_weld", "hub_ring_free", "rim_ring_weld", "rim_ring_free"):
        assert blocks[name].shape[1] == n_th, (name, blocks[name].shape, n_th)
    assert cfgo.nn(cfgo.n_collar_r) != n_th and cfgo.nn(cfgo.n_rim_r) != n_th, (
        "the coupling stopped being a coupling — n_collar_r now equals n_thick, so "
        "this test no longer says anything")


@pytest.mark.parametrize("cfg", SECTOR_CFGS)
def test_the_ring_blocks_keep_the_SHIPPED_radial_order(genes, cfg):
    """`_edge_sets` and `_node_sets` name three boundary sets by SIDE, not by radius.

    `hub_tie` is `j0`, `rim_outer` is `j1`, `rim_inner_free` is `j0` — and they are named
    off `hub_collar_*` running bore -> ring circle while `rim_band_*` runs ring circle ->
    tyre surface.  Laying both rings out the same way round in the filleted blocking
    would be tidier to write and would move three boundary sets in STEP 1b without
    anything going red, because a set of the wrong radius is still a set.  Pinned on the
    COORDINATES, so it cannot be satisfied by a comment.
    """
    blocks, info = fb.filleted_sector(genes, cfg, float(genes[12]), float(genes[13]))
    assert blocks is not None, info["why"]
    expect = {"hub": (ww.HUB_RADIUS_MM - ww.COLLAR_DEPTH_MM, ww.HUB_RADIUS_MM),
              "rim": (ww.rim_inner_radius(ww.HUB_RIM_SPAN_MM), ww.RIM_OUTER_RADIUS_MM)}
    for junction, (r_j0, r_j1) in expect.items():
        for kind in ("weld", "free"):
            b = blocks[f"{junction}_ring_{kind}"]
            for side, want in (("j0", r_j0), ("j1", r_j1)):
                got = np.linalg.norm(fb._side(b, side), axis=1)
                assert np.allclose(got, want, atol=1e-9), (
                    junction, kind, side, float(got.min()), float(got.max()), want)


def test_the_entry_slope_is_what_keeps_the_junction_block_open(genes):
    """The mechanism behind `LAYER_ENTRY_SLOPE`, not just its value.

    Three blocks meet where the fillet block's inner edge leaves the far flank, and 180
    degrees has to be shared between two of them.  At entry 0 the inner edge leaves
    tangent to the flank — which is the junction block's own top edge — and the junction
    block is a cusp.  Re-measured: the chosen slope must beat it by a wide margin, and
    the block that improves must be the junction.
    """
    ship = (float(genes[12]), float(genes[13]))
    chosen = fb.sector_verdict(genes, "coarse", *ship)
    flat = fb.sector_verdict(genes, "coarse", *ship, entry=0.0,
                             end=fb.LAYER_END_OFFSET)
    assert flat["built"] and chosen["built"]
    assert flat["blocks"]["hub_junction"]["min_scaled_jacobian"] < 0.05, flat
    assert chosen["blocks"]["hub_junction"]["min_scaled_jacobian"] > 10.0 * flat[
        "blocks"]["hub_junction"]["min_scaled_jacobian"]


@pytest.fixture(scope="module")
def genome_sweep():
    """One feasible genome per flank orientation, re-drawn rather than read."""
    return fb.sweep_genomes("coarse", per_orientation=1)


def test_the_sector_closing_seam_FOLLOWS_the_flank_orientation(genome_sweep):
    """The `dk` bug, reproduced and then fixed, on a genome that actually flips.

    `sector_blocks` lays both ring blocks out in INCREASING theta whatever the genome
    does, precisely so that "the next sector" is always `k + 1`.  This blocking lays each
    ring out from `theta_Q` toward the fillet, so the sector-closing seam runs to
    `k + dirn` — and `flank_orientation`'s own docstring records that only 16 of 60
    feasible genomes share the shipped one's `(+1, +1)`.  Written as `dk = +1` the seam
    closes for the shipped genome and misses by a WHOLE SECTOR for a flipped one.  Both
    halves are asserted, because a test that only checks the fixed behaviour cannot tell
    you the bug was ever possible.
    """
    flipped = [r for rows in genome_sweep["groups"].values() for r in rows
               if r["built"] and min(r["dk"].values()) < 0]
    if not flipped:
        pytest.skip("no flipped-orientation genome in this draw")
    row = flipped[0]
    vec = np.asarray(row["genes"], float)
    blocks, info = fb.filleted_sector(vec, "coarse", float(vec[12]), float(vec[13]))
    assert blocks is not None, info["why"]
    orientation = ww.flank_orientation(vec, ww.get_config("coarse"),
                                       span_mm=ww.HUB_RIM_SPAN_MM)
    good = fb.sector_seams(blocks, orientation, info["dirn"])
    assert all(x["closes"] for x in good), [x for x in good if not x["closes"]]
    bad = fb.sector_seams(blocks, orientation, {"hub": 1.0, "rim": 1.0})
    opened = [x for x in bad if not x["closes"]]
    assert opened, "dk = +1 no longer opens a seam — this genome does not flip after all"
    assert max(x["max_gap_mm"] for x in opened) > 1.0, (
        "the miss is supposed to be a whole sector, not a tolerance")


def test_the_blocking_is_measured_at_ONE_genome_and_the_others_are_worse(genome_sweep):
    """The scope, pinned so "48/48 across the box" cannot be quoted as "works everywhere".

    The radius sweep holds the centreline fixed, and the centreline is what decides both
    the flank orientation and how much room the fillet has in the sector.  Re-measured on
    freshly drawn feasible genomes whose UNFILLETED sector is clean: some REFUSE outright
    — the hub fillet's tangent point has swept past the next sector's corner — and of
    those that build, not all clear the optimizer's barrier.  That is the difference
    between a blocking fit for STEP 2 (one genome, one mesh) and one fit for the
    optimizer, and it is a finding rather than a failure.
    """
    rows = [r for v in genome_sweep["groups"].values() for r in v]
    assert len(rows) >= 3, rows
    built = [r for r in rows if r["built"]]
    assert all(r["seams_close"] for r in built)
    refused = [r for r in rows if not r["built"]]
    clears = [r for r in built if r["min_scaled_jacobian"] > wo.MIN_SJ_TARGET]
    assert refused or len(clears) < len(built), (
        "every drawn genome now both fits the sector and clears the barrier — the "
        "blocking may have become genome-robust, in which case this test and the "
        "record that ranks STEP 1b behind it both need re-deriving")
    # PART 13 closed the quality half of this gap (the layer profile is now derived
    # against genomes, not the shipped one alone) without touching the refusal half —
    # the hub sector-fit limit does not depend on `entry`/`end` at all.  So this
    # assertion still holds, and it now holds for a different reason than PART 10's.


def test_a_spoke_fold_genome_does_not_move_with_the_layer_profile(report):
    """The one drawn genome `sweep_layer_profile_genomes` excludes, and why.

    Its worst block is the TRIMMED SPOKE, which is `sample(s_grid, eta_grid)` directly —
    built before `entry`/`end` are ever consulted — so no choice of either constant can
    rescue it.  PART 13 found it by accident while re-deriving the profile: a genome
    whose UNFILLETED sector reads clean (its 97-station grid over `[0, 1]` steps over a
    near self-intersection in the flank near `s = 0.05`) has its trim boundary land
    right where the pathology is, because the trimmed grid's stations are spaced over
    `[s_A(hub), s_A(rim)]` instead.  Confirmed rather than assumed: the verdict is
    recomputed at three very different profiles and the spoke block's own min scaled
    Jacobian does not move at all.
    """
    rows = [r for v in report["sector"]["genomes"]["groups"].values() for r in v]
    fold = [r for r in rows if r.get("built") and r["worst_block"] == "spoke"]
    if not fold:
        pytest.skip("no spoke-fold genome in the committed draw")
    row = fold[0]
    genes_vec = np.asarray(row["genes"], float)
    Rh, Rr = row["R_hub_mm"], row["R_rim_mm"]
    sj = set()
    for entry, end in ((0.0, 1.60), (-0.45, 1.60),
                       (fb.LAYER_ENTRY_SLOPE, fb.LAYER_END_OFFSET)):
        v = fb.sector_verdict(genes_vec, "coarse", Rh, Rr, entry, end)
        assert v["built"], v
        sj.add(round(v["blocks"]["spoke"]["min_scaled_jacobian"], 12))
    assert len(sj) == 1, sj
    assert next(iter(sj)) < 0.0, sj


def test_the_genome_diverse_profile_clears_the_barrier_except_the_flank_defect_genome(
        report):
    """The improvement PART 13 measured, pinned against the committed genomes.

    NOT the shipped `LAYER_ENTRY_SLOPE`/`LAYER_END_OFFSET` — those are PART 10's
    single-genome argmax and this is the whole reason PART 13 exists: at that pair, most
    of the drawn box sits under `MIN_SJ_TARGET` (`rim_ring_free`, mostly).  At
    `GENOME_ROBUST_ENTRY`/`GENOME_ROBUST_END` instead, every BUILT genome in the
    committed draw clears it except the one whose own trimmed spoke folds regardless of
    the profile — that one is `test_a_spoke_fold_genome_does_not_move_with_the_layer_
    profile`'s genome and is excluded here for the same reason
    `sweep_layer_profile_genomes` excludes it: it would report the same floor at every
    cell and hide the result being pinned here.  The pair is measured and reported, not
    adopted as the module default — see `WW._fillet_curves`'s docstring for why not.
    """
    rows = [r for v in report["sector"]["genomes"]["groups"].values() for r in v]
    built = [r for r in rows if r["built"] and r["worst_block"] != "spoke"]
    assert len(built) >= 5, built
    for row in built:
        genes_vec = np.asarray(row["genes"], float)
        v = fb.sector_verdict(genes_vec, "coarse", row["R_hub_mm"], row["R_rim_mm"],
                              fb.GENOME_ROBUST_ENTRY, fb.GENOME_ROBUST_END)
        assert v["min_scaled_jacobian"] > wo.MIN_SJ_TARGET, (
            row["R_hub_mm"], row["R_rim_mm"], v)


def test_the_hub_margin_PREDICTS_the_refusal_rather_than_explaining_it(report):
    """The refusal turned into a number, and the number checked as a classifier.

    PART 10 FINDING 6 counted six refusals of sixteen and named the mechanism -- the hub
    fillet's tangent point had swept past the next sector's corner.  A named mechanism is
    not the same as a predictor, and this is the difference: `sector_fit_margin` is
    computed from the geometry ALONE, before any block is attempted, and it must classify
    the committed draw exactly.  Re-derived here rather than read, because a margin that
    agreed with the artifact but not with the code would be the failure worth catching.
    """
    rows = [r for v in report["sector"]["genomes"]["groups"].values() for r in v]
    assert len(rows) == 16, len(rows)
    for row in rows:
        fit = fb.sector_fit_margin(np.asarray(row["genes"], float), "coarse")
        assert fit["hub"]["binds"] == row["fit"]["hub"]["binds"]
        assert fit["hub"]["binds"] is not row["built"], (
            f"the hub margin misclassifies R_hub={row['R_hub_mm']:.4f}: "
            f"binds={fit['hub']['binds']} built={row['built']}")
    assert sum(1 for r in rows if not r["built"]) == 6, (
        "the committed draw's refusal count moved — a finding, not a pass")


def test_clamping_to_the_sector_fit_limit_closes_the_REFUSAL_half(report):
    """The fix, and the two halves of PLAN.md item 2 kept apart.

    Clamping each radius inside the room its own sector has makes every drawn genome
    build; it does NOT make them all clear the barrier, and those are the refusal half and
    the quality half.  Both are asserted, so that a run in which the clamp appeared to
    solve the whole item would register as a finding rather than as a pass.

    And the clamp only counts as free because it is inert on the shipped genome -- every
    other number in this file is measured there.
    """
    fc = report["sector"]["per_config"]["coarse"]["fit_clamp"]
    assert fc["shipped_is_clamped"] is False
    assert fc["shipped_fit"]["hub"]["binds"] is False

    def row(profile, factor):
        return next(r for r in fc["rows"]
                    if r["profile"] == profile and r["factor"] == factor)

    base = row("shipped", None)
    clamped = row("shipped", fb.SECTOR_FIT_CLAMP)
    assert base["n_built"] < base["n_genomes"], "nothing refused — that is a finding"
    assert clamped["n_built"] == clamped["n_genomes"]
    assert clamped["n_clears_target"] > base["n_clears_target"]
    assert clamped["n_clears_target"] < clamped["n_genomes"], (
        "the clamp now clears the barrier everywhere at the SHIPPED profile — the "
        "quality half of item 2 would be closed too, which is a finding")

    # and the clamp is insensitive to its own factor, or it would be a tuned constant
    for f in fc["factors"]:
        assert row("shipped", f)["n_built"] == clamped["n_genomes"]


def test_the_clamp_re_prices_the_layer_profile_PART_13_declined(report):
    """PART 13's premise, re-checked: it declined the profile partly because of a fact.

    "the hub sector-fit refusal is untouched by either choice, so six of sixteen feasible
    genomes still refuse outright regardless" -- so nothing would collect what the
    genome-robust profile bought.  With the clamp that premise is false, and the prize is
    much larger than it was weighed against.  This pins the re-pricing, and pins that the
    profile is STILL not adopted, because PART 13's OTHER reason (the shipped genome's
    deflection-convergence spread) is untouched by the clamp.
    """
    fc = report["sector"]["per_config"]["coarse"]["fit_clamp"]

    def row(profile, factor):
        return next(r for r in fc["rows"]
                    if r["profile"] == profile and r["factor"] == factor)

    k = fb.SECTOR_FIT_CLAMP
    assert row("genome_robust", None)["n_built"] == row("shipped", None)["n_built"], (
        "the profile does not change WHETHER a genome fits its sector, only how well "
        "its blocks mesh — if that stopped being true the two halves have coupled")
    assert row("genome_robust", k)["n_clears_target"] > row("shipped", k)["n_clears_target"]
    assert (row("genome_robust", k)["n_clears_target"]
            > row("genome_robust", None)["n_clears_target"])

    # measured, not adopted: the module constants are still PART 10's
    assert ww.FILLET_LAYER_ENTRY_SLOPE == fb.LAYER_ENTRY_SLOPE
    assert ww.FILLET_LAYER_END_OFFSET == fb.LAYER_END_OFFSET
    assert fb.GENOME_ROBUST_ENTRY != fb.LAYER_ENTRY_SLOPE


def test_the_margin_robust_pair_loses_its_parity_OUT_OF_SAMPLE(report):
    """PLAN §81, pinned because the in-sample reading is the one that would come back.

    `MARGIN_ROBUST_*` was §69's answer to §68's cliff-margin objection and it ties
    `GENOME_ROBUST_*` on the box it was fitted to — 15 of 16, 14 of 14 fold-clean. On
    §78's held-out thirty-two it clears 28 against 31. The tie is the in-sample half and
    the rejection rests on exactly that, so a run in which the two tie out of sample would
    reverse §81's call and has to register as a failure rather than pass quietly.

    The other direction is asserted too: the pair is REJECTED, not bad. It still clears
    twelve more held-out genomes than the pair that ships, and if that stopped being true
    the reason for keeping it in the study at all would be gone.
    """
    per = report["sector"]["per_config"]["coarse"]
    fc, ho = per["fit_clamp"], per["fit_clamp_held_out"]
    k = fb.SECTOR_FIT_CLAMP

    def clears(table, profile):
        return next(r for r in table["rows"]
                    if r["profile"] == profile and r["factor"] == k)

    assert (clears(fc, "margin_robust")["n_clears_target"]
            == clears(fc, "genome_robust")["n_clears_target"]), (
        "the two pairs no longer tie IN SAMPLE — §81's finding is that the tie does not "
        "survive the hold-out, and it needs the tie to exist")
    assert (clears(ho, "margin_robust")["n_clears_target"]
            < clears(ho, "genome_robust")["n_clears_target"]), (
        "the in-sample tie now HOLDS out of sample — that reverses §81's rejection of "
        "`(-0.70, 0.90)` and is a finding, not a broken test")
    assert (clears(ho, "margin_robust")["n_clears_target"]
            > clears(ho, "shipped")["n_clears_target"])

    # and the clamp still builds the whole held-out box at the new profile: the refusal
    # half and the barrier half are separate items and a coupling would merge them
    row = clears(ho, "margin_robust")
    assert row["n_built"] == row["n_genomes"]

    # measured, not adopted, same as the pair above it
    assert ww.FILLET_LAYER_ENTRY_SLOPE == fb.LAYER_ENTRY_SLOPE
    assert (fb.MARGIN_ROBUST_ENTRY, fb.MARGIN_ROBUST_END) != (fb.LAYER_ENTRY_SLOPE,
                                                              fb.LAYER_END_OFFSET)


def test_the_per_genome_factor_is_the_BAND_EDGE_and_is_re_derived_from_the_rows(report):
    """PLAN §82. `CLIFF_PROFILE_FACTOR` is a constant and the surface is a measurement;
    this re-derives one from the other every run so the constant cannot go stale silently
    — the same guard `the_candidate_constant_matches_the_measured_surface` gives the
    `(entry, end)` sets.

    The rule is stated once, here and in the constant's comment: the lowest swept factor
    that still clears `MIN_SJ_TARGET` on as many HELD-OUT genomes as the best factor does.
    Below it the barrier degrades; above it the shipped genome pays margin and convergence
    for nothing. If the sweep ever ends at its own edge the bracket has stopped containing
    the answer, and that fails here rather than being read off a table nobody re-derives.
    """
    cp = report["sector"]["per_config"]["coarse"]["cliff_profile_held_out"]
    rows = sorted(cp["rows"], key=lambda r: -r["factor"])
    best = max(r["n_clears_target"] for r in rows)
    edge = min(r["factor"] for r in rows if r["n_clears_target"] == best)

    assert edge == pytest.approx(fb.CLIFF_PROFILE_FACTOR), (
        f"the band edge measured on the held-out draw is {edge}, and "
        f"CLIFF_PROFILE_FACTOR says {fb.CLIFF_PROFILE_FACTOR}")
    assert edge > min(cp["factors"]), (
        "the flat band reaches the bottom of the swept range — the bracket no longer "
        "contains the edge and the factors have to be extended further")
    # and the band is a band: the factor below the edge must actually be worse, or the
    # edge is an artefact of where the sweep happened to put its rungs
    below = max((r for r in rows if r["factor"] < edge), key=lambda r: r["factor"])
    assert below["n_clears_target"] < best

    # measured, not adopted — the per-genome rule is not wired into the mesh
    assert ww.FILLET_LAYER_ENTRY_SLOPE == fb.LAYER_ENTRY_SLOPE
    assert ww.FILLET_LAYER_END_OFFSET == fb.LAYER_END_OFFSET


def test_the_fold_margin_is_the_SAMPLED_flank_and_not_a_proxy_for_it(report):
    """The closed form checked against the thing it stands in for, on this box.

    `fold_margin` is `min_s(|1/kappa| - t/2)` off the Bezier hodograph -- no sampling of
    the flank enters it.  `flank_reversal_mm` is the sampled statement: project each
    flank node's step on the tangent it came from and take the least.  If the first is a
    valid gate then their SIGNS agree, and the point of asserting it here rather than
    trusting the identity is that `offset_band` uses finite-difference normals in the
    exported outline and exact ones in the mesh, so "the curvature says it folds" and
    "the sampled outline doubles back" are two different computations that happen to be
    about the same fact.

    AND THE SAMPLING IS NAMED, because the second one moves.  At `FOLD_AUDIT_POINTS` the
    two agree on this box; at the 97 stations PART 13's shipped grid uses, the sampled
    test calls both folded genomes healthy and the closed form has already rejected them.
    Both halves are asserted -- the agreement so the closed form is checked rather than
    trusted, and the disagreement at 97 so the reason a closed form is the gate stays on
    the record instead of being an argument in a docstring.

    Re-derived from the genes, not read off the artifact.  The margin is also pinned
    against `study_mesh_quality.fold_margin`, so the repo keeps ONE computation of this.
    """
    import study_mesh_quality as smq
    rows = [r for v in report["sector"]["genomes"]["groups"].values() for r in v]
    assert len(rows) == 16, len(rows)
    folded = []
    for row in rows:
        vec = np.asarray(row["genes"], float)
        f = fb.fold_margin(vec, "coarse")
        assert f["margin_mm"] == pytest.approx(row["fold"]["margin_mm"], abs=1e-9)
        assert f["margin_mm"] == pytest.approx(
            smq.fold_margin(vec, ww.get_config("coarse")), abs=1e-12)
        fine = fb.fold_margin(vec, "coarse", fb.FOLD_AUDIT_POINTS)
        rev = fb.flank_reversal_mm(vec, "coarse", fb.FOLD_AUDIT_POINTS)
        assert (fine["margin_mm"] < 0.0) == (rev < 0.0), (
            f"closed form {fine['margin_mm']:+.6f} and sampled flank {rev:+.6e} disagree "
            f"at {fb.FOLD_AUDIT_POINTS} points")
        if f["folds"]:
            folded.append(vec)
    assert len(folded) == 2, (
        "the committed draw's fold count moved — a finding, not a pass")

    # and at PART 13's own grid the sampled statement changes sides while the closed
    # form does not — which is why the gate is the closed form
    for vec in folded:
        coarse_rev = fb.flank_reversal_mm(vec, "coarse", 97)
        assert coarse_rev > 0.0, (
            "the 97-station flank now catches this fold — PART 13's mechanism has "
            "changed and the record should say so")
        rungs = fb.fold_resolution_ladder(vec, "coarse")
        assert all(r["margin_mm"] < 0.0 for r in rungs), rungs
        spread = max(r["margin_mm"] for r in rungs) - min(r["margin_mm"] for r in rungs)
        assert spread < 0.1 * fb.WG.MIN_FOLD_MARGIN_MM, (
            f"the closed form moved by {spread:.2e} mm across the ladder")


def test_no_fold_clean_genome_INVERTS_a_block_and_the_converse_is_not_claimed(report):
    """The gate's promise, and the exact shape of it.

    One direction is the gate and is asserted: a genome whose flank does not fold builds
    a sector with no inverted block.  The other direction is NOT a property of the part
    -- whether a folded flank shows up as an inverted element depends on whether the
    trim happens to put a station on the dip, which is a property of the grid -- so it is
    asserted as the weaker, true thing: every inverted block in the box belongs to a
    genome the margin already rejected.

    PART 13 found the one such genome by accident and traced it by hand.  This is that
    finding as a classifier, and the asymmetry is the reason the closed form is the gate
    and `sweep_genomes`' mesh-based filter is not.
    """
    rows = [r for v in report["sector"]["genomes"]["groups"].values() for r in v]
    built = [r for r in rows if r["built"]]
    clean = [r for r in built if not r["fold"]["folds"]]
    assert len(clean) >= 8, len(clean)
    for row in clean:
        assert row["all_blocks_valid"], (
            f"a fold-clean genome inverted {row['worst_block']} at "
            f"{row['min_scaled_jacobian']:+.4f} — the gate has a MISS")
    inverted = [r for r in built if not r["all_blocks_valid"]]
    assert len(inverted) == 1, [r["worst_block"] for r in inverted]
    assert inverted[0]["worst_block"] == "spoke"
    assert inverted[0]["fold"]["folds"] is True
    # and the converse is genuinely open: the other folded genome is not inverted here,
    # it is simply refused for an unrelated reason.
    folded = [r for r in rows if r["fold"]["folds"]]
    assert len(folded) == 2 and any(not r["built"] for r in folded)


def test_the_draw_filter_this_box_used_LEAKS_and_the_rate_is_measured(report):
    """Why a gate was needed at all, given `sweep_genomes` already had one.

    Its filter ends with "the unfilleted sector is clean", which is a mesh-based proxy
    for the fold and a good one -- but a proxy.  `sweep_fold_gate` runs both over the
    same Latin-hypercube stream the box is drawn from and counts what gets through.  The
    assertions are one-sided on purpose: the leak must be small enough that the proxy is
    clearly doing most of the work, and nonzero, because a zero would mean this whole
    section is answering a question nobody has.
    """
    fg = report["sector"]["fold_gate"]
    c = fg["counts"]
    assert c["drawn"] >= 512 * 8 and c["geom"] > 100
    assert 0.2 < fg["fold_rate"] < 0.6, fg["fold_rate"]
    assert 0.0 < fg["leak_rate"] < 0.15, (
        f"the mesh-based filter leaks at {fg['leak_rate']:.3f} — if that is 0 the closed "
        "form buys nothing here, and if it is large the filter is not a proxy at all")
    assert c["mesh_clean_folds"] < c["geom_folds"], c
    ag = fg["agreement"]
    assert ag["worst_disagreement_margin_mm"] < fb.FOLD_AGREEMENT_TOL_MM
    assert ag["worst_disagreement_margin_mm"] < 0.1 * fg["limit_mm"], (
        "a disagreement inside the barrier band would make the barrier unreadable")


def test_the_fold_gate_is_INERT_on_the_shipped_genome(genes, report):
    """The same question §57 asked of the clamp, and it has to be asked of every gate.

    Every number in this file is measured at the shipped genome.  A gate that excluded it
    would not be a gate, it would be a promotion — so this is re-derived from
    `best_solution.json` rather than read, and the margin is asserted to clear the
    barrier by a wide factor rather than merely to clear it.
    """
    f = fb.fold_margin(genes, "coarse")
    assert f["folds"] is False and f["binds"] is False
    assert f["margin_mm"] > 10.0 * f["limit_mm"], f
    assert fb.flank_reversal_mm(genes, "coarse") > 0.0
    assert report["sector"]["fold_gate"]["shipped"]["margin_mm"] == pytest.approx(
        f["margin_mm"], rel=1e-9)


def test_the_gate_costs_the_box_two_genomes_and_buys_the_one_defect(report):
    """What applying the gate would do to the arc's own table, on the same genomes.

    `fit_clamp_fold_clean` is `sweep_sector_fit_clamp` over the rows the margin keeps, so
    the two tables differ only by the exclusion.  What must hold: the fold-clean box is
    smaller, still fully buildable under the clamp, and no longer contains a genome that
    inverts -- which is the whole of what the gate buys, stated so that a run where it
    bought more (or less) registers.
    """
    per = report["sector"]["per_config"]["coarse"]
    fc, fcc = per["fit_clamp"], per["fit_clamp_fold_clean"]
    k = fb.SECTOR_FIT_CLAMP

    def row(table, profile, factor):
        return next(r for r in table["rows"]
                    if r["profile"] == profile and r["factor"] == factor)

    assert row(fcc, "shipped", k)["n_genomes"] == row(fc, "shipped", k)["n_genomes"] - 2
    for profile in ("shipped", "genome_robust"):
        clamped = row(fcc, profile, k)
        assert clamped["n_built"] == clamped["n_genomes"], clamped
    # the quality half is not what the gate is for, and it does not close it
    assert (row(fcc, "shipped", k)["n_clears_target"]
            < row(fcc, "shipped", k)["n_genomes"])
    # §54's excluded genome leaves with the gate, so the floor stops being an inversion
    assert row(fcc, "genome_robust", k)["min_scaled_jacobian_range"][0] > 0.0, (
        "an inverted block survived the fold gate — the exclusion did not do its job")
    assert row(fc, "genome_robust", k)["min_scaled_jacobian_range"][0] < 0.0


@pytest.mark.parametrize("cfg", SECTOR_CFGS)
def test_the_recut_does_NOT_rescue_the_faithful_rim(genes, cfg):
    """The tri-block's question, asked from the fillet's side and answered against it.

    §46 promoted the rim tri-block because on the faithful rim — `uncap` blend 0.0 —
    `rim:P_c`'s admissible fillet radius goes 0.561 -> 2.585 mm, and that corner carries
    the wheel's global peak.  The obvious hope after the re-cut is that it fixes blend 0.0
    too.  It does not, and it makes it WORSE: what collapses there is a corner opening to
    180 degrees at `far_end`, and the filleted junction block is SHORTER, so the straight
    corner dominates more of it.  Re-measured rather than read, and asserted as the
    ORDERING — filleted below unfilleted below the barrier — so a future construction that
    fixed it would go red here and reopen the tri-block's ranking.
    """
    ctl = ww.sector_blocks(genes, cfg, uncap=(True, 0.0))
    ctl_sj = min(fb.block_quality(np.asarray(v, float))["min_scaled_jacobian"]
                 for k, v in ctl.items() if not k.startswith("_"))
    blocks = ww.filleted_sector(genes, cfg, uncap=(True, 0.0))
    fil_sj = min(fb.block_quality(np.asarray(v, float))["min_scaled_jacobian"]
                 for k, v in blocks.items() if not k.startswith("_"))
    assert ctl_sj < wo.MIN_SJ_TARGET, ctl_sj
    assert fil_sj < ctl_sj, (fil_sj, ctl_sj)
    # and the default blend is where both are usable
    good = ww.filleted_sector(genes, cfg)
    assert min(fb.block_quality(np.asarray(v, float))["min_scaled_jacobian"]
               for k, v in good.items() if not k.startswith("_")) > wo.MIN_SJ_TARGET


def test_the_filleted_sector_costs_the_unfilleted_one_nothing(genes):
    """The control: `sector_blocks(fillet=None)` is not touched by any of this.

    Everything in STEP 1a is built inside the study from `wheel_wheel`'s primitives, so
    the seven-block sector the tree ships has to come back exactly as clean as it was —
    and the number it is clean AT is what the filleted sector's 0.36 is a degradation
    from, so it is asserted rather than remembered.
    """
    for cfg in SECTOR_CFGS:
        ctl = fb.sector_control(genes, cfg)
        assert ctl["n_blocks"] == 7 and ctl["all_valid"]
        assert ctl["min_scaled_jacobian"] > 0.7, ctl


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


# ---------------------------------------------------------------------------
# §68's CLIFF, AS A COLUMN (PLAN §77, FILLET_PLAN PART 20 measured it by hand)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("end,published", fb.CLIFF_PUBLISHED)
def test_the_cliff_column_reproduces_PART_20s_hand_bisections(genes, end, published):
    """The automated column has to land on the four numbers the record was written from.

    PLAN §68 and FILLET_PLAN PART 20 declined the genome-robust layer profile on these
    four bisections and the margins they imply, so a column that disagreed with them
    would mean either the decision or the code is wrong.  Pinned against the RECORD
    rather than against a re-run of itself, at the precision §68 published to.
    """
    got = fb.cliff_entry(genes, "coarse", end)
    assert got["entry"] is not None, got["why"]
    assert abs(got["entry"] - published) < 1e-4, (end, got["entry"], published)
    assert fb.CLIFF_REASON in got["why"], got["why"]


def test_the_cliff_is_the_WIDTH_PROFILE_refusal_and_not_whichever_comes_first(genes):
    """A bisection that accepted any refusal would report a different limit under this name.

    The blocking refuses for four distinct geometric reasons and only one of them is the
    layer losing its thickness.  `cliff_entry` checks which bounded it and returns `None`
    with the reason when it is something else — so a future change that made, say, the
    sector-fit limit bind first shows up as a missing column rather than as a margin
    quietly measured against the wrong edge.
    """
    got = fb.cliff_entry(genes, "coarse", 1.00)
    assert fb.CLIFF_REASON in got["why"]
    # and the verdict at an entry just past the cliff refuses for exactly that reason
    v = fb.sector_verdict(genes, "coarse", float(genes[12]), float(genes[13]),
                          entry=got["entry"] - 0.01, end=1.00)
    assert not v["built"] and fb.CLIFF_REASON in v["why"], v
    # while just inside it, the sector builds
    v = fb.sector_verdict(genes, "coarse", float(genes[12]), float(genes[13]),
                          entry=got["entry"] + 0.01, end=1.00)
    assert v["built"], v


def test_the_SHIPPED_profile_stands_farther_from_the_cliff_than_any_candidate(genes):
    """§68's finding, as a check: the arc's candidates are all closer to the edge.

    Every pair this arc proposed stands within 0.08 of a hard refusal of the shipped
    genome and the pair that ships stands 0.55 from it — which is the whole reason the
    profile is measured and not adopted.  A future grid that produced a roomier candidate
    would go red here, and that is the outcome that should reopen the call rather than a
    line in a plan file nobody re-reads.
    """
    shipped = fb.cliff_entry(genes, "coarse", fb.LAYER_END_OFFSET)
    shipped_margin = fb.LAYER_ENTRY_SLOPE - shipped["entry"]
    assert shipped_margin == pytest.approx(0.5520, abs=1e-3), shipped_margin
    compared = 0
    for entry, end in fb.LAYER_PROFILE_FINE_CANDIDATES + fb.LAYER_PROFILE_CANDIDATES:
        c = fb.cliff_entry(genes, "coarse", end)
        if c["entry"] is None:
            continue
        compared += 1
        assert entry - c["entry"] < shipped_margin, (entry, end, entry - c["entry"])
    # or the loop above proves nothing: every candidate having no cliff would pass it
    # silently, and the assertion is about the candidates rather than about the shipped
    # pair alone.
    assert compared == len(fb.LAYER_PROFILE_FINE_CANDIDATES
                           + fb.LAYER_PROFILE_CANDIDATES), compared


# ---------------------------------------------------------------------------
# THE HELD-OUT BOX (PLAN §78) — is §74's "16 of 16" an in-sample number?
# ---------------------------------------------------------------------------

def test_the_held_out_draw_is_actually_disjoint_from_the_committed_one(report):
    """A hold-out that shared genomes with the fitted box would flatter every rate on it.

    Checked from the GENES rather than trusted to the seed arithmetic, because the thing
    that would break it is `sweep_genomes`' batch walk (`seed + batch`) overrunning the
    offset — an off-by-one in `max_batches` would overlap the two streams silently and
    every number in this section would quietly become in-sample again.
    """
    sec = report["sector"]
    a = {tuple(round(x, 12) for x in r["genes"])
         for v in sec["genomes"]["groups"].values() for r in v}
    b = {tuple(round(x, 12) for x in r["genes"])
         for v in sec["genomes_held_out"]["groups"].values() for r in v}
    assert len(a) == 16 and len(b) == 32, (len(a), len(b))
    assert not (a & b), f"{len(a & b)} genomes appear in both draws"
    assert sec["genomes_held_out"]["seed"] == (
        fb.GENOME_SWEEP_SEED + fb.GENOME_HELD_OUT_OFFSET)


def test_the_clamp_closes_the_refusal_half_OUT_OF_SAMPLE_too(report):
    """§74's headline was measured on the sixteen genomes the clamp was designed against.

    UNCAP_PLAN PART 9 is the reason this is worth a test rather than an assumption: a rule
    fitted on 104 genomes scored 1.000 in sample and 0.833 held out, and the hold-out
    falsified half of it.  This one survives — every genome of a disjoint draw builds once
    its radii are inside its own sector's room.

    THE BARRIER HALF IS ASSERTED TO STILL BE OPEN, for the same reason the in-sample test
    asserts it: a run in which the clamp appeared to close both halves is a finding and
    must not read as a pass.
    """
    ho = report["sector"]["per_config"]["coarse"]["fit_clamp_held_out"]

    def row(profile, factor):
        return next(r for r in ho["rows"]
                    if r["profile"] == profile and r["factor"] == factor)

    base = row("shipped", None)
    clamped = row("shipped", fb.SECTOR_FIT_CLAMP)
    assert base["n_genomes"] == 32
    assert base["n_built"] < base["n_genomes"], "nothing refused — that is a finding"
    assert clamped["n_built"] == clamped["n_genomes"], (
        f"the clamp does NOT close the refusal half out of sample: "
        f"{clamped['n_built']}/{clamped['n_genomes']}, {clamped['refusals']}")
    assert clamped["n_clears_target"] < clamped["n_genomes"], (
        "the clamp now clears the barrier everywhere on the held-out box too — the "
        "quality half would be closed, which is a finding")


def test_the_held_out_draw_contains_the_first_RIM_refusal(report):
    """And it narrows §57's wording: the mechanism is the junction's, not the hub's.

    Every one of the six in-sample refusals binds at the HUB, and §57 and §74 both wrote
    the predictor up in those words — "the hub margin classifies 16 of 16".  The held-out
    box contains a genome that refuses at the RIM, so the hub margin alone classifies 31
    of 32 and the sector-fit margin at EITHER junction classifies 32 of 32.

    Nothing about the mechanism changed — a tangent point past the next sector's corner is
    the same event at either ring.  What was in-sample was the JUNCTION, and this pins the
    corrected form so the narrow one cannot come back.
    """
    rows = [r for v in report["sector"]["genomes_held_out"]["groups"].values()
            for r in v]
    refused = [r for r in rows if not r["built"]]
    assert refused, "nothing refused in the held-out draw — that is a finding"
    assert any(r["fit"]["rim"]["binds"] for r in refused), (
        "no rim-bound refusal in the held-out box — if the draw changed, §78's "
        "narrowing of §57's wording needs re-reading rather than this test relaxing")

    hub_only = sum(1 for r in rows if r["fit"]["hub"]["binds"] != r["built"])
    either = sum(1 for r in rows
                 if (r["fit"]["hub"]["binds"] or r["fit"]["rim"]["binds"]) != r["built"])
    assert either == len(rows), (either, len(rows))
    assert hub_only < len(rows), (
        "the hub margin alone now classifies the whole held-out box — the rim refusal "
        "is gone and §78's narrowing should be re-read")


# ---------------------------------------------------------------------------
# THE BARRIER HALF, AGAINST A PER-GENOME ENTRY (PLAN §78)
# ---------------------------------------------------------------------------

def test_the_per_genome_profile_DOMINATES_the_global_pair_68_declined(report):
    """§78's finding: the same barrier clearance for several times the cliff margin.

    §68 declined `GENOME_ROBUST_*` because it spends the shipped genome's cliff margin down
    to ~0.056.  A per-genome entry — each genome taking a share of its OWN room, the shape
    §57's clamp works in — clears the SAME 31 of 32 while leaving 0.20-0.36, because a
    global pair has to be safe for the tightest genome in the box and pays for that
    everywhere.

    Pinned because it is the number that reopens §68's first reason, and because the first
    draft of §78 concluded the exact opposite off a miscount — three genomes that build at
    every entry in the bracket were tallied as measurement failures.  If that handling
    regresses, this test is what catches it.
    """
    for key in ("cliff_profile", "cliff_profile_held_out"):
        cp = report["sector"]["per_config"]["coarse"][key]
        assert cp["end"] == fb.GENOME_ROBUST_END
        rows = {r["factor"]: r for r in cp["rows"]}
        assert set(rows) == set(fb.CLIFF_PROFILE_FACTORS)

        # the margin the rule leaves grows as the factor falls, and monotonically —
        # that is the only reason sweeping the factor says anything
        margins = [rows[f]["shipped_margin"] for f in sorted(rows, reverse=True)]
        assert all(b > a for a, b in zip(margins, margins[1:])), margins

        # AND NOT ONE OF THE ADMISSIBLE ONES REACHES THE SHIPPED PAIR'S CLEARANCE.
        #
        # Scoped to the admissible set rather than to the sweep, and the scope is the
        # finding rather than a convenience.  §81 extended the factors down to 0.15 to
        # locate the band's lower edge, and 0.35 / 0.25 / 0.15 DO leave more than the
        # shipped pair's 0.5520 -- by giving up 2, 7 and 15 genomes of the held-out box
        # on the barrier, which is the trade this rule exists to avoid making.  A margin
        # bought that way is not the same claim, so it is not tallied under the same
        # assertion; the barrier is what separates the two and it is asserted directly.
        adm = [f for f in rows if f >= fb.CLIFF_PROFILE_FACTOR]
        assert max(rows[f]["shipped_margin"] for f in adm) < 0.5520, (
            key, {f: rows[f]["shipped_margin"] for f in adm})
        # THE EDGE ITSELF IS A HELD-OUT CLAIM AND IS CHECKED ONLY THERE.  In sample 0.35
        # ties the edge at 15 of 16 rather than falling below it, which is not a
        # contradiction: 16 genomes cannot resolve a one-genome step, and §81's whole
        # correction was that this band has to be located out of sample.
        if key == "cliff_profile_held_out":
            edge_clears = rows[fb.CLIFF_PROFILE_FACTOR]["n_clears_target"]
            for f in [f for f in rows if f < fb.CLIFF_PROFILE_FACTOR]:
                assert rows[f]["n_clears_target"] < edge_clears, (
                    f"factor {f} clears {rows[f]['n_clears_target']} against the "
                    f"adopted edge's {edge_clears} — the band's lower edge has moved "
                    f"off where §82 put it")

        # "no edge to project onto" and "could not be measured" are different claims and
        # must not be tallied together — the first is the SAFEST genome in the box, and
        # folding it into the second understated this rule by three of sixteen once
        # already (§78).
        for r in cp["rows"]:
            assert r["n_built"] + r["n_unevaluable"] == r["n_genomes"], r
            assert r["n_without_cliff"] > 0, (
                "no genome builds across the whole bracket any more — the fallback "
                "branch is now dead code and §78's correction should be re-read")

    # THE DOMINANCE ITSELF, on the held-out box: at least one factor matches the global
    # pair's barrier clearance while leaving several times its margin.
    ho = report["sector"]["per_config"]["coarse"]["fit_clamp_held_out"]
    gr = next(r for r in ho["rows"] if r["profile"] == "genome_robust"
              and r["factor"] == fb.SECTOR_FIT_CLAMP)
    cp = report["sector"]["per_config"]["coarse"]["cliff_profile_held_out"]
    gr_margin = fb.GENOME_ROBUST_ENTRY - cp["shipped_cliff"]["entry"]
    wins = [r for r in cp["rows"]
            if r["n_clears_target"] >= gr["n_clears_target"]
            and r["shipped_margin"] > 3.0 * gr_margin]
    assert wins, (
        f"no per-genome factor matches the global pair's {gr['n_clears_target']} clears "
        f"at 3x its {gr_margin:.4f} margin — §78's finding has moved")
    # and every one of them builds the whole box, or the clears are over a smaller set
    for r in wins:
        assert r["n_built"] == r["n_genomes"], r

    # ADOPTED AT §82 AS A RULE, AND THE SHIPPED DEFAULT STILL DID NOT MOVE.  The two are
    # not in tension: `per_genome_layer_profile` is the adopted mechanism and
    # `FILLET_LAYER_CLIFF_FACTOR` its operating point, while the pair below is what
    # `fillet=True` still takes when nobody asks for the rule.  Flipping that default is
    # a separate step with its own artifact audit — see §82's "what is unchanged".
    assert ww.FILLET_LAYER_ENTRY_SLOPE == fb.LAYER_ENTRY_SLOPE
    assert ww.FILLET_LAYER_END_OFFSET == fb.LAYER_END_OFFSET
    assert ww.FILLET_LAYER_CLIFF_FACTOR == fb.CLIFF_PROFILE_FACTOR
    assert ww.FILLET_LAYER_CLIFF_END == fb.CLIFF_PROFILE_END


def test_the_layer_cliff_has_a_CLOSED_FORM_that_reproduces_the_bisection(genes):
    """§82's adoption rests on this and nothing else: the cliff costs arithmetic.

    `cliff_entry` finds the layer-width cliff by bisecting `sector_verdict` thirty times,
    which is thirty filleted sector builds — far too expensive to sit inside a mesh
    default.  `wheel_wheel._layer_cliff_from_scalars` finds the same number from the two
    scalars the TANGENCY solve already produces, because the width profile is a cubic in
    `u` whose only `entry`-dependence is one linear term.

    If these two ever disagree by more than the bisection's own resolution, the module
    and the study are measuring different cliffs and every number in §78-§82 is about to
    mean two things.
    """
    cfg = fb.CONVERGENCE_LADDER[0] if hasattr(fb, "CONVERGENCE_LADDER") else "coarse"
    module = ww.layer_cliff_entry(genes, cfg, end=fb.CLIFF_PROFILE_END)
    study = fb.cliff_entry(genes, cfg, fb.CLIFF_PROFILE_END)
    assert module["entry"] is not None and study["entry"] is not None
    # 30 halvings of `CLIFF_BRACKET`'s 2.0 is 1.9e-9, which is the floor here
    assert abs(module["entry"] - study["entry"]) < 5e-9, (module["entry"],
                                                          study["entry"])

    # THE SECTOR'S CLIFF IS THE BINDING JUNCTION'S, and the two really are different —
    # if they were equal this test would pass while asserting nothing about `max`.
    per = {j: module["per_junction"][j]["cliff"] for j in ("hub", "rim")}
    assert all(v is not None for v in per.values()), per
    assert per["hub"] != per["rim"], per
    assert module["entry"] == max(per.values()), (module["entry"], per)


def test_the_per_genome_profile_is_the_ADOPTED_operating_point(genes):
    """The rule, its factor, and the margin it leaves the genome every number is at."""
    cfg = "coarse"
    entry, end = ww.per_genome_layer_profile(genes, cfg)
    cliff = ww.layer_cliff_entry(genes, cfg, end=ww.FILLET_LAYER_CLIFF_END)["entry"]
    assert end == ww.FILLET_LAYER_CLIFF_END
    assert entry == pytest.approx(ww.FILLET_LAYER_CLIFF_FACTOR * cliff)

    # what §68's first reason asked for: the rule leaves the shipped genome several
    # times what the global pair it declined would have left it
    margin = entry - cliff
    assert margin == pytest.approx(0.4435, abs=5e-4), margin
    assert margin > 5.0 * (fb.GENOME_ROBUST_ENTRY - cliff)

    # AND THE OPERATING POINT IS SHALLOWER THAN THE SHIPPED ENTRY, which is the whole
    # reason §82's rule is safe against the `_sector_fit_span` defect it also records:
    # the clamp only misreads a layer refusal as "no room" at a STEEP entry, and this
    # rule never asks for one.
    assert entry > ww.FILLET_LAYER_ENTRY_SLOPE, (entry, ww.FILLET_LAYER_ENTRY_SLOPE)
    per = ww.layer_cliff_entry(genes, cfg)["per_junction"]
    assert not any(per[j].get("clamped") for j in ("hub", "rim")), per


def test_the_cliff_bracket_the_study_uses_is_TOO_NARROW_and_the_module_says_so(genes):
    """§82's third sentinel: "builds across the whole bracket" is not "has no edge".

    `study_fillet_block.CLIFF_BRACKET` stops at -2.0 and reports three of the thirty-two
    held-out genomes as having no layer-width edge at all — the SAFEST case, which
    `sweep_cliff_clamped_profile` handles by falling back to a global constant.  All
    three have edges, at -2.51, -2.30 and -2.12.  The module's bracket holds them.

    Pinned as a RELATION between the two constants rather than against those three
    genomes, so it keeps meaning something if the draw changes.
    """
    assert ww.LAYER_CLIFF_BRACKET[0] < fb.CLIFF_BRACKET[0], (
        ww.LAYER_CLIFF_BRACKET, fb.CLIFF_BRACKET)
    # and the module's own refusal threshold is the one the cliff is defined against
    assert ww.LAYER_CLIFF_ZERO == 1e-6
    assert ww.LAYER_CLIFF_SAMPLES == 401
