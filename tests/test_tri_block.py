"""
Pins for `studies/study_tri_block.py` — PLAN.md §37 and §51, UNCAP_PLAN.md Step 3.

WHY THIS FILE EXISTS.  §51 filed the tri-block's re-pricing AS A PROBE — "no driver, no
artifact, no test — precisely so that nobody quotes it as this project's other numbers may
be quoted" — and named what would make it a measurement.  `make triblock` is that, and it
overturns both of the clauses §37 shelved the construction on.  A conclusion that large
has to be re-derivable by whoever doubts it, and it rests on four structural claims rather
than on any one number:

  1. THE ALGEBRA.  §37's six-way constraint is right and its input was not.  The
     admissible free counts are enumerable, §37's own `B` really does force a 1-element
     strip, and three of the five the algebra admits at `medium` do not.
  2. THE SEAMS ARE WHOLE-EDGE.  Every one of the seventeen is a complete edge of both
     blocks it names.  That is §37 clause 1's entire question and it is a claim about
     INDICES, so it is checked on the index arrays and not on a tolerance.
  3. THE NEIGHBOURS ARE SLICED, NOT REBUILT.  `spoke`, `hub_junction` and
     `rim_band_weld` come out of `sector_blocks` and are cut at a node index, so the
     twelve-block sector is the seven-block one PLUS a partition — bit-for-bit, not
     nearly.  If that ever stops being true the comparison against the control stops
     meaning anything.
  4. THE PARTITION COVERS ITS OWN REGION.  Three quads whose areas sum to the quad's is
     the cheap check that they tile the triangle rather than overlap it.

EVERY TEST HERE RE-MEASURES, for PART 7's reason: a committed artifact whose driver takes
a bare default rots silently, and every test that reads it reads the same stale file.  Two
read `study_tri_block.json` at all — the freshness guard, which exists to catch exactly
that, and the self-check pin.

AND ONE TEST IS ABOUT THE SCOPE RATHER THAN THE CONSTRUCTION.  The faithful rim is not
opt-in: adopting it would move the mesh under every genome.  So the gene-box column is
load-bearing, and `test_the_gene_box_is_what_this_does_not_deliver` pins that it is
REPORTED as a refusal rather than rounded off — a driver whose genome sweep quietly went
green would be the one way this file's headline could become a false one.
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
import study_tri_block as tb           # noqa: E402
import study_fillet_block as fb        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT = os.path.join(REPO, "studies", "study_tri_block.json")


@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


@pytest.fixture(scope="module")
def report():
    with open(ARTIFACT) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def coarse(genes):
    """The region and the chosen cell at `coarse`, built fresh."""
    reg = tb.region(genes, "coarse", blend=0.0)
    return reg, tb.tri_sector(reg, 10, (0.124, 0.751, 0.124))


# ---------------------------------------------------------------------------
# 1. THE ALGEBRA — §37 clause 2
# ---------------------------------------------------------------------------

def test_the_partition_algebra_reproduces_section_37s_own_arithmetic():
    """§37's 7/3/1 at `coarse` is right, and this file has to get it before it moves it.

    A re-pricing that could not reproduce the number it is re-pricing would be measuring a
    different partition, and the six counts are the whole of the disagreement.
    """
    sp = tb.splits(10, 8, 4)
    assert sp == {"a1": 7, "a2": 3, "b1": 1, "b2": 7, "c1": 3, "c2": 1}
    shapes = [[sp["a1"], sp["c2"]], [sp["a2"], sp["b1"]], [sp["b2"], sp["c1"]]]
    assert shapes == [[7, 1], [3, 1], [7, 3]], "§37's forced strip is real at §37's own B"


def test_the_six_counts_satisfy_the_constraint_they_are_derived_from():
    """`a1 = b2`, `a2 = c1`, `c2 = b1`, and each side's two pieces sum to the side.

    Derived once and then used everywhere, so it is checked as an identity over the whole
    admissible set rather than at the one cell that ships.
    """
    for A in range(2, 30):
        for C in range(2, 12):
            for B in tb.admissible(A, C):
                sp = tb.splits(A, B, C)
                assert sp["a1"] == sp["b2"]
                assert sp["a2"] == sp["c1"]
                assert sp["c2"] == sp["b1"]
                assert sp["a1"] + sp["a2"] == A
                assert sp["b1"] + sp["b2"] == B
                assert sp["c1"] + sp["c2"] == C
                assert min(sp.values()) >= 1


def test_the_free_count_is_free_and_three_of_mediums_five_have_no_strip():
    """The clause-2 correction, as a count rather than as an argument.

    §37 read `B` off the block it was replacing and got the only value at `coarse` that
    is BOTH admissible and strip-free wrong.  The point is not that 8 was a bad choice —
    it is that it was never a choice.
    """
    assert tb.admissible(10, 4) == (8, 10, 12)
    assert tb.admissible(16, 6) == (12, 14, 16, 18, 20)
    strip_free = [B for B in tb.admissible(16, 6)
                  if min(min(s) for s in ([tb.splits(16, B, 6)["a1"],
                                           tb.splits(16, B, 6)["c2"]],
                                          [tb.splits(16, B, 6)["a2"],
                                           tb.splits(16, B, 6)["b1"]],
                                          [tb.splits(16, B, 6)["b2"],
                                           tb.splits(16, B, 6)["c1"]])) > 1]
    assert strip_free == [14, 16, 18]
    assert [B for B in tb.admissible(10, 4)
            if tb.splits(10, B, 4)["c2"] > 1 and tb.splits(10, B, 4)["a2"] > 1] == [10]


def test_the_admissible_set_is_exactly_the_parity_and_positivity_window():
    """`|A - C| < B < A + C` with `A + B - C` even, and nothing outside it.

    Stated in the driver's docstring as the window; asserted here so that a change to
    `splits` that widened it would have to change this line too.
    """
    A, C = 16, 6
    for B in range(1, A + C + 6):
        want = (abs(A - C) < B < A + C) and ((A + B - C) % 2 == 0)
        assert (tb.splits(A, B, C) is not None) is want, B


# ---------------------------------------------------------------------------
# 2. THE SEAMS — §37 clause 1
# ---------------------------------------------------------------------------

def test_every_seam_the_partition_declares_is_a_WHOLE_edge_of_both_blocks(coarse):
    """§37 clause 1, checked on indices rather than on a tolerance.

    "Whole-edge single ownership is the whole safety net for this module" is a claim about
    which NODES a seam covers, and a partial-edge seam that happened to close to 1e-15
    would still be the thing §37 refused.  So this asserts that each declared side is the
    complete `i0`/`i1`/`j0`/`j1` of its block — which `_side` gives by construction — AND
    that the two sides carry the same node count, which is the half that can fail.
    """
    reg, (blocks, aux) = coarse
    table = tb.seam_table(reg, aux)
    assert len(table) == 17
    for a, sa, b, sb, dk, rev in table:
        ea, eb = tb._side(blocks[a], sa), tb._side(blocks[b], sb)
        ga, gb = np.asarray(blocks[a]), np.asarray(blocks[b])
        assert ea.shape[0] == (ga.shape[1] if sa in ("i0", "i1") else ga.shape[0])
        assert eb.shape[0] == (gb.shape[1] if sb in ("i0", "i1") else gb.shape[0])
        assert ea.shape == eb.shape, f"{a}.{sa} ~ {b}.{sb} counts disagree"


def test_all_seventeen_seams_close_at_both_configs(genes):
    """Both halves of "closes": the counts agree AND the nodes coincide.

    Built fresh at both configs, because the `reverse` flags and the two cut orderings
    are resolved from the genome and the config and a table that is right at one is not
    thereby right at the other.
    """
    for name, B, w in (("coarse", 10, (0.124, 0.751, 0.124)),
                       ("medium", 18, (0.072, 0.803, 0.124))):
        reg = tb.region(genes, name, blend=0.0)
        blocks, aux = tb.tri_sector(reg, B, w)
        sm = tb.seams(blocks, tb.seam_table(reg, aux))
        assert len(sm) == 17
        bad = [f"{s['a']}.{s['side_a']}~{s['b']}.{s['side_b']}" for s in sm
               if not s["closes"]]
        assert not bad, f"{name}: open seams {bad}"
        assert max(s["max_gap_mm"] for s in sm) < 1.0e-12


def test_the_three_seams_the_Y_itself_creates_are_EXACT(coarse):
    """The internal edges are passed as the same array to both blocks, so 0.0 and not 1e-15.

    This is a claim about the code rather than about the geometry, and it is the reason
    the partition cannot introduce a gap of its own: if it ever stops holding, the seam
    tolerance would hide it at 1e-15 while the construction had quietly become two
    derivations of one curve.
    """
    _, (blocks, _) = coarse
    T = np.asarray(blocks["rim_tri_t"], float)
    Q = np.asarray(blocks["rim_tri_q"], float)
    B = np.asarray(blocks["rim_tri_b"], float)
    assert np.abs(T[-1] - Q[0]).max() == 0.0          # M_A -> X
    assert np.abs(T[:, -1] - B[:, 0]).max() == 0.0    # M_C -> X
    assert np.abs(Q[:, -1] - B[-1]).max() == 0.0      # X   -> M_B


def test_the_seam_table_follows_the_GENOME_and_not_a_constant(genes):
    """Which half of a cut block is which depends on the flank orientation, and is read.

    §48 was bitten by exactly this on the sector-closing seam's `dk`, and this file has
    two of them: the ring's weld block is laid out in increasing theta whichever way the
    junction's arc runs, and the hub junction's cross-section is the spoke's row REVERSED
    when the straddling flank is at eta = +1.  Getting the second backwards leaves the
    node COUNTS agreeing at `coarse` — `n_thick` splits 2/2 there — and only the
    coordinates disagree, which is how it was found.
    """
    reg = tb.region(genes, "coarse", blend=0.0)
    blocks, aux = tb.tri_sector(reg, 10, (0.124, 0.751, 0.124))
    hub_pairs = [(a, b) for a, sa, b, sb, dk, rev in tb.seam_table(reg, aux)
                 if a.startswith("spoke") and b.startswith("hub_junction")]
    eta_hub = float(reg["orientation"][0])
    want = ([("spoke_eta_lo", "hub_junction_hi"), ("spoke_eta_hi", "hub_junction_lo")]
            if eta_hub > 0 else
            [("spoke_eta_lo", "hub_junction_lo"), ("spoke_eta_hi", "hub_junction_hi")])
    assert sorted(hub_pairs) == sorted(want)
    # And the mismatched pairing really does close the counts while opening the gap.
    swapped = [(want[0][0], "i0", want[1][1], "i0", 0, eta_hub > 0),
               (want[1][0], "i0", want[0][1], "i0", 0, eta_hub > 0)]
    rows = tb.seams(blocks, tuple(swapped))
    assert all(r["counts_agree"] for r in rows), "the trap: coarse's counts still agree"
    assert not any(r["closes"] for r in rows), "and the coordinates are what catch it"


# ---------------------------------------------------------------------------
# 3. THE NEIGHBOURS ARE SLICED, NOT REBUILT
# ---------------------------------------------------------------------------

def test_the_three_cut_neighbours_are_slices_of_what_the_tree_BUILDS_TODAY(genes):
    """Bit-for-bit, not nearly — because the whole comparison rests on it.

    Every number in the record is "the tri-block against the quad on the SAME geometry".
    If `spoke`, `hub_junction` or `rim_band_weld` were re-derived here rather than cut,
    the 77x would be a comparison of two constructions and not a measurement of a
    partition.
    """
    for name, B, w in (("coarse", 10, (0.124, 0.751, 0.124)),
                       ("medium", 18, (0.072, 0.803, 0.124))):
        reg = tb.region(genes, name, blend=0.0)
        blocks, aux = tb.tri_sector(reg, B, w)
        base = reg["base"]
        j = aux["j_star"]
        assert np.abs(np.asarray(blocks["spoke_eta_lo"])
                      - np.asarray(base["spoke"])[:, :j + 1]).max() == 0.0
        assert np.abs(np.asarray(blocks["spoke_eta_hi"])
                      - np.asarray(base["spoke"])[:, j:]).max() == 0.0
        h = aux["j_hub"]
        assert np.abs(np.asarray(blocks["hub_junction_lo"])
                      - np.asarray(base["hub_junction"])[:, :h + 1]).max() == 0.0
        assert np.abs(np.asarray(blocks["hub_junction_hi"])
                      - np.asarray(base["hub_junction"])[:, h:]).max() == 0.0
        i = aux["i_star"]
        assert np.abs(np.asarray(blocks[aux["first_weld"]])
                      - np.asarray(base["rim_band_weld"])[:i + 1]).max() == 0.0
        assert np.abs(np.asarray(blocks[aux["last_weld"]])
                      - np.asarray(base["rim_band_weld"])[i:]).max() == 0.0
        for k in ("hub_collar_weld", "hub_collar_free", "rim_band_free"):
            assert np.abs(np.asarray(blocks[k]) - np.asarray(base[k])).max() == 0.0


def test_the_cascade_stops_at_the_hub_junction(genes):
    """§51's "cascades once and stops", as a count of blocks.

    Splitting the spoke on a j-line reaches the hub junction because the spoke's hub row
    IS its `left` edge.  It reaches nothing further because the hub junction's split is
    in `j`, and its `j` runs across the collar arc rather than along it — so the collar's
    two blocks keep whole edges.  Seven blocks become twelve and not thirteen.
    """
    reg = tb.region(genes, "coarse", blend=0.0)
    blocks, _ = tb.tri_sector(reg, 10, (0.124, 0.751, 0.124))
    assert len(blocks) == 12
    assert tuple(blocks) == tb.TWELVE_BLOCK_ORDER
    base = ww.sector_blocks(genes, ww.get_config("coarse"), uncap=(True, 0.0))
    assert len([k for k in base if k != "_thetas"]) == 7
    for k in ("hub_collar_weld", "hub_collar_free", "rim_band_free"):
        assert np.asarray(blocks[k]).shape == np.asarray(base[k]).shape


# ---------------------------------------------------------------------------
# 4. THE REGION, AND THAT THE PARTITION COVERS IT
# ---------------------------------------------------------------------------

def test_the_faithful_rim_junction_really_is_a_triangle(genes):
    """The turn at `far_end` is within a degree of straight, at both configs.

    This is UNCAP_PLAN Step 2's finding and it is the premise of the whole file: if the
    corner existed, a quad would sit on the region and there would be nothing to
    partition.  Measured on the region's own curve rather than quoted.
    """
    for name in ("coarse", "medium"):
        reg = tb.region(genes, name, blend=0.0)
        r = tb.region_report(reg)
        assert 179.0 < r["turn_at_far_end_deg"] < 181.0
        # And at the SHIPPED blend the same vertex is a real corner, which is why the
        # tree meshes today.  Same instrument, same genome, one argument changed.
        r1 = tb.region_report(tb.region(genes, name, blend=1.0))
        assert r1["turn_at_far_end_deg"] < 160.0


def test_the_three_quads_tile_the_quad_blocks_own_region(genes):
    """Areas sum, so they cover it rather than overlap it.

    They cannot sum exactly: the free side is a boundary the partition is allowed to
    re-distribute, and a different node placement on the same curve gives a different
    boundary POLYGON.  1e-4 relative is far below that freedom and far above the
    round-off, which is what makes it a check rather than a coincidence.
    """
    for name, B, w in (("coarse", 10, (0.124, 0.751, 0.124)),
                       ("medium", 18, (0.072, 0.803, 0.124))):
        reg = tb.region(genes, name, blend=0.0)
        blocks, _ = tb.tri_sector(reg, B, w)
        quad = tb.block_area_mm2(reg["base"]["rim_junction"])
        tri = sum(tb.block_area_mm2(blocks[k]) for k in tb.TRI_BLOCKS)
        assert abs(tri - quad) / quad < 1.0e-4
        assert abs(tri - quad) / quad > 0.0


def test_the_shared_sides_keep_their_NEIGHBOURS_distribution(genes):
    """The arc stays uniform in theta and the cross section stays the spoke's own row.

    The partition is allowed to choose the free side's nodes and nothing else.  A
    construction that re-distributed a shared side would be changing its neighbour, and
    the seam would close only because the neighbour had been changed to match — which is
    the failure mode a seam check cannot see.
    """
    reg = tb.region(genes, "coarse", blend=0.0)
    blocks, aux = tb.tri_sector(reg, 10, (0.124, 0.751, 0.124))
    # The cross section, in both quads that carry a piece of it.  Not bit-exact, and the
    # reason is `coons_patch` rather than this file: it returns `ruled_v + ruled_u -
    # bilinear`, which on a boundary reduces to that boundary ALGEBRAICALLY and lands
    # 7e-15 mm off it in floating point.  The tolerance is three orders under the seam
    # tolerance and thirteen under the node spacing, so it still separates "the same
    # points" from "a redistribution".
    cross = reg["cross"]
    iC = 2 * aux["splits"]["c2"]
    assert np.abs(np.asarray(blocks["rim_tri_t"])[0] - cross[:iC + 1]).max() < 1.0e-12
    assert np.abs(np.asarray(blocks["rim_tri_b"])[0] - cross[iC:]).max() < 1.0e-12
    # the arc: every node on the ring circle, uniformly spaced in angle across the join
    arc = np.concatenate([np.asarray(blocks["rim_tri_t"])[:, 0],
                          np.asarray(blocks["rim_tri_q"])[1:, 0]])
    assert np.abs(np.linalg.norm(arc, axis=1) - reg["rim_inner"]).max() < 1.0e-9
    th = np.unwrap(np.arctan2(arc[:, 1], arc[:, 0]))
    assert np.abs(np.diff(th) - np.diff(th).mean()).max() < 1.0e-12


# ---------------------------------------------------------------------------
# 5. THE VERDICT, AND ITS CONTROL
# ---------------------------------------------------------------------------

def test_the_control_is_the_collapse_section_37_measured(genes):
    """0.0072-0.0082 on `rim_junction`, and 0.78 at the shipped blend.

    Re-measured rather than read: this is the number the 77x is a multiple of, and PART 7
    is the record of what happens when a baseline is quoted from a file instead.
    """
    for name in ("coarse", "medium"):
        faithful = tb.control(genes, name, 0.0)
        assert faithful["worst_block"] == "rim_junction"
        assert faithful["min_scaled_jacobian"] < 0.01
        assert not faithful["clears_min_sj_target"]
        shipped = tb.control(genes, name, 1.0)
        assert abs(shipped["min_scaled_jacobian"] - 0.7827) < 0.005
        assert shipped["clears_min_sj_target"]


def test_the_tri_block_clears_the_barrier_the_quad_could_not(genes):
    """The headline, re-measured at both configs, against the floor the optimizer enforces.

    `MIN_SJ_TARGET` is imported from `wheel_objective` rather than written down, because
    a floor quoted from memory in a study is a floor that drifts from the one the barrier
    actually applies.
    """
    for name, B, w in (("coarse", 10, (0.124, 0.751, 0.124)),
                       ("medium", 18, (0.072, 0.803, 0.124))):
        reg = tb.region(genes, name, blend=0.0)
        v = tb.sector_verdict(reg, B, w)
        assert v["all_blocks_valid"]
        assert v["non_positive_gauss_elements"] == 0
        assert v["mixed_sign_cells"] == 0
        assert v["min_scaled_jacobian"] > wo.MIN_SJ_TARGET
        assert v["min_scaled_jacobian"] > 0.5
        assert v["min_scaled_jacobian"] / tb.control(
            genes, name, 0.0)["min_scaled_jacobian"] > 50.0


def test_section_51s_probe_was_a_FLOOR_and_this_is_above_it(genes):
    """§51 said "0.25 is a floor rather than an estimate".  It was one.

    Pinned as the ORDERING rather than as a value: what §51 claimed is that a swept
    interior point could only do better than its own un-swept centroid, and a
    construction that ever came back UNDER 0.25 would mean the probe was measuring
    something else.
    """
    reg = tb.region(genes, "coarse", blend=0.0)
    swept = tb.cell(reg, 10, (0.124, 0.751, 0.124))
    centroid = tb.cell(reg, 10, (1 / 3, 1 / 3, 1 / 3))
    assert swept["min_scaled_jacobian"] > 0.25
    assert swept["min_scaled_jacobian"] > centroid["min_scaled_jacobian"]


def test_a_generated_interior_cannot_move_it(genes):
    """The worst corner is on a held boundary, so Winslow changes nothing.

    Same argument as PART 9's route-2 invariance and the same technique, imported from
    `study_fillet_block` rather than re-written.  It is what says the successor is a
    CURVED Y and not a better smoother.
    """
    reg = tb.region(genes, "coarse", blend=0.0)
    w = (0.124, 0.751, 0.124)
    raw = tb.cell(reg, 10, w)["min_scaled_jacobian"]
    smoothed = tb.winslow_column(reg, 10, w)["min_scaled_jacobian"]
    assert abs(smoothed - raw) < 1.0e-9


# ---------------------------------------------------------------------------
# 6. THE SCOPE — the half this does NOT deliver
# ---------------------------------------------------------------------------

def test_the_gene_box_is_what_this_does_NOT_deliver(report):
    """The genome sweep is a refusal, and it has to stay reported as one.

    This is the load-bearing negative: the faithful rim is not opt-in, so a construction
    that folds on a quarter of the gene box is not adoptable, and the one way this file's
    headline could become false is a genome sweep that quietly went green — by drawing
    fewer genomes, by re-sweeping `B` per genome, or by dropping the fixed-rule column.
    Each of those is asserted against.
    """
    for name, per in report["per_config"].items():
        g = per["genomes"]
        assert g["n_genomes"] >= 16, "the sweep must keep drawing four per orientation"
        assert len(g["groups"]) == 4, "all four flank orientations, or it is not the box"
        assert g["n_fixed_w_valid"] < g["n_genomes"], (
            f"{name}: the fixed rule is reported as HOLDING everywhere — if that is now "
            "true it is a finding and this test is the place to record it")
        assert g["n_best_w_valid"] < g["n_genomes"], (
            f"{name}: even a per-genome interior point does not reach every genome")
        assert g["n_fixed_w_valid"] <= g["n_best_w_valid"]
        assert g["all_seams_close"] is True, "the seams are not what fails"


def test_the_free_count_is_a_CONFIG_constant_and_not_a_genome_one(report):
    """`B` sets element counts, so it may not vary with the genome.

    A mesh whose element count depends on the design is a mesh that cannot be compared
    across a search, and the genome sweep would look far better if it were allowed to
    re-choose `B`.  The driver holds it; this says so out loud.
    """
    for per in report["per_config"].values():
        assert per["genomes"]["B"] == per["sector"]["B"]
        assert isinstance(per["genomes"]["B"], int)


def test_nothing_here_is_wired_into_the_mesh_the_tree_BUILDS(genes):
    """`sector_blocks` still returns seven blocks and the shipped blend is untouched.

    The tri-block is a MEASUREMENT of a construction, exactly as the eleven-block filleted
    sector was in §48 before §50 wired it.  Adopting the faithful rim moves the mesh under
    every genome and is a separate decision with its own baseline.
    """
    assert ww.UNCAP_DEFAULT == (True, 1.0)
    blocks = ww.sector_blocks(genes, ww.get_config("coarse"))
    assert len([k for k in blocks if k != "_thetas"]) == 7
    assert "rim_tri_t" not in blocks
    q = {k: fb.block_quality(np.asarray(v, float))
         for k, v in blocks.items() if k != "_thetas"}
    assert min(v["min_scaled_jacobian"] for v in q.values()) > wo.MIN_SJ_TARGET


# ---------------------------------------------------------------------------
# 7. THE ARTIFACT
# ---------------------------------------------------------------------------

def test_the_committed_report_describes_the_mesh_the_tree_BUILDS_TODAY(genes, report):
    """PART 7's freshness guard: re-measure the control and compare it to the file.

    A committed artifact whose driver takes a bare default rots the moment a default
    moves, and every test that READS the artifact reads the same stale number.  So this
    one re-derives the two controls and the chosen cell and holds the file to them.
    """
    for name, per in report["per_config"].items():
        fresh0 = tb.control(genes, name, 0.0)
        fresh1 = tb.control(genes, name, 1.0)
        assert abs(fresh0["min_scaled_jacobian"]
                   - per["control_faithful"]["min_scaled_jacobian"]) < 1.0e-9
        assert abs(fresh1["min_scaled_jacobian"]
                   - per["control_shipped"]["min_scaled_jacobian"]) < 1.0e-9
        reg = tb.region(genes, name, blend=0.0)
        v = tb.sector_verdict(reg, per["sector"]["B"], per["sector"]["w"])
        assert abs(v["min_scaled_jacobian"]
                   - per["sector"]["min_scaled_jacobian"]) < 1.0e-9
        assert v["n_blocks"] == 12 and v["n_seams"] == 17


def test_the_reports_self_checks_all_pass(report):
    """The driver gates its own exit on these; a stale artifact would carry a stale PASS."""
    sc = report["self_checks"]
    assert sc["pass"] is True
    for k, v in sc.items():
        assert v is True, k
    assert sc["algebra_reproduces_section_37"] is True


def test_the_report_is_the_committed_configs_and_the_shipped_genome(report):
    """The gate guard's other half: what is on disk is the run `make triblock` makes."""
    assert report["genome"] == "best_solution.json"
    assert sorted(report["configs"]) == ["coarse", "medium"]
    assert report["faithful_rim_blend"] == 0.0
    assert report["min_sj_target"] == pytest.approx(wo.MIN_SJ_TARGET)
