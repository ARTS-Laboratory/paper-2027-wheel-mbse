"""
Pins for the FILLETED MESH — FILLET_PLAN.md STEP 1b, PLAN.md §50.

`studies/study_fillet_block.py` and `tests/test_fillet_block.py` measure the BLOCKING:
eleven grids and fourteen seams, geometry and Jacobians, nothing assembled.  This file
pins what only assembly can answer — that `build_wheel(genes, cfg, fillet=True)` returns
a mesh that integrates, whose seams merged, whose boundary sets still mean what their
names say, and that is exactly twelve-fold periodic under a real solve.

AND IT PINS THE CONTROL JUST AS HARD.  `fillet=None` is the default and must stay
bit-identical: the shipped wheel, every study, every gate and every committed artifact
are the unfilleted mesh, and a fillet path that perturbs it by 1e-15 has changed the
answer to every question this project has asked.  `tests/test_golden.py` covers that
from the outside; the tests here cover the two mechanisms that could break it — the
block order and the boundary sets are now chosen per mesh rather than being module
constants.

THE SECOND CONSTRUCTION IS DELIBERATE, NOT LEGACY.  `fillet_blocking="spoke"` is
PART 3's arc-on-the-flank-edge construction, which §47 retired as a mesh and which
`make fillet` still measures: PART 6's fold table is a statement about THAT geometry and
would stop being reproducible if it were deleted.  Both are reachable and this file pins
that they are different.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import wheel_genome as wg              # noqa: E402
import wheel_objective as wo           # noqa: E402
import wheel_wheel as ww               # noqa: E402
import study_fillet_fold as ff         # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGS = ("coarse", "medium")


@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


@pytest.fixture(scope="module")
def filleted(genes):
    return {cfg: ww.build_wheel(genes, cfg, fillet=True) for cfg in CFGS}


@pytest.fixture(scope="module")
def plain(genes):
    return {cfg: ww.build_wheel(genes, cfg) for cfg in CFGS}


# ---------------------------------------------------------------------------
# THE CONTROL: `fillet=None` IS UNTOUCHED
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", CFGS)
def test_the_unfilleted_mesh_still_has_exactly_seven_blocks(genes, plain, cfg):
    """The block order is now chosen per mesh, and the default must not have moved.

    Named blocks rather than a count, because a count would pass if the fillet blocking
    were selected and happened to be trimmed to seven.
    """
    m = plain[cfg]
    assert tuple(dict.fromkeys(m.element_block.tolist())) == ww.BLOCK_ORDER
    assert m.fillet is None
    blocks = ww.sector_blocks(genes, cfg, fillet=None)
    assert tuple(k for k in blocks if not k.startswith("_")) == ww.BLOCK_ORDER


@pytest.mark.parametrize("cfg", CFGS)
def test_the_unfilleted_boundary_sets_are_unchanged(plain, cfg):
    """`_edge_sets` and `_node_sets` are now keyed on which blocking built the mesh.

    Pinned on the RADII the sets land on rather than on their sizes: a set that keeps its
    node count and moves to the wrong circle is the failure this guards, and it is
    exactly what laying the filleted rings out the other way round would have caused.
    """
    m = plain[cfg]
    xy = np.asarray(m.coords)
    r = {k: np.linalg.norm(xy[v], axis=1) for k, v in m.node_sets.items()}
    assert np.allclose(r["hub_tie"], ww.HUB_RADIUS_MM - ww.COLLAR_DEPTH_MM, atol=1e-9)
    assert np.allclose(r["rim_outer"], ww.RIM_OUTER_RADIUS_MM, atol=1e-9)
    assert np.allclose(r["rim_inner_free"], ww.rim_inner_radius(m.span_mm), atol=1e-9)


# ---------------------------------------------------------------------------
# THE FILLETED MESH ASSEMBLES
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", CFGS)
def test_the_filleted_mesh_INTEGRATES(filleted, cfg):
    """§44's criterion on the assembled mesh: `det J` at the Gauss points, not "it built".

    `build_wheel` returning is not the same statement as a mesh that integrates —
    `_orient_elements`' own docstring records that its corner shoelace misses sub-element
    folds in 21% of the meshes it accepts — so the check is the quadrature's.
    """
    m = filleted[cfg]
    assert ff.mesh_gauss_verdict(m)["non_positive_elements"] == 0
    q = ww.quality_report(m)
    assert q["n_inverted"] == 0
    assert q["min_scaled_jacobian"] > wo.MIN_SJ_TARGET, q["per_block"]


@pytest.mark.parametrize("cfg", CFGS)
def test_the_filleted_mesh_merged_its_seams(filleted, plain, cfg):
    """Every seam closed, and the ELEVEN blocks really did merge into one body.

    The seam error is the max distance between what an owner and a discarded duplicate
    said the same node was, so it is the assembly's own verdict rather than a re-derived
    one.  The merge COUNT is asserted too: fourteen seams that all report zero error
    while merging nothing would be a table that names edges nobody shares.
    """
    m = filleted[cfg]
    assert m.seam_error_mm < 1e-9, m.seam_error_mm
    assert m.n_merged > plain[cfg].n_merged


@pytest.mark.parametrize("cfg", CFGS)
def test_the_filleted_mesh_has_the_same_boundary_sets_on_the_same_circles(filleted, cfg):
    """The three FEA boundary sets must survive the re-cut, on their own radii.

    `hub_tie` is the tie to the rigid hub, `rim_outer` is where the ground pushes: if
    either moved by a layer the solve would still run and would answer a different
    problem.  `rim_inner_free` is the one that legitimately SHRINKS — the fillet's
    footprint on the ring circle is interior material now, and the free surface that
    replaced it is the fillet arc, which is not this set.
    """
    m = filleted[cfg]
    xy = np.asarray(m.coords)
    r = {k: np.linalg.norm(xy[v], axis=1) for k, v in m.node_sets.items()}
    assert np.allclose(r["hub_tie"], ww.HUB_RADIUS_MM - ww.COLLAR_DEPTH_MM, atol=1e-9)
    assert np.allclose(r["rim_outer"], ww.RIM_OUTER_RADIUS_MM, atol=1e-9)
    assert np.allclose(r["rim_inner_free"], ww.rim_inner_radius(m.span_mm), atol=1e-9)
    for k in ("hub_tie", "rim_outer"):
        assert m.node_sets[k].size == np.unique(m.node_sets[k]).size


def test_the_filleted_mesh_covers_the_whole_hub_tie_circle(filleted, plain):
    """The bore is a closed circle and the re-cut must not have left a gap in it.

    The hub's ring blocks changed both their angular span and their radial node count,
    and a set that is merely "the right radius" can still be missing an arc.  Counted
    against the unfilleted mesh's own set, which is the same circle at the same
    `n_weld` + `n_free`.
    """
    assert (filleted["coarse"].node_sets["hub_tie"].size
            == plain["coarse"].node_sets["hub_tie"].size)
    assert (filleted["coarse"].node_sets["rim_outer"].size
            == plain["coarse"].node_sets["rim_outer"].size)


@pytest.mark.parametrize("cfg", CFGS)
def test_the_fillet_material_is_labelled_spoke(filleted, cfg):
    """Region labels drive the loss terms, so the new blocks may not be unlabelled.

    Both fillet blocks are the SPOKE's material: the fillet is the spoke's corner
    rounded, and calling it hub or rim would move mass between the terms §8 prices.
    """
    m = filleted[cfg]
    for name in ("hub_fillet_a", "hub_fillet_b", "rim_fillet_a", "rim_fillet_b"):
        sel = m.element_block == name
        assert sel.any(), name
        assert set(m.element_region[sel]) == {"spoke"}
    assert set(m.element_region.tolist()) == {"spoke", "hub", "rim"}


# ---------------------------------------------------------------------------
# THE END-TO-END CHECK THAT ALREADY CAUGHT ONE REAL BUG
# ---------------------------------------------------------------------------

def test_the_filleted_axle_drop_is_exactly_12_fold_periodic(genes):
    """delta(phi) = delta(phi+30) on the FILLETED mesh, through a real solve.

    FILLET_PLAN.md Step 1 names this as the check to judge a filleted mesh on, and
    `tests/test_wheel_fea.py` records why: it exercises the mesh, the seams, the sector
    indexing, the load and the solve at once, and it has an exact expected answer.  A
    seam table whose `dk` is wrong, a block rotated into the wrong sector, or a boundary
    set that lost an arc all break it and few other things do.
    """
    import wheel_fem as fem
    a = fem.solve_wheel(ww.build_wheel(genes, "coarse", phase_deg=0.0,
                                       fillet=True))["axle_drop_mm"]
    b = fem.solve_wheel(ww.build_wheel(genes, "coarse", phase_deg=30.0,
                                       fillet=True))["axle_drop_mm"]
    assert abs(a / b - 1.0) < 1e-10, f"{a:.12f} vs {b:.12f}"


# ---------------------------------------------------------------------------
# THE TWO CONSTRUCTIONS, AND THE PATH THAT MUST REFUSE
# ---------------------------------------------------------------------------

def test_both_filleted_constructions_are_reachable_and_they_are_different(genes):
    """PART 3's is still measurable and still folds; PART 10's is eleven blocks and does not.

    `make fillet` sweeps the first one and PART 6's window (0.12-0.24 mm at `coarse`) is a
    statement about it.  Deleting it would make that measurement unreproducible, so it is
    kept behind `fillet_blocking="spoke"` and pinned here as DIFFERENT rather than merely
    present.
    """
    spoke = ww.sector_blocks(genes, "coarse", fillet=True, fillet_blocking="spoke")
    sector = ww.sector_blocks(genes, "coarse", fillet=True)
    assert tuple(k for k in spoke if not k.startswith("_")) == ww.BLOCK_ORDER
    assert tuple(k for k in sector if not k.startswith("_")) == ww.FILLETED_BLOCK_ORDER
    # the retired one still folds at the shipped radii, which is why it was retired
    assert ff.gauss_verdict(np.asarray(spoke["spoke"], float))[
        "non_positive_elements"] > 0
    assert ff.gauss_verdict(np.asarray(sector["spoke"], float))[
        "non_positive_elements"] == 0
    with pytest.raises(ValueError):
        ww.sector_blocks(genes, "coarse", fillet=True, fillet_blocking="winslow")


def test_the_differentiable_path_REFUSES_a_filleted_mesh(genes, filleted):
    """It would otherwise return coordinates that are neither this mesh's nor an error.

    `mesh_coords` rebuilds the sector WITHOUT `fillet` and indexes it with `mesh.owners`.
    A filleted mesh has more raw nodes and a different ownership, so the silent answer is
    a different mesh's coordinates gathered through this one's index — which is worse
    than either a wrong number or a crash, because it is a plausible one.
    """
    m = filleted["coarse"]
    with pytest.raises(NotImplementedError):
        ww.mesh_coords(genes, m, xp=np)
    with pytest.raises(NotImplementedError):
        ww.coord_fn(m)
    # and the unfilleted path is untouched
    plain_mesh = ww.build_wheel(genes, "coarse")
    got = np.asarray(ww.mesh_coords(genes, plain_mesh, xp=np))
    assert np.abs(got - np.asarray(plain_mesh.coords)).max() < 1e-12


def test_the_area_reference_is_WITHHELD_for_a_filleted_mesh(genes, filleted, plain):
    """`modelled_area_reference` models the unfilleted region, so it may not be compared.

    The fillets are not a rounding: they add 8.0-8.1% of the modelled area at the shipped
    radii, which `error_vs_modelled` would book as a discretisation residual against a
    reference that is 2e-4.  Withheld with a reason rather than reported wrong, and the
    measured half is still returned.
    """
    a = ww.area_report(filleted["coarse"])
    assert "reference_unavailable_because" in a
    assert "error_vs_modelled" not in a
    assert a["meshed_mm2"] > 0.0
    a0 = ww.area_report(plain["coarse"])
    assert abs(a0["error_vs_modelled"]) < 1e-2
    added = a["total_modelled_mm2"] / a0["total_modelled_mm2"] - 1.0
    assert 0.05 < added < 0.12, added


@pytest.mark.parametrize("R", [(0.0, 3.0), (0.6636, 0.0)])
def test_the_filleted_blocking_refuses_a_switched_off_end(genes, R):
    """`fillet_blocking="spoke"` lets a radius be zero; the sector blocking does not.

    Not an oversight: the re-cut moves the spoke's end, the junction's left edge and both
    ring blocks together, so "no fillet at this end" is a different blocking rather than
    the same one at R = 0.  Refused with that reason instead of building something whose
    seams happen to close.  The old construction is shown accepting the same argument —
    as a BLOCKING; whether the mesh it makes then folds is PART 6's subject and not this
    test's.
    """
    with pytest.raises(ValueError):
        ww.sector_blocks(genes, "coarse", fillet=R)
    blocks = ww.sector_blocks(genes, "coarse", fillet=R, fillet_blocking="spoke")
    assert tuple(k for k in blocks if not k.startswith("_")) == ww.BLOCK_ORDER
