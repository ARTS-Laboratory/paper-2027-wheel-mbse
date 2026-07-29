"""
M2b verification of the full 360-degree assembled mesh.

The seam test is the one that matters and the reason is worth restating: a seam
mismatch produces a mesh that plots correctly, has a positive Jacobian everywhere,
solves without complaint, and models a wheel with twelve cracks in it.  No solver
diagnostic would notice.  So there are three independent checks on connectivity here —
exact shared coordinates, a single connected component, and exact 12-fold periodicity —
because each catches a different way of getting it wrong.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import wheel_fea as wf            # noqa: E402
import wheel_fem as fem           # noqa: E402
import wheel_genome as wg         # noqa: E402
import wheel_mesh as wm           # noqa: E402
import wheel_wheel as ww          # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


@pytest.fixture(scope="module")
def mesh(genes):
    return ww.build_wheel(genes, "coarse")


# ---------------------------------------------------------------------------
# SEAMS — three independent checks
# ---------------------------------------------------------------------------

def test_seam_coordinates_agree_at_machine_precision(mesh):
    """What an owning and a non-owning block independently computed must agree.

    1e-10 mm is the plan's number.  It is met by four orders of magnitude, and that is
    not luck: every block that touches the spoke band goes through the single
    `wheel_mesh.band_sampler`, and the ring-crossing stations are Newton-refined against
    that same sampler so the shared corners land on the ring circles exactly.  Two
    mathematically equivalent constructions that interpolate different quantities
    disagree at O(h_curve^2) — measured, 3.9e-4 mm, which is what this test read before
    the sampler was made shared.
    """
    assert mesh.seam_error_mm < 1e-10, f"{mesh.seam_error_mm:.3e} mm"


def test_the_mesh_is_one_connected_component(mesh):
    """Element adjacency must connect every element to every other.

    Independent of the coordinate check: coordinates could agree perfectly while the
    seam declarations failed to actually MERGE the node ids, leaving two nodes at the
    same location and a crack between them.  This test would fail and nothing else here
    would.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    conn = mesh.conn
    n_el = conn.shape[0]
    rows = np.repeat(np.arange(n_el), conn.shape[1])
    m = coo_matrix((np.ones(rows.size), (rows, conn.ravel())),
                   shape=(n_el, mesh.n_nodes))
    adj = (m @ m.T)
    n_comp, _ = connected_components(adj, directed=False)
    assert n_comp == 1, f"{n_comp} disconnected components — the wheel has cracks"


def test_duplicate_node_positions_do_not_exist(mesh):
    """No two distinct global nodes may occupy the same point.

    A leftover duplicate is exactly the signature of a seam that was declared but not
    merged, and it is invisible to the coordinate check (which only looks at pairs it
    was told about).
    """
    xy = np.asarray(mesh.coords)
    order = np.lexsort((xy[:, 1], xy[:, 0]))
    d = np.linalg.norm(np.diff(xy[order], axis=0), axis=1)
    assert d.min() > 1e-9, (
        f"{int((d < 1e-9).sum())} coincident node pairs remain unmerged "
        f"(closest {d.min():.3e} mm)")


def test_twelve_fold_periodicity_is_exact(mesh):
    """Rotating the node set by 30 degrees must map it onto itself.

    Catches sector-indexing bugs that the other seam checks cannot: if sector k were
    built from the wrong rotation, or if the inter-sector seam wired sector k to k+2, the
    coordinates would still agree pairwise and the mesh would still be connected.
    """
    xy = np.asarray(mesh.coords)
    th = np.radians(ww.SECTOR_DEG)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    rotated = xy @ R.T
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy).query(rotated)
    assert d.max() < 1e-9, f"12-fold periodicity off by {d.max():.3e} mm"


@pytest.mark.parametrize("seed", [1, 7])
def test_seams_hold_across_the_design_space(seed):
    """The seam guarantee must survive genomes other than the shipped one.

    Only a handful of genomes, because each build is a full 4704-element assembly; the
    200-genome version is `study_wheel_mesh.py`, whose recorded worst case is 3.6e-14 mm.
    """
    import study_wheel_mesh as sww
    vectors, _ = sww.sample_feasible(3, "coarse", seed)
    assert vectors, "sampler produced nothing — the feasibility filter changed"
    checked = 0
    for vec in vectors:
        if max(float(a) for a in ww.arrival_angles(vec, ww.get_config("smoke"))) \
                > ww.MAX_ARRIVAL_DEG:
            continue
        m = ww.build_wheel(vec, "smoke")
        assert m.seam_error_mm < 1e-10, f"{m.seam_error_mm:.3e} mm"
        checked += 1
    if checked == 0:
        pytest.skip("no sampled genome satisfied the arrival-angle constraint")


# ---------------------------------------------------------------------------
# THE PARTITION
# ---------------------------------------------------------------------------

def test_area_matches_the_region_the_mesh_claims_to_model(genes):
    """Mesh area plus the un-meshed rigid hub core, against `modelled_area_reference`.

    The reference is the area of `hub_disk | rim_band | 12 bands clipped to the annulus`,
    DERIVED from the frame constants and the genome down an independent geometric path —
    the exporter's finite-difference offset normals, integrated by exact shoelace plus
    exact circular sectors, against the mesh's analytic-hodograph normals and Q9 Gauss.
    It is not the shipped STEP's region: `wheel_step_export._embed` adds material this
    mesh deliberately does not model, checked separately below.

    It used to be the hardcoded 2469.836, which quietly became a reference to a wheel
    that no longer exists the moment the rim band was thickened.  That is why the check
    is now derived: the test has to fail when the mesh is wrong, not when the design
    changes.
    """
    rep = ww.area_report(ww.build_wheel(genes, "coarse"))
    assert abs(rep["error_vs_modelled"]) < 0.005, (
        f"{rep['total_modelled_mm2']:.4f} vs {rep['reference_modelled_mm2']:.4f} "
        f"({rep['error_vs_modelled']:+.4%})")


def test_the_area_reference_is_derived_not_transcribed(genes):
    """The reference must MOVE with the frame constants, or it is not a reference.

    A swept `rim_outer` changes the region by an exact closed-form annulus, so the
    reference must change by exactly that and no more.  Catches a reference that has been
    re-hardcoded, which is the failure this whole derivation exists to prevent.
    """
    a = ww.modelled_area_reference(genes)
    b = ww.modelled_area_reference(genes, rim_outer=ww.RIM_OUTER_RADIUS_MM + 1.0)
    expect = np.pi * ((ww.RIM_OUTER_RADIUS_MM + 1.0) ** 2 - ww.RIM_OUTER_RADIUS_MM ** 2)
    assert abs((b["total_mm2"] - a["total_mm2"]) - expect) < 1e-9
    # And the closed-form pieces must be exactly closed-form.
    assert abs(a["hub_disk_mm2"] - np.pi * ww.HUB_RADIUS_MM ** 2) < 1e-9


def test_the_polygon_disk_clipper_is_right_on_shapes_with_known_areas():
    """`_clip_polygon_to_disk` carries the reference, so it gets its own check.

    Three regimes with exact answers: polygon inside the disk, disk inside the polygon
    (the containment branch — no vertex is inside and no crossing exists, and returning
    zero there is wrong in a way that shows up only on the reference), and the partial
    overlap that actually occurs.
    """
    square = np.array([[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]])
    dense = np.vstack([square[i] + np.linspace(0, 1, 2000, endpoint=False)[:, None]
                       * (square[(i + 1) % 4] - square[i]) for i in range(4)])
    assert abs(ww._clip_polygon_to_disk(dense, 10.0) - 36.0) < 1e-9
    assert abs(ww._clip_polygon_to_disk(dense, 2.0) - 4.0 * np.pi) < 1e-9
    # Partial overlap: the disk pokes past all four sides while the square's corners
    # poke past the disk, so the answer is the disk less four circular segments.  This is
    # the regime the spoke band is actually in, and it is exact to roundoff because the
    # arcs are integrated analytically rather than chorded.
    R, h = 3.5, 3.0
    exact = np.pi * R ** 2 - 4.0 * (R ** 2 * np.arccos(h / R)
                                    - h * np.sqrt(R ** 2 - h ** 2))
    assert abs(ww._clip_polygon_to_disk(dense, R) - exact) < 1e-12, (
        f"{ww._clip_polygon_to_disk(dense, R):.12f} vs {exact:.12f}")


def test_area_converges_under_refinement(genes):
    """Refinement must drive the area error toward zero, not merely leave it small.

    Two error sources under-cut the true region (the Coons patches and the polygonal
    ring arcs), so a single config agreeing to 0.5% could be two errors cancelling.
    """
    errs = [abs(ww.area_report(ww.build_wheel(genes, c))["error_vs_modelled"])
            for c in ("smoke", "coarse", "medium")]
    assert errs[1] < errs[0] and errs[2] < errs[1], f"not converging: {errs}"


def test_the_embed_difference_from_the_shipped_step_is_the_known_amount(genes):
    """Pin the deliberate ~2% modelling difference so it cannot silently change.

    If someone models `_embed` after all, or changes COLLAR_DEPTH_MM, or alters a ring
    radius, this moves — and it should be a decision, not a surprise.  Cross-checked
    against the STEP manifest's own mass below, which is a different measurement of the
    same difference through a different kernel.
    """
    rep = ww.area_report(ww.build_wheel(genes, "medium"))
    assert -0.025 < rep["error_vs_shipped_step"] < -0.014, (
        f"{rep['error_vs_shipped_step']:+.4%} — expected about -2%")


def test_region_areas_are_individually_right(genes):
    """Each region separately, so a compensating error in two of them cannot hide.

    The rings have exact closed forms.  The spoke region is checked against the
    reference's own clipped-band figure — a different construction (finite-difference
    offset normals, exact shoelace) from the mesh's (analytic hodograph, Q9 Gauss), which
    is what makes the comparison worth making.
    """
    mesh = ww.build_wheel(genes, "medium")
    rep = ww.area_report(mesh)["by_region_mm2"]
    ref = ww.modelled_area_reference(genes)
    hub = np.pi * (ww.HUB_RADIUS_MM ** 2 - (ww.HUB_RADIUS_MM - ww.COLLAR_DEPTH_MM) ** 2)
    rim = np.pi * (ww.RIM_OUTER_RADIUS_MM ** 2 - ww.RIM_RADIUS_MM ** 2)
    assert abs(rep["hub"] / hub - 1) < 2e-3, f"collar {rep['hub']:.4f} vs {hub:.4f}"
    assert abs(rep["rim"] / rim - 1) < 2e-3, f"rim {rep['rim']:.4f} vs {rim:.4f}"
    assert abs(rep["spoke"] / ref["spokes_mm2"] - 1) < 5e-3, (
        f"{rep['spoke']:.4f} vs {ref['spokes_mm2']:.4f}")


# ---------------------------------------------------------------------------
# VALIDITY AND ORIENTATION
# ---------------------------------------------------------------------------

def test_every_element_is_positively_oriented(mesh):
    """Not cosmetic: a negative-Jacobian element contributes NEGATIVE stiffness.

    `_orient_elements` flips whole blocks whose (i, j) indexing is left-handed — a polar
    block indexed (theta, r) is, since r_hat x theta_hat = +z — and then asserts every
    element individually, so a genuinely folded element in a well-oriented block still
    fails rather than being quietly reversed.
    """
    area = ww._signed_area(np.asarray(mesh.coords), mesh.conn)
    assert area.min() > 0, f"{int((area <= 0).sum())} non-positive elements"


def test_scaled_jacobian_clears_the_floor(mesh):
    q = ww.quality_report(mesh)
    assert q["min_scaled_jacobian"] > 0.2, q["per_block"]
    assert q["n_inverted"] == 0


def test_fem_element_table_matches_the_wheel_connectivity():
    """`wheel_fem._NODE_IJ` must match `grid_connectivity`, not just the spoke block.

    The full wheel emits elements from `grid_connectivity` for seven different grid
    shapes.  If the FE node ordering only happened to match the spoke block's shape,
    every ring and junction element would integrate a scrambled geometry while still
    being symmetric and positive definite.
    """
    for order in (1, 2):
        for ni, nj in ((order + 1, order + 1), (2 * order + 1, 3 * order + 1)):
            conn = wm.grid_connectivity(ni, nj, order)[0]
            offsets = np.array([(int(n) // nj, int(n) % nj) for n in conn])
            assert np.array_equal(offsets, fem._NODE_IJ[order]), (
                f"order {order}, grid {ni}x{nj}: {offsets.tolist()}")


def test_node_sets_lie_on_the_radii_they_claim(mesh):
    xy = np.asarray(mesh.coords)
    r = np.linalg.norm(xy, axis=1)
    for name, expect in (("hub_tie", ww.HUB_RADIUS_MM - ww.COLLAR_DEPTH_MM),
                         ("rim_outer", ww.RIM_OUTER_RADIUS_MM),
                         ("rim_inner_free", ww.RIM_RADIUS_MM)):
        got = r[mesh.node_sets[name]]
        assert np.abs(got - expect).max() < 1e-9, (
            f"{name}: radii span {got.min():.6f}..{got.max():.6f}, expected {expect}")


def test_rim_band_has_at_least_three_elements_through_its_thickness():
    """The 1.1 mm rim band is what the whole M4 gate is about.

    Two elements cannot represent bending through the thickness, and under-resolving the
    one component predicted to dominate compliance would make the gate's headline number
    meaningless.
    """
    for name in ("coarse", "medium", "fine"):
        assert ww.CONFIGS[name].n_rim_r >= 3, name


# ---------------------------------------------------------------------------
# THE TWO GENOME-DEPENDENT DECISIONS
# ---------------------------------------------------------------------------

def test_flank_orientation_is_not_a_constant():
    """Both ends' straddling flank must be computed, not assumed.

    Hardcoding the shipped genome's (+1, +1) rejected 44 of 60 feasible genomes, because
    a spoke may bulge either way and an S-shaped centerline makes the two ends
    independent.  This test asserts the design space actually contains other
    combinations, so the computation cannot be "simplified" back to a constant.
    """
    import study_wheel_mesh as sww
    cfg = ww.get_config("smoke")
    vectors, _ = sww.sample_feasible(24, "coarse", 3)
    seen = {ww.flank_orientation(v, cfg) for v in vectors}
    assert len(seen) > 1, (
        f"only {seen} present in 24 feasible genomes — either the sampler changed or "
        f"the orientation really is constant, which would make this test pointless")


def test_flank_orientation_agrees_with_the_geometry(genes):
    """The chosen flank must be the one that is genuinely inside/outside its ring."""
    cfg = ww.get_config("coarse")
    eta_hub, eta_rim = ww.flank_orientation(genes, cfg)
    sample, _ = ww.global_sampler(np.asarray(genes), cfg)

    def radius(s, eta):
        p = sample(np.asarray(s), np.asarray(eta))
        return float(np.hypot(p[0], p[1]))

    assert radius(0.0, eta_hub) < ww.HUB_RADIUS_MM < radius(0.0, -eta_hub)
    assert radius(1.0, -eta_rim) < ww.RIM_RADIUS_MM < radius(1.0, eta_rim)


def test_arrival_angle_predicts_junction_quality():
    """The arrival-angle constraint must actually be the thing that matters.

    The relationship runs the opposite way from intuition — a near-TANGENT arrival gives
    excellent junction blocks and a near-RADIAL one degenerates them, because the end
    cross-section is normal to the centerline.  Asserting the correlation keeps that
    finding from being re-derived backwards, and keeps `MAX_ARRIVAL_DEG` attached to
    evidence.
    """
    import study_wheel_mesh as sww
    cfg = ww.get_config("smoke")
    vectors, _ = sww.sample_feasible(24, "coarse", 5)
    arr, sj = [], []
    for vec in vectors:
        a = max(float(x) for x in ww.arrival_angles(vec, cfg))
        try:
            q = ww.quality_report(ww.build_wheel(vec, "smoke"))["min_scaled_jacobian"]
        except ValueError:
            q = -1.0
        arr.append(a)
        sj.append(q)
    arr, sj = np.array(arr), np.array(sj)
    if np.ptp(arr) < 20.0:
        pytest.skip("sampled arrival angles too clustered to resolve a correlation")
    rho = float(np.corrcoef(arr, sj)[0, 1])
    assert rho < -0.5, f"correlation {rho:+.3f} — expected strongly negative"
    keep = arr <= ww.MAX_ARRIVAL_DEG
    if keep.any():
        assert sj[keep].min() > 0.2, (
            f"a genome inside MAX_ARRIVAL_DEG={ww.MAX_ARRIVAL_DEG} still gave "
            f"minSJ {sj[keep].min():.4f}")


def test_a_spoke_that_misses_its_ring_is_refused(genes):
    """`ring_station` must raise rather than silently build a hinge.

    Past ~80 degrees of arrival both flanks lie outside the ring circle and the spoke
    touches its ring at the single centerline point.  Meshing that would model a pin
    joint while `wheel_fea`'s `fixed_guided` boundary condition assumes a moment
    connection.
    """
    cfg = ww.get_config("smoke")
    sample, s_dense = ww.global_sampler(np.asarray(genes), cfg)
    with pytest.raises(ValueError, match="never crosses"):
        ww.ring_station(sample, s_dense, 5.0, 1.0, 0)


# ---------------------------------------------------------------------------
# THE GUARDS
# ---------------------------------------------------------------------------

def test_coons_corner_mismatch_is_an_error():
    """A swapped or unreversed boundary edge folds the patch; it must not be accepted."""
    bottom = np.array([[0.0, 0.0], [1.0, 0.0]])
    top = np.array([[0.0, 1.0], [1.0, 1.0]])
    left = np.array([[0.0, 0.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [1.0, 1.0]])
    ww.coons_patch(bottom, top, left, right)                 # the consistent case
    with pytest.raises(ValueError, match="corner mismatch"):
        ww.coons_patch(bottom, top, left, right[::-1])
    with pytest.raises(ValueError, match="opposite edges disagree"):
        ww.coons_patch(bottom, top, left, right[:1])


def test_coons_patch_reproduces_a_bilinear_region_exactly():
    """On a region with straight edges the patch must be the exact bilinear map."""
    nu, nv = 5, 4
    corners = np.array([[0.0, 0.0], [3.0, 0.5], [3.5, 2.0], [0.2, 1.8]])
    u = np.linspace(0, 1, nu)[:, None]
    v = np.linspace(0, 1, nv)[:, None]
    lin = lambda a, b, w: (1 - w) * a + w * b
    got = ww.coons_patch(lin(corners[0], corners[1], u),
                         lin(corners[3], corners[2], u),
                         lin(corners[0], corners[3], v),
                         lin(corners[1], corners[2], v))
    U, V = np.meshgrid(u.ravel(), v.ravel(), indexing="ij")
    want = ((1 - U)[..., None] * (1 - V)[..., None] * corners[0]
            + U[..., None] * (1 - V)[..., None] * corners[1]
            + U[..., None] * V[..., None] * corners[2]
            + (1 - U)[..., None] * V[..., None] * corners[3])
    assert np.abs(got - want).max() < 1e-14


def test_a_seam_node_count_mismatch_is_an_error(genes):
    """A WheelConfig that violates a seam invariant must fail loudly.

    The junction's transverse direction is forced to `n_thick` by construction, so the
    reachable way to break a seam is `n_weld`, which is shared between the junction arc
    and the ring weld block.  Corrupting the seam table is the direct test.
    """
    cfg = ww.get_config("smoke")
    good = ww._seam_table((1.0, 1.0), {"hub_junction": (0.3, 0.0),
                                       "rim_junction": (0.2, 0.0)})
    assert len(good) == 8
    bad = list(good)
    bad[0] = ("spoke", "j0", "hub_junction", "i0", 0, False)   # wrong side: n differs
    import unittest.mock as mock
    with mock.patch.object(ww, "_seam_table", return_value=tuple(bad)):
        with pytest.raises(ValueError, match="nodes"):
            ww.build_wheel(genes, cfg)


def test_config_rejects_nonsense():
    with pytest.raises(ValueError, match="n_span"):
        ww.WheelConfig("bad", 0, 4, 10, 3, 10, 3, 10)
    with pytest.raises(ValueError, match="order"):
        ww.WheelConfig("bad", 8, 4, 10, 3, 10, 3, 10, order=3)
    with pytest.raises(KeyError):
        ww.get_config("nope")


# ---------------------------------------------------------------------------
# DIFFERENTIABILITY
# ---------------------------------------------------------------------------

def test_sector_blocks_are_differentiable_in_the_genes(genes):
    """The traced half of the assembly must survive jax and produce finite gradients.

    Only `sector_blocks` is checked, not `build_wheel`: the union-find, the orientation
    decision, and the validity guards are deliberately static/eager, and M7 is where a
    fully jitted path gets built.  What matters now is that no node coordinate is
    computed by something `jax.grad` cannot see through — in particular the Newton
    refinement of the ring stations, which is unrolled precisely so that it can be.
    """
    jax = pytest.importorskip("jax")
    import jax_config  # noqa: F401
    import jax.numpy as jnp

    cfg = ww.get_config("smoke")
    orient = ww.flank_orientation(genes, cfg)

    def total(v):
        blocks = ww.sector_blocks(v, cfg, xp=jnp, orientation=orient)
        blocks.pop("_thetas")
        return sum(jnp.sum(b ** 2) for b in blocks.values())

    g = np.asarray(jax.grad(total)(jnp.asarray(genes)))
    assert np.isfinite(g).all(), g
    # The 12 shape/thickness genes must all move the mesh; R_hub and R_rim are fillet
    # radii and correctly do not reach it.
    assert (np.abs(g[:12]) > 0).all(), g[:12]
    assert np.allclose(g[12:], 0.0)


def test_numpy_and_jax_agree_on_the_node_coordinates(genes):
    jax = pytest.importorskip("jax")            # noqa: F841
    import jax_config  # noqa: F401
    import jax.numpy as jnp

    cfg = ww.get_config("smoke")
    orient = ww.flank_orientation(genes, cfg)
    a = ww.sector_blocks(genes, cfg, xp=np, orientation=orient)
    b = ww.sector_blocks(jnp.asarray(genes), cfg, xp=jnp, orientation=orient)
    a.pop("_thetas")
    b.pop("_thetas")
    for name in a:
        err = np.abs(np.asarray(a[name]) - np.asarray(b[name])).max()
        assert err < 1e-9, f"{name}: numpy vs jax differ by {err:.3e} mm"
