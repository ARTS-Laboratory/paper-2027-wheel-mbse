"""The spoke-block mesh: structure, validity, area, and differentiability.

The mesh is the piece where a defect is quietest.  An inverted element still solves, and
because negative Jacobian means negative stiffness, the optimizer reads it as free
compliance and heads straight for it.  So this file pins:

  structure       connectivity, node counts, boundary sets, element winding
  area            against wheel_fea's own mass integral — independent code   < 0.5 %
  validity        no inverted elements on the shipped design, and the fold
                  margin genuinely predicts inversion across the design space
  gradients       node coordinates differentiate w.r.t. the genes            1e-6
"""

import json
import os

import numpy as np
import pytest

import wheel_fea as W
import wheel_genome as GN
import wheel_geometry as G
import wheel_mesh as M

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPAN = W.HUB_RIM_SPAN_MM


@pytest.fixture(scope="module")
def vec():
    with open(os.path.join(HERE, "best_solution.json")) as fh:
        return GN.genes_to_vector(json.load(fh)["genes"])


@pytest.fixture(scope="module")
def reference_area(vec):
    """Spoke cross-sectional area implied by wheel_fea's own mass integral.

    `total_mass_g = sum(t * width * ds) * rho * n_spokes`, so inverting it gives the area
    of ONE spoke — computed by completely different code (a beam-style line integral, not
    a mesh), which is what makes it a real check.

    Evaluated NOW rather than read out of `best_solution.json`, so the reference and the
    mesh are always in the same geometric frame.  Reading the recorded metric instead
    made this test fail the moment `RIM_RADIUS_MM` moved — not because the mesh was
    wrong, but because the artifact still described a 36.2 mm span while the mesh built a
    35.8 mm one.
    """
    metrics, _ = W.evaluate_design(vec)
    return metrics["total_mass_g"] / (
        W.DENSITY_PLA * W.NUMBER_OF_SPOKES * W.SPOKE_WIDTH_MM)


# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["smoke", "coarse", "medium", "fine"])
def test_config_shapes_are_consistent(name):
    cfg = M.get_config(name)
    conn = M.spoke_block_connectivity(cfg)
    assert conn.shape == (cfg.n_elements, cfg.nodes_per_element)
    assert conn.min() == 0
    assert conn.max() == cfg.n_nodes - 1


@pytest.mark.parametrize("order,npe", [(1, 4), (2, 9)])
def test_every_node_is_used(order, npe):
    """A node no element references is a free-floating DOF: the stiffness matrix is
    then singular and the solve fails in a way that points nowhere near the cause."""
    cfg = M.MeshConfig("t", 7, 3, order=order)
    conn = M.spoke_block_connectivity(cfg)
    assert conn.shape[1] == npe
    assert set(np.unique(conn)) == set(range(cfg.n_nodes))


def test_elements_are_wound_consistently(vec):
    """All element areas must share a sign.  A mixed sign means some elements are
    numbered backwards, which contributes negative stiffness."""
    cfg = M.get_config("coarse")
    X = M.flatten(M.spoke_block_coords_from_vector(vec, cfg, SPAN))
    area = M.element_areas(X, M.spoke_block_connectivity(cfg))
    assert np.all(area > 0) or np.all(area < 0)


def test_q9_corners_are_the_q4_element(vec):
    """Q9's first four nodes must be its corners, in the same order a Q4 would use —
    the quality metrics slice `conn[:, :4]` and rely on it."""
    q4 = M.spoke_block_connectivity(M.MeshConfig("a", 5, 2, order=1))
    q9 = M.spoke_block_connectivity(M.MeshConfig("b", 5, 2, order=2))
    assert q4.shape[0] == q9.shape[0]
    # Same topology, different node numbering (Q9's grid is twice as dense), so compare
    # the geometric corners rather than the ids.
    cfg1, cfg2 = M.MeshConfig("a", 5, 2, order=1), M.MeshConfig("b", 5, 2, order=2)
    X1 = M.flatten(M.spoke_block_coords_from_vector(vec, cfg1, SPAN))
    X2 = M.flatten(M.spoke_block_coords_from_vector(vec, cfg2, SPAN))
    assert np.abs(X1[q4] - X2[q9[:, :4]]).max() < 1e-12


def test_boundary_sets_are_the_block_edges():
    cfg = M.get_config("coarse")
    b = M.boundary_nodes(cfg)
    assert len(b["root"]) == len(b["tip"]) == cfg.n_node_thick
    assert len(b["flank_top"]) == len(b["flank_bot"]) == cfg.n_node_span
    # Corners belong to two sets; everything else is disjoint.
    assert set(b["root"]) & set(b["tip"]) == set()
    assert len(set(b["root"]) & set(b["flank_top"])) == 1


def test_root_and_tip_lie_on_the_end_cross_sections(vec):
    """The root nodes must sit on the s=0 cross-section and the tip nodes on s=1 —
    these are the surfaces the beam model clamps and guides, so the FE boundary has to
    be the same one or the comparison is meaningless."""
    cfg = M.get_config("coarse")
    X = M.spoke_block_coords_from_vector(vec, cfg, SPAN)
    curve, _ = G.bezier_centerline(*[vec[i] for i in range(8)],
                                   span_mm=SPAN, num_points=cfg.n_curve)
    assert np.abs(X[0].mean(axis=0) - curve[0]).max() < 1e-9
    assert np.abs(X[-1].mean(axis=0) - curve[-1]).max() < 1e-9
    # ...and each end cross-section is straight.
    for row in (X[0], X[-1]):
        d = row[-1] - row[0]
        d = d / np.linalg.norm(d)
        offs = (row - row[0]) - np.outer((row - row[0]) @ d, d)
        assert np.abs(offs).max() < 1e-9


def test_nodes_are_spaced_uniformly_by_arc_length(vec):
    """The block is gridded on arc length, not Bezier parameter.  If it were gridded on
    the parameter, elements would bunch where the curve happens to be parameterised
    densely rather than where the geometry needs them."""
    cfg = M.get_config("coarse")
    X = M.spoke_block_coords_from_vector(vec, cfg, SPAN)
    mid = X[:, cfg.n_node_thick // 2, :]
    step = np.linalg.norm(np.diff(mid, axis=0), axis=1)
    assert step.std() / step.mean() < 2e-3


# ---------------------------------------------------------------------------
# AREA — the independent cross-check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,tol_pct", [
    ("smoke", 0.5), ("coarse", 0.5), ("medium", 0.1), ("fine", 0.01),
])
def test_mesh_area_matches_the_mass_integral(vec, reference_area, name, tol_pct):
    cfg = M.get_config(name)
    X = M.flatten(M.spoke_block_coords_from_vector(vec, cfg, SPAN))
    area = M.element_areas(X, M.spoke_block_connectivity(cfg)).sum()
    err = abs(area - reference_area) / reference_area * 100
    assert err < tol_pct, f"{name}: {area:.5f} vs {reference_area:.5f} mm2 ({err:.4f}%)"


def _area_sequence(vec, ns_list, n_thick=8):
    out = []
    for ns in ns_list:
        cfg = M.MeshConfig("r", ns, n_thick, order=2, n_curve=max(600, 4 * ns))
        X = M.flatten(M.spoke_block_coords_from_vector(vec, cfg, SPAN))
        out.append(float(M.element_areas(X, M.spoke_block_connectivity(cfg)).sum()))
    return np.array(out)


def test_area_converges_second_order(vec):
    """Error must fall ~4x per refinement.  A polygonal approximation of a curved
    region is O(h^2); a different observed rate would mean the resampling or the
    normals are wrong, not merely coarse.

    MEASURED SELF-REFERENCED, against the sequence's own successive differences rather
    than against `reference_area`.  That is not a weakening — it is the only way to
    measure this quantity at all, and the version that used `reference_area` was
    measuring something else.

    `reference_area` is `wheel_fea`'s beam-style line integral: independent code, which
    is exactly what makes it valuable as a CROSS-CHECK (the test above), and also an
    approximation carrying its own quadrature error.  Once the mesh error falls to that
    error's level, `|area - ref|` stops being the mesh error and the observed order is
    meaningless.  Extending the sweep two more levels shows it plainly (§14):

        genome     vs beam ref                        self-referenced
        350f4c7    1.986 2.061 2.355  5.103 -3.300    1.962 1.979 2.000 1.998
        36aed36    1.993 2.033 2.187  3.158  0.029    1.979 1.986 2.001 2.001

    An order of 5.1 and then MINUS 3.3 is not a convergence rate; it is the signature of
    a difference of two discretizations passing through zero.  The self-referenced column
    is flat at 2.000 on both genomes, which is the answer this test was always after.

    Nothing about the promoted genome broke this.  The beam reference sits 3.9e-5 mm^2
    from the mesh's own Richardson limit, and the mesh error at ns=512 is 1.0e-4 — only
    2.6x above it.  On the GA/beam genome the same margin is 5.4x, enough to squeak in at
    2.187.  The promoted wheel has a smaller cross-section (52.9 vs 145.7 mm^2), so its
    absolute mesh error reaches the reference's floor one refinement sooner.  The old
    genome was already one level from failing this.
    """
    a = _area_sequence(vec, (64, 128, 256, 512, 1024))
    d = np.diff(a)
    assert np.all(np.abs(d[1:]) < np.abs(d[:-1])), (
        f"the area sequence is not settling monotonically: {a}")
    orders = [float(np.log2(d[i] / d[i + 1])) for i in range(len(d) - 1)]
    assert all(1.7 < r < 2.3 for r in orders), f"observed orders {orders}"


def test_the_mass_integral_agrees_with_the_meshs_own_limit(vec, reference_area):
    """The cross-code claim that `test_area_converges_second_order` used to carry as a
    passenger, stated on its own and to a real tolerance.

    Richardson-extrapolate the mesh sequence to h -> 0 and compare THAT to the beam-style
    line integral.  This is the "independent code" check the file header advertises, and
    separating it from the convergence-rate claim is what lets both be tight: the order
    is 2.000 and the two integrals agree to well under a part in 10^5.
    """
    a = _area_sequence(vec, (64, 128, 256, 512, 1024))
    d = np.diff(a)
    order = float(np.log2(d[-2] / d[-1]))
    limit = a[-1] + d[-1] / (2.0 ** order - 1.0)
    rel = abs(reference_area - limit) / limit
    assert rel < 1e-5, (
        f"beam integral {reference_area:.9f} vs mesh limit {limit:.9f} mm2 "
        f"({rel:.2e} relative) — two independent computations of the same area have "
        f"drifted apart, which is a geometry bug in one of them, not a mesh resolution "
        f"problem")


# ---------------------------------------------------------------------------
# VALIDITY
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["smoke", "coarse", "medium"])
def test_shipped_design_meshes_cleanly(vec, name):
    cfg = M.get_config(name)
    X = M.flatten(M.spoke_block_coords_from_vector(vec, cfg, SPAN))
    q = M.quality_report(X, M.spoke_block_connectivity(cfg))
    assert q["n_inverted"] == 0
    assert q["min_scaled_jacobian"] > 0.9


def test_scaled_jacobian_is_one_for_a_rectangle():
    """Calibration of the metric itself, including that it is insensitive to aspect
    ratio — a long thin rectangle is a valid element and must score 1.0."""
    conn = np.array([[0, 1, 2, 3]])
    for w in (1.0, 0.01):
        rect = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, w], [0.0, w]])
        assert abs(M.scaled_jacobian(rect, conn)[0] - 1.0) < 1e-12


def test_scaled_jacobian_is_negative_for_an_inverted_element():
    """The property the whole gate rests on: a folded element must score < 0, not
    merely small."""
    conn = np.array([[0, 1, 2, 3]])
    bowtie = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert M.scaled_jacobian(bowtie, conn)[0] < 0


def test_fold_margin_predicts_inversion():
    """The closed-form margin must flag every genome whose mesh inverts.

    This is what lets the optimizer avoid the folded region with one curvature
    evaluation instead of a mesh in the loop.  Measured over the full design space
    (study_mesh_quality.py) the threshold in `MIN_FOLD_MARGIN_MM` gives zero misses;
    here we check a smaller sample keeps that property.
    """
    cfg = M.get_config("coarse")
    conn = M.spoke_block_connectivity(cfg)
    low, high, _ = GN.bounds_arrays(W.GENE_SPACE)
    rng = np.random.default_rng(7)

    checked = 0
    for v in rng.uniform(low, high, size=(1500, 14)):
        _, loss = W.evaluate_design(v)
        if loss["x_order"] != 0.0 or loss["hub_overlap"] != 0.0:
            continue
        curve, ctrl = G.bezier_centerline(*v[:8], span_mm=SPAN,
                                          num_points=cfg.n_curve)
        margin = float(G.self_intersection_margin(curve, ctrl, *v[8:12],
                                                  num_points=cfg.n_curve))
        if margin <= G.MIN_FOLD_MARGIN_MM:
            continue                                     # barrier would reject it
        checked += 1
        X = M.flatten(M.spoke_block_coords_from_vector(v, cfg, SPAN))
        q = M.quality_report(X, conn)
        assert q["n_inverted"] == 0, f"margin {margin:.4f} but mesh inverted"
        assert q["min_scaled_jacobian"] > 0.2, (
            f"margin {margin:.4f} passed the barrier but minSJ is "
            f"{q['min_scaled_jacobian']:.4f}")
    assert checked > 20, f"only {checked} genomes survived the filter — weak test"


# ---------------------------------------------------------------------------
# DIFFERENTIABILITY
# ---------------------------------------------------------------------------

def _mesh_scalar(v12):
    cfg = M.get_config("smoke")
    X = M.spoke_block_coords(*[v12[i] for i in range(12)],
                             cfg=cfg, span_mm=SPAN, xp=jnp)
    return jnp.sum(X ** 2)


def test_mesh_coords_gradient_matches_finite_differences(vec):
    """Reverse-mode AD vs central differences, with a STEP-SIZE SWEEP.

    A single step size does not test what it looks like it tests.  Central differences
    trade truncation error (falls as h^2) against roundoff (rises as 1/h), so every
    gene has its own best h and a fixed choice measures the worse of the two effects
    rather than the gradient.  Here the objective is ~1.4e5, so at h=2e-6 the roundoff
    floor alone is eps*f/(2h) ~ 8e-6 — bigger than the 1e-6 one might naively assert.

    So: sweep h over five decades and take each gene's best agreement.  A gene with no
    good h anywhere is a gene whose objective is genuinely not smooth in it, which is
    the thing actually worth catching.
    """
    v = jnp.asarray(np.asarray(vec)[:12])
    g_ad = np.asarray(jax.grad(_mesh_scalar)(v))
    base = np.asarray(v, dtype=float)

    best = np.full_like(base, np.inf)
    for scale in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        for i in range(len(base)):
            h = scale * max(1.0, abs(base[i]))
            up, dn = base.copy(), base.copy()
            up[i] += h
            dn[i] -= h
            fd = (float(_mesh_scalar(jnp.asarray(up)))
                  - float(_mesh_scalar(jnp.asarray(dn)))) / (2 * h)
            rel = abs(g_ad[i] - fd) / max(abs(fd), 1.0)
            best[i] = min(best[i], rel)

    worst = int(np.argmax(best))
    assert best.max() < 1e-8, (
        f"worst gene {W.GENE_NAMES[worst]}: no step size gave agreement "
        f"(best relative error {best.max():.2e})")


def test_mesh_is_jittable_and_backends_agree(vec):
    cfg = M.get_config("smoke")
    Xn = M.spoke_block_coords_from_vector(np.asarray(vec), cfg, SPAN, xp=np)
    f = jax.jit(lambda v: M.spoke_block_coords(*[v[i] for i in range(12)],
                                               cfg=cfg, span_mm=SPAN, xp=jnp))
    Xj = np.asarray(f(jnp.asarray(np.asarray(vec)[:12])))
    assert np.abs(Xn - Xj).max() < 1e-12
