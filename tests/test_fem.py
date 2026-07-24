"""
M3 verification of the plane-stress FE kernel.

Every tolerance here was written down before the test was run, and each one is a
number the plan committed to.  Where a test passes for a *reason other than the one
intended* it says so — the finite-rotation test in particular is only meaningful
because the same check is asserted to FAIL for the linear kernel.

The slenderness sweep (A4) is the gate and lives in `study_beam_agreement.py`, which
produces a report rather than a boolean; `test_a4_exponent_gate` re-runs a reduced
version of it here so CI cannot drift away from the recorded result.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wheel_fea as wf            # noqa: E402
import wheel_fem as fem           # noqa: E402
import wheel_genome as wg         # noqa: E402
import wheel_mesh as wm           # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


def straight_genes(thickness, span=wf.HUB_RIM_SPAN_MM):
    """A straight, uniform-thickness beam expressed in the SAME 14 genes.

    Deliberately not a hand-built rectangular mesh: routing the analytical beam checks
    through the production geometry kernel and the production mesh generator means A1
    and A2 exercise the code that actually runs, including the arc-length resampling
    and the analytic normals.  A separate rectangular-grid path would test neither.
    """
    f = np.array([0.2, 0.4, 0.6, 0.8]) * span
    v = np.zeros(14)
    v[0:8:2] = f
    v[8:12] = thickness
    v[12:14] = 0.5
    return v


# ---------------------------------------------------------------------------
# ELEMENT-LEVEL: the tests that must pass before any beam comparison means anything
# ---------------------------------------------------------------------------

def test_node_table_matches_the_mesh_connectivity():
    """The FE node ordering and the mesh's vertex ordering must be the same permutation.

    A mismatch here yields an element that is still symmetric, still positive definite,
    and still passes a rigid-body test — it just integrates a scrambled geometry.  The
    check rebuilds the expected local grid offsets from `wheel_mesh`'s own index
    arithmetic rather than from a transcribed copy of them.
    """
    for order in (1, 2):
        cfg = wm.MeshConfig("one", 1, 1, order=order)
        conn = wm.spoke_block_connectivity(cfg)[0]
        nt = cfg.n_node_thick
        offsets = np.array([(int(n) // nt, int(n) % nt) for n in conn])
        assert np.array_equal(offsets, fem._NODE_IJ[order]), (
            f"order {order}: mesh gives {offsets.tolist()}, "
            f"wheel_fem._NODE_IJ has {fem._NODE_IJ[order].tolist()}"
        )


def _distorted_patch(order=2, n=4, seed=0):
    """A small block with its INTERIOR nodes randomly displaced.

    A patch test on a rectangular grid is nearly vacuous: the Jacobian is diagonal and
    constant, so a transposed inverse-Jacobian or a mis-scaled reference gradient
    cancels out.  Distortion is what makes the test able to fail.
    """
    cfg = wm.MeshConfig("patch", n, n, order=order)
    coords = wm.flatten(np.asarray(
        wm.spoke_block_coords_from_vector(straight_genes(4.0), cfg,
                                          span_mm=wf.HUB_RIM_SPAN_MM, xp=np)))
    conn = wm.spoke_block_connectivity(cfg)
    bnd = wm.boundary_nodes(cfg)
    boundary = np.unique(np.concatenate(list(bnd.values())))
    interior = np.setdiff1d(np.arange(coords.shape[0]), boundary)

    hx = wf.HUB_RIM_SPAN_MM / (cfg.n_node_span - 1)
    hy = 4.0 / (cfg.n_node_thick - 1)
    rng = np.random.default_rng(seed)
    coords[interior] += rng.uniform(-0.2, 0.2, (interior.size, 2)) * [hx, hy]
    assert wm.scaled_jacobian(coords, conn).min() > 0.2, "distortion inverted an element"
    return coords, conn, boundary, interior, cfg


@pytest.mark.parametrize("order", [1, 2])
def test_patch_test_on_a_distorted_mesh(order):
    """Prescribe u = A x + b everywhere on the boundary; recover it exactly inside.

    Tolerance 1e-12 relative, per the plan.  Anything above 1e-10 is a bug and not
    roundoff: the linear field is in the element's polynomial space, so the discrete
    solution is the exact one up to conditioning.
    """
    coords, conn, boundary, interior, cfg = _distorted_patch(order=order)
    A = np.array([[7e-4, -3e-4], [2e-4, 5e-4]])
    b = np.array([1e-3, -2e-3])
    exact = coords @ A.T + b

    lam, mu = fem.lame(wf.YOUNGS_MODULUS_PLA_MPA, fem.POISSON_RATIO_PLA)
    dm = fem.DofMap(coords.shape[0])
    dm.fix(boundary, exact[boundary])
    dm.free(interior)
    prob = fem.Problem(coords, conn, cfg.order, lam, mu, wf.SPOKE_WIDTH_MM, dm)
    u = fem.solve_linear(prob)["u"].reshape(-1, 2)

    err = np.abs(u[interior] - exact[interior]).max() / np.abs(exact).max()
    assert err < 1e-12, f"patch test order {order}: relative error {err:.3e}"

    # The stress must be constant to the same tolerance — a displacement field can be
    # right at the nodes while the recovered gradient is not.
    st = fem.gauss_stresses(coords, conn, u.ravel(), order=cfg.order, lam=lam, mu=mu)
    s = st["sigma"].reshape(-1, 4)
    spread = np.abs(s - s.mean(axis=0)).max() / np.abs(s).max()
    assert spread < 1e-12, f"stress not constant: relative spread {spread:.3e}"


def test_zero_energy_modes():
    """K on a free-floating mesh has exactly 3 near-zero eigenvalues, and no more.

    A 4th near-zero eigenvalue is an hourglass mode — the classic signature of
    under-integration.  The threshold is set relative to the 4th eigenvalue rather
    than to an absolute number, because the absolute scale is E*h and therefore
    depends on the mesh.
    """
    cfg = wm.CONFIGS["smoke"]
    coords, conn, _ = fem.spoke_coords(straight_genes(2.0), cfg)
    lam, mu = fem.lame(wf.YOUNGS_MODULUS_PLA_MPA, fem.POISSON_RATIO_PLA)
    K = fem.assemble_stiffness(coords, conn, order=cfg.order, lam=lam, mu=mu,
                               width=wf.SPOKE_WIDTH_MM).toarray()
    assert np.abs(K - K.T).max() / np.abs(K).max() < 1e-12, "K is not symmetric"
    ev = np.linalg.eigvalsh(K)
    assert (ev[:3] / ev[3] < 1e-10).all(), f"expected 3 rigid modes, got {ev[:5]}"
    assert ev[3] / ev[3:].max() > 1e-9, f"4th mode is spurious: {ev[:6]}"


@pytest.mark.parametrize("angle_deg", [1.0, 30.0])
def test_finite_rotation_stores_no_energy_under_svk(angle_deg):
    """A rigid rotation of the whole mesh stores exactly zero strain energy under SVK.

    Green-Lagrange gives E = (R^T R - I)/2 = 0 for any rotation, so this is exact and
    not asymptotic — the tolerance is 1e-10 of the energy the LINEAR kernel spuriously
    stores at the same rotation, which is the only scale that makes the ratio
    meaningful.
    """
    cfg = wm.CONFIGS["smoke"]
    coords, conn, _ = fem.spoke_coords(straight_genes(2.0), cfg)
    lam, mu = fem.lame(wf.YOUNGS_MODULUS_PLA_MPA, fem.POISSON_RATIO_PLA)
    th = np.radians(angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    u = (coords @ R.T - coords).ravel()

    kw = dict(order=cfg.order, lam=lam, mu=mu, width=wf.SPOKE_WIDTH_MM)
    e_svk = fem.total_energy(coords, conn, u, nonlinear=True, **kw)
    e_lin = fem.total_energy(coords, conn, u, nonlinear=False, **kw)

    assert abs(e_svk) / e_lin < 1e-10, (
        f"SVK stored {e_svk:.3e} at {angle_deg} deg (linear stores {e_lin:.3e})")

    # And the linear kernel MUST fail the same check, or this is not testing frame
    # indifference — it is testing that the rotation was small.  Asserting a magic
    # threshold on e_lin would only pin the mesh size, so assert the closed form
    # instead: under u = (R - I)x the linear strain is (cos th - 1) I, giving
    # W = 2(lam + mu)(cos th - 1)^2 per unit volume, uniform over the body.
    volume = wf.HUB_RIM_SPAN_MM * 2.0 * wf.SPOKE_WIDTH_MM
    expected = 2.0 * (lam + mu) * (np.cos(th) - 1.0) ** 2 * volume
    assert abs(e_lin - expected) / expected < 1e-3, (
        f"linear spurious energy {e_lin:.6e} != closed form {expected:.6e}")


def test_svk_and_linear_agree_in_the_small_strain_limit(genes):
    """The two kinematics must coincide as the load goes to zero.

    This is the check that the SVK path is the same material and not a different one:
    at 1% of service load the geometric terms are O(1e-4) and the two energies must
    agree to well under 0.1%.
    """
    kw = dict(cfg="coarse", force=wf.FORCE_PER_SPOKE_NEWTONS * 0.01)
    d_lin = fem.spoke_deflection(genes, kinematics="linear", **kw)
    d_svk = abs(fem.solve_linear(
        fem.spoke_problem(genes, kinematics="svk", **kw))["deflection_mm"])
    # Note: solve_linear on the SVK kernel is one Newton step from u=0, which is the
    # correct comparison here — it isolates the tangent, not the equilibrium path.
    assert abs(d_svk - d_lin) / d_lin < 1e-3


def test_equilibrium_residual_is_at_solver_precision(genes):
    res = fem.solve_linear(fem.spoke_problem(genes, "coarse"))
    assert res["residual_rel"] < 1e-9, res["residual_rel"]


def test_unconstrained_dof_is_an_error_not_a_singular_matrix():
    """The DofMap must refuse to build T rather than hand a singular system to spsolve.

    A dropped constraint produces a matrix that `spsolve` "solves" with a warning and
    garbage, which is far harder to notice than an exception.
    """
    dm = fem.DofMap(4)
    dm.free([0, 1])
    with pytest.raises(ValueError, match="never constrained"):
        dm.finalize()
    dm2 = fem.DofMap(2)
    dm2.free([0])
    with pytest.raises(ValueError, match="constrained twice"):
        dm2.fix([0])


# ---------------------------------------------------------------------------
# A1 / A2 — straight beam against closed form
# ---------------------------------------------------------------------------

def _straight_reference(thickness, bc):
    L = wf.HUB_RIM_SPAN_MM
    I = wf.SPOKE_WIDTH_MM * thickness ** 3 / 12.0
    EI = wf.YOUNGS_MODULUS_PLA_MPA * I
    F = wf.FORCE_PER_SPOKE_NEWTONS
    denom = 3.0 if bc == "cantilever" else 12.0
    return F * L ** 3 / (denom * EI)


@pytest.mark.parametrize("bc,denom", [("cantilever", 3.0), ("fixed_guided", 12.0)])
def test_a1_a2_straight_beam_against_closed_form(bc, denom):
    """A1/A2: transversely loaded straight beam at L/t = 50, < 1.0%.

    L/t = 50 puts the shear correction at ~0.03% (0.81 t^2/L^2 for a rectangular
    section at nu = 0.35), so at this slenderness the closed form is essentially
    exact and 1% is a loose bound on the FE error alone.  The 4x ratio between the
    two boundary conditions is the repo's own documented regression
    (`wheel_fea.py:326-331`).
    """
    t = wf.HUB_RIM_SPAN_MM / 50.0
    ref = _straight_reference(t, bc)
    got = fem.spoke_deflection(straight_genes(t), "medium", bc=bc,
                               load_dir=(0.0, -1.0))
    assert abs(got - ref) / ref < 0.01, f"{bc}: FE {got:.5f} vs {ref:.5f} mm"


def test_a1_a2_ratio_is_four():
    """The cantilever/fixed-guided stiffness ratio must be 4, computed not assumed."""
    t = wf.HUB_RIM_SPAN_MM / 50.0
    g = straight_genes(t)
    c = fem.spoke_deflection(g, "medium", bc="cantilever", load_dir=(0.0, -1.0))
    f = fem.spoke_deflection(g, "medium", bc="fixed_guided", load_dir=(0.0, -1.0))
    assert abs(c / f - 4.0) < 0.02, f"ratio {c / f:.4f}"


def test_shear_deformation_is_resolved_not_locked():
    """The FE must be SOFTER than Euler-Bernoulli, by about the Timoshenko amount.

    This is the Q4-vs-Q9 discriminator stated positively.  A locking element is
    *stiffer* than the closed form; a correct one is softer by
    dv/v = 0.81 (t/L)^2 for a rectangular section at nu = 0.35.  At L/t = 8 that is
    1.27%, which is far outside the discretization error at `medium`.
    """
    L = wf.HUB_RIM_SPAN_MM
    t = L / 8.0
    ref = _straight_reference(t, "cantilever")
    got = fem.spoke_deflection(straight_genes(t), "medium", bc="cantilever",
                               load_dir=(0.0, -1.0))
    excess = got / ref - 1.0
    predicted = 0.81 * (t / L) ** 2
    assert excess > 0, f"FE is stiffer than Euler-Bernoulli by {-excess:.4%} — locking"
    assert abs(excess - predicted) / predicted < 0.35, (
        f"shear excess {excess:.4%} vs Timoshenko {predicted:.4%}")


# ---------------------------------------------------------------------------
# A3 — the real curved spoke against Castigliano
# ---------------------------------------------------------------------------

def _castigliano(genes, bc, thickness_scale=1.0, force=None, clip=(0.5, 20.0)):
    curve, _ = wf.generate_bezier_centerline(*[genes[i] for i in range(8)])
    t = [genes[i] * thickness_scale for i in range(8, 12)]
    d, *_ = wf.generalized_spoke_mechanics(
        curve, *t, wf.SPOKE_WIDTH_MM,
        wf.FORCE_PER_SPOKE_NEWTONS if force is None else force,
        genes[12], genes[13], boundary_condition=bc, thickness_clip=clip)
    return d


@pytest.mark.parametrize("bc", ["fixed_guided", "cantilever"])
def test_a3_curved_spoke_against_castigliano(bc, genes):
    """A3: the on-disk genome at 1% of service load, < 5%, and the FE must be softer.

    1% of service load keeps geometric nonlinearity below 0.01%, so the comparison is
    between two LINEAR models and the only differences left are the ones being
    measured: transverse shear, the finite root, and the fact that the beam model
    integrates a 1D centerline while the FE integrates the actual section.
    """
    ref = _castigliano(genes, bc, force=wf.FORCE_PER_SPOKE_NEWTONS * 0.01)
    got = fem.spoke_deflection(genes, "medium", bc=bc,
                               force=wf.FORCE_PER_SPOKE_NEWTONS * 0.01)
    rel = got / ref - 1.0
    assert abs(rel) < 0.05, f"{bc}: FE {got:.6f} vs beam {ref:.6f} ({rel:+.2%})"
    assert rel > 0, f"{bc}: FE is STIFFER than the beam model by {-rel:.2%}"


def test_deflection_converges_under_refinement(genes):
    """Successive refinements must shrink the change, not merely move the answer."""
    d = [fem.spoke_deflection(genes, c, force=wf.FORCE_PER_SPOKE_NEWTONS * 0.01)
         for c in ("smoke", "coarse", "medium")]
    step1, step2 = abs(d[1] - d[0]), abs(d[2] - d[1])
    assert step2 < 0.35 * step1, f"not converging: steps {step1:.3e}, {step2:.3e}"
    assert step2 / d[2] < 1e-3, f"medium is not converged: {step2 / d[2]:.2e}"


# ---------------------------------------------------------------------------
# A4 — THE GATE
# ---------------------------------------------------------------------------

def a4_sweep(genes, bc="fixed_guided", root_bc="clamped", cfg="medium",
             lambdas=(1.0, 0.5, 0.25, 0.125)):
    """FE-vs-Castigliano discrepancy as the section is thinned.  Shared with the study.

    The load is scaled by lambda^3 alongside the thickness so the deflection — and
    hence the strain — stays fixed across the sweep.  Otherwise a thinner section at
    constant force deflects 512x further and the comparison would be measuring
    geometric nonlinearity creeping into a linear FE model instead of the O(t^2)
    beam-theory error.
    """
    out = []
    for lam in lambdas:
        force = wf.FORCE_PER_SPOKE_NEWTONS * 0.01 * lam ** 3
        ref = _castigliano(genes, bc, thickness_scale=lam, force=force,
                           clip=(0.01, 20.0))
        got = fem.spoke_deflection(genes * np.concatenate([np.ones(8),
                                                           np.full(4, lam),
                                                           np.ones(2)]),
                                   cfg, bc=bc, root_bc=root_bc, force=force)
        out.append({"lambda": lam, "beam_mm": ref, "fe_mm": got,
                    "rel_error": got / ref - 1.0})
    return out


def fit_exponent(rows):
    """Slope of log|rel_error| against log(lambda).  2 means the error is O(t^2)."""
    x = np.log(np.array([r["lambda"] for r in rows]))
    y = np.log(np.abs([r["rel_error"] for r in rows]))
    return float(np.polyfit(x, y, 1)[0])


@pytest.mark.parametrize("root_bc", ["clamped", "plane"])
def test_a4_exponent_gate(root_bc, genes):
    """THE M3 GATE: the beam-model discrepancy must decay as O(t^2).

    Single-point agreement at 5% proves nothing — it is one number that could come
    from anything.  This demonstrates convergence to the analytical model in the
    limit where the analytical model is exact, which is the only evidence that the
    element is right rather than coincidentally close.

    Exponent ~0 means a bug.  Exponent ~1 means shear locking (or, here, a root
    condition contaminating the comparison at first order — which is why both root
    treatments are checked).  Required: exponent in [1.7, 2.3] and < 0.5% at
    lambda = 1/8.
    """
    rows = a4_sweep(genes, root_bc=root_bc)
    p = fit_exponent(rows)
    finest = abs(rows[-1]["rel_error"])
    detail = "  ".join(f"l={r['lambda']:.3f}: {r['rel_error']:+.4%}" for r in rows)
    assert 1.7 <= p <= 2.3, f"root_bc={root_bc}: exponent {p:.3f}\n  {detail}"
    assert finest < 0.005, f"root_bc={root_bc}: {finest:.4%} at lambda=1/8\n  {detail}"
