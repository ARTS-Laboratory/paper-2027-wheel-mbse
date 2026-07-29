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

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

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


@pytest.mark.parametrize("order", [1, 2])
def test_traction_patch_test_on_a_distorted_mesh(order):
    """The Neumann half of the patch test, to the same 1e-12.

    A constant stress state sigma is applied as the traction sigma @ n on both flanks
    while the two ends are held at the exact linear displacement.  Recovering the field
    requires the consistent nodal loads to be right, not merely to sum to the right
    total: lumping a quadratic edge equally onto its 3 nodes gives the same resultant
    and fails here.  It is also the only test that exercises `edge_traction_load`,
    which M4's distributed contact pressure and M6's penalty contact both go through.
    """
    coords, conn, _, _, cfg = _distorted_patch(order=order)
    lam, mu = fem.lame(wf.YOUNGS_MODULUS_PLA_MPA, fem.POISSON_RATIO_PLA)

    # Pick the displacement field first, then derive the stress it implies, so the two
    # cannot disagree.
    A = np.array([[7e-4, -3e-4], [-3e-4, 5e-4]])          # symmetric => A is the strain
    exact = coords @ A.T
    sigma = lam * np.trace(A) * np.eye(2) + 2.0 * mu * A

    bnd = wm.boundary_nodes(cfg)
    held = np.unique(np.concatenate([bnd["root"], bnd["tip"]]))
    free = np.setdiff1d(np.arange(coords.shape[0]), held)

    f = np.zeros(2 * coords.shape[0])
    for side in ("flank_bot", "flank_top"):
        f += fem.edge_traction_load(coords, conn, cfg, side,
                                    lambda x, n: sigma @ n, width=wf.SPOKE_WIDTH_MM)

    dm = fem.DofMap(coords.shape[0])
    dm.fix(held, exact[held])
    dm.free(free)
    prob = fem.Problem(coords, conn, cfg.order, lam, mu, wf.SPOKE_WIDTH_MM, dm,
                       f_nodal=f)
    u = fem.solve_linear(prob)["u"].reshape(-1, 2)

    err = np.abs(u[free] - exact[free]).max() / np.abs(exact).max()
    assert err < 1e-12, f"traction patch test order {order}: relative error {err:.3e}"


def test_traction_resultant_equals_the_analytic_force():
    """Sanity on `edge_traction_load` itself: a uniform pressure integrates to p*L*w.

    Weaker than the patch test — any lumping scheme passes it — but it localises a
    wrong out-of-plane width factor or a missing edge Jacobian to this function instead
    of leaving it as an unexplained patch-test failure.
    """
    cfg = wm.MeshConfig("t", 12, 3, order=2)
    coords, conn, _ = fem.spoke_coords(straight_genes(3.0), cfg)
    f = fem.edge_traction_load(coords, conn, cfg, "flank_top", (0.0, -1.0),
                               width=wf.SPOKE_WIDTH_MM)
    # Straight beam, flat top flank of length = span: resultant = 1 MPa * L * w.
    expected = wf.HUB_RIM_SPAN_MM * wf.SPOKE_WIDTH_MM
    got = -f.reshape(-1, 2)[:, 1].sum()
    assert abs(got / expected - 1.0) < 1e-12, f"{got:.6f} vs {expected:.6f}"


def test_zero_energy_modes():
    """K on a free-floating mesh has exactly 3 near-zero eigenvalues, and no more.

    A 4th near-zero eigenvalue is an hourglass mode — the classic signature of
    under-integration.

    The tolerance is DERIVED, not picked.  A rigid mode's eigenvalue is zero in exact
    arithmetic, so numerically it is bounded below by roundoff in assembling K, i.e.
    ~eps * lambda_max.  Measured relative to lambda_4 that floor is
    `eps * lambda_max / lambda_4` = 1.8e-10 for this mesh, and the observed rigid modes
    sit at 1.1e-10 — BELOW the floor, so any round-number bound tighter than it (an
    earlier version of this test used 1e-10) fails for reasons that have nothing to do
    with the element.

    The real content of "exactly 3" is the SEPARATION between the 3rd and 4th
    eigenvalues, which is 9 orders of magnitude and cannot be produced by roundoff.
    """
    cfg = wm.CONFIGS["smoke"]
    coords, conn, _ = fem.spoke_coords(straight_genes(2.0), cfg)
    lam, mu = fem.lame(wf.YOUNGS_MODULUS_PLA_MPA, fem.POISSON_RATIO_PLA)
    K = fem.assemble_stiffness(coords, conn, order=cfg.order, lam=lam, mu=mu,
                               width=wf.SPOKE_WIDTH_MM).toarray()
    assert np.abs(K - K.T).max() / np.abs(K).max() < 1e-12, "K is not symmetric"
    ev = np.linalg.eigvalsh(K)
    floor = np.finfo(float).eps * ev[-1] / ev[3]
    assert (ev[:3] / ev[3] < 10.0 * floor).all(), (
        f"rigid modes {ev[:3] / ev[3]} exceed 10x the roundoff floor {floor:.2e}")
    assert ev[3] / ev[2] > 1e6, (
        f"no clear gap after the 3rd mode — the 4th is spurious: {ev[:6]}")


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

# ---------------------------------------------------------------------------
# A1 / A2 / A3 / A4 — beam agreement
#
# The sweeps live in `study_beam_agreement.py`, which is the deliverable and prints
# the recorded table.  The tests below call into it rather than re-deriving it, so
# there is exactly one definition of each check and CI cannot drift away from the
# published numbers.  They run at `--quick` fidelity (h = t/8, three slenderness
# levels) to stay inside a test suite's time budget; the gate values in
# `study_beam_agreement.json` come from the full run.
# ---------------------------------------------------------------------------

import study_beam_agreement as sba   # noqa: E402


@pytest.fixture(scope="module")
def quick():
    """Run the study's sweeps at h = t/8 instead of t/16, to fit a test budget."""
    saved = sba.MESH_H_OVER_T
    sba.MESH_H_OVER_T = 8
    yield
    sba.MESH_H_OVER_T = saved


def test_a1_a2_straight_beam_against_closed_form(quick):
    """A1/A2: transversely loaded straight beam at L/t = 50, < 1.0% each.

    At L/t = 50 the shear correction is 0.81 (t/L)^2 = 0.032%, so the closed form is
    essentially exact and 1% is a loose bound on the FE error alone.  The 4x ratio
    between the two boundary conditions is the repo's own documented regression
    (`wheel_fea.py:326-331`), computed here rather than assumed.
    """
    r = sba.run_a1_a2()
    for case in r["cases"]:
        assert abs(case["rel_error"]) < 0.01, (
            f"{case['bc']}: FE {case['fe_mm']:.5f} vs closed form "
            f"{case['closed_form_mm']:.5f} ({case['rel_error']:+.4%})")
    assert abs(r["ratio_error"]) < 0.02, f"stiffness ratio {r['stiffness_ratio']:.5f}"


def test_a3_curved_spoke_against_castigliano(quick, genes):
    """A3: the on-disk genome at 1% of service load, < 5%, and the FE must be softer.

    1% of service load keeps geometric nonlinearity below 0.01%, so this compares two
    LINEAR models and the only differences left are the ones being measured:
    transverse shear, the curved-beam neutral-axis shift, and the fact that
    Castigliano integrates a 1D centerline while the FE integrates the real section.
    """
    r = sba.run_a3(genes)
    for case in r["cases"]:
        assert abs(case["rel_error"]) < 0.05, (
            f"{case['bc']}: {case['rel_error']:+.4%}")
        assert case["rel_error"] > 0, (
            f"{case['bc']}: FE is STIFFER than the beam model by "
            f"{-case['rel_error']:.4%}")


def test_a4a_element_gate(quick):
    """THE M3 GATE.  Straight cantilever, exact closed form, four slendernesses.

    Single-point agreement proves nothing.  This requires the discrepancy to decay at
    second order TO A KNOWN COEFFICIENT: excess/(t/L)^2 -> 0.81 (Timoshenko, k = 5/6,
    nu = 0.35).  Shear locking would show as the wrong sign, an exponent near 1, or a
    coefficient near zero — all three are checked.

    The reference here has no discretization of its own, which is why the element gate
    is on the straight beam and not on the curved spoke.  See
    `study_beam_agreement.py`'s docstring for why the plan's original formulation
    could not settle the question.
    """
    r = sba.run_a4a(slendernesses=(8, 16, 32))
    p = r["plane"]
    detail = "  ".join(f"L/t={row['L_over_t']:.0f}: "
                       f"{row['excess_over_t_L_sq']:.4f}" for row in p["rows"])
    assert p["all_positive"], f"FE stiffer than Euler-Bernoulli somewhere\n  {detail}"
    assert 1.7 <= p["fitted_exponent"] <= 2.3, (
        f"exponent {p['fitted_exponent']:.3f}\n  {detail}")
    assert abs(p["coefficient_error_vs_timoshenko"]) < 0.20, (
        f"coefficient {p['mean_coefficient']:.4f} vs Timoshenko "
        f"{sba.TIMOSHENKO_COEFF}\n  {detail}")


def test_a4a_clamped_root_is_a_first_order_contaminant(quick):
    """The clamped root MUST fail the same check, or `root_bc` is not doing anything.

    A fully clamped root forbids the lateral Poisson strain the bending field genuinely
    has there.  Its energy is O(t/L) relative to the beam's — first order — so at high
    slenderness it does not just contaminate the measurement, it reverses the sign and
    the FE comes out stiffer than Euler-Bernoulli.  This test exists so that nobody
    "simplifies" `root_bc` away and then reads the result as shear locking.
    """
    r = sba.run_a4a(slendernesses=(8, 16, 32), root_bcs=("clamped",))
    coeffs = [row["excess_over_t_L_sq"] for row in r["clamped"]["rows"]]
    assert min(coeffs) < 0.5 * sba.TIMOSHENKO_COEFF, (
        f"clamped root behaved like the beam-consistent one: {coeffs} — has the "
        "root BC stopped mattering?")


def test_a4b_curved_spoke_sweep(quick, genes):
    """A4b: the same sweep against Castigliano.  Positive, monotone, never O(t).

    Deliberately NOT gated on an upper bound: the Castigliano error is a sum of
    several O(t^2) effects that partially cancel for this geometry, so the residual
    decays faster than t^2 (measured exponent 2.70) and nothing broken produces that.
    What IS gated is that every discrepancy has the FE softer and every local exponent
    is at least 1.7, which is what rules out locking.
    """
    r = sba.run_a4b(genes, lambdas=(1.0, 0.5, 0.25))
    detail = "  ".join(f"l={row['lambda']:.3f}: {row['rel_error']:+.4%}"
                       for row in r["rows"])
    assert r["all_positive"], f"FE stiffer than Castigliano somewhere\n  {detail}"
    assert r["monotone"], f"discrepancy not decreasing\n  {detail}"
    for e in r["local_exponents"]:
        assert e >= 1.7, f"local exponent {e:.3f} — first-order error\n  {detail}"


def test_deflection_converges_under_refinement(genes):
    """Successive refinements must shrink the change, not merely move the answer."""
    d = [fem.spoke_deflection(genes, c, root_bc="plane",
                              force=wf.FORCE_PER_SPOKE_NEWTONS * 0.01)
         for c in ("smoke", "coarse", "medium")]
    step1, step2 = abs(d[1] - d[0]), abs(d[2] - d[1])
    assert step2 < 0.35 * step1, f"not converging: steps {step1:.3e}, {step2:.3e}"


def test_mesh_resolution_must_scale_with_thickness():
    """The span element size has to resolve the ~t boundary layers, not the part.

    This is the most consequential number M3 produced for M4, so it is pinned: on the
    thinnest section in the A4 sweep, a mesh with h ~ t gets the SIGN of the
    beam-model discrepancy wrong, while h = t/8 does not.  A future mesh-config table
    that sizes elements by the part rather than by the wall will fail here.
    """
    g = sba.scale_thickness(sba.load_genes(os.path.join(REPO, "best_solution.json")),
                            0.125)
    F = wf.FORCE_PER_SPOKE_NEWTONS * sba.PROBE_FORCE_FRACTION * 0.125 ** 3
    ref = sba.castigliano(g, "fixed_guided", F)
    err = {}
    for k in (1, 8):
        cfg = sba.sized_config(g, k=k)
        err[k] = fem.spoke_deflection(g, cfg, root_bc="plane", force=F) / ref - 1.0
    assert err[1] < 0 < err[8], (
        f"expected h~t to read stiff and h=t/8 to read soft, got "
        f"h~t: {err[1]:+.4%}, h=t/8: {err[8]:+.4%}")
