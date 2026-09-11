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
    (`wheel_fea.py:403-408`), computed here rather than assumed.
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
    thinnest section in the A4 sweep, a mesh sized by the PART reads the sign of the
    beam-model discrepancy wrong, while one sized by the WALL does not.  A future
    mesh-config table that sizes elements by the part will fail here.

    THE PROBE IS AIMED FROM A MEASURED CROSSOVER, AND IT DID NOT USED TO BE (§141).
    The discrepancy is monotone in `h`: coarse reads stiff (negative), refined reads
    soft (positive), and it crosses zero at one `h/t`.  That crossover is a function of
    the SHAPE genes, so a probe pinned near it stops demonstrating anything the moment
    the shipped genome moves.  This test used to probe `h/t = 1`, and bisection puts the
    crossover at:

        genome                       h*/t      err at the h/t = 1 probe
        pre-`cb4e3dd`              0.9471            -0.0011%     (probe 5.6% past it)
        shipped (`b729e86`)        1.7978            +0.0087%     (probe 44% short)

    So `h/t = 1` sat five percent from the crossover and the promotion moved the
    crossover 90% coarser, taking the probe with it -- the demonstration was intact the
    whole time and the vehicle had fallen out of the branch.  The probe is now
    `h/t ~ 4`, which is 2.2x past the crossover, and the measured margin there is 113x
    the one it replaces:

        k       h/t      shipped        pre-`cb4e3dd`
        0.25    3.967    -0.125969%     -0.183304%     <- part-sized, reads STIFF
        0.5     1.994    -0.003732%     -0.029813%
        1       1.000    +0.008710%     -0.001109%     <- the old probe, on the fence
        8       0.125    +0.011332%     +0.011553%     <- wall-resolved, reads SOFT

    Measured 2026-09-08 at `lam = 0.125` (`t_min` 0.15 mm).  Both probes hold their sign
    across `lam` in 0.0625..0.5 on both genomes, EXCEPT the wall probe at `lam = 0.0625`
    on the pre-`cb4e3dd` genome (-0.000595%): at `t = 0.075` mm the converged discrepancy
    is itself ~1e-5 and the sign is not resolvable, which is why `lam` stays at 0.125.
    Over 30 genomes drawn uniformly from `GENE_SPACE` the two signs hold 29/30 each, so
    this is generic to the geometry and not a property of the shipped design -- but it is
    not universal, and this test is scoped to the shipped genome.
    """
    g = sba.scale_thickness(sba.load_genes(os.path.join(REPO, "best_solution.json")),
                            0.125)
    F = wf.FORCE_PER_SPOKE_NEWTONS * sba.PROBE_FORCE_FRACTION * 0.125 ** 3
    ref = sba.castigliano(g, "fixed_guided", F)
    err, span = {}, {}
    for k in (0.25, 8):
        cfg = sba.sized_config(g, k=k)
        span[k] = cfg.n_span
        err[k] = fem.spoke_deflection(g, cfg, root_bc="plane", force=F) / ref - 1.0
    assert err[0.25] < 0 < err[8], (
        f"expected a part-sized mesh (h/t~4) to read stiff and a wall-resolved one "
        f"(h=t/8) to read soft, got h/t~4: {err[0.25]:+.4%}, h=t/8: {err[8]:+.4%}")
    # AND THE COARSE READING MUST NOT DRIFT BACK ONTO THE FENCE.  This is the guard the
    # old probe lacked: a sign that holds by 1e-5 is a coin, not a demonstration.
    # Measured 11.1x (shipped) and 15.9x (pre-`cb4e3dd`); 5.0 leaves 2.2x of headroom.
    assert abs(err[0.25]) > 5.0 * abs(err[8]), (
        f"part-sized error {err[0.25]:+.4%} is not clear of the wall-resolved "
        f"{err[8]:+.4%} -- the probe has drifted toward the crossover again")

    # AND THE SAME GUARD IN THE MESH PARAMETER RATHER THAN IN THE QUANTITY, WHICH IS THE
    # ONE THAT WOULD HAVE CAUGHT THIS BEFORE THE PROMOTION DID.  The assertion above is
    # about how big the coarse reading is; this one is about WHERE THE PROBE STANDS
    # relative to the sign change, which is what actually went wrong (§142).
    #
    # It bisects, so it costs about 13 solves -- roughly 4.3 s, against 2 solves for
    # everything above.  That is not the cheap option and it is not meant to be: what it
    # buys is that the failure names the fence instead of the symptom, and it fires while
    # the test is still green.  ON THE PRE-`cb4e3dd` GENOME THE OLD PROBE STOOD AT
    # h/t 0.998 AGAINST A CROSSOVER OF 0.944 -- A MARGIN OF 1.06x -- SO THIS LINE WOULD
    # HAVE BEEN RED THERE, ONE PROMOTION BEFORE THE SIGN ACTUALLY FLIPPED.  Nothing was
    # watching any of §133's nine approach failure; this is that watch, for this one.
    #
    # A self-aiming probe read off `crossover_h_over_t` was the other option and is worse
    # at any price: it is green however far the crossover moves, so it follows the failure
    # around instead of reporting it.
    #
    # The bound is 1.5x.  Measured margins are 2.207x (shipped) and 4.229x
    # (pre-`cb4e3dd`), i.e. 47% and 182% of headroom, against a ratio quantisation of
    # about 1.6% -- `h/t` is `arc_length / n_span / t_min` with `n_span` an INTEGER, so
    # both terms step, by 1/92 at the probe and 1/203 at the crossover (PLAN.md §144 §2).
    # 1.5 is ~30x that step, so an adjacent element count cannot redden it.
    h_probe = (sba.arc_length(g) / span[0.25]
               / float(np.min(np.asarray(g)[8:12])))
    h_star = sba.crossover_h_over_t(g, F)
    assert h_probe > 1.5 * h_star, (
        f"the part-sized probe sits at h/t = {h_probe:.4f} against a sign change at "
        f"{h_star:.4f} -- a margin of {h_probe / h_star:.3f}x, under the 1.5x this test "
        f"needs to keep demonstrating anything.  Re-aim the probe coarser (both figures "
        f"are quantised by the integer n_span, ~1.6% on this ratio), do not lower this "
        f"bound and do not read the probe off the crossover.")


# ---------------------------------------------------------------------------
# THE AXLE DROP IS READ AT A NODE, AND THAT IS A FIRST-ORDER ERROR (PLAN §62)
# ---------------------------------------------------------------------------

def test_the_vertical_displacement_runs_MONOTONICALLY_through_the_bottom():
    """Why `axle_drop_mm`'s nearest-node snap is a first-order error and not a rounding one.

    If `uy` peaked at the bottom, reading it at a node a fraction of a degree away would
    cost a SECOND-order error and nobody would need to care.  It does not peak: the spokes
    are a spiral, so the wheel has no mirror symmetry about the vertical and `uy` has a
    nonzero slope through `theta = -90`.  Measured here rather than argued, because every
    consequence in §62 follows from this one fact.
    """
    import wheel_wheel as ww
    with open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "best_solution.json")) as fh:
        genes = wg.genes_to_vector(json.load(fh)["genes"])
    mesh = ww.build_wheel(genes, "coarse")
    res = fem.solve_wheel(mesh)
    xy = np.asarray(mesh.coords)
    disp = res["u"].reshape(-1, 2)
    pn = np.unique(mesh.edge_sets["rim_outer"])
    th = np.degrees(np.arctan2(xy[pn, 1], xy[pn, 0]))
    d = (th + 90.0 + 180.0) % 360.0 - 180.0
    near = np.abs(d) < 1.0
    o = np.argsort(d[near])
    uy = disp[pn][near][o, 1]
    # strictly monotone across the bottom, so the reading is first-order in the offset
    steps = np.diff(uy)
    assert (steps < 0).all() or (steps > 0).all(), uy
    slope = (uy[-1] - uy[0]) / (d[near][o][-1] - d[near][o][0])
    assert abs(slope) > 0.005, f"slope {slope:.4f} mm/deg — the snap would be harmless"


def _gap_straddling_the_bottom(mesh):
    """The angular gap [deg] between the two rim nodes either side of `theta = -90`.

    HALF OF THIS IS THE EXACT CEILING on `patch_centre_offset_deg`, which is the distance
    from the bottom to the NEAREST of those two.  Measured as the straddling gap and not
    as a median over a window, for two reasons found in §148:

    - The rim is not uniformly discretised.  It is 12 fine blocks of 10 quadratic
      elements, one per 30 deg sector, separated by a single coarse element: at `coarse`
      the two gaps are 0.1465 and 1.3535 deg, 9.2x apart.  A median over the whole rim
      returns the coarse one and would put the ceiling at 0.677 instead of 0.073.
    - A window is undefined exactly where it matters.  If the bottom ever falls between
      the fine blocks the coarse gap is 1.3535 deg, a +/-1 deg window then holds 0 or 1
      nodes, and `median(diff(...))` is `nan` -- which still fails the assertion, but
      reports the failure as a nan rather than as the 9.2x jump it is.

    The straddling gap is exact, always defined, and needs no window.
    """
    xy = np.asarray(mesh.coords)
    pn = np.unique(mesh.edge_sets["rim_outer"])
    th = np.degrees(np.arctan2(xy[pn, 1], xy[pn, 0]))
    d = np.sort((th + 90.0 + 180.0) % 360.0 - 180.0)
    i = int(np.searchsorted(d, 0.0))
    return float(d[i] - d[i - 1])


def _uy_slope_through_bottom(mesh, res):
    """d(uy)/d(theta) [mm/deg] across `theta = -90` on the rim, from the rim nodes.

    The same reading the monotonicity test above takes; needed here three times, once
    per mesh variant, because the snap error is the offset times THIS mesh's slope.
    """
    xy = np.asarray(mesh.coords)
    disp = res["u"].reshape(-1, 2)
    pn = np.unique(mesh.edge_sets["rim_outer"])
    th = np.degrees(np.arctan2(xy[pn, 1], xy[pn, 0]))
    d = (th + 90.0 + 180.0) % 360.0 - 180.0
    near = np.abs(d) < 1.0
    o = np.argsort(d[near])
    uy = disp[pn][near][o, 1]
    dd = d[near][o]
    return (uy[-1] - uy[0]) / (dd[-1] - dd[0])


def test_the_interpolated_drop_is_the_same_number_when_a_node_IS_at_the_bottom():
    """The correction has to be inert where the artefact is absent, or it is a new bias.

    `axle_drop_interp_mm` interpolates `uy` to `theta = -90` exactly.  Where the nearest
    node already sits at the bottom the two readings must agree; where it does not, they
    must differ by roughly the offset times the slope above.  Both halves are asserted so
    that the correction is pinned as a correction rather than as a different quantity.
    """
    import wheel_wheel as ww
    with open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "best_solution.json")) as fh:
        genes = wg.genes_to_vector(json.load(fh)["genes"])
    mesh = ww.build_wheel(genes, "coarse")
    res = fem.solve_wheel(mesh)
    off, gap = res["patch_centre_offset_deg"], (res["axle_drop_mm"]
                                                - res["axle_drop_interp_mm"])
    # A node sits essentially at the bottom.  This is a bound on the MESH and not a
    # fitted number: the nearest node cannot be further than half the LOCAL rim node
    # spacing, and the bottom is refined for the contact patch -- 0.1465 deg here at
    # `coarse` against a 1.3535 deg median around the whole rim, 9.2x coarser.  So `off`
    # cannot exceed 0.0733 deg by construction and 0.10 has 1.36x on the ceiling, not on
    # the reading.  Checked as a ceiling and not just as a value: across the 12 solves
    # tabulated below, `|off| / (local spacing / 2)` never exceeds 1.000.
    assert abs(off) < 0.10, off

    # AND THAT CEILING IS ASSERTED, NOT ASSUMED -- THE POSITIONAL GUARD FOR THIS TEST.
    # `off` is a PHASE: where the bottom happens to fall between two rim nodes, uniform in
    # [0, spacing/2].  So the line above can pass on LUCK once the ceiling rises past
    # 0.10, with probability 0.10/ceiling, and it would keep passing for a while after the
    # premise "a node sits essentially at the bottom" had stopped being true.  Asserting
    # the ceiling instead makes it a theorem: `ceiling < 0.10` implies `abs(off) < 0.10`
    # for every phase, so this fires the moment the guarantee is lost rather than the
    # moment the coin lands badly.  §146 successor 0; the rule is §145 §5 -- where a fence
    # is locatable, prefer the parameter form.
    #
    # THE FENCE IS LOCATABLE AND THE TEST IS MOVING TOWARD IT.  The guarantee holds iff
    # the local spacing is under 0.20 deg, and at `coarse`:
    #
    #     genome              spacing   ceiling   margin to the fence
    #     pre-`cb4e3dd`        0.0937    0.0469         2.134x
    #     shipped              0.1465    0.0733         1.365x
    #
    # One promotion cost 36% of the margin.  At `smoke` it is already gone -- spacing
    # 0.3663, ceiling 0.1831, margin 0.546x -- and the plain mesh there reads
    # off = -0.1404, which fails the line above outright.  So this is not a hypothetical
    # fence: a coarser rim near the bottom is exactly what removes this test's premise,
    # and one more promotion of that size reaches it.
    ceiling = _gap_straddling_the_bottom(mesh) / 2.0
    assert ceiling < 0.10, (
        f"the nearest rim node can be up to {ceiling:.4f} deg from the bottom, so "
        f"`abs(off) < 0.10` above is no longer guaranteed -- it now passes with "
        f"probability {min(1.0, 0.10 / ceiling):.2f} on where the phase happens to land. "
        f"Refine the rim near the contact patch, or move this test to a config that "
        f"still has a node essentially at the bottom.  Do not raise the 0.10.")
    assert abs(gap) < 0.01 * res["axle_drop_interp_mm"]

    # AND THE CORRECTION IS EXACTLY THE FIRST-ORDER SNAP, WHICH IS THE CLAIM THE
    # DOCSTRING ABOVE ALREADY MAKES -- "they must differ by roughly the offset times the
    # slope above".  Until §141 the second half was carried instead by a pair of ratio
    # proxies: the filleted mesh's offset had to exceed the plain one's by 3x (shipped
    # layer profile) and 2x (per-genome rule).  Those were calibrated on the genome
    # before `cb4e3dd` and the promotion took the vehicle away rather than the finding:
    #
    #     coarse mesh          pre-`cb4e3dd`              shipped (`b729e86`)
    #     plain                off -0.0452               off -0.0671
    #     fillet SHIPPED       off -0.1635  (3.62x)      off +0.0413  (0.61x)
    #     fillet per-genome    off -0.1029  (2.28x)      off +0.0920  (1.37x)
    #
    # The filleted offsets did not merely shrink, they CHANGED SIGN, and all three now
    # sit inside the 0.10 deg band the first assertion calls "essentially at the bottom"
    # -- so there was no longer a not-at-the-bottom case for the ratios to compare, and
    # a bigger ratio would have been a fitted number with no vehicle under it.
    #
    # THE SIGN REVERSAL IS NOT A CHANGE OF MECHANISM, WHICH IS THE ONE READING THAT WOULD
    # HAVE MADE THIS THE WRONG REPAIR.  `off` is the signed distance from the bottom to
    # the nearest rim node, so it is bounded by half the local spacing (above) and its
    # SIGN is only which side of the bottom that node happens to land -- a sub-node-
    # spacing phase, and the re-cut rim re-places those nodes.  The mechanism is what
    # survives it: the relationship below holds to 2.52% on BOTH signs of `off`.
    #
    # The relationship is the honest pin and it does not rot: `gap` is `off` times the
    # slope of `uy` through the bottom, measured on the SAME mesh.  Across 2 genomes x
    # 2 configs (`smoke`, `coarse`) x these 3 mesh variants -- 12 solves, offsets from
    # -0.172 to +0.513 deg and BOTH signs -- `gap / (off * slope)` lands in
    # [0.9797, 1.0252], worst deviation 2.52%.  The 10% band below is 4x that.
    for tag, kw in (("plain", {}),
                    ("fillet SHIPPED", dict(fillet=True,
                                            layer_profile=ww.FILLET_LAYER_SHIPPED)),
                    ("fillet per-genome", dict(fillet=True))):
        m = ww.build_wheel(genes, "coarse", **kw)
        r = fem.solve_wheel(m)
        o = r["patch_centre_offset_deg"]
        g = r["axle_drop_mm"] - r["axle_drop_interp_mm"]
        predicted = o * _uy_slope_through_bottom(m, r)
        assert abs(g - predicted) < 0.10 * abs(predicted), (
            f"{tag}: gap {g:+.4e} is not the first-order snap {predicted:+.4e} "
            f"(off {o:+.5f} deg) -- the correction is a different quantity, not a "
            f"correction")
