"""
M6 verification of penalty contact against a rigid frictionless ground.

Two kinds of check, and the split matters.  The first group is closed-form: a straight
segment at uniform penetration has an exactly known energy, resultant AND nodal
distribution, so the kernel can be wrong in a way that no wheel-level invariant would
catch.  The second group is the wheel-level invariants, which are cheap and each one
exercises the whole chain.

The one failure mode neither group would catch on its own is a sign slip in the Macaulay
bracket: a contact model that PULLS still balances globally, still converges, and still
reports a plausible axle drop.  Only the pointwise pressure sees it, which is why
`test_contact_never_pulls` exists separately from the equilibrium checks.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import wheel_fem as fem          # noqa: E402
import wheel_genome as wg        # noqa: E402
import wheel_wheel as WW         # noqa: E402
import study_contact as sc       # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = "smoke"


@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


@pytest.fixture(scope="module")
def mesh(genes):
    return WW.build_wheel(genes, CFG)


@pytest.fixture(scope="module")
def res(mesh):
    return fem.solve_wheel_contact(mesh)


# ---------------------------------------------------------------------------
# THE KERNEL, AGAINST CLOSED FORM
# ---------------------------------------------------------------------------

def _flat_punch(length=4.0, penetration=0.01, eps_n=1e3, width=2.0, n_quad=6):
    """One straight quadratic segment lying `penetration` below a ground at y=0."""
    coords = np.array([[0.0, -penetration],
                       [0.5 * length, -penetration],
                       [length, -penetration]])
    con = fem.RigidGroundContact([[0, 1, 2]], y_ground=0.0, eps_n=eps_n,
                                 width=width, order=2, n_quad=n_quad)
    return coords, con, length, penetration, eps_n, width


def test_uniform_penetration_energy_is_exact():
    """Pi_c = eps_N * d^2/2 * L * w for a flat segment at uniform penetration d.

    Exact, not approximate: the integrand is constant, so any correct quadrature gives
    it.  That makes this a test of the geometry factors — the reference Jacobian, the
    width, and the eps_N convention — rather than of the integration.
    """
    coords, con, L, d, eps, w = _flat_punch()
    u = np.zeros(coords.size)
    assert con.energy(coords, u) == pytest.approx(eps * 0.5 * d * d * L * w, rel=1e-12)


def test_uniform_penetration_resultant_is_exact():
    """The vertical resultant is eps_N * d * L * w, and `total_force` agrees.

    Two routes: the assembled nodal force vector, and the quadrature over the pressure
    field.  They are computed differently — one through `jax.grad` and a scatter, the
    other in plain numpy — so agreement is evidence rather than a restatement.
    """
    coords, con, L, d, eps, w = _flat_punch()
    u = np.zeros(coords.size)
    expected = eps * d * L * w
    assert con.total_force(coords, u) == pytest.approx(expected, rel=1e-12)
    fy = -con.force(coords, u).reshape(-1, 2)[:, 1].sum()
    assert fy == pytest.approx(expected, rel=1e-12)


def test_the_nodal_distribution_is_1_4_1_and_not_equal_lumping():
    """The Q9 edge weights are 1/6, 4/6, 1/6.  Equal lumping gives the SAME resultant.

    This is the single most important test in the file, and it is the contact version of
    the traction patch test: lumping a quadratic edge equally onto its three nodes is
    wrong by a fixed factor, produces exactly the right total load, and is therefore
    invisible to every equilibrium check in the project.  A node-to-surface penalty —
    the obvious shortcut for contact — IS the lumped version.
    """
    coords, con, L, d, eps, w = _flat_punch()
    u = np.zeros(coords.size)
    fy = -con.force(coords, u).reshape(-1, 2)[:, 1]
    total = fy.sum()
    share = fy / total
    assert share == pytest.approx([1 / 6, 4 / 6, 1 / 6], rel=1e-10)
    # And the lumped alternative would have passed the resultant check above.
    assert not np.allclose(share, [1 / 3, 1 / 3, 1 / 3], atol=1e-3)


def test_no_energy_when_clear_of_the_ground():
    """Above the ground the term must vanish identically — value, force and tangent."""
    coords, con, *_ = _flat_punch(penetration=-0.01)      # sitting ABOVE y=0
    u = np.zeros(coords.size)
    assert con.energy(coords, u) == 0.0
    assert np.abs(con.force(coords, u)).max() == 0.0
    assert abs(con.stiffness(coords, u)).max() == 0.0


def test_the_assembled_force_is_the_gradient_of_the_assembled_energy():
    """Catches a wrong gather/scatter, which `jax.grad` cannot protect against.

    The kernel's derivative is exact by construction, but the node indexing that gathers
    a segment's displacements and scatters its force back is hand-written, and a
    transposed or misordered index there yields a force that is still plausible.
    """
    coords, con, *_ = _flat_punch(penetration=0.02)
    rng = np.random.default_rng(0)
    u = rng.normal(scale=1e-3, size=coords.size)
    f = con.force(coords, u)
    h = 1e-7
    for i in (0, 1, 3, 5):
        up, um = u.copy(), u.copy()
        up[i] += h
        um[i] -= h
        fd = (con.energy(coords, up) - con.energy(coords, um)) / (2 * h)
        assert f[i] == pytest.approx(fd, rel=1e-5, abs=1e-10)


def test_the_tangent_is_the_hessian_of_the_energy():
    coords, con, *_ = _flat_punch(penetration=0.02)
    rng = np.random.default_rng(1)
    u = rng.normal(scale=1e-3, size=coords.size)
    K = con.stiffness(coords, u).toarray()
    h = 1e-6
    for i in (1, 3):
        up, um = u.copy(), u.copy()
        up[i] += h
        um[i] -= h
        fd = (con.force(coords, up) - con.force(coords, um)) / (2 * h)
        assert K[i] == pytest.approx(fd, rel=1e-4, abs=1e-8)


def test_only_the_vertical_direction_carries_contact_force():
    """Frictionless against a FLAT ground: only u_y enters the gap, so there is no other
    place for force to appear.  Here that is structural rather than a modelling choice."""
    coords, con, *_ = _flat_punch(penetration=0.02)
    rng = np.random.default_rng(2)
    u = rng.normal(scale=1e-3, size=coords.size)
    fx = con.force(coords, u).reshape(-1, 2)[:, 0]
    assert np.abs(fx).max() == 0.0


# ---------------------------------------------------------------------------
# THE DISPATCH GUARDS
# ---------------------------------------------------------------------------

def test_solve_linear_refuses_a_contact_problem(mesh):
    """It would silently drop the ground and return a load-free field.

    The file already carries one trap of this shape that cannot be fixed — an `svk`
    problem through `solve_linear` returns a linear answer, because the two Hessians are
    equal at u=0.  This one can be, so it is.
    """
    prob = fem.wheel_contact_problem(mesh, indentation_mm=1.5)
    with pytest.raises(ValueError, match="contact"):
        fem.solve_linear(prob)


def test_solve_routes_contact_to_newton_even_under_linear_kinematics(mesh):
    """Contact is a nonlinear boundary condition regardless of the strain measure."""
    prob = fem.wheel_contact_problem(mesh, indentation_mm=1.5, kinematics="linear")
    assert not prob.nonlinear
    out = fem.solve(prob)
    assert "newton" in out, "solve() sent a contact problem down the linear path"


# ---------------------------------------------------------------------------
# THE WHEEL-LEVEL INVARIANTS
# ---------------------------------------------------------------------------

def test_contact_never_pulls(mesh, res):
    """No tension anywhere, which no resultant check can see.

    A sign slip in the bracket makes the ground suck the rim down outside the patch, and
    the totals still balance because the same wrong sign appears on both sides.
    """
    prob = fem.wheel_contact_problem(mesh, indentation_mm=res["axle_drop_mm"])
    p = prob.contact.pressure(np.asarray(mesh.coords), res["u"])
    assert p["pressure_mpa"].min() >= 0.0
    assert (p["pressure_mpa"] > 0).any(), "nothing is in contact at all"
    # Positive pressure only where the surface is actually below the ground.
    assert np.all(p["gap_mm"][p["pressure_mpa"] > 0] < 0.0)


def test_hub_reaction_balances_the_contact_resultant(res):
    assert res["equilibrium_error_n"] < sc.GATE_EQUILIBRIUM_N, res["equilibrium_error_n"]


def test_a_frictionless_flat_ground_applies_no_side_load(res):
    assert abs(res["contact_resultant_n"][0]) < sc.GATE_HORIZONTAL_N


def test_the_secant_hits_the_service_load(res):
    assert res["contact_force_n"] == pytest.approx(fem.TOTAL_FORCE_NEWTONS, rel=1e-8)


def test_penetration_is_negligible_against_the_rim_band(res):
    frac = res["max_penetration_mm"] / sc.RIM_BAND_MM
    assert frac < sc.GATE_PENETRATION_FRAC, frac


def test_the_centre_node_rise_is_not_the_axle_drop(mesh, res):
    """`centre_node_rise_mm` is a diagnostic and reads exactly like the answer.  It isn't.

    Two independent reasons, both measured rather than assumed:

    1. The patch MIGRATES, so the node at theta = -90 is generally not in contact —
       here it ends up 5.0e-3 mm CLEAR of the ground while the wheel is fully loaded.
    2. The contact is interior to a rim segment.  On this mesh a segment spans 3.75 deg
       and the patch is under 1 deg wide, so no rim NODE penetrates at all; the entire
       contact lives between nodes and is only visible to the quadrature.

    The axle drop is the prescribed indentation, which is exact by construction.  This
    test exists so that nobody "simplifies" the report by using the node instead.
    """
    prob = fem.wheel_contact_problem(mesh, indentation_mm=res["axle_drop_mm"])
    xy = np.asarray(mesh.coords)
    disp = res["u"].reshape(-1, 2)
    rim = np.unique(mesh.edge_sets["rim_outer"])
    node_gap = (xy[rim, 1] + disp[rim, 1]) - prob.contact.y_ground

    assert node_gap.min() > 0.0, (
        "a rim node now penetrates the ground — the mesh has become fine enough that "
        "the contact is no longer interior to a segment, so this test's premise (and "
        "the docstring above) need updating")
    assert res["centre_node_rise_mm"] != pytest.approx(res["axle_drop_mm"], abs=1e-4)


@pytest.mark.parametrize("phase", [0.0, 7.0])
def test_axle_drop_is_12_fold_periodic_under_contact(genes, phase):
    """Stronger than M4's version: the load is now an OUTPUT, so the contact search has
    to be periodic too, not just the sector indexing."""
    a = fem.solve_wheel_contact(WW.build_wheel(genes, CFG, phase_deg=phase))
    b = fem.solve_wheel_contact(WW.build_wheel(genes, CFG, phase_deg=phase + 30.0))
    rel = abs(a["axle_drop_mm"] / b["axle_drop_mm"] - 1.0)
    assert rel < sc.GATE_PERIODICITY_REL, (a["axle_drop_mm"], b["axle_drop_mm"])


def test_the_indentation_ramp_does_not_change_the_equilibrium(mesh):
    """A continuation path is a numerical device; the answer cannot depend on it.

    Note this is NOT `solve_nonlinear(steps=...)`, which scales `f_nodal` and `u_pre` — a
    displacement-driven contact problem has neither, so the ramp has to rebuild the
    problem at each partial indentation.
    """
    f = [fem.solve_wheel_contact_at(mesh, 1.5, steps=s)["contact_force_n"]
         for s in (1, 2, 4)]
    assert max(f) / min(f) - 1.0 < sc.GATE_CONTINUATION_REL, f


# ---------------------------------------------------------------------------
# THE HEADLINE
# ---------------------------------------------------------------------------

def test_the_real_patch_is_far_smaller_than_the_assumed_one(res):
    """M6's first half: 3.0 degrees was several times too wide.

    Pinned as a band rather than a value because it moves with mesh and design.  What
    must not change silently is that the assumption was wrong by a large factor and that
    the truth is nearer the Hertz solid-cylinder bound (0.31 deg) that M4 described as a
    lower bound and expected to be exceeded by far.
    """
    assert res["patch_half_deg"] < 0.5 * fem.CONTACT_PATCH_HALF_DEG, res["patch_half_deg"]
    assert res["patch_half_deg"] > fem.hertz_patch_half_angle_deg()


def test_but_the_assumed_patch_got_the_axle_drop_nearly_right(mesh, res):
    """M6's second half, and the reason M4's and M5's conclusions survive.

    This is the opposite of the pattern the last two gates found, so it is worth a test
    of its own: the assumption was badly wrong about the patch and close to right about
    the answer, because the drop is dominated by spoke and rim bending rather than by how
    the last few newtons are spread.
    """
    assumed = fem.solve_wheel(mesh)["axle_drop_mm"]
    rel = abs(res["axle_drop_mm"] / assumed - 1.0)
    assert rel < 0.05, (
        f"real contact moves the axle drop by {rel:.2%} against the assumed patch — "
        f"large enough that M4's and M5's numbers need re-reading")


def test_the_patch_migrates_with_phase(genes):
    """What the fixed model could not represent at all, rather than merely got wrong.

    The assumed patch is pinned at the bottom of the rim by construction, so this effect
    was not small in it — it was absent.
    """
    centres = []
    for ph in (0.0, 10.0, 20.0):
        r = fem.solve_wheel_contact(WW.build_wheel(genes, CFG, phase_deg=ph))
        centres.append(r["patch_centre_deg"])
    assert max(centres) - min(centres) > 0.5, centres


def test_the_sampled_patch_extent_is_biased_not_merely_noisy(mesh):
    """Pins WHY `patch_extent` exists rather than just reporting live Gauss points.

    The distinction is BIAS, not scatter — which took a measurement to establish, since
    the obvious framing (the sampled version is noisier under quadrature refinement) is
    not reliably true: on this mesh the two move by comparable amounts, and which is
    larger depends on the config.

    What IS systematic is the direction.  A Gauss point counts as in contact at any
    penetration however small, so the reported edge always sits at the outermost
    penetrating sample, which is outside the true edge — the sampled half-angle
    overstates by roughly 3x here.  The peak pressure is biased the other way, since a
    sampled maximum can only miss the true peak.  Both are reported as diagnostics and
    neither may be quoted.
    """
    a = fem.solve_wheel_contact(mesh, n_quad=6)
    b = fem.solve_wheel_contact(mesh, n_quad=20)
    assert abs(b["axle_drop_mm"] / a["axle_drop_mm"] - 1.0) < 1e-3

    for r in (a, b):
        assert r["patch_half_deg_sampled"] > 2.0 * r["patch_half_deg"], (
            f"the sampled extent ({r['patch_half_deg_sampled']:.3f}) no longer "
            f"overstates the zero-crossing one ({r['patch_half_deg']:.3f})")
    # Refining the quadrature can only find MORE of the true peak, never less.
    assert b["peak_pressure_mpa_sampled"] > a["peak_pressure_mpa_sampled"]


# ---------------------------------------------------------------------------
# THE M7 BRIDGE
# ---------------------------------------------------------------------------

def test_the_sharp_bracket_has_a_finite_difference_plateau(genes):
    """The plan's hazard, measured rather than assumed — and it does not bite.

    `<z>^2/2` is C^1, so a finite difference across a change in the contact set is
    meaningless.  But the kinks sit at DISCRETE gene values, so a generic design is not
    near one and the plateau exists anyway.  If this ever fails, the C^2 `smoothing_mm`
    branch is the fallback and `study_contact.run_gradient_plateau` measures what
    switching it on would cost.
    """
    rep = sc.run_gradient_plateau(genes, CFG, gene_ids=(8,),
                                  steps=(1e-2, 1e-3, 1e-4), smoothings=(0.0,))
    assert rep["pass"], rep
    assert not rep["smoothing_is_needed"], rep["worst_sharp_plateau"]


def test_the_fillet_genes_have_no_fea_gradient_at_all(genes):
    """A specification for M7, found by M6 rather than looked for.

    `R_hub` and `R_rim` do not enter the meshed geometry — the mesh models no fillets (see
    the M2b gate) — so their derivative through the FEA is not small, it is IDENTICALLY
    zero.  The beam model meanwhile prices them through `stress_concentration_kt`, so they
    are live genes that a gradient-based Stage 3 would find perfectly flat and never move.

    Asserted in both directions.  If a fillet ever gets meshed this fails, which is the
    signal to revisit M7's design — and the classifier must not quietly start reporting a
    run of identical zeros as a clean plateau, which is what it did before it knew the
    difference.
    """
    rep = sc.run_gradient_plateau(genes, CFG, gene_ids=(8, 12, 13),
                                  steps=(1e-2, 1e-3), smoothings=(0.0,))
    assert set(rep["insensitive_genes"]) == {"R_hub", "R_rim"}, rep["insensitive_genes"]
    assert not rep["genes"]["sharp"]["t0"]["insensitive"]
    for name in ("R_hub", "R_rim"):
        b = rep["genes"]["sharp"][name]
        assert all(d == 0.0 for d in b["derivatives"]), b["derivatives"]
        assert b["plateau_decades"] == 0, (
            "an insensitive gene is being scored as if it had a plateau")
