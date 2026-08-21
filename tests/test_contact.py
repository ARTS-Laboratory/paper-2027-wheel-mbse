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

    The reason is that the patch MIGRATES, so the node at theta = -90 is generally not in
    contact at all — measured, it sits clear of the ground while the wheel is fully
    loaded.  The axle drop is the prescribed indentation, which is exact by construction.
    This test exists so that nobody "simplifies" the report by using the node instead.

    IT USED TO ASSERT A SECOND, STRONGER PREMISE — that no rim NODE penetrates, i.e. that
    the whole contact lives between nodes and is visible only to the quadrature — and its
    own error message said that a failure meant "the mesh has become fine enough" and the
    premise needed updating.  It went red on the 2026-08-13 promotion and the message was
    WRONG about the cause, which is why the premise is now recorded rather than asserted.

    CONTACT_PLAN Step 2 measured it.  The rim OD is divided into `n_weld` segments over
    the weld and `n_rim_free` over the free arc, and those are NOT the same size: at
    `coarse`, 10 x 0.1682 deg + 10 x 2.8318 deg = 30.000 deg, a span ratio of 16.8 that is
    identical at every config.  The boundary between the two families sits 1.6824 deg from
    the bottom of the wheel and the patch centre sits at 1.885-2.082 deg, so the patch
    STRADDLES the size step — and whether a node falls inside it depends on which side it
    lands, which moves with the design, the phase and the kinematics.  Measured on the
    same mesh and genome: under linear the patch takes the weld side and holds 3 nodes at
    `medium`; under SVK it takes the free side and holds none.

    Refining does not fix that and was never what changed it.  What the premise was
    really protecting is that the penetration stays negligible, and that has its own
    test one screen up (`test_penetration_is_negligible_against_the_rim_band`, gating
    `max_penetration_mm / RIM_BAND_MM < 1e-3`), which passes on both genomes at every
    rung — 3.6e-4 here, 3x inside the gate.  So the check below is the same bound applied
    at the nodes, and the node COUNT is characterised, not required.
    """
    prob = fem.wheel_contact_problem(mesh, indentation_mm=res["axle_drop_mm"])
    xy = np.asarray(mesh.coords)
    disp = res["u"].reshape(-1, 2)
    rim = np.unique(mesh.edge_sets["rim_outer"])
    node_gap = (xy[rim, 1] + disp[rim, 1]) - prob.contact.y_ground

    # A node MAY be inside the patch — see the docstring.  What may not happen is a node
    # sinking in by a depth that matters against the band it is denting.
    assert -node_gap.min() / sc.RIM_BAND_MM < sc.GATE_PENETRATION_FRAC, (
        f"a rim node penetrates by {-node_gap.min():.3e} mm, "
        f"{-node_gap.min() / sc.RIM_BAND_MM:.2e} of the {sc.RIM_BAND_MM:.1f} mm band")
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

def test_the_real_patch_is_far_smaller_than_the_assumed_one(genes, res):
    """M6's first half: 3.0 degrees was several times too wide.

    Pinned as a band rather than a value because it moves with mesh and design.  What
    must not change silently is that the assumption was wrong by a large factor and that
    the truth is nearer the Hertz solid-cylinder bound (0.31 deg) that M4 described as a
    lower bound and expected to be exceeded by far.

    THE LOWER BOUND IS CHECKED AT `coarse`, NOT AT THE MODULE'S `smoke` FIXTURE, and that
    is the same scoping `test_the_sampled_patch_extent_is_biased_not_merely_noisy` below
    argues for at length — a claim about a CONVERGED number cannot be asked of a mesh that
    does not resolve the thing.  It went red on the 2026-08-13 promotion at 0.2965 deg
    against a bound of 0.3082, and CONTACT_PLAN Step 2 measured that this is a `smoke`
    artefact and nothing else.  `patch_half_deg / hertz`, both genomes, both kinematics:

        genome     kin       smoke    coarse    medium
        e126cc3    linear    0.962     1.200     1.329
        e126cc3    svk       1.304     1.088     1.171
        e4219f3    linear    1.610     1.082     1.139
        e4219f3    svk       1.720     1.295     1.215

    Below the bound in exactly ONE of the twelve cells, and it is the coarsest mesh of the
    pair the objective never runs.  The mechanism is Step 2's: at `smoke` the whole patch
    lives inside one 0.4206 deg rim element, so its zero crossing is interpolated within a
    single element's shape functions rather than resolved.

    The UPPER bound stays on `smoke` — "several times too wide" is a statement about a
    direction and a large factor, which holds at every tier and costs nothing to ask here.
    """
    assert res["patch_half_deg"] < 0.5 * fem.CONTACT_PATCH_HALF_DEG, res["patch_half_deg"]

    r = fem.solve_wheel_contact(WW.build_wheel(genes, "coarse"))
    assert r["patch_half_deg"] > fem.hertz_patch_half_angle_deg(), (
        f"the real patch ({r['patch_half_deg']:.4f} deg at coarse) has fallen below the "
        f"Hertz solid-cylinder bound ({fem.hertz_patch_half_angle_deg():.4f}), which M4 "
        f"called a LOWER bound and expected to be exceeded by far")


def test_the_assumed_patch_no_longer_stands_in_for_contact(mesh, res):
    """M6's second half, RENAMED because its answer changed and the old name asserted it.

    M6 measured that the assumed 3.0 deg patch was badly wrong about the patch and close
    to right about the ANSWER — 1.3% on the axle drop — because the drop is dominated by
    spoke and rim bending rather than by how the last few newtons are spread.  That is no
    longer true of the wheel that ships, and this test is the tripwire that said so.

    WHAT IT DOES AND DOES NOT INDICT.  CONTACT_PLAN Step 1 measured that the assumed patch
    CANNOT REACH the Stage-3 objective: the whole objective path runs
    `fem.solve_wheel_contact` / `wheel_contact_problem`, which has no `patch_half_deg`
    parameter at all, and passing one raises `TypeError`.  So this gate says nothing about
    the shipped wheel's deflection, which is a real-contact number.  What it indicts is
    every record still computed on `fem.solve_wheel`: M4, M5, `studies/study_gnl.py` and
    `studies/study_wheel_fea.py`.  PLAN.md section 19 read it the other way round and
    section 20 retracts that.

    THE DIVERGENCE IS REAL AND REFINEMENT-STABLE, which is why the bound moves rather than
    the test being scoped away.  |real / assumed - 1|:

        genome     kin       smoke    coarse    medium
        e126cc3    linear    5.272%    6.655%    6.250%
        e126cc3    svk       5.224%    6.760%    6.414%
        e4219f3    linear    3.077%    3.773%    3.793%
        e4219f3    svk       3.403%    3.677%    3.795%

    It does not refine away, it is a property of the DESIGN (the promoted wheel's rim is
    20% thicker with `R_rim` at its box maximum, so it conforms less), and it very nearly
    doubled across one promotion.  The band below is therefore set at the shipped genome's
    measured `smoke` value with room to move, and the assertion that carries the meaning is
    the two-sided one: this must not silently return to "the assumption was fine", and it
    must not silently get much worse either.
    """
    assumed = fem.solve_wheel(mesh)["axle_drop_mm"]
    rel = abs(res["axle_drop_mm"] / assumed - 1.0)
    assert 0.02 < rel < 0.08, (
        f"real contact moves the axle drop by {rel:.2%} against the assumed patch, "
        f"outside the 2-8% band measured across both genomes and three tiers — the "
        f"legacy assumed-patch records (M4, M5, study_gnl, study_wheel_fea) need "
        f"re-reading against whichever end this crossed")


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


def test_the_sampled_patch_extent_is_biased_not_merely_noisy(genes):
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

    THIS ONE TEST BUILDS ITS OWN `coarse` MESH INSTEAD OF TAKING THE MODULE'S `smoke`
    FIXTURE, and the reason is the first assertion.  "The converged answer does not care
    about the quadrature" is only true once the patch is resolved, and on `smoke` the
    promoted genome's patch is not: `patch_half_deg` is 0.53 deg against a rim element
    several times that, so the whole contact set lives inside one element and moving from
    6 to 20 Gauss points changes which elements are loaded at all.  Measured (§14):

        genome     smoke      coarse     medium
        350f4c7   -0.3233%   -0.0170%   +0.0195%
        36aed36   -0.0275%   +0.0001%   +0.0030%

    The failure was `smoke`-only and clears by 5x at `coarse`.  The other assertions here
    are about the SIGN of a bias and hold at every tier, so they come along for free.
    Everything else in this file stays on `smoke` — see the module docstring; this is the
    one claim in it that is about a converged number rather than about a direction.
    """
    mesh = WW.build_wheel(genes, "coarse")
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


def test_the_reported_local_element_contains_the_patch_rather_than_merely_being_near_it(
        mesh, res):
    """CONTACT_PLAN Step 2's resolution columns, pinned on the distinction that broke them.

    The rim OD carries TWO element families — `n_weld` segments over the weld and
    `n_rim_free` over the free arc — whose spans differ by 16.8x on the shipped genome and
    30.3x on its predecessor, constant up the whole mesh ladder.  So "the local element
    size at the patch" is a real quantity and `360 / len(rim_outer)` is not it: that is the
    mean of a bimodal distribution and lands between the two.

    The first version of `_patch_resolution` selected the segment whose CENTRE was nearest
    the patch centre, which is a different segment.  At `medium` it picked a 0.1052 deg
    weld segment 0.45 deg away while the patch was sitting inside the 1.7698 deg free
    segment spanning 1.6824-3.4523 — inverting the reported mechanism, and caught only
    because the node counts stopped reconciling with the node pitch.  Containment is the
    question, so containment is what is pinned here.
    """
    r = sc._patch_resolution(mesh, res, "linear")
    lo = r["rim_seg_span_min_deg"]
    hi = r["rim_seg_span_max_deg"]

    # The families are genuinely two, not one with scatter.  If the rim OD ever becomes
    # uniformly divided this fails, which is the signal that the columns below and PLAN.md
    # §20's mechanism describe a mesh that no longer exists.
    assert r["rim_seg_span_ratio"] > 5.0, (
        f"the rim OD's two element families have collapsed to a ratio of "
        f"{r['rim_seg_span_ratio']:.2f}; §20's mechanism assumes they have not")

    # THE PIN: the reported local span must be one of the spans that actually exist, and
    # the segment it came from must contain the patch centre.
    assert lo - 1e-9 <= r["local_seg_span_deg"] <= hi + 1e-9, r["local_seg_span_deg"]

    seg = np.asarray(mesh.edge_sets["rim_outer"])
    xy = np.asarray(mesh.coords)

    def off(idx):
        t = np.degrees(np.arctan2(xy[idx, 1], xy[idx, 0]))
        return (t + 90.0 + 180.0) % 360.0 - 180.0

    a, b = off(seg[:, 0]), off(seg[:, 2])
    span = np.abs((b - a + 180.0) % 360.0 - 180.0)
    centre = res["patch_centre_deg"]
    holds = ((np.minimum(a, b) <= centre) & (centre <= np.maximum(a, b))
             & (span < 180.0))
    assert holds.any(), "no rim segment contains the patch centre at all"
    assert r["local_seg_span_deg"] == pytest.approx(float(span[holds][0])), (
        f"reported local span {r['local_seg_span_deg']:.4f} deg is not the span of the "
        f"segment containing the patch centre ({float(span[holds][0]):.4f} deg) — the "
        f"'nearest centre' selection bug has come back")


# ---------------------------------------------------------------------------
# The committed artifact is the gate, and only a full-fidelity run may be filed as it.
#
# PLAN.md §41.  `main()` refuses at startup, before `load_genes` and before any solving,
# so every case here costs milliseconds.  `load_genes` is stubbed to raise a sentinel:
# reaching it is exactly the statement "the guard let this through", which is what makes
# the allowed direction testable without running the study.
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMITTED = os.path.join(HERE, "studies", "study_contact.json")


class _PastTheGuard(Exception):
    """Raised by the stubbed load_genes: the guard accepted these arguments."""


@pytest.fixture
def past_the_guard(monkeypatch):
    def _stub(_path):
        raise _PastTheGuard
    monkeypatch.setattr(sc, "load_genes", _stub)


@pytest.mark.parametrize("argv, why", [
    (["--quick"], "reduced fidelity: pins the smoke mesh regardless of --config"),
    (["--quick", "--config", "medium"], "--config does not undo --quick"),
    (["--sections", "penalty"], "partial: 1 of 7 sections"),
    (["--kinematics", "svk"], "non-linear"),
])
def test_a_degraded_run_may_not_be_filed_as_the_gate(monkeypatch, past_the_guard,
                                                     argv, why):
    """Refuse to write the committed study_contact.json from anything but the real gate.

    `--quick` is the case this was missing, and it is the dangerous one.  At that fidelity
    G1 reads 4.394e-04 with BOTH halves passing, against 1.7198e-03 and a red
    `regime_pass` at the real config (§39) — so a quick run filed here is not a coarse
    gate, it is a FALSE GREEN standing in for one.  It slipped the original guard because
    `--quick` alone leaves `full` True and `kinematics` "linear", and only those two were
    checked.  The other three cases pin the guard that already existed, so a later edit
    cannot trade one exposure for another.
    """
    before = os.path.getmtime(COMMITTED)
    monkeypatch.setattr(sys, "argv", ["study_contact.py", *argv])

    with pytest.raises(SystemExit) as excinfo:
        sc.main()

    assert excinfo.value.code != 0, f"a run that is {why} was accepted as the gate"
    assert os.path.getmtime(COMMITTED) == before, (
        f"a run that is {why} modified the committed artifact")


@pytest.mark.parametrize("argv", [
    ["--quick", "--out", "study_contact_quick_probe.json"],
    ["--sections", "penalty", "--out", "study_contact_probe.json"],
    [],
])
def test_the_guard_refuses_a_name_not_a_run(monkeypatch, past_the_guard, argv):
    """The refusal is about the NAME.  Redirected degraded runs, and the real gate, pass.

    Pinned in both directions because the cheap way to quieten a noisy guard is to widen
    it until a degraded run cannot happen at all — which would take `make contact` with
    it, whose whole purpose is a one-section run redirected to `CONTACT_OUT` — and the
    cheap way to quieten this test is to let the default path through again.  The empty
    argv case is the gate itself: if that ever starts refusing, `make studies` loses its
    sixth driver.
    """
    monkeypatch.setattr(sys, "argv", ["study_contact.py", *argv])
    with pytest.raises(_PastTheGuard):
        sc.main()
