"""M8a — the objective.

Split the way `test_gradient.py` is: the IDENTITIES that must hold exactly, the
STRUCTURAL FACTS about which genes and terms carry a gradient, and small reruns of
`study_objective.py`'s headline assertions at `smoke`.

Tolerances are read off `study_objective.GATE_*` rather than duplicated here, so a gate
cannot be relaxed in one place and stay strict in the other.

Everything runs on the `smoke` mesh on purpose: a converged contact solve at `coarse` is
expensive enough that a phase-batched objective would put minutes into `make test`.  Only
the exact identities are checked at full precision, and those do not care about the mesh.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import jax_config  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import study_objective as so  # noqa: E402
import wheel_adjoint as WA  # noqa: E402
import wheel_fea as W  # noqa: E402
import wheel_fem as fem  # noqa: E402
import wheel_genome as wg  # noqa: E402
import wheel_mesh as wm  # noqa: E402
import wheel_objective as WO  # noqa: E402
import wheel_wheel as WW  # noqa: E402

CFG = "smoke"
# Two, not one: the stress term aggregates ACROSS phases before it is compared to the
# allowable, so a single-phase stencil cannot tell a per-phase bug from an aggregation one.
N_PHASE = 2


@pytest.fixture(scope="module")
def genes():
    return so.load_genes()


@pytest.fixture(scope="module")
def mesh(genes):
    return WW.build_wheel(genes, CFG)


@pytest.fixture(scope="module")
def solved(genes, mesh):
    delta = float(fem.solve_wheel_contact(mesh)["axle_drop_mm"])
    res = fem.solve_wheel_contact_at(mesh, delta)
    prob = fem.wheel_contact_problem(mesh, indentation_mm=delta)
    return delta, res, prob


# ---------------------------------------------------------------------------
# THE IDENTITIES
# ---------------------------------------------------------------------------

def test_the_stress_refactor_did_not_move_a_single_bit(genes, mesh, solved):
    """`gauss_stresses` was refactored so the QoI and the report share one kernel.

    `lam` and `mu` moved from closed-over Python floats to traced arguments, which is a
    different jaxpr, so whether XLA emits the same instructions is empirical rather than
    obvious.  M4's, M5's and M6's committed reports all read stresses through this path.
    """
    _, res, prob = solved
    for nl in (False, True):
        s0, v0 = so._original_gauss_stresses(
            mesh.coords, mesh.conn, res["u"], order=mesh.cfg.order,
            lam=prob.lam, mu=prob.mu, nonlinear=nl)
        got = fem.gauss_stresses(mesh.coords, mesh.conn, res["u"],
                                 order=mesh.cfg.order, lam=prob.lam, mu=prob.mu,
                                 nonlinear=nl)
        assert np.array_equal(s0, got["sigma"]), (
            f"the factored stress kernel changed sigma at nonlinear={nl} — "
            f"re-read wheel_fem._stress_kernel")
        assert np.array_equal(v0, got["von_mises"])


def test_scaled_jacobian_has_one_definition_in_two_spellings(genes, mesh):
    """The numpy path M2b's report runs through and the jnp path the barrier needs."""
    a = wm.scaled_jacobian(mesh.coords, mesh.conn)
    b = np.asarray(wm.scaled_jacobian(jnp.asarray(mesh.coords), mesh.conn, xp=jnp))
    assert np.abs(a - b).max() < so.GATE_SJ_PATHS_MM


def test_the_pnorm_is_bounded_by_the_true_max(genes, mesh, solved):
    """A p-norm is a smooth LOWER bound on the max.  If it ever exceeded it, the volume
    weights would be wrong — the one way this term can be wrong and still look sane."""
    delta, res, prob = solved
    q = WA._qoi_pnorm_stress(prob)(jnp.asarray(mesh.coords), jnp.asarray(res["u"]),
                                   prob.contact.y_ground)
    assert 0.0 < float(q) <= WA.max_stress(prob, mesh.coords, res["u"]) * (1 + 1e-12)


def test_the_pnorm_rises_monotonically_toward_the_max_with_p(genes, mesh, solved):
    delta, res, prob = solved
    args = (jnp.asarray(mesh.coords), jnp.asarray(res["u"]), prob.contact.y_ground)
    vals = [float(WA._qoi_pnorm_stress(prob, p=p)(*args)) for p in (4.0, 10.0, 30.0)]
    assert vals[0] < vals[1] < vals[2]


def test_the_gauss_exponent_reaches_the_qoi_and_the_default_is_the_module_constant(genes):
    """M8b-i.6 step 1's plumbing, asserted at the objective rather than at the QoI.

    `adjoint_grads` looks a QoI up BY NAME, and the name carries no exponent — which is
    why `STRESS_PNORM_P` was unreachable from `t3_terms` and why M8b-i.5 could measure the
    constraint's non-convergence without being able to test the obvious cause.  The fix is
    the `(name, factory)` form, and this is what says it works: a different
    `stress_gauss_p` must produce a different constraint.

    The DEFAULT is now `STRESS_NOMINAL_P` (4.0) and not `WA.STRESS_PNORM_P` (30.0), which
    is step 2's whole point — the constraint is built on a convergent nominal stress and
    the peak is restored by `Kt`.  The high leg is pinned to `WA.STRESS_PNORM_P` so the
    module constant every historical record was measured at is still reachable and still
    means what it meant.
    """
    phases = WO.phase_stencil(n_phase=N_PHASE, scheme="uniform")
    kw = dict(phases=phases, meshes=WO.phase_meshes(genes, CFG, phases))

    base = WO.t3_terms(genes, CFG, **kw)["report"]
    high = WO.t3_terms(genes, CFG, stress_gauss_p=WA.STRESS_PNORM_P, **kw)["report"]
    pinned = WO.t3_terms(genes, CFG, stress_gauss_p=WO.STRESS_NOMINAL_P, **kw)["report"]

    assert base["stress_gauss_p"] == WO.STRESS_NOMINAL_P, (
        "the default exponent moved off STRESS_NOMINAL_P; the constraint is only "
        "mesh-convergent at the exponent M8b-i.6's sweep chose")
    assert WA.STRESS_PNORM_P == 30.0, (
        "the adjoint's module constant moved; it is the documented default and every "
        "stress magnitude M8a and M8b-i quote was measured at it")
    assert pinned["pnorm_stress_agg_mpa"] == pytest.approx(
        base["pnorm_stress_agg_mpa"], rel=1e-12), (
        "passing the default explicitly changed the answer, so the argument is not "
        "reaching the same code path the default does")
    assert base["pnorm_stress_agg_mpa"] < high["pnorm_stress_agg_mpa"], (
        "p=4 gave the same p-norm as p=30 — the exponent is being discarded, which is "
        "exactly what the bare-string QoI lookup used to do")
    # The true max is a property of the FIELD, not of the exponent used to summarise it.
    assert base["max_stress_mpa"] == pytest.approx(high["max_stress_mpa"], rel=1e-12)
    # So a lower p means a bigger rescale, and `c` is where the two meet.  `c` is a
    # diagnostic now, but it is still the diagnostic that says why.
    assert base["stress_scale_measured"] > high["stress_scale_measured"]


def test_the_probe_takes_no_gradient_and_leaves_the_constraint_alone(genes):
    """`stress_p_probe` is a measurement channel, not a second constraint.

    The sweep exists to ask which exponent has a value.  If asking changed the answer —
    moved the objective, the gradient, or the reported utilisation — the sweep would be
    measuring the sweep.  So the probed report must equal the unprobed one everywhere
    except in `pnorm_by_p`.
    """
    phases = WO.phase_stencil(n_phase=N_PHASE, scheme="uniform")
    kw = dict(phases=phases, meshes=WO.phase_meshes(genes, CFG, phases))

    plain = WO.t3_terms(genes, CFG, **kw)
    probed = WO.t3_terms(genes, CFG, stress_p_probe=(2.0, 8.0), **kw)

    assert plain["values"] == probed["values"], "the probe moved the objective"
    for k in plain["grads"]:
        assert np.array_equal(plain["grads"][k], probed["grads"][k]), (
            f"the probe moved the {k!r} gradient")
    assert set(probed["report"]) == set(plain["report"])
    for k in plain["report"]:
        if k not in ("rows", "pnorm_by_p"):
            assert plain["report"][k] == probed["report"][k], f"the probe moved {k!r}"
    assert plain["report"]["pnorm_by_p"] == {}
    assert sorted(probed["report"]["pnorm_by_p"]) == ["2.0", "8.0"]
    # Each probed exponent carries its per-phase values, so a phase-aggregation bug
    # cannot hide inside the aggregate.
    for k in ("2.0", "8.0"):
        assert len(probed["report"]["pnorm_by_p"][k]["pnorm_stress_mpa"]) == N_PHASE


# ---------------------------------------------------------------------------
# THE STRUCTURAL FACTS
# ---------------------------------------------------------------------------

def test_a_spiral_junction_has_a_flank_with_no_fillet(genes):
    """Not a curiosity — it is why `fillet_flanks` exists.

    At the shipped genome the `eta=-1` flank never crosses the hub circle, so that side
    has no corner to round.  Requiring all four flanks feasible declares a wheel that has
    been printed infeasible, and `ring_station`'s guard against the degenerate bracket is
    numpy-only, so the traced path returns NaN instead of raising.
    """
    hub, rim = WO.fillet_flanks(genes, WW.get_config(CFG))
    assert len(hub) < 2 or len(rim) < 2, (
        "both junctions now have two crossing flanks; the NaN this guards against "
        "cannot arise here, but re-read wheel_objective.fillet_flanks before relaxing it")


def test_the_fillet_margins_are_feasible_at_the_shipped_genome(genes):
    """The shipped genome has been exported to STEP with both fillets built."""
    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes, cfgo)
    m = np.asarray(WO._fillet_margins(jnp.asarray(genes), cfgo, W.S,
                                      W.HUB_RADIUS_MM, flanks))
    assert np.all(m > 0.0), f"margins {m} — a printed wheel scored infeasible"


def test_R_hub_is_dead_at_the_mesh_and_alive_only_in_gene_space(genes):
    """M7 proved `R_hub`/`R_rim` are dead at the mesh: `dcoords/dgene` is exactly zero.

    So any gradient they have comes from gene-space geometry.  THIS TEST USED TO BE CALLED
    "the fillet term is the only gradient R_hub has", and that title was measured false:
    at the shipped genome the fillet margins are `[+4.647, +0.125]`, both feasible, so the
    `fillet` barrier is FLAT and contributes exactly 0.0.  `R_hub`'s live loss gradient is
    `hub_overlap`'s (+645.8) and now also `fillet_cap`'s (+454.0).  `_fillet_margins` still
    gives it a gradient in the MARGIN, which is what the last assertion here checks and is
    a different claim from "in the loss".
    """
    mesh = WW.build_wheel(genes, CFG)
    _, cols = WA.insensitive_genes(genes, mesh)
    assert cols[12] == 0.0 and cols[13] == 0.0, "a fillet is now meshed — re-read M7"

    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes, cfgo)
    J = np.asarray(jax.jacrev(
        lambda v: WO._fillet_margins(v, cfgo, W.S, W.HUB_RADIUS_MM, flanks)
    )(jnp.asarray(genes)))
    assert not np.isnan(J).any(), "NaN in the fillet jacobian — see fillet_flanks"
    assert abs(J[0, 12]) > 1.0, (
        "R_hub lost its gradient in the hub margin; d(hub margin)/dR_hub should be ~2")

    # And the term that actually prices it in the loss.
    _, jt1 = WO._t1_cached_value_and_jacobian(jnp.asarray(genes), cfgo, None, W.S, flanks)
    row = np.asarray(jt1)[WO.T1_NAMES.index("fillet_cap")]
    assert row[12] > 0.0, (
        f"d(fillet_cap)/dR_hub = {row[12]}, so nothing pushes R_hub back under the slot "
        f"it has to fit in")


# ---------------------------------------------------------------------------
# THE BUILDABLE HUB FILLET CAP — PLAN.md §0(a)
# ---------------------------------------------------------------------------

def _both_hub_flanks_cross():
    """A genome whose `eta=-1` hub flank DOES cross the hub circle.

    Nothing on disk exercises that branch — all 16 Stage-2 elites and `best_solution`
    arrive on a shallow spiral where only one flank crosses (see
    `test_a_spiral_junction_has_a_flank_with_no_fillet`).  So the two-crossing path in
    `hub_void_deg` needs a synthetic witness or it is never executed by the suite.
    Found by uniform draws in the box; about 1 in 80 qualifies.
    """
    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    rng = np.random.default_rng(0)
    cfgo = WW.get_config(CFG)
    for _ in range(400):
        g = low + rng.random(14) * (high - low)
        try:
            if len(WO.fillet_flanks(g, cfgo)[0]) == 2:
                return g
        except Exception:                       # a draw the sampler cannot even build
            continue
    pytest.skip("no two-crossing genome found in 400 draws")


def test_the_hub_cap_reproduces_the_measured_void(genes):
    """The analytic slot against the one measured on the BUILT SOLID.

    The hub fillet milestone classified material on sample rings and found the void
    between adjacent spoke roots to be 9.907 deg = 2.196 mm of arc.  This function gets
    there from the genome alone, with no CAD kernel: 9.977 deg, 0.070 deg apart.  The
    difference is `_embed`'s radial plunge starting from the centerline endpoint rather
    than the flank endpoint — see `wheel_wheel.hub_void_deg`.

    0.25 deg is one sample of the OCC classifier `make hubcap` gates this against, i.e.
    the resolution of the instrument, not a tolerance picked to pass.
    """
    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes, cfgo)
    void = float(WW.hub_void_deg(genes, cfgo, W.S, W.HUB_RADIUS_MM, flanks[0]))
    assert abs(void - 9.907) <= 0.25, (
        f"void {void:.4f} deg against 9.907 measured on the solid")

    # TWO LIMITS, and the cap is the smaller.  See `HUB_CAP_SHARE` for why it is a `min`
    # and which of the two `make hubcap` has actually observed binding.
    by_slot = WO.HUB_CAP_SHARE * W.HUB_RADIUS_MM * np.radians(void)
    by_thickness = WO.HUB_CAP_THICKNESS_SHARE * genes[8]
    cap = float(WO.hub_fillet_cap_mm(genes, cfgo, W.S, W.HUB_RADIUS_MM, flanks))
    assert cap == pytest.approx(min(by_slot, by_thickness))
    # At the shipped genome the SLOT is the smaller of the two — 1.106 against 1.288 — so
    # this design also exercises that branch of the min.
    assert by_slot < by_thickness, (
        f"slot {by_slot:.4f} vs thickness {by_thickness:.4f}: the shipped genome no longer "
        f"exercises the slot branch, so the min is untested here")
    # `make hubcap` bisected the real per-corner threshold at this design to 1.300 mm.  The
    # cap must stay UNDER it: over-promising is the defect this exists to remove, and
    # under-promising only leaves fillet on the table.
    assert cap <= 1.300, f"cap {cap:.5f} over-promises against the measured 1.300"


def test_the_thickness_branch_of_the_cap_binds_on_a_thin_root(genes):
    """The other half of the `min`, and it is the half `make hubcap` observes binding.

    The slot binds at the shipped genome, so without this the thickness term could be
    deleted and every other test would still pass.  Thinning `t0` drives the thickness
    limit down (and, incidentally, WIDENS the slot, since a thinner root leaves more gap) —
    so the two limits move in opposite directions and the crossover is reachable inside
    the gene box rather than hypothetical.
    """
    cfgo = WW.get_config(CFG)
    g = np.asarray(genes, dtype=float).copy()
    g[8] = 2.0                                   # MIN_WALL_MM, the low bound on t0
    flanks = WO.fillet_flanks(g, cfgo)
    void = float(WW.hub_void_deg(g, cfgo, W.S, W.HUB_RADIUS_MM, flanks[0]))
    by_slot = WO.HUB_CAP_SHARE * W.HUB_RADIUS_MM * np.radians(void)
    by_thickness = WO.HUB_CAP_THICKNESS_SHARE * g[8]
    cap = float(WO.hub_fillet_cap_mm(g, cfgo, W.S, W.HUB_RADIUS_MM, flanks))

    assert by_thickness < by_slot, (
        f"at t0 = 2.0 the thickness limit {by_thickness:.4f} is still not the smaller "
        f"(slot {by_slot:.4f}) — the crossover is outside the box and the min is dead code")
    assert cap == pytest.approx(by_thickness)
    # And the gradient has to follow the branch that is live.
    d = np.asarray(jax.grad(
        lambda v: WO.hub_fillet_cap_mm(v, cfgo, W.S, W.HUB_RADIUS_MM, flanks)
    )(jnp.asarray(g)))
    assert d[8] == pytest.approx(WO.HUB_CAP_THICKNESS_SHARE)
    assert np.linalg.norm(d[:8]) == 0.0, (
        f"the shape genes still move the cap on the thickness branch: {d[:8]}")


def test_the_hub_cap_is_the_same_in_numpy_and_jnp(genes):
    """The `soft_barrier` / Kt-twin precedent: one function, two backends, one answer."""
    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes, cfgo)
    a = float(WW.hub_void_deg(genes, cfgo, W.S, W.HUB_RADIUS_MM, flanks[0], xp=np))
    b = float(WW.hub_void_deg(jnp.asarray(genes), cfgo, W.S, W.HUB_RADIUS_MM, flanks[0],
                              xp=jnp))
    assert a == pytest.approx(b, rel=1e-12)


def test_the_hub_cap_handles_a_flank_that_never_crosses(genes):
    """The `fillet_flanks` discipline, on the term that now depends on it.

    A flank that does not cross gives `ring_station` a degenerate bracket, and its guard
    is numpy-only — under tracing the division by zero becomes a SILENT NaN in twelve of
    fourteen gradient components.  Both branches are checked: the shipped genome (one
    crossing flank, the common case) and a synthetic two-crossing one.
    """
    cfgo = WW.get_config(CFG)
    for g in (genes, _both_hub_flanks_cross()):
        flanks = WO.fillet_flanks(g, cfgo)
        d = np.asarray(jax.grad(
            lambda v: WO.hub_fillet_cap_mm(v, cfgo, W.S, W.HUB_RADIUS_MM, flanks)
        )(jnp.asarray(g)))
        assert not np.isnan(d).any(), f"NaN in the cap gradient, flanks={flanks[0]}"
        assert np.linalg.norm(d[:10]) > 0.0, "the cap does not move with the shape genes"
        # The cap is pure geometry: the two fillet radii and t3 cannot move the slot.
        assert np.all(d[10:] == 0.0), f"the cap moved with a gene it cannot depend on: {d}"


def test_a_negative_cap_is_a_number_not_a_nan():
    """Adjacent spoke roots can close the slot entirely, and `Kt` has to survive it.

    The companion to `test_kt_saturates_with_a_finite_zero_gradient_and_never_relieves`:
    that one saturates on a small RADIUS, this one on a slot that does not exist.  Measured
    on a real box draw: void -9.06 deg, cap -1.004 mm.  `stress_concentration_kt`'s
    double-`where` is what keeps the `(t/2R)^0.65` off the negative branch it never takes.
    """
    for cap in (-1.004, 0.0, 0.05):
        def f(v):
            return WO.stress_concentration_kt(
                WO.hub_fillet_r_effective(v, jnp.asarray(cap)), v[8])
        g = jnp.asarray(np.array([0.0] * 8 + [8.66, 3.0, 3.0, 2.0, 1.5598, 3.0]))
        val, d = jax.value_and_grad(f)(g)
        assert float(val) == WO.KT_CLAMP[1], f"cap={cap} gave Kt={float(val)}"
        d = np.asarray(d)
        assert np.all(np.isfinite(d)), f"cap={cap} gave a non-finite gradient {d}"
        assert d[12] == 0.0 and d[8] == 0.0, f"cap={cap} gave a live gradient {d}"


def test_R_eff_is_exactly_the_cap_more_than_one_rung_above_it(genes):
    """WHY THE POLYNOMIAL smooth-min AND NOT THE SQRT ONE.

    Outside the blend the result is `min(a, b)` TO THE BIT, so a design a rung above its
    cap is priced on exactly the cap with exactly zero derivative in `R_hub` — an assertion
    rather than an approximation.  The sqrt form would put `R_eff` 1.3% below the cap here
    and invent a fillet reduction nothing measured.
    """
    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes, cfgo)
    cap = WO.hub_fillet_cap_mm(genes, cfgo, W.S, W.HUB_RADIUS_MM, flanks)
    gj = jnp.asarray(genes)

    # The shipped genome sits ~2.7 rungs above its cap.
    assert float(WO.hub_fillet_r_effective(gj, cap)) == float(cap)
    (_, d_hub), _ = WO.junction_kt(genes, cfgo, flanks=flanks)
    assert d_hub[12] == 0.0, (
        f"d(Kt_hub)/dR_hub = {d_hub[12]}, but R_hub is above the slot — buying more of it "
        f"must buy exactly nothing")

    # Well below the cap, nothing is capped and the original physics is back.
    g2 = np.asarray(genes).copy()
    g2[12] = 0.5 * float(cap)
    assert float(WO.hub_fillet_r_effective(jnp.asarray(g2), cap)) == g2[12]
    (_, d2), _ = WO.junction_kt(g2, cfgo, flanks=flanks)
    assert d2[12] < 0.0, "below the cap a bigger fillet must still lower Kt"


def test_the_cap_gradient_matches_a_finite_difference(genes):
    """Solve-free, so it can afford to be tight.

    This is where the arithmetic gene 12 used to carry in
    `test_the_stress_gradient_obeys_the_product_rule` now lives: `R_hub` is exactly inert
    above its cap, so its row there became vacuous.
    """
    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes, cfgo)

    def cap(v):
        return float(WO.hub_fillet_cap_mm(jnp.asarray(v), cfgo, W.S, W.HUB_RADIUS_MM,
                                          flanks))

    d = np.asarray(jax.grad(
        lambda v: WO.hub_fillet_cap_mm(v, cfgo, W.S, W.HUB_RADIUS_MM, flanks)
    )(jnp.asarray(genes)))

    for i in (0, 1, 4, 7, 8):
        best = min(
            abs((cap(_bump(genes, i, +h)) - cap(_bump(genes, i, -h))) / (2 * h) - d[i])
            / max(abs(d[i]), 1e-12)
            for h in (1e-4, 1e-5, 1e-6))
        assert best < 1e-6, f"gene {i}: FD vs adjoint rel error {best:.2e}"


def test_the_t1_term_lists_stay_in_lockstep(genes):
    """Five lists have to agree, and adding the seventh term touched all of them."""
    assert WO.T1_NAMES == WO._T1_WEIGHT_KEYS
    assert set(WO.T1_NAMES) <= set(WO.TERMS)
    assert set(WO.TERMS) <= set(WO.DEFAULT_WEIGHTS)
    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes, cfgo)
    v = WO.t1_vector(jnp.asarray(genes), cfgo, None, W.S, flanks)
    assert len(v) == len(WO.T1_NAMES)


def test_the_fillet_cap_barrier_is_live_at_the_shipped_genome(genes):
    """It has to BITE somewhere, and the shipped genome is 0.45 mm over its slot.

    15 of the 16 Stage-2 elites are above their cap, including both production multi-start
    points, so a barrier that read zero here would be measuring nothing.
    """
    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes, cfgo)
    v, J = WO._t1_cached_value_and_jacobian(jnp.asarray(genes), cfgo, None, W.S, flanks)
    k = WO.T1_NAMES.index("fillet_cap")
    assert float(np.asarray(v)[k]) > 0.0, "the cap barrier is flat on a design over its cap"
    assert np.asarray(J)[k][12] > 0.0


def test_R_rim_is_still_effectively_inert_and_that_is_recorded(genes):
    """A FINDING, not a passing grade.

    `fillet_feasibility` was built to give both fillet genes a gradient and only `R_hub`
    got one.  At the rim the arrival is near-tangential, so moving `R_rim` moves the ring
    locus and the offset point together and the margin is stationary.  If this ever
    starts failing, the rim junction geometry has changed and M8b's gene census and the
    study's verdict both need revisiting.
    """
    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes, cfgo)
    J = np.asarray(jax.jacrev(
        lambda v: WO._fillet_margins(v, cfgo, W.S, W.HUB_RADIUS_MM, flanks)
    )(jnp.asarray(genes)))
    assert abs(J[1, 13]) < 1e-4, (
        f"d(rim margin)/dR_rim is now {J[1, 13]:.3e}; R_rim has become live, which is "
        f"good news that invalidates a documented finding")


def test_the_smoothness_term_no_longer_counts_anything(genes):
    """`400*n_infl` was an integer from `count_nonzero` and had exactly zero gradient.

    Its replacement must have a nonzero gradient in the shape genes, or the rewrite
    bought nothing.
    """
    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes, cfgo)
    k = WO.T1_NAMES.index("smoothness")
    J = np.asarray(jax.jacrev(
        lambda v: WO.t1_vector(v, cfgo, None, W.S, flanks))(jnp.asarray(genes)))
    assert np.linalg.norm(J[k, :8]) > 0.0


def test_the_t1_cache_hits_on_repeat_calls_and_splits_on_cfg(genes):
    """A cache that silently recompiles every call would erase the whole point of jitting it.

    Same genes, same cfg must hit the one cached closure; a different cfg is a different
    static recipe and must trace its own — proving `_t1_cached_value_and_jacobian`'s key
    actually discriminates rather than either never missing or always missing.
    """
    WO._T1_CACHE.clear()
    WO.objective(genes, "smoke", tiers=("t1",))
    WO.objective(genes, "smoke", tiers=("t1",))
    assert len(WO._T1_CACHE) == 1, "same genes/cfg should hit the cache, not recompile"
    WO.objective(genes, "coarse", tiers=("t1",))
    assert len(WO._T1_CACHE) == 2, "a different cfg must trace its own closure"


def test_buckling_is_inert_and_currently_inactive(genes):
    """The one term with no gradient, and the assertion that it costs nothing today.

    The master plan replaces it with `lambda_min(K_t)`; this repo has no eigen-solver, so
    it stays the Euler proxy until M9.  It is exactly zero at the shipped design, which is
    why an inert term is tolerable here.  If it starts firing, this fails and says so.
    """
    _, _, brk = WO.objective(genes, CFG, tiers=("t1",))
    assert brk["terms"]["buckling"]["grad_norm"] == 0.0
    assert brk["terms"]["buckling"]["value"] == 0.0, (
        "the Euler buckling proxy has started to bite, and it has no gradient — "
        "M8b cannot descend it; see PLAN.md scope decision 1")


# ---------------------------------------------------------------------------
# SMALL RERUNS OF THE GATE
# ---------------------------------------------------------------------------

def test_the_closed_form_terms_agree_with_finite_differences(genes):
    rep = so.run_closed_form(genes, CFG)
    assert rep["pass"], f"worst rel {rep['worst_rel']:.3e}"


def test_the_load_control_coupling_is_not_optional(genes):
    """The gate's headline: dropping the coupling does not lengthen the stress gradient,
    it reverses it."""
    rep = so.run_coupling(genes, CFG, gene_ids=(10,), steps=(1e-4, 1e-5))
    assert rep["pass"], f"worst rel vs the full pipeline {rep['worst_rel_total']:.3e}"
    assert rep["worst_rel_frozen_only"] > 1.0, (
        "a frozen-indentation gradient now agrees with the full pipeline, which would "
        "mean the coupling term has become negligible — re-read study_objective's G3")


def test_the_stress_adjoint_has_a_finite_difference_plateau(genes):
    rep = so.run_stress_plateau(genes, CFG, gene_ids=so.QUICK_GENES,
                                steps=(1e-3, 1e-4, 1e-5))
    assert rep["pass"], f"min decades {rep['min_decades']}"


def test_no_term_dominates_the_table_without_moving_the_gradient(genes):
    """Gate 8, and gate 7 in the same run.  The failure mode the GA did not have.

    The census is over the shipped genome AND the elites: inert everywhere is a property
    of the objective, inert at one design is a property of that design.  See
    `study_objective`'s docstring for the measurement that replaced the pointwise form.
    """
    rep = so.run_total(genes, CFG, n_phase=2, gene_ids=(10,), steps=(1e-3, 1e-4),
                       elites=so.load_elites(limit=2))
    assert rep["census_ok"], (
        f"terms {rep['inert']} are worth over {100*so.GATE_INERT_VALUE:.0f}% of the "
        f"loss with a ZERO gradient at every design scored, so no weight can make a "
        f"gradient method reduce them; expected only {list(so.INERT_EXPECTED)}")
    assert rep["min_decades"] >= 1, f"worst rel {rep['worst_best_rel']:.3e}"


@pytest.mark.parametrize("r", [0.0, 0.05, 0.0999, 0.1, 0.25, 0.5, 1.0, 1.5597674, 2.0,
                               3.0, 5.0, 12.0])
@pytest.mark.parametrize("t", [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
def test_the_jnp_kt_is_the_numpy_kt(r, t):
    """The load-bearing new test of M8b-i.6 step 2: two implementations, one formula.

    The constraint prices `Kt` in jnp so it can be differentiated; `wheel_fea` and the STEP
    exporter price it in numpy, because `wheel_fea` must import in the CadQuery env, which
    has no jax (`test_import_hygiene`).  Two implementations of the same physics is exactly
    the arrangement that drifts silently, and a drift here would mean the optimizer is
    descending on a stress concentration the built part does not have.

    The grid spans both clamp bounds and BOTH sides of the degenerate `r < 0.1` branch,
    including `r = 0.0` — which is not hypothetical: it is what the shipped hub actually
    built (`wheel_step_manifest.json`, 0/12 edges filleted).
    """
    assert float(WO.stress_concentration_kt(r, t)) == W.stress_concentration_kt(r, t)


def test_kt_hub_is_priced_on_the_buildable_radius_and_kt_rim_on_the_requested_one(genes):
    """PLAN.md §0(a): the hub's `Kt` reflects the fillet the PART GETS, not the gene.

    WHAT THIS USED TO SAY.  "only the four genes it depends on move it" — `Kt_hub` was
    `kt(R_hub, t0)` and its gradient was nonzero on exactly `{12, 8}`.  That is no longer
    true and must not be: `R_hub` is capped at half the slot between adjacent spoke roots,
    the slot is a function of the eight centerline genes and `t0`, so `Kt_hub` moves with
    all ten.  The shipped genome sits 0.45 mm above its cap, so `dKt_hub/dR_hub` there is
    exactly 0.0 — the optimizer can no longer buy hub fillet it will not receive.

    The RIM is deliberately not capped (its junction has a whole ring band to grow into,
    not a slot), so its half of this test is unchanged and is the control: if the rim's
    row ever starts looking like the hub's, the cap has leaked across the pairing.
    """
    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes, cfgo)
    cap = WO.hub_fillet_cap_mm(genes, cfgo, W.S, W.HUB_RADIUS_MM, flanks)
    (kt_hub, d_hub), (kt_rim, d_rim) = WO.junction_kt(genes, cfgo, flanks=flanks)

    r_eff = min(float(genes[12]), float(cap))       # hard min: far outside the blend
    assert kt_hub == pytest.approx(W.stress_concentration_kt(r_eff, genes[8]))
    assert kt_hub > W.stress_concentration_kt(genes[12], genes[8]), (
        "capping the radius must RAISE Kt — the part is sharper than the gene asked for")
    assert kt_rim == pytest.approx(W.stress_concentration_kt(genes[13], genes[11]))

    # The rim, unchanged: the pairing is still exact and still not misrouted.
    assert d_rim[13] < 0.0, "dKt_rim/dR_rim >= 0: a bigger fillet must relieve, not raise"
    assert d_rim[11] > 0.0, "dKt_rim/dt3 <= 0: a thicker section must raise, not relieve"
    assert set(np.nonzero(d_rim)[0].tolist()) == {13, 11}, (
        f"Kt_rim has a gradient in {sorted(set(np.nonzero(d_rim)[0].tolist()) - {13, 11})}")

    # The hub.  The pairing still holds in the sense that matters — the hub cannot see the
    # rim's genes — but the shape genes are now live and `R_hub` is not.
    assert d_hub[13] == 0.0 and d_hub[11] == 0.0 and d_hub[10] == 0.0, (
        f"Kt_hub moved with a rim gene: {d_hub}")
    assert d_hub[12] == 0.0, (
        f"dKt_hub/dR_hub = {d_hub[12]}, but R_hub is above its cap — buying more of it "
        f"must buy exactly nothing")
    assert d_hub[8] > 0.0, "dKt_hub/dt0 <= 0: a thicker section must raise, not relieve"
    assert np.linalg.norm(d_hub[:8]) > 0.0, (
        "the centerline genes do not move Kt_hub, so the cap is not being differentiated "
        "through — re-read junction_kt on why it must not be frozen")


def test_kt_saturates_with_a_finite_zero_gradient_and_never_relieves():
    """The upper clamp is reachable in-box, so its gradient must be 0.0 and not NaN.

    `t0 = 10.0` with `R_hub = 0.5` gives `Kt = 5.47`, clamped to 3.5 — inside the gene box,
    so a descent can walk there.  `r = 0.0` is the sharper hazard, and it is not
    hypothetical: it is what the shipped hub actually built.  That branch must not evaluate
    `(t/0)**0.65` even on the path not taken, because the `inf` propagates into the
    gradient of the path that WAS taken.  Hence the double-`where`.

    The LOWER clamp is unreachable and is asserted to be: `Kt = 1 + C*(t/2R)^0.65 >= 1` for
    any positive section, so `KT_CLAMP[0]` is a floor the formula already respects.  Worth
    pinning because "clamped to [1.0, 3.5]" reads as two live bounds and only one is.
    """
    lo, hi = WO.KT_CLAMP
    grad = jax.grad(WO.stress_concentration_kt, argnums=(0, 1))

    for r, t in ((0.5, 10.0), (0.0, 5.0), (0.05, 0.5)):
        assert float(WO.stress_concentration_kt(r, t)) == pytest.approx(hi)
        d = [float(x) for x in grad(float(r), float(t))]
        assert all(np.isfinite(d)), f"non-finite gradient {d} at r={r}, t={t}"
        assert d == [0.0, 0.0], f"saturated at {hi} but still has gradient {d}"

    for r in (0.1, 1.0, 5.0, 50.0):
        for t in (0.1, 1.0, 10.0):
            assert float(WO.stress_concentration_kt(r, t)) >= lo


def test_the_report_carries_both_junctions_and_the_headline_is_their_max(genes):
    """Two junctions, two barriers — so two utilisations must be visible.

    The headline `stress_utilisation` is `max(hub, rim)` for reporting only; the CONSTRAINT
    sums a `soft_barrier` on each, because a hard max would zero the gradient of whichever
    junction is not currently worst and reintroduce the argmax kink the p-norm exists to
    blur.  If the report ever showed one number, a run could round the hub while the rim
    was the binding one and nothing would say so.
    """
    phases = WO.phase_stencil(n_phase=1, scheme="uniform")
    rep = WO.t3_terms(genes, CFG, phases=phases,
                      meshes=WO.phase_meshes(genes, CFG, phases))["report"]

    agg, allow = rep["pnorm_stress_agg_mpa"], WO.ALLOWABLE_STRESS_MPA
    for j in ("hub", "rim"):
        assert rep[f"stress_utilisation_{j}"] == pytest.approx(
            rep[f"kt_{j}"] * agg / allow, rel=1e-12), (
            f"the {j} utilisation is not Kt_{j} * sigma_nominal / ALLOWABLE")
    assert rep["stress_utilisation"] == max(rep["stress_utilisation_hub"],
                                            rep["stress_utilisation_rim"])

    # And the report has to say WHICH radius the hub was priced on, or `kt_hub` reads as
    # `kt(R_hub, t0)` and is not.
    assert rep["r_hub_effective_mm"] <= rep["hub_fillet_cap_mm"] + 1e-12
    assert rep["r_hub_effective_mm"] <= genes[12] + 1e-12
    assert rep["kt_hub"] == pytest.approx(
        float(WO.stress_concentration_kt(rep["r_hub_effective_mm"], genes[8])), rel=1e-9)


def test_the_stress_gradient_obeys_the_product_rule(genes, monkeypatch):
    """`d(Kt*agg) = dKt*agg + Kt*dagg`, finite-differenced with the barrier FORCED ACTIVE.

    THIS IS THE CHECK THAT M8b-i.6 step 2 LIVES OR DIES ON, and it has to be made here
    rather than left to gate 7.  The new constraint puts the shipped design at utilisation
    ~0.39, so `soft_barrier` is flat, `stress` and `d_stress` are both exactly zero, and a
    completely wrong product rule would pass every end-to-end gate in the tree.  So the
    allowable is dropped until the barrier is on the quadratic branch, which changes
    nothing about the arithmetic being tested.

    Gene 13 reaches the stress term ONLY through `Kt`, so it isolates `dKt*agg`; genes 8
    and 11 reach it through both factors, so they are the ones that fail if the two
    contributions are added wrongly.

    GENE 12 USED TO BE IN THAT LIST AND IS NOT ANY MORE.  `R_hub` is now priced through
    the buildable cap, and the shipped genome sits above its cap, so `dKt_hub/dR_hub` is
    exactly 0.0 and its row here would assert `0 == 0` — vacuous, and worse, vacuous in a
    way that looks like coverage.  What it used to check has moved to two solve-free tests
    that can afford to be far tighter than this one's 1e-4:
    `test_R_eff_is_exactly_the_cap_more_than_one_rung_above_it` and
    `test_the_cap_gradient_matches_a_finite_difference`.  A shape gene is NOT substituted
    in: `h = 1e-5` through a nonlinear contact solve may not clear the 1e-4 bar, and adding
    one without measuring that first is how a flaky gate gets written.
    """
    monkeypatch.setattr(WO, "ALLOWABLE_STRESS_MPA", 2.0)
    phases = WO.phase_stencil(n_phase=1, scheme="uniform")
    genes = np.asarray(genes, dtype=float)

    def stress_at(g):
        return WO.t3_terms(g, CFG, phases=phases,
                           meshes=WO.phase_meshes(g, CFG, phases))["values"]["stress"]

    out = WO.t3_terms(genes, CFG, phases=phases,
                      meshes=WO.phase_meshes(genes, CFG, phases))
    assert out["values"]["stress"] > 0.0, (
        "the barrier is still flat even at a 2 MPa allowable, so this test is asserting "
        "0 == 0 and would pass with any product rule at all")
    grad = out["grads"]["stress"]

    for gid in (8, 11, 13):
        best = min(
            abs((stress_at(_bump(genes, gid, +h)) - stress_at(_bump(genes, gid, -h)))
                / (2.0 * h) - grad[gid]) / max(abs(grad[gid]), 1e-12)
            for h in (1e-3, 1e-4, 1e-5))
        assert best < 1e-4, (
            f"d(stress)/d({wg.GENE_NAMES[gid]}) is out by {best:.2e} on its whole FD "
            f"ladder — the product rule dKt*agg + Kt*dagg is wrong")


def _bump(genes, gid, h):
    g = np.array(genes, dtype=float)
    g[gid] += h
    return g


def test_the_phase_stencil_is_a_fixed_lattice(genes):
    """RQMC's offset is quantized so `coord_fn`'s cache can hit.

    A continuously-random offset misses on every phase of every step and pays M7's
    measured 0.774 s re-trace eight times per step, which is roughly double the actual
    solving.  Every draw must land on the `n_phase * n_sub` grid.
    """
    rng = np.random.default_rng(0)
    grid = np.arange(8 * 8) * (WO.SECTOR_DEG / 64)
    for _ in range(20):
        ph = WO.phase_stencil(n_phase=8, n_sub=8, scheme="rqmc", rng=rng)
        assert len(ph) == 8
        assert np.abs(ph[:, None] - grid[None, :]).min(axis=1).max() < 1e-12
    assert WO.phase_stencil(n_phase=8, scheme="uniform")[0] == 0.0
    with pytest.raises(ValueError, match="unknown phase scheme"):
        WO.phase_stencil(scheme="sobol")


def test_the_normalized_and_physical_gradients_are_one_chain_rule_apart(genes):
    """Stage 3 works in the unit box; `cy` spans 64 mm and `R_rim` 2.5, so a single
    learning rate in physical units is meaningless."""
    low, high, rng = wg.bounds_arrays(W.GENE_SPACE)
    _, gp, _ = WO.objective(genes, CFG, tiers=("t1", "t2"))
    z = wg.normalize(genes, low, high)
    _, gz, _ = WO.objective(z, CFG, tiers=("t1", "t2"), normalized=True)
    assert np.allclose(gz, gp * rng, rtol=1e-12, atol=0.0)
