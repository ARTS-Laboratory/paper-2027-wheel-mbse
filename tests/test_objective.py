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

import math
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
import wheel_stage3 as S3  # noqa: E402
import wheel_wheel as WW  # noqa: E402

CFG = "smoke"
# Two, not one: the stress term aggregates ACROSS phases before it is compared to the
# allowable, so a single-phase stencil cannot tell a per-phase bug from an aggregation one.
N_PHASE = 2


@pytest.fixture(scope="module")
def genes():
    return so.load_genes()


@pytest.fixture(scope="module")
def genes_over_knee():
    """A design whose hub utilisation is ABOVE `MARGIN_KNEE_UTIL`, and its breakdown.

    Returns `(grad, breakdown)` from one `coarse`, 8-phase evaluation, module-scoped so the
    solve is paid once.

    IT IS THE PREDECESSOR GENOME, NOT A CONSTRUCTED ONE, and that is the cheap choice
    rather than the lazy one: `stage3_buildcap2_slack_medium.json` is `e4219f3`, the wheel
    that shipped before 2026-08-13, and it reads util 0.85506 at `coarse`/8 against the
    current genome's 0.70898 — §19 promoted a design that bought 22 points of hub margin,
    which is exactly what puts the old one above the knee and the new one below it.
    Constructing a thin genome instead would need thickness genes under the 1.2 mm
    `MIN_WALL_MM` floor, i.e. a design the gene box forbids, which is a worse fixture than
    a wheel this project actually shipped.

    `coarse` and 8 phases rather than the module's `smoke`/2 because no genome here reaches
    0.80 at those settings — see `test_but_above_the_knee_the_fillet_radii_are_live`.
    """
    import json
    path = os.path.join(REPO, "stage3_buildcap2_slack_medium.json")
    g = wg.genes_to_vector(json.load(open(path))["genes"])
    _, grad, brk = WO.objective(
        g, "coarse", phases=WO.phase_stencil(n_phase=8, scheme="uniform"))
    return grad, brk


@pytest.fixture(scope="module")
def genes_over_cap():
    """A genome whose `R_hub` sits ABOVE its hub-fillet cap, which several tests below
    need and which the shipped genome no longer is.

    The hub fillet cap only does anything to a design that asks for more radius than the
    inter-spoke slot can hold.  That used to be true of `best_solution.json` — 0.45 mm
    over, ~2.7 rungs — so those tests simply took the shipped genome and the coupling was
    invisible.  PLAN.md §13 promoted a genome with `R_hub` = 0.579 against a cap of 0.624,
    i.e. UNDER it, and every one of those tests went vacuous or false at once.

    `best_solution_ga_beam.json` is the same genome they were written against, pinned
    under a name that never moves (see `tests/test_golden.py`).  Using it here keeps these
    tests about THE CAP rather than about which design happens to ship.  The measured
    9.907 deg void in `test_the_hub_cap_reproduces_the_measured_void` was taken on THIS
    genome's built solid, so the pairing is not merely convenient, it is required.
    """
    return so.load_genes("best_solution_ga_beam.json")


@pytest.fixture(scope="module")
def genes_under_cap(genes):
    """The shipped genome with `R_hub` pulled clear UNDER its cap, and the reason it has
    to be constructed rather than found.

    A barrier's whole job is to be flat where the constraint is satisfied, so the tests
    below need a satisfied design as much as they need a violating one.  That used to be
    the shipped genome: PLAN.md §13's `R_hub` = 0.578951 sat 0.045 mm under a cap of
    0.6240, and the two tests that assert `fillet_cap` reads exactly 0.0 simply took it.

    BUILD_PLAN.md step 3 moved the cap to 0.5724 at that genome — 1.1% BELOW the same
    `R_hub`, because the old cap was fitted against the corner family OCC does not stop at
    and over-promised by 6.7% here.  So the shipped genome is now on the violating side and
    those assertions were false.  Pinning them to a genome that is over its cap by
    construction keeps them about THE BARRIER rather than about which design ships, which
    is the same argument `genes_over_cap` above makes from the other direction.

    0.75x the cap, not 0.99x: `hub_fillet_r_effective`'s `smooth_min` has a blend of width
    `CAP_BLEND_FRAC * cap` around the knee, and a fixture that lands inside it would be
    testing the blend instead of the barrier.
    """
    cfgo = WW.get_config(CFG)
    g = np.asarray(genes, dtype=float).copy()
    flanks = WO.fillet_flanks(g, cfgo)
    g[12] = 0.75 * float(WO.hub_fillet_cap_mm(g, cfgo, W.S, W.HUB_RADIUS_MM, flanks))
    return g


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


def test_R_hub_is_dead_at_the_mesh_and_alive_only_in_gene_space(genes, genes_over_cap,
                                                                genes_under_cap):
    """M7 proved `R_hub`/`R_rim` are dead at the mesh: `dcoords/dgene` is exactly zero.

    So any gradient they have comes from gene-space geometry.  THIS TEST USED TO BE CALLED
    "the fillet term is the only gradient R_hub has", and that title was measured false:
    at the GA/beam genome the fillet margins are `[+4.647, +0.125]`, both feasible, so the
    `fillet` barrier is FLAT and contributes exactly 0.0.  `R_hub`'s live loss gradient is
    `hub_overlap`'s (+645.8) and now also `fillet_cap`'s (+454.0).  `_fillet_margins` still
    gives it a gradient in the MARGIN, which is what the middle assertion here checks and is
    a different claim from "in the loss".

    THE LAST TWO BLOCKS TAKE THEIR OWN GENOMES, THE REST TAKES THE SHIPPED ONE.  Mesh
    insensitivity and the margin gradient hold for any design; `d(fillet_cap)/dR_hub` is a
    barrier gradient, so it needs one design on each side of the constraint and neither
    side may be "whatever ships".  Running the whole test on the reference genome would
    have been the lazy fix and would have stopped checking M7's claim on the part that
    actually ships.
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

    # And the term that actually prices it in the loss — on a design that is OVER its cap,
    # because a barrier's whole job is to be flat when the constraint is satisfied.
    cfg_oc = WW.get_config(CFG)
    flanks_oc = WO.fillet_flanks(genes_over_cap, cfg_oc)
    _, jt1 = WO._t1_cached_value_and_jacobian(
        jnp.asarray(genes_over_cap), cfg_oc, None, W.S, flanks_oc)
    row = np.asarray(jt1)[WO.T1_NAMES.index("fillet_cap")]
    assert row[12] > 0.0, (
        f"d(fillet_cap)/dR_hub = {row[12]}, so nothing pushes R_hub back under the slot "
        f"it has to fit in")

    # And on a design that IS under its cap the same gradient must be exactly zero.
    # Asserted rather than assumed: a barrier that leaks a gradient into a satisfied
    # constraint is a silent bias on every descent that starts feasible.
    #
    # This used to take the shipped genome, which was 0.045 mm under its cap.  It is now
    # 0.0066 mm OVER it — BUILD_PLAN.md step 3 re-fitted the cap against the corner family
    # the exporter actually stops at — so the satisfied side has to be constructed.  See
    # `genes_under_cap`.
    flanks_uc = WO.fillet_flanks(genes_under_cap, cfgo)
    _, jt1_under = WO._t1_cached_value_and_jacobian(
        jnp.asarray(genes_under_cap), cfgo, None, W.S, flanks_uc)
    assert np.asarray(jt1_under)[WO.T1_NAMES.index("fillet_cap")][12] == 0.0, (
        "a genome under its hub cap must get no `R_hub` gradient from `fillet_cap` at all")


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


def test_the_hub_cap_reproduces_the_measured_void(genes_over_cap):
    """The analytic slot against the one measured on the BUILT SOLID.

    The hub fillet milestone classified material on sample rings and found the void
    between adjacent spoke roots to be 9.907 deg = 2.196 mm of arc.  This function gets
    there from the genome alone, with no CAD kernel: 9.977 deg, 0.070 deg apart.  The
    difference is `_embed`'s radial plunge starting from the centerline endpoint rather
    than the flank endpoint — see `wheel_wheel.hub_void_deg`.

    0.25 deg is one sample of the OCC classifier `make hubcap` gates this against, i.e.
    the resolution of the instrument, not a tolerance picked to pass.

    IT MUST USE THE GENOME THE 9.907 WAS MEASURED ON, which is `genes_over_cap`.  The void
    is a property of a design, not a constant — §13's thinner spokes open it to 22.8 deg —
    so pointing this at whatever ships turns a CAD-versus-analytic agreement check into a
    comparison between two different wheels.  Re-measuring on the new solid would need a
    `make hubcap` run in the CAD env; until someone does that, the honest pairing is this
    one.
    """
    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes_over_cap, cfgo)
    void = float(WW.hub_void_deg(genes_over_cap, cfgo, W.S, W.HUB_RADIUS_MM, flanks[0]))
    assert abs(void - 9.907) <= 0.25, (
        f"void {void:.4f} deg against 9.907 measured on the solid")

    # TWO LIMITS, and the cap is the smaller.  See `HUB_CAP_SHARE` for why it is a `min`
    # and which of the two `make hubcap` has actually observed binding.
    a_hub = float(WW.arrival_angles(genes_over_cap, cfgo, span_mm=W.S)[0])
    by_slot = WO.HUB_CAP_SHARE * W.HUB_RADIUS_MM * np.radians(void)
    by_thickness = genes_over_cap[8] * (WO.HUB_CAP_THICKNESS_SHARE
                                        - WO.HUB_CAP_ARRIVAL_SLOPE
                                        * (1.0 - np.cos(np.radians(a_hub))))
    cap = float(WO.hub_fillet_cap_mm(genes_over_cap, cfgo, W.S, W.HUB_RADIUS_MM, flanks))
    assert cap == pytest.approx(min(by_slot, by_thickness))
    # At the GA/beam genome the SLOT is the smaller of the two, so this design exercises
    # that branch of the min.  §13's shipped genome takes the OTHER branch, which is why
    # the two designs between them cover both, and why this assertion belongs to this
    # genome specifically.
    assert by_slot < by_thickness, (
        f"slot {by_slot:.4f} vs thickness {by_thickness:.4f}: the reference genome no "
        f"longer exercises the slot branch, so the min is untested here")
    # `make hubcap` bisected the real per-corner threshold at this design to 1.300 mm.  The
    # cap must stay UNDER it: over-promising is the defect this exists to remove, and
    # under-promising only leaves fillet on the table.
    assert cap <= 1.300, f"cap {cap:.5f} over-promises against the measured 1.300"


def test_the_thickness_branch_of_the_cap_binds_on_a_thin_root(genes):
    """The other half of the `min`, and it is the half `make hubcap` observes binding.

    The slot binds on the GA/beam genome, so without this the thickness term could be
    deleted and every other test would still pass.  Thinning `t0` drives the thickness
    limit down (and, incidentally, WIDENS the slot, since a thinner root leaves more gap) —
    so the two limits move in opposite directions and the crossover is reachable inside
    the gene box rather than hypothetical.

    §13's shipped genome is on the OTHER side of that crossover: at `t0` = 1.2 the
    thickness branch takes the `min`, which is the branch this test is about.  The 2.0
    below is now a value chosen to sit near the crossover rather than the floor it used to
    be — `MIN_WALL_MM` is 1.2.

    THE LAST ASSERTION USED TO BE ITS EXACT OPPOSITE, and that is the point of
    BUILD_PLAN.md step 3.  It read `norm(d[:8]) == 0.0` — "the shape genes still move the
    cap on the thickness branch" was the failure message — because the branch was
    `HUB_CAP_THICKNESS_SHARE * t0` and nothing else could reach it.  A cap that cannot see
    where the spoke points is a cap that cannot tell `350f4c7`, which OCC builds 24/24 at
    its own `R_hub`, from `bc77614`, which OCC builds 12/24 at the same radius: the two
    differ by 29 degrees of hub arrival and by nothing else the old cap took.  The shape
    genes MUST move it now, and `test_the_cap_ranks_two_designs_OCC_disagrees_about` below
    is the same claim stated in millimetres against a measurement.
    """
    cfgo = WW.get_config(CFG)
    g = np.asarray(genes, dtype=float).copy()
    g[8] = 2.0                                   # near the slot/thickness crossover
    flanks = WO.fillet_flanks(g, cfgo)
    void = float(WW.hub_void_deg(g, cfgo, W.S, W.HUB_RADIUS_MM, flanks[0]))
    a_hub = float(WW.arrival_angles(g, cfgo, span_mm=W.S)[0])
    by_slot = WO.HUB_CAP_SHARE * W.HUB_RADIUS_MM * np.radians(void)
    by_thickness = g[8] * (WO.HUB_CAP_THICKNESS_SHARE
                           - WO.HUB_CAP_ARRIVAL_SLOPE
                           * (1.0 - np.cos(np.radians(a_hub))))
    cap = float(WO.hub_fillet_cap_mm(g, cfgo, W.S, W.HUB_RADIUS_MM, flanks))

    assert by_thickness < by_slot, (
        f"at t0 = 2.0 the thickness limit {by_thickness:.4f} is still not the smaller "
        f"(slot {by_slot:.4f}) — the crossover is outside the box and the min is dead code")
    assert cap == pytest.approx(by_thickness)
    # And the gradient has to follow the branch that is live.
    d = np.asarray(jax.grad(
        lambda v: WO.hub_fillet_cap_mm(v, cfgo, W.S, W.HUB_RADIUS_MM, flanks)
    )(jnp.asarray(g)))
    assert d[8] == pytest.approx(WO.HUB_CAP_THICKNESS_SHARE
                                 - WO.HUB_CAP_ARRIVAL_SLOPE
                                 * (1.0 - np.cos(np.radians(a_hub))))
    assert abs(d[0]) > 1e-3 and abs(d[1]) > 1e-3, (
        f"the two genes that aim the spoke at the hub do not move the cap on the "
        f"thickness branch: d(cap)/d(cx1, cy1) = {d[0]:.3e}, {d[1]:.3e}")


def test_the_cap_ranks_two_designs_OCC_disagrees_about(genes):
    """THE TEST THIS ARC EXISTS TO ADD.  It fails on every cap this repo shipped before
    2026-08-10, and it fails by returning the SAME NUMBER TWICE.

    `350f4c7` (the shipped genome) and `bc77614` (Stage 3's SVK descent at `medium`) have
    identical `t0` = 1.2, identical `R_hub` = 0.578951, and voids 1.7 deg apart.  At that
    radius OCC fillets all 24 hub corners on the first and only 12 on the second.  The old
    cap returned 0.6240 for both, to sixteen digits, because the branch that took the `min`
    was `HUB_CAP_THICKNESS_SHARE * t0` — a function of a gene the two designs share.  What
    they do not share is the hub ARRIVAL ANGLE: 19.68 deg against 48.89.

    THE TWO GENOMES ARE CONSTRUCTED, NOT LOADED, and that is deliberate.  `bc77614` lives in
    a Stage-3 run artifact, and a test that pins a calibration to a file some future run
    overwrites is a test with a fuse in it.  `wheel_geometry.control_points` locks P0 at the
    origin, so the hub arrival is `asin(|cx1| / hypot(cx1, cy1))` — a function of two genes
    and nothing else — and rotating P1 about the hub at FIXED RADIUS walks it while moving
    no other gene.  That is the same one-variable control `studies/study_arrival_cap.py`
    uses, and the two stations below are two of its measured rows.

    The bounds are that study's bisected OCC thresholds at `t0` = 1.2, not round numbers:
    0.5847 mm at 20 deg and 0.4127 at 50.  The cap has to stay under each (over-promising is
    the defect) and within 10% of it (a cap of zero would satisfy the first clause and
    destroy every hub fillet in the wheel — `study_hub_cap.GATE_CAP_FLOOR_FRAC` makes the
    same argument at 50%).
    """
    cfgo = WW.get_config(CFG)

    def at_arrival(deg):
        g = np.asarray(genes, dtype=float).copy()
        r = math.hypot(g[0], g[1])
        g[0] = r * math.sin(math.radians(deg))
        g[1] = r * math.cos(math.radians(deg))
        # t0 IS PART OF THE CONSTRUCTION, not something to inherit from the fixture.  The
        # docstring's whole argument is that a calibration pinned to a file a future run
        # overwrites is a test with a fuse in it -- and the first version of this test left
        # exactly that fuse, by constructing the arrival angle but taking t0 from whatever
        # `best_solution.json` happened to hold.  It went off on 2026-08-11: the promoted
        # genome came off the wall floor by 0.07% (t0 = 1.20084257) and the guard below
        # fired, correctly, because the OCC thresholds are only valid at 1.2.  The by_thickness
        # branch is LINEAR in t0, so inheriting it would silently rescale the very quantity
        # being compared.  Pin it, and the guard becomes the assertion it was meant to be.
        g[8] = 1.2
        return g

    OCC_MM = {20.0: 0.5847, 50.0: 0.4127}       # studies/study_arrival_cap.json, t0 = 1.2
    caps = {}
    for deg, occ in OCC_MM.items():
        g = at_arrival(deg)
        assert float(g[8]) == 1.2, "these thresholds were measured at t0 = 1.2"
        flanks = WO.fillet_flanks(g, cfgo)
        cap = float(WO.hub_fillet_cap_mm(g, cfgo, W.S, W.HUB_RADIUS_MM, flanks))
        caps[deg] = cap
        assert cap <= occ, (
            f"at {deg:.0f} deg of hub arrival the cap promises {cap:.4f} mm and OCC "
            f"bisects at {occ:.4f} — the cap over-promises, which is the whole defect")
        assert cap >= 0.90 * occ, (
            f"at {deg:.0f} deg the cap gives away {(1 - cap / occ):.1%} of a fillet the "
            f"part will build; conservative is not the same as vacuous")

    assert caps[20.0] > caps[50.0], (
        f"the cap cannot tell a 20 deg arrival from a 50 deg one — {caps[20.0]:.6f} vs "
        f"{caps[50.0]:.6f}.  OCC can: 24 corners against 12 at the same R_hub")
    assert caps[20.0] / caps[50.0] > 1.3, (
        f"the cap ranks them the right way but only by "
        f"{caps[20.0] / caps[50.0]:.3f}x; OCC measures 1.417x")


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


def test_smooth_min_is_differentiable_at_the_tie_itself():
    """The one point the blend was written for, and the one point it got wrong.

    `smooth_min` is smooth THROUGH `a == b` — that is the entire reason it exists instead
    of `min` — but at exactly `a == b` it used to report a derivative of 1.0 against a true
    two-sided 0.5. Two primitives are non-differentiable at the tie and autodiff picked a
    subgradient for each: `jnp.abs` returns 0 at 0, dropping the blend term, and
    `jnp.minimum` hands the full 1.0 to its first argument. Nothing in the VALUE was ever
    wrong, which is why a value-only test could not see it.

    Measure-zero and never observed to bite — but §17's margin term drives `R_hub` at its
    cap deliberately, so the tie is now an attractor rather than an accident, and a
    gradient that doubles exactly at the attractor is worth a test.
    """
    k = 0.15
    at_tie = float(jax.grad(lambda a: WO.smooth_min(a, 1.0, k))(1.0))
    assert at_tie == pytest.approx(0.5, abs=1e-9), (
        f"d/da smooth_min(a, a) is {at_tie}, not 0.5 — the tie subgradients are back")

    # Continuous INTO the tie from both sides, which is the property the value-only test
    # cannot check and the reason 0.5 is the right answer rather than a convention.
    for eps in (1e-5, 1e-7):
        lo = float(jax.grad(lambda a: WO.smooth_min(a, 1.0, k))(1.0 - eps))
        hi = float(jax.grad(lambda a: WO.smooth_min(a, 1.0, k))(1.0 + eps))
        assert lo == pytest.approx(0.5, abs=1e-4) and hi == pytest.approx(0.5, abs=1e-4)

    # And the fix must not have cost the exactness it is fenced off from: outside the
    # blend, still `min` to the bit, in BOTH directions.
    assert float(WO.smooth_min(5.0, 1.0, k)) == 1.0
    assert float(WO.smooth_min(1.0, 5.0, k)) == 1.0


def test_R_eff_is_exactly_the_cap_more_than_one_rung_above_it(genes_over_cap):
    """WHY THE POLYNOMIAL smooth-min AND NOT THE SQRT ONE.

    Outside the blend the result is `min(a, b)` TO THE BIT, so a design a rung above its
    cap is priced on exactly the cap with exactly zero derivative in `R_hub` — an assertion
    rather than an approximation.  The sqrt form would put `R_eff` 1.3% below the cap here
    and invent a fillet reduction nothing measured.

    "MORE THAN ONE RUNG ABOVE IT" IS THE PRECONDITION, and it is now carried by the fixture
    rather than by whatever ships.  §13's genome sits 0.045 mm UNDER its cap, inside the
    blend, where `R_eff` is deliberately NOT the hard min — see
    `test_the_shipped_genome_is_inside_the_blend_and_is_priced_conservatively` below, which
    is the other half of this and did not exist until a design landed there.
    """
    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes_over_cap, cfgo)
    cap = WO.hub_fillet_cap_mm(genes_over_cap, cfgo, W.S, W.HUB_RADIUS_MM, flanks)
    gj = jnp.asarray(genes_over_cap)

    # This genome sits ~2.7 rungs above its cap.
    assert float(WO.hub_fillet_r_effective(gj, cap)) == float(cap)
    (_, d_hub), _ = WO.junction_kt(genes_over_cap, cfgo, flanks=flanks)
    assert d_hub[12] == 0.0, (
        f"d(Kt_hub)/dR_hub = {d_hub[12]}, but R_hub is above the slot — buying more of it "
        f"must buy exactly nothing")

    # Well below the cap, nothing is capped and the original physics is back.
    g2 = np.asarray(genes_over_cap).copy()
    g2[12] = 0.5 * float(cap)
    assert float(WO.hub_fillet_r_effective(jnp.asarray(g2), cap)) == g2[12]
    (_, d2), _ = WO.junction_kt(g2, cfgo, flanks=flanks)
    assert d2[12] < 0.0, "below the cap a bigger fillet must still lower Kt"


def test_the_shipped_genome_is_inside_the_blend_and_is_priced_conservatively(genes):
    """§13's genome landed in the smooth-min's blend, where nothing had landed before.

    It landed there from ABOVE on 2026-08-10, having landed there from below in §13, and
    the invariant this test exists for is the same either way: `hub_fillet_r_effective`
    must never price the hub SOFTER than the part is built.

      §13 (cap 0.6240):    R_hub 0.578951 sat 0.045 mm UNDER the cap but inside the blend,
                           so `R_eff` came out 0.5727 — about 1.1% below the requested
                           radius, and below what OCC builds.
      BUILD_PLAN step 3
      (cap 0.5720):        the same `R_hub` is now 1.2% OVER the cap.  `R_eff` is 0.5539,
                           4.3% under the requested radius and 5.3% under the 0.5847 the
                           bisection measures OCC accepting at this design.

    The direction is what makes this tolerable rather than a bug: `smooth_min` can only
    pull `R_eff` DOWN, so the optimizer sees a sharper corner than the part has and the
    constraint is conservative.  This test pins the direction and the magnitude so that a
    future change to the blend, or to the cap, cannot silently make it optimistic.

    Worth stating plainly, because PLAN.md §11 and §13 both headline `kt_error_pct` =
    +0.0%: that number is the EXPORTER comparing its own modelled Kt against its own built
    Kt, and it is correct.  The objective prices the same junction on the blended radius
    instead, and now at Kt 2.0533 against the exporter's 2.0235.
    """
    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes, cfgo)
    cap = float(WO.hub_fillet_cap_mm(genes, cfgo, W.S, W.HUB_RADIUS_MM, flanks))
    r_hub = float(genes[12])
    r_eff = float(WO.hub_fillet_r_effective(jnp.asarray(genes), jnp.asarray(cap)))

    k = WO.CAP_BLEND_FRAC * max(cap, WO.MIN_BUILDABLE_R_MM)
    assert abs(r_hub - cap) < k, (
        f"R_hub {r_hub:.6f} is no longer within one blend width ({k:.6f}) of its cap "
        f"{cap:.6f} — this genome has left the blend and the test below is vacuous")
    assert r_eff < min(r_hub, cap), (
        f"R_eff {r_eff:.6f} is not below both R_hub {r_hub:.6f} and the cap {cap:.6f} — "
        f"the blend has stopped blending, or it has started rounding UP")
    assert r_hub - r_eff < 0.05 * r_hub, (
        f"the blend is pulling R_eff {(1 - r_eff / r_hub):.2%} below the requested "
        f"radius; it was 1.1% at §13's genome and 4.3% once the cap was re-fitted, and a "
        f"much larger reduction is a fillet penalty nothing has measured")

    kt_priced, _ = WO.junction_kt(genes, cfgo, flanks=flanks)[0]
    assert float(kt_priced) > float(W.stress_concentration_kt(r_hub, genes[8])), (
        "the constraint must price the hub at least as sharp as the part is built — a "
        "blend that made Kt optimistic would understate utilisation on every design "
        "that lands near its cap")


def test_the_cap_gradient_matches_a_finite_difference(genes, genes_over_cap):
    """Solve-free, so it can afford to be tight.  ONE BLOCK PER BRANCH OF THE `min`.

    This is where the arithmetic gene 12 used to carry in
    `test_the_stress_gradient_obeys_the_product_rule` now lives: `R_hub` is exactly inert
    above its cap, so its row there became vacuous.

    IT WAS ITSELF FOUR-FIFTHS VACUOUS, and BUILD_PLAN.md step 3 is what exposed that.  It
    checked genes (0, 1, 4, 7, 8) on the shipped genome, where the THICKNESS branch takes
    the `min` — and the old thickness branch was `HUB_CAP_THICKNESS_SHARE * t0`, a function
    of gene 8 and nothing else.  Genes 0, 1, 4 and 7 were all exactly zero, both sides, and
    `max(abs(d[i]), 1e-12)` made the comparison 0.0 < 1e-6.  Four genes passed by having
    nothing to say.

    Both branches are checked now, on a genome that takes each:

      THICKNESS/ARRIVAL, on the shipped genome.  Its three real routes are `t0` and the two
      genes that set the hub arrival angle, so those three are what is asserted.  Genes 2-7
      DO reach the cap now — `global_sampler` re-parameterises the whole curve, so the
      arrival leaks a little into every control point — but at 1e-4 of the gradient norm
      and below, which a central difference cannot resolve.  See
      `study_objective.GATE_FD_LIVE_FRAC`; asserting on them would be measuring round-off.

      SLOT, on `genes_over_cap`.  There the void takes the `min` and every one of genes 0-8
      carries a resolvable gradient, so all nine are asserted.
    """
    cfgo = WW.get_config(CFG)

    def check(g, ids, label):
        flanks = WO.fillet_flanks(g, cfgo)

        def cap(v):
            return float(WO.hub_fillet_cap_mm(jnp.asarray(v), cfgo, W.S, W.HUB_RADIUS_MM,
                                              flanks))

        d = np.asarray(jax.grad(
            lambda v: WO.hub_fillet_cap_mm(v, cfgo, W.S, W.HUB_RADIUS_MM, flanks)
        )(jnp.asarray(g)))
        for i in ids:
            assert abs(d[i]) > 0.0, f"{label} gene {i}: the cap has no gradient here at all"
            best = min(
                abs((cap(_bump(g, i, +h)) - cap(_bump(g, i, -h))) / (2 * h) - d[i])
                / abs(d[i])
                for h in (1e-4, 1e-5, 1e-6))
            assert best < 1e-6, f"{label} gene {i}: FD vs adjoint rel error {best:.2e}"

    check(genes, (0, 1, 8), "thickness/arrival branch")
    check(genes_over_cap, tuple(range(9)), "slot branch")


def test_the_t1_term_lists_stay_in_lockstep(genes):
    """Five lists have to agree, and adding the seventh term touched all of them."""
    assert WO.T1_NAMES == WO._T1_WEIGHT_KEYS
    assert set(WO.T1_NAMES) <= set(WO.TERMS)
    assert set(WO.TERMS) <= set(WO.DEFAULT_WEIGHTS)
    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes, cfgo)
    v = WO.t1_vector(jnp.asarray(genes), cfgo, None, W.S, flanks)
    assert len(v) == len(WO.T1_NAMES)


def test_the_fillet_radii_are_not_dead_genes(genes):
    """§15 defect 2, as the regression that would catch it coming back.

    `R_hub` and `R_rim` reach the loss only through `Kt`, and `Kt` reached it only through
    `soft_barrier(util - 1)`, which is identically flat below the knee. So the two genes
    had EXACTLY zero gradient — not small, zero — and a nominally 14-dimensional search ran
    in 8. Over 602 descent steps `R_rim` moved on 0 and `R_hub` on 2, both times only
    because `fillet_cap` was live.

    The assertion is on the SIGN as well as the magnitude, because the magnitude alone
    would pass on a term that priced fillets backwards. More fillet means a lower stress
    concentration, so more fillet must mean less loss: both derivatives are negative, and
    the optimizer's first move on these genes must be to open them up.

    THE KNEE PUT THE GENES BACK TO SLEEP BELOW 0.80, AND THAT IS DELIBERATE (defect 8).
    This test used to ask the question at the shipped genome. `stress_margin` is now
    `soft_barrier(util - MARGIN_KNEE_UTIL)`, so below the knee it is identically flat and
    the fillet genes are once again exactly dead there — which reads like §15 defect 2
    returning and is not. The difference is WHERE:

        defect 2   the genes were flat everywhere below util = 1.0, so they were dead at
                   every design anyone would ship, and the search really did run in 8
                   dimensions.
        the knee   they are flat below util = 0.80, which is the region where the project
                   has decided more fillet is worth nothing (see MARGIN_KNEE_UTIL). They
                   are live exactly where margin is worth buying.

    So the test asks it in both places now. Below the knee, flat is CORRECT and is asserted
    as such — an accidental return to a live-everywhere term would fail here. Above the
    knee, the original claim is unchanged and is what stops defect 2 coming back.
    """
    _, g, brk = WO.objective(genes, CFG, phases=WO.phase_stencil(n_phase=N_PHASE, scheme="uniform"))
    assert brk["report"]["stress_utilisation_hub"] < WO.MARGIN_KNEE_UTIL, (
        "the shipped genome has climbed above the margin knee, so this half of the test is "
        "no longer measuring the below-knee region it was written for")
    assert brk["terms"]["stress_margin"]["value"] == 0.0, (
        "the margin term is live below its own knee — the knee is what makes the policy "
        "'margin below 0.80 is worth nothing' true in the code")
    assert g[12] == 0.0 and g[13] == 0.0, (
        f"dL/dR_hub {g[12]:+.3e}, dL/dR_rim {g[13]:+.3e} — nonzero below the knee, where "
        f"the only thing that prices the fillets should be flat")


def test_but_above_the_knee_the_fillet_radii_are_live(genes_over_knee):
    """The half of defect 2's regression that still has to hold, asked where it applies.

    ONE COARSE EVALUATION, and it is the reason this is a separate test with its own
    fixture: no genome in this repo reaches util 0.80 on the `smoke`/2-phase settings the
    rest of the module runs at, because two uniform phases sample a 30 degree period at 0
    and 15 degrees and read the utilisation ~20% low (DEFECT8_PLAN.md step 1b). Asking an
    above-knee question at a fidelity that cannot produce an above-knee design would pin
    nothing, which is the fuse `test_the_fillet_cap_barrier_is_live_on_a_design_over_its_
    cap` already carries one screen down.

    THE KNEE IS PER-JUNCTION, AND THE FIRST VERSION OF THIS TEST DID NOT NOTICE.
    It asserted both fillet radii carry gradient on a design "above the knee", which
    conflated two junctions with very different utilisations. Measured on this fixture:

        hub  util 0.85506   ABOVE the 0.80 knee   ->  dL/dR_hub < 0, live
        rim  util 0.47963   far BELOW it          ->  dL/dR_rim = 0.0 exactly

    So `R_rim` IS a dead gene under the knee, at every design in this repo — the rim has
    never come close to 0.80 on any genome measured. That is the intended reading rather
    than a regression, and PLAN.md §22 is the independent measurement of it: raising
    `R_rim`'s box ceiling was worth -0.84 of loss under the OLD `util**2` term, and every
    one of the objective's checks on that radius is blind (the mesh models no fillets, so
    mass cannot see it; the FEA cannot see it; the rim has no buildability cap). The knee
    prices that at zero, which is what §22 says it is worth.
    """
    g, brk = genes_over_knee
    u_hub = brk["report"]["stress_utilisation_hub"]
    u_rim = brk["report"]["stress_utilisation_rim"]
    assert u_hub > WO.MARGIN_KNEE_UTIL, u_hub
    assert brk["terms"]["stress_margin"]["value"] > 0.0, (
        "the margin term is flat above its own knee, so it prices nothing where margin is "
        "supposed to be worth buying")
    assert g[12] < 0.0, f"dL/dR_hub is {g[12]:+.3e} — the hub fillet is dead above the knee"

    # The rim, asked against its OWN utilisation rather than the hub's.
    assert u_rim < WO.MARGIN_KNEE_UTIL, (
        f"the rim has reached util {u_rim:.5f}, above the knee — the branch below is no "
        f"longer the one this fixture exercises and the rim gene should now be live")
    assert g[13] == 0.0, (
        f"dL/dR_rim is {g[13]:+.3e} — nonzero while the rim sits at util {u_rim:.5f}, far "
        f"under the knee, which is precisely the free margin PLAN.md §22 measured as worth "
        f"nothing the part pays for")


def test_the_margin_term_prices_and_never_gates(genes):
    """It is an OBJECTIVE, and the distinction is the whole design of it.

    `stress` stays exactly as it was: the wall is still there and still decides
    shippability. `stress_margin` only stops the approach to that wall being free. If it
    ever landed in `BARRIER_TERMS` it would start vetoing promotion candidates for having
    any stress at all, which every real design does — `selection_key` would return tier 2
    on the shipped wheel.
    """
    assert "stress_margin" in WO.OBJECTIVE_TERMS
    assert "stress_margin" not in WO.BARRIER_TERMS
    assert "stress" in WO.BARRIER_TERMS, "the barrier must survive its own successor"
    _, _, brk = WO.objective(genes, CFG, phases=WO.phase_stencil(n_phase=N_PHASE, scheme="uniform"))
    assert brk["terms"]["stress"]["value"] == 0.0, (
        "the shipped genome is under the allowable, so the BARRIER must still read zero — "
        "if it does not, this test is measuring a violation and not the split")
    assert S3.selection_key(brk["total"], brk, genes)[0] == 0, (
        "a live margin term made the shipped genome unpromotable")


def test_the_margin_weight_is_the_exchange_rate_it_claims_to_be(genes):
    """The weight is a POLICY — 1% of utilisation against 1% of mass — so it is pinned
    against the thing it trades with rather than left as a number in a table.

    A weight that silently drifts off its stated rate is worse than an unstated one: the
    comment would go on claiming a calibration the code no longer has. Checked as an order
    of magnitude, not to the digit, because the rate is exact only at the design it was
    derived at and both terms move.

    RE-DERIVED FOR THE KNEE, AND ASKED AT THE REFERENCE RATHER THAN AT THE DESIGN.
    Two things about the old version were wrong once defect 8 was measured:

    1. It computed the cost as `w * (1.01**2 - 1) * util**2`, which is the OLD shape. The
       term is now `soft_barrier(util - MARGIN_KNEE_UTIL)`, so the cost of 1% of
       utilisation is the difference of two clipped squares.
    2. It read `util` off the shipped genome at `smoke`/2 phases — 0.560, against 0.780 at
       the production `medium`/8. That is what turned a policy check into a fidelity check:
       measured on both genomes, `e4219f3` PASSED at every setting while `e126cc3` failed at
       n_phase=2 and passed at n_phase=8, so the gate was discriminating designs AND
       fidelities at once (DEFECT8_PLAN.md step 1b).

    The rate is a property of the WEIGHT and the KNEE, not of whichever genome happens to
    be in the tree, so it is now evaluated at the reference utilisation the weight was
    calibrated at. That is §18's own 0.855, kept deliberately so this change is a change of
    SHAPE at a fixed rate. The mass term still comes from the shipped genome, because it is
    the thing being traded against and it is a real quantity.
    """
    _, _, brk = WO.objective(genes, CFG, phases=WO.phase_stencil(n_phase=N_PHASE, scheme="uniform"))
    u_ref = 0.855
    k = WO.MARGIN_KNEE_UTIL
    one_pct_of_mass = 0.01 * brk["terms"]["mass"]["value"]
    one_pct_of_util = WO.DEFAULT_WEIGHTS["stress_margin"] * (
        max(0.0, 1.01 * u_ref - k) ** 2 - max(0.0, u_ref - k) ** 2)
    assert 0.5 < one_pct_of_util / one_pct_of_mass < 2.0, (
        f"at the reference utilisation {u_ref}, 1% of utilisation costs "
        f"{one_pct_of_util:.4f} against 1% of mass at {one_pct_of_mass:.4f} — the weight "
        f"no longer sets the rate its comment claims")
    # And the knee is the half of the policy the weight cannot express: below it, a
    # percent of utilisation must cost NOTHING, or "worthless below 0.80" is not true.
    below = WO.DEFAULT_WEIGHTS["stress_margin"] * (
        max(0.0, 1.01 * 0.70 - k) ** 2 - max(0.0, 0.70 - k) ** 2)
    assert below == 0.0, f"1% of utilisation at 0.70 costs {below:.4e}, not nothing"


def test_the_fillet_cap_barrier_is_live_on_a_design_over_its_cap(genes_over_cap,
                                                                 genes_under_cap):
    """It has to BITE somewhere, and the GA/beam genome is 0.45 mm over its slot.

    15 of the 16 Stage-2 elites are above their cap, including both production multi-start
    points, so a barrier that read zero on all of them would be measuring nothing.

    THIS USED TO BE CALLED "...AT THE SHIPPED GENOME" and took whichever genome shipped.
    §13 promoted one that is UNDER its cap, at which point the barrier correctly read 0.0
    and the test failed for being satisfied — the constraint working, described as a
    regression.  Both halves are asserted now: live where it must bite, and exactly zero
    where it must not.

    AND NEITHER HALF TAKES THE SHIPPED GENOME ANY MORE.  BUILD_PLAN.md step 3 re-fitted the
    cap and §13's genome crossed to the violating side — 0.578951 against 0.5724 — so a
    test pinned to "whatever ships" would have flipped a second time.  Both sides are
    constructed now, which is the point the first rewrite half-made.
    """
    cfgo = WW.get_config(CFG)
    flanks = WO.fillet_flanks(genes_over_cap, cfgo)
    v, J = WO._t1_cached_value_and_jacobian(
        jnp.asarray(genes_over_cap), cfgo, None, W.S, flanks)
    k = WO.T1_NAMES.index("fillet_cap")
    assert float(np.asarray(v)[k]) > 0.0, "the cap barrier is flat on a design over its cap"
    assert np.asarray(J)[k][12] > 0.0

    flanks_u = WO.fillet_flanks(genes_under_cap, cfgo)
    v_u, _ = WO._t1_cached_value_and_jacobian(
        jnp.asarray(genes_under_cap), cfgo, None, W.S, flanks_u)
    assert float(np.asarray(v_u)[k]) == 0.0, (
        "a genome under its cap must read exactly 0.0 here — a non-zero value would be a "
        "penalty charged to a feasible design")


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


def test_kt_hub_is_priced_on_the_buildable_radius_and_kt_rim_on_the_requested_one(
        genes_over_cap):
    """PLAN.md §0(a): the hub's `Kt` reflects the fillet the PART GETS, not the gene.

    WHAT THIS USED TO SAY.  "only the four genes it depends on move it" — `Kt_hub` was
    `kt(R_hub, t0)` and its gradient was nonzero on exactly `{12, 8}`.  That is no longer
    true and must not be: `R_hub` is capped at half the slot between adjacent spoke roots,
    the slot is a function of the eight centerline genes and `t0`, so `Kt_hub` moves with
    all ten.  This genome sits 0.45 mm above its cap, so `dKt_hub/dR_hub` there is
    exactly 0.0 — the optimizer can no longer buy hub fillet it will not receive.

    IT TAKES `genes_over_cap`, NOT THE SHIPPED GENOME.  Every hub assertion below is the
    OVER-cap branch: `r_eff` is the hard `min`, `dKt_hub/dR_hub` is exactly zero, and Kt is
    strictly raised above what the gene asked for.  None of that is true inside the blend,
    which is where §13's genome sits — that regime has its own test,
    `test_the_shipped_genome_is_inside_the_blend_and_is_priced_conservatively`.

    The RIM is deliberately not capped (its junction has a whole ring band to grow into,
    not a slot), so its half of this test is unchanged and is the control: if the rim's
    row ever starts looking like the hub's, the cap has leaked across the pairing.
    """
    genes = genes_over_cap
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


def test_every_objective_term_is_classified_by_the_svk_rescore_gate():
    """PLAN §25: `make svk` is the promotion gate for feasibility at both fidelities, and
    it had been dead since §18.

    `studies/study_svk_rescore.py` classifies every term explicitly as a BARRIER (must be
    0.0 to be feasible) or a HEADLINE (reported, never gated), and raises on anything it
    does not recognise — deliberately, because an unclassified term would be reported as
    feasible.  §18 added `stress_margin`; the driver predates it; so the gate exited 2 on
    line one for every run from 2026-08-13, and §19's promotion went through without it.

    Nothing noticed because that driver is not part of `make test` and is run only when a
    promotion is pending, which is the worst possible moment to discover it.  This is the
    tripwire: the classification is an invariant between two files, so it belongs here
    rather than in the driver, whose own guard only fires when someone runs it.

    NOT a check that the classification is CORRECT — that is a judgement (`stress_margin`
    is a price, not a wall, and `smoothness` is the cautionary tale in the driver's own
    header).  It checks only that the judgement has been made for every term that exists.
    """
    import study_svk_rescore as S

    classified = set(S.BARRIER_NAMES) | set(S.HEADLINE_NAMES)
    terms = set(WO.TERMS)
    assert not terms - classified, (
        f"objective terms unclassified by study_svk_rescore: {sorted(terms - classified)} "
        "— classify each in BARRIER_NAMES or HEADLINE_NAMES. An unclassified term would be "
        "reported as feasible, and the gate raises rather than guess")
    assert not classified - terms, (
        f"study_svk_rescore classifies terms the objective no longer has: "
        f"{sorted(classified - terms)}")
    assert not set(S.BARRIER_NAMES) & set(S.HEADLINE_NAMES)
