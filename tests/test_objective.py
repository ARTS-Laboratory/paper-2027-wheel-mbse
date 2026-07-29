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


def test_the_fillet_term_is_the_only_gradient_R_hub_has(genes):
    """M7 proved `R_hub`/`R_rim` are dead at the mesh: `dcoords/dgene` is exactly zero.

    So any gradient they have comes from gene-space geometry, and `fillet_feasibility` is
    where it comes from.  Asserted both ways, the way M7's census is.
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
        "R_hub lost its only gradient; d(hub margin)/dR_hub should be about 2")


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


def test_kt_is_differentiable_and_only_the_four_genes_it_depends_on_move_it(genes):
    """`dKt/dg` reaches `R_hub`/`t0` and `R_rim`/`t3`, and nothing else.

    The pairing is asserted in `wheel_fea` and the exporter; this asserts the GRADIENT
    respects it, which is the part that would silently misroute.  A `Kt_hub` that moved
    with `R_rim` would still report the right value and would push the optimizer to round
    the wrong corner.

    The signs are physics: a bigger fillet relieves the concentration, a thicker section
    sharpens it.
    """
    (kt_hub, d_hub), (kt_rim, d_rim) = WO.junction_kt(genes)

    assert kt_hub == pytest.approx(W.stress_concentration_kt(genes[12], genes[8]))
    assert kt_rim == pytest.approx(W.stress_concentration_kt(genes[13], genes[11]))

    for name, d, (i_r, i_t) in (("hub", d_hub, (12, 8)), ("rim", d_rim, (13, 11))):
        assert d[i_r] < 0.0, f"dKt_{name}/dR >= 0: a bigger fillet must relieve, not raise"
        assert d[i_t] > 0.0, f"dKt_{name}/dt <= 0: a thicker section must raise, not relieve"
        moved = set(np.nonzero(d)[0].tolist())
        assert moved == {i_r, i_t}, (
            f"Kt_{name} has a gradient in {sorted(moved - {i_r, i_t})}, which it does not "
            f"depend on — the junction pairing is misrouted")


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


def test_the_stress_gradient_obeys_the_product_rule(genes, monkeypatch):
    """`d(Kt*agg) = dKt*agg + Kt*dagg`, finite-differenced with the barrier FORCED ACTIVE.

    THIS IS THE CHECK THAT M8b-i.6 step 2 LIVES OR DIES ON, and it has to be made here
    rather than left to gate 7.  The new constraint puts the shipped design at utilisation
    ~0.39, so `soft_barrier` is flat, `stress` and `d_stress` are both exactly zero, and a
    completely wrong product rule would pass every end-to-end gate in the tree.  So the
    allowable is dropped until the barrier is on the quadratic branch, which changes
    nothing about the arithmetic being tested.

    Genes 12 and 13 reach the stress term ONLY through `Kt`, so they isolate `dKt*agg`;
    genes 8 and 11 reach it through both factors, so they are the ones that fail if the
    two contributions are added wrongly.
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

    for gid in (8, 11, 12, 13):
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
