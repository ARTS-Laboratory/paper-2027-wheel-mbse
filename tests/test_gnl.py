"""
M5 verification of geometric nonlinearity and the Newton loop.

The other four gates each have a reduced-fidelity rerun in this suite; M5 did not, so
its numbers lived only in `study_gnl.py`'s docstring and nothing failed when the solver
drifted away from them.  This file closes that gap.

Everything here runs on the `smoke` mesh on purpose.  M5's findings are a 3.95% effect
and an order-of-magnitude spread — neither needs a converged mesh to be visible, and an
SVK wheel solve is expensive enough that a converged one would put minutes into `make
test`.  What is checked at full precision is the identities (rigid rotation, continuation
independence), because those have exact expected answers at any resolution.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import wheel_fem as fem          # noqa: E402
import wheel_genome as wg        # noqa: E402
import wheel_wheel as WW         # noqa: E402
import study_gnl as gnl          # noqa: E402
import study_reds_ratio_stability as RS   # noqa: E402  — the retired `max/min` gate's
#                                         # replacement constants and the grid behind them

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = "smoke"


@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


@pytest.fixture(scope="module")
def mesh(genes):
    return WW.build_wheel(genes, CFG)


# ---------------------------------------------------------------------------
# THE IDENTITIES — exact at any resolution
# ---------------------------------------------------------------------------

def test_rigid_rotation_of_the_whole_wheel_stores_no_energy(genes):
    """Frame indifference, through the assembly rather than through one element.

    `tests/test_fem.py` already checks this identity on a single element.  This is not
    the same test: the rotation here passes through twelve rotated sectors and every
    seam, so a sector wired to the wrong neighbour or a seam declared but not merged
    shows up as stored energy in a body that has not deformed.

    The linear kernel's energy is the scale.  Without it "SVK energy below 1e-9 mJ" says
    nothing — at small angles the linear kernel stores little too, and the ratio is the
    only thing that gives the test a chance to fail.
    """
    rep = gnl.run_frame_indifference(genes, CFG, angles=(1.0, 30.0))
    assert rep["pass"], rep
    assert rep["worst_ratio"] < gnl.GATE_ROTATION_ENERGY_RATIO, rep["rows"]
    # The 30-degree row must be the one with real linear energy in it, or the ratio was
    # small because the denominator was, not because the numerator was.
    big = max(rep["rows"], key=lambda r: r["angle_deg"])
    assert big["linear_energy_mJ"] > 1.0, big


def test_the_load_continuation_path_does_not_change_the_equilibrium(genes):
    """One increment or eight, a conservative problem lands in the same place.

    This is the check that a residual norm cannot make: each increment converges happily
    to its own slightly wrong state, so a genuinely unconverged path looks healthy at
    every step and only the endpoints disagree.
    """
    rep = gnl.run_newton_health(genes, CFG, step_counts=(1, 8))
    assert rep["continuation_spread"] < gnl.GATE_CONTINUATION_REL, rep["continuation"]
    # Quadratic is 2; the gate asks only that it is superlinear, since the first
    # iteration is a transient (the SVK internal force at the linear solution is far
    # from equilibrium, so the residual rises before Newton takes hold).
    assert rep["observed_order"] > gnl.GATE_NEWTON_ORDER, rep["residual_history"]
    # Compared on the whole field, not the axle drop: a warm start that landed on a
    # different equilibrium could easily agree at one node.
    assert rep["warm_start_agrees_rel"] < 1e-8, rep["warm_start_agrees_rel"]
    assert rep["pass"], rep


def test_the_work_identity_separates_the_two_kinematics(genes):
    """2U/(F*delta) computed from the ENERGY, so it cannot be an artefact of the drop.

    Neither value is exactly 1, and that is not a defect: `axle_drop_mm` is the CENTRE
    NODE, while the work-conjugate displacement is the pressure-weighted mean over the
    patch.  That offset is worth ~0.8% here and it applies to both kinematics — which is
    why the useful assertion is not "linear equals 1" but "the two disagree".  Measured
    on the smoke mesh: linear 0.9922, SVK 1.0048.

    `tests/test_wheel_fea.py::test_work_identity` pins the linear side properly, by
    bracketing between the centre node and the patch mean.  This test exists for the
    other question: if the two kinematics gave the SAME ratio, the Newton loop bent no
    load-deflection curve and the nonlinear path is not running.
    """
    rep = gnl.run_energy_identity(genes, CFG)
    lin = rep["linear"]["two_u_over_f_delta"]
    svk = rep["svk"]["two_u_over_f_delta"]
    assert abs(lin - 1.0) < 0.02, (
        f"2U/F*delta = {lin:.6f} linearly — too far from 1 to be the centre-node vs "
        f"work-conjugate offset; the energies and the drop have come apart")
    assert abs(svk - lin) > 5e-3, (
        f"linear {lin:.6f} and SVK {svk:.6f} agree — the load-deflection curve is not "
        f"bending, i.e. the nonlinear path is not doing anything")


# ---------------------------------------------------------------------------
# THE HEADLINE — the off-ramp is closed
# ---------------------------------------------------------------------------

def test_geometric_nonlinearity_exceeds_the_plans_two_percent_off_ramp(genes):
    """M5's first finding: the correction is too big to check only at checkpoints.

    The plan's off-ramp was "if < 2%, run the Stage-3 trajectory on the linear model".
    If this ever passes at under 2%, Stage 3 gets substantially cheaper and the decision
    should be revisited deliberately — so it fails loudly rather than drifting.
    """
    rep = gnl.run_load_ladder(genes, CFG, fractions=(0.01, 0.5, 1.0))
    service = rep["service_rel_diff"]
    assert service > gnl.PLAN_GNL_THRESHOLD, (
        f"GNL correction at service load is {service:.2%}, at or below the plan's "
        f"{gnl.PLAN_GNL_THRESHOLD:.0%} off-ramp — Stage 3 may no longer need the "
        f"nonlinear solve in the loop; re-read study_gnl.py before believing it")


def test_the_correction_is_not_a_constant_over_the_design_space(genes):
    """M5's decisive finding, and the half that actually closes the off-ramp.

    A 3.95% correction that were the SAME 3.95% for every design could simply be applied
    once.  It is not: held at a matched axle drop — the control, without which this only
    rediscovers that the correction grows with deflection — its coefficient of variation
    is 0.258 at worst over 20 seeds, against the 0.10 bar the off-ramp would need.

    Mirrors `test_wheel_fea.py`'s beam-blindness rerun, which is the same argument one
    gate earlier — and which had the SAME defect, retired in the same arc.

    ===========================================================================
    `iso_rel_diff_ratio > 3.0` WAS RETIRED IN THE REDS ARC.  DO NOT REINTRODUCE IT.
    ===========================================================================
    It was a `max/min` over the drawn rows — an estimator of the sample RANGE, which grows
    without bound with the number of draws, so it is a property of the sample and not of
    the design space.  Measured by `studies/study_reds_ratio_stability.py` (109 cells; its
    module comment carries the full table and the derivation):

        20 seeds at this test's own n=4 :  ratio 2.167 - 85.501, passing `> 3.0` in 11/20
        n = 4, 8, 12, 16, 24, 48 at seed 7 :  2.167, 4.203, 9.026, 9.026, 51.790, 51.790

    SEED 7 — the seed this test hard-coded — IS THE LOW OUTLIER of the twenty, in this
    study and in the beam one.  That is the entire failure; the SVK path is fine.

    The replacement bounds the CV, which is what `correction_factor_is_defensible` is
    DEFINED as (`cv < 0.10`, study_gnl.py line ~343), so this is PLAN §28's move — a stale
    constant replaced by the claim's own arithmetic — and `cv > 0.14` strictly implies the
    first assertion below.  0.14 is derived: the CV floor over all 109 cells (0.1450,
    which occurs in the BEAM study), floored to two decimals.  This test inherits a bound
    set by the other study and clears it by 1.8x — its own floor over 20 seeds is 0.2577 —
    which is what keeps the constant from being fitted to either run.

    Five seeds rather than one, because a single hard-coded draw is exactly the defect.
    """
    reps = {s: gnl.run_design_space(genes, CFG, n=4, seed=s, max_draws=2000)
            for s in RS.RETIREMENT_SEEDS}
    assert all(r.get("iso_rel_diff_ratio") is not None for r in reps.values()), reps
    assert not any(r["correction_factor_is_defensible"] for r in reps.values()), {
        s: r["iso_rel_diff_cv"] for s, r in reps.items()}
    worst = min(reps.items(), key=lambda kv: kv[1]["iso_rel_diff_cv"])
    assert worst[1]["iso_rel_diff_cv"] > RS.GATE_CORRECTION_CV, (
        f"the GNL correction's CV fell to {worst[1]['iso_rel_diff_cv']:.4f} at seed "
        f"{worst[0]}, under the {RS.GATE_CORRECTION_CV} gate — the correction is becoming "
        f"a single number and the Stage-2.5 off-ramp is reopening at its own 0.10 bar.  "
        f"That is news, not a gate to move: re-run "
        f"studies/study_reds_ratio_stability.py and read PLAN.md §31 first")


def test_the_retired_max_min_gate_is_decided_by_the_sample_size(genes):
    """Keep the REASON `iso_rel_diff_ratio > 3.0` was retired measured, not just asserted.

    The twin of `test_wheel_fea.py`'s pin of the same name.  `run_design_space` still
    publishes `iso_rel_diff_ratio` — deliberately, it is a useful diagnostic — so this
    asserts, executably, that no threshold can sit on it: the retired gate's VERDICT FLIPS
    with `n` at a fixed seed.  Measured at seed 7, the seed the retired test hard-coded:
    2.167 at n=4 and 9.026 at n=12.
    """
    small = gnl.run_design_space(genes, CFG, n=4, seed=7,
                                 max_draws=2000)["iso_rel_diff_ratio"]
    large = gnl.run_design_space(genes, CFG, n=12, seed=7,
                                 max_draws=2000)["iso_rel_diff_ratio"]
    assert small < 3.0 < large, (
        f"max/min over the drawn rows read {small:.3f} at n=4 and {large:.3f} at n=12 — "
        f"it no longer brackets the retired 3.0 gate, so the demonstration that the gate's "
        f"verdict was decided by the sample size has stopped working")


def test_everything_softens_and_nothing_stiffens(genes):
    """The sign, which is the one thing a magnitude check cannot catch.

    A curved flexure loaded to unwind gains moment arm as it deflects, so the tangent
    stiffness falls and every row must come out softer.  A STIFFENING geometric term at
    this scale is the signature of a sign error in the Green-Lagrange strain, and it
    would still produce plausible-looking magnitudes.
    """
    rep = gnl.run_load_ladder(genes, CFG, fractions=(0.01, 0.5, 1.0, 2.0))
    assert rep["all_softer"], [r["rel_diff"] for r in rep["rows"]]


def test_the_correction_enters_at_first_order_in_the_load(genes):
    """Exponent ~1, which single-point agreement cannot establish.

    An implementation that simply ran the linear kernel twice would pass the plan's
    small-load criterion perfectly.  Exponent 0 would mean a constant offset — a
    modelling difference rather than a geometric effect.

    THE PRE-REGISTERED SMALL-LOAD GATE MOVED OUT of this test, to the xfail below, in the
    REDS arc (PLAN.md §31).  It has been red since §14 and is expected to stay red; while
    it lived here it took the exponent assertion — the one this test is named for, and one
    that passes — down with it on every run.  Nothing about either claim changed.
    """
    rep = gnl.run_load_ladder(genes, CFG, fractions=(0.01, 0.1, 0.5, 1.0, 2.0))
    assert 0.7 < rep["fitted_exponent"] < 1.4, rep["fitted_exponent"]


@pytest.mark.xfail(reason=(
    "PLAN.md §14 item 4a decided this pre-registered gate STANDS; SVK_PLAN Step 0 and "
    "§31 (REDS Step 4) re-declared it.  small_load_rel_diff = 0.2007% against a 0.1% "
    "gate — a true statement about a 1.2 mm wall, not a defect.  GATE_SMALL_LOAD_REL is "
    "NOT to be moved.  strict=True via pyproject.toml, so this reopens itself if the "
    "wheel ever passes it."))
def test_the_gnl_correction_is_small_at_one_percent_of_service_load(genes):
    """PRE-REGISTERED, BREACHED, AND DELIBERATELY HELD.  Do not move the gate.

    §14 swept the mesh to find out whether 0.205% on `smoke` was resolution:

        genome      smoke     coarse    medium
        350f4c7    0.2050%   0.2081%   0.2089%      <-- gate is 0.1%
        36aed36    0.0373%   0.0382%   0.0384%

    Converged by `coarse` on both, and mesh-independent to three digits.  The promoted
    1.2 mm wheel is **5.5x more geometrically nonlinear** than the GA/beam one it
    replaced, which is what a thinner, floppier part does.

    THE EXPONENT IS FINE, which is what says this is a real result and not a broken solve.
    `test_the_correction_enters_at_first_order_in_the_load` above passes at 1.0393 inside
    (0.7, 1.4), re-measured by §31.  The correction still enters at first order; what moved
    is the coefficient.  §31 re-measured this gate too, unchanged: 0.0020070 against 1e-3,
    over by 2.01x.

    `GATE_SMALL_LOAD_REL` HAS DELIBERATELY NOT BEEN MOVED, and this is the third arc to
    say so.  `study_gnl.py` records it as "written down BEFORE the study was run, per the
    plan's rule"; §14 refused to move it twice; SVK_PLAN Step 0 re-declared it; §31 refused
    again.  Re-fitting a pre-registered gate to the design that breached it is exactly the
    move that rule exists to prevent, and raising it to 3e-3 or similar is forbidden.

    WHAT THIS IS ACTUALLY WAITING ON, and it is bigger than the gate: §14 called it "the
    most important thing §14 found" — whether LINEAR KINEMATICS IS STILL AN ACCEPTABLE
    DEFAULT for a 1.2 mm wall at all.  If the answer is no, the fix belongs in the physics
    defaults and not in this number.  §31 flagged it as a successor and deliberately did
    not act on it.  See PLAN.md §14 item 4a and §31.
    """
    rep = gnl.run_load_ladder(genes, CFG, fractions=(0.01, 0.1, 0.5, 1.0, 2.0))
    assert abs(rep["small_load_rel_diff"]) < gnl.GATE_SMALL_LOAD_REL, rep


# ---------------------------------------------------------------------------
# THE TRAP
# ---------------------------------------------------------------------------

def test_svk_through_solve_linear_silently_returns_the_linear_answer(mesh):
    """Pin the trap, because it is invisible and it already cost time once.

    `solve_linear` assembles the tangent at u=0, where the SVK Hessian equals the linear
    one EXACTLY.  So an `svk` problem routed through it returns a linear answer with no
    error and no warning.  `solve` dispatches on `prob.nonlinear` and is the only correct
    entry point.

    This test asserts the trap still behaves as documented rather than asserting it has
    been fixed — the equality at u=0 is a mathematical fact, not a bug to remove.  What
    would be a bug is `solve` losing its dispatch, and that is the second assertion.
    """
    prob, _ = fem.wheel_problem(mesh, kinematics="svk")
    assert prob.nonlinear

    linear_path = fem.solve_linear(prob)
    dispatched = fem.solve(prob)

    assert "newton" not in linear_path, (
        "solve_linear grew a Newton block — if it now handles contact or nonlinear "
        "kinematics, this test and the wheel_fem docstring both need rewriting")
    assert "newton" in dispatched, "solve() stopped dispatching on prob.nonlinear"

    rel = abs(dispatched["axle_drop_mm"] / linear_path["axle_drop_mm"] - 1.0) \
        if "axle_drop_mm" in linear_path else None
    if rel is not None:
        assert rel > 1e-3, (
            f"the dispatched SVK answer differs from the linear path by only {rel:.2e} "
            f"— the Newton loop is not doing anything")


def test_a_diverged_solve_raises_rather_than_returning_a_field(mesh):
    """A best-effort field with a warning would poison a Stage-3 gradient undetectably.

    Forced by capping the iteration budget at one, which cannot converge from a cold
    start.  The caller's correct response to divergence is to reject the design step and
    shrink it, and it can only do that if it is told.
    """
    prob, _ = fem.wheel_problem(mesh, kinematics="svk")
    with pytest.raises(fem.NewtonDivergedError):
        fem.solve_nonlinear(prob, max_iter=1, tol=1e-14, tol_energy=1e-30)


def test_stress_recovery_follows_the_solves_kinematics(genes, mesh):
    """The footgun §14 walked into, pinned so it cannot come back.

    `wheel_fem.gauss_stresses` takes `nonlinear=False` by DEFAULT.  That is correct for a
    linear solve and silently wrong for an SVK one — it applies the engineering-strain
    formula to a large displacement field, and the result is not a stress.  It does not
    warn, it does not NaN, and the number it returns is plausible enough to quote.

    Measured on the shipped genome at service load: the correct Cauchy push-forward gives
    a plain-spoke p99 of 19.75 MPa against the linear kernel's 17.27, a real +14.3%.  The
    linear formula on the same SVK field says 46.56 MPa, +169.5%.  An order of magnitude
    apart, and the wrong one is the one you get by accident.

    So `study_wheel_fea.stress_report` now reads `res["meta"]["kinematics"]` rather than
    taking the default.  What this asserts is that it actually does — that an SVK result
    and a linear result do not come back with the same stress, and that the SVK one
    matches an explicit `nonlinear=True` recovery.
    """
    import numpy as np
    import study_wheel_fea as swf
    import wheel_fea as W

    lin = fem.solve_wheel(mesh, kinematics="linear", force=W.TOTAL_FORCE_NEWTONS)
    svk = fem.solve_wheel(mesh, kinematics="svk", force=W.TOTAL_FORCE_NEWTONS)
    assert lin["meta"]["kinematics"] == "linear"
    assert svk["meta"]["kinematics"] == "svk", (
        "the solve stopped recording its own kinematics, which is what stress_report "
        "dispatches on — it will now silently mis-recover every SVK field")

    lam, mu = fem.lame(W.YOUNGS_MODULUS_PLA_MPA, fem.POISSON_RATIO_PLA)

    def p99(res, nonlinear):
        st = fem.gauss_stresses(np.asarray(mesh.coords), mesh.conn, res["u"],
                                order=mesh.cfg.order, lam=lam, mu=mu,
                                nonlinear=nonlinear, cauchy=True)
        return float(np.percentile(st["von_mises"][mesh.element_block == "spoke"], 99.0))

    # stress_report must agree with the EXPLICIT correct recovery on both...
    assert swf.stress_report(mesh, lin)["spoke_block_p99_mpa"] == pytest.approx(
        p99(lin, False), rel=1e-12)
    assert swf.stress_report(mesh, svk)["spoke_block_p99_mpa"] == pytest.approx(
        p99(svk, True), rel=1e-12), (
        "stress_report used the linear strain formula on an SVK field — see this test's "
        "docstring for how large that error is")

    # ...and the wrong recovery must be visibly different, or this test proves nothing.
    wrong = p99(svk, False)
    assert wrong / p99(svk, True) > 1.5, (
        f"the mis-recovery is only {wrong / p99(svk, True):.3f}x here, so this test can "
        f"no longer tell the two apart and is not guarding anything")
