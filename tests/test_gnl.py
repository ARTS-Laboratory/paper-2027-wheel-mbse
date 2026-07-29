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
    rediscovers that the correction grows with deflection — it spans about an order of
    magnitude.

    Mirrors `test_wheel.py`'s beam-blindness rerun, which is the same argument one gate
    earlier.
    """
    rep = gnl.run_design_space(genes, CFG, n=4, seed=7, max_draws=2000)
    assert rep.get("iso_rel_diff_ratio") is not None, rep
    assert not rep["correction_factor_is_defensible"], rep["iso_rel_diff_cv"]
    assert rep["iso_rel_diff_ratio"] > 3.0, rep["iso_rel_diff_ratio"]


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
    """
    rep = gnl.run_load_ladder(genes, CFG, fractions=(0.01, 0.1, 0.5, 1.0, 2.0))
    assert 0.7 < rep["fitted_exponent"] < 1.4, rep["fitted_exponent"]
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
