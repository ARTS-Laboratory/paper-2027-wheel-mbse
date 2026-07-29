"""M8b-i — the Stage-3 optimizer.

Split the way `test_objective.py` is: the IDENTITIES that must hold exactly (the Adam
update, the projection, the lattice, the warm-start channel), the STRUCTURAL FACTS about
what the driver refuses to do, and small reruns of `study_stage3.py`'s headline
assertions at `smoke`.

Tolerances are read off `study_stage3.GATE_*` rather than duplicated here, so a gate
cannot be relaxed in one place and stay strict in the other.

Everything that touches a solve runs on the `smoke` mesh with two phases and a handful
of steps, for the same reason `test_objective.py` does: a phase-batched contact solve at
`coarse` would put minutes into `make test`.  The identities below do not touch a solve
at all and are checked at full precision.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import jax_config  # noqa: E402,F401

import study_stage3 as so3  # noqa: E402
import wheel_adjoint as WA  # noqa: E402
import wheel_fea as W  # noqa: E402
import wheel_genome as wg  # noqa: E402
import wheel_objective as WO  # noqa: E402
import wheel_stage3 as S3  # noqa: E402
import wheel_wheel as WW  # noqa: E402

CFG = "smoke"
N_PHASE = 2


@pytest.fixture(scope="module")
def genes():
    return so3.load_genes()


@pytest.fixture(scope="module")
def bounds():
    return wg.bounds_arrays(W.GENE_SPACE)


@pytest.fixture(scope="module")
def z0(genes, bounds):
    low, high, _ = bounds
    return wg.normalize(genes, low, high)


# ---------------------------------------------------------------------------
# THE IDENTITIES
# ---------------------------------------------------------------------------

def test_adam_matches_a_hand_rolled_reference_for_two_steps():
    """`adam_update` is a pure function so that this test can exist at all.

    Two steps rather than one: the bias correction `1 - b1**t` is the part that is easy
    to get wrong, and at `t = 1` a great many wrong expressions agree with the right one.
    """
    rng = np.random.default_rng(0)
    g1, g2 = rng.normal(size=14), rng.normal(size=14)
    b1, b2, eps, lr = S3.ADAM_B1, S3.ADAM_B2, S3.ADAM_EPS, 0.01

    m = (1 - b1) * g1
    v = (1 - b2) * g1 * g1
    ref1 = lr * (m / (1 - b1 ** 1)) / (np.sqrt(v / (1 - b2 ** 1)) + eps)
    m = b1 * m + (1 - b1) * g2
    v = b2 * v + (1 - b2) * g2 * g2
    ref2 = lr * (m / (1 - b1 ** 2)) / (np.sqrt(v / (1 - b2 ** 2)) + eps)

    d1, m1, v1 = S3.adam_update(g1, np.zeros(14), np.zeros(14), 1, lr)
    d2, _, _ = S3.adam_update(g2, m1, v1, 2, lr)

    assert np.allclose(d1, ref1, rtol=0, atol=1e-15), (
        "the first Adam step disagrees with the textbook update — check the bias "
        "correction, not the moments")
    assert np.allclose(d2, ref2, rtol=0, atol=1e-15), (
        "the second Adam step disagrees; `t` is 1-based in `adam_update` and the bias "
        "correction is the only thing that depends on it")


def test_the_projection_is_exact_and_idempotent():
    z = np.array([-0.3, 0.0, 0.5, 1.0, 1.7] + [0.5] * 9)
    p = S3.project(z)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert p[0] == 0.0 and p[4] == 1.0, "clip must land exactly ON the bound"
    assert np.array_equal(S3.project(p), p), "projection must be idempotent"


def test_the_gradient_clip_preserves_direction_and_reports_the_raw_norm():
    g = np.arange(1.0, 15.0)
    clipped, norm = S3.clip_global_norm(g, 1.0)
    assert np.isclose(norm, np.linalg.norm(g))
    assert np.isclose(np.linalg.norm(clipped), 1.0)
    assert np.allclose(clipped / np.linalg.norm(clipped), g / np.linalg.norm(g)), (
        "clipping is a rescale, not a per-component clamp — the direction is the whole "
        "point of the gradient")
    small = np.array([1e-3] + [0.0] * 13)
    back, _ = S3.clip_global_norm(small, 1.0)
    assert np.array_equal(back, small), "a gradient under the cap must be untouched"


def test_the_cosine_schedule_starts_at_lr_and_ends_at_zero():
    assert S3.cosine_lr(0.01, 0, 100) == pytest.approx(0.01)
    assert S3.cosine_lr(0.01, 100, 100) == pytest.approx(0.0, abs=1e-18)
    assert S3.cosine_lr(0.01, 50, 100) == pytest.approx(0.005)


@pytest.mark.parametrize("n_phase,n_sub", [(8, 8), (4, 4), (2, 8)])
def test_the_rqmc_offset_always_lands_on_the_fixed_lattice(n_phase, n_sub):
    """The whole performance argument for `rqmc` over a continuous shift.

    `coord_fn` keys its jit cache on `float(phase)`, so if a draw could land off the
    `n_phase * n_sub` grid it would re-trace on every step forever.  Checked over many
    draws because a scheme that is usually on the lattice is not on the lattice.
    """
    cell = WO.SECTOR_DEG / n_phase
    lattice = np.arange(n_phase * n_sub) * (cell / n_sub)
    rng = np.random.default_rng(0)
    for _ in range(50):
        for p in WO.phase_stencil(n_phase, n_sub, "rqmc", rng):
            assert np.min(np.abs(lattice - p)) < 1e-12, (
                f"phase {p} is off the {n_phase}x{n_sub} lattice; coord_fn's cache "
                f"would miss on every step — see wheel_stage3's module docstring")


def test_the_warm_vector_is_the_previous_drops_and_nothing_else():
    """`delta0` is the secant's indentation and `wheel_adjoint.py:537` makes that the
    same number as `axle_drop_mm`, which is why no change to `wheel_objective` was
    needed to warm-start.  If that identity ever moves, this is what says so."""
    brk = {"report": {"rows": [{"axle_drop_mm": 1.5}, {"axle_drop_mm": 1.6}]}}
    assert S3.warm_from(brk) == [1.5, 1.6]
    assert S3.warm_from({"report": {}}) is None, (
        "a missing T3 block must give None — the cold-start input — not an empty list")


# ---------------------------------------------------------------------------
# THE STRUCTURAL FACTS
# ---------------------------------------------------------------------------

def test_lbfgsb_refuses_a_stochastic_phase_stencil(z0):
    """A quasi-Newton method fits curvature to differences of gradients; if the stencil
    moves between them the model is being fitted to the stencil.  Adam tolerates that by
    construction and L-BFGS-B does not, so this raises rather than converging to
    something plausible-looking."""
    with pytest.raises(ValueError, match="deterministic"):
        S3.descend_lbfgsb(z0, CFG, steps=1, scheme="rqmc")


def test_an_unknown_start_spec_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown --start"):
        S3.start_points("elite3")


def test_the_probe_weight_sets_keep_every_barrier_on():
    """A "lowest reachable stress" reached through a folded or unmeshable design is a
    bound on nothing.  S9's probes zero OBJECTIVES, never barriers."""
    barriers = S3.T1_BARRIER_NAMES + ("min_sj",)
    for kind in so3.PROBE_ZERO:
        w = so3.probe_weights(kind)
        for b in barriers:
            assert w[b] == WO.DEFAULT_WEIGHTS[b], (
                f"probe {kind!r} disturbed the {b!r} barrier; only deflection, stress, "
                f"mass and phase_ripple may be switched off")
    assert so3.probe_weights("joint") == WO.DEFAULT_WEIGHTS, (
        "the joint probe must be the real objective, unmodified")
    assert so3.probe_weights("stress_only")["deflection"] == 0.0
    assert so3.probe_weights("deflection_only")["stress"] == 0.0


def test_the_t1_precheck_costs_no_mesh_and_no_solve(z0, monkeypatch):
    """The reject rule is only worth having because `tiers=("t1",)` builds nothing.

    `("t1","t2")` would still pay a `build_wheel`, which is the trap this pins shut.
    """
    calls = []
    real = WW.build_wheel
    monkeypatch.setattr(WW, "build_wheel",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    S3.t1_barrier_sum(z0, CFG)
    assert not calls, (
        "the T1 pre-check built a mesh — it must use tiers=('t1',) only, or the "
        "cheap-refusal argument in wheel_stage3's docstring is false")


def test_the_barrier_screen_refuses_only_gross_infeasibility(z0):
    """The regression test for the deadlock that ate a coarse feasibility probe.

    The screen was `b_trial > max(1.0, b_here)` — never increase a barrier.  The shipped
    design carries `hub_overlap` = 52.13; the stress-reducing direction raises it to
    55.41 and halving only walks it back to 52.54, so every trial at every scale was
    refused and all forty steps were abandoned.  The probe then reported its own starting
    point as the lowest reachable stress utilisation.

    Every barrier is already a weighted term in the objective, so a screen that vetoes
    increasing one overrides the weights and forbids the constrained trade the run exists
    to make.  The screen keeps only the job the objective cannot do — refusing to spend a
    solve on geometry that will not mesh — and 1e4 is that scale against barriers that
    sit in the tens.
    """
    assert S3.T1_REJECT >= 1e3, (
        "T1_REJECT is back to a value comparable with a HEALTHY barrier; it is a "
        "gross-infeasibility threshold, not a no-increase rule — see its comment")

    b0, brk = S3.t1_barrier_sum(z0, CFG)
    assert b0 < S3.T1_REJECT, (
        f"the shipped genome's own barrier sum {b0} is above the gross threshold, so "
        f"every step would be refused from the start")
    # A modest worsening of the live barrier must still be admissible.
    assert 1.2 * b0 < S3.T1_REJECT, (
        "a 20% worse barrier is already refused; that is the deadlock rule again")
    assert S3.barrier_sum_of(brk) == b0, (
        "barrier_sum_of and t1_barrier_sum disagree, so the free reading of the current "
        "point's barrier is not the same quantity as the screen's")


def test_a_stuck_run_stops_and_says_so(z0):
    """An abandonment storm must terminate with a record, not run out the budget.

    Halving `lr` on every abandoned step drives it to zero geometrically, so a
    deterministic refusal freezes the run silently.  The coarse probe burned thirty-five
    minutes that way before anyone could see it.
    """
    # Driven through the SOLVE-reject route rather than the barrier screen: the screen's
    # threshold is `max(t1_reject, b_here)`, so the deadlock-escape clause means no value
    # of `t1_reject` alone can force a refusal — which is the property that clause exists
    # to have.  An always-failing evaluator reaches the same abandonment path.
    ev = so3._FaultyEvaluator(CFG, n_fail=10_000,
                              orientation=WW.flank_orientation(
                                  so3.load_genes(), WW.get_config(CFG)))
    rec = S3.descend(z0, CFG, steps=40, n_phase=N_PHASE, scheme="uniform",
                     max_rejects=0, evaluator=ev, verbose=False)
    stops = [e for e in rec["events"] if e["kind"] == "run_stopped_stuck"]
    assert len(stops) == 1, f"a fully-refused run did not stop: {rec['events'][:4]}"
    assert stops[0]["steps_remaining"] > 30, (
        "the run stopped, but only after spending most of its budget standing still")
    assert len(rec["steps"]) - 1 == S3.MAX_CONSECUTIVE_ABANDONED
    lrs = [e["lr_after"] for e in rec["events"] if e["kind"] == "step_abandoned"]
    assert min(lrs) >= S3.DEFAULT_LR * S3.LR_FLOOR_FRAC, (
        "the learning rate decayed below its floor, so a long storm would leave the run "
        "unable to move even once the refusals stopped")


def test_the_run_record_carries_the_iterate_and_the_gradient(z0):
    """S3 and S4 reconstruct the projection and the scale policy offline, which is only
    possible if every step row records `z` and `grad` rather than their norms."""
    rec = S3.descend(z0, CFG, steps=1, n_phase=N_PHASE, scheme="uniform", verbose=False)
    for row in rec["steps"]:
        assert row["z"] is not None and len(row["z"]) == wg.N_GENES
        assert len(row["grad"]) == wg.N_GENES
        assert 0.0 <= min(row["z"]) and max(row["z"]) <= 1.0


# ---------------------------------------------------------------------------
# SMALL RERUNS OF THE GATE
# ---------------------------------------------------------------------------

def test_a_failed_solve_is_a_step_reject_and_the_run_recovers(genes):
    """S5 at `smoke`, both halves: recovery from injected divergences, and a correct
    surrender when every trial fails.

    Fault injection rather than a hunt for a real divergence — the reject path is
    exercised rarely and has to work the first time it is.
    """
    r = so3.run_reject(genes, CFG, n_phase=N_PHASE, steps=1)
    assert r["recovered"]["pass"], (
        f"the driver did not recover from injected NewtonDivergedError: "
        f"{r['recovered']}")
    assert r["restored"]["pass"], (
        f"with every trial failing the iterate must be restored unchanged and the "
        f"learning rate must decay: {r['restored']}")
    assert r["recovered"]["trial_scales"] == [1.0, 0.5], (
        "a rejected trial must halve the step, so the scales are 1, 1/2, 1/4, ...")


def test_the_evaluator_carries_no_state_that_could_stale_a_gradient(z0):
    """S4 is gone, and this is what replaced it: nothing is frozen, so check nothing is.

    S4 used to assert that the `stress_scale` an evaluation was handed equalled the
    previous step's measurement — M8a's gate-7 lesson made into a run-time invariant,
    because `c` re-measured inside a call made value and gradient answers to different
    questions.  M8b-i.6 deleted the frozen factor instead: `c` is anchored to a
    mesh-divergent true max, so the pinned number was a converged answer to nothing.

    The `Evaluator` may still carry the ORIENTATION pin, which is a discrete topological
    decision and must persist (docstring item 3).  What it must not carry is any
    continuous scalar that enters the loss, because that is the shape of state that goes
    stale between a value and its gradient without anything crashing.
    """
    counters = {"n_calls", "mesh_s", "solve_s"}
    ev = S3.Evaluator(CFG)
    phases = WO.phase_stencil(n_phase=N_PHASE, scheme="uniform")
    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)

    before = {k: v for k, v in vars(ev).items() if k not in counters}
    ev(z0, low, high, phases=phases, tiers=("t1",))
    after = {k: v for k, v in vars(ev).items() if k not in counters}

    assert before == after, (
        f"evaluating MUTATED the Evaluator: "
        f"{ {k: (before.get(k), after.get(k)) for k in after if before.get(k) != after[k]} }. "
        f"State written by one call and read by the next is the shape of bug M8a's gate 7 "
        f"caught — see wheel_stage3's docstring item 1")

    rec = S3.descend(z0, CFG, steps=2, n_phase=N_PHASE, scheme="uniform", verbose=False)
    live = [r for r in rec["steps"] if not r["abandoned"]]
    assert live and "stress_scale_measured" in live[0]["report"], (
        "`c` stopped being reported; it is no longer an input but it IS the evidence for "
        "why, and `make m8bi6` reads it")


def test_the_projection_never_freezes_a_gene_whose_gradient_points_inward(genes):
    """S3 at `smoke`, and the reason the run projects instead of reparameterising.

    Six of the fourteen genes sit exactly on a bound at the shipped genome, and under a
    sigmoid map all six would start with a vanishing gradient.  The check is an
    IMPLICATION rather than a hope that some gene happened to move: a gene on a bound
    stays there if and only if the unprojected step would have left the box.
    """
    r = so3.run_trajectory(genes, CFG, steps=2, n_phase=N_PHASE)["projection"]
    assert not r["box_violations"], f"an iterate left the unit box: {r['box_violations']}"
    assert not r["freeze_violations"], (
        f"a gene on a bound did not move even though the step pointed back into the "
        f"box — the projection is freezing genes: {r['freeze_violations']}")
    assert r["pinned_at_start"], (
        "the shipped genome is supposed to have genes exactly on their bounds; if it "
        "no longer does, this test has stopped checking anything")


def test_every_mesh_a_step_builds_is_built_with_the_pinned_orientation(genes, z0,
                                                                       monkeypatch):
    """S6 at `smoke`.  `build_wheel` re-derives orientation from the genes unless it is
    passed, and a mid-run flip changes which block owns which node — the gradient of the
    step that flips it does not exist.

    Checked by watching what `phase_meshes` is actually called with, rather than by
    building a mesh at the opposite orientation: at the shipped genome the `eta=-1` hub
    arrival does not exist at all (the flank never reaches r=12.7 mm and `ring_station`
    says so), which is the same real geometry M8a's fillet term ran into.  A pin can only
    be tested by whether it is THREADED, not by flipping it to something unbuildable.
    """
    pin = tuple(float(x) for x in
                np.asarray(WW.flank_orientation(genes, WW.get_config(CFG))).ravel())
    seen = []
    real = WO.phase_meshes

    def spy(g, cfg, phases, orientation=None):
        seen.append(orientation)
        return real(g, cfg, phases, orientation=orientation)

    monkeypatch.setattr(WO, "phase_meshes", spy)
    rec = S3.descend(z0, CFG, steps=1, n_phase=N_PHASE, scheme="uniform", verbose=False)

    assert seen, "the run built no meshes through phase_meshes — the pin cannot apply"
    for o in seen:
        assert o is not None, (
            "phase_meshes was called without an orientation, so build_wheel re-derived "
            "it from the genes and the pin is a no-op")
        assert tuple(float(x) for x in np.asarray(o).ravel()) == pin
    assert tuple(rec["settings"]["orientation"]) == pin, (
        "the run record must state the orientation it was pinned to, or S6 has nothing "
        "to check against")


# ---------------------------------------------------------------------------
# M8b-i.5 — THE TWO SECTIONS THAT QUALIFY S9's VERDICT
# ---------------------------------------------------------------------------

def test_the_default_sections_are_the_m8bi_gate_in_its_original_order():
    """`--sections` must not silently change what `make studies` measures.

    The default list is the seven sections S1-S10 have always been, in the order they
    have always run, because the report's keys are written in that order and the repo's
    regression discipline compares two reports leaf by leaf.  M8b-i.5's two are opt-in.
    """
    assert so3.DEFAULT_SECTIONS == ("direction", "trajectory", "reject", "schemes",
                                    "warm", "cost", "feasibility")
    assert set(so3.SECTION_HELP) == set(so3.DEFAULT_SECTIONS) | {"mesh_convergence",
                                                                 "multistart"}
    assert all(s in so3.PRINTERS for s in so3.SECTION_HELP), (
        "every section needs a printer, or a selected section runs for hours and then "
        "prints nothing")
    assert so3.parse_sections(",".join(so3.DEFAULT_SECTIONS)) == list(
        so3.DEFAULT_SECTIONS)


def test_an_unknown_section_is_refused_at_startup_rather_than_three_hours_in():
    """A typo must cost a startup, not a `coarse` run that ends in a `KeyError`."""
    with pytest.raises(ValueError, match="unknown section"):
        so3.parse_sections("direction,feasibilty")          # codespell:ignore
    with pytest.raises(ValueError, match="empty"):
        so3.parse_sections(" , ")
    # Order and selection are the caller's, and are preserved verbatim.
    assert so3.parse_sections("multistart,direction") == ["multistart", "direction"]


def test_corner_distance_is_zero_inside_the_box_and_grows_outside():
    """The ranking that decides which elites are worth an hour of descent.

    Both terms are excesses over their OWN limit, which is what makes them addable —
    `util` is already normalised by the allowable and `defl_err` by the target, so the
    mistake this exists to avoid is summing "MPa over" with "mm short".
    """
    assert so3.corner_distance(so3.FEASIBLE_UTIL, so3.FEASIBLE_DEFL_REL) == 0.0
    assert so3.corner_distance(0.5, 0.0) == 0.0
    # ... and the sign of the deflection error cannot matter: too stiff is as infeasible
    # as too soft, which is `wheel_objective`'s two-sided deflection term.
    assert so3.corner_distance(1.0, 0.10) == so3.corner_distance(1.0, -0.10) > 0.0
    # Each violation moves it on its own.
    assert so3.corner_distance(2.0, 0.0) > so3.corner_distance(1.5, 0.0) > 0.0
    assert so3.corner_distance(1.0, 0.20) > so3.corner_distance(1.0, 0.10) > 0.0
    # Doubling the allowable and doubling the target-relative tolerance cost the same,
    # which is the whole claim the sum rests on.
    assert so3.corner_distance(2 * so3.FEASIBLE_UTIL, 0.0) == pytest.approx(
        so3.corner_distance(0.0, 2 * so3.FEASIBLE_DEFL_REL))


def test_a_two_rung_ladder_reports_no_ratio_rather_than_extrapolating_from_nothing():
    """Richardson needs three points.  With two, `_series` must say so.

    A NaN here is the honest answer and a number would be a fabrication — `--quick` runs
    a two-rung ladder as a wiring check, and it must not print a convergence claim.
    """
    rows = [{"config": "smoke", "x": 1.0}, {"config": "coarse", "x": 2.0}]
    s = so3._series(rows, "x")
    assert s["settling"] is None
    assert np.isnan(s["ratio"]) and np.isnan(s["richardson"])
    assert s["direction"] == "rising"
    assert s["rel_change_last_pair"] == pytest.approx(1.0)

    three = rows + [{"config": "medium", "x": 2.5}]
    s3 = so3._series(three, "x")
    # 1 -> 2 -> 2.5: each refinement moved it less than the one before, so the sequence
    # is settling and the extrapolate sits beyond the finest value.
    assert s3["settling"] is True
    assert s3["ratio"] == pytest.approx(2.0)
    assert s3["richardson"] == pytest.approx(3.0)
    assert s3["converged"] is None or s3["converged"] is False


def test_settling_is_not_convergence_and_the_report_must_not_conflate_them():
    """The distinction that a plot title once got wrong.

    `settling` is `ratio > 1` — the successive differences are shrinking, which is
    necessary and nowhere near sufficient.  A sequence can settle while its remaining
    discretisation error is enormous, and reading a verdict off `settling` produced a
    figure captioned "the stress QoI settles under refinement" above a utilisation whose
    GCI was 63%.  `converged` is the claim worth quoting.
    """
    # The measured shipped-genome utilisation: settling, and nowhere near converged.
    util = so3._series([{"config": c, "x": v} for c, v in
                        (("smoke", 1.2617), ("coarse", 1.7128), ("medium", 2.0525))], "x")
    assert util["settling"] is True, "differences ARE shrinking; that is the trap"
    assert util["converged"] is False
    assert util["gci"] > 0.5, f"expected a huge GCI, got {util['gci']}"

    # The axle drop off the very same solves: both.
    drop = so3._series([{"config": c, "x": v} for c, v in
                        (("smoke", 1.4523), ("coarse", 1.4914), ("medium", 1.4986))], "x")
    assert drop["settling"] is True
    assert drop["converged"] is True
    assert drop["gci"] < so3.GATE_LADDER_GCI

    # And the threshold is the one constant, not a number typed twice.
    borderline = so3._series([{"config": c, "x": v} for c, v in
                              (("smoke", 1.0), ("coarse", 2.0), ("medium", 2.5))], "x")
    assert borderline["converged"] == (borderline["gci"] < so3.GATE_LADDER_GCI)


def test_an_unknown_probe_kind_is_refused_rather_than_guessed():
    """A typo in `kinds` must not silently run fewer probes than the caller asked for."""
    with pytest.raises(ValueError, match="unknown probe kind"):
        so3.run_feasibility(np.zeros(14), CFG, steps=1, kinds=("stress",))


def test_a_subset_of_probes_omits_the_keys_it_did_not_measure(genes):
    """`kinds=` is what lets the multi-start re-probes skip `joint`.

    The verdict must then carry only what its probes support.  An absent key is a
    question that was not asked; a NaN reads as a question that was asked and came back
    undefined, and the two get acted on differently.
    """
    f = so3.run_feasibility(genes, CFG, steps=1, n_phase=N_PHASE,
                            kinds=("stress_only",))
    assert list(f["runs"]) == ["stress_only"]
    assert f["kinds"] == ["stress_only"]
    v = f["verdict"]
    assert "defl_err_when_stress_met" in v
    assert "util_when_deflection_met" not in v, (
        "no deflection-only probe ran, so there is no utilisation at which deflection "
        "was met — the key must be absent, not NaN")
    assert "joint_util_end" not in v
    # The two bounds still exist: they are minima over whatever DID run.
    assert np.isfinite(v["min_reachable_util"])
    assert np.isfinite(v["min_reachable_abs_defl_err"])


def test_the_ladder_costs_one_evaluation_per_rung_and_repins_per_design(genes, monkeypatch):
    """S11's cost claim, and the pin that has to be re-derived rather than carried.

    One rung is one evaluation however many exponents are probed — the probe reads them
    off a displacement field the adjoint already converged, which is the whole reason a
    five-exponent sweep is fourteen minutes and not an hour.

    The orientation pin is re-derived per rung for the same reason it is re-derived per
    design: it is a function of the genes and the config, not a run-wide constant.
    """
    seen = []
    real = S3.Evaluator

    class Spy(real):
        def __call__(self, *a, **kw):
            seen.append({"cfg": self.cfg, "orientation": self.orientation,
                         "tiers": kw.get("tiers")})
            return super().__call__(*a, **kw)

    monkeypatch.setattr(S3, "Evaluator", Spy)
    rep = so3.run_mesh_convergence([("shipped", genes)], configs=(CFG,), n_phase=N_PHASE)

    assert len(seen) == 1, f"one rung, one evaluation; got {len(seen)}"
    assert seen[0]["tiers"] == ("t3",), (
        "the ladder pays t1's eager jacrev for barrier numbers it never reads")
    assert seen[0]["orientation"] == tuple(
        float(x) for x in np.asarray(
            WW.flank_orientation(genes, WW.get_config(CFG))).ravel())
    row = rep["designs"][0]["rows"][0]
    # The identity the report's series rest on, at the one rung that ran.  `Kt`, not `c`:
    # `c` is still reported beside it, and the gap between these two numbers is M8b-i.6's
    # finding rather than a discrepancy.
    assert row["stress_utilisation"] == pytest.approx(
        max(row["kt_hub"], row["kt_rim"]) * row["pnorm_stress_agg_mpa"]
        / WO.ALLOWABLE_STRESS_MPA, rel=1e-12)


# ---------------------------------------------------------------------------
# M8b-i.6 STEP 1 — THE p SWEEP
# ---------------------------------------------------------------------------

def test_a_bad_exponent_is_refused_at_startup_rather_than_after_the_meshes():
    """`parse_ladder_p` is `parse_sections`' contract, and the stakes are the same.

    The sweep's whole claim is that it costs no extra solve.  An exponent that only fails
    inside `_qoi_pnorm_stress` would be discovered after the ladder had already paid for
    its meshes and its contact solves, which is exactly the three-hours-then-KeyError that
    `parse_sections` exists to prevent one argument over.
    """
    assert so3.parse_ladder_p("2,4,8,16,30") == [2.0, 4.0, 8.0, 16.0, 30.0]
    assert so3.parse_ladder_p(" 2 , 4 ") == [2.0, 4.0]
    assert so3.parse_ladder_p("") == [], "empty means no sweep, not an error"
    assert so3.parse_ladder_p("2.5") == [2.5], "the interesting p need not be an integer"

    for bad in ("2,four,8", "p30", "2,,x"):
        with pytest.raises(ValueError, match="not a number"):
            so3.parse_ladder_p(bad)
    # A p-norm with p <= 0 is not a norm, and `x**(1/p)` at p=0 is a ZeroDivisionError
    # thrown from inside a jit trace, where it is unreadable.
    for bad in ("0", "-4", "2,-1", "inf"):
        with pytest.raises(ValueError):
            so3.parse_ladder_p(bad)

    # A REPEATED EXPONENT IS THE SAME BUG WEARING A VALID NUMBER.  `t3_terms` keys its
    # accumulator by `p` but appends inside a loop over the raw list, so a duplicate
    # collects two samples per phase against one true max and `_stress_aggregate` dies on
    # the broadcast — after the first solve.  Dropped here rather than raised, because the
    # user's intent is unambiguous and a sweep is not worth failing over.  First occurrence
    # wins, so the reported order still follows the command line.
    assert so3.parse_ladder_p("8,8") == [8.0]
    assert so3.parse_ladder_p("30,2,30,4,2") == [30.0, 2.0, 4.0]
    assert so3.parse_ladder_p("8, 8.0 ,8") == [8.0], "same number, three spellings"


def test_the_probe_reproduces_the_constraint_at_the_exponent_it_was_built_with(genes):
    """THE TEST THE WHOLE SWEEP RESTS ON.

    The probe reads its exponents off the displacement field the adjoint already
    converged, instead of re-solving per `p`.  That is only sound if the cheap path and
    the real path compute the same thing — so probing at the exponent the constraint was
    built with must return the constraint's own numbers, not merely close ones.  Both go
    through `_stress_aggregate` precisely so this can be asserted at 1e-12 rather than
    hoped for.

    If this fails, no other row of a `--ladder-p` report means anything.
    """
    phases = WO.phase_stencil(n_phase=N_PHASE, scheme="uniform")
    _, rep, _ = so3.score(genes, CFG, phases=phases,
                          probe_p=(WO.STRESS_NOMINAL_P, WA.STRESS_PNORM_P))

    # The reproduction is read off the CONSTRAINT'S OWN exponent, which is now
    # STRESS_NOMINAL_P.  And off `stress_utilisation_kt`, not `stress_utilisation`: the
    # probe still reports the old `c * agg` under the latter name, because `c`'s drift with
    # `p` and with the mesh is exactly what step 1 measured and what the sweep exists to
    # show.  The two keys disagreeing is the finding, not a bug.
    at4 = rep["pnorm_by_p"][repr(float(WO.STRESS_NOMINAL_P))]
    assert at4["pnorm_agg_mpa"] == pytest.approx(rep["pnorm_stress_agg_mpa"], rel=1e-12)
    assert at4["stress_scale_measured"] == pytest.approx(
        rep["stress_scale_measured"], rel=1e-12)
    assert at4["stress_utilisation_kt"] == pytest.approx(
        rep["stress_utilisation"], rel=1e-12)

    # And the sweep is a sweep: a different exponent is a different number, in the
    # direction the p-norm's own docstring claims (it approaches the max from below).
    at30 = rep["pnorm_by_p"][repr(float(WA.STRESS_PNORM_P))]
    assert at4["pnorm_agg_mpa"] < at30["pnorm_agg_mpa"], (
        "the p-norm must be non-decreasing in p; if it is not, the probe is not "
        "evaluating the exponent it was handed")
    assert rep["stress_gauss_p"] == pytest.approx(WO.STRESS_NOMINAL_P)


def test_the_sweep_costs_no_extra_solve(genes, monkeypatch):
    """Five exponents, one evaluation — the measurement that makes `--ladder-p` cheap.

    Re-solving per `p` would turn a fourteen-minute ladder into an hour for values a
    convergence study reads and a gradient nothing reads.  This is the assertion that
    stops that regression, and it is deliberately the same `len(seen) == 1` the rung test
    above makes: the invariant is "one rung, one evaluation", however long the list is.
    """
    seen = []
    real = S3.Evaluator

    class Spy(real):
        def __call__(self, *a, **kw):
            seen.append(self.problem_kw.get("stress_p_probe"))
            return super().__call__(*a, **kw)

    monkeypatch.setattr(S3, "Evaluator", Spy)
    probe = (2.0, 4.0, 8.0, 16.0, 30.0)
    rep = so3.run_mesh_convergence([("shipped", genes)], configs=(CFG,), n_phase=N_PHASE,
                                   probe_p=probe)

    assert len(seen) == 1, (
        f"one rung, five exponents, {len(seen)} evaluations — the probe is re-solving")
    assert seen[0] == probe, "the exponents never reached t3_terms"
    row = rep["designs"][0]["rows"][0]
    assert sorted(row["pnorm_by_p"][k]["p"] for k in row["pnorm_by_p"]) == list(probe)
    assert rep["probe_p"] == list(probe)


def test_the_default_ladder_reports_no_sweep_at_all(genes, monkeypatch):
    """M8b-i.5's report, unchanged, when nobody asks for a sweep.

    `make m8bi5` and every test above it must keep writing the report they always wrote —
    a new argument that quietly adds keys or work to the default path is how a milestone's
    artifact stops being comparable to the one before it.
    """
    phases = WO.phase_stencil(n_phase=N_PHASE, scheme="uniform")
    _, rep, _ = so3.score(genes, CFG, phases=phases)
    assert rep["pnorm_by_p"] == {}, "the default path did work nobody asked for"

    called = []
    real = WA._qoi_pnorm_stress
    monkeypatch.setattr(WA, "_qoi_pnorm_stress",
                        lambda prob, **kw: called.append(kw.get("p")) or real(prob, **kw))
    m = so3.run_mesh_convergence([("shipped", genes)], configs=(CFG,), n_phase=N_PHASE)
    assert m["designs"][0]["series_by_p"] == {}
    assert m["probe_p"] == []
    # One factory per phase for the constraint itself, and not one more.
    assert called == [WO.STRESS_NOMINAL_P] * N_PHASE, (
        f"the default path built {len(called)} p-norm factories for {N_PHASE} phases")


def test_the_per_p_series_are_the_same_extrapolation_the_verdict_was_read_from(genes):
    """`_series_by_p` reuses `_series`, so a swept GCI is comparable to the quoted one.

    M8b-i.5's verdict is a GCI against `GATE_LADDER_GCI`.  A sweep that computed its own
    extrapolation could name a `p` "converged" on a standard the verdict was never held
    to, which is a worse failure than not measuring at all — it would look like progress.
    """
    rows = [{"config": c, "pnorm_by_p": {"8.0": {"pnorm_agg_mpa": v,
                                                 "stress_scale_measured": 1.4,
                                                 "stress_utilisation": 2.0 * v}}}
            for c, v in (("smoke", 1.0), ("coarse", 2.0), ("medium", 2.5))]
    by = so3._series_by_p(rows, [8.0])["8.0"]

    direct = so3._series([{"config": r["config"], "x": v} for r, v in
                          zip(rows, (1.0, 2.0, 2.5))], "x")
    assert by["pnorm"] == direct, "the sweep grew its own extrapolation"
    assert by["p"] == 8.0
    assert by["util"]["values"] == [2.0, 4.0, 5.0]
    # A column that does not move gives no successive-difference ratio, so `_series`
    # declines to call it converged rather than calling it trivially converged.  That
    # matters here: `c` is nearly flat at low `p`, and a sweep that read flatness as
    # convergence would report the constraint as fixed at exactly the exponents where the
    # measurement is least informative.
    assert not np.isfinite(by["c"]["ratio"]) and np.isnan(by["c"]["gci"])
    assert by["c"]["converged"] is False

    # An exponent measured at fewer than two rungs is omitted rather than reported as a
    # series of one — the rule `_series` already applies one level down.
    assert so3._series_by_p(rows[:1], [8.0]) == {}
    assert so3._series_by_p(rows, [99.0]) == {}, "an unmeasured p must not appear"


def test_an_unmeasurable_sweep_says_so_instead_of_reporting_a_verdict():
    """"Nothing converged" and "nothing was measurable" must not share a title.

    A two-rung ladder has no GCI at any exponent, so the sweep panel is empty — and an
    empty panel under the words "no exponent gives the stress p-norm a mesh-independent
    value" asserts a finding the axes hold no evidence for.  M8b-i.5 shipped exactly this
    fault in the other direction (a figure captioned "the stress QoI settles under
    refinement" above a 63% GCI), so the third state is spelled out rather than left to
    fall through to the negative one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def panel(values):
        rows = [{"config": c, "pnorm_by_p": {"8.0": {"pnorm_agg_mpa": v,
                                                     "stress_scale_measured": 1.4,
                                                     "stress_utilisation": v}}}
                for c, v in values]
        d = {"label": "x", "rows": rows,
             "series": {"drop": so3._series([{"config": c, "x": 1.0 + 0.001 * i}
                                             for i, (c, _) in enumerate(values)], "x")},
             "series_by_p": so3._series_by_p(rows, [8.0])}
        fig, ax = plt.subplots()
        so3._plot_by_p(ax, {"designs": [d], "configs": [c for c, _ in values]})
        title = ax.get_title()
        plt.close(fig)
        return title

    two = panel([("smoke", 1.0), ("coarse", 2.0)])
    assert "too short" in two and "3 rungs" in two, (
        f"a two-rung sweep reported a verdict it cannot support: {two!r}")

    # Three rungs that do NOT converge: now the negative verdict is earned.
    diverging = panel([("smoke", 1.0), ("coarse", 2.0), ("medium", 2.5)])
    assert "no exponent" in diverging, diverging

    # Three rungs that DO converge: the title names the exponent.
    converging = panel([("smoke", 1.0), ("coarse", 1.1), ("medium", 1.1001)])
    assert "converges up to p = 8" in converging, converging


def test_the_sweep_verdict_is_read_off_every_design_not_just_the_first():
    """A design that lost a rung must not caption the other one's finding.

    The verdict was scoped to `designs[0]`, so a shipped genome whose `medium` rung failed
    would title the panel "no exponent gives the stress p-norm a mesh-independent value"
    over axes in which elite 1's curve visibly dips under the gate — a title asserting the
    opposite of what is plotted beneath it.  That is the fault M8b-i.5 already shipped once
    and had to regenerate the figure for, and per-row `try` on the ladder makes a design
    with no usable series the EXPECTED case rather than a hypothetical one.

    The design is named in the title because a `p` that converges at one genome and not the
    other is not a constraint, and a bare exponent hides which of the two it came from.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def design(label, values):
        rows = [{"config": c, "pnorm_by_p": {"8.0": {"pnorm_agg_mpa": v,
                                                     "stress_scale_measured": 1.4,
                                                     "stress_utilisation": v}}}
                for c, v in values]
        return {"label": label, "rows": rows,
                "series": {"drop": so3._series(
                    [{"config": c, "x": 1.0 + 0.001 * i}
                     for i, (c, _) in enumerate(values)], "x")},
                "series_by_p": so3._series_by_p(rows, [8.0])}

    rungs = [("smoke", 1.0), ("coarse", 1.1), ("medium", 1.1001)]
    # `best_solution` lost its `medium` rung -> two rungs -> no GCI at any p.
    lost = design("best_solution", rungs[:2])
    assert lost["series_by_p"]["8.0"]["pnorm"]["converged"] is None, (
        "fixture is wrong: a two-rung ladder must have no verdict")
    good = design("elite1", rungs)

    fig, ax = plt.subplots()
    so3._plot_by_p(ax, {"designs": [lost, good], "configs": ["smoke", "coarse", "medium"]})
    title = ax.get_title()
    plt.close(fig)

    assert "converges up to p = 8" in title, (
        f"the second design's convergence was thrown away by the first's missing rung: "
        f"{title!r}")
    assert "elite1" in title, f"the title must say WHICH design converged: {title!r}"


def test_a_converged_pnorm_over_a_divergent_constraint_says_both():
    """The p-norm's verdict is not the constraint's, and M8b-i.6 measured them disagreeing.

    `util = c * pnorm / allowable` is a product.  Lowering `p` converges the p-norm (GCI
    0.45% at p=4 against the axle drop's 0.14%) and makes `c = max/pnorm` WORSE, because
    the true max is a singularity the p-norm walks away from — so the constraint's GCI sits
    at 109% exactly where the nominal is best.  A title reporting only "converges up to
    p = 6" over that panel would be the third caption in this milestone to claim more than
    its axes support, and the one most likely to be believed, since it reads as the good
    news everyone was waiting for.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # pnorm settles hard; c is measured as max/pnorm and the max RUNS AWAY up the ladder,
    # so util inherits the divergence.  This is the shape of the real measurement.
    vals = [("smoke", 1.000, 30.0), ("coarse", 1.010, 45.0), ("medium", 1.012, 60.0)]
    rows = [{"config": c, "pnorm_by_p": {"4.0": {"pnorm_agg_mpa": pn,
                                                 "stress_scale_measured": mx / pn,
                                                 "stress_utilisation": mx / 25.0}}}
            for c, pn, mx in vals]
    d = {"label": "best_solution", "rows": rows,
         "series": {"drop": so3._series([{"config": c, "x": 1.0 + 0.001 * i}
                                         for i, (c, _, _) in enumerate(vals)], "x")},
         "series_by_p": so3._series_by_p(rows, [4.0])}
    b = d["series_by_p"]["4.0"]
    assert b["pnorm"]["converged"] is True, "fixture: the nominal must converge"
    assert b["util"]["converged"] is False, "fixture: the constraint must not"

    fig, ax = plt.subplots()
    so3._plot_by_p(ax, {"designs": [d], "configs": ["smoke", "coarse", "medium"]})
    title = ax.get_title()
    plt.close(fig)

    assert "p-norm converges up to p = 4" in title, title
    assert "CONSTRAINT converges at no p" in title, (
        f"the title reported the converged factor and stayed silent on the divergent "
        f"product it is multiplied into: {title!r}")


def test_the_per_p_table_prints_the_decomposition_and_not_just_the_verdict(capsys):
    """`util = c * pnorm / allowable`, so a table showing only `util` hides half the cause.

    M8b-i.5's decomposition is what made this milestone's next step right: `c` measured at
    GCI 5.33% — the BEST-behaved factor — against the p-norm's 47%, which is why step 2
    lowers `p` rather than deleting the rescale.  `_series_by_p` computed the `c` leaf from
    the start and `_print_by_p` never printed it, so the same table that decided the last
    call could not have been read off this one.
    """
    rows = [{"config": c, "pnorm_by_p": {"8.0": {"pnorm_agg_mpa": v,
                                                 "stress_scale_measured": 1.4 + 0.01 * i,
                                                 "stress_utilisation": v}}}
            for i, (c, v) in enumerate([("smoke", 1.0), ("coarse", 1.1), ("medium", 1.15)])]
    d = {"label": "x", "rows": rows,
         "series": {"drop": so3._series([{"config": c, "x": 1.0} for c, _ in
                                         [("smoke", 0), ("coarse", 0), ("medium", 0)]], "x")},
         "series_by_p": so3._series_by_p(rows, [8.0])}
    so3._print_by_p(d, {})
    out = capsys.readouterr().out

    assert "c GCI" in out and "util GCI" in out, (
        f"both factors of the product must be reported, not just the product: {out!r}")
    # The `c` VALUE, not merely a column header — 1.42 is the `medium` rung's measurement.
    assert "1.420" in out, f"the c column is headed but not filled: {out!r}"
    assert "THE CONTROL" in out, "the axle drop makes every GCI above it interpretable"


def test_an_elite_that_will_not_solve_is_recorded_and_the_screen_carries_on(monkeypatch):
    """One bad genome must not cost the other fifteen.

    `study_objective.py`'s G10 table already treats a failure this way and for the same
    reason: a genome that will not mesh is DATA about the elite set, and the alternative
    is a three-hour screen that dies on elite 1 and reports nothing about elites 2-15.
    """
    calls = []

    def fake_score(genes, cfg=so3.DEFAULT_CONFIG, phases=None, n_phase=4,
                   tiers=("t3",)):
        calls.append(len(calls))
        if len(calls) == 2:
            raise RuntimeError("d(force)/d(indentation) came out -1.0")
        # The full set of report keys the screen reads, because a stub that is missing one
        # turns "this elite would not solve" into "every elite would not solve" — which is
        # the exact failure this test exists to forbid, arriving through the test's own
        # scaffolding instead of through the code.
        return 1.0, {"stress_utilisation": 1.4, "max_stress_mpa": 35.0,
                     "stress_utilisation_hub": 1.4, "stress_utilisation_rim": 1.1,
                     "kt_hub": 1.86, "kt_rim": 1.49,
                     "axle_drop_mean_mm": 1.5}, 0.1

    monkeypatch.setattr(so3, "score", fake_score)
    m = so3.run_multistart(CFG, elites=[np.zeros(14)] * 3, n_probe=0)

    assert [r["elite"] for r in m["elites"]] == [0, 2], (
        "the screen stopped at the failure instead of carrying on past it")
    assert [r["elite"] for r in m["failed"]] == [1]
    assert m["failed"][0]["failed"] == "RuntimeError"
    assert m["verdict"]["n_elites_scored"] == 2
    assert m["verdict"]["n_elites_failed"] == 1
    assert m["pass"] is True, (
        "a failed genome is a measurement, not a broken screen; the gate is whether the "
        "screen RAN")
