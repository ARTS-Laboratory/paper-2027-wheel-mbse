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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import jax_config  # noqa: E402,F401

import study_stage3 as so3  # noqa: E402
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


def test_stress_scale_is_the_previous_steps_measurement_exactly(z0):
    """S4 at `smoke`.  This is M8a's gate-7 lesson made into a run-time invariant.

    `c` re-measured inside a call makes value and gradient answers to different
    questions — measured, 10% into the assembled gradient while every individual term
    still matched its own finite difference to 1e-8.  Held fixed forever it stops
    tracking the design.  The policy is "fixed within a step, refreshed between", and
    this is the identity that pins it.
    """
    rec = S3.descend(z0, CFG, steps=2, n_phase=N_PHASE, scheme="uniform", verbose=False)
    live = [r for r in rec["steps"] if not r["abandoned"]]
    checked = 0
    for k in range(1, len(live)):
        used = live[k]["report"]["stress_scale"]
        prev = live[k - 1]["report"]["stress_scale_measured"]
        rel = abs(used - prev) / abs(prev)
        assert rel <= so3.GATE_SCALE_REL, (
            f"step {live[k]['step']} was scored with c={used} but the previous step "
            f"measured {prev}; re-read the stress_scale paragraph in "
            f"wheel_objective.t3_terms")
        checked += 1
    assert checked >= 1, "the run produced no consecutive evaluated steps to check"


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
