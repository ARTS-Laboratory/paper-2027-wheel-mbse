"""
=============================================================================
  STAGE 3 — PROJECTED ADAM ON THE FEA OBJECTIVE
=============================================================================

    .venv-opt/bin/python src/wheel_stage3.py --steps 60
    .venv-opt/bin/python src/wheel_stage3.py --optimizer lbfgsb --phase-scheme uniform

M8a built the scalar and proved its gradient (`wheel_objective.py`, eleven gates, total
gradient against a finite-difference ladder of the whole pipeline at 1.0e-6).  This is
the thing that descends it.  It is a DRIVER AND NOTHING ELSE: no new physics, no new
gradient, no new term.  Every line below is loop, projection, bookkeeping or a policy
decision about what to do when the solver refuses.

WHY A DRIVER NEEDS A DOCSTRING THIS LONG
----------------------------------------
Five of M8a's measured findings are load-bearing HERE rather than there, and each one
has the same failure signature if it is got wrong: the run does not crash, it converges
somewhere slightly wrong, and the report looks exactly like a hard design problem.  That
is the failure mode this project keeps naming and keeps having to catch with a gate.

1.  THE STRESS CONSTRAINT IS A PURE FUNCTION OF THE GENES, AND NOTHING IS FROZEN.
    It used to rescale a p-norm to the true max by a measured ratio `c`, and because that
    rescale is exact only for a CONSTANT factor, each evaluation had to be handed a `c`
    measured at the PREVIOUS point — ordinary adaptive constraint scaling, whose only
    unusual property was that getting it wrong was invisible.  M8a's gate 7 was the
    story: every individual term agreed with its own finite difference to 1e-8 while the
    ASSEMBLED gradient was out by 1e-1, purely because `c` moved inside the difference.

    M8b-i.6 removed the need.  `c` is anchored to the true max, which is M4's crack-tip
    singularity and diverges under mesh refinement, so the frozen number was a converged
    answer to nothing; the constraint is now `Kt(R, t) * sigma_nominal(p=4)`, in which
    both factors are mesh-convergent and `Kt` is differentiated rather than held still
    (`wheel_objective.t3_terms`).  So there is no state to carry between steps, both legs
    of every difference re-evaluate freely, and gate 7 tests the product rule instead of
    the plumbing.  `stress_scale_measured` survives in the report as a diagnostic only —
    it is the evidence for the change, not an input to anything.

2.  THE PHASE STENCIL IS DRAWN FROM A FIXED LATTICE.
    `wheel_wheel.coord_fn` keys its jit cache on `float(phase)`, so a continuously
    random offset misses on every phase of every step and pays M7's measured 0.774 s
    re-trace eight times per step — roughly double the actual solving.
    `wheel_objective.phase_stencil` quantizes the random offset onto an `n_phase*n_sub`
    grid for exactly this reason, and `_COORD_FN_CACHE_MAX` is already 128 to hold it
    (`wheel_wheel.py:897`).  Randomising at all buys something specific: the contact
    facets on the rim discretisation at 0.25 degrees while the stencil samples at 3.75,
    so the artefact ALIASES, and under a fixed stencil it is a deterministic function of
    the genes with a nonzero gradient — precisely what an optimizer chases.  A random
    shift makes it zero-mean noise instead.

3.  FLANK ORIENTATION IS PINNED FOR THE WHOLE RUN.
    `WW.flank_orientation` is a discrete topological decision about which side of the
    ring each spoke flank arrives on.  `build_wheel` re-derives it from the genes unless
    it is passed, and a step that flips it changes which block owns which node — the
    gradient of that step does not exist.  So it is evaluated once at the start and
    threaded through `phase_meshes(orientation=...)` forever after.  It is ALSO
    re-derived every step for reporting only, and a disagreement is recorded as an
    `orientation_flip` event: the run keeps going on the pinned topology, and the event
    says the trajectory has walked into a region where the frozen answer is stale.

4.  WARM STARTING IS FREE AND IS TAKEN.
    `service_qoi_value_and_grad`'s `delta0` is the secant's indentation, and
    `wheel_adjoint.py:537` shows that is the same number reported as `axle_drop_mm`.
    So the next step's warm vector is just the last step's per-phase drops, read off
    `breakdown["report"]["rows"]` and passed back as `warm=`.  `delta0=None` costs an
    extra linear `solve_wheel` per phase to manufacture a starting guess
    (`wheel_fem.py:1841`), and between two Adam steps the design barely moves, so the
    previous answer is a far better guess than anything a cold solve can produce.

5.  A FAILED SOLVE IS A STEP REJECT, NOT A CRASH, AND NOT A ZERO.
    `wheel_objective.py` contains no `try`/`except` anywhere, deliberately:
    `NewtonDivergedError` (`wheel_fem.py:1090`) and the `dF/ddelta <= 0` guard
    (`wheel_adjoint.py:553`) propagate straight out, and the caller owns the policy.
    The policy is here.  A trial step that will not solve is halved and retried, up to
    `--max-rejects`; on exhaustion the iterate is restored and the learning rate decays.
    This matters beyond robustness: `solve_nonlinear` raising because `r @ du >= 0` is
    the tangent losing positive definiteness, which is the only buckling signal this
    repo owns until M9 replaces the Euler proxy with `lambda_min(K_t)`.  Swallowing it
    as a large loss would let the optimizer walk into a buckled design and score it.

WHY PROJECTION AND NOT A SIGMOID
--------------------------------
`z <- clip(z - step, 0, 1)`, not `z = sigmoid(y)`.  Six of the fourteen genes sit
exactly on a bound at the shipped genome — `cy2` at its high bound, `t1`/`t2`/`t3` at
`MIN_WALL_MM`, `R_rim` at its high bound — and a squashing reparameterisation has
vanishing gradient there, so the run would begin with six genes frozen.  Projection has
no such pathology: a gene on a bound whose gradient points inward moves immediately.
`study_stage3.py`'s S3 checks that rather than assuming it.

WHAT THIS RUN IS EXPECTED TO FIND
---------------------------------
M8a measured the shipped design at 513.2 objective units against the GA's 50.4, missing
its deflection target by -26.3% AND exceeding its stress allowable by 24%.  Stage 3 is
therefore being asked for more compliance with negative stress headroom, on a wheel
whose rim band carries a third of the compliance (M4's `compliance_split.rim = 0.324`)
and which no gene touches.  Whether that problem has a feasible point at all is an open
question with numbers attached, and `study_stage3.py`'s feasibility probe answers it
before anyone funds a production run.  No weight, target or allowable is adjusted here
to make the trajectory look better; the infeasibility is the measurement.
=============================================================================
"""

import argparse
import json
import math
import os

import project_paths as PP   # stdlib-only; safe in the jax-free CAD env
import time

import jax_config  # noqa: F401  — must precede every other jax import
import numpy as np
import scipy.optimize

import wheel_fea as W
import wheel_genome as wg
import wheel_objective as WO
import wheel_wheel as WW

HERE = PP.ROOT          # run outputs and the genome live at the repo root
DEFAULT_CONFIG = "coarse"

# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------

DEFAULT_STEPS = 60
DEFAULT_LR = 0.01               # normalised units; the box is [0,1] in every gene
GRAD_CLIP = 1.0                 # global norm, in the same normalised units
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
MAX_REJECTS = 3
DEFAULT_N_PHASE, DEFAULT_N_SUB = 8, 8

# The T1 terms that are FEASIBILITY BARRIERS rather than objectives.  `smoothness` is a
# preference and `buckling` has no gradient, so neither belongs in a reject rule.
T1_BARRIER_NAMES = ("x_order", "hub_overlap", "fold", "arrival", "fillet")

# The GROSS-infeasibility threshold, and the number is the whole lesson.
#
# This was 1.0, with the rule "reject a trial whose barrier sum exceeds the current
# point's" — i.e. never step further into the wall.  That rule DEADLOCKED the stress-only
# feasibility probe at `coarse`: the shipped design carries `hub_overlap` = 52.13, the
# stress-reducing direction raises it to 55.41, and halving the step only walks it back to
# 52.54 — still above 52.13.  Every trial at every scale was refused, all forty steps were
# abandoned, and the probe reported its starting point as its answer.  194 events, zero
# accepted steps, thirty-five minutes of wall clock spent standing still.
#
# The error was conceptual, not a mis-set constant.  Every barrier is ALREADY a term in
# the objective, with a weight chosen to price it; `hub_overlap` at 500 x (violation)^2 is
# how the optimizer is told.  A pre-check that also forbids any increase duplicates that
# job and overrides it, and what it overrides is exactly the constrained trade the run
# exists to make — buying stress with a little hub overlap is a decision the weights are
# there to arbitrate, not one a screen should veto.
#
# So the screen keeps only the job the objective cannot do: refusing to spend a solve on
# geometry that will not mesh.  1e4 is that scale.  Healthy barriers here are tens
# (`hub_overlap` 52 at the shipped genome); M8a's analytically infeasible fillet scored
# 112,000.  `max(T1_REJECT, b_here)` survives as a deadlock escape, and with an absolute
# threshold this high it binds only when the design is already broken.
T1_REJECT = 1.0e4

# An abandonment storm must terminate and say so.  Halving `lr` on every abandoned step
# drives it to zero geometrically, so a deterministic refusal — which is what the deadlock
# above was — freezes the run silently rather than stopping it.
LR_FLOOR_FRAC = 1e-3            # lr never decays below this fraction of its initial value
MAX_CONSECUTIVE_ABANDONED = 5   # ... and this many in a row ends the run, with an event

# Which `report` scalars are worth carrying into every step record.  Cheap, and they are
# what a checkpoint is read for.
REPORT_KEYS = ("axle_drop_mean_mm", "max_stress_mpa", "stress_utilisation",
               "stress_utilisation_hub", "stress_utilisation_rim", "kt_hub", "kt_rim",
               "phase_ripple_std_over_mean", "mesh_mass_g", "min_scaled_jacobian",
               "stress_scale_measured", "buckling_ratio")


# ---------------------------------------------------------------------------
# THE OPTIMIZER, AS PURE FUNCTIONS
# ---------------------------------------------------------------------------

def cosine_lr(lr0, i, n_steps):
    """Cosine decay from `lr0` at step 0 to 0 at step `n_steps`."""
    if n_steps <= 0:
        return lr0
    return lr0 * 0.5 * (1.0 + math.cos(math.pi * min(i, n_steps) / n_steps))


def clip_global_norm(g, max_norm=GRAD_CLIP):
    """Scale `g` down so `||g|| <= max_norm`.  Returns `(g_clipped, norm_before)`."""
    g = np.asarray(g, dtype=float)
    norm = float(np.linalg.norm(g))
    if max_norm is None or max_norm <= 0 or norm <= max_norm or norm == 0.0:
        return g, norm
    return g * (max_norm / norm), norm


def adam_update(g, m, v, t, lr, b1=ADAM_B1, b2=ADAM_B2, eps=ADAM_EPS):
    """One Adam step.  Returns `(delta, m_new, v_new)` with `t` 1-based.

    A pure function on purpose: `tests/test_stage3.py` reproduces it by hand, which is
    only possible if no state is hidden in an object.
    """
    g = np.asarray(g, dtype=float)
    m = b1 * m + (1.0 - b1) * g
    v = b2 * v + (1.0 - b2) * g * g
    m_hat = m / (1.0 - b1 ** t)
    v_hat = v / (1.0 - b2 ** t)
    return lr * m_hat / (np.sqrt(v_hat) + eps), m, v


def project(z):
    """The unit box.  `wheel_genome.clip_to_bounds` with the bounds already normalised."""
    return wg.clip_to_bounds(np.asarray(z, dtype=float), 0.0, 1.0)


# ---------------------------------------------------------------------------
# EVALUATION — THE PER-STEP PROTOCOL
# ---------------------------------------------------------------------------

def warm_from(breakdown):
    """The per-phase indentations to start the next step's secant from.

    `service_qoi_value_and_grad` takes `delta0`, the secant's initial indentation, and
    `wheel_adjoint.py:537` sets `delta = float(sec["axle_drop_mm"])` — the two are the
    same number, so the drops already in the report ARE the warm vector.  Returns `None`
    when there is no T3 block to read, which is the correct input for a cold start.
    """
    rows = breakdown.get("report", {}).get("rows")
    if not rows:
        return None
    return [float(r["axle_drop_mm"]) for r in rows]


def barrier_sum_of(breakdown):
    """The feasibility barriers, read off a breakdown that has already been paid for."""
    t = breakdown["terms"]
    return float(sum(t[k]["value"] for k in T1_BARRIER_NAMES if k in t))


def t1_barrier_sum(z, cfg, weights=None, span_mm=W.S):
    """The feasibility barriers only, with no mesh and no solve.

    `tiers=("t1",)` is the only genuinely mesh-free AND solve-free path — `("t1","t2")`
    still pays a `build_wheel`.

    AND IT IS NOT MICROSECONDS, WHICH IS WORTH KNOWING BEFORE RELYING ON IT — though the
    conclusion is narrower than it first looked, and both halves are worth stating.

    Measured: ~9 s on the first call and a steady ~1.1 s at `coarse` (1.58 s at `smoke`)
    on every call after it, so the cost is real work rather than a trace that amortises.
    `t1_vector` carries no `@jax.jit`, so `jax.jacrev` runs it eagerly and pays python
    dispatch on every op of a several-hundred-point Bezier evaluation.  Against
    `wheel_objective.py:33`'s "~ms" for the whole tier that is three orders out.

    But the TIERING ARGUMENT is about a ratio, and the ratio holds where it is used: T1
    is 0.73% of a full `coarse` evaluation, so refusing a trial for 1 s instead of
    spending 144 s on it is still the trade M8a described.  It is at `smoke` that the
    same second lands against a 7 s solve and becomes 21% — a statement about how cheap
    the mesh is, not about T1.  Quoting either number alone misleads.

    The driver's response is a design decision rather than a complaint.  The current
    point's barrier is read off the breakdown the step already computed
    (`barrier_sum_of`), which was half the cost and is now free; what remains is one call
    per TRIAL, and `--no-t1-precheck` turns even that off.  S10 reports both numbers so
    the trade is visible rather than assumed.  Jitting `t1_vector` would fix it properly
    and belongs in `wheel_objective.py`, which M8b-i does not touch.
    """
    _, _, brk = WO.objective(z, cfg, normalized=True, weights=weights,
                             span_mm=span_mm, tiers=("t1",))
    return barrier_sum_of(brk), brk


class Evaluator:
    """One `objective` call, with the five protocol decisions applied.

    Holds only what must persist BETWEEN steps: the pinned orientation.  Everything that
    varies per call — the iterate, the phases, the warm vector, the tiers — is an
    argument, which is what lets the gate drive it directly and lets `_FaultyEvaluator`
    stand in for it.

    It used to also carry a `stress_scale` refreshed between steps; docstring item 1 says
    why that is gone and why nothing replaced it.
    """

    def __init__(self, cfg=DEFAULT_CONFIG, *, weights=None, orientation=None,
                 span_mm=W.S, **problem_kw):
        self.cfg = cfg
        self.weights = weights
        self.span_mm = span_mm
        self.orientation = orientation
        self.problem_kw = problem_kw
        self.n_calls = 0
        self.mesh_s = 0.0
        self.solve_s = 0.0

    def genes(self, z, low, high):
        return wg.denormalize(np.asarray(z, dtype=float), low, high)

    def __call__(self, z, low, high, *, phases, warm=None, tiers=("t1", "t2", "t3")):
        """`(value, grad_z, breakdown)` at `z`, gradient in normalised units."""
        genes = self.genes(z, low, high)

        t0 = time.time()
        meshes = None
        if "t2" in tiers or "t3" in tiers:
            meshes = WO.phase_meshes(genes, self.cfg, phases,
                                     orientation=self.orientation)
        self.mesh_s += time.time() - t0

        t0 = time.time()
        val, grad, brk = WO.objective(
            z, self.cfg, normalized=True, weights=self.weights, phases=phases,
            meshes=meshes, tiers=tiers, span_mm=self.span_mm,
            warm=warm, **self.problem_kw)
        self.solve_s += time.time() - t0
        self.n_calls += 1
        return val, grad, brk


# ---------------------------------------------------------------------------
# THE DESCENT
# ---------------------------------------------------------------------------

def descend(z0, cfg=DEFAULT_CONFIG, *, steps=DEFAULT_STEPS, lr=DEFAULT_LR, weights=None,
            n_phase=DEFAULT_N_PHASE, n_sub=DEFAULT_N_SUB, scheme="rqmc", seed=0,
            grad_clip=GRAD_CLIP, max_rejects=MAX_REJECTS, t1_reject=T1_REJECT,
            log_every=10, out=None, orientation=None, span_mm=W.S, verbose=True,
            evaluator=None, warm_start=True, t1_precheck=True, **problem_kw):
    """Projected Adam in the unit box.  Returns the run record.

    One `objective` evaluation per accepted step, plus one microsecond T1 call per trial.
    """
    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    z = project(z0)
    rng = np.random.default_rng(seed)

    # `flank_orientation` wants the resolved WheelConfig, not its name.
    wcfg = WW.get_config(cfg)
    if orientation is None:
        orientation = WW.flank_orientation(wg.denormalize(z, low, high), wcfg,
                                           span_mm=span_mm)
    orientation = tuple(float(o) for o in orientation)

    # `evaluator` is an injection point for the gate: S5 has to prove the step-reject
    # path recovers, and waiting for a real divergence to turn up is not a test.
    ev = evaluator if evaluator is not None else Evaluator(
        cfg, weights=weights, orientation=orientation, span_mm=span_mm, **problem_kw)

    events, step_rows = [], []
    t_start = time.time()

    def _draw():
        return WO.phase_stencil(n_phase=n_phase, n_sub=n_sub, scheme=scheme, rng=rng)

    # -- step 0: the starting point, scored.
    phases = _draw()
    t0 = time.time()
    val, grad, brk = ev(z, low, high, phases=phases)
    wall = time.time() - t0
    warm = warm_from(brk) if warm_start else None
    best = {"loss": val, "z": z.copy(), "step": 0, "breakdown": brk}
    step_rows.append(_row(0, val, grad, lr, brk, phases, wall, 0, z=z))
    if verbose:
        _log_step(0, val, grad, lr, brk, wall)
        WO.print_loss_breakdown(brk, "STAGE 3 — START")

    m = np.zeros_like(z)
    v = np.zeros_like(z)
    n_reject = 0
    lr0, n_consecutive = lr, 0

    for i in range(1, steps + 1):
        lr_t = cosine_lr(lr, i - 1, steps)
        g_clipped, g_norm = clip_global_norm(grad, grad_clip)
        delta, m_try, v_try = adam_update(g_clipped, m, v, i, lr_t)

        # Free: the accepted step already evaluated T1 as part of the objective.
        b_here = barrier_sum_of(brk)
        scale, accepted = 1.0, False
        z_trial, val_new, grad_new, brk_new, wall = z, val, grad, brk, 0.0

        for attempt in range(max_rejects + 1):
            z_trial = project(z - scale * delta)

            # -- the refusal, before any mesh is built.  GROSS infeasibility only: the
            # barriers are already terms in the objective, so a screen that also forbids
            # increasing them overrides the weights and vetoes the constrained trade the
            # run exists to make.  See `T1_REJECT` for the deadlock that taught this.
            if t1_precheck:
                b_trial, _ = t1_barrier_sum(z_trial, cfg, weights=weights,
                                            span_mm=span_mm)
                if b_trial > max(t1_reject, b_here):
                    events.append({"step": i, "kind": "t1_reject", "attempt": attempt,
                                   "scale": scale, "barrier": b_trial,
                                   "barrier_here": b_here})
                    scale *= 0.5
                    continue

            phases = _draw()
            t0 = time.time()
            try:
                val_new, grad_new, brk_new = ev(z_trial, low, high, phases=phases,
                                                warm=warm)
            except RuntimeError as err:
                # NewtonDivergedError, the secant's stall, or dF/ddelta <= 0.  All three
                # mean "this design did not solve", which is information, not noise.
                events.append({
                    "step": i, "kind": "solve_reject", "attempt": attempt,
                    "scale": scale, "error": type(err).__name__, "message": str(err)[:400],
                    "n_newton_records": len(getattr(err, "history", []) or [])})
                scale *= 0.5
                warm = None          # the previous indentations are suspect now
                continue
            wall = time.time() - t0
            accepted = True
            break

        if not accepted:
            # Every trial refused.  Keep the iterate, decay the rate, do not update the
            # moments with a step that was never taken.
            n_reject += 1
            n_consecutive += 1
            lr = max(lr * 0.5, lr0 * LR_FLOOR_FRAC)
            events.append({"step": i, "kind": "step_abandoned", "lr_after": lr,
                           "consecutive": n_consecutive})
            step_rows.append(_row(i, val, grad, lr_t, brk, phases, 0.0, n_reject,
                                  z=z, abandoned=True))
            if verbose:
                print(f"[step {i:4d}] ABANDONED after {max_rejects + 1} trials; "
                      f"lr -> {lr:.4g}")
            _persist(out, cfg, z, low, high, ev, step_rows, events, best, t_start,
                     locals_scheme=scheme, seed=seed, n_phase=n_phase, steps=steps)
            if n_consecutive >= MAX_CONSECUTIVE_ABANDONED:
                # Standing still is a result; standing still for the remaining budget is
                # only a waste.  Stop, and make the reason a record rather than a shape
                # in the loss history someone has to squint at.
                events.append({"step": i, "kind": "run_stopped_stuck",
                               "consecutive_abandoned": n_consecutive,
                               "steps_remaining": steps - i})
                if verbose:
                    print(f"[step {i:4d}] STOPPING: {n_consecutive} consecutive "
                          f"abandoned steps, {steps - i} of {steps} not attempted")
                break
            continue

        z, val, grad, brk = z_trial, val_new, grad_new, brk_new
        m, v = m_try, v_try
        n_consecutive = 0
        warm = warm_from(brk) if warm_start else None

        live = WW.flank_orientation(wg.denormalize(z, low, high), wcfg, span_mm=span_mm)
        if tuple(float(o) for o in live) != orientation:
            events.append({"step": i, "kind": "orientation_flip",
                           "pinned": list(orientation),
                           "live": [float(o) for o in live]})

        if val < best["loss"]:
            best = {"loss": val, "z": z.copy(), "step": i, "breakdown": brk}

        step_rows.append(_row(i, val, grad, lr_t, brk, phases, wall, n_reject,
                              z=z, scale=scale, grad_norm_raw=g_norm))
        if verbose and (i % log_every == 0 or i == steps):
            _log_step(i, val, grad, lr_t, brk, wall)
        _persist(out, cfg, z, low, high, ev, step_rows, events, best, t_start,
                 locals_scheme=scheme, seed=seed, n_phase=n_phase, steps=steps)

    record = _record(cfg, z, low, high, ev, step_rows, events, best, t_start,
                     scheme=scheme, seed=seed, n_phase=n_phase, steps=steps)
    if out:
        _write(out, record)
    return record


def _row(i, val, grad, lr, brk, phases, wall, n_reject, z=None, scale=1.0,
         grad_norm_raw=None, abandoned=False):
    r = {"step": i, "loss": float(val), "grad_norm": float(np.linalg.norm(grad)),
         "lr": float(lr), "wall_s": round(float(wall), 3), "trial_scale": float(scale),
         "n_reject_cumulative": int(n_reject), "abandoned": bool(abandoned),
         # The iterate itself.  14 floats per step is nothing, and without it the
         # trajectory cannot be audited after the fact — S3 checks that a gene pinned to
         # a bound actually leaves it, which is only answerable from the z history.
         "z": None if z is None else [float(x) for x in np.asarray(z).ravel()],
         # The gradient too, for the same reason: S3 reconstructs the projection step
         # offline and cannot do that from a norm.
         "grad": [float(x) for x in np.asarray(grad).ravel()],
         "phase_deg": [float(p) for p in np.asarray(phases).ravel()],
         "terms": {k: {"value": d["value"], "grad_norm": d["grad_norm"],
                       "share": d["share"], "grad_share": d["grad_share"]}
                   for k, d in brk["terms"].items()}}
    if grad_norm_raw is not None:
        r["grad_norm_before_clip"] = float(grad_norm_raw)
    rep = brk.get("report", {})
    r["report"] = {k: float(rep[k]) for k in REPORT_KEYS if k in rep}
    return r


def _log_step(i, val, grad, lr, brk, wall):
    rep = brk.get("report", {})
    print(f"[step {i:4d}] loss {val:12.4f}  |grad| {np.linalg.norm(grad):10.4g}  "
          f"lr {lr:8.5f}  drop {rep.get('axle_drop_mean_mm', float('nan')):6.3f} mm  "
          f"util {rep.get('stress_utilisation', float('nan')):6.3f}  {wall:5.2f} s")


def _record(cfg, z, low, high, ev, step_rows, events, best, t_start, *, scheme, seed,
            n_phase, steps):
    genes = wg.denormalize(z, low, high)
    best_genes = wg.denormalize(best["z"], low, high)
    return {
        "settings": {"config": cfg if isinstance(cfg, str) else cfg.name,
                     "optimizer": "adam", "steps": steps, "n_phase": n_phase,
                     "phase_scheme": scheme, "seed": seed,
                     "service_force_n": W.TOTAL_FORCE_NEWTONS,
                     "target_deflection_mm": WO.TARGET_DEFLECTION_MM,
                     "allowable_stress_mpa": WO.ALLOWABLE_STRESS_MPA,
                     "elapsed_s": round(time.time() - t_start, 1),
                     "orientation": [float(o) for o in (ev.orientation or ())],
                     "n_objective_calls": ev.n_calls,
                     "mesh_s": round(ev.mesh_s, 1), "solve_s": round(ev.solve_s, 1)},
        "steps": step_rows,
        "events": events,
        "final": {"z": [float(x) for x in z],
                  "genes": wg.vector_to_genes(genes),
                  "genome_hash": wg.genome_hash(wg.vector_to_genes(genes)),
                  "loss": step_rows[-1]["loss"] if step_rows else None,
                  "bound_saturation": [list(t) for t in
                                       wg.bound_saturation(genes, low, high)]},
        "best": {"step": best["step"], "loss": float(best["loss"]),
                 "z": [float(x) for x in best["z"]],
                 "genes": wg.vector_to_genes(best_genes),
                 "genome_hash": wg.genome_hash(wg.vector_to_genes(best_genes)),
                 "terms": {k: d["value"] for k, d in best["breakdown"]["terms"].items()},
                 "report": {k: float(v) for k, v in best["breakdown"]["report"].items()
                            if isinstance(v, (int, float))}},
    }


def _persist(out, cfg, z, low, high, ev, step_rows, events, best, t_start, *,
             locals_scheme, seed, n_phase, steps):
    """Write the trajectory after every step.

    The GA has no checkpointing at all — `wheel_fea.py` persists only after `ga.run()`
    returns — and a run measured in tens of minutes must survive a kill.
    """
    if not out:
        return
    _write(out, _record(cfg, z, low, high, ev, step_rows, events, best, t_start,
                        scheme=locals_scheme, seed=seed, n_phase=n_phase, steps=steps))


def _write(out, record):
    path = out if os.path.isabs(out) else os.path.join(HERE, out)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(record, fh, indent=1)
    os.replace(tmp, path)           # atomic, so a kill mid-write cannot truncate the run
    return path


# ---------------------------------------------------------------------------
# THE DETERMINISTIC ALTERNATIVE
# ---------------------------------------------------------------------------

def descend_lbfgsb(z0, cfg=DEFAULT_CONFIG, *, steps=DEFAULT_STEPS, weights=None,
                   n_phase=DEFAULT_N_PHASE, scheme="uniform", orientation=None,
                   span_mm=W.S, out=None, verbose=True, **problem_kw):
    """`scipy.optimize.minimize(method="L-BFGS-B")` over the same closure.

    `scheme` MUST be `uniform`.  A quasi-Newton method builds its curvature model from
    differences of gradients taken at different points; if the phase stencil moves
    between those points, the difference contains the stencil's shift as well as the
    design's, and the model is being fitted to noise.  Adam tolerates that by design and
    L-BFGS-B does not, so this raises rather than silently producing a bad Hessian.
    """
    if scheme != "uniform":
        raise ValueError(
            f"L-BFGS-B needs a deterministic objective; phase scheme {scheme!r} makes "
            f"the gradient stochastic and poisons the curvature model. Use "
            f"--phase-scheme uniform, or --optimizer adam.")

    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    z0 = project(z0)
    if orientation is None:
        orientation = WW.flank_orientation(wg.denormalize(z0, low, high),
                                           WW.get_config(cfg), span_mm=span_mm)
    orientation = tuple(float(o) for o in orientation)

    ev = Evaluator(cfg, weights=weights, orientation=orientation, span_mm=span_mm,
                   **problem_kw)
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")

    events, step_rows, state = [], [], {"warm": None, "best": None, "t0": time.time()}

    def fun(z):
        t0 = time.time()
        try:
            val, grad, brk = ev(z, low, high, phases=phases, warm=state["warm"])
        except RuntimeError as err:
            # L-BFGS-B has no reject hook, so a failed solve is reported as a large
            # value with a zero gradient, which its line search reads as "shorten the
            # step".  Recorded as an event so it can never be mistaken for a real point.
            events.append({"step": len(step_rows), "kind": "solve_reject",
                           "error": type(err).__name__, "message": str(err)[:400]})
            state["warm"] = None
            return 1e12, np.zeros_like(np.asarray(z, dtype=float))
        state["warm"] = warm_from(brk)
        i = len(step_rows)
        step_rows.append(_row(i, val, grad, float("nan"), brk, phases,
                              time.time() - t0, 0, z=z))
        if state["best"] is None or val < state["best"]["loss"]:
            state["best"] = {"loss": val, "z": np.asarray(z, dtype=float).copy(),
                             "step": i, "breakdown": brk}
        if verbose:
            _log_step(i, val, grad, float("nan"), brk, time.time() - t0)
        return float(val), np.asarray(grad, dtype=float)

    res = scipy.optimize.minimize(fun, z0, jac=True, method="L-BFGS-B",
                                  bounds=[(0.0, 1.0)] * wg.N_GENES,
                                  options={"maxiter": steps})
    z = project(res.x)
    record = _record(cfg, z, low, high, ev, step_rows, events, state["best"],
                     state["t0"], scheme=scheme, seed=None, n_phase=n_phase, steps=steps)
    record["settings"]["optimizer"] = "lbfgsb"
    record["scipy"] = {"success": bool(res.success), "status": int(res.status),
                       "message": str(res.message), "nit": int(res.nit),
                       "nfev": int(res.nfev)}
    if out:
        _write(out, record)
    return record


# ---------------------------------------------------------------------------
# STARTING POINTS
# ---------------------------------------------------------------------------

def load_genes(path="best_solution.json"):
    with open(os.path.join(HERE, path)) as fh:
        return np.array(list(json.load(fh)["genes"].values()), dtype=float)


def load_elites(path="stage2_elites.json", limit=16):
    """The Stage-2 final population's distinct genomes, best first.

    Duplicated from `study_objective.load_elites` rather than imported: a library module
    importing a study script would invert the dependency this repo keeps one-way.
    """
    p = path if os.path.isabs(path) else os.path.join(HERE, path)
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        rec = json.load(fh)
    return [np.array(list(e["genes"].values()), dtype=float)
            for e in rec.get("elites", [])[:limit]]


def start_points(spec, genome="best_solution.json", elites="stage2_elites.json"):
    """`best` | `rank:N` | `all` -> a list of `(label, genes)`."""
    if spec == "best":
        return [("best_solution", load_genes(genome))]
    if spec == "all":
        pop = load_elites(elites)
        if not pop:
            raise FileNotFoundError(
                f"{elites} not found — run `make elites` to write it, or use --start best")
        return [(f"elite{k}", g) for k, g in enumerate(pop)]
    if spec.startswith("rank:"):
        k = int(spec.split(":", 1)[1])
        pop = load_elites(elites, limit=k + 1)
        if len(pop) <= k:
            raise IndexError(f"{elites} has {len(pop)} elites, no rank {k}")
        return [(f"elite{k}", pop[k])]
    raise ValueError(f"unknown --start {spec!r}; expected 'best', 'all' or 'rank:N'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="M8b Stage-3 descent on the FEA objective")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--optimizer", choices=("adam", "lbfgsb"), default="adam")
    ap.add_argument("--phase-scheme", choices=("rqmc", "uniform", "iid"), default="rqmc",
                    help="rqmc draws the stencil offset from the n_phase*n_sub lattice, "
                         "which keeps coord_fn's jit cache hitting")
    ap.add_argument("--n-phase", type=int, default=DEFAULT_N_PHASE)
    ap.add_argument("--n-sub", type=int, default=DEFAULT_N_SUB)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    ap.add_argument("--max-rejects", type=int, default=MAX_REJECTS)
    ap.add_argument("--t1-reject", type=float, default=T1_REJECT)
    ap.add_argument("--no-t1-precheck", action="store_true",
                    help="skip the gene-space barrier screen on each trial step; it is "
                         "measured at ~20%% of an evaluation, not the microseconds the "
                         "tiering argument assumes (see t1_barrier_sum)")
    ap.add_argument("--start", default="best", help="best | rank:N | all")
    ap.add_argument("--genome", default="best_solution.json")
    ap.add_argument("--elites", default="stage2_elites.json")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", default="stage3_run.json")
    ap.add_argument("--best-out", default="stage3_best.json")
    args = ap.parse_args()

    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    starts = start_points(args.start, args.genome, args.elites)
    runs = []

    for label, genes0 in starts:
        z0 = wg.normalize(genes0, low, high)
        out = args.out if len(starts) == 1 else \
            f"{os.path.splitext(args.out)[0]}_{label}.json"
        print(f"\n{'=' * 78}\n  STAGE 3 — {label}  ({args.optimizer}, {args.config}, "
              f"{args.steps} steps, {args.phase_scheme})\n{'=' * 78}")
        print(wg.describe(genes0, low, high))

        if args.optimizer == "lbfgsb":
            rec = descend_lbfgsb(z0, args.config, steps=args.steps,
                                 n_phase=args.n_phase, scheme=args.phase_scheme,
                                 out=out)
        else:
            rec = descend(z0, args.config, steps=args.steps, lr=args.lr,
                          n_phase=args.n_phase, n_sub=args.n_sub,
                          scheme=args.phase_scheme, seed=args.seed,
                          grad_clip=args.grad_clip, max_rejects=args.max_rejects,
                          t1_reject=args.t1_reject, t1_precheck=not args.no_t1_precheck,
                          log_every=args.log_every, out=out)
        rec["label"] = label
        runs.append(rec)
        print(f"\nwrote {out}  ({rec['settings']['elapsed_s']} s, "
              f"{rec['settings']['n_objective_calls']} objective calls)")

    best = min(runs, key=lambda r: r["best"]["loss"])
    print(f"\n{'=' * 78}\n  BEST OVER {len(runs)} START(S): {best['label']} at step "
          f"{best['best']['step']}, loss {best['best']['loss']:.4f}\n{'=' * 78}")
    print(wg.describe(np.array(list(best["best"]["genes"].values())), low, high))

    h = wg.save_record(
        os.path.join(HERE, args.best_out), best["best"]["genes"],
        source="wheel_stage3.py",
        search={"optimizer": args.optimizer, "config": args.config,
                "steps": args.steps, "phase_scheme": args.phase_scheme,
                "n_phase": args.n_phase, "seed": args.seed, "start": args.start,
                "label": best["label"], "at_step": best["best"]["step"]},
        loss_terms=best["best"]["terms"],
        metrics=best["best"]["report"],
        loss=best["best"]["loss"],
        stage2_loss=[r["steps"][0]["loss"] for r in runs if r["label"] == best["label"]][0],
    )
    print(f"\nwrote {os.path.join(HERE, args.best_out)}  (genome_hash {h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
