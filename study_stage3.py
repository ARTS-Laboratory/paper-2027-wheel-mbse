"""
=============================================================================
  M8b-i GATE — THE STAGE-3 OPTIMIZER, AND WHETHER ITS PROBLEM HAS A SOLUTION
=============================================================================

    .venv-opt/bin/python study_stage3.py            # the gate; exits nonzero on failure
    .venv-opt/bin/python study_stage3.py --quick    # reduced meshes and step counts

M8a proved the objective and its gradient.  This gates the thing that descends it, and
then asks the question M8a left open with numbers attached.

WHY THIS STUDY EXISTS
---------------------
`PLAN.md` records the finding that shapes the whole milestone: at the shipped genome the
FEA objective is 513.2 units against the GA's 50.4, the wheel misses its deflection
target by -26.3% AND exceeds its stress allowable by 24%, and *"whether M8b has a
feasible problem is now an open question"*.  A Stage-3 run against an infeasible problem
and a Stage-3 run against a buggy optimizer produce the same artifact — a trajectory
that does not reach its target — and neither M7's gradient checks nor M8a's objective
gates can tell them apart, because in both cases the gradient is correct.

So the gates below split into two kinds, and the split is the point:

  * S1-S8 are about the OPTIMIZER.  They are pass/fail, and they are what makes a
    negative feasibility result believable rather than suspicious.
  * S9 is about the PROBLEM.  It is reported and deliberately NOT gated on finding
    feasibility.  Gating on it would make an honest negative result look like a broken
    build, which is the one outcome guaranteed to get the wrong thing fixed.

THE GATES, WRITTEN DOWN BEFORE THE RUN
--------------------------------------
  S1  directional derivative vs an FD ladder of the whole pipeline   1e-4, >= 1 rung
  S2  Adam reduces the loss over a deterministic run                 strictly
  S3  the projection is exact, and does not freeze a pinned gene     exact; no violation
  S4  `stress_scale` used == the PREVIOUS step's measurement         1e-12
  S5  a failed solve is a step reject and the run recovers           recovers, logged
  S6  the pinned flank orientation is the one the meshes are built   exact
  S7  rqmc vs uniform vs iid: step-to-step variance and wall cost    rqmc < iid
  S8  warm start vs cold, seconds per evaluation                     reported
  S9  THE FEASIBILITY VERDICT                                        reported, not gated
  S10 cost per step by tier, and the projected production cost       reported

EVERY FINITE DIFFERENCE IS A LADDER, per the project's rule.  M7 lost days to three
gates written at a single step, all three of which failed at `coarse` by one to eight
parts in 1e5 because of the REFERENCE's truncation error rather than the gradient's.

THE ONE THING S1 HAS TO GET RIGHT THAT IS NOT OBVIOUS
------------------------------------------------------
`stress_scale` is held FIXED across the base point and both legs of every difference.
M8a's gate 7 is the reason: the stress term rescales a p-norm to the true max by a
measured ratio, that rescale is exact only for a constant factor, and re-measuring it
inside each call makes the function being differentiated a different function from the
one being evaluated.  Measured, it put 10% into the assembled gradient while every
individual term still matched its own finite difference to 1e-8.  A ladder run without
pinning `c` would fail here and the failure would look like a broken optimizer.

The FD legs are allowed to leave the unit box by a few parts in a thousand.  That is
deliberate: the objective is a smooth function of the genes everywhere, the box is the
OPTIMIZER's constraint rather than the physics', and projecting the legs would make the
difference a difference of two different functions.
=============================================================================
"""

import argparse
import collections
import json
import os
import time

import jax_config  # noqa: F401  — must precede every other jax import
import numpy as np

import wheel_fea as W
import wheel_fem as fem
import wheel_genome as wg
import wheel_objective as WO
import wheel_stage3 as S3
import wheel_wheel as WW

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = "coarse"

# Written down BEFORE the study was run, per the project's rule.
GATE_DIRECTION_REL = 1e-4       # S1  directional derivative vs its FD plateau
GATE_DIRECTION_RUNGS = 1        # S1  ... on at least one rung of the ladder
GATE_BOX_TOL = 0.0              # S3  the projection is exact, not approximate
GATE_SCALE_REL = 1e-12          # S4  `c` used is the previous step's measurement
GATE_WARM_SAVING = 0.0          # S8  reported, not gated: warm must not be SLOWER

# Reported, not gated.  A design is feasible when it is inside both of these.
FEASIBLE_UTIL = 1.0             # S9  stress utilisation at or under the allowable
FEASIBLE_DEFL_REL = 0.05        # S9  within 5% of the 2.0 mm deflection target

# The weight sets the feasibility probe descends.  Every barrier stays on in all three —
# a "lowest reachable stress" that is reached through a folded, self-intersecting or
# unmeshable design is not a bound on anything.
PROBE_ZERO = {"stress_only": ("deflection", "mass", "phase_ripple"),
              "deflection_only": ("stress", "mass", "phase_ripple"),
              "joint": ()}


def load_genes(path="best_solution.json"):
    with open(os.path.join(HERE, path)) as fh:
        return np.array(list(json.load(fh)["genes"].values()), dtype=float)


def load_elites(path="stage2_elites.json", limit=4):
    p = os.path.join(HERE, path)
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        rec = json.load(fh)
    return [np.array(list(e["genes"].values()), dtype=float)
            for e in rec.get("elites", [])[:limit]]


def _bounds():
    return wg.bounds_arrays(W.GENE_SPACE)


def probe_weights(kind):
    """`DEFAULT_WEIGHTS` with the named terms switched off."""
    w = dict(WO.DEFAULT_WEIGHTS)
    for k in PROBE_ZERO[kind]:
        w[k] = 0.0
    return w


# ---------------------------------------------------------------------------
# S1 — THE DESCENT DIRECTION IS A DESCENT DIRECTION
# ---------------------------------------------------------------------------

def run_direction(genes, cfg=DEFAULT_CONFIG, n_phase=4,
                  steps=(1e-3, 1e-4, 1e-5, 1e-6)):
    """The gradient's own directional derivative against an FD ladder of the pipeline.

    M8a's gate 7 differenced the total along the COORDINATE axes.  This differences it
    along `-g/||g||`, which is the only direction the optimizer ever actually moves in,
    and which mixes all fourteen components — so a per-gene sign error that happens to
    cancel in a coordinate check cannot survive here.  The predicted value is exact and
    needs no reference implementation: `grad . (-g/||g||) = -||g||`.
    """
    low, high, _ = _bounds()
    z0 = wg.normalize(genes, low, high)
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    ori = WW.flank_orientation(genes, WW.get_config(cfg))

    ev = S3.Evaluator(cfg, orientation=ori)
    # Prime, then pin.  The first call measures `c`; every call after it is scored with
    # that same `c`, base point and both legs alike.  See the module docstring.
    ev(z0, low, high, phases=phases)
    c = ev.stress_scale
    val0, g0, brk0 = ev(z0, low, high, phases=phases, refresh_scale=False)

    # Both legs start their secant from the BASE point's indentations, which is
    # `study_objective.run_stress_plateau`'s reasoning applied one level up: a cold solve
    # at a perturbed design can converge along a different Newton path — a different
    # contact active set — and the difference of two such solves carries that as noise,
    # which is exactly what eats a plateau.  One starting guess keeps both legs on one
    # branch, and it is also what makes a ten-evaluation ladder affordable at `coarse`.
    warm = S3.warm_from(brk0)

    gnorm = float(np.linalg.norm(g0))
    d = -np.asarray(g0, dtype=float) / max(gnorm, 1e-300)
    predicted = float(np.dot(g0, d))            # == -||g||, to rounding

    rows = []
    for t in steps:
        vp, _, _ = ev(z0 + t * d, low, high, phases=phases, warm=warm,
                      refresh_scale=False)
        vm, _, _ = ev(z0 - t * d, low, high, phases=phases, warm=warm,
                      refresh_scale=False)
        fd = (vp - vm) / (2.0 * t)
        rel = abs(fd - predicted) / max(abs(predicted), 1e-300)
        rows.append({"t": float(t), "fd": float(fd), "rel": float(rel)})

    rungs = int(sum(r["rel"] < GATE_DIRECTION_REL for r in rows))
    out = {"config": cfg if isinstance(cfg, str) else cfg.name, "n_phase": n_phase,
           "steps": list(steps), "rows": rows, "loss": float(val0),
           "grad_norm": gnorm, "predicted": predicted, "stress_scale": float(c),
           "best_rel": float(min(r["rel"] for r in rows)), "rungs": rungs,
           "terms": {k: v["value"] for k, v in brk0["terms"].items()}}
    out["pass"] = bool(rungs >= GATE_DIRECTION_RUNGS)
    return out


# ---------------------------------------------------------------------------
# S2/S3/S4/S6 — ONE DETERMINISTIC RUN, FOUR THINGS READ OFF IT
# ---------------------------------------------------------------------------

def run_trajectory(genes, cfg=DEFAULT_CONFIG, steps=25, n_phase=4, lr=S3.DEFAULT_LR,
                   scheme="uniform"):
    """A run at fixed phases, and the four structural facts its record has to satisfy.

    Deterministic on purpose: S2 asks whether the loss goes DOWN, and under a stochastic
    stencil a rise is not evidence of anything.  The phase schemes are compared
    separately, in S7, which is where that question belongs.
    """
    low, high, _ = _bounds()
    z0 = wg.normalize(genes, low, high)
    t0 = time.time()
    rec = S3.descend(z0, cfg, steps=steps, lr=lr, n_phase=n_phase, scheme=scheme,
                     verbose=False)
    wall = time.time() - t0
    rows = rec["steps"]
    live = [r for r in rows if not r["abandoned"]]

    # -- S2: it descends.
    l0, lN = rows[0]["loss"], rows[-1]["loss"]
    best = min(r["loss"] for r in rows)
    descent = {"loss_start": float(l0), "loss_end": float(lN), "loss_best": float(best),
               "factor": float(l0 / best) if best > 0 else float("inf"),
               "pass": bool(best < l0)}

    # -- S3: the projection is exact, and it does not freeze a gene whose gradient
    # points back into the box.  Reconstructed offline from (z, grad) — a correct
    # projection is an IMPLICATION, not a hope that some gene happened to move: a gene on
    # a bound must stay there if and only if the unprojected step would leave the box.
    box_viol, freeze_viol, unpinned = [], [], []
    for r in rows:
        z = np.asarray(r["z"], dtype=float)
        if np.any(z < -GATE_BOX_TOL) or np.any(z > 1.0 + GATE_BOX_TOL):
            box_viol.append({"step": r["step"], "min": float(z.min()),
                             "max": float(z.max())})
    # Replays `descend`'s update exactly, abandoned steps included: on an abandonment the
    # moments are NOT advanced (the step was never taken), and replaying it any other way
    # would desynchronise `m`/`v` and manufacture violations that never happened.
    m = np.zeros(wg.N_GENES)
    v = np.zeros(wg.N_GENES)
    for k in range(1, len(rows)):
        prev, cur = rows[k - 1], rows[k]
        g, _ = S3.clip_global_norm(np.asarray(prev["grad"], dtype=float))
        delta, m_try, v_try = S3.adam_update(g, m, v, cur["step"], cur["lr"])
        if cur["abandoned"]:
            continue
        m, v = m_try, v_try
        zp = np.asarray(prev["z"], dtype=float)
        zc = np.asarray(cur["z"], dtype=float)
        unproj = zp - delta * cur["trial_scale"]
        for j in range(wg.N_GENES):
            at_low, at_high = zp[j] <= 0.0, zp[j] >= 1.0
            if not (at_low or at_high):
                continue
            wants_in = (at_low and unproj[j] > 0.0) or (at_high and unproj[j] < 1.0)
            moved = abs(zc[j] - zp[j]) > 1e-15
            if wants_in and not moved:
                freeze_viol.append({"step": cur["step"], "gene": wg.GENE_NAMES[j],
                                    "z": float(zp[j]), "unprojected": float(unproj[j])})
            if wants_in and moved:
                unpinned.append({"step": cur["step"], "gene": wg.GENE_NAMES[j],
                                 "from": float(zp[j]), "to": float(zc[j])})
    projection = {"box_violations": box_viol, "freeze_violations": freeze_viol,
                  "n_unpinned_moves": len(unpinned), "unpinned": unpinned[:12],
                  "pinned_at_start": [n for n, _, _, _ in
                                      wg.bound_saturation(genes, low, high, 0.0)],
                  "pass": bool(not box_viol and not freeze_viol)}

    # -- S4: `c` used at step k is the measurement from step k-1, exactly.
    scale_rows, worst = [], 0.0
    for k in range(1, len(live)):
        used = live[k]["report"].get("stress_scale")
        prev_meas = live[k - 1]["report"].get("stress_scale_measured")
        if used is None or prev_meas is None:
            continue
        rel = abs(used - prev_meas) / max(abs(prev_meas), 1e-300)
        worst = max(worst, rel)
        scale_rows.append({"step": live[k]["step"], "used": used,
                           "previous_measured": prev_meas, "rel": rel})
    scale = {"worst_rel": float(worst), "n_checked": len(scale_rows),
             "rows": scale_rows[:8], "gate": GATE_SCALE_REL,
             "pass": bool(scale_rows and worst <= GATE_SCALE_REL)}

    # -- S6: the pin is honoured.  Not "no flip happened" — a flip is a legitimate event
    # — but "the run scored the topology the pin asked for, for every phase".
    #
    # Checked against what the run RECORDED it pinned to, and against the mesh actually
    # built at the final design.  Not against a mesh at the flipped orientation: at the
    # shipped genome the eta=-1 hub arrival does not exist (the flank never reaches
    # r=12.7 mm), so "build it the other way" is not a question the geometry answers.
    final_genes = np.array(list(rec["final"]["genes"].values()), dtype=float)
    pin = tuple(float(x) for x in
                np.asarray(WW.flank_orientation(genes, WW.get_config(cfg))).ravel())
    recorded = tuple(float(x) for x in rec["settings"]["orientation"])
    pinned_mesh = WW.build_wheel(final_genes, cfg, orientation=pin)
    try:
        free = tuple(float(x) for x in
                     np.asarray(WW.build_wheel(final_genes, cfg).orientation).ravel())
    except Exception as exc:                                  # pragma: no cover
        free = f"unbuildable: {type(exc).__name__}"
    built = tuple(float(x) for x in np.asarray(pinned_mesh.orientation).ravel())
    orientation = {
        "pinned": list(pin), "recorded_by_run": list(recorded),
        "mesh_built_with_pin": list(built),
        "free_choice_at_final_design": free if isinstance(free, str) else list(free),
        "free_choice_still_agrees": bool(free == pin),
        "n_flip_events": sum(1 for e in rec["events"]
                             if e["kind"] == "orientation_flip"),
        "pass": bool(built == pin and recorded == pin)}

    return {"config": cfg if isinstance(cfg, str) else cfg.name, "steps": steps,
            "n_phase": n_phase, "scheme": scheme, "lr": lr, "wall_s": round(wall, 1),
            "descent": descent, "projection": projection, "stress_scale": scale,
            "orientation": orientation,
            "loss_history": [float(r["loss"]) for r in rows],
            "util_history": [float(r["report"].get("stress_utilisation", float("nan")))
                             for r in rows],
            "drop_history": [float(r["report"].get("axle_drop_mean_mm", float("nan")))
                             for r in rows],
            "final": rec["final"], "best": rec["best"], "events": rec["events"],
            "pass": bool(descent["pass"] and projection["pass"] and scale["pass"]
                         and orientation["pass"])}


# ---------------------------------------------------------------------------
# S5 — A FAILED SOLVE IS A STEP REJECT
# ---------------------------------------------------------------------------

class _FaultyEvaluator(S3.Evaluator):
    """Raises `NewtonDivergedError` on the first `n_fail` calls after the start point.

    Fault injection rather than a hunt for a real divergence.  A test that waits for the
    solver to fail on its own is a test that runs when the physics feels like it, and
    the whole point of the reject path is that it is exercised rarely and must work the
    first time it is.
    """

    def __init__(self, *a, n_fail=2, **kw):
        super().__init__(*a, **kw)
        self.n_fail = n_fail
        self.n_raised = 0

    def __call__(self, *a, **kw):
        if self.n_calls > 0 and self.n_raised < self.n_fail:
            self.n_raised += 1
            raise fem.NewtonDivergedError(
                "injected: the tangent is not positive definite (slope = r @ du >= 0)")
        return super().__call__(*a, **kw)


def run_reject(genes, cfg=DEFAULT_CONFIG, n_phase=2, steps=2):
    """Two halves: the run survives rejects, and it gives up correctly when it must."""
    low, high, _ = _bounds()
    z0 = wg.normalize(genes, low, high)
    ori = WW.flank_orientation(genes, WW.get_config(cfg))

    # -- recovers: two injected failures, four trials allowed.
    ev = _FaultyEvaluator(cfg, orientation=ori, n_fail=2)
    rec = S3.descend(z0, cfg, steps=steps, n_phase=n_phase, scheme="uniform",
                     max_rejects=3, evaluator=ev, verbose=False)
    rejects = [e for e in rec["events"] if e["kind"] == "solve_reject"]
    moved = not np.allclose(np.asarray(rec["steps"][-1]["z"]), z0)
    recovered = {"n_injected": ev.n_fail, "n_reject_events": len(rejects),
                 "n_steps_recorded": len(rec["steps"]),
                 "trial_scales": [e["scale"] for e in rejects],
                 "error_names": sorted({e["error"] for e in rejects}),
                 "iterate_moved": bool(moved),
                 "pass": bool(len(rejects) == ev.n_fail and moved
                              and len(rec["steps"]) == steps + 1)}

    # -- gives up: every trial fails, so the iterate must be RESTORED, not corrupted.
    ev2 = _FaultyEvaluator(cfg, orientation=ori, n_fail=10_000)
    rec2 = S3.descend(z0, cfg, steps=1, n_phase=n_phase, scheme="uniform",
                      max_rejects=1, lr=S3.DEFAULT_LR, evaluator=ev2, verbose=False)
    abandoned = [e for e in rec2["events"] if e["kind"] == "step_abandoned"]
    z_end = np.asarray(rec2["steps"][-1]["z"], dtype=float)
    restored = {"n_abandoned": len(abandoned),
                "lr_after": abandoned[0]["lr_after"] if abandoned else None,
                "iterate_unchanged": bool(np.array_equal(z_end, z0)),
                "pass": bool(len(abandoned) == 1 and np.array_equal(z_end, z0)
                             and abandoned[0]["lr_after"] < S3.DEFAULT_LR)}

    return {"recovered": recovered, "restored": restored,
            "pass": bool(recovered["pass"] and restored["pass"])}


# ---------------------------------------------------------------------------
# S7 — THE PHASE SCHEMES, ON RUN BEHAVIOUR RATHER THAN ON BIAS
# ---------------------------------------------------------------------------

def run_phase_schemes(genes, cfg=DEFAULT_CONFIG, steps=8, n_phase=4, n_sub=4, seed=0):
    """rqmc / uniform / iid over one budget, scored on noise AND on wall clock.

    M8a's G9 measured the schemes' BIAS against a 64-point reference.  This measures what
    an optimizer actually feels: how much the loss jitters step to step, and what the
    scheme costs.  The cost half is not a footnote — `coord_fn` keys its jit cache on
    `float(phase)`, so `iid` draws a fresh phase every step and re-traces every step,
    which is the concrete reason `phase_stencil` quantizes the rqmc offset onto a lattice
    instead of shifting it continuously.
    """
    low, high, _ = _bounds()
    z0 = wg.normalize(genes, low, high)
    out = {}
    for scheme in ("uniform", "rqmc", "iid"):
        t0 = time.time()
        rec = S3.descend(z0, cfg, steps=steps, n_phase=n_phase, n_sub=n_sub,
                         scheme=scheme, seed=seed, verbose=False)
        wall = time.time() - t0
        loss = np.array([r["loss"] for r in rec["steps"]], dtype=float)
        walls = np.array([r["wall_s"] for r in rec["steps"]], dtype=float)
        # Step-to-step jitter of the loss, relative — the part of the signal that is the
        # stencil moving rather than the design moving.
        d = np.abs(np.diff(loss)) / np.maximum(np.abs(loss[:-1]), 1e-30)

        # First visit to a stencil vs a repeat, split apart.  `coord_fn` keys its jit
        # cache on `float(phase)`, so the first time a lattice point is used it pays a
        # trace and every use after that does not.  That makes the trace a ONE-OFF cost
        # of at most `n_phase * n_sub` per run, not a per-step cost — and a short gate run
        # is nearly all first visits, so a raw median badly overstates what rqmc costs a
        # 300-step run.  `iid` is the case with no steady state at all: every step draws a
        # stencil it has never seen, so it never stops paying.
        seen, first, repeat = set(), [], []
        for r in rec["steps"]:
            key = tuple(r["phase_deg"])
            (repeat if key in seen else first).append(r["wall_s"])
            seen.add(key)
        out[scheme] = {"loss_end": float(loss[-1]), "loss_best": float(loss.min()),
                       "jitter_mean": float(d.mean()), "jitter_max": float(d.max()),
                       "wall_s": round(wall, 1),
                       "wall_per_step_median": float(np.median(walls)),
                       "wall_per_step_max": float(walls.max()),
                       "wall_first_visit_median":
                           float(np.median(first)) if first else float("nan"),
                       "wall_repeat_median":
                           float(np.median(repeat)) if repeat else float("nan"),
                       "n_first_visits": len(first), "n_repeats": len(repeat),
                       "n_distinct_stencils": len(seen),
                       "loss_history": [float(x) for x in loss]}
    out["steps"] = steps
    out["n_phase"] = n_phase
    out["pass"] = bool(out["rqmc"]["jitter_mean"] <= out["iid"]["jitter_mean"])
    return out


# ---------------------------------------------------------------------------
# S8 — WARM START
# ---------------------------------------------------------------------------

def run_warm(genes, cfg=DEFAULT_CONFIG, n_phase=4, n_rep=3):
    """Seconds per evaluation with and without the previous step's indentations.

    Measured at FIXED phases and after a priming call, so the jit trace is paid once and
    is not attributed to either arm.  Otherwise the trace — which at `smoke` measured
    roughly ten times a step's solve — swamps the quantity being measured.
    """
    low, high, _ = _bounds()
    z0 = wg.normalize(genes, low, high)
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    ori = WW.flank_orientation(genes, WW.get_config(cfg))
    ev = S3.Evaluator(cfg, orientation=ori)

    _, _, brk = ev(z0, low, high, phases=phases)         # prime: trace + measure `c`
    warm = S3.warm_from(brk)

    cold_t, warm_t = [], []
    for k in range(n_rep):
        # A slightly different design each rep, so a cached solve cannot flatter either
        # arm; the same designs are used for both.
        z = np.clip(z0 + 1e-3 * (k + 1), 0.0, 1.0)
        t0 = time.time()
        ev(z, low, high, phases=phases, warm=None, refresh_scale=False)
        cold_t.append(time.time() - t0)
        t0 = time.time()
        ev(z, low, high, phases=phases, warm=warm, refresh_scale=False)
        warm_t.append(time.time() - t0)

    cold_m, warm_m = float(np.median(cold_t)), float(np.median(warm_t))
    saving = (cold_m - warm_m) / cold_m if cold_m > 0 else 0.0
    return {"n_phase": n_phase, "n_rep": n_rep,
            "cold_s_median": cold_m, "warm_s_median": warm_m,
            "cold_s": [float(x) for x in cold_t], "warm_s": [float(x) for x in warm_t],
            "saving_frac": float(saving),
            "pass": bool(saving >= GATE_WARM_SAVING)}


# ---------------------------------------------------------------------------
# S9 — THE FEASIBILITY VERDICT.  REPORTED, NOT GATED.
# ---------------------------------------------------------------------------

def run_feasibility(genes, cfg=DEFAULT_CONFIG, steps=40, n_phase=4, lr=S3.DEFAULT_LR):
    """Three descents that answer `PLAN.md:41` — is there a feasible point at all?

    The shipped design misses deflection by -26.3% and exceeds its allowable by 24%
    simultaneously, so "the objective is large" says nothing about which of the two is
    binding.  Descending each constraint ALONE, with every geometric barrier still on,
    bounds each one separately; the joint run then says where the real weighted objective
    actually lands between them.  Two bounds and a landing point is not a Pareto front,
    but it is enough to distinguish "the weights are wrong" from "the box has no feasible
    point", which is the only distinction M8b-ii needs before it is funded.

    THE ANSWER IS NOT A GATE.  A gate that fails when the wheel turns out to be
    infeasible would be reporting a fact about the design as a fact about the code.
    """
    low, high, _ = _bounds()
    z0 = wg.normalize(genes, low, high)
    runs = {}
    for kind in ("stress_only", "deflection_only", "joint"):
        t0 = time.time()
        rec = S3.descend(z0, cfg, steps=steps, lr=lr, weights=probe_weights(kind),
                         n_phase=n_phase, scheme="uniform", verbose=False)
        rows = rec["steps"]
        util = np.array([r["report"].get("stress_utilisation", np.nan) for r in rows])
        drop = np.array([r["report"].get("axle_drop_mean_mm", np.nan) for r in rows])
        err = (drop - WO.TARGET_DEFLECTION_MM) / WO.TARGET_DEFLECTION_MM
        both = np.where((util <= FEASIBLE_UTIL) & (np.abs(err) <= FEASIBLE_DEFL_REL))[0]
        runs[kind] = {
            "wall_s": round(time.time() - t0, 1),
            "loss_start": float(rows[0]["loss"]), "loss_end": float(rows[-1]["loss"]),
            "util_start": float(util[0]), "util_min": float(np.nanmin(util)),
            "util_end": float(util[-1]),
            "drop_start_mm": float(drop[0]), "drop_end_mm": float(drop[-1]),
            "defl_err_start": float(err[0]), "defl_err_end": float(err[-1]),
            "abs_defl_err_min": float(np.nanmin(np.abs(err))),
            "util_history": [float(x) for x in util],
            "defl_err_history": [float(x) for x in err],
            "n_steps_both_satisfied": int(both.size),
            "genes": rec["best"]["genes"],
            "bound_saturation": rec["final"]["bound_saturation"],
            "n_events": len(rec["events"]),
            # A probe that never accepted a step reported its STARTING POINT as its
            # answer, and a bare event count did not make that visible.  These do.
            "event_kinds": dict(collections.Counter(e["kind"] for e in rec["events"])),
            "n_steps_recorded": len(rows),
            "n_steps_accepted": sum(1 for r in rows if not r["abandoned"]) - 1,
            "stopped_stuck": any(e["kind"] == "run_stopped_stuck" for e in rec["events"]),
            "moved": bool(not np.allclose(rows[-1]["z"], rows[0]["z"]))}

    verdict = {
        "min_reachable_util": runs["stress_only"]["util_min"],
        "min_reachable_abs_defl_err": runs["deflection_only"]["abs_defl_err_min"],
        "stress_reachable": bool(runs["stress_only"]["util_min"] <= FEASIBLE_UTIL),
        "deflection_reachable": bool(
            runs["deflection_only"]["abs_defl_err_min"] <= FEASIBLE_DEFL_REL),
        "simultaneously_reached": bool(
            any(runs[k]["n_steps_both_satisfied"] > 0 for k in runs)),
        "util_when_deflection_met": runs["deflection_only"]["util_end"],
        "defl_err_when_stress_met": runs["stress_only"]["defl_err_end"],
        "joint_util_end": runs["joint"]["util_end"],
        "joint_defl_err_end": runs["joint"]["defl_err_end"],
    }
    return {"config": cfg if isinstance(cfg, str) else cfg.name, "steps": steps,
            "n_phase": n_phase, "feasible_util": FEASIBLE_UTIL,
            "feasible_defl_rel": FEASIBLE_DEFL_REL, "runs": runs, "verdict": verdict,
            # PASS means every probe actually DESCENDED — not that it found a feasible
            # point, and not merely that the function returned.
            #
            # The weaker condition was `loss_end == loss_end`, a NaN check, and it is why
            # a deadlocked stress-only probe passed while reporting its starting point as
            # the lowest reachable utilisation.  A bound obtained by never moving is not a
            # bound on anything, and the gate has to be able to tell the difference
            # between "descended and this is as far as it got" and "never took a step".
            "pass": bool(all(runs[k]["moved"] and runs[k]["n_steps_accepted"] > 0
                             for k in runs))}


# ---------------------------------------------------------------------------
# S10 — WHAT A STEP COSTS
# ---------------------------------------------------------------------------

def run_cost(genes, cfg=DEFAULT_CONFIG, n_phase=8, prod_steps=300, prod_starts=4):
    """Seconds by tier, and what that projects to for the M8b-ii production run."""
    low, high, _ = _bounds()
    z0 = wg.normalize(genes, low, high)
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    ori = WW.flank_orientation(genes, WW.get_config(cfg))
    ev = S3.Evaluator(cfg, orientation=ori)
    ev(z0, low, high, phases=phases)                       # prime the traces

    def timed(fn, n=3):
        ts = []
        for _ in range(n):
            t0 = time.time()
            fn()
            ts.append(time.time() - t0)
        return float(np.median(ts))

    t_t1 = timed(lambda: S3.t1_barrier_sum(z0, cfg))
    t_t12 = timed(lambda: ev(z0, low, high, phases=phases[:1], tiers=("t1", "t2"),
                             refresh_scale=False))
    t_full = timed(lambda: ev(z0, low, high, phases=phases, refresh_scale=False), n=2)

    serial_h = prod_steps * prod_starts * t_full / 3600.0
    return {"config": cfg if isinstance(cfg, str) else cfg.name, "n_phase": n_phase,
            "t1_s": t_t1, "t1_t2_s": t_t12, "full_s": t_full,
            "t1_frac_of_full": float(t_t1 / t_full) if t_full else float("nan"),
            "t1_t2_frac_of_full": float(t_t12 / t_full) if t_full else float("nan"),
            "per_phase_s": float(t_full / n_phase),
            "projected_serial_hours": float(serial_h),
            "projected_steps": prod_steps, "projected_starts": prod_starts,
            "pass": True}


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def _print(rep):
    def head(s):
        print(f"\n{s}\n" + "-" * len(s))

    print("=" * 78)
    print("  M8b-i GATE — THE STAGE-3 OPTIMIZER")
    print("=" * 78)

    d = rep["direction"]
    head(f"S1  descent direction vs an FD ladder of the whole pipeline  "
         f"[< {GATE_DIRECTION_REL:.0e}, >= {GATE_DIRECTION_RUNGS}]")
    print(f"    loss {d['loss']:.4f}   |grad| {d['grad_norm']:.6g}   "
          f"predicted g.d = {d['predicted']:.6g}   c = {d['stress_scale']:.5f}")
    print(f"    {'t':>10s}{'central diff':>18s}{'rel err':>14s}")
    for r in d["rows"]:
        print(f"    {r['t']:10.1e}{r['fd']:18.6g}{r['rel']:14.3e}")
    print(f"    best {d['best_rel']:.3e} on {d['rungs']} rung(s)"
          f"   -> {'PASS' if d['pass'] else 'FAIL'}")

    t = rep["trajectory"]
    head(f"S2/S3/S4/S6  a deterministic {t['steps']}-step run at {t['n_phase']} phases")
    de, pr, sc, orr = t["descent"], t["projection"], t["stress_scale"], t["orientation"]
    print(f"    S2  loss {de['loss_start']:.4f} -> {de['loss_end']:.4f} "
          f"(best {de['loss_best']:.4f}, factor {de['factor']:.2f}x)"
          f"   -> {'PASS' if de['pass'] else 'FAIL'}")
    print(f"    S3  box violations {len(pr['box_violations'])}, freeze violations "
          f"{len(pr['freeze_violations'])}, pinned genes freed {pr['n_unpinned_moves']}"
          f"   -> {'PASS' if pr['pass'] else 'FAIL'}")
    print(f"        pinned at start: {', '.join(pr['pinned_at_start']) or '(none)'}")
    print(f"    S4  worst |c_used - c_prev_measured| / c = {sc['worst_rel']:.3e} "
          f"over {sc['n_checked']} steps   -> {'PASS' if sc['pass'] else 'FAIL'}")
    print(f"    S6  pin {orr['pinned']}, run recorded {orr['recorded_by_run']}, "
          f"mesh built {orr['mesh_built_with_pin']}")
    print(f"        free choice at the final design "
          f"{orr['free_choice_at_final_design']} "
          f"({'agrees' if orr['free_choice_still_agrees'] else 'STALE'}), "
          f"{orr['n_flip_events']} flip event(s)"
          f"   -> {'PASS' if orr['pass'] else 'FAIL'}")
    print(f"    ({t['wall_s']} s, {len(t['events'])} events)")

    r = rep["reject"]
    head("S5  a failed solve is a step reject, not a crash and not a zero")
    rc, rs = r["recovered"], r["restored"]
    print(f"    injected {rc['n_injected']} divergences -> {rc['n_reject_events']} "
          f"reject events {rc['error_names']}, trial scales {rc['trial_scales']}, "
          f"iterate moved {rc['iterate_moved']}"
          f"   -> {'PASS' if rc['pass'] else 'FAIL'}")
    print(f"    all trials failing -> {rs['n_abandoned']} abandonment(s), "
          f"lr {S3.DEFAULT_LR} -> {rs['lr_after']}, iterate unchanged "
          f"{rs['iterate_unchanged']}   -> {'PASS' if rs['pass'] else 'FAIL'}")

    p = rep["schemes"]
    head("S7  phase scheme: what the optimizer feels, and what it costs")
    print(f"    {'scheme':<10s}{'loss end':>12s}{'jitter mean':>13s}"
          f"{'s/step 1st':>12s}{'s/step rpt':>12s}{'1st/rpt':>9s}{'stencils':>10s}")
    for s in ("uniform", "rqmc", "iid"):
        d2 = p[s]
        n = d2["n_first_visits"], d2["n_repeats"]
        print(f"    {s:<10s}{d2['loss_end']:12.4f}{d2['jitter_mean']:13.4f}"
              f"{d2['wall_first_visit_median']:12.2f}"
              f"{d2['wall_repeat_median']:12.2f}"
              f"{n[0]:5d}/{n[1]:<3d}{d2['n_distinct_stencils']:10d}")
    print(f"    rqmc jitter <= iid jitter   -> {'PASS' if p['pass'] else 'FAIL'}")
    print(f"\n    The 1st/rpt split is the point: a first visit to a lattice point pays a")
    print(f"    coord_fn jit trace and a repeat does not, so the trace is a ONE-OFF cost")
    print(f"    of at most n_phase*n_sub per run rather than a per-step cost.  `iid` has")
    print(f"    no steady state — every step draws a stencil it has never seen — which is")
    print(f"    the concrete reason phase_stencil quantizes the rqmc offset onto a")
    print(f"    lattice instead of shifting it continuously.")

    w = rep["warm"]
    head("S8  warm start — the previous step's indentations as the secant's guess")
    print(f"    cold {w['cold_s_median']:.3f} s   warm {w['warm_s_median']:.3f} s   "
          f"saving {100 * w['saving_frac']:.1f}%"
          f"   -> {'PASS' if w['pass'] else 'FAIL'}")

    c = rep["cost"]
    head("S10  what a step costs, by tier")
    print(f"    T1 only        {c['t1_s']:8.4f} s   ({100 * c['t1_frac_of_full']:.2f}% "
          f"of a full evaluation)")
    print(f"    T1+T2, 1 phase {c['t1_t2_s']:8.4f} s   "
          f"({100 * c['t1_t2_frac_of_full']:.2f}%)")
    print(f"    full, {c['n_phase']} phases {c['full_s']:8.4f} s   "
          f"({c['per_phase_s']:.3f} s/phase)")
    print(f"    projected serial cost of {c['projected_steps']} steps x "
          f"{c['projected_starts']} starts: {c['projected_serial_hours']:.2f} h")
    print(f"\n    NOTE, and it cuts both ways.  T1's ABSOLUTE cost is "
          f"{c['t1_s']:.2f} s, not the")
    print(f"    '~ms' wheel_objective's tiering table claims — `t1_vector` carries no")
    print(f"    @jax.jit, so `jacrev` runs it eagerly and pays python dispatch per op.")
    print(f"    But as a FRACTION of a `{c['config']}` evaluation it is "
          f"{100 * c['t1_frac_of_full']:.2f}%, so the")
    print(f"    tiering ECONOMICS hold here and the cheap-refusal argument survives; it")
    print(f"    is at `smoke` that the same 1 s lands against a 7 s solve and becomes")
    print(f"    21%.  The fraction is a statement about the mesh, the second is about")
    print(f"    T1.  Jitting `t1_vector` is the real fix and belongs in")
    print(f"    wheel_objective.py, which M8b-i does not touch.")

    f = rep["feasibility"]
    v = f["verdict"]
    head("S9  THE FEASIBILITY VERDICT — reported, NOT gated")
    print(f"    {'probe':<18s}{'loss':>22s}{'utilisation':>22s}"
          f"{'deflection error':>22s}")
    for k in ("stress_only", "deflection_only", "joint"):
        d3 = f["runs"][k]
        print(f"    {k:<18s}{d3['loss_start']:10.2f} ->{d3['loss_end']:10.2f}"
              f"{d3['util_start']:10.3f} ->{d3['util_end']:10.3f}"
              f"{100 * d3['defl_err_start']:9.1f}% ->{100 * d3['defl_err_end']:9.1f}%")
        flag = "" if d3["n_steps_accepted"] > 0 else "   <- NEVER MOVED, bounds nothing"
        print(f"      {d3['n_steps_accepted']:d}/{d3['n_steps_recorded'] - 1} steps "
              f"accepted, events {d3['event_kinds'] or '{}'}"
              f"{', STOPPED STUCK' if d3['stopped_stuck'] else ''}{flag}")
    print(f"\n    lowest reachable stress utilisation   {v['min_reachable_util']:.4f}"
          f"   (feasible at <= {f['feasible_util']:.2f}: "
          f"{'YES' if v['stress_reachable'] else 'NO'})")
    print(f"    lowest reachable |deflection error|   "
          f"{100 * v['min_reachable_abs_defl_err']:.2f}%"
          f"   (feasible at <= {100 * f['feasible_defl_rel']:.0f}%: "
          f"{'YES' if v['deflection_reachable'] else 'NO'})")
    print(f"    stress met at deflection error        "
          f"{100 * v['defl_err_when_stress_met']:.1f}%")
    print(f"    deflection met at utilisation         "
          f"{v['util_when_deflection_met']:.3f}")
    print(f"    BOTH satisfied at any visited design  "
          f"{'YES' if v['simultaneously_reached'] else 'NO'}")
    print(f"    -> {'PASS (probe ran)' if f['pass'] else 'FAIL (probe did not run)'}")

    print(f"\n{'=' * 78}")
    print(f"  OVERALL: {'PASS' if rep['pass'] else 'FAIL'}")
    print("=" * 78)
    print("\n  NOT DONE: the process-parallel phase batch, the multi-fidelity")
    print("  checkpoints and the 300-step multi-start production run. Those are M8b-ii,")
    print("  and S9 above is the measurement that says whether funding them is sensible.")
    print("  `lambda_min(K_t)` remains M9; `buckling` is still the zero-gradient Euler")
    print("  proxy, and a diverged tangent is the only buckling signal this run has.")


def _plot(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

    d = rep["direction"]
    ts = [r["t"] for r in d["rows"]]
    rels = [r["rel"] for r in d["rows"]]
    ax[0].loglog(ts, rels, "o-", lw=1, ms=4, label="central diff vs $-\\|g\\|$")
    ax[0].axhline(GATE_DIRECTION_REL, color="k", ls=":", lw=0.9,
                  label=f"gate {GATE_DIRECTION_REL:.0e}")
    ax[0].set_xlabel("FD step along $-\\hat{g}$  [normalised]")
    ax[0].set_ylabel("relative error")
    ax[0].set_title("the descent direction has a plateau")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7)

    t = rep["trajectory"]
    ax[1].semilogy(t["loss_history"], "o-", lw=1, ms=3, label="deterministic run")
    for s, col in (("rqmc", "C1"), ("iid", "C3")):
        ax[1].semilogy(rep["schemes"][s]["loss_history"], "-", lw=1, color=col,
                       alpha=0.8, label=s)
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("objective")
    ax[1].set_title("Adam descends; iid jitters")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=7)

    f = rep["feasibility"]
    e = 100 * f["feasible_defl_rel"]
    # The feasible set itself, drawn rather than described: util <= 1 AND |err| <= 5%.
    # Whether any trajectory enters it is the whole question S9 exists to answer.
    ax[2].add_patch(Rectangle((-e, 0.0), 2 * e, f["feasible_util"],
                              facecolor="C2", alpha=0.20, edgecolor="C2",
                              lw=1.0, ls="--", zorder=0, label="feasible"))
    lo, hi = np.inf, -np.inf
    for k, col in (("stress_only", "C0"), ("deflection_only", "C1"), ("joint", "C3")):
        r = f["runs"][k]
        u = np.asarray(r["util_history"], dtype=float)
        lo, hi = min(lo, np.nanmin(u)), max(hi, np.nanmax(u))
        ax[2].plot(100 * np.array(r["defl_err_history"]), u, "o-",
                   lw=1, ms=3, color=col, alpha=0.85, label=k)
    # Keep the axis on the trajectories; the feasible box is clipped from below on
    # purpose, since nothing below the reachable utilisation is informative.
    ax[2].set_ylim(min(lo, f["feasible_util"]) - 0.04, hi + 0.04)
    ax[2].plot(100 * f["runs"]["joint"]["defl_err_history"][0],
               f["runs"]["joint"]["util_history"][0], "k*", ms=11,
               label="shipped genome", zorder=5)
    ax[2].axhline(f["feasible_util"], color="k", ls=":", lw=0.9)
    ax[2].set_xlabel("deflection error  [%]")
    ax[2].set_ylabel("stress utilisation")
    # The title states what was measured, so it cannot outlive the measurement.
    v = f["verdict"]
    if v["simultaneously_reached"]:
        title = "the feasible corner is reached"
    elif v["stress_reachable"] and v["deflection_reachable"]:
        title = "each constraint is reachable; the corner is not"
    else:
        title = "a constraint is out of reach on its own"
    ax[2].set_title(title)
    ax[2].grid(alpha=0.3)
    ax[2].legend(fontsize=7, loc="best")

    fig.tight_layout()
    out = os.path.join(HERE, path)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="M8b-i Stage-3 optimizer gate")
    ap.add_argument("--genome", default="best_solution.json")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default="study_stage3.json")
    ap.add_argument("--elites", default="stage2_elites.json")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="reduced meshes and step counts; for the test suite")
    args = ap.parse_args()

    genes = load_genes(args.genome)
    cfg = "smoke" if args.quick else args.config
    # --quick shrinks step counts and the mesh, never a TOLERANCE.  What it genuinely
    # weakens is S9: a 6-step probe bounds nothing, so the verdict it prints is a wiring
    # check rather than a measurement.
    n_phase = 2 if args.quick else 4
    traj_steps = 4 if args.quick else 25
    scheme_steps = 3 if args.quick else 8
    # 20 rather than 40: measured at `coarse`, every probe plateaus well inside it —
    # deflection-only reaches 0.0% error by step 4, joint by step 5, stress-only is at
    # utilisation 1.16 and flattening by step 5.  The cosine schedule anneals over
    # `steps`, so a shorter budget is a faster anneal to the same place, not a truncation.
    feas_steps = 4 if args.quick else 20
    cost_phase = 2 if args.quick else 8
    ladder = (1e-3, 1e-4, 1e-5) if args.quick else (1e-3, 1e-4, 1e-5, 1e-6)

    t0 = time.time()
    rep = {}

    def section(name, fn):
        """Run one gate, announcing it before and after.

        At `coarse` this study is hours and every `run_*` is silent, so without this the
        only observable states are "running" and "done" — which is how a deadlocked probe
        burned thirty-five minutes standing still without anyone being able to see it.
        Flushed, because stdout is a pipe under `make studies` and would otherwise buffer
        the whole run into one block at the end.
        """
        print(f"[{time.time() - t0:7.1f} s] {name} ...", flush=True)
        s = time.time()
        rep[name] = fn()
        print(f"[{time.time() - t0:7.1f} s] {name} done in {time.time() - s:.1f} s"
              f"  -> {'PASS' if rep[name].get('pass') else 'FAIL'}", flush=True)

    section("direction", lambda: run_direction(genes, cfg, n_phase=n_phase, steps=ladder))
    section("trajectory",
            lambda: run_trajectory(genes, cfg, steps=traj_steps, n_phase=n_phase))
    section("reject", lambda: run_reject(genes, cfg, n_phase=n_phase))
    section("schemes",
            lambda: run_phase_schemes(genes, cfg, steps=scheme_steps, n_phase=n_phase))
    section("warm", lambda: run_warm(genes, cfg, n_phase=n_phase))
    section("cost", lambda: run_cost(genes, cfg, n_phase=cost_phase))
    section("feasibility",
            lambda: run_feasibility(genes, cfg, steps=feas_steps, n_phase=n_phase))
    rep["pass"] = bool(rep["direction"]["pass"] and rep["trajectory"]["pass"]
                       and rep["reject"]["pass"] and rep["schemes"]["pass"]
                       and rep["warm"]["pass"] and rep["cost"]["pass"]
                       and rep["feasibility"]["pass"])
    rep["settings"] = {"config": cfg, "genome": args.genome, "quick": args.quick,
                       "service_force_n": W.TOTAL_FORCE_NEWTONS,
                       "target_deflection_mm": WO.TARGET_DEFLECTION_MM,
                       "allowable_stress_mpa": WO.ALLOWABLE_STRESS_MPA,
                       "lr": S3.DEFAULT_LR, "grad_clip": S3.GRAD_CLIP,
                       "elapsed_s": round(time.time() - t0, 1)}

    # Written BEFORE the report is formatted.  At `coarse` this study is an hour of
    # solving and `_print` is string formatting; losing the former to a bug in the latter
    # is a trade nobody would make on purpose.
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(rep, fh, indent=1)
    _print(rep)
    print(f"\nwrote {os.path.join(HERE, args.out)}  "
          f"({rep['settings']['elapsed_s']} s)")
    if not args.no_plot:
        try:
            print(f"wrote {_plot(rep, os.path.splitext(args.out)[0] + '.jpg')}")
        except Exception as exc:                              # pragma: no cover
            print(f"(plot skipped: {exc})")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
