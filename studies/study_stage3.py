"""
=============================================================================
  M8b-i GATE — THE STAGE-3 OPTIMIZER, AND WHETHER ITS PROBLEM HAS A SOLUTION
=============================================================================

    .venv-opt/bin/python studies/study_stage3.py            # the gate; exits nonzero on failure
    .venv-opt/bin/python studies/study_stage3.py --quick    # reduced meshes and step counts

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
  S5  a failed solve is a step reject and the run recovers           recovers, logged
  S6  the pinned flank orientation is the one the meshes are built   exact
  S7  rqmc vs uniform vs iid: step-to-step variance and wall cost    rqmc < iid
  S8  warm start vs cold, seconds per evaluation                     reported
  S9  THE FEASIBILITY VERDICT                                        reported, not gated
  S10 cost per step by tier, and the projected production cost       reported

M8b-i.5 ADDS TWO SECTIONS, AND THEY ARE OPT-IN
----------------------------------------------
    .venv-opt/bin/python studies/study_stage3.py --sections mesh_convergence,multistart \
        --out study_stage3_m8bi5.json                   # or just: make m8bi5

S9's verdict came back "each constraint is reachable alone, neither with the other", and
`PLAN.md` records that it is weaker than it looks in two specific ways.  Both are cheap to
close and either can change the answer, so they are closed before anyone acts on it:

  S11 the stress QoI up the mesh ladder                              reported, not gated
  S12 the same feasibility question from the Stage-2 elites          reported, not gated

S11 exists because utilisation measured 1.2406 at `smoke` and 1.7128 at `coarse` — the
verdict's own input moved 38% for a change of mesh alone.  S12 exists because all three of
S9's descents started at `best_solution.json`, a GA optimum for the BEAM surrogate that
M8a showed is a bad guide to the FEA, and `stage2_elites.json` has held fifteen other
converged genomes that nothing has ever scored.

They are NOT in the default `--sections` and `make studies` does not run them.  At
`coarse` they are ~2 h against the gate's ~2 h 45 m, they answer a question about the
wheel rather than about the code, and a gate nobody can afford to run is not a gate.

EVERY FINITE DIFFERENCE IS A LADDER, per the project's rule.  M7 lost days to three
gates written at a single step, all three of which failed at `coarse` by one to eight
parts in 1e5 because of the REFERENCE's truncation error rather than the gradient's.

S4 WAS HERE, AND WHY IT IS NOT ANY MORE
----------------------------------------
S4 asserted that the `stress_scale` an evaluation used was the PREVIOUS step's
measurement, to 1e-12.  That contract existed because the stress term rescaled a p-norm to
the true max by a measured ratio, the rescale is exact only for a CONSTANT factor, and
re-measuring inside each call made the function being differentiated a different function
from the one being evaluated — measured, 10% into the assembled gradient while every
individual term still matched its own finite difference to 1e-8.  S1 therefore had to pin
`c` across the base point and both legs, or it would have failed and the failure would have
looked like a broken optimizer.

M8b-i.6 deleted the frozen factor rather than the freeze: `c` is anchored to a mesh-divergent
true max, so it was a converged answer to nothing, and the constraint is now
`Kt(R, t) * sigma_nominal(p=4)` with `Kt` differentiated.  Nothing is held across a
difference any more, and S1 is stronger for it — both legs re-evaluate freely, so it now
tests the product rule rather than the pinning.

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

import project_paths as PP
import time

import jax_config  # noqa: F401  — must precede every other jax import
import numpy as np

import wheel_adjoint as WA
import wheel_fea as W
import wheel_fem as fem
import wheel_genome as wg
import wheel_objective as WO
import wheel_pool as WP
import wheel_stage3 as S3
import wheel_wheel as WW

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = "coarse"

# Written down BEFORE the study was run, per the project's rule.
GATE_DIRECTION_REL = 1e-4       # S1  directional derivative vs its FD plateau
GATE_DIRECTION_RUNGS = 1        # S1  ... on at least one rung of the ladder
GATE_BOX_TOL = 0.0              # S3  the projection is exact, not approximate
GATE_WARM_SAVING = 0.0          # S8  reported, not gated: warm must not be SLOWER

# Reported, not gated.  A design is feasible when it is inside both of these.
FEASIBLE_UTIL = 1.0             # S9  stress utilisation at or under the allowable
FEASIBLE_DEFL_REL = 0.05        # S9  within 5% of the 2.0 mm deflection target

# S11.  When does a ladder count as CONVERGED?  Reported, not gated — but the report's
# words and its plot title are driven off this, so it has to be a real criterion.
#
# `study_wheel_fea.run_refinement` uses `finest_error_vs_richardson < 0.005` and calls it
# `criterion_met`.  This is the same standard in GCI form, which is the safety-factored
# version of the same quantity and the more honest one to quote when the observed order is
# far from the formal one.  10x looser than M4's, deliberately: at 5% a quantity is usable
# as a CONSTRAINT even if it is not publication-converged, and the question S11 asks is
# whether the constraint has a value, not whether the value is precise.
#
# NOT `ratio > 1`.  Successive differences shrinking is necessary and nowhere near
# sufficient — a sequence with ratio 1.33 has an observed order of 0.41 and an
# extrapolation 50% beyond its finest rung, which is what "no value" looks like.
GATE_LADDER_GCI = 0.05

# S13.  How much of the ideal speedup must the phase pool actually deliver?
#
# A FRACTION, NOT A MULTIPLE, and that is the device-agnostic half of this milestone.
# "at least 2.5x" passes without effort on a 16-core box and fails on a 4-core laptop that
# is behaving perfectly; `speedup / n_workers` means the same thing on both.  0.35 is set
# low on purpose — it is not a performance target, it is the tripwire for the pool costing
# more than it buys.  The phases are unequal (contact iterations differ by design), the
# parent still runs T1 and T2 serially, and Amdahl takes the rest; anything at or above
# this is the pool working.
GATE_POOL_EFFICIENCY = 0.35

# S13.  How close must a pooled GRADIENT be to the serial one?  Values are gated at exact
# equality and need no tolerance; this is for the gradient alone, and the split is a
# measurement rather than a preference.
#
# WHAT WAS MEASURED, AND WHY EXACT IS NOT AVAILABLE HERE.  Every forward value, every
# report leaf and the stress and phase_ripple gradients come back from the pool bit-for-bit
# identical — 0.0, not "close".  `grads.deflection` does not: it differs by 5.2e-18
# RELATIVE, which is below one ulp of double precision (eps = 2.2e-16).
#
# The cause is not the pool.  Two PLAIN SERIAL runs of one `coarse` adjoint, in two
# separate interpreters with no pool anywhere, already disagreed by 3.33e-16 before any of
# this existed.  Pinning `XLA_FLAGS` (see `wheel_pool.PINNED_ENV`) removed the largest
# source and made a single-worker pool exactly equal to serial; what remains is
# process-history dependent — a process that has already run other phases answers the next
# one differently in its last bit — and lives below the thread pool, in XLA's own codegen.
# It is not reachable from this repo, and it is not this milestone's to fix.
#
# 1e-14 is ~2000x looser than what is observed and still some twelve orders tighter than
# any physical tolerance in this study.  It is a tripwire for a real numerical regression,
# not a fudge factor: `identical_values` is what carries the claim.
GATE_POOL_GRAD_REL = 1e-14

# Leaves whose value is computed FROM a gradient, and which therefore inherit its last-bit
# noise.  `grad_norm` and `grad_share` are per-term reductions of it; `coupling_frac` is a
# ratio of two gradient norms.  Classified by path rather than listed per-term so a new
# gradient-derived report key is covered the day it is added.
_GRAD_DERIVED = ("grad", "coupling_frac")

# The weight sets the feasibility probe descends.  Every barrier stays on in all three —
# a "lowest reachable stress" that is reached through a folded, self-intersecting or
# unmeshable design is not a bound on anything.
PROBE_ZERO = {"stress_only": ("deflection", "mass", "phase_ripple"),
              "deflection_only": ("stress", "mass", "phase_ripple"),
              "joint": ()}

# The sections, and which milestone each answers to.  M8b-i's seven are the default and
# are what `make studies` runs; M8b-i.5's two are opt-in — see `--sections` and `make
# m8bi5`.  Order matters: a default invocation writes the report keys in this order.
SECTION_HELP = {
    "direction":        "S1  the descent direction vs an FD ladder of the pipeline",
    "trajectory":       "S2/S3/S4/S6  one deterministic run, four structural facts",
    "reject":           "S5  a failed solve is a step reject",
    "schemes":          "S7  rqmc vs uniform vs iid",
    "warm":             "S8  warm start vs cold",
    "cost":             "S10 seconds per step by tier",
    "feasibility":      "S9  THE FEASIBILITY VERDICT, from the shipped genome",
    "mesh_convergence": "S11 is the stress QoI converged?  [M8b-i.5]",
    "multistart":       "S12 the same question from the Stage-2 elites  [M8b-i.5]",
    "phase_pool":       "S13 pooled == serial, and what it buys  [M8b-ii]",
}
DEFAULT_SECTIONS = ("direction", "trajectory", "reject", "schemes", "warm", "cost",
                    "feasibility")


def parse_sections(spec):
    """`--sections` -> the list to run, or a `ValueError` naming what was not understood.

    A pure function so it can be tested without paying a jax import, and so a typo costs
    a startup rather than three hours of solving followed by a `KeyError`.
    """
    names = [s.strip() for s in str(spec).split(",") if s.strip()]
    if not names:
        raise ValueError("--sections is empty; expected at least one of "
                         f"{sorted(SECTION_HELP)}")
    unknown = [s for s in names if s not in SECTION_HELP]
    if unknown:
        raise ValueError(f"unknown section(s) {unknown}; expected any of "
                         f"{sorted(SECTION_HELP)}")
    return names


def parse_ladder_p(spec):
    """`--ladder-p` -> the exponents to probe, or a `ValueError` naming the bad one.

    Pure and validated at startup for `parse_sections`' reason, which bites harder here:
    the sweep's whole claim is that it costs no extra solve, so a bad exponent that only
    surfaces at the `_qoi_pnorm_stress` call would be discovered after the ladder had
    already paid for its meshes.  Empty means no sweep, which is the default and leaves
    the report exactly as M8b-i.5 wrote it.

    `p > 0` because the p-norm's `x^(1/p)` is not a norm otherwise, and `p` is a float
    rather than an int because nothing downstream rounds it and the interesting region
    between the master plan's 10 and M4's percentile may not be an integer.

    DUPLICATES ARE DROPPED HERE, not tolerated downstream.  `t3_terms` keys its
    accumulator by exponent (`probe_pn = {v: [] for v in probe_p}`) but appends inside a
    loop over the raw list, so a repeated `p` collects two samples per phase against one
    true max and `_stress_aggregate` dies on the broadcast — after the first solve has
    been paid for, which is precisely the class of failure this function exists to move to
    startup.  First occurrence wins, so the reported order still follows the command line.
    """
    out = []
    for tok in [s.strip() for s in str(spec).split(",") if s.strip()]:
        try:
            v = float(tok)
        except ValueError:
            raise ValueError(f"--ladder-p got {tok!r}, which is not a number; expected "
                             "a comma-separated list of exponents, e.g. 2,4,8,16,30")
        if not (np.isfinite(v) and v > 0.0):
            raise ValueError(f"--ladder-p got {tok!r}; every exponent must be finite and "
                             "positive — a p-norm with p <= 0 is not a norm")
        if v not in out:
            out.append(v)
    return out


def load_genes(path="best_solution.json"):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return np.array(list(json.load(fh)["genes"].values()), dtype=float)


def load_elites(path="stage2_elites.json", limit=4):
    p = os.path.join(PP.ROOT, path)
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
    # No priming call and nothing pinned: the objective is now a pure function of the
    # genes, so the base point and both legs are evaluations of one function.  See the
    # module docstring on what S4 used to guard here.
    val0, g0, brk0 = ev(z0, low, high, phases=phases)

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
        vp, _, _ = ev(z0 + t * d, low, high, phases=phases, warm=warm)
        vm, _, _ = ev(z0 - t * d, low, high, phases=phases, warm=warm)
        fd = (vp - vm) / (2.0 * t)
        rel = abs(fd - predicted) / max(abs(predicted), 1e-300)
        rows.append({"t": float(t), "fd": float(fd), "rel": float(rel)})

    rungs = int(sum(r["rel"] < GATE_DIRECTION_REL for r in rows))
    out = {"config": cfg if isinstance(cfg, str) else cfg.name, "n_phase": n_phase,
           "steps": list(steps), "rows": rows, "loss": float(val0),
           "grad_norm": gnorm, "predicted": predicted,
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

    # S4 was here — see the module docstring.  Nothing is frozen across a step any more,
    # so there is no freeze contract left to assert.

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
            "descent": descent, "projection": projection,
            "orientation": orientation,
            "loss_history": [float(r["loss"]) for r in rows],
            "util_history": [float(r["report"].get("stress_utilisation", float("nan")))
                             for r in rows],
            "drop_history": [float(r["report"].get("axle_drop_mean_mm", float("nan")))
                             for r in rows],
            "final": rec["final"], "best": rec["best"], "events": rec["events"],
            "pass": bool(descent["pass"] and projection["pass"]
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
        ev(z, low, high, phases=phases, warm=None)
        cold_t.append(time.time() - t0)
        t0 = time.time()
        ev(z, low, high, phases=phases, warm=warm)
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

def run_feasibility(genes, cfg=DEFAULT_CONFIG, steps=40, n_phase=4, lr=S3.DEFAULT_LR,
                    kinds=("stress_only", "deflection_only", "joint")):
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

    `kinds` selects which of the three to run, defaulting to all of them so S9 is exactly
    what it always was.  M8b-i.5's multi-start re-probes pass the two BOUNDS alone and
    drop `joint`, which by construction lands between them and adds nothing to a verdict
    about reachability.  The verdict dict then carries only the keys its `kinds` support
    rather than filling the rest with NaN — an absent key is a question that was not
    asked, and `nan` reads as a question that was asked and came back undefined.
    """
    low, high, _ = _bounds()
    z0 = wg.normalize(genes, low, high)
    unknown = [k for k in kinds if k not in PROBE_ZERO]
    if unknown:
        raise ValueError(f"unknown probe kind(s) {unknown}; "
                         f"expected any of {sorted(PROBE_ZERO)}")
    runs = {}
    for kind in kinds:
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

    # Whatever ran, this holds: the lowest utilisation and the smallest deflection error
    # SEEN ANYWHERE, which is the only pair of numbers that bounds anything when the
    # caller has chosen a subset of the probes.  It generalises the old per-probe
    # definition without moving it — checked against the recorded `coarse` run, where the
    # minimum over all three probes IS stress_only's util_min and deflection_only's
    # abs_defl_err_min, to every bit (0.9320225375154423 and 0.0021959376342238).
    verdict = {
        "min_reachable_util": float(np.nanmin(
            [np.nanmin(runs[k]["util_history"]) for k in runs])),
        "min_reachable_abs_defl_err": float(np.nanmin(
            [np.nanmin(np.abs(runs[k]["defl_err_history"])) for k in runs])),
        "simultaneously_reached": bool(
            any(runs[k]["n_steps_both_satisfied"] > 0 for k in runs)),
    }
    verdict["stress_reachable"] = bool(verdict["min_reachable_util"] <= FEASIBLE_UTIL)
    verdict["deflection_reachable"] = bool(
        verdict["min_reachable_abs_defl_err"] <= FEASIBLE_DEFL_REL)
    if "stress_only" in runs:
        verdict["defl_err_when_stress_met"] = runs["stress_only"]["defl_err_end"]
    if "deflection_only" in runs:
        verdict["util_when_deflection_met"] = runs["deflection_only"]["util_end"]
    if "joint" in runs:
        verdict["joint_util_end"] = runs["joint"]["util_end"]
        verdict["joint_defl_err_end"] = runs["joint"]["defl_err_end"]
    return {"config": cfg if isinstance(cfg, str) else cfg.name, "steps": steps,
            "n_phase": n_phase, "kinds": list(kinds), "feasible_util": FEASIBLE_UTIL,
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
    t_t12 = timed(lambda: ev(z0, low, high, phases=phases[:1], tiers=("t1", "t2")))
    t_full = timed(lambda: ev(z0, low, high, phases=phases), n=2)

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
# M8b-i.5 — SCORING ONE DESIGN, WITHOUT DESCENDING FROM IT
# ---------------------------------------------------------------------------

def score(genes, cfg=DEFAULT_CONFIG, phases=None, n_phase=4, tiers=("t3",), probe_p=()):
    """`(loss, report, seconds)` at one design.

    `tiers=("t3",)` on purpose.  Both M8b-i.5 sections want the deflection and the stress,
    and neither wants the geometric barriers — which cost `t1_vector`'s 1.06 s of eager
    `jacrev` per call (S10) to produce numbers this study never reads.

    `c = mean_phase(max/pnorm)` is reported as `stress_scale_measured`, measured at this
    design and this mesh.  It no longer enters the constraint — its drift with the mesh is
    what the ladder is asking about, and the answer (it diverges, because the true max is a
    singularity) is why the constraint stopped using it.

    A fresh `Evaluator` per call, and the orientation re-derived per design: the flank pin
    is a function of the genes, so carrying one elite's pin onto another's mesh would build
    a wheel nobody asked for.

    `probe_p` COSTS NO SOLVE, and that is the whole design of `--ladder-p`.  It rides
    `Evaluator`'s `problem_kw` into `wheel_objective.t3_terms`, which reads each exponent
    off the displacement field the adjoint already converged — the same channel `warm`
    uses.  So one rung stays one evaluation however long the list is — which is what
    `test_the_ladder_costs_one_evaluation_per_rung_and_repins_per_design` asserts, and
    what turns a five-exponent sweep from an hour back into fourteen minutes.
    """
    low, high, _ = _bounds()
    if phases is None:
        phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    ev = S3.Evaluator(cfg, orientation=WW.flank_orientation(genes, WW.get_config(cfg)),
                      stress_p_probe=tuple(probe_p))
    t0 = time.time()
    val, _, brk = ev(wg.normalize(np.asarray(genes, dtype=float), low, high),
                     low, high, phases=phases, tiers=tiers)
    return float(val), brk["report"], time.time() - t0


def corner_distance(util, defl_err):
    """How far outside the feasible box, in multiples of each constraint's OWN limit.

    A ranking heuristic and not a metric — it exists to pick which elites are worth an
    hour of descent, nothing more.  Both terms are dimensionless excesses over their own
    allowance, which is what makes them addable: `util` is already normalised by
    `ALLOWABLE_STRESS_MPA` and `defl_err` by `TARGET_DEFLECTION_MM`, so a raw sum of
    "MPa over" and "mm short" is the mistake this avoids.  Zero anywhere inside the box.
    """
    return (max(0.0, abs(float(util)) / FEASIBLE_UTIL - 1.0)
            + max(0.0, abs(float(defl_err)) / FEASIBLE_DEFL_REL - 1.0))


# ---------------------------------------------------------------------------
# S11 — IS THE STRESS QoI CONVERGED?
# ---------------------------------------------------------------------------

LADDER_CONFIGS = ("smoke", "coarse", "medium")


def _series(rows, key):
    """Richardson extrapolation and a GCI on one column of the ladder.

    Same convention as `study_wheel_fea.run_refinement`, whose `_observed_ratio` is
    imported rather than re-derived — study-to-study, the way `study_gnl` already borrows
    `latin_hypercube` from `study_mesh_quality`.
    """
    from study_wheel_fea import _observed_ratio

    v = np.array([r[key] for r in rows], dtype=float)
    out = {"values": [float(x) for x in v],
           "configs": [r["config"] for r in rows],
           "rel_change_last_pair": float(abs(v[-1] / v[-2] - 1.0)) if len(v) >= 2
           else float("nan"),
           "total_rel_change": float(abs(v[-1] / v[0] - 1.0)) if len(v) >= 2
           else float("nan"),
           "direction": ("rising" if len(v) >= 2 and v[-1] > v[-2]
                         else "falling" if len(v) >= 2 and v[-1] < v[-2] else "flat")}
    if len(v) < 3:
        out.update({"ratio": float("nan"), "richardson": float("nan"),
                    "finest_error_vs_richardson": float("nan"), "gci": float("nan"),
                    "settling": None, "converged": None})
        return out
    ratio = float(_observed_ratio(v))
    r32 = v[-1] - v[-2]
    rich = float(v[-1] + r32 / (ratio - 1.0)) if np.isfinite(ratio) and ratio > 1 \
        else float(v[-1])
    out.update({
        "ratio": ratio,
        "richardson": rich,
        "finest_error_vs_richardson": float(abs(v[-1] / rich - 1.0)) if rich else
        float("nan"),
        "gci": float(1.25 * abs(r32 / v[-1]) / (abs(ratio) - 1.0))
        if np.isfinite(ratio) and abs(ratio) > 1.0 and v[-1] else float("nan"),
        # Two different claims, and conflating them is how a plot title comes to say the
        # opposite of its own data.  `settling` is the weak one: each refinement moved the
        # number less than the one before.  `converged` is the one worth quoting: the
        # remaining discretisation error is inside `GATE_LADDER_GCI`.  Measured here, every
        # stress series is `settling` and none is `converged`, while the axle drop on the
        # very same meshes is both.
        "settling": bool(np.isfinite(ratio) and ratio > 1.0)})
    out["converged"] = bool(np.isfinite(out["gci"]) and out["gci"] < GATE_LADDER_GCI)
    return out


def _series_by_p(ok, probe_p):
    """Per-exponent convergence, off the rows the ladder already measured.

    `_series` is reused verbatim and deliberately: it is a pure function of a list of
    dicts carrying `config` plus one named column, so a synthetic column is all a per-`p`
    series needs, and the GCI the sweep quotes is then the same GCI the p=30 verdict was
    quoted from.  Rewriting the extrapolation for the sweep would let the two disagree,
    which is exactly the failure `_stress_aggregate` exists to prevent one level down.

    A LEAF ABSENT FROM THE ROWS IS SKIPPED, NOT A KeyError.  `util_kt` arrived with
    M8b-i.6 step 2 and the sweeps recorded before it do not carry it; a re-analysis of an
    older `study_stage3_pnorm.json` must still produce every series it does have, because
    the whole value of that file is that its `c` columns are the evidence for the change.
    """
    out = {}
    for v in probe_p:
        k = repr(float(v))
        have = [r for r in ok if k in r.get("pnorm_by_p", {})]
        if len(have) < 2:
            continue
        out[k] = {"p": float(v)}
        for name, field in (("pnorm", "pnorm_agg_mpa"), ("c", "stress_scale_measured"),
                            ("util", "stress_utilisation"),
                            ("util_kt", "stress_utilisation_kt")):
            cols = [{"config": r["config"], "x": r["pnorm_by_p"][k][field]}
                    for r in have if field in r["pnorm_by_p"][k]]
            if len(cols) == len(have):
                out[k][name] = _series(cols, "x")
    return out


def run_mesh_convergence(designs, configs=LADDER_CONFIGS, n_phase=4, probe_p=()):
    """*** THE FIRST HALF OF M8b-i.5 ***  Is the number the verdict rests on converged?

    S9 reported the shipped genome at utilisation 1.7128 and called the problem infeasible.
    The same genome measures 1.2406 at `smoke` — a 38% rise for a change of mesh alone —
    so the verdict's own input is moving, and which way it is still moving decides whether
    the infeasibility is a lower bound or an artifact.

    THREE SERIES, NOT ONE, AND THAT IS THE POINT.  When this was written the constraint was
    `c * pnorm / allowable` with `c = max/pnorm` measured on the mesh, so it inherited
    whatever the max did.  `_qoi_pnorm_stress`'s docstring argues the volume-weighted
    p-norm is a quadrature of an integral and therefore mesh-convergent, while the true max
    is a pointwise peak at an unfilleted junction corner — the same corner `study_wheel_fea`
    blames for its sub-second-order axle-drop rate — and a peak at a stress concentration
    need not converge at all.  If `pnorm` settles while `max` climbs, then `c` is carrying
    the mesh into `util` and "the wheel is infeasible" and "the constraint is measured on a
    quantity that has no mesh-independent value" are different conclusions with different
    fixes.  So all three are extrapolated and reported side by side.

    THAT IS WHAT HAPPENED, and the constraint moved as a result: `util` is now
    `max(Kt_hub, Kt_rim) * pnorm(p=4) / allowable`, with the peak modelled analytically
    instead of measured off the singularity.  The `max` and `c` series stay in the report
    anyway — they are the evidence, and the `util` series only means something next to
    them.

    The stencil is FIXED and uniform across every row, so the only thing varying down a
    ladder is the mesh.  A row that fails to mesh or solve is recorded and the ladder
    continues: at `fine` this is 261k dof through contact, a service-force secant and an
    adjoint, which nothing in this repo has run before.

    `probe_p` — M8b-i.6 STEP 1 — RUNS THE WHOLE LADDER AGAIN AT NO COST.  The three series
    above answered "does the constraint converge" with a flat no, and decomposed it far
    enough to name the p-norm rather than the rescale as the reason.  What they could not
    say is whether the p-norm is unfixable or merely mis-exponented, and that is one
    argument: at p=30 the norm is 1/1.38 of the true max and inherits the corner's r^-0.5
    field, while M4's p99 of the same field converges to 8.61 MPa, so SOME smooth aggregate
    of it has a value.  Each `p` in `probe_p` gets its own `pnorm`/`c`/`util` series off
    the rows already measured — see `_series_by_p` — and the axle drop stays in the report
    as the control that makes them readable.  Values only, no gradient, no extra solve.
    """
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    probe_p = [float(v) for v in probe_p]
    out = []
    for label, genes in designs:
        rows = []
        for cfg in configs:
            t0 = time.time()
            try:
                loss, rep, wall = score(genes, cfg, phases=phases, probe_p=probe_p)
                rows.append({
                    "config": cfg,
                    "n_elements": int(WW.build_wheel(genes, cfg).n_elements),
                    "loss": loss,
                    "max_stress_mpa": float(rep["max_stress_mpa"]),
                    "pnorm_stress_agg_mpa": float(rep["pnorm_stress_agg_mpa"]),
                    "stress_scale_measured": float(rep["stress_scale_measured"]),
                    "kt_hub": float(rep["kt_hub"]), "kt_rim": float(rep["kt_rim"]),
                    "stress_utilisation": float(rep["stress_utilisation"]),
                    "stress_utilisation_hub": float(rep["stress_utilisation_hub"]),
                    "stress_utilisation_rim": float(rep["stress_utilisation_rim"]),
                    "axle_drop_mean_mm": float(rep["axle_drop_mean_mm"]),
                    "defl_err": float((rep["axle_drop_mean_mm"]
                                       - WO.TARGET_DEFLECTION_MM)
                                      / WO.TARGET_DEFLECTION_MM),
                    "stress_gauss_p": float(rep["stress_gauss_p"]),
                    "pnorm_by_p": rep["pnorm_by_p"],
                    "seconds": round(wall, 1)})
                print(f"      {label:<16s}{cfg:<8s}"
                      f"max {rows[-1]['max_stress_mpa']:7.2f} MPa   "
                      f"pnorm {rows[-1]['pnorm_stress_agg_mpa']:7.2f}   "
                      f"util {rows[-1]['stress_utilisation']:6.4f}   "
                      f"({rows[-1]['seconds']:.0f} s)", flush=True)
            except Exception as exc:                              # pragma: no cover
                rows.append({"config": cfg, "failed": type(exc).__name__,
                             "error": str(exc)[:200],
                             "seconds": round(time.time() - t0, 1)})
                print(f"      {label:<16s}{cfg:<8s}FAILED  "
                      f"{type(exc).__name__}: {str(exc)[:100]}", flush=True)
        ok = [r for r in rows if "failed" not in r]
        out.append({
            "label": label, "rows": rows, "n_ok": len(ok),
            "series": {name: _series(ok, key) for name, key in
                       (("max", "max_stress_mpa"),
                        ("pnorm", "pnorm_stress_agg_mpa"),
                        ("c", "stress_scale_measured"),
                        ("util", "stress_utilisation"),
                        ("drop", "axle_drop_mean_mm"))} if len(ok) >= 2 else {},
            "series_by_p": _series_by_p(ok, probe_p) if len(ok) >= 2 else {}})
    return {"configs": list(configs), "n_phase": n_phase, "scheme": "uniform",
            "probe_p": probe_p, "gate_ladder_gci": GATE_LADDER_GCI,
            "allowable_stress_mpa": WO.ALLOWABLE_STRESS_MPA, "designs": out,
            # NOT a convergence gate.  Whether the stress QoI converges is a fact about
            # the mesh and the geometry, and gating on it would report that fact as a
            # broken build — the same mistake S9 exists to avoid one level up.  What is
            # gated is that the ladder RAN: at least two rungs on every design, so a
            # sequence exists to read a direction off.
            "pass": bool(out) and all(d["n_ok"] >= 2 for d in out)}


# ---------------------------------------------------------------------------
# S12 — THE SAME QUESTION FROM SOMEWHERE ELSE
# ---------------------------------------------------------------------------

def run_multistart(cfg=DEFAULT_CONFIG, elites=None, n_phase=4, steps=20,
                   n_probe=2, kinds=("stress_only", "deflection_only"),
                   lr=S3.DEFAULT_LR):
    """*** THE SECOND HALF OF M8b-i.5 ***  Was the verdict about the space, or the basin?

    S9's three descents all started at `best_solution.json` and all plateaued.  That
    supports "no feasible design along three descents from one start", and the gap to "the
    design space contains no feasible point" is exactly what decides whether the genome
    needs new genes.  Two things make the single start suspect rather than merely narrow:
    `best_solution.json` is a GA optimum for the BEAM surrogate, which M8a measured to be a
    bad guide to the FEA — it scored `deflection` at 0.1% of the loss where the FEA says
    33.7% — so the GA optimised toward a corner chosen by a model now known to be wrong;
    and Adam is local, so a plateau is evidence about a basin, not about a space.

    `stage2_elites.json` has held sixteen distinct converged genomes this whole time and
    fifteen of them have never been scored against the FEA objective at all.  So:

      (a) score all sixteen, no descent.  Cheap, and the thing to look for is SPREAD — if
          every elite lands on top of the shipped genome, Stage 2 converged to one basin
          and the multi-start argument for M8b-ii is worth less than it looks.
      (b) re-run the two BOUNDS from the `n_probe` elites nearest the feasible corner.
          `joint` is dropped: it lands between the two by construction, and an hour of
          `coarse` descent buys nothing a bound does not already say.

    A genome that will not mesh is data.  Each elite is scored inside its own `try`, the
    failure is recorded by name, and the screen continues — `study_objective.py`'s G10
    table already does exactly this, and the alternative is one bad genome costing the
    other fifteen.
    """
    if elites is None:
        elites = S3.load_elites(limit=16)
    scored, failed = [], []
    for i, genes in enumerate(elites):
        try:
            loss, rep, wall = score(genes, cfg, n_phase=n_phase)
            err = (rep["axle_drop_mean_mm"] - WO.TARGET_DEFLECTION_MM) \
                / WO.TARGET_DEFLECTION_MM
            row = {"elite": i, "loss": loss,
                   "stress_utilisation": float(rep["stress_utilisation"]),
                   "stress_utilisation_hub": float(rep["stress_utilisation_hub"]),
                   "stress_utilisation_rim": float(rep["stress_utilisation_rim"]),
                   "kt_hub": float(rep["kt_hub"]), "kt_rim": float(rep["kt_rim"]),
                   "max_stress_mpa": float(rep["max_stress_mpa"]),
                   "axle_drop_mean_mm": float(rep["axle_drop_mean_mm"]),
                   "defl_err": float(err),
                   "corner_distance": corner_distance(rep["stress_utilisation"], err),
                   "feasible": bool(rep["stress_utilisation"] <= FEASIBLE_UTIL
                                    and abs(err) <= FEASIBLE_DEFL_REL),
                   "seconds": round(wall, 1)}
            scored.append(row)
            print(f"      elite {i:<2d}  util {row['stress_utilisation']:6.4f}   "
                  f"defl err {100 * err:+7.2f}%   d {row['corner_distance']:6.3f}   "
                  f"({row['seconds']:.0f} s)", flush=True)
        except Exception as exc:                                  # pragma: no cover
            failed.append({"elite": i, "failed": type(exc).__name__,
                           "error": str(exc)[:200]})
            print(f"      elite {i:<2d}  FAILED  {type(exc).__name__}: "
                  f"{str(exc)[:100]}", flush=True)

    ranked = sorted(scored, key=lambda r: r["corner_distance"])
    probes = []
    for row in ranked[:n_probe]:
        i = row["elite"]
        print(f"      probing elite {i} (d {row['corner_distance']:.3f}) "
              f"with {list(kinds)} ...", flush=True)
        f = run_feasibility(elites[i], cfg, steps=steps, n_phase=n_phase, lr=lr,
                            kinds=kinds)
        f["elite"] = i
        f["corner_distance_at_start"] = row["corner_distance"]
        probes.append(f)

    # -- the verdict, restated over every design ANY of this actually visited.
    seen_util = [r["stress_utilisation"] for r in scored]
    seen_err = [abs(r["defl_err"]) for r in scored]
    both = [r for r in scored if r["feasible"]]
    for f in probes:
        for run in f["runs"].values():
            seen_util += list(run["util_history"])
            seen_err += [abs(e) for e in run["defl_err_history"]]
        if f["verdict"]["simultaneously_reached"]:
            both.append({"elite": f["elite"], "from": "probe"})

    spread = {}
    if scored:
        u = np.array([r["stress_utilisation"] for r in scored])
        e = np.array([r["defl_err"] for r in scored])
        spread = {"util_min": float(u.min()), "util_max": float(u.max()),
                  "util_range": float(u.max() - u.min()),
                  "defl_err_min": float(e.min()), "defl_err_max": float(e.max()),
                  "defl_err_range": float(e.max() - e.min()),
                  "n_feasible_as_scored": int(sum(r["feasible"] for r in scored))}

    verdict = {
        "n_elites_scored": len(scored), "n_elites_failed": len(failed),
        "n_starts_probed": len(probes), "kinds": list(kinds),
        # (utilisation, deflection) PAIRS, not distinct designs: each probe re-scores its
        # own starting elite as step 0, so a few designs are measured more than once and
        # calling this a design count would overstate the coverage by exactly that
        # overlap.  What the verdict is quantified over is `n_starts_probed` and
        # `n_elites_scored`; this is the number of measurements behind the two minima.
        "n_points_measured": len(seen_util),
        "min_util_seen": float(np.nanmin(seen_util)) if seen_util else float("nan"),
        "min_abs_defl_err_seen": float(np.nanmin(seen_err)) if seen_err else float("nan"),
        "simultaneously_reached": bool(both),
        "spread": spread}
    return {"config": cfg if isinstance(cfg, str) else cfg.name, "n_phase": n_phase,
            "steps": steps, "feasible_util": FEASIBLE_UTIL,
            "feasible_defl_rel": FEASIBLE_DEFL_REL,
            "elites": scored, "failed": failed, "ranked": [r["elite"] for r in ranked],
            "probes": probes, "verdict": verdict,
            # As everywhere in this study: feasibility is REPORTED.  What is gated is that
            # the screen did its work — some elite got scored, and every re-probe moved and
            # accepted a step.  The second half is S9's rule (`run_feasibility`'s `pass`),
            # which exists because a deadlocked probe once reported its own starting point
            # as the lowest reachable utilisation.
            "pass": bool(scored) and all(f["pass"] for f in probes)}


# ---------------------------------------------------------------------------
# S13 — DOES THE PROCESS-PARALLEL PHASE BATCH BUY ANYTHING, AND IS IT THE SAME ANSWER?
# ---------------------------------------------------------------------------

def _worker_ladder(n_phase):
    """Powers of two up to what this machine can actually run, plus that cap.

    DERIVED FROM THE HOST, NOT WRITTEN DOWN.  A hardcoded `(1, 2, 4, 8)` measures
    oversubscription on a 4-core box — every rung past the cores is workers queueing for a
    core and the "speedup" it reports is an artifact of the machine, not of the code.  `1`
    is on the ladder deliberately: a one-worker pool does no concurrency at all, so its
    time against serial isolates what the pipe and the pickling cost from what the
    parallelism buys.
    """
    top = WP.default_workers(n_phase)
    ladder, n = [], 1
    while n < top:
        ladder.append(n)
        n *= 2
    ladder.append(top)
    return ladder


def _leaf_diffs(a, b, path=""):
    """Every differing leaf of two nested results, as `(path, |diff|, scale)`.

    `scale` is the serial side's own magnitude, so a caller can ask for exactness or for a
    RELATIVE tolerance off the same walk.  A structural mismatch — a missing key, a
    different length, a non-numeric leaf that moved — comes back with `inf`, so it can
    never be excused by any tolerance.
    """
    if isinstance(a, dict):
        if set(a) != set(b):
            return [(path + " (keys)", float("inf"), 0.0)]
        return [d for k in a for d in _leaf_diffs(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, np.ndarray):
        if np.array_equal(a, b):
            return []
        return [(path, float(np.max(np.abs(a - b))), float(np.max(np.abs(a))))]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [(path + " (length)", float("inf"), 0.0)]
        return [d for i, (x, y) in enumerate(zip(a, b))
                for d in _leaf_diffs(x, y, f"{path}[{i}]")]
    if isinstance(a, bool) or not isinstance(a, (int, float)):
        return [] if a == b else [(path, float("inf"), 0.0)]
    return [] if a == b else [(path, abs(a - b), abs(a))]


def _split_diffs(a, b, rel=GATE_POOL_GRAD_REL):
    """`(value_diffs, grad_diffs)` — exact for values, relative for gradients.

    TWO STANDARDS BECAUSE THERE ARE TWO SITUATIONS, not because one of them was hard.

    Values are gated EXACTLY, and they pass exactly: every forward value and every report
    leaf comes back from a pool bit-for-bit.  `pytest.approx` there would accept a pool
    that reduced its phases in completion order — the one failure this section exists to
    catch, since floating-point addition is not associative and an as-completed combine
    looks reproducible on a quiet machine and is not reproducible under load.

    Gradients cannot be gated exactly by anyone, pooled or not: see `GATE_POOL_GRAD_REL`
    for the two-serial-interpreters measurement that says so.  Anything gradient-derived
    therefore gets a relative tolerance, and everything else keeps the strict rule.
    """
    values, grads = [], []
    for path, adiff, scale in _leaf_diffs(a, b):
        if any(tag in path for tag in _GRAD_DERIVED):
            if adiff > rel * scale:
                grads.append((path, adiff, scale))
        else:
            values.append((path, adiff, scale))
    return values, grads


def _worst_rel(a, b):
    """The largest RELATIVE gradient difference, for reporting rather than gating."""
    worst = 0.0
    for path, adiff, scale in _leaf_diffs(a, b):
        if any(tag in path for tag in _GRAD_DERIVED) and scale > 0:
            worst = max(worst, adiff / scale)
    return worst


def run_phase_pool(genes, cfg=DEFAULT_CONFIG, n_phase=8, worker_counts=None, n_rep=2,
                   probe_p=(), prod_steps=300, prod_starts=4):
    """The same evaluation serial and pooled: identical answer, and how much faster.

    TWO CLAIMS, AND ONLY ONE OF THEM TRAVELS.  `identical` is a statement about the code
    and means the same thing on every machine.  The speedup does not — so it is gated as a
    FRACTION OF IDEAL (`speedup / n_workers`) rather than as an absolute multiple.  An
    absolute threshold passes trivially on a big box and fails unfairly on a small one,
    while the efficiency ratio fails in exactly the case worth catching: the pool not
    buying what its workers cost.  `cpu_count` is recorded beside every timing, because a
    wall-clock number without it cannot be compared to one measured elsewhere.

    BOTH SIDES ARE PRIMED BEFORE THE CLOCK STARTS, and without that the measurement is
    nonsense.  The first evaluation in any process pays the jax import, the `wheel_fem`
    kernel traces and one `coord_fn` trace per phase — measured at 0.774 s each — so an
    unprimed serial arm compared against a warm pooled one reports a speedup an order of
    magnitude too large.  What the priming costs is reported as `spawn_s` and
    `first_call_s` rather than hidden: a pool is built once per run and amortises over
    hundreds of steps, and that is a claim the numbers should let a reader check.

    Full tiers, not `("t3",)`.  T1 and T2 run in the PARENT on both paths — the pool only
    touches the phase loop — so including them is what makes the projected hours
    comparable to S10's 48.13 h instead of flattering the result by timing only the part
    that was parallelised.
    """
    low, high, _ = _bounds()
    z0 = wg.normalize(genes, low, high)
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    ori = tuple(float(o) for o in WW.flank_orientation(genes, WW.get_config(cfg)))
    counts = list(_worker_ladder(n_phase) if worker_counts is None else worker_counts)

    def evaluate(pool):
        ev = S3.Evaluator(cfg, orientation=ori, pool=pool, stress_p_probe=tuple(probe_p))
        t0 = time.time()
        out = ev(z0, low, high, phases=phases)
        first = time.time() - t0
        ts = []
        for _ in range(n_rep):
            t0 = time.time()
            out = ev(z0, low, high, phases=phases)
            ts.append(time.time() - t0)
        return out, float(np.median(ts)), first

    (s_val, s_grad, s_brk), serial_s, serial_first = evaluate(None)
    serial = {"loss": s_val, "grad": s_grad, "terms": s_brk["terms"],
              "report": s_brk["report"]}

    rows = []
    for n in counts:
        t0 = time.time()
        pool = WP.PhasePool(n)
        spawn_s = time.time() - t0
        try:
            (val, grad, brk), med, first = evaluate(pool)
        finally:
            pool.close()
        got = {"loss": val, "grad": grad, "terms": brk["terms"], "report": brk["report"]}
        vdiffs, gdiffs = _split_diffs(serial, got)
        rows.append({
            "workers": int(n), "seconds": med, "spawn_s": spawn_s,
            "first_call_s": first,
            "speedup": float(serial_s / med) if med else float("nan"),
            "efficiency": float(serial_s / med / n) if med else float("nan"),
            # The steady-state evidence: once the pool exists and every worker has traced
            # its own slots' phases, does a call cost what the next one will?  This is
            # what slot pinning is for — see `wheel_pool`'s docstring on `coord_fn`.
            "first_over_steady": float(first / med) if med else float("nan"),
            "identical_values": not vdiffs,
            "grads_within": not gdiffs,
            "worst_grad_rel": _worst_rel(serial, got),
            "value_diffs": [[p, f"{d:.3e}"] for p, d, _ in vdiffs[:8]],
            "grad_diffs": [[p, f"{d:.3e}", f"{s:.3e}"] for p, d, s in gdiffs[:8]],
            "projected_hours": float(prod_steps * prod_starts * med / 3600.0),
        })

    best = max(rows, key=lambda r: r["workers"])
    all_identical = all(r["identical_values"] for r in rows)
    all_grads_ok = all(r["grads_within"] for r in rows)
    # On a machine that can only run one worker there is no concurrency to measure and the
    # efficiency gate would fail for a reason that has nothing to do with this code.
    # `identical` still applies, and it is the claim that matters.
    measurable = best["workers"] > 1
    return {"config": cfg if isinstance(cfg, str) else cfg.name, "n_phase": n_phase,
            "cpu_count": os.cpu_count(), "worker_counts": counts, "n_rep": n_rep,
            "serial_s": serial_s, "serial_first_call_s": serial_first,
            "serial_projected_hours": float(prod_steps * prod_starts * serial_s / 3600.0),
            "projected_steps": prod_steps, "projected_starts": prod_starts,
            "rows": rows, "all_identical_values": all_identical,
            "all_grads_within": all_grads_ok,
            "worst_grad_rel": max(r["worst_grad_rel"] for r in rows),
            "best_workers": best["workers"], "best_speedup": best["speedup"],
            "best_efficiency": best["efficiency"],
            "gate_efficiency": GATE_POOL_EFFICIENCY,
            "gate_grad_rel": GATE_POOL_GRAD_REL,
            "efficiency_measurable": measurable,
            "pass": bool(all_identical and all_grads_ok
                         and (not measurable
                              or best["efficiency"] >= GATE_POOL_EFFICIENCY))}


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def head(s):
    print(f"\n{s}\n" + "-" * len(s))


def _print_direction(rep):
    d = rep["direction"]
    head(f"S1  descent direction vs an FD ladder of the whole pipeline  "
         f"[< {GATE_DIRECTION_REL:.0e}, >= {GATE_DIRECTION_RUNGS}]")
    print(f"    loss {d['loss']:.4f}   |grad| {d['grad_norm']:.6g}   "
          f"predicted g.d = {d['predicted']:.6g}")
    print(f"    {'t':>10s}{'central diff':>18s}{'rel err':>14s}")
    for r in d["rows"]:
        print(f"    {r['t']:10.1e}{r['fd']:18.6g}{r['rel']:14.3e}")
    print(f"    best {d['best_rel']:.3e} on {d['rungs']} rung(s)"
          f"   -> {'PASS' if d['pass'] else 'FAIL'}")


def _print_trajectory(rep):
    t = rep["trajectory"]
    head(f"S2/S3/S6  a deterministic {t['steps']}-step run at {t['n_phase']} phases")
    de, pr, orr = t["descent"], t["projection"], t["orientation"]
    print(f"    S2  loss {de['loss_start']:.4f} -> {de['loss_end']:.4f} "
          f"(best {de['loss_best']:.4f}, factor {de['factor']:.2f}x)"
          f"   -> {'PASS' if de['pass'] else 'FAIL'}")
    print(f"    S3  box violations {len(pr['box_violations'])}, freeze violations "
          f"{len(pr['freeze_violations'])}, pinned genes freed {pr['n_unpinned_moves']}"
          f"   -> {'PASS' if pr['pass'] else 'FAIL'}")
    print(f"        pinned at start: {', '.join(pr['pinned_at_start']) or '(none)'}")
    print(f"    S6  pin {orr['pinned']}, run recorded {orr['recorded_by_run']}, "
          f"mesh built {orr['mesh_built_with_pin']}")
    print(f"        free choice at the final design "
          f"{orr['free_choice_at_final_design']} "
          f"({'agrees' if orr['free_choice_still_agrees'] else 'STALE'}), "
          f"{orr['n_flip_events']} flip event(s)"
          f"   -> {'PASS' if orr['pass'] else 'FAIL'}")
    print(f"    ({t['wall_s']} s, {len(t['events'])} events)")


def _print_reject(rep):
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


def _print_schemes(rep):
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


def _print_warm(rep):
    w = rep["warm"]
    head("S8  warm start — the previous step's indentations as the secant's guess")
    print(f"    cold {w['cold_s_median']:.3f} s   warm {w['warm_s_median']:.3f} s   "
          f"saving {100 * w['saving_frac']:.1f}%"
          f"   -> {'PASS' if w['pass'] else 'FAIL'}")


def _print_cost(rep):
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

def _print_probes(f, indent="    "):
    """The probe table and its verdict — shared by S9 and by every multi-start re-probe."""
    v = f["verdict"]
    print(f"{indent}{'probe':<18s}{'loss':>22s}{'utilisation':>22s}"
          f"{'deflection error':>22s}")
    for k, d3 in f["runs"].items():
        print(f"{indent}{k:<18s}{d3['loss_start']:10.2f} ->{d3['loss_end']:10.2f}"
              f"{d3['util_start']:10.3f} ->{d3['util_end']:10.3f}"
              f"{100 * d3['defl_err_start']:9.1f}% ->{100 * d3['defl_err_end']:9.1f}%")
        flag = "" if d3["n_steps_accepted"] > 0 else "   <- NEVER MOVED, bounds nothing"
        print(f"{indent}  {d3['n_steps_accepted']:d}/{d3['n_steps_recorded'] - 1} steps "
              f"accepted, events {d3['event_kinds'] or '{}'}"
              f"{', STOPPED STUCK' if d3['stopped_stuck'] else ''}{flag}")
    print(f"\n{indent}lowest reachable stress utilisation   "
          f"{v['min_reachable_util']:.4f}"
          f"   (feasible at <= {f['feasible_util']:.2f}: "
          f"{'YES' if v['stress_reachable'] else 'NO'})")
    print(f"{indent}lowest reachable |deflection error|   "
          f"{100 * v['min_reachable_abs_defl_err']:.2f}%"
          f"   (feasible at <= {100 * f['feasible_defl_rel']:.0f}%: "
          f"{'YES' if v['deflection_reachable'] else 'NO'})")
    if "defl_err_when_stress_met" in v:
        print(f"{indent}stress met at deflection error        "
              f"{100 * v['defl_err_when_stress_met']:.1f}%")
    if "util_when_deflection_met" in v:
        print(f"{indent}deflection met at utilisation         "
              f"{v['util_when_deflection_met']:.3f}")
    print(f"{indent}BOTH satisfied at any visited design  "
          f"{'YES' if v['simultaneously_reached'] else 'NO'}")


def _print_feasibility(rep):
    f = rep["feasibility"]
    head("S9  THE FEASIBILITY VERDICT — reported, NOT gated")
    _print_probes(f)
    print(f"    -> {'PASS (probe ran)' if f['pass'] else 'FAIL (probe did not run)'}")


def _series_verdict(s):
    """The one place a `_series` becomes English, so the words cannot drift per caller.

    Driven off `converged` and never off `settling` — the M8b-i.5 figure captioned "the
    stress QoI settles under refinement" above a 63%-GCI utilisation is what that costs.
    """
    if s["converged"] is None:
        return "n/a (<3 rungs)"
    if s["converged"]:
        return f"CONVERGED (GCI < {100 * GATE_LADDER_GCI:.0f}%)"
    return "NOT CONVERGED" + ("" if s["settling"] else ", not even settling")


def _series_line(s):
    return (f"{100 * s['total_rel_change']:11.1f}%{100 * s['rel_change_last_pair']:11.1f}%"
            f"{s['ratio']:9.2f}{_order(s):7.2f}{s['richardson']:12.3f}"
            f"{100 * s['gci']:8.2f}%   {s['direction']}, {_series_verdict(s)}")


def _order(s):
    """Observed order from the refinement ratio.  Derived, never stored — a ratio is what
    `_series` measures and `log2` is only the reading of it under 2x refinement."""
    return np.log2(s["ratio"]) if s.get("ratio", 0) > 0 else float("nan")


def _print_by_p(d, m):
    """M8b-i.6 step 1's table: does ANY exponent give the constraint a value?

    The axle drop is reprinted at the foot rather than left three tables up, because it is
    the standard every row above is read against — a GCI means nothing without knowing what
    a converged QoI scores on these same meshes.

    TWO UTILISATION COLUMNS, AND THE GAP BETWEEN THEM IS STEP 2.  `util_c` is the old
    constraint, `c * pnorm / allowable`, kept because it is the evidence: it converges at
    no exponent, at either design.  `util_Kt` is what the constraint became — an analytic
    concentration factor on a nominal stress — and it converges wherever the p-norm does,
    which is what choosing `p = STRESS_NOMINAL_P` buys.

    `c` IS PRINTED, not just stored.  `util = c * pnorm / allowable` is a product, and
    M8b-i.5's one durable methodological lesson is that reporting the decomposition rather
    than the verdict is what made the next step right: the whole reason this milestone
    targets `p` instead of the rescale is that `c` was visible in that table and turned out
    to be the BEST-behaved factor (GCI 5.33%) while the p-norm sat at 47%.  Printing `util`
    against `pnorm` alone would re-hide exactly the factor that decided the last call.
    `_series_by_p` has computed this leaf all along; it simply never reached the page.
    """
    print(f"\n    {'p':<9s}{'pnorm MPa up the ladder':<34s}{'order':>7s}{'GCI':>9s}"
          f"{'c':>8s}{'c GCI':>9s}{'util_c':>9s}{'util GCI':>10s}"
          f"{'util_Kt':>10s}{'util GCI':>10s}   verdict")
    for k in sorted(d["series_by_p"], key=lambda k: d["series_by_p"][k]["p"]):
        b = d["series_by_p"][k]
        pn, c, ut, uk = b["pnorm"], b["c"], b["util"], b.get("util_kt")
        vals = " ".join(f"{v:.3f}" for v in pn["values"])
        # `util_Kt` inherits the p-norm's convergence exactly — `Kt` is a constant of the
        # mesh — so its GCI column IS the p-norm's, and that is the payoff rather than a
        # redundancy: it is the same number the constraint now uses.
        kt_col = (f"{uk['values'][-1]:10.4f}{100 * uk['gci']:9.2f}%" if uk
                  else f"{'':20s}")
        print(f"    {b['p']:<9.4g}{vals:<34s}{_order(pn):7.2f}{100 * pn['gci']:8.2f}%"
              f"{c['values'][-1]:8.3f}{100 * c['gci']:8.2f}%"
              f"{ut['values'][-1]:9.4f}{100 * ut['gci']:9.2f}%"
              f"{kt_col}   {_series_verdict(pn)}")
    drop = d["series"].get("drop")
    if drop:
        print(f"    {'drop':<9s}{' '.join(f'{v:.4f}' for v in drop['values']):<34s}"
              f"{_order(drop):7.2f}{100 * drop['gci']:8.2f}%{'':36s}   "
              f"{_series_verdict(drop)}   <- THE CONTROL")


def _print_mesh_convergence(rep):
    m = rep["mesh_convergence"]
    head(f"S11  IS THE STRESS QoI CONVERGED?  the ladder at {m['n_phase']} fixed phases "
         f"— reported, NOT gated")
    for d in m["designs"]:
        print(f"\n    {d['label']}")
        print(f"    {'config':<9s}{'elements':>10s}{'max MPa':>10s}{'pnorm MPa':>11s}"
              f"{'c=max/pn':>10s}{'Kt':>7s}{'util':>9s}{'drop mm':>10s}{'s':>8s}")
        for r in d["rows"]:
            if "failed" in r:
                print(f"    {r['config']:<9s}{'FAILED':>10s}   {r['failed']}: "
                      f"{r['error'][:40]}")
                continue
            print(f"    {r['config']:<9s}{r['n_elements']:10d}"
                  f"{r['max_stress_mpa']:10.2f}{r['pnorm_stress_agg_mpa']:11.2f}"
                  f"{r['stress_scale_measured']:10.4f}"
                  f"{max(r['kt_hub'], r['kt_rim']):7.3f}"
                  f"{r['stress_utilisation']:9.4f}"
                  f"{r['axle_drop_mean_mm']:10.4f}{r['seconds']:8.0f}")
        if not d["series"]:
            continue
        print(f"    {'series':<9s}{'total move':>12s}{'last pair':>12s}"
              f"{'ratio':>9s}{'order':>7s}{'richardson':>12s}{'GCI':>9s}   verdict")
        for name in ("max", "pnorm", "c", "util", "drop"):
            # `.get`, because a design whose ladder lost a rung has no entry here and a
            # KeyError would throw away every solve this print exists to report.
            s = d["series"].get(name)
            if s is None:
                continue
            print(f"    {name:<9s}{_series_line(s)}")
        if d.get("series_by_p"):
            _print_by_p(d, m)
    print(f"\n    -> {'PASS (ladder ran)' if m['pass'] else 'FAIL (ladder did not run)'}")
    print(f"\n    HOW TO READ THIS.  `util` is `c * pnorm / {m['allowable_stress_mpa']:.0f}`"
          f", so it inherits whatever `pnorm`")
    print(f"    and `c` do.  `drop` is the CONTROL: it comes out of the same solves on the")
    print(f"    same meshes, so any explanation that blames the mesher, the contact solve,")
    print(f"    the phase stencil or the adjoint has to explain why the axle drop is")
    print(f"    unaffected by it.")
    print(f"\n    The hypothesis this section was built to test was that `max` diverges")
    print(f"    while `pnorm` converges — `_qoi_pnorm_stress` argues the volume-weighted")
    print(f"    p-norm is a quadrature of an integral — leaving `c = max/pnorm` to carry")
    print(f"    the mesh into `util`.  Read the `c` row against the `pnorm` row before")
    print(f"    accepting it: at p=30 the p-norm is ~1/1.38 of the max, which is a max in")
    print(f"    disguise, and an unfilleted spoke/ring junction is a 349.5-degree re-entrant")
    print(f"    corner whose r^-0.5 field no mesh resolves — see study_wheel_fea's")
    print(f"    `stress_report`, which measured this at M4 and prescribed a PERCENTILE.")
    print(f"    A p-norm that does not converge is not fixed by a gene or by a weight; it")
    print(f"    is fixed by lowering `p` until it does, and by replacing the rescale-to-max")
    print(f"    with an analytic Kt (`wheel_fea.stress_concentration_kt`).")
    if m.get("probe_p"):
        print(f"\n    THE SWEEP (M8b-i.6 step 1).  Exponents {m['probe_p']} measured off the")
        print(f"    SAME solves — no extra mesh, no extra Newton, no extra adjoint — so the")
        print(f"    rows differ in `p` and in nothing else.  `p = {WA.STRESS_PNORM_P:.0f}` "
              f"is the shipped")
        print(f"    default and must reproduce the `pnorm` row above exactly; if it does not,")
        print(f"    the probe is wrong and no other row here means anything.  A `p` whose GCI")
        print(f"    lands under {100 * GATE_LADDER_GCI:.0f}% is a stress constraint with a "
              f"value, which is the")
        print(f"    prerequisite M8b-i.6 steps 2 and 3 are waiting on.")


def _print_multistart(rep):
    m = rep["multistart"]
    v, sp = m["verdict"], m["verdict"]["spread"]
    head(f"S12  THE SAME QUESTION FROM {v['n_elites_scored']} OTHER STARTS "
         f"— reported, NOT gated")
    print(f"    (a) every Stage-2 elite scored against the FEA objective, no descent")
    print(f"    {'elite':<8s}{'loss':>12s}{'utilisation':>14s}{'defl err':>12s}"
          f"{'max MPa':>10s}{'corner d':>11s}{'feasible':>10s}")
    for r in m["elites"]:
        print(f"    {r['elite']:<8d}{r['loss']:12.2f}{r['stress_utilisation']:14.4f}"
              f"{100 * r['defl_err']:11.2f}%{r['max_stress_mpa']:10.2f}"
              f"{r['corner_distance']:11.3f}{'YES' if r['feasible'] else 'no':>10s}")
    for r in m["failed"]:
        print(f"    {r['elite']:<8d}FAILED  {r['failed']}: {r['error'][:50]}")
    if sp:
        print(f"\n    SPREAD, which is the question: utilisation "
              f"{sp['util_min']:.4f}-{sp['util_max']:.4f} "
              f"(range {sp['util_range']:.4f}),")
        print(f"    deflection error {100 * sp['defl_err_min']:.2f}% to "
              f"{100 * sp['defl_err_max']:.2f}% (range "
              f"{100 * sp['defl_err_range']:.2f} points).")
        print(f"    {sp['n_feasible_as_scored']} of {v['n_elites_scored']} elites are "
              f"feasible as scored, before any descent.")
        # Stated against the measurement, not over it.  A narrow spread would mean Stage 2
        # converged to one basin, and a multi-start over these elites would then be a
        # weaker argument than its count suggests — so the reading has to depend on the
        # number rather than assert one of the two outcomes in advance.
        wide = (sp["defl_err_range"] > 4 * FEASIBLE_DEFL_REL
                or sp["util_range"] > 0.1 * FEASIBLE_UTIL)
        if wide:
            print(f"    That is a WIDE spread: these elites are not one basin, so the "
                  f"probes below")
            print(f"    start somewhere S9's three descents never reached.")
        else:
            print(f"    That is a NARROW spread: Stage 2 converged to ONE basin, and a "
                  f"multi-start")
            print(f"    over these elites is a weaker argument than its count suggests.")

    for f in m["probes"]:
        print(f"\n    (b) elite {f['elite']} re-probed "
              f"(corner distance {f['corner_distance_at_start']:.3f} at the start)")
        _print_probes(f, indent="        ")

    print(f"\n    (c) THE VERDICT, RESTATED OVER {v['n_starts_probed']} PROBED START(S) "
          f"AND {v['n_elites_scored']} SCORED")
    print(f"    (util, deflection) pairs measured {v['n_points_measured']}")
    print(f"    lowest utilisation seen anywhere {v['min_util_seen']:.4f}"
          f"   (feasible at <= {m['feasible_util']:.2f})")
    print(f"    smallest |deflection error| seen {100 * v['min_abs_defl_err_seen']:.2f}%"
          f"   (feasible at <= {100 * m['feasible_defl_rel']:.0f}%)")
    print(f"    BOTH satisfied at any design     "
          f"{'YES' if v['simultaneously_reached'] else 'NO'}")
    print(f"    -> {'PASS (screen ran)' if m['pass'] else 'FAIL (screen did not run)'}")


def _print_phase_pool(rep):
    m = rep["phase_pool"]
    head(f"S13  THE PROCESS-PARALLEL PHASE BATCH  ({m['config']}, {m['n_phase']} phases, "
         f"{m['cpu_count']} cores)")
    print(f"    serial {m['serial_s']:.1f} s  "
          f"(first call {m['serial_first_call_s']:.1f} s, traces included)")
    print("    workers   seconds   speedup   efficiency   1st/steady   values   grad rel")
    for r in m["rows"]:
        print(f"    {r['workers']:7d}  {r['seconds']:8.1f}  {r['speedup']:8.2f}x  "
              f"{r['efficiency']:10.2f}   {r['first_over_steady']:10.2f}   "
              f"{'EXACT' if r['identical_values'] else 'DIFFER':>6s}   "
              f"{r['worst_grad_rel']:8.1e}"
              f"{'' if r['grads_within'] else '  OVER GATE ' + str(r['grad_diffs'][:2])}")
        if not r["identical_values"]:
            print(f"            value diffs: {r['value_diffs'][:3]}")
    print(f"\n    projected production run ({m['projected_steps']} steps x "
          f"{m['projected_starts']} starts):")
    print(f"      serial              {m['serial_projected_hours']:8.2f} h")
    best = max(m["rows"], key=lambda r: r["workers"])
    print(f"      {best['workers']} workers          {best['projected_hours']:8.2f} h"
          f"   <- on {m['cpu_count']} cores; the hours do not travel, the ratio does")
    if not m["efficiency_measurable"]:
        print("\n    efficiency NOT GATED: this machine can run only one worker, so there "
              "is no concurrency to measure. `identical` still applies.")
    print("\n    Values are gated EXACTLY and gradients relatively, because that is what")
    print("    is available: two plain SERIAL runs in separate interpreters already")
    print("    disagree in the adjoint's last bit (3.33e-16), with no pool involved.")
    print(f"\n    -> {'PASS' if m['pass'] else 'FAIL'}"
          f"  (values {'exact' if m['all_identical_values'] else 'DIFFER'}, "
          f"worst grad rel {m['worst_grad_rel']:.1e} vs gate {m['gate_grad_rel']:.0e}, "
          f"efficiency {m['best_efficiency']:.2f} vs gate {m['gate_efficiency']:.2f})")


PRINTERS = {"direction": _print_direction, "trajectory": _print_trajectory,
            "reject": _print_reject, "schemes": _print_schemes, "warm": _print_warm,
            "cost": _print_cost, "feasibility": _print_feasibility,
            "mesh_convergence": _print_mesh_convergence,
            "multistart": _print_multistart, "phase_pool": _print_phase_pool}


def _print_tail(rep):
    print(f"\n{'=' * 78}")
    print(f"  OVERALL: {'PASS' if rep['pass'] else 'FAIL'}")
    print("=" * 78)
    print("\n  DONE, of M8b-ii: the process-parallel phase batch. `--workers` on")
    print("  wheel_stage3.py, and S13 (`make m8bii1`) is the gate: every VALUE bit-for-bit")
    print("  against serial, gradients to 1e-14 because two plain serial interpreters")
    print("  already disagree in the adjoint's last bit with no pool involved.")
    print("  NOT DONE: the multi-fidelity checkpoints, the `t1_vector` jit and the")
    print("  300-step multi-start production run.")
    if "multistart" in rep or "feasibility" in rep:
        print("  The feasibility sections above are the measurement that says whether")
        print("  funding that run is sensible.")
    print("  `lambda_min(K_t)` remains M9; `buckling` is still the zero-gradient Euler")
    print("  proxy, and a diverged tangent is the only buckling signal this run has.")


def _print(rep):
    """Whatever sections `rep` holds, in the order they were run."""
    print("=" * 78)
    print(f"  {rep.get('settings', {}).get('title', 'M8b-i GATE — THE STAGE-3 OPTIMIZER')}")
    print("=" * 78)
    for name in rep:
        if name in PRINTERS:
            PRINTERS[name](rep)
    _print_tail(rep)


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


def _plot_by_p(a, m):
    """GCI against `p` — M8b-i.6 step 1's whole answer on one pair of axes.

    GCI on a log axis because the interesting span is 0.1% to 60%, and the question is
    which side of `GATE_LADDER_GCI` a curve falls on rather than by how much.  The axle
    drop is drawn as a horizontal reference for the reason it is reprinted in the table:
    it is a converged QoI off these same solves, so it fixes what the y-axis means.

    The title states which `p` converged, or that none did.  It is read off `converged`,
    never off `settling` — see `_series`.
    """
    best, best_util, measurable = None, None, False
    for j, d in enumerate(m["designs"]):
        by = d.get("series_by_p") or {}
        if not by:
            continue
        ks = sorted(by, key=lambda k: by[k]["p"])
        ps = [by[k]["p"] for k in ks]
        col = f"C{j}"
        a.plot(ps, [100 * by[k]["pnorm"]["gci"] for k in ks], "o-", color=col, lw=1.3,
               ms=5, label=f"{d['label']}: pnorm")
        a.plot(ps, [100 * by[k]["util"]["gci"] for k in ks], "s--", color=col, lw=1.0,
               ms=4, alpha=0.75, label=f"{d['label']}: util")
        drop = d["series"].get("drop") or {}
        if np.isfinite(drop.get("gci", np.nan)):
            a.axhline(100 * drop["gci"], color=col, ls="-.", lw=0.8, alpha=0.5,
                      label=f"{d['label']}: axle drop (the control)")
        measurable |= any(np.isfinite(by[k]["pnorm"]["gci"]) for k in ks)
        # EVERY design, not just the first.  Scoping this to `j == 0` made the title a
        # property of whichever design happened to be listed first: a shipped genome that
        # lost a rung would caption the figure "no exponent gives the stress p-norm a
        # mesh-independent value" over a panel in which elite 1's curve visibly dips under
        # the gate.  Same fault as the caption M8b-i.5 had to regenerate — a title
        # asserting something the axes underneath it contradict.  The design is named
        # because a `p` that converges at one genome and not the other is not a constraint.
        conv = [by[k]["p"] for k in ks if by[k]["pnorm"]["converged"]]
        if conv and (best is None or max(conv) > best[0]):
            best = (max(conv), d["label"])
        # AND THE CONSTRAINT SEPARATELY.  `util = c * pnorm / allowable` is a PRODUCT, and
        # the p-norm is only one of its two factors.  A title reporting the p-norm's verdict
        # alone captions this panel with the good half and invites the reading that lowering
        # `p` fixed the constraint.  Measured here it does the opposite: the p-norm converges
        # for p <= 6 while `util` converges at NO exponent and is WORST where the p-norm is
        # best, because `c = max/pnorm` diverges faster as `p` falls away from the max.
        convu = [by[k]["p"] for k in ks if by[k]["util"]["converged"]]
        if convu and (best_util is None or max(convu) > best_util[0]):
            best_util = (max(convu), d["label"])

    a.axhline(100 * GATE_LADDER_GCI, color="k", ls=":", lw=1.1)
    a.text(0.02, 100 * GATE_LADDER_GCI, f" converged below {100 * GATE_LADDER_GCI:.0f}%",
           transform=a.get_yaxis_transform(), fontsize=6.5, va="bottom")
    a.axvline(WA.STRESS_PNORM_P, color="k", ls="--", lw=0.7, alpha=0.45)
    a.text(WA.STRESS_PNORM_P, 0.98, f"shipped p={WA.STRESS_PNORM_P:.0f} ",
           transform=a.get_xaxis_transform(), fontsize=6.5, ha="right", va="top",
           rotation=90)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("Gauss-point p-norm exponent  p")
    a.set_ylabel("GCI  [%]   (lower is converged)")
    # Three states, not two.  "No exponent converged" and "no exponent was MEASURABLE"
    # look identical on an empty panel, and only one of them is a finding — a two-rung
    # ladder has no GCI at any `p`, so a title claiming a negative verdict there would
    # assert a conclusion its own axes contain no evidence for.  That is the exact fault
    # M8b-i.5 shipped once already, in the other direction.
    #
    # AND THE p-NORM'S VERDICT IS NOT THE CONSTRAINT'S.  Both are stated, because the
    # measured answer is that they DISAGREE — and a title carrying only the first half
    # ("converges up to p = 6") over a panel whose dashed curves never leave 60-130% would
    # be the third caption in this milestone to claim more than its own axes support.
    if best is None:
        title = ("no exponent gives the stress p-norm a mesh-independent value"
                 if measurable else
                 f"ladder too short to tell — a GCI needs 3 rungs, this has "
                 f"{len(m['configs'])}")
    elif best_util is None:
        # Wrapped, not shortened.  Both halves are the finding, and a title that runs off
        # its own axes into the neighbouring panel is a caption nobody reads to the end.
        title = (f"the stress p-norm converges up to p = {best[0]:.4g}  ({best[1]})\n"
                 f"but the CONSTRAINT converges at no p — the rescale, not the exponent")
    else:
        title = (f"the stress p-norm converges up to p = {best[0]:.4g}  ({best[1]})\n"
                 f"the constraint up to p = {best_util[0]:.4g}  ({best_util[1]})")
    a.set_title(title, fontsize=9)
    a.grid(alpha=0.3, which="both")
    a.legend(fontsize=6.5, loc="best")


def _plot_pool(rep, path):
    """S13 on one sheet: what it costs, and how much of the ideal that is.

    THE IDEAL LINE IS DRAWN, and it is the point of the left panel.  A wall-clock curve
    alone looks impressive on any machine with enough cores; against `serial / n` it shows
    where Amdahl and the unequal phases actually bite, and it is the same picture on four
    cores as on sixty-four.  The core count is in the title because these seconds are the
    one thing in this figure that does not travel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = rep["phase_pool"]
    ws = [r["workers"] for r in m["rows"]]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2))

    ax[0].plot(ws, [r["seconds"] for r in m["rows"]], "o-", lw=1.4, ms=5, label="measured")
    ax[0].plot(ws, [m["serial_s"] / w for w in ws], "--", lw=1, color="0.5",
               label="ideal (serial / workers)")
    ax[0].axhline(m["serial_s"], color="C3", lw=1, ls=":", label="serial")
    ax[0].set_xscale("log", base=2)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("phase workers")
    ax[0].set_ylabel("seconds per evaluation")
    ax[0].set_title(f"one {m['config']} evaluation, {m['n_phase']} phases "
                    f"({m['cpu_count']} cores)", fontsize=9)
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7)

    ax[1].plot(ws, [r["efficiency"] for r in m["rows"]], "o-", lw=1.4, ms=5)
    ax[1].axhline(m["gate_efficiency"], color="C3", lw=1, ls="--",
                  label=f"gate {m['gate_efficiency']:.2f}")
    ax[1].axhline(1.0, color="0.5", lw=1, ls=":", label="ideal")
    ax[1].set_xscale("log", base=2)
    ax[1].set_ylim(0.0, 1.15)
    ax[1].set_xlabel("phase workers")
    ax[1].set_ylabel("speedup / workers")
    same = (f"values exact vs serial, gradients to {m['worst_grad_rel']:.0e}"
            if m["all_identical_values"] and m["all_grads_within"] else
            "POOLED AND SERIAL DISAGREE — see the report")
    ax[1].set_title(f"efficiency; {same}", fontsize=9)
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7)

    fig.suptitle("M8b-ii S13 — the process-parallel phase batch", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = path if os.path.isabs(path) else os.path.join(HERE, path)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _plot_m8bi5(rep, path):
    """The two things that qualify S9's verdict, on one sheet.

    The right panel is drawn on the SAME axes as `_plot`'s third — same feasible
    rectangle, same star for the shipped genome — because the whole point is to read the
    two figures against each other and see whether sixteen other starts land anywhere S9's
    three did not.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    panels = [k for k in ("mesh_convergence", "multistart") if k in rep]
    # M8b-i.6's panel only exists when a sweep was asked for, and it goes FIRST: it is the
    # answer, and the ladder beside it is the evidence the answer is read off.
    if rep.get("mesh_convergence", {}).get("probe_p"):
        panels.insert(0, "by_p")
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 4.4),
                             squeeze=False)
    ax = dict(zip(panels, axes[0]))

    if "by_p" in ax:
        _plot_by_p(ax["by_p"], rep["mesh_convergence"])

    if "mesh_convergence" in ax:
        a, m = ax["mesh_convergence"], rep["mesh_convergence"]
        for j, d in enumerate(m["designs"]):
            ok = [r for r in d["rows"] if "failed" not in r]
            if not ok:
                continue
            n = [r["n_elements"] for r in ok]
            col = f"C{j}"
            a.plot(n, [r["max_stress_mpa"] for r in ok], "o-", color=col, lw=1.2, ms=5,
                   label=f"{d['label']}: true max")
            a.plot(n, [r["stress_utilisation"] * m["allowable_stress_mpa"] for r in ok],
                   "s--", color=col, lw=1.0, ms=4, alpha=0.8,
                   label=f"{d['label']}: c x pnorm (what util uses)")
            a.plot(n, [r["pnorm_stress_agg_mpa"] for r in ok], "^:", color=col, lw=1.0,
                   ms=4, alpha=0.6, label=f"{d['label']}: pnorm")
            for key, style in (("max", "-"), ("pnorm", ":")):
                s = d["series"].get(key, {})
                if np.isfinite(s.get("richardson", np.nan)):
                    a.axhline(s["richardson"], color=col, ls=style, lw=0.7, alpha=0.35)
        a.axhline(m["allowable_stress_mpa"], color="k", ls="--", lw=1.0,
                  label=f"allowable {m['allowable_stress_mpa']:.0f} MPa")
        a.set_xscale("log")
        a.set_xlabel("elements in the wheel mesh")
        a.set_ylabel("stress  [MPa]")
        # The title states what was measured, so it cannot outlive the measurement.  Driven
        # off `converged` (GCI inside GATE_LADDER_GCI) and NOT off `settling`, which is only
        # "the differences are shrinking" and is true of every divergent-looking series
        # here — reading the title off it once produced a plot captioned "the stress QoI
        # settles under refinement" above a utilisation with a 63% GCI.
        s0 = m["designs"][0]["series"] if m["designs"] and m["designs"][0]["series"] else {}
        cmax = s0.get("max", {}).get("converged")
        cpn = s0.get("pnorm", {}).get("converged")
        cdrop = s0.get("drop", {}).get("converged")
        if cmax and cpn:
            a.set_title("the stress QoI converges under refinement")
        elif cpn:
            a.set_title("the p-norm converges; the true max does not")
        elif cdrop:
            # The interesting case, and the measured one: the control converges on the very
            # same meshes, so the stress QoI's failure is the QoI's and not the mesh's.
            a.set_title("the stress QoI does not converge — while the axle drop, "
                        "same meshes, does")
        else:
            a.set_title("the stress QoI does not converge under refinement")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=6.5, loc="best")

    if "multistart" in ax:
        a, m = ax["multistart"], rep["multistart"]
        e = 100 * m["feasible_defl_rel"]
        a.add_patch(Rectangle((-e, 0.0), 2 * e, m["feasible_util"], facecolor="C2",
                              alpha=0.20, edgecolor="C2", lw=1.0, ls="--", zorder=0,
                              label="feasible"))
        for f in m["probes"]:
            for k, col in (("stress_only", "C0"), ("deflection_only", "C1"),
                           ("joint", "C3")):
                if k not in f["runs"]:
                    continue
                r = f["runs"][k]
                a.plot(100 * np.array(r["defl_err_history"]), r["util_history"], "-",
                       lw=1, color=col, alpha=0.85,
                       label=f"elite {f['elite']}: {k}")
        if m["elites"]:
            a.plot([100 * r["defl_err"] for r in m["elites"]],
                   [r["stress_utilisation"] for r in m["elites"]], "o", ms=5,
                   color="C4", alpha=0.9, label=f"{len(m['elites'])} elites, as scored")
        # The shipped genome, from whichever section measured it at this config.
        for d in rep.get("mesh_convergence", {}).get("designs", []):
            row = [r for r in d["rows"]
                   if r.get("config") == m["config"] and "failed" not in r]
            if d["label"].startswith("best") and row:
                a.plot(100 * row[0]["defl_err"], row[0]["stress_utilisation"], "k*",
                       ms=11, label="shipped genome", zorder=5)
        a.axhline(m["feasible_util"], color="k", ls=":", lw=0.9)
        a.set_xlabel("deflection error  [%]")
        a.set_ylabel("stress utilisation")
        v = m["verdict"]
        a.set_title("a feasible design was found"
                    if v["simultaneously_reached"] else
                    f"no feasible design in {v['n_points_measured']} measurements "
                    f"from {v['n_elites_scored']} starts")
        a.grid(alpha=0.3)
        a.legend(fontsize=6.5, loc="best")

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
    ap.add_argument("--sections", default=",".join(DEFAULT_SECTIONS),
                    help="comma-separated, run in the order given.  The default is the "
                         "M8b-i gate, S1-S10.  M8b-i.5's `mesh_convergence` and "
                         "`multistart` are OPT-IN: at `coarse` they are ~2 h on top of "
                         "the gate's ~2 h 45 m, and making `make studies` a five-hour "
                         "job to re-measure something that does not change per commit "
                         "is how a gate stops being run at all.  See `make m8bi5`.")
    ap.add_argument("--ladder-configs", default=",".join(LADDER_CONFIGS),
                    help="the S11 mesh ladder.  `fine` is 261k dof through contact, a "
                         "service-force secant and an adjoint, which nothing in this "
                         "repo has run before — hence opt-in.")
    ap.add_argument("--ladder-p", default="",
                    help="M8b-i.6 step 1: extra Gauss-point p-norm exponents to measure "
                         "up the S11 ladder, e.g. `2,4,8,16,30`.  Empty (the default) "
                         "leaves the report as M8b-i.5 wrote it.  Each exponent is read "
                         "off the displacement field the adjoint already converged, so "
                         "the sweep costs NO extra solve — the whole list is the same ~14 "
                         "min as one column.")
    ap.add_argument("--n-probe", type=int, default=2,
                    help="S12: how many elites to re-descend from, nearest corner first")
    ap.add_argument("--pool-workers", default="",
                    help="S13: comma-separated worker counts to measure, e.g. `1,2,4,8`. "
                         "Empty (the default) derives the ladder from this machine — "
                         "powers of two up to min(n_phase, cpu_count) — so the section "
                         "measures concurrency rather than oversubscription wherever it "
                         "runs.")
    args = ap.parse_args()

    try:
        sections = parse_sections(args.sections)
        probe_p = parse_ladder_p(args.ladder_p)
    except ValueError as exc:
        raise SystemExit(str(exc))

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
    # M8b-i.5.  `--quick` cuts the ladder to two rungs, which is a wiring check and not a
    # convergence measurement — with two points there is no successive-difference ratio
    # and `_series` says so rather than extrapolating from nothing.
    ladder_cfgs = ("smoke", "coarse") if args.quick else \
        tuple(c.strip() for c in args.ladder_configs.split(",") if c.strip())
    # M8b-i.6.  `--quick` keeps the ENDS of whatever sweep was asked for and drops the
    # middle: two exponents exercise the whole probe path, and the two that matter for a
    # wiring check are the extremes, where the p-norm is least and most like the max.
    probe_p = probe_p[:1] + probe_p[-1:] if args.quick and len(probe_p) > 2 else probe_p
    n_elite = 2 if args.quick else 16
    n_probe = 1 if args.quick else args.n_probe
    # M8b-ii S13.  `None` means "ask the machine" — see `_worker_ladder`.  `--quick` pins
    # the ladder to a single 2-worker rung: two workers are spawnable on a one-core CI
    # runner, so the wiring check asserts the same thing everywhere, which a
    # host-derived ladder by construction cannot.
    pool_workers = ([2] if args.quick else
                    [int(w) for w in args.pool_workers.split(",") if w.strip()] or None)

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

    # The registry, not a call.  `--sections` decides which of these run and in what
    # order; the default list is S1-S10 in the order they have always run, so a default
    # invocation writes the same report keys, in the same order, that it always has.
    runners = {
        "direction":
            lambda: run_direction(genes, cfg, n_phase=n_phase, steps=ladder),
        "trajectory":
            lambda: run_trajectory(genes, cfg, steps=traj_steps, n_phase=n_phase),
        "reject":
            lambda: run_reject(genes, cfg, n_phase=n_phase),
        "schemes":
            lambda: run_phase_schemes(genes, cfg, steps=scheme_steps, n_phase=n_phase),
        "warm":
            lambda: run_warm(genes, cfg, n_phase=n_phase),
        "cost":
            lambda: run_cost(genes, cfg, n_phase=cost_phase),
        "feasibility":
            lambda: run_feasibility(genes, cfg, steps=feas_steps, n_phase=n_phase),
        # Two designs: the shipped genome, whose 1.2406 -> 1.7128 is the measurement S11
        # exists to explain, and elite 1.  NOT elite 0 — that is `best_solution.json`
        # bit-for-bit (checked: max|diff| = 0.0 across all fourteen genes), so a ladder
        # over the first two elites would refine the same wheel twice and report the
        # agreement as corroboration.  Elite 1 misses the target from the OTHER side,
        # which is what makes a second ladder worth its minutes.
        "mesh_convergence":
            lambda: run_mesh_convergence(
                [("best_solution", genes)]
                + [(f"elite{k + 1}", g) for k, g in
                   enumerate(S3.load_elites(args.elites, limit=2)[1:])],
                configs=ladder_cfgs, n_phase=n_phase, probe_p=probe_p),
        "multistart":
            lambda: run_multistart(
                cfg, elites=S3.load_elites(args.elites, limit=n_elite),
                n_phase=n_phase, steps=feas_steps, n_probe=n_probe),
        # `cost_phase`, so S13's evaluation is the SAME evaluation S10 projected 48.13 h
        # from.  A speedup quoted against a different phase count is not a speedup on the
        # thing anyone is waiting for.
        "phase_pool":
            lambda: run_phase_pool(genes, cfg, n_phase=cost_phase,
                                   worker_counts=pool_workers),
    }
    for name in sections:
        section(name, runners[name])

    rep["pass"] = all(rep[name]["pass"] for name in sections)
    rep["settings"] = {"config": cfg, "genome": args.genome, "quick": args.quick,
                       "service_force_n": W.TOTAL_FORCE_NEWTONS,
                       "target_deflection_mm": WO.TARGET_DEFLECTION_MM,
                       "allowable_stress_mpa": WO.ALLOWABLE_STRESS_MPA,
                       "lr": S3.DEFAULT_LR, "grad_clip": S3.GRAD_CLIP,
                       "sections": sections, "ladder_p": probe_p,
                       "stress_pnorm_p_default": WA.STRESS_PNORM_P,
                       "elapsed_s": round(time.time() - t0, 1)}
    if sections != list(DEFAULT_SECTIONS):
        rep["settings"]["title"] = (
            "M8b-ii — THE PROCESS-PARALLEL PHASE BATCH" if sections == ["phase_pool"] else
            "M8b-i.6 — WHICH p GIVES THE STRESS CONSTRAINT A VALUE" if probe_p else
            "M8b-i.5 — QUALIFYING THE FEASIBILITY VERDICT")

    # Written BEFORE the report is formatted.  At `coarse` this study is an hour of
    # solving and `_print` is string formatting; losing the former to a bug in the latter
    # is a trade nobody would make on purpose.
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(rep, fh, indent=1)
    _print(rep)
    print(f"\nwrote {os.path.join(HERE, args.out)}  "
          f"({rep['settings']['elapsed_s']} s)")
    if not args.no_plot:
        # Which figure, decided by what actually ran: `_plot`'s triptych needs S1, S2 and
        # S9 and cannot be drawn from an M8b-i.5 report.
        if all(k in rep for k in ("direction", "trajectory", "feasibility")):
            plot = _plot
        elif "phase_pool" in rep and not any(
                k in rep for k in ("mesh_convergence", "multistart")):
            plot = _plot_pool
        else:
            plot = _plot_m8bi5
        try:
            print(f"wrote {plot(rep, os.path.splitext(args.out)[0] + '.jpg')}")
        except Exception as exc:                              # pragma: no cover
            print(f"(plot skipped: {exc})")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
