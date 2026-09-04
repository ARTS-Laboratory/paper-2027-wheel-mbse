"""
=============================================================================
  WHAT THE SWITCH DOES TO THE OPTIMUM, NOT TO THE SCORE —
  IS WIRING THE FILLET IN A RE-SCORE OR A RE-OPTIMISATION?
=============================================================================
    .venv-opt/bin/python studies/study_fillet_optimum.py   (make filletoptimum)

PLAN.md §91's ranked successor 2.  FILLET_PLAN.md STEP 3.

WHY THIS EXISTS, AND WHY IT IS THE LAST THING BEFORE THE DECISION
------------------------------------------------------------------
The fillet arc's remaining blocker has been a DECISION about two numbers since §89, and
both numbers now exist:

    the COST         1.12x per evaluation, not §88's unmeasured 2-3x       (§90)
    the SURROGATE    `Kt` exactly flat above its 0.6657 mm cap while the
                     wheel keeps stiffening 8.8%, shipped genome parked
                     at 99.7% of the cap                                    (§75)

§90 and §91 both ranked the surrogate first and both said "via `make reds-hub-fillet`",
as though it were unrun.  IT WAS RUN ON 2026-08-24 AND IT IS §75 — the sweep is in
`studies/study_reds_hub_share.json` under `sweep_filleted_svk`, fourteen rows, SVK,
regenerated at §85, and §91 quotes its numbers three paragraphs above the ranking that
asks for it.  So that item is not work; it is a stale pointer, and this driver measures
the thing that is genuinely unmeasured instead.

§91 FOUND THE 17x AND IT IS `deflection`, WHICH IS WHY THIS QUESTION EXISTS
----------------------------------------------------------------------------
§91 decomposed §90's 671.66-against-38.79 and found it is not the barrier: every one of
the nine `BARRIER_TERMS` is EXACTLY 0.0 on both meshes, and 99.498% of the gap is
`deflection`, with `mass` the remaining 0.502%.  The mechanism is mean axle drop against
a FIXED two-sided target:

    TARGET_DEFLECTION_MM = 2.0
    unfilleted   1.9011 mm    4.945% under target
    filleted     0.9914 mm   50.43%  under target

`deflection` is `2500 * ((drop - 2.0) / 2.0)^2`, so those are not two scores of one
design — they are one design read as NEARLY ON TARGET by one mesh and as HALF THE TARGET
by the other.  A design 50% under a two-sided quadratic target is not sitting at the
filleted objective's minimum.  If that is right, wiring the fillet in does not re-score
the shipped wheel, it MOVES it: the optimum walks toward more compliance, paying `mass`,
which is the only other term with any share of the gap at all.

That is the promotion-shaped consequence, and it is the one thing a decision record has
to have priced.  "The filleted objective scores the shipped genome at 671.66" is a fact
about a number.  "The filleted objective's optimum is not the shipped genome" is a fact
about the PART, and it is what says whether the switch is followed by a Stage 3 re-run
and a re-promotion (`test_promotion.py` carries what that costs) or by nothing at all.

THE MEASUREMENT: TWO DESCENTS FROM THE SAME START, ONE MESH APART
-------------------------------------------------------------------
Both arms are `wheel_stage3.descend` from the shipped genome, identical in every
argument except which mesh the evaluator builds:

    CONTROL     `wheel_stage3.Evaluator`     — the unfilleted mesh, the objective as it
                                                ships, the one that promoted this genome
    TREATMENT   `_FilletedEvaluator`         — the same call with `fillet=True`

Same start, same 30-step cosine schedule, same lr, same clip, same 8-phase uniform
stencil, same SVK kernel, same pinned flank orientation, same box.  One mesh apart.

THE CRITERION IS A PREFERENCE REVERSAL, AND IT NEEDS NO THRESHOLD
------------------------------------------------------------------
The obvious tests are all worse than they look, and two of them were written into this
driver and taken back out before it was run:

  "the filleted arm walks further in gene space".  NO — `wheel_stage3.adam_update` is
  `lr * m_hat / (sqrt(v_hat) + eps)`, which is SCALE FREE in the gradient: an arm whose
  gradient is 5.6x the other's (§90: |grad| 1179.53 against 212.49) takes about the same
  step, not 5.6x the step.  Distance under Adam measures how consistent the gradient
  DIRECTION was, not how far the start was from a minimum.

  "the filleted arm removes a larger fraction of its own start loss".  NO — that is
  decided before the run by arithmetic.  At `coarse` under SVK the unfilleted loss is
  38.79 and 6.11 of it is `deflection`, so at most ~16% of it is reducible through the
  term that moves; the filleted loss is 671.66 with 635.7 in `deflection`, so ~95% is.
  The comparison would be measuring the size of the error term, which §91 already
  published, and calling it a result.

  "the filleted arm closes more of its own distance to the 2.0 mm target".  NO, AND THIS
  ONE IS WRONG-SIGNED — a design that has to travel FURTHER closes a SMALLER fraction of
  its own gap in a fixed budget, so the more thoroughly the optimum has moved, the more
  this test says it has not.  The smoke probe that caught this had the control closing
  87.7% of a 0.262 mm gap while the treatment closed 18.6% of a 1.040 mm one, moving
  +0.294 mm and +0.193 mm respectively — comparable rates, opposite verdicts.

    THE PRE-REGISTERED CRITERION.  Wiring the fillet in is a RE-OPTIMISATION rather than
    a re-score if THE TWO OBJECTIVES DISAGREE ABOUT WHICH DESIGN IS BETTER — that is, if
    the unfilleted objective ranks the shipped genome above the filleted arm's endpoint
    while the filleted objective ranks them the other way round.

That is a preference REVERSAL, and it is the right test for three reasons.  It needs no
threshold and no normalisation, so nothing in it can be tuned after the fact.  It is
invariant to the 17x — a re-score is any monotone re-labelling of the same ranking, and
a monotone re-labelling CANNOT reverse a pair, so a reversal is exactly the thing a
re-score is unable to produce.  And it is the question a promotion actually asks: not
"what does this design score" but "which design do we ship".

WHAT WOULD FALSIFY IT, WHICH IS THE POINT OF SAYING IT FIRST
--------------------------------------------------------------
No reversal.  If 30 steps of the filleted objective land on a design that the filleted
objective STILL ranks below the shipped genome, then within this budget the switch has
not moved the answer and the fillet is a re-score of a wheel that is already the right
one.  That outcome is live: the treatment arm has a long way to travel and could spend
its whole budget getting worse in `mass` faster than it gets better in `deflection`.

THE CONTROL IS STILL NOT A FORMALITY
--------------------------------------
It supplies the third design in the table and it is what says whether a reversal is about
the MESH or about the schedule.  The shipped genome is the unfilleted objective's own
answer (§26 promoted it off exactly this loss), so the control arm re-descending the
SAME objective from it is the null: whatever it finds is what 30 steps at `coarse` under
an 8-phase uniform stencil buys when the objective has not changed at all.  If the
control's endpoint also reverses the pair, the reversal is the schedule's and not the
fillet's, and that is reported rather than left to be inferred.

AND THE PRICE, WHICH IS THE HALF A DECISION RECORD QUOTES
-----------------------------------------------------------
A reversal says the answer moves.  It does not say what moving costs, and the decision
needs both.  So the artifact also carries, for each arm: the distance walked in the unit
box and the per-gene deltas in millimetres, the `mass` term and the mesh mass in grams,
the mean axle drop against its 2.0 mm target at both ends, the minimum scaled Jacobian,
and whether any of the nine `BARRIER_TERMS` fired at any point on the trajectory.  Those
are the re-optimisation's bill.

THE 3x2 TABLE IS THE INSTRUMENT THE CRITERION IS READ OFF
-----------------------------------------------------------
Three designs — the shipped genome, the control's endpoint, the treatment's endpoint —
each read by BOTH objectives.  Four of the six cells are free (the arms' own first and
last steps); the two off-diagonal ones are separate evaluations at the same stencil,
kernel and pinned orientation, because a cross-evaluation taken at a different stencil
would be comparing two integrands.  Each COLUMN is one objective ranking three designs,
and the criterion above is whether the two columns rank them differently.

WHAT THIS DRIVER DOES NOT DO
-----------------------------
It does not wire the fillet into the objective.  Like §90's and §91's drivers it touches
NO `src/` module — the treatment evaluator is a `studies/` subclass that overrides one
call — so `test_nothing_wires_the_fillet_into_the_objective` is untouched and the scope
gate stands.  It does not promote anything, it does not write `best_solution.json`, and
neither arm's endpoint is a candidate for anything: 30 steps at `coarse` is an
INSTRUMENT for the question "did the start move", not a search.

IT DOES NOT FIND THE FILLETED OPTIMUM EITHER, AND THE NUMBER IS A LOWER BOUND
------------------------------------------------------------------------------
30 steps is half of `wheel_stage3.DEFAULT_STEPS` and a small fraction of a promotion run.
Whatever the treatment arm recovers is therefore a LOWER BOUND on what re-optimising
would recover, and it is labelled that way in the verdict rather than quoted as the
answer — the same reading `extrapolation-against-measurement` asks for whenever a ladder
is still climbing at its top rung.  A lower bound is sufficient for the decision this
feeds: the decision needs to know whether the optimum moves, not where it lands.

THIRTY STEPS AND NOT SIXTY, PRICED
-----------------------------------
§90 measured one filleted 8-phase SVK evaluation at 183.6 s against the unfilleted
163.3 s, plus a one-time jit trace per mesh topology (271.7 s and 1122.0 s).  Thirty
steps of both arms is therefore about three and a half hours, and sixty would be seven.
The question is answered by whether the start moves, which the first steps decide, so the
extra three and a half hours would buy a sharper lower bound on a quantity already
labelled as one.

UNIFORM PHASES, NOT `rqmc`, AND THAT IS A DEPARTURE FROM PRODUCTION
--------------------------------------------------------------------
`descend` defaults to `rqmc`, which is what a promotion run uses and is genuinely
stochastic.  Both arms here take `uniform` instead, matching §91: the two trajectories
have to differ by the MESH and not by two draws of a random offset, and a paired
comparison between arms is worth more here than the unbiasedness that `rqmc` buys a
search.  It is a departure and it is named, not silent.
"""

import argparse
import json
import os
import time

import numpy as np

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import _gate_guard

import jax_config  # noqa: F401
import wheel_fea as W
import wheel_genome as wg
import wheel_objective as WO
import wheel_stage3 as WS
import wheel_wheel as WW

HERE = os.path.dirname(os.path.abspath(__file__))

# `coarse`, 8 uniform phases and SVK are §91's configuration, and this driver's numbers
# have to sit next to §91's table or the two are measuring different pairs.  30 steps is
# priced in the module docstring.
DEFAULT_CONFIG = "coarse"
DEFAULT_PHASES = 8
DEFAULT_STEPS = 30
DEFAULT_KINEMATICS = "svk"
DEFAULT_SCHEME = "uniform"

# The five report fields the verdict reads.  `axle_drop_mean_mm` is the mechanism §91
# named, `mesh_mass_g` is the term it trades against, and the other three are the ones
# that would say a moving optimum walked into a stress or validity problem on the way.
TRACK_KEYS = ("axle_drop_mean_mm", "mesh_mass_g", "min_scaled_jacobian",
              "stress_utilisation", "max_stress_mpa")


class _FilletedEvaluator(WS.Evaluator):
    """`wheel_stage3.Evaluator` with the mesh build's `fillet=` PINNED, and nothing else.

    This was `Evaluator.__call__` copied with ONE keyword added, rather than a hook added
    to `wheel_stage3`, because the scope gate
    (`test_nothing_wires_the_fillet_into_the_objective`) parsed `src/` for a `fillet=`
    keyword reaching any call and had to keep passing while this ran.  **THAT GATE WAS
    INVERTED AT §103**, which wired the fillet into the objective; the subclass stays
    anyway, and now carries the CONTROL as well -- see `_UnfilletedEvaluator` and the
    paragraph below.

    `_FILLET` EXISTS BECAUSE THE CONTROL STOPPED BEING A CONTROL.  Until §103 the control
    arm was the plain `wheel_stage3.Evaluator`, whose mesh build took `build_wheel`'s
    `fillet=None` default.  §103 made `wheel_objective.phase_meshes` pass `fillet=True`,
    so the plain evaluator began building the SAME mesh as this subclass and the A/B
    silently compared a mesh with itself -- the driver still ran, still wrote an artifact,
    and the artifact was two identical arms labelled `control` and `treatment`.  Both arms
    now state their `fillet=` explicitly, so neither depends on a default that can move
    under them again (PLAN.md §105).

    The pooled branch of the parent is dropped rather than reproduced: `descend` never
    gives an INJECTED evaluator a pool, so `self.pool` is always None here and a pooled
    path would be untestable dead code.  `mesh_s`/`solve_s`/`n_calls` are kept because
    `_record` reads all three off the evaluator it was handed.
    """

    _FILLET = True

    def __call__(self, z, low, high, *, phases, warm=None, tiers=("t1", "t2", "t3")):
        genes = self.genes(z, low, high)

        t0 = time.time()
        meshes = None
        if "t2" in tiers or "t3" in tiers:
            meshes = [WW.build_wheel(genes, self.cfg, phase_deg=float(p),
                                     orientation=self.orientation, fillet=self._FILLET)
                      for p in phases]
        self.mesh_s += time.time() - t0

        t0 = time.time()
        val, grad, brk = WO.objective(
            z, self.cfg, normalized=True, weights=self.weights, phases=phases,
            meshes=meshes, tiers=tiers, span_mm=self.span_mm,
            warm=warm, pool=None, orientation=self.orientation, **self.problem_kw)
        self.solve_s += time.time() - t0
        self.n_calls += 1
        return val, grad, brk


class _UnfilletedEvaluator(_FilletedEvaluator):
    """The CONTROL arm, with `fillet=None` stated rather than inherited from a default.

    Subclassing the treatment rather than `WS.Evaluator` is deliberate: the two arms must
    differ in the `fillet=` keyword and in NOTHING else, and sharing one `__call__` is the
    only way to keep that true as `wheel_stage3.Evaluator` changes.  See the sibling's
    docstring for why this class had to exist at all.
    """

    _FILLET = None


def load_genes(path):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


# ---------------------------------------------------------------------------
# THE TWO ARMS
# ---------------------------------------------------------------------------

def arm(z0, cfg, *, filleted, steps, n_phase, kinematics, scheme, seed):
    """One descent from `z0`.  Returns `wheel_stage3.descend`'s own record.

    `orientation` is left to `descend`, which derives it from `z0` and PINS it for the
    run.  Both arms therefore get the SAME orientation from the same start — it is a
    discrete decision about which block owns which node and `flank_orientation` reads the
    genes and the config, never a mesh, so the fillet cannot move it.  Deriving it once
    per arm and letting the two disagree would be a second difference between the arms.
    """
    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    wcfg = WW.get_config(cfg)
    orientation = tuple(float(o) for o in WW.flank_orientation(
        wg.denormalize(z0, low, high), wcfg, span_mm=W.S))
    cls = _FilletedEvaluator if filleted else _UnfilletedEvaluator
    ev = cls(cfg, orientation=orientation, kinematics=kinematics)
    return WS.descend(z0, cfg, steps=steps, n_phase=n_phase, scheme=scheme, seed=seed,
                      evaluator=ev, orientation=orientation, out=None, verbose=True,
                      log_every=1)


def cross_evaluate(z, cfg, *, filleted, n_phase, kinematics, orientation):
    """Score ONE iterate on ONE mesh, outside any descent.

    The two arms already report their endpoints on their OWN mesh — that is the last row
    of each trajectory — so this is called only for the two OFF-DIAGONAL cells: the
    control's endpoint read by the filleted objective, and the treatment's endpoint read
    by the unfilleted one.  Together with the two diagonals and §90's pair at the shipped
    genome that is a 3x2 table, and it is the table a decision record needs: it prices the
    disagreement in BOTH directions, so "the filleted objective's answer is bad by the
    unfilleted objective's reckoning" is a number rather than a worry.

    Same stencil, same kernel, same pinned orientation as the descent that produced `z`.
    A cross-evaluation taken at a different stencil would be comparing two integrands.
    """
    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    genes = wg.denormalize(np.asarray(z, dtype=float), low, high)
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    if filleted:
        meshes = [WW.build_wheel(genes, cfg, phase_deg=float(p),
                                 orientation=orientation, fillet=True) for p in phases]
    else:
        meshes = [WW.build_wheel(genes, cfg, phase_deg=float(p),
                                 orientation=orientation) for p in phases]
    val, grad, brk = WO.objective(z, cfg, normalized=True, phases=phases, meshes=meshes,
                                  orientation=orientation, kinematics=kinematics)
    return {"filleted": filleted, "loss": float(val),
            "grad_norm": float(np.linalg.norm(grad)),
            "terms": {k: d["value"] for k, d in brk["terms"].items()},
            "report": {k: float(brk["report"][k]) for k in TRACK_KEYS
                       if k in brk.get("report", {})}}


def _trajectory(rec):
    """The per-step quantities the verdict reads, off a descend record.

    Abandoned steps are kept in the trajectory rather than filtered out: a step every
    trial refused is a fact about the arm, and dropping it would make a run that stalled
    look like a run that walked.
    """
    rows = rec["steps"]
    out = {"n_steps": len(rows) - 1,
           "loss": [r["loss"] for r in rows],
           "grad_norm": [r["grad_norm"] for r in rows],
           "abandoned": [bool(r["abandoned"]) for r in rows],
           "z": [r["z"] for r in rows]}
    for k in TRACK_KEYS:
        out[k] = [r["report"].get(k) for r in rows]
    for k in ("deflection", "mass"):
        out[k] = [r["terms"][k]["value"] if k in r["terms"] else None for r in rows]
    # Every barrier, summed, per step.  A moving optimum that walks into a barrier is a
    # different consequence from one that does not, and the sum is the cheapest statement
    # of "did any of the nine fire at any point on this trajectory".
    out["barrier_sum"] = [
        float(sum(r["terms"][k]["value"] for k in WO.BARRIER_TERMS if k in r["terms"]))
        for r in rows]
    return out


def _displacement(traj, low, high):
    """How far the arm walked, in the unit box and in physical genes.

    Reported in BOTH because neither alone is readable: the box distance is what the
    optimizer sees and what the two arms are comparable in, and the per-gene physical
    deltas are what a reader can check against a drawing.
    """
    z0 = np.asarray(traj["z"][0], dtype=float)
    z1 = np.asarray(traj["z"][-1], dtype=float)
    dz = z1 - z0
    return {"l2_box": float(np.linalg.norm(dz)),
            "linf_box": float(np.max(np.abs(dz))),
            "per_gene_box": {n: float(d) for n, d in zip(wg.GENE_NAMES, dz)},
            "per_gene_mm": {n: float(d) for n, d in
                            zip(wg.GENE_NAMES, wg.denormalize(z1, low, high)
                                - wg.denormalize(z0, low, high))},
            "start_genes": {n: float(v) for n, v in
                            zip(wg.GENE_NAMES, wg.denormalize(z0, low, high))},
            "end_genes": {n: float(v) for n, v in
                          zip(wg.GENE_NAMES, wg.denormalize(z1, low, high))}}


def verdict(control, treatment, cross):
    """The pre-registered comparison, adjudicated off the 3x2 table.

    The criterion is a PREFERENCE REVERSAL and is stated in the module docstring before
    the run: the two objectives disagree about which design is better.  Nothing in it is
    a threshold, so nothing in it can be tuned after the fact, and a re-score — any
    monotone re-labelling of one ranking — cannot produce one.

    Everything else here is the PRICE, reported and not read by the criterion.
    """
    TARGET = float(WO.TARGET_DEFLECTION_MM)
    c, t = control["trajectory"], treatment["trajectory"]

    # THE TABLE.  Each column is one objective ranking the same three designs.  The
    # diagonals are the arms' own steps; the off-diagonals are `cross_evaluate`.
    table = {
        "shipped": {"unfilleted": c["loss"][0], "filleted": t["loss"][0]},
        "control_endpoint": {"unfilleted": c["loss"][-1],
                             "filleted": cross["control_endpoint_on_filleted"]["loss"]},
        "treatment_endpoint": {
            "unfilleted": cross["treatment_endpoint_on_unfilleted"]["loss"],
            "filleted": t["loss"][-1]}}

    def argmin(col):
        return min(table, key=lambda d: table[d][col])

    # THE DECISIVE PAIR.  `shipped` against the filleted arm's endpoint: the design the
    # tree ships against the design descending the filleted objective produced.  A
    # reversal on THIS pair is what says the switch changes the answer.
    pair = ("shipped", "treatment_endpoint")
    prefers = {col: min(pair, key=lambda d: table[d][col])
               for col in ("unfilleted", "filleted")}
    reversal = prefers["unfilleted"] != prefers["filleted"]

    # THE CONTROL'S OWN PAIR, which says whether a reversal is the MESH's or the
    # SCHEDULE's: the same test with the control's endpoint in place of the treatment's.
    c_pair = ("shipped", "control_endpoint")
    c_prefers = {col: min(c_pair, key=lambda d: table[d][col])
                 for col in ("unfilleted", "filleted")}
    control_reversal = c_prefers["unfilleted"] != c_prefers["filleted"]

    def frac_removed(x):
        first, last = x["loss"][0], x["loss"][-1]
        return (first - last) / first if first else 0.0

    def drop_row(x):
        d = [v for v in x["axle_drop_mean_mm"] if v is not None]
        return {"start_mm": d[0], "end_mm": d[-1], "target_mm": TARGET,
                "start_pct_under": 100.0 * (TARGET - d[0]) / TARGET,
                "end_pct_under": 100.0 * (TARGET - d[-1]) / TARGET,
                "moved_toward_target": bool(abs(d[-1] - TARGET) < abs(d[0] - TARGET))}

    def term_row(x, k):
        v = [q for q in x[k] if q is not None]
        return {"start": v[0] if v else None, "end": v[-1] if v else None,
                "delta": (v[-1] - v[0]) if v else None}

    return {
        "criterion": ("wiring the fillet in is a RE-OPTIMISATION rather than a re-score "
                      "if the two objectives DISAGREE about which design is better — the "
                      "unfilleted objective ranking the shipped genome above the filleted "
                      "arm's endpoint while the filleted objective ranks them the other "
                      "way round.  A re-score is a monotone re-labelling of one ranking "
                      "and cannot reverse a pair."),
        "table": table,
        "argmin_unfilleted": argmin("unfilleted"),
        "argmin_filleted": argmin("filleted"),
        "columns_rank_differently": bool(argmin("unfilleted") != argmin("filleted")),
        "decisive_pair": {"pair": list(pair),
                          "unfilleted_prefers": prefers["unfilleted"],
                          "filleted_prefers": prefers["filleted"],
                          "reversal": bool(reversal)},
        "re_optimisation_not_re_score": bool(reversal),
        # Whether the SCHEDULE alone reverses the same pair.  If this is true the
        # treatment arm's reversal is not evidence about the mesh.
        "control_pair_also_reverses": bool(control_reversal),
        # ---- THE PRICE.  Reported; the criterion does not read any of it. ----
        "price": {
            "fraction_of_start_loss_removed": {"control": frac_removed(c),
                                               "treatment": frac_removed(t)},
            "displacement_l2_box": {
                "control": control["displacement"]["l2_box"],
                "treatment": treatment["displacement"]["l2_box"]},
            "displacement_note": (
                "adam_update is lr * m_hat / (sqrt(v_hat) + eps), scale free in the "
                "gradient — this measures gradient-direction consistency, not distance "
                "from a minimum; see the module docstring"),
            "axle_drop": {"control": drop_row(c), "treatment": drop_row(t)},
            "deflection_term": {"control": term_row(c, "deflection"),
                                "treatment": term_row(t, "deflection")},
            "mass_term": {"control": term_row(c, "mass"),
                          "treatment": term_row(t, "mass")},
            "mesh_mass_g": {"control": term_row(c, "mesh_mass_g"),
                            "treatment": term_row(t, "mesh_mass_g")},
            "min_scaled_jacobian": {"control": term_row(c, "min_scaled_jacobian"),
                                    "treatment": term_row(t, "min_scaled_jacobian")},
            # `soft_barrier` is exactly zero below its knee (§91), so `> 0.0` here is "a
            # barrier fired", not "a barrier is small".
            "barrier_fired_anywhere": {"control": bool(max(c["barrier_sum"]) > 0.0),
                                       "treatment": bool(max(t["barrier_sum"]) > 0.0)},
            "barrier_max": {"control": max(c["barrier_sum"]),
                            "treatment": max(t["barrier_sum"])},
            "n_abandoned": {"control": int(sum(c["abandoned"])),
                            "treatment": int(sum(t["abandoned"]))}},
        # THE LOWER BOUND, LABELLED.  30 steps is not a promotion run; see the docstring.
        # It bounds the PRICE from below.  It does NOT weaken the criterion: a reversal
        # found inside a short budget is still a reversal.
        "is_a_lower_bound": True}


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def _print(rec):
    def head(s):
        print(f"\n{'=' * 78}\n  {s}\n{'=' * 78}")

    v = rec["verdict"]
    print(f"\n{'=' * 78}")
    print("  WHAT THE SWITCH DOES TO THE OPTIMUM — PLAN §91 SUCCESSOR 2")
    print(f"  genome {rec['genome']}, {rec['config']}, {rec['steps']} steps, "
          f"{rec['n_phase']} {rec['scheme']} phases, {rec['kinematics']}")
    print(f"{'=' * 78}")

    for name in ("control", "treatment"):
        a = rec[name]
        t = a["trajectory"]
        head(f"{name.upper()}  —  {'filleted' if a['filleted'] else 'unfilleted'} mesh")
        print(f"    {'step':>5s}{'loss':>13s}{'|grad|':>12s}{'drop mm':>10s}"
              f"{'mass g':>10s}{'deflection':>12s}{'mass term':>11s}{'barrier':>10s}")
        n = t["n_steps"]
        for i in [0] + [k for k in range(1, n + 1) if k % 5 == 0 or k == n]:
            def f(key, w=10, p=4):
                x = t[key][i]
                return f"{x:{w}.{p}f}" if x is not None else " " * w
            print(f"    {i:5d}{t['loss'][i]:13.4f}{t['grad_norm'][i]:12.4f}"
                  f"{f('axle_drop_mean_mm')}{f('mesh_mass_g')}"
                  f"{f('deflection', 12)}{f('mass', 11)}{f('barrier_sum')}")
        d = a["displacement"]
        print(f"    walked  ||dz||_2 {d['l2_box']:.6f}   ||dz||_inf "
              f"{d['linf_box']:.6f}  (unit box)")
        moved = sorted(d["per_gene_box"].items(), key=lambda kv: -abs(kv[1]))[:4]
        print("    largest gene moves  " + "   ".join(
            f"{k} {d['per_gene_mm'][k]:+.4f} mm" for k, _ in moved))

    head("THE VERDICT — RE-SCORE OR RE-OPTIMISATION")
    print(f"    criterion: {v['criterion']}")
    print()
    print("    THE SAME THREE DESIGNS, READ BY BOTH OBJECTIVES")
    print(f"    {'design':<24s}{'unfilleted obj':>16s}{'filleted obj':>16s}")
    for d in ("shipped", "control_endpoint", "treatment_endpoint"):
        row = v["table"][d]
        mark = ("  <-- unfilleted argmin" if d == v["argmin_unfilleted"] else "")
        mark += ("  <-- filleted argmin" if d == v["argmin_filleted"] else "")
        print(f"    {d:<24s}{row['unfilleted']:16.4f}{row['filleted']:16.4f}{mark}")
    print("    each COLUMN is one objective ranking three designs; the diagonal cells "
          "are the arms'")
    print("    own first and last steps, the two off-diagonal ones separate evaluations "
          "at the same")
    print("    stencil, kernel and pinned orientation")
    print()
    dp = v["decisive_pair"]
    print(f"    the decisive pair            {dp['pair'][0]} vs {dp['pair'][1]}")
    print(f"    the unfilleted objective prefers   {dp['unfilleted_prefers']}")
    print(f"    the filleted objective prefers     {dp['filleted_prefers']}")
    print()
    print(f"    RE-OPTIMISATION AND NOT A RE-SCORE: "
          f"{v['re_optimisation_not_re_score']}")
    if v["control_pair_also_reverses"]:
        print("    BUT THE CONTROL'S PAIR REVERSES TOO — the same test with the "
              "control's endpoint in")
        print("    place of the treatment's also flips, so this reversal is the "
              "SCHEDULE's and not the")
        print("    mesh's, and it is not evidence about the fillet.")
    else:
        print("    the control's pair does NOT reverse, so the reversal is the mesh's "
              "and not the")
        print("    schedule's")

    p_ = v["price"]
    head("THE PRICE — REPORTED, AND THE CRITERION READS NONE OF IT")
    print(f"    {'':<38s}{'control':>13s}{'treatment':>13s}")

    def two(label, cv, tv, fmt="13.4f", tail=""):
        # `bool` before `(int, float)`: a bool IS an int in python, and "a barrier fired
        # anywhere" printed as 0.0000 before this line existed.
        def cell(x):
            return f"{str(x):>13s}" if isinstance(x, bool) or not isinstance(
                x, (int, float)) else f"{x:{fmt}}"
        print(f"    {label:<38s}{cell(cv)}{cell(tv)}{tail}")

    two("fraction of start loss removed",
        p_["fraction_of_start_loss_removed"]["control"],
        p_["fraction_of_start_loss_removed"]["treatment"], "13.6f")
    ac, at = p_["axle_drop"]["control"], p_["axle_drop"]["treatment"]
    two("axle drop, start (mm)", ac["start_mm"], at["start_mm"])
    two("axle drop, end (mm)", ac["end_mm"], at["end_mm"],
        tail=f"   target {ac['target_mm']:.1f}")
    two("percent under target, start", ac["start_pct_under"], at["start_pct_under"])
    two("percent under target, end", ac["end_pct_under"], at["end_pct_under"])
    for label, blk in (("`deflection` term", "deflection_term"),
                       ("`mass` term", "mass_term"),
                       ("mesh mass (g)", "mesh_mass_g"),
                       ("min scaled Jacobian", "min_scaled_jacobian")):
        b = p_[blk]
        two(label + ", start", b["control"]["start"], b["treatment"]["start"])
        two(label + ", end", b["control"]["end"], b["treatment"]["end"],
            tail=f"   delta {b['control']['delta']:+.4f} / "
                 f"{b['treatment']['delta']:+.4f}")
    two("a barrier fired anywhere", p_["barrier_fired_anywhere"]["control"],
        p_["barrier_fired_anywhere"]["treatment"])
    two("steps abandoned", p_["n_abandoned"]["control"],
        p_["n_abandoned"]["treatment"], "13d")
    two("||dz||_2 (NOT a criterion — Adam)", p_["displacement_l2_box"]["control"],
        p_["displacement_l2_box"]["treatment"], "13.6f")
    print()
    print(f"    THE PRICE IS A LOWER BOUND: {rec['steps']} steps is half of "
          f"wheel_stage3.DEFAULT_STEPS.  The")
    print("    CRITERION is not — a reversal found inside a short budget is still a "
          "reversal.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genome", default="best_solution.json")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--phases", type=int, default=DEFAULT_PHASES)
    ap.add_argument("--kinematics", default=DEFAULT_KINEMATICS,
                    choices=("linear", "svk"))
    ap.add_argument("--scheme", default=DEFAULT_SCHEME)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="study_fillet_optimum.json")
    args = ap.parse_args()

    # What degrades THIS driver is answering the question at a configuration §91's numbers
    # were not taken at, or with a budget too short for "did the start move" to mean
    # anything.  `linear` is refused outright: §32 says a loss is not comparable across
    # kinematics and Stage 3 descends SVK, so a linear pair adjudicates a different
    # optimizer than the one a promotion would re-run.
    _gate_guard.refuse_degraded_out(ap, args, "study_fillet_optimum.json", [
        (args.config != DEFAULT_CONFIG,
         "--config %s, not the %s §91's decomposition was taken at"
         % (args.config, DEFAULT_CONFIG)),
        (args.kinematics != DEFAULT_KINEMATICS,
         "--kinematics %s — Stage 3 descends svk (§32) and a linear pair adjudicates a "
         "different optimizer" % args.kinematics),
        (args.phases < DEFAULT_PHASES,
         "--phases %d, below the optimizer's %d-point stencil"
         % (args.phases, DEFAULT_PHASES)),
        (args.steps < DEFAULT_STEPS,
         "--steps %d, below the %d this driver's lower bound is quoted at"
         % (args.steps, DEFAULT_STEPS)),
        (args.genome != "best_solution.json", "--genome %s" % args.genome),
    ])

    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    z0 = wg.normalize(load_genes(args.genome), low, high)

    t0 = time.time()
    rec = {"genome": args.genome, "config": args.config, "steps": args.steps,
           "n_phase": args.phases, "scheme": args.scheme,
           "kinematics": args.kinematics, "seed": args.seed,
           "target_deflection_mm": float(WO.TARGET_DEFLECTION_MM),
           "weights": {k: float(v) for k, v in WO.DEFAULT_WEIGHTS.items()},
           "barrier_terms": list(WO.BARRIER_TERMS)}

    # The CONTROL first, so the arm that pays §90's 1122 s filleted trace runs second.
    # Nothing here is timed against anything — the seconds are recorded per arm and never
    # compared — but the trace belongs to the mesh topology rather than to the kernel, and
    # which arm pays it is a real difference in the wall clock, so it is chosen.
    for name, filleted in (("control", False), ("treatment", True)):
        print(f"\n{'#' * 78}\n#  {name.upper()}  "
              f"({'filleted' if filleted else 'unfilleted'} mesh)\n{'#' * 78}")
        t_arm = time.time()
        run = arm(z0, args.config, filleted=filleted, steps=args.steps,
                  n_phase=args.phases, kinematics=args.kinematics,
                  scheme=args.scheme, seed=args.seed)
        traj = _trajectory(run)
        rec[name] = {"filleted": filleted, "trajectory": traj,
                     "displacement": _displacement(traj, low, high),
                     "settings": run["settings"], "events": run["events"],
                     # `final` and `best` carry `genome_hash`, which is how this tree
                     # names a design (§26 promoted `09e8188`).  An arm that walked away
                     # from the shipped genome has a NAME for where it walked to, and a
                     # successor that wants to re-evaluate an endpoint should not have to
                     # re-derive it from 14 floats.
                     "final": run["final"], "best": run["best"],
                     "wall_s": time.time() - t_arm}

    # THE TWO OFF-DIAGONAL CELLS.  Each arm's endpoint read by the OTHER objective, so
    # the disagreement is priced in both directions.  The orientation is the one both
    # descents were pinned to — `descend` derived it from `z0` and neither arm moved it.
    orientation = tuple(rec["control"]["settings"]["orientation"])
    rec["cross"] = {
        "control_endpoint_on_filleted": cross_evaluate(
            rec["control"]["trajectory"]["z"][-1], args.config, filleted=True,
            n_phase=args.phases, kinematics=args.kinematics, orientation=orientation),
        "treatment_endpoint_on_unfilleted": cross_evaluate(
            rec["treatment"]["trajectory"]["z"][-1], args.config, filleted=False,
            n_phase=args.phases, kinematics=args.kinematics, orientation=orientation)}

    rec["verdict"] = verdict(rec["control"], rec["treatment"], rec["cross"])
    rec["wall_s"] = time.time() - t0
    _print(rec)

    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"\n    wrote {args.out}  ({rec['wall_s']:.1f} s)")

    # NO PASS/FAIL, EXIT 0, for the same reason §90's and §91's drivers have none: this
    # answers "does the optimum move" and either answer is the answer.  A driver that
    # exited nonzero on `re_optimisation_not_re_score` would be asserting which one the
    # decision wants to hear.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
