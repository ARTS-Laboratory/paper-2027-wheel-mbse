"""
=============================================================================
  WHAT THE FILLETED OBJECTIVE COSTS — ONE EVALUATION, FORWARD AND ADJOINT,
  AGAINST §88's UNMEASURED "2-3x"
=============================================================================
    .venv-opt/bin/python studies/study_fillet_cost.py           (make filletcost)

PLAN.md §89's ranked successor 1, first half.  FILLET_PLAN.md STEP 3.

WHY THIS EXISTS
---------------
§89 retired §48's last mesh-validity objection to a filleted objective: the barrier puts
`fillet=True` and `fillet=None` on the SAME side of `MIN_SJ_TARGET`, on the same genomes,
with the one failing genome further under it UNFILLETED.  What is left is not a blocker
but a DECISION with two terms, and §89 found one of the two has never been measured at
all:

    §88 ranking item 2:  "the filleted mesh is 2-3x the cost of the unfilleted one"

Nothing in this tree measures that.  §89 said so in as many words — *"the cost half of the
remaining blocker is currently a number with no measurement behind it"* — and noted the
one adjacent number that does exist, the element counts at `coarse` (5952 against 4704,
1.27x), which is not the same quantity as solve time and neither confirms nor refutes it.

This driver is that measurement.  It is not a validity study and takes no view on whether
the fillet SHOULD be wired in; it produces the number the decision needs.

WHAT IS TIMED, AND AT WHICH ALTITUDE
-------------------------------------
Three altitudes, because "cost" is three different quantities and quoting one as the
others is how "2-3x" got written down unmeasured in the first place:

  A. MESH BUILD.  `build_wheel(fillet=)` alone.  Cheap at both settings and reported
     because it is the only part of the ratio anybody has an intuition for.

  B. ONE SOLVE AND ONE ADJOINT.  `wheel_adjoint.solve_and_grad`, cold and warm, on each
     mesh at ITS OWN service indentation.  This is `study_gradient.py`'s G10 method —
     same call, same `timings` block, same two denominators — extended to the filleted
     mesh, which is the thing G10 never ran on.  Each mesh is solved at its own
     equilibrium and not at a shared one: the filleted wheel is stiffer (PART 12's 37.97%),
     so a shared indentation would time one of the two meshes at a state it never
     occupies.

  C. ONE OBJECTIVE EVALUATION — the quantity the decision is actually about.  Eight
     phases, `wheel_objective.objective`, which is eight force-controlled secants and
     eight adjoints plus T1 and T2.  The tier split is taken by DIFFERENCE: a
     `tiers=("t1","t2")` call is milliseconds-to-seconds and a second full T3 would double
     the run for a number subtraction already gives.

BOTH KINEMATICS, AND THE REASON IS THIS ARC'S OWN CAVEAT
---------------------------------------------------------
`wheel_objective.objective` called bare takes `wheel_fem`'s LINEAR kernel default; Stage 3
passes `kinematics="svk"` (§32, and `wheel_stage3.py`'s `--kinematics` default).  So the
number that carries the decision is the SVK row — that is what an optimizer step pays —
and the linear row is reported beside it because it is what anyone re-running `objective`
by hand will get.  FILLET_PLAN.md's header names this exact trap: *"the study drivers do
NOT get svk by default ... A ladder built on those takes linear silently."*

EVERYTHING IS POST-TRACE
-------------------------
`wheel_wheel.coord_fn` is jitted and its cache is keyed on the mesh topology, so the first
evaluation at each setting pays a trace that Stage 3 pays once and amortises over a whole
run.  Every timed call here is preceded by an IDENTICAL warm-up call at the same setting —
identical, and not a cheaper single-phase stand-in, because nothing here has established
that one phase's trace serves the other seven.

The warm-up is not discarded either.  `first_call_s` is its wall time and
`trace_overhead_s` is `first_call_s - eval_s`, which is the trace ALONE: the two calls run
the same arithmetic on the same meshes, so what the first one pays extra is what the jit
cost.  A trace that costs minutes is a real price of switching the default, just not a
PER-EVALUATION one, and it is kept out of the ratio the verdict reads for exactly that
reason.  On the probe that sized this driver it was 117.7 s against a 19.8 s evaluation,
one phase, unfilleted, linear — six times the thing it precedes, and 244.2 s filleted.

AND THE TRACE BELONGS TO THE MESH, NOT TO THE KERNEL, which is why only ONE of the two
kinematics rows can report it.  The same probe, having traced both meshes under `linear`,
paid 26.4 s and 26.7 s for its two SVK first calls against 24.7 s and 25.1 s for the
evaluations behind them — nothing left to trace.  The jit cache is process-wide, so this
driver runs the evaluation section FIRST and the linear row before the svk one, and the
svk row's near-zero `trace_overhead_s` is that fact rather than a glitch.  Section B runs
behind both and its `first_call_s` is NOT a cold trace; it is recorded and not ratioed.

WHAT THIS DOES NOT MEASURE
---------------------------
The other half of §89's item 1 — `R_hub` read through the filleted mesh against the flat
`Kt` surrogate (§75) — is a different experiment on a different apparatus
(`make reds-hub-fillet`) and is not here.  This file answers "what does it cost", not
"what does it buy".

Nothing here writes to `src/`.  The filleted meshes are built in this driver and handed to
`objective(meshes=...)`, which is the parameter `wheel_stage3.Evaluator` already uses, so
`test_nothing_wires_the_fillet_into_the_objective` is untouched and the scope gate stands.
"""

import argparse
import json
import os
import time

import numpy as np

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import _gate_guard

import jax_config  # noqa: F401
import wheel_adjoint as WA
import wheel_fea as W
import wheel_fem as fem
import wheel_genome as wg
import wheel_objective as WO
import wheel_wheel as WW

HERE = os.path.dirname(os.path.abspath(__file__))

SERVICE_FORCE_N = W.TOTAL_FORCE_NEWTONS

# `coarse` is the config `wheel_objective.objective` defaults to and the one §89's barrier
# control was taken at, for the same reason: it is where the objective runs.
DEFAULT_CONFIG = "coarse"

# Both, and the SVK row is the one that carries the decision — see the module docstring.
DEFAULT_KINEMATICS = ("linear", "svk")

# The optimizer's own stencil.  `phase_stencil(scheme="uniform")` with its default
# `n_phase=8`, taken deterministically rather than `rqmc` so two rows of this table differ
# by the fillet and not by a random offset.
DEFAULT_PHASES = 8

# Medians over this many timed calls at altitudes A and B.  Altitude C is ONE call per
# setting: eight secants and eight adjoints is minutes, and the spread that a median
# defends against is measured at B on the same solves.
DEFAULT_REPEATS = 3

# §88 ranking item 2's claim, as a bracket, so the verdict compares against a number
# instead of a sentiment.
PART88_CLAIM = (2.0, 3.0)


def load_genes(path):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


def _med(rows, key):
    return float(np.median([r[key] for r in rows]))


# ---------------------------------------------------------------------------
# A — THE MESH BUILD
# ---------------------------------------------------------------------------

def build_cost(genes, cfg, repeats):
    """`build_wheel` alone, both settings, with the counts §89 already quoted.

    The first build at each setting is timed and reported as `first_build_s` rather than
    thrown away, and the median of the repeats behind it is what the ratio reads.  No
    claim is made here about WHY they differ: the numpy build path is not obviously
    jitted, the sizing probe's very first unfilleted build was already 0.021 s, and
    `first_build_s` is recorded so the question is answerable from the artifact instead
    of from a docstring's guess.
    """
    out = {}
    for fil in (None, True):
        t0 = time.perf_counter()
        WW.build_wheel(genes, cfg, fillet=fil)
        first_s = time.perf_counter() - t0
        rows = []
        for _ in range(int(repeats)):
            t0 = time.perf_counter()
            mesh = WW.build_wheel(genes, cfg, fillet=fil)
            rows.append({"build_s": time.perf_counter() - t0})
        out["filleted" if fil else "unfilleted"] = {
            "build_s": _med(rows, "build_s"),
            "first_build_s": first_s,
            "n_nodes": int(mesh.coords.shape[0]),
            "n_elements": int(mesh.conn.shape[0]),
            "calls": [r["build_s"] for r in rows]}
    u, f = out["unfilleted"], out["filleted"]
    out["ratio_build_s"] = f["build_s"] / max(u["build_s"], 1e-12)
    out["ratio_elements"] = f["n_elements"] / u["n_elements"]
    out["ratio_nodes"] = f["n_nodes"] / u["n_nodes"]
    return out


# ---------------------------------------------------------------------------
# B — ONE SOLVE AND ONE ADJOINT
# ---------------------------------------------------------------------------

def solve_cost(genes, cfg, kinematics, repeats):
    """G10's two denominators on both meshes: forward and gradient, cold and warm.

    The warm row is the one that sizes an optimizer step and it is much the larger ratio —
    G10's own finding, and its reason is unchanged by the fillet: a warm contact solve is
    two Newton iterations while the gradient still needs a full tangent assembly and
    factorisation however good the starting guess was.
    """
    out = {}
    for fil in (None, True):
        mesh = WW.build_wheel(genes, cfg, fillet=fil)
        t0 = time.perf_counter()
        ind = float(fem.solve_wheel_contact(mesh, force=SERVICE_FORCE_N,
                                            kinematics=kinematics)["axle_drop_mm"])
        secant_s = time.perf_counter() - t0

        # NOT a cold trace — the evaluation section ran first and warmed the cache.
        # Timed and recorded anyway, because a first call that is still slow here would
        # mean something was left untraced and the numbers under it are suspect.
        t0 = time.perf_counter()
        warm0 = WA.solve_and_grad(genes, cfg, "contact_force", indentation_mm=ind,
                                  mesh=mesh, kinematics=kinematics)
        trace_s = time.perf_counter() - t0
        u0 = warm0["res"]["u_reduced"]

        cold, warm = [], []
        for _ in range(int(repeats)):
            cold.append(WA.solve_and_grad(
                genes, cfg, "contact_force", indentation_mm=ind, mesh=mesh,
                kinematics=kinematics)["timings"])
            warm.append(WA.solve_and_grad(
                genes, cfg, "contact_force", indentation_mm=ind, mesh=mesh,
                u_reduced0=u0, kinematics=kinematics)["timings"])

        out["filleted" if fil else "unfilleted"] = {
            "service_indentation_mm": ind,
            "secant_s": secant_s,
            "first_call_s": trace_s,
            "cold_forward_s": _med(cold, "forward_s"),
            "warm_forward_s": _med(warm, "forward_s"),
            "gradient_s": _med(warm, "gradient_s"),
            "gradient_over_cold_forward": _med(cold, "gradient_over_forward"),
            "gradient_over_warm_forward": _med(warm, "gradient_over_forward"),
            "cold": cold, "warm": warm}

    u, f = out["unfilleted"], out["filleted"]
    out["ratios"] = {k: f[k] / max(u[k], 1e-12) for k in
                     ("secant_s", "cold_forward_s", "warm_forward_s", "gradient_s")}
    return out


# ---------------------------------------------------------------------------
# C — ONE OBJECTIVE EVALUATION
# ---------------------------------------------------------------------------

def evaluation_cost(genes, cfg, kinematics, n_phase):
    """One `wheel_objective.objective` call each way, post-trace, at eight phases.

    The meshes are built here and passed in.  That is not a shortcut around the scope
    gate: `objective(meshes=...)` is the parameter `wheel_stage3.Evaluator` already hands
    a pooled caller, so this times the objective on exactly the path the optimizer uses,
    with the mesh built at the setting under test.

    THE BUILD IS COUNTED IN, SEPARATELY.  A real evaluation rebuilds every phase mesh at
    the current genes (`phase_meshes`), so `total_s` is build + evaluate and the two are
    also reported apart.

    `value` and `grad_norm` come back with the timings so a reader can see that the two
    rows are the same objective at the same genome on two MESHES, and not two different
    calculations that happen to have been timed.  They differ, and by a lot — 212.4
    against 842.8 on the sizing probe, one phase, linear — because the filleted mesh
    sits deeper into the barrier at this genome, which is §89's "what the fillet costs is
    headroom" in the objective's own units.  That is a finding about the VALUE and it
    does not touch the cost ratio, which is what this driver is for.
    """
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    out = {"n_phase": int(n_phase), "phases_deg": [float(p) for p in phases]}
    for fil in (None, True):
        t0 = time.perf_counter()
        meshes = [WW.build_wheel(genes, cfg, phase_deg=float(p), fillet=fil)
                  for p in phases]
        build_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        WO.objective(genes, cfg, phases=phases, meshes=meshes,
                     kinematics=kinematics)                                # trace
        trace_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        total, grad, bd = WO.objective(genes, cfg, phases=phases, meshes=meshes,
                                       kinematics=kinematics)
        eval_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        WO.objective(genes, cfg, phases=phases, meshes=meshes,
                     tiers=("t1", "t2"), kinematics=kinematics)
        t1t2_s = time.perf_counter() - t0

        out["filleted" if fil else "unfilleted"] = {
            "build_s": build_s, "first_call_s": trace_s, "eval_s": eval_s,
            "trace_overhead_s": trace_s - eval_s,
            "total_s": build_s + eval_s,
            "t1_t2_s": t1t2_s, "t3_s": eval_s - t1t2_s,
            "value": float(total),
            "grad_norm": float(np.linalg.norm(grad)),
            "min_scaled_jacobian": bd["report"].get("min_scaled_jacobian"),
            "mesh_mass_g": bd["report"].get("mesh_mass_g")}

    u, f = out["unfilleted"], out["filleted"]
    out["ratios"] = {k: f[k] / max(u[k], 1e-12) for k in
                     ("build_s", "trace_overhead_s", "eval_s", "total_s",
                      "t1_t2_s", "t3_s")}
    return out


# ---------------------------------------------------------------------------
# THE VERDICT
# ---------------------------------------------------------------------------

def verdict(rec):
    """§88's bracket against the measurement, on the row that carries the decision.

    The claim is about "the filleted mesh", which is a per-EVALUATION statement, so it is
    adjudicated on `total_s` at SVK — build plus evaluate, the whole of what an optimizer
    step pays for one design.  The other altitudes are reported but do not decide: the
    build ratio and the element ratio are both quantities §88 could have meant and neither
    is what an optimizer spends its time on.
    """
    # The verdict is an SVK statement and says so rather than falling back to whatever
    # row happens to be present: a cost ratio taken under the linear kernel is not the
    # one an optimizer step pays, and quoting it as if it were is the substitution
    # FILLET_PLAN's header warns about.
    if "svk" not in rec["evaluation"]:
        return {"kinematics": None, "claim": {"low": PART88_CLAIM[0],
                                              "high": PART88_CLAIM[1]},
                "unadjudicated": "no svk row in this run"}
    svk = rec["evaluation"]["svk"]
    r = svk["ratios"]["total_s"]
    lo, hi = PART88_CLAIM
    return {
        "kinematics": "svk",
        "cost_ratio_total_s": r,
        "cost_ratio_eval_s": svk["ratios"]["eval_s"],
        "claim": {"low": lo, "high": hi},
        "claim_holds": bool(lo <= r <= hi),
        # Reads "the MEASURED RATIO is <direction> the claim", which is the direction
        # the sentence is printed in.
        "direction": ("within" if lo <= r <= hi else
                      "below" if r < lo else "above"),
        "element_ratio": rec["build"]["ratio_elements"],
        "linear_cost_ratio_total_s": rec["evaluation"]["linear"]["ratios"]["total_s"]}


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def _print(rec):
    def head(s):
        print(f"\n{'=' * 78}\n  {s}\n{'=' * 78}")

    print(f"\n{'=' * 78}")
    print("  WHAT THE FILLETED OBJECTIVE COSTS — PLAN §89 SUCCESSOR 1")
    print(f"  genome {rec['genome']}, {rec['config']} mesh, "
          f"{rec['evaluation']['linear']['n_phase']} phases")
    print(f"{'=' * 78}")

    b = rec["build"]
    head("A  THE MESH BUILD")
    print(f"    {'':>12s} {'build s':>10s} {'first s':>10s} {'nodes':>8s} "
          f"{'elements':>10s}")
    for k in ("unfilleted", "filleted"):
        print(f"    {k:>12s} {b[k]['build_s']:10.3f} {b[k]['first_build_s']:10.3f} "
              f"{b[k]['n_nodes']:8d} {b[k]['n_elements']:10d}")
    print(f"    {'ratio':>12s} {b['ratio_build_s']:10.2f} {'':>10s} "
          f"{b['ratio_nodes']:8.3f} {b['ratio_elements']:10.3f}")
    print(f"    the element ratio is the 1.27x §89 quoted, and it is NOT the cost ratio")

    for kin in rec["kinematics"]:
        s = rec["solve"][kin]
        head(f"B  ONE SOLVE AND ONE ADJOINT — kinematics={kin}")
        print(f"    {'':>12s} {'secant s':>9s} {'cold fwd':>9s} {'warm fwd':>9s} "
              f"{'grad s':>8s} {'g/cold':>7s} {'g/warm':>7s} {'delta mm':>9s}")
        for k in ("unfilleted", "filleted"):
            r = s[k]
            print(f"    {k:>12s} {r['secant_s']:9.2f} {r['cold_forward_s']:9.2f} "
                  f"{r['warm_forward_s']:9.2f} {r['gradient_s']:8.2f} "
                  f"{r['gradient_over_cold_forward']:7.3f} "
                  f"{r['gradient_over_warm_forward']:7.2f} "
                  f"{r['service_indentation_mm']:9.4f}")
        print(f"    {'ratio':>12s} {s['ratios']['secant_s']:9.2f} "
              f"{s['ratios']['cold_forward_s']:9.2f} "
              f"{s['ratios']['warm_forward_s']:9.2f} "
              f"{s['ratios']['gradient_s']:8.2f}")
        print(f"    (each mesh at its OWN service indentation — the filleted wheel is "
              f"stiffer)")

    for kin in rec["kinematics"]:
        e = rec["evaluation"][kin]
        head(f"C  ONE OBJECTIVE EVALUATION — kinematics={kin}")
        print(f"    {'':>12s} {'build s':>9s} {'eval s':>9s} {'total s':>9s} "
              f"{'t1+t2 s':>9s} {'t3 s':>9s} {'trace s':>9s}")
        print(f"    {'':>12s} {'':>9s} {'':>9s} {'':>9s} {'':>9s} {'':>9s} "
              f"{'(once)':>9s}")
        for k in ("unfilleted", "filleted"):
            r = e[k]
            print(f"    {k:>12s} {r['build_s']:9.2f} {r['eval_s']:9.2f} "
                  f"{r['total_s']:9.2f} {r['t1_t2_s']:9.2f} {r['t3_s']:9.2f} "
                  f"{r['trace_overhead_s']:9.2f}")
        print(f"    {'ratio':>12s} {e['ratios']['build_s']:9.2f} "
              f"{e['ratios']['eval_s']:9.2f} {e['ratios']['total_s']:9.2f} "
              f"{e['ratios']['t1_t2_s']:9.2f} {e['ratios']['t3_s']:9.2f} "
              f"{e['ratios']['trace_overhead_s']:9.2f}")
        print(f"    value {e['unfilleted']['value']:.6f} -> "
              f"{e['filleted']['value']:.6f}, |grad| "
              f"{e['unfilleted']['grad_norm']:.4e} -> "
              f"{e['filleted']['grad_norm']:.4e}")

    v = rec["verdict"]
    if v.get("unadjudicated"):
        head("THE VERDICT — NOT ADJUDICATED")
        print(f"    {v['unadjudicated']}; §88's claim is about what an optimizer step")
        print(f"    pays, and that is the SVK row (§32).")
        print()
        return
    head("THE VERDICT — §88 RANKING ITEM 2's \"2-3x\", MEASURED")
    print(f"    one evaluation, {rec['config']}, svk, "
          f"{rec['evaluation']['svk']['n_phase']} phases, post-trace, build included")
    print(f"    measured cost ratio      {v['cost_ratio_total_s']:.2f}x")
    print(f"    §88 claimed              {v['claim']['low']:.0f}-"
          f"{v['claim']['high']:.0f}x")
    print(f"    the measured ratio is    {v['direction'].upper()} the claim"
          f"   [§88's \"2-3x\" "
          f"{'HOLDS' if v['claim_holds'] else 'DOES NOT HOLD'}]")
    print(f"    same evaluation, linear  {v['linear_cost_ratio_total_s']:.2f}x   "
          f"(what `objective` takes when called bare — §32)")
    print(f"    element count ratio      {v['element_ratio']:.2f}x   "
          f"(§89's 1.27x, a different quantity)")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genome", default="best_solution.json")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--kinematics", default=",".join(DEFAULT_KINEMATICS))
    ap.add_argument("--phases", type=int, default=DEFAULT_PHASES)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--out", default="study_fillet_cost.json")
    args = ap.parse_args()

    # What degrades THIS driver is measuring a cheaper evaluation than the one the
    # decision is about: fewer phases than the optimizer's stencil, a config the objective
    # does not default to, a missing SVK row (the only row the verdict reads), or a
    # different wheel.
    _gate_guard.refuse_degraded_out(ap, args, "study_fillet_cost.json", [
        (args.config != DEFAULT_CONFIG,
         "--config %s, not the %s the objective defaults to" % (args.config,
                                                                DEFAULT_CONFIG)),
        (set(args.kinematics.split(",")) != set(DEFAULT_KINEMATICS),
         "--kinematics %s, not both — the verdict reads the svk row and the linear row "
         "is what a bare call gets" % args.kinematics),
        (args.phases < DEFAULT_PHASES,
         "--phases %d, below the optimizer's %d-point stencil"
         % (args.phases, DEFAULT_PHASES)),
        (args.repeats < DEFAULT_REPEATS,
         "--repeats %d, below %d" % (args.repeats, DEFAULT_REPEATS)),
        (args.genome != "best_solution.json", "--genome %s" % args.genome),
    ])

    kinematics = tuple(args.kinematics.split(","))
    genes = load_genes(args.genome)

    t0 = time.time()
    rec = {"genome": args.genome, "config": args.config,
           "kinematics": list(kinematics),
           "service_force_n": float(SERVICE_FORCE_N),
           "build": build_cost(genes, args.config, args.repeats),
           "solve": {}, "evaluation": {}}
    # ORDER MATTERS AND IT IS NOT THE REPORT'S ORDER.  The jit cache is process-wide, so
    # whichever section runs first is the only one that can measure a COLD trace.  C is
    # the section the decision reads and the one whose trace is worth a number, so it
    # runs first and B runs behind it, warm.  The report prints A, B, C because that is
    # the order they are read in.
    for kin in kinematics:
        rec["evaluation"][kin] = evaluation_cost(genes, args.config, kin, args.phases)
    for kin in kinematics:
        rec["solve"][kin] = solve_cost(genes, args.config, kin, args.repeats)
    rec["verdict"] = verdict(rec)
    rec["wall_s"] = time.time() - t0
    _print(rec)

    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"    wrote {args.out}  ({rec['wall_s']:.1f} s)")

    # THIS DRIVER HAS NO PASS/FAIL AND EXITS 0.  It measures a cost; there is no
    # threshold a cost has to meet, and §89 asked for the number rather than for a gate.
    # The one thing that WOULD be a failure — the two rows not being evaluations of the
    # same objective — is visible in the printed `value` pair and is not a solver verdict.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
