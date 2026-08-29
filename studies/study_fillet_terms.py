"""
=============================================================================
  WHERE THE 17x LIVES — THE FILLETED OBJECTIVE'S LOSS, TERM BY TERM,
  AGAINST §90's "IF THE BARRIER IS MOST OF IT"
=============================================================================
    .venv-opt/bin/python studies/study_fillet_terms.py        (make filletterms)

PLAN.md §90's ranked successor 2.  FILLET_PLAN.md STEP 3.

WHY THIS EXISTS
---------------
§90 measured what the filleted objective COSTS — 1.12x, against §88's unmeasured "2-3x" —
and on the way past it recorded a difference it deliberately did not attribute:

    the filleted objective returns 671.66 against 38.79 at the shipped genome,
    with |grad| 1179.53 against 212.49                              (§90, svk)

and ranked the attribution second with a reason that is not curiosity:

    "if the barrier is most of it, wiring the fillet in re-weights the whole loss and
     every committed loss number becomes incomparable across the switch (§32's own rule
     about kinematics, applied to the mesh).  That is a promotion-shaped consequence."

This driver is that attribution.  It is `breakdown["terms"]` on both meshes at the same
genome, and it takes no view on whether the fillet should be wired in — it says which of
the fourteen terms the difference is made of, which is the input that decision needs.

THE HYPOTHESIS IS TESTED, NOT ASSUMED
--------------------------------------
§90's sentence names `min_sj` as the suspect and it is the obvious one: the filleted mesh
sits at min scaled Jacobian 0.2877 against 0.7822, which is most of the way from the
unfilleted mesh to `MIN_SJ_TARGET`, and the barrier's weight is 3000.  The suspect is
still a suspect, so the barrier's delta is reported as one row of fourteen and the verdict
prints whether it is most of the gap rather than assuming the sentence was right.

AND THE BARRIER ROW IS EXACT, WHICH MAKES §89's BOX RESULT TRANSFER
-------------------------------------------------------------------
`soft_barrier` is `scale * max(0, violation)^2` — exactly zero below the knee, not merely
small — and `min_sj` sums it over every element.  So `min_sj == 0.0` is not a rounding
statement: it is "every element on this mesh clears `MIN_SJ_TARGET`", and it is the SAME
statement as `min(sj) > MIN_SJ_TARGET`, which is the quantity §89 already swept over 16
in-sample and 32 held-out genomes on both meshes.  Whatever this driver finds for that one
row at the shipped genome, §89's box result carries it to the box for free, and neither
this driver nor a successor has to re-measure it.

TWO ALTITUDES, AND ONLY ONE OF THEM NEEDS A SOLVE
--------------------------------------------------
  A. THE GENE- AND MESH-SPACE TERMS.  `tiers=("t1","t2")`: seven gene-space terms plus
     `buckling`, `mass` and `min_sj`.  Seconds, no FEA, and NO KINEMATICS — T1 reads the
     genes and T2 reads `mesh_coords`, so neither can see a solver kernel and this
     altitude is reported once rather than per kinematics.  It is also where §90's
     hypothesis is decided, because `min_sj` lives here.

  B. THE WHOLE EVALUATION.  `tiers=("t1","t2","t3")`, eight phases, both meshes, both
     kinematics.  T3 adds `deflection`, `stress`, `stress_margin` and `phase_ripple`, and
     each of the four is a function of a SOLVE — so this is the altitude at which a
     difference means the physics moved and not the mesh's arithmetic.

     The T3 deltas are also the ones a promotion has to care about.  A barrier that fires
     on one mesh and not the other prices MESH VALIDITY, which is a property of the
     instrument; `deflection` and `stress` price the WHEEL, and if those are what move,
     the two meshes disagree about the part rather than about themselves.

SEVEN OF THE FOURTEEN TERMS CANNOT DIFFER, AND THAT IS CHECKED
---------------------------------------------------------------
T1 is a function of the genes and the config alone — `fillet_flanks` takes `cfg`, not a
mesh — so `x_order`, `hub_overlap`, `smoothness`, `fold`, `arrival`, `fillet`,
`fillet_cap` and `buckling` MUST come back bit-identical from the two calls.  They are not
dropped from the table for being predictable: they are the internal control that says the
two rows are one objective evaluated on two meshes, and `t1_identical` in the artifact is
that check as a fact rather than as an assumption.  If it ever goes false, the
decomposition below is comparing two different calculations and none of it means anything.

BOTH KINEMATICS, AND THE SVK ROW IS THE ONE THAT CARRIES IT
------------------------------------------------------------
Same caveat as §90 and for the same reason: `wheel_objective.objective` called bare takes
`wheel_fem`'s LINEAR default while Stage 3 passes `kinematics="svk"` (§32), so the SVK row
is what an optimizer step actually descends and the linear row is what a hand re-run gets.
Both of §90's published pairs are reproduced here — 671.66/38.79 at svk and 834.65/122.36
at linear — and a decomposition that did not reproduce them would be measuring something
else.

NOTHING HERE IS TIMED, SO NOTHING HERE IS WARMED UP
-----------------------------------------------------
`study_fillet_cost.py` precedes every timed call with an identical warm-up because the jit
trace would otherwise land inside the number.  This driver reads VALUES, which the trace
does not change, so the warm-up calls are omitted and the run is roughly half of what the
same four evaluations cost there.  The trace is still paid — §90 measured it at 271.7 s
unfilleted and 1122.0 s filleted — and it is paid by whichever kinematics runs first,
which is why `linear` runs first and why the wall clock is not evenly split.
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
import wheel_wheel as WW

HERE = os.path.dirname(os.path.abspath(__file__))

# `coarse` is the config the objective defaults to, the one §90's pair was taken at, and
# the one §89's barrier control used.  A decomposition of §90's numbers has to be taken
# where §90 took them or it is decomposing a different pair.
DEFAULT_CONFIG = "coarse"

# Both, and the SVK row carries the decision — see the module docstring.
DEFAULT_KINEMATICS = ("linear", "svk")

# The optimizer's own stencil, taken deterministically (`uniform`, not `rqmc`) so the two
# rows differ by the fillet and not by a random phase offset.  §90's pair is at eight.
DEFAULT_PHASES = 8

# The eight terms that are functions of the genes alone and so cannot differ between the
# two meshes.  Seven T1 names plus `buckling`, which `objective` computes inside the "t1"
# tier from the numpy beam surrogate.  Checked, not assumed — see the module docstring.
GENE_SPACE_TERMS = tuple(WO.T1_NAMES) + ("buckling",)

# What T2 adds: the two terms that read `mesh_coords` and nothing else.
MESH_SPACE_TERMS = tuple(WO.T2_NAMES)

# And what T3 adds: the four that are functions of a SOLVE.
SOLVE_SPACE_TERMS = ("deflection", "stress", "stress_margin", "phase_ripple")

# The report leaves worth carrying beside the terms.  Not decoration: each one is the
# physical quantity the term above it is computed FROM, so a delta in `deflection` can be
# read as a delta in millimetres of axle drop rather than as an unexplained 400.
REPORT_KEYS = ("axle_drop_mean_mm", "phase_ripple_std_over_mean", "pnorm_stress_agg_mpa",
               "max_stress_mpa", "kt_hub", "kt_rim", "r_hub_effective_mm",
               "hub_fillet_cap_mm", "stress_utilisation_hub", "stress_utilisation_rim",
               "mesh_mass_g", "min_scaled_jacobian")


def load_genes(path):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


def _terms(bd):
    return {k: {"value": v["value"], "grad_norm": v["grad_norm"],
                "share": v["share"], "grad_share": v["grad_share"]}
            for k, v in bd["terms"].items()}


def _compare(unfilleted, filleted):
    """The two term tables against each other, as one row per term.

    `share_of_gap` is `delta / (total_filleted - total_unfilleted)` and it is signed: a
    term that moves the OTHER way gets a negative share and the column still sums to 1.0.
    That is deliberate — a decomposition whose parts do not sum to the whole is not one,
    and rounding a term to zero because it looks small is how a cancellation gets missed.
    """
    gap = filleted["total"] - unfilleted["total"]
    rows = {}
    for k in unfilleted["terms"]:
        u, f = unfilleted["terms"][k], filleted["terms"][k]
        rows[k] = {
            "unfilleted": u["value"], "filleted": f["value"],
            "delta": f["value"] - u["value"],
            "share_of_gap": (f["value"] - u["value"]) / gap if gap else 0.0,
            "grad_norm_unfilleted": u["grad_norm"],
            "grad_norm_filleted": f["grad_norm"],
            "grad_norm_delta": f["grad_norm"] - u["grad_norm"]}
    return {"gap": gap, "rows": rows}


# ---------------------------------------------------------------------------
# A — THE GENE- AND MESH-SPACE TERMS
# ---------------------------------------------------------------------------

def mesh_space_terms(genes, cfg):
    """`tiers=("t1","t2")` on both meshes.  No solve, and so no kinematics.

    This is the altitude §90's hypothesis lives at: `min_sj` is a T2 term, and whether the
    barrier is most of the difference is decided here in seconds rather than in the
    thirty-odd minutes altitude B costs.

    ONE MESH EACH, AT PHASE ZERO.  T2 reads `meshes[0]` and nothing else — `objective`'s
    own `mesh0` — so a phase stencil would be eight builds to feed a term that would read
    the first of them.  Altitude B builds the full stencil because T3 needs it.
    """
    out = {}
    for fil in (None, True):
        mesh = WW.build_wheel(genes, cfg, fillet=fil)
        total, grad, bd = WO.objective(genes, cfg, meshes=[mesh], tiers=("t1", "t2"))
        out["filleted" if fil else "unfilleted"] = {
            "total": float(total), "grad_norm": float(np.linalg.norm(grad)),
            "terms": _terms(bd),
            "report": {k: bd["report"][k] for k in REPORT_KEYS if k in bd["report"]}}
    out["comparison"] = _compare(out["unfilleted"], out["filleted"])
    return out


# ---------------------------------------------------------------------------
# B — THE WHOLE EVALUATION
# ---------------------------------------------------------------------------

def evaluation_terms(genes, cfg, kinematics, n_phase):
    """One full `objective` call each way, eight phases, and the two tables compared.

    The meshes are built here and handed in through `meshes=`, which is the parameter
    `wheel_stage3.Evaluator` already uses for a pooled caller — so this decomposes the
    objective on the path the optimizer runs, with the mesh built at the setting under
    test, and it touches no `src/` module to do it.
    """
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    out = {"n_phase": int(n_phase), "phases_deg": [float(p) for p in phases]}
    for fil in (None, True):
        meshes = [WW.build_wheel(genes, cfg, phase_deg=float(p), fillet=fil)
                  for p in phases]
        t0 = time.time()
        total, grad, bd = WO.objective(genes, cfg, phases=phases, meshes=meshes,
                                       kinematics=kinematics)
        out["filleted" if fil else "unfilleted"] = {
            "total": float(total), "grad_norm": float(np.linalg.norm(grad)),
            "seconds": time.time() - t0,
            "terms": _terms(bd),
            "report": {k: bd["report"][k] for k in REPORT_KEYS if k in bd["report"]}}
    out["comparison"] = _compare(out["unfilleted"], out["filleted"])

    # The control described in the module docstring, as a fact in the artifact.  Bitwise,
    # because these terms are the same float computed twice from the same genes and a
    # tolerance here would be hiding something rather than allowing for something.
    u, f = out["unfilleted"]["terms"], out["filleted"]["terms"]
    out["t1_identical"] = bool(all(u[k]["value"] == f[k]["value"]
                                   for k in GENE_SPACE_TERMS))
    out["t1_disagreements"] = sorted(k for k in GENE_SPACE_TERMS
                                     if u[k]["value"] != f[k]["value"])
    return out


# ---------------------------------------------------------------------------
# THE VERDICT
# ---------------------------------------------------------------------------

def verdict(rec):
    """§90's sentence against the decomposition, on the row that carries the decision.

    Two questions, and they are not the same one.  "Is the barrier most of it" is §90's
    own hypothesis and is answered by `min_sj`'s share.  "Is the difference in the
    PHYSICS" is the one the promotion consequence actually turns on: a gap made of T3 is
    two meshes disagreeing about the wheel, which no amount of barrier tuning reaches.
    """
    if "svk" not in rec["evaluation"]:
        return {"kinematics": None, "unadjudicated": "no svk row in this run"}
    ev = rec["evaluation"]["svk"]
    cmp_ = ev["comparison"]
    rows, gap = cmp_["rows"], cmp_["gap"]

    ranked = sorted(rows.items(), key=lambda kv: -abs(kv[1]["delta"]))
    top, topr = ranked[0]

    def group(names):
        d = sum(rows[k]["delta"] for k in names)
        return {"delta": d, "share_of_gap": d / gap if gap else 0.0}

    barrier = rows["min_sj"]
    return {
        "kinematics": "svk",
        "total_unfilleted": ev["unfilleted"]["total"],
        "total_filleted": ev["filleted"]["total"],
        "gap": gap,
        "ratio": (ev["filleted"]["total"] / ev["unfilleted"]["total"]
                  if ev["unfilleted"]["total"] else None),
        "largest_term": top,
        "largest_term_delta": topr["delta"],
        "largest_term_share_of_gap": topr["share_of_gap"],
        # §90's hypothesis, adjudicated.  "Most of it" is more than half the gap.
        "barrier_delta": barrier["delta"],
        "barrier_share_of_gap": barrier["share_of_gap"],
        "barrier_is_most_of_it": bool(barrier["share_of_gap"] > 0.5),
        "barrier_is_exactly_zero_on_both_meshes": bool(
            barrier["unfilleted"] == 0.0 and barrier["filleted"] == 0.0),
        "gene_space": group(GENE_SPACE_TERMS),
        "mesh_space": group(MESH_SPACE_TERMS),
        "solve_space": group(SOLVE_SPACE_TERMS),
        "t1_identical": ev["t1_identical"],
        "grad_norm_unfilleted": ev["unfilleted"]["grad_norm"],
        "grad_norm_filleted": ev["filleted"]["grad_norm"]}


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def _print(rec):
    def head(s):
        print(f"\n{'=' * 78}\n  {s}\n{'=' * 78}")

    def table(cmp_, note):
        rows = cmp_["rows"]
        print(f"    {'term':<15s}{'unfilleted':>13s}{'filleted':>13s}"
              f"{'delta':>13s}{'share':>9s}{'|g| delta':>12s}")
        for k, r in sorted(rows.items(), key=lambda kv: -abs(kv[1]["delta"])):
            print(f"    {k:<15s}{r['unfilleted']:13.4f}{r['filleted']:13.4f}"
                  f"{r['delta']:13.4f}{r['share_of_gap']:9.3f}"
                  f"{r['grad_norm_delta']:12.4f}")
        print(f"    {'GAP':<15s}{'':>13s}{'':>13s}{cmp_['gap']:13.4f}"
              f"{1.0:9.3f}")
        print(f"    {note}")

    print(f"\n{'=' * 78}")
    print("  WHERE THE 17x LIVES — PLAN §90 SUCCESSOR 2")
    print(f"  genome {rec['genome']}, {rec['config']} mesh, "
          f"{rec['evaluation'][rec['kinematics'][0]]['n_phase']} phases")
    print(f"{'=' * 78}")

    head("A  THE GENE- AND MESH-SPACE TERMS — no solve, no kinematics")
    table(rec["mesh_space"]["comparison"],
          "T1 reads the genes and T2 reads mesh_coords; neither sees a solver kernel")
    for k in ("unfilleted", "filleted"):
        r = rec["mesh_space"][k]["report"]
        print(f"    {k:>12s}  mass {r['mesh_mass_g']:8.4f} g   "
              f"min scaled J {r['min_scaled_jacobian']:8.4f}")

    for kin in rec["kinematics"]:
        e = rec["evaluation"][kin]
        head(f"B  THE WHOLE EVALUATION — kinematics={kin}")
        table(e["comparison"],
              f"t1_identical {e['t1_identical']}"
              + ("" if e["t1_identical"]
                 else "  <-- " + ",".join(e["t1_disagreements"])))
        print(f"    total {e['unfilleted']['total']:.4f} -> "
              f"{e['filleted']['total']:.4f}   |grad| "
              f"{e['unfilleted']['grad_norm']:.4f} -> "
              f"{e['filleted']['grad_norm']:.4f}")
        print(f"    {'':>12s} {'drop mm':>9s} {'ripple':>8s} {'pnorm MPa':>10s} "
              f"{'max MPa':>9s} {'kt_hub':>8s} {'kt_rim':>8s} {'util hub':>9s} "
              f"{'util rim':>9s}")
        for k in ("unfilleted", "filleted"):
            r = e[k]["report"]
            print(f"    {k:>12s} {r['axle_drop_mean_mm']:9.4f} "
                  f"{r['phase_ripple_std_over_mean']:8.4f} "
                  f"{r['pnorm_stress_agg_mpa']:10.4f} {r['max_stress_mpa']:9.4f} "
                  f"{r['kt_hub']:8.4f} {r['kt_rim']:8.4f} "
                  f"{r['stress_utilisation_hub']:9.4f} "
                  f"{r['stress_utilisation_rim']:9.4f}")

    v = rec["verdict"]
    if v.get("unadjudicated"):
        head("THE VERDICT — NOT ADJUDICATED")
        print(f"    {v['unadjudicated']}; §90's pair is an SVK pair (§32).")
        print()
        return
    head("THE VERDICT — §90's \"IF THE BARRIER IS MOST OF IT\"")
    print(f"    one evaluation, {rec['config']}, svk, "
          f"{rec['evaluation']['svk']['n_phase']} phases")
    print(f"    loss  {v['total_unfilleted']:.4f} -> {v['total_filleted']:.4f}"
          f"   gap {v['gap']:.4f}   ratio {v['ratio']:.2f}x")
    print(f"    the barrier `min_sj` is  {v['barrier_share_of_gap'] * 100:6.2f}% of the "
          f"gap   [§90's hypothesis "
          f"{'HOLDS' if v['barrier_is_most_of_it'] else 'DOES NOT HOLD'}]")
    if v["barrier_is_exactly_zero_on_both_meshes"]:
        print(f"    and it is EXACTLY 0.0 on both meshes, which is `soft_barrier`'s "
              f"max(0,.)^2 saying")
        print(f"    every element clears MIN_SJ_TARGET — the same statement §89 swept "
              f"over 32 genomes")
    print(f"    largest single term      {v['largest_term']}  "
          f"{v['largest_term_delta']:+.4f}  "
          f"({v['largest_term_share_of_gap'] * 100:.1f}% of the gap)")
    print(f"    gene-space terms         {v['gene_space']['delta']:+12.4f}  "
          f"({v['gene_space']['share_of_gap'] * 100:6.2f}%)   "
          f"t1_identical {v['t1_identical']}")
    print(f"    mesh-space terms         {v['mesh_space']['delta']:+12.4f}  "
          f"({v['mesh_space']['share_of_gap'] * 100:6.2f}%)")
    print(f"    solve-space terms        {v['solve_space']['delta']:+12.4f}  "
          f"({v['solve_space']['share_of_gap'] * 100:6.2f}%)")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genome", default="best_solution.json")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--kinematics", default=",".join(DEFAULT_KINEMATICS))
    ap.add_argument("--phases", type=int, default=DEFAULT_PHASES)
    ap.add_argument("--out", default="study_fillet_terms.json")
    args = ap.parse_args()

    # What degrades THIS driver is decomposing a different pair than the one §90
    # published: another config, another genome, a shorter stencil, or no SVK row — and
    # the SVK row is the only one the verdict reads.
    _gate_guard.refuse_degraded_out(ap, args, "study_fillet_terms.json", [
        (args.config != DEFAULT_CONFIG,
         "--config %s, not the %s §90's pair was taken at" % (args.config,
                                                              DEFAULT_CONFIG)),
        (set(args.kinematics.split(",")) != set(DEFAULT_KINEMATICS),
         "--kinematics %s, not both — the verdict reads the svk row and the linear row "
         "is what a bare call gets" % args.kinematics),
        (args.phases < DEFAULT_PHASES,
         "--phases %d, below the optimizer's %d-point stencil"
         % (args.phases, DEFAULT_PHASES)),
        (args.genome != "best_solution.json", "--genome %s" % args.genome),
    ])

    kinematics = tuple(args.kinematics.split(","))
    genes = load_genes(args.genome)

    t0 = time.time()
    rec = {"genome": args.genome, "config": args.config,
           "kinematics": list(kinematics),
           "service_force_n": float(W.TOTAL_FORCE_NEWTONS),
           "min_sj_target": float(WO.MIN_SJ_TARGET),
           "weights": {k: float(v) for k, v in WO.DEFAULT_WEIGHTS.items()},
           "mesh_space": mesh_space_terms(genes, args.config),
           "evaluation": {}}
    # `linear` first, so the row that pays §90's 1122 s filleted trace is the one whose
    # seconds nobody reads.  Nothing here is timed — see the module docstring — but the
    # trace belongs to the mesh rather than to the kernel, so which kinematics pays it is
    # a real difference in the wall clock and it is chosen rather than left to argv order.
    for kin in ("linear", "svk"):
        if kin in kinematics:
            rec["evaluation"][kin] = evaluation_terms(genes, args.config, kin,
                                                      args.phases)
    rec["verdict"] = verdict(rec)
    rec["wall_s"] = time.time() - t0
    _print(rec)

    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"    wrote {args.out}  ({rec['wall_s']:.1f} s)")

    # THIS DRIVER HAS NO PASS/FAIL AND EXITS 0, for the same reason `study_fillet_cost.py`
    # does not: §90 asked which terms the difference is made of, and an attribution has no
    # threshold to meet.  The one thing that WOULD be a failure is `t1_identical` going
    # false — that is the two rows not being the same objective — and it is printed and
    # filed rather than raised, because the table beside it is what makes it diagnosable.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
