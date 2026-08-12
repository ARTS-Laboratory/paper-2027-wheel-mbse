"""
=============================================================================
  SVK RE-SCORE — WHAT THE SHIPPED WHEEL AND ITS RIVALS ACTUALLY COST
=============================================================================
    .venv-opt/bin/python studies/study_svk_rescore.py        (make svk)

SVK_PLAN.md step 3.  No optimizer, no descent: forward evaluations of the SAME objective
Stage 3 descends, at a handful of committed genomes, under BOTH kinematics.

WHY IT EXISTS.  Every Stage-3 headline in PLAN.md was computed under `kinematics="linear"`
— which is what `wheel_contact_problem` defaults to — and nothing in any record said so.
PLAN.md §14 caught it late and priced the shipped genome's axle drop at +23.346% under
SVK, then ESTIMATED the stress utilisation at "on the order of 0.91" by scaling a reported
figure by a p99 ratio.  The constraint does not compute a p99.  It computes
`Kt * pnorm(p=4) / ALLOWABLE_STRESS_MPA` per junction (`wheel_objective.py` around the
`utils[name]` loop) and takes the worse of hub and rim.  So the one question this file
answers is not rhetorical and not expensive:

    IS THE SHIPPED GENOME STILL FEASIBLE UNDER SVK — every barrier at 0.0 and
    utilisation below 1.0 — OR IS IT NOT?

DELIBERATELY NOT IN `make studies`, for the reason PLAN.md gives for `m8bi5`, `m9buck`
and `hubcap`: it measures THE WHEEL, NOT THE COMMIT.  Its answer does not move when the
code changes, it costs the better part of an hour at `medium`, and a gate nobody can
afford to run stops being run.  `make svk` runs it on purpose.

THE CONTROL COLUMN IS NOT PADDING, AND IT IS THE FIRST THING TO READ.  `36aed36`
(`best_solution_ga_beam.json`) is the GA/beam genome — a thicker, duller wheel that §14
measured at +3.953% under SVK against the shipped genome's +23.346%.  `run_control`
reproduces those two numbers with study_gnl's own force-controlled single-phase solve
before any of the phase-averaged columns are believed.  If the control does not come back
at +3.953%, this driver is wrong and the shipped column means nothing.  That rule caught
three of §14's eight "the new design broke it" claims.

TWO DIFFERENT AXLE DROPS LIVE IN THIS FILE AND THEY ARE NOT THE SAME NUMBER.
  `run_control`  solves ONE mesh at ONE phase to a FIXED FORCE — study_gnl's quantity,
                 the one §14's ladder reports, and the only one comparable to §14.
  `run_rescore`  is the OBJECTIVE'S quantity: the mean over an 8-phase stencil, which is
                 what `deflection` is actually scored on.
They differ by more than the SVK correction does, so a table that mixed them would look
like a physics finding.  Both are reported, never averaged.

MESHES ARE BUILT ONCE PER GENOME AND SHARED BY BOTH KINEMATICS.  A strain measure is a
property of the kernel, not of the geometry, so rebuilding would only add a chance for the
two columns to differ for a reason that is not the strain measure.
=============================================================================
"""

import argparse
import json
import os
import time

import numpy as np

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import wheel_fea as W
import wheel_fem as fem
import wheel_objective as WO
import wheel_pool as WP
import wheel_wheel as WW

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = "medium"
CONTROL_CONFIG = "coarse"       # what §14's ladder ran at; changing it breaks the control
SERVICE_FORCE_N = W.TOTAL_FORCE_NEWTONS
N_PHASE = 8

# What PLAN.md §14 recorded, written here BEFORE this file was run.  These are the
# reproduction targets for `run_control`, not thresholds on the wheel.
PLAN14_SHIPPED_SERVICE_REL = 0.23346
PLAN14_GA_BEAM_SERVICE_REL = 0.03953
GATE_CONTROL_REL = 0.02         # 2% OF THE CORRECTION, i.e. 0.234 must land in 0.229-0.238

# The terms that are barriers: each is `soft_barrier(...)` of a violation, so 0.0 means
# satisfied and anything positive is a breach.  Named explicitly rather than derived by
# subtracting the three headline terms, so a term added upstream lands in neither list and
# fails loudly in `_barriers` instead of being silently reclassified as a barrier.
BARRIER_NAMES = ("x_order", "hub_overlap", "fold", "arrival", "fillet",
                 "fillet_cap", "buckling", "min_sj", "stress")
# NOT barriers, and `smoothness` is the one that matters here: it is a curvature-rate
# INTEGRAL (`wheel_objective.t1_vector`, "smoothness, REWRITTEN"), so it is positive at
# every real genome — 0.29 at the shipped one.  Listing it as a barrier reports the
# PROMOTED DESIGN as infeasible, which is what the first run of this file did.
HEADLINE_NAMES = ("mass", "deflection", "phase_ripple", "smoothness")

# `36aed36` IS EXPECTED TO COME BACK INFEASIBLE and that is not a finding.  It predates
# the fillet-cap work, and `wheel_objective.DEFAULT_WEIGHTS`' own comment names its
# numbers: "a genome that puts this term at 103.2 against `hub_overlap`'s 52.1 and
# `mass`'s 55.3".  It is here as a control on the SVK CORRECTION — a thicker, duller
# wheel whose correction §14 measured at +3.953% — not as a promotion candidate.

# `minwall 1.2` IS THE SHIPPED GENOME, BIT-FOR-BIT.  Measured: max|gene diff| against
# `best_solution.json` is 0.0, and the shipped file's own note says it was promoted "from
# stage3_minwall_best_1.2.json, verbatim apart from this note".  (So is
# `stage3_minwall_best_1.2_medium.json`, whose `search` block reads `"steps": 0` — that
# artifact is a RE-SCORE, never a descent.)  The row is kept because two rows of one table
# that must agree to the last digit is a free check on the pool reduction and the mesh
# cache, but it is NOT a seventh design and must never be counted as independent evidence:
# there are SIX distinct genomes below.
GENOMES = (
    ("350f4c7 shipped", "best_solution.json"),
    ("36aed36 GA/beam", "best_solution_ga_beam.json"),
    ("elite10", "stage3_prod_best_elite10.json"),
    ("minwall 1.2", "stage3_minwall_best_1.2.json"),
    ("minwall 1.4", "stage3_minwall_best_1.4.json"),
    ("minwall 1.6", "stage3_minwall_best_1.6.json"),
    ("minwall 2.0", "stage3_minwall_best_2.0.json"),
)


def load_genes(path):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return np.array(list(json.load(fh)["genes"].values()), dtype=float)


def _barriers(terms):
    """`(dict of breaches, worst)`.  Raises if the term set moved under this file."""
    missing = set(BARRIER_NAMES + HEADLINE_NAMES) - set(terms)
    extra = set(terms) - set(BARRIER_NAMES + HEADLINE_NAMES)
    if missing or extra:
        raise RuntimeError(
            f"the objective's term set moved: missing {sorted(missing)}, "
            f"unclassified {sorted(extra)}. Classify it in BARRIER_NAMES or "
            f"HEADLINE_NAMES — an unclassified term would be reported as feasible")
    breached = {k: terms[k] for k in BARRIER_NAMES if terms[k] > 0.0}
    return breached, (max(breached.values()) if breached else 0.0)


# ---------------------------------------------------------------------------
# THE CONTROL — reproduce §14's ladder before believing anything else
# ---------------------------------------------------------------------------

def run_control(cfg=CONTROL_CONFIG):
    """§14's two service-load numbers, recomputed from scratch in this tree.

    Force-controlled, one mesh, one phase, `fem.solve_wheel` — study_gnl's `_drop`, not
    the objective's stencil mean.  This is the ONLY quantity in this file that §14 can be
    compared against, and it is run first so that a driver that is wired wrong is caught
    before an hour of `medium` solves has been spent.
    """
    rows = []
    for label, path in (("350f4c7 shipped", "best_solution.json"),
                        ("36aed36 GA/beam", "best_solution_ga_beam.json")):
        genes = load_genes(path)
        mesh = WW.build_wheel(genes, cfg)
        lin = fem.solve_wheel(mesh, kinematics="linear", force=SERVICE_FORCE_N)
        svk = fem.solve_wheel(mesh, kinematics="svk", force=SERVICE_FORCE_N)
        rel = float(svk["axle_drop_mm"] / lin["axle_drop_mm"] - 1.0)
        expect = (PLAN14_SHIPPED_SERVICE_REL if "350f4c7" in label
                  else PLAN14_GA_BEAM_SERVICE_REL)
        rows.append({
            "genome": label, "file": path,
            "linear_mm": float(lin["axle_drop_mm"]),
            "svk_mm": float(svk["axle_drop_mm"]),
            "service_rel_diff": rel,
            "plan14_rel_diff": expect,
            "rel_error_vs_plan14": float(abs(rel - expect) / expect),
            "newton_iterations": int(svk["newton"]["total_iterations"]),
            "pass": bool(abs(rel - expect) / expect < GATE_CONTROL_REL),
        })
    return {"config": cfg, "force_n": SERVICE_FORCE_N, "rows": rows,
            "gate_rel": GATE_CONTROL_REL,
            "pass": bool(all(r["pass"] for r in rows))}


# ---------------------------------------------------------------------------
# THE RE-SCORE — the objective's own quantities, both kinematics, shared meshes
# ---------------------------------------------------------------------------

def _score(genes, cfg, phases, meshes, kinematics, pool=None, orientation=None):
    val, _, brk = WO.objective(
        genes, cfg, normalized=False, phases=phases, meshes=meshes,
        stress_p_probe=(WO.STRESS_NOMINAL_P, 30.0),
        pool=pool, orientation=orientation, kinematics=kinematics)
    terms = {k: d["value"] for k, d in brk["terms"].items()}
    rep = brk["report"]
    breached, worst = _barriers(terms)
    p30 = rep["pnorm_by_p"][repr(30.0)]
    drop = rep["axle_drop_mean_mm"]
    # The probe at the CONSTRAINT'S OWN exponent must reproduce the constraint's own
    # utilisation, because both go through `_stress_aggregate` and `Kt` is the same
    # `max(kt_hub, kt_rim)`.  Asserted rather than assumed: it is the one line that says
    # the p=30 column below is the same construction as the verdict column, differing
    # only in the exponent, and a mismatch would mean this file is reading the wrong key.
    p4 = rep["pnorm_by_p"][repr(WO.STRESS_NOMINAL_P)]
    if abs(p4["stress_utilisation_kt"] - rep["stress_utilisation"]) > 1e-12:
        raise RuntimeError(
            f"the p={WO.STRESS_NOMINAL_P} probe ({p4['stress_utilisation_kt']}) does not "
            f"reproduce the constraint's utilisation ({rep['stress_utilisation']}); the "
            f"probe and the constraint are no longer the same construction")
    return {
        "kinematics": kinematics,
        "loss": float(val),
        "axle_drop_mean_mm": float(drop),
        "axle_drop_min_mm": float(rep["axle_drop_min_mm"]),
        "axle_drop_max_mm": float(rep["axle_drop_max_mm"]),
        "deflection_error_mm": float(drop - WO.TARGET_DEFLECTION_MM),
        "deflection_error_pct": float(100.0 * (drop / WO.TARGET_DEFLECTION_MM - 1.0)),
        "kt_hub": float(rep["kt_hub"]), "kt_rim": float(rep["kt_rim"]),
        "pnorm_stress_agg_mpa": float(rep["pnorm_stress_agg_mpa"]),
        # THE CONSTRAINT'S OWN NUMBER, not a scaled estimate: max(hub, rim) of
        # Kt * pnorm(p=4) / ALLOWABLE.
        "stress_utilisation": float(rep["stress_utilisation"]),
        "stress_utilisation_hub": float(rep["stress_utilisation_hub"]),
        "stress_utilisation_rim": float(rep["stress_utilisation_rim"]),
        # The p=30 aggregate, reported because §14's estimate came from the peak end of
        # the exponent range.  M8b-i.5 measured p=30 to be NOT mesh-convergent (GCI 63%),
        # so it is a diagnostic here and never a verdict.
        "pnorm_p30_utilisation_kt": float(p30["stress_utilisation_kt"]),
        "pnorm_p30_agg_mpa": float(p30["pnorm_agg_mpa"]),
        # Diverges under refinement (PLAN.md §0) — printed, never compared.
        "max_stress_mpa": float(rep["max_stress_mpa"]),
        "mass_g": float(rep["mesh_mass_g"]),
        "phase_ripple_std_over_mean": float(rep["phase_ripple_std_over_mean"]),
        "terms": {k: float(v) for k, v in terms.items()},
        "barriers_breached": {k: float(v) for k, v in breached.items()},
        "worst_barrier": float(worst),
        "feasible": bool(not breached and rep["stress_utilisation"] < 1.0),
    }


def run_rescore(genomes=GENOMES, cfg=DEFAULT_CONFIG, n_phase=N_PHASE, workers=0):
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    rows = []
    pool = WP.PhasePool(workers) if workers else None
    try:
        for label, path in genomes:
            full = os.path.join(PP.ROOT, path)
            if not os.path.exists(full):
                rows.append({"genome": label, "file": path, "missing": True})
                print(f"  {label:<16} MISSING {path}", flush=True)
                continue
            genes = load_genes(path)
            # Pinned once per genome so both kinematics see the same discrete decision;
            # `flank_orientation` is a branch, and letting it be re-derived per column
            # would put a topology change inside a strain-measure comparison.
            orientation = tuple(float(o) for o in
                                WW.flank_orientation(genes, WW.get_config(cfg)))
            t0 = time.time()
            wanted = phases[:1] if pool is not None else phases
            meshes = WO.phase_meshes(genes, cfg, wanted, orientation=orientation)
            row = {"genome": label, "file": path,
                   "mesh_s": round(time.time() - t0, 1)}
            for kin in ("linear", "svk"):
                t1 = time.time()
                row[kin] = _score(genes, cfg, phases, meshes, kin,
                                  pool=pool, orientation=orientation)
                row[kin]["elapsed_s"] = round(time.time() - t1, 1)
                print(f"  {label:<16} {kin:<6} drop {row[kin]['axle_drop_mean_mm']:7.4f} "
                      f"util {row[kin]['stress_utilisation']:6.3f} "
                      f"loss {row[kin]['loss']:10.4f} "
                      f"{'FEASIBLE' if row[kin]['feasible'] else 'INFEASIBLE'} "
                      f"({row[kin]['elapsed_s']} s)", flush=True)
            row["drop_rel_diff"] = float(row["svk"]["axle_drop_mean_mm"]
                                         / row["linear"]["axle_drop_mean_mm"] - 1.0)
            row["util_rel_diff"] = float(row["svk"]["stress_utilisation"]
                                         / row["linear"]["stress_utilisation"] - 1.0)
            rows.append(row)
    finally:
        if pool is not None:
            pool.close()
    return {"config": cfg, "n_phase": n_phase, "scheme": "uniform",
            "workers": workers, "target_deflection_mm": WO.TARGET_DEFLECTION_MM,
            "allowable_stress_mpa": WO.ALLOWABLE_STRESS_MPA, "rows": rows}


# ---------------------------------------------------------------------------

def _print(rep):
    c = rep["control"]
    print("\n" + "=" * 92)
    if c.get("skipped"):
        # Said loudly rather than passed over: without the control, nothing below is
        # known to be comparable with §14, and that is the whole basis for quoting it.
        print("  CONTROL — SKIPPED.  The table below is NOT known to reproduce §14 and "
              "must not be quoted against it.")
        print("=" * 92)
        _print_rescore(rep)
        return
    print(f"  CONTROL — §14's force-controlled service point reproduced at {c['config']}")
    print("=" * 92)
    print(f"  {'genome':<16} {'linear mm':>10} {'svk mm':>10} {'rel':>9} "
          f"{'§14':>9} {'err':>8}  ")
    for r in c["rows"]:
        print(f"  {r['genome']:<16} {r['linear_mm']:10.6f} {r['svk_mm']:10.6f} "
              f"{100 * r['service_rel_diff']:8.3f}% {100 * r['plan14_rel_diff']:8.3f}% "
              f"{100 * r['rel_error_vs_plan14']:7.2f}%  "
              f"{'PASS' if r['pass'] else 'FAIL'}")
    print(f"  control: {'PASS' if c['pass'] else 'FAIL'}"
          f"   (gate: within {100 * c['gate_rel']:.0f}% of §14's correction)")
    _print_rescore(rep)


def _print_rescore(rep):
    rs = rep["rescore"]
    print("\n" + "=" * 92)
    print(f"  RE-SCORE — the objective's own quantities, {rs['config']}, "
          f"{rs['n_phase']} phases, target {rs['target_deflection_mm']} mm")
    print("=" * 92)
    print(f"  {'genome':<16} {'kin':<6} {'drop mm':>9} {'err %':>8} {'util':>7} "
          f"{'p30 util':>9} {'mass g':>8} {'loss':>11}  barriers")
    for r in rs["rows"]:
        if r.get("missing"):
            print(f"  {r['genome']:<16} MISSING {r['file']}")
            continue
        for kin in ("linear", "svk"):
            s = r[kin]
            b = (",".join(f"{k}={v:.3g}" for k, v in s["barriers_breached"].items())
                 or "-")
            print(f"  {r['genome']:<16} {kin:<6} {s['axle_drop_mean_mm']:9.4f} "
                  f"{s['deflection_error_pct']:7.2f}% {s['stress_utilisation']:7.3f} "
                  f"{s['pnorm_p30_utilisation_kt']:9.3f} {s['mass_g']:8.2f} "
                  f"{s['loss']:11.4f}  {b}")
        print(f"  {'':<16} {'Δsvk':<6} {100 * r['drop_rel_diff']:8.3f}% "
              f"{'':>8} {100 * r['util_rel_diff']:6.2f}%")

    print("\n  THE QUESTION THIS FILE EXISTS TO ANSWER")
    for r in rs["rows"]:
        if r.get("missing"):
            continue
        for kin in ("linear", "svk"):
            s = r[kin]
            print(f"    {r['genome']:<16} {kin:<6} "
                  f"{'FEASIBLE  ' if s['feasible'] else 'INFEASIBLE'} "
                  f"util {s['stress_utilisation']:.4f}  worst barrier "
                  f"{s['worst_barrier']:.4g}")


def main():
    ap = argparse.ArgumentParser(description="SVK_PLAN.md step 3 — re-score under SVK")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--n-phase", type=int, default=N_PHASE)
    ap.add_argument("--workers", type=int, default=0,
                    help="run the phase loop across processes; N is also the only memory "
                         "cap, exactly as in wheel_stage3")
    ap.add_argument("--out", default="study_svk_rescore.json")
    ap.add_argument("--only", default="",
                    help="comma-separated substrings; score only matching genomes")
    ap.add_argument("--extra", default="",
                    help="comma-separated label=path pairs appended to GENOMES. Additive "
                         "on purpose: Step 6 must re-score the descent winners at `medium` "
                         "without editing GENOMES, so that this driver at its DEFAULTS "
                         "still reproduces Step 3's recorded artifact unchanged")
    ap.add_argument("--skip-control", action="store_true",
                    help="skip §14's reproduction. Only for a re-run whose control "
                         "already passed in the same tree — it is the check that says "
                         "the rest of the table means anything")
    args = ap.parse_args()

    genomes = GENOMES
    if args.only:
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        genomes = tuple(g for g in GENOMES if any(w in g[0] or w in g[1] for w in want))
    for pair in (s.strip() for s in args.extra.split(",")):
        if not pair:
            continue
        if "=" not in pair:
            raise SystemExit(f"--extra wants label=path pairs, got {pair!r}")
        label, path = (s.strip() for s in pair.split("=", 1))
        if not os.path.exists(os.path.join(PP.ROOT, path)):
            raise SystemExit(f"--extra {label}: no such genome {path!r}")
        genomes = genomes + ((label, path),)

    t0 = time.time()
    print("control ...", flush=True)
    control = ({"skipped": True, "pass": None} if args.skip_control
               else run_control())
    print("re-score ...", flush=True)
    rep = {"control": control,
           "rescore": run_rescore(genomes, args.config, args.n_phase, args.workers)}
    rep["settings"] = {"config": args.config, "n_phase": args.n_phase,
                       "workers": args.workers,
                       "genomes": [list(g) for g in genomes],
                       "service_force_n": SERVICE_FORCE_N,
                       "min_wall_mm": float(W.MIN_WALL_MM),
                       "elapsed_s": round(time.time() - t0, 1)}
    # WRITTEN BEFORE IT IS PRINTED, and that order is deliberate: this run is the better
    # part of an hour at `medium`, and a formatting bug in `_print` must not be able to
    # throw the measurement away.  Same argument as `_persist` in the descent.
    out = os.path.join(HERE, args.out)
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1)
    _print(rep)
    print(f"\nwrote {out}  ({rep['settings']['elapsed_s']} s)")
    # The control is a gate on the DRIVER, not on the wheel: a red control means the
    # numbers above are not comparable to §14 and must not be quoted.
    return 0 if control.get("pass") in (True, None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
