"""
=============================================================================
  WHAT PORTFOLIO IS `DEFAULT_WEIGHTS` ALREADY RUNNING?  THE POINTS CALIBRATION
=============================================================================
    .venv-opt/bin/python studies/study_mbse_calibration.py     (make mbsecal)

MBSE_PLAN.md Step 4.  PLAN.md §97.

WHY THIS EXISTS
---------------
Fourteen numbers at `wheel_objective.py:352-393` decide what the optimiser is trying to
do.  They are individually justified — `stress_margin`'s comment is three paragraphs of
exchange rate — and they have never been justified AGAINST EACH OTHER as a portfolio.
This driver states the portfolio, in points, for the first time.

IT SOLVES NOTHING.  Every number is arithmetic on the weight table and on committed
artifacts, in `study_fillet_kt`'s idiom.  `smoothness` is the one term whose reference
cost needs a genome — its argument is an integral with no reference scale — and that is
read out of a record's `loss_terms`, never re-evaluated.

THE INSTRUMENT THAT DOES NOT WORK, AND WHY THE REPORT LEADS WITH IT
--------------------------------------------------------------------
The naive allocation is proportional to LOSS SHARE, and `best_solution.json` refutes it in
one table: the five objective terms split the loss 0.013% / 99.07% / 0.40% / 0.51% /
0.00%.  Read as an allocation that says the tree cares 99.07% about mass and 0.013% about
stroke.  **It is false, and the reason is the point.**  `deflection` is 0.013% of the loss
because the design sits at 1.99742 mm against a 2.0 mm target — a -0.129% miss.  The term
is small because the requirement is MET.  **Loss share measures satisfaction, not
priority.**  An allocation built on it would strip the weight from every requirement the
optimiser had successfully satisfied and then be surprised when the next descent stopped
satisfying them.

The marginal rate `dL/dx` fails the same way from the other side: at a satisfied quadratic
term it is near zero, so a calibration built on it is a calibration built on where the
last descent happened to stop.

WHAT REPLACES IT
-----------------
For each objective term, a REFERENCE DEVIATION `d_T` — one named physical unit of missing
that requirement — and `c_T = L(d_T)`, the loss that miss costs under `DEFAULT_WEIGHTS`.
`c_T` is a property of the WEIGHT, not of the iterate.  Then `p_cal_T = 100 c_T / sum(c)`.

THE HOLD-OUT, WHICH IS THE PART MBSE_PLAN NAMED AS THE HAZARD
--------------------------------------------------------------
*"Do not calibrate on one genome and call it a rule"* — the error §24 corrected and §73
paid for again.  The calibration is anchored at the shipped genome BY NECESSITY, and this
driver measures what that anchoring is worth by re-deriving `p_cal` at the three genomes
that shipped before it — the promotion chain `350f4c7 -> e4219f3 -> e126cc3 -> 09e8188`.
Those are HOLD-OUTS, not confirmations, and the number they produce is reported whichever
way it comes out.

`best_solution_ga_beam.json` is deliberately NOT in the hold-out set: its `loss_terms`
come from `wheel_fea.evaluate_design`'s objective, not from `wheel_objective`, so its
`smoothness` (10.39, against this era's 0.17) is a different quantity with the same name.
Comparing it would be `state-the-scope-of-a-measurement` all over again.

THE TERM THE MAP CANNOT REACH BY THE OBVIOUS ROUTE
---------------------------------------------------
`DEFAULT_WEIGHTS["phase_ripple"] = 0.0`, so `c = 0`, so `p_cal = 0`, so the obvious map
`w = w_default * p / p_cal` is `0/0` — undefined on the axis a user is most likely to
want to move first, because it is the one this tree has never bought any of.
`wheel_requirements.weights_from_priorities` states the map as COST PER POINT instead,
which has no singularity and is identical everywhere else; this driver prices what one
point of `rolling` buys and what the shipped wheel's own 10.4 points of ripple would cost
at that price.
"""

import argparse
import json
import os

import numpy as np

import project_paths as PP

import wheel_objective as WO
import wheel_requirements as R

HERE = os.path.dirname(os.path.abspath(__file__))

# The promotion chain, oldest first, with the shipped genome last.  Every one carries a
# `loss_terms` block written by `wheel_stage3`, i.e. by THIS objective.
DEFAULT_INPUTS = {
    "350f4c7": "stage3_minwall_best_1.2.json",
    "e4219f3": "stage3_buildcap2_feasible_medium.json",
    "e126cc3": "stage3_margin_best_medium.json",
    "09e8188": "best_solution.json",
}
REFERENCE = "09e8188"

# How many random allocations the conservation check draws.  The property is exact
# algebra, not a statistic, so this is a tripwire against an implementation slip rather
# than a sample: a handful of draws in general position finds a broken map immediately.
N_CONSERVATION_DRAWS = 32
CONSERVATION_TOL = 1e-12


def _load(name):
    with open(os.path.join(PP.ROOT, name)) as fh:
        return json.load(fh)


def loss_share(rec):
    """Section A — the instrument that does not work, computed so it can be refuted."""
    lt = rec["loss_terms"]
    terms = {k: float(lt[k]) for k in WO.OBJECTIVE_TERMS}
    total = sum(terms.values())
    return {"terms": terms, "total": total,
            "share": {k: (v / total if total else 0.0) for k, v in terms.items()},
            "objective_total": total,
            "record_loss": float(rec.get("loss", float("nan")))}


def calibration(smoothness_loss):
    """Sections B and C — `c_T`, `p_cal`, and the map's identity at its own anchor."""
    c = R.reference_costs(smoothness_loss)
    p, _ = R.calibrated_priorities(smoothness_loss)
    w = R.weights_from_priorities(p, smoothness_loss)
    default = dict(WO.DEFAULT_WEIGHTS)
    err = {k: abs(w[k] - default[k]) / max(abs(default[k]), 1e-30) for k in default}
    return {
        "reference_deviation": {k: {"what": v[0], "d": v[1]}
                                for k, v in R.REFERENCE_DEVIATION.items()},
        "costs": c,
        "cost_total": sum(c.values()),
        "cost_per_point": R.cost_per_point(smoothness_loss),
        "p_cal": p.as_dict(),
        "identity_weights": w,
        "identity_max_rel_err": max(err.values()),
        "identity_worst_term": max(err, key=err.get),
    }


def conservation(smoothness_loss, seed=0):
    """Section D — total exchange-rate pressure is invariant under ANY reallocation.

    THE PROPERTY IS WHY THE BUDGET IS 100 AND NOT A UI CONVENTION.  Weights are not
    scale-free in this objective: `BARRIER_TERMS` are absolute, so doubling every
    objective weight does not leave the optimum alone — it HALVES the effective strength
    of every `shall`.  `sum p = 100` is precisely what forbids a user buying more of
    everything and quietly weakening every feasibility barrier in the process.

    The draws include the two extreme allocations — everything on one axis — because the
    map's one special case (`phase_ripple`, whose default weight is zero) is reachable
    only by giving it points, and an all-ripple allocation is the sharpest form of that.
    """
    axes = R.priority_axes()
    base = sum(R.reference_costs(smoothness_loss).values())
    rng = np.random.default_rng(seed)
    draws = []
    for k in axes:                                        # 100 points on one axis each
        draws.append({a: (100.0 if a == k else 0.0) for a in axes})
    for _ in range(N_CONSERVATION_DRAWS):
        x = rng.random(len(axes))
        x = 100.0 * x / x.sum()
        draws.append(dict(zip(axes, x)))
    rows = []
    for pts in draws:
        p = R.Priorities(pts)
        w = R.weights_from_priorities(p, smoothness_loss)
        tot = sum(R.reference_costs(smoothness_loss, w).values())
        rows.append({"points": p.as_dict(), "pressure": tot,
                     "abs_err": abs(tot - base),
                     "w_phase_ripple": w["phase_ripple"],
                     # The barriers must come through UNTOUCHED — a point that reached a
                     # `shall` would be a design that bought its way out of a mesh that
                     # does not integrate.
                     "barriers_unmoved": all(
                         w[b] == WO.DEFAULT_WEIGHTS[b] for b in WO.BARRIER_TERMS)})
    return {"base_pressure": base, "rows": rows,
            "max_abs_err": max(r["abs_err"] for r in rows),
            "all_barriers_unmoved": all(r["barriers_unmoved"] for r in rows)}


def ripple_anchor(smoothness_loss, rec):
    """Section E — the axis with no default weight, priced.

    `w_ripple(p) = cost_per_point * p / d^2` is the map with the singularity removed, and
    `per_point` below is its slope.  The two numbers that make the anchor arguable rather
    than decorative are filed beside it: what one point of `rolling` buys, and what the
    SHIPPED wheel's own ripple would cost at the allocation `mass` currently holds.
    """
    unit = R.cost_per_point(smoothness_loss)
    d = R.RIPPLE_REFERENCE_DEVIATION
    per_point = unit / d ** 2
    p_cal, _ = R.calibrated_priorities(smoothness_loss)
    at_mass_parity = per_point * p_cal.points["mass"]
    shipped = float(rec["metrics"]["phase_ripple_std_over_mean"])
    return {
        "reference_deviation": d,
        "cost_per_point": unit,
        "weight_per_point": per_point,
        "points_held_by_mass": p_cal.points["mass"],
        "weight_at_mass_parity": at_mass_parity,
        "shipped_std_over_mean": shipped,
        "shipped_points_of_ripple": shipped / d,
        "term_at_mass_parity": at_mass_parity * shipped ** 2,
        "mass_term_at_reference": float(rec["loss_terms"]["mass"]),
        "ratio_to_mass_term": at_mass_parity * shipped ** 2
                              / float(rec["loss_terms"]["mass"]),
    }


def holdout(inputs):
    """Section F — re-derive `p_cal` at every genome in the promotion chain.

    Only `smoothness` reads the genome at all, so this measures exactly one thing: how
    much of the allocation is a property of the anchor rather than of the weight table.
    """
    rows = {}
    for label, name in inputs.items():
        rec = _load(name)
        sm = float(rec["loss_terms"]["smoothness"])
        p, c = R.calibrated_priorities(sm)
        rows[label] = {"file": name, "smoothness_loss": sm, "p_cal": p.as_dict(),
                       "cost_total": sum(c.values())}
    ref = rows[REFERENCE]["p_cal"]
    for label, r in rows.items():
        r["max_abs_point_shift_vs_reference"] = max(
            abs(r["p_cal"][k] - ref[k]) for k in ref)
    return rows


def build(inputs):
    ref_rec = _load(inputs[REFERENCE])
    sm = float(ref_rec["loss_terms"]["smoothness"])
    out = {
        "inputs": dict(inputs),
        "reference_genome": REFERENCE,
        "reference_smoothness_loss": sm,
        "loss_share": loss_share(ref_rec),
        "calibration": calibration(sm),
        "conservation": conservation(sm),
        "ripple": ripple_anchor(sm, ref_rec),
        "holdout": holdout(inputs),
    }
    out["checks"] = checks(out)
    out["verdict"] = verdict(out)
    return out


def checks(rec):
    cal, con = rec["calibration"], rec["conservation"]
    c = {
        "identity_at_the_calibration_point": cal["identity_max_rel_err"] <= 1e-12,
        "identity_max_rel_err": cal["identity_max_rel_err"],
        "conservation_under_reallocation": con["max_abs_err"] <= CONSERVATION_TOL,
        "conservation_max_abs_err": con["max_abs_err"],
        "barriers_never_move": con["all_barriers_unmoved"],
        "points_sum_to_100": abs(sum(cal["p_cal"].values()) - 100.0) <= 1e-9,
        # The five point axes ARE `OBJECTIVE_TERMS`, read and not retyped.
        "axes_are_objective_terms": tuple(cal["p_cal"]) == tuple(WO.OBJECTIVE_TERMS),
    }
    c["all_ok"] = bool(c["identity_at_the_calibration_point"]
                       and c["conservation_under_reallocation"]
                       and c["barriers_never_move"] and c["points_sum_to_100"]
                       and c["axes_are_objective_terms"])
    return c


def verdict(rec):
    v = {}
    p = rec["calibration"]["p_cal"]
    v["portfolio"] = {k: round(p[k], 2) for k in p}
    v["portfolio_note"] = (
        "the shipped weight table is a %s portfolio — %.0f%% on mass, %.0f%% on stroke, "
        "%.0f%% on durability, %.1f%% on print finish and nothing at all on rolling.  "
        "That is a defensible allocation and it is nothing like the 99%%-mass reading the "
        "loss breakdown invites.  It has never been stated before this line."
        % ("/".join(("%.1f" if p[k] < 1.0 else "%.0f") % p[k] for k in
                    ("mass", "deflection", "stress_margin", "smoothness", "phase_ripple")),
           p["mass"], p["deflection"], p["stress_margin"], p["smoothness"]))

    ls = rec["loss_share"]["share"]
    v["loss_share_says_mass_is"] = ls["mass"]
    v["calibration_says_mass_is"] = p["mass"] / 100.0
    v["loss_share_refuted"] = bool(abs(ls["mass"] - p["mass"] / 100.0) > 0.2)
    v["loss_share_note"] = (
        "loss share puts mass at %.2f%% and stroke at %.4f%%; the calibration puts them "
        "at %.2f%% and %.2f%%.  The gap is the whole finding: `deflection` is small in "
        "the loss because the requirement is MET (1.99742 mm against 2.0), not because "
        "it is unimportant."
        % (100 * ls["mass"], 100 * ls["deflection"], p["mass"], p["deflection"]))

    hs = [r["max_abs_point_shift_vs_reference"] for r in rec["holdout"].values()]
    v["holdout_max_point_shift"] = max(hs)
    v["calibration_survives_the_holdout"] = bool(max(hs) < 1.0)
    v["holdout_note"] = (
        "re-anchoring the calibration at each of the three genomes that shipped before "
        "the current one moves no share by more than %.3f points, because `smoothness` "
        "is the only term that reads a genome at all and it holds %.2f of the 100.  The "
        "allocation is a property of the WEIGHT TABLE, not of the anchor — which is the "
        "claim `do not calibrate on one genome` requires, measured rather than asserted."
        % (max(hs), p["smoothness"]))

    r = rec["ripple"]
    v["ripple_weight_per_point"] = r["weight_per_point"]
    v["ripple_at_mass_parity_costs"] = r["term_at_mass_parity"]
    v["ripple_note"] = (
        "one point of `rolling` buys weight %.4f.  Bought at parity with `light` (%.2f "
        "points) that is w_ripple = %.1f, and the SHIPPED wheel's own ripple of %.6f — "
        "%.1f points, not one — would then cost %.2f loss units, %.2fx the entire mass "
        "term.  The axis nobody has ever bought is not a small one."
        % (r["weight_per_point"], r["points_held_by_mass"], r["weight_at_mass_parity"],
           r["shipped_std_over_mean"], r["shipped_points_of_ripple"],
           r["term_at_mass_parity"], r["ratio_to_mass_term"]))
    return v


BAR = "=" * 78


def _wrap(text, indent, width=72):
    import textwrap
    return ("\n" + " " * indent).join(textwrap.wrap(text, width))


def _print(rec):
    cal, v, c = rec["calibration"], rec["verdict"], rec["checks"]
    print(BAR)
    print("  WHAT PORTFOLIO IS `DEFAULT_WEIGHTS` RUNNING?  —  MBSE_PLAN.md STEP 4")
    print(BAR)
    print("    reference genome %s (%s), smoothness term %.8f"
          % (rec["reference_genome"], rec["inputs"][rec["reference_genome"]],
             rec["reference_smoothness_loss"]))

    print("\n  A — THE INSTRUMENT THAT DOES NOT WORK: LOSS SHARE")
    print("      term              loss        share")
    ls = rec["loss_share"]
    for k in WO.OBJECTIVE_TERMS:
        print("      %-14s %11.6f %10.4f%%"
              % (k, ls["terms"][k], 100 * ls["share"][k]))
    print("      %-14s %11.6f            (the record's own `loss` is %.6f)"
          % ("total", ls["total"], ls["record_loss"]))

    print("\n  B — THE COST OF A 1% MISS, AND THE ALLOCATION IT IMPLIES")
    print("      term            reference deviation d_T                      d        "
          "c_T        p_cal")
    for k in WO.OBJECTIVE_TERMS:
        rd = cal["reference_deviation"][k]
        print("      %-14s %-44s %8.5f %10.6f %8.2f"
              % (k, rd["what"], rd["d"], cal["costs"][k], cal["p_cal"][k]))
    print("      %-14s %-44s %8s %10.6f %8.2f"
          % ("sum", "", "", cal["cost_total"], sum(cal["p_cal"].values())))
    print("      cost per point %.9f  (= sum(c) / 100, the whole map)"
          % cal["cost_per_point"])

    print("\n  C — THE MAP IS AN IDENTITY AT ITS OWN CALIBRATION POINT")
    print("      term            weights_from_priorities(p_cal)   DEFAULT_WEIGHTS")
    for k in WO.OBJECTIVE_TERMS:
        print("      %-14s %30.10f %17.10f"
              % (k, cal["identity_weights"][k], WO.DEFAULT_WEIGHTS[k]))
    print("      worst relative error %.3e on `%s`"
          % (cal["identity_max_rel_err"], cal["identity_worst_term"]))

    print("\n  D — CONSERVATION.  TOTAL PRESSURE UNDER %d REALLOCATIONS SUMMING TO 100"
          % len(rec["conservation"]["rows"]))
    print("      base pressure %.12f   worst deviation %.3e   barriers unmoved %s"
          % (rec["conservation"]["base_pressure"], rec["conservation"]["max_abs_err"],
             rec["conservation"]["all_barriers_unmoved"]))
    print("      the five single-axis allocations:")
    for r in rec["conservation"]["rows"][:len(WO.OBJECTIVE_TERMS)]:
        on = max(r["points"], key=lambda k: r["points"][k])
        print("        100 points on %-14s pressure %.12f   w_ripple %10.4f"
              % (on, r["pressure"], r["w_phase_ripple"]))

    print("\n  E — THE AXIS WITH NO DEFAULT WEIGHT, PRICED")
    r = rec["ripple"]
    for k in ("reference_deviation", "weight_per_point", "points_held_by_mass",
              "weight_at_mass_parity", "shipped_std_over_mean",
              "shipped_points_of_ripple", "term_at_mass_parity",
              "mass_term_at_reference", "ratio_to_mass_term"):
        print("      %-28s %14.6f" % (k, r[k]))

    print("\n  F — THE HOLD-OUT: p_cal RE-ANCHORED AT EVERY GENOME IN THE PROMOTION CHAIN")
    print("      genome    smoothness   %s   worst shift"
          % "  ".join("%9s" % k[:9] for k in WO.OBJECTIVE_TERMS))
    for label, r in rec["holdout"].items():
        print("      %-9s %10.6f   %s   %10.4f"
              % (label, r["smoothness_loss"],
                 "  ".join("%9.4f" % r["p_cal"][k] for k in WO.OBJECTIVE_TERMS),
                 r["max_abs_point_shift_vs_reference"]))

    print("\n  CHECKS")
    for k in ("identity_at_the_calibration_point", "conservation_under_reallocation",
              "barriers_never_move", "points_sum_to_100", "axes_are_objective_terms"):
        print("    %-38s %s" % (k, "ok" if c[k] else "FAIL"))

    print("\n  VERDICT")
    print("    %s" % _wrap(v["portfolio_note"], 4))
    print("    %s" % _wrap(v["loss_share_note"], 4))
    print("    %s" % _wrap(v["holdout_note"], 4))
    print("    %s" % _wrap(v["ripple_note"], 4))
    print(BAR)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for label, default in DEFAULT_INPUTS.items():
        ap.add_argument("--%s" % label, default=default)
    ap.add_argument("--out", default="study_mbse_calibration.json")
    args = ap.parse_args()

    inputs = {k: getattr(args, k) for k in DEFAULT_INPUTS}
    # It solves nothing, so the only way to weaken it is to read a different set of
    # records — and then the calibration is anchored somewhere else and the hold-out is
    # measuring a different chain.
    import _gate_guard
    _gate_guard.refuse_degraded_out(ap, args, "study_mbse_calibration.json", [
        (inputs[k] != v, "--%s %s, not the committed %s" % (k, inputs[k], v))
        for k, v in DEFAULT_INPUTS.items()])

    rec = build(inputs)
    _print(rec)
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"    wrote {args.out}")

    # GATED, unlike `study_fillet_kt`: the identity and the conservation law are claims
    # that are either true or a broken map, not valuations with no threshold to meet.
    return 0 if rec["checks"]["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
