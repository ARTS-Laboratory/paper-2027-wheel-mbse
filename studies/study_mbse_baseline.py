"""
=============================================================================
  WHAT MISSION DOES THE SHIPPED WHEEL IMPLY?  THE DERIVATIONS, RUN BACKWARDS
=============================================================================
    .venv-opt/bin/python studies/study_mbse_baseline.py        (make mbsebase)

MBSE_PLAN.md Step 0.  PLAN.md §97.

WHY THIS EXISTS
---------------
This tree has spent ninety-six numbered sections optimising a wheel and has never written
down what the wheel is FOR.  The entire mission is two bare literals:

    FORCE_LBS            = 15.0   (wheel_fea.py:151)   -> 66.7233 N
    TARGET_DEFLECTION_MM = 2.0    (wheel_fea.py:156)   -> the whole stroke requirement

Neither carries a derivation.  `MIN_WALL_MM` three lines away carries eighteen lines on
why 1.2; `FORCE_LBS` carries the word `# Loading`.  So: read the four shipped constants,
run `wheel_requirements`' derivations BACKWARDS, and report the all-up weight, sink rate,
ambient and service life they correspond to.  **Nobody currently knows, and the answer is
a finding whether it is flattering or not.**

IT SOLVES NOTHING, which is deliberate — the idiom `study_fillet_kt` established (*"reads
six committed artifacts and solves nothing"*).  Every number here is arithmetic on
constants and on one committed artifact (`best_solution.json`, for the reference
genome's own metrics).  A seventh set of solves would be a second instrument for a
comparison that has to be made on one.

WHAT IT REPORTS, AND WHICH PARTS ARE FINDINGS
----------------------------------------------
  A  THE FOUR CONSTANTS, DECODED.  Each one against the derivation that now produces it.

  B  THE MISSION FAMILY.  `FORCE_LBS = 15.0` is ONE EQUATION IN FOUR UNKNOWNS
     (`auw_kg`, `n_wheels`, `k_asym`, `sink_rate_ms`), so there is a three-parameter
     FAMILY of missions reproducing 66.7233 N and nothing on disk picks between them.
     The table prints the family so `Mission.implied_baseline`'s fiat can be checked
     rather than believed.

  C  **THE HEADLINE, AND IT IS THE STROKE.**  Across every plausible member of that
     family the implied touchdown sink rate comes out in the LOW TENTHS OF A m/s — a
     taxi-speed arrival, not a landing.  That is not a property of the vehicle, which is
     free; it is a property of the 2.0 mm STROKE, which every member shares.  The table
     runs it the other way too: what the modelled load would have to be at a real sink
     rate, and what stroke a real sink rate would need at the modelled load.

  D  THE THERMAL TABLE, WITH ITS HOLD-OUT ERRORS.  Reported so that the retention curves
     are published as what they are — a piecewise-linear INTERPOLATION between asserted
     anchors, not a fitted model.  The hold-out column is how much the table asserts
     between its own points, and at the knee it is not small.

  E  SERVICE LIFE.  `SAFETY_FACTOR = 1.6` with the life it is the factor FOR.

WHAT IT GATES (exit nonzero)
-----------------------------
Four checks, all of them about the derivation reproducing what ships:
  1  `Requirements.from_mission(Mission.implied_baseline())` reproduces `force_n`,
     `target_deflection_mm`, `allowable_stress_mpa` and `min_wall_mm` to 1e-12 relative.
  2  `e_retention(20) == 1.0` and `sigma_retention(20) == 1.0` EXACTLY, so the baseline is
     untouched by construction rather than by luck.
  3  both retention curves are monotone non-increasing on their anchor grids.
  4  an ambient above `t_max_service_c` is REFUSED, not extrapolated.

WHAT IT DOES NOT DO
--------------------
It does not re-score the wheel, propose a weight, or touch `wheel_objective`.  The
calibration is `study_mbse_calibration.py` and the compliance table is
`study_mbse_score.py`.

**AND IT DOES NOT PRESENT THE THERMAL KNOCKDOWN AS A LIFE MODEL.**  Quasi-static only —
see `wheel_requirements.THERMAL_SCOPE_NOTE`, which is printed with section D every time
rather than left in a docstring.
"""

import argparse
import json
import os

import numpy as np

import project_paths as PP
import _gate_guard

import wheel_fea as W
import wheel_requirements as R

HERE = os.path.dirname(os.path.abspath(__file__))

# The family sweep's grid.  Small on purpose: the point is that the family is WIDE and the
# conclusion is the same everywhere in it, which a handful of decades shows better than a
# fine mesh of a thing nothing measures.
AUW_KG = (1.0, 2.0, 3.0, 5.0, 8.0)
N_WHEELS = (3, 4)
K_ASYM = (1.0, 1.5, 2.0)

# Sink rates to run the derivation FORWARD at, in m/s.  0.26 is the implied baseline's own
# and is inserted by the driver.  The rest bracket what small-UAV landing gear is normally
# designed to; FAR 23 / MIL-A-8862 use 7-10 ft/s (2.13-3.05 m/s) for crewed aircraft, and
# those are quoted here as CONTEXT for the scale, not as a requirement on a UAV.
SINK_RATES_MS = (0.5, 1.0, 1.5, 2.0, 3.0)

AMBIENTS_C = (-20.0, 0.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0)
LANDINGS = (100, 1000, 10000, 100000)

TOL_REL = 1e-12


def _rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


def constants_decoded(mission, req):
    """Section A — each shipped constant beside the derivation that now produces it."""
    return {
        "force_n": {"shipped": float(W.TOTAL_FORCE_NEWTONS), "derived": req.force_n,
                    "rel_err": _rel(req.force_n, W.TOTAL_FORCE_NEWTONS),
                    "from": "auw_kg * g / n_wheels * n_land * k_asym"},
        "target_deflection_mm": {
            "shipped": float(W.TARGET_DEFLECTION_MM),
            "derived": req.target_deflection_mm,
            "rel_err": _rel(req.target_deflection_mm, W.TARGET_DEFLECTION_MM),
            "from": "max(STROKE_FLOOR_MM, STROKE_BY_FIELD_CLASS[field_class])"},
        "allowable_stress_mpa": {
            "shipped": float(W.ALLOWABLE_STRESS_MPA),
            "derived": req.allowable_stress_mpa,
            "rel_err": _rel(req.allowable_stress_mpa, W.ALLOWABLE_STRESS_MPA),
            "from": "sigma_ult(20C) * sigma_retention(T) * fff_knockdown / SF(landings)"},
        "min_wall_mm": {
            "shipped": float(W.MIN_WALL_MM), "derived": req.min_wall_mm,
            "rel_err": _rel(req.min_wall_mm, W.MIN_WALL_MM),
            "from": "nozzle_mm * perimeters"},
    }


def mission_family(base):
    """Section B — every vehicle that reproduces 66.7233 N, and the sink rate it needs.

    One row per `(auw_kg, n_wheels, k_asym)`; `sink_rate_ms` is SOLVED, because that is
    the free parameter once the vehicle is named.  A row whose static reaction times
    `k_asym` already exceeds the service load has NO solution — the load factor would
    have to be below 1 — and is reported as such rather than dropped, because "this
    vehicle is already too heavy for the modelled load standing still" is itself a
    reading of the constant.
    """
    rows = []
    for auw in AUW_KG:
        for n in N_WHEELS:
            for k in K_ASYM:
                m = base.__class__(**{**_mission_fields(base), "auw_kg": auw,
                                      "n_wheels": n, "k_asym": k, "sink_rate_ms": 0.0})
                row = {"auw_kg": auw, "n_wheels": n, "k_asym": k,
                       "static_force_n": m.static_force_n,
                       "gear_mass_frac": None}
                try:
                    v = m.sink_rate_for_force(W.TOTAL_FORCE_NEWTONS)
                    m = base.__class__(**{**_mission_fields(m), "sink_rate_ms": v})
                    row.update({"sink_rate_ms": v,
                                "landing_load_factor": m.landing_load_factor,
                                "admissible": True})
                except ValueError as exc:
                    row.update({"sink_rate_ms": None, "landing_load_factor": None,
                                "admissible": False, "why": str(exc).split(" — ")[0]})
                rows.append(row)
    return rows


def _mission_fields(m):
    return {"auw_kg": m.auw_kg, "n_wheels": m.n_wheels, "k_asym": m.k_asym,
            "sink_rate_ms": m.sink_rate_ms, "field_class": m.field_class,
            "ambient_c": m.ambient_c, "landings": m.landings,
            "nozzle_mm": m.nozzle_mm, "perimeters": m.perimeters,
            "material": m.material}


def stroke_against_sink_rate(base):
    """Section C — the same trade seen from both ends, at the baseline vehicle.

    `force_at_sink_rate` holds the 2.0 mm stroke and asks what the load becomes.
    `stroke_for_force` holds the 66.7233 N load and asks what stroke would be needed.
    They are the same equation and neither is more true than the other; printing both is
    what makes the size of the finding legible, because one of them comes out as a
    multiple and the other as a length that can be compared to the wheel.
    """
    n_land_shipped = base.landing_load_factor
    rows = []
    for v in sorted(set(SINK_RATES_MS) | {round(base.sink_rate_ms, 6)}):
        m = base.__class__(**{**_mission_fields(base), "sink_rate_ms": v})
        # Holding the load fixed instead: s_eff = v^2 / (2 g (n_land - 1)).
        stroke_mm = (None if n_land_shipped <= 1.0 else
                     v ** 2 / (2.0 * R.G_MS2 * (n_land_shipped - 1.0))
                     * 1000.0 / R.STROKE_EFFICIENCY)
        rows.append({
            "sink_rate_ms": v,
            "landing_load_factor": m.landing_load_factor,
            "force_n_at_shipped_stroke": m.force_n,
            "force_over_shipped": m.force_n / float(W.TOTAL_FORCE_NEWTONS),
            "stroke_mm_at_shipped_force": stroke_mm,
            "stroke_over_wheel_radius": (None if stroke_mm is None
                                         else stroke_mm / W.RIM_RADIUS_MM),
        })
    return rows


def thermal_table(card):
    """Section D — retention at each ambient, plus the hold-out error at every INTERIOR
    anchor.

    THE HOLD-OUT IS WHAT DECIDES HOW THIS TABLE MAY BE DESCRIBED.  MBSE_PLAN Step 2 asks
    for an anchor predicted within a stated band *"or the curve is reported as an
    interpolation between anchors and NOT as a model.  Say which."*  This says which, with
    the number: each interior anchor is dropped, the two that bracket it are joined, and
    the error at the dropped point is reported.  It is 7-9% through the knee, so the
    curves are published as an INTERPOLATION and nothing here is called a model.
    """
    out = {"anchors": {}, "holdout": {}, "at_ambient": []}
    for what, anchors, fn in (("e", card.e_retention_anchors, card.e_retention),
                              ("sigma", card.sigma_retention_anchors,
                               card.sigma_retention)):
        out["anchors"][what] = [[float(a), float(b)] for a, b in anchors]
        xs = np.array([a[0] for a in anchors], dtype=float)
        ys = np.array([a[1] for a in anchors], dtype=float)
        rows = []
        for i in range(1, len(xs) - 1):
            keep = np.delete(np.arange(len(xs)), i)
            pred = float(np.interp(xs[i], xs[keep], ys[keep]))
            rows.append({"t_c": float(xs[i]), "actual": float(ys[i]), "predicted": pred,
                         "abs_err": abs(pred - ys[i]),
                         "rel_err": abs(pred - ys[i]) / ys[i]})
        out["holdout"][what] = rows
        out["holdout"][what + "_max_rel_err"] = max(r["rel_err"] for r in rows)
    for t in AMBIENTS_C:
        out["at_ambient"].append({
            "ambient_c": float(t),
            "e_retention": card.e_retention(t),
            "sigma_retention": card.sigma_retention(t),
            "e_mpa": card.e_mpa(t),
            "allowable_stress_mpa": card.allowable_stress_mpa(t, R.REFERENCE_LANDINGS),
        })
    out["scope_note"] = R.THERMAL_SCOPE_NOTE
    return out


def life_table(card):
    """Section E — what a service life costs in allowable stress."""
    return [{"landings": n, "fatigue_knockdown": R.fatigue_knockdown(n),
             "safety_factor": R.safety_factor(n),
             "allowable_stress_mpa": card.allowable_stress_mpa(20.0, n)}
            for n in LANDINGS]


def build(genome_path):
    base = R.Mission.implied_baseline()
    req = R.Requirements.from_mission(base)
    shipped = R.Requirements.baseline()
    with open(genome_path) as fh:
        rec = json.load(fh)

    out = {
        "genome_file": os.path.basename(genome_path),
        "implied_baseline_mission": base.as_dict(),
        "implied_baseline_derived": {
            "weight_n": base.weight_n, "static_force_n": base.static_force_n,
            "stroke_mm": base.stroke_mm, "effective_stroke_m": base.effective_stroke_m,
            "landing_load_factor": base.landing_load_factor, "force_n": base.force_n,
            "safety_factor": base.safety_factor, "min_wall_mm": base.min_wall_mm},
        "constants": constants_decoded(base, req),
        "family": mission_family(base),
        "stroke_trade": stroke_against_sink_rate(base),
        "thermal": thermal_table(base.material),
        "life": life_table(base.material),
        "req_hash": {"shipped": shipped.req_hash(), "derived": req.req_hash()},
        # The reference genome's own numbers, for the reader who wants the design beside
        # the requirement.  Read, never recomputed.
        "reference_genome": {
            "axle_drop_mean_mm": rec["metrics"]["axle_drop_mean_mm"],
            "stress_utilisation_hub": rec["metrics"]["stress_utilisation_hub"],
            "mesh_mass_g": rec["metrics"]["mesh_mass_g"],
            "phase_ripple_std_over_mean": rec["metrics"]["phase_ripple_std_over_mean"]},
    }
    out["checks"] = checks(base, req, shipped)
    out["verdict"] = verdict(out)
    return out


def checks(base, req, shipped):
    """The four gated checks.  Each one is a boolean this file computed."""
    card = base.material
    c = {}
    c["derivation_reproduces_constants"] = {
        k: {"rel_err": v["rel_err"], "ok": v["rel_err"] <= TOL_REL}
        for k, v in constants_decoded(base, req).items()}
    c["derivation_reproduces_constants_ok"] = all(
        v["ok"] for v in c["derivation_reproduces_constants"].values())

    c["retention_is_exactly_one_at_20c"] = bool(
        card.e_retention(20.0) == 1.0 and card.sigma_retention(20.0) == 1.0)

    mono = True
    for anchors in (card.e_retention_anchors, card.sigma_retention_anchors):
        ys = [a[1] for a in anchors]
        mono = mono and all(b <= a for a, b in zip(ys, ys[1:]))
    c["retention_is_monotone_non_increasing"] = bool(mono)

    try:
        card.e_retention(card.t_max_service_c + 0.1)
        c["refuses_above_t_max_service"] = False
    except ValueError:
        c["refuses_above_t_max_service"] = True

    # The weight table is untouched by a baseline derivation — the arc's own first rule.
    c["baseline_weights_unmoved"] = bool(req.weights == shipped.weights)
    c["all_ok"] = bool(c["derivation_reproduces_constants_ok"]
                       and c["retention_is_exactly_one_at_20c"]
                       and c["retention_is_monotone_non_increasing"]
                       and c["refuses_above_t_max_service"]
                       and c["baseline_weights_unmoved"])
    return c


def verdict(rec):
    """The findings, as booleans and numbers this file computed rather than as sentences."""
    v = {}
    ok = [r for r in rec["family"] if r["admissible"]]
    v["family_size"] = len(rec["family"])
    v["family_admissible"] = len(ok)
    v["implied_sink_rate_ms_min"] = min(r["sink_rate_ms"] for r in ok)
    v["implied_sink_rate_ms_max"] = max(r["sink_rate_ms"] for r in ok)
    v["every_admissible_member_is_under_1_ms"] = all(r["sink_rate_ms"] < 1.0 for r in ok)
    v["sink_rate_note"] = (
        "across %d admissible members of the mission family the implied touchdown sink "
        "rate spans %.4f to %.4f m/s.  That is a taxi-speed arrival, not a landing, and "
        "it is a property of the 2.0 mm STROKE — which every member shares — and not of "
        "the vehicle, which is free."
        % (len(ok), v["implied_sink_rate_ms_min"], v["implied_sink_rate_ms_max"]))

    one = [r for r in rec["stroke_trade"] if abs(r["sink_rate_ms"] - 1.0) < 1e-12]
    if one:
        r = one[0]
        v["force_multiple_at_1_ms"] = r["force_over_shipped"]
        v["stroke_mm_needed_at_1_ms"] = r["stroke_mm_at_shipped_force"]
        v["stroke_fraction_of_wheel_radius_at_1_ms"] = r["stroke_over_wheel_radius"]
        v["stroke_note"] = (
            "at a 1.0 m/s touchdown the same vehicle needs %.1fx the modelled load over "
            "the shipped 2.0 mm stroke, or %.1f mm of stroke — %.0f%% of the wheel's own "
            "%.1f mm radius — to hold the modelled load.  Neither is available."
            % (r["force_over_shipped"], r["stroke_mm_at_shipped_force"],
               100.0 * r["stroke_over_wheel_radius"], W.RIM_RADIUS_MM))

    v["thermal_is_an_interpolation_not_a_model"] = True
    v["thermal_holdout_max_rel_err"] = max(
        rec["thermal"]["holdout"]["e_max_rel_err"],
        rec["thermal"]["holdout"]["sigma_max_rel_err"])
    v["thermal_note"] = (
        "worst hold-out error over the interior anchors is %.1f%%, so the retention "
        "curves are published as a piecewise-linear INTERPOLATION between asserted "
        "anchors and not as a model.  %s"
        % (100.0 * v["thermal_holdout_max_rel_err"], R.THERMAL_SCOPE_NOTE))

    hot = [r for r in rec["thermal"]["at_ambient"] if r["ambient_c"] == 40.0][0]
    v["allowable_at_40c_over_20c"] = (hot["allowable_stress_mpa"]
                                      / float(W.ALLOWABLE_STRESS_MPA))
    v["shipped_utilisation_at_40c"] = (rec["reference_genome"]["stress_utilisation_hub"]
                                       / v["allowable_at_40c_over_20c"])
    v["shipped_breaches_the_allowable_at_40c"] = bool(
        v["shipped_utilisation_at_40c"] > 1.0)
    v["temperature_note"] = (
        "the shipped wheel sits at util %.4f against a knee of 0.80 and has essentially "
        "no headroom: at 40 C the allowable falls to %.1f%% of its 20 C value, which puts "
        "the SAME stress at util %.4f before the softer modulus is allowed to move the "
        "stress at all.  This is a knockdown arithmetic, NOT a re-solve — "
        "study_mbse_score.py is the one that re-solves."
        % (rec["reference_genome"]["stress_utilisation_hub"],
           100.0 * v["allowable_at_40c_over_20c"], v["shipped_utilisation_at_40c"]))
    return v


# ---------------------------------------------------------------------------
# PRINT
# ---------------------------------------------------------------------------

BAR = "=" * 78


def _print(rec):
    v, c = rec["verdict"], rec["checks"]
    m = rec["implied_baseline_mission"]
    d = rec["implied_baseline_derived"]
    print(BAR)
    print("  WHAT MISSION DOES THE SHIPPED WHEEL IMPLY?  —  MBSE_PLAN.md STEP 0")
    print(BAR)

    print("\n  A — THE FOUR CONSTANTS, DECODED")
    print("      constant                shipped        derived      rel err   from")
    for k, r in rec["constants"].items():
        print("      %-22s %12.6f %12.6f  %10.2e  %s"
              % (k, r["shipped"], r["derived"], r["rel_err"], r["from"]))
    print("\n      the mission it decoded to (see `Mission.implied_baseline` — THE VEHICLE")
    print("      IS A FIAT, THE SINK RATE IS SOLVED):")
    print("        auw %.3f kg on %d wheels, k_asym %.2f, field %s, ambient %.1f C, "
          "%d landings" % (m["auw_kg"], m["n_wheels"], m["k_asym"], m["field_class"],
                           m["ambient_c"], m["landings"]))
    print("        nozzle %.2f mm x %d perimeters -> min wall %.4f mm"
          % (m["nozzle_mm"], m["perimeters"], d["min_wall_mm"]))
    print("        weight %.4f N -> static %.4f N/wheel -> n_land %.4f -> %.4f N"
          % (d["weight_n"], d["static_force_n"], d["landing_load_factor"], d["force_n"]))
    print("        stroke %.3f mm at %.2f efficiency -> s_eff %.6f m"
          % (d["stroke_mm"], R.STROKE_EFFICIENCY, d["effective_stroke_m"]))
    print("        SOLVED SINK RATE  %.6f m/s" % m["sink_rate_ms"])

    print("\n  B — THE MISSION FAMILY.  ONE EQUATION, FOUR UNKNOWNS: every vehicle below")
    print("      reproduces 66.7233 N, and nothing on disk picks between them.")
    print("      auw kg  wheels  k_asym   static N   n_land    sink m/s")
    for r in rec["family"]:
        if r["admissible"]:
            print("      %6.1f  %6d  %6.2f  %9.4f %8.3f    %8.4f"
                  % (r["auw_kg"], r["n_wheels"], r["k_asym"], r["static_force_n"],
                     r["landing_load_factor"], r["sink_rate_ms"]))
        else:
            print("      %6.1f  %6d  %6.2f  %9.4f       —           —   (already over "
                  "the service load standing still)"
                  % (r["auw_kg"], r["n_wheels"], r["k_asym"], r["static_force_n"]))

    print("\n  C — THE HEADLINE.  THE SAME TRADE FROM BOTH ENDS, AT THE BASELINE VEHICLE.")
    print("      sink m/s   n_land    force N at 2.0 mm    xshipped    stroke mm at "
          "66.72 N   x wheel R")
    for r in rec["stroke_trade"]:
        print("      %8.4f %9.3f %19.3f %11.2f %21.3f %11.3f"
              % (r["sink_rate_ms"], r["landing_load_factor"],
                 r["force_n_at_shipped_stroke"], r["force_over_shipped"],
                 r["stroke_mm_at_shipped_force"], r["stroke_over_wheel_radius"]))

    print("\n  D — THE THERMAL TABLE.  AN INTERPOLATION, NOT A MODEL — SEE THE HOLD-OUT.")
    print("      ambient C   E retention   sigma retention      E MPa    allowable MPa")
    for r in rec["thermal"]["at_ambient"]:
        print("      %9.1f %13.4f %17.4f %10.2f %16.4f"
              % (r["ambient_c"], r["e_retention"], r["sigma_retention"], r["e_mpa"],
                 r["allowable_stress_mpa"]))
    print("      hold-out at each INTERIOR anchor (drop it, join its neighbours):")
    print("        curve   T C    actual  predicted   rel err")
    for what in ("e", "sigma"):
        for r in rec["thermal"]["holdout"][what]:
            print("        %-6s %5.1f %9.4f %10.4f %9.2f%%"
                  % (what, r["t_c"], r["actual"], r["predicted"], 100.0 * r["rel_err"]))
    print("      SCOPE: %s" % _wrap(rec["thermal"]["scope_note"], 14))

    print("\n  E — SERVICE LIFE")
    print("      landings   k_fatigue   safety factor   allowable MPa")
    for r in rec["life"]:
        print("      %8d %11.4f %15.4f %15.4f"
              % (r["landings"], r["fatigue_knockdown"], r["safety_factor"],
                 r["allowable_stress_mpa"]))

    print("\n  CHECKS")
    for k, r in c["derivation_reproduces_constants"].items():
        print("    %-24s rel err %.2e   %s"
              % (k, r["rel_err"], "ok" if r["ok"] else "FAIL"))
    for k in ("retention_is_exactly_one_at_20c", "retention_is_monotone_non_increasing",
              "refuses_above_t_max_service", "baseline_weights_unmoved"):
        print("    %-40s %s" % (k, "ok" if c[k] else "FAIL"))

    print("\n  VERDICT")
    print("    every admissible mission is under 1 m/s : %s"
          % v["every_admissible_member_is_under_1_ms"])
    print("      %s" % _wrap(v["sink_rate_note"], 6))
    if "stroke_note" in v:
        print("    %s" % _wrap(v["stroke_note"], 6))
    print("    the thermal curves are an interpolation : %s"
          % v["thermal_is_an_interpolation_not_a_model"])
    print("      %s" % _wrap(v["thermal_note"], 6))
    print("    the shipped wheel breaches at 40 C      : %s"
          % v["shipped_breaches_the_allowable_at_40c"])
    print("      %s" % _wrap(v["temperature_note"], 6))
    print(BAR)


def _wrap(text, indent, width=72):
    import textwrap
    pad = " " * indent
    return ("\n" + pad).join(textwrap.wrap(text, width))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genome", default=PP.BEST_SOLUTION)
    ap.add_argument("--out", default="study_mbse_baseline.json")
    args = ap.parse_args()

    # This driver solves nothing, so it has no fidelity to lower; the only way to weaken
    # it is to point it at a different reference genome, and then its `reference_genome`
    # block describes a wheel that is not the one that ships.
    _gate_guard.refuse_degraded_out(ap, args, "study_mbse_baseline.json", [
        (os.path.abspath(args.genome) != PP.BEST_SOLUTION,
         "--genome %s, not the shipped best_solution.json" % args.genome)])

    rec = build(args.genome)
    _print(rec)
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"    wrote {args.out}")

    # THIS ONE DOES GATE, unlike `study_fillet_kt`, and the difference is what is being
    # claimed: a valuation has no threshold to meet, but "the derivation reproduces the
    # constant that ships" is a claim that is either true or a broken build.
    return 0 if rec["checks"]["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
