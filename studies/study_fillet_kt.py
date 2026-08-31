"""
=============================================================================
  WHAT `Kt * agg` IS WORTH ON A FILLETED MESH — THE DOUBLE COUNT, PRICED,
  AGAINST THE FILLET SURFACE THE MESH NOW RESOLVES
=============================================================================
    .venv-opt/bin/python studies/study_fillet_kt.py             (make filletkt)

PLAN.md §93's Condition B.  FILLET_PLAN.md STEP 3.

WHY THIS EXISTS
---------------
§93 took the decision to wire the fillet into the objective and put two conditions in
front of it.  Condition B was found while §93 was being written, by reading
`wheel_objective.py:1234` rather than the plan files:

    util_j = kt * agg / ALLOWABLE_STRESS_MPA

`agg` is the p-norm of the MESH's own stress field; `kt` is `stress_concentration_kt`, a
closed-form surrogate whose whole purpose is to stand in for a fillet the mesh does not
model.  On a filleted mesh the mesh models it.  §93's own check reads:

    "a written derivation of what replaces `kt * agg` when the fillet is meshed, and a
     measurement of what the current double count is worth at the shipped genome and at
     `b029622`."

This driver is the measurement half.  It reads six committed artifacts and solves nothing,
which is deliberate: every number it needs was already measured, and a seventh set of
solves would be a second instrument for a comparison that has to be made on one.

WHAT IT READS, AND WHY EACH ONE
--------------------------------
  `study_corner_singularity[_fillet][_b029622].json`  — four ladders, two genomes x two
      meshes.  These carry the CORNER CENSUS (which corners are re-entrant, and their
      Williams exponents) and the DIVERGENCE verdicts, including `hub:surface` and
      `rim:surface`: the peak von Mises inside a fixed analytic tube about the fillet arc,
      which is the measured quantity `Kt * agg` is a model OF.  `arc_peak`'s own docstring
      is why the tube is defined analytically — "a tube round a polyline of the arc's own
      nodes would be a slightly different region at every rung, which is the one thing a
      convergence measurement may not have."

  `study_fillet_terms[_b029622].json`  — two genomes x two meshes x two kinematics, eight
      phases, `coarse`, and the report block that carries `pnorm_stress_agg_mpa`,
      `kt_hub`, `kt_rim` and the two utilisations.  This is the objective's OWN reading.

THE THREE CLAIMS THE CONSTRUCTION RESTS ON, AND WHICH ONE THE FILLET MOVES
---------------------------------------------------------------------------
`wheel_objective.py`'s module docstring and the comment at :1160 state them:

  C1  The mesh's true peak DIVERGES under refinement, so it cannot be the constraint.
      (M4, `test_peak_stress_diverges_but_the_field_converges`; M8b-i.6 measured the
      p-norm converging for p <= 6 and `max/pnorm` only for p >= 24, and their product at
      no exponent at either design.)
  C2  `agg` at `STRESS_NOMINAL_P = 4.0` IS mesh-convergent — order 1.76 / 2.18, GCI
      0.45% / 0.20%.
  C3  `Kt(R, t)` maps that nominal up to the peak.

**C1 IS THE ONE §93 SAID THE FILLET MOVES, AND IT DOES NOT.**  §93 wrote "on a filleted
mesh the peak is finite and convergent (§52)" and cited §52's 36.8 / 16.1 MPa.  Those are
§52's FILLET-SURFACE numbers — a local tube probe — and §52's own headline says the
opposite about the wheel: *"ITS HEADLINE IS STILL BLOCKED ON `rim:P_c`"*, the global
maximum sitting on the end cap's corner at every rung from `coarse` up with successive
differences holding a ratio of +1.264.  §93 read the surviving half of §52 and wrote the
conclusion of the half that failed.  This driver reports the filleted `global_max_vm`
ladder at both genomes so the claim is checkable rather than cited.

C3 IS WHERE THE DOUBLE COUNT ACTUALLY LIVES, AND IT IS ONE CORNER PER JUNCTION
-------------------------------------------------------------------------------
`stress_concentration_kt` is Peterson's stepped-beam factor for a FILLETED step.  What it
models is the raiser at `P_t` — the part's own junction corner.  On the unfilleted mesh
`P_t` is re-entrant and singular and `Kt` supplies what the mesh lacks.  On the filleted
mesh `P_t` is a point in the material's INTERIOR at 360 deg and its probe settles.  So
`Kt` prices a feature the mesh now resolves, at both junctions, and that is the double
count — precisely located, and narrower than "the stress term is wrong".

`P_c` is untouched by either `Kt` and by the fillet: still re-entrant on the filleted mesh
at both genomes, still divergent, and it is the END CAP's corner, which the exported solid
does not have (§52; `wheel_wheel.py`'s module docstring prices the cap under WHAT IS AND
IS NOT MODELLED).  No `Kt` was ever meant to model it and none does.

AND THE DIRECTION IS THE OTHER WAY FROM §93's, WHICH IS WHY THIS IS NOT BOOKKEEPING
-------------------------------------------------------------------------------------
§93 argued the error is conservative because `Kt > 1` always.  That is an argument about
one factor of a product.  The other factor, `agg`, is a volume-weighted p-norm at p = 4
over the WHOLE wheel — hub, rim and thick spoke roots, most of the volume and almost none
of the stress — and the thing it stands in for is a local peak.  So the two errors point
opposite ways and neither is settled by inspection.  This driver measures the product
against the fillet surface's own converged peak, which is the first time the two have been
put on the same axis, because on an unfilleted mesh the measured side does not exist.

WHAT IT DOES NOT DO
--------------------
It does not change `wheel_objective`, propose a weight, or run the replacement.  §93's
step 2 asks what the current term is worth and what should replace it; picking the
exponent of a region-restricted p-norm is a sweep on filleted solves, the way M8b-i.6
swept the global one, and it is named in the record rather than guessed at here.

SCOPE, WHICH DIFFERS BETWEEN THE TWO SIDES OF THE COMPARISON AND IS REPORTED WITH IT
--------------------------------------------------------------------------------------
The surface peaks are LINEAR, ONE PHASE, on the ladder's own solves.  `agg` is EIGHT
phases aggregated at `stress_phase_p = 8`, which is >= any single phase's p-norm, and is
reported under both kernels.  **Both scope gaps run the same way** — SVK and eight phases
each raise the measured peak and leave the model where it is — so a modelled peak that
comes out below the measured one is a LOWER BOUND on the gap, not an estimate of it.
"""

import argparse
import json
import os

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import _gate_guard

import jax_config  # noqa: F401
import wheel_objective as WO

HERE = os.path.dirname(os.path.abspath(__file__))

# The six committed artifacts, keyed `(genome_label, what)`.  `shipped` is
# `best_solution.json`; `b029622` is the design the FILLETED objective descends to from it
# (§92's treatment endpoint), filed at the repository root as `fillet_optimum_b029622.json`
# so it can be measured by name.
DEFAULT_INPUTS = {
    ("shipped", "corner_unfilleted"): "study_corner_singularity.json",
    ("shipped", "corner_filleted"): "study_corner_singularity_fillet.json",
    ("shipped", "terms"): "study_fillet_terms.json",
    ("b029622", "corner_unfilleted"): "study_corner_singularity_b029622.json",
    ("b029622", "corner_filleted"): "study_corner_singularity_fillet_b029622.json",
    ("b029622", "terms"): "study_fillet_terms_b029622.json",
}

GENOMES = ("shipped", "b029622")
JUNCTIONS = ("hub", "rim")
KINEMATICS = ("linear", "svk")


def _load(name):
    with open(os.path.join(HERE, name)) as fh:
        return json.load(fh)


def corner_census(unf, fil):
    """The four reference corners before and after, plus what the fillet did to each.

    The census is the whole reason `Kt` can be said to double count at all: `Kt` models a
    fillet, and the question is whether the mesh now has one.  `resolved` is the corner
    going from re-entrant to interior — not "less re-entrant", which would be a matter of
    degree, but the singularity ceasing to exist.
    """
    rows = {}
    for probe in ("hub:P_t", "hub:P_c", "rim:P_t", "rim:P_c"):
        a, b = unf["williams"][probe], fil["williams"][probe]
        rows[probe] = {
            "unfilleted": {"wedge_deg": a["wedge_deg"], "kind": a["kind"],
                           "lambda": a.get("lambda")},
            "filleted": {"wedge_deg": b["wedge_deg"], "kind": b["kind"],
                         "lambda": b.get("lambda")},
            "resolved_by_the_fillet": bool(a["re_entrant"] and not b["re_entrant"]),
        }
    return rows


def divergence_row(rec, probe):
    d = rec["divergence"].get(probe)
    if d is None:
        return None
    return {"peak_mpa": d["peak_mpa"], "increment_ratios": d["increment_ratios"],
            "diverges": d["diverges"], "settled_estimate_mpa": d["settled_estimate_mpa"]}


def objective_reading(terms, kin):
    """The objective's own numbers on each mesh, and the `Kt = 1` counterfactual beside them.

    The counterfactual is arithmetic on the SAME solves and not a second run: `agg` does
    not depend on `Kt`, so dropping the factor is exact rather than a re-evaluation.  Both
    barriers are recomputed through `wheel_objective.soft_barrier` and the artifact's own
    `weights` block — the same function and the same scales the run used, so a difference
    here is the factor and nothing else.
    """
    w = terms["weights"]
    out = {}
    for arm in ("unfilleted", "filleted"):
        r = terms["evaluation"][kin][arm]["report"]
        agg = r["pnorm_stress_agg_mpa"]
        row = {"pnorm_stress_agg_mpa": agg, "max_stress_mpa": r["max_stress_mpa"],
               "junctions": {}}
        for j in JUNCTIONS:
            kt = r["kt_%s" % j]
            util = kt * agg / WO.ALLOWABLE_STRESS_MPA
            util1 = agg / WO.ALLOWABLE_STRESS_MPA
            row["junctions"][j] = {
                "kt": kt,
                "util_as_computed": util,
                "util_with_kt_1": util1,
                "util_overstated_by": util - util1,
                "stress_as_computed":
                    float(WO.soft_barrier(util - 1.0, w["stress"])),
                "stress_with_kt_1":
                    float(WO.soft_barrier(util1 - 1.0, w["stress"])),
                "stress_margin_as_computed":
                    float(WO.soft_barrier(util - WO.MARGIN_KNEE_UTIL,
                                          w["stress_margin"])),
                "stress_margin_with_kt_1":
                    float(WO.soft_barrier(util1 - WO.MARGIN_KNEE_UTIL,
                                          w["stress_margin"])),
            }
        loss = sum(row["junctions"][j]["stress_as_computed"]
                   + row["junctions"][j]["stress_margin_as_computed"] for j in JUNCTIONS)
        loss1 = sum(row["junctions"][j]["stress_with_kt_1"]
                    + row["junctions"][j]["stress_margin_with_kt_1"] for j in JUNCTIONS)
        row["stress_loss_as_computed"] = loss
        row["stress_loss_with_kt_1"] = loss1
        # `kt_worth_in_loss` and not `double_count_worth`, because the name has to be true
        # on BOTH arms: on the unfilleted mesh `Kt` is not a double count, it is the factor
        # doing the job it was written for.  On the filleted arm this same number IS the
        # double count, and the print below says so per row rather than in the column head.
        row["kt_worth_in_loss"] = loss - loss1
        out[arm] = row
    return out


def modelled_against_measured(terms, corner_fil, kin):
    """`Kt_j * agg` on the filleted mesh against that junction's own converged surface peak.

    THE COMPARISON THE CONSTRUCTION CLAIMS TO MAKE.  `Kt * nominal ~ peak` is what
    `wheel_objective`'s docstring says the term is — *"the peak is modelled instead of
    measured"* — so the two sides here are the modelled peak and the measured one, in MPa,
    at the same junction on the same mesh.  It cannot be run on an unfilleted mesh at all,
    which is why it has never been run: there is no converged measured side to compare to.

    `settled_estimate_mpa` and not the `fine` rung, because the surface probe is still
    rising there — the ladder's own geometric tail is the number the divergence block
    reports and refusing it in favour of a rung would understate the measured side.
    """
    r = terms["evaluation"][kin]["filleted"]["report"]
    agg = r["pnorm_stress_agg_mpa"]
    out = {}
    for j in JUNCTIONS:
        d = corner_fil["divergence"].get("%s:surface" % j)
        modelled = r["kt_%s" % j] * agg
        measured = None if d is None else d["settled_estimate_mpa"]
        out[j] = {
            "modelled_peak_mpa": modelled,
            "measured_surface_peak_mpa": measured,
            "measured_over_modelled": None if not measured else measured / modelled,
            "util_modelled": modelled / WO.ALLOWABLE_STRESS_MPA,
            "util_measured": None if measured is None
                             else measured / WO.ALLOWABLE_STRESS_MPA,
            "model_understates": None if measured is None else measured > modelled,
        }
    return out


def build(inputs):
    rec = {
        "allowable_stress_mpa": float(WO.ALLOWABLE_STRESS_MPA),
        "margin_knee_util": float(WO.MARGIN_KNEE_UTIL),
        "stress_nominal_p": float(WO.STRESS_NOMINAL_P),
        "inputs": {"%s/%s" % k: v for k, v in inputs.items()},
        "genomes": {},
    }
    for g in GENOMES:
        unf = _load(inputs[(g, "corner_unfilleted")])
        fil = _load(inputs[(g, "corner_filleted")])
        terms = _load(inputs[(g, "terms")])
        rec["genomes"][g] = {
            "genome_file": terms["genome"],
            "corner_census": corner_census(unf, fil),
            "global_max_vm": {
                "unfilleted": divergence_row(unf, "global_max_vm"),
                "filleted": divergence_row(fil, "global_max_vm"),
            },
            "fillet_surface": {j: divergence_row(fil, "%s:surface" % j)
                               for j in JUNCTIONS},
            "objective": {k: objective_reading(terms, k) for k in KINEMATICS
                          if k in terms["evaluation"]},
            "modelled_against_measured": {
                k: modelled_against_measured(terms, fil, k) for k in KINEMATICS
                if k in terms["evaluation"]},
        }
    rec["verdict"] = verdict(rec)
    return rec


def verdict(rec):
    """Four named findings, each a boolean this file computed rather than a sentence.

    They are stated so a re-run can contradict them.  `premise_survives` going FALSE would
    mean the filleted peak had stopped diverging and the construction really did need
    replacing for §93's stated reason; `model_understates_everywhere` going FALSE would
    mean §93's "the direction is conservative" was right after all.
    """
    v = {}
    div = [rec["genomes"][g]["global_max_vm"]["filleted"]["diverges"] for g in GENOMES]
    v["premise_survives"] = all(div)
    v["premise_survives_note"] = (
        "the FILLETED mesh's global max still diverges at %d of %d genomes, so C1 — the "
        "reason `Kt * nominal` exists — is unchanged by the fillet"
        % (sum(div), len(div)))

    resolved = {g: [p for p, r in rec["genomes"][g]["corner_census"].items()
                    if r["resolved_by_the_fillet"]] for g in GENOMES}
    v["corners_resolved_by_the_fillet"] = resolved
    v["kt_prices_a_corner_the_mesh_resolves"] = all(
        set(resolved[g]) == {"hub:P_t", "rim:P_t"} for g in GENOMES)

    still = {g: [p for p, r in rec["genomes"][g]["corner_census"].items()
                 if r["filleted"]["kind"] == "re_entrant"] for g in GENOMES}
    v["corners_still_singular_on_the_filleted_mesh"] = still

    ratios, unders = [], []
    for g in GENOMES:
        for k, byj in rec["genomes"][g]["modelled_against_measured"].items():
            for j, row in byj.items():
                if row["measured_over_modelled"] is not None:
                    ratios.append(row["measured_over_modelled"])
                    unders.append(bool(row["model_understates"]))
    v["model_understates_everywhere"] = bool(unders) and all(unders)
    v["measured_over_modelled_min"] = min(ratios) if ratios else None
    v["measured_over_modelled_max"] = max(ratios) if ratios else None
    v["direction_note"] = (
        "§93 argued the filleted objective OVERSTATES stress because `Kt > 1`.  Measured "
        "against the fillet surface's own converged peak it understates it by %.2fx to "
        "%.2fx, because the other factor — a whole-wheel p=%g p-norm standing in for a "
        "local peak — errs further the other way.  Both scope gaps (one phase vs eight, "
        "linear vs SVK) raise the measured side, so these are lower bounds."
        % (v["measured_over_modelled_min"], v["measured_over_modelled_max"],
           rec["stress_nominal_p"]) if ratios else "no surface probe available")

    binding = {}
    for g in GENOMES:
        row = rec["genomes"][g]["modelled_against_measured"].get("svk") or \
              rec["genomes"][g]["modelled_against_measured"]["linear"]
        binding[g] = {j: {"util_modelled": row[j]["util_modelled"],
                          "util_measured": row[j]["util_measured"],
                          "modelled_admissible": row[j]["util_modelled"] <= 1.0,
                          "measured_admissible": None if row[j]["util_measured"] is None
                                                 else row[j]["util_measured"] <= 1.0}
                      for j in JUNCTIONS}
    v["admissibility"] = binding
    v["they_disagree_about_admissibility"] = any(
        binding[g][j]["modelled_admissible"] != binding[g][j]["measured_admissible"]
        for g in GENOMES for j in JUNCTIONS
        if binding[g][j]["measured_admissible"] is not None)
    return v


def _print(rec):
    W_ = "=" * 78
    print(W_)
    print("  WHAT `Kt * agg` IS WORTH ON A FILLETED MESH  —  PLAN.md §93 CONDITION B")
    print(W_)
    print(f"    allowable {rec['allowable_stress_mpa']:.4g} MPa    "
          f"margin knee {rec['margin_knee_util']:.2f}    "
          f"nominal p {rec['stress_nominal_p']:g}")

    print("\n  A — THE CORNER CENSUS.  WHICH SINGULARITY DOES THE FILLET ACTUALLY REMOVE?")
    for g in GENOMES:
        print(f"\n    {g} ({rec['genomes'][g]['genome_file']})")
        print("      corner       unfilleted wedge/lambda      filleted wedge/lambda   "
              "  resolved")
        for p, r in rec["genomes"][g]["corner_census"].items():
            u, f = r["unfilleted"], r["filleted"]
            ul = "  -   " if u["lambda"] is None else "%.4f" % u["lambda"]
            fl = "  -   " if f["lambda"] is None else "%.4f" % f["lambda"]
            print("      %-10s   %8.3f  %-11s %s   %8.3f  %-11s %s   %s"
                  % (p, u["wedge_deg"], u["kind"], ul,
                     f["wedge_deg"], f["kind"], fl,
                     "YES" if r["resolved_by_the_fillet"] else "no"))

    print("\n  B — DOES THE PEAK STOP DIVERGING?  (C1, the reason the construction exists)")
    print("      genome    mesh          peak MPa up the ladder            ratios      "
          "verdict")
    for g in GENOMES:
        for arm in ("unfilleted", "filleted"):
            d = rec["genomes"][g]["global_max_vm"][arm]
            print("      %-9s %-11s  %s   %s   %s"
                  % (g, arm, " ".join("%8.3f" % x for x in d["peak_mpa"]),
                     " ".join("%6.3f" % x for x in d["increment_ratios"]),
                     "DIVERGES" if d["diverges"] else "settles"))
    for g in GENOMES:
        for j in JUNCTIONS:
            d = rec["genomes"][g]["fillet_surface"][j]
            if d is None:
                continue
            print("      %-9s %s:surface  %s   %s   %s"
                  % (g, j, " ".join("%8.3f" % x for x in d["peak_mpa"]),
                     " ".join("%6.3f" % x for x in d["increment_ratios"]),
                     "DIVERGES" if d["diverges"]
                     else "settles -> %.4f" % d["settled_estimate_mpa"]))

    print("\n  C — WHAT THE DOUBLE COUNT IS WORTH  (8 phases, coarse; both meshes, so the")
    print("      unfilleted row shows what the same factor is doing where it belongs)")
    print("      genome    kin     mesh        junction  agg      Kt      util    "
          "util(Kt=1)  overstated  stress+margin")
    for g in GENOMES:
        for k, byarm in rec["genomes"][g]["objective"].items():
            for arm in ("unfilleted", "filleted"):
                row = byarm[arm]
                for j in JUNCTIONS:
                    c = row["junctions"][j]
                    print("      %-9s %-7s %-11s %-8s %7.4f  %6.4f  %7.4f  %7.4f    "
                          "%+7.4f    %8.4f"
                          % (g, k, arm, j, row["pnorm_stress_agg_mpa"], c["kt"],
                             c["util_as_computed"], c["util_with_kt_1"],
                             c["util_overstated_by"],
                             c["stress_as_computed"] + c["stress_margin_as_computed"]))
                print("      %-9s %-7s %-11s   stress loss %.4f, without Kt %.4f, "
                      "%s %+.4f"
                      % ("", "", arm, row["stress_loss_as_computed"],
                         row["stress_loss_with_kt_1"],
                         "DOUBLE COUNT WORTH" if arm == "filleted" else "Kt is worth",
                         row["kt_worth_in_loss"]))

    print("\n  D — THE MODELLED PEAK AGAINST THE MEASURED ONE  (filleted mesh, both sides)")
    print("      genome    kin     junction   modelled  measured   ratio   "
          "util_mod  util_meas")
    for g in GENOMES:
        for k, byj in rec["genomes"][g]["modelled_against_measured"].items():
            for j, r in byj.items():
                if r["measured_surface_peak_mpa"] is None:
                    continue
                print("      %-9s %-7s %-9s %8.4f  %8.4f  %6.3fx  %7.4f   %7.4f"
                      % (g, k, j, r["modelled_peak_mpa"],
                         r["measured_surface_peak_mpa"], r["measured_over_modelled"],
                         r["util_modelled"], r["util_measured"]))

    v = rec["verdict"]
    print("\n  VERDICT")
    print("    C1 (the peak diverges) survives the fillet : %s" % v["premise_survives"])
    print("      %s" % v["premise_survives_note"])
    print("    Kt prices a corner the mesh now resolves   : %s"
          % v["kt_prices_a_corner_the_mesh_resolves"])
    for g in GENOMES:
        print("      %-9s resolved %s ; still singular %s"
              % (g, v["corners_resolved_by_the_fillet"][g],
                 v["corners_still_singular_on_the_filleted_mesh"][g]))
    print("    the model UNDERSTATES the measured peak    : %s"
          % v["model_understates_everywhere"])
    print("      %s" % v["direction_note"])
    print("    modelled and measured disagree about admissibility : %s"
          % v["they_disagree_about_admissibility"])
    for g in GENOMES:
        for j in JUNCTIONS:
            b = v["admissibility"][g][j]
            print("      %-9s %-4s util modelled %.4f (%s)   measured %s (%s)"
                  % (g, j, b["util_modelled"],
                     "admissible" if b["modelled_admissible"] else "BREACHED",
                     "  n/a " if b["util_measured"] is None
                     else "%.4f" % b["util_measured"],
                     "n/a" if b["measured_admissible"] is None
                     else ("admissible" if b["measured_admissible"] else "BREACHED")))
    print(W_)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for (g, what), default in DEFAULT_INPUTS.items():
        ap.add_argument("--%s-%s" % (g, what.replace("_", "-")), default=default)
    ap.add_argument("--out", default="study_fillet_kt.json")
    args = ap.parse_args()

    inputs = {k: getattr(args, ("%s_%s" % k)) for k in DEFAULT_INPUTS}
    # What degrades THIS driver is being pointed at a probe artifact.  It solves nothing,
    # so there is no fidelity flag to lower — the only way to weaken it is to read a
    # weaker measurement, and then the verdict is about that measurement instead.
    _gate_guard.refuse_degraded_out(ap, args, "study_fillet_kt.json", [
        (inputs[k] != v, "--%s-%s %s, not the committed %s"
                         % (k[0], k[1].replace("_", "-"), inputs[k], v))
        for k, v in DEFAULT_INPUTS.items()])

    rec = build(inputs)
    _print(rec)
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"    wrote {args.out}")

    # NO PASS/FAIL AND EXIT 0, for `study_fillet_terms`' reason: §93 asked what the term is
    # worth, and a valuation has no threshold to meet.  The verdict's booleans are the
    # findings and they are filed rather than raised, because the tables beside them are
    # what make them diagnosable.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
