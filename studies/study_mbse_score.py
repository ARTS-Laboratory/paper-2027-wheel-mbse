"""
=============================================================================
  THE SHIPPED WHEEL, SCORED AGAINST NAMED REQUIREMENT PROFILES
=============================================================================
    .venv-opt/bin/python studies/study_mbse_score.py           (make mbsescore)

MBSE_PLAN.md Step 5.  PLAN.md §97.

WHY THIS EXISTS
---------------
`wheel_requirements.verify` turns a scored design and a requirement set into a compliance
table — requirement ID, statement, verification method, measured quantity, value, limit,
margin, verdict.  This driver is the thing that fills it in: it re-solves the shipped
genome under a handful of named missions and prints one table per mission.

**IT IS THE ONLY MBSE DRIVER THAT SOLVES**, and it solves forward only — no optimiser, no
descent, no promotion.  Eight phases per profile at `--config coarse` under SVK.

THE CRITERION MUST BE ABLE TO COME BACK BOTH WAYS
--------------------------------------------------
A verifier that cannot fail is not a verifier, it is a formatting exercise.  So this
driver gates TWO things at once:

  1  the `baseline` profile is COMPLIANT, every barrier exactly 0.0, **and its evaluation
     is bit-identical to an evaluation that names no requirements at all** — the scalar,
     all 14 gradient components and every one of the 14 breakdown terms.  A default that
     moved would be a silent re-interpretation of every committed artifact and of the
     five study drivers that re-alias `SERVICE_FORCE_N` (`study_gnl.py:106`,
     `study_contact.py:94`, `study_gradient.py:120`, `study_fillet_cost.py:115`,
     `study_svk_rescore.py:67`).

  2  **at least one other profile comes back NON-COMPLIANT, naming a binding
     requirement.**  The shipped design has essentially no headroom — `stress_utilisation
     _hub = 0.8201` against a knee of 0.80 and `axle_drop_mean_mm = 1.99742` against a
     2.0 mm target — so a modest load or temperature increase should bind on stress.  If
     nothing binds, the requirement layer is not reaching the physics and this exits
     nonzero.

WHY SVK, EXPLICITLY
--------------------
`wheel_fem`'s kernel default is `linear` on purpose (§32) and eleven study drivers never
mention `kinematics` at all — a ladder built on those takes linear silently.  The shipped
genome was descended under SVK (`best_solution.json`'s `search.kinematics`), and a loss
from one strain measure is not comparable with a loss from the other.  So this driver
passes `kinematics` explicitly and prints it in the banner, which is §14's lesson stated
at the one place a human actually looks.

WHAT THE PROFILES ARE, AND WHAT EACH ONE MOVES
-----------------------------------------------
Every profile is `Mission.implied_baseline()` with ONE field changed, so each table is a
one-variable statement rather than a scenario:

  baseline        nothing                    the shipped constants, bit for bit
  hot_day         ambient_c   20 -> 40       E and the allowable BOTH fall
  heavy_payload   auw_kg     3.0 -> 4.5      force_n rises with the vehicle
  rough_field     field_class grass -> rough stroke rises, and the load factor FALLS
                                             with it — the one profile that moves two
                                             requirements in opposite directions
  long_life       landings 1e3 -> 1e5        the safety factor rises, allowable falls

**EVERY HOT PROFILE CARRIES THE SCOPE NOTE.**  `wheel_requirements.THERMAL_SCOPE_NOTE` is
printed under the `hot_day` table every time, not left in a docstring: the knockdown is
QUASI-STATIC — no creep, no fatigue, no thermal expansion, no self-heating, no rate
dependence — and PLA creeps badly above ~45 C, so a static allowable at elevated
temperature is OPTIMISTIC.

WHAT IT DOES NOT DO
--------------------
It does not re-optimise, and it does not promote.  A profile coming back non-compliant is
a statement about the SHIPPED wheel under a requirement it was not designed to, not a
defect.  `wheel_stage3.py --requirements` is what re-optimises.
"""

import argparse
import json
import os
import time
from dataclasses import replace

import numpy as np

import project_paths as PP

import _gate_guard

import jax_config  # noqa: F401
import wheel_genome as wg
import wheel_objective as WO
import wheel_requirements as R

HERE = os.path.dirname(os.path.abspath(__file__))

# One field each.  See the module docstring for what each one moves and why the set is
# one-variable-at-a-time rather than a set of scenarios.
PROFILES = {
    "baseline": {},
    "hot_day": {"ambient_c": 40.0},
    "heavy_payload": {"auw_kg": 4.5},
    "rough_field": {"field_class": "rough"},
    "long_life": {"landings": 100000},
}

BIT_IDENTITY_TERMS = WO.TERMS


def profile_requirements():
    """`{name: (Requirements, changed_fields)}`, one field off the implied baseline each.

    `Requirements.from_mission` and NOT `Requirements.baseline()` even for `baseline`
    itself: the point of the bit-identity check is that the DERIVATION lands on the
    shipped constants, and taking a shortcut here for the one profile that has one would
    make the identity a tautology instead of a measurement.
    """
    base = R.Mission.implied_baseline()
    return {name: (R.Requirements.from_mission(replace(base, **kw)), kw)
            for name, kw in PROFILES.items()}


def score(genes, req, cfg, kinematics, n_phase):
    """One forward evaluation, through `wheel_requirements.score_record`.

    The reshape lives in `src/` and not here so that this driver and the `make mbse`
    front end cannot disagree about which keys go in `metrics` — a compliance table and
    the artifact it claims to describe drifting apart is exactly the class of defect §25
    found in the study drivers.
    """
    t0 = time.time()
    out = R.score_record(genes, req, cfg, n_phase=n_phase, kinematics=kinematics)
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


def bit_identity(genes, cfg, kinematics, n_phase):
    """`objective(genes, req=baseline())` against `objective(genes)`, bit for bit.

    THE LOAD-BEARING TEST OF THE WHOLE ARC, run here as well as in
    `tests/test_requirements.py` because a test proves the code path and this proves the
    number that goes in the record — at the config and the kinematics the profiles below
    are actually scored at, on the genome that actually ships.
    """
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    v0, g0, b0 = WO.objective(genes, cfg, phases=phases, kinematics=kinematics)
    v1, g1, b1 = WO.objective(genes, cfg, phases=phases, kinematics=kinematics,
                              req=R.Requirements.baseline())
    terms = {k: {"plain": b0["terms"][k]["value"], "with_req": b1["terms"][k]["value"],
                 "identical": b0["terms"][k]["value"] == b1["terms"][k]["value"]}
             for k in BIT_IDENTITY_TERMS}
    g0, g1 = np.asarray(g0), np.asarray(g1)
    return {
        "value": {"plain": float(v0), "with_req": float(v1), "identical": v0 == v1},
        "grad_components_identical": int(np.sum(g0 == g1)),
        "grad_n": int(g0.size),
        "grad_max_abs_diff": float(np.max(np.abs(g0 - g1))),
        "terms": terms,
        "all_identical": bool(v0 == v1 and np.array_equal(g0, g1)
                              and all(t["identical"] for t in terms.values())),
    }


def build(genome_path, cfg, kinematics, n_phase):
    rec = wg.load_record(genome_path)
    genes = wg.genes_to_vector(rec["genes"])

    out = {"genome_file": os.path.basename(genome_path), "genome_hash": rec["_hash"],
           "config": cfg, "kinematics": kinematics, "n_phase": n_phase,
           "thermal_scope_note": R.THERMAL_SCOPE_NOTE,
           "profiles": {}}

    out["bit_identity"] = bit_identity(genes, cfg, kinematics, n_phase)

    for name, (req, changed) in profile_requirements().items():
        s = score(genes, req, cfg, kinematics, n_phase)
        table = R.verify(s, req)
        # The process requirement is checked HERE and not as a table row, because the
        # table's `shall` rows are exactly `BARRIER_TERMS` and adding a tenth would make
        # that statement false.  `min_wall_mm` is not a term in the loss at all — it is
        # the LOW BOUND on four genes, so it constrains a DESCENT and not a SCORE, and
        # the only thing there is to check on a finished genome is whether it respects
        # the floor it is being verified against.
        walls = {k: float(rec["genes"][k]) for k in ("t0", "t1", "t2", "t3")}
        out["profiles"][name] = {
            "changed": changed,
            "requirements": req.as_dict(),
            "score": s,
            "table": table,
            "process": {"min_wall_mm": req.min_wall_mm, "thickness_genes": walls,
                        "respects_floor": all(v >= req.min_wall_mm - 1e-12
                                              for v in walls.values())},
        }
    out["checks"] = checks(out)
    out["verdict"] = verdict(out)
    return out


def checks(rec):
    p = rec["profiles"]
    binding = non_compliant(rec)
    c = {
        "baseline_is_compliant": bool(p["baseline"]["table"]["compliant"]),
        "baseline_barriers_all_zero": all(
            r["value"] == 0.0 for r in p["baseline"]["table"]["rows"]
            if r["kind"] == "shall"),
        "baseline_bit_identical_to_no_requirements":
            bool(rec["bit_identity"]["all_identical"]),
        "some_profile_is_non_compliant": bool(binding),
        "non_compliant_profiles": {k: v for k, v in binding.items()},
        "shall_rows_are_exactly_barrier_terms": tuple(
            r["id"] for r in p["baseline"]["table"]["rows"] if r["kind"] == "shall"
        ) == tuple("SHALL-%s" % t.upper().replace("_", "-") for t in WO.BARRIER_TERMS),
        "should_rows_are_exactly_objective_terms": tuple(
            r["id"] for r in p["baseline"]["table"]["rows"] if r["kind"] == "should"
        ) == tuple("SHOULD-%s" % t.upper().replace("_", "-")
                   for t in WO.OBJECTIVE_TERMS),
    }
    c["all_ok"] = bool(c["baseline_is_compliant"] and c["baseline_barriers_all_zero"]
                       and c["baseline_bit_identical_to_no_requirements"]
                       and c["some_profile_is_non_compliant"]
                       and c["shall_rows_are_exactly_barrier_terms"]
                       and c["should_rows_are_exactly_objective_terms"])
    return c


def non_compliant(rec):
    """`{profile: [failing requirement ids]}` — the criterion coming back the other way."""
    out = {}
    for name, p in rec["profiles"].items():
        bad = [r["id"] for r in p["table"]["rows"] if r["verdict"] == "FAIL"]
        if bad:
            out[name] = bad
    return out


def verdict(rec):
    v = {}
    bi = rec["bit_identity"]
    v["bit_identity_note"] = (
        "objective(genes, req=Requirements.baseline()) equals objective(genes) in the "
        "scalar, in %d of %d gradient components (worst |diff| %.3e) and in all %d "
        "breakdown terms."
        % (bi["grad_components_identical"], bi["grad_n"], bi["grad_max_abs_diff"],
           len(bi["terms"])))
    binding = non_compliant(rec)
    v["non_compliant"] = binding
    v["verifier_can_fail"] = bool(binding)
    if binding:
        name = sorted(binding)[0]
        v["binding_note"] = (
            "%d of %d profiles come back NON-COMPLIANT; %s is bound by %s.  The verifier "
            "can return both ways, which is what makes the compliant rows worth reading."
            % (len(binding), len(rec["profiles"]), name, ", ".join(binding[name])))
    else:
        v["binding_note"] = (
            "NO profile came back non-compliant.  That is a RED: the shipped design sits "
            "at util 0.8201 against a knee of 0.80 with no headroom, so a requirement "
            "layer that reaches the physics must be able to bind it.  Either the "
            "requirements are not reaching the solve or the profiles are too mild.")
    rows = {}
    for name, p in rec["profiles"].items():
        m = p["score"]["metrics"]
        rows[name] = {"loss": p["score"]["loss"],
                      "force_n": p["requirements"]["force_n"],
                      "e_mpa": p["requirements"]["e_mpa"],
                      "allowable_stress_mpa": p["requirements"]["allowable_stress_mpa"],
                      "target_deflection_mm": p["requirements"]["target_deflection_mm"],
                      "axle_drop_mean_mm": m["axle_drop_mean_mm"],
                      "stress_utilisation_hub": m["stress_utilisation_hub"],
                      "mesh_mass_g": m["mesh_mass_g"],
                      "compliant": p["table"]["compliant"]}
    v["summary"] = rows
    return v


BAR = "=" * 78


def _wrap(text, indent, width=72):
    import textwrap
    return ("\n" + " " * indent).join(textwrap.wrap(text, width))


def _print_table(name, p):
    t = p["table"]
    print("\n  PROFILE `%s`  %s" % (name, p["changed"] or "(the shipped constants)"))
    req = p["requirements"]
    print("      force %.4f N   target %.3f mm   allowable %.4f MPa   E %.1f MPa   "
          "min wall %.4f mm"
          % (req["force_n"], req["target_deflection_mm"], req["allowable_stress_mpa"],
             req["e_mpa"], req["min_wall_mm"]))
    print("      req_hash %s   provenance %s" % (t["req_hash"], t["provenance"]))
    print("      %-22s %-8s %-34s %14s %12s %10s  %s"
          % ("id", "kind", "quantity", "value", "limit", "margin", "verdict"))
    for r in t["rows"]:
        lim = "         —" if r["limit"] is None else "%12.6f" % r["limit"]
        mar = "        —" if r["margin"] is None else "%10.6f" % r["margin"]
        print("      %-22s %-8s %-34s %14.6f %s %s  %s"
              % (r["id"], r["kind"], r["quantity"], r["value"], lim, mar, r["verdict"]))
        ev = r.get("evidence")
        if ev is not None:
            print("      %-22s %-8s   evidence: %s = %.6f (%s %.4f)"
                  % ("", "", ev["quantity"], ev["value"], ev["sense"], ev["limit"]))
    print("      process: min wall %.4f mm, thickness genes %s -> %s"
          % (p["process"]["min_wall_mm"],
             " ".join("%.4f" % v for v in p["process"]["thickness_genes"].values()),
             "respected" if p["process"]["respects_floor"] else "VIOLATED"))
    print("      COMPLIANT" if t["compliant"] else "      NON-COMPLIANT")


def _print(rec):
    v, c = rec["verdict"], rec["checks"]
    print(BAR)
    print("  THE SHIPPED WHEEL AGAINST NAMED REQUIREMENT PROFILES  —  MBSE_PLAN STEP 5")
    print(BAR)
    print("    genome %s (%s)   config %s   kinematics %s   %d phases"
          % (rec["genome_hash"], rec["genome_file"], rec["config"], rec["kinematics"],
             rec["n_phase"]))

    print("\n  THE BIT-IDENTITY CHECK  (req=baseline() against no requirements at all)")
    bi = rec["bit_identity"]
    print("      scalar        %.17g  vs  %.17g   %s"
          % (bi["value"]["plain"], bi["value"]["with_req"],
             "identical" if bi["value"]["identical"] else "DIFFERENT"))
    print("      gradient      %d/%d components identical, worst |diff| %.3e"
          % (bi["grad_components_identical"], bi["grad_n"], bi["grad_max_abs_diff"]))
    bad = [k for k, t in bi["terms"].items() if not t["identical"]]
    print("      breakdown     %d/%d terms identical%s"
          % (len(bi["terms"]) - len(bad), len(bi["terms"]),
             "" if not bad else "  — DIFFERENT: %s" % bad))

    for name in PROFILES:
        _print_table(name, rec["profiles"][name])
        if rec["profiles"][name]["changed"].get("ambient_c") is not None:
            print("      SCOPE: %s" % _wrap(rec["thermal_scope_note"], 14))

    print("\n  SUMMARY")
    print("      profile          force N   allow MPa    E MPa   target mm    drop mm   "
          "util hub   mass g   verdict")
    for name, r in v["summary"].items():
        print("      %-14s %9.3f %11.4f %8.1f %11.3f %10.5f %10.5f %8.3f   %s"
              % (name, r["force_n"], r["allowable_stress_mpa"], r["e_mpa"],
                 r["target_deflection_mm"], r["axle_drop_mean_mm"],
                 r["stress_utilisation_hub"], r["mesh_mass_g"],
                 "COMPLIANT" if r["compliant"] else "NON-COMPLIANT"))

    print("\n  CHECKS")
    for k in ("baseline_is_compliant", "baseline_barriers_all_zero",
              "baseline_bit_identical_to_no_requirements",
              "some_profile_is_non_compliant", "shall_rows_are_exactly_barrier_terms",
              "should_rows_are_exactly_objective_terms"):
        print("    %-45s %s" % (k, "ok" if c[k] else "FAIL"))

    print("\n  VERDICT")
    print("    %s" % _wrap(v["bit_identity_note"], 4))
    print("    %s" % _wrap(v["binding_note"], 4))
    print(BAR)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genome", default=PP.BEST_SOLUTION)
    ap.add_argument("--config", default="coarse")
    ap.add_argument("--kinematics", choices=("linear", "svk"), default="svk")
    ap.add_argument("--n-phase", type=int, default=8)
    ap.add_argument("--out", default="study_mbse_score.json")
    args = ap.parse_args()

    _gate_guard.refuse_degraded_out(ap, args, "study_mbse_score.json", [
        (args.config != "coarse", "--config %s, not the committed coarse" % args.config),
        (args.kinematics != "svk",
         "--kinematics %s; the shipped genome was descended under svk and a loss from "
         "one strain measure is not comparable with a loss from the other"
         % args.kinematics),
        (args.n_phase != 8, "--n-phase %d, not the committed 8" % args.n_phase),
        (os.path.abspath(args.genome) != PP.BEST_SOLUTION,
         "--genome %s, not the shipped best_solution.json" % args.genome)])

    rec = build(args.genome, args.config, args.kinematics, args.n_phase)
    _print(rec)
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"    wrote {args.out}")
    return 0 if rec["checks"]["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
