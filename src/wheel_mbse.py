"""
=============================================================================
  COMPLIANT WHEEL — THE MBSE FRONT END (a mission and 100 points, in; a
  requirement set, a compliance table and optionally a wheel, out)
=============================================================================
    .venv-opt/bin/python src/wheel_mbse.py --help              (make mbse)

MBSE_PLAN.md Step 7.  PLAN.md §97.

WHAT IT DOES, IN ORDER
-----------------------
  1  builds a `Mission` from the flags and a `Priorities` from `--points`
  2  derives a `Requirements` — force, stroke target, allowable stress, modulus,
     Poisson ratio, printable wall floor, and the whole 14-key weight table
  3  writes it to `--out-requirements` so `wheel_stage3.py --requirements` can read it
  4  scores `--genome` against it and prints the compliance table
  5  with `--descend`, warm-starts a Stage-3 descent from that genome under it

STEPS 1-4 SOLVE ONE FORWARD EVALUATION AND STEP 5 SPENDS A DESCENT.  That split is
deliberate and is MBSE_PLAN's own: *"a requirements layer that can only answer by
spending a descent cannot be debugged — a wrong requirement and a bad descent look the
same from the outside."*  So the default prints a table in seconds and `--descend` is
opt-in.

THE TWO INPUT SURFACES ARE SEPARATE FLAGS BECAUSE THEY ARE SEPARATE THINGS
---------------------------------------------------------------------------
A requirement you cannot choose is not a preference, and a preference is not a
requirement.  Mission flags are ABSOLUTE FACTS entered as numbers; `--points` is a
100-point ZERO-SUM allocation over the five `should`s.  **Points never reach a `shall`** —
`Priorities` refuses an axis that is not in `wheel_objective.OBJECTIVE_TERMS`, and the
budget is not a UI convention but the conservation law that keeps the
objective-against-barrier balance fixed while priorities move (weights are not scale-free
here: barriers are absolute, so scaling every objective weight halves every `shall`).

DEFAULTS REPRODUCE THE SHIPPED WHEEL
-------------------------------------
With no flags at all this derives the mission `Mission.implied_baseline()` — the one
`studies/study_mbse_baseline.py` reads out of the shipped constants — and the requirement
set it produces reproduces `wheel_fea`'s four constants and `DEFAULT_WEIGHTS` key for
key.  `--points` omitted means `DEFAULT_WEIGHTS` UNTOUCHED, not the calibrated allocation
re-derived and re-applied: those two agree to a couple of ulp and "a couple of ulp" is
not a thing a default should introduce into every downstream artifact.
"""

import argparse
import json
import os
import sys

import project_paths as PP

import wheel_fea as W
import wheel_genome as wg
import wheel_requirements as R

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_points(specs, smoothness_loss):
    """`["mass=60", "deflection=40"]` -> a validated `Priorities`.

    An axis the caller does not name is 0, NOT its calibrated share.  Silently topping an
    allocation up to 100 would mean a user who typed `--points mass=60` got 40 points of
    something they never asked for, and the budget's whole job is to make the tradeoff
    explicit.
    """
    axes = R.priority_axes()
    pts = {a: 0.0 for a in axes}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--points wants `axis=value`, got {spec!r}; "
                             f"axes are {list(axes)}")
        name, _, value = spec.partition("=")
        name = name.strip()
        if name not in pts:
            raise SystemExit(
                f"--points {name!r} is not one of the five objective terms {list(axes)}."
                f"  Points may not reach a BARRIER term: a barrier is a `shall` whose "
                f"only admissible value is zero and it is never traded against.")
        pts[name] = float(value)
    return R.Priorities(pts)          # raises unless it sums to 100


def reference_smoothness(path):
    """The reference genome's `smoothness` term — the one input to the points map that
    cannot come from the weight table, because that term's argument is an integral with
    no reference scale.  See `wheel_requirements.reference_costs`."""
    with open(path) as fh:
        return float(json.load(fh)["loss_terms"]["smoothness"])


def build_requirements(args):
    base = R.Mission.implied_baseline()
    fields = {}
    for name in ("auw_kg", "k_asym", "sink_rate_ms", "ambient_c", "nozzle_mm"):
        v = getattr(args, name)
        if v is not None:
            fields[name] = float(v)
    for name in ("n_wheels", "perimeters", "landings"):
        v = getattr(args, name)
        if v is not None:
            fields[name] = int(v)
    if args.field_class is not None:
        fields["field_class"] = args.field_class
    from dataclasses import replace
    mission = replace(base, **fields)

    priorities, smooth = None, None
    if args.points:
        smooth = reference_smoothness(args.reference)
        priorities = parse_points(args.points, smooth)
    return mission, R.Requirements.from_mission(mission, priorities, smooth), priorities


BAR = "=" * 78


def print_requirements(mission, req, priorities):
    print(BAR)
    print("  MBSE — MISSION -> REQUIREMENTS")
    print(BAR)
    print("  MISSION")
    for k, v in mission.as_dict().items():
        print("    %-16s %s" % (k, v))
    print("  DERIVED")
    p = req.provenance
    print("    weight %.4f N  ->  static %.4f N/wheel  ->  n_land %.4f  ->  force %.4f N"
          % (p["weight_n"], p["static_force_n"], p["landing_load_factor"], req.force_n))
    print("    stroke %.4f mm at %.2f efficiency  ->  s_eff %.6f m"
          % (p["stroke_mm"], R.STROKE_EFFICIENCY, p["effective_stroke_m"]))
    print("    E retention %.4f  sigma retention %.4f  SF %.4f (k_fatigue %.4f)"
          % (p["e_retention"], p["sigma_retention"], p["safety_factor"],
             p["fatigue_knockdown"]))
    print("  REQUIREMENTS  (req_hash %s)" % req.req_hash())
    for k in ("force_n", "target_deflection_mm", "allowable_stress_mpa", "min_wall_mm",
              "e_mpa", "nu"):
        print("    %-24s %.6f" % (k, getattr(req, k)))
    if priorities is not None:
        print("  PRIORITIES  (100 points, zero-sum, over the five `should`s)")
        for k, v in priorities.points.items():
            print("    %-16s %6.2f points   ->  weight %.6f" % (k, v, req.weights[k]))
    else:
        print("  PRIORITIES  none given — DEFAULT_WEIGHTS untouched")
    if mission.ambient_c != 20.0:
        import textwrap
        print("  THERMAL SCOPE")
        for line in textwrap.wrap(R.THERMAL_SCOPE_NOTE, 72):
            print("    " + line)


def print_table(table):
    print("\n  COMPLIANCE TABLE  (req_hash %s, provenance %s)"
          % (table["req_hash"], table["provenance"]))
    print("    %-22s %-7s %-34s %14s %12s  %s"
          % ("id", "kind", "quantity", "value", "limit", "verdict"))
    for r in table["rows"]:
        lim = "         —" if r["limit"] is None else "%12.6f" % r["limit"]
        print("    %-22s %-7s %-34s %14.6f %s  %s"
              % (r["id"], r["kind"], r["quantity"], r["value"], lim, r["verdict"]))
    print("    %s" % ("COMPLIANT" if table["compliant"] else "NON-COMPLIANT"))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # -- the mission.  Every default is None and means "the implied baseline's value",
    #    so the flags say what CHANGED rather than restating the whole mission.
    ap.add_argument("--auw-kg", type=float, default=None)
    ap.add_argument("--n-wheels", type=int, default=None)
    ap.add_argument("--k-asym", type=float, default=None)
    ap.add_argument("--sink-rate-ms", type=float, default=None)
    ap.add_argument("--field-class", choices=sorted(R.STROKE_BY_FIELD_CLASS),
                    default=None)
    ap.add_argument("--ambient-c", type=float, default=None)
    ap.add_argument("--landings", type=int, default=None)
    ap.add_argument("--nozzle-mm", type=float, default=None)
    ap.add_argument("--perimeters", type=int, default=None)
    # -- the priorities
    ap.add_argument("--points", action="append", default=[], metavar="AXIS=N",
                    help="repeatable; must sum to 100 over %s"
                         % list(R.priority_axes()))
    ap.add_argument("--reference", default=PP.BEST_SOLUTION,
                    help="the genome the points map is anchored at (its `smoothness` "
                         "term is the one scale the weight table cannot supply)")
    # -- what to do with it
    ap.add_argument("--genome", default=PP.BEST_SOLUTION)
    ap.add_argument("--config", default="coarse")
    ap.add_argument("--kinematics", choices=("linear", "svk"), default="svk")
    ap.add_argument("--n-phase", type=int, default=8)
    ap.add_argument("--out-requirements", default="requirements.json")
    ap.add_argument("--no-score", action="store_true",
                    help="derive and write the requirement set, then stop")
    ap.add_argument("--descend", action="store_true",
                    help="Stage-3 descent from --genome under these requirements")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--best-out", default="mbse_best.json")
    ap.add_argument("--run-out", default="mbse_run.json")
    args = ap.parse_args(argv)

    mission, req, priorities = build_requirements(args)
    print_requirements(mission, req, priorities)
    out_req = os.path.join(PP.ROOT, args.out_requirements) \
        if not os.path.isabs(args.out_requirements) else args.out_requirements
    req.save(out_req)
    print("\n  wrote %s  (req_hash %s)" % (out_req, req.req_hash()))

    if args.no_score:
        return 0

    rec = wg.load_record(args.genome)
    genes = wg.genes_to_vector(rec["genes"])
    scored = R.score_record(genes, req, args.config, n_phase=args.n_phase,
                            kinematics=args.kinematics)
    print("\n  SCORED  genome %s at %s / %s / %d phases   loss %.6f"
          % (rec["_hash"], args.config, args.kinematics, args.n_phase, scored["loss"]))
    table = R.verify(scored, req)
    print_table(table)

    if not args.descend:
        return 0

    # THE FLOOR MOVES BEFORE THE BOX IS READ, exactly as `wheel_stage3.main` does it —
    # `GENE_SPACE` consumes `MIN_WALL_MM` at import, so a later call would leave
    # `bounds_arrays` describing a box the descent is not in.
    req.apply_process()
    import wheel_stage3 as S3
    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    z0 = wg.normalize(genes, low, high)
    print("\n%s\n  STAGE 3 UNDER req_hash %s  (%s, %d steps, %s)\n%s"
          % (BAR, req.req_hash(), args.config, args.steps, args.kinematics, BAR))
    run = S3.descend(z0, args.config, steps=args.steps, n_phase=args.n_phase,
                     scheme="uniform", out=os.path.join(PP.ROOT, args.run_out),
                     workers=args.workers, kinematics=args.kinematics, req=req)
    best = run["best"]
    h = wg.save_record(
        os.path.join(PP.ROOT, args.best_out), best["genes"],
        source="wheel_mbse.py",
        search={"optimizer": "adam", "config": args.config, "steps": args.steps,
                "phase_scheme": "uniform", "n_phase": args.n_phase, "seed": 0,
                "start": "genome", "min_wall_mm": float(W.MIN_WALL_MM),
                "cy_bound_mm": float(W.CY_BOUND_MM), "kinematics": args.kinematics,
                "selection": best["selection"], "label": "mbse",
                "at_step": best["step"], "req_hash": req.req_hash(),
                "requirements_file": args.out_requirements},
        loss_terms=best["terms"], metrics=best["report"], loss=best["loss"],
        # TOP-LEVEL, never inside `genes` — `wheel_genome.save_record` refuses a key
        # there because it would change `genome_hash` for every genome on disk.
        requirements=req.as_dict())
    print("\n  wrote %s  (genome_hash %s, req_hash %s)"
          % (args.best_out, h, req.req_hash()))
    print_table(R.verify({"loss_terms": best["terms"], "metrics": best["report"],
                          "requirements": req.as_dict()}, req))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
