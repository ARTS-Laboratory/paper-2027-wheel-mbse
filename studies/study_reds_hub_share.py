"""REDS Step 3 — the hub compliance share, and PLAN §14 item 4b's unmeasured hypothesis.

`tests/test_wheel_fea.py::test_the_rim_band_holds_a_large_minority_of_the_compliance`
asserts `compliance_split["hub"] < 0.03`.  It has been red since §14, which recorded 0.0321
and called it 7% over.  It is now 0.0417 at the same rung.  §14 left it open deliberately,
and said exactly why:

    "This is the one where the *direction* is surprising: thinner, floppier spokes should
    push compliance toward the spokes and the hub share DOWN.  It went up.  The plausible
    cause is `R_hub` dropping 1.5598 -> 0.5790 — much less material at the hub junction —
    but that is a hypothesis and it has not been measured.  Least urgent of the eight and
    the only one whose sign is not understood."

This driver measures it.  Two experiments, which answer two different questions:

  --sweep   R_hub across its whole gene box (0.4 - 4.0) on ONE genome with everything else
            held fixed.  This is the controlled test of §14's hypothesis: if the hub share
            rises as R_hub falls, the hypothesis lives.  It also records the four GA
            barriers per point, because most of the upper box is not reachable geometry —
            `spoke_overlap_penalty` requires t0 + 2*R_hub + clearance to fit the sector
            chord — and a share quoted at an infeasible R_hub is not a finding about any
            wheel that could be built.

  --rungs   The 2x2 §14's own rule prescribes, design x mesh: {shipped, ga_beam} x
            {smoke, coarse, medium, fine}.  The share drifts UP with refinement, so part
            of the gap to `< 0.03` is discretisation rather than design, and a fixed bound
            on a non-converged quantity is PLAN §29's problem again.  Separating the two
            is what tells you whether `< 0.03` is measuring the wheel or the mesh.

THIS DRIVER DECIDES NOTHING.  §14 said the threshold call is a human's and REDS_PLAN.md
Step 3.3 does not overrule it; the output goes into PLAN.md and then to the user.

Run:
    studies/redsrun.sh studies/study_reds_hub_share.py --sweep
    studies/redsrun.sh studies/study_reds_hub_share.py --rungs
"""

import argparse
import json
import os
import time

import numpy as np

import project_paths as PP
import wheel_fea as W
import wheel_fem as fem
import wheel_genome as wg
import wheel_wheel as WW

HERE = os.path.dirname(os.path.abspath(__file__))

R_HUB_GENE = 12                 # GENE_SPACE index; see wheel_fea.py:307 ({0.4, 4.0})
BARRIERS = ("x_order", "hub_overlap", "fold", "arrival")
GENOMES = {"shipped": "best_solution.json", "ga_beam": "best_solution_ga_beam.json"}


def load(name):
    with open(os.path.join(PP.ROOT, GENOMES[name])) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


def shares(genes, cfg, fillet=None, kinematics="linear"):
    """The three compliance shares plus the axle drop, at one design and one rung.

    `fillet=True` MESHES the junction fillets, which is what turns this sweep from a
    control into FILLET_PLAN Step 3's acceptance test -- see `sweep`.  `kinematics` is
    explicit because `solve_wheel` defaults to the LINEAR kernel and FILLET_PLAN's cost
    section forbids Step 3 taking that default silently.
    """
    mesh = WW.build_wheel(genes, cfg, fillet=fillet)
    res = fem.solve_wheel(mesh, kinematics=kinematics)
    s = res["compliance_split"]
    return {"hub": float(s["hub"]), "spoke": float(s["spoke"]), "rim": float(s["rim"]),
            # What the mesh ACTUALLY built.  `fillet=True` clamps a radius with no room in
            # its own sector (`wheel_wheel.SECTOR_FIT_CLAMP`), and the top of `R_hub`'s
            # gene box is past the shipped genome's limit of 3.130 -- so on this sweep of
            # all sweeps the requested and applied radii diverge, and a row that did not
            # carry both would be plotting a radius the wheel does not have.
            "fillet_radii_mm": (None if mesh.fillet_radii_mm is None
                                else list(mesh.fillet_radii_mm)),
            "fillet_clamped": mesh.fillet_clamped,
            "axle_drop_mm": float(res["axle_drop_mm"]),
            # `axle_drop_mm` reads `uy` at whichever rim node is nearest the ground, and
            # that is a first-order error (PLAN.md §62).  The shares above are ratios of
            # compliances and do not touch it; the drop is carried for context, so both
            # readings are carried and the difference is visible rather than absorbed.
            "axle_drop_interp_mm": float(res["axle_drop_interp_mm"])}


def sweep(name, cfg, points, fillet=None, kinematics="linear"):
    """§14's hypothesis, controlled: move R_hub only, and watch the hub share.

    AND, WITH `fillet=True`, FILLET_PLAN STEP 3's OWN ACCEPTANCE TEST.  Step 3 item 1
    names this exact sweep and states what passing looks like: *"it currently returns the
    same 17 significant figures at every point in the box, and it should not."*  It does
    return them, because the mesh models no fillets -- `R_hub` and `R_rim` are genes 12
    and 13 and `dcoords/dgene` is identically zero for both (`wheel_adjoint`'s header).
    Meshing the fillet is the only thing that gives either of them mechanical feedback.

    THIS IS SENSITIVITY, NOT A GRADIENT, and the difference is not cosmetic.
    `mesh_coords` still refuses a filleted mesh outright, so nothing here reaches the
    optimizer; what a moving row buys is a MEASURED answer to "what does the fillet do to
    the wheel", against which the `Kt` correlation the objective actually prices these two
    genes through can be checked for the first time.
    """
    base = load(name)
    lo, hi = W.GENE_SPACE[R_HUB_GENE]["low"], W.GENE_SPACE[R_HUB_GENE]["high"]
    rows = []
    for r in list(np.linspace(lo, hi, points)) + [float(base[R_HUB_GENE])]:
        v = np.array(base, dtype=float)
        v[R_HUB_GENE] = float(r)
        _, loss = W.evaluate_design(v)
        row = {"R_hub": float(r),
               "is_shipped_value": bool(abs(r - base[R_HUB_GENE]) < 1e-12),
               "barriers": {t: float(loss[t]) for t in BARRIERS},
               "feasible": bool(all(loss[t] <= 0.0 for t in BARRIERS))}
        try:
            row.update(shares(v, cfg, fillet=fillet, kinematics=kinematics))
        except Exception as exc:                                  # pragma: no cover
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    rows.sort(key=lambda x: x["R_hub"])
    return {"genome": name, "config": cfg, "rows": rows,
            "fillet": bool(fillet), "kinematics": kinematics,
            "R_hub_shipped": float(base[R_HUB_GENE]), "box": [lo, hi],
            "spread": _spread(rows)}


def _spread(rows):
    """How much each reported quantity MOVED across the sweep -- the whole point.

    Reported rather than left to a reader's eye because "bit-identical" is the claim
    being tested and a table of 17-significant-figure repeats does not read as one.  A
    zero here is the pre-Step-3 state; anything else is the acceptance test passing.
    """
    out = {}
    for key in ("hub", "spoke", "rim", "axle_drop_mm", "axle_drop_interp_mm"):
        vals = [r[key] for r in rows if key in r]
        if len(vals) < 2:
            continue
        out[key] = {"min": min(vals), "max": max(vals),
                    "range": max(vals) - min(vals),
                    "distinct_values": len(set(vals)),
                    "bit_identical": len(set(vals)) == 1}
    out["n_rows"] = len(rows)
    out["n_solved"] = sum(1 for r in rows if "hub" in r)
    out["n_clamped"] = sum(1 for r in rows
                           if isinstance(r.get("fillet_clamped"), dict)
                           and any(r["fillet_clamped"].values()))
    return out


# One rung ABOVE `fine`, built here rather than added to `wheel_wheel.CONFIGS`.
#
# `fine` is the top of the tree's ladder, and four rungs left the shipped genome's hub
# share still climbing with its last increment LARGER than its second-to-last (+0.0025,
# +0.0016, +0.0020) — which is not enough to say "not converged" without being told what
# the next rung does.  It is a measurement, not a rung the project should acquire, so it
# does not go in CONFIGS: adding a fifth name there would put it in reach of every study
# and test that iterates the ladder, and `wheel_mesh` has its own same-named configs (a
# name crossing between the two modules has silently resolved against the wrong mesh
# before).  Roughly 2x `fine` in each direction.
ULTRA = ("ultra", 384, 10, 40, 8, 40, 7, 40)


def _config(name):
    if name != "ultra":
        return name
    return WW.WheelConfig(*ULTRA, n_curve=9600)


def rungs(configs):
    """Design x mesh, so discretisation drift and design change can be told apart."""
    out = {}
    for name in GENOMES:
        genes = load(name)
        out[name] = {"R_hub": float(genes[R_HUB_GENE]), "rungs": {}}
        for cfg in configs:
            t0 = time.time()
            out[name]["rungs"][cfg] = dict(shares(genes, _config(cfg)),
                                           elapsed_s=round(time.time() - t0, 1))
    return out


def attribute(cfg):
    """§14 asked WHICH variable moved the hub share.  R_hub cannot be it, so: which is?

    One-at-a-time gene swaps from the shipped genome toward `best_solution_ga_beam.json`,
    which is the design the `< 0.03` bound was calibrated on and which still passes it.
    Each row is the shipped wheel with exactly ONE gene replaced, so the column is an
    attribution rather than a correlation.

    Only genes 0-11 are swept.  12 and 13 are `R_hub`/`R_rim`, and the sweep above showed
    the meshed wheel is BIT-IDENTICAL across the whole `R_hub` box — `wheel_wheel.py:44`
    says why: "FILLETS ARE NOT MODELLED".
    """
    ship, beam = load("shipped"), load("ga_beam")
    base = shares(ship, cfg)["hub"]
    target = shares(beam, cfg)["hub"]
    rows = []
    for i in range(12):
        v = np.array(ship, dtype=float)
        v[i] = beam[i]
        try:
            h = shares(v, cfg)["hub"]
        except Exception as exc:                                  # pragma: no cover
            rows.append({"gene": i, "name": W.GENE_NAMES[i], "error": str(exc)})
            continue
        rows.append({"gene": i, "name": W.GENE_NAMES[i],
                     "shipped_value": float(ship[i]), "ga_beam_value": float(beam[i]),
                     "hub": h, "delta": h - base,
                     # How much of the shipped->ga_beam gap this ONE gene closes.
                     "closes_frac": (h - base) / (target - base) if target != base else None})
    rows.sort(key=lambda r: abs(r.get("delta", 0.0)), reverse=True)
    return {"config": cfg, "shipped_hub": base, "ga_beam_hub": target, "rows": rows}


def _print_attribute(rep):
    print(f"\n=== which gene moves the hub share?  one-at-a-time swaps, cfg={rep['config']} ===")
    print(f"  shipped hub={rep['shipped_hub']:.4f}   ga_beam hub={rep['ga_beam_hub']:.4f}"
          f"   (gap {rep['ga_beam_hub'] - rep['shipped_hub']:+.4f})")
    print(f"  {'gene':>6s} {'name':<7s} {'shipped':>9s} {'ga_beam':>9s} {'hub':>8s} "
          f"{'delta':>9s}  closes")
    for r in rep["rows"]:
        if "error" in r:
            print(f"  {r['gene']:>6d} {r['name']:<7s}  {r['error']}")
            continue
        print(f"  {r['gene']:>6d} {r['name']:<7s} {r['shipped_value']:9.4f} "
              f"{r['ga_beam_value']:9.4f} {r['hub']:8.4f} {r['delta']:+9.4f}  "
              f"{r['closes_frac']:+6.1%}")


def _print_sweep(rep):
    mesh = "FILLETED" if rep.get("fillet") else "unfilleted"
    print(f"\n=== R_hub sweep, genome={rep['genome']} cfg={rep['config']} "
          f"mesh={mesh} kinematics={rep.get('kinematics', 'linear')} "
          f"(box {rep['box'][0]} - {rep['box'][1]}, shipped {rep['R_hub_shipped']:.4f}) ===")
    print("   R_hub  applied    hub      spoke     rim     drop_mm   feasible  "
          "binding barrier")
    for r in rep["rows"]:
        if "error" in r:
            print(f"  {r['R_hub']:6.3f}   {r['error']}")
            continue
        worst = max(r["barriers"].items(), key=lambda kv: kv[1])
        mark = "  <-- SHIPPED" if r["is_shipped_value"] else ""
        rad = r.get("fillet_radii_mm")
        clamped = isinstance(r.get("fillet_clamped"), dict) and r["fillet_clamped"]["hub"]
        app = "     -" if rad is None else f"{rad[0]:6.3f}{'*' if clamped else ' '}"
        print(f"  {r['R_hub']:6.3f} {app}  {r['hub']:.4f}   {r['spoke']:.4f}  "
              f"{r['rim']:.4f}  {r['axle_drop_mm']:7.4f}    {str(r['feasible']):5s}    "
              f"{worst[0]}={worst[1]:.4g}{mark}")
    sp = rep.get("spread") or {}
    if sp.get("n_clamped"):
        print(f"  * {sp['n_clamped']} of {sp['n_rows']} rows had R_hub clamped inside its "
              f"own sector's room (limit 3.130 at this genome)")
    # FILLET_PLAN Step 3's acceptance test, stated as a verdict rather than left to the
    # reader to spot in a column of repeated digits.
    for key in ("hub", "axle_drop_mm"):
        s = sp.get(key)
        if not s:
            continue
        verdict = ("BIT-IDENTICAL across the box" if s["bit_identical"]
                   else f"MOVES: range {s['range']:.6g} over {s['distinct_values']} "
                        f"distinct values")
        print(f"  {key:<20s} {s['min']:.10g} .. {s['max']:.10g}   {verdict}")
    ok = [r for r in rep["rows"] if r["feasible"] and "error" not in r]
    if len(ok) >= 2:
        first, last = ok[0], ok[-1]
        # THREE-WAY, AND THE THIRD BRANCH IS THE ONE THAT WAS MISSING.  `>` is False when
        # the two are EXACTLY equal, so on the unfilleted mesh -- where every row of this
        # sweep is bit-identical because `R_hub` moves no node -- this printed "§14's
        # hypothesis IS KILLED".  A mesh that cannot express a hypothesis cannot falsify
        # it, and that line was the falsification's only evidence.  See FILLET_PLAN Step 3.
        if first["hub"] == last["hub"]:
            direction = ("does not move at all — this mesh models no fillets, so it "
                         "cannot answer §14 either way")
        elif first["hub"] > last["hub"]:
            direction = "RISES as R_hub falls — §14's hypothesis SURVIVES"
        else:
            direction = "FALLS as R_hub falls — §14's hypothesis IS KILLED"
        print(f"\n  over the FEASIBLE range {first['R_hub']:.3f} - {last['R_hub']:.3f}: "
              f"hub share {first['hub']:.4f} -> {last['hub']:.4f}")
        print(f"  the hub share {direction}")


def _print_rungs(rep):
    print("\n=== hub compliance share, design x mesh ===")
    cfgs = list(next(iter(rep.values()))["rungs"])
    print(f"  {'genome':10s} {'R_hub':>7s}  " + "  ".join(f"{c:>8s}" for c in cfgs)
          + "   drift smoke->finest")
    for name, d in rep.items():
        vals = [d["rungs"][c]["hub"] for c in cfgs]
        print(f"  {name:10s} {d['R_hub']:7.4f}  " + "  ".join(f"{v:8.4f}" for v in vals)
              + f"   {100 * (vals[-1] / vals[0] - 1):+7.2f}%")
    print("\n  (the gate is `hub < 0.03`)")
    for name, d in rep.items():
        for c in cfgs:
            v = d["rungs"][c]["hub"]
            print(f"    {name:10s} {c:8s} hub={v:.4f}  "
                  f"{'PASSES' if v < 0.03 else f'over by {100 * (v / 0.03 - 1):.1f}%'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true", help="§14's R_hub hypothesis")
    ap.add_argument("--rungs", action="store_true", help="design x mesh 2x2")
    ap.add_argument("--attribute", action="store_true",
                    help="which gene moves the hub share")
    ap.add_argument("--genome", default="shipped", choices=sorted(GENOMES))
    ap.add_argument("--config", default="coarse", help="rung for the sweep")
    ap.add_argument("--points", type=int, default=13)
    ap.add_argument("--fillet", action="store_true",
                    help="mesh the junction fillets (FILLET_PLAN Step 3's acceptance "
                         "test); implies --kinematics svk unless one is given")
    ap.add_argument("--kinematics", default=None, choices=("linear", "svk"),
                    help="default: linear unfilleted, svk filleted -- FILLET_PLAN's cost "
                         "section forbids Step 3 taking the kernel default silently")
    ap.add_argument("--configs", default="smoke,coarse,medium,fine")
    ap.add_argument("--out", default="study_reds_hub_share.json")
    args = ap.parse_args(argv)
    if not (args.sweep or args.rungs or args.attribute):
        ap.error("pick at least one of --sweep / --rungs / --attribute")

    t0, rep = time.time(), {}
    if args.sweep:
        kin = args.kinematics or ("svk" if args.fillet else "linear")
        # ONE KEY PER ARM, because these runs do not SUPERSEDE each other -- the
        # unfilleted sweep IS the control that makes "it stopped being bit-identical" a
        # finding, and overwriting it would delete the thing being compared against.
        # `sweep` keeps the historical default (unfilleted, linear) under the name every
        # reader of this artifact already knows; anything else names its own two arms, so
        # a filleted row can never be read against a differently-kernelled control by
        # accident.  Same reason the driver merges into an existing artifact at all.
        key = ("sweep" if not args.fillet and kin == "linear" else
               f"sweep_{'filleted' if args.fillet else 'unfilleted'}_{kin}")
        rep[key] = sweep(args.genome, args.config, args.points,
                         fillet=True if args.fillet else None, kinematics=kin)
        _print_sweep(rep[key])
    if args.rungs:
        rep["rungs"] = rungs(args.configs.split(","))
        _print_rungs(rep["rungs"])
    if args.attribute:
        rep["attribute"] = attribute(args.config)
        _print_attribute(rep["attribute"])
    rep["settings"] = {"elapsed_s": round(time.time() - t0, 1)}

    path = os.path.join(HERE, args.out)
    existing = {}
    if os.path.exists(path):
        with open(path) as fh:
            existing = json.load(fh)
    existing.update(rep)
    with open(path, "w") as fh:
        json.dump(existing, fh, indent=1)
    print(f"\nwrote {path}  ({rep['settings']['elapsed_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
