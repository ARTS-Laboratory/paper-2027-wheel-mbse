"""
=============================================================================
  THE DEFLECTION GATE'S STANDING — A MESH-CONVERGENCE STUDY ON THE QoI THE
  GATE IS ACTUALLY STATED ON
=============================================================================
    .venv-opt/bin/python studies/study_deflection_gci.py      (make gci)

SVK_PLAN.md's closing item, PLAN.md §26's ranked successor #2.  No optimizer, no descent:
forward evaluations of the shipped genome at every rung of the mesh ladder, under both
kinematics, with a Richardson extrapolation and a GCI on the result.

WHY IT EXISTS.  `±0.3% of the 2.0 mm target` is a PLAN-LEVEL promotion gate — SVK_PLAN
step 5, BUILD_PLAN steps 8 and 10, and the clause that BLOCKED the SVK candidate at
+1.65%.  It is nowhere in the code.  And SVK_PLAN step 6 found that it is satisfiable at
exactly ONE fidelity: the coarse-converged answer read +1.65% at medium, the
medium-converged answer read -1.71% at coarse, and no design in this tree has ever met
±0.3% at both.  Which rung the gate is stated at is therefore a CHOICE, and the number a
design is judged against moves when that choice moves.  SVK_PLAN named the fix in its
closing section and left it open:

    "Give the deflection QoI the GCI treatment M8b-i.5 gave the stress QoI, then state
     the gate against the extrapolated value instead of against whichever rung the
     descent happened to run on."

THE QoI HERE IS THE GATE'S, NOT `run_refinement`'S, and that is the whole reason this file
exists next to `study_wheel_fea.run_refinement` rather than inside it.  That function
already does Richardson and a GCI on an axle drop — but on `fem.solve_wheel(mesh)`: ONE
phase, LINEAR kinematics.  The gate is stated on `axle_drop_mean_mm` under SVK: the mean
over the 8-phase uniform stencil, which is what the `deflection` term is scored on and
what every promotion has been judged by.  Extrapolating the wrong QoI would answer a
question nobody asked, precisely.

THE ORIENTATION IS PINNED ACROSS THE LADDER, for the reason `study_svk_rescore` pins it
across kinematics: `flank_orientation` is a discrete branch, and letting each rung
re-derive it would put a topology change inside a mesh-refinement comparison.  It is
pinned at the FINEST rung — that is the one whose answer the extrapolation is anchored on
— and the per-rung values are recorded so a disagreement is visible rather than absorbed.

GCI, NOT A BARE RICHARDSON.  `run_refinement` assumes a constant refinement ratio and
takes `(d1-d2)/(d2-d3)` as the observed one.  The rungs here are NOT constant-ratio:
h = 1/sqrt(n_elements) gives 1.8257 from coarse to medium and 1.7889 from medium to fine,
a 2% inconsistency.  So this uses Roache's iterative solve for the observed order `p`,
which is stated for exactly that case, and reports the apparent order both ways so the
difference between the two treatments is on the record instead of hidden in a formula.
=============================================================================
"""

import argparse
import json
import math
import os
import time

import numpy as np

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import wheel_mesh as WM
import wheel_objective as WO
import wheel_pool as WP
import wheel_wheel as WW

HERE = os.path.dirname(os.path.abspath(__file__))

# The ladder.  `smoke` is included as a DIAGNOSTIC and excluded from the extrapolation:
# 72 elements is far outside any asymptotic range, and a three-point Richardson is only
# meaningful on points that are in one.  Reported so that claim is checkable.
LADDER = ("smoke", "coarse", "medium", "fine")
EXTRAPOLATE_FROM = ("coarse", "medium", "fine")

GENOME = "best_solution.json"
N_PHASE = 8
SAFETY_FACTOR = 1.25            # Roache's, the customary value for three-grid studies
GATE_PCT = 0.3                  # the plan-level gate, ± this many percent of target

# THE LADDER IS NOT UNIFORMLY REFINED, which is the first objection to any `p` reported
# here: span goes 64 -> 128 -> 256 (x2.0 twice) but thickness goes 6 -> 10 -> 16 (x1.667
# then x1.600), so the cells change aspect ratio as well as size and "the" refinement
# ratio depends on what you call h.  Roache's procedure assumes uniform refinement.  So
# every h that could reasonably be defended is carried, and the conclusion is only stated
# where they AGREE.  They do: under SVK p is 0.48-0.54 and the GCI is 2.4-3.5% for all of
# them, so the verdict below does not rest on the choice.
H_DEFS = {
    "1/sqrt(n_elements)": lambda c: 1.0 / math.sqrt(WM.get_config(c).n_elements),
    "1/n_span": lambda c: 1.0 / WM.get_config(c).n_span,
    "1/n_thick": lambda c: 1.0 / WM.get_config(c).n_thick,
    "1/sqrt(n_nodes)": lambda c: 1.0 / math.sqrt(WM.get_config(c).n_nodes),
}
H_PRIMARY = "1/sqrt(n_elements)"     # the isotropic equivalent; reported as THE number


def load_genes(path):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return np.array(list(json.load(fh)["genes"].values()), dtype=float)


def _h(cfg_name, which=H_PRIMARY):
    """Representative cell size.  The mesh is a structured span x thickness grid, so
    1/sqrt(n_elements) is the isotropic equivalent and is the primary choice; see
    `H_DEFS` for why every alternative is carried alongside it."""
    return H_DEFS[which](cfg_name)


def observed_order(phi, h, tol=1e-12, max_iter=200):
    """Roache's observed order of convergence for THREE grids at a NON-constant ratio.

    Solves Roache's  p = |ln|e32/e21| + q(p)| / ln(r21),
    q(p) = ln((r21^p - s)/(r32^p - s)), by fixed-point iteration from q=0.

    THE INDEX ORDER IS REVERSED FROM ROACHE'S AND THAT IS NOT COSMETIC.  He indexes 1 as
    the FINEST grid; `phi` here runs COARSE -> FINE.  So his r21 (= h2/h1, the finest
    pair) is this function's `h[1]/h[2]`, and his e32/e21 is the RECIPROCAL of the ratio
    formed from `phi` in this order.  The first draft of this function used the
    coarse-pair ratio as the denominator and returned p = 1.2568 for a synthetic p = 1.3.
    The constant-ratio unit cases could not catch it, because r21 == r32 makes the swap
    invisible — which is exactly why the non-constant case is in the test set.

    Returns (p, converged).  `p` is None when the ratio is non-monotone (e32/e21 <= 0),
    which is a real outcome — it means the three points do not sit on a single asymptotic
    trend and NOTHING should be extrapolated from them.
    """
    e21, e32 = phi[1] - phi[0], phi[2] - phi[1]
    if e21 == 0.0 or e32 == 0.0:
        return None, False
    ratio = e32 / e21
    if ratio <= 0.0:                      # non-monotone: oscillatory, not converging
        return None, False
    # Roache's r21 is the FINEST pair; ours is h[1]/h[2].  His e32/e21 is 1/ratio.
    r21_R, r32_R = h[1] / h[2], h[0] / h[1]
    s = math.copysign(1.0, ratio)
    ln_e = -math.log(abs(ratio))                       # = ln|e32/e21| in Roache's order
    p = abs(ln_e) / math.log(r21_R)
    for _ in range(max_iter):
        q = math.log((r21_R ** p - s) / (r32_R ** p - s))
        p_new = abs(ln_e + q) / math.log(r21_R)
        if abs(p_new - p) < tol:
            return p_new, True
        p = p_new
    return p, False


def richardson(phi, h):
    """Extrapolated value, observed order and GCI on the finest pair.

    `phi` is ordered COARSE -> FINE, which is the opposite of how Roache writes it (he
    indexes 1 as finest).  Stated because getting it backwards silently reports the
    extrapolation on the wrong side of the finest point.
    """
    p, ok = observed_order(phi, h)
    out = {"h": [float(x) for x in h], "phi": [float(x) for x in phi],
           "r21": float(h[0] / h[1]), "r32": float(h[1] / h[2]),
           "observed_order_p": None if p is None else float(p),
           "order_converged": bool(ok),
           # The constant-ratio shortcut `run_refinement` uses, for comparison only.
           "naive_ratio": float((phi[1] - phi[0]) / (phi[2] - phi[1]))
           if phi[2] != phi[1] else float("inf")}
    if p is None:
        out.update({"extrapolated_mm": None, "gci_fine": None, "monotone": False})
        return out
    r32 = h[1] / h[2]
    e32 = phi[2] - phi[1]
    ext = phi[2] + e32 / (r32 ** p - 1.0)
    gci = SAFETY_FACTOR * abs(e32 / phi[2]) / (r32 ** p - 1.0)
    out.update({"monotone": True,
                "extrapolated_mm": float(ext),
                "finest_error_vs_extrapolated_pct": float(100.0 * (phi[2] / ext - 1.0)),
                "gci_fine": float(gci),
                "gci_fine_pct": float(100.0 * gci)})
    return out


def run_ladder(genome=GENOME, ladder=LADDER, n_phase=N_PHASE, workers=0):
    genes = load_genes(genome)
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")

    # Pinned at the finest rung; every rung's own value recorded next to it.
    pinned = tuple(float(o) for o in
                   WW.flank_orientation(genes, WW.get_config(ladder[-1])))
    per_rung = {c: tuple(float(o) for o in
                         WW.flank_orientation(genes, WW.get_config(c))) for c in ladder}
    disagree = [c for c, o in per_rung.items()
                if max(abs(a - b) for a, b in zip(o, pinned)) > 1e-12]

    pool = WP.PhasePool(workers) if workers else None
    rows = []
    try:
        for name in ladder:
            cfg = WM.get_config(name)
            t0 = time.time()
            wanted = phases[:1] if pool is not None else phases
            meshes = WO.phase_meshes(genes, name, wanted, orientation=pinned)
            row = {"config": name, "n_elements": int(cfg.n_elements),
                   "n_nodes": int(cfg.n_nodes), "h": _h(name),
                   "orientation_own": per_rung[name],
                   "mesh_s": round(time.time() - t0, 1)}
            for kin in ("linear", "svk"):
                t1 = time.time()
                _, _, brk = WO.objective(
                    genes, name, normalized=False, phases=phases, meshes=meshes,
                    pool=pool, orientation=pinned, kinematics=kin)
                rep = brk["report"]
                drop = float(rep["axle_drop_mean_mm"])
                row[kin] = {
                    "axle_drop_mean_mm": drop,
                    "axle_drop_min_mm": float(rep["axle_drop_min_mm"]),
                    "axle_drop_max_mm": float(rep["axle_drop_max_mm"]),
                    "deflection_error_pct":
                        float(100.0 * (drop / WO.TARGET_DEFLECTION_MM - 1.0)),
                    "stress_utilisation": float(rep["stress_utilisation"]),
                    "elapsed_s": round(time.time() - t1, 1)}
                print(f"  {name:<7} {kin:<6} drop {drop:8.5f} mm  "
                      f"{row[kin]['deflection_error_pct']:+7.3f}%  "
                      f"util {row[kin]['stress_utilisation']:6.3f}  "
                      f"({row[kin]['elapsed_s']} s)", flush=True)
            rows.append(row)
    finally:
        if pool is not None:
            pool.close()

    out = {"genome": genome, "n_phase": n_phase, "scheme": "uniform",
           "target_deflection_mm": WO.TARGET_DEFLECTION_MM, "gate_pct": GATE_PCT,
           "orientation_pinned_at": ladder[-1], "orientation": list(pinned),
           "orientation_disagreements": disagree, "rows": rows,
           "extrapolate_from": list(EXTRAPOLATE_FROM)}

    return analyse(out)


def analyse(out):
    """Richardson, GCI and the h-definition sensitivity, from the measured rows alone.

    Split out from `run_ladder` so `--reanalyse` can redo the arithmetic on a saved
    report without paying for the ladder again — it is 95 minutes, and an analysis bug
    should not cost a re-run.  (One did: see `observed_order`.)
    """
    rows = out["rows"]
    order = [next(i for i, r in enumerate(rows) if r["config"] == c)
             for c in out["extrapolate_from"]]
    cfgs = [rows[i]["config"] for i in order]

    out["refinement"] = {}
    out["h_sensitivity"] = {}
    for kin in ("linear", "svk"):
        phi = [rows[i][kin]["axle_drop_mean_mm"] for i in order]

        by_h = {}
        for name in H_DEFS:
            r = richardson(phi, [_h(c, name) for c in cfgs])
            if r.get("extrapolated_mm") is not None:
                r["extrapolated_error_pct"] = float(
                    100.0 * (r["extrapolated_mm"] / WO.TARGET_DEFLECTION_MM - 1.0))
            by_h[name] = r
        out["h_sensitivity"][kin] = by_h

        res = dict(by_h[H_PRIMARY])
        if res.get("extrapolated_mm") is not None:
            # THE QUESTION THE FILE EXISTS TO ANSWER.  The band is the gate; the GCI is
            # the numerical uncertainty on the extrapolated value.  A design whose GCI is
            # WIDER than the band cannot be adjudicated by the gate at all.
            res["gate_decidable"] = bool(res["gci_fine_pct"] < GATE_PCT)
            res["extrapolated_inside_gate"] = bool(
                abs(res["extrapolated_error_pct"]) <= GATE_PCT)
            # And the same verdict under every other h, so "it depends how you measure
            # the cell" is answered in the artifact rather than in an argument.
            res["gate_decidable_under_all_h"] = all(
                v["gci_fine_pct"] < GATE_PCT for v in by_h.values()
                if v.get("gci_fine_pct") is not None)
            res["inside_gate_under_all_h"] = all(
                abs(v["extrapolated_error_pct"]) <= GATE_PCT for v in by_h.values()
                if v.get("extrapolated_mm") is not None)
            res["p_range"] = [min(v["observed_order_p"] for v in by_h.values()),
                              max(v["observed_order_p"] for v in by_h.values())]
            res["gci_range_pct"] = [min(v["gci_fine_pct"] for v in by_h.values()),
                                    max(v["gci_fine_pct"] for v in by_h.values())]
        out["refinement"][kin] = res
    return out


def _print(rep):
    T = rep["target_deflection_mm"]
    print("\n" + "=" * 78)
    print(f"  DEFLECTION GCI — {rep['genome']}, {rep['n_phase']}-phase uniform stencil")
    print("=" * 78)
    if rep["orientation_disagreements"]:
        print(f"  NOTE: flank_orientation differs from the pinned "
              f"({rep['orientation_pinned_at']}) value at: "
              f"{', '.join(rep['orientation_disagreements'])}")
    else:
        print("  flank_orientation is identical at every rung — the ladder is a pure "
              "mesh refinement.")
    print(f"\n  {'config':<8}{'elem':>7}{'h':>10}"
          f"{'linear mm':>12}{'err %':>9}{'svk mm':>12}{'err %':>9}")
    for r in rep["rows"]:
        print(f"  {r['config']:<8}{r['n_elements']:>7}{r['h']:>10.5f}"
              f"{r['linear']['axle_drop_mean_mm']:>12.5f}"
              f"{r['linear']['deflection_error_pct']:>+9.3f}"
              f"{r['svk']['axle_drop_mean_mm']:>12.5f}"
              f"{r['svk']['deflection_error_pct']:>+9.3f}")
    print(f"\n  Richardson on {' -> '.join(rep['extrapolate_from'])} "
          f"(r21={rep['refinement']['svk']['r21']:.4f}, "
          f"r32={rep['refinement']['svk']['r32']:.4f})")
    for kin in ("linear", "svk"):
        d = rep["refinement"][kin]
        if not d.get("monotone"):
            print(f"    {kin:<7} NON-MONOTONE — the three rungs do not sit on one "
                  f"asymptotic trend; nothing is extrapolated.")
            continue
        print(f"    {kin:<7} p = {d['observed_order_p']:.3f}"
              f"{'' if d['order_converged'] else ' (NOT converged)'}   "
              f"extrapolated {d['extrapolated_mm']:.5f} mm "
              f"({d['extrapolated_error_pct']:+.3f}%)   "
              f"GCI(fine) {d['gci_fine_pct']:.3f}%")
        print(f"            finest is {d['finest_error_vs_extrapolated_pct']:+.3f}% "
              f"from the extrapolated value; naive constant-ratio would read "
              f"{d['naive_ratio']:.4f}")
    print(f"\n  SENSITIVITY TO THE DEFINITION OF h — the ladder is not uniformly refined "
          f"(span x2.0, thickness x1.667 then x1.600)")
    for kin in ("linear", "svk"):
        for name, d in rep["h_sensitivity"][kin].items():
            if d.get("extrapolated_mm") is None:
                print(f"    {kin:<7} {name:<20} NON-MONOTONE")
                continue
            print(f"    {kin:<7} {name:<20} r32={d['r32']:.4f}  p={d['observed_order_p']:.4f}"
                  f"  ext {d['extrapolated_mm']:.5f} mm ({d['extrapolated_error_pct']:+.3f}%)"
                  f"  GCI {d['gci_fine_pct']:.3f}%")

    svk = rep["refinement"]["svk"]
    if svk.get("monotone"):
        print(f"\n  AGAINST THE ±{rep['gate_pct']}% GATE, under SVK:")
        print(f"    extrapolated {svk['extrapolated_error_pct']:+.3f}%  ->  "
              f"{'INSIDE' if svk['extrapolated_inside_gate'] else 'OUTSIDE'} the band")
        print(f"    GCI {svk['gci_fine_pct']:.3f}% vs a ±{rep['gate_pct']}% band  ->  "
              f"the gate {'CAN' if svk['gate_decidable'] else 'CANNOT'} be adjudicated "
              f"at this ladder's resolution")
        print(f"    and under EVERY h: p in [{svk['p_range'][0]:.3f}, "
              f"{svk['p_range'][1]:.3f}], GCI in [{svk['gci_range_pct'][0]:.3f}%, "
              f"{svk['gci_range_pct'][1]:.3f}%] -> decidable under all h: "
              f"{svk['gate_decidable_under_all_h']}, inside gate under all h: "
              f"{svk['inside_gate_under_all_h']}")
    print("=" * 78 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", default=GENOME)
    ap.add_argument("--n-phase", type=int, default=N_PHASE)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--ladder", default=",".join(LADDER))
    ap.add_argument("--out", default=os.path.join(HERE, "study_deflection_gci.json"))
    ap.add_argument("--reanalyse", metavar="REPORT.json",
                    help="redo the arithmetic on a saved report; runs no FEA")
    args = ap.parse_args()

    if args.reanalyse:
        with open(args.reanalyse) as fh:
            rep = analyse(json.load(fh))
        rep.setdefault("settings", {})["reanalysed"] = True
        _print(rep)
        with open(args.out, "w") as fh:
            json.dump(rep, fh, indent=2)
        print(f"  wrote {args.out}  (re-analysis only, no FEA)")
        return

    t0 = time.time()
    rep = run_ladder(args.genome, tuple(args.ladder.split(",")),
                     args.n_phase, args.workers)
    rep["settings"] = {"genome": args.genome, "n_phase": args.n_phase,
                       "workers": args.workers, "ladder": args.ladder,
                       "elapsed_s": round(time.time() - t0, 1)}
    _print(rep)
    with open(args.out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"  wrote {args.out}  ({rep['settings']['elapsed_s']} s)")


if __name__ == "__main__":
    main()
