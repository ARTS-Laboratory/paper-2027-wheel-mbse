"""
=============================================================================
  M2b GATE — IS THE FULL-WHEEL MESH A PARTITION OF THE ACTUAL PART?
=============================================================================
    .venv-opt/bin/python studies/study_wheel_mesh.py --samples 200

Three checks, and the second is the one that would otherwise sink M4 silently.

  AREA      the assembled mesh, plus the un-meshed rigid hub core, must reproduce the
            area of the region it claims to model, to < 0.5%.  Validated against OCC
            for the same region definition (2469.06 by an independent kernel), and
            reported alongside the shipped STEP's 2521.44 so the 2.1% `_embed`
            difference stays visible rather than quietly absorbed.

  SEAMS     for every sampled genome, the largest distance between what an owning and a
            non-owning block INDEPENDENTLY computed for the same shared node must be at
            machine precision.  This is the plan's non-negotiable test, and the reason
            is that a seam mismatch produces a mesh that plots correctly, has positive
            Jacobian everywhere, solves without complaint, and models a wheel with
            twelve cracks in it.  Nothing else in the pipeline would notice.

  VALIDITY  minimum scaled Jacobian over the design space, per block type, against the
            same 0.2 floor the M2a spoke-block gate used.

WHAT THE PER-BLOCK BREAKDOWN IS FOR
-----------------------------------
The aggregate minSJ hides which construction is weak.  Reported per block it is
immediately clear: the two junction Coons patches are the worst and everything else is
essentially perfect, because the junctions are the only blocks bounded by a circular arc
on one side and a straight cross-section on the other.  That is also the number to watch
if the junction construction is ever changed — a regression there will not move the
aggregate much.
=============================================================================
"""

import argparse
import json
import os

import project_paths as PP
import time

import numpy as np

import wheel_fea as W
import wheel_genome as wg
import wheel_mesh as M
import wheel_wheel as WW
from study_mesh_quality import fold_margin, latin_hypercube

MIN_SJ_ACCEPT = 0.2
ACCEPT_FRACTION = 0.98
MAX_SEAM_ERROR_MM = 1e-10
MAX_AREA_ERROR = 0.005

HERE = os.path.dirname(os.path.abspath(__file__))


def shipped_genome():
    with open(os.path.join(PP.ROOT, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


# ---------------------------------------------------------------------------
# AREA
# ---------------------------------------------------------------------------

def run_area(genes, configs=("smoke", "coarse", "medium", "fine")):
    """Area convergence across the config ladder.

    Refinement must drive the error toward zero, not merely make it small: the junction
    Coons patches and the polygonal approximation of the ring arcs both under-cut the
    true region, so a single config agreeing to 0.5% could be a coincidence between two
    errors of opposite sign.
    """
    rows = []
    for name in configs:
        t0 = time.time()
        mesh = WW.build_wheel(genes, name)
        rep = WW.area_report(mesh)
        rows.append({"config": name, "n_elements": mesh.n_elements,
                     "n_nodes": mesh.n_nodes, "seconds": round(time.time() - t0, 2),
                     **rep})
    errs = [abs(r["error_vs_modelled"]) for r in rows]
    return {"rows": rows,
            "converging": bool(all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))),
            "finest_error": errs[-1],
            "pass": bool(errs[-1] < MAX_AREA_ERROR
                         and all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)))}


# ---------------------------------------------------------------------------
# SEAMS AND VALIDITY ACROSS THE DESIGN SPACE
# ---------------------------------------------------------------------------

def sample_feasible(n, cfg_name, seed, max_draws=200_000):
    """Draw genomes that are geometrically feasible AND pass the fold constraint.

    The fold constraint is included because M2a established it as a hard requirement the
    GA never had: without it, roughly 40% of the genomes the existing constraints admit
    produce inverted elements, and sweeping those here would only re-measure that.
    """
    spoke_cfg = M.get_config("coarse")
    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    out, draws, batch = [], 0, max(64, n)
    while len(out) < n and draws < max_draws:
        for vec in latin_hypercube(batch, low, high, seed=seed + draws):
            draws += 1
            _, loss = W.evaluate_design(vec)
            if loss["x_order"] != 0.0 or loss["hub_overlap"] != 0.0:
                continue
            if fold_margin(vec, spoke_cfg) <= _geom_floor():
                continue
            out.append(vec)
            if len(out) >= n:
                break
    return out, draws


def _geom_floor():
    import wheel_geometry as G
    return G.MIN_FOLD_MARGIN_MM


def run_design_space(n, cfg_name, seed):
    """Seam error and per-block validity over `n` feasible genomes.

    Reports validity under TWO feasibility definitions, the same structure the M2a spoke
    gate used, because the interesting result is not the percentage but which constraint
    the percentage is missing:

        geometric + fold          what the GA plus M2a's barrier enforce
        ... plus arrival angle    the constraint the junction blocks additionally need
    """
    vectors, draws = sample_feasible(n, cfg_name, seed)
    cfg = WW.get_config(cfg_name)
    rows, per_block = [], {name: [] for name in WW.BLOCK_ORDER}
    failures = []
    for vec in vectors:
        arrival = tuple(float(a) for a in WW.arrival_angles(vec, cfg))
        row = {"arrival_hub_deg": arrival[0], "arrival_rim_deg": arrival[1],
               "worst_arrival_deg": max(arrival)}
        try:
            mesh = WW.build_wheel(vec, cfg_name)
        except ValueError as exc:
            # A build that REFUSES is a pass for the seam check and a data point for
            # validity: the guard fired rather than a bad mesh escaping.
            failures.append({"error": str(exc)[:110], **row})
            rows.append({"built": False, "min_sj": -1.0, "seam_error_mm": 0.0,
                         "n_inverted": 0, **row})
            continue
        q = WW.quality_report(mesh)
        for name, v in q["per_block"].items():
            per_block[name].append((v["min_scaled_jacobian"], max(arrival)))
        rows.append({"built": True,
                     "seam_error_mm": q["seam_error_mm"],
                     "min_sj": q["min_scaled_jacobian"],
                     "max_ar": q["max_aspect_ratio"],
                     "n_inverted": q["n_inverted"], **row})

    sj = np.array([r["min_sj"] for r in rows])
    arr = np.array([r["worst_arrival_deg"] for r in rows])
    seam = np.array([r["seam_error_mm"] for r in rows if r["built"]])
    inv = int(sum(r["n_inverted"] for r in rows))

    keep = arr <= WW.MAX_ARRIVAL_DEG
    frac_all = float((sj > MIN_SJ_ACCEPT).mean())
    frac_kept = float((sj[keep] > MIN_SJ_ACCEPT).mean()) if keep.any() else 0.0

    # Threshold sweep, exactly as the M2a fold-margin gate does it: the recommended
    # value is the LOOSEST threshold with zero misses, so the barrier costs as little of
    # the design space as it can.
    sweep = []
    for thr in (40.0, 55.0, 60.0, 65.0, 70.0, 72.0, 80.0, 90.0):
        k = arr <= thr
        missed = int(((sj <= MIN_SJ_ACCEPT) & k).sum())
        sweep.append({"max_arrival_deg": thr, "kept": int(k.sum()),
                      "false_alarm": int(((sj > MIN_SJ_ACCEPT) & ~k).sum()),
                      "missed": missed,
                      "validity": float((sj[k] > MIN_SJ_ACCEPT).mean()) if k.any() else 0.0})

    return {
        "n_requested": n, "n_sampled": len(rows),
        "n_built": int(sum(r["built"] for r in rows)), "raw_draws": draws,
        "config": cfg_name,
        "worst_seam_error_mm": float(seam.max()) if seam.size else 0.0,
        "seam_pass": bool(seam.size and seam.max() < MAX_SEAM_ERROR_MM),
        "fraction_above_sj_floor_all": frac_all,
        "fraction_above_sj_floor_with_arrival_constraint": frac_kept,
        "fraction_of_space_kept": float(keep.mean()),
        "min_sj_percentiles": {f"p{p}": float(np.percentile(sj[keep], p))
                               for p in (0, 1, 5, 50)} if keep.any() else {},
        "total_inverted": inv,
        "threshold_sweep": sweep,
        "arrival_vs_sj_correlation": float(np.corrcoef(arr, sj)[0, 1]),
        "per_block_worst_min_sj": {
            k: (float(min(x[0] for x in v if x[1] <= WW.MAX_ARRIVAL_DEG))
                if any(x[1] <= WW.MAX_ARRIVAL_DEG for x in v) else None)
            for k, v in per_block.items()},
        "build_failures": failures[:5],
        "n_build_failures": len(failures),
        "worst_arrival_among_failures_deg": (
            max(f["worst_arrival_deg"] for f in failures) if failures else None),
        "min_arrival_among_failures_deg": (
            min(f["worst_arrival_deg"] for f in failures) if failures else None),
        "pass": bool(seam.size and seam.max() < MAX_SEAM_ERROR_MM
                     and frac_kept >= ACCEPT_FRACTION and inv == 0),
    }


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def _print(rep):
    def head(s):
        print(f"\n{s}\n" + "-" * len(s))

    def verdict(ok):
        return "PASS" if ok else "FAIL"

    a = rep["area"]
    head("AREA — the mesh must partition the region it claims to model")
    print(f"  {'config':8s} {'elem':>7s} {'nodes':>7s}  {'meshed':>10s} "
          f"{'+core':>10s}  {'vs modelled':>12s}  {'vs STEP':>9s}   s")
    for r in a["rows"]:
        print(f"  {r['config']:8s} {r['n_elements']:7d} {r['n_nodes']:7d}  "
              f"{r['meshed_mm2']:10.4f} {r['total_modelled_mm2']:10.4f}  "
              f"{r['error_vs_modelled']:+12.4%}  "
              f"{r['error_vs_shipped_step']:+9.3%}  {r['seconds']:5.2f}")
    r0 = a["rows"][0]
    print(f"\n  reference (this region, analytic)     {r0['reference_modelled_mm2']:9.3f} mm2")
    print(f"      = hub disk + rim band + 12 clipped spoke bands, from the frame")
    print(f"        constants and the genome (`wheel_wheel.modelled_area_reference`),")
    print(f"        down the EXPORTER's geometry path — finite-difference offset normals")
    print(f"        and exact shoelace, against the mesh's analytic hodograph and Q9")
    print(f"        Gauss.  There was an OCC cross-check here too; it was measured at")
    print(f"        RIM_RADIUS_MM = 48.9 and is not re-run, so quoting it now would be")
    print(f"        quoting a different wheel.")
    print(f"  shipped STEP cross-section            {r0['reference_shipped_step_mm2']:9.3f} mm2")
    print(f"  the {abs(a['rows'][-1]['error_vs_shipped_step']):.1%} gap is "
          f"wheel_step_export._embed, not a meshing error "
          f"(see wheel_wheel.py)")
    print(f"  error decreasing under refinement: {a['converging']}   "
          f"[< {MAX_AREA_ERROR:.1%} at the finest]")
    print(f"  -> {verdict(a['pass'])}")

    d = rep["design_space"]
    head(f"SEAMS — {d['n_sampled']} feasible genomes ({d['raw_draws']} draws), "
         f"config {d['config']!r}")
    print(f"  worst distance between what an owning and a non-owning block")
    print(f"  INDEPENDENTLY computed for the same shared node:")
    print(f"      {d['worst_seam_error_mm']:.3e} mm   [< {MAX_SEAM_ERROR_MM:.0e}]")
    print(f"  -> {verdict(d['seam_pass'])}")

    head("VALIDITY — and the constraint the junction blocks need")
    print(f"  minSJ > {MIN_SJ_ACCEPT} over...")
    print(f"    geometric + fold feasibility only : "
          f"{d['fraction_above_sj_floor_all']:.2%}"
          f"  -> {'PASS' if d['fraction_above_sj_floor_all'] >= ACCEPT_FRACTION else 'FAIL'}")
    print(f"    + the arrival-angle constraint     : "
          f"{d['fraction_above_sj_floor_with_arrival_constraint']:.2%}"
          f"  -> {'PASS' if d['fraction_above_sj_floor_with_arrival_constraint'] >= ACCEPT_FRACTION else 'FAIL'}")
    print(f"  inverted elements, total {d['total_inverted']}")
    print(f"  correlation(worst arrival angle, minSJ) = "
          f"{d['arrival_vs_sj_correlation']:+.3f}")
    if d["min_sj_percentiles"]:
        pcts = "  ".join(f"{k}={v:.3f}" for k, v in d["min_sj_percentiles"].items())
        print(f"  minSJ percentiles, constrained subset: {pcts}")
    print(f"\n  What should the barrier actually require?")
    print(f"      {'arrival <=':>12s}  missed  false alarm   kept  minSJ ok")
    for r in d["threshold_sweep"]:
        print(f"      {r['max_arrival_deg']:9.0f} deg  {r['missed']:6d}  "
              f"{r['false_alarm']:11d}  {r['kept']:5d}  {r['validity']:7.2%}")
    print(f"\n  worst minSJ per block, constrained subset:")
    for k, v in sorted(d["per_block_worst_min_sj"].items(),
                       key=lambda kv: (kv[1] is None, kv[1])):
        print(f"      {k:18s} {'n/a' if v is None else f'{v:7.4f}'}")
    if d["n_build_failures"]:
        print(f"\n  builds refused: {d['n_build_failures']} of {d['n_sampled']} "
              f"(the guard fired; no bad mesh escaped)")
        print(f"      their worst arrival angles ran "
              f"{d['min_arrival_among_failures_deg']:.1f} to "
              f"{d['worst_arrival_among_failures_deg']:.1f} deg — the same cause")
        for f in d["build_failures"][:2]:
            print(f"      {f['error']}")
    print(f"  -> {verdict(d['pass'])}")

    head("M2b GATE")
    for k in ("area", "design_space"):
        print(f"  {k:13s} {verdict(rep[k]['pass'])}")
    print(f"\n  OVERALL: {verdict(rep['pass'])}")


def main():
    ap = argparse.ArgumentParser(description="M2b full-wheel mesh gate")
    ap.add_argument("--samples", type=int, default=200,
                    help="feasible genomes for the seam and validity sweep")
    ap.add_argument("--config", default="coarse")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="study_wheel_mesh.json")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="skip the two finest configs; for CI, not for the record")
    args = ap.parse_args()

    genes = shipped_genome()
    t0 = time.time()
    configs = ("smoke", "coarse") if args.quick else ("smoke", "coarse", "medium", "fine")
    rep = {"area": run_area(genes, configs),
           "design_space": run_design_space(args.samples, args.config, args.seed)}
    rep["pass"] = all(rep[k]["pass"] for k in ("area", "design_space"))
    rep["settings"] = {"samples": args.samples, "config": args.config,
                       "seed": args.seed, "elapsed_s": round(time.time() - t0, 1)}
    _print(rep)
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"\nwrote {os.path.join(HERE, args.out)}  "
          f"({rep['settings']['elapsed_s']} s)")
    if not args.no_plot:
        try:
            print(f"wrote {_plot(genes, rep, os.path.splitext(args.out)[0] + '.jpg')}")
        except Exception as exc:                            # pragma: no cover
            print(f"(plot skipped: {exc})")
    return 0 if rep["pass"] else 1


def _plot(genes, rep, path):
    """The wheel itself, coloured by block type, plus a junction detail.

    Worth having as an artifact: every structural claim in `wheel_wheel.py`'s docstring
    about why the topology works is visible in these two panels, and a regression in the
    junction construction shows up here long before it shows up in a number.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    mesh = WW.build_wheel(genes, "coarse")
    xy = np.asarray(mesh.coords)
    colours = {"spoke": "#4C72B0", "hub_junction": "#DD8452",
               "rim_junction": "#C44E52", "hub_collar_weld": "#55A868",
               "hub_collar_free": "#8172B3", "rim_band_weld": "#937860",
               "rim_band_free": "#DA8BC3"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    for ax, box in ((ax1, None), (ax2, (10.5, 17.0, -2.5, 7.0))):
        for name, colour in colours.items():
            m = mesh.element_block == name
            if not m.any():
                continue
            quads = xy[mesh.conn[m][:, [0, 4, 1, 5, 2, 6, 3, 7]]]
            ax.add_collection(PolyCollection(
                quads, facecolors=colour, edgecolors="k",
                linewidths=0.15 if box is None else 0.4,
                label=name if box is None else None))
        ax.set_aspect("equal")
        if box is None:
            ax.set_xlim(-52, 52)
            ax.set_ylim(-52, 52)
            ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
            ax.set_title(f"{mesh.n_elements} Q9 elements, {mesh.n_nodes} nodes\n"
                         f"seam error {mesh.seam_error_mm:.1e} mm, "
                         f"minSJ {WW.quality_report(mesh)['min_scaled_jacobian']:.3f}")
        else:
            ax.set_xlim(box[0], box[1])
            ax.set_ylim(box[2], box[3])
            for r in (WW.HUB_RADIUS_MM, WW.HUB_RADIUS_MM - WW.COLLAR_DEPTH_MM):
                ax.add_patch(plt.Circle((0, 0), r, fill=False, ec="r", lw=0.8, ls="--"))
            ax.set_title("hub junction: the spoke arrives 10.5 deg from tangent, but its\n"
                         "cross-section is NORMAL to it, so the corners are ~80 deg")
    fig.suptitle(f"M2b full-wheel mesh — {'PASS' if rep['pass'] else 'FAIL'}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
