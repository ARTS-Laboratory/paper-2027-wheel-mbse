"""
=============================================================================
  STUDY — MESH QUALITY ACROSS THE DESIGN SPACE          (M2 gate)
=============================================================================
Runs BEFORE the FE solver exists, on purpose.

A mesh that inverts for some genomes does not announce itself: the solve still returns
numbers, the optimizer still descends, and the answer is quietly wrong in a region of
the design space nobody looked at.  The optimizer will then find that region, because
a negative-Jacobian element contributes negative stiffness and looks like free
compliance.  So the acceptance criterion is fixed in advance:

    > 98 % of the FEASIBLE design space must have minimum scaled Jacobian > 0.2

WHAT "FEASIBLE" HAS TO MEAN, AND WHY THAT IS THE FINDING
--------------------------------------------------------
Two definitions are reported side by side, because which one is right is the question
this study answers rather than assumes:

  `feasible_geom`  what `evaluate_design` already enforces — x-ordering and hub
                   crowding (wheel_fea.py:550).
  `meshable`       that, plus a positive fold margin: min(R_curvature - t/2) > 0.

Under the first definition the gate fails outright — roughly half of the genomes the
existing constraints admit have inverted elements.  That is not a meshing defect: the
offset band genuinely turns inside out when the outward offset passes the centre of
curvature, and the mesh is faithfully reporting it.  The existing constraint set is
simply incomplete, and `wheel_fea`'s `smoothness` term was only ever an indirect proxy
for the missing one (wheel_fea.py:596-598 says as much).

So the study also measures whether the closed-form margin PREDICTS inversion.  If it
does, the optimizer can be kept out of the folded region by an analytic barrier costing
one curvature evaluation, with no mesh in the loop — which is the cheapest possible
form of the fix.

    .venv-opt/bin/python studies/study_mesh_quality.py --samples 2000

Writes `study_mesh_quality.json` and, with matplotlib available, a diagnostic figure.
=============================================================================
"""

import argparse
import json
import os

import project_paths as PP
import time

import numpy as np

import wheel_fea as W
import wheel_genome as GN
import wheel_mesh as M

HERE = os.path.dirname(os.path.abspath(__file__))

MIN_SJ_ACCEPT = 0.2          # scaled-Jacobian floor for a usable element
ACCEPT_FRACTION = 0.98       # of the feasible space


def latin_hypercube(n, low, high, seed=0):
    """Stratified sample of the gene box.

    Latin hypercube rather than uniform: with 14 dimensions and only a couple of
    thousand samples, plain uniform sampling leaves large unvisited slabs in every
    single coordinate, which is exactly where an untested genome would come from.
    """
    try:
        from scipy.stats import qmc
        u = qmc.LatinHypercube(d=len(low), seed=seed).random(n)
    except ImportError:                                    # pragma: no cover
        rng = np.random.default_rng(seed)
        u = (rng.permuted(np.tile(np.arange(n), (len(low), 1)), axis=1).T
             + rng.random((n, len(low)))) / n
    return low + u * (high - low)


def fold_margin(vec, cfg):
    """min(radius of curvature - half thickness), the closed-form fold predictor.

    Negative means the outward offset has passed the centre of curvature, so the flank
    turns inside out.  Computable without meshing, which is the point: if it predicts
    mesh inversion reliably it can serve directly as the optimizer's barrier.
    """
    import wheel_geometry as G
    curve, ctrl = G.bezier_centerline(*[vec[i] for i in range(8)],
                                      span_mm=W.HUB_RIM_SPAN_MM,
                                      num_points=cfg.n_curve)
    return float(G.self_intersection_margin(curve, ctrl, *[vec[i] for i in range(8, 12)],
                                            num_points=cfg.n_curve))


def evaluate_one(vec, cfg, conn):
    """Mesh one genome and record everything the gate needs.

    Two notions of feasible, kept separate on purpose:

    `feasible_geom` is what `evaluate_design` already enforces — x-ordering and hub
    crowding (wheel_fea.py:550).  `meshable` adds the fold constraint.  Reporting both
    is what shows whether the existing constraint set is sufficient, which is the
    question this study actually exists to answer.
    """
    _, loss = W.evaluate_design(vec)
    feasible_geom = loss["x_order"] == 0.0 and loss["hub_overlap"] == 0.0
    margin = fold_margin(vec, cfg)
    X = M.spoke_block_coords_from_vector(vec, cfg, W.HUB_RIM_SPAN_MM)
    q = M.quality_report(M.flatten(X), conn)
    return {
        "feasible_geom": bool(feasible_geom),
        "meshable": bool(feasible_geom and margin > 0.0),
        "min_sj": q["min_scaled_jacobian"],
        "n_inverted": q["n_inverted"],
        "max_ar": q["max_aspect_ratio"],
        "area": q["total_area_mm2"],
        "fold_margin": margin,
        "x_order": loss["x_order"],
        "hub_overlap": loss["hub_overlap"],
        "genes": [float(g) for g in vec],
    }


def sweep(target_feasible, cfg_name, seed, extra_vectors=(), max_draws=400_000):
    """Draw until `target_feasible` GEOMETRICALLY FEASIBLE genomes have been meshed.

    Targeting feasible samples rather than raw draws is what gives the gate its
    statistical power: only a few percent of the raw gene box satisfies x-ordering, so
    a fixed raw budget would leave far too few feasible samples to resolve a 2%
    criterion.
    """
    cfg = M.get_config(cfg_name)
    conn = M.spoke_block_connectivity(cfg)
    low, high, _ = GN.bounds_arrays(W.GENE_SPACE)

    rows, n_feasible, draws, batch = [], 0, 0, 0
    t0 = time.perf_counter()
    while n_feasible < target_feasible and draws < max_draws:
        for vec in latin_hypercube(4096, low, high, seed=seed + batch):
            row = evaluate_one(vec, cfg, conn)
            draws += 1
            rows.append(row)
            n_feasible += row["feasible_geom"]
            if n_feasible >= target_feasible:
                break
        batch += 1
        print(f"    {draws} drawn, {n_feasible} feasible "
              f"({time.perf_counter() - t0:.1f}s)")

    for vec in extra_vectors:
        rows.append(evaluate_one(np.asarray(vec, dtype=float), cfg, conn))
    return cfg, rows


def _fraction_above_floor(rows):
    if not rows:
        return 0.0, np.array([])
    sj = np.array([r["min_sj"] for r in rows])
    return float((sj > MIN_SJ_ACCEPT).mean()), sj


def summarize(cfg, rows):
    geom = [r for r in rows if r["feasible_geom"]]
    mesh = [r for r in rows if r["meshable"]]

    frac_geom, _ = _fraction_above_floor(geom)
    frac_mesh, sj_mesh = _fraction_above_floor(mesh)

    # Does the closed-form margin predict mesh inversion?  If it does, the optimizer
    # can be kept out of the folded region for the cost of one analytic evaluation,
    # with no mesh in the loop.
    pred = np.array([r["fold_margin"] < 0.0 for r in geom])
    bad = np.array([r["min_sj"] <= MIN_SJ_ACCEPT for r in geom])
    confusion = {
        "true_positive": int((pred & bad).sum()),
        "false_positive": int((pred & ~bad).sum()),
        "false_negative": int((~pred & bad).sum()),
        "true_negative": int((~pred & ~bad).sum()),
    }

    # A margin of exactly 0 is the fold point, but element quality is already degrading
    # as it is approached.  Sweep the threshold to find what the barrier should actually
    # require, rather than assuming the mathematical boundary is the useful one.
    all_m = np.array([r["fold_margin"] for r in geom])
    all_sj = np.array([r["min_sj"] for r in geom])
    thresholds = []
    for th in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        rejected = all_m < th
        bad = all_sj <= MIN_SJ_ACCEPT
        kept = ~rejected
        thresholds.append({
            "threshold_mm": th,
            "missed_bad_meshes": int((kept & bad).sum()),
            "false_alarms": int((rejected & ~bad).sum()),
            "designs_kept": int(kept.sum()),
            "frac_above_floor_among_kept":
                float((all_sj[kept] > MIN_SJ_ACCEPT).mean()) if kept.any() else 0.0,
        })

    summary = {
        "config": cfg.name,
        "fold_margin_threshold_sweep": thresholds,
        "n_span": cfg.n_span,
        "n_thick": cfg.n_thick,
        "order": cfg.order,
        "n_samples": len(rows),
        "n_feasible_geom": len(geom),
        "n_meshable": len(mesh),
        "geom_feasible_fraction": len(geom) / len(rows),
        "min_sj_accept": MIN_SJ_ACCEPT,
        "accept_fraction_required": ACCEPT_FRACTION,
        # The gate, under both feasibility definitions.
        "frac_above_floor_geom_feasible": frac_geom,
        "frac_above_floor_meshable": frac_mesh,
        "passes_gate_geom_only": bool(frac_geom >= ACCEPT_FRACTION),
        "passes_gate_with_fold_constraint": bool(frac_mesh >= ACCEPT_FRACTION),
        "fold_margin_predicts_inversion": confusion,
        "n_inverted_any": int(sum(r["n_inverted"] > 0 for r in mesh)),
    }
    if sj_mesh.size:
        summary["sj_percentiles_meshable"] = {
            p: float(np.percentile(sj_mesh, p)) for p in (0, 1, 5, 25, 50)
        }
        summary["worst_sj_meshable"] = float(sj_mesh.min())
        summary["max_aspect_over_meshable"] = float(max(r["max_ar"] for r in mesh))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=2000,
                    help="target number of GEOMETRICALLY FEASIBLE genomes to mesh")
    ap.add_argument("--config", default="coarse")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="study_mesh_quality.json")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--rows", metavar="PATH",
                    help="also dump every sampled genome and its metrics here.  Off by "
                         "default: at the gate's 2000 feasible genomes that is 45769 "
                         "raw draws and 28 MB of JSON, and it is exactly reproducible "
                         "from --seed, so it is regenerated rather than committed.  "
                         "The summary is the artifact.")
    args = ap.parse_args()

    print("=" * 70)
    print(f"  MESH QUALITY SWEEP — {args.samples} feasible genomes, "
          f"config {args.config!r}")
    print(f"  gate: >{ACCEPT_FRACTION:.0%} of the feasible space with "
          f"minSJ > {MIN_SJ_ACCEPT}")
    print("=" * 70)

    # Always include the shipped design, so a regression there can never be averaged
    # away by a couple of thousand random samples.
    rec = GN.load_record(os.path.join(PP.ROOT, "best_solution.json"))
    shipped = GN.genes_to_vector(rec["genes"])

    cfg, rows = sweep(args.samples, args.config, args.seed, extra_vectors=[shipped])
    summary = summarize(cfg, rows)

    print(f"\n  drawn                        : {summary['n_samples']}")
    print(f"  geometrically feasible       : {summary['n_feasible_geom']}"
          f"  ({summary['geom_feasible_fraction']:.1%} of draws)")
    print(f"  ...of which meshable         : {summary['n_meshable']}"
          f"  (fold margin > 0)")

    c = summary["fold_margin_predicts_inversion"]
    print(f"\n  Does the closed-form fold margin predict mesh inversion?")
    print(f"    margin<0 and minSJ<=floor  : {c['true_positive']:6d}")
    print(f"    margin<0 but mesh fine     : {c['false_positive']:6d}   (false alarm)")
    print(f"    margin>=0 but mesh bad     : {c['false_negative']:6d}   "
          f"<-- UNEXPLAINED, these would be silent")
    print(f"    margin>=0 and mesh fine    : {c['true_negative']:6d}")

    print(f"\n  minSJ > {MIN_SJ_ACCEPT} over...")
    print(f"    geometric feasibility only : "
          f"{summary['frac_above_floor_geom_feasible']:.2%}  -> "
          f"{'PASS' if summary['passes_gate_geom_only'] else 'FAIL'}")
    print(f"    + the fold constraint      : "
          f"{summary['frac_above_floor_meshable']:.2%}  -> "
          f"{'PASS' if summary['passes_gate_with_fold_constraint'] else 'FAIL'}")
    if "sj_percentiles_meshable" in summary:
        print(f"\n  meshable minSJ percentiles : "
              + "  ".join(f"p{p}={v:.3f}"
                          for p, v in summary["sj_percentiles_meshable"].items()))
        print(f"  worst minSJ (meshable)     : {summary['worst_sj_meshable']:.4f}")
        print(f"  max aspect ratio           : {summary['max_aspect_over_meshable']:.1f}")

    print(f"\n  What should the barrier actually require?")
    print(f"    {'margin >':>10s} {'missed':>7s} {'false alarm':>12s} "
          f"{'kept':>6s} {'minSJ ok':>9s}")
    for t in summary["fold_margin_threshold_sweep"]:
        print(f"    {t['threshold_mm']:10.2f} {t['missed_bad_meshes']:7d} "
              f"{t['false_alarms']:12d} {t['designs_kept']:6d} "
              f"{t['frac_above_floor_among_kept']:8.2%}")

    print(f"\n  GATE: "
          f"{'PASS' if summary['passes_gate_with_fold_constraint'] else 'FAIL'}"
          f"  (with the fold constraint in the feasible set)")

    out = os.path.join(HERE, args.out)
    with open(out, "w") as fh:
        json.dump({"summary": summary,
                   "shipped_genome_hash": rec["_hash"],
                   "provenance": {"samples": args.samples, "config": args.config,
                                  "seed": args.seed, "n_rows": len(rows)}},
                  fh, indent=1)
    print(f"  wrote {out}")

    if args.rows:
        with open(args.rows, "w") as fh:
            json.dump(rows, fh)
        print(f"  wrote {args.rows}  ({len(rows)} rows)")

    if not args.no_plot:
        try:
            _plot(rows, summary, os.path.splitext(out)[0] + ".jpg")
        except Exception as exc:                            # pragma: no cover
            print(f"  (plot skipped: {exc})")

    return 0 if summary["passes_gate_with_fold_constraint"] else 1


def _plot(rows, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feas = [r for r in rows if r["meshable"]]
    sj = np.array([r["min_sj"] for r in feas])
    fm = np.array([r["fold_margin"] for r in feas])
    ar = np.array([r["max_ar"] for r in feas])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#ccc")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.xaxis.label.set_color("#ccc")
        ax.yaxis.label.set_color("#ccc")
        ax.title.set_color("#eee")
        ax.grid(alpha=0.15, color="white")

    ax = axes[0]
    ax.hist(sj, bins=60, color="#00e5ff")
    ax.axvline(MIN_SJ_ACCEPT, color="#ff6e40", ls="--", lw=2, label="floor 0.2")
    ax.set_yscale("log")
    ax.set_xlabel("min scaled Jacobian")
    ax.set_ylabel("feasible genomes")
    ax.set_title(f"Mesh validity ({summary['frac_above_floor_meshable']:.2%} above floor)")
    ax.legend(facecolor="#222", labelcolor="#ccc")

    ax = axes[1]
    ax.scatter(fm, sj, s=4, alpha=0.35, color="#76ff03")
    ax.axhline(MIN_SJ_ACCEPT, color="#ff6e40", ls="--", lw=1.5)
    ax.axvline(0.0, color="#e0e0e0", ls=":", lw=1.5)
    ax.set_xlabel("fold margin: min(R_curv - t/2)  [mm]")
    ax.set_ylabel("min scaled Jacobian")
    ax.set_title("Does the closed-form margin predict mesh failure?")

    ax = axes[2]
    ax.scatter(ar, sj, s=4, alpha=0.35, color="#ffd54f")
    ax.axhline(MIN_SJ_ACCEPT, color="#ff6e40", ls="--", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("max aspect ratio")
    ax.set_ylabel("min scaled Jacobian")
    ax.set_title("Aspect ratio vs shape quality (independent)")

    fig.suptitle("Spoke-block mesh quality over the design space",
                 color="#eee", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"  wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
