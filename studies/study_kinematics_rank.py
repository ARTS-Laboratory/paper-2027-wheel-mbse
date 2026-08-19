"""
=============================================================================
  KINEMATICS_PLAN.md STEP 1 — DOES `linear` RANK DESIGNS THE WAY SVK DOES?
=============================================================================
    .venv-opt/bin/python studies/study_kinematics_rank.py     (make kinrank)

THE ONE QUESTION.  `linear` is `wheel_fem`'s default strain measure and the correction at
service load is 22.75% (KINEMATICS_PLAN Step 0a, reproduced from §14/§31).  A 22.75%
absolute offset does not by itself condemn a SEARCH model: an optimizer does not need the
right deflection, it needs the right ORDERING and the right DESCENT DIRECTION.  So the
criterion this arc registered, BEFORE any of this ran (KINEMATICS_PLAN Step 0c), is:

    R1  ARGMIN IDENTITY, binary and primary.  The lowest-linear-loss genome must be the
        lowest-SVK-loss genome, over the pool AND over its feasible subset.
    R2  Spearman rho >= 0.90 on the FEASIBLE subset (binding) and reported on the full
        pool (diagnostic).  Top-5 sets equal under both orderings.
    R3  cos(grad_linear, grad_svk) >= 0.90 in the NORMALIZED gene space the descent steps
        in, at every probed genome.

R2's BINDING CASE IS THE FEASIBLE SUBSET, AND THAT IS NOT A CONVENIENCE.  `wheel_objective`
adds `soft_barrier(...)` terms that are 0.0 when satisfied and large when not, and a barrier
breach is overwhelmingly a GEOMETRY fact — `x_order`, `hub_overlap`, `fold`, `arrival`,
`fillet`, `fillet_cap` never touch the FEA at all.  Both kinematics therefore see almost the
same number for an infeasible design, and a pool dominated by infeasible genomes would return
rho ~ 1.0 while telling you nothing about the designs an optimizer actually chooses between.
The full-pool figure is reported so that this effect is visible rather than assumed.

WHY NOT JUST EXTEND `study_svk_rescore.py`.  That driver answers a different question — IS
THE SHIPPED GENOME FEASIBLE UNDER SVK — and its `--extra` hook exists so that the bare
`make svk` keeps reproducing SVK_PLAN Step 3's recorded artifact unchanged.  Widening its
GENOMES tuple to 36 rows would break that contract.  Its `_score` IS reused here, imported
rather than copied, so the term-set guard, the p=4 probe identity and the feasibility rule
are the same construction in both files and cannot drift apart.

MESHES ARE BUILT ONCE PER GENOME AND SHARED BY BOTH KINEMATICS, and `flank_orientation` is
pinned per genome, for `study_svk_rescore.py`'s reasons: a strain measure is a property of
the kernel, not the geometry, and letting a discrete branch be re-derived per column would
put a topology change inside a strain-measure comparison.

THE POOL IS EVERY DISTINCT COMMITTED GENOME IN THE TREE PLUS THE 15 STAGE-2 ELITES THAT ARE
NOT ALREADY ONE.  De-duplicated by gene vector, not by filename: `best_solution.json` and
`stage3_knee_best_medium.json` are the same wheel, as are `stage3_margin_best_medium.json`
and `stage3_margin_promote_best.json`, and `stage2_elites.json` rank 0 IS
`best_solution_ga_beam.json`.  A duplicate row would put a tied pair into a rank statistic
for a reason that is not about the wheel.
=============================================================================
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
from scipy import stats

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import wheel_fea as W
import wheel_genome as wg
import wheel_objective as WO
import wheel_pool as WP
import wheel_wheel as WW

import study_svk_rescore as SR

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = "coarse"
N_PHASE = 8

# REGISTERED IN KINEMATICS_PLAN.md STEP 0c BEFORE THIS FILE WAS RUN.  Not to be moved to
# fit the run: R1 is deliberately binary so there is nothing in it to loosen.
GATE_SPEARMAN = 0.90
GATE_GRAD_COSINE = 0.90

# Every distinct genome the tree commits to, in the order they were produced.  Files that
# hold the SAME gene vector are folded together by `_pool` and reported as aliases.
COMMITTED = (
    ("36aed36 ga_beam",      "best_solution_ga_beam.json"),
    ("elite9 prod",          "stage3_prod_best_elite9.json"),
    ("elite10 prod",         "stage3_prod_best_elite10.json"),
    ("minwall 0.8",          "stage3_minwall_best_0.8.json"),
    ("minwall 1.0",          "stage3_minwall_best_1.0.json"),
    ("350f4c7 minwall1.2",   "stage3_minwall_best_1.2.json"),
    ("minwall 1.4",          "stage3_minwall_best_1.4.json"),
    ("minwall 1.6",          "stage3_minwall_best_1.6.json"),
    ("minwall 1.8",          "stage3_minwall_best_1.8.json"),
    ("minwall 2.0",          "stage3_minwall_best_2.0.json"),
    ("minwall 2.2",          "stage3_minwall_best_2.2.json"),
    ("svk-shipped",          "stage3_svk_best_shipped.json"),
    ("svk-elite10",          "stage3_svk_best_elite10.json"),
    ("bc77614 svk-medium",   "stage3_svk_best_medium.json"),
    ("e4219f3 buildcap",     "stage3_buildcap_best_medium.json"),
    ("buildcap2",            "stage3_buildcap2_best_medium.json"),
    ("margin probe",         "stage3_margin_probe_best.json"),
    ("e126cc3 margin",       "stage3_margin_best_medium.json"),
    ("promote check",        "stage3_promote_best.json"),
    ("promote2 check",       "stage3_promote2_best.json"),
    ("09e8188 SHIPPED",      "best_solution.json"),
)

# The genomes R3 probes.  Four, not all 36: a value+grad call is the expensive one and the
# question R3 asks is about the DESCENT, so the points that matter are the ones a descent
# actually sat on — the shipped genome, the incumbent it replaced, the design the linear
# ranking prefers, and the GA/beam control whose correction is 5.5x smaller.
GRAD_PROBES = (
    ("09e8188 SHIPPED",    "best_solution.json"),
    ("e126cc3 margin",     "stage3_margin_best_medium.json"),
    ("350f4c7 minwall1.2", "stage3_minwall_best_1.2.json"),
    ("36aed36 ga_beam",    "best_solution_ga_beam.json"),
)


def _genes(d):
    """BY NAME, not by dict order.

    `study_svk_rescore.load_genes` takes `.values()`, which is correct only while every
    artifact happens to serialise its genes in `GENE_NAMES` order.  This pool reaches into
    `stage2_elites.json` as well, so the ordering is validated rather than assumed —
    `genes_to_vector` raises on a missing or extra key.
    """
    return np.asarray(wg.genes_to_vector(d), dtype=float)


def _genes_from_file(path):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return _genes(json.load(fh)["genes"])


def _key(v):
    return hashlib.sha1(repr(tuple(round(float(x), 12) for x in v)).encode()).hexdigest()[:7]


def _pool(include_elites=True):
    """`[(label, genes, aliases)]`, de-duplicated by GENE VECTOR.

    A tie in the input is not a tie in the wheel, and a rank statistic cannot tell the
    difference.  Aliases are kept so the report says which files collapsed.
    """
    rows, index = [], {}
    for label, path in COMMITTED:
        full = os.path.join(PP.ROOT, path)
        if not os.path.exists(full):
            print(f"  MISSING {path}", flush=True)
            continue
        v = _genes_from_file(path)
        k = _key(v)
        if k in index:
            rows[index[k]][2].append(f"{label} ({path})")
            continue
        index[k] = len(rows)
        rows.append([f"{label}", v, [path]])
    if include_elites:
        with open(os.path.join(PP.ROOT, "stage2_elites.json")) as fh:
            for el in json.load(fh)["elites"]:
                v = _genes(el["genes"])
                k = _key(v)
                if k in index:
                    rows[index[k]][2].append(f"stage2 elite {el['rank']}")
                    continue
                index[k] = len(rows)
                rows.append([f"elite{el['rank']} {el['genome_hash']}", v, ["stage2_elites.json"]])
    return [(a, b, c) for a, b, c in rows]


# ---------------------------------------------------------------------------
# R1 / R2 — the ranking
# ---------------------------------------------------------------------------

def run_rank(pool, cfg=DEFAULT_CONFIG, n_phase=N_PHASE, workers=0):
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    p = WP.PhasePool(workers) if workers else None
    rows = []
    try:
        for i, (label, genes, aliases) in enumerate(pool):
            orientation = tuple(float(o) for o in
                                WW.flank_orientation(genes, WW.get_config(cfg)))
            t0 = time.time()
            wanted = phases[:1] if p is not None else phases
            meshes = WO.phase_meshes(genes, cfg, wanted, orientation=orientation)
            row = {"genome": label, "aliases": aliases, "gene_key": _key(genes),
                   "mesh_s": round(time.time() - t0, 1)}
            for kin in ("linear", "svk"):
                t1 = time.time()
                # A genome that will not solve is a finding, not a reason to lose the run:
                # 36 genomes is over an hour and one NewtonDivergedError must not take the
                # other 35 with it.  Recorded and excluded from the statistics, loudly.
                try:
                    row[kin] = SR._score(genes, cfg, phases, meshes, kin,
                                         pool=p, orientation=orientation)
                    row[kin]["elapsed_s"] = round(time.time() - t1, 1)
                except Exception as exc:                       # noqa: BLE001
                    row[kin] = {"failed": f"{type(exc).__name__}: {exc}"}
                    print(f"  [{i + 1}/{len(pool)}] {label:<22} {kin:<6} FAILED {exc}",
                          flush=True)
                    continue
                s = row[kin]
                print(f"  [{i + 1}/{len(pool)}] {label:<22} {kin:<6} "
                      f"loss {s['loss']:12.4f} drop {s['axle_drop_mean_mm']:7.4f} "
                      f"util {s['stress_utilisation']:6.3f} "
                      f"{'FEAS' if s['feasible'] else 'infeas'} "
                      f"({s['elapsed_s']} s)", flush=True)
            if all("failed" not in row[k] for k in ("linear", "svk")):
                row["loss_rel_diff"] = float(row["svk"]["loss"] / row["linear"]["loss"] - 1.0)
                row["drop_rel_diff"] = float(row["svk"]["axle_drop_mean_mm"]
                                             / row["linear"]["axle_drop_mean_mm"] - 1.0)
            rows.append(row)
    finally:
        if p is not None:
            p.close()
    return {"config": cfg, "n_phase": n_phase, "scheme": "uniform", "workers": workers,
            "rows": rows, "verdict": _verdict(rows)}


def _subsets(rows):
    """`{name: [row]}` — the two pools R2 is evaluated on.

    `feasible` is FEASIBLE UNDER BOTH kinematics.  Taking either one alone would let the
    subset itself be chosen by the thing under test.
    """
    ok = [r for r in rows if "failed" not in r["linear"] and "failed" not in r["svk"]]
    return {"full": ok,
            "feasible": [r for r in ok
                         if r["linear"]["feasible"] and r["svk"]["feasible"]]}


def _rank_block(rows, name):
    if len(rows) < 3:
        return {"subset": name, "n": len(rows), "insufficient": True}
    lin = np.array([r["linear"]["loss"] for r in rows])
    svk = np.array([r["svk"]["loss"] for r in rows])
    rho = stats.spearmanr(lin, svk)
    tau = stats.kendalltau(lin, svk)
    order_l = [rows[i]["genome"] for i in np.argsort(lin)]
    order_s = [rows[i]["genome"] for i in np.argsort(svk)]
    k = min(5, len(rows))
    top_l, top_s = order_l[:k], order_s[:k]
    # THE INVERSION COUNT is over unordered pairs and is the raw form of the same fact rho
    # summarises — reported because "rho = 0.94" and "9 of 120 pairs are the wrong way
    # round" land very differently on a reader deciding whether to trust a search model.
    n_inv = int(sum(1 for i in range(len(rows)) for j in range(i + 1, len(rows))
                    if (lin[i] - lin[j]) * (svk[i] - svk[j]) < 0.0))
    n_pairs = len(rows) * (len(rows) - 1) // 2
    return {
        "subset": name, "n": len(rows),
        "spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue),
        "kendall_tau": float(tau.statistic),
        "order_linear": order_l, "order_svk": order_s,
        "top5_linear": top_l, "top5_svk": top_s,
        "top5_sets_equal": bool(set(top_l) == set(top_s)),
        "argmin_linear": order_l[0], "argmin_svk": order_s[0],
        "argmin_identical": bool(order_l[0] == order_s[0]),
        "top2_inverted": bool(len(rows) >= 2 and set(order_l[:2]) == set(order_s[:2])
                              and order_l[:2] != order_s[:2]),
        "discordant_pairs": n_inv, "n_pairs": n_pairs,
        "discordant_fraction": float(n_inv / n_pairs) if n_pairs else 0.0,
        "r2_pass": bool(float(rho.statistic) >= GATE_SPEARMAN
                        and set(top_l) == set(top_s)),
    }


def _verdict(rows):
    blocks = {name: _rank_block(rs, name) for name, rs in _subsets(rows).items()}
    binding = blocks["feasible"]
    full = blocks["full"]
    r1 = None
    if not full.get("insufficient"):
        r1 = full["argmin_identical"]
        if not binding.get("insufficient"):
            r1 = r1 and binding["argmin_identical"]
    return {"blocks": blocks, "gate_spearman": GATE_SPEARMAN,
            "r1_argmin_identity": r1,
            "r2_rank_agreement": (None if binding.get("insufficient")
                                  else binding["r2_pass"])}


# ---------------------------------------------------------------------------
# R3 — the descent direction
# ---------------------------------------------------------------------------

def run_gradients(cfg=DEFAULT_CONFIG, n_phase=N_PHASE, workers=0, probes=GRAD_PROBES):
    """cos(grad_linear, grad_svk) IN NORMALIZED GENE SPACE, which is where the descent lives.

    `wheel_stage3` steps in the unit box (`objective(..., normalized=True)`), and the chain
    rule that gets there multiplies each component by that gene's range — `cy` spans 64 mm
    and `R_rim` spans 2.5.  A cosine taken in PHYSICAL units would be dominated by whichever
    genes happen to have wide boxes and would not be the angle any optimizer ever sees.

    The two calls SHARE MESHES and a pinned `flank_orientation`, for `run_rank`'s reasons.
    """
    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    p = WP.PhasePool(workers) if workers else None
    rows = []
    try:
        for label, path in probes:
            full = os.path.join(PP.ROOT, path)
            if not os.path.exists(full):
                continue
            genes = _genes_from_file(path)
            z = (genes - low) / (high - low)
            orientation = tuple(float(o) for o in
                                WW.flank_orientation(genes, WW.get_config(cfg)))
            wanted = phases[:1] if p is not None else phases
            meshes = WO.phase_meshes(genes, cfg, wanted, orientation=orientation)
            g = {}
            for kin in ("linear", "svk"):
                t0 = time.time()
                val, grad, _ = WO.objective(z, cfg, normalized=True, phases=phases,
                                            meshes=meshes, pool=p, orientation=orientation,
                                            kinematics=kin)
                g[kin] = {"loss": float(val), "grad": np.asarray(grad, dtype=float),
                          "elapsed_s": round(time.time() - t0, 1)}
            a, b = g["linear"]["grad"], g["svk"]["grad"]
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")
            row = {
                "genome": label, "file": path,
                "loss_linear": g["linear"]["loss"], "loss_svk": g["svk"]["loss"],
                "grad_norm_linear": na, "grad_norm_svk": nb,
                "grad_norm_ratio": float(nb / na) if na > 0 else float("nan"),
                "cosine": cos,
                "angle_deg": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
                # A DIRECTION IS ONLY USEFUL IF IT DESCENDS THE OTHER LOSS.  cos > 0 is the
                # weak form of the question and it is reported separately from the 0.90 bar,
                # because they fail for different reasons and mean different things: cos > 0
                # says a linear step still reduces the SVK loss for SOME step size; cos >=
                # 0.90 says the two optimizers would walk the same way.
                "linear_step_descends_svk": bool(cos > 0.0),
                # Per-gene sign flips: which genes the two models want moved OPPOSITE ways.
                "sign_flips": [wg.GENE_NAMES[i] for i in range(len(a))
                               if a[i] * b[i] < 0.0],
                "elapsed_s": {k: g[k]["elapsed_s"] for k in g},
                "r3_pass": bool(cos >= GATE_GRAD_COSINE),
            }
            rows.append(row)
            print(f"  grad {label:<22} cos {cos:+.4f} ({row['angle_deg']:5.1f} deg)  "
                  f"|g| {na:9.3f} -> {nb:9.3f}  flips {len(row['sign_flips'])}",
                  flush=True)
    finally:
        if p is not None:
            p.close()
    return {"config": cfg, "n_phase": n_phase, "workers": workers,
            "gate_cosine": GATE_GRAD_COSINE, "rows": rows,
            "r3_pass": bool(rows) and all(r["r3_pass"] for r in rows)}


# ---------------------------------------------------------------------------

def _print(rep):
    rk = rep.get("rank")
    if rk:
        print("\n" + "=" * 100)
        print(f"  STEP 1 — LINEAR vs SVK RANKING, {rk['config']}, {rk['n_phase']} phases")
        print("=" * 100)
        print(f"  {'genome':<22} {'lin loss':>12} {'svk loss':>12} {'d loss':>9} "
              f"{'lin drop':>9} {'svk drop':>9} {'lin':>6} {'svk':>6}")
        for r in rk["rows"]:
            if "failed" in r["linear"] or "failed" in r["svk"]:
                print(f"  {r['genome']:<22} FAILED  "
                      f"{r['linear'].get('failed', '')} {r['svk'].get('failed', '')}")
                continue
            print(f"  {r['genome']:<22} {r['linear']['loss']:12.4f} "
                  f"{r['svk']['loss']:12.4f} {100 * r['loss_rel_diff']:8.1f}% "
                  f"{r['linear']['axle_drop_mean_mm']:9.4f} "
                  f"{r['svk']['axle_drop_mean_mm']:9.4f} "
                  f"{'FEAS' if r['linear']['feasible'] else 'infs':>6} "
                  f"{'FEAS' if r['svk']['feasible'] else 'infs':>6}")

        for name in ("full", "feasible"):
            b = rk["verdict"]["blocks"][name]
            print(f"\n  --- {name.upper()} POOL "
                  f"({'BINDING for R2' if name == 'feasible' else 'diagnostic'}) ---")
            if b.get("insufficient"):
                print(f"      only {b['n']} rows — no rank statistic")
                continue
            print(f"      n = {b['n']}   Spearman rho = {b['spearman_rho']:+.4f} "
                  f"(gate {GATE_SPEARMAN})   Kendall tau = {b['kendall_tau']:+.4f}")
            print(f"      discordant pairs {b['discordant_pairs']}/{b['n_pairs']} "
                  f"({100 * b['discordant_fraction']:.1f}%)")
            print(f"      top5 linear : {b['top5_linear']}")
            print(f"      top5 svk    : {b['top5_svk']}")
            print(f"      top5 sets equal: {b['top5_sets_equal']}"
                  f"   argmin identical: {b['argmin_identical']}"
                  f"   ({b['argmin_linear']} vs {b['argmin_svk']})")

    gr = rep.get("gradients")
    if gr:
        print("\n" + "=" * 100)
        print(f"  R3 — DESCENT DIRECTION, normalized gene space, {gr['config']}")
        print("=" * 100)
        print(f"  {'genome':<22} {'cos':>9} {'angle':>8} {'|g| lin':>11} {'|g| svk':>11} "
              f"{'ratio':>7}  sign flips")
        for r in gr["rows"]:
            print(f"  {r['genome']:<22} {r['cosine']:+9.4f} {r['angle_deg']:7.1f}d "
                  f"{r['grad_norm_linear']:11.3f} {r['grad_norm_svk']:11.3f} "
                  f"{r['grad_norm_ratio']:7.2f}  "
                  f"{','.join(r['sign_flips']) if r['sign_flips'] else '-'}")

    v = rep["registered_criterion"]
    print("\n" + "=" * 100)
    print("  THE REGISTERED CRITERION (KINEMATICS_PLAN.md Step 0c, written before this ran)")
    print("=" * 100)
    for k in ("R1_argmin_identity", "R2_rank_agreement", "R3_descent_direction"):
        got = v[k]
        print(f"    {k:<22} {'PASS' if got else ('FAIL' if got is False else 'n/a')}")
    print(f"\n  LINEAR IS AN ACCEPTABLE DEFAULT FOR SEARCH: "
          f"{'YES' if v['linear_is_acceptable'] else 'NO'}")


def main():
    ap = argparse.ArgumentParser(description="KINEMATICS_PLAN Step 1 — the ranking test")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--n-phase", type=int, default=N_PHASE)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="study_kinematics_rank.json")
    ap.add_argument("--no-elites", action="store_true",
                    help="committed genomes only; the 15 stage-2 elites are GA/beam-era "
                         "2.0 mm designs and most land infeasible under today's objective")
    ap.add_argument("--skip-rank", action="store_true")
    ap.add_argument("--skip-grad", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rep = {}
    if not args.skip_rank:
        pool = _pool(include_elites=not args.no_elites)
        print(f"pool: {len(pool)} distinct genomes", flush=True)
        rep["rank"] = run_rank(pool, args.config, args.n_phase, args.workers)
        # WRITTEN AS SOON AS IT EXISTS, before the gradients and before `_print`:
        # `study_svk_rescore.py`'s rule, and this run is over an hour.
        with open(os.path.join(HERE, args.out), "w") as fh:
            json.dump(rep, fh, indent=1, default=float)
    if not args.skip_grad:
        rep["gradients"] = run_gradients(args.config, args.n_phase, args.workers)

    rk = rep.get("rank", {}).get("verdict", {})
    gr = rep.get("gradients", {})
    r1 = rk.get("r1_argmin_identity")
    r2 = rk.get("r2_rank_agreement")
    r3 = gr.get("r3_pass") if gr else None
    rep["registered_criterion"] = {
        "R1_argmin_identity": r1, "R2_rank_agreement": r2, "R3_descent_direction": r3,
        # R1 IS THE VERDICT.  R2 and R3 qualify it and cannot rescue it: a model that
        # selects a different design has failed at the only job a search model has.
        "linear_is_acceptable": bool(r1 and r2 and r3),
        "gate_spearman": GATE_SPEARMAN, "gate_grad_cosine": GATE_GRAD_COSINE,
    }
    rep["settings"] = {"config": args.config, "n_phase": args.n_phase,
                       "workers": args.workers, "elites": not args.no_elites,
                       "elapsed_s": round(time.time() - t0, 1)}
    out = os.path.join(HERE, args.out)
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    _print(rep)
    print(f"\nwrote {out}  ({rep['settings']['elapsed_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
