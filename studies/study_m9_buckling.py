"""M9 Phase 3 — the generalised buckling load factor.

WHY THIS EXISTS RATHER THAN A SECTION IN `study_m9.py`.  That study measured
`lambda_min(K_t)` and FAILED, and its FAIL is a deliverable artifact that must stay
reproducible.  It also turned out to be measuring something else entirely:
`wheel_contact_problem` defaults to `kinematics="linear"`, so `prob.nonlinear` is False and
the displacement threaded into `assemble_stiffness` is IGNORED -- measured at exactly
0.000e+00.  The quantity there is `lambda_min(K_linear + K_contact)`, which contains no
geometric stiffening by construction, which is why it scales as h^2 and is nearly
indifferent to load.

WHAT THIS MEASURES INSTEAD.  Classical linear buckling: with `K_0 = K(u=0)` and
`K_t = K(u_service)` both assembled under SVK kinematics, `K_g = K_t - K_0` is the
geometric term, and `det(K_0 + lambda*K_g) = 0` gives a dimensionless LOAD FACTOR -- how
many times the service load until the tangent goes singular.  Solved as the generalised
symmetric problem `K_g x = mu K_0 x` with `lambda = -1/mu`; the smallest positive lambda is
the most negative mu, hence `which="SA"`.

THE STATE MUST BE SOLVED UNDER SVK TOO, AND THAT IS NOT A DETAIL.  Assembling an SVK
stiffness at a displacement that was converged under LINEAR kinematics gives a different
and larger answer -- measured, on `best_solution` at phase 0:

    smoke   linear state 1.800046   svk state 1.378129
    coarse  linear state 1.785253   svk state 1.359846

The svk-state column is the one that converges under refinement and the one every number
in PLAN.md refers to.  Using the linear state is the same class of error `study_m9` made.

STILL MEASUREMENT-ONLY.  Nothing here is added to the Stage-3 objective, `buckling` stays
inert, and no threshold is invented -- `LOBPCG_RESIDUAL_REL` is deliberately left alone.
Calibrating a gate against a quantity whose independent corroboration is one of this
study's own open sections is how `stress_scale` happened.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import project_paths as PP  # noqa: E402
if PP.SRC not in sys.path:
    sys.path.insert(0, PP.SRC)

import jax_config  # noqa: E402,F401  -- x64 before the first trace
import wheel_fem as fem  # noqa: E402
import wheel_genome as wg  # noqa: E402
import wheel_wheel as WW  # noqa: E402
from wheel_fea import TOTAL_FORCE_NEWTONS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PERIOD_DEG = 30.0
REFERENCE_PHASES = 13
PRODUCTION_PHASES = 4

# Same constant `study_m9` gates its mesh ladder on, for the same reason: a quantity whose
# successive-refinement change exceeds this has no mesh-independent value and cannot carry
# a threshold.  lambda_min(K_t) misses it by twelve; the load factor clears it by ~21x.
GATE_MESH_REL = 0.05

# The load ramp in `run_corroborate`.  Straddles the predicted critical factor on both
# sides on purpose -- a ramp that stops below it can only ever agree.
#
# IT RUNS TO 4x BECAUSE 2x WAS NOT ENOUGH TO SETTLE THE QUESTION.  The first run stopped at
# 2.0 and showed the implied critical total still rising (0.9946 -> 2.2140), which says
# "not a limit point" but leaves open where the limit point actually is.  Extending it
# answers that: lambda(f) approaches 1 from ABOVE as roughly 1 + 0.43/f^2 and never
# crosses, so the fixed point lambda(f) = 1 is at infinity and there is no finite critical
# load in this formulation.  Measured rather than extrapolated, on purpose -- extrapolating
# a critical load from a fitted curve is how a number nobody can defend gets reported.
LOAD_RAMP = (0.5, 0.8, 1.0, 1.2, 1.36, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


def _genes(path):
    with open(path) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


def _designs(path):
    out = [("best_solution", _genes(path))]
    with open(PP.STAGE2_ELITES) as fh:
        elites = json.load(fh)["elites"]
    for row in elites:
        rank = int(row["rank"])
        if rank == 0:
            continue
        out.append((f"elite_{rank}", wg.genes_to_vector(row["genes"])))
    return out


def measure(genes, cfg, phase_deg=0.0, *, force=TOTAL_FORCE_NEWTONS,
            kinematics="svk", contact=False, n_modes=1):
    """One load factor, plus the state it was linearised about.

    `kinematics` and `contact` are exposed because they are exactly the two modelling
    choices PLAN.md lists as unestablished, and `run_variants` sweeps them.  The defaults
    are the formulation the mesh ladder converges under.
    """
    t0 = time.time()
    mesh = WW.build_wheel(genes, cfg, phase_deg=float(phase_deg))
    sec = fem.solve_wheel_contact(mesh, force=float(force), kinematics=kinematics)
    res = fem.solve_wheel_contact_at(mesh, sec["axle_drop_mm"],
                                     u_reduced0=sec["u_reduced"],
                                     kinematics=kinematics)
    prob = fem.wheel_contact_problem(mesh, indentation_mm=sec["axle_drop_mm"],
                                     kinematics=kinematics)

    kw = dict(order=prob.order, lam=prob.lam, mu=prob.mu, width=prob.width)
    # `nonlinear=True` REGARDLESS of how the state was solved: the geometric term is the
    # whole quantity, and the linear kernel's Hessian is independent of u (its own
    # docstring says so), so a linear assembly here returns K_g == 0 exactly.
    K0 = fem.assemble_stiffness(prob.coords, prob.conn, None, nonlinear=True, **kw)
    Kt = fem.assemble_stiffness(prob.coords, prob.conn, res["u"], nonlinear=True, **kw)
    if contact:
        C = prob.contact.stiffness(prob.coords, res["u"])
        K0, Kt = K0 + C, Kt + C
    Kg = Kt - K0

    K0r = (prob.T.T @ K0 @ prob.T).tocsc()
    Kgr = (prob.T.T @ Kg @ prob.T).tocsc()
    mus = spla.eigsh(Kgr, k=int(n_modes), M=K0r, which="SA",
                     return_eigenvectors=False)
    mus = np.sort(np.asarray(mus, dtype=float))          # most negative first
    mu = float(mus[0])
    return {
        "config": cfg,
        "phase_deg": float(phase_deg),
        "force_n": float(force),
        "kinematics": kinematics,
        "contact_in_operator": bool(contact),
        "mu": mu,
        "load_factor": float(-1.0 / mu) if mu else float("inf"),
        "mu_modes": mus.tolist(),
        "axle_drop_mm": float(sec["axle_drop_mm"]),
        "u_max_mm": float(np.abs(res["u"]).max()),
        "reduced_dof": int(K0r.shape[0]),
        "elapsed_s": round(time.time() - t0, 2),
    }


def _series(rows, key="load_factor"):
    vals = np.asarray([r[key] for r in rows if "error" not in r], dtype=float)
    if len(vals) < 2:
        return {"values": vals.tolist(), "last_pair_rel": None, "pass": True}
    rel = abs(vals[-1] / vals[-2] - 1.0) if vals[-2] else float("inf")
    return {"values": vals.tolist(), "last_pair_rel": float(rel),
            "pass": bool(np.isfinite(rel) and rel < GATE_MESH_REL)}


def _guarded(fn, **tag):
    try:
        return fn()
    except Exception as exc:                    # record refusals; never hide them
        return {**tag, "error": f"{type(exc).__name__}: {exc}"}


def run_mesh(designs, configs):
    """The claim that makes this quantity usable at all: it converges under refinement."""
    rows, series = [], {}
    for name, genes in designs:
        drows = []
        for cfg in configs:
            row = _guarded(lambda: measure(genes, cfg), config=cfg)
            row["design"] = name
            rows.append(row)
            drows.append(row)
        series[name] = _series(drows)
    return {"rows": rows, "series": series,
            "n_error_rows": sum(1 for r in rows if "error" in r),
            "pass": bool(all(v["pass"] for v in series.values())
                         and not any("error" in r for r in rows))}


def run_phase(designs, cfg, n_phase):
    """PLAN.md open item 1, half one: the load factor's phase spread is UNMEASURED.

    `study_m9` measured lambda_min's at 0.66-1.59% and concluded a 4-phase production
    stencil would do.  That conclusion does not transfer to a different quantity for free.
    """
    phases = np.linspace(0.0, PERIOD_DEG, n_phase, endpoint=False)
    production = set(np.linspace(0.0, PERIOD_DEG, PRODUCTION_PHASES, endpoint=False))
    rows, comparisons = [], []
    for name, genes in designs:
        ref = []
        for p in phases:
            row = _guarded(lambda: measure(genes, cfg, p), phase_deg=float(p))
            row["design"] = name
            ref.append(row)
        rows.extend(ref)
        ok = [r for r in ref if "error" not in r]
        if len(ok) < 2:
            comparisons.append({"design": name, "pass": False,
                                "note": "too few converged rows"})
            continue
        vals = np.asarray([r["load_factor"] for r in ok])
        pvals = np.asarray([r["load_factor"] for r in ok
                            if any(abs(r["phase_deg"] - p) < 1e-12 for p in production)])
        comparisons.append({
            "design": name,
            "reference_min": float(vals.min()), "reference_max": float(vals.max()),
            "reference_mean": float(vals.mean()), "reference_std": float(vals.std()),
            "spread_pct": float(100.0 * vals.std() / abs(vals.mean())),
            "production_min": float(pvals.min()) if pvals.size else None,
            "production_captures_min_pct": (
                float(100.0 * abs(pvals.min() - vals.min()) / abs(vals.mean()))
                if pvals.size else None),
            "pass": bool(np.all(np.isfinite(vals))),
        })
    return {"n_phase": int(n_phase), "period_deg": PERIOD_DEG, "rows": rows,
            "comparisons": comparisons,
            "pass": bool(all(c["pass"] for c in comparisons))}


def run_design_space(designs, cfg):
    """Open item 1, half two: is 1.36 this DESIGN or this WHEEL?"""
    rows = []
    for name, genes in designs:
        row = _guarded(lambda: measure(genes, cfg), design=name)
        row["design"] = name
        rows.append(row)
    ok = [r for r in rows if "error" not in r]
    vals = np.asarray([r["load_factor"] for r in ok], dtype=float)
    return {"rows": rows,
            "n_designs": len(rows), "n_converged": len(ok),
            "min": float(vals.min()) if vals.size else None,
            "max": float(vals.max()) if vals.size else None,
            "spread_ratio": float(vals.max() / vals.min()) if vals.size else None,
            "n_below_one": int((vals < 1.0).sum()) if vals.size else None,
            "pass": bool(len(ok) == len(rows))}


def run_corroborate(genes, cfg, ramp=LOAD_RAMP):
    """PLAN.md open item 2, AND THE REASON THIS STUDY EXISTS.

    "THE LOAD FACTOR IS ~1.36, AND THAT IS ALARMINGLY TIGHT ... Do not report 1.36 as a
    safety factor until it is checked against something independent."

    The check is self-consistency along a load ramp.  The load factor is measured AT the
    service state, so if it means what it claims, then measuring it at f x service should
    return roughly `lambda_cr / f` -- and the PRODUCT `f * lambda(f)` should be flat at the
    true critical factor.  A quantity that is really just reporting stiffness will not do
    that.  The ramp deliberately runs past the predicted critical point: if the tangent is
    genuinely near-singular at 1.36x, solving at 1.5x and 2.0x should get hard or fail, and
    if it sails through then `K_g ~ K_t - K_0` is not capturing a real limit point.
    """
    rows = []
    for f in ramp:
        row = _guarded(lambda: measure(genes, cfg, 0.0, force=f * TOTAL_FORCE_NEWTONS),
                       load_multiple=float(f))
        row["load_multiple"] = float(f)
        if "error" not in row:
            row["implied_critical_total"] = float(f * row["load_factor"])
        rows.append(row)
    ok = [r for r in rows if "error" not in r]
    prod = np.asarray([r["implied_critical_total"] for r in ok], dtype=float)
    lam = np.asarray([r["load_factor"] for r in ok], dtype=float)
    return {"rows": rows,
            "n_solved": len(ok), "n_refused": len(rows) - len(ok),
            "implied_critical_total": prod.tolist(),
            # THE DECIDING FACT.  A real critical factor is a FIXED POINT: at f = lambda_cr
            # the remaining factor is 1.0.  If lambda(f) > 1 at every load level tested,
            # the structure is stable at every one of them and the "critical load" is
            # simply receding ahead of the load -- which is not a limit point, it is a
            # treadmill, and the value read at service load means nothing as a margin.
            "lambda_min_over_ramp": float(lam.min()),
            "crosses_unity": bool((lam <= 1.0).any()),
            "max_load_multiple_solved": float(max(
                r["load_multiple"] for r in ok)) if ok else None,
            # Flat product => the factor is a property of the structure, not of the load
            # level it was measured at.  This is the number to read.
            "product_spread_pct": (float(100.0 * prod.std() / prod.mean())
                                   if prod.size else None),
            # NO PASS/FAIL.  Either outcome is the result, and a gate here would be a
            # threshold invented before the thing it gates is understood.
            "pass": True}


def run_variants(genes, cfg):
    """Open items 3 and 4: the linearisation point, and contact."""
    rows = []
    for kin in ("svk", "linear"):
        for contact in (False, True):
            row = _guarded(lambda: measure(genes, cfg, 0.0, kinematics=kin,
                                           contact=contact),
                           kinematics=kin, contact_in_operator=contact)
            rows.append(row)
    base = next((r for r in rows if r.get("kinematics") == "svk"
                 and r.get("contact_in_operator") is False and "error" not in r), None)
    for r in rows:
        if base and "error" not in r:
            r["vs_baseline_pct"] = float(
                100.0 * (r["load_factor"] / base["load_factor"] - 1.0))
    return {"rows": rows, "baseline": "svk, contact excluded",
            "pass": bool(not any("error" in r for r in rows))}


def main():
    ap = argparse.ArgumentParser(description="M9 Phase 3 buckling load factor")
    ap.add_argument("--genome", default=PP.BEST_SOLUTION)
    ap.add_argument("--out", default="study_m9_buckling.json")
    ap.add_argument("--config", default="coarse")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    designs = _designs(args.genome)
    quick = args.quick
    selected = designs[:1] if quick else [designs[i] for i in (0, 1, 2)]
    configs = ("smoke", "coarse") if quick else ("smoke", "coarse", "medium")
    n_phase = 3 if quick else REFERENCE_PHASES
    cfg = "smoke" if quick else args.config
    ramp = (1.0, 1.5) if quick else LOAD_RAMP
    out_path = os.path.join(HERE, args.out)

    rep = {
        "settings": {"quick": quick, "configs": list(configs), "section_config": cfg,
                     "reference_phases": n_phase,
                     "production_phases": PRODUCTION_PHASES,
                     "genome": os.path.basename(args.genome),
                     "service_force_n": TOTAL_FORCE_NEWTONS,
                     "gate_mesh_rel": GATE_MESH_REL, "elapsed_s": None},
        "sections_complete": [], "complete": False,
    }

    # Checkpointed after every section, for the reason study_m9.py records: a partial file
    # must be readable and must never be mistakable for a verdict, so `pass` is absent
    # until the run is whole.
    def _flush():
        rep["settings"]["elapsed_s"] = round(time.time() - t0, 1)
        with open(out_path, "w") as fh:
            json.dump(rep, fh, indent=1)

    _flush()
    sections = (
        ("mesh", lambda: run_mesh(selected[:1], configs)),
        ("variants", lambda: run_variants(designs[0][1], cfg)),
        ("corroborate", lambda: run_corroborate(designs[0][1], cfg, ramp)),
        ("phase", lambda: run_phase(selected, cfg, n_phase)),
        ("design_space",
         lambda: run_design_space(selected if quick else designs, cfg)),
    )
    for name, fn in sections:
        rep[name] = fn()
        rep["sections_complete"].append(name)
        print(f"  [{name}] {'PASS' if rep[name]['pass'] else 'FAIL'}"
              f"  at {round(time.time() - t0, 1)} s", flush=True)
        _flush()

    rep["complete"] = True
    rep["pass"] = bool(all(rep[n]["pass"] for n, _ in sections))

    mesh_vals = rep["mesh"]["series"].get("best_solution", {}).get("values", [])
    print("\n" + "=" * 70)
    print("M9 PHASE 3 — THE GENERALISED BUCKLING LOAD FACTOR")
    print("=" * 70)
    if mesh_vals:
        print("  mesh ladder :", "  ".join(f"{v:.6f}" for v in mesh_vals),
              f"   last-pair rel {rep['mesh']['series']['best_solution']['last_pair_rel']:.2e}"
              f" against {GATE_MESH_REL}")
    for c in rep.get("phase", {}).get("comparisons", []):
        if "spread_pct" in c:
            print(f"  phase spread: {c['design']:14s} {c['spread_pct']:.2f}% "
                  f"(min {c['reference_min']:.4f}, max {c['reference_max']:.4f})")
    ds = rep.get("design_space", {})
    if ds.get("min") is not None:
        print(f"  design space: {ds['n_converged']}/{ds['n_designs']} converged, "
              f"{ds['min']:.4f}-{ds['max']:.4f}, {ds['n_below_one']} below 1.0")
    co = rep.get("corroborate", {})
    if co.get("product_spread_pct") is not None:
        print(f"  corroborate : f*lambda(f) spread {co['product_spread_pct']:.2f}% over "
              f"{co['n_solved']} solved / {co['n_refused']} refused")
        print(f"                {['%.4f' % v for v in co['implied_critical_total']]}")
        print(f"                lambda reached {co['lambda_min_over_ramp']:.4f} at "
              f"{co['max_load_multiple_solved']:g}x service; crosses 1.0: "
              f"{co['crosses_unity']}")
        if not co["crosses_unity"]:
            print("                -> NO FIXED POINT: lambda(f) > 1 at every load level "
                  "solved, so this\n"
                  "                   is NOT a critical load factor and must not be "
                  "reported as a margin.")
    for r in rep.get("variants", {}).get("rows", []):
        if "error" not in r:
            print(f"  variant     : {r['kinematics']:6s} contact={str(r['contact_in_operator']):5s}"
                  f"  LF {r['load_factor']:.6f}  {r.get('vs_baseline_pct', 0.0):+.2f}%")
    print(f"\n  OVERALL: {'PASS' if rep['pass'] else 'FAIL'}")
    print(f"  elapsed: {round(time.time() - t0, 1)} s")
    _flush()
    print(f"wrote {out_path}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
