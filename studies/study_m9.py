"""M9 Phase 2 — measure the lowest reduced tangent eigenvalue.

This study is deliberately measurement-only.  It uses the service-force contact path,
records LOBPCG diagnostics, and does not add the eigenvalue to the Stage 3 objective.
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

import wheel_adjoint as WA  # noqa: E402
import wheel_fem as fem  # noqa: E402
import wheel_genome as wg  # noqa: E402
import wheel_wheel as WW  # noqa: E402
from wheel_fea import TOTAL_FORCE_NEWTONS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PERIOD_DEG = 30.0
REFERENCE_PHASES = 13
PRODUCTION_PHASES = 4
GATE_MESH_REL = 0.05
GATE_EIG_REL = 1.0e-6


def _genes(path):
    with open(path) as fh:
        data = json.load(fh)
    return wg.genes_to_vector(data["genes"])


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


def _independent_eigenvalue(meta):
    prob = meta["prob"]
    res = meta["res"]
    K = fem.assemble_stiffness(prob.coords, prob.conn, res["u"], order=prob.order,
                               lam=prob.lam, mu=prob.mu, width=prob.width,
                               nonlinear=prob.nonlinear)
    K = K + prob.contact.stiffness(prob.coords, res["u"])
    Kr = (prob.T.T @ K @ prob.T).tocsc()
    return float(spla.eigsh(Kr, k=1, sigma=0.0, which="LM",
                            return_eigenvectors=False)[0])


def measure(genes, cfg, phase_deg, *, force=TOTAL_FORCE_NEWTONS,
            independent=False, gradient=True):
    diagnostics = {}
    t0 = time.time()
    mesh = WW.build_wheel(genes, cfg, phase_deg=float(phase_deg))
    sec = fem.solve_wheel_contact(mesh, force=force)
    res = fem.solve_wheel_contact_at(mesh, sec["axle_drop_mm"],
                                     u_reduced0=sec["u_reduced"])
    prob = fem.wheel_contact_problem(mesh, indentation_mm=sec["axle_drop_mm"])
    K = fem.assemble_stiffness(prob.coords, prob.conn, res["u"], order=prob.order,
                                lam=prob.lam, mu=prob.mu, width=prob.width,
                                nonlinear=prob.nonlinear)
    K = K + prob.contact.stiffness(prob.coords, res["u"])
    Kr = (prob.T.T @ K @ prob.T).tocsc()
    back_solve = spla.factorized(Kr)
    value, _, eig_diag = WA.lowest_tangent_eigenpair(Kr, back_solve)
    diagnostics[WA.BUCKLING_EIG_NAME] = {**eig_diag, "value": value,
                                         "reduced_dof": int(Kr.shape[0])}
    gradient_norm = None
    gradient_max = None
    if gradient:
        out = WA.service_qoi_value_and_grad(
            genes, cfg, qois=(WA.BUCKLING_EIG_NAME,), force=force,
            phase_deg=float(phase_deg), diagnostics=diagnostics)
        eig = out[WA.BUCKLING_EIG_NAME]
        value = float(eig["value"])
        gradient_norm = float(np.linalg.norm(eig["grad"]))
        gradient_max = float(np.max(np.abs(eig["grad"])))
    row = {
        "config": cfg,
        "phase_deg": float(phase_deg),
        "force_n": float(force),
        "lambda_min": float(value),
        "gradient_norm": gradient_norm,
        "gradient_max": gradient_max,
        "axle_drop_mm": float(sec["axle_drop_mm"]),
        "iterations": int(diagnostics[WA.BUCKLING_EIG_NAME]["iterations"]),
        "residual_rel": float(diagnostics[WA.BUCKLING_EIG_NAME]["residual_rel"]),
        "lobpcg_converged": bool(diagnostics[WA.BUCKLING_EIG_NAME]["converged"]),
        "reduced_dof": int(diagnostics[WA.BUCKLING_EIG_NAME]["reduced_dof"]),
        "elapsed_s": float(time.time() - t0),
    }
    if independent:
        reference = float(spla.eigsh(Kr, k=1, sigma=0.0, which="LM",
                                     return_eigenvectors=False)[0])
        row["independent_lambda_min"] = reference
        row["independent_rel_error"] = abs(row["lambda_min"] / reference - 1.0)
    return row


def _series(rows):
    values = np.asarray([r["lambda_min"] for r in rows if "error" not in r], dtype=float)
    if len(values) < 2:
        return {"values": values.tolist(), "last_pair_rel": None, "pass": True}
    rel = abs(values[-1] / values[-2] - 1.0) if values[-2] else float("inf")
    return {"values": values.tolist(), "last_pair_rel": float(rel),
            "pass": bool(np.isfinite(rel) and rel < GATE_MESH_REL)}


def run_mesh(designs, configs, phases):
    rows = []
    series = {}
    for name, genes in designs:
        design_rows = []
        for cfg in configs:
            for phase in phases:
                try:
                    row = measure(genes, cfg, phase, gradient=(cfg == configs[0]
                                  and phase == phases[0]),
                                  independent=(name == "best_solution"
                                               and phase == phases[0]
                                               and cfg in (configs[0], configs[-1])))
                except Exception as exc:
                    # The `fine` rung is 261k dof and had never been run when this was
                    # written; `run_load` and `run_design_space` already guard this way,
                    # and study_stage3's ladder wraps each row for the same reason.  A
                    # refused row is recorded and FAILS the section (below) rather than
                    # discarding the rungs that did converge.  This cannot catch a cgroup
                    # OOM kill -- that is what the per-section checkpoint in main() is for.
                    row = {"config": cfg, "phase_deg": float(phase),
                           "error": f"{type(exc).__name__}: {exc}"}
                row["design"] = name
                rows.append(row)
                design_rows.append(row)
        for phase in phases:
            phase_rows = [r for r in design_rows if r.get("phase_deg") == float(phase)]
            series[f"{name}:phase={phase:g}"] = _series(phase_rows)
    return {"rows": rows, "series": series,
            "n_error_rows": sum(1 for r in rows if "error" in r),
            "pass": bool(all(v["pass"] for v in series.values())
                         and not any("error" in r for r in rows))}


def run_phase(designs, cfg, n_phase):
    phases = np.linspace(0.0, PERIOD_DEG, n_phase, endpoint=False)
    rows = []
    comparisons = []
    for name, genes in designs:
        ref = [measure(genes, cfg, p, gradient=False) for p in phases]
        production = [r for r in ref if any(abs(r["phase_deg"] - p) < 1e-12
                                             for p in np.linspace(0, PERIOD_DEG,
                                                                  PRODUCTION_PHASES,
                                                                  endpoint=False))]
        vals = np.asarray([r["lambda_min"] for r in ref])
        pvals = np.asarray([r["lambda_min"] for r in production])
        scale = max(float(np.max(np.abs(vals))), 1.0e-30)
        comparisons.append({
            "design": name,
            "reference_min": float(np.min(vals)),
            "reference_max": float(np.max(vals)),
            "reference_mean": float(np.mean(vals)),
            "reference_std": float(np.std(vals)),
            "production_min": float(np.min(pvals)),
            "production_max": float(np.max(pvals)),
            "min_rel_error": float(abs(np.min(pvals) - np.min(vals)) / scale),
            "max_rel_error": float(abs(np.max(pvals) - np.max(vals)) / scale),
            "pass": bool(np.all(np.isfinite(vals)) and np.all(
                [r["lobpcg_converged"] for r in ref])),
        })
        rows.extend([{**r, "design": name} for r in ref])
    return {"n_phase": int(n_phase), "period_deg": PERIOD_DEG,
            "rows": rows, "comparisons": comparisons,
            "pass": bool(all(r["pass"] for r in comparisons))}


def run_load(designs, cfg, forces):
    rows = []
    for name, genes in designs:
        for force in forces:
            try:
                row = measure(genes, cfg, 0.0, force=float(force), gradient=False)
                row["design"] = name
            except Exception as exc:  # record solver health; do not hide it
                row = {"design": name, "force_n": float(force),
                       "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
    return {"rows": rows,
            "pass": bool(all("error" not in r and r["lobpcg_converged"] for r in rows))}


def run_design_space(designs, cfg):
    rows = []
    for name, genes in designs:
        try:
            row = measure(genes, cfg, 0.0, gradient=False)
            row["design"] = name
        except Exception as exc:
            row = {"design": name, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
    return {"rows": rows,
            "pass": bool(all("error" not in r and r["lobpcg_converged"] for r in rows))}


def main():
    ap = argparse.ArgumentParser(description="M9 Phase 2 tangent-eigenvalue study")
    ap.add_argument("--genome", default=PP.BEST_SOLUTION)
    ap.add_argument("--out", default="study_m9.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    designs = _designs(args.genome)
    selected = designs[:1] if args.quick else [designs[i] for i in (0, 1, 2)]
    configs = ("smoke",) if args.quick else ("smoke", "coarse", "medium", "fine")
    n_phase = 2 if args.quick else REFERENCE_PHASES
    mesh_phases = (0.0,) if args.quick else (0.0, 15.0)
    phase_cfg = "smoke" if args.quick else "coarse"
    force_ladder = (TOTAL_FORCE_NEWTONS,) if args.quick else (
        0.25 * TOTAL_FORCE_NEWTONS, 0.5 * TOTAL_FORCE_NEWTONS,
        0.75 * TOTAL_FORCE_NEWTONS, TOTAL_FORCE_NEWTONS)
    out_path = os.path.join(HERE, args.out)
    rep = {
        "settings": {"quick": args.quick, "configs": list(configs),
                     "reference_phases": n_phase, "production_phases": PRODUCTION_PHASES,
                     "genome": os.path.basename(args.genome),
                     "elapsed_s": None},
        "sections_complete": [],
        "complete": False,
    }

    # CHECKPOINT AFTER EVERY SECTION.  The full run is hours and its first section drives
    # the 261k-dof `fine` rung, so it is the most likely thing in this repo to be OOM-killed
    # by its own cgroup cap -- and a SIGKILL is not catchable, so a `try` cannot preserve
    # anything a later section would have kept.  Writing as we go means a kill costs the
    # section it happened in, not the ones already paid for.  `complete` says whether the
    # file is a whole run; `pass` is absent until it is, so a partial file can never be
    # misread as a verdict.
    def _flush():
        rep["settings"]["elapsed_s"] = round(time.time() - t0, 1)
        with open(out_path, "w") as fh:
            json.dump(rep, fh, indent=1)

    _flush()
    sections = (
        ("mesh", lambda: run_mesh(selected, configs, mesh_phases)),
        ("phase", lambda: run_phase(selected, phase_cfg, n_phase)),
        ("load", lambda: run_load(selected, phase_cfg, force_ladder)),
        ("design_space",
         lambda: run_design_space(designs if not args.quick else selected, phase_cfg)),
    )
    for name, fn in sections:
        rep[name] = fn()
        rep["sections_complete"].append(name)
        print(f"  [{name}] {'PASS' if rep[name]['pass'] else 'FAIL'}"
              f"  at {round(time.time() - t0, 1)} s", flush=True)
        _flush()

    rep["complete"] = True
    rep["settings"]["elapsed_s"] = round(time.time() - t0, 1)
    rep["pass"] = bool(rep["mesh"]["pass"] and rep["phase"]["pass"]
                       and rep["load"]["pass"] and rep["design_space"]["pass"])
    print(f"M9 Phase 2: {'PASS' if rep['pass'] else 'FAIL'}")
    print(f"  elapsed: {rep['settings']['elapsed_s']} s")
    print(f"  mesh:   {'PASS' if rep['mesh']['pass'] else 'FAIL'}")
    print(f"  phase:  {'PASS' if rep['phase']['pass'] else 'FAIL'}")
    print(f"  load:   {'PASS' if rep['load']['pass'] else 'FAIL'}")
    print(f"  designs:{'PASS' if rep['design_space']['pass'] else 'FAIL'}")
    _flush()
    print(f"wrote {out_path}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
