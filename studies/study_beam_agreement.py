"""
=============================================================================
  M3 GATE — DOES THE FINITE ELEMENT REPRODUCE BEAM THEORY?
=============================================================================
    .venv-opt/bin/python studies/study_beam_agreement.py

Four checks, in order of how much they prove:

  A1/A2  straight beam vs FL^3/3EI and FL^3/12EI              tolerance 1.0%
  A3     the on-disk genome vs Castigliano at 1% of service   tolerance 5%, FE softer
  A4a    THE ELEMENT GATE — straight-beam slenderness sweep
  A4b    the same sweep on the curved spoke vs Castigliano

WHY A4a IS THE REAL GATE AND A4b IS NOT
---------------------------------------
The plan asked for one slenderness sweep, on the curved spoke against Castigliano,
with the fitted exponent required to be ~2.  Running it exposed a problem with that
formulation: the Castigliano model's error is a SUM of several O(t^2) effects
(transverse shear, the Winkler-Bach neutral-axis shift on a curved member, the
straight-beam EI used for a curved one) which for this particular geometry partially
cancel.  The residual still decays, and faster than t^2 — measured local exponents
2.19, 2.42, 3.57 — but it is not a pure power law and there is no defect that a
STEEPER decay could indicate.  Gating on an upper bound there would fail a correct
element for being too accurate.

So the element gate moved to the straight beam, where the reference is an exact
closed form with no discretization of its own, and where the O(t^2) coefficient is
known analytically:

    delta_FE / (F L^3 / 3EI) - 1  ->  0.81 (t/L)^2      (Timoshenko, k = 5/6, nu = 0.35)

That is a far stronger test.  It does not merely require the discrepancy to shrink;
it requires it to shrink at the right rate TO THE RIGHT NUMBER.  Shear locking would
show up as a discrepancy of the wrong sign, an exponent near 1, or a coefficient near
zero, and all three are checked.

A4b is kept as a reported measurement rather than a pass/fail: its useful output is
the single number at lambda = 1, which is how much the beam model that drove the whole
GA over-stiffens the real spoke.

THE ROOT BOUNDARY CONDITION IS A FIRST-ORDER CONTAMINANT
--------------------------------------------------------
Measured on the straight cantilever, excess over Euler-Bernoulli divided by (t/L)^2:

    L/t        8        16        32        64       128
    plane    0.902     0.882     0.872     0.868     0.913     <- constant, = 0.81 +8%
    clamped  0.597     0.405     0.019    -0.753    -2.248     <- diverges, changes sign

A fully clamped root forbids the lateral Poisson strain eps_yy = -nu*M*y/(EI) that
the bending field genuinely has there.  That is a self-equilibrated end perturbation
decaying over a length ~t whose energy is O(t/L) relative to the beam's — first order,
so at high slenderness it does not merely contaminate the measurement, it DOMINATES
it, and the FE comes out stiffer than Euler-Bernoulli.  Anyone reading that number
without this table would conclude the element shear-locks.

`root_bc="plane"` is therefore mandatory for every beam-theory comparison.  It is not
the right BC for the real wheel, where the root is a meshed junction and collar rather
than an imposed constraint — which is precisely why this only matters here.
=============================================================================
"""

import argparse
import json
import os
import time

import numpy as np

import wheel_fea as wf
import wheel_fem as fem
import wheel_genome as wg
import wheel_mesh as wm

# Reference resolutions.  Both are far finer than production and both are here for a
# measured reason.
#
# BEAM_N_CURVE: `generalized_spoke_mechanics` integrates a polyline, and at its
# production N_CURVE_PTS = 600 that polyline is short by 2.1e-5 relative — which is
# LARGER than the FE-vs-beam discrepancy being measured at lambda = 1/8 (2.8e-5).  At
# 38400 the reference's own error is 7.7e-8, i.e. 0.3% of the smallest number reported.
#
# MESH_H_OVER_T: the span discretization must resolve the root and tip boundary layers,
# which decay over a length ~t.  Measured at lambda = 1/8, the discrepancy moves from
# -0.52% at h = t*2.7 to +0.0027% at h = t/16 — the coarse mesh gets the SIGN wrong.
# `n_thick` is irrelevant by comparison (converged at 4 elements; 24 changes the answer
# by 5e-6 relative).  This is the single most important number in this file for M4:
# element size scales with thickness, not with the part.
BEAM_N_CURVE = 38400
MESH_H_OVER_T = 16
MESH_N_THICK = 4
PROBE_FORCE_FRACTION = 0.01     # keeps geometric nonlinearity below 0.01%

TIMOSHENKO_COEFF = 0.81         # 2(1+nu) / (4k) with k = 5/6, nu = 0.35


def load_genes(path="best_solution.json"):
    with open(path) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


def straight_genes(thickness, span=wf.HUB_RIM_SPAN_MM):
    """A straight uniform beam expressed in the production 14 genes.

    Routed through the real geometry kernel and the real mesh generator on purpose: a
    hand-built rectangular grid would verify a code path that never runs.
    """
    v = np.zeros(14)
    v[0:8:2] = np.array([0.2, 0.4, 0.6, 0.8]) * span
    v[8:12] = thickness
    v[12:14] = 0.5
    return v


def scale_thickness(genes, lam):
    return np.asarray(genes) * np.concatenate([np.ones(8), np.full(4, lam), np.ones(2)])


def arc_length(genes):
    curve, _ = wf.generate_bezier_centerline(*[genes[i] for i in range(8)],
                                             num_points=BEAM_N_CURVE)
    return float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))


def sized_config(genes, k=MESH_H_OVER_T, n_thick=MESH_N_THICK):
    """A mesh whose span element size is t_min/k.  See MESH_H_OVER_T above."""
    t_min = float(np.min(np.asarray(genes)[8:12]))
    n_span = int(np.ceil(k * arc_length(genes) / t_min))
    return wm.MeshConfig("sized", n_span, n_thick, order=2,
                         n_curve=max(BEAM_N_CURVE, 4 * n_span))


def castigliano(genes, bc, force, clip=(0.01, 20.0)):
    curve, _ = wf.generate_bezier_centerline(*[genes[i] for i in range(8)],
                                             num_points=BEAM_N_CURVE)
    d, *_ = wf.generalized_spoke_mechanics(
        curve, *[genes[i] for i in range(8, 12)], wf.SPOKE_WIDTH_MM, force,
        genes[12], genes[13], boundary_condition=bc, thickness_clip=clip)
    return d


def euler_bernoulli(thickness, bc, force, L=wf.HUB_RIM_SPAN_MM):
    I = wf.SPOKE_WIDTH_MM * thickness ** 3 / 12.0
    return force * L ** 3 / ((3.0 if bc == "cantilever" else 12.0)
                             * wf.YOUNGS_MODULUS_PLA_MPA * I)


def local_exponents(x, y):
    """Slope of log|y| against log|x| between each adjacent pair."""
    x, y = np.asarray(x, float), np.asarray(np.abs(y), float)
    return [float(np.log(y[i + 1] / y[i]) / np.log(x[i + 1] / x[i]))
            for i in range(len(x) - 1)]


# ---------------------------------------------------------------------------
# A1 / A2
# ---------------------------------------------------------------------------

def run_a1_a2(slenderness=50.0):
    t = wf.HUB_RIM_SPAN_MM / slenderness
    g = straight_genes(t)
    cfg = sized_config(g)
    F = wf.FORCE_PER_SPOKE_NEWTONS
    rows = []
    for bc in ("cantilever", "fixed_guided"):
        ref = euler_bernoulli(t, bc, F)
        got = fem.spoke_deflection(g, cfg, bc=bc, load_dir=(0.0, -1.0),
                                   root_bc="plane", force=F)
        rows.append({"bc": bc, "closed_form_mm": ref, "fe_mm": got,
                     "rel_error": got / ref - 1.0})
    ratio = rows[0]["fe_mm"] / rows[1]["fe_mm"]
    return {"slenderness_L_over_t": slenderness, "thickness_mm": t,
            "n_span": cfg.n_span, "cases": rows,
            "stiffness_ratio": ratio, "ratio_error": ratio / 4.0 - 1.0,
            "pass": all(abs(r["rel_error"]) < 0.01 for r in rows)
                    and abs(ratio / 4.0 - 1.0) < 0.02}


# ---------------------------------------------------------------------------
# A3
# ---------------------------------------------------------------------------

def run_a3(genes):
    F = wf.FORCE_PER_SPOKE_NEWTONS * PROBE_FORCE_FRACTION
    cfg = sized_config(genes)
    rows = []
    for bc in ("fixed_guided", "cantilever"):
        ref = castigliano(genes, bc, F)
        got = fem.spoke_deflection(genes, cfg, bc=bc, root_bc="plane", force=F)
        rows.append({"bc": bc, "castigliano_mm": ref, "fe_mm": got,
                     "rel_error": got / ref - 1.0})
    return {"force_n": F, "n_span": cfg.n_span, "cases": rows,
            "pass": all(0.0 < r["rel_error"] < 0.05 for r in rows)}


# ---------------------------------------------------------------------------
# A4a — THE ELEMENT GATE
# ---------------------------------------------------------------------------

def run_a4a(slendernesses=(8, 16, 32, 64), root_bcs=("plane", "clamped")):
    """Straight cantilever against the exact closed form, at four slendernesses.

    The reported quantity is the excess divided by (t/L)^2, which Timoshenko says is
    0.81 and which must be CONSTANT if the element converges at second order.
    """
    L = wf.HUB_RIM_SPAN_MM
    F = wf.FORCE_PER_SPOKE_NEWTONS
    out = {}
    for root_bc in root_bcs:
        rows = []
        for r in slendernesses:
            t = L / r
            g = straight_genes(t)
            cfg = sized_config(g)
            ref = euler_bernoulli(t, "cantilever", F)
            got = fem.spoke_deflection(g, cfg, bc="cantilever",
                                       load_dir=(0.0, -1.0), root_bc=root_bc,
                                       force=F)
            excess = got / ref - 1.0
            rows.append({"L_over_t": float(r), "thickness_mm": t,
                         "n_span": cfg.n_span, "closed_form_mm": ref,
                         "fe_mm": got, "excess": excess,
                         "excess_over_t_L_sq": excess / (t / L) ** 2})
        thick = [row["thickness_mm"] for row in rows]
        exps = local_exponents(thick, [row["excess"] for row in rows])
        coeffs = [row["excess_over_t_L_sq"] for row in rows]
        out[root_bc] = {
            "rows": rows,
            "local_exponents": exps,
            "fitted_exponent": float(np.polyfit(np.log(thick),
                                                np.log(np.abs([r["excess"]
                                                               for r in rows])), 1)[0]),
            "mean_coefficient": float(np.mean(coeffs)),
            "coefficient_error_vs_timoshenko": float(np.mean(coeffs)
                                                     / TIMOSHENKO_COEFF - 1.0),
            "all_positive": bool(all(r["excess"] > 0 for r in rows)),
        }
    # The verdict is read off the beam-consistent root only; `clamped` is reported as
    # a diagnostic and is EXPECTED to fail, so it must not vote.
    if "plane" in out:
        p = out["plane"]
        out["pass"] = bool(p["all_positive"]
                           and 1.7 <= p["fitted_exponent"] <= 2.3
                           and abs(p["coefficient_error_vs_timoshenko"]) < 0.20)
    return out


# ---------------------------------------------------------------------------
# A4b — the curved spoke against Castigliano
# ---------------------------------------------------------------------------

def run_a4b(genes, lambdas=(1.0, 0.5, 0.25, 0.125), bc="fixed_guided",
            root_bc="plane"):
    rows = []
    for lam in lambdas:
        g = scale_thickness(genes, lam)
        F = wf.FORCE_PER_SPOKE_NEWTONS * PROBE_FORCE_FRACTION * lam ** 3
        cfg = sized_config(g)
        ref = castigliano(g, bc, F)
        got = fem.spoke_deflection(g, cfg, bc=bc, root_bc=root_bc, force=F)
        rows.append({"lambda": lam, "thickness_mm": float(np.min(g[8:12])),
                     "n_span": cfg.n_span, "force_n": F,
                     "castigliano_mm": ref, "fe_mm": got,
                     "rel_error": got / ref - 1.0})
    exps = local_exponents([r["lambda"] for r in rows],
                           [r["rel_error"] for r in rows])
    return {
        "bc": bc, "root_bc": root_bc, "rows": rows, "local_exponents": exps,
        "fitted_exponent": float(np.polyfit(
            np.log([r["lambda"] for r in rows]),
            np.log(np.abs([r["rel_error"] for r in rows])), 1)[0]),
        "all_positive": bool(all(r["rel_error"] > 0 for r in rows)),
        "monotone": bool(all(abs(rows[i + 1]["rel_error"])
                             < abs(rows[i]["rel_error"])
                             for i in range(len(rows) - 1))),
        # No upper bound: see the module docstring.  A steeper-than-quadratic decay is
        # a better result, not a worse one, and nothing broken produces it.
        "pass": bool(all(r["rel_error"] > 0 for r in rows)
                     and all(e >= 1.7 for e in exps)),
    }


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def _print(report):
    def head(s):
        print(f"\n{s}\n" + "-" * len(s))

    def verdict(ok):
        return "PASS" if ok else "FAIL"

    a = report["A1_A2"]
    head(f"A1 / A2  straight beam vs closed form, L/t = {a['slenderness_L_over_t']:.0f}"
         f" (t = {a['thickness_mm']:.4f} mm, n_span = {a['n_span']})")
    for r in a["cases"]:
        print(f"  {r['bc']:13s} closed form {r['closed_form_mm']:9.5f}   "
              f"FE {r['fe_mm']:9.5f}   {r['rel_error']:+.4%}   [< 1.0%]")
    print(f"  cantilever / fixed-guided stiffness ratio "
          f"{a['stiffness_ratio']:.5f}  ({a['ratio_error']:+.3%} from 4)")
    print(f"  -> {verdict(a['pass'])}")

    a = report["A3"]
    head(f"A3  genome df26647 vs Castigliano at {a['force_n']:.4f} N "
         f"(n_span = {a['n_span']})")
    for r in a["cases"]:
        print(f"  {r['bc']:13s} Castigliano {r['castigliano_mm']:9.5f}   "
              f"FE {r['fe_mm']:9.5f}   {r['rel_error']:+.4%}   "
              f"[< 5%, FE softer]")
    print(f"  -> {verdict(a['pass'])}")

    a = report["A4a"]
    head("A4a  THE ELEMENT GATE — straight cantilever slenderness sweep")
    print(f"  excess over FL^3/3EI, divided by (t/L)^2.  "
          f"Timoshenko says {TIMOSHENKO_COEFF:.2f}, constant.")
    print(f"  {'L/t':>8s} " + " ".join(f"{r['L_over_t']:>10.0f}"
                                       for r in a["plane"]["rows"]))
    for root_bc in ("plane", "clamped"):
        if root_bc not in a:
            continue
        print(f"  {root_bc:>8s} " + " ".join(f"{r['excess_over_t_L_sq']:>10.4f}"
                                             for r in a[root_bc]["rows"]))
    p = a["plane"]
    print(f"\n  root_bc=plane   fitted exponent {p['fitted_exponent']:.4f}   "
          f"[1.7, 2.3]")
    print(f"                  local exponents "
          + ", ".join(f"{e:.3f}" for e in p["local_exponents"]))
    print(f"                  mean coefficient {p['mean_coefficient']:.4f}  "
          f"({p['coefficient_error_vs_timoshenko']:+.2%} vs Timoshenko)  [< 20%]")
    print(f"                  all discrepancies positive (FE softer): "
          f"{p['all_positive']}")
    if "clamped" in a:
        c = a["clamped"]
        print(f"  root_bc=clamped fitted exponent {c['fitted_exponent']:.4f}   "
              f"sign changes: {not c['all_positive']}  <- the O(t) contaminant")
    print(f"  -> {verdict(a['pass'])}")

    a = report["A4b"]
    head("A4b  curved spoke vs Castigliano (reported, not gated on an upper bound)")
    for r, e in zip(a["rows"], list(a["local_exponents"]) + [None]):
        tail = "" if e is None else f"   local exponent {e:.3f}"
        print(f"  lambda {r['lambda']:6.3f}  t {r['thickness_mm']:6.4f} mm  "
              f"n_span {r['n_span']:5d}   {r['rel_error']:+.5%}{tail}")
    print(f"  fitted exponent {a['fitted_exponent']:.4f}   "
          f"all positive {a['all_positive']}   monotone {a['monotone']}")
    print(f"  -> {verdict(a['pass'])}")

    head("M3 GATE")
    for key in ("A1_A2", "A3", "A4a", "A4b"):
        print(f"  {key:6s} {verdict(report[key]['pass'])}")
    print(f"\n  OVERALL: {verdict(report['pass'])}")
    if report["pass"]:
        design = report["A4b"]["rows"][0]["rel_error"]
        print(f"\n  The headline number: at the design point the Castigliano model "
              f"over-stiffens\n  the spoke by {design:.2%}.  That is the entire "
              f"single-spoke error of the model\n  that drove the GA — small, "
              f"one-directional, and now quantified.")


def _plot(report, path):
    """Log-log convergence, with the t^2 reference slope drawn through the data.

    The whole gate is "does this decay at slope 2", so the figure draws the slope-2
    guide rather than leaving the reader to fit a line by eye.  Both root treatments
    are on the left panel because the divergence of the clamped one is the single most
    important thing to be able to see.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    a = report["A4a"]
    for root_bc, style in (("plane", "o-"), ("clamped", "s--")):
        if root_bc not in a:
            continue
        rows = a[root_bc]["rows"]
        t = np.array([r["thickness_mm"] for r in rows])
        e = np.array([r["excess"] for r in rows])
        pos = e > 0
        ax1.loglog(t[pos], e[pos], style, label=f'root_bc="{root_bc}"')
        if (~pos).any():
            ax1.loglog(t[~pos], -e[~pos], style, mfc="none",
                       color=ax1.lines[-1].get_color(),
                       label=f'root_bc="{root_bc}" (NEGATIVE — FE stiffer)')
    rows = a["plane"]["rows"]
    t = np.array([r["thickness_mm"] for r in rows])
    L = wf.HUB_RIM_SPAN_MM
    ax1.loglog(t, TIMOSHENKO_COEFF * (t / L) ** 2, "k:",
               label=f"Timoshenko  {TIMOSHENKO_COEFF} (t/L)$^2$")
    ax1.set_xlabel("thickness t (mm)")
    ax1.set_ylabel("|excess over $FL^3/3EI$|")
    ax1.set_title("A4a — straight cantilever vs exact closed form\n"
                  f"fitted exponent {a['plane']['fitted_exponent']:.3f}, "
                  f"coefficient {a['plane']['mean_coefficient']:.3f}")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    b = report["A4b"]
    rows = b["rows"]
    lam = np.array([r["lambda"] for r in rows])
    e = np.abs([r["rel_error"] for r in rows])
    ax2.loglog(lam, e, "o-", label="FE vs Castigliano")
    ax2.loglog(lam, e[0] * lam ** 2, "k:", label="slope 2 through $\\lambda=1$")
    ax2.set_xlabel("thickness scale $\\lambda$")
    ax2.set_ylabel("|relative discrepancy|")
    ax2.set_title("A4b — curved spoke vs Castigliano\n"
                  f"fitted exponent {b['fitted_exponent']:.3f} "
                  "(faster than $t^2$: the model's\n"
                  "several $O(t^2)$ errors partially cancel)")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.annotate(f"design point: beam model\nover-stiffens by "
                 f"{rows[0]['rel_error']:.2%}",
                 xy=(lam[0], e[0]), xytext=(0.35, 0.85),
                 textcoords="axes fraction", fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=0.8))

    fig.suptitle(f"M3 beam-agreement gate — "
                 f"{'PASS' if report['pass'] else 'FAIL'}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def main():
    ap = argparse.ArgumentParser(description="M3 beam-agreement gate")
    ap.add_argument("--genome", default="best_solution.json")
    ap.add_argument("--out", default="study_beam_agreement.json")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="coarser meshes and 3 slenderness levels; for CI, not for "
                         "the recorded result")
    args = ap.parse_args()

    global MESH_H_OVER_T
    if args.quick:
        MESH_H_OVER_T = 8

    genes = load_genes(args.genome)
    t0 = time.time()
    report = {
        "A1_A2": run_a1_a2(),
        "A3": run_a3(genes),
        "A4a": run_a4a(slendernesses=(8, 16, 32) if args.quick else (8, 16, 32, 64)),
        "A4b": run_a4b(genes, lambdas=(1.0, 0.5, 0.25) if args.quick
                       else (1.0, 0.5, 0.25, 0.125)),
    }
    report["pass"] = all(report[k]["pass"] for k in ("A1_A2", "A3", "A4a", "A4b"))
    report["settings"] = {
        "beam_n_curve": BEAM_N_CURVE, "mesh_h_over_t": MESH_H_OVER_T,
        "mesh_n_thick": MESH_N_THICK, "probe_force_fraction": PROBE_FORCE_FRACTION,
        "poisson": fem.POISSON_RATIO_PLA, "plane": "stress",
        "elapsed_s": round(time.time() - t0, 1),
    }
    _print(report)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {os.path.abspath(args.out)}  "
          f"({report['settings']['elapsed_s']} s)")
    if not args.no_plot:
        try:
            print(f"wrote {_plot(report, os.path.splitext(args.out)[0] + '.jpg')}")
        except Exception as exc:                            # pragma: no cover
            print(f"(plot skipped: {exc})")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
