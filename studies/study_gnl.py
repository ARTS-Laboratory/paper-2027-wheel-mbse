"""
=============================================================================
  M5 GATE — GEOMETRIC NONLINEARITY, AND WHETHER STAGE 3 CAN IGNORE IT
=============================================================================
    .venv-opt/bin/python studies/study_gnl.py

The plan's M5 asks four things: build the Newton loop, verify frame indifference at
finite rotation, verify that linear and SVK agree at 1% of service load, and then
MEASURE how much geometric nonlinearity actually changes the answer — with an explicit
off-ramp, "if < 2%, run the Stage-3 trajectory on the linear model and check GNL only
at checkpoints".

THE ANSWER: 2% is exceeded, and the off-ramp is closed for the same structural reason
M4's was.  At service load the wheel is 3.95% SOFTER under SVK than under linear
kinematics — already twice the threshold — and, more decisively, that correction is not
a property of the wheel that can be applied once.  Held at a FIXED axle drop of 1.667 mm
by scaling the load per genome, the correction still ranges over roughly an order of
magnitude across feasible designs, so it is not recoverable from the linear answer.
Stage 3 needs the nonlinear solve in the loop.

WHERE THE NONLINEARITY LIVES, AND IT IS NOT WHERE THE COMPLIANCE IS
--------------------------------------------------------------------
The residual at the linear solution is concentrated in the rim band within a few degrees
of the contact patch — the thin band locally flattens against the ground, and local
rotation there is what the linear kernel gets wrong.  That is why the correction does not
follow the axle drop: the axle drop is dominated by the spokes (66% of the strain
energy), while the geometric correction is dominated by a local rim effect.  Two
different quantities, and one does not predict the other.

Everything softens rather than stiffens.  That is the physically expected direction here:
a curved flexure loaded to unwind gets a longer moment arm as it deflects, so the
tangent stiffness falls.  A geometric term that came out STIFFENING would be the
signature of a sign error in the Green-Lagrange strain, which is why the direction is
asserted rather than merely reported.

WHAT THE PLAN ASKED FOR THAT HAD TO CHANGE
------------------------------------------
"Residual < 1e-10" is not attainable at low load and this is not a solver defect.  The
reduced tangent of a thin flexure is conditioned around 1e8-1e9, so the sparse
factorization's own backward error puts a floor under the residual that scales with the
load exactly as the load does — normalising cannot remove it.  At 1% of service load the
loop stalls at 1.2e-9.  `wheel_fem.solve_nonlinear` therefore also accepts an
ENERGY-INCREMENT criterion (|du . r| below 1e-14 of the first step's), which is
conditioning-insensitive and says the thing that actually matters for a four-figure
answer.  Both are reported for every solve; see the block comment in `wheel_fem.py`.
=============================================================================
"""

import argparse
import json
import os

import project_paths as PP
import time

import numpy as np

import wheel_fea as W
import wheel_fem as fem
import wheel_genome as wg
import wheel_wheel as WW

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = "coarse"
SERVICE_FORCE_N = W.TOTAL_FORCE_NEWTONS

# Written down BEFORE the study was run, per the plan's rule.
GATE_ROTATION_ENERGY_RATIO = 1e-10   # SVK energy under a 30 deg rigid rotation
GATE_SMALL_LOAD_REL = 1e-3           # linear vs SVK at 1% of service load
GATE_CONTINUATION_REL = 1e-9         # 1 load step vs 8 must give the same equilibrium
GATE_NEWTON_ORDER = 1.5              # observed convergence order, quadratic is 2
PLAN_GNL_THRESHOLD = 0.02            # "if < 2%, keep the trajectory linear"


def load_genes(path="best_solution.json"):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


def _drop(mesh, kinematics, force=SERVICE_FORCE_N, **kw):
    return fem.solve_wheel(mesh, kinematics=kinematics, force=force, **kw)


# ---------------------------------------------------------------------------
# G1 — FRAME INDIFFERENCE ON THE ASSEMBLED WHEEL
# ---------------------------------------------------------------------------

def run_frame_indifference(genes, cfg=DEFAULT_CONFIG, angles=(1.0, 30.0)):
    """A rigid rotation of the WHOLE WHEEL must store exactly zero energy under SVK.

    `tests/test_fem.py` already does this on a single element, and this is the same
    identity — but it is not the same test.  Here the rotation passes through the mesh
    assembly, the twelve rotated sectors, and every seam, so a sector wired to the wrong
    neighbour or a seam declared but not merged shows up as stored energy in a body that
    has not deformed.  It is the cheapest end-to-end check in the file.

    The linear kernel's energy is reported as the scale.  Asserting a bare "SVK energy
    below 1e-9 mJ" would be meaningless without it: at 1 degree the linear kernel stores
    little too, and the ratio is the only thing that says the test had a chance to fail.
    """
    mesh = WW.build_wheel(genes, cfg)
    coords = np.asarray(mesh.coords)
    lam, mu = fem.lame(W.YOUNGS_MODULUS_PLA_MPA, fem.POISSON_RATIO_PLA)
    kw = dict(order=mesh.cfg.order, lam=lam, mu=mu, width=W.SPOKE_WIDTH_MM)

    rows = []
    for a in angles:
        th = np.radians(a)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        u = (coords @ R.T - coords).ravel()
        e_svk = fem.total_energy(coords, mesh.conn, u, nonlinear=True, **kw)
        e_lin = fem.total_energy(coords, mesh.conn, u, nonlinear=False, **kw)
        rows.append({"angle_deg": float(a), "svk_energy_mJ": float(e_svk),
                     "linear_energy_mJ": float(e_lin),
                     "ratio": float(abs(e_svk) / e_lin)})
    worst = max(r["ratio"] for r in rows)
    return {"rows": rows, "worst_ratio": worst, "config": cfg,
            "pass": bool(worst < GATE_ROTATION_ENERGY_RATIO)}


# ---------------------------------------------------------------------------
# G2 — THE LOAD LADDER
# ---------------------------------------------------------------------------

def run_load_ladder(genes, cfg=DEFAULT_CONFIG,
                    fractions=(0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)):
    """Linear vs SVK axle drop over a load ladder, and the RATE at which they diverge.

    The plan's criterion is the single point at 1% of service load (< 0.1%).  A single
    point is weak evidence on its own — an implementation that simply ran the linear
    kernel twice would pass it perfectly — so the ladder is fitted as well.  A geometric
    correction enters at first order in the deflection, hence at first order in the load,
    so the fitted exponent of (svk/linear - 1) against load must be about 1.

    Measured: 0.038% at 1%, 3.95% at service load, 12.8% at 3x, fitted exponent 1.03.
    Exponent 0 would mean a constant offset (a modelling difference, not a geometric
    effect); exponent 2 would mean the linear term had cancelled, which for an
    asymmetric structure like this would need explaining.
    """
    mesh = WW.build_wheel(genes, cfg)
    rows = []
    for fr in fractions:
        lin = _drop(mesh, "linear", SERVICE_FORCE_N * fr)
        nl = _drop(mesh, "svk", SERVICE_FORCE_N * fr)
        rows.append({
            "load_fraction": float(fr),
            "linear_mm": lin["axle_drop_mm"],
            "svk_mm": nl["axle_drop_mm"],
            "rel_diff": float(nl["axle_drop_mm"] / lin["axle_drop_mm"] - 1.0),
            "iterations": int(nl["newton"]["total_iterations"]),
            "backtracks": int(nl["newton"]["total_backtracks"]),
            "criterion": nl["newton"]["criterion"],
            "residual_rel": nl["residual_rel"],
        })
    x = np.log(np.array([r["load_fraction"] for r in rows]))
    y = np.log(np.abs(np.array([r["rel_diff"] for r in rows])))
    exponent = float(np.polyfit(x, y, 1)[0])
    small = min(rows, key=lambda r: r["load_fraction"])
    service = min(rows, key=lambda r: abs(r["load_fraction"] - 1.0))
    return {
        "rows": rows,
        "fitted_exponent": exponent,
        "small_load_fraction": small["load_fraction"],
        "small_load_rel_diff": small["rel_diff"],
        "service_rel_diff": service["rel_diff"],
        # Every entry must soften.  A stiffening geometric term at this scale would be a
        # sign error in the Green-Lagrange strain, not a physical finding.
        "all_softer": bool(all(r["rel_diff"] > 0.0 for r in rows)),
        "pass": bool(abs(small["rel_diff"]) < GATE_SMALL_LOAD_REL
                     and all(r["rel_diff"] > 0.0 for r in rows)
                     and 0.7 < exponent < 1.4),
    }


# ---------------------------------------------------------------------------
# G3 — NEWTON HEALTH
# ---------------------------------------------------------------------------

def run_newton_health(genes, cfg=DEFAULT_CONFIG, step_counts=(1, 2, 4, 8)):
    """Continuation independence, warm-start payoff, and the observed order.

    THE CONTINUATION CHECK IS THE ONE THAT MATTERS.  A load path is a numerical device,
    not physics: for a conservative problem the converged equilibrium cannot depend on
    how many increments were used to reach it.  If it does, the solve is not converged —
    and the failure is invisible to a residual norm, because each step converges happily
    to its own slightly wrong state.

    Warm starting is measured because it is the single largest performance lever once
    this sits inside the Stage-3 loop, where consecutive designs differ by ~1%.  Here the
    warm start is the LINEAR solution, which is the closest thing available on a cold
    call and is free (the linear solve is wanted anyway).

    The observed order is computed from the residual triple around the sharpest drop,
    since the first iteration is a transient — the SVK internal force at the linear
    solution is far from equilibrium, so the residual RISES on step one before Newton
    takes hold.  That rise is expected and is not a defect.
    """
    mesh = WW.build_wheel(genes, cfg)
    rows = []
    for s in step_counts:
        t0 = time.time()
        res = _drop(mesh, "svk", newton=dict(steps=s))
        rows.append({"steps": int(s), "axle_drop_mm": res["axle_drop_mm"],
                     "iterations": int(res["newton"]["total_iterations"]),
                     "backtracks": int(res["newton"]["total_backtracks"]),
                     "residual_rel": res["residual_rel"],
                     "seconds": round(time.time() - t0, 2)})
    d = np.array([r["axle_drop_mm"] for r in rows])
    spread = float(d.max() / d.min() - 1.0)

    # Warm start from the linear solution — same DofMap, so the reduced vectors line up.
    lin = fem.wheel_problem(mesh, kinematics="linear")[0]
    u0 = fem.solve_linear(lin)["u_reduced"]
    prob = fem.wheel_problem(mesh, kinematics="svk")[0]
    cold = fem.solve_nonlinear(prob)
    warm = fem.solve_nonlinear(prob, u_reduced0=u0)

    # Compared on the whole displacement FIELD rather than on the axle drop: a warm start
    # that landed on a different equilibrium could easily agree at one node.
    warm_gap = float(np.abs(warm["u"] - cold["u"]).max()
                     / max(np.abs(cold["u"]).max(), 1e-300))

    hist = cold["newton"]["history"][0]["residuals"]
    order = _observed_order(hist)
    return {
        "continuation": rows,
        "continuation_spread": spread,
        "cold_iterations": int(cold["newton"]["total_iterations"]),
        "warm_iterations": int(warm["newton"]["total_iterations"]),
        "warm_start_agrees_rel": warm_gap,
        "residual_history": [float(x) for x in hist],
        "observed_order": order,
        "worst_residual_rel": float(max(r["residual_rel"] for r in rows)),
        "pass": bool(spread < GATE_CONTINUATION_REL
                     and order > GATE_NEWTON_ORDER
                     and warm_gap < 1e-8),
    }


def _observed_order(residuals):
    """Best observed convergence order p from log r_{k+1} = p log r_k over the history.

    Taken as the maximum over consecutive triples rather than the last one, because the
    final iteration sits on the roundoff floor where the ratio is meaningless.
    """
    r = np.array([x for x in residuals if x > 0.0])
    best = 0.0
    for i in range(len(r) - 2):
        a, b, c = r[i], r[i + 1], r[i + 2]
        if not (a > b > c) or b <= 0 or a <= 0:
            continue
        denom = np.log(b / a)
        if abs(denom) < 1e-12:
            continue
        best = max(best, float(np.log(c / b) / denom))
    return best


# ---------------------------------------------------------------------------
# G4 — THE MEASUREMENT, AND WHETHER IT IS A CONSTANT
# ---------------------------------------------------------------------------

def run_mesh_ladder(genes, configs=("smoke", "coarse", "medium")):
    """Is the correction itself converged?  It has to be, to be worth 4%."""
    rows = []
    for c in configs:
        mesh = WW.build_wheel(genes, c)
        lin, nl = _drop(mesh, "linear"), _drop(mesh, "svk")
        rows.append({"config": c, "n_elements": int(mesh.n_elements),
                     "linear_mm": lin["axle_drop_mm"], "svk_mm": nl["axle_drop_mm"],
                     "rel_diff": float(nl["axle_drop_mm"] / lin["axle_drop_mm"] - 1.0)})
    v = np.array([r["rel_diff"] for r in rows])
    return {"rows": rows, "spread_points": float((v.max() - v.min()) * 100.0),
            "finest_rel_diff": float(v[-1])}


def run_design_space(genes, cfg=DEFAULT_CONFIG, n=12, seed=7, max_draws=6000,
                     target_drop_mm=None):
    """*** THE M5 HEADLINE ***  Is the GNL correction one number, or a design variable?

    The off-ramp — "measure the correction once, then run Stage 3 on the linear model" —
    exists if and only if the correction is roughly constant over the designs the
    optimizer visits.  This is the same question M4 asked of the beam model, and it needs
    the same control, because the correction obviously grows with deflection and a raw
    sample would only rediscover that.

    THE CONTROL: each drawn genome's load is scaled so its LINEAR axle drop equals the
    shipped genome's.  Every row then sits at the same global deformation, and what is
    left is the design dependence.  Without this the sample is worthless — feasible
    random genomes are 3-20x stiffer than the design point, so their corrections are
    small for a reason that has nothing to do with the question.

    If the corrections were a function of the axle drop alone, the scaled column would be
    a constant.  It is not: it spans about an order of magnitude, so no single factor
    converts a linear axle drop into a nonlinear one.  `iso_load_scale` is reported per
    row so the reader can see how hard each genome had to be pushed.
    """
    from study_mesh_quality import latin_hypercube
    import wheel_genome as GN

    if target_drop_mm is None:
        target_drop_mm = _drop(WW.build_wheel(genes, cfg), "linear")["axle_drop_mm"]

    low, high, _ = GN.bounds_arrays(W.GENE_SPACE)
    rows, drawn, batch = [], 0, 0
    while len(rows) < n and drawn < max_draws:
        for vec in latin_hypercube(512, low, high, seed=seed + batch):
            drawn += 1
            v = np.asarray(vec, dtype=float)
            _, loss = W.evaluate_design(v)
            # The same four barriers the GA enforces, so every sample is a genome the
            # optimizer could have returned.
            if any(loss[t] > 0.0 for t in ("x_order", "hub_overlap", "fold", "arrival")):
                continue
            try:
                rows.append(_gnl_row(v, cfg, target_drop_mm))
            except fem.NewtonDivergedError as exc:
                # A design that will not solve at the target deflection is itself a
                # finding — Stage 3 has to survive these — so it is recorded, not hidden.
                rows.append({"genes": [float(x) for x in v], "diverged": str(exc)})
            if len(rows) >= n:
                break
        batch += 1

    shipped = _gnl_row(genes, cfg, target_drop_mm, keep_genes=False)
    ok = [r for r in rows if "diverged" not in r]
    out = {"rows": rows, "shipped": shipped, "n_drawn": drawn,
           "n_diverged": len(rows) - len(ok),
           "target_drop_mm": float(target_drop_mm)}
    if len(ok) >= 3:
        iso = np.array([r["iso_rel_diff"] for r in ok] + [shipped["iso_rel_diff"]])
        raw = np.array([r["service_rel_diff"] for r in ok]
                       + [shipped["service_rel_diff"]])
        out.update({
            "iso_rel_diff_min": float(iso.min()), "iso_rel_diff_max": float(iso.max()),
            "iso_rel_diff_ratio": float(iso.max() / iso.min()),
            "iso_rel_diff_cv": float(iso.std() / iso.mean()),
            "service_rel_diff_min": float(raw.min()),
            "service_rel_diff_max": float(raw.max()),
            # The off-ramp needs ONE number.  10% dispersion would already be generous
            # beside the effects this project chases; the same bar M4 used.
            "correction_factor_is_defensible": bool(iso.std() / iso.mean() < 0.10),
        })
    return out


def _gnl_row(v, cfg, target_drop_mm, keep_genes=True):
    """One genome, measured twice: at service load, and at a matched axle drop."""
    mesh = WW.build_wheel(v, cfg)
    lin = _drop(mesh, "linear")
    nl = _drop(mesh, "svk")
    scale = target_drop_mm / lin["axle_drop_mm"]
    iso_lin = _drop(mesh, "linear", SERVICE_FORCE_N * scale)
    iso_nl = _drop(mesh, "svk", SERVICE_FORCE_N * scale)
    row = {
        "linear_mm": lin["axle_drop_mm"],
        "svk_mm": nl["axle_drop_mm"],
        "service_rel_diff": float(nl["axle_drop_mm"] / lin["axle_drop_mm"] - 1.0),
        "iso_load_scale": float(scale),
        "iso_linear_mm": iso_lin["axle_drop_mm"],
        "iso_rel_diff": float(iso_nl["axle_drop_mm"] / iso_lin["axle_drop_mm"] - 1.0),
        "rim_share": lin["compliance_split"]["rim"],
        "free_arc_fraction": float(WW.spoke_free_arc_fraction(v, cfg)),
        "iterations": int(iso_nl["newton"]["total_iterations"]),
    }
    if keep_genes:
        row["genes"] = [float(x) for x in v]
    return row


def run_energy_identity(genes, cfg=DEFAULT_CONFIG):
    """2U/(F*delta) is exactly 1 for a linear body and is NOT under SVK.

    The M4 test suite ties the reported energies to the reported displacement through
    this identity, so it has to be restated rather than reused: under nonlinear
    kinematics the load-displacement curve is not a straight line and the stored energy
    is no longer half the load times the drop.  The departure is a second, independent
    measure of how nonlinear the problem is — and it is computed from the energy, not
    from the displacement, so it cannot be an artefact of how the axle drop is picked
    off the mesh.
    """
    mesh = WW.build_wheel(genes, cfg)
    out = {}
    for k in ("linear", "svk"):
        r = _drop(mesh, k)
        out[k] = {"energy_mJ": r["strain_energy_mJ"],
                  "axle_drop_mm": r["axle_drop_mm"],
                  "two_u_over_f_delta": float(2.0 * r["strain_energy_mJ"]
                                              / (SERVICE_FORCE_N * r["axle_drop_mm"])),
                  "compliance_split": r["compliance_split"]}
    out["linear_identity_error"] = abs(out["linear"]["two_u_over_f_delta"] - 1.0)
    out["svk_identity_error"] = abs(out["svk"]["two_u_over_f_delta"] - 1.0)
    return out


def run_stress(genes, cfg=DEFAULT_CONFIG):
    """Does GNL move the stress the way it moves the displacement?

    Reported on the plain spoke block's p99, which is the only converged stress number
    this mesh produces — the junction corners are unfilleted re-entrant singularities and
    their pointwise maximum diverges under refinement (see `study_wheel_fea.py`).

    The SVK number is the CAUCHY stress, pushed forward from the 2nd Piola-Kirchhoff
    stress the constitutive law returns.  Comparing S against the linear kernel's sigma
    would mix a frame effect worth a couple of percent into a measurement of the same
    size.
    """
    mesh = WW.build_wheel(genes, cfg)
    lam, mu = fem.lame(W.YOUNGS_MODULUS_PLA_MPA, fem.POISSON_RATIO_PLA)
    out = {}
    for k, nl in (("linear", False), ("svk", True)):
        res = _drop(mesh, k)
        st = fem.gauss_stresses(np.asarray(mesh.coords), mesh.conn, res["u"],
                                order=mesh.cfg.order, lam=lam, mu=mu, nonlinear=nl)
        vm = st["von_mises"][mesh.element_block == "spoke"]
        out[k] = {"spoke_block_p99_mpa": float(np.percentile(vm, 99.0)),
                  "spoke_block_max_mpa": float(vm.max())}
    out["p99_rel_diff"] = float(out["svk"]["spoke_block_p99_mpa"]
                                / out["linear"]["spoke_block_p99_mpa"] - 1.0)
    return out


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def _print(rep):
    p = print
    p("\n" + "=" * 78)
    p("  M5 GATE — GEOMETRIC NONLINEARITY")
    p("=" * 78)

    fi = rep["frame_indifference"]
    p("\nG1  FRAME INDIFFERENCE, whole assembled wheel")
    p(f"    {'angle':>8} {'SVK energy':>14} {'linear energy':>14} {'ratio':>10}")
    for r in fi["rows"]:
        p(f"    {r['angle_deg']:8.1f} {r['svk_energy_mJ']:14.3e} "
          f"{r['linear_energy_mJ']:14.4f} {r['ratio']:10.2e}")
    p(f"    worst ratio {fi['worst_ratio']:.2e} vs {GATE_ROTATION_ENERGY_RATIO:.0e}"
      f"   -> {'PASS' if fi['pass'] else 'FAIL'}")

    ll = rep["load_ladder"]
    p("\nG2  LOAD LADDER — linear vs SVK axle drop")
    p(f"    {'load':>6} {'linear mm':>11} {'SVK mm':>11} {'diff %':>9} "
      f"{'its':>4} {'bt':>3}  criterion")
    for r in ll["rows"]:
        p(f"    {r['load_fraction']:6.2f} {r['linear_mm']:11.5f} {r['svk_mm']:11.5f} "
          f"{100 * r['rel_diff']:9.4f} {r['iterations']:4d} {r['backtracks']:3d}  "
          f"{r['criterion']}")
    p(f"    at {ll['small_load_fraction']:.0%} of service load the two agree to "
      f"{100 * ll['small_load_rel_diff']:.4f}%  (gate {100 * GATE_SMALL_LOAD_REL:.1f}%)")
    p(f"    fitted exponent {ll['fitted_exponent']:.3f} (first order = 1), "
      f"every point softer: {'yes' if ll['all_softer'] else 'NO'}"
      f"   -> {'PASS' if ll['pass'] else 'FAIL'}")

    nh = rep["newton"]
    p("\nG3  NEWTON HEALTH")
    p(f"    {'steps':>6} {'axle drop mm':>14} {'its':>4} {'bt':>3} {'residual':>10}"
      f" {'s':>6}")
    for r in nh["continuation"]:
        p(f"    {r['steps']:6d} {r['axle_drop_mm']:14.9f} {r['iterations']:4d} "
          f"{r['backtracks']:3d} {r['residual_rel']:10.2e} {r['seconds']:6.2f}")
    p(f"    continuation spread {nh['continuation_spread']:.2e} "
      f"(gate {GATE_CONTINUATION_REL:.0e})")
    p(f"    residual history {['%.1e' % x for x in nh['residual_history']]}")
    p(f"    observed order {nh['observed_order']:.2f} (quadratic = 2)")
    p(f"    warm start from the linear solution: {nh['cold_iterations']} -> "
      f"{nh['warm_iterations']} iterations"
      f"   -> {'PASS' if nh['pass'] else 'FAIL'}")

    ml = rep["mesh_ladder"]
    p("\n    correction under mesh refinement")
    for r in ml["rows"]:
        p(f"      {r['config']:>7} {r['n_elements']:7d} el   "
          f"{100 * r['rel_diff']:7.3f} %")
    p(f"      spread {ml['spread_points']:.3f} points")

    ds = rep["design_space"]
    p("\nG4  *** THE HEADLINE *** IS THE CORRECTION ONE NUMBER?")
    p(f"    control: every genome loaded to the same "
      f"{ds['target_drop_mm']:.3f} mm LINEAR axle drop")
    p(f"    {'lin mm':>9} {'GNL @service':>13} {'load x':>8} {'GNL @iso drop':>14} "
      f"{'free arc':>9}")
    for r in ds["rows"] + [ds["shipped"]]:
        if "diverged" in r:
            p(f"    {'--':>9} {'DIVERGED':>13}")
            continue
        p(f"    {r['linear_mm']:9.4f} {100 * r['service_rel_diff']:12.3f}% "
          f"{r['iso_load_scale']:8.2f} {100 * r['iso_rel_diff']:13.3f}% "
          f"{r['free_arc_fraction']:9.3f}")
    if "iso_rel_diff_ratio" in ds:
        p(f"    at matched deflection the correction spans "
          f"{100 * ds['iso_rel_diff_min']:.2f}% to "
          f"{100 * ds['iso_rel_diff_max']:.2f}%  "
          f"(a factor of {ds['iso_rel_diff_ratio']:.1f}), cv "
          f"{ds['iso_rel_diff_cv']:.2f}")
        p(f"    one correction factor would be defensible: "
          f"{'YES' if ds['correction_factor_is_defensible'] else 'NO'}")

    ei = rep["energy_identity"]
    p("\n    2U/(F*delta) — exactly 1 for a linear body, and a second measure of GNL")
    for k in ("linear", "svk"):
        p(f"      {k:>7}  U {ei[k]['energy_mJ']:9.4f} mJ   "
          f"delta {ei[k]['axle_drop_mm']:8.5f} mm   "
          f"2U/Fd {ei[k]['two_u_over_f_delta']:.6f}")

    st = rep["stress"]
    p(f"\n    plain-spoke p99 von Mises: {st['linear']['spoke_block_p99_mpa']:.3f} MPa "
      f"linear -> {st['svk']['spoke_block_p99_mpa']:.3f} MPa SVK "
      f"({100 * st['p99_rel_diff']:+.2f}%)")

    p(f"\n  VERDICT")
    p(f"    GNL at service load: {100 * rep['gnl_service_pct']:.2f}% softer, against the "
      f"plan's {100 * PLAN_GNL_THRESHOLD:.0f}% threshold")
    p(f"    the linear-trajectory off-ramp is "
      f"{'OPEN' if rep['linear_trajectory_is_enough'] else 'CLOSED'}")
    p(f"\n  OVERALL: {'PASS' if rep['pass'] else 'FAIL'}")
    p(f"\n  NOT DONE: M4b still.  Every absolute number rests on E = 2300 MPa and an")
    p(f"            unconditional 0.80 FFF knockdown, both uncalibrated — a +-25%")
    p(f"            band on E swamps the 4% measured here for ABSOLUTE stiffness,")
    p(f"            though not for the design-to-design COMPARISON that Stage 3 makes.")


def main():
    ap = argparse.ArgumentParser(description="M5 geometric-nonlinearity gate")
    ap.add_argument("--genome", default="best_solution.json")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default="study_gnl.json")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="shorter sweeps and a coarser ladder; for CI")
    args = ap.parse_args()

    genes = load_genes(args.genome)
    t0 = time.time()
    rep = {
        "frame_indifference": run_frame_indifference(genes, "smoke"),
        "load_ladder": run_load_ladder(
            genes, args.config,
            (0.01, 0.25, 1.0, 2.0) if args.quick
            else (0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)),
        "newton": run_newton_health(genes, args.config,
                                    (1, 2, 4) if args.quick else (1, 2, 4, 8)),
        "mesh_ladder": run_mesh_ladder(
            genes, ("smoke", "coarse") if args.quick else ("smoke", "coarse", "medium")),
        "design_space": run_design_space(genes, args.config,
                                         n=6 if args.quick else 12),
        "energy_identity": run_energy_identity(genes, args.config),
        "stress": run_stress(genes, args.config),
    }
    rep["gnl_service_pct"] = rep["load_ladder"]["service_rel_diff"]
    # The plan's off-ramp needs BOTH: the effect small enough to ignore, AND constant
    # enough that a one-off correction would be meaningful if it were not.
    rep["linear_trajectory_is_enough"] = bool(
        abs(rep["gnl_service_pct"]) < PLAN_GNL_THRESHOLD
        and rep["design_space"].get("correction_factor_is_defensible", False))
    rep["pass"] = bool(rep["frame_indifference"]["pass"]
                       and rep["load_ladder"]["pass"]
                       and rep["newton"]["pass"])
    rep["settings"] = {"config": args.config, "genome": args.genome,
                       "service_force_n": SERVICE_FORCE_N,
                       "elapsed_s": round(time.time() - t0, 1)}
    _print(rep)
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"\nwrote {os.path.join(HERE, args.out)}  "
          f"({rep['settings']['elapsed_s']} s)")
    if not args.no_plot:
        try:
            print(f"wrote {_plot(rep, os.path.splitext(args.out)[0] + '.jpg')}")
        except Exception as exc:                            # pragma: no cover
            print(f"(plot skipped: {exc})")
    return 0 if rep["pass"] else 1


def _plot(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ll = rep["load_ladder"]["rows"]
    f = np.array([r["load_fraction"] for r in ll])
    ax[0].plot(f, [r["linear_mm"] for r in ll], "o-", label="linear")
    ax[0].plot(f, [r["svk_mm"] for r in ll], "s-", label="SVK")
    ax[0].set_xlabel("load / service load")
    ax[0].set_ylabel("axle drop (mm)")
    ax[0].set_title("the load-deflection curve bends")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].loglog(f, 100 * np.abs([r["rel_diff"] for r in ll]), "o-")
    ax[1].loglog(f, 100 * abs(ll[0]["rel_diff"]) * (f / f[0]), "k--",
                 label="first order")
    ax[1].set_xlabel("load / service load")
    ax[1].set_ylabel("SVK - linear (%)")
    ax[1].set_title(f"exponent {rep['load_ladder']['fitted_exponent']:.2f}")
    ax[1].legend()
    ax[1].grid(alpha=0.3, which="both")

    ds = rep["design_space"]
    ok = [r for r in ds["rows"] if "diverged" not in r] + [ds["shipped"]]
    ax[2].scatter([r["linear_mm"] for r in ok],
                  [100 * r["service_rel_diff"] for r in ok],
                  label="at service load")
    ax[2].scatter([ds["target_drop_mm"]] * len(ok),
                  [100 * r["iso_rel_diff"] for r in ok], marker="x",
                  label=f"at matched {ds['target_drop_mm']:.2f} mm drop")
    ax[2].set_xlabel("linear axle drop (mm)")
    ax[2].set_ylabel("GNL correction (%)")
    ax[2].set_title("the correction is not a function of the drop")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, path)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
