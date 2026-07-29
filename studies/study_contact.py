"""
=============================================================================
  M6 GATE — REAL CONTACT, AND WHAT THE ASSUMED PATCH WAS ACTUALLY WORTH
=============================================================================
    .venv-opt/bin/python studies/study_contact.py

Every full-wheel number before this rested on `CONTACT_PATCH_HALF_DEG = 3.0` — a
constant someone chose, with the ground load spread over it as an elliptical vertical
traction.  M4 measured what that assumption was worth by sweeping it: over a 12x range
the axle drop moved 9.7%, larger than the entire 3.95% geometric-nonlinearity correction
M5 went on to measure.  Unlike that correction, it was not a property of the wheel but a
free parameter.  M6 replaces it with penalty contact against a rigid frictionless ground,
so the patch and the pressure distribution come out of the solve.

THE HEADLINE, AND IT BREAKS THE PATTERN OF THE LAST TWO GATES
------------------------------------------------------------
The assumption was badly wrong about the patch and almost exactly right about the answer.

The real patch is a half-angle of about 0.47 deg, not 3.0 — the assumed patch was SIX
TIMES too wide, and the truth sits nearer the Hertz solid-cylinder bound of 0.31 deg that
M4 explicitly described as a lower bound and expected to be exceeded "far".  And yet the
axle drop moves 1.3%, from 1.676 mm assumed to 1.655 mm measured, because the drop is
dominated by spoke and rim bending rather than by how the last few newtons are spread.

That is worth stating plainly because it is the opposite of what M4 and M5 found.  Those
gates each found an assumed constant standing in for a design variable that ranged over
3-10x, and each closed an off-ramp.  This one confirms an assumption's CONSEQUENCE while
refuting the assumption itself.  A prior written down before the run — "expect the patch
to vary by 3-10x across designs, like everything else has" — was half right: the patch
does vary, but the axle drop's dependence on it is weak enough that it does not matter.

WHAT REAL CONTACT ADDS THAT THE FIXED PATCH COULD NOT REPRESENT AT ALL
----------------------------------------------------------------------
The patch MIGRATES as the wheel rolls.  Its centre swings from -1.21 deg to +1.20 deg
over one 30-degree period, passing through zero where the wheel is stiffest.  The fixed
model pinned the patch at the bottom of the rim by construction, so this effect was not
small in it — it was absent.  It is also the mechanism behind the phase ripple: the
contact point tracks toward the stiff spoke locations.

The ripple itself barely moves: 18.6% peak-to-peak under real contact against M4's 19.8%
with the patch held fixed.  A prediction written into the plan — that real contact would
LOWER the ripple materially, because the rim between spokes is softer and would flatten
more, widening the patch and recruiting more spokes — is not supported.  The
self-levelling effect exists and is worth about a point.  Phase remains a first-class
design objective and Stage 3 still needs the phase quadrature.

TWO NUMBERS THAT ARE NOT NUMBERS, FOR THE SAME REASON AS M4'S PEAK STRESS
-------------------------------------------------------------------------
The peak contact PRESSURE and the patch extent measured from the live quadrature points
are both SAMPLING STATISTICS.  They can only report an edge at one of `n_quad` places per
segment, so refining the quadrature moves them while the displacement field stands still:
6 -> 20 points per segment moves the sampled half-angle from 1.609 to 1.677 deg and the
peak pressure from 5.18 to 5.53 MPa on the medium mesh, where the axle drop moves by 3e-5
relative.  `RigidGroundContact.patch_extent` therefore measures the patch from the ZERO
CROSSING of the gap, which is a property of the solution; the sampled versions are
reported as diagnostics and must not be quoted.

THE M7 BRIDGE, ANSWERED EARLY AND IN THE UNEXPECTED DIRECTION
--------------------------------------------------------------
The plan flagged a hazard: `<z>^2/2` is C^1 but not C^2, so a finite difference in a gene
taken across a change in the contact set is meaningless — the "gene with no
finite-difference plateau" failure M7 gates on.  A C^2 smoothed bracket was carried as
the fallback.

It is not needed.  The sharp bracket has a clean plateau over three decades of step size
(stable to ~1e-5 relative from h=1e-2 down to 1e-5), and switching on the smoothing moves
the derivative by only 0.1%.  The reason is that the kinks are at DISCRETE gene values —
where a quadrature point crosses the surface — so a generic design is not near one.  The
plateau therefore exists almost everywhere and can still fail on the measure-zero set
where a point sits exactly at the patch edge.  `smoothing_mm` is kept for that case and
defaults to 0.0.
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
RIM_BAND_MM = WW.RIM_OUTER_RADIUS_MM - WW.RIM_RADIUS_MM

# Written down BEFORE the study was run, per the project's rule.
GATE_EPS_PLATEAU_REL = 1.0e-3     # axle drop across >= 2 decades of penalty stiffness
GATE_PENETRATION_FRAC = 1.0e-3    # peak penetration / rim band thickness
GATE_EQUILIBRIUM_N = 1.0e-6       # hub reaction vs contact resultant
GATE_HORIZONTAL_N = 1.0e-9        # frictionless flat ground: no side load, at all
GATE_PERIODICITY_REL = 1.0e-9     # 12-fold periodicity of the axle drop
GATE_CONTINUATION_REL = 1.0e-9    # 1 indentation step vs 4 must give the same state
GATE_FD_PLATEAU_REL = 1.0e-3      # best consecutive agreement in the FD ladder


def load_genes(path="best_solution.json"):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


# ---------------------------------------------------------------------------
# G1 — THE PENALTY STIFFNESS IS A PLATEAU, NOT A CHOICE
# ---------------------------------------------------------------------------

def run_penalty_plateau(genes, cfg=DEFAULT_CONFIG,
                        eps_values=(1e2, 1e3, 1e4, 1e5)):
    """Sweep eps_n and require a range over which the answer does not care.

    Too soft and the penetration pollutes the answer; too stiff and the conditioning
    blows up — and there is less headroom here than usual, since M5 established the
    reduced tangent already runs at kappa ~ 1e8-1e9.  So `eps_n` is not a number to pick
    but a plateau to demonstrate: consecutive decades must agree, and the penetration
    must be negligible against the 1.5 mm rim band it is denting.

    Iteration counts are reported because they are the cost of the plateau: the active-set
    search lengthens as the contact law stiffens, which is what eventually ends it.

    THE FIRST VERSION OF THIS GATE ASKED FOR THE WRONG THING, AND MEASUREMENT SAID SO.
    It required two CONSECUTIVE decades to agree within 1e-3.  That is unmeetable by
    construction here, and not because the model is bad: a penalty method's error in the
    axle drop simply IS the penetration, which falls linearly in 1/eps_n.  Measured, the
    drop moves 1.96e-3 mm between eps_n 1e3 and 1e4 while the penetration falls by
    2.17e-3 mm — the same number.  So demanding agreement across two decades demands the
    penalty error already be negligible at the SOFTER of them, where it is deliberately
    not.  The question that matters is whether the DEFAULT is converged, so that is what
    is asked: the default must agree with the next stiffer decade, and the penetration
    must be negligible against the band.  Both are one-sided and anchored where the
    answer is actually taken.
    """
    mesh = WW.build_wheel(genes, cfg)
    rows = []
    for eps in eps_values:
        t0 = time.time()
        try:
            r = fem.solve_wheel_contact(mesh, eps_n=float(eps))
        except Exception as exc:                              # pragma: no cover
            rows.append({"eps_n": float(eps), "failed": str(exc)[:200]})
            continue
        rows.append({
            "eps_n": float(eps),
            "axle_drop_mm": r["axle_drop_mm"],
            "contact_force_n": r["contact_force_n"],
            "max_penetration_mm": r["max_penetration_mm"],
            "penetration_frac_of_band": r["max_penetration_mm"] / RIM_BAND_MM,
            "patch_half_deg": r["patch_half_deg"],
            "iterations": int(r["newton"]["total_iterations"]),
            "seconds": round(time.time() - t0, 2),
        })
    ok = sorted((r for r in rows if "failed" not in r), key=lambda r: r["eps_n"])
    # Reported for context, but NOT the criterion — see the docstring.
    best, run = 0, 0
    for a, b in zip(ok[:-1], ok[1:]):
        if abs(b["axle_drop_mm"] / a["axle_drop_mm"] - 1.0) < GATE_EPS_PLATEAU_REL:
            run += 1
            best = max(best, run)
        else:
            run = 0

    idx = [i for i, r in enumerate(ok)
           if r["eps_n"] == fem.DEFAULT_CONTACT_EPS_N]
    converged, residual_rel = False, float("nan")
    pen_default = float("nan")
    if idx and idx[0] + 1 < len(ok):
        d, nxt = ok[idx[0]], ok[idx[0] + 1]
        pen_default = d["penetration_frac_of_band"]
        residual_rel = abs(nxt["axle_drop_mm"] / d["axle_drop_mm"] - 1.0)
        converged = residual_rel < GATE_EPS_PLATEAU_REL

    return {
        "rows": rows,
        "plateau_decades": int(best),
        "default_eps_n": float(fem.DEFAULT_CONTACT_EPS_N),
        "penetration_frac_at_default": pen_default,
        # How far the default still is from the eps_n -> infinity limit, estimated by the
        # next stiffer decade.  This is the number that says whether the SHIPPED setting
        # is converged, which is the only one the rest of the gate depends on.
        "default_vs_next_decade_rel": residual_rel,
        "worst_penetration_frac": float(
            max((r["penetration_frac_of_band"] for r in ok), default=1.0)),
        "pass": bool(converged and pen_default < GATE_PENETRATION_FRAC),
    }


# ---------------------------------------------------------------------------
# G2 — THE INVARIANTS
# ---------------------------------------------------------------------------

def run_verification(genes, cfg=DEFAULT_CONFIG):
    """Equilibrium, no tension, no side load, periodicity, continuation independence.

    The no-tension check is the one that earns its place.  A penalty law with a sign slip
    PULLS the wheel down onto the ground outside the patch, and that is invisible to every
    resultant check in the file — the total still balances, because the same wrong sign
    appears on both sides.  Only the pointwise pressure sees it.
    """
    mesh = WW.build_wheel(genes, cfg)
    r = fem.solve_wheel_contact(mesh)
    xy = np.asarray(mesh.coords)
    prob = fem.wheel_contact_problem(mesh, indentation_mm=r["axle_drop_mm"])
    p = prob.contact.pressure(xy, r["u"])

    # 12-fold periodicity.  Under contact this tests more than the sector indexing: the
    # load itself is now an output, so the whole contact search has to be periodic too.
    per = []
    for phase in (0.0, 7.0):
        a = fem.solve_wheel_contact(WW.build_wheel(genes, cfg, phase_deg=phase))
        b = fem.solve_wheel_contact(WW.build_wheel(genes, cfg, phase_deg=phase + 30.0))
        per.append({"phase_deg": phase,
                    "delta_mm": a["axle_drop_mm"],
                    "delta_plus_30_mm": b["axle_drop_mm"],
                    "rel_diff": abs(a["axle_drop_mm"] / b["axle_drop_mm"] - 1.0)})
    worst_per = max(x["rel_diff"] for x in per)

    # Continuation independence, on the indentation ramp rather than on a load factor.
    cont = [fem.solve_wheel_contact_at(mesh, r["axle_drop_mm"], steps=s)
            ["contact_force_n"] for s in (1, 2, 4)]
    cont_spread = float(max(cont) / min(cont) - 1.0)

    return {
        "axle_drop_mm": r["axle_drop_mm"],
        "contact_force_n": r["contact_force_n"],
        "target_force_n": float(SERVICE_FORCE_N),
        "hub_reaction_n": r["hub_reaction_n"],
        "equilibrium_error_n": r["equilibrium_error_n"],
        "horizontal_resultant_n": abs(r["contact_resultant_n"][0]),
        "min_pressure_mpa": r["min_pressure_mpa"],
        "n_quad_points_in_contact": r["n_quad_points_in_contact"],
        "centre_node_rise_mm": r["centre_node_rise_mm"],
        "indentation_vs_centre_node_mm": abs(r["axle_drop_mm"]
                                             - r["centre_node_rise_mm"]),
        "max_penetration_mm": r["max_penetration_mm"],
        "periodicity": per,
        "worst_periodicity_rel": float(worst_per),
        "continuation_force_n": cont,
        "continuation_spread": cont_spread,
        "secant_iterations": int(r["secant"]["iterations"]),
        "pass": bool(r["equilibrium_error_n"] < GATE_EQUILIBRIUM_N
                     and abs(r["contact_resultant_n"][0]) < GATE_HORIZONTAL_N
                     and r["min_pressure_mpa"] >= 0.0
                     and worst_per < GATE_PERIODICITY_REL
                     and cont_spread < GATE_CONTINUATION_REL
                     and float(np.min(p["pressure_mpa"])) >= 0.0),
    }


# ---------------------------------------------------------------------------
# G3 — *** THE HEADLINE *** THE EMERGENT PATCH VS THE ASSUMED ONE
# ---------------------------------------------------------------------------

def run_emergent_patch(genes, configs=("smoke", "coarse", "medium")):
    """What the patch actually is, and what assuming 3.0 deg cost.

    Both models are run on the same mesh at the same service load, so the difference is
    the load model and nothing else.  The `n_quad` column exists to show which quantities
    are sampling statistics: the axle drop is insensitive to it, the sampled patch and the
    peak pressure are not.
    """
    rows = []
    for cfg in configs:
        mesh = WW.build_wheel(genes, cfg)
        assumed = fem.solve_wheel(mesh)
        for nq in (6, 20):
            c = fem.solve_wheel_contact(mesh, n_quad=nq)
            rows.append({
                "config": cfg,
                "n_elements": int(mesh.n_elements),
                "n_rim_segments": int(len(mesh.edge_sets["rim_outer"])),
                "deg_per_segment": 360.0 / len(mesh.edge_sets["rim_outer"]),
                "n_quad": int(nq),
                "contact_drop_mm": c["axle_drop_mm"],
                "assumed_drop_mm": assumed["axle_drop_mm"],
                "rel_diff": float(c["axle_drop_mm"] / assumed["axle_drop_mm"] - 1.0),
                "patch_half_deg": c["patch_half_deg"],
                "patch_centre_deg": c["patch_centre_deg"],
                "patch_half_deg_sampled": c["patch_half_deg_sampled"],
                "peak_pressure_mpa_sampled": c["peak_pressure_mpa_sampled"],
                "n_quad_points_in_contact": c["n_quad_points_in_contact"],
                "contact_split": c["compliance_split"],
                "assumed_split": assumed["compliance_split"],
            })
    fine = [r for r in rows if r["config"] == configs[-1]]

    def _sens(key):
        vals = [r[key] for r in fine]
        return float(max(vals) / min(vals) - 1.0) if min(vals) > 0 else float("nan")
    return {
        "rows": rows,
        "hertz_half_deg": float(fem.hertz_patch_half_angle_deg()),
        "assumed_half_deg": float(fem.CONTACT_PATCH_HALF_DEG),
        "measured_half_deg": float(np.mean([r["patch_half_deg"] for r in fine])),
        "assumed_over_measured": float(fem.CONTACT_PATCH_HALF_DEG
                                       / np.mean([r["patch_half_deg"] for r in fine])),
        "drop_rel_diff_at_finest": float(np.mean([r["rel_diff"] for r in fine])),
        "quadrature_sensitivity": {
            "axle_drop": _sens("contact_drop_mm"),
            "patch_half_deg": _sens("patch_half_deg"),
            "patch_half_deg_sampled": _sens("patch_half_deg_sampled"),
            "peak_pressure_sampled": _sens("peak_pressure_mpa_sampled"),
        },
        # The robust statement about the sampled measures is their BIAS, not their
        # scatter.  Sensitivity to `n_quad` is not a reliable discriminator — the two
        # patch measures move by comparable amounts and which is larger depends on the
        # config.  The DIRECTION is systematic: a Gauss point counts as in contact at any
        # penetration however small, so the sampled edge always lands outside the true
        # one, while a sampled maximum can only miss the true peak.
        "sampled_patch_overstates_by": float(
            np.mean([r["patch_half_deg_sampled"] / r["patch_half_deg"] for r in fine])),
    }


# ---------------------------------------------------------------------------
# G4 — PHASE, AND THE PREDICTION THAT DID NOT HOLD
# ---------------------------------------------------------------------------

def run_phase(genes, cfg=DEFAULT_CONFIG, n=13):
    """Axle drop and patch position over one 30-degree period, under real contact.

    Compared against M4's fixed-patch ripple on the same 13-point basis.  The plan
    predicted the ripple would fall materially under real contact; it does not.  What the
    fixed model genuinely could not represent is the patch MIGRATING, which is reported
    here as well and is the mechanism behind the ripple rather than a separate effect.
    """
    rows = []
    for phase in np.linspace(0.0, 30.0, n):
        mesh = WW.build_wheel(genes, cfg, phase_deg=float(phase))
        c = fem.solve_wheel_contact(mesh)
        a = fem.solve_wheel(mesh)
        rows.append({"phase_deg": float(phase),
                     "contact_drop_mm": c["axle_drop_mm"],
                     "assumed_drop_mm": a["axle_drop_mm"],
                     "patch_half_deg": c["patch_half_deg"],
                     "patch_centre_deg": c["patch_centre_deg"]})
    d = np.array([r["contact_drop_mm"] for r in rows])
    a = np.array([r["assumed_drop_mm"] for r in rows])
    ctr = np.array([r["patch_centre_deg"] for r in rows])
    return {
        "rows": rows,
        "contact_ripple_std_over_mean": float(d.std() / d.mean()),
        "contact_peak_to_peak_over_mean": float((d.max() - d.min()) / d.mean()),
        "assumed_ripple_std_over_mean": float(a.std() / a.mean()),
        "assumed_peak_to_peak_over_mean": float((a.max() - a.min()) / a.mean()),
        "patch_centre_range_deg": [float(ctr.min()), float(ctr.max())],
        "patch_centre_swing_deg": float(ctr.max() - ctr.min()),
    }


# ---------------------------------------------------------------------------
# G5 — IS THE PATCH A DESIGN VARIABLE?
# ---------------------------------------------------------------------------

def run_design_space(genes, cfg=DEFAULT_CONFIG, n=8, seed=11, max_draws=4000):
    """The prior said the patch would vary 3-10x like everything else.  Does it?

    And, more to the point for Stage 2: does the DROP care?  Those are different
    questions, and this gate is the first where they come apart — the patch varies and
    the consequence does not follow it.
    """
    from study_mesh_quality import latin_hypercube

    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    rows, drawn, batch = [], 0, 0
    while len(rows) < n and drawn < max_draws:
        for vec in latin_hypercube(256, low, high, seed=seed + batch):
            drawn += 1
            v = np.asarray(vec, dtype=float)
            _, loss = W.evaluate_design(v)
            if any(loss[t] > 0.0 for t in ("x_order", "hub_overlap", "fold", "arrival")):
                continue
            try:
                mesh = WW.build_wheel(v, cfg)
                c = fem.solve_wheel_contact(mesh)
                a = fem.solve_wheel(mesh)
                rows.append({
                    "patch_half_deg": c["patch_half_deg"],
                    "contact_drop_mm": c["axle_drop_mm"],
                    "assumed_drop_mm": a["axle_drop_mm"],
                    "rel_diff": float(c["axle_drop_mm"] / a["axle_drop_mm"] - 1.0),
                    "free_arc_fraction": float(WW.spoke_free_arc_fraction(v, cfg)),
                })
            except Exception as exc:
                # A design that will not solve under contact is a finding in its own
                # right — Stage 3 has to survive these — so it is recorded, not hidden.
                rows.append({"failed": str(exc)[:160]})
            if len(rows) >= n:
                break
        batch += 1

    mesh = WW.build_wheel(genes, cfg)
    ship = fem.solve_wheel_contact(mesh)
    shipped = {
        "patch_half_deg": ship["patch_half_deg"],
        "contact_drop_mm": ship["axle_drop_mm"],
        "assumed_drop_mm": fem.solve_wheel(mesh)["axle_drop_mm"],
    }
    ok = [r for r in rows if "failed" not in r]
    out = {"rows": rows, "shipped": shipped, "n_drawn": drawn,
           "n_failed": len(rows) - len(ok)}
    if len(ok) >= 3:
        hp = np.array([r["patch_half_deg"] for r in ok])
        rd = np.array([r["rel_diff"] for r in ok])
        out.update({
            "patch_half_deg_min": float(hp.min()),
            "patch_half_deg_max": float(hp.max()),
            "patch_half_deg_ratio": float(hp.max() / hp.min()),
            "drop_rel_diff_min": float(rd.min()),
            "drop_rel_diff_max": float(rd.max()),
            "drop_rel_diff_absmax": float(np.abs(rd).max()),
            # The question Stage 2 actually needs answered: can the assumed-patch model
            # stand in for contact across the design space?
            "assumed_patch_is_adequate": bool(np.abs(rd).max() < 0.05),
        })
    return out


# ---------------------------------------------------------------------------
# G6 — CONTACT PLUS GEOMETRIC NONLINEARITY
# ---------------------------------------------------------------------------

def run_kinematics(genes, cfg=DEFAULT_CONFIG):
    """Both nonlinearities at once, which is what Stage 3 will actually run.

    M5 measured 3.95% softening under SVK with the patch held fixed.  If contact changed
    that materially, the two effects would be coupled and neither could be reasoned about
    on its own.
    """
    mesh = WW.build_wheel(genes, cfg)
    out = {}
    for k in ("linear", "svk"):
        r = fem.solve_wheel_contact(mesh, kinematics=k)
        out[k] = {"axle_drop_mm": r["axle_drop_mm"],
                  "patch_half_deg": r["patch_half_deg"],
                  "iterations": int(r["newton"]["total_iterations"]),
                  "equilibrium_error_n": r["equilibrium_error_n"]}
    out["svk_rel_diff"] = float(out["svk"]["axle_drop_mm"]
                                / out["linear"]["axle_drop_mm"] - 1.0)
    # M5's number, measured with the assumed patch — the comparison this exists to make.
    lin = fem.solve_wheel(mesh)["axle_drop_mm"]
    svk = fem.solve_wheel(mesh, kinematics="svk")["axle_drop_mm"]
    out["assumed_patch_svk_rel_diff"] = float(svk / lin - 1.0)
    out["coupling"] = float(out["svk_rel_diff"] - out["assumed_patch_svk_rel_diff"])
    return out


# ---------------------------------------------------------------------------
# G7 — THE M7 BRIDGE: IS THERE A FINITE-DIFFERENCE PLATEAU?
# ---------------------------------------------------------------------------

def run_gradient_plateau(genes, cfg=DEFAULT_CONFIG, gene_ids=(1, 8, 12, 13),
                         steps=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
                         smoothings=(0.0, 1e-3)):
    """Central differences of the contact force in a gene, over a ladder of step sizes.

    The differentiated quantity is the contact force at a FIXED indentation, not the axle
    drop at fixed force: the latter runs a secant loop whose own tolerance would enter the
    difference, which is a numerical artefact rather than a property of the model.

    A plateau is a run of consecutive steps agreeing to `GATE_FD_PLATEAU_REL`.  If the
    sharp bracket has one, the C^2 smoothing is unnecessary and should not be switched on
    — an unneeded smoothing band is exactly the sort of free parameter this gate exists to
    remove.

    SOME GENES HAVE AN IDENTICALLY ZERO DERIVATIVE, AND THAT IS A FINDING RATHER THAN A
    DEGENERATE CASE.  `R_hub` and `R_rim` are fillet radii, and the mesh does not model
    fillets (see the M2b gate) — so the FEA cannot see them at all, while the beam model
    prices them through `stress_concentration_kt`.  A gradient-based Stage 3 would find
    them perfectly flat and never move them.  They are classified here as INSENSITIVE
    rather than being allowed to masquerade as a clean plateau, which is what a run of
    identical zeros would otherwise look like.
    """
    genes = np.asarray(genes, dtype=float)
    delta = fem.solve_wheel(WW.build_wheel(genes, cfg))["axle_drop_mm"]

    def force(v, smooth):
        return fem.solve_wheel_contact_at(WW.build_wheel(np.asarray(v), cfg), delta,
                                          smoothing_mm=smooth)["contact_force_n"]

    out = {"indentation_mm": float(delta), "genes": {}}
    for smooth in smoothings:
        key = "sharp" if smooth == 0.0 else f"smoothed_{smooth:g}mm"
        out["genes"][key] = {}
        # The smoothing comparison establishes ONE number — how much switching it on
        # would move a derivative — so it only needs one gene.  Running it over all of
        # them would double the most expensive section of the gate to say the same thing
        # three times.
        ids = gene_ids if smooth == 0.0 else gene_ids[:1]
        for gi in ids:
            name = wg.GENE_NAMES[gi]
            derivs = []
            for h in steps:
                vp, vm = genes.copy(), genes.copy()
                vp[gi] += h
                vm[gi] -= h
                derivs.append((force(vp, smooth) - force(vm, smooth)) / (2.0 * h))
            d = np.array(derivs)
            # Insensitive: the gene does not enter the meshed geometry at all, so every
            # difference is exact zero.  Scaled against the load, not against an absolute
            # tolerance, so the classification does not depend on the units.
            insensitive = bool(np.abs(d).max() < 1e-9 * SERVICE_FORCE_N)
            if insensitive:
                rel = np.zeros(max(len(d) - 1, 0))
                best_run, best = 0, 0.0
            else:
                rel = np.abs(d[1:] / d[:-1] - 1.0)
                best_run, run = 0, 0
                for x in rel:
                    if x < GATE_FD_PLATEAU_REL:
                        run += 1
                        best_run = max(best_run, run)
                    else:
                        run = 0
                best = float(d[int(np.argmin(rel)) + 1])
            out["genes"][key][name] = {
                "steps": [float(h) for h in steps],
                "derivatives": [float(x) for x in d],
                "consecutive_rel": [float(x) for x in rel],
                "plateau_decades": int(best_run),
                "best_estimate": best,
                "insensitive": insensitive,
            }
    # How much the smoothing would change the answer, if it were switched on.  Only over
    # the genes that HAVE an answer — a ratio against an identically zero derivative is
    # not a large shift, it is undefined.
    sharp = out["genes"]["sharp"]
    live = [n for n, v in sharp.items() if not v["insensitive"]]
    shifts = {}
    for other in [k for k in out["genes"] if k != "sharp"]:
        shifts[other] = {n: float(out["genes"][other][n]["best_estimate"]
                                  / sharp[n]["best_estimate"] - 1.0)
                         for n in live if n in out["genes"][other]}
    out["smoothing_shift"] = shifts
    out["insensitive_genes"] = [n for n, v in sharp.items() if v["insensitive"]]
    out["worst_sharp_plateau"] = (int(min(sharp[n]["plateau_decades"] for n in live))
                                  if live else 0)
    out["smoothing_is_needed"] = bool(live and out["worst_sharp_plateau"] < 1)
    out["pass"] = bool(live and out["worst_sharp_plateau"] >= 1)
    return out


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def _print(rep):
    def head(s):
        print(f"\n{s}\n" + "-" * len(s))

    print("=" * 78)
    print("  M6 GATE — REAL CONTACT")
    print("=" * 78)

    g = rep["penalty"]
    head("G1  THE PENALTY STIFFNESS IS A PLATEAU")
    print(f"    {'eps_n':>9s} {'drop mm':>10s} {'force N':>9s} {'penetration':>12s} "
          f"{'/band':>9s} {'half deg':>9s} {'its':>4s}")
    for r in g["rows"]:
        if "failed" in r:
            print(f"    {r['eps_n']:9.0e}  FAILED  {r['failed'][:44]}")
            continue
        print(f"    {r['eps_n']:9.0e} {r['axle_drop_mm']:10.5f} "
              f"{r['contact_force_n']:9.4f} {r['max_penetration_mm']:12.3e} "
              f"{r['penetration_frac_of_band']:9.2e} {r['patch_half_deg']:9.4f} "
              f"{r['iterations']:4d}")
    print(f"    the axle drop's error IS the penetration, so it falls linearly in "
          f"1/eps_n rather than")
    print(f"    plateauing — the criterion is therefore one-sided and anchored at the "
          f"default:")
    print(f"        default eps_n {g['default_eps_n']:.0e} vs the next stiffer decade: "
          f"{g['default_vs_next_decade_rel']:.2e}  [< {GATE_EPS_PLATEAU_REL:.0e}]")
    print(f"        penetration there {g['penetration_frac_at_default']:.2e} of the "
          f"{RIM_BAND_MM:.1f} mm band  [< {GATE_PENETRATION_FRAC:.0e}]")
    print(f"    (widest run of consecutive agreeing decades, for context: "
          f"{g['plateau_decades']})")
    print(f"    -> {'PASS' if g['pass'] else 'FAIL'}")

    v = rep["verification"]
    head("G2  THE INVARIANTS")
    print(f"    contact force {v['contact_force_n']:.6f} N  vs target "
          f"{v['target_force_n']:.6f} N   (secant: {v['secant_iterations']} its)")
    print(f"    hub reaction  [{v['hub_reaction_n'][0]:+.4f}, "
          f"{v['hub_reaction_n'][1]:+.4f}] N   equilibrium error "
          f"{v['equilibrium_error_n']:.2e}  [< {GATE_EQUILIBRIUM_N:.0e}]")
    print(f"    horizontal resultant {v['horizontal_resultant_n']:.2e} N  "
          f"[< {GATE_HORIZONTAL_N:.0e}]  (frictionless flat ground has none)")
    print(f"    minimum pressure {v['min_pressure_mpa']:.3e} MPa  "
          f"[>= 0; a contact model that PULLS still balances globally]")
    print(f"    12-fold periodicity, worst {v['worst_periodicity_rel']:.2e}  "
          f"[< {GATE_PERIODICITY_REL:.0e}]")
    print(f"    indentation ramp 1 vs 2 vs 4 steps, spread "
          f"{v['continuation_spread']:.2e}  [< {GATE_CONTINUATION_REL:.0e}]")
    print(f"    prescribed indentation vs measured centre node: "
          f"{v['indentation_vs_centre_node_mm']:.3e} mm "
          f"(= the local penetration, {v['max_penetration_mm']:.3e})")
    print(f"    -> {'PASS' if v['pass'] else 'FAIL'}")

    e = rep["patch"]
    head("G3  *** THE HEADLINE *** THE EMERGENT PATCH VS THE ASSUMED 3.0 DEG")
    print(f"    {'cfg':>7s} {'nq':>3s} {'deg/seg':>8s} {'live':>5s} {'contact':>9s} "
          f"{'assumed':>9s} {'diff':>8s} {'half*':>7s} {'ctr*':>7s} {'half_s':>7s} "
          f"{'peak_s':>7s}")
    for r in e["rows"]:
        print(f"    {r['config']:>7s} {r['n_quad']:3d} {r['deg_per_segment']:8.3f} "
              f"{r['n_quad_points_in_contact']:5d} {r['contact_drop_mm']:9.5f} "
              f"{r['assumed_drop_mm']:9.5f} {100*r['rel_diff']:+7.3f}% "
              f"{r['patch_half_deg']:7.4f} {r['patch_centre_deg']:+7.4f} "
              f"{r['patch_half_deg_sampled']:7.3f} "
              f"{r['peak_pressure_mpa_sampled']:7.3f}")
    print(f"    * = from the gap's zero crossing (a property of the solution)")
    print(f"    _s = sampled at the quadrature points (a SAMPLING STATISTIC, not a "
          f"number)")
    print()
    print(f"    Hertz solid-cylinder bound   {e['hertz_half_deg']:.3f} deg")
    print(f"    measured                     {e['measured_half_deg']:.3f} deg")
    print(f"    ASSUMED                      {e['assumed_half_deg']:.3f} deg   "
          f"-> {e['assumed_over_measured']:.1f}x too wide")
    print(f"    and yet the axle drop moves {100*e['drop_rel_diff_at_finest']:+.2f}%")
    qs = e["quadrature_sensitivity"]
    print(f"    quadrature sensitivity (6 -> 20 pts/segment) at the finest mesh:")
    print(f"        axle drop           {qs['axle_drop']:.2e}   <- insensitive")
    print(f"        patch half (zero x) {qs['patch_half_deg']:.2e}")
    print(f"        patch half (sampled){qs['patch_half_deg_sampled']:.2e}")
    print(f"        peak pressure       {qs['peak_pressure_sampled']:.2e}")
    print(f"    but the reliable statement about the sampled measures is their BIAS:")
    print(f"        the sampled patch overstates the real one by "
          f"{e['sampled_patch_overstates_by']:.1f}x, because a Gauss point counts as in")
    print(f"        contact at any penetration, so its edge always lands outside the "
          f"true one")

    p = rep["phase"]
    head("G4  PHASE — THE RIPPLE, AND THE PATCH MIGRATING")
    print(f"    {'phase':>7s} {'contact':>9s} {'assumed':>9s} {'half':>7s} {'centre':>8s}")
    for r in p["rows"]:
        print(f"    {r['phase_deg']:7.2f} {r['contact_drop_mm']:9.5f} "
              f"{r['assumed_drop_mm']:9.5f} {r['patch_half_deg']:7.4f} "
              f"{r['patch_centre_deg']:+8.4f}")
    print(f"    ripple p2p/mean:  contact {100*p['contact_peak_to_peak_over_mean']:.1f}%"
          f"   assumed patch {100*p['assumed_peak_to_peak_over_mean']:.1f}%")
    print(f"    the plan predicted real contact would LOWER this materially; it does "
          f"not")
    print(f"    patch centre swings {p['patch_centre_range_deg'][0]:+.2f} to "
          f"{p['patch_centre_range_deg'][1]:+.2f} deg — an effect the fixed patch "
          f"could not represent at all")

    d = rep["design_space"]
    head("G5  IS THE PATCH A DESIGN VARIABLE?  AND DOES THE DROP CARE?")
    print(f"    {'half deg':>9s} {'contact':>9s} {'assumed':>9s} {'diff':>8s} "
          f"{'free arc':>9s}")
    for r in d["rows"]:
        if "failed" in r:
            print(f"    FAILED  {r['failed'][:60]}")
            continue
        print(f"    {r['patch_half_deg']:9.4f} {r['contact_drop_mm']:9.5f} "
              f"{r['assumed_drop_mm']:9.5f} {100*r['rel_diff']:+7.2f}% "
              f"{r['free_arc_fraction']:9.3f}")
    if "patch_half_deg_ratio" in d:
        print(f"    the patch varies {d['patch_half_deg_ratio']:.1f}x over the design "
              f"space ({d['patch_half_deg_min']:.3f} to "
              f"{d['patch_half_deg_max']:.3f} deg)")
        print(f"    but the assumed-patch model is never worse than "
              f"{100*d['drop_rel_diff_absmax']:.2f}% on the axle drop")
        print(f"    assumed patch adequate as a Stage-2 stand-in: "
              f"{'YES' if d['assumed_patch_is_adequate'] else 'NO'}")

    k = rep["kinematics"]
    head("G6  CONTACT PLUS GEOMETRIC NONLINEARITY")
    print(f"    with contact:       linear {k['linear']['axle_drop_mm']:.5f} -> "
          f"svk {k['svk']['axle_drop_mm']:.5f}   {100*k['svk_rel_diff']:+.2f}%")
    print(f"    with assumed patch: {100*k['assumed_patch_svk_rel_diff']:+.2f}%  "
          f"(M5's number)")
    print(f"    coupling between the two nonlinearities: "
          f"{100*k['coupling']:+.2f} points")

    f = rep["gradient"]
    head("G7  THE M7 BRIDGE — IS THERE A FINITE-DIFFERENCE PLATEAU?")
    for key, block in f["genes"].items():
        print(f"    {key}:")
        for name, b in block.items():
            ladder = "  ".join(f"{x:+.6f}" for x in b["derivatives"])
            print(f"        {name:>6s}  {ladder}")
            if b["insensitive"]:
                print(f"        {'':>6s}  INSENSITIVE — identically zero")
            else:
                print(f"        {'':>6s}  plateau {b['plateau_decades']} decades within "
                      f"{GATE_FD_PLATEAU_REL:.0e}")
    for other, sh in f["smoothing_shift"].items():
        if not sh:
            continue
        worst = max(abs(x) for x in sh.values())
        print(f"    switching on {other} would move the derivative by up to "
              f"{100*worst:.2f}%")
    print(f"    smoothing needed: {'YES' if f['smoothing_is_needed'] else 'NO'}")
    if f["insensitive_genes"]:
        print()
        print(f"    *** {len(f['insensitive_genes'])} GENE(S) HAVE NO FEA GRADIENT AT "
              f"ALL: {', '.join(f['insensitive_genes'])}")
        print(f"        The mesh does not model fillets, so the FEA cannot see the")
        print(f"        fillet radii — while the beam model prices them through")
        print(f"        `stress_concentration_kt`.  A gradient-based Stage 3 would find")
        print(f"        them perfectly flat and never move them.  This is a")
        print(f"        SPECIFICATION for M7, not a defect in the contact model.")
    print(f"    -> {'PASS' if f['pass'] else 'FAIL'}")

    head("VERDICT")
    print(f"    the real patch is {e['measured_half_deg']:.2f} deg, not "
          f"{e['assumed_half_deg']:.1f} — the assumption was "
          f"{e['assumed_over_measured']:.1f}x too wide")
    print(f"    and it cost {100*abs(e['drop_rel_diff_at_finest']):.2f}% on the axle "
          f"drop, so M4's and M5's conclusions stand unchanged")
    print(f"    what real contact adds is the patch MIGRATING with phase "
          f"({p['patch_centre_swing_deg']:.2f} deg of swing)")
    print(f"\n  OVERALL: {'PASS' if rep['pass'] else 'FAIL'}")
    print(f"\n  NOT DONE: friction, and a deformable ground.  Both are frictionless-")
    print(f"            rigid idealisations here.  Friction matters for the rolling")
    print(f"            resistance this project does not model; it does not change a")
    print(f"            vertical-drop measurement at zero tractive effort.")


def main():
    ap = argparse.ArgumentParser(description="M6 real-contact gate")
    ap.add_argument("--genome", default="best_solution.json")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default="study_contact.json")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="reduced meshes and sample counts; for the test suite")
    args = ap.parse_args()

    genes = load_genes(args.genome)
    cfg = "smoke" if args.quick else args.config
    t0 = time.time()

    rep = {}
    # `--quick` drops the softest decade rather than the stiffest.  1e2 is BELOW the
    # plateau by construction (penetration 0.9% of the band, nine times the gate), so a
    # three-point sweep starting there can never show two agreeing decades — that would
    # make quick mode a stricter gate than the real one rather than a cheaper one.
    rep["penalty"] = run_penalty_plateau(
        genes, cfg, eps_values=(1e3, 1e4, 1e5) if args.quick
        else (1e2, 1e3, 1e4, 1e5))
    rep["verification"] = run_verification(genes, cfg)
    rep["patch"] = run_emergent_patch(
        genes, configs=("smoke", "coarse") if args.quick
        else ("smoke", "coarse", "medium"))
    rep["phase"] = run_phase(genes, cfg, n=5 if args.quick else 13)
    rep["design_space"] = run_design_space(genes, cfg, n=3 if args.quick else 8)
    rep["kinematics"] = run_kinematics(genes, cfg)
    rep["gradient"] = run_gradient_plateau(
        genes, cfg,
        # Both fillet radii on purpose in the full run: the claim is that the FEA sees
        # NEITHER, and a report showing only one of them would not establish it.
        gene_ids=(1, 8, 13) if args.quick else (1, 8, 12, 13),
        steps=(1e-2, 1e-3, 1e-4) if args.quick
        else (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
        smoothings=(0.0,) if args.quick else (0.0, 1e-3))

    rep["pass"] = bool(rep["penalty"]["pass"] and rep["verification"]["pass"]
                       and rep["gradient"]["pass"])
    rep["settings"] = {"config": cfg, "genome": args.genome, "quick": args.quick,
                       "eps_n_default": float(fem.DEFAULT_CONTACT_EPS_N),
                       "rim_band_mm": float(RIM_BAND_MM),
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

    g = [r for r in rep["penalty"]["rows"] if "failed" not in r]
    eps = np.array([r["eps_n"] for r in g])
    ax[0].semilogx(eps, [r["axle_drop_mm"] for r in g], "o-")
    ax[0].set_xlabel("penalty stiffness eps_N (N/mm$^3$)")
    ax[0].set_ylabel("axle drop (mm)")
    ax[0].set_title("the plateau, not the choice")
    ax[0].grid(alpha=0.3, which="both")
    ax2 = ax[0].twinx()
    ax2.loglog(eps, [r["penetration_frac_of_band"] for r in g], "s--", color="tab:red")
    ax2.set_ylabel("penetration / band", color="tab:red")

    p = rep["phase"]["rows"]
    ph = [r["phase_deg"] for r in p]
    ax[1].plot(ph, [r["contact_drop_mm"] for r in p], "o-", label="real contact")
    ax[1].plot(ph, [r["assumed_drop_mm"] for r in p], "s--", label="assumed 3$\\degree$")
    ax[1].set_xlabel("phase (deg)")
    ax[1].set_ylabel("axle drop (mm)")
    ax[1].set_title("the ripple barely moves")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    ax[2].plot(ph, [r["patch_centre_deg"] for r in p], "o-", color="tab:green")
    ax[2].axhline(0.0, color="k", lw=0.8, ls=":")
    ax[2].set_xlabel("phase (deg)")
    ax[2].set_ylabel("patch centre (deg from bottom)")
    ax[2].set_title("the patch migrates as it rolls")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, path)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
