#!/usr/bin/env python3
"""CROWN_PLAN.md Step 4: fit the 2D band thickness that stands in for a crowned rim.

The 2D kernel is plane stress and cannot hold a crown (`wheel_geometry`'s crown block), so
CROWN_PLAN decision 3 gives the 2D model a thicker band instead.  This finds, per 3D rim
(b, h), the band thickness t_eq whose 2D drop ratio reproduces the 3D one:

    D2(t_eq, phase) / D2(1.5, phase)  =  D3(b, h, phase) / D3(1.5, 0, phase)

D2 is the objective's own per-phase quantity -- `build_wheel(..., fillet=True)` at `coarse`,
`solve_wheel_contact` at the service force under SVK -- with only `rim_outer` moved
(= RIM_RADIUS_MM + t, the band thickened outward, as `study_wheel_fea.run_rim_sweep` does).
D3 is CROWN_PLAN R3's table: fe3d SVK on each exported part.  Both sides are RATIOS to the
flat 1.5 mm band, so the plane-stress -> 3D offset and the exported junctions' stiffening
cancel out of the fit; CROWN_PLAN decision 0.4 carries them as a separate target factor.

Step 4's check, as written before the run: fit t_eq(b) on the h 1 column, HOLD ONE OUT, and
the held-out point's 2D ratio at its PREDICTED t_eq -- a real solve, not an interpolation --
must land within 1% of its 3D ratio, at both phases.  Every one of the three is held out in
turn.  Two fit forms, each named: `offset` t_eq = b + c (one parameter, least squares on two
points) and `affine` t_eq = a b + c (two parameters, exact on two points).

Writes studies/study_crown_standin.json.  ~25 2D solves at ~30 s.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import wheel_wheel as WW, wheel_fem as fem, wheel_objective as WO, wheel_genome as WG
import project_paths as PP

CFG, PHASES, T_REF = "coarse", (0.0, 3.75), 1.5
SWEEP = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4)

# CROWN_PLAN R3 (and R1's measured ends, sec205 / sec206): 3D SVK axle drop, mm, of the
# exported part on b729e86, h 2.0 / hc 0.25, per phase.
D3 = {(1.5, 0.0): (1.7230894388112434, 1.9362363645978073),
      (0.5, 1.0): (2.0617881541817296, 2.3259441279863386),
      (1.0, 1.0): (1.5515247226875573, 1.6998806417158094),
      (1.5, 1.0): (1.2797470, 1.3548182)}
H = 1.0
COLUMN = [0.5, 1.0, 1.5]


def drop2d(genes, t, phase):
    mesh = WW.build_wheel(genes, CFG, phase_deg=phase, fillet=True,
                          rim_outer=WW.RIM_RADIUS_MM + t)
    return float(fem.solve_wheel_contact(mesh, force=WO.SERVICE_FORCE_N,
                                         kinematics="svk")["axle_drop_mm"])


def t_for_ratio(ts, ratios, target):
    """Band thickness whose 2D ratio is `target`, interpolated in log-log (the ratio is
    monotone falling in t).  Refused outside the sweep rather than extrapolated."""
    lt, lr = np.log(ts), np.log(ratios)
    if not lr.min() <= np.log(target) <= lr.max():
        raise ValueError(f"ratio {target} outside the sweep's {ratios.min()}..{ratios.max()}")
    return float(np.exp(np.interp(np.log(target), lr[::-1], lt[::-1])))


def main():
    rec = json.load(open(PP.BEST_SOLUTION))
    genes = WG.genes_to_vector(rec["genes"])
    t0 = time.time()
    sweep = {p: [] for p in PHASES}
    for t in SWEEP:
        for p in PHASES:
            sweep[p].append(drop2d(genes, t, p))
            print(f"  sweep t {t:.2f}  phase {p:5.2f}  D2 {sweep[p][-1]:.9f}", flush=True)
    ref = {p: sweep[p][SWEEP.index(T_REF)] if T_REF in SWEEP else drop2d(genes, T_REF, p)
           for p in PHASES}
    ts = np.array(SWEEP)
    r3 = {b: [D3[(b, H)][i] / D3[(T_REF, 0.0)][i] for i in range(len(PHASES))] for b in COLUMN}
    # t_eq per point and phase, from the sweep; then CHECKED by a direct solve at it
    teq = {b: [t_for_ratio(ts, np.array(sweep[p]) / ref[p], r3[b][i])
               for i, p in enumerate(PHASES)] for b in COLUMN}
    interp_check = {b: [drop2d(genes, teq[b][i], p) / ref[p] / r3[b][i] - 1.0
                        for i, p in enumerate(PHASES)] for b in COLUMN}
    # one t_eq per rim: the mean over the two phases, since the objective takes one band
    tbar = {b: float(np.mean(teq[b])) for b in COLUMN}

    folds = []
    for out in COLUMN:
        fit = [b for b in COLUMN if b != out]
        x = np.array(fit); y = np.array([tbar[b] for b in fit])
        c_off = float(np.mean(y - x))
        a_aff, c_aff = np.polyfit(x, y, 1)
        for form, pred in (("offset", out + c_off), ("affine", a_aff * out + c_aff)):
            err = [drop2d(genes, pred, p) / ref[p] / r3[out][i] - 1.0
                   for i, p in enumerate(PHASES)]
            folds.append({"held_out_b": out, "form": form, "t_pred": float(pred),
                          "t_eq_measured": tbar[out], "ratio_err": err,
                          "pass_1pct": bool(max(abs(e) for e in err) <= 0.01)})
            print(f"  hold out b {out}  {form:6s}  t_pred {pred:.4f} (t_eq {tbar[out]:.4f})  "
                  f"ratio err {err[0]:+.4%} / {err[1]:+.4%}", flush=True)

    out = {"genome_hash": WG.genome_hash(rec["genes"]), "cfg": CFG, "kinematics": "svk",
           "phases": list(PHASES), "h": H, "t_ref_mm": T_REF,
           "sweep": {"t_mm": list(SWEEP), "drop_2d_mm": {str(p): sweep[p] for p in PHASES}},
           "drop_2d_ref_mm": {str(p): ref[p] for p in PHASES},
           "ratio_3d": {str(b): r3[b] for b in COLUMN},
           "t_eq_mm": {str(b): teq[b] for b in COLUMN},
           "t_eq_mean_mm": {str(b): tbar[b] for b in COLUMN},
           "interp_check_err": {str(b): interp_check[b] for b in COLUMN},
           "folds": folds, "wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(HERE, "study_crown_standin.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: out[k] for k in ("t_eq_mm", "t_eq_mean_mm", "interp_check_err")},
                     indent=1))


if __name__ == "__main__":
    main()
