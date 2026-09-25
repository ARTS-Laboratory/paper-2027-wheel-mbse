#!/usr/bin/env python3
"""CROWN_PLAN.md Step 5: the band's inner-face hoop tension in 2D, calibrated to 3D.

R5 decided the quantity (D4), the calibration (D5) and the pricing (D6) before this ran.
This measures what those decisions left open, at `b729e86`, SVK, the service force, on the
flat 1.5 mm band and on the stand-in (`WO.CROWN_STANDIN`).  Every solve keeps its RAW
inner-face profile (`wheel_adjoint.band_face_profile`: hoop stress, x, y and distance to
the nearest spoke junction per (element, node) pair), and everything below is a reduction
of those profiles, so a different exclusion or exponent is a re-read, not a re-run:

  1. THE JUNCTION EXCLUSION (R5 D4, and its fallback).  The first pass of this driver read
     the face with no exclusion and refuted D4: at the node next to a junction the tension
     climbs with the mesh, 27.05 / 30.43 / 36.35 MPa at coarse / medium / fine (phase 3.75,
     stand-in).  So a scan over d: the smallest d whose max moves at most 3% medium -> fine
     at the two junction-peak phases (0, 3.75) and the mid-span control (11.25), and which
     still reads R3's 3D peak positions (x +1.0 at 3.75, x -2.15 at 0).
  2. D5: c_band(phase) = the 3D inner-face hoop tension of the crown on top (R1, all eight
     phases) over the stand-in's 2D max beyond d; its max/min spread against 1.3; and the
     control c_flat = R3's flat-part 3D (phases 0, 3.75) over the flat 2D max.
  3. D6: the node p-norm of `_qoi_band_tension` (the same sum over the same pairs) against
     the max beyond d, at every stand-in phase; the smallest p of 8 / 16 / 32 / 64 within 2%.

Writes studies/study_band_tension.json.  24 solves, ~25 min.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import wheel_wheel as WW, wheel_adjoint as WA, wheel_objective as WO, wheel_genome as WG
import project_paths as PP

CFG, KIN = "coarse", "svk"
PHASES = [float(p) for p in WO.phase_stencil(scheme="uniform")]
LADDER_CFGS = ("coarse", "medium", "fine")
LADDER_PHASES = (0.0, 3.75, 11.25)          # two junction-peak phases and a mid-span control
EXCLUDE_MM = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
NODE_P = (8.0, 16.0, 32.0, 64.0)
CONVERGED = 0.03                             # R5 D4's falsifier, medium -> fine
PNORM_WITHIN = 0.02                          # R5 D6
SPREAD_MAX = 1.3                             # R5 D5
# R1 (CROWN_PLAN.md): 3D hoop TENSION max on the crown on top's band inner face, MPa,
# post3d_cyl.py over fe3d SVK, b729e86, h 2.0 / hc 0.25, at the eight stencil phases.
HOOP_3D_CROWN = (26.72, 33.80, 29.83, 28.41, 26.49, 24.17, 21.01, 19.97)
# R3: the same instrument on the flat exported part (1.5, 0), phases 0 and 3.75.
HOOP_3D_FLAT = {0.0: 39.07, 3.75: 50.33}
# R3: where the 3D hoop-tension peak sits, x in mm (ground frame), at the two phases read.
PEAK_X_3D = {0.0: -2.15, 3.75: +1.0}


def solve(genes, cfg, phase, rim_outer):
    """One service solve; the inner face's raw profile and the axle drop."""
    mesh = WW.build_wheel(genes, cfg, phase_deg=phase, fillet=True, rim_outer=rim_outer)
    o = WA.service_qoi_value_and_grad(genes, cfg, ("pnorm_stress",), force=WO.SERVICE_FORCE_N,
                                      mesh=mesh, kinematics=KIN)
    f = WA.band_face_profile(o["_meta"]["prob"], o["_meta"]["res"]["u"], mesh)
    return {"cfg": cfg, "phase_deg": phase, "axle_drop_mm": float(o["axle_drop"]["value"]),
            **{k: [float(v) for v in f[k]] for k in f}}


def peak(prof, d):
    """Max hoop stress beyond `d` of every junction, and where."""
    s, dj = np.asarray(prof["s_tt_mpa"]), np.asarray(prof["d_junction_mm"])
    k = int(np.argmax(np.where(dj >= d, s, -np.inf)))
    return {"max_mpa": float(s[k]), "x_mm": prof["x_mm"][k], "d_junction_mm": float(dj[k])}


def pnorm(prof, d, p):
    """`_qoi_band_tension`'s value: the l_p norm of the positive hoop stress beyond `d`."""
    s, dj = np.asarray(prof["s_tt_mpa"]), np.asarray(prof["d_junction_mm"])
    return float(np.sum(np.maximum(s[dj >= d], 0.0) ** p) ** (1.0 / p))


def d_at_x(prof, x):
    """Junction distance of the loaded-side face node nearest `x` (the 3D peak's place)."""
    xs, ys = np.asarray(prof["x_mm"]), np.asarray(prof["y_mm"])
    idx = np.where(ys < 0.0)[0]
    k = idx[np.argmin(np.abs(xs[idx] - x))]
    return {"x_mm": float(xs[k]), "d_junction_mm": prof["d_junction_mm"][k]}


def main():
    rec = json.load(open(PP.BEST_SOLUTION))
    genes = WG.genes_to_vector(rec["genes"])
    t0 = time.time()
    bands = {"flat": WW.RIM_OUTER_RADIUS_MM, "standin": WO.CROWN_STANDIN["rim_outer"]}
    prof = {b: [] for b in bands}
    for b, ro in bands.items():
        for p in PHASES:
            prof[b].append(solve(genes, CFG, p, ro))
            pk = peak(prof[b][-1], 0.0)
            print(f"  {b:8s} phase {p:6.3f}  max {pk['max_mpa']:7.3f}  x {pk['x_mm']:+7.3f}"
                  f"  d_j {pk['d_junction_mm']:.3f}", flush=True)
    ladder = {str(p): {"coarse": prof["standin"][PHASES.index(p)]} for p in LADDER_PHASES}
    for cfg in LADDER_CFGS[1:]:
        for p in LADDER_PHASES:
            ladder[str(p)][cfg] = solve(genes, cfg, p, bands["standin"])
            print(f"  ladder {cfg:7s} phase {p:6.3f}  max(d 0) "
                  f"{peak(ladder[str(p)][cfg], 0.0)['max_mpa']:.4f}", flush=True)

    scan = []
    for d in EXCLUDE_MM:
        rungs = {ph: [peak(ladder[ph][c], d)["max_mpa"] for c in LADDER_CFGS] for ph in ladder}
        moves = {ph: v[2] / v[1] - 1.0 for ph, v in rungs.items()}
        where = {str(p): d_at_x(prof["standin"][PHASES.index(p)], x) for p, x in PEAK_X_3D.items()}
        c_band = [h / peak(r, d)["max_mpa"] for h, r in zip(HOOP_3D_CROWN, prof["standin"])]
        scan.append({"exclude_mm": d, "ladder_max_mpa": rungs, "medium_to_fine": moves,
                     "converged": bool(all(abs(m) <= CONVERGED for m in moves.values())),
                     "peak_3d_positions": where,
                     "reads_3d_peaks": bool(all(w["d_junction_mm"] >= d for w in where.values())),
                     "c_band": c_band, "c_band_spread": max(c_band) / min(c_band)})
        print(f"  d {d:4.2f}  m->f " + " ".join(f"{m:+.3%}" for m in moves.values())
              + f"  c_band spread {scan[-1]['c_band_spread']:.3f}", flush=True)
    chosen = next((s for s in scan if s["converged"] and s["reads_3d_peaks"]), None)
    out = {"genome_hash": WG.genome_hash(rec["genes"]), "cfg": CFG, "kinematics": KIN,
           "phases": PHASES, "rim_outer_mm": bands, "scan": scan,
           "hoop_3d_crown_mpa": list(HOOP_3D_CROWN),
           "hoop_3d_flat_mpa": {str(k): v for k, v in HOOP_3D_FLAT.items()},
           "peak_x_3d_mm": {str(k): v for k, v in PEAK_X_3D.items()},
           "exclude_mm_chosen": None if chosen is None else chosen["exclude_mm"]}
    if chosen is not None:
        d = chosen["exclude_mm"]
        out["peaks"] = {b: [peak(r, d) for r in prof[b]] for b in bands}
        out["c_band"] = chosen["c_band"]
        out["c_band_max"] = max(chosen["c_band"])
        out["c_band_spread"] = chosen["c_band_spread"]
        out["d5_priced"] = bool(chosen["c_band_spread"] <= SPREAD_MAX)
        out["c_flat"] = {str(p): HOOP_3D_FLAT[p] / peak(prof["flat"][PHASES.index(p)], d)["max_mpa"]
                         for p in HOOP_3D_FLAT}
        over = {str(int(p)): max(pnorm(r, d, p) / peak(r, d)["max_mpa"] - 1.0
                                 for r in prof["standin"]) for p in NODE_P}
        out["pnorm_over_max_worst"] = over
        out["node_p_chosen"] = next((float(k) for k, v in over.items() if v <= PNORM_WITHIN),
                                    None)
    out["profiles"] = prof
    out["ladder_profiles"] = {ph: {c: v for c, v in r.items() if c != "coarse"}
                              for ph, r in ladder.items()}
    out["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "study_band_tension.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out.get(k) for k in ("exclude_mm_chosen", "c_band", "c_band_spread",
                                               "d5_priced", "c_flat", "pnorm_over_max_worst",
                                               "node_p_chosen")}, indent=1))


if __name__ == "__main__":
    main()
