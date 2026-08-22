"""
=============================================================================
  DOES THE FEA MESH HAVE THE JUNCTION CORNERS THE SHIPPED PART HAS?
=============================================================================
    .venv-opt/bin/python studies/study_junction_agreement.py   (make junction)

UNCAP_PLAN.md Step 0's instrument, and Step 1's go/no-go in the same run.

WHY THIS EXISTS
---------------
`wheel_wheel.sector_blocks` terminates the spoke ON the ring circle and closes it with a
half END CAP, so its two corners per junction are

    P_t   the straddling flank crossing the circle       (spoke-side leg = the flank)
    P_c   where the end cap meets the circle             (spoke-side leg = t/2)

`wheel_step_export.spoke_profile` does something else.  `_embed` continues BOTH flanks
along one shared straight direction until both are past HUB_EMBED_RADIUS_MM (12.20,
INSIDE the hub disk) or RIM_EMBED_RADIUS_MM (50.25, OUTSIDE the OD), so the shipped solid
has NO end cap and both its corners are flank crossings.

That difference is deliberate and `wheel_wheel.py`'s module docstring documents it: the
mesh models ~1.4% less material, all of it at the junctions, because reproducing
`_embed`'s argmax search would put a non-differentiable step in the coordinate map (M7).
THIS DRIVER DOES NOT ARGUE WITH THAT.  What it measures is the consequence nobody had
written down — PLAN §34 Finding 4, that the wheel's global peak von Mises sits 11-16 um
from `rim:P_c`, i.e. ON a corner the part does not have.

WHAT IT REPORTS
---------------
  A.  the MESH's four corners, and the PART's four, side by side: position, wedge angle,
      Williams lambda, and the spoke-side leg each fillet would have to lie on.
  B.  three EXTENSION candidates -- UNCAP_PLAN Step 1's proposals for replacing the end
      cap with a real corner -- scored against the part's.  `wheel_wheel.py`'s docstring
      says no smooth alternative exists because "the bottom flank's backward tangent
      MISSES the hub circle entirely"; measured, it does NOT miss, so the alternatives are
      real and have to be judged on their numbers rather than excluded.  Measured, the
      one that matches is NOT the flank's own tangent: it is `_embed`'s own blend at each
      ring -- radial at the hub, shared tangent at the rim -- and taken that way the
      agreement is 0.00 deg of wedge at the hub and 0.02 at the rim.

THE VALIDATION THAT MAKES THE RECONSTRUCTION TRUSTWORTHY
--------------------------------------------------------
`_embed` is reproduced here in numpy so this runs in the OPT env with no OCC.  The check
that it is the same geometry is the CROSSING COUNT: the reconstruction must find exactly
2 ring crossings per spoke per ring, i.e. 24 and 24, matching `hub_edges`/`rim_edges` in
`export/wheel_step_manifest.json`.  It is checked, not assumed, and a mismatch is fatal.

WEDGE CONVENTION, stated because two conventions differ by 180 degrees
---------------------------------------------------------------------
`wedge = 360 - void`, and `void` is the angle between the flank leaving the corner into
the free spoke and the ring-circle tangent pointing into the FREE arc.  Computed this
way the mesh's four corners reproduce §30's independently measured wedges (321.10 /
296.75 / 321.33 / 307.94, summed from incident element angles on the fine mesh) to within
0.8 deg -- which is the cross-check that both are measuring the same thing.  Taking the
ACUTE branch instead, which is the tempting shortcut, is wrong by 100 deg at rim:P_c.  This driver
does NOT solve a field and does NOT need one; it is geometry only, and it runs in
seconds rather than the 95 minutes §30 warned about.
=============================================================================
"""

import argparse
import json
import math
import os

import numpy as np

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import wheel_genome as wg
import wheel_wheel as WW
from wheel_fea import generate_bezier_centerline, thicken_3taper_curve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# `wheel_step_export`'s constants, restated rather than imported: that module pulls in
# cadquery at import time and this driver must run in the OPT env.  If either constant
# moves there, this one is wrong -- which `test_junction_agreement.py` is what checks.
HUB_EMBED_MM = WW.HUB_RADIUS_MM - 0.5
RIM_EMBED_MM = WW.RIM_OUTER_RADIUS_MM + 0.25
N_OUTLINE_PTS = 48
N_DENSE = 4001


def _unit(v):
    v = np.asarray(v, float)
    return v / (float(np.hypot(v[0], v[1])) or 1.0)


def williams_lambda(omega_deg):
    """Smallest root in (0, 1) of sin(lambda*omega) + lambda*sin(omega) = 0.

    Same function as `study_corner_singularity.williams_lambda`, duplicated rather than
    imported so this driver has no dependency on that one's import side effects.  Both
    are verified the same way: 360 deg -> 0.5 exactly, 270 deg -> 0.5445.
    """
    w = math.radians(omega_deg)

    def f(L):
        return math.sin(L * w) + L * math.sin(w)

    lo, hi = 1e-9, 1.0 - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def void_and_wedge(Q, f_hat, away_from):
    """Void and wedge at a corner `Q` on a ring circle.

    `f_hat` leaves `Q` along the spoke's boundary, into the free spoke.  `away_from` is a
    point on the circle on the WELDED side, so the free-arc tangent is the one pointing
    away from it.
    """
    Q = np.asarray(Q, float)
    rad = _unit(Q)
    tang = np.array([-rad[1], rad[0]])
    # pick the tangent branch that increases the arc distance from `away_from`
    if float(np.dot(tang, _unit(np.asarray(away_from, float) - Q))) > 0.0:
        tang = -tang
    void = math.degrees(math.acos(float(np.clip(np.dot(_unit(f_hat), tang), -1.0, 1.0))))
    return void, 360.0 - void


def fillet_fit(void_deg, leg_mm, R_mm):
    """Can a fillet of radius `R` sit in this corner, on the leg this corner has?

    A fillet tangent to both faces of a corner needs `T = R / tan(void/2)` of each, so a
    corner with a SHORT leg refuses a large radius no matter how open it is.  That is the
    whole of FILLET_PLAN.md PART 2's `P_c` NO-GO, and it is reported here rather than
    re-derived in prose because every input to it moves: `void` moves with `uncap`, `leg`
    moves with `uncap`, and `R` is a gene.

    `r_max_mm = leg * tan(void/2)` inverts it — the largest radius this corner would
    accept as built — which is the number to compare a gene box against.

    Returns None where there is no leg to measure (the PART's own corners are read off an
    outline, not off a block, and their legs are the flanks).
    """
    if void_deg <= 0.0 or void_deg >= 180.0:
        return None
    half = math.radians(void_deg / 2.0)
    T = R_mm / math.tan(half)
    out = {"radius_mm": float(R_mm), "tangent_length_mm": float(T)}
    if leg_mm is not None and leg_mm > 0.0:
        out["spoke_side_leg_mm"] = float(leg_mm)
        out["t_over_leg"] = float(T / leg_mm)
        out["r_max_on_this_leg_mm"] = float(leg_mm * math.tan(half))
        out["fits"] = bool(T <= leg_mm)
    return out


# ---------------------------------------------------------------------------
# THE PART'S OUTLINE
# ---------------------------------------------------------------------------

def spoke_edges_global(g, n=N_OUTLINE_PTS):
    """`wheel_step_export.spoke_edges_global`, reproduced."""
    curve, _ = generate_bezier_centerline(g["cx1"], g["cy1"], g["cx2"], g["cy2"],
                                          g["cx3"], g["cy3"], g["cx4"], g["cy4"])
    top, bot = thicken_3taper_curve(curve, g["t0"], g["t1"], g["t2"], g["t3"],
                                    return_edges=True)
    top = top.copy(); top[:, 0] += WW.HUB_RADIUS_MM
    bot = bot.copy(); bot[:, 0] += WW.HUB_RADIUS_MM
    idx = np.unique(np.linspace(0, len(top) - 1, n).astype(int))
    return top[idx], bot[idx]


def embed(p_top, p_bot, d_tan, target_r, outward, margin=0.05):
    """`wheel_step_export._embed`, reproduced: one direction and one length for BOTH
    flanks.  Blend 1.0 (pure radial) inward at the hub; a blend search outward at the
    rim.  Returns (direction, run length, blend)."""
    p_top, p_bot = np.asarray(p_top, float), np.asarray(p_bot, float)
    d_tan = _unit(d_tan)
    reach = target_r + margin if outward else target_r - margin
    span = 4.0 * (target_r + float(np.linalg.norm(p_bot)))
    L = np.linspace(0.0, span, 20001)[:, None]
    radial = _unit((p_top + p_bot) / 2.0) * (1.0 if outward else -1.0)
    for blend in (np.linspace(0.0, 1.0, 21) if outward else (1.0,)):
        d = _unit((1.0 - blend) * d_tan + blend * radial)
        r_top = np.linalg.norm(p_top + L * d, axis=1)
        r_bot = np.linalg.norm(p_bot + L * d, axis=1)
        good = ((r_top >= reach) & (r_bot >= reach) if outward
                else (r_top <= reach) & (r_bot <= reach))
        if good.any():
            return d, float(L[int(np.argmax(good)), 0]), float(blend)
    raise RuntimeError(f"no embedding direction reaches r={target_r:.3f}")


def ring_crossings(poly, radius):
    """(point, unit tangent) at every crossing of |p| = radius along a polyline."""
    r = np.linalg.norm(poly, axis=1)
    out = []
    for k in range(len(poly) - 1):
        if (r[k] - radius) * (r[k + 1] - radius) > 0 or r[k] == r[k + 1]:
            continue
        f = (radius - r[k]) / (r[k + 1] - r[k])
        out.append((poly[k] + f * (poly[k + 1] - poly[k]),
                    _unit(poly[k + 1] - poly[k])))
    return out


def part_corners(genes):
    """The shipped part's ring crossings, per ring, with a tangent at each."""
    top, bot = spoke_edges_global(genes, n=N_DENSE)
    t48, b48 = spoke_edges_global(genes)
    d_hub, run_hub, _ = embed(t48[0], b48[0],
                              (t48[0] - t48[1]) + (b48[0] - b48[1]),
                              HUB_EMBED_MM, False)
    d_rim, run_rim, _ = embed(t48[-1], b48[-1],
                              (t48[-1] - t48[-2]) + (b48[-1] - b48[-2]),
                              RIM_EMBED_MM, True)
    out = {}
    for nm, arr in (("top_flank", top), ("bot_flank", bot)):
        a = np.asarray(arr, float)
        path = np.vstack([a[0] + run_hub * d_hub, a, a[-1] + run_rim * d_rim])
        for ring, R in (("hub", WW.HUB_RADIUS_MM), ("rim", WW.rim_inner_radius())):
            for q, t in ring_crossings(path, R):
                out.setdefault(ring, []).append((nm, q, t))
    return out


def _line_circle(p, d, R):
    """First positive `t` with |p + t d| = R, in closed form."""
    p, d = np.asarray(p, float), _unit(d)
    a, b, c = float(d @ d), 2.0 * float(p @ d), float(p @ p) - R * R
    disc = b * b - 4 * a * c
    closest = float(np.linalg.norm(p - float(p @ d) * d))
    if disc < 0:
        return None, closest
    s = math.sqrt(disc)
    ts = [t for t in ((-b - s) / (2 * a), (-b + s) / (2 * a)) if t > 0]
    return (min(ts) if ts else None), closest


def extension_candidates(genes):
    """UNCAP_PLAN Step 1's candidates for replacing the END CAP with a real corner.

    The M7 objection that rules out reproducing `_embed` is about its ARGMAX: it picks a
    length by scanning 20001 candidates and, at the rim, 21 blend directions as well.
    **Neither piece is needed here.**  The LENGTH is unnecessary because the ring
    crossing is a closed-form line/circle intersection; and each of `_embed`'s two
    extreme DIRECTIONS is itself a smooth function of the genes:

      `own_tangent`     the bottom flank's own end tangent
      `shared_tangent`  the mean of the two end tangents -- `_embed` at blend 0.0
      `radial`          the flank midpoint's radial -- `_embed` at blend 1.0

    Which one is faithful is not a matter of taste: **`_embed` itself lands on blend 1.0
    at the hub (radial-inward always reaches, so the hub search is a single step) and on
    blend 0.0 at the rim on the shipped genome.**  So the prediction going in is that
    `radial` matches at the hub and the tangents match at the rim, and this driver is
    what checks it.
    """
    top, bot = spoke_edges_global(genes, n=N_DENSE)
    t48, b48 = spoke_edges_global(genes)
    out = {}
    for ring, R, p, own, shared, sgn in (
            ("hub", WW.HUB_RADIUS_MM, bot[0], bot[0] - bot[1],
             (t48[0] - t48[1]) + (b48[0] - b48[1]), -1.0),
            ("rim", WW.rim_inner_radius(), bot[-1], bot[-1] - bot[-2],
             (t48[-1] - t48[-2]) + (b48[-1] - b48[-2]), +1.0)):
        p = np.asarray(p, float)
        radial = _unit((np.asarray(t48[0 if ring == "hub" else -1], float)
                        + np.asarray(b48[0 if ring == "hub" else -1], float)) / 2.0) * sgn
        out[ring] = {}
        for nm, d in (("own_tangent", _unit(own)), ("shared_tangent", _unit(shared)),
                      ("radial", radial)):
            t, closest = _line_circle(p, d, R)
            if t is None:
                out[ring][nm] = {"reaches": False, "closest_approach_mm": closest}
                continue
            q = p + t * d
            out[ring][nm] = {"reaches": True, "closest_approach_mm": closest,
                             "run_mm": float(t), "point": q, "tangent": -d,
                             "theta_deg": math.degrees(math.atan2(q[1], q[0]))}
    return out


# ---------------------------------------------------------------------------
# THE MESH'S CORNERS
# ---------------------------------------------------------------------------

def mesh_corners(gvec, cfg, uncap=False):
    cfg = WW.get_config(cfg)
    orient = WW.flank_orientation(gvec, cfg)
    b = WW.sector_blocks(gvec, cfg, orientation=orient, uncap=uncap)
    sp = np.asarray(b["spoke"], float)
    out = {}
    for k, ring in enumerate(("hub", "rim")):
        j = np.asarray(b[f"{ring}_junction"], float)
        P_t, P_c = j[0, 0], j[-1, 0]
        # P_t's FREE boundary is the STRADDLING FLANK, which belongs to the spoke block.
        # The junction's own two edges at P_t are the ring arc and the end cross-section,
        # and the cross-section is INTERNAL -- taking it here reads the wrong angle and
        # gave 232 deg where §30 measures 321.
        col = -1 if float(orient[k]) > 0 else 0
        row, step = (0, 1) if ring == "hub" else (-1, -2)
        f_hat = _unit(sp[step, col] - sp[row, col])
        out[ring] = {
            # welded side is toward P_c
            "P_t": {"point": P_t, "f_hat": f_hat, "away": P_c,
                    "leg_mm": float(np.linalg.norm(
                        np.diff(sp[:, col], axis=0), axis=1).sum())},
            # P_c: the end cap leaves toward `far_end`; the welded side is toward P_t
            "P_c": {"point": P_c, "f_hat": _unit(j[-1, 1] - P_c), "away": P_t,
                    "leg_mm": float(np.linalg.norm(j[-1, -1] - P_c))},
        }
    return out


def build(genes, cfg="coarse"):
    gvec = wg.genes_to_vector(genes)
    # THREE meshes, because since 2026-08-18 the DEFAULT is neither of the two this
    # study was written to compare.  `uncap=False` is the pre-flip geometry and is kept
    # as the control; `WW.UNCAP_DEFAULT` is what `build_wheel` now actually builds;
    # `uncap=True` is `_embed`'s own blend at both rings -- the most FAITHFUL rim and
    # the one that is not buildable (min_sj 0.0072).  Reporting only two of the three
    # would put the label "today" on a mesh nobody builds any more.
    mesh = mesh_corners(gvec, cfg, uncap=False)
    mesh_d = mesh_corners(gvec, cfg, uncap=WW.UNCAP_DEFAULT)
    mesh_u = mesh_corners(gvec, cfg, uncap=True)
    part = part_corners(genes)
    ext = extension_candidates(genes)

    n_hub, n_rim = len(part.get("hub", [])), len(part.get("rim", []))
    rec = {"genome_hash": wg.genome_hash(genes), "config": cfg,
           "crossings_per_spoke": {"hub": n_hub, "rim": n_rim},
           "crossings_per_wheel": {"hub": n_hub * WW.NUMBER_OF_SPOKES,
                                   "rim": n_rim * WW.NUMBER_OF_SPOKES},
           "rings": {}}

    R_by_ring = {"hub": float(gvec[12]), "rim": float(gvec[13])}
    for ring, ring_r in (("hub", WW.HUB_RADIUS_MM), ("rim", WW.rim_inner_radius())):
        rows = []
        R_mm = R_by_ring[ring]
        m = mesh[ring]
        for nm in ("P_t", "P_c"):
            e = m[nm]
            void, wedge = void_and_wedge(e["point"], e["f_hat"], e["away"])
            rows.append({"source": "mesh (uncap=False)", "name": nm,
                         "theta_deg": math.degrees(math.atan2(e["point"][1],
                                                              e["point"][0])),
                         "void_deg": void, "wedge_deg": wedge,
                         "lambda_W": williams_lambda(wedge),
                         "spoke_side_leg_mm": e["leg_mm"],
                         "fillet_fit": fillet_fit(void, e["leg_mm"], R_mm)})
        # the same block's second corner under the other two settings.  `P_t` does not
        # move with `uncap` -- it is the spoke's own end row -- so only `P_c` is re-read.
        for src, mm in (("mesh (SHIPPED DEFAULT)", mesh_d), ("mesh (uncap=True)", mesh_u)):
            eu = mm[ring]["P_c"]
            void, wedge = void_and_wedge(eu["point"], eu["f_hat"], eu["away"])
            rows.append({"source": src, "name": "P_c",
                         "theta_deg": math.degrees(math.atan2(eu["point"][1],
                                                              eu["point"][0])),
                         "void_deg": void, "wedge_deg": wedge,
                         "lambda_W": williams_lambda(wedge),
                         "spoke_side_leg_mm": eu["leg_mm"],
                         "fillet_fit": fillet_fit(void, eu["leg_mm"], R_mm)})
        pc = part[ring]
        for nm, q, t in pc:
            other = [x[1] for x in pc if x[0] != nm]
            away = other[0] if other else -q
            # the outline tangent runs hub->rim; into the free spoke is away from the
            # ring at the hub end and toward it at the rim end, so orient on the radial
            f_hat = t if (float(np.dot(t, _unit(q))) > 0) == (ring == "hub") else -t
            void, wedge = void_and_wedge(q, f_hat, away)
            rows.append({"source": "part", "name": nm,
                         "theta_deg": math.degrees(math.atan2(q[1], q[0])),
                         "void_deg": void, "wedge_deg": wedge,
                         "lambda_W": williams_lambda(wedge),
                         "spoke_side_leg_mm": None,
                         "fillet_fit": fillet_fit(void, None, R_mm)})
        for nm, e in ext[ring].items():
            if not e["reaches"]:
                rows.append({"source": "extension", "name": nm, "reaches": False,
                             "closest_approach_mm": e["closest_approach_mm"]})
                continue
            void, wedge = void_and_wedge(e["point"], e["tangent"],
                                         m["P_t"]["point"])
            rows.append({"source": "extension", "name": nm,
                         "theta_deg": e["theta_deg"], "void_deg": void,
                         "wedge_deg": wedge, "lambda_W": williams_lambda(wedge),
                         "spoke_side_leg_mm": None, "run_mm": e["run_mm"],
                         "closest_approach_mm": e["closest_approach_mm"],
                         "fillet_fit": fillet_fit(void, None, R_mm)})
        rec["rings"][ring] = {"ring_r_mm": ring_r, "corners": rows}

    # THE PRICING, GATHERED.  FILLET_PLAN.md PART 2 ruled `P_c` un-filletable on a leg of
    # t/2, measured on the CAPPED mesh.  Every term in that judgement has since moved, so
    # the verdict is recomputed per ring and per uncap setting rather than quoted.
    rec["fillet_fit_summary"] = {}
    for ring in ("hub", "rim"):
        by_src = {}
        for row in rec["rings"][ring]["corners"]:
            ff = row.get("fillet_fit")
            if ff is None or "t_over_leg" not in ff:
                continue
            by_src[f"{row['source']} {row['name']}"] = {
                "void_deg": row["void_deg"],
                "spoke_side_leg_mm": ff["spoke_side_leg_mm"],
                "tangent_length_mm": ff["tangent_length_mm"],
                "t_over_leg": ff["t_over_leg"],
                "r_max_on_this_leg_mm": ff["r_max_on_this_leg_mm"],
                "fits_at_shipped_radius": ff["fits"],
            }
        rec["fillet_fit_summary"][ring] = {"radius_mm": R_by_ring[ring], "by_corner": by_src}

    rec["validation"] = {
        "expected_crossings_per_wheel": 24,
        "hub_ok": n_hub * WW.NUMBER_OF_SPOKES == 24,
        "rim_ok": n_rim * WW.NUMBER_OF_SPOKES == 24,
    }
    rec["validation"]["pass"] = bool(rec["validation"]["hub_ok"]
                                     and rec["validation"]["rim_ok"])
    return rec


def verdict(rec):
    """UNCAP_PLAN Step 1's go/no-go, computed rather than eyeballed.

    The proposal replaces the mesh's `P_c` with the tangent extension.  It is an
    IMPROVEMENT only if the replacement lands closer to the part's second corner than
    the end cap does -- in WEDGE ANGLE, which is what drives the singularity, and the
    driver reports position too because position is what the first look at this got
    wrong.
    """
    out = {}
    for ring, r in rec["rings"].items():
        by = {}
        for row in r["corners"]:
            by.setdefault(row["source"], []).append(row)
        cap = next(x for x in by["mesh (uncap=False)"] if x["name"] == "P_c")
        p_t = next(x for x in by["mesh (uncap=False)"] if x["name"] == "P_t")
        # the part's second corner is the one FURTHEST from the P_t match
        truth = sorted(by["part"],
                       key=lambda x: abs(x["theta_deg"] - p_t["theta_deg"]))[-1]

        def err(row):
            return {"wedge_err_deg": abs(row["wedge_deg"] - truth["wedge_deg"]),
                    "theta_err_deg": abs(row["theta_deg"] - truth["theta_deg"]),
                    "lambda_err": abs(row["lambda_W"] - truth["lambda_W"])}

        cands = {"end_cap (pre-2026-08-18)": err(cap)}
        for src, lab in (("mesh (SHIPPED DEFAULT)", "AS BUILT (the default)"),
                         ("mesh (uncap=True)", "uncap=True (rim not buildable)")):
            row = next((x for x in by.get(src, []) if x["name"] == "P_c"), None)
            if row is not None:
                cands[lab] = err(row)
        for row in by["extension"]:
            cands[row["name"]] = (err(row) if row.get("reaches", True)
                                  else {"reaches": False})
        best = min((k for k, v in cands.items() if "wedge_err_deg" in v),
                   key=lambda k: cands[k]["wedge_err_deg"])
        out[ring] = {
            "truth_theta_deg": truth["theta_deg"],
            "truth_wedge_deg": truth["wedge_deg"],
            "truth_lambda": truth["lambda_W"],
            "candidates": cands,
            "best_on_wedge": best,
            "beats_the_end_cap": bool(best != "end_cap (pre-2026-08-18)"),
            "improvement_deg": (cands["end_cap (pre-2026-08-18)"]["wedge_err_deg"]
                                - cands[best]["wedge_err_deg"]),
        }
    out["go"] = bool(all(out[k]["beats_the_end_cap"] for k in ("hub", "rim")))
    out["go_rim_only"] = bool(out["rim"]["beats_the_end_cap"])
    return out


def _print(rec, ver):
    print(f"\n  genome {rec['genome_hash']}   config {rec['config']}")
    v = rec["validation"]
    print(f"  reconstruction check: {rec['crossings_per_wheel']['hub']} hub and "
          f"{rec['crossings_per_wheel']['rim']} rim crossings per wheel against the "
          f"manifest's 24 and 24 -> {'OK' if v['pass'] else 'MISMATCH'}")
    for ring, r in rec["rings"].items():
        print(f"\n  --- {ring}   ring r = {r['ring_r_mm']:.4f} mm")
        print(f"      {'source':<24}{'name':<16}{'theta':>11}{'wedge':>10}"
              f"{'lambda':>9}{'spoke leg':>12}")
        for row in r["corners"]:
            if row.get("reaches") is False:
                print(f"      {row['source']:<24}{row['name']:<16}"
                      f"   DOES NOT REACH (closest "
                      f"{row['closest_approach_mm']:.4f} mm)")
                continue
            leg = ("" if row["spoke_side_leg_mm"] is None
                   else f"{row['spoke_side_leg_mm']:12.4f}")
            print(f"      {row['source']:<24}{row['name']:<16}"
                  f"{row['theta_deg']:>11.5f}{row['wedge_deg']:>10.2f}"
                  f"{row['lambda_W']:>9.4f}{leg:>12}")

    print("\n  CAN A FILLET SIT IN THESE CORNERS?  T = R / tan(void/2) against the leg it "
          "must lie on")
    print("  (GEOMETRIC admissibility only — whether the MESH can then be built is a "
          "separate gate, `make fillet`)")
    for ring, summ in rec.get("fillet_fit_summary", {}).items():
        print(f"\n      --- {ring}   R = {summ['radius_mm']:.4f} mm")
        print(f"          {'corner':<34}{'void':>8}{'leg mm':>10}{'T mm':>10}"
              f"{'T/leg':>8}{'R_max':>9}   fits")
        for name, c in summ["by_corner"].items():
            print(f"          {name:<34}{c['void_deg']:>8.2f}"
                  f"{c['spoke_side_leg_mm']:>10.4f}{c['tangent_length_mm']:>10.4f}"
                  f"{c['t_over_leg']:>8.2f}{c['r_max_on_this_leg_mm']:>9.4f}"
                  f"   {'yes' if c['fits_at_shipped_radius'] else 'NO'}")

    print("\n  STEP 1 GO/NO-GO — which idealisation lands closest to the part's second "
          "corner?")
    for ring in ("hub", "rim"):
        e = ver[ring]
        print(f"\n      {ring}:  the part's corner is at theta "
              f"{e['truth_theta_deg']:+.5f}, wedge {e['truth_wedge_deg']:.2f} deg, "
              f"lambda {e['truth_lambda']:.4f}")
        print(f"          {'candidate':<32}{'wedge err':>11}{'theta err':>11}"
              f"{'lambda err':>12}")
        for nm, c in e["candidates"].items():
            if "wedge_err_deg" not in c:
                print(f"          {nm:<32}   does not reach the ring circle")
                continue
            mark = "   <- best" if nm == e["best_on_wedge"] else ""
            print(f"          {nm:<32}{c['wedge_err_deg']:>11.2f}"
                  f"{c['theta_err_deg']:>11.4f}{c['lambda_err']:>12.4f}{mark}")
        print(f"          -> beats the end cap: {e['beats_the_end_cap']}"
              f"   (by {e['improvement_deg']:.2f} deg of wedge)")
    print(f"\n  VERDICT: {'GO' if ver['go'] else 'NO-GO at both rings'}"
          f"   |   rim alone: {'GO' if ver['go_rim_only'] else 'NO-GO'}")
    print("  (UNCAP_PLAN.md Step 1.  Geometry only — no field is solved here.)\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genome", default=os.path.join(ROOT, "best_solution.json"))
    ap.add_argument("--config", default="coarse")
    # Absolute by default so a standalone run lands in studies/ rather than the CWD
    # (§33's path defect), and used AS GIVEN when passed, which is what the Makefile
    # relies on -- it passes `studies/...` relative to the repo root.
    ap.add_argument("--out",
                    default=os.path.join(HERE, "study_junction_agreement.json"))
    args = ap.parse_args()

    with open(args.genome) as fh:
        genes = json.load(fh)["genes"]
    rec = build(genes, args.config)
    ver = verdict(rec)
    rec["step1_verdict"] = ver
    _print(rec, ver)

    out = args.out
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1, default=lambda o: (o.tolist() if hasattr(o, "tolist")
                                                        else str(o)))
    print(f"  wrote {out}")
    return 0 if rec["validation"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
