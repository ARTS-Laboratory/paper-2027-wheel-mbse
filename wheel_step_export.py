"""
=============================================================================
  COMPLIANT PLA UAV WHEEL — STEP EXPORTER
=============================================================================
Builds a watertight B-rep solid of the full optimized wheel and writes it to
`wheel.step` (+ a guaranteed-valid `wheel_nofillet.step` fallback) for import into
Fusion 360 / Inventor / SolidWorks.

  Full wheel = solid hub disk  +  12 spiral spokes  +  Ø100 rim band,
  unioned into one solid, with true tangent fillets at the spoke↔hub and
  spoke↔rim junctions (radii = evolved R_hub / R_rim genes).

RUN THIS IN THE CadQuery ENV (Python 3.12), e.g.:
    .venv-cad\\Scripts\\python wheel_step_export.py

It reads `best_solution.json`, produced by running `wheel_fea.py` on the optimizer
(Python 3.14).  The two interpreters share only that JSON file.
=============================================================================
"""

import os
import json
import math
import numpy as np
import cadquery as cq

# wheel_fea imports cleanly with only numpy (pygad/matplotlib are lazy in __main__).
from wheel_fea import (
    generate_bezier_centerline,
    thicken_3taper_curve,
    HUB_RADIUS_MM,
    RIM_RADIUS_MM,
    SPOKE_WIDTH_MM,
    NUMBER_OF_SPOKES,
    DENSITY_PLA,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# --- User-decided solid parameters -----------------------------------------
RIM_OUTER_RADIUS_MM = 50.0     # Ø100 outer rim; spokes merge at RIM_RADIUS_MM (Ø97.8)
HUB_OVERLAP_MM      = 1.0      # extend spoke root inward into the hub disk (clean union)
RIM_OVERLAP_MM      = 0.0      # spoke tip already reaches ~r49.9 into the band; no push-out
N_OUTLINE_PTS       = 48       # interpolation pts per spoke edge (splined → smooth faces)
# Junction edges lie EXACTLY on the ring cylinders (r = hub/rim radius); a tight
# tolerance isolates them from the spoke's own polyline facet edges (nearest ≈0.03 mm off).
FILLET_TOL_MM       = 0.02


def load_genome(path=None):
    path = path or os.path.join(HERE, "best_solution.json")
    with open(path) as fh:
        rec = json.load(fh)
    return rec


def spoke_edges_global(genes):
    """
    Return (top, bot) offset-edge point arrays for one spoke, in the global wheel
    frame (spoke root on the +X hub circle, angle 0), with the ends extended a little
    into the hub disk / rim band so the boolean union is clean.
    """
    g = genes
    curve, _ = generate_bezier_centerline(
        g["cx1"], g["cy1"], g["cx2"], g["cy2"],
        g["cx3"], g["cy3"], g["cx4"], g["cy4"])
    top, bot = thicken_3taper_curve(
        curve, g["t0"], g["t1"], g["t2"], g["t3"], return_edges=True)

    # Local → global: shift X by hub radius (place_sector at angle 0).
    top = top.copy(); top[:, 0] += HUB_RADIUS_MM
    bot = bot.copy(); bot[:, 0] += HUB_RADIUS_MM

    # Downsample but always keep the exact endpoints.
    idx = np.unique(np.linspace(0, len(top) - 1, N_OUTLINE_PTS).astype(int))
    top = top[idx]
    bot = bot[idx]

    # Extend the hub end radially inward and the rim end radially outward.
    for arr in (top, bot):
        r0 = math.hypot(*arr[0])
        arr[0] = arr[0] * (r0 - HUB_OVERLAP_MM) / r0
        r1 = math.hypot(*arr[-1])
        arr[-1] = arr[-1] * (r1 + RIM_OVERLAP_MM) / r1
    return top, bot


def _unit_tan(a, b):
    """Unit XY tangent from point a→b as a cq.Vector (z=0)."""
    v = np.asarray(b, float) - np.asarray(a, float)
    n = math.hypot(v[0], v[1]) or 1.0
    return cq.Vector(float(v[0] / n), float(v[1] / n), 0.0)


def build_one_spoke(genes):
    """
    Closed profile of one spoke, extruded to the face width.  The two long lateral
    edges are single B-splines (smooth NURBS faces in the STEP); the two short end
    caps (blunt rim tip, hub root) are straight lines.  Clamped end tangents pin the
    spline endpoints so the curve cannot overshoot past the rim / into the hub —
    the failure mode of splining the whole closed loop through the ~180° corners.
    Wire runs: top hub→rim, rim cap, bot rim→hub, hub cap (close).
    """
    top, bot = spoke_edges_global(genes)                      # each [N,2], hub→rim
    top_pts = [(float(x), float(y)) for x, y in top]
    bot_rev = [(float(x), float(y)) for x, y in bot[::-1]]    # rim→hub

    top_tan = [_unit_tan(top[0], top[1]), _unit_tan(top[-2], top[-1])]
    bot_tan = [_unit_tan(bot[-1], bot[-2]), _unit_tan(bot[1], bot[0])]  # rim→hub

    wp = (cq.Workplane("XY")
          .spline(top_pts, tangents=top_tan)                  # hub → rim (top)
          .lineTo(*bot_rev[0])                                # rim cap
          .spline(bot_rev, tangents=bot_tan)                  # rim → hub (bot)
          .close())                                           # hub cap
    return wp.extrude(SPOKE_WIDTH_MM)


def build_wheel(genes):
    """Union of hub disk + 12 patterned spokes + rim band → single solid."""
    hub = cq.Workplane("XY").circle(HUB_RADIUS_MM).extrude(SPOKE_WIDTH_MM)
    rim = (cq.Workplane("XY")
           .circle(RIM_OUTER_RADIUS_MM).circle(RIM_RADIUS_MM)
           .extrude(SPOKE_WIDTH_MM))

    spoke0 = build_one_spoke(genes)
    result = hub
    for k in range(NUMBER_OF_SPOKES):
        angle = k * (360.0 / NUMBER_OF_SPOKES)
        result = result.union(spoke0.rotate((0, 0, 0), (0, 0, 1), angle))
    result = result.union(rim)
    return result


def _select_junction_edges(part, target_r):
    """Vertical edges lying exactly on the ring cylinder r=target_r, selected by
    GEOMETRY (full-height Z span + ~constant XY) rather than by curve type: the
    spoke↔ring junctions bound a spline face, so OCC types them BSPLINE — a plain
    `.edges("|Z")` (linear only) would miss them.  The tight XY/radius tol excludes
    the blunt tip-cap corners (different radius) and the ring boundary circles (dz≈0)."""
    out = []
    for e in part.edges().vals():
        bb = e.BoundingBox()
        if bb.zlen < 0.9 * SPOKE_WIDTH_MM:                 # not full-height → not a junction
            continue
        if math.hypot(bb.xlen, bb.ylen) > FILLET_TOL_MM:   # wanders in XY → not vertical
            continue
        c = e.Center()
        if abs(math.hypot(c.x, c.y) - target_r) <= FILLET_TOL_MM:
            out.append(e)
    return out


def fillet_junctions(part, target_r, radius, label):
    """
    Fillet the concave spoke↔ring junction edges on the cylinder r=target_r.

    Strategy: try the full requested radius as one batch (fast, uniform). If OCC
    rejects it, fall back to per-edge with a descending radius search — this
    naturally rounds only the sharp *notch* side of each shallow spiral junction
    (the near-tangent side has no corner and OCC declines it), at the largest radius
    that fits the local material. Symmetric across spokes by construction.

    Returns (part, effective_radii, n_filleted, n_edges).
    """
    edges = _select_junction_edges(part, target_r)
    if not edges:
        print(f"  [{label}] no junction edges near r={target_r:.2f} mm — skipped")
        return part, [], 0, 0

    # 1) Uniform batch at (possibly reduced) radius.
    for r in (radius, radius * 0.6, radius * 0.35):
        try:
            out = part.newObject(edges).fillet(r)
            note = "" if abs(r - radius) < 1e-9 else f"  (reduced from {radius:.3f})"
            print(f"  [{label}] filleted all {len(edges)} edges @ R={r:.3f} mm{note}")
            return out, [r], len(edges), len(edges)
        except Exception:
            pass
    print(f"  [{label}] uniform fillet failed at R≤{radius:.3f}; trying per-edge…")

    # 2) Per-edge: fillet each notch at the largest radius it accepts.
    #    Re-select edges after every successful fillet (topology changes).
    candidate_radii = [radius, radius * 0.6, radius * 0.35, radius * 0.2, 0.4, 0.25]
    candidate_radii = sorted({round(x, 3) for x in candidate_radii if x > 0.1}, reverse=True)
    applied = []
    done = 0
    remaining = len(edges)
    while True:
        edges = _select_junction_edges(part, target_r)
        progressed = False
        for e in edges:
            for r in candidate_radii:
                try:
                    part = part.newObject([e]).fillet(r)
                    applied.append(r)
                    done += 1
                    progressed = True
                    break
                except Exception:
                    continue
            if progressed:
                break  # topology changed; re-select before continuing
        if not progressed:
            break
    if applied:
        lo, hi = min(applied), max(applied)
        rng = f"{lo:.2f} mm" if abs(hi - lo) < 1e-6 else f"{lo:.2f}–{hi:.2f} mm"
        print(f"  [{label}] filleted {done} notch edge(s) @ R={rng} "
              f"(the near-tangent sides have no corner to round)")
    else:
        print(f"  [{label}] no edge accepted a fillet — junctions left square")
    return part, applied, done, remaining


def report(part, genes, metrics):
    solid = part.val()
    bb = solid.BoundingBox()
    vol = solid.Volume()
    print("\n  ---- solid report ----")
    print(f"  valid (OCC)      : {solid.isValid()}")
    print(f"  bounding box     : {bb.xlen:.2f} × {bb.ylen:.2f} × {bb.zlen:.2f} mm "
          f"(expect ≈ {2*RIM_OUTER_RADIUS_MM:.0f} × {2*RIM_OUTER_RADIUS_MM:.0f} × "
          f"{SPOKE_WIDTH_MM:.1f})")
    print(f"  volume           : {vol:.1f} mm³")
    print(f"  solid mass @PLA  : {vol * DENSITY_PLA:.2f} g "
          f"(optimizer wheel-spoke mass was {metrics.get('total_mass_g', float('nan')):.2f} g, "
          f"hub+rim add the rest)")


def main():
    rec = load_genome()
    genes = rec["genes"]
    metrics = rec.get("metrics", {})
    print("=" * 68)
    print("  WHEEL STEP EXPORT")
    print("=" * 68)
    print(f"  Spokes {NUMBER_OF_SPOKES} | Hub Ø{2*HUB_RADIUS_MM:.1f} (solid) | "
          f"Rim merge Ø{2*RIM_RADIUS_MM:.1f} → outer Ø{2*RIM_OUTER_RADIUS_MM:.1f} | "
          f"Face {SPOKE_WIDTH_MM:.1f} mm")
    print(f"  R_hub={genes['R_hub']:.3f} mm  R_rim={genes['R_rim']:.3f} mm  "
          f"t0={genes['t0']:.2f} t3={genes['t3']:.2f}")

    print("\n  Building union (hub + 12 spokes + rim)…")
    wheel = build_wheel(genes)

    # Guaranteed-valid fallback first.
    nofillet_path = os.path.join(HERE, "wheel_nofillet.step")
    cq.exporters.export(wheel, nofillet_path)
    print(f"  Saved fallback   → {nofillet_path}")

    print("\n  Applying tangent fillets…")
    wheel, _, _, _ = fillet_junctions(wheel, HUB_RADIUS_MM, genes["R_hub"], "hub")
    wheel, _, _, _ = fillet_junctions(wheel, RIM_RADIUS_MM, genes["R_rim"], "rim")

    report(wheel, genes, metrics)

    step_path = os.path.join(HERE, "wheel.step")
    cq.exporters.export(wheel, step_path)
    print(f"\n  Saved STEP       → {step_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
