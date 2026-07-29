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
    .venv-cad\\Scripts\\python src\\wheel_step_export.py

It reads `best_solution.json`, produced by running `wheel_fea.py` on the optimizer
(Python 3.14).  The two interpreters share only that JSON file.

Normally you do NOT run this by hand — `wheel_fea.py` invokes it automatically at the
end of a GA run.  It used to be a separate manual step, which is exactly how
`wheel.step` came to sit 11 minutes older than the genome it claimed to describe
while `poster_summary.jpg` showed the current one.  Run standalone, it now warns
`[STALE]` when the STEP on disk predates `best_solution.json`.

Alongside the STEP it writes `wheel_step_manifest.json`: the genome hash the solid
was built from, junction-overlap volumes, and the fillet radii OCC actually managed
versus the ones the optimizer priced into its stress model.  That last pair is worth
reading — they do not currently agree (see `kt_report`).
=============================================================================
"""

import os
import sys
import json

import project_paths as PP   # stdlib-only; safe in the jax-free CAD env
import math
import time
import hashlib
import datetime
import collections
import numpy as np
import cadquery as cq

from OCP.ShapeCustom import ShapeCustom
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.Interface import Interface_Static
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.TopTools import TopTools_IndexedMapOfShape
from OCP.TopExp import TopExp
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCP.BRep import BRep_Tool
from OCP.TopoDS import TopoDS
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepAlgoAPI import BRepAlgoAPI_Check
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_State

# wheel_fea imports cleanly with only numpy (pygad/matplotlib are lazy in __main__).
from wheel_fea import (
    generate_bezier_centerline,
    thicken_3taper_curve,
    stress_concentration_kt,
    HUB_RADIUS_MM,
    RIM_RADIUS_MM,
    SPOKE_WIDTH_MM,
    NUMBER_OF_SPOKES,
    DENSITY_PLA,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = PP.ROOT
EXPORT = PP.EXPORT

# --- User-decided solid parameters -----------------------------------------
RIM_OUTER_RADIUS_MM = 50.0     # Ø100 outer rim; spokes merge at RIM_RADIUS_MM (Ø97.8)

# Embedding radii.  The spoke root is buried this deep inside the hub disk, and the
# spoke tip is run out this far past the outer diameter before being clipped back to
# Ø100.  Both regions are wholly absorbed by the hub disk / rim band, so nothing about
# the extension shape survives into the part — which is exactly why it is safe.
#
# These replace the old HUB_OVERLAP_MM / RIM_OVERLAP_MM radial *deltas*, which were
# applied by moving the last point of each offset-edge array and then interpolating the
# flank spline THROUGH the moved point.  That is what put a cusp in every spoke:
#
#   the first span jumped 3 mm while its clamped tangent pointed 40.7° against a local
#   curve direction of 76.1°, so the interpolating spline looped — self-intersecting at
#   (14.548, 2.807) on the bottom flank and (49.517, 2.878) on the top.  The boolean
#   union trimmed the loops away, but left a 0.041–0.067 mm radius near-cusp behind at
#   r≈14.78, one per spoke, running the full 22.4 mm height as a knife edge.  OCC calls
#   that valid — BRepCheck_Analyzer does not test for cusps — and Parasolid refuses to
#   build a face across it, drops the face, and the spoke imports OPEN.  That, not the
#   swept surfaces, is what Onshape was rejecting.
#
# The embedding is now done with straight tangent segments appended to the wire
# (`_extend_end`).  A straight line adds no curvature, so it cannot cusp, and because
# it continues the flank's own end tangent the join is G1 — not even a kink.
#
# Keep these SHALLOW.  The spoke meets both rings close to tangent, so a straight
# extension mostly travels sideways: aiming 3 mm inside the hub never gets there at
# all (closest approach 11.26 mm) and the run-on needed to try crosses the flanks.
# Nothing is gained by going deeper — the raw profile already penetrates the hub by
# 127 mm³ and the rim by 308 mm³, both far above MIN_JUNCTION_OVERLAP_MM3.  The
# extension only has to bury the blunt end caps just inside each ring.
HUB_EMBED_RADIUS_MM = HUB_RADIUS_MM - 0.5            # 12.20 mm, just inside the hub disk
RIM_EMBED_RADIUS_MM = RIM_OUTER_RADIUS_MM + 0.25     # 50.25 mm, clipped back to Ø100
MIN_JUNCTION_OVERLAP_MM3 = 50.0   # below this the weld is hairline — fail loudly
N_OUTLINE_PTS       = 48       # interpolation pts per spoke edge (splined → smooth faces)
# Vertical-edge screen: a junction edge wanders less than this in XY over its full
# height.  This is a straightness tolerance, NOT a radial one — see `_junction_edges`
# for why the radius stopped being a selector.
FILLET_TOL_MM       = 0.02
# A corner is worth filleting when its MATERIAL wedge exceeds 180 degrees, i.e. it is
# re-entrant.  Measured on the shipped genome, the five families of full-height vertical
# edge separate with a wide margin and nothing lands near the threshold:
#
#     r = 12.8748  x12   354.0 deg   the spoke<->spoke notch at the hub   <- fillet
#     r = 13.9352  x12   152.0 deg   bot[0], the spline/extension kink    <- convex
#     r = 47.5064  x12   180.0 deg   bot[-1], straight through            <- no corner
#     r = 48.5000  x24   194.0 .. 356.0 deg   the rim junctions           <- fillet
#     r = 50.0000  x 1   178.0 deg   the OD trim seam                     <- convex
#
# 185 rather than a rounder number because the nearest thing on either side is 180.0
# (7 deg of margin at 2 deg probe resolution) and 194.0 (9 deg).  Raising it to exclude
# "shallow" corners would drop the twelve 194 deg rim edges, which build fine today.
MIN_FILLET_WEDGE_DEG   = 185.0
FILLET_WEDGE_PROBE_MM  = 0.05    # probe circle radius; must clear the edge's own tolerance
FILLET_WEDGE_SAMPLES   = 180     # 2 deg resolution, ~2.6 s over the whole solid
# Hard floor on curvature radius anywhere in the profile.  The defect above measured
# 0.041 mm.  0.25 mm is well under any fillet we ask for and well over anything a
# 0.4 mm nozzle could print, so a violation always means a construction fault.
MIN_CURVATURE_RADIUS_MM = 0.25


def genome_hash(genes):
    """Short stable fingerprint of a gene dict, so a STEP can be traced to the run
    that produced it.  Canonicalised (sorted keys, fixed float repr) so the same
    genome always hashes the same regardless of dict ordering."""
    canon = json.dumps({k: round(float(v), 12) for k, v in sorted(genes.items())})
    return hashlib.sha256(canon.encode()).hexdigest()[:7]


def load_genome(path=None):
    path = path or PP.BEST_SOLUTION
    with open(path) as fh:
        rec = json.load(fh)
    rec["_source_path"] = path
    rec["_source_mtime"] = os.path.getmtime(path)
    rec["_hash"] = genome_hash(rec["genes"])
    return rec


def warn_if_stale(rec):
    """The bug this whole file was audited for: `wheel.step` sat 11 minutes older
    than `best_solution.json` and nothing said so, so the STEP silently described a
    previous genome while poster_summary.jpg showed the current one.  Running the
    exporter clears this; the warning exists for when it is run standalone."""
    step_path = os.path.join(EXPORT, "wheel.step")
    if not os.path.exists(step_path):
        return
    step_mtime = os.path.getmtime(step_path)
    if step_mtime < rec["_source_mtime"]:
        fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
        age = rec["_source_mtime"] - step_mtime
        print(f"  [STALE] the wheel.step on disk predates the genome by {age/60:.1f} min")
        print(f"          wheel.step         {fmt(step_mtime)}")
        print(f"          best_solution.json {fmt(rec['_source_mtime'])}")
        print( "          → it is about to be regenerated from the newer genome.")


def spoke_edges_global(genes):
    """
    Return (top, bot) offset-edge point arrays for one spoke, in the global wheel
    frame (spoke root on the +X hub circle, angle 0).

    These are the optimizer's thickened curve exactly as `thicken_3taper_curve`
    defines it — no endpoint surgery.  Embedding into the hub disk and rim band is a
    separate, purely straight-line step in `spoke_profile`.  Keeping the two apart is
    the whole fix: the point array that the flank spline interpolates is now smooth by
    construction, so the spline cannot loop.
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
    return top[idx], bot[idx]


def _unit(v):
    n = math.hypot(float(v[0]), float(v[1]))
    return np.asarray(v, float) / (n or 1.0)


def _embed(p_top, p_bot, d, target_r, outward, label, margin=0.05):
    """Straight-line continuation of BOTH flank ends along the shared direction `d`,
    by ONE common length, until both are past radius `target_r`.

    One length for both is what makes this safe: the two segments stay parallel, so the
    quadrilateral they close with the end cap is a trapezoid and cannot self-intersect
    at any depth.  Giving each flank its own length — or its own direction, or a
    two-segment path that turns radially when the tangent ray misses the circle — all
    reintroduce crossings; measured, per-flank tangents gave 6 self-intersections in
    the root and a two-segment fallback gave 1 more at r=11.83.

    Straight lines only, so no curvature is introduced and a cusp is impossible.
    Returns (q_top, q_bot).
    """
    p_top = np.asarray(p_top, float)
    p_bot = np.asarray(p_bot, float)
    d_tan = _unit(d)
    reach = target_r + margin if outward else target_r - margin
    span = 4.0 * (target_r + float(np.linalg.norm(p_bot)))
    L = np.linspace(0.0, span, 20001)[:, None]

    # The junction tangent alone often will not do it.  At the hub the spoke arrives
    # close to tangent, so travelling backwards along it skims the circle — measured,
    # the deepest it ever gets is r=11.26 for one flank while the other stays at 13.64,
    # OUTSIDE the hub.  A root corner left proud of the ring, with the flank curving
    # back in behind it, traps a 0.1 mm² lune: twelve tiny holes through the wheel.
    #
    # So rotate the shared direction toward the ring's radial until the target is
    # reachable, and take the least rotation that works — as tangential as the geometry
    # allows, but always buried.  Still one direction and one length for both flanks,
    # so the extensions stay parallel and cannot cross.
    radial = _unit((p_top + p_bot) / 2.0) * (1.0 if outward else -1.0)
    for blend in np.linspace(0.0, 1.0, 21):
        d = _unit((1.0 - blend) * d_tan + blend * radial)
        r_top = np.linalg.norm(p_top + L * d, axis=1)
        r_bot = np.linalg.norm(p_bot + L * d, axis=1)
        good = ((r_top >= reach) & (r_bot >= reach) if outward
                else (r_top <= reach) & (r_bot <= reach))
        if good.any():
            t = float(L[int(np.argmax(good)), 0])
            return p_top + t * d, p_bot + t * d

    # Cannot happen for a ring the spoke actually meets, but fail loudly rather than
    # silently shipping an unwelded spoke.
    raise RuntimeError(
        f"no {label} extension direction reaches r={target_r:.2f} mm from "
        f"r={np.linalg.norm(p_top):.2f}/{np.linalg.norm(p_bot):.2f} mm")


def _unit_tan(a, b):
    """Unit XY tangent from point a→b as a cq.Vector (z=0)."""
    v = np.asarray(b, float) - np.asarray(a, float)
    n = math.hypot(v[0], v[1]) or 1.0
    return cq.Vector(float(v[0] / n), float(v[1] / n), 0.0)


def spoke_profile(genes):
    """
    Closed PLANAR wire of one spoke.  Not a solid — the whole wheel is assembled in 2D
    and extruded once at the end, because the part is a constant-section prism and
    every 3D construction it used to go through (solid booleans, the 3D fillet solver,
    a tangential outer-diameter trim) was a place to produce geometry OCC accepts and
    Parasolid does not.

    Wire runs: hub extension → top flank spline (hub→rim) → rim extension → tip cap
    (outside Ø100, clipped away later) → rim extension → bottom flank spline
    (rim→hub) → hub extension → close (root cap, buried in the hub disk).

    The two flank splines interpolate the offset points untouched; only straight
    segments reach into the rings.  See HUB_EMBED_RADIUS_MM for why that matters.
    """
    top, bot = spoke_edges_global(genes)                      # each [N,2], hub→rim

    # Both flanks at one end are continued along the SAME direction — the mean of their
    # two end tangents, i.e. the centerline's direction — and by the same length.  Give
    # each flank its own tangent instead and they converge and cross: measured, 6
    # self-intersections in the root alone, because the two tangents close on each other
    # faster than the 2.4 mm root thickness can absorb over 3 mm of travel.
    d_hub = _unit((top[0] - top[1]) + (bot[0] - bot[1]))
    d_rim = _unit((top[-1] - top[-2]) + (bot[-1] - bot[-2]))
    q_top_hub, q_bot_hub = _embed(top[0], bot[0], d_hub,
                                  HUB_EMBED_RADIUS_MM, False, "hub")
    q_top_rim, q_bot_rim = _embed(top[-1], bot[-1], d_rim,
                                  RIM_EMBED_RADIUS_MM, True, "rim")

    P = lambda a: (float(a[0]), float(a[1]))
    top_pts = [P(p) for p in top]
    bot_rev = [P(p) for p in bot[::-1]]                       # rim→hub
    top_tan = [_unit_tan(top[0], top[1]), _unit_tan(top[-2], top[-1])]
    bot_tan = [_unit_tan(bot[-1], bot[-2]), _unit_tan(bot[1], bot[0])]

    return (cq.Workplane("XY")
            .moveTo(*P(q_top_hub))
            .lineTo(*P(top[0]))                               # hub extension, top
            .spline(top_pts, tangents=top_tan)                # top flank, hub → rim
            .lineTo(*P(q_top_rim))                            # rim extension, top
            .lineTo(*P(q_bot_rim))                            # tip cap, beyond Ø100
            .lineTo(*P(bot[-1]))                              # rim extension, bottom
            .spline(bot_rev, tangents=bot_tan)                # bottom flank, rim → hub
            .lineTo(*P(q_bot_hub))                            # hub extension, bottom
            .close()                                          # root cap, buried in hub
            .wire().val())


def profile_health(shape, label="", n=600):
    """Self-intersection and curvature audit of a planar wire or face, run BEFORE
    extruding.  Accepts a face because after filleting the junctions live on the
    profile's INNER wires — the twelve gaps between spokes — not its outer circle.

    This is the check that was missing.  Every existing check passed on a part with a
    41 µm cusp in every spoke, twice over: BRepCheck_Analyzer does not look for cusps,
    and BRepAlgoAPI_Check only ever saw the post-union solid, by which point the
    self-intersecting loop had been trimmed off and only the cusp it left remained.

    Returns a dict; prints and raises nothing on its own — callers decide.
    """
    worst_r, worst_at = _min_curvature(shape, n)

    # One polyline per closed wire, so crossings BETWEEN edges are caught as well as
    # within one.  Sample the WIRE, not its edges: Shape.Edges() comes back in TopExp
    # order, not traversal order, and stitching them in that order manufactures
    # crossings that do not exist (it reported four, two of them between a hub
    # extension and the rim tip).  Wire.positionAt walks arc length in wire order.
    n_bad, m = 0, max(n * 2, 400)
    for w in shape.Wires():
        poly = np.array([[p.x, p.y] for p in (w.positionAt(i / m) for i in range(m + 1))])
        n_bad += len(_self_intersections(poly))

    out = {"min_curvature_radius_mm": round(worst_r, 6),
           "self_intersections": n_bad,
           "worst_at": worst_at}
    tag = f" [{label}]" if label else ""
    print(f"  profile health{tag}: min curvature radius {worst_r:.4f} mm "
          f"(floor {MIN_CURVATURE_RADIUS_MM}) | self-intersections {n_bad}")
    if n_bad:
        print(f"  [SELF-INTERSECTING PROFILE] {n_bad} crossing(s).  The union trims the "
              f"loop\n                              away but leaves a cusp behind — that "
              f"is the Onshape failure.")
    if worst_r < MIN_CURVATURE_RADIUS_MM:
        print(f"  [CUSP] curvature radius {worst_r:.4f} mm at "
              f"{out['worst_at']} is below the {MIN_CURVATURE_RADIUS_MM} mm floor.\n"
              f"         Parasolid will not build a face across it and the spoke will "
              f"import open.")
    return out


def _self_intersections(pts):
    """Crossings between non-adjacent segments of a sampled planar polyline.

    Vectorised over the second index: each segment is tested against every later
    non-neighbouring one at once, which keeps a ~1200-point profile well under a
    second.  Returns the crossing points."""
    P = np.asarray(pts, float)
    A, D = P[:-1], np.diff(P, axis=0)
    out = []
    for i in range(len(A) - 2):
        q, s = A[i + 2:], D[i + 2:]
        r = D[i]
        den = r[0] * s[:, 1] - r[1] * s[:, 0]
        ok = np.abs(den) > 1e-12
        if not ok.any():
            continue
        qp = q[ok] - A[i]
        den = den[ok]
        t = (qp[:, 0] * s[ok][:, 1] - qp[:, 1] * s[ok][:, 0]) / den
        u = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / den
        hit = (t > 1e-9) & (t < 1 - 1e-9) & (u > 1e-9) & (u < 1 - 1e-9)
        out.extend(A[i] + tv * r for tv in t[hit])
    return out


def check_junction_overlap(spoke_face, hub_face, rim_face):
    """Measure how much spoke material actually lies inside the hub disk and inside
    the rim band.  A near-tangent junction can leave these vanishingly small while
    the boolean still reports a valid single solid — a hairline weld that looks fine
    in CAD and snaps in PLA.  Measured in 2D and scaled by the face width, which is
    the same number the old 3D intersection produced and far cheaper.

    Returns (hub_vol, rim_vol) and warns below the floor."""
    area = lambda s: sum(f.Area() for f in s.Faces())
    hub_vol = area(hub_face.intersect(spoke_face)) * SPOKE_WIDTH_MM
    rim_vol = area(rim_face.intersect(spoke_face)) * SPOKE_WIDTH_MM
    print(f"  junction overlap : hub {hub_vol:7.2f} mm³ | rim {rim_vol:7.2f} mm³ "
          f"(floor {MIN_JUNCTION_OVERLAP_MM3:.0f})")
    for label, vol in (("hub", hub_vol), ("rim", rim_vol)):
        if vol < MIN_JUNCTION_OVERLAP_MM3:
            print(f"  [WEAK JUNCTION] {label} overlap {vol:.2f} mm³ is below the floor — "
                  f"this spoke\n                  grazes the ring rather than meeting it. "
                  f"Deepen {label.upper()}_EMBED_RADIUS_MM,\n                  or constrain "
                  f"the optimizer's junction angle.")
    return hub_vol, rim_vol


def _unify_planar(shape):
    """Merge the coplanar faces of a planar boolean result into one face.

    `Shape.clean()` — ShapeUpgrade_UnifySameDomain on the boolean's compound — merges
    fragments within each argument but leaves the hub disk, the rim annulus and the
    twelve-spoke band as three separate faces, however far the linear and angular
    tolerances are loosened.  They sit in a compound rather than a shell, so the
    unifier does not see them as neighbours.  Sewing first supplies that adjacency,
    and the unifier then returns a single face (measured: 3 faces of 506.7 + 341.8 +
    1684.5 mm² become one of 2533.0).  Without this the extrude yields three solids
    sharing walls instead of one wheel.
    """
    sew = BRepBuilderAPI_Sewing(1e-6)
    for f in shape.Faces():
        sew.Add(f.wrapped)
    sew.Perform()
    up = ShapeUpgrade_UnifySameDomain(sew.SewedShape(), True, True, True)
    up.Build()
    return cq.Shape.cast(up.Shape())


def build_profile(genes):
    """The whole wheel as ONE planar face: hub disk + 12 spokes + rim band, clipped to
    the outer diameter.  Returns (face, (hub_overlap_mm3, rim_overlap_mm3)).

    Assembled with a single n-ary fuse rather than 14 sequential ones: fusing one at a
    time leaves the pieces unmerged, because each step re-splits what the previous one
    produced (measured: a compound of 26 faces, invalid).  `clean()` then runs
    ShapeUpgrade_UnifySameDomain to merge the coplanar fragments and the collinear
    outer arcs back into single faces and edges.
    """
    circle = lambda r: cq.Wire.makeCircle(r, cq.Vector(0, 0, 0), cq.Vector(0, 0, 1))
    hub = cq.Face.makeFromWires(circle(HUB_RADIUS_MM))
    rim = cq.Face.makeFromWires(circle(RIM_OUTER_RADIUS_MM), [circle(RIM_RADIUS_MM)])
    envelope = cq.Face.makeFromWires(circle(RIM_OUTER_RADIUS_MM))

    wire = spoke_profile(genes)
    profile_health(wire, "spoke profile")
    spoke0 = cq.Face.makeFromWires(wire)
    overlaps = check_junction_overlap(spoke0, hub, rim)

    spokes = [spoke0.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1),
                            k * 360.0 / NUMBER_OF_SPOKES)
              for k in range(NUMBER_OF_SPOKES)]

    profile = hub.fuse(rim, *spokes).clean()
    # The tip runs out to RIM_EMBED_RADIUS_MM; clip it back to the outer diameter.
    # In 2D that is an exact circle-vs-spline intersection whose arc merges into the
    # rim's own outer circle — unlike the old 3D `intersect(envelope)`, which cut a
    # solid at a shallow angle and left a 0.87 mm³ wedge per spoke.
    profile = _unify_planar(profile.intersect(envelope))

    faces = profile.Faces()
    if len(faces) != 1:
        raise RuntimeError(
            f"profile fuse produced {len(faces)} faces, expected 1 — unification "
            f"failed, and picking one of them would silently export a fragment of the "
            f"wheel (a broken spoke face once left just the bare hub disk behind)")
    return faces[0], overlaps


def extrude_profile(face):
    """One extrusion — the only 3D operation in the whole build."""
    return cq.Workplane(obj=face).toPending().extrude(SPOKE_WIDTH_MM)


def _material_wedge_deg(classifier, x, y, z,
                        radius=FILLET_WEDGE_PROBE_MM, samples=FILLET_WEDGE_SAMPLES):
    """Interior material angle at a vertical edge, by classifying a horizontal ring.

    A solid classifier rather than face normals and a dihedral sign: the sign convention
    needs the two adjacent faces in a known order, and at the 354 degree notch the sum of
    the outward normals nearly cancels (they sit 174 degrees apart), so the bisector that
    would carry the sign is exactly where the arithmetic is worst.  Sampling has neither
    problem, it returns the ANGLE and not just the sign, and it was measured stable across
    a 10x sweep of `radius` (0.30 / 0.10 / 0.03 mm all give 353.4 at the hub notch).
    """
    hit = 0
    for i in range(samples):
        t = 2.0 * math.pi * i / samples
        classifier.Perform(gp_Pnt(x + radius * math.cos(t),
                                  y + radius * math.sin(t), z), 1e-9)
        if classifier.State() != TopAbs_State.TopAbs_OUT:
            hit += 1
    return 360.0 * hit / samples


def _junction_edges(part):
    """Every full-height vertical edge that is a RE-ENTRANT corner, with its wedge angle.

    Returns `[(edge, radius_mm, wedge_deg), ...]`.

    THE RADIUS IS NO LONGER A SELECTOR, and that is the whole point.  This used to keep
    edges within `FILLET_TOL_MM` of a ring radius, on the stated premise that "junction
    vertices lie EXACTLY on the ring circles".  That premise is false: `_embed` drives the
    spoke roots deep enough and far enough tangentially that adjacent spokes lap over each
    other before either reaches the hub circle.  Measured on the shipped genome, the
    unfilleted solid has NO cylindrical face at r=12.70 at all and the hub circle is 360
    degrees of material at mid-sector — there is no spoke/hub junction to find.  What
    exists is a twelve-fold spoke/spoke notch 0.175 mm outside it, nine times the old
    tolerance away, and the export silently reported "no junction edges — skipped".

    Selecting on re-entrancy instead fillets the corners the part actually has, wherever
    `_embed` happened to leave them.  The full-height and straightness screens are kept:
    they are cheap, they are correct, and they cut 61 candidates out of the whole solid
    before the classifier is touched.
    """
    classifier = BRepClass3d_SolidClassifier(part.val().wrapped)
    rows = []
    for e in part.edges().vals():
        bb = e.BoundingBox()
        if bb.zlen < 0.9 * SPOKE_WIDTH_MM:                 # not full-height → not a junction
            continue
        if math.hypot(bb.xlen, bb.ylen) > FILLET_TOL_MM:   # wanders in XY → not vertical
            continue
        c = e.Center()
        wedge = _material_wedge_deg(classifier, c.x, c.y, c.z)
        if wedge >= MIN_FILLET_WEDGE_DEG:
            rows.append((e, math.hypot(c.x, c.y), wedge))
    return rows


def _group_by_ring(rows):
    """Split re-entrant edges into `(hub_rows, rim_rows)` at the midpoint of the rings.

    A coarse split on purpose.  The junctions sit near their rings but not on them, and
    the only thing this has to decide is which requested radius — `R_hub` or `R_rim` —
    each corner is priced against.  Anything finer would re-introduce the radial
    assumption that broke.
    """
    cut = 0.5 * (HUB_RADIUS_MM + RIM_RADIUS_MM)
    return ([r for r in rows if r[1] < cut], [r for r in rows if r[1] >= cut])


def fillet_junctions(part, rows, radius, label):
    """Fillet a group of re-entrant junction corners, all at one radius.

    `rows` comes from `_junction_edges` + `_group_by_ring` — `(edge, radius_mm, wedge_deg)`
    triples.  Taking the selection as an argument rather than doing it here is what lets
    the classifier run ONCE over the solid instead of once per ring, and it keeps the
    measured wedge angle available for the report.

    These belong in 2D on the profile — an extruded arc is an exact
    CYLINDRICAL_SURFACE, which is what Parasolid most wants to see.  OCC will not do
    it: `BRepFilletAPI_MakeFillet2d` (via `Face.fillet2D`) rejects any corner with a
    B-spline edge on it, which is every spoke-flank corner.  Measured on this profile,
    only the line↔arc corners took a fillet — half the junctions, the wrong half — and
    filleting the multi-wire face as a whole returned a single-wire face that
    BRepCheck called invalid.  So the fillets stay in 3D; the body is what needed
    rebuilding, not them.

    One radius for every edge in one operation, so a 12-fold symmetric part cannot come
    out with eleven spokes filleted one way and the twelfth another — the failure mode
    of the old per-edge greedy version, where each success changed the topology under
    the next probe.  Walking the ladder down and taking the largest radius that the
    whole batch accepts is also the honest answer to "what does this junction actually
    allow", which is what `kt_report` prices.

    Returns (part, applied_radius, n_filleted, n_found).  `n_filleted` and `n_found`
    are reported separately because they used to be the same field: a zero could mean
    either "nothing was found" or "everything was found and nothing could be built", and
    those want opposite fixes.

    Returns (part, applied_radius, n_filleted, n_found).
    """
    edges = [r[0] for r in rows]
    if not edges:
        print(f"  [{label}] no re-entrant junction corners found — skipped")
        return part, 0.0, 0, 0
    worst = max(r[2] for r in rows)
    print(f"  [{label}] {len(edges)} re-entrant corners, worst wedge {worst:.1f} deg")

    ladder = []
    r = radius
    while r > MIN_CURVATURE_RADIUS_MM:
        ladder.append(round(r, 3))
        r *= 0.85
    ladder.append(MIN_CURVATURE_RADIUS_MM)

    for r in ladder:
        try:
            out = part.newObject(edges).fillet(r)
            if out.val().isValid():
                note = "" if abs(r - radius) < 1e-9 else f"  (requested {radius:.3f})"
                print(f"  [{label}] filleted all {len(edges)} junction edges "
                      f"@ R={r:.3f} mm{note}")
                return out, r, len(edges), len(edges)
        except Exception:
            pass

        # The two corners at a junction are not mirror images — measured on the shipped
        # genome the rim's twenty-four run from 194° to 356° of material — and a
        # near-cusp accepts nothing.  Falling back to the subset that accepts r keeps the
        # useful half rather than losing both, but only if that subset is a whole multiple
        # of the spoke count, so a 12-fold symmetric wheel can never come out with one
        # spoke unlike the rest.
        ok = []
        for e in edges:
            try:
                part.newObject([e]).fillet(r)      # probe only, result discarded
                ok.append(e)
            except Exception:
                continue
        if not ok or len(ok) % NUMBER_OF_SPOKES or len(ok) == len(edges):
            continue
        try:
            out = part.newObject(ok).fillet(r)     # one operation, so they stay equal
        except Exception:
            continue
        if out.val().isValid():
            print(f"  [{label}] filleted {len(ok)} of {len(edges)} junction edges "
                  f"@ R={r:.3f} mm  (requested {radius:.3f}; "
                  f"{len(edges) - len(ok)} accepted no radius at all)")
            return out, r, len(ok), len(edges)

    print(f"  [{label}] no radius down to {MIN_CURVATURE_RADIUS_MM} mm was accepted by "
          f"all {len(edges)} edges — junctions left square")
    return part, 0.0, 0, len(edges)


# ---------------------------------------------------------------------------
# STEP INTEROPERABILITY
# ---------------------------------------------------------------------------
# Onshape rejected the exported solid:
#
#   wheel_nofillet.step was translated with errors. Parts with faults have been imported.
#   wheel.step was translated with errors.
#   The following 1 part(s) failed validation and have been removed:
#     1 unnamed - number of faults:10
#
# The solid is not broken.  Every check OCC can run passes: BRepCheck_Analyzer(full)
# valid, BRepAlgoAPI_Check with self-intersection clean, one closed manifold shell,
# 1e-7 mm tolerances throughout, shortest edge 0.907 mm, smallest face 20.3 mm², no
# degenerate edges, and 3D-curve-vs-pcurve agreement to 5.9e-8 mm.
#
# It is a REPRESENTATION problem.  `build_one_spoke` extrudes a B-spline wire, which
# OCC writes as SURFACE_OF_LINEAR_EXTRUSION — 24 of them, the spoke flanks.  Parasolid
# (Onshape's kernel) has no such surface type and must approximate every one on
# import.  That explains why the two files fail differently:
#
#   wheel_nofillet.step  51 faces, shortest edge 4.48 mm  → healing closes the drift
#   wheel.step           99 faces, shortest edge 0.907 mm, plus 48 small fillet faces
#                        built directly against those approximated supports → the
#                        fillets no longer meet them in tolerance → 10 faults
# ---------------------------------------------------------------------------

SWEPT_SURFACE_TYPES = ("SurfaceOfLinearExtrusion", "SurfaceOfRevolution", "OffsetSurface")


def despecialize(shape):
    """Convert swept surfaces to B-splines, leaving planes and cylinders analytic.

    ShapeCustom.ConvertToBSpline_s(shape, extrusion, revolution, offset) touches only
    the requested surface classes, so the hub bore and the Ø100 OD stay true cylinders
    and remain selectable as cylindrical faces in Onshape for mates and fillets.  The
    blunt alternative, BRepBuilderAPI_NurbsConvert, also works and is also exact but
    splines all 25 cylinders as well — not worth the downstream cost.

    Verified geometry-exact on the current genome: boolean symmetric difference
    0.00000 mm³ in both directions, shape still valid, tight bbox unchanged.

    ORDER MATTERS.  Running this on the union *before* filleting yields an invalid
    shape; running it on the final filleted solid is valid.  Call it last.
    """
    return cq.Shape.cast(
        ShapeCustom.ConvertToBSpline_s(_topo(shape), True, False, False))


def _topo(x):
    """Underlying TopoDS_Shape from a Workplane, a Shape, or a raw TopoDS_Shape."""
    if hasattr(x, "wrapped"):
        return x.wrapped
    if hasattr(x, "val"):
        return x.val().wrapped
    return x


def _surface_census(shape):
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(_topo(shape), TopAbs_FACE, m)
    c = collections.Counter()
    for i in range(1, m.Size() + 1):
        srf = BRep_Tool.Surface_s(TopoDS.Face_s(m.FindKey(i)))
        c[srf.DynamicType().Name().replace("Geom_", "")] += 1
    return dict(c)


def _min_curvature(shape, n=400):
    """Tightest curvature radius over every edge of a shape, and where it occurs.

    Sampled rather than analytic because the interesting edges are B-splines whose
    worst point is not at a knot.  Cheap: n points per edge, second differences.
    """
    obj = shape.val() if hasattr(shape, "val") else shape
    worst, at = float("inf"), None
    for e in obj.Edges():
        pts = np.array([[p.x, p.y, p.z] for p in (e.positionAt(i / n) for i in range(n + 1))])
        if np.ptp(pts, axis=0).max() < 1e-9:
            continue
        d1 = np.gradient(pts, axis=0)
        d2 = np.gradient(d1, axis=0)
        cr = np.cross(d1, d2)
        kap = np.linalg.norm(cr, axis=1) / np.maximum(
            np.linalg.norm(d1, axis=1) ** 3, 1e-18)
        i = int(kap.argmax())
        r = 1.0 / max(float(kap[i]), 1e-12)
        if r < worst:
            worst, at = r, [round(float(v), 3) for v in pts[i]]
    return worst, at


def step_health(shape, label=""):
    """The diagnostics that localised the Onshape failure, run every export so a
    future geometry change cannot silently reintroduce it.  Returns a dict for the
    manifest and prints a [STEP RISK] line if a swept surface survives."""
    s = _topo(shape)
    census = _surface_census(shape)

    counts = {}
    for typ, name in ((TopAbs_FACE, "faces"), (TopAbs_EDGE, "edges"),
                      (TopAbs_VERTEX, "vertices")):
        m = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(s, typ, m)
        counts[name] = m.Size()

    em = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(s, TopAbs_EDGE, em)
    min_edge, n_degen, max_tol = float("inf"), 0, 0.0
    for i in range(1, em.Size() + 1):
        e = TopoDS.Edge_s(em.FindKey(i))
        g = GProp_GProps()
        BRepGProp.LinearProperties_s(e, g)
        min_edge = min(min_edge, g.Mass())
        n_degen += 1 if BRep_Tool.Degenerated_s(e) else 0
        max_tol = max(max_tol, BRep_Tool.Tolerance_s(e))

    fm = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(s, TopAbs_FACE, fm)
    min_face = float("inf")
    for i in range(1, fm.Size() + 1):
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(TopoDS.Face_s(fm.FindKey(i)), g)
        min_face = min(min_face, g.Mass())

    valid = bool(BRepCheck_Analyzer(s, True).IsValid())
    try:
        chk = BRepAlgoAPI_Check(s, True, True)
        self_int = not bool(chk.IsValid())
    except Exception:
        self_int = None

    min_curv, curv_at = _min_curvature(shape)

    risky = {k: v for k, v in census.items() if k in SWEPT_SURFACE_TYPES}
    health = {
        "surface_census": census, "counts": counts,
        "min_edge_mm": round(min_edge, 6), "min_face_mm2": round(min_face, 4),
        "max_edge_tolerance_mm": max_tol, "degenerate_edges": n_degen,
        "brepcheck_valid": valid, "self_intersecting": self_int,
        "min_curvature_radius_mm": round(min_curv, 6),
        "min_curvature_at": curv_at,
        "swept_surfaces_remaining": risky,
    }

    tag = f" [{label}]" if label else ""
    print(f"\n  ---- STEP health{tag} ----")
    print(f"  surfaces         : {census}")
    print(f"  faces/edges/verts: {counts['faces']} / {counts['edges']} / {counts['vertices']}")
    print(f"  min edge {min_edge:.4f} mm | min face {min_face:.2f} mm² | "
          f"max tol {max_tol:.1e} mm | degenerate {n_degen}")
    print(f"  BRepCheck valid  : {valid}   self-intersecting: {self_int}")
    print(f"  min curvature R  : {min_curv:.4f} mm (floor {MIN_CURVATURE_RADIUS_MM})"
          + (f" at {curv_at}" if min_curv < MIN_CURVATURE_RADIUS_MM else ""))
    if min_curv < MIN_CURVATURE_RADIUS_MM:
        print(f"  [CUSP] an edge turns inside {MIN_CURVATURE_RADIUS_MM} mm.  Parasolid "
              f"will not build a face\n         across a cusp — it drops the face and "
              f"the spoke imports OPEN in Onshape.\n         This is the check that was "
              f"missing while three rounds of STEP tuning\n         chased the wrong "
              f"cause; do not ship a solid that trips it.")
    if risky:
        print(f"  [STEP RISK] {risky} survive in this solid.  Parasolid has no native")
        print( "              equivalent and must approximate them on import — this is")
        print( "              exactly what made Onshape reject the part.  Fix before shipping.")
    return health


def kt_report(genes, metrics, hub, rim):
    """Compare the fillet radii the optimizer PRICED IN against the ones OCC could
    actually build.

    `wheel_fea` charges each design a stress-concentration factor Kt derived from the
    R_hub / R_rim genes.  But a spoke arriving ~6° from tangent to the rim leaves no
    room for the radius it asked for — OCC quietly delivers a fraction of it, and the
    built part is sharper, and therefore more stressed, than the model believes.
    Nothing downstream noticed because main() discarded these return values.

    `hub` and `rim` are `(applied_radius, n_filleted, n_found, worst_wedge_deg)`.

    A SQUARE JUNCTION IS A NUMBER, NOT A NaN.  This used to report `kt_built = nan` when
    nothing was built and then filter non-positive `r_built` out of the mismatch gate
    below — so the worst case in the whole report was the one case that could not raise
    the alarm, and the manifest grew bare `NaN` tokens that are not valid JSON.  A square
    corner is priced through the branch `stress_concentration_kt` already has for it
    (`fillet_radius_mm < 0.1 → 3.5`), which is a real, large, comparable number.

    Returns a list of per-junction dicts for the manifest."""
    rows = []
    print("\n  ---- fillet feasibility (requested vs. built) ----")
    print(f"  {'junction':<9s} {'R_req':>7s} {'R_built':>8s} {'edges':>9s} "
          f"{'wedge':>7s} {'Kt_model':>9s} {'Kt_built':>9s} {'error':>8s}")
    print("  " + "-" * 76)
    for label, r_req, group, t in (
            ("hub", genes["R_hub"], hub, genes["t0"]),
            ("rim", genes["R_rim"], rim, genes["t3"])):
        r_built, n_filleted, n_found, wedge = group
        kt_model = float(stress_concentration_kt(r_req, t))
        kt_built = float(stress_concentration_kt(r_built, t))
        err = (kt_built / kt_model - 1.0) * 100.0
        print(f"  {label:<9s} {r_req:7.3f} {r_built:8.3f} "
              f"{n_filleted:4d}/{n_found:<4d} {wedge:7.1f} {kt_model:9.3f} "
              f"{kt_built:9.3f} {err:+7.1f}%")
        rows.append({"junction": label, "r_requested_mm": r_req, "r_built_mm": r_built,
                     "n_edges_filleted": n_filleted, "n_edges_found": n_found,
                     "worst_wedge_deg": wedge,
                     "kt_modeled": kt_model, "kt_built": kt_built,
                     "kt_error_pct": err})
    print("  " + "-" * 76)

    for row in rows:
        if row["r_built_mm"] <= 0:
            found = row["n_edges_found"]
            why = ("no re-entrant corner was found there at all — see `_junction_edges`"
                   if not found else
                   f"all {found} corners refused every radius down to "
                   f"{MIN_CURVATURE_RADIUS_MM} mm")
            print(f"  [SQUARE JUNCTION] the {row['junction']} junction took no fillet: "
                  f"{why}.\n                    Priced above at Kt = "
                  f"{row['kt_built']:.2f}, the degenerate-fillet value, which is a FLOOR:"
                  f"\n                    a square re-entrant corner is a worse riser "
                  f"than the {row['r_requested_mm']:.2f} mm the\n"
                  f"                    optimizer assumed, not a better one.")

    worst = max(rows, key=lambda r: r["kt_error_pct"], default=None)
    if worst and worst["kt_error_pct"] > 10.0:
        modeled = metrics.get("max_stress_mpa", float("nan"))
        scaled = modeled * worst["kt_built"] / worst["kt_modeled"]
        print(f"  [Kt MISMATCH] the {worst['junction']} fillet was built at "
              f"{worst['r_built_mm']:.2f} mm, not the {worst['r_requested_mm']:.2f} mm")
        print(f"                the optimizer assumed — Kt is {worst['kt_error_pct']:+.1f}% "
              f"higher than modeled.")
        print(f"                As-built peak stress scales {modeled:.2f} → ~{scaled:.2f} MPa.")
        print( "                The fix belongs upstream: nothing in wheel_fea.py stops the")
        print( "                GA proposing a fillet the junction angle cannot accept.")
    return rows


def report(part, genes, metrics, ghash, vol=None):
    """`vol` must be measured BEFORE despecialize().  BRepGProp.VolumeProperties_s
    uses low-order quadrature that is materially inaccurate on B-spline faces: the
    same solid measures 57151 mm³ with analytic spoke flanks and 60199 mm³ once they
    are splined — a 5.3% phantom gain, on geometry whose boolean symmetric difference
    is exactly zero.  Measure first, convert second, or the reported PLA mass lies."""
    solid = part.val()
    bb = solid.BoundingBox()
    if vol is None:
        vol = solid.Volume()
    print("\n  ---- solid report ----")
    print(f"  genome           : {ghash}")
    print(f"  valid (OCC)      : {solid.isValid()}")
    print(f"  solid count      : {len(part.solids().vals())} (expect 1)")
    print(f"  bounding box     : {bb.xlen:.2f} × {bb.ylen:.2f} × {bb.zlen:.2f} mm "
          f"(expect ≈ {2*RIM_OUTER_RADIUS_MM:.0f} × {2*RIM_OUTER_RADIUS_MM:.0f} × "
          f"{SPOKE_WIDTH_MM:.1f})")
    print(f"  volume           : {vol:.1f} mm³")
    print(f"  solid mass @PLA  : {vol * DENSITY_PLA:.2f} g "
          f"(optimizer wheel-spoke mass was {metrics.get('total_mass_g', float('nan')):.2f} g, "
          f"hub+rim add the rest)")
    return vol, solid.isValid()


def _export_with_settings(shape, path, schema=None, surfacecurve=None):
    """Write a STEP with temporarily-overridden writer settings.

    The Interface_Static registry is empty until a STEPControl_Writer exists, so one
    is instantiated first purely to populate it.  Settings are restored afterwards —
    they are process-global, and a leaked schema would silently affect every later
    variant in the same run."""
    writer = STEPControl_Writer()              # also populates the static registry
    old_schema = Interface_Static.CVal_s("write.step.schema")
    old_sc = Interface_Static.IVal_s("write.surfacecurve.mode")
    try:
        if schema is not None:
            Interface_Static.SetCVal_s("write.step.schema", schema)
        if surfacecurve is not None:
            Interface_Static.SetIVal_s("write.surfacecurve.mode", surfacecurve)
        # Drive STEPControl_Writer directly rather than cq.exporters.export — the
        # CadQuery wrapper re-applies its own writer settings, which silently undid
        # write.surfacecurve.mode (the file still carried all 873 PCURVEs).
        writer.Transfer(_topo(shape), STEPControl_AsIs)
        status = writer.Write(path)
        if status != IFSelect_ReturnStatus.IFSelect_RetDone:
            raise RuntimeError(f"STEP write returned status {status}")
    finally:
        Interface_Static.SetCVal_s("write.step.schema", old_schema)
        Interface_Static.SetIVal_s("write.surfacecurve.mode", old_sc)


def main():
    t_start = time.time()
    rec = load_genome()
    genes = rec["genes"]
    metrics = rec.get("metrics", {})
    print("=" * 68)
    print("  WHEEL STEP EXPORT")
    print("=" * 68)
    print(f"  genome {rec['_hash']}  ← {os.path.basename(rec['_source_path'])}")
    warn_if_stale(rec)
    print(f"  Spokes {NUMBER_OF_SPOKES} | Hub Ø{2*HUB_RADIUS_MM:.1f} (solid) | "
          f"Rim merge Ø{2*RIM_RADIUS_MM:.1f} → outer Ø{2*RIM_OUTER_RADIUS_MM:.1f} | "
          f"Face {SPOKE_WIDTH_MM:.1f} mm")
    print(f"  R_hub={genes['R_hub']:.3f} mm  R_rim={genes['R_rim']:.3f} mm  "
          f"t0={genes['t0']:.2f} t3={genes['t3']:.2f}")

    print("\n  Building 2D profile (hub + 12 spokes + rim)…")
    profile, (hub_ovl, rim_ovl) = build_profile(genes)

    # Guaranteed-valid fallback, written before any fillet can complicate it.
    nofillet_path = os.path.join(EXPORT, "wheel_nofillet.step")
    _export_with_settings(despecialize(extrude_profile(profile)), nofillet_path)
    print(f"  Saved fallback   → {nofillet_path}")

    prof_health = profile_health(profile, "wheel profile")

    print("\n  Extruding…")
    wheel = extrude_profile(profile)

    print("\n  Filleting junctions…")
    # Classify once over the whole solid, then split — the classifier is the expensive
    # part and it does not depend on which ring a corner belongs to.
    hub_rows, rim_rows = _group_by_ring(_junction_edges(wheel))
    hub_wedge = max((r[2] for r in hub_rows), default=0.0)
    rim_wedge = max((r[2] for r in rim_rows), default=0.0)
    wheel, hub_r, hub_n, hub_found = fillet_junctions(wheel, hub_rows, genes["R_hub"], "hub")
    # Re-select against the NEW solid: filleting the hub replaces the faces the rim edges
    # are bound to, and OCC will not accept stale edge handles.  Safe to re-run because a
    # fillet introduces only TANGENT edges, which are smooth and fall below
    # MIN_FILLET_WEDGE_DEG — verified by filleting the rim and re-selecting: the 24
    # re-entrant corners become exactly the 12 that refused a radius, and the hub set is
    # untouched.  Does not fire today (the hub accepts nothing); kept because the day it
    # does, a stale handle would be a confusing crash rather than a wrong answer.
    if hub_n:
        _, rim_rows = _group_by_ring(_junction_edges(wheel))
    wheel, rim_r, rim_n, rim_found = fillet_junctions(wheel, rim_rows, genes["R_rim"], "rim")
    kt_rows = kt_report(genes, metrics,
                        (hub_r, hub_n, hub_found, hub_wedge),
                        (rim_r, rim_n, rim_found, rim_wedge))

    # Measure BEFORE despecializing — see report()'s docstring.
    vol_true = wheel.val().Volume()
    vol, valid = report(wheel, genes, metrics, rec["_hash"], vol=vol_true)

    print("\n  Converting swept surfaces for Parasolid/Onshape…")
    wheel_out = despecialize(wheel)
    health = step_health(wheel_out, "wheel.step")

    step_path = os.path.join(EXPORT, "wheel.step")
    _export_with_settings(wheel_out, step_path)
    print(f"\n  Saved STEP       → {step_path}")

    # Manifest: what this STEP actually is, so it can never again be mistaken for a
    # build of a different genome.
    manifest = {
        "genome_hash": rec["_hash"],
        "source": os.path.basename(rec["_source_path"]),
        "source_mtime": datetime.datetime.fromtimestamp(
            rec["_source_mtime"]).isoformat(timespec="seconds"),
        "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "export_seconds": round(time.time() - t_start, 1),
        "solid": {"valid": bool(valid), "volume_mm3": round(vol, 1),
                  "mass_g_pla": round(vol * DENSITY_PLA, 2)},
        "junction_overlap_mm3": {"hub": round(hub_ovl, 2), "rim": round(rim_ovl, 2),
                                 "floor": MIN_JUNCTION_OVERLAP_MM3},
        # `detail` carries found-vs-filleted per junction; these two remain as the count
        # actually built, which is what the old readers expect.
        "fillets": {"hub_edges": hub_n, "rim_edges": rim_n, "detail": kt_rows},
        "profile_health": prof_health,
        "step_health": health,
    }
    manifest_path = os.path.join(EXPORT, "wheel_step_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  Saved manifest   → {manifest_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
