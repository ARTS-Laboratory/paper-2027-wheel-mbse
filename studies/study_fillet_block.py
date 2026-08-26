"""
=============================================================================
  CAN THE FILLET BE A BLOCK, AND CAN THE SECTOR BE BLOCKED AROUND IT?  THE
  REGION HAS TWO CUSPS; THE BLOCK THAT MESHES HAS ITS CORNERS OFF BOTH TANGENT
  POINTS; AND THE SECTOR AROUND IT CLOSES IN ELEVEN BLOCKS AND FOURTEEN SEAMS
=============================================================================
    .venv-opt/bin/python studies/study_fillet_block.py       (make filletblock)

FILLET_PLAN.md STEP 1 RECORD PART 9, and PLAN.md §44/§46's ranked item 1.

WHY THIS EXISTS
---------------
PART 3 (2026-08-17) left the arc with exactly two routes and they have been ranked first
for nine arcs:

    1.  a DEDICATED FILLET BLOCK "covering the curvilinear triangle A - P_t - B, with its
        own seam entries" -- the preferred route;
    2.  a GENERATED SPOKE BLOCK -- transfinite smoothing with a boundary correction that
        decays away from the junction.

`make fillet` (PART 6) settled what "valid" means for either of them: `det J` at the
Gauss points the assembly integrates.  It did not ask whether either route's REGION can
be a block at all.  This file asks that, in geometry only, before anybody spends a week
building one.  It is the same discipline PART 7 and PART 8 applied to Step 0 and to
PART 2 -- re-check the premise before spending on the step.

THE ANSWER, IN ONE LINE EACH
----------------------------
  ROUTE 2 IS DEAD, and not by a tolerance.  What fails in the shipped `fillet=`
  construction is the ANGLE AT THE MOVED CORNER, and both of the curves that make it are
  BOUNDARY curves of the spoke block -- the fillet arc on the flank edge, and the end
  cross-section.  A generated interior is by definition a scheme that moves INTERIOR
  nodes; the three nodes that carry this angle are all boundary nodes.  Measured here by
  running an elliptic (Winslow) interior solve on the block and reporting that the corner
  angle comes back BIT-IDENTICAL.

  ROUTE 1 AS WRITTEN IS DEAD TOO, and this one is a geometry fact nobody had measured.
  The region `A - P_t - B` is not a curvilinear triangle with three usable corners.  A
  fillet TANGENT to both legs meets each of them at zero angle, so the region it adds is
  a CUSP SLIVER: measured interior angles 0.0000 deg at `B` and 0.42-0.60 deg at `A`,
  against 38.06-38.89 deg at `P_t`.  No quad block covers a 0 deg corner, and neither
  does a tri-block -- a tri-block subdivides the region's corners, it does not create
  new ones.

  AND THE SPOKE BLOCK WAS NEVER THE BLOCKER.  PART 3's finding -- "what actually blocks a
  filleted mesh is that the spoke block is ruled" -- is an artefact of WHERE the fillet
  was put, not of the spoke.  Take the arc out of the spoke and end the spoke at the
  tangent station `s_A` instead, and the spoke block is clean at the SHIPPED radii at
  `coarse` and `medium`, where the shipped construction's usable window is 0.12-0.24 mm.
  The fold does not disappear -- it moves, whole, into whatever block then has to carry
  the fillet.

WHAT DOES WORK, MEASURED
------------------------
A BOUNDARY-LAYER block, which is what a structured mesher builds along a fillet:

    j0  the fillet arc `A -> B`                        the free surface
    j1  that arc offset INTO the material              full wall at `A`, depth `d` at `B`
    i0  the spoke's end cross-section at `s_A`         cuts across the flank AT `A`
    i1  a radial cut `B -> B''` of depth `d`           cuts across the ring circle AT `B`

Its four corners are OFF both tangent points, which is the whole trick: the cusp is
interior to an edge instead of being a corner.  Measured `min scaled Jacobian` 0.87-1.00,
zero mixed-sign cells, zero non-positive Gauss points, at the SHIPPED radii, at both
junctions, at `coarse` and `medium`, and stable under refinement -- the first filleted
block in this arc that meshes at the radii that actually ship.

WHAT IT COSTS, WHICH IS THE HONEST OTHER HALF
---------------------------------------------
`i1` cuts ACROSS the ring circle.  That is not incidental -- it is forced, and this file
measures why: at `B` the material on the free-surface side has zero thickness, so a block
that stops at the ring circle there degenerates, and a block that crosses it does not.
So THE RING CIRCLE STOPS BEING THE JUNCTION/COLLAR INTERFACE NEAR THE FILLET, and the
collar (hub) / band (rim) block has to be notched to depth `d`.  That is a re-cut of the
neighbours, not an eighth block bolted on, and it is priced in the report's `price`
section rather than guessed at.

TWO CANDIDATES THAT DO NOT WORK ARE MEASURED HERE TOO, because both are the obvious thing
to try and both fail for the same reason -- they keep a block edge on the pre-fillet
surfaces through a tangent point:

    grown_junction        j0 = [fillet arc] + [ring arc], the junction block grown
    pre_fillet_surfaces   PART 3's region, closed by the two end cross-sections

Both fold, both get WORSE under refinement, and an elliptic interior solve does not
rescue either.

AND THE WHOLE SECTOR, WHICH IS WHAT A BLOCK THAT MESHES IS NOT
-------------------------------------------------------------
A block that meshes is not a mesh.  The block above has an inner edge that CROSSES the
ring circle, so half of it seams to the junction block and half to the ring block, and
one edge with two partners is a partial-edge seam -- which this tree has never had.
`filleted_sector` builds the whole thing: ELEVEN blocks and FOURTEEN seams, every seam
whole-edge, with the fillet block split at the crossing `N`.  Measured at `coarse` and
`medium`, at the shipped radii and across the admissible gene box: every block valid,
every seam closed to under 1.5e-14 mm, worst min scaled Jacobian 0.35 against the
unfilleted sector's 0.78 and `MIN_SJ_TARGET`'s 0.2.

Two things fall out of it that PART 9 could not see from one block.  The cut at `B` has
to reach the ring's FAR boundary -- the hub bore, the rim's outer surface -- because a
shallow one cannot be closed with whole-edge quads at all; and the radius is now bounded
by the SECTOR rather than by the block, at `R_hub = 3.130 mm`, where the fillet's tangent
point reaches the next sector's corner.

WHAT THIS DOES NOT DO
---------------------
It solves no field, touches no `best_solution.json`, and changes no mesh: every block
here is built in this file from `wheel_wheel`'s own primitives and nothing is written
back.  `wheel_wheel.sector_blocks` is called only with `fillet=None` and with the shipped
`fillet=` path, both of which already exist.  It is geometry and Jacobians and it runs in
seconds.

EXIT STATUS follows `make fillet` and `make junction`: nonzero ONLY if a self-check fails
-- the controls, the cusp measurement, or the route-2 invariance.  Never on a
characterisation finding about a candidate block, which is what this exists to report.
=============================================================================
"""

import argparse
import json
import math
import os
import time

import numpy as np

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import _gate_guard
import wheel_genome as wg
import wheel_geometry as WG
import wheel_wheel as WW
import study_fillet_fold as ff

HERE = os.path.dirname(os.path.abspath(__file__))

# The barrier `wheel_objective` puts on the mesh.  Imported rather than repeated, for the
# same reason `study_tri_block` imports it: a floor quoted from memory in a study is a
# floor that drifts from the one the optimizer enforces.
try:
    import wheel_objective as WO
    MIN_SJ_TARGET = float(WO.MIN_SJ_TARGET)
except Exception:                                    # pragma: no cover - import guard
    MIN_SJ_TARGET = 0.2

DEFAULT_CONFIGS = ("coarse", "medium")
DEFAULT_JUNCTIONS = ("hub", "rim")

# The radii swept.  The low end is where `make fillet`'s usable window sits (0.07-0.24
# mm), the high end is the gene box: `R_hub` runs 0.4-4.0 and `R_rim` 0.5-3.0, and the
# shipped genome is 0.6636 / 3.0000.  The cusp claim is a claim about EVERY radius, so
# the grid has to span the box rather than sit on the shipped point.
RADII = (0.05, 0.10, 0.20, 0.40, 1.00, 1.50, 2.00, 3.00, 4.00)


def radius_grid(genes):
    """`RADII` with the genome's OWN two radii merged in.

    The shipped point has to be ON the grid rather than near it: every table in this file
    is read at "the radius that ships", and 0.6636 is not 0.6636060402965218.
    """
    return tuple(sorted(set(RADII) | {float(genes[12]), float(genes[13])}))

# Refinement multipliers on the candidate blocks' long edge.  A construction whose fold
# gets WORSE as its own resolution rises is a construction whose region is wrong, which
# is the distinction this column exists to draw.
REFINEMENTS = (1, 2, 4)

CANDIDATES = ("grown_junction", "pre_fillet_surfaces", "boundary_layer")

# Winslow relaxation for the "does a generated interior rescue it" column.  2000 sweeps
# at omega=0.4 is far past where any of these stop moving; the point is not to build a
# production smoother but to give every candidate the benefit of one.
WINSLOW_ITERS = 2000
WINSLOW_OMEGA = 0.4


def load_genes(path):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


# ---------------------------------------------------------------------------
# THE GEOMETRY, TAKEN FROM THE SHIPPED CONSTRUCTION RATHER THAN RE-DERIVED
# ---------------------------------------------------------------------------

def junction_geometry(genes, cfg, junction, R):
    """Everything the fillet at one junction is made of, at radius `R`.

    `_fillet_tangency` is `wheel_wheel`'s own solve and is called here rather than
    reimplemented: every number below is then a statement about the construction the tree
    ships, not about a second copy of it that could drift.  Returns `None` when no fillet
    of that radius is tangent to both legs, which is the tangency solve's own refusal and
    is a legitimate outcome at the top of the box.
    """
    cfg = WW.get_config(cfg)
    span = WW.HUB_RIM_SPAN_MM
    orientation = WW.flank_orientation(genes, cfg, span_mm=span)
    rim_inner = WW.rim_inner_radius(span)
    sample, s_dense = WW.global_sampler(genes, cfg, span_mm=span)
    s_hub, s_rim = WW.junction_stations(sample, s_dense, orientation, rim_inner)
    is_hub = junction == "hub"
    k = 0 if is_hub else 1
    s_end, s_far = (s_hub, s_rim) if is_hub else (s_rim, s_hub)
    s_ring = 0.0 if is_hub else 1.0
    ring_r = WW.HUB_RADIUS_MM if is_hub else rim_inner
    # How much room the cut at `B` has on the far side of the ring circle: the collar is
    # 5 mm deep inward at the hub, the band 1.1-1.5 mm outward at the rim.
    depth_available = (WW.COLLAR_DEPTH_MM if is_hub
                       else WW.RIM_OUTER_RADIUS_MM - rim_inner)
    eta = 1.0 if float(orientation[k]) > 0 else -1.0
    far_pt = np.asarray(sample(np.asarray(float(s_end)), np.asarray(-eta)), float)
    void_sign = 1.0 if float(np.linalg.norm(far_pt)) > ring_r else -1.0
    P_t = np.asarray(sample(np.asarray(float(s_end)), np.asarray(eta)), float)
    try:
        s_A, A, B, C = WW._fillet_tangency(sample, s_end, s_far, eta, ring_r, R,
                                           void_sign)
    except ValueError as exc:
        return {"tangency": False, "message": str(exc).split(":")[0]}
    blend = WW._uncap_blend(WW.UNCAP_DEFAULT, is_hub)
    Q = np.asarray(WW._uncap_corner(sample, s_ring, eta, ring_r, is_hub, blend, np),
                   float)
    far_end = np.asarray(sample(np.asarray(s_ring), np.asarray(-eta)), float)
    return {"tangency": True, "cfg": cfg, "sample": sample, "eta": eta,
            "void_sign": void_sign, "ring_r": ring_r, "R": R,
            "s_end": float(s_end), "s_A": float(s_A), "s_ring": float(s_ring),
            "s_hub": float(s_hub), "s_rim": float(s_rim),
            "P_t": P_t, "A": A, "B": B, "C": C, "Q": Q, "far_end": far_end,
            "depth_available_mm": float(depth_available)}


def _unit(v):
    n = float(np.linalg.norm(v))
    return np.asarray(v, float) / (n or 1.0)


def _angle_deg(u, v):
    c = float(np.dot(_unit(u), _unit(v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _flank_tangent(g, s, toward):
    """The straddling flank's unit tangent at station `s`, oriented toward `toward`."""
    h = 1.0e-7
    sample, eta = g["sample"], g["eta"]
    p1 = np.asarray(sample(np.asarray(s + h), np.asarray(eta)), float)
    p0 = np.asarray(sample(np.asarray(s - h), np.asarray(eta)), float)
    t = p1 - p0
    p = np.asarray(sample(np.asarray(s), np.asarray(eta)), float)
    return _unit(t if float(np.dot(t, np.asarray(toward, float) - p)) > 0 else -t)


def _circle_tangent(P, toward):
    t = np.array([-P[1], P[0]], float)
    return _unit(t if float(np.dot(t, np.asarray(toward, float) - P)) > 0 else -t)


def _arc_tangent(C, P, toward):
    r = np.asarray(P, float) - np.asarray(C, float)
    t = np.array([-r[1], r[0]], float)
    return _unit(t if float(np.dot(t, np.asarray(toward, float) - P)) > 0 else -t)


# ---------------------------------------------------------------------------
# THE MEASUREMENT THAT KILLS ROUTE 1 AS WRITTEN
# ---------------------------------------------------------------------------

def region_angles(g):
    """The three interior angles of PART 3's region `A - P_t - B`, in degrees.

    PART 3 named this region "the curvilinear triangle A - P_t - B" and proposed covering
    it with a dedicated block.  A fillet is TANGENT to both legs by definition, so at `A`
    the arc leaves along the flank and at `B` it leaves along the ring circle: both
    corners are CUSPS and the region is a sliver, not a triangle.

    `B` is exact -- both curves are circles and the tangency is the solve's own residual,
    so the measured angle is 0 to machine precision.  `A` is not quite: the flank is a
    spline rather than a straight leg, so the arc is tangent to it at a point where the
    flank's own curvature has already turned it by a few tenths of a degree.  That is the
    0.42-0.60 deg below, and it is a curvature term, not slack in the solve.

    The angle at `P_t` is the void -- the same quantity `make junction` prices as the
    corner a fillet has to fit into.  It is reported here so the three are read together:
    38 + 0.6 + 0 does not add up to a meshable block, and the 38 is the only one anybody
    had looked at.
    """
    P_t, A, B, C = g["P_t"], g["A"], g["B"], g["C"]
    at_B = _angle_deg(_circle_tangent(B, P_t), _arc_tangent(C, B, A))
    at_A = _angle_deg(_flank_tangent(g, g["s_A"], P_t), _arc_tangent(C, A, B))
    at_P = _angle_deg(_circle_tangent(P_t, B), _flank_tangent(g, g["s_end"], A))
    at_P_chord = _angle_deg(_circle_tangent(P_t, B), _flank_chord(g) - P_t)
    return {
        "at_B_deg": at_B, "at_A_deg": at_A, "at_P_t_deg": at_P,
        "at_P_t_chord_deg": at_P_chord,
        "chord_minus_tangent_deg": at_P_chord - at_P,
        "leg_flank_mm": float(np.linalg.norm(A - P_t)),
        "leg_ring_chord_mm": float(np.linalg.norm(B - P_t)),
        "arc_length_mm": float(g["R"] * _arc_sweep(C, A, B)),
        "is_cusp_at_B": bool(at_B < 1.0e-6),
        "is_cusp_at_A": bool(at_A < 1.0),
    }


def _flank_chord(g):
    """The spoke block's SECOND flank node, one station in from `P_t`.

    `make junction` measures the void at `P_t` from the mesh, and its `f_hat` is exactly
    this chord -- the reproduction is checked in `tests/test_fillet_block.py` against the
    committed `study_junction_agreement.json`, to the digit.  It is reported next to the
    analytic tangent because the flank is a SPLINE: over one `coarse` station the chord
    has already turned, and the two differ by 0.8 deg at the hub and 0.6 at the rim.

    WHICH ONE IS RIGHT DEPENDS ON THE QUESTION.  For "how much room does a fillet have",
    which is `make junction`'s question, the chord is 0.8 deg OPTIMISTIC and both `P_t`
    verdicts clear by 5-20x anyway, so no verdict there moves.  For "what angle does a
    block corner have", which is this file's question, the tangent is the one that
    decides -- a block corner is a limit, not a chord.  The `P_c` rows of `make junction`
    are unaffected either way: under `uncap` that corner's leg is a STRAIGHT continuation,
    so its chord and its tangent are the same direction exactly.
    """
    cfg = g["cfg"]
    ds = (g["s_rim"] - g["s_hub"]) / (cfg.nn(cfg.n_span) - 1)
    step = ds if g["s_end"] == g["s_hub"] else -ds
    return np.asarray(g["sample"](np.asarray(g["s_end"] + step),
                                  np.asarray(g["eta"])), float)


def _arc_sweep(C, P, Q):
    u = np.asarray(P, float) - np.asarray(C, float)
    v = np.asarray(Q, float) - np.asarray(C, float)
    return abs(math.atan2(u[0] * v[1] - u[1] * v[0], float(np.dot(u, v))))


# ---------------------------------------------------------------------------
# THE MEASUREMENT THAT KILLS ROUTE 2
# ---------------------------------------------------------------------------

def winslow(grid, n_iter=WINSLOW_ITERS, omega=WINSLOW_OMEGA):
    """An elliptic (Winslow) interior solve on an [ni, nj, 2] block, boundary held.

    This is route 2's technique, applied as favourably as possible: it is what "a
    generated block" means, and every candidate here is offered it.  The boundary is held
    fixed because that is what "generated interior" means -- a scheme that moved the
    boundary would be a different construction, not a smoother.
    """
    g = np.array(grid, float)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return g
    for _ in range(n_iter):
        xi = 0.5 * (g[2:, 1:-1] - g[:-2, 1:-1])
        et = 0.5 * (g[1:-1, 2:] - g[1:-1, :-2])
        a = np.sum(et * et, axis=-1)[..., None]
        b = np.sum(xi * et, axis=-1)[..., None]
        c = np.sum(xi * xi, axis=-1)[..., None]
        cross = 0.25 * (g[2:, 2:] - g[2:, :-2] - g[:-2, 2:] + g[:-2, :-2])
        new = (a * (g[2:, 1:-1] + g[:-2, 1:-1])
               + c * (g[1:-1, 2:] + g[1:-1, :-2])
               - 2.0 * b * cross) / (2.0 * (a + c) + 1.0e-300)
        g[1:-1, 1:-1] = (1.0 - omega) * g[1:-1, 1:-1] + omega * new
    return g


def moved_corner(genes, cfg, junction, R):
    """The shipped `fillet=` spoke block's corner at `B`, and whether it is a BOUNDARY
    quantity.

    This is PART 3's "corner interior angle collapses from ~89 deg to 3.60 / 8.52 and the
    cross product changes sign", re-measured on the CURRENT default rather than quoted
    from the capped mesh it was taken on (PART 7's standing lesson).

    The second half is the part that decides route 2.  The angle is carried by three
    nodes: the corner `(0, j_f)`, its neighbour along the flank edge `(1, j_f)`, and its
    neighbour along the end cross-section `(0, j_f-+1)`.  All three are on the block's
    BOUNDARY.  So the reported `winslow_max_boundary_shift_mm` is exactly 0 and
    `angle_after_winslow_deg` is bit-identical to `angle_deg` -- not approximately, and
    not as a matter of how good the smoother is.
    """
    fillet = (R, 0.0) if junction == "hub" else (0.0, R)
    g = np.asarray(WW.sector_blocks(genes, cfg, fillet=fillet,
                                    fillet_blocking="spoke")["spoke"], float)
    g0 = np.asarray(WW.sector_blocks(genes, cfg, fillet=(0.0, 0.0),
                                    fillet_blocking="spoke")["spoke"], float)
    row = 0 if junction == "hub" else -1
    nj = g.shape[1]
    moved = [j for j in (0, nj - 1) if np.linalg.norm(g[row, j] - g0[row, j]) > 1e-9]
    if not moved:
        return None
    j_f = moved[0]
    step_i = 1 if junction == "hub" else -1
    step_j = 1 if j_f == 0 else -1
    corner = g[row, j_f]
    along_flank = g[row + step_i, j_f]
    along_cross = g[row, j_f + step_j]
    ang = _angle_deg(along_flank - corner, along_cross - corner)

    # The node angle above is what PART 3 reported and is resolution-dependent -- it is
    # measured between the corner and its two NEIGHBOURS.  The geometric one is not: it
    # is the angle between the two boundary CURVES, the fillet arc and the straight end
    # cross-section, and it is the quantity a generated interior would have to move.
    geo = junction_geometry(genes, cfg, junction, R)
    tangent_angle = None
    if geo["tangency"]:
        sample = geo["sample"]
        far0 = np.asarray(sample(np.asarray(geo["s_end"]),
                                 np.asarray(-geo["eta"])), float)
        tangent_angle = _angle_deg(_arc_tangent(geo["C"], geo["B"], geo["A"]),
                                   far0 - geo["B"])

    sm = winslow(g)
    bnd = np.zeros(g.shape[:2], bool)
    bnd[0, :] = bnd[-1, :] = bnd[:, 0] = bnd[:, -1] = True
    shift = float(np.abs(sm[bnd] - g[bnd]).max())
    ang_sm = _angle_deg(sm[row + step_i, j_f] - sm[row, j_f],
                        sm[row, j_f + step_j] - sm[row, j_f])
    return {
        "angle_deg": ang,
        "tangent_angle_deg": tangent_angle,
        "unfilleted_angle_deg": _angle_deg(g0[row + step_i, j_f] - g0[row, j_f],
                                           g0[row, j_f + step_j] - g0[row, j_f]),
        "end_cross_section_mm": float(np.linalg.norm(g[row, -1] - g[row, 0])),
        "unfilleted_end_cross_section_mm": float(np.linalg.norm(g0[row, -1]
                                                               - g0[row, 0])),
        "winslow_max_boundary_shift_mm": shift,
        "angle_after_winslow_deg": ang_sm,
        "angle_is_a_boundary_quantity": bool(shift == 0.0 and ang_sm == ang),
    }


# ---------------------------------------------------------------------------
# THE SPOKE, WITH THE FILLET TAKEN OUT OF IT
# ---------------------------------------------------------------------------

def trimmed_spoke(genes, cfg, R_hub, R_rim):
    """The spoke block ended at the TANGENT stations instead of at the ring crossings.

    This is not a new construction: it is `sector_blocks`' own unfilleted spoke over a
    shorter station range, so at `R_hub = R_rim = 0` it is the default block to the bit.
    What it changes is OWNERSHIP -- the material between `s_end` and `s_A` stops being
    the spoke's and becomes whatever block carries the fillet.

    Returns `None` if either tangency solve refuses, or if the two tangent stations cross
    (a fillet longer than the spoke), which is a geometric limit rather than a mesh one.
    """
    cfgo = WW.get_config(cfg)
    span = WW.HUB_RIM_SPAN_MM
    orientation = WW.flank_orientation(genes, cfgo, span_mm=span)
    rim_inner = WW.rim_inner_radius(span)
    sample, s_dense = WW.global_sampler(genes, cfgo, span_mm=span)
    s_hub, s_rim = WW.junction_stations(sample, s_dense, orientation, rim_inner)
    lo, hi = float(s_hub), float(s_rim)
    for junction, R in (("hub", R_hub), ("rim", R_rim)):
        if R <= 0.0:
            continue
        g = junction_geometry(genes, cfg, junction, R)
        if not g["tangency"]:
            return None
        if junction == "hub":
            lo = g["s_A"]
        else:
            hi = g["s_A"]
    if not lo < hi:
        return None
    n_sp, n_th = cfgo.nn(cfgo.n_span), cfgo.nn(cfgo.n_thick)
    s_grid = np.linspace(lo, hi, n_sp)
    eta_grid = np.linspace(-1.0, 1.0, n_th)
    return np.asarray(sample(s_grid[:, None], eta_grid[None, :]), float)


# ---------------------------------------------------------------------------
# THE CANDIDATE BLOCKS
# ---------------------------------------------------------------------------

def cut_depth_mm(g, wall_mm):
    """How far the cut at `B` reaches past the ring circle, in mm.

    Half the wall, capped at half of what the ring has to give.  Half the wall because
    that is the length scale the block's other end is built on; capped because the cut
    lands INSIDE the collar (5 mm deep) at the hub and inside the band (1.1-1.5 mm) at
    the rim, and a cut that reaches through either would be a hole rather than a notch.
    """
    return float(min(0.5 * wall_mm, 0.5 * g["depth_available_mm"]))


def candidate_grown_junction(g, n_long, n_th):
    """The junction block GROWN so its ring edge runs on past `P_t` and onto the fillet.

        j0 = [fillet arc A -> B] + [ring arc B -> Q]        (G1 at `B`)
        i0 = the spoke's end cross-section at `s_A`
        j1 = the far flank from `s_A` to the ring
        i1 = the uncap edge `Q -> far_end`

    The obvious cheap route: it needs no new block and no new seam node count, because
    every ring block already derives its split angle from the spoke's end row.  `B` is an
    interior point of `j0` and the composite is smooth there, so nothing about the edge
    looks wrong.  What is wrong is the REGION: `j0` doubles back at `B` and the material
    between its two branches -- the cusp sliver -- has zero thickness there.
    """
    sample, eta, C = g["sample"], g["eta"], g["C"]
    A, B, Q, far_end = g["A"], g["B"], g["Q"], g["far_end"]
    th_B = math.atan2(B[1], B[0])
    th_q = math.atan2(Q[1], Q[0])
    L_arc = g["R"] * _arc_sweep(C, A, B)
    L_ring = g["ring_r"] * abs(th_q - th_B)
    n_a = max(2, int(round((n_long - 1) * L_arc / (L_arc + L_ring))) + 1)
    n_r = n_long - n_a + 1
    if n_r < 2:
        return None
    j0 = np.concatenate([np.asarray(WW._arc_between(C, A, B, n_a), float)[:-1],
                         np.asarray(WW.arc_points(g["ring_r"], th_B, th_q, n_r), float)])
    s_f = np.linspace(g["s_A"], g["s_ring"], n_long)
    j1 = np.asarray(sample(s_f, np.zeros(n_long) - eta), float)
    i0 = _cross_section(g, g["s_A"], n_th, A)
    i1 = WW._lerp_points(Q, far_end, n_th, np)
    return WW.coons_patch(j0, j1, i0, i1, xp=np)


def candidate_pre_fillet_surfaces(g, n_long, n_th):
    """PART 3's region, closed into a quad by the two end cross-sections.

        j0 = [ring circle P_t -> B] + [fillet arc B -> A]   (G1 at `B`)
        i0 = the end cross-section at `s_end`               (seams to the junction block)
        i1 = the end cross-section at `s_A`                 (seams to the trimmed spoke)
        j1 = the far flank from `s_end` to `s_A`

    This is the nearest thing to "a dedicated fillet block with its own seam entries"
    that is a quad at all: PART 3's triangle plus the slab of spoke between the two cross
    sections, which turns the 0 deg corner at `P_t`-side into an ordinary one.  It keeps
    both pre-fillet surfaces as block edges, and that is what it dies of -- same cusp,
    same place.
    """
    sample, eta, C = g["sample"], g["eta"], g["C"]
    P_t, A, B = g["P_t"], g["A"], g["B"]
    th_t = math.atan2(P_t[1], P_t[0])
    th_B = math.atan2(B[1], B[0])
    L_ring = g["ring_r"] * abs(th_B - th_t)
    L_arc = g["R"] * _arc_sweep(C, A, B)
    n_r = max(2, int(round((n_long - 1) * L_ring / (L_ring + L_arc))) + 1)
    n_a = n_long - n_r + 1
    if n_a < 2:
        return None
    j0 = np.concatenate([np.asarray(WW.arc_points(g["ring_r"], th_t, th_B, n_r),
                                    float)[:-1],
                         np.asarray(WW._arc_between(C, B, A, n_a), float)])
    s_f = np.linspace(g["s_end"], g["s_A"], n_long)
    j1 = np.asarray(sample(s_f, np.zeros(n_long) - eta), float)
    i0 = _cross_section(g, g["s_end"], n_th, P_t)
    i1 = _cross_section(g, g["s_A"], n_th, A)
    return WW.coons_patch(j0, j1, i0, i1, xp=np)


def candidate_boundary_layer(g, n_long, n_th, depth=None):
    """The block a structured mesher builds along a fillet: corners OFF both tangencies.

        j0 = the fillet arc `A -> B`                       the free surface
        j1 = that arc offset into the material             wall at `A` -> `depth` at `B`
        i0 = the spoke's end cross-section at `s_A`        cuts across the flank AT `A`
        i1 = the radial cut `B -> B''`                     cuts across the ring circle

    The offset is taken along the arc's OWN outward normal (`arc - C`), so `j1` is a
    concentric arc of radius `R + w` and can never cusp however small `R` is -- offsetting
    the other way, toward the centre, is what folds a fillet's inner curve.  `w` runs from
    the wall thickness at `A`, so `j1` meets the end cross-section exactly, down to
    `depth` at `B`.

    Both cusps are now INTERIOR to an edge rather than sitting on a corner, which is the
    whole of why this one meshes and the two above do not.
    """
    sample, eta, C = g["sample"], g["eta"], g["C"]
    A, B = g["A"], g["B"]
    i0 = _cross_section(g, g["s_A"], n_th, A)
    far_sA = i0[-1]
    wall = float(np.linalg.norm(far_sA - A))
    d = cut_depth_mm(g, wall) if depth is None else float(depth)
    j0 = np.asarray(WW._arc_between(C, A, B, n_long), float)
    nrm = j0 - np.asarray(C, float)[None, :]
    nrm = nrm / np.linalg.norm(nrm, axis=1)[:, None]
    j1 = j0 + np.linspace(wall, d, n_long)[:, None] * nrm
    j1[0] = far_sA                     # exact corner; the two normals differ by the
    #                                    flank's own turn between `A` and the centreline
    i1 = WW._lerp_points(B, j1[-1], n_th, np)
    return WW.coons_patch(j0, j1, i0, i1, xp=np)


def _cross_section(g, s, n_th, first):
    """The spoke's end cross-section at station `s`, running straddling flank -> far.

    Its first node is replaced by the exact tangent point so the Coons corner check is
    met to the bit rather than to the sampler's round-off; at `s = s_A` the two agree to
    ~1e-13 mm anyway, which the controls report.
    """
    eta = g["eta"]
    row = np.asarray(g["sample"](np.full(n_th, float(s)),
                                 np.linspace(eta, -eta, n_th)), float)
    return np.concatenate([np.asarray(first, float)[None, :], row[1:]], axis=0)


CANDIDATE_FN = {"grown_junction": candidate_grown_junction,
                "pre_fillet_surfaces": candidate_pre_fillet_surfaces,
                "boundary_layer": candidate_boundary_layer}


# ---------------------------------------------------------------------------
# THE VERDICT ON ONE BLOCK
# ---------------------------------------------------------------------------

def block_quality(grid):
    """Mixed-sign cells, scaled Jacobian and Gauss `det J` for one candidate block.

    `cell_verdict` and `gauss_verdict` are imported from `study_fillet_fold` rather than
    re-written: they ARE PART 6's criteria A and C, and a second copy is how two files
    end up reporting the same word for two different measurements -- which is the exact
    failure PART 6 existed to clean up.

    The scaled Jacobian is sign-normalised by its own median for the same reason
    `gauss_verdict` normalises `det J`: a block indexed (theta, r) is left-handed in
    physical space and `build_wheel` flips whole blocks later, so the question here is
    whether any corner disagrees with its own block.
    """
    g = np.asarray(grid, float)
    a, b, c, d = g[:-1, :-1], g[1:, :-1], g[1:, 1:], g[:-1, 1:]

    def cr(p, q, r):
        return ((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1])
                - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))

    sj = []
    for p, q, r in ((a, b, d), (b, c, a), (c, d, b), (d, a, c)):
        n1 = np.linalg.norm(q - p, axis=-1)
        n2 = np.linalg.norm(r - p, axis=-1)
        sj.append(cr(p, q, r) / np.maximum(n1 * n2, 1.0e-300))
    S = np.stack(sj, axis=-1)
    S = S if np.median(S) > 0 else -S
    cells = ff.cell_verdict(g)
    gauss = ff.gauss_verdict(g)
    return {"shape": [int(g.shape[0]), int(g.shape[1])],
            "mixed_sign_cells": cells["mixed_sign_cells"],
            "min_scaled_jacobian": float(S.min()),
            "non_positive_gauss_elements": gauss["non_positive_elements"],
            "min_det_j": gauss["min_det_j"],
            "valid": bool(cells["mixed_sign_cells"] == 0
                          and gauss["non_positive_elements"] == 0)}


# ---------------------------------------------------------------------------
# THE SWEEPS
# ---------------------------------------------------------------------------

def sweep_region(genes, cfg, junctions, radii):
    """The cusp measurement, per junction per radius.  Geometry only -- no mesh."""
    out = {}
    for junction in junctions:
        rows = []
        for R in radii:
            g = junction_geometry(genes, cfg, junction, R)
            if not g["tangency"]:
                rows.append({"radius_mm": float(R), "tangency": False})
                continue
            row = {"radius_mm": float(R), "tangency": True}
            row.update(region_angles(g))
            rows.append(row)
        out[junction] = rows
    return out


def sweep_candidates(genes, cfg, junctions, radii, refinements):
    """Every candidate block, at every radius, at every refinement, plus Winslow."""
    cfgo = WW.get_config(cfg)
    n_th, n_weld = cfgo.nn(cfgo.n_thick), cfgo.nn(cfgo.n_weld)
    out = {}
    for junction in junctions:
        rows = []
        for R in radii:
            g = junction_geometry(genes, cfg, junction, R)
            row = {"radius_mm": float(R), "tangency": bool(g["tangency"])}
            if g["tangency"]:
                row["blocks"] = {}
                for name in CANDIDATES:
                    per = {}
                    for m in refinements:
                        n_long = (n_weld - 1) * m + 1
                        try:
                            grid = CANDIDATE_FN[name](g, n_long, n_th)
                        except ValueError as exc:
                            per[str(m)] = {"built": False, "message": str(exc)[:120]}
                            continue
                        if grid is None:
                            per[str(m)] = {"built": False, "message": "not constructible"}
                            continue
                        q = block_quality(grid)
                        q["built"] = True
                        q["winslow"] = block_quality(winslow(grid))
                        per[str(m)] = q
                    row["blocks"][name] = per
                row["cut_depth_mm"] = cut_depth_mm(
                    g, float(np.linalg.norm(_cross_section(g, g["s_A"], n_th, g["A"])[-1]
                                            - g["A"])))
            rows.append(row)
        out[junction] = rows
    return out


def sweep_spoke_trim(genes, cfg, radii):
    """The trimmed spoke over the sweep, at the OTHER junction's shipped radius.

    Each column moves one radius and holds the other at what the genome ships, so the row
    answers "can the spoke carry this fillet in the wheel as designed" rather than "in a
    wheel with one fillet deleted", which is the question `make fillet` already asked.
    """
    R_hub_ship, R_rim_ship = float(genes[12]), float(genes[13])
    out = {}
    for junction in ("hub", "rim"):
        rows = []
        for R in radii:
            R_hub = R if junction == "hub" else R_hub_ship
            R_rim = R if junction == "rim" else R_rim_ship
            grid = trimmed_spoke(genes, cfg, R_hub, R_rim)
            if grid is None:
                rows.append({"radius_mm": float(R), "built": False})
                continue
            q = block_quality(grid)
            q.update({"radius_mm": float(R), "built": True})
            rows.append(q)
        out[junction] = rows
    return out


# ---------------------------------------------------------------------------
# THE PRICE OF THE ONE THAT WORKS
# ---------------------------------------------------------------------------

def price(genes, cfg, junctions):
    """What the boundary-layer block asks of its neighbours, measured at the shipped R.

    The cut at `B` reaches PAST the ring circle by construction, so the ring circle stops
    being the junction/collar interface over the fillet's footprint.  This section states
    how far past, over how much arc, and how much of the collar/band depth that is -- the
    numbers a blocking has to be designed against, rather than "it needs a notch".
    """
    cfgo = WW.get_config(cfg)
    n_th = cfgo.nn(cfgo.n_thick)
    out = {}
    for junction in junctions:
        R = float(genes[12] if junction == "hub" else genes[13])
        g = junction_geometry(genes, cfg, junction, R)
        if not g["tangency"]:
            out[junction] = {"tangency": False}
            continue
        A, B, P_t = g["A"], g["B"], g["P_t"]
        wall = float(np.linalg.norm(_cross_section(g, g["s_A"], n_th, A)[-1] - A))
        d = cut_depth_mm(g, wall)
        th_t = math.degrees(math.atan2(P_t[1], P_t[0]))
        th_B = math.degrees(math.atan2(B[1], B[0]))
        # Where the block's inner edge actually leaves the ring circle: the notch is the
        # part of `j1` on the far side of it, and its angular extent is what the collar
        # (hub) or band (rim) block has to be re-cut over.
        j1 = candidate_boundary_layer(g, 401, n_th)[:, -1, :]
        rad = np.linalg.norm(j1, axis=1)
        past = ((rad - g["ring_r"]) * g["void_sign"]) < 0.0
        th_j1 = np.degrees(np.arctan2(j1[:, 1], j1[:, 0]))
        notch_deg = float(th_j1[past].max() - th_j1[past].min()) if past.any() else 0.0
        out[junction] = {
            "radius_mm": R,
            "cut_depth_mm": d,
            "ring_depth_available_mm": g["depth_available_mm"],
            "cut_depth_fraction_of_ring": d / g["depth_available_mm"],
            "footprint_deg": abs(th_B - th_t),
            "notch_deg": notch_deg,
            "notch_fraction_of_footprint": notch_deg / max(abs(th_B - th_t), 1e-12),
            "footprint_fraction_of_sector": abs(th_B - th_t) / WW.SECTOR_DEG,
            "spoke_stations_given_up": abs(g["s_A"] - g["s_end"]) / (
                (g["s_rim"] - g["s_hub"]) / (cfgo.nn(cfgo.n_span) - 1)),
            "wall_at_s_A_mm": wall,
        }
    return out


# ---------------------------------------------------------------------------
# THE WHOLE FILLETED SECTOR: EVERY BLOCK, EVERY SEAM
# ---------------------------------------------------------------------------
#
# PART 9 measured ONE block.  A block that meshes is not a mesh: the fillet block's
# inner edge crosses the ring circle, so the block it seams to changes half way along
# that edge, and the ring blocks it lands in have to close as quads with node counts
# that agree.  This section builds the WHOLE sector -- all eleven blocks and all
# fourteen seams -- so that "what does it cost" is a measurement rather than a list of
# worries.  Nothing here is wired into `build_wheel`; it is the same discipline as the
# rest of this file, geometry and Jacobians only.
#
# THE BLOCKING, AND WHY IT IS THE ONE IT IS
# ----------------------------------------
# Per junction the fillet adds TWO blocks and re-cuts the three it touches:
#
#     <j>_fillet_a   the boundary layer along the arc, ABOVE the ring circle
#     <j>_fillet_b   the wedge that crosses it, from the arc down to the ring's FAR
#                    boundary (the hub bore, the rim's outer surface)
#     <j>_junction   unchanged in shape; its end cross-section is replaced by
#                    `fillet_a`'s inner edge and its arc now starts at `N`
#     <j>_ring_weld  the ring's weld block, ending at `N` instead of at `P_t`
#     <j>_ring_free  the ring's free block, starting at the CUT rather than at `P_t`
#
# `N` is where the fillet block's inner edge crosses the ring circle.  It is the whole
# reason for the split: above `N` that edge's partner is the junction block, below it
# the ring, and one edge with two partners is a PARTIAL-EDGE SEAM.  This tree has never
# had one -- `_seam_table`'s docstring calls whole-edge single ownership "the whole
# safety net" -- so the block is split at `N` instead, at the cost of one extra block
# per junction.
#
# WHY THE CUT GOES ALL THE WAY THROUGH THE RING, WHICH IS NOT A PREFERENCE
# -----------------------------------------------------------------------
# PART 9's block stopped at a shallow depth `d` inside the collar.  That cannot be
# closed with whole-edge quad seams, and the reason is mechanical rather than a matter
# of trying harder.  A cut that stops at depth `d` puts the ring's free block's left
# edge -- which spans the ring's whole depth -- against TWO partners, the notch below
# `B''` and the fillet block above it.  Splitting the free block at `|B''|` to fix that
# leaves its own right edge, at the sector boundary, against two partners again, so the
# split propagates round the ring; and the block it propagates into, the sliver between
# the fillet block's inner edge and a concentric circle, is a TRIANGLE, because the
# inner edge is a concentric offset of an arc that is TANGENT to the ring circle and is
# therefore tangent to every circle concentric with it.  Measured: that landing angle is
# 12.86 deg at the hub and 4.38 at the rim (`landing` in the report).  Taking the cut to
# the ring's far boundary instead splits the ring into exactly two quads and terminates.
#
# WHAT THAT FORCES ON THE NODE COUNTS
# -----------------------------------
# The cut carries `n_thick` nodes.  It is the ring free block's left edge, so that
# block is [n_free, n_thick]; its right edge is then `n_thick` too, and that edge is the
# next sector's weld block's left edge, so the weld is [n_weld, n_thick] as well.
# **`n_collar_r` and `n_rim_r` are not used by the filleted blocking** -- the ring's
# radial count becomes `n_thick` (7 -> 9 at `coarse`, 9 -> 13 at `medium`).  That is the
# node-count coupling; it is reported rather than hidden, and it is why `fillet=None`
# has to keep its own path.
#
# THE INNER EDGE'S SHAPE IS THE ONE FREE PARAMETER, AND IT IS MEASURED
# -------------------------------------------------------------------
# The inner edge is the arc offset by `w(u)`, a cubic Hermite in the arc parameter, and
# then a RADIAL dive from `N` to the far boundary.  Two constants set it:
#
#   LAYER_ENTRY_SLOPE  `w'(0)`, as a multiple of `(R + wall) * sweep`.  NEGATIVE on
#                      purpose.  Zero means the layer leaves the end cross-section
#                      tangent to the FAR FLANK -- which is the junction block's own
#                      top edge -- and a zero-degree corner in the junction block is
#                      what that costs: min scaled Jacobian 0.012 at the hub, measured.
#                      Three blocks meet at that node and 180 degrees has to be shared.
#   LAYER_END_OFFSET   `w(1)`, as a multiple of the wall.
#
# The RULE that picks them, so that a re-run picks the same pair for the same reason:
# take the argmax of the WORST block's min scaled Jacobian over the whole box, on the
# published grid.  The surface is a RIDGE, not a peak -- everything from entry -0.35 to
# -0.60 with end 1.4-1.8 sits within 0.02 of the maximum -- so the argmax is a choice
# ON a plateau rather than a tuned point, and the report prints the whole grid so that
# is visible instead of asserted.  Off the ridge it falls away fast and in a way that
# names its own mechanism: end >= 2.4 collapses (0.07-0.29) because the layer thickens
# past what the wall has to give, and entry >= -0.20 collapses (0.19-0.26) because the
# junction block's corner is being squeezed shut.
#
# The dive is RADIAL because an offset whose `w` grows to the ring's full depth is a
# spiral of radius `R + w` about the arc's centre: for `R` small and `w` large it swings
# clean out of the material, and at the gene box's own floor `R_hub = 0.4` it folds the
# weld block.  Measured, both ways round, before this construction was chosen.

# The construction itself now lives in `wheel_wheel` -- `filleted_sector`,
# `_filleted_sector_blocks` and `_seam_table_filleted` -- because STEP 1b wired it into
# `build_wheel` and a second copy here is exactly the drift this file's docstring is
# about.  What stays here is the MEASUREMENT: the verdict on one cell, the sweeps, the
# controls, and the two constants' re-derivation, which needs the profile open and is
# why `wheel_wheel.filleted_sector` exposes `entry` and `end` at all.
LAYER_ENTRY_SLOPE = WW.FILLET_LAYER_ENTRY_SLOPE
LAYER_END_OFFSET = WW.FILLET_LAYER_END_OFFSET
SECTOR_BLOCK_ORDER = WW.FILLETED_BLOCK_ORDER
SECTOR_BLOCK_REGION = WW.FILLETED_BLOCK_REGION
SEAM_TOL_MM = 1.0e-9


def filleted_sector(genes, cfg, R_hub, R_rim,
                    entry=LAYER_ENTRY_SLOPE, end=LAYER_END_OFFSET):
    """`wheel_wheel.filleted_sector`, with its refusal turned back into a verdict.

    The module RAISES when the geometry refuses, because for `build_wheel` a genome it
    cannot block is the same class of event as a rim band of non-positive thickness.
    This file sweeps refusals on purpose -- how often and why is half of what it
    reports -- so it catches the reason and hands it back as data.
    """
    try:
        blocks = WW.filleted_sector(genes, cfg, fillet=(R_hub, R_rim),
                                    entry=entry, end=end)
    except (ValueError, NotImplementedError) as exc:
        return None, {"why": str(exc).split("  See ")[0]}
    dirn = blocks.pop("_dirn")
    applied = blocks.pop("_applied", None)
    blocks.pop("_thetas", None)
    # Every `_`-prefixed key, not the two this file happens to know about.  `_applied`
    # arrived with PLAN §57's clamp and broke four call sites that popped by name; the
    # prefix is the convention the rest of the tree already filters on.
    for k in [k for k in blocks if k.startswith("_")]:
        blocks.pop(k)
    return blocks, {"why": None, "dirn": dirn, "applied": applied}


def _side(block, side):
    b = np.asarray(block, float)
    return {"i0": b[0, :, :], "i1": b[-1, :, :],
            "j0": b[:, 0, :], "j1": b[:, -1, :]}[side]


def _rotate(P, k):
    a = k * math.radians(WW.SECTOR_DEG)
    c, s = math.cos(a), math.sin(a)
    return np.asarray(P, float) @ np.array([[c, s], [-s, c]])


def sector_seams(blocks, orientation, dirn):
    """Every seam's node count agreement and worst coordinate mismatch, in mm.

    A seam CLOSES when the two edges carry the same number of nodes AND those nodes
    coincide.  Both halves are reported: a count mismatch is a blocking error and a
    coordinate mismatch is a construction error, and they are not the same bug.
    """
    out = []
    for a, sa, b, sb, dk, rev in WW._seam_table_filleted(orientation, dirn):
        ea = _side(blocks[a], sa)
        eb = _rotate(_side(blocks[b], sb), dk)
        row = {"a": a, "side_a": sa, "b": b, "side_b": sb, "dk": dk, "reverse": rev,
               "n_a": int(ea.shape[0]), "n_b": int(eb.shape[0])}
        if ea.shape != eb.shape:
            row.update({"counts_agree": False, "max_gap_mm": None, "closes": False})
        else:
            gap = float(np.abs(ea - (eb[::-1] if rev else eb)).max())
            row.update({"counts_agree": True, "max_gap_mm": gap,
                        "closes": gap < SEAM_TOL_MM})
        out.append(row)
    return out


SEAM_TOL_MM = 1.0e-9


def sector_verdict(genes, cfg, R_hub, R_rim, entry=LAYER_ENTRY_SLOPE,
                   end=LAYER_END_OFFSET):
    """One (config, R_hub, R_rim) cell: every block's validity and every seam's gap."""
    blocks, info = filleted_sector(genes, cfg, R_hub, R_rim, entry, end)
    if blocks is None:
        return {"built": False, "why": info["why"], "R_hub_mm": float(R_hub),
                "R_rim_mm": float(R_rim)}
    orientation = WW.flank_orientation(genes, WW.get_config(cfg),
                                       span_mm=WW.HUB_RIM_SPAN_MM)
    per = {k: block_quality(v) for k, v in blocks.items()}
    dirn = info["dirn"]
    seams = sector_seams(blocks, orientation, dirn)
    return {"built": True, "R_hub_mm": float(R_hub), "R_rim_mm": float(R_rim),
            "blocks": per, "dirn": {j: float(v) for j, v in dirn.items()},
            "orientation": [float(o) for o in orientation],
            "n_blocks": len(blocks),
            "min_scaled_jacobian": min(q["min_scaled_jacobian"] for q in per.values()),
            "worst_block": min(per, key=lambda k: per[k]["min_scaled_jacobian"]),
            "non_positive_gauss_elements": sum(q["non_positive_gauss_elements"]
                                               for q in per.values()),
            "mixed_sign_cells": sum(q["mixed_sign_cells"] for q in per.values()),
            "all_blocks_valid": all(q["valid"] for q in per.values()),
            "n_seams": len(seams),
            "seams_close": all(s["closes"] for s in seams),
            "max_seam_gap_mm": max(s["max_gap_mm"] or 0.0 for s in seams),
            "seams": seams}


def sector_control(genes, cfg):
    """The UNFILLETED sector, measured with the same instrument.

    Every number in this section is a difference from the seven-block mesh the tree
    ships, and a degradation is only readable against what it degrades from.
    """
    blocks = WW.sector_blocks(genes, cfg, fillet=None)
    per = {k: block_quality(np.asarray(v, float))
           for k, v in blocks.items() if k != "_thetas"}
    return {"n_blocks": len(per),
            "min_scaled_jacobian": min(q["min_scaled_jacobian"] for q in per.values()),
            "worst_block": min(per, key=lambda k: per[k]["min_scaled_jacobian"]),
            "all_valid": all(q["valid"] for q in per.values())}


def sector_curves(genes, cfg, junction, R,
                  entry=LAYER_ENTRY_SLOPE, end=LAYER_END_OFFSET):
    """`wheel_wheel._fillet_curves` at one junction, with the setup it needs done here.

    The module computes these inside `_filleted_sector_blocks`, which returns BLOCKS.
    This file measures the curves themselves -- where the inner edge crosses the ring
    circle, where the cut lands, how much free ring is left -- so it calls the same
    function with the same arguments rather than re-deriving any of it.
    """
    cfgo = WW.get_config(cfg)
    n_th = cfgo.nn(cfgo.n_thick)
    is_hub = junction == "hub"
    span = WW.HUB_RIM_SPAN_MM
    orientation = WW.flank_orientation(genes, cfgo, span_mm=span)
    rim_inner = WW.rim_inner_radius(span)
    sample, s_dense = WW.global_sampler(genes, cfgo, span_mm=span)
    s_hub, s_rim = WW.junction_stations(sample, s_dense, orientation, rim_inner)
    s_end, s_far = (s_hub, s_rim) if is_hub else (s_rim, s_hub)
    eta = 1.0 if float(orientation[0 if is_hub else 1]) > 0 else -1.0
    ring_r = WW.HUB_RADIUS_MM if is_hub else rim_inner
    s_ring = 0.0 if is_hub else 1.0
    blend = WW._uncap_blend(WW.UNCAP_DEFAULT, is_hub)
    Q = (np.asarray(sample(np.asarray(s_ring), np.asarray(0.0)), float) if blend is None
         else np.asarray(WW._uncap_corner(sample, s_ring, eta, ring_r, is_hub, blend, np),
                         float))
    c = WW._fillet_curves(sample, s_end, s_far, eta, ring_r,
                          far_boundary_radius(junction), R, n_th, Q, entry, end)
    return dict(c, r_far=far_boundary_radius(junction))


def far_boundary_radius(junction):
    """The far side of the ring the cut has to reach: the bore, or the tyre surface."""
    return WW.ring_far_radius(junction == "hub")


def sector_fit_limit(genes, cfg, junction):
    """The radius at which the fillet's tangent point reaches the NEXT sector's corner.

    A geometric limit of the SECTOR, not of the block: PART 9 measured the block itself
    clean out to 4.00 mm, and it still is -- what runs out first is the ring's free
    block, whose angular span this drives to zero.  Bisected rather than estimated,
    because it is the number a gene bound would have to be written against.

    THE BISECTION ITSELF NOW LIVES IN THE MODULE (PART 21).  `wheel_wheel` clamps to this
    limit on the `fillet=True` path, so a second copy here would be two implementations of
    one criterion drifting apart -- the failure this file's own docstrings are about.
    `sector_curves` stays, because the rest of this section measures the CURVES; only the
    root-find is delegated.  The module bisects 40 times against the 80 here, which moves
    the shipped genome's hub limit from 3.1296998810584657 to 3.1296998810549430 -- 3.5e-12
    mm, eight orders under the 1e-4 these margins are quoted to.
    """
    def curves_at(R):
        c = sector_curves(genes, cfg, junction, R)
        return c if "built" in c else dict(c, built=False)

    R = WW._sector_fit_limit(curves_at)
    return {"limited": R is not None, "radius_mm": R}


# PART 10 FINDING 6's refusal, turned from a count into a number that predicts it.  Six of
# sixteen drawn genomes refused the blocking outright and every refusal was the same one --
# the hub fillet's tangent point swept past the next sector's corner at that genome's own
# `R_hub`.  That is `sector_fit_limit` arriving as a GENOME property rather than a radius
# one, so the margin against it is computable per genome, before any block is built.
#
# THE VALUE MOVED INTO THE MODULE AT PART 21, because `wheel_wheel` now clamps to it on the
# `fillet=True` path and a study-local copy would be a second number to keep in step.  Kept
# under this name so every reference in this file and its tests still reads.
SECTOR_FIT_CLAMP = WW.SECTOR_FIT_CLAMP


def sector_fit_margin(genes, cfg, junctions=DEFAULT_JUNCTIONS):
    """Each junction's radius against the radius its own sector has room for.

    `binds` is the prediction: a genome whose drawn radius exceeds its own limit has no
    free ring block to build and the blocking must refuse.  Reported as a MARGIN and not
    as a boolean, because a gate written on a boolean cannot say how close the rest of the
    box is to it -- and `sweep_genomes` finds margins from -0.48 mm to +5.90 mm on genomes
    that all pass the same feasibility filter.
    """
    out = {}
    for j in junctions:
        R = float(genes[12 if j == "hub" else 13])
        L = sector_fit_limit(genes, cfg, j)["radius_mm"]
        out[j] = {"R_mm": R, "limit_mm": L,
                  "margin_mm": None if L is None else float(L - R),
                  "binds": bool(L is not None and R > L)}
    return out


def clamped_radii(margins, factor=SECTOR_FIT_CLAMP):
    """The two radii, each pulled back to a fraction of the room its sector has.

    A FIX rather than a gate, and the two are different things this file must not blur.
    The gate is `binds` above: it costs nothing and loses the genome.  This keeps the
    genome and models a SMALLER fillet than its genes asked for, which is honest for an
    instrument sweeping the box and is honest for an optimizer only if the clamped value
    is what the objective is told it got.  `factor` exists because the limit is where the
    free ring block's span reaches ZERO, and a block of zero span is not a block.
    """
    return tuple(
        m["R_mm"] if m["limit_mm"] is None else min(m["R_mm"], factor * m["limit_mm"])
        for m in (margins["hub"], margins["rim"]))


def landing_angles(genes, cfg, junctions):
    """Why the shallow cut cannot close: the offset lands TANGENT to the ring's circles.

    PART 9's block stopped at depth `d` inside the ring.  Its inner edge is a concentric
    offset of an arc that is tangent to the ring circle, so it is tangent to the circle
    `d` inside it as well -- and a block cornered between the two is a sliver.  Measured
    at the shipped radii: the residual angle is the width profile's own slope and
    nothing else.
    """
    cfgo = WW.get_config(cfg)
    n_th = cfgo.nn(cfgo.n_thick)
    out = {}
    for junction in junctions:
        R = float(genes[12] if junction == "hub" else genes[13])
        g = junction_geometry(genes, cfg, junction, R)
        if not g["tangency"]:
            out[junction] = {"tangency": False}
            continue
        j1 = candidate_boundary_layer(g, 4001, n_th)[:, -1, :]
        wall = float(np.linalg.norm(_cross_section(g, g["s_A"], n_th, g["A"])[-1]
                                    - g["A"]))
        end, prev = j1[-1], j1[-2]
        circ = np.array([-end[1], end[0]], float)
        a = _angle_deg(end - prev, circ)
        out[junction] = {"cut_depth_mm": cut_depth_mm(g, wall),
                         "landing_angle_deg": min(a, 180.0 - a),
                         "sliver_scaled_jacobian": abs(math.sin(math.radians(
                             min(a, 180.0 - a))))}
    return out


def sweep_layer_profile(genes, cfg, entries, ends, box):
    """The two inner-edge constants, swept against the WORST block in the whole box.

    This is the evidence for `LAYER_ENTRY_SLOPE` and `LAYER_END_OFFSET`.  A constant
    that is only asserted is a constant that drifts; this one is re-derived every run,
    and the report carries the whole surface rather than the argmax, so a reader can see
    how wide the plateau is.
    """
    rows = []
    for entry in entries:
        for end in ends:
            worst, where, fails = 9.0, None, 0
            for R_hub, R_rim in box:
                v = sector_verdict(genes, cfg, R_hub, R_rim, entry, end)
                if not v["built"]:
                    fails += 1
                    continue
                if not v["all_blocks_valid"] or not v["seams_close"]:
                    fails += 1
                if v["min_scaled_jacobian"] < worst:
                    worst = v["min_scaled_jacobian"]
                    where = [v["R_hub_mm"], v["R_rim_mm"], v["worst_block"]]
            rows.append({"entry": float(entry), "end": float(end),
                         "worst_min_scaled_jacobian": (None if where is None
                                                       else float(worst)),
                         "worst_at": where, "cells_failed": fails})
    return rows


GENOME_PROFILE_ENTRIES = (-0.30, -0.45, -0.60, -0.70, -0.75, -0.80, -0.90)
GENOME_PROFILE_ENDS = (0.50, 0.60, 0.70, 0.80, 1.00, 1.30, 1.60)

# The argmax of `sweep_layer_profile_genomes`' own grid (FILLET_PLAN.md PART 13).
# MEASURED, NOT ADOPTED as `WW.FILLET_LAYER_ENTRY_SLOPE` / `FILLET_LAYER_END_OFFSET`:
# it clears `MIN_SJ_TARGET` for nine of the ten non-pathological genomes PART 13 drew
# (against four of ten at the shipped constant), but it costs the shipped genome the
# margin PART 12's deflection-convergence finding relied on -- see `WW._fillet_curves`'s
# docstring for the trade, measured both ways.
GENOME_ROBUST_ENTRY = -0.75
GENOME_ROBUST_END = 0.70

# AND THE CELL THAT SURVIVES WHEN THE CLIFF MARGIN IS A CONSTRAINT (PLAN §69, §80).
# MEASURED, NOT ADOPTED, exactly as the pair above is.  §68 declined `GENOME_ROBUST_*`
# for standing 0.056 from a hard refusal of the shipped genome; §69 rebuilt the candidate
# set with that distance as a constraint rather than as an afterthought and this is what
# came out -- box floor 0.2061 over fifteen genomes with none refused, cliff margin
# 0.1577 against `GENOME_ROBUST_*`'s 0.0564, and an increment ratio of 0.437 against
# `study_corner_singularity.SETTLING_RATIO` = 0.75, so it settles where §54's argmax does
# not.  Every one of those numbers is IN-SAMPLE on the sixteen-genome draw, which is what
# §80 ranked first: `sweep_sector_fit_clamp` carries it onto the held-out thirty-two so
# the pair is judged on genomes it was not fitted to.  See PLAN §80 for why the objection
# that stood against adopting it was withdrawn by inventory rather than by measurement.
MARGIN_ROBUST_ENTRY = -0.70
MARGIN_ROBUST_END = 0.90


def sweep_layer_profile_genomes(genes, cfg, genome_rows, entries=GENOME_PROFILE_ENTRIES,
                                ends=GENOME_PROFILE_ENDS, clamp=None, fold_gate=False):
    """PART 10's derivation, re-run against GENOMES rather than one radius box.

    `sweep_layer_profile` picked `LAYER_ENTRY_SLOPE`/`LAYER_END_OFFSET` as the argmax of
    the worst block over a radius box AT ONE GENOME -- the shipped one -- and FINDING 6
    already named the consequence before this file used it: `flank_orientation` is a
    property of the centreline, and a construction measured at the shipped genome alone
    has been measured on a quarter of the design space.  This sweeps the same two
    constants against the worst block over the GENOMES `sweep_genomes` already drew, each
    at its OWN drawn radii, plus the shipped genome at its own.

    One drawn genome is excluded and named rather than silently dropped: a genome whose
    worst block is the TRIMMED SPOKE fails at every `(entry, end)` in the grid, including
    `entry = 0`, because the spoke block samples the centreline directly and neither
    constant reaches it -- see `test_a_spoke_fold_genome_does_not_move_with_the_profile`.
    Folding it into the argmax would not move the argmax; every cell would report the same
    floor and the ridge underneath it would be invisible.  It is a real finding on its own
    and PART 13 gives it its own paragraph rather than burying it in this one.

    `clamp` AND `fold_gate` ARE WHAT PARTS 14 AND 15 MADE POSSIBLE, and they retire both
    of the apologies above.  Ten cells is what was left after six genomes refused their
    sector and one was excluded by hand.  §57's clamp makes the six build; §58's fold
    margin replaces the hand-exclusion with a number taken before the build -- and it
    removes the OTHER folded genome too, which the hand-exclusion never saw, because that
    one refused for an unrelated reason and so never reached the `worst_block` test.  At
    `clamp=SECTOR_FIT_CLAMP, fold_gate=True` the argmax is over fourteen drawn genomes
    plus the shipped one, every one of which builds and every one of which describes a
    part that exists.

    The default is left exactly as PART 13 ran it so that its table keeps reproducing.
    The re-derivation is a second call and the two sit side by side in the artifact.
    """
    if clamp is None and not fold_gate:
        cells = [(np.asarray(r["genes"], float), r["R_hub_mm"], r["R_rim_mm"])
                 for r in genome_rows if r.get("built") and r["worst_block"] != "spoke"]
    else:
        cells = []
        for r in genome_rows:
            if "fit" not in r or "fold" not in r:
                continue
            if fold_gate and r["fold"]["binds"]:
                continue
            if clamp is None:
                if not r.get("built") or r["worst_block"] == "spoke":
                    continue
                R_hub, R_rim = r["R_hub_mm"], r["R_rim_mm"]
            else:
                R_hub, R_rim = clamped_radii(r["fit"], clamp)
            cells.append((np.asarray(r["genes"], float), R_hub, R_rim))
    cells.append((np.asarray(genes, float), float(genes[12]), float(genes[13])))
    rows = []
    for entry in entries:
        for end in ends:
            worst, where, refused = 9.0, None, 0
            for vec, R_hub, R_rim in cells:
                v = sector_verdict(vec, cfg, R_hub, R_rim, entry, end)
                if not v["built"]:
                    refused += 1
                    continue
                if v["min_scaled_jacobian"] < worst:
                    worst = v["min_scaled_jacobian"]
                    where = [float(R_hub), float(R_rim), v["worst_block"]]
            rows.append({"entry": float(entry), "end": float(end),
                         "n_genomes": len(cells),
                         "worst_min_scaled_jacobian": (None if where is None
                                                       else float(worst)),
                         "worst_at": where, "refused": refused})
    return rows


# EVERY PROFILE THAT COULD BE THE TWO-OBJECTIVE ANSWER, so that a negative result is a
# negative over the whole candidate set rather than over a sample of it.  The criterion is
# not a rank and not a tuned cut-off: it is `MIN_SJ_TARGET`, the barrier `wheel_objective`
# actually enforces, plus "refuses no genome" for the reason `profile_argmax` gives.  A
# cell that fails either is not a candidate for a genome-robust default whatever else it
# does, and a cell that passes both has to be priced against the deflection band before
# the two-objective profile can be said not to exist.
#
# Written as a constant so `study_corner_singularity` can price each pair without reading
# this study's artifact -- one study's freshness must not become another's problem, which
# is what PART 7 was about.  `the_candidate_constant_matches_the_measured_surface`
# re-derives it from the grid every run, so it cannot go stale silently.
LAYER_PROFILE_CANDIDATES = ((-0.80, 0.70), (-0.80, 0.80), (-0.80, 1.00),
                            (-0.75, 0.60), (-0.75, 0.70), (-0.75, 0.80),
                            (-0.70, 0.50), (-0.70, 0.60), (-0.70, 0.70), (-0.70, 0.80),
                            (-0.60, 0.50), (-0.60, 0.60), (-0.60, 0.70), (-0.60, 0.80))


# AND THE FRONTIER, which is what keeps the ridge's answer from being an anecdote.  Every
# cell on the ridge has a steep entry, so pricing only those would show eight profiles
# failing the convergence band without showing WHERE the failure begins -- and `entry` is
# the obvious suspect, since it is the one the shipped pair differs from all eight in.
# This walks the whole entry ladder at two ends: the ridge's own (0.70) and the shipped
# one (1.60).  A pure cross product, so unlike the candidate set it cannot go stale.
# THE ANSWER, once the whole candidate set was priced rather than a sample of it: exactly
# one of the fourteen holds the deflection band, and it is the one with the LOWEST
# genome-box floor of the fourteen -- so every ranking rule that takes a top-k of the
# ridge drops it.  Named here because FILLET_PLAN PART 16 turns it into a promotion
# proposal and a proposal needs a symbol; NOT wired into `wheel_wheel`, which is still
# PART 10's pair.  See PART 16 for what adopting it would cost to re-derive.
TWO_OBJECTIVE_ENTRY = -0.80
TWO_OBJECTIVE_END = 1.00

LAYER_PROFILE_FRONTIER_ENDS = (0.70, 1.60)
LAYER_PROFILE_FRONTIER = tuple((float(e), float(n))
                               for n in LAYER_PROFILE_FRONTIER_ENDS
                               for e in GENOME_PROFILE_ENTRIES)

# AND THE REFINEMENT, because the winner above is a grid point of a sweep laid out for a
# different question -- the ends jump 0.80 -> 1.00 -> 1.30 and the entries -0.80 -> -0.90,
# so the band-holding, barrier-clearing region was located but not resolved.  This walks
# the neighbourhood at half the step in both variables.  Reported through the same two
# functions as the coarse grid, so the refinement is the same measurement and not a
# second instrument -- which is the mistake PART 6 caught this arc making once already.
LAYER_PROFILE_FINE_ENTRIES = (-0.70, -0.75, -0.80, -0.85, -0.90, -0.95)
LAYER_PROFILE_FINE_ENDS = (0.85, 0.90, 1.00, 1.10, 1.20, 1.30)
LAYER_PROFILE_FINE_CANDIDATES = ((-0.90, 1.10),
                                 (-0.85, 0.90), (-0.85, 1.00), (-0.85, 1.10),
                                 (-0.80, 0.85), (-0.80, 0.90), (-0.80, 1.00),
                                 (-0.75, 0.85), (-0.75, 0.90),
                                 (-0.70, 0.85), (-0.70, 0.90))


def profile_candidates(table, target=None):
    """Every cell that clears the barrier on all DRAWN GENOMES and refuses none of them.

    "ALL" IS THE DRAW, NOT THE DESIGN SPACE, and this docstring used to say "on all cells"
    without saying whose.  It is the source of the "clears the barrier on the whole box"
    wording in PLAN §59 and §69, and PLAN §72 measured what that box actually reaches: the
    uniform Latin hypercube puts about one genome above 35 degrees of arc span in sixty-four.
    A cell that clears every genome HERE is a cell no genome in `sweep_layer_profile_genomes`
    refused; it is not a cell proved safe over the gene box.

    Sorted by `(entry, end)` rather than by score, because this is a SET and its order
    should not move when a floor wobbles by 0.001.  Refusing cells are excluded for the
    reason `profile_argmax` gives: a cell that loses a genome is not a candidate for a
    genome-robust default, whatever floor it reports over the ones it kept.
    """
    target = MIN_SJ_TARGET if target is None else target
    ok = [r for r in table
          if r["worst_min_scaled_jacobian"] is not None and r["refused"] == 0
          and r["worst_min_scaled_jacobian"] > target]
    ok.sort(key=lambda r: (r["entry"], r["end"]))
    return tuple((float(r["entry"]), float(r["end"])) for r in ok)


# PART 20 / PLAN §68's CLIFF, TURNED FROM FOUR HAND BISECTIONS INTO A COLUMN.
#
# The best genome-box floor on either grid refuses ONE genome, and it is the SHIPPED one --
# `the layer's width profile reaches zero thickness` at the rim, a hard geometric loss of the
# wheel every published number in this project is measured at.  §68 bisected `entry` at four
# `end`s by hand and found every candidate this arc produced standing within 0.08 of that
# edge while the shipped pair stands 0.55 from it, and then wrote the sentence this constant
# exists to retire: *"the distance to that edge was never a column in any table."*
#
# The bracket is wide on the safe side and stops short of -2.0, which is far past any entry
# either grid visits.  Bisection for the same reason `sector_fit_limit` uses one: the
# refusal comes out of `_hermite`'s minimum over a sampled profile and has no closed form.
# THE BRACKET IS THE MODULE'S NOW, AND THE -2.0 IT USED TO BE WAS A DEFECT (PLAN §84).
#
# At -2.0 three of the thirty-two held-out genomes reported "builds across the whole
# bracket" and were read as having no layer-width edge AT ALL -- the safest case.  All
# three have edges, at -2.51, -2.30 and -2.12, and the sentinel was reporting the bracket's
# width rather than anything about the genome.  Bound to the module's constant rather than
# re-stated, so the two cannot drift the way §57's clamp factor nearly did before PART 21.
CLIFF_BRACKET = WW.LAYER_CLIFF_BRACKET
CLIFF_BISECTIONS = WW.LAYER_CLIFF_BISECTIONS
CLIFF_REASON = "width profile reaches zero"

# How far either side of the closed-form cliff the two confirming verdicts are taken.
# Large enough to clear the root-find's own resolution by many orders, small enough that
# nothing else can bind in between: the tightest cliff-to-next-refusal gap measured over
# the held-out draw is ~0.05 of entry, and this is a fiftieth of that.
CLIFF_PROBE = 1e-3

# `cliff_entry`'s OTHER `None`, and it is the opposite of a problem: the genome builds at
# every entry in the bracket, so it has no layer-width edge to stand back from.  Named as a
# constant because §78 needs to tell it apart from "bounded by something else", which is the
# `None` that means a measurement failed.
CLIFF_NO_EDGE = "builds across the whole bracket"

# AND IT IS NOT THE SAFEST CASE.  IT IS THIS BRACKET BEING TOO NARROW (PLAN §82).
#
# The name above says what the bisection OBSERVED and `sweep_cliff_clamped_profile` reads
# it as "this genome has no layer-width edge to project onto", falls back to a global
# constant, and counts the genome as one the rule does not harm.  Measured against
# `wheel_wheel.layer_cliff_entry`, whose bracket runs to -8.0: all three held-out genomes
# reported this way have edges, at -2.51, -2.30 and -2.12, just past the -2.0 below.
#
# THE FINDING SURVIVES IT, which is why the bracket is left where it is rather than
# widened in the same commit that adopts the rule: at the adopted factor the three
# genomes clear the barrier on the rule's own answer exactly as they do on the fallback
# -- 31 of 32 either way, worst J 0.1721 and median 0.3466 to four figures on both.  So
# this is a defect in what the sentinel MEANS and not in any number published off it,
# and widening the bracket is a change to re-date artifacts for on its own.
#
# It is the third sentinel in this arc to carry two meanings under one name -- see §78's
# `n_without_cliff` split, and `wheel_wheel._sector_fit_span`, which §82 records.

# PART 20's four hand bisections, at the precision they were published to.  Kept so the
# automated column is checked against the RECORD rather than against itself -- if the two
# disagree, one of them is wrong and this file should say so before anyone quotes either.
CLIFF_PUBLISHED = ((0.85, -0.845458), (1.00, -0.881143),
                   (1.10, -0.903400), (1.60, -1.001967))

# THE PER-GENOME LAYER PROFILE, PRICED THE WAY §57's CLAMP WAS.  PLAN §78.
#
# §74 closed the REFUSAL half of §48's scope note and left the BARRIER half open: with the
# clamp on, 16 of 16 drawn genomes build and only 8 clear `MIN_SJ_TARGET`.  The profile that
# does close it is `GENOME_ROBUST_*`, which buys 15 of 16 -- and §68 declined it because it
# leaves the SHIPPED genome about 0.07 from a hard refusal against the shipped profile's
# 0.5520.  Every candidate in that section was a GLOBAL pair, and a global pair has to be
# safe for the tightest genome in the box.
#
# `cliff_entry` makes the per-genome version measurable: the layer-width cliff is a property
# of the genome, computable before any block is built, exactly as `sector_fit_limit` is.  So
# each genome can take a share of ITS OWN room instead of every genome sharing one constant.
#
# THE FACTORS ARE SWEPT AND NOTHING IS ADOPTED.  A multiplicative factor leaves a margin of
# `(1 - f) * |cliff|`, which is a much tighter clearance than the shipped pair's 0.5520 at
# any `f` near 1 -- so the margin each factor leaves is REPORTED next to what it buys rather
# than assumed acceptable, and the band is swept because §68's objection was about margin
# and this is the number that has to answer it.
#
# AND THE BAND IS SWEPT TO ITS FLOOR, NOT TO A ROUND NUMBER (PLAN §81).  The first sweep
# stopped at 0.55 and §78 quoted the rule at 0.75, which is the INTERIOR of a band whose
# edge was never located: 0.85 down to 0.55 all clear the same 31 of 32 held out, while
# the margin the factor leaves and what it costs the shipped genome's convergence BOTH
# improve monotonically as it falls.  Two axes improving across a flat third means the
# operating point is the flat band's lower EDGE, and quoting any interior value is the
# same class of choice §81 rejected.  These four continue to where the entry is too
# shallow to buy anything, so the edge is bracketed by measurement rather than assumed.
CLIFF_PROFILE_FACTORS = (0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15)

# AND THE OPERATING POINT THE SWEEP LOCATES.  ADOPTED AT PLAN §82, and the value now
# lives in `wheel_wheel` as `FILLET_LAYER_CLIFF_FACTOR` -- this name is kept so every
# reference in this file and its tests still reads, exactly as `SECTOR_FIT_CLAMP` above
# was kept when PART 21 moved that one.  The rule it parameterises is
# `wheel_wheel.per_genome_layer_profile`, and the cliff it multiplies is a CLOSED FORM
# there rather than the thirty sector builds `cliff_entry` below spends: the two agree to
# 9.1e-10 over the held-out draw, which is the bisection's own resolution.
#
# The admissible set is two conditions and neither is invented here: clear `MIN_SJ_TARGET`
# on the held-out draw, and settle against `study_corner_singularity.SETTLING_RATIO` at
# the shipped genome.  0.95 and 0.85 fail the second (ratios 0.800 and 0.796); 0.35 and
# below fail the first (30, 25 and 16 of 32).  What is left is 0.75 / 0.65 / 0.55 / 0.45,
# all clearing the same 31 of 32, across which BOTH remaining axes improve monotonically
# as the factor falls -- margin 0.202 -> 0.444, cost +0.392% -> +0.106%.  So the operating
# point is the bottom of that band and not a value inside it, which is the whole of what
# §81 rejected about quoting 0.75.
#
# THE COST IS NOT MONOTONE BELOW THE BAND and that is not a law being broken: 0.35 / 0.25 /
# 0.15 read +0.189% / +0.132% / +0.116%, and every one of them is already excluded on the
# barrier.  The monotonicity is a statement about the admissible set, which is where it is
# used, and the ratio is a three-rung estimate that should not be read finer than that.
CLIFF_PROFILE_FACTOR = WW.FILLET_LAYER_CLIFF_FACTOR

# ONE `end`, AND THE REASON.  The cliff moves with `end`, so a per-genome entry rule has to
# name one.  `GENOME_ROBUST_END` is the choice because the whole point of the measurement is
# to face the per-genome rule off against the global pair that buys 15 of 16 -- holding `end`
# at that pair's own value makes the ENTRY rule the only difference between them.  A second
# `end` would double the bisection cost, which is the expensive half of this section.
CLIFF_PROFILE_END = GENOME_ROBUST_END


def cliff_entry(genes, cfg, end, bracket=CLIFF_BRACKET, steps=CLIFF_BISECTIONS,
                R_hub=None, R_rim=None):
    """The `entry` at which THIS genome loses its rim layer, at a fixed `end`.

    Returns `{"entry": float|None, "why": str}`.  `None` means the layer-width cliff is not
    the build's edge for this genome, which is not the same as a margin of zero and must
    not be read as one.  `why` says which of the ways that happened.

    IT DELEGATES TO THE MODULE'S CLOSED FORM NOW (PLAN §84), and the thirty sector builds
    are gone.  The width profile is a cubic in `u` whose only `entry`-dependence is one
    linear term, so `wheel_wheel._layer_cliff_from_scalars` roots it on two scalars the
    tangency solve already produced.  Measured against the bisection this replaces: 9.1e-10
    over the held-out draw, which was that bisection's own resolution.

    THE REASON IS STILL CHECKED, NOT ASSUMED.  A steeper entry is not the only way the
    blocking can refuse, and a cliff reported without checking would happily name a
    sector-fit or tangency limit under this one -- the exact class of error PART 6 caught
    this file making once already.  The closed form cannot make that mistake about the
    LAYER, but it can be right about the layer and wrong about the BUILD, if something else
    binds at a shallower entry.  So two verdicts either side of it are still taken.

    THE OLD BRACKET WAS THE BUG, NOT THE INSTRUMENT.  At `CLIFF_BRACKET = (-2.0, 0.0)` this
    returned "builds across the whole bracket" for three of thirty-two held-out genomes and
    `sweep_cliff_clamped_profile` read that as "no edge to project onto -- the safest case".
    All three have edges, at -2.51, -2.30 and -2.12.  The bracket is the module's now.

    `R_hub`/`R_rim` DEFAULT TO THE GENOME'S OWN GENES, WHICH IS NOT ALWAYS WHAT IT BUILDS AT.
    Six of the sixteen drawn genomes are pulled back by the sector-fit clamp (§74), and a
    cliff measured at a radius the genome will never be built at describes a mesh nobody
    builds.  Every caller that quotes a published cliff -- `CLIFF_PUBLISHED`, the candidate
    table -- passes neither and is unaffected; the clamped box passes both.
    """
    R_h = float(genes[12]) if R_hub is None else float(R_hub)
    R_r = float(genes[13]) if R_rim is None else float(R_rim)

    c = WW.layer_cliff_entry(genes, cfg, fillet=(R_h, R_r), end=float(end))
    if c["entry"] is None:
        return {"entry": None, "why": c["why"]}
    cliff = float(c["entry"])

    # THE REASON IS STILL CHECKED, AND IT COSTS TWO BUILDS INSTEAD OF THIRTY.  The closed
    # form answers "where does the LAYER go to zero", which is the right question by
    # construction -- but not necessarily the question the BUILD answers, because a
    # sector-fit or tangency refusal can bind at a shallower entry and would then be the
    # real edge.  So the two verdicts either side of the closed-form cliff are still taken:
    # just inside it the sector must build, just outside it must refuse for exactly this
    # reason.  That is PART 6's lesson kept, at 1/15th of its old price.
    inside = sector_verdict(genes, cfg, R_h, R_r, entry=cliff + CLIFF_PROBE,
                            end=float(end))
    if not inside["built"]:
        return {"entry": None,
                "why": f"bounded by something else: {inside.get('why', '')}"}
    outside = sector_verdict(genes, cfg, R_h, R_r, entry=cliff - CLIFF_PROBE,
                             end=float(end))
    why = outside.get("why", "")
    if outside["built"] or CLIFF_REASON not in why:
        return {"entry": None,
                "why": f"the closed-form cliff at {cliff:.6f} is not the build's edge: "
                       f"{'builds past it' if outside['built'] else why}"}
    return {"entry": cliff, "why": why}


def profile_candidate_rows(table, genes, cfg, target=None):
    """`profile_candidates`, as rows, with the distance to §68's cliff as a column.

    A SIBLING RATHER THAN A WIDER RETURN TYPE.  `profile_candidates` returns bare
    `(entry, end)` pairs and three things depend on that shape -- the self-check comparing
    it to `LAYER_PROFILE_CANDIDATES`, `build()`, and `study_corner_singularity`, which
    imports the constant and iterates it as pairs.  Widening it to buy one column would
    make a cross-study consumer pay for this file's convenience.

    The cliff is a property of `(genome, cfg, end)` and not of the cell, so it is bisected
    once per distinct `end` -- eleven cells over six ends on the fine grid.
    """
    target = MIN_SJ_TARGET if target is None else target
    ok = [r for r in table
          if r["worst_min_scaled_jacobian"] is not None and r["refused"] == 0
          and r["worst_min_scaled_jacobian"] > target]
    ok.sort(key=lambda r: (r["entry"], r["end"]))
    cliffs = {}
    rows = []
    for r in ok:
        end = float(r["end"])
        if end not in cliffs:
            cliffs[end] = cliff_entry(genes, cfg, end)
        c = cliffs[end]
        rows.append({
            "entry": float(r["entry"]), "end": end,
            "worst_min_scaled_jacobian": r["worst_min_scaled_jacobian"],
            "n_genomes": r.get("n_genomes"),
            "cliff_entry": c["entry"],
            "cliff_margin": (None if c["entry"] is None
                             else float(r["entry"]) - c["entry"]),
            "cliff_why": c["why"]})
    return rows


def profile_argmax(table):
    """The two ranking rules over a `sweep_layer_profile_genomes` grid, and their gap.

    PART 13 ranked cells on "the worst min scaled Jacobian over the genomes that BUILT
    at this (entry, end)".  THAT RULE PAYS A CELL FOR REFUSING A HARD GENOME: the worst
    is taken over the survivors, so a profile steep enough to lose a difficult genome
    reports a better floor than one that keeps it.  PART 13's own argmax happened to sit
    on a cell where nothing refused, so its answer is unaffected -- checked here rather
    than assumed -- but the bias is real and it bites as soon as the cell set reaches the
    steep-entry corner where the layer profile itself starts refusing, which is exactly
    what §57's clamp added.

    `no_refusal` is the corrected rule and is the one to quote.  `over_built` is kept so
    the two can be compared, because the gap between them is the size of the bias.
    """
    ok = [r for r in table if r["worst_min_scaled_jacobian"] is not None]
    if not ok:
        return None
    clean = [r for r in ok if r["refused"] == 0]
    over_built = max(ok, key=lambda r: r["worst_min_scaled_jacobian"])
    no_refusal = (max(clean, key=lambda r: r["worst_min_scaled_jacobian"])
                  if clean else None)
    return {"over_built": over_built, "no_refusal": no_refusal,
            "n_cells": len(table), "n_zero_refusal_cells": len(clean),
            "rules_agree": bool(no_refusal is not None
                                and no_refusal["entry"] == over_built["entry"]
                                and no_refusal["end"] == over_built["end"]),
            "bias": (None if no_refusal is None else
                        float(over_built["worst_min_scaled_jacobian"]
                              - no_refusal["worst_min_scaled_jacobian"]))}


UNCAP_BLENDS = (1.0, 0.5, 0.25, 0.0)


def sweep_uncap(genes, cfg, blends=UNCAP_BLENDS):
    """The filleted blocking against the rim's uncap blend, with the unfilleted control.

    THIS IS A QUESTION ABOUT THE TRI-BLOCK, ASKED FROM THE FILLET'S SIDE.  §37 shelved
    the rim tri-block and §46 promoted it: on the FAITHFUL rim -- `uncap` blend 0.0, the
    geometry the tri-block exists to make buildable -- `rim:P_c`'s admissible fillet
    radius goes 0.561 -> 2.585 mm, and `rim:P_c` carries the wheel's global peak.  So the
    obvious hope after PART 10 is that the re-cut which fixed `P_t` also fixes the rim
    junction at blend 0.0 and retires the tri-block with it.

    IT DOES NOT, AND IT MAKES IT WORSE.  What collapses at blend 0.0 is a corner opening
    to 180 degrees -- the junction block's corner at `far_end`, where the uncap edge
    becomes the flank's own straight continuation -- and the filleted blocking SHORTENS
    that block, so the straight corner dominates more of it, not less.  Measured here at
    every blend, filleted and not, so the tri-block's ranking rests on a number from the
    current construction rather than on §37's from before the flip.
    """
    rows = []
    for blend in blends:
        row = {"rim_blend": float(blend)}
        try:
            ctl = WW.sector_blocks(genes, cfg, uncap=(True, blend))
            per = {k: block_quality(np.asarray(v, float))
                   for k, v in ctl.items() if not k.startswith("_")}
            row["control_min_scaled_jacobian"] = min(
                q["min_scaled_jacobian"] for q in per.values())
            row["control_worst_block"] = min(
                per, key=lambda k: per[k]["min_scaled_jacobian"])
        except Exception as exc:
            row["control_why"] = f"{type(exc).__name__}: {exc}"
        try:
            b = WW.filleted_sector(genes, cfg, uncap=(True, blend))
            per = {k: block_quality(np.asarray(v, float))
                   for k, v in b.items() if not k.startswith("_")}
            row.update({
                "min_scaled_jacobian": min(q["min_scaled_jacobian"]
                                           for q in per.values()),
                "worst_block": min(per, key=lambda k: per[k]["min_scaled_jacobian"]),
                "non_positive_gauss_elements": sum(
                    q["non_positive_gauss_elements"] for q in per.values())})
        except (ValueError, NotImplementedError) as exc:
            row["why"] = str(exc).split("  See ")[0]
        rows.append(row)
    return rows


def sweep_sector(genes, cfg, radii_hub, radii_rim):
    """Every (R_hub, R_rim) on the grid, at the chosen profile."""
    ship_h, ship_r = float(genes[12]), float(genes[13])
    rows = []
    for R_hub in radii_hub:
        for R_rim in radii_rim:
            v = sector_verdict(genes, cfg, R_hub, R_rim)
            v.pop("seams", None)
            v.pop("blocks", None)
            v["is_shipped"] = bool(abs(R_hub - ship_h) < 1e-9
                                   and abs(R_rim - ship_r) < 1e-9)
            rows.append(v)
    return rows


# ---------------------------------------------------------------------------
# THE FOLD GATE -- the OTHER feasibility number, and it was already in the tree
# ---------------------------------------------------------------------------
#
# PART 13 found one drawn genome whose TRIMMED SPOKE is sign-flipped and traced it to a
# near self-intersection in the UNFILLETED flank at `s = 0.051`, which the shipped
# station grid steps over and the fillet's trim happens to land on.  It filed the finding
# and said a feasibility gate that catches the class is a different piece of work.
#
# It is one import.  `wheel_geometry.self_intersection_margin` is exactly that gate --
# closed form, no mesh, no build -- and it is already calibrated (`MIN_FOLD_MARGIN_MM`,
# measured over 2001 genomes) and already in five studies' draw filters (`study_gnl`,
# `study_contact`, `study_wheel_fea`, and the two that read them).  `sweep_genomes` here
# and in `study_tri_block` use the two-term filter that omits it.  Everything below is
# the arithmetic that says what that omission costs.

FOLD_AGREEMENT_TOL_MM = 1.0e-3
FOLD_AUDIT_POINTS = 2000
FOLD_GATE_BATCHES = 40


def _curve_and_ctrl(vec, n):
    import wheel_fea as WFEA
    return WG.bezier_centerline(*[float(vec[i]) for i in range(8)],
                                span_mm=WFEA.HUB_RIM_SPAN_MM, num_points=n)


def fold_margin(genes, cfg, num_points=None):
    """`min_s (|1/kappa(s)| - t(s)/2)`, wrapped the way `sector_fit_margin` is wrapped.

    Same shape and the same purpose: a refusal wants a NUMBER computed from the geometry
    alone, before anything is built, and a study that names a mechanism has usually not
    been asked to predict with it.  The computation is
    `wheel_geometry.self_intersection_margin` -- the optimizer's own fold barrier, not a
    second opinion about it -- and at `num_points=None` this returns exactly what
    `study_mesh_quality.fold_margin` returns for the same config, pinned by a test.

    The two thresholds are NOT the same claim.  `folds` says the part does not exist --
    the outward offset has passed the centre of curvature and the flank turns inside out.
    `binds` says the optimizer's barrier is active, which `MIN_FOLD_MARGIN_MM`'s own
    calibration puts 0.1 mm clear of the fold because element quality degrades before it.

    `num_points` exists so the number's INDEPENDENCE from the sampling can be measured
    rather than asserted.  It comes off the Bezier hodograph in closed form, so refining
    it should move nothing; `sweep_fold_gate` reports how much it actually moves, next to
    how much the sampled statement of the same fact moves, which is the whole argument.
    """
    vec = np.asarray(genes, float)
    n = WW.get_config(cfg).n_curve if num_points is None else int(num_points)
    curve, ctrl = _curve_and_ctrl(vec, n)
    m = float(WG.self_intersection_margin(curve, ctrl,
                                          *[float(vec[i]) for i in range(8, 12)],
                                          num_points=n))
    return {"margin_mm": m, "limit_mm": float(WG.MIN_FOLD_MARGIN_MM),
            "num_points": n, "binds": bool(m < WG.MIN_FOLD_MARGIN_MM),
            "folds": bool(m < 0.0)}


def flank_reversal_mm(genes, cfg, num_points=None):
    """The SAMPLED statement of the same thing: least forward progress along a flank.

    Each flank node's step is projected on the centreline tangent it came from.  Negative
    means the outline doubles back on itself between two stations -- the same defect
    PART 13 found by hand with a 2000-point shoelace resample, stated as a signed scalar
    so the closed form can be checked against it rather than trusted.

    Kept separate from `fold_margin` on purpose, and it is the one that MOVES.  A fold
    narrower than the station spacing is stepped over and this returns a positive number
    for a part that self-intersects -- which is exactly PART 13's genome, and exactly why
    the mesh-based filter `sweep_genomes` uses leaks.  Refine it and the misses go away;
    the closed form did not need refining, because there is no grid in it.
    """
    vec = np.asarray(genes, float)
    n = WW.get_config(cfg).n_curve if num_points is None else int(num_points)
    curve, ctrl = _curve_and_ctrl(vec, n)
    tang = WG.bezier_tangent(ctrl, n)
    band = WG.offset_band(curve, *[float(vec[i]) for i in range(8, 12)], n_across=1,
                          normals=WG.normals_from_tangents(tang))
    return min(float((np.diff(band[:, c, :], axis=0) * tang[:-1]).sum(axis=1).min())
               for c in (0, 1))


FOLD_LADDER_POINTS = (97, 600, 1200, 2000, 4000)


def fold_resolution_ladder(genes, cfg, points=FOLD_LADDER_POINTS):
    """The two statements of "this flank folds", side by side, against the sampling.

    97 is the station count PART 13 named -- the shipped, untrimmed grid whose spacing
    steps over the dip.  It is the interesting rung: whatever the sampled test says
    there, the closed form has already made its call, and the rest of the ladder shows
    which of the two is converging and which is switching sides.
    """
    vec = np.asarray(genes, float)
    return [{"num_points": int(n),
             "margin_mm": fold_margin(vec, cfg, n)["margin_mm"],
             "flank_reversal_mm": flank_reversal_mm(vec, cfg, n)}
            for n in points]


def sweep_fold_gate(cfg, genes, batches=FOLD_GATE_BATCHES, seed=None):
    """How leaky is the gate `sweep_genomes` already had, and is the closed form right?

    Two populations over the SAME Latin-hypercube stream the genome sweep draws from, so
    the leak rate below is the rate that stream feeds the box above and not a rate
    measured somewhere else:

      * genomes passing `evaluate_design`'s geometric pair, and how many of those fold;
      * those that ALSO mesh a clean unfilleted sector -- which is the filter
        `sweep_genomes` actually applies -- and how many of THOSE still fold.

    The second number is what the existing filter misses.  It is a proxy for the fold and
    a good one, but it is a proxy: whether a folded flank shows up as an inverted element
    depends on whether a station lands on the dip, and the shipped grid's spacing is not
    a property of the part.

    `agreement` is the closed form audited against `flank_reversal_mm` on every
    geometrically feasible draw, both AT `FOLD_AUDIT_POINTS`, which is fine enough to
    resolve the folds this box contains.  A disagreement within `FOLD_AGREEMENT_TOL_MM`
    of zero is two ways of straddling the same point; one outside it would mean the
    closed form is not measuring what it claims.

    `resolution` is the other half and is the reason the first half has to name its
    sampling.  The same two quantities are recomputed at the CONFIG's own `n_curve` and
    the shifts compared: the closed form's should be numerical noise, and the sampled
    one's should be whole misses -- folded parts whose dip falls between two stations.
    That is PART 13's genome stated as a rate instead of an anecdote, and it is the same
    failure mode as the mesh-based draw filter, one level down.
    """
    from study_mesh_quality import latin_hypercube
    import wheel_fea as WFEA

    seed = GENOME_SWEEP_SEED if seed is None else seed
    low, high, _ = wg.bounds_arrays(WFEA.GENE_SPACE)
    n = {"drawn": 0, "geom": 0, "geom_folds": 0, "geom_binds_only": 0,
         "mesh_clean": 0, "mesh_clean_folds": 0, "mesh_clean_binds_only": 0}
    worst_disagreement = 0.0
    disagreements = 0
    margin_shift = 0.0
    margin_flips = 0
    sampled_misses = 0
    for batch in range(batches):
        for vec in latin_hypercube(512, low, high, seed=seed + batch):
            n["drawn"] += 1
            _, loss = WFEA.evaluate_design(vec)
            if loss["x_order"] != 0.0 or loss["hub_overlap"] != 0.0:
                continue
            n["geom"] += 1
            f = fold_margin(vec, cfg, FOLD_AUDIT_POINTS)
            n["geom_folds"] += int(f["folds"])
            n["geom_binds_only"] += int(f["binds"] and not f["folds"])
            if f["folds"] != (flank_reversal_mm(vec, cfg, FOLD_AUDIT_POINTS) < 0.0):
                disagreements += 1
                worst_disagreement = max(worst_disagreement, abs(f["margin_mm"]))
            # the SAME two quantities at the config's own sampling, which is what every
            # mesh in this file is built on
            at_cfg = fold_margin(vec, cfg)
            margin_shift = max(margin_shift, abs(at_cfg["margin_mm"] - f["margin_mm"]))
            margin_flips += int(at_cfg["folds"] != f["folds"])
            sampled_misses += int(f["folds"] and flank_reversal_mm(vec, cfg) >= 0.0)
            try:
                if not sector_control(vec, cfg)["all_valid"]:
                    continue
            except Exception:
                continue
            n["mesh_clean"] += 1
            n["mesh_clean_folds"] += int(f["folds"])
            n["mesh_clean_binds_only"] += int(f["binds"] and not f["folds"])
    return {"config": cfg, "seed": int(seed), "batches": int(batches),
            "limit_mm": float(WG.MIN_FOLD_MARGIN_MM), "counts": n,
            # The gate is only free if it leaves the genome every other number in this
            # file is measured at alone -- the same question `shipped_is_clamped` asks of
            # the clamp, and the same answer is needed before either is worth having.
            "shipped": fold_margin(genes, cfg),
            "leak_rate": (n["mesh_clean_folds"] / n["mesh_clean"]
                          if n["mesh_clean"] else None),
            "fold_rate": (n["geom_folds"] / n["geom"] if n["geom"] else None),
            "agreement": {"n": n["geom"], "at_points": FOLD_AUDIT_POINTS,
                          "disagreements": disagreements,
                          "worst_disagreement_margin_mm": worst_disagreement,
                          "tol_mm": FOLD_AGREEMENT_TOL_MM,
                          "closed_form_is_the_sampled_one": bool(
                              worst_disagreement < FOLD_AGREEMENT_TOL_MM)},
            # What each of the two costs when the sampling is the config's rather than
            # the audit's.  These are not error bars on the same thing: one is roundoff
            # and the other is a miss.
            "resolution": {"config_points": WW.get_config(cfg).n_curve,
                           "audit_points": FOLD_AUDIT_POINTS,
                           "closed_form_worst_shift_mm": margin_shift,
                           "closed_form_verdict_flips": margin_flips,
                           "sampled_flank_missed_folds": sampled_misses,
                           "sampled_flank_miss_rate": (sampled_misses / n["geom_folds"]
                                                       if n["geom_folds"] else None)}}


GENOME_SWEEP_SEED = 20260823
GENOME_SWEEP_PER_ORIENTATION = 4

# The held-out box (PLAN §78).  Eight per orientation rather than four because the number
# being tested is a RATE and the in-sample one is 16 genomes wide; the offset is
# UNCAP_PLAN PART 9's and must stay far above `sweep_genomes`' `max_batches`, or the two
# streams overlap and the hold-out is not one.
GENOME_HELD_OUT_PER_ORIENTATION = 8
GENOME_HELD_OUT_OFFSET = 7000


def sweep_genomes(cfg, per_orientation=GENOME_SWEEP_PER_ORIENTATION,
                  seed=GENOME_SWEEP_SEED, max_batches=40):
    """The blocking at OTHER GENOMES, grouped by flank orientation.

    THE REST OF THIS SECTION SWEEPS RADII AT ONE GENOME, AND THAT IS NOT THE GENE BOX.
    `flank_orientation` is a property of the centreline, not of `R_hub`/`R_rim`, and its
    own docstring records that of 60 feasible Latin-hypercube genomes **only 16 have the
    shipped genome's `(+1, +1)`** — so a blocking measured at the shipped genome alone has
    been measured on a quarter of the design space.  THE LAST CLAUSE OF THAT FILTER IS A
    PROXY AND IT LEAKS: "the unfilleted sector is clean" is a statement about one sampling
    of the flank, not about the flank, and `sweep_fold_gate` measures how often a genome
    whose part genuinely self-intersects meshes clean anyway.  The draw is left as it is
    so every number already published against this box reproduces; the closed-form margin
    rides on each row as `fold` instead, and `fit_clamp_fold_clean` re-tallies the box
    over the rows that survive it.  This draws feasible genomes until it
    has some of each of the four orientations and reports the blocking on them.

    It found a real bug the radius sweep could not: the sector-closing seam's `dk`.  It is
    also the honest place for what the blocking COSTS away from the shipped genome, which
    is not the same number, and for how often the fillet at the shipped radii simply does
    not fit the sector.

    Feasibility is `evaluate_design`'s own geometric pair (`x_order`, `hub_overlap`) plus
    the requirement that the UNFILLETED sector is clean — the filleted verdict is only
    readable against a baseline that is itself valid, and a genome whose default mesh
    folds is not this blocking's problem.
    """
    from study_mesh_quality import latin_hypercube
    import wheel_fea as WFEA

    low, high, _ = wg.bounds_arrays(WFEA.GENE_SPACE)
    cfgo = WW.get_config(cfg)
    groups = {}
    batch = 0
    while batch < max_batches and any(
            len(groups.get(o, [])) < per_orientation
            for o in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))):
        for vec in latin_hypercube(512, low, high, seed=seed + batch):
            _, loss = WFEA.evaluate_design(vec)
            if loss["x_order"] != 0.0 or loss["hub_overlap"] != 0.0:
                continue
            try:
                o = tuple(float(x) for x in WW.flank_orientation(vec, cfgo))
            except Exception:
                continue
            if len(groups.setdefault(o, [])) >= per_orientation:
                continue
            try:
                ctl = sector_control(vec, cfg)
            except Exception:
                continue
            if not ctl["all_valid"]:
                continue
            try:
                v = sector_verdict(vec, cfg, float(vec[12]), float(vec[13]))
            except Exception as exc:      # a drawn genome must never kill the driver
                v = {"built": False, "why": f"{type(exc).__name__}: {exc}"}
            row = {"orientation": list(o), "genes": [float(x) for x in vec],
                   "R_hub_mm": float(vec[12]), "R_rim_mm": float(vec[13]),
                   "control_min_scaled_jacobian": ctl["min_scaled_jacobian"],
                   # Carried so that a refusal has a NUMBER and not just a reason: the
                   # margin is computed from the geometry alone and is what says whether
                   # the refusal was predictable before the blocking was attempted.
                   "fit": sector_fit_margin(vec, cfg),
                   # And the other one.  `fit` says whether the fillet has room in the
                   # sector; `fold` says whether the SPOKE the fillet is trimming exists
                   # at all.  The draw filter above tests neither -- it tests whether one
                   # particular sampling of this genome happens to mesh.
                   "fold": fold_margin(vec, cfg),
                   "built": v["built"]}
            if v["built"]:
                open_seams = [f"{x['a']}.{x['side_a']}~{x['b']}.{x['side_b']}"
                              for x in v["seams"] if not x["closes"]]
                row.update({"min_scaled_jacobian": v["min_scaled_jacobian"],
                            "worst_block": v["worst_block"],
                            "seams_close": v["seams_close"],
                            "max_seam_gap_mm": v["max_seam_gap_mm"],
                            "open_seams": open_seams,
                            "all_blocks_valid": v["all_blocks_valid"],
                            "dk": {j: int(d) for j, d in v["dirn"].items()}})
            else:
                row["why"] = v["why"]
            groups[o].append(row)
        batch += 1
    return {"seed": seed, "config": cfg, "per_orientation": per_orientation,
            "groups": {str(list(k)): v for k, v in sorted(groups.items())}}


SECTOR_FIT_FACTORS = (0.99, 0.95, 0.90, 0.75)


def sweep_sector_fit_clamp(genes, cfg, genome_rows, factors=SECTOR_FIT_FACTORS):
    """Does pulling each radius back inside its own sector's room fix the refusals?

    PART 10 FINDING 6 counted six refusals of sixteen and named their mechanism; PART 13
    then declined the genome-robust layer profile partly BECAUSE the refusals were
    untouched by it, so "six of sixteen still refuse outright regardless" and nothing
    would collect what the profile bought.  This measures the other half of that sentence.

    Three profiles are crossed with the clamp on purpose.  The clamp's own worth is the
    `built` column at the SHIPPED profile -- that is the refusal half of PLAN.md's item 2,
    on its own.  Crossing it with the genome-robust profile is what re-prices PART 13's
    decision, which was taken against a box where six genomes could not benefit from any
    profile at all.

    THE THIRD IS `MARGIN_ROBUST_*` AND IT IS HERE TO BE HELD OUT (PLAN §80).  §69 picked
    that pair as the argmax of a candidate set built with the cliff margin as a
    constraint, and picked it on THIS study's sixteen-genome draw.  An in-sample argmax
    adopted on its in-sample number is the error UNCAP_PLAN PART 9 and §78 both exist to
    prevent, and this table already runs on the held-out thirty-two, so carrying the pair
    through it is the whole of the missing check.  It costs one more row per factor.
    """
    profiles = (("shipped", LAYER_ENTRY_SLOPE, LAYER_END_OFFSET),
                ("genome_robust", GENOME_ROBUST_ENTRY, GENOME_ROBUST_END),
                ("margin_robust", MARGIN_ROBUST_ENTRY, MARGIN_ROBUST_END))
    cells = [(np.asarray(r["genes"], float), r["fit"]) for r in genome_rows
             if "fit" in r]
    shipped_fit = sector_fit_margin(np.asarray(genes, float), cfg)
    rows = []
    for label, entry, end in profiles:
        for factor in (None,) + tuple(factors):
            built, clear, vals, n_clamped = 0, 0, [], {"hub": 0, "rim": 0}
            refusals = []
            for vec, fit in cells:
                if factor is None:
                    R_hub, R_rim = fit["hub"]["R_mm"], fit["rim"]["R_mm"]
                else:
                    R_hub, R_rim = clamped_radii(fit, factor)
                    for j, R in (("hub", R_hub), ("rim", R_rim)):
                        n_clamped[j] += int(R < fit[j]["R_mm"])
                try:
                    v = sector_verdict(vec, cfg, R_hub, R_rim, entry, end)
                except Exception as exc:          # a drawn genome must not kill the driver
                    refusals.append(f"{type(exc).__name__}")
                    continue
                if not v["built"]:
                    refusals.append(v["why"].split(":")[0])
                    continue
                built += 1
                vals.append(v["min_scaled_jacobian"])
                clear += int(v["min_scaled_jacobian"] > MIN_SJ_TARGET)
            rows.append({
                "profile": label, "entry": float(entry), "end": float(end),
                "factor": factor, "n_genomes": len(cells),
                "n_built": built, "n_clears_target": clear,
                "n_clamped": n_clamped, "refusals": refusals,
                "min_scaled_jacobian_range": ([min(vals), max(vals)] if vals else None),
                "median_min_scaled_jacobian": (float(sorted(vals)[len(vals) // 2])
                                               if vals else None)})
    return {"config": cfg, "factors": [float(f) for f in factors],
            "clamp_used_in_report": SECTOR_FIT_CLAMP,
            "shipped_fit": shipped_fit,
            # The clamp is only free if it does not touch the genome every published
            # number in this file is measured at.  Stated as a field rather than left to
            # be inferred from the two radii above.
            "shipped_is_clamped": bool(any(
                m["limit_mm"] is not None and m["R_mm"] > SECTOR_FIT_CLAMP * m["limit_mm"]
                for m in shipped_fit.values())),
            "rows": rows}


def sweep_cliff_clamped_profile(genes, cfg, genome_rows,
                                factors=CLIFF_PROFILE_FACTORS,
                                end=CLIFF_PROFILE_END, clamp=SECTOR_FIT_CLAMP):
    """The BARRIER half, priced against a per-genome entry instead of a global pair.

    Each genome is built at `factor * cliff_entry(genome)` -- a share of its OWN room --
    with its radii already pulled inside the sector-fit clamp, since a cliff measured at a
    radius the genome will not be built at describes a mesh nobody builds.

    THE COMPARISON THIS EXISTS FOR is the `genome_robust` row of `sweep_sector_fit_clamp`
    at the same `end`: that pair buys 15 of 16 by spending the shipped genome's cliff margin
    down to ~0.07, and the question is whether a per-genome rule buys the same barrier
    clearance for more margin.  So `shipped_margin` rides on every row -- what the rule
    leaves the one genome every published number in this file is measured at -- and it is
    the number that faces §68's 0.5520.

    THE FALLBACK IS GONE, AND SO IS THE CASE IT EXISTED FOR (PLAN §84).  §78 split
    `cliff_entry`'s `None` in two and had genomes reporting *"builds across the whole
    bracket"* take the global entry instead, on the reading that such a genome has no
    layer-width edge to project onto -- the safest case rather than a failure.  That
    reading was wrong: the three genomes it fired on have edges at -2.51, -2.30 and -2.12,
    outside a bracket that stopped at -2.0, and the sentinel was describing the bracket.
    With the module's bracket every genome in both draws has a cliff, the rule answers for
    all of them, and `n_without_cliff` is kept in the row schema reporting zero rather than
    deleted -- a counter that silently stops existing is how a regression hides.

    Measured before the branch was removed: at the adopted factor the three genomes clear
    the barrier on the rule's own answer exactly as they did on the fallback -- 31 of 32
    either way, worst J 0.1721 and median 0.3466 to four figures on both.  So this changes
    what the rule SAYS for three genomes and changes no number it is quoted on.
    """
    cells = []
    for r in genome_rows:
        if "fit" not in r:
            continue
        vec = np.asarray(r["genes"], float)
        R_hub, R_rim = clamped_radii(r["fit"], clamp)
        c = cliff_entry(vec, cfg, end, R_hub=R_hub, R_rim=R_rim)
        cells.append({"genes": vec, "R_hub": R_hub, "R_rim": R_rim,
                      "cliff": c["entry"], "why": c["why"]})

    shipped = np.asarray(genes, float)
    shipped_fit = sector_fit_margin(shipped, cfg)
    sR_hub, sR_rim = clamped_radii(shipped_fit, clamp)
    shipped_cliff = cliff_entry(shipped, cfg, end, R_hub=sR_hub, R_rim=sR_rim)

    rows = []
    for factor in factors:
        built, clear, vals, margins = 0, 0, [], []
        no_edge, unevaluable, refusals = 0, 0, []
        for c in cells:
            if c["cliff"] is None:
                # The only `None` left is a genuine one: something other than the layer
                # bounds this genome, so the rule cannot be evaluated on it.  The old
                # `CLIFF_NO_EDGE` branch is retired with its bracket (§84); `no_edge` is
                # kept and reported as zero rather than deleted, so the day it stops being
                # zero is visible instead of silent.
                unevaluable += 1
                continue
            entry = float(factor) * float(c["cliff"])
            margins.append(entry - float(c["cliff"]))
            try:
                v = sector_verdict(c["genes"], cfg, c["R_hub"], c["R_rim"], entry, end)
            except Exception as exc:      # a drawn genome must not kill the driver
                refusals.append(f"{type(exc).__name__}")
                continue
            if not v["built"]:
                refusals.append(v["why"].split(":")[0])
                continue
            built += 1
            vals.append(v["min_scaled_jacobian"])
            clear += int(v["min_scaled_jacobian"] > MIN_SJ_TARGET)
        s_entry = (None if shipped_cliff["entry"] is None
                   else float(factor) * float(shipped_cliff["entry"]))
        rows.append({
            "factor": float(factor), "end": float(end),
            "n_genomes": len(cells),
            # Kept apart on purpose: one of these is the rule declining to act and the
            # other is the rule being unable to.
            "n_without_cliff": no_edge, "n_unevaluable": unevaluable,
            "n_built": built, "n_clears_target": clear, "refusals": refusals,
            # What the rule LEAVES, which is the half §68's objection is about.
            "shipped_entry": s_entry,
            "shipped_margin": (None if s_entry is None
                               else s_entry - float(shipped_cliff["entry"])),
            "margin_range": ([min(margins), max(margins)] if margins else None),
            "min_scaled_jacobian_range": ([min(vals), max(vals)] if vals else None),
            "median_min_scaled_jacobian": (float(sorted(vals)[len(vals) // 2])
                                           if vals else None)})
    return {"config": cfg, "end": float(end), "clamp": float(clamp),
            "factors": [float(f) for f in factors],
            "shipped_cliff": shipped_cliff,
            # The two global pairs this rule is measured against, so the comparison does
            # not have to be assembled by hand from another section.
            "reference": {
                "shipped": {"entry": float(LAYER_ENTRY_SLOPE),
                            "end": float(LAYER_END_OFFSET)},
                "genome_robust": {"entry": float(GENOME_ROBUST_ENTRY),
                                  "end": float(GENOME_ROBUST_END)}},
            "per_genome_cliff": [
                {"cliff": c["cliff"], "why": (None if c["cliff"] is not None
                                              else c["why"]),
                 "R_hub_mm": c["R_hub"], "R_rim_mm": c["R_rim"]} for c in cells],
            "rows": rows}


def build_sector_section(genes, configs, junctions):
    """The whole-sector measurement, per config."""
    box_h = tuple(sorted({0.40, 0.60, float(genes[12]), 1.00, 1.50, 2.00, 2.50, 3.00}))
    box_r = tuple(sorted({0.50, 1.00, 1.50, 2.00, 2.50, float(genes[13])}))
    profile_box = ((0.40, 0.50), (float(genes[12]), float(genes[13])),
                   (1.50, 1.50), (3.00, 3.00), (0.40, 3.00), (3.00, 0.50))
    out = {"entry_slope": LAYER_ENTRY_SLOPE, "end_offset": LAYER_END_OFFSET,
           "seam_tol_mm": SEAM_TOL_MM,
           "block_order": list(SECTOR_BLOCK_ORDER),
           "block_region": dict(SECTOR_BLOCK_REGION),
           "gene_box": {"R_hub_mm": list(box_h), "R_rim_mm": list(box_r)},
           "genomes": sweep_genomes(configs[0]),
           # THE HELD-OUT BOX (PLAN §78).  §74's "16 of 16" was measured on the genomes the
           # clamp was designed against, and UNCAP_PLAN PART 9 has just shown this project
           # what an in-sample rate is worth -- 1.000 fitted, 0.833 held out, half the rule
           # falsified.  Same sampler, same filter, same config; a DISJOINT stream.
           #
           # The offset must clear `sweep_genomes`' own batch range, which walks
           # `seed + batch` for `batch < max_batches`: at 40 batches the in-sample run
           # occupies 20260823-20260862 and this one 20267823-20267862.  `+7000` is
           # UNCAP_PLAN PART 9's offset, so the two hold-outs in this project are named the
           # same way.
           "genomes_held_out": sweep_genomes(
               configs[0], per_orientation=GENOME_HELD_OUT_PER_ORIENTATION,
               seed=GENOME_SWEEP_SEED + GENOME_HELD_OUT_OFFSET),
           # The population the box above is drawn from, and what the draw filter misses
           # in it.  Same stream, same config, so the leak rate is this box's own.
           "fold_gate": sweep_fold_gate(configs[0], genes),
           "per_config": {}}
    # The population above as an anecdote you can read: the same two numbers against the
    # sampling, on the genomes the gate actually rejects -- one of which is PART 13's.
    out["fold_gate"]["ladder"] = [
        {"orientation": r["orientation"], "R_hub_mm": r["R_hub_mm"],
         "worst_block": r.get("worst_block"),
         "rungs": fold_resolution_ladder(np.asarray(r["genes"], float), configs[0])}
        for rows in out["genomes"]["groups"].values() for r in rows
        if r["fold"]["folds"]]
    for cfg in configs:
        cfgo = WW.get_config(cfg)
        shipped = sector_verdict(genes, cfg, float(genes[12]), float(genes[13]))
        seams = shipped.pop("seams", [])
        blocks = shipped.pop("blocks", {})
        out["per_config"][cfg] = {
            "control": sector_control(genes, cfg),
            "node_counts": {"n_thick": cfgo.nn(cfgo.n_thick),
                            "n_collar_r": cfgo.nn(cfgo.n_collar_r),
                            "n_rim_r": cfgo.nn(cfgo.n_rim_r),
                            "ring_radial_used": cfgo.nn(cfgo.n_thick)},
            "shipped": shipped,
            "shipped_blocks": blocks,
            "shipped_seams": seams,
            "box": sweep_sector(genes, cfg, box_h, box_r),
            "fit_limit": {j: sector_fit_limit(genes, cfg, j) for j in junctions},
            "landing": landing_angles(genes, cfg, junctions),
            "uncap": sweep_uncap(genes, cfg),
            # The profile sweep is a DERIVATION OF TWO CONSTANTS, not a per-config
            # measurement, and it is the expensive part of this section -- 30 pairs x 6
            # box corners is 180 sectors.  Run at the first config only, and reported
            # with the reason: the box column above shows the two configs agreeing on
            # the worst block to four figures, so a second copy of this surface would
            # cost a minute to say the same thing twice.
            "profile": (sweep_layer_profile(
                genes, cfg,
                (-0.30, -0.35, -0.40, -0.45, -0.50, -0.60),
                (1.20, 1.40, 1.60, 1.80, 2.00), profile_box)
                if cfg == configs[0] else None),
            # The genome-diverse re-derivation.  Expensive for the same reason
            # `profile` is -- 49 (entry, end) pairs x 11 genomes -- so it is run once,
            # at the first config, reusing the genomes `sweep_genomes` already drew
            # rather than drawing a second set.
            "profile_genomes": (sweep_layer_profile_genomes(
                genes, cfg,
                [r for rows in out["genomes"]["groups"].values() for r in rows])
                if cfg == configs[0] else None),
            # The SAME derivation over the cells §57 and §58 added: every drawn genome
            # clamped inside its own sector's room, minus the ones whose spoke does not
            # exist.  §54's argmax was fitted to ten cells because six could not build
            # and one was thrown out by hand; this one is fitted to fifteen, and whether
            # the pair moves is the cheap half of PLAN.md's item 1.
            "profile_genomes_buildable": (sweep_layer_profile_genomes(
                genes, cfg,
                [r for rows in out["genomes"]["groups"].values() for r in rows],
                clamp=SECTOR_FIT_CLAMP, fold_gate=True)
                if cfg == configs[0] else None),
            # The same derivation again on the refined neighbourhood of the winner.
            # PART 17: whether the two-objective cell PART 16 named is the best point of
            # the region or just the first grid point inside it.
            "profile_genomes_fine": (sweep_layer_profile_genomes(
                genes, cfg,
                [r for rows in out["genomes"]["groups"].values() for r in rows],
                entries=LAYER_PROFILE_FINE_ENTRIES, ends=LAYER_PROFILE_FINE_ENDS,
                clamp=SECTOR_FIT_CLAMP, fold_gate=True)
                if cfg == configs[0] else None),
            # The REFUSAL half of PLAN.md's item 2, priced.  Same config as the genome
            # sweep it reads, for the same reason `profile_genomes` is: it re-uses those
            # rows rather than drawing a second set, and the margins in them were the
            # expensive part.
            "fit_clamp": (sweep_sector_fit_clamp(
                genes, cfg,
                [r for rows in out["genomes"]["groups"].values() for r in rows])
                if cfg == configs[0] else None),
            # The same table over the genomes that describe a part that EXISTS.  Same
            # function and the same rows, minus the ones the closed-form fold margin
            # rejects -- so the difference between the two tables is the gate's price,
            # measured on the same genomes rather than on a second draw that would move
            # everything else at the same time.
            "fit_clamp_fold_clean": (sweep_sector_fit_clamp(
                genes, cfg,
                [r for rows in out["genomes"]["groups"].values() for r in rows
                 if not r["fold"]["binds"]])
                if cfg == configs[0] else None),
            # THE SAME TABLE ON GENOMES THE CLAMP HAS NEVER SEEN (PLAN §78).  Identical
            # function and identical factors, so the only thing that differs between this
            # and `fit_clamp` above is the draw -- which is what makes the two rates
            # comparable at all.
            "fit_clamp_held_out": (sweep_sector_fit_clamp(
                genes, cfg,
                [r for rows in out["genomes_held_out"]["groups"].values()
                 for r in rows])
                if cfg == configs[0] else None),
            # THE BARRIER HALF, priced against a per-genome entry rather than a global
            # pair, on both boxes.  See `sweep_cliff_clamped_profile`: this is the half
            # §74 left open and the half §68's declined profile would have closed.
            "cliff_profile": (sweep_cliff_clamped_profile(
                genes, cfg,
                [r for rows in out["genomes"]["groups"].values() for r in rows])
                if cfg == configs[0] else None),
            "cliff_profile_held_out": (sweep_cliff_clamped_profile(
                genes, cfg,
                [r for rows in out["genomes_held_out"]["groups"].values()
                 for r in rows])
                if cfg == configs[0] else None),
        }
        per = out["per_config"][cfg]
        # The ranking rule, applied to both grids and stated rather than left to whoever
        # reads the table.  See `profile_argmax`: the rule PART 13 used pays a cell for
        # refusing a hard genome, and the clamped cell set reaches the corner where that
        # starts to matter.
        per["profile_argmax"] = ({
            "part_13": profile_argmax(per["profile_genomes"]),
            "buildable": profile_argmax(per["profile_genomes_buildable"])}
            if per["profile_genomes"] else None)
        per["profile_candidates_fine"] = ({
            "measured": [list(p) for p in
                         profile_candidates(per["profile_genomes_fine"])],
            "constant": [list(p) for p in LAYER_PROFILE_FINE_CANDIDATES],
            "target": MIN_SJ_TARGET}
            if per["profile_genomes_fine"] else None)
        per["profile_candidates"] = ({
            "measured": [list(p) for p in
                         profile_candidates(per["profile_genomes_buildable"])],
            "constant": [list(p) for p in LAYER_PROFILE_CANDIDATES],
            "target": MIN_SJ_TARGET}
            if per["profile_genomes_buildable"] else None)
        # §68's cliff as a COLUMN, which is the sentence that section ended on: "the
        # distance to that edge was never a column in any table."  Alongside the pair
        # lists rather than inside them, because those are consumed as pairs.
        per["profile_candidate_rows_fine"] = (
            profile_candidate_rows(per["profile_genomes_fine"], genes, cfg)
            if per["profile_genomes_fine"] else None)
        per["profile_candidate_rows"] = (
            profile_candidate_rows(per["profile_genomes_buildable"], genes, cfg)
            if per["profile_genomes_buildable"] else None)
        # The shipped pair's own margin, which is what every candidate above is measured
        # AGAINST.  It is not a candidate -- it does not clear the box floor -- so it
        # cannot appear in the rows, and quoting 0.031 without the 0.552 it is small
        # relative to is how §60 published that cell as a curiosity.
        # The four `end`s PART 20 bisected by hand, re-measured so the self-check can
        # compare the automated column against the published numbers rather than against
        # itself.  Cheap: four bisections, and two of the ends are already cached above.
        per["_cliff_audit"] = ([
            {"end": end, "published": pub,
             "got": cliff_entry(genes, cfg, end)["entry"]}
            for end, pub in CLIFF_PUBLISHED]
            if cfg == configs[0] else None)
        _cliff_shipped = cliff_entry(genes, cfg, LAYER_END_OFFSET)
        per["shipped_profile_cliff"] = {
            "entry": LAYER_ENTRY_SLOPE, "end": LAYER_END_OFFSET,
            "cliff_entry": _cliff_shipped["entry"],
            "cliff_margin": (None if _cliff_shipped["entry"] is None
                             else LAYER_ENTRY_SLOPE - _cliff_shipped["entry"]),
            "cliff_why": _cliff_shipped["why"]}
    return out


# ---------------------------------------------------------------------------
# THE SELF-CHECKS
# ---------------------------------------------------------------------------

def controls(genes, configs):
    """The default mesh is untouched, and the trimmed spoke is the default one at R=0.

    Both matter for the same reason: every number in this file is a DIFFERENCE from the
    shipped construction, so if the baseline moved, all of them moved with it.
    """
    out = {}
    for cfg in configs:
        base = np.asarray(WW.sector_blocks(genes, cfg, fillet=None)["spoke"], float)
        trim0 = trimmed_spoke(genes, cfg, 0.0, 0.0)
        out[cfg] = {
            "trimmed_spoke_at_zero_max_abs_dx_mm": float(np.abs(base - trim0).max()),
            "default_spoke_clean": block_quality(base)["valid"],
        }
        out[cfg]["pass"] = bool(
            out[cfg]["trimmed_spoke_at_zero_max_abs_dx_mm"] < 1.0e-12
            and out[cfg]["default_spoke_clean"])
    return out


def self_checks(rec):
    """The three claims this file exits nonzero on, each computed rather than asserted.

    A characterisation finding -- which candidate folds, what `min_sj` it reaches -- is
    NOT one of them.  These are the structural statements: the tangency is exact, so the
    cusp at `B` is exact; the route-2 angle is a boundary quantity; and the baseline this
    is all measured against is the shipped mesh.
    """
    checks = {}
    checks["controls"] = all(c["pass"] for c in rec["controls"].values())
    cusp = True
    for per in rec["region"].values():
        for rows in per.values():
            for r in rows:
                if r.get("tangency") and not r["is_cusp_at_B"]:
                    cusp = False
    checks["cusp_at_B_is_exact"] = cusp
    inv = True
    for per in rec["route2"].values():
        for row in per.values():
            if row and not row["angle_is_a_boundary_quantity"]:
                inv = False
    checks["route2_angle_is_a_boundary_quantity"] = inv
    # The whole-sector blocking: STRUCTURAL claims only.  Whether a block's min scaled
    # Jacobian is 0.35 or 0.20 is a characterisation finding and is reported, not gated.
    # Whether the seams CLOSE is not: a blocking whose seams do not close is not a
    # blocking, and STEP 1b would be wiring a mesh that cannot be assembled.
    sec = rec.get("sector")
    if sec:
        closes, control = True, True
        for per in sec["per_config"].values():
            if not per["shipped"].get("built"):
                closes = False
                continue
            for s in per["shipped_seams"]:
                if not s["counts_agree"] or not s["closes"]:
                    closes = False
            if not per["control"]["all_valid"]:
                control = False
        checks["sector_seams_close_whole_edge"] = closes
        checks["unfilleted_sector_still_clean"] = control
        # And at EVERY flank orientation, which is a different claim: the sector-closing
        # seam's `dk` follows the genome, and written as +1 it misses by a whole sector.
        # Structural, so it gates.  Whether those genomes' BLOCKS are good enough is a
        # characterisation finding and is reported, not gated.
        rows = [r for v in sec["genomes"]["groups"].values() for r in v if r["built"]]
        checks["sector_seams_close_at_every_orientation"] = bool(
            rows and all(r["seams_close"] for r in rows))
        # The margin is only useful if it PREDICTS.  Structural, so it gates: a margin
        # that classified some other way would be a number this file reports as a cause
        # while the causation ran elsewhere, which is worse than not reporting it.  How
        # MANY genomes refuse is a characterisation finding and is not gated.
        allr = [r for v in sec["genomes"]["groups"].values() for r in v]
        checks["the_hub_margin_predicts_every_refusal"] = bool(
            allr and all(r["fit"]["hub"]["binds"] != r["built"] for r in allr))
        fc = sec["per_config"][rec["configs"][0]].get("fit_clamp")
        if fc:
            # And the clamp is only free if it leaves the genome every other number in
            # this file is measured at alone.
            checks["the_clamp_is_inert_on_the_shipped_genome"] = (
                not fc["shipped_is_clamped"])
        # AND NOTHING HERE GATES WHAT THE MARGIN-ROBUST PAIR CLEARS OUT OF SAMPLE (§80).
        # That number is the answer §80 asked for, and a gate on it would be a gate on the
        # answer.  Nor is "the profile does not change WHICH genomes fit" gated, however
        # much it looks like a law: §68's own cliff is a profile causing a hard refusal,
        # so a difference in the `built` column across profiles is a layer-width refusal
        # and the row's `refusals` field already names it.
        # THE HELD-OUT BOX (PLAN §78), SPLIT THE WAY EVERYTHING ELSE HERE IS.
        #
        # The MECHANISM gates and the RATE does not.  "The hub margin predicts every
        # refusal" is §57's whole claim -- that the refusal is a property of the genome,
        # computable before the blocking is attempted -- and a claim of that shape either
        # survives fresh genomes or is not a mechanism.  How many of the held-out genomes
        # BUILD is exactly the kind of characterisation finding this file reports rather
        # than gates, and gating it would be gating the answer to the question.
        ho = sec.get("genomes_held_out")
        if ho:
            hall = [r for v in ho["groups"].values() for r in v]
            # THE EITHER-JUNCTION FORM, AND THE HOLD-OUT IS WHY.  In-sample every one of
            # the six refusals binds at the HUB, and §57/§74 wrote the mechanism up in
            # those words.  The held-out draw contains the first RIM refusal, so the hub
            # margin alone classifies 31 of 32 and the sector-fit margin at either
            # junction classifies 32 of 32.  The mechanism is unchanged -- a tangent point
            # past the next sector's corner -- and the JUNCTION in its phrasing was an
            # artefact of which genomes the first draw happened to contain.
            checks["the_sector_fit_margin_predicts_every_refusal_held_out"] = bool(
                hall and all(
                    (r["fit"]["hub"]["binds"] or r["fit"]["rim"]["binds"]) != r["built"]
                    for r in hall))
            checks["sector_seams_close_at_every_orientation_held_out"] = bool(
                hall and all(r["seams_close"] for r in hall if r["built"]))
            # And the hold-out has to BE one.  Two draws that overlapped would make the
            # comparison meaningless in the direction that flatters it, so the disjointness
            # is computed from the genes rather than trusted to the seed arithmetic.
            insample = {tuple(round(x, 12) for x in r["genes"])
                        for v in sec["genomes"]["groups"].values() for r in v}
            checks["the_held_out_draw_is_disjoint"] = bool(
                hall and not any(tuple(round(x, 12) for x in r["genes"]) in insample
                                 for r in hall))
        # The fold gate, stated as the ONE-SIDED claim it can actually support.  Whether a
        # folded flank shows up as an inverted element depends on where the trim puts a
        # station, so "folds => the block inverts" is luck and is reported, not gated.
        # "Fold-clean => nothing inverts" is the gate's own promise and gates.
        builtr = [r for r in allr if r["built"]]
        checks["no_fold_clean_genome_inverts_a_block"] = bool(
            builtr and all(r["all_blocks_valid"] for r in builtr
                           if not r["fold"]["folds"]))
        # And the closed form has to BE the thing it stands in for.  A disagreement
        # further than a micron from the fold point would mean it is not.
        fg = sec.get("fold_gate")
        if fg:
            checks["the_closed_form_fold_is_the_sampled_flank"] = bool(
                fg["agreement"]["closed_form_is_the_sampled_one"])
            # And the closed form has to be the one that does not move, or the gate is
            # just a second grid.  Gated because the whole argument for it rests here.
            # The SHIFT is what gates: it bounds every verdict change between the two
            # samplings at once, and the barrier it has to stay inside of is 100x it.
            # How many verdicts actually flipped is reported next to how many folds the
            # sampled test outright missed, and is a characterisation finding.
            checks["the_closed_form_fold_does_not_move_with_the_sampling"] = bool(
                fg["resolution"]["closed_form_worst_shift_mm"] < 0.1 * fg["limit_mm"])
            checks["the_fold_gate_is_inert_on_the_shipped_genome"] = (
                not fg["shipped"]["binds"])
        # The profile pair is only defensible if the cell it comes from kept every
        # genome.  Structural, so it gates: an argmax that won by refusing a hard genome
        # is not an argmax over the box, it is an argmax over a subset the cell chose.
        pam = sec["per_config"][rec["configs"][0]].get("profile_argmax")
        if pam:
            checks["every_profile_argmax_kept_every_genome"] = bool(
                all(a and a["no_refusal"] is not None
                    and a["no_refusal"]["refused"] == 0 for a in pam.values()))
            # And PART 13's published pair has to still BE the corrected argmax on the
            # larger cell set, or the constant it left behind is stale.  Reported as a
            # check because the alternative -- quietly keeping it -- is the failure mode
            # `measured-not-adopted decisions expire` exists to catch.
            nr = pam["buildable"]["no_refusal"]
            checks["the_re_derived_argmax_is_still_PART_13s_pair"] = bool(
                nr and abs(nr["entry"] - GENOME_ROBUST_ENTRY) < 1e-12
                and abs(nr["end"] - GENOME_ROBUST_END) < 1e-12)
        # `LAYER_PROFILE_CANDIDATES` is read by ANOTHER study, the only reason it is
        # a constant rather than a derivation.  Gated so it cannot drift from the surface
        # it names without this file going red first.
        for key, name in (("profile_candidates", "the_candidate_constant"),
                          ("profile_candidates_fine", "the_fine_candidate_constant")):
            pr = sec["per_config"][rec["configs"][0]].get(key)
            if pr:
                checks[f"{name}_matches_the_measured_surface"] = bool(
                    pr["measured"] == pr["constant"])
        # THE CLIFF REPRODUCES PART 20's HAND BISECTIONS.  Those four numbers are quoted
        # in PLAN §68 and FILLET_PLAN PART 20 and are the whole evidence for declining
        # the profile, so the automated column has to land on them or one of the two is
        # wrong.  1e-4 is the precision §68 published to.
        cliffs = sec["per_config"][rec["configs"][0]].get("_cliff_audit")
        if cliffs:
            checks["the_cliff_column_reproduces_PART_20s_bisections"] = bool(
                all(c["got"] is not None and abs(c["got"] - c["published"]) < 1e-4
                    for c in cliffs))
        sh = sec["per_config"][rec["configs"][0]].get("shipped_profile_cliff")
        rows = sec["per_config"][rec["configs"][0]].get("profile_candidate_rows_fine")
        if sh and rows:
            # PART 20's finding, as a check rather than a sentence: every candidate the
            # arc produced stands closer to the cliff than the pair that ships.  A future
            # grid that produced a roomier candidate would go red here, which is the
            # outcome that should reopen the call.
            margins = [r["cliff_margin"] for r in rows if r["cliff_margin"] is not None]
            checks["the_shipped_profile_is_farthest_from_the_cliff"] = bool(
                sh["cliff_margin"] is not None and margins
                and sh["cliff_margin"] > max(margins))
    checks["pass"] = bool(all(v for k, v in checks.items() if k != "pass"))
    return checks


# ---------------------------------------------------------------------------

def build(genes, configs, junctions, radii, refinements):
    rec = {"configs": list(configs), "junctions": list(junctions),
           "radii_mm": [float(R) for R in radii],
           "refinements": list(refinements),
           "shipped_radii_mm": {"hub": float(genes[12]), "rim": float(genes[13])},
           "controls": controls(genes, configs),
           "region": {}, "route2": {}, "spoke_trim": {}, "candidates": {}, "price": {}}
    for cfg in configs:
        rec["region"][cfg] = sweep_region(genes, cfg, junctions, radii)
        rec["route2"][cfg] = {
            j: moved_corner(genes, cfg, j,
                            float(genes[12] if j == "hub" else genes[13]))
            for j in junctions}
        rec["spoke_trim"][cfg] = sweep_spoke_trim(genes, cfg, radii)
        rec["candidates"][cfg] = sweep_candidates(genes, cfg, junctions, radii,
                                                  refinements)
        rec["price"][cfg] = price(genes, cfg, junctions)
    rec["sector"] = build_sector_section(genes, configs, junctions)
    rec["self_checks"] = self_checks(rec)
    return rec


def _shipped_row(rec, cfg, junction, name):
    ship = rec["shipped_radii_mm"][junction]
    for row in rec["candidates"][cfg][junction]:
        if abs(row["radius_mm"] - ship) < 1e-9 and row.get("blocks"):
            return row["blocks"][name]
    return None


def _print(rec):
    print("\n  CONTROLS (the trimmed spoke at R=0 IS the default spoke)")
    for cfg, c in rec["controls"].items():
        print(f"    {cfg:7s} max|dx| = {c['trimmed_spoke_at_zero_max_abs_dx_mm']:.3e} mm"
              f"   {'PASS' if c['pass'] else 'FAIL'}")

    cfg0 = rec["configs"][0]
    print("\n  ROUTE 1 AS WRITTEN: the region `A - P_t - B` has TWO CUSPS, "
          "at every radius")
    print(f"    {'junction':9s} {'R (mm)':>8s} {'at A (deg)':>11s} {'at P_t (deg)':>13s} "
          f"{'at B (deg)':>11s} {'flank leg':>10s} {'ring leg':>9s} "
          f"{'P_t as make junction chords it':>31s}")
    for junction, rows in rec["region"][cfg0].items():
        for r in rows:
            if not r.get("tangency"):
                print(f"    {junction:9s} {r['radius_mm']:8.4f} "
                      f"{'no fillet is tangent to both legs':>46s}")
                continue
            print(f"    {junction:9s} {r['radius_mm']:8.4f} {r['at_A_deg']:11.4f} "
                  f"{r['at_P_t_deg']:13.4f} {r['at_B_deg']:11.4f} "
                  f"{r['leg_flank_mm']:10.4f} {r['leg_ring_chord_mm']:9.4f} "
                  f"{r['at_P_t_chord_deg']:31.4f}")

    print("\n  ROUTE 2: the angle that fails is a BOUNDARY quantity, so no generated "
          "interior moves it")
    for cfg in rec["configs"]:
        for junction, row in rec["route2"][cfg].items():
            if not row:
                continue
            print(f"    {cfg:7s} {junction:4s} corner {row['unfilleted_angle_deg']:7.3f}"
                  f" -> {row['angle_deg']:7.3f} deg   end cross-section "
                  f"{row['unfilleted_end_cross_section_mm']:6.3f} -> "
                  f"{row['end_cross_section_mm']:6.3f} mm   after {WINSLOW_ITERS} "
                  f"Winslow sweeps {row['angle_after_winslow_deg']:7.3f} deg "
                  f"(boundary moved {row['winslow_max_boundary_shift_mm']:.1e} mm)")
            print(f"    {'':7s} {'':4s} and the two BOUNDARY CURVES themselves meet at "
                  f"{row['tangent_angle_deg']:.4f} deg -- the node angle above is that "
                  f"angle sampled, which is why it moves with the config")

    print("\n  THE SPOKE WAS NEVER THE BLOCKER: end it at `s_A` and it is clean")
    for cfg in rec["configs"]:
        for junction, rows in rec["spoke_trim"][cfg].items():
            ok = [r["radius_mm"] for r in rows if r.get("built") and r["valid"]]
            bad = [r["radius_mm"] for r in rows if r.get("built") and not r["valid"]]
            no = [r["radius_mm"] for r in rows if not r.get("built")]
            span = f" ({min(ok):.2f}-{max(ok):.2f} mm)" if ok else ""
            print(f"    {cfg:7s} {junction:4s} clean at {len(ok)}/{len(rows)} swept "
                  f"radii{span}"
                  + (f", folds at {bad}" if bad else "")
                  + (f", no tangency at {no}" if no else ""))

    print("\n  THE CANDIDATE BLOCKS, AT THE SHIPPED RADII")
    print(f"    {'config':7s} {'junction':9s} {'candidate':21s} "
          f"{'mixed cells (1x/2x/4x)':>24s} {'min scaled J':>13s} {'valid':>6s}")
    for cfg in rec["configs"]:
        for junction in rec["junctions"]:
            for name in CANDIDATES:
                per = _shipped_row(rec, cfg, junction, name)
                if per is None:
                    continue
                got = [per[str(m)] for m in rec["refinements"] if str(m) in per]
                mixed = "/".join(str(q.get("mixed_sign_cells", "-")) for q in got)
                sj = got[0].get("min_scaled_jacobian")
                valid = all(q.get("valid") for q in got)
                sjs = f"{sj:13.4f}" if sj is not None else f"{'-':>13s}"
                print(f"    {cfg:7s} {junction:9s} {name:21s} {mixed:>24s} "
                      f"{sjs} {'YES' if valid else 'no':>6s}")

    print("\n  AND THE ONE THAT WORKS DOES SO ACROSS THE WHOLE GENE BOX")
    for cfg in rec["configs"]:
        for junction in rec["junctions"]:
            rows = [r for r in rec["candidates"][cfg][junction] if r.get("blocks")]
            per = [r["blocks"]["boundary_layer"] for r in rows]
            ok = sum(1 for b in per
                     if all(q["valid"] for q in b.values() if q.get("built")))
            sj = min(q["min_scaled_jacobian"] for b in per for q in b.values()
                     if q.get("built"))
            print(f"    {cfg:7s} {junction:4s} valid at {ok}/{len(rows)} swept radii "
                  f"x {len(rec['refinements'])} refinements, worst min scaled J "
                  f"{sj:.4f} (MIN_SJ_TARGET is 0.2)")

    print("\n  WHAT THE ONE THAT WORKS ASKS OF ITS NEIGHBOURS")
    for cfg in rec["configs"]:
        for junction, p in rec["price"][cfg].items():
            if not p.get("radius_mm"):
                continue
            print(f"    {cfg:7s} {junction:4s} cut {p['cut_depth_mm']:.4f} mm past the "
                  f"ring circle ({100.0 * p['cut_depth_fraction_of_ring']:.1f}% of the "
                  f"{p['ring_depth_available_mm']:.2f} mm available), over a "
                  f"{p['footprint_deg']:.3f} deg footprint "
                  f"({100.0 * p['footprint_fraction_of_sector']:.1f}% of the sector); "
                  f"the notch it needs in the ring block is {p['notch_deg']:.3f} deg; "
                  f"the spoke gives up {p['spoke_stations_given_up']:.1f} stations")

    sec = rec.get("sector")
    if sec:
        print("\n  THE WHOLE FILLETED SECTOR: ELEVEN BLOCKS, FOURTEEN SEAMS, "
              "AT THE SHIPPED RADII")
        print(f"    {'config':7s} {'blocks':>6s} {'seams':>5s} {'worst min scaled J':>19s} "
              f"{'worst block':>14s} {'max seam gap (mm)':>18s} "
              f"{'unfilleted control':>19s}")
        for cfg, per in sec["per_config"].items():
            sh = per["shipped"]
            if not sh.get("built"):
                print(f"    {cfg:7s} REFUSED: {sh.get('why')}")
                continue
            print(f"    {cfg:7s} {sh['n_blocks']:6d} {sh['n_seams']:5d} "
                  f"{sh['min_scaled_jacobian']:19.4f} {sh['worst_block']:>14s} "
                  f"{sh['max_seam_gap_mm']:18.2e} "
                  f"{per['control']['min_scaled_jacobian']:19.4f}")
        cfg0 = rec["configs"][0]
        print("\n  EVERY BLOCK OF IT, AT THE SHIPPED RADII, AT "
              f"{cfg0}")
        for name in sec["block_order"]:
            q = sec["per_config"][cfg0]["shipped_blocks"].get(name)
            if q is None:
                continue
            print(f"    {name:16s} {str(q['shape']):>10s}  min scaled J {q['min_scaled_jacobian']:7.4f}"
                  f"   mixed {q['mixed_sign_cells']:3d}   non-positive Gauss "
                  f"{q['non_positive_gauss_elements']:3d}   "
                  f"{'VALID' if q['valid'] else 'FOLDS'}")
        print("\n  EVERY SEAM OF IT, AT "
              f"{cfg0} -- whole edge, matching counts, coincident nodes")
        for s in sec["per_config"][cfg0]["shipped_seams"]:
            gap = "-" if s["max_gap_mm"] is None else f"{s['max_gap_mm']:.2e}"
            print(f"    {s['a']:14s}.{s['side_a']:2s} ~ {s['b']:14s}.{s['side_b']:2s} "
                  f"dk={s['dk']} rev={str(s['reverse']):5s} "
                  f"n {s['n_a']:3d}/{s['n_b']:3d} gap {gap:>9s} mm  "
                  f"{'CLOSES' if s['closes'] else 'OPEN'}")
        print("\n  AND IT CLOSES ACROSS THE GENE BOX")
        for cfg, per in sec["per_config"].items():
            rows = per["box"]
            built = [r for r in rows if r["built"]]
            ok = [r for r in built if r["all_blocks_valid"] and r["seams_close"]]
            sj = min((r["min_scaled_jacobian"] for r in built), default=float("nan"))
            gap = max((r["max_seam_gap_mm"] for r in built), default=float("nan"))
            refused = [r for r in rows if not r["built"]]
            print(f"    {cfg:7s} {len(ok)}/{len(rows)} cells valid AND closed, worst min "
                  f"scaled J {sj:.4f} (MIN_SJ_TARGET is 0.2), worst seam gap {gap:.2e} mm"
                  + (f", {len(refused)} refused geometrically" if refused else ""))
        print("\n  WHAT BOUNDS THE RADIUS IS NOW THE SECTOR, NOT THE BLOCK")
        for cfg, per in sec["per_config"].items():
            for junction, lim in per["fit_limit"].items():
                if lim["limited"]:
                    print(f"    {cfg:7s} {junction:4s} the ring's free block runs out at "
                          f"R = {lim['radius_mm']:.4f} mm -- the tangent point reaches "
                          f"the NEXT sector's corner")
                else:
                    print(f"    {cfg:7s} {junction:4s} fits the sector at every radius "
                          f"swept")
        print("\n  WHY THE SHALLOW CUT CANNOT CLOSE: THE OFFSET LANDS TANGENT")
        for cfg, per in sec["per_config"].items():
            for junction, row in per["landing"].items():
                if not row.get("landing_angle_deg"):
                    continue
                print(f"    {cfg:7s} {junction:4s} PART 9's cut of "
                      f"{row['cut_depth_mm']:.4f} mm lands on the circle it stops at "
                      f"at {row['landing_angle_deg']:6.3f} deg -- a sliver whose scaled "
                      f"Jacobian is {row['sliver_scaled_jacobian']:.4f}")
        print("\n  THE INNER EDGE'S TWO CONSTANTS, RE-DERIVED (worst block over the "
              f"box, at {cfg0})")
        cfgp = sec["per_config"][cfg0]["profile"] or []
        ends = sorted({r["end"] for r in cfgp})
        print("    entry \\ end " + "".join(f"{e:>10.2f}" for e in ends))
        for entry in sorted({r["entry"] for r in cfgp}, reverse=True):
            cells = []
            for e in ends:
                r = next(x for x in cfgp if x["entry"] == entry and x["end"] == e)
                cells.append("    FAILS" if r["cells_failed"]
                             else f"{r['worst_min_scaled_jacobian']:10.4f}")
            star = " <- chosen" if abs(entry - sec["entry_slope"]) < 1e-12 else ""
            print(f"    {entry:11.2f} " + "".join(cells) + star)
        print(f"    the chosen pair is entry {sec['entry_slope']:.2f}, end "
              f"{sec['end_offset']:.2f}")
        print("\n  THE SAME TWO CONSTANTS, RE-DERIVED AGAINST GENOMES RATHER THAN A "
              f"RADIUS BOX (at {cfg0})")
        cfgg = sec["per_config"][cfg0]["profile_genomes"] or []
        if cfgg:
            n_gen = cfgg[0]["n_genomes"]
            gends = sorted({r["end"] for r in cfgg})
            print(f"    {n_gen} genomes (the spoke-fold one excluded -- neither "
                  "constant reaches it)")
            print("    entry \\ end " + "".join(f"{e:>10.2f}" for e in gends))
            for entry in sorted({r["entry"] for r in cfgg}, reverse=True):
                cells = []
                for e in gends:
                    r = next(x for x in cfgg if x["entry"] == entry and x["end"] == e)
                    cells.append(f"{'REFUSE':>10s}" if r["worst_min_scaled_jacobian"]
                                 is None else f"{r['worst_min_scaled_jacobian']:10.4f}")
                star = " <- chosen" if abs(entry - sec["entry_slope"]) < 1e-12 else ""
                print(f"    {entry:11.2f} " + "".join(cells) + star)
            best = max(cfgg, key=lambda r: (r["worst_min_scaled_jacobian"]
                                            if r["worst_min_scaled_jacobian"] is not None
                                            else -9.0))
            print(f"    argmax over genomes: entry {best['entry']:.2f}, end "
                  f"{best['end']:.2f}, worst min scaled J {best['worst_min_scaled_jacobian']:.4f}")

        pam = sec["per_config"][cfg0].get("profile_argmax")
        newg = sec["per_config"][cfg0].get("profile_genomes_buildable") or []
        if pam and newg:
            print("\n  AND RE-DERIVED AGAINST THE CELLS §57 AND §58 ADDED — EVERY DRAWN "
                  "GENOME CLAMPED INSIDE ITS")
            print("  OWN SECTOR'S ROOM, MINUS THE ONES WHOSE SPOKE DOES NOT EXIST")
            gends = sorted({r["end"] for r in newg})
            print(f"    {newg[0]['n_genomes']} cells, against the "
                  f"{cfgg[0]['n_genomes'] if cfgg else '?'} the pair was fitted to")
            print("    entry \\ end " + "".join(f"{e:>10.2f}" for e in gends)
                  + "   refused")
            for entry in sorted({r["entry"] for r in newg}, reverse=True):
                cells, ref = [], []
                for e in gends:
                    r = next(x for x in newg if x["entry"] == entry and x["end"] == e)
                    cells.append(f"{'REFUSE':>10s}" if r["worst_min_scaled_jacobian"]
                                 is None else f"{r['worst_min_scaled_jacobian']:10.4f}")
                    ref.append(r["refused"])
                print(f"    {entry:11.2f} " + "".join(cells)
                      + "   " + "".join(f"{x:2d}" for x in ref))
            for rule in ("over_built", "no_refusal"):
                b = pam["buildable"][rule]
                if b is None:
                    continue
                print(f"    argmax, {rule:11s}: entry {b['entry']:+.2f}, end "
                      f"{b['end']:.2f}, worst {b['worst_min_scaled_jacobian']:.4f}, "
                      f"refused {b['refused']}, bound by {b['worst_at'][2]}")
            print("    THE TWO RULES DISAGREE HERE AND THEY DID NOT BEFORE.  Ranking on "
                  "the worst of the genomes")
            print("    that BUILT pays a cell for refusing a hard one, and the clamped "
                  "cells reach the steep-entry")
            print("    corner where the layer profile itself starts refusing — which the "
                  "ten-cell set never did.")
            p13 = pam["part_13"]
            print(f"    on PART 13's own grid the two rules AGREE "
                  f"({p13['rules_agree']}), so its answer was never biased.")
            nr = pam["buildable"]["no_refusal"]
            if nr is not None:
                print(f"    and on the corrected rule the re-derivation REPRODUCES that "
                      f"answer: entry {nr['entry']:+.2f}, end {nr['end']:.2f} — "
                      f"§54's pair, at {nr['worst_min_scaled_jacobian']:.4f} against its "
                      f"published {p13['no_refusal']['worst_min_scaled_jacobian']:.4f}.")
            print("    so the argmax is NOT what is stale about §54's decision, and the "
                  "whole of what remains")
            print("    on it is the convergence cost measured at the shipped genome.")
            pr = sec["per_config"][cfg0].get("profile_candidates")
            if pr:
                print(f"    every cell that clears {pr['target']:.2f} on all "
                      f"{newg[0]['n_genomes']} and refuses none — the set "
                      f"`make corner-fillet` prices against the band:")
                print("      " + "  ".join(f"({e:+.2f}, {n:.2f})"
                                           for e, n in pr["measured"]))
                print(f"      constant in this module matches the surface: "
                      f"{pr['measured'] == pr['constant']}")
            # PART 20 / PLAN §68's cliff, as the column that section said no table had.
            for key, label in (("profile_candidate_rows", "buildable grid"),
                               ("profile_candidate_rows_fine", "fine grid")):
                rows = sec["per_config"][cfg0].get(key)
                if not rows:
                    continue
                sh = sec["per_config"][cfg0].get("shipped_profile_cliff") or {}
                print(f"\n    HOW FAR EACH CANDIDATE STANDS FROM THE SHIPPED GENOME'S "
                      f"RIM-LAYER CLIFF ({label})")
                print(f"      {'entry':>6s} {'end':>5s} {'box floor':>10s} "
                      f"{'cliff entry':>12s} {'margin':>8s}")
                for r in sorted(rows, key=lambda r: -(r["cliff_margin"] or -1e9)):
                    m = ("     -" if r["cliff_margin"] is None
                         else f"{r['cliff_margin']:8.4f}")
                    ce = ("       -" if r["cliff_entry"] is None
                          else f"{r['cliff_entry']:12.6f}")
                    print(f"      {r['entry']:+6.2f} {r['end']:5.2f} "
                          f"{r['worst_min_scaled_jacobian']:10.4f} {ce} {m}")
                if sh.get("cliff_margin") is not None:
                    best = max((r["cliff_margin"] for r in rows
                                if r["cliff_margin"] is not None), default=None)
                    print(f"      the SHIPPED pair ({sh['entry']:+.2f}, {sh['end']:.2f}) "
                          f"stands {sh['cliff_margin']:.4f} from it"
                          + ("" if best is None else
                             f" — {sh['cliff_margin'] / best:.1f}x the best candidate's "
                             f"{best:.4f}"))
        print("\n  AND AGAINST THE RIM'S UNCAP BLEND, WHICH IS THE TRI-BLOCK'S QUESTION")
        print(f"    {'config':7s} {'rim blend':>9s} {'unfilleted worst':>17s} "
              f"{'filleted worst':>15s} {'worst block':>14s}")
        for cfg, per in sec["per_config"].items():
            for row in per["uncap"]:
                got = (f"{row['min_scaled_jacobian']:15.6f}" if "min_scaled_jacobian"
                       in row else f"{'REFUSED':>15s}")
                print(f"    {cfg:7s} {row['rim_blend']:9.2f} "
                      f"{row.get('control_min_scaled_jacobian', float('nan')):17.6f} "
                      f"{got} {row.get('worst_block', '-'):>14s}")
        print("    the faithful rim (blend 0.00) is NOT rescued by the re-cut and is made "
              "worse by it:")
        print("    what collapses is a corner opening to 180 deg at `far_end`, and the "
              "filleted junction block is shorter, so it dominates more of it.")

        gs = sec["genomes"]
        print("\n  AND AT OTHER GENOMES, WHICH IS WHAT THE RADIUS BOX ABOVE IS NOT")
        print(f"    {'orientation':13s} {'n':>3s} {'refused':>8s} {'seams close':>12s} "
              f"{'min scaled J (built)':>21s} {'clear MIN_SJ_TARGET':>20s}")
        allrows = []
        for key, rows in gs["groups"].items():
            allrows += rows
            built = [r for r in rows if r["built"]]
            sjs = [r["min_scaled_jacobian"] for r in built]
            rng = (f"{min(sjs):.4f} - {max(sjs):.4f}" if sjs else "-")
            print(f"    {key:13s} {len(rows):3d} {len(rows) - len(built):8d} "
                  f"{str(all(r['seams_close'] for r in built)):>12s} {rng:>21s} "
                  f"{sum(1 for x in sjs if x > 0.2):>13d}/{len(sjs):<6d}")
        built = [r for r in allrows if r["built"]]
        sjs = [r["min_scaled_jacobian"] for r in built]
        print(f"    ALL           {len(allrows):3d} {len(allrows) - len(built):8d} "
              f"{str(all(r['seams_close'] for r in built)):>12s} "
              f"{min(sjs):.4f} - {max(sjs):.4f}".ljust(70)
              + f"{sum(1 for x in sjs if x > 0.2):>7d}/{len(sjs):<6d}")
        print("    every REFUSAL is the same one: the hub fillet's tangent point has "
              "passed the next sector's corner")
        print("    -- so the blocking is measured for STEP 2 (one genome, one mesh) and "
              "is NOT yet fit for the OPTIMIZER, which sweeps genomes.")

        fc = sec["per_config"][cfg0].get("fit_clamp")
        if fc:
            print("\n  AND THE REFUSAL IS PREDICTED BY A NUMBER, NOT ONLY EXPLAINED "
                  f"AFTERWARDS (at {cfg0})")
            print(f"      {'orientation':13s} {'R_hub':>7s} {'limit':>8s} "
                  f"{'margin':>8s} {'binds':>6s} {'built':>6s}")
            wrong = 0
            for r in allrows:
                f = r["fit"]["hub"]
                lim = "none" if f["limit_mm"] is None else f"{f['limit_mm']:.4f}"
                mar = "-" if f["margin_mm"] is None else f"{f['margin_mm']:+.4f}"
                wrong += int(f["binds"] == r["built"])
                print(f"      {str(r['orientation']):13s} {f['R_mm']:7.4f} {lim:>8s} "
                      f"{mar:>8s} {str(f['binds']):>6s} {str(r['built']):>6s}")
            print(f"    the hub margin classifies {len(allrows) - wrong}/{len(allrows)}: "
                  "every genome whose own R_hub exceeds its own sector-fit limit refuses, "
                  "and no other does.")
            sfit = fc["shipped_fit"]
            print(f"    the SHIPPED genome is not near it — hub {sfit['hub']['R_mm']:.4f} "
                  f"against a limit of "
                  + ("none" if sfit["hub"]["limit_mm"] is None
                     else f"{sfit['hub']['limit_mm']:.4f}")
                  + f", rim {sfit['rim']['R_mm']:.4f} against "
                  + ("none" if sfit["rim"]["limit_mm"] is None
                     else f"{sfit['rim']['limit_mm']:.4f}")
                  + f" — so the clamp is inert on it: {not fc['shipped_is_clamped']}")

            print("\n  SO WHAT DOES CLAMPING EACH RADIUS INSIDE ITS OWN SECTOR'S ROOM BUY?")
            print(f"      {'profile':14s} {'clamp':>6s} {'built':>7s} {'clears 0.2':>11s} "
                  f"{'clamped h/r':>12s} {'min scaled J range':>21s} {'median':>8s}")
            for row in fc["rows"]:
                k = "none" if row["factor"] is None else f"{row['factor']:.2f}"
                rng = ("-" if row["min_scaled_jacobian_range"] is None else
                       f"{row['min_scaled_jacobian_range'][0]:+.4f}..."
                       f"{row['min_scaled_jacobian_range'][1]:+.4f}")
                print(f"      {row['profile']:14s} {k:>6s} "
                      f"{row['n_built']:4d}/{row['n_genomes']:<2d} "
                      f"{row['n_clears_target']:8d}/{row['n_genomes']:<2d} "
                      f"{row['n_clamped']['hub']:6d}/{row['n_clamped']['rim']:<5d} "
                      f"{rng:>21s} {row['median_min_scaled_jacobian']:8.4f}")
            print("    the clamp is the REFUSAL half of PLAN.md's item 2 and it closes "
                  "it: every drawn genome builds.")
            print("    it is a FIX and not a gate — it models a smaller fillet than the "
                  "genes asked for, which is honest")
            print("    for an instrument sweeping the box and honest for an optimizer "
                  "only if the objective is told")
            print("    the clamped radius.  The `binds` column above is the gate, and it "
                  "costs nothing.")
            print("    MEASURED, NOT ADOPTED: `sector_blocks` and `build_wheel` are "
                  "untouched and still take the")
            print("    radii they are given.")

        # PLAN §78: the same table on genomes the clamp was not designed against, printed
        # BESIDE the in-sample one rather than in its own section, because the only reading
        # of either number that means anything is the comparison.
        ho = sec["per_config"][cfg0].get("fit_clamp_held_out")
        if ho and fc:
            print("\n  AND THE SAME TABLE ON A HELD-OUT DRAW (PLAN §78) — DISJOINT STREAM, "
                  "SAME SAMPLER AND FILTER")

            def _row(table, profile, factor):
                return next((r for r in table["rows"] if r["profile"] == profile
                             and r["factor"] == factor), None)

            print(f"      {'profile':14s} {'clamp':>6s} {'in-sample':>18s} "
                  f"{'held-out':>18s}")
            for profile in ("shipped", "genome_robust", "margin_robust"):
                for factor in (None, SECTOR_FIT_CLAMP):
                    a, b = _row(fc, profile, factor), _row(ho, profile, factor)
                    if not a or not b:
                        continue
                    k = "none" if factor is None else f"{factor:.2f}"
                    print(f"      {profile:14s} {k:>6s} "
                          f"{a['n_built']:6d}/{a['n_genomes']:<3d} built"
                          f"{b['n_built']:7d}/{b['n_genomes']:<3d} built")
                    print(f"      {'':14s} {'':>6s} "
                          f"{a['n_clears_target']:6d}/{a['n_genomes']:<3d} clear"
                          f"{b['n_clears_target']:7d}/{b['n_genomes']:<3d} clear")
            hb = _row(ho, "shipped", SECTOR_FIT_CLAMP)
            if hb:
                print(f"    the clamp builds {hb['n_built']}/{hb['n_genomes']} of the "
                      f"held-out box at the shipped profile.")
                if hb["refusals"]:
                    print(f"    AND IT DOES NOT CLOSE THE REFUSAL HALF OUT OF SAMPLE — "
                          f"{len(hb['refusals'])} refusals remain: "
                          + "; ".join(sorted(set(hb["refusals"]))))

        # And the barrier half, priced against a PER-GENOME entry rather than a global pair.
        for key, label in (("cliff_profile", "in-sample"),
                           ("cliff_profile_held_out", "held-out")):
            cp = sec["per_config"][cfg0].get(key)
            if not cp:
                continue
            sc = cp["shipped_cliff"]
            print(f"\n  THE BARRIER HALF, AGAINST A PER-GENOME ENTRY ({label} box, "
                  f"end {cp['end']:.2f})")
            print(f"      {'factor':>6s} {'built':>8s} {'clears 0.2':>11s} "
                  f"{'no edge':>8s} {'n/a':>4s} {'shipped entry':>14s} "
                  f"{'shipped margin':>15s} {'median J':>9s}")
            for r in cp["rows"]:
                se = ("      -" if r["shipped_entry"] is None
                      else f"{r['shipped_entry']:+14.4f}")
                sm = ("      -" if r["shipped_margin"] is None
                      else f"{r['shipped_margin']:15.4f}")
                mj = ("        -" if r["median_min_scaled_jacobian"] is None
                      else f"{r['median_min_scaled_jacobian']:9.4f}")
                print(f"      {r['factor']:6.2f} {r['n_built']:5d}/{r['n_genomes']:<2d} "
                      f"{r['n_clears_target']:8d}/{r['n_genomes']:<2d} "
                      f"{r['n_without_cliff']:8d} {r['n_unevaluable']:4d} "
                      f"{se} {sm} {mj}")
            if sc["entry"] is not None:
                print(f"    the shipped genome's own cliff at this `end` is "
                      f"{sc['entry']:+.6f}; the SHIPPED PAIR "
                      f"({LAYER_ENTRY_SLOPE:+.2f}, {LAYER_END_OFFSET:.2f}) stands 0.5520 "
                      f"from its own (§68),")
                print(f"    and the GLOBAL pair that buys the barrier half "
                      f"({GENOME_ROBUST_ENTRY:+.2f}, {GENOME_ROBUST_END:.2f}) stands "
                      f"{GENOME_ROBUST_ENTRY - sc['entry']:+.4f} from it.")
            print("    MEASURED, NOT ADOPTED: no module constant moves and this rule is "
                  "not wired into anything.")

        fg = sec.get("fold_gate")
        if fg:
            c = fg["counts"]
            print("\n  AND THE OTHER FEASIBILITY NUMBER: DOES THE SPOKE THE FILLET TRIMS "
                  "EVEN EXIST?")
            print(f"      {'genome':13s} {'fold margin':>12s} {'folds':>6s} "
                  f"{'binds':>6s} {'built':>6s} {'worst block':>14s} {'min scaled J':>13s}")
            for r in allrows:
                f = r["fold"]
                sj = ("-" if r.get("min_scaled_jacobian") is None
                      else f"{r['min_scaled_jacobian']:+.4f}")
                print(f"      {str(r['orientation']):13s} {f['margin_mm']:+12.4f} "
                      f"{str(f['folds']):>6s} {str(f['binds']):>6s} "
                      f"{str(r['built']):>6s} {r.get('worst_block', '-'):>14s} {sj:>13s}")
            folded = [r for r in allrows if r["fold"]["folds"]]
            inverted = [r for r in allrows if r["built"] and not r["all_blocks_valid"]]
            print(f"    {len(folded)}/{len(allrows)} describe a part that does not exist; "
                  f"{len(inverted)}/{len(allrows)} invert a block, and every one of those "
                  "is in the first set.")
            print("    the converse does NOT hold and is not claimed: whether a folded "
                  "flank SHOWS as an inverted")
            print("    element depends on whether a station lands on the dip, which is a "
                  "property of the grid.")
            print("    that is the whole argument for the closed form -- the mesh-based "
                  "filter this box was drawn")
            print(f"    through is a proxy for it, and over {c['drawn']} draws on the "
                  "same stream it leaks:")
            print(f"      pass (x_order, hub_overlap)             {c['geom']:6d}   "
                  f"folded {c['geom_folds']:5d}  ({100 * fg['fold_rate']:.1f}%)")
            print(f"      AND the unfilleted sector meshes clean  {c['mesh_clean']:6d}   "
                  f"folded {c['mesh_clean_folds']:5d}  ({100 * fg['leak_rate']:.1f}%)"
                  "   <- what the draw filter misses")
            ag, rs = fg["agreement"], fg["resolution"]
            print(f"    audited against the SAMPLED flank on all {ag['n']} of them, both "
                  f"at {ag['at_points']} points: {ag['disagreements']} disagreement(s), "
                  f"worst at |margin| = {ag['worst_disagreement_margin_mm']:.2e} mm")
            print(f"    against a barrier of {fg['limit_mm']:.2f} mm — the two are the "
                  "same statement except within a micron of the fold point.")
            print(f"    AND THAT QUALIFIER IS THE POINT.  Recompute both at the config's "
                  f"own {rs['config_points']} points instead:")
            print(f"      the closed form moves by at most "
                  f"{rs['closed_form_worst_shift_mm']:.2e} mm — a hundredth of the "
                  f"barrier — and flips {rs['closed_form_verdict_flips']} verdict(s)")
            print(f"      the sampled flank MISSES "
                  f"{rs['sampled_flank_missed_folds']}/{c['geom_folds']} of the folds "
                  + ("" if rs["sampled_flank_miss_rate"] is None
                     else f"({100 * rs['sampled_flank_miss_rate']:.1f}%) ")
                  + "outright — the dip falls between two stations")
            print("    those are not two error bars on the same thing: one is a "
                  "converging number and the other is")
            print("    a classification miss.  Read it on the rejected genomes "
                  "themselves:")
            for lad in fg.get("ladder", []):
                head = (f"{str(lad['orientation'])} R_hub {lad['R_hub_mm']:.4f}"
                        + (f", worst block {lad['worst_block']}"
                           if lad["worst_block"] else " (refused)"))
                print(f"      {head}")
                print(f"        {'points':>7s} {'closed form':>13s} "
                      f"{'sampled flank':>15s}")
                for rung in lad["rungs"]:
                    note = ("   <- PART 13's grid" if rung["num_points"] == 97 else "")
                    print(f"        {rung['num_points']:7d} {rung['margin_mm']:+13.6f} "
                          f"{rung['flank_reversal_mm']:+15.3e}{note}")
            print("    the closed form makes the same call at every rung including the "
                  "shipped one; the sampled")
            print("    statement changes sides.  That is the whole argument, and it is "
                  "the same failure the draw")
            print("    filter has one level up.  A gate has to be the first kind of "
                  "number.")
            sh = fg["shipped"]
            print(f"    the SHIPPED genome is nowhere near it — margin "
                  f"{sh['margin_mm']:.4f} mm against a limit of {sh['limit_mm']:.2f} — so "
                  f"the gate is inert on it: {not sh['binds']}")

            fcc = sec["per_config"][cfg0].get("fit_clamp_fold_clean")
            if fcc and fc:
                print("\n  AND WHAT THE ARC'S OWN TABLE LOOKS LIKE OVER PARTS THAT EXIST")
                print(f"      {'profile':14s} {'clamp':>6s} {'all 16':>13s} "
                      f"{'fold-clean':>13s}")
                by = {(r["profile"], r["factor"]): r for r in fcc["rows"]}
                for row in fc["rows"]:
                    k = "none" if row["factor"] is None else f"{row['factor']:.2f}"
                    o = by.get((row["profile"], row["factor"]))
                    if o is None:
                        continue
                    print(f"      {row['profile']:14s} {k:>6s} "
                          f"{row['n_built']:2d}/{row['n_genomes']:<2d} built "
                          f"{row['n_clears_target']:2d} clear   "
                          f"{o['n_built']:2d}/{o['n_genomes']:<2d} built "
                          f"{o['n_clears_target']:2d} clear")
                print("    the gate costs the box two genomes and buys back the one "
                      "defect no profile and no clamp")
                print("    reaches — §54's trimmed-spoke fold, which was never this "
                      "blocking's to fix.")

        print("\n  THE NODE COUNT IT FORCES")
        for cfg, per in sec["per_config"].items():
            nc = per["node_counts"]
            print(f"    {cfg:7s} the ring blocks' radial count becomes n_thick = "
                  f"{nc['ring_radial_used']}, not n_collar_r = {nc['n_collar_r']} / "
                  f"n_rim_r = {nc['n_rim_r']}")

    sc = rec["self_checks"]
    print("\n  SELF-CHECKS")
    for k, v in sc.items():
        if k == "pass":
            continue
        print(f"    {k:40s} {'PASS' if v else 'FAIL'}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genome", default="best_solution.json")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    ap.add_argument("--junctions", default=",".join(DEFAULT_JUNCTIONS))
    ap.add_argument("--out", default="study_fillet_block.json")
    args = ap.parse_args()

    configs = tuple(c for c in args.configs.split(",") if c)
    junctions = tuple(j for j in args.junctions.split(",") if j)
    _gate_guard.refuse_degraded_out(ap, args, "study_fillet_block.json", [
        (set(configs) != set(DEFAULT_CONFIGS),
         f"--configs {args.configs} is not the committed "
         f"{','.join(DEFAULT_CONFIGS)}"),
        (set(junctions) != set(DEFAULT_JUNCTIONS),
         f"--junctions {args.junctions} is not the committed "
         f"{','.join(DEFAULT_JUNCTIONS)}"),
        (args.genome != "best_solution.json",
         f"--genome {args.genome} is not the shipped genome"),
    ])

    t0 = time.time()
    genes = load_genes(args.genome)
    rec = build(genes, configs, junctions, radius_grid(genes), REFINEMENTS)
    rec["genome"] = args.genome
    rec["seconds"] = round(time.time() - t0, 2)
    out = os.path.join(HERE, args.out)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
    _print(rec)
    print(f"  wrote {out}  ({rec['seconds']} s)")
    raise SystemExit(0 if rec["self_checks"]["pass"] else 1)


if __name__ == "__main__":
    main()
