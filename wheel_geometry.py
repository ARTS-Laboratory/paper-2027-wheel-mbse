"""
=============================================================================
  COMPLIANT WHEEL — GEOMETRY KERNEL (framework-agnostic)
=============================================================================
One implementation of the spoke geometry, usable from both the numpy world
(`wheel_fea.py`, the GA, the CadQuery exporter) and the JAX world (the mesh
generator and the differentiable FEA).

HOW THE DUAL BACKEND WORKS
--------------------------
Every function takes an array module `xp`, defaulting to numpy:

    curve, ctrl = bezier_centerline(*genes8)                   # numpy
    curve, ctrl = bezier_centerline(*genes8, xp=jax.numpy)     # traced by JAX

The numpy API subset used here — `stack`, `linspace`, `sqrt`, `cumsum`, `clip`,
`gradient`, `matmul`, `linalg.norm` — is spelled identically in `jax.numpy`, so a
single body serves both.  Crucially **this module never imports jax**: doing so would
drag it into `wheel_fea.py`'s import graph, and from there into the CadQuery
environment, which has no jax.  `tests/test_import_hygiene.py` enforces that.

WHY THESE FUNCTIONS LOOK DIFFERENT FROM THE ORIGINALS
-----------------------------------------------------
Three deliberate changes, each of which is verified equivalent by a test rather than
asserted here:

1. `bernstein_matrix` is CACHED.  `wheel_fea.generate_bezier_centerline` rebuilt the
   600x6 basis on every single fitness evaluation, but it does not depend on the genes
   at all — only on `num_points`.  Memoising it is bit-identical and, measured, removes
   21% of a fitness evaluation (0.051 ms of 0.248 ms), or about 4.6 s from a full
   300x300 GA run.

2. `thickness_at_arc_length` is BRANCH-FREE.  The original selected each of the three
   taper zones with a boolean mask and wrote through it (`thickness[mask] = ...`).
   Boolean-mask assignment has no JAX equivalent — the output shape is data-dependent —
   so the same piecewise-linear function is expressed as a sum of clipped ramps.  See
   that function's docstring for the algebra.

3. `offset_band` GENERALISES `thicken_3taper_curve` with a transverse index, so the
   same parameterisation that draws the spoke outline also lays out the structured
   quad mesh through its thickness.  At `n_across=1` it reproduces the original
   outline exactly.

WHAT IS DELIBERATELY *NOT* CHANGED
----------------------------------
The surface normals still come from `xp.gradient` of the sampled centerline, not from
the exact Bezier hodograph, even though the analytic version is available here as
`bezier_tangent`.  The finite-difference normals are what `wheel_step_export.py` used
to build the STEP currently on disk, and switching would silently move every spoke
flank by O(h^2) at the endpoints — precisely where the fillets attach.  `offset_band`
takes a `normals=` argument so the mesh can opt into the analytic ones later, as a
measured change rather than a side effect.
=============================================================================
"""

import functools
import math

import numpy as np

# Degree of the centerline Bezier.  6 control points: P0 and P5 locked at the hub and
# rim, P1..P4 evolvable (8 genes).
BEZIER_DEGREE = 5

# Normalised arc-length breakpoints of the three linear-taper zones.
TAPER_BREAKPOINTS = np.array([0.0, 1 / 3, 2 / 3, 1.0])


# ---------------------------------------------------------------------------
# BERNSTEIN BASIS
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def bernstein_matrix(num_points, degree=BEZIER_DEGREE):
    """The (num_points, degree+1) Bezier basis, evaluated on a uniform parameter grid.

    Independent of the genes, so it is computed once per (num_points, degree) and
    reused.  Always float64 numpy: it is a constant that both backends multiply
    against, and JAX will convert it on first use.
    """
    t = np.linspace(0.0, 1.0, num_points)[:, None]
    n = degree
    return np.array([
        float(math.comb(n, k)) * (1 - t) ** (n - k) * t ** k
        for k in range(n + 1)
    ]).squeeze(axis=-1).T


@functools.lru_cache(maxsize=16)
def bernstein_derivative_matrix(num_points, degree=BEZIER_DEGREE, order=1):
    """Basis for the `order`-th derivative curve (the hodograph).

    d/du of a degree-n Bezier is a degree-(n-1) Bezier over the forward differences of
    the control points, scaled by n.  Returns the (num_points, degree+1-order) basis;
    combine it with `forward_difference_matrix` to map control points to derivatives.
    """
    return bernstein_matrix(num_points, degree - order)


@functools.lru_cache(maxsize=16)
def forward_difference_matrix(degree=BEZIER_DEGREE, order=1):
    """(degree+1-order, degree+1) matrix D with  D @ P = scaled control differences,
    such that the `order`-th derivative curve is `bernstein(deg-order) @ (D @ P)`."""
    D = np.eye(degree + 1)
    n = degree
    for _ in range(order):
        D = n * (D[1:] - D[:-1])
        n -= 1
    return D


# ---------------------------------------------------------------------------
# CENTERLINE
# ---------------------------------------------------------------------------

def control_points(cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4, span_mm, xp=np):
    """The 6 control points as a (6, 2) array, in the spoke's local frame.

    P0 = (0, 0) at the hub and P5 = (span, 0) at the rim are locked; only the four
    interior points are genes.  Built with `stack` rather than an array literal so the
    coordinates can be JAX tracers.
    """
    zero = xp.zeros((), dtype=float)
    def row(a, b):
        return xp.stack([zero + a, zero + b])
    return xp.stack([
        row(0.0, 0.0),
        row(cx1, cy1),
        row(cx2, cy2),
        row(cx3, cy3),
        row(cx4, cy4),
        row(span_mm, 0.0),
    ])


def bezier_centerline(cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4,
                      span_mm, num_points, xp=np):
    """Degree-5 Bezier centerline.  Returns (curve [N,2], control_polygon [6,2]).

    Linear in the genes: `curve = B @ P`, with B constant.  That linearity is why the
    gradient path from genes to node positions is cheap.
    """
    pts = control_points(cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4, span_mm, xp=xp)
    B = xp.asarray(bernstein_matrix(num_points))
    return B @ pts, pts


def bezier_tangent(ctrl_pts, num_points, xp=np, normalize=True):
    """Exact analytic tangent of the centerline, from the hodograph.

    Preferred over differencing the sampled curve wherever the value matters at the
    ENDS: `xp.gradient` is one-sided there and carries O(h^2) error at exactly the two
    stations where the fillets attach.  Not used by the default outline path — see the
    module docstring for why that stays on finite differences.
    """
    D = xp.asarray(forward_difference_matrix(BEZIER_DEGREE, 1))
    Bd = xp.asarray(bernstein_matrix(num_points, BEZIER_DEGREE - 1))
    d = Bd @ (D @ ctrl_pts)
    if not normalize:
        return d
    return d / (xp.linalg.norm(d, axis=1, keepdims=True) + 1e-12)


def bezier_curvature(ctrl_pts, num_points, xp=np):
    """Signed curvature kappa(s) of the centerline, analytically.

        kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^{3/2}

    The mesh needs this: the offset band self-intersects where the radius of curvature
    |1/kappa| drops below half the local thickness, which is the dominant way a genome
    produces an inverted element.  `wheel_fea`'s `smoothness` loss term was an implicit
    and rather indirect guard against the same thing (wheel_fea.py:596-598).
    """
    D1 = xp.asarray(forward_difference_matrix(BEZIER_DEGREE, 1))
    D2 = xp.asarray(forward_difference_matrix(BEZIER_DEGREE, 2))
    d1 = xp.asarray(bernstein_matrix(num_points, BEZIER_DEGREE - 1)) @ (D1 @ ctrl_pts)
    d2 = xp.asarray(bernstein_matrix(num_points, BEZIER_DEGREE - 2)) @ (D2 @ ctrl_pts)
    num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5
    return num / (den + 1e-30)


# ---------------------------------------------------------------------------
# ARC LENGTH
# ---------------------------------------------------------------------------

def segment_lengths(curve, xp=np):
    d = curve[1:] - curve[:-1]
    return xp.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)


def arc_fractions(curve, xp=np, at="nodes"):
    """Normalised cumulative arc length in [0, 1].

    `at="nodes"`  -> length N,   one value per curve point   (outline / thickening)
    `at="midpoints"` -> length N-1, one per segment          (beam integration)

    Both exist because the two callers genuinely need different ones, and conflating
    them silently shifts the taper by half a segment: `thicken_3taper_curve` samples
    thickness at nodes (wheel_fea.py:746) while `generalized_spoke_mechanics` samples
    at segment midpoints (wheel_fea.py:355).
    """
    seg = segment_lengths(curve, xp=xp)
    cum = xp.concatenate([xp.zeros(1, dtype=seg.dtype), xp.cumsum(seg)])
    total = cum[-1] + 1e-12
    if at == "nodes":
        return cum / total
    if at == "midpoints":
        return (cum[:-1] + seg / 2.0) / total
    raise ValueError(f"at must be 'nodes' or 'midpoints', got {at!r}")


# ---------------------------------------------------------------------------
# THICKNESS
# ---------------------------------------------------------------------------

def thickness_at_arc_length(s, t0, t1, t2, t3, xp=np):
    """Piecewise-linear thickness over three taper zones, branch-free.

    The original (wheel_fea.py:252) selected each zone with a boolean mask and wrote
    through it.  That has no JAX equivalent — `s[mask]` has a data-dependent shape —
    so the same function is written as a base value plus three clipped ramps:

        t(s) = t0 + SUM_i (t_{i+1} - t_i) * clip((s - bp_i) / w_i, 0, 1)

    Equivalence: for s in zone i, every ramp j < i is saturated at 1 and contributes
    its full (t_{j+1} - t_j); ramp i contributes (t_{i+1} - t_i)*alpha; ramps j > i are
    clipped to 0.  The saturated terms telescope to (t_i - t0), so the sum collapses to
    t_i + (t_{i+1} - t_i)*alpha — the original lerp.

    NOTE ON THE ORIGINAL'S EPSILON.  The masked version divided by
    `bp[i+1] - bp[i] + 1e-12`.  That guard protects nothing — the zone widths are the
    compile-time constants 1/3 and can never be zero — and it has a real cost: the ramp
    then saturates at 1 - 3e-12 instead of 1, so the taper misses its own interpolation
    nodes by up to 1.2e-11 and `t(1/3) != t1`.  The original concealed that because its
    zone masks were inclusive at BOTH ends, so a point exactly on a breakpoint was
    written twice and the second write (alpha = 0, the exact node) happened to win.

    The epsilon is dropped here, which makes this function the exact piecewise-linear
    interpolant — verified against `np.interp` — and leaves it differing from the old
    masked version by ~3e-12 relative in zone interiors.  That is 5 orders below OCC's
    1e-7 mm modelling tolerance and 3 below the noise already present between two runs
    of the STEP exporter, so no downstream artifact can see it.

    The `1e-12` in `arc_fractions` is a different guard and is kept: a degenerate genome
    really can produce a zero-length curve.
    """
    bp = TAPER_BREAKPOINTS
    nodes = [t0, t1, t2, t3]
    out = xp.zeros_like(xp.asarray(s, dtype=float)) + nodes[0]
    for i in range(3):
        ramp = xp.clip((s - bp[i]) / (bp[i + 1] - bp[i]), 0.0, 1.0)
        out = out + (nodes[i + 1] - nodes[i]) * ramp
    return out


# ---------------------------------------------------------------------------
# OFFSET BAND  (outline, and the spoke's structured mesh block)
# ---------------------------------------------------------------------------

def offset_normals(curve, xp=np):
    """Unit normals from finite differences of the sampled centerline.

    This is exactly what `thicken_3taper_curve` does (wheel_fea.py:750-752), kept
    bit-for-bit because the STEP on disk was built from it.

    NOT suitable for the mesh — see `normals_from_tangents`.  `xp.gradient` is
    one-sided at the two endpoints, so the normal there carries O(h) error and the
    resulting flank position MOVES as the sampling is refined: measured, the top-flank
    tip shifts 8.9 um between n_span 32 and 64, halving on each refinement.  Harmless
    for a 600-point outline; fatal for a mesh-refinement study, where the geometry must
    hold still while only the discretization changes.
    """
    grads = xp.gradient(curve, axis=0)
    n = xp.stack([-grads[:, 1], grads[:, 0]], axis=1)
    return n / (xp.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def normals_from_tangents(tangents, xp=np):
    """Unit normals from exact tangent vectors: rotate each by +90 degrees.

    Resolution-independent, because the tangents come from the Bezier hodograph rather
    than from differencing the samples.  Same sign convention as `offset_normals`
    (n = (-t_y, t_x)), so `offset_band`'s +eta column stays the "top" flank.
    """
    n = xp.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    return n / (xp.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def offset_band(curve, t0, t1, t2, t3, n_across=1, xp=np, normals=None):
    """Node positions of the spoke, swept across its thickness.

        x(s, eta) = C(s) + eta * (t(s)/2) * N_hat(s),    eta in linspace(-1, 1, n_across+1)

    Returns `[N, n_across+1, 2]`.  Column 0 is the -N (bottom) flank and column -1 is
    the +N (top) flank, so `band[:, -1], band[:, 0]` is the `(top, bot)` pair that
    `thicken_3taper_curve(..., return_edges=True)` returns.

    With `n_across=1` this is the outline; with `n_across=6` it is the spoke's mesh
    block.  One function, so the meshed part and the exported part cannot drift apart.

    `normals=` overrides the finite-difference normals (see module docstring).
    """
    s = arc_fractions(curve, xp=xp, at="nodes")
    thick = thickness_at_arc_length(s, t0, t1, t2, t3, xp=xp)
    nrm = offset_normals(curve, xp=xp) if normals is None else normals
    eta = xp.linspace(-1.0, 1.0, n_across + 1)
    # [N,1,2] + [1,K,1] * [N,1,1] * [N,1,2]
    return (curve[:, None, :]
            + eta[None, :, None] * (thick[:, None, None] / 2.0) * nrm[:, None, :])


def outline_edges(curve, t0, t1, t2, t3, xp=np, normals=None):
    """(top, bot) flank point arrays, each [N,2] running hub->rim.

    The exact contract `wheel_fea.thicken_3taper_curve(..., return_edges=True)` and
    `wheel_step_export.spoke_edges_global` (wheel_step_export.py:153) depend on.
    """
    band = offset_band(curve, t0, t1, t2, t3, n_across=1, xp=xp, normals=normals)
    return band[:, 1, :], band[:, 0, :]


def outline_polygon(curve, t0, t1, t2, t3, xp=np, normals=None):
    """Closed polygon [top; bot reversed], for filling in a plot."""
    top, bot = outline_edges(curve, t0, t1, t2, t3, xp=xp, normals=normals)
    return xp.concatenate([top, bot[::-1]])


# ---------------------------------------------------------------------------
# PLACEMENT
# ---------------------------------------------------------------------------

def rotate(points, angle_rad, xp=np):
    c, s = xp.cos(angle_rad), xp.sin(angle_rad)
    R = xp.stack([xp.stack([c, -s]), xp.stack([s, c])])
    return points @ R.T


def place_sector(points, hub_radius, angle_deg=0.0, xp=np):
    """Local spoke frame -> global wheel frame: shift x by the hub radius, then rotate
    into the k-th sector."""
    shifted = points + xp.asarray([hub_radius, 0.0])
    if angle_deg == 0.0:
        return shifted
    return rotate(shifted, xp.asarray(angle_deg) * math.pi / 180.0, xp=xp)


# ---------------------------------------------------------------------------
# MESH-VALIDITY PRECURSOR
# ---------------------------------------------------------------------------

# Barrier threshold for `self_intersection_margin`, in mm.
#
# Zero is the mathematical fold point, but element quality is already degrading well
# before it.  Measured over 2001 feasible genomes (study_mesh_quality.py): requiring
# margin > 0 leaves 6 designs with min scaled Jacobian as low as 0.04 — badly distorted
# but NOT inverted, so nothing would flag them.  Sweeping the threshold:
#
#     margin >   missed bad meshes   false alarms   designs kept
#       0.00                     6             87           1187
#       0.05                     2            116           1154
#       0.10                     0            140           1128
#       0.20                     0            196           1072
#
# 0.1 mm is the smallest threshold with zero misses, and it still retains 95% of the
# designs that a zero threshold would.  Use it as the barrier's target, not 0.
# Largest angle, in degrees FROM THE RING TANGENT, at which a spoke may meet its hub or
# rim circle.  Lives here rather than in `wheel_wheel` because both the optimizer (as a
# barrier) and the mesher (as a validity guard) need it, and `wheel_fea` cannot import
# `wheel_wheel` without a cycle.  See `wheel_wheel.arrival_angles` for the measurement:
# the sense runs the OPPOSITE way from intuition, and the boundary is sharp at 70.6.
MAX_ARRIVAL_DEG = 65.0

MIN_FOLD_MARGIN_MM = 0.1


def self_intersection_margin(curve, ctrl_pts, t0, t1, t2, t3, num_points, xp=np):
    """min_s ( |1/kappa(s)| - t(s)/2 ), the clearance before the offset band folds.

    Negative means the outward offset has passed the centre of curvature and the flank
    has turned inside out — an inverted element in the mesh, and a self-intersecting
    wire in the STEP.  Returned as a smooth signed scalar so it can be used directly as
    a barrier term rather than discovered as a crash.

    This is the explicit form of what the `smoothness` loss term was implicitly
    protecting (wheel_fea.py:596-598).
    """
    kappa = bezier_curvature(ctrl_pts, num_points, xp=xp)
    radius = 1.0 / (xp.abs(kappa) + 1e-30)
    s = arc_fractions(curve, xp=xp, at="nodes")
    half_t = thickness_at_arc_length(s, t0, t1, t2, t3, xp=xp) / 2.0
    return xp.min(radius - half_t)
