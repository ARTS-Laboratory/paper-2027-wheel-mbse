"""
=============================================================================
  COMPLIANT WHEEL — FULL 360 DEGREE STRUCTURED MESH
=============================================================================
    mesh = build_wheel(genes, "coarse")
    mesh.coords     # [n_nodes, 2]
    mesh.conn       # [n_elem, 9]    numpy constant
    mesh.node_sets  # 'hub_tie', 'rim_outer', ...

    mesh_coords(genes, mesh)          # the same coordinates, differentiable (M7)

Seven blocks per 30 degree sector, twelve sectors, one global node numbering with
single ownership on every shared edge.  `wheel_mesh.py` owns the spoke block; this
module owns the four ring/junction block types and the assembly.

WHY THE TOPOLOGY IS CLEAN, WHICH IS NOT OBVIOUS
-----------------------------------------------
The spoke is a spiral that sweeps 45 degrees of angle in a 30 degree sector, so the
first guess is that adjacent spokes must overlap and no partition exists.  Measured,
three facts save it:

  1. RADIUS IS STRICTLY MONOTONE along the centerline (12.700 -> 48.900) while theta
     rises to 45.16 and returns to 0.  So each spoke is a single-valued curve
     theta(r), and twelve rotated copies of a single-valued theta(r) can never
     intersect.  Minimum clearance between adjacent thickened spokes, outside the hub
     circle: 0.891 mm.
  2. The WELD FOOTPRINT on each ring circle is smaller than the sector: 15.85 degrees
     at the hub and 13.18 at the rim, of 30 available.  So the twelve junctions tile
     each ring with a genuine gap between them.
  3. The centerline endpoints are LOCKED at (0,0) and (span,0) in the local frame,
     i.e. exactly on r = 12.700 and r = 48.900 in the global frame.  Since the end
     cross-section is symmetric about the centerline, it crosses its ring circle
     exactly at its own midpoint.  That corner of the junction block is therefore
     available in closed form with no root-find at all.

Fact 1 is what makes a structured mesh possible; facts 2 and 3 are what make the
junction blocks well-shaped instead of slivers.

WHAT IS AND IS NOT MODELLED
---------------------------
The material region is `hub_disk | rim_band | 12 spoke bands clipped to the annulus`.
Two deliberate differences from the shipped STEP, both measured rather than assumed:

  FILLETS ARE NOT MODELLED, and they do not matter for area.  The unfilleted planar
  profile is 2521.4384 mm2 and the filleted solid's cross-section is 2521.73 — the
  twelve fillets are worth 0.29 mm2 in total, 0.01%.  They still matter for STRESS,
  which is why `wheel_fea.stress_concentration_kt` is retained as a post-multiplier
  here rather than deleted (the plan deletes it only once a meshed fillet exists).

  `wheel_step_export._embed` IS NOT REPRODUCED, and it does matter: it adds
  4.27 mm2 per spoke inside the annulus, so this mesh models 2.06% less material than
  the shipped part (2469.84 vs 2521.44), all of it at the junctions where it acts as a
  gusset.  That is a real modelling difference and it is deliberate.  `_embed` picks
  its direction and length by an argmax search over 21 blends x 20001 lengths
  (`wheel_step_export.py:185`) because it must avoid producing a self-intersecting
  CAD spline.  Reproducing that would put a discontinuous, non-differentiable search
  in the middle of the gradient path — exactly the "gene with no finite-difference
  plateau" failure the plan gates on at M7.  A smooth alternative does not exist
  either: the bottom flank's backward tangent MISSES the hub circle entirely (its
  closest approach exceeds 12.7), which is the same fact
  `wheel_step_export._embed`'s comment records from the other side.

  So the junction is cut at the ring circle instead, which is smooth, exact, and
  differentiable.  The 2.06% is reported by `study_wheel_mesh.py` and its stiffness
  consequence is an M4 sensitivity run, not a guess made here.

SEAMS, AND THE FAILURE MODE THEY HIDE
-------------------------------------
A seam mismatch produces a mesh that plots correctly, has positive Jacobian
everywhere, and quietly models a wheel with twelve cracks in it.  Nothing about the
solve complains.  So shared edges are resolved by SINGLE OWNERSHIP: every block writes
its own coordinates, a union-find merges the node ids declared equal, and the lowest
global id in each class becomes the owner.  The non-owner's coordinates are then
discarded — and `check_seams` reports the largest distance between what an owner and a
non-owner INDEPENDENTLY computed for the same node, which must be at machine
precision.  That number is the whole safety net for this module.
=============================================================================
"""

import numpy as np

import wheel_geometry as _geom_kernel
import wheel_mesh as _mesh

# Ring radii.  These are the exporter's, and they are the contract: `HUB_RADIUS_MM`
# and `RIM_RADIUS_MM` come from `wheel_fea`, `RIM_OUTER_RADIUS_MM` is the one
# user-decided solid parameter (`wheel_step_export.py:74`) that the M4 rim study will
# sweep.
from wheel_fea import HUB_RADIUS_MM, RIM_RADIUS_MM, HUB_RIM_SPAN_MM, NUMBER_OF_SPOKES  # noqa: F401

RIM_OUTER_RADIUS_MM = 50.0
SECTOR_DEG = 360.0 / NUMBER_OF_SPOKES


def rim_inner_radius(span_mm=HUB_RIM_SPAN_MM, hub_radius=HUB_RADIUS_MM):
    """Where the spokes merge into the rim, DERIVED from the genome's frame.

    `wheel_fea.HUB_RIM_SPAN_MM` is defined as `RIM_RADIUS_MM - HUB_RADIUS_MM`, and the
    genome's centerline runs from (0,0) to (span, 0) — so the merge radius is not an
    independent constant, it is whatever the span says it is.  Reading it back this way
    rather than importing `RIM_RADIUS_MM` is what lets the rim band be thickened INWARD
    (holding the Ø100 outer diameter) by changing one number, with the mesh, the spoke
    length and the exporter all following automatically.  Importing the constant instead
    would silently mesh a rim in the wrong place the moment the span changed.
    """
    return hub_radius + span_mm

# Depth of the meshed hub collar.  Inside this radius the hub is treated as a rigid
# body, which is the assumption the beam model already makes (`wheel_fea.py:134`); the
# meshed annulus is what preserves root compliance and root fillet stress.  A full disk
# cannot be one structured quad block — a polar grid degenerates at r=0 — and the
# butterfly/O-grid that would fix that buys nothing here.
COLLAR_DEPTH_MM = 5.0


class WheelConfig:
    """Element counts for the full wheel.

    Three of these are not free, because they sit on a shared seam:

      * the junction's transverse direction must have the spoke's thickness node count
        (they share the spoke's end cross-section), so it is `n_thick` and not a
        parameter at all;
      * `n_weld` is simultaneously the junction's along-the-arc element count AND the
        collar/rim-band weld block's angular element count;
      * the weld and free blocks of a ring share a radial edge, so they share `n_r`.

    Sizing rule, from M3: the SPAN element size must resolve the ~t boundary layers at
    the two ends, not the part — see `study_beam_agreement.py`'s MESH_H_OVER_T.  For a
    2 mm wall on a 69 mm arc, h = t/4 means n_span ~ 138.  `n_thick` is nearly
    irrelevant by comparison (converged at 4).
    """

    __slots__ = ("name", "order", "n_curve", "n_span", "n_thick", "n_weld",
                 "n_collar_r", "n_collar_free", "n_rim_r", "n_rim_free")

    def __init__(self, name, n_span, n_thick, n_weld, n_collar_r, n_collar_free,
                 n_rim_r, n_rim_free, order=2, n_curve=2400):
        for k, v in (("n_span", n_span), ("n_thick", n_thick), ("n_weld", n_weld),
                     ("n_collar_r", n_collar_r), ("n_collar_free", n_collar_free),
                     ("n_rim_r", n_rim_r), ("n_rim_free", n_rim_free)):
            if v < 1:
                raise ValueError(f"{k} must be >= 1, got {v}")
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")
        self.name = name
        self.order = order
        self.n_curve = n_curve
        self.n_span, self.n_thick, self.n_weld = n_span, n_thick, n_weld
        self.n_collar_r, self.n_collar_free = n_collar_r, n_collar_free
        self.n_rim_r, self.n_rim_free = n_rim_r, n_rim_free

    def nn(self, n_elem):
        """Node count along a direction with `n_elem` elements."""
        return self.order * n_elem + 1

    @property
    def n_elements(self):
        per_sector = (self.n_span * self.n_thick                      # spoke
                      + 2 * self.n_weld * self.n_thick                # 2 junctions
                      + self.n_collar_r * (self.n_weld + self.n_collar_free)
                      + self.n_rim_r * (self.n_weld + self.n_rim_free))
        return NUMBER_OF_SPOKES * per_sector

    def __repr__(self):
        return (f"WheelConfig({self.name!r}, spoke {self.n_span}x{self.n_thick}, "
                f"weld {self.n_weld}, {self.n_elements} elem)")


CONFIGS = {
    # The rim band gets n_r >= 3 through its 1.1 mm from `coarse` up: it is the
    # component the whole M4 gate is about, and 2 elements cannot represent bending.
    "smoke":  WheelConfig("smoke",   16, 2,  4, 2,  4, 2,  4, n_curve=600),
    "coarse": WheelConfig("coarse",  48, 4, 10, 3, 10, 3, 10, n_curve=1200),
    "medium": WheelConfig("medium",  96, 6, 16, 4, 16, 4, 16, n_curve=2400),
    "fine":   WheelConfig("fine",   192, 8, 28, 6, 28, 5, 28, n_curve=4800),
}


def get_config(cfg):
    if isinstance(cfg, WheelConfig):
        return cfg
    try:
        return CONFIGS[cfg]
    except KeyError:
        raise KeyError(f"unknown wheel config {cfg!r}; have {sorted(CONFIGS)}") from None


# ---------------------------------------------------------------------------
# GEOMETRY IN THE GLOBAL FRAME
# ---------------------------------------------------------------------------

def global_sampler(genes, cfg, span_mm=HUB_RIM_SPAN_MM, hub_radius=HUB_RADIUS_MM,
                   xp=np):
    """`wheel_mesh.band_sampler` shifted into the GLOBAL wheel frame.

    Returns `(sample, s_dense)`.  Everything in this module that touches the band goes
    through `sample`, including the spoke block itself, so shared nodes agree bitwise
    rather than to within the O(h_curve^2) difference between two constructions.
    """
    inner, s_dense, _ = _mesh.band_sampler(*[genes[i] for i in range(12)],
                                           n_curve=cfg.n_curve, span_mm=span_mm, xp=xp)
    shift = xp.stack([xp.zeros(()) + hub_radius, xp.zeros(())])

    def sample(s, eta):
        return inner(s, eta) + shift

    return sample, s_dense


N_STATION_NEWTON = 4


def ring_station(sample, s_dense, radius, eta, near_end, xp=np):
    """Arc-length fraction where the flank `eta` crosses `radius`, to machine precision.

    Two stages, and both are needed:

      1. A bracketing estimate by inverting r(s) with `xp.interp` on the dense grid.
         Valid because radius is strictly monotone along the flanks of any spoke the
         wheel can be built from (the centerline runs 12.700 -> 48.900 without turning
         back).  Accurate only to O(h_curve^2) — 3.9e-4 mm at n_curve = 600.

      2. Newton refinement against `sample` ITSELF, which is what makes |x| = radius
         hold to 1e-15 rather than to 4e-4.  That matters because this point is a corner
         shared by three blocks: the spoke's end cross-section, the junction's Coons
         patch, and the ring's exactly-circular outer boundary.  Off by 4e-4 and the
         three cannot all be right, so the assembled mesh gets a kink at every junction.

    A fixed iteration count keeps the whole thing traceable and differentiable —
    unrolled Newton's derivative converges to the implicit-function derivative, so no
    custom JVP rule is needed.  `near_end` only picks which end of the curve to search
    from, so the two crossings of a flank that leaves and re-enters cannot be confused.
    """
    n = s_dense.shape[0]
    pts = sample(s_dense, xp.zeros_like(s_dense) + eta)
    r = xp.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)

    # Bracket by finding an actual SIGN CHANGE rather than by inverting r(s) globally.
    # Inverting assumes the flank's radius is monotone, and it is not: an S-shaped
    # centerline (cy4 of the opposite sign to cy1) approaches its ring from the far side,
    # so the flank can cross the circle three times.  A global `interp` then returns a
    # point on the wrong branch and Newton converges to the wrong crossing — measured, it
    # left the Coons corner 5e-3 to 2e-2 mm out on 2 of 60 feasible genomes, which the
    # corner guard caught but could not repair.
    #
    # Which crossing is wanted: at the hub the flank starts INSIDE and the FIRST crossing
    # is where it emerges; at the rim it ends OUTSIDE and the LAST crossing is where it
    # enters the band.
    f = r - radius
    changes = ((f[:-1] * f[1:]) < 0).astype(float)
    if xp is np and not changes.any():
        raise ValueError(
            f"flank eta={eta:+.0f} never crosses r={radius:.3f} mm "
            f"(range {float(r.min()):.3f}..{float(r.max()):.3f}) — this spoke does not "
            f"reach the ring, so there is no junction to build")
    # `argmax` returns the FIRST maximum, so weighting by the index turns the same call
    # into "last crossing".  Written this way rather than with a Python `int()` so the
    # index stays traceable: JAX indexes fine with a traced integer, it just cannot be
    # converted to a concrete one.
    weight = changes if near_end == 0 else changes * xp.arange(1, n)
    idx = xp.argmax(weight)
    fa, fb = f[idx], f[idx + 1]
    sa, sb = s_dense[idx], s_dense[idx + 1]
    # Linear inverse within the bracketing segment only, which is unconditionally valid.
    s0 = sa + (sb - sa) * (-fa) / (fb - fa)

    h = 1.0 / (4.0 * n)
    s = s0
    for _ in range(N_STATION_NEWTON):
        def rad(ss):
            p = sample(xp.asarray(ss), xp.asarray(eta))
            return xp.sqrt(p[0] ** 2 + p[1] ** 2)
        f = rad(s) - radius
        dfds = (rad(s + h) - rad(s - h)) / (2.0 * h)
        s = s - f / dfds
    return xp.clip(s, 0.0, 1.0)


# Largest arrival angle (degrees FROM THE RING TANGENT) at which a junction block is
# still well shaped.  See `arrival_angles` for why this constraint exists and which way
# round it goes; `study_wheel_mesh.py` holds the threshold sweep.
#
# Measured over 200 feasible genomes, the boundary is SHARP: the worst-behaved genome
# that fails minSJ > 0.2 arrives at 70.6 degrees, and every threshold at or below 70
# gives zero misses.  65 is taken for 5.6 degrees of margin on a 200-sample estimate; it
# still keeps 82% of the design space, and the shipped genome is at 10.5 degrees.
#
#     arrival <=   60     65     70     72     80     90
#     missed        0      0      0      2      8     21
#     validity   100%   100%   100%  98.8%  95.7%  89.5%
#
# Defined in the geometry kernel so `wheel_fea` can use it as an optimizer barrier
# without importing this module (which would be a cycle).
MAX_ARRIVAL_DEG = _geom_kernel.MAX_ARRIVAL_DEG


def arrival_angles(genes, cfg, span_mm=HUB_RIM_SPAN_MM, hub_radius=HUB_RADIUS_MM,
                   xp=np):
    """Angle between the centerline and its ring's TANGENT at each end, in degrees.

    Closed form, no mesh in the loop, so it can serve directly as an optimizer barrier —
    the same role `wheel_geometry.self_intersection_margin` plays for mesh folding.
    Because the centerline endpoints are locked on the ring circles, the local radial
    direction is just the endpoint's own unit vector, and the arrival angle is
    `arcsin(|d . r_hat|)`.

    THIS IS THE CONSTRAINT THE JUNCTION BLOCKS NEED, and its sense is the opposite of
    what the geometry suggests.  A near-TANGENT arrival — which sounds like the hard case,
    and is what makes `wheel_step_export._embed` and the fillets difficult — gives
    EXCELLENT junction blocks, because the end cross-section is normal to the centerline
    and therefore nearly radial, meeting the ring arc at ~80 degrees.  A near-RADIAL
    arrival lays that cross-section nearly ALONG the arc and the corner angle collapses.
    Measured over 58 feasible genomes, correlation between arrival angle and junction
    minSJ is -0.93 at the hub and -0.91 at the rim:

        arrival      4.9    7.2   64.8   82.2  degrees
        minSJ      0.806  0.804  0.415  0.043

    Past about 80 degrees it stops being a meshing problem and becomes a design one: both
    flanks then lie OUTSIDE the ring circle and the spoke touches its ring at the single
    centerline point.  That is a hinge, not a weld — `ring_station` refuses to build it,
    which is also why `wheel_fea`'s `fixed_guided` boundary condition would be modelling
    something the part does not do.
    """
    sample, _ = global_sampler(genes, cfg, span_mm=span_mm, hub_radius=hub_radius, xp=xp)
    eps = 1e-5
    out = []
    for s_end, s_near in ((0.0, eps), (1.0, 1.0 - eps)):
        p = sample(xp.asarray(s_end), xp.asarray(0.0))
        q = sample(xp.asarray(s_near), xp.asarray(0.0))
        d = q - p
        d = d / xp.sqrt(d[0] ** 2 + d[1] ** 2)
        rhat = p / xp.sqrt(p[0] ** 2 + p[1] ** 2)
        out.append(xp.degrees(xp.arcsin(xp.clip(xp.abs(d[0] * rhat[0] + d[1] * rhat[1]),
                                                0.0, 1.0))))
    return tuple(out)


def weld_footprints_deg(genes, cfg, span_mm=HUB_RIM_SPAN_MM, hub_radius=HUB_RADIUS_MM,
                        orientation=None, xp=np):
    """`(hub, rim)` weld arc footprints in degrees, out of the `SECTOR_DEG` available.

    The weld footprint is the arc between where the straddling flank crosses the ring
    circle and where the centerline endpoint sits on it — exactly the arc the junction
    block covers, and therefore the arc over which the spoke is fused into its ring.

    It is the CAUSE of `spoke_free_arc_fraction`, which is the wheel's dominant
    stiffness variable and the one the beam model cannot see: a long weld consumes more
    of the spoke's arc length into a stiff gusset, leaving less of it free to flex.  See
    that function for the measurement and the control experiment.

    It is NOT a simple function of `arrival_angles` — the footprint also grows with the
    end thickness, and two genomes arriving at 5.5 and 5.7 degrees differ by 10 degrees
    of footprint.  Quote the footprint when explaining stiffness; the arrival angle
    governs junction mesh QUALITY, which is a different question with, confusingly, the
    opposite sense.

    Closed form apart from `ring_station`'s bracket-plus-four-Newton-steps, so this can
    serve as an optimizer term directly, the same way `self_intersection_margin` and
    `arrival_angles` already do.
    """
    cfg = get_config(cfg)
    if orientation is None:
        orientation = flank_orientation(genes, cfg, span_mm=span_mm,
                                        hub_radius=hub_radius)
    sample, s_dense = global_sampler(genes, cfg, span_mm=span_mm,
                                     hub_radius=hub_radius, xp=xp)
    rim_inner = rim_inner_radius(span_mm, hub_radius)
    stations = junction_stations(sample, s_dense, orientation, rim_inner, xp=xp)
    out = []
    for s_end, s_ring, eta in ((stations[0], 0.0, orientation[0]),
                               (stations[1], 1.0, orientation[1])):
        # P_t: the straddling flank where it crosses the ring.  P_c: the centerline
        # endpoint, which the genome LOCKS on the ring circle — the same two points
        # `sector_blocks` uses for the junction patch's arc, so this cannot drift from
        # the mesh it describes (`tests/test_wheel.py` pins the agreement).
        P_t = sample(xp.asarray(s_end), xp.asarray(eta))
        P_c = sample(xp.asarray(s_ring), xp.asarray(0.0))
        d = xp.abs(xp.arctan2(P_t[1], P_t[0]) - xp.arctan2(P_c[1], P_c[0]))
        out.append(xp.degrees(xp.minimum(d, 2.0 * np.pi - d)))
    return tuple(out)


def spoke_free_arc_fraction(genes, cfg, span_mm=HUB_RIM_SPAN_MM,
                            hub_radius=HUB_RADIUS_MM, orientation=None, xp=np):
    """`s_rim - s_hub`: the fraction of the centerline's arc length that actually flexes.

    THE WHEEL'S DOMINANT STIFFNESS VARIABLE, AND THE BEAM MODEL HAS NO TERM FOR IT.
    `generalized_spoke_mechanics` integrates the spoke over its whole hub-to-rim span,
    but the built part is fused into its rings over the two weld arcs, so only the
    middle 67-87% of it is a flexure at all; the rest is a gusset.  Measured on four GA
    winners whose losses agree to 1.2% and whose BEAM deflections agree to 0.5%:

        free arc fraction  0.667  0.755  0.781  0.865
        axle drop (mm)     0.710  1.667  1.857  2.693

    Monotone, and about the right size: a bending compliance goes as the cube of the
    free length, and (0.865/0.667)^3 = 2.2 against an observed 3.8 — the remainder
    being that the consumed arc sits at the ENDS, where the moment arm is longest.

    The control that identifies it: rigidifying the rim band (`rim_modulus_scale=1000`)
    leaves the spread almost untouched (0.211 / 0.445 / 0.497 / 0.681 mm, still 3.2x),
    so this is NOT the rim's free span between welds bending — which was the first and
    wrong explanation.  A single-spoke FEA also agrees with Castigliano to 0.8% for all
    four, so the spoke MODEL is right; what differs is how much spoke there is.

    Reproduce with `wheel_fea.py --seed N --no-export --out /tmp/x.json` for N in
    {1, 42, 2, 3}, then `study_wheel_fea.py --genome /tmp/x.json`.
    """
    cfg = get_config(cfg)
    if orientation is None:
        orientation = flank_orientation(genes, cfg, span_mm=span_mm,
                                        hub_radius=hub_radius)
    sample, s_dense = global_sampler(genes, cfg, span_mm=span_mm,
                                     hub_radius=hub_radius, xp=xp)
    s_hub, s_rim = junction_stations(sample, s_dense, orientation,
                                     rim_inner_radius(span_mm, hub_radius), xp=xp)
    return s_rim - s_hub


def flank_orientation(genes, cfg, span_mm=HUB_RIM_SPAN_MM, hub_radius=HUB_RADIUS_MM):
    """Which flank straddles each ring: `(eta_hub, eta_rim)`, each +1 or -1.

    NOT a constant, and assuming it is one is wrong for most of the design space.  The
    centerline endpoints sit exactly on the ring circles and the end cross-section is
    symmetric about the centerline, so one flank is always inside its ring and the other
    outside — but WHICH depends on the sign of the normal's radial component there, i.e.
    on which way the spoke leaves the hub.  Since `cy1..cy4` span +/-32, a spoke may
    bulge either way.

    The two ends are INDEPENDENT.  An S-shaped centerline (cy4 of the opposite sign)
    approaches the tip from below the axis, so the flank that is inside at the hub can
    also be inside at the rim.  Measured on 60 feasible Latin-hypercube genomes, only 16
    have the shipped genome's (+1, +1) combination; hardcoding it rejected 44 of them.

    Deliberately computed in numpy and treated as a STATIC choice: it is a topological
    fact about the genome, not a smooth parameter, and it changes only when a spoke
    arrives exactly radially at one of its ends.  Passing it as a static argument keeps
    the traced mesh construction free of data-dependent control flow.
    """
    sample, _ = global_sampler(np.asarray(genes, dtype=float), cfg, span_mm=span_mm,
                              hub_radius=hub_radius, xp=np)

    def radius(s, eta):
        p = sample(np.asarray(s), np.asarray(eta))
        return float(np.hypot(p[0], p[1]))

    # At the hub we follow the flank that starts INSIDE and emerges; at the rim, the one
    # that ends OUTSIDE and so must be cut where it enters the band.
    eta_hub = 1.0 if radius(0.0, 1.0) < hub_radius else -1.0
    rim_inner = rim_inner_radius(span_mm, hub_radius)
    eta_rim = 1.0 if radius(1.0, 1.0) > rim_inner else -1.0
    return eta_hub, eta_rim


def junction_stations(sample, s_dense, orientation, rim_inner=None, xp=np):
    """The two arc-length fractions that bound the SPOKE block.

    `s_hub` is where the straddling flank leaves the hub disk and `s_rim` is where the
    other one enters the rim band.  The opposite flank at each end crosses neither
    circle, which is what makes each junction a four-sided region.
    """
    eta_hub, eta_rim = orientation
    rim_inner = rim_inner_radius() if rim_inner is None else rim_inner
    return (ring_station(sample, s_dense, HUB_RADIUS_MM, eta_hub, 0, xp=xp),
            ring_station(sample, s_dense, rim_inner, eta_rim, -1, xp=xp))


# ---------------------------------------------------------------------------
# GENERIC BLOCK BUILDERS
# ---------------------------------------------------------------------------

def coons_patch(bottom, top, left, right, xp=np):
    """Bilinearly blended Coons patch, [nu, nv, 2], from its four boundary node arrays.

        bottom = X(u, 0)   top = X(u, 1)   left = X(0, v)   right = X(1, v)

    Corners must agree: bottom[0]==left[0], bottom[-1]==right[0], top[0]==left[-1],
    top[-1]==right[-1].  Checked, because a swapped or unreversed edge produces a patch
    that is folded rather than merely ugly, and the fold shows up as a negative
    Jacobian a long way downstream.

    The boundary node DISTRIBUTIONS are the caller's business: every edge that is
    shared with another block must be sampled the way that block samples it, which is
    what makes the seam exact rather than merely close.
    """
    bottom, top = xp.asarray(bottom), xp.asarray(top)
    left, right = xp.asarray(left), xp.asarray(right)
    nu, nv = bottom.shape[0], left.shape[0]
    if top.shape[0] != nu or right.shape[0] != nv:
        raise ValueError(f"opposite edges disagree: bottom {bottom.shape[0]} vs top "
                         f"{top.shape[0]}, left {left.shape[0]} vs right {right.shape[0]}")
    # The corner check needs concrete values, so it runs on the numpy path only.  That
    # is not a hole: `build_wheel` assembles in numpy, so every mesh that is actually
    # built and every genome the design-space sweep tries goes through it.  The JAX path
    # evaluates the identical expressions and exists to be differentiated, not to
    # validate.
    if xp is np:
        for a, b, what in ((bottom[0], left[0], "bottom[0]/left[0]"),
                           (bottom[-1], right[0], "bottom[-1]/right[0]"),
                           (top[0], left[-1], "top[0]/left[-1]"),
                           (top[-1], right[-1], "top[-1]/right[-1]")):
            d = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
            if d > 1e-9:
                raise ValueError(f"Coons corner mismatch {what}: {d:.3e} mm apart")

    u = xp.linspace(0.0, 1.0, nu)[:, None, None]
    v = xp.linspace(0.0, 1.0, nv)[None, :, None]
    ruled_v = (1.0 - v) * bottom[:, None, :] + v * top[:, None, :]
    ruled_u = (1.0 - u) * left[None, :, :] + u * right[None, :, :]
    bilinear = ((1.0 - u) * (1.0 - v) * bottom[0]
                + u * (1.0 - v) * bottom[-1]
                + (1.0 - u) * v * top[0]
                + u * v * top[-1])
    return ruled_v + ruled_u - bilinear


def arc_points(radius, theta0, theta1, n_nodes, xp=np):
    """`n_nodes` points uniformly in ANGLE along a circular arc, [n_nodes, 2]."""
    t = xp.linspace(theta0, theta1, n_nodes)
    return xp.stack([radius * xp.cos(t), radius * xp.sin(t)], axis=1)


def polar_block(r0, r1, theta0, theta1, n_node_th, n_node_r, xp=np):
    """Annular sector as a [n_node_th, n_node_r, 2] grid: index 0 is angle, 1 is radius.

    Uniform in both, which for an annulus is also uniform in shape — the aspect ratio
    varies across the block only as r1/r0, and for the rim band (48.9 to 50.0) that is
    1.02.
    """
    t = xp.linspace(theta0, theta1, n_node_th)[:, None]
    r = xp.linspace(r0, r1, n_node_r)[None, :]
    return xp.stack([r * xp.cos(t), r * xp.sin(t)], axis=2)


def _lerp_points(a, b, n, xp):
    """`n` points uniformly along the straight segment a -> b."""
    w = xp.linspace(0.0, 1.0, n)[:, None]
    return (1.0 - w) * xp.asarray(a)[None, :] + w * xp.asarray(b)[None, :]


# ---------------------------------------------------------------------------
# THE SEVEN BLOCKS OF ONE SECTOR
# ---------------------------------------------------------------------------
#
# Every block is an [ni, nj, 2] node grid.  The orientation of each is fixed here and
# relied on by `_seam_table` below, so it is spelled out per block rather than inferred.
#
#   spoke              [n_node_span, n_node_thick]   i: root->tip   j: bot flank->top
#   hub_junction       [n_node_weld, n_node_thick]   i: along the hub arc, P_t->P_c
#                                                    j: arc -> bottom flank
#   rim_junction       [n_node_weld, n_node_thick]   ditto at the rim
#   hub_collar_weld    [n_node_weld, n_node_collar_r]      i: theta   j: r inward->12.7
#   hub_collar_free    [n_node_collar_free, n_node_collar_r]
#   rim_band_weld      [n_node_weld, n_node_rim_r]         i: theta   j: r 48.9->50.0
#   rim_band_free      [n_node_rim_free, n_node_rim_r]

def sector_blocks(genes, cfg, xp=np, span_mm=HUB_RIM_SPAN_MM, orientation=None,
                  rim_outer=RIM_OUTER_RADIUS_MM):
    """The seven node grids of sector 0, as an ordered dict.

    Ordering matters: `build_wheel` gives ownership of a shared node to the block that
    appears FIRST, so the spoke owns its end cross-sections and the collar owns its
    inner boundary.  That is only a labelling choice — `check_seams` verifies the
    discarded coordinates agreed — but it keeps the owner predictable when debugging.
    """
    cfg = get_config(cfg)
    if orientation is None:
        orientation = flank_orientation(genes, cfg, span_mm=span_mm)
    rim_inner = rim_inner_radius(span_mm)
    sample, s_dense = global_sampler(genes, cfg, span_mm=span_mm, xp=xp)
    s_hub, s_rim = junction_stations(sample, s_dense, orientation, rim_inner, xp=xp)
    if rim_outer <= rim_inner:
        raise ValueError(
            f"rim band has non-positive thickness: outer {rim_outer:.3f} <= inner "
            f"{rim_inner:.3f} (span {span_mm:.3f}).  Thickening the band inward means "
            f"SHORTENING the span, not raising the outer radius.")

    n_th = cfg.nn(cfg.n_thick)
    n_weld = cfg.nn(cfg.n_weld)
    n_cr, n_cf = cfg.nn(cfg.n_collar_r), cfg.nn(cfg.n_collar_free)
    n_rr, n_rf = cfg.nn(cfg.n_rim_r), cfg.nn(cfg.n_rim_free)

    # --- the spoke, trimmed to the annulus -----------------------------------
    # Built from `sample` rather than from `wheel_mesh.spoke_block_coords` so that its
    # two end cross-sections are bitwise the same points the junctions use.  The layout
    # is identical (uniform arc length x uniform eta) — it is the same function.
    s_grid = xp.linspace(s_hub, s_rim, cfg.nn(cfg.n_span))
    eta_grid = xp.linspace(-1.0, 1.0, cfg.nn(cfg.n_thick))
    spoke = sample(s_grid[:, None], eta_grid[None, :])

    blocks = {"spoke": spoke}

    # --- the two junctions ---------------------------------------------------
    # Both have the same four-sided structure, which is the payoff from facts 2 and 3
    # in the module docstring:
    #
    #        P_t  ---- arc along the ring circle ---->  P_c   (= the centerline end,
    #         |                                         |      exactly on the circle)
    #   end cross-section                        half of the end cap
    #         |                                         |
    #        B*   <---- the bottom flank ----------     bot_end
    #
    # Corner angles are 79.5 / 90 / 90 / 79.5 degrees rather than the 10.5 the
    # near-tangent arrival suggests, because the end cross-section is NORMAL to the
    # centerline and therefore nearly RADIAL where it meets the ring.
    thetas = {}
    for label, radius, s_end, s_ring, spoke_row, eta in (
            ("hub_junction", HUB_RADIUS_MM, s_hub, 0.0, 0, orientation[0]),
            ("rim_junction", rim_inner, s_rim, 1.0, -1, orientation[1])):
        # The end cross-section comes straight off the spoke block — the same array, so
        # the seam is exact by construction rather than by agreement.  It must run from
        # the straddling flank (P_t, on the ring circle) to the other one (B*), and the
        # spoke block indexes eta from -1 to +1, so the direction depends on `eta`.
        cross = spoke[spoke_row][::-1] if eta > 0 else spoke[spoke_row]
        P_t = cross[0]
        # P_c is the centerline endpoint, which the genome LOCKS at (0,0) / (span,0) in
        # the local frame — i.e. exactly on r = 12.700 / 48.900 after the shift.  And
        # since the end cross-section is symmetric about the centerline, the cap crosses
        # its ring circle exactly at its own midpoint.  No root-find, no error.
        P_c = sample(xp.asarray(s_ring), xp.asarray(0.0))
        far_end = sample(xp.asarray(s_ring), xp.asarray(-eta))

        th_t = xp.arctan2(P_t[1], P_t[0])
        th_c = xp.arctan2(P_c[1], P_c[0])
        s_flank = xp.linspace(s_end, s_ring, n_weld)
        blocks[label] = coons_patch(
            bottom=arc_points(radius, th_t, th_c, n_weld, xp=xp),
            top=sample(s_flank, xp.zeros_like(s_flank) - eta),
            left=cross,
            right=_lerp_points(P_c, far_end, n_th, xp),
            xp=xp)
        thetas[label] = (th_t, th_c)
    blocks["_thetas"] = thetas

    th_hub_t, th_hub_c = thetas["hub_junction"]
    th_rim_t, th_rim_c = thetas["rim_junction"]

    # --- the two rings, each split at the weld footprint ---------------------
    # Splitting rather than grading is what makes the partial seam exact: the weld
    # arc's two ends become BLOCK CORNERS, so a contiguous run of ring nodes coincides
    # with the junction's arc nodes by construction instead of by a distribution that
    # has to be arranged to land on them.
    #
    # Both ring blocks are laid out in INCREASING theta regardless of which side the
    # weld arc falls on, so that `weld.i1 -> free.i0 -> next weld.i0` holds for every
    # genome and the sector tiling closes.  The only thing the orientation changes is
    # whether the junction's arc runs with or against the weld block, which is what
    # `_seam_table` reads.
    sector = np.radians(SECTOR_DEG)
    for ring, r0, r1, th_t, th_c, n_r, n_free in (
            ("hub_collar", HUB_RADIUS_MM - COLLAR_DEPTH_MM, HUB_RADIUS_MM,
             th_hub_t, th_hub_c, n_cr, n_cf),
            ("rim_band", rim_inner, rim_outer,
             th_rim_t, th_rim_c, n_rr, n_rf)):
        lo, hi = xp.minimum(th_t, th_c), xp.maximum(th_t, th_c)
        blocks[f"{ring}_weld"] = polar_block(r0, r1, lo, hi, n_weld, n_r, xp=xp)
        blocks[f"{ring}_free"] = polar_block(r0, r1, hi, lo + sector, n_free, n_r,
                                             xp=xp)
    return blocks


BLOCK_ORDER = ("spoke", "hub_junction", "rim_junction",
               "hub_collar_weld", "hub_collar_free",
               "rim_band_weld", "rim_band_free")

# Which blocks are which material region, for reporting and for the loss terms.
BLOCK_REGION = {"spoke": "spoke", "hub_junction": "spoke", "rim_junction": "spoke",
                "hub_collar_weld": "hub", "hub_collar_free": "hub",
                "rim_band_weld": "rim", "rim_band_free": "rim"}


# ---------------------------------------------------------------------------
# SEAM DECLARATIONS
# ---------------------------------------------------------------------------
#
# Each entry is (block_a, side_a, block_b, side_b, dk, reverse):  side `side_a` of
# `block_a` in sector k is the same set of nodes as side `side_b` of `block_b` in
# sector k + dk, walked backwards if `reverse`.
#
# Sides name a grid boundary: "i0"/"i1" hold i fixed and vary j, "j0"/"j1" the reverse.
#
# The `reverse` flags are not cosmetic.  Every one of them is the consequence of a
# specific orientation choice above — the spoke's end cross-section runs bottom-flank
# to top-flank while the junction's matching edge runs top to bottom, and the ring
# blocks sweep theta the opposite way from the junction arc.  Get one wrong and the
# seam ties node 0 of one edge to node n of the other, twisting the mesh into a shape
# that still has a positive Jacobian everywhere.  `check_seams` is what catches it.
def _seam_table(orientation, thetas):
    """The eight seams, with the four orientation-dependent `reverse` flags resolved.

    `thetas` maps each junction to `(theta_of_P_t, theta_of_P_c)`; whether the junction's
    arc runs with or against its ring block depends only on which of those is larger,
    because the ring blocks are always laid out in increasing theta.
    """
    eta_hub, eta_rim = orientation
    hub_arc_ascends = float(thetas["hub_junction"][0]) < float(thetas["hub_junction"][1])
    rim_arc_ascends = float(thetas["rim_junction"][0]) < float(thetas["rim_junction"][1])
    return (
        # The spoke's two end cross-sections.  `spoke` indexes eta from -1 to +1 while
        # the junction's matching edge starts at the straddling flank, so the direction
        # follows that flank's sign.
        ("spoke", "i0", "hub_junction", "i0", 0, eta_hub > 0),
        ("spoke", "i1", "rim_junction", "i0", 0, eta_rim > 0),
        # Each junction's arc onto its ring's weld block.
        ("hub_junction", "j0", "hub_collar_weld", "j1", 0, not hub_arc_ascends),
        ("rim_junction", "j0", "rim_band_weld", "j0", 0, not rim_arc_ascends),
        # Weld block to free block within a ring.
        ("hub_collar_weld", "i1", "hub_collar_free", "i0", 0, False),
        ("rim_band_weld", "i1", "rim_band_free", "i0", 0, False),
        # Free block to the NEXT sector's weld block — the only seams that close the 360.
        ("hub_collar_free", "i1", "hub_collar_weld", "i0", 1, False),
        ("rim_band_free", "i1", "rim_band_weld", "i0", 1, False),
    )


def _side_indices(shape, side):
    """Flat, row-major node indices along one boundary of an [ni, nj] grid."""
    ni, nj = shape[0], shape[1]
    ids = np.arange(ni * nj).reshape(ni, nj)
    return {"i0": ids[0, :], "i1": ids[-1, :],
            "j0": ids[:, 0], "j1": ids[:, -1]}[side].copy()


class _UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, i):
        p = self.parent
        root = i
        while p[root] != root:
            root = p[root]
        while p[i] != root:            # path compression
            p[i], i = root, p[i]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Lowest id wins, so ownership is deterministic and follows BLOCK_ORDER.
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            self.parent[hi] = lo


# ---------------------------------------------------------------------------
# ASSEMBLY
# ---------------------------------------------------------------------------

class WheelMesh:
    """A full-wheel mesh: coordinates, connectivity, tags, and boundary node sets."""

    __slots__ = ("coords", "conn", "cfg", "element_block", "element_region",
                 "node_sets", "edge_sets", "seam_error_mm", "n_merged", "rim_outer",
                 "genes", "span_mm", "n_spokes", "owners", "orientation", "phase_deg",
                 "_coord_fn")

    def __init__(self, coords, conn, cfg, element_block, element_region,
                 node_sets, edge_sets, seam_error_mm, n_merged,
                 rim_outer=RIM_OUTER_RADIUS_MM, genes=None,
                 span_mm=HUB_RIM_SPAN_MM, n_spokes=NUMBER_OF_SPOKES,
                 owners=None, orientation=None, phase_deg=0.0):
        self.coords = coords
        self.conn = conn
        self.cfg = cfg
        self.element_block = element_block
        self.element_region = element_region
        self.node_sets = node_sets
        # Boundary EDGE segments, [n_seg, order+1] of global node ids, ordered along the
        # boundary.  Node sets alone are not enough for a distributed load: consistent
        # nodal forces need the element edges, and on a quadratic edge the correct
        # weights are 1/6, 4/6, 1/6 rather than equal thirds.
        self.edge_sets = edge_sets
        self.seam_error_mm = seam_error_mm
        self.n_merged = n_merged
        # The one user-decided solid parameter (`wheel_step_export.py:74`), carried on
        # the mesh so a swept value cannot be lost between build and report.
        self.rim_outer = float(rim_outer)
        # The genome and frame this mesh is OF.  Carried so that `area_report` can derive
        # its own reference instead of comparing against a number someone measured once
        # under constants that have since changed.
        self.genes = None if genes is None else np.asarray(genes, dtype=float)
        self.span_mm = float(span_mm)
        self.n_spokes = int(n_spokes)
        # The three things `mesh_coords` needs to rebuild these coordinates without
        # redoing a single discrete decision: which raw node owns each merged one, which
        # way the flanks were oriented, and the phase the wheel was rolled to.  Carried
        # on the mesh rather than recomputed so the differentiable path cannot silently
        # end up describing a DIFFERENT mesh from the one that was solved.
        self.owners = None if owners is None else np.asarray(owners, dtype=np.int64)
        self.orientation = orientation
        self.phase_deg = float(phase_deg)
        self._coord_fn = None                      # lazily built by `coord_fn`

    @property
    def n_nodes(self):
        return int(self.coords.shape[0])

    @property
    def n_elements(self):
        return int(self.conn.shape[0])

    def region_mask(self, region):
        return self.element_region == region

    def __repr__(self):
        return (f"WheelMesh({self.cfg.name!r}, {self.n_nodes} nodes, "
                f"{self.n_elements} elem, seam error {self.seam_error_mm:.2e} mm)")


def _rotate(grid, angle_rad, xp):
    c, s = xp.cos(angle_rad), xp.sin(angle_rad)
    R = xp.stack([xp.stack([c, -s]), xp.stack([s, c])])
    return grid @ R.T


def _sector_coords(genes, cfg, xp, span_mm, n_spokes, orientation, rim_outer,
                   phase_deg):
    """The raw node coordinates of all twelve sectors, before the seams are merged.

    THE ENTIRE TRACED HALF OF `build_wheel`, factored out so that `mesh_coords` can run
    it again under `jax.grad` without also re-running the eager half.  Nothing here
    makes a discrete decision: the orientation is an argument, the block shapes come out
    of `sector_blocks`, and every arithmetic operation goes through `xp`.

    Returns (coords_all [n_raw, 2], shapes, offsets, thetas).
    """
    sector0 = sector_blocks(genes, cfg, xp=xp, span_mm=span_mm,
                            orientation=orientation, rim_outer=rim_outer)
    thetas = sector0.pop("_thetas")

    parts, offsets, shapes = [], {}, {}
    cursor = 0
    # `phase_deg` rolls the whole wheel under the ground.  It belongs HERE and not in
    # the load, because the ground does not move: a rolling wheel keeps its contact at
    # the bottom while the spoke pattern turns underneath it.  Putting the phase in the
    # load instead pushes the wheel sideways at an angle, which is a different load case
    # entirely — and it silently breaks the 12-fold periodicity of the axle drop, which
    # is the cheapest end-to-end check this model has.
    for k in range(n_spokes):
        angle = xp.zeros(()) + np.radians(SECTOR_DEG * k + phase_deg)
        for name in BLOCK_ORDER:
            g = _rotate(sector0[name], angle, xp) if (k or phase_deg) else sector0[name]
            shapes[(k, name)] = (int(g.shape[0]), int(g.shape[1]))
            offsets[(k, name)] = cursor
            cursor += shapes[(k, name)][0] * shapes[(k, name)][1]
            parts.append(g.reshape(-1, 2))

    return xp.concatenate(parts, axis=0), shapes, offsets, thetas


def mesh_coords(genes, mesh, xp=None):
    """`mesh`'s node coordinates as a differentiable function of the genes.

    The mesh's TOPOLOGY IS FROZEN and taken from `mesh` rather than recomputed: the
    seam-ownership table, and — the one that matters — the flank orientation.  So this
    is the derivative of a fixed mesh whose nodes move, which is the only derivative
    that means anything: `flank_orientation` is a discrete decision, and a step that
    flips it changes which block owns which node.  Such a step has no finite-difference
    plateau, and should not: it is a real discontinuity of the design space rather than
    a defect in the gradient.  `study_gradient.py` measures the plateau and would see it.

    `build_wheel` itself is deliberately NOT made traceable.  Its seam check, element
    orientation pass and validity guards all need concrete coordinates, and they are the
    reason the mesh can be trusted; wrapping them in a "skip when tracing" branch is how
    guards stop running.  Instead the traced half is shared (`_sector_coords`) and this
    function reproduces the eager result, which `study_gradient.py` gate 3 and
    `tests/test_gradient.py` both check to 1e-9 mm.

    `xp` defaults to `jax.numpy`, imported lazily.  This module is in the numpy half of
    the import graph — `wheel_geometry`, `wheel_mesh` and this file are all jax-free at
    module scope — and one convenience import at the top is exactly how that stops being
    true (`tests/test_import_hygiene.py` records what it costs).  Passing `xp=np`
    explicitly is what `tests/test_gradient.py` uses to compare the two paths.

    The default path goes through `coord_fn` and is JITTED, which is not an optimisation
    detail: see that function for the measurement.
    """
    if xp is None:
        return coord_fn(mesh)(genes)

    coords_all, _, _, _ = _sector_coords(
        genes, mesh.cfg, xp, mesh.span_mm, mesh.n_spokes, mesh.orientation,
        mesh.rim_outer, mesh.phase_deg)
    return coords_all[xp.asarray(mesh.owners)]


_COORD_FN_CACHE = {}

# SIZED BY THE PHASE LATTICE, NOT BY TASTE.  Phase is part of the key (see `coord_fn`), so
# a Stage-3 step that evaluates an 8-point phase stencil touches 8 entries, and M8's
# quantized RQMC draws that stencil from a fixed 8x8 = 64-phase lattice.  At 32 the cache
# evicted an entry it was about to need on every step — a 100% miss rate on the exact
# workload it exists for, costing the measured 0.774 s re-trace each time.  128 holds the
# whole lattice with room for a second mesh config at a checkpoint.
_COORD_FN_CACHE_MAX = 128


def coord_fn(mesh):
    """The jitted `genes -> coords` map for one mesh, traced once and reused.

    THE JIT IS LOAD-BEARING, NOT A TUNING CHOICE.  `sector_blocks` unrolls a fixed-count
    Newton refinement over `n_curve` stations for every ring crossing, so tracing it is
    expensive — and `jax.vjp` on an untraced closure re-traces on EVERY call.  Measured
    on the `smoke` mesh, that made the vjp 0.774 s against 0.05 s for the entire rest of
    the adjoint: without this, 97% of the cost of a Stage-3 gradient is re-deriving a
    jaxpr that never changes.

    THE CACHE CANNOT BE KEYED ON THE MESH OBJECT, which is the obvious thing and is
    wrong.  Every finite difference, every sweep point and every optimizer step builds a
    NEW mesh at new genes, so a per-object cache misses on literally every call it exists
    to serve while looking like it works.  The key is the STATIC RECIPE instead — element
    counts, orientation, ownership, phase — which is exactly what the traced function
    closes over and is identical across all of those calls.  `owners` is hashed by bytes;
    at 21012 nodes that is ~20 us against the 0.7 s it saves.

    A design at fixed genes is not in the key and must not be: the genes are the traced
    ARGUMENT.  Phase is, because `_sector_coords` branches on it.
    """
    import jax_config  # noqa: F401  — x64 must be set before the first trace
    import jax
    import jax.numpy as jnp

    cfg, span, n_spokes = mesh.cfg, mesh.span_mm, mesh.n_spokes
    orientation, rim_outer, phase = mesh.orientation, mesh.rim_outer, mesh.phase_deg
    owners_np = np.asarray(mesh.owners)
    key = (cfg.name, cfg.order, cfg.n_curve, cfg.n_span, cfg.n_thick, cfg.n_weld,
           cfg.n_collar_r, cfg.n_collar_free, cfg.n_rim_r, cfg.n_rim_free,
           float(span), int(n_spokes), float(rim_outer), float(phase),
           np.asarray(orientation).tobytes(), owners_np.tobytes())

    if mesh._coord_fn is not None:
        return mesh._coord_fn
    f = _COORD_FN_CACHE.get(key)
    if f is None:
        owners = jnp.asarray(owners_np)

        @jax.jit
        def f(v):                                   # noqa: F811
            coords_all, _, _, _ = _sector_coords(v, cfg, jnp, span, n_spokes,
                                                 orientation, rim_outer, phase)
            return coords_all[owners]

        if len(_COORD_FN_CACHE) >= _COORD_FN_CACHE_MAX:
            _COORD_FN_CACHE.pop(next(iter(_COORD_FN_CACHE)))
        _COORD_FN_CACHE[key] = f
    mesh._coord_fn = f
    return f


def build_wheel(genes, cfg="coarse", xp=np, span_mm=HUB_RIM_SPAN_MM,
                n_spokes=NUMBER_OF_SPOKES, orientation=None,
                rim_outer=RIM_OUTER_RADIUS_MM, phase_deg=0.0):
    """Assemble the full 360 degree mesh.

    Sector 0's seven blocks are built once and rotated, so the twelve sectors are
    exact rigid copies rather than twelve independent evaluations that could disagree
    at the seams by roundoff.

    The coordinates it returns are traced when `xp` is `jax.numpy`, but this function as
    a whole is not differentiable and is not meant to be — the seam check and
    `_orient_elements` both read concrete values.  `mesh_coords` above is the
    differentiable path.
    """
    cfg = get_config(cfg)
    if orientation is None:
        orientation = flank_orientation(genes, cfg, span_mm=span_mm)
    coords_all, shapes, offsets, thetas = _sector_coords(
        genes, cfg, xp, span_mm, n_spokes, orientation, rim_outer, phase_deg)
    seams = _seam_table(orientation, thetas)
    n_raw = coords_all.shape[0]

    # --- merge the seams -----------------------------------------------------
    uf = _UnionFind(n_raw)
    pairs = []
    for a_name, a_side, b_name, b_side, dk, reverse in seams:
        for k in range(n_spokes):
            kb = (k + dk) % n_spokes
            ia = _side_indices(shapes[(k, a_name)], a_side) + offsets[(k, a_name)]
            ib = _side_indices(shapes[(kb, b_name)], b_side) + offsets[(kb, b_name)]
            if ia.size != ib.size:
                raise ValueError(
                    f"seam {a_name}.{a_side} <-> {b_name}.{b_side}: {ia.size} vs "
                    f"{ib.size} nodes.  A WheelConfig invariant is violated — see the "
                    f"class docstring for which counts are not free.")
            if reverse:
                ib = ib[::-1]
            pairs.append((ia, ib))
            for a, b in zip(ia, ib):
                uf.union(int(a), int(b))

    # --- the seam check, BEFORE the non-owners' coordinates are discarded -----
    xy = np.asarray(coords_all)
    seam_error = 0.0
    for ia, ib in pairs:
        seam_error = max(seam_error,
                         float(np.abs(xy[ia] - xy[ib]).max()) if ia.size else 0.0)

    # --- relabel to a compact global numbering -------------------------------
    roots = np.array([uf.find(i) for i in range(n_raw)], dtype=np.int64)
    owners = np.unique(roots)
    remap = np.full(n_raw, -1, dtype=np.int64)
    remap[owners] = np.arange(owners.size)
    global_id = remap[roots]
    coords = coords_all[xp.asarray(owners)]

    # --- connectivity --------------------------------------------------------
    conn_blocks, elem_block, elem_region = [], [], []
    for k in range(n_spokes):
        for name in BLOCK_ORDER:
            ni, nj = shapes[(k, name)]
            c = global_id[_mesh.grid_connectivity(ni, nj, cfg.order)
                          + offsets[(k, name)]]
            conn_blocks.append(c)
            elem_block.extend([name] * c.shape[0])
            elem_region.extend([BLOCK_REGION[name]] * c.shape[0])
    conn = np.concatenate(conn_blocks, axis=0).astype(np.int32)
    conn = _orient_elements(np.asarray(coords), conn, np.asarray(elem_block))

    node_sets = _node_sets(shapes, offsets, global_id, n_spokes)
    edge_sets = _edge_sets(shapes, offsets, global_id, cfg, n_spokes)
    return WheelMesh(coords, conn, cfg, np.asarray(elem_block),
                     np.asarray(elem_region), node_sets, edge_sets, seam_error,
                     int(n_raw - owners.size), rim_outer=rim_outer,
                     genes=genes, span_mm=span_mm, n_spokes=n_spokes,
                     owners=owners, orientation=orientation, phase_deg=phase_deg)


def _orient_elements(xy, conn, elem_block):
    """Flip whole blocks whose (i, j) indexing is left-handed in physical space.

    A polar block indexed (theta, r) is left-handed because r_hat x theta_hat = +z, so
    its elements come out with a negative Jacobian — which in an FE assembly is not a
    cosmetic problem: it contributes NEGATIVE stiffness, and the solve happily returns
    an answer.

    The flip is applied to the element vertex ORDER, never to the node grid, so seams
    and node ids are untouched.  It is decided per block from the median signed area
    and then every element is checked, so a genuinely folded element in an otherwise
    well-oriented block still fails rather than being silently reversed.
    """
    conn = conn.copy()
    order = 4 if conn.shape[1] == 4 else 9
    perm = ([0, 3, 2, 1] if order == 4
            else [0, 3, 2, 1, 7, 6, 5, 4, 8])
    for name in np.unique(elem_block):
        m = elem_block == name
        area = _signed_area(xy, conn[m])
        if np.median(area) < 0:
            conn[m] = conn[m][:, perm]
    area = _signed_area(xy, conn)
    if area.min() <= 0:
        bad = int((area <= 0).sum())
        raise ValueError(
            f"{bad} of {area.size} elements have non-positive area after orientation "
            f"(worst {area.min():.4e} mm2) — the mesh is folded, not merely inverted; "
            f"check the fold margin and the Coons corner ordering")
    return conn


def _signed_area(xy, conn):
    """Shoelace over the 4 corner vertices of each element."""
    P = xy[conn[:, :4]]
    x, y = P[:, :, 0], P[:, :, 1]
    return 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)


def _boundary_segments(shape, side, order):
    """Element edges along one boundary of an [ni, nj] grid, [n_seg, order+1] local ids.

    Ordered along the boundary and each segment ordered along itself, so an edge tangent
    can be taken by differencing.
    """
    ids = np.arange(shape[0] * shape[1]).reshape(shape)
    line = {"i0": ids[0, :], "i1": ids[-1, :],
            "j0": ids[:, 0], "j1": ids[:, -1]}[side]
    p = order
    return np.array([line[k * p:k * p + p + 1]
                     for k in range((len(line) - 1) // p)])


def _edge_sets(shapes, offsets, global_id, cfg, n_spokes):
    """Boundary edge segments in the compacted global numbering."""
    def gather(pairs):
        out = []
        for name, side in pairs:
            for k in range(n_spokes):
                segs = _boundary_segments(shapes[(k, name)], side, cfg.order)
                out.append(global_id[segs + offsets[(k, name)]])
        return np.concatenate(out, axis=0)

    return {
        # r = RIM_OUTER: where the ground pushes.
        "rim_outer": gather([("rim_band_weld", "j1"), ("rim_band_free", "j1")]),
        # r = RIM_RADIUS between spokes: free surface, and the one whose shape tells you
        # whether the rim band is bending.
        "rim_inner_free": gather([("rim_band_free", "j0")]),
        # r = HUB_RADIUS - COLLAR_DEPTH: the rigid-hub interface.
        "hub_tie": gather([("hub_collar_weld", "j0"), ("hub_collar_free", "j0")]),
    }


def _node_sets(shapes, offsets, global_id, n_spokes):
    """Boundary node sets the FEA needs, in the compacted global numbering."""
    def gather(pairs):
        out = []
        for name, side in pairs:
            for k in range(n_spokes):
                out.append(global_id[_side_indices(shapes[(k, name)], side)
                                     + offsets[(k, name)]])
        return np.unique(np.concatenate(out))

    return {
        # r = HUB_RADIUS - COLLAR_DEPTH: tied to the rigid hub body.
        "hub_tie": gather([("hub_collar_weld", "j0"), ("hub_collar_free", "j0")]),
        # r = RIM_OUTER: the ground-contact surface.
        "rim_outer": gather([("rim_band_weld", "j1"), ("rim_band_free", "j1")]),
        # r = RIM_RADIUS between spokes: free, and the surface that tells you whether
        # the rim band is bending (M4's compliance_split).
        "rim_inner_free": gather([("rim_band_free", "j0")]),
    }


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

RIGID_CORE_AREA_MM2 = np.pi * (HUB_RADIUS_MM - COLLAR_DEPTH_MM) ** 2


def _clip_polygon_to_disk(poly, radius):
    """Area of a closed polygon intersected with the disk of `radius` about the origin.

    Exact for the case that occurs here — the spoke band crosses each ring circle twice
    — and it stays exact for any even number of crossings whose outside runs each subtend
    less than pi.  Green's theorem does the work: a chord contributes
    (x1*y2 - x2*y1)/2 and a circular arc contributes R^2*dtheta/2, so the clipped area is
    the shoelace of the retained points plus, for every exit/entry pair, the arc's
    contribution minus the chord's.
    """
    p = np.asarray(poly, dtype=float)
    if np.allclose(p[0], p[-1]):
        p = p[:-1]
    r = np.hypot(p[:, 0], p[:, 1])
    inside = r <= radius
    if inside.all():
        return float(_polygon_area(p))
    if not inside.any():
        # No vertex inside means no crossing either (a crossing needs one of each), so
        # the intersection is all-or-nothing: the whole disk when the polygon encloses
        # the origin, otherwise empty.  Does not arise for a spoke band — it straddles
        # both ring circles — but returning 0 unconditionally is wrong in a way that only
        # shows up on the reference value, which is exactly where it must not.
        if abs(_winding_number(p)) < 0.5:
            return 0.0
        return float(np.sign(_polygon_area(p)) * np.pi * radius ** 2)

    # Between an exit crossing and the next entry crossing every vertex is outside and
    # therefore dropped, so the two are ADJACENT in `kept` and the pairing needs no
    # search: the arc that replaces each excursion runs from `exit` to `exit + 1`.
    kept, exits = [], []
    n = len(p)
    for i in range(n):
        j = (i + 1) % n
        if inside[i]:
            kept.append(p[i])
        if inside[i] != inside[j]:
            kept.append(_circle_crossing(p[i], p[j], radius))
            if inside[i]:
                exits.append(len(kept) - 1)
    kept = np.asarray(kept)
    area = _polygon_area(kept)
    for a in exits:
        pa, pb = kept[a], kept[(a + 1) % len(kept)]
        dth = np.arctan2(pb[1], pb[0]) - np.arctan2(pa[1], pa[0])
        dth = (dth + np.pi) % (2.0 * np.pi) - np.pi
        area += 0.5 * radius ** 2 * dth - 0.5 * (pa[0] * pb[1] - pb[0] * pa[1])
    return float(area)


def _polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _winding_number(p):
    """Turns the polygon makes about the origin.  Requires no vertex at the origin."""
    th = np.unwrap(np.arctan2(p[:, 1], p[:, 0]))
    close = np.arctan2(p[0, 1], p[0, 0]) - th[-1]
    close = (close + np.pi) % (2.0 * np.pi) - np.pi
    return float((th[-1] - th[0] + close) / (2.0 * np.pi))


def _circle_crossing(a, b, radius):
    """The point on segment a->b at radius `radius`, by bisection.

    Bisection rather than the quadratic root because it needs no branch selection and the
    segments here are ~0.06 mm long, so 60 halvings put it far below every tolerance in
    this file.
    """
    lo, hi = 0.0, 1.0
    inside_lo = np.hypot(*a) <= radius
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        q = a + mid * (b - a)
        if (np.hypot(*q) <= radius) == inside_lo:
            lo = mid
        else:
            hi = mid
    return a + 0.5 * (lo + hi) * (b - a)


def modelled_area_reference(genes, rim_outer=RIM_OUTER_RADIUS_MM,
                            span_mm=HUB_RIM_SPAN_MM, hub_radius=HUB_RADIUS_MM,
                            n_spokes=NUMBER_OF_SPOKES, n_curve=20001):
    """The area of the region this module models, DERIVED rather than transcribed.

        hub disk + rim band + n_spokes * (spoke band clipped to the annulus)

    Everything follows from the frame constants and the genome, so changing
    `RIM_RADIUS_MM` or re-optimising the genome moves the reference with them instead of
    silently invalidating a number someone measured once.  That mattered: the previous
    hardcoded 2469.836 / 2521.438 were measured at RIM_RADIUS_MM = 48.9 and became wrong
    the moment the band was thickened.

    It is a genuine cross-check and not the same computation twice.  This path uses
    `wheel_fea.thicken_3taper_curve` — the EXPORTER's geometry, with finite-difference
    offset normals — and integrates by exact shoelace plus exact circular sectors.  The
    mesh path uses analytic-hodograph normals, Coons junction patches, and Q9 Gauss
    quadrature.  Agreement is therefore evidence, and the residual is the real
    difference between the two constructions rather than roundoff.

    The SHIPPED STEP is a different region again: `wheel_step_export._embed` adds
    material at each junction that this deliberately does not model, quantified in the
    module docstring.  Use `shipped=True` to fold that measured allowance in.
    """
    import wheel_fea as _fea

    g = np.asarray(genes, dtype=float)
    curve, _ = _fea.generate_bezier_centerline(*g[:8], span_mm=span_mm,
                                               num_points=n_curve)
    poly = np.asarray(_fea.thicken_3taper_curve(curve, *g[8:12]), dtype=float)
    poly = poly + np.array([hub_radius, 0.0])
    # `thicken_3taper_curve` returns [top; reversed bottom], whose winding depends on
    # which way the offset normals point — so it is clockwise for some genomes and
    # counter-clockwise for others.  Normalise here rather than taking abs() at the end:
    # the arc corrections inside the clipper are signed too, and only a consistently
    # oriented loop makes them cancel correctly.
    if _polygon_area(poly) < 0.0:
        poly = poly[::-1]
    rim_inner = rim_inner_radius(span_mm, hub_radius)
    spoke = (_clip_polygon_to_disk(poly, rim_inner)
             - _clip_polygon_to_disk(poly, hub_radius))
    return {
        "hub_disk_mm2": float(np.pi * hub_radius ** 2),
        "rim_band_mm2": float(np.pi * (rim_outer ** 2 - rim_inner ** 2)),
        "spoke_each_mm2": float(spoke),
        "spokes_mm2": float(n_spokes * spoke),
        "total_mm2": float(np.pi * hub_radius ** 2
                           + np.pi * (rim_outer ** 2 - rim_inner ** 2)
                           + n_spokes * spoke),
    }


# `wheel_step_export._embed` adds straight tangent segments that push each spoke further
# into its rings.  Measured at 4.27 mm2 per spoke against this region's definition — a
# deliberate modelling difference, not an error in either kernel, and the reason a
# mesh-vs-STEP comparison lands ~2% low.
EMBED_ALLOWANCE_PER_SPOKE_MM2 = 4.27


def area_report(mesh):
    """Meshed area by region, plus the totals the cross-checks compare against.

    `total_modelled_mm2` adds back the rigid hub core, which is real material that
    carries real mass and is deliberately not meshed (see COLLAR_DEPTH_MM).  Leaving it
    out is how a mesh-vs-CAD area check ends up 7% low and gets "explained" by
    discretization.

    Two reference numbers, and they are not the same thing:

      `reference_modelled_mm2` is the area of the region this module MODELS, computed
      by `modelled_area_reference` from the frame constants and the genome down an
      independent geometric path (the exporter's finite-difference offset normals,
      integrated by exact shoelace plus exact circular sectors).  `total_modelled_mm2`
      should converge to it, and that is the real cross-check.

      `reference_shipped_step_mm2` adds `wheel_step_export._embed`'s measured allowance
      per spoke, which this mesh deliberately does not model.  A mesh-vs-STEP comparison
      lands ~2% low for that reason and it is a modelling decision, not an error in
      either kernel.

    Both were hardcoded constants until the rim band was thickened, at which point they
    silently became references to a wheel that no longer exists.  Deriving them is the
    fix; `mesh.genes` is carried for exactly this.
    """
    xy = np.asarray(mesh.coords)
    area = _signed_area(xy, mesh.conn)
    by_region = {r: float(area[mesh.region_mask(r)].sum())
                 for r in ("spoke", "hub", "rim")}
    meshed = float(area.sum())
    total = meshed + RIGID_CORE_AREA_MM2
    out = {
        "by_region_mm2": by_region,
        "meshed_mm2": meshed,
        "rigid_core_mm2": float(RIGID_CORE_AREA_MM2),
        "total_modelled_mm2": total,
        "rim_outer_mm": mesh.rim_outer,
    }
    if mesh.genes is None:
        return out
    ref = modelled_area_reference(mesh.genes, rim_outer=mesh.rim_outer,
                                  span_mm=mesh.span_mm, n_spokes=mesh.n_spokes)
    ref_shipped = ref["total_mm2"] + mesh.n_spokes * EMBED_ALLOWANCE_PER_SPOKE_MM2
    out.update({
        "reference_breakdown_mm2": ref,
        "reference_modelled_mm2": ref["total_mm2"],
        "reference_shipped_step_mm2": ref_shipped,
        "error_vs_modelled": total / ref["total_mm2"] - 1.0,
        "error_vs_shipped_step": total / ref_shipped - 1.0,
    })
    return out


def quality_report(mesh):
    """Mesh validity, overall and per block type.

    Reported per block because the aggregate hides which construction is the weak one —
    and the answer is informative: the hub junction is the worst block at minSJ 0.88,
    not because it is a bad patch but because it is the one bounded by a circular arc on
    one side and a straight cross-section on the other.
    """
    xy = np.asarray(mesh.coords)
    sj = _mesh.scaled_jacobian(xy, mesh.conn)
    ar = _mesh.aspect_ratio(xy, mesh.conn)
    area = _signed_area(xy, mesh.conn)
    per_block = {}
    for name in BLOCK_ORDER:
        m = mesh.element_block == name
        per_block[name] = {"min_scaled_jacobian": float(sj[m].min()),
                           "max_aspect_ratio": float(ar[m].max()),
                           "n_elements": int(m.sum())}
    return {
        "min_scaled_jacobian": float(sj.min()),
        "max_aspect_ratio": float(ar.max()),
        "n_inverted": int((area <= 0).sum()),
        "min_element_area_mm2": float(np.abs(area).min()),
        "n_nodes": mesh.n_nodes,
        "n_elements": mesh.n_elements,
        "n_merged_nodes": mesh.n_merged,
        "seam_error_mm": mesh.seam_error_mm,
        "per_block": per_block,
    }
