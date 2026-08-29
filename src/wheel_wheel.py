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

  FILLETS ARE NOT MODELLED BY THE DEFAULT MESH, and they are a FIRST-ORDER term, not a
  rounding.  Read from `export/wheel_step_manifest.json`, which is the shipped genome's
  own OCC export (`09e8188`, 2026-08-14) and the only number here that is not transcribed:

        solid 39224.5 mm3   nofillet 36145.8   fillets 3078.77   =  7.85% of the solid

  At `SPOKE_WIDTH_MM = 22.4` that is an unfilleted profile of 1613.6518 mm2 and
  **137.4451 mm2 of fillet, 8.52% of it**.  THIS PARAGRAPH USED TO SAY 24.28 mm2, 0.92%,
  AND THAT HAS BEEN KNOWN WRONG SINCE §14 — which corrected it in
  `test_total_mass_matches_the_step_manifest_within_the_embed_difference` and in PLAN.md
  and never here, so the docstring went on claiming a fillet two orders too small for
  three arcs.  The 0.92% belongs to the era when only twelve of the forty-eight corners
  were built; all forty-eight are now, and the manifest publishes the subtraction OCC
  does exactly.  A PERCENTAGE HERE MOVES WITH THE GENOME — 6.18% at §14's, 8.77% at
  §24's, 7.85% at this one — so read the manifest rather than this sentence, and
  `test_the_fillet_reference_agrees_with_the_STEP_MANIFEST` is what stops it re-staling.

  They still matter for STRESS, which is why `wheel_fea.stress_concentration_kt` is
  retained as a post-multiplier here rather than deleted (the plan deletes it only once a
  meshed fillet exists).

  `fillet=` DOES MODEL THEM, AND SINCE PLAN.md §86 THE REFERENCE FOLLOWS IT.
  `modelled_area_reference(..., fillet=(R_hub, R_rim))` adds the sector blocking's two
  wedges per spoke — 139.1602 mm2 at the shipped genome, 8.64% of the region — so
  `error_vs_modelled` is a discretisation residual on a filleted mesh the same way it is
  on this one.

  AND THAT IS 1.25% FROM THE EXPORTER'S 137.4451, WHICH IS THE CROSS-CHECK RATHER THAN A
  COINCIDENCE (§87).  The two are genuinely different constructions — OCC's edge fillet on
  the EMBEDDED solid against a tangent arc on the un-embedded band, one kernel each — and
  they are not expected to agree exactly: `_embed` moves the corner the exporter rounds.
  Agreement to 1.7 mm2 on 137 says the mesh's fillet is the PART's fillet and not a
  construction of its own.

  `reference_shipped_step_mm2` below is the UNFILLETED cross-section — `_embed`'s
  allowance and nothing else — so an AREA comparison and a MASS comparison against the
  shipped solid are different questions and land in different places.  It is also why
  `area_report` WITHHOLDS that half for a filleted mesh: the anchor and the allowance
  were both measured against the unfilleted profile, so there is nothing like-for-like
  to report.  RE-MEASURED
  2026-08-18 on the shipped genome at `medium`, because the numbers this paragraph used
  to quote ("~1.4% low" and "~2.3% low") predate both the hub fillet milestone and the
  uncap default and were stale by more than the effects they described:

        area vs `reference_shipped_step_mm2`  -2.2205%  capped   ->  -2.0490%  default
        mass vs the FULL solid                -8.2241%  capped   ->  -8.0632%  default
        mass vs the NOFILLET solid            -0.4039%  capped   ->  -0.2292%  default

  ROW 1's LABEL USED TO READ "area vs unfilleted cross-section" AND THAT NAMED THE WRONG
  ANCHOR (§87).  All six values still reproduce exactly; what row 1 is measured against is
  the DERIVED anchor `reference_capped_mm2 + 12 x EMBED_ALLOWANCE_PER_SPOKE_MM2`, not the
  STEP's own unfilleted cross-section.  Against THAT the answer is row 3 — identical,
  because mass and area are the same ratio for a uniform extrusion — so the mesh is
  **-0.2292% from the shipped solid's unfilleted profile, not -2.05%**.

  THE ~1.8% BETWEEN THE TWO ROWS IS `EMBED_ALLOWANCE_PER_SPOKE_MM2` BEING STALE, which is
  §14's OPEN ITEM 6 and not a new finding: the same computation that produced 3.03 gives
  **0.5317 mm2 per spoke** on the shipped genome (1613.6518 - 1607.2718 = 6.3800, / 12).
  It is deliberately NOT replaced — §14: *"Do not guess a new number. Replacing 3.03 with
  0.98 would only re-stale it on the next genome; what is needed is the scaling law."*

  The full-solid column carries the fillet material, which is 7.85% of the solid (see
  above; this line used to say 6.18%, which was §14's genome) and which the default mesh
  does not model at all; the nofillet column does not, which is why it is the small one.
  Neither is a discrepancy in either kernel.  A percentage here moves with every genome —
  see `test_the_embed_difference_from_the_shipped_step_is_the_known_amount` on why the
  pinned invariant is an absolute mm2 and not one of these fractions.

  `wheel_step_export._embed` IS PARTIALLY REPRODUCED SINCE 2026-08-18, and the part that
  is not still matters — but it is SMALL, and this paragraph used to say otherwise.
  `EMBED_ALLOWANCE_PER_SPOKE_MM2` says `_embed` adds 3.03 mm2 per spoke inside the
  annulus; the shipped genome's STEP says 0.5317 (above), of which `UNCAP_DEFAULT` models
  0.2342 by continuing each junction's far flank to its ring circle instead of closing it
  with a half end cap.  `area_report` reports the modelled share as
  `gusset_modelled_per_spoke_mm2`.  So the gap against the DERIVED anchor narrows
  -2.2205% -> -2.0490%, and the gap against the SHIPPED SOLID narrows -0.4039% ->
  **-0.2292%** — 0.2975 mm2 per spoke left over.  That remainder is the real modelling
  difference and it is deliberate; the ~2.05% is the stale constant and is not.

  WHAT STILL IS NOT REPRODUCED IS `_embed`'s ARGMAX, and that was always the part with
  the M7 problem.  It picks its length by a search over 20001 candidates, and at the RIM
  over 21 blend directions as well, because it must avoid producing a self-intersecting
  CAD spline; putting that in the gradient path would be exactly the "gene with no
  finite-difference plateau" failure the plan gates on.  `_uncap_corner` needs neither:
  the ring crossing is CLOSED-FORM, and `_embed`'s DIRECTION at a fixed blend is smooth.

  THIS PARAGRAPH USED TO END "a smooth alternative does not exist either: the bottom
  flank's backward tangent MISSES the hub circle entirely (its closest approach exceeds
  12.7)".  THAT WAS MEASURED AND IT IS FALSE — the closest approach is 12.0771 and the
  tangent reaches.  The conclusion it was defending survived anyway, but not for the
  stated reason and not in the stated form: see PLAN.md 35.  A justification can be sound
  in its conclusion and wrong in its reason, and the wrong reason was hiding this arc.

  The allowance USED TO BE 4.27 and the gap ~2%.  `_embed`'s inward step took the least
  rotation from the junction tangent, which ran 4.516 mm mostly sideways; it now plunges
  radially, 1.788 mm, because the sideways run buried the hub circle under the
  neighbouring spokes and left no junction to fillet.  See HUB_PLAN.md.

  So the junction is cut at the ring circle instead, which is smooth, exact, and
  differentiable.  The gap is reported by `study_wheel_mesh.py` and its stiffness
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

import functools
import math

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


def hub_void_deg(genes, cfg, span_mm=HUB_RIM_SPAN_MM, hub_radius=HUB_RADIUS_MM,
                 crossing_etas=(1.0,), xp=np):
    """The empty arc between ADJACENT spoke roots on the hub circle, in degrees.

    `SECTOR_DEG` minus the arc one spoke's material occupies where it meets the hub, so
    it is simultaneously the slot a fillet has to grow into and — when it goes NEGATIVE —
    the statement that adjacent spokes overlap.  Same arithmetic as `weld_footprints_deg`
    above (two points, an `arctan2` difference, the wrap), on a different pair of points.

    WHY THIS EXISTS.  The hub fillet milestone measured this void on the built solid, by
    classifying material on sample rings, and found 9.907 deg = 2.196 mm of arc — which
    caps ANY hub fillet near half of that while `R_hub`'s box bound is 4.0 mm.  A number
    measured once on one genome cannot constrain an optimizer that moves the genome, and
    the void is a function of the arrival angle and `t0`, not a constant: across the 16
    Stage-2 elites it ranges 0.99 to 1.53 mm of half-arc.  So it is computed here instead,
    from the same `sample` closure the mesh uses.  This function is the geometry; the
    fillet MODEL built on it (how much of the slot one fillet may claim) is
    `wheel_objective.hub_fillet_cap_mm`, deliberately not here — this module states no
    opinion it cannot measure.

    THE NON-CROSSING FLANK, and why the fallback is `s = 0` rather than an error.  On a
    shallow spiral arrival one flank never crosses the hub circle at all (see
    `wheel_objective.fillet_flanks` — at the shipped genome the `eta=-1` flank does not,
    and that is the common case, not the exotic one).  The exporter continues such a flank
    end RADIALLY INWARD, unconditionally at the hub (`wheel_step_export._embed`, blend
    pinned to 1.0), and a radial plunge does not travel in theta — so the angle at which
    that flank enters the hub disk is the angle of its own `s = 0` endpoint.  Measured
    against the built solid: -0.3883 deg here against -0.4358 on the part, i.e. 0.0386 deg
    = 0.0086 mm of arc, 0.4% of the void and 25x below one step of the exporter's radius
    ladder.  The whole void reproduces to 0.070 deg (9.977 here, 9.907 measured).

    `crossing_etas` IS REQUIRED AND HAS NO `None`-DERIVES-IT PATH, unlike
    `weld_footprints_deg`'s `orientation`.  `ring_station`'s no-crossing guard is
    `if xp is np and not changes.any()`, so under tracing a flank that does not cross
    gives a degenerate bracket, a division by zero and a SILENT NaN in twelve of fourteen
    gradient components rather than an error.  The decision of which flanks cross is
    discrete, so it is frozen eagerly by the caller and handed over — exactly the
    discipline `fillet_flanks` exists to enforce, and it must not be optional here.
    """
    cfg = get_config(cfg)
    sample, s_dense = global_sampler(genes, cfg, span_mm=span_mm,
                                     hub_radius=hub_radius, xp=xp)
    theta = []
    for eta in (-1.0, 1.0):
        if eta in crossing_etas:
            s = ring_station(sample, s_dense, hub_radius, eta, 0, xp=xp)
        else:
            s = xp.asarray(0.0)
        P = sample(s, xp.asarray(eta))
        theta.append(xp.arctan2(P[1], P[0]))
    d = xp.abs(theta[0] - theta[1])
    occupied = xp.degrees(xp.minimum(d, 2.0 * np.pi - d))
    return SECTOR_DEG - occupied


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
# JUNCTION FILLETS (FILLET_PLAN.md, opt-in)
# ---------------------------------------------------------------------------
#
# WHICH CORNER THIS FILLETS, AND WHY ONLY ONE OF THE TWO.  Each junction has two
# re-entrant corners in this mesh, `P_t` and `P_c`.  Only `P_t` is a corner the SHIPPED
# PART has: reconstructing `wheel_step_export._embed` in numpy and walking the outline
# puts the part's flank/ring crossing at +6.117480 deg against the mesh's P_t at
# +6.117810 — 0.073 um of arc apart — while the part's SECOND crossing is at -1.526730,
# nowhere near P_c's 0.0, and 25.8 deg away in wedge angle.  `P_c` existed because the
# mesh USED TO close the spoke with a half END CAP at the ring circle, where the exporter
# drives both flanks straight through the ring (`HUB_EMBED_RADIUS_MM`) and so has no cap
# and no P_c.  SINCE 2026-08-18 `UNCAP_DEFAULT` removes the cap here too, and the second
# corner is the far flank's own ring crossing -- exact at the hub (0.01 deg of wedge),
# still 50 deg out at the rim, where fidelity and mesh validity are provably disjoint
# (UNCAP_PLAN Step 2).  The paragraph above is why THIS module fillets only `P_t`.
#
# THE REASON FOR THAT CHANGED WITH THE UNCAP FLIP AND HALF OF IT NO LONGER HOLDS
# (re-priced 2026-08-22, `make junction`, FILLET_PLAN.md PART 8).  PART 2 ruled the
# second corner out on a spoke-side leg of t/2 -- 0.737 mm at the hub and 0.716 at the
# rim, against tangent lengths of 1.090 and 6.069.  Those are the CAPPED numbers.
# Uncapped, the leg is the far flank's extension (0.660 / 0.566, SHORTER) but the corner
# opens from a 63/53 deg void to 92/89, and a wider void needs a shorter tangent:
#
#   corner                 void    leg     T = R/tan(void/2)   T/leg    R_max on that leg
#   hub P_c capped        62.82   0.737         1.087          1.47        0.450
#   hub P_c as built      91.52   0.660         0.646          0.98        0.678   FITS
#   rim P_c as built      89.49   0.566         3.027          5.34        0.561
#   rim P_c uncap=True   141.16   0.911         1.058          1.16        2.585
#
# So at the HUB the shipped `R_hub` (0.6636) now fits geometrically, by 2%.  At the RIM
# -- the corner that carries the wheel's global peak -- it does not, and the shipped
# `R_rim` of 3.0 needs five times the leg that is there.  This module still fillets only
# `P_t`, now because the `P_c` fillet is worth nothing without the rim (where the peak
# is) and because 2% of margin on a 0.66 mm stub ending in a 117 deg kink is not a
# fillet anyone should mesh.  GEOMETRIC admissibility is also not meshability: the
# construction below has a usable window of 0.12-0.24 mm at `coarse` (PART 6).
#
# THE CONSTRUCTION.  A fillet at P_t adds material in the notch between the straddling
# flank and the FREE arc of the ring circle -- a region no block covers.  The two blocks
# ADJACENT to that notch are the spoke (its j1/j0 flank edge) and the ring's FREE block
# (its ring-circle edge); the junction and the weld block meet the notch at the single
# point P_t and so cannot absorb area unless the shared corner moves.
#
# The shared corner therefore moves from P_t to B, THE TANGENT POINT ON THE RING CIRCLE.
# That choice is what keeps this cheap: the ring's two polar blocks stay exact polar
# blocks (their split angle moves from theta_t to theta_B), the junction's ring edge
# stays an exact circular arc (from theta_B instead of theta_t), and every change is
# confined to the spoke.  All eight seam-table entries keep their node counts AND their
# pairings; nothing in `_seam_table` moves, and the other six blocks come out clean.
#
# READ THIS BEFORE USING IT.  THE SPOKE BLOCK FOLDS AT THE SHIPPED RADII, and that is a
# measured result rather than a bug waiting to be fixed by a better blend:
#
#   * a fillet in a ~39 deg notch has tangent length T = R/tan(void/2) = 2.847 R, so the
#     corner must travel 1.889 mm (hub) or 8.358 mm (rim) from P_t -- against end cross
#     sections of 1.474 and 1.431 mm.  T/t is 1.28 at the hub and 5.84 at the rim, and
#     stays above 0.77 everywhere in the R box.
#   * the spoke's end cross-section therefore runs from the MOVED corner to an UNMOVED
#     far-flank point and grows with T: measured 1.452 -> 2.759 mm at the hub and
#     1.417 -> 8.596 mm at the rim.
#   * the corner's interior angle collapses from ~89 deg to 3.60 (hub) / 8.52 (rim) and
#     the cross product changes sign.  `_orient_elements` catches it.
#
# Swept, the largest radius this construction survives is 0.20 mm at `coarse` and 0.10
# at `medium` -- 3 to 30x below what ships, AND TIGHTENING UNDER REFINEMENT, which is
# what says the limit belongs to the construction and not to the geometry.
#
# THE TWO FIXES PART 3 PROPOSED FOR THAT ARE BOTH DEAD, MEASURED 2026-08-22
# (`make filletblock`, FILLET_PLAN.md STEP 1 RECORD PART 9).  PART 3 wrote that a
# filleted mesh needs "a spoke block that is GENERATED (transfinite smoothing with a
# localised boundary correction) or a dedicated fillet block with its own seam entries",
# and that sentence stood here until it was measured:
#
#   * GENERATED SPOKE BLOCK.  What fails is the angle at the moved corner -- 3.601 deg
#     (hub) / 8.524 (rim) between the fillet arc and the end cross-section, 19.800 /
#     12.432 between the two boundary CURVES.  Both curves are BOUNDARY curves of this
#     block, and so are all three nodes carrying the angle.  A generated interior holds
#     the boundary by definition: 2000 Winslow sweeps leave the corner BIT-IDENTICAL.
#   * DEDICATED FILLET BLOCK on `A - P_t - B`.  A fillet is tangent to both legs, so the
#     region it adds is a CUSP SLIVER, not a curvilinear triangle: interior angles
#     0.0000 deg at `B` (exactly -- both curves are circles) and 0.42-0.60 at `A`,
#     against 38.06 / 38.89 at `P_t`, at every radius in the gene box.  No quad covers a
#     0 deg corner and no tri-block invents one, because a tri-block subdivides the
#     region's corners rather than adding to them.
#
# WHAT DOES MESH is a BOUNDARY-LAYER block whose four corners are OFF both tangent
# points: the fillet arc as its free edge, that arc offset INTO the material as its
# inner edge (full wall at `A`, a cut of depth ~t/2 at `B`), the spoke's end cross
# section at `s_A` at one end and a radial cut at `B` at the other.  Measured min scaled
# Jacobian 0.91-1.00 with zero non-positive Gauss points, at every radius from 0.05 to
# 4.00 mm, both junctions, `coarse` and `medium`, 1x/2x/4x refinement.  ITS PRICE IS THE
# CUT AT `B`: it lands INSIDE the collar (0.711 mm, 14% of the 5 mm depth) or the band
# (0.651 mm, 43% of 1.5), so the ring circle stops being the junction/collar interface
# over the fillet's footprint and the ring block has to be notched -- 7.00 deg at the
# hub, 2.94 at the rim.  That is a re-cut of the neighbours, and it is NOT built here.
#
# AND THE SPOKE WAS NEVER THE BLOCKER.  Take the arc off this block's flank edge and end
# the spoke at the tangent station `s_A` instead, and the same ruled block is clean --
# zero mixed cells, zero non-positive Gauss points -- at 0.05 to 4.00 mm at both configs,
# where the construction below has a usable window of 0.12-0.24.  PART 3's "the spoke
# block is ruled" was a statement about WHERE THE FILLET WAS PUT.
#
# THOSE TWO RADII WERE CONTESTED AND ARE NOW SETTLED IN THEIR FAVOUR (2026-08-22,
# `make fillet`, FILLET_PLAN.md STEP 1 RECORD PART 6).  A second sweep on 2026-08-21 put
# the first fold at 4.00 mm (hub) / 3.00 (rim) at `coarse` and 0.40 / 0.40 at `medium`,
# 10-20x more permissive, and the gap was filed open because neither criterion had been
# written down.  `studies/study_fillet_fold.py` reproduces BOTH rows from one sweep and
# names them: 0.20/0.10 counts MIXED-SIGN CELLS in this block, the later row asks only
# whether `build_wheel` raises.  The later row is the weaker instrument -- see
# `_orient_elements` -- and it overstates what is usable by 10-20x.  Measured against
# `det J` at the Gauss points the assembly integrates, the radii this construction
# actually meshes are 0.12-0.24 mm (hub) / 0.11-0.23 (rim) at `coarse` and 0.07-0.11 at
# `medium`: THE ROW ABOVE IS THE ONE TO QUOTE, and it is the window's upper edge to the
# grid step.
#
# THAT WINDOW HAS A LOWER EDGE TOO, which neither sweep had looked for: below it the
# `k0` clamp holds the arc on one cell when the tangent point is nearer than one station,
# and the first element's mid-side node is dragged out of the middle half of its own edge.
# So there is no usable interval `0 < R < R_max` -- an arbitrarily small fillet folds as
# surely as a large one, and both edges are node allocation rather than geometry.
#
# What this IS good for: it is the apparatus that measured all of the above, it is what
# a fillet-block implementation will be checked against, and it is inert unless asked
# for.  `fillet=None` short-circuits to the original construction and is BIT-identical;
# `fillet=(0, 0)` goes through the Coons rebuild and agrees to 2.8e-14 mm, which is an
# independent numerical check that `sample` is affine in eta and so that the unfilleted
# spoke really is the Coons patch of its own boundary curves.


def _newton_from_root(f, x0, xp):
    """One Newton step from an ALREADY-CONVERGED root: same value, the IFT derivative.

    This is the whole trick that makes the filleted geometry differentiable, and it is
    the same argument `wheel_adjoint` makes one level up — differentiate the equation,
    not the iteration that solved it.  `x0` is a root of `f(., p0)` found eagerly by
    bisection; under a trace at `p` near `p0` the step

        x* = x0 - f(x0, p) / (df/dx)(x0, p)

    has `f(x0, p0) = 0`, so its VALUE is `x0` to the bisection's own tolerance, and its
    derivative is `dx*/dp = -(df/dp)/(df/dx)` — the implicit-function-theorem answer,
    exactly.  Differentiating the bisection instead would put 200 halvings and their
    `sign` comparisons into the tape and return a zero gradient, which is the shape of
    error that "right direction, wrong length" is made of.

    On the numpy path there is nothing to refine: `x0` IS the answer and the step would
    only add rounding, so it is skipped and the eager mesh stays bit-identical.
    """
    if xp is np:
        return x0
    import jax
    r, dr = jax.value_and_grad(f)(xp.asarray(x0))
    return xp.asarray(x0) - r / dr


def _fillet_centre(sample, s, eta_s, R, xp=np):
    """The flank point at station `s` and the fillet centre `R` off it, into the void."""
    p = sample(xp.asarray(s), xp.asarray(eta_s))
    mid = sample(xp.asarray(s), xp.asarray(0.0))
    d = p - mid
    n = xp.linalg.norm(d)
    return p, p + R * (d / xp.where(n > 0.0, n, 1.0))


def _fillet_tangency(sample, s_end, s_far, eta_s, ring_r, R, void_sign, xp=np,
                     s_seed=None):
    """Solve for the fillet tangent to the ring circle and to the straddling flank.

    The fillet disk sits in the VOID, so its centre C is at |C| = ring_r + void_sign*R
    (void_sign +1 where the spoke is OUTSIDE the ring — the hub; -1 where it is inside —
    the rim) and at distance R from the flank on the void side.  One equation, one
    unknown: the flank station.

    Returns (s_A, A, B, C).  A is the tangent point on the flank, B the tangent point on
    the ring circle (B = C scaled onto the circle, exact because C is radially offset
    from it by exactly R).

    `s_seed` is a root of this same equation found on a previous EAGER pass, and passing
    one turns the bracketed search off: the scan and the 200 halvings are replaced by
    `_newton_from_root`, which is what makes this function traceable.  The refusal below
    is a property of the bracket and so cannot be raised on that path — see
    `_fillet_curves` on why freezing it is the same decision `mesh_coords` already makes
    about the flank orientation.
    """
    target = ring_r + void_sign * R

    def f(s):
        return xp.linalg.norm(_fillet_centre(sample, s, eta_s, R, xp)[1]) - target

    if s_seed is not None:
        s_A = _newton_from_root(f, s_seed, xp)
    else:
        # Scan from the ring end into the spoke for a sign change.  `f` is negative at
        # the ring end at the hub (the centre is inside radius+R) and positive at the
        # rim; a single monotone crossing is what a fillet that fits looks like.
        ss = np.linspace(float(s_end), float(s_far), 400)
        vals = np.array([float(f(s)) for s in ss])
        hit = np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) <= 0)[0]
        if hit.size == 0:
            raise ValueError(
                f"no fillet of radius {R:.4f} mm is tangent to both the ring at r="
                f"{ring_r:.4f} and the flank anywhere along the spoke: the tangency "
                f"residual stays {vals.min():+.4f}..{vals.max():+.4f} mm.  The fillet is "
                f"larger than the notch can hold.")
        lo, hi = ss[hit[0]], ss[hit[0] + 1]
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if np.sign(float(f(lo))) == np.sign(float(f(mid))):
                lo = mid
            else:
                hi = mid
        s_A = float(0.5 * (lo + hi))
    A, C = _fillet_centre(sample, s_A, eta_s, R, xp)
    B = C * (ring_r / xp.linalg.norm(C))
    return s_A, A, B, C


def _arc_between(C, P, Q, n, xp=np):
    """`n` points along the circle centred `C` through `P` and `Q`, the short way."""
    C = np.asarray(C, float)
    R = 0.5 * (np.linalg.norm(np.asarray(P, float) - C)
               + np.linalg.norm(np.asarray(Q, float) - C))
    a0 = math.atan2(P[1] - C[1], P[0] - C[0])
    a1 = math.atan2(Q[1] - C[1], Q[0] - C[0])
    d = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi      # shortest signed sweep
    t = np.linspace(a0, a0 + d, n)
    return xp.asarray(np.stack([C[0] + R * np.cos(t), C[1] + R * np.sin(t)], axis=1))


def _filleted_spoke(sample, s_hub, s_rim, orientation, rim_inner, n_sp, n_th,
                    genes, fillet, xp):
    """The spoke block with its straddling flank rounded into each ring circle.

    The block stays [n_sp, n_th] and stays a Coons patch of its four edges, which is
    NOT a change of construction: `sample` is affine in eta, so the unfilleted block is
    already exactly the Coons patch of its own boundary curves.  What changes is two
    boundary curves — the straddling flank picks up a fillet arc at each end, and each
    end cross-section now runs from the tangent point `B` instead of from `P_t`.

    The station vector is re-spread rather than re-parametrised: the nodes that used to
    lie between the ring and the tangent point move onto the fillet arc, and the rest
    are spread over what is left.  With both radii zero this is the identity, which is
    why `sector_blocks` short-circuits instead of relying on it.
    """
    if xp is not np:
        raise NotImplementedError(
            "the filleted spoke needs a concrete tangency solve, so it is numpy-only.  "
            "`mesh_coords` — the differentiable path — freezes the mesh topology and "
            "moves nodes, so it is unaffected by this and must not route through here.")
    if fillet is True:
        R_hub, R_rim = float(genes[12]), float(genes[13])
    else:
        R_hub, R_rim = (float(v) for v in fillet)

    ends = []
    for k_eta, s_end, s_far, ring_r, R in (
            (0, s_hub, s_rim, HUB_RADIUS_MM, R_hub),
            (1, s_rim, s_hub, rim_inner, R_rim)):
        if R <= 0.0:
            ends.append(None)
            continue
        eta_s = 1.0 if float(orientation[k_eta]) > 0 else -1.0
        # Which side of the ring circle the spoke is on decides where the fillet's
        # centre sits: outside at the hub, inside at the rim.  Read it off the far
        # flank rather than hard-coding it, so a re-proportioned wheel cannot flip it
        # silently.
        far_pt = np.asarray(sample(np.asarray(float(s_end)),
                                   np.asarray(-eta_s)), float)
        void_sign = 1.0 if float(np.linalg.norm(far_pt)) > ring_r else -1.0
        s_A, A, B, C = _fillet_tangency(sample, s_end, s_far, eta_s, ring_r, R,
                                        void_sign)
        ends.append({"eta_s": eta_s, "s_A": s_A, "A": A, "B": B, "C": C})

    s0, s1 = float(s_hub), float(s_rim)
    ds = (s1 - s0) / (n_sp - 1)
    cap = max(1, (n_sp - 1) // 3)          # a fillet may not eat a third of the spoke
    k0 = 0 if ends[0] is None else int(np.clip(round((ends[0]["s_A"] - s0) / ds),
                                               1, cap))
    k1 = 0 if ends[1] is None else int(np.clip(round((s1 - ends[1]["s_A"]) / ds),
                                               1, cap))
    sA0 = s0 if k0 == 0 else ends[0]["s_A"]
    sA1 = s1 if k1 == 0 else ends[1]["s_A"]
    s_new = np.concatenate([np.linspace(s0, sA0, k0 + 1)[:-1],
                            np.linspace(sA0, sA1, n_sp - k0 - k1),
                            np.linspace(sA1, s1, k1 + 1)[1:]])

    edges = {}
    for e in (-1.0, 1.0):
        pts = np.asarray(sample(np.asarray(s_new), np.full(n_sp, e)), float).copy()
        if ends[0] is not None and ends[0]["eta_s"] == e:
            pts[:k0 + 1] = _arc_between(ends[0]["C"], ends[0]["B"], ends[0]["A"],
                                        k0 + 1)
        if ends[1] is not None and ends[1]["eta_s"] == e:
            pts[n_sp - k1 - 1:] = _arc_between(ends[1]["C"], ends[1]["A"],
                                               ends[1]["B"], k1 + 1)
        edges[e] = pts

    bottom, top = edges[-1.0], edges[1.0]
    return coons_patch(bottom, top,
                       _lerp_points(bottom[0], top[0], n_th, np),
                       _lerp_points(bottom[-1], top[-1], n_th, np), xp=np)


# ---------------------------------------------------------------------------
# THE FILLETED BLOCKING  (FILLET_PLAN.md PART 10, PLAN.md §48)
# ---------------------------------------------------------------------------
#
# `_filleted_spoke` above rounds the corner INSIDE the spoke block, which is the
# construction PART 3 wrote and §47 retired: the fold does not go away, it moves into
# whichever block carries the arc, and there it is a cusp.  What replaces it is a
# re-cut of the sector -- ELEVEN blocks instead of seven -- measured in
# `studies/study_fillet_block.py` before any of it was wired here.  The old
# construction is still reachable, and has to be: `make fillet` measures the radius at
# which it folds, and that measurement is PART 6's.  See `fillet_blocking`.
#
#     <j>_fillet_a   the boundary layer along the fillet arc, ABOVE the ring circle
#     <j>_fillet_b   the wedge that crosses it, from the arc to the ring's FAR side
#     <j>_junction   unchanged in shape; its end cross-section is replaced by
#                    `fillet_a`'s inner edge, and its arc now starts at `N`
#     <j>_ring_weld  the ring's weld block, ending at `N` rather than at `P_t`
#     <j>_ring_free  the ring's free block, starting at the CUT rather than at `P_t`
#
# `N` is where the fillet block's inner edge crosses the ring circle, and the split
# there is why there are two fillet blocks per junction: above `N` that edge's partner
# is the junction block and below it the ring, and one edge with two partners is a
# PARTIAL-EDGE SEAM.  `_seam_table`'s docstring calls whole-edge single ownership "the
# whole safety net", so the block is split instead.
#
# THE CUT REACHES THE RING'S FAR BOUNDARY -- the hub bore, the rim's outer surface --
# AND THAT IS FORCED.  A cut stopping partway gives the ring free block's left edge two
# partners; splitting that block gives its own right edge two partners, so the split
# propagates round the ring; and the block it propagates into is a TRIANGLE, because the
# inner edge is a concentric offset of an arc TANGENT to the ring circle and is therefore
# tangent to every circle concentric with it (12.86 deg at the hub, 4.38 at the rim,
# measured).  Carrying the cut through splits the ring into two quads and terminates.
#
# THE RING'S RADIAL NODE COUNT BECOMES `n_thick`.  The cut carries `n_thick` nodes and is
# the free block's left edge, whose opposite edge is the next sector's weld block's --
# so `n_collar_r` and `n_rim_r` are not used by this blocking (7 -> 9 at `coarse`).
#
# AND ITS SCOPE IS NARROW, WHICH IS MEASURED RATHER THAN HOPED.  Across the shipped
# genome's radius box it is 48/48 valid and closed at `coarse` and `medium`; across
# GENOMES it is not -- 6 of 16 feasible ones refuse the fillet at their own radii and
# only 4 of the 10 that build clear `MIN_SJ_TARGET`.  This is a measurement instrument
# for one design, not a path the optimizer may take.  See PLAN §48.
#
# NARROWED TWICE SINCE, AND ONE HALF OF IT IS GONE.  The REFUSALS are closed by the
# sector-fit clamp below -- 16 of 16 in sample and 32 of 32 on a held-out draw (§74, §78)
# -- and this construction is DIFFERENTIABLE since §79, so "a path the optimizer may take"
# is no longer blocked on either the geometry refusing or on there being no derivative.
#
# AND THE OTHER HALF IS GONE TOO, ON A MEASUREMENT AND NOT ON A CONCESSION (§89).  What
# stood here read *"What still blocks it is the BARRIER half and only that: half of each
# drawn genome box sits under `MIN_SJ_TARGET` (8/16 and 16/32)."*  Both numbers are the
# SHIPPED GLOBAL PAIR's, which stopped being what `fillet=True` builds at §85.  Re-taken
# on the assembled mesh at `coarse` -- the config `wheel_objective.objective` defaults to
# -- with `wheel_mesh.scaled_jacobian`, which is the instrument the barrier itself reads:
#
#                        in sample     held out     median J    worst barrier
#     fillet=True         16 of 16     31 of 32       0.3382          18.2047
#     fillet=None         16 of 16     31 of 32       0.7447         327.1699
#
# `coarse` AND NOT EVERY CONFIG, WHICH IS A SCOPE AND NOT A HEDGE.  The held-out draw reads
# the same 31 of 32 both ways at `medium` too, but `medium` IN SAMPLE has the filleted mesh
# at 15 of 16 against 16 of 16 -- on one genome, whose part self-intersects, where BOTH
# meshes fold an element and the corner-quad metric surfaces the filleted mesh's fold while
# missing the unfilleted one's, which is deeper.  See §89's last section; it is a defect in
# the metric rather than in this construction, and §58's fold gate rejects that genome
# either way.
#
# THE CONTROL IS THE ROW THAT RETIRES THE CLAUSE.  `fillet=None` is the mesh the optimizer
# builds TODAY and it scores the same count; the one held-out genome under the barrier is
# under it either way, and it is under it HARDER unfilleted -- 0.1298 against 0.1775, and
# the barrier TERM `t2_vector` sums is 327.17 against 18.20 on 72 marginal elements
# against 12.  A criterion that does not separate the two meshes cannot be the reason to
# refuse one of them.
#
# WHAT THE FILLET DOES COST IS HEADROOM, NOT CROSSINGS, and that is the honest other half:
# min scaled Jacobian is lower on 31 of the 32 held-out genomes, median 0.338 against
# 0.745.  It spends about half the room above the barrier and crosses it no more often.
#
# SO NOTHING HERE IS A MESH-VALIDITY BLOCKER ANY MORE.  What is left between `fillet=` and
# the objective is a DECISION, and its two terms are COST and the surrogate: §88 puts the
# filleted mesh at "2-3x the cost of the unfilleted one" and `wheel_objective` prices
# `R_hub` through a `Kt` surrogate that is exactly flat over half its feasible range
# (§75).  §89 did not re-measure either -- the cost figure is §88's and carries no
# measurement behind it in this tree, which is worth knowing before it is quoted a third
# time.  Until that decision is taken nothing wires `fillet=` into the objective, with
# `tests/test_corner_singularity.py` holding that as a check.

FILLET_LAYER_ENTRY_SLOPE = -0.45
FILLET_LAYER_END_OFFSET = 1.60

# THE BARRIER HALF'S ANSWER, AND IT IS A RULE RATHER THAN A PAIR.  PLAN §82.
#
# Every candidate §60-§69 weighed was a GLOBAL `(entry, end)`, and a global pair has to be
# safe for the tightest genome in the box and pays for that at every other one.  §68
# declined the best of them (`GENOME_ROBUST_*`, -0.90 / 0.70) on exactly that: it buys 15
# of 16 by spending the shipped genome's layer-width margin down to ~0.06.  §81 then drew
# a held-out box and killed the compromise pair `(-0.70, 0.90)` outright -- 28 of 32 where
# the pair it was meant to replace clears 31.
#
# What clears 31 of 32 for FIVE TIMES that margin is a per-genome entry: each genome takes
# a fixed share of ITS OWN layer-width room, which is the same shape `SECTOR_FIT_CLAMP`
# above already works in.  `end` is held at `GENOME_ROBUST_END` so the ENTRY rule is the
# only difference between the two.
#
# 0.45 IS THE BOTTOM EDGE OF THE ADMISSIBLE BAND, BRACKETED ON BOTH SIDES BY MEASUREMENT,
# and it is not a round number anybody picked.  The admissible set is two conditions and
# neither is invented for this: clear `MIN_SJ_TARGET` on the held-out draw, and settle
# against `study_corner_singularity.SETTLING_RATIO` at the shipped genome.
#
#   factor   held-out clears   settling ratio   settles   cost vs shipped   margin left
#    0.95        29 of 32          0.800          NO          +0.865%          0.040
#    0.85        31 of 32          0.796          NO          +0.804%          0.121
#    0.75        31 of 32          0.684          yes         +0.392%          0.202
#    0.65        31 of 32          0.634          yes         +0.298%          0.282
#    0.55        31 of 32          0.591          yes         +0.247%          0.363
#    0.45        31 of 32          0.466          yes         +0.106%          0.444   <--
#    0.35        30 of 32          0.552          yes         +0.189%          0.524
#    0.25        25 of 32          0.519          yes         +0.132%          0.605
#    0.15        16 of 32          0.483          yes         +0.116%          0.685
#
# 0.95 and 0.85 fail the settling condition; 0.35 and below start losing genomes on the
# barrier.  What is left is 0.75 / 0.65 / 0.55 / 0.45, all clearing the SAME 31 of 32,
# across which both remaining axes improve monotonically as the factor falls.  Two axes
# improving across a flat third means the operating point is the band's lower EDGE and
# not a value inside it -- quoting the interior is the class of choice §81 rejected when
# it found §78 had quoted 0.75 off a sweep that stopped at 0.55 without locating an edge.
#
# THE COST IS NOT MONOTONE BELOW THE BAND (0.35 reads +0.189% against 0.45's +0.106%) and
# that is not a law being broken: those factors are already excluded on the barrier.  The
# monotonicity is a claim about the ADMISSIBLE SET, which is where it is used, and the
# ratio is a three-rung estimate that should not be read finer than that.
FILLET_LAYER_CLIFF_FACTOR = 0.45
FILLET_LAYER_CLIFF_END = 0.70

# AND SINCE §85 IT IS WHAT A FILLETED BUILD TAKES WHEN NOBODY ASKS.  `layer_profile=None`
# on a filleted mesh now means `per_genome_layer_profile`; `"shipped"` is the two constants
# above, kept reachable under a name because `study_fillet_block` re-derives against them
# and a default that cannot be asked for by name is a default nobody can measure.
FILLET_LAYER_SHIPPED = "shipped"

# The range a cliff has to land in to be believed.  Wide enough to hold every cliff in the
# held-out draw with room to spare: the three steepest sit at -2.51, -2.30 and -2.12, and
# `study_fillet_block.CLIFF_BRACKET` stops at -2.0 and reports all three as "builds across
# the whole bracket" -- a sentinel that reads as "no edge at all" and is the OPPOSITE of
# the truth.  See §82; the bracket is the whole of that defect.
#
# IT IS A VALIDITY CHECK SINCE §88 AND WAS A SEARCH BRACKET BEFORE IT.  The cliff is a
# closed form now (`_layer_cliff_from_scalars`), so nothing is bisected and
# `LAYER_CLIFF_BISECTIONS = 90` is gone with the search that used it.  The interval is kept
# because the two sentinels it separates are still separate: a cliff outside it is a
# genome this rule has never seen and must not quietly serve.
LAYER_CLIFF_BRACKET = (-8.0, 0.0)

# The width at which the layer counts as gone.  This is `_fillet_curves`' own refusal
# threshold and must stay equal to it -- the cliff is DEFINED as the entry at which that
# refusal fires, so two thresholds would be two different cliffs.
LAYER_CLIFF_ZERO = 1e-6

# And the sample the refusal is tested on.  `_fillet_curves` takes the minimum of the
# width profile over 401 points rather than solving the cubic exactly, so the cliff is
# defined against THAT minimum: solving it exactly would put the rule and the refusal it
# is measured against 3e-6 of entry apart, in the direction where the rule prescribes an
# entry the construction then refuses.  Measured, at the shipped genome: the exact
# minimum gives -1.862136 and the sampled one -1.862143.
LAYER_CLIFF_SAMPLES = 401

# THE REFUSAL ABOVE, TURNED INTO A BOUND PROJECTION.  PLAN §57 / FILLET_PLAN PART 14.
#
# All six of the refusals in the scope note were the same one -- the fillet's tangent point
# swept past the next sector's corner at that genome's own `R_hub` -- and the radius at
# which that happens is a property of the GENOME, computable before any block is built.
# Comparing each drawn radius to its own sector's limit classifies 16 of 16 with no false
# positive, and pulling each radius back inside that limit takes the drawn box from 10/16
# BUILDING to 16/16.  It is inert at the shipped genome (hub 0.6636 against a limit of
# 3.1297; the rim has no limit at all), so the arc's published numbers are unchanged.
#
# 0.95 RATHER THAN 1.00 because the limit is where the free ring block's angular span
# reaches ZERO, and a block of zero span is not a block.  The factor is insensitive across
# 0.75-0.99 -- every one of them builds all sixteen -- so the test asserts the
# INSENSITIVITY and not this value.
#
# A CLAMP IS NOT A GATE, and the two must not blur.  The gate is exact, costs nothing, and
# LOSES the genome; this keeps the genome and models a SMALLER fillet than its genes asked
# for.  That is honest only if the applied radius is what the caller is told it got, which
# is why `build_wheel` reports `fillet_radii_mm` and `fillet_clamped` and why the clamp
# applies to `fillet=True` alone -- an explicit `fillet=(R_hub, R_rim)` is a request for
# those radii and is honoured exactly, or refused as before.
SECTOR_FIT_CLAMP = 0.95

# The bracket the limit is bisected in, and the step count.  `study_fillet_block`'s own
# `sector_fit_limit` uses [0.05, 8.0] and the two must agree or §57's margins stop being
# reproducible from the mesh.  40 halvings of a 7.95 mm bracket is 7e-12 mm, which is far
# under the 1e-4 the margins are quoted to; the study's 80 was already past double
# precision on the bracket.
SECTOR_FIT_BRACKET = (0.05, 8.0)
SECTOR_FIT_BISECTIONS = 40

FILLETED_BLOCK_ORDER = (
    "spoke",
    "hub_fillet_a", "hub_fillet_b", "rim_fillet_a", "rim_fillet_b",
    "hub_junction", "rim_junction",
    "hub_ring_weld", "hub_ring_free", "rim_ring_weld", "rim_ring_free")

FILLETED_BLOCK_REGION = {
    "spoke": "spoke",
    "hub_fillet_a": "spoke", "hub_fillet_b": "spoke",
    "rim_fillet_a": "spoke", "rim_fillet_b": "spoke",
    "hub_junction": "spoke", "rim_junction": "spoke",
    "hub_ring_weld": "hub", "hub_ring_free": "hub",
    "rim_ring_weld": "rim", "rim_ring_free": "rim"}


def _hermite(y0, y1, m0, m1, u, xp=np):
    u = xp.asarray(u)
    u2, u3 = u * u, u * u * u
    return ((2.0 * u3 - 3.0 * u2 + 1.0) * y0 + (u3 - 2.0 * u2 + u) * m0
            + (-2.0 * u3 + 3.0 * u2) * y1 + (u3 - u2) * m1)


def _unwrap_to(theta, ref):
    return theta + 2.0 * math.pi * round((ref - theta) / (2.0 * math.pi))


def _with_ends(arr, first=None, last=None, xp=np):
    """`arr` with its first and/or last row overwritten, in whichever array flavour.

    Several of the filleted blocking's edges are sampled by one construction and pinned
    at their ends by another — the arc's own endpoints, the tangent point, the corner a
    neighbouring block owns — so that the Coons corner check is met to the bit.  numpy
    does that by assignment and jax cannot; this is the two-line difference, in one
    place, so the construction below reads the same on both paths.
    """
    if xp is np:
        arr = np.array(arr, dtype=float, copy=True)
        if first is not None:
            arr[0] = first
        if last is not None:
            arr[-1] = last
        return arr
    if first is not None:
        arr = arr.at[0].set(first)
    if last is not None:
        arr = arr.at[-1].set(last)
    return arr


def ring_far_radius(is_hub, rim_outer=RIM_OUTER_RADIUS_MM):
    """The far side of the ring the fillet's cut has to reach.

    The hub's collar runs inward to the bore; the rim's band runs outward to the tyre
    surface.  Named rather than inlined because three places have to agree on it: the
    cut, the ring blocks it splits, and the boundary sets that are still keyed on side.
    """
    return (HUB_RADIUS_MM - COLLAR_DEPTH_MM) if is_hub else rim_outer


def _fillet_curves(sample, s_end, s_far, eta, ring_r, r_far, R, n_th, Q,
                   entry=FILLET_LAYER_ENTRY_SLOPE, end=FILLET_LAYER_END_OFFSET,
                   xp=np, roots=None, layer_only=False):
    """Every curve the filleted blocking needs at one junction, or `None` if it refuses.

    Returns a dict; `built` says whether it is a curve set or a refusal.  Refusals are
    GEOMETRIC and are distinguished from mesh quality on purpose: no fillet of that
    radius is tangent to both legs, the width profile would reach zero, the offset never
    crosses the ring circle, or the tangent point has swept past the next sector's
    corner.  None of them is a statement about `det J`, and each carries its own reason
    because `studies/study_fillet_block.py` reports which one, how often — a third of
    feasible genomes refuse, always the same way.

    `entry` is the width profile's slope at `A`, as a multiple of `(R + wall) * sweep`,
    and it is NEGATIVE on purpose.  At zero the inner edge leaves the end cross-section
    TANGENT to the far flank -- which is the junction block's own top edge -- and three
    blocks meet at that node with 180 degrees to share between two of them.  Measured,
    the junction block's min scaled Jacobian is 0.0400 there against 0.4272 at the
    chosen slope.  `end` is the width at `B` as a multiple of the wall.  Both were
    picked by sweeping them against the worst block over the whole box; the surface is a
    ridge and `studies/study_fillet_block.py` prints it.

    FILLET_PLAN.md PART 13 re-ran that same derivation against a box of GENOMES rather
    than one genome's radius box, because a construction tuned at the shipped genome
    alone has been tuned on a quarter of the design space (PART 10 FINDING 6).  A
    genome-diverse ridge exists near `entry = -0.75, end = 0.70` and clears
    `MIN_SJ_TARGET` for nine of the ten non-pathological genomes PART 13 drew, against
    four of ten at this pair.  It is MEASURED, NOT ADOPTED: it costs the shipped genome
    the margin PART 12's deflection-convergence finding was measured against -- the
    coarse..fine spread widens from 0.14% to 0.51%, crossing back over the +-0.3% band
    that PART 12 checked itself against -- and nothing yet consumes the genome-diverse
    path to be worth that trade.  `studies/study_fillet_block.py` prints both surfaces.

    EVERY REFUSAL ABOVE IS A DECISION, AND `roots` IS WHAT FREEZES THEM.  Pass the
    `roots` record a previous EAGER call returned and this function takes no branch on a
    computed value at all: the two bracketed root-finds become `_newton_from_root` steps
    from the seeds in that record, the angle unwraps and the void side become the frozen
    integers and signs it carries, and the four refusals are not re-tested.  That is what
    makes the construction traceable under `xp = jax.numpy`, and it is the SAME decision
    `mesh_coords` already makes about the flank orientation and the seam ownership: a
    step that changes one of them is a genuine discontinuity of the design space, and the
    derivative that means anything is the one of a fixed construction whose nodes move.
    PLAN.md §79.
    """
    frozen = roots is not None
    if frozen:
        void_sign = roots["void_sign"]
        s_A, A, B, C = _fillet_tangency(sample, s_end, s_far, eta, ring_r, R, void_sign,
                                        xp=xp, s_seed=roots["s_A"])
    else:
        far_pt = np.asarray(sample(np.asarray(float(s_end)), np.asarray(-eta)), float)
        void_sign = 1.0 if float(np.linalg.norm(far_pt)) > ring_r else -1.0
        try:
            s_A, A, B, C = _fillet_tangency(sample, s_end, s_far, eta, ring_r, R,
                                            void_sign)
        except ValueError as exc:
            return {"built": False, "why": f"no fillet of R = {R:.4f} mm is tangent to "
                                           f"both legs ({str(exc).split(':')[0]})"}
    i0 = _fillet_cross_section(sample, s_A, eta, n_th, A, xp)
    far_sA = i0[-1]
    wall = xp.linalg.norm(far_sA - A)

    a0 = xp.arctan2(A[1] - C[1], A[0] - C[0])
    a1 = xp.arctan2(B[1] - C[1], B[0] - C[0])
    if frozen:
        dd = (a1 - a0) + roots["dd_turns"] * (2.0 * math.pi)
        sweep = roots["sweep_sign"] * dd
    else:
        dd = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
        sweep = abs(dd)
    R_arc = 0.5 * (xp.linalg.norm(A - C) + xp.linalg.norm(B - C))

    def arc_at(u):
        u = xp.atleast_1d(xp.asarray(u))
        t = a0 + dd * u
        return xp.stack([C[0] + R_arc * xp.cos(t), C[1] + R_arc * xp.sin(t)], axis=1)

    m0 = entry * (R_arc + wall) * sweep
    w1 = end * wall

    def offset_at(u):
        u = xp.atleast_1d(xp.asarray(u))
        w = _hermite(wall, w1, m0, 0.0, u, xp)
        p = arc_at(u)
        nrm = p - xp.asarray(C)[None, :]
        nrm = nrm / xp.linalg.norm(nrm, axis=1)[:, None]
        return p + w[:, None] * nrm

    # THE TWO SCALARS THE LAYER-WIDTH CLIFF IS A CLOSED FORM IN (PLAN §82).  `wall` and
    # `layer_k = (R_arc + wall) * sweep` come out of the TANGENCY solve and do not depend
    # on `entry` or `end` at all, so the width profile is a cubic in `u` whose only
    # `entry`-dependence is the linear term `m0 = entry * layer_k`.  That made the entry
    # at which this junction loses its layer a root-find over arithmetic instead of over
    # thirty sector builds -- and §88 found it is not a root-find at all but a closed form,
    # which is what makes the per-genome profile DIFFERENTIABLE and not merely cheap.  See
    # `_layer_cliff_from_scalars`.  These two scalars are the whole input to it, which is
    # why they are traced values on the frozen path.
    #
    # THEY RIDE ON THE REFUSAL TOO, and that is the point rather than an oversight: the
    # cliff has to be computable AT an entry that refuses, or the only way to find it is
    # to bisect the build, which is what this replaces.
    layer = {"layer_wall": float(wall) if not frozen else wall,
             "layer_k": float((R_arc + wall) * sweep) if not frozen else (R_arc + wall) * sweep}

    # AND `free_span_deg`, COMPUTED HERE RATHER THAN AFTER THE TWO REFUSALS BELOW.
    # PLAN §83.
    #
    # It is a function of `Q` -- the ring corner `uncap` chooses -- and of `B`, the
    # tangency point, and of nothing else: neither `entry` nor `end` reaches it, and the
    # arithmetic below is the same arithmetic that used to sit ninety lines further down.
    # Left there, it was unreachable at exactly the radii where the layer-width and
    # ring-crossing refusals fire, so `_sector_fit_span` saw only `built: False` and read
    # it as "this radius has no room".  At a steep entry that made the sector-fit LIMIT
    # the layer's cliff wearing another name -- measured at the shipped genome, entry
    # -1.40: `R` = 1.60 builds with 8.1465 deg of free ring, bit-identical to the shallow
    # entry's, `R` = 1.65 refuses on layer width, and the reported "limit" came out 1.6285
    # where the free span is 8.15 deg rather than the zero the limit is DEFINED as.
    #
    # `th_N` stays below because it needs `N`, which is the offset root-find and does
    # depend on the profile.  Only the half that does not is hoisted.
    th_q = xp.arctan2(Q[1], Q[0])
    th_B_raw = xp.arctan2(B[1], B[0])
    if frozen:
        th_B = th_B_raw + roots["th_B_turns"] * (2.0 * math.pi)
        dirn = roots["dirn"]
    else:
        th_q, th_B_raw = float(th_q), float(th_B_raw)
        th_B = _unwrap_to(th_B_raw, th_q)
        dirn = 1.0 if th_B > th_q else -1.0
    th_end = th_q + dirn * math.radians(SECTOR_DEG)
    free_span_deg = (th_end - th_B) * dirn * (180.0 / math.pi)
    # Every refusal from here down carries both, so a caller measuring the SECTOR FIT can
    # get its answer at a radius where the LAYER refuses, and vice versa.  The two
    # constraints have different remedies -- a smaller radius, a shallower entry -- and a
    # refusal that names only one of them cannot be told apart by the caller.
    geom = dict(layer, free_span_deg=free_span_deg)

    # AND THIS IS EVERYTHING THE LAYER-WIDTH CLIFF NEEDS, SO THE HARVEST STOPS HERE (§88).
    #
    # Not an optimisation.  Everything below depends on `entry`, and the caller that asks
    # for `layer_only` is `layer_cliff_entry` -- which is COMPUTING the entry and can only
    # have passed a placeholder.  Running on would refine `u_N` by a Newton step from a
    # seed found at a different entry, making it the root of nothing; it is discarded
    # either way, and on the traced path only dead-code elimination keeps that harmless.
    # `built` is `None` rather than `False`: no refusal was tested, which is not the same
    # as none firing, and §84 is what that distinction cost last time.
    if layer_only:
        return dict(geom, built=None, why=None)

    if not frozen and float(_hermite(wall, w1, m0, 0.0,
                                     np.linspace(0.0, 1.0, 401)).min()) <= 1e-6:
        return dict(geom, built=False,
                    why="the layer's width profile reaches zero thickness")

    def past(u):
        return (xp.linalg.norm(offset_at(u)[0]) - ring_r) * void_sign

    if frozen:
        u_N = _newton_from_root(past, roots["u_N"], xp)
    else:
        if past(0.0) < 0.0 or past(1.0) > 0.0:
            return dict(geom, built=False,
                        why="the layer's inner edge does not cross the ring circle once")
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if past(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        u_N = float(0.5 * (lo + hi))
    N = offset_at(u_N)[0]
    L = N * (r_far / xp.linalg.norm(N))

    # `th_q`, `th_B`, `dirn`, `th_end` and `free_span_deg` were hoisted above the two
    # layer-dependent refusals (PLAN §83).  This is the half that could not be: `N` is the
    # offset root-find and moves with the profile.
    th_N_raw = xp.arctan2(N[1], N[0])
    if frozen:
        th_N = th_N_raw + roots["th_N_turns"] * (2.0 * math.pi)
    else:
        th_N_raw = float(th_N_raw)
        th_N = _unwrap_to(th_N_raw, th_q)
    if not frozen and free_span_deg <= 0.0:
        return dict(geom, built=False,
                    why=f"the fillet's tangent point has passed the next sector's corner "
                        f"({free_span_deg:.3f} deg of free ring left)")
    out = {"built": True, **geom,
           "A": A, "B": B, "C": C, "N": N, "L": L, "u_N": u_N, "i0": i0,
           "far_sA": far_sA, "wall_mm": wall, "arc_at": arc_at,
           "offset_at": offset_at, "r_far": float(r_far), "s_A": s_A,
           "th_q": th_q, "th_B": th_B, "th_N": th_N, "th_end": th_end,
           "dirn": dirn,   # `free_span_deg` arrives in `geom`, with every refusal's copy
           "sweep_deg": sweep * (180.0 / math.pi)}
    if not frozen:
        # THE RECORD THAT MAKES THIS TRACEABLE, harvested where the answers are already
        # concrete: the two roots, and the four choices the branches above made on a
        # computed value -- which side of the ring circle the void is on, and the three
        # angle unwraps.  Floats, not integers, because they are consumed as ARITHMETIC
        # on the traced path and `coord_fn` passes them in as an argument rather than
        # closing over them (see there: closing over them would retrace per genome).
        out["roots"] = {
            "s_A": float(s_A), "u_N": float(u_N), "void_sign": float(void_sign),
            "dd_turns": float(round((float(dd) - float(a1 - a0)) / (2.0 * math.pi))),
            "sweep_sign": 1.0 if float(dd) >= 0.0 else -1.0,
            "th_B_turns": float(round((th_B - th_B_raw) / (2.0 * math.pi))),
            "th_N_turns": float(round((th_N - th_N_raw) / (2.0 * math.pi))),
            "dirn": float(dirn)}
    return out


def _fillet_cross_section(sample, s, eta, n_th, first, xp=np):
    """The spoke's end cross-section at `s`, straddling flank -> far flank.

    Its first node is replaced by the exact tangent point so the Coons corner check is
    met to the bit rather than to the sampler's round-off; at `s = s_A` the two agree to
    ~1e-13 mm, and the seam error `build_wheel` reports is that and nothing else.
    """
    row = xp.asarray(sample(xp.full(n_th, s), xp.linspace(eta, -eta, n_th)))
    return xp.concatenate([xp.asarray(first)[None, :], row[1:]], axis=0)


def _sector_fit_span(curves_at, R):
    """`free_span_deg` at one radius, or a negative number when the RADIUS has no room.

    NOT EVERY REFUSAL MEANS NO ROOM, and reading them all that way is the defect §82
    measured and §83 fixes.  This function's answer feeds `_sector_fit_limit`, whose
    output is the radius the clamp pulls back to -- so a refusal counted here is a
    statement that *this radius* is too big.  Of the four refusals `_fillet_curves` can
    make, only the tangency one is: no fillet of that radius is tangent to both legs.

    The layer-width and ring-crossing refusals are statements about the layer PROFILE and
    have a different remedy -- a shallower entry, not a smaller radius.  Counted here they
    made the limit collapse with the profile: measured at the shipped genome, entry -1.40,
    the reported hub "limit" was 1.6285, a radius at which 8.15 deg of free ring remains
    against the ZERO the limit is defined as.  Every one of those refusals now carries
    `free_span_deg` anyway, so the sector fit is answerable at a radius the layer refuses.

    The fourth refusal -- the tangent point past the corner -- carries a `free_span_deg`
    that is already <= 0, so it needs no special case and keeps its own sign.

    THIS DIVERGES FROM `study_fillet_block.sector_fit_limit`, which delegates its
    root-find here (PART 21) and therefore follows automatically.  At the SHIPPED profile
    the two answers are identical and §57's and §74's published margins are unchanged:
    across the full bracket no layer refusal fires at the shipped genome, or at any of the
    32 held-out ones -- 0 of 64 junction-pairs.
    """
    c = curves_at(R)
    if "free_span_deg" in c:
        return float(c["free_span_deg"])
    return -1.0


def _sector_fit_limit(curves_at, bracket=SECTOR_FIT_BRACKET,
                      steps=SECTOR_FIT_BISECTIONS):
    """The radius at which this genome's fillet reaches the next sector's corner.

    `None` means the junction has no limit inside the bracket -- the rim usually does not
    -- and the caller must treat that as "no clamp", never as zero.  Bisected rather than
    solved because `free_span_deg` runs through `_fillet_tangency`'s own root-find; it is
    monotone decreasing in `R`, which is what makes a bisection the right instrument and
    is the same assumption §57 made.
    """
    lo, hi = float(bracket[0]), float(bracket[1])
    if _sector_fit_span(curves_at, hi) > 0.0:
        return None
    if _sector_fit_span(curves_at, lo) <= 0.0:
        return lo
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if _sector_fit_span(curves_at, mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _clamp_to_sector(curves_at, R, factor):
    """`(R_used, clamped)` -- the radius pulled back inside its own sector's room.

    ONE PROBE BEFORE ANY BISECTION, because the bisection is ~40 `_fillet_curves` calls
    and the clamp is inert at almost every genome that matters.  `free_span_deg` is
    monotone in `R`, so free ring left at `R / factor` proves `factor * limit > R` and the
    clamp cannot bite -- one extra call instead of forty.  At the shipped genome that is
    the whole cost.
    """
    if factor is None:
        return float(R), False
    probe = min(float(R) / float(factor), float(SECTOR_FIT_BRACKET[1]))
    if _sector_fit_span(curves_at, probe) > 0.0:
        return float(R), False
    limit = _sector_fit_limit(curves_at)
    if limit is None or float(R) <= factor * limit:
        return float(R), False
    return float(factor * limit), True


def _layer_cliff_from_scalars(wall, layer_k, end, bracket=LAYER_CLIFF_BRACKET, xp=np):
    """The `entry` at which THIS junction's layer reaches zero width, from two scalars.

    `wall` and `layer_k` are what `_fillet_curves` harvests out of the TANGENCY solve,
    and neither depends on the layer profile -- so the width profile

        H(u) = _hermite(wall, end * wall, entry * layer_k, 0, u)
             = a(u) + entry * b(u),   a(u) = h00(u) wall + h01(u) end wall
                                      b(u) = h10(u) layer_k

    is a cubic in `u` whose only `entry`-dependence is ONE LINEAR TERM.

    THAT MAKES THE CLIFF A CLOSED FORM AND NOT A SEARCH (§88).  `b(u) = u (1 - u)^2
    layer_k > 0` strictly inside `(0, 1)`, so for each sample `H(u) <= Z` exactly when
    `entry <= (Z - a(u)) / b(u)`, and the sampled minimum is at or below `Z` exactly when
    ONE of them is:

        cliff = max_u (LAYER_CLIFF_ZERO - a(u)) / b(u)

    over the same `LAYER_CLIFF_SAMPLES` grid `_fillet_curves` takes its minimum on -- the
    two endpoints excluded because `b` vanishes there and `H` is `wall` and `end * wall`,
    both orders above `Z`, so neither can ever be the minimum this is solving for.

    UNTIL §88 THIS BISECTED, 90 halvings of `LAYER_CLIFF_BRACKET`, on the argument that
    `min_u H` is a minimum of functions linear in `entry` and therefore monotone.  That
    argument was right and the closed form is the same root exactly: measured over the
    98 junction-pairs of the shipped genome and all 48 genomes of
    `studies/study_fillet_block.json`, the two agree to **2 ulp, worst relative
    3.999e-16**.  What the closed form buys is a DERIVATIVE -- the bisection's is 90
    `sign` comparisons and a zero, which is the shape of error `_newton_from_root`'s
    docstring is about -- so the per-genome layer profile can now be differentiated
    through instead of frozen.  `xp = jax.numpy` traces it.

    NO MESH IS BUILT.  That is the whole point: `study_fillet_block.cliff_entry` USED TO
    find this number by bisecting `sector_verdict` thirty times, which is thirty filleted
    sector builds, and it agreed with this to 9.1e-10 over the held-out draw (§82) before
    §84 made it delegate here.  A per-genome layer profile is only adoptable because the
    cliff costs arithmetic.

    `None` means the cliff is outside `LAYER_CLIFF_BRACKET` -- the layer survives the
    whole bracket, which for a bracket this wide has not been observed and would be a
    genuine finding rather than a fallback to take.  The upper end is checked too and the
    bisection never checked it: a junction with no layer at ANY negative entry converged
    to the bracket's own zero and reported it as a cliff.  It fires at none of the 98
    pairs, so this is a guard and not a change of answer.  THE CHECK IS SKIPPED UNDER A
    TRACE, because a mesh being differentiated has already been built and its eager pass
    is what made that check.
    """
    u = xp.linspace(0.0, 1.0, LAYER_CLIFF_SAMPLES)[1:-1]
    a = _hermite(wall, end * wall, 0.0, 0.0, u, xp)
    b = _hermite(0.0, 0.0, layer_k, 0.0, u, xp)
    c = xp.max((LAYER_CLIFF_ZERO - a) / b)
    if xp is not np:
        return c
    c = float(c)
    return c if float(bracket[0]) <= c <= float(bracket[1]) else None


def _layer_profile(layer_profile):
    """`(entry, end)`, defaulting to the two measured constants.

    `None` and `FILLET_LAYER_SHIPPED` both return exactly `FILLET_LAYER_ENTRY_SLOPE` and
    `FILLET_LAYER_END_OFFSET` -- the pair the module shipped before §85 moved the filleted
    default onto the per-genome rule.

    Why the pass-through exists at all.  `filleted_sector` already exposed these two so
    `study_fillet_block` could re-derive them against the BLOCKING rather than assert
    them.  FILLET_PLAN PART 16 needs the other half of that derivation -- what a
    candidate pair costs the SOLVE, which is the one reason PART 13's decision still
    rests on -- and that needs a full `build_wheel`, not a sector.  Same reason, one
    level up: the study must not keep a second copy of the assembly.

    IT IS A PASS-THROUGH AND NOT THE PLACE THE DEFAULT LIVES (§85).  It has no `genes`, so
    it cannot answer the per-genome rule -- and that is deliberate rather than a gap:
    `_resolve_layer_profile` settles the profile where the genes ARE, and is now this
    function's only caller.  §88 took the cache key off it: the key reads its record raw,
    because a record holding `None` means the RULE there and this function would map it
    onto the shipped constants.
    """
    if layer_profile is None or layer_profile == FILLET_LAYER_SHIPPED:
        return FILLET_LAYER_ENTRY_SLOPE, FILLET_LAYER_END_OFFSET
    entry, end = layer_profile
    return float(entry), float(end)


def _resolve_layer_profile(layer_profile, genes, cfg, fillet, span_mm=HUB_RIM_SPAN_MM,
                           orientation=None, rim_outer=RIM_OUTER_RADIUS_MM, uncap=None,
                           clamp=SECTOR_FIT_CLAMP, fillet_blocking="sector",
                           xp=np, sector=None, roots=None):
    """The concrete `(entry, end)` this build will use, settled BEFORE anything caches it.

    RESOLVING EAGERLY WAS THE WHOLE DESIGN, and the reason was the jax cache key.  §85
    built it from `repr(_layer_profile(rec["layer_profile"]))`, and leaving `None` in the
    record to be resolved per genome further down meant **two genomes with different
    profiles sharing one key**, the second handed the first's traced geometry -- the exact
    failure the `repr(uncap)` comment beside that key warns about.  Storing the resolved
    pair made the key correct with no change to the key itself.

    §88 INVERTED THAT, AND THE KEY IS CORRECT THE OTHER WAY NOW.  What §85 could not see is
    that a resolved pair is a function of the GENOME, so the key it made correct also became
    genome-dependent -- one jaxpr per design, which is the failure `coord_fn`'s docstring
    describes for the frozen roots.  The cliff is a closed form in two tangency scalars now,
    so `xp`, `sector` and `roots` carry this whole resolution THROUGH the trace: the record
    holds `None` for a per-genome mesh again, the key reads it RAW rather than through
    `_layer_profile`, the two genomes above legitimately share one key, and the gradient
    `mesh_coords` used to REFUSE is returned.

    The eager resolution below is unchanged and is still what `build_wheel` records and
    reports.  What is still frozen is what was always frozen: the flank orientation, the
    seam ownership, the four geometric refusals and the clamp's decision -- discrete
    choices, every one.

    `None` is the per-genome rule (§82, §85) on exactly one path -- `fillet=True` with the
    eleven-block sector blocking.  `FILLET_LAYER_SHIPPED` is the pair the module shipped
    before it; a tuple is itself.  Everywhere else `None` is still the two constants:

      AN EXPLICIT `(R_hub, R_rim)` PAIR DOES NOT GET THE RULE, for the same reason
      `SECTOR_FIT_CLAMP` does not touch one: a caller passing radii is measuring THOSE
      radii, and a profile derived from the genome's own room is not what was asked for.
      `study_fillet_fold` sweeps `fillet=(R, 0.0)` down to a zero rim, where a per-genome
      cliff cannot even be computed.

      `fillet_blocking="spoke"` DOES NOT GET IT EITHER.  §47's construction has no layer
      to profile -- the arc goes onto the spoke block's own flank edge -- so resolving one
      would build an eleven-block sector nobody asked for, purely to answer a question
      that construction does not pose.

      AND THE UNFILLETED PATH NEVER REACHES A LAYER AT ALL.
    """
    if (fillet is not True or layer_profile is not None
            or fillet_blocking != "sector"):
        return _layer_profile(layer_profile)
    return per_genome_layer_profile(genes, cfg, fillet=fillet, span_mm=span_mm,
                                    orientation=orientation, rim_outer=rim_outer,
                                    uncap=uncap, clamp=clamp,
                                    xp=xp, sector=sector, roots=roots)


def _filleted_sector_blocks(sample, cfg, s_hub, s_rim, orientation, rim_inner,
                            rim_outer, genes, fillet, uncap=None,
                            entry=FILLET_LAYER_ENTRY_SLOPE,
                            end=FILLET_LAYER_END_OFFSET,
                            clamp=SECTOR_FIT_CLAMP, xp=np, roots=None,
                            layer_scalars_only=False):
    """The eleven node grids of a filleted sector 0, keyed by `FILLETED_BLOCK_ORDER`.

    Raises `ValueError` when the geometry refuses, with the reason, because the caller
    is `sector_blocks` and a refusal there is the same class of event as the rim band
    having non-positive thickness: a genome this construction cannot build.

    Carries `_applied` alongside `_thetas` and `_dirn` -- the radii actually built and
    which of them the sector-fit clamp moved, see `SECTOR_FIT_CLAMP`.  It rides in the
    dict rather than in a second return value for the same reason those two do: every
    caller already spreads this dict by `FILLETED_BLOCK_ORDER` and would have to grow a
    tuple unpack otherwise.  It is there because a clamp the caller cannot see is exactly
    the defect §57 warns about.

    `_roots` rides there too, and is the third of the same kind: the frozen record that
    lets this construction be re-run under `xp = jax.numpy` with no bracketed search and
    no branch on a computed value.  Pass it back in as `roots` and every discrete choice
    is taken from it instead of being re-made -- see `_fillet_curves` and `mesh_coords`.
    """
    # `UNCAP_DEFAULT` is defined below `sector_blocks`, so it cannot be a default
    # ARGUMENT here without a forward reference; `None` is not a legal `uncap` value and
    # is therefore a safe sentinel for "whatever the module ships".
    uncap = UNCAP_DEFAULT if uncap is None else uncap
    n_th = cfg.nn(cfg.n_thick)
    n_sp = cfg.nn(cfg.n_span)
    n_weld = cfg.nn(cfg.n_weld)
    # THE CLAMP APPLIES TO THE GENE-DERIVED BRANCH ALONE.  `fillet=True` means "this
    # genome's fillets", and a genome whose radius has no room is the case §57 measured;
    # an explicit pair is a request for those exact radii -- every caller that passes one
    # is measuring the radius itself -- so it is honoured or refused, never quietly moved.
    frozen = roots is not None
    if fillet is True:
        radii, clamp_factor = (genes[12], genes[13]), clamp
        if not frozen:
            radii = (float(radii[0]), float(radii[1]))
    else:
        radii, clamp_factor = tuple(float(v) for v in fillet), None

    applied = {"requested_mm": tuple(radii), "radii_mm": None,
               "clamped": {"hub": False, "rim": False},
               "clamp_factor": clamp_factor}
    used = {}
    curves = {}
    harvest = {}
    for junction, R, s_end, s_far, ring_r, k_eta in (
            ("hub", radii[0], s_hub, s_rim, HUB_RADIUS_MM, 0),
            ("rim", radii[1], s_rim, s_hub, rim_inner, 1)):
        if not frozen and R <= 0.0:
            raise ValueError(
                f"the filleted blocking needs a positive radius at both junctions; "
                f"{junction} got {R:.4f}.  Use `fillet=None` for the unfilleted mesh.")
        eta = 1.0 if float(orientation[k_eta]) > 0 else -1.0
        s_ring = 0.0 if junction == "hub" else 1.0
        # `Q` is the ring blocks' other corner and it follows `uncap`, exactly as it
        # does in the unfilleted `sector_blocks`: the centreline endpoint when the
        # junction is capped, where the FAR FLANK crosses when it is not.  Taking the
        # centreline endpoint unconditionally here moves every ring block by the blend
        # and moves the sector-fit limit with it -- measured, 3.130 -> 3.484 mm.
        is_hub_j = junction == "hub"
        blend = _uncap_blend(uncap, is_hub_j)
        Q = (xp.asarray(sample(xp.asarray(s_ring), xp.asarray(0.0)))
             if blend is None else
             xp.asarray(_uncap_corner(sample, s_ring, eta, ring_r, is_hub_j, blend, xp)))
        def curves_at(R_try, _a=(s_end, s_far, eta, ring_r, is_hub_j, Q, junction)):
            s_e, s_f, et, rr, hub_j, q, name = _a
            return _fillet_curves(sample, s_e, s_f, et, rr,
                                  ring_far_radius(hub_j, rim_outer), R_try, n_th, q,
                                  entry, end, xp,
                                  roots[name] if frozen else None,
                                  # ONLY ON THE FROZEN PATH.  The eager harvest is what
                                  # `layer_cliff_entry` reads its refusal REASONS from,
                                  # and a short-circuit tests none of them (§88).
                                  layer_only=layer_scalars_only and frozen)

        if frozen:
            # THE CLAMP'S DECISION IS FROZEN LIKE EVERY OTHER ONE, and the case it does
            # not cover is refused rather than approximated: a clamped radius is
            # `factor * limit(genome)` and does not follow `R` at all, so treating the
            # eager radius as if it did would return a gradient that is wrong in a way
            # nothing downstream could see.  `mesh_coords` refuses that mesh up front.
            R_used, was_clamped = R, roots[junction]["clamped"]
        else:
            R_used, was_clamped = _clamp_to_sector(curves_at, R, clamp_factor)
        applied["clamped"][junction] = was_clamped
        used[junction] = R_used
        c = curves_at(R_used)
        if layer_scalars_only:
            # The two tangency scalars the layer-width cliff is a closed form in, at the
            # radius this genome will actually be BUILT at.  Harvested here rather than
            # in a second copy of this loop because the clamp, the uncap corner and the
            # junction stations all have to agree with the build or the cliff describes
            # a mesh nobody builds -- the exact error §82 caught the study making.
            # A layer refusal still carries them, which is why no `built` check guards
            # this: the cliff has to be computable at an entry that refuses.
            #
            # `built` STAYS `None` ON THE FROZEN PATH, where `layer_only` short-circuited
            # before any refusal was tested.  `bool(None)` would say `False` -- "it
            # refused" -- of a mesh that was built, which is the two-meanings-in-one-value
            # error §84 spent a section on.
            harvest[junction] = {"wall": c.get("layer_wall"), "k": c.get("layer_k"),
                                 "R_mm": R_used, "clamped": bool(was_clamped),
                                 "built": None if c["built"] is None
                                          else bool(c["built"]),
                                 "why": c.get("why")}
            continue
        if not frozen and not c["built"]:
            raise ValueError(
                f"no filleted blocking exists at the {junction}: {c['why']}.  "
                f"See FILLET_PLAN.md PART 10.")
        if not frozen:
            harvest[junction] = dict(c.pop("roots"), clamped=bool(was_clamped))
        c.update({"eta": eta, "s_ring": s_ring, "ring_r": float(ring_r), "Q": Q,
                  "R_mm": R_used, "R_requested_mm": R})
        curves[junction] = c

    applied["radii_mm"] = tuple(used[j] for j in ("hub", "rim"))
    if layer_scalars_only:
        return dict(harvest, _applied=applied)

    lo, hi = curves["hub"]["s_A"], curves["rim"]["s_A"]
    if not frozen and not lo < hi:
        raise ValueError(
            f"the two fillets are longer than the spoke: tangent stations {lo:.4f} and "
            f"{hi:.4f} cross.")

    eta_grid = xp.linspace(-1.0, 1.0, n_th)
    s_grid = xp.linspace(lo, hi, n_sp)
    blocks = {"spoke": xp.asarray(sample(s_grid[:, None], eta_grid[None, :]))}
    thetas, dirn = {}, {}

    for junction in ("hub", "rim"):
        c = curves[junction]
        is_hub = junction == "hub"
        eta, ring_r = c["eta"], c["ring_r"]
        n_free = cfg.nn(cfg.n_collar_free if is_hub else cfg.n_rim_free)

        u_N = c["u_N"]
        P_split = c["arc_at"](u_N)[0]
        cut_N = _lerp_points(P_split, c["N"], n_th, xp)
        ua, ub = xp.linspace(0.0, u_N, n_th), xp.linspace(u_N, 1.0, n_th)
        arc_a = _with_ends(c["arc_at"](ua), c["A"], P_split, xp)
        arc_b = _with_ends(c["arc_at"](ub), P_split, c["B"], xp)
        inner_a = _with_ends(c["offset_at"](ua), c["far_sA"], c["N"], xp)
        inner_b = _lerp_points(c["N"], c["L"], n_th, xp)      # the radial dive
        cut_B = _lerp_points(c["B"], c["L"], n_th, xp)
        blocks[f"{junction}_fillet_a"] = coons_patch(
            bottom=arc_a, top=inner_a, left=c["i0"], right=cut_N, xp=xp)
        fb = coons_patch(bottom=arc_b, top=inner_b, left=cut_N, right=cut_B, xp=xp)
        blocks[f"{junction}_fillet_b"] = fb

        s_flank = xp.linspace(c["s_A"], c["s_ring"], n_weld)
        top = xp.asarray(sample(s_flank, xp.zeros(n_weld) - eta))
        far_end = xp.asarray(sample(xp.asarray(c["s_ring"]), xp.asarray(-eta)))
        Q = c["Q"]
        bottom = _with_ends(arc_points(ring_r, c["th_N"], c["th_q"], n_weld, xp),
                            c["N"], Q, xp)
        top = _with_ends(top, c["far_sA"], far_end, xp)
        blocks[f"{junction}_junction"] = coons_patch(
            bottom=bottom, top=top, left=blocks[f"{junction}_fillet_a"][:, -1, :][::-1],
            right=_lerp_points(Q, far_end, n_th, xp), xp=xp)

        # The ring keeps the SHIPPED radial order -- bore -> ring at the hub, ring ->
        # tyre surface at the rim -- so `_edge_sets` and `_node_sets` keep naming the
        # same sides.  Laying both rings out the same way round is tidier to write and
        # would move `hub_tie`, `rim_outer` and `rim_inner_free` with nothing going red.
        if is_hub:
            r_j0, r_j1 = c["r_far"], ring_r
            th_j0, th_j1 = c["th_N"], c["th_B"]
            corner_j0, corner_j1 = c["L"], c["B"]
            cut = fb[-1, :, :][::-1]                          # L -> B
        else:
            r_j0, r_j1 = ring_r, c["r_far"]
            th_j0, th_j1 = c["th_B"], c["th_N"]
            corner_j0, corner_j1 = c["B"], c["L"]
            cut = fb[-1, :, :]                                # B -> L
        blocks[f"{junction}_ring_weld"] = polar_block(
            r_j0, r_j1, c["th_q"], c["th_N"], n_weld, n_th, xp)
        e_j0 = _with_ends(arc_points(r_j0, th_j0, c["th_end"], n_free, xp), corner_j0,
                          xp=xp)
        e_j1 = _with_ends(arc_points(r_j1, th_j1, c["th_end"], n_free, xp), corner_j1,
                          xp=xp)
        blocks[f"{junction}_ring_free"] = coons_patch(
            bottom=e_j0, top=e_j1, left=cut,
            right=_lerp_points(e_j0[-1], e_j1[-1], n_th, xp), xp=xp)

        thetas[f"{junction}_junction"] = (c["th_N"], c["th_q"])
        dirn[junction] = c["dirn"]

    # Returned in `FILLETED_BLOCK_ORDER` rather than in the order the loop happens to
    # build them.  `sector_blocks`' docstring makes ownership follow the dict order, and
    # the unfilleted dict already matches `BLOCK_ORDER`; a filleted dict that did not
    # would make "first block wins" mean two different things depending on the flag.
    out = {name: blocks[name] for name in FILLETED_BLOCK_ORDER}
    out["_thetas"] = thetas
    out["_dirn"] = dirn
    out["_applied"] = applied
    out["_roots"] = roots if frozen else harvest
    return out


def filleted_sector(genes, cfg, fillet=True, span_mm=HUB_RIM_SPAN_MM,
                    orientation=None, rim_outer=RIM_OUTER_RADIUS_MM, uncap=None,
                    entry=FILLET_LAYER_ENTRY_SLOPE, end=FILLET_LAYER_END_OFFSET,
                    clamp=SECTOR_FIT_CLAMP):
    """The eleven blocks of a filleted sector 0, from the genes, with the profile exposed.

    `sector_blocks(..., fillet=)` is the path `build_wheel` takes and it holds `entry`
    and `end` at their measured values.  This is the same construction with those two
    open, and it exists so that `studies/study_fillet_block.py` can RE-DERIVE them
    against the worst block over the gene box instead of asserting them — the study
    calls this rather than keeping a second copy of the geometry, which is the failure
    mode its own docstring is about.

    For the profile's cost to the SOLVE rather than to the blocking, `sector_blocks`,
    `_sector_coords` and `build_wheel` now take `layer_profile=(entry, end)` as well —
    same two numbers, threaded to the full assembly.  Use that one when a mesh is wanted
    and this one when the blocks are.

    `clamp=None` restores the pre-§57 behaviour — the genes' own radii, refused when they
    have no room — which is what a study re-deriving a published refusal COUNT wants.
    """
    cfg = get_config(cfg)
    if orientation is None:
        orientation = flank_orientation(genes, cfg, span_mm=span_mm)
    rim_inner = rim_inner_radius(span_mm)
    sample, s_dense = global_sampler(genes, cfg, span_mm=span_mm)
    s_hub, s_rim = junction_stations(sample, s_dense, orientation, rim_inner)
    return _filleted_sector_blocks(sample, cfg, s_hub, s_rim, orientation, rim_inner,
                                   rim_outer, genes, fillet, uncap, entry, end, clamp)


def layer_cliff_entry(genes, cfg, fillet=True, end=FILLET_LAYER_CLIFF_END,
                      span_mm=HUB_RIM_SPAN_MM, orientation=None,
                      rim_outer=RIM_OUTER_RADIUS_MM, uncap=None,
                      clamp=SECTOR_FIT_CLAMP, xp=np, sector=None, roots=None):
    """The `entry` at which THIS genome loses its layer, at the radii it is built at.

    Returns `{"entry": float|None, "why": str, "per_junction": {...}}`.  The sector's
    cliff is the BINDING junction's -- the least negative of the two -- because one
    `entry` is spent on the whole sector and the first junction to lose its layer is the
    one that refuses the build.

    THE RADII ARE THE ONES THE MESH WILL USE, resolved through the same clamp, uncap
    corner and junction stations `_filleted_sector_blocks` resolves them through, because
    a cliff measured at a radius the genome will never be built at describes a mesh
    nobody builds.  They are resolved AT THE SHIPPED PROFILE, which is what
    `study_fillet_block` does and is not arbitrary: `_sector_fit_span` counts a
    layer-width refusal as "no room" (see §82), so a clamp resolved at a steep entry
    reports the layer's own cliff under the sector fit's name.  At the shipped profile no
    layer refusal fires anywhere in the radius bracket, so the limit there is the sector
    fit and nothing else -- and the operating point this feeds is shallower still.

    `None` means the layer survives `LAYER_CLIFF_BRACKET` entirely, or that a junction
    refused for a reason that is not the layer -- and the two are KEPT APART in `why`,
    because folding them together is the defect §78 corrected once and §82 found twice
    more.  A caller that treats "no cliff" as "the safest case" must check which it got.

    IT TRACES SINCE §88, and that is what makes the per-genome layer profile
    differentiable rather than frozen.  `xp = jax.numpy` with `roots` — the frozen record
    an eager build returned — takes the same two tangency scalars out of the same harvest
    and hands them to the same closed form, so the traced entry is not a re-derivation of
    the rule but the rule itself.  `sector` is `(sample, s_hub, s_rim, rim_inner)` from a
    caller that has already built them (`sector_blocks` has), so the trace pays for one
    sampler and one pair of junction stations rather than two.

    THE BINDING JUNCTION IS NOT FROZEN.  `xp.maximum` of the two cliffs carries the
    subgradient of whichever binds, so a genome sitting on the crossing gets a one-sided
    derivative rather than a discrete choice baked into the jaxpr — the opposite of the
    treatment `_fillet_curves`' refusals get, and for the opposite reason: this one is a
    kink in a continuous function, not a change of construction.
    """
    cfg = get_config(cfg)
    if orientation is None:
        orientation = flank_orientation(genes, cfg, span_mm=span_mm)
    if sector is None:
        rim_inner = rim_inner_radius(span_mm)
        sample, s_dense = global_sampler(genes, cfg, span_mm=span_mm, xp=xp)
        s_hub, s_rim = junction_stations(sample, s_dense, orientation, rim_inner, xp=xp)
    else:
        sample, s_hub, s_rim, rim_inner = sector
    sc = _filleted_sector_blocks(sample, cfg, s_hub, s_rim, orientation, rim_inner,
                                 rim_outer, genes, fillet, uncap,
                                 FILLET_LAYER_ENTRY_SLOPE, FILLET_LAYER_END_OFFSET,
                                 clamp, xp, roots, layer_scalars_only=True)
    per, cliffs = {}, []
    for junction in ("hub", "rim"):
        h = sc[junction]
        if h["wall"] is None or h["k"] is None:
            per[junction] = {"cliff": None, "why": h["why"], "R_mm": h["R_mm"]}
            continue
        c = _layer_cliff_from_scalars(h["wall"], h["k"], float(end), xp=xp)
        per[junction] = {"cliff": c, "R_mm": h["R_mm"], "clamped": h["clamped"],
                         "why": None if c is not None
                                else "the layer survives the whole bracket"}
        if c is not None:
            cliffs.append(c)
    if len(cliffs) < 2:
        bad = [j for j in ("hub", "rim") if per[j]["cliff"] is None]
        return {"entry": None, "per_junction": per,
                "why": "; ".join(f"{j}: {per[j]['why']}" for j in bad)}
    # NOT `max`, WHICH COMPARES AND WOULD NEED A CONCRETE BOOL.  Same answer eagerly, and
    # the only form that survives a trace.  The eager answer is narrowed back to a Python
    # float because it is written into `studies/*.json` and `np.float64` is not
    # serialisable.
    entry = xp.maximum(cliffs[0], cliffs[1])
    return {"entry": float(entry) if xp is np else entry,
            "per_junction": per, "why": None}


def per_genome_layer_profile(genes, cfg, fillet=True,
                             factor=FILLET_LAYER_CLIFF_FACTOR,
                             end=FILLET_LAYER_CLIFF_END, **kw):
    """`(entry, end)` for this genome: a fixed share of its OWN layer-width room.

    THE RULE §82 ADOPTED, and the shape is `SECTOR_FIT_CLAMP`'s rather than a constant's
    -- each genome takes `factor` of the room it actually has instead of every genome
    sharing one pair that has to be safe for the tightest of them.  It clears
    `MIN_SJ_TARGET` on 31 of 32 held-out genomes where the shipped pair clears 16, and it
    leaves the shipped genome 0.444 of layer-width margin where the best global candidate
    §68 weighed leaves 0.056.

    Raises when the genome has no cliff, rather than falling back to a constant: a
    fallback is where the caller stops being able to tell the rule's answer from a
    default, and the study's version of this fallback was firing on three of thirty-two
    genomes that turned out to HAVE cliffs, just outside its bracket (§82).

    AND SINCE §88 THE PAIR IS TRACED WHEN `xp` IS, so `entry` arrives at
    `_filleted_sector_blocks` as a function of the genes instead of as a constant the
    frozen path held still.  `end` stays a module constant on both paths -- it is the
    rule's operating point and not a function of the genome.
    """
    c = layer_cliff_entry(genes, cfg, fillet=fillet, end=end, **kw)
    if c["entry"] is None:
        raise ValueError(
            f"no per-genome layer profile exists for this genome: {c['why']}.  "
            f"See PLAN.md §82.")
    # ONE EXPRESSION FOR BOTH PATHS: `layer_cliff_entry` has already narrowed its eager
    # answer to a Python float, so no `float()` is needed here and none may be used -- it
    # is what a traced entry would die on.
    return (float(factor) * c["entry"], float(end))


def _seam_table_filleted(orientation, dirn):
    """The fourteen seams of the filleted sector, in `_seam_table`'s own shape.

    Eight more entries than the unfilleted table and two fewer: each ring's
    `weld.i1 ~ free.i0` is gone, because the fillet block now separates those two blocks
    and they meet at a single POINT rather than along an edge.

    `dirn` IS NOT COSMETIC.  `sector_blocks` lays both ring blocks out in INCREASING
    theta whatever the genome does, exactly so that "the next sector" is always `k + 1`.
    This blocking lays each ring out from `theta_Q` toward the fillet instead, so the
    sector-closing seam runs to `k + dirn` -- and `flank_orientation`'s own docstring
    records that only 16 of 60 feasible genomes share the shipped `(+1, +1)`.  Written
    `dk = +1` unconditionally it closes for the shipped genome and MISSES BY A WHOLE
    SECTOR for a flipped one: 12.6 mm at the hub, 50.0 at the rim, measured.
    `build_wheel` takes `(k + dk) % n_spokes`, so a negative `dk` needs nothing from it.
    """
    eta_hub, eta_rim = orientation
    dk_hub, dk_rim = int(dirn["hub"]), int(dirn["rim"])
    return (
        ("spoke", "i0", "hub_fillet_a", "i0", 0, float(eta_hub) > 0),
        ("spoke", "i1", "rim_fillet_a", "i0", 0, float(eta_rim) > 0),
        ("hub_fillet_a", "i1", "hub_fillet_b", "i0", 0, False),
        ("rim_fillet_a", "i1", "rim_fillet_b", "i0", 0, False),
        ("hub_fillet_a", "j1", "hub_junction", "i0", 0, True),
        ("rim_fillet_a", "j1", "rim_junction", "i0", 0, True),
        ("hub_fillet_b", "j1", "hub_ring_weld", "i1", 0, True),
        ("rim_fillet_b", "j1", "rim_ring_weld", "i1", 0, False),
        ("hub_fillet_b", "i1", "hub_ring_free", "i0", 0, True),
        ("rim_fillet_b", "i1", "rim_ring_free", "i0", 0, False),
        ("hub_junction", "j0", "hub_ring_weld", "j1", 0, True),
        ("rim_junction", "j0", "rim_ring_weld", "j0", 0, True),
        ("hub_ring_free", "i1", "hub_ring_weld", "i0", dk_hub, False),
        ("rim_ring_free", "i1", "rim_ring_weld", "i0", dk_rim, False),
    )


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

# ---------------------------------------------------------------------------
# THE SECOND JUNCTION CORNER (UNCAP_PLAN.md Step 2, opt-in)
# ---------------------------------------------------------------------------
#
# WHAT IS WRONG WITH THE END CAP.  The mesh closes the spoke at the ring circle with a
# half end cap, so its second corner per junction sits at the CENTRELINE endpoint --
# theta 0 by construction.  The shipped part has no cap: `wheel_step_export._embed`
# drives both flanks straight through the ring, and its second corner is where the FAR
# FLANK crosses.  Measured (PLAN §35, `make junction`), the cap is wrong by 28.71 deg of
# wedge at the hub and 87.53 at the rim, and the rim error is 84-88 deg on every genome
# in the design history.  In Williams terms the mesh models lambda = 0.5081 where the
# part has 0.6977 -- stress ~ r^-0.492 against r^-0.302 -- which is why the wheel's
# global peak von Mises sits on that corner rather than on a corner the part has.
#
# WHY THIS IS ALLOWED TO BE SMOOTH, when reproducing `_embed` is not.  `_embed`'s
# non-differentiability is entirely in its ARGMAX: a 20001-point length scan, plus a
# 21-point blend scan at the rim.  NEITHER IS NEEDED HERE.
#
#   * the LENGTH is unnecessary -- we want the RING crossing, not `_embed`'s target
#     (12.20 / 50.25), and a ray/circle crossing is a closed-form quadratic;
#   * the DIRECTION at a FIXED blend is a smooth function of the genes, and `_embed`
#     itself pins the blend per ring: its hub branch is hard-coded to `(1.0,)`
#     ("radial-inward always reaches, so the search below is a single step at the hub")
#     and its rim branch searches upward from 0.0.
#
# So: blend 1.0 (radial) at the hub, blend 0.0 (the shared end tangent, which for a band
# of varying thickness is exactly the CENTRELINE tangent) at the rim.  No search, no
# argmax, differentiable end to end, and `mesh_coords` can carry it.
#
# THE RESIDUAL, STATED.  `_embed`'s rim blend is genome-dependent -- 0.0 on seven of the
# eight genomes checked, nonzero on `best_solution_ga_beam`.  Fixing it at 0.0 is the
# deliberate choice: reproducing the search is what M7 forbids, and the two candidates
# differ by under 1 deg of wedge against the end cap's 87, so a fixed blend is still two
# orders of magnitude better than the cap everywhere tested.
#
# WHAT THIS DOES NOT DO: it does not add `_embed`'s buried gusset.  The material inside
# the ring circle is still not modelled, so the docstring's ~1.4% deficit is REDUCED (by
# the sliver between the old cap and the new corner) but not closed.  This moves the
# CORNER, which is what the stress field is about.


def _uncap_blend(uncap, is_hub):
    """Resolve an `uncap=` argument to a blend for one ring, or `None` for "capped".

    Factored out because `modelled_area_reference` has to resolve it EXACTLY as
    `sector_blocks` does.  Two copies of this three-line branch is how the mesh and its
    own area reference end up describing different regions -- which is the failure this
    whole flip had to fix once already.
    """
    want = (uncap[0] if is_hub else uncap[1]) if isinstance(
        uncap, (tuple, list)) else uncap
    if want is False or want is None:
        return None
    # `True` means "whatever blend `_embed` itself uses at this ring".
    return 1.0 if want is True and is_hub else 0.0 if want is True else float(want)


def _uncap_corner(sample, s_ring, eta, ring_r, is_hub, blend, xp):
    """Where the FAR flank meets the ring circle, continued along `_embed`'s direction.

    `blend` is `_embed`'s OWN parameter, and means the same thing: 0.0 is the shared end
    tangent, 1.0 is the ring's radial, and the direction is their normalised mix.  Using
    the exporter's parameter rather than inventing one is what makes "faithful" checkable
    -- `_embed` lands on 1.0 at the hub (its inward branch is hard-coded to a single
    step) and on 0.0 at the rim.

    THE TRADE-OFF THE BLEND CONTROLS, measured.  A blend near 0 is FAITHFUL -- it puts
    the corner within ~1 deg of the part's wedge -- but a tangent continuation is by
    definition SMOOTH, so no corner exists there and the junction block degenerates from
    a quadrilateral into a curvilinear triangle with a ~180 deg vertex.  Measured, the
    shared tangent sits 0.587 deg (hub) and 0.489 deg (rim) off the far flank's own
    tangent, and the resulting `min_scaled_jacobian` is 0.0072.  A blend near 1 leaves a
    real corner (116.67 deg at the hub, 53.02 at the rim) and a well-shaped block, at the
    cost of fidelity.  At the HUB the two agree -- `_embed` uses 1.0, which is also the
    well-shaped choice, so the hub is free.  At the RIM they do not, and that is the
    whole of UNCAP_PLAN Step 2's residual.

    The root is selected by WHICH SIDE the spoke is on rather than by comparing the two
    roots, so there is no branch on a computed value and the expression is differentiable:

      hub   spoke OUTSIDE the ring, ray inward, crosses twice -> the NEAR root
      rim   spoke INSIDE the ring, ray outward, crosses once  -> the positive root
    """
    far_end = sample(xp.asarray(s_ring), xp.asarray(-eta))
    P_c = sample(xp.asarray(s_ring), xp.asarray(0.0))
    sign = -1.0 if is_hub else 1.0
    radial = sign * P_c / xp.sqrt(xp.sum(P_c * P_c))

    # blend 0.0: the mean of the two end tangents.  For a band of varying thickness the
    # two flank tangents average to the CENTRELINE tangent exactly, so take it there and
    # avoid depending on the exporter's 48-point downsample.
    # `sign` is -1 at the hub and +1 at the rim, and the spoke runs from s=0 to s=1, so
    # `s_ring - sign*h` always steps INTO the spoke and `P_c - a0` always points OUT of
    # it -- the direction the flank has to be continued.
    h = 1.0e-4
    a0 = sample(xp.asarray(s_ring - sign * h), xp.asarray(0.0))
    tan = (P_c - a0) / xp.sqrt(xp.sum((P_c - a0) * (P_c - a0)))

    d = (1.0 - blend) * tan + blend * radial
    d = d / xp.sqrt(xp.sum(d * d))

    b = 2.0 * xp.sum(far_end * d)
    c = xp.sum(far_end * far_end) - ring_r * ring_r
    disc = b * b - 4.0 * c
    return far_end + (0.5 * (-b + sign * xp.sqrt(xp.maximum(disc, 0.0)))) * d


# THE DEFAULT, ADOPTED 2026-08-18 (UNCAP_PLAN Step 4, PLAN.md 38).
#
# Read it as "hub: whatever `_embed` itself uses; rim: radial, DELIBERATELY NOT what
# `_embed` uses".  The asymmetry is the whole of Step 2's finding and is not a typo:
#
#   hub   `_embed`'s blend IS 1.0, and 1.0 is also the well-shaped choice, so the hub is
#         faithful and free at the same time -- corner error 28.71 deg -> 0.01 deg,
#         `min_sj` unchanged, `max_aspect_ratio` 20.2715 -> 16.2400.
#   rim   `_embed`'s blend is 0.0, which is FAITHFUL (1.06 deg) and UNBUILDABLE.  A
#         tangent continuation is by definition smooth, so no corner exists there, the
#         block degenerates from a quadrilateral into a curvilinear triangle with a
#         179.35 deg vertex, and `min_sj` collapses 0.782505 -> 0.007208 -- far under
#         `wheel_objective.MIN_SJ_TARGET` (0.2, barrier weight 3000).  The blend sweep
#         has NO OVERLAP between "the peak leaves the artefact" (blend <= 0.15) and "the
#         0.2 gate clears" (blend >= ~0.23), so there is no compromise value to ship.
#
# 1.0 at the rim is therefore not an improvement at the rim; it is a deliberate refusal
# to trade mesh validity for fidelity there, and it leaves the rim corner exactly where
# the capped mesh had it.  WHAT THE FLIP ACTUALLY BUYS IS THE HUB -- and, because the
# peak CASCADES off whichever artefact is worst, a global maximum that lands on a corner
# the part really has.  Fixing the rim properly needs the junction to stop being one
# four-sided block; that is priced in UNCAP_PLAN Step 3, and measured NOT to bind.
#
# Passing `uncap=False` still reproduces the pre-2026-08-18 geometry BIT-FOR-BIT, and is
# what the `capped` rows of `studies/study_junction_agreement.py` report.
UNCAP_DEFAULT = (True, 1.0)


def sector_blocks(genes, cfg, xp=np, span_mm=HUB_RIM_SPAN_MM, orientation=None,
                  rim_outer=RIM_OUTER_RADIUS_MM, fillet=None, uncap=UNCAP_DEFAULT,
                  fillet_blocking="sector", layer_profile=None,
                  fillet_clamp=SECTOR_FIT_CLAMP, fillet_roots=None):
    """The seven node grids of sector 0 — eleven when the fillet is blocked — as a dict.

    Ordering matters: `build_wheel` gives ownership of a shared node to the block that
    appears FIRST, so the spoke owns its end cross-sections and the collar owns its
    inner boundary.  That is only a labelling choice — `check_seams` verifies the
    discarded coordinates agreed — but it keeps the owner predictable when debugging.

    `uncap` replaces each junction's half END CAP with the far flank's own continuation
    to the ring circle (UNCAP_PLAN.md).  `UNCAP_DEFAULT` -- read its comment, the hub and
    rim entries do NOT mean the same thing -- is the default since 2026-08-18; `False`
    restores the pre-flip geometry bit-for-bit; `True` does both rings at `_embed`'s own
    blend and is NOT buildable at the rim; a `(hub, rim)` pair does them independently,
    which is what the pricing needed because THE TWO RINGS DO NOT BEHAVE THE SAME.

    `fillet` rounds the `P_t` corner of each junction (FILLET_PLAN.md).  `None` is the
    unfilleted geometry and is the default; `True` takes the radii from the genome's
    `R_hub`/`R_rim`; a `(R_hub, R_rim)` pair overrides them.

    `fillet_clamp` pulls a gene-derived radius back inside the room its own sector has --
    see `SECTOR_FIT_CLAMP`, which is where the number and its provenance live.  It acts on
    `fillet=True` ONLY, never on an explicit pair, and `None` disables it.  The mesh
    reports what it actually built: `mesh.fillet_radii_mm` and `mesh.fillet_clamped`.

    `fillet_blocking` picks WHICH filleted construction, and the two are not variants of
    one thing — they are the arc's before and after.

      `"sector"` (the default) is PART 10's: ELEVEN blocks, the fillet carried by a
      boundary-layer pair whose corners are off both tangencies, and the ring re-cut
      around it.  This is the one that meshes.

      `"spoke"` is PART 3's, the construction this module shipped from 2026-08-17 and
      §47 retired: the arc goes onto the spoke block's own flank edge and ONLY THE SPOKE
      BLOCK is built differently, because the junctions and the ring blocks both derive
      `theta_t` from the spoke's end row.  It is kept reachable and must stay reachable:
      `make fillet` measures the radius at which it folds, and that measurement is
      PART 6's.  A zero or negative radius switches that end off, which `"sector"` does
      not allow.
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
    n_sp = cfg.nn(cfg.n_span)
    eta_grid = xp.linspace(-1.0, 1.0, cfg.nn(cfg.n_thick))
    if fillet is not None and fillet_blocking == "sector":
        if xp is not np and fillet_roots is None:
            raise NotImplementedError(
                "the filleted blocking's tangency and crossing solves are BRACKETED, so "
                "there is nothing to trace until their roots have been found once "
                "concretely.  Pass `fillet_roots=` — the `_roots` record an eager build "
                "returns — which is what `mesh_coords` does for a filleted mesh; see "
                "`_fillet_curves` and PLAN.md §79.")
        # `build_wheel` has already resolved this and passes a concrete pair; a caller
        # reaching `sector_blocks` directly gets the same default here (§85).  Resolving
        # twice is not possible: a pair is returned unchanged.
        #
        # AND ON THE TRACED PATH `layer_profile` IS `None` AGAIN, DELIBERATELY (§88).
        # `_filleted_gradient_recipe` hands the rule back rather than the pair it produced,
        # so the entry is re-resolved HERE, under `xp`, from this genome's own tangency
        # scalars.  `sector=` passes the sampler and stations already built four lines up,
        # so the trace pays for one of each and not two.
        return _filleted_sector_blocks(sample, cfg, s_hub, s_rim, orientation,
                                       rim_inner, rim_outer, genes, fillet, uncap,
                                       *_resolve_layer_profile(
                                           layer_profile, genes, cfg, fillet,
                                           span_mm=span_mm, orientation=orientation,
                                           rim_outer=rim_outer, uncap=uncap,
                                           clamp=fillet_clamp,
                                           fillet_blocking=fillet_blocking,
                                           xp=xp, roots=fillet_roots,
                                           sector=(sample, s_hub, s_rim, rim_inner)),
                                       fillet_clamp, xp, fillet_roots)
    if fillet is None:
        s_grid = xp.linspace(s_hub, s_rim, n_sp)
        spoke = sample(s_grid[:, None], eta_grid[None, :])
    elif fillet_blocking == "spoke":
        spoke = _filleted_spoke(sample, s_hub, s_rim, orientation, rim_inner, n_sp,
                                n_th, genes, fillet, xp)
    else:
        raise ValueError(f"fillet_blocking must be 'sector' or 'spoke', not "
                         f"{fillet_blocking!r}")

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

        # `Q` is the block's outer corner on the ring circle.  Unfilleted it is the
        # centreline endpoint `P_c` and the right edge is the half END CAP; under
        # `uncap` it is where the FAR FLANK crosses, and the right edge becomes the
        # flank's own continuation to it.  Everything else about the block -- its four
        # sides, their node counts, and every seam entry that reads them -- is unchanged,
        # which is why this is a two-line change rather than a topology change.
        is_hub = label == "hub_junction"
        blend = _uncap_blend(uncap, is_hub)
        Q = (P_c if blend is None else
             _uncap_corner(sample, s_ring, eta, radius, is_hub, blend, xp))

        th_t = xp.arctan2(P_t[1], P_t[0])
        th_q = xp.arctan2(Q[1], Q[0])
        s_flank = xp.linspace(s_end, s_ring, n_weld)
        blocks[label] = coons_patch(
            bottom=arc_points(radius, th_t, th_q, n_weld, xp=xp),
            top=sample(s_flank, xp.zeros_like(s_flank) - eta),
            left=cross,
            right=_lerp_points(Q, far_end, n_th, xp),
            xp=xp)
        thetas[label] = (th_t, th_q)
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
                 "uncap", "fillet", "fillet_radii_mm", "fillet_clamped",
                 "fillet_recipe", "_coord_fn")

    def __init__(self, coords, conn, cfg, element_block, element_region,
                 node_sets, edge_sets, seam_error_mm, n_merged,
                 rim_outer=RIM_OUTER_RADIUS_MM, genes=None,
                 span_mm=HUB_RIM_SPAN_MM, n_spokes=NUMBER_OF_SPOKES,
                 owners=None, orientation=None, phase_deg=0.0,
                 uncap=UNCAP_DEFAULT, fillet=None, applied=None,
                 fillet_recipe=None):
        # Carried so `area_report` can ask `modelled_area_reference` for the region this
        # mesh ACTUALLY builds.  A mesh that does not remember how its junctions were
        # closed cannot be area-checked against anything.
        self.uncap = uncap
        # Carried for the same reason `uncap` is, and for one more: `mesh_coords` would
        # otherwise rebuild the UNFILLETED sector and index it with THIS mesh's owners.
        self.fillet = fillet
        # WHAT WAS ACTUALLY BUILT, not what was asked for.  `fillet=True` with a radius
        # that has no room in its own sector builds a SMALLER fillet (see
        # `SECTOR_FIT_CLAMP`), and a consumer pricing `R_hub` against a deflection has to
        # be able to see that -- §57's condition on the clamp being honest.  Both are
        # `None` off the filleted path, which is not the same as "nothing was clamped".
        self.fillet_radii_mm = None if applied is None else applied["radii_mm"]
        self.fillet_clamped = None if applied is None else applied["clamped"]
        # EVERYTHING PAST `fillet=` THAT MOVES A NODE, so that `mesh_coords` can re-derive
        # THIS mesh rather than the module default's -- the blocking, the layer profile,
        # the clamp factor, and the frozen root record `_fillet_curves` needs to run
        # without a bracketed search.  Same argument as `uncap` below, which was missing
        # from that list once and cost 0.448 mm before anything noticed.
        self.fillet_recipe = fillet_recipe
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
        # The FOUR things `mesh_coords` needs to rebuild these coordinates without
        # redoing a single discrete decision: which raw node owns each merged one, which
        # way the flanks were oriented, the phase the wheel was rolled to, and — since
        # 2026-08-19 — `uncap` (set just above).  Carried on the mesh rather than
        # recomputed so the differentiable path cannot silently end up describing a
        # DIFFERENT mesh from the one that was solved.
        #
        # `uncap` WAS MISSING FROM THAT LIST UNTIL 2026-08-19, AND THE OMISSION WAS
        # INVISIBLE UNTIL PLAN.md §38.  While `UNCAP_DEFAULT` was `False`, the module
        # default and every mesh's value agreed by construction, so `_sector_coords`
        # taking its own default could not disagree with the mesh it was handed.  §38
        # flipped the default to `(True, 1.0)`; the shipped path still agreed, but a mesh
        # built with an explicit `uncap=False` then got traced coordinates from the OTHER
        # geometry — measured at **0.448 mm**, against the 1e-9 mm this path is documented
        # to hold to.  Nothing shipped was wrong, because nothing on the shipped path
        # builds a non-default mesh; the capped-vs-faithful STUDIES are what it breaks,
        # which is precisely the comparison §38 exists to make.
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
                   phase_deg, fillet=None, uncap=UNCAP_DEFAULT,
                   fillet_blocking="sector", layer_profile=None,
                   fillet_clamp=SECTOR_FIT_CLAMP, fillet_roots=None):
    """The raw node coordinates of all twelve sectors, before the seams are merged.

    THE ENTIRE TRACED HALF OF `build_wheel`, factored out so that `mesh_coords` can run
    it again under `jax.grad` without also re-running the eager half.  Nothing here
    makes a discrete decision: the orientation is an argument, the block shapes come out
    of `sector_blocks`, and every arithmetic operation goes through `xp`.

    Returns (coords_all [n_raw, 2], shapes, offsets, thetas, dirn, applied, roots), where
    `dirn` is `None` for the unfilleted blocking and the per-junction sweep direction for
    the filleted one — `build_wheel` needs it to pick the seam table AND to set that
    table's `dk`, which follows the genome (see `_seam_table_filleted`) — `applied` is the
    sector-fit clamp's record and `roots` the frozen root record, both `None` off the
    filleted path.
    """
    sector0 = sector_blocks(genes, cfg, xp=xp, span_mm=span_mm,
                            orientation=orientation, rim_outer=rim_outer,
                            fillet=fillet, uncap=uncap,
                            fillet_blocking=fillet_blocking,
                            layer_profile=layer_profile,
                            fillet_clamp=fillet_clamp, fillet_roots=fillet_roots)
    thetas = sector0.pop("_thetas")
    dirn = sector0.pop("_dirn", None)
    applied = sector0.pop("_applied", None)
    roots = sector0.pop("_roots", None)
    order = FILLETED_BLOCK_ORDER if dirn is not None else BLOCK_ORDER

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
        for name in order:
            g = _rotate(sector0[name], angle, xp) if (k or phase_deg) else sector0[name]
            shapes[(k, name)] = (int(g.shape[0]), int(g.shape[1]))
            offsets[(k, name)] = cursor
            cursor += shapes[(k, name)][0] * shapes[(k, name)][1]
            parts.append(g.reshape(-1, 2))

    return xp.concatenate(parts, axis=0), shapes, offsets, thetas, dirn, applied, roots


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

    A FILLETED MESH IS DIFFERENTIABLE SINCE 2026-08-24 (PLAN.md §79) and freezes
    two more decisions than an unfilleted one does — the tangency and ring-crossing roots
    are seeded from the eager build and refined by one Newton step, and the four
    geometric refusals `_fillet_curves` can make are not re-tested.  Same argument as the
    flank orientation above; `_filleted_gradient_recipe` says what is still refused.

    AND SINCE §88 THAT INCLUDES THE DEFAULT ONE.  `fillet=True` with no `layer_profile` is
    the per-genome rule, whose entry is `FILLET_LAYER_CLIFF_FACTOR * cliff(genes)`; §85
    refused it because the frozen path held that entry constant.  The cliff is a closed
    form in two scalars the tangency solve already produces, so it is re-resolved inside
    the trace and the entry follows the genes — one more root-find NOT frozen, rather than
    one more thing frozen.

    The default path goes through `coord_fn` and is JITTED, which is not an optimisation
    detail: see that function for the measurement.
    """
    if xp is None:
        return coord_fn(mesh)(genes)

    rec = _filleted_gradient_recipe(mesh)
    coords_all, *_ = _sector_coords(
        genes, mesh.cfg, xp, mesh.span_mm, mesh.n_spokes, mesh.orientation,
        mesh.rim_outer, mesh.phase_deg, uncap=getattr(mesh, "uncap", UNCAP_DEFAULT),
        **rec)
    return coords_all[xp.asarray(mesh.owners)]


# The frozen record's field order, and the two junctions'.  It is a FLAT VECTOR on the
# traced path -- see `coord_fn` on why closing over these numbers instead would retrace
# the jaxpr at every genome.
_FILLET_ROOT_JUNCTIONS = ("hub", "rim")
_FILLET_ROOT_FIELDS = ("s_A", "u_N", "void_sign", "dd_turns", "sweep_sign",
                       "th_B_turns", "th_N_turns", "dirn")


def _fillet_roots_vector(roots):
    return np.array([float(roots[j][f]) for j in _FILLET_ROOT_JUNCTIONS
                     for f in _FILLET_ROOT_FIELDS], dtype=float)


def _fillet_roots_from_vector(vec, clamped):
    n = len(_FILLET_ROOT_FIELDS)
    return {j: dict({f: vec[i * n + k] for k, f in enumerate(_FILLET_ROOT_FIELDS)},
                    clamped=bool(clamped[i]))
            for i, j in enumerate(_FILLET_ROOT_JUNCTIONS)}


def _filleted_gradient_recipe(mesh):
    """What `_sector_coords` needs to re-derive THIS mesh, or a refusal saying why not.

    `_sector_coords` would otherwise be called WITHOUT `fillet`, so for a filleted mesh
    it would happily rebuild the UNFILLETED sector, take `mesh.owners` — which index a
    different and longer node list — and return coordinates that are neither the mesh's
    nor an error.  Everything past `fillet=` matters for the same reason `uncap` does
    (see `WheelMesh.__init__`): the layer profile and the clamp factor both move nodes,
    and a traced path taking the module default would describe a different mesh.

    TWO FILLETED MESHES ARE STILL REFUSED, both because the derivative would be wrong
    rather than because it would be hard:

      `fillet_blocking="spoke"` is PART 3's retired construction, and its
      `_filleted_spoke` re-spreads the station vector by ROUNDING a node count — a step
      function of the radius, whose derivative is zero almost everywhere and undefined
      at the jumps.  Nothing but `make fillet` builds it.

      A mesh whose radius the sector-fit clamp MOVED (`SECTOR_FIT_CLAMP`).  There the
      built radius is `factor * limit(genome)` and does not follow `R_hub` at all, so the
      honest gradient in that gene is zero and the honest one in the centreline genes
      runs through the limit's own bisection.  Neither is what freezing the record gives,
      and a plausible wrong length is the failure `wheel_adjoint`'s header is about.  The
      clamp is inert at the shipped genome and over the whole span §75 priced.

    A THIRD CASE WAS REFUSED HERE FROM §85 TO §88 AND IS NOT ANY MORE: the PER-GENOME
    layer profile, whose entry is `FILLET_LAYER_CLIFF_FACTOR * cliff(genes)`.  The cliff
    turned out to be a closed form in two scalars the tangency solve already produces
    (`_layer_cliff_from_scalars`), so it is differentiated rather than held constant.
    THE RULE IS HANDED BACK, NOT THE PAIR IT PRODUCED — `layer_profile: None` — because
    passing the resolved pair is exactly the frozen constant this stopped being.  A
    non-per-genome mesh still passes its pair, and `coord_fn`'s cache key tells the two
    apart by that `None`.
    """
    if getattr(mesh, "fillet", None) is None:
        return {}
    rec = getattr(mesh, "fillet_recipe", None) or {}
    if rec.get("roots") is None:
        raise NotImplementedError(
            "mesh_coords: this mesh was built with `fillet=` and "
            f"`fillet_blocking={rec.get('blocking')!r}`, which carries no root record.  "
            "Only the eleven-block sector blocking has a differentiable path; see "
            "`_filleted_gradient_recipe` and PLAN.md §79.")
    if any(rec["roots"][j]["clamped"] for j in _FILLET_ROOT_JUNCTIONS):
        raise NotImplementedError(
            "mesh_coords: the sector-fit clamp moved this mesh's fillet radius, so the "
            f"radii it was built at {mesh.fillet_radii_mm} are not the genes' "
            "and do not follow them.  The differentiable path would return a gradient "
            "that is plausible and wrong; see `_filleted_gradient_recipe` and PLAN.md "
            "§79.")
    return {"fillet": mesh.fillet, "fillet_blocking": rec["blocking"],
            "layer_profile": None if rec.get("layer_profile_per_genome")
                             else rec["layer_profile"],
            "fillet_clamp": rec["clamp"], "fillet_roots": rec["roots"]}


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

    A FILLETED MESH'S FROZEN ROOTS ARE A TRACED ARGUMENT TOO, and that is the whole
    reason this function still has a cache to speak of.  They depend on the genome — a
    different design has a different tangency station — so closing over them would put a
    design at fixed genes into the jaxpr and re-trace on every finite difference and
    every optimizer step, which is the exact failure the paragraph above describes.
    Passed in as an array instead, the traced recipe is genome-independent again and the
    key stays what it was, plus the blocking and layer profile, which are static and do
    change the geometry.  Measured on the `coarse` filleted mesh, second genome onward:
    2.78 s to trace once, 0.0095 s per call after -- against 2.29 s and 0.0012 s for the
    unfilleted mesh, and against 2.78 s EVERY call if the roots are closed over.
    """
    rec = _filleted_gradient_recipe(mesh)
    import jax_config  # noqa: F401  — x64 must be set before the first trace
    import jax
    import jax.numpy as jnp

    cfg, span, n_spokes = mesh.cfg, mesh.span_mm, mesh.n_spokes
    orientation, rim_outer, phase = mesh.orientation, mesh.rim_outer, mesh.phase_deg
    uncap = getattr(mesh, "uncap", UNCAP_DEFAULT)
    owners_np = np.asarray(mesh.owners)
    # `repr(rec["layer_profile"])` RAW, NOT THROUGH `_layer_profile` (§88).  That helper
    # maps `None` onto the shipped constants, and since §88 `None` in this record means
    # THE PER-GENOME RULE, resolved inside the trace: mapping it would key a per-genome
    # mesh identically to a shipped-pair one and hand the second the first's geometry.
    # The raw `None` also makes the key genome-INDEPENDENT again, which is the whole
    # point of the paragraph above -- a resolved pair in the key re-traces per genome.
    fillet_key = () if not rec else (
        repr(rec["fillet"] if rec["fillet"] is True else tuple(rec["fillet"])),
        rec["fillet_blocking"], repr(rec["layer_profile"]))
    # `repr(uncap)` IS IN THE KEY, and leaving it out is not a cache miss but a WRONG
    # ANSWER: two meshes can share every entry above and differ only in `uncap`, and the
    # second would then be handed the first's traced geometry.  See `WheelMesh.__init__`.
    key = (cfg.name, cfg.order, cfg.n_curve, cfg.n_span, cfg.n_thick, cfg.n_weld,
           cfg.n_collar_r, cfg.n_collar_free, cfg.n_rim_r, cfg.n_rim_free,
           float(span), int(n_spokes), float(rim_outer), float(phase),
           repr(uncap), np.asarray(orientation).tobytes(), owners_np.tobytes(),
           fillet_key)

    if mesh._coord_fn is not None:
        return mesh._coord_fn
    traced = _COORD_FN_CACHE.get(key)
    if traced is None:
        owners = jnp.asarray(owners_np)

        if not rec:
            @jax.jit
            def traced(v):                          # noqa: F811
                coords_all, *_ = _sector_coords(v, cfg, jnp, span, n_spokes,
                                                orientation, rim_outer, phase,
                                                uncap=uncap)
                return coords_all[owners]
        else:
            static = dict(rec)
            static.pop("fillet_roots")
            clamped = tuple(False for _ in _FILLET_ROOT_JUNCTIONS)

            @jax.jit
            def traced(v, root_vec):                # noqa: F811
                coords_all, *_ = _sector_coords(
                    v, cfg, jnp, span, n_spokes, orientation, rim_outer, phase,
                    uncap=uncap,
                    fillet_roots=_fillet_roots_from_vector(root_vec, clamped),
                    **static)
                return coords_all[owners]

        if len(_COORD_FN_CACHE) >= _COORD_FN_CACHE_MAX:
            _COORD_FN_CACHE.pop(next(iter(_COORD_FN_CACHE)))
        _COORD_FN_CACHE[key] = traced
    if rec:
        root_vec = jnp.asarray(_fillet_roots_vector(rec["fillet_roots"]))
        f = functools.partial(traced, root_vec=root_vec)
    else:
        f = traced
    mesh._coord_fn = f
    return f


def build_wheel(genes, cfg="coarse", xp=np, span_mm=HUB_RIM_SPAN_MM,
                n_spokes=NUMBER_OF_SPOKES, orientation=None,
                rim_outer=RIM_OUTER_RADIUS_MM, phase_deg=0.0, fillet=None,
                uncap=UNCAP_DEFAULT, fillet_blocking="sector", layer_profile=None,
                fillet_clamp=SECTOR_FIT_CLAMP):
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
    # THE PROFILE IS SETTLED HERE, ONCE, AND WHAT GOES INTO `fillet_recipe` BELOW IS THE
    # RESOLVED PAIR (§85).  Leaving `None` in that record would leave the jax cache key --
    # which reads it -- identical for two genomes the per-genome rule gives different
    # profiles to.  See `_resolve_layer_profile`.
    per_genome = (fillet is True and layer_profile is None
                  and fillet_blocking == "sector")
    layer_profile = _resolve_layer_profile(
        layer_profile, genes, cfg, fillet, span_mm=span_mm, orientation=orientation,
        rim_outer=rim_outer, uncap=uncap, clamp=fillet_clamp,
        fillet_blocking=fillet_blocking)
    coords_all, shapes, offsets, thetas, dirn, applied, fillet_roots = _sector_coords(
        genes, cfg, xp, span_mm, n_spokes, orientation, rim_outer, phase_deg,
        fillet, uncap, fillet_blocking, layer_profile, fillet_clamp)
    filleted = dirn is not None
    order = FILLETED_BLOCK_ORDER if filleted else BLOCK_ORDER
    region = FILLETED_BLOCK_REGION if filleted else BLOCK_REGION
    seams = (_seam_table_filleted(orientation, dirn) if filleted
             else _seam_table(orientation, thetas))
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
        for name in order:
            ni, nj = shapes[(k, name)]
            c = global_id[_mesh.grid_connectivity(ni, nj, cfg.order)
                          + offsets[(k, name)]]
            conn_blocks.append(c)
            elem_block.extend([name] * c.shape[0])
            elem_region.extend([region[name]] * c.shape[0])
    conn = np.concatenate(conn_blocks, axis=0).astype(np.int32)
    conn = _orient_elements(np.asarray(coords), conn, np.asarray(elem_block))

    node_sets = _node_sets(shapes, offsets, global_id, n_spokes, filleted)
    edge_sets = _edge_sets(shapes, offsets, global_id, cfg, n_spokes, filleted)
    return WheelMesh(coords, conn, cfg, np.asarray(elem_block),
                     np.asarray(elem_region), node_sets, edge_sets, seam_error,
                     int(n_raw - owners.size), rim_outer=rim_outer,
                     genes=genes, span_mm=span_mm, n_spokes=n_spokes,
                     owners=owners, orientation=orientation, phase_deg=phase_deg,
                     uncap=uncap, fillet=fillet, applied=applied,
                     fillet_recipe=None if fillet is None else {
                         "blocking": fillet_blocking, "layer_profile": layer_profile,
                         # Whether that pair came from the RULE or was asked for, which
                         # the pair itself cannot say and `mesh_coords` has to know (§85).
                         "layer_profile_per_genome": bool(per_genome),
                         "clamp": fillet_clamp, "roots": fillet_roots})


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

    WHAT THIS CHECK DOES NOT SEE, MEASURED (`make fillet`, 2026-08-22).  `_signed_area`
    is a shoelace over each element's FOUR CORNERS.  Every config here is `order=2`, so
    one Q9 element spans 2x2 cells and its five mid nodes take no part in that sum: an
    element can fold INSIDE while its corner shoelace stays comfortably positive.
    Sampled over the gene box at `coarse`, 21% of the meshes this function accepts have
    a non-positive `det J` at a Gauss point the assembly integrates -- the very failure
    the paragraph above says it exists to prevent -- and `wheel_mesh.scaled_jacobian`,
    which is corner-only for the same reason, calls a few of those healthy.  What
    actually covers the default path is neither: it is `fold_margin > 0`, which rejects
    the genome before the mesh exists (`studies/study_mesh_quality.py`'s `meshable`).
    That constraint reads genes 0-11 and cannot help the `fillet=` path at all, which is
    why the blind spot is wide open there.  STRENGTHENING THIS CHECK IS NOT THIS ARC'S
    WORK and is not done here; it is recorded so nobody reads a `build_wheel` that
    returns as a mesh that integrates.
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


# The ring blocks' names change under the filleted blocking and their RADIAL ORDER does
# not: `hub_collar_*` and `hub_ring_*` both run bore -> ring circle, `rim_band_*` and
# `rim_ring_*` both run ring circle -> tyre surface.  That is why these three sets can
# stay keyed on side, and it is a property the blocking was built to preserve rather
# than one it happened to have -- `tests/test_fillet_block.py` pins it on the
# coordinates.
_RING_BLOCKS = {False: ("hub_collar_weld", "hub_collar_free",
                        "rim_band_weld", "rim_band_free"),
                True: ("hub_ring_weld", "hub_ring_free",
                       "rim_ring_weld", "rim_ring_free")}


def _edge_sets(shapes, offsets, global_id, cfg, n_spokes, filleted=False):
    """Boundary edge segments in the compacted global numbering."""
    hub_w, hub_f, rim_w, rim_f = _RING_BLOCKS[bool(filleted)]
    def gather(pairs):
        out = []
        for name, side in pairs:
            for k in range(n_spokes):
                segs = _boundary_segments(shapes[(k, name)], side, cfg.order)
                out.append(global_id[segs + offsets[(k, name)]])
        return np.concatenate(out, axis=0)

    return {
        # r = RIM_OUTER: where the ground pushes.
        "rim_outer": gather([(rim_w, "j1"), (rim_f, "j1")]),
        # r = RIM_RADIUS between spokes: free surface, and the one whose shape tells you
        # whether the rim band is bending.  UNDER THE FILLET it is a shorter arc: the
        # fillet's own footprint on the ring circle is interior material now, and the
        # free surface it replaces is the fillet ARC, which is not this set.
        "rim_inner_free": gather([(rim_f, "j0")]),
        # r = HUB_RADIUS - COLLAR_DEPTH: the rigid-hub interface.
        "hub_tie": gather([(hub_w, "j0"), (hub_f, "j0")]),
    }


def _node_sets(shapes, offsets, global_id, n_spokes, filleted=False):
    """Boundary node sets the FEA needs, in the compacted global numbering."""
    hub_w, hub_f, rim_w, rim_f = _RING_BLOCKS[bool(filleted)]
    def gather(pairs):
        out = []
        for name, side in pairs:
            for k in range(n_spokes):
                out.append(global_id[_side_indices(shapes[(k, name)], side)
                                     + offsets[(k, name)]])
        return np.unique(np.concatenate(out))

    return {
        # r = HUB_RADIUS - COLLAR_DEPTH: tied to the rigid hub body.
        "hub_tie": gather([(hub_w, "j0"), (hub_f, "j0")]),
        # r = RIM_OUTER: the ground-contact surface.
        "rim_outer": gather([(rim_w, "j1"), (rim_f, "j1")]),
        # r = RIM_RADIUS between spokes: free, and the surface that tells you whether
        # the rim band is bending (M4's compliance_split).
        "rim_inner_free": gather([(rim_f, "j0")]),
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


def _uncap_reference_poly(poly, curve, hub_radius, rim_inner, uncap):
    """Continue each end's FAR flank to its ring circle, on the REFERENCE's polygon.

    WHY THIS EXISTS.  `modelled_area_reference` is the independent cross-check on the
    mesh's area, and it describes a REGION: `hub disk | rim band | bands clipped to the
    annulus`.  The band is only defined over the centreline's own parameter range, so
    its ends are the straight cross-sections -- exactly the half end caps `uncap`
    removes.  Leave the reference alone and the 2026-08-18 default flip turns
    `error_vs_modelled` from a discretisation residual (-0.024%, converging) into a
    REGION MISMATCH (+0.152%, growing under refinement, because refining the mesh
    resolves a sliver the reference does not contain at all).  That is a false red on a
    real check, and `test_area_converges_under_refinement` caught it.

    HOW MUCH INDEPENDENCE SURVIVES, STATED PLAINLY.  The direction rule is now shared
    with `_uncap_corner` -- it has to be, or the two describe different regions again.
    What stays independent is everything that made this a cross-check rather than the
    same computation twice: the flank endpoints come from `thicken_3taper_curve`'s
    FINITE-DIFFERENCE offset normals against the mesh's analytic hodograph, and the
    integration is exact shoelace plus exact circular sectors against Q9 Gauss.  The
    residual it reports is still the real difference between two constructions.

    THE CLIPPER SUPPLIES THE ARC.  Only the far flank moves; the straddling flank is
    already cut by `_clip_polygon_to_disk` at the ring circle, which is the reference's
    own `P_t`.  Extending the far endpoint to `Q` on the same circle and letting the
    clip close the region between them reproduces the junction block's [arc + flank
    continuation] boundary without this function ever constructing an arc.  That is the
    same "only one corner moves" property the mesh side has (PLAN.md 34 Finding 2).
    """
    poly = np.asarray(poly, dtype=float)
    n = len(poly) // 2
    shift = np.array([hub_radius, 0.0])
    # `poly` is [top; reversed bottom], possibly reversed WHOLE for winding.  Reversal
    # maps index i -> 2n-1-i, which leaves both cross-section EDGES as edges: the hub's
    # is always the closing edge (-1 -> 0) and the rim's is always (n-1 -> n).
    ends = (
        # is_hub, the edge's two indices, the centreline point, the point one step in
        (True, (len(poly) - 1, 0), curve[0] + shift, curve[1] + shift),
        (False, (n - 1, n), curve[-1] + shift, curve[-2] + shift),
    )
    inserts = []
    for is_hub, (ia, ib), p_c, a0 in ends:
        blend = _uncap_blend(uncap, is_hub)
        if blend is None:
            continue
        ring_r = hub_radius if is_hub else rim_inner
        sign = -1.0 if is_hub else 1.0
        # The far flank is the one on the SPOKE side of the ring circle -- outside it at
        # the hub, inside it at the rim.  Reading it off the radii rather than off
        # `flank_orientation` keeps this path independent of the mesh's own decision.
        ra, rb = np.linalg.norm(poly[ia]), np.linalg.norm(poly[ib])
        far_is_a = (ra > rb) if is_hub else (ra < rb)
        far = poly[ia] if far_is_a else poly[ib]
        radial = sign * p_c / np.linalg.norm(p_c)
        tan = (p_c - a0) / np.linalg.norm(p_c - a0)
        d = (1.0 - blend) * tan + blend * radial
        d = d / np.linalg.norm(d)
        # NOTE for any blend other than 1.0: `tan` here steps one CURVE SAMPLE (span/20000
        # mm) where `_uncap_corner` steps 1e-4 in the PARAMETER, so the two tangents differ
        # slightly and the region this reference describes would differ from the mesh's by
        # that much.  At `UNCAP_DEFAULT` the term is multiplied by (1 - 1.0) and the
        # question does not arise; at an intermediate blend, check `error_vs_modelled`
        # before trusting it.
        b = 2.0 * float(far @ d)
        c = float(far @ far) - ring_r * ring_r
        disc = b * b - 4.0 * c
        if disc < 0.0:
            continue                      # no crossing: leave this end capped
        q = far + (0.5 * (-b + sign * math.sqrt(disc))) * d
        # Q goes INTO the cross-section edge, which runs ia -> ib in loop order, so the
        # insertion position is `ib` whichever end of it `far` turned out to be:
        #   far is poly[ia]:  ... -> far -> Q -> straddling -> ...
        #   far is poly[ib]:  ... -> straddling -> Q -> far -> ...
        # Both put the continuation on the far flank and leave the straddling flank for
        # the clipper to cut at the ring circle.
        inserts.append((ib, q))
    for at, q in sorted(inserts, key=lambda t: -t[0]):
        poly = np.insert(poly, at, q, axis=0)
    return poly


def _fillet_reference_wedge(flank, mid, ring_r, void_sign, R):
    """Area ONE junction's fillet ADDS to the region `modelled_area_reference` describes.

    THE REGION, WHICH IS THE WHOLE OF WHY THIS IS NOT A WEDGE FORMULA.  Unfilleted, the
    material's boundary turns a corner where the STRADDLING flank crosses its ring
    circle: flank on one side, circle on the other, void between them.  The sector
    blocking replaces that corner with an arc tangent to both legs, so the material gains
    exactly the curvilinear triangle

        A -> F   along the flank      (`A` the tangency point, `F` the ring crossing)
        F -> B   along the ring circle
        B -> A   along the fillet arc

    and nothing else -- every other boundary the filleted blocking builds (the far flank,
    the cut to the ring's far radius, the radial dive) is a CUT BETWEEN BLOCKS and lies
    strictly inside the region either way.  Verified against the mesh rather than
    asserted: at the shipped genome the residual against `area_report`'s measured total
    falls 0.4412% -> 0.1309% -> 0.0602% -> 0.0362% down the config ladder, and holding
    `fine` while sweeping `n_thick` 8 -> 64 -- the direction that resolves the arcs --
    drives it to -0.0024%, against the UNFILLETED mesh's own -0.0025% at that config.
    A region that was wrong would converge to that error instead of to zero.  See
    PLAN.md §86, which is §50's ranked item, and FILLET_PLAN.md STEP 1 RECORD PART 28.

    `flank` and `mid` are the straddling flank and the centreline, both ordered FROM the
    ring end INTO the spoke, and both from `thicken_3taper_curve`'s own samples -- so the
    tangency is re-solved here on the EXPORTER's finite-difference offset normals rather
    than read off the mesh's analytic hodograph, which is what keeps this an independent
    cross-check instead of the same computation twice.  Measured, the two paths put the
    twelve spokes' fillets at 139.16025 and 139.16031 mm2.

    THE OFFSET DIRECTION IS THE CENTRELINE'S NORMAL, NOT THE FLANK'S, and that is a
    faithful copy of `_fillet_centre` rather than an approximation of it: `C` is the flank
    point pushed `R` along `flank - mid`, exactly as the mesh's tangency solve pushes it,
    so a band of varying width gives the same not-quite-perpendicular circle on both
    paths.  Copying the construction is the point; "improving" it here would measure the
    difference between two fillets instead of between two integrations of one.

    Returns `(area_mm2, i)`, `i` being the flank index the tangency landed in -- the
    caller needs it to tell the two junctions' fillets from a spoke they have eaten.
    """
    r = np.hypot(flank[:, 0], flank[:, 1])
    # `void_sign` is +1 where the spoke is OUTSIDE its ring (the hub) and -1 where it is
    # inside (the rim), so this one expression is "how far into the spoke" at both ends.
    side = (r - ring_r) * void_sign
    cross = np.nonzero(side[:-1] * side[1:] <= 0.0)[0]
    if cross.size == 0:
        raise ValueError(
            f"the straddling flank never crosses r={ring_r:.4f} mm, so this junction has "
            f"no corner for a fillet to round")
    j = int(cross[0])
    F = _circle_crossing(flank[j], flank[j + 1], ring_r)

    d = flank - mid
    centres = flank + R * (d / np.linalg.norm(d, axis=1)[:, None])
    target = ring_r + void_sign * R
    resid = np.hypot(centres[:, 0], centres[:, 1]) - target
    # Bracketed from the ring crossing INTO the spoke, which is the same bracket
    # `_fillet_tangency` scans (`s_end` is that crossing's station).  Starting at the end
    # of the band instead would admit a root on the stub that lies inside the ring.
    tail = resid[j:]
    hit = np.nonzero(np.sign(tail[:-1]) * np.sign(tail[1:]) <= 0.0)[0]
    if hit.size == 0:
        raise ValueError(
            f"no fillet of radius {R:.4f} mm is tangent to both the ring at "
            f"r={ring_r:.4f} and the flank anywhere along the spoke: the tangency "
            f"residual stays {tail.min():+.4f}..{tail.max():+.4f} mm.  The fillet is "
            f"larger than the notch can hold.")
    i = j + int(hit[0])

    def at(t):
        p = flank[i] + t * (flank[i + 1] - flank[i])
        e = p - (mid[i] + t * (mid[i + 1] - mid[i]))
        return p, p + R * e / np.linalg.norm(e)

    lo, hi, sign_lo = 0.0, 1.0, np.sign(resid[i])
    for _ in range(60):
        t = 0.5 * (lo + hi)
        if np.sign(np.hypot(*at(t)[1]) - target) == sign_lo:
            lo = t
        else:
            hi = t
    A, C = at(0.5 * (lo + hi))
    B = C * (ring_r / np.hypot(C[0], C[1]))

    # Green's theorem, one term per boundary piece and both arcs EXACT -- the same
    # integration `_clip_polygon_to_disk` does, for a loop it cannot be handed because
    # two of its three sides are circles of different centres.
    loop = np.concatenate([[A], flank[i:j:-1], [F]])
    acc = float(np.sum(loop[:-1, 0] * loop[1:, 1] - loop[1:, 0] * loop[:-1, 1]))
    dth = math.atan2(B[1], B[0]) - math.atan2(F[1], F[0])
    acc += ring_r ** 2 * ((dth + math.pi) % (2.0 * math.pi) - math.pi)
    a0 = math.atan2(B[1] - C[1], B[0] - C[0])
    a1 = math.atan2(A[1] - C[1], A[0] - C[0])
    dphi = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
    R_arc = 0.5 * (np.linalg.norm(A - C) + np.linalg.norm(B - C))
    acc += R_arc ** 2 * dphi + C[0] * (A[1] - B[1]) - C[1] * (A[0] - B[0])
    return abs(0.5 * acc), i


def _fillet_reference_areas(flanks, mid, hub_radius, rim_inner, radii):
    """Both junctions' fillet areas for ONE spoke, on the reference's own polygon.

    `flanks` is `(top, bottom)`, each indexed by CURVE index -- so the caller has to
    un-reverse `thicken_3taper_curve`'s second half and hand them over before the winding
    normalisation, which reverses the whole loop and swaps them.

    WHICH FLANK STRADDLES IS READ OFF THE RADII, not off `flank_orientation`, for the
    reason `_uncap_reference_poly` reads its far flank the same way: the two ends are
    independent, and deciding it here from the exporter's own points keeps this path
    from inheriting the mesh's decision.  `flank_orientation` answers the identical
    question -- the flank that starts inside the hub circle, the flank that ends outside
    the rim's -- and the two agree by construction, not by being the same call.
    """
    top, bot = flanks
    ends = (("hub", radii[0], hub_radius, 1.0,
             float(np.hypot(*top[0])) < float(np.hypot(*bot[0])), False),
            ("rim", radii[1], rim_inner, -1.0,
             float(np.hypot(*top[-1])) > float(np.hypot(*bot[-1])), True))
    out, reach = {}, {}
    for junction, R, ring_r, void_sign, straddling_is_top, from_rim in ends:
        if not R > 0.0:
            raise ValueError(
                f"the filleted region needs a positive radius at both junctions; "
                f"{junction} got {float(R):.4f}.  Use `fillet=None` for the unfilleted "
                f"region.")
        flank = top if straddling_is_top else bot
        flank, line = (flank[::-1], mid[::-1]) if from_rim else (flank, mid)
        area, i = _fillet_reference_wedge(flank, line, ring_r, void_sign, float(R))
        out[junction] = area
        reach[junction] = (len(mid) - 1 - i) if from_rim else i
    if reach["hub"] >= reach["rim"]:
        raise ValueError(
            f"the two fillets are longer than the spoke: their tangent points meet at "
            f"curve samples {reach['hub']} and {reach['rim']}.")
    return out["hub"], out["rim"]


# `modelled_area_reference` is pure and costs ~50 ms (a 20001-point centreline, a 40002-
# point offset band, and two exact disk clips).  `area_report` needs it TWICE whenever
# `uncap` is active -- once for the region the mesh builds and once for the capped region
# the STEP comparison is anchored on -- so the 2026-08-18 flip doubled it, 51 -> 101 ms.
# This cache takes that back for any repeat caller: 0.27 ms warm.
#
# WHAT IT IS NOT.  It is NOT why `make test` takes what it takes.  I added it believing the
# doubled `area_report` explained the suite's +3m07s against its pre-flip baseline; the
# measurement says otherwise -- 31m46s before the cache, 31m36s after, a 10 s difference,
# so `area_report` was never on the critical path.  Keeping it anyway because the 2x on a
# pure function is real for sweeps and drivers that call it in a loop, but the honest note
# is that this bought throughput, not suite time.  See UNCAP_PLAN.md Step 4.
#
# Bounded like `_COORD_FN_CACHE` and for the same reason: a sweep that walks a gene must
# not grow it without limit.
_AREA_REF_CACHE = {}
_AREA_REF_CACHE_MAX = 64


def modelled_area_reference(genes, rim_outer=RIM_OUTER_RADIUS_MM,
                            span_mm=HUB_RIM_SPAN_MM, hub_radius=HUB_RADIUS_MM,
                            n_spokes=NUMBER_OF_SPOKES, n_curve=20001,
                            uncap=UNCAP_DEFAULT, fillet=None):
    """The area of the region this module models, DERIVED rather than transcribed.

        hub disk + rim band + n_spokes * (spoke band clipped to the annulus)
                            + n_spokes * (two junction fillets, when `fillet` is given)

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
    module docstring.  `area_report` folds that measured allowance in and reports it as
    `reference_shipped_step_mm2`; this function never does.  (That sentence used to say
    "use `shipped=True`", which has not been a parameter of anything here.)

    `fillet=(R_hub, R_rim)` ADDS THE SECTOR BLOCKING'S TWO FILLETS PER SPOKE (PLAN.md
    §50), so that `error_vs_modelled` is a discretisation residual for a filleted mesh
    the same way it is for the default one.  Three things about that argument:

      IT TAKES RADII, NEVER `fillet=True`.  In `sector_blocks` and `build_wheel` that
      flag means "this genome's radii, MOVED BY `SECTOR_FIT_CLAMP` if they have no room",
      and resolving the clamp needs a config, an `uncap` and a layer profile — none of
      which a pure area reference has or should have.  Accepting `True` here would
      therefore make one spelling mean two regions, so it is refused; `area_report`
      passes `mesh.fillet_radii_mm`, which is what was BUILT rather than what was asked
      for.

      IT IS THE `"sector"` BLOCKING'S REGION.  `fillet_blocking="spoke"` is §47's retired
      construction and rounds the flank a different way; `area_report` withholds rather
      than comparing against this.

      IT IS NOT THE EXPORTER'S FILLET.  `wheel_step_export` rounds the SOLID's edges, and
      the module docstring's `reference_shipped_step_mm2` is anchored on the UNFILLETED
      cross-section for that reason.  This term describes the mesh's region and only
      that.

    The two fillets are added, never blended into `spokes_mm2`: at the hub the wedge sits
    outside the ring circle and at the rim inside it, so both fall in the annulus and
    would vanish into the band's own figure — and the mesh tags them `spoke` while part
    of `*_fillet_b` sits inside the ring, so per-REGION agreement is a different (and
    still open) question from the total.
    """
    import wheel_fea as _fea

    if fillet is True or fillet is False:
        raise ValueError(
            "`modelled_area_reference` takes fillet=(R_hub, R_rim) or None, not "
            f"{fillet!r}: `fillet=True` means the CLAMPED radii in `sector_blocks` and "
            "this function cannot resolve the clamp.  Pass `mesh.fillet_radii_mm`.")
    fillet = None if fillet is None else (float(fillet[0]), float(fillet[1]))
    g = np.asarray(genes, dtype=float)
    key = (g.tobytes(), float(rim_outer), float(span_mm), float(hub_radius),
           int(n_spokes), int(n_curve), repr(uncap), repr(fillet))
    hit = _AREA_REF_CACHE.get(key)
    if hit is not None:
        return dict(hit)
    curve, _ = _fea.generate_bezier_centerline(*g[:8], span_mm=span_mm,
                                               num_points=n_curve)
    poly = np.asarray(_fea.thicken_3taper_curve(curve, *g[8:12]), dtype=float)
    shift = np.array([hub_radius, 0.0])
    poly = poly + shift
    # The two flanks, each by CURVE index, taken BEFORE the winding normalisation below
    # reverses the loop and swaps them.  `_fillet_reference_areas` needs them this way
    # round and the clipper does not care.
    half = len(poly) // 2
    flanks = (poly[:half], poly[half:][::-1])
    # `thicken_3taper_curve` returns [top; reversed bottom], whose winding depends on
    # which way the offset normals point — so it is clockwise for some genomes and
    # counter-clockwise for others.  Normalise here rather than taking abs() at the end:
    # the arc corrections inside the clipper are signed too, and only a consistently
    # oriented loop makes them cancel correctly.
    if _polygon_area(poly) < 0.0:
        poly = poly[::-1]
    rim_inner = rim_inner_radius(span_mm, hub_radius)
    # The region has to be the one the MESH builds -- see `_uncap_reference_poly`.
    poly = _uncap_reference_poly(poly, curve, hub_radius, rim_inner, uncap)
    spoke = (_clip_polygon_to_disk(poly, rim_inner)
             - _clip_polygon_to_disk(poly, hub_radius))
    out = {
        "hub_disk_mm2": float(np.pi * hub_radius ** 2),
        "rim_band_mm2": float(np.pi * (rim_outer ** 2 - rim_inner ** 2)),
        "spoke_each_mm2": float(spoke),
        "spokes_mm2": float(n_spokes * spoke),
        "total_mm2": float(np.pi * hub_radius ** 2
                           + np.pi * (rim_outer ** 2 - rim_inner ** 2)
                           + n_spokes * spoke),
    }
    if fillet is not None:
        # The fillet does not touch `uncap`'s corner -- one rounds the STRADDLING flank,
        # the other continues the FAR one -- so `area_report`'s two calls differ by the
        # gusset and by nothing else, exactly as they did before this term existed.
        f_hub, f_rim = _fillet_reference_areas(flanks, curve + shift, hub_radius,
                                               rim_inner, fillet)
        out.update({
            "fillet_radii_mm": [fillet[0], fillet[1]],
            "fillet_hub_each_mm2": float(f_hub),
            "fillet_rim_each_mm2": float(f_rim),
            "fillet_each_mm2": float(f_hub + f_rim),
            "fillets_mm2": float(n_spokes * (f_hub + f_rim)),
        })
        out["total_mm2"] += out["fillets_mm2"]
    if len(_AREA_REF_CACHE) >= _AREA_REF_CACHE_MAX:
        _AREA_REF_CACHE.pop(next(iter(_AREA_REF_CACHE)))
    _AREA_REF_CACHE[key] = dict(out)
    return out


# `wheel_step_export._embed` adds straight segments that push each spoke further into its
# rings.  A deliberate modelling difference, not an error in either kernel, and the reason
# a mesh-vs-STEP comparison lands low.
#
# RE-MEASURED after the hub fillet milestone (see HUB_PLAN.md).  `_embed`'s inward step used
# to take the least rotation from the junction tangent, which ran 4.516 mm mostly sideways
# and buried the hub circle under the neighbouring spokes; it now plunges radially, 1.788 mm,
# and the gusset it leaves in the annulus is correspondingly smaller.  Measured as
# (STEP cross-section - this region's reference) / 12, both from the same genome in one
# process: 2644.3509 - 2607.9634 = 36.3875, i.e. 3.032 per spoke, against 4.356 the same way
# before the change.  The gap it explains narrows from ~1.93% to ~1.38%.
#
# THAT PAIR IS THE PRE-§13 GENOME'S AND THE CONSTANT IS STALE WITH IT.  Its DIFFERENCE is
# the invariant -- see `test_the_embed_difference_from_the_shipped_step_is_the_known_amount`
# -- but the two absolute numbers describe a wheel that was promoted away from on
# 2026-08-06.  The identical computation on the SHIPPED genome, from
# `export/wheel_step_manifest.json` and this module in one process (§87):
#
#     36145.8 mm3 / 22.4 mm = 1613.6518,  capped reference 1607.2718
#     1613.6518 - 1607.2718 = 6.3800, i.e. 0.5317 per spoke
#
# 5.7x smaller, and `reference_shipped_step_mm2` is high by the difference -- 29.6 mm2,
# 1.8% of the wheel.  DELIBERATELY NOT REPLACED.  §14's open item 6 is explicit: "Do not
# guess a new number.  Replacing 3.03 with 0.98 would only re-stale it on the next genome;
# what is needed is the scaling law, derived from `wheel_step_export._embed` the way
# `wheel_geometry.junction_bite` was derived."  0.5317 is a third measurement of a quantity
# that has now read 4.356, 3.032, 0.98 and 0.5317 on four genomes, which is the evidence FOR
# that item rather than a candidate to close it with.
EMBED_ALLOWANCE_PER_SPOKE_MM2 = 3.03


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

    A FILLETED MESH GETS THE FIRST AND NOT THE SECOND (PLAN.md §86).
    `reference_modelled_mm2` follows the fillet, because it is the region this mesh
    builds and `error_vs_modelled` means nothing unless both sides describe it; the
    fillets' own share is reported as `fillet_modelled_mm2` so the two halves can be read
    apart.  The STEP half is WITHHELD instead, and not out of caution: both numbers
    behind it — the 2644.3509 mm2 profile and `EMBED_ALLOWANCE_PER_SPOKE_MM2` — were
    measured against the UNFILLETED cross-section (see the module docstring), so
    `reference_shipped_step_mm2` describes an unfilleted region and there is no
    like-for-like comparison to report.  The exporter's fillet is a different
    construction from this one and pricing one against the other is its own work.
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
    fillet = getattr(mesh, "fillet_radii_mm", None)
    if getattr(mesh, "fillet", None) is not None and fillet is None:
        # THE OTHER FILLETED CONSTRUCTION, and this one still has no reference.
        # `fillet_blocking="spoke"` is §47's retired blocking -- the arc on the spoke
        # block's own flank edge -- which rounds the flank a different way and leaves no
        # `_applied` record to read the built radii out of.  `make fillet` still measures
        # it and it must not be compared against the SECTOR blocking's region, so the
        # withholding this branch used to do for every filleted mesh survives here.
        out["reference_unavailable_because"] = (
            "this mesh is filleted by the `spoke` blocking and "
            "`modelled_area_reference` models the `sector` blocking's region; see "
            "PLAN.md §86")
        return out
    uncap = getattr(mesh, "uncap", UNCAP_DEFAULT)
    kw = dict(rim_outer=mesh.rim_outer, span_mm=mesh.span_mm, n_spokes=mesh.n_spokes)
    ref = modelled_area_reference(mesh.genes, uncap=uncap, fillet=fillet, **kw)
    # TWO references, and only one of them may follow `uncap`.
    #
    # `reference_modelled_mm2` MUST follow it: it is the region this mesh builds, and
    # `error_vs_modelled` is a discretisation residual only if both sides describe the
    # same region.  `reference_shipped_step_mm2` MUST NOT: the shipped solid is a fixed
    # physical thing, and letting it drift with our meshing choice would hide exactly the
    # change this pair of numbers exists to expose.  So the STEP reference stays anchored
    # on the CAPPED region plus the full measured allowance, and the part of `_embed`'s
    # gusset the mesh has started to model is reported separately rather than netted off.
    #
    # THE FILLET IS THE CASE THAT RULE DOES NOT COVER, so the STEP half is withheld
    # instead of anchored -- see the docstring.  It rides on BOTH calls below, which is
    # what keeps `gusset_modelled_mm2` exactly the difference it always was: the fillet
    # rounds the STRADDLING flank and `uncap` continues the FAR one, so the term cancels.
    capped = (_uncap_blend(uncap, True) is None and _uncap_blend(uncap, False) is None)
    ref0 = ref if capped else modelled_area_reference(mesh.genes, uncap=False,
                                                      fillet=fillet, **kw)
    gusset = ref["total_mm2"] - ref0["total_mm2"]
    out.update({
        "reference_breakdown_mm2": ref,
        "reference_modelled_mm2": ref["total_mm2"],
        "reference_capped_mm2": ref0["total_mm2"],
        "gusset_modelled_mm2": float(gusset),
        "gusset_modelled_per_spoke_mm2": float(gusset / mesh.n_spokes),
    })
    if fillet is None:
        ref_shipped = ref0["total_mm2"] + mesh.n_spokes * EMBED_ALLOWANCE_PER_SPOKE_MM2
        out["reference_shipped_step_mm2"] = ref_shipped
    out["error_vs_modelled"] = total / ref["total_mm2"] - 1.0
    if fillet is None:
        out["error_vs_shipped_step"] = total / ref_shipped - 1.0
    else:
        out["fillet_modelled_mm2"] = float(ref["fillets_mm2"])
        out["fillet_modelled_per_spoke_mm2"] = float(ref["fillets_mm2"] / mesh.n_spokes)
        out["step_reference_unavailable_because"] = (
            "`reference_shipped_step_mm2` is anchored on the UNFILLETED cross-section "
            "and `EMBED_ALLOWANCE_PER_SPOKE_MM2` was measured against it, so neither "
            "describes a filleted region; see PLAN.md §86")
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
    for name in dict.fromkeys(mesh.element_block.tolist()):
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
