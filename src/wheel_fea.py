"""
=============================================================================
  COMPLIANT PLA UAV WHEEL — Co-Optimization v2.1
  Senior ME Refactor: Expanded Genome, Rigorous Physics, Enhanced Output
=============================================================================

CHAIN-OF-THOUGHT ENGINEERING NOTES
------------------------------------
0. WHAT CHANGED IN v2.1  (three defects that made v2.0's numbers untrustworthy)
   a. The deflection model was a CANTILEVER — hub clamped, rim end free.  A real
      spoke is fused into a stiff rim ring, which restrains tip rotation and
      tangential motion, so v2.0 over-predicted compliance by ~4× (straight-beam
      check: FL³/3EI vs FL³/12EI).  Replaced by a force-method (Castigliano)
      solve carrying two redundants at the tip.  See BOUNDARY_CONDITION.
   b. The Euler buckling check was DEAD CODE, twice over.  (i) `F_axial = F·cosθ`
      is positive for every monotone-x curve, so `clip(-F_axial, 0, None)` was
      identically zero; the sign convention for compression was inverted.
      (ii) `Pe = π²EI/(Ke·L)²` used L = one 0.06 mm discretization segment, and
      the 1/L² made Pe astronomical.  Euler buckling is a global mode over an
      unbraced length, not a per-element property.  v2.0 runs report
      buckling_ratio = 0.000 for both reasons.  Now: compression-negative axial
      force + an equivalent-column check over the whole spoke.
   c. Mass was INERT as an objective (mass/40 ≈ <1 loss unit against a deflection
      term scaled at 2500).  Now normalised and weighted — see MASS_WEIGHT.
   Also: the sector plot's fillet arcs were drawn at an arbitrary diagonal offset
   rather than tangent to the real geometry, so the figure disagreed with the STEP
   that wheel_step_export.py builds.  Now constructed properly.

1. GENOME (14 genes)
   - Bezier centerline upgraded from degree-3 (4 pts) to degree-5 (6 pts),
     adding two interior control points [P1..P4] that allow S-curves and
     progressive bend profiles.  P0 (hub) and P5 (rim) remain locked.  Interior
     control-point y-range is ±32 mm (v2.0 used ±25, itself down from ±55).  The
     widening is a direct consequence of 0a: compliance goes as ∫F·y²/(EI) ds, so
     recovering the ~4× stiffness of the honest boundary condition needs roughly
     2× the lateral deviation, and the v2.0 winner already sat at cy ≈ 15–17 mm.
   - Thickness parameterized as 4 nodes (t0..t3) across 3 linear-taper zones
     at normalized arc-length breakpoints s=[0, 0.33, 0.67, 1.0].  This lets
     the GA bulk the root/tip independently and create a waisted mid-section.
     Lower bound = MIN_WALL_MM (printable floor) so it can genuinely thin down.
   - Two fillet radii genes: R_hub and R_rim (now truly evolvable, not derived
     from thickness).  Used in the stress-concentration factor (Kt), the sector
     plot, and the CAD output.
   - A smooth-spiral regularizer penalizes curvature reversals (inflections) and
     turn-rate wiggle so the evolved centerline is a clean single-curvature
     spiral rather than a lumpy path that merely hits the deflection target.

2. PHYSICS UPGRADES
   a. Stress-concentration factors (Kt) at the hub and rim fillets are
      computed from the Peterson (1974) empirical formula for a stepped beam
      in bending:  Kt ≈ 1 + C*(t/2R)^0.65  (simplified; C≈1.0 for PLA-FFF).
      The peak bending stress at each end is multiplied by its local Kt.
   b. Curved-beam correction: Winkler-Bach theory adds a curvature factor
      (e/c) to the bending stress for each segment whose radius of curvature
      is within 4× the section height, preventing under-prediction of inner-
      fibre stress.
   c. Euler column buckling as an EQUIVALENT-COLUMN check over the whole unbraced
      spoke: Pe = π²·EI_eff / (Ke·L_total)², where EI_eff is the compliance-
      weighted (harmonic) mean rigidity so the 3-zone taper is respected, and Ke
      is keyed to the boundary condition (KE_BY_BC).  A penalty fires when the
      peak compressive axial load exceeds 0.6·Pe.  Caveat: I = w·t³/12 is
      correctly the weak axis (out-of-plane is t·w³/12, far stiffer at w=22.4mm),
      but for an in-plane-curved member Euler is a proxy — the genuinely competing
      modes are snap-through and lateral-torsional buckling.  Sanity guard, not a
      certification.
   d. FFF anisotropy knockdown: layer-adhesion strength is ~80 % of bulk
      PLA, so the allowable stress is reduced by 0.80 before applying the
      safety factor.

3. GA TUNING FOR 14-GENE SPACE
   - Population 300 (≥20× gene count for diversity in a 14-D landscape).
   - Tournament selection k=5 to maintain selection pressure without
     premature convergence.
   - Uniform crossover + adaptive Gaussian mutation (σ starts at 0.25 of
     gene range, halves every 250 generations via on_generation callback).
     The mutation actually reads that σ now (adaptive_gaussian_mutation).
   - Elitism = 5 to preserve top solutions across mutation sweeps.
   - Smooth penalty functions (quadratic barrier) replace hard -500 cliffs
     so gradient-free search can still descend toward feasibility.

4. SPATIAL SELF-INTERSECTION (hub crowding)
   Adjacent spoke roots must fit within the chord the hub circle allots each
   sector: available = 2·R_hub·sin(pi/N).  The root needs t0 + 2·R_hub_fillet +
   clearance of in-plane width; a quadratic soft penalty fires on any excess.
   NOTE: the face width (z-height) is out-of-plane and is intentionally NOT part
   of this in-plane check — using it (the old code) produced a ~1.5e6 near-
   constant penalty that swamped the entire fitness landscape.
"""

import warnings
import hashlib
import json
import math
import os

import project_paths as PP   # stdlib-only; safe in the jax-free CAD env
import numpy as np

# The shared geometry kernel.  numpy-only at import (it takes its array module as an
# argument rather than importing jax), which is what keeps this file importable from
# the CadQuery environment — see wheel_step_export.py:60-69.
import wheel_geometry as _geom

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(42)

# ---------------------------------------------------------------------------
# PHYSICAL & GEOMETRIC CONSTANTS
# ---------------------------------------------------------------------------
NUMBER_OF_SPOKES    = 12
HUB_RADIUS_MM       = 25.4 / 2          # 12.7 mm

# Where the spokes merge into the rim band.  THIS IS THE RIM-THICKNESS DECISION, because
# the outer diameter is fixed at Ø100 by requirement: the band is
# RIM_OUTER_RADIUS_MM - RIM_RADIUS_MM, so thickening it means moving THIS radius inward,
# which also shortens the spoke.
#
# Was 97.8/2 = 48.9, giving a 1.1 mm band.  M4's full-wheel FEA measured what that costs:
#
#     band    axle drop   rim share of compliance
#     1.1 mm    2.85 mm         44.7%      <- as shipped, +43% over the 2.0 mm target
#     1.3 mm    2.29 mm         39.8%
#     1.5 mm    1.94 mm         35.5%      <- here
#     1.8 mm    1.59 mm         30.8%
#
# A 1.1 mm band held nearly as much compliance as all twelve spokes combined, and the
# beam model the GA optimises against cannot see it at all — so the GA was solving the
# wrong problem, not solving it badly.  1.5 mm is chosen because it brackets the 2.0 mm
# target with the rim share down to a third, and because at that thickness the beam
# model happens to become close to unbiased at the design point (measured FEA/beam 1.00
# there versus 1.43 at 1.1 mm), which is what makes a re-run of the existing GA
# meaningful rather than merely different.
#
# Everything derives from this: the span below, the mesh's ring radius
# (`wheel_wheel.rim_inner_radius`), and the exporter's rim face.  Changing it
# REINTERPRETS every gene on disk — `tests/test_golden.py` fails loudly if the artifact
# was optimised in a different frame.
RIM_RADIUS_MM       = 48.5
HUB_RIM_SPAN_MM     = RIM_RADIUS_MM - HUB_RADIUS_MM   # 35.8 mm
SPOKE_WIDTH_MM      = 22.4              # fixed face width (z-height)

# Material — PLA, FFF anisotropy-corrected
YOUNGS_MODULUS_PLA_MPA  = 2300.0
FFF_KNOCKDOWN           = 0.80          # layer-adhesion reduction
ULTIMATE_STRESS_MPA     = 50.0 * FFF_KNOCKDOWN   # 40 MPa effective
SAFETY_FACTOR           = 1.6
ALLOWABLE_STRESS_MPA    = ULTIMATE_STRESS_MPA / SAFETY_FACTOR   # 25 MPa
DENSITY_PLA             = 1.24e-3       # g/mm³

# Loading
FORCE_LBS                = 15.0
TOTAL_FORCE_NEWTONS      = FORCE_LBS * 4.44822     # 66.72 N
# Conservative: 1/3 of spokes are load-bearing at any stance
FORCE_PER_SPOKE_NEWTONS  = TOTAL_FORCE_NEWTONS / (NUMBER_OF_SPOKES / 3.0)

TARGET_DEFLECTION_MM     = 2.0     # compliant target (raised from 1.0 for more travel)

# Mass objective scaling.  The old `mass/40` contributed <1 loss unit against a
# deflection term scaled at 2500, so mass was a tiebreaker rather than an objective.
# Calibration: a 5 % deflection error costs 2500*0.05² = 6.25, so at WEIGHT=30 about
# 7.6 g of mass now trades against 5 % of the deflection target.  Tune against the
# loss-term table that `evaluate_design` reports for the winner.
MASS_REFERENCE_G         = 36.5    # v2.0 cantilever-era best, as a normaliser
MASS_WEIGHT              = 30.0

# ---------------------------------------------------------------------------
# BOUNDARY CONDITION
# ---------------------------------------------------------------------------
#  "fixed_guided" — hub clamped; the rim end is clamped against rotation and
#                   tangential motion but free to translate radially.  Models a spoke
#                   fused into a stiff rim ring, which is what the part actually is.
#  "cantilever"   — legacy v2.0 model: hub clamped, rim end entirely free.  Retained
#                   for regression against `best_solution_cantilever.json`.  It
#                   over-predicts compliance by ~4× (straight beam: FL³/3EI vs
#                   FL³/12EI), so it is not a defensible design model.
BOUNDARY_CONDITION       = "fixed_guided"

# Euler effective-length factor, keyed to the boundary condition rather than fixed at
# the old hard-coded 0.7 (which was the fixed-pinned figure and no longer matches).
KE_BY_BC = {"cantilever": 2.0, "fixed_guided": 0.5}
KE_BUCKLING              = KE_BY_BC[BOUNDARY_CONDITION]

# Piecewise taper breakpoints in normalized arc-length
# Re-exported from the geometry kernel rather than redefined.  Two copies of the zone
# breakpoints would be two things to keep in step, and a silent half-zone shift between
# the thickness the optimizer prices and the thickness the mesh builds is exactly the
# kind of disagreement this refactor exists to make impossible.
TAPER_BREAKPOINTS = _geom.TAPER_BREAKPOINTS

# Mesh-derived design constraints, from the geometry kernel so there is one copy.
MIN_FOLD_MARGIN_MM = _geom.MIN_FOLD_MARGIN_MM
MAX_ARRIVAL_DEG = _geom.MAX_ARRIVAL_DEG

# Number of discretization points along Bezier
N_CURVE_PTS = 600

# ---------------------------------------------------------------------------
# GENOME LAYOUT  (14 genes total)
# ---------------------------------------------------------------------------
# Indices: CP = Control point
#  0  cx1   — Bezier interior CP1 x
#  1  cy1   — Bezier interior CP1 y
#  2  cx2   — Bezier interior CP2 x
#  3  cy2   — Bezier interior CP2 y
#  4  cx3   — Bezier interior CP3 x
#  5  cy3   — Bezier interior CP3 y
#  6  cx4   — Bezier interior CP4 x
#  7  cy4   — Bezier interior CP4 y
#  8  t0    — thickness at hub root (s=0)
#  9  t1    — thickness at first junction (s=0.33)
# 10  t2    — thickness at second junction (s=0.67)
# 11  t3    — thickness at rim tip (s=1)
# 12  R_hub — hub-root transition fillet radius (mm)   [optimized, was derived]
# 13  R_rim — rim-tip  transition fillet radius (mm)   [optimized, was derived]

S = HUB_RIM_SPAN_MM

# Minimum printable wall (≈5 perimeters @ 0.4 mm nozzle)
MIN_WALL_MM = 2.0
# Lateral clearance between a spoke root and its hub sector (mm)
HUB_CLEARANCE_MM = 0.8

# Lateral half-range for the interior Bezier control points.
#
# DO NOT WIDEN THIS.  It looks like a binding constraint — cy1 and cy3 both sit at
# exactly +32.0 in the v2.1 winner — and the obvious reading is that the centerline
# wanted more room than the box allowed.  Measured, that reading is wrong.  Sweeping the
# bound over {32, 45, 60} at 4 seeds each, 300×300 as usual:
#
#     bound 32:  mean loss 50.30  (sd 0.44)   best 49.78
#     bound 45:  mean loss 52.91  (sd 1.65)   best 51.08
#     bound 60:  mean loss 55.16  (sd 1.64)   best 53.73
#
# Monotonically worse, far outside the seed noise.  And the tell: at bounds 45 and 60 no
# cy gene is pinned at all — the optimizer settles INTERIOR at |cy| ≈ 30–41 and is still
# worse.  So the extra room is reachable and simply not wanted.  The pinning at 32 is a
# boundary artifact of a shallow, y-symmetric, multi-modal landscape, and enlarging the
# box just costs search efficiency at a fixed budget (mean loss over the population
# doubled, 463 → 1076).
#
# Overridable via --cy-bound so this stays a measurable claim rather than a belief.
CY_BOUND_MM = 32.0

GENE_SPACE = [
    # CP1 — near hub (x must stay in [5%, 30%] of span)
    {"low": S * 0.05,  "high": S * 0.30},   # cx1
    {"low": -CY_BOUND_MM, "high": CY_BOUND_MM},   # cy1
    # CP2 — first third to midspan
    {"low": S * 0.22,  "high": S * 0.55},   # cx2
    {"low": -CY_BOUND_MM, "high": CY_BOUND_MM},   # cy2
    # CP3 — midspan to second third
    {"low": S * 0.45,  "high": S * 0.78},   # cx3
    {"low": -CY_BOUND_MM, "high": CY_BOUND_MM},   # cy3
    # CP4 — near rim (x must stay in [70%, 95%] of span)
    {"low": S * 0.70,  "high": S * 0.95},   # cx4
    {"low": -CY_BOUND_MM, "high": CY_BOUND_MM},   # cy4
    # Piecewise thickness nodes (mm) — 4 nodes, 3 taper zones
    # Lower bound = MIN_WALL_MM so the GA can thin down to the printable floor.
    {"low": MIN_WALL_MM,  "high": 10.0},     # t0 (root, hub side)
    {"low": MIN_WALL_MM,  "high":  8.0},     # t1 (junction 1)
    {"low": MIN_WALL_MM,  "high":  8.0},     # t2 (junction 2)
    {"low": MIN_WALL_MM,  "high":  6.0},     # t3 (tip, rim side)
    # Transition fillets (mm) — now first-class evolvable genes
    {"low": 0.5,  "high": 4.0},              # R_hub
    {"low": 0.5,  "high": 3.0},              # R_rim
]

# ---------------------------------------------------------------------------
# HISTORY BUFFERS
# ---------------------------------------------------------------------------
best_fitness_history = []
mean_fitness_history = []
mutation_sigma_global = [0.25]   # mutable via on_generation

# ---------------------------------------------------------------------------
# DEGREE-5 BEZIER CENTERLINE
# ---------------------------------------------------------------------------

def generate_bezier_centerline(cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4,
                                span_mm=HUB_RIM_SPAN_MM, num_points=N_CURVE_PTS):
    """
    Degree-5 Bezier with 6 control points.
    P0 = (0, 0)  [hub, locked]
    P1..P4       [interior, evolvable]
    P5 = (span, 0) [rim, locked]
    Returns (curve_points [N,2], control_polygon [6,2])

    Thin wrapper over `wheel_geometry.bezier_centerline`.  The body moved there so the
    JAX mesh generator and this module compute the same curve from the same code rather
    than from two implementations that agree until someone edits one of them.
    Bit-identical to the previous inline version — see tests/test_geometry_kernel.py.
    Also faster: the Bernstein basis is now cached instead of rebuilt per evaluation.
    """
    return _geom.bezier_centerline(cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4,
                                   span_mm=span_mm, num_points=num_points)

# ---------------------------------------------------------------------------
# PIECEWISE-LINEAR THICKNESS PROFILE
# ---------------------------------------------------------------------------

def thickness_at_arc_length(arc_fractions, t0, t1, t2, t3):
    """
    Returns thickness at each normalized arc-fraction in [0,1].
    Three zones: [0, 1/3], [1/3, 2/3], [2/3, 1].

    Delegates to the branch-free form in `wheel_geometry`, which is the same
    piecewise-linear function written as a sum of clipped ramps so it survives JAX
    tracing (the old boolean-mask assignment does not).  Agreement with the masked
    original is one ULP — 2.2e-16 relative — and is pinned by a test.
    """
    return _geom.thickness_at_arc_length(arc_fractions, t0, t1, t2, t3)

# ---------------------------------------------------------------------------
# STRESS-CONCENTRATION FACTOR (Peterson, 1974 — stepped beam in bending)
# ---------------------------------------------------------------------------

def stress_concentration_kt(fillet_radius_mm, thickness_mm, c_factor=1.0):
    """
    Empirical Kt for a curved fillet at a thickness step.
      Kt ≈ 1 + c_factor * (thickness / (2 * fillet_radius))^0.65
    Clamped to [1.0, 3.5] for physical realism.
    """
    if fillet_radius_mm < 0.1:
        return 3.5  # degenerate fillet → high concentration
    ratio = thickness_mm / (2.0 * fillet_radius_mm)
    kt = 1.0 + c_factor * (ratio ** 0.65)
    return float(np.clip(kt, 1.0, 3.5))

# ---------------------------------------------------------------------------
# WINKLER-BACH CURVED-BEAM CORRECTION
# ---------------------------------------------------------------------------

def curved_beam_factor(radius_of_curvature, section_height):
    """
    Returns curvature correction factor for inner-fibre bending stress.
    kb = 1 + (section_height / (4 * R))  (simplified Winkler-Bach).
    Only significant when R < 4h.

    Accepts scalars or arrays.  The array path matters: this used to be called from a
    599-iteration Python loop inside the fitness function, i.e. ~200M scalar calls per
    GA run, and it was the single hottest line in the program.
    """
    R = np.asarray(radius_of_curvature, dtype=float)
    h = np.asarray(section_height, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kb = 1.0 + h / (4.0 * R)
    kb = np.where(R <= 0, 2.5, kb)
    kb = np.clip(kb, 1.0, 2.5)
    return float(kb) if kb.ndim == 0 else kb

# ---------------------------------------------------------------------------
# CORE MECHANICAL ANALYSIS
# ---------------------------------------------------------------------------

def generalized_spoke_mechanics(curve_points, t0, t1, t2, t3,
                                 spoke_width, force_per_spoke,
                                 R_hub_fillet=0.5, R_rim_fillet=0.5,
                                 youngs_modulus=YOUNGS_MODULUS_PLA_MPA,
                                 boundary_condition=None,
                                 thickness_clip=(0.5, 20.0)):
    """
    Force-method (Castigliano) beam analysis of one curved spoke.

    BOUNDARY CONDITIONS
    -------------------
    Local frame: hub root at (0,0), rim tip at (S,0); the radial service load F acts
    at the tip along -x.  Two redundants are carried at the tip:
        M0 — end moment       (restrains tip rotation)
        H  — tangential force (restrains tip motion along +y)

      "fixed_guided" : both redundants solved from the compatibility conditions
                       dU/dM0 = 0 and dU/dH = 0.  Models a spoke fused into a stiff
                       rim ring — the physically representative case.
      "cantilever"   : M0 = H = 0.  Legacy free-tip model.  This is not a separate
                       code path but the same one with the redundants zeroed, which
                       is algebraically exact: it collapses to M = -F·y and
                       N = -F·cosθ, giving M·m_F = F·y² and N·n_F = F·cos²θ — term
                       for term the v2.0 integrands.  That equivalence is the
                       regression test.

    Internal actions at station s = (x, y), free body taken from s to the tip:
        M(s) = M0 + H·(S - x) - F·y
        N(s) = -F·cosθ + H·sinθ            (cosθ, sinθ) = (dx/ds, dy/ds)
    N is NEGATIVE IN COMPRESSION.  v2.0 used the opposite sign and its buckling
    check consequently never fired.

    Also included: piecewise-linear 3-taper thickness, Kt stress concentration at the
    hub/rim fillets, Winkler-Bach curved-beam correction, and an equivalent-column
    Euler buckling check over the whole unbraced spoke.

    Returns (deflection_mm, max_stress_mpa, total_mass_g, buckling_ratio).
    """
    if boundary_condition is None:
        boundary_condition = BOUNDARY_CONDITION
    n_seg = len(curve_points) - 1

    # --- Arc-length parameterization ---
    dx = np.diff(curve_points[:, 0])
    dy = np.diff(curve_points[:, 1])
    seg_lengths = np.sqrt(dx**2 + dy**2)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]
    arc_fracs_mid = (cumulative[:-1] + seg_lengths / 2.0) / (total_length + 1e-12)

    # --- Piecewise thickness at segment midpoints ---
    thicknesses = thickness_at_arc_length(arc_fracs_mid, t0, t1, t2, t3)
    # Guard against a GA proposing a degenerate section.  The default bounds are the
    # ones every committed artifact was produced under and must not change; the
    # parameter exists so the A4 slenderness sweep can scale the taper down by 8x
    # (t0 = 2.375 -> 0.297) without silently hitting the floor, which would flatten the
    # very convergence trend the sweep is measuring.
    thicknesses = np.clip(thicknesses, thickness_clip[0], thickness_clip[1])

    # --- Section properties per segment ---
    I  = spoke_width * thicknesses**3 / 12.0    # second moment of area
    A  = spoke_width * thicknesses               # cross-section area
    EI = youngs_modulus * I
    EA = youngs_modulus * A

    # --- Local geometry at segment midpoints ---
    safe_len  = np.where(seg_lengths < 1e-8, 1e-8, seg_lengths)
    cos_theta = dx / safe_len
    sin_theta = dy / safe_len
    x_mid     = curve_points[:-1, 0] + dx / 2.0
    y_mid     = curve_points[:-1, 1] + dy / 2.0

    # --- Unit-load fields: moment / axial force per unit of each unknown ---
    #     M = M0·m_M0 + H·m_H + F·m_F ,   N = M0·n_M0 + H·n_H + F·n_F
    m_M0 = np.ones(n_seg)
    m_H  = HUB_RIM_SPAN_MM - x_mid
    m_F  = -y_mid
    n_M0 = np.zeros(n_seg)
    n_H  = sin_theta
    n_F  = -cos_theta

    F = force_per_spoke

    if boundary_condition == "cantilever":
        M0 = 0.0
        H  = 0.0
    else:
        # Flexibility coefficients  a_ij = ∫ mi·mj/EI ds + ∫ ni·nj/EA ds
        # and load terms           b_i  = ∫ mi·m_F/EI ds + ∫ ni·n_F/EA ds
        w_b = seg_lengths / EI          # bending weight per segment
        w_a = seg_lengths / EA          # axial weight per segment
        a11 = np.sum(m_M0 * m_M0 * w_b) + np.sum(n_M0 * n_M0 * w_a)
        a12 = np.sum(m_M0 * m_H  * w_b) + np.sum(n_M0 * n_H  * w_a)
        a22 = np.sum(m_H  * m_H  * w_b) + np.sum(n_H  * n_H  * w_a)
        b1  = np.sum(m_M0 * m_F  * w_b) + np.sum(n_M0 * n_F  * w_a)
        b2  = np.sum(m_H  * m_F  * w_b) + np.sum(n_H  * n_F  * w_a)

        Amat = np.array([[a11, a12], [a12, a22]], dtype=float)
        # Tiny relative ridge: a near-straight, very thin spoke makes this system
        # badly conditioned, and a GA will happily propose exactly that.
        Amat[0, 0] += 1e-12 * (1.0 + abs(a11))
        Amat[1, 1] += 1e-12 * (1.0 + abs(a22))
        rhs = -F * np.array([b1, b2], dtype=float)
        try:
            M0, H = np.linalg.solve(Amat, rhs)
        except np.linalg.LinAlgError:
            M0, H = np.linalg.lstsq(Amat, rhs, rcond=None)[0]

    # --- Internal actions ---
    M = M0 * m_M0 + H * m_H + F * m_F
    N = M0 * n_M0 + H * n_H + F * n_F

    # --- Castigliano deflection in the load direction.  By the envelope theorem the
    #     redundants sit at stationary values of U, so differentiating w.r.t. F while
    #     holding them fixed is exact — no chain terms needed.
    defl_bending = np.sum(M * m_F / EI * seg_lengths)
    defl_axial   = np.sum(N * n_F / EA * seg_lengths)
    total_deflection = abs(defl_bending + defl_axial)

    # --- Radius of curvature per segment (finite-difference of tangent angle) ---
    tangent_angles = np.arctan2(dy, dx)
    d_theta = np.diff(tangent_angles)
    seg_len_mid = (seg_lengths[:-1] + seg_lengths[1:]) / 2.0  # n_seg-1 interior
    with np.errstate(divide="ignore", invalid="ignore"):
        R_curv_interior = np.where(np.abs(d_theta) < 1e-8, 1e6,
                                   seg_len_mid / np.abs(d_theta))
    # Segment i>0 takes the turn at node i; segment 0 borrows segment 1's.  One
    # prepended element is all that is needed to reach n_seg — v2.0 padded both ends
    # and produced n_seg+1, leaving a trailing element that was never read.
    R_curv = np.concatenate([[R_curv_interior[0]], R_curv_interior])

    # --- Curved-beam correction factor per segment (vectorized) ---
    kb = curved_beam_factor(R_curv, thicknesses)

    # --- Bending stress (outer fibre), with curved-beam correction ---
    sigma_bending = np.abs(M) * (thicknesses / 2.0) / I * kb

    # --- Axial stress ---
    sigma_axial = N / A

    # --- Kt at hub (segment 0) and rim (segment -1) ---
    Kt_hub = stress_concentration_kt(R_hub_fillet, thicknesses[0])
    Kt_rim = stress_concentration_kt(R_rim_fillet, thicknesses[-1])

    # Superposing |bending| + |axial| assumes worst-case sign alignment at every
    # station — deliberately conservative for a part we intend to print and load.
    sigma_combined = sigma_bending + np.abs(sigma_axial)
    sigma_combined[0]  *= Kt_hub
    sigma_combined[-1] *= Kt_rim

    max_stress = float(np.max(sigma_combined))

    # --- Euler buckling: equivalent-column check over the WHOLE unbraced spoke ---
    # EI_eff is the compliance-weighted (harmonic) mean rigidity, which is the standard
    # equivalent-column treatment for a stepped/tapered member and so respects the
    # 3-zone taper.  v2.0 applied Pe per discretization segment (L ≈ 0.06 mm), where
    # the 1/L² made Pe astronomical and the check could never fire.
    EI_eff = total_length / (np.sum(seg_lengths / EI) + 1e-30)
    Ke     = KE_BY_BC[boundary_condition]
    Pe     = np.pi**2 * EI_eff / ((Ke * total_length)**2 + 1e-12)
    F_comp = np.clip(-N, 0.0, None)        # compression is negative in N
    buckling_ratio = float(np.max(F_comp) / (0.6 * Pe + 1e-12))

    # --- Mass ---
    total_mass_g = (np.sum(thicknesses * spoke_width * seg_lengths)
                    * DENSITY_PLA * NUMBER_OF_SPOKES)

    return total_deflection, max_stress, total_mass_g, buckling_ratio

# ---------------------------------------------------------------------------
# MONOTONE-X VALIDATION
# ---------------------------------------------------------------------------

def curve_is_monotone_x(curve_points, tol=0.01):
    """Returns True if x-coordinates are strictly increasing (spoke doesn't fold back)."""
    return np.all(np.diff(curve_points[:, 0]) > tol)

# ---------------------------------------------------------------------------
# SMOOTH PENALTY HELPER
# ---------------------------------------------------------------------------

def soft_barrier(violation, scale=1.0):
    """
    Smooth quadratic barrier: 0 when violation<=0, rises as violation².
    Avoids the flat -500 cliff that stalls gradient-free search.
    """
    v = max(0.0, violation)
    return scale * v**2

# ---------------------------------------------------------------------------
# SPATIAL SELF-INTERSECTION (adjacent spokes near hub)
# ---------------------------------------------------------------------------

def hub_available_lateral_mm():
    """
    In-plane lateral room available to a single spoke root at the hub, measured as
    the chord between two adjacent spoke centerlines on the hub circle.
      available = 2 * R_hub * sin(pi / N)   (chord of the 360/N° sector)
    NOTE: face width (z-height) is out-of-plane and deliberately NOT used here.
    """
    return 2.0 * HUB_RADIUS_MM * np.sin(np.pi / NUMBER_OF_SPOKES)

def spoke_overlap_penalty(t0, R_hub):
    """
    In-plane hub-crowding check. The spoke root occupies its full thickness t0
    laterally (t0/2 either side of the centerline) plus the two fillet radii.
    Fires (quadratic) only when that exceeds the chord available per sector.
    """
    available = hub_available_lateral_mm()          # ≈ 6.57 mm at 12 spokes
    required  = t0 + 2.0 * R_hub + HUB_CLEARANCE_MM
    violation = required - available                # mm (O(1))
    return soft_barrier(violation, scale=500.0)

# ---------------------------------------------------------------------------
# SMOOTH-SPIRAL REGULARIZATION
# ---------------------------------------------------------------------------

def curve_smoothness_metrics(curve_points, n_sample=60, deadband_deg=1.0):
    """
    Quantify how 'spiral' vs 'lumpy' a centerline is.
    Returns (n_inflections, turn_rate_variation):
      - n_inflections     : number of curvature sign reversals (0 => single-curvature
                            spiral).  A deadband ignores numerical jitter near κ≈0.
      - turn_rate_variation: Σ|Δ(turn angle)| along the curve (radians); small for a
                            clean arc/spiral, large for a wiggly path.
    The curve is downsampled to n_sample points first so the metric is robust to the
    fine (600-pt) discretization noise.
    """
    idx = np.linspace(0, len(curve_points) - 1, n_sample).astype(int)
    p   = curve_points[idx]
    dx  = np.diff(p[:, 0]); dy = np.diff(p[:, 1])
    ang = np.arctan2(dy, dx)
    dtheta = np.diff(ang)                       # turn angle at each interior node

    deadband = np.deg2rad(deadband_deg)
    sig = dtheta[np.abs(dtheta) > deadband]
    if sig.size > 1:
        n_infl = int(np.count_nonzero(np.diff(np.sign(sig))))
    else:
        n_infl = 0

    turn_rate_variation = float(np.sum(np.abs(np.diff(dtheta))))
    return n_infl, turn_rate_variation

# ---------------------------------------------------------------------------
# PYGAD FITNESS FUNCTION
# ---------------------------------------------------------------------------

# Barrier weights for the two mesh-derived constraints.  Both are hard model-validity
# limits rather than preferences, so they are scaled to dominate the objective (~50
# units total) the moment they are violated, without being cliffs the GA cannot descend.
FOLD_PENALTY_SCALE    = 3000.0
ARRIVAL_PENALTY_SCALE = 3000.0


def fold_penalty(curve_points, ctrl_pts, t0, t1, t2, t3):
    """Barrier on the offset band turning inside out.

    `study_mesh_quality.py` measured that roughly 40% of the genomes the other
    constraints admit have a NEGATIVE-area region: where the centerline's radius of
    curvature drops below half the local thickness, the outward offset passes the centre
    of curvature and the flank folds through itself.  That is not a meshing artifact —
    the part described by such a genome does not exist, and the Castigliano model
    happily returns a deflection for it anyway.

    `smoothness` was only ever an indirect proxy for this (`:596-598` said so).  This is
    the direct statement, and it costs one closed-form curvature evaluation.  The
    threshold of 0.1 mm rather than 0 is measured: at 0 the sweep still admitted 9
    designs with inverted elements, at 0.1 none, and it still keeps 95% of the space.
    """
    margin = _geom.self_intersection_margin(curve_points, ctrl_pts, t0, t1, t2, t3,
                                            num_points=N_CURVE_PTS)
    return soft_barrier(_geom.MIN_FOLD_MARGIN_MM - margin, scale=FOLD_PENALTY_SCALE)


def arrival_penalty(ctrl_pts):
    """Barrier on the spoke meeting its ring too close to RADIALLY.

    The centerline endpoints are locked on the hub and rim circles, so the end
    cross-section — which is normal to the centerline — crosses its ring at
    (90 - arrival) degrees.  A near-tangent arrival therefore gives a healthy weld; a
    near-radial one lays the cross-section along the arc, and past about 80 degrees both
    flanks lie OUTSIDE the ring circle and the spoke touches it at a single point.

    That is a hinge, not a weld — and `BOUNDARY_CONDITION = "fixed_guided"` assumes a
    moment connection at both ends, so the model would be solving a structure the part
    does not have.  `study_wheel_mesh.py` measured the boundary at 70.6 degrees;
    `wheel_wheel.MAX_ARRIVAL_DEG = 65` is that with margin.
    """
    d = _geom.forward_difference_matrix(_geom.BEZIER_DEGREE, 1) @ ctrl_pts
    worst = 0.0
    # Bezier endpoint tangents are the first and last control-polygon edges exactly, and
    # the endpoints sit on the +x axis of their ring, so the radial direction is x-hat
    # and the arrival angle from the tangent is asin(|dx| / |d|).
    for edge in (d[0], d[-1]):
        n = math.hypot(float(edge[0]), float(edge[1]))
        if n > 0.0:
            worst = max(worst, math.degrees(math.asin(min(1.0, abs(edge[0]) / n))))
    return soft_barrier((worst - MAX_ARRIVAL_DEG) / 10.0, scale=ARRIVAL_PENALTY_SCALE)


def evaluate_design(solution):
    """
    Score one genome.  Returns (metrics, loss_terms) as plain dicts.

    The split exists so the GA can sum the terms while __main__ can print the
    breakdown: with seven weighted terms spanning four orders of magnitude, the only
    safe way to tune a weight is to look at what each one actually contributes.  The
    v2.0 mass term (mass/40) and the dead buckling term were both invisible precisely
    because nothing ever printed them.
    """
    (cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4,
     t0, t1, t2, t3, R_hub, R_rim) = solution

    # -------- Geometric monotonicity: soft penalty on x-ordering ----------
    xs = [0.0, cx1, cx2, cx3, cx4, S]
    x_order_penalty = 0.0
    for i in range(len(xs) - 1):
        violation = xs[i] + 2.0 - xs[i + 1]   # must be cx[i]+gap < cx[i+1]
        x_order_penalty += soft_barrier(violation, scale=80.0)

    curve_points, ctrl_pts = generate_bezier_centerline(
        cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4)

    # Check monotone x (soft penalty for fold-back)
    if not curve_is_monotone_x(curve_points, tol=0.005):
        fold_back = -np.min(np.diff(curve_points[:, 0]))
        x_order_penalty += soft_barrier(fold_back, scale=300.0)

    # -------- Fillet radii come straight from the genome now -----------
    #  (previously derived heuristically as R = 0.4·t; now GA-optimized)

    # -------- Physics --------------------------------------------------
    (deflection, max_stress, total_mass,
     buckling_ratio) = generalized_spoke_mechanics(
        curve_points, t0, t1, t2, t3,
        SPOKE_WIDTH_MM, FORCE_PER_SPOKE_NEWTONS,
        R_hub_fillet=R_hub, R_rim_fillet=R_rim)

    # -------- Deflection objective (want exactly TARGET) ---------------
    #  Two-sided on purpose: for a compliant mechanism the travel is the feature, so
    #  being too stiff is penalised exactly as hard as being too soft.
    defl_err = abs(deflection - TARGET_DEFLECTION_MM) / TARGET_DEFLECTION_MM

    # -------- Smooth-spiral regularization -----------------------------
    #  Penalize curvature reversals (want a single-curvature spiral) and general
    #  wiggle.  Without this the GA settles on lumpy centerlines that still hit the
    #  deflection target.  Secondary benefit: it suppresses tight local curvature,
    #  which is what keeps thicken_3taper_curve's naive normal offset from
    #  self-intersecting where R drops below t/2.
    n_infl, turn_var = curve_smoothness_metrics(curve_points)

    loss_terms = {
        "deflection":  2500.0 * (defl_err**2),
        # Normalised and weighted (v2.0: mass/40, which was <1 loss unit and so acted
        # only as a tiebreaker).  See MASS_WEIGHT for the calibration.
        "mass":        MASS_WEIGHT * (total_mass / MASS_REFERENCE_G),
        # Scale 4000: at this deflection target we lean hard on the material, so any
        # genome over the (already SF/FFF-derated) allowable is firmly rejected.
        "stress":      soft_barrier(max_stress / ALLOWABLE_STRESS_MPA - 1.0,
                                    scale=4000.0),
        "buckling":    soft_barrier(buckling_ratio - 1.0, scale=2000.0),
        "x_order":     x_order_penalty,
        "hub_overlap": spoke_overlap_penalty(t0, R_hub),
        "smoothness":  400.0 * n_infl + 120.0 * turn_var,
        # ---- the two constraints the meshing milestones measured -------------
        # Both are closed-form (one curvature evaluation, one endpoint tangent), so
        # they cost nothing next to the Castigliano solve, and both fix cases where the
        # MODEL ITSELF is invalid rather than merely where the design is poor.
        "fold":        fold_penalty(curve_points, ctrl_pts, t0, t1, t2, t3),
        "arrival":     arrival_penalty(ctrl_pts),
    }

    metrics = {
        "deflection_mm":      float(deflection),
        "max_stress_mpa":     float(max_stress),
        "total_mass_g":       float(total_mass),
        "buckling_ratio":     float(buckling_ratio),
        "deflection_error":   float(defl_err),
        "stress_utilization": float(max_stress / ALLOWABLE_STRESS_MPA),
        "n_inflections":      int(n_infl),
        "turn_variation":     float(turn_var),
    }
    return metrics, loss_terms


def pygad_fitness(ga_instance, solution, sol_idx):
    _, loss_terms = evaluate_design(solution)
    return -float(sum(loss_terms.values()))   # PyGAD maximises; lower loss → higher fitness


# ---------------------------------------------------------------------------
# GENERATION CALLBACK (logging + adaptive mutation)
# ---------------------------------------------------------------------------

# Logging / cooling cadence, overridden by __main__ so a --smoke run (8 generations)
# still prints progress instead of falling silent between the 100-generation marks.
LOG_EVERY    = [100]
SIGMA_HALVES = [250]


def on_generation(ga_instance):
    scores = ga_instance.last_generation_fitness
    best_fitness_history.append(-np.max(scores))
    mean_fitness_history.append(-np.mean(scores))

    # Halve mutation sigma periodically (adaptive cooling)
    gen = ga_instance.generations_completed
    if gen > 0 and gen % SIGMA_HALVES[0] == 0:
        mutation_sigma_global[0] *= 0.5
        print(f"  [Gen {gen:4d}] Mutation σ reduced to "
              f"{mutation_sigma_global[0]:.4f}")

    if gen % LOG_EVERY[0] == 0:
        best = -np.max(scores)
        print(f"  [Gen {gen:4d}] Best loss = {best:8.3f}  |  "
              f"Mean loss = {-np.mean(scores):8.3f}")

# ---------------------------------------------------------------------------
# ADAPTIVE GAUSSIAN MUTATION
# ---------------------------------------------------------------------------
# Per-gene ranges, precomputed once from GENE_SPACE for scaling the perturbation.
_GENE_LOW   = np.array([g["low"]  for g in GENE_SPACE])
_GENE_HIGH  = np.array([g["high"] for g in GENE_SPACE])
_GENE_RANGE = _GENE_HIGH - _GENE_LOW


def set_cy_bound(bound_mm):
    """Widen (or narrow) the lateral half-range of the four interior Bezier control
    points, keeping every derived array in step.

    These three module arrays are snapshots of GENE_SPACE taken at import.  Mutating
    GENE_SPACE alone would leave `adaptive_gaussian_mutation` clipping offspring to the
    OLD limits (:690) and `bound_saturation_report` measuring against them (:700), so
    the GA would silently keep the bound it was told to drop.
    """
    global CY_BOUND_MM, _GENE_LOW, _GENE_HIGH, _GENE_RANGE
    CY_BOUND_MM = float(bound_mm)
    for idx in (1, 3, 5, 7):                      # cy1, cy2, cy3, cy4
        GENE_SPACE[idx]["low"]  = -CY_BOUND_MM
        GENE_SPACE[idx]["high"] =  CY_BOUND_MM
    _GENE_LOW   = np.array([g["low"]  for g in GENE_SPACE])
    _GENE_HIGH  = np.array([g["high"] for g in GENE_SPACE])
    _GENE_RANGE = _GENE_HIGH - _GENE_LOW

def adaptive_gaussian_mutation(offspring, ga_instance):
    """
    Custom PyGAD mutation that actually honors the cooling schedule in
    `mutation_sigma_global`.  Each gene is perturbed with probability
    `mutation_probability` by N(0, (σ · gene_range)²), then clipped back into
    [low, high].  σ is halved every 250 generations by `on_generation`, so the
    search explores broadly early and fine-tunes late.
    """
    sigma = mutation_sigma_global[0]
    prob  = ga_instance.mutation_probability
    mask  = np.random.random(offspring.shape) < prob
    noise = np.random.normal(0.0, 1.0, offspring.shape) * (sigma * _GENE_RANGE)
    offspring = offspring + mask * noise
    return np.clip(offspring, _GENE_LOW, _GENE_HIGH)

# ---------------------------------------------------------------------------
# DIAGNOSTICS  (weight tuning and bound saturation)
# ---------------------------------------------------------------------------

# Single source of truth, in wheel_genome.  This ordering is the contract between the
# flat 14-vector `evaluate_design` unpacks positionally (:560) and the dict key set the
# STEP exporter reads, so it must exist exactly once.  Imported lazily inside this
# module's own namespace to keep the import graph flat — wheel_genome is numpy+stdlib,
# so this costs the CadQuery environment nothing.
from wheel_genome import GENE_NAMES  # noqa: E402


def bound_saturation_report(solution, tol_frac=0.01):
    """
    Genes sitting within `tol_frac` of their range from a bound.

    The np.clip in `adaptive_gaussian_mutation` causes boundary accumulation: any
    Gaussian mass landing outside the box piles up exactly on the limit, so a
    saturated gene means the search wanted to go further than the box allows.  That
    is sometimes the answer you want (t0/t1 pinned at MIN_WALL_MM says printability
    is the active constraint) and sometimes a sign the bound is too tight (cy pinned
    at ±32 would say the centerline needs more room to hit the deflection target).

    Returns a list of (gene_name, value, "low"|"high", bound_value).
    """
    hits = []
    for name, val, lo, hi in zip(GENE_NAMES, solution, _GENE_LOW, _GENE_HIGH):
        span = hi - lo
        if val - lo <= tol_frac * span:
            hits.append((name, float(val), "low", float(lo)))
        elif hi - val <= tol_frac * span:
            hits.append((name, float(val), "high", float(hi)))
    return hits


def dump_population(ga, path="stage2_elites.json", n_keep=16):
    """Write the final GA population's best distinct genomes for Stage 3 to start from.

    WHY THIS HAS TO EXIST.  The master plan says Stage 3 "multi-starts all 16 elites",
    and there are no 16 elites anywhere: `best_solution.json` holds ONE genome, pygad is
    constructed without `save_solutions=True`, and `keep_elitism` is 5 at this population
    size.  Without this, M8b's multi-start would have to perturb the single shipped
    genome — and every start would then sit in one basin, which tests far less than
    "monotone decrease from >= 3 of 4 multi-starts" is meant to test.

    Deduplicated by `genome_hash` because elitism carries identical rows forward between
    generations, so the raw population contains repeats and a naive top-16 can be four
    distinct designs wearing sixteen hats.

    Deliberately confined to `__main__`'s call site and gated behind a flag: M0's golden
    test pins `evaluate_design` and the shipped record, and a milestone that needed a
    starting set is not a reason to change what a GA run writes by default.
    """
    pop = np.asarray(ga.population, dtype=float)
    fit = np.asarray(ga.last_generation_fitness, dtype=float)
    order = np.argsort(-fit)                       # pygad maximises: best fitness first

    seen, kept = set(), []
    for i in order:
        h = _genome_hash_vector(pop[i])
        if h in seen:
            continue
        seen.add(h)
        kept.append((pop[i], float(fit[i]), h))
        if len(kept) >= n_keep:
            break

    record = {
        "source": "wheel_fea.py --dump-population",
        "seed": int(ga.random_seed) if getattr(ga, "random_seed", None) is not None
        else None,
        "generations": int(ga.generations_completed),
        "population_size": int(pop.shape[0]),
        "n_distinct_in_final_population": int(len({_genome_hash_vector(p)
                                                   for p in pop})),
        "elites": [
            {"rank": r, "genome_hash": h, "fitness": f, "loss": -f,
             "genes": {name: float(v) for name, v in zip(GENE_NAMES_ORDER, g)}}
            for r, (g, f, h) in enumerate(kept)],
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    with open(out, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\n  [STAGE-2 ELITES]  wrote {len(kept)} distinct genomes to {out}")
    print(f"  final population held {record['n_distinct_in_final_population']} distinct "
          f"designs out of {pop.shape[0]}")
    return out


GENE_NAMES_ORDER = ("cx1", "cy1", "cx2", "cy2", "cx3", "cy3", "cx4", "cy4",
                    "t0", "t1", "t2", "t3", "R_hub", "R_rim")


def _genome_hash_vector(vec):
    """`wheel_genome.genome_hash` on a raw vector, without importing that module.

    `wheel_fea` is the numpy-only half of the import graph and `tests/test_import_hygiene`
    enforces it; the hash is seven characters of sha256 over the same canonical text the
    other two copies use.
    """
    payload = ",".join(f"{float(v):.9g}" for v in vec)
    return hashlib.sha256(payload.encode()).hexdigest()[:7]


def print_loss_breakdown(loss_terms):
    """Per-term contribution to the total loss, as absolute value and % of total."""
    total = sum(loss_terms.values())
    print("\n  [LOSS-TERM BREAKDOWN]")
    print(f"  {'term':<14s} {'value':>12s}  {'share':>8s}")
    print("  " + "-" * 36)
    for name, val in sorted(loss_terms.items(), key=lambda kv: -kv[1]):
        share = (val / total * 100.0) if total > 0 else 0.0
        print(f"  {name:<14s} {val:12.4f}  {share:7.2f}%")
    print("  " + "-" * 36)
    print(f"  {'TOTAL':<14s} {total:12.4f}")
    print("  Tune weights against this table, not by feel: a term at >90% is "
          "swamping\n  the landscape, one at <0.1% is inert.")

# ---------------------------------------------------------------------------
# GEOMETRY HELPERS FOR PLOTTING
# ---------------------------------------------------------------------------

def rotate_points(pts, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T


def thicken_3taper_curve(curve_points, t0, t1, t2, t3, return_edges=False):
    """
    Build the closed filled polygon for a 3-taper piecewise spoke.
    If return_edges=True, returns (top, bot) — the two offset edge point arrays
    (each [N,2], both running hub→rim) so callers (e.g. the STEP exporter) can build
    smooth splines rather than one flattened polygon.  Otherwise returns the closed
    polygon [top; bot reversed] for matplotlib fill.

    Delegates to `wheel_geometry.offset_band` with one point across the thickness.
    The same function with n_across > 1 lays out the spoke's structured mesh block, so
    the meshed part and the exported part are the same surface by construction rather
    than by two implementations happening to agree.
    """
    if return_edges:
        return _geom.outline_edges(curve_points, t0, t1, t2, t3)
    return _geom.outline_polygon(curve_points, t0, t1, t2, t3)


def place_sector(polygon, hub_radius, angle_deg=0.0):
    shifted = polygon.copy()
    shifted[:, 0] += hub_radius
    return rotate_points(shifted, np.deg2rad(angle_deg))

# ---------------------------------------------------------------------------
# FILLET ARC PLOTTING HELPER
# ---------------------------------------------------------------------------

def fillet_arc_points(center, radius, start_angle_deg, end_angle_deg, n=30):
    """Returns (x, y) arrays for a circular arc segment."""
    angles = np.linspace(np.deg2rad(start_angle_deg),
                         np.deg2rad(end_angle_deg), n)
    return (center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles))


def tangent_fillet_arc(ring_radius, fillet_radius, junction_point, edge_dir,
                       outward_normal, outside_ring, n=24, max_reach=6.0):
    """
    Circular arc genuinely tangent to BOTH a ring circle (centred on the wheel origin)
    and a spoke edge, filling the reentrant corner between them.

    v2.0 drew these arcs at an arbitrary diagonal offset from the corner, so the
    figure disagreed with the tangent fillets `wheel_step_export.fillet_junctions`
    actually builds.  This solves the real construction.

      ring_radius    : hub or rim circle radius (mm)
      fillet_radius  : R
      junction_point : point on the ring circle where the spoke outline crosses it
      edge_dir       : unit direction of the outline there
      outward_normal : unit normal pointing OUT of the spoke material
      outside_ring   : True at the hub (centre at |c| = ring + R, in the free space
                       outside the hub disk), False at the rim (|c| = ring - R, in
                       the annulus inside the rim band)

    The centre must be at perpendicular distance R from the edge line AND at
    |c| = ring_radius ± R from the origin.  Writing the first locus as c = Q + u·d
    with Q = junction_point + R·n̂_out, the second gives a quadratic in u:

        u² + 2u(Q·d) + |Q|² - Rc² = 0

    Of the two roots we take the one of smaller |u| — the tangency point nearest the
    junction rather than the one wrapping around the far side of the ring.

    Returns (x, y) arrays, or None when no fillet belongs here: the loci fail to meet
    (corner not reentrant enough for this R), or the solution sits absurdly far from
    the junction.  A None is a legitimate answer — on a shallow spiral junction the
    near-tangent side has no corner to round, which is exactly why
    `wheel_step_export.fillet_junctions` also declines it.
    """
    d = np.asarray(edge_dir, dtype=float)
    d = d / (np.hypot(d[0], d[1]) or 1.0)
    nout = np.asarray(outward_normal, dtype=float)
    nout = nout / (np.hypot(nout[0], nout[1]) or 1.0)
    P = np.asarray(junction_point, dtype=float)

    Rc = ring_radius + fillet_radius if outside_ring else ring_radius - fillet_radius
    if Rc <= 0:
        return None

    Q = P + fillet_radius * nout
    b = float(Q @ d)
    c = float(Q @ Q) - Rc * Rc
    disc = b * b - c
    if disc < 0.0:
        return None                          # loci never meet — corner not reentrant
    root = np.sqrt(disc)
    u = min(-b + root, -b - root, key=abs)
    if abs(u) > max_reach * fillet_radius:
        return None                          # tangency point nowhere near the junction

    centre = Q + u * d
    r_centre = np.hypot(centre[0], centre[1])
    if r_centre < 1e-9:
        return None

    # Tangency points: radial projection onto the ring, and the foot of the
    # perpendicular from the centre onto the edge line.
    t_ring = centre * (ring_radius / r_centre)
    t_edge = P + u * d

    a0 = np.arctan2(t_ring[1] - centre[1], t_ring[0] - centre[0])
    a1 = np.arctan2(t_edge[1] - centre[1], t_edge[0] - centre[0])
    delta = (a1 - a0 + np.pi) % (2.0 * np.pi) - np.pi     # sweep the short way
    return fillet_arc_points(centre, fillet_radius,
                             np.rad2deg(a0), np.rad2deg(a0 + delta), n=n)


def ring_junction_fillets(outline, ring_radius, fillet_radius, n=24):
    """
    Every fillet where a closed spoke outline crosses a ring circle.

    Finding the junction by crossing the outline against the circle — rather than by
    assuming the offset-edge endpoints sit on it — is what makes this correct for the
    shallow spiral geometry the GA actually evolves.  A spoke leaving the hub at ~20°
    to the hub tangent has its two root corners at markedly different radii (one
    inside the hub disk, one well outside), so the endpoints are nowhere near the
    junction and the end-cap segment can be the thing that crosses.

    `outline` must be the closed polygon in the global frame (hub at the origin).
    Returns a list of (x, y) arc tuples — typically one per crossing, and a shallow
    spiral yields one filleted notch per ring rather than two.
    """
    pts = np.asarray(outline, dtype=float)
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r < ring_radius

    # Polygon orientation fixes which normal points out of the material: interior is
    # on the left of the traversal for a CCW loop, on the right for CW.
    x, y = pts[:, 0], pts[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    ccw = signed_area > 0.0

    arcs = []
    n_pts = len(pts)
    for i in range(n_pts):
        j = (i + 1) % n_pts
        if inside[i] == inside[j]:
            continue                              # segment does not cross the circle
        denom = r[j] - r[i]
        if abs(denom) < 1e-12:
            continue
        alpha = (ring_radius - r[i]) / denom
        P = pts[i] + alpha * (pts[j] - pts[i])
        norm_P = np.hypot(P[0], P[1])
        if norm_P < 1e-9:
            continue
        P = P * (ring_radius / norm_P)            # snap exactly onto the circle

        d = pts[j] - pts[i]
        d = d / (np.hypot(d[0], d[1]) or 1.0)
        nout = np.array([d[1], -d[0]]) if ccw else np.array([-d[1], d[0]])

        arc = tangent_fillet_arc(ring_radius, fillet_radius, P, d, nout,
                                 outside_ring=(ring_radius <= HUB_RADIUS_MM), n=n)
        if arc is not None:
            arcs.append(arc)
    return arcs

# ---------------------------------------------------------------------------
# CAD OUTPUT FORMATTER
# ---------------------------------------------------------------------------

def print_cad_data(opt_curve, control_pts, t0, t1, t2, t3, R_hub, R_rim):
    """Print parametric data formatted for CAD import."""
    print("\n" + "=" * 68)
    print("  CAD IMPORT DATA  — Autodesk Inventor / Fusion 360 / SolidWorks")
    print("=" * 68)
    print("\n  [BEZIER CONTROL POINTS — global frame, origin = wheel centre]")
    print(f"  Note: add HUB_RADIUS = {HUB_RADIUS_MM:.3f} mm to all X values\n")
    labels = ["P0 (hub, fixed)", "P1 (CP1)", "P2 (CP2)",
              "P3 (CP3)", "P4 (CP4)", "P5 (rim, fixed)"]
    for i, (lbl, pt) in enumerate(zip(labels, control_pts)):
        gx = pt[0] + HUB_RADIUS_MM
        gy = pt[1]
        print(f"  {lbl:20s}  X = {gx:8.4f} mm   Y = {gy:8.4f} mm")

    print("\n  [PIECEWISE THICKNESS PROFILE — 3 linear-taper zones]")
    bp = TAPER_BREAKPOINTS
    nodes = [t0, t1, t2, t3]
    arc_len = opt_curve  # use full curve for arc length
    dx = np.diff(arc_len[:, 0]); dy = np.diff(arc_len[:, 1])
    total_arc = np.sum(np.sqrt(dx**2 + dy**2))
    for i, (s, t) in enumerate(zip(bp, nodes)):
        print(f"  s = {s:.4f}  (arc = {s * total_arc:6.3f} mm)  "
              f"thickness = {t:.4f} mm")

    print("\n  [TRANSITION FILLETS]")
    print(f"  Hub fillet radius   R_hub = {R_hub:.4f} mm  "
          f"  (Kt ≈ {stress_concentration_kt(R_hub, t0):.2f})")
    print(f"  Rim fillet radius   R_rim = {R_rim:.4f} mm  "
          f"  (Kt ≈ {stress_concentration_kt(R_rim, t3):.2f})")

    print("\n  [SPOKE OUTLINE — 20 evenly-spaced centerline pts]")
    step = len(opt_curve) // 20
    print("  Idx    X (local, mm)    Y (local, mm)")
    for i in range(0, len(opt_curve), step):
        print(f"  {i:3d}   {opt_curve[i, 0]:12.4f}   {opt_curve[i, 1]:12.4f}")
    print("=" * 68)

# ---------------------------------------------------------------------------
# MAIN OPTIMISATION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evolve the compliant wheel genome, then hand off to the STEP "
                    "exporter.")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny population/generation count — exercises every code "
                             "path in seconds. Produces a throwaway genome, not a "
                             "usable design.")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--pop", type=int, default=None,
                        help="population size (sol_per_pop)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed. Passed to both numpy and pygad so a run is "
                             "reproducible end to end.")
    parser.add_argument("--no-export", action="store_true",
                        help="skip the CAD hand-off")
    parser.add_argument("--out", default="best_solution.json",
                        help="genome output path, relative to this file")
    parser.add_argument("--dump-population", action="store_true",
                        help="also write the final GA population's best genomes to "
                             "--elites-out. Stage 3 multi-starts from these; the GA "
                             "keeps only 5 elites internally and pygad is constructed "
                             "without save_solutions, so nothing else records them.")
    parser.add_argument("--elites-out", default="stage2_elites.json",
                        help="where --dump-population writes, relative to this file")
    parser.add_argument("--n-elites", type=int, default=16,
                        help="how many distinct genomes to keep (default 16)")
    parser.add_argument("--cy-bound", type=float, default=CY_BOUND_MM,
                        help=f"lateral half-range for the interior Bezier control "
                             f"points (default {CY_BOUND_MM}). The v2.1 winner has cy1 "
                             f"and cy3 pinned at this bound, so the recorded optimum is "
                             f"a boundary optimum — widening it is a real experiment.")
    args = parser.parse_args()

    if args.cy_bound != CY_BOUND_MM:
        set_cy_bound(args.cy_bound)

    # A --smoke run is for wiring, not for design: it must not clobber the real genome
    # that the STEP on disk was built from.
    if args.smoke and args.out == "best_solution.json":
        args.out = "best_solution_smoke.json"

    n_generations = args.generations if args.generations else (8 if args.smoke else 300)
    n_pop         = args.pop         if args.pop         else (24 if args.smoke else 300)
    LOG_EVERY[0]    = max(1, n_generations // 6)
    SIGMA_HALVES[0] = max(1, n_generations // 2)

    # The module-level np.random.seed(42) at import time is not enough on its own —
    # pygad draws from its OWN Generator, so without random_seed the initial population
    # differed run to run and nothing downstream was reproducible.  That matters now:
    # Stage 2 and Stage 3 record which genome they consumed, and provenance is
    # meaningless if the producing run cannot be repeated.
    np.random.seed(args.seed)

    import matplotlib
    matplotlib.use("Agg")          # headless: no DISPLAY on a build box or over ssh
    import matplotlib.pyplot as plt
    import pygad

    print("=" * 68)
    print("  COMPLIANT PLA UAV WHEEL — Co-Optimisation v2.1"
          + ("   [SMOKE RUN]" if args.smoke else ""))
    print(f"  {NUMBER_OF_SPOKES} spokes | Face width {SPOKE_WIDTH_MM} mm | "
          f"Hub Ø {HUB_RADIUS_MM*2:.1f} mm | Rim Ø {RIM_RADIUS_MM*2:.1f} mm")
    print(f"  Load {FORCE_LBS} lb ({TOTAL_FORCE_NEWTONS:.1f} N total) | "
          f"Target δ = {TARGET_DEFLECTION_MM} mm")
    print(f"  Allowable stress {ALLOWABLE_STRESS_MPA:.1f} MPa "
          f"(PLA 50 MPa × {FFF_KNOCKDOWN} FFF × 1/{SAFETY_FACTOR} SF)")
    print(f"  Beam model: {BOUNDARY_CONDITION}  (Ke = {KE_BUCKLING})")
    print(f"  GA: pop {n_pop} × {n_generations} gen | seed {args.seed}")
    print("=" * 68 + "\n")

    ga = pygad.GA(
        num_generations         = n_generations,
        num_parents_mating      = max(2, min(20, n_pop // 15)),
        fitness_func            = pygad_fitness,
        sol_per_pop             = n_pop,
        num_genes               = 14,
        gene_space              = GENE_SPACE,
        parent_selection_type   = "tournament",
        K_tournament            = 5,
        crossover_type          = "uniform",
        mutation_type           = adaptive_gaussian_mutation,
        mutation_probability    = 0.35,
        keep_elitism            = min(5, max(1, n_pop // 10)),
        on_generation           = on_generation,
        suppress_warnings       = True,
        random_seed             = args.seed,
    )

    ga.run()

    if args.dump_population:
        dump_population(ga, args.elites_out, args.n_elites)

    best_sol, best_fit, _ = ga.best_solution()
    (cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4,
     t0, t1, t2, t3, R_hub, R_rim) = best_sol

    opt_curve, ctrl_pts = generate_bezier_centerline(
        cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4)

    Kt_hub = stress_concentration_kt(R_hub, t0)
    Kt_rim = stress_concentration_kt(R_rim, t3)

    # Re-score through the same entry point the GA used, so the reported metrics and
    # the loss breakdown are guaranteed to describe one evaluation rather than two.
    best_metrics, best_loss_terms = evaluate_design(best_sol)
    defl       = best_metrics["deflection_mm"]
    max_stress = best_metrics["max_stress_mpa"]
    total_mass = best_metrics["total_mass_g"]
    buck_ratio = best_metrics["buckling_ratio"]

    print("\n" + "=" * 68)
    print("  EVOLVED OPTIMAL DESIGN")
    print("=" * 68)
    print(f"  Spoke count              : {NUMBER_OF_SPOKES}")
    print(f"  Face width (fixed)       : {SPOKE_WIDTH_MM:.2f} mm")
    print(f"  Thickness t0 (hub root)  : {t0:.3f} mm")
    print(f"  Thickness t1 (s=0.33)    : {t1:.3f} mm")
    print(f"  Thickness t2 (s=0.67)    : {t2:.3f} mm")
    print(f"  Thickness t3 (rim tip)   : {t3:.3f} mm")
    print(f"  Hub fillet  R_hub        : {R_hub:.3f} mm  (Kt = {Kt_hub:.2f})")
    print(f"  Rim fillet  R_rim        : {R_rim:.3f} mm  (Kt = {Kt_rim:.2f})")
    print("-" * 68)
    print(f"  Per-spoke deflection     : {defl:7.4f} mm  "
          f"(target {TARGET_DEFLECTION_MM} mm,  "
          f"error {abs(defl-TARGET_DEFLECTION_MM)/TARGET_DEFLECTION_MM*100:.1f}%)")
    print(f"  Peak bending stress      : {max_stress:7.2f} MPa  "
          f"(allowable {ALLOWABLE_STRESS_MPA:.1f} MPa, "
          f"utilization {max_stress/ALLOWABLE_STRESS_MPA*100:.1f}%)")
    print(f"  Buckling ratio           : {buck_ratio:7.3f}  "
          f"({'SAFE' if buck_ratio < 1.0 else 'BUCKLING RISK!'})")
    print(f"  Total wheel mass         : {total_mass:7.2f} g")
    print(f"  Beam model               : {BOUNDARY_CONDITION} (Ke = {KE_BUCKLING})")
    print("=" * 68)

    print_loss_breakdown(best_loss_terms)

    saturated = bound_saturation_report(best_sol)
    print("\n  [GENE BOUND SATURATION]  (within 1% of a limit)")
    if not saturated:
        print("  none — every gene converged in the interior of its range.")
    else:
        for name, val, which, bound in saturated:
            print(f"  {name:<6s} = {val:9.4f}  pinned at its {which:<4s} bound "
                  f"({bound:.4f})")
        print("  A pinned t0/t1/t2/t3 means MIN_WALL_MM (printability) is the active\n"
              "  constraint — informative.  A pinned cy means the centerline wanted\n"
              "  more lateral room than GENE_SPACE allows; widen it and re-run.")

    print_cad_data(opt_curve, ctrl_pts, t0, t1, t2, t3, R_hub, R_rim)

    # -----------------------------------------------------------------------
    # PERSIST WINNING GENOME  (hand-off to wheel_step_export.py / CAD)
    # -----------------------------------------------------------------------
    genome_record = {
        "genes": {
            "cx1": float(cx1), "cy1": float(cy1),
            "cx2": float(cx2), "cy2": float(cy2),
            "cx3": float(cx3), "cy3": float(cy3),
            "cx4": float(cx4), "cy4": float(cy4),
            "t0": float(t0), "t1": float(t1),
            "t2": float(t2), "t3": float(t3),
            "R_hub": float(R_hub), "R_rim": float(R_rim),
        },
        "geometry": {
            "hub_radius_mm": HUB_RADIUS_MM,
            "rim_radius_mm": RIM_RADIUS_MM,
            "spoke_width_mm": SPOKE_WIDTH_MM,
            "number_of_spokes": NUMBER_OF_SPOKES,
        },
        "metrics": {
            "deflection_mm": float(defl),
            "max_stress_mpa": float(max_stress),
            "total_mass_g": float(total_mass),
            "buckling_ratio": float(buck_ratio),
            "Kt_hub": float(Kt_hub),
            "Kt_rim": float(Kt_rim),
        },
        # Additive keys (v2.1).  wheel_step_export.py reads only `genes` and
        # `metrics.total_mass_g`, so extending the record is safe.
        # `model` records the physics a genome was optimized under — without it a
        # saved genome is not interpretable, since a cantilever-era and a
        # fixed-guided-era solution differ by ~4× in stiffness.
        "model": {
            "boundary_condition": BOUNDARY_CONDITION,
            "ke_buckling": float(KE_BUCKLING),
            "n_curve_pts": int(N_CURVE_PTS),
            "youngs_modulus_mpa": float(YOUNGS_MODULUS_PLA_MPA),
            "allowable_stress_mpa": float(ALLOWABLE_STRESS_MPA),
            "force_per_spoke_n": float(FORCE_PER_SPOKE_NEWTONS),
            "target_deflection_mm": float(TARGET_DEFLECTION_MM),
            "version": "2.1",
        },
        # Search provenance, distinct from physics: what run produced this genome and
        # inside what box.  `cy_bound_mm` is here because a genome with cy pinned at the
        # bound is a BOUNDARY optimum — its meaning depends on a number that used to
        # live only in the source.  `seed` is here so the run can be repeated at all.
        "search": {
            "seed": int(args.seed),
            "generations": int(n_generations),
            "population": int(n_pop),
            "cy_bound_mm": float(CY_BOUND_MM),
            "smoke": bool(args.smoke),
        },
        "loss_terms": {k: float(v) for k, v in best_loss_terms.items()},
        "bound_saturation": [
            {"gene": n, "value": v, "at": w, "bound": b}
            for n, v, w, b in bound_saturation_report(best_sol)
        ],
    }
    genome_path = os.path.join(PP.ROOT, args.out)
    with open(genome_path, "w") as fh:
        json.dump(genome_record, fh, indent=2)
    print(f"\nSaved winning genome → {genome_path}")

    # -----------------------------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------------------------
    spoke_poly_local  = thicken_3taper_curve(opt_curve, t0, t1, t2, t3)
    spoke_poly_placed = place_sector(spoke_poly_local, HUB_RADIUS_MM, 0.0)

    half_angle  = 360.0 / NUMBER_OF_SPOKES / 2.0
    hub_arcs    = np.linspace(-half_angle, half_angle, 120)
    full_circle = np.linspace(0, 2 * np.pi, 360)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#ccc")
        ax.xaxis.label.set_color("#ccc")
        ax.yaxis.label.set_color("#ccc")
        ax.title.set_color("#eee")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    SPOKE_COLOR  = "#00e5ff"
    CTRL_COLOR   = "#ff6e40"
    CURVE_COLOR  = "#76ff03"
    HUB_RIM_CLR  = "#e0e0e0"
    GRID_ALPHA   = 0.15

    # -- [0,0] GA Convergence -----------------------------------------------
    ax = axes[0, 0]
    gens = np.arange(len(best_fitness_history))
    ax.semilogy(gens, best_fitness_history,
                label="Best loss", color="#00e5ff", lw=2)
    ax.semilogy(gens, mean_fitness_history,
                label="Mean loss", color="#76ff03", lw=1.2, alpha=0.7)
    ax.set_title("GA Convergence (log scale)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Loss value (log)")
    ax.legend(facecolor="#222", labelcolor="#ccc")
    ax.grid(alpha=GRID_ALPHA, color="white")

    # -- [0,1] Single sector + Bezier handles -------------------------------
    ax = axes[0, 1]
    for b_angle in (-half_angle, half_angle):
        ax.plot(
            [0, RIM_RADIUS_MM * 1.12 * np.cos(np.deg2rad(b_angle))],
            [0, RIM_RADIUS_MM * 1.12 * np.sin(np.deg2rad(b_angle))],
            "--", color="#555", lw=0.8)
    ax.plot(HUB_RADIUS_MM * np.cos(np.deg2rad(hub_arcs)),
            HUB_RADIUS_MM * np.sin(np.deg2rad(hub_arcs)),
            color=HUB_RIM_CLR, lw=2)
    ax.plot(RIM_RADIUS_MM * np.cos(np.deg2rad(hub_arcs)),
            RIM_RADIUS_MM * np.sin(np.deg2rad(hub_arcs)),
            color=HUB_RIM_CLR, lw=2)
    ax.fill(spoke_poly_placed[:, 0], spoke_poly_placed[:, 1],
            color=SPOKE_COLOR, alpha=0.75, ec="white", lw=0.5)

    # Bezier control polygon
    ctrl_phys = ctrl_pts.copy()
    ctrl_phys[:, 0] += HUB_RADIUS_MM
    ax.plot(ctrl_phys[:, 0], ctrl_phys[:, 1],
            "o--", color=CTRL_COLOR, lw=1.2, ms=5, label="Bézier handles")
    for i, lbl in enumerate(["P0", "P1", "P2", "P3", "P4", "P5"]):
        ax.annotate(lbl, ctrl_phys[i], textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5,
                    color=CTRL_COLOR, fontweight="bold")

    # Draw the (GA-optimized) transition fillets, constructed tangent to both the ring
    # circle and the spoke outline, so the figure agrees with the STEP that
    # wheel_step_export.fillet_junctions builds.  A shallow spiral junction only has a
    # sharp notch on one side, so expect one arc per ring, not two.
    FILLET_CLR = "#ffd54f"
    fillet_arcs = (ring_junction_fillets(spoke_poly_placed, HUB_RADIUS_MM, R_hub)
                   + ring_junction_fillets(spoke_poly_placed, RIM_RADIUS_MM, R_rim))
    for j, (fx, fy) in enumerate(fillet_arcs):
        ax.plot(fx, fy, color=FILLET_CLR, lw=1.8, zorder=6,
                label="Fillets (tangent, R_hub / R_rim)" if j == 0 else None)
    ax.annotate(f"R_hub = {R_hub:.2f} mm", (HUB_RADIUS_MM, -t0 / 2.0),
                textcoords="offset points", xytext=(4, -12),
                fontsize=7, color=FILLET_CLR)
    ax.annotate(f"R_rim = {R_rim:.2f} mm", (RIM_RADIUS_MM, t3 / 2.0),
                textcoords="offset points", xytext=(-40, 10),
                fontsize=7, color=FILLET_CLR)

    ax.set_aspect("equal")
    ax.set_title(f"1/{NUMBER_OF_SPOKES} Sector — 3-Taper Profile")
    ax.legend(facecolor="#222", labelcolor="#ccc", fontsize=8)
    ax.grid(alpha=GRID_ALPHA, color="white")

    # -- [0,2] Full wheel ---------------------------------------------------
    ax = axes[0, 2]
    ax.plot(HUB_RADIUS_MM * np.cos(full_circle),
            HUB_RADIUS_MM * np.sin(full_circle),
            color=HUB_RIM_CLR, lw=2.5)
    ax.plot(RIM_RADIUS_MM * np.cos(full_circle),
            RIM_RADIUS_MM * np.sin(full_circle),
            color=HUB_RIM_CLR, lw=2.5)
    for k in range(NUMBER_OF_SPOKES):
        angle_k = k * (360.0 / NUMBER_OF_SPOKES)
        rot_poly = rotate_points(spoke_poly_placed, np.deg2rad(angle_k))
        ax.fill(rot_poly[:, 0], rot_poly[:, 1],
                color=SPOKE_COLOR, alpha=0.8, ec="white", lw=0.3)
    ax.set_aspect("equal")
    ax.set_title(f"Full Wheel ({NUMBER_OF_SPOKES}× Symmetry)")
    ax.grid(alpha=GRID_ALPHA, color="white")

    # -- [1,0] Thickness profile along arc ----------------------------------
    ax = axes[1, 0]
    arc_s = np.linspace(0, 1, 500)
    t_profile = thickness_at_arc_length(arc_s, t0, t1, t2, t3)
    ax.fill_between(arc_s, t_profile, alpha=0.4, color=SPOKE_COLOR)
    ax.plot(arc_s, t_profile, color=SPOKE_COLOR, lw=2.5,
            label="Thickness (mm)")
    ax.axvline(1/3, color="#ff6e40", ls="--", lw=1, label="Zone boundaries")
    ax.axvline(2/3, color="#ff6e40", ls="--", lw=1)
    bp_labels = ["t₀ (root)", "t₁", "t₂", "t₃ (tip)"]
    bp_vals   = [t0, t1, t2, t3]
    for bp, bv, blbl in zip(TAPER_BREAKPOINTS, bp_vals, bp_labels):
        ax.plot(bp, bv, "o", color=CTRL_COLOR, ms=7, zorder=5)
        ax.annotate(f"{blbl}\n{bv:.2f}mm", (bp, bv),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7.5, color=CTRL_COLOR)
    ax.set_xlabel("Normalized arc length s")
    ax.set_ylabel("Thickness (mm)")
    ax.set_title("3-Zone Piecewise Linear Thickness Profile")
    ax.legend(facecolor="#222", labelcolor="#ccc", fontsize=8)
    ax.grid(alpha=GRID_ALPHA, color="white")

    # -- [1,1] Isolated centerline curvature --------------------------------
    ax = axes[1, 1]
    ax.plot(opt_curve[:, 0], opt_curve[:, 1],
            "-", color=CURVE_COLOR, lw=2.5, label="Optimized centerline")
    ax.set_xlabel("Local X span (mm)")
    ax.set_ylabel("Local Y deviation (mm)")
    ax.set_title("Isolated Spoke Curvature Mapping")
    ax.axhline(0, color="#555", lw=0.8, ls=":")
    ax.legend(facecolor="#222", labelcolor="#ccc", fontsize=8)
    ax.grid(alpha=GRID_ALPHA, color="white")

    # -- [1,2] Stress utilization bar chart ---------------------------------
    ax = axes[1, 2]
    metrics = {
        "Stress\nutilization": max_stress / ALLOWABLE_STRESS_MPA,
        "Deflection\nerror":   abs(defl - TARGET_DEFLECTION_MM) / TARGET_DEFLECTION_MM,
        "Buckling\nratio":     buck_ratio,
    }
    bar_colors = [
        "#ff6e40" if v > 0.95 else "#76ff03"
        for v in metrics.values()
    ]
    bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                  color=bar_colors, edgecolor="white", lw=0.7)
    ax.axhline(1.0, color="white", ls="--", lw=1, label="Limit = 1.0")
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, color="#eee")
    ax.set_ylabel("Ratio (< 1.0 = safe)")
    ax.set_title("Design Constraint Utilization")
    ax.set_ylim(0, max(1.3, max(metrics.values()) * 1.15))
    ax.legend(facecolor="#222", labelcolor="#ccc", fontsize=8)
    ax.grid(alpha=GRID_ALPHA, color="white", axis="y")

    fig.suptitle(
        f"Compliant PLA UAV Wheel  —  Co-Optimized Shape & 3-Taper Thickness  "
        f"[v2.1 | {NUMBER_OF_SPOKES} spokes]",
        fontsize=14, fontweight="bold", color="#eee"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    # Beside this file, not at an absolute path.  This was hard-coded to
    # C:/Users/crfloyd/github/wheel/, which is a hard failure anywhere but the machine
    # it was written on — and it fires at the very END of a multi-minute GA run, after
    # best_solution.json is already safely on disk but before the CAD hand-off, so it
    # took out the STEP export too.
    # The figure name FOLLOWS --out.  It used to be a fixed "poster_summary.jpg", which
    # meant any experimental run — a bound sweep, a seed comparison, anything writing
    # its genome elsewhere — silently overwrote the committed poster with a throwaway
    # design while leaving best_solution.json alone.  That is the same class of
    # artifacts-disagreeing-with-each-other bug the CAD hand-off exists to prevent.
    # `genome_path` is already resolved against this file's directory, so the figure
    # lands beside whatever genome it actually describes — including out-of-tree ones.
    if args.out == "best_solution.json":
        out_path = os.path.join(os.path.dirname(genome_path), "poster_summary.jpg")
    else:
        out_path = os.path.splitext(genome_path)[0] + "_summary.jpg"
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"\nSaved summary figure → {out_path}")

    # -----------------------------------------------------------------------
    # CAD HAND-OFF
    # -----------------------------------------------------------------------
    # The optimizer (Python 3.14 + pygad) and the STEP exporter (Python 3.12 +
    # CadQuery/OCC) cannot share an interpreter, so the export used to be a
    # second command run by hand.  It got forgotten, and `wheel.step` silently
    # went on describing a previous genome while the figure showed the new one.
    # Driving it from here makes the three artefacts land together, always.
    #
    # Deliberately non-fatal: a GA run costs minutes, a missing .venv-cad or an
    # OCC boolean failure must never destroy one.  Worst case we print the
    # manual command and exit normally.
    import sys
    import subprocess

    here = PP.ROOT

    # venv puts the interpreter in Scripts/python.exe on Windows and bin/python on
    # POSIX.  The old code knew only the Windows layout, so on Linux the hand-off
    # reported "env not found" even when .venv-cad was sitting right there.  Probe both
    # and report whichever is right for this platform.
    cad_candidates = [
        os.path.join(here, ".venv-cad", "bin", "python"),
        os.path.join(here, ".venv-cad", "Scripts", "python.exe"),
    ]
    cad_python = next((p for p in cad_candidates if os.path.isfile(p)), None)
    manual_cmd = (".venv-cad\\Scripts\\python src\\wheel_step_export.py"
                  if os.name == "nt" else
                  ".venv-cad/bin/python src/wheel_step_export.py")

    print("\n" + "=" * 68)
    print("  CAD HAND-OFF → wheel_step_export.py")
    print("=" * 68)
    # The child writes straight to this stdout; flush ours first or, when the run is
    # piped to a file, our block-buffered output lands *after* the child's.
    sys.stdout.flush()
    if args.no_export or args.smoke:
        # The exporter reads best_solution.json unconditionally (load_genome,
        # wheel_step_export.py:125).  A smoke run wrote its genome elsewhere, so
        # exporting here would rebuild the STEP from the *previous* real genome and
        # stamp it with a fresh timestamp — destroying exactly the staleness signal
        # warn_if_stale() exists to give.
        why = "--no-export" if args.no_export else "--smoke (genome is throwaway)"
        print(f"  Skipped: {why}")
        print(f"  Rebuild the STEP with:\n    {manual_cmd}")
    elif cad_python is None:
        print(f"  CadQuery env not found — looked for:")
        for p in cad_candidates:
            print(f"    {p}")
        print(f"  STEP not regenerated.  Build it with `make env-cad`, then run:"
              f"\n    {manual_cmd}")
    else:
        try:
            # Inherit stdout so the exporter's own report (junction overlaps,
            # fillet radii, Kt mismatch) shows up in this run's log.
            proc = subprocess.run(
                [cad_python, os.path.join(PP.SRC, "wheel_step_export.py")],
                cwd=here, timeout=900, check=False,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            if proc.returncode != 0:
                print(f"\n  STEP export exited {proc.returncode} — genome is safe "
                      f"in best_solution.json.\n  Retry with:\n    {manual_cmd}")
        except subprocess.TimeoutExpired:
            print(f"\n  STEP export timed out (>900 s).  Retry with:\n    {manual_cmd}")
        except OSError as exc:
            print(f"\n  Could not launch the CAD env ({exc}).\n  Retry with:\n    {manual_cmd}")

    print("\nDone.")
