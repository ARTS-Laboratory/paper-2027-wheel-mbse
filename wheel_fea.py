"""
=============================================================================
  COMPLIANT PLA UAV WHEEL — Co-Optimization v2.0
  Senior ME Refactor: Expanded Genome, Rigorous Physics, Enhanced Output
=============================================================================

CHAIN-OF-THOUGHT ENGINEERING NOTES
------------------------------------
1. GENOME (14 genes)
   - Bezier centerline upgraded from degree-3 (4 pts) to degree-5 (6 pts),
     adding two interior control points [P1..P4] that allow S-curves and
     progressive bend profiles.  P0 (hub) and P5 (rim) remain locked.  Interior
     control-point y-range is held to ±25 mm (was ±55) to keep the search in
     physically reasonable, non-lumpy territory.
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
   c. Euler column buckling check on each segment (Pe = π²EI / (Ke·L)²) with
      an effective-length factor Ke=0.7 (one end fixed, one pin).  A penalty
      fires if any segment's axial compressive load exceeds 0.6·Pe.
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
import json
import os
import numpy as np
# NOTE: pygad and matplotlib are imported lazily inside __main__ so this module can
# be imported (for its geometry functions) in a minimal env — e.g. the CadQuery
# Python 3.12 venv used by wheel_step_export.py — with only numpy present.

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(42)

# ---------------------------------------------------------------------------
# PHYSICAL & GEOMETRIC CONSTANTS
# ---------------------------------------------------------------------------
NUMBER_OF_SPOKES    = 12
HUB_RADIUS_MM       = 25.4 / 2          # 12.7 mm
RIM_RADIUS_MM       = 97.8 / 2          # 48.9 mm
HUB_RIM_SPAN_MM     = RIM_RADIUS_MM - HUB_RADIUS_MM   # 36.2 mm
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

# Buckling effective-length factor (one-end fixed, one-end pinned)
KE_BUCKLING              = 0.7

# Piecewise taper breakpoints in normalized arc-length
TAPER_BREAKPOINTS = np.array([0.0, 1/3, 2/3, 1.0])

# Number of discretization points along Bezier
N_CURVE_PTS = 600

# ---------------------------------------------------------------------------
# GENOME LAYOUT  (14 genes total)
# ---------------------------------------------------------------------------
# Indices:
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

GENE_SPACE = [
    # CP1 — near hub (x must stay in [5%, 30%] of span)
    {"low": S * 0.05,  "high": S * 0.30},   # cx1
    {"low": -25.0,     "high":  25.0},       # cy1
    # CP2 — first third to midspan
    {"low": S * 0.22,  "high": S * 0.55},   # cx2
    {"low": -25.0,     "high":  25.0},       # cy2
    # CP3 — midspan to second third
    {"low": S * 0.45,  "high": S * 0.78},   # cx3
    {"low": -25.0,     "high":  25.0},       # cy3
    # CP4 — near rim (x must stay in [70%, 95%] of span)
    {"low": S * 0.70,  "high": S * 0.95},   # cx4
    {"low": -25.0,     "high":  25.0},       # cy4
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
    """
    pts = np.array([
        [0.0,   0.0 ],
        [cx1,   cy1 ],
        [cx2,   cy2 ],
        [cx3,   cy3 ],
        [cx4,   cy4 ],
        [span_mm, 0.0],
    ])
    # Degree-5 Bernstein basis
    t = np.linspace(0.0, 1.0, num_points)[:, None]   # (N,1)
    n = 5
    B = np.array([
        _binomial(n, k) * (1 - t)**(n - k) * t**k
        for k in range(n + 1)
    ]).squeeze(axis=-1).T   # (N, 6)
    curve = B @ pts          # (N, 2)
    return curve, pts


def _binomial(n, k):
    """Binomial coefficient as float array-friendly scalar."""
    from math import comb
    return float(comb(n, k))

# ---------------------------------------------------------------------------
# PIECEWISE-LINEAR THICKNESS PROFILE
# ---------------------------------------------------------------------------

def thickness_at_arc_length(arc_fractions, t0, t1, t2, t3):
    """
    Returns thickness at each normalized arc-fraction in [0,1].
    Three zones: [0, 1/3], [1/3, 2/3], [2/3, 1].
    """
    nodes = np.array([t0, t1, t2, t3])
    bp    = TAPER_BREAKPOINTS
    thickness = np.zeros_like(arc_fractions)
    for i in range(3):
        mask = (arc_fractions >= bp[i]) & (arc_fractions <= bp[i + 1])
        alpha = (arc_fractions[mask] - bp[i]) / (bp[i + 1] - bp[i] + 1e-12)
        thickness[mask] = nodes[i] * (1 - alpha) + nodes[i + 1] * alpha
    return thickness

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
    """
    if radius_of_curvature <= 0:
        return 2.5
    ratio = section_height / (4.0 * radius_of_curvature)
    kb = 1.0 + ratio
    return float(np.clip(kb, 1.0, 2.5))

# ---------------------------------------------------------------------------
# CORE MECHANICAL ANALYSIS
# ---------------------------------------------------------------------------

def generalized_spoke_mechanics(curve_points, t0, t1, t2, t3,
                                 spoke_width, force_per_spoke,
                                 R_hub_fillet=0.5, R_rim_fillet=0.5,
                                 youngs_modulus=YOUNGS_MODULUS_PLA_MPA):
    """
    Upgraded mechanics:
      - Piecewise-linear 3-taper thickness profile
      - Kt stress-concentration at hub and rim fillets
      - Winkler-Bach curved-beam correction per segment
      - Castigliano deflection (bending + axial)
      - Euler column buckling check per segment
      - Returns: deflection_mm, max_stress_mpa, total_mass_g,
                 buckling_ratio (max axial_load / 0.6*Pe per segment)
    """
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
    thicknesses = np.clip(thicknesses, 0.5, 20.0)

    # --- Section properties per segment ---
    I  = spoke_width * thicknesses**3 / 12.0    # second moment of area
    A  = spoke_width * thicknesses               # cross-section area

    # --- Local geometry ---
    cos_theta = dx / np.where(seg_lengths < 1e-8, 1e-8, seg_lengths)
    y_mid     = curve_points[:-1, 1] + dy / 2.0

    # --- Bending moment (transverse load F × lateral deviation) ---
    M = force_per_spoke * y_mid

    # --- Axial component (projection of F along spoke axis) ---
    F_axial = force_per_spoke * cos_theta

    # --- Castigliano deflection contributions ---
    defl_bending = np.sum((force_per_spoke * y_mid**2) /
                           (youngs_modulus * I) * seg_lengths)
    defl_axial   = np.sum((force_per_spoke * cos_theta**2) /
                           (youngs_modulus * A) * seg_lengths)
    total_deflection = defl_bending + defl_axial

    # --- Radius of curvature per segment (finite-difference of tangent angle) ---
    tangent_angles = np.arctan2(dy, dx)
    d_theta = np.diff(tangent_angles)
    seg_len_mid = (seg_lengths[:-1] + seg_lengths[1:]) / 2.0  # n_seg-1 interior
    with np.errstate(divide="ignore", invalid="ignore"):
        R_curv_interior = np.where(np.abs(d_theta) < 1e-8, 1e6,
                                   seg_len_mid / np.abs(d_theta))
    # Pad to length n_seg
    R_curv = np.concatenate([[R_curv_interior[0]],
                              R_curv_interior,
                              [R_curv_interior[-1]]])

    # --- Curved-beam correction factor per segment ---
    kb = np.array([curved_beam_factor(R_curv[i], thicknesses[i])
                   for i in range(n_seg)])

    # --- Bending stress (outer fibre), with curved-beam correction ---
    sigma_bending = np.abs(M) * (thicknesses / 2.0) / I * kb

    # --- Axial stress (compressive positive convention) ---
    sigma_axial = F_axial / A

    # --- Kt at hub (segment 0) and rim (segment -1) ---
    Kt_hub = stress_concentration_kt(R_hub_fillet, thicknesses[0])
    Kt_rim = stress_concentration_kt(R_rim_fillet, thicknesses[-1])

    # Apply Kt to the first and last segments
    sigma_combined = sigma_bending + np.abs(sigma_axial)
    sigma_combined[0]  *= Kt_hub
    sigma_combined[-1] *= Kt_rim

    max_stress = float(np.max(sigma_combined))

    # --- Euler buckling check ---
    # Pe = π² E I / (Ke L)²  per segment
    Pe = (np.pi**2 * youngs_modulus * I) / ((KE_BUCKLING * seg_lengths)**2 + 1e-12)
    F_comp = np.clip(-F_axial, 0, None)   # compressive part only
    buckling_ratio = float(np.max(F_comp / (0.6 * Pe + 1e-12)))

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

def pygad_fitness(ga_instance, solution, sol_idx):
    (cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4,
     t0, t1, t2, t3, R_hub, R_rim) = solution

    # -------- Geometric monotonicity: soft penalty on x-ordering ----------
    xs = [0.0, cx1, cx2, cx3, cx4, S]
    x_order_penalty = 0.0
    for i in range(len(xs) - 1):
        violation = xs[i] + 2.0 - xs[i + 1]   # must be cx[i]+gap < cx[i+1]
        x_order_penalty += soft_barrier(violation, scale=80.0)

    curve_points, _ = generate_bezier_centerline(
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
    defl_err   = abs(deflection - TARGET_DEFLECTION_MM) / TARGET_DEFLECTION_MM
    defl_loss  = 2500.0 * (defl_err**2)

    # -------- Mass objective (minimise) --------------------------------
    mass_loss  = total_mass / 40.0

    # -------- Stress penalty (smooth quadratic barrier) ----------------
    #  Scale raised to 4000: at the higher deflection target we lean harder on the
    #  material, so any genome over the (already SF/FFF-derated) allowable is firmly
    #  rejected — more compliance without snapping.
    stress_viol  = max_stress / ALLOWABLE_STRESS_MPA - 1.0
    stress_penalty = soft_barrier(stress_viol, scale=4000.0)

    # -------- Buckling penalty -----------------------------------------
    buck_viol  = buckling_ratio - 1.0
    buck_penalty = soft_barrier(buck_viol, scale=2000.0)

    # -------- Overlap penalty (in-plane hub crowding, gene-driven) ------
    overlap_pen = spoke_overlap_penalty(t0, R_hub)

    # -------- Smooth-spiral regularization -----------------------------
    #  Penalize curvature reversals (want a single-curvature spiral) and general
    #  wiggle.  Without this the GA settles on lumpy centerlines that still hit the
    #  deflection target.
    n_infl, turn_var = curve_smoothness_metrics(curve_points)
    smoothness_penalty = 400.0 * n_infl + 120.0 * turn_var

    total_loss = (defl_loss + mass_loss + stress_penalty + buck_penalty
                  + x_order_penalty + overlap_pen + smoothness_penalty)

    return -total_loss   # PyGAD maximises; lower loss → higher fitness


# ---------------------------------------------------------------------------
# GENERATION CALLBACK (logging + adaptive mutation)
# ---------------------------------------------------------------------------

def on_generation(ga_instance):
    scores = ga_instance.last_generation_fitness
    best_fitness_history.append(-np.max(scores))
    mean_fitness_history.append(-np.mean(scores))

    # Halve mutation sigma every 250 generations (adaptive cooling)
    gen = ga_instance.generations_completed
    if gen > 0 and gen % 250 == 0:
        mutation_sigma_global[0] *= 0.5
        print(f"  [Gen {gen:4d}] Mutation σ reduced to "
              f"{mutation_sigma_global[0]:.4f}")

    if gen % 100 == 0:
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
    """
    # Arc-length fractions
    dx = np.diff(curve_points[:, 0])
    dy = np.diff(curve_points[:, 1])
    segs = np.sqrt(dx**2 + dy**2)
    cum  = np.concatenate([[0.0], np.cumsum(segs)])
    arc_fracs = cum / (cum[-1] + 1e-12)
    thicknesses = thickness_at_arc_length(arc_fracs, t0, t1, t2, t3)

    # Normals along centerline
    grads = np.gradient(curve_points, axis=0)
    norms = np.stack([-grads[:, 1], grads[:, 0]], axis=1)
    norms /= np.linalg.norm(norms, axis=1, keepdims=True) + 1e-12

    half_t = thicknesses[:, None] / 2.0
    top    = curve_points + norms * half_t
    bot    = curve_points - norms * half_t
    if return_edges:
        return top, bot
    return np.vstack([top, bot[::-1]])


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
    import matplotlib.pyplot as plt
    import pygad

    print("=" * 68)
    print("  COMPLIANT PLA UAV WHEEL — Co-Optimisation v2.0")
    print(f"  {NUMBER_OF_SPOKES} spokes | Face width {SPOKE_WIDTH_MM} mm | "
          f"Hub Ø {HUB_RADIUS_MM*2:.1f} mm | Rim Ø {RIM_RADIUS_MM*2:.1f} mm")
    print(f"  Load {FORCE_LBS} lb ({TOTAL_FORCE_NEWTONS:.1f} N total) | "
          f"Target δ = {TARGET_DEFLECTION_MM} mm")
    print(f"  Allowable stress {ALLOWABLE_STRESS_MPA:.1f} MPa "
          f"(PLA 50 MPa × {FFF_KNOCKDOWN} FFF × 1/{SAFETY_FACTOR} SF)")
    print("=" * 68 + "\n")

    ga = pygad.GA(
        num_generations         = 1200,
        num_parents_mating      = 20,
        fitness_func            = pygad_fitness,
        sol_per_pop             = 300,
        num_genes               = 14,
        gene_space              = GENE_SPACE,
        parent_selection_type   = "tournament",
        K_tournament            = 5,
        crossover_type          = "uniform",
        mutation_type           = adaptive_gaussian_mutation,
        mutation_probability    = 0.35,
        keep_elitism            = 5,
        on_generation           = on_generation,
        suppress_warnings       = True,
    )

    ga.run()

    best_sol, best_fit, _ = ga.best_solution()
    (cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4,
     t0, t1, t2, t3, R_hub, R_rim) = best_sol

    opt_curve, ctrl_pts = generate_bezier_centerline(
        cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4)

    Kt_hub = stress_concentration_kt(R_hub, t0)
    Kt_rim = stress_concentration_kt(R_rim, t3)

    (defl, max_stress, total_mass,
     buck_ratio) = generalized_spoke_mechanics(
        opt_curve, t0, t1, t2, t3,
        SPOKE_WIDTH_MM, FORCE_PER_SPOKE_NEWTONS,
        R_hub_fillet=R_hub, R_rim_fillet=R_rim)

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
    print("=" * 68)

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
    }
    genome_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "best_solution.json")
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

    # Draw the (now GA-optimized) transition fillets at the hub root and rim tip.
    # Root corners sit at (HUB_RADIUS, ±t0/2); tip corners at (RIM_RADIUS, ±t3/2)
    # since the centerline endpoints P0/P5 are locked to y=0.
    FILLET_CLR = "#ffd54f"
    fillet_corners = [
        # (corner_x,        corner_y,   radius, arc_start, arc_end)
        (HUB_RADIUS_MM,  t0 / 2.0,  R_hub, 180, 270),   # hub, top
        (HUB_RADIUS_MM, -t0 / 2.0,  R_hub,  90, 180),   # hub, bottom
        (RIM_RADIUS_MM,  t3 / 2.0,  R_rim, 270, 360),   # rim, top
        (RIM_RADIUS_MM, -t3 / 2.0,  R_rim,   0,  90),   # rim, bottom
    ]
    for j, (cxp, cyp, rad, a0, a1) in enumerate(fillet_corners):
        sign_y = 1.0 if cyp >= 0 else -1.0
        # Fillet-arc centre offset diagonally out of the corner into the free space.
        c_x = cxp + (rad if cxp < RIM_RADIUS_MM * 0.5 else -rad)
        c_y = cyp + sign_y * rad
        fx, fy = fillet_arc_points((c_x, c_y), rad, a0, a1, n=24)
        ax.plot(fx, fy, color=FILLET_CLR, lw=1.6,
                label="Fillets (R_hub / R_rim)" if j == 0 else None)
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
        f"[v2.0 | {NUMBER_OF_SPOKES} spokes]",
        fontsize=14, fontweight="bold", color="#eee"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = "C:/Users/crfloyd/github/wheel/poster_summary.jpg"
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"\nSaved summary figure → {out_path}")
    print("\nDone.")
