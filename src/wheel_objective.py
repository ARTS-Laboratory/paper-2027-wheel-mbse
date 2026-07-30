"""
=============================================================================
  THE STAGE-3 OBJECTIVE — every loss term, from the mesh, with its gradient
=============================================================================

    from wheel_objective import objective, print_loss_breakdown
    val, grad, brk = objective(genes, "coarse")
    print_loss_breakdown(brk)

M7 proved the adjoint.  This is what the adjoint is *for*: the scalar Stage 3
descends, assembled from the FEA rather than from the beam surrogate the GA used.

WHY THIS IS A MILESTONE AND NOT A FUNCTION
------------------------------------------
Recomputing the loss terms from the FEA does not adjust the objective, it INVERTS it.
The GA's beam model reports 1.990 mm of axle drop against a 2.0 mm target, so its
`deflection` term is 0.06 — 0.1% of a total loss of 50.4, essentially satisfied.  The
real contact FEA reports a phase-mean drop of 1.503 mm at the same genome, a -24.8%
error, i.e. `2500 * 0.248**2` = 154, three times the whole GA loss.  `mass` (79.2%
today) and `smoothness` (20.6%) stop being the objective.  Stage 3's job is softening
the wheel, and nothing knew that until these terms were evaluated on FEA quantities.

The wheel is also already exactly on the stress barrier (25.08 MPa against a 25.0
allowable), and M4 measured `compliance_split.rim = 0.324` — a third of the compliance
is in a rim band no gene touches.  So the calibration below is being asked for +33%
compliance with no stress headroom, and whether that is a weighting problem or a
FEASIBILITY problem is the question this module exists to make visible BEFORE a
four-hour Stage-3 run answers it by not converging.

THREE TIERS, AND THE TIERING IS THE DESIGN
------------------------------------------
    T1  gene-space   x_order, hub_overlap, fold, arrival, smoothness,
                     fillet_feasibility, buckling         genes only        ~ms
    T2  mesh-space   mass, min_sj                          mesh_coords       ~ms
    T3  solve-space  deflection, stress, phase_ripple      solve + adjoint   ~0.7 s/phase

Not organisation — economics.  T1 and T2 are four orders of magnitude cheaper than T3,
so every feasibility barrier can be evaluated on every trial step of a line search while
the solve runs once.  A single fused objective would make the cheapest constraints cost
what the most expensive one does.

WHAT IS NOT DIFFERENTIABLE IS NAMED, NOT HIDDEN
-----------------------------------------------
Two things carry no gradient, and both are asserted rather than tolerated (gate 8):

  * `buckling` is the legacy Euler equivalent-column proxy, computed by the numpy
    `generalized_spoke_mechanics`.  Porting that solve to jnp is not hard, it is just
    not this milestone; the master plan's replacement is `lambda_min(K_t)` via LOBPCG and
    this repo has no eigen-solver at all.  The term is currently EXACTLY ZERO (ratio
    0.092 against a threshold of 1.0), so it costs nothing today — and the gate asserts
    it is still zero, so the day it starts to bite, the census fails and says so rather
    than quietly handing the optimizer an inert 2000-unit penalty.

  * `R_hub` and `R_rim` have no FEA gradient, because no fillet is meshed —
    `dcoords/dgene` is identically zero, which M7 proved at the mesh with no solver
    involved.  They are NOT dead here: `fillet_feasibility` is a closed-form geometric
    margin and gives both a real gradient in gene space.  That is the whole reason the
    term exists, and the census asserts it both ways.

THE STRESS CONSTRAINT IS NOT THE P-NORM
---------------------------------------
A volume-weighted p-norm over the whole wheel is diluted by the hub, the rim and the
thick spoke roots — most of the volume, almost none of the stress.  At the master plan's
p=10 the p-norm is 40% of the true max, so "p-norm <= 25 MPa" would admit a 60 MPa peak.
`wheel_adjoint.STRESS_PNORM_P` carries the measurement that sets p=30; here the p-norm is
rescaled by the measured `max/pnorm` ratio, held CONSTANT within a step so the gradient
stays exact (the p-norm is positively homogeneous, so a constant rescale is exact rather
than approximate).  The ratio's stability across designs — cv 4.4% at p=30 — is what
makes that defensible, and it is the same test M4 applied to `fea_over_beam`, which
failed it at cv 62% and closed the Stage-2.5 off-ramp.

AND THAT ARGUMENT IS ABOUT THE WRONG FACTOR.  M8b-i.5 put both factors up a three-rung
mesh ladder: `max/pnorm` is the BEST-behaved quantity in the whole decomposition (GCI
5.3%, converged), because it is a ratio of two things that diverge together.  The p-norm
itself is at GCI 47% and the utilisation at 63%, while the axle drop off the very same
solves converges at order 2.44 to GCI 0.14%.  At p=30 the p-norm is 1/1.38 of the true
max — it IS the max in disguise — so it inherits the r^-0.5 field of the unfilleted
349.5-degree spoke/ring corner that `study_wheel_fea.stress_report` identifies as
geometrically a crack.  The exponent is therefore an argument (`stress_gauss_p`) and
`t3_terms` can probe others on the same solve (`stress_p_probe`).

AND THE RESCALE IS GONE.  M8b-i.6 swept ten exponents off one set of solves and settled
it: the p-norm converges for p <= 6, `max/pnorm` only for p >= 24, and their product — the
utilisation, the number Stage 3 was descending on — at NO exponent at either design.  A
factor anchored to a singularity cannot be stabilised by freezing it.  So the peak is
modelled instead of measured: `Kt(R, t) * sigma_nominal(p=4) <= ALLOWABLE`, where
`sigma_nominal` is mesh-convergent (GCI 0.45%) and `Kt` is the same analytic
stress-concentration factor `wheel_fea` and the STEP exporter already price, now
differentiated rather than frozen.  `STRESS_NOMINAL_P` is the default here;
`wheel_adjoint.STRESS_PNORM_P` stays 30.0 so historical records keep their meaning.

PHASE
-----
Not moot: M6 measured 7.0% std/mean and 19.7% peak-to-peak in axle drop over the 30
degree period, against the master plan's 2% "phase is nearly moot" threshold.  So
`phase_ripple` is a design term, not just a quadrature nuisance.

The stencil is a FIXED LATTICE and that is a performance fact.  `wheel_wheel.coord_fn`
keys its jit cache on phase, so a continuously-random RQMC offset misses on every phase
of every step and pays M7's measured 0.774 s re-trace eight times per step — roughly
double the actual solving.  `phase_stencil` therefore quantizes the random offset onto an
`n_phase x n_sub` lattice: genuinely stochastic, unbiased, and cached after one pass.

Randomizing at all buys something specific.  M7 found the contact facets on the rim
discretisation (3.8% of the slope at `coarse`, 17.6% at `smoke`), an artefact that
refines away and exists because no config resolves the 0.484 degree contact patch —
`coarse` gets 1.75 nodes across it against the 8 the master plan assumed.  It lives at the
0.25 degree quadrature spacing while the stencil samples at 3.75 degrees, so it ALIASES.
Under a fixed stencil the aliased artefact is a deterministic function of the genes with a
nonzero gradient, which is exactly what an optimizer chases; a random shift makes it
zero-mean noise instead.
=============================================================================
"""

import jax_config  # noqa: F401  — must precede every other jax import
import jax
import jax.numpy as jnp
import numpy as np

import wheel_adjoint as WA
import wheel_fea as W
import wheel_genome as wg
import wheel_geometry as geom
import wheel_mesh as wm
import wheel_wheel as WW

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

SECTOR_DEG = WW.SECTOR_DEG                  # 30 degrees, the phase period
TARGET_DEFLECTION_MM = W.TARGET_DEFLECTION_MM
ALLOWABLE_STRESS_MPA = W.ALLOWABLE_STRESS_MPA
SERVICE_FORCE_N = W.TOTAL_FORCE_NEWTONS

# Gauss-point p-norm exponent for the NOMINAL stress the constraint is built on — NOT
# `WA.STRESS_PNORM_P` (30.0), which stays the adjoint's documented default so historical
# records keep their meaning.  M8b-i.6 step 1 swept ten exponents off one set of solves
# (`study_stage3_pnorm.json`): 4.0 is the largest whose observed order is still ~2 at BOTH
# designs — 1.76 / 2.18, GCI 0.45% / 0.20%, comparable to the axle drop's 2.44 / 0.14%.
# At p=6 the GCI still clears the 5% gate but the order has collapsed to 0.99 / 1.12, i.e.
# inside the gate by margin rather than by mechanism.  Below p=3 it slides toward a plain
# volume mean and loses the peak sensitivity the constraint exists for.
STRESS_NOMINAL_P = 4.0

# Mesh-validity barrier.  0.2 is `study_wheel_mesh.MIN_SJ_ACCEPT`, the value M2b's
# design-space sweep established as the floor for a usable element; the master plan's
# hard step-reject sits lower, at 0.05, so the barrier pushes back before the reject
# fires rather than instead of it.
MIN_SJ_TARGET = 0.2

# Free arc the spoke root must leave at the hub, on top of `HUB_CLEARANCE_MM`.  The
# master plan's tightening: "so the mesh never sees the degenerate case".
HUB_FREE_ARC_MM = 0.5

# Fillet feasibility margin, mm, with the master plan's 10% safety factor applied where
# the barrier is evaluated rather than baked in here.
MIN_FILLET_MARGIN_MM = 0.1
FILLET_SAFETY = 1.1

TERMS = ("deflection", "mass", "stress", "buckling", "x_order", "hub_overlap",
         "smoothness", "fold", "arrival", "fillet", "min_sj", "phase_ripple")

# Ported unchanged from `wheel_fea.evaluate_design` except where noted.  `smoothness` and
# `fillet` are new scales and are what gate 10's calibration is for; `min_sj` and
# `phase_ripple` are new terms.
DEFAULT_WEIGHTS = {
    "deflection":  2500.0,      # x (rel error)^2, two-sided about the 2.0 mm target
    "mass":        W.MASS_WEIGHT / W.MASS_REFERENCE_G,
    "stress":      4000.0,      # soft_barrier scale on utilisation - 1
    "buckling":    2000.0,      # soft_barrier scale on ratio - 1   [NO GRADIENT]
    "x_order":     80.0,        # per control-point pair
    "hub_overlap": 500.0,
    "smoothness":  1.0,         # applied to the curvature-rate integral, see below
    "fold":        W.FOLD_PENALTY_SCALE,
    "arrival":     W.ARRIVAL_PENALTY_SCALE,
    "fillet":      3000.0,      # same class as fold/arrival: model validity, not taste
    "min_sj":      3000.0,
    "phase_ripple": 0.0,        # off by default; gate 10 reports what turning it on costs
}

# `smoothness` internals.  `400*n_infl` had literally zero gradient (an int from
# `count_nonzero` behind a data-dependent mask) and is replaced by an integral plus a
# smooth sign-reversal penalty.  `KAPPA_REF` is 1/mm and non-dimensionalises the reversal
# product; the curvature-rate integral is scaled to land in the same units as the old
# `120*turn_var` at the shipped design so the weight table stays readable.
KAPPA_REF = 0.05
SMOOTHNESS_CURVATURE_W = 4.0e3
SMOOTHNESS_REVERSAL_W = 60.0


def soft_barrier(violation, scale=1.0):
    """`wheel_fea.soft_barrier` in jnp.

    The original uses the Python builtin `max` (`wheel_fea.py:538`) and so cannot be
    traced.  Same function, C^1 at the knee — which is what an FD plateau needs, and why
    the barrier is quadratic rather than linear in the first place.
    """
    return scale * jnp.maximum(0.0, violation) ** 2


KT_CLAMP = (1.0, 3.5)
KT_EXPONENT = 0.65
KT_DEGENERATE_R_MM = 0.1


def stress_concentration_kt(fillet_radius_mm, thickness_mm, c_factor=1.0):
    """`wheel_fea.stress_concentration_kt` in jnp.  Kt = 1 + C*(t/2R)^0.65, clamped.

    The original (`wheel_fea.py:315-325`) is untraceable three ways — a data-dependent
    `if fillet_radius_mm < 0.1`, `np.clip`, and a `float()` cast — and `wheel_fea` must
    stay numpy-only, because `tests/test_import_hygiene.py` imports it in an interpreter
    with no jax (the CadQuery env has none).  So the twin lives here, for the same reason
    and by the same precedent as `soft_barrier` above.

    Verified bit-identical to the numpy original across a grid spanning both clamp bounds
    and the degenerate branch; see `tests/test_objective.py`.
    """
    r = jnp.asarray(fillet_radius_mm, dtype=float)
    degenerate = r < KT_DEGENERATE_R_MM
    # Double-`where`: the degenerate branch must not evaluate r=0 inside the power, or its
    # `inf` poisons the gradient of the branch that WAS taken.  `R_hub`/`R_rim` are bounded
    # below at 0.5 mm so the branch is unreachable in-box, but `kt_report` calls this with
    # `r_built = 0.0` and the twin has to agree there too.
    safe = jnp.where(degenerate, 1.0, r)
    kt = 1.0 + c_factor * (thickness_mm / (2.0 * safe)) ** KT_EXPONENT
    return jnp.where(degenerate, KT_CLAMP[1], jnp.clip(kt, *KT_CLAMP))


def _kt_hub(g):
    return stress_concentration_kt(g[12], g[8])       # R_hub with t0


def _kt_rim(g):
    return stress_concentration_kt(g[13], g[11])      # R_rim with t3


_KT_HUB_VG = jax.value_and_grad(_kt_hub)
_KT_RIM_VG = jax.value_and_grad(_kt_rim)


def junction_kt(genes):
    """`((Kt_hub, dKt_hub), (Kt_rim, dKt_rim))` — NO MESH, NO SOLVE.

    Pure gene space: one `jax.grad` on a 14-vector, which is why the stress constraint can
    afford an analytic concentration factor at all.  Ten of the fourteen entries of each
    gradient are exactly zero.

    The pairing — `R_hub` with `t0`, `R_rim` with `t3` — is not a choice made here.  It is
    asserted in three independent places already: `wheel_fea.py:494-495` (the beam model),
    `wheel_step_export.py:810-812` (the built-geometry report) and `tests/test_golden.py`.

    `jax.grad` rather than a hand-written derivative gets the clamp's zero-gradient region
    right for free, and the clamp IS reachable in-box (`t0=10.0` with `R_hub=0.5` gives
    `Kt = 5.47 -> 3.5`), though not at either design of interest.
    """
    g = jnp.asarray(genes, dtype=float)
    kt_h, d_h = _KT_HUB_VG(g)
    kt_r, d_r = _KT_RIM_VG(g)
    return ((float(kt_h), np.asarray(d_h, dtype=float)),
            (float(kt_r), np.asarray(d_r, dtype=float)))


# ---------------------------------------------------------------------------
# T1 — GENE SPACE.  No mesh, no solve.
# ---------------------------------------------------------------------------

def fillet_flanks(genes, cfg, span_mm=W.S, hub_radius=W.HUB_RADIUS_MM):
    """Which flanks actually cross their ring — a DISCRETE decision, taken in numpy.

    NOT A DETAIL, AND THE MEASUREMENT SAYS SO.  At the shipped genome the `eta=-1` flank
    NEVER CROSSES the hub circle: the spoke arrives on a shallow spiral, so on that side
    there is no reentrant corner and no fillet.  `wheel_fea.tangent_fillet_arc` reports
    exactly this by returning `None`, and its docstring calls it "a legitimate answer";
    `wheel_step_export.fillet_junctions` declines the same side.

    Requiring all four flanks feasible would therefore declare the SHIPPED DESIGN
    infeasible — measured margins `[-1.21, +4.64, +1.31, -5.87]` mm, two negative — and
    hand Stage 3 a 112,000-unit penalty on a wheel that has been printed.  The two
    negatives are not violations; they are the two sides that have nothing to round.

    `wheel_wheel.ring_station` catches this on the numpy path and raises, but its guard
    is `if xp is np and not changes.any()` — under tracing the check is skipped, the
    bracketing segment is degenerate and the Newton divides by zero.  The result is a
    silent NaN in twelve of fourteen gradient components rather than an error, which is
    why this is decided eagerly here and frozen, exactly like `owners` and `orientation`.
    """
    sample, s_dense = WW.global_sampler(genes, cfg, span_mm=span_mm,
                                        hub_radius=hub_radius, xp=np)
    rim_inner = WW.rim_inner_radius(span_mm=span_mm, hub_radius=hub_radius)
    out = []
    for radius in (hub_radius, rim_inner):
        etas = []
        for eta in (-1.0, 1.0):
            pts = sample(s_dense, np.zeros_like(s_dense) + eta)
            f = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2) - radius
            if bool(((f[:-1] * f[1:]) < 0).any()):
                etas.append(eta)
        out.append(tuple(etas))
    return tuple(out)


def _fillet_margins(genes, cfg, span_mm, hub_radius, flanks, xp=jnp):
    """Signed clearance [mm] for the fillet at each junction — hub, then rim.

    THE ONLY GRADIENT `R_hub` AND `R_rim` HAVE.  M7 proved both are dead at the mesh
    (`dcoords/dgene` identically zero, no solver involved) because no fillet is meshed.
    They are not dead in gene space, and this is where they live.

    `tangent_fillet_arc` refuses a radius at four places (`wheel_fea.py:948`, `:955`,
    `:959`, `:964`) and `fillet_junctions` then walks the radius ladder DOWN, so the
    optimizer silently receives a smaller fillet than it asked for and is never told —
    precisely the discrepancy `kt_report` carries as known-open.  The binding condition of
    the four is the discriminant: the centre must sit at distance R from the flank line
    AND at |c| = ring +/- R from the origin, and those loci meet only if the flank line
    passes within `Rc` of the origin.  Written as a signed distance rather than the raw
    discriminant so the margin is in mm and the barrier's scale means something:

        margin = Rc - dist(origin, flank line through Q = P + R * n_out)

    Positive is feasible.  Where a junction has two crossing flanks the margin is the
    BETTER of them: the fillet goes in the reentrant corner, and one comfortable side is
    enough to round it.  Where it has none there is no fillet to constrain, and the
    margin is +inf.  See `fillet_flanks` for why that case is real and common.
    """
    sample, s_dense = WW.global_sampler(genes, cfg, span_mm=span_mm,
                                        hub_radius=hub_radius, xp=xp)
    rim_inner = WW.rim_inner_radius(span_mm=span_mm, hub_radius=hub_radius)
    eps = 1e-4
    out = []
    for (radius, R, outside, near_end), etas in zip(
            ((hub_radius, genes[12], True, 0), (rim_inner, genes[13], False, 1)), flanks):
        if not etas:
            out.append(jnp.asarray(np.inf))
            continue
        Rc = radius + R if outside else radius - R
        per = []
        for eta in etas:
            s = WW.ring_station(sample, s_dense, radius, eta, near_end, xp=xp)
            P = sample(s, xp.asarray(eta))
            # Flank direction at the crossing, and the flank normal.  Both by a small
            # step rather than an analytic derivative because `sample` is the same
            # closure the mesh uses, and agreeing with the mesh matters more here than
            # two extra digits.
            step = eps if near_end == 0 else -eps
            d = sample(s + step, xp.asarray(eta)) - P
            d = d / xp.sqrt(d[0] ** 2 + d[1] ** 2)
            nv = P - sample(s, xp.asarray(eta * (1.0 - eps)))
            nv = nv / xp.sqrt(nv[0] ** 2 + nv[1] ** 2)
            # BOTH NORMAL ORIENTATIONS, BEST WINS.  `tangent_fillet_arc`'s caller passes
            # a specific `outward_normal`, and the convention flips between the two
            # junctions: at the hub the centre sits at ring+R, OUTSIDE the hub disk, so
            # the offset runs away from the origin; at the rim it sits at ring-R, inside
            # the rim band, so it runs toward it.  Measured at the shipped genome, taking
            # the hub's convention at the rim gives -5.87 mm and declares a wheel that
            # has been printed infeasible.  The question this term asks — does a circle
            # of radius R fit in this corner — does not depend on the convention, so it
            # is asked both ways.  The rim then comes out at +0.13 mm, which is the right
            # answer for a fillet radius pinned at its upper bound.
            b = xp.stack([(P + sgn * R * nv)[0] * d[0] + (P + sgn * R * nv)[1] * d[1]
                          for sgn in (1.0, -1.0)])
            QQ = xp.stack([xp.sum((P + sgn * R * nv) ** 2) for sgn in (1.0, -1.0)])
            per.append(xp.max(Rc - xp.sqrt(xp.maximum(QQ - b * b, 1e-30))))
        out.append(per[0] if len(per) == 1 else xp.maximum(*per))
    return xp.stack(out)


def t1_vector(genes, cfg="coarse", weights=None, span_mm=W.S, flanks=None):
    """The gene-space loss terms as a fixed-order jnp vector.

    A vector rather than a scalar so that `jax.jacrev` gives every term's gradient in one
    backward pass — which is what gate 8's per-term gradient-norm census needs, and what
    the master plan calls for: *"a term with a large value and a tiny gradient is inert in
    a gradient method even though it dominates the table"*.
    """
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    cfgo = WW.get_config(cfg)
    g = genes
    cx = [jnp.zeros(()), g[0], g[2], g[4], g[6], jnp.zeros(()) + span_mm]
    t0, R_hub = g[8], g[12]

    curve, ctrl = geom.bezier_centerline(g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7],
                                         span_mm, W.N_CURVE_PTS, xp=jnp)

    # -- x_order: control points must stay ordered with a 2 mm gap, and the curve must
    # not fold back in x.  The original's `if not curve_is_monotone_x` guard is dropped
    # rather than ported: the fold-back violation it feeds is `-min(diff(x))`, which is
    # already negative exactly when the guard is False, so `soft_barrier` returns zero
    # there anyway.  Same function, no branch.
    x_order = jnp.zeros(())
    for i in range(len(cx) - 1):
        x_order = x_order + soft_barrier(cx[i] + 2.0 - cx[i + 1], w["x_order"])
    x_order = x_order + soft_barrier(-jnp.min(jnp.diff(curve[:, 0])), 300.0)

    # -- hub_overlap, tightened by HUB_FREE_ARC_MM per the master plan.
    available = 2.0 * W.HUB_RADIUS_MM * np.sin(np.pi / W.NUMBER_OF_SPOKES)
    required = t0 + 2.0 * R_hub + W.HUB_CLEARANCE_MM + HUB_FREE_ARC_MM
    hub_overlap = soft_barrier(required - available, w["hub_overlap"])

    # -- fold: the offset band turning inside out.  Already a smooth signed margin.
    margin = geom.self_intersection_margin(curve, ctrl, g[8], g[9], g[10], g[11],
                                           num_points=W.N_CURVE_PTS, xp=jnp)
    fold = soft_barrier(geom.MIN_FOLD_MARGIN_MM - margin, w["fold"])

    # -- arrival: spoke meeting its ring too close to radially.
    a_hub, a_rim = WW.arrival_angles(genes, cfgo, span_mm=span_mm, xp=jnp)
    worst = jnp.maximum(a_hub, a_rim)
    arrival = soft_barrier((worst - WW.MAX_ARRIVAL_DEG) / 10.0, w["arrival"])

    # -- smoothness, REWRITTEN.  See the module docstring: the old `400*n_infl` had no
    # gradient at all.  Two smooth pieces instead: the curvature-rate integral penalises
    # wiggle, and the reversal product penalises curvature changing sign (a spiral is
    # wanted, an S is not) without counting anything.
    kappa = geom.bezier_curvature(ctrl, W.N_CURVE_PTS, xp=jnp)
    seg = geom.segment_lengths(curve, xp=jnp)
    dk = jnp.diff(kappa) / (seg + 1e-30)
    curvature_rate = jnp.sum(dk ** 2 * seg)
    reversal = jnp.sum(jnp.maximum(0.0, -kappa[:-1] * kappa[1:] / KAPPA_REF ** 2) * seg)
    smoothness = w["smoothness"] * (SMOOTHNESS_CURVATURE_W * curvature_rate
                                    + SMOOTHNESS_REVERSAL_W * reversal)

    # -- fillet feasibility.  The barrier is applied to the WORST of the four with the
    # safety factor, and to each individually so a single infeasible fillet cannot be
    # averaged away by three comfortable ones.
    marg = _fillet_margins(genes, cfgo, span_mm, W.HUB_RADIUS_MM, flanks)
    fillet = jnp.sum(soft_barrier(FILLET_SAFETY * MIN_FILLET_MARGIN_MM - marg,
                                  w["fillet"]))

    return jnp.stack([x_order, hub_overlap, smoothness, fold, arrival, fillet])


T1_NAMES = ("x_order", "hub_overlap", "smoothness", "fold", "arrival", "fillet")


# ---------------------------------------------------------------------------
# T2 — MESH SPACE.  Needs node coordinates; needs no solve.
# ---------------------------------------------------------------------------

def t2_vector(genes, mesh, weights=None):
    """`mass` and the mesh-validity barrier, from `mesh_coords` alone.

    Both reach the genes through `wheel_wheel.mesh_coords`, which freezes the topology —
    so this is the derivative of a fixed mesh whose nodes move, and a step that flips the
    flank orientation is a real discontinuity of the design space rather than a defect
    here.  See `mesh_coords`' docstring.

    `min_sj` is a barrier on EVERY element rather than on the worst one.  `min` over
    elements is a max-like kink whose gradient is a delta on whichever element currently
    happens to be worst, so it jumps as the argmin moves — the same objection that makes
    the stress term a p-norm.  Summing a barrier over all elements is smooth, and it also
    tells the optimizer how MANY elements are marginal, which the worst one cannot.
    """
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    coords = WW.mesh_coords(genes, mesh)

    area = WA._area_kernel(mesh.cfg.order)
    conn = jnp.asarray(mesh.conn)
    mass_g = jnp.sum(area(coords[conn])) * W.SPOKE_WIDTH_MM * W.DENSITY_PLA
    mass = w["mass"] * mass_g

    sj = wm.scaled_jacobian(coords, mesh.conn, xp=jnp)
    min_sj = jnp.sum(soft_barrier(MIN_SJ_TARGET - sj, w["min_sj"]))

    return jnp.stack([mass, min_sj]), mass_g


T2_NAMES = ("mass", "min_sj")


# ---------------------------------------------------------------------------
# PHASE
# ---------------------------------------------------------------------------

def phase_stencil(n_phase=8, n_sub=8, scheme="rqmc", rng=None):
    """The phase angles [deg] one objective evaluation is averaged over.

    `rqmc`     an `n_phase`-point uniform stencil shifted by a random offset drawn from
               the `n_sub`-point sub-lattice.  Unbiased and genuinely stochastic, but the
               union of all reachable phases is the fixed `n_phase * n_sub` grid, so
               `coord_fn`'s jit cache and any mesh cache hit after the first pass.  See
               the module docstring for what a continuous offset costs (roughly double).
    `uniform`  the same stencil with no shift.  Deterministic, cheapest, and the one that
               lets the rim's contact faceting alias into a chaseable bias.
    `iid`      independent uniform draws.  Present because the master plan asks for it to
               be settled empirically, and because it is the variance baseline the other
               two are measured against.  Not recommended: 8 i.i.d. samples of a smooth
               periodic integrand give about one digit where the stencil gives four.
    """
    cell = SECTOR_DEG / n_phase
    base = np.arange(n_phase) * cell
    if scheme == "uniform":
        return base
    if scheme == "rqmc":
        rng = np.random.default_rng() if rng is None else rng
        return base + int(rng.integers(n_sub)) * (cell / n_sub)
    if scheme == "iid":
        rng = np.random.default_rng() if rng is None else rng
        return np.sort(rng.random(n_phase) * SECTOR_DEG)
    raise ValueError(f"unknown phase scheme {scheme!r}; "
                     f"expected one of 'rqmc', 'uniform', 'iid'")


def phase_meshes(genes, cfg, phases, orientation=None):
    """One mesh per phase, rebuilt at the current genes.

    `orientation` is passed through so a caller can PIN the flank orientation across an
    optimizer step.  It is a discrete decision; letting it flip mid-trajectory changes
    which block owns which node, and the gradient of the step that flips it does not
    exist.  M8b pins it between checkpoints and reports a flip as an event.
    """
    return [WW.build_wheel(genes, cfg, phase_deg=float(p), orientation=orientation)
            for p in phases]


# ---------------------------------------------------------------------------
# T3 — SOLVE SPACE
# ---------------------------------------------------------------------------

def _stress_aggregate(pn, maxes, stress_phase_p):
    """`(agg, c)` from the per-phase p-norms and the per-phase true maxima.

    Factored out so the PROBE below and the constraint proper cannot drift apart: both
    build `agg` by calling this, so a probe at the constraint's own exponent reproduces the
    constraint's aggregate exactly and is checkable against the thing it is probing.

    `c = mean_phase(max / pnorm)` is now a DIAGNOSTIC ONLY and no longer enters any
    constraint.  M8b-i.6 step 1 is why: `c` chases M4's crack-tip singularity, which
    diverges under refinement (31.02 -> 41.54 -> 48.47 MPa, GCI 34.4%), so `c * agg`
    converges at no exponent at either design.  It survives because the `c` and `c GCI`
    columns of `make m8bi6` are the evidence for that finding, and the sweep must stay
    reproducible.
    """
    pn = np.asarray(pn, dtype=float)
    c = float(np.mean(np.asarray(maxes, dtype=float) / np.maximum(pn, 1e-12)))
    agg = float(np.sum(pn ** stress_phase_p) ** (1.0 / stress_phase_p))
    return agg, c


def _probe_values(prob, u, probe_p):
    """The p-norm of one converged field at each probe exponent, `{p: value}`.

    ONE IMPLEMENTATION, TWO CALLERS.  The serial path calls this with the field the
    adjoint just differentiated; `wheel_pool_worker` calls it in the child, because `prob`
    and `res` are the two things that cannot cross a process boundary.  Written out once
    so the pooled and serial probes cannot drift into two arithmetics that happen to agree
    on the day they were written.
    """
    return {float(v): float(WA._qoi_pnorm_stress(prob, p=float(v))(
        jnp.asarray(prob.coords), jnp.asarray(u), float(prob.contact.y_ground)))
        for v in probe_p}


def t3_terms(genes, cfg="coarse", *, phases=None, meshes=None, weights=None,
             force=SERVICE_FORCE_N, stress_phase_p=8.0, warm=None,
             stress_gauss_p=STRESS_NOMINAL_P, stress_p_probe=(), pool=None,
             orientation=None, **problem_kw):
    """`deflection`, `stress` and `phase_ripple`, phase-aggregated, with gradients.

    One `service_qoi_value_and_grad` per phase — which is one forward solve, one
    assembly, one factorisation and two back-solves, sharing the mesh vjp.  Every value
    it returns is a TOTAL derivative at constant service FORCE: the coupling through
    where the equilibrium moved is included, and it is not a correction.  Measured at the
    shipped genome, dropping it does not lengthen the stress gradient, it REVERSES it
    (+11.22 against a true -2.59 for `t2`).  See `service_qoi_value_and_grad`.

    Aggregation follows the master plan: the mean over the stencil for `deflection`,
    a smoothed max (p-norm, p=8) over the stencil nodes for `stress` — replacing CVaR,
    which at 8 samples is "the worst 1-2" and whose gradient jumps as the argmin switches
    — and std/mean for `phase_ripple`.

    TWO EXPONENTS, AND THEY ARE NOT THE SAME QUANTITY.  `stress_phase_p` aggregates the
    eight PHASE samples; `stress_gauss_p` is the exponent of the volume-weighted p-norm
    over the GAUSS POINTS inside one solve.  It defaults to `STRESS_NOMINAL_P` (4.0) and
    NOT to `wheel_adjoint.STRESS_PNORM_P` (30.0), because M8b-i.5 measured the p=30
    constraint to be NOT MESH-CONVERGENT (GCI 63% on a three-rung ladder whose axle drop
    converges to 0.14% off the same solves) and M8b-i.6 identified the exponent as the
    cause: at p=30 the p-norm is chasing a singular peak.  At p=4 it is a NOMINAL stress
    that converges (GCI 0.45% / 0.20%), and the peak is restored analytically by `Kt`
    below.  Reaching `_qoi_pnorm_stress` needs the `(name, factory)` form of
    `adjoint_grads`' `qois`; the bare string would take the module default and no argument
    could change it.

    `pool` RUNS THE PHASE LOOP IN PARALLEL AND CHANGES NOTHING ELSE.  Given a
    `wheel_pool.PhasePool`, each phase's mesh build, solve and adjoint happen in a worker
    process and only the leaves read below come back; `meshes` is then unused, because a
    built mesh cannot be pickled (`WheelMesh._coord_fn` holds a jitted function) and
    shipping one would cost more than rebuilding it.  `orientation` is needed only on that
    path, for the same reason `phase_meshes` takes it — the flank decision is pinned by
    the caller and the worker cannot re-derive it without risking a flip mid-step.

    The replies come back IN SLOT ORDER, so `drops`, `pn` and the rest are assembled in
    exactly the order the serial loop assembles them and every reduction below sums the
    same floats in the same sequence.  That is what lets S13 gate pooled == serial
    exactly rather than to a tolerance; see `wheel_pool`'s module docstring.

    `stress_p_probe` MEASURES WITHOUT SOLVING.  The p-norm is a pure function of the
    converged displacement field, so any number of exponents can be read off the field the
    adjoint already ran on — `_meta` hands back both `prob` and `res`.  Each probe `p` is
    pushed through `_stress_aggregate`, the same arithmetic the constraint uses, so
    `pnorm_by_p[stress_gauss_p]` reproduces the report's own `stress_utilisation` exactly
    and the probe is checkable against the thing it is probing.  NO GRADIENT is taken at a
    probe `p`: a convergence study reads values, and one adjoint per exponent per phase
    would turn a 14-minute ladder into an hour for numbers nothing reads.
    """
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if phases is None:
        phases = phase_stencil(scheme="uniform")
    if meshes is None and pool is None:
        meshes = phase_meshes(genes, cfg, phases, orientation=orientation)

    probe_p = [float(v) for v in stress_p_probe]
    # The name stays "pnorm_stress" so every downstream key is unchanged; only the factory
    # differs from `QOI["pnorm_stress"]`, and only in `p`.
    qoi = ("pnorm_stress", lambda prob: WA._qoi_pnorm_stress(prob, p=stress_gauss_p))

    drops, dgrads, pn, pgrads, maxes, rows = [], [], [], [], [], []
    probe_pn = {v: [] for v in probe_p}
    pooled = None if pool is None else pool.map_phases([
        {"genes": np.asarray(genes, dtype=float), "cfg": cfg, "phase": float(p),
         "orientation": orientation, "force": force,
         "delta0": None if warm is None else warm[i],
         "stress_gauss_p": stress_gauss_p, "probe_p": tuple(probe_p),
         "problem_kw": problem_kw}
        for i, p in enumerate(phases)])

    for i, p in enumerate(phases):
        if pooled is None:
            o = WA.service_qoi_value_and_grad(
                genes, cfg, (qoi,), force=force, mesh=meshes[i],
                delta0=None if warm is None else warm[i], **problem_kw)
            # The same field the adjoint above differentiated, at a different exponent.
            probes = _probe_values(o["_meta"]["prob"], o["_meta"]["res"]["u"], probe_p)
        else:
            o, probes = pooled[i], pooled[i]["_probe"]
        for v in probe_p:
            probe_pn[v].append(probes[v])
        drops.append(o["axle_drop"]["value"])
        dgrads.append(o["axle_drop"]["grad"])
        pn.append(o["pnorm_stress"]["value"])
        pgrads.append(o["pnorm_stress"]["grad"])
        maxes.append(o["_meta"]["max_stress_mpa"])
        rows.append({"phase_deg": float(p), "axle_drop_mm": float(o["axle_drop"]["value"]),
                     "pnorm_stress_mpa": float(o["pnorm_stress"]["value"]),
                     "max_stress_mpa": float(o["_meta"]["max_stress_mpa"]),
                     "contact_force_n": float(o["_meta"]["contact_force_n"]),
                     "coupling_frac": float(
                         np.linalg.norm(o["pnorm_stress"]["coupling_grad"])
                         / max(np.linalg.norm(o["pnorm_stress"]["grad"]), 1e-30))})

    drops = np.asarray(drops)
    dgrads = np.asarray(dgrads)
    pn = np.asarray(pn)
    pgrads = np.asarray(pgrads)
    n = len(drops)

    # -- deflection: mean drop against the target, two-sided.  For a compliant mechanism
    # the travel IS the feature, so being too stiff is penalised exactly as hard as being
    # too soft — `wheel_fea`'s reasoning, unchanged.
    mean_drop = drops.mean()
    mean_dgrad = dgrads.mean(axis=0)
    err = (mean_drop - TARGET_DEFLECTION_MM) / TARGET_DEFLECTION_MM
    deflection = w["deflection"] * err ** 2
    d_deflection = w["deflection"] * 2.0 * err / TARGET_DEFLECTION_MM * mean_dgrad

    # -- stress: `Kt(R, t) * sigma_nominal <= ALLOWABLE`, one barrier per junction.
    #
    # WHAT THIS REPLACED, AND WHY IT HAD TO GO.  The constraint used to be
    # `c * agg / ALLOWABLE` with `c = mean_phase(max / pnorm)` — a MEASURED rescale from the
    # p-norm up to the true max.  It had a known gradient hazard, handled by making `c` an
    # explicit input the caller froze within a step: the rescale is exact only for a
    # CONSTANT factor, since the p-norm is positively homogeneous of degree 1 and
    # `d(c*pnorm) = c*d(pnorm)` holds if and only if `c` did not move.  (Recomputing it
    # inside a finite difference once put a 10% error into the assembled gradient while
    # every individual term still agreed with its own FD to 1e-8 — how an assembly bug
    # hides.)
    #
    # The freeze was sound and the quantity was not.  M8b-i.6 step 1 swept ten exponents off
    # one set of solves and found the two factors converge in DISJOINT ranges of `p`: the
    # p-norm for `p <= 6`, `c` only for `p >= 24`, and their product at NO exponent at
    # either design.  The reason is structural — `c` is anchored to the true max, which is
    # M4's crack-tip singularity and diverges under refinement — so no amount of freezing
    # fixes it.  A frozen divergent number is still a divergent number.
    #
    # So the peak is no longer measured, it is MODELLED: a converged nominal stress at
    # `STRESS_NOMINAL_P = 4.0` times an analytic, differentiable stress-concentration
    # factor.  `Kt` is a function of the genes, so it is not hoisted or frozen — it is
    # DIFFERENTIATED, and the product rule below is the whole change.
    #
    # TWO JUNCTIONS, TWO BARRIERS, not one aggregated `Kt`.  A hard `max` over the two
    # would zero the gradient of whichever junction is not currently worst and reintroduce
    # exactly the argmax kink the p-norm exists to blur; a p-norm over two samples is an
    # arbitrary exponent for no smoothing benefit.  Summing two `soft_barrier`s is smooth
    # everywhere and lets `R_hub` and `R_rim` both carry a live gradient at once.
    q = stress_phase_p
    agg, c = _stress_aggregate(pn, maxes, q)
    denom = np.sum(pn ** q)
    dagg = (pn ** (q - 1.0))[:, None] * pgrads
    dagg = dagg.sum(axis=0) * denom ** (1.0 / q - 1.0)

    (kt_hub, dkt_hub), (kt_rim, dkt_rim) = junction_kt(genes)
    stress, d_stress, utils = 0.0, np.zeros_like(dagg), {}
    for name, kt, dkt in (("hub", kt_hub, dkt_hub), ("rim", kt_rim, dkt_rim)):
        util_j = kt * agg / ALLOWABLE_STRESS_MPA
        d_util = (dkt * agg + kt * dagg) / ALLOWABLE_STRESS_MPA   # THE PRODUCT RULE
        stress += float(soft_barrier(util_j - 1.0, w["stress"]))
        d_stress = d_stress + 2.0 * w["stress"] * max(0.0, util_j - 1.0) * d_util
        utils[name] = float(util_j)
    util = max(utils.values())          # the headline scalar — REPORTING ONLY

    # -- phase_ripple: std/mean of the drop over the stencil.  A design quantity, not a
    # quadrature nuisance — M6 measured 7.0% at the shipped genome.
    sd = drops.std()
    ripple = sd / mean_drop
    if sd > 0:
        dsd = ((drops - mean_drop)[:, None] * dgrads).sum(axis=0) / (n * sd)
    else:
        dsd = np.zeros_like(mean_dgrad)
    d_ripple = (dsd * mean_drop - sd * mean_dgrad) / mean_drop ** 2
    phase_ripple = w["phase_ripple"] * ripple ** 2
    d_phase_ripple = w["phase_ripple"] * 2.0 * ripple * d_ripple

    # -- the probe.  `c` is re-measured at every exponent, because the point of the sweep is
    # how `c = mean_phase(max/pnorm_p)` moves WITH `p` and with the mesh — that measurement
    # IS M8b-i.6 step 1's finding, and the three original keys are kept unchanged so
    # `make m8bi6` reproduces its `c` / `c GCI` columns bit-identically.
    #
    # `stress_utilisation` here is now a DIFFERENT quantity from the constraint's: it is the
    # old `c * agg / ALLOWABLE`, retained as the diagnostic that shows why it was abandoned.
    # `stress_utilisation_kt` is the new one, and it is the key that reproduces the report's
    # own `stress_utilisation` when probed at `stress_gauss_p`.
    by_p = {}
    kt_max = max(kt_hub, kt_rim)
    for v in probe_p:
        a_v, c_v = _stress_aggregate(probe_pn[v], maxes, q)
        by_p[repr(v)] = {"p": v, "pnorm_agg_mpa": a_v, "stress_scale_measured": c_v,
                         "stress_utilisation": c_v * a_v / ALLOWABLE_STRESS_MPA,
                         "stress_utilisation_kt": kt_max * a_v / ALLOWABLE_STRESS_MPA,
                         "pnorm_stress_mpa": [float(x) for x in probe_pn[v]]}

    return {
        "values": {"deflection": float(deflection), "stress": float(stress),
                   "phase_ripple": float(phase_ripple)},
        "grads": {"deflection": d_deflection, "stress": d_stress,
                  "phase_ripple": d_phase_ripple},
        "report": {"axle_drop_mean_mm": float(mean_drop),
                   "axle_drop_min_mm": float(drops.min()),
                   "axle_drop_max_mm": float(drops.max()),
                   "phase_ripple_std_over_mean": float(ripple),
                   "pnorm_stress_agg_mpa": float(agg),
                   "max_stress_mpa": float(np.max(maxes)),
                   "stress_scale_measured": c,     # diagnostic only, see `_stress_aggregate`
                   "kt_hub": float(kt_hub), "kt_rim": float(kt_rim),
                   "stress_utilisation_hub": utils["hub"],
                   "stress_utilisation_rim": utils["rim"],
                   "stress_utilisation": float(util),
                   "stress_gauss_p": float(stress_gauss_p), "pnorm_by_p": by_p,
                   "n_phase": n, "rows": rows},
    }


# ---------------------------------------------------------------------------
# THE OBJECTIVE
# ---------------------------------------------------------------------------

def objective(genes, cfg="coarse", *, weights=None, phases=None, meshes=None,
              normalized=False, force=SERVICE_FORCE_N, tiers=("t1", "t2", "t3"),
              span_mm=W.S, pool=None, orientation=None, **problem_kw):
    """`(value, grad, breakdown)` — the scalar Stage 3 descends, and its gradient.

    `normalized=True` takes and returns unit-box `z` instead of physical genes.  The
    chain rule `dL/dz = dL/dgenes * (high - low)` lives here and only here: `cy` spans 64
    mm and `R_rim` spans 2.5, so a single learning rate in physical units is meaningless,
    and the bounds are `wheel_fea.GENE_SPACE` read through `wheel_genome.bounds_arrays`.

    `tiers` exists so a line search can re-evaluate the microsecond tiers alone.

    `pool` and `orientation` are NAMED rather than left to ride `**problem_kw`, and that
    is not tidiness: `problem_kw` is splatted into `service_qoi_value_and_grad`, which
    would raise on either of them.  T2 still needs a real mesh, so a pooled caller is
    expected to hand `meshes=[mesh0]` — one mesh, built at the first phase — rather than
    let T2 fall back to `build_wheel` at the default phase and quietly stop matching the
    serial path.  `wheel_stage3.Evaluator` does exactly that.
    """
    low, high, rng = wg.bounds_arrays(W.GENE_SPACE)
    genes = wg.denormalize(np.asarray(genes, dtype=float), low, high) if normalized \
        else np.asarray(genes, dtype=float)
    gj = jnp.asarray(genes)

    values, grads, report = {}, {}, {}

    if "t1" in tiers:
        # Frozen once per evaluation, like the mesh topology — see `fillet_flanks`.
        flanks = fillet_flanks(genes, WW.get_config(cfg), span_mm)
        v1 = np.asarray(t1_vector(gj, cfg, weights, span_mm, flanks))
        j1 = np.asarray(jax.jacrev(t1_vector)(gj, cfg, weights, span_mm, flanks))
        report["fillet_flanks"] = [list(e) for e in flanks]
        for k, name in enumerate(T1_NAMES):
            values[name], grads[name] = float(v1[k]), j1[k]
        # The one term with no gradient, and it is asserted rather than assumed — see the
        # module docstring.  Computed from the numpy beam surrogate.
        _, _, _, ratio = _buckling_ratio(genes)
        wts = dict(DEFAULT_WEIGHTS if weights is None else weights)
        values["buckling"] = float(soft_barrier(ratio - 1.0, wts["buckling"]))
        grads["buckling"] = np.zeros(14)
        report["buckling_ratio"] = float(ratio)

    if "t2" in tiers:
        mesh0 = (meshes[0] if meshes else WW.build_wheel(genes, cfg))
        v2, mass_g = t2_vector(gj, mesh0, weights)
        j2 = np.asarray(jax.jacrev(lambda v: t2_vector(v, mesh0, weights)[0])(gj))
        v2 = np.asarray(v2)
        for k, name in enumerate(T2_NAMES):
            values[name], grads[name] = float(v2[k]), j2[k]
        report["mesh_mass_g"] = float(mass_g)
        report["min_scaled_jacobian"] = float(
            np.min(wm.scaled_jacobian(np.asarray(WW.mesh_coords(gj, mesh0)), mesh0.conn)))

    if "t3" in tiers:
        t3 = t3_terms(genes, cfg, phases=phases, meshes=meshes, weights=weights,
                      force=force, pool=pool, orientation=orientation, **problem_kw)
        values.update(t3["values"])
        grads.update(t3["grads"])
        report.update(t3["report"])

    total = float(sum(values.values()))
    gsum = np.sum([grads[k] for k in values], axis=0)
    if normalized:
        gsum = gsum * rng
        grads = {k: v * rng for k, v in grads.items()}

    breakdown = {"total": total, "terms": {}, "report": report}
    gnorms = {k: float(np.linalg.norm(grads[k])) for k in values}
    gtot = sum(gnorms.values()) or 1.0
    for k in values:
        breakdown["terms"][k] = {
            "value": values[k], "share": values[k] / (total or 1.0),
            "grad_norm": gnorms[k], "grad_share": gnorms[k] / gtot}
    return total, gsum, breakdown


def _buckling_ratio(genes):
    """The legacy Euler equivalent-column check.  NO GRADIENT — see the module docstring.

    Delegates to `wheel_fea.generalized_spoke_mechanics` rather than reimplementing it:
    the value is the one M0's golden test pins, and a second copy that drifted would be a
    silent change to a committed number.
    """
    curve, _ = W.generate_bezier_centerline(*[float(genes[i]) for i in range(8)])
    return W.generalized_spoke_mechanics(
        curve, float(genes[8]), float(genes[9]), float(genes[10]), float(genes[11]),
        W.SPOKE_WIDTH_MM, W.FORCE_PER_SPOKE_NEWTONS,
        R_hub_fillet=float(genes[12]), R_rim_fillet=float(genes[13]))


def insensitive_terms(breakdown, value_share=0.05, grad_abs=1e-9):
    """Terms that matter to the loss and CANNOT BE REDUCED — gradient identically zero.

    The master plan names this as a failure mode the GA did not have: *"a term with a
    large value and a tiny gradient is inert in a gradient method even though it dominates
    the table"*.  `400*n_infl` was exactly that — an integer from `count_nonzero`, 20.6%
    of the GA's loss, derivative zero everywhere — and it is the case this catches.

    ZERO IS THE CRITERION, NOT SMALL, and the difference is not pedantry.  A term with a
    zero gradient can never be reduced by any gradient method at any weight: it is a
    constant the optimizer cannot see.  A term with a small-but-exact gradient is merely
    DOMINATED, which is a statement about the other terms and about where the design
    currently sits, not about whether this one works.  `dominated_terms` reports that
    separately, because it is calibration information rather than a defect.
    """
    return sorted(k for k, d in breakdown["terms"].items()
                  if d["share"] > value_share and d["grad_norm"] <= grad_abs)


def dominated_terms(breakdown, value_share=0.05, grad_share=0.005):
    """Terms worth reading the table for: material value, small share of the gradient.

    Not a failure — see `insensitive_terms`.  Reported so a weight is never tuned on the
    value column alone, which is the master plan's actual instruction.
    """
    return sorted(k for k, d in breakdown["terms"].items()
                  if d["share"] > value_share and d["grad_share"] < grad_share
                  and d["grad_norm"] > 0.0)


def print_loss_breakdown(breakdown, title="STAGE-3 OBJECTIVE"):
    """`wheel_fea.print_loss_breakdown` plus the column that changes the conclusion.

    The gradient-norm share is the addition.  Reading value shares alone is how a weight
    gets tuned to a term that a gradient method cannot act on at all.
    """
    print(f"\n{title}\n" + "-" * len(title))
    print(f"    {'term':<14s}{'value':>12s}{'share':>9s}"
          f"{'|grad|':>13s}{'grad share':>12s}")
    rows = sorted(breakdown["terms"].items(), key=lambda kv: -kv[1]["value"])
    for name, d in rows:
        flag = ""
        if d["share"] > 0.05 and d["grad_norm"] <= 1e-9:
            flag = "  <- INERT (zero gradient)"
        elif d["share"] > 0.05 and d["grad_share"] < 0.005:
            flag = "  <- dominated"
        print(f"    {name:<14s}{d['value']:12.4f}{100*d['share']:8.1f}%"
              f"{d['grad_norm']:13.4g}{100*d['grad_share']:11.1f}%{flag}")
    print(f"    {'TOTAL':<14s}{breakdown['total']:12.4f}")
    r = breakdown.get("report", {})
    if r:
        print()
        for k in ("axle_drop_mean_mm", "phase_ripple_std_over_mean", "max_stress_mpa",
                  "stress_utilisation", "mesh_mass_g", "min_scaled_jacobian",
                  "buckling_ratio"):
            if k in r:
                print(f"    {k:<32s}{r[k]:12.5f}")
