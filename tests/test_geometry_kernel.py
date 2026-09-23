"""The geometry kernel: backend agreement, equivalence to the originals, gradients.

`wheel_geometry` is the single implementation that both the numpy side (the GA, the
CadQuery exporter) and the JAX side (the mesh, the differentiable FEA) will use.  Three
classes of thing have to hold, and each has its own tolerance decided in advance:

  equivalence  the branch-free / cached rewrites must reproduce the functions they
               replaced, or the STEP on disk stops describing the design      1e-15
  backends     numpy and jax.numpy must agree, or the FEA optimises a different
               wheel from the one that gets exported                          1e-12
  gradients    reverse-mode AD must match central finite differences, or every
               gradient-based stage downstream is descending on noise          1e-6
"""

import json
import math
import os

import numpy as np
import pytest

import wheel_fea as W
import wheel_geometry as G

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOL_EQUIV = 1e-15
TOL_BACKEND = 1e-12
TOL_GRAD = 1e-6

SPAN, NPTS = W.HUB_RIM_SPAN_MM, W.N_CURVE_PTS


@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(HERE, "best_solution.json")) as fh:
        return json.load(fh)["genes"]


@pytest.fixture(scope="module")
def g8(genes):
    return [genes[k] for k in ("cx1", "cy1", "cx2", "cy2", "cx3", "cy3", "cx4", "cy4")]


@pytest.fixture(scope="module")
def ts(genes):
    return [genes[k] for k in ("t0", "t1", "t2", "t3")]


@pytest.fixture(scope="module")
def curve(g8):
    return G.bezier_centerline(*g8, span_mm=SPAN, num_points=NPTS)[0]


def _rng_genomes(n, seed=0):
    """Random genomes spanning the whole box, so equivalence is tested across the
    design space rather than at the one point that happens to be on disk."""
    rng = np.random.default_rng(seed)
    low, high, _ = _bounds()
    return rng.uniform(low, high, size=(n, len(low)))


def _bounds():
    import wheel_genome as GN
    return GN.bounds_arrays(W.GENE_SPACE)


# ---------------------------------------------------------------------------
# EQUIVALENCE TO THE FUNCTIONS THIS KERNEL REPLACED
# ---------------------------------------------------------------------------

def test_bernstein_matrix_is_cached_not_recomputed():
    """The basis does not depend on the genes, so it must be memoised — it used to be
    rebuilt on every fitness evaluation."""
    a = G.bernstein_matrix(NPTS)
    b = G.bernstein_matrix(NPTS)
    assert a is b, "bernstein_matrix is not returning a cached object"


def test_bernstein_partition_of_unity():
    """Every Bezier basis sums to 1 at every parameter value — the property that makes
    the curve lie in the convex hull of its control points."""
    B = G.bernstein_matrix(NPTS)
    assert np.abs(B.sum(axis=1) - 1.0).max() < 1e-14


def test_centerline_endpoints_are_locked(curve):
    """P0 at the hub origin and P5 at (span, 0) are not genes."""
    assert np.allclose(curve[0], [0.0, 0.0], atol=1e-12)
    assert np.allclose(curve[-1], [SPAN, 0.0], atol=1e-12)


@pytest.mark.parametrize("i", range(6))
def test_thickness_is_the_exact_linear_interpolant(i):
    """Branch-free ramps vs `np.interp`, which is an independent implementation of the
    same piecewise-linear interpolation.

    This is the rewrite most likely to be subtly wrong: the two forms are algebraically
    identical but associate their floating-point operations differently, and the zone
    widths are not even all equal in binary (2/3 - 1/3 != 1 - 2/3).  Checked on a dense
    grid that includes every breakpoint exactly.
    """
    rng = np.random.default_rng(i)
    s = np.concatenate([
        np.linspace(0.0, 1.0, 20000),
        G.TAPER_BREAKPOINTS,                       # exactly on the zone boundaries
        rng.uniform(0.0, 1.0, 10000),
    ])
    t = rng.uniform(2.0, 10.0, 4)
    expected = np.interp(s, G.TAPER_BREAKPOINTS, t)
    got = G.thickness_at_arc_length(s, *t)
    # Relative, because thickness ranges over 2-10 mm and one ULP of 10.0 is 1.8e-15 —
    # an absolute bound would be testing the magnitude of the numbers, not the code.
    # Observed: exactly 1 ULP (2.1e-16 relative).
    assert np.abs((got - expected) / expected).max() < TOL_EQUIV


@pytest.mark.parametrize("i", range(4))
def test_thickness_tracks_the_superseded_masked_form(i):
    """...and still agrees with the boolean-mask version it replaced, to 1e-11.

    Not tighter, and deliberately so.  The old form divided by
    `bp[i+1] - bp[i] + 1e-12`, a guard on a constant that cannot be zero, which biased
    every zone interior by ~2.4e-12 relative.  The rewrite drops it.  1e-11 is the size
    of that known, understood difference; anything larger would be a new bug.
    """
    rng = np.random.default_rng(100 + i)
    s = np.linspace(0.0, 1.0, 20000)
    t = rng.uniform(2.0, 10.0, 4)

    bp, nodes = G.TAPER_BREAKPOINTS, np.array(t)
    old = np.zeros_like(s)
    for z in range(3):
        mask = (s >= bp[z]) & (s <= bp[z + 1])
        alpha = (s[mask] - bp[z]) / (bp[z + 1] - bp[z] + 1e-12)
        old[mask] = nodes[z] * (1 - alpha) + nodes[z + 1] * alpha

    assert np.abs((G.thickness_at_arc_length(s, *t) - old) / old).max() < 1e-11


def test_thickness_hits_its_nodes_exactly(ts):
    """t(0)=t0, t(1/3)=t1, t(2/3)=t2, t(1)=t3 — the defining property of an
    interpolant, and the one the superseded version got wrong by 1.2e-11.

    THE BOUND WAS `== 0.0` UNTIL 2026-09-08, AND THAT WAS NEVER THIS CLAIM — it was a
    rounding accident that held for the genomes this file happened to be run on.  The
    function is a base value plus three clipped ramps, so `t(bp_k)` is `t0` plus a
    TELESCOPING SUM of k differences; the sum is exact only when its roundings cancel.
    Measured over 20000 uniform draws of `(t0, t1, t2, t3)` across `GENE_SPACE`'s own
    bounds:

        exactly 0.0 ....................  80.09% of draws, NOT 100%
        worst error ....................  1.332268e-15  =  6 ULP
        per node, share not exact ......  bp=0: 0.00%   1/3: 7.89%   2/3: 10.10%   1: 13.49%

    The per-node profile IS the mechanism: `t(0)` is `t0` with no sum and never misses,
    and the miss rate climbs with the number of terms.  So `== 0.0` was a coin this test
    had won on every genome until `b729e86`, which reads 2.220446e-16 — ONE ULP, on `t1`
    and `t2` — and lost it.

    §133 §GROUP B's stated reason SURVIVES ITS CONTROL.  It said the exactness "survives
    only while the t-vector's magnitudes are small".  Drawing all four thicknesses from
    [1.2, 2.0] instead of the full box gives 20000 of 20000 exact, against 80.09% — so
    magnitude is the driver, and `t0` going 1.4738 -> 3.5055 is why this one broke now.

    THE BOUND IS THE GEOMETRIC MIDPOINT OF THE TWO THINGS IT MUST SEPARATE, which is the
    warrant `== 0.0` never had: 75x above the measured worst case over the whole box, and
    120x below the 1.2e-11 defect this test exists to catch.  The `ts` fixture stays on
    the shipped genome deliberately — the claim is about the INTERPOLANT and the bound is
    genome-independent by the sweep above, so reading what ships costs nothing and keeps
    the test a live check that the shipped design is inside that regime.
    """
    got = G.thickness_at_arc_length(G.TAPER_BREAKPOINTS, *ts)
    assert np.abs(got - np.array(ts)).max() < 1e-13


def test_thickness_is_monotone_between_nodes(ts):
    """Piecewise LINEAR: within a zone the value must stay between its two nodes."""
    s = np.linspace(0, 1, 5000)
    got = G.thickness_at_arc_length(s, *ts)
    assert got.min() >= min(ts) - 1e-12
    assert got.max() <= max(ts) + 1e-12


def test_offset_band_at_one_reproduces_thicken_3taper(curve, ts):
    """n_across=1 must give back exactly the outline the exporter builds splines
    through (wheel_step_export.py:216)."""
    top, bot = G.outline_edges(curve, *ts)
    band = G.offset_band(curve, *ts, n_across=1)
    assert np.abs(band[:, 1, :] - top).max() == 0.0
    assert np.abs(band[:, 0, :] - bot).max() == 0.0


def test_offset_band_centre_column_is_the_centerline(curve, ts):
    """With an even n_across there is a column at eta=0, which must be the centerline
    itself — the property the mesh relies on to tie the two flanks together."""
    band = G.offset_band(curve, *ts, n_across=6)
    assert np.abs(band[:, 3, :] - curve).max() < 1e-13


def test_offset_band_columns_are_evenly_spaced(curve, ts):
    """Uniform eta must give uniform through-thickness spacing, or the mesh is graded
    where nobody asked it to be."""
    band = G.offset_band(curve, *ts, n_across=8)
    step = np.linalg.norm(np.diff(band, axis=1), axis=2)
    assert np.abs(step - step[:, :1]).max() < 1e-12


def test_outline_polygon_is_closed_loop(curve, ts):
    poly = G.outline_polygon(curve, *ts)
    assert poly.shape == (2 * len(curve), 2)


@pytest.mark.parametrize("vec", _rng_genomes(8))
def test_wheel_fea_wrappers_agree_with_kernel(vec):
    """wheel_fea's public geometry surface still produces what the kernel does, across
    the design space — this is what the STEP exporter actually calls."""
    g8, ts = list(vec[:8]), list(vec[8:12])
    c_fea, p_fea = W.generate_bezier_centerline(*g8)
    c_ker, p_ker = G.bezier_centerline(*g8, span_mm=SPAN, num_points=NPTS)
    assert np.abs(c_fea - c_ker).max() == 0.0
    assert np.abs(p_fea - p_ker).max() == 0.0

    top_f, bot_f = W.thicken_3taper_curve(c_fea, *ts, return_edges=True)
    top_k, bot_k = G.outline_edges(c_ker, *ts)
    assert np.abs(top_f - top_k).max() == 0.0
    assert np.abs(bot_f - bot_k).max() == 0.0


# ---------------------------------------------------------------------------
# BACKEND AGREEMENT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vec", _rng_genomes(6, seed=1))
def test_numpy_and_jax_agree(vec):
    """Every kernel function, both backends, on random genomes.

    If these drift, the FEA optimises a wheel that is not the one the exporter builds —
    the exact class of silent disagreement the shared kernel exists to prevent.
    """
    g8, ts = list(vec[:8]), list(vec[8:12])

    cn, pn = G.bezier_centerline(*g8, span_mm=SPAN, num_points=NPTS, xp=np)
    cj, pj = G.bezier_centerline(*g8, span_mm=SPAN, num_points=NPTS, xp=jnp)
    assert np.abs(cn - np.asarray(cj)).max() < TOL_BACKEND
    assert np.abs(pn - np.asarray(pj)).max() < TOL_BACKEND

    for at in ("nodes", "midpoints"):
        sn = G.arc_fractions(cn, xp=np, at=at)
        sj = G.arc_fractions(cj, xp=jnp, at=at)
        assert np.abs(sn - np.asarray(sj)).max() < TOL_BACKEND

    sn = G.arc_fractions(cn, xp=np)
    assert np.abs(G.thickness_at_arc_length(sn, *ts, xp=np)
                  - np.asarray(G.thickness_at_arc_length(sn, *ts, xp=jnp))
                  ).max() < TOL_BACKEND
    assert np.abs(G.offset_normals(cn, xp=np)
                  - np.asarray(G.offset_normals(cj, xp=jnp))).max() < TOL_BACKEND
    assert np.abs(G.offset_band(cn, *ts, n_across=5, xp=np)
                  - np.asarray(G.offset_band(cj, *ts, n_across=5, xp=jnp))
                  ).max() < TOL_BACKEND
    assert np.abs(G.bezier_curvature(pn, NPTS, xp=np)
                  - np.asarray(G.bezier_curvature(pj, NPTS, xp=jnp))
                  ).max() < TOL_BACKEND
    assert np.abs(G.bezier_tangent(pn, NPTS, xp=np)
                  - np.asarray(G.bezier_tangent(pj, NPTS, xp=jnp))
                  ).max() < TOL_BACKEND


def test_analytic_curvature_matches_finite_differences(g8):
    """The hodograph curvature is independent code from differencing the samples, so
    agreeing to the FD scheme's own O(h^2) error validates both.

    THAT CLAIM WAS NEVER WRONG.  THE METRIC COULD NOT SEE IT, AND `b729e86` IS THE FIRST
    GENOME WHERE THAT SHOWS.  The error was normalised POINTWISE, by `k_an` at each
    sample — undefined where the curvature passes through zero, which is what an
    INFLECTION POINT is.  The shipped centerline has one and the outgoing centerline does
    not:

        genome            sign flip   min|k_an|   max|k_an|   pointwise    scaled
        b729e86 SHIPS        yes      1.584e-04   1.041e-01   1.956e-03   1.446e-05
        96a0ac5 outgoing     no       3.868e-02   6.623e-02   7.104e-06   6.170e-06

    The pointwise max is attained AT the smallest |k_an| on every refinement, so it does
    not measure truncation error, it measures how near a sample happens to land to the
    zero crossing — and it does not converge: 1.956e-03, 9.053e-03, 9.485e-04, 2.270e-04
    over n = 600, 1200, 2400, 4800, an "order" of -2.21, 3.25, 2.06.

    NORMALISED BY THE CURVE'S OWN SCALE INSTEAD, BOTH GENOMES ARE EXACTLY SECOND ORDER:

        b729e86    1.446e-05  3.608e-06  9.014e-07  2.257e-07     order 2.00 2.00 2.00
        96a0ac5    6.170e-06  1.540e-06  3.846e-07  9.613e-08     order 2.00 2.00 2.00

    THE OLD `1e-4` WAS NOT GENOME-INDEPENDENT AND NOTHING RECORDED THAT.  Over 400
    uniform draws across the eight centerline genes at n = 600, the pointwise metric put
    **1.8%** under 1e-4, spanning 337262x from 2.883e-05 to 9.725 — a 970% "relative
    error" on a curve whose curvature simply crosses zero.  The outgoing genome was one
    of the 1.8%.

    THE NEW BOUND HAS A WARRANT.  The same 400 draws under the scaled metric span **23x**
    — min 1.147e-05, median 5.456e-05, p99 1.946e-04, max 2.588e-04 — and 400 of 400 sit
    under 1e-3, the worst by 3.9x.  The order itself is deliberately NOT asserted: its
    median over those draws is 2.00 but 2.5% land outside [1.8, 2.2], and a pin that is
    97.5% reliable across the box is the kind of thing this file is currently repairing.
    """
    c, p = G.bezier_centerline(*g8, span_mm=SPAN, num_points=NPTS)
    k_an = G.bezier_curvature(p, NPTS)
    d1 = np.gradient(c, axis=0)
    d2 = np.gradient(d1, axis=0)
    k_fd = ((d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
            / (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5)
    interior = slice(5, -5)          # np.gradient is one-sided at the ends
    k_an, k_fd = k_an[interior], k_fd[interior]
    # Scaled by max|k_an|, NOT pointwise: see the docstring.  Pointwise is a division by
    # zero anywhere the curve has an inflection, and the shipped one does.
    assert np.abs(k_an - k_fd).max() / np.abs(k_an).max() < 1e-3


def test_analytic_tangent_is_unit_length(g8):
    _, p = G.bezier_centerline(*g8, span_mm=SPAN, num_points=NPTS)
    tan = G.bezier_tangent(p, NPTS)
    assert np.abs(np.linalg.norm(tan, axis=1) - 1.0).max() < 1e-12


def test_normals_are_unit_and_perpendicular(curve):
    """Unit to 1e-10, perpendicular to machine precision.

    The looser bound on the magnitude is understood, not slop: `offset_normals`
    normalises by `||n|| + 1e-12`, and the raw gradient magnitude here is the
    inter-sample spacing of ~0.06 mm, so the guard costs 1e-12/0.06 = 1.6e-11 of
    relative length.  Unlike the taper's epsilon this one is load-bearing — a genome
    with coincident curve points really does give ||n|| = 0 — so it stays, and the
    error it causes is 4 orders below OCC's 1e-7 mm modelling tolerance.

    Perpendicularity is unaffected by the scaling, so it is held to 1e-12.
    """
    n = G.offset_normals(curve)
    assert np.abs(np.linalg.norm(n, axis=1) - 1.0).max() < 1e-10
    tan = np.gradient(curve, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    assert np.abs((n * tan).sum(axis=1)).max() < 1e-12


# ---------------------------------------------------------------------------
# GRADIENTS  — the property every downstream stage depends on
# ---------------------------------------------------------------------------

def _outline_scalar(vec12):
    """A scalar summary of the spoke outline, differentiable end to end.

    Deliberately touches the whole chain — control points, Bernstein product, arc
    length, thickness ramps, normals, offset — so one number exercises every path a
    gradient will later flow through.
    """
    g8, ts = vec12[:8], vec12[8:12]
    c, _ = G.bezier_centerline(*g8, span_mm=SPAN, num_points=NPTS, xp=jnp)
    band = G.offset_band(c, *ts, n_across=1, xp=jnp)
    return jnp.sum(band ** 2)


def test_outline_jacobian_matches_finite_differences():
    """Reverse-mode AD vs central differences, per gene, at the on-disk design.

    A gene whose AD gradient disagrees with FD is a gene the optimizer cannot move
    correctly — and because the objective is a sum, one bad component is enough to
    steer the whole search wrong.
    """
    with open(os.path.join(HERE, "best_solution.json")) as fh:
        genes = json.load(fh)["genes"]
    v = jnp.asarray([genes[n] for n in W.GENE_NAMES[:12]])

    g_ad = np.asarray(jax.grad(_outline_scalar)(v))

    base = np.asarray(v, dtype=float)
    g_fd = np.zeros_like(base)
    for i in range(len(base)):
        h = 1e-6 * max(1.0, abs(base[i]))
        up, dn = base.copy(), base.copy()
        up[i] += h
        dn[i] -= h
        g_fd[i] = (float(_outline_scalar(jnp.asarray(up)))
                   - float(_outline_scalar(jnp.asarray(dn)))) / (2 * h)

    scale = np.maximum(np.abs(g_fd), 1.0)
    rel = np.abs(g_ad - g_fd) / scale
    worst = int(np.argmax(rel))
    assert rel.max() < TOL_GRAD, (
        f"worst gene {W.GENE_NAMES[worst]}: AD {g_ad[worst]:.8e} vs "
        f"FD {g_fd[worst]:.8e} (rel {rel.max():.2e})"
    )


def test_gradient_is_finite_everywhere_in_the_box():
    """NaN gradients are the classic autodiff trap — a `where` guard that hides a
    division by zero in the value still propagates NaN through the derivative.  Sample
    the box and check."""
    low, high, _ = _bounds()
    rng = np.random.default_rng(3)
    grad = jax.jit(jax.grad(_outline_scalar))
    for vec in rng.uniform(low, high, size=(12, 14)):
        g = np.asarray(grad(jnp.asarray(vec[:12])))
        assert np.all(np.isfinite(g)), f"non-finite gradient at {vec[:12]}"


def test_kernel_is_jittable():
    """The whole geometry path must survive jit — a data-dependent branch or shape
    would fail here rather than deep inside the FEA."""
    f = jax.jit(_outline_scalar)
    v = jnp.asarray(np.linspace(2.0, 20.0, 12))
    assert np.isfinite(float(f(v)))


def test_self_intersection_margin_detects_a_fold():
    """The margin must go negative for a genome whose offset band actually folds.

    Constructed rather than searched for: take the on-disk curve and inflate the
    thickness until half of it exceeds the minimum radius of curvature (~10.96 mm).
    """
    with open(os.path.join(HERE, "best_solution.json")) as fh:
        genes = json.load(fh)["genes"]
    g8 = [genes[k] for k in ("cx1", "cy1", "cx2", "cy2", "cx3", "cy3", "cx4", "cy4")]
    c, p = G.bezier_centerline(*g8, span_mm=SPAN, num_points=NPTS)

    healthy = G.self_intersection_margin(c, p, 2.0, 2.0, 2.0, 2.0, NPTS)
    folded = G.self_intersection_margin(c, p, 40.0, 40.0, 40.0, 40.0, NPTS)
    assert healthy > 0, "the shipped design should not be folding"
    assert folded < 0, "a 40 mm-thick band on a ~11 mm curvature radius must fold"


# ---------------------------------------------------------------------------
# THE TRANSVERSE CROWN
# ---------------------------------------------------------------------------
# Equivalence-class tolerance (1e-15) is the wrong instrument for these: the quantities
# are compared against QUADRATURE, not against a second spelling of the same closed form,
# so the residual is the quadrature's and not the kernel's.  The volume check below is
# the tightest the sampling supports and is stated as such rather than tuned until green.

CROWN_W = W.SPOKE_WIDTH_MM
CROWN_RIM_OUTER = 50.0                  # `wheel_wheel.RIM_OUTER_RADIUS_MM`, spelled here
                                        # rather than imported: this module is the kernel's
                                        # test and must not need the mesher to run.


def test_the_crown_radius_inverts_its_own_sagitta():
    """`crown_radius_mm` is the inverse of `h = R - sqrt(R^2 - (w/2)^2)`, so the round
    trip is the only check that matters and it is exact to machine precision.

    Swept over a decade of crown heights rather than checked at the shipped 1.0 mm: a
    formula that happens to be right at one point is what this repo keeps finding.
    """
    for h in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        r = G.crown_radius_mm(CROWN_W, h)
        assert r - math.sqrt(r * r - (0.5 * CROWN_W) ** 2) == pytest.approx(h, abs=1e-12)
    assert G.crown_radius_mm(CROWN_W) == pytest.approx(63.22, abs=1e-9)


def test_the_crown_relief_is_the_rectangle_minus_the_segment_not_the_segment():
    """THE FAILURE THIS PINS IS AN OFF-BY-2x, AND IT IS EASY TO MAKE.

    The circular segment is the material UNDER the arc; the relief a crown removes is the
    material ABOVE it, between the arc and the cylinder it replaces.  Confusing the two
    reports 14.957 mm^2 where the answer is 7.443 -- a 2.01x error that would look
    plausible in every downstream number.  Checked here as the identity
    `rectangle = relief + segment`, which fails if either term is the other one.
    """
    h = G.CROWN_HEIGHT_MM
    r = G.crown_radius_mm(CROWN_W, h)
    th = math.asin(0.5 * CROWN_W / r)
    segment = r * r * (th - math.sin(th) * math.cos(th))
    relief = G.crown_relief_area_mm2(CROWN_W, h)
    assert relief + segment == pytest.approx(CROWN_W * h, abs=1e-9)
    assert relief == pytest.approx(7.442884186, abs=1e-8)
    assert relief < segment, "the relief is the SMALLER of the two at this aspect ratio"


def test_the_crown_volume_agrees_with_quadrature():
    """The closed form against a direct numerical sweep of the same solid of revolution.

    Independent in the way that matters: `crown_relief_volume_mm3` evaluates an
    antiderivative, this integrates `pi * (R_out^2 - r(z)^2)` over a million samples.  A
    sign slip or a dropped term in the closed form cannot survive both.
    """
    for h in (0.25, 1.0, 2.5):
        r = G.crown_radius_mm(CROWN_W, h)
        z = np.linspace(-0.5 * CROWN_W, 0.5 * CROWN_W, 1_000_001)
        r_z = (CROWN_RIM_OUTER - r) + np.sqrt(r * r - z * z)
        quad = math.pi * np.trapezoid(CROWN_RIM_OUTER ** 2 - r_z ** 2, z)
        got = G.crown_relief_volume_mm3(CROWN_W, CROWN_RIM_OUTER, h)
        assert got == pytest.approx(quad, rel=1e-9), f"crown height {h}"
    assert G.crown_relief_volume_mm3(CROWN_W, CROWN_RIM_OUTER) == pytest.approx(
        2324.240750, abs=1e-5)


def test_the_crown_thins_the_rim_band_below_the_print_floor_over_45_percent_of_the_face():
    """REGISTERING A MANUFACTURABILITY FACT THAT NOTHING ELSE IN THE TREE POLICES.

    `wheel_fea.MIN_WALL_MM` is a floor on the thickness GENES -- it builds `GENE_SPACE`
    (wheel_fea.py:261) -- and has never been a check on the rim band, which is a fixed
    1.5 mm between two fixed constants.  So the crown thins that band with nothing going
    red anywhere: 1.500 mm at the apex, 0.500 mm at the side faces, and below the 1.2 mm
    print floor over 45.08% of the 22.4 mm face.

    This test exists so the number is on the record rather than met on a print.  It is
    DERIVED from the kernel, not transcribed, so raising `CROWN_HEIGHT_MM` moves it
    instead of rotting it -- what would be wrong is a crown whose cost nobody wrote down.
    """
    band_at_edge = CROWN_RIM_OUTER - G.CROWN_HEIGHT_MM - W.RIM_RADIUS_MM
    assert band_at_edge == pytest.approx(0.5, abs=1e-12)
    assert band_at_edge < W.MIN_WALL_MM, (
        "this test is about a band that falls BELOW the print floor; if the geometry has "
        "changed so that it no longer does, delete the test rather than invert it")

    span = G.crown_min_wall_span_mm(CROWN_W, CROWN_RIM_OUTER, W.RIM_RADIUS_MM,
                                    W.MIN_WALL_MM)
    assert span == pytest.approx(12.3031703, abs=1e-6)
    assert 100.0 * (1.0 - span / CROWN_W) == pytest.approx(45.075, abs=1e-3)


def test_the_min_wall_span_clamps_at_both_ends():
    """The span is a width of face and cannot leave [0, w].

    Both clamps are real cases and not defensive padding: a band thinner than the floor
    even at the apex returns 0.0, and one thick enough to clear it at the edges returns
    the whole face -- which is what an uncrowned rim does, and is the continuity check
    that this reduces to the flat answer as the crown vanishes.
    """
    assert G.crown_min_wall_span_mm(CROWN_W, CROWN_RIM_OUTER, 49.5, W.MIN_WALL_MM) == 0.0
    assert G.crown_min_wall_span_mm(CROWN_W, CROWN_RIM_OUTER, 44.0,
                                    W.MIN_WALL_MM) == CROWN_W
    assert G.crown_min_wall_span_mm(CROWN_W, CROWN_RIM_OUTER, W.RIM_RADIUS_MM,
                                    W.MIN_WALL_MM, height_mm=1e-12) == CROWN_W
