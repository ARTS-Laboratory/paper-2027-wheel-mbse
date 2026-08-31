"""The region-restricted p-norm's three defining properties, on synthetic fields.

PLAN.md §94 items 1 and 2, `studies/study_fillet_pnorm.py`.

WHAT THESE PIN, AND WHAT THEY DELIBERATELY DO NOT.  Not one of them reads the committed
artifact, builds a mesh or solves anything, so none of them can go green because a
measurement got worse — the failure mode PLAN §89's "pin the finding, not its symptom"
names.  They pin the CONSTRUCTION: that the region weight is a compactly supported C2
bump, that the aggregation over it is a p-norm, and that swapping the bump for the
indicator reintroduces exactly the step §94 item 2 says it does.  If the fillet's measured
convergence changes tomorrow these still pass, which is the point; if the kernel stops
being smooth or the normalisation drifts, they fail immediately.

Milliseconds — synthetic arrays throughout.
"""

import math

import numpy as np
import pytest

import study_corner_singularity as CS
import study_fillet_pnorm as FP


R = 0.30            # `FP.REFERENCE_RADIUS_MM`, spelled out so the arithmetic below reads


def test_the_region_weight_is_a_compactly_supported_C2_bump():
    """Value, slope and curvature all reach zero AT the support boundary.

    THIS IS THE WHOLE OF ITEM 2's MECHANISM.  A Gauss point crossing `d = r` has to enter
    with zero weight and zero rate of change of weight, or its arrival is a step in the
    loss however smooth the rest of the expression is.  `max(0, 1 - d^2/r^2)^3` is a cubic
    in a quantity that vanishes linearly at the boundary, so `W`, `dW/dd` and `d2W/dd2`
    are all O((r-d)^k) with k >= 1 there and the match to the identically-zero exterior is
    C2.

    Differentiated numerically rather than symbolically on purpose: the study evaluates
    the closed form, and a symbolic check here would be a second copy of it agreeing with
    itself.
    """
    d = np.linspace(0.0, 2.0 * R, 4001)
    w = FP.region_weight(d * d, R, "bump3")

    assert w[0] == pytest.approx(1.0), "the arc itself carries full weight"
    assert np.all(w[d >= R] == 0.0), "compact support: nothing outside the tube"
    assert np.all(np.diff(w[d <= R]) <= 0.0), "monotone decreasing toward the boundary"

    # Approach the boundary from inside and check all three vanish.  The scaling is the
    # assertion, not the smallness: at (1 - eps) the value is O(eps^3), the slope O(eps^2)
    # and the curvature O(eps), so halving eps must divide them by ~8, ~4 and ~2.
    def derivs(eps):
        h = 1e-7
        x = R * (1.0 - eps)
        f = [float(FP.region_weight(np.array([(x + k * h) ** 2]), R, "bump3")[0])
             for k in (-1, 0, 1)]
        return abs(f[1]), abs(f[2] - f[0]) / (2 * h), abs(f[2] - 2 * f[1] + f[0]) / h ** 2

    a, b = derivs(2e-3), derivs(1e-3)
    for got, want in zip((a[0] / b[0], a[1] / b[1], a[2] / b[2]), (8.0, 4.0, 2.0)):
        assert got == pytest.approx(want, rel=0.05)


def test_the_aggregation_is_a_p_norm_and_not_a_weighted_mean():
    """Non-decreasing in `p`, and bounded by the region's own extremes.

    `_qoi_pnorm_stress`' contract, asserted on the region-restricted form: the p-norm is a
    SMOOTH LOWER BOUND that approaches the max from below as `p` grows.  A normalisation
    slip — dividing by the count instead of the weight sum, or forgetting the `1/p` —
    breaks one of these three and none of the others, which is why all three are here.
    """
    rng = np.random.default_rng(0)
    n = 500
    # A real quarter-circle arc of unit radius about the origin, with the sample points
    # scattered through the annulus that straddles its tube — so `d` genuinely varies over
    # [0, r) and the weights are not all 1.0, which is what makes the weighting testable.
    arc = {"centre": np.array([0.0, 0.0]), "radius": 1.0, "a0": 0.0, "a1": 0.5 * math.pi,
           "A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}
    th = rng.uniform(0.0, 0.5 * math.pi, n)
    rad = 1.0 + rng.uniform(-1.5 * R, 1.5 * R, n)
    xy = np.column_stack([rad * np.cos(th), rad * np.sin(th)])
    vol = rng.uniform(0.5, 1.5, n)
    vm = rng.uniform(1.0, 40.0, n)
    ps = (2.0, 4.0, 8.0, 16.0, 30.0)
    got = FP.region_pnorm(arc, xy, vm, vol, 1, ps, R, "bump3")["values_mpa"]
    vals = [got["%g" % p] for p in ps]

    assert all(a <= b for a, b in zip(vals, vals[1:])), \
        "a p-norm is non-decreasing in p; if it is not, the normalisation is wrong"
    assert vals[0] > vm.min() and vals[-1] < vm.max(), \
        "it must sit strictly inside the region's own range at every finite p"

    # And at p = 1 it is the WEIGHTED mean, not the unweighted one.  The reference weights
    # come from `study_corner_singularity._distance_to_arc` — the geometry module, not this
    # one — so the assertion is about the AGGREGATION rather than a second copy of itself.
    d = CS._distance_to_arc(xy, arc)
    assert 0.05 < float((d < R).mean()) < 0.95, "the tube must partition the sample"
    w = FP.region_weight(d * d, R, "bump3") * vol
    one = FP.region_pnorm(arc, xy, vm, vol, 1, (1.0,), R, "bump3")["values_mpa"]["1"]
    assert one == pytest.approx(float(np.sum(w * vm) / w.sum()), rel=1e-12)


def test_the_indicator_steps_where_the_bump_does_not():
    """One point crossing the boundary: a jump under the control, nothing under the bump.

    §94 item 2 in miniature, and the reason the study carries the indicator at all.  The
    study measures this against a real mesh and a real gene (`R_hub`, 0.0592 mm from the
    shipped genome, a 1.08% step in the p-norm at p = 4); here it is reduced to the one
    arithmetic fact underneath that result, so the property is pinned even if nobody ever
    runs the driver again.
    """
    eps = 1e-9
    d2_in = np.array([(R - eps) ** 2])
    d2_out = np.array([(R + eps) ** 2])
    for kernel, tol in (("indicator", None), ("bump3", 1e-20)):
        w_in = float(FP.region_weight(d2_in, R, kernel)[0])
        w_out = float(FP.region_weight(d2_out, R, kernel)[0])
        if tol is None:
            assert w_in == 1.0 and w_out == 0.0, \
                "the indicator's membership is a step BY CONSTRUCTION — that is the defect"
        else:
            assert abs(w_in - w_out) < tol, \
                "the bump must cross its own support boundary continuously"
