"""
=============================================================================
  IS THE JUNCTION CORNER A STRESS SINGULARITY?  MEASURED IN THE FIELD,
  NOT INFERRED FROM THE DEFLECTION LADDER
=============================================================================
    .venv-opt/bin/python studies/study_corner_singularity.py      (make corner)

PLAN.md §29's successor.  §29 read a deflection convergence order of p = 0.502 off the
GCI ladder, matched it to Williams' eigenvalue for the junction's 322 deg wedge, and
called the corner confirmed.  That `p` was wrong — the study had measured its cell size on
`wheel_mesh`'s spoke block while solving on the `wheel_wheel` mesh — and at the corrected
p = 0.638 the agreement is gone.  §29's corner hypothesis went back to being a hypothesis.

FIRST, WHAT THIS FILE DOES NOT DISCOVER.  That the peak stress diverges was established at M4
and is already pinned by
`tests/test_wheel_fea.py::test_peak_stress_diverges_but_the_field_converges`; M8b-i.6 step 2
rebuilt the entire stress constraint around it, and PLAN.md's banner says "the max is not a
number" in those words.  Nothing here overturns or re-establishes any of that.  What the
existing test asserts is THAT the peak grows, by more than 20% over coarse..fine.  What is
missing is the RATE (a growth threshold cannot be compared with a wedge angle, a log-log slope
can), WHICH of the four re-entrant corners does it, and what wedge angle each one actually has.

THIS FILE MEASURES THOSE, WHERE THEY CAN BE MEASURED: the stress field at the corner.
That is a better test than the deflection order for a reason worth stating, because it is
why §29's approach was weak even before the arithmetic error.  `axle_drop_mean_mm` is a
GLOBAL functional; its convergence order is a claim about how a singular local field
pollutes a far-field average, which involves a duality argument nobody here has made (a
smooth functional would converge at 2*lambda, not lambda).  The near-field decay rate and
the divergence of the peak are LOCAL and need no such argument.

THREE MEASUREMENTS, in increasing order of how hard they are to explain away:

  A.  RADIAL DECAY.  sigma(r, theta) ~ K r^(lambda-1) f(theta) near a re-entrant corner.
      Pooling Gauss points by distance alone mixes f(theta) into the fit and gives
      correlations of -0.3 (measured, first attempt).  Binning by log r and taking the
      MAX over theta in each bin removes f: max_theta f is a constant, so the binned
      maxima decay as a pure power of r.

  B.  DIVERGENCE OF THE PEAK.  A singular field has no finite peak, so the nearest-Gauss-
      point stress must GROW under refinement, like h^(lambda-1).  This needs no angular
      treatment and no fit window, which makes it the robust one.  A convergent corner
      gives a slope of zero.  THIS IS THE HEADLINE.

  C.  WHAT THE OPTIMIZER ACTUALLY SEES.  The objective does not use the raw peak; it uses
      a Gauss-weighted p-norm.  Whether THAT diverges is a different question from whether
      the field does, and it is the one that decides if the stress constraint is a
      physical quantity or a mesh setting.

FOURTH, SINCE 2026-08-23: THE SAME THREE ON A FILLETED MESH.  `--fillet genome`
(`make corner-fillet`) is FILLET_PLAN.md's Step 2, and it is one flag on this driver rather
than a second script on purpose — a before/after measured by two instruments is not a
before/after, and this arc has already been bitten by that once (PART 6 found two recorded
fold tables disagreeing by 20x with neither criterion written down).  Three things change
and each is named where it happens:

  * `corner_points` stays UNFILLETED.  The four corners are the arc's reference LOCATIONS
    and may not move when the mesh does.  A filleted `<ring>_junction` patch has the same
    shape and index convention and its `[0, 0]` is NOT `P_t` — it is `N`, 4.34 mm away at
    the rim — so reading them off the filleted blocking would keep the label and move the
    place.  The filleted body's own points come from `fillet_points` under their own names.
  * `P_t` stops being a boundary corner at all, so Williams is WITHHELD there rather than
    printing a crack's 0.5000 for a point in the middle of the material, and the wedge
    search had to stop being able to land on a midside node.
  * "Does the peak diverge" gets a second instrument.  The log-log slope was written for a
    ladder on which every probe was singular; the ratio of successive DIFFERENCES separates
    a settling tail from a divergence, and on the unfilleted ladder every corner comes in at
    0.999 or above.  Both are reported and neither is dropped.

AND THE CONTROL THAT LICENSES ANY OF IT, `--continuity`.  The filleted ladder's axle drop is
38% below the unfilleted one, which is equally well explained by the fillet's stiffness and
by a different model.  The blocking takes an explicit radius pair, so drive it toward zero
and demand the unfilleted wheel back.

WILLIAMS' EIGENVALUE is computed per corner from a wedge angle MEASURED ON THE MESH — the
sum of the incident elements' interior angles at the corner node — rather than quoted from
the export manifest.  The manifest measures the exported SOLID, which is filleted; the
mesh is what has the corner.  Reading the prediction off the wrong body is the same class
of mistake as reading the cell size off the wrong mesh.
=============================================================================
"""

import argparse
import json
import math
import os

import numpy as np

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import wheel_fem as fem
import wheel_genome as wg
import wheel_wheel as WW

HERE = os.path.dirname(os.path.abspath(__file__))

GENOME = "best_solution.json"
LADDER = ("smoke", "coarse", "medium", "fine")
PROBE_RADIUS_MM = 0.3       # "at the corner", for picking the loaded copy and the peak
FIT_MAX_MM = 1.5            # outer limit of the radial fit window
FIT_DECADE = 10.0           # fit the inner decade only, where the singular term leads

# A node in the material's interior sums to exactly 360 deg; the tolerance is for
# floating point, not for a corner that is nearly interior.  `SMOOTH_WEDGE_TOL_DEG` is
# looser on purpose: a node on a DISCRETISED curved boundary sums to 180 plus the arc's
# turn across one node, which is a mesh number and goes to zero under refinement, so the
# band has to hold the coarsest rung's facet angle without holding a real corner.
INTERIOR_WEDGE_DEG = 359.0
SMOOTH_WEDGE_TOL_DEG = 25.0

# TEST B's second instrument.  A divergent peak holds the ratio of its successive
# differences at or above 1 — measured, every unfilleted corner comes in at 0.999 or
# higher — so anything comfortably below separates.  `SETTLED_TAIL_FRACTION` is the
# escape hatch for a series that has already arrived, where the increments are noise and
# their ratio means nothing.
SETTLING_RATIO = 0.75
SETTLED_TAIL_FRACTION = 0.01

# The continuity control's radius ladder, in mm.  Both ends are chosen: the smallest is
# near the construction's floor (below it the boundary layer is thinner than the mesh can
# resolve and the drop reads HIGH, which is a discretisation statement and is reported
# rather than trimmed), the largest is `R_rim`'s own gene bound.  The genome's OWN two
# radii are spliced in at run time rather than written here, so this list cannot go stale
# against a promotion — which is how a committed constant last went wrong in this tree.
CONTINUITY_RADII_MM = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0)


def load_genes(path):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


def williams_lambda(omega_deg):
    """Smallest root in (0, 1) of sin(lambda*omega) + lambda*sin(omega) = 0.

    Mode-I eigenvalue for a traction-free re-entrant wedge of material angle omega;
    stress ~ r^(lambda-1).  Bisection, because the function is smooth and monotone enough
    on (0, 1) that nothing cleverer earns its complexity.  Verified against the two cases
    with known answers: omega = 360 deg (a crack) gives exactly 0.5, and 270 deg gives the
    textbook 0.5445.
    """
    w = math.radians(omega_deg)

    def f(L):
        return math.sin(L * w) + L * math.sin(w)

    lo, hi = 1e-9, 1.0 - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# WHERE THE CORNERS ARE, AND HOW WIDE THEY ARE
# ---------------------------------------------------------------------------

def corner_points(genes, cfg):
    """The four junction corners of sector 0, in the global frame.

    `sector_blocks` lays both junction patches out with `i` along the ring arc and `j`
    from the arc to the far flank, so the two ends of the arc — the patch's [0, 0] and
    [-1, 0] nodes — are exactly P_t (where the straddling flank crosses the ring circle)
    and P_c (the centerline endpoint, locked on the circle by the genome).  Those are the
    two places the ring's free surface stops being free.

    ALWAYS THE UNFILLETED BLOCKING, AND THAT IS THE POINT.  These four points are the
    arc's REFERENCE LOCATIONS: what "the peak at `rim:P_t` stopped diverging" means is
    that the field at the same place in the same wheel stopped diverging, so the place
    may not move when the mesh does.  It would move if this read the filleted blocking,
    and it would move SILENTLY: the filleted `<ring>_junction` patch has the same shape
    and the same index convention, its `[-1, 0]` really is still `P_c`, and its `[0, 0]`
    is `N` — the fillet block's inner-edge crossing — which is 0.47 mm from `P_t` at the
    hub and 4.34 mm at the rim.  A filleted mesh's own points come from `fillet_points`
    under their own names.
    """
    b = WW.sector_blocks(genes, cfg)
    out = {}
    for ring, label in (("hub_junction", "hub"), ("rim_junction", "rim")):
        out[f"{label}:P_t"] = np.asarray(b[ring][0, 0], dtype=float)
        out[f"{label}:P_c"] = np.asarray(b[ring][-1, 0], dtype=float)
    return out


def fillet_points(genes, cfg, fillet):
    """The filleted mesh's OWN named points at both junctions.

    `P_t` is the corner the fillet removes, so it has no counterpart on the filleted
    body; what the filleted body has instead is an arc, and these are the places on it
    where a corner could have survived the construction:

      `A`    the tangent point on the straddling flank,
      `arc`  the middle of the fillet arc,
      `B`    the tangent point on the ring circle,
      `N`    where the fillet block's inner edge crosses the ring circle — the one point
             in the list that is INTERIOR, and the one where four blocks meet.

    A fillet is tangent to both legs, so `A` and `B` should measure a smooth boundary and
    `N` a smooth interior; anything re-entrant here would be a corner the construction
    introduced, which is a thing worth being able to see rather than assume.

    Read off the block grids rather than re-solving the tangency, so this cannot drift
    from what `build_wheel` meshed.  `hub_fillet_a[0, 0]` is `A` and `hub_fillet_b[-1, 0]`
    is `B` because `_filleted_sector_blocks` passes `arc_a`/`arc_b` as each patch's
    `bottom`; `hub_fillet_b[0, -1]` is `N` because `inner_b` is its `top` and starts
    there.  Pinned in `tests/test_corner_singularity.py` on the geometry — `A` and `B` lie
    on one circle of the gene's own radius, and `B` and `N` lie on the ring circle.
    """
    b = WW.sector_blocks(genes, cfg, fillet=fillet)
    out = {}
    for label in ("hub", "rim"):
        fa, fb = b[f"{label}_fillet_a"], b[f"{label}_fillet_b"]
        arc = np.concatenate([fa[:, 0, :], fb[1:, 0, :]])
        out[f"{label}:A"] = np.asarray(fa[0, 0], dtype=float)
        out[f"{label}:arc"] = np.asarray(arc[len(arc) // 2], dtype=float)
        out[f"{label}:B"] = np.asarray(fb[-1, 0], dtype=float)
        out[f"{label}:N"] = np.asarray(fb[0, -1], dtype=float)
    return out


def fillet_arcs(genes, cfg, fillet):
    """Each junction's fillet arc as `(centre, radius, a0, a1)` — analytic, not sampled.

    Recovered by a least-squares circle fit through the arc's own mesh nodes, which is a
    fit only in form: the residual is 7e-14 mm and the radius comes back as the gene's own
    value to twelve digits, because the nodes ARE on a circle by construction.  Fitting
    rather than re-solving `_fillet_tangency` keeps this reading what `build_wheel`
    meshed; a second tangency solve here is the duplicated-geometry drift
    `study_fillet_block.py`'s docstring is about.

    Wanted because three sampled points are not a verdict on a surface.  `A`, `arc` and
    `B` each answer "does the field at THIS spot settle"; what Step 2 asks is whether the
    fillet's peak is a number, and that is a maximum over the whole arc.
    """
    b = WW.sector_blocks(genes, cfg, fillet=fillet)
    out = {}
    for label in ("hub", "rim"):
        fa, fb = b[f"{label}_fillet_a"], b[f"{label}_fillet_b"]
        arc = np.concatenate([fa[:, 0, :], fb[1:, 0, :]])
        M = np.c_[2.0 * arc[:, 0], 2.0 * arc[:, 1], np.ones(len(arc))]
        sol, *_ = np.linalg.lstsq(M, (arc ** 2).sum(axis=1), rcond=None)
        C = np.array([sol[0], sol[1]])
        R = math.sqrt(sol[2] + C @ C)
        resid = float(np.abs(np.linalg.norm(arc - C, axis=1) - R).max())
        a0 = math.atan2(arc[0, 1] - C[1], arc[0, 0] - C[0])
        a1 = math.atan2(arc[-1, 1] - C[1], arc[-1, 0] - C[0])
        a1 = a0 + ((a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi)
        out[label] = {"centre": C, "radius": R, "a0": a0, "a1": a1,
                      "A": arc[0].copy(), "B": arc[-1].copy(), "fit_residual_mm": resid}
    return out


def _distance_to_arc(pts, arc):
    """Perpendicular distance from each point to the arc, folding to the endpoints
    outside the sweep.  Vectorised; `pts` is (n, 2)."""
    C, R, a0, a1 = arc["centre"], arc["radius"], arc["a0"], arc["a1"]
    d = pts - C[None, :]
    r = np.linalg.norm(d, axis=1)
    ang = np.arctan2(d[:, 1], d[:, 0])
    lo, hi = (a0, a1) if a1 >= a0 else (a1, a0)
    # Unwrap each angle into the sweep's own branch before testing containment.
    ang = lo + ((ang - lo) % (2.0 * math.pi))
    inside = ang <= hi
    perp = np.abs(r - R)
    ends = np.minimum(np.linalg.norm(pts - arc["A"][None, :], axis=1),
                      np.linalg.norm(pts - arc["B"][None, :], axis=1))
    return np.where(inside, perp, ends)


def arc_peak(arc, xy, vm, n_sp, radius=PROBE_RADIUS_MM):
    """TEST B, over the fillet's whole free surface rather than at three points on it.

    The peak von Mises within `radius` of the arc, on the rotational copy that carries
    it — the same loaded-copy rule `loaded_copy` uses, for the same reason: the wheel is
    loaded at one contact patch and the twelve images of a fillet see twelve loads.

    THE REGION IS DEFINED ANALYTICALLY AND SO DOES NOT MOVE WITH THE MESH.  A tube round a
    polyline of the arc's own nodes would be a slightly different region at every rung,
    which is the one thing a convergence measurement may not have.
    """
    ca, sa = np.cos(-2.0 * np.pi * np.arange(n_sp) / n_sp), np.sin(
        -2.0 * np.pi * np.arange(n_sp) / n_sp)
    best, best_k = 0.0, None
    for k in range(n_sp):
        rot = np.column_stack([ca[k] * xy[:, 0] - sa[k] * xy[:, 1],
                               sa[k] * xy[:, 0] + ca[k] * xy[:, 1]])
        m = _distance_to_arc(rot, arc) < radius
        if m.any() and float(vm[m].max()) > best:
            best, best_k = float(vm[m].max()), k
    return {"peak_vm_mpa": best, "loaded_copy": best_k}


def probe_points(genes, cfg, fillet=None):
    """Every point the ladder measures: the four reference corners, plus — on a filleted
    mesh — the eight the fillet itself creates.  The four are in both lists on purpose,
    because the before/after comparison is the whole measurement."""
    pts = corner_points(genes, cfg)
    if fillet is not None:
        pts.update(fillet_points(genes, cfg, fillet))
    return pts


def measured_wedge_deg(mesh, point, order):
    """The material wedge at a corner, summed from the mesh's own incident elements.

    Sum of the interior angles at the corner node over every element that owns it.  A node
    interior to the material sums to 360; a smooth boundary node to 180; a re-entrant
    corner to something between 180 and 360.  Midside nodes contribute a straight 180 and
    are skipped, since a Q9 midside cannot be a corner of the geometry.

    This measures the MESHED body.  That is the point: the exported solid is filleted at
    these corners and has no wedge to measure.

    THE NEAREST NODE IS SEARCHED OVER Q9 VERTICES ONLY.  A midside node contributes a
    straight 180 deg and is skipped by the loop below, so a search that can land on one
    returns `wedge_deg = 0.0` and `n_incident_corner_elements = 0` — a number that looks
    like a measurement and is not.  It never fired while every probe was an exact mesh
    node (`node_gap_mm = 0`), which is what the four unfilleted junction corners are.  It
    fires immediately on a FILLETED mesh, where `P_t` is no longer a node at all but a
    point in the material's interior: `rim:P_t` at `coarse` reported 0.00 deg against
    `hub:P_t`'s correct 360.00, purely on which of two equally-good nodes was nearer.
    Restricting to vertices costs nothing on the unfilleted mesh — the regenerated
    unfilleted report is unchanged in every field — and makes the filleted one readable.
    """
    xy = mesh.coords
    verts = np.unique(mesh.conn[:, :4])
    nid = int(verts[np.argmin(np.linalg.norm(xy[verts] - point, axis=1))])
    gap = float(np.linalg.norm(xy[nid] - point))
    conn = mesh.conn
    total = 0.0
    n_inc = 0
    for e in np.where((conn == nid).any(axis=1))[0]:
        corners = conn[e, :4]
        hit = np.where(corners == nid)[0]
        if not len(hit):
            continue                      # midside/centre: straight, contributes nothing
        k = int(hit[0])
        p = xy[nid]
        a = xy[corners[(k + 1) % 4]] - p
        b = xy[corners[(k - 1) % 4]] - p
        cosang = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        total += math.acos(max(-1.0, min(1.0, cosang)))
        n_inc += 1
    return {"node": nid, "node_gap_mm": gap, "wedge_deg": math.degrees(total),
            "n_incident_corner_elements": n_inc}


# ---------------------------------------------------------------------------
# THE FIELD
# ---------------------------------------------------------------------------

def solve_field(genes, cfg, fillet=None):
    mesh = WW.build_wheel(genes, cfg, fillet=fillet)
    res = fem.solve_wheel(mesh)
    lam, mu = fem.lame(fem.YOUNGS_MODULUS_PLA_MPA, fem.POISSON_RATIO_PLA, plane="stress")
    gs = fem.gauss_stresses(mesh.coords, mesh.conn, res["u"], order=cfg.order,
                            lam=lam, mu=mu, nonlinear=False, cauchy=True)
    return mesh, res, gs["xy"].reshape(-1, 2), gs["von_mises"].reshape(-1)


def _copies(p0, n):
    """The n rotational images of a sector-0 point.  The wheel is loaded at ONE contact
    patch, so the n copies of a corner carry n different loads and must not be pooled —
    the loaded copy is picked per corner instead."""
    r0, th0 = math.hypot(*p0), math.atan2(p0[1], p0[0])
    return [np.array([r0 * math.cos(th0 + 2 * math.pi * k / n),
                      r0 * math.sin(th0 + 2 * math.pi * k / n)]) for k in range(n)]


def loaded_copy(p0, xy, vm, n, radius=PROBE_RADIUS_MM):
    """The rotational image of `p0` carrying the highest near-corner stress, and that
    stress.  Returns (point, peak) or (None, 0.0) if no Gauss point is within `radius`."""
    best, best_c = 0.0, None
    for c in _copies(p0, n):
        d = np.linalg.norm(xy - c, axis=1)
        m = d < radius
        if m.any() and vm[m].max() > best:
            best, best_c = float(vm[m].max()), c
    return best_c, best


def radial_decay(xy, vm, centre, n_bins=13):
    """TEST A.  Log-spaced radial bins, max over theta in each, power-law fit on the
    inner decade.  Returns the fitted lambda and the correlation that says whether the
    fit means anything — a corner that is NOT singular gives a slope near zero and a
    correlation near zero, and both are reported rather than one."""
    d = np.linalg.norm(xy - centre, axis=1)
    m = (d > 1e-4) & (d < FIT_MAX_MM)
    if m.sum() < 30:
        return None
    dd, vv = d[m], vm[m]
    edges = np.exp(np.linspace(math.log(dd.min()), math.log(FIT_MAX_MM), n_bins))
    br, bv = [], []
    for i in range(len(edges) - 1):
        s = (dd >= edges[i]) & (dd < edges[i + 1])
        if s.sum() >= 3:
            br.append(math.sqrt(edges[i] * edges[i + 1]))
            bv.append(float(vv[s].max()))
    if len(br) < 5:
        return None
    br, bv = np.array(br), np.array(bv)
    inner = br < br.min() * FIT_DECADE
    if inner.sum() < 4:
        return None
    slope, _ = np.polyfit(np.log(br[inner]), np.log(bv[inner]), 1)
    corr = float(np.corrcoef(np.log(br[inner]), np.log(bv[inner]))[0, 1])
    return {"n_bins": int(inner.sum()),
            "window_mm": [float(br[inner].min()), float(br[inner].max())],
            "slope": float(slope), "lambda": float(slope + 1.0), "corr": corr,
            "bin_r_mm": [float(x) for x in br], "bin_max_vm": [float(x) for x in bv]}


def nearest_probe(point, pts, n_sp):
    """The named probe nearest `point`, over all `n_sp` rotational copies of each.

    The wheel's global peak has been located by hand in FILLET_PLAN's PART 4 and PART 7
    and the answer — `rim:P_c`, 15-24 um away — is load-bearing for the whole arc: it is
    why a fillet at `P_t` cannot deliver Step 2's headline.  It was measured twice in a
    plan file and never by the driver, so it is measured here.
    """
    best, best_d = None, float("inf")
    for name, p0 in pts.items():
        for c in _copies(p0, n_sp):
            d = float(np.linalg.norm(point - c))
            if d < best_d:
                best, best_d = name, d
    return {"nearest_probe": best, "distance_mm": best_d}


def continuity_sweep(genes, cfg_name, radii, fillet_ref=True):
    """DOES THE FILLETED BLOCKING SOLVE THE SAME WHEEL?  The control for the whole run.

    A filleted ladder that reports a 38% shift in `axle_drop_mm` against the unfilleted
    one is reporting either the fillet's stiffness or a different model, and the two are
    indistinguishable from one ladder.  They are separable by a limit: the filleted
    blocking takes an explicit radius pair, so drive it to zero and the mesh must
    reproduce the unfilleted wheel.  It cannot be driven TO zero — `sector_blocks`
    refuses a zero radius at either end, because the re-cut moves four blocks and "no
    fillet here" is a different blocking rather than this one at `R = 0` — so the
    statement is a limit rather than an identity, and both halves are reported.

    One linear solve per radius at one config.  Nothing here is a corner measurement.
    """
    if fillet_ref is None:
        raise ValueError("the continuity control compares a FILLETED ladder against the "
                         "unfilleted wheel; there is nothing to control on a run with "
                         "`fillet=None`.")
    cfg = WW.get_config(cfg_name)
    base = fem.solve_wheel(WW.build_wheel(genes, cfg))["axle_drop_mm"]
    radii = sorted(set(radii) | {float(genes[12]), float(genes[13])})
    rows = []
    for R in radii:
        pair = (float(R), float(R))
        try:
            mesh = WW.build_wheel(genes, cfg, fillet=pair)
        except ValueError as exc:
            rows.append({"R_mm": float(R), "built": False, "why": str(exc)[:120]})
            continue
        drop = float(fem.solve_wheel(mesh)["axle_drop_mm"])
        rows.append({"R_mm": float(R), "built": True, "axle_drop_mm": drop,
                     "rel_to_unfilleted": drop / float(base) - 1.0})
    shipped = float(fem.solve_wheel(
        WW.build_wheel(genes, cfg, fillet=fillet_ref))["axle_drop_mm"])
    return {"config": cfg_name, "unfilleted_axle_drop_mm": float(base),
            "shipped_fillet_axle_drop_mm": shipped,
            "shipped_rel_to_unfilleted": shipped / float(base) - 1.0,
            "rows": rows}


# PART 12 quoted the filleted deflection's convergence as the spread of `axle_drop_mm`
# over `coarse..fine` -- 0.141% -- and checked it against the +-0.3% band this arc exists
# partly to earn back.  Both are repeated here as constants rather than as prose, because
# PART 16 turns that one number into a criterion applied to nine candidate profiles and a
# criterion has to be stated before it is applied.
CONVERGENCE_LADDER = ("coarse", "medium", "fine")
CONVERGENCE_BAND_PCT = 0.3


def profile_convergence(genes, pairs, ladder=CONVERGENCE_LADDER, fillet=True):
    """What each candidate layer profile costs the SOLVE, which the blocking cannot see.

    FILLET_PLAN PART 13 declined the genome-robust profile for two reasons; PART 14 killed
    one of them and PART 16 showed the argmax itself is not stale, so the whole of what is
    left is this: at `GENOME_ROBUST_ENTRY/END` the shipped genome's filleted axle-drop
    spread over `coarse..fine` goes 0.141% -> 0.513% and crosses back over the +-0.3%
    band.  That was measured at ONE alternative pair.  The profile surface is a broad
    ridge, so the question nobody has asked is whether some OTHER cell on it is
    genome-robust AND stays inside the band -- which is the two-objective version, and it
    is three linear solves per pair rather than an optimisation.

    `pairs` comes from `study_fillet_block.LAYER_PROFILE_CANDIDATES` -- every cell that
    clears `MIN_SJ_TARGET` on the whole clamped, fold-clean box and refuses none of it,
    so a negative here is a negative over the entire candidate set and not over a sample.
    A constant rather than a read of that study's artifact, deliberately: one study's
    freshness must not become another's problem.  The shipped pair and the frontier ride
    along as controls.  A pair that refuses to build at any rung is reported with its
    reason and no spread, never silently dropped.
    """
    rows = []
    for entry, end in pairs:
        drops, why = [], None
        for name in ladder:
            try:
                mesh = WW.build_wheel(genes, name, fillet=fillet,
                                      layer_profile=(entry, end))
                drops.append(float(fem.solve_wheel(mesh)["axle_drop_mm"]))
            except Exception as exc:          # a candidate must not kill the driver
                why = f"{type(exc).__name__}: {exc}"
                break
        row = {"entry": float(entry), "end": float(end),
               "ladder": list(ladder), "axle_drop_mm": drops, "why": why}
        if why is None and len(drops) == len(ladder):
            lo, hi = min(drops), max(drops)
            mean = sum(drops) / len(drops)
            row["spread_pct"] = 100.0 * (hi - lo) / mean
            row["inside_band"] = bool(row["spread_pct"] <= CONVERGENCE_BAND_PCT)
        else:
            row["spread_pct"], row["inside_band"] = None, None
        rows.append(row)
    return {"band_pct": CONVERGENCE_BAND_PCT, "ladder": list(ladder), "rows": rows}


def run(genome=GENOME, ladder=LADDER, fillet=None, continuity=None, profiles=False):
    genes = load_genes(genome)
    n_sp = WW.NUMBER_OF_SPOKES
    out = {"genome": genome, "ladder": list(ladder), "n_spokes": n_sp,
           "fillet": (fillet if fillet is None or fillet is True
                      else [float(v) for v in fillet]),
           "probe_radius_mm": PROBE_RADIUS_MM, "rungs": [], "williams": {}}

    finest = WW.get_config(ladder[-1])
    pts = probe_points(genes, finest, fillet)

    for name in ladder:
        cfg = WW.get_config(name)
        mesh, res, xy, vm = solve_field(genes, cfg, fillet=fillet)
        rec = {"config": name, "n_elements": int(mesh.n_elements),
               "n_nodes": int(mesh.n_nodes),
               "h": 1.0 / math.sqrt(mesh.n_elements),
               "axle_drop_mm": float(res["axle_drop_mm"]),
               "global_max_vm_mpa": float(vm.max()), "corners": {}}
        rung_pts = probe_points(genes, cfg, fillet)
        rec["global_peak"] = nearest_probe(xy[int(np.argmax(vm))], rung_pts, n_sp)
        for cname, p0 in rung_pts.items():
            centre, peak = loaded_copy(p0, xy, vm, n_sp)
            entry = {"peak_vm_mpa": peak}
            entry.update(measured_wedge_deg(mesh, p0, cfg.order))
            if name == ladder[-1] and centre is not None:
                entry["radial_decay"] = radial_decay(xy, vm, centre)
            rec["corners"][cname] = entry
        if fillet is not None:
            rec["surfaces"] = {}
            for label, arc in fillet_arcs(genes, cfg, fillet).items():
                e = arc_peak(arc, xy, vm, n_sp)
                e.update({"radius_mm": arc["radius"],
                          "sweep_deg": math.degrees(abs(arc["a1"] - arc["a0"])),
                          "fit_residual_mm": arc["fit_residual_mm"]})
                rec["surfaces"][f"{label}:surface"] = e
        out["rungs"].append(rec)
        gp = rec["global_peak"]
        print(f"  {name:<7} elem {rec['n_elements']:>6}  h {rec['h']:.6f}  "
              f"drop {rec['axle_drop_mm']:.5f}  global vm max "
              f"{rec['global_max_vm_mpa']:8.2f} MPa  at {gp['nearest_probe']} "
              f"+{gp['distance_mm'] * 1000.0:.1f} um", flush=True)

    # Williams, from the wedge angles the FINEST mesh actually has.
    #
    # `kind` IS NOT DECORATION.  On the unfilleted mesh every probe is a re-entrant
    # boundary corner and Williams applies to all four.  On a filleted one it applies to
    # none of the fillet's own points and to neither `P_t`: `P_t` is now INTERIOR (four
    # elements, 360 deg) and an interior node is not a traction-free wedge, so quoting
    # `williams_lambda(360)` there would print a crack's 0.5000 for a point in the middle
    # of the material.  `lambda` is therefore withheld unless the wedge is a re-entrant
    # BOUNDARY corner, and the reason is carried in the record rather than left to the
    # reader.
    fine_rec = out["rungs"][-1]
    for cname, e in fine_rec["corners"].items():
        w = e["wedge_deg"]
        if w >= INTERIOR_WEDGE_DEG:
            kind = "interior"
        elif w > 180.0 + SMOOTH_WEDGE_TOL_DEG:
            kind = "re_entrant"
        elif w > 180.0 - SMOOTH_WEDGE_TOL_DEG:
            kind = "smooth"
        else:
            kind = "convex"
        rec = {"wedge_deg": w, "kind": kind, "re_entrant": bool(kind == "re_entrant"),
               "node_gap_mm": e["node_gap_mm"]}
        if kind == "re_entrant":
            rec["lambda"] = williams_lambda(w)
        out["williams"][cname] = rec
    out["williams_checks"] = {"crack_360deg": williams_lambda(360.0),
                              "textbook_270deg": williams_lambda(270.0)}

    # TEST B: log-log slope of the peak against h.
    lh = np.log([r["h"] for r in out["rungs"]])
    out["divergence"] = {}
    series = {"global_max_vm": [r["global_max_vm_mpa"] for r in out["rungs"]]}
    for cname in pts:
        series[cname] = [r["corners"][cname]["peak_vm_mpa"] for r in out["rungs"]]
    for sname in out["rungs"][-1].get("surfaces", {}):
        series[sname] = [r["surfaces"][sname]["peak_vm_mpa"] for r in out["rungs"]]
    for k, v in series.items():
        v = np.array(v, dtype=float)
        if (v <= 0).any():
            continue
        s_all = float(np.polyfit(lh, np.log(v), 1)[0])
        s_fin = float(np.polyfit(lh[-3:], np.log(v[-3:]), 1)[0])
        # THE SLOPE IS NOT THE ONLY READ, AND ON A FILLETED MESH IT IS NOT THE BEST ONE.
        # `slope_finest3` fits a power law through three rungs and calls anything steeper
        # than -0.15 divergent; that separates a singular corner from a smooth one and it
        # was written for a ladder where every probe was singular.  A BOUNDED quantity
        # still approaching its limit has a nonzero slope over three rungs, and two of the
        # fillet's own probes land there: `hub:B` fits -0.2331 while its successive
        # differences run 1.76, 0.96, 0.22 mm — a ratio falling 0.55 -> 0.23, which is a
        # tail, not a divergence.  The differences are the sharper instrument and both are
        # reported.  A divergent series holds its ratio near or above 1 (`rim:P_c`
        # unfilleted: 0.68 -> 1.00); a convergent one decays.
        d = np.diff(v)
        ratios = [float(b / a) if abs(a) > 1e-12 else float("nan")
                  for a, b in zip(d, d[1:])]
        tail = float(abs(d[-1]) / v[-1])
        out["divergence"][k] = {
            "peak_mpa": [float(x) for x in v],
            "slope_all": s_all, "slope_finest3": s_fin,
            "lambda_from_slope": s_fin + 1.0,
            "growth_over_ladder": float(v[-1] / v[0]),
            "increments_mpa": [float(x) for x in d],
            "increment_ratios": ratios,
            "increment_ratio_finest": ratios[-1] if ratios else None,
            "tail_fraction": tail,
            # SETTLING IS NOT "THE RATIO FELL".  A geometrically convergent sequence has a
            # roughly CONSTANT ratio below 1 — demanding a falling one demands superlinear
            # convergence and would refuse `hub:surface`'s 0.430 -> 0.507, which is textbook
            # behaviour.  What separates settling from diverging is the ratio's DISTANCE
            # below 1, and the second clause catches the case the ratio cannot speak to at
            # all: a series already at its limit, whose increments are noise about zero and
            # whose ratio is therefore arbitrary (`hub:P_t` filleted: 0.12, -0.003, 0.014
            # MPa on a value of 5.5, and a ratio of -4.78).
            "settling": bool(tail < SETTLED_TAIL_FRACTION
                             or (ratios and abs(ratios[-1]) < SETTLING_RATIO
                                 and tail < 0.10)),
            # The limit of the remaining geometric tail, WHERE THERE IS ONE.  Withheld
            # for anything not settling, which is the only discipline that keeps this
            # from being a way to quote a number for a divergent series: summing
            # d * r / (1 - r) needs |r| < 1, and a divergent peak's r is >= 1 by
            # definition.  It is an estimate and the measurement is `peak_mpa[-1]`.
            "settled_estimate_mpa": None,
            # A convergent peak gives ~0.  The threshold is deliberately loose: this
            # separates "diverges" from "converges", not one lambda from another.
            "diverges": bool(s_fin < -0.15)}
        rec = out["divergence"][k]
        if rec["settling"] and ratios and abs(ratios[-1]) < 1.0:
            r = abs(ratios[-1])
            rec["settled_estimate_mpa"] = float(v[-1] + d[-1] * r / (1.0 - r))

    if continuity is not None:
        out["continuity"] = continuity_sweep(genes, continuity, CONTINUITY_RADII_MM,
                                             fillet_ref=fillet)
    if profiles and fillet is not None:
        import study_fillet_block as fbk
        # The ridge, plus BOTH published pairs as controls -- the shipped one because it
        # is what 0.141% was measured at, and the genome-robust one because 0.513% is the
        # number this whole sweep exists to put in context.  De-duplicated, order kept.
        pairs, seen = [], set()
        for p in (tuple(fbk.LAYER_PROFILE_CANDIDATES)
                  + ((fbk.GENOME_ROBUST_ENTRY, fbk.GENOME_ROBUST_END),
                     (fbk.LAYER_ENTRY_SLOPE, fbk.LAYER_END_OFFSET))
                  + tuple(fbk.LAYER_PROFILE_FRONTIER)):
            if p not in seen:
                seen.add(p)
                pairs.append(p)
        out["profiles"] = profile_convergence(genes, pairs, fillet=fillet)
        out["profiles"]["shipped_pair"] = [float(fbk.LAYER_ENTRY_SLOPE),
                                           float(fbk.LAYER_END_OFFSET)]
        out["profiles"]["genome_robust_pair"] = [float(fbk.GENOME_ROBUST_ENTRY),
                                                 float(fbk.GENOME_ROBUST_END)]
        # Which of the priced pairs were CANDIDATES rather than controls.  Carried so a
        # reader (and a test) can tell "holds the band" from "holds the band AND clears
        # the barrier" without re-deriving the candidate set from the other study.
        out["profiles"]["candidates"] = [list(p) for p in fbk.LAYER_PROFILE_CANDIDATES]
    return out


def _print(rep):
    fil = rep.get("fillet")
    tag = ("unfilleted" if fil is None else
           "FILLETED (genome radii)" if fil is True else
           f"FILLETED R = {fil[0]:.4f} / {fil[1]:.4f} mm")
    print("\n" + "=" * 78)
    print(f"  JUNCTION CORNER SINGULARITY — {rep['genome']} — {tag}")
    print("=" * 78)
    w = rep["williams_checks"]
    print(f"  Williams check: crack (360 deg) -> {w['crack_360deg']:.6f} (exact 0.5), "
          f"270 deg -> {w['textbook_270deg']:.4f} (textbook 0.5445)")
    print(f"\n  {'probe':<11}{'wedge deg':>11}{'gap um':>9}{'kind':>13}{'lambda_W':>10}")
    for c, d in rep["williams"].items():
        lam = f"{d['lambda']:>10.4f}" if "lambda" in d else f"{'-':>10}"
        print(f"  {c:<11}{d['wedge_deg']:>11.2f}"
              f"{d['node_gap_mm'] * 1000.0:>9.1f}{d['kind']:>13}{lam}")

    print(f"\n  TEST A — radial decay at the finest rung "
          f"(binned max over theta, inner decade)")
    fine = rep["rungs"][-1]
    for c, e in fine["corners"].items():
        rd = e.get("radial_decay")
        if not rd:
            print(f"    {c:<10} no fit")
            continue
        print(f"    {c:<10} {rd['n_bins']} bins {rd['window_mm'][0]:.4f}-"
              f"{rd['window_mm'][1]:.4f} mm   slope {rd['slope']:+.4f} -> "
              f"lambda {rd['lambda']:.4f}   corr {rd['corr']:+.4f}")

    print(f"\n  TEST B — does the peak DIVERGE under refinement?")
    print(f"    {'quantity':<16}{'peak MPa across the ladder':<44}"
          f"{'d log/d log h':>14}{'lambda':>9}  {'diverges':<6}{'dN/dN-1':>8}")
    for k, d in rep["divergence"].items():
        peaks = "  ".join(f"{x:7.2f}" for x in d["peak_mpa"])
        r = d.get("increment_ratio_finest")
        rs = f"{r:>+8.3f}" if r is not None else f"{'-':>8}"
        est = d.get("settled_estimate_mpa")
        note = f"settling -> {est:.2f} MPa" if est is not None else (
            "settling" if d.get("settling") else "")
        print(f"    {k:<16}{peaks:<44}{d['slope_finest3']:>+14.4f}"
              f"{d['lambda_from_slope']:>9.4f}  {str(d['diverges']):<6}"
              f"{rs}  {note}")
    print("    (a convergent peak gives a slope of ~0; these grow by "
          f"{max(d['growth_over_ladder'] for d in rep['divergence'].values()):.2f}x "
          "across the ladder.  The last column is the ratio of the two finest successive "
          "differences:\n     a divergent series holds it near 1, a settling one decays.)")

    print(f"\n  WHERE THE WHEEL'S GLOBAL MAXIMUM ACTUALLY IS")
    for r in rep["rungs"]:
        g = r["global_peak"]
        print(f"    {r['config']:<8}{r['global_max_vm_mpa']:>9.2f} MPa   "
              f"{g['nearest_probe']:<11} + {g['distance_mm'] * 1000.0:8.1f} um")

    print(f"\n  DEFLECTION ACROSS THE LADDER")
    for r in rep["rungs"]:
        print(f"    {r['config']:<8}{r['axle_drop_mm']:>12.6f} mm")

    if "continuity" in rep:
        c = rep["continuity"]
        print(f"\n  CONTROL — IS THIS THE SAME WHEEL?  R -> 0 against the unfilleted "
              f"mesh at `{c['config']}`")
        print(f"    unfilleted           {c['unfilleted_axle_drop_mm']:>12.6f} mm")
        for row in c["rows"]:
            if not row["built"]:
                print(f"    R = {row['R_mm']:<8.4f}     REFUSED  {row['why']}")
                continue
            print(f"    R = {row['R_mm']:<8.4f}     {row['axle_drop_mm']:>12.6f} mm  "
                  f"{row['rel_to_unfilleted']:+8.2%}")
        print(f"    genome's own pair    {c['shipped_fillet_axle_drop_mm']:>12.6f} mm  "
              f"{c['shipped_rel_to_unfilleted']:+8.2%}")

    if "profiles" in rep:
        pr = rep["profiles"]
        ship, robust = tuple(pr["shipped_pair"]), tuple(pr["genome_robust_pair"])
        print("\n  WHAT EACH CANDIDATE LAYER PROFILE COSTS THE DEFLECTION'S CONVERGENCE "
              "(FILLET_PLAN PART 16)")
        print(f"    the band is +-{pr['band_pct']:.1f}% over "
              + "..".join(pr["ladder"]) + ", which is what PART 12 checked 0.141% against")
        print(f"      {'entry':>6s} {'end':>5s} " + "".join(
            f"{n:>11s}" for n in pr["ladder"])
            + f"{'spread':>9s} {'in band':>8s}   note")
        for row in pr["rows"]:
            pair = (row["entry"], row["end"])
            note = ("<- shipped" if pair == ship else
                    "<- genome-robust (§54)" if pair == robust else "")
            if row["spread_pct"] is None:
                print(f"      {row['entry']:+6.2f} {row['end']:5.2f} "
                      f"{'REFUSED: ' + (row['why'] or '')[:40]:>50s}   {note}")
                continue
            print(f"      {row['entry']:+6.2f} {row['end']:5.2f} "
                  + "".join(f"{d:11.6f}" for d in row["axle_drop_mm"])
                  + f"{row['spread_pct']:8.3f}% {str(row['inside_band']):>8s}   {note}")
        ok = [r for r in pr["rows"] if r["inside_band"]]
        print(f"    {len(ok)}/{len(pr['rows'])} candidates hold the band.")
        by_entry, by_end = {}, {}
        for r in pr["rows"]:
            if r["spread_pct"] is not None:
                by_entry.setdefault(r["entry"], []).append(r["spread_pct"])
                by_end.setdefault(r["end"], []).append(r["spread_pct"])
        print("    NEITHER VARIABLE ALONE PREDICTS IT, which is why the whole candidate "
              "set had to be priced:")
        print(f"      {'entry':>6s}   spread over the ends priced")
        for e in sorted(by_entry, reverse=True):
            print(f"      {e:+6.2f}   "
                  + ", ".join(f"{x:.3f}%" for x in sorted(by_entry[e])))
        print(f"      {'end':>6s}   spread over the entries priced")
        for n in sorted(by_end):
            print(f"      {n:6.2f}   "
                  + ", ".join(f"{x:.3f}%" for x in sorted(by_end[n])))
        print("    the failing set is the MIDDLE of the space — a short end with an entry "
              "steep enough to matter —")
        print("    and it covers almost all of the barrier-clearing region.  Almost.")
        clears = [r for r in pr["rows"] if r["inside_band"] and
                  (r["entry"], r["end"]) in [tuple(p) for p in pr["candidates"]]]
        if clears:
            for r in clears:
                print(f"    THE TWO-OBJECTIVE PROFILE EXISTS: entry {r['entry']:+.2f}, "
                      f"end {r['end']:.2f} — spread {r['spread_pct']:.3f}%, inside the "
                      f"band and BETTER than the shipped pair's.")
            print("    it is one cell of fourteen, and it is the one a rank cut-off over "
                  "the ridge would have")
            print("    dropped: it has the LOWEST genome-box floor of the fourteen that "
                  "clear the barrier.")
        else:
            print("    and no candidate holds the band: the two-objective profile does "
                  "not exist on this grid.")
    print("=" * 78 + "\n")


def parse_fillet(text):
    """`--fillet` as `none`, `genome`, or an explicit `R_hub,R_rim` pair, in mm.

    `genome` rather than `true` because that is what `build_wheel(fillet=True)` means —
    genes 12 and 13 — and a flag reading `--fillet true` invites the reading "on, at
    whatever the default radius is", which there is no such thing as.
    """
    t = (text or "none").strip().lower()
    if t in ("none", "off", ""):
        return None
    if t in ("genome", "true", "on"):
        return True
    parts = [p for p in t.replace(",", " ").split() if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--fillet takes `none`, `genome`, or `R_hub,R_rim` in mm; got {text!r}")
    return tuple(float(p) for p in parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", default=GENOME)
    ap.add_argument("--ladder", default=",".join(LADDER))
    ap.add_argument("--fillet", type=parse_fillet, default=None,
                    help="none (default) | genome | R_hub,R_rim in mm")
    ap.add_argument("--continuity", default=None,
                    help="config name for the R -> 0 control; filleted runs only")
    ap.add_argument("--profiles", action="store_true",
                    help="price each candidate layer profile's deflection convergence "
                         "(FILLET_PLAN PART 16); filleted runs only, ~80 s")
    ap.add_argument("--out", default=os.path.join(HERE, "study_corner_singularity.json"))
    args = ap.parse_args()
    rep = run(args.genome, tuple(args.ladder.split(",")),
              fillet=args.fillet, continuity=args.continuity, profiles=args.profiles)
    _print(rep)
    with open(args.out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
