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
    """
    b = WW.sector_blocks(genes, cfg)
    out = {}
    for ring, label in (("hub_junction", "hub"), ("rim_junction", "rim")):
        out[f"{label}:P_t"] = np.asarray(b[ring][0, 0], dtype=float)
        out[f"{label}:P_c"] = np.asarray(b[ring][-1, 0], dtype=float)
    return out


def measured_wedge_deg(mesh, point, order):
    """The material wedge at a corner, summed from the mesh's own incident elements.

    Sum of the interior angles at the corner node over every element that owns it.  A node
    interior to the material sums to 360; a smooth boundary node to 180; a re-entrant
    corner to something between 180 and 360.  Midside nodes contribute a straight 180 and
    are skipped, since a Q9 midside cannot be a corner of the geometry.

    This measures the MESHED body.  That is the point: the exported solid is filleted at
    these corners and has no wedge to measure.
    """
    xy = mesh.coords
    nid = int(np.argmin(np.linalg.norm(xy - point, axis=1)))
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

def solve_field(genes, cfg):
    mesh = WW.build_wheel(genes, cfg)
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


def run(genome=GENOME, ladder=LADDER):
    genes = load_genes(genome)
    n_sp = WW.NUMBER_OF_SPOKES
    out = {"genome": genome, "ladder": list(ladder), "n_spokes": n_sp,
           "probe_radius_mm": PROBE_RADIUS_MM, "rungs": [], "williams": {}}

    finest = WW.get_config(ladder[-1])
    pts = corner_points(genes, finest)

    for name in ladder:
        cfg = WW.get_config(name)
        mesh, res, xy, vm = solve_field(genes, cfg)
        rec = {"config": name, "n_elements": int(mesh.n_elements),
               "n_nodes": int(mesh.n_nodes),
               "h": 1.0 / math.sqrt(mesh.n_elements),
               "axle_drop_mm": float(res["axle_drop_mm"]),
               "global_max_vm_mpa": float(vm.max()), "corners": {}}
        for cname, p0 in corner_points(genes, cfg).items():
            centre, peak = loaded_copy(p0, xy, vm, n_sp)
            entry = {"peak_vm_mpa": peak}
            entry.update(measured_wedge_deg(mesh, p0, cfg.order))
            if name == ladder[-1] and centre is not None:
                entry["radial_decay"] = radial_decay(xy, vm, centre)
            rec["corners"][cname] = entry
        out["rungs"].append(rec)
        print(f"  {name:<7} elem {rec['n_elements']:>6}  h {rec['h']:.6f}  "
              f"drop {rec['axle_drop_mm']:.5f}  global vm max "
              f"{rec['global_max_vm_mpa']:8.2f} MPa", flush=True)

    # Williams, from the wedge angles the FINEST mesh actually has.
    fine_rec = out["rungs"][-1]
    for cname, e in fine_rec["corners"].items():
        out["williams"][cname] = {
            "wedge_deg": e["wedge_deg"],
            "lambda": williams_lambda(e["wedge_deg"]),
            "re_entrant": bool(e["wedge_deg"] > 180.0)}
    out["williams_checks"] = {"crack_360deg": williams_lambda(360.0),
                              "textbook_270deg": williams_lambda(270.0)}

    # TEST B: log-log slope of the peak against h.
    lh = np.log([r["h"] for r in out["rungs"]])
    out["divergence"] = {}
    series = {"global_max_vm": [r["global_max_vm_mpa"] for r in out["rungs"]]}
    for cname in pts:
        series[cname] = [r["corners"][cname]["peak_vm_mpa"] for r in out["rungs"]]
    for k, v in series.items():
        v = np.array(v, dtype=float)
        if (v <= 0).any():
            continue
        s_all = float(np.polyfit(lh, np.log(v), 1)[0])
        s_fin = float(np.polyfit(lh[-3:], np.log(v[-3:]), 1)[0])
        out["divergence"][k] = {
            "peak_mpa": [float(x) for x in v],
            "slope_all": s_all, "slope_finest3": s_fin,
            "lambda_from_slope": s_fin + 1.0,
            "growth_over_ladder": float(v[-1] / v[0]),
            # A convergent peak gives ~0.  The threshold is deliberately loose: this
            # separates "diverges" from "converges", not one lambda from another.
            "diverges": bool(s_fin < -0.15)}
    return out


def _print(rep):
    print("\n" + "=" * 78)
    print(f"  JUNCTION CORNER SINGULARITY — {rep['genome']}")
    print("=" * 78)
    w = rep["williams_checks"]
    print(f"  Williams check: crack (360 deg) -> {w['crack_360deg']:.6f} (exact 0.5), "
          f"270 deg -> {w['textbook_270deg']:.4f} (textbook 0.5445)")
    print(f"\n  {'corner':<10}{'wedge deg':>11}{'re-entrant':>12}{'lambda_W':>10}")
    for c, d in rep["williams"].items():
        print(f"  {c:<10}{d['wedge_deg']:>11.2f}{str(d['re_entrant']):>12}"
              f"{d['lambda']:>10.4f}")

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
          f"{'d log/d log h':>14}{'lambda':>9}  diverges")
    for k, d in rep["divergence"].items():
        peaks = "  ".join(f"{x:7.2f}" for x in d["peak_mpa"])
        print(f"    {k:<16}{peaks:<44}{d['slope_finest3']:>+14.4f}"
              f"{d['lambda_from_slope']:>9.4f}  {d['diverges']}")
    print("    (a convergent peak gives a slope of ~0; these grow by "
          f"{max(d['growth_over_ladder'] for d in rep['divergence'].values()):.2f}x "
          "across the ladder)")
    print("=" * 78 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", default=GENOME)
    ap.add_argument("--ladder", default=",".join(LADDER))
    ap.add_argument("--out", default=os.path.join(HERE, "study_corner_singularity.json"))
    args = ap.parse_args()
    rep = run(args.genome, tuple(args.ladder.split(",")))
    _print(rep)
    with open(args.out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
