"""Genome -> the picture, using the project's own geometry kernel.

EXACT, NOT AN APPROXIMATION, AND THAT IS THE WHOLE POINT.  Every vertex here comes out of
the same three functions the mesher and the STEP exporter build the real part from:

    generate_bezier_centerline -> thicken_3taper_curve -> place_sector

so a preview that looks wrong IS a wheel that is wrong.  A GUI that re-drew the spoke from
its own idea of a Bezier would be a second implementation to keep in step, and the first
time it drifted it would be lying about a part someone is about to print.  `wheel_fea`'s
own poster figure composes these same calls; what is new here is only that it is served
per-request instead of written once at the end of a GA run.

ONE SECTOR ON THE WIRE, TWELVE ON THE SCREEN.  The wheel is twelve rotations of sector 0 --
that is how `place_sector` builds it and how `wheel_wheel` assembles the mesh -- so the
payload carries one outline and a spoke count, and the browser repeats it with a
transform.  At full `N_CURVE_PTS` the whole wheel is 14,400 vertices; one sector is 1,200,
and at preview resolution it is a couple of hundred.  That is the difference between a
slider that redraws and a slider that stutters.

RESOLUTION IS A PREVIEW KNOB AND NOTHING ELSE.  `num_points` here only sets how finely the
centerline is sampled for DRAWING.  It is not `N_CURVE_PTS`, it does not reach the mesh,
the objective, or anything a run records -- the default is lowered from 600 purely because
a wheel drawn 700 px wide cannot show 600 points per edge.

The fillet arcs are the real ones: `ring_junction_fillets` finds the junction by crossing
the outline against the ring circle rather than assuming where the offset edges end, and
its docstring says it is built to agree with what the exporter constructs in OCC.
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import wheel_fea as W                                             # noqa: E402
import wheel_genome as wg                                         # noqa: E402
import wheel_geometry as G                                        # noqa: E402

PREVIEW_POINTS = 160          # see the module docstring: drawing only


def bounds():
    """The gene box, READ INSIDE THE CALL and never snapshotted.

    `set_min_wall` and `set_cy_bound` rewrite `GENE_SPACE` in place and refresh three
    derived arrays from it; a module-level copy taken at import would keep serving the
    old box to every slider after a run moved the floor.
    """
    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    return [{"name": n, "low": float(lo), "high": float(hi)}
            for n, lo, hi in zip(wg.GENE_NAMES, low, high)]


_RIM_OUTER = None


def _rim_outer_mm():
    """The Ø100 ground-contact radius, which is the one drawn constant `wheel_fea` does
    not hold: it lives in `wheel_wheel` and in the exporter, because it belongs to the
    ASSEMBLY rather than to the beam model. Imported lazily and cached so this module
    stays importable with numpy alone -- `wheel_wheel` pulls in jax."""
    global _RIM_OUTER
    if _RIM_OUTER is None:
        import wheel_wheel
        _RIM_OUTER = float(wheel_wheel.RIM_OUTER_RADIUS_MM)
    return _RIM_OUTER


def constants():
    return {"n_spokes": int(W.NUMBER_OF_SPOKES),
            "hub_radius_mm": float(W.HUB_RADIUS_MM),
            "rim_radius_mm": float(W.RIM_RADIUS_MM),
            "rim_outer_radius_mm": _rim_outer_mm(),
            "span_mm": float(W.HUB_RIM_SPAN_MM),
            "face_width_mm": float(W.SPOKE_WIDTH_MM)}


def _round(a, nd=3):
    """Micron precision on the wire. The printer's nozzle is 400 um."""
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def outline(genes, num_points=PREVIEW_POINTS):
    """One sector's geometry, plus everything needed to draw the other eleven."""
    t0 = time.perf_counter()
    g = dict(genes)
    curve, ctrl = W.generate_bezier_centerline(
        g["cx1"], g["cy1"], g["cx2"], g["cy2"], g["cx3"], g["cy3"], g["cx4"], g["cy4"],
        span_mm=W.HUB_RIM_SPAN_MM, num_points=num_points)
    poly = W.thicken_3taper_curve(curve, g["t0"], g["t1"], g["t2"], g["t3"])
    placed = W.place_sector(poly, W.HUB_RADIUS_MM, 0.0)
    centre = W.place_sector(curve, W.HUB_RADIUS_MM, 0.0)
    handles = W.place_sector(ctrl, W.HUB_RADIUS_MM, 0.0)

    # The fillets the exporter will actually try to build, at both rings.
    fillets = []
    for ring, radius, label in ((W.HUB_RADIUS_MM, g["R_hub"], "hub"),
                                (W.RIM_RADIUS_MM, g["R_rim"], "rim")):
        try:
            for arc in W.ring_junction_fillets(placed, ring, radius):
                # Each arc comes back as a (xs, ys) pair of arrays; transpose so every
                # polyline on the wire has the same (n, 2) shape as `sector`.
                fillets.append({"ring": label,
                                "points": _round(np.asarray(arc).T)})
        except Exception:
            # A degenerate genome can have no crossing to fillet. The outline is still
            # the truth about the shape; the arcs are an annotation on it.
            pass

    # The taper t(s), which is what t0..t3 actually mean.
    s = np.linspace(0.0, 1.0, 100)
    taper = W.thickness_at_arc_length(s, g["t0"], g["t1"], g["t2"], g["t3"])

    # A saturated gene is a boundary optimum, and which kind matters: a pinned `t*`
    # means the printable wall floor is the active constraint, which is a real answer;
    # a pinned `cy*` looks like the box is too tight and measurably is not.
    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    vec = wg.genes_to_vector(g)
    saturated = {name: at for name, _v, at, _b in wg.bound_saturation(vec, low, high)}

    return {
        "sector": _round(placed),
        "centerline": _round(centre),
        "handles": _round(handles),
        "fillets": fillets,
        "taper": {"s": _round(s, 4), "t": _round(taper, 4)},
        "saturated": saturated,
        "constants": constants(),
        "build_ms": round((time.perf_counter() - t0) * 1e3, 3),
    }


def load_genes(path=None):
    import project_paths as PP
    with open(path or PP.BEST_SOLUTION) as fh:
        import json
        rec = json.load(fh)
    return rec["genes"]
