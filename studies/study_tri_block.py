#!/usr/bin/env python
"""
=============================================================================
THE RIM TRI-BLOCK, BUILT — PLAN.md §51's PROBE TURNED INTO A MEASUREMENT
=============================================================================

WHAT THIS ANSWERS.  At the faithful rim — `uncap` blend 0.0, the geometry the exporter's
`_embed` actually uses — the rim junction stops being a quadrilateral.  A tangent
continuation is by definition smooth, so the corner at `far_end` opens to ~180 degrees
and the region becomes a curvilinear TRIANGLE.  A four-sided structured block on a
three-sided region always carries that vertex, and `min_scaled_jacobian` collapses
0.782505 -> 0.007208 (§37, UNCAP_PLAN Step 2).  The fix named there and never built is
the three-quad Y-partition.  §37 priced it and shelved it on two clauses; §51 re-priced
both with a scratch probe, said both were wrong, and filed the probe AS a probe with the
instruction "do not quote these numbers until this exists".  This is what makes them
quotable, and its numbers supersede §51's.

THE UNIT, quoting §51 exactly: "build all twelve blocks and all their seams, sweep the
free count and the interior point, and report validity and seam closure at both configs."

  §37 CLAUSE 1, "it needs PARTIAL-EDGE SEAMS".  The Y-partition splits all three sides
  and two of them are shared — the ring arc with `rim_band_weld`, the end cross-section
  with `spoke`.  That is a partial-edge seam ONLY IF the neighbours may not be split, and
  they may: splitting `rim_band_weld` in theta and the `spoke` along a j-line cascades
  once into `hub_junction` and stops.  SEVEN blocks become TWELVE, whole-edge throughout.
  Measured here as seventeen seams, counted and closed.

  §37 CLAUSE 2, "forced 1-element strips".  The algebra is right and its input was not.
  Writing the three sides as A (the arc, `n_weld`), B (the free side) and C (the cross
  section, `n_thick`), matching opposite sides of the three quads forces

      a1 = b2      a2 = c1      c2 = b1      hence   a1 = (A + B - C) / 2

  §37 read B off the block it was replacing (8 elements at `coarse`) and got 7x1, 3x1,
  3x7 — the strip is real at that B.  **B is the FREE side and its count is free.**  The
  admissible set is every B with |A - C| < B < A + C and A + B - C even, and this file
  enumerates it rather than picking one.

WHAT IS SWEPT, AND WHY THOSE TWO.  The free count B, over its whole admissible set, and
the interior point X, over a barycentric grid on the triangle.  Nothing else is free: the
A side's node distribution is the ring block's (uniform in theta) and the C side's is the
spoke's (uniform in eta), and a construction that re-distributed either would be changing
its neighbour rather than partitioning its own region.  The internal spokes are straight
and nothing is smoothed, so every number here is a FLOOR — an elliptic interior solve
could only raise it, and `study_fillet_block.winslow` is the one that would.

THE RULE THAT PICKS THE CELL, so a re-run picks the same one for the same reason: the
argmax of the WORST tri block's min scaled Jacobian over the published grid.  The grid is
published in full for the same reason §48's was — so that "a choice on a plateau" and "a
tuned point" are told apart by looking rather than by assertion.

WHY THIS IS NOT A SECTION OF `study_fillet_block.py`.  That file measures the FILLETED
blocking at the SHIPPED blend; this measures the UNFILLETED blocking at the FAITHFUL one.
They share no construction, no control and no verdict, and §51's `sweep_uncap` — which
lives there and stays there — is what handed this file its question.

WHAT THIS DOES NOT DO.  It does not wire anything into `build_wheel`, it promotes
nothing, and `best_solution.json` is untouched.  The tri-block is a MEASUREMENT of a
construction here, exactly as the eleven-block filleted sector was in §48 before §50
wired it; whether the faithful rim ships is a separate decision with its own baseline,
because it moves the mesh under every genome.

EXIT STATUS follows `make filletblock`: nonzero ONLY if a self-check fails — the
controls, the area identity, the seam closure, or the algebra reproducing §37's own
numbers.  Never on a characterisation finding about a block, which is what this exists to
report.
=============================================================================
"""

import argparse
import json
import math
import os
import time

import numpy as np

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import _gate_guard
import wheel_genome as wg
import wheel_wheel as WW
import study_fillet_block as fbk

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIGS = ("coarse", "medium")

# The faithful rim.  The hub entry is `_embed`'s own blend AND the well-shaped choice, so
# it is the shipped default and is not what this arc is about; the rim entry is the one
# the tree refuses today.  Passed explicitly rather than as `True` so that a future change
# to `UNCAP_DEFAULT` cannot silently re-aim this file at a different geometry.
FAITHFUL = (True, 0.0)

# The barycentric grid for the interior point, in weights on (P_t, Q, B*).  Two of the
# three are swept and the third closes to 1, so this is a triangle of cells rather than a
# square; `X_GRID_N` sets the spacing on each axis.
X_GRID_N = 19
X_GRID_LO, X_GRID_HI = 0.02, 0.96

# Order-2 elements everywhere in this tree, and the splits have to land on element
# boundaries.  Read from the config rather than assumed, and asserted.
SEAM_TOL_MM = 1.0e-9

# How finely the free side's own curve is sampled before it is re-distributed.  The side
# is a far-flank arc followed by a straight closing segment, and the node count on it is
# what is being swept, so the underlying curve has to be resolved far past any of them.
B_CURVE_SAMPLES = 4001
B_SEGMENT_SAMPLES = 401

# `MIN_SJ_TARGET` is the barrier `wheel_objective` puts on the mesh, and the reason the
# faithful rim is refused today is that 0.0072 sits under it.  Imported rather than
# repeated: a floor quoted from memory in a study is a floor that drifts from the one the
# optimizer enforces.
try:
    import wheel_objective as WO
    MIN_SJ_TARGET = float(WO.MIN_SJ_TARGET)
except Exception:                                    # pragma: no cover - import guard
    MIN_SJ_TARGET = 0.2


def load_genes(path):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


# ---------------------------------------------------------------------------
# THE PARTITION ALGEBRA
# ---------------------------------------------------------------------------

def splits(A, B, C):
    """The Y-partition's six side counts for one (A, B, C), or `None` if there is none.

    `A` is the ring arc's element count (`n_weld`), `C` the end cross-section's
    (`n_thick`), `B` the free side's — the one that is chosen rather than inherited.
    Returns the six pieces as elements:

        a1  P_t -> M_A along the arc          a2  M_A -> Q along the arc
        b1  Q   -> M_B along the free side    b2  M_B -> B* along the free side
        c1  B*  -> M_C along the cross        c2  M_C -> P_t along the cross

    and the three internal spokes follow: |M_A X| = c2, |M_B X| = a2, |M_C X| = a1.
    `None` means either the parity fails (a half-element split) or one piece would be
    zero, which is a degenerate partition rather than a bad one.
    """
    if (A + B - C) % 2:
        return None
    a1 = (A + B - C) // 2
    a2, c2 = A - a1, B - a1
    c1 = C - c2
    if min(a1, a2, c1, c2) < 1:
        return None
    return {"a1": a1, "a2": a2, "b1": c2, "b2": a1, "c1": c1, "c2": c2}


def admissible(A, C):
    """Every free count `B` the Y-partition admits, in increasing order."""
    return tuple(B for B in range(1, A + C) if splits(A, B, C) is not None)


def algebra_section(configs):
    """The whole admissible set at each config, with §37's own choice marked.

    This is the clause-2 re-pricing, and it is enumerated rather than argued: the strip
    §37 reports is real AT ITS OWN `B`, and the table shows both that and the B's that do
    not have one.
    """
    out = {}
    for name in configs:
        cfg = WW.get_config(name)
        A, C = cfg.n_weld, cfg.n_thick
        rows = []
        for B in admissible(A, C):
            sp = splits(A, B, C)
            shapes = [[sp["a1"], sp["c2"]], [sp["a2"], sp["b1"]], [sp["b2"], sp["c1"]]]
            rows.append({
                "B": B, **{k: int(v) for k, v in sp.items()},
                "shapes": shapes,
                "min_side_elements": int(min(min(s) for s in shapes)),
                "has_one_element_strip": bool(min(min(s) for s in shapes) == 1),
                # §37 quoted `B = 8` at `coarse` and never said where it came from.  The
                # cell is marked by ITS OWN ARITHMETIC -- 7/3, 1/7, 3/1 -- rather than by
                # a formula reverse-engineered from one number, because reproducing what
                # §37 published is the check and a guessed formula would not be one.
                "is_section_37_choice": bool(
                    name == "coarse" and [sp["a1"], sp["a2"], sp["c1"], sp["c2"]]
                    == [7, 3, 3, 1])})
        out[name] = {
            "A_arc_elements": int(A), "C_cross_elements": int(C),
            "admissible_B": [int(b) for b in admissible(A, C)],
            "rows": rows,
            "strip_free_B": [r["B"] for r in rows if not r["has_one_element_strip"]]}
    return out


# ---------------------------------------------------------------------------
# THE REGION
# ---------------------------------------------------------------------------

def _resample_arclength(P, n):
    """`n` points spaced uniformly in ARC LENGTH along the polyline `P`."""
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    s = s / s[-1]
    t = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], axis=1)


def _turn_deg(a, b, c):
    """The interior turn at `b` on the path a -> b -> c, in degrees; 180 is straight."""
    u, v = np.asarray(b) - np.asarray(a), np.asarray(c) - np.asarray(b)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return 180.0 - math.degrees(math.acos(float(np.clip(u @ v, -1.0, 1.0))))


def region(genes, cfgname, blend=0.0):
    """The rim junction's three sides at `blend`, plus everything the partition needs.

    Built from `sector_blocks` itself rather than re-derived, so that the spoke's end row
    IS the cross section and the arc IS the one the ring block sees.  A second derivation
    here is exactly the drift `study_fillet_block`'s docstring is about.
    """
    cfg = WW.get_config(cfgname)
    span = WW.HUB_RIM_SPAN_MM
    orientation = WW.flank_orientation(genes, cfg, span_mm=span)
    eta_rim = float(orientation[1])
    rim_inner = WW.rim_inner_radius(span)
    base = WW.sector_blocks(genes, cfg, uncap=(True, blend), orientation=orientation)
    thetas = base.pop("_thetas")
    th_t, th_q = (float(x) for x in thetas["rim_junction"])

    sample, s_dense = WW.global_sampler(genes, cfg, span_mm=span)
    s_hub, s_rim = WW.junction_stations(sample, s_dense, orientation, rim_inner)

    spoke = np.asarray(base["spoke"], float)
    cross = spoke[-1][::-1] if eta_rim > 0 else spoke[-1]     # P_t -> B*
    P_t, B_star = cross[0], cross[-1]
    Q = np.array([rim_inner * math.cos(th_q), rim_inner * math.sin(th_q)])

    # The free side: the FAR flank from the junction station out to the ring station,
    # then the straight closing segment onto the ring circle.  At blend 0 the second
    # piece is the first one's own tangent continuation, which is why the two meet at
    # ~180 degrees -- the whole reason the quad block degenerates.
    sd = np.linspace(float(s_rim), 1.0, B_CURVE_SAMPLES)
    flank = sample(sd, np.zeros_like(sd) - eta_rim)
    far_end = flank[-1]
    B_dense = np.concatenate(
        [flank, WW._lerp_points(far_end, Q, B_SEGMENT_SAMPLES, np)[1:]])

    return {"cfg": cfg, "name": cfgname, "blend": float(blend),
            "orientation": orientation, "eta_rim": eta_rim, "rim_inner": rim_inner,
            "base": base, "thetas": thetas, "th_t": th_t, "th_q": th_q,
            "cross": cross, "B_dense": B_dense, "spoke": spoke,
            "P_t": P_t, "Q": Q, "B_star": B_star, "far_end": far_end,
            "sample": sample, "s_rim": float(s_rim)}


def _bow_mm(reg):
    """How far the arc side departs from its own chord, at its furthest, in mm.

    The straight Y cuts CHORDS across the region, so the quantity that decides whether a
    chord stays inside it is how far the region's long side bows away from one.  Reported
    against the cross section's length in `bow_over_width`, because a 3 mm bow across a
    6 mm region and a 0.2 mm bow across the same region are not the same geometry -- and
    that ratio, not the arc span, is what separates the genomes the straight Y folds on
    from the ones it does not.
    """
    n = reg["cfg"].nn(reg["cfg"].n_weld) * 4
    A = WW.arc_points(reg["rim_inner"], reg["th_t"], reg["th_q"], n)
    d = reg["Q"] - reg["P_t"]
    d = d / np.linalg.norm(d)
    off = A - reg["P_t"]
    return float(np.abs(off[:, 0] * d[1] - off[:, 1] * d[0]).max())


def region_report(reg):
    """The three vertices, the three side lengths, and the vertex that is not one.

    The `far_end` turn is the whole finding of UNCAP_PLAN Step 2 restated as a number: at
    blend 0 it is within a degree of straight, so the fourth vertex a quad block needs
    does not exist on this region.
    """
    B_dense = reg["B_dense"]
    k = B_CURVE_SAMPLES - 1
    arc_len = abs(reg["th_q"] - reg["th_t"]) * reg["rim_inner"]
    cross_len = float(np.linalg.norm(np.diff(reg["cross"], axis=0), axis=1).sum())
    bow = _bow_mm(reg)
    return {
        "bow_mm": bow,
        "bow_over_width": bow / cross_len,
        "P_t": [float(x) for x in reg["P_t"]],
        "Q": [float(x) for x in reg["Q"]],
        "B_star": [float(x) for x in reg["B_star"]],
        "far_end": [float(x) for x in reg["far_end"]],
        "arc_span_deg": float(math.degrees(abs(reg["th_q"] - reg["th_t"]))),
        "A_length_mm": float(arc_len),
        "B_length_mm": float(np.linalg.norm(np.diff(B_dense, axis=0), axis=1).sum()),
        "C_length_mm": cross_len,
        "turn_at_far_end_deg": float(_turn_deg(B_dense[k - 1], B_dense[k],
                                               B_dense[k + 1])),
        "wedge_at_P_t_deg": float(_turn_deg(reg["cross"][1], reg["P_t"], _arc_pt(reg, 1))),
        "wedge_at_Q_deg": float(_turn_deg(_arc_pt(reg, -2), reg["Q"], B_dense[-2])),
        "wedge_at_B_star_deg": float(_turn_deg(B_dense[1], reg["B_star"],
                                               reg["cross"][-2])),
    }


def _arc_pt(reg, i):
    n = reg["cfg"].nn(reg["cfg"].n_weld)
    return WW.arc_points(reg["rim_inner"], reg["th_t"], reg["th_q"], n)[i]


# ---------------------------------------------------------------------------
# THE TWELVE BLOCKS
# ---------------------------------------------------------------------------

TRI_BLOCKS = ("rim_tri_t", "rim_tri_q", "rim_tri_b")

TWELVE_BLOCK_ORDER = (
    "spoke_eta_lo", "spoke_eta_hi",
    "hub_junction_lo", "hub_junction_hi",
    "rim_tri_t", "rim_tri_q", "rim_tri_b",
    "hub_collar_weld", "hub_collar_free",
    "rim_band_weld_q", "rim_band_weld_t", "rim_band_free")

TWELVE_BLOCK_REGION = {
    "spoke_eta_lo": "spoke", "spoke_eta_hi": "spoke",
    "hub_junction_lo": "spoke", "hub_junction_hi": "spoke",
    "rim_tri_t": "spoke", "rim_tri_q": "spoke", "rim_tri_b": "spoke",
    "hub_collar_weld": "hub", "hub_collar_free": "hub",
    "rim_band_weld_q": "rim", "rim_band_weld_t": "rim", "rim_band_free": "rim"}


def _bent_spoke(straight, u, v, frac, p0, p1, q0, q1, bend):
    """One spoke, bent from a straight line toward the two sides it runs between.

    THE CURVED Y, AND WHY THIS PARTICULAR CURVE.  Each spoke is the OPPOSITE edge of two
    of the three quads, and in each of them it faces a piece of the region's own boundary:
    `sC` faces the arc in `rim_tri_t` and the free side in `rim_tri_b`, and so on round.
    Those two curves are `u` and `v` here, already running the spoke's own direction and
    -- because `splits` forces `a1 == b2`, `a2 == c1` and `c2 == b1` -- already carrying
    its exact node count, so no resampling enters and the blend is node-for-node.

    `frac` is where the spoke's own foot sits between them, as a fraction, so the blend
    `(1-frac)*u + frac*v` is the curve the region's two sides say should be there.  It
    does NOT pass through the spoke's endpoints, and the two linear terms are what pin it
    back to them -- which is the Coons correction, and is why the endpoints are exact for
    every `bend` and the three internal seams stay exact by construction rather than by
    tolerance.

    `bend = 0.0` returns `straight` UNTOUCHED, not merely equal to it, so every number this
    file published before the curve existed is reproduced bit for bit rather than
    approximately.
    """
    if bend == 0.0:
        return straight
    t = np.linspace(0.0, 1.0, straight.shape[0])[:, None]
    blend = (1.0 - frac) * u + frac * v
    d0 = straight[0] - ((1.0 - frac) * p0 + frac * p1)
    d1 = straight[-1] - ((1.0 - frac) * q0 + frac * q1)
    return straight + bend * (blend + (1.0 - t) * d0 + t * d1 - straight)


def tri_sector(reg, B, w, bend=0.0):
    """The twelve node grids of a faithful-rim sector 0, or `None` if `B` is inadmissible.

    THE THREE NEIGHBOURS ARE SLICED, NOT REBUILT.  `spoke`, `hub_junction` and
    `rim_band_weld` come out of `sector_blocks` and are cut at a node index, so their
    geometry is bit-for-bit what the tree builds today and every number below is a
    consequence of the PARTITION rather than of a second construction of the same thing.
    Only `rim_junction` is replaced.

    `w` is a barycentric weight triple on (P_t, Q, B*) for the interior point.  `bend` is
    how far the three spokes follow the region rather than cutting across it; 0.0 is the
    straight Y this file built first, and `_bent_spoke` has the construction.
    """
    cfg = reg["cfg"]
    A, C = cfg.n_weld, cfg.n_thick
    sp = splits(A, B, C)
    if sp is None:
        return None
    o = cfg.order
    a1, a2, b1, b2, c1, c2 = (sp[k] for k in ("a1", "a2", "b1", "b2", "c1", "c2"))

    th_t, th_q = reg["th_t"], reg["th_q"]
    th_M = th_t + (th_q - th_t) * (a1 / A)
    cross, P_t, Q, B_star = reg["cross"], reg["P_t"], reg["Q"], reg["B_star"]

    # --- the three sides, each split at its own midpoint ---------------------
    # The two SHARED sides keep their neighbour's distribution exactly: the arc is
    # uniform in theta because `polar_block` is, and the cross section is the spoke's own
    # end row.  Only the FREE side is redistributed, and only because its count is what
    # is being swept.
    A1 = WW.arc_points(reg["rim_inner"], th_t, th_M, o * a1 + 1)      # P_t -> M_A
    A2 = WW.arc_points(reg["rim_inner"], th_M, th_q, o * a2 + 1)      # M_A -> Q
    A1[0], A2[-1] = P_t, Q
    A2[0] = A1[-1]
    Bn = _resample_arclength(reg["B_dense"], o * B + 1)               # B* -> Q
    Bn[0], Bn[-1] = B_star, Q
    iC, iB = o * c2, o * b2
    M_A, M_C, M_B = A1[-1], cross[iC], Bn[iB]

    X = np.asarray(w, float) @ np.stack([P_t, Q, B_star]) / float(np.sum(w))
    lerp = lambda p, q, n: WW._lerp_points(p, q, o * n + 1, np)       # noqa: E731
    sA, sB, sC = lerp(M_A, X, c2), lerp(M_B, X, a2), lerp(M_C, X, a1)

    # The bend, if any.  Each spoke is named by the side its foot is on, and the two
    # curves it is blended toward are the ones it faces across its own two quads.
    sC = _bent_spoke(sC, A1, Bn[:iB + 1], c2 / (c1 + c2), P_t, B_star, M_A, M_B, bend)
    sA = _bent_spoke(sA, cross[:iC + 1], Bn[iB:][::-1], a1 / A, P_t, Q, M_C, M_B, bend)
    sB = _bent_spoke(sB, cross[iC:][::-1], A2[::-1], b2 / B, B_star, Q, M_C, M_A, bend)

    blocks = {}
    # The Y's three quads.  Every internal edge is passed as the SAME array to both of
    # its blocks, so the three seams that the partition itself creates are exact by
    # construction rather than by agreement -- and `sector_seams` still measures them,
    # because "exact by construction" is a claim about code that changes.
    blocks["rim_tri_t"] = WW.coons_patch(bottom=A1, top=sC, left=cross[:iC + 1], right=sA)
    blocks["rim_tri_q"] = WW.coons_patch(bottom=A2, top=sB[::-1], left=sA,
                                         right=Bn[iB:][::-1])
    blocks["rim_tri_b"] = WW.coons_patch(bottom=sC, top=Bn[:iB + 1], left=cross[iC:],
                                         right=sB[::-1])

    # --- the neighbours, cut ------------------------------------------------
    base = reg["base"]
    spoke = np.asarray(base["spoke"], float)
    n_th_nodes = spoke.shape[1]
    # `cross` runs P_t -> B*, and is the spoke's rim row reversed when the straddling
    # flank is at eta = +1.  `M_C` is at index `iC` along it, so the j-line the spoke is
    # cut on is that index mapped back into the spoke's own eta ordering.
    j_star = (n_th_nodes - 1 - iC) if reg["eta_rim"] > 0 else iC
    blocks["spoke_eta_lo"] = spoke[:, :j_star + 1]
    blocks["spoke_eta_hi"] = spoke[:, j_star:]

    hub = np.asarray(base["hub_junction"], float)
    eta_hub = float(reg["orientation"][0])
    j_hub = (n_th_nodes - 1 - j_star) if eta_hub > 0 else j_star
    blocks["hub_junction_lo"] = hub[:, :j_hub + 1]
    blocks["hub_junction_hi"] = hub[:, j_hub:]

    # The ring's weld block is laid out in INCREASING theta whichever way the junction's
    # arc runs, so the split node's index from its low end is `a2` elements when the arc
    # descends and `a1` when it ascends.
    # Its low-theta end is `P_t` when the junction's arc ASCENDS and `Q` when it
    # descends, so which half of the cut is which is a function of the genome and not a
    # constant.  The same trap §48 hit on the sector-closing seam's `dk`, and the reason
    # `_t` and `_q` are named for the CORNER they touch rather than for their index.
    weld = np.asarray(base["rim_band_weld"], float)
    ascends = th_t < th_q
    i_star = o * (a1 if ascends else a2)
    first, second = (("rim_band_weld_t", "rim_band_weld_q") if ascends else
                     ("rim_band_weld_q", "rim_band_weld_t"))
    blocks[first], blocks[second] = weld[:i_star + 1], weld[i_star:]

    for k in ("hub_collar_weld", "hub_collar_free", "rim_band_free"):
        blocks[k] = np.asarray(base[k], float)

    aux = {"splits": {k: int(v) for k, v in sp.items()}, "B": int(B),
           "w": [float(x) for x in w], "bend": float(bend),
           "X": [float(x) for x in X],
           "M_A": [float(x) for x in M_A], "M_B": [float(x) for x in M_B],
           "M_C": [float(x) for x in M_C], "theta_M": float(th_M),
           "j_star": int(j_star), "j_hub": int(j_hub), "i_star": int(i_star),
           "arc_ascends": bool(ascends), "first_weld": first, "last_weld": second}
    return {name: blocks[name] for name in TWELVE_BLOCK_ORDER}, aux


# ---------------------------------------------------------------------------
# THE SEAMS
# ---------------------------------------------------------------------------

def _side(block, side):
    b = np.asarray(block, float)
    return {"i0": b[0, :, :], "i1": b[-1, :, :],
            "j0": b[:, 0, :], "j1": b[:, -1, :]}[side]


def _rotate(P, k):
    a = k * math.radians(WW.SECTOR_DEG)
    c, s = math.cos(a), math.sin(a)
    return np.asarray(P, float) @ np.array([[c, s], [-s, c]])


def seam_table(reg, aux):
    """The SEVENTEEN seams of the twelve-block sector, `reverse` flags resolved.

    Eight of them are the existing table's, re-pointed at whichever half of a cut block
    the edge now belongs to.  Nine are new: three are the Y's own internal edges, three
    are the cuts through `spoke`, `hub_junction` and `rim_band_weld`, and three are the
    two shared sides' halves landing on their partners separately.  **Every one is a
    whole edge of both blocks it names**, which is §37 clause 1's whole question.
    """
    eta_hub, eta_rim = (float(x) for x in reg["orientation"])
    hub_t, hub_c = (float(x) for x in reg["thetas"]["hub_junction"])
    hub_ascends = hub_t < hub_c
    ascends = aux["arc_ascends"]
    # Which half of the cut spoke carries which flank.  `spoke_eta_lo` holds eta = -1.
    # At the rim the straddling flank -- and so `P_t`, and so the `rim_tri_t` side of the
    # cross section -- is at eta = `eta_rim`.
    t_side, b_side = (("spoke_eta_hi", "spoke_eta_lo") if eta_rim > 0
                      else ("spoke_eta_lo", "spoke_eta_hi"))
    # The hub junction's `left` edge is the spoke's hub row REVERSED when the straddling
    # flank there is at eta = +1, so the low-j half of one is the high-j half of the
    # other.  Getting this backwards leaves the node COUNTS agreeing at `coarse`, where
    # `n_thick` splits 2/2, and only the coordinates disagree -- which is why the seam
    # check reports the gap as well as the count.
    if eta_hub > 0:
        pairs = (("spoke_eta_lo", "hub_junction_hi"), ("spoke_eta_hi", "hub_junction_lo"))
    else:
        pairs = (("spoke_eta_lo", "hub_junction_lo"), ("spoke_eta_hi", "hub_junction_hi"))
    first, second = aux["first_weld"], aux["last_weld"]
    return (
        # The cut through the spoke: one whole j-line, the full span.
        ("spoke_eta_lo", "j1", "spoke_eta_hi", "j0", 0, False),
        # The spoke's HUB end cross-section, now in two pieces onto two blocks.
        (pairs[0][0], "i0", pairs[0][1], "i0", 0, eta_hub > 0),
        (pairs[1][0], "i0", pairs[1][1], "i0", 0, eta_hub > 0),
        # The cut through the hub junction, the same j-line continued.
        ("hub_junction_lo", "j1", "hub_junction_hi", "j0", 0, False),
        # The spoke's RIM end cross-section, onto two of the Y's three quads.
        (t_side, "i1", "rim_tri_t", "i0", 0, eta_rim > 0),
        (b_side, "i1", "rim_tri_b", "i0", 0, eta_rim > 0),
        # The Y's own three internal edges.
        ("rim_tri_t", "i1", "rim_tri_q", "i0", 0, False),
        ("rim_tri_t", "j1", "rim_tri_b", "j0", 0, False),
        ("rim_tri_q", "j1", "rim_tri_b", "i1", 0, False),
        # Each junction's arc onto its ring's weld block -- the rim's in two pieces.
        ("hub_junction_lo", "j0", "hub_collar_weld", "j1", 0, not hub_ascends),
        ("rim_tri_t", "j0", "rim_band_weld_t", "j0", 0, not ascends),
        ("rim_tri_q", "j0", "rim_band_weld_q", "j0", 0, not ascends),
        # The cut through the ring's weld block.
        (first, "i1", second, "i0", 0, False),
        # Weld block to free block within a ring.
        ("hub_collar_weld", "i1", "hub_collar_free", "i0", 0, False),
        (second, "i1", "rim_band_free", "i0", 0, False),
        # Free block to the NEXT sector's weld block -- the seams that close the 360.
        ("hub_collar_free", "i1", "hub_collar_weld", "i0", 1, False),
        ("rim_band_free", "i1", first, "i0", 1, False),
    )


def seams(blocks, table):
    """Every seam's node-count agreement and worst coordinate mismatch, in mm.

    A seam CLOSES when the two edges carry the same number of nodes AND those nodes
    coincide.  Both halves are reported: a count mismatch is a blocking error and a
    coordinate mismatch is a construction error, and they are not the same bug.
    """
    out = []
    for a, sa, b, sb, dk, rev in table:
        ea = _side(blocks[a], sa)
        eb = _rotate(_side(blocks[b], sb), dk)
        row = {"a": a, "side_a": sa, "b": b, "side_b": sb, "dk": dk, "reverse": rev,
               "n_a": int(ea.shape[0]), "n_b": int(eb.shape[0])}
        if ea.shape != eb.shape:
            row.update({"counts_agree": False, "max_gap_mm": None, "closes": False})
        else:
            gap = float(np.abs(ea - (eb[::-1] if rev else eb)).max())
            row.update({"counts_agree": True, "max_gap_mm": gap,
                        "closes": gap < SEAM_TOL_MM})
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# AREA — the identity that says the partition covers its own region
# ---------------------------------------------------------------------------

def block_area_mm2(grid):
    """The signed area of a block's boundary polygon, absolute, in mm^2."""
    g = np.asarray(grid, float)
    loop = np.concatenate([g[:, 0], g[-1, 1:], g[::-1, -1][1:], g[0, ::-1][1:-1]])
    x, y = loop[:, 0], loop[:, 1]
    return abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


# ---------------------------------------------------------------------------
# THE SWEEPS
# ---------------------------------------------------------------------------

def block_area_mm2s(blocks):
    """The area the Y's three quads tile between them, in mm^2."""
    return sum(block_area_mm2(blocks[k]) for k in TRI_BLOCKS)


def x_grid(n=X_GRID_N, lo=X_GRID_LO, hi=X_GRID_HI):
    """The barycentric weights swept, as (w_Pt, w_Q, w_Bstar) triples."""
    out = []
    for u in np.linspace(lo, hi, n):
        for v in np.linspace(lo, hi, n):
            if u + v >= 1.0 - lo:
                continue
            out.append((float(u), float(v), float(1.0 - u - v)))
    return tuple(out)


def cell(reg, B, w, bend=0.0):
    """One (B, X, bend) cell: the three quads' validity, and nothing else.

    The sweep runs on the Y alone rather than on the whole sector because the other nine
    blocks do not depend on either swept quantity -- `spoke`, `hub_junction` and
    `rim_band_weld` are CUT at a node index and their interiors never move.  That is
    checked once, in `self_checks`, rather than re-measured 900 times.
    """
    built = tri_sector(reg, B, w, bend)
    if built is None:
        return None
    blocks, aux = built
    q = {k: fbk.block_quality(np.asarray(blocks[k], float)) for k in TRI_BLOCKS}
    return {"B": int(B), "w": [float(x) for x in w], "bend": float(bend),
            "min_scaled_jacobian": min(v["min_scaled_jacobian"] for v in q.values()),
            "worst_block": min(q, key=lambda k: q[k]["min_scaled_jacobian"]),
            "non_positive_gauss_elements": sum(v["non_positive_gauss_elements"]
                                               for v in q.values()),
            "mixed_sign_cells": sum(v["mixed_sign_cells"] for v in q.values()),
            "all_valid": bool(all(v["valid"] for v in q.values())),
            "per_block": q, "shapes": aux["splits"]}


def refine(reg, B, w0, half=0.06, n=9):
    """A finer grid in a box round `w0`, so the argmax is not the grid's own spacing.

    The coarse sweep's cell is 0.052 wide in each weight and the valid fraction is small,
    so "the maximum" could be an artefact of where the grid points fell.  This re-sweeps
    a box a little wider than one coarse cell at a ninth of the spacing and reports both
    numbers; if they differ by more than the coarse grid's own resolution the surface is
    being under-resolved, and that is visible rather than assumed.
    """
    best = None
    for u in np.linspace(max(w0[0] - half, 0.005), w0[0] + half, n):
        for v in np.linspace(max(w0[1] - half, 0.005), w0[1] + half, n):
            if u + v >= 0.995:
                continue
            c = cell(reg, B, (float(u), float(v), float(1.0 - u - v)))
            if c is not None and c["all_valid"] and (
                    best is None or c["min_scaled_jacobian"] > best["min_scaled_jacobian"]):
                best = c
    return best


def winslow_column(reg, B, w):
    """The same cell with an elliptic interior solve on each quad, boundaries held.

    This is the "does a generated interior rescue it" control `study_fillet_block` runs on
    every candidate, and it is imported from there rather than re-written.  It is offered
    as favourably as the construction allows and NO MORE: the Y's three internal spokes
    are BOUNDARIES of two blocks each, so holding them is what per-block smoothing means.
    A scheme that moved them would be a different construction — a curved Y rather than a
    straight one — and is named as a successor rather than measured as this one.
    """
    built = tri_sector(reg, B, w)
    if built is None:
        return None
    blocks, _ = built
    q = {k: fbk.block_quality(fbk.winslow(np.asarray(blocks[k], float)))
         for k in TRI_BLOCKS}
    return {"min_scaled_jacobian": min(v["min_scaled_jacobian"] for v in q.values()),
            "worst_block": min(q, key=lambda k: q[k]["min_scaled_jacobian"]),
            "all_valid": bool(all(v["valid"] for v in q.values())),
            "per_block": {k: v["min_scaled_jacobian"] for k, v in q.items()}}


def sweep(reg, grid):
    """Every admissible `B` crossed with every interior point, and the argmax.

    The whole grid is kept, not just its maximum: the difference between a plateau and a
    tuned point is the difference between a construction that works and one that was
    fitted, and it is only visible in the surface.
    """
    cfg = reg["cfg"]
    rows, best = [], None
    for B in admissible(cfg.n_weld, cfg.n_thick):
        for w in grid:
            c = cell(reg, B, w)
            if c is None:
                continue
            rows.append({k: c[k] for k in
                         ("B", "w", "min_scaled_jacobian", "worst_block",
                          "non_positive_gauss_elements", "all_valid")})
            if c["all_valid"] and (best is None or c["min_scaled_jacobian"]
                                   > best["min_scaled_jacobian"]):
                best = c
    per_B = {}
    for B in admissible(cfg.n_weld, cfg.n_thick):
        r = [x for x in rows if x["B"] == B]
        ok = [x for x in r if x["all_valid"]]
        per_B[str(B)] = {
            "n_cells": len(r), "n_valid": len(ok),
            "best_min_scaled_jacobian": (max(x["min_scaled_jacobian"] for x in ok)
                                         if ok else None),
            "best_w": (max(ok, key=lambda x: x["min_scaled_jacobian"])["w"]
                       if ok else None),
            "shapes": [[splits(cfg.n_weld, B, cfg.n_thick)["a1"],
                        splits(cfg.n_weld, B, cfg.n_thick)["c2"]],
                       [splits(cfg.n_weld, B, cfg.n_thick)["a2"],
                        splits(cfg.n_weld, B, cfg.n_thick)["b1"]],
                       [splits(cfg.n_weld, B, cfg.n_thick)["b2"],
                        splits(cfg.n_weld, B, cfg.n_thick)["c1"]]]}
    # The plateau, measured rather than asserted: how much of the swept grid sits within
    # a tenth of the maximum, at the chosen `B`.
    plateau = None
    if best is not None:
        at_B = [x["min_scaled_jacobian"] for x in rows
                if x["B"] == best["B"] and x["all_valid"]]
        top = best["min_scaled_jacobian"]
        plateau = {"n_cells_at_best_B": len(at_B),
                   "fraction_within_10pct": float(
                       sum(1 for v in at_B if v > 0.9 * top) / max(len(at_B), 1)),
                   "fraction_within_25pct": float(
                       sum(1 for v in at_B if v > 0.75 * top) / max(len(at_B), 1))}
    return {"cells": rows, "per_B": per_B, "best": best, "plateau": plateau,
            "n_grid_points": len(grid)}


# ---------------------------------------------------------------------------
# THE CONTROLS
# ---------------------------------------------------------------------------

def control(genes, cfgname, blend):
    """The SEVEN-block sector at one blend, measured with the same instrument.

    `blend = 0.0` is what the tri-block replaces and is §37's 0.007208 / §51's 0.008176;
    `blend = 1.0` is what the tree ships and is the 0.782 every other number here is a
    fraction of.  A degradation is only readable against what it degrades from.
    """
    cfg = WW.get_config(cfgname)
    blocks = WW.sector_blocks(genes, cfg, uncap=(True, blend))
    per = {k: fbk.block_quality(np.asarray(v, float))
           for k, v in blocks.items() if k != "_thetas"}
    return {"blend": float(blend), "n_blocks": len(per),
            "min_scaled_jacobian": min(q["min_scaled_jacobian"] for q in per.values()),
            "worst_block": min(per, key=lambda k: per[k]["min_scaled_jacobian"]),
            "non_positive_gauss_elements": sum(q["non_positive_gauss_elements"]
                                               for q in per.values()),
            "all_valid": bool(all(q["valid"] for q in per.values())),
            "clears_min_sj_target": bool(min(q["min_scaled_jacobian"]
                                             for q in per.values()) > MIN_SJ_TARGET),
            "per_block": per}


# ---------------------------------------------------------------------------
# THE SECTOR, AT THE CHOSEN CELL
# ---------------------------------------------------------------------------

def sector_verdict(reg, B, w, bend=0.0):
    """All twelve blocks and all seventeen seams at one cell."""
    blocks, aux = tri_sector(reg, B, w, bend)
    per = {k: fbk.block_quality(np.asarray(v, float)) for k, v in blocks.items()}
    tbl = seam_table(reg, aux)
    sm = seams(blocks, tbl)
    quad_area = block_area_mm2(reg["base"]["rim_junction"])
    tri_area = sum(block_area_mm2(blocks[k]) for k in TRI_BLOCKS)
    return {
        "B": int(B), "w": [float(x) for x in w], "bend": float(bend), "aux": aux,
        "n_blocks": len(blocks), "n_seams": len(sm),
        "blocks": per,
        "min_scaled_jacobian": min(q["min_scaled_jacobian"] for q in per.values()),
        "worst_block": min(per, key=lambda k: per[k]["min_scaled_jacobian"]),
        "tri_min_scaled_jacobian": min(per[k]["min_scaled_jacobian"]
                                       for k in TRI_BLOCKS),
        "non_positive_gauss_elements": sum(q["non_positive_gauss_elements"]
                                           for q in per.values()),
        "mixed_sign_cells": sum(q["mixed_sign_cells"] for q in per.values()),
        "all_blocks_valid": bool(all(q["valid"] for q in per.values())),
        "clears_min_sj_target": bool(min(q["min_scaled_jacobian"]
                                         for q in per.values()) > MIN_SJ_TARGET),
        "seams": sm,
        "seams_close": bool(all(s["closes"] for s in sm)),
        "all_seams_whole_edge": True,
        "max_seam_gap_mm": max((s["max_gap_mm"] or 0.0) for s in sm),
        "quad_region_area_mm2": float(quad_area),
        "tri_region_area_mm2": float(tri_area),
        "area_relative_difference": float(abs(tri_area - quad_area) / quad_area),
        "element_count": int(sum((np.asarray(v).shape[0] - 1)
                                 * (np.asarray(v).shape[1] - 1)
                                 // (reg["cfg"].order ** 2) for v in blocks.values())),
    }


# ---------------------------------------------------------------------------
# THE GENE BOX — because this construction would ship for EVERY genome
# ---------------------------------------------------------------------------
#
# THE DIFFERENCE FROM §48, AND IT IS THE WHOLE REASON THIS SECTION IS NOT OPTIONAL.  The
# filleted blocking is opt-in: §48 could measure it at one genome, name six refusals out
# of sixteen, and still hand STEP 2 a usable instrument, because `fillet=` is passed by a
# study and never by the optimizer.  The faithful rim is not opt-in.  Adopting it changes
# `sector_blocks` for every genome the search touches, so a blocking that folds on a
# quarter of the box is not "measured at one genome" — it is unusable, and finding that
# out here is cheaper than finding it out from a barrier.
#
# TWO COLUMNS, AND THEY ARE DIFFERENT CLAIMS.  `fixed_w` applies the SHIPPED genome's own
# barycentric triple to every drawn genome — barycentric weights are scale-free, so this
# is a construction with no free parameter left, which is what would actually ship.
# `best_w` re-sweeps the interior point per genome and is the upper bound a smarter rule
# could reach.  The gap between them is the price of the rule being fixed.
#
# `B` IS NOT SWEPT HERE AND MUST NOT BE.  It sets element counts, and a mesh whose element
# count depends on the genome cannot be compared across a search.  It is a per-config
# constant, chosen once on the shipped genome, and then held.

GENOME_SWEEP_SEED = 20260823
GENOME_SWEEP_PER_ORIENTATION = 4
GENOME_X_GRID_N = 15


def sweep_genomes(cfg_name, B, w_fixed, per_orientation=GENOME_SWEEP_PER_ORIENTATION,
                  seed=GENOME_SWEEP_SEED, max_batches=40):
    """The tri-block at other genomes, grouped by flank orientation.

    Feasibility is `evaluate_design`'s own geometric pair (`x_order`, `hub_overlap`) plus
    the requirement that the genome's UNFILLETED, SHIPPED-BLEND sector is clean: a
    partition's verdict is only readable against a baseline that is itself valid, and a
    genome whose default mesh folds is not this construction's problem.
    """
    from study_mesh_quality import latin_hypercube
    import wheel_fea as WFEA

    low, high, _ = wg.bounds_arrays(WFEA.GENE_SPACE)
    cfgo = WW.get_config(cfg_name)
    grid = x_grid(n=GENOME_X_GRID_N)
    groups, batch = {}, 0
    wanted = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
    while batch < max_batches and any(
            len(groups.get(o, [])) < per_orientation for o in wanted):
        for vec in latin_hypercube(512, low, high, seed=seed + batch):
            _, loss = WFEA.evaluate_design(vec)
            if loss["x_order"] != 0.0 or loss["hub_overlap"] != 0.0:
                continue
            try:
                o = tuple(float(x) for x in WW.flank_orientation(vec, cfgo))
            except Exception:
                continue
            if len(groups.setdefault(o, [])) >= per_orientation:
                continue
            try:
                ctl = control(vec, cfg_name, 1.0)
                if not ctl["all_valid"]:
                    continue
                reg = region(vec, cfg_name, blend=0.0)
                quad = control(vec, cfg_name, 0.0)
            except Exception:
                continue
            rr = region_report(reg)
            row = {"orientation": list(o), "genes": [float(x) for x in vec],
                   "shipped_min_scaled_jacobian": ctl["min_scaled_jacobian"],
                   "faithful_quad_min_scaled_jacobian": quad["min_scaled_jacobian"],
                   # Carried so that a refusal has a MECHANISM and not just a count: the
                   # triangle's own shape is the only thing that changes between genomes,
                   # and `B` and the weights are held.
                   "arc_span_deg": rr["arc_span_deg"],
                   "bow_over_width": rr["bow_over_width"],
                   # And the OTHER mechanism, carried so it can be ruled out rather than
                   # assumed away.  `study_fillet_block`'s fold gate is the closed-form
                   # statement of whether this genome's spoke exists at all, and the same
                   # two-term draw filter above lets folded ones through here too.  The
                   # tri-block does not touch the spoke block, so the expectation is that
                   # it explains nothing about this construction -- which is a claim, and
                   # `the_fold_margin_does_not_explain_the_tri_block` is where it is tested.
                   "fold": fbk.fold_margin(vec, cfg_name),
                   "side_lengths_mm": [rr["A_length_mm"], rr["B_length_mm"],
                                       rr["C_length_mm"]],
                   "wedges_deg": [rr["wedge_at_P_t_deg"], rr["wedge_at_Q_deg"],
                                  rr["wedge_at_B_star_deg"]],
                   "turn_at_far_end_deg": rr["turn_at_far_end_deg"]}
            try:
                fixed = cell(reg, B, w_fixed)
                sec = (sector_verdict(reg, B, w_fixed)
                       if fixed is not None and fixed["all_valid"] else None)
                best = None
                for w in grid:
                    c = cell(reg, B, w)
                    if c is not None and c["all_valid"] and (
                            best is None
                            or c["min_scaled_jacobian"] > best["min_scaled_jacobian"]):
                        best = c
            except Exception as exc:      # a drawn genome must never kill the driver
                row["why"] = f"{type(exc).__name__}: {exc}"
                groups[o].append(row)
                continue
            row.update({
                "fixed_w_valid": bool(fixed is not None and fixed["all_valid"]),
                "fixed_w_min_scaled_jacobian": (fixed["min_scaled_jacobian"]
                                                if fixed is not None else None),
                "fixed_w_worst_block": (fixed["worst_block"] if fixed else None),
                "best_w_valid": bool(best is not None),
                "best_w_min_scaled_jacobian": (best["min_scaled_jacobian"]
                                               if best else None),
                "best_w": (best["w"] if best else None),
                "seams_close": (sec["seams_close"] if sec else None),
                "max_seam_gap_mm": (sec["max_seam_gap_mm"] if sec else None)})
            groups[o].append(row)
        batch += 1
    rows = [r for v in groups.values() for r in v]
    ok = [r for r in rows if r.get("fixed_w_valid")]
    best_ok = [r for r in rows if r.get("best_w_valid")]
    return {"seed": seed, "config": cfg_name, "B": int(B),
            "w_fixed": [float(x) for x in w_fixed],
            "per_orientation": per_orientation, "x_grid_n": GENOME_X_GRID_N,
            "groups": {str(list(k)): v for k, v in sorted(groups.items())},
            "n_genomes": len(rows),
            "n_fixed_w_valid": len(ok), "n_best_w_valid": len(best_ok),
            "n_fixed_w_clears_target": sum(
                1 for r in ok if r["fixed_w_min_scaled_jacobian"] > MIN_SJ_TARGET),
            "n_best_w_clears_target": sum(
                1 for r in best_ok if r["best_w_min_scaled_jacobian"] > MIN_SJ_TARGET),
            "all_seams_close": bool(all(r["seams_close"] for r in ok)) if ok else None,
            "fixed_w_range": ([min(r["fixed_w_min_scaled_jacobian"] for r in ok),
                               max(r["fixed_w_min_scaled_jacobian"] for r in ok)]
                              if ok else None)}


# `sweep_genomes` already tells us WHICH drawn genomes no placement of X can rescue at this
# `B` (`best_w_valid` is False for them) -- that is the "curved Y" question named in
# UNCAP_PLAN Step 3 PART 2 and is not this one.  What it leaves open is whether the FIXED
# rule -- the one with no free parameter left, the one that would actually ship -- can be
# re-derived to reach every genome ITS OWN `best_w` can, the same question §48 PART 13
# asked of the fillet's layer profile.  `GENOME_ROBUST_X_GRID_N` is its own grid, published
# in full for the same reason every grid here is: `sweep`'s own single-genome argmax showed
# a plateau of only 6.9%-8.3% of valid cells, and a joint argmax over sixteen genomes can be
# a far narrower ridge than that -- visible only by publishing the surface, not by trusting
# the one cell an argmax reports.
GENOME_ROBUST_X_GRID_N = 25


def sweep_w_genomes(cfg_name, B, genome_rows, shipped_genes, current_w, grid=None):
    """The interior point's barycentric triple, re-derived against the GENOME BOX.

    `genome_rows` is `sweep_genomes`'s own group rows for this config.  The shipped genome
    is NOT one of them -- `sweep_genomes` draws sixteen OTHER genomes -- so it is appended
    here explicitly and named rather than folded in silently, exactly as
    `study_fillet_block.sweep_layer_profile_genomes` appends its own shipped genome.  A
    genome `sweep_genomes` already marked `best_w_valid: False` is EXCLUDED and named: no
    placement of X rescues it at this `B`, so it would dominate every worst-case comparison
    for a reason this sweep cannot fix and should not be blamed for.

    The objective is `n_clear` first and `worst_min_scaled_jacobian` second, not the
    reverse.  A raw argmax of the worst genome's worst block chases whichever genome is
    CLOSEST to folding, which is a different question from how many genomes clear the
    barrier the optimizer actually enforces -- the published grid at `coarse` has exactly
    one cell where all fixable genomes are simultaneously valid, and it is not the cell a
    worst-case argmax would pick.
    """
    grid = grid if grid is not None else x_grid(n=GENOME_ROBUST_X_GRID_N)
    fixable = [r for r in genome_rows if r.get("best_w_valid")]
    excluded = [r for r in genome_rows if not r.get("best_w_valid")]
    regs = [region(np.asarray(r["genes"], float), cfg_name, blend=0.0) for r in fixable]
    regs.append(region(np.asarray(shipped_genes, float), cfg_name, blend=0.0))
    n_cells = len(regs)

    def stats(w):
        worst, n_valid, n_clear = 9.0, 0, 0
        for reg in regs:
            c = cell(reg, B, w)
            worst = min(worst, c["min_scaled_jacobian"])
            n_valid += int(c["all_valid"])
            n_clear += int(c["min_scaled_jacobian"] > MIN_SJ_TARGET)
        return {"w": [float(x) for x in w], "worst_min_scaled_jacobian": float(worst),
                "n_valid": n_valid, "n_clear": n_clear}

    rows = [stats(w) for w in grid]
    rows.sort(key=lambda r: (-r["n_clear"], -r["worst_min_scaled_jacobian"]))
    best = rows[0]
    at_current = stats(tuple(current_w))
    shipped_reg = regs[-1]
    shipped_at_best = cell(shipped_reg, B, tuple(best["w"]))
    shipped_at_current = cell(shipped_reg, B, tuple(current_w))
    fully_valid = [r for r in rows if r["n_valid"] == n_cells]
    return {
        "n_cells": n_cells, "n_fixable": len(fixable),
        "excluded_arc_span_deg": [r["arc_span_deg"] for r in excluded],
        "grid_n": len(grid),
        "best": best, "at_current_w": at_current,
        "n_fully_valid_cells": len(fully_valid),
        "shipped_min_scaled_jacobian_at_best": shipped_at_best["min_scaled_jacobian"],
        "shipped_clears_target_at_best": bool(
            shipped_at_best["min_scaled_jacobian"] > MIN_SJ_TARGET),
        "shipped_min_scaled_jacobian_at_current": shipped_at_current["min_scaled_jacobian"],
        "shipped_clears_target_at_current": bool(
            shipped_at_current["min_scaled_jacobian"] > MIN_SJ_TARGET),
        "rows": rows,
    }


# THE CURVED Y.  PART 2's Winslow column found that a generated interior changes the
# number by 0.000000, because the Y's three spokes are BOUNDARIES of two blocks each and
# per-block smoothing holds them by definition -- so the only lever left is where the
# spokes GO, and they were straight lines.  `_bent_spoke` is that lever and `bend` is its
# one parameter.  This sweep is the same two-column claim `sweep_genomes` makes, run over
# the (w, bend) plane instead of the w one: a per-genome ceiling, which says whether the
# curve can reach a genome AT ALL, and a joint fixed rule, which says whether one
# construction with no free parameter left can reach it.
BEND_GRID = tuple(round(0.1 * k, 2) for k in range(11))
# The same 25 the interior point's own genome-robust sweep uses, and for the reason that
# sweep found the hard way: at 15 the joint argmax at `coarse` reaches 14 of 16 genomes
# and at 21 it reaches all 16, so the coarser grid was reporting the SPACING and not the
# rule.  Two sweeps answering the same shape of question are on the same grid.
BEND_X_GRID_N = GENOME_ROBUST_X_GRID_N


# THE REFUSAL SEARCH -- PART 6's named experiment.
#
# One genome in the sixteen refuses the curve at every interior point, every bend and every
# admissible free count, and PART 6 found it extremal on three shape quantities at once --
# most widely on the region's interior-angle sum, by 57% of the others' spread.  With ONE
# negative example that is arithmetic rather than evidence: any quantity on which a set of
# one is extremal separates it from a set of fifteen.  A second refusal turns all three
# candidates into testable claims at once.
#
# The draw is a SUPERSET, not a redraw: `sweep_genomes` fills each orientation from the same
# Latin-hypercube stream in the same order, so the first four of each are exactly the box
# every published number is measured on and the next four are new.  Nothing above moves.
#
# `coarse` only and the 15-point interior grid rather than the 25-point one, because this
# section asks WHETHER a ceiling is negative and not what it is -- the published ceilings
# stay where they were measured.
#
# MEASURED AT 16, 32 AND 64 GENOMES, AND THE EXPERIMENT DID NOT DO WHAT IT WAS DESIGNED TO.
# No second refusal appears: the curve reaches 63 of 64.  What the larger box did instead is
# FALSIFY the leading candidate.  PART 6 ranked the interior-angle sum first on a gap worth
# 57% of the reached set's spread; at 32 that is 26% and at 64 it is 4%, because each larger
# draw finds a reached genome closer to the refusal (170.3 -> 164.2 -> 157.9 against the
# refusal's 156.4).  `arc_span_deg`, which PART 6 ranked third and discounted, is the one
# that holds: 0.187 -> 0.187 -> 0.176 across a fourfold box.  A separation that survives a
# 4x draw and one that decays by 14x are different kinds of claim, and only running the box
# out shows which is which.
REFUSAL_SEARCH_PER_ORIENTATION = 16


def _shape(row):
    """The shape numbers PART 6 tested, keyed the way its table reports them."""
    return {"wedge_sum_deg": float(sum(row["wedges_deg"])),
            "wedge_sum_minus_180": float(abs(sum(row["wedges_deg"]) - 180.0)),
            "arc_span_deg": float(row["arc_span_deg"]),
            "bow_over_width": float(row["bow_over_width"]),
            "turn_at_far_end_deg": float(row["turn_at_far_end_deg"]),
            "min_wedge_deg": float(min(row["wedges_deg"])),
            "A_over_C": float(row["side_lengths_mm"][0] / row["side_lengths_mm"][2])}


def _separation(refusals, reached, key):
    """Does `key` separate the refusals from the rest, and by how much of the spread?

    Reported with the gap NORMALISED by the reached set's own spread, because a gap in
    degrees means nothing without knowing how wide the box is in that quantity -- which is
    the whole difference between PART 6's angle sum (57%) and its arc span (19%).
    """
    b = [r[key] for r in refusals]
    g = [r[key] for r in reached]
    if not b or not g:
        return None
    spread = max(g) - min(g)
    low, high = min(g) - max(b), min(b) - max(g)
    gap = max(low, high)
    return {"refusals": b, "reached_min": min(g), "reached_max": max(g),
            "separates": bool(gap > 0.0),
            "refusal_is_low": bool(low > 0.0),
            "gap": float(gap),
            "gap_over_spread": float(gap / spread) if spread > 0 else None}


def sweep_refusal_search(cfg_name, B, w_fixed, shipped_genes, current_w,
                         per_orientation=REFUSAL_SEARCH_PER_ORIENTATION):
    """Draw deeper until a SECOND region refuses the curve, and re-test PART 6's candidates.

    Returns the per-genome verdicts over the enlarged box, the shape numbers for each, and
    the separation statistic for every quantity PART 6 tried -- so a candidate that survives
    a second negative is visibly different from one that does not.
    """
    deep = sweep_genomes(cfg_name, B, w_fixed, per_orientation=per_orientation)
    rows = [r for v in deep["groups"].values() for r in v if "fixed_w_valid" in r]
    bend = sweep_bend_genomes(cfg_name, B, rows, shipped_genes, current_w,
                              grid=x_grid(n=GENOME_X_GRID_N))
    per = bend["per_genome"]
    # `sweep_bend_genomes` appends the shipped genome last; it is not a drawn one and is
    # excluded from the statistic for the same reason it is named separately everywhere else.
    drawn = [(g, _shape(r)) for g, r in zip(per, rows)]
    refusals = [sh for g, sh in drawn if not g["curved_valid"]]
    reached = [sh for g, sh in drawn if g["curved_valid"]]
    keys = ("wedge_sum_deg", "wedge_sum_minus_180", "arc_span_deg", "bow_over_width",
            "turn_at_far_end_deg", "min_wedge_deg", "A_over_C")
    return {"config": cfg_name, "per_orientation": per_orientation,
            "n_genomes": len(drawn), "n_refusals": len(refusals),
            "n_reached": len(reached),
            "x_grid_n": GENOME_X_GRID_N, "bend_grid": list(BEND_GRID),
            "genomes": [{"curved_valid": bool(g["curved_valid"]),
                         "curved_min_scaled_jacobian": g["curved_min_scaled_jacobian"],
                         **sh} for g, sh in drawn],
            "separation": {k: _separation(refusals, reached, k) for k in keys},
            # Which of PART 6's three survive a larger box, named rather than left to be
            # read off the table.
            "still_separating": sorted(
                k for k in keys
                if (_separation(refusals, reached, k) or {}).get("separates"))}


def sweep_bend_genomes(cfg_name, B, genome_rows, shipped_genes, current_w,
                       grid=None, bends=BEND_GRID):
    """The curved Y over the gene box: what it reaches, and at what fixed rule.

    Every genome `sweep_genomes` drew is here, INCLUDING the ones it marked
    `best_w_valid: False` -- those are the whole question, and excluding them is what
    `sweep_w_genomes` had to do and this one must not.  The shipped genome is appended and
    named, as there.

    `bend = 0.0` is in the grid on purpose rather than assumed: the straight-Y column of
    every table below is re-measured here against the identical objective, so the
    comparison is one sweep's own two slices and not this sweep against a remembered
    number from another.
    """
    grid = grid if grid is not None else x_grid(n=BEND_X_GRID_N)
    # A row `sweep_genomes` recorded a `why` on is one whose cell RAISED; it carries the
    # region's shape but no verdict, and asking the same question of it again would raise
    # here too.  `fixed_w_valid` is present only on the path that got a verdict.
    rows_in = [r for r in genome_rows if "fixed_w_valid" in r]
    regs = [region(np.asarray(r["genes"], float), cfg_name, blend=0.0) for r in rows_in]
    regs.append(region(np.asarray(shipped_genes, float), cfg_name, blend=0.0))
    labels = [{"arc_span_deg": r["arc_span_deg"], "bow_over_width": r["bow_over_width"],
               "shipped_genome": False} for r in rows_in]
    shipped_rr = region_report(regs[-1])
    labels.append({"arc_span_deg": shipped_rr["arc_span_deg"],
                   "bow_over_width": shipped_rr["bow_over_width"],
                   "shipped_genome": True})

    # [genome][bend][w] once, and every column below is a slice of it.
    table = [[[cell(reg, B, w, bend) for w in grid] for bend in bends] for reg in regs]

    def ceiling(gi, only_zero_bend):
        best = None
        for bi, bend in enumerate(bends):
            if only_zero_bend and bend != 0.0:
                continue
            for wi, c in enumerate(table[gi][bi]):
                if c is None or not c["all_valid"]:
                    continue
                if best is None or c["min_scaled_jacobian"] > best[0]:
                    best = (c["min_scaled_jacobian"], float(bend), list(grid[wi]))
        return best

    per_genome = []
    for gi, lab in enumerate(labels):
        st, cu = ceiling(gi, True), ceiling(gi, False)
        per_genome.append({
            **lab,
            "straight_valid": st is not None,
            "straight_min_scaled_jacobian": st[0] if st else None,
            "curved_valid": cu is not None,
            "curved_min_scaled_jacobian": cu[0] if cu else None,
            "curved_bend": cu[1] if cu else None,
            "curved_w": cu[2] if cu else None,
            "rescued_by_the_curve": bool(cu is not None and st is None)})

    # The joint fixed rule runs on the genomes SOME (w, bend) reaches, and the ones it
    # does not are excluded and named -- the same convention `sweep_w_genomes` set, and
    # for the same reason: a genome no cell in the plane rescues dominates every
    # worst-case comparison for a reason the rule cannot fix.  Keeping the convention is
    # also what makes the two functions' numbers readable against each other.
    keep = [gi for gi, g in enumerate(per_genome) if g["curved_valid"]]
    rows = []
    for bi, bend in enumerate(bends):
        for wi, w in enumerate(grid):
            worst, n_valid, n_clear = 9.0, 0, 0
            for gi in keep:
                c = table[gi][bi][wi]
                worst = min(worst, c["min_scaled_jacobian"])
                n_valid += int(c["all_valid"])
                n_clear += int(c["min_scaled_jacobian"] > MIN_SJ_TARGET)
            rows.append({"bend": float(bend), "w": [float(x) for x in w],
                         "worst_min_scaled_jacobian": float(worst),
                         "n_valid": n_valid, "n_clear": n_clear})
    key = lambda r: (-r["n_clear"], -r["worst_min_scaled_jacobian"])   # noqa: E731
    straight_rows = [r for r in rows if r["bend"] == 0.0]
    best = min(rows, key=key)
    best_straight = min(straight_rows, key=key)

    # PUBLISHED, BUT NOT ALL OF IT.  Every grid in this file is published whole so that a
    # plateau and a tuned point can be told apart by looking, and the (w, bend) plane is
    # eleven times the last one -- large enough that committing it whole would double the
    # artifact.  So the two slices where the question is actually asked go out whole: the
    # straight Y, and the bend the joint rule picked.  Every other bend goes out as its
    # own argmax, which is what says whether the winning bend is a spike or a shelf.
    per_bend = [min([r for r in rows if r["bend"] == float(b)], key=key) for b in bends]
    at_best_bend = [r for r in rows if r["bend"] == best["bend"]]
    # The 10% band is taken off the MAGNITUDE, not as a multiplier: the joint worst can be
    # negative, and `0.9 * x` moves the wrong way when it is -- a band that excludes the
    # maximum itself is not a plateau measurement, it is a sign error.
    band = best["worst_min_scaled_jacobian"] - 0.1 * abs(best["worst_min_scaled_jacobian"])
    near = [r for r in at_best_bend
            if r["n_clear"] == best["n_clear"]
            and r["worst_min_scaled_jacobian"] >= band]

    # THE REFUSAL, PRICED PROPERLY.  A genome no cell of the (w, bend) plane rescues is
    # this section's load-bearing negative, and at the shipped `B` alone it is a weaker
    # claim than it sounds: `B` is held across the gene box because it sets element
    # counts, but the question "is this region Y-partitionable at all" is not about the
    # count that ships.  So the refusals -- and only they, because this is the expensive
    # sweep -- are re-asked at every admissible `B`.
    cfg = WW.get_config(cfg_name)
    refusals = []
    for gi, lab in enumerate(labels):
        if per_genome[gi]["curved_valid"]:
            continue
        per_B = []
        for Bx in admissible(cfg.n_weld, cfg.n_thick):
            top = None
            for bend in bends:
                for w in grid:
                    c = cell(regs[gi], Bx, w, bend)
                    if c is None:
                        continue
                    if top is None or c["min_scaled_jacobian"] > top[0]:
                        top = (c["min_scaled_jacobian"], float(bend), c["all_valid"])
            per_B.append({"B": int(Bx), "ceiling": top[0], "bend": top[1],
                          "valid": bool(top[2])})
        refusals.append({**lab, "per_B": per_B,
                         "ceiling_over_every_B": max(r["ceiling"] for r in per_B),
                         "valid_at_any_B": any(r["valid"] for r in per_B)})

    shipped = {}
    for tag, r in (("best", best), ("best_straight", best_straight)):
        c = cell(regs[-1], B, tuple(r["w"]), r["bend"])
        shipped[tag] = {"min_scaled_jacobian": c["min_scaled_jacobian"],
                        "clears_target": bool(c["min_scaled_jacobian"] > MIN_SJ_TARGET)}
    c = cell(regs[-1], B, tuple(current_w), 0.0)
    shipped["published_cell"] = {
        "min_scaled_jacobian": c["min_scaled_jacobian"],
        "clears_target": bool(c["min_scaled_jacobian"] > MIN_SJ_TARGET)}

    return {
        "config": cfg_name, "B": int(B), "n_genomes": len(regs),
        "n_cells": len(keep),
        "bend_grid": [float(b) for b in bends], "grid_n": len(grid),
        "per_genome": per_genome,
        "n_straight_valid": sum(1 for r in per_genome if r["straight_valid"]),
        "n_curved_valid": sum(1 for r in per_genome if r["curved_valid"]),
        "n_rescued_by_the_curve": sum(1 for r in per_genome
                                      if r["rescued_by_the_curve"]),
        "refusals_bow_over_width": [r["bow_over_width"] for r in per_genome
                                    if not r["curved_valid"]],
        "refusals": refusals,
        "best": best, "best_straight": best_straight,
        "shipped": shipped,
        "per_bend": per_bend,
        "plateau_at_best_bend": {"n_cells": len(at_best_bend), "n_within_10pc": len(near)},
        "surface_straight": straight_rows,
        "surface_at_best_bend": at_best_bend,
    }


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------

def build(genes, configs, genome_sweep=True):
    rec = {"configs": list(configs), "faithful_rim_blend": FAITHFUL[1],
           "min_sj_target": MIN_SJ_TARGET, "x_grid_n": X_GRID_N,
           "algebra": algebra_section(configs), "per_config": {}}
    grid = x_grid()
    for name in configs:
        reg = region(genes, name, blend=FAITHFUL[1])
        sw = sweep(reg, grid)
        if sw["best"] is not None:
            fine = refine(reg, sw["best"]["B"], sw["best"]["w"])
            # REPORTED, NOT ADOPTED.  The rule this file states is the argmax over the
            # PUBLISHED grid, and it is stated that way on purpose: at `medium` the
            # refinement gains 0.045 by walking `w_Pt` down to the search box's own
            # clamp, and that cell generalises across the gene box WORSE than the grid
            # point it beats.  A number that only exists at four decimal places of one
            # weight is a tuned point, which is the thing §48's ridge rule was written to
            # keep visible rather than to chase.
            sw["refined"] = ({"w": fine["w"],
                              "min_scaled_jacobian": fine["min_scaled_jacobian"],
                              "gain_over_published_grid": (
                                  fine["min_scaled_jacobian"]
                                  - sw["best"]["min_scaled_jacobian"])}
                             if fine is not None else None)
        chosen = sw["best"]
        per = {"region": region_report(reg),
               "control_faithful": control(genes, name, 0.0),
               "control_shipped": control(genes, name, 1.0),
               "sweep": sw}
        if chosen is not None:
            per["sector"] = sector_verdict(reg, chosen["B"], chosen["w"])
            # The neighbours are CUT, so their interiors must be identical to the blocks
            # they were cut from.  Checked here rather than assumed, because "sliced, not
            # rebuilt" is a claim about code.
            blocks, _ = tri_sector(reg, chosen["B"], chosen["w"])
            base = reg["base"]
            cuts = []
            for whole, lo, hi, axis in (("spoke", "spoke_eta_lo", "spoke_eta_hi", 1),
                                        ("hub_junction", "hub_junction_lo",
                                         "hub_junction_hi", 1)):
                w0 = np.asarray(base[whole], float)
                j = np.asarray(blocks[lo]).shape[axis] - 1
                cuts.append({"block": whole, "max_gap_mm": float(max(
                    np.abs(np.asarray(blocks[lo]) - w0[:, :j + 1]).max(),
                    np.abs(np.asarray(blocks[hi]) - w0[:, j:]).max()))})
            w0 = np.asarray(base["rim_band_weld"], float)
            i = int(per["sector"]["aux"]["i_star"])
            first = per["sector"]["aux"]["first_weld"]
            second = per["sector"]["aux"]["last_weld"]
            cuts.append({"block": "rim_band_weld", "max_gap_mm": float(max(
                np.abs(np.asarray(blocks[first]) - w0[:i + 1]).max(),
                np.abs(np.asarray(blocks[second]) - w0[i:]).max()))})
            per["cuts_are_slices"] = cuts
            per["winslow"] = winslow_column(reg, chosen["B"], chosen["w"])
            if genome_sweep:
                per["genomes"] = sweep_genomes(name, chosen["B"], chosen["w"])
                grows = [r for v in per["genomes"]["groups"].values() for r in v]
                per["genome_robust_w"] = sweep_w_genomes(
                    name, chosen["B"], grows, genes, chosen["w"])
                per["curved_y"] = sweep_bend_genomes(
                    name, chosen["B"], grows, genes, chosen["w"])
                # PART 6's named experiment, at the FIRST config only: it draws a superset
                # of the box above and its whole purpose is to find a second negative, so
                # running it twice would cost eight minutes to ask the same question of the
                # same genomes at a resolution that does not change the answer.
                if name == rec["configs"][0]:
                    per["refusal_search"] = sweep_refusal_search(
                        name, chosen["B"], chosen["w"], genes, chosen["w"])
                # The curve moves the three spokes, which are seams.  They are shared
                # arrays so this cannot fail -- which is exactly why it is checked, at
                # the bend the joint rule picked rather than at the one that ships.
                bw = tuple(per["curved_y"]["best"]["w"])
                bb = per["curved_y"]["best"]["bend"]
                per["curved_y"]["sector_at_best"] = {
                    k: sector_verdict(reg, chosen["B"], bw, bb)[k]
                    for k in ("seams_close", "max_seam_gap_mm", "n_seams",
                              "tri_region_area_mm2", "area_relative_difference",
                              "element_count")}
                # THE BOUNDARY, NOT ITS AREA.  Summing three shoelaces cancels the shared
                # spokes only to rounding, so an area comparison cannot be exact and is
                # the wrong instrument for an exact claim.  The claim is that the SIX
                # boundary edges the region owns do not move with the bend, and those are
                # arrays: compared as arrays, at one `w` so that only `bend` differs.
                owned = (("rim_tri_t", (slice(None), 0)), ("rim_tri_t", (0, slice(None))),
                         ("rim_tri_q", (slice(None), 0)), ("rim_tri_q", (-1, slice(None))),
                         ("rim_tri_b", (slice(None), -1)), ("rim_tri_b", (0, slice(None))))
                flat, bent = (tri_sector(reg, chosen["B"], bw, x)[0] for x in (0.0, bb))
                per["curved_y"]["region_sides_are_untouched"] = bool(all(
                    np.array_equal(np.asarray(flat[k])[ix], np.asarray(bent[k])[ix])
                    for k, ix in owned))
                per["curved_y"]["tiled_area_relative_shift"] = float(abs(
                    block_area_mm2s(bent) - block_area_mm2s(flat))
                    / block_area_mm2s(flat))
        rec["per_config"][name] = per

    rec["self_checks"] = self_checks(rec)
    return rec


def self_checks(rec):
    """What must hold for the report to mean anything, as opposed to what it FINDS.

    A block's min scaled Jacobian is a finding and is never gated.  These are the claims
    the file makes about ITSELF: that its control reproduces the number §37 and §51 both
    published, that its partition covers the same region as the block it replaces, that
    every seam it declares actually closes, and that its algebra reproduces §37's own
    arithmetic at §37's own `B`.
    """
    out = {}
    cs = rec["per_config"]
    out["control_reproduces_the_collapse"] = all(
        c["control_faithful"]["worst_block"] == "rim_junction"
        and c["control_faithful"]["min_scaled_jacobian"] < 0.02
        for c in cs.values())
    out["shipped_control_is_the_published_0.78"] = all(
        abs(c["control_shipped"]["min_scaled_jacobian"] - 0.7822) < 0.02
        for c in cs.values())
    out["every_config_found_a_valid_cell"] = all("sector" in c for c in cs.values())
    out["all_seams_close"] = all(c["sector"]["seams_close"]
                                 for c in cs.values() if "sector" in c)
    out["seventeen_seams"] = all(c["sector"]["n_seams"] == 17
                                 for c in cs.values() if "sector" in c)
    out["twelve_blocks"] = all(c["sector"]["n_blocks"] == 12
                               for c in cs.values() if "sector" in c)
    out["partition_covers_the_region"] = all(
        c["sector"]["area_relative_difference"] < 1.0e-4
        for c in cs.values() if "sector" in c)
    out["cuts_are_slices"] = all(
        all(x["max_gap_mm"] == 0.0 for x in c["cuts_are_slices"])
        for c in cs.values() if "cuts_are_slices" in c)
    # THE CURVE MOVES SEAMS AND NOTHING ELSE.  `_bent_spoke` touches only the three
    # internal spokes, so bending them must leave the region's own boundary — and
    # therefore the tiled area — bit-for-bit what the straight Y tiled, at a DIFFERENT
    # interior point and a different bend.  A curve that leaked into the boundary would
    # be a different region silently measured against the same control.
    cy = [c for c in cs.values() if "curved_y" in c and "sector" in c]
    out["the_bend_moves_no_boundary"] = all(
        c["curved_y"]["region_sides_are_untouched"] for c in cy)
    out["the_bend_tiles_the_same_area"] = all(
        c["curved_y"]["tiled_area_relative_shift"] < 1.0e-12 for c in cy)
    out["the_bend_closes_every_seam"] = all(
        c["curved_y"]["sector_at_best"]["seams_close"]
        and c["curved_y"]["sector_at_best"]["n_seams"] == 17 for c in cy)
    out["the_straight_y_is_a_slice_of_the_bend_grid"] = all(
        0.0 in c["curved_y"]["bend_grid"] for c in cy)
    out["every_refusal_was_re-asked_at_every_free_count"] = all(
        len(c["curved_y"]["refusals"]) == (c["curved_y"]["n_genomes"]
                                           - c["curved_y"]["n_curved_valid"])
        and all(not r["valid_at_any_B"] for r in c["curved_y"]["refusals"])
        for c in cy)
    # A NEGATIVE result, gated so it cannot rot into a positive one by assumption.
    # `study_fillet_block`'s fold margin classifies that study's one inverted spoke block
    # 16/16, and the temptation is to reach for it here as a general difficulty predictor.
    # It is not one: the tri-block partitions the rim JUNCTION region and never touches
    # the offset band, and this box's fold-negative genomes are among its easiest.  The
    # check is that the two are not merely uncorrelated but ANTI-informative -- the worst
    # cell at the fixed rule is a fold-CLEAN genome -- so a future run where the fold
    # margin does start explaining refusals here shows up as a failure and gets looked at.
    def _fold_is_not_the_story(per):
        grows = [r for v in per["genomes"]["groups"].values() for r in v
                 if "fold" in r and r.get("fixed_w_min_scaled_jacobian") is not None]
        if not grows:
            return True
        worst = min(grows, key=lambda r: r["fixed_w_min_scaled_jacobian"])
        return not worst["fold"]["folds"]

    # The refusal search's statistic is a set of refusals against a set of reached, so it
    # is vacuous if either is empty.  Structural, so it gates.  WHICH quantity separates is
    # a finding and is reported, never gated -- the whole result is that the leading
    # candidate changed when the box grew.
    rsx = [per["refusal_search"] for per in rec["per_config"].values()
           if "refusal_search" in per]
    if rsx:
        out["the_refusal_search_has_both_classes"] = all(
            r["n_refusals"] > 0 and r["n_reached"] > 0 for r in rsx)
        # And it must be a SUPERSET of the published box, or it is a different experiment.
        out["the_refusal_search_is_a_superset_of_the_box"] = all(
            r["n_genomes"] >= per["genomes"]["n_genomes"]
            for per, r in ((p, p["refusal_search"]) for p in rec["per_config"].values()
                           if "refusal_search" in p))
    out["the_fold_margin_does_not_explain_the_tri_block"] = all(
        _fold_is_not_the_story(per) for per in rec["per_config"].values())
    rows = rec["algebra"].get("coarse", {}).get("rows", [])
    s37 = [r for r in rows if r["is_section_37_choice"]]
    out["algebra_reproduces_section_37"] = bool(
        s37 and s37[0]["B"] == 8 and s37[0]["shapes"] == [[7, 1], [3, 1], [7, 3]])
    out["pass"] = all(v for k, v in out.items() if k != "pass")
    return out


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def _print(rec):
    print("\n" + "=" * 78)
    print("  THE RIM TRI-BLOCK, BUILT — §51's probe, measured")
    print("=" * 78)

    print("\n  §37 CLAUSE 2 — THE ADMISSIBLE FREE COUNTS, ENUMERATED")
    for name, al in rec["algebra"].items():
        print(f"    {name}  A = {al['A_arc_elements']} (n_weld)   "
              f"C = {al['C_cross_elements']} (n_thick)")
        print(f"      {'B':>3s} {'a1/a2':>7s} {'b1/b2':>7s} {'c1/c2':>7s} "
              f"{'block shapes':>22s}  {'strip?':>7s}")
        for r in al["rows"]:
            sh = "  ".join(f"{s[0]}x{s[1]}" for s in r["shapes"])
            mark = " <- §37" if r["is_section_37_choice"] else ""
            print(f"      {r['B']:3d} {r['a1']:3d}/{r['a2']:<3d} {r['b1']:3d}/{r['b2']:<3d} "
                  f"{r['c1']:3d}/{r['c2']:<3d} {sh:>22s}  "
                  f"{'YES' if r['has_one_element_strip'] else 'no':>7s}{mark}")
        print(f"      strip-free B: {al['strip_free_B']}")

    print("\n  THE REGION — why a quad cannot sit on it")
    print(f"    {'config':8s} {'arc span':>9s} {'A mm':>8s} {'B mm':>8s} {'C mm':>8s} "
          f"{'turn at far_end':>16s}")
    for name, c in rec["per_config"].items():
        r = c["region"]
        print(f"    {name:8s} {r['arc_span_deg']:8.4f}° {r['A_length_mm']:8.4f} "
              f"{r['B_length_mm']:8.4f} {r['C_length_mm']:8.4f} "
              f"{r['turn_at_far_end_deg']:15.4f}°")
    print("    180° is straight: the fourth vertex a quad block needs does not exist "
          "on this region.")

    print("\n  THE SWEEP — best worst-block min scaled Jacobian per free count")
    for name, c in rec["per_config"].items():
        print(f"    {name}")
        print(f"      {'B':>3s} {'shapes':>18s} {'valid cells':>12s} "
              f"{'best min scaled J':>18s} {'best w (P_t,Q,B*)':>26s}")
        for B, v in c["sweep"]["per_B"].items():
            sh = " ".join(f"{s[0]}x{s[1]}" for s in v["shapes"])
            best = ("%.6f" % v["best_min_scaled_jacobian"]
                    if v["best_min_scaled_jacobian"] is not None else "none valid")
            wv = ("(%.3f, %.3f, %.3f)" % tuple(v["best_w"])) if v["best_w"] else "-"
            print(f"      {B:>3s} {sh:>18s} {v['n_valid']:6d}/{v['n_cells']:<5d} "
                  f"{best:>18s} {wv:>26s}")
        pl = c["sweep"]["plateau"]
        if pl:
            print(f"      plateau at the chosen B: {pl['fraction_within_10pct']:.1%} of "
                  f"valid cells within 10% of the maximum, "
                  f"{pl['fraction_within_25pct']:.1%} within 25%")

    print("\n  THE VERDICT — the twelve-block sector against the seven-block controls")
    print(f"    {'config':8s} {'shipped (1.0)':>14s} {'faithful quad':>14s} "
          f"{'faithful TRI':>14s} {'x':>7s} {'clears 0.2':>11s} {'seams':>13s}")
    for name, c in rec["per_config"].items():
        if "sector" not in c:
            print(f"    {name:8s}  no valid cell")
            continue
        s = c["sector"]
        f0 = c["control_faithful"]["min_scaled_jacobian"]
        print(f"    {name:8s} {c['control_shipped']['min_scaled_jacobian']:14.6f} "
              f"{f0:14.6f} {s['min_scaled_jacobian']:14.6f} "
              f"{s['min_scaled_jacobian'] / f0:6.1f}x "
              f"{str(s['clears_min_sj_target']):>11s} "
              f"{sum(1 for x in s['seams'] if x['closes'])}/{s['n_seams']} close")
    for name, c in rec["per_config"].items():
        if "sector" not in c:
            continue
        s = c["sector"]
        print(f"    {name}: B = {s['B']}, worst block {s['worst_block']}, "
              f"worst seam gap {s['max_seam_gap_mm']:.2e} mm, "
              f"region area matches to {s['area_relative_difference']:.2e}")

    print("\n  DOES A GENERATED INTERIOR CHANGE IT?  (Winslow, boundaries held)")
    for name, c in rec["per_config"].items():
        wc = c.get("winslow")
        if not wc or "sector" not in c:
            continue
        raw = c["sector"]["tri_min_scaled_jacobian"]
        shp = c["sector"]["blocks"][wc["worst_block"]]["shape"]
        print(f"    {name:8s} straight-spoke {raw:.6f} -> smoothed "
              f"{wc['min_scaled_jacobian']:.6f}  ({wc['min_scaled_jacobian'] - raw:+.6f}, "
              f"worst {wc['worst_block']} at {shp[0]}x{shp[1]} nodes)")
    print("    the Y's three spokes are BOUNDARIES of two blocks each, so per-block "
          "smoothing holds them.")
    print("    it changes nothing, and that is the finding: the worst corner is ON a "
          "held boundary, so this")
    print("    number is set by where the Y's spokes GO and not by how the interiors are "
          "filled — which is why")
    print("    a CURVED Y is the successor and a better smoother is not.")

    print("\n  THE GENE BOX — this construction is not opt-in, so one genome is not "
          "a measurement")
    for name, c in rec["per_config"].items():
        g = c.get("genomes")
        if not g:
            continue
        print(f"    {name}  B = {g['B']}, fixed w = "
              f"({g['w_fixed'][0]:.3f}, {g['w_fixed'][1]:.3f}, {g['w_fixed'][2]:.3f})")
        print(f"      {'orientation':14s} {'n':>3s} {'fixed w valid':>14s} "
              f"{'best w valid':>13s} {'fixed w min scaled J':>21s}")
        for key, rows in g["groups"].items():
            ok = [r for r in rows if r.get("fixed_w_valid")]
            bw = [r for r in rows if r.get("best_w_valid")]
            rng = (f"{min(r['fixed_w_min_scaled_jacobian'] for r in ok):.4f} - "
                   f"{max(r['fixed_w_min_scaled_jacobian'] for r in ok):.4f}"
                   if ok else "-")
            print(f"      {key:14s} {len(rows):3d} {len(ok):8d}/{len(rows):<5d} "
                  f"{len(bw):7d}/{len(rows):<5d} {rng:>21s}")
        print(f"      ALL            {g['n_genomes']:3d} "
              f"{g['n_fixed_w_valid']:8d}/{g['n_genomes']:<5d} "
              f"{g['n_best_w_valid']:7d}/{g['n_genomes']:<5d}"
              + (f"   clears {MIN_SJ_TARGET}: {g['n_fixed_w_clears_target']}"
                 f"/{g['n_fixed_w_valid']} fixed, {g['n_best_w_clears_target']}"
                 f"/{g['n_best_w_valid']} best"))
        print(f"      every seam closes on the valid ones: {g['all_seams_close']}")
        rows = [r for v in g["groups"].values() for r in v]
        bad = [r for r in rows if not r.get("fixed_w_valid")]
        good = [r for r in rows if r.get("fixed_w_valid")]
        if bad and good:
            sb = sorted(r["arc_span_deg"] for r in bad)
            sg = sorted(r["arc_span_deg"] for r in good)
            print(f"      the ones it folds on are the WIDE arcs — arc span "
                  f"{sb[0]:.3f}-{sb[-1]:.3f} deg against {sg[0]:.3f}-{sg[-1]:.3f} on the "
                  f"ones it does not")
            print(f"      (the shipped genome's is {c['region']['arc_span_deg']:.3f} deg, "
                  f"at the very bottom of the box; the two ranges "
                  f"{'SEPARATE' if sb[0] > sg[-1] else 'OVERLAP'})")
            wb = sorted(r["bow_over_width"] for r in bad)
            wg = sorted(r["bow_over_width"] for r in good)
            print(f"      the arc's BOW over the region's own width separates them "
                  f"{'CLEANLY' if wb[0] > wg[-1] else 'no better'}: "
                  f"{wb[0]:.3f}-{wb[-1]:.3f} against {wg[0]:.3f}-{wg[-1]:.3f} — the "
                  f"straight Y cuts chords,")
            print(f"      and what a chord cannot survive is a side that bows away from "
                  f"one by a fair fraction of the region's width")
        folded = [r for r in rows if r.get("fold", {}).get("folds")]
        if folded and good:
            worst = min((r for r in rows
                         if r.get("fixed_w_min_scaled_jacobian") is not None),
                        key=lambda r: r["fixed_w_min_scaled_jacobian"])
            print(f"      and it is NOT the fold margin, which is the other feasibility "
                  f"number the same draw filter misses:")
            print(f"      {len(folded)}/{len(rows)} of this box describe a spoke that "
                  f"self-intersects, and they sit at fixed-rule "
                  + ", ".join(f"{r['fixed_w_min_scaled_jacobian']:+.4f}"
                              for r in folded)
                  + f" — while the WORST cell in the box, at "
                  f"{worst['fixed_w_min_scaled_jacobian']:+.4f}, has margin "
                  f"{worst['fold']['margin_mm']:+.4f} mm and folds nothing.")
            print(f"      the tri-block partitions the rim JUNCTION and never touches the "
                  f"offset band, so this is the expected answer — recorded because the "
                  f"expectation is worth a number.")

    for name, c in rec["per_config"].items():
        rs = c.get("refusal_search")
        if not rs:
            continue
        print(f"\n  WHAT MAKES A REGION IMPOSSIBLE — THE BOX DRAWN OUT TO "
              f"{rs['n_genomes']} GENOMES ({name})")
        print(f"    {rs['n_refusals']} refuse the curve at every bend and every free "
              f"count, {rs['n_reached']} are reached")
        print(f"      {'quantity':22s} {'refusal':>10s} {'reached range':>22s} "
              f"{'gap':>9s} {'/spread':>8s}")
        for k, v in rs["separation"].items():
            if v is None:
                continue
            rng = f"[{v['reached_min']:.3f}, {v['reached_max']:.3f}]"
            gs = (f"{v['gap_over_spread']:8.3f}" if v["separates"] else f"{'-':>8s}")
            print(f"      {k:22s} {v['refusals'][0]:10.3f} {rng:>22s} "
                  f"{v['gap']:+9.3f} {gs}")
        print(f"    still separating: {', '.join(rs['still_separating'])}")
        print("    a separation that survives a fourfold box and one that decays with it "
              "are different claims —")
        print("    the interior-angle sum went 0.573 -> 0.257 -> 0.041 of the spread over "
              "16, 32 and 64 genomes,")
        print("    while the arc span held at 0.187 -> 0.187 -> 0.176.  NO second refusal "
              "appeared, so every")
        print("    statistic here is still one against many and the arc span is a "
              "CANDIDATE, not a mechanism.")

    print("\n  THE INTERIOR POINT, RE-DERIVED AGAINST THE GENOME BOX -- MEASURED, NOT "
          "ADOPTED")
    for name, c in rec["per_config"].items():
        gr = c.get("genome_robust_w")
        if not gr:
            continue
        cur, best = gr["at_current_w"], gr["best"]
        print(f"    {name}  n_cells = {gr['n_cells']} ({gr['n_fixable']} drawn + shipped), "
              f"excluded (no `w` rescues them): "
              f"{[round(a, 1) for a in gr['excluded_arc_span_deg']]} deg")
        print(f"      current w {tuple(round(x, 3) for x in cur['w'])}: "
              f"worst {cur['worst_min_scaled_jacobian']:.4f}, "
              f"{cur['n_valid']}/{gr['n_cells']} valid, "
              f"{cur['n_clear']}/{gr['n_cells']} clear {MIN_SJ_TARGET}")
        print(f"      genome-robust w {tuple(round(x, 3) for x in best['w'])}: "
              f"worst {best['worst_min_scaled_jacobian']:.4f}, "
              f"{best['n_valid']}/{gr['n_cells']} valid, "
              f"{best['n_clear']}/{gr['n_cells']} clear {MIN_SJ_TARGET}   "
              f"({gr['n_fully_valid_cells']}/{gr['grid_n']} grid cells are fully valid)")
        print(f"      the SHIPPED genome at the genome-robust w: "
              f"{gr['shipped_min_scaled_jacobian_at_best']:.4f} "
              f"(clears {MIN_SJ_TARGET}: {gr['shipped_clears_target_at_best']}), "
              f"against {gr['shipped_min_scaled_jacobian_at_current']:.4f} today")
    print("    a fixed rule exists that reaches nearly every genome its own best-per-genome")
    print("    point can reach, at a per-config cost to the shipped genome's own margin "
          "shown above (zero at")
    print("    some configs, real at others).  Reported and not shipped as THE rule, on "
          "§53's `blend 0.0`")
    print("    precedent: nothing yet reads `chosen` besides this file's own headline "
          "table, so adopting it")
    print("    changes a quoted number and nothing else.")

    print("\n  THE CURVED Y — the spokes FOLLOW the region instead of cutting across it")
    for name, c in rec["per_config"].items():
        cy = c.get("curved_y")
        if not cy:
            continue
        b, bs, sh = cy["best"], cy["best_straight"], cy["shipped"]
        print(f"    {name}  B = {cy['B']}, {cy['grid_n']} interior points x "
              f"{len(cy['bend_grid'])} bends, {cy['n_genomes']} genomes "
              f"({cy['n_genomes'] - 1} drawn + shipped)")
        print(f"      per-genome ceiling   straight {cy['n_straight_valid']}"
              f"/{cy['n_genomes']} valid -> curved {cy['n_curved_valid']}"
              f"/{cy['n_genomes']}, {cy['n_rescued_by_the_curve']} RESCUED by the bend "
              f"alone")
        print(f"      (the straight half is RE-MEASURED here on this sweep's own "
              f"{cy['grid_n']}-point grid, so it is not the gene box block's "
              f"`best w valid` above, which is on a {c['genomes']['x_grid_n']}-point one)")
        print(f"      one fixed rule       straight {bs['n_valid']}/{cy['n_cells']} valid, "
              f"{bs['n_clear']} clear {MIN_SJ_TARGET}, worst "
              f"{bs['worst_min_scaled_jacobian']:+.4f}")
        print(f"                             curved {b['n_valid']}/{cy['n_cells']} valid, "
              f"{b['n_clear']} clear {MIN_SJ_TARGET}, worst "
              f"{b['worst_min_scaled_jacobian']:+.4f}  at bend {b['bend']:.2f}, w "
              f"{tuple(round(x, 3) for x in b['w'])}")
        pl = cy["plateau_at_best_bend"]
        frac = pl["n_within_10pc"] / pl["n_cells"]
        shape = "SHELF" if frac >= 0.10 else "ridge" if frac >= 0.01 else "SPIKE"
        print(f"      and the winning cell is a {shape}: {pl['n_within_10pc']}"
              f"/{pl['n_cells']} interior points at that bend are within 10% of it — "
              f"the argmax at each bend runs")
        print("        " + "  ".join(
            f"{r['bend']:.1f}:{r['n_valid']}/{r['n_clear']}" for r in cy["per_bend"])
            + f"   (valid/clear, out of {cy['n_cells']})")
        pub = sh["published_cell"]["min_scaled_jacobian"]
        print(f"      the SHIPPED genome   {pub:.4f} at the published cell, "
              f"{sh['best']['min_scaled_jacobian']:.4f} at the joint curved rule")
        sa = cy["sector_at_best"]
        print(f"      all {sa['n_seams']} seams still close at that bend: "
              f"{sa['seams_close']} (max gap {sa['max_seam_gap_mm']:.2e} mm), and the "
              f"three quads still tile the region to "
              f"{sa['area_relative_difference']:.1e}")
        print(f"      {'bow/width':>9s} {'arc deg':>8s} {'straight':>9s} "
              f"{'curved':>9s} {'bend':>5s}")
        for r in sorted(cy["per_genome"], key=lambda x: -x["bow_over_width"]):
            f = (lambda v: "FOLD" if v is None else "%+.4f" % v)
            tag = ("  <- SHIPPED" if r["shipped_genome"] else
                   "  RESCUED" if r["rescued_by_the_curve"] else
                   "  refuses at every bend" if not r["curved_valid"] else "")
            bd = "--" if r["curved_bend"] is None else "%.2f" % r["curved_bend"]
            print(f"      {r['bow_over_width']:9.3f} {r['arc_span_deg']:8.2f} "
                  f"{f(r['straight_min_scaled_jacobian']):>9s} "
                  f"{f(r['curved_min_scaled_jacobian']):>9s} {bd:>5s}{tag}")
        for r in cy["refusals"]:
            pb = ", ".join(f"B={x['B']} {x['ceiling']:+.4f}" for x in r["per_B"])
            print(f"      the {r['arc_span_deg']:.1f}-deg refusal is not the free count "
                  f"either — over EVERY admissible B its ceiling is "
                  f"{r['ceiling_over_every_B']:+.4f}, valid at none")
            print(f"        ({pb})")
    print("    the bend is INERT where the region is fat and decisive where it is a "
          "sliver, which is the")
    print("    same statement as the bow column: it is a correction to cutting chords, "
          "and a fat region's")
    print("    chords needed no correction.  But the bow does NOT explain the survivor — "
          "the LARGEST bow in")
    print("    the box is a genome the curve reaches, and the one that refuses has a "
          "smaller one.  So the bow")
    print("    says where the bend is needed and not what makes a region impossible, and "
          "that is still open.")
    print("    Measured, not adopted — `bend` defaults to 0.0 and every number above "
          "this block is the")
    print("    straight Y this file published first.")

    print("\n  SELF-CHECKS")
    for k, v in rec["self_checks"].items():
        if k == "pass":
            continue
        print(f"    {k:44s} {'PASS' if v else 'FAIL'}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genome", default="best_solution.json")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    ap.add_argument("--out", default="study_tri_block.json")
    args = ap.parse_args()

    configs = tuple(c for c in args.configs.split(",") if c)
    _gate_guard.refuse_degraded_out(ap, args, "study_tri_block.json", [
        (set(configs) != set(DEFAULT_CONFIGS),
         f"--configs {args.configs} is not the committed "
         f"{','.join(DEFAULT_CONFIGS)}"),
        (args.genome != "best_solution.json",
         f"--genome {args.genome} is not the shipped genome"),
    ])

    t0 = time.time()
    genes = load_genes(args.genome)
    rec = build(genes, configs)
    rec["genome"] = args.genome
    rec["seconds"] = round(time.time() - t0, 2)
    out = os.path.join(HERE, args.out)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
    _print(rec)
    print(f"  wrote {out}  ({rec['seconds']} s)")
    raise SystemExit(0 if rec["self_checks"]["pass"] else 1)


if __name__ == "__main__":
    main()
