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


def region_report(reg):
    """The three vertices, the three side lengths, and the vertex that is not one.

    The `far_end` turn is the whole finding of UNCAP_PLAN Step 2 restated as a number: at
    blend 0 it is within a degree of straight, so the fourth vertex a quad block needs
    does not exist on this region.
    """
    B_dense = reg["B_dense"]
    k = B_CURVE_SAMPLES - 1
    arc_len = abs(reg["th_q"] - reg["th_t"]) * reg["rim_inner"]
    return {
        "P_t": [float(x) for x in reg["P_t"]],
        "Q": [float(x) for x in reg["Q"]],
        "B_star": [float(x) for x in reg["B_star"]],
        "far_end": [float(x) for x in reg["far_end"]],
        "arc_span_deg": float(math.degrees(abs(reg["th_q"] - reg["th_t"]))),
        "A_length_mm": float(arc_len),
        "B_length_mm": float(np.linalg.norm(np.diff(B_dense, axis=0), axis=1).sum()),
        "C_length_mm": float(np.linalg.norm(np.diff(reg["cross"], axis=0),
                                            axis=1).sum()),
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


def tri_sector(reg, B, w):
    """The twelve node grids of a faithful-rim sector 0, or `None` if `B` is inadmissible.

    THE THREE NEIGHBOURS ARE SLICED, NOT REBUILT.  `spoke`, `hub_junction` and
    `rim_band_weld` come out of `sector_blocks` and are cut at a node index, so their
    geometry is bit-for-bit what the tree builds today and every number below is a
    consequence of the PARTITION rather than of a second construction of the same thing.
    Only `rim_junction` is replaced.

    `w` is a barycentric weight triple on (P_t, Q, B*) for the interior point.
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
           "w": [float(x) for x in w], "X": [float(x) for x in X],
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

def x_grid(n=X_GRID_N, lo=X_GRID_LO, hi=X_GRID_HI):
    """The barycentric weights swept, as (w_Pt, w_Q, w_Bstar) triples."""
    out = []
    for u in np.linspace(lo, hi, n):
        for v in np.linspace(lo, hi, n):
            if u + v >= 1.0 - lo:
                continue
            out.append((float(u), float(v), float(1.0 - u - v)))
    return tuple(out)


def cell(reg, B, w):
    """One (B, X) cell: the three quads' validity, and nothing else.

    The sweep runs on the Y alone rather than on the whole sector because the other nine
    blocks do not depend on either swept quantity -- `spoke`, `hub_junction` and
    `rim_band_weld` are CUT at a node index and their interiors never move.  That is
    checked once, in `self_checks`, rather than re-measured 900 times.
    """
    built = tri_sector(reg, B, w)
    if built is None:
        return None
    blocks, aux = built
    q = {k: fbk.block_quality(np.asarray(blocks[k], float)) for k in TRI_BLOCKS}
    return {"B": int(B), "w": [float(x) for x in w],
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

def sector_verdict(reg, B, w):
    """All twelve blocks and all seventeen seams at one cell."""
    blocks, aux = tri_sector(reg, B, w)
    per = {k: fbk.block_quality(np.asarray(v, float)) for k, v in blocks.items()}
    tbl = seam_table(reg, aux)
    sm = seams(blocks, tbl)
    quad_area = block_area_mm2(reg["base"]["rim_junction"])
    tri_area = sum(block_area_mm2(blocks[k]) for k in TRI_BLOCKS)
    return {
        "B": int(B), "w": [float(x) for x in w], "aux": aux,
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
