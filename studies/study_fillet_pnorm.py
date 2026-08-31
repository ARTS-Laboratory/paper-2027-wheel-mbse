"""
=============================================================================
  THE REGION-RESTRICTED P-NORM OVER THE FILLET ARC — THE SMOOTH REGION
  WEIGHT, AND THE EXPONENT SWEEP THAT SETS ITS `p`
=============================================================================
    .venv-opt/bin/python studies/study_fillet_pnorm.py       (make filletpnorm)

PLAN.md §94's ranked successor 1 — items 1 and 2 of "WHAT REPLACES `Kt * agg`".
FILLET_PLAN.md STEP 3.

WHY THIS EXISTS
---------------
§94 measured that the objective's stress term is inert on a filleted mesh — every
utilisation under `MARGIN_KNEE_UTIL`, both barriers exactly zero in value and gradient —
while `arc_peak`, the mesh's own converged reading of the fillet surface, calls the hub
breached at both designs.  Its proposed replacement keeps the two-junction two-barrier
shape and swaps the modelled factor for the measured field:

    util_j  =  sigma_fillet_j(mesh) / ALLOWABLE_STRESS_MPA          no Kt

and it put four things in front of that, of which this driver measures the first two:

  1.  IT MUST BE DIFFERENTIABLE, AND `arc_peak` IS AN ARGMAX.  "Replace with a
      volume-weighted p-norm over the tube and sweep the exponent on filleted solves the
      way M8b-i.6 swept the global one.  CHECK: the largest `p` whose observed order is
      still ~2."
  2.  THE REGION MOVES WITH THE GENES.  "The tube is fixed in mesh terms but its
      membership flips as the arc moves with `R_hub`/`R_rim`/`t0`/`t3`, and a flipping
      membership is a step in the loss.  Needs a smooth weight in distance-to-arc, not an
      indicator."

THE QUANTITY
------------
    sigma_fillet_j(p, r) = ( SUM_g W_g V_g vm_g^p / SUM_g W_g V_g )^(1/p)     [MPa]

        W_g = max(0, 1 - d_g^2 / r^2)^3            the SMOOTH region weight
        d_g = distance from Gauss point g to junction j's ANALYTIC fillet arc
        V_g = the Gauss point's own volume weight, `wheel_fem._volume_kernel`

read on the loaded rotational copy, which is `arc_peak`'s rule and is here for its reason:
the wheel is loaded at one contact patch and the twelve images of a fillet see twelve
loads.  `width` is left out of `V_g` deliberately — it is a constant and cancels between
numerator and denominator, and carrying it would imply this driver knows which width the
objective will use.

WHY THAT KERNEL AND NOT A GAUSSIAN.  Three properties, all of them needed:

  *  IT IS A POLYNOMIAL IN `d^2`, so it never forms `sqrt(d^2)` and never meets the kink
     `|x|` has at zero — which is exactly where the weight is largest.
  *  IT IS C2 AT ITS OWN SUPPORT BOUNDARY.  Value, first and second derivative all vanish
     at `d = r`, so a Gauss point entering or leaving the region does so with zero weight
     and zero slope.  That is what item 2 asks for, and section C measures what its
     absence costs: an indicator steps 4.30% in the region mass and 1.08% / 0.27% in the
     measure at `p` = 4 / 16 at §52's tube radius, 0.0592 mm from the shipped `R_hub`;
     the bump sits at the sampling floor at every radius and exponent.
  *  ITS SUPPORT IS COMPACT AND IS EXACTLY `arc_peak`'s TUBE.  A Gaussian would have the
     first two properties and an unbounded tail, which puts the whole wheel back into the
     measure at low `p` — the dilution §94 measured `Kt * agg` losing 1.68x to 2.76x to.

`d_g` ITSELF IS SMOOTH WHERE IT MATTERS, and this is measured rather than assumed.
`study_corner_singularity._distance_to_arc` folds to the endpoints outside the sweep; the
fold is C1 (both branches give `|r - R|` with zero angular derivative at the tangent
point), and the one genuine kink it has — `min(dA, dB)` on the chord's perpendicular
bisector — sits on the far side of the circle at a distance of about `2R`, where `W` is
identically zero.  Inside the sweep `d = |r - R|` and the absolute value never activates:
the fillet is concave, its centre of curvature is in the VOID, and the closest Gauss point
to either centre measures `r / R` = 1.085 (hub) and 1.011 (rim) at the shipped genome.

WHAT THE SWEEP FOUND — §94's EXPECTATION HOLDS, AND IT IS NOT THE BINDING CONSTRAINT
--------------------------------------------------------------------------------------
§94 wrote: "the region is small and far more uniform than the whole wheel, so it should
tolerate a higher `p` than the global 4.0 — that is a measurement and not an assumption."
**Measured, it does, everywhere the region is resolved**, and by a long way: `p` = 30 at
the rim, which is the top of this sweep and never breaks, and 24 / 16 at the hub.  Against
`STRESS_NOMINAL_P` = 4.0 that is between four and seven times the global exponent.

**BUT THE EXPONENT IS NOT WHAT DECIDES WHETHER THE QUANTITY CONVERGES, AND A SECOND
PARAMETER HAS TO BE SET FIRST.**  `SUM_g W_g V_g` — the measure's own denominator — is a
quadrature of `INTEGRAL W dV` over a region fixed ANALYTICALLY, with no displacement field
in it at all, so its ladder is pure quadrature error and it bounds every claim the
numerator can make.  At the tube radius §52's `arc_peak` uses (0.30 mm) that mass moves
3.04% between `medium` and `fine` at the shipped hub and 2.85% at `b029622` — while at the
rim, same radius, it moves 0.008% and 0.256%.

The diagnostic that explains the split is reported beside every cell.  A junction's fillet
is meshed as two blocks and its `b` half — the one welded to the ring — carries the coarser
first element layer: 0.358 / 0.237 / 0.178 mm up the ladder at the shipped hub against
0.156 / 0.103 / 0.077 at the rim.  So a 0.30 mm tube is **1.7 cells deep at the hub and
3.9 at the rim** at the finest rung, and a quadrature rule cannot resolve a weight function
it samples fewer than about twice across.

So the sweep had to be two-dimensional — the exponent AND the tube radius — because the
second is what decides whether the first means anything, and `RADII_MM` is a sweep axis
rather than a constant.  Widening the tube fixes the hub: it resolves at 0.45 mm (mass
drift 0.095% and 0.205%) and stays resolved at 0.90.

WHAT IS RECOMMENDED, AND WHY IT IS A CROSS-CELL ANSWER
--------------------------------------------------------
`p` and `r` are constants of the objective, not per-design choices, so whatever is picked
has to hold at every junction of every genome the optimizer can reach.  Two of the five
radii are resolved in all four cells — 0.45 and 0.90 — and the smaller is preferred
because the term exists to be LOCAL to the fillet surface.  At `r` = 0.45 mm the largest
exponent clearing observed order 1.50 in every cell, on both `h` definitions, is `p` = 16:

    cell             sigma_fillet   util    order   GCI     what the objective reads today
    shipped/hub        28.5385     1.1415    2.33   1.11%              0.5803
    shipped/rim        12.6678     0.5067    2.71   0.70%              0.3859
    b029622/hub        41.8999     1.6760    1.77   2.09%              0.7954
    b029622/rim        19.6115     0.7845    3.95   0.37%              0.5057

The hub is over the allowable at BOTH designs under the replacement, which is §94's
finding surviving the move from `arc_peak` to a differentiable proxy.

AND THE PROXY STILL UNDERSTATES, WHICH IS THE P-NORM'S DOCUMENTED BEHAVIOUR AND NOT A
DEFECT TO BE FIXED BY RAISING `p`.  `_qoi_pnorm_stress`: it "is a smooth lower bound that
approaches the max from below as p grows", and at `p` = 16 it sits at 77.6% to 79.5% of
`arc_peak` — the surface peak is 1.26x to 1.29x the replacement's reading.  Against the
1.68x to 2.76x §94 measured for `Kt * agg` that is most of the gap closed and none of it
denied, and "conservative" is the wrong word for either: both err in the direction of
calling a breached fillet safe, and the replacement errs less.  `p` cannot be raised past
16 to close the rest, because 16 is where the ORDER floor binds — which is the whole
content of the sweep.

THE TWO JUNCTIONS DISAGREE ABOUT `p`, AND THAT IS A RESULT RATHER THAN NOISE.  At the rim
the observed order RISES with `p` — 2.2 to 3.1 across the sweep at `r` = 0.30 — so the
floor never binds and `p` = 30 passes.  At the hub it FALLS — 8.0 down to 1.5 at
`r` = 0.45 — so the floor binds from above, which is M8b-i.6 step 1's shape.  Both limits
are consistent with the same mechanism: as `p` grows the p-norm tracks the local peak, so
its order approaches `arc_peak`'s own (1.28 at the hub, 2.17 at the rim, on `h_fillet`),
and the hub's peak is the one this ladder resolves worst.  The hub's low-`p` orders of 6
to 11 come with GCI of 0.01% and are the opposite artefact — three points a hair apart
give Roache an error ratio near zero and he answers with a large `p`, which is not a
measurement of anything.

`h` IS THE FILLET BLOCK'S, NOT THE WHEEL'S, AND BOTH ARE CARRIED
------------------------------------------------------------------
`study_deflection_gci`'s H_DEFS block is the rule and its warning is the precedent: it
drew `h` from the wrong mesh once and inflated every reported `p` by 1.25x.  This measure
is LOCAL to the fillet block, whose grid goes 5x5 / 9x9 / 13x13 / 17x17 nodes — linear
refinement ratios 2.0 / 1.5 / 1.333, against the whole wheel's 1.617 / 1.556 by element
count.  Those are different ladders and they give different `p` off the same numbers, so
`h_fillet` is primary, `h_global` is carried, and no verdict here is stated except where
the two agree.

WHAT IS OUT OF SCOPE, AND IT IS THE SAME SCOPE §94 STATED
-----------------------------------------------------------
LINEAR kinematics at ONE phase, which is `study_corner_singularity`'s ladder and therefore
the one `arc_peak`'s numbers were taken on.  Both gaps raise the measured side.  This
driver does not touch `wheel_objective`, does not wire anything into the loss, and takes
no view on the exchange rate (§94 item 4) or on the `P_c` exclusion (§94 item 3).

It does report two things those two items need as inputs.  The distance from each arc to
each reference corner, which is the fact item 3's argument rests on and which comes out at
2.75 mm at the hub and 6.50 at the rim — nine and twenty-two tube radii clear, so the
exclusion is a geometric fact here and not a modelling choice.  And what the RAW
substitution fires at today's weights, which is item 4's input: `soft_barrier` at the
recommended cell reads `stress` 80.13 and `stress_margin` 37.91 at the shipped genome and
1827.88 and 249.39 at `b029622`, against `0.0000` for both at both designs today.
"""

import argparse
import json
import math
import os
import time

import numpy as np

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import _gate_guard

import jax_config  # noqa: F401
import wheel_wheel as WW
import wheel_fem as fem
import wheel_fea as WF
import wheel_genome as WG
import wheel_objective as WO
import study_corner_singularity as CS
import study_deflection_gci as SG

HERE = os.path.dirname(os.path.abspath(__file__))

GENOMES = (("shipped", "best_solution.json"),
           ("b029622", "fillet_optimum_b029622.json"))

# `smoke` is a DIAGNOSTIC and is excluded from every Richardson, for
# `study_deflection_gci`'s reason: a three-point extrapolation is only meaningful on
# points inside one asymptotic range, and the fillet block is 4x4 elements at `smoke`.
LADDER = ("smoke", "coarse", "medium", "fine")
EXTRAPOLATE_FROM = ("coarse", "medium", "fine")

# The tube radii.  0.30 is `study_corner_singularity.PROBE_RADIUS_MM`, the region
# `arc_peak` reads and therefore the only radius at which this measure and §52's number
# are the same region; the rest bracket it by a factor of three either side, because the
# module docstring's finding is that the radius and not the exponent is what decides
# whether the ladder resolves the region.
RADII_MM = (0.20, 0.30, 0.45, 0.60, 0.90)
REFERENCE_RADIUS_MM = CS.PROBE_RADIUS_MM

# M8b-i.6 step 1 swept ten exponents; this sweeps nine over the same span, bracketing both
# `STRESS_NOMINAL_P` (4.0, what the objective's global p-norm uses) and
# `WA.STRESS_PNORM_P` (30.0, the adjoint's documented default).
EXPONENTS = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 30.0)

# The indicator is carried as the CONTROL, not as a candidate.  It is what `arc_peak`
# uses and what §94 item 2 says may not be used, and section C is the measurement of the
# difference; without it in the tables there is nothing for the smooth weight to be
# smoother THAN.
KERNELS = ("bump3", "indicator")

JUNCTIONS = ("hub", "rim")

# Section C.  `R_hub` because it is the gene the replacement term exists to give a live
# gradient to — §15 DEFECT 1 is that `stress` and the fillet barriers are the only paths
# from `R_hub` into the loss — and the hub because §94 measured it at 1.46x and 2.20x the
# allowable while the objective read 0.58 and 0.80.
SMOOTH_GENE = "R_hub"
SMOOTH_CONFIG = "coarse"
# HOW FAR TO LOOK FOR A CROSSING, AND WHY IT IS THIS WIDE.  +-0.10 mm at 2.5e-3 was the
# first setting and it found two crossings at `r` = 0.30 mm and NONE at 0.45 — the wider
# tube's boundary happens to run through a sparser part of the mesh near this genome.  A
# null there would have left the defect unpriced at the very radius the recommendation
# names, so the window is widened rather than the null reported: same 101 builds, 2.5x the
# range.  Clamped to the gene's own box, since a scan outside it measures nothing.
SMOOTH_HALF_WIDTH_MM = 0.25
SMOOTH_SCAN_STEP_MM = 5.0e-3
SMOOTH_BISECT_TOL_MM = 1e-9     # far below the finest sampling step below
SMOOTH_STEPS_MM = (2.5e-4, 2.5e-5)   # a decade apart: a jump stays, a slope shrinks 10x
SMOOTH_SAMPLES = 9
SMOOTH_P = 4.0                  # `STRESS_NOMINAL_P`, so the step is quoted at the
                                # exponent the objective's own nominal is built at


# ---------------------------------------------------------------------------
# THE MEASURE
# ---------------------------------------------------------------------------

def region_weight(d2, r_sup, kernel):
    """`W` from the SQUARED distance to the arc — see the module docstring.

    Taking `d2` rather than `d` is the point and not an optimisation: `bump3` is a cubic
    in `1 - d^2/r^2`, so on this path the square root is never formed and the kink `|x|`
    has at zero — which sits exactly on the arc, where the weight peaks — never exists.
    """
    u = np.maximum(0.0, 1.0 - d2 / (r_sup * r_sup))
    return (u > 0.0).astype(float) if kernel == "indicator" else u ** 3


def _rotations(n_sp):
    """The `n_sp` rotations that bring each spoke's copy onto sector 0's arc.

    `arc_peak` rotates the FIELD and holds the analytic arc still, and this follows it
    exactly: rotating the arc instead would refit a circle per copy and put the fit's
    7e-14 mm residual into a comparison between copies.
    """
    a = -2.0 * np.pi * np.arange(n_sp) / n_sp
    return np.cos(a), np.sin(a)


def region_pnorm(arc, xy, vm, vol, n_sp, exponents, r_sup, kernel):
    """Every exponent at one radius, off ONE field, on the loaded copy.

    ONE SOLVE, NINE EXPONENTS, FIVE RADII — M8b-i.6 step 1's property and its reason.
    The p-norm is a pure function of the converged displacement field, so re-solving per
    `p` would buy nothing and would turn a four-minute driver into hours.

    THE LOADED COPY IS PICKED ON THE LARGEST EXPONENT and the choice is then held for all
    of them, so a row of this table is one region on one copy rather than nine regions
    that might not be the same one.  Picking per exponent would let low `p` — which is
    nearly a volume average and barely sees the load — choose a different spoke from the
    one carrying the peak, and the ladder would then be comparing two features.  The copy
    is reported so a change up the ladder is visible; measured, it does not change
    (copy 9 at the hub and 8 at the rim, every rung, both genomes).

    `vm` is raised to `p` on the SUPPORTED points only.  Outside the support `W` is
    exactly 0.0 and `0.0 * vm**30` is 0.0, so the mask changes no value — it keeps
    `vm ** 30` off 50 000 Gauss points that contribute nothing.
    """
    ca, sa = _rotations(n_sp)
    best, best_k, best_mass, best_n = None, None, 0.0, 0
    for k in range(n_sp):
        rot = np.column_stack([ca[k] * xy[:, 0] - sa[k] * xy[:, 1],
                               sa[k] * xy[:, 0] + ca[k] * xy[:, 1]])
        d = CS._distance_to_arc(rot, arc)
        w = region_weight(d * d, r_sup, kernel) * vol
        tot = float(w.sum())
        if tot <= 0.0:
            continue
        m = w > 0.0
        wm, vmm = w[m], vm[m]
        vals = {p: float((np.sum(wm * vmm ** p) / tot) ** (1.0 / p)) for p in exponents}
        if best is None or vals[exponents[-1]] > best[exponents[-1]]:
            best, best_k, best_mass = vals, k, tot
            best_n = int((d < r_sup).sum())
    return {"values_mpa": {"%g" % p: v for p, v in best.items()},
            "loaded_copy": best_k, "region_mass_mm3": best_mass,
            "n_gauss_in_tube": best_n}


def region_mass(arc, xy, vol, n_sp, r_sup, kernel, copy=0):
    """`SUM_g W_g V_g` on ONE named copy, with no displacement field anywhere in it.

    THE DIAGNOSTIC THE WHOLE OF SECTION A RESTS ON.  The arc is analytic and does not move
    with the mesh, so this is a quadrature of a fixed integral and its ladder is pure
    quadrature error — nothing about the solve, the load, the kinematics or the stress can
    enter it.  It is the denominator of every `region_pnorm` above, so whatever it fails
    to converge at, nothing built on it converges better.

    On a fixed genome any copy gives the same number to rotational symmetry, so `copy=0`
    is not a choice about where to look; it is a statement that there is nothing to look
    for.
    """
    ca, sa = _rotations(n_sp)
    rot = np.column_stack([ca[copy] * xy[:, 0] - sa[copy] * xy[:, 1],
                           sa[copy] * xy[:, 0] + ca[copy] * xy[:, 1]])
    d = CS._distance_to_arc(rot, arc)
    return float((region_weight(d * d, r_sup, kernel) * vol).sum())


def gauss_geometry(mesh, cfg):
    """`(gauss xy [n, 2], gauss volume weights [n])` for a built mesh, no solve.

    The coordinates come from `fem.gauss_stresses` with a zero displacement and unit
    moduli, which is the same `xy` a real solve reports — it is `N @ Xe`, and neither the
    shape functions nor the node coordinates know what `u` was.  Calling the reporting
    path rather than re-deriving `N @ Xe` here is `gauss_stresses`' own rule: two copies
    of a geometric map is the drift a shared kernel exists to prevent.
    """
    xy = np.asarray(fem.gauss_stresses(mesh.coords, mesh.conn,
                                       np.zeros(2 * mesh.n_nodes), order=cfg.order,
                                       lam=1.0, mu=1.0)["xy"]).reshape(-1, 2)
    vol = np.asarray(fem._volume_kernel(cfg.order)(
        np.asarray(mesh.coords)[np.asarray(mesh.conn)])).reshape(-1)
    return xy, vol


# ---------------------------------------------------------------------------
# THE RESOLUTION DIAGNOSTIC — WHY A CELL OF THE GRID CONVERGES OR DOES NOT
# ---------------------------------------------------------------------------

def first_layer_depth_mm(genes, cfg, label):
    """The depth of the element layer that touches the arc, per fillet half.

    THE NUMBER THAT EXPLAINS EVERY BAD CELL IN SECTION B.  A junction's fillet is meshed
    as two blocks — `a` toward the junction, `b` toward the ring weld — and both start ON
    the arc and run away from it.  `r_sup` divided by this is how many element layers deep
    the tube is, and a quadrature cannot resolve a weight it samples fewer than twice
    across.  The two halves are reported separately because they are not close: at the
    shipped hub the `a` half is 0.144 mm at `coarse` and the `b` half is 0.358, and the
    `b` half is the one that decides.
    """
    b = WW.sector_blocks(genes, cfg, fillet=True)
    arc = CS.fillet_arcs(genes, cfg, True)[label]
    out = {}
    for half in ("a", "b"):
        g = np.asarray(b["%s_fillet_%s" % (label, half)])
        d = CS._distance_to_arc(g.reshape(-1, 2), arc).reshape(g.shape[:2])
        out[half] = float(np.abs(d[:, 1] - d[:, 0]).mean())
    return out


def fillet_block_elements(genes, cfg):
    """Elements along one side of a fillet block — the local `h`'s denominator.

    4 / 8 / 12 / 16 up the ladder against the wheel's 960 / 5952 / 15552 / 37632 elements,
    i.e. LINEAR ratios 2.0 / 1.5 / 1.333 where `1/sqrt(n_elements)` gives 1.617 / 1.556.
    Both are defensible and they are not the same ladder; see the module docstring.
    """
    return int(np.asarray(WW.sector_blocks(genes, cfg,
                                           fillet=True)["hub_fillet_a"]).shape[0]) - 1


def corner_clearances_mm(genes, cfg):
    """Distance from each fillet arc to each of `probe_points`' named corners.

    §94 item 3 — "`P_c` is excluded by construction and the exclusion must be argued, not
    assumed" — needs one fact before it can be argued at all: that the tube does not
    contain `P_c`.  It does not, and by a wide margin.  Recorded here rather than left to
    the reader because a region measure that silently swallowed a divergent corner would
    converge exactly as badly as section A's worst cells do, for a completely different
    reason.
    """
    arcs = CS.fillet_arcs(genes, cfg, True)
    pts = CS.probe_points(genes, cfg, True)
    return {lab: {n: float(CS._distance_to_arc(np.array([p]), arc)[0])
                  for n, p in sorted(pts.items())}
            for lab, arc in arcs.items()}


# ---------------------------------------------------------------------------
# SECTIONS A AND B — THE LADDER, AND EVERY (radius, exponent) OFF IT
# ---------------------------------------------------------------------------

def _richardson(phi, h):
    """`study_deflection_gci.richardson`, with its one deflection-shaped key renamed.

    That module returns `extrapolated_mm` because its QoI is an axle drop.  Ours is a
    stress in MPa and filing it under a `_mm` name would be a unit error sitting in a
    committed artifact.  Renamed here and not there: the key is that module's contract
    with its own artifact and its own tests, and this is the only caller that needs it to
    say something else.
    """
    out = SG.richardson(list(phi), list(h))
    out["extrapolated_mpa"] = out.pop("extrapolated_mm")
    return out


def run_ladder(genome_path, ladder=LADDER, radii=RADII_MM, exponents=EXPONENTS):
    """One genome, one filleted linear solve per rung, every cell off those solves."""
    genes = CS.load_genes(genome_path)
    n_sp = WW.NUMBER_OF_SPOKES
    rungs = []
    for name in ladder:
        cfg = WW.get_config(name)
        t0 = time.time()
        mesh, res, xy, vm = CS.solve_field(genes, cfg, fillet=True)
        _, vol = gauss_geometry(mesh, cfg)
        arcs = CS.fillet_arcs(genes, cfg, True)
        n_fil = fillet_block_elements(genes, cfg)
        rec = {"config": name, "n_elements": int(mesh.n_elements),
               "n_nodes": int(mesh.n_nodes), "n_fillet_elements": n_fil,
               "h_fillet": 1.0 / n_fil,
               "h_global": 1.0 / math.sqrt(mesh.n_elements),
               "axle_drop_mm": float(res["axle_drop_mm"]),
               "global_max_vm_mpa": float(vm.max()),
               "layer_profile": [float(v) for v in
                                 WW.per_genome_layer_profile(genes, cfg, fillet=True)],
               "seconds": None, "junctions": {}}
        for lab in JUNCTIONS:
            arc = arcs[lab]
            cell = first_layer_depth_mm(genes, cfg, lab)
            j = {"arc_radius_mm": float(arc["radius"]),
                 "arc_sweep_deg": math.degrees(abs(arc["a1"] - arc["a0"])),
                 "arc_fit_residual_mm": float(arc["fit_residual_mm"]),
                 "first_layer_depth_mm": cell,
                 "arc_peak_mpa": CS.arc_peak(arc, xy, vm, n_sp)["peak_vm_mpa"],
                 "cells": {}}
            for r in radii:
                cell_rec = {"cells_across": {h: r / cell[h] for h in ("a", "b")}}
                for kn in KERNELS:
                    cell_rec[kn] = region_pnorm(arc, xy, vm, vol, n_sp, exponents, r, kn)
                    # AND THE SAME MASS WITH THE FIELD REMOVED FROM THE COPY CHOICE TOO.
                    # `region_pnorm` reports its own denominator, but it picked the copy
                    # by maximising the p-norm — so that number is field-free in its
                    # value and not in its provenance, and section A's whole claim is
                    # that no field enters it.  Copy 0 costs one more rotation and makes
                    # the claim literally true; the two agree to `mass_copy_spread`
                    # below, which is what says the distinction was immaterial.
                    m0 = region_mass(arc, xy, vol, n_sp, r, kn, copy=0)
                    cell_rec[kn]["region_mass_copy0_mm3"] = m0
                    cell_rec[kn]["mass_copy_spread_rel"] = abs(
                        m0 / cell_rec[kn]["region_mass_mm3"] - 1.0)
                j["cells"]["%g" % r] = cell_rec
            rec["junctions"][lab] = j
        rec["seconds"] = time.time() - t0
        rungs.append(rec)
        print("    %-7s elem %6d  fillet block %2dx%-2d  h_fil %.5f  drop %.5f  "
              "global vm max %8.2f MPa  %5.1fs"
              % (name, rec["n_elements"], n_fil, n_fil, rec["h_fillet"],
                 rec["axle_drop_mm"], rec["global_max_vm_mpa"], rec["seconds"]),
              flush=True)
    return genes, rungs


def converge(rungs, extrapolate_from=EXTRAPOLATE_FROM, radii=RADII_MM,
             exponents=EXPONENTS):
    """Richardson on every cell, on BOTH `h` definitions.

    Reported as a pair rather than a choice, and the verdict below only speaks where the
    pair agrees — `study_deflection_gci`'s rule, and the reason it carries four `h`
    definitions instead of one.
    """
    idx = {r["config"]: r for r in rungs}
    rows = [idx[c] for c in extrapolate_from]
    hs = {"h_fillet": [r["h_fillet"] for r in rows],
          "h_global": [r["h_global"] for r in rows]}
    out = {}
    for lab in JUNCTIONS:
        js = [r["junctions"][lab] for r in rows]
        e = {"arc_peak": {k: _richardson([j["arc_peak_mpa"] for j in js], h)
                          for k, h in hs.items()}, "radii": {}}
        for r in radii:
            key = "%g" % r
            cells = [j["cells"][key] for j in js]
            rr = {"cells_across_fine": cells[-1]["cells_across"]}
            for kn in KERNELS:
                mass = [c[kn]["region_mass_copy0_mm3"] for c in cells]
                rr[kn] = {
                    "region_mass_mm3": [float(v) for v in mass],
                    "mass_copy_spread_rel": max(c[kn]["mass_copy_spread_rel"]
                                                for c in cells),
                    # THE BOUND ON EVERYTHING ELSE IN THIS BLOCK.  No field enters the
                    # mass, so a mass that moves 3% between the last two rungs is 3% of
                    # quadrature error sitting under every exponent beside it.
                    "mass_finest_pair_pct":
                        float(100.0 * abs(mass[-1] / mass[-2] - 1.0)),
                    "mass": {k: _richardson(mass, h) for k, h in hs.items()},
                    "exponents": {},
                }
                for p in exponents:
                    pk = "%g" % p
                    phi = [c[kn]["values_mpa"][pk] for c in cells]
                    rr[kn]["exponents"][pk] = {
                        "values_mpa": [float(v) for v in phi],
                        **{k: _richardson(phi, h) for k, h in hs.items()}}
            e["radii"][key] = rr
        out[lab] = e
    return out


# ---------------------------------------------------------------------------
# SECTION C — ITEM 2, MEASURED: WHAT A FLIPPING MEMBERSHIP COSTS
# ---------------------------------------------------------------------------

def _tube_counts_all_copies(genes, cfg, label, r_sup):
    """The indicator's membership on every rotational copy, as `n_sp` integers.

    THE BISECTION VARIABLE, AND THE CHECK THAT IT IS COPY-INDEPENDENT IN ONE CALL.
    `build_wheel` replicates one sector, so the twelve copies are the same geometry and a
    boundary crossing has to happen on all of them at once.  That is what lets section C
    bracket a crossing on cheap MESH BUILDS and then measure the p-norm on the LOADED copy
    — two different copies of one event.  Asserted rather than assumed because if it were
    ever false the crossing would be located for a region the sweep does not read.
    """
    return _tube_count(genes, cfg, label, r_sup,
                       copy=range(WW.NUMBER_OF_SPOKES))


def _tube_count(genes, cfg, label, r_sup, copy=0):
    """Gauss points strictly inside the tube on one copy — or on several, if `copy` is a
    range, which is how the copy-independence check above pays for one mesh build."""
    mesh = WW.build_wheel(genes, cfg, fillet=True)
    xy, _ = gauss_geometry(mesh, cfg)
    ca, sa = _rotations(WW.NUMBER_OF_SPOKES)
    arc = CS.fillet_arcs(genes, cfg, True)[label]
    ks = [copy] if isinstance(copy, int) else list(copy)
    out = []
    for k in ks:
        rot = np.column_stack([ca[k] * xy[:, 0] - sa[k] * xy[:, 1],
                               sa[k] * xy[:, 0] + ca[k] * xy[:, 1]])
        out.append(int((CS._distance_to_arc(rot, arc) < r_sup).sum()))
    return out[0] if isinstance(copy, int) else out


def find_crossing(genes0, gene_index, cfg, label, r_sup, copy=0,
                  half_width=SMOOTH_HALF_WIDTH_MM, step=SMOOTH_SCAN_STEP_MM,
                  tol=SMOOTH_BISECT_TOL_MM):
    """Scan for a gene value at which a Gauss point crosses the tube boundary, then bisect.

    THE SCAN HAS TO BE WIDE AND THAT IS ITSELF THE FIRST RESULT.  The first attempt swept
    +-0.04 mm about the shipped `R_hub` and found the indicator perfectly smooth — because
    the fillet block is regenerated with the arc, so its Gauss points travel WITH the
    boundary and crossings are rare rather than dense.  Sampling blindly and reporting
    "no step found" would have been a false negative on a defect that is certainly there.
    §94's own check says it: "at a genome with an element near the boundary."

    Returns None when the bracket holds no crossing, which is a real outcome and not an
    error — it means this gene and this radius do not exercise the defect, and the caller
    reports the distance searched rather than a verdict.
    """
    n = int(round(half_width / step))
    xs = genes0[gene_index] + step * np.arange(-n, n + 1)
    lo_b, hi_b, _ = WG.bounds_arrays(WF.GENE_SPACE)
    xs = xs[(xs >= lo_b[gene_index]) & (xs <= hi_b[gene_index])]
    counts = []
    for x in xs:
        g = genes0.copy()
        g[gene_index] = x
        counts.append(_tube_count(g, cfg, label, r_sup, copy))
    counts = np.array(counts)
    jumps = np.nonzero(np.diff(counts))[0]
    if not len(jumps):
        return {"found": False, "scanned_half_width_mm": half_width,
                "scanned_range_mm": [float(xs[0]), float(xs[-1])],
                "n_gauss_in_tube": int(counts[0])}
    # The crossing NEAREST the genome, because the question the number answers is how far
    # a Stage 3 run starting here would have to walk to meet one.
    i = int(jumps[np.argmin(np.abs(xs[jumps] - genes0[gene_index]))])
    lo, hi = float(xs[i]), float(xs[i + 1])
    c_lo = int(counts[i])
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        g = genes0.copy()
        g[gene_index] = mid
        if _tube_count(g, cfg, label, r_sup, copy) == c_lo:
            lo = mid
        else:
            hi = mid
    return {"found": True, "scanned_half_width_mm": half_width, "copy": copy,
            "scanned_range_mm": [float(xs[0]), float(xs[-1])],
            "crossing_value_mm": 0.5 * (lo + hi),
            "bracket_mm": [lo, hi],
            "distance_from_genome_mm": abs(0.5 * (lo + hi) - genes0[gene_index]),
            "n_gauss_in_tube": [c_lo, int(counts[i + 1])],
            "n_crossings_in_scan": int(len(jumps))}


def _sample_measure(genes, cfg, label, r_sup, exponents, copy):
    """The mass and the p-norm at EVERY exponent, both kernels, on a NAMED copy.

    The copy is passed in rather than re-picked per sample, for `region_pnorm`'s reason
    one level up: a max over copies is itself an argmax, and a sweep whose argmax moved
    mid-window would show a step that had nothing to do with the region weight.

    EVERY EXPONENT, BECAUSE THE STEP IS NOT THE SAME SIZE AT ALL OF THEM AND THE FIRST
    DRAFT QUOTED IT AT ONE.  A crossing admits a point at the tube's EDGE, where the
    stress is lowest, so a high `p` — which weights the peak — feels it less than a low
    one.  Reporting the step only at `STRESS_NOMINAL_P` = 4.0 would price the defect at
    an exponent nobody proposes to use; reporting it only at the recommended `p` would
    hide how it behaves under the exponent the objective has today.  Both, and the
    solves are shared, so the whole ladder is free.
    """
    mesh, res, xy, vm = CS.solve_field(genes, cfg, fillet=True)
    _, vol = gauss_geometry(mesh, cfg)
    arc = CS.fillet_arcs(genes, cfg, True)[label]
    ca, sa = _rotations(WW.NUMBER_OF_SPOKES)
    rot = np.column_stack([ca[copy] * xy[:, 0] - sa[copy] * xy[:, 1],
                           sa[copy] * xy[:, 0] + ca[copy] * xy[:, 1]])
    d = CS._distance_to_arc(rot, arc)
    out = {}
    for kn in KERNELS:
        w = region_weight(d * d, r_sup, kn) * vol
        tot = float(w.sum())
        out["%s_mass" % kn] = tot
        for p in exponents:
            out["%s_p%g" % (kn, p)] = float((np.sum(w * vm ** p) / tot) ** (1.0 / p))
    return out


def _step_stats(values):
    """`max|first difference|` against its own median — the discontinuity statistic.

    WHY THIS AND NOT A SECOND DIFFERENCE.  The first attempt used `max|d2| / delta^2`,
    which is the textbook curvature estimator and is the wrong instrument here: at the
    step sizes a gene sweep can afford it is dominated by round-off amplified through the
    mesh build, and it grew at the same rate for the region measures AND for the total
    mesh volume, which has no region weight in it at all.  A jump of size `J` puts `J`
    into ONE first difference and leaves its neighbours on the smooth trend, so the ratio
    of the largest to the median is a clean signal that needs no noise model — and, since
    a smooth function's differences all shrink with `delta` while a jump does not, the
    ratio GROWS by the step ratio when the sampling is refined.  That is the test.
    """
    v = np.asarray(values, dtype=float)
    d1 = np.abs(np.diff(v))
    med = float(np.median(d1))
    return {"mean": float(v.mean()), "max_abs_step": float(d1.max()),
            "median_abs_step": med,
            "max_over_median": float(d1.max() / med) if med > 0.0 else None,
            "max_step_pct_of_mean": float(100.0 * d1.max() / abs(v.mean()))}


def smoothness(genome_path, gene=SMOOTH_GENE, cfg_name=SMOOTH_CONFIG,
               label="hub", r_sup=REFERENCE_RADIUS_MM, p=SMOOTH_P,
               exponents=EXPONENTS, steps=SMOOTH_STEPS_MM, samples=SMOOTH_SAMPLES):
    """§94 item 2, measured: sample ACROSS a located crossing at two step sizes.

    The window is centred on the crossing and offset by half a step, so the crossing falls
    strictly BETWEEN two samples at every step size rather than on one of them.  Without
    that the finer window walks past the crossing and reports everything smooth — which is
    what the pilot did, and it is a false negative that looks exactly like a pass.
    """
    genes0 = CS.load_genes(genome_path)
    idx = WG.GENE_NAMES.index(gene)
    cfg = WW.get_config(cfg_name)

    # THE LOADED COPY, RESOLVED FIRST, AND EVERYTHING BELOW READS IT.  The first version
    # of this section bracketed and sampled on copy 0, which is a spoke a quarter turn
    # from the contact patch: its p-norm came back at 1.50 MPa against the loaded copy's
    # 18.31, so the step was being quoted as a fraction of a number the loss never sees.
    # The membership crossing is copy-INDEPENDENT — `build_wheel` replicates one sector —
    # so locating it on cheap mesh builds and reading it on the loaded copy is one event
    # seen twice, and the check below is what says so rather than assuming it.
    mesh, _res, xy, vm = CS.solve_field(genes0, cfg, fillet=True)
    _, vol = gauss_geometry(mesh, cfg)
    arc = CS.fillet_arcs(genes0, cfg, True)[label]
    copy = region_pnorm(arc, xy, vm, vol, WW.NUMBER_OF_SPOKES, (p,), r_sup,
                        "bump3")["loaded_copy"]
    per_copy = _tube_counts_all_copies(genes0, cfg, label, r_sup)

    lo, hi, _ = WG.bounds_arrays(WF.GENE_SPACE)
    out = {"gene": gene, "gene_index": idx, "config": cfg_name, "junction": label,
           "region_radius_mm": r_sup, "exponent": p,
           "exponents": [float(v) for v in exponents], "loaded_copy": copy,
           "tube_count_per_copy": per_copy,
           "tube_count_is_copy_independent": len(set(per_copy)) == 1,
           "gene_bounds_mm": [float(lo[idx]), float(hi[idx])],
           "genome_value_mm": float(genes0[idx]), "crossing": None, "windows": {}}
    print("    scanning %s for a tube-boundary crossing at r = %.2f mm, copy %d ..."
          % (gene, r_sup, copy), flush=True)
    cross = find_crossing(genes0, idx, cfg, label, r_sup, copy)
    out["crossing"] = cross
    if not cross["found"]:
        return out
    for step in steps:
        xs = cross["crossing_value_mm"] + step * (
            np.arange(samples) - samples // 2 + 0.5)
        rows = []
        for x in xs:
            g = genes0.copy()
            g[idx] = x
            rows.append(_sample_measure(g, cfg, label, r_sup, exponents, copy))
        w = {"step_mm": step,
             "gene_values_mm": [float(x) for x in xs],
             "series": {k: [float(r[k]) for r in rows] for k in rows[0]},
             "stats": {k: _step_stats([r[k] for r in rows]) for k in rows[0]}}
        out["windows"]["%g" % step] = w
        print("      step %-9g  p=%g: indicator max/median %8.1f    bump3 %6.1f"
              % (step, p, w["stats"]["indicator_p%g" % p]["max_over_median"],
                 w["stats"]["bump3_p%g" % p]["max_over_median"]), flush=True)
    return out


# ---------------------------------------------------------------------------
# THE VERDICT
# ---------------------------------------------------------------------------

# The order a converged quantity has to show.  M8b-i.6 step 1 accepted 1.76 / 2.18 and
# rejected 0.99 / 1.12 without writing a number down, so there is no inherited threshold
# to use; 1.5 is the midpoint of that gap.  IT IS A GATE INVENTED FROM ONE PRECEDENT, so
# the verdict is reported at three floors rather than one and the tables carry every
# order, which is the same discipline `study_deflection_gci` applies to `h`.
ORDER_FLOORS = (1.25, 1.5, 1.75)

# What the region's own quadrature may drift over the finest pair before nothing built on
# it can be quoted.  0.5% is `study_deflection_gci`'s GATE_PCT (0.3%) loosened by the fact
# that this is an error bound on a denominator rather than the answer itself; both the
# measured value and this threshold are in the table, so a reader who prefers 0.3% can
# apply it.
MASS_DRIFT_PCT = 0.5


def verdict(conv, radii=RADII_MM, exponents=EXPONENTS):
    """The largest `p` that survives, per cell — and the cells where no `p` can.

    TWO GATES AND THEY ARE NOT THE SAME QUESTION.  The mass gate asks whether the mesh
    resolves the REGION; the order gate asks whether the FIELD over it converges.  A cell
    that fails the first has nothing to say about the second, and reporting an order there
    is what produced this file's stray 7s and 9s: three points a hair apart give Roache an
    error ratio near zero and he answers with a large `p`, which reads like fast
    convergence and is actually no signal at all.
    """
    out = {"order_floors": list(ORDER_FLOORS), "mass_drift_gate_pct": MASS_DRIFT_PCT,
           "cells": {}, "clean_cells": [], "largest_p_overall": {}}
    for gname, per_j in conv.items():
        for lab, e in per_j.items():
            for r in radii:
                key = "%g" % r
                blk = e["radii"][key]["bump3"]
                mass_ok = blk["mass_finest_pair_pct"] <= MASS_DRIFT_PCT
                per_floor = {}
                for floor in ORDER_FLOORS:
                    good = []
                    for p in exponents:
                        o = blk["exponents"]["%g" % p]
                        a = o["h_fillet"]["observed_order_p"]
                        b = o["h_global"]["observed_order_p"]
                        # BOTH `h` DEFINITIONS OR NEITHER: the module docstring's rule.
                        if a is not None and b is not None and min(a, b) >= floor:
                            good.append(p)
                    per_floor["%g" % floor] = {
                        "exponents_passing": good,
                        "largest_p": max(good) if good else None}
                cell = {"genome": gname, "junction": lab, "radius_mm": r,
                        "mass_finest_pair_pct": blk["mass_finest_pair_pct"],
                        "region_resolved": bool(mass_ok),
                        "cells_across_fine": e["radii"][key]["cells_across_fine"],
                        "by_order_floor": per_floor}
                out["cells"]["%s/%s/%s" % (gname, lab, key)] = cell
                if mass_ok and per_floor["%g" % 1.5]["largest_p"] is not None:
                    out["clean_cells"].append("%s/%s/%s" % (gname, lab, key))
    # §94's CHECK, answered per junction: the largest `p` that survives BOTH gates
    # anywhere on the grid, and at which radius it did.
    for gname, per_j in conv.items():
        for lab in per_j:
            best = None
            for k, c in out["cells"].items():
                if c["genome"] != gname or c["junction"] != lab:
                    continue
                if not c["region_resolved"]:
                    continue
                lp = c["by_order_floor"]["1.5"]["largest_p"]
                if lp is not None and (best is None or lp > best[0]):
                    best = (lp, c["radius_mm"])
            out["largest_p_overall"]["%s/%s" % (gname, lab)] = (
                None if best is None else {"largest_p": best[0], "radius_mm": best[1]})

    # AND THE CROSS-CELL ANSWER, WHICH IS THE ONE THE WIRING STEP NEEDS.  `p` and `r` are
    # constants of the objective, not per-design choices: whatever is picked has to hold
    # at every junction of every genome the optimizer can reach, and this arc has two
    # genomes to check that on.  So the recommendation is the SMALLEST radius resolved in
    # every cell — smallest because the term exists to be LOCAL to the fillet surface —
    # paired with the largest `p` that clears the floor in every cell at that radius.
    per_radius = {}
    for r in radii:
        key = "%g" % r
        cells = [c for c in out["cells"].values() if c["radius_mm"] == r]
        entry = {"resolved_everywhere": all(c["region_resolved"] for c in cells),
                 "worst_mass_drift_pct": max(c["mass_finest_pair_pct"] for c in cells),
                 "largest_p_all_cells": {}}
        for floor in ORDER_FLOORS:
            fk = "%g" % floor
            common = set(exponents)
            for c in cells:
                common &= set(c["by_order_floor"][fk]["exponents_passing"])
            entry["largest_p_all_cells"][fk] = max(common) if common else None
        per_radius[key] = entry
    out["per_radius"] = per_radius
    ok = [r for r in radii
          if per_radius["%g" % r]["resolved_everywhere"]
          and per_radius["%g" % r]["largest_p_all_cells"]["1.5"] is not None]
    out["radii_resolved_everywhere"] = [float(r) for r in ok]
    out["recommended"] = (
        None if not ok else
        {"radius_mm": float(min(ok)),
         "exponent": per_radius["%g" % min(ok)]["largest_p_all_cells"]["1.5"]})
    return out


def price_recommendation(rep):
    """The recommended cell read as a UTILISATION, and what the raw substitution fires.

    `util_j = sigma_fillet_j / ALLOWABLE_STRESS_MPA` is the replacement §94 wrote down, so
    the division is not an interpretation and the barrier values are the objective's own
    `soft_barrier` at the objective's own `DEFAULT_WEIGHTS` — nothing here is fitted or
    chosen.  IT IS NOT THE EXCHANGE RATE, which is §94 item 4 and stays out of scope: what
    these numbers say is what the substitution costs BEFORE anyone re-derives a weight,
    which is exactly the quantity item 4 needs as its input.
    """
    rec = rep["verdict"]["recommended"]
    if rec is None:
        return None
    rk, pk = "%g" % rec["radius_mm"], "%g" % rec["exponent"]
    out = {"radius_mm": rec["radius_mm"], "exponent": rec["exponent"],
           "allowable_stress_mpa": WO.ALLOWABLE_STRESS_MPA,
           "margin_knee_util": WO.MARGIN_KNEE_UTIL,
           "weights": {k: WO.DEFAULT_WEIGHTS[k] for k in ("stress", "stress_margin")},
           "cells": {}}
    for gname, per_j in rep["convergence"].items():
        for lab, e in per_j.items():
            blk = e["radii"][rk]["bump3"]["exponents"][pk]
            v = blk["values_mpa"][-1]
            u = v / WO.ALLOWABLE_STRESS_MPA
            ap = e["arc_peak"]["h_fillet"]["phi"][-1]
            out["cells"]["%s/%s" % (gname, lab)] = {
                "sigma_fillet_mpa": v, "util": u,
                "arc_peak_mpa": ap, "arc_peak_util": ap / WO.ALLOWABLE_STRESS_MPA,
                "fraction_of_arc_peak": v / ap,
                "observed_order_p": blk["h_fillet"]["observed_order_p"],
                "gci_fine_pct": blk["h_fillet"].get("gci_fine_pct"),
                "stress": float(WO.soft_barrier(
                    u - 1.0, WO.DEFAULT_WEIGHTS["stress"])),
                "stress_margin": float(WO.soft_barrier(
                    u - WO.MARGIN_KNEE_UTIL, WO.DEFAULT_WEIGHTS["stress_margin"]))}
    return out


# ---------------------------------------------------------------------------
# BUILD AND REPORT
# ---------------------------------------------------------------------------

def build(genomes=GENOMES, ladder=LADDER, radii=RADII_MM, exponents=EXPONENTS,
          smooth=True):
    rep = {"genomes": dict(genomes), "ladder": list(ladder),
           "extrapolate_from": list(EXTRAPOLATE_FROM),
           "radii_mm": list(radii), "reference_radius_mm": REFERENCE_RADIUS_MM,
           "exponents": list(exponents), "kernels": list(KERNELS),
           "kinematics": "linear", "n_phase": 1,
           "kernel_formula": "W = max(0, 1 - d^2/r^2)^3",
           "rungs": {}, "convergence": {}, "corner_clearances_mm": {},
           "priced": None, "smoothness": None}
    for gname, path in genomes:
        print("  %s  (%s)" % (gname, path), flush=True)
        genes, rungs = run_ladder(path, ladder, radii, exponents)
        rep["rungs"][gname] = rungs
        rep["convergence"][gname] = converge(rungs, EXTRAPOLATE_FROM, radii, exponents)
        rep["corner_clearances_mm"][gname] = corner_clearances_mm(
            genes, WW.get_config(ladder[-1]))
    rep["verdict"] = verdict(rep["convergence"], radii, exponents)
    rep["priced"] = price_recommendation(rep)
    if smooth:
        print("  SECTION C — the smooth region weight", flush=True)
        # AT §52's RADIUS AND AT THE RECOMMENDED ONE, BECAUSE THEY ARE NOT THE SAME
        # NUMBER AND THE FIRST DRAFT ONLY HAD THE FIRST.  0.30 mm is the region
        # `arc_peak` reads and is where the defect is largest — one crossing is a
        # bigger share of a smaller tube.  The recommended radius is the one a wired
        # term would actually use, and reporting only the worse of the two would
        # overstate what the proposed configuration suffers.
        rec = rep["verdict"]["recommended"]
        rr = [REFERENCE_RADIUS_MM]
        if rec is not None and rec["radius_mm"] not in rr:
            rr.append(rec["radius_mm"])
        rep["smoothness"] = {}
        for r in rr:
            print("    r_sup = %.2f mm" % r, flush=True)
            rep["smoothness"]["%g" % r] = smoothness(genomes[0][1], r_sup=r,
                                                     exponents=exponents)
    return rep


def _bar(title):
    print("\n" + "=" * 78 + "\n  " + title + "\n" + "=" * 78)


def _print(rep):
    _bar("A  THE REGION'S OWN MASS — NO FIELD, PURE QUADRATURE ERROR")
    print("    `SUM W_g V_g` over a region fixed ANALYTICALLY.  Whatever this fails to")
    print("    converge at, nothing built on it converges better.\n")
    print("    Read on copy 0, so no field enters it even through a copy choice; the")
    print("    loaded copy's own denominator agrees to `mass_copy_spread_rel`.\n")
    print("    genome   junc  r_sup  cells across a/b      bump3 mass C/M/F"
          "               d(F,M)  indicator d(F,M)  copy spread")
    for gname, per_j in rep["convergence"].items():
        for lab, e in per_j.items():
            for r in rep["radii_mm"]:
                blk = e["radii"]["%g" % r]
                ca = blk["cells_across_fine"]
                b, i = blk["bump3"], blk["indicator"]
                print("    %-8s %-5s %.2f   %4.1f / %4.1f   %9.6f %9.6f %9.6f  "
                      "%7.3f%%          %7.3f%%     %.1e"
                      % (gname, lab, r, ca["a"], ca["b"], *b["region_mass_mm3"],
                         b["mass_finest_pair_pct"], i["mass_finest_pair_pct"],
                         b["mass_copy_spread_rel"]))

    _bar("B  THE EXPONENT SWEEP — ORDER (h_fillet / h_global) AND GCI, bump3")
    for gname, per_j in rep["convergence"].items():
        for lab, e in per_j.items():
            ap = e["arc_peak"]["h_fillet"]
            print("\n    %s / %s      arc_peak %s MPa   order %s (fillet) / %s (global)"
                  % (gname, lab,
                     " ".join("%.3f" % v for v in ap["phi"]),
                     "%.2f" % ap["observed_order_p"] if ap["observed_order_p"] else "none",
                     "%.2f" % e["arc_peak"]["h_global"]["observed_order_p"]
                     if e["arc_peak"]["h_global"]["observed_order_p"] else "none"))
            print("      r_sup  resolved" + "".join("   p=%-4g" % p
                                                    for p in rep["exponents"]))
            for r in rep["radii_mm"]:
                blk = e["radii"]["%g" % r]["bump3"]
                ok = "  yes " if blk["mass_finest_pair_pct"] <= MASS_DRIFT_PCT else "  NO  "
                line_v = "      %.2f  %s " % (r, ok)
                line_o = "                   "
                line_g = "                   "
                for p in rep["exponents"]:
                    o = blk["exponents"]["%g" % p]
                    a = o["h_fillet"]["observed_order_p"]
                    bb = o["h_global"]["observed_order_p"]
                    g = o["h_fillet"].get("gci_fine_pct")
                    line_v += " %7.3f" % o["values_mpa"][-1]
                    line_o += " %3s/%3s" % ("%.1f" % a if a else "  -",
                                            "%.1f" % bb if bb else "  -")
                    line_g += " %6.2f%%" % g if g is not None else "      --"
                print(line_v + "   MPa at fine")
                print(line_o + "   order fillet/global")
                print(line_g + "   GCI")

    _bar("C  THE SMOOTH REGION WEIGHT — WHAT A FLIPPING MEMBERSHIP COSTS")
    if not rep["smoothness"]:
        print("    not run")
    for rk in sorted(rep["smoothness"] or {}, key=float):
        s = rep["smoothness"][rk]
        print("\n    ---- r_sup = %s mm%s ----" % (
            rk, "   (§52's `arc_peak` tube)" if float(rk) == REFERENCE_RADIUS_MM
            else "   (RECOMMENDED)"))
        c = s["crossing"]
        if not c["found"]:
            print("    no crossing within +-%.3f mm of %s = %.5f"
                  % (c["scanned_half_width_mm"], s["gene"], s["genome_value_mm"]))
            continue
        print("    %s = %.6f, box [%.1f, %.1f].  Nearest tube-boundary crossing at"
              % (s["gene"], s["genome_value_mm"], *s["gene_bounds_mm"]))
        print("    %s = %.7f — %.4f mm away, %.2f%% of the box; %d crossings in +-%.2f mm."
              % (s["gene"], c["crossing_value_mm"], c["distance_from_genome_mm"],
                 100.0 * c["distance_from_genome_mm"]
                 / (s["gene_bounds_mm"][1] - s["gene_bounds_mm"][0]),
                 c["n_crossings_in_scan"], c["scanned_half_width_mm"]))
        print("    Gauss points in the tube: %d -> %d, on loaded copy %d "
              "(%d per copy, all %d equal: %s).  %s junction.\n"
              % (*c["n_gauss_in_tube"], s["loaded_copy"], s["tube_count_per_copy"][0],
                 len(s["tube_count_per_copy"]), s["tube_count_is_copy_independent"],
                 s["junction"]))

        # THE MASS, THE OBJECTIVE'S CURRENT NOMINAL EXPONENT, AND THE RECOMMENDED ONE.
        # A crossing admits a point at the tube's EDGE where the stress is lowest, so a
        # high `p` feels it less — quoting one exponent would price the defect either at
        # a `p` nobody proposes or at a `p` the objective does not have.
        rec_p = (rep["verdict"]["recommended"] or {}).get("exponent")
        quantities = ["indicator_mass", "bump3_mass",
                      "indicator_p%g" % s["exponent"], "bump3_p%g" % s["exponent"]]
        if rec_p is not None and rec_p != s["exponent"]:
            quantities += ["indicator_p%g" % rec_p, "bump3_p%g" % rec_p]
        print("      step_mm    quantity            mean      max|step|   median|step|"
              "    ratio   step % of mean")
        for skey in sorted(s["windows"], key=float, reverse=True):
            w = s["windows"][skey]
            for q in quantities:
                st = w["stats"][q]
                print("      %-9g  %-18s %9.5f  %10.3e  %10.3e  %7.1f   %8.4f%%"
                      % (w["step_mm"], q, st["mean"], st["max_abs_step"],
                         st["median_abs_step"], st["max_over_median"],
                         st["max_step_pct_of_mean"]))
            print()
        fine = min(s["windows"], key=float)
        for kn in ("indicator", "bump3"):
            print("      %-9s step %% of mean at step %s, by p:  %s"
                  % (kn, fine,
                     "  ".join("%g:%.4f%%"
                               % (e, s["windows"][fine]["stats"]["%s_p%g" % (kn, e)]
                                  ["max_step_pct_of_mean"]) for e in s["exponents"])))
        print("      The step SHRINKS with `p`: the crossing admits a point at the tube's")
        print("      EDGE, where the stress is lowest, so a peak-weighted exponent feels")
        print("      it least.  It is a step at every one of them.")
    _bar("THE VERDICT — §94's CHECK: THE LARGEST p WHOSE ORDER IS STILL ~2")
    v = rep["verdict"]
    print("    Two gates.  The MASS gate asks whether the mesh resolves the region; the")
    print("    ORDER gate asks whether the field over it converges.  A cell failing the")
    print("    first has nothing to say about the second.\n")
    print("    cell                     resolved   largest p at order floor "
          + " / ".join("%.2f" % f for f in v["order_floors"]))
    for k in sorted(v["cells"]):
        c = v["cells"][k]
        lp = [c["by_order_floor"]["%g" % f]["largest_p"] for f in v["order_floors"]]
        print("    %-24s %-10s %s   (mass drift %.3f%%)"
              % (k, "yes" if c["region_resolved"] else "NO",
                 " / ".join("%5s" % ("%g" % x if x is not None else "none") for x in lp),
                 c["mass_finest_pair_pct"]))
    print("\n    §94's CHECK, per junction — the largest p surviving BOTH gates anywhere:")
    for k in sorted(v["largest_p_overall"]):
        b = v["largest_p_overall"][k]
        print("      %-20s %s" % (k, "NO EXPONENT QUALIFIES" if b is None
                                  else "p = %g, at r_sup = %.2f mm"
                                       % (b["largest_p"], b["radius_mm"])))
    print("\n    Clean cells (region resolved AND some p at order >= 1.50): %s"
          % (", ".join(sorted(v["clean_cells"])) or "none"))

    print("\n    AND THE CROSS-CELL ANSWER — `p` and `r` are constants of the objective,")
    print("    so what is picked has to hold at every junction of every genome:")
    print("      r_sup   resolved everywhere   worst mass drift   largest p at "
          + " / ".join("%.2f" % f for f in v["order_floors"]))
    for r in rep["radii_mm"]:
        e = v["per_radius"]["%g" % r]
        print("      %.2f    %-19s %7.3f%%            %s"
              % (r, "yes" if e["resolved_everywhere"] else "NO",
                 e["worst_mass_drift_pct"],
                 " / ".join("%5s" % ("%g" % e["largest_p_all_cells"]["%g" % f]
                                     if e["largest_p_all_cells"]["%g" % f] is not None
                                     else "none") for f in v["order_floors"])))
    pr = rep.get("priced")
    if pr is None:
        print("\n    NO (r, p) IS RESOLVED AT EVERY CELL — nothing to recommend.")
    else:
        print("\n    RECOMMENDED: r_sup = %.2f mm, p = %g  (smallest radius resolved"
              % (pr["radius_mm"], pr["exponent"]))
        print("    everywhere, largest p clearing order 1.50 in every cell), priced at")
        print("    ALLOWABLE = %.1f MPa, weights stress %g / stress_margin %g:\n"
              % (pr["allowable_stress_mpa"], pr["weights"]["stress"],
                 pr["weights"]["stress_margin"]))
        print("      cell             sigma_fillet   util    arc_peak  its util  frac  "
              " order   GCI      stress  stress_margin")
        for k in sorted(pr["cells"]):
            c = pr["cells"][k]
            print("      %-16s %8.4f %8.4f  %8.4f %8.4f  %.3f  %5s %6s  %9.4f %9.4f"
                  % (k, c["sigma_fillet_mpa"], c["util"], c["arc_peak_mpa"],
                     c["arc_peak_util"], c["fraction_of_arc_peak"],
                     "%.2f" % c["observed_order_p"] if c["observed_order_p"] else "none",
                     "%.2f%%" % c["gci_fine_pct"] if c["gci_fine_pct"] is not None
                     else "--",
                     c["stress"], c["stress_margin"]))
        print("\n    The two barriers above are what the RAW substitution fires at today's")
        print("    weights.  That is §94 item 4's input and not its answer — the exchange")
        print("    rate was derived at util 0.855 under a construction this replaces.")

    _bar("SCOPE")
    print("    LINEAR kinematics, ONE phase — `study_corner_singularity`'s ladder, which")
    print("    is the one `arc_peak`'s numbers were taken on.  Both gaps raise the")
    print("    measured side (§94: filleted max_stress 34.68 linear against 38.37 SVK).")
    print("    Nothing here is wired into `wheel_objective` and nothing is promoted.")
    print("\n    Arc-to-corner clearances at the finest rung, for §94 item 3:")
    for gname, per_j in rep["corner_clearances_mm"].items():
        for lab in JUNCTIONS:
            d = per_j[lab]
            print("      %-8s %s arc -> %s:P_c %7.4f mm, %s:P_t %7.4f mm  "
                  "(tube radius %.2f)"
                  % (gname, lab, lab, d["%s:P_c" % lab], lab, d["%s:P_t" % lab],
                     REFERENCE_RADIUS_MM))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ladder", default=",".join(LADDER))
    ap.add_argument("--radii", default=",".join("%g" % r for r in RADII_MM))
    ap.add_argument("--exponents", default=",".join("%g" % p for p in EXPONENTS))
    ap.add_argument("--no-smoothness", action="store_true",
                    help="skip section C (item 2); the ladder alone is ~2 min")
    ap.add_argument("--out", default="study_fillet_pnorm.json")
    args = ap.parse_args()

    ladder = tuple(s.strip() for s in args.ladder.split(",") if s.strip())
    radii = tuple(float(s) for s in args.radii.split(",") if s.strip())
    exps = tuple(float(s) for s in args.exponents.split(",") if s.strip())

    # What degrades THIS driver: a shorter ladder (Richardson needs `EXTRAPOLATE_FROM`
    # entire), a thinner sweep on either axis, or section C skipped — which would file an
    # artifact whose §94-item-2 half is simply absent while every other field reads normal.
    _gate_guard.refuse_degraded_out(ap, args, "study_fillet_pnorm.json", [
        (tuple(ladder) != LADDER, "--ladder %s, not the committed %s"
                                  % (args.ladder, ",".join(LADDER))),
        (set(radii) < set(RADII_MM), "--radii %s drops one of the committed %s"
                                     % (args.radii, ",".join("%g" % r for r in RADII_MM))),
        (set(exps) < set(EXPONENTS), "--exponents %s drops one of the committed %s"
                                     % (args.exponents,
                                        ",".join("%g" % p for p in EXPONENTS))),
        (args.no_smoothness, "--no-smoothness leaves §94 item 2 unmeasured"),
    ])

    t0 = time.time()
    rep = build(GENOMES, ladder, radii, exps, smooth=not args.no_smoothness)
    rep["seconds"] = time.time() - t0
    _print(rep)
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(rep, fh, indent=2)
    print("\n    wrote %s  (%.1f s)" % (args.out, rep["seconds"]))

    # NO PASS/FAIL AND EXIT 0, for `study_fillet_kt`'s reason: §94 asked what `p` the
    # region measure supports, and "none, and here is the quantity that is actually
    # limiting" is an answer rather than a failure.  The verdict block carries it.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
