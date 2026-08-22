"""
=============================================================================
  AT WHAT RADIUS DOES THE FILLETED SPOKE BLOCK FOLD?  THREE CRITERIA, AND
  WHY TWO OF THEM DISAGREED BY 20x
=============================================================================
    .venv-opt/bin/python studies/study_fillet_fold.py            (make fillet)

FILLET_PLAN.md STEP 1 RECORD PART 5's open discrepancy, and PLAN.md §43's ranked item 1.

WHY THIS EXISTS
---------------
`wheel_wheel.sector_blocks(..., fillet=)` rounds the `P_t` corner of each junction.  It is
opt-in, inert by default, and PART 3 called it "what a fillet-block implementation will be
checked against" -- so it is the arc's measuring instrument, and the number it exists to
produce is THE LARGEST RADIUS IT SURVIVES.  That number was recorded twice and disagreed:

    PART 3  (2026-08-17)   0.20 mm at `coarse`,  0.10 at `medium`,  both junctions
    PART 5  (2026-08-21)   4.00 / 3.00 coarse,   0.40 / 0.40 medium  (hub / rim)

PART 5 filed the gap open rather than resolving it, because PART 3's criterion was never
written down beyond the word "survives" and NO SCRIPT OF EITHER SURVIVED.  Both were
scratch measurements written up in prose.  This file is the apparatus, so the next person
re-runs it instead of re-deriving it.

THE ANSWER: THEY MEASURED DIFFERENT THINGS, AND BOTH NUMBERS ARE RIGHT
----------------------------------------------------------------------
Three criteria are computed here for every radius, on the same mesh, in one pass:

  A.  `block_cells` -- PART 3's.  The spoke block's BILINEAR CELLS: a cell is bad when
      its four corner cross products do not all share a sign.  PART 3's own write-up
      reports exactly this column ("mixed-sign cells") for the seven blocks, which is
      what identifies it.

  B.  `build_wheel` -- PART 5's.  Does `wheel_wheel.build_wheel` raise.  Its guard is
      `_orient_elements`, a shoelace over the FOUR CORNERS of each element.

  C.  `gauss` -- neither's.  Is `det J` positive at all 3x3 Gauss points of every Q9
      element, i.e. AT THE POINTS THE FE ASSEMBLY ACTUALLY INTEGRATES.  Reported twice:
      on the SPOKE BLOCK alone, which is the like-for-like comparison with A and is
      computable even when `build_wheel` refuses, and on the WHOLE ASSEMBLED MESH
      (`mesh_gauss`) when it builds.  The two differ at large radii, where the fold
      leaves the spoke and appears in the blocks the moved corner drags with it, so
      `mesh_gauss` -- builds AND integrates -- is the one to read as "usable".

B is the weakest of the three and the reason is structural rather than a matter of
tolerance.  Every config here is `order=2`, so ONE Q9 ELEMENT SPANS 2x2 CELLS, and its
shoelace uses only its four corner nodes -- the five mid nodes, and therefore the whole
interior, are invisible to it.  A cell can invert inside an element whose corner
shoelace is comfortably positive.  That is not hypothetical: measured below, `build_wheel`
accepts meshes with dozens of non-positive Gauss points.

So PART 5's row is not a more permissive measurement of the same thing.  It is the
radius at which a guard that cannot see the fold finally notices, and it overstates what
is usable by 10-20x.  PART 3's row is within one grid step of criterion C's answer.
**PART 3's table is the one to quote.**

WHAT CRITERION C SAYS, WHICH IS SHARPER THAN EITHER RECORDED ROW
----------------------------------------------------------------
There is no interval `0 < R < R_max` that this construction meshes validly.  Small radii
fold too.  What exists is a NARROW WINDOW, and both of its edges are properties of node
allocation rather than of geometry:

  * BELOW the window the fillet arc is forced onto one cell by `_filleted_spoke`'s
    `k0 = clip(round((s_A - s0) / ds), 1, cap)`, whose lower clamp of 1 puts the arc's far
    node at a fraction `mid_frac` of the first Q9 element's edge.  As R -> 0 that fraction
    -> 0, and a quadratic edge whose mid node sits at 6% of its length is singular.  The
    window opens where `mid_frac` climbs back to ~0.4.
  * ABOVE the window `round(...)` steps from 1 to 2, the arc claims a second cell, and the
    element straddling its end inverts.

Both edges move under refinement because `ds` does, which is the mechanism behind the
qualitative claim PART 3 and PART 5 agreed on: THE LIMIT TIGHTENS UNDER REFINEMENT, so it
belongs to the construction and not to the notch.  Criterion C names which part of the
construction.

WHAT THIS DOES NOT DO
---------------------
It does not solve a field, does not touch `best_solution.json`, and does not change the
default mesh -- `fillet=None` is the default everywhere and the controls below re-confirm
it is bit-identical.  It is geometry and Jacobians, and it runs in under a minute.

EXIT STATUS follows `make junction`: nonzero ONLY if a self-check fails -- the two
controls, or the reproduction of the two contested tables.  Never on a characterisation
finding about the fillet, which is what this exists to report.
=============================================================================
"""

import argparse
import json
import os
import time

import numpy as np

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import _gate_guard
import wheel_fem as fem
import wheel_genome as wg
import wheel_mesh as WM
import wheel_wheel as WW

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIGS = ("coarse", "medium")
DEFAULT_JUNCTIONS = ("hub", "rim")

# Feasible genomes sampled for the default-path section.  1000 at `coarse` is ~20 s and
# resolves a 1-in-100 effect, which is the size of the thing being looked for.
DEFAULT_DRAWS = 1000

# The fine grid resolves the window edges to 0.01 mm; the tail carries the radii PART 5
# swept, so its table is reproduced from the same run rather than from a second sweep.
FINE_GRID = tuple(round(0.01 * k, 2) for k in range(1, 51))
TAIL_GRID = (0.60, 0.80, 1.00, 1.20, 1.60, 2.00, 3.00, 4.00)

# The grid the two contested tables were taken on.  PART 5 records cells at 0.40, 0.80,
# 1.20, 2.00 and 3.00 over "0.05 .. 4.00"; PART 3 quotes 0.20 and 0.10.  Both are
# reproduced against THIS grid, so "largest surviving" means the same thing it meant
# there -- the largest grid point with no fold at it or below it.
LEGACY_GRID = (0.05, 0.10, 0.20, 0.30, 0.40, 0.80, 1.20, 2.00, 3.00, 4.00)

# What the two rows claimed, so the reproduction is checked rather than eyeballed.
PART3_LARGEST_SURVIVING = {("coarse", "hub"): 0.20, ("coarse", "rim"): 0.20,
                           ("medium", "hub"): 0.10, ("medium", "rim"): 0.10}
PART5_FIRST_FOLD = {("coarse", "hub"): 4.00, ("coarse", "rim"): 3.00,
                    ("medium", "hub"): 0.40, ("medium", "rim"): 0.40}

CRITERIA = ("block_cells", "build_wheel", "gauss", "mesh_gauss")


def load_genes(path):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


# ---------------------------------------------------------------------------
# THE THREE CRITERIA
# ---------------------------------------------------------------------------

def cell_verdict(grid):
    """PART 3's criterion on an [ni, nj, 2] block: mixed-sign bilinear cells.

    Each cell's four corner cross products are computed the way
    `wheel_mesh.scaled_jacobian` computes an element's, but on the CELL rather than on
    the Q9 element, which is what makes this the strictest of the three at small radii.
    A cell is bad when the four do not share a sign; that is the "mixed-sign cells"
    column PART 3 tabulated for all seven blocks.
    """
    P = np.asarray(grid, float)
    c = np.stack([P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]], axis=2).reshape(-1, 4, 2)
    signs = []
    for k in range(4):
        a = c[:, (k + 1) % 4] - c[:, k]
        b = c[:, (k - 1) % 4] - c[:, k]
        signs.append(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    S = np.stack(signs, axis=1)
    mixed = int(((S > 0).any(axis=1) & (S < 0).any(axis=1)).sum())
    x, y = c[:, :, 0], c[:, :, 1]
    area = 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)
    area = area if np.median(area) > 0 else -area      # the block may be left-handed
    return {"mixed_sign_cells": mixed,
            "non_positive_area_cells": int((area <= 0).sum()),
            "min_abs_cell_area_mm2": float(np.abs(area).min())}


def gauss_verdict(grid):
    """Criterion C on one block: `det J` at the 3x3 Gauss points of its Q9 elements.

    Uses `wheel_fem.gauss_volume`, i.e. THE SAME KERNEL the assembly integrates with,
    rather than a re-derivation -- the point of this criterion is that it is not a
    proxy.  The sign is normalised by the median because a block indexed (theta, r) is
    left-handed in physical space and `build_wheel` flips it later; what is being asked
    here is whether any point disagrees with its own block, not which way round the
    block is numbered.
    """
    g = np.asarray(grid, float)
    ni, nj = g.shape[:2]
    conn = WM.grid_connectivity(ni, nj, 2)
    vol = np.asarray(fem.gauss_volume(g.reshape(-1, 2), conn, order=2, width=1.0))
    vol = vol if np.median(vol) > 0 else -vol
    bad = (vol <= 0).any(axis=1)
    n_ej = (nj - 1) // 2
    return {"non_positive_elements": int(bad.sum()),
            "min_det_j": float(vol.min()),
            "bad_elements_ij": [[int(e) // n_ej, int(e) % n_ej]
                                for e in np.nonzero(bad)[0][:8]]}


def build_verdict(genes, cfg, fillet):
    """PART 5's criterion: does `build_wheel` raise, and on how many elements."""
    try:
        mesh = WW.build_wheel(genes, cfg, fillet=fillet)
    except ValueError as exc:
        return {"raises": True, "message": str(exc).split(" — ")[0]}, None
    return {"raises": False, "message": None}, mesh


def mesh_gauss_verdict(mesh):
    """Criterion C on the ASSEMBLED wheel, and which blocks own the bad elements.

    `build_wheel` has already flipped whole left-handed blocks by the time this runs, so
    no sign normalisation is wanted here: a negative `det J` at this point is a genuine
    disagreement with the mesh the solver would assemble, not an indexing convention.
    """
    vol = np.asarray(fem.gauss_volume(np.asarray(mesh.coords), mesh.conn,
                                      order=mesh.cfg.order, width=1.0))
    bad = (vol <= 0).any(axis=1)
    names, counts = np.unique(np.asarray(mesh.element_block)[bad], return_counts=True)
    return {"non_positive_elements": int(bad.sum()),
            "min_det_j": float(vol.min()),
            "by_block": {str(n): int(c) for n, c in zip(names, counts)}}


# ---------------------------------------------------------------------------
# THE DIAGNOSTICS THAT SAY WHICH PART OF THE CONSTRUCTION IS AT ITS LIMIT
# ---------------------------------------------------------------------------

def diagnostics(genes, cfg, junction, R, grid):
    """Node allocation and end-cross-section growth at one radius.

    `arc_cells` is `_filleted_spoke`'s `k0` (or `k1` at the rim) read back off the
    geometry rather than recomputed, so it cannot drift from the construction.  The
    arc's centre is recoverable exactly: `_fillet_tangency` puts `B` on the ring circle
    RADIALLY BELOW the centre (`B = C * ring_r / |C|`), so `C` is the ring-end station
    pushed `R` along its own radius, and the stations on the arc are the ones exactly
    `R` from it.

    `mid_frac` is where the first element's mid-side node sits along its own edge.  A Q9
    edge is singular at its end once that falls outside (0.25, 0.75), and the measured
    fold window opens at ~0.4 -- see the module docstring.
    """
    g = np.asarray(grid, float)
    row, step = (0, 1) if junction == "hub" else (-1, -1)
    # The straddling flank is the eta side the fillet was cut into: the one whose end
    # point moved.  Compare against the unfilleted block rather than assuming a side.
    g0 = np.asarray(WW.sector_blocks(genes, cfg, fillet=(0.0, 0.0))["spoke"], float)
    moved = [j for j in (0, g.shape[1] - 1)
             if np.linalg.norm(g[row, j] - g0[row, j]) > 1e-9]
    j_f = moved[0] if moved else g.shape[1] - 1
    j_o = 0 if j_f else g.shape[1] - 1

    line = g[::step, j_f]
    on_arc = 0
    if R > 0.0:
        p0 = line[0]
        for sign in (+1.0, -1.0):                    # hub: centre outside; rim: inside
            C = p0 * (1.0 + sign * R / float(np.linalg.norm(p0)))
            d = np.abs(np.linalg.norm(line - C, axis=1) - R)
            if d[1] < 1e-7:
                on_arc = int(np.argmax(d > 1e-7)) - 1 if (d > 1e-7).any() else len(d) - 1
                break
    p0, p1, p2 = line[0], line[1], line[2]
    mid_frac_f = float(np.linalg.norm(p1 - p0) / np.linalg.norm(p2 - p0))
    o = g[::step, j_o]
    mid_frac_o = float(np.linalg.norm(o[1] - o[0]) / np.linalg.norm(o[2] - o[0]))
    return {"arc_cells": int(on_arc),
            "mid_frac_fillet_flank": mid_frac_f,
            "mid_frac_far_flank": mid_frac_o,
            "end_cross_section_mm": float(np.linalg.norm(g[row, -1] - g[row, 0]))}


# ---------------------------------------------------------------------------
# THE SWEEP
# ---------------------------------------------------------------------------

def sweep_one(genes, cfg, junction, radii):
    """Every criterion at every radius, for one config and one junction."""
    base = np.asarray(WW.sector_blocks(genes, cfg, fillet=(0.0, 0.0))["spoke"], float)
    base_xs = float(np.linalg.norm(base[0 if junction == "hub" else -1, -1]
                                   - base[0 if junction == "hub" else -1, 0]))
    rows = []
    for R in radii:
        fillet = (R, 0.0) if junction == "hub" else (0.0, R)
        row = {"radius_mm": float(R)}
        try:
            spoke = np.asarray(WW.sector_blocks(genes, cfg, fillet=fillet)["spoke"], float)
        except ValueError as exc:
            row["sector_blocks_raises"] = str(exc)[:120]
            rows.append(row)
            continue
        row["block_cells"] = cell_verdict(spoke)
        row["gauss"] = gauss_verdict(spoke)
        row["build_wheel"], mesh = build_verdict(genes, cfg, fillet)
        row["mesh_gauss"] = None if mesh is None else mesh_gauss_verdict(mesh)
        row.update(diagnostics(genes, cfg, junction, R, spoke))
        row["end_cross_section_ratio"] = row["end_cross_section_mm"] / base_xs
        row["folds"] = {"block_cells": row["block_cells"]["mixed_sign_cells"] > 0,
                        "build_wheel": row["build_wheel"]["raises"],
                        "gauss": row["gauss"]["non_positive_elements"] > 0,
                        # a mesh that will not build cannot be integrated either
                        "mesh_gauss": (mesh is None
                                       or row["mesh_gauss"]["non_positive_elements"] > 0)}
        rows.append(row)
    return rows


def _clean_windows(rows, criterion):
    """Maximal runs of consecutive swept radii with no fold, as [lo, hi] pairs."""
    out, run = [], []
    for r in rows:
        if "folds" in r and not r["folds"][criterion]:
            run.append(r["radius_mm"])
        elif run:
            out.append([run[0], run[-1]])
            run = []
    if run:
        out.append([run[0], run[-1]])
    return out


def summarize(rows, grid):
    """Largest surviving radius and first fold, per criterion, on a stated grid.

    Both statistics are grid-dependent by construction -- "the largest radius that
    survives" is only ever the largest one that was TRIED -- so the grid is carried in
    the report next to them.  PART 3 and PART 5 both omitted theirs, and half the reason
    their rows looked irreconcilable is that nobody could tell whether the difference was
    the criterion or the sample points.
    """
    on_grid = [r for r in rows if r["radius_mm"] in grid and "folds" in r]
    out = {"grid_mm": list(grid)}
    for crit in CRITERIA:
        first = next((r["radius_mm"] for r in on_grid if r["folds"][crit]), None)
        largest = None
        for r in on_grid:
            if r["folds"][crit]:
                break
            largest = r["radius_mm"]
        out[crit] = {"largest_surviving_mm": largest, "first_fold_mm": first}
    return out


def build(genes, configs, junctions, radii):
    rec = {"configs": list(configs), "junctions": list(junctions),
           "radii_mm": [float(R) for R in radii], "sweep": {}, "summary": {}}
    for cfg in configs:
        for junction in junctions:
            key = f"{cfg}:{junction}"
            rows = sweep_one(genes, cfg, junction, radii)
            rec["sweep"][key] = rows
            rec["summary"][key] = {
                "legacy_grid": summarize(rows, LEGACY_GRID),
                "fine_grid": summarize(rows, FINE_GRID),
                "clean_windows_mm": {c: _clean_windows(rows, c) for c in CRITERIA},
            }
    return rec


# ---------------------------------------------------------------------------
# THE SELF-CHECKS
# ---------------------------------------------------------------------------

def controls(genes, configs):
    """`fillet=None` and `fillet=(0, 0)` must be the same mesh, and both must be clean.

    PART 3 measured the second at 2.842e-14 mm and called it an independent numerical
    check that `sample` is affine in eta -- that the unfilleted spoke already IS the
    Coons patch of its own boundary curves, which is why swapping the construction in
    costs nothing at zero radius.  Re-checked here because every number in this file is
    a DIFFERENCE from that baseline, so a drift in it would move all of them.
    """
    out = {}
    for cfg in configs:
        a = np.asarray(WW.sector_blocks(genes, cfg, fillet=None)["spoke"], float)
        b = np.asarray(WW.sector_blocks(genes, cfg, fillet=(0.0, 0.0))["spoke"], float)
        out[cfg] = {
            "max_abs_dx_mm": float(np.abs(a - b).max()),
            "none_clean": (cell_verdict(a)["mixed_sign_cells"] == 0
                           and gauss_verdict(a)["non_positive_elements"] == 0),
            "zero_clean": (cell_verdict(b)["mixed_sign_cells"] == 0
                           and gauss_verdict(b)["non_positive_elements"] == 0),
            "build_wheel_none_raises": build_verdict(genes, cfg, None)[0]["raises"],
        }
        out[cfg]["pass"] = bool(out[cfg]["max_abs_dx_mm"] < 1e-12
                                and out[cfg]["none_clean"] and out[cfg]["zero_clean"]
                                and not out[cfg]["build_wheel_none_raises"])
    return out


def reconcile(rec):
    """Reproduce both contested tables, and state which criterion each one was.

    This is the whole point of the file, so it is computed and checked rather than
    written into the prose: if a future change to `_filleted_spoke` moves either row,
    this driver goes red and says which one moved.
    """
    part3, part5, ok = {}, {}, True
    for key, summ in rec["summary"].items():
        cfg, junction = key.split(":")
        got3 = summ["legacy_grid"]["block_cells"]["largest_surviving_mm"]
        got5 = summ["legacy_grid"]["build_wheel"]["first_fold_mm"]
        want3 = PART3_LARGEST_SURVIVING.get((cfg, junction))
        want5 = PART5_FIRST_FOLD.get((cfg, junction))
        part3[key] = {"recorded_mm": want3, "measured_mm": got3,
                      "agrees": want3 is None or got3 == want3}
        part5[key] = {"recorded_mm": want5, "measured_mm": got5,
                      "agrees": want5 is None or got5 == want5}
        ok = ok and part3[key]["agrees"] and part5[key]["agrees"]
    return {
        "part3_criterion": "block_cells (mixed-sign bilinear cells in the spoke block)",
        "part5_criterion": "build_wheel (4-corner shoelace per Q9 element, after orientation)",
        "part3_largest_surviving": part3,
        "part5_first_fold": part5,
        "pass": bool(ok),
    }


def mechanism(rec):
    """Pin both edges of the usable window to the construction detail that sets them.

    Stated as measured coincidences rather than as prose, because both are claims a
    change to `_filleted_spoke` would falsify and nobody would notice:

      UPPER EDGE.  The last usable radius is the last one whose fillet arc occupies ONE
      cell.  `k0 = clip(round((s_A - s0) / ds), 1, cap)` stepping to 2 is what ends the
      window -- a node-allocation event, not a geometric one.

      LOWER EDGE.  Below the window the same clamp's LOWER bound is what bites: it holds
      the arc on one cell when the tangent point is much nearer than one station, which
      drags the first Q9 element's mid-side node toward its own end.  A quadratic edge
      whose mid node sits at a fraction outside (0.25, 0.75) is singular at that end;
      measured, the window opens as that fraction climbs back through ~0.4.

    Also recorded: whether PART 3's criterion agrees with the upper edge on the FINE
    grid.  It does, at every cell -- which is the strongest form of the reconciliation.
    """
    out = {}
    for key, rows in rec["sweep"].items():
        win = rec["summary"][key]["clean_windows_mm"]["mesh_gauss"]
        by_r = {r["radius_mm"]: r for r in rows if "folds" in r}
        entry = {"usable_window_mm": win[0] if win else None}
        if win:
            lo, hi = win[0]
            above = [R for R in sorted(by_r) if R > hi]
            below = [R for R in sorted(by_r) if R < lo]
            first_above = by_r[above[0]] if above else None
            last_below = by_r[below[-1]] if below else None
            entry["upper_edge_arc_cells"] = by_r[hi]["arc_cells"]
            entry["first_fold_above_arc_cells"] = (
                None if first_above is None else first_above["arc_cells"])
            entry["upper_edge_is_arc_cell_step"] = bool(
                first_above is not None
                and by_r[hi]["arc_cells"] == 1 and first_above["arc_cells"] == 2)
            entry["end_cross_section_ratio_at_upper_edge"] = by_r[hi][
                "end_cross_section_ratio"]
            entry["mid_frac_bracket_at_lower_edge"] = [
                None if last_below is None else last_below["mid_frac_fillet_flank"],
                by_r[lo]["mid_frac_fillet_flank"]]
        fine = rec["summary"][key]["fine_grid"]["block_cells"]["largest_surviving_mm"]
        entry["part3_criterion_fine_grid_mm"] = fine
        entry["part3_criterion_matches_upper_edge"] = bool(win and fine == win[0][1])
        out[key] = entry
    return out


def default_path_blindness(n_feasible, cfg_name, seed=0):
    """Is the guard's blind spot reachable WITHOUT a fillet?  Sampled over the gene box.

    Asked because a blind spot demonstrated only on an opt-in path is a note about that
    path, while one reachable on the default path is a defect in the tree's mesh
    validity.  It is the latter for `build_wheel`'s guard alone -- and it is NOT, on this
    sample, for the constraint set as a whole:

      * `feasible_geom` (what `evaluate_design` enforces: x-ordering and hub crowding) is
        not enough.  A fifth of the meshes that BUILD have non-positive Gauss detJ, and
        some of those also clear `study_mesh_quality`'s minSJ >= 0.2 acceptance floor --
        which is corner-only too, so the two checks are blind in the same way.
      * adding `fold_margin > 0` -- the closed-form self-intersection predictor, the
        third of `meshable` -- cuts that rate by ~60x but does NOT take it to zero.
        THAT is what mostly covers the default path, and it covers it for a reason
        unrelated to elements: it rejects the genome before the mesh exists.
        The residue is small and it is real; it is not this arc's to chase, and it is
        recorded here rather than argued away.

    Which is exactly why it cannot help the fillet path.  `fold_margin` reads genes 0-11,
    the centreline and the thickness; `R_hub` and `R_rim` are genes 12 and 13 and it
    never sees them.  The one constraint that closes this hole on the default path is
    blind to the parameter that opens it here.

    Sampled rather than argued, with a fixed seed so the artifact reproduces.
    """
    from study_mesh_quality import fold_margin, latin_hypercube, MIN_SJ_ACCEPT
    import wheel_fea as W
    import wheel_genome as GN

    low, high, _ = GN.bounds_arrays(W.GENE_SPACE)
    cfg = WW.get_config(cfg_name)
    n = {"feasible_drawn": 0, "built": 0, "built_non_positive_gauss": 0,
         "built_non_positive_gauss_and_min_sj_ok": 0, "meshable": 0,
         "meshable_non_positive_gauss": 0,
         "meshable_non_positive_gauss_and_min_sj_ok": 0}
    batch = 0
    while n["feasible_drawn"] < n_feasible and batch < 200:
        for vec in latin_hypercube(2048, low, high, seed=seed + batch):
            _, loss = W.evaluate_design(vec)
            if loss["x_order"] != 0.0 or loss["hub_overlap"] != 0.0:
                continue
            n["feasible_drawn"] += 1
            margin = fold_margin(vec, cfg)
            try:
                mesh = WW.build_wheel(vec, cfg_name)
            except ValueError:
                if n["feasible_drawn"] >= n_feasible:
                    break
                continue
            n["built"] += 1
            xy = np.asarray(mesh.coords)
            bad = mesh_gauss_verdict(mesh)["non_positive_elements"] > 0
            sj_ok = float(WM.scaled_jacobian(xy, mesh.conn).min()) >= MIN_SJ_ACCEPT
            n["built_non_positive_gauss"] += bad
            n["built_non_positive_gauss_and_min_sj_ok"] += bad and sj_ok
            if margin > 0.0:
                n["meshable"] += 1
                n["meshable_non_positive_gauss"] += bad
                n["meshable_non_positive_gauss_and_min_sj_ok"] += bad and sj_ok
            if n["feasible_drawn"] >= n_feasible:
                break
        batch += 1
    n = {k: int(v) for k, v in n.items()}
    n["config"] = cfg_name
    n["seed"] = seed
    n["min_sj_accept"] = MIN_SJ_ACCEPT
    # Measured, not asserted: move both fillet genes across their whole box and see
    # whether the margin notices.  It does not, and that is the load-bearing half of the
    # paragraph above.
    probe = np.array(load_genes("best_solution.json"), float)
    m0 = float(fold_margin(probe, cfg))
    moved = probe.copy()
    moved[12], moved[13] = float(high[12]), float(high[13])
    n["fold_margin_shipped"] = m0
    n["fold_margin_fillet_genes_at_box_top"] = float(fold_margin(moved, cfg))
    n["fold_margin_reads_fillet_genes"] = bool(
        n["fold_margin_fillet_genes_at_box_top"] != m0)
    return n


def guard_blindness(rec):
    """How far `build_wheel`'s guard lets a folded mesh through, in radii and in points.

    Reported as a characterisation rather than a check.  `_orient_elements` is a shoelace
    over each element's four CORNERS; at `order=2` an element spans 2x2 cells and its
    five mid nodes take no part in that sum, so a fold inside the element is invisible to
    it.  The gap below is the measured size of that blindness on this construction.
    """
    out = {}
    for key, rows in rec["sweep"].items():
        blind = [r for r in rows if "folds" in r
                 and not r["folds"]["build_wheel"] and r["folds"]["mesh_gauss"]]
        out[key] = {
            "n_radii_accepted_with_non_positive_gauss": len(blind),
            "worst_min_det_j": min([r["mesh_gauss"]["min_det_j"] for r in blind],
                                   default=None),
            "max_non_positive_elements": max(
                [r["mesh_gauss"]["non_positive_elements"] for r in blind], default=0),
            "example_radius_mm": blind[0]["radius_mm"] if blind else None,
        }
    return out


# ---------------------------------------------------------------------------

def _print(rec):
    ctl = rec["controls"]
    print("\n  CONTROLS (fillet=None vs fillet=(0,0); both must be clean)")
    for cfg, c in ctl.items():
        print(f"    {cfg:7s} max|dx| = {c['max_abs_dx_mm']:.3e} mm   "
              f"{'PASS' if c['pass'] else 'FAIL'}")

    print("\n  THE SWEEP, on the grid the two contested tables used")
    print(f"    {'':16s} {'PART 3 criterion':>22s} {'PART 5 criterion':>22s} "
          f"{'builds AND integrates':>24s}")
    print(f"    {'config:junction':16s} {'largest surviving':>22s} "
          f"{'first fold':>22s} {'usable window(s), mm':>24s}")
    for key, summ in rec["summary"].items():
        g3 = summ["legacy_grid"]["block_cells"]["largest_surviving_mm"]
        g5 = summ["legacy_grid"]["build_wheel"]["first_fold_mm"]
        win = summ["clean_windows_mm"]["mesh_gauss"]
        wtxt = ", ".join(f"{lo:.2f}-{hi:.2f}" for lo, hi in win) or "none"
        print(f"    {key:16s} {g3 if g3 is not None else 'none':>22} "
              f"{g5 if g5 is not None else 'none':>22} {wtxt:>24s}")

    rc = rec["reconciliation"]
    print("\n  RECONCILIATION (FILLET_PLAN.md PART 5's open discrepancy)")
    for key in rec["summary"]:
        a, b = rc["part3_largest_surviving"][key], rc["part5_first_fold"][key]
        print(f"    {key:16s} PART 3 recorded {a['recorded_mm']}, measured "
              f"{a['measured_mm']}  {'OK' if a['agrees'] else 'DISAGREES'}"
              f"   |   PART 5 recorded {b['recorded_mm']}, measured "
              f"{b['measured_mm']}  {'OK' if b['agrees'] else 'DISAGREES'}")

    print("\n  BOTH EDGES OF THE USABLE WINDOW ARE NODE ALLOCATION, NOT GEOMETRY")
    for key, m in rec["mechanism"].items():
        if m["usable_window_mm"] is None:
            print(f"    {key:16s} no usable radius on this grid")
            continue
        lo_frac = m["mid_frac_bracket_at_lower_edge"]
        print(f"    {key:16s} window {m['usable_window_mm'][0]:.2f}-"
              f"{m['usable_window_mm'][1]:.2f} mm   "
              f"upper edge = arc cells 1->2: "
              f"{'YES' if m['upper_edge_is_arc_cell_step'] else 'no'}   "
              f"lower edge = mid-side fraction {lo_frac[0]:.3f} -> {lo_frac[1]:.3f}   "
              f"PART 3's criterion lands on the upper edge: "
              f"{'YES' if m['part3_criterion_matches_upper_edge'] else 'no'}")

    print("\n  WHAT build_wheel's GUARD DOES NOT SEE")
    for key, g in rec["guard_blindness"].items():
        print(f"    {key:16s} accepts {g['n_radii_accepted_with_non_positive_gauss']:3d} "
              f"swept radii whose meshes have non-positive Gauss detJ "
              f"(worst {g['worst_min_det_j']}, up to "
              f"{g['max_non_positive_elements']} elements)")

    d = rec.get("default_path_blindness")
    if d:
        print(f"\n  AND IT IS REACHABLE WITHOUT A FILLET AT ALL "
              f"({d['feasible_drawn']} feasible draws at {d['config']})")
        print(f"    built by build_wheel                 {d['built']}")
        print(f"      with non-positive Gauss detJ       "
              f"{d['built_non_positive_gauss']}"
              f"  ({100.0 * d['built_non_positive_gauss'] / max(d['built'], 1):.1f}%)")
        print(f"        and min scaled Jacobian >= {d['min_sj_accept']}   "
              f"{d['built_non_positive_gauss_and_min_sj_ok']}"
              f"   <- both corner-only checks blind at once")
        print(f"    ALSO fold_margin > 0 ('meshable')    {d['meshable']}")
        print(f"      with non-positive Gauss detJ       "
              f"{d['meshable_non_positive_gauss']}"
              f" ({d['meshable_non_positive_gauss_and_min_sj_ok']} of them also minSJ "
              f"ok)")
        print(f"                                             <- mostly covered, and not "
              f"by an element check: fold_margin reads")
        print(f"                                                genes 0-11, and "
              f"R_hub/R_rim are 12 and 13, so it cannot cover the fillet")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genome", default="best_solution.json")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    ap.add_argument("--junctions", default=",".join(DEFAULT_JUNCTIONS))
    ap.add_argument("--coarse-grid", action="store_true",
                    help="skip the 0.01 mm sweep and use the legacy grid only")
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS,
                    help="feasible genomes to sample for the default-path section")
    ap.add_argument("--out", default="study_fillet_fold.json")
    args = ap.parse_args()

    # A degraded run may not be filed under the committed artifact's name (PLAN §41,
    # §43).  Refused at startup.  What degrades THIS driver is losing a config the two
    # contested tables were taken at, losing the fine grid that resolves the window
    # edges, or measuring a different wheel.
    _gate_guard.refuse_degraded_out(ap, args, "study_fillet_fold.json", [
        (args.coarse_grid, "--coarse-grid, which cannot resolve the fold window"),
        (set(args.configs.split(",")) != set(DEFAULT_CONFIGS),
         "--configs %s, not the reconciliation's %s"
         % (args.configs, ",".join(DEFAULT_CONFIGS))),
        (set(args.junctions.split(",")) != set(DEFAULT_JUNCTIONS),
         "--junctions %s, not both" % args.junctions),
        (args.genome != "best_solution.json", "--genome %s" % args.genome),
        (args.draws < DEFAULT_DRAWS,
         "--draws %d, below the %d the default-path section is sampled at"
         % (args.draws, DEFAULT_DRAWS)),
    ])

    configs = tuple(args.configs.split(","))
    junctions = tuple(args.junctions.split(","))
    genes = load_genes(args.genome)
    # The two radii that SHIP are swept as well, at both junctions.  They are the numbers
    # the arc exists to mesh, and a sweep that steps over them would have to be read by
    # interpolation.
    shipped = {float(genes[12]), float(genes[13])}
    radii = LEGACY_GRID if args.coarse_grid else tuple(
        sorted(set(FINE_GRID) | set(TAIL_GRID) | set(LEGACY_GRID) | shipped))

    t0 = time.time()
    rec = build(genes, configs, junctions, radii)
    rec["controls"] = controls(genes, configs)
    rec["reconciliation"] = reconcile(rec)
    rec["mechanism"] = mechanism(rec)
    rec["guard_blindness"] = guard_blindness(rec)
    rec["default_path_blindness"] = default_path_blindness(args.draws, configs[0])
    rec["genome"] = args.genome
    rec["shipped_radii_mm"] = {"R_hub": float(genes[12]), "R_rim": float(genes[13])}
    rec["wall_s"] = time.time() - t0
    _print(rec)

    passed = (rec["reconciliation"]["pass"]
              and all(c["pass"] for c in rec["controls"].values()))
    rec["pass"] = bool(passed)
    out = os.path.join(HERE, args.out)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1, default=lambda o: (o.tolist() if hasattr(o, "tolist")
                                                        else str(o)))
    print(f"  wrote {out}   ({rec['wall_s']:.1f} s)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
