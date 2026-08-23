"""PLAN.md §0(a) — the analytic hub-fillet cap against what OCC actually accepts.

`wheel_objective.hub_fillet_cap_mm` claims to know the largest hub fillet the part can
build, from the genome alone and with no CAD kernel in the loop.  Stage 3 now believes that
claim twice — a barrier pushes `R_hub` under it, and `Kt_hub` is priced on it — so the
claim needs a measurement behind it rather than one recorded number from one export.

THREE SECTIONS, AND ONLY THE FIRST TWO ARE GATES.

  `void`       The load-bearing one, and it involves no filleting at all.  A ring of points
               just outside the hub circle is classified against the profile FACE, the runs
               of empty are measured, and the mean run is compared to `hub_void_deg`.  This
               tests the geometry claim directly, with nothing else in the way.

  `occ_limit`  What radius OCC will actually accept on a hub corner, found by BISECTION.

               NOT by reading the ladder, and that distinction is a measurement rather than
               a preference.  The obvious criterion — "the largest ladder rung below the cap
               is what gets built" — is FALSE at the shipped genome: `_fillet_ladder(1.5598)`
               is 1.5598, 1.3258, 1.1269, 0.9579, the largest rung under the 1.1057 cap is
               0.9579, and OCC took 1.1269.  The rungs straddle the cap and which side they
               land on is an accident of where `R_hub` happens to start.  So the acceptance
               threshold is measured where it actually is.

  `sweep`      Reported, never gated: the cap against `R_hub` for every design on disk.
               This is the evidence that a fixed bound would have been right for exactly one
               genome — the caps span 0.99 to 1.53 mm across the 16 Stage-2 elites.

TWO FAMILIES, NAMED BY WEDGE ANGLE, AND EITHER CAN BE THE ONE THAT BINDS.  The hub's
twenty-four corners split into twelve SQUARE-ON ones at a wedge of 266-270 deg, limited by
the root thickness and by the hub ARRIVAL ANGLE, and twelve NEAR-CUSP ones at 294-332 deg,
limited by the slot.  Which family stops the exporter is a property of the design: the
square-on one at `t0` = 1.2, where everything ships, and the near-cusp one on the `t0` = 2.55
Stage-2 elites.  So the gate reads the WORST of the twenty-four rather than a named half —
until 2026-08-10 it read the larger half of a RANK split, which is not the same partition's
label on every design and is how a cap over-promising by up to 1.62x read green.

The old note here said the near-cusp family's arrival-angle limit was "a different mechanism
the cap does not model".  It is modelled as of 2026-08-10 for the square-on family
(`wheel_objective.HUB_CAP_ARRIVAL_SLOPE`, calibrated by `studies/study_arrival_cap.py`).  For
the near-cusp family it is still not: `HUB_CAP_SHARE` is that family's limit as the arrival
goes to zero and is conservative above it.  BUILD_PLAN.md steps 1-4.

Out of `make studies` for the `m8bi5` reason and one more: it needs BOTH interpreters, and
what it measures is OCC's behaviour on this shape rather than anything this commit changed.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import project_paths as PP  # noqa: E402
if PP.SRC not in sys.path:
    sys.path.insert(0, PP.SRC)

import wheel_fea as W  # noqa: E402
import wheel_genome as wg  # noqa: E402
import wheel_objective as WO  # noqa: E402
import wheel_wheel as WW  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CAD_PY = os.path.join(ROOT, ".venv-cad", "bin", "python")

# One sample of the OCC ring classifier below — the RESOLUTION OF THE INSTRUMENT, not a
# tolerance chosen to make the gate pass.  The analytic void and the two independent solid
# measurements on file (9.907 from the milestone's ring sampling, "about 10.0" from
# `_embed`'s own comment) sit 0.070 deg apart, so this runs at ~3.5x margin.
RING_SAMPLES = 14400
GATE_VOID_DEG = 0.25

# The ring is classified just OUTSIDE the hub circle: on it, the corners themselves are
# boundary samples and the classifier's answer there is a coin toss.
RING_OFFSET_MM = 0.01

# THE GATE IS ONE-SIDED, AND THAT IS THE MODEL'S ACTUAL CLAIM.
#
# This started as a two-sided "the cap is within one ladder rung of what OCC accepts", on
# the assumption that `0.5 * slot` WAS the limit.  The first run of this driver falsified
# that (see `wheel_objective.HUB_CAP_SHARE`), and the cap became a `min` of two limits, only
# one of which has ever been observed binding.  A `min` of a measured limit and a
# deliberately-unvalidated one does not claim to be tight — it claims to be SAFE.  Gating it
# on tightness would be gating a claim nothing in the code makes, and the honest way to fail
# such a gate is to loosen the model until it passes, which is backwards.
#
# So: the cap may never promise more than OCC will give (that is the defect being removed),
# and it may not collapse to something vacuously small (a cap of zero is trivially "never
# over-promising" and would sail through a pure one-sided test while destroying every hub
# fillet in the wheel).  The 1% is the bisection's own resolution; the 0.5 floor says the cap
# has to be the right ORDER, which is all a two-term min with one unvalidated share can
# honestly assert.
GATE_CAP_OVERPROMISE = 1.01
GATE_CAP_FLOOR_FRAC = 0.5

BISECT_REL = 0.01               # bisect the acceptance threshold to 1% of its own value

# The wedge angle that separates the two hub-corner families.  Measured wedges cluster at
# 266-270 (square-on) and 294-332 (near-cusp) across every design and every station of
# `studies/study_arrival_cap.py`; 285 is the middle of that 24 deg gap.  A split constant,
# not a threshold anything is gated on — `families_are_clean` reports whether a corner ever
# landed near it.
CUSP_WEDGE_DEG = 285.0


_CAD_SNIPPET = r"""
import json, math, sys
import wheel_step_export as X
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_State

payload = json.loads(sys.argv[1])
n = payload["ring_samples"]
ring_r = payload["ring_r"]
bisect_rel = payload["bisect_rel"]
do_bisect = payload["bisect"]

def void_runs(face, radius):
    # Angular widths [deg] of the maximal empty runs on a ring, circularly.
    out = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        cl = BRepClass_FaceClassifier(
            face, gp_Pnt(radius * math.cos(t), radius * math.sin(t), 0.0), 1e-9)
        out.append(cl.State() == TopAbs_State.TopAbs_OUT)
    if all(out):
        return [360.0]
    if not any(out):
        return []
    # Rotate so index 0 is MATERIAL, then every run of empties is contiguous.
    k = out.index(False)
    out = out[k:] + out[:k]
    runs, cur = [], 0
    for v in out:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return [360.0 * c / n for c in runs]

def accepts(part, edge, r):
    try:
        return bool(part.newObject([edge]).fillet(r).val().isValid())
    except Exception:
        return False

def threshold(part, edge, lo, hi):
    # Largest radius this corner accepts, bracketed to `bisect_rel` of itself.
    if not accepts(part, edge, lo):
        return 0.0
    if accepts(part, edge, hi):
        return hi                       # above the bracket; reported as such
    while (hi - lo) > bisect_rel * lo:
        mid = 0.5 * (lo + hi)
        if accepts(part, edge, mid):
            lo = mid
        else:
            hi = mid
    return lo

rows = []
for d in payload["designs"]:
    rec = {"design": d["name"]}
    try:
        profile, _ = X.build_profile(d["genes"])
        part = X.extrude_profile(profile)
        rec["void_runs_deg"] = void_runs(profile.wrapped, ring_r)
        if do_bisect:
            hub, _rim = X._group_by_ring(X._junction_edges(part))
            rec["n_hub_corners"] = len(hub)
            hi = d["hi_mm"]
            rec["thresholds_mm"] = [
                threshold(part, e, X.MIN_CURVATURE_RADIUS_MM, hi) for e, _r, _w in hub]
            # THE PER-CORNER WEDGE, PAIRED WITH ITS OWN THRESHOLD.  `_group_by_ring`
            # already carries it and this used to drop it on the floor, which is why the
            # shallow/square split downstream had to be taken at the MEDIAN on the
            # assumption the families are equal-sized.  With the wedge in hand the split
            # is observable instead, and BUILD_PLAN.md step 2 is about the per-corner
            # relationship rather than the per-junction one.  Same order as
            # `thresholds_mm` -- index i of one is index i of the other.
            rec["wedges_deg"] = [float(w) for _e, _r, w in hub]
            rec["bisect_hi_mm"] = hi
    except Exception as exc:
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
    rows.append(rec)

print("RESULT:" + json.dumps(rows))
"""


def _genes(path):
    with open(path) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


def _designs():
    """Every genome on disk, `best_solution` first.

    The gate runs on three of them and the reason each is there is different: the shipped
    genome because it is the one every recorded number describes, elite14 because it has
    the TIGHTEST cap on disk (0.9898 mm), and elite13 because it is the ONE design whose
    `R_hub` is already under its cap — a design the cap must not bind, which makes it a
    falsifiable negative control rather than a fourth confirmation.
    """
    out = [("best_solution", _genes(PP.BEST_SOLUTION))]
    with open(PP.STAGE2_ELITES) as fh:
        for row in json.load(fh)["elites"]:
            out.append((f"elite{int(row['rank'])}", wg.genes_to_vector(row["genes"])))
    return out


def _extra_designs(specs):
    """`--extra LABEL=PATH` -> [(label, gene vector)], in the order given.

    Genome files that are not `best_solution.json` and not a Stage-2 elite have no place in
    `_designs()`, which walks the provenance chain.  A Stage-3 answer is not on that chain
    — it is a search result — so it arrives by name and by path, which also means the
    report says exactly which file each row came from (`settings.extra`).
    """
    out = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--extra wants LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        label, path = label.strip(), path.strip()
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
        if not os.path.exists(path):
            raise SystemExit(f"--extra {label}: no such file {path}")
        out.append((label, _genes(path)))
    return out


def analytic_cap(genes, cfg="coarse"):
    """`(void_deg, cap_mm, r_hub_mm)` from the genome alone — no CAD."""
    cfgo = WW.get_config(cfg)
    flanks = WO.fillet_flanks(genes, cfgo)
    void = float(WW.hub_void_deg(genes, cfgo, W.S, W.HUB_RADIUS_MM, flanks[0]))
    cap = float(WO.hub_fillet_cap_mm(genes, cfgo, W.S, W.HUB_RADIUS_MM, flanks))
    return void, cap, float(genes[12])


def _run_cad(designs, caps, bisect):
    payload = {
        "ring_samples": RING_SAMPLES,
        "ring_r": W.HUB_RADIUS_MM + RING_OFFSET_MM,
        "bisect_rel": BISECT_REL,
        "bisect": bool(bisect),
        # The DICT form: `wheel_step_export.build_profile` keys by gene name, and the
        # 14-vector is the optimizer side's representation, not the exporter's.
        #
        # `hi_mm` brackets the bisection from ABOVE and has to clear whichever limit
        # actually binds, or every corner reports the bracket instead of its threshold.
        # `1.2 * t0` is comfortably above the 0.505 * t0 the square-on law predicts at its
        # most generous (a tangential arrival) and further above it at every steeper one,
        # and `2 * slot cap` covers the case where the slot is the binding one.  NOTE that
        # `1.2 * t0` dominates on every design measured so far, so re-fitting the shares
        # downward on 2026-08-10 did not move a single bracket — and the near-cusp family
        # on the `t0` = 1.2 designs stays CENSORED at 1.44 for exactly the same reason it
        # was before.
        "designs": [{"name": n, "genes": wg.vector_to_genes(g), "cap_mm": caps[n],
                     "hi_mm": max(2.0 * caps[n], 1.2 * float(g[8]), 1.0)}
                    for n, g in designs],
    }
    proc = subprocess.run(
        [CAD_PY, "-c", _CAD_SNIPPET, json.dumps(payload)], cwd=ROOT,
        env={**os.environ, "PYTHONPATH": SRC},
        capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"CAD subprocess failed:\n{proc.stderr[-3000:]}")
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:"))
    return json.loads(line[len("RESULT:"):])


# ---------------------------------------------------------------------------
# SECTIONS
# ---------------------------------------------------------------------------

def run_void(selected, cad_rows):
    """The analytic void against the one OCC measures on the profile.  THE GATE."""
    rows = []
    for (name, genes), cad in zip(selected, cad_rows):
        void, cap, r_hub = analytic_cap(genes)
        rec = {"design": name, "void_analytic_deg": void, "cap_mm": cap,
               "r_hub_mm": r_hub}
        if "error" in cad:
            rec.update({"error": cad["error"], "pass": False})
            rows.append(rec)
            continue
        runs = cad["void_runs_deg"]
        rec["n_runs"] = len(runs)
        rec["void_occ_mean_deg"] = float(np.mean(runs)) if runs else 0.0
        rec["void_occ_spread_deg"] = float(max(runs) - min(runs)) if runs else 0.0
        rec["delta_deg"] = abs(void - rec["void_occ_mean_deg"])
        # Three conditions.  The spread one is free: twelve identical spokes must leave
        # twelve identical gaps, so a spread above one sample says the profile is not
        # twelve-fold symmetric and every other number here is an average over unlike
        # things.
        rec["pass"] = bool(rec["n_runs"] == W.NUMBER_OF_SPOKES
                           and rec["delta_deg"] <= GATE_VOID_DEG
                           and rec["void_occ_spread_deg"] <= GATE_VOID_DEG)
        rows.append(rec)
    return {"rows": rows, "gate_void_deg": GATE_VOID_DEG,
            "ring_r_mm": W.HUB_RADIUS_MM + RING_OFFSET_MM,
            "pass": bool(all(r["pass"] for r in rows))}


def run_occ_limit(selected, cad_rows):
    """What OCC accepts, bisected, against the cap.  THE OTHER GATE."""
    rows = []
    for (name, genes), cad in zip(selected, cad_rows):
        _void, cap, r_hub = analytic_cap(genes)
        rec = {"design": name, "cap_mm": cap, "r_hub_mm": r_hub}
        if "error" in cad or "thresholds_mm" not in cad:
            rec.update({"error": cad.get("error", "no thresholds"), "pass": False})
            rows.append(rec)
            continue
        th = sorted(cad["thresholds_mm"])
        rec["n_corners"] = len(th)
        rec["thresholds_mm"] = th
        half = len(th) // 2
        # Two families by construction — twelve square-on corners and twelve shallow
        # near-cusp ones.  The split is at the median because the clusters are known to be
        # equal-sized (one per flank per spoke), not because the gap is looked for.
        shallow, square = th[:half], th[half:]
        rec["shallow_mean_mm"] = float(np.mean(shallow)) if shallow else 0.0
        rec["square_mean_mm"] = float(np.mean(square)) if square else 0.0

        # THE GATE READS THE WORST CORNER, and it did not used to.  BUILD_PLAN.md step 4.
        #
        # It read `square_mean_mm` — the mean of the LARGER half of a rank split — on the
        # reading that the larger family is the square-on one the cap models.  Pairing each
        # threshold with its own wedge (step 1) showed the rank split does not carry that
        # label: on `350f4c7` the smaller half IS the 270 deg square-on family and the
        # larger half is the 328 deg cusp, and on elite14 it is the other way round.  So the
        # gate was comparing the cap against whichever family happened to rank higher, which
        # on three of five designs is the family that is NOT the limit.  A cap over-promising
        # against the corner OCC actually stops at by 1.067x to 1.615x reported
        # `over_promises: false` on all five.
        #
        # The exporter's ladder is driven by the corner that refuses first, so the honest
        # comparison needs no families at all: the MINIMUM over all twenty-four.  Kept
        # label-free on purpose — `worst_corner_mm` cannot be mis-assigned the way a named
        # half can.
        rec["worst_corner_mm"] = float(min(th))
        r_occ = rec["worst_corner_mm"]
        rec["rel_error"] = abs(r_occ / cap - 1.0) if cap > 0 else float("inf")
        # `cap / r_occ`: below 1.0 the cap under-promises (safe), above it over-promises
        # (the defect).  Reported as the headline because it is the number the gate reads.
        rec["conservatism"] = (cap / r_occ) if r_occ > 0 else float("inf")
        rec["over_promises"] = bool(rec["conservatism"] > GATE_CAP_OVERPROMISE)
        rec["vacuously_small"] = bool(rec["conservatism"] < GATE_CAP_FLOOR_FRAC)
        rec["pass"] = bool(not rec["over_promises"] and not rec["vacuously_small"])

        # ---- BUILD_PLAN.md step 1/2: the per-corner view, added, nothing above changed.
        #
        # THE SPLIT ABOVE IS TAKEN AT THE MEDIAN and says so — "because the clusters are
        # known to be equal-sized, not because the gap is looked for".  With the wedge now
        # paired to its own threshold that assumption is CHECKABLE, and an assumption that
        # can be checked and is not is how a calibration goes stale.  `split_agrees_with_
        # wedge` is False when ordering the corners by wedge does not reproduce the same
        # two families ordering them by threshold does — in which case the shallow/square
        # means above are averages over unlike things and must not be quoted.
        wedges = cad.get("wedges_deg")
        if wedges and len(wedges) == len(cad["thresholds_mm"]):
            pairs = sorted(zip(wedges, cad["thresholds_mm"]))
            rec["corners"] = [{"wedge_deg": w, "threshold_mm": t} for w, t in pairs]
            by_wedge = [t for _w, t in pairs]
            rec["split_agrees_with_wedge"] = bool(
                sorted(by_wedge[:half]) == sorted(th[:half])
                or sorted(by_wedge[half:]) == sorted(th[:half]))
            # THE FAMILIES, NAMED BY THE THING THAT DEFINES THEM.  Every wedge measured so
            # far is either 266-270 (a square-on corner: the flank meets the hub circle at
            # right angles, and the root THICKNESS is what limits it) or 294-332 (a
            # near-cusp corner, which the SLOT between adjacent roots limits).  The 24 deg
            # gap between those two bands is wide and `CUSP_WEDGE_DEG` sits in the middle
            # of it; if a design ever lands a corner inside the gap, `families_are_clean`
            # goes False rather than silently sorting it into one side.
            sq = [t for w, t in pairs if w < CUSP_WEDGE_DEG]
            cu = [t for w, t in pairs if w >= CUSP_WEDGE_DEG]
            rec["square_on_mm"] = float(np.mean(sq)) if sq else None
            rec["near_cusp_mm"] = float(np.mean(cu)) if cu else None
            rec["binds_at_wedge_deg"] = float(min(pairs, key=lambda p: p[1])[0])
            rec["families_are_clean"] = bool(
                len(sq) == len(cu) == half
                and all(w <= 280.0 or w >= 290.0 for w, _t in pairs))
        # The mechanism the cap MODELS AS OF 2026-08-10, reported beside the thing it
        # predicts.  Free — it is analytic and already in the objective.  It used to be
        # labelled "named-but-never-modelled"; `HUB_CAP_ARRIVAL_SLOPE` is that label coming
        # off.
        a_hub, a_rim = WW.arrival_angles(genes, WW.get_config("coarse"), xp=np)
        rec["arrival_hub_deg"] = float(a_hub)
        rec["arrival_rim_deg"] = float(a_rim)
        rows.append(rec)
    return {"rows": rows, "gate_cap_overpromise": GATE_CAP_OVERPROMISE,
            "gate_cap_floor_frac": GATE_CAP_FLOOR_FRAC,
            "bisect_rel": BISECT_REL,
            "pass": bool(all(r["pass"] for r in rows))}


T0_SWEEP = (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


def t0_sweep_designs(base_name="best_solution", stations=T0_SWEEP):
    """One shape, `t0` swept across its whole box — the measurement the disk cannot give.

    Every genome on disk sits at `t0` between 2.468 and 2.627, i.e. 6% of a box that runs
    2.0 to 10.0, so measuring more of them cannot say whether the thickness law holds
    anywhere except at one thickness.  Holding the centerline and sweeping `t0` does, and
    it does something better as well: a thicker root leaves a NARROWER void, so the two
    candidate limits move in opposite directions and their crossover is observable rather
    than assumed.  That crossover is the whole justification for taking a `min`.
    """
    by_name = dict(_designs())
    base = np.asarray(by_name[base_name], dtype=float)
    out = []
    for t0 in stations:
        g = base.copy()
        g[8] = t0
        out.append((f"t0_{t0:g}", g))
    return out


def run_t0_sweep(rows, cad_rows, stations=T0_SWEEP):
    """Threshold against `t0` and against the slot, over the whole thickness box."""
    out = []
    for (name, genes), cad in zip(rows, cad_rows):
        # BOTH LIMITS SEPARATELY, not just the `min` — the whole point of the sweep is to
        # watch them cross, and a column that silently reports whichever one won cannot
        # show that.  `analytic_cap` returns the min, which is the shipped cap.
        void, cap, _r = analytic_cap(genes)
        by_slot = WO.HUB_CAP_SHARE * W.HUB_RADIUS_MM * np.radians(void)
        # The arrival is CONSTANT down this sweep — only `t0` moves — so the arrival factor
        # is a constant multiplier here.  Written out rather than folded into a share so the
        # column stays comparable with `wheel_objective`'s own branch.
        a_hub = float(WW.arrival_angles(genes, WW.get_config("coarse"), xp=np)[0])
        by_thickness = float(genes[8]) * (WO.HUB_CAP_THICKNESS_SHARE
                                          - WO.HUB_CAP_ARRIVAL_SLOPE
                                          * (1.0 - np.cos(np.radians(a_hub))))
        rec = {"design": name, "t0_mm": float(genes[8]), "void_deg": void,
               "arrival_hub_deg": a_hub,
               "cap_mm": cap, "slot_cap_mm": float(by_slot),
               "thickness_cap_mm": by_thickness,
               "binds": "slot" if by_slot < by_thickness else "thickness"}
        if "thresholds_mm" not in cad:
            rec["error"] = cad.get("error", "no thresholds")
            out.append(rec)
            continue
        th = sorted(cad["thresholds_mm"])
        half = len(th) // 2
        rec["n_corners"] = len(th)
        rec["shallow_mean_mm"] = float(np.mean(th[:half])) if half else 0.0
        rec["square_mean_mm"] = float(np.mean(th[half:])) if half else 0.0
        # `share_of_t0` IS THE CALIBRATION THIS SECTION EXISTS TO PRODUCE, so it has to come
        # from the square-on family by NAME.  It used to come from `square_mean_mm`, the
        # larger half of a rank split, which is the same mis-labelling `run_occ_limit`
        # carried — see there.  It happened to pick the right family down this sweep,
        # because a thicker root closes the slot and drags the cusp family under; that is
        # luck, and it stops being luck the moment the base design changes.
        wedges = cad.get("wedges_deg")
        if wedges and len(wedges) == len(cad["thresholds_mm"]):
            pairs = sorted(zip(wedges, cad["thresholds_mm"]))
            sq = [t for w, t in pairs if w < CUSP_WEDGE_DEG]
            rec["square_on_mm"] = float(np.mean(sq)) if sq else 0.0
            rec["near_cusp_mm"] = float(np.mean([t for w, t in pairs
                                                 if w >= CUSP_WEDGE_DEG])) or 0.0
            # A MEAN OVER A FAMILY IS ONLY A MEASUREMENT IF THE FAMILY IS TIGHT, and down
            # this sweep it is not always: the `t0` = 6 row reports a square-on share of
            # 0.29 against 0.50-0.54 at every other intact station, and nothing in the row
            # said whether that is one shape or twelve unlike corners averaged.  Reported
            # so the next run answers it instead of the next reader guessing.
            rec["square_on_spread"] = (float(max(sq) / min(sq) - 1.0)
                                       if sq and min(sq) > 0 else None)
            rec["corners"] = [{"wedge_deg": w, "threshold_mm": t} for w, t in pairs]
        else:
            rec["square_on_mm"] = rec["square_mean_mm"]
        r_sq = rec["square_on_mm"]
        rec["at_bracket"] = bool(r_sq >= cad["bisect_hi_mm"] - 1e-9)
        rec["share_of_t0"] = r_sq / genes[8]
        rec["cap_over_occ"] = (rec["cap_mm"] / min(th)) if min(th) > 0 else float("inf")
        # A ROW THAT IS NOT MEASURING THE SAME FEATURE.  Past t0 ~ 3 the void collapses and
        # then goes negative — adjacent roots have merged — so there is no spoke-to-hub
        # corner left and `_junction_edges` reports whatever re-entrant edges the merged
        # blob happens to have.  Those thresholds are real numbers about a different shape,
        # and averaging them into a calibration of the spoke-to-hub fillet would be reading
        # the instrument through the wrong window.  Kept in the report, excluded from the fit.
        #
        # THE CORNER COUNT IS THE DIRECT TEST AND IT USED TO BE THE VOID.  `void > 1.0` is a
        # proxy for "the roots have not merged", and BUILD_PLAN.md step 4 measured it too
        # permissive: at `t0` = 8 the void still reads 2.577 deg and `_junction_edges`
        # already finds **48** corners rather than 24, so that row was in the fit while
        # describing a different junction.  Two corners per spoke is what an intact
        # spoke-to-hub junction has; anything else is the merged blob, whatever the void
        # says.  Observed rather than assumed, which is the same move `split_agrees_with_
        # wedge` made for the family split.
        n_ok = rec["n_corners"] == 2 * W.NUMBER_OF_SPOKES
        rec["topology_is_intact"] = bool(n_ok)
        rec["same_feature"] = bool(void > 1.0 and n_ok and not rec["at_bracket"])
        out.append(rec)
    good = [r for r in out if r.get("same_feature")]
    return {"rows": out, "t0_values": list(stations),
            "n_same_feature": len(good),
            "t0_valid_range": ([min(r["t0_mm"] for r in good),
                                max(r["t0_mm"] for r in good)] if good else None),
            "share_of_t0_min": min((r["share_of_t0"] for r in good), default=None),
            "share_of_t0_max": max((r["share_of_t0"] for r in good), default=None),
            # WHAT THE MODEL CLAIMS AT THIS SWEEP'S ARRIVAL, so the two columns can be read
            # against each other.  Constant down the sweep — only `t0` moves — and it is
            # the number `share_of_t0` has to stay ABOVE for the fit to be conservative.
            "model_share_of_t0": (out[0]["thickness_cap_mm"] / out[0]["t0_mm"]
                                  if out else None),
            "arrival_hub_deg": out[0]["arrival_hub_deg"] if out else None,
            # Reported, not gated: this section EXISTS to calibrate, so gating it on the
            # constant it is measuring would be circular.
            "pass": True}


def run_sweep(designs):
    """Every design on disk.  REPORTED, NEVER GATED — this is the picture, not a claim."""
    rows = []
    for name, genes in designs:
        void, cap, r_hub = analytic_cap(genes)
        cfgo = WW.get_config("coarse")
        a_hub, _a_rim = WW.arrival_angles(genes, cfgo, span_mm=W.S)
        rows.append({"design": name, "void_deg": void, "cap_mm": cap, "r_hub_mm": r_hub,
                     "binds": bool(r_hub > cap), "over_by_mm": r_hub - cap,
                     "arrival_hub_deg": float(a_hub), "t0_mm": float(genes[8])})
    n_bind = sum(r["binds"] for r in rows)
    return {"rows": rows, "n_binding": n_bind, "n_designs": len(rows),
            "cap_min_mm": min(r["cap_mm"] for r in rows),
            "cap_max_mm": max(r["cap_mm"] for r in rows),
            "pass": True}


def plot(rep, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sw = rep.get("sweep")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    if sw:
        rows = sw["rows"]
        caps = [r["cap_mm"] for r in rows]
        rh = [r["r_hub_mm"] for r in rows]
        col = ["tab:red" if r["binds"] else "tab:green" for r in rows]
        ax.scatter(caps, rh, c=col, s=42, zorder=3)
        lim = [0.9 * min(caps + rh), 1.1 * max(caps + rh)]
        ax.plot(lim, lim, "k--", lw=1, label="R_hub = cap")
        ax.set_xlabel("buildable cap [mm]")
        ax.set_ylabel("R_hub requested [mm]")
        ax.set_title(f"{sw['n_binding']} of {sw['n_designs']} designs are over their cap\n"
                     f"(red = the cap binds)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    ax = axes[1]
    if sw:
        ax.scatter([r["arrival_hub_deg"] for r in sw["rows"]],
                   [r["cap_mm"] for r in sw["rows"]], s=42, c="tab:blue", zorder=3)
        ax.set_xlabel("hub arrival angle [deg]")
        ax.set_ylabel("buildable cap [mm]")
        ax.set_title("the cap is a function of the GENOME\n"
                     f"(spans {sw['cap_min_mm']:.3f}–{sw['cap_max_mm']:.3f} mm)")
        ax.grid(alpha=0.3)

    ax = axes[2]
    ol = rep.get("occ_limit")
    if ol and ol["rows"]:
        names = [r["design"] for r in ol["rows"] if "thresholds_mm" in r]
        x = np.arange(len(names))
        ax.bar(x - 0.2, [r["cap_mm"] for r in ol["rows"] if "thresholds_mm" in r],
               0.4, label="analytic cap", color="tab:blue")
        ax.bar(x + 0.2, [r["worst_corner_mm"] for r in ol["rows"] if "thresholds_mm" in r],
               0.4, label="OCC worst corner (the gate)", color="tab:orange")
        ax.plot(x, [r["square_on_mm"] for r in ol["rows"] if "thresholds_mm" in r],
                "kv", ms=7, label="square-on family")
        ax.plot(x, [r["near_cusp_mm"] for r in ol["rows"] if "thresholds_mm" in r],
                "k^", ms=7, label="near-cusp family")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("radius [mm]")
        ax.set_title("the cap against what OCC takes")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("PLAN.md §0(a) — the buildable hub fillet cap", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


SECTIONS = ("void", "occ_limit", "t0_sweep", "sweep")


def main():
    ap = argparse.ArgumentParser(description="the hub-fillet cap against OCC")
    ap.add_argument("--out", default="study_hub_cap.json")
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--designs", default="best_solution,elite14,elite13",
                    help="which genomes the CAD sections measure; `sweep` always "
                         "reports every design on disk")
    # ADDITIVE, and that is load-bearing.  `studies/study_hub_cap.json` is the calibration
    # evidence behind HUB_CAP_THICKNESS_SHARE (PLAN.md's Artifacts section: the `occ_limit`
    # and `t0_sweep` blocks "should stay reproducible"), so this driver at its DEFAULTS has
    # to keep producing that file.  Both flags below default to exactly what was committed;
    # naming either one is what asks for something new.  Same shape as
    # `study_svk_rescore.py --extra`, for the same reason.
    ap.add_argument("--extra", action="append", default=[], metavar="LABEL=PATH",
                    help="additional genome files to measure, e.g. "
                         "--extra bc77614=stage3_svk_best_medium.json.  Repeatable.  "
                         "Added to --designs' selection and to the `sweep` table.")
    ap.add_argument("--t0-sweep", default=",".join(f"{v:g}" for v in T0_SWEEP),
                    help="thickness stations for the t0_sweep section.  THE DEFAULT IS "
                         "THE COMMITTED ONE and starts at 2.0; HUB_CAP_THICKNESS_SHARE is "
                         "fitted on [2.0, 2.6] and the shipped floor is 1.2, so pass the "
                         "lower stations explicitly to measure below the fit.")
    args = ap.parse_args()

    want = [s.strip() for s in args.sections.split(",") if s.strip()]
    unknown = [s for s in want if s not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown section(s) {unknown}; expected any of {list(SECTIONS)}")

    t0 = time.time()
    all_designs = _designs()
    extra = _extra_designs(args.extra)
    all_designs = all_designs + extra
    by_name = dict(all_designs)
    picked = [n.strip() for n in args.designs.split(",") if n.strip()]
    # An --extra genome is measured because it was named; asking for it twice in --designs
    # is not an error but must not double the CAD work.
    picked += [n for n, _g in extra if n not in picked]
    missing = [n for n in picked if n not in by_name]
    if missing:
        raise SystemExit(f"no such design(s) {missing}; have {sorted(by_name)}")
    selected = [(n, by_name[n]) for n in picked]

    try:
        t0_stations = tuple(float(v) for v in args.t0_sweep.split(",") if v.strip())
    except ValueError:
        raise SystemExit(f"--t0-sweep wants a comma-separated number list, "
                         f"got {args.t0_sweep!r}")

    rep = {"settings": {"sections": want, "designs": picked,
                        "ring_samples": RING_SAMPLES, "bisect_rel": BISECT_REL,
                        "config": "coarse", "elapsed_s": None,
                        # The stations and the extras go in the RECORD, not just in argv.
                        # A calibration artifact that does not say which thicknesses it
                        # measured is the misattribution risk PLAN.md flags for
                        # MIN_WALL_MM, and this file IS the calibration.
                        "t0_stations": list(t0_stations),
                        "extra": {n: p for n, p in
                                  (e.split("=", 1) for e in args.extra)}}}

    cad_rows = []
    needs_cad = [s for s in want if s in ("void", "occ_limit")]
    if needs_cad or "t0_sweep" in want:
        if not os.path.exists(CAD_PY):
            raise SystemExit(f"no CAD env at {CAD_PY} — run `make env-cad`")
    if needs_cad:
        caps = {n: analytic_cap(g)[1] for n, g in selected}
        cad_rows = _run_cad(selected, caps, bisect="occ_limit" in want)
    if "void" in want:
        rep["void"] = run_void(selected, cad_rows)
    if "occ_limit" in want:
        rep["occ_limit"] = run_occ_limit(selected, cad_rows)
    if "t0_sweep" in want:
        sw_rows = t0_sweep_designs(stations=t0_stations)
        sw_caps = {n: analytic_cap(g)[1] for n, g in sw_rows}
        rep["t0_sweep"] = run_t0_sweep(sw_rows, _run_cad(sw_rows, sw_caps, bisect=True),
                                       stations=t0_stations)
    if "sweep" in want:
        rep["sweep"] = run_sweep(all_designs)

    rep["settings"]["elapsed_s"] = round(time.time() - t0, 1)
    rep["pass"] = bool(all(rep[s]["pass"] for s in want))

    print(f"\nhub fillet cap: {'PASS' if rep['pass'] else 'FAIL'}"
          f"   ({rep['settings']['elapsed_s']} s)")
    if "void" in want:
        print(f"  void      {'PASS' if rep['void']['pass'] else 'FAIL'}"
              f"   (gate {GATE_VOID_DEG:.3f} deg)")
        for r in rep["void"]["rows"]:
            if "error" in r:
                print(f"    {r['design']:<16} ERROR {r['error']}")
            else:
                print(f"    {r['design']:<16} analytic {r['void_analytic_deg']:7.4f}  "
                      f"OCC {r['void_occ_mean_deg']:7.4f}  "
                      f"delta {r['delta_deg']:6.4f}  runs {r['n_runs']}")
    if "occ_limit" in want:
        print(f"  occ_limit {'PASS' if rep['occ_limit']['pass'] else 'FAIL'}"
              f"   (cap/OCC must be in "
              f"[{GATE_CAP_FLOOR_FRAC:.2f}, {GATE_CAP_OVERPROMISE:.2f}] — one-sided:"
              f" never promise more than the part gives)")
        for r in rep["occ_limit"]["rows"]:
            if "thresholds_mm" not in r:
                print(f"    {r['design']:<16} ERROR {r.get('error')}")
            else:
                flag = ("OVER-PROMISES" if r["over_promises"] else
                        "vacuously small" if r["vacuously_small"] else "safe")
                fam = (f"[square-on {r['square_on_mm']:.4f}  "
                       f"near-cusp {r['near_cusp_mm']:.4f}  "
                       f"binds at wedge {r['binds_at_wedge_deg']:.0f}]"
                       if r.get("square_on_mm") is not None else "")
                print(f"    {r['design']:<16} cap {r['cap_mm']:.4f}  "
                      f"OCC worst {r['worst_corner_mm']:.4f}  "
                      f"cap/OCC {r['conservatism']:.3f}  {flag:<15}{fam}")
        print("    The gate reads the WORST of the twenty-four corners, because that is the")
        print("    one the exporter's ladder stops at.  Which FAMILY that is differs by")
        print("    design — square-on at t0 = 1.2, near-cusp on the t0 = 2.55 elites — so it")
        print("    is named by wedge angle rather than by rank.  BUILD_PLAN.md step 4.")
    if "t0_sweep" in want:
        s = rep["t0_sweep"]
        print(f"  t0_sweep  (calibration, not gated)   "
              f"square-on share of t0 {s['share_of_t0_min']:.4f}–{s['share_of_t0_max']:.4f}"
              f"   model claims {s['model_share_of_t0']:.4f} at arrival "
              f"{s['arrival_hub_deg']:.2f} deg"
              if s["share_of_t0_min"] is not None else "  t0_sweep  no usable rows")
        print(f"    {'design':<10} {'t0':>6} {'void':>7} {'slot':>8} {'thick':>8} "
              f"{'cap':>8} {'OCC sq':>8} {'/t0':>7} {'binds':>10}")
        for r in s["rows"]:
            if "square_mean_mm" not in r:
                print(f"    {r['design']:<10} ERROR {r.get('error')}")
                continue
            note = ("  [AT BRACKET]" if r.get("at_bracket")
                    else "" if r.get("same_feature")
                    else "  [void closed — NOT the same junction, excluded from the fit]")
            print(f"    {r['design']:<10} {r['t0_mm']:6.2f} {r['void_deg']:7.3f} "
                  f"{r['slot_cap_mm']:8.4f} {r['thickness_cap_mm']:8.4f} "
                  f"{r['cap_mm']:8.4f} {r['square_on_mm']:8.4f} "
                  f"{r['share_of_t0']:7.4f} {r['binds']:>10}" + note)
    if "sweep" in want:
        s = rep["sweep"]
        print(f"  sweep     {s['n_binding']} of {s['n_designs']} designs over their cap; "
              f"caps span {s['cap_min_mm']:.4f}–{s['cap_max_mm']:.4f} mm")

    path = os.path.join(HERE, args.out)
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=1)
    jpg = os.path.splitext(path)[0] + ".jpg"
    try:
        plot(rep, jpg)
        print(f"wrote {path} and {jpg}")
    except Exception as exc:                      # a missing backend is not a gate failure
        print(f"wrote {path}  (plot skipped: {type(exc).__name__}: {exc})")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
