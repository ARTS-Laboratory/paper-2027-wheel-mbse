"""
=============================================================================
  KINEMATICS_PLAN.md STEP 1 — DOES `linear` RANK DESIGNS THE WAY SVK DOES?
=============================================================================
    .venv-opt/bin/python studies/study_kinematics_rank.py     (make kinrank)

THE ONE QUESTION.  `linear` is `wheel_fem`'s default strain measure and the correction at
service load is 22.75% (KINEMATICS_PLAN Step 0a, reproduced from §14/§31).  A 22.75%
absolute offset does not by itself condemn a SEARCH model: an optimizer does not need the
right deflection, it needs the right ORDERING and the right DESCENT DIRECTION.  So the
criterion this arc registered, BEFORE any of this ran (KINEMATICS_PLAN Step 0c), is:

    R1  ARGMIN IDENTITY, binary and primary.  The lowest-linear-loss genome must be the
        lowest-SVK-loss genome, over the pool AND over its feasible subset.
    R2  Spearman rho >= 0.90 on the FEASIBLE subset (binding) and reported on the full
        pool (diagnostic).  Top-5 sets equal under both orderings — a clause that
        ABSTAINS below n = 6, where the slice is the whole subset and it cannot fail
        (PLAN.md §130 §3, fixed at §132; `_rank_block`).  Each block carries `r2_binds`
        saying whether ITS `r2_pass` is the one the verdict took, because both blocks
        compute the field and only one is read (§132 successor 0; `R2_BINDING_SUBSET`).
    R3  cos(grad_linear, grad_svk) >= 0.90 in the NORMALIZED gene space the descent steps
        in, at every probed genome.

R2's BINDING CASE IS THE FEASIBLE SUBSET, AND THAT IS NOT A CONVENIENCE.  `wheel_objective`
adds `soft_barrier(...)` terms that are 0.0 when satisfied and large when not, and a barrier
breach is overwhelmingly a GEOMETRY fact — `x_order`, `hub_overlap`, `fold`, `arrival`,
`fillet`, `fillet_cap` never touch the FEA at all.  Both kinematics therefore see almost the
same number for an infeasible design, and a pool dominated by infeasible genomes would return
rho ~ 1.0 while telling you nothing about the designs an optimizer actually chooses between.
The full-pool figure is reported so that this effect is visible rather than assumed.

WHY NOT JUST EXTEND `study_svk_rescore.py`.  That driver answers a different question — IS
THE SHIPPED GENOME FEASIBLE UNDER SVK — and its `--extra` hook exists so that the bare
`make svk` keeps reproducing SVK_PLAN Step 3's recorded artifact unchanged.  Widening its
GENOMES tuple to 36 rows would break that contract.  Its `_score` IS reused here, imported
rather than copied, so the term-set guard, the p=4 probe identity and the feasibility rule
are the same construction in both files and cannot drift apart.

MESHES ARE BUILT ONCE PER GENOME AND SHARED BY BOTH KINEMATICS, and `flank_orientation` is
pinned per genome, for `study_svk_rescore.py`'s reasons: a strain measure is a property of
the kernel, not the geometry, and letting a discrete branch be re-derived per column would
put a topology change inside a strain-measure comparison.

THE POOL IS EVERY DISTINCT COMMITTED GENOME IN THE TREE PLUS THE 15 STAGE-2 ELITES THAT ARE
NOT ALREADY ONE.  De-duplicated by gene vector, not by filename: `best_solution.json` and
`stage3_knee_best_medium.json` are the same wheel, as are `stage3_margin_best_medium.json`
and `stage3_margin_promote_best.json`, and `stage2_elites.json` rank 0 IS
`best_solution_ga_beam.json`.  A duplicate row would put a tied pair into a rank statistic
for a reason that is not about the wheel.

ONE OF THOSE THREE ALIASES IS NO LONGER TRUE, AND IT COSTS THE POOL A GENOME — PLAN.md
§129.  §115 promoted `b729e86` into `best_solution.json` and PRESERVED the outgoing
`09e8188` as `stage3_knee_best_medium.json` precisely so it would not be lost, which
turned the two files from one wheel into two.  `COMMITTED` below never listed the knee
file — it did not need to while the alias held — so the outgoing shipped genome is now
the one design in the tree this driver cannot see.  Measured 2026-09-07 over every
tracked file carrying a genome: 39 distinct gene vectors on disk, 36 reachable here, and
the three that are not are `stage3_knee_best_medium.json`, `defect5_step100.json` and
`fillet_optimum_b029622.json`.  The other two aliases still hold, checked the same day.
Whether the missing three belong in the pool is §116's successor 3 — a decision about
what "the shipped genome and its rivals" means, not a fact this docstring can settle —
so the list is left as it is and the gap is named instead.
=============================================================================
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
from scipy import stats

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import wheel_fea as W
import wheel_genome as wg
import wheel_objective as WO
import wheel_pool as WP
import wheel_wheel as WW

import study_svk_rescore as SR

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = "coarse"
N_PHASE = 8

# REGISTERED IN KINEMATICS_PLAN.md STEP 0c BEFORE THIS FILE WAS RUN.  Not to be moved to
# fit the run: R1 is deliberately binary so there is nothing in it to loosen.
GATE_SPEARMAN = 0.90
GATE_GRAD_COSINE = 0.90

# WHICH SUBSET DECIDES R2 — one literal, three readers (§132 successor 0).  `_verdict`
# consumes only this block's `r2_pass`, but `_rank_block` computes the field for BOTH
# subsets, so an artifact can carry `blocks.full.r2_pass: false` beside
# `registered_criterion.R2_rank_agreement: true` — both correct, and grep-readable as a
# contradiction.  `study_kinematics_rank_filleted.json` does exactly that today.  The
# terminal output has always disambiguated it ("BINDING for R2" against "diagnostic") and
# the ARTIFACT never did; `r2_binds` below puts the same sentence in the file, next to the
# field that provokes the question rather than in a header a grep will not see.
#
# WHY THE MARKER RATHER THAN DROPPING THE VERB, which was the other option filed at §132.
# Removing `r2_pass` from the diagnostic block would subtract a field from
# `study_kinematics_rank.json`, and that file is §32's evidence — §130's header note is
# explicit that overwriting it falsifies a closed arc.  An additive marker leaves every
# recorded number in place.  The full block is not purely diagnostic either: R1 reads its
# `argmin_identical`, so "the full pool does not bind" would itself be false.  It is R2
# specifically that this subset does not decide, and that is what the name says.
R2_BINDING_SUBSET = "feasible"

# Every distinct genome the tree commits to, in the order they were produced.  Files that
# hold the SAME gene vector are folded together by `_pool` and reported as aliases.
COMMITTED = (
    ("36aed36 ga_beam",      "best_solution_ga_beam.json"),
    ("elite9 prod",          "stage3_prod_best_elite9.json"),
    ("elite10 prod",         "stage3_prod_best_elite10.json"),
    ("minwall 0.8",          "stage3_minwall_best_0.8.json"),
    ("minwall 1.0",          "stage3_minwall_best_1.0.json"),
    ("350f4c7 minwall1.2",   "stage3_minwall_best_1.2.json"),
    ("minwall 1.4",          "stage3_minwall_best_1.4.json"),
    ("minwall 1.6",          "stage3_minwall_best_1.6.json"),
    ("minwall 1.8",          "stage3_minwall_best_1.8.json"),
    ("minwall 2.0",          "stage3_minwall_best_2.0.json"),
    ("minwall 2.2",          "stage3_minwall_best_2.2.json"),
    ("svk-shipped",          "stage3_svk_best_shipped.json"),
    ("svk-elite10",          "stage3_svk_best_elite10.json"),
    ("bc77614 svk-medium",   "stage3_svk_best_medium.json"),
    ("e4219f3 buildcap",     "stage3_buildcap_best_medium.json"),
    ("buildcap2",            "stage3_buildcap2_best_medium.json"),
    ("margin probe",         "stage3_margin_probe_best.json"),
    ("e126cc3 margin",       "stage3_margin_best_medium.json"),
    ("promote check",        "stage3_promote_best.json"),
    ("promote2 check",       "stage3_promote2_best.json"),
    # `b729e86` since §115 (2026-09-06), not `09e8188` — the label is the only thing that
    # was ever hardcoded here, the genes are read live, and `best_solution.json`'s own
    # note flagged this label as outstanding at the promotion.  Corrected at PLAN.md §129
    # rather than left cosmetic: since the promotion the label names a DIFFERENT genome
    # that is still on disk under another name, so a stale label here and a live read
    # below no longer disagree about a caption — they disagree about which wheel the row
    # is.  See the pool note in the header.
    ("b729e86 SHIPPED",      "best_solution.json"),
)

# The genomes R3 probes.  Four, not all 36: a value+grad call is the expensive one and the
# question R3 asks is about the DESCENT, so the points that matter are the ones a descent
# actually sat on — the shipped genome, the incumbent it replaced, the design the linear
# ranking prefers, and the GA/beam control whose correction is 5.5x smaller.
#
# THAT LINEAGE IS ONE PROMOTION OUT OF DATE — PLAN.md §129.  Row 1 is `b729e86` now, and
# "the incumbent it replaced" is `09e8188` (`stage3_knee_best_medium.json`), not
# `e126cc3`, which is two promotions back.  The four ROWS are unchanged: they are still
# four points a descent sat on, which is the property the list was chosen for, and
# re-choosing them by today's lineage is a decision about the probe set rather than a
# correction to it.  What is fixed is the sentence claiming they are something they are
# not.
GRAD_PROBES = (
    ("b729e86 SHIPPED",    "best_solution.json"),
    ("e126cc3 margin",     "stage3_margin_best_medium.json"),
    ("350f4c7 minwall1.2", "stage3_minwall_best_1.2.json"),
    ("36aed36 ga_beam",    "best_solution_ga_beam.json"),
)


def _genes(d):
    """BY NAME, not by dict order.

    `study_svk_rescore.load_genes` takes `.values()`, which is correct only while every
    artifact happens to serialise its genes in `GENE_NAMES` order.  This pool reaches into
    `stage2_elites.json` as well, so the ordering is validated rather than assumed —
    `genes_to_vector` raises on a missing or extra key.
    """
    return np.asarray(wg.genes_to_vector(d), dtype=float)


def _genes_from_file(path):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return _genes(json.load(fh)["genes"])


def _key(v):
    return hashlib.sha1(repr(tuple(round(float(x), 12) for x in v)).encode()).hexdigest()[:7]


def _pool(include_elites=True):
    """`[(label, genes, aliases)]`, de-duplicated by GENE VECTOR.

    A tie in the input is not a tie in the wheel, and a rank statistic cannot tell the
    difference.  Aliases are kept so the report says which files collapsed.
    """
    rows, index = [], {}
    for label, path in COMMITTED:
        full = os.path.join(PP.ROOT, path)
        if not os.path.exists(full):
            print(f"  MISSING {path}", flush=True)
            continue
        v = _genes_from_file(path)
        k = _key(v)
        if k in index:
            rows[index[k]][2].append(f"{label} ({path})")
            continue
        index[k] = len(rows)
        rows.append([f"{label}", v, [path]])
    if include_elites:
        with open(os.path.join(PP.ROOT, "stage2_elites.json")) as fh:
            for el in json.load(fh)["elites"]:
                v = _genes(el["genes"])
                k = _key(v)
                if k in index:
                    rows[index[k]][2].append(f"stage2 elite {el['rank']}")
                    continue
                index[k] = len(rows)
                rows.append([f"elite{el['rank']} {el['genome_hash']}", v, ["stage2_elites.json"]])
    return [(a, b, c) for a, b, c in rows]


# ---------------------------------------------------------------------------
# R1 / R2 — the ranking
# ---------------------------------------------------------------------------

def run_rank(pool, cfg=DEFAULT_CONFIG, n_phase=N_PHASE, workers=0):
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    p = WP.PhasePool(workers) if workers else None
    rows = []
    try:
        for i, (label, genes, aliases) in enumerate(pool):
            t0 = time.time()
            wanted = phases[:1] if p is not None else phases
            row = {"genome": label, "aliases": aliases, "gene_key": _key(genes)}
            # THE MESH BUILD IS INSIDE THE HANDLER, AND IT WAS NOT — PLAN.md §129.
            # `_score`'s own refusals have been recorded since this file was written; the
            # build above them was not, so a genome with NO filleted mesh took the whole
            # run down and the report is written only after `run_rank` RETURNS — an hour
            # of scoring and nothing on disk.  That is not hypothetical: measured
            # 2026-09-07 at this driver's own `coarse` default, `elite11` (`fc7aeb1`)
            # raises `MeshRefusedError` at the rim, and it is row 32 of 36.
            try:
                orientation = tuple(float(o) for o in
                                    WW.flank_orientation(genes, WW.get_config(cfg)))
                meshes = WO.phase_meshes(genes, cfg, wanted, orientation=orientation)
            except Exception as exc:                           # noqa: BLE001
                row["mesh_s"] = round(time.time() - t0, 1)
                why = f"{type(exc).__name__}: {exc}"
                for kin in ("linear", "svk"):
                    row[kin] = {"failed": why}
                print(f"  [{i + 1}/{len(pool)}] {label:<22} MESH   FAILED {exc}",
                      flush=True)
                rows.append(row)
                continue
            row["mesh_s"] = round(time.time() - t0, 1)
            for kin in ("linear", "svk"):
                t1 = time.time()
                # A genome that will not solve is a finding, not a reason to lose the run:
                # 36 genomes is over an hour and one NewtonDivergedError must not take the
                # other 35 with it.  Recorded and excluded from the statistics, loudly.
                try:
                    row[kin] = SR._score(genes, cfg, phases, meshes, kin,
                                         pool=p, orientation=orientation)
                    row[kin]["elapsed_s"] = round(time.time() - t1, 1)
                except Exception as exc:                       # noqa: BLE001
                    row[kin] = {"failed": f"{type(exc).__name__}: {exc}"}
                    print(f"  [{i + 1}/{len(pool)}] {label:<22} {kin:<6} FAILED {exc}",
                          flush=True)
                    continue
                s = row[kin]
                print(f"  [{i + 1}/{len(pool)}] {label:<22} {kin:<6} "
                      f"loss {s['loss']:12.4f} drop {s['axle_drop_mean_mm']:7.4f} "
                      f"util {s['stress_utilisation']:6.3f} "
                      f"{'FEAS' if s['feasible'] else 'infeas'} "
                      f"({s['elapsed_s']} s)", flush=True)
            if all("failed" not in row[k] for k in ("linear", "svk")):
                row["loss_rel_diff"] = float(row["svk"]["loss"] / row["linear"]["loss"] - 1.0)
                row["drop_rel_diff"] = float(row["svk"]["axle_drop_mean_mm"]
                                             / row["linear"]["axle_drop_mean_mm"] - 1.0)
            rows.append(row)
    finally:
        if p is not None:
            p.close()
    return {"config": cfg, "n_phase": n_phase, "scheme": "uniform", "workers": workers,
            "rows": rows, "verdict": _verdict(rows)}


def _subsets(rows):
    """`{name: [row]}` — the two pools R2 is evaluated on.

    `feasible` is FEASIBLE UNDER BOTH kinematics.  Taking either one alone would let the
    subset itself be chosen by the thing under test.
    """
    ok = [r for r in rows if "failed" not in r["linear"] and "failed" not in r["svk"]]
    return {"full": ok,
            "feasible": [r for r in ok
                         if r["linear"]["feasible"] and r["svk"]["feasible"]]}


def _rank_block(rows, name):
    if len(rows) < 3:
        return {"subset": name, "n": len(rows), "insufficient": True,
                "r2_binds": name == R2_BINDING_SUBSET}
    lin = np.array([r["linear"]["loss"] for r in rows])
    svk = np.array([r["svk"]["loss"] for r in rows])
    rho = stats.spearmanr(lin, svk)
    tau = stats.kendalltau(lin, svk)
    order_l = [rows[i]["genome"] for i in np.argsort(lin)]
    order_s = [rows[i]["genome"] for i in np.argsort(svk)]
    k = min(5, len(rows))
    top_l, top_s = order_l[:k], order_s[:k]
    # `None`, not `True`, WHEN THE SLICE IS THE WHOLE SUBSET.  `k = min(5, n)`, so at
    # n <= 5 both slices hold every scored genome and the two sets are equal BY
    # CONSTRUCTION — for any pair of orderings whatsoever, including exactly reversed
    # ones.  Reporting `True` there made R2's second clause read as a condition that had
    # been checked and had passed when nothing had been checked at all, and it did:
    # PLAN.md §130 cleared R2 on a binding subset of five that way.  `r3_pass` below
    # already takes this shape for the same reason — an artifact has to keep "measured
    # and agreed" distinguishable from "not measurable here".  Below n = 6 the registered
    # R2 therefore degrades to the bare Spearman, which is what it has always been; the
    # change is that it now says so.  §132 and KINEMATICS_PLAN step 0c carry the note.
    sets_equal = None if k == len(rows) else bool(set(top_l) == set(top_s))
    # THE INVERSION COUNT is over unordered pairs and is the raw form of the same fact rho
    # summarises — reported because "rho = 0.94" and "9 of 120 pairs are the wrong way
    # round" land very differently on a reader deciding whether to trust a search model.
    n_inv = int(sum(1 for i in range(len(rows)) for j in range(i + 1, len(rows))
                    if (lin[i] - lin[j]) * (svk[i] - svk[j]) < 0.0))
    n_pairs = len(rows) * (len(rows) - 1) // 2
    return {
        "subset": name, "n": len(rows),
        "spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue),
        "kendall_tau": float(tau.statistic),
        "order_linear": order_l, "order_svk": order_s,
        "top5_linear": top_l, "top5_svk": top_s,
        "top5_sets_equal": sets_equal,
        "argmin_linear": order_l[0], "argmin_svk": order_s[0],
        "argmin_identical": bool(order_l[0] == order_s[0]),
        "top2_inverted": bool(len(rows) >= 2 and set(order_l[:2]) == set(order_s[:2])
                              and order_l[:2] != order_s[:2]),
        "discordant_pairs": n_inv, "n_pairs": n_pairs,
        "discordant_fraction": float(n_inv / n_pairs) if n_pairs else 0.0,
        # `is not False`: a vacuous clause cannot fail the gate and must not silently
        # pass it either, so it abstains and rho carries R2 alone.
        "r2_pass": bool(float(rho.statistic) >= GATE_SPEARMAN
                        and sets_equal is not False),
        # Whether `_verdict` READS the `r2_pass` directly above it.  False here does not
        # mean the subset was not measured — every statistic in this block is real; it
        # means R2's registered verdict was not taken from it.
        "r2_binds": name == R2_BINDING_SUBSET,
    }


def _verdict(rows):
    blocks = {name: _rank_block(rs, name) for name, rs in _subsets(rows).items()}
    binding = blocks[R2_BINDING_SUBSET]
    full = blocks["full"]
    r1 = None
    if not full.get("insufficient"):
        r1 = full["argmin_identical"]
        if not binding.get("insufficient"):
            r1 = r1 and binding["argmin_identical"]
    return {"blocks": blocks, "gate_spearman": GATE_SPEARMAN,
            "r1_argmin_identity": r1,
            "r2_rank_agreement": (None if binding.get("insufficient")
                                  else binding["r2_pass"])}


# ---------------------------------------------------------------------------
# R3 — the descent direction
# ---------------------------------------------------------------------------

def run_gradients(cfg=DEFAULT_CONFIG, n_phase=N_PHASE, workers=0, probes=GRAD_PROBES):
    """cos(grad_linear, grad_svk) IN NORMALIZED GENE SPACE, which is where the descent lives.

    `wheel_stage3` steps in the unit box (`objective(..., normalized=True)`), and the chain
    rule that gets there multiplies each component by that gene's range — `cy` spans 64 mm
    and `R_rim` spans 2.5.  A cosine taken in PHYSICAL units would be dominated by whichever
    genes happen to have wide boxes and would not be the angle any optimizer ever sees.

    The two calls SHARE MESHES and a pinned `flank_orientation`, for `run_rank`'s reasons.
    """
    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    p = WP.PhasePool(workers) if workers else None
    rows = []
    try:
        for label, path in probes:
            full = os.path.join(PP.ROOT, path)
            if not os.path.exists(full):
                continue
            genes = _genes_from_file(path)
            z = (genes - low) / (high - low)
            # SAME HANDLER AS `run_rank`, AND R3 HAD NONE AT ALL — PLAN.md §129.  Both
            # calls below reach `mesh_coords`, which refuses a mesh the sector-fit clamp
            # moved off its genes (`clamp_reject`, §108/§110), and `36aed36` is probe 4 of
            # 4 and does exactly that on the filleted mesh — §116.3 at `medium`, confirmed
            # 2026-09-07 at this driver's `coarse`.  Unhandled, that killed the run AFTER
            # `run_rank`'s report was written and BEFORE `registered_criterion` ever was,
            # so `make kinrank` could not produce a verdict at all.  A refused probe is
            # recorded and excluded from `r3_pass`, never counted as a cosine of zero:
            # R3 asks how far apart two gradients point, and "there is no gradient here"
            # is a different answer from "they point 90 degrees apart".
            try:
                orientation = tuple(float(o) for o in
                                    WW.flank_orientation(genes, WW.get_config(cfg)))
                wanted = phases[:1] if p is not None else phases
                meshes = WO.phase_meshes(genes, cfg, wanted, orientation=orientation)
                g = {}
                for kin in ("linear", "svk"):
                    t0 = time.time()
                    val, grad, _ = WO.objective(z, cfg, normalized=True, phases=phases,
                                                meshes=meshes, pool=p,
                                                orientation=orientation, kinematics=kin)
                    g[kin] = {"loss": float(val), "grad": np.asarray(grad, dtype=float),
                              "elapsed_s": round(time.time() - t0, 1)}
            except Exception as exc:                           # noqa: BLE001
                rows.append({"genome": label, "file": path,
                             "failed": f"{type(exc).__name__}: {exc}"})
                print(f"  grad {label:<22} FAILED {exc}", flush=True)
                continue
            a, b = g["linear"]["grad"], g["svk"]["grad"]
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")
            row = {
                "genome": label, "file": path,
                "loss_linear": g["linear"]["loss"], "loss_svk": g["svk"]["loss"],
                "grad_norm_linear": na, "grad_norm_svk": nb,
                "grad_norm_ratio": float(nb / na) if na > 0 else float("nan"),
                "cosine": cos,
                "angle_deg": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
                # A DIRECTION IS ONLY USEFUL IF IT DESCENDS THE OTHER LOSS.  cos > 0 is the
                # weak form of the question and it is reported separately from the 0.90 bar,
                # because they fail for different reasons and mean different things: cos > 0
                # says a linear step still reduces the SVK loss for SOME step size; cos >=
                # 0.90 says the two optimizers would walk the same way.
                "linear_step_descends_svk": bool(cos > 0.0),
                # Per-gene sign flips: which genes the two models want moved OPPOSITE ways.
                "sign_flips": [wg.GENE_NAMES[i] for i in range(len(a))
                               if a[i] * b[i] < 0.0],
                "elapsed_s": {k: g[k]["elapsed_s"] for k in g},
                "r3_pass": bool(cos >= GATE_GRAD_COSINE),
            }
            rows.append(row)
            print(f"  grad {label:<22} cos {cos:+.4f} ({row['angle_deg']:5.1f} deg)  "
                  f"|g| {na:9.3f} -> {nb:9.3f}  flips {len(row['sign_flips'])}",
                  flush=True)
    finally:
        if p is not None:
            p.close()
    ok = [r for r in rows if "failed" not in r]
    return {"config": cfg, "n_phase": n_phase, "workers": workers,
            "gate_cosine": GATE_GRAD_COSINE, "rows": rows,
            "n_probes": len(rows), "n_refused": len(rows) - len(ok),
            # `None`, not `False`, when nothing differentiated: `registered_criterion`
            # already reads a `None` as not-passing, and the two states have to stay
            # distinguishable in the artifact — "every probe disagreed" and "no probe
            # could be taken" are different findings about the same gate.
            "r3_pass": (bool(all(r["r3_pass"] for r in ok)) if ok else None)}


# ---------------------------------------------------------------------------

def _print(rep):
    rk = rep.get("rank")
    if rk:
        print("\n" + "=" * 100)
        print(f"  STEP 1 — LINEAR vs SVK RANKING, {rk['config']}, {rk['n_phase']} phases")
        print("=" * 100)
        print(f"  {'genome':<22} {'lin loss':>12} {'svk loss':>12} {'d loss':>9} "
              f"{'lin drop':>9} {'svk drop':>9} {'lin':>6} {'svk':>6}")
        for r in rk["rows"]:
            if "failed" in r["linear"] or "failed" in r["svk"]:
                print(f"  {r['genome']:<22} FAILED  "
                      f"{r['linear'].get('failed', '')} {r['svk'].get('failed', '')}")
                continue
            print(f"  {r['genome']:<22} {r['linear']['loss']:12.4f} "
                  f"{r['svk']['loss']:12.4f} {100 * r['loss_rel_diff']:8.1f}% "
                  f"{r['linear']['axle_drop_mean_mm']:9.4f} "
                  f"{r['svk']['axle_drop_mean_mm']:9.4f} "
                  f"{'FEAS' if r['linear']['feasible'] else 'infs':>6} "
                  f"{'FEAS' if r['svk']['feasible'] else 'infs':>6}")

        for name in ("full", "feasible"):
            b = rk["verdict"]["blocks"][name]
            print(f"\n  --- {name.upper()} POOL "
                  f"({'BINDING for R2' if name == R2_BINDING_SUBSET else 'diagnostic'}"
                  f") ---")
            if b.get("insufficient"):
                print(f"      only {b['n']} rows — no rank statistic")
                continue
            print(f"      n = {b['n']}   Spearman rho = {b['spearman_rho']:+.4f} "
                  f"(gate {GATE_SPEARMAN})   Kendall tau = {b['kendall_tau']:+.4f}")
            print(f"      discordant pairs {b['discordant_pairs']}/{b['n_pairs']} "
                  f"({100 * b['discordant_fraction']:.1f}%)")
            print(f"      top5 linear : {b['top5_linear']}")
            print(f"      top5 svk    : {b['top5_svk']}")
            eq = b['top5_sets_equal']
            print(f"      top5 sets equal: "
                  f"{'n/a — the slice IS the subset' if eq is None else eq}"
                  f"   argmin identical: {b['argmin_identical']}"
                  f"   ({b['argmin_linear']} vs {b['argmin_svk']})")

    gr = rep.get("gradients")
    if gr:
        print("\n" + "=" * 100)
        print(f"  R3 — DESCENT DIRECTION, normalized gene space, {gr['config']}")
        print("=" * 100)
        print(f"  {'genome':<22} {'cos':>9} {'angle':>8} {'|g| lin':>11} {'|g| svk':>11} "
              f"{'ratio':>7}  sign flips")
        for r in gr["rows"]:
            if "failed" in r:                       # the refusal handler's row, §129
                print(f"  {r['genome']:<22} FAILED  {r['failed']}")
                continue
            print(f"  {r['genome']:<22} {r['cosine']:+9.4f} {r['angle_deg']:7.1f}d "
                  f"{r['grad_norm_linear']:11.3f} {r['grad_norm_svk']:11.3f} "
                  f"{r['grad_norm_ratio']:7.2f}  "
                  f"{','.join(r['sign_flips']) if r['sign_flips'] else '-'}")

    v = rep["registered_criterion"]
    print("\n" + "=" * 100)
    print("  THE REGISTERED CRITERION (KINEMATICS_PLAN.md Step 0c, written before this ran)")
    print("=" * 100)
    for k in ("R1_argmin_identity", "R2_rank_agreement", "R3_descent_direction"):
        got = v[k]
        print(f"    {k:<22} {'PASS' if got else ('FAIL' if got is False else 'n/a')}")
    print(f"\n  LINEAR IS AN ACCEPTABLE DEFAULT FOR SEARCH: "
          f"{'YES' if v['linear_is_acceptable'] else 'NO'}")


def main():
    ap = argparse.ArgumentParser(description="KINEMATICS_PLAN Step 1 — the ranking test")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--n-phase", type=int, default=N_PHASE)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="study_kinematics_rank_filleted.json")
    ap.add_argument("--no-elites", action="store_true",
                    help="committed genomes only; the 15 stage-2 elites are GA/beam-era "
                         "2.0 mm designs and most land infeasible under today's objective")
    ap.add_argument("--skip-rank", action="store_true")
    ap.add_argument("--skip-grad", action="store_true")
    args = ap.parse_args()

    # TWO NAMES FROM ONE PATH, AND ONE OF THEM IS CLOSED — PLAN.md §176, deciding §130
    # successor 0 in the shape §132 §5 left room for.  `study_kinematics_rank_filleted.json`
    # is the gate: the mesh `wheel_objective` has solved since §103, and a four-worker re-run
    # reproduces it bit for bit (§175 §2(c)).  `study_kinematics_rank.json` is §32's evidence
    # on the unfilleted mesh, which no run at this commit can reproduce, so EVERY run is
    # refused under that name — and the rho -0.83 `wheel_stage3`'s `--kinematics` default
    # cites exists only there.  `--workers` is a scheduling knob and is not guarded, for
    # `_gate_guard`'s `--seed` reason.  Imported here, not at the top, so no line above
    # moves: `PLAN.md` cites this file by line.
    import _gate_guard
    _gate_guard.refuse_degraded_out(ap, args, "study_kinematics_rank.json", [
        (True, "§32's closed evidence, measured on the unfilleted mesh, which no run on "
               "this tree reproduces (PLAN.md §176)"),
    ])
    _gate_guard.refuse_degraded_out(ap, args, "study_kinematics_rank_filleted.json", [
        (args.config != DEFAULT_CONFIG,
         "--config %s, not the gate's %s" % (args.config, DEFAULT_CONFIG)),
        (args.n_phase != N_PHASE,
         "--n-phase %d, not the gate's %d" % (args.n_phase, N_PHASE)),
        (args.no_elites, "--no-elites, which drops the 15 stage-2 elite rows from the pool"),
        (args.skip_rank, "--skip-rank, which drops R1 and R2"),
        (args.skip_grad, "--skip-grad, which drops R3"),
    ])

    t0 = time.time()
    rep = {}
    if not args.skip_rank:
        pool = _pool(include_elites=not args.no_elites)
        print(f"pool: {len(pool)} distinct genomes", flush=True)
        rep["rank"] = run_rank(pool, args.config, args.n_phase, args.workers)
        # WRITTEN AS SOON AS IT EXISTS, before the gradients and before `_print`:
        # `study_svk_rescore.py`'s rule, and this run is over an hour.
        with open(os.path.join(HERE, args.out), "w") as fh:
            json.dump(rep, fh, indent=1, default=float)
    if not args.skip_grad:
        rep["gradients"] = run_gradients(args.config, args.n_phase, args.workers)

    rk = rep.get("rank", {}).get("verdict", {})
    gr = rep.get("gradients", {})
    r1 = rk.get("r1_argmin_identity")
    r2 = rk.get("r2_rank_agreement")
    r3 = gr.get("r3_pass") if gr else None
    rep["registered_criterion"] = {
        "R1_argmin_identity": r1, "R2_rank_agreement": r2, "R3_descent_direction": r3,
        # R1 IS THE VERDICT.  R2 and R3 qualify it and cannot rescue it: a model that
        # selects a different design has failed at the only job a search model has.
        "linear_is_acceptable": bool(r1 and r2 and r3),
        "gate_spearman": GATE_SPEARMAN, "gate_grad_cosine": GATE_GRAD_COSINE,
    }
    rep["settings"] = {"config": args.config, "n_phase": args.n_phase,
                       "workers": args.workers, "elites": not args.no_elites,
                       "elapsed_s": round(time.time() - t0, 1)}
    out = os.path.join(HERE, args.out)
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    _print(rep)
    print(f"\nwrote {out}  ({rep['settings']['elapsed_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
