#!/usr/bin/env python
"""
=============================================================================
A BEND THAT IS A FUNCTION OF THE GENOME, FIT-FREEZE-SCORED
(PLAN.md §56 successor 2; UNCAP_PLAN.md STEP 3 RECORD, PART 4 -> PART 10)
=============================================================================

WHAT THIS ANSWERS.  PART 4 built the curved Y with one constant `bend`, jointly optimised
with the interior point `w` over `study_tri_block.sweep_bend_genomes`'s grid, and named
the gap it left: at `medium` the per-genome CEILING (any (w, bend) at all) reaches 16 of
17 genomes while the one FIXED (w, bend) rule that would actually ship reaches only 13 of
16 -- wider than the straight Y's own gap, because the curve enlarged what is reachable
without making a constant rule any better at reaching it.  `bow_over_width` was named as
"the obvious argument to fit against", with the warning that the per-genome argmax `bend`
values "are argmaxes over a plateau and their scatter is partly that" -- so this file does
NOT fit against those argmaxes.  It fits against the thing that actually matters: how many
genomes a rule leaves VALID and CLEAR, evaluated at `w` HELD FIXED to the tri-block's own
published cell (PLAN §55; the same `w` `study_tri_rule.py` reads from `sweep.best`), because
a genome-dependent `w` is a different, unasked question and this session's scope is the
bend alone.

TWO ONE-PARAMETER FAMILIES, FIT UNDER THE SAME PROTOCOL, SO THE COMPARISON IS FAIR.
Fitting a genome-dependent rule against an already-fitted historical constant (the joint
(w, bend) grid's own `best["bend"]`, chosen over a much larger search and a DIFFERENT w)
would not isolate what the genome-dependence itself buys.  So both families here are
fit on the SAME fit stream, by the SAME argmax-with-parsimony-tie-break rule, and scored
ONCE on the SAME hold-out stream:

  CONSTANT   bend(bow) = b,  b in `study_tri_block.BEND_GRID` (0.0, 0.1, ..., 1.0).
             The zero-genome-information baseline -- what "one number, no argument" can
             reach at THIS w.

  LINEAR     bend(bow) = clip(k * bow, 0, 1),  k in `K_GRID` below.  Zero at bow = 0 by
             construction, which is exactly PART 4 / test_tri_block's
             `test_the_bend_is_INERT_where_the_region_is_fat` requirement -- a genome
             whose region is fat needs no argument to keep `bend` at 0, because the family
             puts it there itself.

Selection, on the FIT stream only: maximise `n_clear` first, `n_valid` second,
`worst_min_scaled_jacobian` third -- `sweep_bend_genomes`'s own tie-break order, because a
rule chosen on `n_clear` alone and ties broken by mesh quality has a chance of
generalising and one chosen on the raw Jacobian sum does not (§48 PART 13's reasoning,
reused).  Ties beyond that go to the SMALLER parameter -- fewer things not built, or a
gentler slope -- the same parsimony bias `study_tri_rule.fit_rule` uses for fewer fires.

THE STREAMS.  Fresh disjoint draws of `study_tri_block.sweep_genomes`'s UNIFORM 16-genome
box (this is PART 4's own population, not §72's arc-span band -- PART 4's gap is measured
on the uniform box, and re-aiming to the band is a different, unasked question).  The
shipped genome is carried alongside every stream and scored under every frozen rule, never
used to fit, as a check that a genome-dependent rule which improves the box does not cost
the one genome the objective actually ships.

EXIT STATUS follows the family's convention: nonzero ONLY if a self-check fails -- stream
disjointness, an empty fit stream, or a non-finite frozen parameter.  A held-out score that
does not beat the constant is a finding and exits zero -- this file states plainly, in
`self_checks["linear_beats_constant"]`, whether it did.

WHAT THIS DOES NOT DO.  It does not wire a genome-dependent bend into `build_wheel` or
`sector_blocks` -- `bend` stays a `study_tri_block` argument nothing outside its own
studies passes -- and it does not move `w`, `best_solution.json`, or the tri-block's
published cell.  It is a measurement for the adoption decision (Step 3 of this session),
not the decision itself.
=============================================================================
"""

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import _gate_guard
import study_tri_block as tb

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIGS = ("coarse", "medium")

# Registered seeds for `sweep_genomes`' own batching (`seed + b`, up to 40 batches per the
# committed convention) -- spaced 100 apart, as `study_tri_rule.py`'s first registration
# established for exactly this sweep.
SEEDS = {
    "coarse": {"fit": 20268100, "holdout": 20268200},
    "medium": {"fit": 20268300, "holdout": 20268400},
}

# The constant family's own candidates: the tri-block's own published bend grid, so the
# baseline is not a straw man built for this file.
CONSTANT_GRID = tuple(float(b) for b in tb.BEND_GRID)

# The linear family's slope candidates.  §56 measured the largest bow the UNIFORM sampler
# draws at ~0.5 and PLAN §72 measured a CONDITIONED draw reaching 1.25 -- this file's
# streams are uniform, so a slope that saturates `bend` to 1.0 by bow ~0.03-0.06 (k = 16-30)
# covers the box this file actually draws from, with smaller slopes covering genomes that
# need only a partial bend.
K_GRID = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0)

TRIBLOCK_ARTIFACT = os.path.join(HERE, "study_tri_block.json")


# ---------------------------------------------------------------------------
# THE TWO FAMILIES
# ---------------------------------------------------------------------------

def constant_rule(b):
    return lambda bow: float(b)


def linear_rule(k):
    return lambda bow: float(min(1.0, max(0.0, k * bow)))


# ---------------------------------------------------------------------------
# EVALUATION -- one rule, one stream, at the FIXED w
# ---------------------------------------------------------------------------

def evaluate(rows, bend_of_bow, w, B, cfg_name):
    """`(n, n_valid, n_clear, worst_min_scaled_jacobian)` for one rule on one stream.

    A row whose region cannot even be rebuilt from its own genes (should not happen --
    `sweep_genomes` already filtered on the SHIPPED-blend mesh being clean -- but a study
    driver must never let one drawn genome kill the run) is dropped and counted.
    """
    n_valid = n_clear = 0
    worst = 9.0
    dropped = 0
    for r in rows:
        try:
            reg = tb.region(np.asarray(r["genes"], float), cfg_name, blend=0.0)
            c = tb.cell(reg, B, w, bend_of_bow(r["bow_over_width"]))
        except Exception:
            dropped += 1
            continue
        if c is None:
            dropped += 1
            continue
        worst = min(worst, c["min_scaled_jacobian"])
        n_valid += int(c["all_valid"])
        n_clear += int(c["min_scaled_jacobian"] > tb.MIN_SJ_TARGET)
    return {"n": len(rows) - dropped, "dropped": dropped,
            "n_valid": n_valid, "n_clear": n_clear,
            "worst_min_scaled_jacobian": float(worst)}


def _select(scored):
    """The tie-broken argmax over `(param, ev)` pairs -- pure, so it is testable without
    building a single mesh.  Maximise `n_clear`, then `n_valid`, then the worst block's
    own margin; ties go to the SMALLER parameter, the same parsimony bias
    `study_tri_rule.fit_rule` uses for fewer fires."""
    best = None
    for p, ev in scored:
        key = (-ev["n_clear"], -ev["n_valid"], -ev["worst_min_scaled_jacobian"], p)
        if best is None or key < best[0]:
            best = (key, p, ev)
    _, p, ev = best
    return {"param": p, "in_sample": ev}


def _fit(rows, candidates, rule_of, w, B, cfg_name):
    """Argmax over `candidates` on `rows`, tie-broken toward the smaller parameter."""
    return _select((p, evaluate(rows, rule_of(p), w, B, cfg_name)) for p in candidates)


# ---------------------------------------------------------------------------
# STREAMS
# ---------------------------------------------------------------------------

def gene_key(genes):
    blob = ",".join(repr(float(x)) for x in genes).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def draw_stream(cfg, seed, B, w_fixed):
    """The uniform 16-genome box `sweep_genomes` already draws for the committed cell."""
    sw = tb.sweep_genomes(cfg, B, w_fixed, seed=seed, max_batches=40)
    rows = [r for v in sw["groups"].values() for r in v if "bow_over_width" in r]
    return {"seed": seed, "rows": rows,
            "orientations_complete": (len(sw["groups"]) == 4
                                      and all(len(v) == tb.GENOME_SWEEP_PER_ORIENTATION
                                             for v in sw["groups"].values()))}


def streams_disjoint(streams):
    seen, dup = {}, None
    for s in streams:
        for r in s["rows"]:
            k = gene_key(r["genes"])
            if k in seen:
                dup = (seen[k], s["seed"], k)
                return False, dup
            seen[k] = s["seed"]
    return True, None


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------

def frozen_cells(path=TRIBLOCK_ARTIFACT):
    with open(path) as fh:
        rec = json.load(fh)
    out = {}
    for name, per in rec["per_config"].items():
        best = per.get("sweep", {}).get("best")
        if best is None:
            continue
        out[name] = {"B": int(best["B"]), "w": [float(x) for x in best["w"]]}
    return out


def build(configs, triblock_artifact=TRIBLOCK_ARTIFACT):
    cells = frozen_cells(triblock_artifact)
    missing = [c for c in configs if c not in cells]
    if missing:
        raise SystemExit(
            f"{triblock_artifact} has no chosen cell for {missing}; "
            f"run `make triblock` first")
    genes = tb.load_genes("best_solution.json")

    streams = {cfg: {"fit": draw_stream(cfg, SEEDS[cfg]["fit"],
                                        cells[cfg]["B"], tuple(cells[cfg]["w"])),
                     "holdout": draw_stream(cfg, SEEDS[cfg]["holdout"],
                                            cells[cfg]["B"], tuple(cells[cfg]["w"]))}
              for cfg in configs}
    flat = [s for cfg in configs for s in streams[cfg].values()]
    disjoint, dup = streams_disjoint(flat)

    shipped_region = {cfg: tb.region(np.asarray(genes, float), cfg, blend=0.0)
                      for cfg in configs}
    shipped_bow = {cfg: tb.region_report(shipped_region[cfg])["bow_over_width"]
                  for cfg in configs}

    results = {}
    for cfg in configs:
        B, w = cells[cfg]["B"], tuple(cells[cfg]["w"])
        fit_s, hold_s = streams[cfg]["fit"], streams[cfg]["holdout"]

        const_fwd = _fit(fit_s["rows"], CONSTANT_GRID, constant_rule, w, B, cfg)
        lin_fwd = _fit(fit_s["rows"], K_GRID, linear_rule, w, B, cfg)
        const_rev = _fit(hold_s["rows"], CONSTANT_GRID, constant_rule, w, B, cfg)
        lin_rev = _fit(hold_s["rows"], K_GRID, linear_rule, w, B, cfg)

        def shipped_under(rule_of, param):
            bend = rule_of(param)(shipped_bow[cfg])
            c = tb.cell(shipped_region[cfg], B, w, bend)
            return {"bend": bend,
                    "min_scaled_jacobian": c["min_scaled_jacobian"] if c else None,
                    "clears_target": bool(c and c["min_scaled_jacobian"]
                                          > tb.MIN_SJ_TARGET)}

        results[cfg] = {
            "B": B, "w": list(w), "bow_over_width_shipped": shipped_bow[cfg],
            "seeds": {"fit": fit_s["seed"], "holdout": hold_s["seed"]},
            "streams": {"fit": {"n": len(fit_s["rows"]),
                               "orientations_complete": fit_s["orientations_complete"]},
                       "holdout": {"n": len(hold_s["rows"]),
                                  "orientations_complete": hold_s["orientations_complete"]}},
            "forward": {
                "constant": {**const_fwd,
                            "holdout_scored_once": evaluate(
                                hold_s["rows"], constant_rule(const_fwd["param"]),
                                w, B, cfg),
                            "shipped": shipped_under(constant_rule, const_fwd["param"])},
                "linear": {**lin_fwd,
                          "holdout_scored_once": evaluate(
                              hold_s["rows"], linear_rule(lin_fwd["param"]), w, B, cfg),
                          "shipped": shipped_under(linear_rule, lin_fwd["param"])},
            },
            "swapped": {
                "constant": {**const_rev,
                            "holdout_scored_once": evaluate(
                                fit_s["rows"], constant_rule(const_rev["param"]),
                                w, B, cfg)},
                "linear": {**lin_rev,
                          "holdout_scored_once": evaluate(
                              fit_s["rows"], linear_rule(lin_rev["param"]), w, B, cfg)},
            },
        }

    rec = {
        "protocol": {
            "families": {"constant": "bend(bow) = b, b in BEND_GRID",
                        "linear": "bend(bow) = clip(k * bow, 0, 1), k in K_GRID"},
            "w_held_fixed_to": "the tri-block's own published cell (PLAN §55)",
            "selection": "max n_clear, then n_valid, then worst_min_sj; "
                        "ties: smaller parameter",
            "order": "fit -> freeze -> score holdout once -> swap",
            "seeds": SEEDS, "k_grid": list(K_GRID),
            "constant_grid": list(CONSTANT_GRID),
        },
        "min_sj_target": tb.MIN_SJ_TARGET,
        "per_config": results,
    }

    checks = {}
    checks["streams_pairwise_disjoint"] = disjoint
    checks["duplicate_gene_vectors"] = dup
    checks["orientations_complete_everywhere"] = all(
        s["orientations_complete"] for s in flat)
    checks["fit_streams_nonempty"] = all(
        len(streams[cfg]["fit"]["rows"]) > 0 for cfg in configs)
    checks["every_config_produced_finite_parameters"] = all(
        math.isfinite(r["forward"]["constant"]["param"])
        and math.isfinite(r["forward"]["linear"]["param"])
        for r in results.values())
    # Reported, not gated: whether the genome-dependent family actually beats the
    # zero-information constant on data neither has seen.  A "no" is this file's answer,
    # not its failure.
    checks["linear_beats_constant_on_holdout"] = {
        cfg: (r["forward"]["linear"]["holdout_scored_once"]["n_clear"],
              r["forward"]["constant"]["holdout_scored_once"]["n_clear"])
        for cfg, r in results.items()}
    checks["pass"] = all(v for k, v in checks.items()
                         if isinstance(v, bool))
    rec["self_checks"] = checks
    return rec


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def _print(rec):
    print("\n" + "=" * 78)
    print("  A GENOME-DEPENDENT BEND — constant vs. linear(bow), fit/freeze/score")
    print("=" * 78)
    for cfg, r in rec["per_config"].items():
        print(f"\n  {cfg.upper()}   B = {r['B']}   w = ({r['w'][0]:.3f}, "
              f"{r['w'][1]:.3f}, {r['w'][2]:.3f})   shipped bow/width "
              f"{r['bow_over_width_shipped']:.5f}")
        for name, tag in (("constant", "CONSTANT  bend = b"),
                          ("linear", "LINEAR    bend = clip(k*bow, 0, 1)")):
            f = r["forward"][name]
            h = f["holdout_scored_once"]
            print(f"    {tag}")
            print(f"      fit-chosen param: {f['param']:.4f}   "
                  f"in-sample n_clear/n_valid: {f['in_sample']['n_clear']}/"
                  f"{f['in_sample']['n_valid']} of {f['in_sample']['n']}")
            print(f"      HOLD-OUT, scored once: n_clear/n_valid "
                  f"{h['n_clear']}/{h['n_valid']} of {h['n']}, worst min SJ "
                  f"{h['worst_min_scaled_jacobian']:.4f}")
            if "shipped" in f:
                s = f["shipped"]
                print(f"      shipped genome under this frozen rule: bend "
                      f"{s['bend']:.4f}, min SJ {s['min_scaled_jacobian']:.4f}, "
                      f"clears target: {s['clears_target']}")
    print("\n  SELF-CHECKS")
    for k, v in rec["self_checks"].items():
        if k == "pass":
            continue
        print(f"    {k:44s} {v}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    ap.add_argument("--triblock-artifact",
                    default=os.path.basename(TRIBLOCK_ARTIFACT))
    ap.add_argument("--out", default="study_tri_bend.json")
    args = ap.parse_args()

    configs = tuple(c for c in args.configs.split(",") if c)
    _gate_guard.refuse_degraded_out(ap, args, "study_tri_bend.json", [
        (set(configs) != set(DEFAULT_CONFIGS),
         f"--configs {args.configs} is not the committed "
         f"{','.join(DEFAULT_CONFIGS)}"),
        (os.path.basename(args.triblock_artifact) != "study_tri_block.json",
         "--triblock-artifact is not PART 2's committed measurement"),
    ])

    t0 = time.time()
    rec = build(configs, os.path.join(HERE, args.triblock_artifact))
    rec["seconds"] = round(time.time() - t0, 2)
    out = os.path.join(HERE, args.out)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
    _print(rec)
    print(f"  wrote {out}  ({rec['seconds']} s)")
    raise SystemExit(0 if rec["self_checks"]["pass"] else 1)


if __name__ == "__main__":
    main()
