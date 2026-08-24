#!/usr/bin/env python
"""
 =============================================================================
 THE TRI-BLOCK'S FOLD RULE, CALIBRATED ON A PROPER HOLD-OUT PROTOCOL
 (PLAN.md §53 successor 1; UNCAP_PLAN.md STEP 3 RECORD, PART 3)
 =============================================================================

WHAT THIS ANSWERS.  PART 2 named the mechanism — "the ones it folds on are the WIDE
weld arcs" — and left open whether a RULE on that mechanism can predict the fold across
the gene box.  A conjunctive form has been circulating informally,

    arc > t_wide   OR   (arc > t_conj  AND  min_wedge < t_wedge)

with hand-read thresholds and an in-sample score of 1.000.  An in-sample 1.000 is not a
measurement; it is one degree freer than the data it sits on.  This file calibrates the
form the only way that means anything: FIT ON ONE STREAM, FREEZE, SCORE ON A DISJOINT
ONE — then swap.

THE LABEL, FIXED BEFORE ANYTHING RAN.  Fold means `best_w_valid == False`: NO interior
point on the published barycentric grid makes the three tri blocks valid.  That is a
statement about the straight-Y CONSTRUCTION, which is what §53's successor asks about;
`fixed_w_valid` would instead score the shipped weights and is carried below only as a
secondary column, never used to fit.

THE PRE-REGISTERED PROTOCOL, written before any hold-out was drawn:

  1. STREAMS.  Fresh genome draws from `study_tri_block.sweep_genomes` itself — same
     feasibility filter, same orientation balance, same B and w_fixed the committed
     measurement chose — differing ONLY in their LHS seed.  Every stream carries its
     own seed, because the candidate pool is a function of the seed alone and two
     streams sharing one would draw the same candidates:

         coarse   fit 20260800 + 20260900      hold-out   20261000
         medium   fit 20261100 + 20261200      hold-out   20261300

     Pairwise disjointness of ALL SIX streams is asserted on hashed gene vectors, and
     a duplicate would fail the run rather than shrink a number.  (A first spacing
     attempt put the seeds tens apart and FAILED that assertion: `sweep_genomes`
     draws its batch `b` candidates from `seed + b`, up to forty batches, so nearby
     seeds literally share candidate batches.  The registered seeds are therefore
     spaced a hundred apart — a correctness fix, not a re-roll.)

     AMENDMENT, RECORDED RATHER THAN SILENT.  The first registration drew ONE seed per
     role, and its fit streams came back 16/16 CLEAN at both configs — the fold event
     is rarer across fresh draws than PART 2's committed stream suggested, and a fit
     stream with no folds fits nothing except a vacuous rule (the driver refused to
     file one, which is what its self-check is for).  The amendment pools TWO
     independent fit draws per config, doubling n and making a fold-free fit stream
     loud instead of likely; the hold-out stays a SINGLE draw, never touched by
     fitting, and its seed did not move.  What the first registration measured is
     quoted in the record as a finding: fold rates vary strongly between draws.

  2. THE GRID, ANCHORED TO THE FIT STREAM.  Candidate thresholds are the midpoints
     between consecutive DISTINCT feature values of the fit stream, plus a sentinel
     infinity that disables a branch — so the family can lose a clause if the data says
     so, and no threshold is ever read off a hold-out genome.

  3. SELECTION, FIT STREAM ONLY.  Maximise accuracy; ties broken toward FEWER fires,
     then toward the lexicographically smallest threshold triple.  Deterministic, and
     biased against a rule that cries wolf.

  4. FREEZE, THEN SCORE ONCE.  The chosen triple is written into the artifact before
     the hold-out is touched, and the hold-out is scored EXACTLY once under it.  The
     swap (fit on the hold-out's draw, score on the pooled fit draws) is a second,
     independent replication of step 4, not a refit of step 3's output.

  5. HONESTY COLUMNS.  Every accuracy is reported beside n, class balance, and the
     per-branch fire counts — so a wide-arc-branch counterexample is NAMED (genes, arc,
     wedges) rather than averaged away — and beside PART 2's own ceiling, because at
     the current B even the best-per-genome re-sweep only reaches 15/16 and 12/16, so
     a rule alone may not suffice whatever its held-out score.

  6. REFERENCES, CLEARLY LABELLED AS OUTSIDE THE PROTOCOL.  Two of them, both scored on
     the SAME hold-out the frozen rule faces:
       - the INFORMAL rule whose hand-read thresholds motivated this file
         (arc > 36.16 OR (arc > 30 AND min_wedge < 17.12)) — never fitted here, quoted
         only so its held-out behaviour is visible next to the calibrated one;
       - the frozen rules against PART 2's committed stream (seed 20260823), the
         stream the informal numbers circulated on.

EXIT STATUS follows `make triblock`: nonzero ONLY if a self-check fails — stream
disjointness, orientation balance, or an empty class in a fit stream.  A held-out
accuracy, including an embarrassing one, is a finding and exits zero.

WHAT THIS DOES NOT DO.  It wires nothing into `build_wheel` or `sector_blocks`, adopts
no rule, moves no threshold anywhere else in the tree, and leaves `best_solution.json`
alone.  A rule that predicts where the construction folds is a SCREEN for §53's
successor work, not a fix for it; the curved Y remains the only lever the Winslow
column names.
=============================================================================
"""

import argparse
import json
import math
import os
import time

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import _gate_guard
import study_tri_block as tb

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIGS = ("coarse", "medium")

# The registered seeds.  All six are pairwise >= 100 apart: the candidate pool for
# batch `b` is `seed + b` (up to forty batches), so nearby seeds share candidate
# batches and disjointness fails — measured, not assumed.  Each fit role pools TWO
# draws (the recorded amendment: a single fit draw came back fold-free at both configs
# on the first registration); each hold-out is ONE draw whose role never changes.
SEEDS = {
    "coarse": {"fit": (20260800, 20260900), "holdout": (20261000,)},
    "medium": {"fit": (20261100, 20261200), "holdout": (20261300,)},
}

# The informal rule that motivated this file — hand-read thresholds, in-sample 1.000.
# Never fitted here; scored on the same hold-outs as the calibrated rule so the two
# can be read side by side, and labelled as outside the protocol wherever it appears.
INFORMAL_RULE = {"t_wide": 36.16, "t_conj": 30.0, "t_wedge": 17.12}

# The committed measurement's chosen cell, read rather than re-swept: the streams must
# be measured at the same B and interior point PART 2 published, or the labels are not
# comparable to anything.
TRIBLOCK_ARTIFACT = os.path.join(HERE, "study_tri_block.json")


# ---------------------------------------------------------------------------
# FEATURES, LABEL, RULE FAMILY
# ---------------------------------------------------------------------------

def features(row):
    """The two features PART 2's mechanism names: the weld arc's span and the
    triangle's tightest wedge."""
    return (float(row["arc_span_deg"]), min(float(w) for w in row["wedges_deg"]))


def label(row):
    """Fold = no interior point exists.  Rows the sweep could not label are excluded
    upstream and counted, never guessed."""
    return not bool(row["best_w_valid"])


def labelled(rows):
    ok = [r for r in rows if r.get("best_w_valid") is not None]
    return ok, len(rows) - len(ok)


def predict(feat, rule):
    """`arc > t_wide OR (arc > t_conj AND min_wedge < t_wedge)`.

    A sentinel `inf` disables its branch cleanly: `feat > inf` is False, so a rule the
    fit wants to strip to one clause is representable inside the family rather than
    forced to keep a clause nobody selected.
    """
    arc, mw = feat
    if rule["t_wide"] != math.inf and arc > rule["t_wide"]:
        return True
    return (rule["t_conj"] != math.inf and rule["t_wedge"] != math.inf
            and arc > rule["t_conj"] and mw < rule["t_wedge"])


def branch_of(feat, rule):
    """Which clause fired, so per-branch behaviour is reportable rather than implied."""
    arc, mw = feat
    if rule["t_wide"] != math.inf and arc > rule["t_wide"]:
        return "wide"
    if (rule["t_conj"] != math.inf and rule["t_wedge"] != math.inf
            and arc > rule["t_conj"] and mw < rule["t_wedge"]):
        return "conjunctive"
    return None


# ---------------------------------------------------------------------------
# THE PRE-REGISTERED GRID AND SELECTION
# ---------------------------------------------------------------------------

def threshold_grid(values):
    """Midpoints between consecutive DISTINCT values, plus the disabling sentinel.

    Anchored to whatever stream is passed — which, by the protocol, is always the fit
    stream.  Midpoints rather than the values themselves so no threshold sits exactly
    on a fit genome (a rule that fires on equality with its own anchor is the kind of
    in-sample luck this file exists to price out).
    """
    uniq = sorted(set(float(v) for v in values))
    cuts = [(a + b) / 2.0 for a, b in zip(uniq, uniq[1:])]
    return cuts + [math.inf]


def fit_rule(rows):
    """Grid search on the FIT stream only, under the registered tie-breaks.

    Selection key, minimised: (-accuracy, n_fires, t_wide, t_conj, t_wedge).  Accuracy
    first; among equally accurate rules prefer the one that fires less often; then the
    deterministic lexicographic preference so the same fit stream always yields the
    same triple.  Returns the rule and its own in-sample confusion, which is reported
    as IN-SAMPLE and never quoted as the result.
    """
    feats = [features(r) for r in rows]
    ys = [label(r) for r in rows]
    arcs = [f[0] for f in feats]
    mws = [f[1] for f in feats]
    wide_grid = threshold_grid(arcs)
    conj_grid = threshold_grid(arcs)
    wedge_grid = threshold_grid(mws)
    best = None
    for tw in wide_grid:
        for tc in conj_grid:
            for tg in wedge_grid:
                rule = {"t_wide": tw, "t_conj": tc, "t_wedge": tg}
                preds = [predict(f, rule) for f in feats]
                acc = sum(p == y for p, y in zip(preds, ys)) / len(ys)
                fires = sum(preds)
                key = (-acc, fires, tw, tc, tg)
                if best is None or key < best[0]:
                    best = (key, rule, acc, fires)
    _, rule, acc, fires = best
    conf = confusion([predict(f, rule) for f in feats], ys)
    return {"rule": rule, "in_sample": dict(conf, n=len(ys), accuracy=acc,
                                            fires=fires)}


def confusion(preds, ys):
    tp = sum(1 for p, y in zip(preds, ys) if p and y)
    fp = sum(1 for p, y in zip(preds, ys) if p and not y)
    tn = sum(1 for p, y in zip(preds, ys) if not p and not y)
    fn = sum(1 for p, y in zip(preds, ys) if not p and y)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ---------------------------------------------------------------------------
# SCORING A FROZEN RULE
# ---------------------------------------------------------------------------

def score_frozen(rule, rows):
    """Score once, frozen.  Also reports per-branch fires and NAMES every counterexample
    the wide-arc branch takes, because a branch that eats a clean genome is the exact
    defect an averaged accuracy hides."""
    ok, dropped = labelled(rows)
    preds, ys = [], []
    per_branch = {"wide": {"tp": 0, "fp": 0},
                  "conjunctive": {"tp": 0, "fp": 0}}
    false_fires = []
    missed = []
    for r in ok:
        f, y = features(r), label(r)
        p = predict(f, rule)
        preds.append(p)
        ys.append(y)
        b = branch_of(f, rule)
        if b is not None:
            per_branch[b]["tp" if y else "fp"] += 1
            if not y:
                false_fires.append({
                    "branch": b, "arc_span_deg": f[0],
                    "min_wedge_deg": f[1], "wedges_deg": r["wedges_deg"],
                    "orientation": r["orientation"],
                    "best_w_min_scaled_jacobian":
                        r.get("best_w_min_scaled_jacobian")})
        elif y:
            missed.append({"arc_span_deg": f[0], "min_wedge_deg": f[1],
                           "orientation": r["orientation"]})
    conf = confusion(preds, ys)
    return {"n": len(ok), "unlabelled_dropped": dropped,
            **conf,
            "accuracy": (conf["tp"] + conf["tn"]) / len(ok) if ok else None,
            "fold_rate": (conf["tp"] + conf["fn"]) / len(ok) if ok else None,
            "per_branch": per_branch,
            "false_fires": false_fires,
            "missed_folds": missed}


# ---------------------------------------------------------------------------
# STREAMS
# ---------------------------------------------------------------------------

def frozen_cells(path=TRIBLOCK_ARTIFACT):
    """B and w_fixed per config, from PART 2's committed choice."""
    with open(path) as fh:
        rec = json.load(fh)
    out = {}
    for name, per in rec["per_config"].items():
        best = per.get("sweep", {}).get("best")
        if best is None:
            continue
        out[name] = {"B": int(best["B"]), "w": [float(x) for x in best["w"]]}
    return out


def draw_stream(cfg, cell, seeds):
    """One stream = the union of one draw per registered seed.  A single-seed stream
    passes `(seed,)`; the pooled fit streams pass two."""
    rows, draws = [], []
    for seed in seeds:
        sw = tb.sweep_genomes(cfg, cell["B"], tuple(cell["w"]),
                              seed=seed, max_batches=40)
        rows.extend(r for v in sw["groups"].values() for r in v)
        draws.append({"seed": seed, "n_genomes": sw["n_genomes"],
                      "n_fixed_w_valid": sw["n_fixed_w_valid"],
                      "n_best_w_valid": sw["n_best_w_valid"],
                      "orientations_complete": (
                          len(sw["groups"]) == 4
                          and all(len(v) == tb.GENOME_SWEEP_PER_ORIENTATION
                                  for v in sw["groups"].values()))})
    return {"config": cfg, "seeds": list(seeds), "B": cell["B"],
            "w_fixed": list(cell["w"]), "rows": rows,
            "draws": draws,
            "orientations_complete": all(d["orientations_complete"]
                                         for d in draws)}


def gene_key(row):
    import hashlib
    blob = ",".join(repr(float(x)) for x in row["genes"]).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def streams_disjoint(streams):
    seen, dup = {}, None
    for s in streams:
        for r in s["rows"]:
            k = gene_key(r)
            if k in seen:
                dup = (seen[k], tuple(s["seeds"]), k)
                return False, dup
            seen[k] = tuple(s["seeds"])
    return True, None


def class_balance(rows):
    ok, _ = labelled(rows)
    folds = sum(1 for r in ok if label(r))
    return {"n": len(ok), "folds": folds, "clean": len(ok) - folds}


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------

def build(configs, triblock_artifact=TRIBLOCK_ARTIFACT):
    cells = frozen_cells(triblock_artifact)
    missing = [c for c in configs if c not in cells]
    if missing:
        raise SystemExit(
            f"{triblock_artifact} has no chosen cell for {missing}; "
            f"run `make triblock` first")

    # STEP 1 — draw the six streams, and assert they are six populations.
    streams = {}
    for cfg in configs:
        streams[cfg] = {
            "fit": draw_stream(cfg, cells[cfg], SEEDS[cfg]["fit"]),
            "holdout": draw_stream(cfg, cells[cfg], SEEDS[cfg]["holdout"]),
        }
    flat = [s for cfg in configs for s in streams[cfg].values()]
    disjoint, dup = streams_disjoint(flat)

    # STEPS 2-4 — fit, freeze, score once; then the swap as an independent replicate.
    results = {}
    for cfg in configs:
        fit_s, hold_s = streams[cfg]["fit"], streams[cfg]["holdout"]
        fitted_fwd = fit_rule(fit_s["rows"])
        frozen = fitted_fwd["rule"]
        fitted_rev = fit_rule(hold_s["rows"])
        results[cfg] = {
            "frozen_cell": {"B": cells[cfg]["B"], "w": cells[cfg]["w"]},
            "seeds": {"fit": list(SEEDS[cfg]["fit"]),
                      "holdout": list(SEEDS[cfg]["holdout"])},
            "streams": {
                "fit": {k: v for k, v in fit_s.items() if k != "rows"},
                "holdout": {k: v for k, v in hold_s.items() if k != "rows"}},
            "class_balance": {"fit": class_balance(fit_s["rows"]),
                              "holdout": class_balance(hold_s["rows"])},
            "forward": {
                "frozen_rule": frozen,
                "in_sample_fit": fitted_fwd["in_sample"],
                "informal_rule_on_same_holdout": {
                    "rule": dict(INFORMAL_RULE), "outside_the_protocol": True,
                    **score_frozen(INFORMAL_RULE, hold_s["rows"])},
                "holdout_scored_once": score_frozen(frozen, hold_s["rows"])},
            "swapped": {
                "frozen_rule": fitted_rev["rule"],
                "in_sample_fit": fitted_rev["in_sample"],
                "holdout_scored_once": score_frozen(
                    fitted_rev["rule"], fit_s["rows"])},
        }

    # STEP 6 — references on PART 2's own stream, labelled as outside the protocol.
    with open(triblock_artifact) as fh:
        committed = json.load(fh)
    for cfg in configs:
        g = committed["per_config"][cfg].get("genomes")
        if not g:
            continue
        rows = [r for v in g["groups"].values() for r in v]
        results[cfg]["corroboration_part2_stream"] = {
            "seed": g["seed"], "outside_the_protocol": True,
            **score_frozen(results[cfg]["forward"]["frozen_rule"], rows)}

    rec = {
        "protocol": {
            "label": "not best_w_valid (no interior point exists on the grid)",
            "family": "arc > t_wide OR (arc > t_conj AND min_wedge < t_wedge)",
            "grid": "midpoints of consecutive distinct FIT-stream values + inf",
            "selection": "max fit accuracy; ties: fewer fires, then lexicographic",
            "order": "fit -> freeze -> score holdout once -> swap",
            "amendment": ("first registration drew one seed per role; its fit "
                          "streams came back fold-free at both configs, so the fit "
                          "side pools two independent draws. A first seed spacing "
                          "tens apart shared candidate batches (batch b draws from "
                          "seed + b) and failed disjointness; the registered seeds "
                          "are >= 100 apart."),
            "informal_rule": dict(INFORMAL_RULE),
            "seeds": SEEDS,
        },
        "min_sj_target": tb.MIN_SJ_TARGET,
        "per_config": results,
        "_streams_for_checks": {
            cfg: {role: [gene_key(r) for r in s["rows"]]
                  for role, s in pair.items()}
            for cfg, pair in streams.items()},
    }

    checks = {}
    checks["six_streams_pairwise_disjoint"] = disjoint
    checks["duplicate_gene_vectors"] = dup
    checks["orientations_complete_everywhere"] = all(
        s["orientations_complete"] for s in flat)
    empty = {f"{cfg}/{role}": class_balance(s["rows"])
             for cfg in configs for role, s in streams[cfg].items()
             if class_balance(s["rows"])["folds"] == 0}
    checks["fit_streams_have_both_classes"] = all(
        class_balance(streams[cfg]["fit"]["rows"])["folds"] > 0
        and class_balance(streams[cfg]["fit"]["rows"])["clean"] > 0
        for cfg in configs)
    checks["stream_with_no_folds_at_all"] = empty or None
    checks["every_config_produced_a_finite_frozen_rule"] = all(
        math.isfinite(r["forward"]["frozen_rule"]["t_wide"])
        or math.isfinite(r["forward"]["frozen_rule"]["t_conj"])
        for r in results.values())
    checks["pass"] = all(v for k, v in checks.items()
                         if isinstance(v, bool))
    rec["self_checks"] = checks
    return rec


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def _fmt_conf(c, n=None):
    base = (f"TP {c['tp']:2d}  FP {c['fp']:2d}  TN {c['tn']:2d}  FN {c['fn']:2d}"
            + (f"   n {c['n']}" if "n" in c else "")
            + (f"   accuracy {c['accuracy']:.3f}" if c.get("accuracy") is not None
               else ""))
    return base


def _print(rec):
    print("\n" + "=" * 78)
    print("  THE TRI-BLOCK'S FOLD RULE, CALIBRATED — fit, freeze, score disjoint")
    print("=" * 78)
    print("\n  label: fold = NOT best_w_valid (no interior point exists)")
    print("  family: " + rec["protocol"]["family"])
    for cfg, r in rec["per_config"].items():
        cb = r["class_balance"]
        print(f"\n  {cfg.upper()}   B = {r['frozen_cell']['B']}   "
              f"w_fixed = ({r['frozen_cell']['w'][0]:.3f}, "
              f"{r['frozen_cell']['w'][1]:.3f}, {r['frozen_cell']['w'][2]:.3f})")
        print(f"    streams: fit seeds {r['seeds']['fit']} "
              f"(folds {cb['fit']['folds']}/{cb['fit']['n']}), "
              f"hold-out seed {r['seeds']['holdout']} "
              f"(folds {cb['holdout']['folds']}/{cb['holdout']['n']})")
        for d in r["streams"]["fit"]["draws"] + r["streams"]["holdout"]["draws"]:
            print(f"      draw seed {d['seed']}: n_genomes {d['n_genomes']}, "
                  f"best-w folds {d['n_genomes'] - d['n_best_w_valid']}, "
                  f"orientations complete: {d['orientations_complete']}")
        for tag, direction in (("FORWARD (pooled fit -> hold-out, scored once)",
                                "forward"),
                               ("SWAPPED (fit on the hold-out draw -> pooled fit)",
                                "swapped")):
            d = r[direction]
            ru = d["frozen_rule"]
            parts = []
            parts.append("arc > " + (
                f"{ru['t_wide']:.3f}" if math.isfinite(ru["t_wide"]) else "--"))
            if math.isfinite(ru["t_conj"]) and math.isfinite(ru["t_wedge"]):
                parts.append(f"OR (arc > {ru['t_conj']:.3f} AND "
                             f"min_wedge < {ru['t_wedge']:.3f})")
            print(f"\n    {tag}")
            print(f"      frozen rule:  {' '.join(parts)}")
            ins = d["in_sample_fit"]
            print(f"      IN-SAMPLE (the fit stream, never the result):  {_fmt_conf(ins)}")
            h = d["holdout_scored_once"]
            print(f"      HOLD-OUT, scored once under the freeze:           "
                  f"TP {h['tp']:2d}  FP {h['fp']:2d}  TN {h['tn']:2d}  FN {h['fn']:2d}"
                  f"   accuracy {h['accuracy']:.3f}"
                  + (f"   (fold rate {h['fold_rate']:.3f}, n {h['n']})"
                     if h["fold_rate"] is not None else f"   (n {h['n']})"))
            wb, cbj = h["per_branch"]["wide"], h["per_branch"]["conjunctive"]
            print(f"      branches: wide TP {wb['tp']} FP {wb['fp']};  "
                  f"conjunctive TP {cbj['tp']} FP {cbj['fp']}")
            for cx in h["false_fires"]:
                where = ("WIDE-BRANCH" if cx["branch"] == "wide"
                         else "CONJUNCTIVE-BRANCH")
                print(f"        {where} FALSE FIRE: arc "
                      f"{cx['arc_span_deg']:.3f} deg, min wedge "
                      f"{cx['min_wedge_deg']:.3f} deg, orientation "
                      f"{tuple(cx['orientation'])} — the construction is CLEAN there "
                      f"(best w min scaled J {cx['best_w_min_scaled_jacobian']})")
            for m in h["missed_folds"]:
                print(f"        MISSED FOLD: arc {m['arc_span_deg']:.3f} deg, "
                      f"min wedge {m['min_wedge_deg']:.3f} deg, orientation "
                      f"{tuple(m['orientation'])}")
            if direction == "forward":
                inf = r["forward"]["informal_rule_on_same_holdout"]
                print(f"      THE INFORMAL RULE on the same hold-out "
                      f"(arc > {inf['rule']['t_wide']} OR (arc > "
                      f"{inf['rule']['t_conj']} AND min_wedge < "
                      f"{inf['rule']['t_wedge']})) — outside the protocol:")
                acc = (f"   accuracy {inf['accuracy']:.3f}"
                       if inf["accuracy"] is not None else "")
                print(f"        TP {inf['tp']:2d}  FP {inf['fp']:2d}  "
                      f"TN {inf['tn']:2d}  FN {inf['fn']:2d}{acc}"
                      f"   (n {inf['n']})")
                for cx in inf["false_fires"]:
                    print(f"        it FIRES CLEAN at arc "
                          f"{cx['arc_span_deg']:.3f} deg, min wedge "
                          f"{cx['min_wedge_deg']:.3f} deg ({cx['branch']} branch)")
                for m in inf["missed_folds"]:
                    print(f"        it MISSES a fold at arc {m['arc_span_deg']:.3f} "
                          f"deg, min wedge {m['min_wedge_deg']:.3f} deg")
        cor = r.get("corroboration_part2_stream")
        if cor:
            print(f"\n    REFERENCE on PART 2's committed stream (seed "
                  f"{cor['seed']}) — outside the protocol, quoted only against the "
                  f"informal numbers:")
            print(f"      TP {cor['tp']:2d}  FP {cor['fp']:2d}  TN {cor['tn']:2d}  "
                  f"FN {cor['fn']:2d}   accuracy {cor['accuracy']:.3f}   (n {cor['n']})")
    print("\n  SELF-CHECKS")
    for k, v in rec["self_checks"].items():
        if k == "pass":
            continue
        shown = "PASS" if v is True else ("none" if v is None else str(v))
        print(f"    {k:44s} {shown}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    ap.add_argument("--triblock-artifact",
                    default=os.path.basename(TRIBLOCK_ARTIFACT))
    ap.add_argument("--out", default="study_tri_rule.json")
    args = ap.parse_args()

    configs = tuple(c for c in args.configs.split(",") if c)
    _gate_guard.refuse_degraded_out(ap, args, "study_tri_rule.json", [
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
