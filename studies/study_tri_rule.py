#!/usr/bin/env python
"""
 =============================================================================
 THE TRI-BLOCK'S FOLD RULE, CALIBRATED ON A PROPER HOLD-OUT PROTOCOL
 (PLAN.md §73 successor 1; UNCAP_PLAN.md STEP 3 RECORD, PART 9 -> PART 10)
 =============================================================================

WHAT THIS ANSWERS.  PART 9 (PLAN §73) named the mechanism -- the SMALLEST interior wedge
angle, conjunctive with the weld arc's span -- and found a rule on it that scores 1.000
in-sample and 0.833 held out, falsifying its own wide-arc branch outright.  It explicitly
declined to publish either threshold: "an in-sample 1.000 would have been the wrong number
to publish."  This file is that calibration, run properly: FIT ON ONE STREAM, FREEZE,
SCORE ON A DISJOINT ONE -- then swap.

THIS IS THE SAME APPARATUS RE-AIMED, NOT A NEW ONE.  Its first registration (recorded in
this file's own history) was shaken down against `best_w_valid` -- whether ANY interior
point on the published grid makes the STRAIGHT Y valid -- over `sweep_genomes`' UNIFORM
draw.  That is §53's question, already closed by §55.  PART 9's open question is
`curved_valid` -- whether ANY (bend, interior point) makes the CURVED Y valid -- over
`sweep_arc_span_band`'s draw, CONDITIONED on arc span > 30 deg (§72), because the fold event
is essentially unsampled by a uniform draw (about one genome in sixty-four exceeds 35
degrees of arc span) and PART 9's own 22/40 refusal rate lives inside that band and nowhere
else.  Re-aiming needed two things `sweep_arc_span_band` did not have: a `seed_base` kwarg
so a caller can draw a band independent of the committed one, and a `genes` field on every
band row so two streams' disjointness can be asserted from the genes themselves rather than
assumed from the seeds.  Both are now in `study_tri_block.sweep_arc_span_band`.

THE LABEL, FIXED BEFORE ANYTHING RAN.  Fold means `curved_valid == False`: no (bend,
interior point) on the published grids makes the three tri blocks valid.  That is PART 9's
own subject, and it is read directly off each band row's `curved_valid` -- `_shape`'s
`min_wedge_deg`/`arc_span_deg` are exactly the two features PART 9 names, computed once by
`sweep_arc_span_band` and carried on the row rather than re-derived here.

THE PRE-REGISTERED PROTOCOL, written before any hold-out was drawn:

  1. STREAMS.  Fresh band draws from `study_tri_block.sweep_arc_span_band` itself -- same
     `arc_span_deg > 30` screen, same feasibility filter, same B and w_fixed PART 2's
     committed measurement chose -- differing ONLY in their LHS seed base:

         coarse   fit 20262000              hold-out   20263000
         medium   fit 20264000              hold-out   20265000

     Bases must be spaced >= `ARC_BAND_MAX_BATCHES` (400) apart, because a band draw
     consumes batch `b` from `seed_base + b` and may use up to 400 of them reaching its
     40-row target; the registered bases are 500-1000 apart, and pairwise disjointness of
     every stream drawn is asserted on hashed gene vectors rather than assumed from the
     spacing.  ONE fit seed per config is registered, not two: the class balance this
     apparatus's first registration had to guard against (a fold-free uniform draw) is a
     property of the UNIFORM sampler PART 8 diagnosed and fixed by conditioning on arc
     span -- the committed band already runs 22/40 (55%) -- so a single band draw is not
     expected to come back one-class.  `AMENDMENT_FIT_SEED` is reserved per config and used
     -- with the amendment recorded rather than silent, the same discipline the first
     registration set -- ONLY if the drawn fit stream turns out to have an empty class.

  2. THE GRID, ANCHORED TO THE FIT STREAM.  Candidate thresholds are the midpoints
     between consecutive DISTINCT feature values of the fit stream, plus a sentinel
     infinity that disables a branch -- so the family can lose a clause if the data says
     so, and no threshold is ever read off a hold-out genome.

  3. SELECTION, FIT STREAM ONLY.  Maximise accuracy; ties broken toward FEWER fires,
     then toward the lexicographically smallest threshold triple.  Deterministic, and
     biased against a rule that cries wolf.

  4. FREEZE, THEN SCORE ONCE.  The chosen triple is written into the artifact before
     the hold-out is touched, and the hold-out is scored EXACTLY once under it.  The
     swap (fit on the hold-out's draw, score on the fit draw) is a second, independent
     replication of step 4, not a refit of step 3's output.

  5. HONESTY COLUMNS.  Every accuracy is reported beside n, class balance, and the
     per-branch fire counts -- so a wide-arc-branch counterexample is NAMED (genes, arc,
     wedge, shape) rather than averaged away -- and beside PART 2's own ceiling, because at
     the current B even the best-per-genome re-sweep only reaches 15/16 and 12/16, so
     a rule alone may not suffice whatever its held-out score.

  6. REFERENCES, CLEARLY LABELLED AS OUTSIDE THE PROTOCOL.  Two of them, both scored on
     the SAME hold-out the frozen rule faces:
       - the INFORMAL rule PART 9 fitted in-sample and refused to publish
         (arc > 36.16 OR (arc > 30 AND min_wedge < 17.12)) -- never fitted here, quoted
         only so its held-out behaviour is visible next to the calibrated one;
       - the frozen rule against PART 9's own committed band (seed base
         GENOME_SWEEP_SEED + 1000), the stream the informal numbers circulated on -- at
         `coarse` only, because that is the only config the committed artifact ran the
         band for.

EXIT STATUS follows `make triblock`: nonzero ONLY if a self-check fails -- stream
disjointness, a band that did not reach its target, or an empty class in a fit stream that
the reserved amendment seed did not fix.  A held-out accuracy, including an embarrassing
one, is a finding and exits zero.

WHAT THIS DOES NOT DO.  It wires nothing into `build_wheel` or `sector_blocks`, adopts
no rule, moves no threshold anywhere else in the tree, and leaves `best_solution.json`
alone.  A rule that predicts where the construction folds is a SCREEN for the tri-block's
adoption decision, not a fix for it; the curved Y remains the only lever the Winslow
column names, and Step 2 of this session's own successor list is what tries to make its
bend a function of the genome.

=============================================================================
 HISTORY -- THE FIRST REGISTRATION, KEPT RATHER THAN DELETED
=============================================================================
The apparatus below was built and shaken down against a DIFFERENT label first:
`best_w_valid` (no interior point makes the STRAIGHT Y valid) over `sweep_genomes`'
uniform draw -- §53's question.  That shakedown is the reason this file's protocol
machinery exists in the shape it does (the pooled-fit amendment in particular), and its
result is worth keeping as a finding even though it answers a question §55 already
closed: every refit landed in-sample 1.000 on DIFFERENT thresholds; held-out
1.000/0.906-0.969 with one-to-two-fold classes; the informal 36.16/30/17.12 rule was not
beaten by the calibrated one on any stream; and a first registration with one seed per
role came back fold-free in both fit streams at both configs, which is what forced the
pooled-fit amendment in the first place.  None of that is re-run here -- re-aiming a
one-genome-sweep apparatus at a forty-genome band is not a matter of changing a label
string, and the streams below are fresh draws under the new one.
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

# The registered seed bases.  Spaced >= `ARC_BAND_MAX_BATCHES` (400) apart -- a band draw
# consumes batch `b` from `seed_base + b` and may use up to 400 of them reaching its
# 40-row target, so nearer bases would share candidate batches -- measured, not assumed,
# by `streams_disjoint` below.  One fit seed per config; see the module docstring for why
# this apparatus does not pre-emptively pool two the way its first registration had to.
SEEDS = {
    "coarse": {"fit": (20262000,), "holdout": (20263000,)},
    "medium": {"fit": (20264000,), "holdout": (20265000,)},
}

# Reserved, used ONLY if a config's fit stream comes back with an empty class -- the same
# amendment discipline the first registration recorded rather than hid.  500 apart from
# its config's primary fit seed, still >= ARC_BAND_MAX_BATCHES.
AMENDMENT_FIT_SEED = {"coarse": 20262500, "medium": 20264500}

# The informal rule PART 9 fitted in-sample and explicitly declined to publish.  Never
# fitted here; scored on the same hold-outs as the calibrated rule so the two can be read
# side by side, and labelled as outside the protocol wherever it appears.
INFORMAL_RULE = {"t_wide": 36.16, "t_conj": 30.0, "t_wedge": 17.12}

# The committed measurement's chosen cell, read rather than re-swept: the streams must
# be measured at the same B and interior point PART 2 published, or the labels are not
# comparable to anything.
TRIBLOCK_ARTIFACT = os.path.join(HERE, "study_tri_block.json")


# ---------------------------------------------------------------------------
# FEATURES, LABEL, RULE FAMILY
# ---------------------------------------------------------------------------

def features(row):
    """The two features PART 9's mechanism names: the weld arc's span and the
    triangle's tightest wedge -- both already on the row via `study_tri_block._shape`."""
    return (float(row["arc_span_deg"]), float(row["min_wedge_deg"]))


def label(row):
    """Fold = no (bend, interior point) on the published grids makes the curved Y valid.
    Rows the sweep could not label are excluded upstream and counted, never guessed."""
    return not bool(row["curved_valid"])


def labelled(rows):
    ok = [r for r in rows if r.get("curved_valid") is not None]
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

    Anchored to whatever stream is passed -- which, by the protocol, is always the fit
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
                    "branch": b, "arc_span_deg": f[0], "min_wedge_deg": f[1],
                    "bow_over_width": r.get("bow_over_width"),
                    "turn_at_far_end_deg": r.get("turn_at_far_end_deg"),
                    "genes": r.get("genes")})
        elif y:
            missed.append({"arc_span_deg": f[0], "min_wedge_deg": f[1],
                           "bow_over_width": r.get("bow_over_width"),
                           "turn_at_far_end_deg": r.get("turn_at_far_end_deg"),
                           "genes": r.get("genes")})
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


def draw_band_stream(cfg, cell, seed_bases, shipped_genes):
    """One band stream = the union of one `sweep_arc_span_band` draw per registered seed
    base.  Unlike the first registration's uniform pool, a band draw is CONDITIONED on
    arc span > 30 deg (§72) and its rows already carry `curved_valid` / `arc_span_deg` /
    `min_wedge_deg` via `study_tri_block._shape` -- no re-derivation needed here.
    """
    rows, draws = [], []
    for seed_base in seed_bases:
        res = tb.sweep_arc_span_band(cfg, cell["B"], tuple(cell["w"]), shipped_genes,
                                     tuple(cell["w"]), seed_base=seed_base)
        rows.extend(res["genomes"])
        draws.append({"seed_base": seed_base, "n_drawn": res["n_drawn"],
                      "n_in_band": res["n_in_band"], "n_meshable": res["n_meshable"],
                      "reached_target": res["n_meshable"] >= tb.ARC_BAND_TARGET})
    return {"config": cfg, "seed_bases": list(seed_bases), "B": cell["B"],
            "w_fixed": list(cell["w"]), "rows": rows, "draws": draws,
            "reached_target": all(d["reached_target"] for d in draws)}


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
                dup = (seen[k], tuple(s["seed_bases"]), k)
                return False, dup
            seen[k] = tuple(s["seed_bases"])
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
    genes = tb.load_genes("best_solution.json")

    # STEP 1 -- draw the fit and hold-out band per config.  If a fit stream comes back
    # one-class, the reserved amendment seed is drawn and the amendment is recorded --
    # never silently, and the hold-out is never touched by this decision.
    streams, amendments = {}, {}
    for cfg in configs:
        fit_s = draw_band_stream(cfg, cells[cfg], SEEDS[cfg]["fit"], genes)
        cb = class_balance(fit_s["rows"])
        if cb["folds"] == 0 or cb["clean"] == 0:
            extra = AMENDMENT_FIT_SEED[cfg]
            fit_s = draw_band_stream(cfg, cells[cfg],
                                     SEEDS[cfg]["fit"] + (extra,), genes)
            amendments[cfg] = (f"registered fit seed {SEEDS[cfg]['fit']} came back "
                              f"one-class ({cb}); pooled with reserved seed {extra}")
        streams[cfg] = {
            "fit": fit_s,
            "holdout": draw_band_stream(cfg, cells[cfg], SEEDS[cfg]["holdout"], genes),
        }
    flat = [s for cfg in configs for s in streams[cfg].values()]
    disjoint, dup = streams_disjoint(flat)

    # STEPS 2-4 -- fit, freeze, score once; then the swap as an independent replicate.
    results = {}
    for cfg in configs:
        fit_s, hold_s = streams[cfg]["fit"], streams[cfg]["holdout"]
        fitted_fwd = fit_rule(fit_s["rows"])
        frozen = fitted_fwd["rule"]
        fitted_rev = fit_rule(hold_s["rows"])
        results[cfg] = {
            "frozen_cell": {"B": cells[cfg]["B"], "w": cells[cfg]["w"]},
            "seed_bases": {"fit": list(fit_s["seed_bases"]),
                          "holdout": list(SEEDS[cfg]["holdout"])},
            "amendment": amendments.get(cfg),
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

    # STEP 6 -- the reference on PART 9's own committed band, labelled outside the
    # protocol.  Only `coarse` carries a committed `arc_span_band` section -- PART 8/9's
    # own driver runs it at the first config alone.
    with open(triblock_artifact) as fh:
        committed = json.load(fh)
    for cfg in configs:
        band = committed["per_config"][cfg].get("arc_span_band")
        if not band or not band.get("genomes"):
            continue
        results[cfg]["corroboration_committed_band"] = {
            "seed_base": tb.GENOME_SWEEP_SEED + 1000, "outside_the_protocol": True,
            **score_frozen(results[cfg]["forward"]["frozen_rule"], band["genomes"])}

    rec = {
        "protocol": {
            "label": "not curved_valid (no (bend, interior point) reaches this region)",
            "family": "arc > t_wide OR (arc > t_conj AND min_wedge < t_wedge)",
            "grid": "midpoints of consecutive distinct FIT-stream values + inf",
            "selection": "max fit accuracy; ties: fewer fires, then lexicographic",
            "order": "fit -> freeze -> score holdout once -> swap",
            "sampler": "sweep_arc_span_band, conditioned on arc_span_deg > 30 (PLAN §72)",
            "amendment": (amendments or
                         "none needed: every registered fit stream had both classes"),
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
    # Named without a count: unlike the first registration's fixed six, an amendment
    # seed can bring this to five or six depending on how many configs needed one.
    checks["all_streams_pairwise_disjoint"] = disjoint
    checks["duplicate_gene_vectors"] = dup
    checks["band_streams_reached_their_target"] = all(
        s["reached_target"] for s in flat)
    empty = {f"{cfg}/{role}": class_balance(s["rows"])
             for cfg in configs for role, s in streams[cfg].items()
             if class_balance(s["rows"])["folds"] == 0
             or class_balance(s["rows"])["clean"] == 0}
    checks["fit_streams_have_both_classes"] = all(
        class_balance(streams[cfg]["fit"]["rows"])["folds"] > 0
        and class_balance(streams[cfg]["fit"]["rows"])["clean"] > 0
        for cfg in configs)
    checks["stream_with_one_class_only"] = empty or None
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
    print("  THE TRI-BLOCK'S FOLD RULE, CALIBRATED -- fit, freeze, score disjoint")
    print("=" * 78)
    print("\n  label: " + rec["protocol"]["label"])
    print("  family: " + rec["protocol"]["family"])
    print("  sampler: " + rec["protocol"]["sampler"])
    if rec["protocol"]["amendment"] != "none needed: every registered fit stream had both classes":
        print(f"  AMENDMENT: {rec['protocol']['amendment']}")
    for cfg, r in rec["per_config"].items():
        cb = r["class_balance"]
        print(f"\n  {cfg.upper()}   B = {r['frozen_cell']['B']}   "
              f"w_fixed = ({r['frozen_cell']['w'][0]:.3f}, "
              f"{r['frozen_cell']['w'][1]:.3f}, {r['frozen_cell']['w'][2]:.3f})")
        print(f"    streams: fit seed base(s) {r['seed_bases']['fit']} "
              f"(folds {cb['fit']['folds']}/{cb['fit']['n']}), "
              f"hold-out seed base {r['seed_bases']['holdout']} "
              f"(folds {cb['holdout']['folds']}/{cb['holdout']['n']})")
        for d in r["streams"]["fit"]["draws"] + r["streams"]["holdout"]["draws"]:
            print(f"      draw seed_base {d['seed_base']}: n_drawn {d['n_drawn']}, "
                  f"n_in_band {d['n_in_band']}, n_meshable {d['n_meshable']}, "
                  f"reached target: {d['reached_target']}")
        for tag, direction in (("FORWARD (fit -> hold-out, scored once)", "forward"),
                               ("SWAPPED (fit on the hold-out draw -> fit draw)",
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
                      f"{cx['min_wedge_deg']:.3f} deg, bow/width "
                      f"{cx['bow_over_width']:.4f} — the construction is CLEAN there")
            for m in h["missed_folds"]:
                print(f"        MISSED FOLD: arc {m['arc_span_deg']:.3f} deg, "
                      f"min wedge {m['min_wedge_deg']:.3f} deg, bow/width "
                      f"{m['bow_over_width']:.4f}")
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
        cor = r.get("corroboration_committed_band")
        if cor:
            print(f"\n    REFERENCE on PART 9's committed band (seed base "
                  f"{cor['seed_base']}) — outside the protocol, quoted only against the "
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
