"""
Pins for `studies/study_tri_rule.py` — PLAN.md §73 successor 1, UNCAP_PLAN.md
STEP 3 RECORD, PART 9 -> PART 10.

WHY THIS FILE EXISTS.  A conjunctive fold rule for the tri-block's curved-Y construction
circulated with an in-sample score of 1.000, which is one degree freer than the data it
sits on.  The driver calibrates it under a pre-registered protocol — fit on one band
stream, freeze the thresholds, score ONCE on a disjoint stream, swap — and this file pins
the four structural claims the protocol's honesty rests on:

  1. THE FAMILY IS WHAT THE DOCSTRING SAYS.  `predict` is the exact two-branch
     disjunction, with a sentinel infinity that disables a branch rather than faking a
     threshold nobody selected.
  2. THE GRID IS ANCHORED TO THE FIT STREAM ALONE.  Midpoints between consecutive
     DISTINCT values plus sentinels — so no threshold sits on its own anchor, and no
     hold-out value can enter the grid by construction.
  3. FITTING IS DETERMINISTIC AND BIASED AGAINST CRYING WOLF.  Same fit stream, same
     triple; ties broken toward fewer fires, then lexicographically.
  4. SCORING IS READ-ONLY UNDER THE FREEZE, AND NAMES ITS MISTAKES.  Per-branch fire
     counts, wide-branch counterexamples carried with their features, missed folds
     likewise — an averaged accuracy is exactly what this protocol exists to refuse.

EVERYTHING HERE RUNS ON SYNTHETIC ROWS except the two artifact pins and the scope pin,
so the suite stays seconds-fast and never re-draws a stream.

AND ONE TEST IS ABOUT SCOPE, as in `test_tri_block.py`: nothing here is wired into the
mesh the tree builds, and no rule is adopted anywhere.
"""

import json
import math
import os
import sys

import numpy as np              # noqa: F401  (kept for parity with sibling files)
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import wheel_genome as wg              # noqa: E402
import wheel_objective as wo           # noqa: E402
import wheel_wheel as ww               # noqa: E402
import study_tri_block as tb           # noqa: E402
import study_tri_rule as tr            # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT = os.path.join(REPO, "studies", "study_tri_rule.json")


def row(arc, min_wedge, curved_valid):
    """A band row, shaped like `study_tri_block._shape` + `curved_valid` + `genes` —
    what `sweep_arc_span_band` actually carries.  Only `min_wedge_deg` (not the full
    three-wedge triple) ever fed `features()`, so callers pass it directly."""
    return {"genes": [0.0] * 14,
            "arc_span_deg": arc, "min_wedge_deg": min_wedge,
            "curved_valid": curved_valid,
            "bow_over_width": 0.1, "turn_at_far_end_deg": 150.0,
            "wedge_sum_deg": 180.0, "wedge_sum_minus_180": 0.0, "A_over_C": 1.0}


# ---------------------------------------------------------------------------
# 1. THE RULE FAMILY
# ---------------------------------------------------------------------------

R = {"t_wide": 36.16, "t_conj": 30.0, "t_wedge": 17.12}


def test_predict_is_the_registered_two_branch_disjunction():
    assert tr.predict((40.0, 90.0), R) is True          # wide branch alone
    assert tr.predict((31.0, 10.0), R) is True          # conjunctive branch alone
    assert tr.predict((31.0, 20.0), R) is False         # past t_conj, wedge too fat
    assert tr.predict((29.0, 10.0), R) is False         # under both arc cuts
    assert tr.predict((36.16, 90.0), R) is False        # STRICT inequality on t_wide
    assert tr.predict((30.0, 5.0), R) is False          # strict on t_conj too


def test_the_sentinel_disables_its_branch_cleanly():
    wide_only = {"t_wide": 36.16, "t_conj": math.inf, "t_wedge": math.inf}
    assert tr.predict((40.0, 1.0), wide_only) is True
    assert tr.predict((31.0, 1.0), wide_only) is False   # conjunctive branch is OFF
    none_at_all = {"t_wide": math.inf, "t_conj": math.inf, "t_wedge": math.inf}
    assert all(tr.predict(f, none_at_all) is False
               for f in [(100.0, 1.0), (0.0, 0.0)])


def test_branch_of_names_which_clause_fired():
    assert tr.branch_of((40.0, 90.0), R) == "wide"
    assert tr.branch_of((31.0, 10.0), R) == "conjunctive"
    assert tr.branch_of((10.0, 10.0), R) is None
    assert tr.branch_of((40.0, 90.0),
                        {"t_wide": math.inf, "t_conj": math.inf,
                         "t_wedge": math.inf}) is None


# ---------------------------------------------------------------------------
# 2. THE GRID — anchored to whatever stream it is handed
# ---------------------------------------------------------------------------

def test_grid_takes_midpoints_of_consecutive_distinct_values():
    g = tr.threshold_grid([10.0, 20.0, 40.0])
    assert g[:2] == [15.0, 30.0]
    assert g[-1] == math.inf                            # the disabling sentinel
    assert len(g) == 3


def test_grid_collapses_duplicates_and_never_sits_on_a_value():
    g = tr.threshold_grid([5.0, 5.0, 5.0, 9.0])
    assert g == [7.0, math.inf]
    vals = [1.0, 2.0, 3.0]
    for t in tr.threshold_grid(vals):
        if math.isfinite(t):
            assert t not in vals                        # off its own anchors


# ---------------------------------------------------------------------------
# 3. FITTING — deterministic, parsimonious on ties
# ---------------------------------------------------------------------------

def _fit_rows():
    # Four folds at wide arcs, four clean at narrow ones, plus one clean wide genome
    # (a potential false positive) and one folded narrow one (a potential miss).
    return [
        row(41.0, 19.0, False),
        row(38.0, 18.0, False),
        row(45.0, 25.0, False),
        row(39.0, 21.0, False),
        row(5.0, 25.0, True),
        row(8.0, 30.0, True),
        row(11.0, 22.0, True),
        row(6.0, 26.0, True),
        row(37.0, 28.0, True),            # clean, inside the wide band
        row(12.0, 9.0, False),            # folded, narrow arc
    ]


def test_fitting_is_deterministic_on_the_same_fit_stream():
    a = tr.fit_rule(_fit_rows())
    b = tr.fit_rule(list(reversed(_fit_rows())))
    assert a["rule"] == b["rule"]                       # order-invariant too


def test_fitting_prefers_fewer_fires_among_equal_accuracy_rules():
    rows = [
        row(50.0, 19.0, False),
        row(10.0, 25.0, True),
    ]
    out = tr.fit_rule(rows)
    # Any t_conj/t_wedge pair that never fires scores the same as the bare wide cut;
    # the registered tie-break must land on rules that do NOT add spurious fires.
    fired = sum(tr.predict(tr.features(r), out["rule"]) for r in rows)
    assert fired == 1
    assert out["rule"]["t_wide"] == 30.0                # midpoint of {10, 50}


def test_in_sample_confusion_is_reported_as_in_sample():
    out = tr.fit_rule(_fit_rows())
    ins = out["in_sample"]
    assert {"tp", "fp", "tn", "fn", "n", "accuracy", "fires"} <= set(ins)
    assert ins["n"] == len(_fit_rows())


# ---------------------------------------------------------------------------
# 4. SCORING — read-only under the freeze, mistakes NAMED
# ---------------------------------------------------------------------------

FROZEN = {"t_wide": 36.16, "t_conj": 30.0, "t_wedge": 17.12}


def test_score_frozen_reports_branch_counts_and_names_counterexamples():
    holdout = [
        row(41.0, 19.0, False),           # wide TP
        row(37.5, 28.0, True),            # wide FP -> false fire
        row(31.0, 10.0, False),           # conjunctive TP
        row(32.0, 20.0, True),            # conjunctive FP -> false fire
        row(35.9, 30.0, False),           # missed fold
        row(4.0, 24.0, True),             # true negative
    ]
    s = tr.score_frozen(FROZEN, holdout)
    assert s["n"] == 6
    assert (s["tp"], s["fp"], s["tn"], s["fn"]) == (2, 1, 2, 1)
    assert s["per_branch"]["wide"] == {"tp": 1, "fp": 1}
    assert s["per_branch"]["conjunctive"] == {"tp": 1, "fp": 0}
    assert len(s["false_fires"]) == 1
    cx = s["false_fires"][0]
    assert cx["branch"] == "wide" and abs(cx["arc_span_deg"] - 37.5) < 1e-12
    assert len(s["missed_folds"]) == 1
    assert abs(s["missed_folds"][0]["arc_span_deg"] - 35.9) < 1e-12


def test_score_frozen_reports_a_conjunctive_false_fire_when_it_happens():
    loose = {"t_wide": math.inf, "t_conj": 30.0, "t_wedge": 25.0}
    s = tr.score_frozen(loose, [row(32.0, 20.0, True)])
    assert (s["fp"], s["per_branch"]["conjunctive"]["fp"]) == (1, 1)
    assert s["false_fires"][0]["branch"] == "conjunctive"


def test_score_frozen_drops_unlabelled_rows_and_counts_them():
    bad = row(40.0, 19.0, False)
    bad["curved_valid"] = None                          # the sweep could not say
    s = tr.score_frozen(FROZEN, [bad, row(4.0, 24.0, True)])
    assert s["unlabelled_dropped"] == 1 and s["n"] == 1
    assert (s["tp"], s["fp"], s["tn"], s["fn"]) == (0, 0, 1, 0)


def test_scoring_does_not_mutate_the_holdout_or_the_freeze():
    import copy
    holdout = [row(41.0, 19.0, False),
               row(4.0, 24.0, True)]
    before_rows, before_rule = copy.deepcopy(holdout), copy.deepcopy(FROZEN)
    tr.score_frozen(FROZEN, holdout)
    assert holdout == before_rows and FROZEN == before_rule


# ---------------------------------------------------------------------------
# 5. STREAMS — disjointness is asserted, not assumed
# ---------------------------------------------------------------------------

def _stream(seed_bases, arcs):
    # The gene vector encodes the seed too, so two streams with different seed bases can
    # never collide by construction of the fixture itself.
    return {"seed_bases": tuple(seed_bases),
            "rows": [{"genes": [float(seed_bases[0] * 1000 + a)] + [0.0] * 13,
                      "arc_span_deg": a, "min_wedge_deg": 25.0,
                      "curved_valid": a < 20.0} for a in arcs]}


def test_disjointness_accepts_six_distinct_streams():
    ok, dup = tr.streams_disjoint([
        _stream((1, 2), [5.0, 6.0]), _stream((3,), [5.0, 6.0]),
        _stream((4, 5), [5.0, 6.0]), _stream((6,), [5.0, 6.0]),
        _stream((7,), [5.0, 6.0]), _stream((8,), [5.0, 6.0])])
    assert ok and dup is None


def test_disjointness_catches_a_shared_genome_and_names_it():
    shared = _stream((9,), [7.0])
    other = _stream((2,), [8.0])
    ok, dup = tr.streams_disjoint([shared, other, shared])
    assert not ok
    assert dup[0] == (9,) and dup[1] == (9,)            # both sides name the same seeds


# ---------------------------------------------------------------------------
# 6. SCOPE — nothing here is wired into the mesh the tree builds
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


def test_nothing_here_is_wired_into_the_mesh_the_tree_BUILDS(genes):
    """Same pin as the tri-block's own file: a screen for successor work must not
    quietly become a change to what ships."""
    assert ww.UNCAP_DEFAULT == (True, 1.0)
    blocks = ww.sector_blocks(genes, ww.get_config("coarse"))
    assert len([k for k in blocks if k != "_thetas"]) == 7
    assert "rim_tri_t" not in blocks
    src = open(os.path.join(REPO, "src", "wheel_wheel.py")).read()
    assert "study_tri_rule" not in src


# ---------------------------------------------------------------------------
# 7. THE ARTIFACT
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def report():
    with open(ARTIFACT) as fh:
        return json.load(fh)


def test_the_committed_run_passed_its_own_protocol_checks(report):
    assert report["self_checks"]["pass"] is True
    assert report["self_checks"]["all_streams_pairwise_disjoint"] is True
    assert report["self_checks"]["band_streams_reached_their_target"] is True
    assert report["self_checks"]["fit_streams_have_both_classes"] is True


def test_the_artifact_carries_a_frozen_rule_per_config_and_scores_both_ways(report):
    for cfg, r in report["per_config"].items():
        fwd, swp = r["forward"], r["swapped"]
        for d, other in ((fwd, "holdout"), (swp, "fit")):
            ru = d["frozen_rule"]
            assert math.isfinite(ru["t_wide"]) or math.isfinite(ru["t_conj"])
            h = d["holdout_scored_once"]
            assert h["n"] == r["class_balance"][other]["n"]
            assert h["accuracy"] is not None


def test_the_informal_rule_is_scored_on_the_same_holdout_and_labelled(report):
    for cfg, r in report["per_config"].items():
        inf = r["forward"]["informal_rule_on_same_holdout"]
        assert inf["outside_the_protocol"] is True
        assert inf["rule"] == report["protocol"]["informal_rule"]
        h = r["forward"]["holdout_scored_once"]
        assert inf["n"] == h["n"]          # the same hold-out genomes


def test_the_corroboration_column_is_labelled_outside_the_protocol(report):
    """Only `coarse` carries a committed `arc_span_band` section — PART 8/9's own
    driver runs that sweep at the first config alone — so `medium` legitimately has no
    corroboration column rather than a stale or fabricated one."""
    found_one = False
    for cfg, r in report["per_config"].items():
        cor = r.get("corroboration_committed_band")
        if cor is None:
            continue
        found_one = True
        assert cor["outside_the_protocol"] is True
        assert cor["seed_base"] == tb.GENOME_SWEEP_SEED + 1000
    assert found_one


def test_the_label_is_the_curved_construction_on_the_conditioned_band(report):
    assert "not curved_valid" in report["protocol"]["label"]
    assert "sweep_arc_span_band" in report["protocol"]["sampler"]
    assert report["min_sj_target"] == wo.MIN_SJ_TARGET
