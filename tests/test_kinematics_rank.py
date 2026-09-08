"""
Pins for `study_kinematics_rank._rank_block` — R2's second clause, and the derived verdict
block both committed kinrank artifacts carry. PLAN.md §130 successor 2.

WHY THIS FILE EXISTS. The driver that answers KINEMATICS_PLAN.md step 0c's registered
criterion had no test of any kind, and §130 found the criterion quietly losing half of R2:
`_rank_block` slices `k = min(5, len(rows))` and reported `top5_sets_equal` as
`set(order_linear[:k]) == set(order_svk[:k])`, so **at n <= 5 both slices are the whole
subset and the sets are equal by construction** — for any pair of orderings, including
exactly reversed ones. R2's gate is `rho >= GATE_SPEARMAN and top5_sets_equal`, so below
n = 6 it silently stopped being two conditions and became a bare Spearman. It is not a
hypothetical: the filleted run (§130) cleared R2 on a binding subset of exactly five.

WHAT IS PINNED, and the second half is the one that would have caught this.

1. The clause ABSTAINS (`None`) where it cannot fail, and BINDS again at n = 6. The
   abstain case is asserted together with the fact that makes it vacuous — the two sets
   are equal while the two orderings are reversed — so this file states the defect it
   pins rather than only the repair.
2. Each committed artifact still REPRODUCES its own `verdict` block from its own stored
   `rank.rows`. The verdict is pure post-processing over rows the run measured, so a
   change to `_rank_block` that nobody re-derived leaves a driver that no longer describes
   its own output; nothing was watching. Cheap — no mesh, no solve, milliseconds — which
   is what makes it a test and not a study.

`study_kinematics_rank.json` IS THE CONTROL AND IT MUST COME BACK NULL. It is §32's
evidence and KINEMATICS_PLAN.md step 1's subject, its binding subset is ten, and the fix
above therefore cannot touch it: k = 5 is a proper subset there. Measured at the fix —
byte-identical verdict — and asserted here so that a later change to this block cannot
move a closed arc's record without a test saying so.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import study_kinematics_rank as KR       # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIES = os.path.join(REPO, "studies")

COMMITTED_ARTIFACTS = ("study_kinematics_rank.json",
                       "study_kinematics_rank_filleted.json")


def _rows(lin, svk):
    """The three fields `_rank_block` reads, and nothing else."""
    return [{"genome": f"g{i}", "linear": {"loss": float(a)}, "svk": {"loss": float(b)}}
            for i, (a, b) in enumerate(zip(lin, svk))]


def test_the_top5_clause_abstains_where_the_slice_is_the_whole_subset():
    """Five rows ranked exactly backwards: the sets are equal, the orders are opposite."""
    b = KR._rank_block(_rows([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), "feasible")

    assert b["n"] == 5
    assert b["top5_sets_equal"] is None
    # The vacuity itself, stated: this is what the old `bool(...)` reported as True.
    assert set(b["top5_linear"]) == set(b["top5_svk"])
    assert b["top5_linear"] != b["top5_svk"]
    # rho = -1 carries R2 alone, and it fails.
    assert b["spearman_rho"] == pytest.approx(-1.0)
    assert b["r2_pass"] is False


def test_r2_rests_on_the_spearman_alone_below_six():
    """The §130 shape: rho clears the bar, and nothing else was ever checked."""
    b = KR._rank_block(_rows([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]), "feasible")

    assert b["top5_sets_equal"] is None
    assert b["spearman_rho"] >= KR.GATE_SPEARMAN
    assert b["r2_pass"] is True


def test_the_top5_clause_binds_again_at_six():
    """One adjacent swap across the 5/6 boundary: rho = 0.9429 passes, the sets differ.

    The swap is chosen so the two clauses disagree — a gate that had collapsed to the
    Spearman would read PASS here.
    """
    b = KR._rank_block(_rows([1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 60, 50]), "feasible")

    assert b["n"] == 6
    assert b["spearman_rho"] >= KR.GATE_SPEARMAN
    assert b["top5_sets_equal"] is False
    assert b["r2_pass"] is False


@pytest.mark.parametrize("name", COMMITTED_ARTIFACTS)
def test_the_committed_kinrank_artifacts_reproduce_their_own_verdict_block(name):
    with open(os.path.join(STUDIES, name)) as fh:
        rep = json.load(fh)

    stored = rep["rank"]["verdict"]
    rebuilt = KR._verdict(rep["rank"]["rows"])

    assert json.dumps(rebuilt, sort_keys=True) == json.dumps(stored, sort_keys=True)


@pytest.mark.parametrize("name", COMMITTED_ARTIFACTS)
def test_exactly_one_block_binds_r2_and_it_is_the_one_the_verdict_took(name):
    """§132 successor 0: the artifact now says IN PLACE which `r2_pass` decided R2.

    `_rank_block` computes `r2_pass` for both subsets and `_verdict` reads one of them, so
    `study_kinematics_rank_filleted.json` carries `blocks.full.r2_pass: false` beside
    `registered_criterion.R2_rank_agreement: true` — both correct, and readable as a
    contradiction by anyone grepping the field.  The terminal output always drew the
    distinction ("BINDING for R2" against "diagnostic") and the file never did.

    Pinned as an IDENTITY between the marker and the verdict rather than as
    `r2_binds == (subset == "feasible")`, which would restate the constant and pass however
    wrong the wiring got.  If someone re-points `R2_BINDING_SUBSET`, the marker and
    `r2_rank_agreement` move together or this fails.
    """
    with open(os.path.join(STUDIES, name)) as fh:
        blocks = json.load(fh)["rank"]["verdict"]["blocks"]

    binding = [b for b in blocks.values() if b["r2_binds"]]
    assert len(binding) == 1, f"{len(binding)} blocks claim to bind R2"

    rebuilt = KR._verdict(json.load(open(os.path.join(STUDIES, name)))["rank"]["rows"])
    if not binding[0].get("insufficient"):
        assert rebuilt["r2_rank_agreement"] == binding[0]["r2_pass"]


def test_the_diagnostic_blocks_marker_does_not_claim_its_numbers_are_unmeasured():
    """`r2_binds: false` is about consumption, not about measurement.

    The full pool is NOT purely diagnostic — R1 reads its `argmin_identical` — so the
    marker had to be named for the one criterion the subset does not decide.  A reader who
    took it as "this block was not measured" would discard a real statistic, and R1 would
    then have no full-pool operand at all.
    """
    rows = _rows([1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 60, 50])
    full = KR._rank_block(rows, "full")

    assert full["r2_binds"] is False
    assert full["spearman_rho"] > 0 and full["n"] == 6
    assert "argmin_identical" in full, "R1's operand must survive the marker"
