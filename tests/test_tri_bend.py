"""
Pins for `studies/study_tri_bend.py` — PLAN.md §56 successor 2, UNCAP_PLAN.md
STEP 3 RECORD, PART 4 -> PART 10.

WHY THIS FILE EXISTS.  PART 4 left a fixed (constant) `bend` reaching fewer genomes than
the per-genome ceiling, and warned against fitting a genome-dependent rule against the
per-genome argmax `bend` values because they are "argmaxes over a plateau and their
scatter is partly that."  This driver instead fits two ONE-parameter families —
CONSTANT and LINEAR-in-`bow_over_width` — against how many genomes a rule leaves valid
and clear, under the same fit/freeze/score-once discipline `study_tri_rule.py` uses, so
that whether genome-dependence helps at all is answered on data neither family has seen.

WHAT IS TESTED WITHOUT A MESH.  `_select`'s tie-broken argmax, `constant_rule`/
`linear_rule`'s pure arithmetic (including the zero-at-bow-zero guarantee the plan asks
for), and `streams_disjoint`.  `evaluate()` itself calls real geometry
(`study_tri_block.region`/`cell`) and is exercised on the shipped genome alone, which is
fast — the expensive part of this file's own driver is drawing sixteen fresh genomes per
stream, not evaluating one.

AND ONE TEST IS ABOUT SCOPE, as in the sibling files: nothing here is wired into the mesh
the tree builds, and no rule is adopted anywhere.
"""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import wheel_genome as wg              # noqa: E402
import wheel_objective as wo           # noqa: E402
import wheel_wheel as ww               # noqa: E402
import study_tri_block as tb           # noqa: E402
import study_tri_bend as tbnd          # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT = os.path.join(REPO, "studies", "study_tri_bend.json")


# ---------------------------------------------------------------------------
# 1. THE TWO FAMILIES — pure arithmetic
# ---------------------------------------------------------------------------

def test_constant_rule_ignores_bow():
    f = tbnd.constant_rule(0.4)
    assert f(0.0) == 0.4 and f(1.25) == 0.4


def test_linear_rule_is_zero_at_zero_bow_by_construction():
    """The plan's own requirement — a fat region needs no argument to keep `bend` at 0 —
    is a structural property of this family, not a fitted one."""
    for k in (0.0, 1.0, 5.0, 30.0):
        assert tbnd.linear_rule(k)(0.0) == 0.0


def test_linear_rule_clips_to_the_unit_interval():
    f = tbnd.linear_rule(10.0)
    assert f(0.05) == pytest.approx(0.5)
    assert f(1.0) == 1.0          # would be 10.0 unclipped
    assert f(-1.0) == 0.0         # never negative, even off a malformed bow


# ---------------------------------------------------------------------------
# 2. SELECTION — tie-broken argmax, no mesh required
# ---------------------------------------------------------------------------

def _ev(n_clear, n_valid, worst, n=16):
    return {"n": n, "dropped": 0, "n_valid": n_valid, "n_clear": n_clear,
            "worst_min_scaled_jacobian": worst}


def test_select_maximises_n_clear_first():
    out = tbnd._select([(0.0, _ev(10, 12, 0.1)), (2.0, _ev(13, 13, -0.5))])
    assert out["param"] == 2.0


def test_select_breaks_a_clear_tie_on_n_valid():
    out = tbnd._select([(0.0, _ev(10, 12, 0.1)), (2.0, _ev(10, 14, -0.5))])
    assert out["param"] == 2.0


def test_select_breaks_a_double_tie_on_worst_min_sj_then_smaller_param():
    out = tbnd._select([(4.0, _ev(10, 12, 0.05)), (2.0, _ev(10, 12, 0.05))])
    assert out["param"] == 2.0                           # smaller param wins the full tie
    out2 = tbnd._select([(2.0, _ev(10, 12, 0.01)), (4.0, _ev(10, 12, 0.20))])
    assert out2["param"] == 4.0                          # better margin wins first


# ---------------------------------------------------------------------------
# 3. EVALUATE — real geometry, but one genome is cheap
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shipped_genes():
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


@pytest.fixture(scope="module")
def cell(shipped_genes):
    with open(os.path.join(REPO, "studies", "study_tri_block.json")) as fh:
        rec = json.load(fh)
    best = rec["per_config"]["coarse"]["sweep"]["best"]
    return int(best["B"]), tuple(float(x) for x in best["w"])


def test_evaluate_reproduces_the_committed_control_at_bend_zero(shipped_genes, cell):
    """`bend=0` is the straight Y, and the committed `sector` block already measured its
    min scaled Jacobian at this exact (B, w) — this pins `evaluate` against that number
    rather than trusting it blind."""
    B, w = cell
    row = {"genes": [float(x) for x in shipped_genes], "bow_over_width": 0.0}
    ev = tbnd.evaluate([row], tbnd.constant_rule(0.0), w, B, "coarse")
    with open(os.path.join(REPO, "studies", "study_tri_block.json")) as fh:
        rec = json.load(fh)
    published = rec["per_config"]["coarse"]["sector"]["min_scaled_jacobian"]
    assert ev["n"] == 1 and ev["dropped"] == 0
    assert ev["worst_min_scaled_jacobian"] == pytest.approx(published, abs=1e-6)


def test_evaluate_drops_a_genome_it_cannot_build_rather_than_raising(cell):
    B, w = cell
    bad = {"genes": [1.0e6] * 14, "bow_over_width": 0.0}   # nonsense, must not raise
    ev = tbnd.evaluate([bad], tbnd.constant_rule(0.0), w, B, "coarse")
    assert ev["dropped"] == 1 and ev["n"] == 0


# ---------------------------------------------------------------------------
# 4. STREAMS — disjointness is asserted, not assumed
# ---------------------------------------------------------------------------

def _stream(seed, genes_list):
    return {"seed": seed,
            "rows": [{"genes": g, "bow_over_width": 0.1} for g in genes_list]}


def test_streams_disjoint_accepts_distinct_seeds_and_genomes():
    ok, dup = tbnd.streams_disjoint([
        _stream(1, [[1.0] * 14, [2.0] * 14]), _stream(2, [[3.0] * 14, [4.0] * 14])])
    assert ok and dup is None


def test_streams_disjoint_catches_a_shared_genome():
    shared = [5.0] * 14
    ok, dup = tbnd.streams_disjoint([
        _stream(1, [shared]), _stream(2, [[6.0] * 14, shared])])
    assert not ok and dup[0] == 1 and dup[1] == 2


# ---------------------------------------------------------------------------
# 5. SCOPE — nothing here is wired into the mesh the tree builds
# ---------------------------------------------------------------------------

def test_nothing_here_is_wired_into_the_mesh_the_tree_BUILDS(shipped_genes):
    assert ww.UNCAP_DEFAULT == (True, 1.0)
    blocks = ww.sector_blocks(shipped_genes, ww.get_config("coarse"))
    assert len([k for k in blocks if k != "_thetas"]) == 7
    assert "rim_tri_t" not in blocks
    src = open(os.path.join(REPO, "src", "wheel_wheel.py")).read()
    assert "study_tri_bend" not in src


# ---------------------------------------------------------------------------
# 6. THE ARTIFACT
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def report():
    with open(ARTIFACT) as fh:
        return json.load(fh)


def test_the_committed_run_passed_its_own_protocol_checks(report):
    assert report["self_checks"]["pass"] is True
    assert report["self_checks"]["streams_pairwise_disjoint"] is True
    assert report["self_checks"]["orientations_complete_everywhere"] is True
    assert report["self_checks"]["fit_streams_nonempty"] is True


def test_w_is_held_fixed_to_the_tri_blocks_own_published_cell(report):
    with open(os.path.join(REPO, "studies", "study_tri_block.json")) as fh:
        tri = json.load(fh)
    for cfg, r in report["per_config"].items():
        best = tri["per_config"][cfg]["sweep"]["best"]
        assert r["B"] == int(best["B"])
        assert r["w"] == pytest.approx([float(x) for x in best["w"]])


def test_the_linear_familys_bend_is_zero_at_the_shipped_genomes_own_bow(report):
    """The shipped genome's own `bow_over_width` is tiny (§56) — this is the artifact-
    level version of the structural guarantee `test_linear_rule_is_zero_at_zero_bow_by_
    construction` pins in the abstract."""
    for cfg, r in report["per_config"].items():
        s = r["forward"]["linear"]["shipped"]
        assert abs(s["bend"]) < 0.05, (cfg, s)


def test_a_negative_result_is_reported_not_hidden(report):
    """This file's whole point is that "the genome-dependent rule did not beat the
    constant" must be a visible outcome, not a reason to withhold the artifact."""
    lb = report["self_checks"]["linear_beats_constant_on_holdout"]
    assert set(lb) == set(report["per_config"])
    for cfg, (lin_clear, const_clear) in lb.items():
        assert isinstance(lin_clear, int) and isinstance(const_clear, int)
