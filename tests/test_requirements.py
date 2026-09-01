"""The requirements layer: it must reach the physics, and it must not move a default.

MBSE_PLAN.md.  PLAN.md §97.

TWO CLAIMS THAT PULL IN OPPOSITE DIRECTIONS, AND BOTH ARE GATED HERE.

  1  **NOTHING MOVED.**  `objective(genes, req=Requirements.baseline())` is bit-identical
     to `objective(genes)` — the scalar, all 14 gradient components and every one of the
     14 breakdown terms.  A default that moved is a silent re-interpretation of every
     committed artifact on disk and of the five study drivers that re-alias
     `SERVICE_FORCE_N` (`study_gnl.py:106`, `study_contact.py:94`,
     `study_gradient.py:120`, `study_fillet_cost.py:115`, `study_svk_rescore.py:67`).

  2  **AND YET IT REACHES.**  Two requirement sets differing ONLY in
     `allowable_stress_mpa` must give different `stress`/`stress_margin` IN THE SAME
     INTERPRETER, and likewise for `target_deflection_mm` and `deflection`.  This is a
     CACHE AUDIT BY TEST AND NOT BY READING: `_T1_CACHE` keys on
     `(cfg.name, span_mm, flanks, _t1_weights_key(weights))` (`wheel_objective.py:908`),
     `_KT_CACHE` keys without weights (:533) and `wheel_wheel._COORD_FN_CACHE` (:2760)
     keys on the static mesh recipe.  A stale jit trace returning the old answer is
     exactly the failure these two tests exist for, and it is invisible to inspection.

  1 without 2 is a layer that changes nothing because it is not wired.  2 without 1 is a
  layer that silently rescored the tree.  Neither alone is worth having.

Everything runs at `smoke` with 4 phases, which is the smallest mesh that still solves —
these are contract tests, not convergence tests, and the claims are exact equalities that
do not get truer on a finer mesh.
"""

import json
import os
from dataclasses import replace

import numpy as np
import pytest

import wheel_fea as W
import wheel_genome as wg
import wheel_objective as WO
import wheel_requirements as R

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG, N_PHASE = "smoke", 4


@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(HERE, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


@pytest.fixture(scope="module")
def shipped_record():
    with open(os.path.join(HERE, "best_solution.json")) as fh:
        return json.load(fh)


def _evaluate(genes, **kw):
    phases = WO.phase_stencil(n_phase=N_PHASE, scheme="uniform")
    return WO.objective(genes, CFG, phases=phases, **kw)


# ---------------------------------------------------------------------------
# STEP 1 — the baseline IS the constants
# ---------------------------------------------------------------------------

def test_baseline_reproduces_every_shipped_constant_exactly():
    req = R.Requirements.baseline()
    assert req.force_n == W.TOTAL_FORCE_NEWTONS
    assert req.target_deflection_mm == W.TARGET_DEFLECTION_MM
    assert req.allowable_stress_mpa == W.ALLOWABLE_STRESS_MPA
    assert req.min_wall_mm == W.MIN_WALL_MM
    assert req.e_mpa == W.YOUNGS_MODULUS_PLA_MPA
    assert req.nu == 0.35


def test_baseline_weights_are_DEFAULT_WEIGHTS_key_for_key():
    """`==` on the dict, not on a subset: a weight the baseline forgot would be a term
    the optimiser stopped seeing, and a weight it added would be one nothing reads."""
    assert R.Requirements.baseline().weights == WO.DEFAULT_WEIGHTS


def test_the_implied_mission_derives_the_shipped_constants():
    """Step 0's claim, gated.  1e-12 relative and not `==`: `min_wall` is `0.4 * 3`,
    which is 1.2000000000000002, and the finding is that the derivation is right — not
    that IEEE 754 multiplication is exact."""
    m = R.Mission.implied_baseline()
    req = R.Requirements.from_mission(m)
    for got, want in ((req.force_n, W.TOTAL_FORCE_NEWTONS),
                      (req.target_deflection_mm, W.TARGET_DEFLECTION_MM),
                      (req.allowable_stress_mpa, W.ALLOWABLE_STRESS_MPA),
                      (req.min_wall_mm, W.MIN_WALL_MM),
                      (req.e_mpa, W.YOUNGS_MODULUS_PLA_MPA)):
        assert abs(got - want) <= 1e-12 * abs(want)


def test_the_implied_sink_rate_round_trips_exactly():
    """The one field `Mission.implied_baseline` SOLVES rather than states."""
    m = R.Mission.implied_baseline()
    assert m.force_n == W.TOTAL_FORCE_NEWTONS


# ---------------------------------------------------------------------------
# STEP 1 — a budget that does not bind is not a budget
# ---------------------------------------------------------------------------

def test_the_five_priority_axes_are_exactly_OBJECTIVE_TERMS():
    """Read and not retyped — same names, same order.  A term promoted from objective to
    barrier changes this on its next import, which is the correct failure."""
    assert R.priority_axes() == tuple(WO.OBJECTIVE_TERMS)


@pytest.mark.parametrize("points, why", [
    ({"deflection": 50, "mass": 40, "stress_margin": 5, "smoothness": 4,
      "phase_ripple": 0}, "sums to 99"),
    ({"deflection": 50, "mass": 50, "stress_margin": 5, "smoothness": 0,
      "phase_ripple": 0}, "sums to 105"),
    ({"deflection": 110, "mass": -10, "stress_margin": 0, "smoothness": 0,
      "phase_ripple": 0}, "negative points"),
    ({"deflection": 50, "mass": 50}, "missing three axes"),
])
def test_priorities_reject_an_allocation_that_is_not_a_budget(points, why):
    with pytest.raises(ValueError):
        R.Priorities(points)


def test_points_may_not_reach_a_barrier_term():
    """A barrier is a `shall` whose only admissible value is zero.  You cannot buy your
    way out of a mesh that does not integrate."""
    pts = {a: 0.0 for a in R.priority_axes()}
    pts["deflection"] = 100.0
    pts["hub_overlap"] = 0.0                       # a BARRIER_TERM
    with pytest.raises(ValueError, match="objective terms"):
        R.Priorities(pts)


# ---------------------------------------------------------------------------
# STEP 2 — the thermal knockdown
# ---------------------------------------------------------------------------

def test_retention_is_exactly_one_at_20c():
    """EXACTLY, so the baseline is untouched by construction rather than by luck."""
    card = R.PLA_FFF
    assert card.e_retention(20.0) == 1.0
    assert card.sigma_retention(20.0) == 1.0
    assert card.e_mpa(20.0) == W.YOUNGS_MODULUS_PLA_MPA
    assert card.allowable_stress_mpa(20.0, R.REFERENCE_LANDINGS) == W.ALLOWABLE_STRESS_MPA


@pytest.mark.parametrize("what", ["e_retention_anchors", "sigma_retention_anchors"])
def test_retention_is_monotone_non_increasing_on_the_anchor_grid(what):
    ys = [a[1] for a in getattr(R.PLA_FFF, what)]
    assert all(b <= a for a, b in zip(ys, ys[1:]))


def test_above_t_max_service_is_refused_and_not_extrapolated():
    """A linear extrapolation of a collapsing modulus is not a knockdown, it is a guess
    with a sign."""
    with pytest.raises(ValueError, match="t_max_service_c"):
        R.PLA_FFF.e_retention(R.PLA_FFF.t_max_service_c + 0.1)
    with pytest.raises(ValueError, match="coldest"):
        R.PLA_FFF.sigma_retention(-100.0)


def test_a_material_card_that_gets_stiffer_when_heated_is_refused():
    with pytest.raises(ValueError, match="monotone"):
        replace(R.PLA_FFF,
                e_retention_anchors=((20.0, 1.0), (40.0, 1.1), (60.0, 0.2)))


def test_hotter_is_softer_and_weaker_and_colder_is_neither():
    card = R.PLA_FFF
    assert card.e_mpa(40.0) < card.e_mpa(20.0) < card.e_mpa(0.0)
    a = card.allowable_stress_mpa
    assert a(40.0, 1000) < a(20.0, 1000) < a(0.0, 1000)


def test_a_longer_life_only_ever_costs_allowable_stress():
    prev = None
    for n in (100, 1000, 10_000, 100_000, 1_000_000):
        v = R.PLA_FFF.allowable_stress_mpa(20.0, n)
        if prev is not None:
            assert v <= prev
        prev = v
    assert R.safety_factor(R.REFERENCE_LANDINGS) == W.SAFETY_FACTOR


# ---------------------------------------------------------------------------
# STEP 3 — THE LOAD-BEARING TEST: nothing moved
# ---------------------------------------------------------------------------

def test_req_baseline_is_bit_identical_to_naming_no_requirements(genes):
    """The whole arc rests on this one.  Scalar, all 14 gradient components, all 14
    breakdown terms — `==`, not `approx`."""
    v0, g0, b0 = _evaluate(genes)
    v1, g1, b1 = _evaluate(genes, req=R.Requirements.baseline())
    assert v1 == v0
    assert np.array_equal(np.asarray(g1), np.asarray(g0))
    for term in WO.TERMS:
        assert b1["terms"][term]["value"] == b0["terms"][term]["value"], term
        assert np.array_equal(np.asarray(b1["terms"][term]["grad_norm"]),
                              np.asarray(b0["terms"][term]["grad_norm"])), term


def test_the_two_new_keywords_default_to_the_module_constants(genes):
    """Named explicitly, at today's values, they must also change nothing."""
    v0, g0, _ = _evaluate(genes)
    v1, g1, _ = _evaluate(genes,
                          target_deflection_mm=WO.TARGET_DEFLECTION_MM,
                          allowable_stress_mpa=WO.ALLOWABLE_STRESS_MPA)
    assert v1 == v0 and np.array_equal(np.asarray(g1), np.asarray(g0))


def test_req_together_with_a_keyword_it_sets_is_refused_not_resolved(genes):
    """A precedence rule is a rule someone has to remember at the call site, and getting
    it wrong scores a design against a requirement nobody chose."""
    req = R.Requirements.baseline()
    for kw in ({"force": 50.0}, {"weights": dict(WO.DEFAULT_WEIGHTS)},
               {"target_deflection_mm": 1.0}, {"allowable_stress_mpa": 10.0},
               {"E": 1000.0}, {"nu": 0.3}):
        with pytest.raises(ValueError, match="req="):
            _evaluate(genes, req=req, **kw)


# ---------------------------------------------------------------------------
# STEP 3 — AND YET IT REACHES.  The cache audit, by test.
# ---------------------------------------------------------------------------

def test_moving_the_allowable_moves_the_stress_terms_in_the_same_interpreter(genes):
    base = R.Requirements.baseline()
    softer = replace(base, allowable_stress_mpa=base.allowable_stress_mpa * 0.5)
    _, _, b0 = _evaluate(genes, req=base)
    _, _, b1 = _evaluate(genes, req=softer)
    assert b1["terms"]["stress_margin"]["value"] > b0["terms"]["stress_margin"]["value"]
    assert b1["report"]["stress_utilisation_hub"] > b0["report"]["stress_utilisation_hub"]
    # and back again, in the SAME interpreter: a cache that keyed on the first call
    # would now return the second call's answer.
    _, _, b2 = _evaluate(genes, req=base)
    assert b2["terms"]["stress_margin"]["value"] == b0["terms"]["stress_margin"]["value"]


def test_moving_the_stroke_target_moves_the_deflection_term(genes):
    base = R.Requirements.baseline()
    longer = replace(base, target_deflection_mm=base.target_deflection_mm * 2.0)
    _, _, b0 = _evaluate(genes, req=base)
    _, _, b1 = _evaluate(genes, req=longer)
    assert b1["terms"]["deflection"]["value"] != b0["terms"]["deflection"]["value"]
    assert b1["report"]["target_deflection_mm"] == longer.target_deflection_mm


def test_moving_the_force_moves_the_solved_wheel(genes):
    base = R.Requirements.baseline()
    heavy = replace(base, force_n=base.force_n * 1.5)
    _, _, b0 = _evaluate(genes, req=base)
    _, _, b1 = _evaluate(genes, req=heavy)
    assert b1["report"]["axle_drop_mean_mm"] > b0["report"]["axle_drop_mean_mm"]
    assert b1["report"]["service_force_n"] == heavy.force_n


def test_moving_the_modulus_moves_the_solved_wheel_and_the_buckling_barrier(genes):
    """`buckling` is a BARRIER and it read `W.FORCE_PER_SPOKE_NEWTONS` and the 20 C
    modulus off the module until MBSE_PLAN Step 3 — so a hot or heavy requirement set was
    checked for buckling AT THE SHIPPED LOAD, silently."""
    base = R.Requirements.baseline()
    hot = replace(base, e_mpa=base.e_mpa * 0.5)
    _, _, b0 = _evaluate(genes, req=base)
    _, _, b1 = _evaluate(genes, req=hot)
    assert b1["report"]["axle_drop_mean_mm"] > b0["report"]["axle_drop_mean_mm"]
    assert b1["report"]["buckling_ratio"] != b0["report"]["buckling_ratio"]

    heavy = replace(base, force_n=base.force_n * 3.0)
    _, _, b2 = _evaluate(genes, req=heavy)
    assert b2["report"]["buckling_ratio"] > b0["report"]["buckling_ratio"]


def test_a_bare_buckling_ratio_call_is_the_call_it_has_always_been(genes):
    """The two new arguments default to the module constants, so M0's golden number is
    reachable without naming them."""
    assert WO._buckling_ratio(genes) == WO._buckling_ratio(
        genes, force=W.TOTAL_FORCE_NEWTONS,
        youngs_modulus=W.YOUNGS_MODULUS_PLA_MPA)


# ---------------------------------------------------------------------------
# STEP 3 — the requirements reach the POOL WORKERS
# ---------------------------------------------------------------------------

def test_pooled_equals_serial_under_a_non_baseline_requirement_set(genes):
    """MBSE_PLAN Step 3's third check, as a test rather than as a `make` invocation.

    `force`, `E` and `nu` are the three routed quantities that must survive being PICKLED
    into a worker process (`wheel_objective.py:1127` ships `problem_kw`).  Defaulted
    there, the pooled arm would score the shipped mission while the serial arm scored the
    given one — and this comparison is exact, not to a tolerance, which is what
    `wheel_pool.PINNED_ENV` and the Makefile's `XLA_FLAGS` line buy.
    """
    import wheel_pool as WP
    import wheel_stage3 as S3

    base = R.Requirements.baseline()
    req = replace(base, force_n=base.force_n * 1.37, e_mpa=base.e_mpa * 0.83,
                  allowable_stress_mpa=base.allowable_stress_mpa * 0.71,
                  target_deflection_mm=base.target_deflection_mm * 1.19)
    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    z = wg.normalize(genes, low, high)
    phases = WO.phase_stencil(n_phase=N_PHASE, scheme="uniform")
    import wheel_wheel as WW
    ori = tuple(float(o) for o in WW.flank_orientation(genes, WW.get_config(CFG)))

    ev = S3.Evaluator(CFG, orientation=ori, req=req)
    v_ser, g_ser, b_ser = ev(z, low, high, phases=phases)

    pool = WP.PhasePool(2)
    try:
        ev_p = S3.Evaluator(CFG, orientation=ori, pool=pool, req=req)
        v_pool, g_pool, b_pool = ev_p(z, low, high, phases=phases)
    finally:
        pool.close()

    assert v_pool == v_ser
    assert np.array_equal(np.asarray(g_pool), np.asarray(g_ser))
    for term in WO.TERMS:
        assert b_pool["terms"][term]["value"] == b_ser["terms"][term]["value"], term
    # and it really was a different mission from the default one
    assert b_ser["report"]["service_force_n"] == req.force_n
    assert b_ser["report"]["allowable_stress_mpa"] == req.allowable_stress_mpa


# ---------------------------------------------------------------------------
# STEP 4 — the calibration
# ---------------------------------------------------------------------------

def test_the_map_is_an_identity_at_its_own_calibration_point(shipped_record):
    """A map that is not an identity at its anchor is not a re-parameterisation, it is a
    change.  `1e-12` and not `==`: `w * (c/S) * (S/100) * 100 / d^2` is a multiply-divide
    round trip and lands a couple of ulp out on two of the five terms."""
    sm = shipped_record["loss_terms"]["smoothness"]
    p, _ = R.calibrated_priorities(sm)
    w = R.weights_from_priorities(p, sm)
    assert set(w) == set(WO.DEFAULT_WEIGHTS)
    for k, v in WO.DEFAULT_WEIGHTS.items():
        assert w[k] == pytest.approx(v, rel=1e-12, abs=1e-15), k


def test_the_calibration_reproduces_the_portfolio_the_plan_states(shipped_record):
    """51.35 / 42.80 / 5.56 / 0.29 / 0.00.  If this moves, MBSE_PLAN.md is wrong and the
    code is right — but somebody has to be told."""
    p, _ = R.calibrated_priorities(shipped_record["loss_terms"]["smoothness"])
    assert p.points["mass"] == pytest.approx(51.35, abs=0.01)
    assert p.points["deflection"] == pytest.approx(42.80, abs=0.01)
    assert p.points["stress_margin"] == pytest.approx(5.56, abs=0.01)
    assert p.points["smoothness"] == pytest.approx(0.29, abs=0.01)
    assert p.points["phase_ripple"] == 0.0


def test_total_exchange_rate_pressure_is_invariant_under_any_reallocation(shipped_record):
    """THE CONSERVATION LAW, and it is why the budget is 100.  Weights are not scale-free
    here: barriers are absolute, so scaling every objective weight halves every `shall`."""
    sm = shipped_record["loss_terms"]["smoothness"]
    axes = R.priority_axes()
    base = sum(R.reference_costs(sm).values())
    rng = np.random.default_rng(0)
    draws = [{a: (100.0 if a == k else 0.0) for a in axes} for k in axes]
    for _ in range(8):
        x = rng.random(len(axes))
        draws.append(dict(zip(axes, 100.0 * x / x.sum())))
    for pts in draws:
        w = R.weights_from_priorities(R.Priorities(pts), sm)
        assert sum(R.reference_costs(sm, w).values()) == pytest.approx(base, abs=1e-12)
        for b in WO.BARRIER_TERMS:
            assert w[b] == WO.DEFAULT_WEIGHTS[b], b


def test_the_ripple_axis_is_reachable_even_though_its_default_weight_is_zero(shipped_record):
    """The obvious map `w = w_default * p / p_cal` is `0/0` here.  Stated as cost per
    point it is not, and it is 0.0 at 0 points — which is the identity the test above
    needs."""
    sm = shipped_record["loss_terms"]["smoothness"]
    axes = R.priority_axes()
    pts = {a: 0.0 for a in axes}
    pts["phase_ripple"], pts["mass"] = 50.0, 50.0
    w = R.weights_from_priorities(R.Priorities(pts), sm)
    assert w["phase_ripple"] > 0.0 and np.isfinite(w["phase_ripple"])

    p_cal, _ = R.calibrated_priorities(sm)
    assert R.weights_from_priorities(p_cal, sm)["phase_ripple"] == 0.0


def test_a_priority_set_actually_moves_the_weight_the_objective_uses(genes, shipped_record):
    sm = shipped_record["loss_terms"]["smoothness"]
    axes = R.priority_axes()
    pts = {a: 0.0 for a in axes}
    pts["mass"] = 100.0
    req = R.Requirements.from_mission(R.Mission.implied_baseline(),
                                      R.Priorities(pts), sm)
    _, _, b = _evaluate(genes, req=req)
    assert b["terms"]["deflection"]["value"] == 0.0
    assert b["terms"]["mass"]["value"] > 0.0


# ---------------------------------------------------------------------------
# STEP 5 — the compliance table
# ---------------------------------------------------------------------------

def test_the_shall_rows_are_exactly_BARRIER_TERMS_and_the_shoulds_OBJECTIVE_TERMS(
        shipped_record):
    t = R.verify(shipped_record, R.Requirements.baseline())
    shall = [r for r in t["rows"] if r["kind"] == "shall"]
    should = [r for r in t["rows"] if r["kind"] == "should"]
    assert [r["quantity"].split(".")[-1] for r in shall] == list(WO.BARRIER_TERMS)
    assert len(should) == len(WO.OBJECTIVE_TERMS)
    assert [r["id"] for r in should] == [
        "SHOULD-%s" % t_.upper().replace("_", "-") for t_ in WO.OBJECTIVE_TERMS]


def test_a_missed_should_never_makes_a_design_non_compliant(shipped_record):
    """THE WHOLE SHALL/SHOULD DISTINCTION.  The shipped wheel misses two `should`s — it
    is over `MASS_REFERENCE_G` and over the margin knee — and it ships."""
    t = R.verify(shipped_record, R.Requirements.baseline())
    missed = [r["id"] for r in t["rows"] if r["verdict"] == "MISSED"]
    assert missed, "expected the shipped wheel to miss at least one `should`"
    assert t["compliant"]


def test_a_breached_shall_makes_a_design_non_compliant(shipped_record):
    bad = json.loads(json.dumps(shipped_record))
    bad["loss_terms"]["hub_overlap"] = 1e-9
    t = R.verify(bad, R.Requirements.baseline())
    assert not t["compliant"]
    assert [r["verdict"] for r in t["rows"] if r["id"] == "SHALL-HUB-OVERLAP"] == ["FAIL"]


def test_a_record_scored_under_other_requirements_is_refused(shipped_record):
    """`warn_if_stale`'s discipline, applied to requirements — and it matters more,
    because a stale STEP file looks wrong and a stale utilisation looks like a number."""
    base = R.Requirements.baseline()
    rec = dict(shipped_record, requirements=base.as_dict())
    R.verify(rec, base)                                    # matched: fine
    other = replace(base, force_n=base.force_n * 2.0)
    with pytest.raises(ValueError, match="REFUSED"):
        R.verify(rec, other)
    assert "MISMATCH" in R.verify(rec, other, strict=False)["provenance"]


def test_a_record_with_no_requirements_block_is_reported_and_not_refused(shipped_record):
    """Every artifact committed before this arc is such a record."""
    t = R.verify(shipped_record, R.Requirements.baseline())
    assert "unstated" in t["provenance"]


# ---------------------------------------------------------------------------
# STEP 6 — provenance
# ---------------------------------------------------------------------------

def test_req_hash_ignores_provenance_but_not_a_number():
    """Two missions that derive the same constants are the same requirement for the
    optimiser; one number apart they are not."""
    a = R.Requirements.baseline()
    b = replace(a, provenance={"anything": "else"})
    assert a.req_hash() == b.req_hash()
    assert replace(a, force_n=a.force_n + 1e-6).req_hash() != a.req_hash()
    assert replace(a, weights=dict(a.weights, mass=1.0)).req_hash() != a.req_hash()


def test_a_hand_edited_requirements_file_is_refused_on_load(tmp_path):
    req = R.Requirements.baseline()
    path = tmp_path / "req.json"
    req.save(str(path))
    assert R.load(str(path)).req_hash() == req.req_hash()

    d = json.loads(path.read_text())
    d["force_n"] = d["force_n"] * 2.0
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="req_hash"):
        R.load(str(path))


def test_a_requirements_run_records_its_hash_beside_the_box():
    """`search_block` carries `min_wall_mm` and `cy_bound_mm` to say what SPACE a genome
    is a boundary optimum of; `req_hash` says what PROBLEM it is the optimum of."""
    import argparse
    import wheel_stage3 as S3
    args = argparse.Namespace(
        optimizer="adam", config="coarse", steps=5, phase_scheme="uniform",
        n_phase=8, seed=0, start="best", kinematics="svk")
    req = R.Requirements.baseline()
    assert S3.search_block(args, "l", 1)["req_hash"] is None
    b = S3.search_block(args, "l", 1, req=req, requirements_file="req.json")
    assert b["req_hash"] == req.req_hash()
    assert b["requirements_file"] == "req.json"


def test_the_min_wall_floor_reaches_the_gene_box_and_comes_back():
    """`apply_process` is `set_min_wall`, which rewrites `GENE_SPACE` and re-snapshots the
    three arrays the GA clips against — the only way the floor moves inside one
    interpreter at all."""
    old = float(W.MIN_WALL_MM)
    try:
        req = replace(R.Requirements.baseline(), min_wall_mm=1.6)
        req.apply_process()
        assert W.MIN_WALL_MM == 1.6
        low, _, _ = wg.bounds_arrays(W.GENE_SPACE)
        assert list(low[8:12]) == [1.6, 1.6, 1.6, 1.6]
    finally:
        W.set_min_wall(old)
    low, _, _ = wg.bounds_arrays(W.GENE_SPACE)
    assert list(low[8:12]) == [old] * 4


def test_the_two_threaded_requirements_are_read_at_call_time_not_bound_at_def_time():
    """`t3_terms` must resolve its two new keywords from the module INSIDE the body.

    Written as `allowable_stress_mpa=ALLOWABLE_STRESS_MPA` the default binds once, at
    import, and `tests/test_objective.py`'s
    `monkeypatch.setattr(WO, "ALLOWABLE_STRESS_MPA", 2.0)` — the only lever that file has
    for pushing the stress barrier off zero — stops reaching the term.  That test then
    asserts `0 == 0` and passes.  It is how this arc first wrote the signature and it
    went red, which is the only reason it is not still written that way.

    A signature assertion and not a solve: the finding IS the binding time, the
    behavioural half is already gated by `test_objective.py`'s product-rule test, and a
    second contact solve to re-check it would cost seconds per run forever.
    """
    import inspect
    sig = inspect.signature(WO.t3_terms).parameters
    for name in ("target_deflection_mm", "allowable_stress_mpa"):
        assert sig[name].default is None, (
            "%s's default is bound at `def` time, so a monkeypatched module global no "
            "longer reaches the term and every test that moves one silently stops "
            "testing anything" % name)


def test_min_wall_and_requirements_together_are_refused_before_either_takes_effect():
    """Both flags set the printable wall floor and the record can only carry one.

    THE ORDER IS THE WHOLE TEST.  The refusal detects `--min-wall` by comparing it
    against `W.MIN_WALL_MM`, so it has to run BEFORE `set_min_wall` — placed after, the
    comparison is false by construction, the refusal never fires and the requirement
    file silently wins a floor the user typed on the command line.  That is how it was
    first written.  A subprocess, because this is an `argparse` exit and because the
    point is that it happens before anything is descended.
    """
    import subprocess
    import sys
    req = R.Requirements.baseline()
    path = os.path.join(HERE, "tests", "_tmp_req_minwall.json")
    try:
        req.save(path)
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "src", "wheel_stage3.py"),
             "--requirements", path, "--min-wall", "1.6", "--steps", "1",
             "--config", "smoke"],
            cwd=HERE, capture_output=True, text=True, timeout=600)
    finally:
        if os.path.exists(path):
            os.remove(path)
    assert proc.returncode != 0, proc.stdout[-2000:]
    assert "--min-wall and --requirements" in proc.stderr, proc.stderr[-2000:]


# ---------------------------------------------------------------------------
# HYGIENE — the CAD interpreter must be able to import this
# ---------------------------------------------------------------------------

def test_importing_wheel_requirements_does_not_pull_in_jax():
    """Mirrors `wheel_genome.py`'s contract; `wheel_objective` is reached only through
    `_objective_module()`, inside function bodies."""
    import subprocess
    import sys
    code = ("import sys\n"
            "import wheel_requirements\n"
            "print('LEAKED:' + ','.join(m for m in ('jax', 'pygad', 'matplotlib')\n"
            "                           if m in sys.modules))\n")
    proc = subprocess.run([sys.executable, "-c", code],
                          cwd=os.path.join(HERE, "src"), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "LEAKED:", proc.stdout
