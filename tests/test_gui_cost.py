"""Pins for the GUI's memory verdict on a Stage-3 descent -- PLAN.md §174, §173 successor 0.

NO TEST IMPORTED THE GUI BEFORE THIS FILE, and `gui/catalog.py`'s cost model with
`gui/jobs.plan`'s refusal is the one live consumer of a memory figure that ACTS on it
(§170 §4, row 11): it refused every pooled `coarse` descent the box runs and admitted the
`-1` pool it priced as serial.  Both halves were re-derived from `wheel_pool.POOL_GIB`, so
what is pinned here is the measured runs, not the arithmetic -- a model that refuses a pool
the box held, or prices one under what it held, goes red.  That holds for the shapes the GUI
launches, which pass no `--fidelity-check-every`: the check's second Evaluator took a
`medium` parent to 15.47 GiB (§174), past the 11 the pair carries.

Every reading is set: `plan` reads free and total memory from `jobs`, and `-1` is priced by
`wheel_pool.default_workers`, which reads its own.  `catalog` imports that same module
object, so patching `WP` reaches it.
"""

import os
import sys

import wheel_pool as WP

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "gui"))
import jobs  # noqa: E402

BOX_TOTAL_GIB = 61.37   # MemTotal of the box §170, §171 and §173 ran their pools on


def _plan(monkeypatch, available, **params):
    monkeypatch.setattr(WP, "_available_gib", lambda: available)
    monkeypatch.setattr(WP.os, "cpu_count", lambda: 24)
    monkeypatch.setattr(jobs, "available_gib", lambda: available)
    monkeypatch.setattr(jobs, "_total_gib", lambda: BOX_TOTAL_GIB)
    monkeypatch.setattr(jobs, "running_jobs", lambda: [])
    return jobs.plan("stage3", params)


def test_the_pools_the_box_held_are_admitted_priced_above_what_they_held(monkeypatch):
    """§171 §2: four `coarse` workers at `-1`'s own argv summed 49.719 GiB of kernel marks
    with 57.05 available; §170's `--workers 4` summed 49.613.  §173 §3: three `medium`
    workers summed 45.003 over 100 steps with 57.76 available.

    The affine model priced the first at 81.5 GiB and refused it, and x1.35 on a 55 GiB
    upper bound would cap it at 74.2 on a 61.37 GiB box, a cap that cannot bind.
    """
    for workers in (4, -1):
        pre = _plan(monkeypatch, 57.05, config="coarse", workers=workers)
        assert not pre["blockers"], pre["blockers"]
        assert pre["peak_gib"] == 55.0 > 49.719
        assert pre["memory_max_gib"] == 55.0, "an upper bound is its own cap"
    pre = _plan(monkeypatch, 57.76, config="medium", workers=-1)
    assert not pre["blockers"], pre["blockers"]
    assert pre["peak_gib"] == 47.0 > 45.003


def test_minus_one_is_priced_at_the_width_it_will_pick_not_as_serial(monkeypatch):
    """§170 §4: `-1` was priced as width 1, 43.4 GiB at `coarse`, while it launched four."""
    assert _plan(monkeypatch, 57.05, config="coarse", workers=-1)["peak_gib"] == 55.0
    assert _plan(monkeypatch, 40.0, config="coarse", workers=-1)["peak_gib"] == 11 + 2 * 11


def test_a_config_default_workers_refuses_is_blocked_with_its_reason(monkeypatch):
    """`fine` has no measured pair.  `-1` there raises wherever memory can be read (§167 §2),
    and the preview says so first rather than pricing it at a coarser rung's numbers."""
    pre = _plan(monkeypatch, 57.05, config="fine", workers=-1)
    assert any("no measured pool memory" in b for b in pre["blockers"]), pre["blockers"]


def test_an_explicit_count_is_the_callers_below_the_machine_and_refused_above_it(
        monkeypatch):
    """Four `medium` workers budget 59 GiB: over the 57.76 free, which §173 §2's pool made
    real by step 3, and under the 61.37 total -- a warning, because `--workers 4` is the
    caller's (§167).  Eight budget 107 and cannot fit the machine at all."""
    pre = _plan(monkeypatch, 57.76, config="medium", workers=4)
    assert not pre["blockers"], pre["blockers"]
    assert any("exceeds" in w for w in pre["warnings"]), pre["warnings"]
    assert _plan(monkeypatch, 57.76, config="medium", workers=8)["blockers"]


def test_serial_keeps_the_affine_estimate_and_its_headroom(monkeypatch):
    """Serial is §173 successor 2's and unchanged here: 43.4 GiB at `coarse`, capped x1.35."""
    pre = _plan(monkeypatch, 57.05, config="coarse", workers=0)
    assert pre["peak_gib"] == 43.4
    assert pre["memory_max_gib"] == round(43.4 * jobs.MEMORY_HEADROOM, 1)
    assert not pre["blockers"], pre["blockers"]
