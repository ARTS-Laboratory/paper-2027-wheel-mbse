"""The phase pool: does it give the same answer as serial, and does it clean up?

TWO KINDS OF TEST, AND THEY ARE DELIBERATELY SEPARATE.

The transport tests drive `PhasePool` against a STUB worker (`_stub_worker`, written to a
tmp_path and injected through `PhasePool(script=...)`).  Slot ordering, error frames,
respawn and child reaping are properties of the pool, not of the adjoint, and testing them
through a real solve would pay a jax import and a mesh build per assertion — which in
practice means testing them once, badly, or not at all.  The stub makes each of them a
sub-second, deterministic check.

`test_a_pooled_evaluation_equals_the_serial_one_exactly` is the other kind, and it is the
one the milestone rests on.  It pays a real `smoke` evaluation twice because there is no
cheaper way to say the thing that has to be said: pooled and serial agree to the BIT, not
to a tolerance.  Floating-point addition is not associative, so an as-completed reduction
over phases would make the aggregate depend on which worker finished first; `map_phases`
returns in slot order precisely so that this test can use `==`.

`workers=2` is a literal rather than `default_workers`, because the suite must assert the
same thing on every machine — and two workers are spawnable on a one-core CI runner.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import wheel_pool as WP

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A worker that does no FEA: it echoes its slot, can be told to sleep so that completion
# order and slot order disagree, and can be told to raise or to die outright.
_STUB_WORKER = '''
import os, sys, time
sys.path.insert(0, {src!r})
import wheel_pool as WP


def main():
    me = int(sys.argv[1])
    while True:
        task = WP._recv(sys.stdin.buffer)
        if task is None or task.get("kind") == "stop":
            return 0
        if task.get("die"):
            os._exit(9)
        time.sleep(float(task.get("sleep", 0.0)))
        try:
            if task.get("boom"):
                raise RuntimeError("the stub was asked to fail")
            if task.get("diverge"):
                import wheel_fem as WF
                raise WF.NewtonDivergedError("stub divergence", history=[{{}}] * 7)
            WP._send(sys.stdout.buffer,
                     {{"ok": True, "result": {{"slot": task["slot"], "worker": me}}}})
        except Exception as exc:
            WP._send(sys.stdout.buffer, {{"ok": False, "error": WP._error_frame(exc)}})


raise SystemExit(main())
'''


@pytest.fixture
def stub(tmp_path):
    path = tmp_path / "stub_worker.py"
    path.write_text(_STUB_WORKER.format(src=os.path.join(HERE, "src")))
    return str(path)


# ---------------------------------------------------------------------------
# SIZING — the one place core count is consulted
# ---------------------------------------------------------------------------

def test_default_workers_never_exceeds_the_phase_count_or_the_core_count(monkeypatch):
    monkeypatch.setattr(WP.os, "cpu_count", lambda: 4)
    assert WP.default_workers(8) == 4, "must not exceed the cores it has"
    assert WP.default_workers(2) == 2, "must not exceed the phases there are — a worker "\
                                       "with no slot is a jax import for nothing"
    monkeypatch.setattr(WP.os, "cpu_count", lambda: 64)
    assert WP.default_workers(8) == 8


def test_default_workers_survives_a_machine_that_cannot_count_its_cores(monkeypatch):
    """`os.cpu_count()` returns None on platforms that will not answer.

    One worker is the answer there, not a `TypeError` — the point of auto-sizing is that
    it runs everywhere.
    """
    monkeypatch.setattr(WP.os, "cpu_count", lambda: None)
    assert WP.default_workers(8) == 1


def test_the_worker_env_pins_every_thread_count():
    """All five, and `src/` still reachable.

    Pinning four of the five would leave whichever library reads the fifth spinning up a
    core-count pool per worker — the oversubscription this exists to prevent, minus the
    one variable nobody checked.
    """
    env = WP.worker_env({"PATH": "/usr/bin"})
    for name, value in WP.PINNED_ENV.items():
        assert env[name] == value, f"{name} is not pinned"
    assert os.path.join(HERE, "src") in env["PYTHONPATH"].split(os.pathsep)


def test_xla_is_pinned_because_it_is_what_makes_the_gradient_reproducible():
    """Named explicitly, because it is the entry that is easy to drop as noise.

    Measured: two plain serial runs of one `coarse` adjoint in separate interpreters, no
    pool involved, agree on every forward VALUE to the bit and disagree on the GRADIENT by
    3.33e-16 — XLA's CPU intra-op thread pool does not associate its reductions the same
    way twice, and nothing in OMP/MKL/OPENBLAS/NUMEXPR reaches it.  Delete this entry and
    `test_a_pooled_evaluation_equals_the_serial_one_exactly` starts failing for a reason
    that looks like the pool's fault and is not.
    """
    flags = WP.PINNED_ENV["XLA_FLAGS"]
    assert "intra_op_parallelism_threads=1" in flags
    assert "--xla_cpu_multi_thread_eigen=false" in flags


def test_the_worker_env_lets_a_deliberate_override_through():
    """`setdefault`, not assignment: someone debugging with threads set is driving.

    The cost is that S13's `identical` check fails — loudly, with a diff — which is the
    right way to find out, and better than an override that silently does nothing.
    """
    env = WP.worker_env({"OMP_NUM_THREADS": "8"})
    assert env["OMP_NUM_THREADS"] == "8"
    assert env["XLA_FLAGS"] == WP.PINNED_ENV["XLA_FLAGS"], "the rest must still be pinned"


def test_the_worker_env_layers_pythonpath_rather_than_replacing_it():
    env = WP.worker_env({"PYTHONPATH": "/somewhere/else"})
    assert "/somewhere/else" in env["PYTHONPATH"].split(os.pathsep)


# ---------------------------------------------------------------------------
# TRANSPORT
# ---------------------------------------------------------------------------

def test_results_come_back_in_slot_order_not_completion_order(stub):
    """The invariant the bit-identity claim is built on.

    Slot 0 is told to sleep longest, so it finishes last.  If `map_phases` returned in
    completion order the list would come back reversed, every reduction in `t3_terms`
    would sum the same floats in a different sequence, and pooled == serial would hold
    only by luck of the scheduler.
    """
    with WP.PhasePool(4, script=stub) as pool:
        out = pool.map_phases([{"sleep": 0.4 - 0.1 * i} for i in range(4)])
    assert [r["slot"] for r in out] == [0, 1, 2, 3]


def test_every_slot_lands_on_its_own_pinned_worker(stub):
    """Slot `i` goes to worker `i % n_workers` and nowhere else.

    This is the whole reason the pool is hand-rolled rather than `multiprocessing.Pool`:
    `coord_fn` keys its jit cache on `float(phase)` and M7 measured a miss at 0.774 s, so
    a phase that wanders between workers pays that on every step forever.
    """
    with WP.PhasePool(3, script=stub) as pool:
        out = pool.map_phases([{} for _ in range(7)])
    assert [r["worker"] for r in out] == [0, 1, 2, 0, 1, 2, 0]


def test_more_phases_than_workers_still_returns_one_reply_per_phase(stub):
    with WP.PhasePool(2, script=stub) as pool:
        out = pool.map_phases([{} for _ in range(5)])
    assert [r["slot"] for r in out] == [0, 1, 2, 3, 4]


def test_a_worker_that_raises_reaches_the_parent_as_that_exception(stub):
    with WP.PhasePool(2, script=stub) as pool:
        with pytest.raises(RuntimeError, match="the stub was asked to fail"):
            pool.map_phases([{"boom": True}, {}])


def test_a_diverged_solve_arrives_as_NewtonDivergedError_not_a_bare_RuntimeError(stub):
    """`descend` records `type(err).__name__` in its `solve_reject` event.

    A run record saying every failure was a `RuntimeError` has lost the distinction
    between "Newton diverged" and "the wheel got softer the harder it was pressed", which
    is the distinction that says whether the design or the solver is at fault.
    """
    import wheel_fem as WF

    with WP.PhasePool(2, script=stub) as pool:
        with pytest.raises(WF.NewtonDivergedError) as caught:
            pool.map_phases([{}, {"diverge": True}])
    assert len(caught.value.history) == 7, "the iteration count did not survive transport"


def test_decode_error_falls_back_to_RuntimeError_for_an_unknown_type():
    """A new exception type in the adjoint must degrade to a rejected step.

    `descend` catches `RuntimeError`; anything else escapes the trial loop and kills the
    run.  So an unrecognised name is deliberately NOT re-raised as itself.
    """
    exc = WP._decode_error({"type": "SomeFutureError", "message": "hello",
                            "n_newton_records": 0})
    assert type(exc) is RuntimeError
    assert "hello" in str(exc)


def test_decode_error_marks_the_message_as_coming_from_a_worker():
    exc = WP._decode_error({"type": "RuntimeError", "message": "x", "n_newton_records": 0})
    assert "[phase worker]" in str(exc), (
        "a traceback that does not say which process it came from sends whoever reads it "
        "looking in the parent")


def test_an_unpicklable_task_names_the_key_that_broke(stub):
    with WP.PhasePool(1, script=stub) as pool:
        with pytest.raises(TypeError, match="problem_kw"):
            pool.map_phases([{"problem_kw": lambda: None}])


# ---------------------------------------------------------------------------
# LIFECYCLE
# ---------------------------------------------------------------------------

def test_a_dead_worker_is_respawned_so_the_next_evaluation_proceeds(stub):
    """One segfault must cost one trial, not the run.

    Without the respawn the pool stays broken, `descend` rejects four attempts, abandons
    the step, and ends via `run_stopped_stuck` five steps later — hours of solving lost to
    one child.  With it, the failure is the `solve_reject` the reject path already handles
    and S5 already tests.
    """
    with WP.PhasePool(2, script=stub) as pool:
        with pytest.raises(WP.PoolDeadError):
            pool.map_phases([{"die": True}, {}])
        out = pool.map_phases([{}, {}])
    assert [r["slot"] for r in out] == [0, 1]


def test_closing_the_pool_reaps_every_child(stub):
    pool = WP.PhasePool(3, script=stub)
    procs = list(pool.procs)
    pool.map_phases([{}, {}, {}])
    pool.close()
    for k, proc in enumerate(procs):
        assert proc.poll() is not None, f"worker {k} is still running after close()"


def test_closing_twice_is_not_an_error(stub):
    """`descend` closes in a `finally` and callers use `with`; both can happen."""
    pool = WP.PhasePool(1, script=stub)
    pool.close()
    pool.close()


def test_a_closed_pool_refuses_work_instead_of_hanging(stub):
    pool = WP.PhasePool(1, script=stub)
    pool.close()
    with pytest.raises(WP.PoolDeadError):
        pool.map_phases([{}])


def test_a_hung_worker_becomes_an_error_rather_than_a_hang(stub):
    """A blocking read on a pipe has no portable timeout, so the pool supplies one."""
    with WP.PhasePool(1, script=stub, timeout=0.5) as pool:
        with pytest.raises(WP.PoolDeadError, match="died or hung"):
            pool.map_phases([{"sleep": 30.0}])


def test_zero_workers_is_refused_rather_than_silently_meaning_serial():
    with pytest.raises(ValueError):
        WP.PhasePool(0)


# ---------------------------------------------------------------------------
# THE IMPORT CONTRACT
# ---------------------------------------------------------------------------

def test_wheel_pool_imports_without_jax():
    """The parent half must stay cheap.

    `wheel_stage3` imports it, but so could anything that only wants to size a pool or
    read `PINNED_ENV`, and dragging in jax to answer `default_workers(8)` is the kind
    of import creep `tests/test_import_hygiene.py` exists to catch on the other half of
    the tree.  `_decode_error` imports `wheel_fem` lazily, inside the function, for exactly
    this reason.
    """
    code = ("import sys\nimport wheel_pool\n"
            "print('LEAKED:' + ','.join(m for m in ['jax'] if m in sys.modules))\n")
    proc = subprocess.run([sys.executable, "-c", code], cwd=os.path.join(HERE, "src"),
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "LEAKED:", (
        "`import wheel_pool` pulled in jax; keep the wheel_fem import inside "
        "_decode_error")


# ---------------------------------------------------------------------------
# THE CLAIM
# ---------------------------------------------------------------------------

def test_a_pooled_evaluation_matches_the_serial_one():
    """Values BIT-IDENTICAL, gradients to 1e-14.  What M8b-ii item 1 is allowed to claim.

    THE TWO STANDARDS ARE A MEASUREMENT, NOT A COMPROMISE.  Every forward value and every
    report leaf comes back from the pool bit-for-bit — that half is gated with `==` and
    passes with `==`.  The gradient cannot be, by anyone: two PLAIN SERIAL runs of one
    `coarse` adjoint in separate interpreters, with no pool anywhere, already disagree by
    3.33e-16, because XLA's CPU codegen answers differently depending on what the process
    has already run.  Pinning `XLA_FLAGS` removed the largest source; the remainder sits
    below this repo.  Observed here: 5.2e-18 relative, against a 1e-14 gate.

    Slow on purpose — a real `smoke` evaluation twice — because there is no cheaper way to
    compare the two paths' arithmetic.  The probe exponents are included even though they
    cost no solve: the probe is the ONE place the two branches read from different
    sources (the serial path off `_meta["prob"]`, the pooled path off the worker's reply),
    so leaving it out would skip the only asymmetry there is.

    The comparison itself is `study_stage3`'s, imported rather than reimplemented, so this
    test and S13 cannot come to describe two different standards.
    """
    import study_stage3 as so3
    import wheel_objective as WO
    import wheel_wheel as WW

    with open(os.path.join(HERE, "best_solution.json")) as fh:
        genes = np.array(list(json.load(fh)["genes"].values()), dtype=float)

    cfg = "smoke"
    phases = WO.phase_stencil(n_phase=2, scheme="uniform")
    orientation = tuple(float(o) for o in
                        WW.flank_orientation(genes, WW.get_config(cfg)))
    probe = (4.0, 30.0)

    meshes = WO.phase_meshes(genes, cfg, phases, orientation=orientation)
    serial = WO.t3_terms(genes, cfg, phases=phases, meshes=meshes, stress_p_probe=probe)
    with WP.PhasePool(2) as pool:
        pooled = WO.t3_terms(genes, cfg, phases=phases, pool=pool,
                             orientation=orientation, stress_p_probe=probe)

    vdiffs, gdiffs = so3._split_diffs(serial, pooled)
    assert not vdiffs, (
        f"a VALUE moved between the pooled and serial paths: {vdiffs[:8]}. These are "
        f"gated exactly because they pass exactly — this is a real difference, not "
        f"floating-point noise")
    assert not gdiffs, (
        f"a gradient moved by more than {so3.GATE_POOL_GRAD_REL:.0e} relative: "
        f"{gdiffs[:8]}")
