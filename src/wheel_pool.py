"""A phase-pinned worker pool: the eight solves of one evaluation, run at once.

WHAT THIS EXISTS FOR.  S10 measured a full 8-phase evaluation at 144.4 s at `coarse`, of
which `per_phase_s = 18.05` x 8 is the loop in `wheel_objective.t3_terms`.  A production
search of 300 steps x 4 starts is therefore 48.13 h serial, which is not a search anyone
runs.  The eight phases are FULLY INDEPENDENT — each needs only
`(genes, cfg, phase, orientation, force, delta0)` and returns three floats plus two
14-vectors — so the loop is the one place in this tree where parallelism is free of any
physics argument.

WHY THIS IS NOT `multiprocessing.Pool`, and all three reasons are load-bearing:

1.  THREAD SETTINGS MUST BE IN PLACE BEFORE NUMPY AND JAX ARE IMPORTED.  `OMP_NUM_THREADS`
    and its siblings are read by OpenBLAS/MKL at import and `XLA_FLAGS` by XLA at its
    first trace, and `spawn` offers no hook that runs that early — `initializer=` fires
    after the unpickling that already imported everything.  (`XLA_FLAGS` is not decoration
    here: without it the adjoint's GRADIENT is not reproducible across processes at all.
    See `PINNED_ENV`.)  An explicit `Popen(env=...)` is unambiguous.  `fork` would be
    early enough and is
    unusable for two separate reasons: forking a process that has already initialised jax
    is a documented hazard, and `fork` does not exist on Windows, so anything built on it
    is POSIX-only by construction.  This mirrors the CAD hand-off in
    `wheel_fea.py` — explicit env, explicit cwd, hard timeout, non-fatal child failure.

2.  PHASE SLOTS MUST PIN TO WORKERS.  `wheel_wheel.coord_fn` keys its jit cache on
    `float(phase)` and M7 measured a miss at 0.774 s against 0.05 s for the entire rest of
    the adjoint.  `Pool.map` gives no control over which worker sees which task, so a
    phase would land on a cold worker at random and pay that 0.774 s.  Here slot `i` goes
    to worker `i % n_workers` and nowhere else, so worker `k` only ever traces the phases
    of its own slots.  An rqmc stencil draws its offset from the `n_sub`-point sub-lattice,
    so one slot spans at most `n_sub` = 8 distinct phases: 8 cache entries per worker at
    `workers=8`, 16 at `workers=4`, against `_COORD_FN_CACHE_MAX = 128`.

3.  RESULTS MUST COMBINE IN SLOT ORDER.  `map_phases` returns a list indexed by slot
    regardless of which worker finished first.  Floating-point addition is not
    associative, so an as-completed reduction would make the aggregate depend on the
    scheduler; in slot order the arithmetic in `t3_terms` is the arithmetic it always was,
    and `study_stage3`'s S13 gates pooled == serial EXACTLY rather than approximately.

NOTHING UNPICKLABLE CROSSES.  `WheelMesh.__slots__` holds `_coord_fn`, a jitted
`PjitFunction` after the first `mesh_coords` call, and the adjoint's `_meta` carries the
sparse problem, the displacement field and the mesh itself.  So the boundary sits INSIDE
the loop body: the worker builds its own mesh from the genes and returns only the leaves
`t3_terms` actually reads.

THIS MODULE MUST NOT IMPORT JAX.  It is the parent half, it is imported by
`wheel_stage3`, and keeping it stdlib+numpy means a caller can size a pool, inspect the
worker environment or decode an error frame without paying for the heavy half of the
tree.  `tests/test_pool.py` asserts it in a fresh interpreter.
"""

import os
import pickle
import struct
import subprocess
import sys
import threading
import time

import project_paths as PP

WORKER_SCRIPT = os.path.join(PP.SRC, "wheel_pool_worker.py")

# THE THREAD PIN, AND `XLA_FLAGS` IS THE ENTRY THAT EARNS IT.
#
# The first four are the obvious half: N workers each spinning up a core-count-sized BLAS
# pool oversubscribes any machine by roughly N x.  `XLA_FLAGS` is the half that was
# MEASURED, and it is a correctness setting rather than a performance one.
#
# WHAT WAS MEASURED.  Two plain serial runs of one `coarse` adjoint, in two separate
# interpreters, with no pool anywhere: the forward values agree to the bit, and the
# GRADIENT does not — max |diff| 3.33e-16, which the objective's weights amplify to ~3e-12
# in the assembled `dL/dz`.  XLA's CPU backend sizes an intra-op thread pool from the
# machine and its parallel reductions do not associate the same way twice; nothing in
# OMP/MKL/OPENBLAS/NUMEXPR touches that pool.  Pinning it to one thread makes the same
# comparison exactly zero.
#
# AND IT IS FREE.  One `coarse` phase, median of 3 after priming: 20.43 s unpinned, 19.84 s
# pinned.  XLA's threads were buying nothing here — the work is a scipy sparse
# factorisation and a handful of small jax kernels — while costing reproducibility and, in
# a pool, an 8 x 16 oversubscription of a 16-core box.
#
# The values are constants rather than anything computed: the parallelism in this design
# lives at the PHASE level, wherever it runs.  The Makefile and `conftest.py` set the same
# five for the parent; these are here so a worker spawned from a bare interpreter, with no
# make and no pytest, is still pinned.
PINNED_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}

# How long to wait for one phase reply before declaring the worker hung.  A `coarse` phase
# is 18 s and a `medium` one 30 s; 900 s is the same order of slack `wheel_fea` gives the
# CAD hand-off, and it exists to turn a deadlock into an error rather than to police
# performance.
DEFAULT_TIMEOUT_S = 900.0

# How long a worker gets to exit after being asked politely, before `terminate()`.
SHUTDOWN_GRACE_S = 10.0

_LEN = struct.Struct(">I")      # fixed endianness; the two ends may not be the same build


def worker_env(base=None):
    """The environment a worker is spawned into: threads pinned, `src/` reachable.

    `setdefault`, not assignment: someone debugging with `XLA_FLAGS` or `OMP_NUM_THREADS`
    already set is driving on purpose, and silently overriding them would make the
    override look broken.  The cost of letting them through is that S13's `identical`
    check fails, loudly, with a diff — which is the right way to find out.

    `PYTHONPATH` is layered rather than replaced because the Makefile exports it for
    exactly this reason, and a worker that cannot `import wheel_objective` fails in a way
    that looks like a missing dependency rather than a missing path.
    """
    env = dict(os.environ if base is None else base)
    for name, value in PINNED_ENV.items():
        env.setdefault(name, value)
    existing = env.get("PYTHONPATH", "")
    if PP.SRC not in existing.split(os.pathsep):
        env["PYTHONPATH"] = (PP.SRC + os.pathsep + existing) if existing else PP.SRC
    return env


def default_workers(n_phase):
    """How many workers this machine should run for `n_phase` phases.

    THE ONLY PLACE CORE COUNT IS CONSULTED.  Everything else takes an explicit integer, so
    a study can pin a ladder and a test can assert the same thing on every machine.

    Capped by `n_phase` as well as by cores, and that half matters just as much: a ninth
    worker for eight phases is a ninth jax import, a ninth interpreter's worth of resident
    memory, and no work.  `os.cpu_count()` returns `None` on platforms that cannot answer,
    which is a one-worker answer rather than a crash.
    """
    return max(1, min(int(n_phase), os.cpu_count() or 1))


def _send(fh, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    fh.write(_LEN.pack(len(payload)))
    fh.write(payload)
    fh.flush()


def _read_exactly(fh, n):
    """`n` bytes, or `None` if the stream ended first.  `read` on a pipe may short-count."""
    chunks, got = [], 0
    while got < n:
        block = fh.read(n - got)
        if not block:
            return None
        chunks.append(block)
        got += len(block)
    return b"".join(chunks)


def _recv(fh):
    head = _read_exactly(fh, _LEN.size)
    if head is None:
        return None
    body = _read_exactly(fh, _LEN.unpack(head)[0])
    return None if body is None else pickle.loads(body)


def _error_frame(exc):
    """The picklable summary of an exception, built worker-side.

    The exception object itself is not sent.  `NewtonDivergedError.history` is a list of
    per-iteration dicts that can run to hundreds of entries, and the only thing any caller
    reads from it is its length (`wheel_stage3.descend` records `n_newton_records`).
    """
    return {"type": type(exc).__name__, "message": str(exc),
            "n_newton_records": len(getattr(exc, "history", []) or [])}


def _decode_error(frame):
    """Rebuild a raisable exception from an error frame.

    THE TYPE NAME IS PRESERVED ON PURPOSE.  `descend`'s `solve_reject` event records
    `type(err).__name__`, and a run record that says every failure was a bare
    `RuntimeError` has lost the distinction between "Newton diverged" and "the wheel got
    softer the harder it was pressed" — which is the distinction that tells you whether
    the design or the solver is at fault.  Unknown names fall back to `RuntimeError`,
    which is what `descend` catches, so a new exception type in the adjoint degrades to a
    rejected step rather than to an escaped exception.
    """
    import wheel_fem as WF

    message = f"[phase worker] {frame['message']}"
    n = int(frame.get("n_newton_records", 0))
    if frame["type"] == "NewtonDivergedError":
        return WF.NewtonDivergedError(message, history=[{}] * n)
    return RuntimeError(message)


class PoolDeadError(RuntimeError):
    """A worker died or hung.  A `RuntimeError`, so `descend` rejects the step."""


class PhasePool:
    """`n_workers` persistent interpreters, each pinned to its own phase slots.

    Persistent because the costs a worker pays once are the ones worth avoiding: the jax
    import, the `wheel_fem` kernel traces, and the `coord_fn` trace for each of its slots'
    phases.  A pool rebuilt per step would pay all of them per step and lose to serial.

    Use it as a context manager, or call `close()` from a `finally`.  Eight orphaned
    interpreters holding a `medium` mesh each is not a leak anyone notices until the
    machine swaps.
    """

    def __init__(self, n_workers, *, python=None, timeout=DEFAULT_TIMEOUT_S, env=None,
                 script=None):
        if int(n_workers) < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers!r}")
        self.n_workers = int(n_workers)
        # `script` exists so the TRANSPORT can be tested without an FEA solve.  Slot
        # ordering, error frames, respawn and reaping are properties of this class, not of
        # the adjoint, and a suite that could only exercise them through a real solve
        # would pay a jax import per assertion and test them barely or not at all.
        self.script = script or WORKER_SCRIPT
        # `sys.executable` is the parent's own interpreter, which is already the right
        # venv on every platform.  `wheel_fea`'s probe for `.venv-cad/bin/python` vs
        # `Scripts/python.exe` exists because that hand-off deliberately crosses
        # interpreters; this one must not, so it must not copy those hardcoded paths.
        self.python = python or sys.executable
        self.timeout = float(timeout)
        self.env = worker_env(env)
        self.procs: list = [None] * self.n_workers
        self._closed = False
        for k in range(self.n_workers):
            self._start(k)

    # -- lifecycle ---------------------------------------------------------------

    def _start(self, k):
        self.procs[k] = subprocess.Popen(
            [self.python, self.script, str(k)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            # stderr is INHERITED: a worker traceback has to reach the terminal, and
            # stdout is the data channel so it cannot carry text.
            #
            # Default buffering, NOT `bufsize=0`.  An unbuffered pipe hands back a raw
            # stream whose `write` is allowed to short-count, which would silently
            # truncate a frame; the buffered writer plus an explicit `flush` in `_send`
            # writes all of it or raises.
            stderr=None, cwd=PP.ROOT, env=self.env)

    def close(self):
        """Ask, then terminate, then kill.  All three are cross-platform `Popen` calls."""
        if self._closed:
            return
        self._closed = True
        for proc in self.procs:
            if proc is None:
                continue
            try:
                _send(proc.stdin, {"kind": "stop"})
            except (OSError, ValueError):
                pass                      # already gone; the wait below reaps it
        for proc in self.procs:
            if proc is None:
                continue
            try:
                proc.wait(timeout=SHUTDOWN_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=SHUTDOWN_GRACE_S)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            for stream in (proc.stdin, proc.stdout):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        self.procs = [None] * self.n_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _kill(self, k):
        proc = self.procs[k]
        if proc is None:
            return
        proc.kill()
        proc.wait()
        for stream in (proc.stdin, proc.stdout):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        self.procs[k] = None

    def _respawn(self, k):
        """Replace a dead worker so the NEXT trial works.

        Without this a single segfault poisons every later trial: `descend` would reject
        four attempts, abandon the step, and end the run via `run_stopped_stuck` five
        steps later — hours of solving lost to one child.  With it, only the trial in
        flight is lost, which is a case the reject path already handles and S5 already
        tests.
        """
        self._kill(k)
        self._start(k)

    # -- the dispatch ------------------------------------------------------------

    def map_phases(self, tasks):
        """One reply per task, IN SLOT ORDER, whatever order the workers finished in.

        Dispatched in ROUNDS: round `r` sends slot `r * n_workers + k` to worker `k`, then
        collects all of that round's replies before sending the next.  Every worker is
        busy within a round, which is where the speedup comes from, and no more than one
        task and one reply are ever outstanding on a pipe — so the pool cannot deadlock
        against a full pipe buffer no matter how large a payload grows.  With the usual
        `n_phase = 8` and `workers = 8` there is exactly one round.

        Reads happen on one short-lived thread per worker so the whole round shares a
        single deadline.  A blocking `read` on a pipe cannot be given a timeout portably —
        `selectors` does not accept pipes on Windows — and a hung child must become an
        error rather than a hang.
        """
        if self._closed:
            raise PoolDeadError("this PhasePool is closed")
        tasks = list(tasks)
        for i, task in enumerate(tasks):
            _check_picklable(i, task)

        out = [None] * len(tasks)
        for start in range(0, len(tasks), self.n_workers):
            batch = [(start + k, k) for k in range(self.n_workers)
                     if start + k < len(tasks)]
            for slot, k in batch:
                try:
                    _send(self.procs[k].stdin, dict(tasks[slot], kind="phase", slot=slot))
                except (OSError, ValueError, AttributeError) as exc:
                    self._respawn(k)
                    raise PoolDeadError(
                        f"phase worker {k} was not writable (slot {slot}): {exc}") from exc

            replies = {}
            threads = []
            for slot, k in batch:
                t = threading.Thread(target=self._collect, args=(k, slot, replies),
                                     daemon=True)
                t.start()
                threads.append((t, slot, k))
            # ONE deadline for the round, not one per worker.  Joining each thread with
            # the full timeout would let a round wait `n_workers * timeout` before
            # reporting a hang, which at the default is most of a day.
            deadline = time.monotonic() + self.timeout
            for t, _, _ in threads:
                t.join(timeout=max(0.0, deadline - time.monotonic()))

            for _, slot, k in threads:
                reply = replies.get(slot)
                if reply is None:
                    self._respawn(k)
                    raise PoolDeadError(
                        f"phase worker {k} died or hung on slot {slot}; it has been "
                        f"respawned, so the next evaluation should proceed")
                if isinstance(reply, BaseException):
                    self._respawn(k)
                    raise PoolDeadError(
                        f"phase worker {k} could not be read on slot {slot}: "
                        f"{reply}") from reply
                if not reply.get("ok"):
                    raise _decode_error(reply["error"])
                out[slot] = reply["result"]
        return out

    def _collect(self, k, slot, replies):
        proc = self.procs[k]
        if proc is None:
            return
        try:
            reply = _recv(proc.stdout)
        except (OSError, ValueError, EOFError, pickle.UnpicklingError) as exc:
            replies[slot] = exc
            return
        if reply is not None:
            replies[slot] = reply


def _check_picklable(i, task):
    """Fail on the offending key, not on `pickle`'s own message.

    Everything in a task is a float, a string, a tuple or a small array — until someone
    threads a live object through `problem_kw`, at which point the default failure is a
    `PicklingError` naming a type with no hint of where it came from.
    """
    for key, value in task.items():
        try:
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:              # noqa: BLE001 — any pickler failure
            raise TypeError(
                f"phase task {i} key {key!r} is not picklable ({type(value).__name__}); "
                f"only plain data crosses to a phase worker") from exc
