"""Detached jobs, and the on-disk registry that survives this process.

THE ONE DECISION THIS FILE EXISTS TO ENFORCE: a run is never a child of the GUI.  Close
the browser, stop the server, `kill -9` it -- the descent keeps descending, and the next
server to start finds it again by reading the same directories.  Everything else here
follows from that.

HOW A JOB IS STARTED.  `systemd-run --user --unit=wheelgui-<id> --collect`, which is the
same incantation the Makefile pastes into six recipe comments for long runs, and it buys
two things beyond detachment: a cgroup, so cancelling kills the worker pool too and not
just the parent, and `MemoryMax=`, so a descent that runs away is OOM-killed inside its
own cgroup instead of taking the desktop down -- which the Makefile records happening
twice.  Where there is no systemd user manager there are two fallbacks and neither can cap
memory; see THREE DETACH MECHANISMS below.

HOW A JOB IS WATCHED, AND WHY NOT BY ASKING SYSTEMD.  `--collect` garbage-collects the
unit the moment it exits, taking its exit status with it, so a GUI that asked
`systemctl show -p ExecMainStatus` would find nothing for every job that had already
finished -- which is most of them, most of the time.  So the wrapper writes its own
evidence instead:

    pid   the shell's pid, written by the shell itself
    rc    the exit code, written after the command returns

`rc` existing IS the definition of finished, and it is durable, readable by anything, and
independent of both systemd and this process.  A job with no `rc` whose pid is gone did
not finish -- it was killed or the box went down -- and that is a THIRD state worth
distinguishing from success and failure, because it is what an OOM looks like from here.

The pid is read back from the file the job wrote about itself rather than resolved later
by pattern-matching a process list: a pgrep that matches the wrong `python src/...` and
then reports a 32-minute suite "finished" in four seconds is a failure mode this project
has already paid for once.  Where WE created the process rather than handing it to a
service manager, `Popen.pid` is used instead -- that is the kernel naming the process it
just made for us, which is the same evidence by a shorter route, not a guess.

THREE DETACH MECHANISMS, ONE REGISTRY.  Nothing below the launch decision knows which ran:

    systemd   Linux with a user manager.  The only one that also gives a cgroup and
              MemoryMax, which is why it is preferred wherever it exists.
    posix     Linux without one, and macOS.  `start_new_session=True`, which IS what
              setsid(1) does -- and the binary is not on macOS at all, so calling it was
              a portability bug rather than a belt-and-braces second copy.
    windows   DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, and a `.cmd` wrapper.

Only the first can cap memory.  `plan()` refuses on TOTAL memory, which every platform
reports exactly, and only warns on AVAILABLE, which off Linux is an approximation -- so
the hard guard does not rest on the soft number.
"""

import ctypes
import glob
import json
import os
import random
import shlex
import signal
import string
import subprocess
import sys
import time

import catalog

IS_WINDOWS = os.name == "nt"
IS_MACOS = sys.platform == "darwin"

ROOT = catalog.ROOT
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

# What fraction of physical RAM a job is allowed to ask for before we refuse outright.
# The box has been taken down twice by two concurrent descents; 43.4 GiB of a 61 GiB
# machine is one descent's measured peak, so the ceiling has to sit above that and well
# below two of them.
MEMORY_CEILING_FRACTION = 0.85
MEMORY_HEADROOM = 1.35          # cap = estimate x this, so a good estimate is not fatal
# A cap is a guard against runaway, not a budget to hold a run to. Set below what a
# run genuinely needs it turns a working job into a swapping one -- measured here on
# 2026-09-09, when a 9 GiB cap put a `smoke` descent into 1.5 GiB of swap.
MEMORY_FLOOR_GIB = 8.0

_ALIVE = ("running", "queued")


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _age_s(job):
    try:
        return time.time() - time.mktime(time.strptime(job["started_at"],
                                                       "%Y-%m-%dT%H:%M:%S"))
    except (KeyError, ValueError):
        return 1e9


class _MEMORYSTATUSEX(ctypes.Structure):
    """The Win32 struct. Field order and widths are the API's, not ours.

    Declarable on every platform -- `ctypes.Structure` is portable and only
    `ctypes.windll`, down in `_win_memory`, is not.
    """
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _win_memory():
    """(total, available) bytes of physical RAM. Exact on both counts."""
    st = _MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(st)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        raise OSError("GlobalMemoryStatusEx failed")
    return st.ullTotalPhys, st.ullAvailPhys


def _total_gib():
    """Physical RAM. EXACT ON EVERY PLATFORM, which is why the hard refusal uses it.

    `sysconf` covers Linux and macOS; Windows has neither of those two names and needs
    the Win32 call.
    """
    if IS_WINDOWS:
        return _win_memory()[0] / 2**30
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30


def _macos_available_gib():
    """macOS has no MemAvailable, and this is an APPROXIMATION OF ONE, not a substitute.

    Free + inactive + speculative + purgeable pages: the four `vm_stat` buckets a new
    allocation can take without swapping.  It is not the kernel's own estimate the way
    Linux's MemAvailable is, and it does not account for compressed memory -- macOS keeps
    almost nothing genuinely free, so `pages free` alone would report a 64 GiB box as
    having a few hundred MiB and block every job on it.

    Only ever produces a WARNING (see `plan`), never a refusal.  The refusal is on
    `_total_gib`, which is exact here.
    """
    out = subprocess.run(["vm_stat"], capture_output=True, timeout=10).stdout.decode()
    page = 4096
    first = out.splitlines()[0] if out else ""
    if "page size of" in first:
        page = int(first.split("page size of")[1].split()[0])
    want = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    pages = 0
    for line in out.splitlines():
        name, _, value = line.partition(":")
        if name.strip() in want:
            pages += int(value.strip().rstrip("."))
    return pages * page / 2**30


def available_gib():
    """What a new job can take without swapping.

    Linux's MemAvailable is the kernel's own answer and the only exact one of the three;
    `_macos_available_gib` says why its reading is an approximation, and Windows'
    ullAvailPhys is exact again.  Falls back to total on any failure, which errs toward
    letting a run start -- correct here, because this number only ever warns.
    """
    try:
        if IS_WINDOWS:
            return _win_memory()[1] / 2**30
        if IS_MACOS:
            return _macos_available_gib()
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    return _total_gib()


def _have_systemd():
    """Whether there is a systemd USER manager to hand a unit to.

    Short-circuited off Linux rather than probed: `systemctl` is absent on Windows and on
    stock macOS, so the probe would only ever cost a subprocess timeout to learn what the
    platform already says.  A Linux box without a user manager -- a container, a bare
    session -- still has to be asked, and answers `False` through the probe below.
    """
    if IS_WINDOWS or IS_MACOS:
        return False
    try:
        out = subprocess.run(["systemctl", "--user", "is-system-running"],
                             capture_output=True, timeout=5).stdout.decode().strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return out in ("running", "degraded")


HAVE_SYSTEMD = _have_systemd()

# Which of the three the launch will use.  Decided once, here, so that the printout, the
# `/api/machine` panel and `launch()` can never disagree about it.
DETACH = "systemd-run --user" if HAVE_SYSTEMD else ("windows" if IS_WINDOWS else "posix")


# ---------------------------------------------------------------------------
# THE REGISTRY
# ---------------------------------------------------------------------------

def _job_path(rundir):
    return os.path.join(rundir, "job.json")


def _read(rundir):
    try:
        with open(_job_path(rundir)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write(job):
    """Atomic, because the server may be reading this while a refresh rewrites it."""
    path = _job_path(job["rundir"])
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(job, fh, indent=2)
    os.replace(tmp, path)


def _unit_alive(unit):
    """Whether a transient unit is still up.

    Only ever consulted as a SECOND opinion, never as the record of how a job ended:
    `--collect` garbage-collects the unit the instant it exits, so `inactive` here means
    "gone", which is true of a clean success and of an OOM kill alike.  `rc` is what
    separates those, and it is checked first.
    """
    try:
        out = subprocess.run(["systemctl", "--user", "is-active", unit],
                             capture_output=True, timeout=10).stdout.decode().strip()
    except (OSError, subprocess.SubprocessError):
        return True                    # cannot tell: do not declare a live job dead
    return out in ("active", "activating", "reloading")


_STILL_ACTIVE = 259               # Win32 STILL_ACTIVE, the sentinel GetExitCodeProcess
                                  # returns for a process that has not exited


def _win_pid_alive(pid):
    """Liveness on Windows, WITHOUT os.kill.

    `os.kill(pid, 0)` is the POSIX idiom and on Windows it is a footgun: CPython
    implements os.kill there by calling TerminateProcess for any signal that is not
    CTRL_C_EVENT or CTRL_BREAK_EVENT, so the "harmless" probe would kill the very job it
    was asked about -- and would kill it with exit code 0, which this module would then
    read back off `rc` as a clean success.
    """
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    k32 = ctypes.windll.kernel32
    handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True           # cannot tell: do not declare a live job dead
        return code.value == _STILL_ACTIVE
    finally:
        k32.CloseHandle(handle)


def _pid_alive(pid):
    if not pid:
        return False
    try:
        if IS_WINDOWS:
            return _win_pid_alive(pid)
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def refresh(job):
    """Recompute a job's state from what the job itself left on disk.

    Called on every listing, which is what makes reattach work: nothing about a job's
    state lives in this process, so a server that has just started knows exactly as much
    as one that launched it.
    """
    if job.get("state") not in _ALIVE:
        return job
    rundir = job["rundir"]
    rc_path = os.path.join(rundir, "rc")
    if os.path.exists(rc_path):
        try:
            with open(rc_path) as fh:
                code = int(fh.read().strip() or 1)
        except (OSError, ValueError):
            code = 1
        job["state"] = "finished" if code == 0 else "failed"
        job["exit_code"] = code
        job["ended_at"] = job.get("ended_at") or _now()
        _write(job)
        return job
    # A job whose pid we never captured -- because the wrapper had not written it yet,
    # or wrote something unreadable -- would otherwise sit in `queued` forever after
    # being killed from outside this GUI.  The unit is the second opinion.  Age-gated,
    # because a unit is not registered the instant `systemd-run` returns.
    if job.get("unit") and _age_s(job) > 15 and not _unit_alive(job["unit"]):
        job["state"] = "interrupted"
        job["ended_at"] = job.get("ended_at") or _now()
        _write(job)
        return job
    if job.get("pid") is None:
        pid_path = os.path.join(rundir, "pid")
        if os.path.exists(pid_path):
            try:
                with open(pid_path) as fh:
                    job["pid"] = int(fh.read().strip())
                job["state"] = "running"
                _write(job)
            except OSError:
                pass
            except ValueError:
                # A pid file that is not a number means the wrapper was mangled between
                # here and the shell.  Surface it: a job stuck in `queued` forever while
                # its process runs happily is the worst of both readings.
                job["state"] = "running"
                job["note"] = f"pid file is not a number ({pid_path})"
                _write(job)
        return job
    if _pid_alive(job["pid"]):
        job["state"] = "running"
        return job
    # No rc, and the pid the job reported for itself is gone.  Killed, OOMed, or the box
    # went down -- deliberately NOT called "failed", because nothing reported a status.
    job["state"] = "interrupted"
    job["ended_at"] = job.get("ended_at") or _now()
    _write(job)
    return job


def list_jobs():
    jobs = []
    for path in sorted(glob.glob(os.path.join(RUNS, "*", "job.json")), reverse=True):
        job = _read(os.path.dirname(path))
        if job:
            jobs.append(refresh(job))
    return jobs


def get(job_id):
    for job in list_jobs():
        if job["id"] == job_id:
            return job
    return None


def running_jobs():
    return [j for j in list_jobs() if j["state"] in _ALIVE]


# ---------------------------------------------------------------------------
# LAUNCH
# ---------------------------------------------------------------------------

def plan(target_key, params):
    """What would happen, without doing it: argv, cost, memory verdict, blockers.

    The confirm dialog is built from this, and so is the refusal.  Separated from
    `launch` so the UI can show the price before anything is spent -- the estimate that
    priced `stage3` 27x under lived unchallenged in a comment for months precisely
    because nothing ever printed it next to a decision.
    """
    target = catalog.resolve(target_key)
    rundir = os.path.join(RUNS, "PREVIEW")
    argv = target.argv(params, rundir)
    cost = target.cost(params) if target.cost else None
    avail = available_gib()
    total = _total_gib()
    out = {"key": target_key, "label": target.label, "heavy": target.heavy,
           "argv": argv, "command": " ".join(shlex.quote(a) for a in argv),
           "available_gib": round(avail, 1), "total_gib": round(total, 1),
           "blockers": [], "warnings": []}
    if cost:
        seconds, gib, basis = cost
        cap = max(MEMORY_FLOOR_GIB, gib * MEMORY_HEADROOM)
        out.update(seconds=seconds, peak_gib=gib, basis=basis,
                   memory_max_gib=round(cap, 1))
        if gib > total * MEMORY_CEILING_FRACTION:
            out["blockers"].append(
                f"needs about {gib:.1f} GiB and this box has {total:.0f} GiB -- "
                f"reduce workers, or drop a mesh rung")
        elif gib > avail:
            out["warnings"].append(
                f"estimated peak {gib:.1f} GiB exceeds the {avail:.1f} GiB free now")
        if seconds > 3600:
            out["warnings"].append(f"estimated wall time {seconds / 3600:.1f} hours")
    writes = target.writes(params) if target.writes else []
    if writes:
        out["writes"] = writes
        out["warnings"].append(
            "this run OVERWRITES committed evidence: " + ", ".join(writes) +
            ". A study artifact records what was measured at the commit that filed it; "
            "refreshing one on a new genome rewrites the evidence rather than updating "
            "the number. Commit it with the driver, or not at all.")
    if target.heavy:
        live = [j for j in running_jobs() if j.get("heavy")]
        if live:
            out["blockers"].append(
                f"a heavy job is already running ({live[0]['label']}) -- this box runs "
                f"out of memory before it runs out of cores")
    return out


def _env():
    """The environment `make` would have given us.

    `wheel_pool.worker_env()` is the canonical construction and is used rather than a
    hand-built dict: it layers PYTHONPATH instead of replacing it, and it `setdefault`s
    the five pinned thread/XLA variables so a deliberate override still passes through.
    Those five are a correctness setting -- without XLA_FLAGS the adjoint gradient is not
    bit-reproducible across processes -- not a performance knob.
    """
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import wheel_pool
    return wheel_pool.worker_env()


def _write_wrapper(rundir, argv, log):
    """Write the script the job actually runs, and return its path.

    THE WRAPPER GOES IN A FILE, NOT ON THE COMMAND LINE, AND THAT IS A BUG FIX.
    `systemd-run` builds a unit whose ExecStart is subject to systemd's OWN variable
    expansion, in which `$$` is the escape for a literal dollar -- so an inline
    `echo $$ > pid` reaches the shell as `echo $ > pid` and writes the character instead
    of the process id.  The job then runs perfectly while this module, reading a pid file
    containing `$`, reports it as never having started.  Putting the script in a file
    means systemd passes a path and expands nothing.

    It also leaves the exact command that ran sitting in the run directory, which is the
    first thing anyone wants when a job did something surprising -- and that is worth as
    much on the `.cmd` side, where the quoting is the part nobody can reconstruct.

    The two dialects are not a translation of one another.  `cmd` has no `$$`, so the
    Windows wrapper writes no pid file at all and `launch` takes `Popen.pid` instead; and
    its exit code is `%ERRORLEVEL%` read on the next line rather than `$?`.

    THE PARENTHESES AROUND THAT `echo` ARE LOAD-BEARING.  `%ERRORLEVEL%` expands to a
    NUMBER, so the obvious `echo %ERRORLEVEL%>"rc"` becomes `echo 1>"rc"` -- and a digit
    written hard against a `>` is cmd's syntax for a STREAM NUMBER, not text to echo.  It
    redirects stdout and writes an empty `rc`, which this module reads back as an
    unparseable exit code on every job that finishes.  `(echo %ERRORLEVEL%)>"rc"` puts the
    digit inside a block where nothing can mistake it for a file descriptor.  The `>>` on
    the command line above is safe by contrast only because a space precedes it.
    """
    if IS_WINDOWS:
        # list2cmdline quotes for the argv parser CreateProcess hands the child, which is
        # not the same grammar as cmd's own -- a path holding `&`, `^` or `%` would still
        # break here.  Repo paths and numeric parameters are what actually goes through.
        script_path = os.path.join(rundir, "run.cmd")
        with open(script_path, "w", newline="\r\n") as fh:
            fh.write("@echo off\n"
                     "rem Written by gui/jobs.py. This is exactly what ran.\n"
                     f'cd /d "{ROOT}" || exit /b 1\n'
                     f'{subprocess.list2cmdline(argv)} >> "{log}" 2>&1\n'
                     f'(echo %ERRORLEVEL%)>"{os.path.join(rundir, "rc")}"\n')
        return script_path

    script_path = os.path.join(rundir, "run.sh")
    with open(script_path, "w") as fh:
        fh.write("#!/bin/sh\n"
                 "# Written by gui/jobs.py. This is exactly what ran.\n"
                 f"cd {shlex.quote(ROOT)} || exit 1\n"
                 f'echo $$ > {shlex.quote(os.path.join(rundir, "pid"))}\n'
                 f'{" ".join(shlex.quote(a) for a in argv)} '
                 f'>> {shlex.quote(log)} 2>&1\n'
                 f'echo $? > {shlex.quote(os.path.join(rundir, "rc"))}\n')
    os.chmod(script_path, 0o755)
    return script_path


def launch(target_key, params, force=False):
    target = catalog.resolve(target_key)
    pre = plan(target_key, params)
    if pre["blockers"] and not force:
        raise RuntimeError("; ".join(pre["blockers"]))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    job_id = f"{stamp}_{target.key.replace(':', '-')}_{tag}"
    rundir = os.path.join(RUNS, job_id)
    os.makedirs(rundir, exist_ok=True)

    argv = target.argv(params, rundir)
    log = os.path.join(rundir, "stdout.log")

    script_path = _write_wrapper(rundir, argv, log)

    unit, pid = None, None
    if HAVE_SYSTEMD:
        unit = f"wheelgui-{job_id}"
        cmd = ["systemd-run", "--user", f"--unit={unit}", "--collect",
               f"--working-directory={ROOT}", "--quiet"]
        if pre.get("memory_max_gib"):
            cmd.append(f"--property=MemoryMax={pre['memory_max_gib']:.1f}G")
        for name, value in _env().items():
            if name in ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "XLA_FLAGS"):
                cmd.append(f"--setenv={name}={value}")
        cmd += ["/bin/sh", script_path]
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        # No pid to take: systemd made the process, not us, and the wrapper reports its
        # own through the `pid` file.
    elif IS_WINDOWS:
        # DETACHED_PROCESS drops the console the server holds, so closing the app does
        # not deliver a console-close to the job; CREATE_NEW_PROCESS_GROUP is what makes
        # the whole tree killable as one in `cancel`.
        proc = subprocess.Popen(["cmd", "/c", script_path], cwd=ROOT, env=_env(),
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                creationflags=(subprocess.DETACHED_PROCESS |
                                               subprocess.CREATE_NEW_PROCESS_GROUP))
        pid = proc.pid
    else:
        # `start_new_session=True` IS setsid(1) -- it calls setsid(2) between fork and
        # exec.  The old code spawned the BINARY as well, which is absent on macOS and
        # was doing nothing on Linux that this flag was not already doing.
        proc = subprocess.Popen(["/bin/sh", script_path], cwd=ROOT, env=_env(),
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
        pid = proc.pid

    job = {"id": job_id, "key": target_key, "label": target.label,
           "group": target.group, "progress": target.progress,
           "phases": list(target.phases), "heavy": target.heavy,
           "params": params, "argv": argv, "rundir": rundir, "log": log,
           "unit": unit, "pid": pid, "state": "running" if pid else "queued",
           "started_at": _now(), "ended_at": None, "exit_code": None,
           "estimate": {k: pre.get(k) for k in
                        ("seconds", "peak_gib", "basis", "memory_max_gib")},
           "outputs": target.outputs(params, rundir) if target.outputs else {}}
    _write(job)
    return refresh(job)


def cancel(job_id):
    """Stop a job. The cgroup is the unit of killing, so the worker pool goes too."""
    job = get(job_id)
    if job is None:
        raise KeyError(job_id)
    if job["state"] not in _ALIVE:
        return job
    if job.get("unit"):
        subprocess.run(["systemctl", "--user", "stop", job["unit"]],
                       capture_output=True, timeout=30)
    elif job.get("pid"):
        # THE TREE, NOT THE PROCESS, in all three cases: the thing being cancelled is a
        # shell whose child is a worker POOL, and killing only the shell orphans the
        # workers still holding the memory this was called to release.
        try:
            if IS_WINDOWS:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(int(job["pid"]))],
                               capture_output=True, timeout=30)
            else:
                os.killpg(os.getpgid(int(job["pid"])), signal.SIGTERM)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    job["state"] = "cancelled"
    job["ended_at"] = _now()
    _write(job)
    return job


def forget(job_id):
    """Drop a finished job's directory. Refuses while it is still running."""
    import shutil
    job = get(job_id)
    if job is None:
        raise KeyError(job_id)
    if job["state"] in _ALIVE:
        raise RuntimeError("cancel it before forgetting it")
    shutil.rmtree(job["rundir"], ignore_errors=True)
