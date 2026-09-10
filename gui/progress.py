"""Reading how far along a job is.

THE PIPELINE HAS EXACTLY THREE PROGRESS SHAPES and pretending otherwise would mean either
one useless bar for everything or a parser per target:

  descent   the only one with a structured channel.  `wheel_stage3._persist` rewrites the
            whole trajectory to `<out>.tmp` and `os.replace`s it after EVERY step, so the
            file is always complete and never half-written -- deliberately, so a kill
            cannot truncate it.  That makes it pollable: `len(steps) / settings.steps` is
            the fraction, and every step carries its own `wall_s`.
  ga        stdout only.  `wheel_fea` prints `[Gen N] Best loss = ...` and its banner
            states the generation count, so the denominator is in the log too.
  phases    stdout only, and the phases are whatever the recipe echoes.  `make studies`
            runs nine drivers in sequence and make echoes each command as it starts them.

plus `pytest`, which is really a phase reader that gets its percentage for free because
pytest already prints `[ NN%]`, and `plain`, which admits it does not know.

WHY THE ETA USES A MEDIAN AND NOT A MEAN.  Step 0 of a descent pays for the jit trace:
1391 s against a 210 s steady state in the run this parser was written against.  A mean
over all steps carries that spike forever and reports an ETA that is wrong for the whole
run; a median over the recent tail forgets it after a few steps.

MTIME CACHING IS NOT AN OPTIMISATION HERE.  A `medium` trajectory is 340 KB-1 MB and is
rewritten every step; re-parsing it for every client poll of every job would be the
server's entire cost.  Parse on change, serve the summary.
"""

import json
import os
import re
import statistics

_CACHE = {}

_STEP_RE = re.compile(
    rb"\[step\s+(\d+)\]\s+loss\s+([-\d.eE+]+)\s+\|grad\|\s+([-\d.eE+]+)\s+"
    rb"lr\s+([-\d.eE+]+)\s+drop\s+([-\d.eE+nan]+)\s+mm\s+util\s+([-\d.eE+nan]+)\s+"
    rb"([\d.]+)\s+s")
_GEN_RE = re.compile(rb"\[Gen\s+(\d+)\]\s+Best loss\s*=\s*([-\d.eE+]+)\s*\|\s*"
                     rb"Mean loss\s*=\s*([-\d.eE+]+)")
_GA_TOTAL_RE = re.compile(rb"GA:\s*pop\s*(\d+)\s*\S+\s*(\d+)\s*gen")
_S3_TOTAL_RE = re.compile(rb"STAGE 3 .*?\(\w+,\s*\w+,\s*(\d+) steps")
_PCT_RE = re.compile(rb"\[\s*(\d+)%\]")
_PYTEST_SUMMARY_RE = re.compile(
    rb"(\d+) passed(?:, (\d+) failed)?|(\d+) failed")


def _stat(path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _cached(path, parse):
    key = _stat(path)
    if key is None:
        return None
    hit = _CACHE.get(path)
    if hit and hit[0] == key:
        return hit[1]
    value = parse(path)
    _CACHE[path] = (key, value)
    return value


def tail(path, nbytes=16384):
    """The last chunk of a log, as text.

    Read as BYTES and decoded with `errors="replace"`: these logs carry XLA's own stderr
    noise and occasionally non-UTF-8, and a decode error here would blank the one thing a
    user looks at when a run misbehaves.
    """
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - nbytes))
            data = fh.read()
    except OSError:
        return ""
    if size > nbytes:
        data = data.split(b"\n", 1)[-1]
    return data.decode("utf-8", "replace")


def _eta(per_step, remaining):
    if not per_step or remaining <= 0:
        return None
    recent = per_step[-12:]
    return statistics.median(recent) * remaining


# ---------------------------------------------------------------------------
# DESCENT
# ---------------------------------------------------------------------------

def _parse_trajectory(path):
    try:
        with open(path) as fh:
            run = json.load(fh)
    except (OSError, ValueError):
        return None
    steps = run.get("steps") or []
    settings = run.get("settings") or {}
    total = int(settings.get("steps") or 0)
    last = steps[-1] if steps else {}
    # THE ROW COUNT IS NOT THE STEP COUNT.  Row 0 is the starting iterate -- a 122-step
    # run files 123 rows, numbered 0..122 -- so `len(steps)` reports 101% at the finish.
    # The row's own `step` index is the honest numerator.
    done = int(last.get("step") or 0)
    report = last.get("report", {}) or {}
    per_step = [float(s.get("wall_s") or 0.0) for s in steps if s.get("wall_s")]
    best = run.get("best") or {}
    return {
        "fraction": (done / total) if total else None,
        "done": done, "total": total,
        "eta_seconds": _eta(per_step, max(0, total - done)),
        "elapsed_seconds": settings.get("elapsed_s"),
        "detail": {
            "loss": last.get("loss"),
            "|grad|": last.get("grad_norm"),
            "lr": last.get("lr"),
            "axle drop (mm)": report.get("axle_drop_mean_mm"),
            "utilisation": report.get("stress_utilisation"),
            "mass (g)": report.get("mesh_mass_g"),
            "min scaled J": report.get("min_scaled_jacobian"),
        },
        "series": [{"step": s.get("step"), "loss": s.get("loss"),
                    "drop": (s.get("report") or {}).get("axle_drop_mean_mm"),
                    "util": (s.get("report") or {}).get("stress_utilisation")}
                   for s in steps],
        # tier 0 is the only shippable one; 1 and 2 are not promotion candidates.
        "tier": (best.get("selection") or {}).get("tier"),
        "best_loss": best.get("loss"),
        "events": run.get("events") or [],
    }


def _descent_from_log(path):
    """Fallback for the window before the trajectory file exists, and for a run whose
    --out we did not get to redirect."""
    text = tail(path, 65536).encode("utf-8", "replace")
    rows = _STEP_RE.findall(text)
    if not rows:
        return None
    total_m = _S3_TOTAL_RE.search(text)
    total = int(total_m.group(1)) if total_m else 0
    done = int(rows[-1][0])          # the step index, not a count -- see _parse_trajectory
    per_step = [float(r[6]) for r in rows]
    last = rows[-1]

    def _f(v):
        try:
            return float(v)
        except ValueError:
            return None
    return {"fraction": (done / total) if total else None, "done": done, "total": total,
            "eta_seconds": _eta(per_step, max(0, total - done)),
            "detail": {"loss": _f(last[1]), "|grad|": _f(last[2]), "lr": _f(last[3]),
                       "axle drop (mm)": _f(last[4]), "utilisation": _f(last[5])},
            "series": [{"step": int(r[0]), "loss": _f(r[1])} for r in rows],
            "events": []}


# ---------------------------------------------------------------------------
# GA / PYTEST / PHASES
# ---------------------------------------------------------------------------

def _ga(path):
    text = tail(path, 65536).encode("utf-8", "replace")
    rows = _GEN_RE.findall(text)
    total_m = _GA_TOTAL_RE.search(tail(path, 4096).encode("utf-8", "replace")) \
        or _GA_TOTAL_RE.search(text)
    total = int(total_m.group(2)) if total_m else 0
    if not rows:
        return {"fraction": None, "done": 0, "total": total, "detail": {}, "series": []}
    done = int(rows[-1][0])
    return {"fraction": (done / total) if total else None, "done": done, "total": total,
            "detail": {"best loss": float(rows[-1][1]), "mean loss": float(rows[-1][2])},
            "series": [{"step": int(r[0]), "loss": float(r[1])} for r in rows]}


def _pytest(path):
    text = tail(path, 32768).encode("utf-8", "replace")
    pct = _PCT_RE.findall(text)
    summary = None
    for line in reversed(text.splitlines()):
        if b"passed" in line or b"failed" in line or b"error" in line:
            summary = line.decode("utf-8", "replace").strip()
            break
    return {"fraction": (int(pct[-1]) / 100.0) if pct else None,
            "done": int(pct[-1]) if pct else 0, "total": 100,
            "detail": {"result": summary} if summary else {}, "series": []}


def _phases(path, names):
    text = tail(path, 131072)
    seen = [n for n in names if n in text]
    current = seen[-1] if seen else None
    return {"fraction": (len(seen) / len(names)) if names else None,
            "done": len(seen), "total": len(names),
            "detail": {"stage": current} if current else {},
            "phases": [{"name": n, "done": n in seen} for n in names], "series": []}


# ---------------------------------------------------------------------------

def read(job):
    """Progress for one job. Never raises -- a job whose log is unreadable still lists."""
    kind = job.get("progress", "plain")
    log = job.get("log")
    out = {"kind": kind, "fraction": None, "detail": {}, "series": []}
    try:
        if kind == "descent":
            traj = (job.get("outputs") or {}).get("trajectory")
            got = _cached(traj, _parse_trajectory) if traj else None
            if got is None and log:
                got = _cached(log, _descent_from_log)
            if got:
                out.update(got)
        elif kind == "ga" and log:
            out.update(_cached(log, _ga) or {})
        elif kind == "pytest" and log:
            out.update(_cached(log, _pytest) or {})
        elif kind == "phases" and log:
            names = tuple(job.get("phases") or ())
            got = _cached(log, lambda p: _phases(p, names)) if names else None
            if got:
                out.update(got)
    except Exception as exc:                       # a bad log must not break the listing
        out["error"] = f"{type(exc).__name__}: {exc}"
    if job.get("state") in ("finished",):
        out["fraction"] = 1.0
    if out.get("fraction") is not None:
        out["fraction"] = min(1.0, max(0.0, out["fraction"]))
    return out
