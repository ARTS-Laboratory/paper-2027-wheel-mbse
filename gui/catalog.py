"""What this GUI can run, as data.

ONE ENTRY PER THING A USER CAN LAUNCH, and every entry answers the same four questions:
how to start it, where its output goes, how to read its progress, and what it will cost.

WHY A CATALOG RATHER THAN A FORM PER TARGET.  The Makefile has ~55 phony targets and most
of them are one-off study drivers whose answers are already recorded in PLAN.md.  A
bespoke panel each would be forty forms describing experiments nobody is re-running; a
table of specs plus one generic renderer covers all of them and keeps the special-casing
in one file where it can be read at a glance.

TWO LAUNCH STYLES, AND THE CHOICE IS NOT COSMETIC.

  `make <target> VAR=value`  is preferred wherever the recipe is parameterised enough,
  because the Makefile exports PYTHONPATH and five pinned thread/XLA variables and those
  are a CORRECTNESS setting: without XLA_FLAGS the adjoint gradient is not bit-reproducible
  across processes (Makefile's own measurement, 3.33e-16 disagreement on a `coarse`
  adjoint).  Shelling out to make inherits them and cannot forget.

  Direct argv is the fallback, used where a recipe has no override variables to redirect
  its output with -- `stage3` is the case that forces it: the recipe is a bare
  `$(PY_OPT) src/wheel_stage3.py` with no VAR hooks, and two runs at the default
  --out/--best-out clobber each other in the repo root.  Direct argv MUST be paired with
  `wheel_pool.worker_env()` (jobs.py does this), never a hand-built environment.

COSTS CARRY THEIR BASIS.  Every number in COST below is either measured and cited, or
extrapolated and said to be.  A GUI that shows "about an hour" without saying where the
hour came from is how a 27x-under estimate survived in the `stage3` recipe comment for
months.
"""

import os
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A venv puts its interpreter in `bin/python` everywhere except Windows, where it is
# `Scripts/python.exe`.  Both names are built here rather than probed for existence: an
# absent venv must reach the user as the failure it is -- "no such file: .venv-opt/bin/
# python", naming the env that was not built -- and not as a target list that silently
# came back short.
_VENV_BIN = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
PY_OPT = os.path.join(ROOT, ".venv-opt", *_VENV_BIN)
PY_CAD = os.path.join(ROOT, ".venv-cad", *_VENV_BIN)


# ---------------------------------------------------------------------------
# COST MODEL
# ---------------------------------------------------------------------------
# Seconds for ONE PHASE of a value+gradient evaluation, by mesh config, IN STEADY STATE.
#
# `coarse` IS THE MEASURED ONE: 151.42 s for a full 8-phase evaluation on the filleted
# mesh (Makefile's `stage3` recipe comment, re-measured 2026-09-03), so 18.93 s a phase.
# `medium` is that scaled by the paired 273 s / 58.6 s value+grad reading in the
# `svk-medium` comment -> 4.66x.  `fine` is medium x3, from the `gci` comment's
# medium->fine ratio.  `smoke` is scaled DOWN from coarse by element count and is the
# least trustworthy row here -- see FIRST_EVAL_S.
SECONDS_PER_PHASE = {"smoke": 3.2, "coarse": 18.93, "medium": 88.2, "fine": 264.6}

# THE FIRST EVALUATION IS NOT LIKE THE OTHERS AND LEAVING IT OUT MISPRICES SHORT RUNS.
# In `stage3_refillet_shipped_r2.log` step 0 took 1391 s against a ~210 s steady state:
# the jit trace of the adjoint is a one-off of roughly 1180 s that every run pays whether
# it takes three steps or three hundred.  A five-step job priced without it is wrong by
# more than an order of magnitude, which is exactly the direction the `stage3` recipe
# comment records being wrong in for months.
#
# Measured at `coarse`.  Applied at every rung because the trace is dominated by graph
# construction rather than by mesh size -- which is what a `smoke` calibration on
# 2026-09-09 showed: 3 steps at `smoke` had not reached step 0 after six minutes.
FIRST_EVAL_S = 1180.0

# Peak resident set for ONE descent process, GiB, as an AFFINE FIT ON TWO MEASURED POINTS
# rather than a proportion.
#
#   coarse  43.4 GiB   Makefile `stage3` recipe comment, "in one process"
#   smoke  >22.9 GiB   measured here 2026-09-09 under a 40 GiB cap, still climbing when
#                      the run was stopped -- so the intercept below is a FLOOR
#
# A PROPORTIONAL MODEL WAS TRIED FIRST AND WAS WRONG.  Scaling 43.4 GiB by element count
# priced `smoke` at 7.2 GiB; the cgroup cap that produced put the run into swap at 8.9 GiB
# and it went on to exceed 22.9.  Most of a descent's memory is the jax/XLA working set
# and the adjoint's factorisation, and neither shrinks with the mesh.  So: a large
# mesh-independent baseline, plus a mesh-proportional part fitted so `coarse` lands on its
# measured value.
#
# ABOVE `coarse` THIS IS A LOWER BOUND.  A direct sparse solve grows faster than its
# element count, and under-stating memory is the dangerous direction on a box this project
# has already OOMed twice.
GIB_BASELINE = 23.0
GIB_DESCENT_COARSE = 43.4
GIB_PER_WORKER = 12.7           # pre-fillet `svk` comment; now only a config POOL_GIB lacks

# `n_span * n_thick` per spoke block, from `wheel_wheel.CONFIGS`, normalised to coarse.
ELEMENT_RATIO = {"smoke": 32 / 192, "coarse": 1.0, "medium": 576 / 192, "fine": 1536 / 192}

SECONDS_BASIS = {
    "smoke": "steady-state scaled DOWN from coarse by element count and NOT measured; "
             "the one-off trace dominates a short run",
    "coarse": "measured 2026-09-03 -- 151.42 s / 8 phases on the filleted mesh",
    "medium": "coarse x 4.66 (paired 273 s vs 58.6 s value+grad)",
    "fine": "medium x 3 (gci ladder ratio)",
}
GIB_BASIS = ("affine on two measured points -- 43.4 GiB at coarse, >22.9 GiB at smoke; "
             "a LOWER bound above coarse")
POOL_BASIS = ("pool: parent + workers x worker from wheel_pool.POOL_GIB, kernel marks of live "
              "descents rounded up to whole GiB -- an UPPER bound (PLAN.md §167, §171, §173)")


def _wheel_pool():
    """`src/wheel_pool`, imported the way `jobs._env` imports it and for the same reason: its
    numbers are the ones the launched run obeys, so a copy here would be a second place for
    them to drift."""
    import sys
    if os.path.join(ROOT, "src") not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, "src"))
    import wheel_pool
    return wheel_pool


def descent_gib(cfg, width):
    """`(peak GiB, is an upper bound)` for a descent at `cfg` with `width` phase workers.

    A POOL IS PRICED FROM `wheel_pool.POOL_GIB`, the pair `--workers -1` sizes itself by.
    The affine model's 12.7 GiB a worker priced four `coarse` workers at 81.5 GiB against
    49.72 measured, and refused every pooled `coarse` descent this box runs (PLAN.md §170
    §4, §171 §2).  Serial, and a pool at a config with no measured pair, stay affine.
    """
    pair = _wheel_pool().POOL_GIB.get(cfg) if width >= 2 else None
    if pair:
        worker, parent = pair
        return parent + width * worker, True
    ratio = ELEMENT_RATIO.get(cfg, 1.0)
    one = GIB_BASELINE + (GIB_DESCENT_COARSE - GIB_BASELINE) * ratio
    return one + GIB_PER_WORKER * ratio * max(0, width - 1), False


@dataclass(frozen=True)
class Param:
    """One knob the user can turn before launching."""
    name: str
    kind: str                    # int | float | str | choice | bool
    default: object
    choices: tuple = ()
    help: str = ""


@dataclass(frozen=True)
class Target:
    key: str
    label: str
    group: str                   # pipeline | mbse | gates | studies
    blurb: str
    progress: str                # descent | ga | phases | pytest | plain
    params: tuple = ()
    phases: tuple = ()           # for progress == "phases": the ordered stage names
    heavy: bool = False          # needs the confirm dialog and the concurrency guard
    interpreter: str = "opt"     # opt | cad
    # argv(params, rundir) -> list[str].  rundir is where output must land.
    argv: object = None
    # cost(params) -> (seconds, peak_gib, basis[, peak_gib is an upper bound]) | None
    cost: object = None
    # outputs(params, rundir) -> {label: path}; files the run is expected to write.
    outputs: object = field(default=None)
    # writes(params) -> [committed artifact names this run would OVERWRITE], or None.
    # A study artifact is EVIDENCE: PLAN.md 119 declined to refresh the MBSE ones on a new
    # genome precisely because "refreshing one on a new genome does not update a stale
    # number; it rewrites the evidence under a conclusion the tree still makes."  So a
    # GUI must not rewrite one as a side effect of someone pressing Run -- it must say so
    # first, and default to not doing it.
    writes: object = field(default=None)


def _truthy(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def _make(target, **vars_):
    """A `make` invocation with variable overrides."""
    return ["make", target] + [f"{k}={v}" for k, v in vars_.items() if v not in (None, "")]


def _cfg(p):
    return p.get("config", "coarse")


# ---------------------------------------------------------------------------
# THE PIPELINE MAIN LINE
# ---------------------------------------------------------------------------

def _stage3_argv(p, rundir):
    """Direct argv, NOT `make stage3`.

    The recipe is a bare `$(PY_OPT) src/wheel_stage3.py` with no override variables, so
    there is no way to redirect --out/--best-out through make, and both default to fixed
    names in the repo root.  Two GUI jobs at the defaults would overwrite one another's
    trajectory mid-descent.  jobs.py supplies the pinned environment make would have.
    """
    argv = [PY_OPT, "-u", os.path.join("src", "wheel_stage3.py"),
            "--config", str(p.get("config", "coarse")),
            "--steps", str(p.get("steps", 60)),
            "--n-phase", str(p.get("n_phase", 8)),
            "--phase-scheme", str(p.get("phase_scheme", "rqmc")),
            "--kinematics", str(p.get("kinematics", "svk")),
            "--workers", str(p.get("workers", 0)),
            "--start", str(p.get("start", "best")),
            "--log-every", str(p.get("log_every", 1)),
            "--out", os.path.join(rundir, "stage3_run.json"),
            "--best-out", os.path.join(rundir, "stage3_best.json")]
    # --min-wall and --requirements together are REFUSED by wheel_stage3 itself (the floor
    # sets 4 of the 14 genes at the optimum and the record can only carry one).  Pass at
    # most one, and let the driver do the refusing if a caller forces both.
    if p.get("requirements"):
        argv += ["--requirements", str(p["requirements"])]
    elif p.get("min_wall"):
        argv += ["--min-wall", str(p["min_wall"])]
    if p.get("genome"):
        argv += ["--genome", str(p["genome"])]
    return argv


def _stage3_cost(p):
    cfg = _cfg(p)
    per = SECONDS_PER_PHASE.get(cfg)
    if per is None:
        return None
    steps = int(p.get("steps", 60))
    n_phase = int(p.get("n_phase", 8))
    workers = int(p.get("workers", 0))
    starts = 16 if p.get("start") == "all" else 1
    # `-1` is priced at the width `default_workers` gives it now, the call the run makes a
    # moment later.  Priced as serial it was 43.4 GiB against the 49.72 its four workers held
    # (PLAN.md §171 §2).  A config it refuses raises here, and `jobs.plan` blocks on that.
    if workers < 0:
        workers = _wheel_pool().default_workers(n_phase, cfg)
    # Workers parallelise the phase loop, so wall time divides by the effective width.
    width = max(1, min(workers, n_phase)) if workers > 0 else 1
    seconds = FIRST_EVAL_S + steps * starts * n_phase * per / width
    gib, bound = descent_gib(cfg, width)
    return (seconds, gib, SECONDS_BASIS[cfg] + "; " + (POOL_BASIS if bound else GIB_BASIS),
            bound)


TARGETS = [
    Target(
        key="smoke", label="Smoke (GA)", group="pipeline",
        blurb="Tiny GA run that exercises every code path end to end. Proves the pipeline "
              "is wired up; the genome it produces is throwaway.",
        progress="ga", interpreter="opt",
        argv=lambda p, rundir: _make("smoke"),
        cost=lambda p: (25.0, 1.0, "order-of-magnitude; `make help` says seconds"),
    ),
    Target(
        key="ga", label="GA search", group="pipeline",
        blurb="Full GA over the analytical beam model, then hands off to the STEP "
              "exporter. Writes best_solution.json in the repo root.",
        progress="ga", interpreter="opt",
        argv=lambda p, rundir: _make("ga"),
        cost=lambda p: (600.0, 2.0, "order-of-magnitude; `make help` says minutes"),
    ),
    Target(
        key="elites", label="GA + elites", group="pipeline",
        blurb="The same GA run, plus the final population's distinct genomes as "
              "stage2_elites.json -- the multi-start set Stage 3 begins from.",
        progress="ga", interpreter="opt",
        argv=lambda p, rundir: _make("elites"),
        cost=lambda p: (600.0, 2.0, "order-of-magnitude; `make help` says minutes"),
    ),
    Target(
        key="stage3", label="Stage-3 descent", group="pipeline",
        blurb="Projected Adam on the FEA objective. Writes its trajectory every step, so "
              "this is the one target with a real progress bar and a real ETA.",
        progress="descent", heavy=True, interpreter="opt",
        params=(
            Param("config", "choice", "coarse", ("smoke", "coarse", "medium", "fine"),
                  "mesh fidelity; cost scales ~4.7x per rung"),
            Param("steps", "int", 60, (), "descent steps"),
            Param("n_phase", "int", 8, (), "phase stencil width"),
            Param("phase_scheme", "choice", "rqmc", ("rqmc", "uniform", "iid"),
                  "rqmc shifts the stencil by a random lattice offset; uniform does not"),
            Param("kinematics", "choice", "svk", ("svk", "linear"),
                  "svk since PLAN.md 32; linear does not rank designs the same way"),
            Param("workers", "int", 0, (),
                  "0 serial, -1 auto, N literal. N is also the only memory cap"),
            Param("start", "choice", "best", ("best", "all"), "start point(s)"),
            Param("min_wall", "float", "", (), "printable wall floor mm (blank = shipped)"),
            Param("requirements", "str", "", (),
                  "path to a requirements.json from the MBSE panel (blank = shipped mission)"),
        ),
        argv=_stage3_argv, cost=_stage3_cost,
        outputs=lambda p, rundir: {
            "trajectory": os.path.join(rundir, "stage3_run.json"),
            "best genome": os.path.join(rundir, "stage3_best.json")},
    ),
    Target(
        key="export", label="STEP export", group="pipeline",
        blurb="Rebuild the CAD solid from a genome. Runs in .venv-cad -- the only thing "
              "that does -- and writes a .step, a no-fillet fallback and a manifest.",
        progress="plain", interpreter="cad",
        params=(Param("genome", "str", "", (), "genome record (blank = best_solution.json)"),),
        argv=lambda p, rundir: _make("export", EXPORT_GENOME=p.get("genome", "")),
        cost=lambda p: (57.1, 1.5, "measured -- export_seconds in wheel_step_manifest.json"),
    ),
]

# ---------------------------------------------------------------------------
# THE MBSE FAMILY
# ---------------------------------------------------------------------------
TARGETS += [
    Target(
        key="mbsebase", label="Baseline derivation", group="mbse",
        blurb="MBSE_PLAN Step 0: the derivations run BACKWARDS -- what all-up weight, sink rate, ambient and service life the four shipped constants imply. Solves nothing.",
        progress="plain", interpreter="opt",
        params=(Param("refresh_artifact", "bool", False, (),
                      "write studies/study_mbse_baseline.json itself instead of a copy in this job's "
                      "run directory. That file is committed EVIDENCE -- refreshing it "
                      "rewrites the record, it does not update it"),),
        argv=lambda p, rundir: _make(
            "mbsebase", MBSEBASE_OUT=("" if _truthy(p.get("refresh_artifact"))
                                else os.path.join(rundir, "study_mbse_baseline.json"))),
        writes=lambda p: ["studies/study_mbse_baseline.json"] if _truthy(p.get("refresh_artifact")) else [],
        cost=lambda p: (1.0, 0.3, "measured -- `make help` says ~1 s"),
    ),
    Target(
        key="mbsecal", label="Weight calibration", group="mbse",
        blurb="MBSE_PLAN Step 4: what portfolio DEFAULT_WEIGHTS is already running, in points, with the loss-share reading refuted. Solves nothing.",
        progress="plain", interpreter="opt",
        params=(Param("refresh_artifact", "bool", False, (),
                      "write studies/study_mbse_calibration.json itself instead of a copy in this job's "
                      "run directory. That file is committed EVIDENCE -- refreshing it "
                      "rewrites the record, it does not update it"),),
        argv=lambda p, rundir: _make(
            "mbsecal", MBSECAL_OUT=("" if _truthy(p.get("refresh_artifact"))
                                else os.path.join(rundir, "study_mbse_calibration.json"))),
        writes=lambda p: ["studies/study_mbse_calibration.json"] if _truthy(p.get("refresh_artifact")) else [],
        cost=lambda p: (1.0, 0.3, "measured -- `make help` says ~1 s"),
    ),
    Target(
        key="mbsescore", label="Five profiles scored", group="mbse",
        blurb="MBSE_PLAN Step 5: the shipped wheel against five named requirement profiles, a compliance table each. Gates that at least one comes back NON-COMPLIANT -- a verifier that cannot fail is a formatting exercise.",
        progress="plain", interpreter="opt",
        params=(Param("refresh_artifact", "bool", False, (),
                      "write studies/study_mbse_score.json itself instead of a copy in this job's "
                      "run directory. That file is committed EVIDENCE -- refreshing it "
                      "rewrites the record, it does not update it"),),
        argv=lambda p, rundir: _make(
            "mbsescore", MBSESCORE_OUT=("" if _truthy(p.get("refresh_artifact"))
                                else os.path.join(rundir, "study_mbse_score.json"))),
        writes=lambda p: ["studies/study_mbse_score.json"] if _truthy(p.get("refresh_artifact")) else [],
        cost=lambda p: (7 * 8 * SECONDS_PER_PHASE["coarse"], 8.0, "7 evaluations with gradients at coarse/svk/8-phase"),
    ),
]

# ---------------------------------------------------------------------------
# GATES
# ---------------------------------------------------------------------------
STUDY_GATES = ("study_mesh_quality", "study_wheel_mesh", "study_beam_agreement",
               "study_wheel_fea", "study_gnl", "study_contact", "study_gradient",
               "study_objective", "study_stage3")

TARGETS += [
    Target(
        key="test", label="Test suite", group="gates",
        blurb="pytest in .venv-opt. 957 collected; xfail_strict is on, so an xfail that "
              "starts passing is a failure.",
        progress="pytest", heavy=True, interpreter="opt",
        argv=lambda p, rundir: _make("test"),
        cost=lambda p: (29 * 60.0, 20.0,
                        "measured 2026-09-09 (PLAN 154): 957 collected in 29 min"),
    ),
    Target(
        key="studies", label="Verification gates", group="gates",
        blurb="The nine milestone gates, sequentially. Each writes a JSON report and "
              "exits nonzero when its own measurement is untrustworthy.",
        progress="phases", phases=STUDY_GATES, heavy=True, interpreter="opt",
        argv=lambda p, rundir: _make("studies"),
        cost=lambda p: (3 * 3600.0, 20.0, "dominated by study_stage3 (M8b-i, ~2 h 45 m)"),
        # The recipe is nine hardcoded commands at their default --out, with no variable
        # to redirect them. Rewriting the nine gate artifacts IS what this target is for,
        # so it is not blocked -- but it is said out loud before it happens.
        writes=lambda p: ["studies/%s.json" % g for g in STUDY_GATES],
    ),
]

BY_KEY = {t.key: t for t in TARGETS}


def generic_target(make_target):
    """Any other .PHONY target, run as-is with a plain log and no progress model.

    The escape hatch that makes "control the whole process" true rather than nearly true.
    No cost estimate is offered, because inventing one for a driver whose recipe comment
    was not read would be worse than saying nothing.
    """
    return Target(
        key=f"make:{make_target}", label=f"make {make_target}", group="studies",
        blurb="Run this Makefile target as-is. No progress model and no cost estimate -- "
              "read its recipe comment in the Makefile before starting a long one.",
        progress="plain", heavy=True, interpreter="opt",
        argv=lambda p, rundir, t=make_target: _make(t),
    )


def resolve(key):
    if key in BY_KEY:
        return BY_KEY[key]
    if key.startswith("make:"):
        return generic_target(key.split(":", 1)[1])
    raise KeyError(f"no such target {key!r}")
