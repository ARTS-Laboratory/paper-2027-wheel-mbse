"""Every study driver refuses to file a degraded run under its committed name.

PLAN.md §43.  §41 found that `study_contact --quick` wrote smoke-mesh data into the
committed `study_contact.json`, where G1 reads 4.394e-04 with both halves passing against
1.7198e-03 and a red `regime_pass` at the real config — a FALSE GREEN standing in for a
RED one.  `study_contact` was the only driver with any guard; the other eight had none.

TWO DIRECTIONS, AND THE FIRST ONE IS THE DANGEROUS TEST TO GET WRONG.  A guard that fires
on the recipe's own invocation would take `make studies` down — five hours, nine drivers —
and it would do it at the END of each driver's run, since `make` only sees the exit
status.  So the exact argv of every Makefile target that writes one of these artifacts is
asserted to pass.

The guard is reached and then execution is stopped, so these cost milliseconds rather than
the recipe's five hours: `refuse_degraded_out` is wrapped to run for real and then raise.
That means the REAL conditions are evaluated with the REAL parsed arguments — a test that
stubbed the guard out would assert nothing.

WHAT §131 CHANGED, AND WHY THIS FILE IS THE REASON THE GAP LASTED FOUR MONTHS.  §129.5
censused the guard across every driver in `studies/` and found fourteen with none, ten of
them exposed — `study_m9.py --quick --out study_m9.json` was §41's invocation with §41's
consequence, unobstructed.  This file could not see any of it, and not because it was
missing rows: it covered the nine drivers in `make studies` and never passed `--out` at
all, so it asserted things about DEFAULT names only.  A guard that is wrong about a
driver's second name is invisible to a test that only ever supplies the first — and the
drivers here write more names than they have defaults.  The `--out` column is §129's
successor 2 asking for exactly that, and the mutation catcher below now asserts the whole
CALL SEQUENCE rather than that some guard fired.
"""

import ast
import glob
import json
import os
import re
import shlex
import signal
import subprocess
import sys

import pytest

import _gate_guard
import project_paths as PP
# Cheap at import: the driver's own top level is argparse/glob/json/os/time, and both cell
# functions import `study_wheel_fea`/`study_gnl` lazily inside themselves.
import study_reds_ratio_stability as reds


class _GuardPassed(Exception):
    """The real guard was called and did not refuse."""


_RUNAWAY_S = 20   # far outside any guard-and-stop path, far inside any real study


class _DriverRanAway(BaseException):
    """A driver got past every guard this table knows about.

    A `BaseException` on purpose: several drivers wrap their work in `except Exception`,
    and a timeout those swallow is a timeout that does not stop anything.
    """


# (module, the committed names it guards IN CALL ORDER, argvs that must PASS, degraded
#  argvs that must be refused)
#
# The passing argvs are copied from the Makefile targets that write these artifacts.  If a
# target changes, this table has to change with it, and the first assertion below is what
# says so.
#
# THE SECOND AND THIRD COLUMNS ARE §131's REPAIR AND THE REASON THIS FILE COULD NOT SEE
# §129.5's GAP.  Until §131 every row named ONE artifact, passed NO `--out`, and gave one
# recipe argv — so a guard that was wrong about a driver's SECOND name was invisible to a
# test that only ever supplied the first, and four of the drivers below have more than one
# name or more than one gate.  `study_stage3` writes four tracked artifacts from four
# targets and guarded one of them; `study_corner_singularity` writes two; and three
# drivers spell their single name two ways (an absolute default, and the `studies/...`
# the Makefile passes) so a row that exercised only the default could not reach the
# spelling people actually type.
DRIVERS = [
    ("study_mesh_quality",   ["study_mesh_quality.json"],
     [["--samples", "2000"]],
     [["--samples", "500"], ["--config", "smoke"], ["--no-plot"]]),
    ("study_wheel_mesh",     ["study_wheel_mesh.json"],
     [["--samples", "200"]],
     [["--quick"], ["--samples", "50"], ["--config", "smoke"], ["--no-plot"]]),
    ("study_beam_agreement", ["study_beam_agreement.json"],
     [[]],
     [["--quick"], ["--genome", "stage3_knee_best_medium.json"], ["--no-plot"]]),
    ("study_wheel_fea",      ["study_wheel_fea.json"],
     [[]],
     [["--quick"], ["--config", "smoke"], ["--no-plot"]]),
    ("study_gnl",            ["study_gnl.json"],
     [[]],
     [["--quick"], ["--config", "smoke"], ["--no-plot"]]),
    ("study_contact",        ["study_contact.json"],
     [[]],
     [["--quick"], ["--config", "medium"], ["--kinematics", "svk"],
      ["--sections", "penalty"], ["--no-plot"]]),
    ("study_gradient",       ["study_gradient.json"],
     [[]],
     [["--quick"], ["--config", "smoke"], ["--kinematics", "svk"], ["--no-plot"]]),
    ("study_objective",      ["study_objective.json"],
     [[]],
     [["--quick"], ["--config", "smoke"], ["--elites", "elite10.log"], ["--no-plot"]]),
    # FOUR NAMES, FOUR TARGETS: `studies`, `m8bi5`, `m8bi6`, `m8bii1`.  The three
    # secondary argvs are what §129.5 measured as accepted — `--sections` is degrading for
    # `study_stage3.json` and IS the gate for the other three, so each name carries its
    # own answer and the recipe column has to hold all four.
    ("study_stage3",         ["study_stage3.json", "study_stage3_m8bi5.json",
                              "study_stage3_pnorm.json", "study_stage3_pool.json"],
     [[],
      ["--sections", "mesh_convergence,multistart", "--out", "study_stage3_m8bi5.json"],
      ["--sections", "mesh_convergence", "--ladder-p", "1,2,3,4,6,8,12,16,24,30",
       "--out", "study_stage3_pnorm.json"],
      ["--sections", "phase_pool", "--out", "study_stage3_pool.json"]],
     [["--quick"], ["--config", "smoke"], ["--sections", "direction"],
      ["--ladder-p", "1,2,3"], ["--no-plot"],
      # The secondary names, which had no guard at all until §131.
      ["--sections", "mesh_convergence,multistart", "--config", "smoke",
       "--out", "study_stage3_m8bi5.json"],
      ["--sections", "mesh_convergence", "--out", "study_stage3_m8bi5.json"],
      ["--sections", "mesh_convergence", "--ladder-p", "1,2,3",
       "--out", "study_stage3_pnorm.json"],
      ["--sections", "phase_pool", "--no-plot", "--out", "study_stage3_pool.json"]]),
    # §131: the drivers §129.5 censused as EXPOSED — a fidelity or section flag reaching a
    # tracked artifact with nothing in the way.  `study_kinematics_rank` is the tenth and
    # is not here: it was being worked in a concurrent session, and its guard waits on
    # which of its two artifacts is canonical (§130 successor 0).
    ("study_m9",             ["study_m9.json"],
     [["--out", "study_m9.json"]],
     [["--quick", "--out", "study_m9.json"],
      ["--genome", "best_solution_ga_beam.json", "--out", "study_m9.json"]]),
    ("study_m9_buckling",    ["study_m9_buckling.json"],
     [["--out", "study_m9_buckling.json"]],
     [["--quick", "--out", "study_m9_buckling.json"],
      ["--config", "smoke", "--out", "study_m9_buckling.json"],
      ["--genome", "best_solution_ga_beam.json", "--out", "study_m9_buckling.json"]]),
    ("study_svk_rescore",    ["study_svk_rescore.json"],
     [["--config", "medium", "--workers", "0", "--out", "study_svk_rescore.json"]],
     [["--skip-control", "--out", "study_svk_rescore.json"],
      ["--only", "shipped", "--out", "study_svk_rescore.json"],
      ["--config", "coarse", "--out", "study_svk_rescore.json"],
      ["--n-phase", "4", "--out", "study_svk_rescore.json"]]),
    # THREE targets, ONE name, and all three PASS: the writer merges under a key per arm,
    # so `--sweep` alone does not drop `rungs`.  The refusals are fidelity INSIDE an arm,
    # which the merge does not protect.
    ("study_reds_hub_share", ["study_reds_hub_share.json"],
     [["--sweep", "--attribute", "--rungs", "--config", "coarse",
       "--configs", "smoke,coarse,medium,fine,ultra",
       "--out", "study_reds_hub_share.json"],
      ["--sweep", "--fillet", "--config", "coarse",
       "--out", "study_reds_hub_share.json"],
      ["--rungs", "--fillet", "--kinematics", "linear",
       "--configs", "smoke,coarse,medium,fine,ultra",
       "--out", "study_reds_hub_share.json"]],
     [["--sweep", "--config", "smoke", "--out", "study_reds_hub_share.json"],
      ["--sweep", "--points", "3", "--out", "study_reds_hub_share.json"],
      ["--sweep", "--genome", "ga_beam", "--out", "study_reds_hub_share.json"],
      ["--rungs", "--configs", "smoke,coarse", "--out", "study_reds_hub_share.json"]]),
    # No Makefile target: the gate is this driver's own defaults, checked against the
    # committed artifact's two rungs rather than against the five-rung table in its
    # docstring, which was never filed here.
    ("study_knee_rungs",     ["study_knee_rungs.json"],
     [[]],
     [["--rungs", "smoke"], ["--rungs", "smoke,coarse,medium,fine,ultra"],
      ["--kinematics", "linear"], ["--phases", "4"],
      ["--genome", "best_solution_ga_beam.json"]]),
    ("study_hub_cap",        ["study_hub_cap.json"],
     [["--out", "study_hub_cap.json"]],
     [["--sections", "void", "--out", "study_hub_cap.json"],
      ["--designs", "best_solution", "--out", "study_hub_cap.json"],
      ["--t0-sweep", "2,3", "--out", "study_hub_cap.json"]]),
    # TWO artifacts from two targets, and the flags that separate them are degrading for
    # one and mandatory for the other — the case a per-DRIVER guard cannot express.
    ("study_corner_singularity",
     ["studies/study_corner_singularity.json",
      "studies/study_corner_singularity_fillet.json"],
     [["--genome", "best_solution.json", "--ladder", "smoke,coarse,medium,fine",
       "--out", "studies/study_corner_singularity.json"],
      ["--genome", "best_solution.json", "--ladder", "smoke,coarse,medium,fine",
       "--fillet", "genome", "--continuity", "coarse", "--profiles",
       "--out", "studies/study_corner_singularity_fillet.json"]],
     [["--ladder", "smoke", "--out", "studies/study_corner_singularity.json"],
      ["--fillet", "genome", "--out", "studies/study_corner_singularity.json"],
      ["--ladder", "smoke,coarse,medium,fine",
       "--out", "studies/study_corner_singularity_fillet.json"],
      ["--fillet", "genome", "--continuity", "coarse",
       "--out", "studies/study_corner_singularity_fillet.json"]]),
    ("study_deflection_gci", ["studies/study_deflection_gci.json"],
     [["--genome", "best_solution.json", "--ladder", "smoke,coarse,medium,fine",
       "--workers", "0", "--out", "studies/study_deflection_gci.json"]],
     [["--ladder", "smoke", "--out", "studies/study_deflection_gci.json"],
      ["--n-phase", "4", "--out", "studies/study_deflection_gci.json"],
      ["--genome", "best_solution_ga_beam.json",
       "--out", "studies/study_deflection_gci.json"]]),
    ("study_junction_agreement", ["studies/study_junction_agreement.json"],
     [["--genome", "best_solution.json", "--config", "coarse",
       "--out", "studies/study_junction_agreement.json"]],
     [["--config", "smoke", "--out", "studies/study_junction_agreement.json"],
      ["--genome", "best_solution_ga_beam.json",
       "--out", "studies/study_junction_agreement.json"]]),
]

_IDS = [d[0] for d in DRIVERS]


@pytest.fixture
def guard_stops_here(monkeypatch, tmp_path):
    """Run the real guard, then stop — and make `studies/` unreachable while we do.

    THE SECOND HALF IS NOT BELT-AND-BRACES, IT IS THE LESSON THIS FILE WAS WRITTEN BY.
    An earlier draft stopped execution ONLY by having this wrapper raise.  That is fine
    while every driver calls the guard — and catastrophic the moment one does not, which
    is exactly the mutation these tests exist to catch: with the call removed, the wrapper
    never fires, `main()` runs the whole study, and it writes its report to the COMMITTED
    artifact.  Mutating one driver to check the tests could fail did precisely that,
    overwriting `studies/study_mesh_quality.json` and `.jpg` and leaving two stray
    `_probe` files behind.  A test for an artifact-clobbering defect must not be able to
    clobber the artifact.

    Every driver writes through its module-level `HERE` (`os.path.join(HERE, args.out)`),
    so pointing that at `tmp_path` bounds the damage to a temp dir no matter how far
    execution gets.  The guard-was-called assertion below then turns "the driver ran"
    from silent corruption into a plain failure.

    §131 ADDED TWO MORE CONTAINMENTS, BOTH BECAUSE THE WRAPPER NO LONGER RAISES ON THE
    FIRST CALL.  A driver with more than one gate calls the guard once per artifact, and
    stopping at the first call would leave every later one unexercised — `make m8bi5`'s
    argv returns early from `study_stage3.json`'s guard, so the m8bi5 guard is the only
    one that can judge it.  So the wrapper runs them all and raises at the row's LAST
    name, which widens the window in which a driver that has lost a guard call keeps
    executing:

    * `chdir(tmp_path)`, with a `studies/` under it.  Three drivers use `args.out` AS
      GIVEN rather than joining `HERE`, and the Makefile hands them `studies/study_x.json`
      relative to the repo root — a spelling the `HERE` monkeypatch above cannot reach.
      Without this the tests would write the real artifacts through the real path.
    * `signal.alarm`, re-armed per argv.  The full studies behind these drivers are minutes to hours; a
      driver that never reaches its last guard would otherwise run one for real rather
      than fail.  Twenty seconds is far outside any guard-and-stop path (the whole file
      is under a second) and far inside any real study.
    """
    real = _gate_guard.refuse_degraded_out
    calls = []

    def arm(last_name):
        """Wrap the guard so the row's LAST name is where execution stops.  Returns
        `calls`, cleared, so each argv is judged on its own call sequence.

        The alarm is re-armed here rather than once per test: a row runs several argvs,
        and an alarm that has already fired on the first would leave the rest unbounded.
        """
        del calls[:]
        signal.alarm(_RUNAWAY_S)

        def wrapper(ap, args, committed, degraded):
            # `committed` is a name or a tuple of spellings of one name; the first is the
            # canonical one, which is what the table lists.
            name = committed if isinstance(committed, str) else committed[0]
            calls.append(name)
            real(ap, args, committed, degraded)   # SystemExit if this run is degraded
            if name == last_name:
                raise _GuardPassed                # allowed — stop before any solving

        monkeypatch.setattr(_gate_guard, "refuse_degraded_out", wrapper)
        return calls

    for name, *_ in DRIVERS:
        monkeypatch.setattr(__import__(name), "HERE", str(tmp_path))
    (tmp_path / "studies").mkdir()
    monkeypatch.chdir(tmp_path)

    def _ran_away(signum, frame):
        raise _DriverRanAway(
            f"a driver ran for {_RUNAWAY_S} s without reaching its last guard — it has "
            f"lost a guard call, and this alarm is all that stopped it running the real "
            f"study")

    old_handler = signal.signal(signal.SIGALRM, _ran_away)
    try:
        yield arm
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)




def _main(name):
    return __import__(name).main


@pytest.mark.parametrize("name, names, recipes, _degraded", DRIVERS, ids=_IDS)
def test_the_recipe_invocation_is_not_refused(monkeypatch, guard_stops_here,
                                              name, names, recipes, _degraded):
    """Every Makefile invocation that writes one of these artifacts must reach the work.

    This is the assertion protecting the recipes from their own guard.  It fails loudly if
    a condition is written against the wrong default — e.g. guarding `--samples != 2000`
    when the target passes exactly 2000, or comparing `--config` to a literal the driver
    does not actually default to.

    §131: a driver may have several such invocations, and `study_corner_singularity`'s two
    are the case that makes the point — `--fillet genome --continuity coarse --profiles`
    is `corner-fillet`'s gate and would be a degraded run under `corner`'s name.  Both
    must pass here, which is only possible because each artifact carries its own list.

    The `calls` assertion is the mutation catcher, and §131 strengthened it from "the
    guard was called" to "EVERY name this driver is supposed to guard was reached, in
    order".  Dropping the second of four calls is exactly §129.5's defect, and the old
    single-name form could not see it.
    """
    for recipe in recipes:
        calls = guard_stops_here(names[-1])
        monkeypatch.setattr(sys, "argv", [f"{name}.py", *recipe])
        with pytest.raises(_GuardPassed):
            _main(name)()
        assert calls == names, (
            f"{name} {recipe} guarded {calls}, not {names} — a name a driver writes and "
            f"does not guard can be overwritten by a degraded run, which is the whole "
            f"defect §43 closes and the gap §129.5 censused")


@pytest.mark.parametrize("name, _names, _recipes, degraded", DRIVERS, ids=_IDS)
def test_degraded_runs_are_refused_by_name(monkeypatch, guard_stops_here,
                                           name, _names, _recipes, degraded):
    """Each degraded invocation must exit nonzero rather than overwrite the artifact."""
    for argv in degraded:
        guard_stops_here(_names[-1])
        monkeypatch.setattr(sys, "argv", [f"{name}.py", *argv])
        with pytest.raises(SystemExit) as excinfo:
            _main(name)()
        assert excinfo.value.code != 0, f"{name} {argv} was accepted as the gate"


@pytest.mark.parametrize("name, names, _recipes, degraded", DRIVERS, ids=_IDS)
def test_an_explicit_out_lets_a_degraded_run_through(monkeypatch, guard_stops_here,
                                                     name, names, _recipes, degraded):
    """The refusal is about the NAME.  Redirected, every degraded run is allowed.

    Pinned because the cheap way to quieten a noisy guard is to widen it until the
    degraded run cannot happen at all — which would take `make m8bi5`, `make m8bi6`,
    `make m8bii1` and `make contact` with it, all four of which are partial or redirected
    runs that pass their own `--out` for exactly this reason.

    §131 gave the first three of those four their own guarded names, which does NOT
    weaken this test: a redirected run is still allowed, and what changed is that
    `study_stage3_m8bi5.json` is no longer a redirection — it is a gate with a list of
    its own.  `make contact`'s case is untouched and is the sharper one: its documented
    invocation writes a COMMIT-PINNED baseline (`study_contact_e126cc3_svk.json`,
    Makefile's `contact` block) under four degrading flags, so a guard keyed on "is this
    name tracked" would refuse the recipe that created the evidence.
    """
    for argv in degraded:
        guard_stops_here(names[-1])
        monkeypatch.setattr(sys, "argv",
                            [f"{name}.py", *argv, "--out", f"{name}_probe.json"])
        with pytest.raises(_GuardPassed):
            _main(name)()


def test_the_helper_collects_every_reason_rather_than_the_first():
    """A guard that reports one problem at a time is one people route around.

    Checked on the helper directly: the drivers pass their conditions in, and the
    message-building is the part shared by all nine.
    """
    seen = {}

    class _AP:
        def error(self, msg):
            seen["msg"] = msg
            raise SystemExit(2)

    class _Args:
        out = "study_x.json"

    with pytest.raises(SystemExit):
        _gate_guard.refuse_degraded_out(
            _AP(), _Args(), "study_x.json",
            [(True, "reason-one"), (False, "not-this"), (True, "reason-two")])

    assert "reason-one" in seen["msg"] and "reason-two" in seen["msg"]
    assert "not-this" not in seen["msg"]


def test_the_helper_is_silent_when_out_was_redirected():
    class _AP:
        def error(self, msg):                      # pragma: no cover — must not run
            raise AssertionError(f"refused a redirected run: {msg}")

    class _Args:
        out = "somewhere_else.json"

    _gate_guard.refuse_degraded_out(_AP(), _Args(), "study_x.json",
                                    [(True, "degraded")])


# ---------------------------------------------------------------------------
# `study_reds_ratio_stability` — THE ELEVENTH EXPOSED DRIVER, AND THE ONE THE
# TABLE ABOVE CANNOT HOLD (§131 successor 3)
# ---------------------------------------------------------------------------
# EVERY ROW OF `DRIVERS` ASSERTS A PROPERTY OF AN ARGV.  That works because in every one
# of those drivers `--out` defaults to the committed name and a fidelity flag on the SAME
# invocation degrades it, so the guard can read `args` and a test can decide the case from
# argv alone.  THIS DRIVER IS NOT SHAPED THAT WAY, and putting it in the table would state
# something false about it.
#
# Its fidelity flags — `--seed`, `--n`, `--min-wall` — belong to the CELL invocation,
# which prints one JSON line into `$(REDS_CELLS)` and never touches a tracked name.  The
# only invocation that writes the artifact is `--which collect`, which accepts those flags
# and IGNORES them: measured, `--which collect --n 3 --min-wall 1.2 --seed 999` over a
# fixed `--glob` reproduces the unflagged output byte-for-byte.  So no argv distinguishes
# a gate run from a degraded one; what distinguishes them is THE SET OF CELLS THE GLOB
# FINDS, and `make reds-ratio`'s own collect argv is a gate run or a degraded one
# depending on whether the fan-out that preceded it left 109 cells in `/tmp` or 3.
#
# That is also why the first table test's promise — "every Makefile invocation must reach
# the work" — cannot be made about this recipe from argv, and why these tests build the
# cell directory instead of quoting a command line.


def _cells():
    """The committed artifact's own 109 rows, which `collect` accepts as a cell source."""
    import project_paths as PP
    with open(os.path.join(PP.ROOT, "studies",
                           "study_reds_ratio_stability.json")) as fh:
        return json.load(fh)["cells"]


@pytest.fixture
def reds_cells(tmp_path, monkeypatch):
    """Write cell files into a temp dir and contain the driver's writes to another.

    Contained the same way the table's fixture is and for the same reason: these tests
    aim a run at `studies/study_reds_ratio_stability.json`, and if the guard call were
    ever deleted that run must land in a temp tree rather than on the evidence.  This
    driver uses `args.out` AS GIVEN — it never joins `HERE` — so `chdir` with a `studies/`
    beneath it is the containment that matters, not the `HERE` monkeypatch.
    """
    def write(cells, name="cells"):
        d = tmp_path / name
        d.mkdir()
        for i, c in enumerate(cells):
            (d / f"cell{i:03d}.json").write_text(json.dumps(c))
        return str(d / "*.json")

    (tmp_path / "studies").mkdir()
    monkeypatch.chdir(tmp_path)
    return write


COMMITTED = "studies/study_reds_ratio_stability.json"


def test_a_full_grid_is_filed_under_the_committed_name(reds_cells):
    """The gate run — 66 beam and 43 gnl cells, every one holding the rows it asked for."""
    glob_pat = reds_cells(_cells())
    assert reds.main(["--which", "collect", "--glob", glob_pat,
                      "--out", COMMITTED]) == 0
    with open(COMMITTED) as fh:
        assert len(json.load(fh)["cells"]) == reds.N_CELLS_BEAM + reds.N_CELLS_GNL


def test_a_short_glob_may_not_be_filed_under_the_committed_name(reds_cells):
    """§132 §5's shape, and the defect this guard closes.

    Measured before the guard existed: 3 real cells beside 106 EMPTY files — what the
    fan-out leaves when cells die, since `> $(REDS_CELLS)/...json` creates the file before
    the process runs and `2>/dev/null` hides the reason — collected to a 3-cell artifact
    of 843 bytes against 31039, EXIT 0, with `_table` still printing
    "correction_factor_is_defensible true in 0/1 cells  (never — the conclusion holds...)".
    The verdict line survives; every cell under it is gone.
    """
    glob_pat = reds_cells(_cells()[:3])
    with pytest.raises(SystemExit) as excinfo:
        reds.main(["--which", "collect", "--glob", glob_pat, "--out", COMMITTED])
    assert excinfo.value.code != 0
    assert not os.path.exists(COMMITTED), "a short grid was filed as the gate"


def test_an_empty_cell_file_is_silent_and_a_truncated_one_is_not(reds_cells):
    """The two failure shapes are not the same, and only one of them needs the guard.

    `collect` wraps its whole-file `json.loads` in a `try`, but the per-line fallback
    beneath it is OUTSIDE that `try` — so a half-written cell line RAISES and nothing is
    written at all, which is already safe.  A zero-byte file has no lines, contributes no
    cells, and returns cleanly.  The empty file is the one the fan-out actually produces.
    """
    d = os.path.dirname(reds_cells(_cells()[:3]))
    open(os.path.join(d, "empty.json"), "w").close()
    assert len(reds.collect(os.path.join(d, "*.json"))) == 3, "an empty file was not silent"

    with open(os.path.join(d, "truncated.json"), "w") as fh:
        fh.write('{"ratio": 1.5, "cv"')
    with pytest.raises(json.JSONDecodeError):
        reds.collect(os.path.join(d, "*.json"))


def test_a_cell_that_could_not_draw_its_rows_may_not_be_filed(reds_cells):
    """The report's own claim is that every cell returned exactly the rows it asked for.

    A full 109-cell grid can still be degraded one cell at a time, which is the condition
    neither of the count checks can see.  `n_drawn` is NOT this quantity — it counts the
    LHS candidates rejected on the way to `n_rows` and runs 7.5x to 92.5x above it in
    every committed cell.
    """
    cells = [dict(c) for c in _cells()]
    cells[0]["n_rows"] = cells[0]["n"] - 1
    glob_pat = reds_cells(cells)
    with pytest.raises(SystemExit) as excinfo:
        reds.main(["--which", "collect", "--glob", glob_pat, "--out", COMMITTED])
    assert excinfo.value.code != 0
    assert not os.path.exists(COMMITTED)


def test_the_cell_flags_are_inert_on_the_collect_path_and_the_guard_must_not_read_them(
        reds_cells):
    """THE PIN ON THE GUARD'S SHAPE, not on its symptom.

    `--n`, `--seed` and `--min-wall` degrade a CELL and mean nothing to a collect.  A
    guard rewritten to read them — the shape every other driver in this file uses — would
    refuse this run, which is a full grid and IS the gate.  That regression is invisible
    to every other test here, because the cheap way to write this guard is also the wrong
    one for this driver alone.
    """
    glob_pat = reds_cells(_cells())
    assert reds.main(["--which", "collect", "--glob", glob_pat,
                      "--n", "3", "--min-wall", "1.2", "--seed", "999",
                      "--out", COMMITTED]) == 0


def test_a_short_grid_is_allowed_once_it_is_redirected(reds_cells):
    """The refusal is about the NAME here too: re-tabulating a partial grid is routine."""
    glob_pat = reds_cells(_cells()[:3])
    assert reds.main(["--which", "collect", "--glob", glob_pat,
                      "--out", "studies/reds_ratio_probe.json"]) == 0


# ---------------------------------------------------------------------------
# THE TABLE ABOVE IS HAND-COPIED FROM THE MAKEFILE.  THIS READS THE MAKEFILE.
# (§131 successor 1 — the check that lived in a scratch harness)
# ---------------------------------------------------------------------------
# §131 §5 ran every guarded target's real argv through its real parser and real guard and
# found 29 targets / 37 invocations, every guard reached, not one refusal — and then the
# harness was thrown away.  What it was protecting against is a recipe that refuses ITSELF:
# rename `REDS_RATIO_OUT`, or add a flag to a recipe, and the hand-copied column above still
# agrees with itself while `make studies` dies five hours in, at the END of a run, because
# `make` only ever sees the exit status.
#
# THIS IS THE CHEAP HALF AND IT DOES NOT RUN A DRIVER.  §131 §5's harness ran
# `study_fillet_condition_a` to completion and rewrote its artifact — that driver's three
# guard calls sit in conditional paths, so "stop at the last guard" means "run the study".
# Nothing here executes a driver: `make -n` resolves the variables, and the argvs are
# compared against the table, whose own three tests already prove those argvs pass the real
# guard.  Two cheap assertions composed beat one expensive one that can rewrite evidence.


def _make_n_invocations():
    """Every guarded driver invocation the Makefile's own targets expand to.

    THE CONTINUATION JOIN IS NOT A DETAIL, IT IS THE WHOLE ENUMERATION.  These recipes are
    written across several physical lines with trailing backslashes, so a scan that reads
    `make -n` line by line captures `study_contact.py --genome best_solution.json \\` and
    nothing after it.  Measured while writing this: 20 targets / 34 invocations without the
    join, 34 targets / 49 with it — §129.5's census defect exactly, an enumeration that
    silently cannot see half its candidates.

    `shlex.split` IS DELIBERATELY NOT WRAPPED.  A trailing backslash raises there, and the
    draft of this scan caught that and skipped the invocation — which is how the 20/34
    reading looked like a complete census instead of a broken one.  Unwrapped, the same
    breakage is an error rather than a smaller number, and the test below pins the count
    anyway so neither failure mode is quiet.
    """
    phony = subprocess.run(["grep", "-m1", "^.PHONY:", "Makefile"],
                           cwd=PP.ROOT, capture_output=True, text=True).stdout
    out = {}
    for target in phony.split(":", 1)[1].split():
        r = subprocess.run(["make", "-n", target],
                           cwd=PP.ROOT, capture_output=True, text=True)
        if r.returncode:
            continue
        for line in r.stdout.replace("\\\n", " ").splitlines():
            m = re.search(r"studies/(study_[A-Za-z0-9_]+)\.py(.*)$", line)
            if not m or m.group(1) not in _GUARDED_MODULES:
                continue
            # `make reds-ratio`'s fan-out passes `--which $$0` inside `xargs bash -c`: a
            # shell template, not an argv, and there is nothing for a parser to check.
            if re.search(r"\$[0-9{(]|\$\$", m.group(2)):
                continue
            out.setdefault(m.group(1), set()).add(tuple(shlex.split(m.group(2))))
    return out


def _guarded_names(module_path):
    """Every committed name this module's `refuse_degraded_out` calls defend, via AST.

    Read statically rather than by importing, because the point is to enumerate without
    running anything, and `committed` is a name or a tuple of spellings of one name.
    """
    names = set()
    for node in ast.walk(ast.parse(open(module_path).read())):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)) \
                != "refuse_degraded_out":
            continue
        if len(node.args) < 3:
            continue
        names |= {c.value for c in ast.walk(node.args[2])
                  if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    return names


_GUARDED_MODULES = {
    os.path.basename(p)[:-3]
    for p in glob.glob(os.path.join(PP.ROOT, "studies", "study_*.py"))
    if "refuse_degraded_out" in open(p).read()
}

# The guarded drivers the Makefile invokes that have NO row above.  FROZEN, and it fails in
# BOTH directions on purpose: adding a guarded driver to a recipe without a row grows it,
# and giving one of these a row without deleting it here shrinks it.  §131 successor 1 reads
# as a formatting complaint — "the check lives in a scratch harness" — and the census behind
# it is the real finding: HALF the guarded drivers the Makefile invokes (17 of 34) have
# never had their recipe checked by anything.  `study_reds_ratio_stability` is on this list
# and stays on it: its guard is keyed on collected rows, so no argv can decide its case and
# the tests above cover it instead.
_NO_TABLE_ROW = {
    "study_fillet_block", "study_fillet_condition_a", "study_fillet_cost",
    "study_fillet_fold", "study_fillet_kt", "study_fillet_optimum",
    "study_fillet_pnorm", "study_fillet_pnorm_box", "study_fillet_terms",
    "study_fillet_wiring", "study_mbse_baseline", "study_mbse_calibration",
    "study_mbse_score", "study_reds_ratio_stability", "study_tri_bend",
    "study_tri_block", "study_tri_rule",
}


def test_the_enumeration_can_actually_see_the_makefile():
    """The anti-vacuity pin, and it exists because this enumeration silently failed once.

    A scan whose regex stops matching finds nothing, and every assertion below then passes
    over an empty set — the same defect class §132 fixed in `study_kinematics_rank`'s R2:
    a check that cannot fail at the size it is evaluated at.  These floors are the measured
    counts less a little slack, so a recipe may be added or retired without touching them
    but a BROKEN scan cannot pass.
    """
    invocations = _make_n_invocations()
    assert len(invocations) >= 30, (
        f"only {len(invocations)} guarded modules found in `make -n` output — the scan is "
        f"broken, not the Makefile (34 when this was written)")
    assert sum(len(v) for v in invocations.values()) >= 45
    # The continuation join, pinned by the invocation it was added for: `make contact`
    # spans four physical lines and carries five flags.
    assert any(len(argv) >= 8 for argv in invocations["study_contact"]), \
        "multi-line recipes are being truncated at the backslash again"


def test_every_recipe_argv_appears_in_the_table():
    """What `make` expands must be what the table above claims it expands to.

    The table's argvs are hand-copied, and its own three tests prove those argvs reach the
    work rather than the guard.  That proves nothing about the command people type unless
    something checks the two are the same string — which is §131 successor 1.

    ONLY INVOCATIONS AIMED AT A GUARDED NAME ARE COMPARED.  `make contact`'s second
    invocation redirects to `study_contact_step2.json`, which no guard defends, and the
    guard's whole design is that a redirected run is allowed — the table does not need to
    know about it and `test_an_explicit_out_lets_a_degraded_run_through` is where that
    property is pinned.
    """
    invocations = _make_n_invocations()
    for name, names, recipes, _degraded in DRIVERS:
        if name not in invocations:
            continue
        guarded = _guarded_names(os.path.join(PP.ROOT, "studies", f"{name}.py"))
        aimed = {argv for argv in invocations[name]
                 if "--out" not in argv or argv[argv.index("--out") + 1] in guarded}
        assert aimed == {tuple(r) for r in recipes}, (
            f"{name}: `make -n` expands to {sorted(aimed)} but the table claims "
            f"{sorted(tuple(r) for r in recipes)} — one of the two is stale, and if it is "
            f"the table then this driver's recipe is unchecked")


def test_the_set_of_unchecked_recipes_has_not_grown():
    """Half the guarded drivers the Makefile invokes have no row above.  Pinned, both ways.

    Not a red: it is a census, frozen so it can only be changed deliberately.  A new
    guarded driver wired into a recipe with no table row fails this, which is the §129.5
    shape — a maintenance gap that nothing announces.
    """
    unchecked = set(_make_n_invocations()) - {d[0] for d in DRIVERS}
    assert unchecked == _NO_TABLE_ROW, (
        f"unexpectedly missing a row: {sorted(unchecked - _NO_TABLE_ROW)}; "
        f"newly covered, delete from _NO_TABLE_ROW: {sorted(_NO_TABLE_ROW - unchecked)}")
