"""Every study driver refuses to file a degraded run under its committed name.

PLAN.md §43.  §41 found that `study_contact --quick` wrote smoke-mesh data into the
committed `study_contact.json`, where G1 reads 4.394e-04 with both halves passing against
1.7198e-03 and a red `regime_pass` at the real config — a FALSE GREEN standing in for a
RED one.  `study_contact` was the only driver with any guard; the other eight had none.

TWO DIRECTIONS, AND THE FIRST ONE IS THE DANGEROUS TEST TO GET WRONG.  A guard that fires
on the recipe's own invocation would take `make studies` down — five hours, nine drivers —
and it would do it at the END of each driver's run, since `make` only sees the exit
status.  So the recipe's exact argv, as written in the `studies:` target, is asserted to
pass for all nine.

The guard is reached and then execution is stopped, so these cost milliseconds rather than
the recipe's five hours: `refuse_degraded_out` is wrapped to run for real and then raise.
That means the REAL conditions are evaluated with the REAL parsed arguments — a test that
stubbed the guard out would assert nothing.
"""

import sys

import pytest

import _gate_guard


class _GuardPassed(Exception):
    """The real guard was called and did not refuse."""


# (module, the `studies:` target's argv, degraded argvs that must be refused)
#
# The recipe argv is copied from the Makefile's `studies:` block.  If that target changes,
# this table has to change with it, and the first assertion below is what says so.
DRIVERS = [
    ("study_mesh_quality",   ["--samples", "2000"],
     [["--samples", "500"], ["--config", "smoke"], ["--no-plot"]]),
    ("study_wheel_mesh",     ["--samples", "200"],
     [["--quick"], ["--samples", "50"], ["--config", "smoke"], ["--no-plot"]]),
    ("study_beam_agreement", [],
     [["--quick"], ["--genome", "stage3_knee_best_medium.json"], ["--no-plot"]]),
    ("study_wheel_fea",      [],
     [["--quick"], ["--config", "smoke"], ["--no-plot"]]),
    ("study_gnl",            [],
     [["--quick"], ["--config", "smoke"], ["--no-plot"]]),
    ("study_contact",        [],
     [["--quick"], ["--config", "medium"], ["--kinematics", "svk"],
      ["--sections", "penalty"], ["--no-plot"]]),
    ("study_gradient",       [],
     [["--quick"], ["--config", "smoke"], ["--kinematics", "svk"], ["--no-plot"]]),
    ("study_objective",      [],
     [["--quick"], ["--config", "smoke"], ["--elites", "elite10.log"], ["--no-plot"]]),
    ("study_stage3",         [],
     [["--quick"], ["--config", "smoke"], ["--sections", "direction"],
      ["--ladder-p", "1,2,3"], ["--no-plot"]]),
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
    """
    real = _gate_guard.refuse_degraded_out
    calls = []

    def wrapper(ap, args, committed, degraded):
        calls.append(committed)
        real(ap, args, committed, degraded)      # SystemExit if this run is degraded
        raise _GuardPassed                       # allowed — stop before any solving

    monkeypatch.setattr(_gate_guard, "refuse_degraded_out", wrapper)
    for name, *_ in DRIVERS:
        monkeypatch.setattr(__import__(name), "HERE", str(tmp_path))
    return calls




def _main(name):
    return __import__(name).main


@pytest.mark.parametrize("name, recipe, _degraded", DRIVERS, ids=_IDS)
def test_the_recipe_invocation_is_not_refused(monkeypatch, guard_stops_here,
                                              name, recipe, _degraded):
    """`make studies`' own argv must reach the work for all nine drivers.

    This is the assertion protecting the recipe from its own guard.  It fails loudly if a
    condition is written against the wrong default — e.g. guarding `--samples != 2000`
    when the target passes exactly 2000, or comparing `--config` to a literal the driver
    does not actually default to.
    """
    monkeypatch.setattr(sys, "argv", [f"{name}.py", *recipe])
    with pytest.raises(_GuardPassed):
        _main(name)()
    assert guard_stops_here == [f"{name}.json"], (
        f"{name} never called the guard — a driver that skips it can overwrite its own "
        f"committed artifact, which is the whole defect §43 closes")


@pytest.mark.parametrize("name, _recipe, degraded", DRIVERS, ids=_IDS)
def test_degraded_runs_are_refused_by_name(monkeypatch, guard_stops_here,
                                           name, _recipe, degraded):
    """Each degraded invocation must exit nonzero rather than overwrite the artifact."""
    for argv in degraded:
        monkeypatch.setattr(sys, "argv", [f"{name}.py", *argv])
        with pytest.raises(SystemExit) as excinfo:
            _main(name)()
        assert excinfo.value.code != 0, f"{name} {argv} was accepted as the gate"


@pytest.mark.parametrize("name, _recipe, degraded", DRIVERS, ids=_IDS)
def test_an_explicit_out_lets_a_degraded_run_through(monkeypatch, guard_stops_here,
                                                     name, _recipe, degraded):
    """The refusal is about the NAME.  Redirected, every degraded run is allowed.

    Pinned because the cheap way to quieten a noisy guard is to widen it until the
    degraded run cannot happen at all — which would take `make m8bi5`, `make m8bi6`,
    `make m8bii1` and `make contact` with it, all four of which are partial or redirected
    runs that pass their own `--out` for exactly this reason.
    """
    for argv in degraded:
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
