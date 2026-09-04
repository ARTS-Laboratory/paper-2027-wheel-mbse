"""The A/B in `studies/study_fillet_optimum.py` must compare two DIFFERENT meshes.

PLAN.md §105.  This file exists because of a failure mode no green suite caught: §103 made
`wheel_objective.phase_meshes` pass `fillet=True`, and the control arm -- which was the
plain `wheel_stage3.Evaluator`, relying on `build_wheel`'s `fillet=None` DEFAULT -- began
building the same mesh as the treatment.  The driver still ran, still wrote its artifact,
and the artifact was two identical arms labelled `control` and `treatment`.

THE ASSERT IS ON THE MESH REQUEST, NOT ON THE CLASS NAMES OR THE `_FILLET` ATTRIBUTE.  A
test that pinned either of those would go green again the day someone rewires `arm`'s
selector back to a class whose default has moved -- which is exactly the bug.  What is
pinned here is the property the study depends on: the two arms ask `build_wheel` for
different geometry.
"""
import pytest

import wheel_genome as wg
import wheel_wheel as WW
import wheel_fea as W

study_fillet_optimum = pytest.importorskip("study_fillet_optimum")

# BOUND AT IMPORT, AND THE SPY CALLS THIS RATHER THAN `WW.build_wheel`.  The helper below
# runs TWICE in one test, and `monkeypatch.setattr` does not undo between calls -- so a spy
# that delegated to whatever `WW.build_wheel` currently is would wrap the PREVIOUS spy and
# append the second arm's calls to the first arm's list.  That is not hypothetical: it read
# `[True, None]` for the treatment and made the arms look identical when they were not.
_REAL_BUILD_WHEEL = WW.build_wheel


def _fillet_kwargs_for(monkeypatch, *, filleted):
    """Every `fillet=` the arm's evaluator hands `build_wheel`, without solving anything.

    `arm` is not called: it runs a descent.  The evaluator it would build is constructed
    the same way and invoked at `tiers=("t2",)`, which reaches the mesh build and stops
    short of the adjoint -- the mesh request is what this file is about, and a solve here
    would make the test cost minutes for a fact settled in milliseconds.
    """
    seen = []

    def spy(genes, cfg, **kw):
        seen.append(kw.get("fillet", "ABSENT"))
        return _REAL_BUILD_WHEEL(genes, cfg, **kw)

    monkeypatch.setattr(WW, "build_wheel", spy)

    low, high, _ = wg.bounds_arrays(W.GENE_SPACE)
    genes = study_fillet_optimum.load_genes("best_solution.json")
    z0 = wg.normalize(genes, low, high)
    wcfg = WW.get_config("smoke")
    orientation = tuple(float(o) for o in WW.flank_orientation(genes, wcfg, span_mm=W.S))

    cls = (study_fillet_optimum._FilletedEvaluator if filleted
           else study_fillet_optimum._UnfilletedEvaluator)
    ev = cls("smoke", orientation=orientation)
    ev(z0, low, high, phases=(0.0,), tiers=("t2",))
    return seen


def test_the_two_arms_do_not_request_the_same_mesh(monkeypatch):
    treatment = _fillet_kwargs_for(monkeypatch, filleted=True)
    control = _fillet_kwargs_for(monkeypatch, filleted=False)

    assert treatment, "the treatment arm built no mesh -- the spy saw nothing"
    assert control, "the control arm built no mesh -- the spy saw nothing"
    assert all(f is True for f in treatment), (
        f"the treatment arm must request the filleted mesh, got {treatment}")
    assert all(f is None for f in control), (
        f"the control arm must request the UNFILLETED mesh explicitly, got {control} -- "
        f"if this reads True the A/B has collapsed onto one mesh again (PLAN.md §105); if "
        f"it reads 'ABSENT' the arm is relying on build_wheel's default, which is what let "
        f"§103 collapse it silently the first time")
    assert set(treatment) != set(control), (
        "both arms requested the same geometry -- the study is comparing a mesh with "
        "itself and its artifact's two arms are the same number twice")


def test_the_control_arm_is_not_the_bare_stage3_evaluator():
    """The specific regression: `arm`'s selector must not fall back to a moving default.

    Narrower than the test above and kept separate on purpose -- this one names the exact
    line that broke (`cls = _FilletedEvaluator if filleted else WS.Evaluator`), so a
    failure here says WHERE, while a failure above says WHAT.
    """
    import wheel_stage3 as WS
    ctrl = study_fillet_optimum._UnfilletedEvaluator
    assert ctrl is not WS.Evaluator, (
        "the control arm is the bare wheel_stage3.Evaluator again, whose mesh build has "
        "followed wheel_objective.phase_meshes into fillet=True since §103")
    assert issubclass(ctrl, WS.Evaluator), (
        "the control must still BE a wheel_stage3.Evaluator -- the two arms may differ in "
        "the fillet keyword and in nothing else")
