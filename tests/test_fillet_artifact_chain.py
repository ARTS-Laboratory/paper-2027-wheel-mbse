"""
Pins for the two DERIVED artifacts at the end of the fillet chain — `study_fillet_kt.json`
and `study_fillet_wiring.json`. PLAN.md §119 successor 3.

WHY THIS FILE EXISTS. Neither driver had a test of any kind, and both are pure readers:
they solve nothing and derive everything from artifacts other drivers committed. §119
named the gap — *"`studies/study_fillet_kt.json` AND `study_fillet_wiring.json` READ THE
CORNER ARTIFACTS, and no test covers that dependency"* — and the chain is one hop longer
than that sentence says, because the profile pairs enter it from a third driver:

    study_fillet_block.py   LAYER_PROFILE_CANDIDATES
        -> study_corner_singularity.py  (--profiles imports it as `fbk`)
        -> study_corner_singularity_fillet.json
        -> study_fillet_kt.json  and  study_fillet_wiring.json
    study_junction_agreement.json  -> study_fillet_wiring.json

A derived artifact that solves nothing cannot go stale on its own. It goes stale when an
INPUT is regenerated under it, or — `study_fillet_wiring` only — when the one live read it
does have moves. Nothing was watching either.

WHAT IS PINNED. That each driver still reads the artifacts it is documented to read, and
that each committed artifact still REPRODUCES from the inputs on disk. A stored-number
test would pass forever after an input moved; rebuilding is the whole point, and it is
cheap enough to be a test because neither driver solves a field (0.7 s and 0.4 s measured
2026-09-07).

`test_study_fillet_wiring_reproduces_from_its_committed_inputs` IS RED ON PURPOSE — see
its docstring. It is the tripwire doing its job on the day it was installed.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import study_fillet_kt as kt              # noqa: E402
import study_fillet_wiring as fw          # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIES = os.path.join(REPO, "studies")

# Fields a rebuild is allowed to differ on because they time the run rather than measure
# the wheel.  Neither driver writes one today; the tuple is here so that adding a
# timestamp later does not turn both reproduction tests red for no reason.
VOLATILE = ("wall_s", "when", "generated")


def _committed(name):
    with open(os.path.join(STUDIES, name)) as fh:
        return json.load(fh)


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _first_difference(a, b, path=""):
    """The first path where two records disagree, so a failure names the field."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}/{key} present in only one record"
            found = _first_difference(a[key], b[key], f"{path}/{key}")
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path} length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = _first_difference(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    return None if a == b else f"{path}: committed {a!r}, rebuilt {b!r}"


# ---------------------------------------------------------------------------
# THE DEPENDENCY ITSELF
# ---------------------------------------------------------------------------

def test_both_drivers_still_read_the_corner_artifacts():
    """The edge §119 named, pinned as an edge rather than left to the prose.

    If someone re-points either driver at a probe artifact — `..._probe.json`, a scratch
    `--out` from an exploratory run — the reproduction tests below would still pass
    against the new inputs and say nothing.  This is the check that the chain being
    reproduced is the chain the tree documents.
    """
    assert kt.DEFAULT_INPUTS[("shipped", "corner_unfilleted")] == \
        "study_corner_singularity.json"
    assert kt.DEFAULT_INPUTS[("shipped", "corner_filleted")] == \
        "study_corner_singularity_fillet.json"
    assert fw.DEFAULT_INPUTS["corner_shipped"] == "study_corner_singularity_fillet.json"
    assert fw.DEFAULT_INPUTS["junction"] == "study_junction_agreement.json"


def test_the_kt_census_is_the_corner_artifacts_own_numbers():
    """`corner_census` COPIES four probes out of two artifacts; it does not re-derive them.

    So the committed census is a statement about those two files and nothing else, and it
    is the sharpest form of the dependency: refresh either corner artifact without
    refreshing `study_fillet_kt.json` and these rows describe a mesh that no longer
    exists, with every field still present and every verdict still computed.
    """
    unf = _committed("study_corner_singularity.json")
    fil = _committed("study_corner_singularity_fillet.json")
    census = _committed("study_fillet_kt.json")["genomes"]["shipped"]["corner_census"]
    for probe, row in census.items():
        for tag, src in (("unfilleted", unf), ("filleted", fil)):
            assert row[tag]["wedge_deg"] == src["williams"][probe]["wedge_deg"], probe
            assert row[tag]["kind"] == src["williams"][probe]["kind"], probe
            assert row[tag].get("lambda") == src["williams"][probe].get("lambda"), probe


# ---------------------------------------------------------------------------
# THE COMMITTED ARTIFACTS, REBUILT
# ---------------------------------------------------------------------------

def test_study_fillet_kt_reproduces_from_its_committed_inputs():
    """`study_fillet_kt` reads six artifacts and makes no live read at all.

    That makes it exactly reproducible from the tree, and it reproduces today — which is
    the useful half of the finding: this artifact is CONSISTENT with its inputs, so
    whatever is stale about the corner ladders under it is stale there and not here.
    """
    rebuilt = kt.build(dict(kt.DEFAULT_INPUTS))
    diff = _first_difference(_strip(_committed("study_fillet_kt.json")), _strip(rebuilt))
    assert diff is None, diff


def test_study_fillet_wiring_reproduces_from_its_committed_inputs():
    """REFRESHED 2026-09-07 once both frozen inputs were — see PLAN §119 successor 1's
    corner half and §126 (junction half, peer session).

    This driver makes ONE live read (`build()` loads `best_solution.json` for
    `mesh_fillet_arcs`), so refreshing it folds in both the corner/junction artifact
    updates and that live read at once. Checked before filing, the same way §120 checked
    the three drivers it pinned: every categorical verdict below survived unchanged —
    only three numeric wedge-error fields moved, all downstream of the junction refresh
    (`hub_wedge_err_end_cap_deg` 28.705 -> 6.426, `hub_wedge_err_as_built_deg` 0.0080 ->
    0.0183, `rim_wedge_err_as_built_deg` 50.612 -> 32.343). `test_the_wiring_verdict_
    survives_the_rebuild` is the pin on that half; this test is the bookkeeping half.
    """
    rebuilt = fw.build()
    diff = _first_difference(_strip(_committed("study_fillet_wiring.json")),
                             _strip(rebuilt))
    assert diff is None, diff


def test_the_wiring_verdict_survives_the_rebuild():
    """The half of the artifact above that is NOT stale, pinned separately.

    Split out because a single red on the whole record cannot distinguish "ten numeric
    fields drifted" from "a finding moved", and those call for opposite responses.  While
    this passes, the red above is bookkeeping.  If it ever fails, the refresh has become
    §119 successor 1's problem and stops being a refresh.
    """
    assert fw.build()["verdict"] == _committed("study_fillet_wiring.json")["verdict"]
