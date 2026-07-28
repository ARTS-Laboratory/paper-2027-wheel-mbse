"""The fillet contract, checked through the artifact rather than through the exporter.

`_junction_edges`, `fillet_junctions` and `kt_report` live in the CadQuery env and had no
test coverage at all — which is how a `NaN` came to sit in the manifest reporting the
part's worst stress riser while the mismatch gate that exists to catch exactly that
filtered it out.

Most of this reads `wheel_step_manifest.json`, which is committed and parses in either
env.  That is deliberate: the manifest IS the contract between the two interpreters, and
a test that needs cadquery cannot run in `make test`.  The one check that genuinely needs
the solid is guarded on `.venv-cad` existing.
"""

import json
import os
import subprocess

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAD_PY = os.path.join(HERE, ".venv-cad", "bin", "python")

# One corner per spoke at the hub; two per spoke at the rim.  Pinned as exact counts
# because a selector that silently drifts onto the wrong family of edges — the flank
# kinks at r=13.94 and r=47.51 are also twelve-fold — would still produce a plausible
# looking manifest.
EXPECTED_EDGES = {"hub": 12, "rim": 24}


def _strict(token):
    raise ValueError(f"manifest contains the non-JSON constant {token!r}")


@pytest.fixture(scope="module")
def manifest():
    with open(os.path.join(HERE, "wheel_step_manifest.json")) as fh:
        return json.load(fh, parse_constant=_strict)


def test_manifest_is_strict_json(manifest):
    """No bare NaN / Infinity tokens.

    `json.load` accepts them as a Python extension; `jq`, `JSON.parse` and every strict
    parser do not.  The manifest used to carry two, from `kt_report`'s square-junction
    path — so this is a regression test on that specific fix, and the `parse_constant`
    hook in the fixture is what enforces it.
    """
    assert manifest["fillets"]["detail"]


def test_every_junction_reports_a_number_not_a_nan(manifest):
    """THE regression: a square junction must be priced, not excused.

    When nothing could be built, `kt_report` routes through the branch
    `wheel_fea.stress_concentration_kt` already has for a degenerate fillet
    (`< 0.1 mm -> 3.5`).  That makes `kt_error_pct` a real, comparable number instead of
    a NaN that the mismatch gate silently drops.
    """
    import math
    for row in manifest["fillets"]["detail"]:
        for key in ("kt_modeled", "kt_built", "kt_error_pct", "worst_wedge_deg"):
            assert key in row, f"{row['junction']}: missing {key}"
            assert math.isfinite(row[key]), f"{row['junction']}.{key} = {row[key]}"
        assert row["kt_built"] >= 1.0


def test_a_junction_built_sharper_than_asked_is_priced_worse(manifest):
    """Direction check, and it has been wrong in the reporting before.

    A fillet smaller than requested is a SHARPER corner, so its Kt must come out HIGHER
    than the modeled one — never lower.  A sign slip here would let an under-built fillet
    read as a safety margin.
    """
    for row in manifest["fillets"]["detail"]:
        if row["r_built_mm"] < row["r_requested_mm"] - 1e-9:
            assert row["kt_built"] > row["kt_modeled"], row
            assert row["kt_error_pct"] > 0.0, row
        else:
            assert abs(row["kt_error_pct"]) < 1e-9, row


def test_found_and_filleted_counts_are_distinguishable(manifest):
    """`hub_edges: 0` used to mean both "found none" and "built none", indistinguishably.

    Those want opposite fixes — a selector bug versus a geometry that cannot accept a
    fillet — so the manifest has to say which.
    """
    for row in manifest["fillets"]["detail"]:
        n_found = row["n_edges_found"]
        n_filleted = row["n_edges_filleted"]
        assert n_found == EXPECTED_EDGES[row["junction"]], (
            f"{row['junction']}: found {n_found} re-entrant corners, expected "
            f"{EXPECTED_EDGES[row['junction']]} — the edge selector has moved")
        assert 0 <= n_filleted <= n_found
        assert (n_filleted > 0) == (row["r_built_mm"] > 0.0), row


def test_the_selected_corners_are_re_entrant(manifest):
    """The selector's whole premise: only a corner with material on both sides of the
    gap can take a fillet.  Below 180 degrees it is convex and there is nothing to round.
    """
    for row in manifest["fillets"]["detail"]:
        assert row["worst_wedge_deg"] > 180.0, row


def test_the_hub_junction_is_still_the_known_open_problem(manifest):
    """Pin the discrepancy so it cannot regress quietly, in EITHER direction.

    Measured: `_embed` laps adjacent spokes over the hub circle, leaving a 354 degree
    spoke-to-spoke notch that OCC refuses at every radius down to
    `MIN_CURVATURE_RADIUS_MM`.  So the hub ships square and `kt_error_pct` is about +88%.

    If this test fails because the hub now takes a fillet, that is GOOD NEWS and the fix
    is to update the expectation — but it must be noticed, because the as-built peak
    stress it implies (25.1 -> ~47.2 MPa against a 25 MPa allowable) is the single
    largest known error in the shipped part.
    """
    hub = next(r for r in manifest["fillets"]["detail"] if r["junction"] == "hub")
    assert hub["r_built_mm"] == 0.0, (
        f"the hub junction now builds a {hub['r_built_mm']:.3f} mm fillet — the open "
        f"discrepancy in CLAUDE.md is fixed and this test should say so")
    assert hub["worst_wedge_deg"] > 350.0, hub
    assert hub["kt_error_pct"] > 50.0, hub


# ---------------------------------------------------------------------------
# NEEDS THE CAD ENV
# ---------------------------------------------------------------------------

_CAD_CROSS_CHECK = r"""
import json, math, sys
import numpy as np
import wheel_step_export as X
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_State

genes = json.load(open("best_solution.json"))["genes"]
profile, _ = X.build_profile(genes)
part = X.extrude_profile(profile)
rows = X._junction_edges(part)

# Independent measurement: the 2D FACE classifier on the profile, against the 3D SOLID
# classifier `_junction_edges` uses.  Different OCC algorithm on a different shape, so an
# inverted inside/outside sense in one of them cannot cancel.
face = profile.wrapped
def wedge2d(cx, cy, rad=X.FILLET_WEDGE_PROBE_MM, n=X.FILLET_WEDGE_SAMPLES):
    hit = 0
    for i in range(n):
        t = 2.0 * math.pi * i / n
        cl = BRepClass_FaceClassifier(
            face, gp_Pnt(cx + rad * math.cos(t), cy + rad * math.sin(t), 0.0), 1e-9)
        if cl.State() != TopAbs_State.TopAbs_OUT:
            hit += 1
    return 360.0 * hit / n

out = []
for e, r, w in rows:
    c = e.Center()
    out.append({"r": r, "wedge_solid": w, "wedge_face": wedge2d(c.x, c.y)})
print("RESULT:" + json.dumps(out))
"""


@pytest.mark.skipif(not os.path.exists(CAD_PY), reason="no .venv-cad on this machine")
def test_the_wedge_classifier_agrees_with_an_independent_probe():
    """Two OCC classifiers, two shapes, one answer per corner.

    This is the check that the re-entrancy predicate is actually right.  A silently
    inverted sense would select the twelve CONVEX flank kinks instead of the twelve
    re-entrant notches, and every downstream count and radius would still look sane —
    the manifest tests above cannot tell the difference.
    """
    proc = subprocess.run([CAD_PY, "-c", _CAD_CROSS_CHECK], cwd=HERE,
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:"))
    rows = json.loads(line[len("RESULT:"):])

    assert len(rows) == sum(EXPECTED_EDGES.values()), len(rows)
    for row in rows:
        # One probe-sample of slack: the two classifiers can disagree on a boundary
        # sample, and the sample spacing is 360/FILLET_WEDGE_SAMPLES degrees.
        assert abs(row["wedge_solid"] - row["wedge_face"]) <= 360.0 / 180 + 1e-9, row
        assert row["wedge_solid"] > 180.0, row

    # And the corners must be where the geometry says they are, not on the flank kinks.
    hub = sorted(r["r"] for r in rows if r["r"] < 30.0)
    rim = sorted(r["r"] for r in rows if r["r"] >= 30.0)
    assert len(hub) == EXPECTED_EDGES["hub"] and len(rim) == EXPECTED_EDGES["rim"]
    # The hub corner is NOT on the hub circle — that is the whole finding.
    assert all(abs(r - 12.8748) < 0.01 for r in hub), hub
    assert all(r > 12.7 for r in hub), (
        "a hub corner landed inside the hub circle; _embed's overshoot has changed")
    assert all(abs(r - 48.5) < 0.01 for r in rim), rim
