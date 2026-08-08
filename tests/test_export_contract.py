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

# Two corners per spoke at each ring, one per flank.  Pinned as exact counts because a
# selector that silently drifts onto the wrong family of edges — the flank kinks at
# r=13.94 and r=47.51 are also twelve-fold — would still produce a plausible looking
# manifest.
#
# The hub was 12 until the hub fillet milestone, and that 12 was the bug: `_embed` ran the
# root caps 22.3 deg sideways, adjacent spokes lapped over the hub circle before either
# reached it, and the twelve corners being counted were spoke-to-spoke knife edges standing
# 0.175 mm OUTSIDE the circle rather than the twenty-four spoke-to-hub corners on it.  See
# HUB_PLAN.md.
EXPECTED_EDGES = {"hub": 24, "rim": 24}


def _strict(token):
    raise ValueError(f"manifest contains the non-JSON constant {token!r}")


@pytest.fixture(scope="module")
def manifest():
    with open(os.path.join(HERE, "export", "wheel_step_manifest.json")) as fh:
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
        # A junction is priced by its WORST corner, so a positive built radius means every
        # corner got one.  This used to read `n_filleted > 0`, which was the same statement
        # only while a junction could not be partly built — it can be, and the rim was.
        assert (row["r_built_mm"] > 0.0) == (n_filleted == n_found), row


def test_the_junction_is_priced_at_its_worst_corner(manifest):
    """`fillet_families` is the per-family record; `r_built_mm` is the worst of them.

    The rule this pins is why the rim's `kt_error_pct` is not +0.0% any more.  It used to
    report the radius that WAS applied, so twelve square corners out of twenty-four read as
    a part matching its model exactly.  A stress riser is not an average.
    """
    for row in manifest["fillets"]["detail"]:
        families = row["fillet_families"]
        assert sum(f["n_edges"] for f in families) == row["n_edges_filleted"], row
        assert all(f["radius_mm"] > 0.0 for f in families), row
        # Descending: the ladder walks down, and each pass takes the largest radius its
        # remaining corners accept.
        radii = [f["radius_mm"] for f in families]
        assert radii == sorted(radii, reverse=True), row
        assert all(r <= row["r_requested_mm"] + 1e-12 for r in radii), (
            f"{row['junction']}: a family was built LARGER than requested, which prices "
            f"a fillet nobody asked for as margin: {radii} vs {row['r_requested_mm']}")
        if row["n_edges_filleted"] == row["n_edges_found"] and families:
            assert row["r_built_mm"] == min(radii), row


def test_the_built_fillet_radii_are_exact_ladder_rungs(manifest):
    """`wheel_objective.FILLET_LADDER_DECAY` is a copy, and this is what keeps it honest.

    The optimizer side needs the ladder's step size — it sets the blend width of the
    `smooth_min` that caps `R_hub`, on the argument that two radii closer than one rung
    cannot produce different built parts.  It cannot IMPORT the number: `wheel_step_export`
    needs cadquery and `tests/test_import_hygiene.py` pins that split.  So the two are tied
    together through the artifact instead — every radius the exporter actually built is
    `r_requested * 0.85**n` for a whole `n`, exactly.

    Measured on the shipped manifest: the hub's two families are rungs 2 and 9, agreeing to
    the last bit.  If `_fillet_ladder` ever changes its decay or starts rounding its rungs
    (it did once — `round(r, 3)` reported a part built BLUNTER than requested), this fails
    here rather than silently mis-sizing the cap's blend.
    """
    import math

    import wheel_objective as WO

    decay = WO.FILLET_LADDER_DECAY
    for row in manifest["fillets"]["detail"]:
        req = row["r_requested_mm"]
        for fam in row["fillet_families"]:
            n = math.log(fam["radius_mm"] / req) / math.log(decay)
            assert abs(n - round(n)) < 1e-9, (
                f"{row['junction']}: built radius {fam['radius_mm']!r} is not "
                f"{req!r} * {decay}**n — it is n = {n:.6f}.  Either the exporter's ladder "
                f"changed or it started rounding, and wheel_objective.FILLET_LADDER_DECAY "
                f"is now wrong about what the exporter can build")


def test_the_selected_corners_are_re_entrant(manifest):
    """The selector's whole premise: only a corner with material on both sides of the
    gap can take a fillet.  Below 180 degrees it is convex and there is nothing to round.
    """
    for row in manifest["fillets"]["detail"]:
        assert row["worst_wedge_deg"] > 180.0, row


def test_the_exporter_and_the_constraint_price_the_same_kt(manifest):
    """One formula, two interpreters, and now also two implementations.

    The manifest's `kt_modeled` comes from `wheel_fea.stress_concentration_kt` in the CAD
    env; the stress constraint prices `Kt` with `wheel_objective`'s jnp twin, because the
    numpy original cannot be traced.  Since M8b-i.6 step 2 that factor IS the constraint —
    `Kt(R, t) * sigma_nominal <= ALLOWABLE` — so a drift between the two would mean the
    optimizer is descending on a concentration the exporter does not think the part has.

    Checkable at all only because both sides are now one closed-form expression of the
    genes.  The genome is pinned by hash, so this cannot pass by comparing the wrong
    design to itself.
    """
    import numpy as np
    import wheel_genome as wg
    import wheel_objective as WO

    with open(os.path.join(HERE, "best_solution.json")) as fh:
        genes = json.load(fh)["genes"]
    assert wg.genome_hash(genes).startswith(manifest["genome_hash"]), (
        f"the manifest was exported from genome {manifest['genome_hash']} but "
        f"best_solution.json now hashes to {wg.genome_hash(genes)}; re-export before "
        f"reading anything else in this file")

    for row, (r_gene, t_gene) in zip(manifest["fillets"]["detail"],
                                     (("R_hub", "t0"), ("R_rim", "t3"))):
        assert row["r_requested_mm"] == pytest.approx(genes[r_gene], rel=1e-12), (
            f"{row['junction']}: the exporter requested {row['r_requested_mm']} but the "
            f"gene says {genes[r_gene]}")
        for label, radius, expect in (("modeled", row["r_requested_mm"], row["kt_modeled"]),
                                      ("built", row["r_built_mm"], row["kt_built"])):
            twin = float(WO.stress_concentration_kt(radius, genes[t_gene]))
            assert twin == pytest.approx(expect, rel=1e-12), (
                f"{row['junction']} kt_{label}: the exporter says {expect} and the "
                f"constraint's jnp twin says {twin} — the two implementations of "
                f"Kt = 1 + C*(t/2R)^0.65 have drifted")
        assert np.isfinite(row["kt_error_pct"])


def test_the_hub_junction_exists_and_every_corner_of_it_is_filleted(manifest):
    """The hub fillet milestone, pinned so it cannot regress quietly.

    WHAT THIS USED TO SAY.  `r_built_mm == 0.0`, `worst_wedge_deg > 350`,
    `kt_error_pct > 50` — the hub shipped square at Kt 3.5 against a modelled 1.861, so
    as-built hub utilisation was 1.88x whatever Stage 3 reported.  M8b-i.6 step 2 had
    raised the stakes by giving `R_hub` a live gradient: the optimizer paid for a fillet
    in utilisation and did not get one.

    WHAT CLOSED IT.  `_embed`'s inward step took the least rotation from the junction
    tangent, a 4.516 mm run that swung the root cap 22.3 deg out of a 30 deg sector, so
    adjacent spokes lapped over the hub circle before either reached it and the circle
    stopped existing.  It now plunges radially — 1.788 mm, 0.57 deg — and the twenty-four
    spoke-to-hub corners are back on r = 12.7.  `fillet_junctions` then fillets the
    leftover family instead of returning after the first one.

    WHAT USED TO BE OPEN, and is not any more.  On the old shipped genome the shallow corner
    of a near-tangent arrival took only 0.361 mm against a requested 1.560, so the hub split
    into two families and `kt_error_pct` sat at +73.4%.  The genome promoted in PLAN.md §13
    asks for `R_hub` = 0.579, under §5's cap of 0.624 for that genome, and OCC builds it on
    all 24 corners in ONE family: `kt_error_pct` +0.0%, the first shipped part whose fillets
    are the ones its stress model priced.  The worst-corner rule still applies and is still
    asserted — there is simply only one corner radius to be worst now.

    THE OTHER HALF OF THIS NOTE USED TO BE WRONG.  It said the inter-spoke gap "caps ANY hub
    fillet near 1.1 mm", on the strength of half the 2.196 mm void agreeing with the
    1.127 mm OCC built.  `make hubcap` bisected what OCC will actually accept on a single hub
    corner and got 1.300 mm: the 1.127 was a LADDER RUNG taken by a whole twelve-edge family,
    not a limit, and the agreement was a coincidence.  The real limit tracks `t0/2`, not the
    slot — three designs whose voids span a 54% range give thresholds 3.4% apart.  PLAN.md §5
    is the record; `wheel_objective.hub_fillet_cap_mm` is the constraint that now knows it.

    Capping `R_hub` does not move `kt_error_pct` DIRECTLY — the cap does not model the
    shallow corner that used to set it.  But a genome that lands under the cap never asks
    OCC for a radius the shallow corner has to refuse, and that is how §13's genome reaches
    +0.0%.  The `< 88.0` bound below is therefore still the real gate: it is what catches a
    regression back toward a square hub, and it must not be tightened onto the current
    genome's 0.0, which is a property of THIS design and not of the exporter.
    """
    hub = next(r for r in manifest["fillets"]["detail"] if r["junction"] == "hub")
    assert hub["r_built_mm"] > 0.0, (
        f"the hub junction is square again — every corner must take some radius.  "
        f"{hub}")
    assert hub["n_edges_filleted"] == hub["n_edges_found"] == EXPECTED_EDGES["hub"], hub
    # 332 deg measured, against the 354 deg spoke-to-spoke knife edge that preceded it.
    # Anything back above 350 means the spokes are lapping again.
    assert hub["worst_wedge_deg"] < 350.0, (
        f"{hub}\na wedge this close to a cusp is the spoke-to-spoke notch, not a "
        f"spoke-to-hub corner — `_embed` is running sideways again")
    # THIS USED TO ASSERT `== 2`, and that was pinning a PATHOLOGY as an invariant.  Two
    # families means OCC refused the requested radius on one flank and had to fall back;
    # the promoted genome (§13) keeps `R_hub` under the cap and gets ONE family covering
    # all 24 edges at the full requested radius, which is the outcome the whole hub-fillet
    # milestone was aiming at — and the old assertion called it a failure.
    #
    # What this actually needs to guarantee is that no family is silently abandoned.  The
    # edge count above already covers that; this makes the families account for it too, so
    # a fallback split is still fine and a DROPPED family is not.
    assert hub["fillet_families"], f"{hub}\nno fillet families recorded at all"
    assert sum(f["n_edges"] for f in hub["fillet_families"]) == hub["n_edges_filleted"], (
        f"{hub}\nthe families do not account for every filleted edge — `fillet_junctions` "
        f"abandoned one, which is the bug that left the hub square")
    assert hub["r_built_mm"] == pytest.approx(
        min(f["radius_mm"] for f in hub["fillet_families"])), (
        f"{hub}\n`r_built_mm` must be the WORST corner's radius — the junction is priced "
        f"at its weakest fillet, not its best")
    assert hub["kt_error_pct"] < 88.0, (
        f"{hub}\nas-built hub utilisation is "
        f"{hub['kt_built'] / hub['kt_modeled']:.2f}x what the constraint reports; it was "
        f"1.88x when the hub shipped square, and it must not go back")


# ---------------------------------------------------------------------------
# THE WEAK-JUNCTION CHECK
#
# It had no coverage at all until now, which is how it came to spend the whole
# MIN_WALL_MM sweep reporting a proxy for t0 as if it were a verdict on the weld.  See
# `wheel_geometry.junction_bite`.
# ---------------------------------------------------------------------------

def test_the_junction_check_reports_a_bite_not_just_a_volume(manifest):
    """The manifest has to carry the normalised number, or nothing can check it.

    A raw mm³ is not falsifiable across designs: it is quadratic in the root thickness,
    so the same junction reads 18.12 mm³ at t0=1.20 and 78.53 mm³ at t0=2.48.  The bite
    divides that out, and `t_mm` is recorded next to it so the division is auditable
    from the artifact alone.
    """
    block = manifest["junction_overlap_mm3"]
    assert set(block) >= {"hub", "rim", "bite", "t_mm", "pass", "bite_floor"}, (
        f"junction block is missing the normalised fields — re-export.  Got {block}")
    for ring in ("hub", "rim"):
        for value in (block[ring], block["bite"][ring], block["t_mm"][ring]):
            assert isinstance(value, (int, float)) and value > 0.0, block
        assert isinstance(block["pass"][ring], bool), block


def test_the_bite_is_the_volume_divided_by_the_right_thickness(manifest):
    """THE cross-interpreter check: recompute the bite here, from the genes.

    The exporter prices the hub on t0 and the rim on t3 — the same pairing the stress
    constraint uses, because `thickness_at_arc_length` is exactly t0 at s=0 and t3 at
    s=1.  Swapping them is a one-character mistake that the raw volumes cannot reveal,
    and on the shipped genome (t0=2.48, t3=2.00) it moves the hub bite by 54%.  This
    test runs in the jax env against a manifest written by the CAD env, so it is also
    what keeps `MIN_JUNCTION_BITE` from being defined twice.
    """
    import wheel_genome as wg
    from wheel_fea import SPOKE_WIDTH_MM
    from wheel_geometry import junction_bite

    with open(os.path.join(HERE, "best_solution.json")) as fh:
        genes = json.load(fh)["genes"]
    assert wg.genome_hash(genes).startswith(manifest["genome_hash"]), (
        f"the manifest was exported from genome {manifest['genome_hash']} but "
        f"best_solution.json now hashes to {wg.genome_hash(genes)}; re-export")

    block = manifest["junction_overlap_mm3"]
    for ring, t_key in (("hub", "t0"), ("rim", "t3")):
        # abs=1e-4 because the manifest rounds to 4 dp.  Loose enough for the rounding,
        # tight enough for the failure this exists for: the shipped t0 and t3 are 0.48 mm
        # apart, and a swap is 4800x this tolerance.
        assert block["t_mm"][ring] == pytest.approx(genes[t_key], abs=1e-4), (
            f"{ring} was priced at t={block['t_mm'][ring]} but the gene {t_key} says "
            f"{genes[t_key]} — the exporter has the two rings' thicknesses crossed")
        expect = junction_bite(block[ring], genes[t_key], SPOKE_WIDTH_MM)
        assert block["bite"][ring] == pytest.approx(expect, abs=1e-4), (
            f"{ring}: manifest bite {block['bite'][ring]} but "
            f"{block[ring]} mm³ / (t² · W) = {expect}")


def test_the_shipped_junctions_clear_their_own_floor(manifest):
    """And `pass` is the floor comparison, not an independently written opinion.

    The margin assertion is deliberately loose.  0.25 is a geometric floor — half of
    what every genome measured so far achieves — not a limit fitted to a failure, since
    this repo has never produced a junction that failed one.  So this pins that the
    shipped part is not near it, and leaves the exact value free to move when a real
    negative example turns up.
    """
    from wheel_geometry import MIN_JUNCTION_BITE

    block = manifest["junction_overlap_mm3"]
    assert block["bite_floor"] == MIN_JUNCTION_BITE, (
        f"the manifest was written against a floor of {block['bite_floor']} and the "
        f"code now says {MIN_JUNCTION_BITE} — re-export")
    for ring in ("hub", "rim"):
        assert block["pass"][ring] is (block["bite"][ring] >= MIN_JUNCTION_BITE), (
            f"{ring}: pass={block['pass'][ring]} disagrees with "
            f"{block['bite'][ring]} >= {MIN_JUNCTION_BITE}")
        assert block["bite"][ring] >= MIN_JUNCTION_BITE, (
            f"the SHIPPED wheel's {ring} weld is below the floor at "
            f"{block['bite'][ring]} root thicknesses — this is not a threshold to "
            f"loosen without reading wheel_geometry.junction_bite first")


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
    # cwd is the ROOT, because the snippet opens `best_solution.json` relatively; `src/` is
    # handed over explicitly rather than inherited, so this passes under a bare `pytest`
    # that never loaded conftest.py as well as under `make test`.
    proc = subprocess.run(
        [CAD_PY, "-c", _CAD_CROSS_CHECK], cwd=HERE,
        env={**os.environ, "PYTHONPATH": os.path.join(HERE, "src")},
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
    # The hub corners are ON the hub circle, and that is the whole hub fillet fix.  They
    # used to sit at r = 12.8748, twelve of them rather than twenty-four, because they were
    # spoke-to-spoke notches standing outside a hub circle `_embed` had buried.
    assert all(abs(r - 12.7) < 0.01 for r in hub), (
        f"{hub}\na hub corner is not on the hub circle — `_embed` is running sideways "
        f"again and the spokes are lapping over it")
    assert all(abs(r - 48.5) < 0.01 for r in rim), rim


_MASS_KEY_PROBE = r"""
import json
import wheel_step_export as X
print("RESULT:" + json.dumps([
    X.optimizer_spoke_mass({"total_mass_g": 41.5, "mesh_mass_g": 58.7}),
    X.optimizer_spoke_mass({"mesh_mass_g": 58.7}),
    X.optimizer_spoke_mass({}),
]))
"""


@pytest.mark.skipif(not os.path.exists(CAD_PY), reason="no .venv-cad on this machine")
def test_the_solid_report_finds_a_mass_from_either_optimizer():
    """The mass cross-check must survive promoting a Stage-3 genome.

    `report()` compares the OCC solid's mass against the optimizer's own spoke mass, and
    that is one of the few places the FEA and CadQuery pipelines are checked against each
    other at all.  It read `metrics['total_mass_g']` through a `.get(..., nan)` — a key
    only the GA/beam writer produces.  A Stage-3 descent writes `mesh_mass_g` instead, so
    promoting one exported cleanly and printed `nan g`: the check did not fail, it went
    silent, on exactly the genome that ships.

    The GA key wins when both are present so the historical artifact keeps reporting the
    number it always did, and the source key is returned rather than normalised away
    because the two are the same ROLE measured two different ways (analytic beam area vs
    integration over the FEA mesh) and are not interchangeable to a reader.
    """
    proc = subprocess.run(
        [CAD_PY, "-c", _MASS_KEY_PROBE], cwd=HERE,
        env={**os.environ, "PYTHONPATH": os.path.join(HERE, "src")},
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")][0]
    both, stage3_only, empty = json.loads(line[len("RESULT:"):])

    assert both == [41.5, "total_mass_g"]
    assert stage3_only == [58.7, "mesh_mass_g"]
    # No key at all still must not crash the export — but it must SAY so, not print nan
    # next to a label claiming the optimizer reported it.
    assert empty[0] != empty[0], "expected nan"
    assert "no mass key" in empty[1]
