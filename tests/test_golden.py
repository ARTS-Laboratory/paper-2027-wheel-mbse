"""Golden regression against the artifacts already on disk.

These lock down the CURRENT behaviour of `wheel_fea.evaluate_design` before any of the
three-stage refactor touches it.  `wheel_fea.py` is 1310 lines with no other tests and a
load-bearing import contract (`wheel_step_export.py:60-69` imports from it across an
interpreter boundary), so this file is the safety net for every later milestone.

The expected values are not hand-transcribed — they are read back out of
`best_solution.json` and `wheel_step_manifest.json`, which the last real run wrote.
"""

import hashlib
import json
import os

import pytest

import wheel_fea as W

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Machine-precision, not a loose tolerance: `evaluate_design` is deterministic and the
# recorded values came out of this exact code path (wheel_fea.py:998).  Anything above
# this is a real behaviour change, not float noise.
TOL = 1e-9


@pytest.fixture(scope="module")
def record():
    with open(os.path.join(HERE, "best_solution.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def scored(record):
    """Score the on-disk genome through the same entry point the GA uses."""
    vec = [record["genes"][name] for name in W.GENE_NAMES]
    return W.evaluate_design(vec)


def test_gene_names_match_json_keys(record):
    """GENE_NAMES (wheel_fea.py:682) is the ordering contract between the flat 14-vector
    the GA works in and the dict the exporter reads.  If these ever diverge, every
    genome on disk is silently reinterpreted."""
    assert sorted(W.GENE_NAMES) == sorted(record["genes"])
    assert len(W.GENE_NAMES) == 14


def test_the_artifact_was_produced_by_these_constants(record):
    """The genome only means something in the frame it was optimised in.

    `HUB_RIM_SPAN_MM` is derived from `RIM_RADIUS_MM`, and the genome's centerline runs
    from (0,0) to (span,0) — so changing the rim merge radius silently reinterprets
    every gene on disk.  Without this check the golden test would keep passing while
    comparing a genome against a DIFFERENT wheel, because it reads its expected values
    out of the same stale artifact.

    If this fails, the fix is to re-run the GA, not to edit the number.
    """
    geom = record["geometry"]
    assert geom["rim_radius_mm"] == pytest.approx(W.RIM_RADIUS_MM), (
        f"best_solution.json was produced with RIM_RADIUS_MM="
        f"{geom['rim_radius_mm']} but the module now says {W.RIM_RADIUS_MM} — the "
        f"artifact is stale and must be regenerated (`make ga`)")
    assert geom["hub_radius_mm"] == pytest.approx(W.HUB_RADIUS_MM)
    assert geom["spoke_width_mm"] == pytest.approx(W.SPOKE_WIDTH_MM)
    assert geom["number_of_spokes"] == W.NUMBER_OF_SPOKES


def test_loss_terms_reproduce(record, scored):
    """Every term the artifact recorded must still evaluate to the same number.

    Driven by the artifact's own keys rather than a hardcoded list, so ADDING a loss
    term does not break the golden test — the test's job is that the recorded numbers
    still reproduce, not that the term set is frozen.  A term that DISAPPEARS still
    fails, which is the direction that matters.
    """
    _, loss_terms = scored
    recorded = record["loss_terms"]
    assert recorded, "the artifact records no loss terms"
    missing = sorted(set(recorded) - set(loss_terms))
    assert not missing, f"evaluate_design no longer produces {missing}"
    for term, expected in recorded.items():
        assert loss_terms[term] == pytest.approx(expected, rel=TOL, abs=TOL), term


@pytest.mark.parametrize("metric", ["deflection_mm", "max_stress_mpa",
                                    "total_mass_g", "buckling_ratio"])
def test_metrics_reproduce(record, scored, metric):
    metrics, _ = scored
    assert metrics[metric] == pytest.approx(record["metrics"][metric],
                                            rel=TOL, abs=TOL)


def test_kt_reproduces(record):
    """Kt_hub/Kt_rim are computed in __main__ (wheel_fea.py:989) rather than inside
    evaluate_design, so they need their own check."""
    g = record["genes"]
    assert W.stress_concentration_kt(g["R_hub"], g["t0"]) == pytest.approx(
        record["metrics"]["Kt_hub"], rel=TOL)
    assert W.stress_concentration_kt(g["R_rim"], g["t3"]) == pytest.approx(
        record["metrics"]["Kt_rim"], rel=TOL)


def test_fitness_is_negative_total_loss(record, scored):
    """pygad maximises, so pygad_fitness must stay the negated sum (wheel_fea.py:629).
    A sign flip here would invert the entire search."""
    _, loss_terms = scored
    vec = [record["genes"][name] for name in W.GENE_NAMES]
    assert W.pygad_fitness(None, vec, 0) == pytest.approx(-sum(loss_terms.values()),
                                                          rel=TOL)


def _genome_hash(genes):
    """Byte-identical copy of wheel_step_export.py:117.

    Deliberately duplicated rather than imported: that module needs CadQuery and runs in
    a different interpreter, so importing it here would make this test unrunnable in
    env-opt.  The duplication is safe precisely because this test pins the result.
    """
    canon = json.dumps({k: round(float(v), 12) for k, v in sorted(genes.items())})
    return hashlib.sha256(canon.encode()).hexdigest()[:7]


def test_genome_hash_matches_manifest(record):
    """The manifest records which genome the STEP on disk was built from.  This is the
    traceability invariant the whole pipeline's provenance rests on, and it breaks if
    anyone adds a key inside `genes`."""
    with open(os.path.join(HERE, "export", "wheel_step_manifest.json")) as fh:
        manifest = json.load(fh)
    assert _genome_hash(record["genes"]) == manifest["genome_hash"]


def test_evaluate_design_is_pure(record, scored):
    """Stage 3 will call evaluate_design on arbitrary vectors outside the GA, and the
    mesh/FEA path will call the geometry helpers concurrently.  Scoring twice must give
    identical results and must not mutate the input."""
    vec = [record["genes"][name] for name in W.GENE_NAMES]
    before = list(vec)
    metrics_a, loss_a = W.evaluate_design(vec)
    assert vec == before
    metrics_b, loss_b = scored
    assert loss_a == loss_b
    assert metrics_a == metrics_b
