"""The GA driver's CLI contract: reproducibility, and not clobbering real artifacts.

Reproducibility is not cosmetic here.  Stage 2 and Stage 3 record which genome they
consumed (`input_genome_hash`), and that provenance is worthless if the run that
produced the genome cannot be repeated.  Before this, `pygad.GA` was constructed without
`random_seed` — the module-level `np.random.seed(42)` did not reach pygad's own
Generator, so the initial population differed every run.
"""

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Smallest run that still exercises every code path: init, fitness, tournament,
# crossover, mutation, elitism, the on_generation callback, scoring the winner, the
# genome write, the figure, and the CAD hand-off branch.  ~1 s.
TINY = ["--smoke", "--generations", "2", "--pop", "8"]


def _run(tmp_path, name, *extra):
    out = tmp_path / name
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "src", "wheel_fea.py"),
         *TINY, "--out", str(out), *extra],
        cwd=HERE, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    with open(out) as fh:
        return json.load(fh), proc.stdout


def test_same_seed_gives_identical_genome(tmp_path):
    a, _ = _run(tmp_path, "a.json")
    b, _ = _run(tmp_path, "b.json")
    assert a["genes"] == b["genes"], (
        "two runs at the same seed diverged — random_seed is not reaching pygad, and "
        "no Stage-2/3 provenance record can be trusted."
    )
    assert a["loss_terms"] == b["loss_terms"]


def test_different_seed_gives_different_genome(tmp_path):
    """Guards the opposite failure: a seed that is accepted but ignored would make
    every run identical and look 'reproducible' while actually being frozen."""
    a, _ = _run(tmp_path, "a.json")
    c, _ = _run(tmp_path, "c.json", "--seed", "7")
    assert a["genes"] != c["genes"]


def test_smoke_does_not_touch_real_artifacts(tmp_path):
    """A --smoke run must never rewrite best_solution.json or re-stamp wheel.step.

    The exporter reads best_solution.json unconditionally (wheel_step_export.py:125), so
    running it after a smoke run would rebuild the STEP from the PREVIOUS real genome
    with a fresh mtime — destroying the staleness signal warn_if_stale() exists to give.
    """
    # Paths, not bare names: the STEP artifacts moved to export/ while the genome and
    # its poster stay at the root, beside each other.
    targets = ["best_solution.json", "poster_summary.jpg",
               os.path.join("export", "wheel.step"),
               os.path.join("export", "wheel_step_manifest.json"),
               os.path.join("export", "wheel_nofillet.step")]
    before = {t: os.path.getmtime(os.path.join(HERE, t)) for t in targets
              if os.path.exists(os.path.join(HERE, t))}
    assert before, "no real artifacts on disk to protect — check the repo state"

    _, stdout = _run(tmp_path, "a.json")

    for t, mtime in before.items():
        assert os.path.getmtime(os.path.join(HERE, t)) == mtime, (
            f"a --smoke run modified {t}"
        )
    assert "Skipped:" in stdout, "smoke run should skip the CAD hand-off"


def test_out_of_tree_run_writes_nothing_into_the_repo(tmp_path):
    """A run with --out elsewhere must leave the repo directory alone entirely.

    Regression: --out redirected the genome but NOT the summary figure, which stayed
    hard-coded to poster_summary.jpg.  A parameter sweep therefore overwrote the
    committed poster with a throwaway design while best_solution.json still described
    the real one — the artifacts-disagreeing failure the CAD hand-off exists to prevent.
    """
    before = {f: os.path.getmtime(os.path.join(HERE, f)) for f in os.listdir(HERE)
              if os.path.isfile(os.path.join(HERE, f))}

    _run(tmp_path, "elsewhere.json")

    after = {f: os.path.getmtime(os.path.join(HERE, f)) for f in os.listdir(HERE)
             if os.path.isfile(os.path.join(HERE, f))}
    touched = [f for f in after if f not in before or after[f] != before.get(f)]
    assert touched == [], f"an out-of-tree run wrote into the repo: {touched}"

    # ...and the figure did get written, beside its genome.
    assert (tmp_path / "elsewhere_summary.jpg").exists()


def test_smoke_redirects_default_output():
    """Without --out, --smoke must retarget itself away from best_solution.json."""
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "src", "wheel_fea.py"), *TINY],
        cwd=HERE, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "best_solution_smoke.json" in proc.stdout
    smoke = os.path.join(HERE, "best_solution_smoke.json")
    try:
        assert os.path.exists(smoke)
    finally:
        for f in (smoke, os.path.join(HERE, "poster_summary_smoke.jpg")):
            if os.path.exists(f):
                os.remove(f)


@pytest.mark.parametrize("key", ["genes", "geometry", "metrics", "model",
                                 "loss_terms", "bound_saturation"])
def test_genome_record_schema(tmp_path, key):
    """The record layout Stage 2/3 and the exporter depend on."""
    rec, _ = _run(tmp_path, "a.json")
    assert key in rec


def test_genes_block_has_exactly_the_14_keys(tmp_path):
    """`genome_hash` (wheel_step_export.py:117) hashes sorted(genes.items()), so one
    extra key inside `genes` silently changes every hash in the repo.  This is the
    invariant Stage 3 must also honour when it rewrites the genome."""
    import wheel_fea
    rec, _ = _run(tmp_path, "a.json")
    assert sorted(rec["genes"]) == sorted(wheel_fea.GENE_NAMES)
