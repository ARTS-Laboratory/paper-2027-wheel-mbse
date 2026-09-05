"""Every genome file's `genes` block must be serialised in `GENE_NAMES` order.

THE TWO READERS DISAGREE ABOUT WHETHER ORDER MATTERS, AND ONLY ONE OF THEM SAYS SO.

  `wheel_genome.genome_hash`  is order-INDEPENDENT — it hashes `sorted(genes.items())`
                              (wheel_genome.py:122-132), so it cannot see a permutation.
  `wheel_genome.genes_to_vector` is order-SAFE — it indexes by name (wheel_genome.py:48).
  `wheel_stage3.load_genes`   is order-DEPENDENT — `list(json.load(fh)["genes"].values())`,
                              with no name lookup anywhere (wheel_stage3.py:980-982).

So a genome file written in sorted key order (`R_hub, R_rim, cx1, ...` rather than
`cx1, cy1, ...`) would carry the CORRECT `genome_hash`, satisfy `test_promotion.py`'s pins,
export identically through `wheel_step_export`, and hand a SCRAMBLED 14-vector to any descent
started with `--genome`. `t0` would be read as `cx4`. Nothing else in the tree catches it,
because everything else that reads a gene dict reads it by name.

THIS IS A REGRESSION GUARD, NOT A BUG FIX, AND THE DISTINCTION IS WORTH KEEPING. All 28
genome files at the repo root were checked when this was written (2026-09-05) and all 28 were
already canonical — nothing on disk is wrong today. The hazard is latent and it stays latent
only while every writer happens to produce canonical order. `wheel_genome.save_record`
preserves the insertion order of the dict handed to it (`{"genes": genes}` then `json.dump`
with no `sort_keys`, wheel_genome.py:165-170), so "happens to" is doing real work in that
sentence: any caller that builds its dict from a `sorted()`, a merge, or a comprehension over
an unordered source silently breaks the invariant while every existing check stays green.

`studies/stage3_resume_genome.py` is the writer that prompted this — it copies a `genes` block
out of a Stage-3 trajectory and back into a genome file, which is exactly the shape of
operation that could permute one.

This is a file-level check: no FEA, no meshing, no jax. It costs milliseconds.
"""

import json
import os

import pytest

import wheel_genome as WG

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _genome_files():
    """Every repo-root JSON that is a genome record, by the same test
    `test_promotion.py:172-181` uses to find them — a dict with a dict `genes`."""
    found = []
    for name in sorted(os.listdir(HERE)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(HERE, name)) as fh:
                rec = json.load(fh)
        except (ValueError, OSError):
            continue
        if isinstance(rec, dict) and isinstance(rec.get("genes"), dict):
            found.append((name, rec["genes"]))
    return found


def test_there_are_genome_files_to_check():
    """The walk above returns [] just as happily when the glob breaks as when the tree is
    clean, and a silently empty parametrisation is a green test that checks nothing."""
    assert len(_genome_files()) >= 20, (
        "fewer than 20 genome records found at the repo root — the walk that finds them has "
        "probably broken, not the tree. 28 were present when this test was written.")


@pytest.mark.parametrize("name,genes", _genome_files(),
                         ids=[n for n, _ in _genome_files()])
def test_genes_are_serialised_in_canonical_order(name, genes):
    keys = list(genes.keys())
    assert keys == WG.GENE_NAMES, (
        f"{name}'s `genes` block is not in GENE_NAMES order.\n"
        f"  on disk : {keys}\n"
        f"  expected: {WG.GENE_NAMES}\n"
        "`genome_hash` cannot see this (it sorts first) and neither can any reader that goes "
        "by name, but `wheel_stage3.load_genes` takes `.values()` positionally — this file "
        "would start a descent from a permuted genome and every other check would stay green.")


@pytest.mark.parametrize("name,genes", _genome_files(),
                         ids=[n for n, _ in _genome_files()])
def test_gene_values_are_plain_numbers(name, genes):
    """`test_promotion.py:176-181` json-loads every one of these files inside a `try` that
    catches only `ValueError` and `OSError`, then hashes the result. A non-numeric gene value
    would raise `TypeError` out of that loop and turn an unrelated test red, so the constraint
    belongs here where the failure names the actual file."""
    bad = {k: v for k, v in genes.items() if not isinstance(v, (int, float))
           or isinstance(v, bool)}
    assert not bad, f"{name} has non-numeric gene values: {bad}"
