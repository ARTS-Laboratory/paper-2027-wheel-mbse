"""
=============================================================================
  COMPLIANT WHEEL — GENOME (names, bounds, normalisation, provenance)
=============================================================================
The 14-gene genome and everything that reads or writes it, in one place.

Stage 3 optimises in a normalised unit box rather than in millimetres, because the
genes span wildly different ranges — `cy*` covers 64 mm while `R_rim` covers 2.5 mm —
so a single gradient step size is meaningless in raw units.  `normalize`/`denormalize`
are that mapping.

THE CONTRACT THIS FILE PROTECTS
-------------------------------
`genome_hash` is the fingerprint the whole pipeline's traceability rests on.  It
hashes `sorted(genes.items())`, so **adding even one key inside `genes` changes the
hash of every genome ever recorded** and silently breaks every staleness comparison in
the repo.  New information belongs in new TOP-LEVEL keys of the record, which
`wheel_step_export.load_genome` (wheel_step_export.py:125) and `main` (:807)
demonstrably ignore.

numpy + stdlib only.  This module is imported by the numpy-side tooling and must never
pull in jax; see tests/test_import_hygiene.py.
=============================================================================
"""

import hashlib
import json
import os

import numpy as np

# Ordering contract between the flat 14-vector the GA and SGD work in, and the dict the
# exporter reads.  `evaluate_design` unpacks positionally in exactly this order
# (wheel_fea.py:560), so the two must never drift.
GENE_NAMES = [
    "cx1", "cy1", "cx2", "cy2", "cx3", "cy3", "cx4", "cy4",
    "t0", "t1", "t2", "t3",
    "R_hub", "R_rim",
]

N_GENES = len(GENE_NAMES)


# ---------------------------------------------------------------------------
# VECTOR <-> DICT
# ---------------------------------------------------------------------------

def genes_to_vector(genes, xp=np):
    """dict -> ordered 14-vector."""
    missing = [n for n in GENE_NAMES if n not in genes]
    if missing:
        raise KeyError(f"genome is missing {missing}")
    return xp.asarray([float(genes[n]) for n in GENE_NAMES])


def vector_to_genes(vec):
    """ordered 14-vector -> dict, with the exact key set the exporter expects."""
    vec = np.asarray(vec, dtype=float).ravel()
    if vec.size != N_GENES:
        raise ValueError(f"expected {N_GENES} genes, got {vec.size}")
    return {name: float(v) for name, v in zip(GENE_NAMES, vec)}


# ---------------------------------------------------------------------------
# BOUNDS AND NORMALISATION
# ---------------------------------------------------------------------------

def bounds_arrays(gene_space):
    """(low, high, range) as float arrays, from wheel_fea's GENE_SPACE list-of-dicts."""
    low = np.array([g["low"] for g in gene_space], dtype=float)
    high = np.array([g["high"] for g in gene_space], dtype=float)
    return low, high, high - low


def normalize(vec, low, high):
    """Raw genes (mm) -> unit box [0,1]^14.

    No `xp` parameter: plain arithmetic against numpy constants already works
    unchanged on numpy arrays and JAX tracers alike.
    """
    return (vec - low) / (high - low)


def denormalize(z, low, high):
    """Unit box -> raw genes (mm)."""
    return low + z * (high - low)


def bound_saturation(vec, low, high, tol_frac=0.01):
    """Genes sitting within `tol_frac` of a bound, as (name, value, 'low'|'high', bound).

    A saturated gene is a boundary optimum, and whether that is informative or a
    warning depends on the gene: a pinned `t*` means MIN_WALL_MM printability is the
    active constraint, which is real.  A pinned `cy*` looks like the box is too tight
    but measurably is not — see CY_BOUND_MM in wheel_fea.py.
    """
    vec = np.asarray(vec, dtype=float)
    span = high - low
    out = []
    for name, v, lo, hi, w in zip(GENE_NAMES, vec, low, high, span):
        if w <= 0:
            continue
        if v - lo <= tol_frac * w:
            out.append((name, float(v), "low", float(lo)))
        elif hi - v <= tol_frac * w:
            out.append((name, float(v), "high", float(hi)))
    return out


def clip_to_bounds(vec, low, high, xp=np):
    """Project into the box.  `evaluate_design` does NOT clip (wheel_fea.py:550), so a
    gradient step that leaves the box is scored without complaint and the thickness is
    then silently clamped deep inside the physics (wheel_fea.py:359).  Stage 3 projects
    explicitly instead."""
    return xp.clip(vec, low, high)


# ---------------------------------------------------------------------------
# PROVENANCE
# ---------------------------------------------------------------------------

def genome_hash(genes):
    """Short stable fingerprint of a gene dict.

    Byte-identical to `wheel_step_export.genome_hash` (wheel_step_export.py:117), and
    deliberately DUPLICATED rather than imported: that module needs CadQuery and runs in
    a different interpreter, so importing it here would make this module unusable in
    env-opt.  `tests/test_golden.py` pins the result against the value recorded in
    wheel_step_manifest.json, which is what makes the duplication safe.
    """
    canon = json.dumps({k: round(float(v), 12) for k, v in sorted(genes.items())})
    return hashlib.sha256(canon.encode()).hexdigest()[:7]


def load_record(path):
    """Read a genome record and attach source provenance, mirroring
    `wheel_step_export.load_genome` so the two agree on what a record is."""
    with open(path) as fh:
        rec = json.load(fh)
    if "genes" not in rec:
        raise KeyError(f"{path} has no 'genes' block")
    rec["_source_path"] = os.path.abspath(path)
    rec["_source_mtime"] = os.path.getmtime(path)
    rec["_hash"] = genome_hash(rec["genes"])
    return rec


def save_record(path, genes, **extra_blocks):
    """Write a genome record.

    Enforces the invariant that makes the pipeline traceable: `genes` carries exactly
    the 14 canonical keys and nothing else.  Everything else — metrics, FEA results,
    stage-3 history — goes in top-level blocks, which downstream readers ignore safely.
    """
    genes = {k: float(v) for k, v in genes.items()}
    if sorted(genes) != sorted(GENE_NAMES):
        extra = sorted(set(genes) - set(GENE_NAMES))
        missing = sorted(set(GENE_NAMES) - set(genes))
        raise ValueError(
            f"`genes` must hold exactly the {N_GENES} canonical keys — "
            f"unexpected {extra}, missing {missing}. Adding a key inside `genes` "
            f"changes genome_hash for every genome and breaks all staleness checks; "
            f"put it in a top-level block instead."
        )
    record = {"genes": genes}
    record.update(extra_blocks)
    # Private keys are provenance attached at load time, never persisted.
    record.pop("_hash", None)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    return genome_hash(genes)


def describe(vec, low, high):
    """One-line-per-gene table, for logging a design at a checkpoint."""
    vec = np.asarray(vec, dtype=float)
    sat = {n: w for n, _, w, _ in bound_saturation(vec, low, high)}
    lines = [f"  {'gene':<6s} {'value':>10s} {'low':>9s} {'high':>9s}  {'frac':>6s}"]
    for name, v, lo, hi in zip(GENE_NAMES, vec, low, high):
        frac = (v - lo) / (hi - lo) if hi > lo else 0.0
        mark = f"  <- pinned {sat[name]}" if name in sat else ""
        lines.append(f"  {name:<6s} {v:10.4f} {lo:9.3f} {hi:9.3f}  {frac:6.3f}{mark}")
    return "\n".join(lines)
