"""REDS Step 1 — is `max/min over the drawn rows` a statistic that can carry a gate?

Two tests assert `ratio > 3.0` on a max/min taken over a Latin-hypercube sample:

    tests/test_wheel_fea.py::test_the_beam_to_wheel_ratio_is_not_a_constant
        -> study_wheel_fea.run_beam_blindness(...)["fea_over_beam_ratio"]
    tests/test_gnl.py::test_the_correction_is_not_a_constant_over_the_design_space
        -> study_gnl.run_design_space(...)["iso_rel_diff_ratio"]

A max/min is an estimator of the RANGE of the underlying distribution, and the range of a
sample grows without bound with the number of draws.  So the quantity a `> 3.0` gate is
placed on is not a property of the design space at all — it is a property of the design
space AND the sample size AND the seed.  This driver measures both dependences so the
claim is earned rather than asserted.

It also records the statistic the tests' FIRST assertion rests on — the coefficient of
variation behind `correction_factor_is_defensible` — at every cell, because the argument
for retiring max/min is only admissible if the conclusion survives on a stable statistic.

Run:
    .venv-opt/bin/python studies/study_reds_ratio_stability.py --which beam --seed 7 --n 6
    .venv-opt/bin/python studies/study_reds_ratio_stability.py --which gnl  --seed 7 --n 4

One cell per invocation, printed as a single JSON line, so a caller can fan the grid out
across cores.  `--collect` merges a directory of those lines into the report table.
"""

import argparse
import glob
import json
import os
import time

import project_paths as PP

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# WHAT THE TWO TESTS IMPORT — the retirement of `max/min > 3.0`, as constants
# ---------------------------------------------------------------------------
# Measured 2026-08-15 by this driver: 109 cells, written to
# `studies/study_reds_ratio_stability.json`.  20 seeds at each test's own `n`, an `n`
# sweep at seed 7, and — for the beam study, whose statistic is a property of the GENE BOX
# — the whole thing again at the default 1.2 mm wall floor as well as the 2.0 mm floor the
# test pins.  Every cell returned exactly the rows it asked for, so none of the numbers
# below is a draw-exhaustion artefact.
#
#   beam @2.0   ratio 1.570 - 30.129 over 20 seeds (n=6)   passes `> 3.0` in  7/20
#               n = 6, 12, 24, 48, 96  ->  2.413, 9.995, 34.968, 34.968, 48.123
#   beam @1.2   ratio 1.767 - 32.655 over 20 seeds (n=6)   passes `> 3.0` in 11/20
#   gnl         ratio 2.167 - 85.501 over 20 seeds (n=4)   passes `> 3.0` in 11/20
#               n = 4, 8, 12, 16, 24, 48  ->  2.167, 4.203, 9.026, 9.026, 51.790, 51.790
#
# SEED 7 — THE ONE BOTH TESTS HARD-CODE — IS A LOW OUTLIER IN BOTH.  That is the whole of
# the two failures; nothing about the wheel changed.
#
# BOTH STATISTICS ARE SAMPLE-DEPENDENT, and pretending otherwise would be the wrong
# lesson.  The difference that decides which one can carry a gate is WHICH SIDE OF ITS
# BOUND the sampling noise can carry it across:
#
#   * `max/min` is an estimator of the RANGE, so it grows without bound with `n` — and it
#     grows THROUGH `3.0` from the failing side (beam 2.413 at n=6 to 34.968 at n=24).
#     Its verdict is decided by the draw: 13/20 seeds fail it on a tree where the
#     conclusion it stands for is true at 20/20.
#   * the CV is an estimator of a population parameter.  It moves with the sample too —
#     gnl runs 0.258 to 1.676 over the `n` sweep — but every one of those moves is AWAY
#     from the 0.10 bar.  Its floor over all 109 cells is 0.1450 (beam @2.0, seed 0), and
#     `correction_factor_is_defensible` is False in 109/109.  No draw flips the verdict.
#
# THE DERIVATION OF THE GATE, so it is not a picked number: take the CV's floor over the
# whole grid and floor it to two decimals.  0.1450 -> 0.14.  One constant for both tests,
# set by the worse of the two, so neither is tuned to its own run — the gnl test inherits
# a bound its own floor (0.2577) clears by 1.8x and did not choose.
#
# THIS IS NOT A LOOSENING, and the form matters (PLAN §28): `correction_factor_is_
# defensible` is DEFINED in both studies as `cv < 0.10`, so the CV is the claim's own
# arithmetic and `max/min` never was.  `cv > 0.14` strictly implies `cv > 0.10` implies
# `not correction_factor_is_defensible` — the assertion that was already in both tests and
# already passing.  So the replacement is strictly stronger than the surviving half of the
# old test, at a bound 40% inside the claim's own pre-registered bar.
#
# It is NOT strictly stronger than the retired half, and that comparison is not available:
# relative margins across a heavy-tailed range estimator and a CV are not commensurable
# (on the 7 beam seeds where `> 3.0` passed, it did so by 1.03x-10.04x; `cv > 0.14` passes
# those same seeds by 2.11x-4.11x).  The claim made here is the §28 one — the retired
# constant was never the claim's arithmetic — not a margin comparison.
OFF_RAMP_CV_BAR = 0.10        # both studies' `correction_factor_is_defensible` bar
GATE_CORRECTION_CV = 0.14     # floor(measured CV floor 0.1450, 2 dp)
RETIREMENT_SEEDS = (0, 1, 2, 3, 4)   # the first five, NOT selected for difficulty:
#                                    # beam @2.0 floor 0.1450, gnl floor 0.2653.  Taking
#                                    # the first five rather than the five worst is what
#                                    # keeps the ensemble from being fitted to the grid.


def cell_beam(seed, n, min_wall=2.0):
    """One `run_beam_blindness` cell, under the test's own conditions.

    The test wraps the call in `set_min_wall(2.0)` and the statistic is a property of the
    GENE BOX, not of the genome (see the test's docstring, §14) — so measuring it at the
    default 1.2 floor is measuring a different quantity.  `--min-wall` selects which,
    because the reason the test pins 2.0 is that the `> 3.0` margin was calibrated there,
    and if that margin is retired the pin has no rationale left.  Restored in `finally`
    for the same reason the test does it: a leaked floor is baked into module-scoped
    fixtures elsewhere.
    """
    import study_wheel_fea as swf
    import wheel_fea as wf

    genes = swf.load_genes()
    before = wf.MIN_WALL_MM
    try:
        wf.set_min_wall(min_wall)
        rep = swf.run_beam_blindness(genes, "smoke", n=n, seed=seed)
    finally:
        wf.set_min_wall(before)
    return {
        "ratio": rep.get("fea_over_beam_ratio"),
        "cv": rep.get("fea_over_beam_cv"),
        "defensible": rep.get("correction_factor_is_defensible"),
        "vmin": rep.get("fea_over_beam_min"),
        "vmax": rep.get("fea_over_beam_max"),
        "n_rows": len(rep["rows"]),
        "n_drawn": rep["n_drawn"],
    }


def cell_gnl(seed, n, min_wall=None):
    """One `run_design_space` cell, under the test's own conditions (smoke, max_draws)."""
    import json as _json

    import study_gnl as gnl
    import wheel_genome as wg

    with open(os.path.join(PP.ROOT, "best_solution.json")) as fh:
        genes = wg.genes_to_vector(_json.load(fh)["genes"])
    rep = gnl.run_design_space(genes, "smoke", n=n, seed=seed, max_draws=2000)
    return {
        "ratio": rep.get("iso_rel_diff_ratio"),
        "cv": rep.get("iso_rel_diff_cv"),
        "defensible": rep.get("correction_factor_is_defensible"),
        "vmin": rep.get("iso_rel_diff_min"),
        "vmax": rep.get("iso_rel_diff_max"),
        "n_rows": len(rep["rows"]),
        "n_drawn": rep["n_drawn"],
        "n_diverged": rep["n_diverged"],
    }


CELLS = {"beam": cell_beam, "gnl": cell_gnl}


def collect(pattern):
    """Merge cells from either shape: one-JSON-line-per-cell, or a merged report.

    The fan-out writes one cell per file as a single line; the merged report this driver
    writes is indented and nests them under "cells".  Accepting both is what lets
    `--glob` be pointed at this driver's OWN output — re-tabulating a saved grid without
    re-solving it, which is most of what anyone reading it back wants to do.
    """
    cells = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            text = fh.read()
        try:
            blob = json.loads(text)
        except json.JSONDecodeError:
            blob = None
        if isinstance(blob, dict) and "cells" in blob:
            cells.extend(blob["cells"])
        elif isinstance(blob, dict):
            cells.append(blob)
        else:
            cells.extend(json.loads(ln) for ln in text.splitlines()
                         if ln.strip().startswith("{"))
    return cells


def _table(cells, which):
    """One table per (study, gene box).

    The beam statistic is a property of the GENE BOX — its test pins the wall floor at 2.0
    while the project ships 1.2 — so cells from the two floors are two different
    populations and pooling them into one row would read as seed noise.  Split.
    """
    rows = [c for c in cells if c["which"] == which]
    for wall in sorted({c.get("min_wall") for c in rows}, key=lambda w: (w is None, w)):
        _one_table(which, [c for c in rows if c.get("min_wall") == wall], wall)


def _one_table(which, rows, wall):
    if not rows:
        return
    n_test = {"beam": 6, "gnl": 4}[which]
    print(f"\n=== {which}" + (f", wall floor {wall}" if wall is not None else "") + " ===")
    seeds = sorted((c for c in rows if c["n"] == n_test), key=lambda c: c["seed"])
    if seeds:
        print(f"  seed sweep at the test's own n={n_test}:")
        print("    seed  " + "  ".join(f"{c['seed']:>7d}" for c in seeds))
        print("    ratio " + "  ".join(f"{c['ratio']:>7.3f}" for c in seeds))
        print("    cv    " + "  ".join(f"{c['cv']:>7.3f}" for c in seeds))
        rr = [c["ratio"] for c in seeds]
        print(f"    range {min(rr):.3f} - {max(rr):.3f}  "
              f"(spread {max(rr) / min(rr):.2f}x); "
              f"passes >3.0 in {sum(r > 3.0 for r in rr)}/{len(rr)}")
        print(f"    cv    {min(c['cv'] for c in seeds):.3f} - "
              f"{max(c['cv'] for c in seeds):.3f}   (bar is 0.10)")
    ns = sorted((c for c in rows if c["seed"] == 7), key=lambda c: c["n"])
    if len(ns) > 1:
        print("  n sweep at seed=7:")
        print("    n     " + "  ".join(f"{c['n']:>7d}" for c in ns))
        print("    ratio " + "  ".join(f"{c['ratio']:>7.3f}" for c in ns))
        print("    cv    " + "  ".join(f"{c['cv']:>7.3f}" for c in ns))
    bad = [c for c in rows if c["defensible"]]
    print(f"  correction_factor_is_defensible true in {len(bad)}/{len(rows)} cells"
          + ("  <-- THE DIAGNOSIS IS WRONG, STOP" if bad else "  (never — the "
             "conclusion holds on the stable statistic in every cell)"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=sorted(CELLS) + ["collect"], required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--min-wall", type=float, default=2.0,
                    help="beam only: gene-box wall floor to measure in")
    ap.add_argument("--glob", default=None, help="collect: shell glob of cell files")
    ap.add_argument("--out", default=None, help="collect: merged json path")
    args = ap.parse_args(argv)

    if args.which == "collect":
        cells = collect(args.glob)
        for w in ("beam", "gnl"):
            _table(cells, w)
        if args.out:
            with open(args.out, "w") as fh:
                json.dump({"cells": cells}, fh, indent=1)
            print(f"\nwrote {args.out}  ({len(cells)} cells)")
        return 0

    t0 = time.time()
    out = CELLS[args.which](args.seed, args.n, args.min_wall)
    out.update({"which": args.which, "seed": args.seed, "n": args.n,
                "min_wall": args.min_wall if args.which == "beam" else None,
                "elapsed_s": round(time.time() - t0, 1)})
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
