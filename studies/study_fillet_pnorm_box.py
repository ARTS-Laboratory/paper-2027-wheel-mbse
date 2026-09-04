"""
=============================================================================
  (r, p) ACROSS §82's THIRTY-TWO HELD-OUT GENOMES — IS 16 A CONSTANT OF THE
  OBJECTIVE, OR A PROPERTY OF THE TWO DESIGNS THAT WERE MEASURED
=============================================================================
    .venv-opt/bin/python studies/study_fillet_pnorm_box.py   (make filletpnormbox)

PLAN.md §99's ranked successor 1 (named at §95 as item 2, unchanged through §96-§99).
FILLET_PLAN.md STEP 3.

WHY THIS EXISTS
---------------
§95 swept `(r, p)` at two genomes — shipped and `b029622` — and found r=0.45 mm,
p=16 the cross-cell answer at both: the smallest radius resolved everywhere, paired
with the largest exponent clearing observed order 1.50 in every cell at that radius.
Its own scope note said what two genomes cannot say:

    "TWO GENOMES IS NOT A DESIGN SPACE.  r = 0.45 and p = 16 hold at the shipped
     genome and at b029622 and the two disagree about the hub's largest p by 50%
     (24 against 16), which is the visible edge of a spread nothing here has
     bounded.  §82's thirty-two held-out genomes are the instrument for that and
     this does not use them."

This is that measurement.  It reuses `study_fillet_pnorm`'s quantity, kernel and
Richardson machinery UNCHANGED — nothing here is re-derived — and widens `genomes`
from two named paths to the held-out draw's own gene vectors.  `verdict()`'s
cross-cell answer already aggregates over every `(genome, junction)` cell it is
handed, so running it on 28 genomes instead of 2 costs no new logic: the same
function that produced "r=0.45, p=16 at two designs" now says whether that holds
at the box.

THE DRAW IS READ, NOT REPEATED.  `studies/study_fillet_block.json`'s
`sector.genomes_held_out` is the committed artifact `test_fillet_block.py` already
holds disjoint from the in-sample draw (§78, §82's own instrument).  This driver
reads its 32 gene vectors and their `built` flags rather than re-running
`sweep_genomes`, which would re-spend several hundred `evaluate_design` /
`sector_control` calls to reproduce a set already on disk under the same seed.

SIX OF THIRTY-TWO ARE EXCLUDED, AND THE REASON IS §82's, NOT THIS DRIVER'S.
`built: false` means the genome's filleted sector construction refuses AT ITS OWN
RADII — the barrier half's own admissible-set question, already closed at §82 and
not reopened here.  A refusal there means `wheel_wheel.build_wheel(..., fillet=True)`
has nothing to solve, so each excluded genome's own recorded reason is carried into
this artifact rather than re-discovered.

SCOPE: the same as §95's — LINEAR kinematics, ONE phase, `coarse..fine` (`smoke` is
excluded from every Richardson for `study_deflection_gci`'s reason and is dropped
from the ladder here rather than solved and discarded, which is the only departure
from `study_fillet_pnorm`'s own default ladder).  Nothing here touches
`wheel_objective` and nothing is promoted; `test_nothing_wires_the_fillet_into_the_
objective` is untouched.
"""

import argparse
import json
import os
import tempfile
import time

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import _gate_guard

import jax_config  # noqa: F401
import wheel_wheel as WW
import wheel_genome as WG
import study_fillet_block as FB
import study_fillet_pnorm as SP

HERE = os.path.dirname(os.path.abspath(__file__))
FILLET_BLOCK_ARTIFACT = os.path.join(HERE, "study_fillet_block.json")

# `smoke` dropped — it is excluded from every Richardson (`SP.EXTRAPOLATE_FROM`) and
# solving it 28 times would spend ~30% more wall time on rungs nothing below reads.
LADDER = ("coarse", "medium", "fine")

FIXED_RADIUS_MM = 0.45   # §95 / §99's taken radius
FIXED_EXPONENT = 16.0    # §95 / §99's taken exponent


# ---------------------------------------------------------------------------
# THE HELD-OUT DRAW, READ FROM ITS OWN COMMITTED ARTIFACT
# ---------------------------------------------------------------------------

def held_out_records(source=FILLET_BLOCK_ARTIFACT):
    """The 32 held-out genomes as `{name, orientation, genes, built, why}` dicts.

    Flattened out of `sector.genomes_held_out.groups` in a fixed order (sorted
    orientation key, then draw order within it) so `ho00`..`ho31` name the same
    genome on every run.
    """
    with open(source) as fh:
        blk = json.load(fh)
    ho = blk["sector"]["genomes_held_out"]
    expected_seed = FB.GENOME_SWEEP_SEED + FB.GENOME_HELD_OUT_OFFSET
    if ho["seed"] != expected_seed:
        raise SystemExit(
            "%s's held-out draw is seeded %d, not %d (GENOME_SWEEP_SEED + "
            "GENOME_HELD_OUT_OFFSET) — it is not §82's draw; regenerate with "
            "`make filletblock` before running this driver." % (
                source, ho["seed"], expected_seed))
    out = []
    idx = 0
    for orient_key in sorted(ho["groups"]):
        for row in ho["groups"][orient_key]:
            out.append({"name": "ho%02d" % idx, "orientation": orient_key,
                        "genes": row["genes"], "built": bool(row["built"]),
                        "why": row.get("why")})
            idx += 1
    return out, int(ho["seed"]), int(ho["per_orientation"])


def write_genome_files(records, tmpdir):
    """One genome file per record, in `best_solution.json`'s own shape.

    `sweep_genomes` stores each held-out genome as a bare 14-vector
    (`[float(x) for x in vec]`); `study_corner_singularity.load_genes` reads a
    NAMED dict and converts it with `wheel_genome.genes_to_vector` — the two are
    the same information, `vector_to_genes` undoes the vector-to-dict step
    `sweep_genomes` never needed to take.
    """
    paths = []
    for rec in records:
        p = os.path.join(tmpdir, rec["name"] + ".json")
        with open(p, "w") as fh:
            json.dump({"genes": WG.vector_to_genes(rec["genes"])}, fh)
        paths.append((rec["name"], p))
    return paths


# ---------------------------------------------------------------------------
# THE LADDER, GENOME BY GENOME — `study_fillet_pnorm.build`, WITH ONE CHANGE:
# A GENOME THAT REFUSES TO SOLVE DOES NOT TAKE THE OTHERS' WORK WITH IT
# ---------------------------------------------------------------------------

def build_box(genomes, ladder=LADDER, radii=SP.RADII_MM, exponents=SP.EXPONENTS):
    """`study_fillet_pnorm.build`'s body, minus section C, plus a per-genome try.

    §95's `build` has no failure path because both its genomes are known-good —
    one shipped, one an optimiser's own output.  Twenty-six of these are neither:
    `built` in `study_fillet_block.json` says only that the SECTOR construction
    accepts the genome at its own radii, a cheaper and different check than an
    actual filleted linear solve at three mesh densities.  A solve that refuses
    here is recorded and skipped — `sweep_genomes`' own rule, "a drawn genome must
    never kill the driver" — rather than losing every genome solved before it.
    """
    rep = {"genomes": dict(genomes), "ladder": list(ladder),
           "extrapolate_from": list(SP.EXTRAPOLATE_FROM),
           "radii_mm": list(radii), "reference_radius_mm": SP.REFERENCE_RADIUS_MM,
           "exponents": list(exponents), "kernels": list(SP.KERNELS),
           "kinematics": "linear", "n_phase": 1,
           "kernel_formula": "W = max(0, 1 - d^2/r^2)^3",
           "rungs": {}, "convergence": {}, "corner_clearances_mm": {},
           "solve_failures": {}, "smoothness": None}
    for gname, path in genomes:
        print("  %s  (%s)" % (gname, path), flush=True)
        try:
            genes, rungs = SP.run_ladder(path, ladder, radii, exponents)
        except Exception as exc:
            print("    REFUSED: %s: %s" % (type(exc).__name__, exc), flush=True)
            rep["solve_failures"][gname] = "%s: %s" % (type(exc).__name__, exc)
            continue
        rep["rungs"][gname] = rungs
        rep["convergence"][gname] = SP.converge(rungs, SP.EXTRAPOLATE_FROM, radii,
                                                exponents)
        rep["corner_clearances_mm"][gname] = SP.corner_clearances_mm(
            genes, WW.get_config(ladder[-1]))
    rep["verdict"] = SP.verdict(rep["convergence"], radii, exponents)
    rep["priced"] = SP.price_recommendation(rep)
    return rep


# ---------------------------------------------------------------------------
# THE FIXED-CELL READING — §95/§99 ALREADY TOOK (r, p) = (0.45, 16); DOES IT HOLD
# ---------------------------------------------------------------------------

def fixed_cell_rows(rep, r=FIXED_RADIUS_MM, p=FIXED_EXPONENT):
    """Every `(genome, junction)`'s own reading at the ALREADY-TAKEN cell.

    `verdict()`'s cross-cell answer picks whatever radius and exponent survive
    everywhere, which could in principle move off (0.45, 16) once the box is
    added.  §99 has already spent that pair on a weight (89.21).  This is the
    direct question: at the pair actually taken, what does the box do to it.
    """
    import wheel_objective as WO
    rk, pk = "%g" % r, "%g" % p
    rows = []
    for gname, per_j in rep["convergence"].items():
        for lab, e in per_j.items():
            blk = e["radii"][rk]["bump3"]
            o = blk["exponents"][pk]
            v = o["values_mpa"][-1]
            rows.append({
                "genome": gname, "junction": lab,
                "mass_finest_pair_pct": blk["mass_finest_pair_pct"],
                "region_resolved": blk["mass_finest_pair_pct"] <= SP.MASS_DRIFT_PCT,
                "sigma_fillet_mpa": v,
                "util": v / WO.ALLOWABLE_STRESS_MPA,
                "order_h_fillet": o["h_fillet"]["observed_order_p"],
                "order_h_global": o["h_global"]["observed_order_p"],
                "order_clears_1_50": (
                    o["h_fillet"]["observed_order_p"] is not None
                    and o["h_global"]["observed_order_p"] is not None
                    and min(o["h_fillet"]["observed_order_p"],
                           o["h_global"]["observed_order_p"]) >= 1.5)})
    return rows


def _print_box(rep, records, built, excluded):
    print("\n" + "=" * 78)
    print("  THE HELD-OUT DRAW")
    print("=" * 78)
    print("  seed %d, %d per orientation, %d of %d genomes' filleted sector builds "
          "at its own radii" % (rep["held_out_seed"], rep["held_out_per_orientation"],
                                len(built), len(records)))
    for r in excluded:
        print("    excluded  %-5s orientation %-14s %s"
              % (r["name"], r["orientation"], r["why"]))
    if rep["solve_failures"]:
        print("  %d genome(s) built their sector but refused the ladder solve:"
              % len(rep["solve_failures"]))
        for g, why in rep["solve_failures"].items():
            print("    REFUSED  %-10s %s" % (g, why))

    SP._print(rep)

    print("\n" + "=" * 78)
    print("  THE ALREADY-TAKEN CELL, ACROSS THE BOX — r_sup = %g mm, p = %g "
          "(§95 / §99)" % (FIXED_RADIUS_MM, FIXED_EXPONENT))
    print("=" * 78)
    rows = fixed_cell_rows(rep)
    print("    %-10s %-5s  resolved  mass_drift%%  order(fil/glob)   util"
          % ("genome", "junc"))
    for row in sorted(rows, key=lambda r: r["util"], reverse=True):
        of = "%5.2f" % row["order_h_fillet"] if row["order_h_fillet"] else "  -  "
        og = "%5.2f" % row["order_h_global"] if row["order_h_global"] else "  -  "
        print("    %-10s %-5s  %-8s  %9.4f%%   %s / %s     %.4f%s"
              % (row["genome"], row["junction"],
                 "yes" if row["region_resolved"] else "NO",
                 row["mass_finest_pair_pct"], of, og, row["util"],
                 "" if row["order_clears_1_50"] else "   <- order floor MISSED"))
    n_ok = sum(1 for r in rows if r["region_resolved"] and r["order_clears_1_50"])
    n_breach = sum(1 for r in rows if r["util"] > 1.0)
    print("\n    %d of %d cells clear both gates at (0.45, 16); %d of %d read over "
          "the allowable (util > 1.0, informational — LINEAR, one phase, no fitness "
          "verdict)." % (n_ok, len(rows), n_breach, len(rows)))

    rec = rep["verdict"]["recommended"]
    print("\n    THE BOX'S OWN CROSS-CELL ANSWER: %s"
          % ("NONE — no (r, p) resolves and clears the order floor everywhere"
             if rec is None else "r_sup = %g mm, p = %g" % (rec["radius_mm"],
                                                            rec["exponent"])))
    if rec is not None:
        print("    against the two-genome answer of r_sup = %g mm, p = %g."
              % (FIXED_RADIUS_MM, FIXED_EXPONENT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ladder", default=",".join(LADDER))
    ap.add_argument("--radii", default=",".join("%g" % r for r in SP.RADII_MM))
    ap.add_argument("--exponents", default=",".join("%g" % p for p in SP.EXPONENTS))
    ap.add_argument("--held-out-source", default=FILLET_BLOCK_ARTIFACT)
    ap.add_argument("--out", default="study_fillet_pnorm_box.json")
    args = ap.parse_args()

    ladder = tuple(s.strip() for s in args.ladder.split(",") if s.strip())
    radii = tuple(float(s) for s in args.radii.split(",") if s.strip())
    exps = tuple(float(s) for s in args.exponents.split(",") if s.strip())

    _gate_guard.refuse_degraded_out(ap, args, "study_fillet_pnorm_box.json", [
        (not {"coarse", "medium", "fine"} <= set(ladder),
         "--ladder %s drops a rung the Richardson extrapolation needs" % args.ladder),
        (set(radii) < set(SP.RADII_MM), "--radii %s drops one of the committed %s"
                                        % (args.radii,
                                           ",".join("%g" % r for r in SP.RADII_MM))),
        (set(exps) < set(SP.EXPONENTS), "--exponents %s drops one of the committed %s"
                                        % (args.exponents,
                                           ",".join("%g" % p for p in SP.EXPONENTS))),
        (os.path.abspath(args.held_out_source) != FILLET_BLOCK_ARTIFACT,
         "--held-out-source %s is not §82's committed draw" % args.held_out_source),
    ])

    records, seed, per_orientation = held_out_records(args.held_out_source)
    built = [r for r in records if r["built"]]
    excluded = [r for r in records if not r["built"]]

    tmpdir = tempfile.mkdtemp(prefix="fillet_pnorm_box_")
    genomes = list(SP.GENOMES) + write_genome_files(built, tmpdir)

    t0 = time.time()
    rep = build_box(genomes, ladder, radii, exps)
    rep["seconds"] = time.time() - t0
    rep["held_out_seed"] = seed
    rep["held_out_per_orientation"] = per_orientation
    rep["held_out_source"] = os.path.relpath(args.held_out_source, PP.ROOT)
    rep["held_out_included"] = [r["name"] for r in built]
    rep["held_out_excluded"] = [{"name": r["name"], "orientation": r["orientation"],
                                 "why": r["why"]} for r in excluded]
    rep["n_genomes"] = len(genomes)

    _print_box(rep, records, built, excluded)

    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(rep, fh, indent=2)
    print("\n    wrote %s  (%.1f s, %d genomes)" % (args.out, rep["seconds"],
                                                     len(genomes)))

    # NO PASS/FAIL AND EXIT 0, for `study_fillet_pnorm`'s reason: whether 16 is a
    # constant of the objective or a property of two designs is a measurement, not
    # a threshold.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
