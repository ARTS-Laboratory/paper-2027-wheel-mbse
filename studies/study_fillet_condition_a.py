"""
=============================================================================
  CONDITION A — THE CONVERGENCE LADDER AT `b029622`, SVK, EIGHT PHASES
=============================================================================
    .venv-opt/bin/python studies/study_fillet_condition_a.py   (make filletconda)

PLAN.md §93's sequence step 1 / §99's ranked successor 2.  FILLET_PLAN.md STEP 3.

WHY THIS EXISTS
---------------
§93 named two conditions in front of wiring the fillet into the objective.  Condition B
(the stress term counts the fillet twice) was measured at §94 and closed — the double
count is worth exactly `0.0000`.  Condition A is still open, and §93 stated its check in
full:

    "The convergence ladder at `b029622`, SVK, eight phases, coarse/medium/fine, both
     meshes.  CHECK: the filleted mesh's axle-drop spread is materially smaller than the
     unfilleted mesh's on the SAME design — PART 12's ordering reproduced where it is
     load-bearing.  AND the admissibility disagreement resolves: at the finest rung, does
     the design read 1.0112 utilisation or 0.7954?  If the filleted mesh is not the
     converged one, this decision is reopened and nothing below happens."

FILLET_PLAN STEP 1 PART 12 measured the ordering this reproduces — filleted axle-drop
spread 0.141% against unfilleted 1.216% over `coarse..fine` — but at the SHIPPED genome,
under LINEAR kinematics, at ONE phase.  §94 measured the 1.0112 / 0.7954 disagreement at
`b029622`, under SVK, at eight phases — but at `coarse` alone, no mesh-density ladder at
all.  Neither number has been read at the place the other one is stated: the design and
kernel where the two meshes disagree (`b029622`, SVK) crossed with the ladder that says
whether either reading survives refinement.  This driver is that crossing.

THE QUANTITY IS THE OBJECTIVE'S OWN, READ ON TWO MESHES
----------------------------------------------------------
`wheel_objective.objective(..., meshes=meshes, kinematics="svk")` does not know whether
`meshes` came from `WW.build_wheel(..., fillet=True)` or the default — it reads whatever
field it is handed.  So a forward evaluation (no descent, no Stage 3) at each of
`coarse`/`medium`/`fine`, on each of the two mesh constructions, is `study_fillet_kt.py`'s
own `coarse`-only comparison extended along the one axis it never swept — the same
pattern `study_fillet_optimum._FilletedEvaluator` and `study_deflection_gci.run_ladder`
already use, combined rather than re-derived: `WW.build_wheel(fillet=...)` for the mesh
construction, `WO.objective(..., kinematics="svk")` for the reading, `study_deflection_gci
.richardson` for the extrapolation.

`axle_drop_mean_mm` is PART 12's own QoI; `stress_utilisation` is `wheel_objective.py`'s
`util_j = kt * agg / ALLOWABLE_STRESS_MPA`, the same field §94's table reads, on whichever
mesh built `agg`.

THE ORIENTATION IS PINNED ONCE, AT THE FINEST RUNG, ACROSS BOTH MESHES
--------------------------------------------------------------------------
`flank_orientation` reads the genes and the config, never a mesh — it is a discrete branch
about which sector block owns which flank, decided before any mesh is built — so the
fillet cannot move it and one pin serves both arms.  Pinning at the finest rung and
carrying it down is `study_deflection_gci`'s own rule, for its own reason: the
extrapolation is anchored on the finest point.

SCOPE
-----
SVK, eight phases, uniform stencil, `coarse..fine` — Condition A's own words, no `smoke`
(it is not part of the check and the ladder does not need a fourth point Richardson
never reads).  Nothing here touches `wheel_objective` and nothing is promoted;
`test_nothing_wires_the_fillet_into_the_objective` is untouched.  **This is a gate, not a
score**: a filleted spread that is NOT materially smaller, or an admissibility reading
that does not converge, is Condition A failing and this arc's own §93 says what that
means — "this decision is reopened and nothing below happens."
"""

import argparse
import json
import math
import os
import resource
import time

import numpy as np

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import _gate_guard

import jax_config  # noqa: F401
import wheel_genome as wg
import wheel_objective as WO
import wheel_wheel as WW
import study_deflection_gci as SG

HERE = os.path.dirname(os.path.abspath(__file__))

GENOME = "fillet_optimum_b029622.json"
N_PHASE = 8
KINEMATICS = "svk"
LADDER = ("coarse", "medium", "fine")
MESH_TYPES = ("unfilleted", "filleted")
# The orientation is pinned at the LADDER's finest rung regardless of which cell a given
# process is solving (see `pinned_orientation` below) — a single `--rung coarse` process
# still has to agree with a `--rung fine` one, or the two cannot be merged into one file.
FINEST = LADDER[-1]

TRACK_KEYS = ("axle_drop_mean_mm", "axle_drop_min_mm", "axle_drop_max_mm",
              "stress_utilisation", "mesh_mass_g", "min_scaled_jacobian",
              "max_stress_mpa")


def load_genes(path):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


def pinned_orientation(genes):
    return tuple(float(o) for o in WW.flank_orientation(genes, WW.get_config(FINEST)))


def peak_rss_gb():
    """This process's peak resident set, from the kernel's own accounting rather than a
    sampled `free -h` — exact for a one-cell process, since nothing else has run in it.
    `ru_maxrss` is kB on Linux (`man getrusage`); GiB matches how the wall-clock notes
    elsewhere in this repo (`Makefile:727`'s "20.6 GB") were read off `free -h`.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2


def solve_cell(genes, cfg_name, filleted, phases, pinned, kinematics):
    """One (rung, mesh construction) cell: build the phase stencil's meshes and read the
    objective on them.  Factored out of `run_ladder` so a single-cell process (`--rung`)
    and the whole-ladder path compute the identical entry rather than two implementations
    of the same six numbers drifting apart.
    """
    t0 = time.time()
    # `build_wheel`'s `fillet` is a three-way switch (`None` unfilleted, `True` this
    # genome's radii, a pair an explicit override) and NOT a bool — passing the loop's own
    # `False` would be read as a malformed radii pair, not as "unfilleted".
    meshes = [WW.build_wheel(genes, cfg_name, phase_deg=float(p), orientation=pinned,
                             fillet=True if filleted else None)
             for p in phases]
    mesh_s = time.time() - t0
    t1 = time.time()
    val, grad, brk = WO.objective(genes, cfg_name, normalized=False, phases=phases,
                                  meshes=meshes, orientation=pinned, kinematics=kinematics)
    solve_s = time.time() - t1
    rep = brk["report"]
    m0 = meshes[0]
    entry = {"n_elements": int(m0.n_elements), "n_nodes": int(m0.n_nodes),
             "h": 1.0 / math.sqrt(m0.n_elements),
             **{k: float(rep[k]) for k in TRACK_KEYS},
             "loss": float(val), "grad_norm": float(np.linalg.norm(grad)),
             "mesh_s": round(mesh_s, 1), "solve_s": round(solve_s, 1)}
    label = "filleted" if filleted else "unfilleted"
    print("  %-7s %-10s elem %6d  drop %8.5f mm  util %7.4f  "
          "min_sj %.4f  (%.1fs)"
          % (cfg_name, label, entry["n_elements"], entry["axle_drop_mean_mm"],
             entry["stress_utilisation"], entry["min_scaled_jacobian"],
             mesh_s + solve_s), flush=True)
    return entry


def run_ladder(genome=GENOME, ladder=LADDER, n_phase=N_PHASE, kinematics=KINEMATICS,
               resume_rows=(), checkpoint_path=None):
    """`resume_rows` are already-computed rows from a prior, interrupted run of the SAME
    (genome, n_phase, kinematics) — each rung this driver has already paid for (both mesh
    types, six figures each) is a real cost, and this run has been killed mid-ladder
    without explanation twice.  A rung already present in `resume_rows` is reused rather
    than recomputed; `checkpoint_path`, if given, is rewritten after every completed rung
    so a third kill loses at most one rung, not the whole ladder.

    This is the whole-ladder, one-process path.  It has been killed three times short of
    `fine` — see `main`'s `--rung` for the one-cell-per-process path that survives that.
    """
    genes = load_genes(genome)
    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    pinned = pinned_orientation(genes)

    done = {r["config"]: r for r in resume_rows}
    rows = []
    for name in ladder:
        if name in done:
            print("  %-7s  (resumed from checkpoint)" % name, flush=True)
            rows.append(done[name])
            continue
        row = {"config": name, "meshes": {}}
        for filleted in (False, True):
            label = "filleted" if filleted else "unfilleted"
            row["meshes"][label] = solve_cell(genes, name, filleted, phases, pinned,
                                              kinematics)
        rows.append(row)
        if checkpoint_path is not None:
            with open(checkpoint_path, "w") as fh:
                json.dump({"genome": genome, "n_phase": n_phase, "kinematics": kinematics,
                          "ladder": list(ladder), "orientation_pinned_at": FINEST,
                          "orientation": list(pinned), "rows": rows,
                          "complete": False}, fh, indent=2)
    return {"genome": genome, "n_phase": n_phase, "kinematics": kinematics,
           "ladder": list(ladder), "orientation_pinned_at": FINEST,
           "orientation": list(pinned), "rows": rows}


# ---------------------------------------------------------------------------
# ONE CELL PER PROCESS
# ---------------------------------------------------------------------------
# `run_ladder` above has been killed three times short of `fine` — the last time ~1h25m
# in, mid-JIT-compile on the fourth of six cells, with a background-process memory cap
# reported as the cause (not a system-wide OOM: `dmesg`/`journalctl` show nothing, and
# `free -h` shows tens of GB free once the process is gone).  `medium filleted` — the
# cell that killed that run — has separately SUCCEEDED at ~45 GB peak RSS when it was the
# only thing a fresh process ever computed.  JAX does not release compiled executables
# between cells, so a long-lived process's baseline climbs across the ladder even though
# each cell's own cost does not; a fresh process per cell returns that memory to the OS
# on exit, which is the one configuration this repo has actually observed to survive.
#
# `study_m9.py` solved the same class of problem (`fine` at 261k dof, `study_m9.py:141`:
# "the most likely thing in this repo to be OOM-killed by its own cgroup cap -- and a
# SIGKILL is not catchable") with a per-SECTION checkpoint and a `complete` flag that
# stays false, with no verdict, until every section is in.  This adopts the same shape at
# per-CELL granularity, because a cell here (up to 47 minutes, `medium filleted`) is
# already section-sized, and per-rung checkpointing (this file's first version) lost the
# 17.5 minutes already spent on `medium unfilleted` when the run died one cell later.

def _fresh_report(genome, n_phase, kinematics, ladder, pinned):
    return {"genome": genome, "n_phase": n_phase, "kinematics": kinematics,
           "ladder": list(ladder), "orientation_pinned_at": FINEST,
           "orientation": list(pinned), "rows": [], "complete": False}


def merge_cell(out_path, genome, n_phase, kinematics, ladder, pinned, cfg_name, label,
              entry):
    """Read `out_path` (if it names a report for this SAME genome/n_phase/kinematics),
    set `rows[cfg_name].meshes[label]` to `entry`, and write it back.  Never touches a
    cell other than the one just computed, so two single-cell processes racing on
    DIFFERENT cells cannot clobber each other's result — only the same cell run twice can,
    which `main`'s skip-if-present check already avoids.
    """
    if os.path.exists(out_path):
        with open(out_path) as fh:
            report = json.load(fh)
        if not (report.get("genome") == genome and report.get("n_phase") == n_phase
                and report.get("kinematics") == kinematics):
            report = _fresh_report(genome, n_phase, kinematics, ladder, pinned)
    else:
        report = _fresh_report(genome, n_phase, kinematics, ladder, pinned)

    rows_by_cfg = {r["config"]: r for r in report["rows"]}
    row = rows_by_cfg.setdefault(cfg_name, {"config": cfg_name, "meshes": {}})
    row["meshes"][label] = entry
    # Ordered by `ladder` first, then anything outside it (e.g. `--rung smoke`, a probe
    # of this path rather than a Condition A rung) appended rather than dropped — this
    # filter exists to ORDER the rows for `_print`, not to exclude a computed cell.
    order = [n for n in ladder if n in rows_by_cfg] + [n for n in rows_by_cfg
                                                        if n not in ladder]
    report["rows"] = [rows_by_cfg[name] for name in order]
    report["complete"] = False
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def cell_done(report, cfg_name, label):
    row = next((r for r in report.get("rows", ()) if r["config"] == cfg_name), None)
    if row is None or label not in row["meshes"]:
        return False
    return "error" not in row["meshes"][label]


def run_one_cell(out_path, genome, n_phase, kinematics, ladder, cfg_name, filleted):
    label = "filleted" if filleted else "unfilleted"
    genes = load_genes(genome)
    pinned = pinned_orientation(genes)

    if os.path.exists(out_path):
        with open(out_path) as fh:
            prior = json.load(fh)
        if (prior.get("genome") == genome and prior.get("n_phase") == n_phase
                and prior.get("kinematics") == kinematics
                and cell_done(prior, cfg_name, label)):
            print("  %-7s %-10s  (already computed, skipping)" % (cfg_name, label),
                 flush=True)
            return

    phases = WO.phase_stencil(n_phase=n_phase, scheme="uniform")
    try:
        entry = solve_cell(genes, cfg_name, filleted, phases, pinned, kinematics)
        entry["peak_rss_gb"] = round(peak_rss_gb(), 2)
    except Exception as exc:
        # A refused cell is RECORDED rather than left to crash the process with nothing
        # written — `study_m9.py`'s pattern (`study_m9.py:141-146`) for the same class of
        # never-before-run rung.  This cannot catch a cgroup memory kill (SIGKILL is not
        # catchable); it is here for the failures that ARE Python exceptions.
        entry = {"error": f"{type(exc).__name__}: {exc}",
                 "peak_rss_gb": round(peak_rss_gb(), 2)}
        print("  %-7s %-10s  FAILED: %s" % (cfg_name, label, entry["error"]), flush=True)

    merge_cell(out_path, genome, n_phase, kinematics, ladder, pinned, cfg_name, label,
              entry)


def spread_pct(values):
    """`100 * (max - min) / coarse` over the ladder — FILLET_PLAN STEP 1 PART 12's own
    definition, reproduced exactly so the two numbers are comparable.  `values` is
    ordered coarse -> fine; PART 12's table is monotone in every row it reports, where
    this collapses to `(fine - coarse) / coarse`, and stated as a max-min spread so a
    non-monotone ladder is not silently read as smaller than it is.
    """
    v = np.asarray(values, dtype=float)
    return float(100.0 * (v.max() - v.min()) / v[0])


def analyse(out):
    rows = out["rows"]
    result = {"spread_pct": {}, "richardson": {}, "finest": {}, "utilisation": {}}
    for label in MESH_TYPES:
        drops = [r["meshes"][label]["axle_drop_mean_mm"] for r in rows]
        hs = [r["meshes"][label]["h"] for r in rows]
        result["spread_pct"][label] = spread_pct(drops)
        result["richardson"][label] = SG.richardson(drops, hs)
        result["finest"][label] = {k: rows[-1]["meshes"][label][k] for k in TRACK_KEYS}
        result["utilisation"][label] = [r["meshes"][label]["stress_utilisation"]
                                        for r in rows]
    result["spread_ordering_reproduces_part_12"] = (
        result["spread_pct"]["filleted"] < result["spread_pct"]["unfilleted"])
    fin_u = result["finest"]["unfilleted"]["stress_utilisation"]
    fin_f = result["finest"]["filleted"]["stress_utilisation"]
    result["admissibility_at_finest"] = {
        "unfilleted_util": fin_u, "unfilleted_breached": fin_u > 1.0,
        "filleted_util": fin_f, "filleted_breached": fin_f > 1.0,
        "disagreement_survives_refinement": (fin_u > 1.0) and not (fin_f > 1.0)}
    return result


def _print(rep):
    print("\n" + "=" * 78)
    print("  CONDITION A — %s, %s, %d-phase, %s"
          % (rep["genome"], rep["kinematics"].upper(), rep["n_phase"],
             "/".join(rep["ladder"])))
    print("=" * 78)
    a = rep["analysis"]
    print("\n  AXLE DROP (mm), coarse -> fine, both meshes:")
    print("    %-11s" % "config" + "".join("%12s" % c for c in rep["ladder"])
          + "     spread%")
    for label in MESH_TYPES:
        vals = [r["meshes"][label]["axle_drop_mean_mm"] for r in rep["rows"]]
        print("    %-11s" % label + "".join("%12.6f" % v for v in vals)
              + "   %7.3f%%" % a["spread_pct"][label])
    print("\n  PART 12's ordering (filleted spread < unfilleted spread) at the SHIPPED "
          "genome, LINEAR, one phase: 0.141%% < 1.216%%.  Here, at %s, %s, %d phases:"
          % (rep["genome"], rep["kinematics"].upper(), rep["n_phase"]))
    print("    filleted %.3f%% < unfilleted %.3f%%  ->  %s"
          % (a["spread_pct"]["filleted"], a["spread_pct"]["unfilleted"],
             "REPRODUCES" if a["spread_ordering_reproduces_part_12"] else "DOES NOT HOLD"))

    print("\n  RICHARDSON on axle_drop_mean_mm, h = 1/sqrt(n_elements):")
    for label in MESH_TYPES:
        r = a["richardson"][label]
        if r["observed_order_p"] is None:
            print("    %-11s order did not converge — extrapolation not reported"
                  % label)
        else:
            print("    %-11s order %.2f   extrapolated %.6f mm   GCI(fine) %.3f%%"
                  % (label, r["observed_order_p"], r["extrapolated_mm"],
                     r["gci_fine_pct"]))

    print("\n  STRESS UTILISATION (util_j = kt * agg / ALLOWABLE), coarse -> fine:")
    print("    %-11s" % "config" + "".join("%12s" % c for c in rep["ladder"]))
    for label in MESH_TYPES:
        print("    %-11s" % label
              + "".join("%12.4f" % v for v in a["utilisation"][label]))
    fa = a["admissibility_at_finest"]
    print("\n  AT THE FINEST RUNG (%s): unfilleted %.4f (%s), filleted %.4f (%s)."
          % (rep["ladder"][-1], fa["unfilleted_util"],
             "BREACHED" if fa["unfilleted_breached"] else "clears",
             fa["filleted_util"],
             "BREACHED" if fa["filleted_breached"] else "clears"))
    print("  §94's coarse-only reading was 1.0112 (unfilleted, breached) / 0.7954 "
          "(filleted, clears).")
    print("  Admissibility disagreement survives refinement: %s"
          % ("YES" if fa["disagreement_survives_refinement"] else "NO"))

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    ok = a["spread_ordering_reproduces_part_12"] and fa["disagreement_survives_refinement"]
    print("  %s" % (
        "CONDITION A HOLDS: the filleted mesh's axle-drop spread is materially smaller "
        "and the admissibility disagreement survives to the finest rung. The filleted "
        "mesh is the converged one on this evidence."
        if ok else
        "CONDITION A DOES NOT HOLD AS STATED — see the two lines above. §93: \"this "
        "decision is reopened and nothing below happens.\""))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genome", default=GENOME)
    ap.add_argument("--kinematics", default=KINEMATICS)
    ap.add_argument("--n-phase", type=int, default=N_PHASE)
    ap.add_argument("--ladder", default=",".join(LADDER))
    ap.add_argument("--out", default="study_fillet_condition_a.json")
    ap.add_argument("--rung", choices=LADDER + ("smoke",),
                    help="compute ONE (rung, --mesh) cell in this process and merge it "
                         "into --out, rather than the whole ladder.  Each cell gets a "
                         "fresh interpreter and returns its memory to the OS on exit -- "
                         "see the module docstring above run_one_cell for why.  `smoke` "
                         "is accepted here (only) as a cheap end-to-end check of this "
                         "path; it is not part of Condition A's own ladder.")
    ap.add_argument("--mesh", choices=MESH_TYPES, help="required with --rung")
    ap.add_argument("--finalise", action="store_true",
                    help="run the analysis/verdict on --out's already-computed cells and "
                         "mark it complete, without computing anything.  Refuses if any "
                         "cell in --ladder is missing or errored.")
    args = ap.parse_args()

    if args.rung and not args.mesh:
        ap.error("--rung requires --mesh")
    if args.finalise and args.rung:
        ap.error("--finalise and --rung are two different steps; pass one at a time")

    ladder = tuple(s.strip() for s in args.ladder.split(",") if s.strip())
    out_path = os.path.join(HERE, args.out)

    if args.rung:
        # `--rung smoke` is a probe of THIS path, not a cell of Condition A's own ladder
        # (coarse/medium/fine only) -- it may never land in the gate's own artifact, so
        # this refuses on the filename directly rather than trusting `--out` to be typed
        # correctly, the same posture `refuse_degraded_out` takes for everything else.
        if args.rung == "smoke" and args.out == "study_fillet_condition_a.json":
            ap.error("--rung smoke is a probe of the per-cell path, not a Condition A "
                     "rung -- pass an explicit --out (e.g. --out smoke_probe.json)")
        if args.rung != "smoke":
            _gate_guard.refuse_degraded_out(ap, args, "study_fillet_condition_a.json", [
                (args.genome != GENOME,
                 "--genome %s, not §93's named %s" % (args.genome, GENOME)),
                (args.kinematics != KINEMATICS,
                 "--kinematics %s — Condition A is stated under svk" % args.kinematics),
                (args.n_phase != N_PHASE,
                 "--n-phase %d, not the committed %d" % (args.n_phase, N_PHASE)),
            ])
        run_one_cell(out_path, args.genome, args.n_phase, args.kinematics, ladder,
                    args.rung, args.mesh == "filleted")
        return 0

    if args.finalise:
        _gate_guard.refuse_degraded_out(ap, args, "study_fillet_condition_a.json", [
            (args.genome != GENOME, "--genome %s, not §93's named %s" % (args.genome,
                                                                         GENOME)),
            (args.kinematics != KINEMATICS,
             "--kinematics %s — Condition A is stated under svk" % args.kinematics),
            (args.n_phase != N_PHASE, "--n-phase %d, not the committed %d" % (args.n_phase,
                                                                              N_PHASE)),
            (not {"coarse", "medium", "fine"} <= set(ladder),
             "--ladder %s drops a rung Condition A names" % args.ladder),
        ])
        if not os.path.exists(out_path):
            ap.error("--out %s does not exist yet -- run --rung for each cell first"
                     % args.out)
        with open(out_path) as fh:
            rep = json.load(fh)
        missing = [(name, label) for name in ladder for label in MESH_TYPES
                  if not cell_done(rep, name, label)]
        if missing:
            ap.error("not every cell is computed yet: %s"
                     % ", ".join("%s/%s" % (n, l) for n, l in missing))
        rep["analysis"] = analyse(rep)
        rep["complete"] = True
        _print(rep)
        with open(out_path, "w") as fh:
            json.dump(rep, fh, indent=2)
        print("\n    wrote %s (finalised)" % args.out)
        return 0

    _gate_guard.refuse_degraded_out(ap, args, "study_fillet_condition_a.json", [
        (args.genome != GENOME, "--genome %s, not §93's named %s" % (args.genome, GENOME)),
        (args.kinematics != KINEMATICS,
         "--kinematics %s — Condition A is stated under svk" % args.kinematics),
        (args.n_phase != N_PHASE, "--n-phase %d, not the committed %d" % (args.n_phase,
                                                                          N_PHASE)),
        (not {"coarse", "medium", "fine"} <= set(ladder),
         "--ladder %s drops a rung Condition A names" % args.ladder),
    ])

    resume_rows = ()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            prior = json.load(fh)
        if (prior.get("genome") == args.genome and prior.get("n_phase") == args.n_phase
                and prior.get("kinematics") == args.kinematics):
            resume_rows = prior.get("rows", ())
            if resume_rows:
                print("resuming from checkpoint: %d rung(s) already computed (%s)"
                      % (len(resume_rows), ", ".join(r["config"] for r in resume_rows)))

    t0 = time.time()
    rep = run_ladder(args.genome, ladder, args.n_phase, args.kinematics,
                     resume_rows=resume_rows, checkpoint_path=out_path)
    rep["analysis"] = analyse(rep)
    rep["complete"] = True
    rep["seconds"] = time.time() - t0
    _print(rep)

    with open(out_path, "w") as fh:
        json.dump(rep, fh, indent=2)
    print("\n    wrote %s  (%.1f s)" % (args.out, rep["seconds"]))

    # NO PASS/FAIL AND EXIT 0.  Condition A failing is a live, meaningful outcome
    # §93 already named ("this decision is reopened") — a driver that exited nonzero on
    # it would be asserting which answer the decision wants to hear.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
