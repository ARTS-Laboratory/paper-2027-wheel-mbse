"""BOUNDARY_PLAN.md Step 0 — price defect 5 as wasted descent, not as loss share.

    .venv-opt/bin/python studies/study_boundary_waste.py      (make boundarywaste)

§21 priced `soft_barrier`'s zero-gradient-at-the-knee defect at 0.61% of loss. Step 0 asks a
different question: over the Stage-3 runs already on disk, how much of each run's wall-clock
is spent *past* the last iterate where every barrier held, versus pushed further into
violation by a term with no slope to hold it there? §19 measured one run at 55/100 iterates
and 2.8 of 6h20m. This driver is the census that says whether that ratio generalises.

**IT READS TWENTY-FIVE COMMITTED ARTIFACTS AND SOLVES NOTHING.** Every `stage3_*.json` with
a `steps` trace already carries `steps[i]["terms"][name]["value"]` for each term and
`steps[i]["wall_s"]`, so the answer is arithmetic on what is already on disk — the same
reason `study_fillet_wiring.py` reads seven committed artifacts instead of adding an eighth
instrument to a comparison that has to be made on the ones that produced the claims.

BARRIER_TERMS IS HARD-CODED, NOT IMPORTED. `wheel_objective` pulls in jax to define it
(`wheel_objective.py:118-120`), and nothing else this driver does needs jax. The nine names
below are copied from `wheel_objective.BARRIER_TERMS` (`wheel_objective.py:399-400`) at
2026-09-04; if that tuple ever changes, it will not change silently here — every step's
`terms` dict is read in full regardless, and `all_term_keys_observed` in the output reports
every key any run actually carries, so a name added or dropped there shows up as a diff
against the list below without this driver needing to run jax to find it.

TWO FEASIBILITY QUESTIONS, KEPT SEPARATE ON PURPOSE. "Every barrier holds" is the general
question BOUNDARY Step 0 asks. `fillet_cap` gets its own column because §19's own run — the
one the arc's whole cost estimate rests on — was carried by exactly that term ("wasting 45
steps of a 6h20m run", BOUNDARY_PLAN.md). Two of the twenty-five runs (`stage3_run_elite9`,
`stage3_run_elite10`) predate `fillet_cap` existing at all: their `terms` dicts have no such
key. Silently reading a missing key as 0.0 would call every one of their steps feasible on a
term that was never evaluated — the same "a sentinel with two meanings" trap this tree has
been caught by before. So a run missing a barrier key gets `null` in the per-term index, not
a manufactured feasible run, and is named in `runs_with_partial_schema`.

SCOPE, STATED RATHER THAN DISCOVERED BY A READER LATER: every one of the twenty-five
`steps`-bearing artifacts was produced 2026-08-01/03, before §103's fillet switch, and none
carries a `fillet` key in `settings`. This driver checks that claim against every run's
`settings` rather than asserting it, and reports the check. **What it prices is defect 5
against the objective §103 replaced.** Whether the same ratio holds on the objective the
optimizer now runs is a `fillet=True` Stage-3 descent's question, not this driver's — see
PLAN.md §105 for why that descent is not cheap.
"""

import argparse
import glob
import json
import os

import project_paths as PP  # noqa: F401  (puts src/ on the path; stdlib-only, see its own
                             # docstring — safe to import without paying for jax)

HERE = os.path.dirname(os.path.abspath(__file__))

BARRIER_TERMS = ("stress", "buckling", "x_order", "hub_overlap", "fold",
                 "arrival", "fillet", "fillet_cap", "min_sj")


def _run_record(path):
    with open(path) as fh:
        doc = json.load(fh)
    steps = doc.get("steps")
    if not isinstance(steps, list) or not steps or "terms" not in steps[0]:
        return None  # a "*_best*"/"*_check*" summary artifact — no step trace to census

    present = tuple(k for k in BARRIER_TERMS if k in steps[0]["terms"])
    missing = tuple(k for k in BARRIER_TERMS if k not in present)
    wall = [float(s.get("wall_s") or 0.0) for s in steps]
    n = len(steps)

    def _last_feasible(keys):
        if not keys:
            return None  # nothing to judge feasibility on at all
        last = -1
        for i, s in enumerate(steps):
            t = s["terms"]
            if all(float(t[k]["value"]) == 0.0 for k in keys):
                last = i
        return last

    last_all = _last_feasible(present)
    last_cap = _last_feasible(("fillet_cap",)) if "fillet_cap" in present else None

    settings = doc.get("settings") or {}
    all_keys = set()
    for s in steps:
        all_keys |= set(s["terms"].keys())

    return {
        "config": settings.get("config"),
        "kinematics": settings.get("kinematics"),
        "carries_fillet_setting": "fillet" in settings,
        "n_steps": n,
        "barrier_keys_present": list(present),
        "barrier_keys_missing": list(missing),
        "total_wall_s": sum(wall),
        "last_all_feasible_index": last_all,
        "wall_s_past_all_feasible": sum(wall[last_all + 1:]),
        "last_fillet_cap_feasible_index": last_cap,
        "wall_s_past_fillet_cap_feasible": (
            None if last_cap is None else sum(wall[last_cap + 1:])),
        "all_term_keys_observed": sorted(all_keys),
    }


def build():
    excluded = []
    runs = {}
    for path in sorted(glob.glob(os.path.join(PP.ROOT, "stage3_*.json"))):
        name = os.path.basename(path)
        rec = _run_record(path)
        if rec is None:
            excluded.append(name)
        else:
            runs[name] = rec

    total_wall_s = sum(r["total_wall_s"] for r in runs.values())
    past_all_s = sum(r["wall_s_past_all_feasible"] for r in runs.values())
    cap_rows = [r for r in runs.values() if r["last_fillet_cap_feasible_index"] is not None]
    past_cap_s = sum(r["wall_s_past_fillet_cap_feasible"] for r in cap_rows)
    cap_wall_s = sum(r["total_wall_s"] for r in cap_rows)
    partial = sorted(n for n, r in runs.items() if r["barrier_keys_missing"])
    bites = sorted(n for n, r in runs.items() if r["wall_s_past_all_feasible"] > 0.0)
    pre_switch = all(not r["carries_fillet_setting"] for r in runs.values())

    return {
        "barrier_terms_used": list(BARRIER_TERMS),
        "runs": runs,
        "excluded_no_step_trace": excluded,
        "runs_with_partial_schema": partial,
        "runs_where_defect_bites": bites,
        "all_pre_fillet_switch": pre_switch,
        "totals": {
            "n_runs": len(runs),
            "n_excluded": len(excluded),
            "total_wall_h": total_wall_s / 3600.0,
            "wall_past_all_feasible_h": past_all_s / 3600.0,
            "pct_past_all_feasible": (100.0 * past_all_s / total_wall_s
                                       if total_wall_s else 0.0),
            "n_runs_with_fillet_cap": len(cap_rows),
            "wall_past_fillet_cap_feasible_h": past_cap_s / 3600.0,
            "pct_past_fillet_cap_feasible_of_cap_runs": (
                100.0 * past_cap_s / cap_wall_s if cap_wall_s else 0.0),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="study_boundary_waste.json")
    args = ap.parse_args()

    rep = build()
    t = rep["totals"]
    print(f"runs: {t['n_runs']} included, {t['n_excluded']} excluded (no step trace)")
    print(f"all pre-fillet-switch (no 'fillet' in settings): {rep['all_pre_fillet_switch']}")
    print(f"total wall: {t['total_wall_h']:.2f} h")
    print(f"  past last all-barriers-feasible iterate: {t['wall_past_all_feasible_h']:.2f} h "
          f"({t['pct_past_all_feasible']:.1f}%)")
    print(f"  past last fillet_cap-feasible iterate:   "
          f"{t['wall_past_fillet_cap_feasible_h']:.2f} h "
          f"({t['pct_past_fillet_cap_feasible_of_cap_runs']:.1f}% of the "
          f"{t['n_runs_with_fillet_cap']} runs that carry the term)")
    print(f"runs with a partial barrier schema: {rep['runs_with_partial_schema']}")
    print(f"runs where the defect bites at all: {rep['runs_where_defect_bites']}")

    out = os.path.join(HERE, args.out)
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
