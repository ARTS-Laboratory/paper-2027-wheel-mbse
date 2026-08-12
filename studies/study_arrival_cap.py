"""BUILD_PLAN.md step 2 — does the per-corner OCC threshold depend on the HUB ARRIVAL ANGLE?

IN `studies/` RATHER THAN IN A SCRATCHPAD, and the distinction is the one PLAN.md §0 draws.
`eps_n_check.py` and `h2_check.py` were one-question falsifications whose answer was a table
and whose artifact nothing depended on, so they stayed in a scratchpad.  This one started
that way and is not that: **a constant fitted to these rows SHIPS**, which puts it in
`study_hub_cap.json`'s category — calibration evidence behind a number in `wheel_objective`,
and it has to stay reproducible for the same reason.  §0 says as much about `h2_check.py`
itself: "Promoting it into a real study driver is the FIRST piece of phase-3 work."

Deliberately OUT of `make studies`, for `study_hub_cap.py`'s two reasons: it needs BOTH
interpreters, and what it measures is OCC's behaviour on this shape rather than anything a
commit changed.

NOT A GATE.  It calibrates; gating a section on the constant it is measuring is circular,
which is the argument `run_t0_sweep` already makes for itself ("Reported, not gated: this
section EXISTS to calibrate").

THE CONTROL VARIABLE, and it is exact rather than approximate.  `control_points` locks
P0 = (0,0) at the hub, so the first control-polygon edge is exactly `(cx1, cy1)` and
`arrival_penalty`'s hub angle is `asin(|cx1| / hypot(cx1, cy1))` — a function of TWO genes
and nothing else.  Rotating P1 about the hub at FIXED RADIUS therefore walks the arrival
angle while moving no other gene at all.  That is the sweep `t0_sweep` is for the thickness
branch, and it is what the n=2 comparison in step 1 could not be.

TWO BASES, because step 1's two thickness groups disagreed on the SIGN.  If the flip is real
it will show up here as two curves of opposite slope; if it was an artefact of five designs
that differ in everything, both curves will agree.
"""
import json
import math
import os
import sys

import numpy as np

ROOT = "/home/eric-bodhi/github/wheel"
sys.path.insert(0, os.path.join(ROOT, "studies"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import study_hub_cap as H          # noqa: E402  reuse the bisection rig verbatim
import wheel_genome as wg          # noqa: E402
import wheel_wheel as WW           # noqa: E402
import project_paths as PP         # noqa: E402

CFG = WW.get_config("coarse")
STATIONS_DEG = (5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)


def _elite(rank):
    for row in json.load(open(PP.STAGE2_ELITES))["elites"]:
        if int(row["rank"]) == rank:
            return wg.genes_to_vector(row["genes"])
    raise KeyError(rank)


def set_hub_arrival(base, deg):
    """Rotate P1 about the hub to put the hub arrival angle at `deg`.  Nothing else moves."""
    g = np.asarray(base, dtype=float).copy()
    r = math.hypot(g[0], g[1])
    g[0] = r * math.sin(math.radians(deg))
    g[1] = r * math.cos(math.radians(deg))
    return g


def measure(label, base):
    designs = []
    for d in STATIONS_DEG:
        designs.append((f"{label}_a{d:g}", set_hub_arrival(base, d)))
    caps = {n: H.analytic_cap(g)[1] for n, g in designs}
    cad = H._run_cad(designs, caps, bisect=True)

    print(f"\n=== {label}   t0 = {float(base[8]):.3f} mm,  |P1| = "
          f"{math.hypot(base[0], base[1]):.4f} mm ===")
    print(f"{'target':>7s} {'arrival':>8s} {'void':>7s} {'wedgeLO':>8s} {'wedgeHI':>8s} "
          f"{'thrLO':>8s} {'thrHI':>8s} {'restrict':>9s} {'restr/t0':>9s}")
    rows = []
    for (name, g), c in zip(designs, cad):
        target = float(name.split("_a")[1])
        a_hub, _a_rim = WW.arrival_angles(g, CFG, xp=np)
        void, cap, _r = H.analytic_cap(g)
        if "thresholds_mm" not in c:
            print(f"{target:7.1f} {float(a_hub):8.3f}   ERROR {c.get('error')}")
            continue
        pairs = sorted(zip(c["wedges_deg"], c["thresholds_mm"]))
        half = len(pairs) // 2
        lo_w = float(np.mean([w for w, _t in pairs[:half]]))
        hi_w = float(np.mean([w for w, _t in pairs[half:]]))
        lo_t = float(np.mean([t for _w, t in pairs[:half]]))
        hi_t = float(np.mean([t for _w, t in pairs[half:]]))
        restrict = min(lo_t, hi_t)
        t0 = float(g[8])
        bracket = c["bisect_hi_mm"]
        flag = "  <- restrictive family AT BRACKET (censored)" if restrict >= bracket - 1e-9 else ""
        print(f"{target:7.1f} {float(a_hub):8.3f} {void:7.3f} {lo_w:8.2f} {hi_w:8.2f} "
              f"{lo_t:8.4f} {hi_t:8.4f} {restrict:9.4f} {restrict / t0:9.4f}{flag}")
        rows.append({"target_deg": target, "arrival_deg": float(a_hub), "void_deg": void,
                     "wedge_lo": lo_w, "wedge_hi": hi_w, "thr_lo": lo_t, "thr_hi": hi_t,
                     "restrictive_mm": restrict, "restrictive_over_t0": restrict / t0,
                     "which": "lo_wedge" if lo_t <= hi_t else "hi_wedge",
                     "cap_mm": cap, "at_bracket": bool(restrict >= bracket - 1e-9)})
    return rows


if __name__ == "__main__":
    out = {}
    shipped = wg.genes_to_vector(json.load(open(os.path.join(ROOT, "best_solution.json")))["genes"])
    out["350f4c7_t0_1.2"] = measure("350f4c7_t0_1.2", shipped)
    out["elite13_t0_2.55"] = measure("elite13_t0_2.55", _elite(13))

    print("\n\n==== MONOTONICITY, the question this script exists to answer ====")
    for k, rows in out.items():
        good = [r for r in rows if not r["at_bracket"]]
        if len(good) < 3:
            print(f"{k:22s} too few uncensored rows ({len(good)})")
            continue
        xs = [r["arrival_deg"] for r in good]
        ys = [r["restrictive_mm"] for r in good]
        d = np.diff(ys)
        sign = ("DECREASING" if all(v < 0 for v in d) else
                "INCREASING" if all(v > 0 for v in d) else "NON-MONOTONE")
        rho = float(np.corrcoef(xs, ys)[0, 1])
        print(f"{k:22s} {sign:14s}  pearson {rho:+.4f}  "
              f"range {min(ys):.4f}..{max(ys):.4f} mm over "
              f"arrival {min(xs):.1f}..{max(xs):.1f} deg  (n={len(good)})")
        print(f"{'':22s} which family binds: "
              f"{sorted({r['which'] for r in good})}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "study_arrival_cap.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote arrival_sweep.json")
