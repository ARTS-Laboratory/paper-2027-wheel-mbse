"""Does the shipped genome's hub utilisation converge, and does it cross the knee?

PLAN.md §38, and the 2026-08-20 ruling that came out of it.  `stress_margin` is a
`soft_barrier` with its knee at `MARGIN_KNEE_UTIL`, so whether the term does any work at
all turns on which side of 0.80 `util_hub` lands on — and the shipped genome sits near it
either way.  One rung cannot answer that; the LADDER can, and what it says is not "above"
or "below" but "the sequence does not settle".

Measured 2026-08-19/20 on the faithful mesh under SVK at 8 uniform phases:

    rung     dofs    util_hub   increment   ratio    headroom
    smoke     22k    0.75490                          +5.64%
    coarse    42k    0.77876     0.02386             +2.65%
    medium   106k    0.78519     0.00643    0.269    +1.85%
    fine     265k    0.78979     0.00460    0.716    +1.28%
    ultra    587k    0.79347     0.00367    0.799    +0.82%

THE RATIO CLIMBS TOWARD 1 INSTEAD OF DECAYING, so every extrapolation lands ABOVE the knee
(0.8080 at the last ratio, 0.8027 at the previous) while every measurement lands below it.
That is the whole finding.  The drift is in `agg`, the field p-norm, NOT in the geometry:
`kt_hub` moves -0.0197% across the same five rungs, first order and converged, and in the
OPPOSITE direction — which this driver also prints, because the ruling depends on it.

`ultra` is study_reds_hub_share.py's ULTRA and is deliberately NOT in `WW.CONFIGS`: a fifth
name there would put it in reach of every study and test that iterates the ladder, and
`wheel_mesh` has its own same-named configs.  It costs 4878 s at 14.8 GB.

NOT ON `make studies`.  Like `svk`, `m8bi5`, `m9buck` and `hubcap`, this measures THE WHEEL,
NOT THE COMMIT, and its top rung costs more than the whole recipe.
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, HERE)
import jax_config  # noqa
import wheel_genome as wg, wheel_objective as WO, wheel_wheel as WW

ULTRA = WW.WheelConfig("ultra", 384, 10, 40, 8, 40, 7, 40, n_curve=9600)


def _config(name):
    return ULTRA if name == "ultra" else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", default="best_solution.json")
    ap.add_argument("--rungs", default="smoke,coarse",
                    help="comma-separated; `ultra` is 4878 s at 14.8 GB, `fine` 2596 s")
    ap.add_argument("--kinematics", default="svk", choices=("linear", "svk"))
    ap.add_argument("--phases", type=int, default=8)
    ap.add_argument("--out", default="study_knee_rungs.json")
    a = ap.parse_args()

    g = wg.genes_to_vector(json.load(open(os.path.join(REPO, a.genome)))["genes"])
    ph = WO.phase_stencil(n_phase=a.phases, scheme="uniform")
    rungs = [r.strip() for r in a.rungs.split(",") if r.strip()]

    # `kt` first: it is cheap, it is geometry only, and it is the half of `util = kt * agg`
    # that has to be ruled out before the drift can be attributed to the field.
    kt = {}
    print(f"{'rung':<8} {'kt_hub':>12} {'kt_rim':>12}")
    for r in rungs:
        cfg = WW.get_config(_config(r))
        fl = WO.fillet_flanks(g, cfg, WW.HUB_RIM_SPAN_MM)
        (kh, _), (kr, _) = WO.junction_kt(g, cfg, span_mm=WW.HUB_RIM_SPAN_MM, flanks=fl)
        kt[r] = {"kt_hub": float(kh), "kt_rim": float(kr)}
        print(f"{r:<8} {float(kh):12.8f} {float(kr):12.8f}")
    sys.stdout.flush()

    print(f"\nknee = {WO.MARGIN_KNEE_UTIL}   kinematics = {a.kinematics}   "
          f"phases = {a.phases} uniform")
    print(f"{'rung':<8} {'util_hub':>9} {'util_rim':>9} {'margin':>10} {'headroom %':>11} "
          f"{'drop_mm':>9} {'s':>8}")
    rows, prev = {}, None
    for r in rungs:
        t0 = time.time()
        val, grad, brk = WO.objective(g, _config(r), phases=ph, kinematics=a.kinematics)
        t, rep = brk["terms"], brk["report"]
        u = float(rep["stress_utilisation_hub"])
        row = dict(kt[r], rung=r, kinematics=a.kinematics, util_hub=u,
                   util_rim=float(rep["stress_utilisation_rim"]),
                   margin=float(t["stress_margin"]["value"]),
                   headroom_pct=(WO.MARGIN_KNEE_UTIL - u) / WO.MARGIN_KNEE_UTIL * 100.0,
                   drop_mm=float(rep["axle_drop_mean_mm"]), loss=float(val),
                   crosses_knee=bool(u >= WO.MARGIN_KNEE_UTIL),
                   seconds=round(time.time() - t0, 1))
        if prev is not None:
            row["increment"] = u - prev
        prev = u
        rows[r] = row
        print(f"{r:<8} {u:9.5f} {row['util_rim']:9.5f} {row['margin']:10.6f} "
              f"{row['headroom_pct']:+11.2f} {row['drop_mm']:9.5f} {row['seconds']:8.1f}")
        sys.stdout.flush()
        json.dump({"knee": WO.MARGIN_KNEE_UTIL, "genome": a.genome, "rows": rows},
                  open(os.path.join(HERE, a.out), "w"), indent=1)

    incs = [rows[r]["increment"] for r in rungs if "increment" in rows[r]]
    if len(incs) >= 2:
        ratios = [incs[i] / incs[i - 1] for i in range(1, len(incs))]
        print(f"\nincrement ratios {['%.3f' % x for x in ratios]}")
        r_last = ratios[-1]
        if 0.0 < r_last < 1.0:
            lim = prev + incs[-1] * r_last / (1.0 - r_last)
            print(f"geometric limit at the last ratio: {lim:.5f}  "
                  f"({'ABOVE' if lim >= WO.MARGIN_KNEE_UTIL else 'below'} the knee)")
            print("  ...and a ratio that RISES toward 1 means this understates the limit.")
    print(f"\nwrote {os.path.join(HERE, a.out)}")


if __name__ == "__main__":
    main()
