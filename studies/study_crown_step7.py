#!/usr/bin/env python3
"""CROWN_PLAN.md Step 7's read, as R7 registered it before the descent launched.

Reads `stage3_crown_standin.json` (the Step 7 descent) and reports, at its tier-0 selected
genome:
  1. every gene's move from `b729e86`, in millimetres and in normalised units, and which
     genes end on a bound;
  2. the spoke profile overlaid on `b729e86`'s (`.jpg`);
  3. R7's attribution: each live term's gradient, from one `Evaluator` call per term with
     every other weight zeroed (the band term rides `band`, so it is isolated by zeroing
     them all), restricted to the FREE genes and projected on deflection's.  R7's rule:
     the band "sets the drop" if its projection cancels at least half of deflection's.
     Done twice, because `t2` ends 0.002 mm off its 1.2 mm floor: once with `t2` free
     (strictly off the bound) and once with it counted as pinned.
The isolated gradients must sum to the run's own recorded gradient at the selected step;
the difference is printed, not asserted.

Run alone on the box, after the descent, under `wheel_pool.PINNED_ENV` (the Makefile's
five variables) — ~15 min, three `coarse` SVK eight-phase calls serial.
"""
import json
import sys

import numpy as np

import wheel_fea as W
import wheel_genome as wg
import wheel_geometry as G
import wheel_objective as WO
import wheel_stage3 as S3

RUN = sys.argv[1] if len(sys.argv) > 1 else "stage3_crown_standin.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "studies/study_crown_step7"

rec = json.load(open(RUN))
st = rec["settings"]
W.set_min_wall(st["min_wall_mm"])
low, high, rng = wg.bounds_arrays(W.GENE_SPACE)
names = list(json.load(open("best_solution.json"))["genes"])
assert len(names) == len(W.GENE_SPACE)
row = {r["step"]: r for r in rec["steps"]}
z0 = np.asarray(rec["steps"][0]["z"])
best = rec["best"]
zb = np.asarray(best["z"])
g0, gb = wg.denormalize(z0, low, high), wg.denormalize(zb, low, high)
sel = best["step"]
last = rec["steps"][-1]

out = {"run": RUN, "selected_step": sel, "last_step": last["step"],
       "selected_loss": best["loss"], "last_loss": last["loss"],
       "selected_report": best["report"], "last_report": last["report"],
       "genes": {}}
print(f"selected step {sel} (loss {best['loss']:.4f}); last step {last['step']} "
      f"(loss {last['loss']:.4f})")
print(f"{'gene':7s} {'start':>9s} {'end':>9s} {'move':>9s} {'dz':>8s}  bound")
for k, n in enumerate(names):
    at = "LOW" if zb[k] <= 1e-12 else ("HIGH" if zb[k] >= 1 - 1e-12 else "")
    out["genes"][n] = {"start": g0[k], "end": gb[k], "move": gb[k] - g0[k],
                       "dz": zb[k] - z0[k], "bound": at}
    print(f"{n:7s} {g0[k]:9.4f} {gb[k]:9.4f} {gb[k]-g0[k]:+9.4f} {zb[k]-z0[k]:+8.4f}  {at}")

# ---- the overlay -------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def poly(g):
    d = dict(zip(names, g))
    curve, ctrl = G.bezier_centerline(*(d[f"c{a}{i}"] for i in range(1, 5) for a in "xy"),
                                      W.S, 200)
    p = G.outline_polygon(curve, d["t0"], d["t1"], d["t2"], d["t3"])
    return G.place_sector(p, W.HUB_RADIUS_MM), G.place_sector(curve, W.HUB_RADIUS_MM)

fig, ax = plt.subplots(figsize=(10, 4.5))
for g, lab, col in ((g0, "b729e86 (start)", "0.55"), (gb, f"step 7, step {sel}", "C3")):
    p, c = poly(g)
    ax.fill(p[:, 0], p[:, 1], color=col, alpha=0.25, lw=0)
    ax.plot(np.r_[p[:, 0], p[:1, 0]], np.r_[p[:, 1], p[:1, 1]], color=col, lw=1.2, label=lab)
    ax.plot(c[:, 0], c[:, 1], color=col, lw=0.6, ls="--")
ax.axvline(W.HUB_RADIUS_MM, color="k", lw=0.5)
ax.axvline(W.RIM_RADIUS_MM, color="k", lw=0.5)
ax.set_aspect("equal")
ax.set_xlabel("x, mm (hub face at 12.7, rim inner face at 48.5)")
ax.set_ylabel("y, mm")
ax.legend(loc="upper left", fontsize=8)
ax.set_title("CROWN_PLAN Step 7: one spoke, start vs selected genome")
fig.tight_layout()
fig.savefig(OUT + ".jpg", dpi=130)

# ---- free-gene attribution (R7) ---------------------------------------------------
# One Evaluator call per term, weights zeroed but for that term; the band term rides
# `band` and has no DEFAULT_WEIGHTS row, so it is isolated by zeroing every weight.
standin = {"rim_outer": st["rim_outer_mm"], "drop_factor": st["drop_factor"],
           "band": st["band"]}
phases = list(row[sel]["phase_deg"])
orient = tuple(st["orientation"])
full = row[sel]["terms"]
live = [k for k, v in full.items() if v["value"] != 0.0 or v["grad_norm"] != 0.0]
t3 = {"deflection", "stress", "stress_margin", "phase_ripple", "band_margin"}
zeros = {k: 0.0 for k in WO.DEFAULT_WEIGHTS}
grads, vals = {}, {}
for k in live:
    w = dict(zeros)
    kw = dict(standin)
    if k == "band_margin":
        pass
    else:
        w[k] = WO.DEFAULT_WEIGHTS[k]
        kw["band"] = None
    tiers = ("t3",) if k in t3 else ("t1", "t2")
    ev = S3.Evaluator(st["config"], weights=w, orientation=orient,
                      kinematics=st["kinematics"], **kw)
    v, g, brk = ev(zb, low, high, phases=phases, tiers=tiers)
    grads[k], vals[k] = np.asarray(g), v
    print(f"  {k:14s} value {v:12.6f} (run {full[k]['value']:12.6f})  |g| "
          f"{np.linalg.norm(g):10.4g} (run {full[k]['grad_norm']:10.4g})", flush=True)

gsum = np.sum(list(grads.values()), axis=0)
grun = np.asarray(row[sel]["grad"])
print("sum of isolated grads vs the run's recorded grad at the selected step: "
      f"max |diff| {np.max(np.abs(gsum - grun)):.3e}, |run| {np.linalg.norm(grun):.4g}")

d = grads["deflection"]
att = {}
for label, pinned in (("t2_free", {n for n, o in out["genes"].items() if o["bound"]}),
                      ("t2_pinned", {n for n, o in out["genes"].items() if o["bound"]}
                       | {"t2"})):
    free = np.array([n not in pinned for n in names])
    df = np.where(free, d, 0.0)
    nf2 = float(df @ df) or 1.0
    print(f"\n[{label}] pinned {sorted(pinned)}; |deflection grad|: all "
          f"{np.linalg.norm(d):.4g}, free {np.linalg.norm(df):.4g}")
    att[label] = {"pinned": sorted(pinned), "terms": {}}
    for k, g in grads.items():
        gf = np.where(free, g, 0.0)
        a = {"proj_on_defl_free": float(gf @ df) / nf2,
             "cos_free": float(gf @ df) / (np.linalg.norm(gf) * np.linalg.norm(df) or 1.0),
             "norm_all": float(np.linalg.norm(g)), "norm_free": float(np.linalg.norm(gf))}
        att[label]["terms"][k] = a
        print(f"  {k:14s} proj on defl(free) {a['proj_on_defl_free']:+9.4f}  "
              f"cos {a['cos_free']:+.3f}  |g| all {a['norm_all']:10.4g} "
              f"free {a['norm_free']:10.4g}")
out["term_grads"] = {k: g.tolist() for k, g in grads.items()}
out["attribution"] = att
out["gsum_vs_run_maxdiff"] = float(np.max(np.abs(gsum - grun)))
json.dump(out, open(OUT + ".json", "w"), indent=1)
print("wrote", OUT + ".json", OUT + ".jpg")
