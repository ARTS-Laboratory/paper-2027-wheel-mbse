#!/usr/bin/env python3
"""CROWN_PLAN.md Step 0.5: the interlayer coupons, as a printable part and a registered rule.

Step 8's last gate is `s_zz`, the tension that pulls the printed layers apart, and it has no
allowable in this tree: 0.3's 12.5 MPa is a placeholder, half of `wheel_fea`'s in-plane 25.
Decision 0.5 replaces it with a measurement made on the user's printer. The ratio of the
across-the-layers strength to the along-the-layers strength, from coupons printed and pulled
the same way, scales the in-plane allowable:

    allowable_zz = ALLOWABLE_STRESS_MPA * mean(sigma_Z) / mean(sigma_XY)

Pulling both sets on the same rig cancels a shared scale error, which is why decision 0.5
takes the ratio and not sigma_Z alone.

ONE GEOMETRY, TWO ORIENTATIONS. The XY set is printed lying flat and pulled along the
extrusions, which is how the band's hoop tension runs. The Z set is the SAME file stood on
its end and pulled across the layers, which is how `s_zz` runs. Only the direction of the
pull relative to the layers differs, so the ratio carries nothing but anisotropy.

  gauge 3.0 x 3.0 mm (9 mm^2), 10 mm straight, R 12.7 transitions (ASTM D638 Type V's), to
  20 mm grips, 3.0 mm thick throughout, a 4.4 mm pin hole (M4 clearance) 8 mm from each
  end. The gauge is small so a 50 kg hand-held scale can break the in-plane control: 40 MPa
  x 9 mm^2 = 360 N = 36.7 kgf. The grips are wide so a Z coupon breaks in its gauge, not
  at a hole: the net section beside the hole carries 0.19x the gauge stress, so a pin-loaded
  hole with Kt up to ~4 still stays under it.

    .venv-cad/bin/python studies/study_coupons.py
        -> export/coupon_step05.step, export/coupon_step05.stl, studies/study_coupons.json
    .venv-cad/bin/python studies/study_coupons.py --score coupons.csv

The CSV has one row per coupon: `orientation,width_mm,thickness_mm,peak_kgf,break`, with
orientation `xy` or `z`, both gauge dimensions measured by caliper, and `break` given as
`gauge`, `transition` or `grip`. Only gauge breaks are counted, which is standard practice.
The rule `--score` applies is CROWN_PLAN R19's, registered before any coupon was printed:

  R = mean(sigma_Z) / mean(sigma_XY), SE(R) by first-order propagation of both means' SEs.
  PASS       R - 2 SE >= s_zz / ALLOWABLE      (the gate is decided outside the scatter)
  FAIL       R + 2 SE <  s_zz / ALLOWABLE
  UNDECIDED  otherwise; print five more of each orientation and pool them.

It is read for `be96531`, the candidate (s_zz 17.12 MPa, R17), and for `b729e86`, the
shipped part (16.78, R1). With both on one line, a ratio between the two thresholds fails
only the candidate, and a ratio under both fails the part already shipping.
"""
import argparse, csv, json, math, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import wheel_fea as WF   # numpy only, importable in env-cad (tests/test_import_hygiene.py)

GAUGE_W, GAUGE_L, THICK = 3.0, 10.0, 3.0
GRIP_W, GRIP_L = 20.0, 16.0
ARC_R = 12.7                        # ASTM D638 Type V's transition radius
HOLE_D, HOLE_FROM_END = 4.4, 8.0    # M4 clearance, printed holes close up ~0.1-0.2 mm
KGF_N = 9.80665

# max s_zz tension in the band over eight phases, SVK, h 2.0 / hc 0.25, one mesh per phase
S_ZZ_MPA = {"be96531": 17.12,       # CROWN_PLAN R17, phase 11.25
            "b729e86": 16.78}       # CROWN_PLAN R1, phase 7.5

OUT_STEM = os.path.join(HERE, "..", "export", "coupon_step05")
OUT_JSON = os.path.join(HERE, "study_coupons.json")


def _outline():
    """The half-length profile as (x, y) points, gauge centre at the origin."""
    g, w, W = GAUGE_L / 2, GAUGE_W / 2, GRIP_W / 2
    rise = W - w
    a = math.sqrt(ARC_R ** 2 - (ARC_R - rise) ** 2)   # the arc's run along x
    th = math.asin(a / ARC_R) / 2
    mid = (g + ARC_R * math.sin(th), w + ARC_R * (1 - math.cos(th)))
    return g, w, W, a, mid


def build():
    import cadquery as cq
    g, w, W, a, mid = _outline()
    L2 = g + a + GRIP_L
    s = (cq.Workplane("XY").moveTo(-g, -w).lineTo(g, -w)
         .threePointArc((mid[0], -mid[1]), (g + a, -W)).lineTo(L2, -W).lineTo(L2, W)
         .lineTo(g + a, W).threePointArc(mid, (g, w)).lineTo(-g, w)
         .threePointArc((-mid[0], mid[1]), (-g - a, W)).lineTo(-L2, W).lineTo(-L2, -W)
         .lineTo(-g - a, -W).threePointArc((-mid[0], -mid[1]), (-g, -w)).close()
         .extrude(THICK))
    x_hole = L2 - HOLE_FROM_END
    s = s.faces(">Z").workplane().pushPoints([(-x_hole, 0), (x_hole, 0)]).hole(HOLE_D)
    os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
    cq.exporters.export(s, OUT_STEM + ".step")
    cq.exporters.export(s, OUT_STEM + ".stl", tolerance=0.005, angularTolerance=0.05)

    area = GAUGE_W * THICK
    net = (GRIP_W - HOLE_D) * THICK
    need = {k: v / WF.ALLOWABLE_STRESS_MPA for k, v in S_ZZ_MPA.items()}
    rec = {
        "what": "CROWN_PLAN Step 0.5 interlayer coupon: one geometry, printed flat (xy) and on end (z)",
        "files": [os.path.relpath(OUT_STEM + e, os.path.join(HERE, "..")) for e in (".step", ".stl")],
        "gauge_w_mm": GAUGE_W, "gauge_l_mm": GAUGE_L, "thick_mm": THICK,
        "grip_w_mm": GRIP_W, "grip_l_mm": GRIP_L, "arc_r_mm": ARC_R, "arc_run_mm": a,
        "hole_d_mm": HOLE_D, "hole_from_end_mm": HOLE_FROM_END,
        "overall_l_mm": 2 * L2, "volume_mm3": s.val().Volume(),
        "gauge_area_mm2": area, "hole_net_area_mm2": net, "hole_net_stress_ratio": area / net,
        "ultimate_mpa": WF.ULTIMATE_STRESS_MPA, "allowable_mpa": WF.ALLOWABLE_STRESS_MPA,
        "s_zz_mpa": S_ZZ_MPA, "ratio_needed": need,
        "xy_break_at_ultimate_kgf": WF.ULTIMATE_STRESS_MPA * area / KGF_N,
        "z_break_at_needed_kgf": {k: r * WF.ULTIMATE_STRESS_MPA * area / KGF_N
                                  for k, r in need.items()},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))


def score(path):
    rows = list(csv.DictReader(open(path)))
    sig = {"xy": [], "z": []}
    for r in rows:
        if r["break"].strip() != "gauge":
            continue
        a = float(r["width_mm"]) * float(r["thickness_mm"])
        sig[r["orientation"].strip().lower()].append(float(r["peak_kgf"]) * KGF_N / a)
    for k in sig:
        print(f"{k:>2}: n {len(sig[k])} gauge breaks of "
              f"{sum(r['orientation'].strip().lower() == k for r in rows)}, "
              + (f"mean {statistics.mean(sig[k]):.2f} MPa, sd {statistics.stdev(sig[k]):.2f}"
                 if len(sig[k]) > 1 else "too few to read"))
    if min(len(v) for v in sig.values()) < 5:
        print("INSUFFICIENT: R19 needs five gauge breaks per orientation")
        return
    m = {k: statistics.mean(v) for k, v in sig.items()}
    cv2 = sum((statistics.stdev(v) / m[k]) ** 2 / len(v) for k, v in sig.items())
    R = m["z"] / m["xy"]
    se = R * math.sqrt(cv2)
    print(f"R = {R:.4f}, SE {se:.4f}, 2-SE band {R - 2*se:.4f} .. {R + 2*se:.4f}; "
          f"allowable_zz {WF.ALLOWABLE_STRESS_MPA * R:.2f} MPa")
    print(f"control: sigma_XY {m['xy']:.2f} MPa against wheel_fea's ultimate "
          f"{WF.ULTIMATE_STRESS_MPA:.1f} (reported, not gated)")
    for k, s_zz in S_ZZ_MPA.items():
        need = s_zz / WF.ALLOWABLE_STRESS_MPA
        call = ("PASS" if R - 2 * se >= need else
                "FAIL" if R + 2 * se < need else "UNDECIDED")
        print(f"{k}: s_zz {s_zz:.2f} MPa needs R >= {need:.4f}: {call}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--score", metavar="CSV")
    args = ap.parse_args()
    score(args.score) if args.score else build()
