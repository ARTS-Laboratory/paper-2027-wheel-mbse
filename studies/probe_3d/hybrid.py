"""A prismatic wheel whose section is the STEP's inside r < R and the twin's outside it
(mode hub), the reverse (mode rim), or all STEP (mode step).  Both inputs are extrusions
over z in [0, 22.4]; the section is taken at z = 11.2, glued at r = R where the two bodies
coincide to 9 um (sec202 sec2), and extruded 22.4.
usage: hybrid.py STEP TWIN MODE R OUT.step [TOL]"""
import sys, gmsh
step, twin, mode, R, out = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
tol = float(sys.argv[6]) if len(sys.argv) > 6 else 0.02
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
occ = gmsh.model.occ
def section(path):
    v = occ.importShapes(path)
    sl = occ.addRectangle(-60, -60, 11.2, 120, 120)
    s, _ = occ.intersect(v, [(2, sl)])
    assert len(s) == 1, s
    return s
sS, sT = section(step), section(twin)
occ.synchronize()
aS, aT = occ.getMass(2, sS[0][1]), occ.getMass(2, sT[0][1])
if mode == "step":
    body = sS; occ.remove(sT, recursive=True)
else:
    inner, outer = (sS, sT) if mode == "hub" else (sT, sS)
    d1 = occ.addDisk(0, 0, 11.2, R, R); d2 = occ.addDisk(0, 0, 11.2, R, R)
    a, _ = occ.intersect(inner, [(2, d1)])
    b, _ = occ.cut(outer, [(2, d2)])
    occ.synchronize()
    ma, mb = sum(occ.getMass(2, t) for _, t in a), sum(occ.getMass(2, t) for _, t in b)
    gmsh.option.setNumber("Geometry.ToleranceBoolean", tol)
    body, _ = occ.fuse(a, b)
    occ.synchronize()
    print(f"pieces inner {len(a)} {ma:.4f}  outer {len(b)} {mb:.4f}  sum {ma + mb:.4f}")
assert len(body) == 1, body
occ.synchronize()
A = occ.getMass(2, body[0][1])
print(f"{mode} R={R}: section STEP {aS:.4f} twin {aT:.4f} hybrid {A:.4f}  faces {len(body)}")
occ.translate(body, 0, 0, -11.2)
ext = occ.extrude(body, 0, 0, 22.4)
occ.synchronize()
vols = [e for e in ext if e[0] == 3]
assert len(vols) == 1
print("volume %.4f  = section x 22.4 %.4f" % (occ.getMass(3, vols[0][1]), 22.4 * A))
gmsh.write(out)
gmsh.finalize()
