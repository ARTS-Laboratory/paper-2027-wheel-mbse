"""Distance from every 2D boundary node (medium mesh) to the flat STEP's surface at z = 11.2.
Every 2D node lies inside the solid (checked), so the distance is how far INSIDE it sits:
zero where the two boundaries coincide."""
import sys, gmsh, numpy as np
SP = sys.argv[1]
d = np.load(f"{SP}/mesh2d_medium.npz"); xy, conn = d["xy"], d["conn"]
# Q9 edges: corner pairs (0,1)(1,2)(2,3)(3,0) with mid nodes 4..7
E = []
for (i, j, k) in [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)]:
    E.append(np.c_[conn[:, i], conn[:, j], conn[:, k]])
E = np.vstack(E); key = np.sort(E[:, :2], 1)
_, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
bnd = E[cnt[inv] == 1]
bn = np.unique(bnd); P = xy[bn]; r = np.hypot(*P.T)
keep = r > 7.7 + 1e-6                      # the hub tie circle is inside the solid disk
P, bn = P[keep], bn[keep]
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.occ.importShapes(f"{SP}/flat.step"); gmsh.model.occ.synchronize()
faces = [t for _, t in gmsh.model.getEntities(2)]
side = []
for s in faces:
    bb = gmsh.model.getBoundingBox(2, s)
    if bb[5] - bb[2] > 20: side.append((s, bb))      # through-thickness side faces only
coords = np.c_[P, np.full(len(P), 11.2)]
best = np.full(len(P), np.inf)
for s, bb in side:
    sel = np.where((P[:, 0] > bb[0] - 1) & (P[:, 0] < bb[3] + 1) & (P[:, 1] > bb[1] - 1) & (P[:, 1] < bb[4] + 1))[0]
    if len(sel) == 0: continue
    cp, _ = gmsh.model.getClosestPoint(2, s, coords[sel].ravel().tolist())
    dist = np.linalg.norm(np.array(cp).reshape(-1, 3) - coords[sel], axis=1)
    best[sel] = np.minimum(best[sel], dist)
gmsh.finalize()
r = np.hypot(*P.T)
print("boundary nodes", len(P))
np.savez(f"{SP}/bdist.npz", P=P, d=best)
for lo, hi in [(7.7, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 48.49), (48.49, 49.99), (49.99, 50.1)]:
    name = f"{lo}-{hi}"
    s = (r >= lo) & (r < hi)
    b = best[s]
    if not s.any(): continue
    print(f"{name:20s} n={s.sum():5d}  median {np.median(b):.5f}  p90 {np.percentile(b,90):.5f}  max {b.max():.5f} mm  frac>0.005 {np.mean(b>0.005):.3f}")
np.savez(f"{SP}/bdist.npz", P=P, d=best)
