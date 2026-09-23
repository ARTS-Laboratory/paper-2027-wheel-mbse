"""A 3D solid whose cross-section IS the 2D mesh: every boundary loop of the medium Q9 mesh
(corner AND mid-side nodes, in order) becomes one closed spline, the face is extruded 22.4."""
import sys, gmsh, numpy as np
SP = sys.argv[1]
d = np.load(f"{SP}/mesh2d_medium.npz"); xy, conn = d["xy"], d["conn"]
E = np.vstack([np.c_[conn[:, i], conn[:, k], conn[:, j]] for (i, j, k) in [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)]])
key = np.sort(E[:, [0, 2]], 1)
_, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
bnd = E[cnt[inv] == 1]
nxt = {}
for a_, m_, b_ in bnd:
    nxt.setdefault(a_, []).append((m_, b_)); nxt.setdefault(b_, []).append((m_, a_))
assert all(len(v) == 2 for v in nxt.values())
seen, loops = set(), []
for start in list(nxt):
    if start in seen: continue
    loop, prev, cur = [start], None, start
    while True:
        seen.add(cur)
        opts = [o for o in nxt[cur] if o[1] != prev] if prev is not None else nxt[cur][:1]
        m_, b_ = opts[0]
        if b_ == start: loop.append(m_); break
        loop += [m_, b_]; prev, cur = cur, b_
    loops.append(np.array(loop))
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
occ = gmsh.model.occ
wires, areas = [], []
for L in loops:
    P = xy[L]
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    pts = [occ.addPoint(x, y, 0) for x, y in P]
    n = len(pts); k = max(1, round(n / 20)); cuts = [round(i * n / k) for i in range(k)] + [n]
    curves = [occ.addSpline([pts[j % n] for j in range(cuts[i], cuts[i + 1] + 1)]) for i in range(k)]
    wires.append(occ.addCurveLoop(curves)); areas.append(abs(area))
order = np.argsort(areas)[::-1]
print("loops", len(loops), "polygon areas", [round(areas[i], 3) for i in order])
faces = [occ.addPlaneSurface([w]) for w in wires]
fa = [occ.getMass(2, f) for f in faces]
print("face areas   ", [round(fa[i], 3) for i in order])
outer = faces[order[0]]
cut, _ = occ.cut([(2, outer)], [(2, faces[i]) for i in order[1:]])
occ.synchronize()
A = sum(occ.getMass(2, t) for _, t in cut)
print("section area %.4f  (2D mesh %.4f)" % (A, d["area"].sum()))
ext = occ.extrude(cut, 0, 0, 22.4)
occ.synchronize()
v = [e for e in ext if e[0] == 3]
print("volumes", len(v), "total %.4f  (2D area x 22.4 = %.4f)" % (sum(occ.getMass(3, t) for _, t in v), 22.4 * d["area"].sum()))
gmsh.write(f"{SP}/twin.step")
gmsh.finalize()
