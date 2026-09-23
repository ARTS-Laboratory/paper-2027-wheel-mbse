"""Half-model P2 tet mesh of an exported wheel STEP, in the 2D kernel's frame.

  - the hub interior r < 7.7 mm is cut away and the r = 7.7 face is the tie (u = 0),
    which is `wheel_fem`'s rigid fixed `hub_tie` at the same radius;
  - the solid is cut at the mid-plane z = 11.2 and only z <= 11.2 is kept (symmetry);
  - a refinement box sits over the ground contact at the bottom (y ~ -50).

usage: mesh3d.py STEP OUT.npz H_GLOBAL H_CONTACT
"""
import sys, time, gmsh, numpy as np

step, out, h, hc = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
R_TIE, ZMID = 7.7, 11.2

gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("General.NumThreads", 8)
occ = gmsh.model.occ
v = occ.importShapes(step)
cyl = occ.addCylinder(0, 0, -5, 0, 0, 40, R_TIE)
top = occ.addBox(-60, -60, ZMID, 120, 120, 20)
body, _ = occ.cut(v, [(3, cyl), (3, top)])
occ.synchronize()
assert len(body) == 1, body
vol = body[0][1]

# classify boundary faces by points sampled on them (a cylinder's centroid is on its axis)
tie, sym, od = [], [], []
for _, s in gmsh.model.getBoundary(body, oriented=False):
    lo, hi = gmsh.model.getParametrizationBounds(2, s)
    uv = [lo[0] + (hi[0] - lo[0]) * a for a in np.linspace(0.05, 0.95, 7)]
    vv = [lo[1] + (hi[1] - lo[1]) * a for a in np.linspace(0.05, 0.95, 7)]
    P = np.array(gmsh.model.getValue(2, s, [c for u in uv for w in vv for c in (u, w)])).reshape(-1, 3)
    r = np.hypot(P[:, 0], P[:, 1])
    if np.all(np.abs(P[:, 2] - ZMID) < 1e-6):
        sym.append(s)
    elif np.all(np.abs(r - R_TIE) < 1e-4):
        tie.append(s)
    elif np.all(r > 48.9) and np.ptp(P[:, 2]) > 1.0:
        od.append(s)            # the rim's outer surface, flat or crowned
gmsh.model.addPhysicalGroup(3, [vol], 1)
gmsh.model.addPhysicalGroup(2, tie, 11); gmsh.model.addPhysicalGroup(2, sym, 12)
gmsh.model.addPhysicalGroup(2, od, 13)

f = gmsh.model.mesh.field
b = f.add("Box")
f.setNumber(b, "VIn", hc); f.setNumber(b, "VOut", h)
f.setNumber(b, "XMin", -4.0); f.setNumber(b, "XMax", 4.0)
f.setNumber(b, "YMin", -51.0); f.setNumber(b, "YMax", -47.0)
f.setNumber(b, "ZMin", -1.0); f.setNumber(b, "ZMax", 12.0)
f.setNumber(b, "Thickness", 3.0)
f.setAsBackgroundMesh(b)
gmsh.option.setNumber("Mesh.MeshSizeMax", h)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 8)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
gmsh.option.setNumber("Mesh.Algorithm3D", 10)
gmsh.option.setNumber("Mesh.ElementOrder", 2)
gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)
gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)   # mid-edge nodes ON the geometry
t0 = time.time(); gmsh.model.mesh.generate(3); tm = time.time() - t0

tags, x, _ = gmsh.model.mesh.getNodes()
x = x.reshape(-1, 3); idx = np.full(int(tags.max()) + 1, -1); idx[tags] = np.arange(len(tags))
et, _, en = gmsh.model.mesh.getElements(3, vol)
assert list(et) == [11], et                      # 11 = 10-node tet
tet = idx[en[0].reshape(-1, 10)]

def facets(group_faces):
    tris = []
    for s in group_faces:
        et, _, en = gmsh.model.mesh.getElements(2, s)
        assert list(et) == [9], et              # 9 = 6-node triangle
        tris.append(idx[en[0].reshape(-1, 6)])
    return np.vstack(tris)

def nodes_of(group_faces):
    return np.unique(facets(group_faces))

np.savez(out, x=x, tet=tet, tie=nodes_of(tie), sym=nodes_of(sym), od_tri=facets(od),
         meta=np.array([h, hc, tm, len(tie), len(sym), len(od)]))
print(f"{step.split('/')[-1]} h={h} hc={hc} tets={len(tet)} nodes={len(x)} dof={3*len(x)} "
      f"mesh {tm:.1f}s faces tie/sym/od {len(tie)}/{len(sym)}/{len(od)} "
      f"od tris {len(facets(od))}")
gmsh.finalize()
