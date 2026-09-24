"""von Mises AT THE OD-SURFACE NODES of every tet near the bottom (|x|<10, y<-40): the 3D
twin of wheel_adjoint.rim_band_surface_stress.  Profile along x, max, and where.
usage: post3d_surface.py MESH.npz:RESULT.npz [...]   (reads fe3d.py beside itself, as post.py does)"""
import sys, numpy as np
src = open(sys.argv[0].replace("post3d_surface.py", "fe3d.py")).read()
exec("import numpy as np\n" + src[src.index("# ---- tet10"):src.index("QL, QW = tet_quad()")])
E, NU = 2300.0, 0.35; LAM = E*NU/((1+NU)*(1-2*NU)); MU = E/(2*(1+NU))
Lnode = np.zeros((10, 4)); Lnode[np.arange(4), np.arange(4)] = 1.0
for k, (i, j) in enumerate(EDGES): Lnode[4 + k, [i, j]] = 0.5
_, dNn = tet_shape(Lnode)                                  # [10 eval, 10 nodes, 3]
for tag in sys.argv[1:]:
    mesh, res = tag.split(":")
    d = np.load(mesh); X, T, OD = d["x"], d["tet"], d["od_tri"]; U = np.load(res)["u"].reshape(-1, 3)
    cen = X[T[:, :4]].mean(1)
    sel = np.where((np.abs(cen[:, 0]) < 10) & (cen[:, 1] < -40))[0]
    Xe = X[T[sel]]
    J = np.einsum("eai,paj->epij", Xe, dNn)
    G = np.einsum("paj,epji->epai", dNn, np.linalg.inv(J))
    gu = np.einsum("eai,epaj->epij", U[T[sel]], G); eps = 0.5*(gu+np.swapaxes(gu,-1,-2))
    sig = LAM*np.trace(eps, axis1=-2, axis2=-1)[..., None, None]*np.eye(3) + 2*MU*eps
    if "'kinematics': 'svk'" in str(np.load(res)["rep"][0]):      # Cauchy, as fe3d.py reports
        F = gu + np.eye(3); Eg = 0.5*(np.swapaxes(F, -1, -2) @ F - np.eye(3))
        Sg = LAM*np.trace(Eg, axis1=-2, axis2=-1)[..., None, None]*np.eye(3) + 2*MU*Eg
        sig = F @ Sg @ np.swapaxes(F, -1, -2) / np.linalg.det(F)[..., None, None]
    dev = sig - np.trace(sig, axis1=-2, axis2=-1)[..., None, None]*np.eye(3)/3
    vm = np.sqrt(1.5*np.einsum("epij,epij->ep", dev, dev))
    on = np.isin(T[sel], np.unique(OD))
    P = Xe                                                  # node coords per (e, p)
    v = np.where(on, vm, -1); k = np.unravel_index(np.argmax(v), v.shape); p = P[k]
    print(tag.split("/")[-1], f"OD-node max vM {v[k]:.3f} at x {p[0]:+.3f} z {p[2]:.3f}")
    for zlo, zhi, lab in ((11.2 - 1e-6, 11.3, "mid-plane z=11.2"), (-1e-6, 1e-6, "free face z=0"), (-1, 12, "all z")):
        m = on & (P[..., 2] >= zlo) & (P[..., 2] <= zhi)
        x, vv = P[..., 0][m], vm[m]
        prof = [vv[(x >= a) & (x < a + 0.5)].max() if np.any((x >= a) & (x < a + 0.5)) else np.nan for a in np.arange(-6, 6, 0.5)]
        print(f"   {lab:17s} max {vv.max():6.2f} | " + " ".join(f"{q:5.1f}" for q in prof))
