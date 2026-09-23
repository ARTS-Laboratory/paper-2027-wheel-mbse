"""von Mises from a saved 3D solve, over |x| < 10, y < -40, split rim band / below band.

usage: post.py MESH.npz:RESULT.npz [...]
"""
import sys, numpy as np
src = open(sys.argv[0].replace("post.py", "fe3d.py")).read()
exec("import numpy as np\n" + src[src.index("# ---- tet10"):src.index("t0 = time.time()\nrows")].replace("X[T[idx]]", "Xg[Tg[idx]]"))
E, NU = 2300.0, 0.35; LAM = E*NU/((1+NU)*(1-2*NU)); MU = E/(2*(1+NU))
for tag in sys.argv[1:]:
    mesh, res = tag.split(":")
    d = np.load(mesh); Xg, Tg = d["x"], d["tet"]; U = np.load(res)["u"].reshape(-1, 3)
    cen = Xg[Tg[:, :4]].mean(1)
    sel = np.where((np.abs(cen[:, 0]) < 10) & (cen[:, 1] < -40))[0]
    G, wd = element_grads(sel)
    gu = np.einsum("eai,eqaj->eqij", U[Tg[sel]], G); eps = 0.5*(gu+np.swapaxes(gu,-1,-2))
    tr = np.trace(eps, axis1=-2, axis2=-1)[..., None, None]
    sig = LAM*tr*np.eye(3) + 2*MU*eps
    dev = sig - np.trace(sig, axis1=-2, axis2=-1)[..., None, None]*np.eye(3)/3
    vm = np.sqrt(1.5*np.einsum("eqij,eqij->eq", dev, dev))
    xq = np.einsum("qa,eai->eqi", QN, Xg[Tg[sel]]); r = np.hypot(xq[..., 0], xq[..., 1])
    band = r > 48.5 - 1e-9
    def top(mask, label):
        v = np.where(mask, vm, -1); k = np.unravel_index(np.argmax(v), v.shape)
        p = xq[k]; print(f"   {label:34s} vM {v[k]:7.3f} MPa at x {p[0]:7.3f} y {p[1]:8.3f} z {p[2]:6.3f}  r {np.hypot(p[0],p[1]):.3f}")
    print(tag.split("/")[-1])
    top(np.ones_like(band), "max, |x|<10, y<-40 (any)")
    top(band, "max in rim band r>=48.5")
    top(band & (np.abs(xq[..., 0] - 1.4) < 2.0), "rim band, within 2 mm of patch")
    top(~band, "max below band (junction/spoke)")
