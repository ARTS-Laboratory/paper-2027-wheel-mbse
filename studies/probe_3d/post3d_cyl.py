"""Band stress in CYLINDRICAL components, to split von Mises into what the print cares about.
Printed flat: layers stack along z (the axle), so s_tt (hoop) and s_rr, s_rt lie IN the layers;
s_zz is ACROSS layers (interlayer tension), s_rz / s_tz are interlayer shear.
Reports, over the rim band (r >= 48.45) near the bottom (|x|<10, y<-40), at element nodes:
vM max; max tensile principal (and its direction's z-share); max hoop tension; max hoop
compression; max s_zz tension; max interlayer shear.  usage: post3d_cyl.py MESH:RESULT ..."""
import sys, numpy as np
src = open(sys.argv[0].replace("post3d_cyl.py", "fe3d.py")).read()
exec("import numpy as np\n" + src[src.index("# ---- tet10"):src.index("QL, QW = tet_quad()")])
E, NU = 2300.0, 0.35; LAM = E*NU/((1+NU)*(1-2*NU)); MU = E/(2*(1+NU))
Lnode = np.zeros((10, 4)); Lnode[np.arange(4), np.arange(4)] = 1.0
for k, (i, j) in enumerate(EDGES): Lnode[4 + k, [i, j]] = 0.5
_, dNn = tet_shape(Lnode)
for tag in sys.argv[1:]:
    mesh, res = tag.split(":")
    d = np.load(mesh); X, T = d["x"], d["tet"]; U = np.load(res)["u"].reshape(-1, 3)
    cen = X[T[:, :4]].mean(1)
    sel = np.where((np.abs(cen[:, 0]) < 10) & (cen[:, 1] < -40))[0]
    Xe = X[T[sel]]
    J = np.einsum("eai,paj->epij", Xe, dNn)
    G = np.einsum("paj,epji->epai", dNn, np.linalg.inv(J))
    gu = np.einsum("eai,epaj->epij", U[T[sel]], G); eps = 0.5*(gu+np.swapaxes(gu,-1,-2))
    sig = LAM*np.trace(eps, axis1=-2, axis2=-1)[..., None, None]*np.eye(3) + 2*MU*eps
    if "'kinematics': 'svk'" in str(np.load(res)["rep"][0]):
        F = gu + np.eye(3); Eg = 0.5*(np.swapaxes(F, -1, -2) @ F - np.eye(3))
        Sg = LAM*np.trace(Eg, axis1=-2, axis2=-1)[..., None, None]*np.eye(3) + 2*MU*Eg
        sig = F @ Sg @ np.swapaxes(F, -1, -2) / np.linalg.det(F)[..., None, None]
    P = Xe; r = np.hypot(P[..., 0], P[..., 1])
    c, s = P[..., 0]/r, P[..., 1]/r
    er = np.stack([c, s, 0*c], -1); et = np.stack([-s, c, 0*c], -1); ez = np.zeros_like(er); ez[..., 2] = 1
    comp = lambda a, b: np.einsum("epi,epij,epj->ep", a, sig, b)
    stt, szz, srz, stz = comp(et, et), comp(ez, ez), comp(er, ez), comp(et, ez)
    dev = sig - np.trace(sig, axis1=-2, axis2=-1)[..., None, None]*np.eye(3)/3
    vm = np.sqrt(1.5*np.einsum("epij,epij->ep", dev, dev))
    w, v = np.linalg.eigh(sig); s1 = w[..., 2]; zshare = v[..., 2, 2]**2
    band = r >= 48.45
    def at(q, lab, fn=np.argmax):
        qq = np.where(band, q, -np.inf if fn is np.argmax else np.inf); k = np.unravel_index(fn(qq), qq.shape); p = P[k]
        return f"{lab} {q[k]:7.2f} @x{p[0]:+5.2f} r{r[k]:5.2f} z{p[2]:5.2f}"
    print(tag.split("/")[-1].replace("_free.npz", ""))
    print("   " + at(vm, "vM") + " | " + at(s1, "s1(tens)") + f" zshare {zshare[np.unravel_index(np.argmax(np.where(band,s1,-np.inf)), s1.shape)]:.2f}")
    print("   " + at(stt, "hoop+") + " | " + at(stt, "hoop-", np.argmin))
    print("   " + at(szz, "s_zz+ (interlayer tension)") + " | " + at(np.hypot(srz, stz), "interlayer shear"))
    k = np.unravel_index(np.argmax(np.where(band, vm, -1)), vm.shape)
    print(f"   at the vM peak: hoop {stt[k]:+.2f}  s_zz {szz[k]:+.2f}  s_rr {comp(er,er)[k]:+.2f}  s1 {s1[k]:+.2f}  s3 {w[k][0]:+.2f}")
