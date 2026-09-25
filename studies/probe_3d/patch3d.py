"""Contact-patch metrics and a scrub-torque proxy, from a finished `fe3d.py` solve (CROWN_PLAN.md §4).

The pressure is fe3d's own: the penalty eps_n * penetration at the same 6x6 Gauss points on the
same P2 OD triangles, gap measured on the REFERENCE surface through u_y alone, ground at
y = -r_out + drop.  So the integral of p dA IS fe3d's contact force, and that equality is the
control printed on every row: it must reproduce `contact_force_half_n` (33.36165 N).  fe3d's
candidate window is not stored, so every OD triangle below y -45 is read; fe3d refuses a patch
within 0.1 mm of its window, so nothing outside it penetrates, and the control would show it.

Reported per solve, half model mirrored at the mid-plane z = 11.2:
  patches    connected components of active triangles (sharing a node), with each one's share of
             the force; CROWN_PLAN.md Step 0.1 passes a rim only if this is 1 at every phase
  length     x extent;  width  2 (11.2 - z_min);  area  the active quadrature area, doubled
  p_peak, p_mean = F / area
  scrub      M / mu = int p |q - c| dA over the full patch, c the pressure centroid (x_c, 11.2):
             Coulomb pivot torque per unit friction coefficient, N mm.  mu scales out of every
             comparison between rims.  lever = scrub / F, the pressure-weighted mean arm, mm.
A proxy for turn-in resistance on flat, rigid, frictionless ground under a radial load; it is not
a steering model and the tree has no side load or camber to make one.

usage: patch3d.py MESH.npz:RESULT.npz ...
"""
import sys, ast, numpy as np

EPS_N, ZMID = 1.0e4, 11.2


def tri_quad(n=6):                               # fe3d.py's rule, verbatim
    g, w = np.polynomial.legendre.leggauss(n); g = (g + 1) / 2; w = w / 2
    u, v = np.meshgrid(g, g, indexing="ij"); wu, wv = np.meshgrid(w, w, indexing="ij")
    xi = u.ravel(); eta = (v * (1 - u)).ravel(); W = (wu * wv * (1 - u)).ravel()
    return xi, eta, W


xi, eta, TW = tri_quad(6)
L = np.c_[1 - xi - eta, xi, eta]
TN = np.zeros((len(xi), 6)); TdN = np.zeros((len(xi), 6, 2))
dLt = np.array([[-1, -1], [1, 0], [0, 1]], float)
for i in range(3):
    TN[:, i] = L[:, i] * (2 * L[:, i] - 1); TdN[:, i] = (4 * L[:, i] - 1)[:, None] * dLt[i]
for k, (i, j) in enumerate([(0, 1), (1, 2), (0, 2)]):
    TN[:, 3 + k] = 4 * L[:, i] * L[:, j]
    TdN[:, 3 + k] = 4 * (L[:, i][:, None] * dLt[j] + L[:, j][:, None] * dLt[i])


def components(tris):
    """Connected components of triangles that share a node (union-find on node ids)."""
    parent = {}
    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for t in tris:
        r0 = find(t[0])
        for n in t[1:]:
            rn = find(n)
            if rn != r0:
                parent[rn] = r0
    return np.array([find(t[0]) for t in tris])


print(f"{'solve':34s} {'F_half':>9s} {'patches (force share)':>24s} {'length':>7s} {'width':>6s} "
      f"{'area':>7s} {'p_peak':>7s} {'p_mean':>7s} {'scrub':>8s} {'lever':>6s}")
for tag in sys.argv[1:]:
    mesh, res = tag.split(":")
    d = np.load(mesh); r = np.load(res)
    X, OD = d["x"], d["od_tri"]; U = r["u"].reshape(-1, 3)
    rep = ast.literal_eval(str(r["rep"][0]).replace("np.float64(", "("))
    tri = OD[np.all(X[OD, 1] < -45.0, axis=1)]
    Xt = X[tri]
    Xq = np.einsum("qa,tai->tqi", TN, Xt)
    Jt = np.einsum("tai,qaj->tqij", Xt, TdN)
    dA = np.linalg.norm(np.cross(Jt[..., 0], Jt[..., 1]), axis=-1) * TW
    uy = np.einsum("qa,ta->tq", TN, U[tri, 1])
    g = Xq[..., 1] + uy - (-rep.get("r_out", 50.0) + rep["axle_drop_mm"])   # pre-9f296d0: 50
    p = EPS_N * np.maximum(-g, 0.0)                              # [tri, q]
    f = p * dA
    F = f.sum()
    act = (p > 0).any(1)
    lab = components(tri[act])
    share = sorted((f[act][lab == c].sum() / F for c in np.unique(lab)), reverse=True)
    x, z = Xq[..., 0], Xq[..., 2]
    xc = (f * x).sum() / F
    scrub = 2.0 * (f * np.hypot(x - xc, z - ZMID)).sum()        # full patch, per unit mu
    on = p > 0
    name = res.split("/")[-1].replace(".npz", "")
    shares = " ".join(f"{s:.3f}" for s in share[:3]) + (" .." if len(share) > 3 else "")
    print(f"{name:34s} {F:9.5f} {len(share):3d} ({shares:>18s}) {x[on].max() - x[on].min():7.3f} "
          f"{2 * (ZMID - z[on].min()):6.3f} {2 * dA[on].sum():7.3f} {p.max():7.2f} "
          f"{2 * F / (2 * dA[on].sum()):7.2f} {scrub:8.3f} {scrub / (2 * F):6.3f}")
    assert abs(F - rep["contact_force_half_n"]) < 1e-6 * rep["contact_force_half_n"], (
        f"{name}: int p dA {F} is not fe3d's contact force {rep['contact_force_half_n']} -- "
        f"this is not the pressure the solve balanced")
