"""Linear-elastic P2-tet solve of a half wheel against a rigid frictionless flat ground.

The 2D kernel's contact problem, in 3D: the hub tie (r = 7.7 face) is fixed, the ground
y = -50 + delta indents the rim OD, penalty eps_n = 1e4 N/mm^3 on the reference surface,
and delta is found by secant so the contact force equals the service load.  Half model,
so the target is TOTAL_FORCE / 2 and u_z = 0 on the mid-plane.

Contact is the only nonlinearity and it lives on a few thousand DOF, so K is factored ONCE
and the problem is condensed onto the candidate contact y-DOF exactly (S = E_C' K^-1 E_C),
after which every Newton / secant step is dense and small.  The factor is PARDISO's SPD
mode (13.3 s and 5.1e8 factor nonzeros at 651k free DOF, against 32.2 s and 1.0e9 for the
general mode), and the condensation is LAZY: columns of S are computed only for the nodes
the patch needs, grown whenever a disallowed point penetrates, so at exit the restricted
problem IS the unrestricted one.  1115 of 2423 columns at 2.0 mm; the full condensation
reproduced the lazy answer to 1.7e-12 relative.

The candidate window is [x0, x1] because at phase 0 the patch is NOT at the bottom point:
the 2D kernel puts its centre 1.37 mm (strain) to 1.50 mm (stress) toward +x.  A first run
with a +/-1.5 mm window truncated the patch and read 1.5160 mm; the assertion below now
refuses any patch that comes within 0.1 mm of the window edge.

usage: fe3d.py MESH.npz OUT.npz [--faces free|clamped_z] [--x0 0] [--x1 3]
                                [--seed0 0.8] [--seed1 1.9] [--kinematics linear|svk]
  --faces clamped_z   u_z = 0 on the z = 0 face as well: exact plane strain for an
                      extrusion, the control that checks this code against the 2D kernel
  --kinematics svk    St. Venant-Kirchhoff, `wheel_fem`'s law in 3D: Green-Lagrange strain,
                      the same energy, contact on the reference surface through u_y alone.
                      Solved AFTER the linear answer, from it, by full Newton on (u, delta):
                      the consistent tangent plus the penalty Hessian, the service load as a
                      bordered constraint (two solves per factor), Armijo backtracking on
                      W + eps/2 int pen^2 - F delta.  Modified Newton on the linear K was
                      tried first and diverged: at the linear answer the SVK residual is
                      11-21x the load, the thin spokes' membrane strain that K has no term
                      for.  Reports both drops.  PLAN.md sec204.
"""
import sys, time, argparse, numpy as np, scipy.sparse as sp
import pypardiso

ap = argparse.ArgumentParser()
ap.add_argument("mesh"); ap.add_argument("out")
ap.add_argument("--faces", default="free", choices=["free", "clamped_z"])
ap.add_argument("--x0", type=float, default=0.0)
ap.add_argument("--x1", type=float, default=3.0)
ap.add_argument("--block", type=int, default=300)
ap.add_argument("--seed0", type=float, default=0.8)
ap.add_argument("--seed1", type=float, default=1.9)
ap.add_argument("--kinematics", default="linear", choices=["linear", "svk"])
ap.add_argument("--tol", type=float, default=1e-10)
a = ap.parse_args()

E, NU, EPS_N = 2300.0, 0.35, 1.0e4
F_TARGET = 66.7233 / 2.0
R_OUT = 50.0
LAM = E * NU / ((1 + NU) * (1 - 2 * NU)); MU = E / (2 * (1 + NU))

d = np.load(a.mesh)
X, T, TIE, SYM, OD = d["x"], d["tet"], d["tie"], d["sym"], d["od_tri"]
N = len(X); ndof = 3 * N
t_start = time.time()

# ---- tet10 (gmsh ordering: edges 01 12 02 03 23 13) --------------------------------
EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]

def tet_quad():
    q = []
    for aa, w in ((0.0927352503108912, 0.01224884051939366),
                  (0.3108859192633006, 0.01878132095300264)):
        for k in range(4):
            L = np.full(4, aa); L[k] = 1 - 3 * aa; q.append((L, w))
    b = 0.4544962958743504; w = 0.007091003462846911
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        L = np.full(4, 0.5 - b); L[i] = b; L[j] = b; q.append((L, w))
    L = np.array([p[0] for p in q]); W = np.array([p[1] for p in q])
    return L, W

def tet_shape(L):
    """N [Q,10] and dN/dxi [Q,10,3] at barycentric points L [Q,4]."""
    dL = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    Nn = np.zeros((len(L), 10)); dN = np.zeros((len(L), 10, 3))
    for i in range(4):
        Nn[:, i] = L[:, i] * (2 * L[:, i] - 1)
        dN[:, i] = (4 * L[:, i] - 1)[:, None] * dL[i]
    for k, (i, j) in enumerate(EDGES):
        Nn[:, 4 + k] = 4 * L[:, i] * L[:, j]
        dN[:, 4 + k] = 4 * (L[:, i][:, None] * dL[j] + L[:, j][:, None] * dL[i])
    return Nn, dN

QL, QW = tet_quad()
# the rule must integrate every monomial of degree <= 5 exactly over the reference tet
from math import factorial
for p in range(4):
    for q_ in range(4 - p):
        for r in range(6 - p - q_):
            if p + q_ + r > 5: continue
            exact = factorial(p) * factorial(q_) * factorial(r) / factorial(p + q_ + r + 3)
            got = np.sum(QW * QL[:, 1] ** p * QL[:, 2] ** q_ * QL[:, 3] ** r)
            assert abs(got - exact) < 1e-14, (p, q_, r, got, exact)
QN, QdN = tet_shape(QL)
assert np.allclose(QN.sum(1), 1) and np.allclose(QdN.sum(1), 0)

def element_grads(idx):
    Xe = X[T[idx]]                                   # [e,10,3]
    J = np.einsum("eai,qaj->eqij", Xe, QdN)          # dx_i/dxi_j
    det = np.linalg.det(J)
    assert det.min() > 0, "inverted element"
    G = np.einsum("qaj,eqji->eqai", QdN, np.linalg.inv(J))   # dN/dx
    return G, det * QW

t0 = time.time()
rows, cols, vals = [], [], []
GW = []
vol = 0.0
CH = 20000
for s in range(0, len(T), CH):
    idx = np.arange(s, min(s + CH, len(T)))
    G, wd = element_grads(idx)
    vol += wd.sum()
    Ke = (LAM * np.einsum("eq,eqai,eqbj->eaibj", wd, G, G)
          + MU * np.einsum("eq,eqaj,eqbi->eaibj", wd, G, G))
    Kd = MU * np.einsum("eq,eqak,eqbk->eab", wd, G, G)
    for i in range(3):
        Ke[:, :, i, :, i] += Kd
    dof = (3 * T[idx][:, :, None] + np.arange(3)).reshape(len(idx), 30)
    rows.append(np.repeat(dof, 30, axis=1).ravel().astype(np.int32))
    cols.append(np.tile(dof, (1, 30)).ravel().astype(np.int32))
    vals.append(Ke.reshape(len(idx), 900).ravel())
    if a.kinematics == "svk":
        GW.append((dof.ravel(), T[idx], G, wd))
K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                  shape=(ndof, ndof))
del rows, cols, vals
t_asm = time.time() - t0
print(f'assembled {t_asm:.1f}s nnz {K.nnz}', flush=True)

# ---- constraints ------------------------------------------------------------------
fixed = np.zeros(ndof, bool)
for c in range(3): fixed[3 * TIE + c] = True
fixed[3 * SYM + 2] = True
if a.faces == "clamped_z":            # u_z = 0 on the z = 0 face too: exact plane strain
    fixed[3 * np.where(np.abs(X[:, 2]) < 1e-9)[0] + 2] = True
free = np.where(~fixed)[0]
pos = np.full(ndof, -1); pos[free] = np.arange(len(free))
Kff = K[free][:, free].tocsr()

# ---- contact candidates -----------------------------------------------------------
def tri_quad(n=6):
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
assert abs(TW.sum() - 0.5) < 1e-14

ctri = OD[np.all((X[OD, 1] < -45.0) & (X[OD, 0] > a.x0) & (X[OD, 0] < a.x1), axis=1)]
Xt = X[ctri]
Xq = np.einsum("qa,tai->tqi", TN, Xt)
Jt = np.einsum("tai,qaj->tqij", Xt, TdN)
dA = np.linalg.norm(np.cross(Jt[..., 0], Jt[..., 1]), axis=-1) * TW
cnodes = np.unique(ctri)
cdof = 3 * cnodes + 1
assert not fixed[cdof].any()
cloc = np.searchsorted(cnodes, ctri)                   # tri node -> candidate index
m = len(cnodes)
# B maps candidate u_y [m] to the y-displacement at every contact quad point
B = sp.csr_matrix((np.broadcast_to(TN[None], (len(ctri),) + TN.shape).ravel(),
                   (np.repeat(np.arange(len(ctri) * len(xi)), 6),
                    np.broadcast_to(cloc[:, None, :], (len(ctri), len(xi), 6)).ravel())),
                  shape=(len(ctri) * len(xi), m))
yq = Xq[..., 1].ravel(); wq = dA.ravel(); xq = Xq[..., 0].ravel(); zq = Xq[..., 2].ravel()

# ---- factor once (SPD, upper triangle); condense LAZILY onto the nodes contact needs --
t0 = time.time()
solver = pypardiso.PyPardisoSolver(mtype=2)
Kup = sp.triu(Kff, format="csr")
solver.factorize(Kup)
t_fact = time.time() - t0
print(f'factored {t_fact:.1f}s m={m}', flush=True)
S = np.zeros((m, m)); have = np.zeros(m, bool)
cpos = pos[cdof]
t_cond = 0.0
def add_columns(idx):
    global t_cond
    idx = np.setdiff1d(idx, np.where(have)[0])
    t = time.time()
    for s in range(0, len(idx), a.block):
        cols_ = idx[s:s + a.block]
        R = np.zeros((len(free), len(cols_))); R[cpos[cols_], np.arange(len(cols_))] = 1.0
        S[:, cols_] = solver.solve(Kup, R)[cpos]
        del R
    have[idx] = True
    t_cond += time.time() - t
    print(f'  +{len(idx)} columns -> {have.sum()} of {m}  ({t_cond:.1f}s)', flush=True)

npt = len(ctri) * len(xi)
PT_NODES = np.broadcast_to(cloc[:, None, :], (len(ctri), len(xi), 6)).reshape(npt, 6)

def contact_state(uc, delta, allowed=None):
    yg = -R_OUT + delta
    g = yq + B @ uc - yg
    pen = np.maximum(-g, 0.0)
    if allowed is not None:
        pen = np.where(allowed, pen, 0.0)
    fc = EPS_N * (B.T @ (pen * wq))                   # +y force on the candidates
    act = pen > 0
    return g, pen, fc, act

def solve_at(delta, uc0):
    """Contact allowed only at points whose nodes all have columns; grow until no
    disallowed point penetrates.  At convergence this is the unrestricted solution."""
    uc = uc0.copy()
    while True:
        allowed = have[PT_NODES].all(1)
        for it in range(100):
            g, pen, fc, act = contact_state(uc, delta, allowed)
            r = uc - S @ fc
            Ba = B[act]
            Kc = EPS_N * (Ba.T @ sp.diags(wq[act]) @ Ba).toarray()
            uc = uc - np.linalg.solve(np.eye(m) + S @ Kc, r)
            g2, pen2, fc2, act2 = contact_state(uc, delta, allowed)
            if np.array_equal(act2, act) and np.abs(uc - S @ fc2).max() < 1e-13 * max(1.0, np.abs(uc).max()):
                break
        else:
            raise RuntimeError("Newton did not settle")
        gall = contact_state(uc, delta)[0]
        bad = (gall < 0) & ~allowed
        if not bad.any():
            return uc, it + 1
        # add the penetrating points' nodes and every candidate within 0.3 mm of them
        pts = np.c_[xq[bad], zq[bad]]
        cx = X[cnodes][:, [0, 2]]
        near = np.where(np.min(np.linalg.norm(cx[:, None, :] - pts[None, ::7, :], axis=-1), axis=1) < 0.3)[0]
        add_columns(np.union1d(np.unique(PT_NODES[bad]), near))

# seed: every candidate within the window's middle, by the 2D patch
add_columns(np.where((X[cnodes, 0] > a.seed0) & (X[cnodes, 0] < a.seed1))[0])
t0 = time.time()
hist = []
d0, d1 = 1.6, 1.7
uc = np.zeros(m)
uc, it0 = solve_at(d0, uc); f0 = contact_state(uc, d0)[2].sum(); hist.append((d0, f0, it0))
uc1, it1 = solve_at(d1, uc); f1 = contact_state(uc1, d1)[2].sum(); hist.append((d1, f1, it1))
for k in range(30):
    if abs(f1 - F_TARGET) <= 1e-10 * F_TARGET: break
    d2 = d1 - (f1 - F_TARGET) * (d1 - d0) / (f1 - f0)
    d0, f0, uc = d1, f1, uc1
    d1 = d2
    uc1, it = solve_at(d1, uc); f1 = contact_state(uc1, d1)[2].sum(); hist.append((d1, f1, it))
delta, uc = d1, uc1
t_sec = time.time() - t0

# ---- SVK: full Newton from the linear answer ----------------------------------------
def svk_chunks(u, what):
    """Yield per-chunk SVK quantities: 'W' energy, 'f' internal force, 'K' tangent."""
    U = u.reshape(-1, 3)
    for dofs, Te, G, wd in GW:
        F = np.einsum("eai,eqaj->eqij", U[Te], G) + np.eye(3)
        Eg = 0.5 * (np.einsum("eqki,eqkj->eqij", F, F) - np.eye(3))
        trE = np.trace(Eg, axis1=-2, axis2=-1)
        if what == "W":
            yield np.sum(wd * (0.5 * LAM * trE ** 2 + MU * np.einsum("eqij,eqij->eq", Eg, Eg)))
            continue
        Sg = LAM * trE[..., None, None] * np.eye(3) + 2 * MU * Eg
        if what == "f":
            yield dofs, np.einsum("eq,eqik,eqkj,eqaj->eai", wd, F, Sg, G)
            continue
        # dP_iJ/dF_kL = delta_ik S_JL + lam F_iJ F_kL + mu (F_iM F_kM delta_JL + F_iL F_kJ)
        Bf = np.einsum("eqim,eqam->eqai", F, G)
        Ke = (LAM * np.einsum("eq,eqai,eqbk->eaibk", wd, Bf, Bf)
              + MU * np.einsum("eq,eqbi,eqak->eaibk", wd, Bf, Bf)
              + MU * np.einsum("eq,eqik,eqab->eaibk", wd, np.einsum("eqim,eqkm->eqik", F, F),
                               np.einsum("eqaj,eqbj->eqab", G, G)))
        Kg = np.einsum("eq,eqaj,eqjl,eqbl->eab", wd, G, Sg, G)
        for i in range(3):
            Ke[:, :, i, :, i] += Kg
        yield dofs, Ke

def svk_W(u):
    return sum(svk_chunks(u, "W"))

def svk_fint(u):
    """Internal force of the SVK energy, [ndof]: f_a = sum_q w F S dN_a/dX."""
    f = np.zeros(ndof)
    for dofs, fe in svk_chunks(u, "f"):
        f += np.bincount(dofs, fe.ravel(), minlength=ndof)
    return f

def svk_K(u):
    r_, c_, v_ = [], [], []
    for dofs, Ke in svk_chunks(u, "K"):
        d30 = dofs.reshape(-1, 30)
        r_.append(np.repeat(d30, 30, axis=1).ravel().astype(np.int32))
        c_.append(np.tile(d30, (1, 30)).ravel().astype(np.int32))
        v_.append(Ke.reshape(len(d30), 900).ravel())
    return sp.csr_matrix((np.concatenate(v_), (np.concatenate(r_), np.concatenate(c_))),
                         shape=(ndof, ndof))

def potential(u, dlt):
    """W + eps/2 int pen^2 - F delta: stationary in u AND delta exactly at equilibrium
    under the service load, since d/d(delta) of the penalty term is the contact force."""
    pen = contact_state(u[cdof], dlt)[1]
    return svk_W(u) + 0.5 * EPS_N * np.sum(pen * pen * wq) - F_TARGET * dlt

svk_hist, t_svk = [], 0.0
if a.kinematics == "svk":
    t0 = time.time()
    delta_lin = delta
    rhs = np.zeros(len(free)); rhs[cpos] = contact_state(uc, delta)[2]
    u = np.zeros(ndof); u[free] = solver.solve(Kup, rhs)
    solver.free_memory(everything=True); del S
    # frame indifference, the check a linear kernel would also pass for a SMALL rotation:
    # a 30-degree rigid rotation must produce no force at all
    th = np.radians(30.0); R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    f_rot = np.abs(svk_fint((X @ R.T - X).ravel())).max()
    f_lin = np.abs(K @ u).max()
    assert f_rot < 1e-9 * f_lin, ("SVK force under a rigid rotation", f_rot, f_lin)
    # it must be the linear force to first order: halving u quarters the gap to K u
    e1 = np.abs(svk_fint(1e-3 * u) - 1e-3 * (K @ u)).max()
    e2 = np.abs(svk_fint(5e-4 * u) - 5e-4 * (K @ u)).max()
    assert 3.9 < e1 / e2 < 4.1, ("SVK force is not K u + O(u^2)", e1, e2)
    # and force, tangent and energy must be one function: central differences at u_lin.
    # The error is truncation (the energy is quartic): on a 445k-DOF dev mesh it read
    # 4.06e-1, 4.06e-3, 4.06e-5, 4.06e-7 (tangent) at hh 1e-2 .. 1e-5 -- exactly h^2.
    v = np.random.default_rng(0).standard_normal(ndof) * 1e-2; v[fixed] = 0.0
    hh = 1e-5
    fd_f = (svk_fint(u + hh * v) - svk_fint(u - hh * v)) / (2 * hh)
    Kt = svk_K(u)
    e_K = np.abs(fd_f - Kt @ v).max() / np.abs(Kt @ v).max()
    e_W = abs((svk_W(u + hh * v) - svk_W(u - hh * v)) / (2 * hh) - svk_fint(u) @ v) / abs(svk_fint(u) @ v)
    assert e_K < 1e-5 and e_W < 1e-5, ("SVK tangent/energy inconsistent", e_K, e_W)
    print(f'SVK checks: rigid 30 deg {f_rot:.2e} (K u {f_lin:.2e}); O(u^2) ratio {e1 / e2:.4f}; '
          f'FD tangent {e_K:.1e}, FD energy {e_W:.1e}', flush=True)
    del fd_f, v
    Ef = sp.csr_matrix((np.ones(m), (cpos, np.arange(m))), shape=(len(free), m))   # E_C
    for it in range(40):
        fint = svk_fint(u)
        g, pen, fc, act = contact_state(u[cdof], delta)
        Rf = fint[free] - Ef @ fc
        gd = fc.sum() - F_TARGET
        res = np.sqrt(Rf @ Rf + gd * gd) / F_TARGET
        if it > 0 and res < a.tol:
            why = "residual"; svk_hist.append((it, float(res), float(delta), 0.0, 0.0, 0)); break
        Ba = B[act]
        Kc = EPS_N * (Ba.T @ sp.diags(wq[act]) @ Ba)                    # m x m
        bd = EPS_N * (Ba.T @ wq[act]); sd = EPS_N * wq[act].sum()
        A = (Kt if it == 0 else svk_K(u))[free][:, free] + Ef @ Kc @ Ef.T
        Aup = sp.triu(A, format="csr"); del A
        slv = pypardiso.PyPardisoSolver(mtype=2)
        slv.factorize(Aup)
        x = slv.solve(Aup, np.c_[-Rf, Ef @ bd])
        slv.free_memory(everything=True); del Aup, slv
        x1, x2 = x[:, 0], x[:, 1]
        dd = (-gd + bd @ x1[cpos]) / (sd - bd @ x2[cpos])
        du = x1 + dd * x2
        # energy backtracking (Armijo), as wheel_fem.solve_nonlinear does
        P0 = potential(u, delta); slope = Rf @ du + gd * dd
        assert slope < 0, ("not a descent direction", slope)
        # solve_nonlinear's second criterion: the residual norm has a roundoff floor that
        # grows with the mesh (twin 7.5 clamped sat at 1.9e-9 for 35 steps, the drop fixed
        # to 1e-12 from step 4), the energy increment does not.  The step it fires on is
        # still TAKEN: it is a quadratic Newton step, and stopping before it left the dev
        # mesh's drop 1.4e-9 short
        dE0 = -slope if it == 0 else dE0
        last = -slope <= 1e-14 * dE0
        alpha, nb = 1.0, 0
        while True:
            ut = u.copy(); ut[free] += alpha * du
            if potential(ut, delta + alpha * dd) <= P0 + 1e-4 * alpha * slope or nb >= 20:
                break
            alpha *= 0.5; nb += 1
        u, delta = ut, delta + alpha * dd
        svk_hist.append((it, float(res), float(delta), float(alpha * dd),
                         float(alpha * np.abs(du).max()), nb))
        print(f'  svk {it:2d}  |R|/F {res:.3e}  drop {delta:.12f}  d(drop) {alpha * dd:+.3e}  '
              f'max|du| {alpha * np.abs(du).max():.3e}  alpha {alpha:g}  active {act.sum()}  '
              f'({time.time() - t0:.0f}s)', flush=True)
        if last:
            why = "energy"; break
    else:
        raise RuntimeError("SVK Newton did not converge")
    print(f'  svk {it:2d}  |R|/F {res:.3e}  converged ({why})', flush=True)
    uc = u[cdof]
    t_svk = time.time() - t0

# ---- recover the full field and report ---------------------------------------------
g, pen, fc, act = contact_state(uc, delta)
# the patch must sit strictly inside the candidate window, or it was truncated
xe0, xe1 = xq.min(), xq.max()
assert xq[act].min() > xe0 + 0.1 and xq[act].max() < xe1 - 0.1, ('PATCH TRUNCATED', xq[act].min(), xq[act].max(), xe0, xe1)
if a.kinematics == "linear":
    rhs = np.zeros(len(free)); rhs[cpos] = fc
    uf = solver.solve(Kup, rhs)
    u = np.zeros(ndof); u[free] = uf
U = u.reshape(-1, 3)
assert np.abs(U[cnodes, 1] - uc).max() < 1e-9 * np.abs(uc).max()
reac = (K @ u) if a.kinematics == "linear" else svk_fint(u)
hub_fy = reac[3 * TIE + 1].sum()
lowest = cnodes[np.argmin(X[cnodes, 1])]

# von Mises over every element whose centroid is within the patch neighbourhood
cen = X[T[:, :4]].mean(1)
near = np.where((cen[:, 0] > a.x0 - 1.0) & (cen[:, 0] < a.x1 + 1.0) & (cen[:, 1] < -45.0))[0]
G, wd = element_grads(near)
gu = np.einsum("eai,eqaj->eqij", U[T[near]], G)
eps = 0.5 * (gu + np.swapaxes(gu, -1, -2))
sig = LAM * np.trace(eps, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) + 2 * MU * eps
if a.kinematics == "svk":             # Cauchy: F S F^T / det F, as `wheel_fem` reports it
    F = gu + np.eye(3); Eg = 0.5 * (np.swapaxes(F, -1, -2) @ F - np.eye(3))
    Sg = LAM * np.trace(Eg, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) + 2 * MU * Eg
    sig = F @ Sg @ np.swapaxes(F, -1, -2) / np.linalg.det(F)[..., None, None]
dev = sig - np.trace(sig, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
vm = np.sqrt(1.5 * np.einsum("eqij,eqij->eq", dev, dev))
xqp = np.einsum("qa,eai->eqi", QN, X[T[near]])
kmax = np.unravel_index(np.argmax(vm), vm.shape)

rep = dict(
    m_columns=int(have.sum()),
    mesh=a.mesh, faces=a.faces, n_tet=len(T), n_nodes=N, ndof=ndof, n_free=len(free),
    volume_half_mm3=vol, m_candidates=m, n_contact_tris=len(ctri),
    axle_drop_mm=delta, lowest_node_rise_mm=float(U[lowest, 1]),
    max_penetration_mm=float(pen.max()), contact_force_half_n=float(fc.sum()),
    hub_reaction_half_fy_n=float(hub_fy),
    patch_x_mm=[float(xq[act].min()), float(xq[act].max())],
    patch_z_mm=[float(zq[act].min()), float(zq[act].max())],
    peak_pressure_mpa=float(EPS_N * pen.max()),
    vm_max_mpa=float(vm.max()), vm_max_at=xqp[kmax].tolist(),
    secant=hist, t_assemble=t_asm, t_factor=t_fact, t_condense=t_cond, t_secant=t_sec,
    t_total=time.time() - t_start)
if a.kinematics == "svk":
    rep.update(kinematics="svk", axle_drop_linear_mm=delta_lin,
               svk_ratio=delta / delta_lin, svk_passes=svk_hist, svk_converged=why, t_svk=t_svk)
for k, v in rep.items():
    print(f"  {k:24s} {v}")
np.savez(a.out, u=u, rep=np.array([repr(rep)]))
