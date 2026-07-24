"""
=============================================================================
  COMPLIANT WHEEL — PLANE-STRESS FINITE ELEMENT KERNEL
=============================================================================
One spoke, isoparametric quads, direct sparse solve.

    prob = spoke_problem(genes, cfg="coarse", bc="fixed_guided")
    res  = solve_linear(prob)
    res["deflection_mm"]        # tip motion along the load direction

WRITE THE ENERGY, DERIVE EVERYTHING ELSE
----------------------------------------
`element_energy` is the ONLY physics in this file.  The internal force is its
gradient and the tangent stiffness is its Hessian, both produced by `jax.grad` /
`jax.hessian` and `vmap`ped over elements:

    f_e = grad(Pi_e)     K_e = hessian(Pi_e)

There is no B-matrix, no hand-differentiated stress recovery, and no separately
coded tangent that can drift out of step with the residual.  That is where
hand-rolled nonlinear FEA usually dies, and the whole class of bug is simply
absent.  It also makes the rigid-body test exact by construction rather than by
cancellation: if the energy is frame-indifferent, so is everything derived from it.

The cost is one Hessian per element per assembly.  At the `fine` config that is
4096 elements x an 18x18 dense block, which `vmap` does in milliseconds — the
sparse factorization dominates by orders of magnitude.

KINEMATICS: BOTH, FROM DAY ONE
------------------------------
`kinematics="linear"` uses the engineering strain eps = sym(grad u);
`kinematics="svk"` uses the Green-Lagrange strain E = (F^T F - I)/2 with the same
strain-energy function.  The two differ by five lines and coincide to O(|grad u|^2),
so shipping both now costs nothing and lets the finite-rotation test be run against
BOTH kernels.  That matters: a rigid-body test that passes for the linear kernel too
is not testing frame indifference, it is testing that the rotation was small.
M3 verifies with `linear`; M5 switches the default and adds the Newton loop.

PLANE STRESS, AND WHY IT IS THE RIGHT CHOICE *FOR VALIDATION*
-------------------------------------------------------------
Plane stress with the Lame constant lam = E*nu/(1-nu^2) reproduces Euler-Bernoulli
bending exactly in the slender limit: the beam solution sigma_yy = 0 satisfies plane
stress identically, so EI = E*w*t^3/12 with PLAIN E — the same E the Castigliano
model uses.  Apples to apples, which is what makes the A1-A4 comparisons a check
rather than an argument.

The real part is a different question.  The spoke is 22.4 mm wide and 2 mm thick
(aspect 11); a wide beam suppresses anticlastic curvature and behaves closer to
plane STRAIN, i.e. an effective modulus of E/(1-nu^2) = 1.14*E.  That 14% is larger
than most effects this project is chasing, so `plane="strain"` exists and which one
was used is recorded in the result dict.  Do not let it be picked silently.

BOUNDARY CONDITIONS
-------------------
The tip cross-section is a 3-DOF RIGID BODY (Ux, Uy, Theta) with every tip node tied
to it.  That is not a convenience: beam theory assumes plane sections remain plane
and normal, and a rigid end section enforces exactly that assumption, so the FE model
and the beam model are answering the same question.  It also removes the "how was the
tip traction distributed?" ambiguity entirely — the load is a point force on the
rigid body.

  bc="cantilever"    all three rigid DOF free            (legacy free-tip model)
  bc="fixed_guided"  Theta = 0 and the transverse DOF = 0; the load-direction DOF
                     is free.  Models a spoke fused into a stiff rim ring.

These mirror `wheel_fea.generalized_spoke_mechanics` exactly, including its two
redundants: M0 (end moment) is the reaction of Theta = 0 and H (tangential force) is
the reaction of the transverse constraint.

The ROOT has two treatments, and the difference is not cosmetic:

  root_bc="clamped"  u = 0 on every root node.  Simple and conservative, but it also
                     forbids the lateral Poisson strain eps_yy = -nu*M*y/(EI) that
                     the bending field genuinely has there.  That is a self-
                     equilibrated end perturbation decaying over ~t, and its energy
                     is O(t/L) relative to the beam's — a FIRST-order contamination
                     of a comparison whose whole point is to measure an O(t^2) effect.
  root_bc="plane"    plane sections remain plane (displacement along the root tangent
                     is zero at every root node) and the centerline node is pinned,
                     but the section may stretch laterally.  This is precisely what
                     Euler-Bernoulli assumes and nothing more.

A4 is reported under both.  If the two disagree about the exponent, the root
treatment was the thing being measured.
=============================================================================
"""

import jax_config  # noqa: F401  — must precede every other jax import
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import wheel_mesh as _mesh
from wheel_fea import (
    HUB_RIM_SPAN_MM,
    SPOKE_WIDTH_MM,
    YOUNGS_MODULUS_PLA_MPA,
    FORCE_PER_SPOKE_NEWTONS,
)

POISSON_RATIO_PLA = 0.35


# ---------------------------------------------------------------------------
# SHAPE FUNCTIONS
# ---------------------------------------------------------------------------
#
# The (xi, eta) node table below MUST match `wheel_mesh.spoke_block_connectivity`'s
# vertex ordering.  A permutation here would still pass the patch test and still give
# a symmetric positive-definite K — it would just quietly model a different element.
# `tests/test_fem.py::test_node_table_matches_the_mesh_connectivity` pins the pairing
# by rebuilding the ordering from the mesh module's own index arithmetic.
#
# Local 1D node index == the local grid offset, for both orders:
#   order 1: offsets {0,1}   -> xi in {-1, +1}
#   order 2: offsets {0,1,2} -> xi in {-1, 0, +1}
_NODE_IJ = {
    1: np.array([(0, 0), (1, 0), (1, 1), (0, 1)]),
    2: np.array([(0, 0), (2, 0), (2, 2), (0, 2),
                 (1, 0), (2, 1), (1, 2), (0, 1),
                 (1, 1)]),
}


def _lagrange_1d(order, x):
    """Value and derivative of the 1D Lagrange basis at `x`, as [n, ...] arrays."""
    if order == 1:
        n = np.array([(1.0 - x) / 2.0, (1.0 + x) / 2.0])
        d = np.array([-0.5, 0.5])
    else:
        n = np.array([x * (x - 1.0) / 2.0, 1.0 - x * x, x * (x + 1.0) / 2.0])
        d = np.array([x - 0.5, -2.0 * x, x + 0.5])
    return n, d


def _gauss_1d(order):
    """Gauss-Legendre rule that integrates the element stiffness exactly on an
    undistorted element: 2 points for Q4, 3 for Q9.

    Full integration on purpose.  Reduced integration is the usual dodge for Q4 shear
    locking, but it buys hourglass modes — and the fix for locking here is Q9, which
    the mesh module already defaults to.
    """
    if order == 1:
        p = np.array([-1.0, 1.0]) / np.sqrt(3.0)
        w = np.array([1.0, 1.0])
    else:
        p = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    return p, w


def _element_tables(order):
    """Shape-function values and reference gradients at every Gauss point.

    These depend only on the element order, never on the genes, so they are numpy
    constants evaluated once and closed over by the traced element energy.

    Returns N [ngp, nen], dN [ngp, nen, 2], w [ngp].
    """
    gp, gw = _gauss_1d(order)
    ij = _NODE_IJ[order]
    a, b = ij[:, 0], ij[:, 1]
    N, dN, W = [], [], []
    for xi, wxi in zip(gp, gw):
        nx, dx = _lagrange_1d(order, xi)
        for eta, weta in zip(gp, gw):
            ny, dy = _lagrange_1d(order, eta)
            N.append(nx[a] * ny[b])
            dN.append(np.stack([dx[a] * ny[b], nx[a] * dy[b]], axis=1))
            W.append(wxi * weta)
    return np.asarray(N), np.asarray(dN), np.asarray(W)


_TABLES = {order: _element_tables(order) for order in (1, 2)}


# ---------------------------------------------------------------------------
# CONSTITUTIVE
# ---------------------------------------------------------------------------

def lame(E, nu, plane="stress"):
    """(lambda, mu) for the 2D strain-energy density.

    Plane stress uses the reduced lambda = E*nu/(1-nu^2), which is what makes a
    uniaxial state give sigma = E*eps exactly and hence makes beam bending give
    EI = E*I with plain E.
    """
    mu = E / (2.0 * (1.0 + nu))
    if plane == "stress":
        lam = E * nu / (1.0 - nu * nu)
    elif plane == "strain":
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    else:
        raise ValueError(f"plane must be 'stress' or 'strain', got {plane!r}")
    return lam, mu


def _strain(grad_u, nonlinear):
    """Engineering strain, or Green-Lagrange if `nonlinear`.

    Green-Lagrange is exactly frame-indifferent: under u = (R - I)x it gives
    E = (R^T R - I)/2 = 0 identically, so a rigid rotation of any magnitude stores
    zero energy.  The linear strain does not, which is the whole content of the
    finite-rotation test.
    """
    if nonlinear:
        F = grad_u + jnp.eye(2)
        return 0.5 * (F.T @ F - jnp.eye(2))
    return 0.5 * (grad_u + grad_u.T)


def _energy_density(eps, lam, mu):
    """St. Venant-Kirchhoff: W = lam/2 tr(eps)^2 + mu eps:eps.

    With the linear strain this IS linear elasticity; with Green-Lagrange it is SVK.
    One function, so the two kinematics cannot disagree about the material.
    """
    tr = eps[0, 0] + eps[1, 1]
    return 0.5 * lam * tr * tr + mu * jnp.sum(eps * eps)


# ---------------------------------------------------------------------------
# ELEMENT
# ---------------------------------------------------------------------------

def element_energy(Xe, ue, lam, mu, width, dN, w, nonlinear):
    """Total strain energy of one element.  Xe, ue are [nen, 2]; scalar out.

    `width` is the out-of-plane face width, which enters as a single multiplicative
    factor and never touches the kinematics.
    """
    # J[g,k,i] = dx_i/dxi_k, i.e. the TRANSPOSE of the usual Jacobian matrix.  So
    # dxi_k/dx_i is inv(J)[i,k] and not inv(J)[k,i] — the index order below is
    # load-bearing.  Getting it backwards is invisible on an axis-aligned rectangular
    # element (J is diagonal) and silently wrong on every curved or rotated one, which
    # is exactly the geometry this project meshes.  The distorted-mesh patch test is
    # what catches it.
    J = jnp.einsum("gnk,ni->gki", dN, Xe)
    detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    Jinv = jnp.linalg.inv(J)
    dNdx = jnp.einsum("gnk,gik->gni", dN, Jinv)
    grad_u = jnp.einsum("ni,gnj->gij", ue, dNdx)

    eps = jax.vmap(_strain, in_axes=(0, None))(grad_u, nonlinear)
    W = jax.vmap(_energy_density, in_axes=(0, None, None))(eps, lam, mu)
    return jnp.sum(W * detJ * w) * width


def _element_kernels(order, nonlinear):
    """vmapped (energy, force, stiffness) over elements, jitted.

    Cached per (order, nonlinear) so the trace happens once per process rather than
    once per solve — which matters a lot inside the Stage-3 optimizer loop.
    """
    key = (order, nonlinear)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    _, dN, w = _TABLES[order]
    dN_j, w_j = jnp.asarray(dN), jnp.asarray(w)

    def one(Xe, ue, lam, mu, width):
        return element_energy(Xe, ue, lam, mu, width, dN_j, w_j, nonlinear)

    axes = (0, 0, None, None, None)
    energy = jax.jit(jax.vmap(one, in_axes=axes))
    force = jax.jit(jax.vmap(jax.grad(one, argnums=1), in_axes=axes))
    stiff = jax.jit(jax.vmap(jax.hessian(one, argnums=1), in_axes=axes))
    _KERNEL_CACHE[key] = (energy, force, stiff)
    return _KERNEL_CACHE[key]


_KERNEL_CACHE = {}


# ---------------------------------------------------------------------------
# ASSEMBLY
# ---------------------------------------------------------------------------

def _edof(conn):
    """[n_elem, 2*nen] global DOF index per element.  DOF of node n is (2n, 2n+1)."""
    return (np.asarray(conn)[:, :, None] * 2 + np.arange(2)[None, None, :]) \
        .reshape(len(conn), -1)


def assemble_stiffness(coords, conn, u=None, *, order, lam, mu, width,
                       nonlinear=False):
    """Global tangent stiffness as a CSR matrix, [2N, 2N].

    For the linear kernel the Hessian is independent of `u`, so `u=None` (evaluate at
    zero) is exact rather than an approximation.
    """
    coords = jnp.asarray(coords)
    conn_np = np.asarray(conn)
    Xe = coords[conn_np]
    ue = jnp.zeros_like(Xe) if u is None else jnp.asarray(u).reshape(-1, 2)[conn_np]

    _, _, stiff = _element_kernels(order, nonlinear)
    nen = conn_np.shape[1]
    Ke = np.asarray(stiff(Xe, ue, lam, mu, width)).reshape(len(conn_np), 2 * nen,
                                                           2 * nen)
    ed = _edof(conn_np)
    rows = np.repeat(ed, 2 * nen, axis=1).ravel()
    cols = np.tile(ed, (1, 2 * nen)).ravel()
    n_dof = 2 * coords.shape[0]
    return sp.coo_matrix((Ke.ravel(), (rows, cols)),
                         shape=(n_dof, n_dof)).tocsr()


def internal_force(coords, conn, u, *, order, lam, mu, width, nonlinear=False):
    """Global internal force vector, [2N].  Zero at u=0 for both kinematics."""
    coords = jnp.asarray(coords)
    conn_np = np.asarray(conn)
    Xe = coords[conn_np]
    ue = jnp.asarray(u).reshape(-1, 2)[conn_np]
    _, force, _ = _element_kernels(order, nonlinear)
    fe = np.asarray(force(Xe, ue, lam, mu, width))          # [ne, nen, 2]
    out = np.zeros(2 * coords.shape[0])
    np.add.at(out, _edof(conn_np).ravel(), fe.reshape(-1))
    return out


def total_energy(coords, conn, u, *, order, lam, mu, width, nonlinear=False):
    """Total strain energy, a scalar.  Used by the rigid-body test."""
    coords = jnp.asarray(coords)
    conn_np = np.asarray(conn)
    energy, _, _ = _element_kernels(order, nonlinear)
    ue = jnp.asarray(u).reshape(-1, 2)[conn_np]
    return float(jnp.sum(energy(coords[conn_np], ue, lam, mu, width)))


# ---------------------------------------------------------------------------
# CONSTRAINTS
# ---------------------------------------------------------------------------

class DofMap:
    """Master-slave elimination: u_full = T @ u_reduced + u_prescribed.

    Every constraint in this project is linear in the displacements, so a single
    sparse T handles all of them uniformly — homogeneous and inhomogeneous Dirichlet,
    skew (direction-only) constraints, and rigid ties — and the reduced system
    T^T K T is symmetric positive definite whenever the full one is.

    The alternative (penalty springs, or row-and-column zeroing) either perturbs the
    answer by an amount nobody tracks or cannot express a rigid tie at all.

    Every DOF must be claimed exactly once; `finalize` asserts it.  An unclaimed DOF
    is a free floating node, which produces a singular matrix; a doubly-claimed one
    silently drops the first constraint.
    """

    def __init__(self, n_nodes):
        self.n_nodes = int(n_nodes)
        self.n_full = 2 * self.n_nodes
        self.u_pre = np.zeros(self.n_full)
        self.named = {}
        self._claimed = np.zeros(self.n_full, dtype=bool)
        self._rows, self._cols, self._vals = [], [], []
        self._n_red = 0

    # -- internals ---------------------------------------------------------
    def _claim(self, dofs):
        dofs = np.atleast_1d(dofs)
        if self._claimed[dofs].any():
            bad = dofs[self._claimed[dofs]]
            raise ValueError(f"DOF {bad.tolist()} constrained twice")
        self._claimed[dofs] = True

    def _new_reduced(self, name=None):
        idx = self._n_red
        self._n_red += 1
        if name is not None:
            self.named[name] = idx
        return idx

    def _entry(self, full_dof, red_dof, value):
        self._rows.append(int(full_dof))
        self._cols.append(int(red_dof))
        self._vals.append(float(value))

    # -- constraint types --------------------------------------------------
    def free(self, node_ids):
        """Unconstrained nodes: each gets its own two reduced DOF."""
        for n in np.atleast_1d(node_ids):
            dofs = [2 * n, 2 * n + 1]
            self._claim(dofs)
            for d in dofs:
                self._entry(d, self._new_reduced(), 1.0)

    def fix(self, node_ids, values=(0.0, 0.0)):
        """Prescribe both components of a node.  `values` may be per-node [n, 2]."""
        node_ids = np.atleast_1d(node_ids)
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = np.tile(values, (len(node_ids), 1))
        for n, v in zip(node_ids, values):
            self._claim([2 * n, 2 * n + 1])
            self.u_pre[2 * n] = v[0]
            self.u_pre[2 * n + 1] = v[1]

    def constrain_direction(self, node_ids, dhat):
        """Enforce u . dhat = 0, leaving motion along the perpendicular free.

        One reduced DOF per node: the amplitude along dhat_perp.  This is how the
        'plane sections remain plane, but may stretch laterally' root condition is
        expressed without rotating the whole system into a local frame.
        """
        d = np.asarray(dhat, dtype=float)
        if d.ndim == 1:
            d = np.tile(d, (len(np.atleast_1d(node_ids)), 1))
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        perp = np.stack([-d[:, 1], d[:, 0]], axis=1)
        for n, p in zip(np.atleast_1d(node_ids), perp):
            self._claim([2 * n, 2 * n + 1])
            r = self._new_reduced()
            self._entry(2 * n, r, p[0])
            self._entry(2 * n + 1, r, p[1])

    def rigid(self, node_ids, ref_xy, coords, free=("ux", "uy", "th"), prefix="rb"):
        """Tie nodes to a 3-DOF rigid body about `ref_xy`.

            u_x = Ux - Theta * (y - yc)
            u_y = Uy + Theta * (x - xc)

        Constrained components are simply omitted from T, which pins them at zero.
        Returns the dict of reduced indices for the free rigid DOF.
        """
        node_ids = np.atleast_1d(node_ids)
        red = {k: self._new_reduced(f"{prefix}_{k}") for k in ("ux", "uy", "th")
               if k in free}
        xc, yc = float(ref_xy[0]), float(ref_xy[1])
        for n in node_ids:
            self._claim([2 * n, 2 * n + 1])
            dx = float(coords[n, 0]) - xc
            dy = float(coords[n, 1]) - yc
            if "ux" in red:
                self._entry(2 * n, red["ux"], 1.0)
            if "uy" in red:
                self._entry(2 * n + 1, red["uy"], 1.0)
            if "th" in red:
                self._entry(2 * n, red["th"], -dy)
                self._entry(2 * n + 1, red["th"], dx)
        return red

    def finalize(self):
        """Sparse T, [n_full, n_reduced].  Raises if any DOF went unclaimed."""
        if not self._claimed.all():
            missing = np.flatnonzero(~self._claimed)
            raise ValueError(
                f"{missing.size} DOF were never constrained or freed "
                f"(first few: {missing[:6].tolist()}); K would be singular"
            )
        return sp.coo_matrix((self._vals, (self._rows, self._cols)),
                             shape=(self.n_full, self._n_red)).tocsr()


# ---------------------------------------------------------------------------
# PROBLEM DEFINITION AND SOLVE
# ---------------------------------------------------------------------------

class Problem:
    """A fully specified linear FE problem: mesh, material, constraints, load."""

    __slots__ = ("coords", "conn", "order", "lam", "mu", "width", "nonlinear",
                 "T", "u_pre", "dofmap", "f_nodal", "f_reduced", "load_dir",
                 "meta")

    def __init__(self, coords, conn, order, lam, mu, width, dofmap,
                 f_nodal=None, f_reduced=None, load_dir=None, nonlinear=False,
                 meta=None):
        self.coords = np.asarray(coords, dtype=float)
        self.conn = np.asarray(conn)
        self.order = order
        self.lam, self.mu, self.width = float(lam), float(mu), float(width)
        self.nonlinear = bool(nonlinear)
        self.dofmap = dofmap
        self.T = dofmap.finalize()
        self.u_pre = dofmap.u_pre
        n_dof = 2 * self.coords.shape[0]
        self.f_nodal = np.zeros(n_dof) if f_nodal is None else np.asarray(f_nodal)
        self.f_reduced = (np.zeros(self.T.shape[1]) if f_reduced is None
                          else np.asarray(f_reduced))
        self.load_dir = load_dir
        self.meta = dict(meta or {})


def solve_linear(prob):
    """Direct sparse solve of the constrained linear system.

    Returns a dict with the full displacement field, the reduced solution, strain
    energy, an equilibrium residual, and — when the problem carries a rigid tip —
    the tip motion along the load direction.
    """
    K = assemble_stiffness(prob.coords, prob.conn, order=prob.order, lam=prob.lam,
                           mu=prob.mu, width=prob.width, nonlinear=prob.nonlinear)
    T = prob.T
    Kr = (T.T @ K @ T).tocsc()
    fr = T.T @ (prob.f_nodal - K @ prob.u_pre) + prob.f_reduced
    u_red = spla.spsolve(Kr, fr)
    u_full = T @ u_red + prob.u_pre

    out = {
        "u": u_full,
        "u_reduced": u_red,
        "energy": 0.5 * float(u_full @ (K @ u_full)),
        # Equilibrium in the reduced (unconstrained) space.  A nonzero value here is
        # a solver failure, not a reaction: reactions live in the eliminated DOF.
        "residual": float(np.linalg.norm(Kr @ u_red - fr)),
        "residual_rel": float(np.linalg.norm(Kr @ u_red - fr)
                              / max(np.linalg.norm(fr), 1e-300)),
        "n_dof_reduced": int(Kr.shape[0]),
        "meta": prob.meta,
    }

    named = prob.dofmap.named
    if prob.load_dir is not None and ("tip_ux" in named or "tip_uy" in named):
        dx, dy = prob.load_dir
        disp = 0.0
        if "tip_ux" in named:
            disp += dx * u_red[named["tip_ux"]]
        if "tip_uy" in named:
            disp += dy * u_red[named["tip_uy"]]
        # Positive = the tip moved WITH the load.
        out["deflection_mm"] = float(disp)
        out["tip"] = {k: float(u_red[v]) for k, v in named.items()
                      if k.startswith("tip_")}
    return out


# ---------------------------------------------------------------------------
# STRESS RECOVERY
# ---------------------------------------------------------------------------

def gauss_stresses(coords, conn, u, *, order, lam, mu, nonlinear=False):
    """Cauchy (linear) / 2nd Piola-Kirchhoff (SVK) stress at every Gauss point.

    Returns dict with `sigma` [n_elem, ngp, 2, 2], `von_mises` [n_elem, ngp], and the
    Gauss points' physical coordinates.  Von Mises is the plane-stress form
    sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2), which assumes sigma_zz = 0 — true by
    construction under plane stress and NOT true under plane strain, where the
    reported value is an underestimate.
    """
    N, dN, _ = _TABLES[order]
    coords = jnp.asarray(coords)
    conn_np = np.asarray(conn)
    Xe = coords[conn_np]
    ue = jnp.asarray(u).reshape(-1, 2)[conn_np]
    dN_j = jnp.asarray(dN)

    def per_elem(Xe1, ue1):
        J = jnp.einsum("gnk,ni->gki", dN_j, Xe1)
        dNdx = jnp.einsum("gnk,gik->gni", dN_j, jnp.linalg.inv(J))
        grad_u = jnp.einsum("ni,gnj->gij", ue1, dNdx)
        eps = jax.vmap(_strain, in_axes=(0, None))(grad_u, nonlinear)
        tr = eps[:, 0, 0] + eps[:, 1, 1]
        return lam * tr[:, None, None] * jnp.eye(2)[None] + 2.0 * mu * eps

    sigma = np.asarray(jax.jit(jax.vmap(per_elem))(Xe, ue))
    sxx, syy, sxy = sigma[..., 0, 0], sigma[..., 1, 1], sigma[..., 0, 1]
    vm = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)
    xy = np.asarray(jnp.einsum("gn,eni->egi", jnp.asarray(N), Xe))
    return {"sigma": sigma, "von_mises": vm, "xy": xy}


# ---------------------------------------------------------------------------
# THE SINGLE-SPOKE PROBLEM
# ---------------------------------------------------------------------------

def spoke_coords(genes, cfg, span_mm=HUB_RIM_SPAN_MM):
    """Flat [n_nodes, 2] node coordinates for one spoke block, plus conn and sets."""
    cfg = _mesh.get_config(cfg)
    grid = _mesh.spoke_block_coords_from_vector(np.asarray(genes, dtype=float),
                                                cfg, span_mm=span_mm, xp=np)
    return (_mesh.flatten(np.asarray(grid)),
            _mesh.spoke_block_connectivity(cfg),
            _mesh.boundary_nodes(cfg))


def spoke_problem(genes, cfg="coarse", *, bc="fixed_guided", root_bc="clamped",
                  load_dir=(-1.0, 0.0), force=FORCE_PER_SPOKE_NEWTONS,
                  E=YOUNGS_MODULUS_PLA_MPA, nu=POISSON_RATIO_PLA, plane="stress",
                  width=SPOKE_WIDTH_MM, span_mm=HUB_RIM_SPAN_MM,
                  kinematics="linear", coords=None):
    """Build the single-spoke problem that mirrors `generalized_spoke_mechanics`.

    `load_dir` must be axis aligned: the guided constraint restrains the rigid tip's
    OTHER translation, and expressing that for an oblique direction would need a
    rotated rigid DOF for no benefit here (the service load is radial = -x, and the
    straight-beam checks load transversely = -y).

    `coords` overrides the generated mesh, which is what the distorted-mesh patch test
    and the mesh-perturbation studies use.
    """
    cfg = _mesh.get_config(cfg)
    gen_coords, conn, bnd = spoke_coords(genes, cfg, span_mm=span_mm)
    coords = gen_coords if coords is None else np.asarray(coords, dtype=float)

    d = np.asarray(load_dir, dtype=float)
    d = d / np.linalg.norm(d)
    if not (np.isclose(abs(d[0]), 1.0) or np.isclose(abs(d[1]), 1.0)):
        raise ValueError(f"load_dir must be axis aligned, got {load_dir}")
    along = "ux" if abs(d[0]) > 0.5 else "uy"

    lam, mu = lame(E, nu, plane)
    dm = DofMap(coords.shape[0])

    root, tip = bnd["root"], bnd["tip"]
    if set(root) & set(tip):
        raise ValueError("root and tip node sets overlap; n_span is too small")

    # --- root ---
    if root_bc == "clamped":
        dm.fix(root)
    elif root_bc == "plane":
        # Tangent of the root cross-section's normal, i.e. the centerline direction
        # there.  Taken from the meshed geometry so it stays correct if `coords` was
        # overridden.
        seam = coords[root]
        across_vec = seam[-1] - seam[0]                 # along the cross-section
        tangent = np.array([-across_vec[1], across_vec[0]])
        tangent /= np.linalg.norm(tangent)
        mid = root[len(root) // 2]
        others = np.array([n for n in root if n != mid])
        dm.constrain_direction(others, tangent)
        dm.fix([mid])
    else:
        raise ValueError(f"root_bc must be 'clamped' or 'plane', got {root_bc!r}")

    # --- tip: a rigid cross-section, per the docstring ---
    tip_ref = coords[tip].mean(axis=0)
    if bc == "cantilever":
        free = ("ux", "uy", "th")
    elif bc == "fixed_guided":
        free = (along,)
    else:
        raise ValueError(f"bc must be 'cantilever' or 'fixed_guided', got {bc!r}")
    red = dm.rigid(tip, tip_ref, coords, free=free, prefix="tip")

    # --- everything else is free ---
    special = set(np.concatenate([root, tip]).tolist())
    dm.free(np.array([n for n in range(coords.shape[0]) if n not in special]))

    T = dm.finalize()
    f_red = np.zeros(T.shape[1])
    f_red[red[along]] = force * d[0 if along == "ux" else 1]

    prob = Problem(coords, conn, cfg.order, lam, mu, width, dm,
                   f_reduced=f_red, load_dir=d,
                   nonlinear=(kinematics == "svk"),
                   meta={"config": cfg.name, "bc": bc, "root_bc": root_bc,
                         "plane": plane, "kinematics": kinematics,
                         "E_mpa": float(E), "nu": float(nu),
                         "width_mm": float(width), "force_n": float(force),
                         "load_dir": [float(d[0]), float(d[1])],
                         "n_elements": int(cfg.n_elements)})
    # `Problem.__init__` calls finalize a second time; DofMap is idempotent under it
    # (it only reads accumulated triplets), so both objects see the same T.
    return prob


def spoke_deflection(genes, cfg="coarse", **kw):
    """Convenience: the tip deflection magnitude, which is what the beam model returns."""
    return abs(solve_linear(spoke_problem(genes, cfg, **kw))["deflection_mm"])
