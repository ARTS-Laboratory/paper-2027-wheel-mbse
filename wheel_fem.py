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
M3 verifies with `linear`; M5 added the Newton loop (`solve_nonlinear`).  The DEFAULT
stays `linear` in `spoke_problem` and `wheel_problem` — M4's committed report and the
beam comparisons are linear results and must stay reproducible — so `svk` is opt-in per
call.  Note that `solve_linear` assembles the tangent at u=0, where the SVK Hessian
equals the linear one exactly; routing an `svk` problem through it therefore returns a
LINEAR answer silently.  Go through `solve`, which dispatches on `prob.nonlinear`.

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
    TOTAL_FORCE_NEWTONS,
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


def _stress_kernel(order, nonlinear, cauchy):
    """vmapped `(Xe, ue, lam, mu) -> sigma[n_elem, ngp, 2, 2]` over elements, jitted.

    Split out of `gauss_stresses` so that the reported stress and the quantity M8's
    optimizer differentiates are THE SAME FUNCTION.  `gauss_stresses` returns numpy and
    takes `u` as data, so it cannot be a QoI; this returns a traced array and can.  The
    alternative — a second von Mises written in `wheel_adjoint.py` — is a constitutive
    law with two definitions, and under SVK the two would differ by the push-forward
    `F S F^T / det F`: a few percent, in the same direction everywhere, i.e. exactly the
    bias that reads as physics rather than as a bug.

    `lam`/`mu` are traced arguments with `in_axes=None`, matching `_element_kernels`, so
    the cache key is the shape-and-branch tuple only and a change of material does not
    re-trace.
    """
    key = (order, nonlinear, cauchy)
    if key in _STRESS_CACHE:
        return _STRESS_CACHE[key]
    _, dN, _ = _TABLES[order]
    dN_j = jnp.asarray(dN)

    def per_elem(Xe1, ue1, lam, mu):
        J = jnp.einsum("gnk,ni->gki", dN_j, Xe1)
        dNdx = jnp.einsum("gnk,gik->gni", dN_j, jnp.linalg.inv(J))
        grad_u = jnp.einsum("ni,gnj->gij", ue1, dNdx)
        eps = jax.vmap(_strain, in_axes=(0, None))(grad_u, nonlinear)
        tr = eps[:, 0, 0] + eps[:, 1, 1]
        s = lam * tr[:, None, None] * jnp.eye(2)[None] + 2.0 * mu * eps
        if not (nonlinear and cauchy):
            return s
        F = grad_u + jnp.eye(2)[None]
        detF = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
        return jnp.einsum("gik,gkl,gjl->gij", F, s, F) / detF[:, None, None]

    _STRESS_CACHE[key] = jax.jit(jax.vmap(per_elem, in_axes=(0, 0, None, None)))
    return _STRESS_CACHE[key]


_STRESS_CACHE = {}


def gauss_volume(coords, conn, *, order, width):
    """Gauss-point volume weights [n_elem, ngp] in mm^3 — `w_g * detJ_g * width`.

    The weights the p-norm stress is averaged with.  An unweighted p-norm over Gauss
    points would score a mesh rather than a wheel: refining the rim would move the answer
    by changing how many points sit in a lightly loaded region.
    """
    return np.asarray(_volume_kernel(order)(jnp.asarray(coords)[np.asarray(conn)])) * width


def _volume_kernel(order):
    if order in _VOLUME_CACHE:
        return _VOLUME_CACHE[order]
    _, dN, w = _TABLES[order]
    dN_j, w_j = jnp.asarray(dN), jnp.asarray(w)

    def one(Xe):
        J = jnp.einsum("gnk,ni->gki", dN_j, Xe)
        detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        return detJ * w_j

    _VOLUME_CACHE[order] = jax.jit(jax.vmap(one))
    return _VOLUME_CACHE[order]


_VOLUME_CACHE = {}


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
# EDGE TRACTIONS  (consistent nodal loads)
# ---------------------------------------------------------------------------
#
# M4 applies a distributed pressure over an assumed contact patch; M6 replaced it with
# penalty contact on the same boundary (see `RigidGroundContact` below, which reuses this
# quadrature).  Both paths are live — M4's committed report is an assumed-patch result —
# so the traction path is not optional and it gets its own patch test.  Lumping a
# traction equally onto the nodes of a
# quadratic edge is wrong by a fixed factor (the correct Q9 edge weights are
# 1/6, 4/6, 1/6, not 1/3 each) and the error is invisible in the total load, which is
# exactly the kind of mistake an equilibrium check does not catch.

def boundary_edge_segments(cfg, side):
    """Node ids of each boundary edge segment, plus the element each belongs to.

    Returns (segments [n_seg, order+1], elements [n_seg]).  The owning element is
    returned so the OUTWARD normal can be determined geometrically — by requiring it
    to point away from the element centroid — instead of by a per-side sign table that
    would have to be re-derived for every new block type.
    """
    cfg = _mesh.get_config(cfg)
    p, ns, nt = cfg.order, cfg.n_node_span, cfg.n_node_thick
    ids = np.arange(ns * nt).reshape(ns, nt)

    if side == "flank_bot":
        line, elems = ids[:, 0], [(ei, 0) for ei in range(cfg.n_span)]
    elif side == "flank_top":
        line, elems = ids[:, -1], [(ei, cfg.n_thick - 1) for ei in range(cfg.n_span)]
    elif side == "root":
        line, elems = ids[0, :], [(0, ej) for ej in range(cfg.n_thick)]
    elif side == "tip":
        line, elems = ids[-1, :], [(cfg.n_span - 1, ej) for ej in range(cfg.n_thick)]
    else:
        raise ValueError(f"unknown side {side!r}")

    segs = np.array([line[k * p:k * p + p + 1] for k in range(len(elems))])
    # Element ordering must match `spoke_block_connectivity`'s `for ei: for ej:` loop.
    elem_ids = np.array([ei * cfg.n_thick + ej for ei, ej in elems])
    return segs, elem_ids


def edge_traction_load(coords, conn, cfg, side, traction, width=SPOKE_WIDTH_MM):
    """Consistent nodal load vector [2N] from a traction on one boundary side.

    `traction` is either a constant (tx, ty) or a callable `f(x, n) -> (tx, ty)` given
    the Gauss point's physical position and the OUTWARD unit normal there — the
    callable form is what a pressure (`-p * n`) or a stress state (`sigma @ n`) needs.
    """
    cfg = _mesh.get_config(cfg)
    coords = np.asarray(coords, dtype=float)
    conn = np.asarray(conn)
    segs, elem_ids = boundary_edge_segments(cfg, side)
    gp, gw = _gauss_1d(cfg.order)
    out = np.zeros(2 * coords.shape[0])
    const = None if callable(traction) else np.asarray(traction, dtype=float)

    for seg, eid in zip(segs, elem_ids):
        X = coords[seg]                                  # [p+1, 2]
        centroid = coords[conn[eid][:4]].mean(axis=0)
        for xi, w in zip(gp, gw):
            N, dN = _lagrange_1d(cfg.order, xi)
            dx = dN @ X                                  # dx/dxi
            jac = np.linalg.norm(dx)
            n = np.array([dx[1], -dx[0]]) / jac
            x = N @ X
            if n @ (x - centroid) < 0:                   # force it outward
                n = -n
            t = const if const is not None else np.asarray(traction(x, n), dtype=float)
            contrib = np.outer(N, t) * (w * jac * width)  # [p+1, 2]
            out[2 * seg] += contrib[:, 0]
            out[2 * seg + 1] += contrib[:, 1]
    return out


def segment_traction_load(coords, segments, traction, n_nodes, order=2,
                          width=SPOKE_WIDTH_MM, inward_point=(0.0, 0.0)):
    """Consistent nodal loads from a traction on an arbitrary set of boundary edges.

    The full-wheel version of `edge_traction_load`: `segments` is [n_seg, order+1] of
    global node ids along a boundary (`WheelMesh.edge_sets`), so no block structure is
    assumed.  The outward normal is disambiguated by requiring it to point AWAY from
    `inward_point`, which for either of the wheel's rings is the origin.

    Same quadrature as the block version, and the same reason it matters: on a quadratic
    edge the correct nodal weights are 1/6, 4/6, 1/6, and lumping equally gives exactly
    the right resultant with the wrong distribution — an error no equilibrium check can
    see.
    """
    coords = np.asarray(coords, dtype=float)
    segments = np.asarray(segments)
    gp, gw = _gauss_1d(order)
    out = np.zeros(2 * int(n_nodes))
    const = None if callable(traction) else np.asarray(traction, dtype=float)
    ref = np.asarray(inward_point, dtype=float)

    for seg in segments:
        X = coords[seg]
        for xi, w in zip(gp, gw):
            N, dN = _lagrange_1d(order, xi)
            dx = dN @ X
            jac = np.linalg.norm(dx)
            n = np.array([dx[1], -dx[0]]) / jac
            x = N @ X
            if n @ (x - ref) < 0:
                n = -n
            t = const if const is not None else np.asarray(traction(x, n), dtype=float)
            contrib = np.outer(N, t) * (w * jac * width)
            out[2 * seg] += contrib[:, 0]
            out[2 * seg + 1] += contrib[:, 1]
    return out


# ---------------------------------------------------------------------------
# PENALTY CONTACT  (M6)
# ---------------------------------------------------------------------------
#
# CONTACT IS AN ENERGY TERM, WHICH IS WHY IT FITS HERE AT ALL.
#
# Everything above derives the internal force and the tangent from one scalar energy.
# A penalty contact law is a potential too,
#
#     Pi_c(u) = integral over Gamma of eps_N * phi(-g(x, u)) dGamma_0,
#     g       = (Y + u_y) - y_ground          (signed gap; negative = penetration)
#     phi(z)  = <z>^2 / 2                     (Macaulay bracket)
#
# so the contact force is grad Pi_c and the contact tangent is its Hessian, both from
# `jax.grad` / `jax.hessian` exactly as for the bulk.  There is no separately coded
# contact stiffness that can drift out of step with the contact force, which is where
# hand-rolled contact usually dies -- the same argument the module docstring makes
# for the bulk element.
#
# INTEGRATED, NOT LUMPED.  The energy is integrated over the boundary with the same
# machinery `segment_traction_load` uses, for the same reason: on a quadratic edge the
# correct nodal weights are 1/6, 4/6, 1/6, and a node-to-surface penalty is the lumped
# version of this.  Lumping gets the resultant right and the distribution wrong, and no
# equilibrium check can see the difference.
#
# The quadrature is DENSER than the traction rule (`n_quad` below, default 6 against the
# 3-point element rule) and that is not gold-plating.  The contact patch edge falls
# *inside* a segment, and the quadrature is the only thing that resolves where: with the
# 3-point rule the edge can only sit at one of three places per segment, which quantises
# the patch width and puts a staircase into any sweep over load or phase.  Gauss points
# are far cheaper here than mesh refinement, since the integrand is 1-D.
#
# Integration is over the REFERENCE boundary (`jac` from X alone), consistent with the
# SVK bulk energy, which is also written on the reference configuration.
#
# ON THE SMOOTHNESS OF phi, WHICH IS AN M7 QUESTION ASKED EARLY.
#
# <z>^2/2 is C^1 but not C^2: phi'' jumps from 0 to 1 as a point comes into contact.
# Newton does not mind -- Pi stays C^1, the tangent stays SPD, and the descent-direction
# guarantee `solve_nonlinear` relies on is untouched -- but a FINITE DIFFERENCE in a gene
# taken across a change in the contact set is meaningless, and that is precisely the
# "gene with no finite-difference plateau" failure the plan gates on at M7.
#
# So `smoothing_mm` offers a C^2 alternative: phi is replaced by a cubic over a band of
# that width, chosen to match value, slope AND curvature at both ends of the band, so it
# is genuinely C^2 rather than merely continuous.  It defaults to 0.0 (sharp), because a
# smoothing band that is not needed is exactly the kind of free parameter M6 exists to
# remove.  `study_contact.py` measures whether it is needed.

def _gauss_1d_n(n):
    """Gauss-Legendre rule with an arbitrary number of points, on [-1, 1]."""
    p, w = np.polynomial.legendre.leggauss(int(n))
    return np.asarray(p, dtype=float), np.asarray(w, dtype=float)


def _contact_tables(order, n_quad):
    """Boundary shape functions at the contact Gauss points.

    Returns N [ngp, p+1], dN [ngp, p+1], w [ngp].  Depends only on (order, n_quad), so
    it is a numpy constant closed over by the traced energy.
    """
    gp, gw = _gauss_1d_n(n_quad)
    N, dN = [], []
    for xi in gp:
        n_, d_ = _lagrange_1d(order, xi)
        N.append(n_)
        dN.append(d_)
    return np.asarray(N), np.asarray(dN), gw


def _penalty_phi(z, smoothing):
    """phi(z) ~ <z>^2 / 2, either sharp (C^1) or cubic-smoothed over a band (C^2).

    The smoothed branch is

        z <= 0      : 0
        0 < z < d   : z^3 / (6d)
        z >= d      : z^2/2 - d*z/2 + d^2/6

    which matches value, slope and curvature at both z=0 and z=d, so phi'' is continuous
    everywhere.  As d -> 0 it recovers the sharp bracket.  Note it is NOT an upper or
    lower bound on it -- it is softer just inside contact and offset by O(d*z) outside,
    which is why `d` has to be small against the penetration scale rather than merely
    small in absolute terms.
    """
    if smoothing <= 0.0:
        pos = jnp.maximum(z, 0.0)
        return 0.5 * pos * pos
    d = smoothing
    return jnp.where(
        z <= 0.0,
        0.0,
        jnp.where(z < d,
                  z ** 3 / (6.0 * d),
                  0.5 * z * z - 0.5 * d * z + d * d / 6.0),
    )


_CONTACT_CACHE = {}


def _contact_kernels(order, n_quad, smoothing):
    """vmapped (energy, force, stiffness) over boundary segments, jitted.

    Cached per (order, n_quad, smoothing) so the trace happens once per process rather
    than once per Newton iteration -- and contact is assembled every iteration, unlike
    the linear bulk tangent.
    """
    key = (order, int(n_quad), float(smoothing))
    if key in _CONTACT_CACHE:
        return _CONTACT_CACHE[key]
    N, dN, w = _contact_tables(order, n_quad)
    N_j, dN_j, w_j = jnp.asarray(N), jnp.asarray(dN), jnp.asarray(w)

    def one(Xs, us, y_ground, eps_n, width):
        xg = N_j @ Xs                       # [ngp, 2] reference position
        ug = N_j @ us                       # [ngp, 2] displacement
        jac = jnp.linalg.norm(dN_j @ Xs, axis=1)
        gap = (xg[:, 1] + ug[:, 1]) - y_ground
        phi = _penalty_phi(-gap, smoothing)
        return eps_n * jnp.sum(phi * jac * w_j) * width

    axes = (0, 0, None, None, None)
    energy = jax.jit(jax.vmap(one, in_axes=axes))
    force = jax.jit(jax.vmap(jax.grad(one, argnums=1), in_axes=axes))
    stiff = jax.jit(jax.vmap(jax.hessian(one, argnums=1), in_axes=axes))
    _CONTACT_CACHE[key] = (energy, force, stiff)
    return _CONTACT_CACHE[key]


class RigidGroundContact:
    """Frictionless penalty contact between a boundary and a rigid horizontal line.

    The ground is `y = y_ground` and pushes UP; the wheel's hub is fixed, so in this
    frame the ground indenting the rim is what the axle drop measures.  Frictionless and
    flat means the traction acts along the GROUND's normal -- vertical everywhere -- which
    is the same choice `_ground_traction` documents and for the same reason: using the
    rim's radial normal instead produces a spurious horizontal resultant.  Here that
    choice is structural rather than a convention, because only `u_y` enters the gap.

    `eps_n` is a penalty stiffness in N/mm per mm^2 of contact area.  It is not a number
    to pick but a plateau to demonstrate -- see `study_contact.py`.
    """

    __slots__ = ("segments", "y_ground", "eps_n", "width", "order", "n_quad",
                 "smoothing_mm")

    def __init__(self, segments, y_ground, eps_n, *, width=SPOKE_WIDTH_MM, order=2,
                 n_quad=6, smoothing_mm=0.0):
        self.segments = np.asarray(segments)
        self.y_ground = float(y_ground)
        self.eps_n = float(eps_n)
        self.width = float(width)
        self.order = int(order)
        self.n_quad = int(n_quad)
        self.smoothing_mm = float(smoothing_mm)

    # -- the three consumers, mirroring the bulk trio ------------------------

    def energy(self, coords, u):
        """Contact potential, a scalar [mJ]."""
        Xs, us = self._gather(coords, u)
        energy, _, _ = _contact_kernels(self.order, self.n_quad, self.smoothing_mm)
        return float(jnp.sum(energy(Xs, us, self.y_ground, self.eps_n, self.width)))

    def force(self, coords, u):
        """Contact contribution to the INTERNAL force, [2N].

        Sign convention matches `internal_force`: this is +grad Pi_c, so it enters the
        residual with the same sign as the bulk internal force.
        """
        Xs, us = self._gather(coords, u)
        _, force, _ = _contact_kernels(self.order, self.n_quad, self.smoothing_mm)
        fe = np.asarray(force(Xs, us, self.y_ground, self.eps_n, self.width))
        out = np.zeros(2 * np.asarray(coords).shape[0])
        np.add.at(out, self._sdof().ravel(), fe.reshape(-1))
        return out

    def stiffness(self, coords, u):
        """Contact contribution to the tangent, [2N, 2N] sparse."""
        Xs, us = self._gather(coords, u)
        _, _, stiff = _contact_kernels(self.order, self.n_quad, self.smoothing_mm)
        nsn = self.segments.shape[1]
        Ke = np.asarray(stiff(Xs, us, self.y_ground, self.eps_n, self.width)) \
            .reshape(len(self.segments), 2 * nsn, 2 * nsn)
        sd = self._sdof()
        rows = np.repeat(sd, 2 * nsn, axis=1).ravel()
        cols = np.tile(sd, (1, 2 * nsn)).ravel()
        n_dof = 2 * np.asarray(coords).shape[0]
        return sp.coo_matrix((Ke.ravel(), (rows, cols)),
                             shape=(n_dof, n_dof)).tocsr()

    # -- diagnostics the gate reads ----------------------------------------

    def pressure(self, coords, u):
        """Contact pressure at every quadrature point, with its position and gap.

        Returned rather than reduced so the gate can check the two things a resultant
        cannot see: that the pressure is nowhere NEGATIVE (a contact model that pulls
        still balances globally), and where the patch edge actually falls.
        """
        coords = np.asarray(coords, dtype=float)
        segs = self.segments
        N, dN, w = _contact_tables(self.order, self.n_quad)
        X = coords[segs]                                   # [nseg, p+1, 2]
        U = u.reshape(-1, 2)[segs]
        xg = np.einsum("gi,sij->sgj", N, X)
        ug = np.einsum("gi,sij->sgj", N, U)
        jac = np.linalg.norm(np.einsum("gi,sij->sgj", dN, X), axis=2)
        gap = (xg[:, :, 1] + ug[:, :, 1]) - self.y_ground
        pen = -gap
        if self.smoothing_mm <= 0.0:
            dphi = np.maximum(pen, 0.0)
        else:
            d = self.smoothing_mm
            dphi = np.where(pen <= 0.0, 0.0,
                            np.where(pen < d, pen ** 2 / (2.0 * d), pen - 0.5 * d))
        return {
            "x": xg[:, :, 0].ravel(),
            "y": xg[:, :, 1].ravel(),
            "theta_deg": np.degrees(np.arctan2(xg[:, :, 1] + ug[:, :, 1],
                                               xg[:, :, 0] + ug[:, :, 0])).ravel(),
            "gap_mm": gap.ravel(),
            "pressure_mpa": (self.eps_n * dphi).ravel(),
            "weight": (jac * w[None, :] * self.width).ravel(),
        }

    def patch_extent(self, coords, u, n_sample=64):
        """Angular extent of the contact patch, from the ZERO CROSSING of the gap.

        Measured independently of the quadrature, and it has to be.  The obvious measure
        -- the angular span of the Gauss points whose pressure is positive -- is a
        SAMPLING STATISTIC, not a property of the solution: it can only report edges at
        one of `n_quad` places per segment, so refining the quadrature moves it even
        though the displacement field has not changed.  Measured on the shipped genome,
        going from 6 to 20 points per segment moved that number from 1.609 to 1.677 deg
        on the medium mesh while the axle drop moved by 3e-5 relative.

        The gap itself is smooth (a deformed boundary against a straight line), so its
        zero crossing is well defined and converges.  This resamples the same shape
        functions at `n_sample` points per segment and interpolates the crossing.

        Returns (half_deg, centre_deg), both nan when nothing is in contact.
        """
        coords = np.asarray(coords, dtype=float)
        xi = np.linspace(-1.0, 1.0, int(n_sample))
        N = np.array([_lagrange_1d(self.order, x)[0] for x in xi])
        X = coords[self.segments]
        U = np.asarray(u).reshape(-1, 2)[self.segments]
        xg = np.einsum("gi,sij->sgj", N, X) + np.einsum("gi,sij->sgj", N, U)
        gap = xg[:, :, 1] - self.y_ground

        # Reference angle, so the measure is an arc of the undeformed rim rather than of
        # a boundary that has itself moved.
        xr = np.einsum("gi,sij->sgj", N, X)
        th = np.degrees(np.arctan2(xr[:, :, 1], xr[:, :, 0]))
        off = (th + 90.0 + 180.0) % 360.0 - 180.0

        o, gp = off.ravel(), gap.ravel()
        order_ = np.argsort(o)
        o, gp = o[order_], gp[order_]
        if not np.any(gp < 0.0):
            return float("nan"), float("nan")

        edges = []
        sign = gp < 0.0
        for i in np.nonzero(sign[:-1] != sign[1:])[0]:
            g0, g1 = gp[i], gp[i + 1]
            if g1 == g0:
                continue
            edges.append(o[i] + (o[i + 1] - o[i]) * (0.0 - g0) / (g1 - g0))
        if len(edges) < 2:                      # patch runs off the sampled range
            inside = o[sign]
            return float((inside.max() - inside.min()) / 2.0), float(inside.mean())
        lo, hi = min(edges), max(edges)
        return float((hi - lo) / 2.0), float((hi + lo) / 2.0)

    def total_force(self, coords, u):
        """Vertical resultant of the contact pressure [N], by quadrature.

        Computed from the pressure field rather than by summing the nodal force vector,
        so it is an INDEPENDENT check on the assembly rather than a restatement of it.
        """
        p = self.pressure(coords, u)
        return float(np.sum(p["pressure_mpa"] * p["weight"]))

    def max_penetration(self, coords, u):
        p = self.pressure(coords, u)
        return float(max(0.0, -p["gap_mm"].min()))

    # -- internals ---------------------------------------------------------

    def _gather(self, coords, u):
        Xs = jnp.asarray(coords)[self.segments]
        us = jnp.asarray(u).reshape(-1, 2)[self.segments]
        return Xs, us

    def _sdof(self):
        return (self.segments[:, :, None] * 2 + np.arange(2)[None, None, :]) \
            .reshape(len(self.segments), -1)


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
                 "contact", "meta")

    def __init__(self, coords, conn, order, lam, mu, width, dofmap,
                 f_nodal=None, f_reduced=None, load_dir=None, nonlinear=False,
                 contact=None, meta=None):
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
        self.contact = contact
        self.meta = dict(meta or {})


def solve_linear(prob):
    """Direct sparse solve of the constrained linear system.

    Returns a dict with the full displacement field, the reduced solution, strain
    energy, an equilibrium residual, and — when the problem carries a rigid tip —
    the tip motion along the load direction.

    RAISES on a contact problem rather than ignoring the contact term.  Contact is
    inherently nonlinear — the whole content of the model is which points are touching —
    so a linear solve of it is not an approximation, it is a different problem with no
    ground in it at all.  Returning that silently would recreate the trap this file
    already has one of: an `svk` problem routed through here comes back LINEAR without
    complaint, because the SVK Hessian equals the linear one exactly at u=0.  That one
    cannot be fixed (the equality is a fact); this one can, so it is.
    """
    if prob.contact is not None:
        raise ValueError(
            "this problem carries penalty contact, which is inherently nonlinear — "
            "`solve_linear` would silently drop the ground and return a load-free "
            "field.  Use `solve` (which dispatches) or `solve_nonlinear`.")
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

    _report_tip(prob, u_red, out)
    return out


def _report_tip(prob, u_red, out):
    """Attach `deflection_mm` / `tip` when the problem carries a rigid tip body."""
    named = prob.dofmap.named
    if prob.load_dir is None or not ("tip_ux" in named or "tip_uy" in named):
        return out
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
# NONLINEAR SOLVE  (M5)
# ---------------------------------------------------------------------------
#
# NEWTON ON THE ENERGY, NOT ON THE RESIDUAL.
#
# The total potential is
#
#     Pi(u_red; s) = U_int(u_full)  -  s * (f_nodal . u_full + f_reduced . u_red)
#     u_full       = T @ u_red + s * u_pre
#
# and everything the loop needs is a derivative of it: the residual is
# `grad_{u_red} Pi` and the tangent is its Hessian, exactly as in the linear path.
# `s` is the load-continuation factor and scales the PRESCRIBED displacement too, so
# a continuation step is a genuinely proportional load case rather than a partial
# force applied against fully imposed kinematics.
#
# Having Pi in closed form is what makes the Armijo line search free.  Backtracking on
# ||r|| instead is the common shortcut and it is strictly worse here: ||r|| is not a
# merit function for a nonconvex problem — it has minima at unstable equilibria — while
# Pi is the thing physically being minimised, and a Newton direction from an SPD tangent
# is guaranteed to be a descent direction for it.  So the line search cannot stall on a
# direction that is genuinely good.
#
# A FAILED SOLVE RAISES.  It does not return a best-effort field with a warning.  The
# Stage-3 gradient is computed at an assumed-converged state, and one silently
# unconverged solve poisons a gradient in a way that is nearly undebuggable after the
# fact; the caller's correct response is to reject the design step and shrink it, which
# it can only do if it is told.
#
# TWO CONVERGENCE CRITERIA, BECAUSE A RESIDUAL TOLERANCE ALONE IS NOT ATTAINABLE.
#
# The plan asks for a residual below 1e-10 relative.  That is met at service load and is
# NOT met at 1% of it: the run stalls at 1.2e-9 and burns every iteration trying.  This
# is not a solver defect and no amount of iterating fixes it.  The reduced tangent of a
# thin flexure has condition number ~1e8-1e9, so the sparse factorization returns an
# increment with that much relative error, and the residual it can drive to is bounded
# below by roughly kappa * eps regardless of the load.  Normalising by ||f|| does not
# help, because the FLOOR scales with the load exactly as the load does.
#
# So the loop also accepts Bathe's ENERGY-INCREMENT criterion: |du . r|, the energy the
# next step would release, measured against the first step's.  That quantity is
# dimensionally an energy, is insensitive to conditioning, and says something a residual
# norm cannot — "the remaining unconverged energy is 1e-14 of the total", which is the
# statement that actually matters for a result quoted to four figures.  It is reported
# alongside the residual so a solve can never look converged on a technicality.

class NewtonDivergedError(RuntimeError):
    """A nonlinear solve did not reach either convergence criterion.

    Carries `.history` (the per-iteration record) so a caller that reduces a step size
    in response can log *why* without re-running the solve.
    """

    def __init__(self, message, history=None):
        super().__init__(message)
        self.history = list(history or [])


def _potential(prob, u_full, u_red, scale):
    """Total potential energy at a trial state.  The line search's merit function.

    The contact potential is part of Pi, not a side condition — which is exactly why the
    Armijo search keeps working once contact is switched on: Pi is still the thing being
    minimised, and a Newton direction from an SPD tangent still descends it.
    """
    U = total_energy(prob.coords, prob.conn, u_full, order=prob.order, lam=prob.lam,
                     mu=prob.mu, width=prob.width, nonlinear=prob.nonlinear)
    if prob.contact is not None:
        U += prob.contact.energy(prob.coords, u_full)
    return U - scale * (float(prob.f_nodal @ u_full) + float(prob.f_reduced @ u_red))


def solve_nonlinear(prob, *, steps=1, tol=1e-10, tol_energy=1e-14, max_iter=25,
                    max_backtracks=20, armijo_c=1e-4, u_reduced0=None):
    """Newton-Raphson with energy line search and load continuation.

    `steps` splits the load into that many equal proportional increments, each solved
    from the previous converged state.  One step suffices for this wheel at service
    load (measured in `study_gnl.py`: 5 iterations, zero backtracks); the machinery
    exists because the Stage-3 optimizer will visit designs that do not.

    Converged when EITHER the relative reduced residual ||r||/||f_ref|| is below `tol`
    OR the energy increment |du . r| has fallen to `tol_energy` of the first step's.
    See the block comment above for why the second criterion is not a fudge — at low
    load it is the only one that is attainable.

    `u_reduced0` warm-starts from a previous solve, which is the single most valuable
    performance lever once this sits inside an optimizer loop.

    Returns the same keys as `solve_linear` plus a `newton` block recording, per load
    step, the iteration count, the residual and energy-increment histories, how many
    line-search backtracks were taken, and which criterion actually fired.

    Raises `NewtonDivergedError` rather than returning an unconverged field.
    """
    T, n_red = prob.T, prob.T.shape[1]
    kw = dict(order=prob.order, lam=prob.lam, mu=prob.mu, width=prob.width,
              nonlinear=prob.nonlinear)

    f_ref = float(np.linalg.norm(T.T @ prob.f_nodal + prob.f_reduced))
    if f_ref == 0.0 and prob.contact is not None:
        # A displacement-driven contact problem has NO applied force: the contact
        # pressure is the entire load.  Scaling the residual by 1.0 would silently turn
        # the relative tolerance into an absolute one, so the reference is the contact
        # force at u=0 — the load the initial indentation is already applying, which is
        # the right order of magnitude and is free to evaluate.
        f0 = prob.contact.force(prob.coords, np.zeros(2 * prob.coords.shape[0]))
        f_ref = float(np.linalg.norm(T.T @ f0)) or 1.0
    elif f_ref == 0.0:                     # pure prescribed-displacement problem
        f_ref = float(np.linalg.norm(prob.u_pre)) or 1.0

    u_red = (np.zeros(n_red) if u_reduced0 is None
             else np.asarray(u_reduced0, dtype=float).copy())
    log = []

    def residual(u_r, scale):
        u_f = T @ u_r + scale * prob.u_pre
        fint = internal_force(prob.coords, prob.conn, u_f, **kw)
        if prob.contact is not None:
            fint = fint + prob.contact.force(prob.coords, u_f)
        r = T.T @ (fint - scale * prob.f_nodal) - scale * prob.f_reduced
        return u_f, r, float(np.linalg.norm(r)) / f_ref

    for k in range(1, int(steps) + 1):
        scale = k / float(steps)
        resid_hist, dE_hist, backtracks = [], [], 0
        why, dE_ref = None, None

        for it in range(int(max_iter)):
            u_full, r, rn = residual(u_red, scale)
            resid_hist.append(rn)
            if rn <= tol:
                why = "residual"
                break

            K = assemble_stiffness(prob.coords, prob.conn, u_full, **kw)
            if prob.contact is not None:
                K = K + prob.contact.stiffness(prob.coords, u_full)
            Kr = (T.T @ K @ T).tocsc()
            du = spla.spsolve(Kr, -r)
            if not np.all(np.isfinite(du)):
                raise NewtonDivergedError(
                    f"tangent solve produced non-finite increment at load step {k}"
                    f"/{steps}, iteration {it}", log)

            # `slope` is EXACT, because r really is grad Pi rather than an approximation
            # of it.  So a non-negative value is not line-search noise — it says the
            # tangent was not positive definite, i.e. this equilibrium is unstable.  Say
            # that, instead of letting the backtrack loop grind alpha down to nothing.
            slope = float(r @ du)
            if slope >= 0.0:
                raise NewtonDivergedError(
                    f"Newton direction is not a descent direction (slope {slope:.3e}) "
                    f"at load step {k}/{steps}, iteration {it} — the tangent is not "
                    f"positive definite, i.e. this equilibrium is unstable", log)

            dE = abs(slope)
            dE_hist.append(dE)
            if dE_ref is None:
                dE_ref = dE
            if dE <= tol_energy * dE_ref:
                # The step is at the floor.  Take it — it can only help — and stop.
                u_red = u_red + du
                resid_hist.append(residual(u_red, scale)[2])
                why = "energy"
                break

            pi0 = _potential(prob, u_full, u_red, scale)
            alpha, taken = 1.0, 0
            while taken < int(max_backtracks):
                trial = u_red + alpha * du
                pi = _potential(prob, T @ trial + scale * prob.u_pre, trial, scale)
                if pi <= pi0 + armijo_c * alpha * slope:
                    break
                alpha *= 0.5
                taken += 1
            else:
                raise NewtonDivergedError(
                    f"line search failed after {max_backtracks} backtracks at load "
                    f"step {k}/{steps}, iteration {it} (residual {rn:.3e})", log)
            backtracks += taken
            u_red = u_red + alpha * du

        log.append({"load_scale": scale, "iterations": len(resid_hist) - 1,
                    "residual_rel": resid_hist[-1], "residuals": resid_hist,
                    "energy_increments": dE_hist, "backtracks": backtracks,
                    "converged": why is not None, "criterion": why,
                    "energy_increment_rel": (dE_hist[-1] / dE_ref) if dE_hist else 0.0})
        if why is None:
            raise NewtonDivergedError(
                f"load step {k}/{steps} did not converge in {max_iter} iterations "
                f"(relative residual {resid_hist[-1]:.3e} vs tol {tol:.1e}; energy "
                f"increment {log[-1]['energy_increment_rel']:.3e} vs tol "
                f"{tol_energy:.1e})", log)

    u_full = T @ u_red + prob.u_pre
    out = {
        "u": u_full,
        "u_reduced": u_red,
        "energy": total_energy(prob.coords, prob.conn, u_full, **kw),
        "residual_rel": log[-1]["residual_rel"],
        "residual": log[-1]["residual_rel"] * f_ref,
        "n_dof_reduced": int(n_red),
        "meta": prob.meta,
        "newton": {"steps": int(steps), "tol": float(tol),
                   "tol_energy": float(tol_energy),
                   "criterion": log[-1]["criterion"],
                   "energy_increment_rel": log[-1]["energy_increment_rel"],
                   "total_iterations": sum(s["iterations"] for s in log),
                   "total_backtracks": sum(s["backtracks"] for s in log),
                   "history": log},
    }
    _report_tip(prob, u_red, out)
    return out


def solve(prob, **kw):
    """Dispatch on the problem's own kinematics AND on whether it carries contact.

    Kept separate from `solve_linear` so that the linear path stays a single sparse
    solve with no Newton scaffolding around it — that is the path M3's beam gate and
    M4's refinement ladder run thousands of times.

    Contact forces the Newton path regardless of `nonlinear`: a penalty contact law is a
    nonlinear boundary condition even under linear kinematics, since which points are
    touching is itself the unknown.  Routing on `prob.nonlinear` alone would send a
    linear-kinematics contact problem to `solve_linear`, which now raises — but getting
    the dispatch right here is what makes that guard a backstop rather than the only
    line of defence.
    """
    if prob.nonlinear or prob.contact is not None:
        return solve_nonlinear(prob, **kw)
    return solve_linear(prob, **kw)


# ---------------------------------------------------------------------------
# STRESS RECOVERY
# ---------------------------------------------------------------------------

def gauss_stresses(coords, conn, u, *, order, lam, mu, nonlinear=False, cauchy=True):
    """Cauchy stress at every Gauss point.

    Returns dict with `sigma` [n_elem, ngp, 2, 2], `von_mises` [n_elem, ngp], and the
    Gauss points' physical coordinates.  Von Mises is the plane-stress form
    sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2), which assumes sigma_zz = 0 — true by
    construction under plane stress and NOT true under plane strain, where the
    reported value is an underestimate.

    Under SVK the constitutive law returns the 2nd Piola-Kirchhoff stress, which lives
    on the REFERENCE configuration and is not what a strain gauge measures.  `cauchy`
    pushes it forward, sigma = F S F^T / det F, so the number is comparable to the
    linear kernel's.  That is not a detail here: at 1% strain and a few degrees of
    rotation the push-forward is worth a couple of percent, the same order as the
    geometric effect M5 sets out to measure, so leaving it out would let the two
    confound.  `cauchy=False` returns the raw S for anyone who wants it.
    The stress itself comes from `_stress_kernel`, which is also what M8's p-norm stress
    QoI differentiates.  Two copies of a constitutive law is exactly the drift this file's
    "write the energy, derive everything else" rule exists to prevent, and a QoI that
    disagreed with the reported stress by a push-forward would be invisible in an
    optimizer trace.
    """
    N, _, _ = _TABLES[order]
    coords = jnp.asarray(coords)
    conn_np = np.asarray(conn)
    Xe = coords[conn_np]
    ue = jnp.asarray(u).reshape(-1, 2)[conn_np]

    sigma = np.asarray(_stress_kernel(order, nonlinear, cauchy)(Xe, ue, lam, mu))
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


def spoke_deflection(genes, cfg="coarse", *, newton=None, **kw):
    """Convenience: the tip deflection magnitude, which is what the beam model returns.

    Dispatches on `kinematics`, so `kinematics="svk"` here really does run the Newton
    loop.  Before M5 it did not — `solve_linear` assembles the tangent at u=0, where the
    SVK Hessian equals the linear one exactly, so an "svk" spoke deflection was silently
    a linear answer.
    """
    prob = spoke_problem(genes, cfg, **kw)
    return abs(solve(prob, **(newton or {}))["deflection_mm"])


# ---------------------------------------------------------------------------
# THE FULL-WHEEL PROBLEM
# ---------------------------------------------------------------------------
#
# Hub RIGID AND FIXED, ground pressure distributed over an assumed patch on the rim OD.
#
# The plan's alternative — hub free in y, loaded with F_y, ground reacting — is the same
# problem seen from the other frame, but it is SINGULAR as stated: with the load applied
# as a pressure rather than as a constraint, nothing removes the rigid vertical
# translation.  Fixing the hub and letting the contact patch rise is well posed, gives
# the identical number for a linear model, and makes the hub reaction an independent
# equilibrium check for free.
#
# `axle_drop` is therefore the RISE of the contact patch relative to the fixed hub, which
# is exactly the drop of the axle relative to fixed ground.

# The contact phase is NOT a load parameter — it rolls the wheel, so it lives on
# `wheel_wheel.build_wheel(phase_deg=...)`.  The ground stays at the bottom.
CONTACT_PATCH_HALF_DEG = 3.0   # see `study_wheel_fea.py` for the sensitivity sweep


def hertz_patch_half_angle_deg(force=TOTAL_FORCE_NEWTONS, radius=50.0,
                               width=SPOKE_WIDTH_MM, E=YOUNGS_MODULUS_PLA_MPA,
                               nu=POISSON_RATIO_PLA):
    """Line-contact half-angle for a SOLID cylinder on a rigid plane, in degrees.

    A lower bound on the real patch, and a small one — about 0.3 degrees here.  The real
    rim is a 1.1 mm band over spokes rather than a solid cylinder, so it flattens far more
    and the patch is wider; but the Hertz number is worth having because it says the patch
    is small enough that resolving it is a mesh-density question, not an afterthought.
    """
    e_star = E / (1.0 - nu ** 2)
    a = np.sqrt(4.0 * force * radius / (np.pi * width * e_star))
    return float(np.degrees(a / radius))


def _ground_traction(half_deg):
    """Elliptical (Hertz-shaped) ground traction over the patch at the bottom.

    Two choices here, both deliberate.

    VERTICAL, not radial.  For frictionless contact against a rigid FLAT ground the two
    bodies share a tangent plane, so the traction acts along the ground's normal, which
    is vertical everywhere in the patch — not along the rim's own radial direction.  The
    two agree to cos(half-angle), which is 0.1% at 3 degrees but 2% at 12, and using the
    radial normal also produces a spurious horizontal resultant that the hub then has to
    react.

    ELLIPTICAL, not uniform.  A uniform patch has a pressure step at its edges, and a
    step in the traction puts a stress singularity into a model whose peak stress is one
    of the reported numbers.  The magnitude is arbitrary — the caller rescales the
    assembled load so the vertical resultant is exactly the wheel load, which also
    removes any dependence on how the quadrature happened to sample the shape.
    """
    centre = np.radians(-90.0)
    half = np.radians(half_deg)

    def traction(x, n):
        th = np.arctan2(x[1], x[0])
        d = (th - centre + np.pi) % (2.0 * np.pi) - np.pi
        if abs(d) >= half:
            return (0.0, 0.0)
        return (0.0, np.sqrt(max(0.0, 1.0 - (d / half) ** 2)))

    return traction


def wheel_problem(mesh, *, force=TOTAL_FORCE_NEWTONS,
                  patch_half_deg=CONTACT_PATCH_HALF_DEG,
                  E=YOUNGS_MODULUS_PLA_MPA, nu=POISSON_RATIO_PLA, plane="stress",
                  width=SPOKE_WIDTH_MM, kinematics="linear", rim_modulus_scale=1.0):
    """The M4 problem: rigid fixed hub, ground pressure on the rim OD.

    `rim_modulus_scale` multiplies the rim band's stiffness only.  Setting it to 1000
    makes the rim effectively rigid, which is the independent cross-check on the
    strain-energy compliance split: the axle drop should fall by exactly the rim's
    reported share.
    """
    coords = np.asarray(mesh.coords, dtype=float)
    lam, mu = lame(E, nu, plane)

    f = segment_traction_load(coords, mesh.edge_sets["rim_outer"],
                              _ground_traction(patch_half_deg),
                              n_nodes=mesh.n_nodes, order=mesh.cfg.order, width=width)
    fy = float(f.reshape(-1, 2)[:, 1].sum())
    if fy <= 0.0:
        raise ValueError(
            f"patch produced a downward or zero vertical resultant ({fy:.4e} N) — "
            f"half-angle {patch_half_deg} deg may be too small for this mesh to resolve")
    f *= force / fy                       # exact resultant, whatever the quadrature did

    dm = DofMap(mesh.n_nodes)
    tie = mesh.node_sets["hub_tie"]
    dm.fix(tie)
    dm.free(np.setdiff1d(np.arange(mesh.n_nodes), tie))

    prob = Problem(coords, mesh.conn, mesh.cfg.order, lam, mu, width, dm,
                   f_nodal=f, nonlinear=(kinematics == "svk"),
                   meta={"config": mesh.cfg.name, "plane": plane,
                         "kinematics": kinematics, "E_mpa": float(E), "nu": float(nu),
                         "width_mm": float(width), "force_n": float(force),
                         "patch_half_deg": float(patch_half_deg),
                         "rim_modulus_scale": float(rim_modulus_scale),
                         "n_elements": mesh.n_elements, "n_nodes": mesh.n_nodes})
    if rim_modulus_scale != 1.0:
        prob.meta["rim_scaled"] = True
    return prob, f


def element_strain_energy(coords, conn, u, *, order, lam, mu, width, nonlinear=False):
    """Strain energy of every element, [n_elem].  The compliance split reads this.

    For a linear body under one load system U = F*delta/2, so the FRACTION of total
    strain energy stored in a region is exactly that region's fraction of the axle drop.
    That makes `compliance_split` a measured decomposition rather than a modelling
    assumption — no substitution experiment or rule of thumb needed, though
    `rim_modulus_scale` provides one as a cross-check.
    """
    coords = jnp.asarray(coords)
    conn_np = np.asarray(conn)
    energy, _, _ = _element_kernels(order, nonlinear)
    ue = jnp.asarray(u).reshape(-1, 2)[conn_np]
    return np.asarray(energy(coords[conn_np], ue, lam, mu, width))


def solve_wheel(mesh, *, newton=None, **kw):
    """Solve and report axle drop, the compliance split, and the equilibrium checks.

    `kinematics="svk"` routes through `solve_nonlinear`; `newton` is its keyword dict.
    """
    scale = kw.get("rim_modulus_scale", 1.0)
    prob, f = wheel_problem(mesh, **kw)

    if prob.nonlinear:
        if scale != 1.0:
            raise NotImplementedError(
                "rim_modulus_scale is a linear cross-check on the compliance split and "
                "has no nonlinear path; the scaled assembly is built by hand from a "
                "single (lam, mu) and would not match the element kernel's tangent")
        res = solve_nonlinear(prob, **(newton or {}))
    elif scale != 1.0:
        # Scaling one region's modulus needs two assemblies, because `Problem` carries a
        # single (lam, mu).  Cheaper and clearer than threading per-element material
        # through the element kernel for what is a one-off cross-check.
        K = _assemble_scaled(mesh, prob, scale)
        T = prob.T
        Kr = (T.T @ K @ T).tocsc()
        fr = T.T @ prob.f_nodal
        u_red = spla.spsolve(Kr, fr)
        res = {"u": T @ u_red, "u_reduced": u_red,
               "residual_rel": float(np.linalg.norm(Kr @ u_red - fr)
                                     / max(np.linalg.norm(fr), 1e-300)),
               "n_dof_reduced": int(Kr.shape[0]), "meta": prob.meta}
        res["energy"] = 0.5 * float(res["u"] @ (K @ res["u"]))
    else:
        res = solve_linear(prob)

    u = res["u"]
    xy = np.asarray(mesh.coords)
    disp = u.reshape(-1, 2)

    # Axle drop: the rise of the lowest point of the wheel relative to the fixed hub.
    # Reported two ways because they answer slightly different questions — the centre
    # node is "how far did the ground move", the patch mean is less sensitive to how the
    # pressure was distributed.
    patch_nodes = np.unique(mesh.edge_sets["rim_outer"])
    th = np.degrees(np.arctan2(xy[patch_nodes, 1], xy[patch_nodes, 0]))
    d = (th + 90.0 + 180.0) % 360.0 - 180.0
    half = kw.get("patch_half_deg", CONTACT_PATCH_HALF_DEG)
    in_patch = patch_nodes[np.abs(d) <= half]
    centre_node = patch_nodes[np.argmin(np.abs(d))]

    res["axle_drop_mm"] = float(disp[centre_node, 1])
    res["axle_drop_patch_mean_mm"] = (float(disp[in_patch, 1].mean())
                                      if in_patch.size else float("nan"))
    res["n_nodes_in_patch"] = int(in_patch.size)

    # Compliance split, exact for a linear body under one load system.  Under SVK it is
    # still an exact decomposition of the stored energy, but U = F*delta/2 no longer
    # holds, so it stops being exactly a first-order sensitivity of the axle drop.  The
    # gap between 2U/(F*delta) and 1 is itself a measure of the geometric nonlinearity
    # and `study_gnl.py` reports it.
    lam, mu = prob.lam, prob.mu
    e = element_strain_energy(xy, mesh.conn, u, order=mesh.cfg.order, lam=lam, mu=mu,
                              width=prob.width, nonlinear=prob.nonlinear)
    if scale != 1.0:
        e = np.where(mesh.region_mask("rim"), e * scale, e)
    _attach_compliance_split(res, mesh, e)

    # Equilibrium: the hub reaction must be the applied load, to solver precision.  An
    # independent check on the whole assembly — mesh, constraints, quadrature, solve —
    # and nearly free, since K*u is already formed.
    if prob.nonlinear:
        # K @ u is NOT the internal force once the kinematics are nonlinear, so the
        # equilibrium check has to go through the residual the Newton loop drove to
        # zero.  Using K @ u here would report a fictitious reaction that happens to be
        # close, which is the worst possible behaviour for a check.
        fint = internal_force(xy, mesh.conn, u, order=mesh.cfg.order, lam=lam, mu=mu,
                              width=prob.width, nonlinear=True)
        reaction = (fint - prob.f_nodal).reshape(-1, 2)
    else:
        K = assemble_stiffness(xy, mesh.conn, order=mesh.cfg.order, lam=lam, mu=mu,
                               width=prob.width)
        reaction = (K @ u - prob.f_nodal).reshape(-1, 2)
    tie = mesh.node_sets["hub_tie"]
    res["applied_force_n"] = [float(prob.f_nodal.reshape(-1, 2)[:, i].sum())
                              for i in (0, 1)]
    if scale == 1.0:
        res["hub_reaction_n"] = [float(reaction[tie, i].sum()) for i in (0, 1)]
        res["equilibrium_error_n"] = float(np.abs(
            np.array(res["hub_reaction_n"]) + np.array(res["applied_force_n"])).max())
    return res


def _attach_compliance_split(res, mesh, e):
    """Region shares of the strain energy, from per-element energies already computed."""
    total = float(e.sum())
    res["strain_energy_mJ"] = total
    res["compliance_split"] = {r: float(e[mesh.region_mask(r)].sum() / total)
                               for r in ("spoke", "hub", "rim")}


def _assemble_scaled(mesh, prob, scale):
    """K with the rim band's modulus multiplied by `scale`."""
    xy = np.asarray(mesh.coords)
    base = assemble_stiffness(xy, mesh.conn, order=mesh.cfg.order, lam=prob.lam,
                              mu=prob.mu, width=prob.width)
    rim = mesh.region_mask("rim")
    extra = assemble_stiffness(xy, mesh.conn[rim], order=mesh.cfg.order,
                               lam=prob.lam * (scale - 1.0),
                               mu=prob.mu * (scale - 1.0), width=prob.width)
    return (base + extra).tocsr()


# ---------------------------------------------------------------------------
# THE FULL-WHEEL CONTACT PROBLEM  (M6)
# ---------------------------------------------------------------------------
#
# DISPLACEMENT DRIVEN, WHICH THE EXISTING FRAME ALREADY WANTED.
#
# `wheel_problem` above records why the hub is fixed rather than the ground: with the
# load applied as a pressure, nothing else removes the rigid vertical translation.  That
# argument gets stronger with real contact, because before contact establishes there is
# no pressure at all, so a force-driven contact problem is singular at its own starting
# point.  Prescribing the ground's position instead is well posed from the first
# iteration.
#
# It also makes the reported quantity exact rather than inferred: with a rigid FLAT
# ground indenting by `delta`, the rise of the wheel's lowest point IS `delta`, so the
# axle drop is an input and the wheel load is the output.  `solve_wheel_contact` then
# inverts that with a secant iteration to answer the question M4 asked -- what is the
# axle drop at service load -- and each solve warm-starts from the previous one.
#
# The measured centre-node displacement is reported ALONGSIDE the prescribed indentation
# rather than in place of it.  They differ by exactly the local penetration, so their
# gap is a free consistency check on the penalty stiffness: if it is not small, `eps_n`
# is too soft and the answer is being reported for a ground that the wheel sank into.

# N/mm^3.  Pressure is `eps_n * penetration`, so this is "how many MPa per mm of
# penetration" -- which is what makes the scale pickable at all: the peak pressure here
# is a few MPa, so 1e4 buys a penetration of ~3e-4 mm, 2e-4 of the 1.5 mm rim band.
# It is not a tuned constant; `study_contact.py` demonstrates the plateau it sits in.
DEFAULT_CONTACT_EPS_N = 1.0e4


def wheel_contact_problem(mesh, *, indentation_mm, eps_n=DEFAULT_CONTACT_EPS_N,
                          E=YOUNGS_MODULUS_PLA_MPA, nu=POISSON_RATIO_PLA,
                          plane="stress", width=SPOKE_WIDTH_MM, kinematics="linear",
                          n_quad=6, smoothing_mm=0.0):
    """Rigid fixed hub, rigid frictionless ground indenting the rim OD by `indentation_mm`.

    No applied force: the contact pressure is the entire load.  Compare `wheel_problem`,
    which assumes both the patch and the pressure shape.
    """
    coords = np.asarray(mesh.coords, dtype=float)
    lam, mu = lame(E, nu, plane)

    # The ground sits below the undeformed OD by the indentation.  `phase_deg` rolls the
    # wheel (see `wheel_wheel.build_wheel`); the ground never moves, which is the whole
    # point of the frame and what the periodicity check tests.
    y_ground = -float(mesh.rim_outer) + float(indentation_mm)
    contact = RigidGroundContact(mesh.edge_sets["rim_outer"], y_ground, eps_n,
                                 width=width, order=mesh.cfg.order, n_quad=n_quad,
                                 smoothing_mm=smoothing_mm)

    dm = DofMap(mesh.n_nodes)
    tie = mesh.node_sets["hub_tie"]
    dm.fix(tie)
    dm.free(np.setdiff1d(np.arange(mesh.n_nodes), tie))

    return Problem(coords, mesh.conn, mesh.cfg.order, lam, mu, width, dm,
                   nonlinear=(kinematics == "svk"), contact=contact,
                   meta={"config": mesh.cfg.name, "plane": plane,
                         "kinematics": kinematics, "E_mpa": float(E), "nu": float(nu),
                         "width_mm": float(width),
                         "indentation_mm": float(indentation_mm),
                         "eps_n": float(eps_n), "n_quad": int(n_quad),
                         "smoothing_mm": float(smoothing_mm),
                         "contact": True,
                         "n_elements": mesh.n_elements, "n_nodes": mesh.n_nodes})


def solve_wheel_contact_at(mesh, indentation_mm, *, newton=None, steps=1,
                           u_reduced0=None, **kw):
    """One contact solve at a prescribed indentation.  Reports the load it produced.

    `steps` ramps the GROUND down in that many equal increments, warm-starting each from
    the last.  This is not the same knob as `solve_nonlinear(steps=...)` and cannot be:
    that one scales `f_nodal` and `u_pre`, and a displacement-driven contact problem has
    neither -- the ground's position lives in the contact object, so the only way to
    continue it is to rebuild the problem.  Ramping matters because the difficulty here
    is the contact SET, not the magnitude: opening the gap gradually lets points come
    into contact a few at a time instead of all at once from a cold start, which is what
    the iteration counts respond to.

    One step suffices at the default `eps_n`.  The ramp exists for the stiffer end of the
    plateau sweep and for the designs the Stage-3 optimizer will visit.

    THE ITERATION BUDGET IS LARGER HERE, AND THE RESIDUAL HISTORY IS NOT MONOTONE.
    `solve_nonlinear`'s default of 25 is calibrated for smooth SVK, which converges in 5.
    Contact does not converge that way at all: the iterations are an ACTIVE-SET SEARCH,
    and until the set of points in contact is right the tangent is the wrong operator, so
    the residual wanders on a noisy plateau.  Measured on the shipped genome at the
    default `eps_n`, the last dozen residuals run

        2.4e-4  3.0e-4  1.0e-4  1.2e-4  4.1e-5  4.6e-5  2.7e-5  3.9e-5  2.5e-5  1.3e-5
        6.0e-5  6.6e-16

    -- rising as often as falling, then dropping fifteen orders of magnitude in ONE step
    the moment the set settles, because at that point Newton is solving a smooth problem
    exactly.  So a criterion based on residual DECREASE would read this as divergence and
    give up one iteration before the answer, and a budget sized from the SVK experience
    stops in the middle of the plateau.  Neither is a solver defect.
    """
    prob = None
    res = None
    warm = u_reduced0
    n = max(1, int(steps))
    nk = dict(newton or {})
    nk.setdefault("max_iter", 60)
    for k in range(1, n + 1):
        d = float(indentation_mm) * k / n
        prob = wheel_contact_problem(mesh, indentation_mm=d, **kw)
        res = solve_nonlinear(prob, **dict(nk, u_reduced0=warm))
        warm = res["u_reduced"]
    _attach_contact_report(res, mesh, prob, indentation_mm)
    res["contact_steps"] = n
    return res


def _attach_contact_report(res, mesh, prob, indentation_mm):
    """Axle drop, patch geometry, compliance split and equilibrium for a contact solve."""
    u = res["u"]
    xy = np.asarray(mesh.coords)
    disp = u.reshape(-1, 2)
    con = prob.contact

    # The axle drop is the prescribed indentation by construction.  The centre-node
    # displacement is reported beside it as the consistency check described above.
    rim_nodes = np.unique(mesh.edge_sets["rim_outer"])
    th = np.degrees(np.arctan2(xy[rim_nodes, 1], xy[rim_nodes, 0]))
    d = (th + 90.0 + 180.0) % 360.0 - 180.0
    centre_node = rim_nodes[np.argmin(np.abs(d))]

    res["axle_drop_mm"] = float(indentation_mm)
    res["centre_node_rise_mm"] = float(disp[centre_node, 1])
    res["max_penetration_mm"] = con.max_penetration(xy, u)

    p = con.pressure(xy, u)
    live = p["pressure_mpa"] > 0.0
    res["contact_force_n"] = con.total_force(xy, u)
    res["min_pressure_mpa"] = float(p["pressure_mpa"].min())
    res["n_quad_points_in_contact"] = int(live.sum())

    # THE patch measure: the gap's zero crossing, independent of the quadrature.
    half, centre = con.patch_extent(xy, u)
    res["patch_half_deg"] = half
    res["patch_centre_deg"] = centre

    # Both of these are SAMPLED maxima over the quadrature points, so they are reported
    # as diagnostics and must not be quoted as converged numbers — the same status the
    # unfilleted junction's peak stress has in M4.  The peak pressure in particular
    # climbs as `n_quad` rises, because more points sample nearer the true peak.
    res["peak_pressure_mpa_sampled"] = float(p["pressure_mpa"].max())
    res["patch_half_deg_sampled"] = (
        float(np.abs((p["theta_deg"][live] + 90.0 + 180.0) % 360.0 - 180.0).max())
        if live.any() else 0.0)

    lam, mu = prob.lam, prob.mu
    e = element_strain_energy(xy, mesh.conn, u, order=mesh.cfg.order, lam=lam, mu=mu,
                             width=prob.width, nonlinear=prob.nonlinear)
    _attach_compliance_split(res, mesh, e)

    # Equilibrium.  The contact force enters the internal force, so the hub reaction is
    # `f_int - f_contact` at the tied nodes and must balance the contact resultant.  The
    # resultant itself is recomputed by quadrature over the pressure field in
    # `total_force`, so this is two independent routes to the same number rather than
    # one number restated.
    fint = internal_force(xy, mesh.conn, u, order=mesh.cfg.order, lam=lam, mu=mu,
                          width=prob.width, nonlinear=prob.nonlinear)
    fc = con.force(xy, u)
    reaction = (fint + fc).reshape(-1, 2)
    tie = mesh.node_sets["hub_tie"]
    res["hub_reaction_n"] = [float(reaction[tie, i].sum()) for i in (0, 1)]
    # The ground pushes UP along its own normal, so a frictionless flat contact can have
    # no horizontal resultant at all.  This is structural here rather than a convention:
    # only u_y enters the gap, so a nonzero fx would mean the assembly is wrong.
    res["contact_resultant_n"] = [float(-fc.reshape(-1, 2)[:, i].sum()) for i in (0, 1)]
    # Equal and OPPOSITE, so the check is the sum — same convention as `solve_wheel`,
    # where it is `hub_reaction + applied_force`.
    res["equilibrium_error_n"] = float(abs(res["hub_reaction_n"][1]
                                           + res["contact_force_n"]))
    return res


def solve_wheel_contact(mesh, *, force=TOTAL_FORCE_NEWTONS, newton=None,
                        tol_rel=1e-8, max_iter=20, delta0=None, **kw):
    """Solve for the indentation that produces `force`, and report that state.

    A secant iteration on `delta -> contact_force(delta) - force`.  The response is
    near-linear, so the bracket is not worth the extra solves; a secant from two nearby
    points converges in a handful.  Every solve after the first warm-starts from the
    previous converged field, which `solve_nonlinear` supports directly.

    RAISES if it does not converge, for the same reason the Newton loop does: a returned
    "close enough" state is a wrong load case, and a Stage-3 gradient computed at one is
    undebuggable after the fact.
    """
    # The linear-model axle drop is the natural first guess and costs one linear solve.
    if delta0 is None:
        delta0 = solve_wheel(mesh)["axle_drop_mm"]
    d0 = float(delta0)
    d1 = d0 * 1.05

    res = solve_wheel_contact_at(mesh, d0, newton=newton, **kw)
    f0 = res["contact_force_n"]
    warm = res["u_reduced"]
    history = [{"indentation_mm": d0, "force_n": f0}]

    for it in range(int(max_iter)):
        if abs(f0 - force) <= tol_rel * force:
            res["secant"] = {"iterations": it, "history": history,
                             "converged": True}
            res["meta"] = dict(res["meta"], target_force_n=float(force))
            return res
        r1 = solve_wheel_contact_at(mesh, d1, newton=newton, u_reduced0=warm, **kw)
        f1 = r1["contact_force_n"]
        history.append({"indentation_mm": d1, "force_n": f1})
        if f1 == f0:
            raise RuntimeError(
                f"secant stalled: indentations {d0:.6f} and {d1:.6f} give the same "
                f"contact force {f1:.6f} N")
        d_next = d1 - (f1 - force) * (d1 - d0) / (f1 - f0)
        if not np.isfinite(d_next) or d_next <= 0.0:
            raise RuntimeError(
                f"secant produced a non-physical indentation {d_next!r} from "
                f"({d0:.6f}, {f0:.4f}) and ({d1:.6f}, {f1:.4f})")
        d0, f0, res, warm = d1, f1, r1, r1["u_reduced"]
        d1 = d_next

    raise RuntimeError(
        f"contact load control did not reach {force:.4f} N in {max_iter} iterations; "
        f"last force {f0:.6f} N at indentation {d0:.6f} mm")
