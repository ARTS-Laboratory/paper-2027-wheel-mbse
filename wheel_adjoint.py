"""
=============================================================================
  COMPLIANT WHEEL — GRADIENTS BY IMPLICIT DIFFERENTIATION  (M7)
=============================================================================
    val, g = value_and_grad(genes, "coarse", "axle_drop")     # g is [14]

Stage 3 is a projected optimizer that calls one function.  If that function returns a
gradient which is right to plotting accuracy and no better, every hour spent on the
optimizer is wasted — a line search cannot tell a wrong gradient from a hard problem.
So this module ships the derivative and `study_gradient.py` ships the measurements that
say it is correct, hardest evidence first.

WRITE THE ENERGY, DERIVE EVERYTHING ELSE — ONE LEVEL UP
-------------------------------------------------------
`wheel_fem.py` gets the internal force and the tangent from `jax.grad` and
`jax.hessian` of one element energy, so they cannot drift apart.  The same argument
applies to the sensitivity, and with more force: a hand-derived `dr/dp` that disagrees
with the residual by a term produces a gradient that is *plausible* — right direction,
wrong length — which is the hardest kind of bug to see from an optimizer trace.

At a converged state the reduced residual is itself a derivative of the potential:

    Pi(c, u_f, y) = U_bulk(c, u_f) + Pi_contact(c, u_f, y)
    u_f           = T @ u_r + u_pre                        [T static: fixed topology]
    r(c, u_r)     = T^T grad_{u_f} Pi = grad_{u_r} Pi      [ = 0 at the solution ]
    K_r           = hess_{u_r} Pi                          [ what Newton assembled ]

so for a quantity of interest Q(c, u_f, y), differentiating r = 0 gives

    K_r adj_r = -T^T (dQ/du_f)                one sparse solve; K_r is symmetric
    adj_f     = T @ adj_r
    dQ/dc     = dQ/dc|_u + grad_c [ adj_f . grad_{u_f} Pi(c, u_f, y) ]
    dQ/dgenes = vjp_{mesh_coords}(dQ/dc)

The third line is the whole adjoint: ONE `jax.grad` of a scalar.  There is no separately
coded sensitivity operator, for the same reason `RigidGroundContact` has no separately
coded contact stiffness.

TWO THINGS THAT FALL OUT RATHER THAN BEING CODED
-------------------------------------------------
The total contact force is `+dPi/dy_ground` exactly — the penalty potential's derivative
with respect to the ground's position IS the resultant it is pressing with.  So
`contact_force` is not a re-implementation of `RigidGroundContact.total_force`'s
quadrature but a derivative of the contact law itself, and the two agreeing is a check
this module gets for free rather than an assumption it makes (gate 2b).

And `d(force)/d(indentation)` is the same adjoint with the same `adj`, differentiated
with respect to `y_ground` instead of `c`.  That is what makes the load-controlled
quantity cheap.

THE AXLE DROP AT SERVICE FORCE IS AN IMPLICIT FUNCTION, NOT A DIFFERENTIATED SECANT
-----------------------------------------------------------------------------------
`solve_wheel_contact` finds `delta*` with `F(delta*, p) = F_service`.  Differentiating
that iteration would put its own convergence tolerance into the answer, which is a
property of the stopping rule and not of the wheel — the same argument M6's G7 makes
when it differentiates the force at fixed indentation rather than the drop at fixed
force.  So the quotient is taken instead:

    d(delta)/dp = -(dF/dp at fixed delta) / (dF/d delta at fixed p)

numerator and denominator from one adjoint solve.

WHAT IS NOT DIFFERENTIABLE HERE, STATED RATHER THAN HIDDEN
-----------------------------------------------------------
`mesh_coords` freezes the flank orientation and the seam ownership.  Those are discrete
decisions, and a step that flips one is a genuine discontinuity of the design space,
not a defect in this module.  `study_gradient.py` measures the finite-difference
plateau and would see it.

And two genes have an IDENTICALLY ZERO gradient.  `R_hub` and `R_rim` are fillet radii;
the mesh models no fillets, so they do not enter `mesh_coords` at all and the zero
appears in `dcoords/dgenes` before any solver is involved.  M6 found this through the
solve; it is sharper than that.  A gradient-based Stage 3 would optimise 12 of 14 genes
and never notice, so it is classified and gated on rather than left to be discovered.

numpy in, numpy out.  Stage 3's optimizer (projected Adam, `scipy.optimize.minimize`)
never traces the outer loop, so there is no `custom_vjp` wrapper here — a code path with
no consumer is exactly what this project's rules reject.  The pieces are separated so
that adding one later is mechanical.
=============================================================================
"""

import functools
import time

import jax_config  # noqa: F401  — must precede every other jax import
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as spla

import wheel_fem as fem
import wheel_genome as wg
import wheel_wheel as WW
from wheel_fea import DENSITY_PLA, TOTAL_FORCE_NEWTONS


# ---------------------------------------------------------------------------
# THE POTENTIAL, AND THE TWO DERIVATIVES THE ADJOINT NEEDS
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _kernels(order, nonlinear, n_quad, smoothing):
    """(Pi, grad_u Pi, mixed) for one element/contact configuration, jitted.

    `conn` and `segments` are ARGUMENTS rather than closed-over constants so that jit
    specialises on shapes and the cache survives a change of mesh config — otherwise
    every solve at a new `cfg` would re-trace, and the trace is ~0.5 s against a 0.9 s
    warm solve.

    `mixed` returns both halves of the adjoint's second term at once —
    `grad_c [adj . r]` and `d/dy [adj . r]` — because they share the whole backward pass
    and the second one is what makes load control cheap.
    """
    e_bulk, _, _ = fem._element_kernels(order, nonlinear)
    e_con, _, _ = fem._contact_kernels(order, n_quad, smoothing)

    def Pi(coords, u_full, y_ground, conn, segs, lam, mu, width, eps_n, cwidth):
        u2 = u_full.reshape(-1, 2)
        return (jnp.sum(e_bulk(coords[conn], u2[conn], lam, mu, width))
                + jnp.sum(e_con(coords[segs], u2[segs], y_ground, eps_n, cwidth)))

    grad_u = jax.jit(jax.grad(Pi, argnums=1))

    def contracted(coords, u_full, y_ground, adj, *rest):
        return jnp.vdot(adj, grad_u(coords, u_full, y_ground, *rest))

    mixed = jax.jit(jax.grad(contracted, argnums=(0, 2)))
    return jax.jit(Pi), grad_u, mixed


def _static_args(prob):
    """The trailing arguments `_kernels`' functions take, as jax arrays."""
    con = prob.contact
    return (jnp.asarray(prob.conn), jnp.asarray(con.segments),
            prob.lam, prob.mu, prob.width, con.eps_n, con.width)


def potential_fn(prob):
    """`Pi(coords, u_full, y_ground)` for a contact problem — a convenience closure.

    RAISES on a problem carrying an external load.  `wheel_problem`'s pressure is a
    consistent nodal load computed FROM the coordinates (`segment_traction_load`) and
    then frozen into `f_nodal`, so a potential written as `U - f.u` with that `f` held
    constant is missing a term, and its gradient would be wrong by exactly the amount
    the load moves with the mesh.  It would still look reasonable.  M6 already paid for
    one silent wrong-problem trap (`solve_linear` on a contact problem); this is the
    same failure in a new place, so it raises here too.
    """
    if prob.contact is None:
        raise ValueError(
            "the adjoint is written for the displacement-driven contact problem, where "
            "the contact pressure is the entire load; this problem carries no contact")
    if np.any(prob.f_nodal) or np.any(prob.f_reduced):
        raise ValueError(
            "this problem carries an external load vector, which was computed from the "
            "coordinates and then frozen — differentiating a potential that holds it "
            "constant silently drops the load's own dependence on the mesh.  Use the "
            "displacement-driven contact problem (`wheel_contact_problem`).")
    con = prob.contact
    Pi, _, _ = _kernels(prob.order, prob.nonlinear, con.n_quad, con.smoothing_mm)
    args = _static_args(prob)
    return lambda c, u, y: Pi(c, u, y, *args)


# ---------------------------------------------------------------------------
# QUANTITIES OF INTEREST
# ---------------------------------------------------------------------------
#
# Each is `Q(coords, u_full, y_ground) -> scalar`, written in jnp so that both partial
# derivatives the adjoint needs come from `jax.grad`.  M7 ships solve outputs only; the
# seven loss terms and the p-norm stress are M8's, and putting them here early would
# mean calibrating weights against a gradient nothing has verified yet.

def _qoi_contact_force(prob):
    """Total vertical contact force [N] = +dPi/dy_ground.

    Not a second implementation of `RigidGroundContact.total_force`'s quadrature — the
    derivative of the contact potential with respect to the ground's position IS the
    resultant.  So this cannot disagree with the contact law that was solved, and its
    agreement with the independent quadrature is evidence rather than a tautology.
    """
    Pi = potential_fn(prob)
    return jax.grad(Pi, argnums=2)


def _qoi_strain_energy(prob):
    """Bulk strain energy [mJ].  Excludes the penalty potential, which is a numerical
    artefact of the contact law rather than energy stored in the wheel."""
    e_bulk, _, _ = fem._element_kernels(prob.order, prob.nonlinear)
    conn = jnp.asarray(prob.conn)
    lam, mu, width = prob.lam, prob.mu, prob.width

    def Q(coords, u_full, y_ground):
        return jnp.sum(e_bulk(coords[conn], u_full.reshape(-1, 2)[conn],
                              lam, mu, width))
    return Q


@functools.lru_cache(maxsize=None)
def _area_kernel(order):
    _, dN, w = fem._TABLES[order]
    dN_j, w_j = jnp.asarray(dN), jnp.asarray(w)

    def one(Xe):
        J = jnp.einsum("gnk,ni->gki", dN_j, Xe)
        detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        return jnp.sum(detJ * w_j)
    return jax.jit(jax.vmap(one))


def _qoi_mass(prob):
    """Mass [g] from the meshed element areas.

    Independent of `u_full`, so its adjoint solve has a zero right-hand side and the
    adjoint term is EXACTLY zero — which makes it the cheapest correctness check in the
    module: the geometry-only gradient must come out equal to a direct `jax.grad`
    through `mesh_coords`, with the whole solve contributing nothing.
    """
    area = _area_kernel(prob.order)
    conn = jnp.asarray(prob.conn)
    width = prob.width

    def Q(coords, u_full, y_ground):
        return jnp.sum(area(coords[conn])) * width * DENSITY_PLA
    return Q


QOI = {
    "contact_force": _qoi_contact_force,
    "strain_energy": _qoi_strain_energy,
    "mass": _qoi_mass,
}


# ---------------------------------------------------------------------------
# THE ADJOINT
# ---------------------------------------------------------------------------

def solve_and_grad(genes, cfg="coarse", qoi="contact_force", *, indentation_mm,
                   phase_deg=0.0, mesh=None, u_reduced0=None, steps=1, newton=None,
                   **problem_kw):
    """Solve at a prescribed indentation and return d(qoi)/d(genes), [14].

    Returns a dict with `value`, `grad`, `d_dindentation`, the forward solve `res`, the
    `mesh`, and a `timings` block — the last one so that gate 10 can report what the
    gradient costs as a fraction of the solve without a second instrumented code path.

    `d_dindentation` is the TOTAL derivative of the quantity with respect to the ground
    position, which for `contact_force` is the tangent stiffness of the whole wheel
    against the ground and is the denominator `axle_drop_value_and_grad` needs.
    """
    genes = np.asarray(genes, dtype=float)
    t = {}

    t0 = time.time()
    if mesh is None:
        mesh = WW.build_wheel(genes, cfg, phase_deg=phase_deg)
    res = fem.solve_wheel_contact_at(mesh, indentation_mm, steps=steps, newton=newton,
                                     u_reduced0=u_reduced0, **problem_kw)
    t["forward_s"] = time.time() - t0

    t0 = time.time()
    # Rebuilt rather than reached for: `solve_wheel_contact_at` owns the ramp and does
    # not return its problem, and rebuilding is deterministic and cheap.  Keeping
    # `wheel_fem.py` untouched is worth more than the milliseconds — M4's, M5's and M6's
    # committed reports all run through it.
    prob = fem.wheel_contact_problem(mesh, indentation_mm=indentation_mm,
                                     **problem_kw)
    out = adjoint_grad(prob, mesh, res["u"], genes, qoi)
    t["gradient_s"] = time.time() - t0
    t["gradient_over_forward"] = t["gradient_s"] / max(t["forward_s"], 1e-12)

    out.update(res=res, mesh=mesh, timings=t,
               indentation_mm=float(indentation_mm), qoi=qoi)
    return out


def adjoint_grad(prob, mesh, u_full, genes, qoi="contact_force"):
    """The four lines of the docstring's derivation, at an already-converged state.

    Separated from the solve so that `study_gradient.py` can drive it against a state
    produced some other way — in particular the unrolled-Newton reference, which must
    differentiate the same converged field rather than one of its own.
    """
    Q = QOI[qoi](prob) if isinstance(qoi, str) else qoi(prob)
    con = prob.contact
    args = _static_args(prob)
    _, _, mixed = _kernels(prob.order, prob.nonlinear, con.n_quad, con.smoothing_mm)

    c_j = jnp.asarray(prob.coords)
    u_j = jnp.asarray(u_full)
    y = float(con.y_ground)
    T = prob.T

    value = float(Q(c_j, u_j, y))
    dQ_dc, dQ_du, dQ_dy = jax.grad(Q, argnums=(0, 1, 2))(c_j, u_j, y)

    # -- the adjoint solve.  K_r is the SAME operator Newton converged on, reassembled
    # at the converged state; `solve_nonlinear` does not hand its last factorisation
    # back, and rebuilding costs a fraction of one Newton iteration.
    K = fem.assemble_stiffness(prob.coords, prob.conn, u_full, order=prob.order,
                               lam=prob.lam, mu=prob.mu, width=prob.width,
                               nonlinear=prob.nonlinear)
    K = K + con.stiffness(prob.coords, u_full)
    Kr = (T.T @ K @ T).tocsc()
    rhs = -(T.T @ np.asarray(dQ_du))
    adj_r = spla.spsolve(Kr, rhs)
    if not np.all(np.isfinite(adj_r)):
        raise RuntimeError(
            "the adjoint solve produced a non-finite multiplier — the tangent at the "
            "converged state is singular, which means the forward solve stopped at a "
            "state Newton could not have converged to")
    adj_f = jnp.asarray(T @ adj_r)

    d_dc, d_dy = mixed(c_j, u_j, y, adj_f, *args)
    dQ_dc = dQ_dc + d_dc
    d_dindentation = float(dQ_dy) + float(d_dy)

    # -- chain through the mesh.  `mesh_coords` freezes the topology; see its docstring
    # for which discrete decisions that hides and why hiding them is right.
    _, vjp = jax.vjp(lambda v: WW.mesh_coords(v, mesh), jnp.asarray(genes))
    grad = np.asarray(vjp(dQ_dc)[0])

    return {"value": value, "grad": grad, "d_dindentation": d_dindentation,
            "adjoint_norm": float(np.linalg.norm(adj_r)),
            "dQ_du_norm": float(np.linalg.norm(np.asarray(dQ_du)))}


# ---------------------------------------------------------------------------
# THE LOAD-CONTROLLED QUANTITY
# ---------------------------------------------------------------------------

def axle_drop_value_and_grad(genes, cfg="coarse", *, force=TOTAL_FORCE_NEWTONS,
                             phase_deg=0.0, mesh=None, newton=None, delta0=None,
                             tol_rel=1e-8, max_iter=20, **problem_kw):
    """Axle drop at service force, and its gradient, [14].

    The value comes from `solve_wheel_contact`'s secant; the gradient does NOT come from
    differentiating it.  `F(delta*, p) = force` gives

        d(delta)/dp = -(dF/dp)/(dF/d delta)

    and both pieces are one adjoint at the converged `delta*`: the numerator is the
    `contact_force` gradient, the denominator is the same adjoint contracted against
    `y_ground`.  So the secant's stopping tolerance enters the VALUE, where it is a
    stated 1e-8, and not the derivative, where it would masquerade as physics.
    """
    genes = np.asarray(genes, dtype=float)
    if mesh is None:
        mesh = WW.build_wheel(genes, cfg, phase_deg=phase_deg)
    # The secant's own knobs are named here rather than swept into `problem_kw`, which
    # goes on to `wheel_contact_problem` and would raise on them.  They belong to the
    # load control and only to it — which is the whole point of the quotient below: the
    # tolerance lands in the VALUE and never in the derivative.
    res = fem.solve_wheel_contact(mesh, force=force, newton=newton, delta0=delta0,
                                  tol_rel=tol_rel, max_iter=max_iter, **problem_kw)
    delta = float(res["axle_drop_mm"])

    out = solve_and_grad(genes, cfg, "contact_force", indentation_mm=delta, mesh=mesh,
                         u_reduced0=res["u_reduced"], newton=newton, **problem_kw)
    dF_ddelta = out["d_dindentation"]
    if not np.isfinite(dF_ddelta) or dF_ddelta <= 0.0:
        raise RuntimeError(
            f"d(force)/d(indentation) came out {dF_ddelta!r}; a wheel that gets softer "
            f"the harder it is pressed is not a load case, it is a solver failure")

    return {"value": delta, "grad": -out["grad"] / dF_ddelta,
            "contact_force_n": out["value"], "d_force_d_indentation": dF_ddelta,
            "force_grad": out["grad"], "res": res, "mesh": mesh,
            "timings": out["timings"], "qoi": "axle_drop"}


# ---------------------------------------------------------------------------
# WHAT STAGE 3 CALLS
# ---------------------------------------------------------------------------

def value_and_grad(genes, cfg="coarse", qoi="axle_drop", **kw):
    """(value, grad[14]).  The single entry point the optimizer needs.

    `axle_drop` is load-controlled and needs no `indentation_mm`; every other quantity
    is reported at a prescribed indentation and requires one.
    """
    if qoi == "axle_drop":
        out = axle_drop_value_and_grad(genes, cfg, **kw)
    else:
        out = solve_and_grad(genes, cfg, qoi, **kw)
    return out["value"], out["grad"]


# ---------------------------------------------------------------------------
# THE GENES THE MESH CANNOT SEE
# ---------------------------------------------------------------------------

def insensitive_genes(genes, mesh, tol=0.0):
    """Which genes do not move a single node of the mesh.

    Measured on `dcoords/dgenes` rather than on a finite difference of the solve, which
    is what makes the statement clean: a run of identical zeros in an FD ladder looks
    exactly like a perfect plateau, and M6's gate had to special-case it.  Here the zero
    is structural and visible before any solver runs.

    Returns (names, column norms).  `tol=0.0` is deliberate — these columns are not
    small, they are absent, and a tolerance would invite them to become small.
    """
    genes = jnp.asarray(np.asarray(genes, dtype=float))
    jac = jax.jacfwd(lambda v: WW.mesh_coords(v, mesh))(genes)   # [n_nodes, 2, 14]
    col = np.asarray(jnp.sqrt(jnp.sum(jac ** 2, axis=(0, 1))))
    names = [wg.GENE_NAMES[i] for i in range(len(col)) if col[i] <= tol]
    return names, col
