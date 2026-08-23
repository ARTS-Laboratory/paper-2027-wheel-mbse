"""
M7 verification of the implicit-differentiation gradient.

Three kinds of check, and the split is the point.

The first group is IDENTITIES — the differentiated potential must be the potential
Newton solved, and the differentiated mesh must be the mesh that was solved.  Both hold
to machine precision or the gradient is a correct derivative of the wrong problem, which
is the failure that survives every plausibility check an optimizer can make.

The second is the adjoint against BRUTE FORCE: unroll the whole Newton loop on a tiny
mesh and differentiate it with `jax.grad`.  No finite difference is involved, so this
one's tolerance is set by linear algebra rather than by step size.  It is the check to
fix first if several fail together.

The third is the two structural facts M7 measured: `mass` has no adjoint term at all
(its right-hand side is exactly zero), and two genes have no gradient at all (the mesh
models no fillets).  Both are asserted in BOTH directions, because each would otherwise
be indistinguishable from a bug that produced zeros for a different reason.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import jax_config              # noqa: E402,F401
import jax                     # noqa: E402
import jax.numpy as jnp        # noqa: E402

import scipy.sparse.linalg as spla  # noqa: E402

import study_gradient as sg    # noqa: E402
import wheel_adjoint as WA     # noqa: E402
import wheel_fem as fem        # noqa: E402
import wheel_genome as wg      # noqa: E402
import wheel_wheel as WW       # noqa: E402
from wheel_fea import DENSITY_PLA   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = "smoke"
INDENT_MM = 1.65


@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


@pytest.fixture(scope="module")
def mesh(genes):
    return WW.build_wheel(genes, CFG)


@pytest.fixture(scope="module")
def solved(genes, mesh):
    prob = fem.wheel_contact_problem(mesh, indentation_mm=INDENT_MM)
    return prob, fem.solve_nonlinear(prob, max_iter=60)


# ---------------------------------------------------------------------------
# THE IDENTITIES
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phase_deg", [0.0, 7.5])
def test_mesh_coords_reproduces_build_wheel(genes, phase_deg):
    """The differentiable mesh must BE the solved mesh, not merely resemble it.

    `mesh_coords` shares `_sector_coords` with `build_wheel` and then applies the
    ownership table the eager pass computed, so this is a refactor tripwire: if the two
    paths ever diverge, every gradient in the project is a correct derivative of a mesh
    nobody solved, and nothing else in the suite would notice.
    """
    m = WW.build_wheel(genes, CFG, phase_deg=phase_deg)
    c = np.asarray(WW.mesh_coords(jnp.asarray(genes), m))
    assert c.shape == np.asarray(m.coords).shape
    assert np.abs(c - np.asarray(m.coords)).max() < sg.GATE_MESH_COORDS_MM


@pytest.mark.parametrize("build_order", [("default", "capped"), ("capped", "default")])
def test_mesh_coords_reproduces_build_wheel_at_a_NON_DEFAULT_uncap(genes, build_order):
    """The same tripwire, for a mesh built with an explicit `uncap` — which is the case
    the test above cannot see, and the case that was BROKEN between 2026-08-18 and
    2026-08-19.

    `mesh_coords` and `coord_fn` both called `_sector_coords` WITHOUT passing `uncap`, so
    the traced half silently took the module default. While `UNCAP_DEFAULT` was `False`
    that could not disagree with anything: the default and every mesh's value were the
    same object. PLAN.md §38 flipped the default to `(True, 1.0)` and the omission became
    a wrong answer — a mesh built `uncap=False` was handed the FAITHFUL geometry's
    coordinates, measured at **0.448 mm** against this file's 1e-9 mm gate. The shipped
    path never noticed, because nothing on it builds a non-default mesh; what it broke is
    exactly the capped-vs-faithful comparison §38 exists to make.

    `build_order` is parametrized because the second half of the defect was the
    `_COORD_FN_CACHE` key, which omitted `uncap` too. Two meshes differing ONLY in `uncap`
    share every other key entry, so whichever was built first won and the other silently
    got its jitted function. A single-order test passes with the cache still wrong.
    """
    kw = {"default": {}, "capped": {"uncap": False}}
    meshes = [(name, WW.build_wheel(genes, CFG, **kw[name])) for name in build_order]
    for name, m in meshes:
        c = np.asarray(WW.mesh_coords(jnp.asarray(genes), m))
        assert np.abs(c - np.asarray(m.coords)).max() < sg.GATE_MESH_COORDS_MM, name
        # ... and via the explicit-numpy path, which is a separate call site.
        c_np = np.asarray(WW.mesh_coords(genes, m, xp=np))
        assert np.abs(c_np - np.asarray(m.coords)).max() < sg.GATE_MESH_COORDS_MM, name
    # The two meshes must actually DIFFER, or the assertions above are vacuous.
    (_, a), (_, b) = meshes
    assert np.abs(np.asarray(a.coords) - np.asarray(b.coords)).max() > 1e-3


def test_grad_of_the_potential_is_the_assembled_residual(solved):
    """`grad_u Pi` vs `internal_force + contact.force`, assembled by scatter.

    Two genuinely different computations of one vector — one a reverse-mode derivative
    of a summed energy, the other a vmapped element force scattered with `np.add.at`.
    They must agree to machine precision, because the adjoint linearises the first and
    Newton converged the second; a mismatch means the two operators differ by a term and
    the gradient is consistently, plausibly wrong.
    """
    prob, res = solved
    con = prob.contact
    _, grad_u, _ = WA._kernels(prob.order, prob.nonlinear, con.n_quad, con.smoothing_mm)
    r_ad = np.asarray(grad_u(jnp.asarray(prob.coords), jnp.asarray(res["u"]),
                             float(con.y_ground), *WA._static_args(prob)))
    r_np = fem.internal_force(prob.coords, prob.conn, res["u"], order=prob.order,
                              lam=prob.lam, mu=prob.mu, width=prob.width,
                              nonlinear=prob.nonlinear) \
        + con.force(prob.coords, res["u"])
    assert np.abs(r_ad - r_np).max() / np.abs(r_np).max() < sg.GATE_RESIDUAL_REL


def test_contact_force_is_the_derivative_of_the_contact_potential(solved):
    """`dPi/dy_ground` is the resultant, and it agrees with integrating the pressure.

    The QoI is defined as the derivative rather than as a second quadrature, so it
    cannot drift from the contact law that was solved.  `total_force` integrates the
    pressure field independently, which makes this evidence rather than a tautology.
    """
    prob, res = solved
    f_energy = float(WA.QOI["contact_force"](prob)(
        jnp.asarray(prob.coords), jnp.asarray(res["u"]),
        float(prob.contact.y_ground)))
    f_quad = prob.contact.total_force(prob.coords, res["u"])
    assert abs(f_energy / f_quad - 1.0) < 1e-9


def test_the_adjoint_refuses_a_problem_whose_load_moves_with_the_mesh(mesh):
    """`wheel_problem`'s pressure is computed FROM the coordinates and then frozen.

    A potential written as `U - f.u` with that `f` held constant is missing a term, so
    its gradient would be wrong by exactly how much the load moves with the mesh — and it
    would look entirely reasonable.  M6 already paid for one silent wrong-problem trap
    (`solve_linear` on a contact problem returning a ground-free field); this refuses
    rather than rebuilding it here.

    Both guards are exercised.  `wheel_problem` trips the first one (no contact at all),
    so the load guard needs a contact problem with a load bolted onto it to be reached —
    which is the configuration that would actually be dangerous, since it looks solvable.
    """
    unloaded, _ = fem.wheel_problem(mesh)
    with pytest.raises(ValueError, match="no contact"):
        WA.potential_fn(unloaded)

    loaded = fem.wheel_contact_problem(mesh, indentation_mm=INDENT_MM)
    loaded.f_nodal = np.ones_like(loaded.f_nodal)
    with pytest.raises(ValueError, match="external load"):
        WA.potential_fn(loaded)


# ---------------------------------------------------------------------------
# THE ADJOINT AGAINST BRUTE FORCE
# ---------------------------------------------------------------------------

def test_adjoint_matches_a_fully_unrolled_newton(genes):
    """The highest-signal check in the suite: no finite difference anywhere.

    The whole nonlinear solve is unrolled on a 336-DOF order-1 mesh and differentiated
    with `jax.grad`, so the comparison isolates adjoint correctness from step-size noise
    entirely.  Order 1 makes it a poor wheel model and that is irrelevant — the question
    is whether the adjoint equals the derivative of THE SAME SOLVE.
    """
    rep = sg.run_unrolled(genes, cold_iters=8, warm_iters=(1,))
    assert rep["pass"], rep["cold"]["worst_rel"]
    assert rep["cold"]["worst_rel"] < sg.GATE_UNROLLED_REL, rep["cold"]
    # The two must also be differentiating the SAME state, or agreement of the
    # derivatives would be a coincidence.  The tolerance is 1e-9 rather than machine
    # precision because the ladder is shortened here to keep the test fast: 8 cold
    # iterations leave a residue the gate's 14 do not (the gate measures 0.0).
    assert rep["cold"]["value_rel"] < 1e-9, rep["cold"]["value_rel"]


# ---------------------------------------------------------------------------
# THE TWO STRUCTURAL FACTS
# ---------------------------------------------------------------------------

def test_mass_has_no_adjoint_term_at_all(genes, mesh):
    """A geometry-only quantity must get nothing from the solve, and it must not need to.

    `mass` does not depend on the displacement, so `dQ/du` is exactly zero, the adjoint
    right-hand side is zero, and the multiplier is zero.  Its gradient must therefore
    equal a direct `jax.grad` straight through `mesh_coords` with no FEA in it — which
    makes this the cheapest possible end-to-end check on the chain rule through the mesh.
    """
    out = WA.solve_and_grad(genes, CFG, "mass", indentation_mm=INDENT_MM, mesh=mesh)
    assert out["dQ_du_norm"] == 0.0
    assert out["adjoint_norm"] == 0.0

    area = WA._area_kernel(mesh.cfg.order)
    conn = jnp.asarray(mesh.conn)

    def mass_of(v):
        c = WW.mesh_coords(v, mesh)
        return jnp.sum(area(c[conn])) * fem.SPOKE_WIDTH_MM * DENSITY_PLA

    direct = np.asarray(jax.grad(mass_of)(jnp.asarray(genes)))
    assert abs(float(mass_of(jnp.asarray(genes))) / out["value"] - 1.0) < 1e-12
    live = np.abs(direct) > 0
    assert np.abs(out["grad"][live] / direct[live] - 1.0).max() < 1e-10


def test_the_fillet_genes_do_not_move_a_single_node(genes, mesh):
    """M6 found this through the solve; here it is one level lower and sharper.

    `R_hub` and `R_rim` are fillet radii and the mesh models no fillets, so they do not
    enter `mesh_coords` at all — `dcoords/dgene` is EXACTLY zero and no solver, contact
    law or step size is involved.  A gradient-based Stage 3 would find them perfectly
    flat and never move them, which is why the assertion is a census in both directions:
    exactly these two, and every other gene demonstrably alive.

    If a fillet ever gets meshed this fails, and that is the signal to revisit Stage 3's
    handling of the two rather than a regression to paper over.
    """
    names, col = WA.insensitive_genes(genes, mesh)
    assert set(names) == set(sg.INSENSITIVE_EXPECTED), names
    for n in sg.INSENSITIVE_EXPECTED:
        assert col[wg.GENE_NAMES.index(n)] == 0.0
    live = [i for i, n in enumerate(wg.GENE_NAMES) if n not in sg.INSENSITIVE_EXPECTED]
    assert min(col[i] for i in live) > 0.0, "a third gene has gone dead"


# ---------------------------------------------------------------------------
# M9 PHASE 1 — THE NEW BUCKLING EIGENVALUE (MECHANISM ONLY, NOT YET A CONSTRAINT)
# ---------------------------------------------------------------------------
#
# `wheel_adjoint._qoi_buckling_eig` is `v^T K_r v == lambda_min(K_r)` for a FIXED
# eigenvector `v`, differentiated through the same adjoint every other quantity in this
# module uses.  Nothing here is wired into `wheel_objective`'s loss yet — no margin, no
# threshold, no phase aggregation — see PLAN.md's M9 plan for why that has to wait for a
# measurement pass.  These tests only ask: does LOBPCG find the true smallest eigenvalue,
# and does the adjoint's gradient of it agree with a finite difference.

def _bump(genes, gid, h):
    g = np.array(genes, dtype=float)
    g[gid] += h
    return g


def test_buckling_eigenvalue_is_the_true_smallest(solved, mesh, genes):
    """LOBPCG's `lambda_min` cross-checked against an independent eigensolver.

    LOBPCG, preconditioned by the adjoint's own factorisation, is new to this repo.
    `eigsh(..., sigma=0, which="LM")` is shift-invert Lanczos via ARPACK — a completely
    different algorithm — so agreement is evidence the two converge to the SAME
    eigenvalue, not both landing on a wrong one the same way.
    """
    prob, res = solved
    out = WA.adjoint_grads(prob, mesh, res["u"], genes, [WA.BUCKLING_EIG_NAME])[
        WA.BUCKLING_EIG_NAME]

    con = prob.contact
    K = fem.assemble_stiffness(prob.coords, prob.conn, res["u"], order=prob.order,
                               lam=prob.lam, mu=prob.mu, width=prob.width,
                               nonlinear=prob.nonlinear)
    K = K + con.stiffness(prob.coords, res["u"])
    Kr = (prob.T.T @ K @ prob.T).tocsc()
    true_min = float(spla.eigsh(Kr, k=1, sigma=0.0, which="LM",
                                return_eigenvectors=False)[0])

    assert abs(out["value"] - true_min) / abs(true_min) < 1e-6, (
        out["value"], true_min)


def test_buckling_eigenvalue_gradient_matches_finite_difference(genes, mesh):
    """`d(lambda_min(K_r))/dgenes` via Hadamard's formula, finite-differenced.

    Fixed indentation, not the service-force secant, isolates the new mechanism from
    load-control coupling — the same level `test_mass_has_no_adjoint_term_at_all` checks
    at. Steps are scaled by the gene's own range, matching `study_gradient.run_plateau`'s
    convention (`cy` spans 64 mm, `R_rim` spans 2.5 mm; a shared absolute step asks each
    gene a different-sized question).
    """
    out = WA.solve_and_grad(genes, CFG, WA.BUCKLING_EIG_NAME,
                            indentation_mm=INDENT_MM, mesh=mesh)
    grad = out["grad"]
    assert np.abs(grad).max() > 0.0, "the buckling eigenvalue gradient came out all zero"

    rng = sg._ranges()

    def buckling_at(g):
        return WA.solve_and_grad(g, CFG, WA.BUCKLING_EIG_NAME,
                                 indentation_mm=INDENT_MM)["value"]

    for gid in (0, 4, 8, 10):
        name = wg.GENE_NAMES[gid]
        best = min(
            abs((buckling_at(_bump(genes, gid, rng[gid] * h))
                 - buckling_at(_bump(genes, gid, -rng[gid] * h)))
                / (2.0 * rng[gid] * h) - grad[gid]) / max(abs(grad[gid]), 1e-12)
            for h in (1e-3, 1e-4, 1e-5))
        assert best < 1e-4, (
            f"d(lambda_min)/d({name}) is out by {best:.2e} on its whole FD ladder")


def test_buckling_eigenvalue_is_blind_to_the_fillet_genes(genes, mesh):
    """`R_hub`/`R_rim` don't enter `mesh_coords`; this term inherits the same zero M7
    found for every other quantity in this module (`test_the_fillet_genes_do_not_move_a_
    single_node`) — a structural fact about the mesh, not something specific to this term.
    """
    out = WA.solve_and_grad(genes, CFG, WA.BUCKLING_EIG_NAME,
                            indentation_mm=INDENT_MM, mesh=mesh)
    for name in sg.INSENSITIVE_EXPECTED:
        assert out["grad"][wg.GENE_NAMES.index(name)] == 0.0


# ---------------------------------------------------------------------------
# THE GATE'S HEADLINE NUMBERS, RERUN SMALL
# ---------------------------------------------------------------------------

def test_every_live_gene_has_a_finite_difference_plateau(genes):
    """The master plan's M7 gate: a gene with no plateau is a gene the objective is not
    smooth in, and it must be found before Stage 3 rather than during it.

    Run over one curvature gene, one taper gene and one fillet radius — the last so that
    the insensitive branch is exercised here too, since that is where a classifier bug
    would hide behind a run of identical zeros that looks like a perfect plateau.
    """
    rep = sg.run_plateau(genes, CFG, indentation_mm=INDENT_MM, gene_ids=(6, 8, 12),
                         steps=(1e-2, 1e-3, 1e-4, 1e-5))
    assert rep["census_ok"], rep["insensitive_genes"]
    assert rep["pass"], rep["worst_rel_to_adjoint"]
    assert rep["rows"]["R_hub"]["insensitive"]
    assert rep["rows"]["cx4"]["plateau_decades"] >= sg.GATE_PLATEAU_DECADES


def test_random_directions_agree_with_the_adjoint(genes):
    """Per-gene differences can each agree and still be wrong together.

    A transposed index or a mis-scaled column cancels in every single-gene difference and
    only appears once several genes move at once.
    """
    rep = sg.run_directional(genes, CFG, indentation_mm=INDENT_MM, n=3)
    assert rep["pass"], [r["rel"] for r in rep["rows"]]


def test_no_config_resolves_the_contact_patch(genes):
    """The master plan's own mitigation for contact faceting is not met by any mesh.

    Risk #7 says "set rim N_theta for >= 8 nodes in the patch", and that rule was written
    when the patch was ASSUMED to be 3 deg wide.  M6 measured 0.484, six times narrower,
    so every config in the project is far below the rule its author intended.

    Asserted here as a fast geometric statement — no solve, just node counts against M6's
    measured patch — because it is a SPECIFICATION for M8 rather than a defect: the gate
    measures that the resulting faceting refines away, and this pins the number that says
    how much refining would be needed.  If a config ever does reach 8, this fails, and
    that is the signal to revisit M8's phase quadrature rather than a regression.
    """
    patch_half_deg = 0.484                     # study_contact.json, medium mesh
    counts = {}
    for name in ("smoke", "coarse", "medium"):
        m = WW.build_wheel(genes, name)
        counts[name] = 2.0 * patch_half_deg / sg._rim_node_spacing_deg(m)
    assert counts["smoke"] < counts["coarse"] < counts["medium"], counts
    assert max(counts.values()) < 8.0, (
        f"a config now resolves the patch ({counts}); the phase-faceting finding and "
        f"M8's quadrature plan both need revisiting")


def test_the_axle_drop_gradient_is_not_the_secants_derivative(genes):
    """The load-controlled quantity, against a finite difference of the whole solve.

    The gradient comes from the implicit-function quotient rather than from
    differentiating the secant, so the secant's stopping tolerance lands in the value and
    not in the derivative.  The tolerance here is the looser one for exactly that reason:
    the REFERENCE runs the secant twice.
    """
    rep = sg.run_axle_drop(genes, CFG, gene_ids=(6,))
    assert rep["d_force_d_indentation"] > 0.0, "the wheel got softer under more load"
    assert rep["pass"], rep["rows"]
