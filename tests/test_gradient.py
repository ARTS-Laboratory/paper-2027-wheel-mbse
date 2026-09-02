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

THE SECOND OF THOSE IS A FACT ABOUT THE MESH AND NOT ABOUT THE WHEEL, and since
2026-08-24 (PLAN.md §79) there is a mesh where it is false: `build_wheel(fillet=True)`
models the fillets and `mesh_coords` no longer refuses it.  Every census here is still
taken on the UNFILLETED mesh, which is the one Stage 3 builds, and is still correct.
`study_gradient.run_filleted` is the same census on the other mesh, and
`tests/test_filleted_mesh.py` holds the pins for it.
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


# ---------------------------------------------------------------------------
# §79 — THE SAME GATE ON THE MESH THAT MODELS THE FILLETS
# ---------------------------------------------------------------------------

def test_the_filleted_gate_runs_and_inverts_the_census(genes):
    """`study_gradient.run_filleted`, reduced — the gate that certifies §79's path.

    Reduced the way every other section of this file reduces its gate: `smoke` only, two
    steps instead of three, one gene through the axle drop. What is NOT reduced is any
    tolerance — the identity is still 1e-9 mm and the jacobian still 1e-6 relative
    against a central difference of `build_wheel(fillet=True)` itself.

    G11f RIDES ALONG SINCE §88, and `R_rim` is in `gene_ids` for it rather than for the
    rows above: it is the gene the shipped genome's cliff hangs on, so it is the one where
    a frozen layer profile gets a visibly wrong answer.
    """
    rep = sg.run_filleted(genes, CFG, configs=("smoke",), gene_ids=(8, 12, 13),
                          jac_steps=(1e-4, 1e-6), drop_gene_ids=(12,),
                          drop_steps=(1e-4, 1e-6))
    assert rep["census"]["unfilleted_dead"] == list(sg.INSENSITIVE_EXPECTED)
    assert rep["census"]["filleted_dead"] == [], rep["census"]["filleted_dead"]
    assert rep["census"]["fillet_genes_rank_first"], rep["census"]["ranked"]
    assert rep["worst_identity_mm"] < sg.GATE_FILLET_MESH_MM, rep["identity"]
    assert rep["jacobian"]["worst_rel"] < sg.GATE_FILLET_JAC_REL, rep["jacobian"]
    assert rep["axle_drop"]["worst_rel"] < sg.GATE_SECANT_REL, rep["axle_drop"]["rows"]
    # the two the fillet is worth something for are the two that were dead
    for r in rep["axle_drop"]["rows"]:
        assert r["unfilleted_adjoint"] == 0.0, r
        assert abs(r["adjoint"]) > 1e-3, r
    assert rep["refusals"]["ok"], rep["refusals"]
    # TWO refusals now, not three: §88 made the per-genome layer profile differentiable
    # and G11f measures it instead. The key stays, pinned at `None`, so an old artifact
    # and a new one cannot be read as saying the same thing.
    assert rep["refusals"]["per_genome_layer_profile"] is None
    assert rep["refusals"]["spoke_blocking"] and rep["refusals"]["sector_fit_clamped"]

    pg = rep["per_genome"]
    assert pg["ok"], pg
    assert pg["numpy_path_max_abs_mm"] == 0.0, pg["numpy_path_max_abs_mm"]
    assert pg["identity_max_abs_mm"] < sg.GATE_FILLET_MESH_MM, pg
    assert pg["worst_rel_rule"] < sg.GATE_FILLET_JAC_REL, pg["rows"]
    assert pg["pair"][0] == pytest.approx(
        WW.FILLET_LAYER_CLIFF_FACTOR * pg["cliff_entry"])
    # AND THE POINT OF THE SECTION: a frozen layer profile is not a small approximation of
    # the rule. `R_rim` is the gene the shipped genome's cliff hangs on — the rim binds —
    # so the gradient §85 refused to return is wrong there by four orders more than the
    # gate the honest one clears.
    rim = next(r for r in pg["rows"] if r["gene"] == "R_rim")
    assert rim["rel_frozen"] > 1e-2, rim
    assert rim["rel_frozen"] > 1e4 * rim["rel_rule"], rim
    assert pg["binding_junction"] == "rim", pg["cliff_per_junction"]
    assert rep["pass"], rep


# ---------------------------------------------------------------------------
# THE REGION-RESTRICTED P-NORM'S ARC  (PLAN.md §102)
# ---------------------------------------------------------------------------
# `_qoi_region_pnorm` reads the fillet surface instead of modelling it with `Kt`, and the
# region it reads is a tube about the junction's own arc.  Everything below is about that
# arc being (a) the arc the mesh actually built and (b) a differentiable function of the
# genes — because if either fails, the term prices a feature that is not there and gives
# `R_hub`/`R_rim` a gradient that means nothing.

@pytest.fixture(scope="module")
def filleted(genes):
    return WW.build_wheel(genes, CFG, fillet=True)


def test_the_arc_node_ids_are_the_arcs_own_nodes(genes, filleted):
    """`fillet_arc_nodes` matches by coordinate, and the match has to be EXACT.

    The ids are recovered by looking each arc point up in `mesh.coords` rather than by
    carrying `build_wheel`'s block offsets around, so the thing that makes it sound is
    that the arc points ARE mesh nodes — not nearly, exactly.  A near-miss would be
    silent: the term would weight a neighbouring node and price a slightly different
    region, at every genome, forever.
    """
    import study_corner_singularity as CS

    coords = np.asarray(filleted.coords)
    blocks = WW.sector_blocks(genes, filleted.cfg, fillet=True)
    for label in ("hub", "rim"):
        a = np.asarray(blocks[f"{label}_fillet_a"])
        b = np.asarray(blocks[f"{label}_fillet_b"])
        arc = np.concatenate([a[:, 0, :], b[1:, 0, :]])
        ids = WW.fillet_arc_nodes(filleted, label)

        assert len(ids) == len(arc)
        assert len(set(ids.tolist())) == len(ids), \
            f"{label}: two arc points matched the same node"
        worst = np.abs(coords[ids] - arc).max()
        assert worst == 0.0, \
            f"{label}: arc point missed its node by {worst:.3e} mm — measured, this is 0.0"

        # And it is the same arc `study_corner_singularity` fits, which is what makes
        # §95's whole `(r, p)` measurement transfer to the objective.
        ref = CS.fillet_arcs(genes, filleted.cfg, True)[label]
        C, R, _, _, _, _ = WA._arc_from_nodes(jnp.asarray(coords[ids]))
        assert abs(float(R) - ref["radius"]) < 1e-11, f"{label}: radius disagrees"
        assert np.linalg.norm(np.asarray(C) - ref["centre"]) < 1e-11, \
            f"{label}: centre disagrees"


def test_an_unfilleted_mesh_is_refused_rather_than_given_a_wrong_arc(mesh):
    """There is no arc on an unfilleted mesh, and the honest answer is to say so.

    The failure this prevents is not hypothetical — it is §85's, generalised: a path that
    quietly produces SOMETHING for a mesh that cannot support the quantity is how a
    refusal turns into a wrong number.  What an unfilleted mesh has at `P_t` is the
    singularity the fillet exists to remove (§52, §94), and a tube round it would be a
    region-restricted p-norm over a field that diverges.
    """
    with pytest.raises(ValueError, match="filleted mesh"):
        WW.fillet_arc_nodes(mesh, "hub")


def test_the_arc_rides_the_genes_through_mesh_coords(genes, filleted):
    """`d(fitted radius)/d(R_hub)` is 1, and every other gene's entry is 0.

    THE PREMISE THE WHOLE TERM RESTS ON.  `adjoint_grads` differentiates `Q` with respect
    to the COORDINATES and chains that back to the genes through one shared vjp; the arc
    is not an argument of `Q`, so if it did not ride the coordinates it would be frozen,
    and the region would be a constant region while the geometry moved under it.  It does
    ride them: the fit reads the arc's own nodes, the nodes move with `mesh_coords`, and
    the radius that comes back is the gene.

    The other twelve entries are half the check.  A fit contaminated by the mesh AROUND
    the arc would still give `dR/dR_hub` near 1 and would smear gradient onto `t0`, `cy2`
    and the rest — so the test that the fit is reading the arc is that nothing else moves
    it.  They are NOT identically zero and the tolerance says so: measured at `smoke`, the
    largest is 3.4e-13 on `t0`, twelve orders under the entry that should be 1.  That is
    the least-squares solve's own round-off, not a path — the arc's nodes genuinely do
    move with `t0`, and what the fit extracts from that motion is a radius that does not.
    Asserting `== 0.0` here fails on arithmetic noise while proving nothing extra.
    """
    for label, gid in (("hub", wg.GENE_NAMES.index("R_hub")),
                       ("rim", wg.GENE_NAMES.index("R_rim"))):
        ids = WW.fillet_arc_nodes(filleted, label)

        def radius_of(g):
            return WA._arc_from_nodes(WW.mesh_coords(g, filleted)[ids])[1]

        value, grad = jax.value_and_grad(radius_of)(jnp.asarray(genes))
        grad = np.asarray(grad)
        assert abs(float(value) - genes[gid]) < 1e-9, \
            f"{label}: the fitted radius is not the gene"
        assert abs(grad[gid] - 1.0) < 1e-7, \
            f"{label}: d(radius)/d({wg.GENE_NAMES[gid]}) = {grad[gid]:.9f}, want 1"
        others = np.delete(grad, gid)
        worst = float(np.abs(others).max())
        assert worst < 1e-10, (
            f"{label}: the arc fit picked up {worst:.3e} of gradient from "
            f"{wg.GENE_NAMES[int(np.argmax(np.abs(others)))]} — measured, the largest "
            "round-off entry is 3.4e-13")


def test_the_region_pnorms_gradient_matches_a_central_difference(genes, filleted):
    """§94's item 2 CHECK, which asked for exactly this: "the FD test the assembled
    gradient already has".

    The hub, because it is the junction §94 measured at 1.46x and 2.20x the allowable
    while the objective read 0.58 and 0.80, and `R_hub` because giving that gene a live
    gradient is what the replacement term is FOR.  One junction and one gene rather than
    four of each: a filleted mesh costs a ~2 min XLA compile before it costs a solve, and
    the property being pinned is that the adjoint chain closes, which one gene proves and
    four do not prove any harder.

    MEASURED 2026-09-02 at `smoke`, shipped genome, indentation 1.65 mm:

        h        FD               adjoint          rel
        1e-3    -2.85986560e+00  -3.03772422e+00   5.855e-02
        1e-4    -3.03891570e+00  -3.03772422e+00   3.922e-04
        1e-5    -3.03772422e+00  -3.03772422e+00   1.745e-09

    The ladder is the point and the tolerance is set against its BEST rung, the way
    `test_the_buckling_eigenvalue_gradient_matches_a_central_difference` does: at 1e-3 the
    step is large enough that the FD is measuring curvature, and a single step size would
    be reporting on the step rather than on the gradient.

    THE SIGN IS THE SANITY CHECK.  It is negative: opening the fillet radius lowers the
    stress in the fillet.  A term that came back positive here would be pricing margin
    backwards, and no FD agreement would make that right.
    """
    ids = WW.fillet_arc_nodes(filleted, "hub")
    n_sp = filleted.n_spokes

    def qoi(prob):
        return WA._qoi_region_pnorm(prob, ids, n_sp)

    def value_at(g):
        m = WW.build_wheel(g, CFG, fillet=True)
        return WA.solve_and_grad(g, CFG, qoi, indentation_mm=INDENT_MM,
                                 mesh=m)["value"]

    out = WA.solve_and_grad(genes, CFG, qoi, indentation_mm=INDENT_MM, mesh=filleted)
    grad = out["grad"]
    gid = wg.GENE_NAMES.index("R_hub")

    assert grad[gid] < 0.0, (
        f"d(fillet stress)/d(R_hub) = {grad[gid]:+.6e} — opening the fillet must not "
        "raise the stress in it")

    rng = sg._ranges()
    best = min(
        abs((value_at(_bump(genes, gid, rng[gid] * h))
             - value_at(_bump(genes, gid, -rng[gid] * h)))
            / (2.0 * rng[gid] * h) - grad[gid]) / abs(grad[gid])
        for h in (1e-3, 1e-4, 1e-5))
    assert best < 1e-6, (
        f"d(fillet stress)/d(R_hub) is out by {best:.2e} on its whole FD ladder")


def test_the_region_pnorm_wakes_the_two_genes_defect_1_measured_at_zero(genes, filleted):
    """§15 DEFECT 1, reversed — and this is the affirmative case for the whole switch.

    DEFECT 1 is that `R_hub` and `R_rim` are DEAD: the only paths from a fillet radius
    into the loss are `stress` and the fillet barriers, all of them flat unless breached,
    and measured at the shipped genome on 2026-08-12 `dL/dR_hub` and `dL/dR_rim` are both
    EXACTLY 0.0 — a nominally 14-dimensional search running in 8.  `wheel_objective`'s
    own comment says so at the `stress_margin` block.

    On a filleted mesh read by this term they are not dead, because the radius now moves
    the geometry the field is computed on rather than only a closed-form surrogate that
    is flat above its cap.  Measured at `smoke`: all 14 genes carry a nonzero entry, at
    both junctions.

    This asserts the two that were zero, not all fourteen — the other twelve were never
    the defect, and pinning "all 14 are nonzero" would go red for a reason that has
    nothing to do with §15 if some unrelated gene ever legitimately drops out.
    """
    for label in ("hub", "rim"):
        ids = WW.fillet_arc_nodes(filleted, label)
        out = WA.solve_and_grad(
            genes, CFG, lambda prob: WA._qoi_region_pnorm(prob, ids, filleted.n_spokes),
            indentation_mm=INDENT_MM, mesh=filleted)
        for name in ("R_hub", "R_rim"):
            entry = out["grad"][wg.GENE_NAMES.index(name)]
            assert entry != 0.0, (
                f"{label}: d(fillet stress)/d({name}) came back exactly 0.0 — that is "
                "§15 DEFECT 1's dead gene, which this term exists to wake")
