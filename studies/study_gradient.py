"""
=============================================================================
  M7 GATE — THE ADJOINT, AND WHETHER THE OBJECTIVE IS SMOOTH IN ALL 14 GENES
=============================================================================
    .venv-opt/bin/python studies/study_gradient.py

Stage 3 is a projected optimizer that calls one function.  A gradient that is right to
plotting accuracy and no better cannot be told apart from a hard problem by any line
search, so it would produce a run that looks like it is working and is not.  M7's job is
therefore not to write an adjoint — that is fifty lines — but to prove one.

The checks run HARDEST EVIDENCE FIRST, which is the master plan's ordering and its
reason: G1 compares the adjoint against brute-force differentiation of the same solve
and involves no finite difference at all, so it isolates adjoint correctness from
step-size noise.  Everything after it is a statement about smoothness rather than about
the derivative.

THE GATES, WRITTEN DOWN BEFORE THE RUN
---------------------------------------
    G1  adjoint vs fully unrolled Newton, `tiny` config, all 14 genes    1e-8
    G2  grad_u Pi vs the assembled internal + contact force              1e-12
        and the contact force as dPi/dy vs the pressure quadrature
    G3  mesh_coords vs build_wheel's own coordinates, every config       1e-9 mm
    G4  per-gene FD plateau vs the adjoint, h in {1e-2..1e-7} x range    >= 1 decade
    G5  10 random directional derivatives, over a step ladder            1e-5
    G6  adjoint vs FD at 400 points across one gene, at two steps        see below
    G7  phase faceting must refine away                                  see below
    G8  insensitive-gene census                                          exactly 2
    G9  axle drop gradient through the secant                            1e-5
    G10 gradient cost as a fraction of the forward solve                 reported

G1 RUNS ON A DELIBERATELY BAD MESH, AND THAT IS THE POINT
----------------------------------------------------------
`tiny` is order 1, 336 reduced DOF, 0.9 MB per dense Newton iteration — small enough
that the entire nonlinear solve can be unrolled and differentiated with `jax.grad`.
Q4 shear locking makes it a poor wheel model and that is irrelevant here: the question
is whether the adjoint equals the derivative of THE SAME SOLVE, and A4 in
`study_beam_agreement.py` already owns the question of whether the element is right.

Two versions are run and both are reported.  From a COLD start the whole iteration is
unrolled, active-set search included.  From the CONVERGED state a single step is exact
by construction — `u1 = u0 - K^-1 r` has `du1/dp = -K^-1 dr/dp` there, because the term
carrying `dK/dp` is multiplied by a residual that is zero — so the warm ladder is a
consistency check on the machinery rather than independent evidence, and it is labelled
that way instead of being quoted as two passes.

EVERY FINITE DIFFERENCE HERE IS A LADDER, AND THREE GATES FAILED BEFORE THEY WERE
---------------------------------------------------------------------------------
G4, G5 and G9 were each first written at a single step of 1e-4 of the gene's range, and
each failed at `coarse` by one to eight parts in 1e5.  All three had the same cause and
it was not the gradient: it was the REFERENCE's truncation error.  Measured over a
ladder, the directional disagreement falls 1.2e-2, 8.3e-5, 2.1e-5, 6.2e-7 as the step
shrinks by decades while the adjoint does not move at all.  A single-step check has no
plateau — which is the master plan's own criticism of single-point agreement, applied to
its own items 2, 3 and the load-controlled quantity.

G6 REPLACES "MUST BE VISIBLY SMOOTH" WITH A NUMBER
---------------------------------------------------
The master plan asks for a 400-point sweep with no visible staircase.  A plot is not a
gate.  Comparing the adjoint against a central difference at every one of those points
costs the same solves, is strictly stronger, and LOCALISES a leak instead of merely
reporting one.

The outliers arrive in short RUNS, and that is the signature rather than a defect: a
central difference of step `h` straddles a C^1 kink for every sweep point within `h` of
it, so a sweep sampled finer than `h` must produce a run `2h` wide.  The criterion is
therefore the one that separates a kink from a wrong gradient — shrink `h` and count
again.  A kink's window is proportional to `h`; a wrong gradient does not care what `h`
is.

G7 IS THE ONE THAT FOUND SOMETHING, AND ITS FIRST CRITERION WAS WRONG
----------------------------------------------------------------------
Phase is not a gene, so Stage 3 never needs d(drop)/d(phase) — but the master plan's
risk #7 is contact faceting on the rim discretisation, which the phase quadrature would
then chase.  The criterion written down first was the worst second difference of
force(phase) against its own median.  Measured, that ratio is 6 at 120 samples and 29 at
400: it grows with sampling, because it is dominated by how much the WHEEL'S curvature
varies over a period.  A statistic that worsens the harder you look at a fixed physical
curve is not measuring the discretisation, so it was replaced — recorded here rather
than quietly relaxed, exactly as M6 recorded its unmeetable penalty-plateau criterion.

What replaced it measures the thing itself: the slope of force(phase), detrended, over a
window sampled far finer than the contact quadrature spacing — and then REFINED, because
an artefact must shrink with the mesh while a property of the wheel must not.

The faceting is real and it does refine away.  What it exposes is that the master plan's
own mitigation for risk #7 — "set rim N_theta for >= 8 nodes in the patch" — was written
when the patch was ASSUMED to be 3 deg wide.  M6 measured 0.484, so no config in this
project reaches even two nodes across the patch, and the phase quadrature M8 plans to
use cannot assume the spectral accuracy claimed for it until the rim resolves the patch.
That is a specification for M8, produced the same way M6's dead-gene finding was: by
measuring something the plan expected to be uninteresting.
=============================================================================
"""

import argparse
import json
import os

import project_paths as PP
import time

import numpy as np

import _gate_guard

import jax_config  # noqa: F401
import jax
import jax.numpy as jnp

import wheel_adjoint as WA
import wheel_fea as W
import wheel_fem as fem
import wheel_genome as wg
import wheel_wheel as WW

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = "coarse"
SERVICE_FORCE_N = W.TOTAL_FORCE_NEWTONS

# The unrolled-Newton reference mesh.  Order 1 on purpose; see the module docstring.
TINY = WW.WheelConfig("tiny", 3, 1, 1, 1, 1, 1, 1, order=1, n_curve=200)
TINY_DELTA0_MM = 0.1              # secant seed only; see `run_unrolled`

# Written down BEFORE the study was run, per the project's rule.
GATE_UNROLLED_REL = 1.0e-8        # G1  adjoint vs differentiating the solve itself
GATE_RESIDUAL_REL = 1.0e-12       # G2  grad_u Pi is the residual Newton drove to zero
GATE_MESH_COORDS_MM = 1.0e-9      # G3  the differentiable mesh IS the solved mesh
GATE_FD_PLATEAU_REL = 1.0e-4      # G4  consecutive FD steps agreeing
GATE_PLATEAU_DECADES = 1          # G4  how wide the agreement must be
GATE_DIRECTIONAL_REL = 1.0e-5     # G5  random directions catch per-gene cancellation
GATE_SWEEP_REL = 1.0e-3           # G6  what counts as an outlier along a dense sweep
GATE_FACET_MUST_FALL = True       # G7  see `run_phase_smoothness` — the criterion
                                  #     written here first measured the wrong thing and
                                  #     the data said so; both are recorded there
GATE_SECANT_REL = 1.0e-5          # G9  looser: the secant's own 1e-8 enters the value
INSENSITIVE_EXPECTED = ("R_hub", "R_rim")   # G8  the census, not a tolerance
GATE_FILLET_MESH_MM = 1.0e-9      # G11 the same identity as G3, on the FILLETED mesh

# WHICH LAYER PROFILE G11 IS MEASURED AT, NAMED RATHER THAN INHERITED (PLAN §85).
#
# §79 measured the filleted mesh's differentiability at what was then `fillet=True`'s
# default -- the shipped pair -- and §85 moved that default onto the per-genome rule.  Two
# things followed and neither was optional.  The rule's entry is `FILLET_LAYER_CLIFF_FACTOR
# * cliff(genes)`, which did NOT follow the genes on the frozen path, so `mesh_coords`
# refused those meshes outright: G11 would not have run at all.  And even if it had,
# silently re-measuring §79's numbers on a different mesh would leave this file reporting
# one section's gate against another section's geometry.
#
# THE FIRST REASON IS GONE AND THE SECOND IS NOT (§88).  The cliff is differentiable now
# and the rule is measured, in G11f, on its own rows.  G11a-e stay on the named pair: the
# gate they carry is §79's, and moving it onto a different geometry in the same change that
# made that geometry available is how a threshold stops being comparable to itself.
G11_LAYER_PROFILE = WW.FILLET_LAYER_SHIPPED
GATE_FILLET_JAC_REL = 1.0e-6      # G11 the mesh jacobian against a central difference of
                                  #     `build_wheel(fillet=True)` — which re-runs both
                                  #     bracketed root-finds, so it cannot share a bug
                                  #     with the traced path
FILLET_LIVE_EXPECTED = ()         # G11 the census, INVERTED: nothing is dead here

# THE GATES ABOVE ARE NOT PER-KINEMATICS, AND THAT IS THE POINT OF `--kinematics`.
# M7 has only ever been run under linear kinematics, because `wheel_contact_problem`
# defaults to it and no section here overrode the default.  So the adjoint Stage 3 would
# use under `--kinematics svk` is unproven — see SVK_PLAN.md step 1, which exists to prove
# it against these same numbers.  The tolerances are NOT relaxed for SVK: an adjoint is
# either the derivative of its own solve or it is not, and G1 in particular contains no
# finite difference for a step size to excuse.
#
# `linear` REMAINS THE DEFAULT and that is load-bearing rather than cautious:
# `study_gradient.json` in the tree is a committed artifact measured under it, and a
# changed default would re-baseline every number in it silently.
DEFAULT_KINEMATICS = "linear"

# Genes carried through the expensive per-gene sections in --quick.  One curvature gene,
# one taper gene, and one fillet radius: the last is there so that quick mode still
# exercises the insensitive branch, which is where a classifier bug would hide.
QUICK_GENES = (6, 8, 12)


def load_genes(path="best_solution.json"):
    with open(os.path.join(PP.ROOT, path)) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


def _ranges():
    return wg.bounds_arrays(W.GENE_SPACE)[2]


def _service_indentation(genes, cfg, kinematics=DEFAULT_KINEMATICS):
    """The indentation that carries service load, as the point every check is taken at.

    Taken once and reused: the per-gene sections differentiate the contact FORCE at a
    fixed indentation, which is what M6's G7 established as the right primitive — the
    drop at fixed force runs a secant loop whose tolerance would enter the difference.

    `kinematics` reaches here too, and it has to: the SVK equilibrium sits at a different
    indentation from the linear one (2.409 mm against 1.953 on the shipped genome — see
    SVK_PLAN.md), so taking the SVK gradient at the LINEAR indentation would check the
    adjoint at a state the wheel never occupies.  Every section takes its point from this
    function for exactly that reason.
    """
    return float(fem.solve_wheel_contact(WW.build_wheel(genes, cfg),
                                         force=SERVICE_FORCE_N,
                                         kinematics=kinematics)["axle_drop_mm"])


# ---------------------------------------------------------------------------
# G1 — THE ADJOINT AGAINST DIFFERENTIATING THE SOLVE ITSELF
# ---------------------------------------------------------------------------

def run_unrolled(genes, indentation_mm=None, cold_iters=14, warm_iters=(1, 2, 3),
                 kinematics=DEFAULT_KINEMATICS):
    """Unroll the Newton loop on `tiny` and differentiate it with `jax.grad`.

    No finite difference anywhere, so this is the only check in the file whose tolerance
    is set by linear algebra rather than by step size — which is why it is first and why
    it is the one to fix before reading any other number in this report.
    """
    genes = np.asarray(genes, dtype=float)
    mesh = WW.build_wheel(genes, TINY)
    if indentation_mm is None:
        # `delta0` is given explicitly because the default seed comes from
        # `solve_wheel`'s assumed 3 deg patch, and `tiny` has 10 deg of rim per segment
        # — it cannot resolve that patch and `wheel_problem` rightly refuses.  The seed
        # is a starting point for the secant, not a claim about the wheel.
        indentation_mm = float(fem.solve_wheel_contact(
            mesh, force=SERVICE_FORCE_N, delta0=TINY_DELTA0_MM,
            kinematics=kinematics)["axle_drop_mm"])

    prob = fem.wheel_contact_problem(mesh, indentation_mm=indentation_mm,
                                     kinematics=kinematics)
    res = fem.solve_nonlinear(prob, max_iter=60)
    adjoint = WA.adjoint_grad(prob, mesh, res["u"], genes, "contact_force")

    con = prob.contact
    Pi_k, _, _ = WA._kernels(prob.order, prob.nonlinear, con.n_quad, con.smoothing_mm)
    args = WA._static_args(prob)
    y = float(con.y_ground)
    T = jnp.asarray(prob.T.toarray())
    u_pre = jnp.asarray(prob.u_pre)
    Q = WA.QOI["contact_force"](prob)
    g_j = jnp.asarray(genes)
    u_star = jnp.asarray(res["u_reduced"])

    def pi_red(v, u_r):
        c = WW.mesh_coords(v, mesh)
        return Pi_k(c, T @ u_r + u_pre, y, *args)

    d_du = jax.grad(pi_red, argnums=1)
    d2_du2 = jax.hessian(pi_red, argnums=1)

    def unrolled(v, u0, n_iter):
        u_r = u0
        for _ in range(int(n_iter)):
            u_r = u_r - jnp.linalg.solve(d2_du2(v, u_r), d_du(v, u_r))
        c = WW.mesh_coords(v, mesh)
        return Q(c, T @ u_r + u_pre, y)

    def rel(d):
        d = np.asarray(d)
        a = adjoint["grad"]
        scale = np.maximum(np.abs(a), np.abs(d))
        live = scale > 0.0
        out = np.zeros_like(scale)
        out[live] = np.abs(d[live] - a[live]) / scale[live]
        return d, float(out.max())

    zero = jnp.zeros_like(u_star)
    cold, cold_rel = rel(jax.grad(unrolled)(g_j, zero, cold_iters))
    cold_value = float(unrolled(g_j, zero, cold_iters))
    warm = {}
    for n in warm_iters:
        d, r = rel(jax.grad(unrolled)(g_j, u_star, n))
        warm[str(n)] = {"grad": [float(x) for x in d], "worst_rel": r}

    return {
        "config": "tiny", "order": int(TINY.order),
        "kinematics": kinematics, "nonlinear": bool(prob.nonlinear),
        "n_reduced_dof": int(prob.T.shape[1]),
        "indentation_mm": float(indentation_mm),
        "adjoint_grad": [float(x) for x in adjoint["grad"]],
        "adjoint_value": adjoint["value"],
        "cold": {"iterations": int(cold_iters), "grad": [float(x) for x in cold],
                 "value": cold_value, "worst_rel": cold_rel,
                 "value_rel": abs(cold_value / adjoint["value"] - 1.0)},
        "warm": warm,
        "pass": bool(cold_rel < GATE_UNROLLED_REL),
    }


# ---------------------------------------------------------------------------
# G2/G3 — THE TWO IDENTITIES THE WHOLE CONSTRUCTION RESTS ON
# ---------------------------------------------------------------------------

def run_identities(genes, cfg=DEFAULT_CONFIG, configs=("smoke", "coarse"),
                   indentation_mm=1.65, kinematics=DEFAULT_KINEMATICS):
    """Is the differentiated potential the solved one, and the differentiated mesh the
    solved mesh?

    Both are identities rather than approximations, so they are checked at machine
    precision.  They are cheap and they are the two places where a refactor could
    silently point the gradient at a different problem: `grad_u Pi` drifting from the
    assembled residual would mean the adjoint linearises an operator Newton never used,
    and `mesh_coords` drifting from `build_wheel` would mean it moves the nodes of a
    mesh that was never solved.
    """
    out = {"mesh_coords": [], "residual": [], "contact_force": []}

    for name in configs:
        mesh = WW.build_wheel(genes, name)
        for phase in (0.0, 7.5):
            m = mesh if phase == 0.0 else WW.build_wheel(genes, name, phase_deg=phase)
            c = np.asarray(WW.mesh_coords(jnp.asarray(genes), m))
            out["mesh_coords"].append(
                {"config": name, "phase_deg": phase,
                 "max_abs_mm": float(np.abs(c - np.asarray(m.coords)).max())})

        prob = fem.wheel_contact_problem(mesh, indentation_mm=indentation_mm,
                                         kinematics=kinematics)
        res = fem.solve_nonlinear(prob, max_iter=60)
        u = res["u"]
        con = prob.contact

        _, grad_u, _ = WA._kernels(prob.order, prob.nonlinear, con.n_quad,
                                   con.smoothing_mm)
        r_ad = np.asarray(grad_u(jnp.asarray(prob.coords), jnp.asarray(u),
                                 float(con.y_ground), *WA._static_args(prob)))
        r_np = fem.internal_force(prob.coords, prob.conn, u, order=prob.order,
                                  lam=prob.lam, mu=prob.mu, width=prob.width,
                                  nonlinear=prob.nonlinear) + con.force(prob.coords, u)
        scale = max(np.abs(r_np).max(), 1e-300)
        out["residual"].append({"config": name,
                                "max_abs_rel": float(np.abs(r_ad - r_np).max() / scale)})

        # The contact force is dPi/dy_ground; `total_force` integrates the pressure.
        # Two routes to one number, neither derived from the other.
        f_energy = float(WA.QOI["contact_force"](prob)(
            jnp.asarray(prob.coords), jnp.asarray(u), float(con.y_ground)))
        f_quad = con.total_force(prob.coords, u)
        out["contact_force"].append({"config": name, "from_dPi_dy": f_energy,
                                     "from_quadrature": f_quad,
                                     "rel": abs(f_energy / f_quad - 1.0)})

    out["kinematics"] = kinematics
    out["worst_mesh_coords_mm"] = max(r["max_abs_mm"] for r in out["mesh_coords"])
    out["worst_residual_rel"] = max(r["max_abs_rel"] for r in out["residual"])
    out["worst_force_rel"] = max(r["rel"] for r in out["contact_force"])
    out["pass"] = bool(out["worst_mesh_coords_mm"] < GATE_MESH_COORDS_MM
                       and out["worst_residual_rel"] < GATE_RESIDUAL_REL
                       and out["worst_force_rel"] < GATE_RESIDUAL_REL * 1e6)
    return out


# ---------------------------------------------------------------------------
# G4/G8 — THE FD PLATEAU, AND THE GENES THAT HAVE NO DERIVATIVE TO PLATEAU
# ---------------------------------------------------------------------------

def run_plateau(genes, cfg=DEFAULT_CONFIG, indentation_mm=None, gene_ids=None,
                steps=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
                kinematics=DEFAULT_KINEMATICS):
    """Central differences of the contact force in each gene, over a ladder of steps.

    Steps are scaled by the GENE'S OWN RANGE, as the master plan specifies: `cy` spans
    64 mm and `R_rim` spans 2.5 mm, so a shared absolute step is a different question
    asked of each gene.

    THE INSENSITIVE GENES ARE CLASSIFIED BEFORE THE PLATEAU LOGIC RUNS, because a run of
    identical zeros is indistinguishable from a perfect plateau and would otherwise be
    reported as the cleanest result in the table.  The classification here is not a
    tolerance on the differences but a statement about the mesh: `dcoords/dgene` is
    exactly zero, so no solver is involved and there is nothing for a step size to
    resolve.
    """
    genes = np.asarray(genes, dtype=float)
    mesh = WW.build_wheel(genes, cfg)
    if indentation_mm is None:
        indentation_mm = _service_indentation(genes, cfg, kinematics)
    rng = _ranges()
    if gene_ids is None:
        gene_ids = tuple(range(wg.N_GENES))

    dead_names, col = WA.insensitive_genes(genes, mesh)
    base = WA.solve_and_grad(genes, cfg, "contact_force",
                             indentation_mm=indentation_mm, mesh=mesh,
                             kinematics=kinematics)
    warm = base["res"]["u_reduced"]

    def force(v):
        return fem.solve_wheel_contact_at(WW.build_wheel(np.asarray(v), cfg),
                                          indentation_mm, u_reduced0=warm,
                                          kinematics=kinematics
                                          )["contact_force_n"]

    rows = {}
    for gi in gene_ids:
        name = wg.GENE_NAMES[gi]
        ad = float(base["grad"][gi])
        if name in dead_names:
            rows[name] = {"adjoint": ad, "insensitive": True,
                          "coord_sensitivity": float(col[gi]),
                          "steps": [], "derivatives": [], "consecutive_rel": [],
                          "plateau_decades": None, "best_rel_to_adjoint": None}
            continue
        derivs = []
        for h_rel in steps:
            h = h_rel * rng[gi]
            vp, vm = genes.copy(), genes.copy()
            vp[gi] += h
            vm[gi] -= h
            derivs.append((force(vp) - force(vm)) / (2.0 * h))
        d = np.array(derivs)
        # THE PLATEAU IS MEASURED AGAINST THE ADJOINT, not against the ladder's own
        # previous rung.  That is the master plan's wording — "< 1e-4 relative in the
        # plateau", in a section whose first item established the adjoint — and it is the
        # stronger statement: consecutive agreement says the difference has stopped
        # moving, which a systematically biased difference also satisfies.
        #
        # It also matters for the SMALL derivatives.  `cx1` moves the contact force by
        # 0.032 N/mm where `t2` moves it by 46, so the roundoff floor — set by the 66 N
        # response, not by the derivative — bites `cx1` a thousand times sooner in
        # relative terms.  Its ladder is a clean V with a single rung at the bottom
        # (1.9e-2, 1.1e-3, 2.9e-6, 1.8e-4, 6.6e-4, 9.4e-3): truncation on one side,
        # roundoff on the other.  That is a plateau one decade wide and a correct
        # gradient, and the consecutive-rung criterion scored it zero.
        rel_adj = np.abs(d / ad - 1.0)
        cons = np.abs(d[1:] / d[:-1] - 1.0)
        best_run, run = 0, 0
        for x in rel_adj:
            run = run + 1 if x < GATE_FD_PLATEAU_REL else 0
            best_run = max(best_run, run)
        rows[name] = {
            "adjoint": ad, "insensitive": False,
            "coord_sensitivity": float(col[gi]),
            "steps": [float(x) for x in steps],
            "step_mm": [float(x * rng[gi]) for x in steps],
            "derivatives": [float(x) for x in d],
            "rel_to_adjoint": [float(x) for x in rel_adj],
            "consecutive_rel": [float(x) for x in cons],
            "plateau_decades": int(best_run),
            "best_rel_to_adjoint": float(rel_adj.min()),
            "best_step": float(steps[int(np.argmin(rel_adj))]),
        }

    live = [n for n, r in rows.items() if not r["insensitive"]]
    census_ok = sorted(dead_names) == sorted(INSENSITIVE_EXPECTED)
    return {
        "config": cfg, "kinematics": kinematics,
        "indentation_mm": float(indentation_mm),
        "rows": rows,
        "insensitive_genes": list(dead_names),
        "insensitive_expected": list(INSENSITIVE_EXPECTED),
        "census_ok": bool(census_ok),
        "coord_sensitivity": {wg.GENE_NAMES[i]: float(col[i]) for i in range(len(col))},
        "worst_plateau_decades": (min(rows[n]["plateau_decades"] for n in live)
                                  if live else 0),
        "worst_rel_to_adjoint": (max(rows[n]["best_rel_to_adjoint"] for n in live)
                                 if live else float("inf")),
        "pass": bool(live and census_ok
                     and min(rows[n]["plateau_decades"] for n in live)
                     >= GATE_PLATEAU_DECADES
                     and max(rows[n]["best_rel_to_adjoint"] for n in live)
                     < GATE_FD_PLATEAU_REL),
    }


# ---------------------------------------------------------------------------
# G5 — RANDOM DIRECTIONS
# ---------------------------------------------------------------------------

def run_directional(genes, cfg=DEFAULT_CONFIG, indentation_mm=None, n=10, seed=0,
                    steps=(1e-4, 1e-5, 1e-6), kinematics=DEFAULT_KINEMATICS):
    """Directional derivatives along random unit vectors in normalised gene space.

    Per-gene differences can agree one at a time and still be wrong together: a
    transposed index or a mis-scaled column cancels in every single-gene difference and
    shows up the moment several genes move at once.  Normalised because the directions
    have to be isotropic in the space the OPTIMIZER moves in, not in millimetres.

    OVER A LADDER OF STEPS, NOT AT ONE.  Run at a single `h_rel = 1e-4` this reported a
    uniform 2-8e-5 disagreement in all ten directions and failed — and the ladder is what
    identified that as the FINITE DIFFERENCE'S truncation error rather than the
    gradient's: it falls 1.2e-2, 8.3e-5, 2.1e-5, 6.2e-7 as the step shrinks by decades,
    which is a property of the reference and not of the thing being checked.  A
    single-step check has no plateau, which is the master plan's own criticism of
    single-point agreement applied to its own item 3.
    """
    genes = np.asarray(genes, dtype=float)
    if indentation_mm is None:
        indentation_mm = _service_indentation(genes, cfg, kinematics)
    rng = _ranges()
    base = WA.solve_and_grad(genes, cfg, "contact_force",
                             indentation_mm=indentation_mm, kinematics=kinematics)
    warm = base["res"]["u_reduced"]
    grad = base["grad"]
    gen = np.random.default_rng(seed)

    def force(v):
        return fem.solve_wheel_contact_at(WW.build_wheel(np.asarray(v), cfg),
                                          indentation_mm, u_reduced0=warm,
                                          kinematics=kinematics
                                          )["contact_force_n"]

    rows = []
    for _ in range(int(n)):
        d = gen.normal(size=wg.N_GENES)
        d /= np.linalg.norm(d)
        rels, fds, ads = [], [], []
        for h_rel in steps:
            step = h_rel * rng * d                  # normalised direction, raw units
            ad = float(grad @ step)
            fd = 0.5 * (force(genes + step) - force(genes - step))
            ads.append(ad)
            fds.append(fd)
            rels.append(float(abs(fd - ad) / max(abs(ad), 1e-300)))
        k = int(np.argmin(rels))
        rows.append({"steps": [float(s) for s in steps], "rel_ladder": rels,
                     "adjoint": ads[k], "fd": fds[k], "rel": rels[k],
                     "best_step": float(steps[k])})
    worst = max(r["rel"] for r in rows)
    return {"rows": rows, "n": int(n), "steps": [float(s) for s in steps],
            "kinematics": kinematics,
            "worst_rel": worst, "pass": bool(worst < GATE_DIRECTIONAL_REL)}


# ---------------------------------------------------------------------------
# G6 — THE DENSE SWEEP, WHICH IS WHERE A STAIRCASE WOULD SHOW
# ---------------------------------------------------------------------------

def run_dense_sweep(genes, cfg="smoke", gene_id=6, n=400, span_rel=0.02,
                    indentation_mm=None, steps=(1e-4, 1e-5),
                    kinematics=DEFAULT_KINEMATICS):
    """Adjoint vs central difference at every point of a fine sweep across one gene.

    The master plan asks for a 400-point sweep that is visibly smooth.  A plot is not a
    gate.  Comparing the adjoint against a difference at every point costs the same
    solves, is strictly stronger, and LOCALISES a leak rather than merely reporting one.

    THE OUTLIERS COME IN SHORT RUNS, AND THAT IS THE SIGNATURE RATHER THAN A PROBLEM.
    The first version of this gate allowed a few isolated bad points and no adjacent
    pair, on the reasoning that a C^1 kink corrupts one sample.  It corrupts a WINDOW: a
    central difference of step `h` straddles a kink for every sweep point within `h` of
    it, so a sweep sampled finer than `h` must produce a run, and the run is `2h` wide.
    Measured at 400 points across 2% of `cx4`, the bad points fall in clusters of about
    four at a regular spacing — which is what isolated kinks at discrete gene values look
    like when observed with a finite difference, not what a broken gradient looks like.

    So the criterion is the one that distinguishes those two: SHRINK `h` AND COUNT AGAIN.
    A kink's window is proportional to `h` and its outlier count must fall with it; a
    genuinely wrong gradient does not care what `h` is.  That is a measurement rather
    than a threshold, and it is why two step sizes are run.
    """
    genes = np.asarray(genes, dtype=float)
    rng = _ranges()
    if indentation_mm is None:
        indentation_mm = _service_indentation(genes, cfg, kinematics)
    half = 0.5 * span_rel * rng[gene_id]
    xs = np.linspace(genes[gene_id] - half, genes[gene_id] + half, int(n))

    warm = None
    ad, val, fds = [], [], [[] for _ in steps]
    for x in xs:
        v = genes.copy()
        v[gene_id] = x
        out = WA.solve_and_grad(v, cfg, "contact_force",
                                indentation_mm=indentation_mm, u_reduced0=warm,
                                kinematics=kinematics)
        warm = out["res"]["u_reduced"]
        ad.append(float(out["grad"][gene_id]))
        val.append(float(out["value"]))

        def force(vv):
            return fem.solve_wheel_contact_at(WW.build_wheel(vv, cfg), indentation_mm,
                                              u_reduced0=warm,
                                              kinematics=kinematics)["contact_force_n"]
        for j, h_rel in enumerate(steps):
            h = h_rel * rng[gene_id]
            vp, vm = v.copy(), v.copy()
            vp[gene_id] += h
            vm[gene_id] -= h
            fds[j].append((force(vp) - force(vm)) / (2.0 * h))

    ad = np.array(ad)
    ladders = []
    for j, h_rel in enumerate(steps):
        fd = np.array(fds[j])
        rel = np.abs(fd - ad) / np.maximum(np.abs(ad), 1e-300)
        bad = np.flatnonzero(rel >= GATE_SWEEP_REL)
        runs, run = [], 0
        for i in range(len(rel)):
            if rel[i] >= GATE_SWEEP_REL:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)
        ladders.append({
            "h_rel": float(h_rel), "fd": fd.tolist(), "rel": rel.tolist(),
            "median_rel": float(np.median(rel)), "worst_rel": float(rel.max()),
            "n_outliers": int(bad.size), "n_clusters": len(runs),
            "longest_run": int(max(runs, default=0)),
            "outlier_positions": [float(xs[i]) for i in bad[:20]],
        })

    shrinks = all(ladders[j + 1]["n_outliers"] < ladders[j]["n_outliers"]
                  for j in range(len(ladders) - 1))
    return {
        "config": cfg, "gene": wg.GENE_NAMES[gene_id], "gene_id": int(gene_id),
        "kinematics": kinematics,
        "n": int(n), "span_rel": float(span_rel), "steps": [float(s) for s in steps],
        "indentation_mm": float(indentation_mm),
        "x": xs.tolist(), "value": val, "adjoint": ad.tolist(),
        "ladders": ladders,
        "outliers_shrink_with_step": bool(shrinks),
        "median_rel": ladders[-1]["median_rel"],
        "pass": bool(shrinks and ladders[-1]["median_rel"] < GATE_FD_PLATEAU_REL),
    }


# ---------------------------------------------------------------------------
# G7 — THE PHASE SWEEP, AND THE FACETING THAT DOES LIVE IN IT
# ---------------------------------------------------------------------------

def _phase_sweep(genes, cfg, phases, indentation_mm, kinematics=DEFAULT_KINEMATICS):
    """Contact force at fixed indentation over a list of phases, warm-started along.

    THE WARM START IS A SPEED OPTIMISATION AND UNDER SVK IT CAN LAND OUTSIDE THE BASIN.
    Measured on the shipped genome at `smoke`, 60 phases, indentation 2.2095 mm: the chain
    runs to phase 25.0 and then refuses with `Newton direction is not a descent direction
    (slope 5.662e+02) ... the tangent is not positive definite`, while a COLD solve at that
    identical phase converges without complaint to 71.9097 N.  So the equilibrium exists
    and is reachable; what failed is the guess it was approached from.

    Under LINEAR kinematics this cannot happen — `assemble_stiffness`'s Hessian is
    independent of `u` (its own docstring says so, and PLAN.md §0(1) H2(a) measured the
    difference at exactly 0.000e+00), so a bad starting guess costs iterations and nothing
    else.  Under SVK the tangent is a function of the state, and a state carried across a
    0.5 deg phase jump is not guaranteed to sit anywhere near the one it is being used to
    find.

    Falling back to cold is therefore the right repair rather than a workaround: it
    changes which starting guess is used and not which equilibrium is reported.  The
    phases where it fired are RETURNED rather than swallowed, because "the warm start
    stopped working here" is exactly the kind of observation this repo has been bitten by
    losing.
    """
    warm, vals, patch, retried = None, [], [], []
    for p in phases:
        mesh = WW.build_wheel(genes, cfg, phase_deg=float(p))
        try:
            res = fem.solve_wheel_contact_at(mesh, indentation_mm, u_reduced0=warm,
                                             kinematics=kinematics)
        except fem.NewtonDivergedError:
            res = fem.solve_wheel_contact_at(mesh, indentation_mm,
                                             kinematics=kinematics)
            retried.append(float(p))
        warm = res["u_reduced"]
        vals.append(res["contact_force_n"])
        patch.append(res["patch_half_deg"])
    return np.array(vals), np.array(patch), retried


def _rim_node_spacing_deg(mesh):
    """Angular spacing of the rim's contact-boundary NODES, in degrees.

    Counted off the mesh's own `rim_outer` edge set rather than derived from the config's
    element counts, so it cannot drift from what was actually meshed.

    This is the number the master plan's risk #7 mitigation is written in terms of — "set
    rim N_theta for >= 8 nodes in the patch" — and that rule was written when the patch
    was ASSUMED to be 3 deg wide.  M6 measured 0.484.
    """
    n_seg = int(np.asarray(mesh.edge_sets["rim_outer"]).shape[0])
    return 360.0 / (n_seg * mesh.cfg.order)


def run_phase_smoothness(genes, configs=("smoke", "coarse"), n_period=200,
                         window_deg=3.0, window_step_deg=0.05, indentation_mm=None,
                         kinematics=DEFAULT_KINEMATICS):
    """Does the contact facet on the rim discretisation as the wheel rolls?

    THE CRITERION WRITTEN DOWN BEFORE THE RUN MEASURED THE WRONG THING, AND THE DATA IS
    WHAT SAID SO.  It asked for the worst second difference of force(phase) against its
    own median to stay under 10x.  Measured, that ratio is 6 at 120 points and 29 at 400
    — it GROWS with sampling, because it is dominated by how much the wheel's real
    curvature varies over a period, which is a property of the wheel and not of the
    discretisation.  A statistic that gets worse the harder you look at a fixed physical
    curve is not measuring the discretisation.  Recorded rather than quietly relaxed, the
    way M6's penalty-plateau gate was.

    What faceting actually is: the SLOPE picking up a ripple at the spacing of the
    contact quadrature points.  So the sweep is taken over a window at a step far finer
    than that spacing, the slope is detrended, and the residual is reported as a fraction
    of the mean slope.  And the test that settles whether it is an artefact is
    REFINEMENT: a mesh-scale ripple must shrink when the rim is refined, while a real
    feature of the wheel must not.  That is the gate.

    The window is chosen as the steepest `window_deg` of the period rather than fixed,
    because a facet is a perturbation of the slope and is least ambiguous where there is
    a slope to perturb.
    """
    genes = np.asarray(genes, dtype=float)
    lead = configs[0]
    if indentation_mm is None:
        indentation_mm = _service_indentation(genes, lead, kinematics)

    # -- the whole period once, for the ripple and to locate the window
    phases = np.linspace(0.0, 30.0, int(n_period), endpoint=False)
    v_period, _, retried_period = _phase_sweep(genes, lead, phases, indentation_mm,
                                               kinematics)
    w = max(2, int(round(window_deg / (30.0 / n_period))))
    drop = np.array([abs(v_period[min(i + w, n_period - 1)] - v_period[i])
                     for i in range(n_period - 1)])
    start = float(phases[int(np.argmax(drop))])

    win = np.arange(start, start + window_deg + 1e-9, window_step_deg)
    rows = []
    for name in configs:
        v, patch, retried = _phase_sweep(genes, name, win, indentation_mm, kinematics)
        slope = np.gradient(v, win)
        # Detrend with a moving average wide enough to average over the quadrature
        # spacing but narrow enough to follow the real curve.
        spacing = _rim_node_spacing_deg(WW.build_wheel(genes, name))
        k = max(3, int(round(2.0 * spacing / window_step_deg)) | 1)
        pad = k // 2
        trend = np.convolve(np.pad(slope, pad, mode="edge"), np.ones(k) / k,
                            mode="valid")
        resid = slope - trend
        mean_slope = float(np.mean(np.abs(trend)))
        rows.append({
            "config": name,
            "rim_node_spacing_deg": float(spacing),
            "quadrature_spacing_deg": float(spacing * 2.0 / 6.0),
            "patch_half_deg": float(np.nanmedian(patch)),
            "nodes_in_patch": float(2.0 * np.nanmedian(patch) / spacing),
            "mean_abs_slope": mean_slope,
            "facet_amplitude": float(np.std(resid)),
            "facet_fraction": float(np.std(resid) / max(mean_slope, 1e-300)),
            "slope_min": float(slope.min()), "slope_max": float(slope.max()),
            "phase_deg": win.tolist(), "force_n": v.tolist(),
            "cold_retry_phases_deg": retried,
        })

    falls = all(rows[i + 1]["facet_fraction"] < rows[i]["facet_fraction"]
                for i in range(len(rows) - 1))
    return {
        "configs": list(configs), "kinematics": kinematics,
        "indentation_mm": float(indentation_mm),
        "window_start_deg": start, "window_deg": float(window_deg),
        "window_step_deg": float(window_step_deg),
        "period_phase_deg": phases.tolist(), "period_force_n": v_period.tolist(),
        "peak_to_peak_over_mean": float((v_period.max() - v_period.min())
                                        / v_period.mean()),
        "rows": rows,
        "cold_retry_phases_deg": retried_period,
        "n_cold_retries": len(retried_period) + sum(len(r["cold_retry_phases_deg"])
                                                    for r in rows),
        "facet_falls_with_refinement": bool(falls),
        "facet_ratio": (rows[0]["facet_fraction"] / rows[-1]["facet_fraction"]
                        if len(rows) > 1 and rows[-1]["facet_fraction"] > 0 else None),
        "pass": bool(len(rows) > 1 and falls),
    }


# ---------------------------------------------------------------------------
# G9/G10 — THE LOAD-CONTROLLED GRADIENT, AND WHAT IT COSTS
# ---------------------------------------------------------------------------

def run_axle_drop(genes, cfg=DEFAULT_CONFIG, gene_ids=(6, 8, 12),
                  steps=(1e-4, 1e-5, 1e-6), kinematics=DEFAULT_KINEMATICS,
                  fd_tol_rel=1e-9, fd_max_iter=40):
    """The quantity Stage 3 actually optimises, against a finite difference of the whole
    load-controlled solve — the only check here that exercises the secant.

    THE SECANT'S TOLERANCE **IS** WHAT LIMITS THE REFERENCE, AND THIS DOCSTRING USED TO
    SAY IT WAS NOT.  The old text read: "Tightening it to 1e-11 moves the difference by
    nothing at all — ten identical digits — so the residue is ordinary truncation in the
    outer difference."  That measurement was real, and it was taken on the mesh of the
    day.  It does not generalise.  On the 2026-08-18 uncap default the same experiment
    gives the opposite answer, on gene `cx4` at `smoke`:

        fd tol_rel   1e-8            1e-9            1e-10        rel vs adjoint
        capped       -0.0307547723   -0.0307547723   -0.0307547723   1.85e-06
        hub only     -0.0303006155   -0.0303006155   -0.0303006155   3.29e-06
        rim only     -0.0212447596   -0.0212447596   -0.0212447596   4.75e-06
        BOTH         -0.0209239528   -0.0209216681   -0.0209216681   1.07e-04 -> 1.99e-06

    One decade on the REFERENCE moves G9 from 10x over its gate to 5x under it, and only
    on the one setting where both rings are uncapped.  So `fd_tol_rel` is a parameter now
    rather than an inherited default, and it is set where the reference has stopped
    depending on it.

    THE ADJOINT WAS NEVER THE PROBLEM, and that is checkable independently: split the
    quotient and finite-difference each half separately, and `dF/d delta` agrees to
    2.79e-06 and `dF/dp` to 2.03e-06 on that same setting — both already as good as
    capped.  It was the whole-solve reference that was under-converged, in the one place
    a stopping rule can still reach the derivative.

    That is still the point of computing the gradient by the implicit-function quotient:
    the secant's tolerance lands in the VALUE, where it is a stated 1e-8.  What this
    correction adds is that a FINITE-DIFFERENCE REFERENCE built on the same solve does
    NOT get that protection — it differences two separately-terminated secants, and their
    termination bias does not cancel.  A tolerance that is invisible in one mesh's
    reference can be first-order in another's.  Do not re-derive "the secant does not
    matter" from a single mesh again.

    NOT LOOSENED: `GATE_SECANT_REL` is untouched at 1e-5.  This tightens the reference the
    gate is measured against, in the direction that makes the check harder to pass by
    accident.  1e-11 STALLS the secant ("indentations 1.477174 and 1.477174 give the same
    contact force"), which is the float64 floor on the force, so 1e-9 is two decades of
    margin from where it breaks and one decade past where it stops mattering.
    """
    genes = np.asarray(genes, dtype=float)
    rng = _ranges()
    out = WA.axle_drop_value_and_grad(genes, cfg, force=SERVICE_FORCE_N,
                                      kinematics=kinematics)

    def drop(v):
        return fem.solve_wheel_contact(WW.build_wheel(v, cfg),
                                       force=SERVICE_FORCE_N,
                                       tol_rel=fd_tol_rel, max_iter=fd_max_iter,
                                       kinematics=kinematics)["axle_drop_mm"]

    rows = []
    for gi in gene_ids:
        ad = float(out["grad"][gi])
        rels, fds = [], []
        for h_rel in steps:
            h = h_rel * rng[gi]
            vp, vm = genes.copy(), genes.copy()
            vp[gi] += h
            vm[gi] -= h
            fd = (drop(vp) - drop(vm)) / (2.0 * h)
            fds.append(float(fd))
            rels.append(float(abs(fd - ad) / max(abs(ad), 1e-300))
                        if ad != 0.0 else float(abs(fd)))
        k = int(np.argmin(rels))
        rows.append({"gene": wg.GENE_NAMES[gi], "adjoint": ad,
                     "steps": [float(s) for s in steps], "fd_ladder": fds,
                     "rel_ladder": rels, "fd": fds[k], "rel": rels[k],
                     "best_step": float(steps[k])})
    live = [r for r in rows if r["adjoint"] != 0.0]
    worst = max((r["rel"] for r in live), default=float("inf"))
    return {
        "config": cfg, "kinematics": kinematics, "axle_drop_mm": out["value"],
        "fd_tol_rel": float(fd_tol_rel),
        "contact_force_n": out["contact_force_n"],
        "d_force_d_indentation": out["d_force_d_indentation"],
        "grad": [float(x) for x in out["grad"]],
        "rows": rows, "worst_rel": worst,
        "pass": bool(worst < GATE_SECANT_REL),
    }


# ---------------------------------------------------------------------------
# G11 — THE FILLETED MESH, WHICH HAD NO DERIVATIVE AT ALL UNTIL 2026-08-24
# ---------------------------------------------------------------------------

def run_filleted(genes, cfg=DEFAULT_CONFIG, configs=("smoke", "coarse"),
                 gene_ids=(8, 12, 13), jac_steps=(1e-4, 1e-5, 1e-6),
                 drop_gene_ids=(12, 13), drop_steps=(1e-4, 1e-5, 1e-6),
                 kinematics=DEFAULT_KINEMATICS, fd_tol_rel=1e-9, fd_max_iter=40):
    """G8's census, INVERTED — and the two genes it names now have a gradient.

    G8 reports that `R_hub` and `R_rim` have an identically zero derivative because the
    mesh models no fillets, and calls that "sharper than M6's version" because the zero
    is in `dcoords/dgenes` before any solver runs.  It is still true of the mesh Stage 3
    builds.  What changed on 2026-08-24 (PLAN.md §79) is that the FILLETED mesh —
    `build_wheel(fillet=True)`, the eleven-block sector blocking — is no longer refused by
    `mesh_coords`, so the same census can be taken on a mesh where those two genes move
    material, and the answer is that NOTHING is dead.

    THE REFERENCE IS THE EAGER BUILD, NOT THE TRACED PATH'S OWN ARITHMETIC.  Both of the
    filleted construction's root-finds are bracketed and are replaced under trace by one
    Newton step from a frozen seed (`wheel_wheel._newton_from_root`), so the risk that
    matters is a derivative that is self-consistent and wrong.  Central-differencing
    `build_wheel(fillet=True)` cannot be self-consistent with it: it re-scans, re-brackets
    and re-bisects at every perturbed genome, and it compares the WHOLE coordinate array
    rather than a scalar that could cancel.

    G11f IS THE ONE MEASURED AT THE RULE RATHER THAN AT THE NAMED PAIR (PLAN §88).
    `fillet=True` bare takes the per-genome layer profile, whose entry is a function of the
    genes; §85 refused a gradient through it and §88 made the cliff a closed form and
    differentiated it.  The row that says whether that was worth doing is not `rel_rule` —
    which only says the new path is right — but `rel_frozen`, the jacobian §85 would have
    returned had it not refused, measured against the same central difference.

    AND THE END OF THE CHAIN IS THE AXLE DROP, for the same reason G9 exists: a mesh
    derivative that is right and a solve that never sees it is not a gradient.  Section 75
    priced this arm at 12.49% of axle drop over `R_hub`'s feasible range, measured as a
    SENSITIVITY — a sweep of separate solves.  These rows are the derivative of the same
    quantity at a point, against a finite difference of the whole load-controlled solve on
    filleted meshes, and the reference secant runs at the same tightened `fd_tol_rel` G9
    uses and for the same reason.
    """
    genes = np.asarray(genes, dtype=float)
    rng = _ranges()
    out = {"configs": list(configs), "identity": [], "gate_mm": GATE_FILLET_MESH_MM}

    # --- G11a  the differentiable filleted mesh IS the solved filleted mesh ----
    for name in configs:
        for phase in (0.0, 7.5):
            m = WW.build_wheel(genes, name, phase_deg=phase, fillet=True,
                               layer_profile=G11_LAYER_PROFILE)
            traced = np.asarray(WW.mesh_coords(jnp.asarray(genes), m))
            eager_np = np.asarray(WW.mesh_coords(genes, m, xp=np))
            out["identity"].append(
                {"config": name, "phase_deg": phase, "n_nodes": int(m.n_nodes),
                 "max_abs_mm": float(np.abs(traced - np.asarray(m.coords)).max()),
                 "numpy_path_max_abs_mm":
                     float(np.abs(eager_np - np.asarray(m.coords)).max()),
                 "radii_mm": [float(r) for r in m.fillet_radii_mm],
                 "clamped": {k: bool(v) for k, v in m.fillet_clamped.items()}})

    mesh_f = WW.build_wheel(genes, cfg, fillet=True,
                            layer_profile=G11_LAYER_PROFILE)
    mesh_u = WW.build_wheel(genes, cfg)

    # --- G11b  the census, both ways round ------------------------------------
    dead_u, col_u = WA.insensitive_genes(genes, mesh_u)
    dead_f, col_f = WA.insensitive_genes(genes, mesh_f)
    rank = sorted(range(len(col_f)), key=lambda i: -col_f[i])
    out["census"] = {
        "unfilleted_dead": list(dead_u), "filleted_dead": list(dead_f),
        "expected_unfilleted": list(INSENSITIVE_EXPECTED),
        "expected_filleted": list(FILLET_LIVE_EXPECTED),
        "coord_sensitivity_unfilleted": {wg.GENE_NAMES[i]: float(col_u[i])
                                         for i in range(len(col_u))},
        "coord_sensitivity_filleted": {wg.GENE_NAMES[i]: float(col_f[i])
                                       for i in range(len(col_f))},
        "ranked": [wg.GENE_NAMES[i] for i in rank],
        "fillet_genes_rank_first": bool(
            {wg.GENE_NAMES[rank[0]], wg.GENE_NAMES[rank[1]]} == {"R_hub", "R_rim"}),
        "ok": bool(sorted(dead_u) == sorted(INSENSITIVE_EXPECTED)
                   and sorted(dead_f) == sorted(FILLET_LIVE_EXPECTED))}

    # --- G11c  the mesh jacobian against a central difference of `build_wheel` --
    def eager_coords(v):
        return np.asarray(WW.build_wheel(np.asarray(v, float), cfg, fillet=True,
                                         layer_profile=G11_LAYER_PROFILE).coords)

    def column(gid):
        e = np.zeros(len(genes))
        e[gid] = 1.0
        _, d = jax.jvp(lambda v: WW.mesh_coords(v, mesh_f),
                       (jnp.asarray(genes),), (jnp.asarray(e),))
        return np.asarray(d)

    jac_rows = []
    for gi in gene_ids:
        ad = column(gi)
        rels, hs = [], []
        for h_rel in jac_steps:
            h = h_rel * rng[gi]
            vp, vm = genes.copy(), genes.copy()
            vp[gi] += h
            vm[gi] -= h
            fd = (eager_coords(vp) - eager_coords(vm)) / (2.0 * h)
            hs.append(float(h))
            rels.append(float(np.abs(fd - ad).max()
                              / max(float(np.abs(fd).max()), 1e-300)))
        k = int(np.argmin(rels))
        jac_rows.append({"gene": wg.GENE_NAMES[gi],
                         "max_abs_dcoord_dgene_mm": float(np.abs(ad).max()),
                         "steps": [float(s) for s in jac_steps], "h_mm": hs,
                         "rel_ladder": rels, "rel": rels[k],
                         "best_step": float(jac_steps[k])})
    out["jacobian"] = {"config": cfg, "rows": jac_rows,
                       "worst_rel": max(r["rel"] for r in jac_rows),
                       "gate_rel": GATE_FILLET_JAC_REL}

    # --- G11d  the axle drop, filleted, against the whole load-controlled solve -
    ad_f = WA.axle_drop_value_and_grad(genes, cfg, force=SERVICE_FORCE_N,
                                       mesh=mesh_f, kinematics=kinematics)
    ad_u = WA.axle_drop_value_and_grad(genes, cfg, force=SERVICE_FORCE_N,
                                       mesh=mesh_u, kinematics=kinematics)

    def drop(v):
        return fem.solve_wheel_contact(
            WW.build_wheel(np.asarray(v, float), cfg, fillet=True,
                           layer_profile=G11_LAYER_PROFILE),
            force=SERVICE_FORCE_N, tol_rel=fd_tol_rel, max_iter=fd_max_iter,
            kinematics=kinematics)["axle_drop_mm"]

    drop_rows = []
    for gi in drop_gene_ids:
        a = float(ad_f["grad"][gi])
        rels, fds = [], []
        for h_rel in drop_steps:
            h = h_rel * rng[gi]
            vp, vm = genes.copy(), genes.copy()
            vp[gi] += h
            vm[gi] -= h
            fd = (drop(vp) - drop(vm)) / (2.0 * h)
            fds.append(float(fd))
            rels.append(float(abs(fd - a) / max(abs(a), 1e-300)))
        k = int(np.argmin(rels))
        drop_rows.append({"gene": wg.GENE_NAMES[gi], "adjoint": a,
                          "unfilleted_adjoint": float(ad_u["grad"][gi]),
                          "steps": [float(s) for s in drop_steps],
                          "fd_ladder": fds, "rel_ladder": rels,
                          "fd": fds[k], "rel": rels[k],
                          "best_step": float(drop_steps[k])})
    out["axle_drop"] = {
        "config": cfg, "kinematics": kinematics, "fd_tol_rel": float(fd_tol_rel),
        "filleted_mm": float(ad_f["value"]), "unfilleted_mm": float(ad_u["value"]),
        "grad_filleted": [float(x) for x in ad_f["grad"]],
        "grad_unfilleted": [float(x) for x in ad_u["grad"]],
        "rows": drop_rows, "worst_rel": max(r["rel"] for r in drop_rows),
        "gate_rel": GATE_SECANT_REL}

    # --- G11e  what is still refused, and that it still refuses ----------------
    refusals = {}
    spoke = WW.build_wheel(genes, cfg, fillet=(0.10, 0.10), fillet_blocking="spoke")
    try:
        WW.mesh_coords(genes, spoke, xp=np)
        refusals["spoke_blocking"] = None
    except NotImplementedError as exc:
        refusals["spoke_blocking"] = str(exc)
    clamped = genes.copy()
    clamped[12] = 8.0                      # far past any sector's room; §57's case
    mc = WW.build_wheel(clamped, cfg, fillet=True,
                        layer_profile=G11_LAYER_PROFILE)
    try:
        WW.mesh_coords(clamped, mc, xp=np)
        refusals["sector_fit_clamped"] = None
    except NotImplementedError as exc:
        refusals["sector_fit_clamped"] = str(exc)
    refusals["clamped_radii_mm"] = [float(r) for r in mc.fillet_radii_mm]
    # A THIRD CASE WAS REFUSED HERE FROM §85 TO §88 AND IS NOT ANY MORE.  `fillet=True`
    # with no `layer_profile` is the per-genome rule, whose entry the frozen path held
    # constant; §85 wrote *"the day it stops refusing is the day someone made the cliff
    # differentiable and this census says so"*.  §88 made it, so the case moved OUT of this
    # dict and into G11f below, where it is measured rather than counted.  It is not
    # dropped silently: `per_genome_layer_profile` is kept as a key, pinned at `None`, so a
    # reader of the old artifact sees the same name change meaning in one place.
    refusals["per_genome_layer_profile"] = None
    refusals["per_genome_no_longer_refused_since"] = "PLAN.md §88"
    refusals["ok"] = bool(refusals["spoke_blocking"]
                          and refusals["sector_fit_clamped"])
    out["refusals"] = refusals

    # --- G11f  THE PER-GENOME RULE, WHICH G11e USED TO COUNT AS A REFUSAL ------
    #
    # `fillet=True` bare is the default a filleted build takes (§85), and its entry is
    # `FILLET_LAYER_CLIFF_FACTOR * cliff(genes)`.  Two questions, and the second is the one
    # that says whether §88 was worth doing:
    #
    #   does the traced path reproduce and differentiate the mesh the rule builds?
    #   -- `rel_rule`, the same jvp-against-eager-central-difference G11c makes.
    #
    #   HOW BIG IS THE TERM THE FROZEN PATH WOULD HAVE MISSED?  -- `rel_frozen`, the
    #   jacobian of the mesh built at the rule's pair HELD FIXED, against the central
    #   difference of the rule itself.  That is exactly the gradient §85 refused to return,
    #   and its error is the cliff's own contribution.  A small number here would mean the
    #   refusal was never worth much; it is not small.
    mp = WW.build_wheel(genes, cfg, fillet=True)
    pair = tuple(float(v) for v in mp.fillet_recipe["layer_profile"])
    mfz = WW.build_wheel(genes, cfg, fillet=True, layer_profile=pair)
    cliff = WW.layer_cliff_entry(genes, cfg)

    def rule_coords(v):
        return np.asarray(WW.build_wheel(np.asarray(v, float), cfg, fillet=True).coords)

    def col(mesh, gid):
        e = np.zeros(len(genes))
        e[gid] = 1.0
        _, d = jax.jvp(lambda v: WW.mesh_coords(v, mesh),
                       (jnp.asarray(genes),), (jnp.asarray(e),))
        return np.asarray(d)

    pg_rows = []
    for gi in gene_ids:
        a_rule, a_frozen = col(mp, gi), col(mfz, gi)
        rels_r, rels_f = [], []
        for h_rel in jac_steps:
            h = h_rel * rng[gi]
            vp, vm = genes.copy(), genes.copy()
            vp[gi] += h
            vm[gi] -= h
            fd = (rule_coords(vp) - rule_coords(vm)) / (2.0 * h)
            scale = max(float(np.abs(fd).max()), 1e-300)
            rels_r.append(float(np.abs(fd - a_rule).max() / scale))
            rels_f.append(float(np.abs(fd - a_frozen).max() / scale))
        k = int(np.argmin(rels_r))
        pg_rows.append({"gene": wg.GENE_NAMES[gi],
                        "max_abs_dcoord_dgene_mm": float(np.abs(a_rule).max()),
                        "steps": [float(s) for s in jac_steps],
                        "rel_ladder": rels_r, "rel_rule": rels_r[k],
                        "rel_frozen": rels_f[k], "best_step": float(jac_steps[k]),
                        "cliff_term_rel": float(np.abs(a_rule - a_frozen).max()
                                                / max(float(np.abs(a_frozen).max()),
                                                      1e-300))})
    # AND THE TRACE IS SHARED ACROSS GENOMES AGAIN, which the resolved pair in `coord_fn`'s
    # cache key made impossible: two per-genome meshes have different pairs and would have
    # keyed apart, re-tracing the 2.8 s jaxpr on every finite difference and every optimizer
    # step.  The record holds `None` now and the entry is resolved inside the trace, so the
    # key is genome-independent -- and a SHIPPED-pair mesh must still key apart from it.
    # A FRESH MESH, because `mp` already carries a `_coord_fn` from the rows above and
    # would short-circuit the cache entirely -- the count would read 0 and say nothing.
    WW._COORD_FN_CACHE.clear()
    m_first = WW.build_wheel(genes, cfg, fillet=True)
    np.asarray(WW.coord_fn(m_first)(genes))
    n_after_first = len(WW._COORD_FN_CACHE)
    other = genes.copy()
    other[12] += 0.2
    other[3] += 0.05
    m_other = WW.build_wheel(other, cfg, fillet=True)
    got = np.asarray(WW.coord_fn(m_other)(other))
    n_after_second = len(WW._COORD_FN_CACHE)
    np.asarray(WW.coord_fn(WW.build_wheel(genes, cfg, fillet=True,
                                          layer_profile=G11_LAYER_PROFILE))(genes))
    out["per_genome"] = {
        "config": cfg, "pair": list(pair),
        "cliff_entry": float(cliff["entry"]),
        "cliff_per_junction": {j: float(cliff["per_junction"][j]["cliff"])
                               for j in ("hub", "rim")},
        "binding_junction": max(("hub", "rim"),
                                key=lambda j: cliff["per_junction"][j]["cliff"]),
        "factor": float(WW.FILLET_LAYER_CLIFF_FACTOR),
        "identity_max_abs_mm": float(np.abs(
            np.asarray(WW.mesh_coords(jnp.asarray(genes), mp))
            - np.asarray(mp.coords)).max()),
        "numpy_path_max_abs_mm": float(np.abs(
            np.asarray(WW.mesh_coords(genes, mp, xp=np))
            - np.asarray(mp.coords)).max()),
        "rows": pg_rows,
        "worst_rel_rule": max(r["rel_rule"] for r in pg_rows),
        "worst_rel_frozen": max(r["rel_frozen"] for r in pg_rows),
        "trace_shared": {
            "after_one_genome": n_after_first,
            "after_two_genomes": n_after_second,
            "second_genome_pair": [float(v) for v in
                                   m_other.fillet_recipe["layer_profile"]],
            "second_genome_identity_mm": float(
                np.abs(got - np.asarray(m_other.coords)).max()),
            "after_a_shipped_pair_mesh": len(WW._COORD_FN_CACHE)},
        "gate_rel": GATE_FILLET_JAC_REL, "gate_mm": GATE_FILLET_MESH_MM}
    out["per_genome"]["ok"] = bool(
        out["per_genome"]["worst_rel_rule"] < GATE_FILLET_JAC_REL
        and out["per_genome"]["identity_max_abs_mm"] < GATE_FILLET_MESH_MM
        and out["per_genome"]["numpy_path_max_abs_mm"] == 0.0
        and n_after_first == 1 and n_after_second == 1
        and out["per_genome"]["trace_shared"]["after_a_shipped_pair_mesh"] == 2)

    out["worst_identity_mm"] = max(r["max_abs_mm"] for r in out["identity"])
    out["worst_numpy_identity_mm"] = max(r["numpy_path_max_abs_mm"]
                                         for r in out["identity"])
    out["pass"] = bool(out["worst_identity_mm"] < GATE_FILLET_MESH_MM
                       and out["census"]["ok"]
                       and out["jacobian"]["worst_rel"] < GATE_FILLET_JAC_REL
                       and out["axle_drop"]["worst_rel"] < GATE_SECANT_REL
                       and refusals["ok"]
                       and out["per_genome"]["ok"])
    return out


def run_cost(genes, cfg=DEFAULT_CONFIG, indentation_mm=None, repeats=3,
             kinematics=DEFAULT_KINEMATICS):
    """What the gradient costs, and against WHICH forward solve — the ratio is not one
    number and reporting it as one would mis-size Stage 3.

    Both denominators are measured because they answer different questions.  Against a
    COLD solve the gradient is a rounding error, which is the master plan's framing
    ("~10-15% on top of the forward solve").  But Stage 3 warm-starts every design from
    the last one, and a warm contact solve is only two Newton iterations — while the
    gradient still needs a full tangent assembly and factorisation no matter how good the
    starting guess was.  So the ratio that actually sizes the phase batch is the warm
    one, and it is much the larger of the two.

    Everything here runs after the jit trace, which the first call pays for and no later
    one does.  `wheel_wheel.coord_fn`'s docstring records what that trace costs and why
    it is cached rather than re-taken.
    """
    genes = np.asarray(genes, dtype=float)
    mesh = WW.build_wheel(genes, cfg)
    if indentation_mm is None:
        indentation_mm = _service_indentation(genes, cfg, kinematics)
    warm0 = WA.solve_and_grad(genes, cfg, "contact_force",
                              indentation_mm=indentation_mm, mesh=mesh,
                              kinematics=kinematics)                      # trace
    u0 = warm0["res"]["u_reduced"]

    cold, warm = [], []
    for _ in range(int(repeats)):
        cold.append(WA.solve_and_grad(genes, cfg, "contact_force",
                                      indentation_mm=indentation_mm, mesh=mesh,
                                      kinematics=kinematics)["timings"])
        warm.append(WA.solve_and_grad(genes, cfg, "contact_force",
                                      indentation_mm=indentation_mm, mesh=mesh,
                                      u_reduced0=u0,
                                      kinematics=kinematics)["timings"])

    def med(rows, key):
        return float(np.median([r[key] for r in rows]))

    return {"config": cfg, "kinematics": kinematics, "cold": cold, "warm": warm,
            "cold_forward_s": med(cold, "forward_s"),
            "warm_forward_s": med(warm, "forward_s"),
            "gradient_s": med(warm, "gradient_s"),
            "gradient_over_cold_forward": med(cold, "gradient_over_forward"),
            "gradient_over_warm_forward": med(warm, "gradient_over_forward")}


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def _print(rep):
    def head(s):
        print(f"\n{s}\n" + "-" * len(s))

    print("=" * 78)
    print("  M7 GATE — GRADIENTS BY IMPLICIT DIFFERENTIATION")
    print("=" * 78)
    # Printed at the top, not buried in the settings block: a report read without knowing
    # which kinematics produced it is a report that can be quoted against the wrong wheel,
    # which is the misattribution SVK_PLAN.md's standing rules exist to prevent.
    kin = rep.get("settings", {}).get("kinematics", rep["unrolled"].get("kinematics"))
    if kin is not None:
        print(f"  kinematics: {str(kin).upper()}")

    u = rep["unrolled"]
    head("G1  *** THE ONE THAT MATTERS *** ADJOINT VS DIFFERENTIATING THE SOLVE")
    print(f"    {u['config']} config, order {u['order']}, "
          f"{u['n_reduced_dof']} reduced DOF, indentation {u['indentation_mm']:.4f} mm")
    print(f"    {'gene':>6s} {'adjoint':>16s} {'unrolled (cold)':>17s} {'rel':>10s}")
    for i, name in enumerate(wg.GENE_NAMES):
        a, c = u["adjoint_grad"][i], u["cold"]["grad"][i]
        r = abs(c - a) / max(abs(a), abs(c)) if max(abs(a), abs(c)) > 0 else 0.0
        print(f"    {name:>6s} {a:+16.9e} {c:+17.9e} {r:10.2e}")
    print(f"    cold start, {u['cold']['iterations']} unrolled Newton iterations, "
          f"no implicit-function theorem anywhere")
    print(f"    worst relative difference {u['cold']['worst_rel']:.3e}  "
          f"[< {GATE_UNROLLED_REL:.0e}]")
    print(f"    warm ladder (exact by construction from the converged state — a "
          f"consistency check, not evidence):")
    for k, v in u["warm"].items():
        print(f"        {k} step(s): worst {v['worst_rel']:.3e}")
    print(f"    -> {'PASS' if u['pass'] else 'FAIL'}")

    d = rep["identities"]
    head("G2/G3  THE TWO IDENTITIES UNDERNEATH EVERYTHING ELSE")
    for r in d["mesh_coords"]:
        print(f"    mesh_coords vs build_wheel  {r['config']:>7s} phase "
              f"{r['phase_deg']:4.1f}   {r['max_abs_mm']:.3e} mm")
    for r in d["residual"]:
        print(f"    grad_u Pi vs assembled f    {r['config']:>7s}          "
              f"{r['max_abs_rel']:.3e} rel")
    for r in d["contact_force"]:
        print(f"    dPi/dy vs pressure integral {r['config']:>7s}   "
              f"{r['from_dPi_dy']:.6f} vs {r['from_quadrature']:.6f} N   "
              f"{r['rel']:.2e}")
    print(f"    [mesh < {GATE_MESH_COORDS_MM:.0e} mm; residual < "
          f"{GATE_RESIDUAL_REL:.0e}]")
    print(f"    -> {'PASS' if d['pass'] else 'FAIL'}")

    p = rep["plateau"]
    head("G4/G8  THE FD PLATEAU, GENE BY GENE")
    ladder_steps = next((r["steps"] for r in p["rows"].values() if r["steps"]), [])
    print(f"    {'gene':>6s} {'adjoint':>15s} "
          + " ".join(f"{x:>10.0e}" for x in ladder_steps))
    for name, r in p["rows"].items():
        if r["insensitive"]:
            print(f"    {name:>6s} {r['adjoint']:+15.8e}   INSENSITIVE — "
                  f"|dcoords/dgene| = {r['coord_sensitivity']:.1e}, identically zero")
            continue
        ladder = " ".join(f"{x:10.4g}" for x in r["derivatives"])
        print(f"    {name:>6s} {r['adjoint']:+15.8e} {ladder}")
        print(f"    {'':>6s} {'':>15s} plateau {r['plateau_decades']} decades within "
              f"{GATE_FD_PLATEAU_REL:.0e}; best agreement with the adjoint "
              f"{r['best_rel_to_adjoint']:.2e}")
    print(f"    steps are fractions of each GENE'S OWN range, not absolute mm")
    print(f"    narrowest plateau {p['worst_plateau_decades']} decades  "
          f"[>= {GATE_PLATEAU_DECADES}]")
    print(f"    worst adjoint-vs-FD {p['worst_rel_to_adjoint']:.2e}  "
          f"[< {GATE_FD_PLATEAU_REL:.0e}]")
    print()
    print(f"    *** {len(p['insensitive_genes'])} GENE(S) HAVE NO GRADIENT AT ALL: "
          f"{', '.join(p['insensitive_genes'])}")
    print(f"        The mesh models no fillets, so the fillet radii do not enter")
    print(f"        `mesh_coords` and dcoords/dgene is EXACTLY zero — the zero is")
    print(f"        structural and visible before any solver runs, which is sharper")
    print(f"        than M6's version of this finding.  Stage 3 must keep a")
    print(f"        non-gradient term for these two or it will silently optimise 12.")
    print(f"        census matches the expected set: "
          f"{'YES' if p['census_ok'] else 'NO — THIS IS A FAILURE'}")
    print(f"    -> {'PASS' if p['pass'] else 'FAIL'}")

    v = rep["directional"]
    head("G5  RANDOM DIRECTIONS — WHAT PER-GENE DIFFERENCES CANCEL")
    print(f"    {'#':>2s} {'adjoint':>16s} {'fd':>16s} {'best h':>8s} "
          f"{'rel':>9s}   ladder over {', '.join(f'{s:.0e}' for s in v['steps'])}")
    for i, r in enumerate(v["rows"]):
        print(f"    {i:2d} {r['adjoint']:+16.9e} {r['fd']:+16.9e} "
              f"{r['best_step']:8.0e} {r['rel']:9.2e}   "
              + "  ".join(f"{x:.1e}" for x in r["rel_ladder"]))
    print(f"    worst {v['worst_rel']:.2e}  [< {GATE_DIRECTIONAL_REL:.0e}]")
    print(f"    the ladder is the point: at a single h=1e-4 every direction reads "
          f"2-8e-5, which is")
    print(f"    the DIFFERENCE'S truncation error and not the gradient's — it falls "
          f"with h and the")
    print(f"    gradient does not move")
    print(f"    -> {'PASS' if v['pass'] else 'FAIL'}")

    s = rep["sweep"]
    head("G6  THE DENSE SWEEP — ADJOINT VS FD AT EVERY POINT")
    print(f"    {s['n']} points across {s['span_rel']*100:.1f}% of {s['gene']}'s range, "
          f"{s['config']} mesh")
    print(f"    {'h/range':>9s} {'median rel':>11s} {'worst rel':>11s} "
          f"{'outliers':>9s} {'clusters':>9s} {'longest':>8s}")
    for L in s["ladders"]:
        print(f"    {L['h_rel']:9.0e} {L['median_rel']:11.2e} {L['worst_rel']:11.2e} "
              f"{L['n_outliers']:9d} {L['n_clusters']:9d} {L['longest_run']:8d}")
    print(f"    outliers shrink as the FD step shrinks: "
          f"{'YES' if s['outliers_shrink_with_step'] else 'NO'}")
    print(f"    that is the test that separates a C^1 kink from a wrong gradient: a "
          f"kink corrupts")
    print(f"    a window of width h around itself, so its outlier count is "
          f"proportional to h; a")
    print(f"    wrong gradient does not care what h is.  The clusters are the windows.")
    print(f"    median at the finest step {s['median_rel']:.2e}  "
          f"[< {GATE_FD_PLATEAU_REL:.0e}]")
    print(f"    -> {'PASS' if s['pass'] else 'FAIL'}")

    f = rep["phase"]
    head("G7  *** THE M8 SPECIFICATION *** DOES THE CONTACT FACET AS IT ROLLS?")
    print(f"    ripple {100*f['peak_to_peak_over_mean']:.1f}% p2p/mean over one 30 deg "
          f"period")
    print(f"    slope measured over the steepest {f['window_deg']:.1f} deg "
          f"(from {f['window_start_deg']:.2f}) at {f['window_step_deg']:.3f} deg steps")
    print(f"    {'cfg':>7s} {'node deg':>9s} {'quad deg':>9s} {'patch half':>11s} "
          f"{'nodes/patch':>12s} {'slope':>18s} {'facet':>9s}")
    for r in f["rows"]:
        print(f"    {r['config']:>7s} {r['rim_node_spacing_deg']:9.4f} "
              f"{r['quadrature_spacing_deg']:9.4f} {r['patch_half_deg']:11.4f} "
              f"{r['nodes_in_patch']:12.2f} "
              f"[{r['slope_min']:+7.3f},{r['slope_max']:+7.3f}] "
              f"{100*r['facet_fraction']:8.1f}%")
    print(f"    facet = std of the detrended slope, as a fraction of the mean slope")
    verdict = ("YES" if f["facet_falls_with_refinement"]
               else "NO — it is not an artefact and M8 cannot refine it away")
    ratio = f"  ({f['facet_ratio']:.1f}x)" if f["facet_ratio"] else ""
    print(f"    falls with refinement: {verdict}{ratio}")
    print(f"    -> {'PASS' if f['pass'] else 'FAIL'}")
    print()
    print(f"    *** THE MASTER PLAN'S RISK #7 IS REAL, AND ITS OWN MITIGATION IS NOT MET")
    print(f"        \"set rim N_theta for >= 8 nodes in the patch\" was written when the")
    print(f"        patch was ASSUMED to be 3 deg wide.  M6 measured 0.484, so no")
    print(f"        config in this project reaches even 2 nodes across it, and the")
    print(f"        slope carries a ripple at the quadrature spacing as a result.")
    print(f"        It converges, so it is an artefact rather than a property of the")
    print(f"        wheel — but M8's phase quadrature cannot assume the spectral")
    print(f"        accuracy the master plan claims for a trapezoid rule until the rim")
    print(f"        resolves the patch.")

    a = rep["axle_drop"]
    head("G9  THE QUANTITY STAGE 3 ACTUALLY OPTIMISES")
    print(f"    axle drop {a['axle_drop_mm']:.6f} mm at {a['contact_force_n']:.4f} N; "
          f"dF/d(delta) {a['d_force_d_indentation']:.4f} N/mm")
    for r in a["rows"]:
        print(f"    {r['gene']:>6s}  adjoint {r['adjoint']:+.9e}  fd "
              f"{r['fd']:+.9e}  rel {r['rel']:.2e}  at h {r['best_step']:.0e}   "
              + "  ".join(f"{x:.1e}" for x in r["rel_ladder"]))
    print(f"    worst {a['worst_rel']:.2e}  [< {GATE_SECANT_REL:.0e}]")
    print(f"    the reference secant runs at fd_tol_rel {a.get('fd_tol_rel', 1e-9):.0e}, "
          f"and THAT TOLERANCE IS WHAT LIMITS")
    print(f"    THIS — the opposite of what this line said until 2026-08-19.  On the "
          f"uncapped default,")
    print(f"    cx4 at smoke reads 1.07e-04 at tol 1e-8 and 1.99e-06 at 1e-9: one decade "
          f"on the")
    print(f"    REFERENCE moves G9 from 10x over this gate to 5x under it.  (1e-11 is not "
          f"an option;")
    print(f"    the secant stalls at the float64 floor.)  See `run_axle_drop`'s docstring "
          f"for the table.")
    print(f"    -> {'PASS' if a['pass'] else 'FAIL'}")

    c = rep["cost"]
    head("G10  WHAT THE GRADIENT COSTS, AND AGAINST WHICH SOLVE")
    print(f"    {c['config']} mesh, after the jit trace")
    print(f"    gradient                       {c['gradient_s']:.3f} s")
    print(f"    forward, cold start            {c['cold_forward_s']:.3f} s   "
          f"-> gradient/forward {c['gradient_over_cold_forward']:.3f}")
    print(f"    forward, warm start            {c['warm_forward_s']:.3f} s   "
          f"-> gradient/forward {c['gradient_over_warm_forward']:.2f}")
    print(f"    the master plan predicted 0.10-0.15, which is the COLD ratio and is met.")
    print(f"    the ratio that sizes Stage 3 is the WARM one: the optimizer warm-starts")
    print(f"    every design from the last, so its forward solve is two Newton")
    print(f"    iterations, while the gradient still needs a full tangent assembly and")
    print(f"    factorisation however good the starting guess was.")

    q = rep.get("filleted")
    if q is not None:
        head("G11  *** G8's CENSUS, INVERTED *** THE FILLETED MESH HAS A DERIVATIVE")
        for r in q["identity"]:
            print(f"    mesh_coords vs build_wheel(fillet=True)  {r['config']:>7s} "
                  f"phase {r['phase_deg']:4.1f}  {r['n_nodes']:6d} nodes  "
                  f"{r['max_abs_mm']:.3e} mm   (numpy path "
                  f"{r['numpy_path_max_abs_mm']:.3e})")
        print(f"    [< {GATE_FILLET_MESH_MM:.0e} mm, the same identity G3 makes of the "
              f"unfilleted mesh]")
        c = q["census"]
        print()
        print(f"    census, unfilleted mesh:  dead = "
              f"{', '.join(c['unfilleted_dead']) or '(none)'}")
        print(f"    census, FILLETED mesh:    dead = "
              f"{', '.join(c['filleted_dead']) or '(NONE — all 14 genes move nodes)'}")
        print(f"    {'gene':>6s} {'|dcoords/dgene| unfilleted':>27s} "
              f"{'filleted':>14s}")
        for name in wg.GENE_NAMES:
            print(f"    {name:>6s} {c['coord_sensitivity_unfilleted'][name]:27.6e} "
                  f"{c['coord_sensitivity_filleted'][name]:14.6e}")
        print(f"    the two that were dead now rank FIRST and SECOND of fourteen: "
              f"{'YES' if c['fillet_genes_rank_first'] else 'NO'}")
        j = q["jacobian"]
        print()
        print(f"    the mesh jacobian against a central difference of "
              f"`build_wheel(fillet=True)` itself")
        print(f"    ({j['config']} mesh; the reference re-brackets and re-bisects at "
              f"every perturbed genome)")
        print(f"    {'gene':>6s} {'max |dc/dg| mm/mm':>18s} {'rel':>10s} "
              f"{'at h':>10s}   ladder")
        for r in j["rows"]:
            print(f"    {r['gene']:>6s} {r['max_abs_dcoord_dgene_mm']:18.6e} "
                  f"{r['rel']:10.2e} {r['best_step']:10.0e}   "
                  + "  ".join(f"{x:.1e}" for x in r["rel_ladder"]))
        print(f"    worst {j['worst_rel']:.2e}  [< {GATE_FILLET_JAC_REL:.0e}]")
        a = q["axle_drop"]
        print()
        print(f"    and the end of the chain: the axle drop at service force")
        print(f"    filleted {a['filleted_mm']:.6f} mm against unfilleted "
              f"{a['unfilleted_mm']:.6f} mm")
        for r in a["rows"]:
            print(f"    {r['gene']:>6s}  adjoint {r['adjoint']:+.9e}  fd "
                  f"{r['fd']:+.9e}  rel {r['rel']:.2e}  at h {r['best_step']:.0e}   "
                  + "  ".join(f"{x:.1e}" for x in r["rel_ladder"]))
            print(f"    {'':>6s}  the same gene on the UNFILLETED mesh: "
                  f"{r['unfilleted_adjoint']:+.9e}")
        print(f"    worst {a['worst_rel']:.2e}  [< {GATE_SECANT_REL:.0e}, G9's gate and "
              f"G9's tightened reference secant at {a['fd_tol_rel']:.0e}]")
        print()
        print(f"    STILL REFUSED, because the derivative would be WRONG and not because")
        print(f"    it would be hard — both raise:")
        for k in ("spoke_blocking", "sector_fit_clamped"):
            msg = q["refusals"][k]
            print(f"      {k:24s} {'refused' if msg else '*** DID NOT REFUSE ***'}")
        print(f"      the clamped mesh was built at radii "
              f"{q['refusals']['clamped_radii_mm']} against a requested 8.0 mm at the "
              f"hub")
        print()
        pg = q["per_genome"]
        print(f"    AND THE PER-GENOME LAYER PROFILE, WHICH WAS THE THIRD REFUSAL FROM "
              f"§85 TO §88:")
        print(f"      cliff {pg['cliff_entry']:+.9f} (hub "
              f"{pg['cliff_per_junction']['hub']:+.6f}, rim "
              f"{pg['cliff_per_junction']['rim']:+.6f} — the {pg['binding_junction']} "
              f"binds), x {pg['factor']} -> entry {pg['pair'][0]:+.9f}")
        print(f"      identity {pg['identity_max_abs_mm']:.3e} mm traced, "
              f"{pg['numpy_path_max_abs_mm']:.3e} mm on the numpy path "
              f"[< {pg['gate_mm']:.0e} mm]")
        print(f"      {'gene':>6s}  {'rel vs the RULE':>16s}  "
              f"{'rel if the pair were FROZEN':>28s}   cliff term")
        for r in pg["rows"]:
            print(f"      {r['gene']:>6s}  {r['rel_rule']:>16.2e}  "
                  f"{r['rel_frozen']:>28.2e}   {r['cliff_term_rel']:.2e}")
        print(f"      worst {pg['worst_rel_rule']:.2e} [< {pg['gate_rel']:.0e}] against "
              f"{pg['worst_rel_frozen']:.2e} for the gradient §85 refused to return")
        ts = pg["trace_shared"]
        print(f"      one traced jaxpr across genomes: {ts['after_one_genome']} entry "
              f"after one genome, {ts['after_two_genomes']} after a second at pair "
              f"{tuple(round(v, 6) for v in ts['second_genome_pair'])} "
              f"(identity {ts['second_genome_identity_mm']:.3e} mm), "
              f"{ts['after_a_shipped_pair_mesh']} once a SHIPPED-pair mesh is added")
        print(f"    -> {'PASS' if q['pass'] else 'FAIL'}")
        print()
        print(f"    *** WHAT THIS DOES AND DOES NOT CHANGE.  Nothing wires the fillet "
              f"into the")
        print(f"        objective: `wheel_objective` still prices `R_hub` through a "
              f"`Kt` surrogate")
        print(f"        that section 75 measured EXACTLY FLAT above the cap, and section "
              f"48's")
        print(f"        surviving clause — half of a drawn genome box sits under "
              f"`MIN_SJ_TARGET` —")
        print(f"        still stands against letting the optimizer take this path.  G8 "
              f"above is")
        print(f"        still the census of the mesh Stage 3 actually builds.")

    head("VERDICT")
    print(f"    the adjoint reproduces brute-force differentiation of the same solve "
          f"to {u['cold']['worst_rel']:.1e}")
    print(f"    every live gene has a finite-difference plateau; "
          f"{len(p['insensitive_genes'])} genes have no gradient to have one, and the "
          f"zero is in the MESH")
    print(f"    the objective IS faceted in phase at the contact quadrature spacing "
          f"({100*f['rows'][0]['facet_fraction']:.0f}% of the slope at "
          f"{f['rows'][0]['config']}), and it refines away — an M8 mesh requirement, "
          f"not a gradient defect")
    if q is not None:
        print(f"    the two genes with no gradient have one on a FILLETED mesh, where "
              f"they rank first")
        print(f"    and second of fourteen — the mesh Stage 3 builds is still the "
              f"unfilleted one, and")
        print(f"    is still blind to them")
    print(f"\n  OVERALL: {'PASS' if rep['pass'] else 'FAIL'}")
    print(f"\n  NOT DONE: the loss terms.  M7 differentiates SOLVE OUTPUTS; the seven")
    print(f"            objective terms and the p-norm stress are M8's, and")
    print(f"            calibrating their weights against a gradient nothing had")
    print(f"            verified is the order this gate exists to prevent.")


def main():
    ap = argparse.ArgumentParser(description="M7 gradient gate")
    ap.add_argument("--genome", default="best_solution.json")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default="study_gradient.json")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="reduced meshes and sample counts; for the test suite")
    # SVK_PLAN.md step 1.  Every gate in this file has only ever been measured under the
    # linear default, so the adjoint Stage 3 would descend on under SVK is unproven.  The
    # DEFAULT MUST STAY `linear`: `study_gradient.json` is a committed artifact and the
    # `--kinematics svk` run writes its own `--out`.
    ap.add_argument("--kinematics", choices=("linear", "svk"),
                    default=DEFAULT_KINEMATICS,
                    help="kinematics for every solve in the report; linear is the "
                         "committed default and svk is SVK_PLAN.md step 1")
    args = ap.parse_args()

    # A degraded run may not be filed under the committed artifact's name (PLAN.md
    # §43).  Refused at startup, before any solving.  See `_gate_guard`.
    _gate_guard.refuse_degraded_out(ap, args, "study_gradient.json", [
        (args.quick, "--quick (reduced fidelity)"),
        (args.config != DEFAULT_CONFIG, "--config %s, not the gate's %s" % (args.config, DEFAULT_CONFIG)),
        (args.genome != "best_solution.json", "--genome %s" % args.genome),
        (args.kinematics != "linear", "--kinematics %s" % args.kinematics),
        (args.no_plot, "--no-plot, which would refresh the .json and leave the "
                       "committed .jpg stale"),
    ])

    genes = load_genes(args.genome)
    cfg = "smoke" if args.quick else args.config
    kin = args.kinematics
    t0 = time.time()

    rep = {}
    # G1 first, and it stays a real check in --quick: it is the only one with no finite
    # difference in it, so weakening it would weaken the number the rest of the report is
    # read against.  Quick mode shortens the unrolled ladder (the cost is one
    # differentiated Newton iteration each) and keeps the tolerance.
    rep["unrolled"] = run_unrolled(
        genes, cold_iters=8 if args.quick else 14,
        warm_iters=(1,) if args.quick else (1, 2, 3), kinematics=kin)
    rep["identities"] = run_identities(
        genes, cfg, configs=("smoke",) if args.quick else ("smoke", "coarse"),
        kinematics=kin)
    delta = _service_indentation(genes, cfg, kin)
    rep["plateau"] = run_plateau(
        genes, cfg, indentation_mm=delta,
        gene_ids=QUICK_GENES if args.quick else None,
        steps=(1e-2, 1e-3, 1e-4, 1e-5) if args.quick
        else (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7), kinematics=kin)
    rep["directional"] = run_directional(
        genes, cfg, indentation_mm=delta, n=3 if args.quick else 10,
        steps=(1e-4, 1e-6) if args.quick else (1e-4, 1e-5, 1e-6), kinematics=kin)
    rep["sweep"] = run_dense_sweep(genes, "smoke", n=60 if args.quick else 400,
                                   kinematics=kin)
    # Two configs always: the whole content of G7 is whether the faceting CONVERGES, and
    # one mesh cannot answer that.  Quick mode shortens the window instead.
    rep["phase"] = run_phase_smoothness(
        genes, configs=("smoke", "coarse"),
        n_period=60 if args.quick else 200,
        window_deg=1.0 if args.quick else 3.0, kinematics=kin)
    rep["axle_drop"] = run_axle_drop(
        genes, cfg, gene_ids=(6,) if args.quick else (6, 8, 12),
        steps=(1e-4, 1e-6) if args.quick else (1e-4, 1e-5, 1e-6), kinematics=kin)
    rep["cost"] = run_cost(genes, cfg, indentation_mm=delta,
                           repeats=1 if args.quick else 3, kinematics=kin)
    # G11 last: it is the only section that builds a mesh the shipped path never builds,
    # so a failure here must not be able to shorten the report the other ten produce.
    rep["filleted"] = run_filleted(
        genes, cfg, configs=("smoke",) if args.quick else ("smoke", "coarse"),
        gene_ids=(8, 12) if args.quick else (8, 12, 13),
        jac_steps=(1e-4, 1e-6) if args.quick else (1e-4, 1e-5, 1e-6),
        drop_gene_ids=(12,) if args.quick else (12, 13),
        drop_steps=(1e-4, 1e-6) if args.quick else (1e-4, 1e-5, 1e-6),
        kinematics=kin)

    rep["pass"] = bool(rep["unrolled"]["pass"] and rep["identities"]["pass"]
                       and rep["plateau"]["pass"] and rep["directional"]["pass"]
                       and rep["sweep"]["pass"] and rep["phase"]["pass"]
                       and rep["axle_drop"]["pass"] and rep["filleted"]["pass"])
    rep["settings"] = {"config": cfg, "genome": args.genome, "quick": args.quick,
                       "kinematics": kin,
                       "service_force_n": float(SERVICE_FORCE_N),
                       "indentation_mm": float(delta),
                       "elapsed_s": round(time.time() - t0, 1)}

    _print(rep)
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"\nwrote {os.path.join(HERE, args.out)}  "
          f"({rep['settings']['elapsed_s']} s)")
    if not args.no_plot:
        try:
            print(f"wrote {_plot(rep, os.path.splitext(os.path.join(HERE, args.out))[0] + '.jpg')}")
        except Exception as exc:                            # pragma: no cover
            print(f"(plot skipped: {exc})")
    return 0 if rep["pass"] else 1


def _plot(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

    p = rep["plateau"]
    for name, r in p["rows"].items():
        if r["insensitive"] or not r["steps"]:
            continue
        ax[0].loglog(r["steps"], np.maximum(r["rel_to_adjoint"], 1e-16), "o-",
                     lw=1, ms=3, label=name)
    ax[0].axhline(GATE_FD_PLATEAU_REL, color="k", ls=":", lw=0.9)
    ax[0].set_xlabel("step / gene range")
    ax[0].set_ylabel("|FD / adjoint - 1|")
    ax[0].set_title("every live gene has a plateau")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=6, ncol=2)

    s = rep["sweep"]
    for L in s["ladders"]:
        ax[1].semilogy(s["x"], np.maximum(L["rel"], 1e-16), ".", ms=2,
                       label=f"h/range = {L['h_rel']:.0e}")
    ax[1].axhline(GATE_SWEEP_REL, color="k", ls=":", lw=0.9)
    ax[1].set_xlabel(f"{s['gene']} (mm)")
    ax[1].set_ylabel("|FD / adjoint - 1|")
    ax[1].set_title(f"{s['n']} points: kink windows shrink with h")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[1].locator_params(axis="x", nbins=5)
    ax[1].ticklabel_format(axis="x", useOffset=False)

    f = rep["phase"]
    for r in f["rows"]:
        ph = np.array(r["phase_deg"])
        v = np.array(r["force_n"])
        ax[2].plot(ph, np.gradient(v, ph), "-", lw=1.0,
                   label=f"{r['config']}  facet {100*r['facet_fraction']:.0f}%")
    ax[2].set_xlabel("phase (deg)")
    ax[2].set_ylabel("d(contact force)/d(phase)  (N/deg)")
    ax[2].set_title("the facet, and that it refines away")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, path)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
