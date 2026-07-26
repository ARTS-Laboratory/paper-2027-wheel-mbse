# M7 — Gradients: the adjoint, and whether the objective is smooth in all 14 genes

> **STATUS: DONE.** All ten gates pass at `coarse` in 1079 s; `study_gradient.json` /
> `.jpg` are committed. Test suite 201 -> 213.
>
> **The headline.** The adjoint reproduces brute-force differentiation of the same solve
> to **9.3e-12**, three orders inside the 1e-8 gate, with no finite difference involved.
> `grad_u Pi` equals the assembled internal + contact force to **exactly zero**. Every
> live gene has an FD plateau agreeing with the adjoint to better than **2.9e-6**. The
> gradient costs **0.48 s** at `coarse`.
>
> **What the plan got right:** the energy formulation collapsed the adjoint to one
> `jax.grad` of a scalar; `wheel_fem.py` needed no changes at all; the two fillet genes
> are dead and the census caught it.
>
> **What the plan got wrong, and the measurements that said so:**
> 1. *`R_hub` and `R_rim` are dead at the MESH, not through the solve.* `dcoords/dgene` is
>    identically zero — no solver, contact law or step size is involved. Sharper than
>    M6's version of the same finding, and it is what the gate now asserts.
> 2. *Three gates were measuring the finite difference, not the gradient.* G4, G5 and G9
>    were each written at a single step of 1e-4 of the gene range and each failed at
>    `coarse` by one to eight parts in 1e5. One cause: the REFERENCE's truncation error.
>    Over a ladder the directional disagreement falls 1.2e-2, 8.3e-5, 2.1e-5, 6.2e-7
>    while the adjoint does not move. Tightening the secant from 1e-8 to 1e-11 changed
>    G9 by *nothing at all* — ten identical digits — which is what ruled the secant out.
> 3. *G6's "no adjacent outliers" was wrong about how a kink looks.* A central difference
>    of step `h` straddles a C¹ kink for every sweep point within `h`, so a sweep finer
>    than `h` must produce a run. Measured: 23 outliers in 6 clusters at h=1e-4, **2 at
>    h=1e-5**, median falling 4.4e-5 -> 3.2e-7. The criterion is now the shrinkage.
> 4. *The gradient does not cost "10-15% on top of the forward solve" in the way that
>    matters.* Against a cold solve it is 4.3%, meeting the prediction. Against the WARM
>    solve Stage 3 will actually run it is **2.4x**, because a warm contact solve is two
>    Newton iterations while the gradient still needs a full tangent assembly and
>    factorisation. That is the number that sizes M8's phase batch.
>
> **Unplanned finding, and the most consequential one for M8:** the contact **facets on
> the rim discretisation as the wheel rolls**. The slope of force(phase) carries a ripple
> at the quadrature-point spacing worth **17.6% of the slope at `smoke` and 3.8% at
> `coarse`** — it refines away (4.6x for a 2.5x rim refinement), so it is an artefact
> rather than a property of the wheel. What it exposes is that the master plan's own
> mitigation for its risk #7 — "set rim N_theta for >= 8 nodes in the patch" — was
> written when the patch was *assumed* to be 3° wide. M6 measured 0.484°, so `smoke` gets
> **0.56** nodes across the patch and `coarse` **1.75**. No config in this project
> reaches two. M8's phase quadrature cannot assume the spectral accuracy the master plan
> claims for it until the rim resolves the patch.
>
> **A performance fact that is load-bearing rather than tuning:** `jax.vjp` on an untraced
> closure re-traces `sector_blocks` on every call — 0.774 s against 0.05 s for the entire
> rest of the adjoint. And the obvious cache, keyed on the mesh object, misses on every
> call it exists to serve, because every finite difference and every optimizer step builds
> a new mesh. `wheel_wheel.coord_fn` keys on the static recipe instead.

## Context

M7 is the master plan's gradient milestone:

> **M7 — Gradients (~2d).** `custom_vjp` implicit-diff solve; the four gradient checks in
> order. *Gate:* a gene with no FD plateau means the objective isn't smooth in it — find
> the term and fix it before Stage 3.

Everything downstream of M7 is a projected optimizer that calls one function. A gradient
that is right to plotting accuracy and no better cannot be told apart from a hard problem
by any line search, so it would produce a Stage-3 run that looks like it is working and is
not. M7's deliverable is therefore not an adjoint — that is fifty lines — but a *proved*
one, plus the measurements of smoothness that say the objective is differentiable at all.

M6 already answered M7's advertised hazard in the unexpected direction: the sharp Macaulay
bracket has a clean three-decade FD plateau, so the C² smoothing is unnecessary and
`smoothing_mm` stays 0.0. What M6 surfaced instead — `R_hub` and `R_rim` having no FEA
gradient — is the finding that shapes this milestone.

**Four scope decisions, taken before writing any code:**

1. **Solve outputs only.** `contact_force` at fixed indentation, `axle_drop` at service
   force, `strain_energy`, `mass`. The seven loss terms and the p-norm stress are M8's;
   calibrating their weights against a gradient nothing had verified is the order this
   gate exists to prevent.
2. **numpy `value_and_grad` only, no `custom_vjp` wrapper.** Stage 3's optimizer is
   numpy-side and never traces the outer loop, so a wrapper would be a code path with no
   consumer. The pieces are separated so that adding one is mechanical.
3. **The two fillet-blind genes are accepted, classified and gated on**, not fixed.
   Meshing fillets is its own milestone (`replicated-roaming-quill.md`) and would re-open
   M2b's mesh-validity gate. Stage 3 keeps a non-gradient term for them.
4. **`CLAUDE.md` stays deleted.** M7 documents itself in `wheel_adjoint.py`'s docstring
   and in `study_gradient.py`, the way M5 and M6 currently do.

---

## Formulation

**Write the energy, derive everything else — one level up.** `wheel_fem.py` gets the
internal force and the tangent from `jax.grad` / `jax.hessian` of one element energy so
they cannot drift apart. The same argument applies with more force to the sensitivity: a
hand-derived `∂r/∂p` that disagrees with the residual by a term produces a gradient that is
*plausible* — right direction, wrong length — which is the hardest kind of bug to see from
an optimizer trace.

At a converged state the reduced residual is itself a derivative of the potential:

```
Pi(c, u_f, y) = U_bulk(c, u_f) + Pi_contact(c, u_f, y)
u_f           = T @ u_r + u_pre                       [T static: the topology is frozen]
r(c, u_r)     = T^T grad_{u_f} Pi = grad_{u_r} Pi     [ = 0 at the solution ]
K_r           = hess_{u_r} Pi                         [ what Newton assembled ]
```

so for a quantity of interest `Q(c, u_f, y)`:

```
K_r adj_r = -T^T (dQ/du_f)                one sparse solve; K_r is symmetric
adj_f     = T @ adj_r
dQ/dc     = dQ/dc|_u + grad_c [ adj_f . grad_{u_f} Pi(c, u_f, y) ]
dQ/dgenes = vjp_{mesh_coords}(dQ/dc)
```

The third line is the whole adjoint: one `jax.grad` of a scalar.

**Two things fall out rather than being coded.** The total contact force is `+dPi/dy_ground`
exactly, so `contact_force` is a derivative of the contact law rather than a second
quadrature of it — and its agreement with `RigidGroundContact.total_force` becomes evidence
instead of an assumption. And `dF/d(indentation)` is the same adjoint with the same
multiplier, contracted against `y_ground` instead of `c`.

**The axle drop at service force is an implicit function, not a differentiated secant.**
`F(delta*, p) = F_service` gives `d(delta)/dp = -(dF/dp)/(dF/d delta)`, both pieces from
one adjoint. Differentiating the secant would put its own stopping tolerance into the
derivative, where it would masquerade as physics — the argument M6's G7 already makes.

**What is not differentiable is named rather than hidden.** `mesh_coords` freezes the flank
orientation and the seam ownership. Those are discrete decisions and a step that flips one
is a genuine discontinuity of the design space, not a defect.

---

## Gates — written down before the run

| # | gate | threshold |
|---|---|---|
| 1 | adjoint vs **fully unrolled Newton**, `tiny` config, all 14 genes | 1e-8 relative |
| 2 | `grad_u Pi` vs the assembled `internal_force + contact.force` | 1e-12 relative |
| 3 | `mesh_coords` vs `build_wheel`'s own coordinates, every config | < 1e-9 mm |
| 4 | per-gene FD plateau **vs the adjoint**, all 14, `h ∈ {1e-2…1e-7} × gene range` | ≥ 1 decade within 1e-4 |
| 5 | 10 random directional derivatives, over a step ladder | 1e-5 relative |
| 6 | adjoint vs FD at 400 points across one gene, at two steps | outliers shrink with `h` |
| 7 | phase faceting must refine away | see below |
| 8 | insensitive-gene census | **exactly** `{R_hub, R_rim}` |
| 9 | `axle_drop` gradient through the secant | 1e-5 relative |
| 10 | gradient cost as a fraction of the forward solve | reported |

**Gate 1 is the one that matters**, and it is first for the master plan's stated reason: it
compares the adjoint against brute-force differentiation of the same solve and involves no
finite difference at all, so it isolates adjoint correctness from step-size noise. `tiny`
is order 1 with 336 reduced DOF — 0.9 MB per dense Newton iteration, so the whole nonlinear
solve unrolls and differentiates. Order 1 makes it a poor *wheel* model, which is
irrelevant to the question being asked and is stated in the report so nobody quotes it.

**Gates 4, 5 and 9 are ladders, and were not at first.** Each was written at a single step
of 1e-4 of the gene range and each failed at `coarse` by one to eight parts in 1e5 — all
three because of the *reference's* truncation error, not the gradient's. A single-step
check has no plateau, which is the master plan's own criticism of single-point agreement
applied to its own items. Gate 4 also measures the plateau against the **adjoint** rather
than against the ladder's previous rung: that is the plan's literal wording, it is the
stronger statement (a systematically biased difference also stops moving), and it is what
lets a gene like `cx1` — whose derivative is a thousand times smaller than `t2`'s, so the
roundoff floor set by the 66 N response bites it a thousand times sooner — be scored on
whether its FD is *right* rather than on whether it is *stationary*.

**Gate 6 replaces "must be visibly smooth" with a number.** A plot is not a gate. Comparing
the adjoint against a difference at all 400 points costs the same solves, is strictly
stronger, and *localises* a leak. The outliers arrive in short **runs**, and that is the
signature rather than a defect: a central difference of step `h` straddles a C¹ kink for
every sweep point within `h` of it, so a sweep sampled finer than `h` must produce a run
`2h` wide. The criterion is therefore the one that separates a kink from a wrong gradient
— shrink `h` and count again.

**Gate 7's first criterion was wrong and the data said so.** It asked for the worst second
difference of `force(phase)` against its own median. Measured, that ratio is 6 at 120
samples and 29 at 400 — it *grows* with sampling, because it is dominated by how much the
wheel's real curvature varies over a period. A statistic that worsens the harder you look
at a fixed physical curve is not measuring the discretisation. Replaced with the thing
itself: the detrended slope of `force(phase)` over a window sampled far finer than the
contact quadrature spacing, measured at two mesh refinements. An artefact must shrink with
the mesh; a property of the wheel must not.

---

## Files

- **`wheel_wheel.py`** — the traced half of `build_wheel` factored into `_sector_coords`;
  the static recipe (`owners`, `orientation`, `phase_deg`) carried on `WheelMesh`; new
  `mesh_coords` and `coord_fn`. `build_wheel`'s eager guards stay unconditional — they are
  the reason the mesh can be trusted, and a "skip when tracing" branch is how guards stop
  running. `coord_fn`'s jit is load-bearing, not tuning: without it `jax.vjp` re-traces
  `sector_blocks` on every call and 97% of a Stage-3 gradient is re-derived jaxpr.
- **`wheel_fem.py`** — **unchanged**. The adjoint reassembles `K_r` from the public
  `assemble_stiffness` + `contact.stiffness`; keeping this file untouched is worth more
  than the milliseconds, since M4's, M5's and M6's committed reports all run through it.
- **`wheel_adjoint.py`** — new: the potential as one jnp function, the QoI registry,
  `solve_and_grad`, `axle_drop_value_and_grad`, `value_and_grad`, `insensitive_genes`.
- **`study_gradient.py`** — the M7 gate.
- **`tests/test_gradient.py`** — the identities, the unrolled-Newton comparison, the two
  structural facts, and a small rerun of the gate's headline assertions.
- **`Makefile`** — `study_gradient.py` added to `studies`.

---

## Verification

```bash
.venv-opt/bin/python study_gradient.py --quick     # fast loop during development
.venv-opt/bin/python study_gradient.py             # the gate; exits nonzero on failure
make test
make studies                                       # all seven gates
```

**The regression that matters most**: M4's, M5's and M6's committed reports must not move.
M7 adds a derivative; it does not touch the mesh, the element, the contact law or the
solver. A `git diff` on `study_wheel_fea.json`, `study_beam_agreement.json`,
`study_gnl.json` and `study_contact.json` after `make studies` is the check. The one change
that could leak is the `wheel_wheel.py` refactor, which is why gate 3 compares
`mesh_coords` against `build_wheel`'s own output on every config.
