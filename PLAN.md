# M8b-i.5 — the stress constraint is measured on a singularity, and the verdict it produced is not a number

> **STATUS: BOTH CHECKS COMPLETE. `make m8bi5` ran to completion at `coarse` in 8273 s
> (2 h 18 m), both sections PASS.** `study_stage3_m8bi5.json` and `.jpg` are the record.
> `study_stage3.py` gained a `--sections` selector and two sections; `tests/test_stage3.py`
> gained 9 tests (19 -> 28, suite 249 -> 258, all green); `Makefile` gained `m8bi5`.
> **No physics module was touched** — `wheel_objective`, `wheel_stage3`, `wheel_adjoint`,
> `wheel_fem`, `wheel_wheel` are byte-identical. `make studies` is unchanged and still
> ~2 h 45 m; the two new sections are opt-in for exactly that reason.
>
> ## The headline: M8b-i answered the wrong question, and M4 had already said so
>
> S9 reported "each constraint is reachable alone, neither with the other", with the stress
> half quantified at **utilisation 1.7128**. **That number does not exist.** The quantity it
> is computed from does not converge under mesh refinement, and the ladder carries its own
> control proving the fault is in the QoI rather than in the mesh or the solve:
>
> | shipped genome, same meshes and same solves | smoke | coarse | medium | ratio | order | GCI | |
> |---|---|---|---|---|---|---|---|
> | **axle drop [mm]** — the deflection constraint | 1.4523 | 1.4914 | 1.4986 | 5.42 | 2.44 | **0.14%** | **CONVERGED** |
> | p-norm stress, p=30 [MPa] | 23.173 | 29.789 | 34.619 | 1.37 | 0.45 | 47.20% | not converged |
> | true max [MPa] | 31.020 | 41.539 | 48.465 | 1.52 | 0.60 | 34.43% | not converged |
> | c = mean_phase(max/pnorm) | 1.3611 | 1.4375 | 1.4822 | 1.71 | 0.77 | 5.33% | not converged (3.39% and converged at elite 1) |
> | **stress utilisation** | 1.2617 | **1.7128** | 2.0525 | 1.33 | 0.41 | **63.01%** | **not converged** |
>
> Two quantities out of one displacement field, on one mesh ladder, in one run. **The
> deflection converges at second order to 0.14%. The stress utilisation is at 63% and
> rising.** Nothing about the meshing, the contact solve, the phase stencil or the adjoint
> can explain a 450x gap in convergence between two QoIs sharing all of them.
>
> `CONVERGED` means GCI < `study_stage3.GATE_LADDER_GCI` = 5%, the GCI form of
> `study_wheel_fea.run_refinement`'s `finest_error_vs_richardson < 0.005` and deliberately
> 10x looser: the question is whether the constraint has a value at all, not whether the
> value is precise. **It is emphatically not `ratio > 1`.** Every stress series here has
> `ratio > 1` — the differences do shrink — and reading the verdict off that produced a
> figure captioned *"the stress QoI settles under refinement"* above a utilisation with a
> 63% GCI. `_series` now reports `settling` and `converged` as separate leaves and
> `tests/test_stage3.py` pins the distinction with these very numbers.
>
> The `coarse` rung reproduces S9's input exactly — 1.7128 against the recorded
> 1.7128182748822758 — so this is the same quantity the verdict rests on, not a neighbour.
> Elite 1 behaves identically and worse: utilisation 1.4238 -> 2.2403 -> 2.8220, order 0.49,
> GCI 63.84%, while its axle drop converges to GCI 0.08%.
>
> ### Why, in M4's own words
>
> `study_wheel_fea.stress_report`, `study_wheel_fea.py:101`, written at M4 and never acted on:
>
> > **THE POINTWISE MAX IS NOT A NUMBER** — it diverges under refinement (rim region 30.3,
> > 38.9, 44.9, 52.4 MPa across the config ladder) because this mesh has no fillets, and an
> > unfilleted spoke/ring junction is a 349.5 degree re-entrant corner: geometrically **a
> > crack**. Its stress field goes as r^-0.5 and no mesh resolves it.
> >
> > The p99 of the same field **converges cleanly (8.84, 8.78, 8.61, 8.61)**, which is what a
> > point singularity of measure zero looks like. **Quote the percentile; quote the max only
> > to say it is a singularity.**
>
> Stage 3's stress term does the opposite of that prescription, twice, and the two compound:
>
> 1. **`STRESS_PNORM_P = 30.0`** (`wheel_adjoint.py:244`). `_qoi_pnorm_stress`'s docstring
>    argues the volume-weighted p-norm is "a quadrature of an integral and is
>    mesh-convergent". That argument holds for a bounded field. At p=30 the norm is 1/1.38
>    of the max — it *is* the max in disguise — and it inherits r^-0.5. **Measured here at
>    order 0.45, against the axle drop's 2.44 on the same meshes.** This is the finding that
>    was not already on record, and it is the one that matters: the smooth surrogate does
>    not rescue the constraint.
> 2. **The rescale to the true max.** `t3_terms` multiplies that p-norm by
>    `c = mean_phase(max/pnorm)` and compares the product to the allowable
>    (`wheel_objective.py:501-510`), so the constraint is a comparison against the singular
>    quantity M4 named, by construction.
>
> **But note which of the two is doing the damage, because it is not the one the plan
> predicted.** `c` is the *best*-behaved factor in the whole table — GCI 5.33% at the
> shipped genome and 3.39% at elite 1, converged by the criterion above — because it is a
> ratio of two quantities that diverge together. The p-norm, the factor
> `_qoi_pnorm_stress`'s docstring calls mesh-convergent, is at 47%. **Removing the rescale
> alone would fix almost nothing; lowering `p` is the load-bearing change.**
>
> There is **no percentile QoI anywhere in `wheel_objective` or `wheel_adjoint`.** M4's
> prescription was never carried into the optimizer, and M8a's gates could not catch it
> because every one of them checks the stress term against *itself* — its own finite
> difference, its own p-norm, its own rescale — and all of those agree beautifully on any
> single mesh. **Nothing in the repo differentiated the stress QoI with respect to `cfg`
> until now.**
>
> ## What survives, what does not
>
> **Does not survive — every stress magnitude M8a and M8b-i quote.** "42.82 MPa against
> 25.0", "utilisation 1.7128", "min reachable utilisation 0.932", "M8a's headline understated
> it by 38%" and the whole of the previous PLAN.md's headline table. All are `coarse`
> readings of a crack tip. The `smoke`-to-`coarse` rise that the last milestone read as
> *"peak stress rises with refinement, as a resolved stress concentration does, so the verdict
> is a lower bound"* was the right observation and the wrong inference: an unresolvable
> singularity looks exactly like a resolving concentration from two points, which is why the
> third rung and the axle-drop control were worth 2 h 18 m.
>
> **Survives — the deflection half, entirely.** The axle drop converges (GCI 0.14%), and the
> 16-elite screen found designs that **meet the deflection target with no descent at all**:
> elite 9 at **+2.29%** and elite 10 at **+1.65%**, both inside the +-5% box. Fifteen of
> those sixteen had never been scored against the FEA objective.
>
> **Survives — that the two constraints trade against each other**, now over 16 independent
> Stage-2 optima and 4 fresh probes rather than 3 descents from 1 start. But the trade is
> quantified in the same broken stress units, so its *magnitude* is as unreliable as S9's.
>
> ## S12 — the screen, and the verdict restated over 16 starts
>
> Sixteen distinct starts, two of them re-descended. S9's own start is not a seventeenth:
> elite 0 IS `best_solution.json`.
>
> **(a) all 16 elites at `coarse`, no descent.** Spread: utilisation **1.0093-2.3500**,
> deflection error **-77.27% to +28.38%**. Emphatically not one basin, which is what makes
> the multi-start argument for M8b-ii real. Elite 0 is `best_solution.json` bit-for-bit
> (max|diff| = 0.0 over all fourteen genes) — hence the ladder's second design is elite 1.
>
> | elite | utilisation | deflection error | corner d | |
> |---|---|---|---|---|
> | 11 | **1.0093** | -77.27% | 14.464 | at the allowable, far too stiff |
> | 0 (= shipped) | 1.7128 | -25.43% | 4.799 | |
> | **9** | 1.7740 | **+2.29%** | 0.774 | **inside the deflection box** |
> | **10** | 1.8605 | **+1.65%** | 0.860 | **inside the deflection box** |
> | 7 | 1.9660 | +7.93% | 1.552 | |
> | 12, 13, 8, 6 | 2.04-2.11 | +13.8 to +18.2% | 2.8-3.7 | |
> | 3, 5, 15, 4, 1, 2, 14 | 2.18-2.35 | +19.1 to +28.4% | 4.0-6.0 | |
>
> **(b) the two nearest elites re-probed**, `stress_only` and `deflection_only`, 20 steps each,
> all four accepted 20/20 steps with zero events:
>
> | start | probe | utilisation | deflection error |
> |---|---|---|---|
> | elite 9 | stress_only | 1.774 -> 1.487 | +2.3% -> **-58.3%** |
> | elite 9 | deflection_only | 1.774 -> **1.770** | +2.3% -> +0.6% (min 0.04%) |
> | elite 10 | stress_only | 1.860 -> **1.005** (min 0.893) | +1.7% -> **-77.7%** |
> | elite 10 | deflection_only | 1.860 -> **1.878** | +1.7% -> +0.3% |
>
> **(c)** 100 (utilisation, deflection) pairs measured, from 16 scored starts and 2 probed.
> Lowest utilisation anywhere **0.8933** (feasible). Smallest |deflection error| anywhere
> **0.04%** (feasible). **Both at the same design: NO** — at every start, in the same shape
> S9 found: meeting deflection costs utilisation ~1.77-1.88, meeting stress costs -58% to
> -78% of deflection.
>
> ## What the plan got right, and wrong
>
> **Right.** Splitting the work so the cheap check ran before the 48-hour production run —
> 2 h 18 m bought the discovery that the constraint being optimised has no value. Making the
> two sections opt-in rather than part of `make studies`. The per-row `try`/`except`, which
> cost nothing here but is why one bad genome cannot take the other fifteen.
>
> **Ranking elites by an explicit corner distance rather than by Stage-2 loss, which turned
> out to matter more than expected.** Elites 9 and 10 — the two probed, the two nearest
> feasibility, the two that meet the deflection target as scored — are **10th and 11th of 16
> by Stage-2 loss**. The five *best* Stage-2 genomes (elites 0-4, losses 50.41-50.86) rank
> 11th, 13th, 14th, 10th and 12th by corner distance. Spearman rho between Stage-2 loss and
> FEA corner distance is **-0.285** (p = 0.28, n = 16): not significant, and *negatively*
> signed. The surrogate ranking carries no usable signal about FEA feasibility. Had the
> screen probed "the best elites" it would have probed the four worst-placed designs in the
> set — which is M8a's "the beam surrogate is a bad guide to the FEA" showing up as an
> operational trap rather than a scoring discrepancy.
>
> **Wrong, and it was nearly a silent error.** The plan predicted the failure mode as
> *"`max` diverges while `pnorm` settles, so `c` carries the mesh into `util`"*, and specified
> reporting all three to tell them apart. Measured, that is backwards: `c` is the
> **best**-behaved of the three (GCI 5.33%, and 3.39% at elite 1) and the p-norm is nearly as
> bad as the max. Had the section reported `util` alone — the obvious design — the conclusion
> would have been "the stress QoI does not converge" with no way to see that the p=30
> exponent rather than the rescale is the cause, and step 1 below would have been the wrong
> step. **Reporting the decomposition rather than the verdict is what made the next step
> right.**
>
> **Wrong, and it reached the artifact.** The first `study_stage3_m8bi5.jpg` was captioned
> *"the stress QoI settles under refinement"*, because the title was driven off `settling`
> (`ratio > 1`) rather than a convergence criterion. The figure asserted the opposite of the
> data underneath it. Fixed by splitting `settling` from `converged` and driving every
> verdict — title, printed label, PLAN.md — off the latter. The JSON and figure were then
> regenerated from the stored ladder ROWS, with every previously-written leaf asserted
> bit-identical; no solve was repeated, because `_series` is a pure function of the rows.
>
> ---

## THE NEXT STEP — M8b-i.6, give the stress term a value

**This is now a prerequisite for everything downstream, and it is cheap.** Do not add genes,
do not touch the targets, do not re-weight, and do not run Stage 3 at scale until the
constraint has a mesh-independent value. Every one of those decisions is a decision about a
stress number, and there is not one yet.

### 1. Find the `p` at which the ladder settles — hours, not days

The measurement already exists. `run_mesh_convergence` is parameterised by everything except
`p`, which is currently frozen at `wheel_adjoint.STRESS_PNORM_P = 30.0` and reaches
`_qoi_pnorm_stress(prob, p=...)` through `adjoint_grads` -> `QOI["pnorm_stress"]`.

```bash
# thread `p` from wheel_objective.t3_terms down to _qoi_pnorm_stress, then:
.venv-opt/bin/python study_stage3.py --sections mesh_convergence \
    --ladder-p 2,4,8,16,30 --out study_stage3_pnorm.json
```

Report the observed order and GCI of the p-norm series **per p**, against the axle drop's
2.44 / 0.14% as the standard of what a converged QoI looks like on these meshes. The
prediction, which the measurement should be allowed to refute: p in the 4-8 range is a smooth
percentile and converges; p=30 is a max in disguise and does not. M4's p99 converging to
8.61 MPa is the evidence that *some* smooth aggregate of this field has a value.

**Do not skip the axle-drop column when adding `--ladder-p`.** It is the control that makes
every stress row interpretable, and it costs nothing — it comes out of the same solve.

### 2. Replace the rescale-to-max with an analytic Kt

Once a converged nominal stress exists, the constraint should be

```
Kt(fillet_radius, thickness) * sigma_nominal_converged  <=  ALLOWABLE_STRESS_MPA
```

`wheel_fea.stress_concentration_kt(fillet_radius_mm, thickness_mm, c_factor=1.0)`
(`wheel_fea.py:315`) already exists for precisely this, is analytic, is differentiable in the
thickness genes `t0..t3`, and is clamped to [1.0, 3.5]. M4 names it in the same breath as the
singularity: *"That the corner is nearly a crack is exactly WHY the real part is filleted
there and why `wheel_fea.stress_concentration_kt` exists."*

This deletes `stress_scale` from the objective entirely — and with it the M8a gate-7 hazard,
the S4 discipline, the `refresh_scale` plumbing in `wheel_stage3.Evaluator` and the pinning
in every finite difference. **The single most delicate mechanism in the Stage-3 driver exists
only to make a rescale-to-a-singularity differentiable.** Removing it is a simplification, not
a cost.

`wheel_step_export.kt_report` documents the fillet-feasibility fix as open; that is the
alternative route (mesh the fillet and measure the peak for real) and it is the expensive
one. The analytic Kt is the cheap one and is what the beam surrogate already assumes.

### 3. Only then, re-run S9 and S12

The feasibility question is genuinely open again, and the answer may well be different: the
elite screen already shows deflection met at three designs, and the stress side has never
been evaluated on a quantity with a value. `run_feasibility` and `run_multistart` take the
new objective unchanged.

### The decision that follows, which is a human's — unchanged in list, deferred in time

Rim-band genes / revisit the targets / accept a Pareto point / change material. Every one of
them needs a stress number. None of them should be argued before step 1.

---

## M8b-ii and beyond — unchanged, and still downstream

- **Process-parallel phase batch, a precondition rather than an optimization.** 48.13 h
  serial for 300 steps x 4 starts at 144.4 s per 8-phase evaluation. `OMP_NUM_THREADS=1`
  before the numpy import; phase slots pinned to workers so each traces only its share of the
  lattice.
- **Multi-fidelity checkpoints, and `medium` is cheaper than budgeted.** Measured here on a
  4-phase `tiers=("t3",)` evaluation: `medium` 243 s against `coarse` 87 s, **2.8x**, not the
  4x the last plan assumed. Take that pair from the elite-1 ladder, not the shipped genome's
  (142 s -> 255 s, 1.8x): the shipped genome ran first and its `coarse` rung carries the
  `coord_fn` jit trace for that mesh size, so its ratio flatters `medium`. Both rows are in
  `study_stage3_m8bi5.json`.
- **Jit `t1_vector`** in `wheel_objective.py` — 1.06 s of eager dispatch per call.
- **M9** remains `lambda_min(K_t)` via LOBPCG, replacing the zero-gradient Euler `buckling`
  proxy, which is still exactly 0.0 and still asserted to be.

---

## How to run any of this

```bash
.venv-opt/bin/python study_stage3.py --quick            # wiring check, ~13 min, see NOTE
.venv-opt/bin/python study_stage3.py                    # the M8b-i gate, S1-S10, ~2 h 45 m
make m8bi5                                              # S11 + S12, ~2 h 18 m
make test                                               # 257 tests
make studies                                            # all nine gates; does NOT run m8bi5
```

`--sections` selects and orders sections; the default is the seven S1-S10 sections in their
original order, so a default run writes the same report keys it always has (verified). The
M8b-i.5 pair is opt-in. `--ladder-configs smoke,coarse,medium,fine` adds a fourth rung; `fine`
is 261k dof through contact plus a service-force secant plus an adjoint, which nothing in this
repo has yet run, so each row is wrapped in its own `try`.

**NOTE, a known and pre-existing failure: `study_stage3.py --quick` exits 1 on S8.** Measured
cold 6.16 s against warm 6.32 s, -2.6%. It is a `smoke`-tier artifact and not a regression:
`run_warm` is untouched by M8b-i.5, at `smoke` the 960-element solve is a small share of an
evaluation dominated by meshing and dispatch, and cold always runs first within each rep. At
`coarse` — the gate that counts — S8 passes at +2.4%. No test asserts it, so `make test` is
unaffected. Fix it by giving `run_warm` more reps at `smoke` or by scoping S8 to `coarse`;
do not "fix" it by relaxing `GATE_WARM_SAVING`.

## State of the working tree, for whoever picks this up

> **Ignore version control entirely. Do not commit, branch, stage, revert or otherwise
> touch git — it is not part of this project's workflow and nothing here depends on it.**

`make test` (257) is green. `make m8bi5` and `study_stage3.py` at `coarse` are both green as
the tree stands.

Changed by M8b-i.5: `study_stage3.py`, `tests/test_stage3.py`, `Makefile`, `PLAN.md`, and the
new artifacts `study_stage3_m8bi5.json` / `.jpg`. Nothing else. `study_stage3.json` and its
`.jpg` are M8b-i's and still describe that run — the stress magnitudes in them are the ones
this milestone invalidates, and they are left in place as the record of what was measured
rather than edited after the fact.
