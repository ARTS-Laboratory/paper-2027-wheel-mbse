# M8b-i — the Stage-3 optimizer, and whether its problem has a solution

> **STATUS: CODE COMPLETE. ALL TEN GATES PASS AT `coarse` IN 9941 s (2 h 46 m).**
> `wheel_stage3.py`, `study_stage3.py`, `tests/test_stage3.py`, `study_stage3.json` and
> `.jpg` are new; only the `Makefile` is amended. **No existing module was touched** —
> the warm-start channel, the tier selector, the `stress_scale` input, the orientation pin
> and the 128-entry `coord_fn` cache were all built for this in M8a. Test suite 230 -> 249,
> all green.
>
> | gate | measured at `coarse` | threshold |
> |---|---|---|
> | S1 directional derivative vs an FD ladder of the whole pipeline | **9.5e-6**, 2 rungs | 1e-4, >= 1 |
> | S2 Adam descends, deterministic 25 steps | **2303.05 -> 953.70**, 2.41x | strictly |
> | S3 projection exact; no gene frozen against an inward gradient | **0 box, 0 freeze**, 4 genes freed | exact; none |
> | S4 `stress_scale` used == previous step's measurement | **0.000e+00** over 25 steps | 1e-12 |
> | S5 a failed solve is a step reject, and a storm terminates | **recovers; restores; stops** | recovers, logged |
> | S6 the flank-orientation pin is what the meshes are built with | **[1,1]**, 0 flips | exact |
> | S7 rqmc vs uniform vs iid, step-to-step jitter | **0.148 vs 0.085 vs 0.187** | rqmc < iid |
> | S8 warm start vs cold, per evaluation | **68.44 -> 66.81 s**, 2.4% | not slower |
> | S9 **the feasibility verdict** | reported below | **not gated** |
> | S10 cost per step by tier, and the production projection | reported below | — |
>
> **M8a's two debts are cleared.** `study_objective.py` ran to completion at `coarse`
> (PASS, 4030 s) and `make studies` ran the full sweep (**exit 0, all eight gates PASS**).
> The regression check is now executed rather than argued: two reports came back
> **byte-identical**, and across the other six **every single differing leaf is a
> wall-clock timing or a ratio of timings** — 24 of 4283 in `study_gradient`, 5 of 423 in
> `study_contact`, 1 of 175 in `study_wheel_mesh`. No physics number moved anywhere.
> `study_objective` was then run a second time independently and reproduced itself to 4
> leaves of 667, all timing. The only thing not yet executed as one command is `make
> studies` *including* the new `study_stage3.py` entry; every one of the nine has been run
> and passed individually.
>
> ## The headline: **the problem is infeasible, and by more than M8a thought**
>
> Each constraint is reachable **alone**. Neither is reachable **with the other**.
>
> | probe | stress utilisation | deflection error |
> |---|---|---|
> | stress only | 1.713 -> **0.972** (min 0.932) | -25.4% -> **-77.0%** |
> | deflection only | 1.713 -> **1.820** | -25.4% -> **+1.4%** (min abs 0.22%) |
> | joint, real weights | 1.713 -> **1.203** | -25.4% -> **-54.9%** |
>
> Stress can be met, at 77% short of the deflection target. Deflection can be met, at 82%
> over the allowable. **No design visited by any of the three descents satisfies both**,
> and the joint run settles in between rather than finding a corner neither single-objective
> run could see. `study_stage3.jpg`'s third panel is the whole result: the feasible box sits
> in the corner, untouched, and the frontier sweeps past it.
>
> **And M8a's headline understated it by 38%.** `PLAN.md`'s "31.02 MPa against 25.0" is a
> **`smoke`** measurement — this milestone reproduces it exactly at `smoke` (31.01 MPa,
> utilisation 1.2406) and measures **42.82 MPa, utilisation 1.7128**, at `coarse`. Peak
> stress rises with refinement, as a resolved stress concentration does, so the verdict
> above is a *lower* bound on the infeasibility and the mesh convergence of the stress QoI
> is now the load-bearing open question. M4's `compliance_split.rim = 0.324` remains the
> standing explanation for the deflection half: a third of the compliance sits in a rim band
> no gene touches.
>
> **What the plan got right.** Projection rather than sigmoid reparameterisation — 4 genes
> left their bounds during the 25-step run, and all six start pinned, so a squashing map
> would have begun with them frozen. The `stress_scale` discipline, which held to *exactly*
> zero over 25 steps. The prediction that warm-starting needed no change to
> `wheel_objective` (`delta0` and `axle_drop_mm` are the same number,
> `wheel_adjoint.py:537`). The quantized lattice: `iid` drew 9 stencils in 9 steps and
> **never once got a cache repeat**, while `rqmc` paid 4 first visits at 163 s and then ran
> at 73 s. And the gate-before-consume split itself, which is what stopped a 48-hour
> production run being launched at an infeasible problem.
>
> **What the plan got wrong, and the measurements that said so.**
>
> 1. *The T1 pre-check rule deadlocked the feasibility probe, and the first coarse run
>    reported the deadlock as a result.* The rule was "reject a trial whose barrier sum
>    exceeds the current point's". The shipped design carries `hub_overlap` = 52.13; the
>    stress-reducing direction raises it to **55.41**, and halving only walks it back to
>    **52.54** — still above. Every trial at every scale was refused, all 40 steps were
>    abandoned, and the probe reported **its own starting point** as the lowest reachable
>    utilisation: 194 events, **zero accepted steps**, 35 minutes standing still. The error
>    was conceptual. Every barrier is *already* a weighted term in the objective —
>    `hub_overlap` at 500·(violation)² is how the optimizer is told — so a screen that also
>    forbids increasing one overrides the weights and vetoes precisely the constrained trade
>    the probe exists to make. `T1_REJECT` is now **1e4**, a gross "this will not mesh"
>    threshold against barriers that sit in the tens.
> 2. *And S9's gate could not tell a frozen run from a converged one, which is why it
>    passed.* The pass condition was `loss_end == loss_end` — a NaN check. "The function
>    returned" counted as success. It now requires every probe to have **accepted at least
>    one step and moved**, which is the condition that would have caught this
>    automatically. A bound obtained by never moving is not a bound on anything.
> 3. *The cost model was out by a factor of 26, in the direction that matters.* The plan
>    budgeted 0.68 s/phase from M7 — but that figure is a warm single-phase gradient with a
>    prebuilt mesh, not an objective evaluation. Measured: **18.05 s/phase, 144.4 s for an
>    8-phase evaluation**, which projects the production run to **48.13 hours serial**
>    against the plan's 1.8. Independently corroborated by M8a's own `cheap_fraction` of
>    1.22%, which implies a ~114 s `coarse` evaluation. **Process parallelism is a
>    precondition for M8b-ii, not an optimization of it.**
> 4. *T1 is not "~ms", but the tiering argument survives anyway, and quoting either number
>    alone misleads.* `t1_vector` carries no `@jax.jit`, so `jacrev` runs it eagerly:
>    **1.06 s** per call at `coarse`, three orders off `wheel_objective.py:33`. As a
>    *fraction* it is **0.73%** of a full `coarse` evaluation, so refusing a trial for 1 s
>    instead of spending 144 s on it is still the trade M8a described. At `smoke` the same
>    second lands against a 7 s solve and becomes 21% — a statement about how cheap the
>    mesh is, not about T1.
> 5. *A three-hour study that prints nothing until it finishes is unobservable.* The first
>    coarse run gave no signal for 3 h 20 m, which is how the deadlock stayed invisible.
>    Per-section progress is now printed and flushed.

## Context

M8a's closing line was an open question with numbers attached: *"Whether M8b has a feasible
problem is now an open question."* Answering it by launching the production run and watching
it fail to converge was the expensive option, and a Stage-3 run against an infeasible problem
is indistinguishable from one against a buggy optimizer — in both cases the trajectory stalls
short of target and in both cases the gradient is correct.

So M8b splits. **M8b-i is the optimizer, its gate, and a cheap probe that answers the
question first.** M8b-ii is the process-parallel batch, the multi-fidelity checkpoints and
the production run — and its scope is now a different question than it was, because the
answer came back *no*.

**No constant was relaxed.** `TARGET_DEFLECTION_MM`, `ALLOWABLE_STRESS_MPA` and
`DEFAULT_WEIGHTS` are exactly as M8a ported them. Re-weighting until a trajectory looked
convergent would have hidden the one result this milestone existed to produce.

---

## Architecture — the per-step protocol is the design

`wheel_stage3.py` is a driver and nothing else: no new physics, no new gradient, no new
term. What it *does* own is five decisions, each of which fails silently if got wrong — the
run does not crash, it converges somewhere slightly wrong and looks like a hard problem.

1. **`stress_scale` is refreshed between steps and pinned within one.** The p-norm→max
   rescale is exact only for a *constant* factor (`wheel_objective.py:487`); M8a's gate 7
   measured 10% of error in the assembled gradient while every individual term still matched
   its own FD to 1e-8. Each evaluation is handed the ratio measured at the previous point,
   so value and gradient at any one point answer the same question. S4 holds this to exactly
   zero.
2. **Phases come from a fixed lattice and meshes are built once per step.**
   `--phase-scheme {rqmc,uniform,iid}` ships here; `rqmc` is the default. Meshes are built
   once via `phase_meshes` and passed as `meshes=`, or `objective` rebuilds eight of them
   per call.
3. **Flank orientation is pinned for the whole run** and threaded through
   `phase_meshes(orientation=...)`. It is re-derived every step *for reporting only*; a
   disagreement is an `orientation_flip` event. S6 checks the pin by watching what
   `phase_meshes` is actually called with — not by building the flipped mesh, because at the
   shipped genome the `eta=-1` hub arrival **does not exist** (the flank never reaches
   r = 12.7 mm), which is the same real geometry M8a's fillet term ran into.
4. **Warm starting is free and is taken**: the previous step's per-phase drops are the next
   step's `delta0`. Worth 2.4% per evaluation, measured.
5. **A failed solve is a step reject.** `wheel_objective` contains no `try`/`except` by
   design, so the driver owns the policy: halve, retry to `--max-rejects`, then restore the
   iterate and decay the rate. A non-descent tangent *is* the only buckling signal this repo
   owns until M9. Five consecutive abandonments now stop the run with a
   `run_stopped_stuck` event rather than burning the budget in place.

**Projection, not reparameterisation.** `z <- clip(z - step, 0, 1)`. Six of fourteen genes
sit exactly on a bound at the shipped genome, and a sigmoid map has vanishing gradient there.
S3 verifies the *implication* rather than hoping a gene moved: a gene on a bound stays there
if and only if the unprojected step would have left the box.

---

## The feasibility probe

Three descents at `coarse`, 20 steps each, differing only in which **objectives** are zeroed
— every geometric barrier stays on in all three, because a "lowest reachable stress" reached
through a folded or unmeshable design bounds nothing. 20 steps rather than 40 because every
probe plateaus well inside it, measured.

The verdict is **reported and deliberately not gated**. Gating on feasibility would report a
fact about the wheel as a fact about the code, and would get the wrong thing fixed.

---

## Files

- **`wheel_stage3.py`** — new. Projected Adam, `--optimizer lbfgsb` as the deterministic
  alternative (which *refuses* a stochastic stencil rather than fitting curvature to noise),
  the five protocol decisions, incremental atomic `stage3_run.json`, and
  `stage3_best.json` through `wheel_genome.save_record`.
- **`study_stage3.py`** — new. S1–S10 and the probe.
- **`tests/test_stage3.py`** — new, 19 tests. Tolerances read off `study_stage3.GATE_*`.
  Two of them are regressions on the deadlock above.
- **`Makefile`** — `study_stage3.py` added to `studies` and to `help`; new `stage3:` target.

---

## Verification

```bash
.venv-opt/bin/python study_stage3.py --quick     # fast loop during development
.venv-opt/bin/python study_stage3.py             # the gate; exits nonzero on failure
make test                                        # 249 tests
make studies                                     # all nine gates
```

`study_stage3.py` writes its JSON **before** formatting its report: at `coarse` the run is
hours and `_print` is string formatting, and losing the former to a bug in the latter is a
trade nobody would make on purpose.

---

## THE NEXT STEP — M8b-i.5, and why it comes before any design change

**Read this first: the verdict above is weaker than it looks, and the weakness is
specific.** S9 ran three descents, all from `best_solution.json`, all at `coarse`. That
supports *"no feasible design was found along three descents from one start at one
fidelity"*. It does **not** support *"the design space contains no feasible point"*, and the
difference decides whether the genome needs new genes or not.

Two things make the single start suspect rather than merely narrow:

- `best_solution.json` is a converged GA optimum for the **beam surrogate**, and M8a
  established that surrogate is a bad guide to the FEA — it scored `deflection` at 0.1% of
  the loss where the FEA says 33.7%, and ranked `mass` first at 79.2% where the FEA puts it
  fourth at 10.8%. The GA optimised toward a corner chosen by a model we now know is wrong.
- Adam is a local method and every probe plateaued. A plateau is evidence about a basin, not
  about a space.

`stage2_elites.json` holds **16 distinct converged genomes** at Stage-2 losses 50.41–52.53,
already on disk, and nothing has ever evaluated the FEA objective at 15 of them.

### The work, in order

**1. Mesh convergence of the stress QoI.** This gates everything, because if utilisation is
still climbing the infeasibility is larger than reported and no weight or gene discussion is
meaningful yet.

Measured so far at the shipped genome: utilisation **1.2406** at `smoke`, **1.7128** at
`coarse` — 31.01 -> 42.82 MPa. Add `medium` (and `fine` if it is affordable) and report the
sequence. This is the same question M4 asked of `fea_over_beam` and M8a asked of the
`max/pnorm` ratio: does the number settle, and if not, which way is it going.

New study, or a section appended to `study_stage3.py` — a section is preferable, because the
verdict it qualifies lives there. Budget: a `medium` 8-phase evaluation is roughly 4x
`coarse`'s 144 s, so a handful of designs at 1–4 phases is ~30–60 min.

**2. Multi-start feasibility screen.** Two stages, cheap then expensive:

```bash
# (a) score all 16 elites, no descent — ~16 x 144 s = 38 min at coarse
#     report utilisation and deflection error per elite; look for spread
# (b) re-run the stress-only and deflection-only probes from the 2-3 elites
#     that sit closest to the feasible corner — ~1 h
```

`wheel_stage3.start_points("all")` and `study_stage3.load_elites()` already read the file;
`run_feasibility` already takes `genes`, so (b) is a loop over starts rather than new
machinery. What changes is the **verdict**, which must become "over N starts" or stay silent.

**Total ~2 h, against 48 h for a production run, and either half can change the decision.**

### The decision that follows, which is a human's

Only once the two checks are in. If the infeasibility survives them:

1. **Add rim-band genes.** M4's `compliance_split.rim = 0.324` — a third of the compliance
   sits in a band no gene touches, and `deflection_only` reached its target only by driving
   utilisation to 1.82. This is the "the optimizer needs more freedom" answer, and it is
   substantial: new genes, `GENE_SPACE`, mesh parameterisation, and a Stage-2 re-run, i.e.
   everything downstream of the genome moves.
2. **Revisit the targets.** 2.0 mm at 25 MPa may be over-specified for PLA at this geometry.
   Cheapest path to a feasible problem, but it is a decision about what the wheel is *for*.
3. **Accept a Pareto point.** Run Stage 3 with explicit trade weights, ship the best
   compromise, and document that neither constraint is met.
4. **Change material or process.** A higher-strength filament raises the allowable directly.

**Do not resolve this by re-weighting.** The weights arbitrate trades inside a feasible set;
they cannot manufacture one, and tuning them until the plot looks convergent would bury the
only result this milestone produced.

---

## M8b-ii and beyond — unchanged, but now downstream of the above

- **Process-parallel phase batch, and it is a precondition rather than an optimization.**
  48.13 h serial for 300 steps × 4 starts. `OMP_NUM_THREADS=1` before the numpy import, and
  phase slots pinned to workers so each traces only its own share of the lattice — the S7
  first-visit/repeat split (163 s vs 73 s) is exactly the cost that pinning preserves.
- **Multi-fidelity checkpoints**, warm-starting each `medium` phase from the converged
  `coarse` state; measured, not assumed.
- **Jit `t1_vector`** in `wheel_objective.py` — 1.06 s of eager dispatch per call, paid on
  every trial step.
- **M9** remains `λ_min(K_t)` via LOBPCG, replacing the zero-gradient Euler `buckling`
  proxy, which is still exactly 0.0 and still asserted to be.

---

## State of the working tree, for whoever picks this up

> **Ignore version control entirely. Do not commit, branch, stage, revert or otherwise
> touch git — it is not part of this project's workflow and nothing here depends on it.**

`make test` (249) and `study_stage3.py` at `coarse` are both green as the tree stands. New
files: `wheel_objective.py`, `study_objective.py`, `wheel_stage3.py`, `study_stage3.py`,
`tests/test_objective.py`, `tests/test_stage3.py`, `stage2_elites.json`, and the
`study_objective.*` / `study_stage3.*` artifacts. Also changed: `Makefile`, `PLAN.md`, and
M8a's five module edits (`wheel_adjoint`, `wheel_fea`, `wheel_fem`, `wheel_mesh`,
`wheel_wheel`).

The five re-run `study_*.json` reports differ from their previous contents **only in
wall-clock timings and ratios of timings**, verified leaf by leaf — no physics number moved.
