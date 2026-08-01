# PLAN.md — the next changes

> **Ignore version control entirely. Do not commit, branch, stage, revert or otherwise
> touch git — it is not part of this project's workflow and nothing here depends on it.**

---

## Where the tree stands — the minimum a fresh session needs

**M8b-i.6 step 2 landed.** The stress constraint is no longer a p-norm rescaled to the true
max by a measured ratio. It is now

```
Kt(R, t) * sigma_nominal(p=4)  <=  ALLOWABLE_STRESS_MPA        # = 25.0
```

with **one `soft_barrier` per junction, summed** — hub priced on `(R_hub, t0)`, rim on
`(R_rim, t3)`. `Kt = 1 + C*(t/2R)^0.65` clamped to [1.0, 3.5], differentiated by `jax.grad`,
not frozen. `stress_scale` is gone from `t3_terms` and `objective`; `stress_scale_measured`
survives in the report as a read-only diagnostic because it is the *evidence* for the change.

Why it had to change: `c = max/pnorm` is anchored to M4's crack-tip singularity, which
diverges under refinement, so `c * pnorm` converged at **no exponent at either design**. The
sweep that proved it is `study_stage3_pnorm.json` (`make m8bi6`).

**Key constants.** `wheel_objective.STRESS_NOMINAL_P = 4.0` is the `t3_terms` default.
`wheel_adjoint.STRESS_PNORM_P` stays **30.0** — it is the documented default every historical
record was measured at, and a test pins it there.

**Measured after the change** (`medium` rung):

| design | Kt_hub | Kt_rim | sigma_nom(p=4) | **util** | util GCI | field max |
|---|---|---|---|---|---|---|
| best_solution | 1.861 | 1.490 | 5.507 MPa | **0.4099** | **0.45%** | 48.47 MPa |
| elite 1 | 1.871 | 1.490 | 6.765 MPa | **0.5063** | **0.20%** | 71.40 MPa |

Someone will put `util = 0.41` next to `max = 48.5 MPa` and panic. The answer is M4's,
unchanged: **the max is not a number.** It diverges 31.02 → 41.54 → 48.47 under refinement.

**Gates, all green:** `make test` **383 passed** (357 before the phase pool; 269 before
M8b-i.6, whose Kt-twin equivalence test is parameterised 84 ways). Gate 7 `min_decades`
**2**, `worst_best_rel` **2.009e-07** — both better than the 1 / 1.820e-05 baseline.
`make m8bi6`'s `pnorm_by_p` block reproduces the step-1 sweep **bit-identically**,
0.000e+00 on every value and every GCI including the `c` columns; `max_stress_mpa` and
`axle_drop_mean_mm` are identical too, confirming no physics moved.

**M8b-ii item 1 also landed: the phase loop is process-parallel.** `--workers` on
`wheel_stage3.py`, gated by S13 (`make m8bii1`, PASS). **3.95× at 8 workers on 16 cores,
and the production run projects 46.46 h → 11.77 h.** Pooled values are bit-identical to
serial; gradients agree to 2.1e-16. Details, and the `XLA_FLAGS` finding that made the
comparison meaningful, are in "the next changes" below.

**M8b-ii item 2 also landed: `t1_vector` is jitted and cached.** T1 and T2 run in the
PARENT on every path (serial or pooled), so once the phases were parallel they became
the whole of Amdahl's serial fraction — this is why the item mattered more than its raw
number suggested. `wheel_objective._t1_cached_value_and_jacobian` replaces the old
`t1_vector(...)` call plus a separate `jax.jacrev(t1_vector)(...)` call with one
`jax.jit`-compiled closure (`jax.vjp` + `jax.vmap` over an identity basis, sharing the
one forward pass) cached on `(cfg.name, span_mm, flanks, weights-projected-to-the-six-
T1-keys)`, exactly the `coord_fn` idiom — genes are the sole traced argument, never in
the key. `t1_vector` itself is untouched; the cache is a new call path in `objective()`
only. **Measured (S10, `coarse`): 1.06 s → 0.0054 s, ~195×, 0.73% → 0.003% of a full
evaluation.** `make test` **384 passed** (383 before). `t2_vector` was deliberately left
alone — its own eager surface is small (`mesh_coords` already routes through the
jitted, cached `coord_fn`) and there's no measured complaint to fix; revisit only if a
future S10 run shows it's material.

**M8b-ii item 3 also landed: a periodic fidelity check exists on `descend()`.** There
was no spec for "multi-fidelity checkpoints" beyond the budget number (`medium` 2.8×
`coarse`) — chose the narrowest reading: `fidelity_check_every`/`fidelity_check_cfg`
forward-evaluate the just-accepted iterate at a second config every N steps, t3 tier
only, discard the gradient, and attach the raw result to that step's row under
`"fidelity_check"`. Pure observation — the discarded gradient never reaches
`m`/`v`/`delta`/`z`, proven by a test that runs the same seed with the feature on and
off and asserts `z`/`grad`/`loss` are bit-identical at every step. No disagreement
threshold is invented (matches the M8b-i.6 lesson about `stress_scale`'s old rescale:
don't calibrate a threshold with no measurement behind it); a second-fidelity solve
failure is a `fidelity_check_failed` event, not an abort. CLI:
`--fidelity-check-every`/`--fidelity-check-config`, off by default. `descend_lbfgsb` is
untouched (not the default path, and its `fun` is called on every line-search
evaluation, not just accepted steps — "every N steps" would mean something else there).
`make test` **391 passed** (384 before).

**The hub fillet landed too — §3 below, full record in `HUB_PLAN.md`.** `_embed`'s inward
step now plunges radially instead of running 4.5 mm sideways, so the hub circle exists again
and **all 24 of its corners are filleted, from 0 of 12**; `fillet_junctions` no longer
abandons the family that refused the first rung, which also takes the rim from 12 of 24 to
24 of 24; and a junction is now priced by its **worst** corner, which is why the rim's
`kt_error_pct` moved off a +0.0% that had twelve square corners hiding behind it. `make test`
**396 passed** (395 before, one of which was the contract test pinning the old broken hub).

---

## THE VERDICT REVERSED: the problem is FEASIBLE, and always was

S9 called this design space infeasible. That verdict was read off `c * pnorm`, which has no
mesh-independent value. Re-scored on a constraint that does:

| elite | util | defl err | corner distance |
|---|---|---|---|
| **10** | **0.4548** | **+1.65%** | **0.000  ← FEASIBLE** |
| **9** | **0.4468** | **+2.29%** | **0.000  ← FEASIBLE** |
| 7 | 0.4667 | +7.93% | 0.586 |
| 12 | 0.4710 | +13.78% | 1.757 |
| 0 (`best_solution`) | 0.4063 | −25.43% | 4.086 |

All 16 scored, none failed. **Every elite is stress-feasible** (util 0.23–0.53 against 1.0);
the binding constraint is now deflection alone. Two elites are inside the feasible box on
both. `corner_distance` is zero only when `util <= 1.0` **and** `|defl_err| <= 5%`.

Spread across the 16: utilisation 0.2272–0.5272, deflection error −77.27% to +28.38%. **That
is a wide spread — these elites are not one basin**, which is what made S9's three descents
from a single start a statement about a basin rather than about the space.

### And the bound descents agree — `make m8bi5` complete, 9086.3 s, OVERALL: PASS

Both probed starts satisfy **both** constraints at a visited design, 20/20 steps accepted,
no reject events:

| start | probe | utilisation | deflection error |
|---|---|---|---|
| elite 9 | `stress_only` | 0.447 → 0.456 | 2.3% → −1.4% |
| elite 9 | `deflection_only` | 0.447 → 0.443 | 2.3% → **0.04%** |
| elite 10 | `stress_only` | 0.455 → 0.458 | 1.7% → 0.2% |
| elite 10 | `deflection_only` | 0.455 → 0.448 | 1.7% → 0.3% |

Restated over 2 probed starts and 16 scored designs, 100 (util, deflection) pairs measured:

```
lowest utilisation seen anywhere   0.2272   (feasible at <= 1.00)
smallest |deflection error| seen   0.04%    (feasible at <= 5%)
BOTH satisfied at any design       YES
```

**S9 said "each constraint is reachable alone, neither with the other." That is now false.**
Driving deflection to 0.04% error *lowered* utilisation (0.447 → 0.443). The two constraints
were never in tension; the tension was an artifact of measuring stress against a singularity.
The old bound "min reachable utilisation 0.932" is **invalid** — it is `c * pnorm` at p=30.

**So the genome does not need new genes to reach feasibility.** The question changes from
"can this wheel be built" to "how good a wheel can be built".

---

## The next changes, in order

### 0. WHAT TO DO NEXT — three things, in this order

**~~(a) `R_hub`'s bound against the buildable ceiling.~~ DONE — the constraint learned the
cap. §5 below is the record.** Of the three ways out, the chosen one was to teach the
constraint rather than drop the bound to 1.1: the void is a function of the arrival angle and
`t0`, not a constant, and across the 16 Stage-2 elites the cap spans **0.9898 to 1.5265 mm**,
so a fixed bound would have been right for exactly one genome. The 4.0 box bound is
untouched; the cap does the work.

**~~(b) The production multi-start run.~~ DONE, BOTH STARTS — §6 below is the record.**
Elites 9 and 10, mass as the objective, `--workers 4`, `--phase-scheme uniform`, fidelity
check off. Everything it was waiting on landed and is measured: the phase pool, the jitted
`t1_vector`, the fidelity check. `make prod9` / `make prod10` are the targets.

**Headline: 73.689 g → 58.715 g (elite 9, −20.3%) and 70.937 g → 58.660 g (elite 10,
−17.3%), with every barrier at exactly 0.0 and deflection error inside ±0.3% on both.** Mass
ends at 97% of the loss and deflection at 46% of its gradient — the inversion §1 predicted,
arrived at twice.

**And the two starts land 0.09% apart in mass with 1.8 mm of daylight between their spoke
centerlines.** The optimum is a **valley, not a well**: at the 2.0 mm wall floor a family of
centerlines has the same length and meets the same deflection. That makes further starts
pointless and the leftover freedom interesting — see §6, and it is a better argument for §2's
M9 than §2 makes for itself.

**The floor is now the binding design decision.** All four thickness genes sit on
`MIN_WALL_MM = 2.0` at both answers, so a manufacturing constant, not the FEA, sets 4 of the
14 genes. It is on "The decision that is a human's" with a number attached.

**The starts must run SEQUENTIALLY.** One descent at `coarse` with 4 workers sits flat at
**~12.7 GB anonymous** with the fidelity check off; two of them do not fit in 31 GB. §1's
"two starts × 4 workers beats one start × 8" is a CPU-efficiency argument that ignores
memory, and acting on it is what produced the measurement — see the warning there. The
17 GB once quoted there was measured with the fidelity check ON and before the `uniform`
finding; both are fixed now, and the `Makefile`'s `PROD_FIDELITY` default is **0**
accordingly.

**(c) `make m9` in full.** Only `--quick` has ever been run, and M9 phase 3 — promoting
`lambda_min(K_t)` from a mechanism to a constraint, with a margin, a threshold and a phase
aggregation rule — is deliberately blocked on that measurement (`tests/test_gradient.py`
says so in as many words). **It is now FIRST — (a) and (b) are done and the machine is
free** — and §6 strengthened the case for it: the production answer is a valley with real
geometric freedom left in it that the mass objective cannot spend, and `buckling` is still
the one constraint in the loss with a gradient of exactly 0.0. Budget it carefully: the full
driver is 3 designs × 4 configs including the **never-run 261k-dof `fine` rung** at 13
phases, against 38.6 s for `--quick` at 1 design / `smoke` / 2 phases. That extrapolation
is not linear and `fine`'s memory is unmeasured, so run it capped like §0(b)'s descents
rather than bare.

**The STEP consumer is Autodesk Inventor now, not Onshape.** Onshape was temporary. The
`wheel.step` interop history in `wheel_step_export.py` is Parasolid's, and Inventor is a
different kernel, so that record is history rather than a live constraint and re-testing the
Onshape import is NOT on this list. What still travels: `despecialize` exists because a
`SURFACE_OF_LINEAR_EXTRUSION` is not universally supported and the hub bore and OD are kept
analytic on purpose, and `wheel_nofillet.step` is still written before any fillet is
attempted. Whether Inventor is happy with the 48 fillet faces and the 0.162 mm shortest edge
the hub fillet added is unmeasured — worth one import, not a milestone.

### 1. M8b-ii — make the optimizer runnable at scale

Feasible points exist; the optimizer could not search for better ones in reasonable time.
The phase batch is parallel, `t1_vector` is jitted, and the fidelity check exists, all
measured; only the production run itself is open.

- **~~Process-parallel phase batch.~~ DONE — `--workers`, gated by S13 (`make m8bii1`).**

  `src/wheel_pool.py` (`PhasePool`) + `src/wheel_pool_worker.py`. Measured at `coarse`,
  8 phases, on a **16-core** box — the hours below do not travel to another machine, the
  efficiency column does:

  | workers | s / eval | speedup | efficiency | values vs serial | grad rel |
  |---|---|---|---|---|---|
  | serial | 139.4 | — | — | — | — |
  | 1 | 136.7 | 1.02× | 1.02 | **exact** | 0.0 |
  | 2 | 76.4 | 1.82× | 0.91 | **exact** | 0.0 |
  | 4 | 47.6 | 2.93× | 0.73 | **exact** | 2.1e-16 |
  | 8 | 35.3 | 3.95× | 0.49 | **exact** | 2.1e-16 |

  **The production projection is 46.46 h → 11.77 h** (300 steps × 4 starts).

  `0` is serial and stays the DEFAULT, so every gate and every committed artifact still
  runs the path they were measured on. `-1` sizes the pool to `min(n_phase, cpu_count)`;
  `N` is taken literally and is the only cap on memory, since the auto-size counts cores
  and knows nothing about RAM.

  Slot `i` is pinned to worker `i % n`, which is why this is not `multiprocessing.Pool`:
  `coord_fn` keys its jit cache on `float(phase)` and a miss is 0.774 s, so a phase that
  wanders between workers pays that forever. Replies come back in **slot order**, so every
  reduction sums the same floats in the same sequence as serial.

  **`XLA_FLAGS` is pinned along with the four thread counts, and it is a correctness
  setting.** Measured: two *plain serial* runs of one `coarse` adjoint, in two separate
  interpreters with **no pool anywhere**, agree on every forward value to the bit and
  disagree on the GRADIENT by 3.33e-16 — XLA's CPU thread pool does not associate its
  reductions the same way twice, and nothing in OMP/MKL/OPENBLAS/NUMEXPR reaches it.
  Pinned, that comparison is exactly zero, and it costs nothing: 19.84 s against 20.43 s
  for one `coarse` phase. Set in the `Makefile`, in `conftest.py`, and in
  `wheel_pool.PINNED_ENV` for a worker started by hand.

  **S13 gates values EXACTLY and gradients at 1e-14, and the split is a measurement.**
  Exact bit-identity of the gradient is not available to *anyone* here, pooled or not —
  what remains after the XLA pin is process-history dependent (a process that has already
  run other phases answers the next one differently in its last bit) and lives below this
  repo, in XLA's own codegen. Observed 2.1e-16 against a 1e-14 gate. Every value, every
  report leaf and the stress and ripple gradients are 0.0.

  **The thread pin moved no physics**: `make m8bi6` re-run and diffed against the previous
  `study_stage3_pnorm.json` — **2038 non-timing leaves, 0 differ**, gate verdict unchanged.
  The artifact in the tree is the earlier one, restored, because the two differed only in
  wall clock and the earlier run's timings were measured on a quiet machine.

  **End to end**, 2 steps from elite 9 at `coarse`, `--phase-scheme uniform`, serial vs
  `--workers -1`: **639.4 s → 201.6 s (3.17×)**, every step's loss equal to the last bit
  and the same final `genome_hash` (`99fea84`), no events either side, no orphaned
  workers. 3.17× rather than S13's 3.95× because the T1 precheck and step 0's tracing are
  a larger share of a 3-call run than of a steady-state one.

  **Do not read that exactness as a guarantee.** A trajectory can diverge in its last bits
  whenever a gradient difference survives the Adam update into the accepted iterate; it
  happened not to at this design. The claim S13 supports is per-evaluation — values exact,
  gradients to 1e-14 — not per-trajectory.

- **~~Jit `t1_vector`.~~ DONE — `_t1_cached_value_and_jacobian`, `wheel_objective.py`.**
  1.06 s → 0.0054 s at `coarse` (S10), ~195×. Details above, in "M8b-ii item 2 also
  landed."
- **~~Multi-fidelity checkpoints.~~ DONE — `fidelity_check_every`/`fidelity_check_cfg`
  on `wheel_stage3.descend()`, `--fidelity-check-every`/`--fidelity-check-config` on the
  CLI.** There was no spec for this beyond the budget number above (`medium` is 2.8×
  `coarse`, 243 s vs 87 s, not the 4× once assumed — still true, taken from the elite-1
  ladder rather than the shipped genome's for the reason already stated: the shipped
  genome runs first and its `coarse` rung carries the `coord_fn` jit trace). Chose the
  narrowest reading: a PURE OBSERVATION, off by default. Every N accepted steps (and at
  step 0), the just-accepted iterate is forward-evaluated a second time at a second
  config, t3 tier only, and the gradient that call returns is discarded — it can never
  reach `m`/`v`/`delta`/`z`, so it can only report on the trajectory, not redirect it.
  Result goes on that step's row under `"fidelity_check"` (raw `report` dict, not a
  diff/ratio — no disagreement threshold is invented here, on purpose: this repo already
  removed one derived, unrevisited quantity, `stress_scale`'s old rescale role, once its
  assumptions stopped holding, and a guessed threshold would repeat that with no
  measurement behind it yet). A second-fidelity solve failure is recorded as a
  `fidelity_check_failed` event and does not abort the run. A scheduled step that lands
  on an abandoned step is skipped, not rescheduled. `descend_lbfgsb` is untouched —
  `--optimizer lbfgsb` isn't the default and its `fun(z)` is called on every line-search
  evaluation rather than only accepted steps, so "every N steps" would mean something
  different there. `make test` **391 passed** (384 before). Recommended CLI value for
  the production run below: `--fidelity-check-every 25` or `50` against
  `--fidelity-check-config medium`, adding roughly 49 or 24 minutes to an 11.77 h run.

  **MEASURED, and that estimate was ~3× too low: one check costs 604.6 s (10.1 min).**
  `settings.fidelity_check_solve_s` from the first production attempt, one call, `medium`
  against a `coarse` run. So every-50 over 300 steps is 7 checks ≈ **1.2 h**, not 24 min,
  and every-25 is ≈ 2.4 h. The estimate above assumed the check rides the phase pool; it
  does not — `descend` builds `ev_fc` as a **wholly separate serial `Evaluator`**, on
  purpose, so the check pays 8 phases end to end with no parallelism at all.

  **It also costs ~3.4 GB, permanently.** That second `Evaluator` retains its own jit cache
  and mesh for the life of the run, so the parent goes 4.5 → 7.9 GB across the FIRST check
  and stays there. On a memory-bound box this is the first thing to turn off:
  `--fidelity-check-every 0`, which is what `make prod9`'s `PROD_FIDELITY` exposes. The
  check is a pure observation, so dropping it costs evidence, never trajectory.
- **Then the production multi-start run.** Start from elites 9 and 10, not
  `best_solution.json` — that is a GA optimum for the BEAM surrogate, which M8a measured as a
  bad guide to the FEA, and it sits at −25.43% deflection error. The 16-elite spread is wide
  (util range 0.30, deflection range 105.66 points), so a multi-start genuinely samples
  different basins rather than re-running one.

  **The objective it should descend is now mass**, not feasibility. Both constraints are
  satisfiable together and the barriers are flat at every feasible design, so `mass` is the
  only term with anything left to give — it was 19.6% of the loss at the shipped genome
  against deflection's 61.3%, and that ratio inverts once deflection is met.

  **Run it with `--workers`.** At 11.77 h for 300 × 4 this is now affordable. But 8 workers
  is not obviously the right setting: 4 gets 2.93× at 0.73 efficiency on a quarter of a
  16-core box, so **two starts × 4 workers beats one start × 8** and finishes the same
  budget sooner. Decide that against the machine it actually runs on.

  **DO NOT DO THAT. The paragraph above is CPU arithmetic on a machine that runs out of
  MEMORY first, and it was tried.** Measured on the 16-core / 31 GB box: one descent at
  `coarse` with 4 workers holds **~17 GB anonymous** (parent ~4.5 GB with checks off, 7.9 GB
  with them on; four workers ~3.3 GB each) and worker RSS was **still climbing** at step 5
  (2.2 → 3.4 GB), so the steady-state ceiling is unknown. Two starts is ~34-42 GB against
  31 GB of RAM and a 2 GB swapfile.

  **That 17 GB is the WORST case and has since been beaten down to a flat ~12.7 GB** — it
  was measured with `rqmc` (64 retained jit traces, see below) and the fidelity check ON
  (+3.4 GB in the parent, permanently). With `--phase-scheme uniform` and
  `--fidelity-check-every 0`, which is what §6's completed run used and what the `Makefile`
  now defaults to, the footprint is flat rather than climbing and the ceiling is known.
  **Two starts still do not fit** — 25 GB against 31 GB and a 2 GB swapfile leaves nothing
  for the desktop — so the sequential rule stands on the smaller number too.

  What that costs is not a slow run, it is **the desktop**: `systemd-oomd` kills the entire
  `user@1000.service` slice when memory pressure holds above 50% for 20 s, which drops the
  user to the login screen AND kills every run and terminal in that slice. The kernel OOM
  killer is not involved and `dmesg` shows nothing — `journalctl -u systemd-oomd` is where
  the evidence is. So on a memory-bound host: **run the starts one at a time, capped**, e.g.
  `systemd-run --user --unit=wheel-prod9 -p MemoryMax=20G --collect make prod9`, which
  inverts the failure — the kernel kills the run at its own cap instead of oomd killing the
  session. `--workers` is then bounded by RAM/worker, not by cores.

  **RESOLVED: S13's 47.6 s/eval is right, and `rqmc` re-tracing was the entire gap.**
  Under `rqmc` the per-step cost wandered 132/130/128/**52** on one run and 180/135/132 on
  another, which read like noise. Under `uniform`, on the same box and the same 4 workers:
  step 0 is 177.8 s (tracing), then 55.3, 46.7, and a steady 44-47 s thereafter — median
  **48.7 s**, and the last five steps at 46.3/46.4/47.4/44.9/44.4. **300 steps is ~3.8 h.**

  So roughly **80 s of every `rqmc` step was JIT compilation, not solving.** The trace cache
  below explains the wall clock and the memory with one mechanism; there was never a second
  problem. Still read `settings.elapsed_s / n_objective_calls` off a run before projecting,
  but S13's ladder travels once the phase set is fixed.

- **THE PRODUCTION RUN MUST USE `--phase-scheme uniform`, AND THE REASON IS MEMORY.**
  This is the finding that killed the first two production attempts, and neither
  `--workers` nor the fidelity check is the lever on it.

  `rqmc` — the DEFAULT, and what S13 and every projection above assumed — redraws the
  stencil every step from the `n_sub`-point sub-lattice. Verified from a run record: step 0
  drew phases at 2.812°, 6.562°, ... and step 1 drew 2.344°, 6.094°, ..., all different. So
  a run visits `n_phase * n_sub` = **64 distinct phase values**, and `wheel_wheel.coord_fn`
  keys its jit cache on `float(phase)` (`_COORD_FN_CACHE_MAX = 128`, FIFO), so **all 64
  traces are retained and none is ever evicted**.

  Measured: ~0.4 GB per retained trace. Worker RSS climbed 2.2 → 3.9 GB over three steps
  while the parent sat flat at 4.5 GB, and the run peaked at **18.9 GB and was OOM-killed at
  step 3**. The pool holds all 64 traces however the slots are divided, so **`--workers 2`
  does not help** — it redistributes the same cache and saves only one process baseline.

  **`uniform` was then measured and it holds.** The cache saturates at 8 traces after step
  0, and the run sits FLAT at **12.73 GB anon through step 27** with `memory.events` all
  zero — the 20 GB cap is never approached, rather than approached slowly. That is the
  difference between a bounded phase set and an open one, and it is why this is a
  correctness constraint on the run configuration rather than a tuning knob.

  The jit cache was designed for WARMTH — M7 measured a miss at 0.774 s, which is why slots
  are pinned to workers at all (§ the phase pool above). Nothing sized it against a stencil
  that keeps producing new keys. `uniform` is not a workaround here: it is the scheme
  `make m8bii1`'s end-to-end pooled-vs-serial check already ran on.

  **What it costs is the randomised quadrature**, and that is a real change to how the
  phase average is estimated, not a free win. It is a deliberate trade of variance
  reduction for a run that can physically complete.

### 2. M9 — `lambda_min(K_t)` via LOBPCG, replacing the Euler `buckling` proxy

**This milestone promoted M9.** `buckling` has a gradient of exactly 0.0 and is asserted zero
by the inert-term census (`INERT_EXPECTED = ("buckling",)`). With stress no longer binding, it
is **the only constraint left in the objective that a gradient method cannot act on**. A
diverged tangent is the only real buckling signal the run has today.

**M9 phase 1: all three new tests pass, including contact penalty smoothing.**

### 3. ~~The hub fillet~~ — DONE as geometry, and it turned up a bound the genome cannot honour

**The hub junction exists and all 24 of its corners are filleted** (it was 0 of 12). Full
record in **`HUB_PLAN.md`**; the short version:

`_embed`'s inward step took the *least rotation from the junction tangent* that reached the
hub — a 4.516 mm run that swung the root cap **22.3° out of a 30° sector**, so adjacent spokes
lapped over the hub circle before either reached it and the circle stopped existing. Measured:
at r = 12.71 and r = 12.80 the ring is 360° material; the first void is a 0.16° sliver at
r = 12.8748, which is the 354° notch OCC was refusing. There was no spoke↔hub junction to
fillet. It now **plunges radially** — 1.788 mm, 0.57° — and the 24 corners are back on
r = 12.700, worst wedge 332°.

`fillet_junctions` also stopped abandoning the leftovers: it re-selects the corners that
refused a rung and walks the ladder again for them, so both flanks get a radius. That was
never a hub-only bug — it is why the **rim** shipped 12 of 24.

| junction | found | filleted | families | R worst | Kt model | Kt built | error |
|---|---|---|---|---|---|---|---|
| hub, before | 12 (off-circle) | **0** | — | 0.000 | 1.861 | 3.500 | +88.1% |
| hub, after | **24** | **24** | 12 @ 1.127, 12 @ 0.361 | 0.361 | 1.861 | 3.228 | **+73.4%** |
| rim, before | 24 | 12 | 12 @ 3.000 | 3.000 *(as priced then)* | 1.490 | 1.490 | +0.0% |
| rim, after | 24 | **24** | 12 @ 3.000, 12 @ 0.308 | 0.308 | 1.490 | 3.149 | **+111.4%** |

**The pricing decision changed with it: a junction is priced by its WORST corner.** The old
rule priced the radius that *was* applied, which is how the rim reported +0.0% while twelve of
its corners shipped square. The rim therefore looks worse than it ever has while its geometry
strictly improved; `fillet_families` keeps every per-family radius, so nothing is lost.

**The new finding, and it is a human's decision.** The void between adjacent spokes at the hub
circle is **9.907°, 2.196 mm of arc**, narrowing outward. Two fillets growing into that slot
can each take about half of it — OCC accepted **1.127 mm**, against 1.098 = half the gap. So
**`R_hub` cannot be built above about 1.1 mm on this genome and its bound is 4.0 mm.** The
constraint can price a fillet the part can never build, which is the same class of discrepancy
this milestone existed to remove, one level up. Either the bound comes down, or the constraint
learns the cap, or the spoke count/width changes.

Knock-ons, both bookkeeping: `EMBED_ALLOWANCE_PER_SPOKE_MM2` re-measured 4.27 → **3.03** (the
radial plunge leaves less gusset in the annulus), narrowing the deliberate mesh-vs-STEP AREA
gap −1.93% → **−1.384%**; and the fillets stopped being negligible — 0.29 mm² of cross-section
when 12 of 48 corners were built, **24.28 mm² (0.92%)** now that all 48 are — so the MASS gap
is a different number from the area gap, **−2.277%**, and the difference between them is the
fillet material. Export cost went 14.6 s → 230.5 s; `wheel.step` carries 111 faces against 75
and a 0.162 mm shortest edge. OCC calls it valid, self-intersection clean, no degenerate
edges; no other kernel has looked at it since (Inventor is the consumer now — see §0).

### 4. Minor, known, pre-existing — DONE

**~~`study_stage3.py --quick` exits 1 on S8.~~ FIXED — scoped to `coarse`, not more reps
at `smoke`.** The old failure (cold 6.16 s vs warm 6.32 s, −2.6%) was a `smoke`-tier
noise-floor artifact — the 960-element solve is too small a share of an evaluation
dominated by meshing/dispatch to resolve the warm-start saving at all, so more reps
would have chased a signal too small to measure rather than fixed anything. `run_warm`'s
section-registry entry now always calls it at `DEFAULT_CONFIG` ("coarse"), regardless of
`--quick` — "the gate that counts," per the diagnosis already on file. Re-measured under
`--quick --sections warm`: cold 43.3 s, warm 39.1 s, **saving 9.8%, PASS** (up from the
`smoke`-tier +2.4% quoted before, both comfortably above `GATE_WARM_SAVING = 0.0`, left
untouched). The section's report and printed output now carry which config it ran at,
so this doesn't silently look like a `smoke` number in a `--quick` run. No test asserted
the old behavior, so `make test` is unaffected; **391 passed** afterward.

### 5. §0(a) — the constraint learned the buildable hub fillet cap. DONE

`R_hub`'s box bound is still 4.0 mm. What changed is that the loss now knows what the part
can build, in two places, and the number is computed from the genome rather than remembered
from one export.

**The geometry.** `wheel_wheel.hub_void_deg` measures the empty arc adjacent spoke roots
leave on the hub circle — `SECTOR_DEG` minus the arc one root occupies, from `ring_station`
and an `arctan2`, the same arithmetic as `weld_footprints_deg`. It lives in `wheel_wheel`
because that module is jax-free and therefore importable in `.venv-cad`, which is what would
make "let the exporter consume the cap too" a small change later.

**That part is validated.** `make hubcap`'s `void` section classifies 14400 points on a ring
just outside the hub circle in OCC and measures the empty runs. Analytic against measured,
on three designs: **9.977 / 9.825, 8.931 / 8.800, 13.774 / 13.575** — 0.13 to 0.20° apart,
exactly the residual `hub_void_deg`'s docstring predicts from `_embed` plunging radially
from the centerline endpoint rather than the flank endpoint. Twelve runs every time.

**THE CAP MODEL IS A `min` OF TWO LIMITS, AND THE GATE IS WHY.** It was written as
`0.5 × slot` alone, on the strength of the hub fillet milestone's one recorded export
(void 2.196 mm, OCC built 1.127). Bisecting what OCC will *actually* accept on each of the
24 hub corners falsified that:

| design | void | slot arc | `0.5×arc` | **OCC, bisected** | t0 |
|---|---|---|---|---|---|
| best_solution | 9.977° | 2.2115 | 1.1057 | **1.3000** | 2.4774 |
| elite14 | 8.931° | 1.9795 | 0.9898 | **1.3445** | 2.5536 |
| elite13 | 13.774° | 3.0531 | 1.5265 | **1.3340** | 2.5536 |

The 1.127 on file was a **ladder rung accepted for a whole twelve-edge family**, not the
limit. And the limit does not track the slot: the void spans a **54% range** while the
threshold moves **3.4%**. elite14 and elite13 have *identical* `t0` and thresholds 0.7%
apart across the widest void gap in the set. It tracks **`t0/2`**, to 0.7%.

So `hub_fillet_cap_mm` is now `min(0.5 × slot_arc, 0.52 × t0)`. The slot term is kept, and
kept at an unvalidated 0.5, because it is the only one that knows a *closed* slot admits no
fillet at any radius — as adjacent roots converge the void goes to zero and then negative,
and the thickness term cannot see that at all. Both facts are in the constants' docstrings,
labelled as measurement and as assumption respectively.

**`make hubcap` PASSES, and the split confirms the `min`:**

| design | binding limit | cap | OCC | cap/OCC |
|---|---|---|---|---|
| best_solution | slot | 1.1057 | 1.3030 | 0.849 |
| elite14 | slot | 0.9898 | 1.3384 | 0.740 |
| **elite13** | **thickness** | 1.3279 | 1.3274 | **1.000** |
| **t0 = 2.0** | **thickness** | 1.0400 | 1.0395 | **1.000** |

Where the thickness term binds it predicts what OCC accepts to **0.05%**. Where the slot
term binds — the deliberately unvalidated 0.5 — it is conservative by 15–26%, which is the
harmless direction. That asymmetry is the model's shape, stated honestly rather than tuned
away.

**The thickness law is calibrated on `t0 ∈ [2.0, 2.6]` and is not known outside it.** Five
points there (0.5197 / 0.5254 from the sweep, 0.5247 / 0.5263 / 0.5224 from the designs) sit
in a 1.3% band. The sweep's rows above `t0 ≈ 3` report 0.63–0.94, and they are not evidence
against it: the void has collapsed and gone negative by then (−0.92° at `t0` = 6, −9.51° at
10), adjacent roots have merged, and there is no spoke-to-hub corner left for the number to
be about. Those rows are marked `same_feature: false` and excluded from the fit. In that
regime the slot term is negative and takes the `min` anyway — the cap is right there for a
different reason: no fillet exists at any radius, which is what a negative cap says.
`MIN_WALL_MM` is 2.0 and every design on disk sits at 2.468–2.627, so the calibrated band
covers everything the GA has produced.

**The gate is ONE-SIDED, and that is the model's actual claim.** It began as a two-sided
"within one ladder rung of OCC", written when `0.5 × slot` was believed to *be* the limit. A
`min` of one measured limit and one unvalidated one does not claim to be tight, it claims to
be **safe** — and the way to pass a tightness gate that a safe model fails is to loosen the
model until it passes, which is backwards. So: `cap/OCC ∈ [0.5, 1.01]`. Never promise more
than the part gives; and do not collapse to something vacuously small, since a cap of zero
would sail through a pure one-sided test while destroying every hub fillet in the wheel.

**Both halves of the fix, because either alone leaves half the defect.**
`fillet_cap = soft_barrier(R_hub − cap, 500)` pushes the gene under the slot; and `_kt_hub`
prices `Kt` on `smooth_min(R_hub, cap)` so the stress constraint stops crediting a fillet
that will not exist. `Kt_hub` moved **1.8609 → 2.0766, +11.6%** — the part is sharper than
the gene was asking for, and now the loss says so.

**`junction_kt` is no longer pure 14-vector gene space, and that is the change.** The cap
depends on the eight centerline genes and `t0` through `ring_station`, so `Kt_hub` does too.
It is **differentiated, not frozen** — freezing it and applying the chain rule by hand is
exactly the `stress_scale` failure mode this module already paid for once. Still no mesh and
no solve: it is a sampled curve, jitted and cached on `(cfg.name, span_mm, flanks)` the same
way `t1_vector` is. The consequence worth knowing: **`dKt_hub/dR_hub` is now EXACTLY 0.0**
at the shipped genome, because it sits 2.7 ladder rungs above its cap and buying more `R_hub`
there buys nothing. Below the cap the original physics is back (−1.99).

**`make hubcap` does not read the ladder, and that is a measurement too.** The obvious
criterion — "the largest ladder rung below the cap is what gets built" — is **false**: the
rungs from 1.5598 are 1.5598 / 1.3258 / 1.1269 / 0.9579, the largest under the 1.1057 cap is
0.9579, and OCC takes 1.3000. The rungs straddle the cap and which side they land on is an
accident of where `R_hub` starts. So the driver bisects the threshold where it actually is.

Four sections: `void` and `occ_limit` gate; `t0_sweep` is the **calibration** and is
deliberately not gated (gating a section on the constant it exists to measure is circular);
`sweep` is the picture. The three gated designs are chosen for different reasons — the
shipped genome, elite14 (tightest slot, 0.9898), and elite13, the one design already under
its cap and therefore a negative control rather than a fourth confirmation.

**`t0_sweep` exists because the disk cannot calibrate this.** All 17 designs sit at
`t0` between 2.468 and 2.627 — **6% of a box that runs 2.0 to 10.0** — so measuring more of
them says nothing about whether the thickness law holds at any other thickness. Sweeping
`t0` on one fixed shape does, and it does better than that: a thicker root leaves a
*narrower* void, so the two limits move in opposite directions and their crossover is
observed rather than assumed. That crossover is the entire justification for the `min`.

**15 of the 16 elites are above their cap**, including elites 9 and 10, which are the two
starts §0(b)'s production run begins from. That is what makes this a prerequisite rather
than a tidy-up.

**Verified.** `make test` **406 passed** (396 before). `make hubcap` PASS. Gate 7
(`study_objective.py --quick`) **OVERALL: PASS** — and it is the independent check on the new
term's gradient, because G4 finite-differences every entry of `T1_NAMES` generically:
`fillet_cap` agrees to **3.098e-09** over 11 live genes, which is the assurance that
differentiating through `ring_station` and the `min` is right. G8's inert census is still
`[] ⊆ ("buckling",)`. `make export` was not re-run and must not need to be: nothing on the
CAD side changed, and the manifest is byte-identical.

**A 4-step `coarse` descent from the shipped genome does what it should, and exercises BOTH
branches of the `min` on the way:**

| step | R_hub | cap | R_eff | Kt_hub | `fillet_cap` | t0 | binding |
|---|---|---|---|---|---|---|---|
| 0 | 1.5598 | 1.1057 | 1.1057 | 2.0766 | 103.07 | 2.4774 | slot |
| 2 | 1.4949 | 1.1705 | 1.1705 | 1.9967 | 52.60 | 2.3290 | slot |
| 4 | 1.4722 | 1.1875 | 1.1875 | 1.9748 | 40.53 | 2.2836 | **thickness** |

`R_hub` walks down, the cap walks *up* as `t0` thins, the barrier decays 103 → 41, and the
binding limit crosses from slot to thickness with no discontinuity — `0.52 × 2.2836 = 1.1875`
is the step-4 cap exactly. The gap is closing rather than closed (0.454 → 0.285 mm in four
steps on a decaying `lr`), which is what four steps should look like.

**Two things this does NOT do, stated so nobody expects them.** It does not move
`kt_error_pct` off +73.4%: the manifest's `r_built = 0.361 mm` is set by the SHALLOW
near-cusp corner, an arrival-angle limit the cap does not model. And it does not touch
`stress_concentration_kt`, the exporter, or the GA's numpy `spoke_overlap_penalty` — the
cap is applied by the caller, so the Kt twin stays bit-identical to `wheel_fea`'s and
`test_golden.py` does not move.

**Open, and flagged as the natural next item: `hub_overlap` is now subsumed.** Its chord
proxy wants `t0 + 2·R_hub + 1.3 ≤ 6.574` and reports a **+0.323 mm violation** on a wheel
that has been printed, while the true void at the same design is **+2.20 mm of clearance** —
it assumes the root sits square across the sector, and the shipped root arrives 10.5° from
tangent. `void < 0` already is "adjacent spokes overlap", so the new barrier covers the
collision case too. It was left in place deliberately: it is 47.9% of the T1+T2 value and
98.7% of its gradient norm, it is named in `wheel_stage3`'s deadlock post-mortem, and five
study drivers plus `tests/test_mesh.py` read it as a feasibility filter. Retiring a
48%-of-the-loss term inside a bound fix is how the `stress_scale` problem happened. The
`sweep` section of `make hubcap` prints the head-to-head that would justify it.

### 6. §0(b) — the production descent. DONE, BOTH STARTS. It found a 20% lighter wheel.

`make prod9`, `--start rank:9 --steps 300 --workers 4 --phase-scheme uniform
--fidelity-check-every 0`, `coarse`, 8 phases, seed 0. Record: `stage3_prod_elite9.json`,
genome `stage3_prod_best_elite9.json`. **7051.5 s for 150 objective calls = 47.0 s per
evaluation**, which is S13's 47.6 s ladder entry to within 1.3% — the projection travelled.

**Stopped by the operator at step 149 of 300, on convergence.** Loss improved 240.96 over
steps 0–50, 0.274 over 50–100 and 0.031 over 100–149; the last nine steps moved it 3.8e-03.
Zero rejects, zero abandoned steps, `events` empty.

| | step 0 (elite 9 as scored) | step 149 | |
|---|---|---|---|
| loss | 291.036 | **49.771** | |
| mesh mass | 73.689 g | **58.715 g** | **−20.3%** |
| axle drop, mean | 2.0491 mm | 1.9942 mm | **+2.46% → −0.29%** |
| stress utilisation | 0.5444 | 0.6018 | feasible at ≤ 1.0 |
| `hub_overlap` | 107.736 | **0.0** | |
| `fillet_cap` | 118.184 | **0.0** | |
| `Kt_hub` | 2.0978 | 2.0627 | |
| hub fillet cap | 1.1061 mm | 1.0400 mm | thickness-binding |
| `R_hub` effective | 1.1061 (**at** cap) | 0.9106 (**under** cap) | |
| buckling ratio | 0.0871 | 0.0663 | |
| min scaled Jacobian | 0.8892 | 0.8892 | |

**Every barrier in the objective is exactly 0.0 at the answer.** `x_order`, `hub_overlap`,
`fold`, `arrival`, `fillet`, `fillet_cap`, `buckling`, `min_sj`, `stress`, `phase_ripple` —
all of them. Three terms are left: **mass 48.259 (96.96% of the value, 44.4% of the gradient),
deflection 0.0214 (0.04% of the value, 46.5% of the GRADIENT), smoothness 1.491 (3.0% / 9.2%)**.

That split is the whole result. **Deflection is worth four hundredths of a percent of the loss
and nearly half of its gradient**: it is not paying for anything, it is *holding mass up*. The
descent is riding the deflection constraint down to the lightest wheel that still meets it,
which is exactly the regime §1 said the run would enter — "19.6% of the loss at the shipped
genome against deflection's 61.3%, and that ratio inverts once deflection is met."

**THE FOUR THICKNESS GENES ALL SATURATE AT THE LOWER BOUND, and that is the finding to argue
about.** `final.bound_saturation` is `t0, t1, t2, t3` all `low` at **2.0**, which is
`MIN_WALL_MM`. Minimising mass under a deflection constraint makes the wheel **as thin as it
is allowed to be everywhere**, and buys its stiffness back entirely from the eight centerline
genes. So the reported 58.715 g is not "the lightest wheel in this genome" — it is **the
lightest wheel at the 2.0 mm manufacturing floor**, and the floor, not the physics, is what
set four of fourteen genes. Whether 2.0 mm is the right floor for the actual process is a
human's question, and it is now worth about 20% of the mass. It belongs on the list below.

**§5 earned its place here.** Elite 9 started **above** its buildable hub fillet cap —
`fillet_cap` was 118.18, the second-largest term in the starting loss at 40.6% — and the
descent drove `R_hub` from 1.1061 (pinned at the cap) to 0.9106, comfortably under a cap that
had itself moved to 1.0400 as `t0` thinned. Without §0(a) this run would have spent 300 steps
buying a fillet the part cannot build. `hub_overlap` went 107.74 → 0.0 the same way, which
also answers §5's open question in one direction: that term is **satisfiable**, not binding at
the optimum, so retiring it is still optional rather than urgent.

### Elite 10 ran the full 300 steps, and the answer is a FLAT VALLEY, not a point

`make prod10`, same flags, a clean 300 from `rank:10` — the first attempt had been stopped at
step 83, well short of converged. **13656.6 s / 301 calls = 45.4 s per evaluation**, zero
rejects, zero abandoned steps, `events` empty, `Result=success` at a **13.7 GB** unit peak
against the 20 GB cap and **0 B** swap. Written by `main()`, so unlike elite 9's this genome
needed no reconstruction (`genome_hash eddcfc2`).

| | step 0 (elite 10 as scored) | step 300 | |
|---|---|---|---|
| loss | 308.847 | **49.7376** | |
| mesh mass | 70.937 g | **58.660 g** | **−17.3%** |
| axle drop, mean | 2.0379 mm | 1.9941 mm | **+1.90% → −0.30%** |
| stress utilisation | 0.5444 | 0.5904 | feasible at ≤ 1.0 |
| `hub_overlap` / `fillet_cap` | 159.194 / 87.010 | **0.0 / 0.0** | |
| `R_hub` effective | 1.2251 (**at** cap) | 0.9666 (**under** cap 1.0400) | |

Same shape of answer as elite 9: every barrier exactly 0.0, all four thickness genes pinned
low at 2.0, and the surviving split **mass 96.94% of the value / 44.0% of the gradient,
deflection 0.04% / 46.7%, smoothness 3.0% / 9.3%**. Converged hard — the last 50 steps moved
the loss 7e-04 and `|grad|` sat at 94.4 from step 100 onward.

**Now the comparison, and it is the point of running a multi-start at all:**

| | elite 9 (step 149) | elite 10 (step 300) | gap |
|---|---|---|---|
| loss | 49.7706 | **49.7376** | −0.033, **0.066%** |
| mesh mass | 58.7145 g | **58.6604 g** | −0.054 g, **0.092%** |
| axle drop, mean | 1.99415 mm | 1.99408 mm | 0.004% |
| stress utilisation | 0.6018 | 0.5904 | |
| `Kt_hub` | 2.0627 | 2.0223 | |

**Two starts, same mass to 0.09% — and GENUINELY DIFFERENT GEOMETRY.** This is not one basin
reached twice. The centerline genes disagree by far more than the objective does:

```
cy4   17.131 vs 15.312   -1.819 mm   10.6%
cx3   24.328 vs 26.011   +1.683 mm    6.9%
R_rim  2.339 vs  2.749   +0.411 mm   17.6%
cy2   23.586 vs 24.580   +0.994 mm    4.2%
t0..t3   2.000 vs 2.000       0.000    0.0%   <- both on the floor
```

So the optimum is a **valley, not a well**: at the 2.0 mm wall floor, mass is essentially the
spoke path length, deflection is the binding constraint, and there is a **family** of
centerlines with the same length that meet it. The 14-gene space is *underdetermined* by
(mass, deflection) — which is exactly why two starts 105 deflection points apart in Stage 2
land 0.09% apart in mass with 1.8 mm of daylight between their spokes.

**What that means for the multi-start: it was worth running, and it says more starts are
not.** A third start would be expected to find a third point in the same valley at the same
mass. The remaining freedom is not something the mass objective can spend — it is free
capacity to satisfy something the loss is not currently asking for (buckling margin,
manufacturability, fatigue at the hub). That is a better argument for M9 than anything in §2.

**The honest caveat: this is not apples to apples, and it slightly favours elite 10.** Elite 9
stopped at 149 while elite 10 ran to 300. At the SAME step the ordering is already elite 10's
— its step-150 loss is 49.7613 against elite 9's 49.7706 at 149 — and it then improved a
further 0.024 over steps 150–300. Extrapolating elite 9's remaining 150 steps at elite 10's
rate puts it near 49.747, i.e. **~0.01 of loss and ~0.02 g behind**, not ahead. The
conclusion is unchanged either way, but the 0.09% is an upper bound on the gap rather than a
measurement of it. Closing it properly costs one more 3.8 h `make prod9`, and nothing on this
list depends on the answer.

**Both records carry `workers`, `cpu_count` and `phase_scheme` in `settings`**, so the
47.0 s can be read on another machine. Memory held flat under a 20 GB `systemd-run` cap with
no `memory.events` and no oomd kill, which is the `uniform` + fidelity-off configuration §1
argues for and not a coincidence.

**One artifact correction.** `stage3_prod_best_elite9.json` is reconstructed by hand rather
than written by `main()`, because the run was stopped rather than completed — that is stated
inside the file. An earlier reconstruction took **step 148** (loss 49.771005); the record's
argmin over all 150 rows is **step 149** (loss 49.770621) and the record's own `best` block
already carried it with the full metric set. The file now carries step 149. The 3.8e-04
difference moves no gene past the fourth decimal; it is corrected because a file named `best`
should be the argmin, not because the design changed.

---

## The decision that is a human's

Unchanged in list — rim-band genes / revisit the targets / accept a Pareto point / change
material — but **the premise moved.** Every one of those was blocked on "we need a stress
number first". The number exists now, and it says the current 14-gene space already contains
designs meeting both targets.

Adding rim-band genes to *reach* feasibility is no longer justified. Adding them to reduce
mass, or to buy margin, is a different argument and needs to be made on its own terms.

**And §6 added one to the list, with a number attached: `MIN_WALL_MM = 2.0`.** The production
descent drives **all four** thickness genes onto that floor and leaves them there, so the
2.0 mm wall is what sets 4 of the 14 genes at the answer — not the FEA, not the deflection
target, not the stress constraint. Every gram below 58.715 is on the other side of it. This
is the first item on this list that is no longer a preference: it is a manufacturing
parameter that is now provably binding, and it is worth asking the process what it can
actually hold before asking the optimizer for anything else.

---

## Where gate 7 no longer helps, and what replaced it

`QUICK_GENES` now includes 12 and 13 (`R_hub`, `R_rim`) so `dKt/dg` is finite-differenced.
**It does not currently test that.** With utilisation at 0.375/0.300 the `soft_barrier` is
flat, so `stress` and `d_stress` are exactly zero and neither gene reaches the loss through
`Kt`, and `R_rim`'s row is `0 == 0`.

**CORRECTION, measured: `R_hub`'s +645.8 adjoint is `hub_overlap`'s, not the `fillet`
barrier's.** This paragraph said `fillet` from M8b-i.6 step 2 until §0(a) checked it. At the
shipped genome the fillet margins are `[+4.647, +0.125]`, both feasible, so that barrier is
flat and `d(fillet)/dR_hub` is exactly 0.0. Every unit of the 645.8 comes from
`hub_overlap`, whose chord proxy is violated by +0.323 mm there. §0(a) added a second live
term, `fillet_cap` at +454.0.

The product rule is tested by **`test_the_stress_gradient_obeys_the_product_rule`**
(`tests/test_objective.py`), which monkeypatches `ALLOWABLE_STRESS_MPA` down to 2.0 to force
the barrier onto its quadratic branch, then FDs genes 8, 11 and 13. **That test, not gate 7,
is what says `dKt*agg + Kt*dagg` is right.** Gene 12 was dropped from that list by §0(a):
`R_hub` is now priced through the buildable cap and the shipped genome sits above its cap,
so `dKt_hub/dR_hub` is exactly zero and its row there asserted `0 == 0`. What it used to
check moved to two solve-free tests that can afford to be far tighter —
`test_R_eff_is_exactly_the_cap_more_than_one_rung_above_it` and
`test_the_cap_gradient_matches_a_finite_difference`. Gene 13 stays in `QUICK_GENES` because
it costs nothing and becomes a live check the moment a design is stress-binding.

---

## How to run any of this

```bash
.venv-opt/bin/python studies/study_objective.py --quick   # gate 7 fast path, ~9 min
.venv-opt/bin/python studies/study_objective.py           # the full M8a gate, > 50 min
.venv-opt/bin/python studies/study_stage3.py --quick      # wiring check, ~13 min, see S8 note
.venv-opt/bin/python studies/study_stage3.py              # the M8b-i gate, S1-S10, ~2 h 45 m
make m8bi5                                                # S11 + S12, ~2 h 31 m
make m8bi6                                                # the p sweep, ~14 min
make m8bii1                                               # S13, the phase pool, ~30 min
make hubcap                                               # the hub-fillet cap vs OCC, ~10 min
make prod9 / make prod10                                  # §0(b), one start each, SEQUENTIAL
make test                                                 # 406 tests, ~22 min
make export                                               # rebuild wheel.step, ~4 min
make studies                                              # all gates; NOT m8bi5/m8bi6/m8bii1
```

**`--workers` runs the phase loop across processes.** `0` (the default) is serial, `-1`
sizes the pool to `min(n_phase, cpu_count)`, `N` is literal and is the only memory cap:

```bash
.venv-opt/bin/python src/wheel_stage3.py --start rank:9 --steps 300 --workers -1
```

The run record carries `workers` and `cpu_count` in its `settings`, because a wall-clock
number without them cannot be read on another machine.

**On a memory-bound host, launch a production run CAPPED and DETACHED.** `systemd-oomd`
kills the whole `user@1000.service` slice at 50% memory pressure for 20 s, which drops the
user to the login screen and takes every terminal and run in that slice with it — the
kernel OOM killer is not involved, so `dmesg` is empty and `journalctl -u systemd-oomd` is
where the evidence is. A cgroup cap inverts that: the kernel kills the run at its own
limit instead.

```bash
systemd-run --user --unit=wheel-prod9 -p MemoryMax=20G --collect \
    --working-directory=$PWD /usr/bin/make prod9
```

`systemctl --user show wheel-prod9.service -p MemoryCurrent -p ActiveState` reads its
state, and the unit's journal records the peak and the verdict
(`Failed with result 'oom-kill'`, `18.9G memory peak`). It also survives the terminal
closing, which a plain child process does not.

**The `prod` targets run the interpreter with `-u`, and that is not cosmetic.** Python
block-buffers stdout when it is not a TTY, and under `systemd-run` it is a journal socket —
so without `-u` a detached descent emits **nothing** to `journalctl` until the buffer fills
or the process exits. Measured on the elite-10 re-run, which predates the flag: 162 steps
in, **zero** `[step ...]` lines in the unit's journal. That makes `-f` useless for progress
and hides a traceback until exit. On a run that predates the flag, progress has to be read
off the clock instead — `ExecMainStartTimestamp` against ~180 s for step 0 (tracing) plus
~47 s per step thereafter — and `MemoryCurrent` is the only live health signal.

Run a study driver directly and it needs `src/` on the path:
`PYTHONPATH=src .venv-opt/bin/python studies/study_stage3.py`. The Makefile exports it, so
anything driven by `make` is already covered — including the CAD hand-off, which
`src/wheel_fea.py` spawns into `.venv-cad` itself.

`--sections` selects and orders sections. `--ladder-p` takes any comma-separated exponents,
dedupes them, and **costs no extra solve** — every exponent is read off the displacement field
the adjoint already converged. `--ladder-configs smoke,coarse,medium,fine` adds a fourth rung;
`fine` is 261k dof and has never been run, so each row is wrapped in its own `try`.

**`make m8bi6` overwrites `studies/study_stage3_pnorm.json`.** Back it up before re-running if you
need to diff against it — its `pnorm_by_p` block is the step-1 evidence and must stay
reproducible.

---

## Repo layout

```
best_solution.json  stage2_elites.json     the provenance chain, read by BOTH envs
poster_summary.jpg                         written beside the genome it describes
src/        the modules — imported flat (`import wheel_fea as W`)
            project_paths.py  ROOT/SRC/STUDIES/EXPORT, stdlib only so the CAD env
                              can import it through wheel_fea
            wheel_pool.py     the parent half of the phase pool.  stdlib+numpy, NO jax
                              (a test asserts it), so sizing a pool or reading
                              PINNED_ENV costs nothing
            wheel_pool_worker.py  the child.  pins threads BEFORE its first import,
                              which is the whole reason it is a separate process
studies/    the 10 study drivers AND their .json/.jpg output, together
export/     what the CadQuery env produces: wheel.step, wheel_nofillet.step,
            wheel_step_manifest.json
tests/      conftest.py at the ROOT puts src/ on sys.path and into PYTHONPATH
```

**Imports were not rewritten.** `src/` reaches the interpreter three ways —
`pyproject.toml`'s `pythonpath` for pytest, `export PYTHONPATH` in the Makefile, and the
root `conftest.py` (which also seeds `os.environ` so the three subprocess-spawning tests
behave the same under bare `pytest` as under `make test`). A package with `__init__.py`
was rejected: `tests/test_import_hygiene.py` imports `wheel_fea` in a **jax-free**
interpreter, and an `__init__` importing jax-dependent siblings would break the CAD env.

A study driver's own `HERE` is `studies/`, so `os.path.join(HERE, args.out)` still puts
output beside the driver and was left alone. Only the INPUTS moved to `PP.ROOT` / `PP.EXPORT`.

---

## Artifacts

`study_stage3.json` / `.jpg` are M8b-i's record and `study_stage3_m8bi5.*` are M8b-i.5's; both
describe those runs and are deliberately left unedited rather than corrected after the fact.
`study_stage3_pnorm.*` were regenerated by step 2 — the `pnorm_by_p` leaves are bit-identical
to step 1's, and the top-level rows now carry the new constraint plus a `util_kt` column.

`study_stage3_pool.*` are S13's, and **half of that file describes this machine rather than
this commit** — the seconds and the projected hours are 16-core numbers. What travels is
`identical_values`, `worst_grad_rel`, and the efficiency column. Re-run `make m8bii1` on a
different host and expect different hours and a different ladder; expect the same verdict.

`study_hub_cap.*` are §5's, and unlike the above they describe **OCC's behaviour on this
shape** rather than this machine or this commit. They are the calibration behind
`HUB_CAP_THICKNESS_SHARE` and the falsification of the slot-only model, so the
`occ_limit` and `t0_sweep` blocks are the evidence and should stay reproducible. The wall
clock (583 s for three designs plus the eight-point sweep) is dominated by OCC fillet probes
— roughly 220 of them per design — and does travel between machines only loosely.

`stage3_prod_elite9.json` / `stage3_prod_elite10.json` are §6's, and they are the only
artifacts here that are a **SEARCH RESULT** rather than a gate, a calibration or a machine
description — nothing in them passes or fails. They sit at the repo root beside
`best_solution.json` and `stage2_elites.json` because that is where the provenance chain
lives and they are the next link in it. `stage3_prod_best_elite9.json` is the genome, and
it is the one file in the tree written by hand rather than by the code that names it — §6
says why and the file says so itself. `stage3_prod_best_elite10.json` (`eddcfc2`) was
written by `main()` at the end of a completed 300-step run and needs no such note; **it is
the better of the two answers and the one to carry forward.**

The elite-9 record stops at step 149 and the elite-10 record runs to 300, so **the two are
not directly comparable step for step** — §6 states the size of that asymmetry and which way
it leans. Neither file is a gate: re-running either produces a different wall clock and,
because §1's S13 note applies, a trajectory that may differ in its last bits. What should
reproduce is the *shape* of the answer — every barrier zero, four thicknesses on the floor,
mass ~97% of the loss against deflection ~46% of the gradient.
