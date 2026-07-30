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

### 1. M8b-ii — make the optimizer runnable at scale

Feasible points exist; the optimizer could not search for better ones in reasonable time.
The phase batch is now parallel and that is measured; the other three bullets are open.

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

- **Multi-fidelity checkpoints.** `medium` is 2.8× `coarse`, not the 4× budgeted (243 s vs
  87 s). Take that pair from the **elite-1** ladder, not the shipped genome's — the shipped
  genome runs first and its `coarse` rung carries the `coord_fn` jit trace.
- **Jit `t1_vector`** (`wheel_objective.py`) — 1.06 s of eager dispatch per call, measured by
  S10. **Worth more now than that number suggests, and S13 says why.** T1 and T2 run in the
  PARENT on both paths, so they are the whole of Amdahl's serial fraction once the phases
  are parallel: efficiency falls 0.91 → 0.73 → **0.49** from 2 to 8 workers, and at 8 the
  parallel part is ~26 s against ~9 s of serial remainder. 0.73% of a serial evaluation is
  a much larger share of a pooled one.

  `flanks` is a tuple of at most four `±1.0` floats — a DISCRETE structure with 16 possible
  values — so it belongs in a `_T1_CACHE` key beside `cfg.name`/`weights`/`span_mm`, in the
  `coord_fn` idiom, and the cache will hit on essentially every step. One jitted function
  returning `(value, jacobian)` beats two calls.
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

### 2. M9 — `lambda_min(K_t)` via LOBPCG, replacing the Euler `buckling` proxy

**This milestone promoted M9.** `buckling` has a gradient of exactly 0.0 and is asserted zero
by the inert-term census (`INERT_EXPECTED = ("buckling",)`). With stress no longer binding, it
is **the only constraint left in the objective that a gradient method cannot act on**. A
diverged tangent is the only real buckling signal the run has today.

### 3. The hub fillet — a geometry milestone this change made expensive

The constraint prices `Kt(R_hub, t0) = 1.861` and lets `R_hub` carry a live gradient. The part
builds `Kt = 3.5`: **0 of 12 hub edges filleted**, because `_embed` laps adjacent spokes over
the hub circle into a 354° notch OCC refuses at every radius down to `MIN_CURVATURE_RADIUS_MM`.
As-built utilisation is therefore **~1.88× whatever Stage 3 reports** (`kt_built/kt_modeled`).

At util 0.41 that still lands under 1.0, so it does not threaten feasibility — but it is a real
gap between the modelled and printed part, and it is now load-bearing rather than cosmetic.

**The decision already taken, deliberately:** price `r_requested`, gate on `kt_error_pct`.
Clamping `R_hub` to what OCC can build would pin `Kt_hub` at the constant 3.5 and kill the
gradient this whole change exists to create. `tests/test_export_contract.py` pins the
discrepancy in both directions and states the multiplier in its failure text.

The fix is in `_embed` / `fillet_junctions`, not in the constraint.

### 4. Minor, known, pre-existing

**`study_stage3.py --quick` exits 1 on S8.** Cold 6.16 s vs warm 6.32 s, −2.6%. A `smoke`-tier
artifact: the 960-element solve is a small share of an evaluation dominated by meshing and
dispatch, and cold always runs first within each rep. At `coarse` — the gate that counts — S8
passes at +2.4%. No test asserts it, so `make test` is unaffected. Fix by giving `run_warm`
more reps at `smoke` or by scoping S8 to `coarse`. **Do not "fix" it by relaxing
`GATE_WARM_SAVING`.**

---

## The decision that is a human's

Unchanged in list — rim-band genes / revisit the targets / accept a Pareto point / change
material — but **the premise moved.** Every one of those was blocked on "we need a stress
number first". The number exists now, and it says the current 14-gene space already contains
designs meeting both targets.

Adding rim-band genes to *reach* feasibility is no longer justified. Adding them to reduce
mass, or to buy margin, is a different argument and needs to be made on its own terms.

---

## Where gate 7 no longer helps, and what replaced it

`QUICK_GENES` now includes 12 and 13 (`R_hub`, `R_rim`) so `dKt/dg` is finite-differenced.
**It does not currently test that.** With utilisation at 0.375/0.300 the `soft_barrier` is
flat, so `stress` and `d_stress` are exactly zero and neither gene reaches the loss through
`Kt` — `R_hub`'s +645.8 adjoint comes from the geometric `fillet` barrier, and `R_rim`'s row
is `0 == 0`.

The product rule is tested by **`test_the_stress_gradient_obeys_the_product_rule`**
(`tests/test_objective.py`), which monkeypatches `ALLOWABLE_STRESS_MPA` down to 2.0 to force
the barrier onto its quadratic branch, then FDs genes 8, 11, 12, 13. **That test, not gate 7,
is what says `dKt*agg + Kt*dagg` is right.** Genes 12/13 stay in `QUICK_GENES` because they
cost nothing and become live checks the moment a design is stress-binding.

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
make test                                                 # 383 tests, ~12 min
make studies                                              # all gates; NOT m8bi5/m8bi6/m8bii1
```

**`--workers` runs the phase loop across processes.** `0` (the default) is serial, `-1`
sizes the pool to `min(n_phase, cpu_count)`, `N` is literal and is the only memory cap:

```bash
.venv-opt/bin/python src/wheel_stage3.py --start rank:9 --steps 300 --workers -1
```

The run record carries `workers` and `cpu_count` in its `settings`, because a wall-clock
number without them cannot be read on another machine.

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
