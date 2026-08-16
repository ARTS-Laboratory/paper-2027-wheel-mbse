# SVK_PLAN.md — should Stage 3 descend on a linear solve at all?

> **Ignore version control entirely. Do not commit, branch, stage, revert or otherwise
> touch git — it is not part of this project's workflow and nothing here depends on it.**

> **THIS FILE IS WRITTEN TO BE READ ONE STEP AT A TIME, BY A SESSION THAT HAS NEVER SEEN
> THE OTHERS.** Every step carries **Why / Read first / Do / Gate / Record**. Do the step,
> write its numbers into its own **Record** block, tick the status table, stop. The next
> session starts from this file and PLAN.md and needs nothing else.

**The parent item is PLAN.md §14 item 4a.** Read that section before Step 1; everything
below assumes it. This file is the milestone; PLAN.md gets a §15 only at Step 7.

---

## Status

| step | what | status |
|---|---|---|
| 0 | baseline `make test` | **DONE** — 431 passed / 2 failed, both known (2026-08-08) |
| 1 | prove the adjoint under SVK | **DONE — ALL 10 GATES PASS UNDER SVK, unmodified** (2026-08-08). Also found: `studies/study_gradient.json` is stale |
| 2 | measure what SVK costs per evaluation | **DONE — 1.36x time, 1.05x memory** (2026-08-09). 5.3 h projected per 300-step start; MemoryMax 16G |
| 3 | re-score the shipped genome + candidates under SVK | **DONE — SHIPPED GENOME IS FEASIBLE UNDER SVK**, util 0.875 vs §14's estimated ~0.91 (2026-08-09). Control reproduces §14 to 5 s.f. But the ranking of every design INVERTS |
| 4 | THE DECISION | **DONE — OPTION A, descend under SVK** (2026-08-09) |
| 5 | the SVK descent | **DONE — BOTH RUNS PASS ALL THREE GATE CLAUSES** (2026-08-09). Candidate `ae7092c`: 37.451 g (-1.74 g vs shipped), deflection -0.043%, every barrier 0.0, util 0.899. Also found: `R_rim` and `R_hub` are DEAD GENES (zero gradient in 602 of 604 steps) |
| 6 | export, check, promote | **DONE — NOTHING PROMOTED, AND THAT IS THE RESULT** (2026-08-10). `bc77614` clears every FEA gate at `medium` (defl −0.041%, 37.414 g, all barriers 0.0) then fails the export: `kt_error_pct` +11.9% at the hub, as-built utilisation **1.046**. Control `350f4c7` is 0.0% — a regression this arc introduced. `best_solution.json` UNCHANGED |
| 7 | write the record into PLAN.md | **DONE — §15 WRITTEN, BANNER AMENDED, §0 BULLET ADDED** (2026-08-10) |

**CLOSED 2026-08-10, all eight steps done, nothing promoted — and that was the result.** Two of
the five successors this arc parked have since been built (buildability → PLAN §16, stress
margin → PLAN §18), and they are what let the first promotion since go through on 2026-08-13
(PLAN §19). See the annotated **Parked** section at the end of this file for where each one
landed. Anything after 2026-08-10 lives in `PLAN.md` §16–§19 and `BUILD_PLAN.md`.

---

## The problem, in one page

`wheel_contact_problem` defaults to **`kinematics="linear"`** (`src/wheel_fem.py:1693`),
and nothing in the Stage-3 path has ever overridden it. So **every headline number for the
shipped genome `350f4c7` was computed under linear kinematics.** PLAN.md §14 item 4a
measured what that costs, at `coarse`, on the full load ladder:

| load | linear mm | SVK mm | rel_diff | | GA/beam `36aed36` rel_diff |
|---|---|---|---|---|---|
| 0.01x | 0.019530 | 0.019570 | +0.208% | | +0.038% |
| 0.25x | 0.488241 | 0.514371 | +5.352% | | +0.963% |
| 0.50x | 0.976483 | 1.084109 | +11.022% | | +1.943% |
| **1.00x** | **1.952966** | **2.408898** | **+23.346%** | | **+3.953%** |
| 2.00x | 3.905931 | 5.939932 | +52.075% | | +8.193% |
| 3.00x | 5.858897 | 10.909429 | +86.203% | | +12.758% |

The GA/beam column reproduces M5's recorded numbers exactly (0.038% / 3.95% / 12.8%),
which is what makes the other column trustworthy.

**Two consequences, and they are the whole motivation:**

- **Deflection.** `TARGET_DEFLECTION_MM = 2.0` (`src/wheel_fea.py:156`) and the objective
  wants to hit it *exactly* — a two-sided squared relative error at weight 2500
  (`src/wheel_objective.py:250, 978`). Linear says 1.953 mm, essentially on target. SVK
  says **2.409 mm, 20% over**. The design was tuned to a target it does not hit.
- **Stress utilisation.** The 0.799 headline is a linear-field number. Plain-spoke p99
  goes 17.274 → 19.746 MPa under SVK, **+14.3%**, which puts utilisation **on the order of
  0.91** against an allowable of 1.0. **That is an estimate, not a measurement** — it
  scales the reported figure by the p99 ratio, while the constraint aggregates
  `Kt * sigma_nominal(p=4) / ALLOWABLE_STRESS_MPA` (`src/wheel_objective.py:1025`) and a
  p-norm at p=30 on the `medium` rung. Step 3 replaces it with the number the constraint
  actually sees.

The GNL gate (`GATE_SMALL_LOAD_REL = 1e-3`, `studies/study_gnl.py:70`) is the tripwire
that found this, and §14 decided **it stands**: relaxing it would silence the only
automatic warning that linear kinematics no longer describes this part.

### Two things established by reading the code, which set the shape of this plan

**1. The plumbing already exists. Only the CLI flag is missing.**
`kinematics` rides `**problem_kw` the entire way down:

```
wheel_stage3.Evaluator.__init__(**problem_kw)        src/wheel_stage3.py:319
  -> self.problem_kw                                          :325
  -> WO.objective(..., **self.problem_kw)                      :356
       wheel_objective.objective(**problem_kw)       src/wheel_objective.py:1094
         -> t3_terms(..., **problem_kw)                              :1154
              t3_terms(**problem_kw)                                  :873
                -> WA.service_qoi_value_and_grad(..., **problem_kw)   :947
                   wheel_adjoint.service_qoi_value_and_grad  src/wheel_adjoint.py:640,653
                     -> fem.solve_wheel_contact(**problem_kw)
                     -> fem.solve_wheel_contact_at(**problem_kw)
                     -> fem.wheel_contact_problem(**problem_kw)
```

The phase pool forwards it too: `t3_terms` puts `problem_kw` in the task dict
(`src/wheel_objective.py:940`) and `src/wheel_pool_worker.py:66` splats it. The adjoint's
own kernels already dispatch on `prob.nonlinear` (`src/wheel_adjoint.py:161, 190, 400`),
as does the von Mises recovery (`:249, 301`). **What is missing is `--kinematics` on
`wheel_stage3.py`, and `main()` forwarding it** — `main()` currently calls `descend(...)`
with no `problem_kw` at all (`src/wheel_stage3.py:918`).

**2. The adjoint has never been validated under SVK.** Every section of
`studies/study_gradient.py` builds its problem at the default —
`fem.wheel_contact_problem(mesh, indentation_mm=indentation_mm)` at line 187 in
`run_unrolled`, and `_service_indentation` at line 160 likewise. M7's gate is a
linear-kinematics gate. **Turning SVK on today would spend hours descending on a gradient
nothing has checked**, which is the exact failure mode this repo has spent three
milestones removing. Step 1 is therefore a hard prerequisite for Steps 5–7.

### And one thing that makes the cost question genuinely open

**Contact already forces the Newton path under linear kinematics.** `wheel_fem.solve`
routes on `prob.nonlinear or prob.contact is not None` (`src/wheel_fem.py:1274`), because
a penalty contact law is a nonlinear boundary condition whichever kinematics is in force —
which points are touching is itself the unknown. So SVK does not *add* Newton to this
problem. It adds iterations and a more expensive element kernel. **The penalty could be
1.4× or 4×. Nobody has measured it.** Step 2 does, and Step 5 is not launched until it
has.

### Standing rules for every step below

- **Measure before touching a threshold.** PLAN.md §14's governing rule. If an SVK run
  fails a gate, that is the finding — do not loosen the gate to fit it. `make m9`'s
  OVERALL: FAIL was the most useful result in the file.
- **Measure both genomes.** Anything claimed about `350f4c7` gets the same measurement on
  `best_solution_ga_beam.json` (`36aed36`). That comparison killed three of §14's eight
  "the new design broke it" readings.
- **Linear stays the default everywhere.** Every committed artifact must still reproduce
  bit-for-bit with no flag passed. A new default is a re-baselining, and this repo does
  not re-baseline silently.
- **`--phase-scheme uniform` is a correctness setting, not a preference.** See PLAN.md §1:
  `rqmc` retains 64 jit traces at ~0.4 GB each and OOM-killed a production run at step 3.
- **Drive long runs through `make`, and cap them.** The five pinned env vars at
  `Makefile:29-34` are what make the adjoint bit-reproducible, and they reach the parent
  only via make. On this box (24 cores / 61 GB) a descent still gets
  `systemd-run --user -p MemoryMax=... --collect`, because that inverts the failure: the
  kernel kills the run at its own cap instead of `systemd-oomd` killing the whole
  `user@1000.service` slice and dropping the desktop to the login screen.

---

## Step 0 — Baseline the gate

**Why.** PLAN.md §14 closes with: *"the count should be 2 failed / 431 passed; that
arithmetic has NOT been confirmed by a full run."* Steps 1, 2 and 6 all get read as a diff
against this number, so it has to be measured rather than inferred. It is also the cheapest
possible check that the tree is where the last session left it.

**Read first.** PLAN.md §14, "The gate after all of it" (the last subsection).

**Do.**

```
make test 2>&1 | tee /tmp/svk_step0_test.log
```

~25 min. Nothing else meaningful should be running on the box.

**Gate.** Exactly **two** failures, and both of them these:

- `tests/test_gnl.py::test_the_correction_enters_at_first_order_in_the_load` — the
  pre-registered GNL gate, deliberately red (§14 item 4a).
- `tests/test_wheel_fea.py::test_the_rim_band_holds_a_large_minority_of_the_compliance` —
  hub compliance share 0.0321 against `< 0.03`, deliberately red (§14 item 4b).

**Any third failure is a finding and stops the arc here.** Diagnose it before proceeding —
it means something moved that nobody recorded, and every measurement below would inherit
it.

**Record.** Passed/failed counts, the wall clock, and the exact node id of every red.

```
STEP 0 RECORD — RUN 2026-08-08.  GATE: PASS.
  make test:            431 passed / 2 failed in 1374.52 s (22:54)
  matches §14's 431/2:  YES, exactly
  box:                  24 cores / 61 GB, load 0.38, 55 GB free, nothing else running
  reds, both expected:
    tests/test_gnl.py::test_the_correction_enters_at_first_order_in_the_load
    tests/test_wheel_fea.py::test_the_rim_band_holds_a_large_minority_of_the_compliance
```

**§14's arithmetic was right.** It predicted 431/2 without ever running the full suite —
423 passed at §13, plus items 1/2/3 closing six failures, plus the test
`test_stress_recovery_follows_the_solves_kinematics` that item 4a added, plus the extra
test from splitting `test_area_converges_second_order`. That now has a measurement behind
it instead of a sum.

**Both reds reproduce their recorded values to the digit**, which is worth more than the
count — it says the tree is exactly where §14 left it and no measurement below inherits a
drift:

| red | measured now | §14 recorded | gate |
|---|---|---|---|
| GNL, `small_load_rel_diff` | **0.0020499** | 0.2050% at `smoke` | `< 1e-3` |
| hub compliance share | **0.032076694850181206** | 0.0321 | `< 0.03` |

Two details from the failure output that matter to Step 1, both of them encouraging:

- **The GNL test's first assertion passes.** `fitted_exponent` = **1.0343**, inside
  `0.7 < e < 1.4`. The correction still enters at first order in the load; what moved is
  the coefficient. The test's own docstring says so, and it is the reason §14 concluded
  "the SVK path is behaving" — the SVK *solve* is not in question, only whether Stage 3
  should be using it.
- **The measured 0.20499% is the `smoke` rung**, matching §14's sweep (`smoke` 0.2050% /
  `coarse` 0.2081% / `medium` 0.2089%) rather than one of the finer ones. So this red is a
  cheap tripwire, not an expensive one — useful to know before Step 1 starts adding SVK
  runs to the gate.

*(Aside, not a repo fact: `make test 2>&1 | tee ...` reports `tee`'s exit status, so the
pipeline exits 0 while make itself returns Error 1. Read the summary line, not `$?`.)*

---

## Step 1 — Prove the adjoint under SVK. **THE PREREQUISITE.**

**Why.** M7's gate is what licenses Stage 3 to believe its gradient, and it has only ever
been run under linear kinematics. Under SVK the element kernels change
(`fem._element_kernels(order, nonlinear)`), the tangent is state-dependent, and the
`d_dindentation` quotient the secant depends on is a different operator. None of that is
*expected* to break — the adjoint is written generically against `prob.nonlinear` — but
"expected" is exactly the word `stress_scale` was justified with for two milestones. This
step costs an hour and removes the possibility that everything after it is measuring a
broken derivative.

**Read first.**

- `studies/study_gradient.py` — the header, `GATE_*` at lines 125–138, and `main()` at
  line 966.
- `run_unrolled` at line 168. **This is the section that matters.** It unrolls the Newton
  loop on the `tiny` config and differentiates it with `jax.grad`, so it contains no
  finite difference anywhere and its tolerance (`GATE_UNROLLED_REL = 1e-8`) is set by
  linear algebra rather than by step size. Its docstring: *"this is the only check in the
  file whose tolerance is set by linear algebra... which is why it is first and why it is
  the one to fix before reading any other number in this report."*
- `src/wheel_adjoint.py:161, 190, 400` — the three places `prob.nonlinear` already selects
  the kernel, which is why this is a kwarg change and not a derivation.

**Do.**

1. Add `--kinematics` to `studies/study_gradient.py`:

   ```python
   ap.add_argument("--kinematics", choices=("linear", "svk"), default="linear",
                   help="...")
   ```

   **Default `linear`, and that is load-bearing** — `study_gradient.json` in the tree is a
   committed artifact and must still reproduce with no flag.

2. Thread it into every section that builds a problem or a solve. **The complete list**,
   from `grep -n "wheel_contact_problem\|solve_wheel_contact\|solve_and_grad"
   studies/study_gradient.py` — 15 call sites, every one of which currently takes the
   linear default:

   | line | function | call |
   |---|---|---|
   | 160 | `_service_indentation` | `solve_wheel_contact` |
   | 182, 185 | `run_unrolled` (G1) | `solve_wheel_contact`, `wheel_contact_problem` |
   | 271 | `run_identities` (G2/G3) | `wheel_contact_problem` |
   | 333, 338 | `run_plateau` (G4) | `solve_and_grad`, `solve_wheel_contact_at` |
   | 438, 445 | `run_directional` (G5) | `solve_and_grad`, `solve_wheel_contact_at` |
   | 508, 515 | `run_dense_sweep` (G6) | `solve_and_grad`, `solve_wheel_contact_at` |
   | 569 | `run_phase_smoothness` (G7) | `solve_wheel_contact_at` |
   | 696 | `run_axle_drop` (G9) | `solve_wheel_contact` |
   | 749, 755, 758 | `run_cost` | `solve_and_grad` ×3 |

   `WA.solve_and_grad` takes `**problem_kw` and passes it to both the solve and the
   problem rebuild (`src/wheel_adjoint.py:443, 451`), so those sites need the kwarg but no
   restructuring. The sections should take `kinematics` as a plain keyword argument
   threaded from `main()`, **not** as module state — a module-level default is how a run
   gets misattributed to the wrong kinematics, which is the mistake PLAN.md flags for
   `MIN_WALL_MM`.

   Note `run_cost` (line 729) is measurement-only and has no gate, but it is the section
   that reports gradient-cost-over-forward-solve — which is **exactly the ratio Step 2
   needs**, so run it under SVK too rather than skipping it.

3. Record the kinematics in `rep["settings"]` so a report cannot be misattributed — the
   same reason PLAN.md gives for the `MIN_WALL_MM` sweep writing its floor into each arm.

4. Run `--quick` first (minutes), then the full gate:

   ```
   .venv-opt/bin/python studies/study_gradient.py --kinematics svk --quick \
       --out study_gradient_svk_quick.json
   .venv-opt/bin/python studies/study_gradient.py --kinematics svk \
       --out study_gradient_svk.json
   ```

   `--out` keeps `study_gradient.json` untouched.

5. **Then re-run the default** and confirm it is bit-identical to the committed
   `study_gradient.json` on every non-timing leaf. This is the proof that the plumbing is
   inert when unused — the same check `make m8bi6` ran after the `XLA_FLAGS` pin ("2038
   non-timing leaves, 0 differ").

**Gate.** The **unmodified** linear thresholds, all of them:

| id | gate | value |
|---|---|---|
| G1 | `GATE_UNROLLED_REL` | 1e-8 |
| G2 | `GATE_RESIDUAL_REL` | 1e-12 |
| G3 | `GATE_MESH_COORDS_MM` | 1e-9 |
| G4 | `GATE_FD_PLATEAU_REL` / `GATE_PLATEAU_DECADES` | 1e-4 / 1 |
| G5 | `GATE_DIRECTIONAL_REL` | 1e-5 |
| G6 | `GATE_SWEEP_REL` | 1e-3 |
| G9 | `GATE_SECANT_REL` | 1e-5 |

**Do not move one of these to make SVK pass.** G1 is the one that decides the step: it has
no finite difference in it, so a G1 failure is an adjoint defect and not a step-size
artifact. If G1 fails, **the arc stops and Step 4 becomes "the SVK adjoint is wrong, here
is the evidence"** — which is a perfectly good milestone result and much cheaper than
finding out at step 200 of a descent.

If G4/G5/G6 fail while G1 passes, suspect the finite-difference step ladder rather than the
adjoint: SVK's curvature is larger, so the plateau can move. Widen the *ladder* (more step
sizes), never the tolerance, and say so in the record.

**Record.**

```
STEP 1 RECORD — DONE (2026-08-08).  THE ADJOINT IS CORRECT UNDER SVK.
  two full coarse runs, THIS TREE, THIS GENOME, all ten gates, thresholds UNMODIFIED:
      studies/study_gradient_svk.json        svk      1215.9 s   OVERALL: PASS
      studies/study_gradient_lin_check.json  linear    996.7 s   OVERALL: PASS

  gate                       svk            linear        threshold   both
  G1  unrolled           5.893e-11        4.555e-11         1e-8      PASS
  G2  force identity     2.970e-11        2.462e-11         1e-12*    PASS
  G3  mesh coords mm     3.553e-14        3.553e-14         1e-9      PASS
  G4  plateau rel        5.616e-06        2.910e-06         1e-4      PASS
      plateau decades      2                3               >=1       PASS
  G5  directional        3.379e-06        8.056e-06         1e-5      PASS
  G6  sweep median       6.626e-06        6.298e-06         1e-3      PASS
  G7  facet ratio          3.06             3.83          falls       PASS
  G9  secant             1.079e-06        3.576e-07         1e-5      PASS
  * G2 residual is exactly 0.0 in both; the number shown is worst_force_rel.

  SVK IS NOT UNIFORMLY THE WORSE-CONDITIONED SIDE.  It is better on G5 (2.4x) and has a
  narrower FD plateau on G4 (2 decades against 3) — neither near a threshold.  There is
  no gate here that SVK squeaks through.
```

**G1 PASSES UNDER SVK, and that is the result this step existed to get.** The adjoint
reproduces brute-force differentiation of its own SVK solve to 5.9e-11 — the same quality
as linear's 4.6e-11, against a gate neither had to be given any slack on. `nonlinear=True`
in the record confirms the SVK kernel is actually engaged rather than silently falling
back, which is the failure mode `tests/test_gnl.py::test_svk_through_solve_linear_silently
_returns_the_linear_answer` exists to catch one level down.

The two runs converge to the same contact force (66.7233 N, the service load by
construction) at **different indentations** — 0.104934 against 0.104360 mm. That 0.55% is
small only because `tiny` is a crude mesh at a small drop; at `smoke` on the real genome
the same comparison is **2.2095 against 1.8048 mm, +22.4%**, reproducing §14's +23.3% at
`coarse`. It is also why `_service_indentation` had to take the kinematics: taking the SVK
gradient at the LINEAR indentation would check the adjoint at a state the wheel never
occupies.

#### The finding: SVK warm starts can leave the basin, and it is the DRIVER, not the adjoint

`--quick --kinematics svk` did not finish the first time. `_phase_sweep` refused at phase
25.0 with:

```
NewtonDivergedError: Newton direction is not a descent direction (slope 5.662e+02)
at load step 1/1, iteration 0 — the tangent is not positive definite
```

**A cold solve at that identical phase converges without complaint, to 71.9097 N.** So the
equilibrium exists and is reachable; what failed was the guess it was approached from.
`_phase_sweep` carries the converged displacement field across a 0.5 deg phase jump, and
under SVK that can land where `K(u)` is genuinely indefinite. Under linear kinematics it
cannot — the Hessian is independent of `u` (PLAN.md §0(1) H2(a) measured the difference at
exactly 0.000e+00), so a bad guess costs iterations and nothing else.

Repaired by falling back to a cold solve and **returning the phases where that fired**
rather than swallowing them (`cold_retry_phases_deg` / `n_cold_retries` in the G7 block).
That changes which starting guess is used, not which equilibrium is reported. Under SVK at
`smoke` it fired **8 times, at phases 25.0–29.0**; under linear, **0**.

**What this means for Steps 2 and 5, and it is not alarming.** Stage 3's cross-step warm
vector is *scalar indentations* seeding the secant (`warm_from`, `src/wheel_stage3.py:227`
→ `delta0`, `src/wheel_objective.py:938`), not displacement fields carried across phases.
The only displacement warm start in the objective path re-solves at essentially the same
indentation on the same mesh (`src/wheel_adjoint.py:648`), which is a close guess. And
`descend` **already** catches `NewtonDivergedError` at `src/wheel_stage3.py:495`: it
records a `solve_reject` event, halves the trial step and drops the warm vector. So a
divergence under SVK costs a step, not a run. **Step 2 should count `solve_reject` events
under SVK** — that is now a named thing to measure rather than a surprise.

#### `--quick` IS NOT A GATE, AND THE CONTROL IS WHAT SAYS SO

The quick SVK run came back `OVERALL: FAIL` on G5, G6, G7 and G9. Before reading one word
of that as an SVK defect, the same run under linear:

| gate | linear `--quick` | svk `--quick` | lin | svk |
|---|---|---|---|---|
| G1 unrolled | 4.047e-11 | 5.897e-11 | PASS | PASS |
| G4 plateau | 4.924e-06 | 3.136e-06 | PASS | PASS |
| G5 directional | 1.588e-05 | 1.579e-05 | **FAIL** | **FAIL** |
| G6 sweep median | 5.988e-06 | 8.018e-06 | **FAIL** | **FAIL** |
| G7 facet ratio | 1.326 | 0.790 | PASS | **FAIL** |
| G9 axle drop | 4.322e-05 | 9.001e-05 | **FAIL** | **FAIL** |

**`--quick` fails under linear too — `OVERALL: FAIL` on the committed default.** It is a
reduced-fidelity smoke mode for the test suite (its own `--help` says so), and its step
ladder drops the middle rungs where the plateau lives. G5 is the clearest reading: 1.588e-05
linear against 1.579e-05 SVK, i.e. **SVK is marginally the better of the two**, and both
miss a 1e-5 gate. Nothing there is about kinematics.

That left exactly **one** genuine difference in quick mode — **G7, the facet ratio: 1.33
(refines away) under linear against 0.79 (does not) under SVK** — and one gate where SVK was
2x worse while both failed anyway (G9).

**Both were quick-mode artifacts, and the full run says so.** At `coarse`, G7's facet ratio
is **3.06 under SVK against 4.58 under linear, and falls with refinement in both**; G9 is
**1.079e-06 SVK against 1.111e-06 linear**, i.e. SVK is marginally the *better* of the two,
both an order of magnitude inside a 1e-5 gate. The quick G7 runs 60 phases over a 1.0 deg
window and was never a number to conclude from.

*(The 8 cold retries were all in the smoke period sweep that LOCATES the window, phases
25–29; the window itself needed none. **At full fidelity the count is 0** — `n_cold_retries`
is 0 in `studies/study_gradient_svk.json`. The warm-start failure is real and the repair
stays, but it is a smoke-mesh phenomenon, not something the production rungs hit.)*

#### THE COMMITTED `study_gradient.json` IS STALE, AND FINDING THAT OUT WAS THE INERTNESS CHECK

The plan's step 5 said: re-run the default and confirm it reproduces the committed
`studies/study_gradient.json` on every non-timing leaf. **It does not — 3755 of 4239
non-timing leaves differ**, including the physics: linear axle drop **1.8746 mm** fresh
against **1.6546 mm** committed, same genome file, same config, same kinematics.

That is not the `--kinematics` plumbing, and it is not SVK. The dates say what it is:

```
studies/study_gradient.json     2026-08-03      <- the artifact
best_solution.json              2026-08-06      <- THE GENOME IT DESCRIBES, REPLACED
src/wheel_fea.py                2026-08-06
```

**The committed report describes a wheel that is no longer in `best_solution.json`.** The
MIN_WALL sweep replaced the genome three days after the study was last run, and the study
was never re-run. Any reading of `study_gradient.json` as "the gradient report for the
shipped genome" is wrong, and this arc made that mistake for about an hour: the first
version of the table above used the committed artifact as its linear column and reported a
**+39.6%** SVK axle-drop difference. **The real number, against a control measured on the
same genome in the same session, is +23.26%** — which independently reproduces PLAN.md
§14's +23.346% almost exactly, and that agreement is itself the check that the fresh
control is the trustworthy one.

*This is the repo's own "always measure the control" rule catching a live error, and it is
the second time in this arc — the first was `--quick` (below).*

**Since the artifact could not serve as the control, the inertness claim was proved where
it is actually checkable** — that `kinematics="linear"` is exactly what
`wheel_contact_problem` already defaults to (`src/wheel_fem.py`), so passing it explicitly
changes nothing. Built at `smoke` on the shipped genome, 15 problem fields compared:

```
CONTROL   default vs default            differs: contact, dofmap   <- object identity only
TEST      default vs explicit "linear"  differs: contact, dofmap   <- IDENTICAL TO CONTROL
SENTINEL  default vs explicit "svk"     differs: contact, dofmap, meta, nonlinear
```

The control establishes the floor (two default calls already differ on two freshly
constructed sub-objects), the test matches it exactly, and **the sentinel proves the
comparison has the power to see a real change** — without it, "no difference" would be an
untested claim. `--kinematics linear` is inert.

#### The physics the full run hands to Steps 3 and 4

Measurements, not estimates, both columns from this tree and this genome:

```
at the service load 66.7233 N, coarse, run_axle_drop
    axle drop        2.310580 mm   svk   vs    1.874573 mm   linear    +23.26%
    dF/d(indentation)  24.6809 N/mm svk   vs     36.0203 N/mm linear    -31.48%
contact patch half-width, coarse (G7 block)
    patch_half_deg      0.9338      svk   vs       0.5924      linear   +57.6%
    nodes_in_patch      2.490       svk   vs       1.580       linear
whole-report wall clock
    1215.9 s svk vs 996.7 s linear  ->  1.22x   (a first cost signal for step 2)
```

**The wheel is 31% softer at the service point under SVK** — one number, and it is the
whole milestone in miniature. The wider patch is worth noting because it cuts the *other*
way on M8's risk #7: SVK puts more nodes across the contact patch, not fewer.

**Still to reconcile in Step 3, flagged rather than smoothed over.** This study's linear
axle drop is 1.8746 mm; PLAN.md §14's load ladder reports 1.9530 mm at 1.0x, both linear
and both `coarse`. Closer than the stale artifact suggested but still not the same number,
so they remain *different quantities* — `run_axle_drop` solves for the indentation that
produces the service force at a pinned orientation, the ladder applies load through
`study_gnl`'s path. Step 3 must measure **the one the objective actually optimises**, via
`t3_terms`, and treat neither of these as that number.

Cosmetic, recorded so nobody re-derives it: the G7 slope window lands at `window_start
0.9 deg` under SVK against `3.3 deg` under linear, flipping the reported sign of
`slope_min`/`slope_max`. The study picks the steepest 3 deg window and under SVK that is
the other flank of the same ripple. Not a defect.

---

## Step 2 — Measure what SVK costs per evaluation

**Why.** Step 5 is a multi-hour run on a memory-bound box, and PLAN.md records two
production attempts killed by guessing at exactly this. The projection also decides
`--workers`, which is bounded by RAM per worker and not by cores. And the answer is
genuinely unknown in both directions: contact already runs Newton under linear kinematics
(`src/wheel_fem.py:1274`), so the SVK penalty is extra iterations plus a heavier kernel,
not a new solver.

**Read first.** PLAN.md §1, the whole "M8b-ii" section — in particular the S13 worker
ladder, the `rqmc` memory finding, and the paragraph beginning "DO NOT DO THAT."
`src/wheel_stage3.py:300-360` (`Evaluator`) and `:847-950` (`main`).

**Do.**

1. Add the flag to `src/wheel_stage3.py`'s `main()` (near line 849):

   ```python
   ap.add_argument("--kinematics", choices=("linear", "svk"), default="linear",
                   help="...")
   ```

2. Forward it. `main()` presently calls both optimizers with no `problem_kw`; both accept
   `**problem_kw` (`descend` at `:366-372`, `descend_lbfgsb` at `:703-705`):

   - `descend(z0, args.config, ..., kinematics=args.kinematics)` — `src/wheel_stage3.py:922`
   - `descend_lbfgsb(z0, args.config, ..., kinematics=args.kinematics)` — `:918`

   From there it reaches `Evaluator(**problem_kw)` at `:418` and the fidelity-check
   evaluator at `:431`, both already splatting.

3. Put it in `search_block(...)` (`src/wheel_stage3.py:814`) so every run record names the
   kinematics it descended on. A record that does not say is a record that will be misread — exactly the
   misattribution risk PLAN.md flags for `MIN_WALL_MM`.

4. **Extend `tests/test_pool.py`** with the S13 contract under SVK: one pooled evaluation
   against one serial evaluation, **values exactly equal, gradients within 1e-14**. The
   linear path already holds this; there is no reason SVK should not, and if it does not
   that is a finding about `wheel_pool_worker.py:66`'s forwarding.

5. Measure, at `coarse`, 8 phases, `--phase-scheme uniform`, `--fidelity-check-every 0`,
   from `best_solution.json`, a handful of steps only. Read `elapsed_s /
   n_objective_calls` off the run record rather than timing by hand — PLAN.md's own
   instruction. Watch RSS with `systemd-run` accounting or `/proc/<pid>/status`.

**Gate.** No pass/fail; this is a measurement. But **two numbers must exist before Step 5
is allowed to launch**: seconds per evaluation, and peak anonymous RSS with the intended
`--workers`.

**Record.**

```
STEP 2 RECORD — DONE (2026-08-09).  SVK COSTS ~1.36x TIME AND ~5% MEMORY.
  coarse, 8 phases, uniform, fidelity off, from best_solution.json, this tree, this
  genome, the five pinned env vars exported by hand to exactly Makefile:29-34.
  s/eval read off `elapsed_s / n_objective_calls` in the run record, never hand-timed.

  |        | s/eval serial |  s/eval workers 4 | Newton iters | peak anon RSS       |
  |        |  all / steady |      all / steady |    per solve | serial /  workers 4 |
  |--------|---------------|-------------------|--------------|---------------------|
  | linear | 193.7 / 129.2 |      72.5 /  45.9 |        26.00 | 7.12 GiB / 12.56 GiB|
  | svk    | 234.2 / 165.7 |      91.1 /  62.3 |        26.75 | 7.08 GiB / 13.16 GiB|

  "all" divides by every objective call; "steady" drops call 0, which is JIT warm-up
  (322.7 s serial, 152.2 s pooled) and is paid once per run, not once per step.  A
  300-step projection is a steady-state number, so `steady` is the one Step 5 uses.

  SVK penalty                     1.36x  (workers 4, steady: 62.3 / 45.9)
                                  1.28x  (serial, steady)
                                  1.17x  (JIT warm-up only)
  4-worker speedup                2.81x linear, 2.66x svk  (not 4x; the phase loop is
                                  not the whole evaluation)
  projected 300-step descent      3.9 h linear   ->   5.3 h svk   (one start, 4 workers)
  two starts, sequentially                             ~10.5 h
  MemoryMax to use in Step 5      16G  (13.16 GiB measured + 21% headroom; sampled at
                                  4 Hz, so the measurement is a floor)
  pooled == serial under SVK      values BIT-IDENTICAL, gradients within 1e-14: PASS
                                  `tests/test_pool.py::test_a_pooled_SVK_evaluation_
                                  matches_the_serial_one`

TWO INDEPENDENT CORROBORATIONS OF THE RIG, both unplanned:
  - 12.56 GiB measured for a 4-worker LINEAR descent reproduces `make prod10`'s help
    text ("one descent holds ~12.7 GB anon"), which was measured months earlier by a
    different route.
  - 3.9 h projected for a 300-step linear descent reproduces `make prod9`'s "~4 h".
  The rig is measuring the same thing the Makefile's own numbers were measured on.

WHERE THE SVK PENALTY IS, AND IT IS NOT WHERE THE PLAN GUESSED
  The plan's framing was "extra iterations plus a heavier kernel."  Measured at the
  service point, coarse, over the same 8-phase stencil (forward solves only):

      iterations / solve   26.00 lin -> 26.75 svk    1.03x
      seconds / solve       8.94 lin -> 10.11 svk    1.13x
      seconds / ITERATION  0.344 lin ->  0.378 svk   1.10x

  SVK ADDS ALMOST NO NEWTON ITERATIONS.  Contact already spends 26 of them under
  LINEAR kinematics — `wheel_fem.solve` routes on `prob.nonlinear or prob.contact is
  not None` (`src/wheel_fem.py:1274`), so the Newton loop was always there and SVK
  merely changes what each iteration costs.  Backtracks actually FELL (8 -> 4).

  But the forward solve is only 1.13x while the full evaluation is 1.36x, so most of
  the penalty is in the GRADIENT, not the forward solve: the SVK tangent assembly and
  the nonlinear vJP kernels (`src/wheel_adjoint.py:161,190,400`).  That is worth
  knowing before anyone tries to buy the cost back with a solver tolerance — there is
  no iteration count to trade away.

  Softer, as expected: at a FIXED 2.0 mm indentation the same mesh carries 58.99 N
  under SVK against 71.24 N under linear at phase 0.

WHAT THE COST RUNS INCIDENTALLY MEASURED, AND IT IS STEP 3'S QUESTION
  Every run started from the shipped genome, so step 0 of each is a free re-score of
  `350f4c7` at coarse, 8 phases:

      | 350f4c7, coarse, 8-phase | linear  |   svk   |
      | axle drop mean           | 1.99923 | 2.36935 |  +18.51%
      | stress_utilisation       | 0.78304 | 0.85914 |  +9.72%   (allowable 1.0)
      | deflection TERM          | 0.00037 |  85.262 |  x230000
      | every barrier            |   0.0   |   0.0   |
      | total loss               |  32.505 | 117.766 |

  SO: §14's ESTIMATE OF "~0.91" WAS PESSIMISTIC.  The constraint actually sees 0.859,
  and the shipped genome is STILL FEASIBLE under SVK — every barrier exactly 0.0 and
  utilisation below 1.0.  What breaks is not feasibility, it is the TARGET: the
  deflection term goes from essentially perfect to 72% of the whole loss.
  This is `coarse`; Step 3 asks the same question at `medium` with a control.

AND THE THING THAT DECIDES STEP 4
  The shipped genome is a LOCAL OPTIMUM OF THE LINEAR OBJECTIVE AND IS NOT ONE UNDER
  SVK.  Three Adam steps, 4 workers, same settings:

      linear   best stays step 0, 350f4c7, loss 32.505   (no step improved on it)
      svk      loss 117.766 -> 36.827 -> 35.437 -> 33.436, drop 2.369 -> 1.982 mm,
               hash d38c0cc, every barrier still 0.0, utilisation 0.798

  THREE STEPS RECOVER THE DEFLECTION TARGET, for +0.66 g (39.194 -> 39.852 g) and
  +9% phase ripple (0.0823 -> 0.1043).  That is one data point, not a descent, and the
  mass it costs is exactly what Step 5 has to measure properly.  But it says the SVK
  descent is not starting from a bad place, and it says the linear descent had nowhere
  left to go — which is what "descended on the wrong objective" looks like from inside.

CODE THAT LANDED, AND ONE BUG THE STEP FOUND IN ITSELF
  `src/wheel_stage3.py`   `--kinematics {linear,svk}`, forwarded to BOTH optimizers,
                          recorded in `search_block` and in the run record's settings.
  `tests/test_pool.py`    the S13 contract under SVK, sharing ONE helper with the
                          linear test so the two kinematics cannot drift into two
                          standards, plus a sentinel asserting the two kinematics
                          return DIFFERENT answers — without it the equivalence would
                          hold no matter what the pool did with the key.
  `tests/test_stage3.py`  `test_the_run_record_carries_the_kinematics_it_actually_
                          descended`, the exact shape of the existing wall-floor test,
                          and `_Args.kinematics` so `search_block`'s contract holds.

  The first version read `problem_kw` inside `_record`, where it does not exist, and
  crashed with a `NameError` AFTER a 7.5-minute solve had completed.  It now reads
  `ev.problem_kw` — the very dict the Evaluator splats into the solver, so the record
  cannot disagree with what was solved.  `search_block` likewise lost its
  `getattr(args, "kinematics", "linear")` fallback: a default there would report
  "linear" for an SVK run whose caller forgot the field, which is the exact failure
  this key exists to prevent.  Both are now covered by tests; `tests/test_stage3.py`
  is 51 passed.
```

---

## Step 3 — Re-score the shipped genome and the candidates under SVK

**Why.** §14's "utilisation on the order of 0.91" is an estimate built by scaling a
reported figure by a p99 ratio. The constraint does not compute a p99 — it computes
`Kt * sigma_nominal(p=4) / ALLOWABLE_STRESS_MPA` (`src/wheel_objective.py:1025`) with a
soft barrier per junction, and reports a p=30 p-norm alongside. **Whether the shipped wheel
is still feasible under SVK is currently unknown**, and it is knowable for the cost of a
few forward solves with no optimizer involved. This step also decides whether Step 5 starts
from the shipped genome, from a `minwall` arm, or from elite 10's answer.

**Read first.** PLAN.md §13 ("What shipped", "The design this buys, stated honestly") and
§11's `medium` re-score, which is the template for what a re-score report should contain.
`src/wheel_objective.py:870-1090` (`t3_terms`).

**Do.**

1. New driver `studies/study_svk_rescore.py`, plus a `make svk` target. **Both now
   exist** (`Makefile` `svk:`, `SVK_CONFIG ?= medium`, `SVK_WORKERS ?= 4`).

   **Keep it out of `make studies`**, for the reason PLAN.md gives for `m8bi5`, `m9buck`
   and `hubcap`: it measures **the wheel, not the commit**. Its answer does not change per
   commit, and a gate nobody can afford to run stops being run.

   **Two axle drops live in that file and they are not the same number**, which is a
   decision taken while writing it rather than a detail: `run_control` solves ONE mesh at
   ONE phase to a FIXED FORCE — study_gnl's quantity, and the only one §14 can be compared
   against — while `run_rescore` reports the OBJECTIVE'S quantity, the mean over the
   8-phase stencil that `deflection` is actually scored on. They differ by more than the
   SVK correction does, so a table that mixed them would read as a physics finding. Both
   are reported; neither is averaged into the other.

   The driver also asserts that its p=4 probe reproduces the constraint's own
   `stress_utilisation` exactly. That is the line which says the p=30 column and the
   verdict column are the same construction differing only in the exponent — without it
   the diagnostic and the verdict could drift apart silently, which is precisely how
   §14's "~0.91" estimate came to stand in for a number nobody had computed.

2. For each genome, at `medium`, under **both** kinematics, report every term:

   | genome | file |
   |---|---|
   | `350f4c7` | `best_solution.json` (shipped) |
   | `36aed36` | `best_solution_ga_beam.json` (the control) |
   | elite 10 answer | `stage3_prod_best_elite10.json` |
   | the 1.2 arm | `stage3_minwall_best_1.2.json` |
   | bracketing arms | `stage3_minwall_best_1.4.json`, `_1.6.json`, `_2.0.json` |

   Columns: `axle_drop_mm`, deflection error vs the 2.0 mm target, `Kt_hub`, `Kt_rim`,
   `sigma_nominal(p=4)`, **`stress_utilisation` as the constraint computes it**, the p=30
   p-norm, `max_stress_mpa` (a diagnostic, not a number — see §0's standing note that the
   max diverges under refinement), every barrier term, and the total loss.

3. **The control column is not padding.** `36aed36` under SVK must reproduce §14's
   +3.953% at service load. If it does not, the driver is wrong and the shipped column
   means nothing. Pre-registered in the driver as `GATE_CONTROL_REL = 0.02` — within 2%
   **of the correction**, so +23.346% must land in 22.9–23.8% and +3.953% in 3.87–4.03%.
   It gates the DRIVER, not the wheel: a red control means the table must not be quoted,
   not that the wheel changed.

**Gate.** No pass/fail. One question answered: **is the shipped genome feasible under
SVK** — every barrier at 0.0 and utilisation < 1.0 — or is it not?

**Record.**

```
STEP 3 RECORD — DONE 2026-08-09.  `make svk SVK_WORKERS=4 SVK_CONFIG=medium`
                                 -> studies/study_svk_rescore.json
  47.8 min wall, peak anon RSS 16.48 GiB (4 workers, medium), exit 0

  THE ANSWER TO THE GATE:  the shipped genome IS FEASIBLE under SVK.
    every barrier exactly 0.0, utilisation 0.8754 against an allowable of 1.0.
    §14 ESTIMATED "on the order of 0.91" by scaling a reported figure by a p99 ratio.
    The number the CONSTRAINT computes is 0.875 — §14 was pessimistic by ~4 points of
    margin, and the wheel has 12.5% of stress headroom, not 9%.
```

**The control — it reproduces §14 exactly, not merely within the gate.**

| genome | linear mm | SVK mm | rel | §14 | err | verdict |
|---|---|---|---|---|---|---|
| `350f4c7` shipped | 1.9529657 | 2.4088977 | **23.346%** | 23.346% | 1.6e-5 | PASS |
| `36aed36` GA/beam | 1.6666335 | 1.7325172 | **3.953%** | 3.953% | 2.6e-5 | PASS |

`GATE_CONTROL_REL = 0.02` allowed 2% of drift and got 0.0016% and 0.0026% — five
significant figures against numbers `studies/study_gnl.py` computed weeks earlier by a
different route. The driver is not approximately right, it is reproducing.

**A second, unplanned control landed with the first row.** `350f4c7 linear` came back
`loss 32.73762364435313, util 0.7985623196237454, drop 2.0193060279307797` — every digit
of the committed `stage3_minwall_best_1.2_medium.json`, an artifact written 2026-08-05 by
`wheel_stage3.py --steps 0`, a code path this driver shares nothing with. Two independent
controls, one on the physics and one on the whole scoring pipeline, both green.

**The table.** `medium`, 8 uniform phases, target 2.0 mm, service load.

| genome | kin | drop mm | err % | util | p30 util | mass g | loss | barriers |
|---|---|---|---|---|---|---|---|---|
| `350f4c7` shipped | lin | 2.0193 | +0.97% | 0.799 | 10.747 | 39.19 | 32.7376 | — |
| `350f4c7` shipped | **svk** | **2.3947** | **+19.74%** | **0.875** | 12.462 | 39.19 | **129.8963** | — |
| `36aed36` GA/beam | lin | 1.4973 | −25.14% | 0.493 | 3.023 | 67.27 | 369.9661 | hub_overlap 52.1, fillet_cap 103 |
| `36aed36` GA/beam | svk | 1.5577 | −22.12% | 0.507 | 3.116 | 67.27 | 334.2914 | hub_overlap 52.1, fillet_cap 103 |
| elite10 | lin | 2.0041 | +0.21% | 0.594 | 4.312 | 58.66 | 49.7265 | — |
| elite10 | **svk** | **2.1418** | **+7.09%** | **0.624** | 4.687 | 58.66 | **62.2812** | — |
| minwall 1.2 | lin | 2.0193 | +0.97% | 0.799 | 10.747 | 39.19 | 32.7376 | — |
| minwall 1.2 | svk | 2.3947 | +19.74% | 0.875 | 12.462 | 39.19 | 129.8963 | — |
| minwall 1.4 | lin | 2.0151 | +0.75% | 0.744 | 7.890 | 42.82 | 35.5178 | — |
| minwall 1.4 | svk | 2.2847 | +14.24% | 0.799 | 8.937 | 42.82 | 86.0519 | — |
| minwall 1.6 | lin | 2.0098 | +0.49% | 0.671 | 5.598 | 47.03 | 39.0645 | — |
| minwall 1.6 | svk | 2.2146 | +10.73% | 0.712 | 6.238 | 47.03 | 67.7961 | — |
| minwall 2.0 | lin | 2.0038 | +0.19% | 0.594 | 4.379 | 58.55 | 49.7082 | — |
| minwall 2.0 | svk | 2.1416 | +7.08% | 0.624 | 4.759 | 58.55 | 62.2346 | — |

**`minwall 1.2` IS `350f4c7`, bit-for-bit** — `max|Δgenes| = 0.0`, and the shipped file's
own note says it was promoted from that arm verbatim. The two rows agree on every float in
the record except `elapsed_s`, which is the free determinism check that makes the rest of
the table comparable row-to-row. **Six distinct designs, seven rows.** Recorded in the
driver next to `GENOMES` so the next reader does not count it twice.

### The three findings

**1. SVK moves exactly ONE term.** Not "the loss got worse" — the term breakdown is
bit-identical under both kinematics on `mass`, `smoothness`, `phase_ripple` and all nine
barriers, and `deflection` carries the entire difference:

| | mass | smoothness | deflection lin | deflection **svk** | stress barrier |
|---|---|---|---|---|---|
| `350f4c7` | 32.2145 | 0.2902 | 0.2330 | **97.3916** | 0.0 both |
| elite10 | 48.2140 | 1.5017 | 0.0108 | **12.5654** | 0.0 both |

That is the whole arc in one row: the shipped wheel's deflection penalty is **418× larger**
under the strain measure it is actually judged on, and it is 75.0% of its total loss.
Nothing else in the objective is a function of the strain measure, which is why the
`stress` barrier stays at 0.0 even as utilisation climbs 0.799 → 0.875.

**2. The correction is a function of the DESIGN, not a constant.** It falls monotonically
as the wheel gets stiffer, +18.6% → +13.4% → +10.2% → +6.9%, tracking wall thickness
across four arms and reproducing on a fifth design (elite10, +6.87%) that reached similar
stiffness by a different route. §14 measured it on two genomes and quoted +23.346%; it is
not one number, it is a curve.

**3. The ranking INVERTS, on every comparison in the table.** The linear loss column is
monotone increasing in wall thickness; the SVK column is monotone decreasing. Same six
designs, same objective, opposite orderings:

| | linear | SVK |
|---|---|---|
| best | **1.2 (32.74)** | **2.0 (62.23)** ≈ elite10 (62.28) |
| worst (feasible) | 2.0 (49.71) | 1.2 (129.90) |

`350f4c7` was promoted over elite10 because it won by 17 points under linear. Under SVK it
loses by 68. **The `minwall` sweep that chose the 1.2 mm floor ran entirely under linear,
and its conclusion is reversed by this table.** That is a bigger blast radius than §14
anticipated — the open question was one deflection number, and the answer implicates a
design decision made three sections earlier.

**Not a finding:** `36aed36` INFEASIBLE under both. Pre-registered in the driver as
expected — it predates the fillet-cap work, and `DEFAULT_WEIGHTS`' own comment names it as
the genome that puts that term at 103.2. It is a control on the correction, not a
candidate.

**Read `p30 util` as a diagnostic and nothing else.** It sits at 10.747 for the shipped
genome — if the constraint used p=30 the promoted wheel would be infeasible by 10×. It is
not mesh-convergent (GCI 63%, M8b-i.5), which is exactly why `STRESS_NOMINAL_P = 4.0` is
the constraint. The column is here to show the two are the same construction differing
only in the exponent; the driver asserts the p=4 probe reproduces `stress_utilisation` to
1e-12, and it does.

**For Step 5.** Peak anon RSS was **16.48 GiB at `medium` with 4 workers** — above Step 2's
13.16 GiB at `coarse`, and above the 16G cap Step 2 set. The descent runs at `coarse`
(`wheel_stage3.DEFAULT_CONFIG`), so 16G still holds for it, but **any `medium` work at 4
workers needs 20G**, and the fidelity check is `medium`. This is the second reason
`--fidelity-check-every 0` is not optional.

---

## Step 4 — THE DECISION

**Why.** This is the scope call PLAN.md §14 item 4a hands to a human, and Steps 1–3 exist
to make it decidable rather than rhetorical. **Write the options and their measured costs
down before launching anything** — the same pre-registration discipline `study_gnl.py`
used, which is the only reason its gate meant anything when it fired.

**Read first.** The Record blocks of Steps 1, 2 and 3, above. Nothing else.

**Do.** Fill in this table from the recorded numbers and pick one.

| option | what it means | cost | what it leaves broken |
|---|---|---|---|
| **A — descend under SVK** | Step 5. The optimizer sees the physics it is judged on. | Step 2's projection × 2 starts | Nothing structural. Every prior Stage-3 record becomes a linear-kinematics record. |
| **B — keep linear, re-target** | Lower `TARGET_DEFLECTION_MM` so the SVK drop lands on 2.0. | one short descent | Calibrating a target against a model known to be wrong by 23%. The correction is load-dependent (`c·f^1.03`), so it holds at one load only. |
| **C — accept 2.409 mm** | Restate the design's spec as what it is. | nothing | The stress margin, which Step 3 measures. If util ≥ 1.0 this option does not exist. |
| **D — stop** | Step 1 failed; the SVK adjoint is wrong. | nothing | Everything above. This is a legitimate milestone result. |

**Gate.** A decision, written into this file with its reasoning, before Step 5 runs.

**Record.**

```
STEP 4 RECORD — DECIDED 2026-08-09, on the numbers in Steps 1-3 and before Step 5 ran.

  decision:   A — descend under SVK.
```

**Why A.** Three things had to be true for A to be the answer, and Steps 1–3 measured all
three rather than assuming them:

1. **The SVK adjoint is correct.** Step 1, all 10 gates, thresholds *unmodified* —
   including G1, which unrolls Newton and differentiates it with no finite difference
   anywhere at `GATE_UNROLLED_REL = 1e-8`. Without this, A descends 5.3 h on a gradient
   nobody checked, and D is the answer instead.
2. **It is affordable.** Step 2: 1.36× time, 1.05× memory, 5.3 h per start, 16G cap. Two
   starts sequentially is ~10.5 h on a box that is otherwise idle. This is a large number
   but it is a *known* one.
3. **There is something to descend to.** Step 2's incidental probe: 3 Adam steps under
   SVK took the shipped genome 117.766 → 33.436 and pulled the drop 2.369 → 1.982 mm for
   +0.66 g, every barrier still 0.0. The same 3 steps under linear could not improve on
   the start at all. **The shipped genome is a local optimum of the LINEAR objective and
   is demonstrably not one under SVK** — so the descent has somewhere to go, and that was
   measured before committing the 10.5 h rather than hoped for.

**Why not B (keep linear, re-target the deflection).** Step 3 killed this option outright,
and not by the argument the plan anticipated. The plan's objection was that the correction
is load-dependent (`c·f^1.03`), so a re-target holds at one load only. The stronger
objection is finding 2: **the correction is a function of the design** — +4.0% on the
GA/beam wheel, +6.9% on elite10, +18.6% on the shipped one, falling monotonically with
stiffness across four `minwall` arms. A re-targeted constant is calibrated to *the design
that existed when you calibrated it*, and the optimizer's whole job is to move the design.
Every step of such a descent would invalidate its own target. B is not merely approximate,
it is unstable under the thing it is meant to enable.

**Why not C (accept 2.409 mm).** C is *available* — that is Step 3's gate answer, and it
was not available before this arc. The wheel is feasible: every barrier 0.0, utilisation
0.875, 12.5% of stress headroom. C is rejected on value, not on legality: it means
shipping a 19.74% spec miss when the same table already contains feasible designs at
+7.09% (elite10) and +7.08% (`minwall` 2.0), and when a 3-step probe moved the shipped
genome most of the way for 0.66 g. Accepting a miss that large while holding measured
evidence that it is cheap to fix is not a defensible engineering call.

**Why not D (stop).** D is the answer only if Step 1 failed. It passed on all 10 gates.

```
  what it costs:
    ~10.5 h of wall clock, two 300-step descents run SEQUENTIALLY under
    `systemd-run --user -p MemoryMax=16G --collect`, per Step 2's measurement.
    Nothing else in the tree changes until Step 6.

  what it leaves open — and item 3 is new, from Step 3:
    1. Every Stage-3 record in PLAN.md becomes a LINEAR-KINEMATICS record.  §14's
       headlines are not wrong, they are answers to a different question, and Step 7
       has to say so in the banner rather than quietly re-baselining them.
    2. `studies/study_gradient.json` is stale (found in Step 1) and is NOT refreshed by
       this arc.
    3. THE 1.2 mm WALL FLOOR IS NOW UNSUPPORTED.  `make minwall-*` chose it under linear
       kinematics, and Step 3 reverses that sweep's ordering over the whole 1.2-2.0 range.
       This arc does NOT re-derive the floor — it descends at 1.2 for the reason below —
       so the floor question is left open, explicitly, for a successor.
```

**The one design decision inside A, and it is a real one.** Step 3's ladder says the SVK
optimum over the *measured* designs is at the thick end, so descending from the shipped
genome at `--min-wall 1.2` looks like starting in the worst place. It is still right, for
three reasons. A floor is a **constraint, not a target** — 1.2 permits the optimizer to
choose any thickness above it, and if SVK genuinely wants thicker walls it will walk there
from inside the same box. Holding the box **identical across both starts** is what makes
the two runs comparable; changing the floor and the kinematics in one run would measure
neither, which is the argument the `minwall-` block already makes for its own control arm.
And the outcome is **informative either way**: if the descent drives the walls back onto
1.2 under SVK, that is evidence the floor survives its own reversal; if it leaves them
above, the floor was never the binding constraint and item 3 above is answered for free.

**The two starts.** `best_solution.json` (39.19 g, SVK loss 129.90) and
`stage3_prod_best_elite10.json` (58.66 g, SVK loss 62.28). The plan named these two before
Step 3 ran and Step 3 supports the choice for a better reason than the plan had: they
**bracket the mass/compliance trade** — the lightest feasible design and the stiffest — and
the SVK optimum is somewhere between them. `minwall 2.0` is *not* a third start despite
holding the table's best SVK loss: at 58.55 g against elite10's 58.66 g and 62.2346 against
62.2812, it is the same design reached by a 125-step re-descent that its own Makefile
comment predicted would "barely move". A third start there would spend 5.3 h re-measuring
elite10's basin.

---

## Step 5 — The SVK descent *(Step 4 chose A — RUNNING since 2026-08-09 11:30)*

**Why.** If the physics the wheel is judged on is SVK, the optimizer should descend on
SVK. Nothing else in this arc changes the wheel.

**Read first.** PLAN.md §1's memory rules and the `prod9`/`prod10` comment block in the
`Makefile` (lines 180–260) — every constraint there applies unchanged, and it is written
out at length because two runs died learning it.

**Do.**

1. Add `make svk9` / `svk10`-shaped targets modelled on `prod9`/`prod10`
   (`Makefile:250-260`), with:

   - `--kinematics svk`
   - `--min-wall 1.2` (the shipped floor — `src/wheel_fea.py:236`)
   - `--phase-scheme uniform` (**correctness**, see the standing rules)
   - `--fidelity-check-every 0` (costs 604.6 s per check and ~3.4 GB permanently)
   - `--workers` from Step 2's memory number, **not** from the core count
   - **distinct `--out` AND `--best-out`** — both default to fixed names under the repo
     root and `main()` writes `--best-out` unconditionally, so two runs at the defaults
     silently clobber each other's genome
   - `-u` on the interpreter, or a detached run emits nothing to the journal for hours

2. Starts: the shipped genome (`--start best`) and `stage3_prod_best_elite10.json`
   (`--start best --genome ...`), unless Step 3 says otherwise.

3. **Sequentially**, each capped:

   ```
   systemd-run --user --unit=wheel-svk9 -p MemoryMax=<step 2>G --collect \
       --working-directory=$PWD /usr/bin/make svk9
   ```

**Gate.** Every barrier at exactly 0.0 at the answer, deflection error inside ±0.3%
**measured under SVK**, and mass below the shipped 39.194 g on the mesh. A run that does
not clear these is a result, not a failure — record it and go back to Step 4.

> **PRE-REGISTERED 2026-08-09, BEFORE THE RUNS LANDED — the mass clause is expected to
> fail, and it must NOT be moved when it does.**
>
> This gate was written before Step 3, when the working assumption was that SVK would cost
> some deflection and the descent would buy it back cheaply. Step 3 says otherwise: the
> SVK loss ladder is monotone *decreasing* in wall thickness over 1.2–2.0 mm, and the
> table's best SVK design (elite10, 58.66 g) is **50% heavier** than the shipped genome
> (39.19 g). If SVK's answer to a 19.74% deflection miss is more material — and three
> independent comparisons in Step 3 say it is — then "mass below 39.194 g" is asking the
> descent to beat the shipped wheel on the one axis the shipped wheel was over-optimized
> on, under the strain measure that says so.
>
> **Recording that breach is the finding.** The standing rule is `measure before touching
> a threshold`, and its companion — never re-fit a pre-registered gate to a design that
> breached it — is exactly what `make m9` had to obey. So: if mass comes back above
> 39.194 g with every barrier at 0.0 and deflection inside ±0.3%, the honest statement is
> **"SVK costs this design N grams"**, and that number is the real deliverable of this
> arc. It is not licence to relax the gate, and Step 6 promotes on the barriers and the
> deflection, with the mass delta stated plainly as a cost.

**Record.** Start genome, final genome hash, mass, deflection error, utilisation, every
barrier, steps accepted/rejected, wall clock, peak RSS, and whether the projection from
Step 2 held.

```
STEP 5 RECORD — RUNNING.  Launched 2026-08-09 11:30 -0400, sequential, each under
  `systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --collect`.

  run 1  make svk-shipped   best_solution.json            -> stage3_svk_best_shipped.json
  run 2  make svk-elite10   stage3_prod_best_elite10.json -> stage3_svk_best_elite10.json

  step 0 of run 1: loss 117.7664  |grad| 8874  drop 2.369 mm  util 0.859  186.7 s
    — reproduces Step 2's incidental probe (117.766) exactly, so the descent starts
      where it was measured to start.  186.7 s is jit compilation; Step 2 projects
      62.3 s/step steady, i.e. ~5.3 h per run.

  IN-FLIGHT, run 1, steps 50-100 — THE STRESS BARRIER IS BLIND BELOW 1.0, AND THE GATE
  INHERITS THAT BLINDNESS.  Utilisation is climbing monotonically while the loss sits
  still:

    step   50   loss 32.3637   drop 1.996   util 0.827
    step   60   loss 32.7969   drop 1.970   util 0.825
    step   70   loss 32.0993   drop 1.994   util 0.832
    step   80   loss 32.1428   drop 1.977   util 0.836
    step   90   loss 31.9544   drop 1.972   util 0.848
    step  100   loss 32.2034   drop 1.959   util 0.868

  Fifty steps bought 0.4% of loss and spent 0.041 of utilisation, and the run is now
  ABOVE the 0.859 it started from.  The mechanism is not subtle and it is not a bug:
  `stress` is `soft_barrier(util - 1.0, 4000)` (`src/wheel_objective.py:1027`) and
  `soft_barrier` is `scale * max(0, violation)**2` (`:290`), so it is identically zero
  AND HAS IDENTICALLY ZERO GRADIENT for every util <= 1.0.  Below the knee the optimizer
  cannot see stress at all; it sees only mass, which it reduces by thinning the wall.
  The barrier is a wall to stop at, never a cost to trade against.

  This matters for the gate, which reads "every barrier at exactly 0.0".  A design parked
  at util 0.999 satisfies that clause EXACTLY and ships with no stress margin whatsoever,
  and with a weight of 4000 against a loss of ~32 the barrier will slam it back to
  precisely there if it crosses.  The gate as pre-registered cannot distinguish 0.5 from
  0.999.  DO NOT FIX THIS BY MOVING THE GATE MID-RUN — the arc's own rule is that a
  pre-registered gate is never re-fitted to the design that breached it.  Record the final
  utilisation as a number in the table, treat "barrier 0.0 at util > 0.95" as a pass with
  a stated caveat rather than a clean pass, and carry the missing margin clause into
  Step 7 as a defect in the objective, not in this run.  The shipped genome has form
  here: this file's own header (`src/wheel_objective.py:23`) notes the wheel was already
  sitting "exactly on the stress barrier (25.08 MPa against a 25.0 allowable)".

  RESOLVED AT CONVERGENCE — THE EXTRAPOLATION ABOVE WAS WRONG, THE MECHANISM WAS NOT.
  Utilisation did not run to the barrier.  It plateaued at 0.899 and stayed there for the
  last 180 steps:

    step  110  0.888     step  200  0.899
    step  120  0.899     step  250  0.899
    step  150  0.899     step  300  0.8989

  The 50->100 slope (0.827 -> 0.868) projected ~0.99 by step 300 and that projection is
  NOT borne out; it stopped 0.10 short of the knee.  Why it stopped is worth knowing,
  because it is not the stress barrier doing the stopping: at step 300 all four wall genes
  t0..t3 sit ON the 1.2 mm floor (`final.bound_saturation` = t0,t1,t2,t3 all "low").  The
  run ran out of wall to thin before it ran out of stress margin.  The 0.899 is therefore
  an artefact of WHERE THE FLOOR IS, not evidence that the objective protects margin — set
  the floor lower and the same blind gradient would keep going.  The zero-gradient finding
  stands exactly as written and still belongs in Step 7 as an objective defect; what
  changes is only that this particular run was saved by a different constraint, so the
  0.95 caveat clause never fires and run 1 is a CLEAN pass rather than a caveated one.

  AND THAT IS THE SECOND FINDING: the answer is pinned to the 1.2 mm floor, and Step 3
  established that the floor was chosen by a `minwall` sweep run under LINEAR kinematics
  whose ranking SVK INVERTS.  So this design sits hard against a constraint whose
  justification this arc has already invalidated.  That is not a reason to move the floor
  now — moving it mid-arc would be re-fitting — but the 1.2 mm number is doing real work
  in the result and Step 7 must say so.  Re-deriving the floor under SVK is the natural
  successor to this arc and is named in the closing section.

  | run | steps | final loss | mass g | defl err (svk) | util | worst barrier | wall |
  |-----|-------|------------|--------|----------------|------|---------------|------|
  | shipped | 300/300 | **30.8207** | **37.451** | **-0.043%** | 0.8989 | **0.0** | 4.90 h |
  | elite10 | 300/300 | 30.8245 | 37.449 | -0.044% | 0.9085 | **0.0** | 4.94 h |

  BOTH RUNS PASS ALL THREE PRE-REGISTERED CLAUSES.  Run 2: 301 calls, 17792.3 s = 4.94 h,
  59.1 s/step, genome_hash c4f207c, one handled solve_reject (step 128, above), 0
  abandoned.  Step 2's 62.3 s/step projection held for the second time.

  RUN 1 — PASSES THE PRE-REGISTERED GATE ON ALL THREE CLAUSES, unmodified:
    - every barrier exactly 0.0        x_order, hub_overlap, fold, arrival, fillet,
                                       fillet_cap, buckling, min_sj, stress — all 0.0
    - deflection inside +/-0.3% (svk)  1.99913 mm vs 2.0 target = -0.043%, 7x inside
    - mass below the shipped 39.194 g  37.451 g, -1.743 g (-4.45%)

  So SVK did not cost this design grams; it SAVED 1.74 g while moving deflection from
  2.369 mm (+18.5% over target, §14's real number) to 2.0 mm.  The pre-registered "SVK
  costs this design N grams" clause does not fire.

  final genome_hash ae7092c   loss 117.7664 -> 30.8207 (3.82x)   best step == step 300
  0 rejected steps, 0 abandoned, 0 events — the descent never fought the line search.

  term-by-term, step 0 -> step 300:
    deflection   85.2617 -> 0.00047     the whole point; SVK's error is now noise
    mass         32.2145 -> 30.7820
    smoothness    0.2902 -> 0.0382
    all barriers  0.0    -> 0.0         never entered a barrier at any step
  metrics:            step 0      step 300
    axle_drop_mean    2.36935     1.99913
    mesh_mass_g      39.19436    37.45146
    stress_util       0.85914     0.89889
    max_stress_mpa  160.57577   220.21472   <- see caveat below
    min_scaled_jac    0.71086     0.60663   <- degraded 15%, barrier still 0.0
    buckling_ratio    0.15728     0.14742
    phase_ripple      0.08235     0.09593   <- term still 0.0 (inside deadband)

  CAVEAT ON max_stress_mpa: the raw peak grew +37% (160.6 -> 220.2 MPa) while the
  utilisation the constraint actually sees grew +4.7% (0.859 -> 0.899).  The two are
  decoupling because the constraint is `Kt * pnorm(p=4) / 25.0`, and a p=4 aggregate is
  deliberately insensitive to a singular corner peak — that is what `Kt` and
  `stress_scale_measured` (23.96) exist to bridge.  This is by design and p=30 is NOT
  mesh-convergent (GCI 63%, M8b-i.5), so the p=4 number is the right one to gate on.  But
  a 37% growth in the raw peak against a 4.7% growth in the aggregate is worth one line in
  §15: the aggregate is tracking the field, not the corner, and the corner moved more.

  TIMING — STEP 2'S PROJECTION HELD.  wall 17646.8 s = 4.90 h for 301 objective calls.
  Steady-state 58.18 s/step mean (min 33.2, max 74.6) against Step 2's projected 62.3,
  i.e. the projection was 6.6% CONSERVATIVE.  mesh 4.3 s, solve 17636.6 s — 99.94% of the
  run is inside the solver, which is where Step 2 said the 1.36x SVK penalty lives.
  Memory: peak cgroup 14466 MiB observed at step ~110 against the 16 GiB cap, swap 0,
  no OOM, no kill.  The 16G cap chosen in Step 2 was right and had ~1.5 GiB of headroom.

  ONE THING RUN 1 DOES NOT ESTABLISH, carried into Step 6: this descent ran at `coarse`.
  Step 3's re-score ran at `medium`, where the same shipped genome read 2.3947 mm against
  coarse's 2.36935 — a 1.1% fidelity gap.  A deflection converged to -0.043% at coarse is
  NOT thereby inside +/-0.3% at medium.  Step 6 must re-score the promotion candidate at
  medium under SVK BEFORE promoting, and if it lands outside +/-0.3% there, that is a
  finding about the rung the descent was run on, not licence to promote anyway.

  run 2 (elite10) launched 2026-08-09 16:24:38 -0400, same cap, same targets.

  RUN 1 REPRODUCED THROUGH AN INDEPENDENT DRIVER.  While run 2 was in flight, run 1's
  saved genome was re-scored by `study_svk_rescore.py` (new `--extra label=path` flag,
  additive so the driver at its DEFAULTS still reproduces Step 3's artifact unchanged),
  at `coarse`, workers 0, control skipped:

    from stage3_svk_best_shipped.json   from study_svk_rescore.py --extra
      drop  1.99913 mm                    drop  1.9991 mm
      util  0.89889                       util  0.8989
      loss  30.82069                      loss  30.8207
      mass  37.45146 g                    mass  37.45 g

  Every digit agrees.  The descent's own reported answer is therefore reproducible from
  the saved genome by a code path that shares only the objective — so the Stage-3 run
  record is not reporting an internal state that the genome does not actually encode.
  (Control was SKIPPED in this invocation, so these numbers are a self-consistency check
  ONLY and must not be quoted against §14.  The medium-rung check in Step 6 runs WITH the
  control.)

  AND THE THIRD FINDING, which the re-score handed over for free — THE NEW DESIGN IS MORE
  GEOMETRICALLY NONLINEAR THAN THE ONE IT REPLACES, NOT LESS:

    genome                     linear mm    svk mm    Δsvk
    350f4c7 shipped (Step 3)      2.0193    2.3947   18.593%
    ae7092c svk-descended         1.6254    1.9991   22.996%

  The descent did not find a design where the linear/SVK gap closes; it found one where
  the gap is 4.4 points WIDER.  That is the mirror image of the shipped genome: the old
  design hit 2.0 mm under linear and read 2.39 under SVK, the new one hits 2.0 mm under
  SVK and reads 1.63 under linear (-18.7%).  Both are specialised to the kinematics they
  were descended on, and this arc has only moved WHICH model is the honest one.  The
  practical consequence for §15: after promotion, any linear-kinematics number computed on
  the shipped wheel is MORE wrong than it was before this arc, not less, and every study
  in the tree that still defaults to `kinematics="linear"` inherits that.  Naming those is
  Step 7's job.

  (Timings in that run are contaminated — it shared the box with run 2's four workers —
  so its 505 s linear vs 285 s svk says nothing about Step 2's measured 1.36x and is not
  offered as evidence against it.)

  RUN 2, STEP 128 — THE LOAD-CONTROL SECANT CANNOT ALWAYS REACH 1e-8 UNDER SVK.  One
  `solve_reject` event, handled, run continued:

    {"step": 128, "kind": "solve_reject", "attempt": 0, "scale": 1.0,
     "error": "RuntimeError",
     "message": "contact load control did not reach 66.7233 N in 20 iterations;
                 last force 66.723265 N at indentation 2.197308 mm"}

  This is NOT a crash and NOT a hang, though it prints a traceback on stderr and reads
  like one.  `wheel_pool_worker.py:88-98` catches every worker-side exception and reports
  it to the parent as the `solve_reject` that `descend` already knows how to handle; the
  traceback on stderr is for the person debugging, not a fatal.  The run took the next
  step normally.

  What actually failed is worth a line in §15.  `solve_wheel_contact`
  (`src/wheel_fem.py:1841`) is a secant on indentation with `tol_rel=1e-8`, i.e. it wants
  the contact force inside 6.7e-7 N of 66.7233 N.  It stalled at 66.723265 N — a residual
  of 3.5e-5 N, or 5.2e-7 RELATIVE, about 52x above the tolerance it is asked to hit.  The
  load is physically reached to within a part in two million; what fails is the outer
  secant's ability to resolve a force difference smaller than the noise floor of the inner
  Newton solve that produces it, and SVK raises that floor.  The function raises rather
  than returning the state, which is correct and documented — "a returned 'close enough'
  state is a wrong load case, and a Stage-3 gradient computed at one is undebuggable
  after the fact" (`:1850`).

  TWO CONSEQUENCES, and neither is "loosen tol_rel":
    - RUN 1 IS UNAFFECTED.  301 objective calls, `n_reject_cumulative` 0, zero events.
      Every evaluation behind the promoted answer met 1e-8.
    - DO NOT RELAX IT MID-ARC.  Loosening `tol_rel` now would make run 2 incomparable to
      run 1 AND would change the setting under which run 1's answer was computed, on the
      evidence of the run that breached it — the exact re-fit this arc forbids.  The
      finding is that 1e-8 is not universally achievable under SVK, and the right response
      is to measure the inner solve's noise floor and set the outer tolerance from it,
      as its own piece of work.  Named in the closing section.

  ==========================================================================
  BOTH RUNS IN — WHAT THE AGREEMENT IS, AND WHAT IT IS NOT
  ==========================================================================

  The two losses agree to 0.012% (30.8207 vs 30.8245) and the two masses to 0.008%
  (37.4515 vs 37.4486 g) from starts 19 g apart.  That is NOT a shared point optimum.
  The genomes differ:

    gene    run1 (ae7092c)  run2 (c4f207c)   diff
    cx1         4.835197        5.187072    +7.28%
    cy1         4.297267        4.594920    +6.93%
    cy4         4.686810        4.887004    +4.27%
    R_hub       0.578951        0.557862    -3.64%
    cx2..cy3    (all within 1.6%)
    t0,t1,t2,t3 1.200000        1.200000     BOTH ON THE FLOOR
    R_rim       2.749468        2.749468     IDENTICAL TO 6 D.P.

  So the objective is FLAT along a manifold: the spline control points move up to 7.3%
  for a fourth-decimal change in loss.  Report the agreement as "both starts reach the
  same basin and the same headline numbers", never as "the optimizer found THE answer".
  A future run from a third start will land somewhere else on the same shelf.

  AND THE FOURTH FINDING, which falls straight out of the gradient trace and is the
  sharpest thing this arc has produced — TWO OF THE FOURTEEN GENES ARE DEAD:

    over all 602 steps of both runs
      R_rim   nonzero gradient on    0 steps    never moved, either run
      R_hub   nonzero gradient on    2 steps    (run 2 only, steps 24 and 73)
      stress term > 0 on             0 steps    max util over both runs 0.9089

  And the two steps where R_hub had a gradient are EXACTLY the two steps where the
  `fillet_cap` barrier was live (0.0026 and 0.0088).  That is the whole mechanism, and it
  closes the loop with the zero-gradient finding above: the two fillet radii reach the
  loss ONLY through `stress` (via kt_hub/kt_rim) and through the fillet barriers.
  `stress` is identically flat below util 1.0, and the fillet barriers are identically
  zero unless breached.  So for essentially every step of every run, NOTHING IN THE
  OBJECTIVE CAN SEE EITHER FILLET RADIUS.

  The consequence is concrete and it changes how the answer must be described: run 1's
  `R_hub` 0.5790 and `R_rim` 2.7495 ARE NOT OPTIMISATION RESULTS.  They are constants
  inherited from `best_solution.json` and carried untouched through 300 steps.  Nothing in
  this arc has ever chosen them, and the same blind path was in place for every Stage-3
  run behind §14.  The search is nominally 14-dimensional; it is actually 12-dimensional,
  and with all four thicknesses pinned on the floor the live subspace is the 8 spline
  coordinates — precisely where the two runs still disagree by up to 7.3%.

  This makes the stress-margin term the highest-value follow-on by some distance: adding
  it does not merely stop the optimizer spending margin it is not charged for, it
  UNFREEZES TWO DESIGN VARIABLES that no run in this repo's history has been able to move.

  PROMOTION CANDIDATE: RUN 1, ae7092c.  Both pass, so the choice is on the tie-breaks:
    - lower loss                 30.8207  vs 30.8245
    - more stress margin         util 0.8989 vs 0.9085
    - better element quality     min_sj 0.6066 vs 0.6055
    - already independently reproduced (the --extra re-score above)
  Run 2 is lighter by 0.0029 g.  That is 2.9 mg, 0.008%, far inside mesh discretisation
  noise, and it is not a reason to prefer a design with 0.010 less stress margin.

  ==========================================================================
  STEP 6'S MEDIUM CHECK — THE CANDIDATE DOES NOT PROMOTE.  DEFLECTION IS
  +1.65% AT `medium`, AGAINST A +/-0.3% GATE.
  ==========================================================================

  `make svk SVK_ONLY=350f4c7 SVK_EXTRA=svk-shipped=stage3_svk_best_shipped.json`
  at `medium`, 8 phases, 4 workers, 967.9 s, CONTROL PASS (§14's ladder reproduced to
  0.00% on both genomes, so this table is comparable to §14):

    genome           kin      drop mm    err %    util   mass g       loss
    350f4c7 shipped  linear    2.0193    +0.97%   0.799   39.19    32.7376
    350f4c7 shipped  svk       2.3947   +19.74%   0.875   39.19   129.8963
    ae7092c svk-desc linear    1.6498   -17.51%   0.831   37.45   107.4570
    ae7092c svk-desc svk       2.0330    +1.65%   0.937   37.45    31.5001

  The candidate converged to -0.043% at `coarse` and reads +1.65% at `medium`.  That is
  5.5x the gate, and it is the exact failure the Step 5 record pre-registered a week of
  hours ago: "a deflection converged to -0.043% at coarse is NOT thereby inside +/-0.3% at
  medium ... if it lands outside +/-0.3% there, that is a finding about the rung the
  descent was run on, not licence to promote anyway."  SO IT IS NOT PROMOTED.  The rule
  holds even though the check was one I added rather than one Step 4 pre-registered —
  especially then.

  UTILISATION IS ALSO WORSE THAN THE DESCENT BELIEVED: 0.937 at `medium` against 0.899 at
  `coarse`.  The descent could not have traded on this even in principle (the stress term
  has no gradient below 1.0, above), but what would ship has 0.063 of margin, not 0.101,
  and 0.937 is close enough to the 0.95 caveat line to matter.

  THE CONTROL IS WHAT MAKES THIS READABLE, AND IT SAYS THE GATE WAS NEVER A `medium` GATE.
  The INCUMBENT fails +/-0.3% at `medium` too — +0.97% under the very kinematics it was
  descended on.  No design in this repo has ever met +/-0.3% at `medium`; the gate has only
  ever been met at the rung the descent ran on, and this arc inherited that assumption
  without testing it.  That is a finding about the gate, not about this candidate, and it
  is why the response is to re-converge at `medium` rather than to move the number.

  AND ON THE THING THE ARC EXISTS TO FIX, THE CANDIDATE IS A LARGE IMPROVEMENT.  Judged
  under the honest model (svk, medium), against the incumbent under the same:
        deflection error   +19.74%  ->  +1.65%     (12x closer to target)
        mass                39.19 g ->  37.45 g    (-1.74 g)
        stress margin        0.125  ->   0.063     (worse, and it is the real cost)
  Not promoting is NOT a judgement that the candidate is bad.  It is 12x better on the
  headline.  It is a judgement that "12x better" is not the gate, and the gate was written
  down first.

  A PROCESS FINDING FOR §15, and it is the cheapest lesson here: Step 5 ran with
  `--fidelity-check-every 0`.  `descend` HAS the machinery for exactly this
  (`src/wheel_stage3.py:384`, `_fidelity_check` at `:272`) — it forward-evaluates the
  accepted step at a second fidelity and attaches the result to the step row.  It is "a
  PURE OBSERVATION" and could not have redirected the descent, but with
  `--fidelity-check-every 25 --fidelity-check-config medium` the coarse/medium gap would
  have been on the record at step 0 instead of discovered after 9.8 h of descending.
  Turning it off saved perhaps 4% of wall clock and cost the arc a run.

  DECISION — RE-CONVERGE AT `medium`, WARM-STARTED FROM ae7092c.  Not a fresh descent: the
  design is 1.65% from target, not 19.74%, so what is needed is the last of the error
  removed at the rung that will be believed, and the cost is the honest one to pay.
  Measured from this run: one value+grad at medium/4 workers is 273 s against coarse's
  58.6 s, i.e. 4.7x, so ~100 steps is ~7.6 h.  This does not move a threshold, does not
  re-fit a gate, and does not promote anything on a failed check — it meets the existing
  gate at the fidelity that matters.  Run it with the fidelity check ON, pointing the
  other way (`--fidelity-check-config coarse`), so the record carries both rungs.

  ==========================================================================
  `make svk-medium` — bc77614 PASSES AT `medium`.  100 steps, 6.29 h,
  101 calls + 5 fidelity calls, 0 rejects, 0 events, best at step 92.
  ==========================================================================

    clause                        required        bc77614 at `medium`
    every barrier exactly 0.0     0.0             0.0, all nine
    deflection error (svk)        +/-0.3%         -0.041%   (1.99919 mm)
    mass below shipped 39.194 g   < 39.194        37.414 g  (-1.781 g, -4.54%)

  Also lighter than the coarse candidate it replaces (37.414 vs 37.451 g) and lower loss
  at the rung both are now measured on (30.7829 vs 31.5001).  224 s/step against the 273 s
  projected from the Step 6 re-score — the projection was 18% conservative.

  AND THE FIDELITY CHECK, POINTED BACK AT `coarse`, GIVES THE MIRROR IMAGE — WHICH IS THE
  REAL RESULT HERE.  The medium answer evaluated at coarse, every 25 steps:

    step   0   coarse drop 1.99913 mm   (this is ae7092c, and it reproduces exactly)
    step  25   coarse drop 1.96080 mm
    step  50   coarse drop 1.96900 mm
    step  75   coarse drop 1.96516 mm
    step 100   coarse drop 1.96571 mm   =  -1.71%

  So: the COARSE-converged answer reads +1.65% at medium, and the MEDIUM-converged answer
  reads -1.71% at coarse.  The two rungs disagree by ~1.7% on this wheel and NO DESIGN CAN
  SATISFY +/-0.3% AT BOTH.  The gate is satisfiable at exactly one fidelity, and which one
  is a CHOICE, not a property of the design.  This arc chooses `medium`, because it is the
  finer of the two and because §14's control ladder is stated there — but the honest
  sentence for §15 is that the wheel is now specialised to a rung as well as to a
  kinematics, and a third rung would move it again.  The right long-term answer is a mesh
  convergence study on `axle_drop_mean_mm` (the GCI treatment M8b-i.5 gave the stress
  QoI), so the gate can be stated against an extrapolated value instead of a rung.  Named
  in the closing section.

  THE DEAD GENES STAYED DEAD AT `medium`, which rules out the one benign explanation.
  `R_hub` 0.578951 and `R_rim` 2.749468 are UNCHANGED to six decimals from ae7092c through
  another 100 steps at a finer fidelity.  It was not a coarse-mesh artefact; the radii are
  structurally invisible to the objective, exactly as the gradient trace said.

  All four walls remain pinned on the 1.2 mm floor, so the floor is load-bearing for this
  answer too, and the caveat recorded above carries over unchanged.

  UTILISATION IS THE ONE NUMBER THAT GOT WORSE AND IT SHOULD BE READ AS THE COST OF THIS
  ARC: 0.799 (shipped, linear, medium) -> 0.875 (shipped, svk, medium) -> 0.935 (bc77614,
  svk, medium).  Below the 0.95 caveat line, so this is a CLEAN pass on the pre-registered
  wording, but the wheel ships with 0.065 of stress margin against the incumbent's 0.125.
  Roughly half of that is not a design change at all — it is the correction from measuring
  the same wheel honestly (0.799 -> 0.875).  The rest was spent by an optimizer that
  cannot see stress below 1.0.  Say both halves in §15.

  ==========================================================================
  STEP 6 STOPS HERE.  bc77614 IS NOT PROMOTED.  IT PASSES EVERY FEA GATE AND
  IS NOT BUILDABLE AT THE STRESS CONCENTRATION IT WAS PRICED AT.
  ==========================================================================

  `make export EXPORT_GENOME=stage3_svk_best_medium.json`, and the SAME export run on the
  incumbent as the control the repo's rules require:

    genome              worst wedge   hub built                    kt_error_pct
    350f4c7 shipped        328.0 deg  24/24 @ 0.579 mm                    0.0%
    bc77614 svk-medium     308.0 deg  12 @ 0.579, 12 @ 0.418 mm         +11.9%

  THE INCUMBENT BUILDS EXACTLY AS MODELLED.  THE CANDIDATE DOES NOT.  This is a regression
  introduced by this arc, not a defect it inherited, and the control is the only reason
  that sentence can be said with confidence.

  What it costs, in the only units that matter:
    Kt at the hub      modelled 2.0235   as built 2.2643   +11.9%
    peak stress        modelled  294.02 MPa   as built ~329.01 MPa
    UTILISATION        modelled    0.9347      AS BUILT     1.0461
  The objective priced the hub junction at a radius OCC then refused on half the edges.
  Corrected for what was actually built, the wheel is at 1.046 of allowable — INFEASIBLE
  as built, on the one constraint this whole arc has been circling.

  Everything else in the export is clean, which is what makes the failure legible rather
  than ambiguous: OCC valid True, 1 solid, bbox 100.00 x 100.00 x 22.40 mm, BRepCheck
  valid, self-intersecting False, degenerate 0, min curvature R 0.4184 mm against the 0.25
  floor, junction bite floor 0.25 satisfied.  Only the fillet feasibility is red.

  AND IT IS THE THREE EARLIER FINDINGS COLLECTING THEIR DEBT, IN ORDER:
    1. `stress` has zero gradient below util 1.0, so nothing prices stress margin.
    2. Therefore `R_hub` and `R_rim` are dead genes — the ONLY paths from a fillet radius
       into the loss are `stress` and the fillet barriers, both flat here.
    3. Therefore the descent was free to swing the spline until the hub arrival went
       shallow (wedge 328 -> 308 deg) with NOTHING IN THE OBJECTIVE OBJECTING, and it
       could not have compensated by asking for a smaller radius either, because R_hub is
       exactly the gene it cannot move.
  The exporter's own diagnostic reaches the same place unprompted: "what is left is the
  shallow corner of a near-tangent arrival, which is the arrival angle, i.e. the genome."

  WHY NOT JUST LOWER R_hub BY HAND until modelled == built.  Because that is fitting the
  geometry to the check after seeing the check fail, it would have to be done by hand
  precisely because the optimizer cannot do it, and it would leave the same blind spot in
  place for the next design.  The defect is that the objective cannot see buildability,
  and the fix belongs in the objective.

  SO THE ARC'S RESULT IS A MEASUREMENT, NOT A PROMOTION, and Step 4 pre-registered exactly
  this outcome: "a run that does not clear these is a result, not a failure — record it
  and go back to Step 4."  `best_solution.json` is UNCHANGED and still holds 350f4c7.
  Nothing was promoted, nothing was re-baselined, and `wheel.step` was regenerated from
  the unchanged incumbent genome (manifest genome_hash 350f4c7) purely to obtain the
  control.

  WHAT THE ARC DID ESTABLISH, and it is not small: SVK is the honest kinematics and the
  shipped wheel is 2.39 mm not 1.95 mm; a design exists that is 12x closer to target and
  1.78 g lighter under that kinematics; and the objective has four named, measured defects
  that together explain why that design cannot ship.  §15 writes all of it.
```

**One change made while run 1 was in flight** (`src/wheel_stage3.py:946`, tests re-run,
51 passed): the console banner now prints the kinematics —
`STAGE 3 — best_solution (adam, coarse, 300 steps, uniform, svk)`. The run *record* has
carried the key since Step 2, but a record is read after the fact, and the failure this
whole arc exists to correct is that four hours of descent looked identical from the
outside whichever strain measure was underneath. Run 1 predates the edit; run 2 shows it.

---

## Step 6 — Export, check, promote *(blocked on Step 5)*

**Why.** §11–§13 established what a promotion requires, and it is not a file copy. The
last one turned up a weak-junction check that was really measuring `t0` (§12) and a hub
fillet family that had never been built (§3).

**Read first.** PLAN.md §11 ("What promotion still needs"), §12, and §13 ("The export,
old → new").

**Do.**

1. `make export EXPORT_GENOME=<candidate>.json` — which names artifacts after the genome's
   stem and **cannot overwrite the shipped STEP** (`Makefile:294-301`).
2. Check, against §13's table: `kt_error_pct` at both junctions (the shipped genome is
   +0.0% at both — the first part whose built fillets match the ones its stress model
   priced; a regression here is a real cost), `junction_bite` against
   `MIN_JUNCTION_BITE = 0.25`, `fillets.volume_mm3` and its share of the solid,
   `step_health`, smallest edge, face count.
3. The Inventor import. That is what gated §13's promotion and it is the consumer now.
4. Only then `best_solution.json`. **Leave `tests/test_golden.py` reading
   `best_solution_ga_beam.json`** — §10 decoupled them precisely so a promotion cannot
   re-baseline the regression net.
5. `make test` in full, diffed against **Step 0**, test by test. Any new red is triaged,
   not accommodated.

**Gate.** OCC-valid, Inventor-clean, and a `make test` whose every difference from Step 0
is accounted for.

**Record.** The old → new manifest diff, all 33 geometric keys, and the gate count.

---

## Step 7 — Write the record into PLAN.md

**Why.** PLAN.md is the tree's memory and this file is one milestone's working notes. §15
is where the arc becomes something a fresh session finds without being told to look.

**Do.** Add a **§15** to PLAN.md in the voice the rest of the file uses — what moved, what
is now stale, what was measured and what was assumed — plus a one-line entry in §0's
bulleted list. If the shipped genome changed at Step 6, update the top-of-file banner:
every per-design number in §1–§14 then describes the *previous* wheel, exactly as §13's
banner had to say.

State plainly, whatever the outcome: **before this milestone, every Stage-3 number in the
repo was a linear-kinematics number.** That is true even if Step 4 chooses C.

**Record.**

```
STEP 7 RECORD — DONE (2026-08-10).  THE ARC IS CLOSED.

  THREE EDITS TO PLAN.md, and nothing else in the tree moved:

  1. §15 appended after §14 — "STAGE 3 WAS DESCENDING ON THE WRONG PHYSICS.  It can now
     descend on the right physics, and the wheel that finds cannot be built.  NOTHING
     PROMOTED."  Eleven subsections: what landed in the code, the adjoint proof, the cost,
     the re-score and its three findings, the decision, the two descents, the fidelity
     trap, the export refusal, THE FOUR DEFECTS IN THE OBJECTIVE, what is now stale, the
     gate, the five successors, the artifacts.
  2. §0's bulleted list — one entry, same shape as the §13/§14 entries, ending on
     "`best_solution.json` is UNCHANGED".
  3. THE TOP-OF-FILE BANNER — a third paragraph.

  ON THE BANNER, because Step 7 as written did not require it.  The instruction was
  "if the shipped genome changed at Step 6, update the banner."  IT DID NOT CHANGE, so
  that clause does not fire.  The banner was amended anyway, on the other clause: it
  exists to say READ THIS BEFORE TRUSTING ANY NUMBER BELOW, and "every Stage-3 deflection
  and utilisation in §1-§14 is a linear-kinematics number" is exactly that kind of fact.
  The paragraph states the genome did NOT change in the same breath, so nobody reads a
  caveat as a promotion.

  WHAT §15 DELIBERATELY DOES NOT DO
    - It does not correct one number in §1-§14.  Those sections are not wrong; they are
      answers to a different question, and §13's banner set the precedent for saying so
      rather than editing history.
    - It does not add to the "## Artifacts" prose section.  §11-§14 did not either — they
      carry their own artifact notes inside themselves, and §15 does the same.
    - It does not soften the four defects into "future work".  They are stated as measured
      defects with their measurements attached, because the terminal finding of this arc
      is that they are the reason nothing shipped.

  THE FOUR THINGS §15 IS WRITTEN TO MAKE UNMISSABLE, in the order a fresh session hits them:
    banner  -> every Stage-3 number below is linear, and the genome did not change
    §0      -> the one-paragraph version, with the +11.9% and the dead genes in it
    §15     -> the evidence, the controls, and the five successors in priority order
    successor 1 is BUILDABILITY, and it is the one that has to be fixed before ANY SVK
    descent can ship.  2 (stress margin) and 3 (the wall floor) are subsumed by it as far
    as promotion is concerned, and §15 says so rather than listing five equal items.

  THE GATE.  `make test` re-run at the close of the arc, first full run since Step 0.
  MEASURED, NOT INFERRED:  433 passed / 2 failed in 1468.57 s (24:28).

      Step 0 (2026-08-08)      431 passed / 2 failed   1374.52 s
      Step 7 (2026-08-10)      433 passed / 2 failed   1468.57 s
      delta                     +2 passed / 0 failed

  The +2 is EXACTLY the two tests this arc added, and nothing else moved:
      tests/test_pool.py::test_a_pooled_SVK_evaluation_matches_the_serial_one
      tests/test_stage3.py::test_the_run_record_carries_the_kinematics_it_actually_
          descended
  Both reds are the same two, still deliberate:
      tests/test_gnl.py::test_the_correction_enters_at_first_order_in_the_load
      tests/test_wheel_fea.py::test_the_rim_band_holds_a_large_minority_of_the_compliance
  And the second one reproduces 0.032076694850181206 -- every digit of Step 0's value and
  of the one Sec 14 recorded two days before that.  The tree did not drift under this arc.

  Sec 14 had to close on an unconfirmed sum ("that arithmetic has NOT been confirmed by a
  full run").  This arc predicted its own count and then measured it, which is the whole
  difference and is worth the 24 minutes.
```

---

## Parked, on purpose

Named here so they are not lost, and deliberately **not** in this arc:

- **PLAN.md §14 item 4b** — the hub compliance share, 0.0321 against `< 0.03`. The one
  open item whose *sign* is not understood: thinner, floppier spokes should push
  compliance toward the spokes and the hub share down, and it went up. Hypothesis on file
  (`R_hub` 1.5598 → 0.5790), unmeasured.
- **PLAN.md §14 item 5** — `make hubcap` at the 1.2 mm floor. `HUB_CAP_SHARE` is
  calibrated on `t0 ∈ [2.0, 2.6]` and the shipped genome sits at 1.2. What saves it today
  is *which branch binds* — the thickness branch, at 0.52 × 1.2 = 0.624 mm — so the slot
  share is not actually being extrapolated. A design both thin and tight-slotted would
  extrapolate it for real and nothing would say so.
- **PLAN.md §14 item 6** — `EMBED_ALLOWANCE_PER_SPOKE_MM2 = 3.03` against a measured
  0.98 mm²/spoke. Do not guess a new number; what is needed is the scaling law, derived
  from `wheel_step_export._embed` the way `wheel_geometry.junction_bite` was.
  `solid.volume_nofillet_mm3` now makes the true gusset measurable per genome.
- **Gate 1 at a 1.2 mm floor.** §14 item 1: *"Re-deriving Gate 1 at a 1.2 mm floor is real
  work and has not been done."* `test_the_free_arc_fraction_is_not_constant_over_the_design_space`
  is the same shape and is the next one to move if the floor does.
- **`HUB_PLAN.md` is referenced by PLAN.md §3 and §0 but is not present in the tree.**
  Not this arc's problem; noted so the next reader does not go looking.

**Raised BY this arc, and its natural successors** — all three found in Step 5, all three
deliberately left alone because acting on any of them mid-arc would be re-fitting:

- **Re-derive the minimum-wall floor under SVK.** Run 1's answer is pinned on all four
  wall genes at 1.2 mm, and Step 3 showed the `minwall` sweep that chose 1.2 was run under
  linear kinematics with a ranking SVK inverts. The floor is load-bearing in the shipped
  result and its justification no longer holds. This is the single most valuable follow-on.

  > **STATUS 2026-08-13 — OVERTAKEN, NOT DONE.** The sweep was never re-run under SVK, so the
  > justification for 1.2 is still the linear-kinematics one. But the *premise* expired: once
  > PLAN §18 priced stress margin, the design chose to be thicker on its own, and **no wall
  > gene is on the floor in the wheel that now ships** (`e126cc3`: `t0` +359.4 µm, `t3`
  > +263.0 µm, `t2` +52.4 µm, `t1` +4.7 µm). The floor is no longer what stops the descent, so
  > this is no longer "the single most valuable follow-on" — it is a correctness item about a
  > constant that is currently slack. See PLAN §19.

- **Give the objective a stress-margin term.** `stress` is `soft_barrier(util - 1.0, 4000)`
  and is identically zero, with identically zero gradient, for every `util <= 1.0`
  (`src/wheel_objective.py:1027`, `:290`). The optimizer cannot see stress below the knee,
  so it will spend margin it is not charged for. Run 1 stopped at 0.899 only because the
  wall floor stopped it first, not because anything valued the remaining 0.101. **And it
  freezes two genes solid:** `R_hub` and `R_rim` reach the loss only through `stress` and
  the fillet barriers, both identically flat here, so they had zero gradient in 602 of the
  604 steps of Step 5 and `R_rim` has never moved in this repo's history. Fixing the
  margin term is also what makes the fillet radii optimisable at all.

  > **STATUS 2026-08-13 — DONE, PLAN §18 (`BUILD_PLAN.md` step 9).** `stress_margin = w * util**2`
  > per junction, `w` = 20.0 derived as a 1%-mass-for-1%-utilisation exchange rate, in
  > `OBJECTIVE_TERMS` so it prices without ever gating. The prediction above was exactly right
  > and was confirmed by direct measurement before the change: `dL/dR_hub` and `dL/dR_rim` were
  > both **exactly `+0.000000e+00`** at the shipped genome. Both are live now, FD-checked to
  > 1.8e-7, and `R_rim` moved for the first time in this repo's history — straight to its box
  > ceiling of 3.0, which is where it still sits. Two things this bullet did not anticipate:
  > the term found a measure-zero `smooth_min` tie-derivative bug, and `util**2` turns out to
  > have no knee (**defect 8**, PLAN §19) — the price of margin falls only 28% from `util`
  > 0.996 to 0.780, so the term never stops wanting margin and the design stops only when mass
  > catches up.

- **Make the objective see BUILDABILITY.** The arc's terminal finding: `bc77614` clears
  every FEA gate and then cannot be built at the stress concentration it was priced at
  (`kt_error_pct` +11.9%, as-built utilisation 1.046 against a modelled 0.935), because
  nothing in the loss prices the hub arrival angle or the fillet the exporter can actually
  cut. The exporter already computes `kt_built` — the missing piece is a term that charges
  the difference, which would also give `R_hub` its first real gradient. This is the one
  that has to be fixed before ANY SVK descent can ship, and it subsumes the wall-floor and
  stress-margin items below as far as promotion is concerned.

  > **STATUS 2026-08-13 — DONE, PLAN §16 (the whole of `BUILD_PLAN.md` steps 1–7).** This bullet
  > became its own seven-step arc. The differentiable cap is
  > `min(by_slot, by_thickness)` with `by_thickness = t0 * (0.505 - 0.48*(1 - cos arrival))`,
  > fitted conservatively under all fourteen OCC sweep stations, and it gave `cx1`/`cy1` a live
  > gradient into buildability. It worked: the wheel promoted in §19 builds **24/24 edges at
  > both junctions at the full requested radius, `kt_error_pct` +0.0%**, at a worst hub wedge
  > of 326° — harsher than the 314° corner the model was fitted against. The +11.9% that
  > closed this arc is gone.
  >
  > Two defects were found *inside* the fix and are not closed: `--best-out` selected by loss
  > and ignored feasibility (**defect 6**, fixed in §17 — it has since caught two real over-cap
  > promotions), and the quadratic barrier cannot hold a boundary against live opposition
  > (**defect 5**, open, and now the thing that limits every descent — §19 lost 45 of 100 steps
  > to it).

- **Put a mesh-convergence study on `axle_drop_mean_mm`.** `coarse` and `medium` disagree
  by ~1.7% on this wheel, and the ±0.3% deflection gate is satisfiable at exactly one rung
  — the coarse answer reads +1.65% at medium, the medium answer −1.71% at coarse. Give the
  deflection QoI the GCI treatment M8b-i.5 gave the stress QoI, then state the gate against
  the extrapolated value instead of against whichever rung the descent happened to run on.

  > **STATUS 2026-08-13 — STILL OPEN, and the gap has halved on its own.** No GCI study has been
  > done. But §19's five scheduled coarse-against-medium checks measured the disagreement along
  > a whole trajectory, and it shrinks as the design gets thicker: **−1.59% at `e4219f3`**
  > (1.96822 coarse against 1.99996 medium) and **−0.86% at `e126cc3`** (1.99079 against
  > 2.00806). Still not a ±0.3% gate at both rungs, so the item stands.
  >
  > **A caution for whoever picks it up:** mesh refinement is now the *smaller* of the two
  > deflection errors. PLAN §19 measured the assumed-3.0°-contact-patch model at **5.27%**
  > against real contact on the promoted wheel, up from 3.08% on its predecessor — six times
  > the mesh disagreement, and growing as the optimizer works. Extrapolating the deflection QoI
  > to zero mesh size converges on the wrong answer more precisely unless the patch model is
  > fixed first. That is why the contact model, not this study, is §19's successor #1.

  > **CLOSED 2026-08-14 — DONE, and the answer is that the gate cannot be adjudicated. See
  > PLAN.md §29.** `make gci` / `studies/study_deflection_gci.py` ran the ladder on the gate's
  > own QoI (`axle_drop_mean_mm`, 8-phase, SVK): the observed order is **p = 0.638** and the
  > **GCI on the finest rung is 2.749%, nine times the ±0.3% band** — so the band was never
  > deciding anything about the design. Extrapolated deflection **2.05700 mm (+2.850%)**; every
  > rung flatters the wheel. The verdict holds under all four defensible definitions of `h`:
  > the SMALLEST GCI any of them gives is 2.42%, already 8× the band.
  >
  > *(Numbers corrected 2026-08-15. The first version of this entry read p = 0.502 / GCI 2.804%
  > / 2.05789 mm, from a study that took its cell size from `wheel_mesh`'s spoke-block ladder
  > while solving on `wheel_wheel`'s 12-sector wheel. The gate verdict is unaffected — see
  > PLAN §29 for why the GCI barely moved while p moved 25%.)*
  >
  > **The caution above was right about the ranking and wrong about the conclusion.** Mesh
  > refinement being the smaller error does not make it a small error: at 2.8% it already
  > swamps the gate. And the "gap has halved on its own" note is falsified — §26's scheduled
  > checks put it at **−1.077 pp** on the shipped genome, up 25% from `e126cc3`'s −0.863 pp.
  >
  > The absolute band is retired in favour of a same-rung comparison against the incumbent
  > (PLAN §29's call). Earning it back needs the junction FILLETED IN THE FEA MODEL — Q9
  > elements give p ≈ 2 on a smooth solution and this ladder gives 0.638, the patch control in
  > `study_wheel_fea` rules out the other candidate, and the FEA model is sharp exactly where
  > the exported solid is round (24/24 filleted, `kt_error_pct` +0.0%).
  >
  > **The exponent does NOT confirm the corner, and the claim that it did is retracted
  > (2026-08-15).** Williams' eigenvalue for the junction's 322°/320° material wedge is
  > λ = 0.5030, and the corrected measurement is p = 0.638 — they do not agree. The reported
  > agreement was the `h` error above, exactly: `0.5023 × ln(1.826)/ln(1.616) = 0.628`.
  >
  > **The corner is real, but it never needed this study to establish it — PLAN §30.** M4
  > already had it, `test_peak_stress_diverges_but_the_field_converges` has pinned it since
  > §14, and M8b-i.6 step 2 rebuilt the stress constraint around it. `make corner` (8.5 s) adds
  > the RATE — peak von Mises 61.9 → 99.1 → 126.3 → 150.6 MPa, slope −0.44 on log h where a
  > convergent peak gives zero — plus the four corners individually and their mesh-measured
  > wedge angles (321.1° / 296.8° / 321.3° / 307.9°, Williams λ = 0.503–0.514 against 0.38–0.56
  > from the rates: the mechanism, not the number). **Filleting the FEA junction stays the top
  > item.** The deflection order `p` was simply the wrong instrument for the question, and the
  > right one had been sitting in the test suite the whole time.

- **Set the load-control tolerance from the inner solve's noise floor.** `tol_rel=1e-8` in
  `solve_wheel_contact` (`src/wheel_fem.py:1842`) is not universally achievable under SVK:
  run 2 stalled at 5.2e-7 relative, 52x above it. Measure the floor, then set the outer
  tolerance from the measurement — do not pick a looser round number.

  > **STATUS 2026-08-13 — STILL OPEN, untouched.** Nothing since has measured the inner solve's
  > noise floor. No SVK descent since has stalled on it either: §19's 100-step `medium` run
  > exited 0 with no convergence events, so the item is real but has not been costing anything
  > observable.

---

### Where the five successors stand (2026-08-13)

| successor | status |
|---|---|
| Make the objective see BUILDABILITY | **DONE** — PLAN §16, `BUILD_PLAN.md` steps 1–7 |
| Give the objective a stress-margin term | **DONE** — PLAN §18, `BUILD_PLAN.md` step 9 |
| Re-derive the min-wall floor under SVK | **OVERTAKEN** — the floor is slack on the shipped wheel |
| Mesh-convergence on `axle_drop_mean_mm` | **OPEN** — gap halved to −0.86%; a larger model error now dominates |
| Load-control tolerance from the noise floor | **OPEN** — untouched, and not observably costing anything |

This arc ended with nothing promoted. The two successors it named as prerequisites were both
built, and on 2026-08-13 the first promotion since — `e4219f3` → `e126cc3`, PLAN §19 — went
through on the strength of them: buildable at 24/24 edges with +0.0% `Kt` error, and hub
utilisation down from **0.9964 to 0.7795**. The genome this arc could not ship, `bc77614`, was
blocked by exactly the two things those successors fixed.

**Read `PLAN.md` §16–§19 and `BUILD_PLAN.md` for anything after 2026-08-10. This file is
closed.**
