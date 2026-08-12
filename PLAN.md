# PLAN.md — the next changes

> **Ignore version control entirely. Do not commit, branch, stage, revert or otherwise
> touch git — it is not part of this project's workflow and nothing here depends on it.**

> **THE SHIPPED GENOME CHANGED ON 2026-08-06 — READ THIS BEFORE TRUSTING ANY NUMBER BELOW.**
> `best_solution.json` is now the Stage-3 1.2 mm optimum, genome **`350f4c7`**, 39.194 g on
> the mesh / 47.63 g as an OCC solid. Everywhere in §1–§12 that a table, a study or a
> sentence says **"best_solution"** it means the genome that used to be in that file: the
> v2.1 GA/beam optimum, **`36aed36`**, `t0` 2.48 mm, 74.12 g. That genome is preserved
> unchanged as `best_solution_ga_beam.json`, and **none of those measurements are wrong** —
> they simply describe *the old wheel* now. §9's 1.36 load factor and §0's utilisation table
> are both in that category. Every driver in `studies/` defaults to `best_solution.json`, so
> **re-running any study today measures the new wheel and will not reproduce the old page.**
> §13 is the record, and it lists the numbers that moved.
>
> **AND: `wheel_fea.MIN_WALL_MM` DEFAULTS TO 1.2 AS OF THE SAME DAY**, down from 2.0. The
> shipped genome is a 1.2 mm design and the old default put all four of its thickness genes
> outside the box they are supposed to live in. **The GA and every driver now search down
> to 1.2 mm by default** — if you are reading a run whose numbers assume a 2.0 mm floor,
> that run predates this. See §13.
>
> **AND, 2026-08-10: EVERY STAGE-3 NUMBER IN §1–§14 IS A LINEAR-KINEMATICS NUMBER.**
> `wheel_contact_problem` defaults to `kinematics="linear"` and nothing in the Stage-3 path
> ever overrode it. The shipped wheel deflects **2.409 mm, not the 1.953 the optimizer saw**,
> and carries **0.875 of allowable, not 0.799** — measured, at `medium`, and it is **still
> feasible** with every barrier at 0.0. None of those numbers is wrong; they are answers to a
> different question. **THE SHIPPED GENOME DID NOT CHANGE** — `best_solution.json` is still
> `350f4c7`, because the wheel the SVK descent found clears every FEA gate and then **does
> not build** (`kt_error_pct` +11.9% at the hub, as-built utilisation 1.046). §15 is the
> record and `SVK_PLAN.md` is the evidence. Linear remains the default everywhere on purpose;
> `--kinematics svk` is opt-in on `wheel_stage3.py` and `study_gradient.py`.

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

**Gates, all green:** `make test` **427 passed** (425 before §8's second pass; 406 before
the wall-floor work; 383 at M8b-ii; 357 before the phase pool; 269 before M8b-i.6, whose
Kt-twin equivalence test is parameterised 84 ways). Gate 7 `min_decades`
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

### 0. WHAT TO DO NEXT

**READ §8, §9 AND §10 FIRST — THEY CLOSED ITEMS (1) AND (2) AND CHANGED WHAT (3) IS.**
One line each:

- **§8 — `MIN_WALL_MM` is priced, and the floor DOES stop binding — at the ends, near
  1.5 mm.** EIGHT 125-step arms, 2.2 mm down to 0.8 mm. `t3` lifts off the floor at
  1.4 mm and `t0` at 1.0 mm, both settling on floor-independent values (**`t0 ≈ 1.45`,
  `t3 ≈ 1.6`**); `t1`/`t2` stay pinned at every floor. **The first pass's headline
  (`29.5 g/mm`, "never stops binding") is RETRACTED** — it was fitted entirely inside the
  band where all four genes were on the bound, and the marginal cost actually falls 4x
  across the full band (~31 → ~20 → ~8 g/mm). 2.0 → 1.2 mm is worth 19.4 g; below 1.2 mm
  is worth 3.3 g. Item (2) is CLOSED.
- **§9 — M9 phase 3: the load factor is mesh-convergent and is NOT a safety factor.**
  `λ(f) > 1` at all 11 load levels out to **4× service**, zero solver refusals, so there
  is no fixed point and the "1.36" recedes as fast as the load advances. Phase 3 stays
  blocked, now for a MEASURED reason. Item (1)'s successor is CLOSED as a question.
- **§10 — item (3) is DEFERRED ON PURPOSE, and the golden test has been decoupled from
  it.** §8 makes elite 10 optimal *at the 2.0 floor only*, so which genome should ship
  now depends on a floor decision that has not been made. `best_solution.json` is
  untouched; `tests/test_golden.py` now reads `best_solution_ga_beam.json` so a future
  promotion is a one-file change that cannot re-baseline the regression net.
- **§11 — THE FLOOR DECISION LANDED (2026-08-05): the process can hold 1.2 mm.** So the
  candidate is `stage3_minwall_best_1.2.json`, **39.194 g**, and §10's two prerequisites
  are done: the `medium` re-score confirms the mass to five figures with every barrier
  still at 0.0, and the export is OCC-valid with **`kt_error_pct` = +0.0% at BOTH
  junctions** — the first genome whose built fillets match the ones its stress model
  priced. **It is still NOT promoted**: the exporter's `[WEAK JUNCTION]` check fired on
  the hub (18.12 mm³ against a 50 mm³ floor) and the smallest edge fell to 0.0087 mm.
  Deepening the embed is measured and does NOT fix it. Read §11 before touching
  `best_solution.json`.
- **§12 — THAT CHECK WAS MEASURING `t0`, AND IT IS FIXED (2026-08-05).** Junction overlap
  is quadratic in the root thickness, so the fixed 50 mm³ floor ranked genomes by how
  thick they were and called it a verdict on the weld — which is why **elite 10 failed it
  too, at 48.72 mm³**. Normalised, the three genomes measure **0.562 / 0.544 / 0.571**
  root thicknesses of penetration across a 2× range in `t0`: the same junction, three
  times over. `MIN_JUNCTION_OVERLAP_MM3` is gone, replaced by
  `wheel_geometry.junction_bite` and `MIN_JUNCTION_BITE = 0.25` — **a geometric floor, not
  a calibrated one**, since no junction in this repo has ever actually failed. Still a
  warning, never a raise. Three tests where there were none. Both genomes re-exported and
  the manifest diff proves it is reporting-only: **not one geometric number moved.**
  **`make test` 430 passed.** Promotion now waits on exactly one thing — the Inventor
  import of `export/stage3_minwall_best_1.2.step`.
- **§13 — PROMOTED (2026-08-06). That import passed, and `best_solution.json` is now
  `350f4c7`.** The shipped wheel is the 1.2 mm Stage-3 optimum: **74.12 g → 47.63 g as an
  OCC solid**, `kt_error_pct` **+0.0% at both junctions** (the first shipped part whose
  fillets match the ones its stress model priced), one fillet family per junction instead
  of two. The export is identical to §11's candidate in all 33 geometric keys. The price is
  stated plainly in §13: **stress utilisation 0.41 → 0.80** at the `medium` rung, still
  inside the barrier with every constraint term at exactly 0.0, but no longer comfortable.
  `best_solution_ga_beam.json` keeps the old genome and `tests/test_golden.py` keeps
  reading it, so nothing was re-baselined. **Every per-design number in §1–§12 now describes
  the old wheel** — see the banner at the top of this file.
  **`make test` came back 410 passed / 20 failed, and TEN OF THOSE WERE FINDINGS, NOT
  PINS.** The most important was that `MIN_WALL_MM` still defaulted to **2.0** while the
  shipped genome is a 1.2 mm design, putting all four thickness genes outside the default
  gene box — **the default is now 1.2**, which closed three failures on its own. The gate
  stands at **423 passed / 8 failed**. Read §13's gate section before running any driver on
  this genome.
- **§14 — those eight triaged (2026-08-06). Three were about the TESTS, and the
  "eight findings about a thinner wheel" reading was half wrong.** Every one was diagnosed
  by sweeping something — the mesh tier, the gene box, the reference — on the promoted
  genome AND on `best_solution_ga_beam.json`, so "the new design broke it" always had to
  survive a comparison against the design it replaced. Three times it did not: the area
  order is **exactly 2.000** on both genomes once measured self-referenced instead of
  against a beam integral that has its own error floor; the p99 ratio test is a divergence
  detector misfiring on an already-converged quantity, and the OLD genome fails it too one
  tier down; the contact failure is smoke-only and clears by 5x at `coarse`. Two more were
  tests pinning a **pathology** as an invariant. **Six thresholds touched, four of them
  now TIGHTER.** The exporter publishes `fillets.volume_mm3` (**6.18% of the solid**, not
  the 0.92% its docstring claimed), which turned the mass budget from a fitted band into a
  named one — and exposed a new open item: `EMBED_ALLOWANCE_PER_SPOKE_MM2 = 3.03` is a
  `t`-proxy and the real gusset is **0.98 mm²/spoke**. **Left deliberately red: the
  pre-registered GNL gate and the hub compliance share** (0.0321, sign not understood).
  Also still open: `make hubcap` at 1.2.
- **§14 item 4a — THE GNL GATE STANDS, and chasing it down found the biggest thing on this
  page.** The 1%-load gate is a tripwire for the service-load number, and the full ladder
  says the shipped wheel's axle drop at service load is **+23.3% under SVK against linear**
  (the GA/beam wheel: +3.95%, reproducing M5 exactly). Every headline for this genome was
  computed under `kinematics="linear"`. **Deflection is 2.409 mm, not the 1.953 the optimizer
  saw — 20% past the 2.0 mm target it was tuned to hit exactly**, and utilisation goes from
  0.799 to roughly **0.91** against an allowable of 1.0. Relaxing the gate would silence the
  only automatic warning that linear kinematics no longer describes this part; **the real
  question is whether Stage 3 should descend on a linear solve at a 1.2 mm wall at all**, and
  that is a scope decision. Also fixed en route: `study_wheel_fea.stress_report` was applying
  the LINEAR strain formula to SVK fields — a **+169.5% artefact against a real +14.3%** — and
  now dispatches on `res["meta"]["kinematics"]`.
- **§15 — STAGE 3 CAN NOW DESCEND UNDER SVK, AND THE WHEEL THAT DESCENT FINDS CANNOT BE BUILT.
  NOTHING PROMOTED (2026-08-10).** Working notes in `SVK_PLAN.md`. **Before this milestone
  every Stage-3 number in this repo was a linear-kinematics number** — read §15 before
  quoting a deflection or a utilisation out of §1–§14. The adjoint was proved correct under
  SVK first (**all ten M7 gates, thresholds unmodified**, including G1's unrolled-Newton
  check at 5.9e-11 against 1e-8); SVK costs **1.36×** time and 1.05× memory, and the penalty
  is in the gradient, not in Newton iterations. Re-scoring six designs at `medium` says the
  shipped wheel **IS still feasible** (util **0.875**, not §14's estimated ~0.91, every
  barrier 0.0) but that **the design ranking INVERTS** — which reverses §8's `minwall` sweep
  over the whole 1.2–2.0 mm range. Two 300-step SVK descents both cleared every
  pre-registered clause, and a `medium` re-convergence (`bc77614`) is **1.78 g lighter and
  12× closer to the deflection target** than the incumbent under the honest kinematics. **It
  is not promoted: it does not build.** `kt_error_pct` **+11.9%** at the hub, as-built
  utilisation **1.046** against a modelled 0.935, where the incumbent is +0.0% — a regression
  this arc introduced, caught by the control. The cause is **four measured defects in the
  objective**: `stress` has identically zero gradient below util 1.0, so `R_hub` and `R_rim`
  are **dead genes** (nonzero gradient on 2 of 602 steps), so nothing objected when the
  descent swung the hub arrival shallow — and the 1.2 mm floor, not the stress margin, is
  what stopped the run. **`best_solution.json` is UNCHANGED and still holds `350f4c7`.**


**The previous three items — (a) the hub-fillet cap, (b) the production descent, (c)
`make m9` in full — are ALL DONE. §5, §6 and §7 are their records.** A one-paragraph
summary of each is at the end of this section; read those three sections for anything real.
What follows is the new list, and it is ordered by *cost to decide*, not by importance.

**The new items are NUMBERED (1)(2)(3) and the closed ones keep their LETTERS (a)(b)(c)**,
because §5, §6 and §7 are titled after the letters and a fresh session reading "§0(b)" must
land on the production descent, not on the wall-thickness sweep. Letters are history;
numbers are open.

**~~(1) SETTLE WHY `lambda_min` IS MESH-DEPENDENT.~~ DONE. BOTH HYPOTHESES RUN. The answer
is that `study_m9`'s `K_t` is not a tangent at all — and a mesh-convergent replacement
exists and was measured.** Full record in H1 and H2 below; the two-line version:

- **`kinematics="linear"` is the default on `wheel_contact_problem`, so `prob.nonlinear` is
  False and the displacement passed to `assemble_stiffness` is IGNORED — measured at exactly
  0.000e+00 change.** §7's quantity is λ_min(K_linear + K_contact), which has no geometric
  stiffening in it. Every symptom §7 recorded follows from that one fact.
- **The generalised load factor converges: 1.378129 / 1.359846 / 1.356669 at smoke / coarse /
  medium**, error ~dof^−1.87, Richardson limit ≈1.3560, `medium` within 0.049% of it. It
  passes `GATE_MESH_REL` with 21× margin where λ_min fails it by twelve.

**What is now open is phase 3 itself**, plus four things H2 did NOT establish — see the end
of H2. **The load factor came out ≈1.36 on one design, which is alarmingly tight and must not
be quoted as a safety factor until it is independently checked.**

The original framing follows, because the reasoning is what made the cheap test worth running:

**~~H1 — THE CONTACT PENALTY.~~ RUN, AND REFUTED. It also corroborated H2.** `best_solution`,
`coarse`, phase 0, `eps_n` over two decades, ~2 min. Three independent discriminators, all
negative:

| eps_n | λ_min | vs default | λ_min, contact block REMOVED |
|---|---|---|---|
| 1e3 (0.1×) | 4.860825e-03 | 1.0069× | 4.703656e-03 |
| 3e3 | 4.894088e-03 | 1.0138× | 4.703656e-03 |
| **1e4 (default)** | **4.827619e-03** | 1.0000× | 4.703656e-03 |
| 3e4 | 4.851094e-03 | 1.0049× | 4.703656e-03 |
| 1e5 (10×) | 4.882208e-03 | 1.0113× | 4.703656e-03 |

1. **`eps_n` varied 100×; λ_min varied 1.0044×** — and *non-monotonically*, so even that
   0.4% is solve-to-solve variation rather than a trend. A penalty mode tracks its penalty
   roughly linearly. This does not track it at all.
2. **Ablating the contact block entirely moves λ_min by 2.6%**, and the ablated value is
   identical to 7 digits at every `eps_n`. The contact term is a couple of percent of the
   quantity; it is not the quantity.
3. **The eigenvector is a GLOBAL mode.** 5.05% of its energy sits on `rim_outer`, which is
   2.28% of the nodes — concentration factor 2.2, i.e. essentially none. The **top 10 nodes
   hold 0.11%**. Participation ratio **0.54**: the mode is spread over roughly half of every
   node in the mesh.

**Point 3 is why this was worth the two minutes even though it came back negative.** It is
independent corroboration of H2 from a different direction: λ ∝ h² is what the smallest
eigenvalue of an unnormalised discrete elliptic operator does, and "a global mode spread over
half the mesh" is what that mode looks like. The scaling exponent and the eigenvector now
agree. **H2 is the remaining explanation and it is no longer competing with anything.**

The driver is `eps_n_check.py`, written to a scratchpad rather than to `studies/` on purpose:
it is a one-question falsification with no gate and no artifact anyone should depend on, and
its answer is this table. Re-deriving it is ~2 min if it is ever doubted.

**~~H2 — THE FORMULATION.~~ RUN, AND CONFIRMED. A mesh-convergent buckling quantity EXISTS,
and it is cheap.** But the check that came with it found something bigger, so read that first.

##### H2(a) — `study_m9`'s "K_t" IS NOT A TANGENT. It has zero geometric stiffening in it.

`wheel_contact_problem` defaults to **`kinematics="linear"`**, so `prob.nonlinear` is
**False**, and `study_m9.measure()` never overrides it. `assemble_stiffness`'s own docstring
says the linear Hessian "is independent of `u`, so `u=None` (evaluate at zero) is exact
rather than an approximation." **Measured, not inferred** — assemble at the converged `u`
and at zero and diff:

| config | \|u\|max | K(linear) change with u | K(svk) change with u |
|---|---|---|---|
| smoke | 1.6153 mm | **0.000e+00** | 2.732e-02 |
| coarse | 1.6663 mm | **0.000e+00** | 8.769e-03 |

**Exactly zero.** The `res["u"]` that `study_m9` threads into `assemble_stiffness` is inert;
the kernel ignores it. So the quantity §7 measured is **λ_min(K_linear + K_contact)** — the
smallest eigenvalue of a fixed linear elliptic operator plus a penalty block. It was never a
tangent eigenvalue, and it contains **no buckling information by construction.**

That single fact explains every symptom §7 recorded, with nothing left over:
- **h², to within 3%, nine times** — that is what λ_min of a fixed linear operator does.
- **1.022× over a 4× load ladder** — the only load path into it is the contact block, and H1
  measured that block at 2.6% of the value.
- **A global mode, participation ratio 0.54** — a bulk discrete mode, not a structural one.

The three findings were never independent. They are one defect seen from three sides.

##### H2(b) — the generalised load factor, and it converges

Under **SVK kinematics** so a geometric term exists at all: `K_0 = K(u=0)`,
`K_t = K(u_service)`, `K_g = K_t − K_0`, then `(K_0 + λ K_g)x = 0` solved as the generalised
symmetric problem `K_g x = μ K_0 x` with `λ = −1/μ`. `λ` is a dimensionless **load factor** —
how many times the service load until the tangent goes singular.

| config | reduced dof | **load factor** | vs previous | `study_m9`'s λ_min, same mesh |
|---|---|---|---|---|
| smoke | 8,904 | 1.378129 | — | 2.275954e-02 |
| coarse | 41,064 | 1.359846 | **−1.327%** | 4.827619e-03 (−78.8%) |
| medium | 104,712 | 1.356669 | **−0.234%** | 1.921939e-03 (−60.2%) |

**Converging, and fast.** Successive changes are −0.018283 then −0.003176, ratio **5.76**, so
the error goes as **dof^−1.87**. Richardson-extrapolating puts the limit at **≈ 1.3560**, and
`medium` is **0.049%** from it. Against `GATE_MESH_REL = 0.05` the coarse→medium step passes
with **21× of margin**, on the same meshes where λ_min(K_t) fails it by twelve.

**So M9 phase 3 has a quantity.** It is dimensionless, mesh-convergent, physically meaningful,
constructible from existing functions with no new assembly, and costs 3.3 / 23.9 / 71.4 s at
the three rungs. That is the whole of what item (1) set out to establish.

##### What this does NOT establish — four things, before anything is built on it

1. **One design, one phase.** `best_solution` at phase 0.0 only. §7 measured λ_min's phase
   spread at 0.66–1.59%; the load factor's is **unmeasured**, and so is its spread across the
   16 elites. Both are needed before a threshold or a phase aggregation rule.
2. **THE LOAD FACTOR IS ≈ 1.36, AND THAT IS ALARMINGLY TIGHT.** It says this wheel buckles at
   36% above service load. That is either a real and important design finding — the Euler
   `buckling` proxy has been reporting ratios of 0.066–0.087 and would not have shown it — or
   the `K_g ≈ K_t − K_0` approximation is off. **Do not report 1.36 as a safety factor until
   it is checked against something independent.** It is the most consequential number this
   session produced and the least corroborated.
3. **`K_g` is linearised about the SERVICE state, not the reference state.** Textbook
   eigenvalue buckling prestresses `K_g` at a reference load and scales it linearly; this
   forms it at the converged service state instead. For a load factor near 1 the two nearly
   coincide, which is convenient here and would stop being true for a stiffer design.
4. **Contact is excluded from `K_0` and `K_g`.** Buckling is computed on the bulk only, while
   the load actually arrives through the contact patch. H1 puts the contact block at 2.6% of
   the old quantity; its effect on the *load factor* is unmeasured.

**Both hypotheses were worth running, and the cheap one paid twice.** H2 was always the more
likely — h^1.94 to h^2.03 across nine refinements is a bulk discrete mode, not a localised
penalty — but that was **inference from a scaling exponent**, and inference is what
`stress_scale` was. H1 died in two minutes and its eigenvector diagnostic turned out to be
positive evidence *for* H2 rather than mere absence of evidence against it; then H2's own
setup surfaced H2(a), which no amount of reasoning about eigenvalue scaling would have found.

Drivers: `eps_n_check.py` and `h2_check.py`, both in a scratchpad rather than `studies/` —
one-question falsifications with no gate and no artifact anything should depend on. Their
answers are these tables. Promoting `h2_check.py` into a real study driver is the FIRST
piece of phase-3 work, because items 1–4 above all need it.

**(2) PUT A NUMBER ON `MIN_WALL_MM`. The only item here that changes the WHEEL rather than
the model.** §6 measured that all four thickness genes pin to the 2.0 mm floor at both
production answers, so a manufacturing constant — not the FEA, not the deflection target, not
the stress constraint — sets **4 of the 14 genes**, and every gram below 58.660 is on the far
side of it. It sits on "The decision that is a human's" with **no quantification at all**,
which is not a decidable state.

  **The experiment: short descents at `MIN_WALL_MM` ∈ {1.6, 1.8, 2.0, 2.2}**, from the elite-10
  answer rather than from `rank:10`, and **well short of 300 steps.** Measured off elite 10's
  record, distance from its own final loss: **step 100 is +0.190%, step 125 +0.096%, step 150
  +0.048%, step 200 +0.011%.** So ~125 steps costs ~1.6 h and resolves to a tenth of a
  percent — far finer than a 0.2 mm floor change will move the mass. That turns "should the
  floor come down?" into "0.2 mm of floor buys N grams."

  **`MIN_WALL_MM` is NOT a parameter, and this is a code change before it is a run.** It is
  `src/wheel_fea.py:219`, and it is consumed at **import time** by the `GENE_BOUNDS` list
  literal at lines 259–262 (`t0`/`t1`/`t2`/`t3` low bounds). So it cannot be varied inside one
  interpreter without rebuilding the bounds; the sweep is either four separate processes with
  the constant overridden per process, or a small change making the floor an argument that
  `GENE_BOUNDS` is built from. **Prefer the latter and drive the sweep from one place** —
  editing a module constant four times by hand is how a run gets misattributed to the wrong
  floor. Four points, sequential and capped, is ~6.5 h; see the memory rules below.

**(3) DECIDE WHETHER THE PRODUCTION GENOME BECOMES THE SHIPPED GENOME. A human's call, and
it is the pending consequence of §6.** `best_solution.json` is still the GA optimum for the
BEAM surrogate — the genome this very file calls "a bad guide to the FEA," sitting at
**−25.43% deflection error** — and **both environments and every study driver read it**.
`stage3_prod_best_elite10.json` is 58.660 g, inside the feasible box on both constraints,
with every barrier at exactly 0.0.

  Promoting it is not a file copy. It needs `make export` (~4 min) and then a look at whether
  the hub fillets still build on the new geometry: §3's ladder and §5's cap are both
  genome-dependent, the new `R_hub` is **0.9666 against a cap of 1.0400** (under it, which is
  the good direction), and `kt_error_pct`'s +73.4% was set by a shallow near-cusp corner the
  cap does not model. Nobody has looked at 58.660 g of geometry in OCC. Expect the manifest
  to move, and re-measure rather than assume.

**HOUSEKEEPING, stated because it was skipped rather than passed.** `make test` has NOT been
run since §7's changes to `studies/study_m9.py`. Nothing outside the `Makefile` imports that
module — verified by grep over `tests/`, `src/` and the `Makefile` — and `--quick` exercised
every changed path before the full run, so the exposure is close to nil. It is still the
repo's gate and it is still unrun. ~22 min.

#### What just closed, in one line each

- **(a) `R_hub` vs the buildable ceiling — §5.** The constraint learned the cap rather than
  the bound coming down to 1.1: the void is a function of arrival angle and `t0`, spanning
  **0.9898–1.5265 mm** across the 16 elites, so a fixed bound would have been right for
  exactly one genome. The 4.0 box bound is untouched.
- **(b) The production multi-start descent — §6.** `make prod9` / `make prod10`. **73.689 →
  58.715 g (elite 9, −20.3%)** and **70.937 → 58.660 g (elite 10, −17.3%)**, every barrier at
  exactly 0.0, deflection error inside ±0.3% on both. The two land **0.09% apart in mass with
  1.8 mm of daylight between their spoke centerlines** — the optimum is a **valley, not a
  well**, so more starts are pointless and the leftover geometric freedom is the interesting
  part.
- **(c) `make m9` in full — §7.** **OVERALL: FAIL, and the failure is the result.** λ_min(K_t)
  has no mesh-independent value; phase 3 cannot be built as specified. `buckling` stays inert,
  and that is now a deliberate hold rather than an oversight — item (1) above is how it gets
  unblocked.

**THE MEMORY RULES BELOW DESCRIBE THE OLD 16-CORE / 31 GB BOX — see §8 for this one.**
The current machine is **24 cores / 61 GB**, one descent measured at a **13.56 GB peak**,
and three arms ran concurrently under 15 GB caps with no oomd event. What did NOT travel
is the speedup: three concurrent descents run at **~95 s/step each against ~38-47 s alone**,
because the binding resource is **memory bandwidth** (twelve concurrent `spsolve` LUs),
not cores or capacity. Keep the `systemd-run` caps regardless — they make a failure local
instead of taking the desktop's whole `user@1000.service` slice with it.

**THE ORIGINAL RULE, AS MEASURED ON THE OLD BOX:** A descent at `coarse` with 4 workers
sits flat at **~12.7 GB anonymous** with the fidelity check off; **two do not fit in 31 GB**,
so starts run **SEQUENTIALLY** and **capped** (`systemd-run --user -p MemoryMax=20G`). §1's
"two starts × 4 workers beats one start × 8" is CPU arithmetic that ignores memory, and
acting on it is what took the desktop down. The 17 GB quoted there was measured with the
fidelity check ON and before the `uniform` finding; `PROD_FIDELITY` now defaults to **0**.
`--phase-scheme uniform` is a **correctness** setting for the run, not a preference — see §1.

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

**M9 PHASE 2 HAS RUN IN FULL (§7), AND §0(1) THEN FOUND WHY IT FAILED — THIS SECTION'S TITLE
IS NOW WRONG.** `lambda_min(K_t)` diverges under refinement at a clean h², 28.8× across the
ladder against 1.02× across a 4× load ladder, with an independent `eigsh` cross-check
(3.4e-10) ruling out the solver. The cause is that **there is no `K_t`**:
`wheel_contact_problem` defaults to `kinematics="linear"`, so the displacement threaded into
`assemble_stiffness` is ignored — measured at exactly 0.000e+00 — and the quantity is
`lambda_min(K_linear + K_contact)`.

So the paragraph above still describes the *problem* correctly — `buckling` is inert and a
diverged tangent is the only real signal today — but **"via LOBPCG" was never the hard part
and the standard eigenproblem was the wrong question.** The replacement is the generalised
`det(K_0 + λ·K_g) = 0` under SVK kinematics, whose load factor **is** mesh-convergent
(1.3560 in the limit, `medium` within 0.049%, 21× inside `GATE_MESH_REL`). §0(1) H2 is the
measurement and lists the four things it does not yet establish.

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

### 7. §0(c) — `make m9` in full. RAN. **OVERALL: FAIL, and the failure is the result.**

2930.7 s (49 min), 3.1 GB peak against a 22 GB cap — `fine` is 261k dof but this is a 2D
problem and memory was never the constraint. All four sections completed; `study_m9.json` is
whole (`complete: true`). Every section reports FAIL, and **they do not fail for the same
reason** — one is physics and three are a placeholder constant. Separating those is the work.

> **SUPERSEDED IN ITS DIAGNOSIS, NOT IN ITS NUMBERS — read §0(1) H2(a) with this section.**
> Everything measured below stands. What changed is *why*: `wheel_contact_problem` defaults
> to `kinematics="linear"`, so `prob.nonlinear` is False and the displacement this driver
> threads into `assemble_stiffness` is **ignored — measured at exactly 0.000e+00 change**.
> The quantity below is therefore `λ_min(K_linear + K_contact)`, not a tangent eigenvalue at
> all, and it contains no geometric stiffening by construction. The h² scaling, the 1.022×
> load response and the global eigenvector are not three findings — they are one defect seen
> from three sides. A generalised load factor formed under SVK kinematics **does** converge
> (1.3560 in the limit, `medium` within 0.049%); §0(1) H2(b) is that measurement.

#### THE FINDING: `lambda_min(K_t)` has no mesh-independent value. It scales as h².

The `fine` rung had never been run. It is what makes this measurable:

| best_solution, phase 0 | dof | λ_min |
|---|---|---|
| smoke | 8,904 | 2.275954e-02 |
| coarse | 41,064 | 4.827619e-03 |
| medium | 104,712 | 1.921939e-03 |
| **fine** | **261,864** | **7.898831e-04** |

Nine refinement steps across three designs give **λ ~ dof^−0.970 to dof^−1.014**, i.e.
**h^1.94 to h^2.03**. That is h², to within 3%, every time, with no sign of settling.
`last_pair_rel` is 0.589 against `GATE_MESH_REL = 0.05` — off by twelve, and flat.

**It is not a solver artifact, and that is the load-bearing check.** `measure()` runs an
independent `spla.eigsh` shift-invert reference at the first and last rungs, and it agrees
with LOBPCG to **2.668e-12 at smoke and 3.377e-10 at fine**. The number is genuinely
`λ_min(Kr)`. What diverges is the *quantity*, not the computation of it.

**Three spreads, and they settle the question:**

```
mesh ladder, ONE design, smoke -> fine        28.8x
design space, 16 designs at fixed mesh         1.457x
load ladder, ONE design, 4x the force          1.022x
```

**λ_min varies 28.8× with element size, 1.46× across the entire Stage-2 design space, and
1.02× across a 4× load range.** A buckling indicator is supposed to be dominated by load and
approach zero as the structure approaches its critical point. This one is dominated by the
mesh and is nearly indifferent to force — and over that 4× ladder it *rises* (4.723957e-03 →
4.827619e-03), which is the wrong sign for compressive geometric softening. It also tracks
stiffness rather than stability: against `1/axle_drop` over the 16 designs the correlation is
+0.91, though that is carried by one outlier (elite_11, axle drop 0.4305 against ~2.5) and
falls to **+0.45** without it, so read it as suggestive rather than established.

**This is M4 repeating, and §2 is the thing it lands on.** M8b-i.6 rewrote the stress
constraint because `c = max/pnorm` was anchored to a crack-tip singularity that diverges
under refinement. **M9 phase 3 proposes promoting λ_min(K_t) to a constraint with a margin, a
threshold and a phase aggregation rule.** A threshold calibrated against this number would be
`stress_scale` a second time: right at the mesh it was fitted on and meaningless at any other.
**Phase 3 is BLOCKED on a reformulation, not on more measurement.**

The likely cause is the formulation rather than the code. λ_min(K_t) is solved as a
**standard** eigenproblem, so it carries the dimensions of stiffness and its spectrum shrinks
with element size — h² is exactly what an unnormalised discrete operator's smallest eigenvalue
does. Classical linear buckling is **generalised** — `det(K_0 + λ·K_g) = 0` — and returns a
dimensionless *load factor* that is mesh-convergent by construction. That is the shape of the
fix, and it is a change to what is being asked, not to how well it is answered. **This is a
hypothesis from the scaling, not a measured result**, and it should be checked at two rungs
before anything is built on it.

#### The other three FAILs are one unmeasured constant, and Phase 2 existed to measure it

`phase`, `load` and `design_space` have **zero error rows and every λ finite**. They fail
solely on `lobpcg_converged`, which is `residual_rel <= wheel_adjoint.LOBPCG_RESIDUAL_REL`,
and that constant is **1.0e-7** carrying this comment:

```
LOBPCG_MAXITER = 200      # unmeasured starting point — Phase 2 measures iteration counts
LOBPCG_TOL = 1.0e-8       # explicit Phase 2 residual target; measured below
```

So the flag is not broken — **it is reporting that a self-declared placeholder was
optimistic, which is one of the things this study was for.** Measured, achievable
`residual_rel` degrades with problem size:

| | dof | residual_rel |
|---|---|---|
| smoke | 8,904 | 7e-09 – 1.5e-07 |
| coarse | 41,064 | 7.7e-08 – 9.4e-07 |
| medium | 104,712 | 1.0e-07 – 2.7e-06 |
| fine | 261,864 | 2.5e-06 – 6.3e-06 |

1.0e-7 is met at `smoke` and essentially nowhere else; 5 of 24 mesh rows, 13 of 39 phase rows,
2 of 16 design rows. **Do not "fix" this by loosening the constant to whatever passes** — that
is tuning a gate until it agrees, which §5 already refused once. What the number should be
depends on what precision the *constraint* needs, and there is no constraint yet because of
the finding above. Record it, leave it, and set it when phase 3 has a formulation.

`LOBPCG_MAXITER = 200` is NOT binding — iteration counts came in at 13–22 everywhere,
including `fine`. That half of the placeholder is fine.

#### What the study measured that IS usable

- **Phase dependence of λ_min is mild**: std/mean over 13 reference phases is **0.66% /
  1.15% / 1.59%** on the three designs, against the stress ripple's 9.8%. The 4-phase
  production stencil catches the reference min to 0.5–1.9% and the max to 1.7–4.1%. If a
  reformulated eigenvalue behaves similarly, `PRODUCTION_PHASES = 4` will be enough.
- **All 16 elites are finite and well-conditioned** at `coarse`, spread 1.457×, no solver
  refusals anywhere in 91 measurements.
- **`fine` is affordable**: 143–210 s per solve, 3.1 GB peak. Nothing about the 261k-dof rung
  needs special handling, which was unknown before this run.

#### Driver changes this run required, and why they are not cosmetic

`run_mesh` had **no `try` around its rows** while `run_load` and `run_design_space` did, and
`run_mesh` is both the first section and the only one that drives `fine`. Worse, the report
was written **once, at the end of `main()`** — so any failure erased every completed section,
and a cgroup SIGKILL at the cap is not catchable by any `try`. Both are fixed: rows are
guarded and a refused row is recorded and **fails** the section (`n_error_rows`, plus an
explicit error check in `pass`) rather than discarding the rungs that did converge; and the
report is **checkpointed after every section**, with `complete` and `sections_complete` so a
partial file can never be misread as a verdict (`pass` is absent until the run is whole).
`_series` skips error rows, so a truncated ladder cannot pass vacuously on `len < 2`.
Validated on `--quick` before the full run. This is the same precedent §1 records for
study_stage3's ladder, applied to the driver that needed it more.

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
target, not the stress constraint. Every gram below 58.660 is on the other side of it. This
is the first item on this list that is no longer a preference: it is a manufacturing
parameter that is now provably binding, and it is worth asking the process what it can
actually hold before asking the optimizer for anything else.

**§8 HAS NOW MEASURED WHAT THE FLOOR IS WORTH, and it narrows this item rather than
closing it.** The floor is only sovereign over `t1`/`t2`; below ~1.5 mm the optimizer picks
`t0` and `t3` for itself and holds them there. So the question to put to the process is no
longer "what wall can you hold?" in the abstract — it is **"can you hold 1.2 mm?"**, worth
19.4 g against today's 2.0 mm, with everything below that worth only 3.3 g more. That is a
single yes/no with a number attached, which is what makes it decidable now.

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
make m9                                                   # M9 phase 2 in full, ~49 min.
                                                          # 3 designs x smoke/coarse/medium/fine
                                                          # x 13 phases.  3.1 GB peak; FAILS by
                                                          # design now -- see §7, exit 1 is the
                                                          # verdict, not a crash
make prod9 / make prod10                                  # §0(b), one start each, SEQUENTIAL
make m9buck                                               # §9, M9 phase 3: the generalised
                                                          # load factor + the load ramp that
                                                          # shows it is NOT a safety factor.
                                                          # ~34 min at coarse ALONE (~81 min
                                                          # beside 3 descents), ~3 GB
make minwall-<floor>                                      # §8, what the wall floor costs.
                                                          # Ran at 2.2/2.0/1.8/1.6/1.4/1.2/
                                                          # 1.0/0.8.  125 steps each; 2.0 is
                                                          # the CONTROL and the others are
                                                          # unreadable without it.  ~2-4 h
                                                          # each; four abreast is fine on 24
                                                          # cores / 61 GB and costs ~111-120
                                                          # s/step vs ~40 alone.
                                                          # NOTE: floors are a pattern rule,
                                                          # so NEVER add them to .PHONY --
                                                          # make skips pattern search for
                                                          # phony targets and every arm
                                                          # silently no-ops with exit 0.
make test                                                 # 427 tests, ~22 min
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

`studies/study_m9.json` is §7's, and it is the one artifact here whose **FAIL is the
deliverable**. `make m9` exits 1 by design: the `mesh` section fails on physics (h²
divergence) and the other three on `LOBPCG_RESIDUAL_REL = 1e-7`, a constant that says in its
own comment that Phase 2 was meant to measure it. Do not make this file pass by loosening
either gate. It also carries `complete`, `sections_complete` and per-section checkpoints, so
a killed run leaves a readable partial with no `pass` key rather than nothing — the `fine`
rung is 261k dof and had never been run when the guard was written. It turned out to cost
3.1 GB and 143–210 s per solve, so the guard was not needed this time; it is kept because
that was not knowable in advance and is now the only record of it.

The elite-9 record stops at step 149 and the elite-10 record runs to 300, so **the two are
not directly comparable step for step** — §6 states the size of that asymmetry and which way
it leans. Neither file is a gate: re-running either produces a different wall clock and,
because §1's S13 note applies, a trajectory that may differ in its last bits. What should
reproduce is the *shape* of the answer — every barrier zero, four thicknesses on the floor,
mass ~97% of the loss against deflection ~46% of the gradient.

---

### 8. §0(2) — what `MIN_WALL_MM` COSTS. DONE, IN TWO PASSES. The floor DOES stop binding — at the ENDS, near 1.5 mm — and the first pass's headline was wrong.

`make minwall-<floor>` for **2.2 / 2.0 / 1.8 / 1.6** (pass 1) and **1.4 / 1.2 / 1.0 / 0.8**
(pass 2), 125 steps each from `stage3_prod_best_elite10.json` (not `rank:10`), `coarse`,
8 phases, `uniform`, 4 workers, seed 0. **All EIGHT arms ran the full 125 steps with ZERO
events and zero rejects.** Records: `stage3_minwall_<floor>.json`, genomes
`stage3_minwall_best_<floor>.json`.

| floor | loss | mesh mass | axle drop | util | `t0` | `t1` | `t2` | `t3` | on floor |
|---|---|---|---|---|---|---|---|---|---|
| 2.2 | 56.1389 | 65.734 g | 1.9924 | 0.5731 | 2.2000 | 2.2000 | 2.2000 | 2.2000 | **4/4** |
| **2.0** | 49.7254 | 58.551 g | 1.9936 | 0.5910 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | **4/4** |
| 1.8 | 43.9892 | 52.458 g | 1.9955 | 0.6416 | 1.8000 | 1.8000 | 1.8000 | 1.8000 | **4/4** |
| 1.6 | 39.0107 | 47.026 g | 1.9970 | 0.6655 | 1.6000 | 1.6000 | 1.6000 | 1.6000 | **4/4** |
| 1.4 | 35.3760 | 42.821 g | 1.9990 | 0.7345 | 1.4000 | 1.4000 | 1.4000 | **1.5102** | 3/4 |
| 1.2 | 32.5051 | 39.194 g | 1.9992 | 0.7830 | 1.2000 | 1.2000 | 1.2000 | **1.4614** | 3/4 |
| 1.0 | 31.5398 | 37.899 g | 2.0004 | 0.7752 | **1.3826** | 1.0000 | 1.0000 | **1.5894** | 2/4 |
| 0.8 | 30.0993 | 35.911 g | 1.9991 | 0.8094 | **1.4638** | 0.8000 | 0.8000 | **1.6074** | 2/4 |

#### THE FINDING: the two ENDS pick a thickness and hold it. The mid-span never does.

**`t3` lifts off the floor at 1.4 mm and `t0` at 1.0 mm, and neither tracks the floor
afterwards.** Across a floor moving 1.4 → 0.8 mm, `t3` sits at **1.5102 / 1.4614 / 1.5894
/ 1.6074** and `t0` at **1.3826 / 1.4638**. Those are floor-INDEPENDENT values: the
optimizer is choosing them, which is the whole thing the sweep existed to detect.

**`t1` and `t2` are pinned `low` at every one of the eight floors, down to 0.8 mm.** They
never decide. The split is physical: `t0` and `t3` price the hub and rim junctions through
`Kt(R, t)`, so thinning them costs stress at a concentration and something pushes back.
Mid-span carries no local riser — it is pure mass — so it thins until the floor stops it.

The practical form: **the wheel wants ~1.45 mm at the hub, ~1.6 mm at the rim, and as
thin as the process can print in between.**

#### THE RETRACTION: `29.5 g/mm` was an artifact of only sampling floors where everything was pinned

The first pass measured four floors, found `mass ∝ floor^1.05`, and reported **29.5 g/mm
with no diminishing returns**. That fit was excellent — ~1% on all four points — and it
was excellent *because all sixteen thickness genes in it were sitting on their bound.* A
sweep in which every gene is pinned is measuring the floor, not the design, and it will
look linear no matter what the design would have done given room.

Extending the band breaks it:

```
2.2 -> 2.0 : -7.183 g  =  35.9 g/mm
2.0 -> 1.8 : -6.093 g  =  30.5 g/mm
1.8 -> 1.6 : -5.432 g  =  27.2 g/mm     <- last all-pinned interval
1.6 -> 1.4 : -4.205 g  =  21.0 g/mm     <- t3 lifts off
1.4 -> 1.2 : -3.627 g  =  18.1 g/mm
1.2 -> 1.0 : -1.296 g  =   6.5 g/mm     <- t0 lifts off
1.0 -> 0.8 : -1.988 g  =   9.9 g/mm
```

**The marginal cost falls by 4x across the band and the slope was never constant** — even
inside pass 1 it ran 35.9 → 27.2, which a `floor^1.05` fit absorbed into an exponent. Use
the interval table, not a single number. The bands worth quoting:

| band | marginal | total |
|---|---|---|
| 2.2 → 1.6 mm | ~31 g/mm | −18.7 g |
| 1.6 → 1.2 mm | ~20 g/mm | −7.8 g |
| 1.2 → 0.8 mm | ~8 g/mm | −3.3 g |

**Going 2.0 → 1.2 mm is worth 19.4 g, a third of the wheel. Going 1.2 → 0.8 mm is worth
3.3 g.** So the process conversation is worth having down to about 1.2 mm and is
approximately pointless below it.

**THIS IS THE FOURTH TIME THIS REPO HAS SHIPPED A CONVERGENT-LOOKING NUMBER THAT MEASURED
THE WRONG THING**, after `stress_scale`, `λ_min(K_t)` and the load factor (§9). The
pattern holds exactly: the quantity fit beautifully over the range sampled, and the
disproof came from **varying something other than what had been varied so far** — here,
extending the floor past the point where the answer stops touching the bound. The lesson
generalises to sweeps: *a fit taken entirely inside a region where a constraint is active
describes the constraint, not the system.*

#### NOTHING PHYSICAL BINDS ANYWHERE IN THE BAND — including at 0.8 mm

The `stress` loss term is **exactly 0.00000 at all eight floors**, as are `buckling`,
`min_sj`, `fillet_cap`, `fold`, `arrival`, `hub_overlap` and `x_order`. Only `mass`,
`smoothness` and `deflection` are ever nonzero. Utilisation reaches only **0.8094 at
0.8 mm** against an allowable of 1.0, and hub utilisation is the max at every floor
(`u_hub` 0.573 → 0.809, `u_rim` 0.428 → 0.572).

So the flattening is **not** stress switching on. It is the deflection target being bought
back with geometry: bending stiffness goes as `t³`, so past ~1.2 mm the centerline has to
distort to hold 2.0 mm of drop, and `smoothness` climbs **0.18 → 0.58** paying for it.
Axle drop stays within ±0.05% of target at every floor — the spec is met everywhere, it
just costs more geometry to meet.

**Two cautions that are NOT in the loss.** `min_scaled_jacobian` degrades monotonically
**0.8929 → 0.6366** as the floor drops; the `min_sj` barrier never fires, but the mesh is
measurably worse at 0.8 mm and any conclusion there rests on a poorer discretisation than
the ones above it. And `R_hub` moves non-monotonically (0.982 → 0.579 → 0.682) while
`kt_hub` stays pinned near 2.0-2.08 at every floor — the optimizer is trading fillet
radius against `t0` to hold the concentration factor constant, which is why `t0` lifting
off at 1.0 mm coincides with `R_hub` reversing direction. **1.2 → 1.0 is a branch change,
not a smooth continuation**, and it is why that one interval's marginal cost (6.5 g/mm) is
lower than its neighbour's on both sides.

#### What this does and does not settle

**Settled:** the optimizer picks the end thicknesses, the floor sets the mid-span, and the
floor is worth having a process conversation about down to ~1.2 mm.

**Not settled:** whether 0.8-1.0 mm is manufacturable at all, and whether the `coarse` mesh
is trustworthy at `min_sj = 0.64`. **A promotion candidate below 1.2 mm should be re-scored
at `medium` before anyone believes its mass.** No arm below 1.6 mm has been mesh-converged.

**Provenance gap in these artifacts.** The eight `stage3_minwall_best_*.json` were written
before `search_block()` existed, so their `search` blocks carry the optimizer settings but
**not** `min_wall_mm`. The floors are recoverable from `stage3_minwall_<floor>.json`
(`settings.min_wall_mm`) and from the pinned `t1`/`t2` values. Future `--best-out` records
carry the floor; these eight do not.

#### The control arm is what makes the other three readable, and it earned its place twice

`minwall-2.0` restarts from its own converged answer at its own floor, so it measures the
protocol rather than the floor. **It found a real defect in the experiment design before
the other arms had produced anything.**

§0(2) justified 125 steps from a measurement off elite 10's record — "step 125 is +0.096%
from its final loss". **That measurement does not transfer to a restart.** It was taken
INSIDE a running descent, where Adam's `m`/`v` were warm and the cosine `lr` had already
decayed. Restarting cold at the optimum with `lr = 0.01` and zeroed moments kicks the
iterate straight back out: the control's loss went 49.7376 → 49.95 → **51.09** over three
steps, a 2.7% excursion, before settling.

It recovered — final **49.7256**, best **49.7254 at step 61**, mass **58.551 g** against
elite 10's 58.660 g, i.e. **−0.19%**. So the transient costs nothing and the protocol is
sound. But that is a MEASURED conclusion, and without the control arm the 1.6 result would
have been reported against a start value from a different run under a different schedule.

**The control drift is the sweep's noise floor: ±0.19% in mass.** Every difference in the
table above is 50-100x that, which is what makes them signal.

#### What this run measured about the MACHINE, and it contradicts §1

This box is **24 cores / 61 GB / 7 GB swap**, not the 16-core / 31 GB box every number in
§1 was measured on. The memory rules relax — one descent measured a **13.56 GB peak**
(§1's 12.7 GB estimate travelled), so three fit where one used to.

**But the parallel speedup does NOT track cores or RAM, because the binding resource is
MEMORY BANDWIDTH.** Measured across both passes: one arm alone runs at **~38-47 s/step**;
three arms concurrently at **~95 s/step each**; **four arms concurrently at ~111-120
s/step each** (pass 2, 16 pool workers on 24 cores). Going 3 → 4 arms bought **~17% more
aggregate throughput for 33% more concurrency**, and against a single arm the aggregate
speedup is **~1.5x, not the ~4x** core count and capacity predict.

The same effect distorts any wall-clock read off a shared box, in both directions: §9's
buckling study took **4851 s** beside three descents and **2045 s** alone, and a
single-step `smoke` descent — normally a minute or two — took **6m41s wall / 11m13s CPU**
while four arms were running. **Never quote a wall clock from this repo without saying
what else was on the machine.** The inner loop is `spla.spsolve` (`wheel_fem.py:1183`), a sparse direct LU called
once per Newton iteration, and twelve of those saturate the memory controller.

Two consequences. **The GPU is irrelevant** — `requirements-opt.txt` pins CPU
`jax==0.11.0`/`jaxlib==0.11.0`, and even with a CUDA build the expensive half is SciPy's
LU on the CPU, while the `XLA_FLAGS` pin that makes the adjoint bit-reproducible is a
CPU-thread-pool setting. **And S13's efficiency ladder does not travel across concurrent
RUNS** — it measured workers inside ONE evaluation, which is a different contention
regime. If a single descent's wall clock matters, the lever is the solver (reusing
factorisations, or a preconditioned iterative solve), not the hardware.

**Pass 1** ran three-at-a-time with the fourth held back, each under
`systemd-run --user -p MemoryMax=15G --collect`, on the reasoning that four concurrently
is ~54 GB of 61 and slice-level memory pressure is what took the desktop down twice before.

**Pass 2 ran all four concurrently with no caps, and it was fine** — the box stayed at
~4 GB used with 40 GB free throughout, and all four arms completed with zero events. The
15 GB-per-arm figure that motivated the caps came from a `medium`-rung measurement; a
`coarse` arm with 4 workers is nowhere near it. Caps are still the right default for
`medium` or for anything left unattended, but they are not needed to run this sweep.

---

### 9. M9 PHASE 3 — THE LOAD FACTOR CONVERGES AND IS NOT A SAFETY FACTOR. `make m9buck`.

`studies/study_m9_buckling.py`, **2045 s** at `coarse`, **OVERALL: PASS** — and unlike §7
the PASS is not the interesting part. Artifact: `studies/study_m9_buckling.json`.

Budget 2045 s only on an otherwise-idle box. An earlier identical run took **4851 s**
because three `minwall-*` descents were sharing the machine — 2.4x, from contention alone,
with no setting changed. The inner loop is a sparse direct LU, so it is memory-bandwidth
bound and co-scheduled runs steal from each other far more than core count suggests (same
effect as §8's note on the sweep). Do not read a wall-clock off a shared run and project
from it.

**The formulation is now reproducible from the repo, which it was not.** §0(1) H2's
numbers lived only in a scratchpad `h2_check.py` that is no longer on disk, so this driver
had to re-derive them. It does, exactly:

| config | reduced dof | load factor | PLAN.md H2(b) |
|---|---|---|---|
| smoke | 8,904 | **1.378129** | 1.378129 |
| coarse | 41,064 | **1.359846** | 1.359846 |
| medium | 104,712 | **1.356669** | 1.356669 |

`last_pair_rel` **2.34e-03** against `GATE_MESH_REL = 0.05` — inside by **21x**, on the
same meshes where `λ_min(K_t)` misses by twelve.

**THE STATE MUST BE SOLVED UNDER SVK, NOT ONLY ASSEMBLED UNDER IT — AND PLAN.md DID NOT
SAY SO.** H2's prose describes `K_0 = K(u=0)`, `K_t = K(u_service)` without stating which
kinematics produced `u_service`, and the obvious reading — reuse the existing
linear-kinematics contact solve — is **wrong by +31%**:

| state solved under | smoke | coarse |
|---|---|---|
| `kinematics="linear"` | 1.800046 | 1.785253 |
| **`kinematics="svk"`** | **1.378129** | **1.359846** |

That is the same class of error `study_m9` made (a linear state threaded into a nonlinear
operator), one level up, and it was found by prototyping the formulation before writing a
driver on it. It is now documented in the driver header and the `m9buck` recipe.

#### THE FINDING: `lambda(f) > 1` AT EVERY LOAD LEVEL. There is no critical point.

A real critical factor is a **fixed point**: at `f = λ_cr` the remaining factor is 1.0.
Measured on `best_solution` at `coarse`, eleven load levels, **zero solver refusals**:

| f (× service) | 0.5 | 1.0 | 1.36 | 2.0 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|
| **λ(f)** | 1.9891 | 1.3598 | 1.2162 | 1.1070 | 1.0481 | **1.0263** |
| f·λ(f) | 0.9946 | 1.3598 | 1.6540 | 2.2140 | 3.1443 | **4.1051** |

**λ approaches 1 from ABOVE as roughly `1 + 0.43/f²` and never crosses it.** The implied
critical load recedes exactly as fast as the load advances — a treadmill, not a limit
point. `crosses_unity: false` is now a field in the artifact.

**Why that is decisive rather than suggestive.** `λ = 1` is precisely the condition
`det(K_0 + 1·K_g) = det(K_t) = 0`. So `λ(f) − 1` measures how close the CONVERGED TANGENT
is to singular, and it never reaches zero. The quantity is real and it is telling the
truth; what it is not is a load factor.

**So: 1.36 must not be reported as a safety factor** — PLAN.md's warning was right, and
this is the measurement that settles it. **What can be said instead is stronger and more
useful: the wheel is stable to at least 4× service load**, on eleven converged nonlinear
SVK contact solves with a non-singular tangent throughout. The Euler `buckling` proxy
(ratios 0.066-0.087) would never have shown that.

#### What else the study established

- **Phase spread 1.25-2.35%** over 13 reference phases on three designs, against
  `λ_min`'s 0.66-1.59%. `PRODUCTION_PHASES = 4` would be adequate *if* the quantity were
  ever promoted. Open item 1, half one: CLOSED.
- **Design space: 16/16 converged, 1.0912-2.4032, a 2.20x spread, none below 1.0.** So
  1.36 is *this design*; elites 1 and 2 sit near 1.10. Open item 1, half two: CLOSED.
- **Contact in the operator is +1.46%** at `coarse` (+3.38% at `smoke`, so it shrinks
  under refinement). Open item 4: CLOSED, and it is small.

#### Phase 3 stays blocked, and the reason is now measured rather than suspected

The quantity is mesh-convergent, phase-stable, design-discriminating and cheap. It fails
exactly one requirement — **load-independence** — and that is the one a constraint needs.
A threshold calibrated at service load would be `stress_scale` a third time.
`LOBPCG_RESIDUAL_REL` stays at 1e-7 and `buckling` stays inert; §7's rule ("record it,
leave it, set it when phase 3 has a formulation") is still not satisfied.

**THIS IS THE THIRD TIME THIS REPO HAS HIT THE SAME PATTERN**, and it is the most
transferable thing on this page: `stress_scale`'s `c = max/pnorm`, `λ_min(K_t)`, and now
the load factor. Each was well-posed, each converged or looked convergent, and each
measured the wrong thing. **In all three cases mesh convergence was treated as evidence of
meaning, and in all three the disproof came from varying something OTHER than the mesh** —
the exponent, the kinematics, the load. A quantity that has only ever been refined has not
been tested.

---

### 10. §0(3) — DEFERRED, DELIBERATELY. And the golden test no longer blocks it.

**Not done, and that is a decision rather than a slip.** §8 measured that
`stage3_prod_best_elite10.json` (58.660 g) is optimal **at the 2.0 mm floor only**, and
there are now **eight** candidates on disk spanning 65.734 g down to 35.911 g. Whichever
floor the process can actually hold determines which one should ship; promoting 58.660 g
now would ship a genome that one process conversation could supersede.
`best_solution.json` is **untouched**.

**The shortlist, given §8.** 1.2 mm (**39.194 g**) is the value-for-money point — it
captures 19.4 g of the available 22.6 g, and everything below it is worth 3.3 g total.
1.6 mm (**47.026 g**) is the conservative pick and the thinnest floor at which all four
genes are still pinned, i.e. the last one whose behaviour is fully understood. **Nothing
below 1.6 mm should be promoted without a `medium`-rung re-score first** — `min_sj` falls
to 0.71 at 1.2 mm and 0.64 at 0.8 mm, and no sub-1.6 arm has been mesh-converged.

**What WAS done is the part that made promotion risky, and it is now permanent.**
`tests/test_golden.py` — the repo's stated "safety net for every later milestone" — read
`geometry`, `loss_terms` and `metrics` out of `best_solution.json` and recomputed them
through `wheel_fea.evaluate_design`. That coupling meant any promotion forced a choice
between two bad outcomes:

1. **Break the golden test.** The Stage-3 record's `loss_terms` are the FEA objective's 13
   terms (`fillet_cap`, `min_sj`, `phase_ripple`, ...); `evaluate_design` produces the
   beam surrogate's 9. `test_loss_terms_reproduce` fails outright on the mismatch.
2. **Re-baseline it.** Regenerate the blocks with `evaluate_design` so it passes — at
   which point the test checks that function against numbers the function just produced,
   and **detects nothing**.

So the GA/beam reference is preserved as **`best_solution_ga_beam.json`**, carrying a
`note` block saying what it is, and `test_golden.py` reads that file. Its job is
regression detection on `evaluate_design`, which has nothing to do with which genome
ships. **11 tests pass against the pinned file, and `best_solution.json` is now free to
move whenever the floor decision lands** — a one-file change with no test churn.

The two files are byte-identical in every block today. That is the starting condition, not
an invariant.

**What promotion will still need when it happens**, unchanged from §0(3): `make export`
(~4 min, expect nearer 230 s now all 48 corners fillet), and then a real look at the
manifest rather than an assumption — `r_built`, `kt_error_pct`, face count, shortest edge,
and the AREA and MASS gaps (−1.384% / −2.277% today). Nobody has yet looked at any
Stage-3 geometry in OCC. `R_hub` on the elite-10 genome is **0.9820** in the file against
a cap of 1.0400, i.e. under it, which is the good direction — but the sub-1.6 mm arms run
`R_hub` down to **0.579**, and nothing has checked that OCC can actually build a fillet
that small next to a 1.2 mm wall. **Export the candidate before trusting its mass.**

**Two latent breaks on this path were found and fixed while the sweep ran**, both of which
would have failed SILENTLY rather than loudly, which is why they were worth catching first:

- **`--best-out` records did not carry the box they were descended in.** The GA writer
  records `min_wall_mm`/`cy_bound_mm` (`wheel_fea.py:1393`) and the Stage-3 writer did
  not — so the eight sweep genomes, every one of them a boundary optimum, were
  distinguishable only by reading their own pinned `t` values back out. Now
  `wheel_stage3.search_block()`, split out of `main()` so it is testable without a solve
  (the cheapest run that reaches that line still costs minutes).
- **The exporter's mass cross-check went `nan` on a Stage-3 genome.** `report()` read
  `metrics['total_mass_g']` through a `.get(..., nan)`; the GA writes that key and a
  descent writes `mesh_mass_g`. The export succeeded and printed `nan g` — losing one of
  the only checks that compares the FEA and CadQuery pipelines against each other, on
  exactly the genome that ships. Now `wheel_step_export.optimizer_spoke_mass()`, which
  returns the value **and which key it came from**: the two are the same role measured two
  different ways (analytic beam area vs integration over the FEA mesh) and normalising
  them into one number would hide that from the reader.

---

### 11. THE FLOOR DECISION LANDED: **the process can hold 1.2 mm** (2026-08-05). The 1.2 mm arm is re-scored, exported, and it is NOT promoted yet — one check fired.

§10's shortlist asked a single yes/no with 19.4 g attached. The answer is **yes**, so the
promotion candidate is `stage3_minwall_best_1.2.json` (genome `350f4c7`, **39.194 g**),
and §10's two prerequisites have now both been run against it.

**`best_solution.json` is still untouched.** Two things came back from the export that a
human should see before it moves — see "What fired" below.

#### The `medium` re-score: the mass is real and both constraints still hold

§8 and §10 both said nothing below 1.6 mm may be believed until it is re-scored on a
finer mesh, because no sub-1.6 arm had ever been mesh-converged. One forward evaluation
of the converged genome at `medium`, 8 phases, `uniform`, 4 workers, **271.8 s**:
`stage3_minwall_1.2_medium.json`, genome echoed to `stage3_minwall_best_1.2_medium.json`.

| quantity | `coarse` | `medium` | change |
|---|---|---|---|
| **mesh mass** | 39.19436 g | **39.19433 g** | **−0.00008%** |
| axle drop (target 2.0) | 1.99923 mm | **2.01931 mm** | deflection error **0.0% → +1.00%** |
| **utilisation** | 0.7830 | **0.7986** | +1.98%, against an allowable of 1.0 |
| `min_scaled_jacobian` | 0.71086 | **0.71114** | **+0.04%** |
| `phase_ripple` | 0.09570 | 0.09421 | −1.56% |
| loss | 32.5051 | 32.7376 | +0.72% |
| `max_stress_mpa` | 140.89 | 182.42 | **+29.5%** — the singularity, as always |

**Every barrier term is still exactly 0.00000 at `medium`** — `stress`, `min_sj`,
`fillet_cap`, `fold`, `arrival`, `hub_overlap`, `x_order`, `buckling`, `phase_ripple`.
Only `mass`, `smoothness` and `deflection` are nonzero, exactly as at `coarse`.

Three things worth reading off that table:

- **The mass survives refinement to five figures**, which is what the re-score existed to
  establish. 39.194 g is a real number.
- **`min_sj` does NOT degrade under refinement** — 0.7109 → 0.7111. §8's caution ("the
  mesh is measurably worse at 0.8 mm and any conclusion rests on a poorer
  discretisation") was about mesh quality falling as the *floor* falls; it does not
  compound as the mesh refines. The thin wall is what sets it, not the resolution.
- **`max_stress_mpa` moving +29.5% while `utilisation` moves +1.98% is the M4 result
  restated**, and it is the reason the constraint was rewritten in M8b-i.6. Anyone
  alarmed by 182 MPa against a 25 MPa allowable is reading the divergent field max.

**What this is NOT: a re-optimisation at `medium`.** It is `--steps 0` — the `coarse`
answer scored once on a finer mesh. It says the design is what `coarse` said it was; it
does not say a `medium` descent would stop here.

#### The export: for the FIRST time the Kt the optimizer priced is the Kt the part has

`export/stage3_minwall_best_1.2.step` (+ `_nofillet.step`, `_step_manifest.json`),
**67.3 s**, OCC valid, one solid, not self-intersecting, bounding box exact.

| junction | R requested | R built | edges | Kt model | Kt built | **error** |
|---|---|---|---|---|---|---|
| hub | 0.5790 | **0.5790** | **24/24**, ONE family | 2.0235 | 2.0235 | **+0.0%** |
| rim | 2.7495 | **2.7495** | **24/24**, ONE family | 1.4226 | 1.4226 | **+0.0%** |

Against the shipped genome's **+73.4%** and **+111.4%**. §3's whole finding — that a
junction priced by its worst corner reports what twelve square corners were hiding — has
no bite here because **there is no second family**: every corner takes the requested
radius at the first ladder rung. That is also why the export costs 67 s against 230 s.

**The mechanism is that small fillets build.** §5's cap model predicted this and the
descent obeyed it: `R_hub` = 0.579 sits under `hub_fillet_cap_mm` = 0.624, and 2.749 is
under the rim's slot. The genome stopped asking for fillets the part cannot build, so the
stress model and the solid finally agree. **`kt_error_pct` is now +0.0% rather than +73.4%
— the discrepancy this milestone existed to remove is gone on this genome.**

Solid mass **47.63 g** (whole wheel; the optimizer's 39.194 g is spoke material only, and
hub + rim add the rest) against the shipped genome's 74.12 g, **−35.7%**.

#### WHAT FIRED, and it is why this is not promoted in the same breath

1. **`[WEAK JUNCTION]` — hub overlap 18.12 mm³ against a 50.0 mm³ floor.** The shipped
   genome is 78.53 mm³ and the rim here is 68.13 mm³, so it is the **hub** end alone. The
   exporter's own words: "this spoke grazes the ring rather than meeting it. Deepen
   `HUB_EMBED_RADIUS_MM`, or constrain the optimizer's junction angle." Cause is not
   mysterious — the root is `t0` = 1.2 mm against 2.477 mm on the shipped genome, and the
   embed is only 0.5 mm deep (`HUB_EMBED_RADIUS_MM = HUB_RADIUS_MM − 0.5`), so there is
   simply less material buried inside the hub disk. **The union still produced a valid
   single solid with all 24 hub corners on the circle**, so this is a warning about weld
   robustness, not a build failure.

   **MEASURED, AND IT KILLS THE SUGGESTED FIX: deepening `HUB_EMBED_RADIUS_MM` does not
   get there.** Profile-only sweep, no fillets, three genomes, embed depth 0.5 → 2.0 mm
   (`embed_depth_probe.py`, a scratchpad one-question probe like `eps_n_check.py`):

   | embed depth | 1.2 mm genome | elite 10 (`t0` 2.00) | `best_solution` (`t0` 2.48) |
   |---|---|---|---|
   | **0.50 (shipped)** | **18.12** | **48.72  ← also under** | 78.53 |
   | 1.00 | 22.64 | 52.54 | 80.68 |
   | 1.50 | 27.20 | 56.30 | 82.86 |
   | 2.00 | 31.73 | 60.09 | 85.01 |

   Four times the embed depth buys the 1.2 mm genome **13.6 mm³**, and it needs 31.9.
   Extrapolating the 9.1 mm³/mm slope puts the crossing near **4 mm of embed** — a third
   of the 12.7 mm hub radius, with twelve roots converging inside it. That is not a
   constant to nudge; it is a different construction. (Self-intersections stayed **0** and
   min curvature never moved at any depth on any genome, so *"keep these SHALLOW"* was not
   what bit here — the sweep is safe, it just does not reach.)

   **What the sweep actually shows is that the 50 mm³ floor tracks root thickness, and it
   was set when roots were ~2.5 mm.** The overlap is superlinear in `t0` — 1.2 → 2.0 →
   2.48 mm of root gives 18.1 → 48.7 → 78.5 mm³ at the shipped depth — because a
   near-tangent arrival makes the buried region a wedge. **Elite 10, the candidate this
   repo was going to promote before today, is at 48.72 against the same 50 floor**, i.e.
   the check was already about to fire at the 2.0 mm floor and nobody had run the export
   to see it. So the question is not "how do we rescue this genome's weld" but **"what is
   `MIN_JUNCTION_OVERLAP_MM3` actually asserting, and is a fixed volume the right form of
   it for a thin-walled part?"** That is a modelling decision, and it is the one blocking
   promotion. **ANSWERED IN §12: no, it is not — the fixed volume was a `t0` proxy, and
   normalising it out shows all three genomes at the same 0.56 root thicknesses of
   engagement. The constant is gone and the candidate passes.**
2. **The smallest features got ~19x smaller.** `min_edge` **0.0087 mm** (shipped 0.162)
   and `min_face` **0.195 mm²** (shipped 3.63). OCC is happy — `BRepCheck` valid, 0
   degenerate edges, tolerance 1e-07 — and `min curvature R` is **0.579 mm**, still well
   over the 0.25 mm floor and over a 0.4 mm nozzle. But an 8.7 µm edge is exactly the kind
   of feature another kernel drops silently, and **Inventor has never imported anything
   from this repo** (§0). This is the one import that is now worth doing.

Neither is a reason the 1.2 mm floor was the wrong call. Both are reasons the *export* of
that genome is not yet a shippable part.

#### The exporter can build a candidate without touching the shipped STEP

`wheel_step_export.py` read `best_solution.json` unconditionally, so "export the candidate
before trusting its mass" (§10) was impossible to do in the right ORDER — the only way to
export a genome was to promote it first. It now takes **`--genome <record.json>`**, and
`output_paths()` names the three artifacts after that file's stem unless it IS the shipped
genome, which keeps `wheel.step` / `wheel_nofillet.step` / `wheel_step_manifest.json`
exactly as they were. **A candidate cannot overwrite the shipped STEP**, which would have
recreated on purpose the failure this file was audited for. `--out-prefix` overrides.
`make export EXPORT_GENOME=stage3_minwall_best_1.2.json` drives it. No arguments is
byte-identical behaviour to before, which is what the GA hand-off (`wheel_fea.py:1658`)
still calls.

`warn_if_stale` takes the step path it is actually about and prints the real source name,
rather than saying `best_solution.json` whatever it just read.

**`make test` 427 passed** (1279.97 s), unchanged from §8 — the flag is additive and the
no-argument path is the one every test and the hand-off already exercised.

#### What promotion still needs

`best_solution.json` is unchanged; `tests/test_golden.py` is already decoupled (§10), so
the promotion itself remains a one-file change. Before it:

1. ~~A decision on `MIN_JUNCTION_OVERLAP_MM3`.~~ **RESOLVED — see §12. The constant was
   measuring `t0`.** It is gone; the check now gates on `wheel_geometry.junction_bite`,
   the 1.2 mm candidate passes at 0.562 root thicknesses against a 0.25 floor, and
   nothing about the candidate's geometry changed.
2. ~~**One Inventor import** of `export/stage3_minwall_best_1.2.step`, against a 0.0087 mm
   edge and a 0.195 mm² face.~~ **DONE 2026-08-06 — it imported clean. THE PROMOTION
   HAPPENED; see §13.** Both items on this list are closed and this section is history.
3. Note that promoting also moves the WHEEL the studies describe: every study driver reads
   `best_solution.json`, so §9's load factor (1.36 on the *shipped* genome) and every other
   per-design number on this page is about a genome that would no longer be the shipped one.
   Nothing is invalidated — they are labelled by design — but they stop describing "the
   wheel" and start describing "the old wheel".

---

### 12. THE WEAK-JUNCTION CHECK WAS MEASURING `t0`. `MIN_JUNCTION_OVERLAP_MM3` IS GONE (2026-08-05).

§11 left one thing blocking promotion: the exporter's `[WEAK JUNCTION]` check fired on the
1.2 mm candidate at **18.12 mm³** against `MIN_JUNCTION_OVERLAP_MM3 = 50.0`. §11 had already
ruled out the exporter's own suggested fix (4× the hub embed depth buys 13.6 mm³ of the 31.9
needed) and noted that **elite 10 fails the same check at 48.72 mm³**. That second fact was
the tell, and it is now the whole answer.

#### The measurement

`check_junction_overlap` intersects ONE spoke's 2D outline with the hub disk and multiplies
by the 22.4 mm face width. That volume is **quadratic in the root thickness**: once through
the band width, and again because a near-tangent band of that width crosses the circle over a
proportionally longer arc. Divide it out — `overlap_mm3 / (t² · W)`, the weld's penetration
measured in ROOT THICKNESSES:

| genome | `t0` | hub mm³ | **bite** |
|---|---|---|---|
| `stage3_minwall_best_1.2.json` | 1.20 | 18.12 | **0.562** |
| `stage3_prod_best_elite10.json` | 2.00 | 48.72 | **0.544** |
| `best_solution.json` (shipped) | 2.48 | 78.53 | **0.571** |

**The raw volumes span 4.3×. The bites agree to 3%.** These are the same junction three
times over, and the old constant "failed" two of them purely for being thin. It was never a
verdict on the weld; it was `t0` with a threshold on it. That is why the thinnest design
failed it, why elite 10 failed it, and why no amount of embed depth was ever going to help.

#### What replaced it

`wheel_geometry.junction_bite(overlap_mm3, t_mm, width_mm)` and `MIN_JUNCTION_BITE = 0.25`.
They live in `wheel_geometry` for the same reason `MAX_ARRIVAL_DEG` does — it is the one
module both interpreters can import, so the exporter (CAD env) and the test that audits its
manifest (jax env) share one definition rather than two that can drift.

The geometric argument: a spoke arriving radially and burying depth `d` contributes
`overlap = t·d·W`, so the ratio is exactly `d/t`, and a grazing spoke drives it to 0 — which
is the failure the check exists for.

**0.25 IS A GEOMETRIC FLOOR, NOT A CALIBRATED ONE, AND THE CODE SAYS SO.** Three samples
that agree to 3% cannot calibrate a threshold, and this repo has never produced a junction
that actually failed, so there is no negative example to fit against. 0.25 is half of what
every measured design achieves. Replace it the moment a real failure turns up.

`check_junction_overlap` **still only warns, and must keep only warning.** It runs inside the
GA's export hand-off (`wheel_fea.py:1658`), which checks nothing but the return code — so
raising there would throw away a finished optimization run over a heuristic. The number goes
to the manifest instead, where a test can see it. The warning text also stopped advising
"deepen `HUB_EMBED_RADIUS_MM`", which §11 measured and ruled out.

#### It still catches what it exists for

Negative control: recede the hub ring toward the spoke root so the spoke genuinely only
grazes it, and the metric falls monotonically and fires.

| hub ring r | hub mm³ | bite | |
|---|---|---|---|
| 12.70 (real) | 18.12 | 0.562 | |
| 12.55 | 13.20 | 0.409 | |
| 12.40 | 9.27 | 0.287 | |
| 12.30 | 7.25 | **0.225** | **[WEAK JUNCTION]** |
| 12.25 | 6.43 | **0.199** | **[WEAK JUNCTION]** |

Read honestly: this shows the new metric detects grazing, not that it beats the old one on
this axis — the old floor would have fired here too. The evidence that the *form* was wrong
is the invariance table above, not this sweep.

#### The manifest, and the tests it now has

`junction_overlap_mm3` keeps `hub` and `rim` in mm³ under the same names, and gains `bite`,
`t_mm`, `pass` and `bite_floor`. `floor` is gone rather than repurposed: it was a mm³ number
and the new one is in root thicknesses, so a reader who missed the change gets a `KeyError`
instead of a plausible wrong comparison.

Three tests in `tests/test_export_contract.py`, where there were **none** — nothing under
`tests/` referenced this check before today, which is how it spent the entire `MIN_WALL_MM`
sweep reporting a `t0` proxy without anything noticing:

- the block carries the normalised fields at all;
- **the bite reconstructs from the genes** — recomputed in the jax env, from a manifest
  written by the CAD env, which is what pins the hub to `t0` and the rim to `t3`. On the
  shipped genome a swap moves the hub bite 54%, and the raw volumes cannot reveal it;
- the shipped part clears its own floor, and `pass` IS the floor comparison rather than an
  independently written opinion.

#### The re-exports prove the change is reporting-only

`make export` on the unchanged shipped genome: `junction_overlap_mm3` is **the only
non-volatile key in the manifest that moved**. `genome_hash` 36aed36, `solid.volume_mm3`
59777.4, `mass_g_pla` 74.12, every `fillets.detail` row (hub +73.4%, rim +111.4%),
`profile_health` and `step_health` are all identical. Shipped bite: hub **0.571**, rim
**4.379**, both pass.

`make export EXPORT_GENOME=stage3_minwall_best_1.2.json`: hub bite **0.562**, rim **1.424**,
both pass, **no warning**. Solid 38415.2 mm³ / 47.63 g, `kt_error_pct` +0.0% at both
junctions, min edge 0.0087 mm — every number identical to §11. Nothing about the candidate's
geometry changed; only what the exporter says about it.

**`make test` 427 + 3 = 430 passed.**

#### What is left

~~Promotion now waits on **one thing**: the Inventor import of
`export/stage3_minwall_best_1.2.step`, against that 0.0087 mm edge and 0.195 mm² face.~~
**That import passed on 2026-08-06 and the genome is promoted — §13.** `best_solution.json`
is no longer untouched; it is `350f4c7`.

---

### 13. PROMOTED. `best_solution.json` IS THE 1.2 mm STAGE-3 GENOME (2026-08-06).

The Inventor import of `export/stage3_minwall_best_1.2.step` passed — that was the last
gate §11 and §12 left standing, and it was the one nobody in this repo could measure from
a script: whether a 0.0087 mm edge and a 0.195 mm² face survive a real MCAD translator.
They do. So the promotion happened.

`best_solution.json` now holds **`350f4c7`**, verbatim from `stage3_minwall_best_1.2.json`
apart from a `note` block recording where it came from. The old shipped genome, **`36aed36`**,
is intact in `best_solution_ga_beam.json` — which is not a courtesy copy, it is the file
`tests/test_golden.py` reads (§10), and that decoupling is exactly why this promotion did
**not** re-baseline the regression net onto numbers regenerated by the function the net
exists to check.

#### What shipped

| gene | old `36aed36` | new `350f4c7` |
|---|---|---|
| `t0` | 2.4774 | **1.2000** |
| `t1` | 2.0000 | **1.2000** |
| `t2` | 2.0000 | **1.2000** |
| `t3` | 2.0000 | **1.4614** |
| `R_hub` | 1.5598 | **0.5790** |
| `R_rim` | 3.0000 | **2.7495** |

Three of the four thicknesses sit exactly on the 1.2 mm wall floor. `R_hub` 0.579 is well
under §5's cap of 0.624 for this genome, which is why every corner filleted at the radius
the stress model priced.

#### The export, old → new

`make export` on the promoted file. **The result is identical to the candidate export in
§11 and §12 — all 33 geometric keys, byte for byte; the only manifest key that differs
between them is `source`.** That is the check that matters here: promotion is a rename of
which genome is "the" genome, and it must not perturb geometry. It did not.

Stronger, and worth recording because it retires the obvious doubt: `export/wheel.step` and
`export/stage3_minwall_best_1.2.step` are **byte-identical apart from the `FILE_NAME`
timestamp** — same length, `diff` over everything past the header returns zero lines. The
file now shipping as `wheel.step` is *literally* the file that went into Inventor, not a
rebuild that ought to match it. Same for the `_nofillet` pair.

| manifest key | old | new |
|---|---|---|
| `solid.volume_mm3` | 59777.4 | **38415.2** |
| `solid.mass_g_pla` | 74.12 g | **47.63 g** |
| `fillets` hub `kt_error_pct` | +73.4% | **+0.0%** |
| `fillets` rim `kt_error_pct` | +111.4% | **+0.0%** |
| hub fillet families | 1.127 mm ×12, 0.361 ×12 | **0.579 mm ×24** |
| rim fillet families | 3.000 mm ×12, 0.308 ×12 | **2.749 mm ×24** |
| `junction_overlap_mm3.bite` hub / rim | 0.571 / 4.379 | **0.562 / 1.424** |
| `step_health.min_edge_mm` | 0.162221 | **0.008711** |
| `step_health.min_face_mm2` | 3.6338 | **0.1951** |
| `step_health.min_curvature_radius_mm` | 0.308309 | **0.578934** |
| surface census | 26 Plane / 48 BSpline | **14 Plane / 60 BSpline** |

**26.5 g of PLA, 35.8% of the solid, for a part that is now the first one this repo has
built whose fillets match the ones its stress model priced.** The `kt_error_pct` collapse
is not a fillet improvement — it is the consequence of `R_hub` dropping under the cap, so
OCC never has to fall back to a second, smaller family. The single-family rows are the
evidence.

The two `step_health` numbers that got WORSE are the ones the Inventor import was for.
The minimum edge fell 19× and the minimum face 19×; both are geometry the exporter has
always produced and neither is a defect, but 0.0087 mm is small enough that a translator
was entitled to drop it. It didn't.

#### The design this buys, stated honestly

Both rows below are the `medium` rung under the M8b-i.6 Kt formulation, so they *are*
comparable — unlike the two records' raw metric blocks, which use different key names for
different measurements (`total_mass_g` is the beam surrogate's analytic area, `mesh_mass_g`
is integrated over the FEA mesh; **do not subtract them**).

| | old `36aed36` | new `350f4c7` |
|---|---|---|
| stress utilisation (hub) | 0.4099 | **0.7986** |
| stress utilisation (rim) | — | 0.5594 |
| `Kt_hub` / `Kt_rim` | 1.861 / 1.490 | **2.031 / 1.423** |
| field max (mesh-divergent, **not** the constraint) | 48.47 MPa | 182.4 MPa |
| `min_scaled_jacobian` | — | 0.7111 |

**Utilisation roughly doubled.** That is what 26.5 g costs, and it is the honest headline of
this promotion. The barrier is still satisfied — every term in `loss_terms` that is a
constraint reads exactly `0.0`, `stress` included, and 0.799 is inside 1.0 — but the margin
is now thin where it used to be comfortable. The field max moving 48.5 → 182.4 MPa is **not**
a 3.8× stress increase and must not be read as one: §0 and M4 both record that the field max
diverges under refinement and is not a number the project constrains. `min_sj` 0.711 clears
the 0.64 threshold §8 flagged as the trustworthy floor for a Stage-3 re-score.

#### What is now stale, and it is a lot

Every driver in `studies/` defaults to `best_solution.json`. Nothing in those recorded
studies is *wrong*, but from today they describe **the old wheel**:

- **§9's M9 phase-3 load factor of 1.36** is on `36aed36`. It was never a safety factor
  (§9 is emphatic) and it is now also not this wheel's number. Re-running `make m9` measures
  `350f4c7`.
- **§0's utilisation table** (`best_solution` at 0.4099, elite 1 at 0.5063) is `36aed36`
  and `stage3_prod_best_elite*`.
- Every per-design figure in §1–§12 that names `best_solution`.

The banner at the top of this file says the same thing, because a fresh session reads that
before it reads this.

#### The gate: **20 failed, 410 passed** — and that is the real story of this promotion

`make test` on the promoted genome. Every failure is a test that had quietly encoded a
property of `36aed36` as if it were a property of the PROJECT. Nothing in `src/` broke.
But they are not all the same kind of stale, and lumping them together would be the
mistake — **ten were pins, ten are findings.**

**THE ONE THAT MATTERED MOST, and it was not a test problem — RESOLVED, see below.**
`wheel_fea.MIN_WALL_MM` still defaulted to **2.0** while the promoted genome has
`t0 = t1 = t2 = 1.2`, `t3 = 1.4614`. **All four thickness genes were OUTSIDE the repo's
default gene box**, at `z` = −0.100, −0.133, −0.133, −0.135. `wheel_stage3.descend`
projects its start into the box before stepping, so any driver that loaded
`best_solution.json` without `--min-wall 1.2` would have **silently lifted all four
thicknesses to 2.0 mm and optimised a different, heavier wheel without saying a word.**
That is what `test_a_failed_solve_is_a_step_reject_and_the_run_recovers` was actually
reporting when it said `iterate_unchanged: False` — a fault-injection test catching a box
problem, which is not what it was written for and is not something reading would have
found. §8 anticipated this exact shape of bug and
`test_a_start_below_a_raised_floor_is_projected_up_onto_it` describes it — for the 2.2 mm
arm. It became true of the shipped genome. **`MIN_WALL_MM` now defaults to 1.2.**

**Fixed here — ten pins, all verified passing.** In each case the test's INTENT survives
and the assertion was the thing that was wrong:

| test | what it had pinned | what it pins now |
|---|---|---|
| `test_the_hub_junction_..._filleted` | the hub splits into **2** fillet families | families account for every filleted edge; `r_built_mm` is the worst family's radius. Two families was the old genome's FALLBACK — asserting it made the fix look like a regression |
| `test_the_normalized_gradient_follows_the_moved_floor` ×2 | that the shipped genome violates `hub_overlap` | reads `best_solution_ga_beam.json`. Its own vacuity guard caught this, which is the guard working |
| `test_the_embed_difference_...` | −1.4% of the wheel | the absolute **36.36 mm²** gusset. The gap moved 36.595 → 36.501 mm²; only the denominator changed |
| `test_genome_hash_matches_manifest` | the GA/beam genome against the SHIPPED manifest | the shipped genome against the shipped manifest. **§10's decoupling missed this one assertion** and it was invisible while the two files were identical |
| `test_R_hub_is_dead_at_the_mesh_...` | `d(fillet_cap)/dR_hub > 0` at whatever ships | over-cap genome for the barrier, shipped genome for M7's mesh claim, **plus** a new assertion that the barrier is exactly 0.0 on a feasible design |
| `test_the_hub_cap_reproduces_the_measured_void` | 9.907 deg, measured on the OLD solid | the genome it was measured on. The void is 22.8 deg on the new design — a property of a design, not a constant |
| `test_R_eff_is_exactly_the_cap_more_than_one_rung_above_it` | shipped genome is above its cap | the over-cap fixture; the name was always the precondition |
| `test_the_fillet_cap_barrier_is_live_at_the_shipped_genome` | shipped genome is 0.45 mm over its slot | renamed `..._on_a_design_over_its_cap`; asserts live where it must bite AND zero where it must not |
| `test_kt_hub_is_priced_on_the_buildable_radius_...` | the hard-`min` branch | the over-cap fixture, since every hub assertion in it is the over-cap branch |

A new `genes_over_cap` fixture reads `best_solution_ga_beam.json` for the five cap tests.
That file was pinned by §10 so the golden test could not be re-baselined; it turns out to
be load-bearing for a second reason, which is that **it is the only genome in the repo
that is over its hub fillet cap.**

**A new test was added, for a regime nothing had ever landed in.**
`test_the_shipped_genome_is_inside_the_blend_and_is_priced_conservatively`. `R_hub` =
0.5790 against a cap of 0.6240 — under it, but by less than the smooth-min's blend width,
so `hub_fillet_r_effective` returns **0.5727**, about 1.1% below the radius OCC actually
builds. **This qualifies the `kt_error_pct` = +0.0% headline above and should be read next
to it**: that +0.0% is the EXPORTER comparing its own modelled Kt against its own built Kt,
and it is correct. The OBJECTIVE prices the same junction at Kt **2.0308** against the
exporter's **2.0235**. The blend can only pull `R_eff` down, so the constraint is
CONSERVATIVE, not optimistic — which is why this is a footnote and not a defect. The new
test pins the direction so a future change to the blend cannot quietly reverse it.

#### The ten that are findings, NOT pins — none of these has been touched

Each is the new design genuinely behaving differently. Re-tuning these thresholds would
convert a measurement into a green checkmark, so none of them has been.

> **§14 REVISED THIS TABLE AND THE HEADING OVER IT IS TOO STRONG.** Read §14 before
> trusting any row. Of the eight that were still red, only two survive as "the new design
> genuinely behaving differently" — the GNL gate and the hub compliance share. Three were
> the TESTS being wrong in ways the old genome also exposed once measured, and two were
> tests pinning a pathology. The rows below record what was measured on the day; the
> `what it means` column is the reading that §14 went on to check, and it does not hold
> for the arrival angle, the p99, the area order, the contact patch, or the mass budget.

| test | measured | gate | what it means |
|---|---|---|---|
| `test_a_failed_solve_is_a_step_reject_and_the_run_recovers` | `iterate_unchanged: False` | must be `True` | **the box problem above.** Fix the default, not the test |
| `test_the_arrival_angle_makes_the_junction_a_near_crack` | material wedge **315.4 deg** | `> 340` | the new junction is **materially less crack-like**. Good news — and it means `study_wheel_fea.py`'s convergence-rate explanation is now about a wedge the wheel no longer has |
| `test_peak_stress_diverges_but_the_field_converges` | p99 changes 0.028 → **0.016** | `d2 < 0.3·d1` | the plain-spoke p99 settles more slowly on a 1.2 mm wall. Probably wants a finer rung, not a looser gate |
| `test_the_correction_enters_at_first_order_in_the_load` | GNL at 1% load **0.205%** | `< 0.1%` | **the thin wheel is measurably more geometrically nonlinear.** The fitted exponent is 1.034, so the physics is right — it is the magnitude that moved |
| `test_the_correction_is_not_a_constant_over_the_design_space` | iso ratio **2.69** | `> 3.0` | M5's "the correction is not a constant" is weaker around this design. M5's conclusion is not overturned, but its margin is |
| `test_the_rim_band_holds_a_large_minority_of_the_compliance` | hub share **0.0321** | `< 0.03` | rim 0.306 and spoke 0.662 are both fine; the hub takes marginally more of a floppier wheel |
| `test_total_mass_matches_the_step_manifest_within_the_embed_difference` | mesh 44.36 g vs solid **47.63 g**, −6.9% | −3.0% to −1.2% | the gap is gusset (≈1.01 g, unchanged) **plus fillet material, which grew from ~0.68 g to ~2.26 g** because all 48 corners now build at full radius on a 36% lighter wheel. Explainable, but it is a budget that has to be restated, not a bound to widen |
| `test_area_converges_second_order` | orders 1.986 / 2.061 / **2.355** | `1.7 < r < 2.3` | the finest rung overshoots second order on the thin section |
| `test_random_directions_agree_with_the_adjoint` | worst rel **1.49e-5** | tighter | directional FD agreement degraded on the thinner geometry |
| `test_the_sampled_patch_extent_is_biased_not_merely_noisy` | axle drop moves **0.32%** between `n_quad` 6 and 20 | `< 0.1%` | the contact solve is more quadrature-sensitive on a softer wheel |

**Read the middle four together.** More geometric nonlinearity, a slower-settling stress
field, a less singular corner and a softer contact response are all the same wheel being
thinner and floppier. None of them says the design is unsafe — the constraint that governs
is `stress_utilisation`, which is 0.799 against 1.0. They say the MODELS were calibrated
on a stiffer part, and several of the conclusions written on this page were measured on
one.

#### THE FLOOR NOW DEFAULTS TO 1.2 (same day, decided after the gate)

`wheel_fea.MIN_WALL_MM` **2.0 → 1.2**. Three perimeters at a 0.4 mm nozzle rather than
five. §8 measured what the floor costs, §11 took the decision that the process can hold
1.2, §13 promoted a 1.2 mm genome — and until this change the box still said that genome
was infeasible. It is still settable per run (`set_min_wall`, `--min-wall`); what moved is
the default, and **a default's job is to describe the wheel that ships.**

**What this changes for anyone running anything.** The GA and every driver now search down
to 1.2 mm by default. A run whose numbers assume a 2.0 mm floor predates this — including
every Stage-2 elite and the GA/beam genome, both of which were produced inside the old box.
Nothing on disk was regenerated.

**Two stale claims fell out of it, both corrected in place:**

- `wheel_objective.HUB_CAP_SHARE`'s comment said "`MIN_WALL_MM` is 2.0 and every design on
  disk sits at 2.468–2.627, so the calibrated band covers the reachable design space's
  lower half". Both halves were false the moment the genome was promoted, and the law is
  explicitly "CALIBRATED ON [2.0, 2.6] AND NOT KNOWN OUTSIDE IT" — the shipped genome sits
  at `t0` = 1.2, **below the band**. What saves it is which branch binds: the cap at
  `t0` = 1.2 is 0.624 mm = `HUB_CAP_THICKNESS_SHARE` × 1.2 exactly, so the THICKNESS term
  takes the `min` and the slot share is not being extrapolated on the shipped part. **A
  design that was thin AND had a tight slot would extrapolate it for real, and nothing
  would say so.** Re-run `make hubcap` at the new floor before trusting the slot branch
  under 2.0 mm.
- Two tests hardcoded `2.0` as "the default floor" (`test_gene_space.py`,
  `test_stage3.py`). Both now read the module. In each the claim was "unchanged" or "what
  the module says", and the literal quietly turned it into "still 2.0".

**The gate after the floor change: 8 failed, 423 passed** (was 20/410). The floor change
closed three by itself — `test_a_failed_solve_is_a_step_reject_and_the_run_recovers`, which
was the one reporting the box problem in the first place, plus
`test_random_directions_agree_with_the_adjoint` and
`test_the_correction_is_not_a_constant_over_the_design_space`. **It also broke one, and that
one is worth reading rather than re-tuning.**

#### `test_the_beam_to_wheel_ratio_is_not_a_constant`: the box moved, not the wheel

It failed at **2.686** against a `> 3.0` gate. The obvious reading — that the promoted
genome made Gate 1's headline weaker — is wrong, and one measurement settles it:

| genome | floor | `fea_over_beam_ratio` |
|---|---|---|
| `best_solution_ga_beam.json` | 2.0 | **4.943** |
| `best_solution.json` (shipped) | 2.0 | **4.943** |
| `best_solution_ga_beam.json` | 1.2 | **2.686** |
| `best_solution.json` (shipped) | 1.2 | **2.686** |

**Identical down the genome column, to every digit.** `run_beam_blindness` draws a Latin
hypercube from the GENE BOX and computes the statistic over the drawn rows only — the
`genes` argument is deliberately excluded from the statistics, and its own comment says why
("it was optimised and the others were drawn, so pooling them would understate the
spread"). The number is a property of the box; the test's fixture never touched it.

The mechanism is written in that same function: "feasible random spokes are typically
10–100x stiffer than the 2.0 mm target, **because the wall-thickness floor is what binds
there**." Dropping the floor to 1.2 lets the hypercube draw thinner, floppier spokes, which
moves the drawn population toward the beam target and COMPRESSES the max/min ratio.
4.943 → 2.686 is that compression, and it is the correct answer for the box the project now
searches.

**Gate 1's conclusion is not overturned.** `correction_factor_is_defensible` is still
`False` — that assertion passed — and a 2.7x spread is still not a constant. What breached
is the `> 3.0` margin, a number picked in a box with a 2.0 mm floor. The mirror image
happened on the gnl side: `test_the_correction_is_not_a_constant_over_the_design_space`
measured 2.694 in the old box and now passes. **Two tests of the same shape swapped sides
of the same threshold when the box moved.** Whether 3.0 is still the right margin at a
1.2 mm floor is a judgement about Gate 1's claim, not about this promotion, so it has been
left alone rather than tuned down to fit.

Everything else in the ten findings above is unaffected by the floor change — they are
properties of the geometry, not of the box.


### 14. THE EIGHT THAT WERE LEFT. Five open items, three closed by measurement (2026-08-06).

§13 promoted the wheel and left **8 failed / 423 passed**, deliberately untouched. This
section is the triage. It is not eight problems — it is five, and the ones that turned out
to be about the tests rather than the wheel have been fixed.

**The rule that governed all of it: measure before touching a threshold.** Every item below
was diagnosed by sweeping something (the mesh tier, the gene box, the reference) on BOTH the
promoted genome and `best_solution_ga_beam.json`, so that "the new design broke it" always
had to survive a comparison against the design it replaced. Three times it did not.

| # | item | status |
|---|---|---|
| 1 | two tests pinned a PATHOLOGY as an invariant | **CLOSED** — both inverted |
| 2 | the mass test asserted a budget it could not measure | **CLOSED** — exporter now publishes the fillet term |
| 3 | four convergence failures, one suspected common mechanism | **CLOSED** — hypothesis wrong; three separate causes found, two fixed |
| 4a | the pre-registered GNL gate | **DECIDED — it stands, and linear kinematics is the real question** |
| 4b | the hub compliance share | **OPEN — a human's call** |
| 5 | `make hubcap` at the 1.2 mm floor | **OPEN** |
| 6 | `EMBED_ALLOWANCE_PER_SPOKE_MM2` is stale (new, found by item 2) | **OPEN** |

---

#### Item 1 — two tests pinned a pathology as an invariant. CLOSED.

Same mistake §13 already fixed once, when `fillet_families == 2` turned out to be pinning an
OCC radius fallback as though it were a requirement.

**`test_the_arrival_angle_makes_the_junction_a_near_crack`** asserted `material_wedge > 340`
and measured **315.4**. Read the name: it pins the junction being nearly a crack. A design
whose spokes arrive less tangentially has a *less* crack-like junction, which is an
improvement, and it broke the test. Renamed to
**`test_the_junction_is_re_entrant_enough_to_be_singular`** and rewritten against a bound
that is a property of the BOX rather than of one design: `MAX_ARRIVAL_DEG` caps the arrival
angle for every genome the optimizer can reach, so the wedge is at least
`360 - MAX_ARRIVAL_DEG` = **295 deg everywhere**. That is all the sibling
`test_peak_stress_diverges_but_the_field_converges` needs, and it never has to be re-fitted
when a genome ships.

**`test_the_beam_to_wheel_ratio_is_not_a_constant`** — the 2x2 table in §13 proved the
statistic is a property of the gene box and does not depend on `genes` at all. It now calls
`set_min_wall(2.0)` for the duration of the draw, with a `finally` that restores
unconditionally (`tests/test_stage3.py` caches bounds in a module-scoped fixture, so a leaked
floor would surface somewhere else entirely). Gate 1's conclusion was never in question:
`correction_factor_is_defensible` is `False` in both boxes. What is pinned is the margin, in
the box the margin was calibrated in. **Re-deriving Gate 1 at a 1.2 mm floor is real work and
has not been done.**

Note `test_the_free_arc_fraction_is_not_constant_over_the_design_space` is the same shape —
a box statistic from the same `run_beam_blindness` call — and currently passes at 0.05. It
has been left alone rather than churned, but it is the next one to move if the floor does.

---

#### Item 2 — the mass budget was two unmeasured terms. CLOSED, and it exposed a stale constant.

`test_total_mass_matches_the_step_manifest_within_the_embed_difference` asserted
`-3.0% < m/manifest - 1 < -1.2%` and measured **-6.9%**. Its docstring decomposed the gap
into "~1.4% gusset plus 0.92% fillets" — and **the manifest published neither number**. Both
were fitted to one wheel. When the band broke there was nothing to look at.

The fillet term did not need estimating. `wheel_step_export` already builds
`wheel_nofillet.step` as its guaranteed-valid fallback, so the material the fillets add is a
subtraction OCC does exactly. The exporter now measures that solid's volume before
despecializing (same rule as `vol_true`, because the two are subtracted) and publishes:

```
solid.volume_nofillet_mm3   36042.6
fillets.volume_mm3           2372.53      = 6.176% of the solid
```

**6.18%, against the 0.92% the old docstring claimed.** That single correction is most of the
missing 6.9 points. Re-exporting was verified reporting-only: a key-by-key diff of the
manifest before and after shows **exactly two keys added and not one other value changed** —
`genome_hash`, `volume_mm3`, `mass_g_pla`, every `fillets.detail` row and `step_health` all
byte-identical.

The test is now a budget rather than a band: subtract the published fillet mass, and what is
left must be the gusset alone — positive, and under 1.5% of the solid. It measures **0.70%**.
It also pins the new field against a sign or unit slip (`fillets.volume_mm3` must equal the
difference of the two published volumes, and must be positive, because filleting a re-entrant
corner adds material).

**And that is how the stale constant surfaced.** Closing the budget with named terms left a
residual that would not go away under refinement:

| tier | mesh g | + fillet + gusset | residual |
|---|---|---|---|
| smoke | 44.2841 | 48.2360 | **-0.6060 g** |
| coarse | 44.3562 | 48.3081 | **-0.6781 g** |
| medium | 44.3641 | 48.3160 | **-0.6860 g** |
| fine | 44.3668 | 48.3187 | **-0.6887 g** |

Converging, not shrinking — so it is not discretization. It is
`EMBED_ALLOWANCE_PER_SPOKE_MM2 = 3.03`, and the promoted genome's actual gusset is
**0.98 mm² per spoke**. See item 6. The rewritten test does not use the constant at all.

---

#### Item 3 — four convergence failures. The common-mechanism hypothesis was WRONG.

The hypothesis was element aspect ratio: `MeshConfig` fixes `n_thick` and `n_span` as
*counts*, so root element thickness went 2.4774/6 = 0.413 mm → 1.2/6 = 0.200 mm while
spanwise length did not move, doubling the slenderness of every root element. Plausible, and
it would have explained all four at once. **It explains none of them.** Three separate causes,
found by three separate sweeps:

**`test_area_converges_second_order` — the reference, not the mesh.** `n_thick` has *literally
zero* effect on this statistic (8, 16, 32 give bit-identical orders — the area of a quad block
does not care how it is subdivided through the thickness). Extending the sweep two levels past
where the test stopped shows what is actually happening:

| genome | vs the beam reference | self-referenced |
|---|---|---|
| `350f4c7` | 1.986 2.061 **2.355** 5.103 **-3.300** | 1.962 1.979 **2.000** 1.998 |
| `36aed36` | 1.993 2.033 2.187 3.158 0.029 | 1.979 1.986 **2.001** 2.001 |

An order of 5.1 and then minus 3.3 is not a convergence rate — it is a difference of two
discretizations passing through zero. `reference_area` is `wheel_fea`'s beam-style line
integral: independent code, which is what makes it a good CROSS-CHECK, and an approximation
carrying its own error. The mesh converges at **exactly 2.000 on both genomes**. Nothing broke;
the promoted wheel has a smaller section (52.9 vs 145.7 mm²) so its absolute error reaches the
reference's floor one refinement sooner. **The GA/beam genome was already one level from
failing this.** Split into two tests — a self-referenced order, and a Richardson-limit
agreement with the beam integral to 1e-5 — and both got *tighter*, not looser.

**`test_peak_stress_diverges_but_the_field_converges` — the test's own documented false alarm,
one tier up.** It asserted `d2 < 0.3*d1` on the p99's successive differences.

| genome | smoke | coarse | medium | fine | d2/d1 | d2/p99 |
|---|---|---|---|---|---|---|
| `350f4c7` | 18.327 | 17.274 | 17.246 | 17.230 | **0.573** | 0.094% |
| `36aed36` | 8.842 | 8.782 | 8.612 | 8.605 | 0.042 | 0.082% |

`d2/d1` fails; `d2/p99` says the p99 has settled to under a tenth of a percent. A
successive-difference ratio is a divergence detector that only means anything while `d1` is
still real discretization error — once converged, both are tail and the ratio is arbitrary.
The old comment already named this failure mode at `smoke`, and the GA/beam genome **fails the
identical test at 2.816 if the window starts one tier lower**. The window had to be hand-picked
per design, which is the tell. Now pinned as the CONTRAST the docstring is actually about —
the max grows 38.9% over coarse..fine while the p99 moves 0.26%, a ratio of relative drifts
that is dimensionless and does not care which tier a design converges on.

**`test_the_sampled_patch_extent_is_biased_not_merely_noisy` — genuinely resolution, and it
clears at `coarse`.**

| genome | smoke | coarse | medium |
|---|---|---|---|
| `350f4c7` | **-0.3233%** | -0.0170% | +0.0195% |
| `36aed36` | -0.0275% | +0.0001% | +0.0030% |

On `smoke` the promoted genome's contact patch is `patch_half_deg` = 0.53 deg against a rim
element several times that: the whole contact set lives inside one element, so going from 6 to
20 Gauss points changes which elements are loaded at all. The failure is smoke-only and clears
by **5x** at `coarse`. That one test now builds its own `coarse` mesh; everything else in
`test_contact.py` stays on `smoke`, because everything else in it is a claim about the
*direction* of a bias, which holds at every tier.

**`test_the_correction_enters_at_first_order_in_the_load` — not resolution. Real.** See item 4.

---

#### Item 4a — THE GNL GATE. DECIDED: IT STANDS, AND IT IS THE MOST IMPORTANT THING §14 FOUND.

Swept for mesh sensitivity first, and there is none:

| genome | smoke | coarse | medium |
|---|---|---|---|
| `350f4c7` | 0.2050% | 0.2081% | **0.2089%** |
| `36aed36` | 0.0373% | 0.0382% | **0.0384%** |

Converged by `coarse`, mesh-independent to three digits, fitted exponent 1.037 against 1.011.
The correction still enters at first order — **what moved is the coefficient, not the
exponent**, and the assertion the test is named for passes.

**But 0.1% vs 0.2% at 1% of load is not what this is about.** `rel_diff = c·f^1.03`, so the
1%-load number and the service-load number are the same fact stated twice, and the gate is a
tripwire for the second one. The full ladder at `coarse`:

| load | linear mm | SVK mm | rel_diff | | GA/beam rel_diff |
|---|---|---|---|---|---|
| 0.01x | 0.019530 | 0.019570 | +0.208% | | +0.038% |
| 0.25x | 0.488241 | 0.514371 | +5.352% | | +0.963% |
| 0.50x | 0.976483 | 1.084109 | +11.022% | | +1.943% |
| **1.00x** | **1.952966** | **2.408898** | **+23.346%** | | **+3.953%** |
| 2.00x | 3.905931 | 5.939932 | +52.075% | | +8.193% |
| 3.00x | 5.858897 | 10.909429 | +86.203% | | +12.758% |

The GA/beam column reproduces M5's recorded numbers exactly (0.038% / 3.95% / 12.8%), which is
what makes the other column trustworthy. **The shipped wheel's axle drop at service load is
23.3% larger under SVK than the linear model says**, against 3.95% for the wheel it replaced.

**What that costs, in the two numbers the design is judged on.** Every headline for this
genome was computed under `kinematics="linear"` — `wheel_contact_problem`'s default:

- **Deflection.** `TARGET_DEFLECTION_MM` is 2.0 and the objective wants to hit it *exactly*.
  Linear says 1.953 mm — 2.4% under target, essentially on it. SVK says **2.409 mm, 20% over**.
  The design was tuned to a target it does not actually hit.
- **Stress utilisation.** The 0.799 headline is a linear-field number. Plain-spoke p99 goes
  17.274 → 19.746 MPa under SVK, **+14.3%**, which puts utilisation on the order of **0.91**
  against an allowable of 1.0. *(An estimate: it scales the reported figure by the p99 ratio,
  while the constraint itself aggregates a p-norm at p=30 on the `medium` rung. The margin
  falls from ~20% to ~9%; it does not vanish.)*

**So the gate has not been moved, and the case for moving it is now weaker, not stronger.**
`study_gnl.py` records it as *"written down BEFORE the study was run, per the plan's rule"* —
and the thing it was written to catch is exactly the thing that happened. Relaxing 1e-3 to
accommodate 0.209% would silence the only automatic warning that **linear kinematics no longer
describes this part.** The real open question is not the threshold; it is whether Stage 3
should be descending on a linear solve at all for a wall this thin. That is a scope decision
and it is a human's.

##### The footgun this turned up — FIXED

The first attempt at the stress half of that measurement returned **+169.5%**, and it was
wrong. `wheel_fem.gauss_stresses` takes `nonlinear=False` by default, and
`study_wheel_fea.stress_report` was calling it without the argument — correct for the linear
solves it was written for, and **silently wrong for an SVK one**, applying the
engineering-strain formula to a large displacement field. No warning, no NaN, and 46.56 MPa is
plausible enough to quote against a 25 MPa allowable. The true figure is 19.75.

`stress_report` now reads `res["meta"]["kinematics"]`, which `wheel_problem` sets down both
paths, and passes `nonlinear=` accordingly. Pinned by
`tests/test_gnl.py::test_stress_recovery_follows_the_solves_kinematics`, which asserts the
correct recovery AND that the wrong one is still visibly different — a guard that stops being
a guard if the two ever converge.

#### Item 4b — OPEN, still a human's call. The hub compliance share.

`test_the_rim_band_holds_a_large_minority_of_the_compliance`:
hub share **0.0321** against `< 0.03`, 7% over. Rim 0.306 and spoke 0.662 are both comfortably
inside. This is the one where the *direction* is surprising: thinner, floppier spokes should
push compliance toward the spokes and the hub share DOWN. It went up. The plausible cause is
`R_hub` dropping 1.5598 → 0.5790 — much less material at the hub junction — but that is a
hypothesis and it has not been measured. **Least urgent of the eight and the only one whose
sign is not understood.**

---

#### Item 5 — OPEN. `make hubcap` at the 1.2 mm floor.

Carried forward unchanged from §13. `HUB_CAP_SHARE` is calibrated on `t0` in [2.0, 2.6] and
the shipped genome sits at 1.2. What saves it today is *which branch binds* — the thickness
branch, at 0.52 × 1.2 = 0.624 mm exactly — so the slot share is not actually being
extrapolated on this part. A design that were both thin AND tight-slotted would extrapolate it
for real and nothing would say so.

---

#### Item 6 — OPEN, and NEW. `EMBED_ALLOWANCE_PER_SPOKE_MM2` is a `t`-proxy, exactly like `MIN_JUNCTION_OVERLAP_MM3` was.

Fell out of item 2. The constant is **3.03 mm²**; the promoted genome's measured gusset is
**0.98 mm² per spoke** (settling from 1.01 at `coarse` to 0.978 at `fine`). Root thickness
fell by 2.06x and the gusset fell by 3.1x, so it is not a constant and it is not linear in `t`
either — the same shape of error §12 removed from the junction check, where a fixed mm³ floor
turned out to be a proxy for `t0` and nothing else.

**Do not guess a new number.** Replacing 3.03 with 0.98 would only re-stale it on the next
genome; what is needed is the scaling law, derived from `wheel_step_export._embed` the way
`wheel_geometry.junction_bite` was derived. The exporter now publishes
`solid.volume_nofillet_mm3`, which makes the true gusset measurable per genome for the first
time — subtract the mesh's own modelled volume from it — so the calibration data is one export
away for any design on disk.

**One thing to know while it stands.**
`test_the_embed_difference_from_the_shipped_step_is_the_known_amount` compares
`reference_shipped_step_mm2 - total_modelled_mm2` against `n_spokes * 3.03`, and
`reference_shipped_step_mm2` is *defined* as `reference_modelled_mm2 + n_spokes * 3.03`. Since
the mesh agrees with `reference_modelled_mm2` to 4e-5, that test is very nearly tautological:
it checks the mesh matches its own geometric reference, which is worth checking, but it cannot
see the constant being wrong and did not.

---

#### The gate after all of it

Item 1 closed two failures, item 2 one, item 3 three. **`make test`: 2 failed, 430 passed in
1452 s (24:12)** — nothing else meaningful on the box, and the export and the probe sweeps had
already finished. (423 → 430 rather than 429: splitting `test_area_converges_second_order` into
a convergence claim and a cross-code claim added a test.) The two reds are the GNL gate and the
hub compliance share, both item 4, both deliberately left.

Item 4a then added `test_stress_recovery_follows_the_solves_kinematics`, so the count should
be **2 failed / 431 passed**; that arithmetic has NOT been confirmed by a full run. What was
re-run after item 4a is `tests/test_gnl.py` and `tests/test_wheel_fea.py` — the two files it
touched — and both come back with only the two known reds. Nothing was
re-tuned to fit: of the six thresholds touched, four got tighter (the area order is now
measured against 2.000 rather than a contaminated reference, the beam ratio is pinned in its
own box, the p99 gained a 10x separation requirement, the mass budget went from a fitted 1.8
point band to a named 1.5% bound) and two moved a test to a mesh that resolves what it is
measuring.

---

### 15. STAGE 3 WAS DESCENDING ON THE WRONG PHYSICS. It can now descend on the right physics, and the wheel that descent finds cannot be built. **NOTHING PROMOTED** (2026-08-10).

**BEFORE THIS MILESTONE, EVERY STAGE-3 NUMBER IN THIS REPO WAS A LINEAR-KINEMATICS NUMBER.**
`wheel_contact_problem` defaults to `kinematics="linear"` (`src/wheel_fem.py:1693`) and
nothing in the Stage-3 path had ever overridden it. That is the one sentence to carry out of
this section. Those numbers are not *wrong* — they are answers to a different question, and
§14 item 4a is where the question got asked.

**`best_solution.json` IS UNCHANGED and still holds `350f4c7`.** Nothing was promoted,
nothing was re-baselined, and the banner at the top of this file still describes the shipped
wheel. `export/wheel.step` was regenerated from that same unchanged genome (manifest
`genome_hash` `350f4c7`) purely to obtain a control for the export check below.

The working notes are **`SVK_PLAN.md`**, seven steps, each with its own pre-registered gate
and its own Record block. This is the summary; that file is the evidence.

#### What landed in the code, and it is small

The plumbing already existed — `kinematics` rides `**problem_kw` from `Evaluator` all the way
to `wheel_contact_problem`, the adjoint kernels already dispatch on `prob.nonlinear`
(`src/wheel_adjoint.py:161, 190, 400`), and `wheel_pool_worker.py:66` already splats it. What
was missing was a CLI flag.

- **`src/wheel_stage3.py`** — `--kinematics {linear,svk}`, default `linear`, forwarded to
  **both** optimizers, recorded in `search_block` and in the run record's settings, and
  **printed in the console banner** (`:954`). The record reads `ev.problem_kw` — the very
  dict the `Evaluator` splats into the solver — so the record cannot disagree with what was
  solved. `search_block` has **no `getattr(args, "kinematics", "linear")` fallback** on
  purpose: a default there would report "linear" for an SVK run whose caller forgot the
  field, which is the exact misattribution the key exists to prevent.
- **`studies/study_gradient.py`** — `--kinematics`, threaded as a plain keyword argument
  through all 15 call sites that build a problem or a solve, **never as module state**, and
  recorded in `rep["settings"]`.
- **`studies/study_svk_rescore.py`** + `make svk` — scores any set of genomes under **both**
  kinematics with no optimizer involved. Deliberately **out of `make studies`**, for the
  reason `m8bi5`/`m9buck`/`hubcap` are: it measures the wheel, not the commit.
- **`make svk-shipped` / `svk-elite10` / `svk-medium`** — the descents, modelled on
  `prod9`/`prod10`, with distinct `--out` AND `--best-out` so two runs cannot clobber each
  other's genome.
- **Tests** — the S13 pooled-equals-serial contract under SVK (sharing one helper with the
  linear test so the two kinematics cannot drift into two standards, plus a sentinel
  asserting the two kinematics return *different* answers, without which the equivalence
  would hold no matter what the pool did with the key), and
  `test_the_run_record_carries_the_kinematics_it_actually_descended`.

**`--kinematics linear` is inert**, proved where it is checkable: `linear` is exactly what
`wheel_contact_problem` already defaults to. Built at `smoke` on the shipped genome, 15
problem fields compared — the control (default vs default) differs on `contact` and `dofmap`
by object identity alone, the test (default vs explicit `linear`) matches the control
exactly, and a **sentinel** (default vs explicit `svk`) also moves `meta` and `nonlinear`,
which is what gives the comparison the power to see a real change.

#### THE PREREQUISITE: the adjoint is correct under SVK. All ten gates, thresholds unmodified

M7's gate had only ever been run under linear kinematics. Two full `coarse` runs, this tree,
this genome:

| gate | SVK | linear | threshold |
|---|---|---|---|
| **G1 unrolled** | **5.893e-11** | **4.555e-11** | **1e-8** |
| G2 force identity | 2.970e-11 | 2.462e-11 | 1e-12 (residual exactly 0.0 in both) |
| G3 mesh coords mm | 3.553e-14 | 3.553e-14 | 1e-9 |
| G4 plateau rel / decades | 5.616e-06 / 2 | 2.910e-06 / 3 | 1e-4 / ≥1 |
| G5 directional | 3.379e-06 | 8.056e-06 | 1e-5 |
| G6 sweep median | 6.626e-06 | 6.298e-06 | 1e-3 |
| G9 secant | 1.079e-06 | 3.576e-07 | 1e-5 |

**G1 is the one that decided the arc.** It unrolls the Newton loop and differentiates it with
`jax.grad`, so it contains no finite difference anywhere and its tolerance is set by linear
algebra rather than by step size. The SVK adjoint reproduces brute-force differentiation of
its own SVK solve to 5.9e-11 against a 1e-8 gate. **SVK is not uniformly the
worse-conditioned side** — it is 2.4× better on G5 — and no gate was given any slack.

Two things this step found on the way, both of them the "always measure the control" rule
catching a live error:

- **`--quick` IS NOT A GATE.** The quick SVK run came back `OVERALL: FAIL` on G5/G6/G7/G9.
  The same run under **linear** also comes back `OVERALL: FAIL`, on the committed default:
  it is a reduced-fidelity smoke mode whose step ladder drops the rungs where the FD plateau
  lives. G5 reads 1.588e-05 linear against 1.579e-05 SVK — SVK marginally the *better* of
  two failures. At full fidelity both G7 and G9 are clean and SVK is again the better side
  on G9.
- **`studies/study_gradient.json` IS STALE, and is NOT refreshed by this arc.** 3755 of 4239
  non-timing leaves differ from a fresh run at the same settings, including the physics
  (linear axle drop **1.8746 mm** fresh against **1.6546 mm** committed). The dates say why:
  the artifact is 2026-08-03, `best_solution.json` was replaced 2026-08-06. **The committed
  report describes a wheel that is no longer in the file it names.** This arc used it as a
  control for about an hour and reported a +39.6% SVK difference on the strength of it; the
  real figure, against a control measured in the same session, is **+23.26%** — which
  independently reproduces §14's +23.346%.

**And one repair to `study_gradient.py` itself.** `_phase_sweep` carries a converged
displacement field across a 0.5 deg phase jump, and under SVK that can land where `K(u)` is
genuinely indefinite (`NewtonDivergedError`, "the tangent is not positive definite") at a
phase where a **cold solve converges without complaint**. Under linear kinematics it cannot —
§0's H2(a) measured `K`'s dependence on `u` at exactly 0.000e+00. Repaired by falling back to
a cold solve and **returning the phases where that fired** (`cold_retry_phases_deg`) rather
than swallowing them: it changes which starting guess is used, never which equilibrium is
reported. It fires 8 times at `smoke` and **0 times at `coarse`** — a smoke-mesh phenomenon.
Stage 3 is not exposed to it: its cross-step warm start is *scalar indentations* seeding a
secant, not displacement fields carried across phases.

#### WHAT SVK COSTS: 1.36× time, 1.05× memory — and the penalty is in the GRADIENT

`coarse`, 8 phases, `uniform`, fidelity off, from `best_solution.json`, s/eval read off
`elapsed_s / n_objective_calls` in the run record rather than hand-timed.

| | s/eval serial (all/steady) | s/eval, 4 workers | Newton iters/solve | peak anon RSS, 4 workers |
|---|---|---|---|---|
| linear | 193.7 / 129.2 | 72.5 / **45.9** | 26.00 | 12.56 GiB |
| svk | 234.2 / 165.7 | 91.1 / **62.3** | 26.75 | 13.16 GiB |

"steady" drops call 0, which is JIT warm-up and is paid once per run. **SVK adds almost no
Newton iterations** — contact already spends 26 of them under linear kinematics, because
`wheel_fem.solve` routes on `prob.nonlinear or prob.contact is not None`
(`src/wheel_fem.py:1274`), so the Newton loop was always there. Backtracks actually *fell*,
8 → 4. The forward solve is only **1.13×** while the full evaluation is **1.36×**, so most of
the penalty is the SVK tangent assembly and the nonlinear vJP kernels. **There is no
iteration count to trade away**, which is worth knowing before anyone tries to buy the cost
back with a solver tolerance.

Two unplanned corroborations that the rig measures what the Makefile's own numbers were
measured on: 12.56 GiB for a 4-worker linear descent reproduces `make prod10`'s help text
("~12.7 GB anon"), and 3.9 h projected for a 300-step linear descent reproduces `prod9`'s
"~4 h". **The 62.3 s/step projection then held twice**, on two 300-step runs, at 58.2 and
59.1 s/step — 6.6% conservative.

Pooled-equals-serial holds under SVK: values **bit-identical**, gradients within 1e-14.

#### THE RE-SCORE: §14's "~0.91" was pessimistic, and the design ranking INVERTS

`make svk`, `medium`, 8 uniform phases, 47.8 min, six distinct designs under both kinematics.
The driver is gated on a **pre-registered control** (`GATE_CONTROL_REL = 0.02`) that
reproduces §14's ladder to five significant figures — 23.346% and 3.953%, errors 1.6e-5 and
2.6e-5 — so the table is comparable to §14 rather than merely internally consistent. A
*second*, unplanned control landed with the first row: `350f4c7 linear` came back
`loss 32.73762364435313` — every digit of the committed `stage3_minwall_best_1.2_medium.json`,
written five days earlier by a code path this driver shares nothing with.

| genome | kin | drop mm | err % | util | mass g | loss |
|---|---|---|---|---|---|---|
| `350f4c7` shipped | lin | 2.0193 | +0.97% | 0.799 | 39.19 | 32.7376 |
| `350f4c7` shipped | **svk** | **2.3947** | **+19.74%** | **0.875** | 39.19 | **129.8963** |
| elite10 | lin | 2.0041 | +0.21% | 0.594 | 58.66 | 49.7265 |
| elite10 | svk | 2.1418 | +7.09% | 0.624 | 58.66 | 62.2812 |
| minwall 1.4 | svk | 2.2847 | +14.24% | 0.799 | 42.82 | 86.0519 |
| minwall 1.6 | svk | 2.2146 | +10.73% | 0.712 | 47.03 | 67.7961 |
| minwall 2.0 | svk | 2.1416 | +7.08% | 0.624 | 58.55 | 62.2346 |

*(`minwall 1.2` **is** `350f4c7` bit-for-bit, `max|Δgenes| = 0.0`; the two rows agree on every
float but `elapsed_s`, which is a free determinism check on the rest of the table. `36aed36`
is infeasible under **both** kinematics — pre-registered as expected, it predates the
fillet-cap work — and is a control on the correction, not a candidate.)*

**Three findings, and the third is the one with the blast radius.**

1. **SVK moves exactly ONE term.** `mass`, `smoothness`, `phase_ripple` and all nine barriers
   are **bit-identical** under both kinematics; `deflection` carries the entire difference —
   0.2330 → **97.3916** on the shipped genome, 418×, and 75% of its total loss. Nothing else
   in the objective is a function of the strain measure, which is why the `stress` *barrier*
   stays at 0.0 even as utilisation climbs 0.799 → 0.875.
2. **The correction is a function of the DESIGN, not a constant.** It falls monotonically with
   stiffness — +18.6% → +13.4% → +10.2% → +6.9% across the four `minwall` arms, reproducing
   on a fifth design that reached similar stiffness by a different route (elite10, +6.87%).
   §14 measured it on two genomes and quoted +23.346%; **it is not one number, it is a curve.**
3. **The ranking INVERTS on every comparison in the table.** The linear loss column is
   monotone *increasing* in wall thickness; the SVK column is monotone *decreasing*.
   `350f4c7` was promoted over elite10 because it won by 17 points under linear; under SVK it
   loses by 68. **The `minwall` sweep of §8 that chose the 1.2 mm floor ran entirely under
   linear kinematics, and this table reverses its ordering over the whole 1.2–2.0 mm range.**

**And the gate's own question, answered: the shipped genome IS FEASIBLE under SVK.** Every
barrier exactly 0.0, utilisation **0.8754** against an allowable of 1.0. §14's "on the order
of 0.91" was an estimate built by scaling a reported p99 by a ratio; the number the
constraint actually computes is 0.875. **§14 was pessimistic by ~4 points of margin.**

Read the `p30 util` column in `studies/study_svk_rescore.json` as a diagnostic and nothing
else — it sits at 10.747 for the shipped genome, and if the constraint used p=30 the promoted
wheel would be infeasible by 10×. It is not mesh-convergent (GCI 63%, M8b-i.5), which is
exactly why `STRESS_NOMINAL_P = 4.0` is the constraint. The driver asserts its p=4 probe
reproduces `stress_utilisation` to 1e-12, so the two columns are the same construction
differing only in the exponent — the line that stops the diagnostic and the verdict drifting
apart, which is precisely how "~0.91" came to stand in for a number nobody had computed.

#### THE DECISION, taken on those numbers and written down before anything launched

**Descend under SVK.** Three things had to be true and all three were measured, not assumed:
the SVK adjoint is correct (G1, unmodified); it is affordable (1.36×, 5.3 h per start, 16G);
and **there is something to descend to** — three Adam steps under SVK took the shipped genome
117.766 → 33.436 and pulled the drop 2.369 → 1.982 mm for +0.66 g, while the same three steps
under linear could not improve on the start at all. **The shipped genome is a local optimum of
the LINEAR objective and is demonstrably not one under SVK.**

Re-targeting `TARGET_DEFLECTION_MM` under linear was rejected not on the anticipated ground
(the correction is load-dependent, `c·f^1.03`) but on finding 2: **the correction is a
function of the design**, so a re-targeted constant is calibrated to the design that existed
when you calibrated it, and the optimizer's whole job is to move the design. It is not merely
approximate, it is unstable under the thing it is meant to enable. Accepting 2.409 mm was
*available* — that is the feasibility answer above, and it was not available before this arc —
and was rejected on value: a 19.74% spec miss while the same table holds feasible designs at
+7% and a 3-step probe moves the shipped genome most of the way for 0.66 g.

**The descents held the 1.2 mm floor**, deliberately, even though the ladder says the SVK
optimum over the *measured* designs is at the thick end. A floor is a **constraint, not a
target** — if SVK wants thicker walls it can walk there from inside the same box — and holding
the box identical across both starts is what makes them comparable. The outcome is
informative either way. It came back binding; see the successors.

#### THE DESCENTS: two starts, both pass, and the answer is a SHELF not a point

Two 300-step `coarse` descents, sequential, each under `systemd-run --user --scope -p
MemoryMax=16G -p MemorySwapMax=0 --collect`, from starts 19 g apart.

| run | start | steps | loss | mass g | defl err (svk) | util | worst barrier | wall |
|---|---|---|---|---|---|---|---|---|
| 1 `ae7092c` | `350f4c7` | 300/300 | **30.8207** | **37.451** | **−0.043%** | 0.8989 | **0.0** | 4.90 h |
| 2 `c4f207c` | elite10 | 300/300 | 30.8245 | 37.449 | −0.044% | 0.9085 | **0.0** | 4.94 h |

**Both clear all three pre-registered clauses, unmodified** — every one of the nine barriers
exactly 0.0, deflection 7× inside ±0.3%, and mass below the shipped 39.194 g. Run 1 took the
loss 117.766 → 30.8207 with **0 rejected steps and 0 events**; the descent never fought the
line search. Term by term, step 0 → 300: `deflection` 85.2617 → **0.00047**, `mass` 32.2145 →
30.7820, `smoothness` 0.2902 → 0.0382, every barrier 0.0 throughout.

> **The gate pre-registered its own likely breach and it did not fire.** Written before the
> re-score, the mass clause was expected to fail — if SVK's answer to a 19.74% deflection miss
> is more material, "mass below 39.194 g" asks the descent to beat the shipped wheel on the
> one axis it was over-optimized on. Instead **SVK did not cost this design grams; it saved
> 1.74 g** while moving deflection from 2.369 mm to 2.000 mm. The clause "SVK costs this
> design N grams" does not fire, and it is recorded here because it was written down first.

**The two runs agree to 0.012% on loss and 0.008% on mass, and that is NOT a shared point
optimum.** The genomes differ: `cx1` by 7.28%, `cy1` by 6.93%, `cy4` by 4.27%, while
`t0..t3` sit on the 1.2 mm floor in both and `R_rim` is identical to six decimals. **The
objective is flat along a manifold** — the spline control points move up to 7.3% for a
fourth-decimal change in loss. Report it as "both starts reach the same basin and the same
headline numbers", never as "the optimizer found THE answer".

Run 1 was **reproduced through an independent driver** while run 2 was in flight
(`study_svk_rescore.py --extra`, additive so the driver at its defaults still reproduces the
Step 3 artifact unchanged): drop 1.9991, util 0.8989, loss 30.8207, every digit. So the
Stage-3 run record is not reporting an internal state the saved genome does not encode.

**Run 2 hit one `solve_reject` at step 128, handled, and it is worth a line.**
`solve_wheel_contact` (`src/wheel_fem.py:1841`) is a secant on indentation with
`tol_rel=1e-8`; it stalled at 66.723265 N against a 66.7233 N target — a residual of
**5.2e-7 relative, 52× above the tolerance it is asked to hit**. The load is physically
reached to within a part in two million. What fails is the *outer* secant's ability to resolve
a force difference smaller than the noise floor of the inner Newton solve that produces it,
and SVK raises that floor. The function raises rather than returning the state, which is
correct and documented. `wheel_pool_worker.py:88-98` reports it to the parent as the
`solve_reject` that `descend` already knows how to handle — it prints a traceback and is not
a crash. **Run 1 is unaffected: 301 calls, `n_reject_cumulative` 0.** The tolerance was **not**
loosened; see the successors.

#### THE FIDELITY TRAP: the ±0.3% deflection gate is satisfiable at exactly ONE rung

The `coarse` candidate `ae7092c` was re-scored at `medium` before promotion, with the control
on — and it reads **+1.65%**, 5.5× the gate. Not promoted. That check was pre-registered in
the descent's own record before the descent finished: *"a deflection converged to −0.043% at
coarse is NOT thereby inside ±0.3% at medium ... that is a finding about the rung the descent
was run on, not licence to promote anyway."* The rule holds even though the check was one
this arc added rather than one the plan pre-registered — especially then.

**The control is what makes it readable, and it says the gate was never a `medium` gate.** The
INCUMBENT fails ±0.3% at `medium` too — **+0.97%**, under the very kinematics it was descended
on. **No design in this repo has ever met ±0.3% at `medium`.** So the response was to
re-converge at `medium`, not to move the number: `make svk-medium`, 100 steps warm-started
from `ae7092c`, 6.29 h, 224 s/step against a projected 273.

`bc77614` **passes every clause at `medium`**: all nine barriers 0.0, deflection **−0.041%**
(1.99919 mm), mass **37.414 g** (−1.781 g against the shipped 39.194). It is also lighter and
lower-loss than the coarse candidate it replaces.

**And the fidelity check, pointed back at `coarse`, gives the mirror image — which is the real
result here.** The medium answer reads **−1.71%** at coarse. The coarse answer reads +1.65% at
medium. **The two rungs disagree by ~1.7% on this wheel and no design can satisfy ±0.3% at
both.** Which rung the gate is stated against is a **choice, not a property of the design**;
this arc chose `medium` because it is the finer and because §14's control ladder is stated
there. The honest sentence is that the wheel is now specialised to a rung as well as to a
kinematics, and a third rung would move it again.

**A cheap process lesson.** Both descents ran with `--fidelity-check-every 0`. `descend` has
had the machinery for exactly this since §1 item 3 (`src/wheel_stage3.py:384`,
`_fidelity_check` at `:272`) — a pure observation that cannot redirect the descent, but with
`--fidelity-check-every 25 --fidelity-check-config medium` the coarse/medium gap would have
been on the record at step 0 instead of after 9.8 h of descending. Turning it off saved
perhaps 4% of wall clock and cost the arc a run.

#### WHY NOTHING IS PROMOTED: `bc77614` clears every FEA gate and is not buildable at the stress concentration it was priced at

`make export EXPORT_GENOME=stage3_svk_best_medium.json`, with the same export run on the
incumbent as the control this file's rules require:

| genome | worst wedge | hub fillets built | `kt_error_pct` |
|---|---|---|---|
| `350f4c7` shipped | 328.0 deg | 24/24 @ 0.579 mm | **0.0%** |
| `bc77614` svk-medium | 308.0 deg | 12 @ 0.579, 12 @ 0.418 mm | **+11.9%** |

**The incumbent builds exactly as modelled. The candidate does not** — and the control is the
only reason that sentence can be said with confidence. This is a regression this arc
introduced, not one it inherited. §13's +0.0% at both junctions was the first shipped part
whose built fillets matched the ones its stress model priced, and it is worth exactly this
much.

What it costs, in the only units that matter:

```
Kt at the hub    modelled 2.0235      as built 2.2643      +11.9%
peak stress      modelled 294.02 MPa  as built ~329.01 MPa
UTILISATION      modelled   0.9347    AS BUILT   1.0461    <- INFEASIBLE AS BUILT
```

Everything else in the export is clean, which is what makes the failure legible rather than
ambiguous: OCC valid, 1 solid, bbox 100.00 × 100.00 × 22.40 mm, BRepCheck valid, no
self-intersection, 0 degenerate, min curvature R 0.4184 mm against the 0.25 floor, junction
bite floor satisfied. **Only the fillet feasibility is red.**

**Lowering `R_hub` by hand until modelled == built was rejected.** It is fitting the geometry
to the check after seeing the check fail; it would have to be done by hand *precisely
because* the optimizer cannot do it; and it leaves the same blind spot in place for the next
design. The defect is that the objective cannot see buildability, and the fix belongs in the
objective.

#### THE FOUR DEFECTS IN THE OBJECTIVE, in the order they collected their debt

This is the part of §15 that outlives the numbers. All four were **measured** in this arc, and
all four were deliberately left alone, because acting on any of them mid-arc would have been
re-fitting a gate to the run that breached it.

1. **`stress` HAS ZERO GRADIENT BELOW `util = 1.0`.** It is `soft_barrier(util - 1.0, 4000)`
   (`src/wheel_objective.py:1027`) and `soft_barrier` is `scale * max(0, v)**2` (`:290`), so
   it is identically zero *and identically flat* for every `util <= 1.0`. Below the knee the
   optimizer cannot see stress at all; it sees mass, and it thins the wall. **The barrier is a
   wall to stop at, never a cost to trade against.** Measured: the `stress` term was > 0 on
   **0 of 602 descent steps**.
2. **THEREFORE `R_hub` AND `R_rim` ARE DEAD GENES.** The only paths from a fillet radius into
   the loss are `stress` and the fillet barriers, and both are identically flat unless
   breached. Over all 602 steps of both descents, **`R_rim` had a nonzero gradient on 0 steps
   and `R_hub` on 2** — and those two steps are *exactly* the two where the `fillet_cap`
   barrier was live. They stayed frozen to six decimals through another 100 steps at
   `medium`, which rules out the one benign explanation: it is not a coarse-mesh artefact.
   **Run 1's `R_hub` 0.5790 and `R_rim` 2.7495 are not optimisation results.** They are
   constants inherited from `best_solution.json` and carried untouched, and the same blind
   path was in place for every Stage-3 run behind §6, §8 and §14. The search is nominally
   14-dimensional; with all four thicknesses on the floor the live subspace is the **8 spline
   coordinates** — precisely where the two runs still disagree by 7.3%.
3. **THEREFORE THE DESCENT SWUNG THE HUB ARRIVAL SHALLOW WITH NOTHING OBJECTING** (wedge
   328 → 308 deg), and it could not have compensated by asking for a smaller radius either,
   because `R_hub` is exactly the gene it cannot move. The exporter's own diagnostic reaches
   the same place unprompted: *"what is left is the shallow corner of a near-tangent arrival,
   which is the arrival angle, i.e. the genome."*
4. **THE 1.2 mm WALL FLOOR IS DOING THE STOPPING, AND ITS JUSTIFICATION NO LONGER HOLDS.**
   Utilisation climbed monotonically through the descent and **plateaued at 0.899 for the last
   180 steps** — not because anything valued the remaining margin, but because at step 300 all
   four wall genes sit **on the 1.2 mm floor** (`final.bound_saturation`: t0..t3 all "low").
   The run ran out of wall to thin before it ran out of stress margin; set the floor lower and
   the same blind gradient would keep going. And the floor itself was chosen by §8's `minwall`
   sweep **under linear kinematics**, whose ranking finding 3 above inverts.

*(An in-flight extrapolation in the working notes projected utilisation reaching ~0.99 by step
300 from the 50→100 slope. It did not; it stopped 0.10 short of the knee. The mechanism above
is unchanged — what was wrong was the projection, and the run was saved by a different
constraint than the one that was worrying.)*

**Utilisation is the one number that got worse, and it should be read as the cost of this
arc:** 0.799 (shipped, linear, medium) → 0.875 (shipped, **svk**, medium) → 0.935 (`bc77614`,
svk, medium). Roughly **half of that is not a design change at all** — it is the correction
from measuring the same wheel honestly. The rest was spent by an optimizer that cannot see
stress below 1.0. Both halves are real.

One more caveat on the descent's own trace: `max_stress_mpa` grew **+37%** (160.6 → 220.2 MPa)
while the utilisation the constraint sees grew **+4.7%**. The two decouple because the
constraint is `Kt * pnorm(p=4) / 25.0` and a p=4 aggregate is deliberately insensitive to a
singular corner peak — that is what `Kt` exists to bridge, and p=30 is not mesh-convergent. The
p=4 number is the right one to gate on. But **the aggregate is tracking the field and the
corner moved more**, and that is the same corner the export then refused to fillet.

#### What is now stale, and what to read as a linear-kinematics number

- **Every Stage-3 deflection and utilisation number in §1–§14 is a linear-kinematics number.**
  §13's headline 0.80 utilisation and §14's 0.799 are `medium`-rung linear figures; the same
  wheel reads **0.875** under SVK. §13's implicit "2.0 mm deflection" is 1.953 linear and
  **2.409 SVK**. Nothing in those sections needs correcting — they need reading with the
  kinematics named.
- **`studies/study_gradient.json` is stale** for a second and unrelated reason (above): it
  describes the pre-`350f4c7` genome. Not refreshed here.
- **Every study driver in `studies/` still defaults to `kinematics="linear"`**, which remains
  the repo-wide default and was held so on purpose: every committed artifact must still
  reproduce bit-for-bit with no flag passed. A new default is a re-baselining and this repo
  does not re-baseline silently. `study_gradient.py` and `study_svk_rescore.py` are the two
  that can now be told otherwise.
- **`36aed36` is barely affected and that is not luck** — +3.95% against +23.3%. The
  correction tracks compliance, and the old wheel is 74 g of it. Any comparison between the
  two genomes under linear kinematics is comparing one number that is nearly right against one
  that is 20% off.

#### The gate

`make test` had not been run in full since the arc's baseline (**431 passed / 2 failed**,
1374.52 s, reproducing §14's predicted arithmetic exactly, both reds the deliberate ones —
the GNL gate at `small_load_rel_diff` 0.0020499 and the hub compliance share at
0.032076694850181206, each matching its recorded value to the digit).

**Re-run at the close of this arc, and it was measured rather than inferred: `make test` —
433 passed / 2 failed in 1468.57 s (24:28).** The arithmetic predicted 433/2 (Step 0's 431/2
plus exactly the two tests this arc added — `test_a_pooled_SVK_evaluation_matches_the_serial
_one` and `test_the_run_record_carries_the_kinematics_it_actually_descended`) and the run
confirms it. **The two reds are the same two, and the hub compliance share reproduces
0.032076694850181206 — every digit of §14's recorded value, and of Step 0's.** No new red,
nothing accommodated, and the +2 is fully accounted for. §14 had to close on an unconfirmed
sum; this one did not.

#### The successors, in priority order

All five were found by this arc and all five are named rather than acted on.

1. **MAKE THE OBJECTIVE SEE BUILDABILITY.** This is the one that has to be fixed before **any**
   SVK descent can ship, and it subsumes 2 and 3 as far as promotion is concerned. Nothing in
   the loss prices the hub arrival angle or the fillet the exporter can actually cut. The
   exporter already computes `kt_built` — the missing piece is a term that charges the
   difference, which would also give `R_hub` its first real gradient.
2. **GIVE THE OBJECTIVE A STRESS-MARGIN TERM.** Defect 1. It does not merely stop the optimizer
   spending margin it is not charged for — it **unfreezes two design variables that no run in
   this repo's history has been able to move**.
3. **RE-DERIVE THE MINIMUM-WALL FLOOR UNDER SVK.** Defect 4. The floor is load-bearing in every
   answer this arc produced and §8's justification for it is reversed by the re-score table.
4. **PUT A MESH-CONVERGENCE STUDY ON `axle_drop_mean_mm`.** Give the deflection QoI the GCI
   treatment M8b-i.5 gave the stress QoI, then state the ±0.3% gate against an extrapolated
   value instead of against whichever rung the descent happened to run on. Today the gate is
   satisfiable at exactly one rung and the choice of rung is undeclared.
5. **SET THE LOAD-CONTROL TOLERANCE FROM THE INNER SOLVE'S NOISE FLOOR.** `tol_rel=1e-8` in
   `solve_wheel_contact` is not universally achievable under SVK. Measure the floor, then set
   the outer tolerance from the measurement — **do not pick a looser round number**, and do
   not touch it on the evidence of the run that breached it.

#### Artifacts

Search results and measurements, none of them gates except where noted. At the repo root:
`stage3_svk_best_shipped.json` (`ae7092c`, run 1) and `stage3_svk_shipped.json` (its run
record), `stage3_svk_best_elite10.json` (`c4f207c`) / `stage3_svk_elite10.json`,
`stage3_svk_best_medium.json` (`bc77614`, the `medium` re-convergence) /
`stage3_svk_medium.json`. In `studies/`: `study_svk_rescore.json` (the Step 3 table, and it
**is** gated — `GATE_CONTROL_REL`), `study_svk_step6.json` (the same driver's `medium` check
on the coarse candidate, the one that stopped the promotion), `study_gradient_svk.json` and
`study_gradient_lin_check.json` (the ten adjoint gates under each kinematics, `"pass": true`
in both), plus their `--quick` counterparts, which are **not** gates and fail under both
kinematics. In `export/`: `stage3_svk_best_medium.step`, its `_nofillet` companion and its
manifest — **the artifact that refused**, kept because the +11.9% `kt_error_pct` is the arc's
terminal finding and re-deriving it costs an export.

**`export/wheel.step` was regenerated from the unchanged incumbent `350f4c7`** to obtain the
control for that comparison; its manifest `genome_hash` says so.

#### What this arc established, stated without the hedging

SVK is the honest kinematics for this part. The shipped wheel deflects **2.39 mm, not 1.95**,
and carries **0.875 of allowable, not 0.799** — and it is still feasible, which was not known
before. A design exists that is **12× closer to the deflection target and 1.78 g lighter**
under that kinematics. And the objective has **four named, measured defects** that together
explain why that design cannot ship. The deliverable of this milestone is a measurement, not
a promotion, and the plan pre-registered that outcome: *"a run that does not clear these is a
result, not a failure — record it and go back."*

---

### 16. THE OBJECTIVE CAN NOW SEE BUILDABILITY. **PROMOTED** — `best_solution.json` is `e4219f3` (2026-08-11).

§15 ended with a wheel that was 12× closer to the deflection target, 1.78 g lighter, and
**impossible to build**: `kt_error_pct` +11.9% at the hub, as-built utilisation 1.046, OCC
filleting 12 of 24 corners at the radius the optimizer had asked for. The deliverable was a
measurement and the instruction was to go back and fix the objective. This is that arc.

**The wheel that ships now builds 24/24 at both junctions at the full requested radius,
`kt_error_pct` +0.0% / +0.0%.** It weighs 37.568 g against 39.194, deflects 1.99996 mm against
a 2.0 mm target (**−0.002%**, where the incumbent was **+20.5%** once re-priced), and carries
0.996 of allowable priced / 0.987 as built. Every barrier is exactly 0.0.

#### The defect was not that the cap was missing. It was that the cap was WRONG, in both branches

`hub_fillet_cap_mm` existed since §5 and returned `min(by_slot, by_thickness)`. The `min`
structure was right. **Both of its arguments were the wrong shape.**

- `by_thickness` was `0.52 * t0` — a function of a gene that says nothing about the corner.
  §15's two candidate genomes have **identical `t0` = 1.2 and identical `R_hub` = 0.578951**,
  and OCC fillets all 24 hub corners on one and only 12 on the other. The old cap returned
  **0.6240 for both, to sixteen digits.** What they do not share is the **hub arrival angle**:
  19.68° against 48.89°. That was the missing variable, and it was already differentiable —
  `control_points` locks P0 at the origin, so arrival is `asin(|cx1| / hypot(cx1, cy1))`, a
  function of two genes and nothing else.
- `by_slot` was `0.5 * R_hub_ring * radians(void)`. It had **never been the binding branch
  before**, so its constant had never been tested. When the arrival fix made it bind, it was
  found to over-promise by up to **1.62×**. Re-fitted under the smallest of eight measured
  ratios, 0.3096.

The replacement, fitted on OCC ground truth and calibrated over [5°, 60°]:

```
by_thickness = t0 * (0.505 - 0.48 * (1 - cos(arrival_hub)))
by_slot      = 0.30 * R_hub_ring * radians(hub_void_deg)
cap          = min(by_slot, by_thickness)
```

`(1 - cos)` rather than a quadratic because it is bounded, monotone on [0°, 90°], and flat at
0° where a tangential arrival should be insensitive. The fit sits **under all 14 sweep
stations** (1.45–3.29% under thin, 4.00–11.61% under thick) and 3.3% / 3.1% under two
out-of-sample designs. **The conservatism is deliberate and it is the design margin** — a point
that matters again below. Cap ÷ OCC-worst across five designs went 1.067 / 1.615 / 1.199 /
1.463 / 1.479 → 0.979 / 0.969 / 0.827 / 0.967 / 0.969.

#### Two hub corner families, named by wedge, and the one that binds today

OCC's hub corners fall in two bands with a measured 24° gap: **SQUARE-ON** at 266–270°, limited
by root thickness and arrival, and **NEAR-CUSP** at 294–332°, limited by the slot. Which binds
is a property of the design, not of the wheel: square-on at `t0` = 1.2, near-cusp on the
`t0` = 2.55 elites. `study_hub_cap.CUSP_WEDGE_DEG = 285.0` splits them.

**The promoted wheel's worst hub wedge is 314.0° — NEAR-CUSP.** The arc was built around the
arrival branch and the arrival branch is what got the descent here, but **the slot branch is
what binds on the design that ships.** Its own arrival law is unfitted and parked (0.31 at
3.4° rising to 0.70 at 50°, ~2.2× on the table); that is now the live successor.

#### What the descent did, and the two headline results

Re-descending under SVK at `medium` with the corrected cap produced the arc's first real
finding immediately: **the incumbent `350f4c7`, re-priced on a fillet radius that can actually
be built, is INFEASIBLE at utilisation 1.051.** Its shipped metrics read 0.783 because they
were computed at `coarse`, under **linear** kinematics, against the superseded 0.624 mm cap.
Every number in that file came from a model this arc showed to be wrong in a specific way.
**Shipping it was the risky option; that is why the promotion happened.**

The first descent then failed its own pre-registered gate 3-of-4, with `R_hub` pinned at a box
floor of **0.5** that was a bare literal in `GENE_SPACE` with no comment behind it — twice the
exporter's `MIN_CURVATURE_RADIUS_MM`. The cap cleared at arrival ≤ 39.893° and the run
converged at 40.542°: **0.650° short, with no legal move left in that gene.**

The floor moved to **0.4 mm — one extrusion width** at the 0.4 mm nozzle `MIN_WALL_MM` is three
perimeters of. Deliberately **not** 0.25: `MIN_CURVATURE_RADIUS_MM` is a *fault detector*
("well under any fillet we ask for, so a violation always means a construction fault") and
`MIN_BUILDABLE_R_MM` exists only to keep a blend width positive. Adopting either as a design
floor would have destroyed the detector by putting legal designs on it. `R_rim` unchanged.

The re-descent, warm-started from the same control so it differed in **exactly one thing**,
put `fillet_cap` at **exactly 0.000000 by step 3** and `R_hub` roamed a 0.135 mm interior band.

Second headline, and a correction to one issued mid-arc: the first descent reported `t0` coming
**off the `MIN_WALL_MM` floor for the first time in the project** (1.2714). With `R_hub` free it
is **pinned low again**. The observation was real; the stated cause was too general. A thicker
wall buys buildable fillet **only when the radius cannot move.**

#### THE THREE DEFECTS THIS ARC ADDED TO §15's FOUR

§15 named four defects in the objective. **This arc fixed none of them** — it fixed the
*buildability model*, which is a different thing — and found three more. All three are about
the penalty formulation rather than the physics, and all three cost real time before they were
understood.

5. **A QUADRATIC SOFT BARRIER CANNOT CONVERGE TO EXACTLY ZERO UNDER OPPOSITION.**
   `soft_barrier` is `scale * max(0, v)**2`, so **its gradient vanishes at its own knee.** A
   larger `R_hub` lowers `kt_hub` and relieves `stress`; a smaller one relieves `fillet_cap`.
   The two push against each other and settle where their quadratic gradients cancel — which is
   necessarily **just inside both**, because neither can generate force at `v = 0`. The
   re-descent's selected best sits at `fillet_cap` 0.000546 and `stress` 0.000751, converged to
   six figures over its last seven steps, with `stress_utilisation` 1.00031. **This is a
   property of the formulation, not of the design**, and it is the same shape of finding as the
   Step 3 gate clause that had to be retired as unsatisfiable. It is the mirror image of
   §15's defect 1: below the knee the barrier is invisible, and *at* the knee it is powerless.
6. **`wheel_stage3.py` SELECTS `--best-out` BY LOSS AND IGNORES FEASIBILITY ENTIRELY.**
   55 of 101 iterates in the re-descent had every one-sided barrier at exactly 0.0. The one the
   code picked was not among them. Selecting the lowest-loss **feasible** iterate is the
   standard rule for a penalised constrained descent and arguably the only correct one; the
   difference here costs **0.030% of a loss that is 99.8% mass.** Recorded, not patched — a
   change to the selection rule is a change to every future run and belongs in its own arc.
   **Until it is fixed, never promote `--best-out` without re-checking feasibility by hand.**
7. **FEASIBILITY MUST BE CHECKED WITH SLACK AT EVERY FIDELITY, NOT AT ONE.** Step 82 — the
   lowest-loss strictly feasible iterate, and the first thing promoted — clears the cap at
   `medium` by **53 nm** and violates it at `coarse` by **93 nm**. Its feasibility depends on
   which mesh you ask. Worse, sitting on the knee of `max(0, v)**2`, whose second derivative is
   discontinuous there, makes a central difference straddle the kink: it fails
   `study_objective.run_closed_form` at **2.705e-06** against a 1e-6 tolerance. **G4 was not
   reporting a code bug. It was reporting where the design sat.** Step 71 was promoted instead:
   **2951 nm** of slack, 0.000e+00 on that gate, a better deflection (−0.002% vs +0.051%) and
   **more** stress margin (0.99639 vs 0.99998), for +0.13% of loss and +0.048 g.

   The reasoning that produced step 82 was that the cap is already fitted 1.45–11.61% under OCC
   truth, so padding the numerical slack would be inventing a second margin. **That is correct
   about buildability and irrelevant to numerical robustness** — OCC built step 82 24/24 at
   +0.0%. Two distinct failure modes; the argument for one was applied to the other.

#### The gene box changed, and that reinterprets history in exactly one gene

`GENE_SPACE[12]['low']` went **0.5 → 0.4**. Raw genomes on disk are stored **by name** and are
unaffected. **The normalized `z` traces inside historical Stage-3 run JSONs are not:** gene 12
now decodes to a different physical radius. Any pre-2026-08-11 `steps[i]["z"][12]` read with
today's `GENE_SPACE` is wrong by `0.1 * (1 - z)` mm. Decode with the box that was in force.

It also has a live consequence: **`test_the_beam_to_wheel_ratio_is_not_a_constant` is a
casualty of the box change, not of the promotion.** `run_beam_blindness` draws a Latin
hypercube from the box, so its statistic is genome-independent — measured at **4.943223** with
the 0.5 floor (matching §14's documented 4.943) and **2.412764** with 0.4, *identically for both
genomes*. Gate 1's actual conclusion, `correction_factor_is_defensible == False`, holds in both
boxes; only the `> 3.0` margin, calibrated in the old box, moved. **Not re-tuned** — the test's
own docstring is explicit that re-deriving it is a judgement about Gate 1, not a test edit.

#### The gate

`make test` closes at **6 failed / 430 passed**, against §15's 2 / 433. 436 collected both
times. Nothing was deleted, skipped, xfailed, or re-thresholded. The one test edit in the arc
was a bug fix in a test the arc itself had added four steps earlier: the new cap test
constructed its arrival angle but **inherited `t0` from `best_solution.json`** — the exact fuse
its own docstring warns against, and `by_thickness` is linear in `t0`. Completing the
construction (`g[8] = 1.2`) made its guard the assertion it was meant to be.

Of the six red, one is the gene-box casualty above and **five are characterisation tests
genuinely invalidated by shipping a materially different wheel.** They pin findings about the
design space and are written to fail loudly when the premise moves:

| test | `350f4c7` | `e4219f3` | gate |
|---|---|---|---|
| `self_intersection_margin_detects_a_fold` | −7.064 | **+0.282** | < 0 |
| `correction_is_not_a_constant_over_the_design_space` | 3.383 | **1.542** | > 3.0 |
| `correction_enters_at_first_order_in_the_load` | 0.00205 | **0.00258** | < 0.001 |
| `rim_band_holds_a_large_minority_of_the_compliance` | 0.0321 | **0.0508** | < 0.03 |
| `a_thicker_rim_monotonically_stiffens_the_wheel` | 2.2496 | **1.637** | straddle 2.0 |

The last two were already red and both moved further out. Three causes worth knowing:

- The **fold detector is not broken.** It builds its positive case by inflating the *shipped*
  spine to a 40 mm band against an assumed ~11 mm curvature radius. The new spine is straighter
  — healthy margin 11.94 → 19.28, min curvature 12.26 mm — so 40 mm no longer folds it, by
  0.28 mm. It needs a thicker band or a fold constructed independently of the shipped genome.
- The **hub compliance share is the one real physical cost.** A 0.457 mm hub fillet is more
  compliant than a 0.579 mm one, so the hub's share of strain energy rose 0.0321 → 0.0508.
  **This is the price of a buildable radius and no iterate choice recovers it.**
- The **rim straddle says the new design is markedly more mesh-sensitive.** The two genomes
  agree at `medium` (1.99996 vs 1.99923 mm) and differ by **27%** at `smoke` (1.637 vs 2.2496
  at rim_outer 49.7), because every thickness gene sits on the 1.2 mm floor and a coarse mesh
  resolves a thin wall badly. **Do not trust a `smoke`-rung number on this genome.**

#### The successors, in priority order

1. **The slot branch's own arrival law.** It is what binds on the wheel that ships (wedge
   314.0°, NEAR-CUSP) and it is unfitted: 0.31 at 3.4° to 0.70 at 50°, ~2.2× unclaimed. The
   experiment is the mirror of `study_hub_cap.run_t0_sweep` — hold arrival fixed, walk the void.
2. **§15's successor 2, the stress-margin term**, now with a second argument behind it.
   `R_hub` still goes inert the moment it is feasible: no gradient from `fillet_cap` when
   satisfied, none from `stress` below its knee. Defects 1, 2 and 5 are one defect seen from
   three sides, and a term that prices margin instead of walling it fixes all three.
3. **The `--best-out` selection rule** (defect 6). Cheap, mechanical, and it removes a
   promotion hazard that this arc walked into twice.
4. **Re-derive the five characterisation gates** against a design whose walls are all on the
   floor. Real work and a judgement about what each gate should now say — not a threshold edit.

#### Artifacts

`stage3_buildcap2_medium.json` (the re-descent trace, 101 steps), `stage3_buildcap2_slack_medium.json`
(step 71, promoted) and `stage3_buildcap2_feasible_medium.json` (step 82, kept because defect 7
is only legible with both). `stage3_promote2_best.json` is the canonical `--steps 0` re-score
that became `best_solution.json`. The predecessor `350f4c7` is preserved byte-identical as
`stage3_minwall_best_1.2.json` and `stage3_minwall_best_1.2_medium.json`;
`best_solution_ga_beam.json` is untouched, so `tests/test_golden.py` is **not** re-baselined.
`export/wheel.step` and its manifest are rebuilt from `e4219f3`. Full step-by-step record in
`BUILD_PLAN.md`, steps 3 through 6c.

#### What this arc established, stated without the hedging

The objective has priced a buildable hub fillet since §5, and from 2026-08-06 to 2026-08-10 it
priced it with **a constant fitted an octave away from the floor every shipped design sits on**,
returning the same number for two designs OCC disagrees about. That is fixed, and the wheel that
ships is the first in this repo that the optimizer priced and the kernel built **at the same
radius**. The cost is honest and it is in the table above: a thinner, stiffer, more
mesh-sensitive wheel with 5.1% of its compliance in the hub, and five findings about the design
space that were measured on a design that no longer ships. The three new defects are all in the
penalty formulation, all found by gates rather than by reasoning — **two hypotheses about the
G4 failure were plausible, confidently held, and wrong before a measurement settled it** — and
the ordering lesson is procedural and cheap: **promote, export, then test**, and run the suite
after a gene-box change before anything else moves.

### 17. DEFECT 6 IS FIXED, AND THE SUCCESSOR RANKED #1 IN §16 IS WORTH NOTHING. Measured, not argued (2026-08-12).

Two things, one small and one that corrects §16's own conclusion.

#### `--best-out` no longer selects an infeasible genome

Defect 6 was that `wheel_stage3.py` reported the lowest-loss iterate of a run, and the loss
is a weighted sum in which the barriers are terms like any other. So the reported iterate is
whichever bought the most objective for the least constraint — and a barrier is not a thing
that can be bought. That is a category error, not a tuning failure, and no amount of descent
fixes it.

`wheel_objective.BARRIER_TERMS` / `OBJECTIVE_TERMS` now split the weight table by what a term
answers: *may this ship* or *how good is it*. The split is asserted complete against `TERMS`
at import, so a term added without a classification fails loudly rather than defaulting to
"can never make a design unshippable". `wheel_stage3.selection_key` ranks in three tiers —
feasible with slack, feasible on the knife edge, in violation — and the same key is used
within a run and across multi-start runs. Tier 2 is still ranked, so a run with nothing
feasible reports its least-violating iterate rather than nothing at all, and the banner says
which tier it is returning.

The band, `MIN_CAP_SLACK_MM = 1e-3`, is defect 7 made operational: **feasibility is
fidelity-dependent**, so a barrier reading exactly 0.0 certifies only that *this mesh* saw no
violation. Step 82 cleared the cap at `medium` by 53 nm and violated it at `coarse` by 93 nm.

Replayed against the real 101-step trace (`stage3_buildcap2_medium.json`), which is now a
regression test rather than a paragraph:

| iterate | loss | cap slack | old rule | new rule |
|---|---|---|---|---|
| step 93 | **30.8914** (the minimum) | **−1.045 µm** | **reported** | tier 2 |
| step 82 | 30.9008 | +0.052 µm | promoted off it | tier 1 |
| step 75 | 30.9406 | +10.936 µm | — | **tier 0, selected** |
| step 71 | 30.9421 | +3.096 µm | — | tier 0 (shipped) |

53 of 101 iterates are tier 0. The old rule reported a violating one out of that.

#### The new rule picks step 75, and step 75 should not ship — which is the finding

| | step 71 (shipped) | step 75 (rule's pick) |
|---|---|---|
| loss | 30.9421 | **30.9406** |
| mass | 37.5678 g | **37.5556 g** |
| deflection error | **−0.002%** | +0.110% |
| stress utilisation | **0.9964** | 0.9988 |
| R_hub | **0.4571 mm** | 0.4510 mm |

Step 75 is lower-loss and worse at every margin: it buys 12 mg with deflection error, stress
headroom, and 6 µm of hub fillet. `best_solution.json` stays at `e4219f3`; nothing was
re-promoted, re-exported, or re-scored.

This is the sharpest available argument for the stress-margin term. Fixing the selection rule
removed the barrier-versus-objective confusion and left the *objective's own* indifference to
margin fully exposed: among tier-0 iterates the rank is still loss, and loss prefers 12 mg to
every margin the design has. **Defects 1, 2 and 5 and this are one defect seen from four
sides.**

#### §16 ranked the slot arrival law #1. It cannot pay, and here is the measurement

§16's argument was that the slot branch "is what binds on the wheel that ships (wedge 314.0°,
NEAR-CUSP)". That conflated two different things, and they disagree on this wheel:

- the **wedge family of the worst OCC corner** — near-cusp, 314.0°, true; and
- the **analytic branch that sets the cap** — thickness, 0.4601 mm against the slot branch's
  1.6221 mm, **253% away from binding**.

The corner OCC finds hardest and the branch the optimizer feels are not the same object. Even
granting the whole re-fit — the measured near-cusp share at the shipped arrival is 0.60
against the modelled 0.30 — the slot branch would move to 3.2476 mm, **7× above the binding
branch**. It changes the cap by exactly zero, on this wheel and on any wheel near it.

Meanwhile the branch that *does* bind is already tight. Against the square-on family at
`t0` = 1.2, the floor the shipped wheel sits on:

| arrival | OCC / t0 | model / t0 | model is |
|---|---|---|---|
| 5.14° | 0.5105 | 0.5031 | 1.5% conservative |
| 20.10° | 0.4872 | 0.4758 | 2.4% conservative |
| 30.06° | 0.4524 | 0.4404 | 2.7% conservative |
| 40.03° | 0.4059 | 0.3925 | 3.4% conservative |
| 50.00° | 0.3439 | 0.3336 | 3.1% conservative |
| 59.96° | 0.2722 | 0.2653 | 2.6% conservative |

Conservative everywhere, never over-promising, and never by more than 3.4%. At the shipped
arrival of 41.748° it is 3.2% conservative. **There is no fillet radius left on the table on
the branch that binds** — which is what §16 set out to achieve and is worth stating as an
achieved result rather than leaving implied.

One more reason the §16 successor was not ready to run, and it is a datum already in the tree
being read for a second purpose. BUILD_PLAN.md step 4 records `350f4c7_t0_1.2`'s near-cusp
threshold as `≥1.44` at all seven arrival stations, censored at the bisection bracket, and
dismisses it correctly — "never binding in the measured range, which is all this arc needs
from it." That is true for computing a **cap**, where a censored non-binding branch costs
nothing. It is fatal for **fitting the branch itself**, which is what the successor proposes:
the share then reads 0.323 → 0.264 across arrival, and that decline is a constant numerator
over a growing arc, an artifact of the ceiling rather than a law. So the slot law rests on
`elite13_t0_2.55` alone, n = 1, and fitting on one uncensored design is the same mistake §16
exists to correct. The sweep would have to be re-run with a raised bracket first.

#### The successors, re-ranked by what they are now measured to be worth

1. **The stress-margin term** (§15's successor 2). Promoted to #1 on the evidence above:
   `R_hub` goes inert the moment it is feasible, and the step-75-versus-71 comparison shows
   the objective will spend every margin the design has for 12 mg. This is the one with a
   payoff.
2. **Re-derive the five characterisation gates** against a design whose walls are all on the
   floor. Unchanged in rank, still real work and still a judgement about what each gate should
   say, not a threshold edit.
3. **The slot branch's arrival law.** Demoted from #1 to #3 and re-scoped: it is a
   correctness fix to a branch that is 253% from binding, not a source of fillet radius. If it
   is done, it needs the arrival sweep re-run with a raised bisection ceiling first, because
   the existing data is censored on one of its two designs.

### 18. THE OBJECTIVE CAN NOW SEE STRESS MARGIN. Two dead genes are alive and the wall came off the floor. **NOTHING PROMOTED** (2026-08-12).

§15's defect 1, named on 2026-08-10 and left alone twice since because acting on it mid-arc
would have been re-fitting a gate to the run that breached it. §17 gave it a second argument
that was not available in §15 — with selection fixed, the objective's *own* indifference to
margin was the only thing left explaining step 75 — so it is now the ranked successor and this
is it.

#### The defect, re-measured before it was touched

`stress` is `soft_barrier(util - 1)`, identically zero **and identically flat** for every
`util <= 1`. Below the knee the optimizer cannot see stress at all: it sees mass, and it thins
the wall. The consequence is not bad weights, it is **dead genes** — the only routes from a
fillet radius into the loss are `stress` and the fillet barriers, all flat unless breached.

Measured at the shipped genome on 2026-08-12, before any change: **`dL/dR_hub` and `dL/dR_rim`
are both exactly `+0.000000e+00`.** Not small. Zero. A nominally 14-dimensional search was
running in 8, which is what §15 measured a different way (over 602 steps `R_rim` moved on 0 and
`R_hub` on 2, both times only because `fillet_cap` was live).

#### The term, and the weight as a stated policy

`stress_margin = w * util^2`, summed over the same two junctions as the barrier and for the
same reason — a `max` would zero the gradient of whichever junction is not currently worst. It
is in `OBJECTIVE_TERMS`, never `BARRIER_TERMS`: `stress` is untouched, the wall still decides
shippability, and this only stops the approach to it being free.

The weight is an exchange rate, so it is derived rather than picked. The mass term is 30.88 at
37.57 g, so 1% of mass is 0.309 of loss; 1% of hub utilisation at `util` = 0.855 costs
`w * (1.01^2 - 1) * 0.855^2` = 0.0147 `w`. Indifference is `w` = 21.0. Shipped at **20.0** —
rounded *down*, toward buying less margin, which is the conservative direction for a term whose
purpose is to move the optimum. Quadratic rather than linear is the second half of the policy:
the exchange rate steepens as margin disappears, so the last 10% of utilisation costs far more
than the first. At the shipped genome the term lands at 20.23, 13.6% of the loss against mass's
20.7%.

The hand-written product rule was FD-checked, and this was not a formality: `dkt_hub`'s `R_hub`
path was previously multiplied by `max(0, util - 1)` = 0 at every feasible design, so an error
in it could not have been observed.

| gene | analytic | central FD | rel err |
|---|---|---|---|
| 12 `R_hub` | −1.234936e+01 | −1.234936e+01 | 1.79e-07 |
| 13 `R_rim` | −7.237940e-01 | −7.237940e-01 | 1.10e-09 |
| 8 `t0` | +3.336761e+02 | +3.336762e+02 | 3.67e-07 |

#### What a 40-step SVK probe did with it

`stage3_margin_probe.json`, `coarse`, 40 steps, uniform 8-phase, SVK, from the shipped genome.
1 h 57 m, 7.4 GB peak, clean exit, converged (last three losses 51.7487 / 51.7478 / 51.7474).

| | step 0 | step 40 |
|---|---|---|
| loss | 57.0300 | 51.7474 |
| `R_hub` | 0.45711 | **0.62435** (cap 0.62700) |
| `R_rim` | 2.74947 | **3.00000 — box maximum** |
| `t0` | 1.20084 | **1.36183 — off the 1.2 floor** |
| utilisation | 0.95921 | 0.80982 |
| axle drop | 1.96822 mm (−1.6%) | 1.99780 mm (−0.11%) |
| mass | 37.568 g | 39.410 g (**+4.9%**) |

Four things, one of them unplanned.

**`R_hub` converged to 0.4% under its own cap.** It rose until the geometry stopped it and
settled just below, `fillet_cap` exactly 0.0 for the last dozen steps. The design now asks for
the largest fillet it can actually build — which is what §16's cap was for and what no descent
before this could do.

**The cap itself rose, 0.46006 to 0.62700.** Nothing pushes the cap; it is a function of `t0`
and the hub arrival. The optimizer reshaped the hub to make room for a fillet that was, for the
first time, worth having. §16's cap model and this term composed into a behaviour neither was
designed to produce.

**Two of the four wall genes came off the floor** — `t0` to 1.362 and `t3` to 1.326. §15's
defect 4 said the descent "ran out of wall to thin before it ran out of stress margin, and set
the floor lower and the same blind gradient would keep going." That is now false for `t0` and
`t3`: with margin priced, the design chooses to be thicker. `t1` and `t2` are still pinned.

**`R_rim` hit its box ceiling of 3.0 and stayed there for 26 steps.** That ceiling was set while
`R_rim` was a dead gene, so no descent has ever tested it. It is now the binding constraint on
the rim fillet, and it is an untested number rather than a physical limit.

#### Defect 5 did not bite, and an intermediate reading of this run said it would

Mid-descent `fillet_cap` oscillated — 0.0, 0.44, 0.0, 0.24 — and at step 11 `R_hub` sat 0.022 mm
**above** its cap. Read at step 11 that is a soft-barrier equilibrium, defect 5 becoming
operational the moment the barrier acquired live opposition, and it was recorded here as such
before the run finished. **The full run says otherwise.** Those excursions are transients from
the cap moving faster than `R_hub` could track it; once the cap stabilised, `R_hub` settled
under it and stayed. 26 of 41 iterates are feasible and the converged point is one of them.
Defect 5 remains real and remains unfixed — it just is not what limits this term.

#### One measure-zero bug, found because the term made it reachable

`smooth_min`'s derivative at exactly `a == b` was **1.0 against a true two-sided 0.5**. Two
primitives are non-differentiable at the tie and autodiff picks a subgradient for each: `jnp.abs`
returns 0 at 0, which drops the blend term entirely, and `jnp.minimum` hands the full 1.0 to its
first argument. The value was never wrong, which is why the existing value-only exactness test
could not see it. Fixed by writing the min symmetrically as `(a + b - |a - b|)/2`, fenced inside
the blend by a `where` because that form is not bit-exact and exactness outside the blend is the
property the function exists for. Never observed to bite — recorded and fixed because this term
drives `R_hub` at its cap deliberately, which turns the tie from an accident into an attractor.

#### The gate

`6 failed, 438 passed` — **the same six reds as §17, no new ones**, and +4 from this section's
own tests. That is worth stating plainly because it was not the expected outcome: changing a
term in the objective changes the loss at every design, and a suite with loss numbers pinned in
it would have gone red in bulk. It did not, which says the gates are pinned to physics and
provenance rather than to the objective's arithmetic. The six are §16's, unchanged and
unrelated: five characterisation gates invalidated by the shipped wheel's geometry and one
gene-box casualty.

#### NOTHING IS PROMOTED, and the reason is not caution

`stage3_margin_probe_best.json` (`3ca40c1`) is a 40-step **coarse** probe under a **changed
objective**. Its loss is not comparable to any number in §16 or §17 — the same discontinuity
§14 recorded for linear-versus-SVK, for the same reason. And it costs **+4.9% mass**, which is
a real design change that deserves a medium-fidelity descent and an export check rather than an
assertion. `best_solution.json` is untouched at `e4219f3`.

#### The successors, re-ranked again

1. **A production descent under the new objective** — `medium`, SVK, from the shipped genome,
   with an export check. This is what turns §18 from a demonstration into a candidate, and the
   +4.9% mass is the thing it has to justify.
2. **`R_rim`'s box ceiling.** Newly binding, never tested, and cheap: the question is whether
   3.0 mm is a real limit or a number typed in when the gene was dead.
3. **Re-derive the five characterisation gates.** Unchanged, and now with more to re-derive.
4. **Defect 5**, the quadratic barrier that cannot hold a boundary against live opposition. Not
   what limited this run, but the opposition is new and it will be back.
5. **The slot branch's arrival law.** Still 253% from binding; still correctness, not payoff.
