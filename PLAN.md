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
2. **One Inventor import** of `export/stage3_minwall_best_1.2.step`, against a 0.0087 mm
   edge and a 0.195 mm² face. **This is now the only thing standing between the candidate
   and promotion.**
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

Promotion now waits on **one thing**: the Inventor import of
`export/stage3_minwall_best_1.2.step`, against that 0.0087 mm edge and 0.195 mm² face. See
§11's "What promotion still needs". `best_solution.json` is untouched.
