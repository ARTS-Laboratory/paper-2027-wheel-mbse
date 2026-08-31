# MBSE_PLAN.md — a requirements layer: what is this wheel for?

**Open arc #9. Created 2026-08-31, carried forward from nothing — this is the first arc in
the tree that is about the PROBLEM STATEMENT rather than about the solution. Nothing
started.**

**VERSION CONTROL IS PART OF THIS PROJECT'S WORKFLOW — CHANGED 2026-08-19.** The rule that
stood here read *"Ignore version control entirely. Do not commit, branch, stage, revert or
otherwise touch git."* **It is superseded.** The rules live in `PLAN.md`'s header block and
**only** there, so they cannot drift across ten files: one commit per finished unit of work
on `feature`, `make test` green first, never while a study driver is mid-write, a study
commit carries its regenerated `.json` and `.jpg`, a promotion is one atomic commit and never
one file — and **commits carry no assistant or tool attribution, no `Co-Authored-By:`
trailer, no session link, no generated-with footer.**

---

## Why this arc exists

**This tree has spent ninety-four numbered sections optimising a wheel and has never written
down what the wheel is for.** Every other arc asks whether the answer is right. This one
asks what the question was.

The entire mission — the vehicle, the landing, the environment, the service life — is
compressed into two bare literals in `src/wheel_fea.py`:

```
  FORCE_LBS            = 15.0    (:151)   ->  66.72 N, and that is the WHOLE vehicle
  TARGET_DEFLECTION_MM = 2.0     (:156)   ->  the WHOLE stroke requirement
```

Neither carries a derivation. Compare them to their neighbours: `MIN_WALL_MM` at `:236`
carries eighteen lines on why 1.2 and what §8/§11/§13 measured; `CY_BOUND_MM` at `:259`
carries a three-row sweep table proving the bound is not binding; `R_hub`'s floor at `:307`
carries twenty-five lines on one extrusion width. `FORCE_LBS = 15.0` carries the word
`# Loading`. `TARGET_DEFLECTION_MM = 2.0` carries `# compliant target (raised from 1.0 for
more travel)` — raised for whom, landing what, from what height, is not recorded anywhere in
the tree.

Four consequences, each of which this arc is meant to end:

1. **Nobody can say what vehicle this wheel is for.** 15 lb is 66.72 N. Divided by how many
   wheels, at what landing load factor, for what all-up mass? Three different vehicles
   produce that number and the tree does not say which one it is.

2. **There is no temperature anywhere in the repository.** `grep -riE
   'temperatur|thermal|celsius|glass.trans|\bTg\b|ambient|anneal|creep'` over `src/`
   returns **zero hits — not one**. Over `studies/`, `tests/`, the `Makefile` and the other
   ten `.md` files it returns **four lines, every one a false positive**: a cosine LR
   schedule that "anneals" (`study_stage3.py:2109-2110`, two lines), "import creep"
   (`test_pool.py:286`), and "creeping to ~0.808" (`PLAN.md:6579`).
   `YOUNGS_MODULUS_PLA_MPA = 2300.0` and `ULTIMATE_STRESS_MPA = 40.0` are
   single-point values at an **unstated** temperature. A PLA part is a thermoplastic part:
   its modulus is a strong function of ambient well below Tg, and this tree models a
   summer afternoon and a winter morning as the same wheel.

3. **`ALLOWABLE_STRESS_MPA`, `TARGET_DEFLECTION_MM` and the weight table cannot be varied
   at all.** `force`, `E` and `nu` already thread as keywords through the whole solve path
   and even survive the process pool (`wheel_objective.py:1115` ships `problem_kw`). The
   other three are read as module globals inside the loss — `wheel_objective.py:1153`
   (target), `:1234-1235` and `:1272-1273` (allowable). `tests/test_objective.py:1257` has
   to `monkeypatch.setattr(WO, "ALLOWABLE_STRESS_MPA", 2.0)` to move it, which is the tell.
   Only `MIN_WALL_MM` and `CY_BOUND_MM` have setters (`wheel_fea.py:840`, `:852`) and only
   they are recorded into a genome record's `search` block.

4. **`DEFAULT_WEIGHTS` is an unacknowledged requirements allocation.** Fourteen numbers at
   `wheel_objective.py:352-393` decide what the optimiser is trying to do, and they are
   individually justified but never justified *against each other* as a portfolio. This
   arc's calibration says what portfolio they are — and the answer is **not** the one the
   loss breakdown suggests. See below.

**What this arc is NOT.** It adds no gene, no mesh block, no solver capability, and no
geometry. The wheel it produces at the baseline requirement set is the wheel that ships
today, bit for bit. It is plumbing plus a calibration plus a verifier.

---

## The scope, decided before the arc opened, and not up for re-litigation inside it

**THE WHEEL IS THE GEAR.** There is no leg, strut, shock absorber, axle part or aircraft
anywhere in `src/`. `wheel_step_export.py:9-11` states the whole assembly: *"Full wheel =
solid hub disk + 12 spiral spokes + Ø100 rim band, unioned into one solid, with true tangent
fillets at the spoke↔hub and spoke↔rim junctions"*. The curved spokes ARE
the springs — that is what "compliant wheel" means and it is why `TARGET_DEFLECTION_MM`
exists at all. This arc drives that part's load case, allowables, stroke target and mass
budget. It does not grow a landing-gear assembly.

**Ø100 IS FROZEN FOR THIS ARC.** Ground clearance and prop clearance are real requirements
and they want `RIM_RADIUS_MM`. They cannot have it here. `wheel_fea.py:113-137` states the
cost: changing it *"REINTERPRETS every gene on disk"*, and `tests/test_golden.py` fails
loudly if the shipped artifact was optimised in a different frame. Every requirement axis
in this arc leaves the genome frame intact, which is what lets `best_solution.json` and the
golden pin stay meaningful from Step 0 to Step 8. **Diameter is named here as a future axis
with its price stated, and that is all.**

**`NUMBER_OF_SPOKES = 12` IS NOT A PARAMETER.** It is baked into `SECTOR_DEG` (`wheel_wheel.py:174`),
the mesh's twelve-fold periodicity, the `/3` in `FORCE_PER_SPOKE_NEWTONS`
(`wheel_fea.py:154`), and the exporter. Do not let a requirement reach it.

**SCORING LANDS BEFORE RE-OPTIMISATION.** A requirements layer that can only answer by
spending a descent cannot be debugged: a wrong requirement and a bad descent look the same
from the outside. Steps 0-5 produce a working MBSE loop that solves nothing new and runs in
seconds. Step 6 spends compute.

---

## What is already known — do not re-derive, re-read

**The shall/should distinction is ALREADY IN THE CODE, and it is asserted.**
`wheel_objective.py:394-401`:

```
  BARRIER_TERMS   = ("stress", "buckling", "x_order", "hub_overlap", "fold",
                     "arrival", "fillet", "fillet_cap", "min_sj")
  OBJECTIVE_TERMS = ("deflection", "mass", "stress_margin", "smoothness", "phase_ripple")

  assert set(BARRIER_TERMS).isdisjoint(OBJECTIVE_TERMS), "a term is one or the other"
  assert set(BARRIER_TERMS) | set(OBJECTIVE_TERMS) == set(TERMS)
```

with the comment: *"a barrier is a term whose only admissible value is zero, so it answers
'may this design ship' and never trades against `mass` or `deflection`, which answer 'how
good is it'."* That IS the MBSE **shall** / **should** spine, already written, already
enforced, already load-bearing (it exists because defect 6 promoted an infeasible design on
2026-08-11 by selecting on loss alone). **Reuse it. Do not invent a second one.**

**The exchange-rate idiom is already the house language for a weight.**
`wheel_objective.py:356-363`, on `stress_margin: 325.0`:

> *"This is the one weight in the table that sets an exchange rate rather than a scale: 1%
> of utilisation against 1% of mass at the shipped genome. 328.49 makes 1% of utilisation
> cost 1% of mass at `MARGIN_KNEE_UTIL`'s reference point (util 0.855, §18's own), rounded
> DOWN for §18's reason: the rounding buys LESS margin, which is the conservative direction
> for a term whose purpose is to move the optimum."*

The calibration below is that paragraph, generalised to all five objective terms.

**Three of the five requirement quantities already thread; two do not.**

```
  quantity              status today                                          work
  --------------------  ----------------------------------------------------  --------
  force                 keyword on objective(:1309), t3_terms(:1046);         route it
                        rides to pool workers at :1111-1115
  E, nu                 ride **problem_kw -> service_qoi_value_and_grad       nothing
                        (wheel_adjoint.py:646) -> wheel_contact_problem;
                        survive the pool (wheel_objective.py:1115)
  min_wall              set_min_wall(wheel_fea.py:852) + --min-wall            route it
  target_deflection     MODULE GLOBAL, read at wheel_objective.py:1153-1155   PLUMB IT
  allowable_stress      MODULE GLOBAL, read at :1234-1235 and :1272-1273      PLUMB IT
```

So the plumbing surface is **two constants**, not the 195 grep hits the constant names
produce.

**The shipped design is already ON the knee.** `best_solution.json` metrics:
`stress_utilisation_hub = 0.8201` against `MARGIN_KNEE_UTIL = 0.80`, and
`axle_drop_mean_mm = 1.99742` against a 2.0 target (−0.129%). It has essentially no stress
headroom. Any requirement profile that raises the load or the temperature by a modest
amount should bind on stress, which is what makes Step 5's verifier falsifiable rather than
decorative.

---

## The two input surfaces, and the split IS the design

A requirement you cannot choose is not a preference, and a preference is not a requirement.
Conflating them is how a trade study becomes a wish.

```
  MISSION — absolute facts, entered as numbers        PRIORITIES — 100 points, zero-sum
  ------------------------------------------------    ---------------------------------
    auw_kg, n_wheels, sink_rate  ->  force_n            light         ->  mass
    field_class, sink_rate       ->  target_deflection  soft landing  ->  deflection
    ambient_c                    ->  e_mpa, allowable   durability    ->  stress_margin
    landings                     ->  safety_factor      rolling       ->  phase_ripple
    nozzle_mm, perimeters        ->  min_wall_mm        print finish  ->  smoothness
```

**The five priority axes are EXACTLY `OBJECTIVE_TERMS`.** Not approximately, not mapped —
the same five names in the same order. Points move `should`s. **Points never reach
`BARRIER_TERMS`**: you cannot buy your way out of a mesh that does not integrate, a spoke
that folds through itself, or a fillet that does not fit in its sector. Those are `shall`s
and their only admissible value is zero.

### The mission axes, each with what it reaches AND what it does not

**1. All-up weight (`auw_kg`) and wheel count (`n_wheels`).** The vehicle. Reaches
`force_n`. This is the axis that is currently `FORCE_LBS = 15.0` and nothing else.

**2. Touchdown sink rate (`sink_rate_ms`).** Reaches `force_n` through the landing load
factor. This is the axis that makes a *compliant* wheel a compliant wheel: the wheel absorbs
½·m·v² over its stroke, so sink rate and stroke are the same trade seen from two ends. A
2.0 mm stroke is a very short stroke, and the load factor it implies is the number this
arc will make visible for the first time.

**3. Runway length (`runway_m`) and field class (`field_class`).** Reaches `sink_rate` (a
short field forces a steeper approach and a firmer arrival) and `target_deflection` (a rough
field needs travel). **AND NOTHING ELSE — this must be stated in the record and not
softened.** The FEA applies a **purely radial** load on **flat, rigid, frictionless**
ground (`wheel_fem.wheel_contact_problem:1717`, `RigidGroundContact:652`). There is no
braking load, no side load, no obstacle bump, no rolling friction and no tyre. A short-field
braking case is genuinely out of model, and a plan that lets "runway length" imply otherwise
is lying about what was verified.

**4. Ambient temperature (`ambient_c`).** Reaches `e_mpa` AND `allowable_stress_mpa`, and it
is the only axis that pulls the design in two directions at once: hotter PLA is softer,
which *helps* a stroke requirement, and weaker, which *hurts* a stress requirement. That is
what makes it the most interesting axis in the set and the one most worth having. **New
physics — see Step 2, and read its scope note before quoting any hot-day result.**

**5. Service life (`landings`).** Reaches `safety_factor` and therefore
`allowable_stress_mpa`. Pure policy today: `SAFETY_FACTOR = 1.6` at `wheel_fea.py:146` with
no stated life behind it.

**6. Process (`nozzle_mm`, `perimeters`).** Reaches `min_wall_mm`, which already has a
setter and a flag. Worth including because `MIN_WALL_MM` **sets 4 of the 14 genes at the
optimum** — the shipped genome has `t1 = t2 = 1.2` exactly on the floor — so it is one of
the highest-leverage numbers in the tree and it is currently derived in prose
(`# 3 perimeters @ 0.4 mm nozzle`) and written as a literal. Nothing in code knows 1.2 came
from 0.4.

### The priority axes

```
  point axis        term            what buying it does
  ---------------   --------------  --------------------------------------------------
  light             mass            grams off, at the cost of everything else
  soft landing      deflection      hit the stroke target rather than drift off it
  durability        stress_margin   keep utilisation under MARGIN_KNEE_UTIL = 0.80
  rolling           phase_ripple    make axle drop uniform as the wheel turns
  print finish      smoothness      a clean single-curvature spiral, no lumps
```

`rolling` is a real UAV concern and the term for it exists and is **switched off**:
`DEFAULT_WEIGHTS["phase_ripple"] = 0.0`, with the comment *"off by default; gate 10 reports
what turning it on costs"*. The shipped wheel has
`phase_ripple_std_over_mean = 0.1044` — a 10% variation in axle drop through one 30° sector
— that nothing currently prices. This arc pays that debt (Step 4).

---

## The points -> weights map, and why LOSS SHARE will not do it

**The naive map is proportional-to-loss, and `best_solution.json` refutes it in one table.**
At the shipped genome, the five objective terms and their share of the objective loss:

```
  term             loss        share
  deflection      0.004146    0.0127%
  mass           32.440977   99.0727%
  stress_margin   0.131681    0.4021%
  smoothness      0.167828    0.5125%
  phase_ripple    0.000000    0.0000%
  ----------------------------------
  total          32.744632   (= the record's own `loss`, exactly)
```

Read as an allocation this says the tree cares 99.07% about mass and 0.013% about stroke.
**That is false, and the reason it is false is the whole point of this section.**
`deflection` is 0.0127% of the loss because the design sits at 1.99742 mm against a 2.0 mm
target — a −0.129% miss. The term is small because the requirement is *met*, not because it
is unimportant. **Loss share measures satisfaction, not priority.** An allocation built on
it would strip the weight from every requirement the optimiser had successfully satisfied
and then be surprised when the next descent stopped satisfying them.

### What replaces it: the cost of a 1% miss

For each objective term `T`, define a **reference deviation** `d_T` — one named physical
unit of missing that requirement — and let `c_T` be the loss that miss costs at the
reference genome under `DEFAULT_WEIGHTS`:

```
  term            reference deviation d_T                          form        c_T
  --------------  -----------------------------------------------  ---------   --------
  mass            1% of MASS_REFERENCE_G (0.365 g)                  linear      0.300000
  deflection      1% relative error on TARGET_DEFLECTION_MM         quadratic   0.250000
  stress_margin   1% of utilisation above MARGIN_KNEE_UTIL (0.80)   quadratic   0.032500
  smoothness      1% of the curvature-rate integral                 STEP 4      0.001678
  phase_ripple    1% of std/mean axle drop                          STEP 4      0.000000
```

`c_T = L(d_T)` — the cost of a full 1% miss measured from that term's own satisfied point —
and **not** `dL/dx` at the current iterate. The marginal rate is the wrong instrument here
for exactly the reason the loss share was: at a satisfied quadratic term the marginal rate
is near zero, so a calibration built on it would be a calibration built on where the last
descent happened to stop. `L(d_T)` is a property of the weight, not of the iterate.

**The calibration allocation, and the finding it produces:**

```
    p_T^cal = 100 * c_T / sum(c)

  term             c_T        p_cal
  --------------  --------   ------
  mass            0.300000    51.35
  deflection      0.250000    42.80
  stress_margin   0.032500     5.56
  smoothness      0.001678     0.29
  phase_ripple    0.000000     0.00
  --------------------------------
  sum c = 0.584178             100
```

**THIS IS THE ARC'S FIRST REAL FINDING AND IT IS AVAILABLE BEFORE ANY CODE IS WRITTEN.**
The shipped weight table is a **51/43/6/0.3/0** portfolio — roughly half on mass, roughly
half on stroke, a twentieth on durability, and nothing at all on rolling or print finish.
That is a defensible allocation and it is nothing like the 99%-mass reading the loss
breakdown invites. It has never been stated. Step 4 must reproduce these numbers from the
code rather than from this file, and the two must agree.

### The map back, and why the budget is 100 and zero-sum

```
    w_T(p) = DEFAULT_WEIGHTS[T] * p_T / p_T^cal
```

Because `w_T` is proportional to `p_T` and `p_T^cal` is proportional to `c_T`, the total
exchange-rate pressure the objective exerts is

```
    sum_T c_T(p)  =  sum_T c_T^cal * p_T / p_T^cal  =  (sum c^cal / 100) * sum_T p_T
```

which is **invariant exactly when `sum_T p_T = 100`.**

**The budget is not a UI convention. It is the conservation law that keeps the
objective-against-barrier balance fixed while priorities move.** Weights are not
scale-free here: `BARRIER_TERMS` are absolute, so multiplying every objective weight by two
does not leave the optimum alone — it halves the effective strength of every `shall`. The
constraint `sum p = 100` is precisely what forbids a user from buying more of everything and
quietly weakening every feasibility barrier in the process. **Any change to this map must
preserve that property, and Step 4 has a test for it.**

### The one term the map cannot reach, and it is not a detail

`DEFAULT_WEIGHTS["phase_ripple"] = 0.0`, so `c = 0`, so `p^cal = 0`, so
`w = w_default * p / p^cal` is `0/0`. The map is undefined on the axis a user is most likely
to want to move first, because it is the one the tree has never bought any of.

It needs its own anchor, **measured and not assumed**: calibrate `w_ripple` so that one
reference ripple deviation costs the same as one reference mass deviation at the reference
genome, then carry the shipped `(std/mean)² = 0.1044²` beside it so the number can be
argued with. The same treatment applies to any future term whose default weight is zero.

---

## The mission -> constants derivations

Each is stated here so Step 1 implements an argument rather than an opinion.

```
  W        = auw_kg * 9.80665                     N        all-up weight
  F_static = W / n_wheels                          N        static, per wheel
  n_land   = 1 + v_z^2 / (2 * g * s_eff)                    energy balance over the stroke
  force_n  = F_static * n_land * k_asym            N        k_asym: one wheel lands first

  e_mpa     = E_20c          * e_retention(ambient_c)
  allowable = sigma_ult_20c  * sigma_retention(ambient_c) * fff_knockdown / SF(landings)
  SF(N)     = SF_base * k_fatigue(N)
  min_wall  = nozzle_mm * perimeters
```

**`n_land` depends on `s_eff`, which is the stroke requirement, which is downstream.** That
is a real circularity and it is resolved by ORDERING, stated in the module docstring and not
left to the reader: field class and a floor set the stroke first, then the load factor
follows from it. Do not solve the fixed point. A fixed-point stroke would make the load a
function of the design and this repo loads to a FORCE, not to an indentation — see
`service_qoi_value_and_grad`'s docstring (`wheel_adjoint.py:649-662`), whose entire subject
is that the distinction is not a correction but the term.

---

## The cost, stated up front

Steps 0-5 **solve nothing new**. `study_mbse_baseline` and `study_mbse_calibration` read
committed artifacts and do arithmetic, in the idiom `study_fillet_kt` established
(*"reads six committed artifacts and solves nothing, which is deliberate"*).
`study_mbse_score` re-scores the shipped genome at `coarse` under a handful of profiles:
8 phases × ~0.7 s, so seconds to low minutes per profile.

Step 6 is the only expensive one, and it is deliberately warm-started: ~58.6 s per
value+grad at `coarse` with 4 workers, so a 30-step descent from `best_solution.json` is
tens of minutes. **Do not run a `medium` production descent inside this arc** — that is
~226 s/step, ~6.3 h for 100 steps, and this arc has nothing to promote.

**Budget the plumbing before the calibration.** If `objective(genes, req=baseline())` cannot
be made bit-identical to `objective(genes)`, that is the finding and it should surface in
hours, not after a calibration has been built on top of a moved default.

---

## THE PLAN

### Step 0 — What mission does the shipped wheel imply?

Run the derivations **backwards**. Read `FORCE_LBS = 15.0`, `TARGET_DEFLECTION_MM = 2.0`,
`SAFETY_FACTOR = 1.6` and `MIN_WALL_MM = 1.2` and report the all-up weight, sink rate,
ambient and service life they correspond to. **Nobody currently knows, and the answer is a
finding whether it is flattering or not.**

Driver: `studies/study_mbse_baseline.py`. Solves nothing.

- **CHECK:** `Requirements.from_mission(baseline_mission)` reproduces all four constants to
  1e-12.
- **CHECK:** the implied load factor is reported explicitly. A 2.0 mm stroke is short; if
  the implied `n_land` is absurd for any plausible UAV, **that is the headline of this
  step** and it must be written up as such rather than tuned away.
- If no plausible mission produces 66.72 N, file that and pick the baseline by fiat, stated
  as a fiat.

### Step 1 — `src/wheel_requirements.py`, with no consumers

`MaterialCard`, `Mission`, `Priorities`, `Requirements`, the derivations above, `req_hash`.
numpy + stdlib only, no jax, mirroring `wheel_genome.py`'s hygiene contract — the CAD
interpreter must be able to import it. Nothing imports it yet.

- **CHECK:** `Requirements.baseline()` gives `force_n == 66.72`,
  `allowable_stress_mpa == 25.0`, `target_deflection_mm == 2.0`, `min_wall_mm == 1.2`,
  `e_mpa == 2300.0`, `nu == 0.35`, and `weights == DEFAULT_WEIGHTS` **key for key,
  exactly**.
- **CHECK:** `Priorities` rejects sums other than 100, negative points, and unknown axis
  names. A budget that does not bind is not a budget.

### Step 2 — the temperature model, with its anchors and its scope

The only genuinely new physics in the arc. `e_retention(T)` and `sigma_retention(T)` as
piecewise-linear curves over **cited anchor points**, with a hard refusal above
`t_max_service_c`. This tree requires a measurement or a citation behind every constant, so
the anchors go in the docstring with their sources, in the form `MIN_WALL_MM` and
`CY_BOUND_MM` already use.

**THE SCOPE NOTE IS PART OF THE DELIVERABLE AND MUST NOT BE SOFTENED.** This is a
**quasi-static** knockdown: no creep, no fatigue, no thermal expansion, no self-heating, no
rate dependence. PLA creeps badly above ~45 °C, and a static allowable at elevated
temperature is **optimistic**. Every hot profile the verifier reports must carry that
sentence.

- **CHECK:** `e_retention(20.0) == 1.0` and `sigma_retention(20.0) == 1.0` **exactly**, so
  the baseline is untouched by construction rather than by luck.
- **CHECK:** monotone non-increasing on the anchor grid.
- **CHECK:** an anchor held out of the fit is predicted within a stated band — or the curve
  is reported as an interpolation between anchors and NOT as a model. Say which.
  A fitted rule believed without a hold-out is how §73 lost half an arc.

### Step 3 — thread it, and prove nothing moved

Make `allowable_stress_mpa` and `target_deflection_mm` keywords on `t3_terms` and
`objective`, defaulting to today's values. Route `force`, `E`, `nu`, `min_wall`. Two
constants; everything else already threads.

- **CHECK — THE LOAD-BEARING TEST OF THE WHOLE ARC.**
  `objective(genes, req=Requirements.baseline())` equals `objective(genes)` in the scalar
  value, **all 14 gradient components**, and **every one of the 14 breakdown terms**, bit
  for bit. A default that moved is a silent re-interpretation of every committed artifact
  and of the five study files that re-alias `SERVICE_FORCE_N` (`study_gnl.py:106`,
  `study_contact.py:94`, `study_gradient.py:120`, `study_fillet_cost.py:115`,
  `study_svk_rescore.py:67`).
- **CHECK — the cache audit, BY TEST AND NOT BY READING.** `_T1_CACHE` keys on
  `(cfg.name, span_mm, flanks, _t1_weights_key(weights))` (`wheel_objective.py:908`);
  `_KT_CACHE` keys without weights (`:533`); `wheel_wheel._COORD_FN_CACHE` (`:2760`) keys on the
  static mesh recipe. Two requirement sets differing **only** in `allowable_stress_mpa`
  must give different `stress`/`stress_margin` **in the same interpreter**. A stale jit
  trace returning the old answer is exactly the failure this check exists for.
- **CHECK:** `make m8bii1` — pooled == serial, bit-identical — still passes with a
  **non-baseline** requirement set, proving the requirements reach the workers and are not
  silently defaulted there.

### Step 4 — the calibration, measured rather than asserted

`studies/study_mbse_calibration.py`: measure `c_T` at the shipped genome from the code,
derive `p^cal`, implement `weights_from_priorities`, and anchor `phase_ripple` from scratch.

- **CHECK — identity:** `weights_from_priorities(p_cal) == DEFAULT_WEIGHTS` to floating
  point. The map must be an identity at its own calibration point or it is not a
  re-parameterisation, it is a change.
- **CHECK — the table above is reproduced from `src/`**, not copied from this file. If
  `p_cal` does not come back as 51.35 / 42.80 / 5.56 / 0.29 / 0.00, this file is wrong and
  the driver is right.
- **CHECK — conservation:** total exchange-rate pressure is invariant under any
  reallocation summing to 100, to floating point.
- **CHECK — ripple:** its anchor is a measured number filed beside the shipped
  `std/mean = 0.1044` it prices, not a guess.

### Step 5 — scoring and verification, no optimiser

`verify(record, req)` -> a compliance table: requirement ID, statement, verification
method, measured quantity, value, limit, margin, verdict. Every measured quantity **already
exists** in `best_solution.json`'s `metrics` block — `axle_drop_mean_mm`,
`stress_utilisation_hub`/`_rim`, `mesh_mass_g`, `min_scaled_jacobian`, `buckling_ratio` —
plus the `loss_terms` barriers, which verify as "exactly 0.0".

`studies/study_mbse_score.py` scores the shipped genome against named profiles: baseline,
hot day, heavy payload, rough field, long service life.

- **CHECK — THE CRITERION MUST BE ABLE TO COME BACK BOTH WAYS.** The baseline profile shows
  every barrier `0.0` and reproduces `best_solution.json`'s `loss_terms` exactly, **AND at
  least one profile returns NON-COMPLIANT naming a binding requirement.** A verifier that
  cannot fail is not a verifier, it is a formatting exercise. The shipped design sits at
  `stress_utilisation_hub = 0.8201` against a knee of 0.80, so a modest load or temperature
  increase should bind on stress. **Probe this on `smoke` before spending a real run.**
- **CHECK:** the compliance table's `shall` rows are exactly `BARRIER_TERMS` and its
  `should` rows are exactly `OBJECTIVE_TERMS`, read from `wheel_objective` rather than
  retyped, so the two cannot drift.

### Step 6 — re-optimisation under a requirement set

`--requirements <path>` on `wheel_stage3`, warm-started from `best_solution.json`. The
output record grows a **top-level** `requirements` block carrying the derived set and
`req_hash`; `search_block` (`wheel_stage3.py:883`) records `req_hash` beside `min_wall_mm`
and `cy_bound_mm`.

- **CHECK:** `--requirements baseline.json` at `coarse`, 5 steps, reproduces a plain run's
  iterates bit for bit.
- **CHECK:** a record whose `req_hash` does not match the requirements it is being scored
  against is **refused**, not silently compared — the discipline `warn_if_stale`
  (`wheel_step_export.py:196`) already applies to STEP files, applied to requirements.

### Step 7 — the front end

`make mbse` — takes a mission and a 100-point allocation, emits the derived requirements,
the compliance table, and with `--descend` the re-optimised genome.

### Step 8 — write up

A `PLAN.md` section (`## §95`, append-only, never renumbered) carrying: the implied baseline
mission from Step 0, the `51/43/6/0.3/0` calibration table, the thermal anchors with their
scope note, the bit-identity result, and the profile that came back non-compliant with its
binding requirement. Then `WHAT MOVED`, then `#### The successors, ranked`.

Add the arc to `PLAN.md`'s *Open arcs* table as row 9.

---

## What must NOT happen

- **Do not add a key inside `genes`.** `wheel_genome.save_record` (`:148-164`) refuses it,
  and the refusal states why: it *"changes `genome_hash` for every genome and breaks all
  staleness checks"*. The requirements block is **top-level**, beside `search`, `metrics`
  and `loss_terms`.

- **Do not move a single default.** If `Requirements.baseline()` does not reproduce
  `wheel_fea`'s constants exactly, **the derivation is wrong, not the constant.** Changing a
  default here turns every silent omission in every driver into a wrong answer.

- **Do not let points reach `BARRIER_TERMS`.** A barrier is a `shall`. Priorities move
  `OBJECTIVE_TERMS` only, and the disjoint/exhaustive assert at `wheel_objective.py:398-401`
  must stay green.

- **Do not put diameter, spoke count or face width in the allocation.** Ø100 is frozen for
  this arc for the reason `wheel_fea.py:113-137` states; `NUMBER_OF_SPOKES = 12` is not a
  parameter; `SPOKE_WIDTH_MM = 22.4` is the extrude depth the whole 2D plane-stress model
  rests on.

- **Do not touch `studies/study_deflection_gci.py:72`.** It defines its own
  `SAFETY_FACTOR = 1.25` and that is **Roache's GCI safety factor**, entirely unrelated to
  `wheel_fea.SAFETY_FACTOR = 1.6`. Two different `SAFETY_FACTOR`s live in this repo and a
  global rename would silently corrupt a convergence gate.

- **Do not claim runway length reaches the physics further than it does.** Radial load,
  flat rigid frictionless ground. No braking, no side load, no obstacle bump, no rolling
  friction. Runway length reaches the model **only** through sink rate and stroke.

- **Do not present the thermal knockdown as a life model.** Quasi-static only. Name it on
  every hot profile, every time.

- **Do not calibrate on one genome and call it a rule.** The calibration is anchored at the
  shipped genome by necessity; say so, and treat a second genome as the hold-out rather than
  as a confirmation. That is the exact error §24 corrected and §73 paid for again.

- **Do not regenerate a committed study artifact as a side effect**, and never commit while
  a driver is mid-write — the drivers rewrite `studies/*.json` and `studies/*.jpg` in place
  over minutes to hours and those files are tracked on purpose.
