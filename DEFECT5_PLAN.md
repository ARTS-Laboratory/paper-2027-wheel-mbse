# DEFECT5_PLAN.md — the quadratic barrier cannot hold a boundary against live opposition

> **Ignore version control entirely. Do not commit, branch, stage, revert or otherwise
> touch git — it is not part of this project's workflow and nothing here depends on it.**

> **THIS FILE IS WRITTEN TO BE READ ONE STEP AT A TIME.** Every step carries
> **Why / Read first / Do / Gate / Record**. Do the step, write its numbers into its own
> Record block, tick the status table, stop.

**The parent item is PLAN.md §20's ranked successor #1.** §19 is where the defect was
measured; §20 is where it became #1. This file is the milestone; PLAN.md gets a §21.

---

## Status

| step | what | status |
|---|---|---|
| 0 | baseline the gate | **INHERITED from CONTACT_PLAN Step 3** — 7 failed / 438 passed (2026-08-13) |
| 1 | measure the equilibrium law, and predict the overshoot before measuring it | **DONE — BARRIER DERIVATIVE EXACT TO 6 s.f.; the run had NOT reached its own fixed point** (2026-08-13). Fixed point is **5.163 µm**, not §19's 4.1. Defect 5 and defect 8 are coupled |
| 2 | THE DESIGN DECISION — pre-registered before anything is built | **DONE — THE DECISION IS *NONE OF THEM*. The converged "unbuildable" iterate EXPORTS 24/24 AT +0.0% Kt ERROR** (2026-08-13). Defect 5's realised cost is 0.61% of loss, not a lost design. Arc stops |
| 3 | implement, opt-in, with the C¹/FD gate | **NOT RUN** — Step 2 says the fix is not justified |
| 4 | cheap probe: `coarse`, short, does the overshoot go | **NOT RUN** |
| 5 | production `medium` SVK descent, §19's own trajectory as control | **NOT RUN** — the 6 h 20 m this arc existed to spend |
| 6 | export, check, and the promotion call | **NOT RUN.** `best_solution.json` untouched |
| 7 | write the record into PLAN.md as §21 | **DONE** (2026-08-13) |

Genomes, by hash: shipped `e126cc3` = `best_solution.json`; §19's run record is
`stage3_margin_medium.json` (101 steps, the control for Step 5).

---

## The defect, in one page

`soft_barrier(v) = scale * max(0, v)**2` (`src/wheel_objective.py:382`), so

```
    d(barrier)/dv = 2 * scale * max(0, v)        ->  ZERO at the knee
```

Against a term that pulls steadily outward with force `P`, the barrier therefore does not
hold `v = 0`. It settles where the two balance:

```
    v*  =  P / (2 * scale)                                   THE EQUILIBRIUM LAW
```

**The overshoot is not a failure of the descent. It is the stationary point the objective
defines.** §18 created the opposition when it gave the loss a `stress_margin` term pushing
`R_hub` up; §19 then measured the consequence — **74 of 101 iterates in violation, steps 56
through 100 every one of them, converging 4.1 µm over the hub fillet cap** — and only §17's
`selection_key` (defect 6) rescued a shippable genome, step 55, from a run whose whole
converged tail was unbuildable. **45 steps of a 6 h 20 m run.**

### The specific geometry, which makes the law testable rather than rhetorical

`fillet_cap` prices `v = genes[12] - hub_fillet_cap_mm(...)` (`:824`) — the **raw gene**
against the cap, so `dv/dR_hub = 1` exactly and the law above has no chain rule in it. The
opposing pull arrives through `Kt_hub`, which is priced on
`hub_fillet_r_effective = smooth_min(R_hub, cap, k)` (`:739`) with
`k = CAP_BLEND_FRAC * max(cap, 0.25)` and `CAP_BLEND_FRAC = 0.15`.

That matters, because `smooth_min` is **exact outside the blend**: more than `k` above the
cap, `R_eff` is the cap to the last bit and `dR_eff/dR_hub` is exactly 0.0 — the pull would
vanish and the barrier would drive `R_hub` back. So a stable equilibrium can only exist
*inside* the blend. Measured at `e126cc3`:

```
    cap        0.735233 mm          blend width k   110.3 um
    overshoot    4.1 um    =  3.72% of the blend width
```

It is deep inside. The mechanism is consistent, the equilibrium is real, and **the law
predicts a number that has not been measured: `P = 2 * 500 * 4.1e-3 = 4.10` loss units per
mm of `R_hub`.** Step 1 measures `dL_stress_margin/dR_hub` directly and checks it.

### What the fix is actually for, stated before choosing one

**Not "make the barrier exact."** Any quadratic barrier has `v* = P/(2w) > 0`; it can be
made small and never zero. And §19's export shows 4 µm of overshoot is not what makes a
design unbuildable — OCC built the selected iterate 24/24 at +0.0% `Kt` error.

**The cost is that the CONVERGED POINT IS NOT THE ANSWER.** The run must fall back to an
earlier feasible iterate and discards everything after it. So the goal is: **the point the
descent converges to should be the point that ships.** That rules out simply raising the
weight, which buys a smaller `v*` at worse conditioning and still leaves the converged point
strictly infeasible.

### The constraint any candidate must respect

`soft_barrier`'s docstring: quadratic **"C¹ at the knee — which is what an FD plateau needs,
and why the barrier is quadratic rather than linear in the first place."** A linear barrier
restores gradient at the knee and breaks the property M7's gradient gates rest on. So a
candidate must have **nonzero gradient at the constraint boundary AND remain C¹ there.**

It also has **19 call sites** in `wheel_objective.py`. Changing `soft_barrier` globally
re-baselines every committed artifact, which this repo does not do silently — so the change
is opt-in, defaulting to today's behaviour, exactly as `--kinematics` was.

### Standing rules, inherited

- **Measure before touching a threshold.** A red gate is the finding.
- **Measure both genomes**, and use §19's own run as the control for Step 5.
- **Today's behaviour stays the default.** Every committed artifact must still reproduce
  with no flag passed.
- **Check the target binds before ranking work** — twice now a #1 successor has not
  survived it (§17, §20). Step 1 is that check for this one.

---

## Step 1 — Measure the equilibrium law

**Why.** The law `v* = P/(2·scale)` is a derivation. This repo has twice shipped a
derivation that did not hold (`stress_scale`, §19's patch framing), and §20 closed by saying
the cheapest decisive check beats the plausible mechanism. If `P` measured at the converged
genome does not reproduce §19's 4.1 µm overshoot, the diagnosis is wrong and the fix would
be aimed at the wrong thing.

**Read first.** `src/wheel_objective.py:382` (`soft_barrier`), `:824` (the `fillet_cap`
violation), `:731-740` (`hub_fillet_r_effective` and the blend), PLAN.md §19's "Defect 5
bit" subsection.

**Do.** At `e126cc3`, no optimizer:

1. **From `stage3_margin_medium.json` alone** — the violation trajectory
   `v_k = r_hub - cap` per step, the `fillet_cap` term, and where it settles. Confirm the
   tail is an equilibrium (flat) rather than a drift.
2. **Per-term gradients at the converged genome**, using `objective(..., weights=...)` with
   every weight zeroed but one. That isolates `dL_term/dR_hub` for `fillet_cap` and
   `stress_margin` through the public API rather than reaching into the closure.
3. **The prediction, registered here before it is run:** at the converged iterate the two
   must be **equal and opposite** in the `R_hub` component, and
   `|dL_stress_margin/dR_hub| = 2 * 500 * v*`. With §19's `v* = 4.1 µm` that is **4.10**.

**Gate.** No pass/fail — but the arc does not proceed to Step 2 unless (3) holds to within
the step-to-step variation of the trajectory tail. A miss is the finding.

**Record — part 1, the trajectory, from `stage3_margin_medium.json` alone (no solving).**

```
STEP 1 RECORD, PART 1 — DONE (2026-08-13).  THE TAIL IS AN EQUILIBRIUM, NOT A DRIFT.
  74 of 101 iterates violate            reproduces §19 exactly
  converged overshoot, step 100         4.112 um    reproduces §19's "4.1 um over"
  tail, steps >= 80                     mean 4.500 um, std 0.513, min 3.843, max 5.525
  worst overshoot anywhere              36.667 um (step 12)
```

| step | cap mm | `R_hub` mm | v µm | `fillet_cap` | loss |
|---|---|---|---|---|---|
| 0 | 0.460207 | 0.457112 | −3.096 | 0.0 | 58.4115 |
| 30 | 0.659344 | 0.651037 | −8.307 | 0.0 | 52.1642 |
| **55** | 0.735233 | 0.727953 | **−7.279** | 0.0 | **51.3892 ← selected** |
| **56** | 0.733937 | 0.732812 | **−1.126** | 0.0 | 51.6558 |
| 57–100 | | | **all > 0** | | |
| 80 | 0.759841 | 0.763684 | +3.843 | 7.4e−03 | 51.1319 |
| 100 | 0.766966 | 0.771078 | **+4.112** | 8.45e−03 | 51.0771 |

**A free control fell out.** The violation recovered two independent ways — denormalising
`z[12]` and subtracting the reported cap, versus `sqrt(fillet_cap / 500)` from the term value
— agrees to **0.000 nm** across all 74 violating iterates. That confirms `w_cap = 500` and
that the barrier's argument is the raw gene against the cap (`:824`), which is what makes the
law free of any chain rule.

**And it corrects §19 on a detail.** §19 wrote *"Every single step from 56 to 100"* is in
violation and *"last feasible iterate: step 55 of 100."* Measured, **step 56 is feasible**
(v = −1.126 µm, `fillet_cap` exactly 0.0); the violating run is **57–100**. Step 55 is the
**selected** iterate — the lowest-loss tier-0 one — which is a different thing from the last
feasible one, and `selection_key` picked it over 56 on loss (51.3892 against 51.6558), both
being tier 0. The count of 74 is unaffected and every other §19 number here reproduces.

**The tail is flat with 0.5 µm of scatter around 4.5 µm** — which is what an equilibrium
looks like and not what a barrier losing ground looks like. The descent is not failing to
converge; it is converging to a point 4 µm outside the feasible set, exactly as the law says
it must.

**Record — part 2, the gradient balance.**

```
STEP 1 RECORD, PART 2 — DONE (2026-08-13).  THE BARRIER SIDE IS EXACT.  THE RUN HAD NOT
REACHED ITS OWN FIXED POINT, AND THAT IS THE CORRECTION TO THE PREDICTION AS REGISTERED.
  medium, SVK, the same 8 uniform phases the run used, genome = step 100 of
  stage3_margin_medium.json, terms isolated by zeroing every other weight.

  violation v at step 100                          4.1121 um
  dL_fillet_cap/dR_hub      measured               +4.112098
                            predicted 2*w_cap*v    +4.1121      <- EXACT, 6 s.f.
  dL_stress_margin/dR_hub   measured               -5.162679
  sum                                              -1.050581    <- NOT zero

  EVERY OTHER TERM'S dL/dR_hub AT THIS ITERATE IS IDENTICALLY 0.0.
    stress_margin  -5.162679
    fillet_cap     +4.112098
    TOTAL          -1.050581
```

**The barrier's derivative is exactly `2·w·v`** — 4.112098 measured against 4.1121 predicted,
six significant figures. That half of the law needed no assumption and now has none.

**What is refuted is my registration, not the law: step 100 is not a stationary point.** I
registered "the two must be equal and opposite at the converged iterate," which presumes the
run converged in `R_hub`. It did not — there is **1.05 of net outward gradient left** at the
step the budget ended on. So the correct test is self-consistency rather than balance:

```
    fixed point   v* = P / (2 * w_cap) = 5.162679 / 1000 = 5.163 um
    trajectory tail (steps >= 80)       3.843 - 5.525 um, mean 4.500, std 0.513
```

**The predicted fixed point lies inside the tail's own oscillation band**, and the run at
step 100 sits 1.05 µm below it and still climbing. The law describes an attractor the descent
was still approaching when the step budget ran out. That is a *stronger* statement about
defect 5 than the one I registered — the converged overshoot §19 reported as 4.1 µm is not
the endpoint, it is a snapshot of a trajectory heading for **5.2 µm**.

**And nothing was assumed away.** Sweeping all fourteen terms found **exactly two** with a
nonzero `dL/dR_hub`. `stress` contributes 0.0 because utilisation is below 1.0 and §15's
defect 1 makes that barrier identically flat there; every other term does not see `R_hub` at
all. So the equilibrium is a two-body problem and the law has no hidden third force in it.

#### DEFECT 5 AND DEFECT 8 ARE COUPLED, WHICH NEITHER §19 NOR §20 SAID

The opposing pull is **the `stress_margin` term §18 added**, and defect 8 is precisely that
`util**2` *"never stops wanting margin"* — its marginal price falls only 28% across the whole
range. So `P` does not decay as the design improves, and a barrier whose restoring force is
`2·w·v` must therefore sit permanently outside its own boundary. **Defect 8 is what keeps
defect 5 fed.** Fixing the barrier shape addresses the symptom at the boundary; fixing the
margin term's knee would reduce `P` itself. Step 2 must price both rather than assume the
barrier is the only lever — and it means the two successors PLAN.md §20 ranked #1 and #3 are
one problem seen from two sides.

**The sizing this hands to Step 2**, which is the number the whole arc turns on:

```
    to put the fixed point AT or INSIDE the boundary, a candidate must supply
    5.163 loss-units/mm of restoring slope AT v = 0
    -- equivalently, shift the knee by  delta >= P/(2*w) = 5.163 um.
```

---

## Step 2 — THE DESIGN DECISION

**Why.** Pre-registration, for the reason `study_gnl.py`'s gate meant something when it
fired. Write the candidates and their costs down before building one.

**Do.** Fill in from Step 1 and pick. The candidates that satisfy both constraints
(nonzero gradient at the boundary, C¹ there):

| candidate | shape | keeps C¹? | self-calibrating? | new constants |
|---|---|---|---|---|
| **A — shifted knee** | barrier on `v + δ`, so the knee sits δ inside the feasible set | yes | **no** | `δ` |
| **B — augmented Lagrangian** | `w·max(0,v)² + λ·v`, `λ` updated between accepted steps | yes | **yes** | update rule, `λ₀` |
| **C — raise the weight** | `v* = P/(2w)`; buy a smaller overshoot | yes | no | none |

**C is the control, not a candidate** — it cannot make the converged point feasible, only
less infeasible, and Step 1's law says by how much. It is worth costing because it is free.

**A's δ is a calibrated constant, and this repo has removed two of those** (`stress_scale`,
`MIN_JUNCTION_OVERLAP_MM3`) for going unrevisited — but Step 1's law makes δ *derivable*
rather than fitted: δ ≥ P/(2w) is exactly the condition that puts the equilibrium at or
inside the boundary. That is the difference between a calibration and a bound, and it is the
argument for A if A is chosen.

**Gate.** One option, with the reason, before Step 3 begins.

**Record.**

```
STEP 2 RECORD — DONE (2026-08-13).  THE DECISION IS *NONE OF THEM*.  THE ARC STOPS HERE.
  Defect 5 is real, its law is now exact, and its measured cost does not justify the fix.
  Steps 3-6 are NOT RUN.  best_solution.json untouched.  No barrier was changed.
```

**Step 2a, the check I owed my own successor.** Three arcs running, a #1-ranked successor has
failed the "does the target bind" test twice (§17, §20). So before choosing a barrier shape,
I asked what defect 5 actually costs — and the answer is **one export**, not a 6 h descent.

**§19's converged, "unbuildable" step 100 was exported. It builds perfectly.**

```
junction    R_req  R_worst     edges   wedge  Kt_model  Kt_built    error
hub         0.771    0.771     24/24     326.0     2.033     2.033    +0.0%
rim         3.000    3.000     24/24     326.0     1.405     1.405    +0.0%
OCC valid True | single solid | BRepCheck valid | self-intersecting False
degenerate edges 0 | min curvature R 0.7711 mm (floor 0.25) | 50.44 g
```

Genome `2674f42`, written to `defect5_step100.json`, exported as
`export/defect5_step100.step`. **It is not a promotion candidate** and was built only to
answer this question.

**Why it builds, and it is not luck.** The `fillet_cap` barrier defends the *modelled* cap,
and BUILD_PLAN Step 3 fitted that cap **2.4–3.9% pessimistic on purpose**. The fixed point
this arc measured sits at **0.673% of the cap** (5.163 µm on 0.767 mm) and step 100 at
**0.536%** — four to seven times inside the conservatism. BUILD_PLAN Step 6 already recorded
the same thing from the other direction: *"the Step 5 barrier was firing on a design that
builds — a 0.90% overshoot of a cap deliberately fitted 2.4–3.9% pessimistic."* This arc
measured a **smaller** overshoot than that one.

**And the design it discarded is the same wheel.** Step 55 (shipped) against step 100:

| | step 55 | step 100 | |
|---|---|---|---|
| loss | 51.3892 | 51.0771 | **−0.61%** |
| axle drop | 2.00806 (+0.40% off target) | 1.99487 (**−0.26%**) | better |
| hub utilisation | 0.77952 | 0.76957 | −1.28% |
| mesh mass | 40.509 g | 40.662 g | +0.38% |
| `Kt` hub | 2.06875 | 2.06085 | −0.38% |

**Every metric moves under 1%.** So defect 5's entire realised cost in §19 was **0.61% of
loss on a design that builds at +0.0% `Kt` error** — not a lost design, and not an
unbuildable one.

#### The decision, and the reason for each option not taken

- **A (shifted knee), B (augmented Lagrangian), C (raise the weight)** all aim at `v* = 0`.
  **`v* = 0` is not the right target.** The barrier's zero is placed at a cap deliberately
  4–7× more conservative than the overshoot, so driving `v` to zero buys nothing the part
  cares about and gives up real fillet radius.
- **D (fix defect 8 to reduce `P`)** would change which design is optimal and re-open §19's
  exchange rate. That is a much larger decision than a barrier shape and it should not ride
  in on the back of one.
- **Promoting step 100 instead** — considered and declined. 0.61% of loss and +0.38% of mass
  is not worth a re-promotion, a fresh export as `wheel.step`, and the regression-net churn
  §16 and §19 both paid for. The wheel that ships is the right wheel.

**What defect 5 actually is, restated so nobody re-ranks it on §19's framing.** It is not a
buildability defect. It is a **classification** defect: the tier system marks iterates
infeasible against a modelled cap that is itself conservative by several times the violation,
so it discards buildable designs. Anyone who fixes it should fix *that* — the placement of
the boundary, not the shape of the barrier — and should note that a tolerance band is another
calibrated constant, which is what BUILD_PLAN Step 3 spent an arc removing.

#### What this hands to the next arc, and why it is the strongest item left

`R_rim` sits **exactly on its box ceiling of 3.0 for 80 of 101 steps**, from step 21 onward,
including both the shipped iterate and the converged one:

```
R_rim box [0.5, 3.0]:  step 0  2.749468   step 55  3.000000   step 100  3.000000
```

**That is the one successor whose target is provably binding** — the optimizer is sitting on
the bound and would cross it if allowed — where the last three #1 candidates each turned out
not to bind. The ceiling has never been tested and the change is a gene-box edit plus a short
probe. It is now #1.

---

## Step 3 — Implement, opt-in, and prove the gradient survives

**Do.** The new shape behind a flag defaulting to today's behaviour, threaded like
`--kinematics`; recorded in `search_block` so a run record names which barrier it descended
on. Tests: the barrier's own algebra, the C¹ property at the knee, and that the flag is
inert when unused (a value diff against a run with no flag).

**Gate.** `make test` at **7 failed / 438 passed + the new tests**, no new red. And
`studies/study_gradient.py`'s G4 FD plateau **unmodified** — a barrier that breaks the
plateau is refused, whatever it does for feasibility.

**Record.** *(pending)*

---

## Step 4 — The cheap probe

**Do.** `coarse`, SVK, ~40 steps from `e126cc3`, new barrier on, `--min-wall 1.2`. §18's
probe is the template and its settings are the ones to match.

**Gate.** Registered before the run: the converged iterate is **tier 0** — `fillet_cap`
exactly 0.0 — and `selection_key` picks the last step rather than an early one. Anything
else and Step 5 does not launch.

**Record.** *(pending)*

---

## Step 5 — Production `medium` SVK descent

**Do.** Every knob identical to §19's run (`medium`, SVK, 100 steps, uniform 8-phase, seed 0,
4 workers, `--fidelity-check-every 25 --fidelity-check-config coarse`, from the shipped
genome, `--min-wall 1.2`) so the **only** difference is the barrier. ~6 h 20 m, capped.

**Gate.** Against §19's own trajectory as control: **74/101 violating iterates must fall to
few or none**, the selected iterate should be at or near the last step, and the loss at the
selected iterate must not be worse than §19's selected iterate (step 55, 51.3892).

**Record.** *(pending)*

---

## Step 6 — Export, check, and the promotion call

**Do.** `make export`, the manifest checks §19 ran (24/24 both junctions, `kt_error_pct`,
bite, min curvature), feasibility at **both** fidelities with slack (§16's trap), and then
the call — which is mine to make, not to escalate.

**Record.** *(pending)*

---

## Step 7 — Write the record into PLAN.md as §21

**Record.** *(pending)*
