# DEFECT8_PLAN.md — `util**2` has no knee, so the margin term never stops wanting margin

> **Ignore version control entirely. Do not commit, branch, stage, revert or otherwise
> touch git — it is not part of this project's workflow and nothing here depends on it.**

**The parent item is PLAN.md §22's ranked successor #1.** Named in §19, measured red by §18's
own rate gate, and landed on from three directions: it keeps defect 5 fed (§21) and it is what
makes `R_rim`'s ceiling look profitable (§22). This file is the milestone; PLAN.md gets a §23.

---

## Status

| step | what | status |
|---|---|---|
| 0 | baseline the gate | **INHERITED** — 7 failed / 438 passed (2026-08-13) |
| 1 | measure the defect as a curve, and measure the red gate's own cause | **DONE — SHAPE DEFECT CONFIRMED (2.0x where ~40x is wanted); the red gate is design AND fidelity, and the best evidence is a PASSING row** (2026-08-13) |
| 2 | THE SHAPE DECISION | **DONE — `soft_barrier(util − 0.80)`: the same function as the `stress` wall, knee moved in** (2026-08-13) |
| 3 | implement, with the rate gate re-derived | **DONE — GATE 7/438 → 6/440, THE RATE GATE WENT GREEN, NO NEW REDS** (2026-08-13) |
| 4 | probe, then production descent if it earns one | **OPEN** — the objective moved, so `e126cc3` is no longer its optimum. Nothing promoted |
| 5 | write the record into PLAN.md as §23 | **DONE** (2026-08-13) |

---

## The defect

`stress_margin` sums `w * util_j**2` over both junctions (`src/wheel_objective.py:1206`), so

```
    marginal price of margin  =  d(term)/d(util)  =  2 * w * util
```

which is **proportional to `util`**. A stated exchange rate can therefore be correct at exactly
one design. §18 set `w = 20` to make 1% of utilisation trade against 1% of mass at `util`
0.855; §19 then measured the consequence and named it defect 8:

> The quadratic was chosen in §18 so "the exchange rate steepens as margin disappears". It
> does — but far too weakly to encode the policy actually intended… **margin below roughly 0.8
> is close to worthless on this part, and margin above 0.95 is close to priceless. A quadratic
> cannot express a knee.**

---

## Step 1 — Measure the defect as a curve, and measure the red gate's own cause

**Why.** Two separate claims are bundled in §19's paragraph: that the quadratic is the wrong
shape, and that §18's rate gate going red is evidence of it. The first is about the objective;
the second is about a test. Four arcs running, this project has repeatedly found the second
kind of claim to be about the test rather than the design (§14 ×3, §20 ×2). They are measured
apart here.

**Record.**

```
STEP 1 RECORD — DONE (2026-08-13).  BOTH CLAIMS MEASURED, AND THEY ARE DIFFERENT CLAIMS.
  The SHAPE defect is real and worse than §19's "28%": the price varies 2.0x across the
  whole range where the stated policy needs ~40x.
  The RED GATE is the design AND the fidelity: e4219f3 passes at n_phase=2 where e126cc3
  fails, so the test discriminates -- but the threshold is only crossed at n_phase=2.
  The strongest evidence for defect 8 is neither: it is the 0.952 -> 0.607 drift of the
  stated rate between the two genomes at coarse/8, WHERE THE GATE PASSES.
```

### 1a — the shape, as a curve. The price is nearly flat across the whole range

Marginal price `d(term)/d(util)`, normalised to its value at `util` 0.855 where `w` was
calibrated, for the shipped shape and three candidates:

| `util` | **`util**2` (shipped)** | `util**6` | knee@0.80, n=2 | knee@0.80, n=4 |
|---|---|---|---|---|
| 0.50 | **0.585** | 0.068 | 0.000 | 0.000 |
| 0.60 | **0.702** | 0.170 | 0.000 | 0.000 |
| 0.70 | **0.819** | 0.368 | 0.000 | 0.000 |
| 0.78 | **0.912** | 0.632 | 0.000 | 0.000 |
| 0.855 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.90 | **1.053** | 1.292 | 1.818 | 6.011 |
| 0.95 | **1.111** | 1.694 | 2.727 | 20.285 |
| 0.99 | **1.158** | 2.081 | 3.455 | 41.226 |

**The shipped term's price varies by 2.0× across the entire meaningful range** — 0.585 at
`util` 0.50 against 1.158 at 0.99. It pays nearly as much for margin at a junction loafing at
half its allowable as at one about to yield. §19 quoted the drift as 28% between two specific
designs; measured across the range it is worse than that, because the ratio it should deliver
is not 2× but something closer to the 41× the knee gives.

**§19's stated policy maps onto `knee@0.80, n=4` almost exactly** — identically zero below 0.8
("close to worthless"), 20× at 0.95 and 41× at 0.99 ("close to priceless"). That is not a fit;
it is the policy sentence written as a function.

### 1b — the red gate is the design AND the fidelity, and the control is what says so

§18's `test_the_margin_weight_is_the_exchange_rate_it_claims_to_be` reads
`stress_utilisation_hub` at `tests/test_objective.py`'s module settings, which are
**`CFG = "smoke"` and `N_PHASE = 2`**. Reproduced exactly, then swept:

| genome | `n_phase` | cfg | `util` hub | ratio | gate [0.5, 2.0] |
|---|---|---|---|---|---|
| **`e126cc3`** | **2** | **smoke** | **0.56008** | **0.379** | **FAIL ← the live red** |
| `e126cc3` | 2 | coarse | 0.58406 | 0.412 | **FAIL** |
| `e126cc3` | 8 | smoke | 0.68096 | 0.560 | PASS |
| `e126cc3` | 8 | coarse | 0.70898 | 0.607 | PASS |
| `e126cc3` | 8 | medium (§19) | 0.77952 | 0.734 | PASS |
| `e4219f3` | 2 | smoke | 0.65569 | 0.560 | PASS |
| `e4219f3` | 2 | coarse | 0.71801 | 0.671 | PASS |
| `e4219f3` | 8 | smoke | 0.78659 | 0.806 | PASS |
| `e4219f3` | 8 | coarse | 0.85506 | 0.952 | PASS |

**THE CONTROL CORRECTS THE FIRST READING OF THIS TABLE, AND THE CORRECTION IS THE RESULT.**
On the shipped genome alone the verdict flips on `n_phase` and nothing else, which reads like a
pure test artefact. **It is not**: `e4219f3` passes at `n_phase = 2` in both configs, so the
test is not simply fragile — it discriminates between the two designs at every one of the four
settings, and the shipped genome is the lower one every time. The promotion moved the design to
where this gate fires; `n_phase = 2` is what carries it across the line rather than what
invents the drift. Both halves are true and only measuring both genomes separates them.

*(Two uniform phases sample a 30° period at 0° and 15°, which is not a phase average — PLAN.md
§1 already records what few samples of this integrand are worth. That is why the threshold is
crossed there and not at 8.)*

**And the row that matters most is a PASS.** `e4219f3` → `e126cc3` at `coarse`/8 takes the
ratio **0.952 → 0.607**, a 36% degradation of the stated policy at a fidelity where the gate
never fires. **That is defect 8 in one number, and it is better evidence than the red is**:
`util**2` means the rate degrades quadratically exactly as the design improves, so the policy
decays fastest when it is working. A gate watching for that should be reading the *trend*
across designs, not a threshold at one.

So §19's Group B reading — "one is defect 8 measured" — is right in substance and points at
the wrong number. The shape defect is 1a and the 0.952 → 0.607 drift; the red itself is that
drift plus an under-sampled fidelity.

### 1c — the fix is low-risk, and that is measured rather than hoped

The shipped design sits at `util_hub` **0.77952** at `medium`. A knee placed at **0.80** is
therefore essentially *at* the shipped design: the term would read ~0 there and stop pushing.
**§19's descent stopped where the correct policy would have put it deliberately** — 0.9964 →
0.7795 — while the quadratic that drove it there could not express that reason. So replacing
the shape re-states why the shipped wheel is right; it does not condemn it.

What it *would* change is every run after this one: the term would stop paying for margin the
part does not need, which is exactly the mechanism §22 measured making `R_rim`'s ceiling look
worth −0.84 of loss it never earns.

---

## Step 2 — THE SHAPE DECISION

**Do.** Pick from 1a's candidates, and state the knee as a policy number the way `w` was.
Open questions to settle first, both cheap:

- **Does the knee belong at 0.80, and is it one number or two?** The hub sits at 0.780 and the
  rim at 0.527. A single knee at 0.80 makes the rim's contribution identically zero forever,
  which is arguably correct (the rim is not the constraint) and arguably hides it.
- **`n = 2` or `n = 4`?** 1a says n=4 encodes "priceless above 0.95" and n=2 does not. But n=4
  near a knee is very stiff, and `soft_barrier`'s own docstring is a reminder that the FD
  plateau is what pays for shape choices here — so the M7 gradient gates decide this, not
  taste.
- **`stress_margin` is in `OBJECTIVE_TERMS`, not `BARRIER_TERMS`** (`:364`). A knee'd term
  that reads exactly 0.0 at the shipped design is behaviourally a barrier; check whether
  `selection_key` (§17, defect 6) still classifies correctly.

**Record.**

```
STEP 2/3 RECORD — DONE (2026-08-13).  SHIPPED THE SHAPE, NOT A NEW DESIGN.
  src/wheel_objective.py   MARGIN_KNEE_UTIL = 0.80, stated as policy like the weight;
                           stress_margin becomes soft_barrier(util_j - MARGIN_KNEE_UTIL)
                           per junction -- literally the `stress` wall's own function with
                           the knee moved from 1.0 in to 0.80.
                           DEFAULT_WEIGHTS["stress_margin"] 20.0 -> 325.0.
  tests/test_objective.py  the rate gate re-derived; the dead-gene regression split into a
                           below-knee and an above-knee half; `genes_over_knee` fixture.
  make test                6 failed / 440 passed  (was 7 / 438)
                           §18's rate gate WENT GREEN.  No new red.  +1 test.
```

**The three decisions, and the reason each beat its alternative.**

**One knee, per junction.** "Margin below 0.8 is worthless" is a statement about stress
margin, not about which junction; the rim reading zero is the intended meaning. Two knees
would be two policy numbers where the physics gives one.

**`n = 2`, reusing `soft_barrier`.** A quartic encodes "priceless above 0.95" (41× at util
0.99 against this shape's 3.5×) and was rejected: that steepness is **already the `stress`
wall's job**, and §18's own comment says so — *"`stress` stays exactly as it is: the wall is
still there, this only stops the approach to it being free."* Taking `n = 2` makes the term
the existing `soft_barrier` with a different knee, so there is no new function, no new C¹
argument to make, and the M7 FD plateau is the one already paid for.

**The weight is derived, not fitted.** At §18's own reference utilisation of 0.855 the exact
weight for "1% of utilisation costs 1% of mass" is **328.49**; 325.0 is that rounded **down**,
which is §18's stated rounding direction (down buys less margin — the conservative way for a
term whose purpose is to move the optimum). Holding the reference fixed is what makes this a
change of SHAPE at constant rate rather than shape and rate at once. The ~16× jump in the
number is entirely because the argument shrank from `util` to `util − 0.80`.

**Measured at the shipped genome, every fidelity: `stress_margin` = 0.0, gradient 0.0.** The
wheel that ships has enough margin and the term now stops paying for more.

#### Two consequences the tests forced, and neither was in the plan

**1. The knee puts the fillet genes back to sleep below 0.80, and that is not §15 defect 2
returning.** Defect 2 was flat below util **1.0** — dead at every design anyone would ship, a
14-gene search running in 8. The knee is flat below **0.80**, the region the project has
decided fillet is worth nothing in. The regression test now asserts *both* halves: flat below
the knee is asserted as CORRECT (an accidental return to a live-everywhere term fails there),
and live above it is unchanged.

**2. `R_rim` IS a dead gene under the knee, at every design in this repo — and §22 says that
is right.** The first version of the above-knee test asserted both radii carry gradient on a
design "above the knee". That conflated two junctions: on `e4219f3` the hub is at **0.85506**
and the rim at **0.47963**, so `dL/dR_rim` is exactly 0.0. The rim has never come near 0.80 on
any genome measured.

That is the same conclusion §22 reached from the opposite direction — raising `R_rim`'s box
ceiling was worth −0.84 of loss under the old `util**2` while every check the objective has on
that radius is blind (mass cannot see it, the FEA cannot see it, the rim has no buildability
cap). **The old term made `R_rim` look valuable; the knee prices it at what §22 measured it to
be worth, which is nothing.** Two arcs, opposite directions, same answer.

#### What is NOT done

**Step 4. The objective moved, so `e126cc3` is no longer its optimum** — it sits at util 0.780
against a knee at 0.800, so the margin term is inert and `mass` now has nothing opposing it
until utilisation climbs back to 0.80. A descent would find a slightly lighter wheel sitting
*at* the knee. That is the run this change earns and it has not been made. **Nothing is
promoted and `best_solution.json` is untouched.**
