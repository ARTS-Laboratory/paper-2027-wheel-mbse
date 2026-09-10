# BOUNDARY_PLAN.md — defect 5's boundary placement

**Open arc #8, lowest ranked. Created 2026-08-16, carried forward from PLAN §19 and §21.
~~Nothing started.~~**

**STATUS CORRECTED 2026-09-05 — PLAN §114. STEP 0 IS ANSWERED (§112, 2026-09-04) and the arc
stays at #8.** The wasted-descent ratio does not generalise — four of twenty-five committed
runs, all `medium`/SVK, 17.5%–42.8% each. See the Step 0 record at the foot of this file. The
header was left claiming nothing had started in the same change that appended that record;
this corrects it.

**VERSION CONTROL IS PART OF THIS PROJECT'S WORKFLOW — CHANGED 2026-08-19.** The rule that
stood here read *"Ignore version control entirely. Do not commit, branch, stage, revert or
otherwise touch git."* **It is superseded.** The rules live in `PLAN.md`'s header block and
**only** there, so they cannot drift across ten files: one commit per finished unit of work
on `feature`, `make test` green first, never while a study driver is mid-write, a study
commit carries its regenerated `.json` and `.jpg`, a promotion is one atomic commit and never
one file — and **commits carry no assistant or tool attribution, no `Co-Authored-By:`
trailer, no session link, no generated-with footer.**

---

## Why this arc exists — and why it is ranked last

**Defect 5 is real, understood, and unfixed:**

> `soft_barrier(v) = scale * max(0, v)**2` has **zero gradient at its own knee**, so against a
> term that pushes steadily outward it cannot hold the boundary — it settles at whatever
> overshoot makes the quadratic's slope match the opposing pull.

§19 measured the cost in wall-clock rather than in theory: **74 of 101 iterates in violation,
every step from 56 to 100**, each 1.4–11.0 µm over its hub fillet cap, with the run's loss
still improving the whole way. **45 steps of a 6 h 20 m run were spent descending into the
unbuildable.**

§18 had forecast the opposite ("defect 5 did not bite") from a 40-step `coarse` probe. That was
a true statement about the probe and a false forecast about everything after it — the tell was
that the opposition the barrier had to hold against *did not exist until §18 created it*.

**But the fix arc already ran and chose nothing.** `DEFECT5_PLAN.md` (deleted 2026-08-16; the
record is PLAN §21) reached its Step 2 with:

> **THE DECISION IS *NONE OF THEM*. THE ARC STOPS HERE.**

and §21 demoted the residual to last place with a number attached: **worth 0.61% of loss on the
evidence.** That number is why this is #8 and not higher. Check it before doing any work here —
this project's own memory is that a successor ranked #1 was worth zero because the branch it
fixed was 253% from binding.

## Why it has not simply been closed

Two reasons it survives as an open item rather than a closed one:

1. **The mitigation is a selection rule, not a fix.** `selection_key` returns the last tier-0
   iterate rather than the lowest-loss one, and §19 records that this is *"the only reason this
   run produced a shippable genome at all"* — twice in a row it caught a real over-cap
   promotion. The optimizer still descends into the unbuildable; the project just no longer
   *ships* from there. That is a correct guard and an unpaid cost.
2. **The cost is compute, and compute is what this project spends.** 45 wasted steps of a 6 h
   20 m run is 2.8 hours. If the descent is re-run often — and `KINEMATICS_PLAN.md` may force
   exactly that — the 0.61% loss figure understates what the defect is worth, because it prices
   the *answer* and not the *search*.

**That second point is the one thing that could re-rank this arc**, and it is measurable.

## THE PLAN

### Step 0 — re-price it, on the axis §21 did not use

§21 priced defect 5 at 0.61% **of loss**. Price it instead as **wasted descent**: over the
existing Stage-3 runs on disk (`stage3_prod_elite9.json`, `stage3_prod_elite10.json`,
`stage3_svk_*.json`, `stage3_margin_*.json`), count for each run:

- the iterate index of the last feasible point,
- the total iterates,
- the wall-clock past the last feasible point.

§19's run gives 55/100 and 2.8 hours. **If that ratio holds across runs, the defect costs
roughly half of every Stage-3 descent**, which is a different-order finding from 0.61% of loss
and would move this arc well up the list.

**If it does not hold — if §19 was the outlier — record that and leave the arc at #8.**

### Step 1 — only if Step 0 re-ranks it: revisit the rejected fixes

The `DEFECT5_PLAN.md` arc rejected all candidates, and §21 is the record of why. **Read that
record before proposing anything**, because re-proposing a rejected fix without new evidence is
the failure mode this tree keeps catching.

What Step 0 supplies that the original arc did not have is a *cost on the search rather than on
the answer*. A fix rejected as "not worth 0.61% of loss" may be worth half a descent. The
candidates themselves do not change; the threshold for accepting one does.

The mechanism to attack is specific and does not need re-derivation: **zero gradient at the
knee**. Any candidate has to hold a boundary against steady outward pressure, and the standard
answers (an exact penalty with a non-zero subgradient at the knee, an augmented-Lagrangian
multiplier, or a log-barrier with a schedule) each trade differently against
differentiability — which matters because `wheel_adjoint` has to push a gradient through it.

### Step 2 — whatever the outcome, do not remove the selection guard

`selection_key` returning the last tier-0 iterate is correct independently of whether defect 5
is fixed, and it has caught two real over-cap promotions. A barrier fix does not retire it.

## What must NOT happen

- **Do not re-propose a fix `DEFECT5_PLAN` already rejected without new evidence.** Step 0 is
  what would constitute new evidence.
- **Do not weaken or remove `selection_key`'s tier-0 rule.**
- **Do not forecast from a short `coarse` probe.** §18 did exactly that and was wrong in a way
  that cost 2.8 hours of a production run. Any claim that a fix holds must be measured on a run
  long enough for the opposing term to develop.
- **Do not re-descend and promote inside this arc.**

---

## RE-EXAMINED AFTER THE FILLET SWITCH — 2026-09-03. **DEFECT 5 DOES NOT REACH THE STRESS WALL, AND §99's MARGIN POLICY IS WHY.**

PLAN.md §106.  The obvious reading of §103 against this arc is that it re-ranks it hard:
§103 measured the shipped genome OVER the stress wall with `stress` reading 12.391 where it
read 0.0 before, so a barrier that *"cannot hold the boundary"* is now active at the starting
point of the descent ranked #1.  **Measured, it is not.**  `stress` does not stand alone at
its knee any more:

```
  stress         = 4000.0 * max(0, u - 1.00)^2     BARRIER      d/du at u=1.0 =  0.00
  stress_margin  =   89.21 * max(0, u - 0.80)^2    OBJECTIVE    d/du at u=1.0 = 35.68
```

`stress_margin` is in `OBJECTIVE_TERMS`, so it is in the loss at every step, and its knee
sits 0.20 INSIDE the wall.  At the wall it contributes a restoring slope of
`2 * 89.21 * 0.2` = **35.684** where the barrier contributes zero, so any outward pull under
35.68 cannot push utilisation past 1.0 at all.  **The stress wall is the first boundary in
this tree that defect 5's mechanism does not reach, and it got that way as a side effect of
§99's margin policy rather than as a defect-5 fix.**

**THIS NARROWS THE ARC RATHER THAN RE-RANKING IT.**  The eight remaining barriers —
`buckling`, `x_order`, `hub_overlap`, `fold`, `arrival`, `fillet`, `fillet_cap`, `min_sj` —
have no companion objective term with an inner knee, so defect 5 reaches all of them
undiluted, including `fillet_cap`, which is the one §19 measured wasting 45 steps of a
6h20m run.  Step 0 is unchanged and still the thing that would re-rank this.

**AND IT SUPPLIES THE NEW EVIDENCE THIS FILE REQUIRES.**  *"Do not re-propose a fix
`DEFECT5_PLAN` already rejected without new evidence."*  An inner-kneed companion term for
`fillet_cap` is not a re-proposal: `stress_margin` did not exist when that arc ran, and the
shape is now demonstrated to work on a live barrier at zero cost to differentiability.

## STEP 0 ANSWERED — 2026-09-04. THE RATIO DOES NOT GENERALISE. FOUR OF TWENTY-FIVE RUNS, ALL `medium`/SVK, 17.5%–42.8% EACH — THE ARC STAYS AT #8

PLAN.md §112.  `studies/study_boundary_waste.py` (`make boundarywaste`) reads every
committed `stage3_*.json` with a `steps` trace — 25 of them — and prices, per run, the wall
spent past the last iterate where every barrier held.  No solve: the steps array already
carries `terms.<name>.value` and `wall_s`.

§19's own run (`stage3_margin_medium.json`) reconstructs exactly — `settings.elapsed_s` =
22834.0 s = 6 h 20 m to the word, `fillet_cap` nonzero at 74 of 101 steps, last recovery at
step 56, permanently violated from 57 to 100 — so the driver is calibrated on the one row
this arc has always had a hand check for.

**Across all 25: 7.58 h of 75.24 h (10.1%) is spent past the last all-barriers-feasible
iterate.  But it is not spread across the descent generally — it is four runs:**

```
  stage3_margin_medium.json      medium/svk   56/101   2.57 h   42.8%
  stage3_buildcap_medium.json    medium/svk   61/101   2.28 h   37.8%
  stage3_knee_medium.json        medium/svk   74/101   1.46 h   24.6%
  stage3_buildcap2_medium.json   medium/svk   82/101   1.06 h   17.5%
```

All five `medium`/SVK full descents on disk are named above or here: the fifth,
`stage3_svk_medium.json`, loses 0.00 h.  Every one of the three 301-step `coarse` full
descents loses 0.00 h.  `stage3_minwall_2.2.json` loses a shallow 0.06 h (3.0%) late in a
`coarse` run — a different shape, not the same finding smaller.  `stage3_run_elite9.json`
(3 steps, unfilleted) never reaches feasibility at all, and it and `stage3_run_elite10.json`
predate `fillet_cap` existing in the term set — reported as a schema gap, not defaulted to
"feasible" on a term neither run ever evaluated.

**Step 0's own trigger resolves to its second branch.** *"If the ratio holds across runs,
the defect costs roughly half of every Stage-3 descent... if not, record that and leave the
arc at #8."* It does not hold as a property of the descent; it holds 4-of-5 times on
`medium`+SVK specifically, at 17.5%–42.8% each. **THE ARC STAYS AT #8.** Step 1 is therefore
not opened by this section — §106's inner-kneed-companion candidate for `fillet_cap` remains
the only thing that would re-rank it, unchanged from the 2026-09-03 block above.

**Scope, checked and not assumed:** all 25 artifacts predate §103 — none carries a `fillet`
key in `settings`. This prices defect 5 against the objective §103 replaced. Whether the
`medium`+SVK conditionality survives on the filleted objective is a question for the
Stage-3 re-run ranked #1 tree-wide, not for this driver.

What moved: `studies/study_boundary_waste.py`, `studies/study_boundary_waste.json`,
`Makefile`, PLAN.md §112, and this block. `best_solution.json` untouched, no threshold
moved.
