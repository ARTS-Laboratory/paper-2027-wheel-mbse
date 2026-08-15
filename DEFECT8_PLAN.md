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
| 4 | probe, then production descent if it earns one | **DONE — PROMOTED, `e126cc3` → `09e8188`** (2026-08-14). 6 h 19 m, 101 calls, exit 0, all five gate clauses pass. **48.64 g solid against 50.25 (−3.20%)**, deflection −0.129% against +0.403%, util 0.780 → 0.820, export 19/19 with `Kt` +0.0% at both junctions, feasible at BOTH fidelities, Inventor import clean. THE PROBE WAS SKIPPED on instruction |
| 5 | write the record into PLAN.md as §23 | **DONE** (2026-08-13). Step 4's own record is **§26**, and the gate breaks it uncovered are **§25** (2026-08-14) |

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

---

## Step 4 — The production descent under the knee *(RUNNING since 2026-08-13 20:56)*

**Why.** Step 3 changed the objective and nothing else, so every design number in this repo
is still an answer to the OLD question. `e126cc3` sits at util 0.780 against a knee at 0.800:
the margin term reads exactly 0.0 on it and `mass` has nothing opposing it until utilisation
climbs back to the knee. Whether that is worth grams — and how many — is a run, not an
argument.

**THE PROBE WAS SKIPPED, DELIBERATELY AND ON INSTRUCTION.** The status table's "probe, then
production descent if it earns one" wanted §18's 40-step `coarse` probe first. It was not run:
the production descent was launched directly. Recorded here rather than quietly, because the
probe is what would have caught a bad launch in 40 minutes instead of six hours. The two
things it would have gated are argued below instead — and neither is a guess about the
objective's shape, which is what Step 1–3 already measured.

**`make knee`** (`Makefile`, added with this step), capped and detached exactly as prod9/prod10:

```
systemd-run --user --unit=wheel-knee -p MemoryMax=16G --collect \
    --working-directory=$PWD /usr/bin/make knee
```

**EVERY KNOB IS §19'S** — `medium`, SVK, 100 steps, uniform 8-phase, seed 0, 4 workers,
`--fidelity-check-every 25 --fidelity-check-config coarse`, from the shipped genome,
`--min-wall 1.2`. So **§19's own run is an exact control** and the only difference between
`stage3_knee_medium.json` and `stage3_margin_medium.json` is `stress_margin`'s shape. Distinct
`--out`/`--best-out` for the reason every descent in the `Makefile` has them: clobbering the
control would destroy it. ~226 s/step measured on the three runs this mirrors, so ~6.3 h.

**Gate, registered before the numbers land.**

1. **Step 0 must read `loss` ≈ 33.674, not 51.389.** Step 0 *is* the shipped genome, and the
   only thing that moved is that its `stress_margin` of 17.7156 under `w * util**2` becomes
   **exactly 0.0** under the knee (util 0.7795 < 0.800). 51.3892 − 17.7156 = **33.6736**. This
   is the cheap half of what the skipped probe would have gated: if step 0 reads anything
   else, the run is not measuring the change Step 3 made and it stops there.
2. **The descent must go TO the knee and stop, not through it.** `stress_utilisation_hub`
   0.7795 → ~0.80, mass below **40.509 g**, and the `stress` wall untouched at 0.0. Utilisation
   running on to 0.9+ would say `w = 325` is too weak to hold its own knee — a Step 2 finding,
   not a run failure, and it would be reported as one.
3. **Tier 0 at the selected iterate** — every barrier exactly 0.0, `fillet_cap` included — and
   `selection_key` picking a late step rather than an early one. §16's trap and defect 6.
4. **Deflection**, and the gate needs amending before it is applied — **the incumbent fails
   it.** ±0.3% of the 2.0 mm target is a PLAN-LEVEL promotion gate (SVK_PLAN step 5,
   BUILD_PLAN steps 8 and 10, where it is the clause that BLOCKED the SVK candidate at
   +1.65%); it is nowhere in the code. **§19 promoted `e126cc3` at +0.403%** — outside it. §19
   records the number honestly ("+0.40% off the 2.0 mm target: 8 µm, physically irrelevant on
   this part, but the direction is real") but does **not** note that it crossed a gate every
   earlier arc treated as binding, so the tree has a gate that quietly stopped binding at the
   last promotion. Judging this candidate only against the band would block a design better
   than the wheel actually shipping; judging it only against the incumbent would retire the
   gate by default. **So both are measured — inside ±0.3%, and no worse than the incumbent's
   +0.403% — and if they disagree the disagreement is the finding**, to be settled in the
   promotion call rather than by picking the convenient one now.
5. The `coarse` fidelity checks at 25/50/75/100 are observation only and cannot redirect the
   descent. They go on the record for the rung gap SVK_PLAN step 6 was caught by.

**And what the gate deliberately does NOT include: mass.** The knee'd term is a PRICE, so a
run that buys utilisation and gives back grams is the term working, not failing. §19 refuted
its own +4.9% and confirmed the trade; this arc's number is whatever it is, and the promotion
call at Step 6 is where it has to justify itself against `e126cc3`.

**Two things observed while it runs, recorded here so they are not re-derived later.**

**The 16G cap was too tight and was raised to 32G mid-run.** Launched at the 16G every other
descent block in the `Makefile` uses, the unit sat at its ceiling — `memory.current` 15.3 GiB,
**2936 forced direct reclaims**, `oom_kill` 0. It held the control's pace throughout
(224–239 s/step against §19's 236/191/227), so no timing or number in this arc is affected;
the cap was raised with `systemctl --user set-property` because five more hours one allocation
from a kill switch is not a risk worth carrying for nothing. The box now reports **61 GiB
total, 49 free** — the "two descents do not fit in 31 GB" rule the memory blocks inherit was
written for a smaller machine and should be re-derived before it is trusted again.

**One `solve_reject`, and it is NOT the control's.** Step 1 threw `NewtonDivergedError` —
*line search failed after 20 backtracks at load step 1/1, iteration 5, residual 1.419e-10* —
and recovered on the retry. §19's run also had exactly one reject, but a different one: a
`RuntimeError` from contact load control at **step 83**. Same count, different failure, at
opposite ends of the descent. Worth a line in the record rather than a conclusion: the knee'd
objective takes a different first step, and the very small residual at failure suggests the
line search, not the physics.

**Record.**

```
STEP 4 RECORD — RUN DONE (2026-08-14).  A CANDIDATE, `09e8188`, AND IT IS LIGHTER ON BOTH
  MASS MEASURES WITH DEFLECTION BACK ON TARGET.  Promotion call pending two checks.
  22737.6 s (6 h 19 m), 101 objective calls, exit 0, 3 rejects.  Control: 22834 s, 101 calls.
  stage3_knee_medium.json / stage3_knee_best_medium.json.
```

**The gate, clause by clause.**

| # | registered before the run | measured | |
|---|---|---|---|
| 1 | step 0 reads 33.674, not 51.389 | **33.6736** | **PASS**, exactly |
| 2 | util → ~0.80 and stops; mass under 40.509 g | 0.780 → **0.820**, 39.470 g | **PASS** |
| 3 | tier 0 at the selected iterate, late step | tier 0, key `[0.0, 0.0, 32.7446]`, **step 74/100** | **PASS** |
| 4 | deflection: inside ±0.3% **and** no worse than the incumbent | **−0.129%** against shipped **+0.403%** | **PASS both ways** |
| 5 | `coarse` fidelity checks on the record | steps 0 and 25 recorded, neither failed | done |

Gate 4's two readings **agree**, so the ±0.3% question this arc surfaced — a plan-level gate the
incumbent violates — does not have to be settled here. It still wants settling.

**The candidate against the shipped wheel, same objective, same rung, same kinematics.** Step 0
*is* `e126cc3` under the knee, so this is apples to apples:

| | `e126cc3` (step 0) | `09e8188` (step 74) | |
|---|---|---|---|
| loss | 33.6736 | **32.7446** | −0.9290 |
| `mass` term | 33.2954 | 32.4410 | |
| `deflection` term | 0.0406 | **0.0041** | 10× better |
| `smoothness` | 0.3376 | 0.1678 | |
| `stress_margin` | **0.0** (inert, util < knee) | 0.1317 (live) | by construction |
| mesh mass | 40.509 g | **39.470 g** | −1.04 g |
| **OCC solid mass** | 50.25 g | **48.64 g** | **−1.61 g, −3.20%** |
| fillet mass (§24) | 4.406 g | 3.818 g | −0.588 g |
| axle drop | 2.00806 mm, +0.403% | **1.99742 mm, −0.129%** | |
| util hub / rim | 0.7795 / 0.5274 | 0.8201 / 0.5454 | |
| `R_hub` | 0.7280 | 0.6636 | |
| `R_rim` | 3.0000 | **3.0000** | still pinned at the ceiling |
| `t1` / `t2` | 1.2047 / 1.2524 | **1.2000 / 1.2000** | both onto the 1.2 floor |

**It is lighter on both mass measures, and by MORE on the built one than the modelled one** —
−3.20% of solid against −2.57% of mesh — because it also gave back 0.588 g of the fillet mass
§24 measured the objective cannot see. That is the first movement in this repo where the
unpriced mass moved in the *same* direction as the priced mass rather than against it.

**What it paid.** Utilisation 0.7795 → 0.8201, i.e. four points of the twenty-two §19 bought.
That is the knee's stated policy doing exactly what Step 2 specified — margin below 0.80 is
worthless, so the design settles just above it — and it is the thing a promotion has to accept
rather than a surprise. The `stress` wall at 1.0 is untouched and `stress_margin` is live at
0.1317, i.e. the term is now paying attention where §18 wanted it to.

**Export: 19 checks, 0 failures.** 24/24 edges at both junctions, one family each, `Kt` error
**+0.0%** at both — the §13 property the shipped part was promoted for, preserved. Bite
0.5344 hub / 1.6416 rim against a 0.25 floor. Hub fillet under its analytic cap with +0.02616
of slack. BRepCheck valid, no self-intersections, no degenerate edges. The geometry simplified:
**faces 123 → 111, edges 363 → 327**, min edge 0.0351 → 0.0287 mm, min face 0.7865 → 0.6417 mm²,
profile min curvature radius 4.65 → 6.88 mm (a *flatter* profile).

**Defect 5 still bites, and measurably less than in the control.** Iterates violating
`fillet_cap`: **46 of 101, against §19's 74**, and the selected iterate moves from step 55 to
**74**. The last 25 steps are still all violating in both runs, so §21's diagnosis stands
unchanged — the quadratic barrier cannot hold the boundary against live opposition — and the
defect-6 selection rule is again the only reason a shippable iterate came out.

**Three rejects against the control's one, and two are the control's own failure.** Step 1
`NewtonDivergedError` (line search, 20 backtracks, residual 1.419e-10); steps 25 and 76
`RuntimeError` from contact load control — *the same failure §19 hit at step 83*, at forces of
66.723336 and 66.723501 N against a 66.7233 N target, i.e. converged to six figures and failing
on the iteration cap rather than on physics. All three recovered on retry; none abandoned.

**Two things the max-stress column will invite and neither is the constraint.**
`max_stress_mpa` goes 128.83 → 166.35 (+29%) and `stress_scale_measured` 14.37 → 19.35 (+35%).
Both are the mesh-divergent field max and its diagnostic, which M8b-i.6 step 2 removed from the
constraint precisely because they converge at no exponent. The constraint is `util` = 0.8201,
under the wall. `min_scaled_jacobian` 0.8266 → 0.7828 is a real mesh-quality loss and is worth
watching, not gating on.

**Feasibility at both fidelities — and the gate that was supposed to answer it was dead.**
`make svk` is the §16-trap check and it could not run: two independent breaks, written up as
PLAN.md **§25**. `stress_margin` was unclassified in `study_svk_rescore.py` (§18 added the
term, the driver predates it), so the driver had exited 2 on line one for every run since
2026-08-13 — **including through §19's promotion** — and its §14 control was wired to
`best_solution.json` while comparing against a constant measured on `350f4c7`, which §19 made a
different wheel. Both fixed, the control now reproduces §14 at **0.00%** on both rows, and a
tripwire test was added so the classification break cannot recur silently.

**`coarse`, after the fix — the candidate is feasible under BOTH kinematics with every barrier
at 0.0:**

| genome | kin | drop mm | err | util | mass g | loss | barriers |
|---|---|---|---|---|---|---|---|
| shipped `e126cc3` | linear | 1.7153 | −14.23% | 0.7090 | 40.51 | 84.2905 | **0** |
| shipped `e126cc3` | svk | 1.9908 | −0.46% | 0.7707 | 40.51 | 33.6859 | **0** |
| **`09e8188`** | linear | 1.6777 | −16.11% | 0.7354 | 39.47 | 97.5232 | **0** |
| **`09e8188`** | svk | **1.9761** | **−1.20%** | **0.8057** | **39.47** | **32.9769** | **0** |

The SVK correction on the candidate is 17.783% against the shipped 16.061%, i.e. **this design
is slightly more nonlinear than the one it would replace** — consistent with it being thinner,
and not a gate.

**`medium`, the promotion rung — §16's TRAP IS CLEARED. Feasible at BOTH fidelities:**

| cfg | genome | util hub | deflection | mass g | `stress_margin` | barriers |
|---|---|---|---|---|---|---|
| coarse | shipped | 0.7707 | −0.460% | 40.509 | 0.0000 | **0** |
| coarse | **`09e8188`** | 0.8057 | −1.196% | 39.470 | 0.0105 | **0** |
| medium | shipped | 0.7795 | +0.403% | 40.509 | 0.0000 | **0** |
| medium | **`09e8188`** | **0.8201** | **−0.129%** | **39.470** | 0.1317 | **0** |

The `medium`/SVK row reproduces the descent's own selected loss to the digit — **32.7446** —
which is the check that the re-score and the descent are scoring the same thing. Control PASS
at 0.00% on both rows.

**The rung gap is present and is not new:** the candidate moves −1.196% → −0.129% between
rungs, the shipped genome −0.460% → +0.403%, i.e. ~1 pp in the same direction for both. That is
the gap SVK_PLAN step 6 was caught by, behaving identically on the two designs. At `coarse` the
candidate reads outside ±0.3%; **`medium` is the promotion rung** (SVK_PLAN step 6 settled
that), and there it is well inside.

**THE CALL — PROMOTED (2026-08-14). The Inventor import was clean.** It is better than the
incumbent on every axis a promotion turns on: 1.61 g lighter as built, 1.04 g lighter as
modelled, deflection −0.129% against +0.403%, `Kt` +0.0% at both junctions, tier 0, feasible at
both fidelities under both kinematics with every barrier at 0.0, export clean at 19/19. It
costs four of §19's twenty-two points of utilisation headroom, which is the knee's stated policy
rather than a side effect.

`best_solution.json` is **`09e8188`**, copied verbatim from `stage3_knee_best_medium.json` apart
from a `note` recording provenance — a field the shipped file used to carry and lost at §19.
`e126cc3` is preserved as `stage3_margin_best_medium.json`. `export/wheel.step` rebuilt (48.64 g,
24/24 both junctions, `Kt` +0.0%). `tests/test_golden.py` verified still reading
`best_solution_ga_beam.json`. PLAN.md's top banner amended — it had been stale since §16, two
promotions, which is the same defect as §25 in the mechanism meant to prevent it.

**This is the first promotion in this repo where the objective, the FEA, the exporter and the
gate that checks all three were each verified against the same genome in the same session.**
That is only remarkable because §25 established it had not been true for the previous one.
