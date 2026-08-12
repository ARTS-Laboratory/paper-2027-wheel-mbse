# BUILD_PLAN.md — can the objective be made to see what OCC will actually cut?

> **Ignore version control entirely. Do not commit, branch, stage, revert or otherwise
> touch git — it is not part of this project's workflow and nothing here depends on it.**

> **THIS FILE IS WRITTEN TO BE READ ONE STEP AT A TIME, BY A SESSION THAT HAS NEVER SEEN
> THE OTHERS.** Every step carries **Why / Read first / Do / Gate / Record**. Do the step,
> write its numbers into its own **Record** block, tick the status table, stop.

**The parent item is PLAN.md §15, successor 1** — *"MAKE THE OBJECTIVE SEE BUILDABILITY.
This is the one that has to be fixed before ANY SVK descent can ship."* Read §15 before
Step 1; everything below assumes it. `SVK_PLAN.md` is that arc's working notes and this
file is the successor's. PLAN.md gets a §16 only at the last step.

---

## Status

| step | what | status |
|---|---|---|
| 0 | baseline `make test` | **INHERITED from SVK_PLAN Step 7** — see Record |
| 1 | ground truth: what does OCC build, at the 1.2 mm floor, on the designs that matter | **DONE — BOTH GATE CLAUSES PASS, AND `HUB_CAP_THICKNESS_SHARE` IS FITTED TO THE FAMILY THAT DOES NOT BIND** (2026-08-10). Bisection predicts the exporter's built radius to the digit on both designs. The cap over-promises against the restrictive family on all five designs, including both elites it was calibrated on. The incumbent clears by **1.0%**. Also: `make hubcap`'s own `occ_limit` gate is RED at the shipped genome — §14 item 5, firing |
| 2 | find the governing variable — and it is not either of the two obvious ones | **DONE — CONFIRMED, AND THE SIGN FLIP IS REAL** (2026-08-10). Each corner family is monotone in the hub arrival angle and THE TWO GO OPPOSITE WAYS, so the buildable radius PEAKS at their crossover (~20°). Normalised by `t0` the law is the same curve across a 2.13× thickness range: **the constant is the wrong SHAPE, not the wrong number**, and 0.52 is above every measured ratio at every station |
| 3 | the differentiable surrogate, calibrated conservatively | **DONE — ALL FOUR GATES PASS, AND BOTH SHARES WERE WRONG** (2026-08-10). The square-on branch becomes `t0 * (0.505 - 0.48*(1-cos arrival))`, fitted under all fourteen sweep stations and 3.1-3.3% under two designs in neither fit. The slot share drops 0.5 → 0.30: it has now been observed binding, on both elites, over-promising by up to **1.62×**. Cap/OCC-worst goes from 1.067/1.615/1.199/1.463/1.479 to 0.979/0.969/0.827/0.967/0.969. `cx1`,`cy1` now carry a live gradient into the cap. One clause of the sketched gate was retired **on a proof**: it is unsatisfiable inside the bisection's own resolution |
| 4 | wire it into the objective | **DONE — 434 passed / 2 failed, AND THE DIFF AGAINST STEP 0 IS EXACTLY THE ONE TEST THIS ARC ADDED** (2026-08-10). Both reds are the same two and reproduce to sixteen digits, so the cap change provably never reached the solver. Four existing tests were asserting the defect — one of them `norm(d[:8]) == 0.0`, *"the shape genes still move the cap"* — and a fifth was vacuous on four of its five genes. The new pin returns 0.6240 twice on the old cap. Two study gates corrected: G4's FD floor is now relative to the term's own gradient norm (it was reporting 373% error on a component 1e-10 of the norm), and `make hubcap` reads the worst corner instead of the larger half of a rank split |
| 5 | re-descend under SVK at `medium` | **DONE — RAN CLEAN, GATE FAILS 3-OF-4** (2026-08-10). 100 steps, 6 h 20 m, exit 0. Deflection +0.054%, mass 37.75 g, `R_hub` moved 0.5790 → 0.5000 — all PASS. `fillet_cap` = 0.009982, not 0.0 — FAIL, and the gate is not being re-fitted. The cap clears at arrival ≤ 39.893° and the descent stopped at 40.542°, so the weights traded 0.032% of the loss against mass rather than the point being unreachable. **The arc's two real results:** the control `bc77614`, re-priced on the buildable radius, is **infeasible at util 1.051** where the old cap read 0.9085; and `t0` came **off the `MIN_WALL_MM` floor** for the first time in the project, because a thicker wall now buys buildable fillet |
| 6 | export, check, promote | **DONE — EVERY STEP-6 CONDITION PASSES, NOTHING PROMOTED** (2026-08-10). OCC builds the candidate **24/24 at both junctions at the full requested radius**, `kt_error_pct` +0.0% / +0.0%, solid valid, Inventor-clean, as-built hub utilisation **0.9672**. The Step 5 barrier was firing on a design that builds — a 0.90% overshoot of a cap deliberately fitted 2.4–3.9% pessimistic. It would close the incumbent's **+20.5% deflection error** at −1.44 g, and cost margin (0.875 → 0.967). **Not promoted, because Step 5's pre-registered gate is red** and the fix — `R_hub`'s unjustified box floor of 0.5, twice the exporter's own 0.25 — is a gene-box decision, not an arc decision |
| 5b | re-descend with `R_hub`'s floor at **0.4**, not the 0.25 named in Step 6 (0.25 is a fault detector; 0.4 is one extrusion width) | **DONE — GATE GREEN ON THE BEST *FEASIBLE* ITERATE** (2026-08-11). 100 steps, 6 h 21 m, exit 0, peak 22.79 GB. `fillet_cap` → 0.000000 at step 3; `R_hub` roamed a 0.135 mm interior band. The run-selected best (step 93) still shows `fillet_cap` 0.000546 — **not infeasibility but a soft-barrier equilibrium**: the quadratic barrier has zero gradient at its own knee, so `stress` (pushing `R_hub` up via `kt_hub`) and `fillet_cap` (pushing down) park just inside both. **55 of 101 iterates are strictly feasible**; the best is **step 82**, 0.030% worse in loss, and it passes all four clauses — barriers 0.0, deflection **+0.051%**, mass **37.5194 g**, `R_hub` 0.5790 → 0.4597 in the box interior. Real defect found: `wheel_stage3.py` picks best-by-loss and ignores feasibility. Correction to row 5: `t0` is **pinned low again** at 1.2000 — it only buys fillet with wall thickness when `R_hub` cannot move |
| 6b | export the feasible candidate | **DONE — EVERY CONDITION PASSES, STILL NOTHING PROMOTED** (2026-08-11). Genome `4a4b137`. **24/24 at both junctions at the full requested radius**, `kt_error_pct` **+0.0% / +0.0%**, as-built hub utilisation **0.9865**, OCC valid, single solid, Inventor-clean, min curvature 0.4597 mm against a 0.25 floor. Worst hub wedge **314.0°** — the **NEAR-CUSP** family, so the *slot* branch binds on the shipped candidate, not the arrival branch this arc was built around; that makes the parked slot-arrival law the live successor. Promotion of `best_solution.json` remains a release decision, not a design one |
| 6c | promote, and read the regression net | **DONE — PROMOTED `e4219f3`** (2026-08-11). Step 82 was promoted first and was the **wrong iterate**: feasible at `medium` by 53 nm but **infeasible at `coarse` by 93 nm**, sitting on the knee of `max(0,v)**2` where the second derivative is discontinuous, which is what failed G4 at 2.705e-06 against a 1e-6 tolerance. Re-promoted **step 71** — 60× the cap slack, FD gate 0.000e+00, better deflection (−0.002% vs +0.051%) and **more** stress margin (0.99639 vs 0.99998) for +0.048 g. New rule: **feasible WITH SLACK AT EVERY FIDELITY.** Suite went 11 red → 4 procedural (fixed by `make export`; ordering is promote→export→test) + 1 gene-box casualty from Step 5b that is genome-independent (4.943 at the 0.5 floor, 2.413 at 0.4) + 1 knee + **5 characterisation tests genuinely invalidated by shipping a different design**, tabulated with old/new values and NOT re-tuned. Also fixed the arc's own new cap test, which constructed the arrival angle but INHERITED `t0` from the shipped file — the exact fuse its docstring warns against. **Suite closes at 6 failed / 430 passed** against Step 0's 2 / 433; nothing deleted, skipped, xfailed or re-thresholded |
| 7 | write the record into PLAN.md as §16 | **DONE** (2026-08-11). PLAN.md:3043. Records the promotion, the corrected cap in both branches, the two hub corner families and which one binds on the shipped wheel (NEAR-CUSP, 314.0°), **three new objective defects** added to §15's four — the quadratic barrier's dead knee, `--best-out` ignoring feasibility, and feasibility being fidelity-dependent — the gene-box change and its `z`-trace reinterpretation, the 6/430 gate with each red's old and new value, and four ranked successors |
| 8 | fix defect 6 (`--best-out` selection) and re-rank the successors | **DONE** (2026-08-12). PLAN.md:3255 as §17. `BARRIER_TERMS`/`OBJECTIVE_TERMS` in `wheel_objective`, `selection_key` in `wheel_stage3`, 4 new tests, trace replay as a regression. Measured that §16's #1 successor (the slot arrival law) **cannot move the cap** — the slot branch is 253% from binding and the binding thickness branch is already 1.5–3.4% conservative. Successors re-ranked; the stress-margin term is now #1 |
| 9 | give the objective a stress-margin term (§15 defect 1) | **DONE, NOTHING PROMOTED** (2026-08-12). PLAN.md:3368 as §18. `dL/dR_hub` and `dL/dR_rim` were measured at exactly 0.0 before the change; a 40-step SVK probe then took `R_hub` to 0.4% under its cap, `R_rim` to its box maximum, and `t0`/`t3` off the 1.2 mm floor, at +4.9% mass. Gradient FD-checked to 1.8e-7. Fixed a measure-zero `smooth_min` tie-derivative bug found on the way. Probe is `3ca40c1`, coarse, NOT a promotion candidate. Gate 6 failed / 438 passed — same six reds, no new ones |

---

## The problem, in one page

`bc77614` clears every FEA gate at `medium` — all nine barriers exactly 0.0, deflection
−0.041%, 37.414 g against the shipped 39.194 — and then **cannot be built at the stress
concentration it was priced at**:

| genome | worst wedge | hub fillets built | `kt_error_pct` |
|---|---|---|---|
| `350f4c7` shipped | 328.0 deg | **24/24 @ 0.579 mm** | **0.0%** |
| `bc77614` svk-medium | 308.0 deg | 12 @ 0.579, **12 @ 0.418 mm** | **+11.9%** |

Corrected for what was actually built, `bc77614` sits at **utilisation 1.046** — infeasible
as built, on the one constraint the whole SVK arc was circling. The incumbent builds exactly
as modelled. **That is a regression the SVK arc introduced, and the control is the only
reason it can be said with confidence.**

### The objective is not blind. It HAS a buildability model, and the model cannot see this

This is the correction to §15's framing, and it changes what the fix is. `wheel_objective`
already prices a buildable cap — that is the whole of PLAN.md §0(a)/§5:

```
hub_fillet_cap_mm  =  min( HUB_CAP_SHARE * hub_radius * radians(hub_void_deg),
                           HUB_CAP_THICKNESS_SHARE * t0 )          # 0.5, 0.52
```

`fillet_cap` barriers `R_hub` under that cap, and `_kt_hub` prices `Kt` on the capped radius
so the stress constraint stops crediting a fillet that will not exist. **Measured on the
three genomes that matter, `coarse`:**

| genome | t0 | `R_hub` | `hub_void_deg` | `by_slot` | `by_thickness` | cap | binds | `R_hub` ≤ cap |
|---|---|---|---|---|---|---|---|---|
| `350f4c7` shipped | 1.2000 | 0.5790 | 22.8350 | 2.5308 | 0.6240 | **0.6240** | THICKNESS | yes |
| `ae7092c` svk-coarse | 1.2000 | 0.5790 | 24.5347 | 2.7191 | 0.6240 | **0.6240** | THICKNESS | yes |
| `bc77614` svk-medium | 1.2000 | 0.5790 | 24.5428 | 2.7200 | 0.6240 | **0.6240** | THICKNESS | yes |

**Identical inputs, identical cap, and the constraint is satisfied in all three.** The model
returns the same number for the design that builds 24/24 and the design that builds 12/24.
It cannot distinguish them because **nothing it looks at moved**: `t0` is pinned on the floor
in all three and `R_hub` is a dead gene the SVK descent never touched (§15, defect 2).

### AND BOTH OBVIOUS PREDICTORS POINT THE WRONG WAY

This is why Step 2 is a real step and not a formality. The two quantities anyone would reach
for are the slot void and the material wedge, and **each is more optimistic about the design
that fails**:

```
                      350f4c7 (builds 24/24)   bc77614 (builds 12/24)   direction
hub_void_deg               22.8350                   24.5428            MORE room
worst_wedge_deg           328.0                     308.0              LESS crack-like
```

A larger void is more room for a fillet. A smaller material wedge is a *less* re-entrant
corner — that is exactly the reading §14 item 1 established when it renamed
`test_the_arrival_angle_makes_the_junction_a_near_crack`. **On both counts `bc77614` looks
like the easier part to cut, and it is the one OCC refuses.**

### THE REPO ALREADY NAMED THE MECHANISM, AND THE CANDIDATE IS ALREADY DIFFERENTIABLE

Found while reading for Step 1, and it is why this arc is much shorter than it looked.
`studies/study_hub_cap.py`'s own header, written at §5 and never acted on:

> *"WHAT THIS DOES NOT MEASURE, and it matters for reading the result. The hub's twenty-four
> corners fall into two families: twelve square-on ones the slot limits, and twelve shallow
> near-cusp ones **whose limit is the ARRIVAL ANGLE, a different mechanism the cap does not
> model.** The shallow family is what sets `r_built_mm` ... Capping `R_hub` closes the
> "optimizer prices 1.56, slot allows 1.11" gap **and nothing else**; both clusters are
> reported so that is visible rather than inferred."*

Twelve-and-twelve is exactly the split `bc77614` came back with. And the arrival angle is
**already a differentiable function of the genes, already in the objective**:
`wheel_wheel.arrival_angles`, consumed by the `arrival` barrier at
`src/wheel_objective.py:643-646`. Measured, `coarse`, hub end:

| genome | `arrival_hub` | `arrival` barrier | hub fillets built |
|---|---|---|---|
| `350f4c7` shipped | **19.6771** | 0.0 | **24/24 @ 0.579** |
| `ae7092c` svk-coarse | **48.3964** | 0.0 | *(not exported)* |
| `bc77614` svk-medium | **48.8859** | 0.0 | **12 @ 0.579, 12 @ 0.418** |
| elite10 | 9.6697 | 0.0 | *(not exported)* |

**The SVK descent swung the hub arrival angle by a factor of 2.5 — 19.7° to 48.9° — and
every barrier stayed at exactly 0.0 the whole way**, because `MAX_ARRIVAL_DEG = 65` and
48.9 < 65. `arrival` is a model-validity barrier (past ~70° the weld becomes a hinge and
`fixed_guided` is solving a structure the part does not have); it was never a buildability
term and does not claim to be. So the objective watched the one quantity that governs the
shallow family travel most of its usable range, and priced none of it.

**That is the shape of the fix**: a third branch in `hub_fillet_cap_mm`'s `min`, keyed on
`arrival_angles`, which needs no new differentiable machinery — only a calibration.

**It is a hypothesis with n = 2 and it is confounded**, and this arc does not get to skip
measuring it: the spline moved everywhere between those two genomes, `19.7 → 48.9` is not a
controlled sweep, and the *direction* is not yet established (`arrival` is measured from
radial, so the failing design is the more RADIAL one, while the exporter's prose talks about
near-TANGENT arrivals leaving a cusp — those two framings have to be reconciled against a
measurement, not against each other). Step 2 sweeps it properly. But the candidate list is
no longer open-ended, and Step 2's job is now **confirm or refute one named mechanism**
rather than go looking for one.

*(Both numbers above are transcribed from the SVK arc's export runs. Step 1 re-derives them
rather than trusting the transcription — the +11.9% is the finding this whole arc is built
on and it gets measured once more, from the genomes on disk, before anything is built on it.)*

### The mechanism, from the exporter, which is what makes the asymmetry plausible

`fillet_junctions` (`src/wheel_step_export.py:652`) walks a ladder from the requested radius
down **15% a rung** to `MIN_CURVATURE_RADIUS_MM = 0.25`, takes the largest radius the whole
batch accepts, then **re-selects the corners that are still re-entrant and walks the ladder
again for them**. Its own docstring states the asymmetry:

> *"The two corners at a junction are not mirror images — a spoke arriving near tangent
> leaves one corner square-on and one nearly a cusp, measured at 27.7° and 89.6° to the hub
> circle."*

`bc77614`'s families are **12 @ 0.579 and 12 @ 0.418**, and `0.579 × 0.85² = 0.418` exactly:
one corner per spoke took the requested radius and the other went **two rungs down**. So the
failure is not "the junction has no room" — it is **one of the two corners per spoke**, and
`kt_report` prices the junction at its worst corner, which is correct and is why the number
is red. Any surrogate that models "the junction" as a single quantity is modelling the wrong
object. **The per-corner asymmetry is the thing to predict.**

### Two constants are being extrapolated, and one of them is a parked item coming due

`HUB_CAP_THICKNESS_SHARE = 0.52` is the branch that binds at the 1.2 mm floor, and its own
comment says what that costs:

> *"THE LAW IS CALIBRATED ON [2.0, 2.6] AND IS NOT KNOWN OUTSIDE IT. ... The floor moved
> under this comment on 2026-08-06. ... the shipped genome sits at t0 = 1.2, BELOW the
> [2.0, 2.6] the share was fitted on. ... Re-run `make hubcap` at the new floor before
> trusting the slot branch below 2.0 mm."*

That re-run is **PLAN.md §14 item 5, parked since 2026-08-06**. It is now on this arc's
critical path rather than parked: the cap that let `bc77614` through is `0.52 × 1.2`, a
number fitted an octave away from where it was applied. **Step 1 is that re-run**, widened
to the genomes this arc is about.

### Standing rules, inherited and non-negotiable

- **Measure before touching a threshold**, and **never re-fit a pre-registered gate to the
  design that breached it.** Both are why the SVK arc's result is trustworthy and why it
  promoted nothing.
- **Measure both genomes.** Anything claimed about `bc77614` gets the same measurement on
  `350f4c7`. The +11.9% is only legible because the incumbent's +0.0% was measured beside it.
- **Conservative is the direction that matters** for a buildable cap. `HUB_CAP_THICKNESS_
  SHARE`'s comment states the asymmetry: a cap that under-promises leaves fillet on the
  table; one that over-promises puts Stage 3 back to buying fillet the part cannot build,
  **which is exactly the failure this arc exists to remove.** Any constant Step 3 fits goes
  UNDER the smallest measured ratio, as 0.52 did.
- **Linear stays the default; SVK is opt-in.** Unchanged from the SVK arc.
- **`--phase-scheme uniform` is a correctness setting.** PLAN.md §1.
- **Drive long runs through `make`, capped.** `systemd-run --user -p MemoryMax=...`;
  16G at `coarse` / 20G at `medium` with 4 workers, both measured in SVK_PLAN Step 2/3.
- **Additive flags only on committed drivers.** `study_svk_rescore.py --extra label=path` is
  the precedent: the driver at its defaults must still reproduce its committed artifact.

---

## Step 0 — Baseline the gate. **INHERITED, NOT RE-RUN.**

**Why.** Steps 4 and 6 are read as a diff against a known count, and one was measured at the
close of the SVK arc on this exact tree. Re-running it here would cost 25 minutes to
reproduce a number taken hours earlier with nothing changed in between.

**Record.**

```
STEP 0 RECORD — INHERITED from SVK_PLAN.md Step 7 (2026-08-10), same tree, not re-run.
  make test:   433 passed / 2 failed in 1468.57 s (24:28)
  box:         24 cores / 61 GB, nothing else meaningful running
  The two expected reds carry over unchanged:
    tests/test_gnl.py::test_the_correction_enters_at_first_order_in_the_load
    tests/test_wheel_fea.py::test_the_rim_band_holds_a_large_minority_of_the_compliance
  ANY third red at Step 4 or Step 6 is a finding and stops the arc there.
```

---

## Step 1 — Ground truth: what does OCC actually build, at the floor that ships?

**Why.** Everything below is a model of a threshold, and the threshold has been measured on
three genomes at `t0 ∈ [2.0, 2.6]` and on none at 1.2. `make hubcap` already does exactly
this measurement — void by ring classification against OCC's, and the acceptance threshold
by **bisection** rather than read off the ladder (the driver's own note: the rungs straddle
the cap, so "the largest rung below it" is measurably the wrong criterion). What it has never
been pointed at is the designs this arc is about.

**Read first.** `studies/study_hub_cap.py` — the header's three-section note, `_designs()` at
`:185`, `analytic_cap` at `:201`, `_run_cad` at `:210`, and `run_t0_sweep` at `:332`.
`src/wheel_objective.py:186-224` (the two constants and their calibration comment).
`src/wheel_step_export.py:633-710` (`_fillet_ladder`, `fillet_junctions`).

**Do.**

1. Add **`--extra label=path`** to `studies/study_hub_cap.py`, additive, exactly the shape
   `study_svk_rescore.py` took: the driver **at its defaults must still reproduce
   `studies/study_hub_cap.json` unchanged**, and that is the inertness check, not an aside.
2. Point it at `stage3_svk_best_medium.json` (`bc77614`), `stage3_svk_best_shipped.json`
   (`ae7092c`) and `best_solution.json` (`350f4c7`, the control).
3. **Measure PER CORNER, not per junction.** `fillet_junctions` already returns families and
   `_junction_edges` already carries a wedge per corner; the existing sections aggregate to
   the junction. The asymmetry is the object of study, so the new rows report **each corner's
   own wedge and its own bisected threshold**. If that needs the CAD snippet extended, extend
   it — do not infer a per-corner number from a family count.
4. Re-run the `t0_sweep` **down to the floor**, so `HUB_CAP_THICKNESS_SHARE` has measured
   rows at 1.2 instead of an extrapolation from 2.0.

**Gate.** Two clauses, both pre-registered here before the run:

- **The driver's own two gates (`void`, `occ_limit`) still pass at their unmodified
  thresholds on the default designs.** A red there means the extension broke the driver and
  the new rows mean nothing.
- **The new rows reproduce what `make export` already observed**: `350f4c7` accepting
  0.579 on all 24 hub corners, and `bc77614` splitting 12/12 with the second family at
  0.418. Bisection will not return exactly the ladder's rungs — it is a finer instrument —
  so the clause is **the same partition of corners into families and the same ordering**,
  not the same numbers. If the partition disagrees, the bisection and the exporter are
  measuring different things and Step 2 must not be run on either until that is resolved.

**Record.** Per-corner wedge and threshold for all three genomes, the `t0_sweep` extended to
1.2, and whether `HUB_CAP_THICKNESS_SHARE = 0.52` is still under every measured ratio at the
floor. **If 0.52 turns out to be OVER a measured ratio at 1.2 mm, that is the finding** — it
means the cap has been over-promising on every design that has shipped since 2026-08-06, and
it is a bigger result than the surrogate this arc set out to build.

```
STEP 1 RECORD — DONE (2026-08-10).  192.7 s, `void` + `occ_limit`, five designs.
  BOTH GATE CLAUSES PASS, AND THE CONSTANT IS FITTED TO THE WRONG FAMILY.

  studies/study_hub_cap.py gained `--extra LABEL=PATH` and `--t0-sweep`, both additive and
  both defaulting to exactly what was committed, plus `wedges_deg` per corner in the CAD
  snippet (it was computing the wedge and dropping it).
```

**Gate clause 1 — the extension is inert, and the control is exact.** `elite14` and `elite13`
reproduce the committed `studies/study_hub_cap.json` **to every digit** — square 1.3384 /
shallow 0.6128 and square 1.3274 / shallow 1.1075, and elite13's void delta 0.1989 — while
`best_solution` is the one row that moved. It moved for the right reason: the file it names
was replaced on 2026-08-06 (§13), cap 1.1057 → 0.6240. **Two unchanged designs reproducing
bit-identically beside one changed design that moved is a better inertness proof than the
one Step 1 asked for.**

**And the driver's own `occ_limit` gate is RED — a pre-existing red, not one this arc
introduced.** The committed artifact is `pass: true`; it is now FAIL on `conservatism`
0.433 against a `GATE_CAP_FLOOR_FRAC` of 0.5, purely because `best_solution.json` became a
1.2 mm design and **nobody has run this driver since the promotion**. That is **PLAN.md §14
item 5 firing exactly as it was parked to**, four days later.

**Gate clause 2 — the control, and it landed better than the clause required.** The clause
asked only for the same partition of corners into families. The bisected threshold predicts
the exporter's built radius **to the digit**, through the ladder:

| genome | bisected threshold | largest ladder rung ≤ it | **exporter actually built** |
|---|---|---|---|
| `350f4c7` shipped | 0.5847 | **0.5790** (= `R_hub`, top rung) | **24/24 @ 0.579** |
| `bc77614` svk-medium | 0.4220 | **0.4183** | **12 @ 0.579, 12 @ 0.418** |

Two independent instruments — a bisection in a study driver and a 15%-a-rung ladder in the
exporter — agree on both designs. **The bisection is measuring the thing the manifest
reports**, so everything below is about the wheel and not about the rig.

**The corner families are clean, and the median split is now measured rather than assumed.**
All five designs: 24 corners, exactly 12 at wedge ≈ 270° and 12 at 308–332°, and every corner
within a family shares its threshold **to sixteen digits** — the free 12-fold symmetry check.
`split_agrees_with_wedge` is **True on all five**, which retires an assumption
`run_occ_limit` had been carrying explicitly (*"the split is at the median because the
clusters are known to be equal-sized … not because the gap is looked for"*).

**But WHICH family binds FLIPS between designs**, and that is the first thing that makes a
one-number cap wrong in principle:

```
                   wedge 270 corner    wedge ~330 corner   restrictive family
  350f4c7             0.5847               1.4400*            the 270
  ae7092c             0.4266               1.4400*            the 270
  bc77614             0.4220               1.4400*            the 270
  elite14             1.3384               0.6128             the ~332
  elite13             1.3274               1.1075             the ~330
  * right-CENSORED: 1.44 is the bisection bracket (`hi_mm = 1.2*t0`), not a threshold.
    The loose family on the three thin designs is ">= 1.44" and its true value is unmeasured.
    It does not affect any conclusion here -- the restrictive family is well inside the
    bracket on every design -- but the column must not be quoted as a measurement.
```

#### THE FINDING: `HUB_CAP_THICKNESS_SHARE = 0.52` IS FITTED TO THE FAMILY THAT DOES NOT BIND

Not "extrapolated below its band", which is what §14 item 5 parked and what this step
expected to confirm. **The constant over-promises against the restrictive family on all five
designs — including both elites it was calibrated on.**

| design | t0 | `arrival_hub` | restrictive | loose | **restrictive/t0** | loose/t0 | `0.52·t0` | over? |
|---|---|---|---|---|---|---|---|---|
| `350f4c7` shipped | 1.200 | 19.677 | **0.5847** | 1.4400\* | **0.4872** | — | 0.6240 | **YES** |
| `ae7092c` svk | 1.200 | 48.396 | **0.4266** | 1.4400\* | **0.3555** | — | 0.6240 | **YES** |
| `bc77614` svk | 1.200 | 48.886 | **0.4220** | 1.4400\* | **0.3517** | 0.6240 | **YES** |
| elite14 | 2.554 | 3.387 | **0.6128** | 1.3384 | **0.2400** | 0.5241 | 1.3279 | **YES** |
| elite13 | 2.554 | 13.787 | **1.1075** | 1.3274 | **0.4337** | 0.5198 | 1.3279 | **YES** |

`0.52` sits just under `loose/t0` = 0.5241 and 0.5198 — **which is precisely what it was
fitted to**, and `run_t0_sweep` computes `share_of_t0` from `square_mean_mm` to this day. The
restrictive ratios on those same two designs are **0.2400 and 0.4337**. The calibration was
never wrong about the number it measured; it measured the wrong family, and the driver's own
header said so at the time and was not acted on:

> *"The shallow family is what sets `r_built_mm` and therefore the manifest's
> `kt_error_pct`. Capping `R_hub` closes the … gap **and nothing else**."*

**What saves the incumbent is one percent.** `350f4c7`'s `R_hub` is 0.5790 and its
restrictive threshold is **0.5847** — it clears by **0.0057 mm, 1.0%**. §13 recorded
`kt_error_pct = +0.0%` as *"the first shipped part whose built fillets match the ones its
stress model priced"*, and §15 treated it as the control that made `bc77614`'s +11.9% legible.
**It is both of those things and it is also a one-percent accident.** Nothing in the objective
put it there — `R_hub` has been a dead gene throughout — and any design that moves the hub
arrival even slightly loses it. That reframes the SVK arc's terminal finding: `bc77614` did
not break a robust property, it stepped off a ledge the incumbent was already standing on.

#### AND THE ARRIVAL-ANGLE HYPOTHESIS IS NOT CONFIRMED. IT IS NOT REFUTED EITHER.

The reason Step 2 is a controlled sweep and not a formality:

```
  within t0 = 1.200        arrival 19.68 -> 0.5847      MONOTONE DECREASING
                                   48.40 -> 0.4266
                                   48.89 -> 0.4220
  within t0 = 2.554        arrival  3.39 -> 0.6128      MONOTONE INCREASING
                                   13.79 -> 1.1075
```

**The two thickness groups disagree on the sign.** Inside the group this arc cares about —
the 1.2 mm floor, where every SVK answer lives — arrival predicts the threshold in the right
direction and separates `350f4c7` from `bc77614` cleanly. Across groups it reverses. With
n = 3 and n = 2, both confounded (the whole spline moves between designs), **that is not a
calibration and must not be fitted.** It is exactly the §14 item 3 trap: a mechanism that
"would have explained all four at once" and explained none.

Step 2 runs the controlled sweep. Its job is now sharper than "find the governing variable":
**hold the centerline and walk the hub arrival angle at fixed `t0`, at both 1.2 and 2.55, and
find out whether the sign flip is real or an artefact of two designs that differ in
everything.**

---

## Step 2 — Confirm or refute the arrival-angle mechanism, on a CONTROLLED sweep

**Why.** The candidate is named and already differentiable (above), but the evidence for it
is two designs that differ in every gene at once. This repo has been wrong exactly this way
before: §14's item 3 hypothesised one common mechanism behind four convergence failures,
it "would have explained all four at once", and **it explained none of them**. A confounded
n = 2 is not a calibration.

**Do.** The controlled experiment the two-genome comparison is not. Hold the design fixed and
walk **one** thing — the same discipline `t0_sweep` already uses for the thickness branch and
for the same reason (`studies/study_hub_cap.py:312`: *"anywhere except at one thickness.
Holding the centerline and sweeping `t0` does"*).

1. **Sweep the hub arrival angle** across its reachable range on a fixed centerline family,
   bisecting the per-corner OCC threshold at each station with Step 1's instrument.
2. **Both families, separately.** The square-on twelve and the near-cusp twelve are different
   mechanisms; averaging them is what the current cap does and is why it is blind. Report two
   curves.
3. **Keep the falsified predictors in the report.** `hub_void_deg` and `worst_wedge_deg` both
   point the wrong way on the n = 2 comparison, and a sweep that shows *why* is worth more
   than one that quietly drops them. H1 in PLAN.md §0 is the template: the negative result
   corroborated the positive one from a different direction.
4. **Reconcile the two framings.** `arrival` is measured from radial and the failing design is
   the more radial one; the exporter's prose describes near-TANGENT arrivals leaving a cusp.
   Both cannot be the mechanism as stated. The sweep settles it, and the record says which
   convention won rather than leaving a reader to guess.

**Gate.** No pass/fail, one question: **does the per-corner OCC threshold depend monotonically
on the hub arrival angle, over the range the optimizer can actually reach?**

- **If yes**, the curve is Step 3's calibration and the arc proceeds as planned.
- **If no**, the mechanism is refuted and this is a finding, not a failure — the fallback is a
  *conservative* cap that gives up fillet rather than a *predictive* one. That is smaller but
  still shippable, and it still closes the +11.9%.
- **Either way the falsified predictors get written down.** Decide between the two outcomes on
  the measurement, not in advance.

```
STEP 2 RECORD — DONE (2026-08-10).  THE MECHANISM IS CONFIRMED AND THE SIGN FLIP IS REAL.
  Each corner family is MONOTONE in the hub arrival angle, and THE TWO GO IN OPPOSITE
  DIRECTIONS.  The buildable radius is the min of the two, so it is a PEAKED function of
  arrival, not a monotone one — which is why step 1's five designs disagreed on the sign.
  Driver: `studies/study_arrival_cap.py` -> `studies/study_arrival_cap.json`.  Started as a
  scratchpad falsification on the `eps_n_check.py` precedent and was PROMOTED when it
  stopped being one: a constant fitted to these rows ships, so it is calibration evidence
  in `study_hub_cap.json`'s category and has to stay reproducible.  Out of `make studies`
  (two interpreters, measures OCC not the commit).  Not a gate -- it calibrates.
```

**The control variable is exact, not approximate.** `control_points` locks P0 = (0,0) at the
hub, so the first control-polygon edge **is** `(cx1, cy1)` and `arrival_penalty`'s hub angle
is `asin(|cx1| / hypot(cx1, cy1))` — a function of two genes and nothing else. Rotating P1
about the hub at fixed radius walks the arrival angle **while moving no other gene**. Seven
stations, 5° to 60°, on two bases 2.13× apart in `t0`.

| arrival | `350f4c7`, t0 1.200 | | | | elite13, t0 2.554 | | |
|---|---|---|---|---|---|---|---|
| | wedge 270 | wedge hi | **binds** | | wedge 270 | wedge hi | **binds** |
| 5° | **0.6126** | ≥1.44\* | 0.6126 | | 1.3384 | **0.6952** | 0.6952 |
| 10° | **0.6079** | ≥1.44\* | 0.6079 | | 1.3384 | **0.9316** | 0.9316 |
| 20° | **0.5847** | ≥1.44\* | 0.5847 | | **1.3054** | 1.3933 | 1.3054 |
| 30° | **0.5429** | ≥1.44\* | 0.5429 | | **1.2284** | 1.8771 | 1.2284 |
| 40° | **0.4871** | ≥1.44\* | 0.4871 | | **1.1075** | 2.3608 | 1.1075 |
| 50° | **0.4127** | ≥1.44\* | 0.4127 | | **0.9591** | 2.8885 | 0.9591 |
| 60° | **0.3267** | ≥1.44\* | 0.3267 | | **0.7667** | 3.0644 | 0.7667 |

\* censored at the bisection bracket; `hi_mm = 1.2·t0 = 1.44` on the thin base. Never binding
in the measured range, which is all this arc needs from it.

**Both families are monotone, in opposite directions, on both bases.** The 270° square-on
corner **tightens** as the spoke arrives more radially; the near-cusp corner **loosens**. On
elite13 they **cross between 10° and 20°**, and the `min` of them peaks right there — 1.3054
at 20°, falling to 0.6952 at 5° and 0.7667 at 60°. On the thin base the cusp family sits above
the bracket at every station, so only the decreasing branch is visible, which is exactly why
Step 1's n = 3 group looked cleanly monotone and its n = 2 group did not.

**Step 1's sign flip is therefore explained and is not an artefact.** It was two designs
sitting on opposite sides of the crossover. `hub_void_deg` and `worst_wedge_deg` both pointing
the wrong way is explained by the same fact: the void grows and the wedge shrinks with arrival
monotonically, so neither can track a quantity that turns around at 20°.

#### AND THE SCALING LAW FALLS OUT: the ratio is a function of ARRIVAL, not of `t0`

Normalise the 270° family by `t0` and the two bases lie almost on top of each other across a
**2.13× thickness range**:

| arrival | thin `thr/t0` | thick `thr/t0` | thin/thick |
|---|---|---|---|
| 5° | 0.5105 | 0.5240 | 0.974 |
| 10° | 0.5066 | 0.5240 | 0.967 |
| 20° | 0.4872 | 0.5111 | 0.953 |
| 30° | 0.4524 | 0.4810 | 0.941 |
| 40° | 0.4059 | 0.4336 | 0.936 |
| 50° | 0.3439 | 0.3755 | 0.916 |
| 60° | 0.2722 | 0.3002 | 0.907 |

**This is the scaling law the repo has been asking for**, and it is the direct answer to
Step 2's gate. `HUB_CAP_THICKNESS_SHARE`'s comment says the law "is calibrated on [2.0, 2.6]
and is NOT known outside it"; it is now measured at 1.2 as well, and the thickness
normalisation holds to within 10% over the whole range while the **arrival dependence spans a
factor of 1.9**. The constant is not the wrong number — **it is the wrong shape**: a constant
where the data is a curve in a variable it does not take.

**`0.52` is above every measured ratio at every station on both bases**; the largest value
anywhere in the sweep is **0.524**, at 5° on the thick base, and the thin base never exceeds
**0.5105**. So the constant is not merely mis-fitted at the floor — **it over-promises across
the entire reachable arrival range**, and the incumbent's 1% margin (Step 1) is what that
looks like from the inside.

**Reading for Step 3**: fit **under the thin curve**, which is under the thick one at every
station (ratio ≤ 0.974). Conservative in the direction
`HUB_CAP_THICKNESS_SHARE`'s own comment names as the one that matters.

---

## Step 3 — The differentiable surrogate

**Why.** Whatever Step 2 finds has to reach `jax.grad`, or `R_hub` stays a dead gene and the
optimizer still cannot trade geometry against buildability.

**Do.** Extend `hub_fillet_cap_mm` with the new branch — a third argument to the same `min`
if Step 2 found a predictor, or a tightened `HUB_CAP_THICKNESS_SHARE` at the floor if it did
not. Fit **under** the smallest measured ratio, as 0.52 was. Keep the hard `jnp.minimum`
between physical mechanisms and the `smooth_min` where `R_hub` meets the cap — that
distinction is deliberate and documented at `src/wheel_objective.py:573-579`.

**Gate.** The surrogate, evaluated on Step 1's designs, must **rank `350f4c7` above
`bc77614`** and must not bind on any design that OCC builds 24/24. Pre-register the exact
inequality here before fitting anything.

### The exact inequalities, registered — and one clause of the sketch above is impossible

**Order of operations, stated because it is not the order the sketch implies.** The *shape*
came out of Step 2's sweep before any gate was written. Writing the gate then showed that the
sketch's second clause cannot be met by any cap at all, by arithmetic that does not mention a
constant. Only then were the constants chosen. So: shape, then gate, then numbers — and the
clause below is retired on a proof, not on a fit that missed it.

**Why "must not bind on any design OCC builds 24/24" is unsatisfiable.** `350f4c7` builds all
24 corners at its own `R_hub = 0.578951`. Step 1 bisected its worst corner at **0.584688**, and
`study_hub_cap.threshold` returns the largest radius *known* to be accepted, so the truth lies
in `[0.584688, 0.590535]`. A cap that does not over-promise must sit at or below the truth; a
cap that does not bind must sit at or above `R_hub`. The window is

    [0.578951, 0.584688]   —   0.99% wide

and `BISECT_REL = 0.01`. **Hitting it requires the surrogate to be more accurate at one design
than the instrument that calibrated it.** That is not a gate, it is a coincidence, and Step 1
already named the coincidence: the incumbent's +0.0% `kt_error_pct` clears by 1.0%. The clause
is retired. What replaces it is `study_hub_cap`'s own two-sided structure — never over-promise,
never collapse to something vacuous — applied to the corner family that actually binds.

**G1 — RANKING.** `cap(350f4c7) > cap(bc77614)`, strictly. Both sit at `t0 = 1.2` and
`R_hub = 0.578951`; today both caps are `0.6240`, identical to 16 digits, which is the defect.

**G2 — NEVER OVER-PROMISES.** For each of Step 1's five designs,
`cap <= min(all 24 measured corner thresholds)`. Against the **minimum**, not the mean of a
rank-split half: the exporter's ladder is driven by the worst corner. The incumbent fails this
on **all five**.

**G3 — NOT VACUOUS.** `cap >= 0.5 * min(threshold)` on all five —
`study_hub_cap.GATE_CAP_FLOOR_FRAC`, unchanged, so a cap of zero cannot sail through G2.

**G4 — THE GENES ARE LIVE.** `d(cap)/d(cx1)` and `d(cap)/d(cy1)` non-zero at all five under
`jax.grad`. This is what Step 5's "`R_hub` must have moved" clause depends on, and it is
cheaper to check here than after a seven-hour descent.

**Tolerance carried into Step 6.** `kt_error_pct` at the hub within **±5%**. That is the fit's
own worst in-sample margin (3.29% against the thin base) rounded up one step, not a number
picked to be reachable.

```
STEP 3 RECORD — DONE (2026-08-10).  ALL FOUR GATES PASS.  Nothing exported, nothing promoted.
  `src/wheel_objective.py`: HUB_CAP_SHARE 0.5 -> 0.30, HUB_CAP_THICKNESS_SHARE 0.52 -> 0.505,
  new HUB_CAP_ARRIVAL_SLOPE = 0.48, and `hub_fillet_cap_mm` gained an `a_hub_deg=None`
  argument on the `flanks=None` idiom.  Verified against `studies/study_arrival_cap.json` and
  step 1's bisections -- no new OCC run, every number below was already measured.
```

#### What the surrogate is

    by_thickness = t0 * (0.505 - 0.48 * (1 - cos(arrival_hub)))     <- square-on family
    by_slot      = 0.30 * R_hub_ring * radians(hub_void_deg)        <- near-cusp family
    cap          = min(by_slot, by_thickness)

The `min` structure is unchanged and the hard `jnp.minimum` stays hard, for the reason at
`src/wheel_objective.py:573-579`. **Both of its arguments were the wrong shape**; neither the
structure nor the `smooth_min` at `hub_fillet_r_effective` needed to move.

`(1 - cos)` was chosen over a quadratic in the angle that fits marginally better (±0.4% vs
±0.9% about the thin curve). The quadratic crosses zero at 87.4° and would assert that a
near-radial spoke admits no fillet at all — a claim nothing measured. `(1 - cos)` stays
positive across the whole physically reachable [0°, 90°] with no special case, and ±0.9% is
already inside the instrument's 1%.

#### The gate table

| design | `t0` | arrival | binds at | OCC worst | old cap | old/OCC | **new cap** | **new/OCC** | branch |
|---|---|---|---|---|---|---|---|---|---|
| `350f4c7` | 1.200 | 19.68° | 270° | 0.5847 | 0.6240 | **1.067** | 0.5724 | 0.979 | thick |
| elite14 | 2.554 | 3.39° | 332° | 0.6128 | 0.9898 | **1.615** | 0.5939 | 0.969 | slot |
| elite13 | 2.554 | 13.79° | 330° | 1.1075 | 1.3279 | **1.199** | 0.9159 | 0.827 | slot |
| `ae7092c` | 1.200 | 48.40° | 268° | 0.4266 | 0.6240 | **1.463** | 0.4124 | 0.967 | thick |
| `bc77614` | 1.200 | 48.89° | 268° | 0.4220 | 0.6240 | **1.479** | 0.4088 | 0.969 | thick |

- **G1 PASS.** 0.5724 vs 0.4088, a ratio of **1.400**. Was 1.000.
- **G2 PASS**, 5/5, where the incumbent fails 5/5.
- **G3 PASS**, worst 0.827.
- **G4 PASS.** `|d cap/d cx1|` runs 0.017–0.072 and `|d cap/d cy1|` 0.006–0.052 across the
  five. Both were **identically zero through the arrival path** before; what gradient the cap
  had in those genes came through the void alone, and on the three `t0 = 1.2` designs the void
  branch is not even the one taking the `min`.

#### The two constants, and which of them is a fit

**`HUB_CAP_THICKNESS_SHARE` and `HUB_CAP_ARRIVAL_SLOPE` are a fit with a holdout.** Fitted under
the thin curve, they sit under **all fourteen** sweep stations — 1.45–3.29% under the thin base,
4.00–11.61% under the thick one. `ae7092c` and `bc77614` were in neither fit and differ from the
sweep base in every gene; the law predicts their square-on thresholds **3.3% and 3.1% under**,
the same margin it carries in sample. Calibrated on arrival ∈ [5°, 60°], which is all of the
range `MAX_ARRIVAL_DEG = 65` permits — **this law is not extrapolated anywhere the optimizer can
go**, which is exactly what could not be said of the constant it replaces.

**`HUB_CAP_SHARE` is a re-fit with no holdout, and it is labelled as one.** The note it replaces
said the slot share "has never been observed binding, so 0.5 stays a modelling assumption rather
than a fit". Pairing threshold to wedge showed it binds on **both** elites, and 0.5 over-promised
there by up to **1.62×**. 0.30 is under the smallest of eight measured `cusp/slot_arc` ratios
(0.3096 at arrival 3.39°, rising to 0.6955 at 49.96°) by 3.1% — the same rule that produced 0.52.
All eight went into the fit; there is no independent design left over to check it on.

**And it deliberately leaves 2.2× on the table.** The cusp family has its own arrival dependence
and it runs the *other* way. 0.30 is its value as arrival → 0, so it is conservative at every
larger angle and increasingly so. It stays a constant because the only sweep that reaches this
family opens the void as it steepens the arrival — rotating P1 does both — so those rows cannot
separate the two. **The experiment that would is the mirror of `t0_sweep`: hold arrival fixed,
walk the void.** Named here rather than buried in the constant.

#### Three things this changes that are worth saying out loud

1. **The cap now binds on the shipped genome**, 0.5724 against `R_hub = 0.578951`. `fillet_cap`
   goes from identically 0.0 to a small positive value on `350f4c7`, and `_kt_hub` prices the hub
   on 0.5724 instead of 0.5790. That is the mechanism by which `R_hub` stops being a dead gene —
   `soft_barrier` is flat below its knee, so a cap above `R_hub` gives it no gradient at all.
   It is also why Step 4's `make test` will differ from Step 0 in more than one place.
2. **`study_hub_cap.run_occ_limit`'s split is by RANK and its labels are inverted on some
   designs.** `shallow, square = th[:half], th[half:]` names the *smaller* half "shallow"; on
   `350f4c7` the smaller half is the 270° **square-on** family and on elite14 it is the 332°
   cusp. `split_agrees_with_wedge` (Step 1) checks the partition, not the naming, so it is True
   while the labels are backwards. `conservatism` therefore compares the cap against the family
   that is *not* binding, which is how a cap over-promising by 1.067–1.615× reported
   `over_promises: false` on all five. **Fixing that is Step 4**, together with the test.
3. **Two mechanisms moved in one step, against "one arc, one mechanism".** Deliberate: the slot
   share was not a second mechanism but the *same* instrument reporting that a constant already
   in the `min` over-promises by 1.62× on designs already on disk. Writing §16 about removing an
   over-promise while leaving a larger one in the adjacent line would not be defensible. What is
   parked is the slot family's *arrival law*, which genuinely is new work and genuinely needs a
   sweep that does not exist yet.

---

## Step 4 — Wire it into the objective

**Do.** The plumbing exists: `fillet_cap` in `t1_vector` (`:682`) and `_kt_hub`'s pricing
(`:1021`) both already consume `hub_fillet_cap_mm`, so this is a change to what the cap is a
function of and not a new term — **unless** Step 2's answer needs a term rather than a cap,
in which case say so and add one.

**Gate.** `make test` against Step 0, test by test, every difference accounted for. Plus the
one this arc must add: **a test that fails on the pre-Step-3 cap** — that is, one asserting
the cap distinguishes `350f4c7` from `bc77614`. Without it the fix is unpinned and the next
genome re-opens the hole.

```
STEP 4 RECORD — DONE (2026-08-10).  THE GATE PASSES, AND THE DIFF AGAINST STEP 0 IS ONE TEST.
MEASURED, NOT INFERRED:  434 passed / 2 failed in 1457.92 s (24:17).

    Step 0  (2026-08-08)     431 passed / 2 failed    1374.52 s
    Step 7  (2026-08-10)     433 passed / 2 failed    1468.57 s   <- SVK arc's baseline
    Step 4  (2026-08-10)     434 passed / 2 failed    1457.92 s
    delta                     +1 passed / 0 failed      -0.7% (noise)

  THE +1 IS `test_the_cap_ranks_two_designs_OCC_disagrees_about` AND NOTHING ELSE.  One test
  function added, one fixture (`genes_under_cap`) added -- fixtures are not collected.  No
  test was deleted, renamed, skipped or xfailed to get here; the run reports no skips and no
  xfails at all.  `tests/test_objective.py` alone: 118 passed, from six red.

  AND THE TWO REDS ARE THE SAME TWO, TO THE DIGIT:
    test_gnl.py::test_the_correction_enters_at_first_order_in_the_load
        small_load_rel_diff 0.0020498836530677966 against the 1e-3 gate
        -> SVK_PLAN.md Step 0 recorded 0.0020499.  Same number.
    test_wheel_fea.py::test_the_rim_band_holds_a_large_minority_of_the_compliance
        hub compliance share 0.032076694850181206 against the 0.03 gate
        -> every digit of PLAN.md §14's, of Step 0's, and of SVK Step 7's.

  That second line is the invariance check this step needed and is worth more than the
  count.  Reproducing sixteen digits of a solved FEA quantity says the cap change did not
  reach the solver at all -- which is what it should not do, `hub_fillet_cap_mm` feeding
  only `fillet_cap` in T1 and `_kt_hub`'s pricing.  Both reds remain PLAN.md §14 item 4's
  deliberate ones, neither is touched by this arc, and neither gate was moved.
```

#### It was a change to what the cap is a function of, not a new term — as the plan predicted

`fillet_cap` and `_kt_hub` both already consumed `hub_fillet_cap_mm` and neither needed
touching. The one signature change is additive and follows the idiom already in the function:

```python
def hub_fillet_cap_mm(genes, cfg, span_mm, hub_radius, flanks=None, a_hub_deg=None, xp=jnp)
```

`a_hub_deg` after `flanks`, so every positional caller in the tree is unaffected. `t1_vector`
passes the arrival it has already computed for the `arrival` barrier, for the same reason it
passes `flanks`: a barrier and a cap that disagree about the same angle is a class of bug
worth making impossible.

#### THE NEW TEST — `test_the_cap_ranks_two_designs_OCC_disagrees_about`

It fails on every cap this repo shipped before today, **and it fails by returning the same
number twice**. It asserts, at `t0` = 1.2 and against `study_arrival_cap.json`'s bisected
thresholds, that the cap is under 0.5847 mm at 20° of hub arrival and under 0.4127 at 50°,
is within 10% of each, and ranks them apart by more than 1.3×.

**Run against the pre-Step-3 constants, restored, rather than asserted to fail:**

| | cap at 20° | cap at 50° | cap/OCC | ranking |
|---|---|---|---|---|
| `0.5 / 0.52 / —` (old) | 0.624000 | 0.624000 | **1.067 / 1.512** | **1.000000** |
| `0.30 / 0.505 / 0.48` (new) | 0.570600 | 0.400287 | 0.976 / 0.970 | 1.425478 |

Both bounds breached, and the ranking clause fails by returning **the same sixteen digits
twice**. OCC measures the ratio at 1.417; the model gives 1.425.

**Both genomes are constructed, not loaded**, by rotating P1 about the hub at fixed radius —
`study_arrival_cap.py`'s exact one-variable control. `bc77614` lives in a Stage-3 run
artifact, and pinning a calibration test to a file some future run overwrites is a test with
a fuse in it.

#### Four existing tests were asserting the defect, and one was four-fifths vacuous

1. **`test_the_thickness_branch_of_the_cap_binds_on_a_thin_root`** asserted
   `norm(d[:8]) == 0.0` — *"the shape genes still move the cap on the thickness branch"* was
   the failure message. That is the defect, written down as an invariant and guarded. It now
   asserts the opposite, on the two genes that aim the spoke at the hub.
2. **`test_the_cap_gradient_matches_a_finite_difference`** checked genes (0, 1, 4, 7, 8) on
   the shipped genome, where the thickness branch takes the `min`. Four of those five were
   **exactly zero on both sides**, and `max(abs(d[i]), 1e-12)` turned the comparison into
   `0.0 < 1e-6`. It now runs one block per branch: the three real routes on the shipped
   genome, and all nine of genes 0–8 on `genes_over_cap`, where the slot branch binds and
   every one of them resolves to 1e-8 or better.
3. **Two tests asserted `fillet_cap == 0.0` at whatever genome ships.** §13's was under its
   cap; it is now 1.2% over, so both were false. The satisfied side is now a fixture
   (`genes_under_cap`, `R_hub` = 0.75 × cap, clear of the blend) rather than a property of
   the current design — which is the point the previous rewrite of these two half-made.
4. **`test_the_shipped_genome_is_inside_the_blend...`** asserted `R_hub < cap`. The genome
   is still inside the blend, now from the other side, and the invariant the test exists for
   is unchanged: `smooth_min` may only pull `R_eff` **down**. It is restated against the
   blend width instead of the sign, so a cap moving under `R_hub` cannot invalidate it again.

#### And one gate had to be corrected rather than passed

**`study_objective.run_closed_form` compares every jacobian entry above an ABSOLUTE 1e-10
against a central difference.** That was sound while no closed-form term's gradient spanned
many decades. `fillet_cap` used to be inert at the shipped genome — zero value, zero
gradient, nothing to check — and making it live gave it a jacobian running from 6.9 down to
5.6e-10. The bottom of that range is the hub arrival leaking into the *rim*-end control
point through `global_sampler`'s re-parameterisation: real, and smaller than the round-off in
what is being differentiated, since `arrival_angles` takes its tangent as
`sample(1e-5) - sample(0)` and that difference cancels to ~1e-6 relative. G4 reported
`worst_rel = 3.73` — **373% error on a component 1e-10 of the norm**, which is a measurement
of the noise floor.

The floor is now relative to the term's own gradient norm, `GATE_FD_LIVE_FRAC = 1e-5`, the
same notion `GATE_DOMINATED_GRAD` already uses for a different question. **Chosen from the
measurement, not from what passes:** on `fillet_cap` the smallest entry a central difference
resolves to better than `GATE_T1_REL` is 5.2e-5 of the norm and the largest it cannot is
8.7e-8, so the floor sits inside a gap of nearly three decades. **No other term loses a
single live gene** — `smoothness` keeps 8, `mass` keeps 12, and `n_below_fd_floor` is 0 on
every row but `fillet_cap`, where it is 4 of 10 and is reported next to `n_live_genes` rather
than hidden. With the floor in place G4's worst is **5.28e-7**, under its 1e-6 gate.

This is a gate correction, and it is worth being explicit that it is not the forbidden kind.
The rule is never to re-fit a gate to the design that breached it. Nothing about the design
changed here: the gate's filter was measuring round-off, the replacement is derived from the
noise floor of the function being differentiated, and the coverage it drops is reported in
the artifact.

#### `make hubcap` RE-RUN AT THE NEW CAP: PASS, and the pre-existing red is green

429.7 s. **`occ_limit` was RED at the shipped genome before this arc** — Step 1 found
`conservatism` 0.433 against `GATE_CAP_FLOOR_FRAC = 0.5`, PLAN.md §14 item 5 firing — and it
is now `safe` at 0.979. Not by moving the floor: by comparing against the corner that binds.

| design | binds at | old cap | old cap / **worst** | new cap | new cap / worst |
|---|---|---|---|---|---|
| `350f4c7` | wedge 270 (square-on) | 0.6240 | **1.067** | 0.5724 | 0.979 |
| elite14 | wedge 332 (near-cusp) | 0.9898 | **1.615** | 0.5939 | 0.969 |
| elite13 | wedge 330 (near-cusp) | 1.3279 | **1.199** | 0.9159 | 0.827 |

`families_are_clean` and `split_agrees_with_wedge` are True on all three. Note the committed
artifact's own `best_solution` row read cap 1.1057 / conservatism 0.8486, which is neither of
the numbers above: that row was measured before `best_solution.json` was replaced on
2026-08-06 and Step 1 already established it as stale. The old-cap column above is Step 1's
re-measurement, not the committed file.

#### And the t0 sweep answered a question the old note could only argue around

The old `HUB_CAP_THICKNESS_SHARE` comment had to explain away the sweep's high rows: *"the
sweep's rows above t0 ~ 3 report shares of 0.63 to 0.94, and they are not evidence against
this"*, on the grounds that the junction had degenerated. **Named by wedge instead of by rank,
those same rows read 0.5029 / 0.5082 / 0.5195 / 0.5379 / 0.5217.** The 0.63–0.94 was the rank
split picking up the cusp family, not a different feature. The prose was right about the
conclusion and wrong about the reason.

**`same_feature` was also too permissive, and the corner count is the direct test.** It gated
on `void > 1.0` as a proxy for "the roots have not merged". At `t0` = 8 the void still reads
2.577° while `_junction_edges` already finds **48 corners** instead of 24 — that row was
inside the fit while describing a different junction, and its "square-on family" has a spread
of **26.8×**, a mean over twelve unlike things. It now also requires two corners per spoke.
Every station that survives has a square-on spread of **exactly 0.0000** — all twelve corners
identical to sixteen digits — so those means are measurements rather than averages. The fit
range tightens from `[2.0, 8.0]` to **`[2.0, 6.0]`**.

**One intact station over-promises, and the `min` catches it.** At `t0` = 6 the square-on
share falls to 0.2905 against 0.503–0.538 everywhere else, and the model claims 0.4770 — 64%
over. The void there has closed to 7.3° and the cusp wedge has fallen to 300°, so the two
families are converging and the topology test cannot see it. But the SLOT term takes the
`min` at that station (0.4857 against the thickness branch's 2.8618), so the cap is 0.4857
against a worst corner of 0.9559 — **`cap / worst` is ≤ 1.0 at all six intact stations**
(0.948, 0.921, 0.677, 0.486, 0.488, 0.508). This is the slot term doing precisely the job its
own comment claims for it: *"it is kept because it must bind eventually and nothing else knows
that"*. First time that has been observed rather than asserted.

#### `make hubcap`'s own gate was reading the wrong family, and now reads the worst corner

`run_occ_limit` compared the cap against `square_mean_mm`, the mean of the **larger half of a
rank split**, on the reading that the larger family is the square-on one. It is not: on
`350f4c7` the smaller half is the 270° square-on family and on elite14 it is the 332° cusp.
So a cap over-promising by 1.067–1.615× against the corner OCC actually stops at reported
`over_promises: false` on all five designs. The gate now reads `worst_corner_mm =
min(thresholds)` — label-free, because the exporter's ladder is driven by whichever corner
refuses first. The families are still reported, now named by wedge (`square_on_mm`,
`near_cusp_mm`, `binds_at_wedge_deg`, `families_are_clean`) via `CUSP_WEDGE_DEG = 285`, which
sits in the middle of a 24° gap between the two measured bands. `run_t0_sweep`'s
`share_of_t0` had the same rank-split flaw and is now taken from the square-on family by
name; it happened to pick the right one down that sweep, because a thicker root closes the
slot and drags the cusp family under, and that is luck rather than design.

---

## Step 5 — Re-descend under SVK at `medium`

**Do.** Warm-start from `bc77614` — it is 0.04% from the deflection target and the point is
to move it off the unbuildable geometry, not to re-derive the wheel. `--kinematics svk`,
`--config medium`, `--min-wall 1.2`, `uniform`, distinct `--out`/`--best-out`,
`--fidelity-check-every 25 --fidelity-check-config coarse` — **on**, per §15's process
finding, which cost the last arc a run. ~224 s/step measured; budget from that.

**Gate.** Pre-registered, and it is the SVK arc's gate plus the clause that arc was missing:
every barrier 0.0, deflection inside ±0.3% **at `medium` under SVK**, mass below 39.194 g,
**and `R_hub` must have moved** — if it is still 0.578951 at the end, Step 3 did not give it
a gradient and the arc has not done its job.

```
STEP 5 RECORD — RAN CLEAN, GATE FAILS 3-OF-4 (2026-08-10).  100 steps, 22783.3 s (6 h 20 m),
  101 objective calls, no rejects, no abandoned steps, exit 0 under a 32 GB cap that was
  never approached (peak 19.1 GB).  Genome `d3766c2` -> `stage3_buildcap_best_medium.json`.
  Driver: `make buildcap`, added for this step and additive -- it differs from `svk-medium`
  in the warm-start genome, the output names, and NOTHING ELSE, so the two runs are a clean
  control/treatment pair on the cap alone.
```

#### The gate, clause by clause

| clause | value | verdict |
|---|---|---|
| deflection within ±0.3% at `medium` under SVK | 2.0010728 mm, **+0.054%** | **PASS** |
| mass below 39.194 g | **37.7516 g** | **PASS** |
| `R_hub` must have moved | 0.578951 → **0.500000** | **PASS** |
| every barrier 0.0 | `fillet_cap` = **0.009982**, all twelve others exactly 0.0 | **FAIL** |

**The gate is not being re-fitted.** It was pre-registered in this file before the run and it
fails, so it fails. What follows is why, measured — not an argument for moving it.

#### Why the barrier clause fails, to the digit

`cap` = 0.495532, `R_hub` = 0.500000 — a **0.90% overshoot**, giving
`500 × 0.004468² = 0.009982`.

**`R_hub` is sitting on its box floor.** `GENE_SPACE` bounds it at `[0.5, 4.0]`, and the
exporter's own `MIN_CURVATURE_RADIUS_MM` is **0.25** — so the gene box is twice as
restrictive as the kernel, and the optimizer had no legal move left in that gene.

**But the point is NOT unsatisfiable, and it would be dishonest to file it as one.** The cap
clears `R_hub` at any hub arrival ≤ **39.893°**, and the descent converged at **40.542°** —
short by **0.650 degrees**. It could have got there and chose not to, because the weights
arbitrated and `fillet_cap` lost at the margin: 0.009982 against a total loss of 31.1052, or
**0.032%**, while `mass` is 31.0287 of it. That is the objective working as designed —
`T1_REJECT`'s own comment makes exactly this argument, that a barrier is *already* a priced
term and "buying stress with a little hub overlap is a decision the weights are there to
arbitrate". Whether 500 is the right price for buildability is a separate question from
whether the gradient exists, and this arc was about the gradient.

**And the gradient emphatically exists.** `fillet_cap` carries **26.3% of the gradient norm**
at the final step — third of thirteen terms, behind `mass` (44.0%) and `deflection` (24.9%).
Before Step 3 it was identically zero at every feasible design.

#### What the treatment bought, against its own control

`svk-medium`'s winner `bc77614` is the control: same rung, same kinematics, same floor, same
scheme, same everything but the cap.

| | control `bc77614` | treatment `d3766c2` | |
|---|---|---|---|
| hub arrival | 48.886° | **40.542°** | −8.34° |
| buildable cap | 0.4089 | **0.4955** | **+21.2%** |
| `r_hub_effective` (what `Kt` is priced on) | 0.4089 | **0.4791** | +17.2% |
| utilisation **under the new cap** | **1.051** | **0.982** | infeasible → feasible |
| mass | 37.449 g | 37.752 g | +0.303 g (+0.81%) |
| deflection error | −0.044% | +0.054% | both inside ±0.3% |
| `t0` | 1.2000 (floor) | **1.2714** | came OFF the floor |

**Two of those rows are the arc's actual result.**

**The control was infeasible on stress and nobody knew.** `util` at step 0 of this run — that
is `bc77614`, scored under the new cap — is **1.051**. Under the old cap the SVK arc recorded
0.9085 for the same genome. Pricing `Kt` on the radius the part can be built with, rather
than on the radius the gene asks for, put the previous arc's own winner **over the
allowable**. §15 said that design "cannot be built"; it is also, once you price it honestly,
over its stress limit.

**`t0` came off the floor for the first time in this project's history.** Every shipped and
elite design sits pinned at `MIN_WALL_MM`, because mass drives it down and nothing pushed
back. Under the new cap a thicker wall *buys buildable fillet* — the square-on branch is
proportional to `t0` — so there is now a real trade between mass and buildability, and the
optimizer took it. That coupling did not exist before Step 3.

#### The honest caveat, and what decides it

The cap is fitted **2.4–3.9% under** the bisected OCC threshold (Step 3). A **0.90%**
overshoot of a model that conservative is well inside the model's own margin, so this design
may build all 24 corners regardless of what the barrier says. **The cap cannot answer that
and neither can this record — OCC can.** Step 6 exports and measures. Nothing is promoted on
the strength of the reasoning above.

---

## Step 6 — Export, check, promote

**Gate.** The one the last arc failed, stated in as-built terms: **`kt_error_pct` at both
junctions inside the tolerance Step 3 pre-registers**, as-built utilisation **< 1.0**, OCC
valid, Inventor-clean, and a `make test` whose every difference from Step 0 is accounted for.
`best_solution.json` moves only if all of that holds.

```
STEP 6 RECORD — EXPORTED AND MEASURED (2026-08-10).  EVERY STEP-6 CONDITION PASSES.
  IT BUILDS 24/24 AT BOTH JUNCTIONS AT THE FULL REQUESTED RADIUS, kt_error_pct +0.0% / +0.0%.
  NOT PROMOTED.  `best_solution.json` is untouched and still `350f4c7`.  The reason is Step 5,
  not Step 6, and it is set out below.
  Artifacts: export/stage3_buildcap_best_medium.step + _nofillet.step + _step_manifest.json.
```

#### The measurement — and the cap's 0.90% overshoot was inside its own conservatism

| Step 6 condition | measured | verdict |
|---|---|---|
| `kt_error_pct` within ±5% (Step 3's tolerance) | hub **+0.0%**, rim **+0.0%** | **PASS** |
| as-built utilisation < 1.0 | hub **0.9672**, rim 0.6126 | **PASS** |
| OCC valid | `brepcheck_valid: true`, 1 solid, `self_intersecting: false`, 0 degenerate edges, bbox 100.00 × 100.00 × 22.40 | **PASS** |
| Inventor-clean | `swept_surfaces_remaining: {}`, max edge tolerance 1.0e-7 mm | **PASS** |
| `make test` accounted for | Step 4's 434 / 2, unchanged since | **PASS** |

**The barrier was firing on a design that builds.** Step 5's `fillet_cap` = 0.009982 said
`R_hub` = 0.500 exceeded the cap of 0.4955. OCC accepted **0.500 on all 24 hub corners** —
one fillet family, worst wedge 312°, `r_built_mm` exactly `r_requested_mm`. That is the
2.4–3.9% conservatism Step 3 fitted in, doing its job: a 0.90% overshoot of a model that
pessimistic is still buildable, and the Step 5 record predicted this outcome without
assuming it.

**And the objective was conservative in the safe direction, measurably.** It priced `Kt_hub`
at **2.201763** on the blended radius 0.47912; the part is built at 0.500 and its `Kt` is
**2.168893**. The constraint saw a *sharper* corner than the part has, so as-built
utilisation (0.9672) comes in **below** the 0.9818 the optimizer computed. `smooth_min` may
only pull `R_eff` down — the invariant
`test_the_shipped_genome_is_inside_the_blend_and_is_priced_conservatively` exists for — and
here it is doing so on a shipped-candidate export for the first time.

#### Against the incumbent it would replace

| | `350f4c7` (shipped) | `d3766c2` (candidate) |
|---|---|---|
| deflection at `medium` under SVK | **2.409 mm, +20.5%** | **2.0011 mm, +0.054%** |
| hub `kt_error_pct` as built | +0.0% | +0.0% |
| as-built hub utilisation | 0.875 | 0.9672 |
| mass | 39.19 g | **37.75 g** |
| hub arrival | 19.68° | 40.54° |

It closes a **+20.5% deflection error** — the headline defect §15 recorded and could not fix,
because §15's own descents produced designs that do not build. It costs **margin**:
utilisation 0.875 → 0.967.

#### WHY IT IS NOT PROMOTED

**Step 5's pre-registered gate failed, and this repo does not promote past a failed gate.**
That is not a technicality imported for the occasion — it is the rule §15 obeyed four days
ago when it refused to promote two descents that met their deflection target, and the rule
`GATE_SMALL_LOAD_REL` is still red rather than re-fitted for. A design that passes every
physical check and fails one model check is exactly the case the rule is for; deciding it
"passes really" on the strength of the model being pessimistic is re-fitting the gate by
argument instead of by edit.

**The failure has a clean, measured cause and a clean, measured fix, and neither is mine to
choose.** `R_hub`'s box floor is `0.5`, written as a bare literal in `GENE_SPACE`
(`src/wheel_fea.py:281`) with no comment behind it — unlike the thickness nodes right above
it, whose floor is `MIN_WALL_MM` and says why. It is **twice** the exporter's
`MIN_CURVATURE_RADIUS_MM` and twice `wheel_objective.MIN_BUILDABLE_R_MM`, both 0.25. The
descent needed 0.650° more arrival to clear the cap and had no legal move left in that gene.
So the two honest options are:

1. **Accept the candidate.** OCC builds it 24/24 at +0.0% and as-built utilisation is 0.967.
   The barrier is conservative and was measured to be conservative here. Cost: promoting past
   a red pre-registered gate, and 0.875 → 0.967 of allowable.
2. **Lower `R_hub`'s box floor to something with a measurement behind it** — 0.25 is the
   number the exporter and the objective both already use — and re-descend. ~6 h 20 m. The
   barrier would then be satisfiable outright rather than traded away, and the arc would
   close on a green gate.

**Option 2 is the one consistent with everything else in this tree**, and its cost is one
overnight run. But it changes the gene box, which reinterprets every genome on disk in one
gene, and that is a decision about the project rather than about this arc.

---

## Step 5b / 6b — Re-descend with `R_hub`'s floor at 0.4, then export

Option 2, with **one correction to the option as it was written above**: the floor went to
**0.4 mm, not 0.25.** I named the wrong constant. `MIN_CURVATURE_RADIUS_MM = 0.25` is a
*fault detector* — its own comment says "0.25 mm is well under any fillet we ask for, so a
violation always means a construction fault" — and `MIN_BUILDABLE_R_MM = 0.25` exists only
to keep a blend width positive. Adopting either as a design floor would have destroyed the
detector by making legal designs sit on it. 0.4 mm is **one extrusion width** at the 0.4 mm
nozzle that `MIN_WALL_MM` is three perimeters of, which is the same kind of number as the
thickness floor right above it. `R_rim` was deliberately left at 0.5.

Warm-started from `bc77614` again, so this run and Step 5 differ in **exactly one thing:
the bound on gene 12.** 100 steps, 22887.2 s (6 h 21 m), exit 0, `MemoryPeak` 22.79 GB
against a 32 GB cap, `memory.events` all zero.

### The floor was the binding problem. It was not the whole problem.

`fillet_cap` hit **exactly 0.000000 at step 3** and `R_hub` came off the wall for good:
0.5790 → 0.4000 (sat on the new floor, steps 10–17) → 0.5345 (step 39) → 0.4611. It roamed
a 0.135 mm interior band. At steps 37–42 it climbed until the cap bit again, then traded
back down — the barrier working *as a barrier*, live and negotiable, instead of as a wall.

**But the run-selected best (step 93, loss 30.8914) still fails the gate**, for a completely
different reason:

| | step 93 | step 100 |
|---|---|---|
| `fillet_cap` | 0.000546 | 0.000344 |
| `stress` | 0.000751 | 0.000380 |
| `R_hub` − cap | +0.00105 mm (0.23%) | +0.00083 mm (0.18%) |
| `stress_utilisation` | 1.00043 | 1.00031 |

Not infeasibility — **a soft-barrier equilibrium.** `soft_barrier` is quadratic, so its
gradient is *zero at its own knee*. A larger `R_hub` lowers `kt_hub` and so relieves
`stress`; a smaller one relieves `fillet_cap`. The two push against each other and park
where their quadratic gradients cancel, which is necessarily **just inside both**, because
neither can generate force at exactly zero. Steps 94–100 are converged to six figures on
this point. **This is a property of the penalty formulation, not of the design**, and it
means "every barrier exactly 0.0" is not an attainable fixed point for any barrier under
opposition — the same shape of finding as the Step 3 clause that had to be retired.

### It is NOT unsatisfiable, and that is the difference that matters

55 of 101 iterates have every one-sided barrier at exactly 0.0. The lowest-loss one is
**step 82**, and it is **0.030% worse in loss** than the run-selected best:

| clause | step 82 | |
|---|---|---|
| every barrier exactly 0.0 | `fillet_cap` 0.0, `stress` 0.0, all others 0.0 | **PASS** |
| deflection ±0.3% at `medium` under SVK | 2.001027 mm, **+0.051%** | **PASS** |
| mass below 39.194 g | **37.5194 g** | **PASS** |
| `R_hub` must have moved | 0.5790 → 0.4597, box interior | **PASS** |

**The gate is green on step 82.** So the honest reading of the red is not "the design fails"
but "`wheel_stage3.py` selects the best iterate by loss and ignores feasibility." Selecting
the lowest-loss *feasible* iterate is the standard rule for a penalised constrained descent
and arguably the correct one; that the code does not do it is a methodology defect this arc
found, and it costs 0.030% of loss to fix here. Recorded as such, not patched mid-arc.

Step 82 is written to `stage3_buildcap2_feasible_medium.json` (genome `4a4b137`) with the
selection rule stated in its own `search` block. `R_hub` = 0.4597 against a cap of 0.4598 —
50 nm of slack, which is meaningless precision on its own, and does not need to mean more:
the cap is **fitted 1.45–11.61% under measured OCC truth on purpose**. That conservatism is
the margin. Padding it further here would be inventing a second one.

### One Step 5 headline does not reproduce, and it should not have

Step 5 reported `t0` coming **off the `MIN_WALL_MM` floor** for the first time in the
project (1.2000 → 1.2714). At step 82, `t0` is **1.2000, pinned low again**. The Step 5
observation was real but its stated cause was too general: a thicker wall buys buildable
fillet *only when `R_hub` cannot move*. Give the radius its own room and the wheel stops
paying for fillet in wall thickness, which is the cheaper trade and the expected one. The
Step 5 row above is left as written, with this correction attached rather than folded in.

### Step 6b — export

`make export EXPORT_GENOME=stage3_buildcap2_feasible_medium.json`, 59.7 s:

| condition | result | |
|---|---|---|
| fillets built | **24/24 hub, 24/24 rim, at the full requested radius** | PASS |
| `kt_error_pct` within ±5% | **+0.0% / +0.0%** | PASS |
| as-built hub utilisation < 1.0 | **0.9865** (priced 0.99998) | PASS |
| OCC valid, single solid | `True`, 1 solid, not self-intersecting | PASS |
| Inventor-clean | max tol 1.0e-07, degenerate 0, min edge 0.0156 mm | PASS |
| min curvature R | 0.4597 mm against a 0.25 floor | PASS |
| junction bite | hub 0.627 t, rim 1.620 t, floor 0.25 t | PASS |

Worst hub wedge is **314.0°** — the NEAR-CUSP family, so the cap's slot branch is what is
binding on the shipped candidate, not the arrival branch the arc was built around. The
arrival branch did its work getting here; the parked slot-arrival law (§"Parked") is
therefore the live successor, not a nice-to-have.

**Every Step 5b and Step 6b condition passes.** Promotion followed in Step 6c below.

---

## Step 6c — Promotion, and what the regression net said about it

Promoted. `best_solution.json` is genome **`e4219f3`**, step 71 of the Step 5b run.

### Step 82 was the wrong iterate, and the test net is what caught it

Step 82 was promoted first, on the rule "lowest-loss iterate with every barrier exactly 0.0."
That rule is right and the application of it was not: it never asked *how* feasible. Measured
after the fact, at `coarse`:

| | R_hub | cap | slack | `run_closed_form` |
|---|---|---|---|---|
| step 82 | 0.459736 | 0.459643 | **−93 nm (violating)** | 2.705e-06, **FAIL** |
| step 71 | 0.457112 | 0.460062 | **+2951 nm** | 0.000e+00, PASS |
| step 63 | 0.461312 | 0.462189 | +877 nm | 0.000e+00, PASS |
| step 58 | 0.459593 | 0.476952 | +17359 nm | 0.000e+00, PASS |

**Step 82's feasibility is fidelity-dependent** — feasible at `medium` by 53 nm, infeasible at
`coarse` by 93 nm. Sitting on the knee of `max(0, v)**2`, whose second derivative is
discontinuous at `v = 0`, makes a central difference straddle the kink, so the analytic
gradient and the FD estimate disagree by 2.7× the 1e-6 tolerance. The G4 gate was not
reporting a code bug; it was reporting **where the design sits**.

Step 71 costs +0.13% of loss and +0.048 g, and buys 60× the slack, a *better* deflection
(−0.002% against +0.051%) and **more** stress margin (0.99639 against 0.99998). Step 82 was a
false economy on a loss that is 99.8% mass. **The rule is now: feasible WITH SLACK AT EVERY
FIDELITY, not feasible at the one rung the descent happened to run on.**

The reasoning that produced step 82 was: the cap is already fitted 1.45–11.61% under OCC truth,
so padding the numerical slack would be inventing a second margin. That is correct about
*buildability* — OCC built step 82 24/24 at +0.0% — and irrelevant to *numerical* robustness,
which is a separate failure mode and the one that actually bit.

### Export (step 71)

24/24 at both junctions at the full requested radius, `kt_error_pct` **+0.0% / +0.0%**, OCC
valid, single solid, min curvature 0.4571 mm against a 0.25 floor, junction bite hub 0.629 t /
rim 1.606 t. Worst hub wedge **314.0°**, NEAR-CUSP, as in Step 6b.

### The regression net: 2 red before, and what changed

The first `make test` after promoting read **11 failed / 425 passed**. Four causes, and only
one of them was a design problem:

1. **Four were procedural.** The genome was promoted without re-exporting, so the manifest
   still described `350f4c7`. The tests say so verbatim — *"re-export before reading anything
   else in this file."* `make export` cleared all four. **The lesson is ordering: promote,
   export, THEN test.**
2. **One is a gene-box casualty, genome-independent.**
   `test_the_beam_to_wheel_ratio_is_not_a_constant` reads **4.943223** at the old 0.5 floor —
   matching §14's documented 4.943 — and **2.412764** at 0.4, *identically for both genomes*,
   exactly as its own docstring predicts ("a property of the gene box, not of `genes`"). It
   would have gone red at Step 5b whichever design shipped. **`make test` should have been run
   at Step 5b, before promoting; running it only afterwards bundled a box change and a genome
   change into one red suite.** Gate 1's actual conclusion,
   `correction_factor_is_defensible == False`, holds in both boxes; only the `> 3.0` margin,
   calibrated in the 0.5 box, moved. Not re-tuned — the docstring is explicit that re-deriving
   it is a judgement about Gate 1, not a test edit.
3. **One was step 82 on the barrier knee**, fixed by step 71 as above.
4. **The rest are genuine consequences of shipping a different design** and are recorded
   below rather than adjusted.

### Where the suite finished

**6 failed / 430 passed in 1348.36 s**, against Step 0's 2 failed / 433 passed. The arithmetic:
436 collected both times (434 + 2 at Step 4, +1 for the arc's new cap test = 436, unchanged
here). Four went green with `make export`; one went green with step 71; one was a test bug of
this arc's own making, fixed; **four are new and real, and the two pre-existing reds both moved
further out.** No test was deleted, skipped, xfailed, or had a threshold re-tuned.

### The characterisation tests that are red, with numbers

None of these is a test bug and none is a correctness gate; they pin findings about the design
space and are written to fail loudly when the premise moves. Re-deriving each is real work.

| test | old `350f4c7` | new | gate |
|---|---|---|---|
| `self_intersection_margin_detects_a_fold` | −7.064 | **+0.282** | must be < 0 |
| `correction_is_not_a_constant_over_the_design_space` | 3.383 | **1.542** | must be > 3.0 |
| `correction_enters_at_first_order_in_the_load` | 0.00205 | **0.00258** | must be < 0.001 |
| `rim_band_holds_a_large_minority_of_the_compliance` (hub share) | 0.0321 | **0.0508** | must be < 0.03 |
| `a_thicker_rim_monotonically_stiffens_the_wheel` (drop @ rim 49.7) | 2.2496 | **1.637** | must straddle 2.0 |

The last two were already red before this arc; both moved further out. Three notes on cause:

- The **fold detector** is not broken. It builds its positive case by inflating the *shipped*
  spine to a 40 mm band against an assumed ~11 mm curvature radius. The new spine is
  straighter — healthy margin 11.94 → 19.28, min curvature radius 12.26 mm — so 40 mm no
  longer folds it, by 0.28 mm. The fix is a thicker constructed band or a fold built
  independently of the shipped genome; both are test edits and neither was made.
- The **hub compliance share** is the one real physical cost: a 0.457 mm hub fillet is more
  compliant than a 0.579 mm one, so the hub's share of strain energy rose 0.0321 → 0.0508.
  This is the price of the buildable radius and it is not recoverable by iterate choice.
- The **rim straddle** shows the new design is markedly more mesh-sensitive. The two genomes
  agree at `medium` (1.99996 vs 1.99923 mm) and differ by **27%** at `smoke` (1.637 vs 2.2496
  at rim_outer 49.7), because every thickness gene now sits on the 1.2 mm floor and a coarse
  mesh resolves a thin wall badly. Worth knowing before trusting any `smoke`-rung number on
  this genome.

### The arc's own new test had the fuse its docstring warns about

Promoting step 71 cleared G4 and immediately reddened
`test_the_cap_ranks_two_designs_OCC_disagrees_about` — **the test this arc exists to add**, on
its own guard:

```
assert float(g[8]) == 1.2, "these thresholds were measured at t0 = 1.2"
```

Step 71's `t0` is **1.20084257**, 0.07% off the wall floor, and the OCC thresholds it compares
against (0.5847 mm at 20°, 0.4127 at 50°) were bisected at exactly 1.2. The guard was right to
fire. What was wrong is that it could: the docstring argues at length that the two genomes are
**constructed, not loaded**, because "a test that pins a calibration to a file some future run
overwrites is a test with a fuse in it" — and then constructed the arrival angle by rotating P1
while inheriting `t0` from `best_solution.json`. That is precisely the fuse, left in by the
same test that names it. `by_thickness` is **linear in `t0`**, so an inherited value silently
rescales the quantity under comparison.

Fixed by completing the construction — `g[8] = 1.2` alongside `g[0]` and `g[1]` — which makes
the guard the assertion it was always meant to be. **This is a test bug fix, not a threshold
re-tune**, and it is the only test edit made in Step 6c.

### Two errors made during this diagnosis, recorded because the reasoning was wrong twice

The G4 failure was first attributed to the numpy/jnp `scaled_jacobian` identity. That sub-gate
reads **0.000e+00 on both genomes**; the failing row was `t1/fillet_cap`. It was then
attributed to the `min(by_slot, by_thickness)` crossover kink. The branches are **253% apart**
(slot 1.622 vs thickness 0.460) and thickness binds decisively — also wrong. The cause was
found only by measuring the cap slack directly. Both wrong guesses were plausible and neither
survived a measurement, which is the argument for measuring first.

---

## Step 7 — Write the record into PLAN.md as §16

Same shape as §15. And whatever the outcome, state plainly: **the objective has priced a
buildable hub fillet since §5, and from 2026-08-06 to 2026-08-10 it priced it with a constant
fitted an octave away from the floor every shipped design sits on.**

---

## Step 8 — Defect 6, and a measurement that demotes §16's own top successor (2026-08-12)

Not part of the seven-step arc; done after it closed, and it changes what §16 says to do next.

### The selection rule

`wheel_objective.BARRIER_TERMS` / `OBJECTIVE_TERMS` split the weight table by what a term
answers — *may this ship* versus *how good is it* — asserted complete against `TERMS` at
import so a new term cannot default into "harmless". `wheel_stage3.selection_key` ranks three
tiers (feasible with slack / knife edge / violating), used identically within a run and across
multi-start runs, with the tier in the banner, in the run record, and in the `--best-out`
provenance block. `MIN_CAP_SLACK_MM = 1e-3` is defect 7 operationalised.

Replayed against the real 101-step trace, now `test_the_rule_reranks_the_run_that_exposed_it`:

| iterate | loss | cap slack | old rule | new rule |
|---|---|---|---|---|
| 93 | **30.8914** (min) | **−1.045 µm** | **reported** | tier 2 |
| 82 | 30.9008 | +0.052 µm | promoted off it | tier 1 |
| 75 | 30.9406 | +10.936 µm | — | **tier 0, selected** |
| 71 | 30.9421 | +3.096 µm | — | tier 0 (shipped) |

53 of 101 are tier 0. Four new tests in `tests/test_stage3.py`; the file is green at 55 passed.

### The new rule picks step 75 and step 75 does not ship

Lower loss, worse everywhere: −12 mg bought with +0.110% deflection error (against 71's
−0.002%), 0.9988 utilisation against 0.9964, and a hub fillet 6 µm smaller. `best_solution.json`
stays `e4219f3` — nothing re-promoted, re-exported or re-scored. The selection fix removed the
barrier/objective confusion and left the objective's **own** indifference to margin exposed,
which is the argument for the stress-margin term stated in numbers rather than in principle.

### §16 ranked the slot arrival law #1; it cannot pay

§16 said the slot branch "is what binds on the wheel that ships (wedge 314.0°, NEAR-CUSP)".
That conflated the **wedge family of the worst OCC corner** (near-cusp, true) with the
**analytic branch that sets the cap** (thickness, 0.4601 mm against slot's 1.6221 mm — 253%
from binding). Granting the entire re-fit — measured near-cusp share 0.60 at the shipped
arrival against the modelled 0.30 — the slot branch moves to 3.2476 mm, **7× above the binding
branch**, and the cap does not move at all.

The branch that does bind is already tight: 1.5–3.4% conservative across 5–60° at `t0` = 1.2,
3.2% at the shipped arrival of 41.748°, never over-promising. **No fillet radius is left on the
binding branch.**

Also: step 4's `≥1.44` censoring note, correct for computing a cap, is disqualifying for
fitting the slot branch — the law would rest on `elite13` alone. Re-run the sweep with a raised
bracket before attempting it.

### Successors, re-ranked

1. The stress-margin term. 2. Re-derive the five characterisation gates. 3. The slot arrival
law, demoted from #1 and re-scoped as correctness rather than payoff.

---

## Parked, on purpose

- **PLAN.md §15 successors 2–5** — the stress-margin term, the wall floor under SVK, the
  mesh-convergence study on `axle_drop_mean_mm`, and the load-control tolerance. Successor 2
  is the closest of the four to this arc: it is the *other* reason `R_hub` has no gradient,
  and if Step 5's gate fails on the "`R_hub` must have moved" clause, that is the reason.
  It is deliberately not bundled — one arc, one mechanism.
- **PLAN.md §14 items 4b and 6** — the hub compliance share and
  `EMBED_ALLOWANCE_PER_SPOKE_MM2`. Untouched by this arc.
- **The rim junction.** Every measurement above is the hub, because the hub is where the
  +11.9% is. `R_rim` is equally dead and the rim has no cap model at all. If Step 2's
  predictor is geometric rather than hub-specific it should apply there too — **check, do not
  assume**, and if it does not, say so rather than generalising quietly.
