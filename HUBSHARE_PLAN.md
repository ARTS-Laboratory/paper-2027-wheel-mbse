# HUBSHARE_PLAN.md — should hub compliance be an objective term?

**~~Open arc #3.~~ Created 2026-08-16 from PLAN §31 item 4's filed successor. ~~Nothing
started.~~**

**STATUS CORRECTED 2026-09-05 — PLAN §114. THIS ARC IS CLOSED — 2026-09-04, §109.** The
answer is NO: the gate is green, the bound is `0.0117`, and there is no deficit for an
objective term to close. The closing record is at the foot of this file — three lines below a
header that claimed nothing had started, which is how long that contradiction stood.

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

§31 measured the hub compliance share in full, decided the threshold question, and filed the
design question here. **The threshold is settled and is not what this arc is about.**

Settled in §31, do not reopen without new evidence:

- ~~`compliance_split["hub"] < 0.03` **stays at 0.03**, and the assertion stays a strict
  `xfail`.~~ **SUPERSEDED 2026-09-04 — PLAN §109, and this arc's closing record at the foot of
  this file.** §31's call rested on two legs and both are gone: the deficit was a mesh
  artefact after all, and `best_solution_ga_beam.json` — the design the bound was calibrated
  on — is REFUSED by the filleted mesh, so leg (A) has no referent there. The gate reads the
  filleted build against `< 0.0117`, passes, and is renamed
  `tests/test_wheel_fea.py::test_the_hub_junction_holds_a_small_minority_of_the_compliance`.
- ~~§14's hypothesis — that the share rose because `R_hub` fell — is **structurally
  impossible**. Fillets are not meshed, so the wheel is bit-identical across the whole
  `R_hub` box.~~ **REOPENED 2026-08-24 BY NEW EVIDENCE — PLAN §75, FILLET_PLAN STEP 3 RECORD
  PART 1.** The second sentence is true and is the reason the first does not follow: a mesh
  with no fillets cannot express this hypothesis, so a bit-identical sweep is the instrument
  reporting its own blindness rather than the wheel reporting indifference. On a FILLETED
  mesh at `coarse` under SVK the sweep gives eleven distinct values of fourteen rows, and
  the hub share runs 0.007755 -> 0.003703 as `R_hub` goes 0.400 -> 1.900 — **it rises as
  `R_hub` falls, monotonically, which is §14's direction.** This bullet said "do not reopen
  without new evidence"; this is the new evidence.
- The bound is neither unreachable nor a mesh artefact: `best_solution_ga_beam.json` meets it
  **converged**, 0.0139–0.0143 across five rungs, with 53% to spare.

**What is open is the design question:** the shipped wheel holds **3.2× the hub compliance
share of the design it replaced**, and §31 identified the lever but did not pull it.

## The measurement already in hand — `make reds-hub`, ~40 s

One-at-a-time gene swaps from the shipped genome toward `best_solution_ga_beam.json` at
`coarse` (hub share 0.0417 → 0.0138):

| gene | shipped | ga_beam | hub | closes the gap by |
|---|---|---|---|---|
| `cy4` | 6.4375 | 29.2919 | **0.0132** | **102.4%** |
| `cy3` | 9.4191 | 24.3248 | 0.0219 | 70.9% |
| `cy1` | 8.7212 | 27.9529 | 0.0250 | 59.7% |
| `cy2` | 11.8088 | 31.7187 | 0.0255 | 57.9% |
| `t0` | 1.4738 | 2.4774 | 0.0561 | **−51.9%** |

**`cy4` alone takes it under the bound.** The shipped spoke is much flatter (cy 6.4–11.8
against 24–32), and a flatter spoke feeds moment into the hub junction instead of storing it
in its own bending. The thickness genes push the *other* way — a thicker root raises the hub
share — which is why §14's instinct about thickness looked wrong when it was right in sign and
simply swamped.

Design × mesh, five rungs:

| genome | smoke | coarse | medium | fine | ultra | drift |
|---|---|---|---|---|---|---|
| shipped | 0.0392 | 0.0417 | 0.0433 | 0.0453 | 0.0463 | **+18.3%** |
| ga_beam | 0.0139 | 0.0138 | 0.0139 | 0.0141 | 0.0143 | +2.4% |

## THE FIRST QUESTION, AND IT MAY CLOSE THE ARC FOR FREE

**Is a high hub compliance share actually bad?** Nothing in this project has established that
it is. It is a share of a partition that sums to 1 — the hub holding 4.6% instead of 1.4%
means the spokes and rim hold correspondingly less, and the wheel's *deflection* and *stress*
targets are both already constrained directly. A bound on an energy share is a proxy, and the
tree has been sharp elsewhere about proxies that outlive their justification (§28, §29, §31).

**So step 0 is not "add the term". It is "find out whether the term would buy anything".**

If the answer is that the share is a diagnostic rather than a constraint, this arc closes by
recording that, and §31's `xfail` becomes a characterisation pin rather than a deficit — which
is a *better* outcome than adding a term.

**And check `FILLET_PLAN.md` first.** The shipped genome's share does not converge (0.0392 →
0.0463, still climbing at the finest rung) while the ga_beam design's does. The unfilleted
re-entrant corner is the prime suspect, and if the fillet arc lands first this number may move
on its own. Do not build an objective term on a quantity that is not converged.

## THE PLAN

### Step 0 — does the share predict anything the objective does not already see?

Cheap, and it decides whether the rest of the arc is worth running. Over the elite pool
(`stage3_prod_elite9.json`, `stage3_prod_elite10.json`, `stage2_elites.json`):

- correlate hub share against **axle drop** and against **utilisation**. If it is strongly
  correlated with quantities the objective already constrains, a term on it is redundant.
- correlate it against anything the objective *cannot* see. That is the only case where it
  earns a place.

**Register what would justify a term BEFORE measuring**, per §14's standing rule.

### Step 1 — only if Step 0 justifies it: is the term reachable and cheap?

`cy4` is gene 7 and already in the box, so no gene-space change is needed. The term would go
in `wheel_objective`. Two things to check before writing it:

1. **Is it differentiable through the existing adjoint?** `wheel_adjoint` computes the
   gradient; a compliance-split term is an energy ratio and needs to survive `jax.grad`.
   `study_gradient.py`'s finite-difference check is the gate.
2. **What does it cost the other targets?** `cy4` at the ga_beam value takes the hub share
   under the bound, but it is a *large* move in a gene the deflection target also depends on.
   Price the trade before proposing it — a term that buys 3% of compliance share for grams or
   for deflection margin is not obviously worth having.

### Step 2 — write it up and put the trade in front of the user with numbers

Same discipline as §31: measure, then hand over. If the term costs mass or deflection margin,
that is a Pareto choice and it belongs on PLAN.md's "The decision that is a human's" list.

## What must NOT happen

- ~~**The `< 0.03` bound is not moved.** Settled in §31 with the ga_beam control as the
  evidence; this arc is about the design, not the threshold.~~ **LIFTED 2026-09-04 — PLAN
  §109.** It was moved, to `0.0117`, and the clause above is why it took a section to do:
  the ga_beam control this prohibition names is the very thing that stopped existing on the
  mesh the objective solves. The prohibition it protected against — re-fitting a gate to the
  design under test — is intact and was applied: the new bound is rescaled by the REFERENCE
  design's mesh factor (2.550x), never by the shipped genome's (4.115x).
- **No objective term is added before Step 0 shows it buys something.** Adding a term to make
  a red test green is exactly the move this tree keeps refusing.
- **Do not build on a non-converged quantity.** Check `FILLET_PLAN.md`'s status first.
- **`best_solution.json` is not re-descended and promoted inside this arc.**

---

## RE-MEASURED ON THE FAITHFUL MESH — 2026-08-19. THE PREMISE SURVIVES.

PLAN §38 flipped `wheel_wheel`'s `uncap` default to `(True, 1.0)`, which replaced the half
end cap at the spoke/ring junctions with the far flank's own continuation. That moved hub
compliance **−17.9% at `coarse` and −18.7% at `medium`** — the same junction this arc's
suspect lives at, and the same order as the drift the arc calls disqualifying. So the
obvious hypothesis was that the end cap WAS the non-convergence.

**It is not.** `make reds-hub` re-run on the shipped default (45.3 s,
`studies/study_reds_hub_share_UNCAPPED.json`), against the capped table above:

| rung | capped | faithful | change |
|---|---|---|---|
| smoke | 0.0392 | 0.0325 | −17.1% |
| coarse | 0.0417 | 0.0342 | −18.0% |
| medium | 0.0433 | 0.0352 | −18.7% |
| fine | 0.0453 | 0.0365 | −19.4% |
| ultra | 0.0463 | **0.0371** | −19.9% |
| **drift smoke→ultra** | **+18.3%** | **+14.05%** | |

The artefact was a **near-constant multiplicative offset of ~18–20%, not the drift**. The
share still climbs monotonically across all five rungs and is *still climbing at `ultra`*.
Removing ~19% of the level removed only 4.3 points of the 18.3% drift, which is what a
constant offset does to a ratio and no more.

**So "do not build on a non-converged quantity" stands, and `FILLET_PLAN.md` is still the
gate on this arc.** The unfilleted re-entrant corner remains the prime suspect precisely
because the one OTHER junction artefact has now been removed and the drift barely moved.
That is a stronger reason to wait for the fillet arc than the arc had before, not a weaker
one: the field of suspects narrowed by one and the symptom did not.

### What DID change, and what it does not change

- **The gate deficit shrank without closing.** `hub < 0.03`: over by 39% (capped `coarse`)
  → **14.0%** (faithful `coarse`); but **23.5% at `ultra`**, because the drift is intact.
  `tests/test_wheel_fea.py::test_the_hub_junction_holds_under_three_percent_of_the_compliance`
  stays an `xfail` and still fails — which, under `xfail_strict = true`, is the outcome that
  keeps the suite green. **Nothing here reopens the `< 0.03` bound.**
- ~~**§14's hypothesis is killed harder than before.** The `R_hub` sweep is now
  *bit-identical* — 0.0342 at every one of the 14 sample points across the whole 0.4–4.0
  box, feasible and infeasible alike. The plan above says "structurally impossible, fillets
  are not meshed"; the faithful mesh shows it to the last digit. (The driver still prints
  the canned line "the hub share FALLS as `R_hub` falls", which is false when the column is
  constant. Cosmetic, in `studies/study_reds_hub_share.py`, and deliberately not touched
  here.)~~

  **RETRACTED 2026-08-24 — PLAN §75.** A bit-identical column is not a sharper kill, it is
  the absence of a test: the sweep moves a gene the mesh does not read. **And the parenthesis
  was the tell.** The canned line was noticed, correctly diagnosed as false, and filed as
  *cosmetic* — but a verdict function that reports a falsification off an exact tie is not a
  formatting bug, it is the only evidence the bullet had. It now has a third branch naming
  the reason. On a filleted mesh the hypothesis survives; see the reopened bullet above.
- **The attribution re-ranked, and `cy1`/`cy2` swapped.** One-at-a-time swaps at `coarse`:

  | gene | capped closes | faithful closes |
  |---|---|---|
  | `cy4` | 102.4% | **114.9%** |
  | `cy3` | 70.9% | 81.1% |
  | `cy2` | 57.9% | **65.5%** |
  | `cy1` | **59.7%** | 46.7% |
  | `t0` | −51.9% | −46.7% |

  `cy4` alone still overshoots the bound, so Step 1's "no gene-space change is needed"
  holds. But **the capped table's ordering was not safe to quote**: `cy1` lost 13 points and
  fell below `cy2`. Any future statement of the form "the second/third strongest lever" has
  to be re-read off the faithful table.

**Step 0 is still not started, and this does not start it.** This measurement only refreshes
the inputs Step 0 would use and re-checks the prohibition that gates the whole arc. The
design question — whether a high hub share is actually bad — is untouched and still open.

---

## RE-MEASURED ON THE FILLETED MESH — 2026-09-03. **THE GATE IS CLEARED AND THE ARC'S OWN QUESTION IS ANSWERED.**

PLAN.md §106.  This arc gates itself in three places on one event — *"check
`FILLET_PLAN.md`'s status first"*, *"the unfilleted re-entrant corner is the prime
suspect"*, *"if the fillet arc lands first this number may move on its own"*.  **The event
happened at §103 and nothing read this file.**  No driver could have answered it either:
`study_reds_hub_share.rungs()` takes no `fillet` argument, so every rung it has ever
measured took `build_wheel`'s unfilleted default, and `make reds-hub-fillet` runs `--sweep`,
not `--rungs`.  Measured by calling the module's own `shares()` directly, which is what
`rungs()` calls:

| genome | mesh | smoke | coarse | medium | fine | ultra | drift | vs 0.03 |
|---|---|---|---|---|---|---|---|---|
| shipped | plain | 0.032489 | 0.034188 | 0.035237 | 0.036483 | 0.037053 | +14.05% | **OVER by 23.5%** |
| shipped | FILLETED | 0.008079 | 0.008308 | 0.008409 | 0.008442 | 0.008463 | +4.75% | **UNDER by 71.8%** |
| ga_beam | plain | 0.013823 | 0.013722 | 0.013849 | 0.014043 | — | +1.60% | UNDER by 53.2% |
| ga_beam | FILLETED | 0.005312 | 0.005381 | 0.005406 | 0.005417 | — | +1.98% | UNDER by 81.9% |

**[CORRECTED 2026-09-04 — PLAN §109. The two `—` cells are not missing data.** `ga_beam` at
`ultra` is **0.014144** and was already committed in this driver's `rungs` block, so the
table read shipped drift over FIVE rungs against ga_beam drift over FOUR; with `ultra` in it
is **+2.32%**, not +1.60%. The filleted cell is now measured at **0.005423**, drift **+2.09%**.
Shipped filleted drift is **+4.74%** and the tail's slowest increment ratio is **1.640**, both
misread off this table's own rounded cells. Every measured value reproduces and no verdict
moves; see the closing record at the foot of this file.]**

**The plain column reproduces this arc's committed `rungs` block to 3.3e-7 at all five
rungs**, returns its stated +14.05% drift, and returns its stated 23.5% deficit at `ultra` —
three matches, so the filleted column is the same instrument on a different mesh.

**CONVERGENCE IS SETTLED BY THE INCREMENTS, NOT THE DRIFT.**
Plain: `+0.001699, +0.001049, +0.001246, +0.000570` — **not monotone**, rising at
`medium`->`fine`, which is why this quantity could never be called converged.
Filleted: `+0.000229, +0.000101, +0.000033, +0.000021` — **falling at every rung**.
Richardson on the filleted tail at its slowest observed ratio (1.57) puts the limit at
~0.00850, so `ultra` is converged to ~0.4%, and every rung from `smoke` up is already inside
the bound by more than 70%.

**So all three gating sentences resolve, and the prime suspect this file named was right:**
the unfilleted re-entrant corner was not merely a cause of the drift, it was 76% of the
LEVEL as well (0.036483 -> 0.008442 at `fine`, a 4.32x reduction).

**AND THAT ANSWERS THE DESIGN QUESTION, NOT JUST THE GATE.**  Step 0 asks *"find out whether
the term would buy anything"*.  On the mesh the objective now solves the shipped wheel holds
0.008442 against a 0.03 bound — **72% of margin** — so there is no deficit left for a term
to close.  This file already calls that the better outcome: *"If the answer is that the
share is a diagnostic rather than a constraint, this arc closes by recording that ... which
is a better outcome than adding a term."*

**SCOPE, BECAUSE IT BOUNDS A COMPARISON.**  `ga_beam` asks `R_hub` 1.5598 / `R_rim` 3.0 and
the mesh applies (0.667, 0.8952) with BOTH junctions clamped; the shipped genome is
unclamped.  The shipped-to-ga_beam ratio on the filleted mesh — 1.56x — therefore compares a
clamped build against an unclamped one and **is not a design comparison**.  The shipped
column is clean and carries the finding.

**WHAT IS LEFT IS A JUDGEMENT, AND IT IS THE ONE §31 RESERVED.**
`tests/test_wheel_fea.py::test_the_hub_junction_holds_under_three_percent_of_the_compliance`
is `xfail(strict=True)` and says *"this reopens itself the day the wheel passes it."*  The
wheel passes it — on the mesh the objective solves — and the test does not reopen, because
its `mesh` fixture is `ww.build_wheel(genes, CFG)`, the bare default.  It is still red,
still green as an xfail, and now records a deficit **of the instrument rather than of the
design**.  Deciding what it should assert closes this arc.

---

## CLOSED — 2026-09-04. **THE GATE IS GREEN, THE BOUND IS `0.0117`, AND THE ARC'S DESIGN QUESTION IS ANSWERED NO.**

PLAN.md §109, which is §106's successor 1 and the judgement §108 declined to make as a typo
fix. Three things happened, in order, and only the third was a decision.

**1. THE LADDER EXISTS ON DISK NOW.** §106's filleted column was a live measurement produced
by calling `shares()` by hand, because `rungs()` took no `fillet` argument — so the arc's
conclusion was un-reproducible from any driver. `rungs()` now forwards `fillet` and
`kinematics`, `make reds-hub-fillet-rungs` runs it, and `studies/study_reds_hub_share.json`
carries both arms under `rungs` and `rungs_filleted_linear`. Linear on both sides, so the
only difference between the two ladders is the mesh.

**THE PLAIN ARM REPRODUCES BIT-IDENTICALLY** — all forty values, every digit, zero
difference against the committed block. §106 claimed 3.3e-7 and was quoting a bound rather
than a measurement; the agreement is exact. Every §106 filleted value reproduces to ≤4.2e-7,
which is the rounding of its own six-decimal table.

| genome | mesh | smoke | coarse | medium | fine | ultra | drift | vs bound |
|---|---|---|---|---|---|---|---|---|
| shipped | plain | 0.032489 | 0.034188 | 0.035237 | 0.036483 | 0.037053 | +14.05% | over 0.03 by 23.5% |
| shipped | FILLETED | 0.008079 | 0.008308 | 0.008409 | 0.008442 | 0.008463 | +4.74% | **under 0.0117 by 29.0%** |
| ga_beam | plain | 0.013823 | 0.013722 | 0.013849 | 0.014043 | **0.014144** | **+2.32%** | under 0.03 by 54.3% |
| ga_beam | FILLETED | 0.005312 | 0.005381 | 0.005406 | 0.005417 | **0.005423** | +2.09% | under 0.0117 by 54.0% |

**2. THREE CORRECTIONS TO §106's TABLE, NONE OF WHICH MOVE ITS VERDICT.** All three are the
same fault: a quantity read off the rounded prose table instead of the values behind it.

- **`ga_beam` at `ultra` was marked `--` and 0.014144 was committed on disk the whole time.**
  So the table compared shipped drift over FIVE rungs against ga_beam drift over FOUR. With
  `ultra` in, ga_beam plain drifts **+2.32%**, not +1.60%.
- **Shipped filleted drift is +4.74%, not +4.75%** — 0.008462759/0.008079409 = 1.047448.
  §106 divided its own 6-dp table.
- **The filleted tail's slowest increment ratio is 1.640, not 1.57** — 1.571 is 33/21 off the
  rounded increments. Richardson at 1.640 puts the limit at 0.008495, converged to 0.375%.

The convergence verdict is unaffected and does not depend on the extrapolation at all: for
the filleted ladder to reach even the OLD 0.03 its increments would have to stop decaying to
a ratio of **1.00095**, against the 1.640 measured.

**3. THE CALL. THE MESH MOVES, THE BOUND MOVES WITH IT, AND THE XFAIL IS GONE.**

§31 made its call on two legs and named them. Both are gone:

- **(B) "not a mesh artefact" is FALSIFIED.** It was one. 76% of the level was the unfilleted
  re-entrant corner (0.036483 → 0.008442 at `fine`, 4.32x).
- **(A) "achievable — ga_beam clears it by 53%" has NO REFERENT.** `ga_beam` asks `R_hub`
  1.5598 / `R_rim` 3.0, and an explicit `fillet=(1.5598, 3.0)` is **refused**: *"the fillet's
  tangent point has passed the next sector's corner (−8.400 deg of free ring left)."* The
  filleted `ga_beam` row above is `fillet=True`, CLAMPED to (0.667, 0.895). **The design the
  bound was calibrated on cannot be built on the mesh the objective solves.**

**AND THE BOUND COULD NOT BE TRANSPORTED BY A FACTOR, WHICH IS THE PART THAT IS NOT OBVIOUS.**
The mesh does not rescale this quantity — it rescales it **differently per design**. At
`coarse` the shipped genome falls **4.115x** and `ga_beam` falls **2.550x**, because the
unfilleted corner penalises a thin hub junction far harder than a thick one (§30's mechanism
on the rim corner, again). There is no single factor for `0.03` to ride across, which is why
"point it at the filleted mesh" was never a one-line change.

**`0.0117` PRESERVES §31's WARRANT INSTEAD OF §31's NUMBER.** What made `0.03` defensible was
leg (A) — the reference design cleared it by 54.3%. So rescale by the REFERENCE design's own
factor and never by the design under test, which is §14's rule applied rather than broken:

```
0.03 * (0.005381416728939758 / 0.013722004451848286) = 0.011765...   ->  0.0117
```

rounded DOWN so the gate is never looser than its derivation. `ga_beam` keeps **54.0%** of
margin against the 54.3% it had; the shipped genome clears by **29.0%**, not the 72% that
carrying `0.03` across would have shown.

**THE SCOPE LIMIT, WITH ITS DIRECTION.** That factor comes from the CLAMPED `ga_beam`, so
`0.0117` is a bound derived from a stand-in. The stand-in's hub fillet (0.667 mm) lands
within 0.51% of the shipped genome's (0.664 mm); its rim fillet (0.895 mm) does not (3.0 mm).
§14's direction — hub share RISES as `R_hub` FALLS, confirmed by §75 on a mesh that can
express it — says an unclamped `ga_beam` would read lower, so the true factor is smaller and
the honest bound is **tighter** than 0.0117. `0.0117` is the loose end, and the shipped
genome's 29.0% is an upper bound on its own margin.

**STEP 0's ANSWER IS NO, AND THAT CLOSES THE ARC.** *"Find out whether the term would buy
anything."* On the mesh the objective solves the shipped wheel holds 0.008308 at `coarse`,
converged, inside a bound derived without reference to it. There is no deficit for an
objective term to close. This file already called that the better outcome. §31's `cy4` route
is not retired — it is de-prioritised, and it stays filed.

**WHAT DID NOT HAPPEN.** `best_solution.json` is untouched, no threshold but this one moved,
and the wheel did not improve. The mesh stopped putting a singularity where the fillet is.
