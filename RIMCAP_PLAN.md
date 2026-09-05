# RIMCAP_PLAN.md — a rim cap model

**Open arc #5. Created 2026-08-16, carried forward from PLAN §22 and §24. ~~Nothing
started.~~**

**STATUS CORRECTED 2026-09-05 — PLAN §114.** Step 0 was run (`THE 'CHECK FIRST' TRIGGER
FIRED` below — 17 of 38 genomes rim-clamped, 2598 iterates scored), and **both halves of what
this arc was ranked for are now superseded: the arc is PARKED — see the park record at the
foot of this file.**

**VERSION CONTROL IS PART OF THIS PROJECT'S WORKFLOW — CHANGED 2026-08-19.** The rule that
stood here read *"Ignore version control entirely. Do not commit, branch, stage, revert or
otherwise touch git."* **It is superseded.** The rules live in `PLAN.md`'s header block and
**only** there, so they cannot drift across ten files: one commit per finished unit of work
on `feature`, `make test` green first, never while a study driver is mid-write, a study
commit carries its regenerated `.json` and `.jpg`, a promotion is one atomic commit and never
one file — and **commits carry no assistant or tool attribution, no `Co-Authored-By:`
trailer, no session link, no generated-with footer.**

---

## Why this arc exists, and what §24 changed about it

The hub side has a cap model — `wheel_objective.hub_fillet_cap_mm` — built by §16 and
BUILD_PLAN steps 5 and 6, which lets the objective see how much fillet the geometry can
actually accept at the hub. **The rim side has no equivalent.** `R_rim`'s box ceiling of 3.0 is
an untested literal.

§21 promoted raising that ceiling to #1. §22 then measured it and found the opposite:

> **`R_rim`'s CEILING IS A TRAP, NOT A PRIZE. The bound binds; raising it would harvest loss
> the part does not pay for.**

§24 then narrowed what a cap model is *for*, and this is the correction that matters:

> "A cliff that sharp, sitting exactly on a round number, is a geometric coincidence on **one
> genome**, not a manufacturing limit — and OCC accepts the rim fillet to within 0.4% of 4.0,
> which is **33% above the ceiling of 3.0**. So buildability does **not** justify that ceiling.
> The mass does, on its own. §22's successor #3 — a rim cap model — is therefore still wanted,
> but for the reason §16 wanted the hub's: **to know the boundary as a function of `t3` and the
> rim arrival angle rather than at one design.** What it is *not* is the thing standing between
> `R_rim` and 4.0."

**So the deliverable is a function, not a number.** Anyone who runs this arc hoping to unlock
`R_rim` = 4.0 has misread it; §22 already measured that the part does not want it.

## What is already known

- OCC accepts a rim fillet to within **0.4% of 4.0 mm** on the shipped genome — 33% above the
  box ceiling of 3.0. Buildability is not the binding constraint at the rim.
- The **mass** is what makes a large `R_rim` unattractive, independently.
- The hub-side precedent is `wheel_objective.hub_fillet_cap_mm`, and `wheel_wheel.py` around
  line 402 documents the void the hub cap is built on — *"simultaneously the slot a fillet has
  to grow into and — when it goes NEGATIVE —"* — with the explicit note that the fillet MODEL
  lives in `wheel_objective`, **deliberately not in `wheel_wheel`**, which "states no fillet
  model". Follow that separation.

## Check first — this arc may be reshaped by `FILLET_PLAN.md`

The hub cap model exists because the objective needed to price a fillet the **mesh cannot
see**. If `FILLET_PLAN.md` lands and the FEA meshes fillets directly, the reason for a
closed-form cap model changes substantially — the objective could price `R_rim` through the
solve instead of through a correlation. **Read `FILLET_PLAN.md`'s status before starting.**

## THE PLAN

### Step 0 — establish the boundary as a function, on a grid

Sweep `t3` × rim arrival angle and record, for each cell, the largest rim fillet the geometry
accepts. Two things must be recorded per cell and not conflated:

- what the **closed-form slot** allows (the `wheel_wheel` void calculation), and
- what **OCC actually accepts** when the solid is built.

§24's finding is precisely that these two disagreed on one genome and the disagreement was read
as a manufacturing limit. Reporting both is what prevents repeating that.

### Step 1 — fit the cap, mirroring the hub

Produce `rim_fillet_cap_mm(t3, arrival_angle)` in `wheel_objective`, alongside
`hub_fillet_cap_mm`. It must be differentiable — `wheel_adjoint` needs it — and it must be
validated by finite difference through `study_gradient.py` the same way the hub's was.

### Step 2 — does it bind?

**Apply this tree's standing test before ranking any further work on it:** measure how far the
shipped design sits from the new cap. §31's memory of this project is that a successor ranked
#1 was worth zero because the branch it fixed was 253% from binding.

If the rim cap does not bind at the shipped design, record that and stop. The model is still
worth having as a guard, but it does not justify further work.

### Step 3 — write up

A PLAN.md section with the grid, both boundaries, the fit, and the binding check.

## What must NOT happen

- **Do not raise `R_rim`'s ceiling as part of this arc.** §22 measured that the bound binds and
  that raising it harvests loss the part does not pay for. That is a separate decision with its
  own evidence, and this arc does not supply new evidence for it.
- **Do not put the fillet model in `wheel_wheel`.** That module states no fillet model, by
  design; the hub's lives in `wheel_objective` and the rim's belongs beside it.
- **Do not calibrate the cap on one genome.** That is the exact error §24 corrected.

---

## THE "CHECK FIRST" TRIGGER FIRED — 2026-09-03. **FILLET_PLAN LANDED AT §103, AND THIS ARC'S DELIVERABLE IS NARROWER AND ITS GAP IS LIVE.**

PLAN.md §106.  This file's own tripwire — *"If `FILLET_PLAN.md` lands and the FEA meshes
fillets directly, the reason for a closed-form cap model changes substantially ... Read
`FILLET_PLAN.md`'s status before starting"* — became true on 2026-09-03 and nothing read it,
because a parked arc is a file nobody opens.  Every clause of it now holds:
`wheel_objective.phase_meshes` passes `fillet=True` unconditionally (`:1015`), `util_j` is the
junction's own region p-norm with `Kt` absent, and §102 gave `R_rim` a nonzero gradient entry
for the first time.

**SO THE HALF THIS ARC WAS RANKED FOR IS GONE.**  Step 1 asks for `rim_fillet_cap_mm` in
`wheel_objective`, differentiable, *"mirroring the hub"* — but §103 demoted `Kt`,
`hub_fillet_cap_mm` and `hub_fillet_r_effective` to **reporting only** (`:1261`).  The rim
does not need a stress surrogate; the solve prices it.

**WHAT THE HUB CAP STILL DOES IS THE HALF NOBODY RESTATED, AND THERE ARE THREE LIMITS, NOT
TWO.**  `_fillet_margins`'s own docstring says the first two are *"a SEPARATE mechanism ...
neither subsumes the other"*; the third is in neither and, since §103, runs inside every
objective evaluation:

| limit | asks | reads | lives in |
|---|---|---|---|
| `fillet` barrier (3000) | does a circle of R fit the re-entrant **corner**? | `g[12]` & `g[13]` | `wheel_objective:836` |
| `fillet_cap` (500) + selection tier | does it fit the **slot** between adjacent spokes? | **`g[12]` only** | `wheel_objective:861`, `wheel_stage3:223` |
| `SECTOR_FIT_CLAMP` | does it fit its own **sector**? | both | `wheel_wheel:1527` — mesh, silent |

**STEP 0, RUN: THE GAP BINDS.**  Enumerated over every genome kept on disk (16 GA elites, the
`best` of all 25 `stage3_*.json` runs carrying one, and the two named genomes; 38 unique):
**17 of 38 have the rim fillet clamped**, up to **88.6%** of the requested radius
(`stage2_elites#7`: 2.7868 asked, 0.3184 built).  All 17 have t1 barrier sums of **43.77 to
712.2** against `T1_REJECT` = **1.0e4**, so every one of them reaches a mesh build —
`descend` rejects a trial only above that threshold, deliberately, because *"every barrier is
ALREADY a term in the objective"*.  `best_solution_ga_beam.json` is cut 70.2% at a sum of
453.8.  The shipped genome is unclamped, with corner margins [4.0271, 10.7491] mm.

**AND IT BINDS ON THE PATH, NOT ONLY AT THE END POINTS.**  Those 38 are where runs stopped.
Every Stage-3 iterate on disk was scored too — 25 committed `studies/stage3_*.json` runs,
denormalised to physical genes, **2598 unique iterates**, each built at `coarse` with
`fillet=True`:

```
  REACH a mesh build (t1 sum <= T1_REJECT=10000)    2598  (100.0%)
  REFUSED the filleted build outright                  0
  rim fillet CLAMPED                                 117        hub clamped  187

  run                        clamped  iterates   idx range   max cut
  stage3_minwall_2.2.json         44       126      #5-#56      18.0%
  stage3_prod_elite10.json        43       300      #0-#42      87.0%
  stage3_prod_elite9.json         24       150      #0-#23      82.3%
  stage3_run_elite10.json          4         4      #1-#4       85.7%
  stage3_run_elite9.json           2         2      #1-#2       81.6%
```

**Not one iterate of 2598 is screened out by `T1_REJECT`**, which retires the "unreachable
geometry" defence: the screen never fires on the search path at all.  And the clamp hides the
gene from its own optimiser — over `stage3_prod_elite10`'s first fifteen steps the gene moves
**2.3%** (2.8283 -> 2.7625) while the radius actually built moves **237%** (0.3668 -> 1.2369),
because the applied radius tracks the sector other genes are opening, not `g[13]`.  The barrier
sum falls monotonically (570.7 -> 84.7) throughout, so the descent reads steady progress.
§102 gave `R_rim` its first nonzero gradient entry; for the first 40-odd steps of the elite-10
descents that entry is for a radius the mesh overwrites.

*A note on method, because the mechanism generalises.*  The first pass filtered to strictly
t1-feasible iterates (every barrier exactly 0), assuming the rest unreachable.  That filter
keeps **1195 of 2598** — a plausible 46% — and **0 of the 117 clamps**: the two sets are
disjoint, so it did not bias the count, it deleted the signal and returned a confident null.
The screen the descent actually applies is `t1_barrier_sum > max(T1_REJECT, b_here)`.

**THE NEXT STEP IS NOT THE CAP MODEL.**  The objective already reads `g[13]` for the CORNER;
what is missing is the SECTOR limit, applied silently by the mesh.  Make the clamp VISIBLE
first — report `fillet_clamped` and the applied radii the way `hub_fillet_cap_mm` is already
reported — and decide whether it needs a barrier once a run can show how often it fires.

**AND A STALE DOCSTRING, WITH ITS CONCLUSION INTACT.**  `_fillet_margins` states *"At the
shipped genome the margins are `[+4.647, +0.125]`"* (`src/wheel_objective.py:640`).  Those are
**`ga_beam`'s** margins, measured to four decimals here — the genome that was shipped when the
line was written.  The current shipped genome reads `[4.0271, 10.7491]`, so the rim figure is
off by 86x.  The docstring's ARGUMENT survives (it uses the margins only to show the barrier
is flat there, and 4.0271 is comfortably feasible), but the numbers are one promotion out of
date.

---

# THE PARK — 2026-09-05. **BOTH HALVES OF THIS ARC ARE SUPERSEDED, AND THE SECOND ONE WAS CLOSED AS IMPOSSIBLE RATHER THAN DONE.**

PLAN.md §114. This arc was ranked #5 for two deliverables. Neither is reachable as written,
and they stopped being reachable for different reasons and in different sections.

## HALF ONE — THE CAP MODEL. SUPERSEDED BY §103.

Step 1 asks for `rim_fillet_cap_mm(t3, arrival_angle)` in `wheel_objective`, differentiable,
finite-difference validated, *"mirroring the hub"*. The thing it mirrors no longer exists in
that role: §103 demoted `Kt`, `hub_fillet_cap_mm` and `hub_fillet_r_effective` to **reporting
only**. Verified in the code as it stands, `src/wheel_objective.py:1261-1265`:

> *"`Kt`/`hub_fillet_cap_mm`/`hub_fillet_r_effective` no longer feed `util_j` — kept for
> REPORTING ONLY. They are a purely geometric feasibility question ... unrelated to and
> untouched by which quantity prices stress."*

**A surrogate mirroring a surrogate that was retired is not a deliverable.** The rim does not
need a stress model; the solve prices it directly through the region p-norms.

## HALF TWO — MAKE THE CLAMP VISIBLE. TRIED AT §110, AND IT CANNOT BE DONE THE PRESCRIBED WAY.

The block above ends *"THE NEXT STEP IS NOT THE CAP MODEL ... report `fillet_clamped` and the
applied radii the way `hub_fillet_cap_mm` is already reported."* **That was attempted at §110
(commit `f21ec7d`) and reverted, on a reason worth keeping.**

`hub_fillet_cap_mm` is reportable because it is computed on the corner barrier's own path,
which never touches the mesh. The sector-fit clamp is not: it is discovered inside
`_filleted_gradient_recipe`, which runs only because `mesh_coords` was called — and
`mesh_coords` **raises before either report dict exists**. A clamped evaluation never reaches
a return statement to hang a key on. Built and tested against `ga_beam`, §106's own clamped
witness, the keys came back constant-by-construction — always `False` — and were reverted
rather than shipped.

Confirmed in the tree today: `wheel_stage3.REPORT_KEYS` (`:179-183`) carries `kt_hub`,
`kt_rim`, `hub_fillet_cap_mm` and `r_hub_effective_mm` and **does not carry `fillet_clamped`**,
which exists only as a `Mesh` attribute (`wheel_wheel.py:2583`). What shipped instead is
visibility in the **event record**: `FilletClampRefusedError` (`wheel_wheel.py:1605`) and
`wheel_stage3._reject_kind` classifying it as `clamp_reject` (`:206`, `:216-217`), split out
of `solve_reject` where S5's calibration had been silently absorbing it.

**So the prescription was not merely completed by someone else — it was shown to be
unreachable, and a different mechanism was substituted.** §110's own successor list records
the same conclusion: *"make the rim clamp visible — closed here: visible in the event record,
not the report dict, because the report dict is unreachable from a clamped evaluation."*

## WHAT IS NOT PARKED

Two things, named so they are not lost with the arc:

1. **Whether the clamp needs a BARRIER.** §106 deferred this explicitly to *"once a run can
   show how often it fires"*, and no run can yet: all 25 committed Stage-3 artifacts predate
   §103's fillet switch, so the clamp has never fired on disk. This is **blocked behind the
   Stage-3 re-run** (§113, still #1 tree-wide), not behind anything in this file. When that
   run lands, its `clamp_reject` event count is the measurement, and it is now recorded
   because §110 made the event distinguishable.
2. **Step 0's grid** — the closed-form slot against OCC acceptance — remains unspent. Step 0's
   enumeration ran (17 of 38 genomes rim-clamped, 2598 iterates scored) and is above; the grid
   itself was never built.

## SCOPE, WITH ITS DIRECTION

The gap this arc named is **real and still open** — the objective reads `g[13]` for the corner
and the SECTOR limit is still applied by the mesh rather than priced by the objective. What
changed is that neither of this arc's two routes to closing it survives. If the re-run shows
`clamp_reject` firing often, the answer is a barrier on the sector limit, which is neither
Step 1 nor the report half. **Parking this arc does not park the gap.**

**WHAT DID NOT HAPPEN.** No code was written for this decision, `best_solution.json` is
untouched, no threshold moved. One loose end from the block above IS now closed and is not
part of this decision: the stale `_fillet_margins` docstring quoting `[+4.647, +0.125]` as the
shipped genome's margins was corrected at §112 (`89a370e`) — it now names `ga_beam` as the
genome those belong to and records the current `[4.0271, 10.7491]` alongside.
