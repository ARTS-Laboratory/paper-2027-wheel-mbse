# RIMCAP_PLAN.md — a rim cap model

**Open arc #5. Created 2026-08-16, carried forward from PLAN §22 and §24. Nothing started.**

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
