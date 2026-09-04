# WALLPIN_PLAN.md — re-derive Gate 1 at the 1.2 mm floor, and drop the beam test's 2.0 mm pin

**Open arc #4. Created 2026-08-16 from PLAN §31's recorded loose end. Small. Nothing started.**

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

`tests/test_wheel_fea.py::test_the_beam_to_wheel_ratio_is_not_a_constant` wraps its measurement
in `wf.set_min_wall(2.0)`. **The only stated reason for that pin was that the `> 3.0` margin
had been calibrated in a 2.0 mm box** — and §31 retired that margin. The pin now has no
rationale left, and it is a global-state mutation inside a test, which the test's own comment
flags as hazardous:

> "Restore unconditionally. `tests/test_stage3.py` takes its bounds in a MODULE-scoped fixture
> that never recomputes, so a floor leaked from here would not merely persist — it would be
> baked in and fail somewhere else entirely."

**The project ships a 1.2 mm design.** `MIN_WALL_MM = 1.2` has been the default since
2026-08-06 (§13). So the test measures a gene box the project does not use, for a reason that
no longer exists.

## What §31 already measured — do not re-derive, re-run to confirm

§31 measured the replacement statistic in **both** boxes rather than move the pin, precisely so
this arc would be cheap. `make reds-ratio` (~3 min, 109 cells) reproduces all of it:

| box | CV over 20 seeds at n=6 | **CV floor** | gate | margin |
|---|---|---|---|---|
| 2.0 mm (the pin) | 0.145 – 0.575 | **0.1450** | 0.14 | 3.6% |
| 1.2 mm (shipped) | 0.195 – 0.613 | **0.1948** | 0.14 | **39%** |

**Dropping the pin changes no verdict, and it makes the gate less tight, not more.** The 1.2
box clears the same 0.14 bound by 39% where the pinned 2.0 box clears it by 3.6%.

`correction_factor_is_defensible` is `False` in **all 66 beam cells across both boxes**, so
Gate 1's conclusion — the Stage-2.5 off-ramp does not exist — is unaffected by the choice.

## WHY THIS IS STILL A JUDGEMENT AND NOT A ONE-LINE EDIT

§14 said, in terms:

> "Re-deriving Gate 1's margin at a 1.2 mm floor is a real piece of work and a judgement about
> Gate 1; it is not a test edit, and it has not been done here."

§31 did the *measurement* half and deliberately left the *judgement* half. The judgement is:
**which gene box is Gate 1 a statement about?** Gate 1's conclusion is "no single beam-to-wheel
correction factor exists over the design space the optimizer searches" — and the optimizer
searches the 1.2 box. On that reading the pin is not merely unjustified, it makes the test
answer a question about a space the GA never visits.

The counter-argument, which should be written down and then accepted or rejected on the
record: Gate 1's historical numbers (§14's 4.943 / 2.686 table) were taken at 2.0, and changing
the box makes the test's value discontinuous with its own history.

## THE PLAN

### Step 0 — confirm

`make reds-ratio`. Gate: the two CV floors above reproduce (0.1450 and 0.1948) and
`correction_factor_is_defensible` is `False` in 66/66 beam cells.

### Step 1 — make the judgement, and record the argument on both sides

Decide which box Gate 1 is a statement about. **Write the argument down before the edit**, in
a PLAN.md section, so the next arc inherits the reasoning and not just the diff.

### Step 2 — if the judgement is "the 1.2 box", the edit

1. Remove the `set_min_wall(2.0)` / `finally: set_min_wall(before)` wrapper from
   `test_the_beam_to_wheel_ratio_is_not_a_constant` **and** from
   `test_the_retired_max_min_gate_is_decided_by_the_sample_size` in the same file.
2. **Re-check the instability pin.** It asserts `small < 3.0 < large` on the max/min ratio at
   n=6 vs n=24. At the 1.2 floor those read **2.686 and 43.587**, which still brackets 3.0 —
   but confirm rather than assume, and update the docstring's measured values.
3. Update both docstrings' measured tables to the 1.2 box, and update `studies/
   study_reds_ratio_stability.py`'s module comment, which currently presents 2.0 as the test's
   box.
4. `make test`. Nothing else should move — but `test_the_free_arc_fraction_is_not_constant_
   over_the_design_space` calls `run_beam_blindness` **without** the pin already, so check it
   is still consistent with whatever the neighbouring tests now do.

### Step 3 — if the judgement is "the 2.0 box", say so where the pin is

Leave the code alone and replace the docstring's "loose end for a human" note with the decided
reason, so the next reader does not re-open it. A pin with a recorded justification is fine; a
pin whose justification was retired is not.

## What must NOT happen

- **The `cv > 0.14` gate is not re-derived to suit whichever box is chosen.** It was derived
  from the floor across both boxes precisely so this choice could not move it. If the 1.2 box
  is adopted the gate stays 0.14 and simply gains margin.
- **`max/min` is not reinstated.** See §31; both test files carry a DO NOT REINTRODUCE block.
- **No test deleted.**

---

## PREMISE CHECKED AGAINST THE FILLET SWITCH — 2026-09-03. **INTACT, AND THE JUDGEMENT GAINS A SECOND DIMENSION.**

PLAN.md §106.  Verified mechanically: `set_min_wall(2.0)` still wraps both named tests
(`tests/test_wheel_fea.py:370`, `:407`) and is restored in a `finally`; `MIN_WALL_MM = 1.2`
(`src/wheel_fea.py:236`); and the measurement path is untouched by §103 — `run_beam_
blindness`'s FEA side is `_blindness_row`, which calls `WW.build_wheel(v, cfg)` with **no
`fillet=` argument** (`studies/study_wheel_fea.py:465`), the unfilleted default.  §31's CV
floors and the 66/66 `correction_factor_is_defensible` result still reproduce from their
driver.  **Nothing in this arc's evidence moved.**

**WHAT MOVED IS THE SUBJECT.**  Gate 1's conclusion is *"no single beam-to-wheel correction
factor exists over the design space the optimizer searches"*, and since §103 the optimizer's
wheel is the FILLETED one, which §91 measured as disagreeing with the unfilleted mesh by
**47.85% of axle drop**.  Step 1 asks *which gene box* Gate 1 is a statement about; there is
now a second question of exactly the same kind — **which MESH** — and it was not on this
file's list.  It does not weaken Gate 1 (a correction factor is less likely to hold across
two meshes than one), and it costs nothing to write both arguments down in one sitting.
