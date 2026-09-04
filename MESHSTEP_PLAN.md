# MESHSTEP_PLAN.md — the 16.8–30.3× element-size step on the rim OD

**Open arc #6. Created 2026-08-16, carried forward from PLAN §20. Small. Nothing started.**

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

The rim OD is **not uniformly divided.** Per 30° sector it gets `n_weld` segments over the weld
arc and `n_rim_free` over the free arc, and they are not the same size. §20 measured it at
`coarse` on the shipped genome:

```
10 × 0.1682°  +  10 × 2.8318°  =  30.000°   exactly
```

**A 16.8× step.** And it is worse elsewhere: the ratio is **constant up the mesh ladder and a
function of the design** — 16.8 on `e126cc3`, 30.3 on `e4219f3` — tracking the weld footprint.

> **Refining buys a smaller element and the same step.**

That last sentence is the whole problem. This is not a discretisation error that goes away with
mesh refinement; it is a property of how the rim is divided, and it is invariant under the one
lever the project normally reaches for.

## Why it matters — the contact patch sits on the step

At `medium` the weld/free boundary sits at **+1.6824°** and the patch centre at **+1.885° to
+2.082°**. The patch is **smaller than the element containing it in all twelve cells**
(`patch/seg` 0.39–0.46 at `medium`), so which nodes the patch sees is decided by whether its
edge reaches back across the step:

| kinematics | patch interval | reaches back past 1.6824°? | nodes | quad points |
|---|---|---|---|---|
| linear | [1.5429, 2.3623] | **yes, by 0.1395°** | **3** | 11 |

A 0.14° margin is what stands between the patch seeing three nodes and seeing fewer. That is a
knife-edge, and it moves with the design because the weld footprint does.

§19 attributed a Group C red to this and **§20 found §19's stated cause was wrong** — the real
mechanism is the element-size step, which is also why the patch-spill bound came into scope.

## The prerequisite — §21 said the cheap G7 measurement comes first

§21 ranked this item with an explicit precondition:

> "**The rim OD element-size step** (§20), still needing its cheap G7 measurement first."

**Do that measurement before anything else in this arc.** It is cheap and it is the analogue of
§30's lesson — `make corner` at 8.5 s answered what `make gci` spent 95 minutes on. Find out
what the step is actually costing before designing a fix for it.

## THE PLAN

### Step 0 — reproduce, and measure the cost

1. Reproduce the ratio at `coarse` and `medium` on the shipped genome (16.8× expected) and
   confirm it is invariant up the ladder.
2. **The G7 measurement**: what does the step cost the quantity that matters? Not the element
   ratio itself — that is a mesh statistic — but the effect on `axle_drop` and on the contact
   solution. Compare against a rim OD divided uniformly at the same total element count.

**If the cost is negligible, close the arc here and record it.** A 16.8× step that changes no
reported number is a mesh wart, not a defect, and §31's lesson is that a statistic nobody has
priced should not drive work.

### Step 1 — only if it binds: the fix, and it is a meshing change

Grade the rim OD division so the weld and free arcs meet at comparable element size, rather
than each taking a fixed count. This touches `wheel_wheel`'s rim blocks and therefore the seam
merge; `test_axle_drop_is_exactly_12_fold_periodic` (1e-10) is the guard.

Note the constraint recorded in `WheelConfig`'s docstring: **`n_weld` is simultaneously the
junction's along-the-arc element count AND the collar/rim-band weld block's angular element
count**, and the weld and free blocks of a ring **share a radial edge, so they share `n_r`**.
Three of the config's numbers are not free. A grading change has to respect that or the seams
stop closing.

### Step 2 — re-check what §20 attributed to this

§20 brought the patch-spill bound into scope on the strength of this mechanism, and
`test_only_the_rim_od_near_the_bottom_is_loaded` was fixed under CONTACT_PLAN Step 3. Confirm
that fix is still correct under a graded mesh, and that the patch still sees enough nodes.

## What must NOT happen

- **Do not refine to fix it.** The ratio is invariant up the ladder; refining is the
  intuitive move and it is measurably useless here.
- **Do not change the patch geometry to dodge the step.** The assumed 3° patch is what M4's
  committed report was measured with, and `study_wheel_fea.py`'s header says re-deriving it
  under contact "would silently change every quoted value".
- **Do not skip the G7 cost measurement** and go straight to the meshing change.

---

## RE-MEASURED ON THE FILLETED MESH — 2026-09-03. **THE HEADLINE NUMBER IS 2.650x, AND THE FILLET DID STEP 1 AS A SIDE EFFECT.**

PLAN.md §106.  Geometry only, no solve — rim-OD nodes at `RIM_OUTER_RADIUS_MM` = 50.0 on
`best_solution.json`:

| mesh | weld band | free band | weld node step | free node step | STEP RATIO |
|---|---|---|---|---|---|
| coarse plain | 1.8746° | 28.1254° | 0.093731° | 1.406269° | **15.003** |
| coarse FILLETED | 8.2186° | 21.7814° | 0.410929° | 1.089071° | **2.650** |
| medium plain | 1.8746° | 28.1254° | 0.058582° | 0.878918° | **15.003** |
| medium FILLETED | 8.2186° | 21.7814° | 0.256831° | 0.680669° | **2.650** |

**THE MECHANISM, IN ONE LINE THIS FILE DID NOT HAVE:** the two bands get the SAME element
count, so the step ratio IS the arc-length ratio of the bands.  That is why the ratio is
invariant up the ladder and a function of the design — both reproduce here, on both meshes.
The fillet widens the weld band 4.384x (0.8180 mm -> 3.5860 mm of arc at the OD, a
difference of 2.77 mm against an applied `R_rim` of 3.0), so the ratio falls by that factor.

**Step 1 is therefore already done, by accident.**  It asks to *"grade the rim OD division
so the weld and free arcs meet at comparable element size"*.  2.650x is not 1.0, but Step 0's
own instruction — *"if the cost is negligible, close the arc here and record it"* — has to be
re-asked at 2.65x before any meshing change is designed.

**AND THE "WHY IT MATTERS" CLAUSE IS MEASURED AGAINST A BOUNDARY THAT MOVED.**  The contact
patch straddling the step by 0.1395° is read against a weld/free boundary at +0.9373° on the
plain mesh; on the filleted mesh it is at **+4.1093°**.  Where the patch now falls needs a
contact solve and is NOT measured here.

**A SECOND STALENESS, INDEPENDENT OF THE FILLET.**  `16.8` is not this genome's number even
unfilleted — it reads **15.003**.  This file says the ratio tracks the design, and the
shipped genome has been `09e8188` since §26, so the title's range was one promotion out of
date before §103 touched it.  Two stale mechanisms; only one is the fillet.
