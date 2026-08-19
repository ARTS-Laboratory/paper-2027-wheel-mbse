# FILLET_PLAN.md — mesh the junction fillets

**Open arc #2. Created 2026-08-16. Nothing started. NOT CHEAP — read the cost section first.**

**VERSION CONTROL IS PART OF THIS PROJECT'S WORKFLOW — CHANGED 2026-08-19.** The rule that
stood here read *"Ignore version control entirely. Do not commit, branch, stage, revert or
otherwise touch git."* **It is superseded.** The rules live in `PLAN.md`'s header block and
**only** there, so they cannot drift across ten files: one commit per finished unit of work
on `feature`, `make test` green first, never while a study driver is mid-write, a study
commit carries its regenerated `.json` and `.jpg`, a promotion is one atomic commit and never
one file — and **commits carry no assistant or tool attribution, no `Co-Authored-By:`
trailer, no session link, no generated-with footer.**

**DO NOT START THIS BEFORE `KINEMATICS_PLAN.md` IS ANSWERED.** A filleted-mesh ladder run
under the wrong kinematics measures the wrong thing, and this arc is far too expensive to run
twice. See the ranking note in PLAN.md.

---

## Why this arc exists

**The exported solid has filleted junctions. The FEA mesh does not.** `wheel_wheel.py:44`
says it in three words — *"FILLETS ARE NOT MODELLED"* — and the mesh meets the rings at a
sharp re-entrant corner, recovering `Kt` by a correction factor instead.

This has been "the top open item" since §29 and has been named-not-started through §30 and
§31. It is not one defect; it is the common cause behind four separate things the project has
already had to work around:

1. **Peak stress diverges under refinement**, so "the max is not a number" (M4, pinned since
   §14 by `test_peak_stress_diverges_but_the_field_converges`). The whole stress constraint
   had to be rebuilt around a p-norm × `Kt` because of it (M8b-i.6 step 2).
2. **The ±0.3% absolute deflection band could not be adjudicated.** §29 retired it to a
   relative clause: *"no worse than the incumbent measured identically"*. §29's own note —
   *"What would earn the absolute band back: fillet the junction in the FEA model"*.
3. **§29 spent 95 minutes** extrapolating a global functional and mis-identified the mechanism
   from its convergence order, because the singularity was in the way.
4. **NEW, §31: `R_hub` and `R_rim` — 2 of the 14 genes — are invisible to the FEA entirely.**
   Sweeping `R_hub` across its whole box (0.4 → 4.0) leaves the solved wheel **bit-identical**:
   hub share `0.04165644522132511` and axle drop `1.6207901051335216` at every point. The
   optimizer prices its two fillet genes through the beam model's `Kt` correlation and the
   buildability barriers only. **They have no mechanical feedback at all.**

Item 4 is the sharpest form of the argument and it is the newest. It is not that the stress
field is wrong near the corner — it is that two genes are steering on a correlation.

## What is ALREADY MEASURED, and must not be re-derived

**§30 built the validation target for this arc, at a cost of 8.5 seconds.**
`studies/study_corner_singularity.py` (`make corner`, 8.5 s, 1.56 GB) gives:

- **four named corners** with measured wedge angles,
- **per-corner divergence rates** (−0.44 on log h globally),
- the localisation, so a filleted model can be checked corner by corner rather than on a
  single global number.

**That is what a filleted mesh has to move.** Do not begin by re-running `make gci` (95
minutes) — §30's explicit lesson is that the cheap local measurement already answered the
question the expensive global one was run for.

**§30 also retracted a Williams agreement** because `study_deflection_gci.py` drew every `h`
from `wheel_mesh` while `wheel_objective` solved on `wheel_wheel` — **both modules export
configs named `smoke`/`coarse`/`medium`/`fine` and they are different meshes.** Any `h` this
arc computes must come from the same module that solved the field. This has bitten the tree
once already and it inflated a convergence order by 25%.

## The cost, stated up front

This is the expensive item and it has been deferred five times for that reason. The mesh
generator builds sector 0's seven blocks and rotates them; a fillet at the hub and rim
junctions changes the block topology at both seams, which is the part that is not cheap. §16
through §24 exist to build and price the fillets *as geometry*; none of that work touched the
mesh.

**Budget the topology work before the ladder.** If the blocks cannot be filleted without
breaking the seam merge, that is the finding and it should surface in days, not after a
ladder run.

## THE PLAN

### Step 0 — baseline the four corners

`make corner` on the shipped genome. Record the four wedge angles and per-corner rates. This
is the "before" and it costs 8.5 s. Also record `make test` green, because this arc will touch
`wheel_wheel.build_wheel`, which most of the suite runs through.

### Step 1 — can the blocks be filleted at all? Answer this before anything else

The seam merge is the risk. `build_wheel` merges sector seams through a union-find and asserts
the twelve sectors are exact rigid copies. A fillet at the hub and rim junctions changes the
block boundaries that seam table describes.

Produce a filleted mesh for **one sector** and check:

- the seam table still closes,
- element quality at the fillet (`studies/study_mesh_quality.py` exists for this),
- `test_axle_drop_is_exactly_12_fold_periodic` still holds to 1e-10 — the rotational
  periodicity check that already caught one genuine bug.

**If the topology cannot be made to close, stop and report that.** It is a legitimate outcome
and it is worth knowing in days rather than weeks.

### Step 2 — the ladder, on the corners rather than on a global functional

With a filleted mesh, re-run `make corner`. The claim to test is specific:

- the **peak stress stops diverging** — this is the one that unlocks quoting a max,
- the per-corner rates move toward the non-singular case,
- `test_peak_stress_diverges_but_the_field_converges` **starts failing**, which its own
  docstring anticipates: *"either fillets were added (good, update this test) or the stress
  recovery changed"*.

That test failing is the success condition, not a regression. Update it deliberately.

### Step 3 — what it unlocks, priced one at a time

Only after Step 2 confirms the singularity is gone:

1. **`R_hub` and `R_rim` become live FEA genes.** Re-run the `R_hub` sweep from
   `studies/study_reds_hub_share.py --sweep`; it should stop being bit-identical. **That
   sweep is the cleanest possible acceptance test for this whole arc** — it currently returns
   the same 17 significant figures at every point in the box, and it should not.
2. **The absolute deflection band** (§29) can be reconsidered. Note this is a *plan-level*
   gate; re-instating it is a separate decision with its own record.
3. **The stress constraint** can be revisited — M8b-i.6 rebuilt it around a divergent max, and
   `stress_scale_measured` survives in the report as the evidence for that change.
4. **`HUBSHARE_PLAN.md`** — the hub compliance share is currently not converged for the
   shipped genome and the unfilleted corner is the prime suspect. This arc may resolve that
   one for free; check before doing work there.
   **RE-MEASURED 2026-08-19 (PLAN §38's uncap flip), and the suspect is now the ONLY one
   left.** The five-rung figures above were taken on the capped mesh; on the shipped default
   they are 0.0325 → **0.0371**, drift **+14.05%** (was 0.0392 → 0.0463, +18.3%). Removing the
   half end cap — the other junction artefact, and a ~19% cut in the LEVEL — bought only 4.3
   points of the drift, i.e. it behaved as a constant multiplicative offset. **The quantity is
   still not converged and is still climbing at `ultra`**, so this arc still gates that one,
   and Step 3 item 1's sweep is still bit-identical across the whole box. Quote the faithful
   numbers, not the capped ones. See `HUBSHARE_PLAN.md`'s 2026-08-19 section.

## What must NOT happen

- **Do not start before `KINEMATICS_PLAN.md` is answered.**
- **Do not begin with `make gci`.** 95 minutes for a question `make corner` answers in 8.5.
- **Do not take `h` from a different module than the one that solved the field.** §30's
  retraction is the worked example.
- **Do not delete `test_peak_stress_diverges_but_the_field_converges`** when it starts
  failing. Update it, with the new measurement in the docstring. No test deleted (§31).
- **Nothing in `best_solution.json` is touched.** A filleted mesh changes what every measured
  number means; re-descending and promoting inside the same arc would make it impossible to
  tell the mesh change from the design change.

---

# STEP 0 RECORD — 2026-08-16. The baseline reproduces BIT-IDENTICALLY, and the prerequisite is checked.

**`make corner` re-run on the shipped genome: 8.8 s, and the report differs from §30's
committed `studies/study_corner_singularity.json` in ZERO fields.** Every wedge angle, every
Williams eigenvalue, every per-corner divergence slope. The "before" is locked and it is the
committed artifact — no need to keep a private copy.

```
  corner      wedge deg  re-entrant  lambda_W     peak growth over the ladder   diverges
  hub:P_t        321.10        True    0.5032          5.14x                      True
  hub:P_c        296.75        True    0.5144          2.56x                      True
  rim:P_t        321.33        True    0.5031          5.58x                      True
  rim:P_c        307.94        True    0.5079          2.43x                      True
  global_max_vm                                        2.43x  (61.92 -> 150.59 MPa)
```

The driver's own Williams check is sound — 360 deg crack -> exactly 0.500000, 270 deg ->
0.5445 (textbook) — so these exponents are trustworthy.

## THE PREREQUISITE PLAN.md NAMED IS NOW ANSWERED, AND IT IS A CAVEAT RATHER THAN A BLOCKER

PLAN.md's ranking note said the one thing to check before starting this arc is whether the
study drivers silently take `linear` after §32. **They do.**
`studies/study_corner_singularity.py` **mentions `kinematics` zero times** and calls
`fem.solve_wheel(mesh)` bare at line 167, so it takes `wheel_problem`'s kernel default, which
§32 deliberately left at `linear` — on a wheel §32 measured as 23.16% softer under SVK at
service load.

**This does NOT invalidate Step 0, and the reason is worth stating precisely.** Williams'
wedge solution is itself a LINEAR-ELASTIC result, and the quantity this arc must move is the
EXPONENT — whether the peak diverges at all. A geometric-nonlinearity correction changes the
amplitude of the field, not the order of its corner singularity. The driver reproducing the
360 deg crack at exactly 0.5 is the evidence that it is measuring the exponent correctly.

**What DOES inherit the linear kernel, and must say so wherever it is quoted:**

- the absolute peak stresses in MPa (61.92 -> 150.59 across the ladder),
- anything this arc concludes about `R_hub` / `R_rim` SENSITIVITY, which is Step 3 item 1.

Step 3's `study_reds_hub_share.py --sweep` acceptance test is the one to watch: it is a
sensitivity claim, so it must be run under SVK explicitly or its result is about the wrong
physics. **Do not let Step 3 take the kernel default.**

# STEP 1 RECORD — 2026-08-16, IN PROGRESS. The seam risk is real but it is NOT uniform: the two corner families cost differently.

Measured by building sector 0's blocks and locating each corner in EVERY block that owns a
node within 1e-9, rather than by reading it out of the source:

```
  corner     blocks meeting at it                                              seams touched
  hub:P_t    spoke(i0,j1)  hub_junction(i0,j0)  hub_collar_weld(i1,j1)  hub_collar_free(i0,j1)   3 of 8
  hub:P_c    hub_junction(i1,j0)  hub_collar_weld(i0,j1)                                          1 of 8
  rim:P_t    spoke(i1,j1)  rim_junction(i0,j0)  rim_band_weld(i1,j0)    rim_band_free(i0,j0)      3 of 8
  rim:P_c    rim_junction(i1,j0)  rim_band_weld(i0,j0)                                            1 of 8
```

**THE COST IS NOT UNIFORM AND IT IS INVERSELY ALIGNED WITH THE PAYOFF.**

- **`P_c` is the weld arc's START and only TWO blocks meet there.** It is a corner of the
  junction block and the ring's weld block, and it touches exactly one seam
  (`junction.j0 <-> weld.j1`). This is the cheap one.
- **`P_t` is the weld arc's END and FOUR blocks meet there** — it is simultaneously a corner
  of the spoke, the junction, the ring's weld block AND the ring's free block, because the
  weld footprint is SPLIT at exactly this point (`sector_blocks`: "splitting rather than
  grading is what makes the partial seam exact"). Moving `P_t` to a fillet tangent point
  perturbs three of the eight seam-table entries at once.

**And `P_t` is the pair that diverges fastest** — 5.14x and 5.58x growth against `P_c`'s 2.56x
and 2.43x. The corners that most need the fillet are the expensive ones.

## WHAT THIS MEANS FOR THE STEP 1 GO/NO-GO

The encouraging half: the junction blocks ALREADY EXIST as separate mediating blocks between
the spoke and the ring, and the invariant `build_wheel` actually enforces at a seam is a NODE
COUNT (`if ia.size != ib.size: raise ValueError`), not a shape. A fillet that curves a block's
boundary while preserving its `[ni, nj]` grid does not disturb the union-find merge at all.

The hard half is `P_t` specifically: the fillet material sits in the re-entrant notch between
the spoke's free flank and the ring circle, which is a region **no current block covers**, and
`P_t` is the shared endpoint of the weld/free split. Either a new fillet block is inserted
there (new seams, new counts) or `hub_collar_free`'s i0 edge and the spoke's j1 edge are grown
to absorb it (same counts, new curves). **The second is the one to try first** — it is the
only one that leaves the seam table's eight entries untouched.

# STEP 1 RECORD, PART 2 — 2026-08-17. HALF THE CORNERS ARE NOT THE PART'S CORNERS.

Before building anything I priced the fillet against the geometry it has to sit in, and
that measurement reframed the arc. **`P_t` is the shipped part's corner to seven
decimal places. `P_c` is a MESH ARTEFACT the part does not have.**

## HOW THE TWO GEOMETRIES DIFFER, MEASURED

`wheel_wheel.sector_blocks` terminates the spoke **ON** the ring circle and closes it
with a half **END CAP** (`right=_lerp_points(P_c, far_end, n_th)`), so its two corners
per junction are `P_t` (the straddling flank crossing the circle) and `P_c` (where that
end cap meets the circle).

`wheel_step_export.spoke_profile` does something else. `_embed` continues **both**
flanks along one shared straight direction until both are past `HUB_EMBED_RADIUS_MM`
(12.20, *inside* the hub disk) or `RIM_EMBED_RADIUS_MM` (50.25, *outside* the OD). The
root cap is buried and the tip cap is clipped, so **the shipped solid has no end cap at
either ring.** Its two corners per junction are both flank-crossings.

I reimplemented `_embed` in numpy (no OCC — it is 20 lines and pure geometry: blend 1.0
radial-inward at the hub, a blend search outward at the rim) and walked the resulting
closed outline for every crossing of r = 12.70 and r = 48.50. **Two crossings per spoke
per ring, i.e. 24 and 24 — exactly the `hub_edges`/`rim_edges` the shipped manifest
counts.** So the reconstruction is the part.

```
  ring   corner            theta (deg)     wedge (deg)   spoke-side leg
  hub    MESH P_t           +6.117810        321.29       41.1182 mm  (flank)
  hub    PART top_flank     +6.117480        322.02       41.1182 mm  (flank)
                            ^^^^ agree to 3.3e-4 deg = 0.073 um of arc
  hub    MESH P_c           +0.000000        297.33        0.7369 mm  (END CAP = t0/2)
  hub    PART bot_flank     -1.526730        268.47       41.1182 mm  (flank)
                            ^^^^ 1.527 deg = 0.338 mm apart, and 28.7 deg of wedge apart

  rim    MESH P_t           +1.360780        321.29       41.1182 mm  (flank)
  rim    PART top_flank     +1.360760        321.07       41.1182 mm  (flank)
                            ^^^^ agree to 2e-5 deg = 17 nm of arc
  rim    MESH P_c           +0.000000        307.39        0.7156 mm  (END CAP = t3/2)
  rim    PART bot_flank     -1.315860        219.90       41.1182 mm  (flank)
                            ^^^^ 1.316 deg = 1.114 mm apart, and 87.5 deg of wedge apart
```

The wedges above are computed from block geometry; they reproduce §30's independently
measured mesh wedges (321.10 / 296.75 / 321.33 / 307.94) to within 0.6 deg, which is
the cross-check that says both measurements are of the same thing.

**The manifest's `worst_wedge_deg` belongs to the `P_t` family**: hub 322.0 against
322.02 computed, rim 320.0 against 321.07 at the exporter's 2 deg probe
resolution.

## WHAT THAT DOES TO THE FILLET, PRICED

A fillet needs two legs. Tangent length `T = R / tan(void/2)`, at the shipped
`R_hub = 0.6636`, `R_rim = 3.0000`:

```
  corner    void (deg)   T (mm)    ring-arc leg           spoke-side leg
  hub:P_t      38.708    1.8893    5.2937 mm free arc     41.1182 mm     OK
  hub:P_c      62.668    1.0900    5.2937 mm free arc      0.7369 mm     1.48x THE LEG
  rim:P_t      39.489    8.3583   24.2427 mm free arc     41.1182 mm     OK
  rim:P_c      52.606    6.0692   24.2427 mm free arc      0.7156 mm     8.48x THE LEG
```

Along the **ring arc** both fillets fit with room: the two corners of a ring share the
free arc between them (`P_c [weld] P_t [free] P_c(next)`), and `T(P_t) + T(P_c)` is
2.979 mm of 5.294 (56.3%) at the hub and 14.428 mm of 24.243 (59.5%) at the rim. **The
seam-collision worry that ranked this arc as expensive does not bind.**

Along the **spoke side** it is `P_c` that fails, and it fails because the leg it would
have to lie on is half a wall thickness. **A fillet cannot be tangent to a face 8.5
times shorter than its own tangent length.** That is not a mesh-resolution problem and
no refinement fixes it.

## THE GO/NO-GO, SPLIT

**`P_t`: GO.** It is the part's corner, the mesh already puts it in the right place to
0.073 um, the fillet fits on both legs, and **it is the fast-diverging pair** — 5.14x
and 5.58x growth over the ladder against `P_c`'s 2.56x and 2.43x. The corners worth
filleting are the ones that are real.

**`P_c`: NO-GO, and filleting it would be a mistake rather than merely hard.** It is
not a corner of the shipped part. Rounding it in the FEA model would move the mesh
*away* from the geometry that ships, and it cannot be done at the shipped radii anyway.

**The end cap is the artefact, not the corner.** The honest long-run fix for the second
corner is to stop capping the spoke at the ring and embed it the way the exporter does,
which puts the mesh's second corner at the flank crossing (-1.527 deg / -1.316 deg)
where the part's is. **That is a junction-topology change, not a fillet, and it is
filed rather than done** — it moves the welded volume and therefore mass, hub share and
every calibrated constant downstream of them. Planned in `UNCAP_PLAN.md`.

> **CORRECTION, same day.** An earlier draft of this record said the mesh/part
> difference was undocumented. **It is not.** `wheel_wheel.py`'s module docstring, under
> *WHAT IS AND IS NOT MODELLED*, states that `_embed` is not reproduced, quantifies it at
> 3.03 mm2 per spoke and ~1.4% of material "all of it at the junctions where it acts as a
> gusset", and calls it deliberate. What is NOT in that justification — which weighs area
> and stiffness — is PART 4 below: that the idealisation manufactures the corner carrying
> the wheel's global peak stress. The corner-provenance table above stands; the claim
> that nobody had written the difference down does not.

# STEP 1 RECORD, PART 3 — 2026-08-17. STEP 1 IS ANSWERED. THE SEAM TABLE SURVIVES; THE SPOKE BLOCK DOES NOT.

**Built it.** `wheel_wheel.sector_blocks` and `build_wheel` now take an opt-in
`fillet=` (`None` default, `True` for the genome's radii, or an explicit
`(R_hub, R_rim)` pair). The construction is the plan's option 2 — grow the existing
blocks, add no seams — implemented as: move the four-block corner from `P_t` to **`B`,
the fillet's tangent point on the ring circle**, and let everything else follow.

## THE PART OF THE PLAN'S RISK ASSESSMENT THAT WAS RIGHT, AND THE PART THAT WAS NOT

**The seam merge was never the problem.** Moving the corner to `B` needs *no code at
all* in six of the seven blocks, because the junctions and both ring blocks already
derive `theta_t` from the spoke's own end row. All eight seam entries keep their node
counts and their pairings, `_seam_table` is untouched, and measured on the shipped
genome at `coarse` the six non-spoke blocks come out **clean — zero mixed-sign cells:**

```
  block             shape     min |cell area|   mixed-sign cells
  hub_junction      (21, 9)      5.565e-03            0
  rim_junction      (21, 9)      6.322e-03            0
  hub_collar_weld   (21, 7)      7.817e-02            0
  hub_collar_free   (21, 7)      9.891e-02            0
  rim_band_weld     (21, 7)      1.114e-01            0
  rim_band_free     (21, 7)      2.069e-01            0
  spoke             (97, 9)      5.866e-03            4     <- at (0,7) and (94..95,6..7)
```

**The spoke block is what fails**, and `build_wheel` stops on it:
`12 of 4704 elements have non-positive area after orientation (worst -3.07e-02 mm2)` —
one per sector, i.e. the same corner twelve times.

## WHY, AND WHY NO BLEND FIXES IT

A fillet in a **~39 degree notch** is a long thin scoop: `T = R/tan(void/2) = 2.847 R`.
So the corner cannot move a little — it has to move `T`:

```
  junction   R (mm)    void (deg)    T (mm)    wall t (mm)    T / t
  hub        0.6636      38.708      1.8893      1.4738        1.28
  rim        3.0000      39.489      8.3582      1.4313        5.84
```

The spoke's end cross-section then runs from the **moved** corner to an **unmoved**
far-flank point, so it grows with `T`. Measured:

```
                    end cross-section     corner interior angle    cross product
  hub  unfilleted        1.4519 mm              88.678 deg            -0.079
  hub  filleted          2.7590 mm               3.601 deg            +0.008   SIGN FLIP
  rim  unfilleted        1.4165 mm              88.987 deg            +0.076
  rim  filleted          8.5963 mm               8.524 deg            -0.065   SIGN FLIP
```

An 8.60 mm "cross-section" on a 1.43 mm wall, adjacent to a 1.43 mm one, is not a
recoverable element. **And the corner angle collapse is not a blending artefact**: at
`B` the fillet arc leaves along the ring circle's tangent (that is what tangency means)
heading back toward `P_t`, while the cross-section to the far flank also heads back
toward the spoke. The two edges of the block are nearly parallel *by construction*. No
choice of interior blend changes the angle between two boundary curves.

**Nor does moving the corner elsewhere on the arc rescue it.** The arc's closest
approach to `P_t` is `R/sin(void/2) - R` = 1.339 mm at the hub and **5.880 mm at the
rim**, still 4x the wall — and every intermediate position costs the ring blocks their
exact-polar form, which was the whole reason to prefer option 2.

**This is not narrow to the shipped genome.** Holding the notch at its measured angle
and sweeping the gene box, `T/t` runs 0.77 -> 7.73 for `R_hub` (0.4 -> 4.0) and
0.97 -> 5.84 for `R_rim` (0.5 -> 3.0). The tangent length is comparable to the wall
**everywhere in the box**, and exceeds it over almost all of it.

## THE SWEEP, AND THE ONE NUMBER THAT SAYS "CONSTRUCTION, NOT GEOMETRY"

Largest radius this construction survives, each junction swept with the other off:

```
                coarse     medium      shipped
  hub          0.200 mm   0.100 mm     0.6636     3.3x / 6.6x too small
  rim          0.200 mm   0.100 mm     3.0000    15.0x / 30.0x too small
```

Both fold at an end-cross-section ratio of ~1.3x the wall, and **the limit TIGHTENS
under refinement.** A geometric limit — "the fillet does not fit in the notch" — would
be mesh-independent. This one halves from `coarse` to `medium`, which is the evidence
that what fails is the ruled interior of the spoke block, not the fillet.

**So do not read 0.2 mm as a physical bound.** It is a property of building the spoke
as a Coons patch of its four boundary curves.

## THE CONTROL

`fillet=None` short-circuits, so the default path is **bit-identical** — verified at
`smoke`, `coarse` and `medium`: same node count, `max|dx| = 0`, seam error unchanged at
3.20e-14 mm. `fillet=(0, 0)` instead goes through the whole Coons rebuild and agrees
with the default to **2.842e-14 mm** at all three configs. That second number is worth
keeping: it is an independent numerical confirmation that `sample` is affine in `eta`,
and therefore that the unfilleted spoke already *is* the Coons patch of its own
boundary curves — which is why swapping the construction in costs nothing when the
radii are zero.

`tests/test_wheel.py`, `test_mesh.py`, `test_golden.py`, `test_import_hygiene.py`: 73
passed. **`make test` after the change reads 476 passed, 2 xfailed, 0 failed, exit 0
(28 m 36 s)** — the same counts and the same two xfails as the §33 baseline taken before
it (476/2/0, 28 m 32 s). The `fillet=` parameter is inert unless asked for, and the
suite confirms it.

## STEP 1'S ANSWER

**The topology closes. The block interiors do not.** The plan asked whether the blocks
can be filleted without breaking the seam merge, and the answer is yes — that risk was
mis-ranked, and it cost one afternoon to retire rather than the weeks the arc was
budgeted for. What actually blocks a filleted mesh is that **the spoke block is ruled,
and a fillet whose tangent length is 1.3-5.8x the wall thickness cannot be absorbed by
ruling.**

Two routes remain, and both are real work rather than a tweak:

1. **A dedicated fillet block** (the plan's option 1) covering the curvilinear triangle
   `A - P_t - B`, with its own seam entries. Now the *preferred* route, and the reason
   has changed: not because option 2 breaks the seams — it does not — but because the
   fillet is geometrically comparable in size to the blocks around it and deserves to
   be one.
2. **A generated spoke block** — transfinite smoothing with a boundary correction that
   decays away from the junction — which would also fix the ruled block's other
   weakness, that any end correction currently propagates linearly down all 41 mm.

**Not attempted, and deliberately:** `study_mesh_quality.py` at the fillet and
`test_axle_drop_is_exactly_12_fold_periodic` on a filleted mesh. Both need a mesh that
assembles, and this one does not get past `_orient_elements`. Running them on the 0.2 mm
mesh that *does* assemble would be measuring a fillet 3-15x smaller than the one that
ships, which answers nothing.

**Step 2 is not reachable from here** and should not be attempted until one of the two
routes above exists.

# STEP 1 RECORD, PART 4 — 2026-08-17. STEP 2's SUCCESS CONDITION IS AT THE ARTEFACT CORNER, SO A FILLET BLOCK WOULD NOT DELIVER IT.

Checked before ranking the fillet block, because §17's standing lesson is to ask whether
the target binds. **It does not.**

`studies/study_corner_singularity.json` reports `global_max_vm_mpa` equal to `rim:P_c`'s
`peak_vm_mpa` at every rung (61.92 / 99.13 / 126.25 / 150.59). Two independent
statistics agreeing four times is worth checking rather than reading as coincidence, so
I located the peak directly — `argmax` over the whole Gauss-point field, then measured to
all twelve rotational copies of each corner:

```
  cfg      kinematics    global max      rim:P_c    rim:P_t    hub:P_c    hub:P_t
  coarse   kernel(lin)   99.1302 MPa    0.016190   1.151194   35.783821  35.881663
  coarse   svk           99.4487 MPa    0.016190   1.151194   35.783821  35.881663
  medium   kernel(lin)  126.2484 MPa    0.010745   1.151901   35.789256  35.887123
  medium   svk          126.3069 MPa    0.010745   1.151901   35.789256  35.887123
```

**The peak is 11-16 um from `rim:P_c`. It is on that corner, and SVK does not move it** —
same location to the digit, 0.3% in magnitude, which also retires the worry that the
corner driver's linear kernel could have put the peak in the wrong place. At `fine` the
corner ranking is `rim:P_c` 150.59 > `hub:P_c` 120.92 > `hub:P_t` 96.22 > `rim:P_t`
75.40: **both artefact corners outrank both real ones.**

Step 2 says the claim to test is that *"the peak stress stops diverging — this is the one
that unlocks quoting a max"*. **Filleting `P_t` cannot do that**, because the peak is not
at `P_t`. Routes 1 and 2 above round the corners that are real and leave the corner that
carries the maximum exactly as it is.

*The honest qualifier:* `P_t` diverges FASTER — 5.14x / 5.58x against `P_c`'s 2.56x /
2.43x, which its larger wedge predicts. At `fine` the two rim corners are a factor 2
apart and closing, so `P_t` overtakes somewhere past the current ladder. **On every mesh
anyone actually runs, `P_c` dominates; asymptotically `P_t` does.** Both have to go.

**This re-ranks the arc.** Removing the end cap — filed at the bottom of PART 2 as
future work — is now the FIRST thing, not the last: it deletes both `P_c` corners
outright rather than rounding them, it is the only change that makes the mesh agree with
the part, and it plausibly makes the fillet easy, because once both corners are
flank/ring crossings with 41 mm legs the end cross-section that folded the spoke block
does not exist. Planned in `UNCAP_PLAN.md`. See PLAN.md §34 Finding 4.
