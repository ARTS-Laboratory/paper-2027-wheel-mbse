# FILLET_PLAN.md — mesh the junction fillets

**Open arc #2. Created 2026-08-16. STEPS 0, 1, 1b AND 2 ARE DONE — 2026-08-23, PART 12.
STEP 3 IS NOT, AND IS BLOCKED.** The state, in four lines, so nobody re-derives it from
twelve PARTs:

- **Step 0** — the four-corner baseline. Done, and REFRESHED at PART 7 after §38's uncap
  flip made the committed one describe a mesh the tree had stopped building.
- **Step 1 / 1b** — can the blocks be filleted, and does the mesh integrate. **Yes**, via a
  re-cut of the sector into ELEVEN blocks (PART 10) wired into `build_wheel(fillet=)` at
  PART 11. `fillet=None` is bit-identical.
- **Step 2** — the ladder on the corners. **Run at PART 12.** The singularity at `P_t` is
  gone and the fillet's surface peak is a number; **the HEADLINE is not delivered**, because
  the wheel's peak is on `rim:P_c`, which needs the rim tri-block and not a fillet. The
  filleted wheel also deflects 37.97% less than the unfilleted mesh said it would.
- **Step 3** — what it unlocks. **Blocked on two things that are not this arc's**: the rim
  tri-block (PLAN §52 ranked 1) and making the filleted blocking genome-robust (PLAN §48) —
  6 of 16 feasible genomes refuse it, so `fillet=` is a MEASUREMENT INSTRUMENT for one
  genome and is not a path the optimizer may take.

**NOT CHEAP — read the cost section first.**

**VERSION CONTROL IS PART OF THIS PROJECT'S WORKFLOW — CHANGED 2026-08-19.** The rule that
stood here read *"Ignore version control entirely. Do not commit, branch, stage, revert or
otherwise touch git."* **It is superseded.** The rules live in `PLAN.md`'s header block and
**only** there, so they cannot drift across ten files: one commit per finished unit of work
on `feature`, `make test` green first, never while a study driver is mid-write, a study
commit carries its regenerated `.json` and `.jpg`, a promotion is one atomic commit and never
one file — and **commits carry no assistant or tool attribution, no `Co-Authored-By:`
trailer, no session link, no generated-with footer.**

**~~DO NOT START THIS BEFORE `KINEMATICS_PLAN.md` IS ANSWERED.~~ SATISFIED — §32, 2026-08-16.**
The concern was that a filleted-mesh ladder run under the wrong kinematics measures the wrong
thing. §33's item 3 then checked this arc's actual driver and turned the blocker into a
CAVEAT that has to be quoted rather than a gate: `studies/study_corner_singularity.py` calls
`fem.solve_wheel(mesh)` bare and so takes the LINEAR kernel default. The corner EXPONENTS
survive it — Williams' solution is itself linear-elastic — and any claim about stress
MAGNITUDES or about `R_hub`/`R_rim` sensitivity inherits the linear kernel and says so. PART
12's 38% deflection number is in that second class.

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

> **RUN 2026-08-23 — PART 12. Read the three claims below against the three answers there:
> the headline is NO, the per-corner rates are YES at `P_t` and NO at `P_c`, and the third
> is NO for a reason that is correct rather than incidental.** `make corner-fillet`, 22 s,
> artifact `studies/study_corner_singularity_fillet.json`.

With a filleted mesh, re-run `make corner`. The claim to test is specific:

- the **peak stress stops diverging** — this is the one that unlocks quoting a max,
- the per-corner rates move toward the non-singular case,
- `test_peak_stress_diverges_but_the_field_converges` **starts failing**, which its own
  docstring anticipates: *"either fillets were added (good, update this test) or the stress
  recovery changed"*.

That test failing is the success condition, not a regression. Update it deliberately.

### Step 3 — what it unlocks, priced one at a time

> **BLOCKED AS OF PART 12, AND THE BLOCKER IS NOT "Step 2 failed".** The precondition below
> reads *"only after Step 2 confirms the singularity is gone"*, and Step 2 confirms it is
> gone AT `P_t` and present at `P_c` — so the precondition is half met and the half that is
> missing needs the RIM TRI-BLOCK rather than more fillet work. Item 1 below is separately
> blocked on genome robustness: `R_hub`/`R_rim` now have mechanical feedback (38% of axle
> drop, monotone in `R`), but feedback on a mesh the optimizer may not build is not
> feedback. See PLAN §52's ranking.

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

> **THIS BASELINE IS SUPERSEDED — READ PART 7 (2026-08-22) BEFORE USING ANY NUMBER BELOW.**
> It was taken on the CAPPED mesh; §38 flipped `UNCAP_DEFAULT` on 2026-08-18 and this
> driver takes that default. The two `P_c` rows are wrong by 28.7 and 36.9 deg of wedge
> and every peak here is ~28% high. The committed artifact has been refreshed and the
> current baseline is in PART 7. **The `P_t` rows below are still correct** — they moved by
> 0.03 and 0.01 deg — and the prerequisite section under this table is unaffected.

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

> **CONTESTED 2026-08-21, RESOLVED IN THIS TABLE'S FAVOUR 2026-08-22 — see PART 6.**
> Re-swept by the criterion "does `build_wheel` raise", the first fold lands at 4.00/3.00
> mm (`coarse`) and 0.40/0.40 (`medium`), 10-20x more permissive; the disagreement was
> filed open because this table's criterion was not recorded and its apparatus did not
> survive. `make fillet` now reproduces BOTH rows from one sweep and names them: **this
> one counts mixed-sign cells and lands on the usable window's upper edge exactly, at
> every cell; the later one is `build_wheel`'s corner-only guard reading late.** Quote
> this row. The conclusion drawn below stands, with its mechanism sharpened.

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

> **RE-MEASURED ON THE UNCAPPED DEFAULT, 2026-08-22 — PART 7.** The conclusion of this
> part is UNCHANGED: the peak is still on `rim:P_c`, 15-24 um away, under both kinematics
> and both stress measures. Two things in it are not: SVK moves the magnitude by 4.3-4.5%
> (solve) rather than 0.3%, and the ranking sentence below — *"both artefact corners
> outrank both real ones"* — is **false on the current default**, where `hub:P_t` has
> overtaken `hub:P_c`. Exactly one artefact corner is left and it is the rim's.

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

---

# STEP 1 RECORD, PART 5 — 2026-08-21. THE UNCAP FLIP DID NOT MOVE THE FOLD. PART 4's PREDICTION IS RETIRED.

PART 4 re-ranked this arc on a prediction: removing the end cap "plausibly makes the
fillet easy, because once both corners are flank/ring crossings with 41 mm legs the end
cross-section that folded the spoke block does not exist." **PLAN.md §38 adopted
`UNCAP_DEFAULT` on 2026-08-18 and nobody came back to check.** This is that check, run
before spending the arc rather than after. **The prediction is wrong and the blocker
stands.**

## THE MEASUREMENT — A/B ON THE SAME CODE, SHIPPED GENOME

At the shipped radii (`fillet=True`), the failure is **byte-identical** capped vs
uncapped:

```
  cfg      uncap        result   _orient_elements
  coarse   (False,1.0)  FOLDS    12 of 4704 elements non-positive, worst -3.0725e-02 mm2
  coarse   (True, 1.0)  FOLDS    12 of 4704 elements non-positive, worst -3.0725e-02 mm2
  medium   (False,1.0)  FOLDS    36 of 12288 elements non-positive, worst -5.6587e-02 mm2
  medium   (True, 1.0)  FOLDS    36 of 12288 elements non-positive, worst -5.6587e-02 mm2
```

Swept per junction with the other off, **all 16 cells agree cell-for-cell** between
capped and uncapped — same build/fold verdict at every radius on
`0.05 .. 4.00 mm`, at both `coarse` and `medium`.

## WHY, STRUCTURALLY — AND WHY THIS IS NOT THE §38 PLUMBING BUG IT LOOKS LIKE

Identical-to-the-digit A/B results are the signature of a parameter that never reaches
the path, which is exactly the defect §38 found in `mesh_coords`/`coord_fn`. **Checked,
and it is not that.** `uncap` is consumed at `wheel_wheel.py:1067-1074` — `_uncap_blend`
and `_uncap_corner`, inside the **junction** block, setting where its right edge lands.
The **spoke** block never receives `uncap`, and neither does the unfilleted path
(`_sector_coords`'s `if fillet is None` branch samples `s_grid x eta_grid` directly).

**The fold is in the spoke block.** So the cap could not have been what folds it, and the
identical numbers are correct construction rather than a silent default.

PART 4's reasoning was about `P_c` disappearing. But the fold is at **`P_t`'s** fillet
moving the shared corner to `B` on the ring circle, and growing the spoke's end
cross-section from the moved corner to an unmoved far-flank point. **The cap is not on
that path.** PART 4 named the right consequence of uncapping — `P_c` really is gone,
`wheel_wheel.py:644-648` records it — and drew the wrong conclusion about this block.

## WHAT IS UNCHANGED

Both routes out of PART 3 stand exactly as written: **(1) a dedicated fillet block** with
its own seam entries, still the preferred one, or **(2) a generated spoke block** —
transfinite smoothing with a localised boundary correction. **Step 2 remains unreachable
until one of them exists.** Nothing about the arc's cost estimate improves.

## ONE DISCREPANCY, FILED OPEN RATHER THAN EXPLAINED AWAY

> **CLOSED 2026-08-22 by PART 6.** Both criteria are now named and both rows reproduce
> from one sweep (`make fillet`). PART 3's is the one to quote; this section's row is
> `build_wheel`'s corner-only guard, which cannot see a fold inside a Q9 element and
> overstates what is usable by 10-20x. The rest of this section stands as written.

PART 3 recorded "largest radius this construction survives" as **0.200 mm at `coarse`,
0.100 at `medium`**, both junctions, and `wheel_wheel.py`'s docstring quotes those as
current fact. Re-swept today, the **first fold** lands at:

```
                 hub        rim
  coarse       4.00 mm    3.00 mm
  medium       0.40 mm    0.40 mm
```

10-20x more permissive, though the qualitative signature PART 3 rested its argument on
**does reproduce**: the limit tightens sharply under refinement (10x here, 2x there),
which is what says the limit belongs to the construction and not to the geometry. The
`medium` hub row is also non-monotonic — folds at 0.40-0.80, builds again at 1.20-2.00,
folds at 3.00 — which is `_filleted_spoke`'s `cap = (n_sp-1)//3` node clamp saturating,
not a second geometric regime.

**This is NOT asserted to be drift, and the docstring is NOT edited on the strength of
it.** Today's criterion is "does `build_wheel` raise"; PART 3's criterion is not recorded
beyond the phrase above, and **no test or script survives to re-run it** — it was a
scratch measurement written up in prose. The two may simply be measuring different
things. Reconstructing PART 3's apparatus is the only way to tell, and guessing is the
error PLAN.md §41 was written about.

**Whoever takes route 1 or route 2 has to build that apparatus anyway** — PART 3 already
says this construction "is what a fillet-block implementation will be checked against" —
so the reconciliation should happen there, as its first act, not as a separate errand.

---

# STEP 1 RECORD, PART 6 — 2026-08-22. THE APPARATUS EXISTS. PART 3's TABLE IS RIGHT, PART 5's IS A WEAK GUARD READING LATE, AND THE WINDOW HAS A LOWER EDGE NEITHER SWEEP LOOKED FOR.

PART 5 filed its discrepancy open and named the fix: *"whoever takes route 1 or route 2
has to build that apparatus anyway ... so the reconciliation should happen there, as its
first act, not as a separate errand."* This is that first act. **The apparatus is
`studies/study_fillet_fold.py` (`make fillet`, 37.8 s, geometry and Jacobians only, no
field solved), it is committed with its report, and it is under test —
`tests/test_fillet_fold.py`, 20 tests, on a construction that had none.**

## THE RECONCILIATION: BOTH ROWS REPRODUCE, FROM ONE SWEEP, AND THEY ARE DIFFERENT
## CRITERIA

```
  config:junction    PART 3 criterion       PART 5 criterion     builds AND integrates
                     largest surviving      first fold           usable window (mm)
  coarse:hub              0.20                  4.00                 0.12 - 0.24
  coarse:rim              0.20                  3.00                 0.11 - 0.23
  medium:hub              0.10                  0.40                 0.07 - 0.11
  medium:rim              0.10                  0.40                 0.07 - 0.11
```

Every recorded figure comes back on its own grid, to the digit. **They were never
measuring the same thing:**

- **PART 3 counted MIXED-SIGN CELLS in the spoke block** — its own write-up tabulates
  exactly that column for all seven blocks, which is what identifies the criterion.
- **PART 5 asked whether `build_wheel` raises.**

**And PART 5's is the weaker instrument, structurally rather than by tolerance.**
`_orient_elements` is a shoelace over each element's four CORNERS; every config is
`order=2`, so one Q9 element spans 2x2 cells and its five mid nodes take no part in that
sum. A fold inside an element is invisible to the check that exists to catch folds.
Measured, `build_wheel` accepts 29-45 of the swept radii at each junction whose meshes
carry up to 72 elements with a non-positive Jacobian.

**So PART 3's row is the one to quote, and the docstring is now edited to say so.** Not
because it is older but because it is within one grid step of the criterion that decides:
`det J` at the Gauss points the FE assembly actually integrates.

## THE THIRD CRITERION SAYS SOMETHING SHARPER THAN EITHER ROW

**There is no usable interval `0 < R < R_max`. An arbitrarily small fillet folds too.**
What exists is a window with two edges, and BOTH are node allocation rather than geometry:

- **Upper edge — exact at all four cells.** `k0 = clip(round((s_A - s0)/ds), 1, cap)`
  steps from 1 to 2 and the element straddling the arc's end inverts. At `coarse` the hub
  is usable at 0.24 and folded at 0.25; the notch, the tangent length and the end
  cross-section move by under 2% across that step. **PART 3's criterion lands on this edge
  exactly, at every cell** — 0.20/0.10 are simply the legacy grid's nearest points below it.
- **Lower edge — the same clamp's LOWER bound.** With the tangent point nearer than one
  station, `k0` is held at 1 and the first Q9 element's mid-side node is dragged toward
  its own end; a quadratic edge is singular once that fraction leaves the middle half.
  Measured, the window opens as the fraction climbs back through ~0.4 (0.378 -> 0.403 at
  the coarse hub, 0.412 -> 0.458 at the medium rim).

Both edges move with `ds`, which is the mechanism behind the one claim PART 3 and PART 5
already agreed on — the limit tightens under refinement, so it belongs to the construction.
**Criterion C names which part of the construction.** PART 3's "the spoke block is ruled"
was right about the block and incomplete about the reason: at the shipped radii the ruled
interior does fold, but the window's edges are set by where nodes land, not by ruling.

## THE FINDING THAT IS NOT ABOUT THE FILLET

`build_wheel`'s fold guard does not see sub-element folds, and that is not confined to the
opt-in path. Sampled over the gene box at `coarse` (1000 geometrically feasible draws,
fixed seed, in the report):

```
  built by build_wheel                             839
    with non-positive Gauss detJ                   177   (21.1%)
      and min scaled Jacobian >= 0.2                 4   both corner-only checks blind at once
  ALSO fold_margin > 0 ("meshable")                615
    with non-positive Gauss detJ                     2   (0 of them minSJ-ok)
```

**What covers the default path is `fold_margin`, and it is not an element check at all** —
it rejects the genome before the mesh exists. Which is exactly why it cannot cover this
arc: `fold_margin` reads genes 0-11, and `R_hub`/`R_rim` are 12 and 13. Pinned by
`test_fold_margin_cannot_see_the_fillet_genes`, which moves both across their whole box
and finds the margin unchanged to the bit.

**Strengthening the guard is NOT this arc's work and was not done.** It is recorded in
`_orient_elements`' docstring so nobody reads a `build_wheel` that returns as a mesh that
integrates. Whoever takes route 1 or route 2 needs a Jacobian-level check anyway — the
apparatus now has one.

## WHAT IS UNCHANGED, AND WHAT THIS COST

Both routes out of PART 3 stand exactly as written: **(1) a dedicated fillet block** with
its own seam entries, still preferred, or **(2) a generated spoke block**. Step 2 remains
unreachable until one exists. Nothing was promoted, `best_solution.json` is untouched, the
default mesh is bit-identical (re-verified as a control in the study and in the tests), and
no threshold moved.

What changed is that the instrument both routes will be judged against is now
re-runnable, named, tested, and it answers a question neither route could previously ask:
**"is this fillet mesh valid?" now has a criterion — `det J` at the Gauss points — instead
of three criteria that disagree by 20x.**

---

# STEP 1 RECORD, PART 7 — 2026-08-22. STEP 0's BASELINE DESCRIBED A MESH THE TREE STOPPED BUILDING FOUR DAYS LATER. PART 4's FINDING SURVIVES; PART 4's RANKING DOES NOT.

PART 5 re-checked its own prediction against §38's uncap flip and found it wrong. **The
same question had never been asked of Step 0.** `studies/study_corner_singularity.py`
calls `build_wheel(genes, cfg)` bare, so it takes `UNCAP_DEFAULT`; §38 flipped that on
2026-08-18; the committed artifact is from 2026-08-16. **Step 0's "before" — the thing
Step 2's success is measured against, and which this file says in terms is "locked and it
is the committed artifact" — was a capped-mesh record.**

`make corner` costs 8 s. It has now been re-run and the artifact is refreshed.

## WHAT MOVED, AND BY HOW MUCH

```
  corner       wedge deg              lambda_W            peak MPa at `fine`
               capped -> uncapped     capped -> uncapped   capped -> uncapped
  hub:P_t      321.10 -> 321.13       0.5032 -> 0.5032      96.22 ->  85.93
  hub:P_c      296.75 -> 268.08       0.5144 -> 0.5477     120.92 ->  66.77
  rim:P_t      321.33 -> 321.32       0.5031 -> 0.5031      75.40 ->  60.69
  rim:P_c      307.94 -> 271.02       0.5079 -> 0.5429     150.59 -> 108.57
  global_max                                               150.59 -> 108.57  (-27.9%)
```

**The `P_t` pair does not move.** It is the part's corner and its wedge is set by the
spoke geometry, not by a mesh option — 0.03 deg and 0.01 deg across a flip that moved the
other two by 28.7 and 36.9. That asymmetry is now pinned per family in
`tests/test_corner_singularity.py` instead of one band across all four.

**The `P_c` pair is barely re-entrant now** — 268.1 and 271.0 deg, against the part's
268.47 (hub) and 219.90 (rim, still 51 deg out, exactly as `UNCAP_PLAN` Step 2 said it
would be). The hub gap reads 0.39 deg here against §36's 0.01: §36 measured its
constructed corner against the part directly, while this is the `fine` mesh's wedge summed
from incident element angles, whose agreement with the geometric value PART 2 put at
±0.6 deg. The two are consistent; do not read 0.39 as a discrepancy with §36.

Their `lambda` rose above the `< 0.53` window a test had been asserting since §30, which is
the assertion that went red on the refresh and was updated with the measurement rather than
widened silently.

## PART 4's FINDING SURVIVES THE FLIP. RE-MEASURED, NOT ASSUMED

PART 4 located the wheel's global peak by `argmax` over the whole Gauss field and measured
it to all twelve rotational copies of each corner. Re-run on the current default:

```
  cfg      solve     recovery    global max      nearest corner
  coarse   linear    linear       73.7282 MPa    rim:P_c at 23.8 um
  coarse   svk       linear       77.0312 MPa    rim:P_c at 23.8 um
  coarse   svk       svk          87.5002 MPa    rim:P_c at 23.8 um
  medium   linear    linear       91.1521 MPa    rim:P_c at 15.3 um
  medium   svk       linear       95.0301 MPa    rim:P_c at 15.3 um
  medium   svk       svk         109.0447 MPa    rim:P_c at 15.3 um
```

**The peak is still on `rim:P_c`, 15-24 um away, under both kinematics and both stress
measures** — and `global_max_vm_mpa` equals `rim:P_c`'s `peak_vm_mpa` at all four rungs of
the refreshed report, as it did before. **So PART 4's conclusion stands unchanged: a
fillet at `P_t` cannot deliver Step 2's headline, because the peak is not at `P_t`.**
Routes 1 and 2 round the corners that are real and leave the corner that carries the
maximum exactly as it is.

*One PART 4 number does not survive:* it recorded that SVK "does not move it — 0.3% in
magnitude". On the uncapped mesh the solve alone moves it **4.3-4.5%**, and the SVK stress
RECOVERY moves it another 12-13%. The location is unmoved to the digit; the magnitude is
an order of magnitude more kinematics-sensitive than it was. PART 4 conflated the two
sensitivities into one column and the table above separates them.

## WHAT DOES NOT SURVIVE: PART 4's RANKING SENTENCE

PART 4 wrote *"both artefact corners outrank both real ones"*, from the capped ranking
`rim:P_c` 150.59 > `hub:P_c` 120.92 > `hub:P_t` 96.22 > `rim:P_t` 75.40. On the current
default it is

```
  rim:P_c 108.57  >  hub:P_t 85.93  >  hub:P_c 66.77  >  rim:P_t 60.69
```

**`hub:P_c` has fallen below `hub:P_t`.** §38's flip fixed the hub's artefact corner
(wedge error 28.71 -> 0.4 deg) and its stress fell 45%. **Exactly one artefact corner is
left, and it is the rim's** — the one `UNCAP_PLAN` Step 2 proved needs a topology change
and §37 retired as not binding. So the arc's shape is now:

- the peak sits on the *only* remaining artefact corner,
- removing that corner needs the rim tri-block, priced and filed at §37 (partial-edge
  seams, forced 1-element strips, and it buys "only rim corner fidelity"),
- and the second-ranked corner is now a REAL one that the fillet WOULD address, and it is
  the fastest-diverging of the four (5.14x over the ladder).

**That last point is new and it is the only thing in four parts that argues UP for routes
1 and 2.** It does not change the go/no-go — Step 2's success condition is still measured
at a corner the fillet cannot reach — but it means a fillet at `P_t` is no longer
addressing the fourth-ranked corner. It is addressing the second.

## THE PROCESS POINT, WHICH IS THE SHARPEST ONE

The uncap commit (`c416cb5`, 2026-08-19) **edited `tests/test_corner_singularity.py`** — it
repaired a tie-break red in that very file — and left `studies/study_corner_singularity.json`
next to it untouched. Nothing went red, because every test that reads the corner artifact
reads the same stale file. `make studies` would not have caught it either: this driver is
one of the cheap ones and is not on that recipe.

`tests/test_corner_singularity.py` now carries
`test_the_committed_report_describes_the_mesh_the_tree_BUILDS_TODAY`, which rebuilds the
finest rung and compares wedge angles and element counts against the committed report. It
is geometry only, runs in well under a second, and it would have gone red on 2026-08-19.

---

# STEP 1 RECORD, PART 8 — 2026-08-22. PART 2's `P_c` NO-GO WAS PRICED ON THE END CAP. UNCAPPED IT FLIPS AT THE HUB, HOLDS AT THE RIM — AND §37's REASON FOR SHELVING THE TRI-BLOCK IS MEASURABLY WRONG IN ONE CLAUSE.

PART 7 refreshed Step 0's baseline and found PART 4's finding intact. This is the same
question asked of PART 2: **its `P_c` NO-GO was measured on the capped mesh, and every term
in it has since moved.** The leg, the void angle, and which geometry is the default all
changed at §38. `make junction` costs 4 s and now computes the pricing rather than leaving
it to arithmetic in a plan file.

## THE RE-PRICED TABLE, FROM `studies/study_junction_agreement.json`

```
  corner                        void    leg mm    T mm   T/leg    R_max    fits at shipped R
  hub  P_t                     38.86   41.1182   1.8812   0.05   14.504         yes
  hub  P_c  capped             62.82    0.7369   1.0867   1.47    0.450         NO   <- PART 2
  hub  P_c  AS BUILT           91.52    0.6600   0.6462   0.98    0.678         YES  <- flipped
  rim  P_t                     39.45   41.1182   8.3661   0.20   14.745         yes
  rim  P_c  capped             52.57    0.7156   6.0738   8.49    0.354         NO   <- PART 2
  rim  P_c  AS BUILT           89.49    0.5664   3.0270   5.34    0.561         NO
  rim  P_c  uncap=True        141.16    0.9114   1.0577   1.16    2.585         NO
```

`R_max = leg * tan(void/2)` is the largest radius the corner would accept as built, which
is the number to compare a gene box against.

**The legs got SHORTER at both rings** (0.737 -> 0.660 hub, 0.716 -> 0.566 rim) and the
verdicts still improved, which is what says the VOID ANGLE is what moved the answer:
uncapping opens `P_c` from 63/53 deg to 92/89, and a wider void needs a shorter tangent.

- **At the hub the NO-GO flips.** `T/leg` 1.47 -> 0.98; the shipped `R_hub` of 0.6636 fits
  under an `R_max` of 0.678. **By 2%**, on a 0.66 mm stub that ends in a 117 deg kink —
  admissible on paper and not a fillet anyone should mesh.
- **At the rim it does not.** 8.49 -> 5.34. The shipped `R_rim` of 3.0 needs five times the
  leg that is there. **And the rim is the one that matters**, because `rim:P_c` carries the
  wheel's global peak (PART 4, re-measured in PART 7).

## THE CLAUSE THIS FALSIFIES

§37 shelved the rim tri-block on the finding that it buys *"only rim corner fidelity — not
convergence, not the fillet, not a quotable peak"*. **"Not the fillet" is wrong, and the
factor is large.** On the faithful rim (`uncap=True`, the geometry the tri-block exists to
make buildable) `rim:P_c`'s admissible radius goes **0.561 mm -> 2.585 mm, a factor of
4.6** — because the corner opens to a 141 deg void and its leg grows to 0.911 mm.

§37 was not sloppy. It asked whether the tri-block unblocks the fillet arc's *known*
blocker, which was and is the ruled spoke block at `P_t`, and correctly answered no. **The
question nobody asked is whether it unblocks the fillet at `P_c`** — because PART 2 had
ruled `P_c` out permanently, on numbers that stopped being true two days later.

## WHAT IT DOES NOT BUY, WHICH IS THE HONEST OTHER HALF

**2.585 mm is still short of the shipped 3.0.** The faithful rim does not admit the fillet
this wheel ships, by 14%. Three things follow and they should not be run together:

1. `R_rim` is a GENE, and §22/§24 already measured its 3.0 ceiling as a trap — *"raising it
   would harvest loss the part does not pay for"*, and §24 priced the fillets at 4.406 g,
   8.77% of the part. A design at `R_rim <= 2.58` is inside the box, not a compromise
   invented to fit the mesh. **But choosing a radius to make the MESH work is choosing the
   design to fit the model, and that is backwards. It is a finding, not a plan.**
2. **Geometric admissibility is not meshability.** Every radius in the table above is far
   outside the construction's usable window of 0.12-0.24 mm at `coarse` (PART 6). A corner
   that accepts a fillet on its legs still has to be meshed by something that works.
3. The tri-block itself is unbuilt and §37 priced it honestly: partial-edge seams against
   a module whose docstring calls whole-edge single ownership "the whole safety net", and
   forced 1-element strips.

## WHERE THIS LEAVES THE ARC

The chain is now closed and every link is measured:

- Step 2's success condition is a non-divergent peak; the peak is on `rim:P_c` (PART 4,
  PART 7).
- A fillet is the only thing that removes a singularity (§37 CHECK 1).
- `rim:P_c` cannot take a fillet as built — 5.34x its leg (this part).
- It can take one at 4.6x the radius on the faithful rim, which needs the tri-block (§36).
- And `P_t`, the corner routes 1 and 2 do reach, is now the SECOND-ranked corner rather
  than the fourth (PART 7).

**So the tri-block is no longer a fidelity nicety filed behind this arc. It is on the only
measured path to Step 2's success condition**, and the fillet arc's two routes are
necessary but not sufficient. Ranked in PLAN §46.

---

# STEP 1 RECORD, PART 9 — 2026-08-22. BOTH OF PART 3's ROUTES ARE DEAD, THE SPOKE BLOCK WAS NEVER THE BLOCKER, AND THE BLOCK THAT DOES MESH IS THE ONE WHOSE CORNERS ARE OFF BOTH TANGENT POINTS.

PLAN §44 and §46 ranked route 1 or route 2 first, for the ninth arc. This is that item,
started the way PART 7 and PART 8 were: **re-check the step's premise before spending on
the step.** Route 1 has a premise — that the region it names can be a block — and it had
never been measured. It cannot.

**The apparatus is `studies/study_fillet_block.py` (`make filletblock`, ~42 s, geometry
and Jacobians only, no field solved), committed with its report, and under test —
`tests/test_fillet_block.py`, 44 tests.** It calls `wheel_wheel`'s own `_fillet_tangency`
and `coons_patch` rather than a second copy, so every number below is a statement about
the construction the tree ships. `make fillet` settled what *valid* means (§44: `det J` at
the Gauss points); this asks the question in front of it.

## FINDING 1 — ROUTE 2 CANNOT REACH THE ANGLE THAT FAILS. IT IS A BOUNDARY QUANTITY

PART 3 diagnosed the fold as the spoke's **corner interior angle collapsing** from ~89 deg
to 3.601 (hub) / 8.524 (rim). Re-measured on the current default — the numbers reproduce,
and PART 3's four figures are the one place in this arc a pre-flip number may be quoted
forward, because the fillet is at `P_t` and §38's flip left `P_t` alone to 0.01 deg (PART 7):

```
  cfg      junction   corner angle         end cross-section    the two BOUNDARY CURVES
  coarse   hub        88.678 -> 3.601      1.452 -> 2.759 mm         19.8003 deg
  coarse   rim        88.987 -> 8.524      1.416 -> 8.596 mm         12.4318 deg
  medium   hub        89.070 -> 10.543     1.452 -> 2.759 mm         19.8004 deg
  medium   rim        89.264 -> 10.418     1.416 -> 8.596 mm         12.4318 deg
```

Two things in that table. First, **the 3.601 is a SAMPLED angle** — corner to its two
neighbours — which is why it reads 10.5 at `medium`; the angle between the two boundary
*curves* is 19.80 / 12.43 and agrees between configs to under 0.001 deg. Both are now
reported so nobody has to work out which a quoted number was.

Second, and this is what decides route 2: **the fillet arc is on the spoke block's flank
EDGE and the end cross-section is its end EDGE.** All three nodes carrying the angle are
boundary nodes. Route 2 is "a generated spoke block", and every generating scheme —
transfinite, elliptic, Winslow — holds the boundary and moves the interior. Measured: 2000
Winslow sweeps move the boundary by **exactly 0.0 mm** and return the corner angle
**bit-identical**. Pinned as an equality, not a tolerance, because the claim is not that
smoothing barely helps — it is that the quantity is out of a smoother's reach by
construction.

## FINDING 2 — ROUTE 1 AS WRITTEN NAMES A REGION WITH TWO ZERO-DEGREE CORNERS

PART 3's route 1 is *"a dedicated fillet block covering the curvilinear triangle
`A - P_t - B`, with its own seam entries"*. **It is not a curvilinear triangle.** A fillet
is tangent to both legs by definition, so it meets each of them at zero angle and the
region it adds is a **cusp sliver**:

```
  junction   R (mm)   at A (deg)   at P_t (deg)   at B (deg)
  hub        0.05        0.6031      38.0556       0.0000
  hub        0.6636      0.6027      38.0556       0.0000     <- shipped
  hub        4.00        0.5297      38.0556       0.0000
  rim        0.05        0.4374      38.8886       0.0000
  rim        3.0000      0.4503      38.8886       0.0000     <- shipped
  rim        4.00        0.4651      38.8886       0.0000
```

**`B` is zero exactly**, and that is construction rather than tolerance: both curves
meeting there are circles, and `_fillet_tangency` places the arc's centre radially above
`B` by exactly `R`, so the tangency is not solved for. **`A` is 0.42-0.60 deg and the
residue is the spline flank's own curvature**, not slack in the bisection — across a 30x
span of radius it stays inside a 0.2 deg band, which is what a curvature term does and
what a convergence residue does not.

**No quad block covers a 0 deg corner, and a tri-block does not rescue it.** A tri-block
*subdivides* a region's corners — its three quads inherit the region's three vertices, one
each — so the smallest corner any decomposition of `A - P_t - B` can offer is that
region's own smallest, and that is `B`'s zero. `38 + 0.6 + 0` is not a triangle anybody
meshes, and **the 38 is the only one of the three anybody had looked at.**

Both obvious ways to close the region into a quad were built and measured, and both fail
the same way, because both keep a block edge on a **pre-fillet surface through a tangent
point**:

```
  candidate                        mixed cells at the shipped R, 1x / 2x / 4x refinement
                                   coarse:hub   coarse:rim   medium:hub   medium:rim
  grown_junction                    3 / 4 / 9    7 /10/ 19    4 /10/ 17   12 /18/ 30
    (junction block's ring edge run on past P_t and onto the fillet arc)
  pre_fillet_surfaces               6 / 8 /15    8 /11/ 21    8 /12/ 23   12 /17/ 32
    (PART 3's region, closed by the two end cross-sections)
```

**Both get WORSE under refinement**, which is the distinction that matters: a fold that
shrinks when you refine is a resolution problem, a fold that grows means the region is
wrong. And an elliptic interior solve — route 2's technique, offered to route 1's
candidates as the fair test — rescues neither.

## FINDING 3 — THE SPOKE BLOCK WAS NEVER THE BLOCKER

PART 3's headline has stood since 2026-08-17: *"what actually blocks a filleted mesh is
that the spoke block is ruled, and a fillet whose tangent length is 1.3-5.8x the wall
thickness cannot be absorbed by ruling."* **That is a statement about where the fillet was
put, not about ruling.**

Take the arc off the spoke's flank edge and end the spoke at the tangent station `s_A`
instead — the same ruled Coons block, a shorter station range, nothing else changed:

```
  config   junction   trimmed spoke, det J at the Gauss points
  coarse   hub        clean at 10/10 swept radii, 0.05 - 4.00 mm
  coarse   rim        clean at 10/10 swept radii, 0.05 - 4.00 mm
  medium   hub        clean at 10/10 swept radii, 0.05 - 4.00 mm
  medium   rim        clean at 10/10 swept radii, 0.05 - 4.00 mm
```

Against the shipped construction's usable window of **0.12-0.24 mm** at `coarse` and
0.07-0.11 at `medium` (PART 6). At `R = 0` the trimmed spoke is the default block **to the
bit** (max\|dx\| = 0.000e+00), so this is not a new construction — it is an ownership
change. **The fold does not disappear; it moves, whole, into whatever block then carries
the fillet.** Which is the right place for it, and is why the next finding is possible.

## FINDING 4 — THE BLOCK THAT MESHES, AND IT MESHES ACROSS THE WHOLE GENE BOX

What a structured mesher builds along a fillet is a **boundary layer**, and its defining
property is that **none of its corners sits on a tangency**:

```
  j0   the fillet arc  A -> B                         the free surface
  j1   that arc offset INTO the material              full wall at A, depth d at B
  i0   the spoke's end cross-section at s_A           cuts ACROSS the flank at A
  i1   a radial cut  B -> B''  of depth d             cuts ACROSS the ring circle at B
```

Both cusps become interior points of an edge instead of corners, and that is the whole of
why it works. The offset is taken along the arc's own outward normal, so `j1` is a
concentric arc of radius `R + w` and can never cusp however small `R` is — offsetting the
other way, toward the centre, folds, and that mutation is a test rather than a comment.

```
  config   junction   mixed cells 1x/2x/4x   min scaled J at shipped R   worst over the box
  coarse   hub            0 / 0 / 0                  0.9615                   0.9146
  coarse   rim            0 / 0 / 0                  0.9910                   0.9149
  medium   hub            0 / 0 / 0                  0.9668                   0.9226
  medium   rim            0 / 0 / 0                  0.9937                   0.9228
```

Zero non-positive Gauss points at **every radius from 0.05 to 4.00 mm**, both junctions,
both configs, three refinements. `wheel_objective.MIN_SJ_TARGET` is 0.2 with a barrier
weight of 3000 — the floor §38 measured the faithful rim's junction block collapsing under
at 0.0072. **This block clears it by more than 4x everywhere tested. It is the first
filleted block in this arc that meshes at the radii that actually ship.**

## FINDING 5 — WHAT IT COSTS, WHICH IS NOT OPTIONAL AND IS NOT AN EIGHTH BLOCK

The cut at `B` lands on **the far side of the ring circle** — inside the collar at the hub,
inside the band at the rim. That is not a tuning choice: at `B` the material on the
free-surface side has zero thickness, so a block that stops at the ring circle there
degenerates and a block that crosses it does not. **The ring circle therefore stops being
the junction/collar interface over the fillet's footprint.**

```
  junction   cut depth      of the ring's       fillet footprint    notch needed    spoke gives up
             (mm)           available depth     on the ring         in the ring     (coarse/medium)
  hub        0.7110         14.2% of 5.00       7.125 deg (23.8%)     7.003 deg      3.7 / 7.3 stations
  rim        0.6513         43.4% of 1.50       9.137 deg (30.5%)     2.935 deg     16.6 / 33.1 stations
```

So the remaining work is a **re-cut of the neighbours**, not a block bolted on: notch the
collar/band block over the footprint, split the spoke at `s_A`, and extend `_seam_table`.
That is real work and it is not done here. What is done is that it now has a target that
is known to mesh, instead of two that are known not to.

## ONE THING THIS FOUND ON THE WAY, WHICH IS ABOUT `make junction`

`make junction`'s `void_deg` at `P_t` — the quantity PART 8's whole re-pricing table is
built on — is **the angle to the spoke block's SECOND flank node, not to the flank's
tangent**. Reproduced from the committed artifact to the digit (38.86057256986532 at the
hub, 39.45464066055605 at the rim) and now reported next to the tangent:

```
  junction   void, one-node chord   void, analytic tangent   gap
  hub             38.8606                 38.0556           0.805 deg
  rim             39.4546                 38.8886           0.566 deg
```

The chord is **0.8 deg optimistic** because the flank is a spline that has already turned
over one `coarse` station. **No verdict in PART 8 moves on it:** both `P_t` rows clear by
5-20x, and the `P_c` rows are unaffected because under `uncap` that corner's leg is a
straight continuation whose chord and tangent are the same direction exactly. Which one is
right depends on the question — for "how much room does a fillet have" the chord is a fair
sample; for "what angle does a block corner have" only the tangent decides, because a
block corner is a limit and not a chord.

## WHAT IS UNCHANGED

**Nothing was promoted, `best_solution.json` is untouched and still 2026-08-14, the default
mesh is bit-identical, and no threshold moved.** `wheel_wheel.sector_blocks` is unchanged
except for its comment block, which had been carrying PART 3's two routes as the way
forward and now carries this. Every candidate block in this arc is built inside the study
from `wheel_wheel`'s own primitives; none of them is wired into `build_wheel` yet.

**Step 2 remains unreachable**, and PART 8's chain is untouched by any of this: the peak is
still on `rim:P_c`, which still refuses a fillet as built at 5.34x its leg. What changed is
that the corner routes 1 and 2 were aimed at — `P_t`, the SECOND-ranked corner since
PART 7 — now has a construction that can carry a fillet at the shipped radius, and the two
routes that had been ranked first for nine arcs are both retired with a measurement rather
than deferred again.

---

# STEP 1 RECORD, PART 10 — 2026-08-23. THE WHOLE SECTOR CLOSES: ELEVEN BLOCKS, FOURTEEN WHOLE-EDGE SEAMS. THE CUT PART 9 PRICED CANNOT CLOSE AT ANY DEPTH SHORT OF THE RING'S FAR SIDE, AND THAT IS A PROOF RATHER THAN A PREFERENCE.

PART 9 measured ONE block and said what remained was "a re-cut of the neighbours, not an
eighth block". **A block that meshes is not a mesh.** The boundary-layer block's inner
edge CROSSES the ring circle, so half of it seams to the junction block and half to the
ring block — one edge, two partners, which is a partial-edge seam and which this tree has
never had. `_seam_table`'s docstring calls whole-edge single ownership *"the whole safety
net"*.

**`studies/study_fillet_block.py` now builds the WHOLE filleted sector** — every block and
every seam, geometry and Jacobians only, nothing wired into `build_wheel` — and
`tests/test_fillet_block.py` grew from 44 tests to 66. `make filletblock` went from ~42 s
to ~85 s.

## FINDING 1 — IT CLOSES. ELEVEN BLOCKS, FOURTEEN SEAMS, EVERY SEAM WHOLE-EDGE

```
  spoke                trimmed to [s_A(hub), s_A(rim)]        [97, 9] at coarse
  <j>_fillet_a         the layer along the arc, ABOVE the ring circle    [9, 9]
  <j>_fillet_b         the wedge that crosses it, arc -> ring's FAR side [9, 9]
  <j>_junction         unchanged in shape; its left edge is now fillet_a's inner edge
  <j>_ring_weld        the weld block, ending at N instead of at P_t
  <j>_ring_free        the free block, starting at the CUT instead of at P_t
```

`N` is where the fillet block's inner edge crosses the ring circle, and the split at `N`
is the whole reason there are two fillet blocks per junction rather than one. Measured, at
the shipped radii:

```
  config   blocks  seams   worst min scaled J   worst block      max seam gap
  coarse      11     14         0.3594          rim_ring_free      7.11e-15 mm
  medium      11     14         0.3623          rim_ring_free      1.42e-14 mm
```

and across the admissible gene box — `R_hub` 0.40-3.00, `R_rim` 0.50-3.00, 48 cells at
each config — **48/48 valid AND closed at both configs, worst min scaled Jacobian 0.3569,
worst seam gap 1.42e-14 mm.** `MIN_SJ_TARGET` is 0.2.

**Both halves of "closes" are measured**, because they are different bugs: the node counts
agreeing is a blocking error, the nodes coinciding is a construction error. The seam table
is written in `_seam_table`'s own `(block_a, side_a, block_b, side_b, dk, reverse)` shape.
Two of its fourteen `reverse` flags depend on the genome, against four of the shipped
table's eight — this blocking lays each ring out from `theta_Q` toward the fillet, so
"does the junction's arc ascend" stops being a question. **Two of the `dk` values depend
on the genome instead, which the shipped table's never do, and FINDING 6 is how that was
found.**

## FINDING 2 — PART 9's SHALLOW CUT CANNOT BE CLOSED WITH WHOLE-EDGE QUADS, AND THE REASON IS MECHANICAL

PART 9's block stopped `d` = 0.711 mm (hub) / 0.651 (rim) inside the ring and priced *"a
notch of 7.003 deg at the hub and 2.935 at the rim"*. **That notch has no whole-edge
blocking.** The chain is short and each link was built before it was believed:

1. A cut that stops at depth `d` leaves the ring's free block's left edge — which spans
   the ring's whole depth — with TWO partners: the notch below `B''` and the fillet block
   above it.
2. Splitting the free block at `|B''|` to fix that leaves ITS right edge, at the sector
   boundary, with two partners. The split propagates round the whole ring.
3. And the block it propagates into is a TRIANGLE. The fillet block's inner edge is a
   concentric offset of an arc that is TANGENT to the ring circle, so it is tangent to
   every circle concentric with it — the sliver between the two closes at
   **12.864 deg at the hub and 4.381 at the rim**, a scaled Jacobian of 0.2226 and
   **0.0764**, the latter below `MIN_SJ_TARGET` outright.

Taking the cut to the ring's FAR boundary instead — the hub bore, the rim's outer surface
— splits the ring into exactly two quads and terminates. `L` lands there exactly, and the
dive is radial, both pinned to 1e-12.

## FINDING 3 — WHAT IT COSTS, AND THE PRICE IS NOT THE ONE PART 9 NAMED

PART 9 priced the notch in degrees of arc. The bill that actually arrives is two other
things.

**The ring's radial node count is forced.** The cut carries `n_thick` nodes; it is the
free block's left edge, so that block is `[n_free, n_thick]`; its right edge is the next
sector's weld block's left edge, so the weld is too. **`n_collar_r` and `n_rim_r` are not
used by the filleted blocking at all** — 7 -> 9 at `coarse`, 9 -> 13 at `medium`. That is
the node-count coupling STEP 1a was told to expect, and it is why `fillet=None` needs its
own path rather than a flag.

**And the sector's worst block gets worse, by a factor of 2.2.** The unfilleted sector is
0.7827 / 0.7829 (its own worst is `rim_junction`); the filleted one is 0.3594 / 0.3623.
Still 1.8x above the barrier, and it is a degradation that has to be quoted with what it
degrades from, so the control is measured in the same run by the same instrument.

## FINDING 4 — WHAT BOUNDS `R_hub` IS NOW THE SECTOR, NOT THE BLOCK

PART 9 measured the boundary-layer block clean from 0.05 to 4.00 mm and **it still is**.
What runs out first is the ring's FREE block: at

```
  R_hub = 3.1297 mm
```

the fillet's tangent point `B` has swept past the NEXT sector's corner and there is no
free ring left to block. Bisected, not estimated, because it is the number a gene bound
would have to be written against — and `R_hub`'s box runs to 4.0. The rim never binds:
its footprint fits at every radius swept. **The two statements have to travel together**;
"the block is clean to 4.00" is true and is no longer the whole answer.

## FINDING 5 — THE INNER EDGE HAS TWO CONSTANTS, AND THE RULE THAT PICKS THEM IS IN THE DRIVER

The inner edge is the arc offset by a cubic-Hermite width `w(u)`, then a RADIAL dive from
`N` to the far boundary. Two constants set it, and both are re-derived every run rather
than asserted:

```
  LAYER_ENTRY_SLOPE  -0.45   w'(0), as a multiple of (R + wall) * sweep
  LAYER_END_OFFSET    1.60   w(1), as a multiple of the wall
```

The rule is the argmax of the worst block over the whole box, on a grid the report prints
in full — and the surface is a RIDGE, not a peak: everything from entry -0.35 to -0.60
with end 1.4-1.8 sits within 0.02 of the maximum. Off it, the fall names its own
mechanism. **`entry = 0` is the one that matters**: it makes the inner edge leave the end
cross-section TANGENT to the far flank, which is the junction block's own top edge, and
the junction block becomes a cusp — **min scaled Jacobian 0.0400 against the chosen slope's 0.4272, measured.** Three blocks
meet at that node and 180 degrees has to be shared between two of them; a boundary layer
that takes the full wall takes all of it.

**The dive is radial for a reason that was measured both ways round.** An offset whose `w`
grows to the ring's full depth is a spiral of radius `R + w` about the arc's centre: for
`R` small and `w` large it swings clean out of the material, and **at the gene box's own
floor, `R_hub = 0.4`, it folds the weld block.** The radial dive is clean at that floor
and everywhere above it.

## FINDING 6 — THE RADIUS BOX IS NOT THE GENE BOX, AND MEASURING THE DIFFERENCE FOUND A BUG AND A LIMIT

Everything above sweeps `R_hub` and `R_rim` **at one genome**. `flank_orientation` is a
property of the CENTRELINE, and its own docstring records that of 60 feasible
Latin-hypercube genomes **only 16 have the shipped genome's `(+1, +1)`**. A blocking
measured at the shipped genome alone has been measured on a quarter of the design space,
and §47's phrase "the whole gene box" meant the radii.

Sixteen freshly drawn feasible genomes — `evaluate_design`'s geometric pair, plus the
requirement that the UNFILLETED sector is clean, four per orientation, seeded — say two
things.

**A bug the radius sweep could not reach.** `sector_blocks` lays both ring blocks out in
INCREASING theta whatever the genome does, exactly so that "the next sector" is always
`k + 1`. This blocking lays each ring out from `theta_Q` toward the fillet, so the
sector-closing seam runs to `k + dirn`. Written as `dk = +1` it closes for the shipped
genome and **misses by a whole sector for a flipped one — 12.6 mm at the hub, 50.0 at the
rim.** Fixed, and both halves are under test: the seam closes at `dk = dirn`, and forcing
`dk = +1` re-opens it by more than a millimetre. `build_wheel` takes `(k + dk) % n_spokes`,
so a negative `dk` costs it nothing.

**And a limit that is the honest scope of everything above.** With that fixed, **every one
of the built cells closes, at all four orientations, to 1.4e-14 mm.** The BLOCKS are a
different story:

```
  orientation    drawn   refused   seams close   min scaled J (built)   clear 0.2
  (-1, -1)         4        1         True        0.1216 - 0.2269         1/3
  (-1, +1)         4        1         True        0.1194 - 0.2898         1/3
  (+1, -1)         4        1         True       -0.0508 - 0.2684         2/3
  (+1, +1)         4        3         True        0.1519 - 0.1519         0/1
  ALL             16        6         True       -0.0508 - 0.2898         4/10
```

**Every refusal is the same one** — the hub fillet's tangent point has swept past the next
sector's corner at that genome's own `R_hub` — which is FINDING 4's limit arriving as a
genome property rather than a radius one. And one built cell folds outright, in the
TRIMMED SPOKE rather than in anything new.

**So the blocking is fit for STEP 2 and is not yet fit for the optimizer**, and those are
different requirements rather than degrees of the same one. Step 2 re-runs `make corner`
on ONE filleted mesh at the shipped genome; the optimizer sweeps genomes and would meet
the refusals and the barrier. Nothing above is retracted — the shipped genome's radius box
is 48/48 at both configs — but "48/48 across the box" must not be quoted as "works
everywhere", and there is now a test whose whole job is to stop that.

## WHAT IS UNCHANGED

**Nothing was promoted, `best_solution.json` is untouched and still 2026-08-14, the
default mesh is bit-identical, and no threshold moved.** `wheel_wheel.py` was not edited
at all: every block here is built inside the study from its primitives, and
`sector_blocks(genes, cfg, fillet=None)` is measured as a control in the same run — seven
blocks, all valid, worst 0.7827.

## WHAT STEP 1b IS NOW

A wiring job with no unknown GEOMETRY left in it, and with its scope now named. `sector_blocks` gains four blocks and a
different ring layout under `fillet=`; `_seam_table` **gains eight entries and loses
two** — the two it loses are `weld.i1 ~ free.i0` in each ring, because the fillet block
now separates them and they meet at a single POINT; the ring's radial count comes from
`n_thick`; and `BLOCK_ORDER`, `BLOCK_REGION`, `_edge_sets`, `_node_sets` and
`modelled_area_reference` follow the eleven names in `SECTOR_BLOCK_ORDER`.

**One thing in that list is a trap and is already disarmed.** `_edge_sets` and
`_node_sets` name `hub_tie`, `rim_outer` and `rim_inner_free` by SIDE — `j0`, `j1`, `j0`
— and those names are correct only because `hub_collar_*` runs bore -> ring circle while
`rim_band_*` runs ring circle -> tyre surface. Laying both rings out the same way round
in the filleted blocking is tidier to write and would move three boundary sets with
nothing going red, because a set of the wrong radius is still a set. The blocking here
keeps the shipped order per ring, and a test pins it on the COORDINATES. The seam table STEP 1b copies is in the driver, in the shape
`_seam_table` already uses, and it is under test. What STEP 1b still has to prove is the
thing geometry cannot: that `build_wheel(genes, cfg, fillet=True)` returns a mesh with
zero non-positive Gauss points, that `check_seams` passes on it, and that
`test_axle_drop_is_exactly_12_fold_periodic` holds to 1e-10 on the FILLETED mesh.

**And FINDING 6 sets its scope.** `fillet=True` lands as a MEASUREMENT INSTRUMENT for one
genome, which is exactly what Step 2 needs; it does not land as a path the optimizer may
take, and it must not be wired into `wheel_objective` or the GA on the strength of this.
Making it genome-robust is separate work, and after this PART it has a measured price:
6 of 16 feasible genomes refuse the shipped-radius fillet outright, and 6 of the 10 that
build sit under the barrier.

---

# STEP 1 RECORD, PART 11 — 2026-08-23. STEP 1b IS DONE. `build_wheel(fillet=True)` RETURNS AN ELEVEN-BLOCK MESH THAT INTEGRATES, AND `fillet=None` IS BIT-IDENTICAL. STEP 2 IS REACHABLE FOR THE FIRST TIME.

PART 10 measured the blocking and called what was left "a wiring job with no unknown
geometry in it". This is that job. `wheel_wheel.sector_blocks` grew a second filleted
construction, `_seam_table` grew a filleted twin, and `BLOCK_ORDER`, `BLOCK_REGION`,
`_edge_sets`, `_node_sets`, `quality_report`, `area_report` and `mesh_coords` all stopped
being written as though there were one blocking.

## THE ACCEPTANCE CRITERIA, EACH MEASURED

```
  config   nodes            elements        min scaled J     seam error      non-positive
                                            (assembled)                      Gauss points
  coarse   21012 -> 26196   4704 -> 5952    0.7822 -> 0.3517   3.06e-14 mm        0
  medium   53124 -> 66468  12288 -> 15552   0.7826 -> 0.3575   3.24e-14 mm        0
```

and `test_axle_drop_is_exactly_12_fold_periodic` — Step 1's own named check, the one
`tests/test_wheel_fea.py` records as exercising the mesh, the seams, the sector indexing,
the load and the solve at once — **holds on the FILLETED mesh through a real solve:
1.016e-11 at phase 0 and 7.492e-12 at phase 7, against the 1e-10 the arc asked for.**

`min scaled J` here is `quality_report`'s, over assembled Q9 elements, and it is a
slightly different instrument from PART 10's per-cell 0.3594; both say the same thing
about the same block, `rim_ring_free`, and both clear `MIN_SJ_TARGET` by 1.8x.

## AND THE CONTROL, WHICH IS THE HALF THAT COULD HAVE GONE WRONG SILENTLY

**`fillet=None` is bit-identical.** Coordinates, connectivity, all three node sets, all
three edge sets and the seam error, hashed at `smoke`, `coarse` and `medium` against the
same build from the previous commit: **identical, every one.** That is not a formality.
The block order and the boundary sets are now chosen per mesh rather than being module
constants, and both of those are ways for the default wheel — which every study, every
gate and every committed artifact describes — to move by something too small to notice
and too large to ignore.

## THE BUG THE PORT FOUND, AND HOW IT ANNOUNCED ITSELF

`Q` is the ring blocks' other corner. Under `uncap` it is where the FAR FLANK crosses the
ring circle, not the centreline endpoint — `sector_blocks` has read it that way since
§38 — and the first port of the blocking took the centreline endpoint unconditionally.
Nothing raised. What caught it is that **the study's own numbers stopped reproducing**:
the sector-fit limit moved 3.1297 -> 3.4836 mm and the worst block 0.3594 -> 0.3592, and
those were committed values with a plan section behind them. Fixed, and both reproduce
exactly.

*And the fix has a second witness.* The filleted mesh models **+8.7625% (coarse) /
+8.6965% (medium)** more area than the unfilleted one, against **§24's 8.77%** for the
fillets' share of the part — measured on the CAD solid, by mass, three arcs ago and by a
completely different computation. With the bug in place that number read 8.06%. The
agreement is not proof and should not be quoted as one — a 2-D area fraction equals a mass
fraction only for a uniform extrusion, which this part is — but two independent paths
landing within 0.01 of a point is worth more than either alone.

## TWO FILLETED CONSTRUCTIONS, AND WHY BOTH STAY

`sector_blocks(..., fillet_blocking=)` selects.

  `"sector"`, the default, is PART 10's eleven blocks.

  `"spoke"` is PART 3's — the arc on the spoke block's own flank edge, which §47
  retired as a mesh. **It is kept because `make fillet` measures it**: PART 6's usable
  window of 0.12-0.24 mm at `coarse` is a statement about THAT geometry, and deleting the
  geometry would make the measurement unreproducible while leaving the table in the plan
  file. `studies/study_fillet_fold.py` and its tests now name it explicitly, and its
  artifact came back identical apart from its wall-clock field, so it is not re-committed.

The sector blocking **refuses a zero radius at either end**, where the spoke one allows
it. Not an oversight: the re-cut moves the spoke's end, the junction's left edge and both
ring blocks together, so "no fillet at this end" is a different blocking rather than the
same one at `R = 0`.

## THE TWO THINGS THAT NOW REFUSE RATHER THAN ANSWER

**`mesh_coords` and `coord_fn` refuse a filleted mesh.** They rebuild the sector WITHOUT
`fillet` and index it with `mesh.owners`; a filleted mesh has 26196 owners against 21012,
so the silent answer was a different mesh's coordinates gathered through this one's index.
That is worse than a wrong number and worse than a crash, because it is a *plausible*
number. `WheelMesh` carries `fillet` for the same reason it carries `uncap`.

**`area_report` withholds its reference for a filleted mesh.** `modelled_area_reference`
derives its region from the exporter's geometry, which has no fillet, so
`error_vs_modelled` would book the fillets' 8.76% as a discretisation residual against a
reference that is otherwise good to 2e-4. Withheld with a reason; the measured half is
still returned. Making the reference fillet-aware is real work and is **not** a closed
form — the fillet's legs are a spline and a circle, not two straight lines, so the
inscribed-wedge formula does not apply — and it is ranked, not done.

## THE STUDY NO LONGER KEEPS A COPY

`studies/study_fillet_block.py` built the eleven blocks itself in PART 10, which was right
then and is exactly the drift its own docstring warns about now that the construction
ships. It calls `wheel_wheel.filleted_sector`, `_filleted_sector_blocks` and
`_seam_table_filleted` instead, and keeps what it is for: the verdict on one cell, the
sweeps, the controls, and the two profile constants' re-derivation — which is why the
module exposes `entry` and `end` at all. **Every measured number in the regenerated
report is unchanged**; the only diff is the wording of six refusal messages, which are now
the module's.

## WHAT THIS DOES NOT DO

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, no threshold
moved, and the default mesh bit-identical.** `fillet=` is opt-in and nothing in the tree
passes it except this arc's own drivers and tests.

**And PART 10's scope stands unchanged.** The filleted mesh is a MEASUREMENT INSTRUMENT
for one genome. 6 of 16 feasible genomes refuse it at their own radii and 6 of the 10 that
build sit under the barrier, so it must not be wired into `wheel_objective` or the GA —
and `mesh_coords` refusing is now the mechanical guarantee of that rather than a note.

## STEP 2 IS REACHABLE

For the first time in the arc. It re-runs `make corner` on a filleted mesh and asks
whether the junction corner's peak still diverges under refinement once the corner is
rounded — which is the question the whole arc exists to answer. It deserves its own
session: it re-ranks the arc, and PART 8's chain says the peak is on `rim:P_c`, which this
fillet does not reach.

---

# STEP 1 RECORD, PART 12 — 2026-08-23. STEP 2 IS RUN. THE FILLET REMOVES THE SINGULARITY AT BOTH CORNERS IT REACHES AND STEP 2's HEADLINE IS STILL NOT DELIVERED, EXACTLY WHERE PART 8's CHAIN SAID IT WOULD NOT BE — AND THE THING THAT MOVED IS THE DEFLECTION.

PART 11 made Step 2 reachable. This is Step 2: the same ladder, the same driver, one flag.
`make corner-fillet` costs 22 s and writes `studies/study_corner_singularity_fillet.json`
next to the unfilleted `studies/study_corner_singularity.json`, which was refreshed from
the same commit so the two are one instrument rather than two.

## THE ANSWER, IN THE ORDER THE ARC ASKED THE QUESTIONS

**Step 2 named three things it would look for. None of the three came back as a plain
yes, and the shape of each answer is the finding.**

```
  the claim                                          verdict
  the peak stress stops diverging                    NO    the peak is on `rim:P_c`
  the per-corner rates move toward non-singular      SPLIT  `P_t` completely; `P_c` not at all
  `test_peak_stress_diverges...` starts failing      NO    and it should not — see below
```

## THE HEADLINE IS NOT DELIVERED, AND THE REASON WAS PREDICTED AND IS NOW MEASURED

`rim:P_c` still carries the wheel's global maximum at every rung from `coarse` up, located
by `argmax` over the whole Gauss field rather than asserted — **25.0 um away at `fine`** —
and it still diverges:

```
  quantity          peak MPa   smoke    coarse   medium    fine     d(N)/d(N-1)
  global max   unfilleted      48.21    73.73    91.15   108.57       +0.999
  global max   FILLETED        28.85    37.82    47.31    59.31       +1.264
```

That is PART 8's chain arriving intact: Step 2's success condition is a non-divergent peak;
the peak is on `rim:P_c`; `rim:P_c` is the END CAP's corner, not the part's; this fillet is
tangent to `P_t` and does not reach it. **Its wedge is unmoved — 271.02 deg unfilleted
against 270.85 filleted** — which is what licenses comparing the two columns at all, and is
pinned.

*One thing did move and it should not be over-read.* `rim:P_c`'s magnitude fell 45% (108.57
-> 59.31 at `fine`) because the fillet stiffened the whole wheel, and its RATE steepened
(-0.4092 -> -0.4877, ratio 0.999 -> 1.264). The rate is not comparable across the two
blockings: the filleted ring's radial node count comes from `n_thick` rather than
`n_rim_r`, so the mesh in the near field of `P_c` is a different mesh. The wedge is the
thing that sets the exponent and the wedge is unchanged. **The claim is "still diverges",
not "diverges faster".**

## AND THE SINGULARITY AT `P_t` IS GONE. NOT REDUCED — GONE

`P_t` is not a corner on the filleted body. It is a point in the material's INTERIOR:

```
  corner    wedge deg                 kind                node gap um
            unfil -> filleted
  hub:P_t   321.13 -> 360.00     re-entrant -> interior      134.8
  rim:P_t   321.32 -> 360.00     re-entrant -> interior      103.9
  hub:P_c   268.08 -> 268.21     re-entrant -> re-entrant      0.0
  rim:P_c   271.02 -> 270.85     re-entrant -> re-entrant      0.0
```

and the field at the same place stops running away:

```
  probe                 peak MPa across the ladder            d(N)/d(N-1)   verdict
  hub:P_t  unfilleted    16.71   50.03   66.59   85.93           +1.168     diverges
  hub:P_t  FILLETED       5.37    5.49    5.49    5.51           (noise)    settled
  rim:P_t  unfilleted     12.11   37.46   48.45   60.69           +1.113     diverges
  rim:P_t  FILLETED        3.81    3.59    3.96    3.98           +0.048     settling
```

**The successive differences are the instrument here, not the log-log slope.** The slope
was written for a ladder on which every probe was singular; a bounded quantity still
approaching its limit has a nonzero slope over three rungs. The differences are
unambiguous: every one of the four unfilleted corners holds its ratio at **0.999 or above**,
and the filleted `P_t` pair's last increments are 0.014 and 0.018 MPa on values of 5.51 and
3.98 — **0.25% and 0.44% of the value**. Both readings are in the artifact and the printed
table carries both columns; neither replaced the other.

## THE NUMBER THE FILLET DOES HAVE, WHICH IS A MAXIMUM OVER A SURFACE AND NOT AT A POINT

Three sampled points are not a verdict on a surface, so the driver takes the peak over the
whole fillet arc — a tube of fixed radius about the arc, defined analytically from the
fitted circle so the REGION does not move when the mesh does:

```
  fillet surface   peak MPa   smoke   coarse   medium    fine    ratios          -> limit
  hub (R 0.6636)              28.85   33.11    34.94    35.86    0.430, 0.507    36.8 MPa
  rim (R 3.0000)              12.90   15.10    15.87    16.05    0.347, 0.242    16.1 MPa
```

**That is the arc's central claim, delivered locally.** The sharp corner's stress at `fine`
is 85.93 / 60.69 MPa and is a mesh setting; the rounded one's is 35.9 / 16.1 MPa and is a
number. The last rung moves it by 2.6% and 1.2%.

*The three POINT probes on the arc are noisier than the surface and the surface is what to
quote.* `hub:arc`'s differences run 6.01, 0.56, 1.67 MPa — a ratio of 2.96 off a small
middle term — because a max in a ball around one node is partly sampling the mesh. Nothing
on the arc is re-entrant: `A` and `B` measure 183.0-184.5 deg and the arc midpoints 188.1-188.3,
which is 180 plus one node's worth of the arc's turn, and `N` — where four blocks meet — is
interior at 360.00. **The construction introduces no corner of its own**, which was the one
way this could have gone badly wrong quietly.

## THE RESULT NOBODY WENT LOOKING FOR: THE FILLETED WHEEL DEFLECTS 38% LESS, AND ITS DEFLECTION CONVERGES

```
  axle drop mm      smoke     coarse    medium     fine     spread over coarse..fine
  unfilleted      1.499486  1.551645  1.562981  1.570505          1.216%
  FILLETED        0.902237  0.962456  0.963816  0.962579          0.141%
```

**Two separate findings and they are the arc's own reasons 2 and 4.**

*The magnitude.* **`-37.97%` at the shipped radii, at `coarse`.** Stated the other way
round, because that is the way it bears on every number this project has quoted: **the
unfilleted mesh reports an axle drop 61.2% HIGHER than the filleted one.** This project's
FEA has been solving a wheel without its fillets and quoting the deflection of one; §24
priced the fillets at 4.406 g and 8.77% of the part by mass, and this says what that mass
is worth structurally. `R_rim = 3.0` mm puts a tangent length of 8.5 mm on a 41 mm flank,
at the root, where the bending moment is — a cantilever's root 20% carries 49% of its tip
deflection, so the order is right.

*And it is reason 4 of this arc answered, with the qualifier that matters.* §31's finding
was that sweeping `R_hub` across its whole box leaves the solved wheel **bit-identical** —
hub share and axle drop to seventeen digits at every point — so two of the fourteen genes
steer on a `Kt` correlation and a buildability barrier with no mechanical feedback at all.
On the filleted mesh they have feedback, and it is large and monotone: the control table
below moves the axle drop from 1.567 to 0.765 mm across the radius ladder. **They are still
invisible to the OPTIMIZER**, which may not build this mesh (PART 10's scope), so what
changed is that the feedback now exists and is measured, not that the genes are live.

*The convergence.* The unfilleted axle drop is still climbing at `fine` and spans 1.22%
over `coarse..fine`; the filleted one spans **0.141%** and is flat from `coarse` up. That is
the singular field polluting a global functional, which is the mechanism §29 spent 95
minutes failing to identify from the convergence order.

**AND THE SCOPE OF BOTH NUMBERS HAS TO TRAVEL WITH THEM.** This is
`solve_wheel`'s `axle_drop_mm` — ONE phase, LINEAR kinematics, one genome, the corner
driver's own cheap solve. §33's item 3 registered that caveat for this exact driver before
Step 2 was run: it mentions `kinematics` zero times and calls `fem.solve_wheel(mesh)` bare,
so it takes the kernel default, and *"any claim it makes about stress MAGNITUDES or about
`R_hub`/`R_rim` sensitivity inherits the linear kernel and must say so"*. **The -38% is a
magnitude and an `R` sensitivity, so it says so.** What §33 also says survives is the part
of this record that is about EXPONENTS — Williams' wedge solution is itself linear-elastic
and the driver reproduces the 360 deg crack at exactly 0.5 — so *"`P_t`'s singularity is
gone and `P_c`'s is not"* does not inherit the caveat. PART 7 measured SVK moving the peak
magnitude by 4.3-4.5% in the solve on the uncapped mesh; whether it moves a 38% deflection
change by more is unmeasured. The ±0.3% band §29 retired belongs to the GATE's QoI,
`axle_drop_mean_mm`: eight phases under both kinematics, which is `make gci` at 95 minutes
and 20.6 GB, re-run post-flip only three sections ago at §49. **So this does not earn the
absolute band back.** What it says is that the mechanism §29 named as the obstacle —
a singular corner polluting the functional — is measurably the obstacle, and that it is
gone on the filleted mesh for the cheap single-phase surrogate. Whether the gate's own QoI
follows is `make gci --fillet`, and it is ranked rather than run.

*And the fillet being measured is the PART's fillet, at the PART's radius, on the PART's
corners — checked against the shipped manifest rather than assumed.*
`export/wheel_step_manifest.json` (genome `09e8188`, 2026-08-14) reports both junctions
built at exactly what was requested:

```
  junction   r_requested   r_built    edges     worst wedge deg   Kt error
  hub          0.663606   0.663606   24 / 24        322.0           0.0%
  rim          3.000000   3.000000   24 / 24        320.0           0.0%
```

**Those wedges are `P_t`'s** — the mesh measures 321.13 and 321.32 — and `P_c`'s 268 / 271
appears nowhere in the manifest. So the exporter fillets exactly the corner family this
blocking fillets, at exactly the radius it uses, and neither body rounds `P_c`. That is
the reason the mesh's remaining artefact corner is an artefact: **the part does not have
it**, which is `UNCAP_PLAN` Step 2's finding and the tri-block's whole warrant.

The mass agrees too: PART 11 recorded the filleted mesh modelling **+8.76% more area**
against **§24's 8.77%** for the fillets' share of the part, measured on the CAD solid by
mass three arcs ago. PART 11's caveat on that agreement stands and is not weakened here —
a 2-D area fraction equals a mass fraction only for a uniform extrusion, which this part
is — but taken with the manifest it is why the 38% is a statement about this wheel and not
about a shape the mesher made up.

*The exporter is not always so obliging and the check is not a formality.* `kt_report`'s
own docstring records the rim once shipping twelve of its twenty-four corners square while
reporting `kt_error_pct = +0.0%`. At this genome it did not, and this run is at this
genome.

## THE CONTROL, WITHOUT WHICH NONE OF THE ABOVE IS SAFE

A 38% shift between two meshes is equally well explained by the fillet's stiffness and by a
different model. **The filleted blocking takes an explicit radius pair, so drive it toward
zero and demand the unfilleted wheel back.**

```
  R (both ends) mm    axle drop mm    vs unfilleted      (abridged; full ladder in the
  unfilleted            1.551645                            artifact, 13 radii)
  0.02                  1.567343         +1.01%
  0.05                  1.549024         -0.17%
  0.10                  1.519177         -2.09%
  0.40                  1.384217        -10.79%
  0.80                  1.236657        -20.30%
  2.00                  0.936259        -39.66%
  3.00                  0.764794        -50.71%
  genome's own pair     0.962456        -37.97%
```

It is a LIMIT and not an identity — `sector_blocks` refuses `R = 0` at either end, because
the re-cut moves four blocks and "no fillet here" is a different blocking rather than this
one at zero — so the residual at the smallest radius is asserted rather than hidden.
**-0.17% at R = 0.05 mm**, and the `+1.01%` one rung below it is the boundary layer getting
thinner than `coarse` resolves, which is a statement about the control. Monotone in `R`
everywhere above the floor. **The two blockings solve the same wheel.**

## WHAT THIS DOES NOT DO, AND ONE TEST THAT DELIBERATELY DOES NOT CHANGE

**`test_peak_stress_diverges_but_the_field_converges` still passes and was not touched.**
Step 2 anticipated it failing — *"either fillets were added (good, update this test)"* — and
it does not, for a reason that is correct rather than incidental: it calls
`ww.build_wheel(genes, cfg)` bare, so it measures the DEFAULT mesh, which is unfilleted and
bit-identical. Updating it would be asserting a filleted result against an unfilleted
measurement.

**Step 2's part C is not run.** "What the optimizer actually sees" is the Gauss-weighted
p-norm, and measuring it on a filleted mesh means `make gci` — 95 minutes and 20.6 GB — on a
mesh that PART 10's scope says the optimizer may not take. Ranked, not done.

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, no threshold moved,
and the default mesh bit-identical.** The refreshed unfilleted artifact was diffed field by
field against the committed one: **every measured value identical**, the only additions
being `fillet`, `global_peak`, `kind`, `node_gap_mm` and the successive-difference columns.

**And PART 10's scope is unchanged and is now under test.** `fillet=` remains a MEASUREMENT
INSTRUMENT for one genome — 6 of 16 feasible genomes refuse it at their own radii — so it is
still not wired into `wheel_objective` or the GA, and
`test_nothing_wires_the_fillet_into_the_objective` PARSES five `src/` modules and refuses
any `fillet=` or `fillet_blocking=` keyword argument that is not `None` — so it stays true
by check rather than by nobody remembering. It parses rather than greps because both greps
that would do the job here are wrong: `wheel_objective` has a LOCAL `fillet = jnp.sum(...)`
that is the fillet-margin barrier and has nothing to do with the mesh, and `wheel_fea`
passes `R_hub_fillet=` to the EXPORTER, where the fillets have always been geometry.

## ONE BUG THE PORT FOUND, AND IT WAS THE MEASUREMENT THAT FOUND IT

`measured_wedge_deg` took the nearest node of ANY kind. A Q9 midside contributes a straight
180 deg and is skipped by the angle sum, so a search that lands on one returns **0.00 deg
and zero incident elements** — a number shaped like a measurement. It could never fire while
every probe was an exact vertex, which the four unfilleted corners are. On the filleted mesh
`P_t` is interior and not a node at all, and `rim:P_t` reported **0.00 deg against
`hub:P_t`'s correct 360.00**, purely on which of two equally-near nodes came first.
Restricted to Q9 vertices; the unfilleted report is unchanged in every field.

## WHAT STEP 2 LEAVES

The chain PART 8 closed is confirmed end to end, and the ranking it implies is unchanged:
**the rim tri-block is still the only measured path to Step 2's headline**, because it is
what removes `rim:P_c`. What is new is that the fillet's own half is no longer a promise —
`P_t`'s singularity is gone, the fillet surface has a peak that is a number, and the
deflection moved 38% and started converging. Two of this arc's four reasons for existing
are answered; the one the headline names is not.

---

# STEP 1 RECORD, PART 13 — 2026-08-23. THE LAYER PROFILE RE-DERIVED AGAINST GENOMES CLEARS THE BARRIER FOR NINE OF NINE, NOT FOUR OF NINE — AND IT IS MEASURED RATHER THAN SHIPPED, BECAUSE ADOPTING IT SPENDS PART 12's OWN RESULT FOR NOTHING YET BOUGHT

PLAN.md ranked "make the filleted sector blocking genome-robust" (§48) as item 2 behind
the tri-block's own successor. This is that item, worked from PART 10's own genome
sweep: of the ten drawn genomes that build at their own radii, six sit under
`MIN_SJ_TARGET`, always at `rim_ring_free` or a neighbour — the same failure PART 10
FINDING 5 traced to `LAYER_ENTRY_SLOPE`/`LAYER_END_OFFSET` being an argmax over ONE
genome's radius box. `studies/study_fillet_block.py` gained
`sweep_layer_profile_genomes`, which runs the same argmax against the genomes
`sweep_genomes` already drew instead, and it is under test in
`tests/test_fillet_block.py`.

## THE RIDGE EXISTS, AND IT IS NARROWER THAN PART 10's

```
  entry \ end     0.50    0.60    0.70    0.80    1.00    1.30    1.60
       -0.30     0.072   0.065   0.058   0.052   0.040   0.024   0.009
       -0.45     0.197   0.190   0.184   0.177   0.164   0.139   0.119  <- shipped
       -0.60     0.228   0.238   0.221   0.206   0.182   0.154   0.132
       -0.70     0.217   0.234   0.236   0.220   0.194   0.164   0.141
       -0.75     0.213   0.229   0.244   0.227   0.200   0.169   0.145  <- argmax
       -0.80     0.208   0.227   0.239   0.235   0.206   0.174   0.150
       -0.90     0.198   0.218   0.231   0.244   0.219   0.185   0.159
```

worst min scaled Jacobian over ten genomes (nine drawn plus the shipped one at its own
radii), one excluded and named below. The argmax sits at `entry = -0.75, end = 0.70`,
worst 0.2444 — against the shipped pair's 0.1194, which is what "six of ten under the
barrier" was measuring. **At the argmax, all nine clear; at the shipped pair, four of
nine do** (the tenth, shipped-genome cell clears at both, always).

## ONE DRAWN GENOME IS NOT ON THAT RIDGE AT ALL, AND IT IS A DIFFERENT BUG

One genome's worst block is the TRIMMED SPOKE, at `min_scaled_jacobian = -0.0508` —
sign-flipped, not merely thin. Neither `entry` nor `end` moves it, at all, because the
spoke block is `sample(s_grid, eta_grid)` directly and is built before either constant
is consulted — confirmed bit-identical across three widely separated profiles in
`test_a_spoke_fold_genome_does_not_move_with_the_layer_profile`.

Traced to source: this genome's UNFILLETED flank has a near self-intersection around
`s = 0.051` on one edge (`eta = -1`) — a fine 2000-point resample finds the shoelace
sign flip there, magnitude -0.405. **The shipped, untrimmed 97-station grid over
`[0, 1]` steps over it** — its nearest stations (s = 0.042, 0.052) bracket the dip
without any corner actually crossing zero, so `sweep_genomes`' own "unfilleted sector
is clean" feasibility gate passes this genome. **The fillet's TRIM moves the grid to
`[s_A(hub), s_A(rim)]` instead**, and for this genome that puts a station almost exactly
on top of the dip, where the untrimmed grid did not. The fillet did not create this
defect — it re-sampled a flank that already had it, at a resolution the shipped mesh's
own station spacing happens not to expose. Filed rather than fixed: strengthening
`sweep_genomes`' feasibility gate to catch this class of defect is a different piece of
work than either successor PLAN.md ranked, and is not this arc's to do unprompted.

## MEASURED, NOT ADOPTED — AND THE REASON IS A CONCRETE COST, NOT CAUTION

The genome-diverse pair was NOT made `FILLET_LAYER_ENTRY_SLOPE`/`FILLET_LAYER_END_OFFSET`.
`study_fillet_block.py` carries it as `GENOME_ROBUST_ENTRY`/`GENOME_ROBUST_END` instead,
reported and tested but not wired anywhere `build_wheel` reaches. The reason is PART 12's
own result, re-checked against the trade before assuming it survives:

```
  quantity (shipped genome, coarse..fine spread)      PART 10 pair    genome-robust pair
  worst block min scaled Jacobian (shipped R)              0.3569              0.2221
  filleted axle-drop spread, coarse..fine                   0.141%              0.513%
```

PART 12's deflection-convergence finding — *"the filleted one is flat from `coarse` up
... 0.141%"* — was checked against the +-0.3% band this arc exists partly to earn back.
At the genome-robust pair that spread **more than triples and crosses back over that
band**, on the SAME shipped genome PART 12 measured. Nothing today is wired to the
sector blocking that would collect the genome-robustness this pair buys — the hub
sector-fit refusal (PART 10 FINDING 4/6) is untouched by either choice, so six of
sixteen feasible genomes still refuse outright regardless — so adopting it now would
spend a working, already-published result on a benefit nothing yet reaches. `blend 0.0`
in §53 set the precedent for this call: measure it, report it, do not ship it until
something needs it.

## WHAT WOULD CHANGE THE CALL

Two things, and neither happened here: (1) the hub sector-fit refusal gets its own fix
or its own exposed margin, so genome robustness stops being partial the moment this
profile is adopted, or (2) something is actually wired to consume the genome-diverse
path — which PLAN.md's own ranking still puts behind item 2 finishing. Until one of
those is true, re-deriving the SAME argmax again would reproduce this table; it would
not change the decision.

## WHAT IS UNCHANGED

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, `FILLET_LAYER_
ENTRY_SLOPE`/`FILLET_LAYER_END_OFFSET` untouched at PART 10's values, and the default
mesh (and the default filleted mesh) bit-identical** — `studies/study_corner_
singularity_fillet.json` was regenerated and reproduces PART 12's numbers to the bit.
The only artifact that moved is `studies/study_fillet_block.json`, which gained the
`profile_genomes` section above and nothing else; every previously-committed field in it
reproduces unchanged.

## WHAT PART 13 LEAVES

Item 2 in PLAN.md's ranking ("make the filleted blocking genome-robust", §48) is now
split into two named, independently-priced pieces instead of one: the QUALITY half
(`rim_ring_free` and neighbours sitting under the barrier) has a known, tested fix that
is not yet worth shipping, and the REFUSAL half (the hub sector-fit limit) is
untouched and is where the next unit of work on this item belongs — it is the one that
still costs six of sixteen genomes outright, at either profile. A third, unrelated
finding — the flank near-self-intersection one genome's trim exposed — is filed above
rather than folded into either.

---

# STEP 1 RECORD, PART 14 — 2026-08-23. THE REFUSAL HALF HAS A FIX, IT IS ONE LINE OF ARITHMETIC AGAINST A NUMBER THIS FILE ALREADY COMPUTED, AND IT RE-PRICES PART 13'S DECISION

PART 13 split PLAN.md's item 2 into a QUALITY half and a REFUSAL half and said the
refusal half "is where the next unit of work on this item belongs". This is that unit.
It also named exactly what would change its own call — *"(1) the hub sector-fit refusal
gets its own fix or its own exposed margin"* — and both of those turn out to be the same
number.

## THE REFUSAL WAS ALREADY PREDICTABLE AND NOBODY HAD ASKED IT TO PREDICT

FINDING 4 bisected `sector_fit_limit`: the radius at which the fillet's tangent point
reaches the NEXT sector's corner and the ring's free block runs out of span. FINDING 6
then found six refusals of sixteen and named their mechanism as that same limit "arriving
as a genome property rather than a radius one". What was never done is the obvious next
step — compute the limit **per drawn genome** and compare it to that genome's own `R_hub`.

`sector_fit_margin` does. On the committed draw at `coarse`:

```
  R_hub   limit   margin    binds   built          R_hub   limit   margin    binds   built
 0.7278  3.1846  +2.4569    False    True         0.7737  1.2503  +0.4766    False    True
 0.6021  3.6500  +3.0480    False    True         0.7065  3.8477  +3.1412    False    True
 0.8888  0.7220  -0.1668     True   False         0.5635  6.4615  +5.8980    False    True
 0.9002  1.5617  +0.6615    False    True         1.0398  0.9759  -0.0640     True   False
 0.8029  1.6434  +0.8405    False    True         1.1008  1.3555  +0.2547    False    True
 2.1081  1.7410  -0.3671     True   False         2.0455  1.6816  -0.3639     True   False
 0.9790  4.5276  +3.5486    False    True         0.5533  0.0699  -0.4833     True   False
 0.4411  0.5904  +0.1494    False    True         1.7547  1.5901  -0.1646     True   False
```

**It classifies 16 of 16.** Every genome whose own `R_hub` exceeds its own limit refuses;
no other one does. That makes the refusal a *feasibility number computable before any
block is attempted* rather than an exception caught afterwards, which is what a gate needs
and what `evaluate_design`'s `x_order`/`hub_overlap` pair already are.
`the_hub_margin_predicts_every_refusal` is a self-check, and
`test_the_hub_margin_PREDICTS_the_refusal_rather_than_explaining_it` re-derives it rather
than reading the artifact.

**The margins are not marginal.** They run from **-0.4833 mm to +5.8980 mm** across
sixteen genomes that all pass the same feasibility filter, and one genome's limit is
**0.0699 mm** — a sector with essentially no room for a hub fillet at all. This is a
genuinely wide feature of the design space, not a boundary effect.

## AND THE FIX IS THE SAME NUMBER USED THE OTHER WAY

`clamped_radii` pulls each radius back to a fraction of its own limit.
`sweep_sector_fit_clamp` crosses that with both layer profiles:

```
  profile         clamp    built   clears 0.2   clamped h/r   min scaled J range    median
  shipped          none    10/16       4/16         0/0        -0.0508...+0.2898    0.1780
  shipped          0.95    16/16       8/16         6/1        -0.0508...+0.2898    0.2081
  genome_robust    none    10/16       9/16         0/0        -0.0508...+0.4592    0.2872
  genome_robust    0.95    16/16      15/16         6/1        -0.0508...+0.5156    0.2915
```

**The refusal half closes completely: 10/16 built becomes 16/16.** At the shipped profile
that is 4/16 clearing the barrier becoming 8/16 — the refusal half was worth four genomes,
and the quality half is still holding eight back.

**And the clamp is free at the shipped genome.** Hub `R = 0.6636` against a limit of
`3.1297`; the rim has no limit at all at any radius swept. `shipped_is_clamped` is False
and `the_clamp_is_inert_on_the_shipped_genome` is a self-check, so every other number in
this file is measured on a genome the clamp does not touch.

**It is insensitive to its own factor.** 0.99, 0.95, 0.90 and 0.75 all give 16/16, and
0.99..0.90 give the identical quality range. The factor exists only because the limit is
where the free block's span reaches ZERO and a block of zero span is not a block; it is
not a tuned constant, and the test asserts the insensitivity rather than the value.

**A GATE AND A FIX ARE DIFFERENT THINGS AND THIS FILE MUST NOT BLUR THEM.** The `binds`
column is a gate: it costs nothing, it is exact, and it loses the genome. The clamp keeps
the genome and models a **smaller fillet than its genes asked for** — honest for an
instrument sweeping the box, and honest for an optimizer *only if the objective is told
the clamped radius*. If `R_hub`/`R_rim` become live FEA genes (Step 3 item 1) the clamp is
exactly a bound projection, which is standard — but the projected value has to be what is
reported back, or the mesh and the genome describe different parts.

## WHAT THIS DOES TO PART 13'S DECISION

PART 13 declined the genome-robust layer profile for two reasons. The first stands
untouched: at that pair the shipped genome's filleted axle-drop spread more than triples
(0.141% -> 0.513%) and crosses back over the +-0.3% band, and nothing about the clamp
changes that — it is a measurement at the shipped genome, which the clamp is inert on.

The second does not. PART 13 wrote:

> the hub sector-fit refusal (PART 10 FINDING 4/6) is untouched by either choice, so six
> of sixteen feasible genomes still refuse outright regardless — so adopting it now would
> spend a working, already-published result on a benefit nothing yet reaches.

**That premise is now false**, and the prize it was weighed against was too small by more
than half. With the clamp, the genome-robust profile takes the draw from 8/16 clearing the
barrier to **15/16** — the single exception being the genome whose own TRIMMED SPOKE
folds, which PART 13 already isolated and which no choice of `entry`/`end` reaches. Against
"9 of 10 built genomes" measured over a box where six could not participate, this is 15 of
16 over the whole draw.

**The call does not change today, and the reason it does not is now a single reason
instead of two.** The shipped genome's deflection-convergence spread is a real, published
cost and nothing yet consumes the genome robustness — `fillet=` is still not wired into
`wheel_objective` or the GA. But "nothing yet consumes it" is a statement about ordering,
not about worth, and the worth just went up by a factor of two. `test_the_clamp_re_prices_
the_layer_profile_PART_13_declined` pins the re-pricing and pins that the constants have
NOT moved.

## WHAT IS UNCHANGED

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, `FILLET_LAYER_
ENTRY_SLOPE`/`FILLET_LAYER_END_OFFSET` still at PART 10's values, `sector_blocks` and
`build_wheel` untouched and still taking the radii they are given, and no threshold
moved.** The clamp exists as `clamped_radii` in the study and is called by nothing outside
it. The regenerated `studies/study_fillet_block.json` was diffed field-by-field against the
committed one: **purely additive** — the `fit` block on each genome row, the `fit_clamp`
section, and two self-checks. Every previously-committed field reproduces exactly.

`make filletblock` is now ~180 s rather than ~85 s — the margins are 32 bisections — and
the Makefile's help says so.

## WHAT PART 14 LEAVES

1. **The quality half, which is now the whole of item 2.** Eight of sixteen still sit under
   the barrier at the shipped profile with the clamp in place, and the genome-robust
   profile takes that to 15/16 at a cost measured only at the shipped genome. The next
   question is whether that cost can be avoided — a profile derived against genomes AND
   constrained to hold the shipped genome's convergence spread, which is a two-objective
   version of PART 13's argmax and has not been attempted.
2. **Re-derive the profile argmax against the CLAMPED draw.** `GENOME_ROBUST_ENTRY/END` is
   the argmax over ten genomes because six could not build. Sixteen can now. The pair may
   or may not move; it has not been asked.
3. **The trimmed-spoke fold**, unchanged and still the one genome no profile reaches.
4. **The flank near-self-intersection** filed in PART 13, unchanged.

---

# STEP 1 RECORD, PART 15 — 2026-08-23. THE FLANK DEFECT PART 13 FILED HAS A GATE, THE GATE IS THE OPTIMIZER'S OWN FOLD BARRIER, AND THE ONLY REASON IT DID NOT CATCH THIS IS THAT THIS STUDY'S DRAW FILTER DOES NOT ASK IT

PART 13 found one drawn genome whose TRIMMED SPOKE is sign-flipped at
`min_scaled_jacobian = -0.0508`, traced it to a near self-intersection in the
**unfilleted** flank at `s = 0.051`, and filed it: *"strengthening `sweep_genomes`'
feasibility gate to catch this class of defect is a different piece of work."*  PART 14
left it as item 4 and PLAN.md's §57 ranked it first.  This is that unit.

It is one import, and the same shape §57 turned out to have.

## THE GATE ALREADY EXISTED, IS ALREADY CALIBRATED, AND FIVE STUDIES ALREADY USE IT

`wheel_geometry.self_intersection_margin` is `min_s (|1/kappa(s)| - t(s)/2)`: the
clearance before the outward offset passes the centre of curvature and the flank turns
inside out.  Closed form off the Bezier hodograph, no mesh, no build.  Its threshold
`MIN_FOLD_MARGIN_MM = 0.1` is not a guess either — `study_mesh_quality` measured it over
2001 genomes and its docstring carries the sweep, 0.1 being the smallest threshold with
zero missed bad meshes.  `wheel_objective` has had it as a live barrier term (`fold`)
since the objective was rewritten, and `study_gnl`, `study_contact` and `study_wheel_fea`
all gate their draws on `loss["fold"] > 0.0`.

`study_fillet_block.sweep_genomes` and `study_tri_block.sweep_genomes` use the two-term
filter `x_order`/`hub_overlap` and then a mesh-based clause — "the unfilleted sector is
clean".  Neither asks the fold.  **The gate the two blocking studies needed was already in
the tree, already calibrated, and already used by half the studies that draw genomes.**

## IT CLASSIFIES THE BOX, AND ONE-SIDEDLY, WHICH IS THE HONEST SHAPE

```
  genome         fold margin  folds  binds  built    worst block  min scaled J
  [-1.0, -1.0]       +2.9333  False  False   True  rim_ring_free       +0.1216
  [-1.0, -1.0]       +0.2648  False  False   True  rim_ring_free       +0.1698
  [-1.0, -1.0]       +0.1244  False  False  False              -             -
  [-1.0, -1.0]       +3.0257  False  False   True  rim_ring_free       +0.2269
  [-1.0,  1.0]       +0.9291  False  False   True  rim_ring_free       +0.1194
  [-1.0,  1.0]       +0.1299  False  False  False              -             -
  [-1.0,  1.0]       +0.8083  False  False   True  rim_ring_free       +0.2898
  [-1.0,  1.0]       +1.8251  False  False   True  rim_ring_free       +0.1780
  [ 1.0, -1.0]       -0.3436   True   True  False              -             -
  [ 1.0, -1.0]       +2.1106  False  False   True  rim_ring_free       +0.2081
  [ 1.0, -1.0]       -0.0131   True   True   True          spoke       -0.0508   <- PART 13's
  [ 1.0, -1.0]       +1.5729  False  False   True  rim_ring_free       +0.2684
  [ 1.0,  1.0]       +1.9088  False  False  False              -             -
  [ 1.0,  1.0]       +2.7720  False  False   True  rim_ring_free       +0.1519
  [ 1.0,  1.0]       +1.8734  False  False  False              -             -
  [ 1.0,  1.0]       +3.6313  False  False  False              -             -
```

**Two of sixteen describe a part that does not exist.  One of sixteen inverts a block, and
it is in that pair.**  No fold-clean genome inverts anything — that is
`no_fold_clean_genome_inverts_a_block`, and it gates.  The margin also separates cleanly
in this box: the two rejects are at `-0.3436` and `-0.0131`, and the next smallest is
`+0.1244`, so the calibrated 0.1 mm barrier excludes exactly the two and costs no false
positive.

**The converse is NOT claimed and must not be.**  Whether a folded flank SHOWS UP as an
inverted element depends on whether a station lands on the dip, which is a property of the
grid and not of the part — the other folded genome refuses for an unrelated reason (its
own sector-fit limit, §57) and would very likely have built.  A gate that promised
"folds <=> inverts" would be promising something about the sampling.  The gate promises
the direction that is a property of the geometry: **fold-clean => nothing inverts.**

## THE FILTER IT REPLACES IS A PROXY, AND THE LEAK RATE IS MEASURED

Over 20480 draws on the same Latin-hypercube stream this box is drawn from:

```
  pass (x_order, hub_overlap)               1454   folded   514  (35.4%)
  AND the unfilleted sector meshes clean     925   folded    25  (2.7%)   <- the leak
```

The mesh clause is doing real work — it removes 489 of the 514 — but it is a proxy and it
leaks at about one in forty.  The drawn box of sixteen got two, which is that rate.

## WHY A CLOSED FORM AND NOT SIMPLY A FINER GRID

Because the two behave differently under refinement, and the difference is a difference in
kind.  `flank_reversal_mm` is the SAMPLED statement of the same fact — project each flank
node's step on the tangent it came from, take the least, negative means the outline
doubles back.  Audited against the closed form on all 1454 geometrically feasible draws at
2000 points: **1 disagreement, at |margin| = 2.09e-04 mm**, which is two ways of straddling
the same point 500x inside the barrier.  Recompute both at the config's own 1200 points:

  * the closed form moves by at most **1.59e-03 mm** and flips **0** verdicts;
  * the sampled flank **misses 2 of 514 folds outright** — the dip falls between stations.

Read on the two rejected genomes themselves, at the 97 stations PART 13's shipped grid
uses:

```
  [1.0, -1.0]  R_hub 1.7547 (refused)      [1.0, -1.0]  R_hub 0.7065, worst block spoke
   points   closed form  sampled flank      points   closed form  sampled flank
       97     -0.343621     +2.350e-02          97     -0.012575     +5.371e-03
      600     -0.343621     -3.294e-03         600     -0.013091     -4.118e-04
     1200     -0.343621     -2.025e-03        1200     -0.013084     -2.252e-04
     2000     -0.343621     -1.307e-03        2000     -0.013028     -1.381e-04
     4000     -0.343621     -6.883e-04        4000     -0.013090     -6.951e-05
```

**At PART 13's own grid the sampled test declares both folded parts healthy and the closed
form has already rejected both** — the second column is constant to six decimal places
across a 40x refinement while the first changes sign.  That is PART 13's anecdote as a
mechanism: not "the grid happened to step over it", but "any grid can, and this quantity
has no grid in it."  It is also the same failure the draw filter has one level up, which is
why the answer is a closed form and not a finer mesh.

## WHAT THE ARC'S OWN TABLE LOOKS LIKE OVER PARTS THAT EXIST

`fit_clamp_fold_clean` is §57's table run again over the rows the margin keeps — same
function, same genomes, minus the two — so the two columns differ only by the exclusion:

```
  profile         clamp        all 16    fold-clean
  shipped          none 10/16 built  4 clear    9/14 built  4 clear
  shipped          0.95 16/16 built  8 clear   14/14 built  7 clear
  genome_robust    none 10/16 built  9 clear    9/14 built  9 clear
  genome_robust    0.95 16/16 built 15 clear   14/14 built 14 clear
```

**With the gate, the genome-robust profile under §57's clamp clears the barrier on every
genome in the box.**  §54's "15 of 16, the exception being the trimmed-spoke genome" had a
named exception; over parts that exist there is no exception, because the exception was
never this blocking's defect.  §54 already excluded that genome from its argmax BY HAND —
the gate is that exclusion made principled, automatic, and taken before the build rather
than after it.

This does not change the adoption call on the profile.  PART 13's surviving reason — the
shipped genome's filleted deflection-convergence spread going 0.141% -> 0.513% across the
+-0.3% band — is measured at the shipped genome and is untouched by anything here.

## AND WHAT IT IS NOT: THE TRI-BLOCK

`study_tri_block.sweep_genomes` uses the same seed and the same stream and draws **the same
sixteen genomes** — verified, not assumed.  So the fold-negative pair is literally the same
pair there, and the temptation is to reach for the margin as a general difficulty
predictor.  It is not one, and the negative is recorded and gated rather than left implied:

```
  coarse   2/16 fold; they sit at fixed-rule +0.5337 and +0.2104 — both above the barrier
           the WORST cell in the box, -0.9597, has margin +0.1299 mm and folds nothing
  medium   1/16 folds, at +0.4756
           the WORST cell, -1.0000, has margin +2.7720 mm and folds nothing
```

The tri-block partitions the rim JUNCTION region and never touches the offset band, so
this is the expected answer — recorded because the expectation is worth a number, and
because §56's "what makes a region impossible" is still unnamed and one more candidate is
now ruled out rather than untried.

## MEASURED, NOT ADOPTED — AND WHAT "NOT ADOPTED" MEANS HERE IS SPECIFIC

**The draw is unchanged.**  `sweep_genomes` still uses its two-term filter and still draws
the same sixteen genomes, deliberately: applying the gate to the draw would replace two
genomes and move every number §54, §55, §56 and §57 published against this box at the same
time as the gate was being evaluated.  Instead the margin rides on every row as `fold`, and
the box is re-tallied over the survivors in `fit_clamp_fold_clean` — the same function on
the same genomes, so the difference between the two tables IS the gate's price and nothing
else is confounded with it.

Applying it to the draw is now a one-word change (`"fold"` into the filter tuple) with its
cost already measured, and it is filed below rather than taken here.

## WHAT IS UNCHANGED

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, `MIN_FOLD_MARGIN_MM`
untouched, `wheel_objective`'s barrier untouched, `sector_blocks` and `build_wheel`
untouched, and no draw filter changed.**  The gate is inert on the shipped genome by a wide
margin — 14.3655 mm against a limit of 0.10 — which is asserted rather than assumed, for
the same reason §57 asserted the clamp was.  Both artifacts were diffed field-by-field
against the committed ones and are **purely additive**: `study_fillet_block.json` gains the
`fold` block on each genome row, the `fold_gate` section, `fit_clamp_fold_clean`, and four
self-checks; `study_tri_block.json` gains the `fold` block on each row and one self-check.
Every previously-committed field in both reproduces exactly.

`make filletblock` is ~200 s rather than ~180 s and the Makefile's help says so;
`make triblock` is unchanged at ~290 s.

## WHAT PART 15 LEAVES

1. **Apply the gate to the draw and re-derive the box.**  Priced above and not taken: two
   genomes leave, two replacements arrive, and every genome-box number in §54 through §57
   moves at once.  Worth doing as its own unit, where the movement is the subject rather
   than a side effect.
2. **The quality half, still.**  Eight of sixteen under the barrier at the shipped profile
   with the clamp; seven of fourteen over parts that exist.  The two-objective profile
   PART 14 named — genome-robust AND holding the shipped genome's convergence spread — has
   still not been attempted, and re-deriving the argmax against the fourteen buildable
   fold-clean genomes is still the cheap first half of it.
3. **The trimmed-spoke fold is CLOSED as a defect** and reclassified: it was never a defect
   of the blocking, and there is now a number that says so before the build.

---

# STEP 1 RECORD, PART 16 — 2026-08-23. THE TWO-OBJECTIVE PROFILE EXISTS. IT IS ONE CELL OF FOURTEEN, IT BEATS THE SHIPPED PAIR ON BOTH OBJECTIVES AT ONCE, AND EVERY SHORTLIST I TRIED FIRST THREW IT AWAY

PART 13 declined the genome-robust layer profile for two reasons.  PART 14 falsified one
of them.  This part takes the other one apart, and the answer is not the one PART 13, PART
14, PART 15 or the first two-thirds of this part expected.

## FIRST, THE CHEAP HALF: THE ARGMAX IS NOT WHAT IS STALE

`GENOME_ROBUST_ENTRY/END = (-0.75, 0.70)` is the argmax over **ten** cells, which is what
was left after six genomes refused their sector and one was excluded by hand.  §57's clamp
makes the six build and §58's fold margin retires the hand-exclusion, so the same
derivation now runs over **fifteen**: fourteen drawn genomes clamped inside their own
sectors, all fold-clean, plus the shipped genome.

Re-derived, the argmax appears to MOVE — to `(-0.90, 0.80)`, worst 0.2440 against
`(-0.75, 0.70)`'s 0.2430 — and it does not.  **That cell refuses one of the fifteen.**

```
  entry \ end     0.50    0.60    0.70    0.80    1.00    1.30    1.60    refused
        -0.30   0.0723  0.0650  0.0582  0.0518  0.0400  0.0238  0.0091    0 0 0 0 0 0 0
        -0.45   0.1973  0.1901  0.1835  0.1773  0.1643  0.1392  0.1194    0 0 0 0 0 0 0
        -0.60   0.2279  0.2310  0.2190  0.2060  0.1816  0.1537  0.1320    0 0 0 0 0 0 0
        -0.70   0.2165  0.2340  0.2347  0.2201  0.1937  0.1637  0.1406    0 0 0 0 0 0 0
        -0.75   0.2125  0.2285  0.2430  0.2274  0.1999  0.1688  0.1450    1 0 0 0 0 0 0
        -0.80   0.2076  0.2271  0.2394  0.2347  0.2061  0.1740  0.1495    1 1 0 0 0 0 0
        -0.90   0.1981  0.2175  0.2313  0.2440  0.2189  0.1845  0.1586    2 1 1 1 1 0 0
```

**Ranking cells on "the worst over the genomes that BUILT" pays a cell for refusing a hard
genome.**  PART 13's rule, and its own argmax happened to sit where nothing refused, so its
answer was never biased — checked, not assumed.  But the clamped cell set reaches the
steep-entry corner where the layer profile itself starts refusing, which the ten-cell set
never did, and there the bias bites.  `profile_argmax` reports both rules and the gap
between them; the corrected one is `no_refusal`, and on it the re-derivation **reproduces
PART 13's pair exactly**: `(-0.75, 0.70)`, 0.2430 against its published 0.2444.

So the argmax is not stale.  Everything rested on the convergence cost.

## THE EXPENSIVE HALF, AND WHAT IT NEEDED FIRST

PART 13's surviving reason is one number: at `(-0.75, 0.70)` the shipped genome's filleted
axle-drop spread over `coarse..fine` goes 0.141% -> 0.513% and crosses the ±0.3% band.
**That was measured at one alternative pair**, and the profile surface is a broad ridge, so
the question nobody had asked is whether some OTHER cell is genome-robust and stays inside
the band.  Three linear solves per pair — the whole ladder is nine seconds.

It needed the profile threaded to a full `build_wheel`, not just to a sector.
`filleted_sector` already exposed `entry`/`end` for the blocking; `sector_blocks`,
`_sector_coords` and `build_wheel` now take `layer_profile=(entry, end)` for the solve.
`None` is the shipped pair, and
`test_the_layer_profile_pass_through_is_BIT_IDENTICAL_at_its_default` asserts bitwise
equality three ways — omitted, `None`, and the constants passed explicitly — with
`test_the_layer_profile_actually_MOVES_the_filleted_mesh` on the other side so the
bit-identity is not vacuous.

## AND THEN I GOT IT WRONG TWICE, WHICH IS THE PART WORTH KEEPING

**Shortlist 1 — the top eight of the ridge.**  All eight failed the band, 0.460% to 0.512%,
and all eight had a steep entry.  The obvious reading: genome robustness needs a steep
entry, the band forbids one, no two-objective profile exists.  A clean negative, and it was
about to be written down.

**Shortlist 2 — the entry ladder, to check that reading.**  It falsified it immediately:
at end 1.60 *every* entry from -0.45 to -0.90 holds the band (0.121%-0.209%).  The cost is
not carried by entry.  The obvious new reading: it is carried by `end`.

**That was wrong too**, and only enumerating the whole candidate set showed it:

```
   entry   spread over the ends priced          end   spread over the entries priced
   -0.30   0.189%, 1.412%                      0.50   0.500%, 0.516%
   -0.45   0.141%, 0.459%                      0.60   0.460%, 0.462%, 0.474%
   -0.60   0.145%, 0.462%, 0.462%, ...         0.70   0.189%, 0.459%, ..., 0.512%
   -0.70   0.121%, 0.460%, ..., 0.516%         0.80   0.462%, 0.475%, 0.482%, 0.490%
   -0.75   0.135%, 0.474%, 0.482%, 0.512%      1.00   0.110%
   -0.80   0.110%, 0.152%, 0.490%, 0.493%      1.60   0.121%, ..., 0.209%, 1.412%
```

Every entry straddles the band and every end but two straddles it.  **Neither variable
alone predicts the cost.**  The failing set is the MIDDLE of the space — a short end with
an entry steep enough to matter — and it covers almost all of the barrier-clearing region.

Almost.

## THE RESULT

```
  profile                       genome-box floor (15 cells)      convergence, coarse..fine
  shipped        (-0.45, 1.60)   0.1194   under MIN_SJ_TARGET       0.141%   inside the band
  §54's pair     (-0.75, 0.70)   0.2430   CLEARS                    0.512%   OUTSIDE
  (-0.80, 1.00)                  0.2061   CLEARS                    0.110%   inside, and BETTER
```

**`entry = -0.80, end = 1.00` clears the barrier on all fifteen cells, refuses none of
them, and holds the deflection band more tightly than the pair that ships.**  It dominates
the shipped profile on both measured objectives at once: the genome-box floor goes 0.1194
-> 0.2061, +73%, and the convergence spread goes 0.141% -> 0.110%.  Re-derived
independently of the driver before being written here.

It is one cell of fourteen.  **And it is the one every shortlist drops**, because it has
the LOWEST genome-box floor of the fourteen that clear the barrier — a top-k rule over the
ridge ranks it fourteenth of fourteen.  The candidate set is now defined by the criterion
that actually matters rather than by a rank: every cell clearing `MIN_SJ_TARGET` on the
whole clamped fold-clean box while refusing none of it.  `LAYER_PROFILE_CANDIDATES` is that
set, `the_candidate_constant_matches_the_measured_surface` re-derives it every run, and
`study_corner_singularity` prices all fourteen.

## MEASURED, NOT ADOPTED — AND HERE THAT IS A DEFERRAL, NOT A DECLINE

The three previous times this arc wrote "measured, not adopted" it was because adoption
would have traded away a published result.  **This one would not**: nothing measured here
gets worse.  The reason it is not adopted in this part is different and is about the size
of the audit, not the merit of the change.

`FILLET_LAYER_ENTRY_SLOPE`/`FILLET_LAYER_END_OFFSET` are the default for every filleted
mesh the tree builds.  They do not touch `best_solution.json`, the shipped unfilleted mesh,
the optimizer or the exporter — `fillet=` is an instrument and `build_wheel`'s default is
still `fillet=None` — but they are the geometry underneath **every filleted number this
arc has published**: PART 12's 38% deflection reduction and its whole corner ladder, the
arc-surface peaks, `study_fillet_block`'s entire radius box, the uncap table, and the
seam-closure claims at both configs.  Changing them re-dates all of it at once.  This tree
has a memory of exactly that going wrong, and the discipline it produced is that a default
change audits every re-derivation that took it bare.

So the pair is named (`TWO_OBJECTIVE_ENTRY`/`TWO_OBJECTIVE_END`), measured, tested and left
unwired, and the promotion is PART 17 with its own baseline.

## WHAT IS UNCHANGED

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14,
`FILLET_LAYER_ENTRY_SLOPE`/`FILLET_LAYER_END_OFFSET` still PART 10's, `GENOME_ROBUST_ENTRY/
END` still PART 13's, `sector_blocks` and `build_wheel` still returning the same meshes
bit-for-bit at their defaults, and no draw filter changed.**  Both artifacts were diffed
field-by-field: `study_corner_singularity_fillet.json` gains exactly one key (`profiles`)
and changes nothing; `study_fillet_block.json` gains `profile_genomes_buildable`,
`profile_argmax`, `profile_candidates` and three self-checks, and changes nothing.

`make filletblock` is ~270 s; `make corner-fillet` is ~180 s rather than ~22 s, and the
Makefile's help says so.

## WHAT PART 16 LEAVES

1. **PART 17: adopt `(-0.80, 1.00)`.**  The audit is the work, not the decision.  Every
   filleted artifact re-derived and re-dated, PART 12's convergence and 38% figure
   re-measured, the two constants moved, and `test_promotion.py`'s checklist extended to
   cover a layer-profile change the way it covers a genome change.
2. **A finer grid around `(-0.80, 1.00)`.**  It is a grid point of a sweep laid out for a
   different question — ends jump 0.80 -> 1.00 -> 1.30 and entries -0.80 -> -0.90.  The
   band-holding, barrier-clearing region has been located but not resolved, and its best
   point is not known to be this one.
3. **Why the middle of the space fails.**  Every cell with a short end and a steep entry
   spreads ~0.5%; the mechanism is unnamed, and PART 12's own reading — that the fillet's
   convergence comes from removing the corner singularity — does not obviously predict a
   region that gets WORSE than the shipped pair while still being filleted.
