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
