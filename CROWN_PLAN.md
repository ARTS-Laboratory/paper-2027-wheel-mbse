# CROWN_PLAN.md — a crowned rim that steers through its contact patch, on a partly compliant base

**Open arc #11. Created 2026-09-24 from PLAN §206 §7.0, the trade §206 handed back. STEP 0 IS
DECIDED (§5, on the user's delegation, same day). No solve run; one free read of fields already on
disk is recorded at the foot and changed decision 0.3.** This file is the plan as written before any of it runs; records go at the foot, and the plan
above them is not rewritten to match them.**

**CURRENT STATE (2026-09-26): Steps 0–7 are done and recorded (R1–R8). The user passed `64e5068`'s spokes at Step 8's look gate. NEXT: Step 8's 3D run — R9 at the foot has the registered predictions and the exact commands (`studies/probe_3d/q_step8.sh`).**

**VERSION CONTROL** follows `PLAN.md`'s header block, which is the only place the rules are
stated: one commit per finished unit of work on `feature`, `make test` green first, never
while a study driver is mid-write, a study commit carries its artifacts, a promotion is one
atomic commit and never one file, and commits carry no assistant or tool attribution.

---

## 1. THE DECISIONS ALREADY TAKEN (the user, 2026-09-24)

1. **The rim is PARTLY compliant: a crown on top of a THINNER base.** Not the cut crown (the
   rim does the flexing, band 1.14–1.79x, §205) and not the full 1.5 mm base under a 1 mm
   crown (band 1.16x worst, but the part 34.1% too stiff, §206). Somewhere between.
2. **The crown's job is steering, through the contact patch.** Its height is chosen for what
   the patch does, not as a deflection lever. 1 mm was never derived (§200 calls it a design
   constant, asked for directly).
3. **The 2D model gets a thicker band that stands in for the crown**, and the spokes are
   re-optimised against that model. **But not before the rest of this plan is laid out**, the
   rim is chosen, and the stand-in is shown to reproduce 3D.
4. **A smaller crown is measured**, not assumed.

**The standing concern, stated so the plan answers it:** the spokes on `b729e86` look right,
the flat rim does not, and a re-optimisation may give spokes that look worse. §6 is the
answer, and it has a gate the user holds.

---

## 2. WHAT IS ALREADY KNOWN, AND THE FRAME THAT MAKES IT A MAP

**The spokes are identical in every crowned part measured so far.** Flat, cut crown and crown
on top all carry genome `b729e86`; the crown lives in the CAD layer only (§200, §206: the
solver is byte-identical). What changed between the parts is the rim alone.

**The two crowned parts are two points on ONE family.** The cut crown was a 1 mm crown whose
apex sat at r 50.0 over a band starting at `RIM_RADIUS_MM` 48.5: **1 mm crown on a 0.5 mm
base**. The crown on top is **1 mm crown on a 1.5 mm base**. Same arc (R 63.22 mm), shifted
1 mm radially. So the rim is two numbers — **base b** (band thickness at the side faces) and
**crown height h** — and "a thinner base" is a point between measured ends:

```
  (b, h)        what             3D SVK 8-phase mean   band worst / 25 MPa              source
  (1.5, 0.0)    flat, modelled   1.8102 mm  -9.5%      1.31x (OD, 3D SVK)               §204
  (1.5, 0.0)    flat, exported   NOT RUN under SVK     1.03x (OD, phase 0 linear only)  §203 §5
  (0.5, 1.0)    cut crown        2.0852 mm  +4.3%      1.79x (OD); inner face unread    §205
  (1.5, 1.0)    crown on top     1.3178 mm  -34.1%     0.97x OD, 1.16x INNER            §206
```

**Edge OD and apex OD follow from (b, h)** with `RIM_RADIUS_MM` held at 48.5 (so no gene on
disk is reinterpreted — `wheel_fea`'s comment above `RIM_RADIUS_MM` prices moving it):
edge Ø = 97 + 2b, apex Ø = 97 + 2b + 2h. The cut crown is 98 / 100; on top is 100 / 102.

**A HYPOTHESIS, not a finding — the base is the dial on how far the spokes must move.** At h 1
the current spokes put the drop at 2.085 mm with b 0.5 and 1.318 with b 1.5. Linear in b, the
drop hits 2.0 at b ≈ 0.61 — next to the cut crown and its 1.7x band. Band bending goes as
roughly b³, so linear interpolation is crude and the real curve is what Step 2 measures. What
it suggests: **every base stiff enough to fix the band needs softer spokes**, and the thicker
the base, the further they move.

**AND A SECOND HYPOTHESIS, on the user's "spokes should be the compliant mechanism".**
`rim_sweep` in `studies/study_wheel_fea.json` (2D, `medium`, the genome of 2026-08-23, not
`b729e86`) holds the rim's share of the strain energy at 0.28–0.33 across bands 0.9–4.5 mm
while the drop falls 3.7x. If that holds for this genome, a stiffer rim does not hand its
compliance to the spokes — it spreads the load over more spokes, and each spoke flexes less.
The spokes then have to be softer than a naive "rim share moves to the spokes" estimate says.
One old genome, 2D, linear; Step 2's map tests it on this one.

---

## 3. THE STAND-IN: A THICKER 2D BAND — AND WHY IT MUST BE CALIBRATED, NOT ASSUMED

**The 2D mesh already takes the band as a kwarg**: `wheel_wheel.build_wheel(..., rim_outer=)`,
which `rim_sweep` used to thicken the band OUTWARD with `RIM_RADIUS_MM` fixed. So the stand-in
needs no new geometry and moves no gene. **It is not threaded through the objective**:
`wheel_objective` never passes `rim_outer`, so every `build_wheel` call on the scoring path
takes the default 50.0. Threading it is a change to a default's reach, and the cache keys must
carry it (memory: a flipped default turns silent omissions into wrong answers).

**THE OBVIOUS STAND-IN IS THE MEAN THICKNESS, b + (2/3)h** — the crown's segment area over the
face, 14.957 / 22.4 = 0.668 mm at h 1. §206 checked it once and it agreed. **Checked on both
crowned parts, it over-predicts both, and not by the same factor:**

```
                       stand-in     rim_sweep predicts   3D measured, crown alone      over-read
  crown on top         1.5 -> 2.168   -25.6% / -25.9%     about -22.1% to -22.5%        ~1.15x
  cut crown            1.5 -> 1.168   +25.6% / +26.5%     about +16.5% to +17.0%        ~1.55x

  predicted: rim_sweep interpolated linearly / in log-log.  measured: the 8-phase linear
  ratio part/twin (R_lin 0.75518 §206, 1.13474 §205) divided by the exported flat part's
  junction stiffening, 2.60-3.04% (§202-§204) -- so the flat-part baseline, not the twin.
```

§206's "within 1.1 points" compared the stand-in against R_lin, which **includes** the
junction stiffening; the crown alone is 3–4 points smaller. **This is a hypothesis with three
confounds** — `rim_sweep` is a different genome, it is 2D where the measurement is 3D, and the
junction stiffening was measured at phase 0 and applied to an 8-phase ratio. It is enough to
say the stand-in must be FITTED per (b, h) against 3D, and that a thin base under a crown is
where a mean-thickness rule is worst: the thin side faces bend far more than their share of
the mean says.

**Separate question, kept separate:** the stand-in stands in for the CROWN. The plane-stress →
3D offset (−9.34% for the modelled flat body, §204) and the exported junctions (2.6–3.0%
stiffer) are two other offsets the objective does not model. Folding all three into one
fitted thickness would let the band carry errors that belong to the kinematics and the
fillets. **Recommended: fit the stand-in to the crown's ratio only, and handle the other two
as a target correction** — which is §204 §7.0's route question, re-opened. Step 0 decides.

---

## 4. "STEERING THROUGH THE CONTACT PATCH" HAS NO INSTRUMENT YET

The tree's physics is a **purely radial load on flat, rigid, frictionless ground**
(`wheel_requirements` module docstring). No side load, no friction, no camber, no yaw. So
steering itself cannot be measured here. What CAN be measured from the 3D solves this tree
already runs, with nothing new but post-processing:

- **Patch geometry per phase**: length along the rolling direction, width across the face,
  peak and mean pressure. `fe3d.py` already reports patch extents (§205 §2, §206 §3).
- **Patch stability over the stencil**: the cut crown's patch ran 2.2 to 10.2 mm long and
  split into TWO patches at 3.75 (§205 §2); the crown on top gave one strip at every phase
  (§206 §3). A patch that changes shape as the wheel rolls steers differently as it rolls.
- **A pivot (scrub) torque proxy**: M = μ ∫ p · |r − r_c| dA over the patch, Coulomb sliding,
  from the contact pressures already on disk. μ scales it out of every COMPARISON, so the
  ratio between rims is μ-free. This is a proxy for turn-in resistance, not a steering model.

What would need NEW capability, and is out of scope unless Step 0 says otherwise: a
tilted-ground (camber) solve to show the crown keeps the patch centred, and a side-load
solve for lateral stiffness. Both are extensions of `fe3d.py`'s ground plane.

---

## 5. THE STEPS, EACH WITH ITS CHECK

**STEP 0 — DECISIONS. TAKEN 2026-09-24 ON THE USER's DELEGATION** ("go with your
recommendations"). Each can be overruled; none has spent anything yet.

- 0.1 **The steering metric — DECIDED: patch stability is a pass/fail, the scrub proxy is
  the figure of merit, width and length are reported.** One contact patch at every stencil
  phase (the cut crown failed this at 3.75, §205 §2); among rims that pass, a lower scrub
  proxy relative to the flat rim is better. **Camber is out of scope** until the map exists:
  it needs a tilted-ground solve this tree does not have, and the map may make it moot.
- 0.2 **The apex OD cap — DECIDED: Ø102**, what the part has shipped at since §206. With
  `RIM_RADIUS_MM` at 48.5 that is **b + h ≤ 2.5**, and every point on Step 2's grid is legal.
  Ø100 at the EDGES is not required: edge Ø = 97 + 2b is under 100 for every b < 1.5.
- 0.3 **The band — DECIDED: a PRICED TERM on hoop TENSION, not a barrier, not a waiver, and
  NOT von Mises; plus an interlayer check that only 3D can make.** The reason is the record
  at the foot (R1): the part is printed flat, so hoop stress runs along the filaments (the
  strong direction) while the only stress across the layers is `s_zz` — and the 2D kernel is
  PLANE STRESS, so `s_zz` is identically zero in everything the optimiser sees. So:
  - **In 2D (Step 5):** the band report reads the max principal TENSION on both faces,
    spoke-flank zones excluded, against the existing 25 MPa allowable, priced with a knee
    the way `stress_margin` is. Not a barrier: the in-plane margin to the printed strength is
    real (R1: worst 33.85 MPa against 40 MPa printed ultimate on the current part), and a
    barrier would forbid the thinner base the user chose. Von Mises is retired for the band
    because it scores the OD's compression (harmless, and the user's point) as if it were
    tension.
  - **In 3D only (Steps 2 and 8):** max `s_zz` tension in the band, read at every map point
    and gated on the final part. **The interlayer allowable is NOT KNOWN.** This tree has no
    Z-direction strength for PLA; the placeholder is half the in-plane allowable, 12.5 MPa,
    and it is a placeholder — the current part already reads 16.78 against it (R1). **Step
    0.5 replaces it with a measurement.**
- 0.4 **The target correction — DECIDED: the stand-in stands in for the crown only; the 3D
  and junction offsets enter as ONE measured factor on the 2D drop inside the `deflection`
  term**, an explicit argument whose default 1.0 is bit-identical to today. Measured on
  `b729e86` from §204/§205 figures before the descent, re-measured on the result at Step 8,
  which is its falsifier. **Not through the requirements layer's stroke:** in
  `wheel_requirements` the stroke also sets the landing load factor (`Mission.stroke_mm`'s
  docstring), so re-stating the target there would move the LOAD with it.
- 0.5 **NEW — AN INTERLAYER STRENGTH, MEASURED (the user, with a printer).** Tensile coupons
  printed so the pull is ACROSS the layers, same material, nozzle, layer height and
  temperature as the wheel, and a set printed along the layers as the control against
  `wheel_fea`'s 50 × 0.80 = 40 MPa. The ratio between them is the number 0.3's placeholder
  stands in for. Not blocking Steps 1–4; blocking Step 8's gate.
- **Spoke look (§6) — DECIDED: warm start from `b729e86`, no frozen genes, a reported
  distance and the user's gate.** A trust region is the fallback if the gate refuses the
  first result, not the first move: constraining before seeing what the descent wants
  would hide whether the rim choice was the problem.

**STEP 1 — PARAMETRIC (b, h) IN THE CAD LAYER.** `wheel_geometry`'s `CROWN_HEIGHT_MM` and
the exporter's fixed 1.5 mm base become two arguments with today's values as defaults.
*Check:* at (1.5, 1.0) the STEP is byte- or volume-identical to `9f296d0`'s (65.35 g OCC);
at (0.5, 1.0) OCC volume matches §200's cut part (45638.5 mm³, 56.59 g); the solver stays
byte-identical, the §200 probe re-run (88 values, 0 ULP).

**STEP 2 — THE 3D MAP, ON `b729e86`.** A grid in (b, h), spokes held fixed so the rim is the
only variable. Proposed, pending 0.2:

```
            h 0.0          h 0.5          h 1.0
  b 0.5     new            new            MEASURED (§205)
  b 1.0     new            new            new
  b 1.5     new (flat      new            MEASURED (§206)
             exported,
             SVK)
```

Seven new points. Per point, §205's queue: 8 phases, SVK, h 2.0 / hc 0.25, 16–20 GiB and
~8–11 min per phase under a `MemoryMax` scope, so **~75 min per point, ~9 h for seven**,
serial. A cheaper first pass is phases 0 and 3.75 only (3.75 was the worst band phase on both
measured parts). *Read at each point:* drop (8-phase mean); band hoop TENSION and `s_zz`
tension on BOTH faces (`post3d_cyl.py`, R1), each peak classified by distance from the
nearest spoke (§206 §4); patch metrics and the scrub proxy (§4); mass. *Registered before the run:* a predicted drop and band figure per
point, from interpolating the measured ends — so the map tests §2's two hypotheses rather
than just filling a table. The worst band peak at the chosen point gets refined (hc 0.25 →
0.20), as §206 Q5/Q6.

**STEP 3 — CHOOSE THE RIM (the user), from the map.** The map answers: for each (b, h), how
far the current spokes are from the 2.0 mm target, what the band carries, and what the patch
does. The smaller crown (h 0.5) is on the grid by construction.

**STEP 4 — FIT THE STAND-IN.** For the chosen h, fit the 2D band thickness t_eq(b, h) that
reproduces the 3D crown ratio, at the objective's setting (`coarse`, SVK). Fit on the grid
points, **hold one out** before believing the fit. *Check:* the held-out point within 1% of
its 3D ratio. If the fit fails, the stand-in idea fails, and that goes back to the user before
any descent.

**STEP 5 — THE BAND REPORT READS TENSION ON BOTH FACES** (0.3). `wheel_adjoint.rim_band_surface_stress`
reads von Mises on the OD only, on purpose (§203), and §206 showed it would have passed a part that is over at
the inner fibre. It reads the inner face too, excluding a distance from each spoke flank
where the re-entrant corner is singular (§199). *Check:* the 2D kernel shows the same
outer/inner split §206 measured in 3D (§206 successor 1). Then wire it per Step 0.3.

**STEP 6 — THREAD THE STAND-IN THROUGH THE OBJECTIVE.** `rim_outer` reaches `build_wheel` on
every scoring path, and the cache keys. *Check:* at the default, every committed loss and
gradient is 0 ULP unchanged; at t_eq, the objective's drop matches Step 4's 2D figure.

**STEP 7 — RE-DESCEND THE SPOKES, WARM-STARTED FROM `b729e86`.** Against the stand-in rim,
the band term of 0.3, and the target of 0.4. *Report:* how far every gene moved, and the
spoke profile overlaid on `b729e86`'s.

**STEP 8 — THE SPOKE GATE (the user), THEN 3D.** §6. Then the new part's 3D run: 8-phase
SVK mean inside [1.90, 2.10]; band hoop tension on both faces as priced (0.3); `s_zz` under
0.5's MEASURED interlayer allowable; patch metrics against the chosen rim's Step 2 values.
*Falsifier:* a 3D mean outside the band means the stand-in or the target correction was wrong, and the part does not ship.

**STEP 9 — PROMOTE** through `tests/test_promotion.py`'s checklist: never one file.

---

## 6. THE SPOKES — WHAT PROTECTS THE ONES THAT LOOK RIGHT

- **`b729e86` stays in `best_solution.json` until Step 8's gate passes.** Nothing in this plan
  replaces it before the user has seen what replaces it.
- **The rim choice sets how far the spokes move** (§2's first hypothesis). Step 2's map shows
  that distance BEFORE any descent, so the rim can be picked knowing its cost in spoke change.
  If the only rims that fix the band need spokes that look wrong, that is a Step 3 finding,
  not a surprise at Step 8.
- **Warm start, and a reported distance.** The descent starts at `b729e86`, and Step 7
  reports gene-by-gene movement and the profile overlay, not just the loss.
- **If the look matters as a requirement, it can be a constraint.** Options, not chosen:
  freeze the genes that set the spoke's shape family and descend on thickness only, or bound
  each gene's move (a trust region). Either makes "the spokes still look like these" a
  measured property instead of a hope. Worth deciding at Step 0 if the concern is strong.
- **The fallback is on the map.** A thinner base needs less spoke change and costs band
  stress. If the new spokes are refused, the map already holds the next-best rim.

---

## 7. WHAT MUST NOT HAPPEN

- No descent before Steps 4–6 pass: a descent against an unvalidated stand-in spends a run
  optimising the wrong rim, which is the 2D-flat-rim problem this file exists to fix.
- No steering claim from a radial, frictionless solve. The patch metrics are patch metrics.
- No moving `RIM_RADIUS_MM`: every gene on disk is framed by it.
- No crediting the crown's mass to the objective until the stand-in is threaded (§200 §4's
  reason still holds until Step 6 closes it).

---

## RECORDS

### R1 — 2026-09-24. WHAT THE BAND STRESS IS, FOR A PART PRINTED FLAT. A FREE READ, NO SOLVE.

**The question (the user):** the wheel is printed on its flat side because of anisotropy, and
compression at the contact patch will not snap it — so how much does the band stress matter?
Every band figure in PLAN §202–§206 is von Mises, which cannot answer that: it has no sign and
no direction. So the fields already on disk were re-read in cylindrical components.

**Instrument:** `studies/probe_3d/post3d_cyl.py` — `post3d_surface.py`'s arithmetic (same
tet10 shape functions, same Cauchy push-forward under SVK), over the band `r >= 48.45` near
the bottom, at element nodes, projected onto hoop / radial / axial. Printed flat, the layers
are planes of constant `z`: hoop and radial stresses lie IN the layers, along the perimeters;
`s_zz` pulls the layers APART; `s_rz`, `s_tz` shear them. **Control:** its von Mises column
reproduces §206 §4's all-node figures to the printed digit at seven of the eight
crown-on-top phases, and at five of the seven cut-crown ones §206 §5 quotes. The three that
differ — 24.36 against 24.62, 45.78 against 45.92, 53.33 against 53.39, all phases 0 and 3.75
— are the peaks §206 located at r 48.450, on this read's region boundary.

**Fields:** §206's eight crown-on-top SVK solves and §205's seven readable cut-crown SVK
solves, `b729e86`, h 2.0 / hc 0.25, one local size. The flat body's 3D fields were not found
on disk and are NOT read.

```
  CROWN ON TOP (b 1.5, h 1)   0.00   3.75   7.50  11.25  15.00  18.75  22.50  26.25
  hoop TENSION, max          26.72  33.80  29.83  28.41  26.49  24.17  21.01  19.97
  hoop COMPRESSION, max     -32.83 -43.74 -44.14 -42.91 -41.53 -39.61 -37.41 -31.58
  s_zz tension (interlayer)  11.11  16.58  16.78  16.65  16.27  15.61  13.84  10.36
  interlayer shear            6.54   5.79   6.71   6.42   5.92   6.53   6.56   6.36

  CUT CROWN (b 0.5, h 1)      0.00   3.75   7.50  11.25  15.00  18.75  22.50
  hoop TENSION, max          51.61  61.07  41.34  43.82  44.20  41.88  36.33
  hoop COMPRESSION, max     -43.93 -51.20 -49.08 -50.43 -50.17 -48.51 -45.38
  s_zz tension (interlayer)  17.77  23.86  24.79  30.40  33.53  33.13  29.49

  MPa, Cauchy, SVK.  Tension peaks: the band's INNER face (r 48.45-48.50), under the load,
  at or near the face's mid-plane (z 10.0-11.2 of the 11.2 half-width).  Compression peaks:
  the apex, r 51.00 / 50.00.  Max principal tension points along the hoop at every peak
  (z-share of its direction 0.00 at all fifteen).
```

**What it says, each against what could have refuted it:**

1. **The user is right about the patch.** The OD under the load is in hoop COMPRESSION, 32–44
   MPa on the part, and the max principal tension there is ~0. This could have come back
   tensile at the patch edges and did not. Two of §205's seven OD readings on the cut crown,
   at 7.5 and 11.25, are this compression — `s1` +0.70 and −0.52 at those points. Those two
   phases are still over allowable, but by the inner face's TENSION (41–44 MPa), which the OD
   reading never saw.
2. **The tension that matters is on the INNER face, under the load, along the filaments.**
   The band sags under the patch like a beam between spokes: OD compressed, inner face
   stretched, and the stretch is hoop — the direction printing flat makes strongest. On the
   current part it is 20–34 MPa against `wheel_fea`'s printed ultimate of 50 × 0.80 = 40 MPa:
   a safety factor of 1.18 at the worst phase (3.75, on the fillet toe) and 1.34–1.41 at
   mid-span (7.5, 11.25), against the 1.6 policy.
   **Margin eroded, not a fracture at the design load.** On the cut crown it reached 61 MPa
   at 3.75, over the printed ultimate — though 0 and 3.75 sit at r 48.47, where §206 §5 says
   the junction corner lives, unclassified and unrefined here.
3. **THE FINDING THE QUESTION DID NOT ASK FOR: THE CROWN PULLS THE LAYERS APART.** At the
   same inner-face points, `s_zz` is 10–17 MPa on the part and 18–34 on the cut crown. Part of
   that is only Poisson constraint in a wide band (ν · hoop ≈ 0.35 × 33 = 11.6 at 3.75); the
   ratio `s_zz / hoop` at the peak point runs 0.41–0.66 on the part and 0.34–0.81 on the cut
   crown, above ν at every phase but the cut crown's 0 and 3.75, and the excess is the band bending ACROSS the face under a patch that loads only its
   middle. **This is the one rim stress printing flat makes WORSE, not better**, and the 2D
   kernel cannot see it: plane stress sets `s_zz` to zero. A thinner base raises it — the cut
   crown, the thinnest base measured, carries twice the part's.
   **Hypothesis, not finding:** that the crown is what drives the excess. The flat body's 3D
   `s_zz` was not read, so "flat would be lower" has no falsifier yet; Step 2's (1.5, 0)
   point is it.

**So, how big a deal:** contact compression, none. Hoop tension on the inner face, moderate —
it eats the safety factor and is worth pricing so a thinner base does not spend it silently,
but it runs in the strong direction. **Interlayer tension under the crown is the one to
watch**, because it is in the weak direction, grows as the base thins, and has no allowable
in this tree. What would settle it is 0.5's coupons, not another solve.

**Scope:** one genome, one load case (the design landing load, equilibrium 33.36 N per
solve), one local mesh size at every peak, nodal values at element nodes without averaging.
No flat-body field. No peak classified by spoke distance beyond what §206 §4 already did for
the crown-on-top phases.


### R2 — 2026-09-25. STEP 1 DONE: THE RIM IS TWO ARGUMENTS. AND A FREE READ OF THE PATCH: THE CUT CROWN FAILS STEP 0.1, THE CROWN ON TOP PASSES.

**What changed:** `wheel_step_export` takes `--base-mm` (b) and `--crown-mm` (h), threaded
through `build_profile(genes, rim_outer_mm)` and `crown_rim(solid, rim_outer_mm, height_mm)`.
Unset is the shipped rim through the constants themselves. A non-shipped rim writes under a
`_b.._h..` stem, so a map point cannot land in `wheel.step`. The spoke wire does NOT move with
b: its tip still runs out to `RIM_EMBED_RADIUS_MM` 50.25 and is clipped at the new OD, so inside
that OD the face is the shipped face cut by a circle. h 0 is the flat rim (`crown_rim` returns
the solid). `wheel_geometry` did not need to change: every crown function there already took
`height_mm`.

**Step 1's three checks, each of which could have failed:**

1. **Default: byte-identical.** HEAD's own re-export of `b729e86` differs from the committed
   `export/wheel.step` only in the `FILE_NAME` timestamp line, and the parametric build differs
   from HEAD's only there too. The manifests are identical except `exported_at` / `export_seconds`.
2. **(0.5, 1.0) reproduces §200's cut part: 45638.5 mm³, 56.59 g**, to the printed digit. So do
   `volume_nofillet_mm3` 44665.9, `fillets.volume_mm3` 972.6 and the surface census (Plane 14,
   BSpline 61, Cylinder 36, swept `{}`). OCC's crown after-minus-before 4642.56 equals the closed
   form 4642.56. §200 built this solid by CUTTING down from Ø100; this builds it up from Ø98.
3. **The solver is untouched by construction, not by probe.** The diff is `wheel_step_export.py`
   and one test. The exporter is imported by nothing on the jax side (`wheel_fea` spawns it as
   a subprocess). **Deviation from the plan as written:** §200's 88-value probe was NOT re-run.
   It was asked for on the premise that `wheel_geometry` would change, and it did not.

**Pinned by** `test_export_contract.py::test_a_thinner_rim_base_is_the_shipped_face_cut_by_a_circle`.
The (b 1.5) face minus the (b 0.5) face is exactly the 49–50 annulus: 311.017672 against π(50² −
49²) = 311.017673 mm². The test also checks that the hub overlap does not move, that the rim
overlap shrinks, the flat identity, and refusals at r 48.5 and 50.25. Three mutants: a clip left
at 50 and h 0 building an arc both fail it. A band left at 50 inside a clip at 49 builds the SAME
face; only the rim-overlap assertion, added for it, kills it.

**A FINDING THE STEP DID NOT ASK FOR: THE JUNCTION BITE IS A PROPERTY OF THE CONSTRUCTION, NOT
OF THE SOLID.** The same solid (check 2) reports rim bite **0.7206 t** in §200's manifest and
**0.240 t** in this one — under the 0.25 floor, `[WEAK JUNCTION]`. `check_junction_overlap`
intersects the spoke with the UNCROWNED 2D band: §200's was 1.5 mm and this one's is 0.5 mm, and
neither sees the crown. So every b 0.5 map point will print the warning. It is a true
statement about the side-face band and blind to the crown. The flag warns; it does not gate.
Nothing here is changed for it.

**THE PATCH, READ FREE FROM FIELDS ALREADY ON DISK** — new instrument `studies/probe_3d/patch3d.py`,
Step 0.1's metrics. The pressure is fe3d's own: the same penalty, the same 6×6 points on the
same P2 OD triangles, the same reference-surface gap. **Control: ∫p dA reproduces fe3d's
`contact_force_half_n` 33.36165 N at all 15 fields**, and the length, width and peak pressure
reproduce each log's `patch_x_mm` / `patch_z_mm` / `peak_pressure_mpa` (checked at the cut
crown's 0 / 3.75 / 7.5).

```
  CROWN ON TOP (1.5, 1)   0.00   3.75   7.50  11.25  15.00  18.75  22.50  26.25
  patches                    1      1      1      1      1      1      1      1
  length, mm              2.29   2.73   2.71   2.64   2.59   2.54   2.48   2.28
  width, mm               2.46   2.36   2.41   2.43   2.44   2.47   2.46   2.47
  scrub M/mu, N mm        44.8   48.3   48.5   48.2   47.8   47.4   46.8   44.9
  lever = M/(mu F), mm   0.671  0.724  0.726  0.722  0.717  0.710  0.701  0.672

  CUT CROWN (0.5, 1)      0.00   3.75   7.50  11.25  15.00  18.75  22.50
  patches                    1  2 (.90/.10) 1     1      1      1      1
  length, mm              2.20   8.91  10.48   7.88   5.79   4.39   3.54
  width, mm               2.50   2.30   1.60   2.00   2.34   2.60   2.62
  scrub M/mu, N mm        44.4   93.5  196.6  129.1   90.4   71.2   60.4
  lever, mm              0.666  1.401  2.947  1.935  1.354  1.067  0.905

  F = 2 x 33.36165 N every row.  M/mu = int p |q - c| dA over the full (mirrored) patch,
  c the pressure centroid.  Phase 26.25 of the cut crown is not read, as in R1.
```

**What it says:** the crown on top keeps one short strip at every phase, and its scrub lever
moves 8% over the stencil. The cut crown's patch runs from 2.20 to 10.48 mm long (4.8x),
splits at 3.75, and its scrub lever swings **4.4x** as it rolls. **By Step 0.1 the cut crown
FAILS** (two patches), and it would be a poor steering rim even without the split: the proxy
for turn-in resistance changes 4.4x across one spoke pitch of roll. The mechanism is not
measured here. That the thin band sags onto the ground between spokes, so the crown stops
setting the patch, is the obvious reading, and the patch's position relative to the spokes
was not classified.
**Hypothesis, not finding:** that the base, not the crown, sets the patch's stability. The two
parts differ in b only, but b also changes the drop 1.6x, and the patch length is confounded
with the drop. Step 2's (1.0, 1.0) point splits them: its predicted drop sits between the two
parts'.

**Scope:** one genome, one mesh rung (h 2.0 / hc 0.25, the default box). Contact outside the
x ±4 box sits on coarser triangles (§205), which is where the cut crown's long patches run. The
scrub proxy is a pressure moment on frictionless, rigid, flat ground under a radial load.

### STEP 2 — PREDICTIONS, REGISTERED BEFORE THE FIRST SOLVE

First pass: the seven new points at phases 0 and 3.75, SVK, h 2.0 / hc 0.25, default box,
`fe3d --r-out` = 48.5 + b + h, windows §205's final. Queue order: (1.5, 0), (1.0, 1.0),
(1.0, 0.5), (1.5, 0.5), (1.0, 0), (0.5, 0.5), (0.5, 0).

**H-MEAN** (the mean-thickness stand-in §3 doubts): q(b, h) = q(1.5, 1) · (t̄(1.5, 1) / t̄)^k,
with t̄ = b + 2h/3 and k fitted per phase and per quantity on the two measured h 1 parts.
**H-LIN** (§2's first hypothesis): the drop is linear in b at fixed h, so it can only be tested
at (1.0, 1.0).

```
                          k     (0.5,0) (0.5,.5) (1.0,0) (1.0,.5) (1.0,1) (1.5,0) (1.5,.5)
  drop  ph 0     H-MEAN  0.770   3.960   2.672    2.322   1.860    1.566   1.699   1.456
  drop  ph 3.75  H-MEAN  0.873   4.874   3.120    2.661   2.070    1.704   1.868   1.568
  drop  ph 0     H-LIN                                              1.671
  drop  ph 3.75  H-LIN                                              1.840
  hoop T ph 0    H-MEAN  1.063   127.1    73.8     60.8    44.8     35.3    39.5    31.9
  hoop T ph 3.75 H-MEAN  0.956   137.2    84.2     70.8    53.8     43.4    48.0    39.7
  s_zz  ph 0     H-MEAN  0.759    33.8    22.9     20.0    16.1     13.6    14.7    12.6
  s_zz  ph 3.75  H-MEAN  0.588    39.3    29.1     26.1    22.1     19.3    20.6    18.3
```

The falsifiers, each one a reading the first pass makes:

- **P1, (1.0, 1.0) drop: H-MEAN against H-LIN**, 1.566 / 1.671 at phase 0 and 1.704 / 1.840 at
  3.75, 6.7–8.0% apart. The closer one survives. If neither is within 3% at both phases, both
  are refuted and the Step 3 map cannot be interpolated; it has to be measured.
- **P2, (1.5, 0) drop: H-MEAN against the flat estimate.** The estimate is §204's modelled-flat
  SVK twin, 1.7858 / 2.0278, times §202–§204's junction stiffening of 0.970–0.974, giving
  1.732–1.739 / 1.967–1.975. H-MEAN reads 1.699 / 1.868, 1.9–2.4% / 5.3–5.7% below it. That is already
  a doubt about H-MEAN at h 0. The junction factor was measured linear and at phase 0 only, so
  if the estimate misses, the stiffening does not carry to SVK or to 3.75.
- **P3, (1.5, 0) interlayer tension: does the crown drive R1's excess?** The instrument is
  `post3d_cyl.py`'s new last line: `s_zz / hoop` at the SAME node as the band's hoop-tension
  peak. **R1's "0.41–0.66 at the peak point" was not that.** It divided the two band MAXIMA,
  which sit at nearby but different nodes. Same-node, the crown on top reads:

  ```
    phase           0.00   3.75   7.50  11.25  15.00  18.75  22.50  26.25
    crown on top   0.390  0.476  0.557  0.586  0.614  0.646  0.653  0.409
    cut crown      0.344  0.388  0.578  0.678  0.742  0.787  0.805    --
  ```

  The first pass reads phases 0 and 3.75 only, where the crown on top is 0.390 / 0.476. R1's
  hypothesis says the flat band sits at ν: **flat ≤ 0.37 at both phases supports it; flat at or
  above 0.390 / 0.476 at both refutes it (the crown is not the driver); between is undecided.**
  The discriminating phases are mid-span, 7.5–22.5, which the first pass does not read; an
  undecided P3 is the reason to read them. H-MEAN's s_zz 14.7 / 20.6 MPa is registered as
  well. Its hoop 39.5 / 48.0 has no in-tree bound: §204's flat 32.68 MPa is OD von Mises, a
  different quantity on a different body.
- **P4, the patch.** Every b ≥ 1.0 crowned point shows one patch at both phases. Every h 0 point
  has a scrub lever at least 3x the crown on top's 0.67–0.72 mm: a flat rim contacts across
  its face. A b 0.5 point with ONE patch at 3.75 would say the split is not a thin-base
  property.

### R3 — 2026-09-25. STEP 2's FIRST PASS, AND STEP 3 DECIDED: THE RIM STAYS (1.5, 1.0). IT IS THE ONLY RIM ON THE GRID WHOSE BAND IS UNDER THE PRINTED ULTIMATE, AND IT SITS ON THE Ø102 CAP.

**What ran:** the seven new points of §5's grid, each exported by Step 1's `--base-mm` /
`--crown-mm` from `b729e86`, at phases 0 and 3.75. SVK, h 2.0 / hc 0.25, default box,
`fe3d --r-out` 48.5 + b + h, 32 GiB `MemoryMax` scopes, one serial queue, 438–1334 s per
accepted solve (refused attempts up to 1765 s) and 13.8–18.1 GiB peak. The two measured parts (§205, §206) fill the h 1 column. **Every
solve below passes `patch3d.py`'s control**: ∫p dA reproduces fe3d's 33.36165 N.

```
  (b, h)      mass g   drop 0     drop 3.75   H-MEAN err    hoop T 0 / 3.75   s_zz 0 / 3.75   patches   lever mm
  (1.5, 0)    59.47    1.72309    1.93624     +1.4 / +3.7    39.07 / 50.33     14.66 / 18.21   1 / 1     4.13 / 4.64
  (1.5, 0.5)  62.40    1.46596    1.59516     +0.7 / +1.7    32.11 / 45.29     12.93 / 22.52   1 / 1     0.78 / 0.90
  (1.5, 1)    65.35    1.27975    1.35482     anchor         26.72 / 33.80     11.11 / 16.58   1 / 1     0.67 / 0.72
  (1.0, 0)    55.13    2.51160    2.73806 w8  +8.2 / +2.9    60.30 / 58.74     21.67 / 21.30   1 / 2     4.25 / 6.18
  (1.0, 0.5)  58.02    1.90185    2.14986     +2.3 / +3.9    45.40 / 58.35     16.54 / 25.90   1 / 1     0.78 / 1.12
  (1.0, 1)    60.95    1.55152    1.69988     -0.9 / -0.2    35.45 / 49.49     13.40 / 24.09   1 / 1     0.67 / 0.84
  (0.5, 0)    50.83    NOT SOLVED -- see below
  (0.5, 0.5)  53.70    2.93322 w4 3.00910 w8  +9.8 / -3.6    69.99 / 57.91     23.78 / 19.98   1 / 2     0.90 / 5.32
  (0.5, 1)    56.59    2.06179    2.32594     anchor         51.61 / 61.07     17.77 / 23.86   1 / 2     0.67 / 1.40

  drop: SVK axle drop, mm.  hoop T: max hoop tension on the band's inner face (r >= 48.45,
  |x| < 10), MPa, Cauchy, element nodes, post3d_cyl.py; the printed ultimate is 50 x 0.80 =
  40 MPa (wheel_fea), the allowable 25.  s_zz: max interlayer tension in the band, same
  instrument.  lever: scrub M / (mu F), patch3d.py.  w4 / w8: fe3d's window widened 4 / 8 mm
  each side.  mass: OCC, from each export's manifest; (1.5, 0) is §200's flat 59.47 g.
```

**The registered predictions (R2), each against the reading:**

- **P1 — H-MEAN SURVIVES, H-LIN IS REFUTED.** At (1.0, 1.0): 1.55152 / 1.69988 measured, H-MEAN
  1.566 / 1.704 (−0.9% / −0.2%), H-LIN 1.671 / 1.840 (−7.1% / −7.6%), each as measured over
  predicted.
  §2's "linear in b, the drop hits 2.0 at b ≈ 0.61" falls with H-LIN. **H-MEAN holds for the
  drop at h 1 only.** Its error grows as the crown shrinks: 0.2–0.9% at h 1, 0.7–9.8% at h 0.5,
  1.4–8.2% at h 0, and always the same sign (too stiff) at h ≤ 0.5 but one. **Hypothesis, not
  finding:** a crown stiffens more than its 2h/3 share of mean thickness says. The confound is
  that k was fitted on two points per phase, so the error pattern could be the fit's.
  **H-MEAN FAILS FOR BAND STRESS AT 3.75:** 43.4 predicted, 49.49 read at (1.0, 1); 39.7
  predicted, 45.29 read at (1.5, 0.5). At phase 0 it was within 1.3% at five of the six new
  points (5.2% at (0.5, 0.5)).
- **P2 — THE FLAT ESTIMATE MISSES LOW, AND IT IS THE JUNCTION FACTOR.** (1.5, 0) reads 1.72309 /
  1.93624 against 1.732–1.739 / 1.967–1.975. Measured against §204's twin, the exported flat
  part is **3.51% stiffer at phase 0 and 4.52% at 3.75 under SVK**, against the 2.60–3.04%
  measured LINEAR at phase 0 (§202–§204). So the junction stiffening grows under SVK and off
  phase 0. That matters to Step 0.4's single target factor, which folds it in.
- **P3 — SUPPORTED, AND THE HEADLINE IS THE OTHER HALF.** The flat (1.5, 0) reads same-node
  `s_zz / hoop` **0.354 / 0.341**, at ν, under the 0.37 bar at both phases. The crown on top reads
  0.390 / 0.476. So the crown does push the ratio above ν, as R1 hypothesised. **But in
  absolute terms the flat band carries MORE interlayer tension, not less: 14.66 / 18.21
  against 11.11 / 16.58 MPa**, because its hoop tension is higher still (39.07 / 50.33 against
  26.72 / 33.80). R1's "the crown pulls the layers apart" is right about the ratio and wrong
  about the stress, at the two phases read. The mid-span phases, where the crowned ratio
  reaches 0.65, were not read on the flat.
- **P4 — HOLDS.** Every crowned b ≥ 1.0 point shows one patch at both phases. Every h 0 point has
  a lever of 4.13–6.18 mm, 6.2–8.5x the crown on top's 0.671 / 0.724 at the same phase. **The split is a thin-rim
  property, not a crown property**: (0.5, 0.5) splits at 3.75 (0.788 / 0.212) as (0.5, 1) did,
  and so does the flat (1.0, 0) (0.846 / 0.154). Step 0.1 fails three points: (0.5, 1),
  (0.5, 0.5) and (1.0, 0).

**Where the band peaks sit.** At every point the hoop-tension peak is at x −2.0…−2.3 (phase 0)
or x +0.85…+1.12 (phase 3.75), r 48.45–48.50. That is the same place on every rim, so the spoke
geometry sets it. At 3.75 it is where §206 Q5 put the crown on top's peak: 0.92 mm from the
spoke body on the fillet toe, and under refinement it moved +0.007%, not the +10.6% a corner
singularity would. No peak on the NEW points was refined. My own distance-to-spoke read did not
reproduce §206 §4's column (0.07 / 0.43 mm against 0.52 / 0.92 at the crown on top's 0 / 3.75),
so no distance is quoted for them.

**AN INSTRUMENT DEFECT, CAUGHT BY THE CONTROL.** fe3d builds its contact candidates from OD
triangles whose EVERY node lies inside the window, and its truncation check reads only those
triangles. At (1.0, 0) phase 3.75, with the window widened 4 mm, three coarse triangles (about
1.7 mm across, outside the refined box) straddled the window's edge. The ground penetrated them
at up to 0.42 µm, carrying 0.70 N that the solve never saw: 34.062 against 33.362 N. fe3d
reported success. Re-solved widened 8 mm, the control passes, and the drop moved
**−0.006%** (2.73822 → 2.73806) with the band figures unchanged. It cost nothing here. Only
`patch3d.py`'s control can see it, which is a reason to run the control on every solve. All 15
§205/§206 fields passed it (R2), so nothing committed is affected. fe3d is not changed.

**(0.5, 0) IS NOT SOLVED.** At phase 0 the SVK equilibrium walked out of both windows tried:
drop 14.1 mm at the default window, 12.7 mm with 29 active points at +4 mm, each time with the
patch on the window's edge. Its phase 3.75 was stopped by hand. Nothing is claimed about it
beyond that. It is thinner than the cut crown, which already fails the band and Step 0.1.

**STEP 3 — DECIDED ON THE STANDING DELEGATION (overrulable, nothing spent on it): THE RIM STAYS
(1.5, 1.0), the part as shipped since §206.**
- **The band decides it.** At phase 3.75, (1.5, 1) is the only point under the printed ultimate
  (33.80 MPa; every other point reads 45.29–61.07). It is also the lowest in interlayer
  tension at both phases (11.11 / 16.58). Its full eight phases are already measured (§206), and
  its 3.75 peak is already refined (§206 Q5). The worst band phase, 3.75, reads 1.35x the 25 MPa
  allowable. Step 0.3 prices that; it does not forbid it.
- **The patch agrees.** It has one strip at all eight phases, the lowest scrub lever on the grid
  and the flattest over the stencil (R2).
- **What it costs is what §2's first hypothesis said.** It is the stiffest and heaviest point
  (65.35 g), 34.1% under target (§206). **The spokes must soften the most here, and Step 7's
  distance-from-`b729e86` report will say how much.** §6's fallback on the map is (1.5, 0.5):
  +0.19–0.24 mm of drop and 2.95 g lighter, but 45.29 MPa at 3.75, 1.13x the printed ultimate.
- **THE Ø102 CAP BINDS, AND IT IS THE ONE CONSTRAINT THIS CHOICE SITS ON.** (1.5, 1) is the
  grid's corner at b + h = 2.5. Both directions that lower band tension on the grid, more base
  and more crown, go past Ø102. What lies beyond is unmeasured and is the user's to open
  (Step 0.2 set the cap). A rim past the cap would also be stiffer, which moves the spokes further.

**So Step 4 fits the stand-in for h 1.** The h 1 column (b 0.5 / 1.0 / 1.5) is the grid to fit on
and hold one out from. The chosen rim is its end point.

**Scope:** one genome, phases 0 and 3.75 only, for every point but the two measured ends. One
mesh rung. No new peak refined. The grid itself is the only evidence for "more base, more crown
lowers band tension": hoop tension falls along every row and column at phase 0. At 3.75 it
breaks at b 0.5, where (0.5, 0.5) reads below both (0.5, 1) and (1.0, 0.5).

### R4 — 2026-09-25. STEP 4 FAILS AS WRITTEN: NO FIT ALONG b REPRODUCES 3D TO 1%. THE CHOSEN RIM's OWN STAND-IN, t_eq 2.144 mm, MISSES TWO OF SIX HELD-OUT PHASES BY 1.5% AND THE EIGHT-PHASE MEAN BY 0.45%. BACK TO THE USER BEFORE ANY DESCENT, AS STEP 4 REQUIRES.

**Instrument:** `studies/study_crown_standin.py` → `study_crown_standin.json`. The 2D drop is the
objective's own per-phase quantity: `build_wheel(..., fillet=True)` at `coarse`,
`solve_wheel_contact` at `SERVICE_FORCE_N` under SVK, with only `rim_outer` = 48.5 + t moved.
**Control: at t 1.5 it reproduces `best_solution.json`'s recorded figures to ten digits.**
Phase 7.5 gives 2.269685753 = `axle_drop_max_mm`, 22.5 gives 1.789749979 = `axle_drop_min_mm`,
and the mean over the eight stencil phases, 1.992026, = `axle_drop_mean_mm`. The fit targets are
RATIOS to the flat exported (1.5, 0) in 3D and to t 1.5 in 2D, so decision 0.4's offsets cancel.
The sweep interpolation (t 1.0–2.4, step 0.2, log-log) checks against a direct solve at each
t_eq to 0.10% or better.

**The column fit (h 1, phases 0 / 3.75):**

```
  b      3D ratio (0 / 3.75)    t_eq mm (0 / 3.75)    t_eq mean   t_eq - b
  0.5    1.19657 / 1.20127      1.2365 / 1.1948       1.2156      0.716
  1.0    0.90043 / 0.87793      1.6924 / 1.7104       1.7014      0.701
  1.5    0.74271 / 0.69972      2.1399 / 2.1485       2.1442      0.644

  hold out   form     t_pred    ratio error (0 / 3.75)
  b 0.5      offset   1.1728    +5.18% / +1.41%
  b 0.5      affine   1.2587    -1.71% / -3.69%
  b 1.0      either   1.6799    +0.58% / +1.92%
  b 1.5      offset   2.2085    -2.48% / -2.56%
  b 1.5      affine   2.1872    -1.73% / -1.68%
```

**No fold passes the 1% check at both phases. Step 4's criterion, as written, FAILS.** t_eq(b) is
concave, and the ratio moves about 2% per 0.03 mm of band at 3.75. So a stand-in interpolated
along b is not good to 1%, even between measured points.
Predictions (registered 05:39, in the session scratchpad):
- **S4.1 holds:** 2.144 is inside 1.95–2.15 and under the mean-thickness 2.168; 1.216 is inside
  1.20–1.35 and over 1.168. §3's direction is confirmed at both ends.
- **S4.2 holds for `offset`, fails for `affine`:** the affine form does not pass even interpolating
  at b 1.0.
- **S4.3 is refuted:** the largest phase split is 0.042 mm (b 0.5), under the predicted 0.05.

**THE PHASE HOLD-OUT ON THE CHOSEN RIM, (1.5, 1).** This is the test that matters once Step 3 put
the rim on a measured grid point. It fits one band thickness at two phases and checks it at the
six the objective also averages over. It needed the flat exported (1.5, 0) at all eight phases in
3D: six new fe3d SVK solves, 425–608 s and 14.3–17.0 GiB each, all on the default window, all
passing `patch3d.py`'s force control, one patch at every phase. S4.4 was registered at 05:57,
before any held-out number.

```
  phase    3D ratio   2D ratio at t_eq 2.1442   error
   0.00    0.74270    0.74257                   -0.02%   fitted
   3.75    0.69972    0.69950                   -0.03%   fitted
   7.50    0.68256    0.69300                   +1.53%
  11.25    0.72116    0.73235                   +1.55%
  15.00    0.76196    0.76900                   +0.93%
  18.75    0.78751    0.78957                   +0.26%
  22.50    0.78870    0.78694                   -0.22%
  26.25    0.76913    0.76547                   -0.48%
  8-phase mean ratio  3D 0.74105   2D 0.74437   +0.45%
```

**S4.4 FAILS: 7.5 and 11.25 are over 1%.** The mean is within 0.45%. The EXPECTATION beside it is
refuted in shape: I expected the phases near the fit to pass and mid-sector to fail, and it is
the reverse. The 2D band tracks the ratio's swing across the sector (0.683–0.789 in 3D,
0.693–0.790 in 2D) but not its dip just after 3.75.

**TWO NUMBERS THIS RECORD MEASURES FOR THE FIRST TIME:**
- **The flat EXPORTED part's 3D SVK eight-phase mean is 1.77026 mm**, 11.5% under the 2.0 mm
  target. The per-phase drops are 1.72309 / 1.93624 / 2.02807 / 1.89284 / 1.74007 / 1.63183 /
  1.59244 / 1.61750. §204's 1.8102 was the modelled twin. **3D / 2D for that body is 0.88867**:
  Step 0.4's single target factor, measured on the flat exported part, across all eight phases.
- **Its patch at every phase is one full-face strip, lever 4.12–4.79 mm**, 6.2–6.6x the crown
  on top's 0.671–0.726 at every one of the eight phases.

**WHAT THIS LEAVES FOR THE USER (Step 4's rule: a failed fit goes back before any descent):**
1. **Is the eight-phase MEAN the right criterion?** The objective's `deflection` term scores the
   mean, and there the stand-in is within 0.45%. The per-phase 1% bar came from Step 4's
   wording, not from a requirement. The per-phase misses (up to 1.55%) would enter
   `phase_ripple`, not `deflection`. If the mean is accepted, t_eq = 2.144 mm for (1.5, 1),
   and the stand-in is the Ø100 gene-frame band thickened to r 50.644 in 2D.
2. **If per-phase 1% is required**, one band thickness cannot do it. The next idea to try is a
   PHASE-DEPENDENT band or a stiffness scale on the band (`rim_modulus_scale` exists in
   `wheel_fem`), which is a different stand-in and a new Step 4.
3. **Nothing here changes Steps 5 and 6's code work.** Step 7's descent waits on 1 or 2.

**Scope:** one genome, one 3D mesh rung, the 2D at `coarse` only. The column fit uses phases 0 and
3.75 only. The phase hold-out uses one rim.

**WHERE THE FIELDS ARE** (session scratchpad, not the tree):
`/tmp/claude-1000/-home-eric-bodhi-github-wheel/cd083dce-06e6-46c8-b903-b5b7bebfe30e/scratchpad/map/`.
It holds `m_<b>_<h>_<phase>.npz` / `r_<b>_<h>_<phase>.npz` for every R3 point and the six new flat
phases, the STEPs and manifests, `q.log` (every solve's line), `windows.txt`, and the queue
scripts. The six held-out 2D drops are in `../d2_phases.json`, and the registered predictions
are in `../PREDICTIONS_step4.txt`. The §206 crown-on-top fields are in `ea223ab2-…/scratchpad/p3d/`
and §205's cut-crown fields in `afcbf479-…/scratchpad/`.

### R5 — 2026-09-25. R4's QUESTION DECIDED ON THE USER's DELEGATION ("decide the next design decisions"): THE STAND-IN IS JUDGED ON THE EIGHT-PHASE MEAN, t_eq STAYS 2.1442 mm, NOT REFITTED, AND STEPS 5–6 GET THEIR DESIGN. NO SOLVE; EVERY FIGURE BELOW IS RE-DERIVED FROM R3, R4, §206 AND `study_crown_standin.json`.

**A correction first.** §2's table gives the crown on top's 3D SVK eight-phase mean as 1.3178 mm.
That is §206's line (B), `1.8102 x R_svk`, a CARRIED estimate. §206's line (A), the mean of the
eight measured drops, is **1.31186 mm**, and it is the figure R4's 0.74105 divides: 1.31186 /
1.77026 = 0.74105. Nothing downstream used 1.3178. The table is left as written.

**D1 — THE CRITERION IS THE EIGHT-PHASE MEAN, AND t_eq = 2.1442 mm IS KEPT, NOT REFITTED.**
- The per-phase drop enters the loss only through its mean: `deflection` is
  `(mean_drop - target) / target`, squared. `phase_ripple` is the only term that reads the phases
  one by one, and its weight is 0.0 in `DEFAULT_WEIGHTS` and in `best_solution.json`'s own
  descent (`loss_terms.phase_ripple` 0.0). So R4's 1.53% / 1.55% at 7.5 and 11.25 reach no term
  the descent sees.
- **Budget:** +0.45% of the mean is 0.006 mm, against Step 8's [1.90, 2.10] half-width of
  0.10 mm, 17x inside it.
- **Why not refit on all eight phases:** the +0.45% is an error at SIX HELD-OUT phases, the only
  held-out evidence the stand-in has. A fit to the mean makes it zero by construction and leaves
  nothing to check the stand-in against until Step 8.
- **Why not R4's option 2:** a phase is a rotation of one geometry, so a band thickness that
  depends on phase is not a wheel. `rim_modulus_scale` is also one scalar and cannot bend a
  phase pattern. Either one is a new Step 4, spending solves on an error that enters no term.
- **The cost, recorded so it is not rediscovered:** the stand-in over-reads the ripple. Taking the
  eight 3D crowned drops (flat per-phase drop x R4's 3D ratio) against flat x R4's 2D ratio,
  std/mean is **3.77% measured and 4.41% predicted, 1.17x.** **If `phase_ripple` is ever given a
  weight under the stand-in, D1 re-opens.**

**D2 — THE TARGET FACTOR IS K = 0.88867**, R4's 3D / 2D for the flat exported part, all eight
phases, as 0.4 decided. It is an explicit `drop_factor` on `t3_terms` that scales `drops` and
`dgrads` as they are assembled, so `deflection` and the report read predicted-3D millimetres and
`phase_ripple` is unmoved (it has no scale). `None` skips the multiply. **Predicted at `b729e86`
on the stand-in:** K x 1.48280 (the 2D eight-phase mean at t_eq, 1.992026 x 0.74437) =
**1.31773 mm, +0.45% over the measured 1.31186.** The combined per-phase factor 3D / 2D(t_eq)
is **0.87620 at phase 0 and 0.87370 at 3.75**, and it is Step 8's first falsifier (D7).

**D3 — ONE SWITCH, NOT THREE FLAGS.** The stand-in's three values, `rim_outer` 48.5 + 2.1442 =
50.6442, `drop_factor` 0.88867 and the band weight (D6), are one record in `wheel_objective`
and one `wheel_stage3 --crown-standin`. The run's `search` block records all three. A rim
thickened without its factor, or a factor without the band term, scores a wheel nobody measured.

**D4 — THE 2D BAND QUANTITY IS THE TANGENTIAL STRESS ON THE INNER FACE, ITS POSITIVE PART.** On a
traction-free face the normal and shear tractions are zero, so the in-plane max principal IS the
tangential stress: no square root and no branch. It is read on `rim_inner_free`, at element
nodes without averaging, as `rim_band_surface_stress` reads the OD. That report is not touched.
**REGISTERED: NO SPOKE-FLANK EXCLUSION IS NEEDED FOR TENSION.** §199's confirmed singular corner is
a COMPRESSION corner, `hub:P_c`. `rim:P_c` does not resolve, and `rim:P_t` is regularised
(λ 1.0493). §203's 30.06 / 34.60 / 41.08 MPa climb was von Mises over the whole band, which
includes the compression corner. **Falsifier:** the inner-face tension max at phase 3.75 moves
more than 3% from `medium` to `fine`. If it does, the exclusion distance comes from that ladder,
and it must not exclude R3's 3D peak locations (x +0.85 to +1.12 at 3.75, x −2.0 to −2.3 at 0).

**D5 — THE 2D TENSION IS CALIBRATED TO 3D BY ONE MEASURED FACTOR, TAKEN AT ITS MAXIMUM.**
c_band(φ) = R1's 3D inner-face hoop tension on the crown on top (26.72 … 19.97 MPa, all eight
phases) divided by the 2D value at t_eq. The priced factor is **max over φ of c_band**, so it
over-prices by construction.
- **Control:** c_flat at (1.5, 0), phases 0 and 3.75, is R3's 39.07 / 50.33 divided by the 2D
  value at t 1.5. There the 2D band IS the part's geometry, so c_flat is the plane-stress offset
  alone, and c_band / c_flat is the stand-in's share.
- **Registered:** c_band > 1 at every phase, because a thicker 2D band carries the same moment at
  a lower surface stress.
- **Falsifier for pricing at all:** max / min of c_band over the eight phases above 1.3. Then the
  2D quantity is not tracking the 3D mechanism, and the band stays a report. That goes in the
  next record; it does not stop the arc.

**D6 — PRICED THE WAY `stress_margin` IS, AT ITS EXCHANGE RATE.**
`band_margin = soft_barrier(util_band − MARGIN_KNEE_UTIL, w)`, where:
- `util_band = c_band x phase_pnorm(node_pnorm(s_tt+)) / allowable`;
- w = `DEFAULT_WEIGHTS["stress_margin"]`, one exchange rate for utilisation (§99);
- phases aggregate through `_pnorm_and_grad` at `stress_phase_p`;
- the node exponent is the smallest of {8, 16, 32} within 2% of the nodal max at all eight phases
  at t_eq.

No wall (0.3). `DEFAULT_WEIGHTS["band_margin"]` is 0.0, and D3's switch sets it.
- **Registered at `b729e86`:** util_band about 1.3 (R1's 3.75 row is 1.352 x 25 MPa), so the
  term is about 20–27, a large share of today's loss of 52.57.
- The Step 5 driver reports the term and the cosine between its gradient and `deflection`'s.
  **A strongly opposed pair (cosine below −0.8) is recorded, not acted on:** the descent is the
  instrument that prices that trade.

**D7 — STEP 8's FIRST PASS HAS A REGISTERED FALSIFIER, AND A MISS IS NOT A HALT.** On the Step 7
genome, at phases 0 and 3.75, the combined factor 3D / 2D(t_eq) should hold to within 2% of
0.87620 / 0.87370. That tests two hypotheses no measurement here can reach: that t_eq, fitted on
`b729e86`'s spokes, transfers to softer ones, and that K does. A miss refits t_eq on the new
spokes at (1.5, 1), re-descends from a warm start, and records which one moved.

**Order:** Step 6 (threading, default path 0 ULP) before Step 5 (the band term), then Step 7.
Step 5's calibration needs only `build_wheel(rim_outer=)`, which exists, but its gradient check
needs the threaded objective. **Unchanged:** the rim stays (1.5, 1.0). 0.5's coupons block
Step 8's `s_zz` gate only. `b729e86` stays in `best_solution.json` until Step 8.

### R6 — 2026-09-25. STEPS 6 AND 5 DONE. D4 IS REFUTED: THE 2D INNER FACE HAS A CORNER BESIDE EVERY SPOKE, AND THE BAND IS READ 0.5 mm CLEAR OF IT. THERE D5 PRICES (c_band SPREAD 1.185), THE TERM READS 55.9 AT `b729e86`, AND ITS GRADIENT OPPOSES THE DEFLECTION TERM's AT COSINE −0.990.

**Step 6 (`b9aa118`).** The stand-in reaches `phase_meshes`, `t3_terms`, `objective`, the pool worker and both descents. Into the descents it rides `problem_kw`, the route `force` already takes. Every call site is edited in its own line count, and new code sits at the file ends.
- **The default path did not move.** The full suite is green, including the requirements layer's bit-identity gate.
- **On the stand-in the objective reproduces R4.** At the six held-out phases it gives R4's direct 2D solves exactly. Its predicted 3D mean is 1.31779 mm against the measured 1.31186 (+0.45%, as D2 registered).
- **Two mutants go red.** Removing `rim_outer` from the pool worker breaks the pooled-equals-serial check. An Evaluator that builds meshes without it is refused by `objective`.
- **Found, not acted on:** on the stand-in, junction utilisation falls from 0.953 to 0.756, under the 0.80 knee.
- **The citation sweep's registered prediction missed on its instrument.** A 212-row first attempt was reflowed to 19 rows, every one anchored on a line the change edits in place. I predicted 11 with a `wheel_objective.py:N` grep. The difference is bare `:N` tokens carried from an earlier owner, which that grep cannot see. The 19 are left as they are (memory, §159): each still points at its line, with that line's new text.

**Step 5. Instrument:** `studies/study_band_tension.py` → `study_band_tension.json`.
- **What it solves:** 24 SVK service solves at `b729e86`, each keeping the RAW inner-face profile (`wheel_adjoint.band_face_profile`: hoop stress, x, y and distance to the nearest spoke junction, per element node). The 24 are the flat band and the stand-in at the eight stencil phases, at `coarse`, plus the stand-in at phases 0, 3.75 and 11.25 at `medium` and `fine`.
- **Everything below is a reduction of those profiles.** The driver's first pass read maxima only and was overwritten. It is the d 0 row here, from solves that reproduce it to the printed digit.
- Predictions were registered before the second pass, in the session scratchpad (`PREDICTIONS_step5b.txt`).

**D4 — REFUTED.** At the face node beside a junction the tension climbs with refinement:

```
  stand-in, inner face, no exclusion     coarse    medium    fine      node's distance to the junction
  phase 3.75  (x +2.125)                 27.05     30.43     36.35     0.188 / 0.117 / 0.067 mm
  phase 0     (x -1.049)                 28.63     33.43     40.89
  phase 11.25 (mid-span control)         19.75     20.35     20.20
```

- **The two junction phases climb, and the climb accelerates:** +12.5% then +19.4% at 3.75, +16.8% then +22.3% at 0. The node closes on the corner at each rung. The mid-span control moves −0.76% from medium to fine.
- **This is §203's corner, on its tension side.** §203 says "the band's inner face meets each rim junction at a re-entrant corner that survives the fillet". D4 leaned on §199's compression-corner result and did not read that sentence.
- **The corner node is the peak at four of the eight coarse phases** (0, 3.75, 22.5, 26.25). R3's 3D peaks at 0 and 3.75 sit about 1 mm further out, on the face node 1.154 mm from the junction.

**THE EXCLUSION, FROM THE LADDER, AS D4's FALLBACK SAID. P6.1 IS REFUTED AS REGISTERED:** no exclusion d both keeps R3's 3D peaks in the read and moves ≤ 3% from medium to fine at all three ladder phases.

```
  d mm              0.25     0.50     0.75     1.00     1.25     1.50     2.00
  m->f, phase 0    +12.04    -2.65    +6.16    -4.16    -7.09    -0.13    +3.28   %
  m->f, phase 3.75  +0.26    -3.40    -3.40    -3.40    -3.40    +0.16    +1.11   %
  reads 3D peaks     yes      yes      yes      yes      no       no       no     (the 1.154 node)
```

**DECIDED: d = 0.5 mm, on these grounds and not on the 3% bar, which it misses at 3.75 by 0.40 points.**
1. The objective runs at `coarse`. There the face nodes sit 0.188, 1.154 and 2.12 mm from a junction, so every d from 0.25 to 1.0 drops exactly the corner node and nothing else. 0.5 leaves 0.31 mm and 0.65 mm to the nodes either side.
2. Beyond it, the medium-to-fine moves are −2.65 / −3.40 / −0.76%, and non-monotone across d, where d 0 climbed 19–22%. **Hypothesis, not finding:** the scatter is where each mesh puts its first node on a stress that falls with distance from the corner. Three meshes cannot separate that from a slow climb.

**D5 — PRICES, AND ITS "c_band > 1" HOLDS ONCE THE CORNER IS OUT.** Beyond 0.5 mm:

```
  phase          0.00   3.75   7.50   11.25  15.00  18.75  22.50  26.25
  3D (R1)       26.72  33.80  29.83  28.41  26.49  24.17  21.01  19.97   MPa
  2D stand-in   19.77  23.29  21.91  19.75  18.08  15.85  13.75  15.49   MPa, coarse, beyond 0.5 mm
  c_band        1.352  1.452  1.361  1.439  1.465  1.525  1.528  1.289
```

- **Spread 1.185, under the 1.3 bar (P6.3 holds).** With no exclusion it was 1.717, and c_band was 0.933 and 0.888 at 0 and 26.25: the corner node made those phases read above the 3D figure.
- **The priced factor is the max, 1.5277.**
- **Control: c_flat is 1.273 / 1.393 at phases 0 / 3.75.** So plane stress alone under-reads the 3D inner-face tension about 1.3–1.4x on the part's own geometry, and the stand-in adds only 4–6% (c_band / c_flat 1.062 / 1.042). **Hypothesis, not finding:** the gap is the band's bending ACROSS the face, which R1's `s_zz` shows and plane stress cannot hold. That rests on one genome and one 3D rung.

**D6.** `node_p` = 64, the smallest of 8 / 16 / 32 / 64 whose node p-norm is within 2% of the max at all eight phases: 1.3%, against 3.3% at 32 (P6.4 holds). The weight is `stress_margin`'s 89.21.
- **At `b729e86` on the whole stand-in, in the objective: util_band 1.5917 and `band_margin` 55.912.** R5's registered 20–27 is refuted: it left out D5's max-c and the phase p-norm.
- **A revised estimate, 46–56, was registered before the figure was computed, and it held.**
- **The term over-reads the 3D worst phase (33.80 / 25 = 1.352) by 17.7%:** 5.2% from the max-c (1.5277 against 3.75's 1.452) and 11.7% from the eight-phase p-norm at `stress_phase_p` 8. Conservative by construction, as D5 decided, and the size is now on the record.

**THE HEADLINE TRADE: cos(∇band_margin, ∇deflection) = −0.990.**
- Softening the spokes, which the target needs, raises the band's tension. Every thickness gene's band gradient is negative, and its deflection gradient positive.
- At the start the band's gradient is 0.0715x the deflection's in norm. P7.1 below quotes 0.076, which is the t2 component alone.
- Recorded and not acted on, as D6 said.

**Registered for Step 7, before any descent (P7.1):** the descent lands SHORT of the target, at a predicted-3D eight-phase mean of 1.90–1.97 mm. The crude linear estimate is 1.92, from the gradient ratio and both terms' shapes. **Below 1.90 fails Step 8's window by construction.** That would mean the band's price, not the stand-in, sets the drop, and it goes back as a trade to decide: the band's knee or weight against the target.

**A DEVIATION FROM R5 D6, RECORDED:** the band term is NOT in `OBJECTIVE_TERMS` or `DEFAULT_WEIGHTS`. That tuple is the requirements layer's list of priority axes (`wheel_requirements.priority_axes`), and a new axis reshapes its SHOULD rows, reference deviations and weight derivation, which Step 5 did not ask for. The term exists only where the stand-in is: `WO.CROWN_STANDIN["band"]` is its third value, and `--crown-standin` applies all three. With `band` None the breakdown is unchanged.

**Pinned by:**
- `test_gradient.py`: the band QoI's adjoint against a central difference on `t3`.
- `test_objective.py`: the term absent unless named, and priced as `stress_margin` when it is.
- `test_pool.py`: pooled equals serial to the bit on the whole stand-in, band term included.
- `test_stage3.py`: the switch is all three values or none, and the run record reads the stand-in off the evaluator.

**Scope:** one genome, SVK, the service load, 2D `coarse` for the calibration, the ladder at three phases on the stand-in only. The 3D figures are R1 / R3's: one 3D rung, nodal values.

### R7 — 2026-09-25. STEP 7 REGISTERED BEFORE LAUNCH: `svk-shipped`'s RECIPE ON THE STAND-IN, 300 STEPS AT FOUR WORKERS. AND P7.1 WAS DERIVED ON A GENE THE DESCENT CANNOT MOVE: `t2` SITS ON THE 1.2 mm FLOOR IN `b729e86`, AS DOES `t1`.

**The argv.** `svk-shipped`'s own flags (`Makefile`, the `svk-shipped` recipe), with `--crown-standin` added and distinct outputs:

```
wheel_stage3 --crown-standin --start best --genome best_solution.json --config coarse
  --kinematics svk --min-wall 1.2 --steps 300 --workers 4 --phase-scheme uniform
  --n-phase 8 --fidelity-check-every 0 --log-every 1
  --out stage3_crown_standin.json --best-out stage3_crown_standin_best.json
```

It runs under a `systemd-run --user` scope with MemoryMax 55G and MemorySwapMax 0, the cap `svk-shipped` names. A `/proc` side log outside the scope reads every process's VmHWM every 30 s.

**Two departures from the plan as handed over (60 steps, serial), and why.**
- **300 steps, not the default 60.** P7.1 predicts where the descent SETTLES. The start is 34% short of the target (predicted-3D 1.3178 mm, deflection term 290.9, against the band term's 55.9). `cosine_lr` decays to zero at the last step, so a 60-step run that lands short could be out of budget, not balanced by the band. That is a second cause, and it would make P7.1 unreadable. 300 is the recipe's length, and §181 ran it at `coarse` to exit 0 in 6.98 h.
- **Four workers, not serial.** The `coarse` pool is measured: `POOL_GIB` is (11, 11) over 300 steps (§171, §181). `test_pool` pins pooled equal to serial to the bit on the whole stand-in, band term included (U3). Serial would take about 18 h at `b729e86`'s ~215 s/step.

**P7.2 — memory. Registered:** no worker's VmHWM passes 11.0 GiB, and the scope never reaches its cap. The stand-in band is 2.144 mm thicker, so the mesh is larger than any mesh `POOL_GIB` was measured on. **Falsifier:** a worker mark over 11.0 means the pair does not cover the stand-in. That gets recorded whether or not the run finishes.

**Wall time: an estimate, not a registered prediction.** 7–11 h. Steady s/step will be read from steps 2–5 and projected before step 10.

**THE CONFOUND IN P7.1, FOUND WHILE WRITING THIS.**
- R6 quotes the band / deflection gradient ratio as 0.076 "for the t2 component alone", and the 1.92 estimate rests on it.
- `b729e86` has `t1` = `t2` = 1.2 mm, exactly `--min-wall`: normalised z = 0.0 on both.
- The deflection term wants those genes THINNER, and `project` holds them at the floor. So the component P7.1 was sized on cannot move.
- So a drop that lands short has at least three plausible causes: the band's price, the wall floor, or another term (`stress_margin`, `mass`, `smoothness`).
- **P7.1 is left exactly as registered. What changes is how it is read.**

**THE READ THAT SEPARATES THEM, registered now.** At the selected genome, one serial objective call returns each term's gradient.
1. List the genes on a bound.
2. Restrict each term's gradient to the FREE genes, and project it on deflection's free-gene gradient.
- **"The band sets the drop"** needs the band's projection to cancel at least half of deflection's there.
- **If it is the floor instead:** the free-gene deflection gradient is small against its full-gene norm, and the floor genes carry the rest. Then the trade to decide is the floor against the target, not the band's knee.

**Also reported, as Step 7 and §6 require:**
- every gene's move, in millimetres and in normalised units;
- the spoke profile overlaid on `b729e86`'s;
- the tier-0 selection step, and whether it differs from the literal last step, as it did at `b729e86` (§115's fillet_cap pathology).

### R8 — 2026-09-25. STEP 7 DONE: THE DESCENT LANDS AT A PREDICTED-3D MEAN OF 1.9463 mm, INSIDE P7.1's 1.90–1.97. BY R7's REGISTERED READ THE BAND SETS THE DROP: ON THE FREE GENES IT CANCELS 72–83% OF THE DEFLECTION GRADIENT. THE FLOOR IS ALSO ACTIVE: THE WHOLE LOSS STILL PUSHES `t1` INTO 1.2 mm. AND THE BAND GOES UP 9.8% TO GET THERE.

**The run.** R7's argv, exit 0, 23714.5 s (6.59 h, 77.8 s/step over steps 2–300). R7's 7–11 h wall estimate was high.
- Record: `stage3_crown_standin.json`. Selected genome: `stage3_crown_standin_best.json`, genome_hash `64e5068`, at **step 142**, loss 140.078, tier 0.
- **It is not the lowest loss in the run** (139.953, step 290). By `selection_key`, 208 of the 301 steps carry a barrier and 2 more sit inside the 1 µm cap slack. `fillet_cap` is nonzero at 163 steps, first at step 30; `x_order` and `arrival` account for the rest. Step 142 is the lowest loss among the 91 tier-0 steps. It is the same pathology `b729e86` was selected past (§115), at the same rate: 86 of 123 steps carried a barrier in that run's resumed leg (69.9%), 208 of 301 here (69.1%).

**P7.2 HOLDS.** No process passed 11.0 GiB in 789 samples, 30 s apart.
- Workers' VmHWM: 10.087 / 10.213 / 10.281 / 10.357 GiB. Parent: 10.404 GiB.
- Scope: at most 50.47 GiB of its 55G cap. The box never had under 2.80 GiB available.
- So `POOL_GIB`'s `coarse` pair (11, 11) covers the stand-in mesh, 300 steps, with 0.643 GiB to spare on the largest worker.

**P7.1 HOLDS: 1.9463 mm, inside 1.90–1.97.** The crude estimate was 1.92, which is 0.026 mm under the landing.
- P7.1 was sized on `t2`'s gradient (R7), and `t2` did not move. So the hit is on the registered range, not on its derivation.
- Against Step 8's window [1.90, 2.10], the landing is 0.046 mm inside the low edge. D7's 2% per-phase factor check is worth 0.039 mm at this drop, so Step 8's first pass can fail on the factor alone.

```
  at step 0 (b729e86 on the stand-in)  ->  step 142 (64e5068)
  predicted-3D mean drop       1.3178 -> 1.9463 mm       phase ripple (std/mean)  4.16% -> 2.00%
  stress_utilisation           0.756  -> 0.918           kt_hub / kt_rim   3.124 / 1.815 -> 2.841 / 2.061
  band_utilisation (priced)    1.5917 -> 1.7483          buckling_ratio    0.156 -> 0.197
  mesh mass (stand-in mesh)    59.77  -> 57.59 g         min scaled Jacobian 0.408 -> 0.333
  loss 400.79 -> 140.08: deflection 290.88 -> 1.80, band_margin 55.91 -> 80.22,
                          mass 49.13 -> 47.34, smoothness 4.87 -> 8.32, stress_margin 0 -> 2.40
```

**THE GENES** (`studies/study_crown_step7.py` → `.json`, `.jpg`):

```
  gene      b729e86     64e5068      move mm    move z    bound
  cx1        2.4566      2.0751     -0.3815    -0.0426
  cy1       24.9130     27.9515     +3.0385    +0.0475
  cx2       19.6900     19.6900      0          0         HIGH (at start too)
  cy2       21.6545     25.0173     +3.3629    +0.0525
  cx3       23.1354     22.5104     -0.6250    -0.0529
  cy3       16.7656     15.0860     -1.6796    -0.0262
  cx4       25.1988     25.0600     -0.1388    -0.0155    LOW
  cy4        6.4416      5.0186     -1.4230    -0.0222
  t0         3.5055      2.9863     -0.5192    -0.0590
  t1         1.2000      1.2000      0          0         LOW (at start too)
  t2         1.2000      1.2021     +0.0021    +0.0003    (0.002 mm off the floor)
  t3         2.4547      1.9406     -0.5141    -0.1071
  R_hub      0.5710      0.6009     +0.0299    +0.0083
  R_rim      1.6802      0.8857     -0.7945    -0.3178
```

- **The largest move is `R_rim`**, the rim-junction fillet: −0.795 mm, −0.318 of its range. That is why `kt_rim` rose 1.815 → 2.061.
- **The spoke thinned at both ends** (`t0` −0.52, `t3` −0.51 mm) and its arch rose about 3 mm near the hub (`cy1`, `cy2`).
- **The overlay** (`study_crown_step7.jpg`) shows the same shape family, a taller and thinner arch. Whether it still "looks right" is the user's call at Step 8 (§6).

**R7's ATTRIBUTION, AS REGISTERED.** Isolated per-term gradients at `64e5068`.
- **The instrument checks out.** Every term reproduces the run's value and gradient norm to the printed digit. The five sum to the run's recorded gradient within 3.8e-5 on a norm of 163.1 (2.3e-7 relative). **Not bit-exact, cause unmeasured.** The hypothesis is that the descent's Newton solves were warm-started and these were not.
- **The driver reproduces itself.** Run twice, from the scratchpad and then from `studies/`, the per-term gradients agree to 0.0.

```
  projection on deflection's free-gene gradient    t2 free      t2 pinned
  |deflection|, free / all                         640.5/809.0  453.8/809.0   (0.792 / 0.561)
  band_margin                                      -0.8347      -0.7165
  stress_margin                                    -0.0554      -0.1147
  smoothness                                       -0.0291      -0.0579
  mass                                             +0.0359      -0.0069
  net                                              +0.1167      +0.1040
```

- **By R7's rule the band sets the drop.** It cancels 83% (t2 free) or 72% (t2 pinned) of deflection's free-gene component, where the bar was half. Its cosine with deflection there is −0.978.
- **The floor half of R7's rule does not fire.** The free genes carry 79% / 56% of deflection's norm, which is not "small".
- **But the floor is ACTIVE, and R7's rule did not ask about that.** On `t1` the summed loss gradient is **+128.1**. Deflection's +493.8 outweighs the band's −383.1, so the loss as a whole still wants `t1` thinner, and only the bound stops it. `t2` sums to +58.6 with 0.002 mm to give.
- **So both constraints hold the landing.** The band on the free genes, the floor on `t1` / `t2`. **Hypothesis, not finding:** relaxing either would move the drop up. Gradients at one point cannot say by how much. Only a descent with the floor lowered, or the band re-weighted, can, and each changes one thing.
- **The net +0.10 to +0.12 is not zero.** Step 142 is not a stationary point on the free genes. It is where the tier-0 floor stopped the selection, not where the descent settled. The lr was 0.0055 there. The literal last step's drop, 1.9519, is 0.0056 mm higher.

**THE BAND GOT WORSE, AND THAT IS THE TRADE THE DESCENT PRICED.** The priced utilisation rose from 1.5917 to 1.7483 (+9.8%).
- **Hypothesis, not finding:** assume R6's 17.7% over-read of the 3D worst phase carries to the new spokes. Then the 3D inner-face worst phase moves from R1's 33.80 to about **37.1 MPa**. That is against the 25 MPa allowable and the 40 MPa printed ultimate (R1), leaving about 7% to the ultimate.
- That rests on the same transfer D7 exists to test. Step 8's 3D run measures it, and it is the number to read first there.

**Scope.** One descent, one start, `coarse`, SVK, uniform 8 phases, the stand-in rim. The attribution is a first-order read at one genome. Every drop here is PREDICTED 3D (the 2D drop × `drop_factor` 0.88867), until Step 8 measures one.

**NEXT: STEP 8's GATE IS THE USER's (§6).** The spoke look (overlay), then the 3D run. `b729e86` stays in `best_solution.json`.

### R9 — 2026-09-26. STEP 8's LOOK GATE PASSED BY THE USER ("Spokes look decent"). `64e5068` IS EXPORTED, AND THE 3D RUN's PREDICTIONS ARE REGISTERED HERE, BEFORE ANY SOLVE, WITH THE COMMAND THAT RUNS IT.

**The gate (§6).** The user reviewed `export/stage3_crown_standin_best.step` and R8's overlay and passed the spokes. No trust region and no frozen genes. `b729e86` stays in `best_solution.json` until the 3D run below clears Step 8.

**The export.** `make export EXPORT_GENOME=stage3_crown_standin_best.json` → `export/stage3_crown_standin_best.step` (+ `_nofillet.step`, `_step_manifest.json`), on the default rim, which is the chosen (1.5, 1.0).
- OCC: one valid solid, Ø102.00 × 22.40 mm, 50864.3 mm³, 63.07 g PLA (`b729e86` on the same rim: 65.35 g).
- STEP health: BRepCheck valid, not self-intersecting, min curvature R 0.5108 mm (floor 0.25).

**2D per-phase at `64e5068` on the stand-in.** One serial `t3_terms` call, `coarse` SVK, the whole `CROWN_STANDIN`. It reproduces the pooled run's step-142 mean, 1.946262 mm, and band utilisation, 1.74828. The raw 2D drops (before K) and the band's node p-norm per phase:

```
  phase    2D drop b729e86   2D drop 64e5068   band 2D p64 MPa   c_band (R6)   pred 3D hoop T   R1 3D at b729e86
   0.00       1.45825            2.16693             21.11           1.352          28.53             26.72
   3.75       1.55358            2.23456             24.53           1.452          35.61             33.80
   7.50       1.57288            2.25325             23.75           1.361          32.32             29.83
  11.25       1.53990            2.23501             23.16           1.439          33.33             28.41
  15.00       1.48631            2.19887             21.54           1.465          31.56             26.49
  18.75       1.43828            2.16122             20.37           1.525          31.07             24.17
  22.50       1.40843            2.13631             18.23           1.528          27.85             21.01
  26.25       1.40540            2.13450             16.82           1.289          21.69             19.97
  mean                           2.19008   x K 0.88867 = 1.94626 predicted 3D
```

**REGISTERED FOR STEP 8's 3D RUN (P8.x), before any solve:**
- **P8.1, D7's check, run first:** 3D / 2D(t_eq) holds to within 2% of `b729e86`'s combined factors, 0.87620 at phase 0 and 0.87370 at 3.75.
  - Phase 0: predicted **1.8987 mm**, band [1.8607, 1.9366].
  - Phase 3.75: predicted **1.9523 mm**, band [1.9133, 1.9914].
  - **A miss is not a halt (D7).** Refit t_eq on these spokes at (1.5, 1), re-descend warm, and record which transfer moved.
  - **Confound, stated now:** P8.1 tests two transfers at once, t_eq's and K's. A miss says one failed, not which. Only D7's refit separates them.
- **P8.2, Step 8's own gate:** the eight-phase 3D SVK mean is in [1.90, 2.10].
  - Point prediction: **1.9376 mm**. That is K × the 2D mean, less the +0.45% by which the stand-in over-read `b729e86`'s measured 3D mean (1.31773 predicted, 1.31186 measured).
  - The low edge is 0.038 mm away, inside D7's 2% (0.039 mm). So this is a live test, not a formality.
  - **Falsifier (Step 8's):** outside the window, the part does not ship.
- **P8.3, the band:** inner-face hoop tension, worst at phase 3.75, **35.6–37.1 MPa**.
  - Two routes give the range: c_band(φ) × the 2D p-norm per phase (35.61, and the p-norm under-reads the max by ≤ 1.3%, D6), and R8's global ratio (37.1). Every phase's prediction is in the table.
  - It is over the 25 MPa allowable at every phase, and that is priced, not barred (0.3).
  - **Falsifier:** any phase over **40 MPa**, R1's printed ultimate. That goes back to the user as the band weight or knee against the target, before any promotion.
- **P8.4, `s_zz`:** reported, not gated. Step 0.5's coupons are outstanding, and the 12.5 MPa placeholder is already exceeded on `b729e86` (16.78, R1).
- **P8.5, the patch (Step 0.1):** ONE patch at every phase is the pass/fail. **Hypothesis, not a gate:** the scrub lever is 0.67–0.85 mm, against `b729e86`'s 0.671–0.726 on this rim (R4). Softer spokes should lengthen the patch a little.
- **Cost, an estimate:** 8 solves × 430–750 s at 14–18 GiB (R4's six flat phases: 425–608 s, 14.3–17.0 GiB), so ~60–100 min.

**HOW TO RUN IT** — nothing else on the box:

```
cd <repo root>
tmux new-session -d -s step8 "OUT=<a scratch dir> studies/probe_3d/q_step8.sh"
```

- **`q_step8.sh` is R4's queue with the genome swapped.** It copies the STEP and the five `probe_3d` tools into `OUT`, then:
  - runs phases 0 and 3.75 first, then the other six;
  - rotates each phase with `rot.py`;
  - meshes at h 2.0 / hc 0.25 (gmsh retried up to 3×, §203);
  - solves `fe3d.py --faces free --kinematics svk --r-out 51.0` in a 32G scope, retrying a refused window at +4 then +8 mm;
  - ends with `patch3d.py` → `collate_patch.txt` and `post3d_cyl.py` → `collate_cyl.txt`.
- **Its log is `OUT/q.log`, one line per mesh and per solve.** Watch that file for `solve|NO MESH|FAILED|DONE`. A queue without a watch finished 2.5 h unseen (R3).
- **The 3D venv and GL libs** are the defaults in the script (`V3D`, `GLLIB`), under a 2026-09-23 scratchpad in `/tmp`. If `/tmp` has been wiped, rebuild per PLAN.md §202's toolchain: a python3.12 venv from `requirements-3d.txt`, plus libGLU / libOpenGL unpacked from `.deb`s.

**What the next record (R10) reads, in this order:**
1. **D7 at phases 0 and 3.75** (P8.1), as soon as those two solves land. A miss can stop the queue early.
2. **Every solve's force control:** `patch3d.py`'s ∫p dA must equal fe3d's 33.36165 N. The window edge is blind otherwise (R3).
3. **The eight-phase mean** against [1.90, 2.10] (P8.2).
4. **The band's inner-face hoop tension** per phase against P8.3's table and the 40 MPa bar.
5. **`s_zz`** (P8.4), and the **patch count and lever** (P8.5).

- **If Step 8 passes, Step 9 is the promotion.** It goes through `tests/test_promotion.py`'s checklist, as one atomic commit, never one file.
- **Promotion's other open question: the drivers' measured constants were taken on `b729e86`.** The stand-in record (`WO.CROWN_STANDIN`) and the `--crown-standin` switch describe a rim, not a genome, and do not move.

### R10 — 2026-09-26. STEP 8's 3D RUN: THE DROP PASSES (1.9421 mm, D7 HOLDS AT BOTH PHASES), BUT THE BAND DOES NOT. THE IN-LAYER PRINCIPAL TENSION AT THE PHASE-0 JUNCTION TOE IS 39.30 MPa, 39.65 REFINED, AGAINST THE 40 MPa PRINTED ULTIMATE. P8.3's HOOP READ PASSES BY ITS LETTER AND UNDER-READS THAT POINT. `64e5068` IS NOT PROMOTED.

**What ran.** `q_step8.sh` exactly as R9 gives it, one serial queue, nothing else on the box. Eight SVK solves, 519–664 s each, 16.9–18.7 GiB peak, every one on its default window at the first try. Then two solves R9 did not register, both at phase 0 and both explained below: a re-solve on a window widened 4 mm (649 s, 19.0 GiB), and a refinement at hc 0.20 (1018 s, 23.2 GiB).

**THE FORCE CONTROL CAUGHT R3's WINDOW DEFECT AT PHASE 0, AND IT COST NOTHING.**
- `patch3d.py`'s ∫p dA read 33.36768 N against fe3d's 33.36165 N (+1.8e-4). Its assertion stopped the script, so no other phase was read.
- The cause is the one R3 recorded. Phase 0's patch reaches x −1.034, past the window's x0 −1.00, which was fitted to `b729e86`'s patch. One OD triangle straddling that edge penetrates 0.39 µm and carries the 0.0060 N. fe3d never saw it, because its candidates and its truncation check use only triangles wholly inside the window.
- Re-solved on x [−5.00, 8.00], the control passes to 1e-13. The drop moved −1.4e-7 relative, 1.9170804 → 1.9170802 mm. The band figures are unchanged to the printed digit.
- The other seven phases pass at the first try, each to 1e-13. **All eight fields below pass the control.**

**P8.1 (D7) HOLDS AT BOTH PHASES.** The combined factor 3D / 2D(t_eq) sits within 2% of `b729e86`'s:
- Phase 0: **1.91708 mm** against 1.8987 [1.8607, 1.9366]. Factor 0.88470 against 0.87620, **+0.97%**.
- Phase 3.75: **1.97033 mm** against 1.9523 [1.9133, 1.9914]. Factor 0.88175 against 0.87370, **+0.92%**.
- **No refit.** R9's confound stands: this is t_eq and K tested together, and a pass says neither failed by more than the other made up.

**P8.2, STEP 8's DROP GATE, HOLDS: THE EIGHT-PHASE 3D SVK MEAN IS 1.94209 mm, INSIDE [1.90, 2.10].**
- That is 0.042 mm clear of the low edge, and 2.9% under the 2.0 mm target.
- Against R9's point prediction 1.9376: **+0.23%**. Against plain K × the 2D mean, 1.94626: **−0.21%**. The −0.45% correction R9 carried over from `b729e86` did not transfer: the stand-in over-read this genome by 0.21%, not 0.45%. With two genomes, nothing here says which of the two predictors is the better one.
- 3D / 2D over the eight phases is 0.88677, against K 0.88867 (−0.21%).

```
  phase    3D SVK drop    3D/2D(t_eq)   vs K x 2D   in-layer s1 max   hoop+ max   P8.3 hoop   s_zz max   lever mm (b729e86)
   0.00     1.91708        0.88470       -0.45%       39.30             37.32       28.53       16.22      0.681 (0.671)
   3.75     1.97033        0.88175       -0.78%       33.31             32.75       35.61       16.19      0.733 (0.724)
   7.50     1.99102        0.88362       -0.57%       31.75             31.75       32.32       17.17      0.733 (0.726)
  11.25     1.98091        0.88631       -0.27%       31.13             31.13       33.33       17.11      0.731 (0.722)
  15.00     1.95440        0.88882       +0.02%       29.93             29.93       31.56       16.83      0.727 (0.717)
  18.75     1.92395        0.89021       +0.17%       28.25             28.25       31.07       16.54      0.722 (0.710)
  22.50     1.90256        0.89058       +0.22%       26.25             26.25       27.85       16.06      0.716 (0.701)
  26.25     1.89649        0.88849       -0.02%       28.80             27.77       21.69       14.99      0.705 (0.672)
  mean      1.94209        0.88677       -0.21%
  phase 0 at hc 0.20:  1.91711 (+0.0017%)             39.65             37.06

  MPa, Cauchy, element nodes, band r >= 48.45, |x| < 10, post3d_cyl.py.  s1: max principal
  tension, its direction in the layer (z-share 0.00 at every phase).  hoop+: its hoop
  component's max, P8.3's quantity.  Phase 0 is the widened-window field.  One patch at
  every phase, 100% of the force.
```

**P8.3 PASSES BY ITS LETTER: NO PHASE's HOOP TENSION REACHES 40 MPa. ITS SHAPE IS REFUTED, AND ITS COMPONENT IS THE WRONG ONE AT THE POINT THAT MATTERS.**
- **The worst phase is 0, not 3.75.** The hoop reads 37.32 MPa, 0.6% over the registered range's top of 37.1. Phases 0 and 26.25, which are neighbours across the 30° period, read **+30.8% and +28.0%** over their per-phase predictions. The other six read 1.8–9.1% under.
- **R8's global-ratio route came closer than R9's per-phase one.** R8 put the worst phase at about 37.1 MPa, 0.6% under the measured 37.32. R9's c_band(φ) × p-norm put the worst at 3.75.
- **On `b729e86`, s1 and hoop agree to within 1.1 MPa at all eight phases** (re-read now from §206's fields: s1 27.80 / 33.85 / 29.83 / 28.42 / 26.49 / 24.17 / 21.01 / 20.91). That is R1's "the max principal points along the hoop at every peak", and P8.3 was written on it. **On `64e5068` at phase 0 they split by 1.98 MPa at the same node:** s1 39.30, hoop 37.32. The direction is still in the layer, with no z-share, but it is no longer the cylinder's hoop.
- **The 40 MPa bar is the printed ultimate for tension along the filaments**, 50 × 0.80 (`wheel_fea`, R1). s1 with no z-share IS that tension, and hoop is one component of it. So the bar applies to s1: **39.30 MPa at the rung every other figure uses, a factor of 1.018 to the ultimate. `b729e86`'s worst is 33.85 (1.182).**
- **Refined, it climbs.** Q20, registered in the session scratchpad before the mesh existed, used hc 0.25 → 0.20 with a box x [−3.0, 2.5] × y [−51, −47.5] over the toe:
  - Q20.1, "the hoop peak moves < 1%", HOLDS: 37.32 → 37.06, −0.70%.
  - Q20.2, "the drop moves < 0.05%", HOLDS: +0.0017%.
  - **The s1 peak rose 39.30 → 39.65 (+0.9%), a factor of 1.009.** One refinement step cannot say whether or where it converges, so the 39.65 is quoted as a reading, not a limit. It is read at surface element nodes, the reading §203 found within ±1% under refinement on the band's OD. This peak is on the inner face, at a toe.
- **Hoop, s1 and the location on each genome, measured:**

```
            b729e86 peak     junction footprint    64e5068 peak     junction footprint
  phase     MPa @ x          r 48.2-48.44          MPa @ x          r 48.2-48.44
   0.00     26.72 @ -2.10    [-1.93, 4.56]         37.32 @ -1.43    [-1.31, 2.27]
   3.75     33.80 @ +0.85    [ 1.23, 7.67]         32.75 @ +1.62    [ 1.86, 5.32]
  26.25     19.97 @ -5.17    [-5.07, 1.39]         27.77 @ -0.65    [-4.47, -0.90]
```

- **Measured:** every one of the six peaks sits 0.10–0.38 mm outside the edge of the material that joins the band (the junction's footprint just inside the inner face). The footprint is **3.6 mm wide on `64e5068` against 6.5 mm on `b729e86`**. R8's largest gene move was `R_rim`, the rim-junction fillet, 1.680 → 0.886 mm.
- **Hypothesis, not finding:** the narrower junction left a toe beside the load at phases 0 and 26.25. There the band bends hardest, and the principal direction turns off the hoop along the fillet's surface. The two genomes differ in every spoke gene, so the runs behind this differ in more than one way. The design that could refute it is the successor below.
- **Implied c_band on `64e5068`** (3D hoop / 2D p64) is 1.768 / 1.335 / 1.337 / 1.344 / 1.390 / 1.387 / 1.440 / 1.651. That is **1.31x and 1.28x R6's** at phases 0 and 26.25, and 0.91–0.98x at the other six. The 2D term reads 0.5 mm clear of the corner (R6), and its calibration was taken on `b729e86`'s junction. The descent narrowed that junction under a price that could not see the toe.

**P8.4, `s_zz`:** 14.99–17.17 MPa, worst at 7.5, against `b729e86`'s 16.78 (+2.3%). Step 0.5's coupons are still outstanding, so there is no allowable to read it against. The 12.5 MPa placeholder is exceeded by both parts.

**P8.5 HOLDS: ONE PATCH AT EVERY PHASE, THE LEVER 0.681–0.733 mm** (hypothesis range 0.67–0.85). It is longer than `b729e86`'s at every phase: +1.0% to +2.1% at six phases, +1.5% at 0, and +4.9% at 26.25. Softer spokes, a longer patch, as registered.

**THE CALL: STEP 8 DOES NOT PASS, AND `64e5068` IS NOT PROMOTED.** `b729e86` stays in `best_solution.json`. The drop is right and the patch is right. The band fails the reason its bar exists: in-layer tension at 98–99% of the printed ultimate, rising under refinement, where the shipped part has 18% margin. R9 registered that a band over 40 MPa "goes back as the band weight or knee against the target". The component P8.3 named reads 37 MPa there only because it is the wrong one at a fillet toe. **Lesson, for any later band prediction: register the in-layer max principal, not a cylindrical component.** Hoop is the same number only where the surface is a cylinder.

**SUCCESSOR (R11), REGISTERED HERE BEFORE ANY SOLVE: ONE GENE.** `64e5068` with `R_rim` alone restored to `b729e86`'s 1.6802 mm, exported on the (1.5, 1.0) rim, SVK at phases 0 and 3.75, windows widened 4 mm.
- 2D on the stand-in (the run's own Evaluator, serial, `coarse`, SVK, step 142's phases):
  - **Control:** `64e5068` reproduces R9's predicted-3D mean, 1.9462621 mm.
  - With `R_rim` restored, the predicted-3D mean is **1.87661 mm (−3.58%)**, under Step 8's 1.90 edge. So restoring the fillet alone costs the drop window.
  - `kt_rim` 2.061 → 1.700, `stress_utilisation` 0.918 → 0.887.
  - **`band_utilisation` 1.7483 → 1.7389 (−0.5%).** The 2D band term barely registers the fillet. If R11.1 holds, that blindness is how the descent could spend it.
- **Registered:**
  - **R11.1:** if the hypothesis holds, the junction footprint widens toward `b729e86`'s 6.5 mm, and the worst in-layer s1 over the two phases falls to **≤ 35 MPa**, a factor ≥ 1.14.
  - **Falsifier:** ≥ 38 MPa. Then the fillet is not the lever, and the thinner spoke ends (`t0`, `t3`) are next.
  - **R11.2:** each phase's 3D drop moves by the 2D MEAN ratio, 0.96421, within 1.5%: phase 0 **1.8485 mm**, phase 3.75 **1.8998 mm**. The ratio is an eight-phase mean applied per phase, hence the wider band.
- **What follows from each outcome:**
  - **If R11.1 holds:** the band read has to reach the toe, or `R_rim` needs a floor. Either way Step 7 re-runs, and the descent then has to find the lost 3.6% of drop elsewhere.
  - **If it fails:** the toe is set by the spoke ends, and the same 2D blindness question moves to `t0` / `t3`.

**Scope.** One genome, one 3D rung (h 2.0 / hc 0.25), and one refinement step at one phase. The 2D is at `coarse`. The toe hypothesis rests on two genomes that differ in every spoke gene. The fields are in the session scratchpad, `/tmp/claude-1000/-home-eric-bodhi-github-wheel/a3321b31-5874-4406-9372-37eea1a1a028/scratchpad/step8/`: `m_s7_*.npz` / `r_s7_*.npz`, `r_s7_0_w4.npz`, `m_s7_0_q20.npz` / `r_s7_0_q20.npz`, `q.log`, `collate_cyl.txt`, `collate_patch_noassert.txt`, the registered `PREDICTIONS_q20.txt`, and `../rrim2d.py` / `../rrim2d.out` for the 2D rows above.

### R11 — 2026-09-26. R10's SUCCESSOR, RUN: RESTORING `R_rim` ALONE TAKES THE PHASE-0 TOE FROM 39.30 TO 31.04 MPa (−21%), AND THE DROP PAYS 4.1–4.7%. BOTH REGISTERED PREDICTIONS HOLD. THE TOE IS THE FILLET's, AND THE 2D BAND TERM CANNOT SEE IT.

**What ran.** `64e5068`'s genome with `R_rim` set to `b729e86`'s 1.6801683 mm, nothing else changed.
- Export: `src/wheel_step_export.py --genome <scratch>/s7_rrim.json --out-prefix <scratch>/s7_rrim`. An absolute prefix keeps the candidate out of `export/`. On the default (1.5, 1.0) rim: 51418.5 mm³, 63.76 g (+0.69 g), BRepCheck valid, min curvature R 0.5108 mm.
- Two SVK solves at phases 0 and 3.75, h 2.0 / hc 0.25, on R10's windows widened 4 mm, in 32G scopes: 656 / 630 s, 18.7 / 16.8 GiB.
- Both pass `patch3d.py`'s force control to 1e-13, one patch each, lever 0.676 / 0.731 mm.

```
                          64e5068            R_rim -> 1.6802       change
  phase 0   drop mm       1.91708            1.82681               -4.71%   (R11.2: 1.8485, -1.17%)
            s1 MPa        39.30 @ x -1.43    31.04 @ x -1.84       -21.0%
            hoop+ MPa     37.32              29.87
            s_zz MPa      16.22              12.64
            footprint     [-1.31, 2.27]      [-1.69, 3.75]         3.58 -> 5.44 mm
  phase 3.75 drop mm      1.97033            1.88916               -4.12%   (R11.2: 1.8998, -0.56%)
            s1 MPa        33.31 @ x +1.62    33.70 @ x +1.10       +1.2%
            hoop+ MPa     32.75              33.64
            s_zz MPa      16.19              16.30
            footprint     [ 1.86, 5.32]      [ 1.46, 6.88]         3.46 -> 5.42 mm

  Same instruments as R10: post3d_cyl.py (band r >= 48.45, |x| < 10, element nodes, Cauchy), and
  the junction footprint as the x-extent of material at r 48.2-48.44.  b729e86: 6.49 / 6.44 mm.
```

**R11.1 HOLDS: the worst s1 over the two phases is 33.70 MPa, ≤ 35.** The falsifier was ≥ 38. The ratio to the printed ultimate is 1.187, where `b729e86` sits at 1.182.
- **The headline rests on a one-variable design.** Only `R_rim` differs between the two genomes, and the outcome could have come back ≥ 38.
- Both peaks still sit on the footprint's edge, 0.15 / 0.36 mm outside. The fillet moves the toe away from the load at phase 0. At 3.75 the toe was not beside the load on either genome, and s1 moved +1.2%.
- **Scope:** two phases on one genome. Phase 26.25, R10's other toe phase, was not solved. That it follows phase 0 is an expectation, not a finding.

**R11.2 HOLDS:** each drop is within 1.5% of the 2D mean ratio applied per phase (−1.17% / −0.56%).
- **The fillet is worth 4.1–4.7% of the drop.** On the 2D mean that is 1.9463 → 1.8766 predicted-3D, **under Step 8's 1.90 edge.**
- **So R_rim alone is no fix.** The part it gives is safe in the band and too stiff for the target. This is the trade R8 found the descent making, now priced in 3D: the fillet's 0.8 mm bought 3.6% of drop, and 18% of in-layer margin at the toe.

**WHY THE DESCENT SPENT IT: THE BAND TERM IS BLIND TO THE FILLET.**
- Across the same one-gene change, 2D `band_utilisation` moves −0.5% (1.7483 → 1.7389, R10), while the 3D toe falls 21%.
- The band term reads the inner face 0.5 mm clear of the 2D corner (R6). The toe is where that exclusion sits.
- `kt_rim` does track the fillet, 2.061 → 1.700 (×0.825). But it has fed no loss term since §103 (it stays for the geometric report, per `t3_terms`' docstring).
- **Hypothesis, not finding:** one gene pair cannot calibrate either quantity against the 3D toe.

**SUCCESSOR (R12), REGISTERED:** make the descent see the toe, and measure in 2D before any descent.
- **The test:** find a 2D quantity whose `64e5068` → `R_rim`-restored ratio matches 3D's at both phases solved: **0.790 at phase 0 and 1.012 at 3.75, each within 5%**.
- **Candidates:**
  1. The band's inner-face tension read with `exclude_mm` under 0.5, with R6's corner checked under refinement at each exclusion.
  2. `kt_rim`, phase by phase.
- **A candidate that passes** gets priced, and Step 7 re-runs with it.
- **The fallback** is a floor on `R_rim`: a bound, not a price. Its value is not derivable from one pair. At the old value it costs the 3.6% the drop window does not have.
- **`b729e86` stays in `best_solution.json`.**

**Fields:** `/tmp/claude-1000/-home-eric-bodhi-github-wheel/a3321b31-5874-4406-9372-37eea1a1a028/scratchpad/r11/`. It holds `m_*.npz` / `r_*.npz`, `q.sh` (the queue), `q.log`, `collate_*.txt`, `junction.txt`, the genome JSON, the STEP and the manifest.
