# CROWN_PLAN.md — a crowned rim that steers through its contact patch, on a partly compliant base

**Open arc #11. Created 2026-09-24 from PLAN §206 §7.0, the trade §206 handed back. STEP 0 IS
DECIDED (§5, on the user's delegation, same day). No solve run; one free read of fields already on
disk is recorded at the foot and changed decision 0.3.** This file is the plan as written before any of it runs; records go at the foot, and the plan
above them is not rewritten to match them.**

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

