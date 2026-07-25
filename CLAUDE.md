# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A two-stage design pipeline for a compliant (flexure-spoke) PLA UAV wheel: a genetic
algorithm co-optimizes the spoke centerline shape and thickness taper against an
analytical beam model, then a CAD script turns the winning genome into a watertight
STEP solid for import into Fusion/Onshape/SolidWorks — two scripts and the JSON
artifacts they exchange.

Work is in progress to extend this into a three-stage pipeline (GA → nonlinear FEA →
gradient refinement); see `.claude/plans/` for the plan and its decision gate.

**The M4 gate has answered that decision gate, and the answer is that Stage 2 is
required.** The plan's Stage-2.5 off-ramp — correct the beam model with one factor and
skip the FEA — is closed: four GA winners whose beam deflections agree to 0.5% have axle
drops spanning **0.72 to 2.71 mm**. The beam model is accurate as a *spoke* model (0.8%
against a single-spoke FEA) and blind as a *wheel* model. See "The M4 gate" below for
the mechanism and for what Stage 2 must add.

## Commands

```bash
make env      # build BOTH virtualenvs (.venv-opt and .venv-cad)
make test     # pytest in env-opt — 168 tests, ~60 s (one is skipped without .venv-cad)
make smoke    # tiny GA run, seconds, proves the pipeline is wired up
make ga       # full GA run (300 pop × 300 gen, ~4 min), then hands off to the exporter
make export   # rebuild wheel.step from the existing best_solution.json
make studies  # all four verification gates (below), ~2 min

.venv-opt/bin/python study_mesh_quality.py --samples 2000   # M2a spoke-mesh gate
.venv-opt/bin/python study_wheel_mesh.py --samples 200      # M2b full-wheel mesh gate
.venv-opt/bin/python study_beam_agreement.py                # M3 beam-agreement gate
.venv-opt/bin/python study_wheel_fea.py                     # M4 full-wheel FEA gate
```

The four `study_*.py` scripts are **gates, not tests**: they produce measured reports
whose numbers are quoted in this file, and they exit nonzero on failure. The test
suite re-runs reduced versions of each so CI cannot drift from the published tables.

Useful flags on `wheel_fea.py`: `--seed N`, `--generations N`, `--pop N`,
`--cy-bound N`, `--no-export`, and `--out PATH` (which also relocates the summary
figure beside the genome, so experimental runs never touch committed artifacts).

**Two environments, and they cannot be merged.** `env-opt` runs the optimizer;
`env-cad` runs the exporter, which imports `wheel_fea` *across the interpreter
boundary*. Do not pin numpy in `requirements-cad.txt` — CadQuery 2.8 resolves it to
2.4.6, and forcing env-opt's version makes pip backtrack to a CadQuery whose `numba`
has no cp312 wheel, failing with a misleading "Failed to build numba".

The CAD hand-off is deliberately non-fatal (a GA run costs minutes; a missing env must
never destroy one), and it is skipped entirely under `--smoke` / `--no-export` so a
throwaway genome can never cause the STEP to be rebuilt from the *previous* real one.

## Architecture

### Module layout

```
        wheel_geometry.py     ← array module INJECTED (xp=numpy | jax.numpy)
          (never imports jax)
                │
   ┌────────────┼──────────────┬──────────────┐
   │            │              │              │
wheel_fea.py  wheel_genome.py  wheel_mesh.py  jax_config.py
   │            │                 │      │           │
   │            │    wheel_wheel.py      └── wheel_fem.py   (the solver)
   │            │    (full 360° assembly)
   │            │
   └── re-exports ──► wheel_step_export.py   (CAD env: numpy + cadquery)
```

`jax_config.py` enables float64 and must be imported before any other jax use.
float32 is not a performance trade here, it is a correctness one: the patch test's
1e-12 tolerance is below float32 epsilon, and a finite-difference gradient check has
no plateau at all because the roundoff floor sits above the discretization error at
every step size. Both failures read as bugs in the physics.

`wheel_geometry.py` is the single implementation of the spoke geometry, used by both
the numpy side (GA, exporter) and the JAX side (mesh, FEA). It takes its array module
as an argument rather than importing one, which is what lets one body serve both — and
what keeps `wheel_fea.py` importable from the CadQuery env.

**Two epsilons that look alike and are not.** `offset_normals` divides by
`||n|| + 1e-12`; that guard is load-bearing (a degenerate genome really can produce
coincident curve points) and it costs 1.6e-11 of relative normal length, because the
raw gradient magnitude is the ~0.06 mm sample spacing. `thickness_at_arc_length` used
to divide by `zone_width + 1e-12`; that one guarded a compile-time constant of 1/3 that
can never be zero, while stopping the taper from reaching its own interpolation nodes
(`t(1/3) != t1` by 1.2e-11). It has been removed. Verified downstream: re-exporting the
STEP changes no geometric quantity at all — same volume, face/edge/vertex counts,
fillet radii, and Kt.

### The mesh

`wheel_mesh.py` builds a structured quad block from the same offset-band
parameterisation the exporter uses, so the meshed part and the exported part are the
same surface by construction. Node positions are differentiable in the genes;
connectivity is a numpy constant. Q9 biquadratic by default — a Q4 mesh shear-locks in
bending, and a 2 mm flexure is exactly where that silently reports a design as stiffer
and safer than it is.

`wheel_mesh.band_sampler` is **the** evaluator for the spoke band, and everything that
touches the band goes through it — the spoke block and both junction patches. That is
not tidiness: two constructions that are mathematically identical but interpolate
different quantities disagree at O(h_curve²). Measured, the junction and the spoke block
once placed their shared cross-section **3.9e-4 mm apart**, which leaves a real kink at
every one of the 24 junctions and makes a 1e-10 seam assertion impossible to state
honestly.

`wheel_mesh.py` holds the single spoke block (what M3's verification consumes, since the
beam model clamps at the root and guides the tip). `wheel_wheel.py` assembles the full
360° wheel — see below.

**Mesh normals must come from the analytic hodograph, not `xp.gradient`.**
`offset_normals` (finite differences) is one-sided at the endpoints, so the flank
position there carries O(h) error and the *geometry itself moves* as the mesh refines —
measured, the top-flank tip walked 8.9 µm between n_span 32 and 64, halving on each
refinement. A convergence study on that mesh measures the geometry settling down rather
than the discretization. `normals_from_tangents` is resolution-independent (verified:
zero shift over a 32× refinement range). The exporter path keeps the finite-difference
normals, because the STEP on disk was built from them.

### Mesh validity is a hard constraint the GA never had

`study_mesh_quality.py` sweeps the design space (2000 feasible genomes) and is the M2a
gate, on the spoke block alone. Result:

| feasibility definition | minSJ > 0.2 |
|---|---|
| x-ordering + hub crowding (what `evaluate_design` enforces) | **59.5%** — FAIL |
| ...plus a fold-margin constraint | **99.0%** — PASS |

Roughly 40% of the genomes the existing constraints admit have **inverted elements**.
That is not a meshing defect — the offset band genuinely turns inside out where the
outward offset passes the centre of curvature, and `wheel_fea`'s `smoothness` term was
only ever an indirect proxy for the missing constraint (`wheel_fea.py:596-598` says so).

The closed-form `self_intersection_margin` predicts it, so the barrier costs one
curvature evaluation with no mesh in the loop. Use `MIN_FOLD_MARGIN_MM = 0.1`, not 0 —
measured, a threshold of 0 still admits 12 designs with minSJ as low as 0.027, while 0.1
gives **zero misses at 100.00% validity** and still keeps 93% of the design space.
This matters because an inverted element has negative stiffness, which reads to the
optimizer as free compliance — it will actively seek that region.

### wheel_wheel.py — the full 360° mesh, and why its topology works

Seven blocks per 30° sector × 12 sectors: `spoke`, `hub_junction`, `rim_junction`, and
each ring split into a `_weld` and a `_free` block. Splitting the rings rather than
grading them is what makes the *partial* seam exact — the weld arc's two ends become
block corners, so a contiguous run of ring nodes coincides with the junction's arc nodes
by construction instead of by a node distribution someone has to arrange to land on them.

**The spoke sweeps 45° of angle in a 30° sector, so the first guess is that no partition
exists.** Three measured facts save it:

1. **Radius is strictly monotone** along the centerline (12.700 → 48.500) while θ rises
   to 47.23° and returns to 0. Each spoke is therefore a single-valued θ(r), and twelve
   rotated copies of a single-valued θ(r) *cannot* intersect. Minimum clearance between
   adjacent thickened spokes outside the hub circle: **0.304 mm** for the shipped genome.
2. The weld footprint on each ring is **19.63°** (hub) and **15.65°** (rim) of the 30°
   available, so the twelve junctions tile each ring with a real gap. That footprint is
   also the wheel's dominant stiffness variable — see `spoke_free_arc_fraction` and the
   M4 gate.
3. The centerline endpoints are **locked on the ring circles** — (0,0) and (span,0)
   locally is r=12.700 and r=48.500 globally — and the end cross-section is symmetric
   about the centerline, so it crosses its ring exactly at its own midpoint. That
   junction corner is closed-form with no root-find.

Fact 1 makes a structured mesh possible; 2 and 3 make the junction blocks well-shaped
rather than slivers (corner angles ~80°/90°, not the 4–6° the near-tangent arrival
suggests, because the cross-section is *normal* to the centerline). Measured on the
shipped genome, the worst block is the hub junction at minSJ **0.889** — and it is worst
not because it is a bad patch but because it is the one bounded by a circular arc on one
side and a straight cross-section on the other.

**The arrival-angle constraint, and it runs the opposite way from intuition.** Junction
quality is set by the angle between the centerline and its ring's *tangent*, correlation
**−0.81** over 200 sampled genomes. A near-**tangent** arrival — the case that makes
`_embed` and the fillets hard — gives *excellent* junctions; a near-**radial** arrival
lays the cross-section nearly along the arc and collapses the corner:

| arrival from tangent | 4.9° | 7.2° | 64.8° | 82.2° |
|---|---|---|---|---|
| junction minSJ | 0.806 | 0.804 | 0.415 | **0.043** |

Past ~80° it stops being a meshing problem and becomes a design one: both flanks lie
outside the ring circle and the spoke touches its ring at the single centerline point.
That is a hinge, not a weld, and `ring_station` refuses to build it — which also means
`wheel_fea`'s `fixed_guided` BC would be modelling a moment connection the part doesn't
have. `arrival_angles` is closed-form with no mesh in the loop, so it can be an optimizer
barrier directly, exactly like `self_intersection_margin`. The boundary is sharp (the
refused builds ran 80.7–89.0°); `MAX_ARRIVAL_DEG = 65.0` gives zero misses with margin
and keeps 82% of the space. The shipped genome is at **4.4°** (hub) / **6.4°** (rim).

**Which flank straddles which ring is genome-dependent, and the two ends are
independent.** `cy1..cy4` span ±32 so a spoke may bulge either way, and an S-shaped
centerline approaches the tip from the far side — so the flank that is inside at the hub
can also be inside at the rim. Hardcoding the shipped genome's `(+1, +1)` rejected **44
of 60** feasible genomes. `flank_orientation` computes it in numpy and it is passed as a
*static* argument: it is a topological fact about the genome, not a smooth parameter.

### The M2b gate

| check | result |
|---|---|
| area vs the region modelled | −0.032% at `coarse` → **−0.0024%** at `fine`, converging |
| **seam error over 200 genomes** | **3.5e-14 mm** (required < 1e-10) |
| minSJ > 0.2, geometric + fold feasibility only | 90.00% — FAIL |
| ...plus the arrival-angle constraint | **100.00%** — PASS |
| inverted elements | 0 |

**The area reference is derived, not transcribed.** `wheel_wheel.modelled_area_reference`
computes `hub disk + rim band + 12 clipped spoke bands` from the frame constants and the
genome, down the *exporter's* geometry path (finite-difference offset normals, exact
shoelace plus exact circular sectors) against the mesh's (analytic hodograph, Coons
patches, Q9 Gauss). Two different constructions agreeing to 2e-5 relative is evidence;
a hardcoded number is not. This was learned the hard way — the previous references
2469.836 / 2521.438 were measured at `RIM_RADIUS_MM = 48.9` and silently became
references to a wheel that no longer exists the moment the band was thickened. The OCC
cross-check (2469.06) is from that same dead frame and is no longer quoted.

**What is modelled**: `hub_disk ∪ rim_band ∪ 12 spoke bands clipped to the annulus`. Two
deliberate differences from the shipped STEP, both measured:

- **Fillets are not modelled, and for area they don't matter** — measured at 0.29 mm²
  (0.01%) for all twelve. They still matter for *stress*, which is why
  `stress_concentration_kt` is retained rather than deleted.
- **`wheel_step_export._embed` is not reproduced, and it does matter**: it adds about
  **4.3 mm² per spoke** inside the annulus, so this mesh models **1.93% less material**
  than the shipped part, all of it at the junctions where it acts as a gusset. The same
  gap shows up independently in mass — 72.43 g from the mesh against the manifest's
  73.84 g, −1.91% — which is a different kernel measuring the same difference. `_embed`
  picks its direction and length by an `argmax` over 21 blends × 20001 lengths, so
  reproducing it would put a discontinuous, non-differentiable search in the gradient
  path — the "gene with no FD plateau" failure M7 gates on. A smooth alternative does not
  exist either: the bottom flank's backward tangent **misses the hub circle entirely**,
  which is the same fact `_embed`'s own comment records from the other side.

The collar is meshed only over r ∈ [7.7, 12.7]; inside that the hub is rigid (the
assumption the beam model already makes). **Any area check must add the rigid core back**
— π·7.7² = 186.27 mm², 7.5% of the wheel. Leaving it out is how a mesh-vs-CAD comparison
comes out 7% low and gets "explained" by discretization.

### wheel_fem.py — write the energy, derive everything else

`element_energy` is the only physics in the file. The internal force is its gradient
and the tangent stiffness is its Hessian, both via `jax.grad`/`jax.hessian` and
`vmap`ped over elements. There is no B-matrix, no hand-differentiated stress recovery,
and no separately coded tangent that can drift out of step with the residual — the
whole class of bug where hand-rolled nonlinear FEA usually dies is structurally absent.
It also makes frame indifference a property of one function rather than a cancellation.

Both kinematics ship: `linear` (ε = sym ∇u) and `svk` (Green–Lagrange), differing by
five lines and sharing one strain-energy function. That is deliberate — a rigid-body
test that passes for the linear kernel too is not testing frame indifference, it is
testing that the rotation was small. M3 verifies with `linear`; M5 adds the Newton loop.

Plane stress with `λ = Eν/(1−ν²)` reproduces Euler–Bernoulli exactly in the slender
limit with **plain E**, the same E the Castigliano model uses, which is what makes the
beam comparison a check rather than an argument. The real 22.4 mm × 2 mm section
(aspect 11) behaves closer to plane *strain* — effective modulus 1.14·E — so
`plane="strain"` exists and the choice is recorded in the result. That 14% is larger
than most effects this project chases; do not let it be picked silently.

Constraints go through one sparse transform, `u_full = T·u_red + u_pre`
(`DofMap`): homogeneous and inhomogeneous Dirichlet, skew direction-only constraints,
and rigid ties all in the same machinery, giving an SPD reduced system. `DofMap`
raises rather than handing a singular matrix to `spsolve`, because a dropped
constraint otherwise produces garbage with only a warning.

`edge_traction_load` builds consistent nodal loads on any of the four block sides, with
the outward normal determined geometrically against the owning element's centroid
rather than from a per-side sign table that every new block type would have to extend.
The correct Q9 edge weights are 1/6, 4/6, 1/6 — equal lumping gives the same resultant
and the wrong answer, which no equilibrium check catches.

### Two boundary-condition findings that dominate every beam comparison

**The root BC is a first-order contaminant, and `clamped` is the wrong one.** A fully
clamped root forbids the lateral Poisson strain `ε_yy = −νMy/(EI)` the bending field
genuinely has there. That perturbation decays over ~t and its energy is **O(t/L)**
relative to the beam's — first order, in a comparison whose entire purpose is to
measure an O(t²) effect. Measured on a straight cantilever, excess over `FL³/3EI`
divided by (t/L)² (Timoshenko says 0.81, constant):

| L/t | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| `root_bc="plane"` | 0.905 | 0.883 | 0.873 | 0.865 |
| `root_bc="clamped"` | 0.600 | 0.411 | 0.031 | **−0.733** |

The clamped root does not merely contaminate the number at high slenderness, it
**reverses its sign** — the FE comes out stiffer than Euler–Bernoulli, which reads
exactly like shear locking. Use `root_bc="plane"` for every beam-theory comparison.
It is *not* the right BC for the real wheel, where the root is a meshed junction and
collar rather than an imposed constraint, which is why this only matters here.

**Element size scales with the wall thickness, not with the part.** The span
discretization has to resolve the root and tip boundary layers, whose length scale is
`t`. Measured at the thinnest section of the A4 sweep (t = 0.25 mm), the FE-vs-beam
discrepancy runs −0.52% at `h ≈ t`, +0.0013% at `h = t/4`, +0.00275% at `h = t/32` —
the coarse mesh gets the **sign** wrong. `n_thick` is nearly irrelevant by comparison
(converged at 4 elements; going to 24 moves the answer by 5e-6 relative). This is the
most consequential number M3 produced for M4: a mesh-config table that sizes elements
by the part will silently misreport thin features. `study_beam_agreement.sized_config`
implements the rule and `test_mesh_resolution_must_scale_with_thickness` pins it.

Corollary for reference values: `generalized_spoke_mechanics` integrates a polyline,
and at its production `N_CURVE_PTS = 600` that polyline is short by **2.1e-5
relative** — larger than the discrepancy being measured at λ = 1/8. The study runs the
beam reference at 38400 points, where its own error is 7.7e-8.

### The M3 gate, and why it is not the gate the plan asked for

The plan's A4 was one slenderness sweep, on the curved spoke against Castigliano, with
the fitted exponent required to be ≈ 2. Running it showed that formulation cannot
settle the question: Castigliano's error is a **sum** of several O(t²) effects
(transverse shear, the Winkler-Bach neutral-axis shift, straight-beam EI on a curved
member) which for this geometry partially cancel. The residual decays, and *faster*
than t² — local exponents 2.19, 2.42, 3.58, fitted 2.70 — and no defect produces a
steeper decay. Gating on an upper bound there fails a correct element for being too
accurate.

So the element gate moved to the **straight beam**, whose reference is an exact closed
form with no discretization of its own and whose O(t²) coefficient is known
analytically. Measured: fitted exponent **2.02**, local exponents 2.034/2.018/2.014,
mean coefficient **0.881 vs Timoshenko's 0.810** (+8.8%), every discrepancy positive.
That requires the error not just to shrink but to shrink at the right rate *to the
right number*, which single-point agreement cannot do. Locking would appear as the
wrong sign, an exponent near 1, or a coefficient near zero; all three are checked.

The curved-spoke sweep is kept as a reported measurement. Its useful output is the one
number at λ = 1: **the Castigliano model over-stiffens the real spoke by 0.81%**
(fixed-guided; 0.09% cantilever). That is the entire single-spoke error of the model
that drove the whole GA — small, one-directional, and now quantified. It is *not* the
error of the model applied to the wheel, which is what M4 measures and which the rim
band is expected to dominate.

### The M4 gate — what the wheel actually does

Full-wheel linear FEA: hub rigid and fixed at r = 7.7, ground pressure over an assumed
patch at the bottom of the rim. `study_wheel_fea.py`.

**THE HEADLINE: the beam model is an accurate spoke model and a blind wheel model, and
the blindness is not a bias that can be corrected.** The plan's Stage-2.5 off-ramp was
"if the beam model is merely biased, apply one factor and skip Stages 2 and 3". That
factor would be `axle_drop / beam_deflection`, and it is not a number — it ranges over a
factor of ~30 across the feasible design space. Worse, at the design point: four GA
seeds whose *losses agree to 1.2%* and whose *beam deflections agree to 0.5%*

| seed | 1 | 42 (shipped) | 2 | 3 |
|---|---|---|---|---|
| beam δ (mm) — what the GA optimised | 1.994 | 1.990 | 1.997 | 2.000 |
| **axle drop (mm) — what the wheel does** | **0.720** | **1.682** | **1.873** | **2.712** |
| axle drop, rim rigidified (mm) | 0.211 | 0.445 | 0.497 | 0.681 |
| spoke free arc fraction | 0.667 | 0.755 | 0.781 | 0.865 |
| rim weld footprint (°) | 24.44 | 15.65 | 14.38 | 7.53 |

A factor of **3.8** in the quantity the whole project optimises, across genomes the
objective cannot tell apart. Reproduce with `wheel_fea.py --seed N --no-export --out
/tmp/x.json`.

**The mechanism is the spoke's effective length, and identifying it took a control.**
A single-spoke FEA agrees with Castigliano to **0.76–0.80%** for all four — M3's measured
number, reproduced — so the spoke model is not what is wrong. What differs is how much
spoke there is: `generalized_spoke_mechanics` integrates the full hub→rim span, while the
built part is fused into its rings over the two weld arcs, so only
`wheel_wheel.spoke_free_arc_fraction` of it flexes. A bending compliance goes as the cube
of the free length and (0.865/0.667)³ = 2.2 against an observed 3.8, the remainder being
that the consumed arc sits at the *ends*, where the moment arm is longest.

The first explanation offered was the rim's free span between welds, and the control
falsifies it: rigidifying the rim band leaves the spread at **3.2×**. Had it been rim
bending, the four would have collapsed onto each other. This is why `run_beam_blindness`
reports the rigid-rim column per row.

**This is a specification for Stage 2, not only a negative result.**
`spoke_free_arc_fraction` is closed-form with no mesh in the loop, exactly like
`self_intersection_margin` and `arrival_angles`, so a GA term on it captures most of the
effect with no FEA at all.

**The compliance split** (with the 1.5 mm band and the re-run genome):

| where the compliance is | share of strain energy |
|---|---|
| 12 spokes | 66.2% |
| **1.5 mm rim band** | **32.4%** |
| hub collar | 1.4% |

It was **44.7%** with the 1.1 mm band that shipped before M4. The conclusion is robust —
the share moves 0.6 points across the whole mesh ladder and stays in 26–33% over a 12×
sweep of the assumed patch size — and the rim remains a first-order term the beam model
omits *entirely*.

**Two different questions get called "how much is the rim", and they have different
answers.** The energy share (32.4%) is exactly a *first-order sensitivity*: for a linear
body under one load system `U = Fδ/2`, so scaling a region's compliance by (1+ε) moves δ
by share·ε. Rigidifying the rim instead removes **73.3%** of the axle drop, because a
stiff rim does not merely stop storing energy — it **spreads the load**. With a floppy
rim, two or three spokes near the contact carry everything; with a stiff one every spoke
shares it. Quote both or neither.

**Rim thickness sweep.** The band that would hit 2.0 mm with the *current* genome is
**1.26 mm** against the 1.50 mm built, i.e. the built band overshoots and the wheel comes
out at **1.68 mm**, 16% stiffer than target. That is headroom, not a defect, and it is
not worth chasing: the 2.0 mm target is a beam-model quantity, and the headline above
shows retuning the band to hit it for *this* genome would miss for the next one. The fix
belongs in Stage 2's objective.

**Phase ripple**: 7.4% std/mean, 19.8% peak-to-peak over one 30° period (1.68 mm with a
spoke under the contact, 1.38 mm between spokes) — halved from the 14.3%/38.9% of the
1.1 mm band, but still large. The plan's M6 asked whether the phase machinery is needed —
it is. This also makes phase a design objective in its own right.

### Two M4 numbers that are NOT converged, and why that is a geometry fact

**Peak stress is not a number here.** It diverges under refinement — rim region 30.3,
38.9, 44.9, 52.4 MPa across the ladder — because this mesh has no fillets, and an
unfilleted spoke/ring junction is a **349.5° re-entrant corner: geometrically a crack**.
Its field goes as r^−0.5 and no mesh resolves it. Only the p99 converges: **8.84, 8.78,
8.61, 8.61 MPa** in the plain spoke block, against the beam model's 25.1 MPa peak.

**Excluding the junction blocks is not enough to escape it.** The plain spoke block's own
*maximum* also diverges (27.7, 35.6, 46.1 MPa) because the block's end cross-section sits
one element from the corner and still samples the singular field. Only a percentile is
safe. That the corner is nearly a crack is exactly *why* the real part is filleted there
and why `stress_concentration_kt` exists.

**The same crack caps the convergence rate of every global quantity.** The axle drop's
observed refinement ratio is 1.66 where clean second order would be 4, so the plan's
"axle drop within 0.5% of the Richardson value" criterion is **just missed** — the finest
mesh is 0.504% off, GCI 0.63%. This is not a meshing defect and refining further will not
fix it: `run_refinement` now runs the control automatically, widening the contact patch
to 12° / 89 nodes, which leaves the rate at 1.61 against 1.66 — so the patch is not the
cause. Meeting 0.5% requires meshed fillets, which requires the fillet-feasibility fix
`kt_report` documents as open. A 0.5% bound on the axle drop is two orders of magnitude
below the 3.8× effect the gate measures and far below the ±20–30% uncertainty in `E`, so
the gate's conclusions stand.

### Two M4 things that were not done, and one that does not exist

- **The CalculiX cross-check was not run** — no `ccx` on this machine. It remains the one
  recommended check not performed, and the only one that would independently catch a
  systematically wrong assembly. The distorted-mesh patch test, M3's closed-form beam
  agreement, and exact equilibrium stand in for it.
- **M4b, measuring the printed wheel, has not happened.** Every absolute number above
  rests on `E = 2300 MPa` and an unconditional 0.80 FFF knockdown, both uncalibrated.
  ±20–30% material uncertainty swamps most effects this project chases.
- **The plan's mirror-symmetry check at φ = 0° and 15° does not exist for this geometry.**
  The wheel is *chiral* — twelve spokes all spiral the same way — so a reflection maps it
  onto a different wheel. Rotational periodicity is the real invariant and it is checked
  to 1e-10 instead. It earned its place: it caught a formulation bug where the contact
  phase moved the patch *around the rim* rather than rolling the wheel under a fixed
  ground, which is a different load case (and, with the traction taken along the rim's
  radial normal rather than the ground's vertical one, was also applying a spurious side
  load). `build_wheel(phase_deg=...)` rolls the wheel; the ground never moves.

### The genome is the interface

Everything flows through `best_solution.json`. `wheel_fea.py.__main__` writes it;
`wheel_step_export.py` reads it and hashes the `genes` dict (`genome_hash`) so a STEP can
never be silently attributed to a different design. Extra top-level keys are safe to add —
the exporter reads only `genes` and `metrics.total_mass_g`.

The 14 genes (`GENE_SPACE`, `wheel_fea.py:180`): 8 interior Bézier control-point
coordinates (degree-5 centerline, endpoints locked at hub and rim), 4 thickness nodes
`t0..t3` over 3 linear-taper zones, and 2 fillet radii `R_hub` / `R_rim`.

**`RIM_RADIUS_MM` is the frame the genome is expressed in, and it changed.** M4's rim
sweep moved it 48.9 → **48.5** (band 1.1 → 1.5 mm, thickened *inward* so Ø100 is held),
which shortens `HUB_RIM_SPAN_MM` 36.2 → 35.8 and therefore **reinterprets every gene on
disk**. `tests/test_golden.py::test_the_artifact_was_produced_by_these_constants` fails
loudly if the artifact was optimised in a different frame, because a stale genome in a new
frame is a different spoke that still loads, meshes, and solves without complaint.

### wheel_fea.py — import-safe by design

Top-level imports are numpy only; **pygad and matplotlib are imported inside
`if __name__ == "__main__"`**. This is load-bearing: the exporter runs in the CadQuery env
and does `from wheel_fea import ...` to reuse the *exact* geometry functions
(`generate_bezier_centerline`, `thicken_3taper_curve`, `stress_concentration_kt`) plus the
dimensional constants. Adding a top-level heavyweight import breaks the exporter.

Physics lives in `generalized_spoke_mechanics` (`wheel_fea.py:308`) — a Castigliano
force-method solve with two redundants at the rim tip. Scoring lives in `evaluate_design`
(`wheel_fea.py:550`), which returns `(metrics, loss_terms)` as separate dicts so `__main__`
can print the per-term breakdown; `pygad_fitness` just negates the sum. With seven weighted
terms spanning four orders of magnitude, **tune weights by reading the loss-term table, not
by reasoning about the constants** — the v2.0 mass and buckling terms were both effectively
dead and nobody noticed because nothing printed them.

`BOUNDARY_CONDITION` (`wheel_fea.py:141`) switches between `"fixed_guided"` (the real part:
spoke fused into a stiff rim ring) and `"cantilever"` (legacy free-tip). They differ by ~4×
in stiffness, so a genome is not interpretable without the `model` block that gets saved
alongside it. Cantilever is not a separate code path — it is the same solve with both
redundants zeroed, and that algebraic collapse is the regression check on the solver.

### wheel_step_export.py — 2D first, one extrusion

The wheel is assembled entirely as a **planar face** (hub disk + 12 spokes + rim band, one
n-ary fuse, then clipped to Ø100) and extruded exactly once. Every 3D construction it used
to use produced geometry OCC accepts and Parasolid rejects.

Two hard-won invariants, both documented at length in the source:

- **Never move a flank point to embed the spoke into a ring.** Interpolating the spline
  through a displaced endpoint loops the curve and leaves a ~40 µm cusp that OCC calls
  valid and Onshape drops the face over. Embedding is straight tangent segments appended to
  the wire (`_embed` / `spoke_profile`), which cannot add curvature.
- **`despecialize()` runs last, and volume is measured before it.** Converting swept
  surfaces to B-splines must happen on the *filleted* solid (doing it earlier yields an
  invalid shape), and `BRepGProp` quadrature over B-spline faces inflates volume ~5% on
  geometry whose boolean difference from the original is exactly zero.

`profile_health` is the check the earlier versions lacked: `BRepCheck_Analyzer` does not
test for cusps, and `BRepAlgoAPI_Check` only ever saw the post-union solid after the
self-intersecting loop had been trimmed away.

### Known open discrepancy

`kt_report` compares the fillet radii the optimizer priced into its stress model against
what OCC could actually build. Check the `fillets.detail` block in
`wheel_step_manifest.json` before trusting a reported stress number.

**There is no spoke↔hub junction on this part.** `_embed` uses one common extension
length for both flanks, so driving `bot[0]` (r=13.935) inward to `HUB_EMBED_RADIUS_MM` =
12.20 forces `top[0]` to overshoot to r=9.77 and swings the extension ~22° tangentially.
Adjacent spokes lap over each other before either reaches the hub circle. Measured on the
shipped genome:

| | |
|---|---|
| cylindrical faces in the unfilleted solid | r=48.5 ×12, r=50.0 ×1 — **nothing at r=12.7** |
| material wedge at the hub circle, mid-sector | **360.0°** — buried, not a boundary anywhere |
| what exists instead | a 12-fold spoke↔spoke notch at r=12.8748, **354° of material** |

That 6° notch is **sharper than the 349.5° corner M4 calls a crack**, and OCC refuses
every radius down to `MIN_CURVATURE_RADIUS_MM` = 0.25 mm — correctly, since rounding a 6°
notch at even 0.25 mm would have to run 4.8 mm along both flanks. **So the hub ships
square**, and the as-built peak stress scales 25.1 → **~47.2 MPa against a 25 MPa
allowable**. This is the largest known error in the shipped part.

Two things this used to hide, both now fixed:

- **`_select_junction_edges` picked edges within ±0.02 mm of a ring radius**, on the
  stated premise that "junction vertices lie EXACTLY on the ring circles". False — the
  hub corner is 0.175 mm out, nine times the tolerance, so the export printed
  `no junction edges near r=12.70 mm — skipped` and moved on. `_junction_edges` now
  selects by **re-entrancy** (material wedge > `MIN_FILLET_WEDGE_DEG` = 185°, measured by
  probing a ring with a solid classifier), so it fillets the corners that exist wherever
  `_embed` left them. Radius is a label for grouping, never a selector. The five families
  of full-height vertical edge separate cleanly — 354° (hub notch), 152° (`bot[0]` kink,
  *convex*), 180° (`bot[-1]`, straight), 194–356° (the 24 rim corners), 178° (OD seam) —
  so nothing lands near the threshold. Note the flank kinks are convex, not shallow
  re-entrant: a "sharpness" threshold above 185° would wrongly drop the twelve 194° rim
  edges that build fine today.
- **`kt_report` emitted `NaN` when nothing was built, and the `[Kt MISMATCH]` gate
  filtered non-positive `r_built` out** — so the worst case in the report was the one
  case that could not raise the alarm, and the manifest carried bare `NaN` tokens that
  are not valid JSON. A square junction is now priced through the branch
  `stress_concentration_kt` already had for it (`< 0.1 mm → 3.5`), giving
  `kt_error_pct = +88.1%` and firing the gate.

`fillets.detail` now carries `n_edges_found` / `n_edges_filleted` separately — a bare `0`
used to mean both "the selector found nothing" and "everything was found and nothing
could be built", which want opposite fixes.

**The fix that remains is upstream**, and it needs a term the plan's design lacks:
`tangent_fillet_arc`'s discriminant runs on the raw thickened outline and reports *2* hub
arcs where the CAD builds *0*, because it never sees `_embed`. Any GA barrier needs an
adjacent-spoke clearance term as well — M4's mesh work already computes one (**0.304 mm**
for this genome) and it goes negative precisely when the hub junction is destroyed.

Also note the mass figures are not comparable: `metrics.total_mass_g` counts spokes only
(48.56 g), while `solid.mass_g_pla` in the manifest includes the hub disk and rim band
(73.84 g). `study_wheel_fea.wheel_mass_g` computes the whole solid from the mesh and gets
72.43 g — 1.91% under the manifest, the same `_embed` difference as the area, and now
read from the manifest at report time rather than transcribed.

## Tests

`tests/` is the safety net for the three-stage refactor. What it pins:

- **Golden regression** — `evaluate_design` on the on-disk genome reproduces
  `best_solution.json`'s `metrics` and `loss_terms` to 1e-9 (it currently matches to
  ~1e-15). Values are read back out of the artifact, not transcribed, so the test stays
  valid when the genome changes.
- **Import hygiene** — a subprocess asserts `import wheel_fea` leaves `jax`, `pygad`,
  and `matplotlib` out of `sys.modules`, plus every symbol
  `wheel_step_export.py:60-69` reaches for still exists. This is the contract that
  makes the two-interpreter split work, and it is one stray convenience import away
  from breaking in an env nobody runs tests in.
- **CLI contract** — same seed gives an identical genome (and a *different* seed gives
  a different one, which catches a seed that is accepted but ignored), and no
  experimental run writes anything into the repo directory.
- **The FE element** — a patch test on a *distorted* mesh to 1e-12 relative, for Q4 and
  Q9, on displacement and on recovered stress, in both a Dirichlet and a traction
  version. A patch test on a rectangular grid is nearly vacuous: the Jacobian is
  diagonal, so a transposed inverse-Jacobian cancels out. Distortion is what makes the
  test able to fail — and it did, on the first run, catching exactly that transposition.
  The traction version is the only check on `edge_traction_load`, which M4's distributed
  contact pressure and M6's penalty contact both go through; lumping a quadratic edge
  equally onto its three nodes gives the right resultant and fails it.
  Plus exactly 3 zero-energy modes (a 4th would be an hourglass mode), and finite
  rotation at 30° storing zero energy under SVK while the linear kernel's spurious
  energy is asserted to match `2(λ+μ)(cos θ − 1)²V` in closed form.
- **The FE-to-mesh coupling** — `wheel_fem._NODE_IJ` must be the same permutation as
  `wheel_mesh.spoke_block_connectivity`'s vertex ordering. A mismatch yields an element
  that is still symmetric, still positive definite, and still passes a rigid-body test;
  it just integrates a scrambled geometry. The test rebuilds the expected ordering from
  the mesh module's own index arithmetic rather than from a transcribed copy.
- **The beam gates** — reduced-fidelity reruns of `study_beam_agreement.py`'s A1/A2/A3
  and both A4 sweeps, including an assertion that `root_bc="clamped"` *fails* the
  element gate, so nobody can "simplify" the root treatment away and then read the
  result as shear locking.
- **The full-wheel FEA** — exact 12-fold periodicity of the axle drop (1e-10), hub
  reaction equal and opposite to the applied load, a zero horizontal resultant (the
  ground pushes along ITS normal, not the rim's), and the work identity δ = 2U/F tying
  the reported energies to the reported displacement so the compliance split cannot
  drift from the drop it decomposes. Plus assertions that the peak stress **diverges**
  and the plain-spoke p99 settles, and that the junction is still a >340° wedge — so if
  fillets are ever meshed, the tests say so rather than silently changing meaning. The
  p99 ladder starts at `coarse`, not `smoke`: on the smoke mesh the p99 is not yet in its
  asymptotic range, and a successive-difference test there reads convergence as
  divergence.
- **That the beam model cannot be corrected** — a reduced-fidelity rerun of
  `run_beam_blindness` asserting the beam-to-wheel ratio spans more than 3× and that
  `spoke_free_arc_fraction` is **not** constant over the design space. The second half
  matters as much as the first: an explanatory variable that does not vary explains
  nothing, and a Stage-2 objective built on one would be worthless. If the first ever
  passes, the entire justification for Stages 2 and 3 needs re-reading.
- **The fillet contract, through the artifact** (`tests/test_export_contract.py`) — the
  exporter lives in `env-cad` and `make test` runs in `env-opt`, so most of this reads
  the committed `wheel_step_manifest.json`: strict JSON with no `NaN`, finite `kt_built`
  and `kt_error_pct`, exact re-entrant corner counts (12 hub / 24 rim), and that a fillet
  built *smaller* than requested is priced *worse* — a sign slip there would let an
  under-built fillet read as safety margin. One test is guarded on `.venv-cad` existing
  and cross-checks the solid classifier against a 2D face classifier on every corner:
  different OCC algorithm, different shape, so an inverted inside/outside sense cannot
  cancel. That matters because an inverted sense would select the twelve *convex* flank
  kinks instead of the twelve re-entrant notches and every count in the manifest would
  still look sane.
- **That the area reference is derived** — the reference must move by exactly the
  closed-form annulus when `rim_outer` is swept, which catches a reference that has been
  re-hardcoded. Plus `_clip_polygon_to_disk` against three shapes with exact answers,
  including the containment branch where returning zero is wrong in a way that shows up
  only on the reference.
- **The assembled wheel, three independent ways** — a seam mismatch produces a mesh that
  plots correctly, has positive Jacobian everywhere, solves without complaint, and models
  a wheel with twelve cracks in it. No solver diagnostic notices. So: shared coordinates
  agree to 1e-10; the element graph is **one connected component** (catches seams declared
  but not merged); no two global nodes coincide (catches the same, differently); and
  12-fold periodicity is exact to 1e-9 (catches sector-indexing bugs the other three
  cannot — wiring sector k to k+2 leaves all of them happy). Plus the area partition per
  region, `MAX_ARRIVAL_DEG` being attached to a measured correlation rather than a guess,
  and an assertion that `flank_orientation` is **not** constant over the design space so
  it cannot be "simplified" back to one.

## Artifacts

`best_solution.json` (genome + metrics + loss breakdown + gene bound saturation),
`wheel_step_manifest.json` (what the STEP actually is: genome hash, source mtime, fillet
feasibility, profile/STEP health), `wheel.step`, `wheel_nofillet.step` (guaranteed-valid
fallback written before any fillet), `poster_summary.jpg`, plus the four gate reports
`study_mesh_quality.json`, `study_wheel_mesh.json`, `study_beam_agreement.json` and
`study_wheel_fea.json` (and their figures). All are committed. The gate reports hold **summaries only** — the raw
sampled genomes are behind `study_mesh_quality.py --rows PATH` because at the gate's 2000
feasible genomes that is 45769 draws and 28 MB of JSON, exactly reproducible from
`--seed`. The exporter
prints `[STALE]` when the STEP on disk predates the genome — this happened for real once,
which is why `wheel_fea.py` now drives the export itself instead of leaving it a manual step.

When a run reports genes pinned at their `GENE_SPACE` bounds: a pinned `t*` means
`MIN_WALL_MM` printability is the active constraint (informative, fine).

A pinned `cy*` looks like the same signal but **is not** — do not widen `CY_BOUND_MM`.
This was measured (sweep over {32, 45, 60}, 4 seeds each):

| bound | mean loss | sd | best |
|---|---|---|---|
| 32 | 50.30 | 0.44 | 49.78 |
| 45 | 52.91 | 1.65 | 51.08 |
| 60 | 55.16 | 1.64 | 53.73 |

Monotonically worse, far outside the noise. The tell is that at bounds 45 and 60 **no
`cy` gene is pinned at all** — the optimizer settles interior at |cy| ≈ 30–41 and is
still worse, so the extra room is reachable and simply not wanted. The pinning at 32 is
a boundary artifact of a shallow, y-symmetric, multi-modal landscape; enlarging the box
just costs search efficiency at a fixed budget. Reproduce with
`wheel_fea.py --cy-bound N --seed S --no-export --out /tmp/x.json`.
