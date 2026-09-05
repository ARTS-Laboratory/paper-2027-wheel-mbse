# UNCAP_PLAN.md — the mesh's second junction corner is manufactured, and it carries the peak stress

**~~Open arc #2, promoted to the top.~~ Created 2026-08-17 from PLAN §34 Findings 1 and 4.
~~Nothing started.~~ NOT CHEAP — read the cost section, and read the premise section first
because it is unusually load-bearing.**

**STATUS CORRECTED 2026-09-05 — PLAN §114. TWO THINGS IN THAT HEADER WERE FALSE.** *"Nothing
started"* — STEP 3 RECORD PARTS 2 through 10 are in this file, recorded at §104, and PART 10
already carries `THE DECISION: THE FAITHFUL RIM IS NOT ADOPTED`. And *"#2, promoted to the
top"* was this file's claim about itself, never ratified: `PLAN.md`'s open-arcs table gives #2
to `FILLET_PLAN.md` and carried **no UNCAP row at all** until §114 added one. **The arc is now
PARKED — see the park record at the foot of this file.** Read the records, not the plan above
them; the plan text is kept unedited because a plan quietly rewritten to match its outcome is
not a record.

**VERSION CONTROL IS PART OF THIS PROJECT'S WORKFLOW — CHANGED 2026-08-19.** The rule that
stood here read *"Ignore version control entirely. Do not commit, branch, stage, revert or
otherwise touch git."* **It is superseded.** The rules live in `PLAN.md`'s header block and
**only** there, so they cannot drift across ten files: one commit per finished unit of work
on `feature`, `make test` green first, never while a study driver is mid-write, a study
commit carries its regenerated `.json` and `.jpg`, a promotion is one atomic commit and never
one file — and **commits carry no assistant or tool attribution, no `Co-Authored-By:`
trailer, no session link, no generated-with footer.**

**Read PLAN.md §34 and `FILLET_PLAN.md`'s STEP 1 RECORD PARTS 2 and 4 first.** This arc
exists because of measurements taken there. It does not cancel `FILLET_PLAN.md`; it
*blocks* it, and §34 Finding 4 is the reason.

---

## Why this arc exists — and what is already known and settled

**The mesh's `P_c` corner is not a corner the shipped part has, and it is where the
wheel's global peak stress lives.**

Two measured facts, both from §34:

1. The mesh terminates the spoke on the ring circle and closes it with a half **end
   cap**; the exporter drives both flanks straight through the ring, so the shipped solid
   has no cap. `P_t` matches the part to 0.073 um of arc; the mesh's `P_c` sits 1.53 deg
   (hub) and 1.32 deg (rim) from where the part's second corner actually is, with wedge
   angles 28.7 and 87.5 deg apart.
2. The global peak von Mises is **11-16 um from `rim:P_c`** — on it — at `coarse` and
   `medium`, under both `linear` and `svk`, against 1.15 mm to `rim:P_t`. At `fine` both
   artefact corners outrank both real ones: `rim:P_c` 150.59 > `hub:P_c` 120.92 >
   `hub:P_t` 96.22 > `rim:P_t` 75.40 MPa.

**SETTLED, DO NOT REOPEN WITHOUT NEW EVIDENCE:**

- **The difference is documented and deliberate.** `wheel_wheel.py`'s module docstring
  (*WHAT IS AND IS NOT MODELLED*) says `_embed` is not reproduced, prices it at 3.03 mm2
  per spoke / ~1.4% of material "all of it at the junctions where it acts as a gusset",
  and states the reason: `_embed` picks its length by an argmax over 20001 candidates
  (and 21 blend directions at the rim), and putting that in the coordinate map would be
  the non-differentiable-gene failure M7 gates on. **That justification is sound and this
  arc does not attack it.** Reproducing `_embed` is not the proposal.
- **The end cap is FORCED, not chosen.** The bottom flank never crosses either ring
  circle along the spline — closest approach **0.6589 mm** (hub) and **0.5656 mm** (rim).
  With the spoke cut at the ring circle there is simply no second corner to put `P_c` on,
  so "just move `P_c`" is not an option and must not be re-proposed.
- **`P_t` is real, correctly placed, and is NOT what this arc touches.**

## THE ONE THING THE RECORD GETS WRONG, AND IT IS THIS ARC'S OPENING

The docstring's justification ends: *"A smooth alternative does not exist either: the
bottom flank's backward tangent MISSES the hub circle entirely (its closest approach
exceeds 12.7)."*

**Measured on the shipped genome, that closest approach is 12.0771 mm — inside 12.700.
It reaches.** A straight extension of the bottom flank along its own end tangent crosses
r = 12.700 after 1.7815 mm and r = 48.500 after 0.9190 mm. That is a line/circle
intersection in closed form: **smooth, differentiable in the genes, no argmax, no blend
search.** The M7 objection that correctly rules out reproducing `_embed` does not reach
it.

So a smooth alternative *does* exist. **That is not the same as a smooth alternative
being better, and this arc's whole risk is confusing the two:**

```
                                     hub theta (deg)    distance from the part's corner
  the shipped part's 2nd corner         -1.52673              -
  the mesh's end cap (today)            +0.00000            1.53 deg
  a bottom-flank tangent extension      -8.73684            7.21 deg      <- WORSE
```

**On position, the end cap is five times closer to the truth than the smooth alternative
is.** Position is not the only thing that matters — wedge angle drives the singularity
and is what Finding 4 is about — but any claim that the tangent extension is an
improvement has to be *measured*, and the one number available so far points the other
way.

## The cost, stated up front

**This changes what every measured number means.** The junction block is bounded by the
end cap, so replacing it moves welded volume, and therefore mass, the hub/rim compliance
split, and every constant calibrated against them. Expect `tests/test_golden.py` and
anything with a pinned constant to redden, and every `study_*` artifact to need a refresh
(`make studies` — 4 h 15 m for the four slowest drivers alone, §33).

**So do not build anything until Step 1 says it is worth building.** Steps 0 and 1 are
cheap and can kill the arc.

## THE PLAN

### Step 0 — the instrument, and the baseline

Port this session's numpy reconstruction of `_embed` into
`studies/study_junction_agreement.py`: for a genome, report every flank/ring crossing of
the PART outline beside the MESH's corners, with wedge angles and spoke-side legs — i.e.
reproduce §34 Finding 1's table from scratch. **That table is this arc's
before-and-after instrument, and it costs seconds, not the 95 minutes `make gci` costs.**

Validate it the way §34 did: the crossing count must come out 24 and 24, matching the
shipped manifest's `hub_edges`/`rim_edges`.

Baseline is already in hand — `make test` 476/2/0 and `study_corner_singularity.json`
unchanged since §30. Confirm; do not re-run.

### Step 1 — is the tangent extension actually better? Answer this before any mesh work

**This is the go/no-go and it needs no mesh at all.** For the tangent-extension geometry,
compute at both rings, in 2D:

- the second corner's position, against the part's -1.52673 / -1.31586;
- its **wedge angle**, against the part's 268.47 / 219.90 and the cap's 297.18 / 307.43;
- Williams' lambda for that wedge, i.e. whether the singularity gets weaker or stronger;
- the weld footprint it implies — 14.85 deg at the hub and 2.74 deg at the rim, both
  inside the 30 deg sector, so the tiling survives, but check the neighbouring sector's
  clearance rather than assuming it.

**The claim to test:** the tangent extension puts the second corner closer to the part's
*in wedge angle* even though it is further away *in position*, and lambda moves toward
the part's. **If it does not, stop and report that** — the end cap is then the better of
the two available idealisations, Finding 4 becomes a recorded limitation rather than a
task, and this arc closes without touching the mesh. That is a legitimate and cheap
outcome, and it is the one the position table above currently predicts.

### Step 2 — only if Step 1 says go: build it, and price it one number at a time

The junction block's `right` edge stops being the end cap and becomes the extension. Node
counts are unchanged, so `_seam_table` should survive untouched — §34 Finding 2 showed a
corner can move without disturbing it, and the same argument applies here.

Then measure, against Step 0's baseline, at `coarse` and `medium`: mass,
`compliance_split`, axle drop, `min_scaled_jacobian`, and the four corner peaks.
**Report the deltas; change no threshold.** A gate that reddens here is evidence.

### Step 3 — the payoff, checked rather than assumed

1. **Does the global peak leave the artefact corner?** Re-run `make corner`. The specific
   claim: the new global max sits on a corner the part has.
2. **Does the fillet become tractable?** Re-run `FILLET_PLAN`'s radius sweep. The
   prediction is that without an end cap the cross-section that folded the spoke block
   does not exist. **If it still folds at 0.2 mm, that prediction was wrong and it gets
   recorded as wrong.**
3. Only then is `FILLET_PLAN.md` Step 2 reachable.

## What must NOT happen

- **Do not reproduce `_embed`.** Its argmax search is exactly the M7 failure the
  docstring rules out, and that ruling is correct.
- **Do not "just move `P_c`" to the far-flank crossing.** There is no such crossing: the
  bottom flank misses both circles by 0.66 / 0.57 mm. Measured, settled.
- **Do not treat the smooth alternative as an improvement because it is smooth.** Step 1
  exists precisely to stop that, and the position table predicts it is worse.
- **Nothing in `best_solution.json` is touched; nothing is promoted or re-descended
  inside this arc.** A junction-topology change and a design change in the same arc are
  indistinguishable afterwards.
- **Do not loosen a threshold that reddens** (§19), and **do not delete a test that starts
  failing** — update it with the new measurement in its docstring (§31).
- **Do not take the kernel default for a SENSITIVITY claim** (§33). Step 3 item 2 in
  particular.
- **Do not begin with `make gci`** (§30).

## Honest summary of this arc's odds

The finding that motivates it is solid: the peak stress is measured, twice, on a corner
the part does not have. **The remedy is not.** The only smooth remedy identified so far
lands further from the part's geometry than the thing it would replace, and Step 1 is
built to find that out in an afternoon rather than after the mesh is rebuilt. **Treat a
Step 1 "no" as the expected outcome, not a failure** — it would convert Finding 4 from an
open task into a known, quantified limitation of the model, which is worth having written
down either way.

---

# STEP 0 + STEP 1 RECORD — 2026-08-17. GO, AND BY TWO ORDERS OF MAGNITUDE.

**Step 1's answer is the opposite of what this plan predicted.** The plan said to treat a
"no" as the expected outcome, on the strength of one number: the bottom flank's own
tangent lands 7.21 deg from the part's hub corner where the end cap lands 1.53. **That
number was right and the conclusion drawn from it was wrong**, because it tested the
wrong extension direction.

## STEP 0 — the instrument

`studies/study_junction_agreement.py` (`make junction`, ~4 s, geometry only, no field
solved). It reproduces `_embed` in numpy, walks the part's outline for every ring
crossing, and tabulates them beside the mesh's corners with wedge angles and Williams
lambda.

**Validated two ways.** The crossing count comes out **24 hub and 24 rim per wheel**,
matching the shipped manifest's `hub_edges`/`rim_edges`; and the mesh's four corners
reproduce §30's independently measured wedges — summed from incident element angles on
the `fine` mesh — to within 0.8 deg:

```
  corner      this driver    §30 (make corner)   delta
  hub:P_t        321.14           321.10          0.04
  hub:P_c        297.18           296.75          0.43
  rim:P_t        320.55           321.33          0.78
  rim:P_c        307.43           307.94          0.51
```

*Two numbers in the earlier record are corrected by this.* PART 2 gave the part's second
corners as 271.53 and 320.10 deg. Those came from a probe that took the ACUTE branch of
the void angle by assumption rather than choosing the free-arc branch by construction.
Measured properly they are **268.47 and 219.90**. The hub number barely moves; **the rim
number moves 100 deg**, and it moves in the direction that matters.

## STEP 1 — the answer

`_embed`'s non-differentiability is entirely in its ARGMAX: a 20001-point length scan and,
at the rim, a 21-point blend scan. **Neither is needed.** The length is unnecessary — the
ring crossing is a closed-form line/circle intersection — and each of `_embed`'s two
extreme directions is a smooth function of the genes. So the candidates are:

```
  own_tangent      the bottom flank's own end tangent
  shared_tangent   the mean of the two end tangents   = `_embed` at blend 0.0
  radial           the flank midpoint's radial        = `_embed` at blend 1.0
```

**And `_embed` tells you which to expect**: its hub branch is hard-coded to `(1.0,)` —
*"Radial-inward always reaches, so the search below is a single step at the hub"* — while
its rim branch searches upward from 0.0. So radial should match at the hub and a tangent
at the rim. **It does, essentially exactly:**

```
  ring   candidate          wedge err (deg)   theta err (deg)   lambda err
  hub    end_cap (today)         28.71            1.5267         0.0329
  hub    own_tangent             70.45            7.2101         0.2866
  hub    shared_tangent          67.29            6.3115         0.2631
  hub    radial                   0.00            0.0000         0.0000   <- exact
  rim    end_cap (today)         87.53            1.3159         0.1896
  rim    own_tangent              1.49            0.0658         0.0075
  rim    shared_tangent           0.02            0.0201         0.0001   <- best
  rim    radial                  50.61            0.8010         0.1540
```

**Not a one-genome coincidence.** Across eight genomes spanning the whole design history
— the shipped wheel, the GA/beam ancestor, both ends of the min-wall sweep, the knee, the
SVK run, a buildcap run and `defect5_step100`:

```
  genome                                  hub best   err     rim best         err    cap hub  cap rim
  best_solution.json                      radial    0.00    shared_tangent   0.02     28.71    87.53
  best_solution_ga_beam.json              radial    0.00    own_tangent      0.97      4.81    84.00
  stage3_minwall_best_0.8.json            radial    0.00    shared_tangent   0.01     21.67    88.22
  stage3_minwall_best_2.2.json            radial    0.00    shared_tangent   0.13      6.85    87.45
  stage3_knee_best_medium.json            radial    0.00    shared_tangent   0.02     28.71    87.53
  defect5_step100.json                    radial    0.00    shared_tangent   0.05     22.29    86.76
  stage3_svk_best_medium.json             radial    0.00    shared_tangent   0.00     50.90    88.37
  stage3_buildcap2_feasible_medium.json   radial    0.00    shared_tangent   0.00     43.52    88.34
```

**`radial` is exact at the hub on every genome tested.** At the rim a tangent wins on all
eight, within 1 deg.

## WHY THIS MATTERS MORE THAN THE ARC EXPECTED

Look at the `cap rim` column: **the end cap is wrong by 84-88 deg of wedge at the rim on
every genome.** That is not a shipped-genome quirk, it is a systematic property of
capping the spoke at the ring circle. In Williams terms the mesh models lambda = 0.5081
where the part has 0.6977 — **stress ~ r^-0.492 against the part's r^-0.302.**

**That is the mechanism behind PLAN §34 Finding 4.** The wheel's global peak von Mises
sits 11-16 um from `rim:P_c` not because the rim junction is the most loaded place on the
wheel, but because the mesh puts a far sharper corner there than the part has. The
artefact is not a small distortion; it is nearly 90 degrees of wedge.

## THE RESIDUAL RISK, NAMED

**The rim's best direction is genome-dependent** — `shared_tangent` on seven of eight,
`own_tangent` on `best_solution_ga_beam`. That is `_embed`'s blend search reappearing, and
a search is exactly what must not enter the coordinate map. **The mitigation is that it
does not need to:** shared and own tangent differ by under 1 deg of wedge, against the end
cap's 87, so **fixing** the rim direction to `shared_tangent` unconditionally is smooth,
differentiable, search-free, and still two orders of magnitude better than today. Step 2
should fix it and record the residual, not reproduce the search.

Second, smaller: this driver only needs the extension to reach the RING circle (48.50),
not `_embed`'s target (50.25). The failure mode that motivates `_embed`'s search is
therefore much less likely here, but Step 2 must still check the extension reaches on
every genome it is asked about rather than assume it.

## STEP 2 IS GO

Fix blend 1.0 (radial) at the hub and blend 0.0 (shared tangent) at the rim, take the ring
crossing in closed form, and the mesh's second corner reproduces the part's to under 1 deg
of wedge everywhere tested — against 5-90 deg today. **The cost section of this plan still
applies in full**: this moves welded volume, mass, hub share and every constant calibrated
against them, and it must be priced one number at a time with no threshold touched.

---

# STEP 2 RECORD — 2026-08-18. THE HUB IS FREE. THE RIM NEEDS A TOPOLOGY CHANGE, AND THE REASON IS EXACT.

**Built, opt-in.** `sector_blocks`/`build_wheel` take `uncap=` — `False` (default, today's
geometry), `True` (both rings at `_embed`'s own blend), or a `(hub, rim)` pair whose
entries may be `False`, `True`, or an explicit blend in [0, 1]. **The blend is `_embed`'s
own parameter and means the same thing**: 0.0 the shared end tangent, 1.0 the ring radial.

The change to the junction is two lines. Its outer corner `Q` is `P_c` when capped and the
far flank's ring crossing when not; the right edge is the cap or the flank's continuation.
**Every other side, every node count and every seam entry is untouched** — the same
property §34 Finding 2 established for a moving corner.

**Control:** `uncap=False` is **bit-identical** at `smoke`/`coarse`/`medium` (`max|dx| =
0`). The uncapped mesh assembles with the same node and element counts and a seam error of
3.15e-14 mm.

## THE FIDELITY IT BUYS

```
  ring   corner error vs the part      wedge        theta       lambda
  hub    end cap (today)              28.71 deg    1.5267 deg   0.0329
  hub    AS BUILT (uncap)              0.01 deg    0.0080 deg   0.0000
  rim    end cap (today)              87.53 deg    1.3159 deg   0.1896
  rim    AS BUILT (uncap, blend 0)     1.06 deg    0.0526 deg   0.0053
```

The rim's 1.06 deg is slightly worse than the 0.02 deg `shared_tangent` scored in Step 1,
because the mesh takes the ANALYTIC centreline tangent where the exporter takes a chord
of its 48-point downsample. **That is a deliberate choice** — the analytic tangent is
smooth and carries no dependence on the exporter's discretisation — and 1.06 against the
cap's 87.53 is not a trade worth undoing.

## THE PHYSICS, PRICED ONE RING AT A TIME (`coarse`, no threshold touched)

```
  uncap            axle drop   comp hub  comp rim   area mm2   vs STEP   max vM   peak at
  False (today)     1.620790   0.041656  0.316117  1420.6161   -2.236%   99.130   rim:P_c
  (True, False)     1.594888   0.034111  0.315402  1421.9600   -2.154%   98.418   rim:P_c
  (False, True)     1.542107   0.041466  0.308861  1424.5844   -1.995%   74.833   hub:P_c
  (True, True)      1.517569   0.033985  0.308347  1425.9283   -1.913%   49.246   hub:P_t
```

**The peak CASCADES, and that is the cleanest confirmation §34 Finding 4 could have
asked for.** Fix the rim artefact and the maximum jumps to the *hub* artefact; fix both
and it lands on `hub:P_t` — a corner the part has, matched to 0.073 um. **The model was
overstating its own peak by 2x (99.13 -> 49.25 MPa), entirely on corners the part does not
have.**

Also: the area deficit against the shipped STEP narrows from -2.236% to -1.913%, because
the slivers between the old cap and the new corner are part of `_embed`'s gusset. **It
does not close** — the material inside the ring circle is still not modelled, and this arc
never claimed to add it.

**Flagged for `HUBSHARE_PLAN.md`, not acted on:** hub compliance falls 0.041656 ->
0.033985 (-18.4%). `test_the_hub_junction_holds_under_three_percent_of_the_compliance`
gates on `< 0.03` and is a strict xfail. **It still fails** — 0.0340 is not 0.03 — but the
deficit §31 accepted is more than half closed by a mesh fix rather than a design change.
`HUBSHARE_PLAN` must be re-read before anyone works it.

## WHERE IT BREAKS, AND WHY IT IS NOT A BLEND THAT CAN FIX IT

`min_scaled_jacobian` collapses **0.782505 -> 0.007208**. Localised, it is entirely one
block:

```
  block              capped     uncapped
  hub_junction      0.796181   0.796983     <- unchanged; slightly BETTER
  rim_junction      0.782505   0.007208     <- the whole of it
  every other       >0.999     >0.999
```

**The hub is free** — `min_sj` unchanged at 0.782505, and `max_aspect_ratio` *improves*
20.2715 -> 16.2400.

The rim's cause is exact and geometric. Measured, the angle between the far flank's own
tangent at the ring and each candidate direction:

```
                              hub          rim
  centreline tangent (0.0)   0.587 deg    0.489 deg    <- SMOOTH: no corner exists here
  radial (1.0)             116.669 deg   53.017 deg    <- a real corner
```

**A tangent continuation is by definition smooth**, so at blend 0 the junction stops being
a quadrilateral and becomes a curvilinear TRIANGLE — three sides: the ring arc, the
cross-section, and one smooth [flank + extension] curve. A four-sided structured block on
a three-sided region always carries a ~180 deg vertex; measured, 179.35 deg, and
sin(179.35 deg) = 0.011 is the 0.0072 scaled Jacobian. **No interior blend, node
distribution or smoothing changes the angle between two boundary curves.**

`_embed` uses blend 1.0 at the hub, which is *also* the well-shaped choice — that is the
whole reason the hub is free and the rim is not.

## THE BLEND SWEEP, AND WHY NO VALUE OF IT SHIPS

`MIN_SJ_TARGET = 0.2` (`wheel_objective`, barrier weight 3000; `tests/test_mesh.py` and
`tests/test_wheel.py` both assert `> 0.2`).

```
  rim blend   wedge err   min_sj    clears 0.2   max vM   peak at
   (capped)      87.53   0.782505      YES       99.130   rim:P_c
       0.00       1.06   0.007208       no       49.246   hub:P_t
       0.10       3.53   0.089500       no       49.326   hub:P_t
       0.15       5.96   0.132625       no       49.368   hub:P_t
       0.20       8.47   0.176818       no       49.753   rim:P_c
       0.22       9.49   0.194744       no       50.371   rim:P_c
       0.25      11.05   0.221845      YES       51.309   rim:P_c
       0.30      13.69   0.267429      YES       52.895   rim:P_c
       1.00      50.61   0.782226      YES       73.728   rim:P_c
```

**The two conditions are disjoint.** The peak leaves the artefact only at blend <= 0.15;
the gate is cleared only at blend >= ~0.23. There is no overlap.

**And the gate-clearing end is worse than it looks.** At blend 0.25 `min_sj` is 0.2218
against a floor of 0.2000 — the margin falls from 0.58 to 0.02, so `min_sj` would become
an ACTIVE constraint against a 3000-weight barrier for every genome the optimiser tries.
That changes what the search does, which is a far larger consequence than the corner it
was meant to fix. **Do not ship a blend compromise.**

## STEP 2's ANSWER

**Hub: DONE and free.** Corner error 28.71 -> 0.01 deg, `min_sj` unchanged, aspect ratio
better, hub compliance -18.4%. The only thing standing between it and being the default is
that the default flip is a separate decision with its own baseline refresh.

**Rim: BLOCKED on a topology change, and the blocker is proven rather than suspected.**
The junction must stop being one four-sided block. The fix is a three-quad decomposition
of the triangular region — which buys blend-0 fidelity (1.06 deg) *and* a well-shaped
block, because it removes the 180 deg vertex instead of trading against it. That is new
seam entries and new counts: the first thing in this arc that genuinely touches
`_seam_table`.

**`make test` after Step 2 reads 476 passed, 2 xfailed, 0 failed, exit 0 (28 m 39 s)** — the same counts, the same two deliberately-held xfails and the same runtime to within 3 s as the §33 and §34 baselines (476/2/0, 28 m 32 s and 28 m 36 s). `build_wheel` is what most of the suite runs through, so this is the check that `uncap=` is genuinely inert at its default rather than merely inert on the three configs probed directly.

**Not done, deliberately:** the default is still `uncap=False`, nothing is promoted,
`best_solution.json` is untouched, and no threshold was moved.

---

# STEP 3 RECORD — 2026-08-18. THE TRI-BLOCK DOES NOT BIND. STOPPED BEFORE BUILDING IT.

§36 ranked the rim three-quad block as successor #1. **Before building it I did the two
checks that its value rests on, and both came back against it.** Neither needed the
tri-block to exist.

## THE PARTITION ALGEBRA, DERIVED FIRST BECAUSE IT SETS THE PRICE

A triangle admits no partition into two quadrilaterals — a diagonal gives two triangles,
and joining midpoints of opposite sides leaves the offending vertex in one piece. The
minimum is the three-quad Y-partition, and it **splits all three sides**. Two of the
three are shared: the ring arc (with `rim_band_weld`) and the end cross-section (with
`spoke`). **So it needs PARTIAL-EDGE SEAMS**, and whole-edge single ownership is what
`wheel_wheel.py`'s docstring calls "the whole safety net for this module".

Worse, the element counts are not free. Writing the three sides as A (arc, `n_weld`), B
(free), C (cross-section, `n_thick`), split as a1+a2, b1+b2, c1+c2, matching opposite
sides of the three quads forces

```
    a1 = b2      a2 = c1      c2 = b1
```

so an EVEN split would require `n_weld == n_thick`, which no config satisfies (10 vs 4 at
`coarse`, 16 vs 6 at `medium`). Uneven splits do solve — at `coarse` a1=7, a2=3, c1=3,
c2=1, b1=1, b2=7 — but they produce blocks of **7x1, 3x1 and 7x3 elements**, and the
`c2 = 1` strip is forced, not chosen.

## CHECK 1 — DOES UNCAPPING STOP THE PEAK DIVERGING?  NO.

M4's problem was never the peak's value at one config; it is that the peak has no limit,
which is why M8b-i.6 rebuilt the stress constraint around a p-norm x `Kt`. Growth over
smoke -> coarse -> medium:

```
                              smoke   coarse   medium   growth
  CAPPED (today)   global    61.922   99.130  126.248    2.04x
  uncap rim 1.0    global    48.208   73.728   91.152    1.89x
  uncap rim 0.0    global    34.217   49.246   65.647    1.92x
```

**Essentially unchanged.** Uncapping moves the peak and shrinks it; it does not converge
it — and it never could, because every corner stays re-entrant (`P_t` at 321-322 deg,
lambda ~0.503, and its per-corner rate is 3.91x capped against 3.99x uncapped, i.e. flat).
**Only a fillet removes a singularity.** The peak's MAGNITUDE is not a quantity this
project can quote, so buying a further 28% of it is buying something unusable.

## CHECK 2 — DOES UNCAPPING UNBLOCK THE FILLET?  NO.

§36's successor 3 warned that §34's 0.2 mm fillet ceiling was measured assuming an end
cap. Re-measured on uncapped meshes:

```
  uncap                hub R_max   rim R_max
  False                    0.200       0.200
  (True, 1.0)              0.200       0.200
  (True, 0.0)              0.200       0.200
  True                     0.200       0.200
```

**Identical.** Obvious once measured: §34's fold is in the SPOKE block — its ruled
interior and its end cross-section running from the moved fillet corner to an unmoved
far-flank point — and `uncap` changes the JUNCTION block. They do not interact. The
caveat is discharged and §34's ceiling stands unqualified.

## SO WHAT WOULD THE TRI-BLOCK ACTUALLY BUY?

Only **rim corner fidelity**: wedge error 50.61 deg (blend 1.0, free) -> 1.06 deg
(blend 0.0, tri-block). Not convergence, not the fillet, and not a quotable peak.

Against: partial-edge seams breaking the module's core invariant, three blocks for one,
and forced 1-element strips.

**It does not bind. Filed, not built.**

## WHAT TO TAKE INSTEAD

```
  option                     min_sj    rim wedge err   peak growth   machinery
  capped (today)            0.782505      87.53 deg       2.04x       -
  uncap (True, 1.0)         0.782226      50.61 deg       1.89x       none
  uncap (True, 0.0)         0.007208       1.06 deg       1.92x       tri-block
```

**`uncap=(True, 1.0)` is free and strictly better than today on every measured axis** —
`min_sj` 0.782226 against 0.782505 (0.03% apart, well inside the 0.2 floor's 0.58 of
margin), hub corner EXACT at 0.01 deg, rim corner error more than halved, and
`max_aspect_ratio` 20.27 -> 16.24. It needs no new machinery and no seam change.

**The rim is a spectrum with no faithful free point, and that is the honest summary of
this arc**: the hub's idealisation was both wrong and cheap to fix; the rim's is wrong and
expensive, and the thing it is wrong about turns out not to be load-bearing.

---

# STEP 4 RECORD — 2026-08-18. THE DEFAULT IS FLIPPED, AND "FREE" WAS THE WRONG WORD.

`UNCAP_DEFAULT = (True, 1.0)` is now the default of `sector_blocks`, `_sector_coords`
and `build_wheel`. `uncap=False` still reproduces the pre-flip geometry bit-for-bit.
Nothing in `best_solution.json` moved; this is a model change, not a promotion.

**READ THIS BEFORE QUOTING STEP 2 OR PLAN 37.** Both of them called the hub uncap
"free", and I repeated it when I ranked this work. That claim was true on the axes it was
measured on — `min_scaled_jacobian`, `max_aspect_ratio`, corner error, seam error, node
and element counts — and those are all mesh-quality axes. **It was never measured on the
objective, on the adjoint, or on the area bookkeeping, and it does not hold on any of the
three.** Flipping the default cost six red tests and two source changes beyond the flip
itself. The flip is still right; "free" was a claim about a subset presented as a claim
about the whole, and that is the error worth carrying forward from this step.

## WHAT `make test` SAID, BEFORE ANY REPAIR

`476 passed, 2 xfailed` -> **`6 failed, 470 passed, 2 xfailed` (34 m 16 s, exit 2).**
Runtime is not comparable to the 28 m 3x baselines: diagnostics were running alongside it.

```
  1  test_corner_singularity :: the wedge is measured on the mesh that has the corner
  2  test_gradient          :: the axle drop gradient is not the secant's derivative
  3  test_objective         :: but above the knee the fillet radii are live
  4  test_wheel             :: area converges under refinement
  5  test_wheel             :: the embed difference from the shipped step is the known amount
  6  test_wheel_fea         :: total mass matches the step manifest within the embed difference
```

Four are now green, and none of the four was fixed by moving a threshold. One is a
genuine open regression. One is a genuine finding that invalidates a fixture.

## THE AREA REFERENCE HAD TO FOLLOW THE MESH, AND IT WAS RIGHT TO SHOUT

Three of the six reds (4, 5, 6) are one cause. `modelled_area_reference` describes a
REGION — `hub disk | rim band | 12 bands clipped to the annulus` — and the band is only
defined over the centreline's own parameter range, so its ends ARE the straight
cross-sections that `uncap` removes. The mesh grew by 2.8187 mm² at `medium` and the
reference did not, which turned `error_vs_modelled` from a discretisation residual into a
region mismatch:

```
  err_vs_modelled        smoke      coarse      medium     converging?
  capped               -0.18034%   -0.02431%   -0.00851%       yes
  flipped, ref stale   +0.00345%   +0.15194%   +0.16686%       NO  <- grows with refinement
  flipped, ref fixed   -0.17107%   -0.02284%   -0.00795%       yes
```

**The stale-reference row is the tell, and `test_area_converges_under_refinement` is the
test that caught it.** A region mismatch does not shrink when you refine the mesh — it is
resolved BETTER — so the error grows. Its `smoke` entry, +0.0034%, is the near-perfect
cancellation that test's docstring was written to catch ("a single config agreeing to
0.5% could be two errors cancelling"); it read three times better than the true answer
while being three times more wrong. Six years of that convention paid for itself here.

**The fix, in `_uncap_reference_poly`:** continue each end's far flank to its ring circle
on the reference's OWN polygon, and let `_clip_polygon_to_disk` supply the arc. Only the
far flank moves — the straddling one is already cut at the ring circle, which is the
reference's own `P_t` — so this is the same "only one corner moves" property the mesh
side has (34 Finding 2). Residual mesh-vs-reference after the fix: **0.0088 mm²**, of the
same order as the capped construction difference.

**WHAT INDEPENDENCE THIS COSTS, SAID PLAINLY.** The DIRECTION rule is now shared with
`_uncap_corner`, and it has to be, or the two describe different regions again. What
stays independent is what made this a cross-check rather than the same computation twice:
the flank endpoints come from `thicken_3taper_curve`'s FINITE-DIFFERENCE offset normals
against the mesh's analytic hodograph, and the integration is exact shoelace plus exact
circular sectors against Q9 Gauss. `_uncap_blend` was factored out of `sector_blocks` for
the same reason — two copies of a three-line branch is how a mesh and its own area
reference end up describing different regions.

## AND THE STEP REFERENCE HAD TO **NOT** FOLLOW IT

The opposite call, one function later, and getting it wrong the other way would have been
worse than the bug. `reference_shipped_step_mm2` describes the SHIPPED SOLID, which is a
fixed physical thing. Letting it drift with our meshing choice would have hidden exactly
the change this pair of numbers exists to expose — `error_vs_shipped_step` would have
read -2.2205% -> -2.2162%, i.e. "nothing happened", when what actually happened is that
the mesh started modelling a tenth of `_embed`'s gusset.

So the STEP reference stays anchored on the CAPPED region plus the full measured
allowance, and the newly-modelled part is reported separately as
`gusset_modelled_per_spoke_mm2` instead of being netted off:

```
                      err_vs_modelled   err_vs_shipped   gusset modelled/spoke
  capped                  -0.00851%        -2.22050%           0.0000 mm2
  DEFAULT (True, 1.0)     -0.00795%        -2.04901%           0.2342 mm2
```

`EMBED_ALLOWANCE_PER_SPOKE_MM2` is untouched at 3.03: it still means what it meant, and
the mesh's share of it is now derived rather than folded into the constant. Both
downstream pins (tests 5 and 6) were repaired by SUBTRACTING or ADDING BACK that derived
term, with every threshold left exactly where it was — 0.5 mm² of slack in test 5, the
`0.5 < per_spoke < 2.0` band in test 6, which reassembles to 0.567 mm²/spoke from the
0.333 left over plus the 0.234 the mesh now carries.

## TEST 1 WAS FRAGILE, NOT WRONG

`test_the_wedge_is_measured_on_the_mesh_that_has_the_corner` picked its "interior node"
with `argsort(abs(r - 49.4))[0]`. **A whole ring of nodes sits at exactly r = 49.500000**,
so that expression was decided by `argsort`'s tie-break, and the flip reordered the seam
merge's node ownership enough to move the winner from a Q9 vertex to a midside. Midsides
are skipped by `measured_wedge_deg` BY DESIGN, so it returned 0.0 against an expected 360.
Nothing about the mesh, its quality or the wedge changed. Fixed by restricting the
candidates to `conn[:, :4]`, which is what the docstring always said, plus an assertion
that the node has four incident elements so it cannot silently degrade to luck again.

## TEST 2 — THE ADJOINT WAS NEVER WRONG, AND IT TOOK THREE MEASUREMENTS TO SAY SO

G9 (`the axle drop gradient is not the secant's derivative`) went **10x over its 1e-5
gate**, at 1.072e-04. That is the one red that could have blocked the flip: a degraded
gradient on the quantity Stage 3 descends is not a bookkeeping problem.

**Measurement 1 — localise it by ring.** Each ring alone is clean; only both together is
not, which already rules out a per-ring geometric defect:

```
  setting                axle drop    worst_rel    gate 1e-5
  capped                  1.477177    6.985e-07      pass
  hub only  (True,False)  1.457596    4.574e-07      pass
  rim only  (False,1.0)   1.442084    5.689e-07      pass
  BOTH      (True,1.0)    1.422966    1.072e-04      FAIL
```

**Measurement 2 — split the quotient.** `d(delta*)/dp = -(dF/dp)/(dF/d delta)`, and
`d_dindentation` is the half no gate covers. Finite-difference each half separately:

```
  setting     dF/d delta adj    FD        rel      dF/dp adj      FD        rel
  capped        45.77602    45.77604   4.91e-07    1.40782851  1.40782325  3.73e-06
  BOTH          47.54066    47.54079   2.79e-06    0.99463181  0.99462978  2.03e-06
```

**Both halves are already as accurate as capped.** So the adjoint reproduces the whole
quotient to ~4e-7, and yet the whole-solve difference disagreed with it by 1.07e-04 —
a hundred times either half's own error. The error was not in the thing under test.

**Measurement 3 — the reference.** It is the SECANT's stopping rule, in the FD reference:

```
  fd tol_rel      1e-8            1e-9            1e-10       rel vs adjoint
  capped       -0.0307547723   -0.0307547723   -0.0307547723    1.85e-06
  hub only     -0.0303006155   -0.0303006155   -0.0303006155    3.29e-06
  rim only     -0.0212447596   -0.0212447596   -0.0212447596    4.75e-06
  BOTH         -0.0209239528   -0.0209216681   -0.0209216681    1.07e-04 -> 1.99e-06
```

One decade on the reference and G9 goes from 10x over to 5x under. `GATE_SECANT_REL`
is untouched at 1e-5; what moved is `run_axle_drop`'s own `fd_tol_rel`, now a named
parameter at 1e-9 — two decades clear of where the secant STALLS at 1e-11 on the float64
floor of the force, one decade past where the reference stops depending on it.

**AND A DOCSTRING IN THIS REPO SAID THE OPPOSITE, ON EVIDENCE.** `run_axle_drop` read:
"THE SECANT'S TOLERANCE IS NOT WHAT LIMITS THIS, WHICH WAS WORTH MEASURING RATHER THAN
ASSUMING... Tightening it to 1e-11 moves the difference by nothing at all — ten identical
digits." That measurement was real and it was taken on the mesh of the day. **It does not
generalise, and the mesh it does not generalise to is one decade away.** The FD reference
differences two SEPARATELY TERMINATED secants and their termination bias does not cancel;
whether that bias is visible depends on where each secant happens to stop, which depends
on the mesh. The docstring is corrected in place with the table above, because it
documents a parameter's value and a wrong reason there would be read as a licence.

**This is the same shape as PLAN 35's `_embed` correction and it is now a pattern worth
naming: a measured justification is only as general as the configuration it was measured
on, and nothing in a docstring records which configuration that was.** Both times the
conclusion looked safe, both times one config change flipped it, and both times the cost
of re-measuring was minutes.

## TEST 3 — THE ONE THAT IS A FINDING, NOT A REPAIR

`test_but_above_the_knee_the_fillet_radii_are_live` asks whether `R_hub` carries gradient
on a design ABOVE `MARGIN_KNEE_UTIL` (0.80). Its fixture, `stage3_buildcap2_slack_medium`,
read **hub utilisation 0.85506 on the capped mesh and 0.77297 on the uncapped one** — the
same genome, at the same `coarse` / 8-uniform-phase settings, now BELOW the knee. The test
does not fail because the coupling broke; it fails because its fixture stopped satisfying
its own premise.

**Why the utilisation moved, and why the direction is right.** The stress constraint is
built on a p = 4 Gauss p-norm times `Kt` post-multipliers (M8b-i.6). Uncapping deletes two
of the four re-entrant corners per junction — the two the SHIPPED PART DOES NOT HAVE — so
the p-norm falls. Section 36 measured the same effect on the raw maximum, where it is
larger: 99.13 -> 49.25 MPa, a factor of two. At p = 4 it is about 10%.

**THE CONSEQUENCE IS NOT A TEST PROBLEM AND IT IS THE MOST IMPORTANT THING IN THIS STEP.**
The shipped genome records `stress_utilisation` = 0.8201, which is ABOVE the 0.80 knee, so
`stress_margin` was a LIVE term with a loss of 0.1317 when the wheel was descended. Under
the uncapped mesh the same design falls below the knee and that term goes to zero. In
other words: **part of what shaped the shipped design was a stress term reading corners
that only exist in the mesh.** That is exactly the failure 36 predicted when it found the
peak sitting 11-16 um from `rim:P_c`, now visible in the objective rather than in a
diagnostic.

This does NOT make the flip wrong — the new number is the more faithful one. It makes the
flip **a re-pricing of the objective**, which is a much larger claim than "free", and it
is why nothing here promotes anything: the shipped genome is now a design that was
optimised against a stress term it would no longer feel.

## THE BASELINE §37 ASKED FOR — MASS, HUB SHARE, AXLE DROP

Shipped genome, one solve per cell, `uncap=False` against the new default.

```
  quantity                      coarse                          medium
                        capped    DEFAULT      pct      capped    DEFAULT      pct
  axle_drop_mm         1.620790   1.551645   -4.266%   1.639893   1.562981   -4.690%
  compliance hub       0.041656   0.034188  -17.929%   0.043349   0.035237  -18.714%
  compliance rim       0.316117   0.311276   -1.531%   0.320262   0.315592   -1.458%
  compliance spoke     0.642226   0.654536   +1.917%   0.636389   0.649172   +2.009%
  mesh_mass_g           39.4590    39.5377   +0.199%    39.4661    39.5444   +0.198%
  meshed_mm2          1420.6161  1423.4489   +0.199%  1420.8700  1423.6887   +0.198%
  err_vs_shipped_STEP  -2.2359%   -2.0636%      --     -2.2205%   -2.0490%      --
  min_scaled_jacobian  0.782505   0.782226   -0.036%   0.782782   0.782608   -0.022%
  max_aspect_ratio      20.2715    16.2400  -19.887%    24.3255    19.4878  -19.887%
  global max vM (MPa)   99.1302    73.7282  -25.625%   126.2484    91.1521  -27.799%

  n_nodes / n_elements  identical      n_inverted 0 both      seam ~3.1e-14 both
  peak sits at          rim:P_c in every cell — rim blend 1.0 keeps that artefact
```

**The wheel is stiffer by 4.3-4.7% and the hub junction is 18% less compliant**, both at
both rungs, and both are the sliver of gusset the junction now carries. Mass rises 0.2%,
which is the same sliver: 0.0787 g on 39.46, and the STEP area gap narrows a matching
0.17 points. `max_aspect_ratio` improves 19.9% at BOTH rungs — the same figure twice,
because the improvement is the junction block's shape and not a resolution effect.

**The peak is still at `rim:P_c` and that is by construction.** Rim blend 1.0 buys mesh
validity and leaves the rim corner where it was (50.61 deg of wedge error), so the global
maximum stays on the rim artefact — it just gets 26-28% smaller. §36's cascade to
`hub:P_t` needs rim blend 0, which is the setting that collapses `min_sj` to 0.0072.
Nothing here changes that trade; UNCAP_PLAN Step 3 priced the only way out and measured it
not to bind.

**Hub compliance -18.4% is flagged for `HUBSHARE_PLAN.md` and NOT acted on here.**
`test_the_hub_junction_holds_under_three_percent_of_the_compliance` gates on < 0.03, is a
strict xfail, and STILL FAILS at 0.0342 — but more than half the deficit §31 accepted and
decided on turns out to be a mesh artefact rather than a property of the design. §31's
decision was made on the capped number. **`HUBSHARE_PLAN.md` must be re-read before anyone
works it, and §31's conclusion re-derived rather than quoted.**

## THE KNEE, MEASURED ACROSS THE ARCHIVE — AND `stress_margin` IS FLAT AT EVERY DESIGN ON DISK

Rather than assume the fixture was unlucky, the question was asked of the whole archive's
most stressed designs. All at `coarse` / 8 uniform phases, the fixture's own settings:

```
  genome                              util hub   util rim   knee 0.80   stress_margin
  stage3_minwall_best_0.8.json         0.78664    0.55144      no          --
  stage3_buildcap2_slack_medium.json   0.77297    0.47862      no        0.000000
  best_solution.json  (SHIPPED)        0.71316    0.47421      no        0.000000
```

`stage3_minwall_best_0.8` is the THINNEST WALL in the tree and therefore its most stressed
design. It is under the knee. **The shipped genome's `stress_margin` measures exactly
0.000000 under the new default** — against the 0.13168 recorded in `best_solution.json`
when it was descended on the capped mesh.

**So the term is flat at every design currently on disk.** That is not a broken term — a
knee'd penalty is SUPPOSED to be flat below its knee, and §23 built it that way
deliberately. It is a statement about where the archive sits once the stress field stops
counting corners the part does not have: **the whole archive is below the knee, and the
shipped design was pushed there by a term that, on the faithful mesh, would never have
pushed.**

Caution on one comparison: `best_solution.json` records `stress_utilisation` 0.8201 at
`medium` with 8 phases, and 0.71316 above is `coarse`. Those are different rungs and
should not be differenced. The clean paired measurement is the fixture's, same genome and
same settings on both sides: **0.85506 -> 0.77297, -9.6%.** The directly measured fact
about the shipped wheel is the `stress_margin` = 0.000000, which needs no pairing.

## HOW TEST 3 WAS LEFT

**Split, following §31's precedent in this same repo.** While the two halves shared one
function, the RIM half — which passes, and which is a claim about a different junction —
was not being checked on any run. So:

  - `test_below_the_knee_the_rim_fillet_radius_is_dead` — LIVE. `u_rim` under the knee and
    `dL/dR_rim == 0.0`. Holds before and after the flip; the rim is at 0.47-0.55 on every
    genome measured.
  - `test_but_above_the_knee_the_fillet_radii_are_live` — STRICT XFAIL, with the table
    above as its reason. `MARGIN_KNEE_UTIL` stays at 0.80. `xfail_strict` reopens it the
    day a design goes above the knee on the faithful mesh.

**Nothing measured says `R_hub` stopped carrying gradient above the knee.** What broke is
the premise, not the claim, and the xfail says so in those words so nobody later reads it
as evidence the coupling regressed.

## WHAT ACTUALLY CHANGED IN THE TREE

```
  src/wheel_wheel.py
    UNCAP_DEFAULT = (True, 1.0)          new module constant, with the hub/rim asymmetry
                                         written out -- the two entries do NOT mean the
                                         same thing and the comment says why
    sector_blocks / _sector_coords /     uncap=False -> uncap=UNCAP_DEFAULT
      build_wheel
    _uncap_blend(uncap, is_hub)          factored out of the junction loop, because
                                         `modelled_area_reference` must resolve `uncap`
                                         EXACTLY as `sector_blocks` does
    _uncap_reference_poly(...)           the reference-side continuation
    modelled_area_reference(..., uncap=)  follows the mesh
    WheelMesh.uncap                      carried, so `area_report` can ask for the region
                                         this mesh actually builds
    area_report                          `reference_capped_mm2`, `gusset_modelled_mm2`,
                                         `gusset_modelled_per_spoke_mm2`; the STEP
                                         reference re-anchored on the capped region
    module docstring                     `_embed` is PARTIALLY reproduced now; the
                                         "a smooth alternative does not exist" claim
                                         corrected; the stale ~1.4%/~2.3% area and mass
                                         percentages re-measured (they were stale by more
                                         than the effect they described)

  studies/study_gradient.py
    run_axle_drop(fd_tol_rel=1e-9, ...)  the FD reference's own tolerance, named; the
                                         docstring's "the secant does not matter" claim
                                         corrected with the table that disproves it

  studies/study_junction_agreement.py    reports THREE meshes now -- capped, the shipped
                                         default, and uncap=True -- because the default is
                                         neither of the two it was written to compare, and
                                         labelling one of them "today" would be false

  tests/test_corner_singularity.py       interior node restricted to Q9 vertices
  tests/test_wheel.py                    embed pin subtracts the derived modelled gusset
  tests/test_wheel_fea.py                mass budget adds the derived modelled gusset back
```

**No threshold was moved.** `MIN_SJ_TARGET` 0.2, `GATE_SECANT_REL` 1e-5,
`MARGIN_KNEE_UTIL` 0.80, `EMBED_ALLOWANCE_PER_SPOKE_MM2` 3.03, the 0.5 mm² area slack and
the 0.5-2.0 gusset band are all exactly where they were. The two test repairs that touch
numbers do so by adding or subtracting a term `area_report` DERIVES, so they cannot
re-stale the way a transcribed constant would, and both reduce to the original assertion
when `uncap=False`.

## THE SUITE, AND A FOURTH WRONG CAUSE — CAUGHT BY MEASURING IT

```
  run                                     result                       wall
  pre-flip baselines (33, 34, 36)   476 passed,  2 xfailed, exit 0   28m32 / 28m36 / 28m39
  after the flip, before repair       6 FAILED, 470 passed           34m16  (contaminated)
  after repair                      476 passed,  3 xfailed, exit 0   31m46
  after the area-reference cache    476 passed,  3 xfailed, exit 0   31m36
```

479 tests where there were 478, because `test_but_above_the_knee_the_fillet_radii_are_live`
was split in two; one of the two is the new strict xfail, so `passed` holds at 476 and
`xfailed` goes 2 -> 3. Exit 0 twice, ten seconds apart, which is itself the evidence that
the 34m16 was contention and not the tree.

**+3m07s against the pre-flip baseline, and I got its cause wrong.** `area_report` genuinely
doubled — 51 -> 101 ms — because with `uncap` active it needs `modelled_area_reference`
twice, and the suite asks for the same genome repeatedly. That was a real regression and a
plausible three minutes, so I cached the reference. Warm `area_report` went 101 ms ->
**0.27 ms, a factor of 190** — and the suite moved **10 seconds**. `area_report` was never
on the critical path.

**That is the fourth wrong cause in this arc and the third today**, and the only one that
cost anything was the one I acted on before measuring. §34 mis-ranked the seam merge, §36
mis-ranked the tri-block, §37/§38's own G9 story ran through two wrong suspects before the
FD reference, and here I shipped a cache on a hypothesis the very next run refuted. The
cache is KEPT, because a 2x on a pure function is real for sweeps and drivers that call it
in a loop — but it is documented as buying throughput, not suite time, and the comment in
`wheel_wheel.py` says so in those words. **A fix that is correct is not thereby the fix for
the thing you attributed it to.**

The residual +3 minutes is **unattributed**, and the one contributor known by construction
is this arc's own `fd_tol_rel` 1e-8 -> 1e-9 in `run_axle_drop`, which buys G9's reference
more secant iterations. The rest is presumed to be the uncapped mesh's solves converging
differently; nobody has measured it and this record does not claim otherwise. **The new
baseline is 31m40s, measured twice and consistent to 10 s.**

## WHAT IS DELIBERATELY NOT DONE

**`best_solution.json` is untouched.** Its `metrics` block now describes a mesh the tree
no longer builds by default — `axle_drop_mean_mm`, `min_scaled_jacobian`,
`pnorm_stress_agg_mpa`, `max_stress_mpa`, `stress_utilisation` all move under the flip.
Re-scoring it is a promotion-shaped operation with a promotion-shaped checklist
(`tests/test_promotion.py`), and it belongs to whoever decides whether the genome should
be re-descended on the re-priced objective rather than merely re-measured on it. **Do not
quote `best_solution.json`'s metrics against a default-built mesh without saying which
mesh produced them.**

**`make studies` is NOT refreshed.** §37 named it as part of this baseline and it is the
one item of that list left open. Two things are worth recording rather than assuming:

  - **The reason it "cannot run" is stale.** The standing belief was that
    `study_gnl.py` exits 1 on the shipped wheel's breached small-load gate, stopping the
    recipe at line 5 of 9. §33 already fixed that: `main` now returns
    `0 if rep["solver_is_correct"] else 1`, and records `pass_means` = "solver_is_correct;
    NOT the small-load regime gate". The blocker is cost, not exit status.
  - **The cost is real**: §33 puts the four slowest drivers alone at ~4h15m, and every
    committed `studies/*.json` now describes the capped mesh. They should be refreshed in
    one pass, not piecemeal, and §15's precedent says record staleness rather than patch
    it mid-arc.

**No genome is re-descended.** The objective moved; nothing was re-optimised against it.

## `make studies` — REFRESHED FIVE OF NINE, AND THE OLD BLOCKER WAS HIDING A NEW ONE

```
  1  study_mesh_quality      ran
  2  study_wheel_mesh        PASS   refreshed
  3  study_beam_agreement    PASS   refreshed
  4  study_wheel_fea         PASS   refreshed
  5  study_gnl               PASS   refreshed   <- the driver believed to block this recipe
  6  study_contact           FAIL   artifacts written, then exit 1 -> make stopped
  7  study_gradient          not reached
  8  study_objective         not reached
  9  study_stage3            not reached
                                                 23m15s total
```

**`study_gnl` PASSES**, printing §33's own `pass_means`: "solver_is_correct (PLAN.md §33);
NOT the small-load regime gate". The standing belief that this recipe cannot complete
because `study_gnl` exits 1 on the breached small-load gate was **a full arc out of date**
— §33 decoupled them and nobody re-ran the recipe to find out. The cost of checking was one
`grep` of `main`'s return statement.

**And it was masking a failure one line further down.** `study_contact`'s G1 — the penalty
stiffness plateau — reads **1.021e-03 against a `< 1e-03` gate**. No test pins this gate;
it is checked only by `make studies`, which had not reached it in weeks.

### THE FLIP CAUSED IT, AND THE GATE WAS ALREADY AT 96% OF ITS LIMIT

Measured directly rather than inferred, `coarse`, the two solves G1 differences:

```
                       drop @1e4   drop @1e5   penetration @1e4      rel     gate 1e-3
  capped (uncap=False)  1.532714    1.531242       5.9878e-04     9.610e-04   PASS
  DEFAULT (True, 1.0)   1.463794    1.462301       5.5167e-04     1.021e-03   FAIL
```

Both halves of the cause are visible. The denominator — the axle drop — falls 4.5%, which
is the flip. The numerator rises 1.4%. Product: 9.610e-04 -> 1.021e-03, over by 2%.

**But look at what the gate is normalised by.** G1's numerator is a penetration difference,
set by `eps_n` and the contact force; its denominator is the axle drop. **Making the wheel
STIFFER — that is, making the model MORE faithful — makes this gate HARDER to pass, with
nothing about the contact model having changed.** And the other half of the same criterion,
the half anchored to a physical length rather than to the answer, moves the RIGHT way:
penetration falls 5.99e-04 -> 5.52e-04 mm, 3.68e-04 of the 1.5 mm band, comfortably inside
its own 1e-3.

**So: the physically-anchored half improves and the self-normalised half fails.**

### WHAT IS DONE ABOUT IT: NOTHING, DELIBERATELY

`GATE_EPS_PLATEAU_REL` is **not moved** and G1 is **not re-normalised**. §19 forbids the
first, and the second is the same act in better clothes — a gate re-anchored until it
passes is not a gate. It is also not this arc's call: `run_penalty_plateau`'s docstring
records that this criterion was revised ONCE already, on measurement ("THE FIRST VERSION OF
THIS GATE ASKED FOR THE WRONG THING, AND MEASUREMENT SAID SO"), and a second revision
deserves the same standard and its own decision.

What is recorded is the pair of facts that decision needs: **the gate was at 96% of its
limit before this arc touched anything**, so it was one small change away from red in any
case; and **its normalisation prices model fidelity as a regression**, which is worth
knowing before anyone tunes `eps_n` to satisfy it.

**Artifacts on disk are now mixed and this is stated so nobody mistakes it for a clean
refresh**: five drivers describe the uncapped mesh, `study_contact.json`/`.jpg` describe it
with a red gate (they are written before the non-zero exit), and `study_gradient`,
`study_objective` and `study_stage3` still describe the CAPPED mesh.

## THE SUCCESSORS, RANKED

1. **Decide what the re-priced stress term means for the shipped genome.** This is the
   only item that follows from the finding rather than from the flip. The shipped design
   was descended with `stress_margin` live at 0.1317 and would now feel none of it. Two
   coherent answers — "re-descend on the faithful objective" and "the design is fine and
   the term was doing no useful work" — and they are distinguishable by one descent.
   That measurement is DONE and it came back "none": the thinnest wall in the tree reads
   0.78664, the shipped genome 0.71316, and the shipped genome's `stress_margin` is
   exactly 0.000000 against the 0.13168 it was descended with. So the question is no
   longer "is the term inactive" but "was it doing useful work when it was active".
2. **`make studies`**, in one pass, once nothing else is moving.
3. **`FILLET_PLAN.md`** — unchanged by this step. §37 discharged the caveat on its 0.2 mm
   ceiling; the blocker is still that the spoke block is ruled.
4. **§32's successors 3 and 4** — §8's wall-floor economics under SVK, and the kernel
   default. **Checked, and the premise holds**: every `stage3_minwall_*.json` in the tree
   is dated 2026-08-03 with `kinematics` unset, i.e. measured on the LINEAR kernel at
   `coarse`/125 steps. The re-run under SVK is genuinely unmeasured. Pure compute.
5. **The rim tri-block**, still filed, still not binding (UNCAP_PLAN Step 3).
   **SUPERSEDED 2026-08-23 — it is BUILT; see STEP 3 RECORD, PART 2 at the end of this
   file, and PLAN §53.**

---

# STEP 3 RECORD, PART 2 — 2026-08-23. THE TRI-BLOCK IS BUILT. BOTH OF STEP 3's CLAUSES ARE RETIRED, IT MESHES AT 77x THE BLOCK IT REPLACES — AND A THIRD OBSTACLE NEITHER §37 NOR §51 NAMED IS WHAT ACTUALLY STOPS IT

> **SCOPE — THE BOX HERE IS THE UNIFORM DRAW, NOT THE DESIGN SPACE (PART 8, PLAN §76).**
> Every genome-box figure in this part comes from `study_mesh_quality.latin_hypercube`
> over the full gene box at `GENOME_SWEEP_SEED`. That sampler puts about **one genome
> above 35 degrees of arc span in sixty-four**, and a draw conditioned on arc span
> reaches bows to 1.25 against its maximum of 0.54. The numbers below are correct about
> what was drawn; read "the box" as "this draw" throughout.

Step 3 stopped **before building it**, on two checks that needed nothing built, and was
right to: neither of them was about whether the partition works. §51 re-priced both with a
scratch probe, said both were wrong, filed the probe **as** a probe — *"no driver, no
artifact, no test — precisely so that nobody quotes it as this project's other numbers may
be quoted"* — and named the unit that would make it a measurement. This is that unit.

Driver `studies/study_tri_block.py`, artifact `studies/study_tri_block.json`,
`make triblock`, 23 tests in `tests/test_tri_block.py`, 14 seconds.

**§51's probe numbers are superseded by everything below and may now be quoted from
here.**

## THE ALGEBRA — CLAUSE 2 IS RETIRED, AND STEP 3's OWN ARITHMETIC WAS NEVER WRONG

Step 3 derived the constraint and it is correct. Writing the three sides as A (the ring
arc, `n_weld`), B (the free side) and C (the end cross-section, `n_thick`), matching
opposite sides of the three quads forces `a1 = b2`, `a2 = c1`, `c2 = b1`, hence

```
    a1 = (A + B - C) / 2
```

**The input was wrong, not the algebra.** Step 3 took `B = 8` at `coarse` — the count the
block it was replacing happened to carry — and got 7x1, 3x1, 7x3 with a forced 1-element
strip. The driver reproduces that exactly and gates its own exit on doing so. But **B is
the free side and its count was never inherited**: the admissible set is every B with
`|A - C| < B < A + C` and `A + B - C` even, enumerated rather than argued.

```
  config  A   C     admissible B          strip-free B
  coarse  10  4     8, 10, 12             10
  medium  16  6     12, 14, 16, 18, 20    14, 16, 18
```

At `coarse` the strip-free choice is unique and Step 3 missed it by one grid point. At
`medium` three of the five have no strip at all.

## THE SEAMS — CLAUSE 1 IS RETIRED, AND IT IS A CLAIM ABOUT INDICES

Step 3: *"it splits all three sides, two of which are shared ... **so it needs PARTIAL-EDGE
SEAMS**, and whole-edge single ownership is what this module's docstring calls the whole
safety net."* True **only if the neighbours may not be split**, and §48 is the whole
argument that they may.

`rim_band_weld` is cut in theta at `M_A`; the `spoke` is cut along a j-line at `M_C`; that
cascades once into `hub_junction`, because the spoke's hub row IS the junction's `left`
edge, and **stops** there, because the junction's cut runs across the collar arc rather
than along it. **Seven blocks become twelve and SEVENTEEN seams, every one a whole edge of
both blocks it names.**

```
  the twelve                          the seventeen, by kind
  spoke_eta_lo / spoke_eta_hi         1  the cut through the spoke
  hub_junction_lo / hub_junction_hi   2  the spoke's hub cross-section, in two pieces
  rim_tri_t / rim_tri_q / rim_tri_b   1  the cut through the hub junction
  hub_collar_weld / hub_collar_free   2  the spoke's rim cross-section, onto two quads
  rim_band_weld_q / rim_band_weld_t   3  the Y's own internal edges
  rim_band_free                       3  the two arcs onto their ring blocks
                                      1  the cut through the ring's weld block
                                      2  weld to free, both rings
                                      2  free to the NEXT sector's weld — closes the 360
```

**Every seam closes, at both configs, worst gap 7.1e-15 mm**, and the Y's own three are
exactly 0.0 because each internal edge is passed as the SAME array to both of its blocks.

## AND IT MESHES — BY A FACTOR OF 77

```
  config   shipped blend 1.0   faithful QUAD   faithful TRI       x   clears 0.2
  coarse       0.782735          0.008176        0.626233      76.6x     YES
  medium       0.782926          0.008251        0.581582      70.5x     YES
```

`MIN_SJ_TARGET` is 0.2 with a barrier weight of 3000, and it is imported from
`wheel_objective` rather than written down. The tri-block clears it by **3.1x** and lands
at **80%** of the shipped mesh's own 0.783 — while delivering the 1.06 degree corner
fidelity that was the whole point. **§51's probe said ~0.25 and called it "a floor rather
than an estimate". It was one.**

The three neighbours are **sliced, not rebuilt**: `spoke`, `hub_junction` and
`rim_band_weld` come out of `sector_blocks` and are cut at a node index, bit-for-bit, so
the 77x is a measurement of a partition and not a comparison of two constructions. Pinned
at `== 0.0`, not at a tolerance. The three quads' areas sum to the quad block's to 1e-5
relative — they tile the region rather than overlap it, and the residual is the free
side's own re-distribution, which the partition is allowed.

## THE THIRD OBSTACLE, WHICH IS THE ONE THAT ACTUALLY STOPS IT

**The faithful rim is not opt-in.** §48 could measure the filleted blocking at one genome,
name six refusals out of sixteen and still hand STEP 2 a usable instrument, because
`fillet=` is passed by a study and never by the optimizer. Adopting blend 0.0 changes
`sector_blocks` for **every genome the search touches**. So the gene box is the
measurement and one genome is not.

Sixteen freshly drawn feasible genomes, four per flank orientation, `B` held at its
per-config value because element counts may not depend on the design:

```
  config   fixed rule valid   best-per-genome valid   clears 0.2 (fixed)   seams
  coarse       12/16                15/16                  11/12          all close
  medium       10/16                12/16                  10/10          all close
```

`fixed rule` applies the shipped genome's own barycentric triple — barycentric weights are
scale-free, so that is a construction with no free parameter left, which is what would
actually ship. `best-per-genome` re-sweeps the interior point and is **the upper bound any
adaptive rule could reach**. It is not 16/16 either.

**The mechanism is named, not just counted.** The ones it folds on are the WIDE weld arcs:
15.9-41.2 degrees against 3.7-11.9 on the ones it does not, at `coarse`, where the two
ranges separate cleanly. The shipped genome's arc is **2.73 degrees**, at the very bottom
of the box. A barycentric point tuned on a 2.7-degree sliver is in the wrong place on a
40-degree triangle.

**So the CONSTRUCTION is proved and the RULE THAT PLACES ITS INTERIOR POINT is not.**

## AND A GENERATED INTERIOR CANNOT HELP — WHICH NAMES THE SUCCESSOR

A Winslow solve on each quad's interior, boundaries held, changes the number by
**0.000000** at both configs. The worst corner is ON a held boundary. The Y's three spokes
are boundaries of two blocks each, so per-block smoothing holds them by definition — the
number is set by **where the Y's spokes go**, not by how the interiors are filled.

Same shape as PART 9's route-2 invariance, and the same conclusion: **a curved Y is the
successor; a better smoother is not.**

## THE RULE THAT PICKS THE CELL, AND ONE PLACE IT WAS NOT FOLLOWED AND SHOULD NOT BE

The argmax of the worst tri block's min scaled Jacobian over the **published** grid, and
the grid is published in full so a plateau and a tuned point are told apart by looking.
**Here it is a tuned point and the report says so**: only 6.9% of valid cells at `coarse`
and 8.3% at `medium` sit within 10% of the maximum, and only 29/173 and 24/173 cells are
valid at all.

A finer local re-sweep is reported and **deliberately not adopted**. At `medium` it gains
0.045 by walking `w_Pt` down to the search box's own clamp, and that cell generalises
across the gene box *worse* than the grid point it beats. A number that only exists at
four decimal places of one weight is the thing §48's ridge rule was written to keep
visible rather than to chase.

## THE BUG THE INSTRUMENT FOUND, RECORDED BECAUSE OF HOW IT HID

The first seam table paired the wrong halves of the cut spoke and the cut hub junction.
The hub junction's `left` edge is the spoke's hub row **reversed** when the straddling
flank there is at eta = +1, so the low-j half of one is the high-j half of the other. At
`coarse` `n_thick` splits 2/2, so **the node counts still agreed** — only the coordinates
disagreed, by 0.62 mm. A seam check that reported closure as a single boolean would have
read "17/17 counts agree" and said nothing. It reports the count and the gap separately,
which is why this took one run to find. `test_the_seam_table_follows_the_GENOME_and_not_a_constant`
reproduces the mismatch and asserts that the counts still agree under it.

The second was the ring's weld block: its low-theta end is `P_t` when the junction's arc
ascends and `Q` when it descends, so which half is which follows the genome. Exactly the
trap §48 hit on the sector-closing seam's `dk`.

## WHAT IS UNCHANGED

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, no threshold
moved, `UNCAP_DEFAULT` still `(True, 1.0)`, and `sector_blocks` still returns seven
blocks.** Blend 0.0 is measured, never adopted. `test_nothing_here_is_wired_into_the_mesh_the_tree_BUILDS`
pins all of it.

## THE SUCCESSORS THIS STEP LEAVES

1. **A rule for the interior point that holds across the gene box.** This is the whole of
   what is missing, and it is now a well-posed problem rather than an open one: 16 drawn
   genomes with their triangles' arc spans, side lengths and three wedge angles are in the
   artifact, the failure mode is the wide arc, and the upper bound any rule can reach at
   the current `B` is 15/16 and 12/16 — so a rule alone may not be enough and the curved Y
   below may be needed with it.
2. **The curved Y.** The Winslow column says the number is set by where the spokes go, and
   they are straight lines today. This is the only lever the measurement points at.
3. **Then, and only then, the decision to adopt the faithful rim** — which is a model
   change with its own baseline: full `make test`, `make studies`, and the mass /
   hub-share / axle-drop deltas, exactly as Step 4 needed for the flip.

---

# STEP 3 RECORD, PART 3 — 2026-08-23. A FIXED RULE DOES REACH THE CEILING, AND FOR ONE CONFIG IT COSTS THE SHIPPED GENOME NOTHING TO GET THERE

> **SCOPE — THE BOX HERE IS THE UNIFORM DRAW, NOT THE DESIGN SPACE (PART 8, PLAN §76).**
> Every genome-box figure in this part comes from `study_mesh_quality.latin_hypercube`
> over the full gene box at `GENOME_SWEEP_SEED`. That sampler puts about **one genome
> above 35 degrees of arc span in sixty-four**, and a draw conditioned on arc span
> reaches bows to 1.25 against its maximum of 0.54. The numbers below are correct about
> what was drawn; read "the box" as "this draw" throughout.

PART 2's item 1 asked whether a FIXED barycentric triple — the one with no free parameter
left, the one that would actually ship — can be re-derived to reach every genome its own
`best_w` reaches, rather than the sixteen-genome argmax `sweep` performs at one genome
alone. `sweep_w_genomes`, new in `studies/study_tri_block.py`, answers it: a joint argmax
of the worst genome's worst block, over the same sixteen genomes `sweep_genomes` already
draws PLUS the shipped genome (named and appended, exactly as
`study_fillet_block.sweep_layer_profile_genomes` does for its own shipped genome), against
a dedicated 25x25 barycentric grid published in full (`GENOME_ROBUST_X_GRID_N`).

**The objective is `n_clear` first and the worst value second, not the reverse.** A raw
argmax of the worst genome's worst block chases whichever genome sits closest to folding,
which turned out to be a DIFFERENT question from how many genomes clear the barrier the
optimizer actually enforces — at `coarse` the published grid has exactly one cell where
every fixable genome is simultaneously valid, and it is not the cell a worst-case argmax
would have picked, nor is it the cell that clears the most genomes. Ranking by
`(n_clear, worst)` finds the one that does both.

```
config   n_cells   current w valid/clear   genome-robust w valid/clear   shipped genome
coarse      16           13 / 12                  15 / 13               0.6262 (UNCHANGED)
medium      13           11 / 11                  13 / 12               0.4336 (was 0.5816)
```

`n_cells` is the drawn genomes `sweep_w_genomes` can even ask the question of — the sixteen
`sweep_genomes` draws, MINUS the one no `w` rescues at that config (`best_w_valid: False`:
`41.2°` alone at `coarse`; `35.3°, 15.9°, 41.2°, 22.7°` at `medium`) — PLUS the shipped
genome, appended and counted in. It is not the same denominator as PART 2's "12/16" /
"10/16" table, which counts against all sixteen drawn genomes without excluding the
unfixable one; both are reported in the artifact and neither should be read as the other.

**At `coarse` this is free.** The genome-robust cell leaves the shipped genome's own min
scaled Jacobian at 0.6262 — bit-identical to today's single-genome rule — while lifting two
more of the drawn genomes off the fold and one more over the barrier. There is no reason
not to prefer it if this arc is picked up.

**At `medium` it is not free, but it is cheap.** The shipped genome's own number drops from
0.5816 to 0.4336 — the quoted multiplier over the collapse falls from roughly 70x to
roughly 52x — while two more drawn genomes stop folding and one more clears the barrier.
0.4336 still clears `MIN_SJ_TARGET` by more than double, so nothing that reads this file's
headline table as "does the tri-block clear the barrier" would see a different answer; only
the specific multiplier would print a smaller number.

**It is measured, not adopted, and the reason is different from every prior use of that
phrase in this project.** `blend 0.0` and §48/§54's fillet profile were reported-not-shipped
because adopting them would move a mesh the OPTIMIZER or an already-published FEA result
depends on. Nothing here does: the tri-block is not wired into `sector_blocks`, so
`chosen` — the cell `sweep`'s single-genome argmax picks — is read by nothing except this
file's own printed table and `per["sector"]` in the committed artifact. Adopting the
genome-robust cell would change a quoted number and nothing else. It is left unshipped
anyway, so that a single stated rule ("the argmax over the published grid, at the shipped
genome") keeps meaning one thing, and so that the choice of which number this arc quotes
going forward is made by whoever next picks it up rather than by this session substituting
one argmax for another inside a file that already had a settled headline.

**What this retires, and what it does not.** PART 2's item 1 asked whether "a rule alone
may not be enough" — a FIXED rule reaches full validity on every genome its own `best_w` (a
free per-genome parameter) reaches at `medium` (13/13), and within one genome of it at
`coarse` (15/16), which is the strongest form of "a rule holds across the gene box" this
file can state without inventing a new construction. What it does not do is rescue the
genome no `w` reaches at either config (41.2°, plus three more that are unreachable at
`medium` alone) — that is still the curved Y's question, unchanged from PART 2, and the
ceiling itself says a rule alone cannot close it.

`tests/test_tri_block.py::test_a_genome_robust_w_reaches_more_of_the_box_without_being_adopted`
pins the comparison at the two named cells (re-derived fresh, not read from the artifact)
and that `chosen` has not moved.

---

# STEP 3 RECORD, PART 4 — 2026-08-23. THE CURVED Y IS BUILT. IT RESCUES WHAT PART 2 SAID IT WOULD, AND THE ONE GENOME LEFT REFUSES FOR A REASON THE MEASUREMENT DOES NOT NAME

> **SCOPE — THE BOX HERE IS THE UNIFORM DRAW, NOT THE DESIGN SPACE (PART 8, PLAN §76).**
> Every genome-box figure in this part comes from `study_mesh_quality.latin_hypercube`
> over the full gene box at `GENOME_SWEEP_SEED`. That sampler puts about **one genome
> above 35 degrees of arc span in sixty-four**, and a draw conditioned on arc span
> reaches bows to 1.25 against its maximum of 0.54. The numbers below are correct about
> what was drawn; read "the box" as "this draw" throughout.

PART 2's Winslow column found that an elliptic interior solve changes the tri-block's
number by **0.000000**, because the Y's three spokes are BOUNDARIES of two blocks each and
per-block smoothing holds them by definition. Its conclusion was that the number is set by
where the spokes GO, and they were straight lines — so a curved Y was the successor and a
better smoother was not. PART 3 then closed the interior point's half. This closes the
other one.

## WHAT THE CURVE IS, AND WHY THIS ONE

`_bent_spoke` in `studies/study_tri_block.py`. Each spoke is the opposite edge of two of
the three quads, and in each of them it faces a piece of the region's own boundary: `sC`
faces the arc in `rim_tri_t` and the free side in `rim_tri_b`, `sA` faces the cross and the
free side, `sB` faces the cross and the arc. Those two curves are what the spoke is bent
TOWARD — the blend `(1-frac)*u + frac*v` at the fraction where the spoke's own foot sits
between them, which is the curve the region's two sides say should be there. That blend
does not pass through the spoke's endpoints; two linear terms pin it back to them, which is
the Coons correction and is why **the endpoints are exact for every bend and the three
internal seams stay exact by construction rather than by tolerance**.

Three things fall out of the construction rather than being arranged:

  * **It needs no resampling.** `splits` already forces `a1 == b2`, `a2 == c1` and
    `c2 == b1`, so each spoke and the two curves it blends carry the SAME node count. The
    blend is node-for-node.
  * **`bend = 0.0` returns the straight spoke untouched**, not an array equal to it. Every
    number this file published before the curve existed is reproduced bit for bit —
    verified by diffing the regenerated artifact against the committed one, where the only
    non-additive change is the wall clock.
  * **The bend moves seams and nothing else.** The six boundary edges the region owns are
    bit-identical across bends (`the_bend_moves_no_boundary`), the tiled area shifts by
    7e-16 relative — one ULP of three shoelaces whose shared spokes cancel only to rounding
    — and all 17 seams still close to 7.11e-15 mm at the bend the joint rule picks.

## WHAT IT REACHES

`sweep_bend_genomes` runs the same two-column claim `sweep_genomes` makes — a per-genome
ceiling and one fixed rule — over the (w, bend) plane instead of the w one: 308 interior
points x 11 bends x 17 genomes (the sixteen drawn plus the shipped one, named and
appended), at both configs.

```
                        per-genome ceiling            one FIXED (w, bend) rule
config   straight    curved   rescued      straight valid/clear/worst   curved valid/clear/worst
coarse     16/17     16/17       0            15/16  13  -0.0204          16/16  13  +0.0478
medium     14/17     16/17       2            13/16  12  -1.0000          13/16  13  -0.4875
```

**At `coarse` the curve turns a fixed rule that folds into one that does not.** The best
fixed straight `w` leaves one of the sixteen reachable genomes inverted at -0.0204; the same
`w` at **bend 0.20** puts every one of them valid, with the joint floor at **+0.0478**. And
it costs the shipped genome nothing: 0.6262 at the published cell, 0.6262 at the joint
curved rule, because the bend is inert there.

**At `medium` it does not raise the count, and it raises the ceiling instead.** Two genomes
that fold at every placement of the interior point become valid once the spokes may follow
the region — 35.3° reaching **0.3464** and 15.9° **0.1698** — and a third, 22.7°, goes from
0.0780 to **0.2366**, over `MIN_SJ_TARGET` rather than merely off the fold. The fixed rule's
count is unchanged at 13/16 but it clears the barrier on one more and lifts the joint floor
from -1.0000 to -0.4875. The shipped genome pays 0.5816 -> 0.4384 to sit at that joint rule
— still more than double the barrier, and still a choice nothing is forced into.

**THE BEND IS INERT WHERE THE REGION IS FAT.** This is the finding that says the curve is
not a knob being tuned. Of seventeen genomes at `coarse`, exactly ONE wants a non-zero bend;
at `medium`, four do. Every other genome's own optimum is bend 0.0 — the straight Y — and
at the published cell the shipped genome's number moves across the WHOLE bend range by
0.000000 at `coarse` and 0.001 at `medium`. A correction to cutting chords does nothing
where the chords were fine.

## THE MECHANISM, AND THE HALF OF IT THAT IS NOT ONE

PART 2 named the straight Y's failure mode as **the wide weld arc**. That was right about
the correlation and wrong about the quantity. The arc span is an angle; what a chord cannot
survive is a side that **bows away from one by a fair fraction of the region's own width**,
and `bow_over_width` — the arc's greatest departure from its chord, over the cross
section's length — is now in `region_report` and in every genome row.

It separates cleanly at `coarse`: the genomes the fixed straight rule folds on run
**0.264-0.498**, the ones it does not run **0.009-0.129**, with nothing between. It is also
what tells 18.5° (bow 0.149, drawn at `medium` and valid there at bend 0) from 15.9° (bow
0.264, folded at `medium` at every straight placement) — two genomes the arc-span story
gets backwards, because 18.5° is a wide arc across a fat 4.24 mm region and 15.9° is a
narrower one across a 1.77 mm sliver.

**And it does not explain the survivor, which is stated here because it would be easy to
imply that it does.** The genome with the LARGEST bow in the whole box — 35.3° at 0.498 — is
one the curve reaches, at both configs. The one that refuses has a bow of 0.491, smaller.
**[READ "THE WHOLE BOX" AS "THE UNIFORM DRAW" — PART 8.** A draw conditioned on arc span
reaches bows to 1.25 against this sampler's 0.54, so 0.498 is not the largest bow the design
space holds. The comparison between the two genomes stands; the superlative does not.**]**
So the bow says where the bend is NEEDED; it does not say what makes a region impossible,
and that quantity is still unnamed. `test_what_the_curve_does_NOT_reach_stays_reported`
asserts the non-separation rather than the separation, so that a future run which does
separate them is a finding and not a silent pass.

## THE REFUSAL, PRICED AT EVERY FREE COUNT

One drawn genome — 41.2°, sides 34.88 / 29.41 / 6.32 mm — folds at every interior point AND
every bend, at both configs. Because `B` is held across the gene box for a reason that has
nothing to do with this question (it sets element counts, and a mesh whose count depends on
the design cannot be compared across a search), the refusal is re-asked at **every
admissible free count** rather than at the one that ships:

```
coarse   B = 8  -0.0134    B = 10 -0.0195    B = 12 -0.0189
medium   B = 12 -0.0141    B = 14 -0.0194    B = 16 -0.0180    B = 18 -0.0174   B = 20 -0.0202
```

Valid at none of them. The ceiling is **-0.0134**, a hair below zero rather than a
collapse, and a hand probe outside the published search box (pushing `w_Bstar` to 0, i.e.
the interior point onto the P_t-Q chord) only reaches -0.0064 — so it is not the box
clamping the answer either. `every_refusal_was_re-asked_at_every_free_count` is a
self-check.

## WHAT IS UNCHANGED

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, no threshold
moved, `UNCAP_DEFAULT` still `(True, 1.0)`, and `sector_blocks` still returns seven
blocks.** `bend` defaults to 0.0, `chosen` is still `sweep`'s single-genome argmax at the
straight Y, and `per["sector"]` still reports 0.626233 / 0.581582 and 76.6x / 70.5x.
`test_nothing_here_is_wired_into_the_mesh_the_tree_BUILDS` and
`test_the_bend_is_OFF_by_default_and_off_means_untouched` pin both halves.

`make triblock` is now ~290 s rather than ~15 s, and the Makefile's help says so.

## THE SUCCESSORS THIS PART LEAVES

1. **What makes a region impossible.** The bow explains the straight Y's folds and does not
   explain the one refusal that survives the curve. Sixteen genomes with their bows, arc
   spans, side lengths, three wedge angles and per-`B` ceilings are in the artifact, and
   the refusal's ceiling is -0.013 rather than -0.9 — a hair, not a collapse. Whatever
   separates it from the 0.498-bow genome the curve reaches is one quantity and it is not
   yet measured.
2. **A bend that is a FUNCTION of the genome rather than a constant.** The per-genome
   ceiling at `medium` reaches 16 of 17; one fixed (w, bend) reaches 13 of 16. The gap is
   larger than the straight Y's was, because the curve enlarged what is reachable without
   making a constant rule better at reaching it. `bow_over_width` is the obvious argument
   for such a function and the per-genome optima are in the artifact to fit against —
   though note that they are argmaxes over a plateau and their scatter is partly that.
3. **Then, and only then, the decision to adopt the faithful rim** — unchanged from PART 2.
   A model change with its own baseline: full `make test`, `make studies`, and the mass /
   hub-share / axle-drop deltas.

---

# STEP 3 RECORD, PART 5 — 2026-08-23. THE FOLD MARGIN IS RULED OUT AS THE THING THAT MAKES A REGION IMPOSSIBLE, WHICH IS ONE CANDIDATE OFF PART 4's OWN LIST

> **SCOPE — THE BOX HERE IS THE UNIFORM DRAW, NOT THE DESIGN SPACE (PART 8, PLAN §76).**
> Every genome-box figure in this part comes from `study_mesh_quality.latin_hypercube`
> over the full gene box at `GENOME_SWEEP_SEED`. That sampler puts about **one genome
> above 35 degrees of arc span in sixty-four**, and a draw conditioned on arc span
> reaches bows to 1.25 against its maximum of 0.54. The numbers below are correct about
> what was drawn; read "the box" as "this draw" throughout.

PART 4 left "what makes a region impossible" as its first successor and said the
separating quantity is *"one quantity and it is not yet measured"*.  Nothing here measures
it.  What this does is remove a candidate that was about to be reached for, and remove it
with a number rather than an argument.

`study_fillet_block`'s PART 15 built a fold gate: `wheel_geometry.self_intersection_margin`,
the closed-form clearance before a spoke's offset band turns inside out, which classifies
that study's one inverted block exactly and catches a class of drawn genome whose part does
not exist.  **Both blocking studies draw from the same seed and the same stream, and draw
the same sixteen genomes** — verified rather than assumed — so those genomes are in this
box too, and the fold margin is now carried on every row here as `fold`.

It explains nothing about this construction:

```
  coarse   2/16 fold.  They sit at fixed-rule +0.5337 and +0.2104 — both above the barrier,
           and one of them is the best cell in its whole orientation group.
           The WORST cell in the box, -0.9597, has margin +0.1299 mm and folds nothing.
  medium   1/16 folds, at +0.4756.
           The WORST cell, -1.0000, has margin +2.7720 mm and folds nothing.
```

This is the expected answer and that is exactly why it is written down.  The tri-block
partitions the rim JUNCTION region; the fold margin is a statement about the spoke's OFFSET
BAND, which no block in this partition touches.  Two boxes drawn from the same sixteen
genomes, two different constructions, and the feasibility number that classifies one of
them 16/16 is anti-informative on the other — the hardest cell here is fold-clean at both
configs.

`the_fold_margin_does_not_explain_the_tri_block` gates it, and
`test_the_fold_margin_is_NOT_what_makes_a_region_impossible` pins it per genome.  Both are
written so that a future run in which the fold margin DOES start explaining refusals here
fails rather than passing quietly, because that would be a finding: it would mean the two
regions are coupled in a way nothing in this file currently believes.

## WHAT IS UNCHANGED

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, no threshold
moved, `UNCAP_DEFAULT` still `(True, 1.0)`, `bend` still defaults to 0.0, and no draw
filter changed** — the margin is reported on each row, not applied to the draw.
`study_tri_block.json` was diffed field-by-field against the committed one and is **purely
additive**: sixteen `fold` blocks per config and one self-check, 33 fields in all, with
every previously-committed field reproducing exactly.  `make triblock` is unchanged at
~290 s.

## WHAT PART 5 LEAVES

Unchanged from PART 4, minus one candidate.  The separating quantity is still unnamed; the
bow does not explain the surviving refusal, and now neither does the fold margin.  What is
known about it is narrow and worth restating: the refusal's ceiling over every admissible
free count is **-0.013**, a hair rather than a collapse, and it sits at bow 0.491 while the
box's largest bow, 0.498, is a genome the curve reaches.

---

# STEP 3 RECORD, PART 6 — 2026-08-23. THE REFUSAL IS EXTREMAL ON THE REGION'S INTERIOR-ANGLE SUM, BY MORE THAN HALF THE SPREAD OF EVERYTHING ELSE — AND WITH ONE NEGATIVE EXAMPLE THAT IS A CANDIDATE, NOT A MECHANISM

> **SCOPE — THE BOX HERE IS THE UNIFORM DRAW, NOT THE DESIGN SPACE (PART 8, PLAN §76).**
> Every genome-box figure in this part comes from `study_mesh_quality.latin_hypercube`
> over the full gene box at `GENOME_SWEEP_SEED`. That sampler puts about **one genome
> above 35 degrees of arc span in sixty-four**, and a draw conditioned on arc span
> reaches bows to 1.25 against its maximum of 0.54. The numbers below are correct about
> what was drawn; read "the box" as "this draw" throughout.

PART 4 left "what makes a region impossible" as its first successor: one drawn genome folds
at every interior point, every bend and every admissible free count, at both configs, and
neither the bow (PART 4) nor the fold margin (PART 5) explains it.  This asks the artifact's
remaining shape numbers.

**Three of them separate the refusal from all fifteen reached genomes, at both configs:**

```
  quantity              refusal   others min   others max      gap   gap/spread
  coarse
    wedge sum deg       156.371      170.330      194.703   13.960        0.573
    |wedge sum - 180|    23.629        0.174       14.703    8.927        0.614
    arc_span_deg         41.209        3.714       35.312    5.897        0.187
  medium
    wedge sum deg       156.667      170.489      194.906   13.822        0.566
    |wedge sum - 180|    23.333        0.308       14.906    8.427        0.577
    arc_span_deg         41.217        3.714       35.324    5.893        0.186
```

**The interior-angle sum is the widest: the refusal sits 14 degrees below the minimum of the
other fifteen, a gap that is 57% of their entire spread.**  Arc span separates too but only
by 19% of its spread, and it is the quantity PART 4's print already uses for the STRAIGHT
Y's folds — where the ranges genuinely do separate — so its appearance here is partly that
same effect.  `bow_over_width`, `turn_at_far_end_deg`, the smallest wedge and the A/C side
ratio all fail to separate, which is PART 4's negative reconfirmed and extended.

## AND WITH ONE NEGATIVE EXAMPLE THIS IS NOT YET A MECHANISM

Stated plainly because it would be easy to write this up as an answer: **any quantity on
which the single refusal happens to be extremal will "separate" a set of one from a set of
fifteen.**  What makes the angle sum more interesting than that is the size of the gap
relative to the spread, and that it is stable across configs — not that it is derived.

A derivation was attempted and does not hold.  In the plane, Gauss-Bonnet gives
`interior angle sum = 180 deg + total boundary turning`, which would have made the sum a
statement about how concave the region is.  Measured, the three sides' turnings correlate
with `sum - 180` at only **0.355** across the box — because the sides are not consistently
oriented between flank orientations, so the signed total is not comparable genome to genome.
The identity is presumably recoverable with the orientations reconciled; it is not
recovered here, and the interpretation is therefore withheld rather than asserted.

## WHAT WOULD SETTLE IT

**A second refusal.**  The box has one because sixteen genomes is a small draw, and every
statistic above is a set of one against a set of fifteen.  Drawing further until a second
region refuses the curve at every bend and every free count would turn all four candidates
into testable claims at once, and would cost about one more `make triblock` — the curved-Y
sweep is the expensive part at roughly 290 s for sixteen genomes, so thirty-two is ten
minutes.  Until then the honest statement is the one this part opens with: the refusal is
extremal on the interior-angle sum by more than half the spread, and that is where to look.

## WHAT IS UNCHANGED

**Nothing promoted, no code changed, no artifact regenerated** — every number above is read
from the committed `study_tri_block.json` or computed from the genes it carries.

---

# STEP 3 RECORD, PART 7 — 2026-08-24. THE EXPERIMENT PART 6 NAMED, RUN: NO SECOND REFUSAL AT SIXTY-FOUR GENOMES, AND THE CANDIDATE IT WAS BUILT TO CONFIRM COLLAPSED FROM 57% OF THE SPREAD TO 4%

> **SCOPE — THE BOX HERE IS THE UNIFORM DRAW, NOT THE DESIGN SPACE (PART 8, PLAN §76).**
> Every genome-box figure in this part comes from `study_mesh_quality.latin_hypercube`
> over the full gene box at `GENOME_SWEEP_SEED`. That sampler puts about **one genome
> above 35 degrees of arc span in sixty-four**, and a draw conditioned on arc span
> reaches bows to 1.25 against its maximum of 0.54. The numbers below are correct about
> what was drawn; read "the box" as "this draw" throughout.

PART 6 ranked the region's interior-angle sum first among three separators and said plainly
that with one negative example it was arithmetic rather than evidence, naming the fix: draw
deeper until a second region refuses the curve.  `sweep_refusal_search` is that, at 16, 32
and 64 genomes.  The draw is a SUPERSET — `sweep_genomes` fills each orientation from the
same Latin-hypercube stream in the same order, so the first four of each are exactly the
published sixteen and nothing above moves.

**No second refusal appeared.  The curve reaches 63 of 64.**  So the experiment did not
produce what it was designed to produce — and it produced something better:

```
  gap / reached-set spread          16 genomes   32 genomes   64 genomes
    interior-angle sum                  0.573        0.257        0.041
    |angle sum - 180|                   0.614        0.502        0.070
    arc_span_deg                        0.187        0.187        0.176
```

Each larger draw finds a reached genome closer to the refusal in angle sum — 170.3, then
164.2, then 157.9, against the refusal's 156.4 — so **the quantity PART 6 ranked first
decays by a factor of fourteen as the box grows, and the one it ranked third and discounted
is the one that holds.**  At 64 genomes `arc_span_deg` is the only separator with a gap
worth more than a tenth of its spread, and no drawn genome has ever exceeded arc span
35.312 while the refusal sits at 41.209.

`bow_over_width` is now decisively out rather than merely unhelpful: a reached genome has
bow 0.540 against the refusal's 0.491, so the refusal is not even extremal on it.
`turn_at_far_end_deg`, the smallest wedge and the A/C ratio stay out.

## WHAT THIS IS AND IS NOT

**It is a demotion, not a promotion.**  Arc span survives a fourfold box; it is still one
refusal against sixty-three, and "the largest arc span in the draw refuses" remains a set of
one.  What has changed is that two of the three candidates are now known to be small-sample
artefacts and were caught being so — a separation that decays as the box grows and one that
holds are different kinds of claim, and only running the box out distinguishes them.

**And the reason a second refusal is hard to find is itself a result:** the LHC draw
produces almost nothing above 35 degrees of arc span.  Sixty-four genomes yielded exactly
one above that, and it is the refusal.  So the next experiment is not "draw more" — it is
**draw CONDITIONED on large arc span**, populating 35-45 degrees deliberately, and see
whether refusals cluster there.  That is a different sampler, not a bigger one, and it is
the first version of this question that could return a mechanism rather than a candidate.

## WHAT IS UNCHANGED

**Nothing promoted, no threshold moved, `bend` still defaults to 0.0, and every previously
committed field in `study_tri_block.json` reproduces exactly** — the artifact gains the
`refusal_search` section, a `num_points` key inside each row's `fold` block, and two
self-checks.  `make triblock` is ~445 s rather than ~290 s and the search is the difference.

---

# STEP 3 RECORD, PART 8 — 2026-08-24. CONDITION THE DRAW ON ARC SPAN AND 22 OF 40 REGIONS REFUSE THE CURVE, AGAINST 1 OF 64 UNIFORM. IT IS A RISK FACTOR WITH A RATE — AND STILL NOT A GATE

PART 7 ran the box out to 64 genomes, found no second refusal, and identified why: the
uniform Latin hypercube puts about one genome above 35 degrees of arc span in sixty-four,
so the band where the refusal lives is essentially unsampled.  It named the fix — a
different sampler, not a bigger one — and this is that.

Screening the stream on `arc_span_deg` before meshing costs nothing (the region and its
report are cheap; the control mesh is not), so the band is reachable directly:

```
  29582 drawn -> 67 above 30 degrees -> 40 of those mesh clean
  22 of the 40 REFUSE the curve at every bend and every admissible free count       55.0%
  the uniform box, for comparison                                          1 of 64   1.6%
```

**A 35x enrichment.**  So the arc span is not a coincidence of one draw: regions with a wide
arc really are far more often impossible, and PART 6's third-ranked candidate — the one it
discounted, and the only one PART 7's fourfold box did not decay — is a real risk factor
with a measured rate.

## AND IT IS STILL NOT A GATE, WHICH IS THE HALF THAT KEEPS THIS HONEST

Inside the band the two classes **overlap**:

```
  refusals   arc span 30.27 - 44.41
  reached    arc span 30.08 - 36.14
```

So the arc span predicts HOW OFTEN a region is impossible and not WHICH one is.  Neither
does anything else here: the refusals' interior-angle sums run 151.8 to 187.5 and their
bows 0.25 to 1.25, both straight across the reached ranges.  Whatever picks the individual
out of the band is still unnamed, and the band is now the right place to look for it —
forty genomes with a 55% failure rate is a far better testbed than sixteen with one.

The conditioned draw also reaches regions the uniform one never produced: bows up to 1.25
against a uniform-box maximum of 0.54.  That is worth knowing on its own — every "the box
spans X" statement in PARTS 1-7 is a statement about what the UNIFORM sampler reaches.

## WHAT IS UNCHANGED

**Nothing promoted, no threshold moved, `bend` still defaults to 0.0, and the draw the
published numbers rest on is untouched** — the band is its own stream (seed offset +1000)
so it shares no genome with the box, which is exactly what makes the two rates comparable.
`study_tri_block.json` gains the `arc_span_band` section and two self-checks; every
previously committed field reproduces exactly.  `make triblock` is ~670 s and the Makefile
says so.

## WHAT PART 8 LEAVES

1. **What picks the refusal out of the band.**  Forty genomes, 55% failing, all shape
   numbers overlapping — the first genuinely well-conditioned version of §56's question.
2. **Every "the box spans X" claim in PARTS 1-7 is about the uniform sampler**, and the
   band shows it reaches less than was assumed.  Worth a pass over those statements.

---

# STEP 3 RECORD, PART 9 — 2026-08-24. WHAT PICKS THE REFUSAL OUT OF THE BAND IS THE SMALLEST CORNER ANGLE, AT AUC 0.043 — AND A HELD-OUT DRAW CUTS THE FITTED RULE FROM 1.000 TO 0.833 AND FALSIFIES HALF OF IT

PART 8 left forty band genomes at a 55% failure rate as "the first well-conditioned version
of §56's question".  With twenty-two refusals against eighteen reached this is a
classification problem rather than a set of one, so it can be scored properly.

## THE ANSWER IS THE SMALLEST WEDGE, AND IT IS NOT CLOSE

Concordance (AUC) of each shape quantity between the 22 refusals and 18 reached:

```
  min_wedge_deg          0.043      <- a random refusal has a SMALLER minimum corner
  wedge_sum_deg          0.227         angle than a random reached region 95.7% of the time
  wedge_sum_minus_180    0.679
  arc_span_deg           0.667
  turn_at_far_end_deg    0.366
  A_over_C               0.429
  bow_over_width         0.472      <- no signal at all
```

**The smallest interior angle of the curvilinear triangle.**  Which is mechanically the
right shape: a transfinite blend on a region with a very sharp corner has to squeeze a
structured grid into it, and that is what folds.

**And it is a quantity PART 6 tested and dismissed** — in the uniform box the refusal's
19.284 sat inside the reached range [10.053, 36.971].  The reason is now visible: ten of
the sixty-three uniform reached genomes have a minimum wedge under 17 degrees and **none of
them has an arc span above 30**.  A sharp corner is harmless in a narrow region; it is only
in the wide-arc regime that it decides anything.  PART 6 could not have seen that, because
its sampler never put the two together.

## AND THEN A HOLD-OUT, WHICH IS THE PART WORTH KEEPING

A conjunctive rule fits the 104 genomes measured so far **perfectly** — `arc > 36.16 OR
(arc > 30 AND min_wedge < 17.12)`, 23 of 23 refusals caught, zero false positives, accuracy
1.000.  Both thresholds are fitted on the same observations they are scored on, and one of
them lands 0.02 degrees from the nearest counterexample, so that number is not evidence.

Scored on a fresh band drawn from a disjoint stream (seed offset +7000), thresholds frozen:

```
  30 held-out band genomes, 12 refuse (40% base rate)
    accuracy 0.833   precision 0.733   recall 0.917
    11 caught, 1 missed, 4 false positives, 14 cleared
```

Above the majority-class baseline of 0.600, and a long way below 1.000.  **And the hold-out
falsifies half the rule outright**: a region with arc span **39.97** and min wedge 20.27
was REACHED, so "a wide enough arc always refuses" — which looked like six for six — is
simply false.  The other three errors are all the corner branch firing too early
(min wedge 16.59, 16.69, 17.08 at arcs 31-32, all reached), so 17.12 is too aggressive.

## WHAT IS ESTABLISHED AND WHAT IS NOT

**Established:** the minimum wedge angle is the dominant separator inside the difficult
regime (AUC 0.043 over 40 genomes); the arc span sets how often a region is in that regime
at all (35x enrichment, PART 8); and the two are conjunctive — neither works alone, which
is why six sections of single-quantity searching found nothing.

**Not established:** any threshold.  The fitted pair scores 1.000 in sample and 0.833 out,
and its wide-arc branch has a counterexample.  What §56 asked for was a mechanism, and a
mechanism with an unvalidated threshold is where this stands.

## WHAT IS UNCHANGED

**Nothing promoted, no code changed, no artifact regenerated.**  Every number above is
computed from `study_tri_block.json`'s committed `arc_span_band` and `refusal_search`
sections except the hold-out, whose stream is named here so it can be re-run.

---

# STEP 3 RECORD, PART 10 — 2026-09-03. THE MECHANISM CALIBRATES AND ITS THRESHOLDS DO NOT, THE GENOME-DEPENDENT BEND BUYS EXACTLY NOTHING, AND THE ADOPTION QUESTION IS NO LONGER THE ONE THIS ARC HAS BEEN ASKING — SINCE §103 THE TREE BUILDS A FILLETED SECTOR AND THE Y-PARTITION IS MEASURED AGAINST THE ONE IT STOPPED BUILDING

PART 2's successor 3, PART 4's successor 3, and PART 9's successor 1, in one session.  Two
of the three are measurements and they are below.  The third is the decision this file has
deferred at the end of PART 2 and again at the end of PART 4 — *"then, and only then, the
decision to adopt the faithful rim"* — and it is taken here, against a blocker neither
deferral could have named, because it did not exist until eleven days after PART 9.

## FIRST, THE PREMISE — AND BOTH HALVES OF IT HAD MOVED

Every ranking this arc carries traces to §46 and §52: the wheel's global peak sits on
`rim:P_c`, a corner the part does not have, and the tri-block is the only measured path to
removing it.  Both halves of that were checked before any study was re-run, because a
successor whose value rests on a premise is worth exactly what the premise is worth.

**THE QUANTITY IT WAS RANKED TO PROTECT IS NO LONGER READ BY ANYTHING THAT DECIDES.**
§103 replaced `util_j`'s `Kt * agg` surrogate with a per-junction region p-norm on the
fillet's own arc.  `wheel_objective.py:1255` now reads

```
    agg, c = _stress_aggregate(pn, maxes, q)          # whole-wheel pnorm — REPORTING ONLY
```

and the `stress` wall and `stress_margin` term at `:1300-1310` loop over `agg_hub` and
`agg_rim` alone.  Checked mechanically rather than read: inside `t3_terms` the bare `agg`
is assigned at `:1255` and read at exactly ONE place — `:1354`, the report key
`pnorm_stress_agg_mpa`.  Every other occurrence of the identifier in the module is a
comment or a different function's own local (`_stress_aggregate`'s and
`_pnorm_and_grad`'s), and the `probe_p` sweep builds its own `a_v` rather than reading
this one.  **So `rim:P_c`'s divergent peak reaches no barrier, no objective term and no
gradient.**  What the tri-block would buy today is a reporting number
and the mesh's own geometric fidelity — which is worth having, and is not what "the whole
remaining path to a quotable peak" priced.

**AND THE MESH THE TREE BUILDS IS NOT THE MESH THIS FILE PARTITIONS.**  §103 also made
`fillet=True` unconditional in `wheel_objective.phase_meshes` and
`wheel_pool_worker.run_phase`: every mesh the objective solves on is the ELEVEN-block
filleted sector.  `study_tri_block.region()` (`studies/study_tri_block.py:228`) calls
`WW.sector_blocks(genes, cfg, uncap=(True, blend), orientation=orientation)` with **no
`fillet=`** — the unfilleted seven-block one.  Re-measured at the shipped genome on this
tree, `MIN_SJ_TARGET` = 0.2:

```
  faithful rim (uncap blend 0.0)        coarse      medium
    UNFILLETED sector, worst block     0.008176    0.008251
    FILLETED sector, worst block       0.000343    0.003334
  filleted sector at the SHIPPED uncap 0.359414    0.362312
```

The filleted re-cut makes the faithful rim **worse**, by 23.8x at `coarse` and 2.5x at
`medium` — `tests/test_fillet_block.py::test_the_recut_does_NOT_rescue_the_faithful_rim`
has asserted that ordering since PART 10 of the fillet arc and it still holds.  The
degenerate vertex survives the fillet (the fillet rounds `P_t`; the ~180-degree corner is at
`far_end`), so the region is still a curvilinear triangle and still needs partitioning —
**but the twelve-block Y this file built is a partition of the seven-block sector, and
there is no version of it for the eleven-block one.**

## PART 9's SUCCESSOR 1 — THE THRESHOLDS, CALIBRATED.  THE MECHANISM SURVIVES; NO TRIPLE DOES

`make trirule`, 958.8 s.  Driver `studies/study_tri_rule.py`, artifact
`studies/study_tri_rule.json`, 20 tests in `tests/test_tri_rule.py`.

The apparatus is the one parked on `origin/tri-rule-holdout` at `8f0a725` since
2026-08-24, cherry-picked and **re-aimed**, which is what its own commit message said it
needed: it was shaken down on `best_w_valid` — the STRAIGHT Y over `sweep_genomes`' uniform
draw, §53's question and closed by §55 — and PART 9's subject is `curved_valid` over
`sweep_arc_span_band`'s draw, conditioned on arc span > 30 deg.  The two enabling changes
were already in that commit: a `seed_base` kwarg, and a `genes` field on every band row so
disjointness is asserted from the genes rather than assumed from the seeds.

Four fresh band streams, ~200 s each, seed bases 1000 apart (a band draw consumes batch `b`
from `seed_base + b` and may use up to 400 of them, so nearer bases would share candidate
batches verbatim).  Fit, freeze, score the hold-out once, then swap:

```
  config  role      seed base   drawn   in band   meshed   folds
  coarse  fit       20262000    25290      57       40      18/40
  coarse  hold-out  20263000    32047      62       40      15/40
  medium  fit       20264000    35309      70       40      17/40
  medium  hold-out  20265000    30895      66       40      16/40
```

```
  config  direction  frozen rule                                        in-sample  HELD OUT  majority
  coarse  forward    arc > 38.444 OR (arc > 30.556 AND wedge < 17.352)     0.975     0.875     0.625
  coarse  swapped    arc > 37.779 OR (arc > 30.391 AND wedge < 16.310)     0.975     0.900     0.550
  medium  forward    arc > 36.569 OR (arc > 30.537 AND wedge < 14.826)     0.975     0.800     0.600
  medium  swapped    arc > 41.813 OR (arc > 33.272 AND wedge < 18.664)     0.925     0.900     0.575
```

**THE MECHANISM IS CONFIRMED AND IT IS NOT A COINCIDENCE OF PART 9's DRAW.**  Every
direction beats its own majority baseline by 20 to 35 points, on genomes the fit never
saw, at four independent draws PART 9 did not have.  The conjunction of a wide arc with a
tight wedge is real.

**AND NO THRESHOLD SURVIVES.**  In sample every fit lands 0.925-0.975 and each lands on a
DIFFERENT triple.  Between the two directions the spread is 0.67 deg on `t_wide` and 1.04
on `t_wedge` at `coarse` — tolerable — and **5.24, 2.74 and 3.84 deg at `medium`**, which
is not.  Worse for the calibrated rule's dignity: **at `medium` forward it is BEATEN on its
own hold-out by the hand-read informal rule it exists to replace, 0.800 against 0.850**,
and at `coarse` forward the two tie exactly at 0.875.  PART 9 declined to publish
`36.16 / 30 / 17.12` because both numbers were fitted on the data they were scored on.
A proper protocol has now been run and it does not hand back a better triple; it hands back
the same family with its thresholds visibly stream-dependent.  **The screen is real, its
constants are not, and PART 9's refusal to publish them is upheld rather than repaired.**

The five hold-out errors at `coarse` forward are all one shape — four conjunctive false
fires at wedges 16.84-17.33 and arcs 30.8-31.8, plus one wide-branch false fire at arc
39.17 / wedge 20.09 — i.e. the same "the corner branch fires too early" and "a wide enough
arc does not always refuse" that PART 9 found, reproduced on fresh genomes.  Its
counterexample class was not a fluke either.

Scored on PART 9's OWN committed band (seed base `GENOME_SWEEP_SEED + 1000`), labelled
outside the protocol: the `coarse` forward rule reaches 0.900 over n = 40.

## PART 4's SUCCESSOR 2 — A BEND THAT IS A FUNCTION OF THE GENOME.  IT BUYS EXACTLY NOTHING

`make tribend`, 23.9 s.  Driver `studies/study_tri_bend.py`, artifact
`studies/study_tri_bend.json`, 15 tests in `tests/test_tri_bend.py`.

PART 4 named `bow_over_width` as "the obvious argument to fit against" and warned in the
same breath that the per-genome optima "are argmaxes over a plateau and their scatter is
partly that."  So nothing here is fitted against those argmaxes.  Two ONE-parameter
families are fit against what actually matters — how many genomes a rule leaves valid and
clear — under the same fit/freeze/score-once discipline, at `w` held fixed to this file's
own published cell, on fresh disjoint 16-genome uniform boxes:

```
  CONSTANT   bend(bow) = b,                      b in BEND_GRID (0.0 ... 1.0)
  LINEAR     bend(bow) = clip(k * bow, 0, 1),    k in (0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30)
```

The linear family is zero at zero bow by construction, which is
`test_the_bend_is_INERT_where_the_region_is_fat` made structural instead of measured.

```
  config  family    fit param   in-sample clear/valid   HELD OUT clear/valid   worst min SJ
  coarse  constant    0.0            14 / 16                 9 / 13             -0.9950
  coarse  linear      0.0 (k)        14 / 16                 9 / 13             -0.9950
  medium  constant    0.4             9 / 10                 9 / 12             -0.1930
  medium  linear      3.0 (k)         9 / 10                 9 / 12             -0.1912
```

**Nine against nine at both configs.**  The genome-dependent family does not reach one more
genome than the number it was built to beat, on data neither had seen.  And the swap says
the same thing louder: fitting on the hold-out draw picks `0.0` for BOTH families at BOTH
configs — a freshly drawn uniform box does not want a bend at all.

**AT `coarse` THE CONSTANT FAMILY ITSELF PICKS ZERO**, which is the honest version of PART
4's own `coarse` column (0 rescued there).  The curve's two rescues live at `medium`, on the
0.498-bow and 0.264-bow genomes, and a rule fit on sixteen fresh genomes does not
rediscover them.

**ONE THING THE NULL RESULT DOES BUY, AND IT IS SMALL.**  At `medium` the linear rule
reaches the same held-out coverage while asking the shipped genome for `bend` = **0.0291**
instead of the constant's **0.40** — 0.5816 against 0.5820, i.e. the published straight-Y
number to four figures.  Genome-dependence does not reach more genomes; it reaches the same
ones without imposing a bend on a genome whose region is fat.  That is a property worth
recording and it is not a reason to adopt anything.

## THE DECISION: THE FAITHFUL RIM IS NOT ADOPTED, AND THE REASON IS NEW

Three sub-decisions, each stated with its own reason.

**THE MESH — DECLINED.**  `UNCAP_DEFAULT` keeps its `(True, 1.0)` rim entry and
`sector_blocks` keeps its seven blocks.  Not on the ground this arc has tracked since PART
2 — that the interior-point rule folds on a quarter of the gene box, which is still true
(fixed-w 12/16 and 10/16, ceiling 15/16 and 12/16, and the 41.2-degree genome refuses at
every interior point, every bend and every admissible `B` at both configs) — but on the
harder ground above: **the construction is a partition of a sector the tree stopped
building on 2026-09-03**, and on the sector it does build the faithful rim is worse than the
one this partition was measured against.  A rule that held across the whole gene box would
not change that.

**THE QUOTED CELL — DECLINED, AND THE REASON HAS CHANGED FROM PART 3's.**  PART 3 measured
a genome-robust interior point, left it unadopted, and said in terms that *"the choice of
which number this arc quotes going forward is made by whoever next picks it up."*  This is
that pickup, and it declines — not because the cell is worse (at `coarse` it costs the
shipped genome nothing, 0.6262 either way, while fixing two more drawn genomes) but because
the number it would change describes an un-adoptable construction.  One stated rule — the
argmax over the published grid at the shipped genome — has meant one thing across seven
sections of `PLAN.md`; substituting a different argmax inside a file whose construction is
now blocked for an unrelated reason spends that consistency and buys a better figure for a
mesh nobody will build.  **PART 3's "there is no reason not to prefer it" was correct when
written and is superseded by the premise, not by the measurement.**

**THE CURVE — CONFIRMED AT 0.0, NOW BY MEASUREMENT.**  `bend` keeps its 0.0 default.  PART
4 left it there as an unadopted measurement; this section makes it the fitted answer at
`coarse` and a tie at `medium`.

## WHAT IS UNCHANGED

**Nothing promoted, `best_solution.json` untouched, no threshold moved anywhere in the
tree, `UNCAP_DEFAULT` still `(True, 1.0)`, `sector_blocks` still seven blocks, `bend` still
0.0, and the default mesh bit-identical.**  Pinned by
`test_nothing_here_is_wired_into_the_mesh_the_tree_BUILDS` in all three tri test files.

`studies/study_tri_block.json` IS regenerated — 665.3 s, 19 of 19 self-checks — because the
cherry-picked commit adds a `genes` field to every `arc_span_band` row and the committed
artifact therefore stopped reproducing from its own driver the moment the pick landed.  The
diff was checked field by field rather than eyeballed: **41 differences, of which one is
`seconds` and forty are the `genes` additions on the forty band rows.  Nothing else moved.**

## WHAT PART 10 LEAVES

1. **A Y-PARTITION OF THE FILLETED RIM JUNCTION, OR A DECISION THAT `rim:P_c`'s FIDELITY IS
   NOT WORTH ONE.**  This replaces "a rule for the interior point that holds across the gene
   box" as the binding constraint.  The rule is still imperfect and no longer decides
   anything: the eleven-block sector needs its own partition before any rule on the
   seven-block one can ship.  And the honest form of the alternative is on the table —
   since §103 the peak `rim:P_c` carries feeds no barrier and no gradient, so the cost of
   never fixing it is a reporting number and a fidelity claim, not a wrong answer.
2. **THE SCREEN IS USABLE EVEN THOUGH ITS CONSTANTS ARE NOT.**  A band draw costs ~200 s
   because 25-35k genomes are drawn to find forty above 30 degrees.  A fold screen at
   0.875-0.900 held-out accuracy is a real instrument for conditioning such a draw, and it
   is now measured rather than hand-read — provided it is refit per stream and never quoted
   as a constant.
3. **WHAT MAKES THE 41.2-DEGREE REGION IMPOSSIBLE IS STILL UNNAMED**, and it is now the only
   part of PART 6's question still open: the bow explains where the bend is needed, the
   wedge-and-arc conjunction explains which regions fold, and neither separates the genome
   that refuses everything from the 0.498-bow one the curve reaches.

---

# THE PARK — 2026-09-05. **`rim:P_c`'s FIDELITY IS NOT WORTH A Y-PARTITION. PART 10's ITEM 1 RESOLVES TO ITS SECOND BRANCH, AND THE ARC IS PARKED WITHOUT ADOPTION.**

PLAN.md §114. PART 10 (§104, 2026-09-03) left the arc holding one binding choice: *"a
Y-partition of the FILLETED rim junction, or a decision that `rim:P_c`'s fidelity is not worth
one."* Nobody has taken it in the fifteen sections since. **This takes the second branch.**

## THE PREMISE, RE-CHECKED MECHANICALLY RATHER THAN READ OFF PART 10

Both legs were re-verified against the tree as it stands today, not quoted from the record
that established them:

1. **The mesh this file partitions is not the mesh the tree builds.**
   `wheel_wheel.sector_blocks` takes `fillet=None` by default and its own docstring reads
   *"The seven node grids of sector 0 — eleven when the fillet is blocked"*
   (`wheel_wheel.py:2282-2286`). `study_tri_block.region()` calls it with **no `fillet=`**
   (`studies/study_tri_block.py:228`) — the seven-block sector. Every mesh the objective
   actually solves passes `fillet=True`: `wheel_objective.py:1015` and
   `wheel_pool_worker.py:63`, unconditional since §103. **The twelve-block Y this file built
   partitions a sector the tree stopped building**, and PART 10 already measured the
   inversion: the faithful rim's worst block goes 23.8x worse at `coarse` and 2.5x at
   `medium` once filleted.

2. **`rim:P_c` reaches nothing that decides anything.** `grep -rn "rim:P_c" src/` returns
   **zero matches** — the string survives only in `tests/`, in study drivers and in committed
   study artifacts. The quantity it contaminates is assigned once and read once:
   `agg, c = _stress_aggregate(pn, maxes, q)` at `wheel_objective.py:1257`, carrying the
   comment *"whole-wheel pnorm — REPORTING ONLY"*, and read at `:1356` into
   `"pnorm_stress_agg_mpa"`, plus the `stress_utilisation` diagnostic that `:1333` records as
   *"retained as the diagnostic that shows why it was abandoned"*. Neither `BARRIER_TERMS`
   nor `OBJECTIVE_TERMS` (`:399-401`) names it. What prices stress on the live path is the
   two per-junction region p-norms, `agg_hub`/`agg_rim`.

## THE REASON, AND IT IS NOT THE ONE THIS ARC CARRIED

This arc was created because **the wheel's global peak stress lived on a corner the shipped
part does not have**. That was true and it was worth chasing. It is not why the arc stops.

The arc stops because the payoff it was ranked for has been disconnected from every consumer.
Step 3.1 asks whether the global peak leaves the artefact corner — but since §103 that peak is
a **reporting number**, so the answer changes a diagnostic and no decision. Step 3.2 predicts
that removing the end cap makes the fillet tractable — **inverted**: on the mesh the tree now
builds, the faithful rim is 23.8x worse, not better. Step 3.3 says *"only then is FILLET_PLAN
Step 2 reachable"* — and FILLET Step 2 landed independently at §103 without this arc. **All
three steps are answered, unreachable, or moot, and none of them by being done.**

So the cost of never partitioning the eleven-block junction is a fidelity claim about a
reporting number. That is not worth the measured price of the alternative: PART 10 put a band
draw at ~200 s because 25-35k genomes must be drawn to find forty above 30 degrees, before any
partition work begins at all.

## WHAT WOULD REOPEN IT

**One thing, and it is not a refinement of this arc's own argument: `rim:P_c` acquiring a
consumer.** If a future term, barrier or gradient reads the whole-wheel p-norm — or if
`stress_utilisation` is ever promoted back out of diagnostic status — the fidelity of the
corner it is computed on stops being cosmetic and this arc becomes live again, on the
eleven-block sector. The park is a statement about what reads the quantity, not about whether
the corner is real. **The corner is still manufactured and the mesh still puts the global peak
on it; nothing downstream cares.**

The fold screen (PART 10's item 2) is **not parked with the arc** — it is a usable instrument
for conditioning a band draw, refit per stream, and it outlives this file.

## SCOPE, WITH ITS DIRECTION

This parks the arc against the objective **as §103 left it**. The direction of the error is
worth naming: if the Stage-3 re-run (§113) or anything after it re-wires the whole-wheel
p-norm into a live term, this decision is wrong in the direction of having under-invested in
mesh fidelity — recoverable, because the seven-block partition work in this file is kept, not
deleted. It is wrong in no direction that costs a wrong answer today, because today the
quantity is read by a report key.

**WHAT DID NOT HAPPEN.** `best_solution.json` is untouched, no threshold moved, no mesh
default changed, and no code was written for this decision. The seven-block Y-partition, the
calibrated thresholds and the fold screen all stay on file exactly as PART 10 left them. What
moved is this file's header, its row in `PLAN.md`'s open-arcs table — which it did not have
until now — and this record.
