# UNCAP_PLAN.md — the mesh's second junction corner is manufactured, and it carries the peak stress

**Open arc #2, promoted to the top. Created 2026-08-17 from PLAN §34 Findings 1 and 4.
Nothing started. NOT CHEAP — read the cost section, and read the premise section first
because it is unusually load-bearing.**

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
