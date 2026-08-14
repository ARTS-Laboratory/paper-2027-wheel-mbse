# CONTACT_PLAN.md — is the contact model the objective ACTUALLY uses still describing the wheel it just chose?

> **Ignore version control entirely. Do not commit, branch, stage, revert or otherwise
> touch git — it is not part of this project's workflow and nothing here depends on it.**

> **THIS FILE IS WRITTEN TO BE READ ONE STEP AT A TIME, BY A SESSION THAT HAS NEVER SEEN
> THE OTHERS.** Every step carries **Why / Read first / Do / Gate / Record**. Do the step,
> write its numbers into its own **Record** block, tick the status table, stop.

**The parent item is PLAN.md §19's ranked successor #1.** Read §19's "Group A" subsection
before Step 1; everything below assumes it. This file is the milestone; PLAN.md gets a §20
at Step 5.

---

## Status

| step | what | status |
|---|---|---|
| 0 | baseline `make test`, enumerate the eleven reds | **DONE — 11 failed / 433 passed, exactly §19's count** (2026-08-13) |
| 1 | falsify the premise: does the assumed 3.0° patch reach the objective? | **DONE — REFUTED. IT CANNOT EVEN BE PASSED TO IT** (2026-08-13). §19's Group A harm claim is retracted; see Record |
| 2 | is the real-contact axle drop still mesh-convergent on `e126cc3`? | **DONE — ALL THREE GATE CLAUSES PASS, AND THE PROMOTED GENOME CONVERGES 2x BETTER THAN THE ONE IT REPLACED** (2026-08-13). The mechanism behind §19's Group A is a **16.8x element-size step on the rim OD** that the patch straddles |
| 3 | triage the reds this arc owns | **DONE — 7 failed / 438 passed, exactly 11 − 4 reds, NO NEW RED** (2026-08-13). Four reds, four different kinds of wrong; one test added |
| 4 | THE DECISION | **DONE — BRANCH A. Successor #1 is RETIRED, not deferred: the harm it claimed does not exist** (2026-08-13). Defect 5 becomes #1 |
| 5 | write the record into PLAN.md as §20 | **DONE — §20 WRITTEN, §19 AMENDED IN PLACE, BUILD_PLAN SUCCESSORS SUPERSEDED** (2026-08-13) |

**The two genomes, resolved by hash so no step can be misattributed.** `wheel_genome.genome_hash`
over each file's `genes` block:

| role | file | hash |
|---|---|---|
| shipped | `best_solution.json` ≡ `stage3_margin_best_medium.json` | `e126cc3` |
| control (predecessor) | `stage3_buildcap2_slack_medium.json` | `e4219f3` |

---

## The problem, in one page

PLAN.md §19 promoted `e4219f3` → `e126cc3` and ranked six successors. It put this first:

> 1. **The contact patch model.** Group A. This is the only red gate in the tree that indicts
>    the wheel currently in `best_solution.json`, it biases the quantity the objective is
>    steering by 5.27%, and it gets worse the harder the optimizer works.

That ranking rests on one sentence: *"The objective computes deflection against an assumed
3.0° patch."* **The call chain says otherwise.** Stage 3's deflection is `axle_drop` out of
`wheel_adjoint.service_qoi_value_and_grad` (`src/wheel_adjoint.py:606`), and every solve on
that path is the **real penalty contact** problem:

```
wheel_objective.t3_terms                       src/wheel_objective.py:947
  -> WA.service_qoi_value_and_grad                  src/wheel_adjoint.py:642,648,653
       -> fem.solve_wheel_contact
       -> fem.solve_wheel_contact_at
       -> fem.wheel_contact_problem                 src/wheel_fem.py:1691
```

`CONTACT_PATCH_HALF_DEG = 3.0` (`src/wheel_fem.py:1444`) is consumed by `wheel_problem` /
`solve_wheel` — the M4 pressure-driven model — and by nothing else. Its live callers are
`studies/study_gnl.py`, `studies/study_wheel_fea.py`, `tests/test_gnl.py`,
`tests/test_wheel_fea.py` and one comparison test in `tests/test_contact.py`. Nothing in
`src/wheel_objective.py`, `src/wheel_adjoint.py` or `src/wheel_fea.py` reaches it.

**This is §17's lesson presenting itself again** — a successor ranked #1 on a mechanism that
turns out not to bind — so Step 1 is a falsification, not construction, and it costs minutes.

### The concern that IS live, and it is a different one

`src/wheel_objective.py:107-114` already records that the objective's **own** real-contact
patch is under-resolved:

> M7 found the contact facets on the rim discretisation (3.8% of the slope at `coarse`, 17.6%
> at `smoke`), an artefact that refines away and exists because **no config resolves the 0.484
> degree contact patch — `coarse` gets 1.75 nodes across it** against the 8 the master plan
> assumed. It lives at the 0.25 degree quadrature spacing while the stencil samples at 3.75
> degrees, so it ALIASES.

§19 then measured that `t3` +20% and `R_rim` at its box maximum **halved the real patch** —
0.4963° → 0.2965° half-angle at `smoke`, below the 0.3082° Hertz bound — and that a rim node
now **penetrates** by 0.247 µm, which trips the premise `tests/test_contact.py:243` exists to
guard ("the mesh has become fine enough that the contact is no longer interior to a segment,
so this test's premise… needs updating").

So the question that can actually indict the shipped wheel is not *assumed vs. real*. It is:
**on `e126cc3`, is the real-contact axle drop the objective steers by still mesh-convergent,
or did the promotion walk the design into the under-resolved corner of its own contact
model?** Nobody has measured it, and it needs forward solves and no optimizer.

### Standing rules, inherited and non-negotiable

- **Measure before touching a threshold.** PLAN.md §14's governing rule. A red gate is the
  finding; do not loosen it to fit.
- **Measure both genomes.** Anything claimed about `e126cc3` gets the same measurement on
  `e4219f3`. That comparison killed three of §14's eight readings and demoted §16's #1
  successor in §17.
- **Linear stays the default everywhere.** Every committed artifact must still reproduce with
  no flag passed. A new default is a re-baselining, and this repo does not re-baseline
  silently.
- **A committed study artifact is not a control.** `studies/study_gradient.json` was three
  days stale and cost SVK_PLAN Step 1 an hour. Controls are measured in-session.
- **One arc, one mechanism.** Reds outside this arc's mechanism stay red with their cause
  recorded.

---

## Step 0 — Baseline the gate, and enumerate the eleven reds by node id

**Why.** §19 reports `11 failed / 433 passed`, but the tree carries the promotion's
uncommitted edits and §19 names only four of the eleven. Step 3 is read as a diff against
this list, so it has to be measured rather than inferred.

**Read first.** PLAN.md §19, "The gate: `11 failed, 433 passed`".

**Do.**

```
make test 2>&1 | tee /tmp/contact_step0_test.log
grep -E "^(FAILED|ERROR)" /tmp/contact_step0_test.log
```

~23 min. *(Aside, already on file in SVK_PLAN Step 0: `make test 2>&1 | tee` reports `tee`'s
exit status, so read the summary line, not `$?`.)*

**Gate.** Exactly **11 failed / 433 passed**. A twelfth failure is a finding and stops the arc
— it means something moved that nobody recorded, and every measurement below would inherit it.

**Record.**

```
STEP 0 RECORD — RUN 2026-08-13.  GATE: PASS.
  make test:            11 failed / 433 passed in 1465.16 s (24:25)
  matches §19's 11/433: YES, exactly
  box:                  24 cores / 61 GB
```

The eleven, tagged to §19's groups. §19 named four of them; the other seven are named here
for the first time, and **the third Group A member was not the one §19's prose implied**.

| # | node id | group |
|---|---|---|
| 1 | `test_contact.py::test_the_centre_node_rise_is_not_the_axle_drop` | **A** |
| 2 | `test_contact.py::test_the_real_patch_is_far_smaller_than_the_assumed_one` | **A** |
| 3 | `test_contact.py::test_but_the_assumed_patch_got_the_axle_drop_nearly_right` | **A** |
| 4 | `test_objective.py::test_the_margin_weight_is_the_exchange_rate_it_claims_to_be` | **B** — defect 8 |
| 5 | `test_export_contract.py::test_the_bite_is_the_volume_divided_by_the_right_thickness` | **C** |
| 6 | `test_wheel_fea.py::test_only_the_rim_od_near_the_bottom_is_loaded` | **C** |
| 7 | `test_gnl.py::test_the_correction_enters_at_first_order_in_the_load` | inherited (§14, deliberate) |
| 8 | `test_wheel_fea.py::test_the_rim_band_holds_a_large_minority_of_the_compliance` | inherited (§14, deliberate) |
| 9 | `test_gnl.py::test_the_correction_is_not_a_constant_over_the_design_space` | inherited |
| 10 | `test_wheel_fea.py::test_the_beam_to_wheel_ratio_is_not_a_constant` | inherited |
| 11 | `test_wheel_fea.py::test_a_thicker_rim_monotonically_stiffens_the_wheel` | inherited |

3 + 1 + 2 + 5 = 11, which is §19's split ("six new, five inherited") arriving at the same
arithmetic from the node ids rather than from the count. Two of the five inherited are
SVK_PLAN Step 0's deliberate pair, red since §14 — §19 called all five "§16's", which is loose;
they are recorded correctly here.

**The three Group A failure texts, because Step 2 reproduces all three independently.** All at
`smoke`, the tier `tests/test_contact.py` pins (`CFG = "smoke"`):

| # | the assertion that fired | measured |
|---|---|---|
| 1 | `node_gap.min() > 0.0` | **−2.465046677855298e-04 mm** |
| 2 | `patch_half_deg > hertz_patch_half_angle_deg()` | **0.2965060836013539 > 0.3082244481571778** — false |
| 3 | `rel < 0.05` (real contact vs assumed patch) | **0.052721891442461866** |

Row 2 is the one §19's prose did not name. Its docstring expects the real patch to sit *above*
the Hertz solid-cylinder bound — M4 called that bound a lower bound and "expected to be
exceeded by far" — and on the promoted genome it has fallen **below** it, by 3.8%. That is a
statement about the wheel and its mesh, not about the assumed patch, so **it is the only Group
A red that Step 1 cannot dispose of** and it belongs to Step 2's question rather than Step 3's.

---

## Step 1 — Falsify the premise: does the assumed 3.0° patch reach the objective at all?

**Why.** The entire ranking of successor #1 turns on this, and it costs minutes. Code reading
says no — but inference from reading is exactly what `stress_scale` was justified with for two
milestones.

**Read first.** `src/wheel_adjoint.py:606-670`, `src/wheel_fem.py:1493-1532`
(`wheel_problem`) and `:1691-1720` (`wheel_contact_problem`), `tests/test_contact.py:288`.

**Do.** Three discriminators at `e126cc3`, `coarse`, one phase, under **both** kinematics:

1. **Identity.** The objective's `report["axle_drop_mean_mm"]` against
   `fem.solve_wheel_contact(mesh)["axle_drop_mm"]` on the same mesh.
2. **Non-identity.** The same against `fem.solve_wheel(mesh)["axle_drop_mm"]` — the assumed
   patch. Without this leg, "the numbers match" has no power; this is what establishes that
   the comparison could have seen a difference.
3. **Sentinel.** Thread `patch_half_deg` through the objective's `**problem_kw`. Inert or
   refused; either is a proof, and a refusal is the stronger of the two.

**Gate.** Binary and pre-registered: the objective's deflection is a real-contact quantity, or
it is not. No threshold to move.

**Record.**

```
STEP 1 RECORD — DONE (2026-08-13).  THE PREMISE IS REFUTED, BY REFUSAL.
  driver: scratchpad `patch_reaches_objective.py` (no gate, no artifact — the same status
  `eps_n_check.py` and `h2_check.py` have).  genome e126cc3, coarse, phase 0.0, the five
  pinned env vars exported to exactly Makefile:29-34.

  |                                   |     linear      |       svk       |
  | 1 objective axle_drop_mean_mm     | 1.5892237160122678 | 1.9132197613988573 |
  |   solve_wheel_contact axle_drop_mm| 1.5892237160122678 | 1.9132197613988573 |
  |   BIT-IDENTICAL                   |      True       |      True       |
  | 2 solve_wheel (ASSUMED 3.0 deg)   | 1.7025307168359758 | 2.0519394047655464 |
  |   rel, objective vs assumed       |     6.655e-02   |     6.760e-02   |
  | 3 patch_half_deg=1.0 and 6.0      | TypeError: wheel_contact_problem() got an
  |   through the objective           | unexpected keyword argument 'patch_half_deg'

  THE SENTINEL IS THE RESULT.  The assumed patch does not merely fail to influence the
  objective — it CANNOT BE PASSED TO IT.  `wheel_contact_problem` has no such parameter,
  so the constant is structurally unreachable from the Stage-3 loss.  That is a stronger
  statement than bit-identity, which is what the leg was written to accept.
```

**§19's Group A harm claim is RETRACTED.** §19 wrote: *"The promoted wheel's true deflection
is therefore near 1.90 mm against a 2.0 mm target, not the 2.008 mm the objective reports."*
The comparison is the other way round — **2.008 mm is the real-contact number**, and it is the
assumed-patch model that sits ~6.7% away from it. Nothing in the loss is biased by the 3.0°
constant, and "it gets worse the harder the optimizer works" does not describe a quantity the
optimizer can see.

**§19's measurements are not wrong; only the model they indict is.** The divergence is real,
it is larger at `coarse` than the 5.27% §19 measured at `smoke`, and it grew with the
promotion. What it says is that **the legacy assumed-patch model no longer stands in for
contact on this design** — which retires M4's and M5's standing as descriptions of the current
wheel, and puts a live caveat on `studies/study_gnl.py` and `studies/study_wheel_fea.py`,
both of which still solve `fem.solve_wheel`. That is a real finding and a much cheaper one
than the mesh study §19's ranking implied.

**What Step 1 does NOT establish**, stated before anything is built on it: it says the 3.0°
constant is unreachable, not that the real-contact quantity is trustworthy. Those are
different claims and Step 2 measures the second one.

---

## Step 2 — The live question: is the real-contact axle drop still mesh-convergent on `e126cc3`?

**Why.** This is what §19 should have ranked first, and it is the only thing in this area that
can indict the shipped wheel. `src/wheel_objective.py:110` already says `coarse` resolves the
patch with 1.75 nodes; the promotion halved the patch. If the drop the loss minimises moves
with the mesh on the new design, the optimizer is steering by an aliased quantity — and unlike
the assumed-patch comparison, nothing downstream corrects it.

**Read first.** `src/wheel_objective.py:95-115`; `studies/study_contact.py`, in particular
`run_emergent_patch` (`:259`), `run_kinematics` (`:428`) and `run_design_space` (`:360`),
which already compute nearly every column this step needs; `src/wheel_fem.py:749`
(`patch_extent`, the quadrature-independent measure) and `:1776` (`_attach_contact_report`).

**Do.** **Extend `studies/study_contact.py` rather than writing a driver.** It already takes
`--genome`, `--config` and `--out` (`:728-732`).

1. Add `--kinematics {linear,svk}`, **default `linear`**, threaded as a plain keyword argument
   into every section that builds a problem — not module state, which is how a run gets
   misattributed. Copy SVK_PLAN Step 1's change to `studies/study_gradient.py`, including
   recording the kinematics in the report's `settings`.
2. Widen `run_emergent_patch`'s rows with the three quantities this step needs and §19
   measured by hand: worst rim-node gap (does a node penetrate), nodes and quadrature points
   across the patch, and `patch_half_deg / hertz_patch_half_angle_deg()`.

Then the matrix — forward solves only, no optimizer:

| genome | kinematics | configs |
|---|---|---|
| `e126cc3` (shipped) | svk **and** linear | smoke, coarse, medium |
| `e4219f3` (control) | svk **and** linear | smoke, coarse, medium |

`--out study_contact_<hash>_<kin>.json`; the committed `studies/study_contact.json` stays
untouched, and **is not a control** — it is stale (see the standing rules).

**Gate.** Pre-registered before the run, on `e126cc3` under **svk**, the kinematics the design
was chosen under:

- `axle_drop_mm`, coarse → medium relative change **< `GATE_MESH_REL` = 0.05**, with a
  Richardson/GCI estimate reported beside it.
- `n_quad_points_in_contact` at `medium` **≥ 2** — the patch is resolved by more than one
  sample.
- worst rim-node gap **> 0** at `coarse` and `medium`, or the penetration reported in mm with
  the segment span beside it.

A red here is the finding and is not fixed by moving the gate.

**Record.**

```
STEP 2 RECORD — DONE (2026-08-13).  GATE: PASS, ALL THREE CLAUSES.
  `make contact` x 4 cells, sequential, systemd-run --user -p MemoryMax=24G --collect.
  patch section only, smoke/coarse/medium, n_quad=6 (the default, and the path Stage 3
  runs).  149-216 s per cell; peak RSS 1.11 GiB, nowhere near the cap.

  THE GATE, on e126cc3 under SVK — the genome and the kinematics that ship:
    coarse -> medium relative change   0.897%   [< GATE_MESH_REL = 0.05]   PASS, 5.6x
    n_quad_points_in_contact @ medium  3        [>= 2]                     PASS
    worst rim-node gap @ coarse/med    +2.148e-04 / +1.567e-04 mm  [> 0]   PASS
```

**The four cells.** `n_quad=6`; `nodes`/`quad` are rim nodes and Gauss points inside the
patch; `gap` is the worst signed rim-node gap, negative meaning a node has sunk in.

| genome | kin | cfg | drop mm | vs prev | half° | /Hertz | nodes | quad | gap mm |
|---|---|---|---|---|---|---|---|---|---|
| `e126cc3` | lin | smoke | 1.534718 | — | 0.2965 | 0.962 | 1 | 3 | −2.465e−04 |
| | | coarse | 1.589224 | +3.552% | 0.3700 | 1.200 | 2 | 7 | −2.602e−04 |
| | | medium | 1.603610 | **+0.905%** | 0.4097 | 1.329 | 3 | 11 | −1.725e−04 |
| `e126cc3` | **svk** | smoke | 1.846588 | — | 0.4020 | 1.304 | 0 | 1 | +5.668e−04 |
| | | coarse | 1.913220 | +3.608% | 0.3353 | 1.088 | 0 | 2 | +2.148e−04 |
| | | medium | 1.930376 | **+0.897%** | 0.3610 | 1.171 | 0 | 3 | +1.567e−04 |
| `e4219f3` | lin | smoke | 1.354943 | — | 0.4963 | 1.610 | 0 | 1 | +1.115e−03 |
| | | coarse | 1.421990 | +4.948% | 0.3335 | 1.082 | 0 | 2 | +4.446e−04 |
| | | medium | 1.444874 | **+1.609%** | 0.3510 | 1.139 | 0 | 2 | +3.895e−04 |
| `e4219f3` | **svk** | smoke | 1.751404 | — | 0.5301 | 1.720 | 0 | 2 | +1.278e−03 |
| | | coarse | 1.857884 | +6.080% | 0.3991 | 1.295 | 0 | 2 | +1.921e−03 |
| | | medium | 1.888894 | **+1.669%** | 0.3744 | 1.215 | 1 | 2 | −2.472e−04 |

**The control reproduces §19's Group A table on BOTH genomes, to the digit** — `e126cc3`
smoke: patch 0.2965°, /Hertz 0.962, gap −0.247 µm; `e4219f3` smoke: /Hertz 1.610, gap
+1.115 µm. §19 measured those by hand; this driver recomputes them from the solve. That is
what licenses the rest of the table.

#### 1. The quantity the objective steers by IS mesh-convergent, and it converges BETTER than before

| genome | kin | coarse→medium | change ratio | Richardson limit | finest off it |
|---|---|---|---|---|---|
| **`e126cc3`** | **svk** | **+0.897%** | 3.884 | 1.936326 mm | **0.307%** |
| `e4219f3` | svk | +1.669% | 3.434 | 1.901637 mm | 0.670% |
| `e126cc3` | lin | +0.905% | 3.789 | 1.608769 mm | **0.321%** |
| `e4219f3` | lin | +1.609% | 2.930 | 1.456732 mm | 0.814% |

**The promoted genome converges better than the one it replaced, under both kinematics, by
a factor of 2.2 to 2.5.** §19 ranked this successor first on the words *"it gets worse the
harder the optimizer works."* Measured against the control that sentence describes is
reversed: the harder the optimizer worked, the better resolved the answer got. It holds on
the second measure too — the rim OD's element-size step is **16.8× on the promoted wheel
against 30.3× on its predecessor**, because a 20% thicker rim widens the weld footprint. This
is the second time in this arc that the standing "measure both genomes" rule has overturned a
reading, and the third time in the project (§14 ×3, §17, here).

#### 2. THE MECHANISM BEHIND ALL OF GROUP A: a 16.8x element-size step, and the patch sits on it

The rim OD is not uniformly divided. Per 30° sector it gets `n_weld + n_rim_free` segments,
and **the two families are not the same size**. Measured, exactly, at `coarse`:

```
  n_weld     = 10 segments x 0.1682 deg  =  1.682 deg of weld arc
  n_rim_free = 10 segments x 2.8318 deg  = 28.318 deg of free arc
                                   total = 30.000 deg  = SECTOR_DEG
```

The span ratio is **constant across the mesh ladder and a function of the DESIGN**: 16.8× on
`e126cc3` at all three rungs, **30.3×** on `e4219f3` at all three. It tracks the weld
footprint — the promoted wheel's rim is 20% thicker, so its weld arc is wider (1.683° against
0.960° at `medium`) and its step is the gentler of the two. On `e126cc3` at `medium` the
weld/free boundary therefore sits at **+1.6824°**, and the contact patch centre sits at
**+1.885° to +2.082°** on a patch whose own half-width is 0.30–0.53°.

**And the patch is smaller than the element it sits in, in every one of the twelve cells.**
`patch / local segment`, where the local segment is the one *containing* the patch centre:

| genome | kin | smoke | coarse | medium |
|---|---|---|---|---|
| `e126cc3` | lin | 0.08 | 0.26 | **0.46** |
| `e126cc3` | svk | 0.11 | 0.24 | **0.41** |
| `e4219f3` | lin | 0.14 | 0.23 | **0.39** |
| `e4219f3` | svk | 0.15 | 0.27 | **0.41** |

**So the patch straddles a 16.8–30.3× discontinuity in element size, always lands inside a
single element on the coarse side of it, and which nodes it sees depends on whether its edge
happens to reach back across the step.** That is `wheel_objective.py:107-114`'s aliasing —
*"an artefact that refines away and exists because no config resolves the contact patch… it
ALIASES"* — located, measured, and given a mechanism rather than a symptom. Note the ratio
column refines **not at all**: going smoke → medium buys a smaller element but the same step,
so this is not something the mesh ladder fixes.

It explains the whole of Group A with nothing left over, which is the test §0(1)'s H2 set:

- **Why `nodes` and `quad` jump between kinematics on the same mesh.** At `medium` the weld
  arc is `[0, 1.6824°]` in 0.1052° elements and the free arc is `[1.6824°, 30°]` in 1.7698°
  ones. **Both patches sit inside the free element**, which is wider than the whole patch —
  what differs is whether the patch's near edge spills back across the boundary:

  | kin | centre | half | patch interval | reaches back past 1.6824°? | nodes | quad |
  |---|---|---|---|---|---|---|
  | linear | +1.9526 | 0.4097 | [1.5429, 2.3623] | **yes, by 0.1395°** | **3** | 11 |
  | svk | +2.0820 | 0.3610 | [1.7210, 2.4430] | no, misses by 0.0386° | **0** | 3 |

  The three linear nodes are the weld nodes at 1.5773 and 1.6298 plus the boundary node at
  1.6824, on a 0.0526° weld pitch; the free element's own nodes are at 1.6824, 2.5673 and
  3.4523, so once the patch clears the boundary there is nothing inside it. Same mesh, same
  genome, same load — **a 3.7x difference in how well the patch is sampled, decided by
  0.04° of where its edge fell.**
- **Why `deg_per_segment` never described the resolution.** `360 / len(rim_outer)` is the
  mean of a bimodal distribution: 0.938° at `medium`, against a local truth of either 0.105°
  or 1.770°. The column has been in this study since M6 and is an average nothing is
  measured at.
- **Why a node penetrates on one design and not another.** A node is inside the patch only
  when the patch is on the weld side; on the free side there is no node to penetrate. It is
  a statement about where the patch landed, not about how deep the contact is — the
  penetration itself is 1.1e-4 to 3.6e-4 of the 1.5 mm rim band on every row above, **9x to
  3x inside `GATE_PENETRATION_FRAC = 1e-3`**, which G1 gates and which passes everywhere.

#### 3. What the gate does NOT say, stated before anything is built on it

- **`n_quad_points_in_contact = 3` at `medium` under SVK is thin**, and it clears a bar of 2
  that this arc pre-registered rather than one anybody derived. The drop converges anyway,
  for M6's original reason — it is dominated by spoke and rim bending, not by how the last
  few newtons are spread — but "converges" and "well resolved" are different claims and only
  the first is measured here.
- **Three rungs is three rungs.** The Richardson limit and the 0.307% are read off
  smoke/coarse/medium with an observed change ratio, not an assumed order.
- **Nothing here measures the GRADIENT.** The claim is that the value is mesh-convergent.
  Whether `d(drop)/d(genes)` is equally well behaved across the weld/free boundary is not
  measured, and the aliasing mechanism in (2) is precisely the kind that would show up in a
  derivative before it showed up in a value. That is the honest successor to this step.

---

## Step 3 — Triage the reds this arc owns, and fix only those on its own mechanism

**Why.** §19 diagnosed several reds and deliberately fixed none, on the rule that loosening a
tolerance in the same session as the promotion that reddened it is the pattern §5 and §16 warn
about. This is a different arc, so that rule is satisfied — but "one arc, one mechanism" still
holds.

**Read first.** Step 0's node ids, Steps 1 and 2's records, `tests/test_contact.py:220-301`,
`tests/test_wheel_fea.py:69-89`.

**Do.** Three in scope, each decided against measurement rather than argument:

1. **`test_but_the_assumed_patch_got_the_axle_drop_nearly_right`** (`tests/test_contact.py:288`)
   — Step 1 says this is a divergence detector on the **legacy model**, not a wheel gate, and
   it has done its job. Re-characterise with the measured value on **both** genomes and a
   docstring naming which model it indicts and which records inherit the error (M4, M5,
   `study_gnl`, `study_wheel_fea`). Do not delete, skip or xfail it, and do not widen 0.05
   without stating what the number now is.
2. **The rim-node-penetration assertion** (`tests/test_contact.py:243`) — its own docstring
   pre-authorises the update. Rewrite the premise to what Step 2 measured; keep the pointwise
   no-pull check intact.
3. **`test_only_the_rim_od_near_the_bottom_is_loaded`** (`tests/test_wheel_fea.py:69`) — this
   is pure assumed-patch load assembly (`fem.wheel_problem(mesh, patch_half_deg=3.0)`), and
   §19's diagnosis is a **corrected formula, not a loosened tolerance**:
   `element_deg = SECTOR_DEG / (n_weld + n_rim_free)` is one *node pitch*, but rim-OD elements
   are quadratic and span **two**. Verify against `mesh.cfg.order` and the measured 2.8318°
   span, and derive the factor from the order rather than hard-coding 2.

**Out of scope, said explicitly:** the Group C manifest-bite tolerance
(`test_the_bite_is_the_volume_divided_by_the_right_thickness` — an export-precision defect,
not a contact one), Group B / defect 8, and §16's five characterisation reds.

**Gate.** `make test` closes at **11 − (number fixed)** failed with **no new red**, and nothing
deleted, skipped, xfailed or re-thresholded without its old and new values tabulated on both
genomes.

**Record.**

```
STEP 3 RECORD — DONE (2026-08-13).  GATE: PASS.  FOUR REDS FIXED, AND THEY ARE FOUR
DIFFERENT KINDS OF WRONG.  Nothing deleted, skipped, xfailed or re-thresholded blind.
  make test:   7 failed / 438 passed in 1460.18 s (24:20)
  against Step 0's 11 / 433:  exactly 11 - 4 reds, and NO NEW RED.
  438 rather than 437 because this arc ADDED one test — the containment pin on
  `_patch_resolution`, `test_the_reported_local_element_contains_the_patch_rather_
  than_merely_being_near_it`.  An intermediate run before it existed read 7 / 437.
  the seven left, all out of scope by the split registered before the work:
    1 Group C   test_export_contract.py::test_the_bite_is_the_volume_divided_by_...
    1 Group B   test_objective.py::test_the_margin_weight_is_the_exchange_rate_...
    5 inherited test_gnl.py::test_the_correction_is_not_a_constant_over_the_design_space
                test_gnl.py::test_the_correction_enters_at_first_order_in_the_load
                test_wheel_fea.py::test_the_rim_band_holds_a_large_minority_of_the_compliance
                test_wheel_fea.py::test_the_beam_to_wheel_ratio_is_not_a_constant
                test_wheel_fea.py::test_a_thicker_rim_monotonically_stiffens_the_wheel
```

**A fourth red came into scope once Step 2 found the mechanism.** The plan scoped three
(Group A) plus the Group C patch-spill bound, and left the Group C manifest-bite tolerance
out. That split still holds — but the reason the patch-spill bound is in scope turned out to
be stronger than "it is about the rim OD": **it is the same 16.8× element-size step**, and
§19's stated cause for it is wrong.

| # | red | what it actually indicts | change |
|---|---|---|---|
| 1 | `test_the_centre_node_rise_is_not_the_axle_drop` | a **premise**, not the wheel | premise recorded, bound moved to the one that carries the meaning |
| 2 | `test_the_real_patch_is_far_smaller_than_the_assumed_one` | a **`smoke`-tier artefact** | lower-bound clause scoped to `coarse` |
| 3 | `test_but_the_assumed_patch_got_the_axle_drop_nearly_right` | the **legacy assumed-patch model** | renamed, re-banded, two-sided |
| 6 | `test_only_the_rim_od_near_the_bottom_is_loaded` | a **constant standing in for a mesh fact** | bound read off the mesh |

**1 — the premise was true for the wrong reason, and its own error message said so.** It
asserted `node_gap.min() > 0` with the message *"the mesh has become fine enough that the
contact is no longer interior to a segment."* Refinement is not what changed it: Step 2's
table shows a node inside the patch at `smoke` and none at `medium` under SVK, which is the
wrong direction for a refinement story. What changed it is which side of the weld/free
boundary the patch edge fell on. The check now applies the bound that actually matters —
penetration against the rim band, the same quantity
`test_penetration_is_negligible_against_the_rim_band` gates at `1e-3` and which passes at
**3.6e-4** — and the node count is characterised in the docstring rather than required.

**2 — one cell in twelve, and it is the mesh the objective never runs.** `patch_half_deg /
hertz` across the Step 2 matrix: **0.962** at `e126cc3`/linear/`smoke`, and **1.082 to 1.720
everywhere else**. At `smoke` the entire patch lives inside one 0.4206° rim element, so its
zero crossing is interpolated within a single element's shape functions rather than resolved.
The lower-bound clause now builds its own `coarse` mesh — the identical scoping, for the
identical reason, that `test_the_sampled_patch_extent_is_biased_not_merely_noisy` already
carries three tests below it, and that §4 applied to `run_warm`. The upper bound ("several
times too wide") stays on `smoke`: it is a claim about a direction and a large factor.

**3 — this one is real, refinement-stable, and it is the finding rather than the artefact.**
`|real / assumed − 1|` at `n_quad=6`:

| genome | kin | smoke | coarse | medium |
|---|---|---|---|---|
| `e126cc3` | lin | 5.272% | 6.655% | 6.250% |
| `e126cc3` | **svk** | 5.224% | **6.760%** | **6.414%** |
| `e4219f3` | lin | 3.077% | 3.773% | 3.793% |
| `e4219f3` | svk | 3.403% | 3.677% | 3.795% |

It does not refine away, it is a property of the design, and it very nearly doubled across
one promotion. So the test keeps its job and loses its name: `test_the_assumed_patch_no_
longer_stands_in_for_contact`, banded **two-sided at 2–8%** so that a silent return to "the
assumption was fine" fails just as loudly as a further blow-out. The docstring records what
Step 1 established — that this indicts M4, M5, `study_gnl.py` and `study_wheel_fea.py`, and
**not** the shipped wheel's deflection.

**6 — §19 had the ratio right and the cause wrong, and the right fix is neither constant.**
The bound was `SECTOR_DEG / (n_weld + n_rim_free)` = 1.5° at `coarse`. §19 diagnosed the gap
as quadratic elements spanning two node pitches and proposed a factor of 2. Measured, the
element order has nothing to do with it: the sector is **10 × 0.1682° of weld arc + 10 ×
2.8318° of free arc = 30.000° exactly**, so the expression returns the *mean of two sizes
16.8× apart* while the element straddling the patch edge is the free one, **2.8318°, which
is 1.888× the bound** — §19's 1.9×, arrived at by a different route. A factor of 2 would have
been a second constant standing in for a number the mesh already knows, which is the same
defect one level along. The bound is now the measured widest rim-OD element.

---

## Step 4 — THE DECISION, pre-registered here so it cannot be rationalised later

**Why.** Steps 1–3 cost hours; what follows could cost a mesh change plus a 6 h 20 m
re-descent. SVK_PLAN Step 4 is the template — write the options and their measured costs down
before launching anything.

**Do.** Take Step 2's gate verdict and follow the branch registered before the run:

- **Branch A — Step 2 green.** The objective's deflection is mesh-convergent on `e126cc3`.
  Successor #1 is **retired**, with the retraction recorded rather than quietly dropped: the
  red gate was about the legacy assumed-patch model, Step 3 re-characterised it, and there is
  no defect on the shipped wheel here. The re-ranked #1 becomes **defect 5** — the
  `max(0,v)**2` barrier's dead knee, which cost 45 steps of a 6 h 20 m run and stands between
  this project and any *converged* buildable design — and that is a **new arc, not this one**.
- **Branch B — Step 2 red.** The patch resolution is a live defect on the design that ships,
  and the options get costed from the numbers rather than sketched: local rim-OD refinement at
  the patch (a mesh change × every descent, and it re-baselines every committed artifact); a
  patch-resolution barrier in the objective so the optimizer cannot walk into the aliased
  corner; or accept-and-document with a measured error bar on `axle_drop_mean_mm`. Pick one,
  with its cost, and only then build.

**Record.**

```
STEP 4 RECORD — DONE (2026-08-13).  BRANCH A, on the criterion registered before the run.
  Step 2's gate is green on all three clauses at the genome and kinematics that ship, with
  5.6x of margin on the clause that matters.  Successor #1 as PLAN.md §19 stated it is
  RETIRED.  It is not deferred and not re-scoped: the harm it claimed does not exist.
```

**What is retired, precisely.** §19 ranked first: *"the only red gate in the tree that
indicts the wheel currently in `best_solution.json`, it biases the quantity the objective is
steering by 5.27%, and it gets worse the harder the optimizer works."* All three clauses are
now measured and none survives:

| §19's claim | measured |
|---|---|
| "indicts the wheel in `best_solution.json`" | it indicts `fem.solve_wheel`, which **cannot be reached** from the objective — `TypeError` (Step 1) |
| "biases the quantity the objective is steering by 5.27%" | the objective steers by the real-contact drop; the 5.27% is the *other* model's error, and the objective's own quantity is mesh-convergent to **0.897%** (Step 2) |
| "gets worse the harder the optimizer works" | the promoted genome converges **2.2–2.5× better** than its predecessor, and its element-size step is 16.8× against 30.3× (Step 2) |

**And §19's deflection correction is withdrawn.** *"The promoted wheel's true deflection is
therefore near 1.90 mm against a 2.0 mm target, not the 2.008 mm the objective reports."*
The 2.008 mm **is** the real-contact figure at `medium`. Nothing about the promoted wheel's
deflection needs re-reading; the records that do are M4's, M5's, `study_gnl.py`'s and
`study_wheel_fea.py`'s, and Step 3's renamed test now says so where a reader will hit it.

**This does not reverse §19's promotion, and it does not weaken it.** Every argument §19 made
*for* `e126cc3` — 12.0% on the objective against step 0 of its own run, tier 0 at both
fidelities, 24/24 at both junctions with +0.0% `Kt` error, and hub utilisation 0.9964 → 0.7795
— is untouched by this arc. What moves is one line of the *against* column and the successor
ranking built on it.

#### The new successor this arc did find, and why it is NOT ranked first

The rim OD carries a **16.8–30.3× element-size step**, the contact patch always sits inside a
single element on the coarse side of it, and **the step does not refine away** — every rung
buys a smaller element and the same ratio. That is a real property of the discretisation and
nobody had written it down.

**It is ranked below defect 5 anyway, and the reason is the rule §17 wrote.** §17 measured
that §16's #1 successor could not move the cap because the branch it fixed was 253% from
binding, and this arc has now overturned a second #1 for the same class of reason. So the
question asked of this one before ranking it is *what does it cost*, and the answer is: **not
measured, and the one adjacent measurement says "little."** M7 already quantified the facet
artefact this step produces — 3.8% of the phase slope at `coarse`, 17.6% at `smoke` — and
recorded that **it refines away**; `phase_stencil`'s randomisation exists to turn what remains
into zero-mean noise (`wheel_objective.py:101-114`). Meanwhile Step 2 measured that the
*value* converges. What is unmeasured is the **gradient**, which is exactly where an aliased
facet would show up first and where a converged value proves nothing.

So the successor is stated as the cheap measurement rather than as the fix:
**run `study_gradient.py`'s G7 phase-smoothness section at `medium` on `e126cc3` under SVK,
and see whether the facet ratio still falls.** If it does, the step is a documented property
and nothing more. If it does not, *then* there is a case for local rim-OD refinement, and it
can be costed against a measurement instead of against a mechanism.

#### The successors, re-ranked

1. **Defect 5** — the `max(0,v)**2` barrier's dead knee. Promoted to #1 by default and on
   evidence: it cost **45 of 100 steps** of a 6 h 20 m run, and it stands between this project
   and any *converged* buildable design. It is the only item on this list with a measured
   price.
2. **`R_rim`'s box ceiling of 3.0** — pinned throughout §19's descent and at convergence,
   still never tested, still cheap.
3. **Defect 8**, and with it §18's rate gate, still red at 0.379 against [0.5, 2.0].
4. **The rim OD element-size step** — new, above. Ranked here because its cost is unmeasured
   and the adjacent measurement says small; the *first* piece of work is the G7 run, not a
   mesh change.
5. **The Group C manifest-bite tolerance** — diagnosed in §19, out of scope here, still red.
6. **Re-derive the five inherited characterisation gates.**

---

## Step 5 — Write the record into PLAN.md as §20

**Why.** PLAN.md is what a fresh session reads, and §19's successor list currently points the
next session at a mesh study it may not need.

**Do.** Add `### 20.` after §19 carrying: Step 1's verdict, the retraction stated as plainly as
§19 stated the claim, Step 2's mesh-convergence matrix, the reds triaged with old and new
values on both genomes, the gate count, and the re-ranked successors. Amend §19's Group A
paragraph in place with a pointer to §20 — **do not edit its numbers**, which were correctly
measured; only their interpretation moved. Point `BUILD_PLAN.md`'s successor list here.

**Record.**

```
STEP 5 RECORD — DONE (2026-08-13).
  PLAN.md          new section 20 appended (PLAN.md:3717).
                   section 19's Group A paragraph carries a retraction banner ABOVE it;
                   its numbers are untouched, because they all reproduce.
                   section 19's successor list: #1 struck through, pointing at 20.
  BUILD_PLAN.md    step 10's "Successors" block marked SUPERSEDED, pointing at 20.
  CONTACT_PLAN.md  this file — the working notes.
```

**The arc closes here. Nothing promoted, `best_solution.json` untouched, and that was never in
question** — this arc could only have moved the ranking, not the wheel.
