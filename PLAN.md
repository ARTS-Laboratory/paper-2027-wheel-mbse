# PLAN.md — the next changes

> **VERSION CONTROL IS NOW PART OF THIS PROJECT'S WORKFLOW — CHANGED 2026-08-19.** The rule
> that stood here read *"Ignore version control entirely. Do not commit, branch, stage, revert
> or otherwise touch git — it is not part of this project's workflow and nothing here depends
> on it."* **It is superseded by everything below.** The nine sibling plan files that carried
> the same sentence — `FILLET_PLAN.md`, `HUBSHARE_PLAN.md`, `KINEMATICS_PLAN.md`,
> `MESHSTEP_PLAN.md`, `RIMCAP_PLAN.md`, `WALLPIN_PLAN.md`, `BOUNDARY_PLAN.md`,
> `EXPORTPREC_PLAN.md`, `UNCAP_PLAN.md` — **were all swept the same day**, and each now carries
> a short superseding note pointing back here. **The rules are stated in this block and only
> here**: a sibling that restates them in full is a sibling that will drift, so the notes carry
> a summary and a pointer, never a second copy.
>
> **Anonymous authorship.** Commits carry the repository's own identity and nothing else. No
> `Co-Authored-By:` trailer, no assistant or tool attribution, no session link, no
> generated-with footer. A commit that names a tool is a commit to amend.
>
> **Cadence — one commit per finished unit of work**, where *finished* means the thing a
> numbered section or a plan Step describes, not a checkpoint mid-edit. A source fix and the
> regression test that pins it are ONE commit; the plan-file record of a measurement is a
> separate one. Prefer many small commits to an end-of-arc sweep. The tree as this rule is
> written carries **36 modified, 6 deleted and 13 untracked** files against `origin/feature`
> — that state is the reason the rule changed.
>
> **Green before commit.** `make test` passes, or the message says which test is red and why
> that is intended. An `xfail` that starts passing is a FAILURE, not a bonus —
> `xfail_strict = true`. **Never commit while a study driver is mid-write**: the drivers
> rewrite `studies/*.json` and `studies/*.jpg` in place over minutes to hours, and those files
> are tracked on purpose.
>
> **A study commit carries its artifacts.** Changing `studies/study_x.py` without the
> regenerated `study_x.json` and `study_x.jpg` commits a script that no longer describes its
> own output — the same rule `.gitignore` already states for `best_solution.json`.
>
> **A promotion is its own commit, and it is never one file.** `best_solution.json` moves
> together with the banner, the drivers carrying measured constants, and `make svk`;
> `tests/test_promotion.py` carries the checklist. One atomic commit, complete, with the
> genome hash in the subject.
>
> **Branches.** Work lands on `feature`. `main` is the promotion line — nothing reaches it
> that has not passed the promotion checklist above.
>
> **Message shape.** A subject naming what changed and where
> (`wheel_wheel: thread uncap through the differentiable mesh path`), then a body giving the
> measurement or the reason. The numbers are the point: a commit that says *"fix mesh bug"*
> has thrown away the 0.448 mm that made it a bug.
>
> **`PLAN.md` is BOTH tracked AND listed in `.gitignore`, and the ignore line is a no-op.**
> Git ignores nothing it already tracks, so this file is versioned today — but were it ever
> untracked, it would leave the repository silently. Decide which was intended before relying
> on either.

> **THE SIX CLOSED ARC FILES WERE DELETED ON 2026-08-16, AND THEIR CITATIONS WERE NOT.**
> `BUILD_PLAN.md`, `CONTACT_PLAN.md`, `DEFECT5_PLAN.md`, `DEFECT8_PLAN.md`, `REDS_PLAN.md`
> and `SVK_PLAN.md` are gone. Every one was a **closed** arc with a final record, and every
> one is summarised in a numbered section of this file — but **about 70 comments, docstrings
> and Makefile lines still cite them by name**, 30 of those to `SVK_PLAN.md` and 22 to
> `BUILD_PLAN.md`. Those citations are now historical: they point at working notes that no
> longer exist, and the surviving record is **this file's numbered sections**. Do not treat a
> dangling `see SVK_PLAN.md step N` as a missing file to restore — read the section instead:
>
> | deleted file | what it was | where the record lives now |
> |---|---|---|
> | `SVK_PLAN.md` | the SVK descent and its 18 steps | §15, §16, §26 |
> | `BUILD_PLAN.md` | buildability: hub fillet cap, `R_hub` floor 0.5→0.4, minwall arms | §16, §18, §24 |
> | `CONTACT_PLAN.md` | the contact-patch arc and the 11-red Step 0 table | §19, §20, §31 |
> | `DEFECT8_PLAN.md` | the margin-weight defect and the `09e8188` candidate | §23, §26 |
> | `DEFECT5_PLAN.md` | defect 5, closed as "none of them" | §21 |
> | `REDS_PLAN.md` | the five inherited reds | §31 |
>
> **`HUB_PLAN.md` was ALREADY dangling before this cleanup** — `wheel_wheel.py:73` says "See
> HUB_PLAN.md" and no such file was deleted today because none existed. Do not attribute that
> one to the deletion above; the hub fillet milestone's record is §16 and §24.
>
> **The open work is no longer in this file.** It is one file per arc — see *Open arcs* below.

---

## Open arcs — one file each, created 2026-08-16

These replace the deleted plan files. Each is a live arc with its own steps; this table is the
index and the ranking. **Arc 1 is CLOSED (§32, 2026-08-16); 2–8 are unstarted.**

> **ARC 2 IS NO LONGER "UNSTARTED" — Step 0 and Step 1 are answered (§34, §38, §44, §47).**
> The seam merge survives untouched, half the mesh's corners are not the part's, and as of
> **§44 the instrument is reconciled, committed and tested** (`make fillet`). **§47 retired
> both of the routes that stood here** — a dedicated block on `A - P_t - B` (that region has
> two 0 deg corners) and a generated spoke block (the angle that fails is between two
> BOUNDARY curves) — **and it retired "what blocks a filleted mesh is the spoke block" with
> them**: end the spoke at the tangent station and it is clean across the whole gene box.
> What replaces them is a boundary-layer block, measured to mesh at min scaled Jacobian
> 0.91+ everywhere in the box, whose price is a notch in the ring block (`make filletblock`).
> Step 2 stays unreachable until that re-cut exists, and it must be judged on `det J` at the
> Gauss points, because `build_wheel` returning is not the same statement as a mesh that
> integrates.
>
> **AND AS OF §48 (2026-08-23) THE RE-CUT EXISTS AND IS MEASURED, THOUGH IT IS NOT WIRED.**
> The whole filleted sector closes — **eleven blocks, fourteen whole-edge seams**, 48/48
> cells across the shipped genome's radius box at `coarse` and `medium`, worst min scaled
> Jacobian 0.357 against the unfilleted sector's 0.783. **PART 9's shallow cut is proved
> uncloseable**: the cut has to reach the ring's far side. Two prices — the ring's radial
> node count becomes `n_thick`, and `R_hub` is bounded at 3.130 mm by the SECTOR rather
> than by the block. **And its scope is now named**: the radius box is not the gene box,
> and on 16 feasible genomes spanning all four flank orientations, 6 refuse the fillet
> outright and only 4 of the 10 that build clear the barrier. `fillet=True` is on course to
> be a MEASUREMENT INSTRUMENT for one genome — which is what Step 2 needs — and not an
> optimizer path.
>
> **AND AS OF §50 (2026-08-23) IT IS WIRED. `build_wheel(genes, cfg, fillet=True)` RETURNS
> A MESH THAT INTEGRATES** — eleven blocks, zero non-positive Gauss points at `coarse` and
> `medium`, seam error 3e-14 mm, and exactly twelve-fold periodic under a real solve.
> `fillet=None` is bit-identical. `mesh_coords` REFUSES a filleted mesh, which is §48's
> scope note made mechanical.
>
> **AND AS OF §52 (2026-08-23) STEP 2 IS RUN, WITH A SPLIT VERDICT.** `make corner-fillet`,
> 22 s, same driver as `make corner` with one flag. **At `P_t` — the PART's own corner, the
> one this fillet reaches — the singularity is GONE**: on the filleted body it is not a
> corner but an interior point, and the fillet SURFACE's peak settles to **36.8 / 16.1 MPa**
> where the sharp corner ran to 85.9 / 60.7 and was still climbing. **STEP 2's HEADLINE IS
> NOT DELIVERED**: the wheel's global maximum is still on `rim:P_c`, the END CAP's artefact
> corner, which the fillet does not reach and which only the RIM TRI-BLOCK removes — §46's
> chain arriving intact, and why the tri-block is now ranked 1. **And the unfilleted wheel
> has been overstating its own deflection — its axle drop is 61.2% HIGHER than the filleted
> mesh's, equivalently the fillet takes 37.97% off it** (single-phase, linear, one genome), with
> the filleted drop also CONVERGING — 0.14% over `coarse..fine` against 1.22%. Scope
> unchanged: `fillet=` is a measurement instrument for one genome, is not wired into the
> optimizer, and that is now a test rather than a note.
>
> **AND AS OF §53 (2026-08-23) THE RIM TRI-BLOCK — THE THING §52's `rim:P_c` BLOCKER WAITS
> ON — IS BUILT, AND IT IS NOT ADOPTABLE YET.** `make triblock`, 14 s. Both of §37's
> clauses are retired: **TWELVE blocks and SEVENTEEN whole-edge seams**, closing at
> 7.1e-15 mm, and the "forced 1-element strip" was §37's own choice of a FREE count. It
> meshes at **0.626 / 0.582 against the quad block's 0.0082 / 0.0083 — a factor of 77** and
> 3.1x over `MIN_SJ_TARGET`, at the 1.06 degree fidelity that was the point. **But the
> faithful rim is not opt-in**, so the gene box is the measurement, and there it folds on
> 4/16 and 6/16 genomes — the WIDE weld arcs, 15.9-41.2 deg against the shipped genome's
> 2.73. The construction is proved; **the rule that places its interior point is not**, and
> that is a third obstacle neither §37 nor §51 named.

| # | file | the question | cost |
|---|---|---|---|
| ~~1~~ | ~~`KINEMATICS_PLAN.md`~~ | **CLOSED 2026-08-16 — §32. NO, not for search.** ρ = **−0.83** over the feasible pool; `wheel_stage3.py --kinematics` now defaults to `svk`, at 1.49× | settled for 3549 s + 303 s |
| 2 | `FILLET_PLAN.md` | Mesh the junction fillets. **Steps 0-2 DONE (§50, §52).** `R_hub`/`R_rim` move the solved wheel by 38% on the filleted mesh and are still invisible to the OPTIMIZER, which may not take that mesh. What is left is Step 3, behind genome-robustness — **and the rim tri-block is no longer part of that queue: §53 BUILT it, and it has the same genome problem** | Steps 0-2 spent; Step 3 not cheap |
| 3 | `HUBSHARE_PLAN.md` | Should hub compliance be an objective term? `cy4` alone moves it by 102% of the gap (§31) | medium |
| 4 | `WALLPIN_PLAN.md` | Re-derive Gate 1 at the 1.2 mm floor and drop the beam test's 2.0 mm pin (§14's reserved judgement, measured by §31) | small |
| 5 | `RIMCAP_PLAN.md` | A rim cap model — the boundary as a function of `t3` and rim arrival angle, not at one design (§22, §24) | medium |
| 6 | `MESHSTEP_PLAN.md` | The 16.8–30.3× rim-OD element-size step the contact patch sits on (§20) | small, needs G7 first |
| 7 | `EXPORTPREC_PLAN.md` | Make the exporter write the overlap at 4 dp instead of 2 — §28's better fix, deferred | small, touches a shipped artifact |
| 8 | `BOUNDARY_PLAN.md` | Defect 5's boundary placement, worth 0.61% of loss (§21) | small, lowest ranked |

**Ranking note — SETTLED.** 1 was put ahead of 2 deliberately, against §30's ranking, on the
argument that it was cheap and that "if the answer is *linear is not acceptable*, it changes
what a filleted-mesh ladder should even be measuring." **That was the right call and it paid.**
The answer was *not acceptable*, and it arrived for 3549 s of `make kinrank` plus 303 s of
`study_gnl` — against the fillet arc's "expensive, not cheap".

**2 IS NOW UNBLOCKED, AND IT INHERITS A CONCRETE INSTRUCTION RATHER THAN A WARNING.** The
fillet ladder must be run under **SVK**, and after §32 it gets that by default from
`wheel_stage3.py` — but **the study drivers do NOT**: `wheel_fem`'s kernel defaults are still
`linear` on purpose (§32), and 11 drivers never mention `kinematics` at all. A ladder built on
those takes linear silently. That is the one thing to check before starting 2.

**§32's own successors 1 and 2 are cheaper than any of 2–8 and should go first**: `make studies`
has been an unrunnable gate since 2026-08-06 and takes four drivers down with it, and two
committed study artifacts describe a wheel promoted out of their named file three times ago.

> **BOTH CLOSED — §33, 2026-08-16.** And §32's premise was wrong in the useful direction: the
> recipe died at **line 4** (`study_wheel_fea`), not line 5, so it never reached `study_gnl`
> and fixing M5 alone would have unblocked nothing. Two gates were exiting nonzero on
> *characterisation findings* rather than on broken solves; both now exit on solver
> correctness, with **neither threshold moved**. `study_gnl.json` describes the shipped wheel
> again (`service_rel_diff` 0.23160, not the GA/beam 3.95%). A third defect surfaced only by
> running the recipe twice: **seven of the nine drivers wrote artifacts to the CWD**, so every
> committed `.jpg` had been stale since 2026-08-03. **`make studies` now runs all nine drivers
> green** — the four that had never been reached cost 4 h 15 m between them and every one
> PASSES. **Arc 2's prerequisite is also now checked — see §33's successor 3:
> `study_corner_singularity.py` takes the linear kernel default.**

> **§33's LAST SENTENCE WAS OVER-OPTIMISTIC, AND §40 IS THE ONE THAT SETTLES IT.** "`make
> studies` now runs all nine drivers green" was written on 2026-08-16 having reached **five**
> of nine: `study_contact`'s G1 was red behind the old blocker and stopped the recipe there
> (§38). §39 split that gate's solver verdict from its characterisation verdict, and on
> **2026-08-20 the recipe completed end to end, exit 0 — §40**. 5:02:52, 30.8 GiB, all nine
> PASS on `solver_is_correct`, with `study_gnl` and `study_contact` both printing a red
> characterisation finding and exiting 0 anyway. **That is the first completion since
> 2026-08-06**, and it has happened exactly once.

> **THE SHIPPED GENOME IS `09e8188`, PROMOTED 2026-08-14 (§26). READ THE CHAIN BEFORE
> TRUSTING ANY PER-DESIGN NUMBER IN THIS FILE.**
>
> ```
>   36aed36  GA/beam optimum      →  best_solution_ga_beam.json, pinned, never moves
>   350f4c7  §13, 2026-08-06      →  stage3_minwall_best_1.2.json   (the §14 control genome)
>   e4219f3  §16, 2026-08-11      →  stage3_buildcap2_feasible_medium.json
>   e126cc3  §19, 2026-08-13      →  stage3_margin_best_medium.json
>   09e8188  §26, 2026-08-14      →  best_solution.json  ← SHIPPED.  48.64 g OCC / 39.47 g mesh
> ```
>
> **AND THE BANNER BELOW WENT STALE FOR TWO PROMOTIONS, WHICH IS ITSELF THE WARNING.** It
> declared `350f4c7` shipped and said in terms that the shipped genome had not changed; that
> stopped being true at **§16 on 2026-08-11** and stayed wrong through §19. `SVK_PLAN.md` step
> 7 requires this banner to be amended whenever the shipped genome moves and it was not, twice.
> The paragraphs below are kept **as written**, scoped to their own dates, because their
> content is still correct history — only their claim about *what ships today* was wrong. This
> is the same defect §25 found in the study drivers, in the one place specifically designed to
> prevent it.
>
> **THE SHIPPED GENOME CHANGED ON 2026-08-06 — READ THIS BEFORE TRUSTING ANY NUMBER BELOW.**
> `best_solution.json` is now the Stage-3 1.2 mm optimum, genome **`350f4c7`**, 39.194 g on
> the mesh / 47.63 g as an OCC solid. Everywhere in §1–§12 that a table, a study or a
> sentence says **"best_solution"** it means the genome that used to be in that file: the
> v2.1 GA/beam optimum, **`36aed36`**, `t0` 2.48 mm, 74.12 g. That genome is preserved
> unchanged as `best_solution_ga_beam.json`, and **none of those measurements are wrong** —
> they simply describe *the old wheel* now. §9's 1.36 load factor and §0's utilisation table
> are both in that category. Every driver in `studies/` defaults to `best_solution.json`, so
> **re-running any study today measures the new wheel and will not reproduce the old page.**
> §13 is the record, and it lists the numbers that moved.
>
> **AND: `wheel_fea.MIN_WALL_MM` DEFAULTS TO 1.2 AS OF THE SAME DAY**, down from 2.0. The
> shipped genome is a 1.2 mm design and the old default put all four of its thickness genes
> outside the box they are supposed to live in. **The GA and every driver now search down
> to 1.2 mm by default** — if you are reading a run whose numbers assume a 2.0 mm floor,
> that run predates this. See §13.
>
> **AND, 2026-08-10: EVERY STAGE-3 NUMBER IN §1–§14 IS A LINEAR-KINEMATICS NUMBER.**
> `wheel_contact_problem` defaults to `kinematics="linear"` and nothing in the Stage-3 path
> ever overrode it. The shipped wheel deflects **2.409 mm, not the 1.953 the optimizer saw**,
> and carries **0.875 of allowable, not 0.799** — measured, at `medium`, and it is **still
> feasible** with every barrier at 0.0. None of those numbers is wrong; they are answers to a
> different question. **THE SHIPPED GENOME DID NOT CHANGE *ON THAT DATE*** —
> `best_solution.json` was still `350f4c7` **as of 2026-08-10**, because the wheel the SVK
> descent found clears every FEA gate and then **does not build** (`kt_error_pct` +11.9% at the
> hub, as-built utilisation 1.046). §15 is the record and `SVK_PLAN.md` is the evidence.
> *(It changed three times after this paragraph was written — §16, §19, §26 — and the sentence
> was not amended at any of them; see the chain at the top. The 2026-08-10 claim stands for
> 2026-08-10 and for nothing after it.)*
>
> **AND THAT PARAGRAPH'S LAST SENTENCE WAS OVERTURNED ON 2026-08-16 — §32.** It used to read
> *"Linear remains the default everywhere on purpose; `--kinematics svk` is opt-in on
> `wheel_stage3.py` and `study_gradient.py`."* **Both halves were wrong by then.** The
> descents stopped taking that default at **§16 on 2026-08-11** — `best_solution.json`'s own
> `search` block reads `"kinematics": "svk"`, and §16, §19 and §26 all passed the flag
> explicitly — so **the shipped wheel was NOT optimised on the wrong physics**, and the
> sentence had been describing a state of affairs three promotions out of date. And "on
> purpose" is no longer defensible for search at all: §32 measured that **linear does not RANK
> designs the way SVK does** (Spearman **−0.83** over the feasible pool, **−0.8333** on §8's
> controlled minwall ladder, where linear's best arm is SVK's worst of eight).
>
> **AS OF §32 THE DEFAULT DEPENDS ON WHICH LAYER YOU ARE IN, AND THE SPLIT IS DELIBERATE:**
>
> | layer | default | why |
> |---|---|---|
> | `wheel_stage3.py --kinematics` (**the search**) | **`svk`** | §32: linear returns a different design. Costs 1.49× |
> | `wheel_stage3.descend(**problem_kw)` | `linear` | records written before the key existed must keep meaning what they meant (`test_the_run_record_carries_the_kinematics_it_actually_descended`) |
> | `wheel_fem.solve_wheel` / `wheel_problem` / `wheel_contact_problem` (**the kernel**) | `linear` | a REPORTING question §32 did not measure; ~470 tests and 11 study drivers take it silently. Filed, not done |
>
> **So `--kinematics linear` is now the opt-in one for a descent, and it is still supported** —
> `study_gnl`, `study_kinematics_rank` and the M5/M7 controls all need it as an arm.
> **Four live recipes changed behaviour**: `stage3`, `prod9`, `prod10`, `minwall-%`. Nothing
> that ships changed, because everything that shipped already passed the flag.

---

## Where the tree stands — the minimum a fresh session needs

> **`make test` READS `0 failed` (§31, 2026-08-15).** It had read `5 failed` since §19, and
> establishing "no new red" meant opening `CONTACT_PLAN.md` to look up which five. It no
> longer does. Two of the five were a seed lottery on a max/min statistic and are **fixed**;
> one asserted an absolute deflection at the least converged mesh in the tree and is
> **fixed**; two are accepted findings and are now **strict `xfail`s** carrying a `reason=`
> that names the section which decided them. `xfail_strict = true` is set in
> `pyproject.toml`, so **an xfail that starts passing is a failure** and either one reopens
> itself automatically. **A non-zero failure count now means something again — treat any red
> as new.** Nothing is left waiting on a decision: the hub compliance bound was measured,
> put up, and **resolved in §31 item 4 — it stays at `0.03` as an accepted deficit**, with
> the curvature route (`cy4`) filed as a design successor for Stage 2/3's objective.

**M8b-i.6 step 2 landed.** The stress constraint is no longer a p-norm rescaled to the true
max by a measured ratio. It is now

```
Kt(R, t) * sigma_nominal(p=4)  <=  ALLOWABLE_STRESS_MPA        # = 25.0
```

with **one `soft_barrier` per junction, summed** — hub priced on `(R_hub, t0)`, rim on
`(R_rim, t3)`. `Kt = 1 + C*(t/2R)^0.65` clamped to [1.0, 3.5], differentiated by `jax.grad`,
not frozen. `stress_scale` is gone from `t3_terms` and `objective`; `stress_scale_measured`
survives in the report as a read-only diagnostic because it is the *evidence* for the change.

Why it had to change: `c = max/pnorm` is anchored to M4's crack-tip singularity, which
diverges under refinement, so `c * pnorm` converged at **no exponent at either design**. The
sweep that proved it is `study_stage3_pnorm.json` (`make m8bi6`).

**Key constants.** `wheel_objective.STRESS_NOMINAL_P = 4.0` is the `t3_terms` default.
`wheel_adjoint.STRESS_PNORM_P` stays **30.0** — it is the documented default every historical
record was measured at, and a test pins it there.

**Measured after the change** (`medium` rung):

| design | Kt_hub | Kt_rim | sigma_nom(p=4) | **util** | util GCI | field max |
|---|---|---|---|---|---|---|
| best_solution | 1.861 | 1.490 | 5.507 MPa | **0.4099** | **0.45%** | 48.47 MPa |
| elite 1 | 1.871 | 1.490 | 6.765 MPa | **0.5063** | **0.20%** | 71.40 MPa |

Someone will put `util = 0.41` next to `max = 48.5 MPa` and panic. The answer is M4's,
unchanged: **the max is not a number.** It diverges 31.02 → 41.54 → 48.47 under refinement.

**Gates, all green:** `make test` **427 passed** (425 before §8's second pass; 406 before
the wall-floor work; 383 at M8b-ii; 357 before the phase pool; 269 before M8b-i.6, whose
Kt-twin equivalence test is parameterised 84 ways). Gate 7 `min_decades`
**2**, `worst_best_rel` **2.009e-07** — both better than the 1 / 1.820e-05 baseline.
`make m8bi6`'s `pnorm_by_p` block reproduces the step-1 sweep **bit-identically**,
0.000e+00 on every value and every GCI including the `c` columns; `max_stress_mpa` and
`axle_drop_mean_mm` are identical too, confirming no physics moved.

**M8b-ii item 1 also landed: the phase loop is process-parallel.** `--workers` on
`wheel_stage3.py`, gated by S13 (`make m8bii1`, PASS). **3.95× at 8 workers on 16 cores,
and the production run projects 46.46 h → 11.77 h.** Pooled values are bit-identical to
serial; gradients agree to 2.1e-16. Details, and the `XLA_FLAGS` finding that made the
comparison meaningful, are in "the next changes" below.

**M8b-ii item 2 also landed: `t1_vector` is jitted and cached.** T1 and T2 run in the
PARENT on every path (serial or pooled), so once the phases were parallel they became
the whole of Amdahl's serial fraction — this is why the item mattered more than its raw
number suggested. `wheel_objective._t1_cached_value_and_jacobian` replaces the old
`t1_vector(...)` call plus a separate `jax.jacrev(t1_vector)(...)` call with one
`jax.jit`-compiled closure (`jax.vjp` + `jax.vmap` over an identity basis, sharing the
one forward pass) cached on `(cfg.name, span_mm, flanks, weights-projected-to-the-six-
T1-keys)`, exactly the `coord_fn` idiom — genes are the sole traced argument, never in
the key. `t1_vector` itself is untouched; the cache is a new call path in `objective()`
only. **Measured (S10, `coarse`): 1.06 s → 0.0054 s, ~195×, 0.73% → 0.003% of a full
evaluation.** `make test` **384 passed** (383 before). `t2_vector` was deliberately left
alone — its own eager surface is small (`mesh_coords` already routes through the
jitted, cached `coord_fn`) and there's no measured complaint to fix; revisit only if a
future S10 run shows it's material.

**M8b-ii item 3 also landed: a periodic fidelity check exists on `descend()`.** There
was no spec for "multi-fidelity checkpoints" beyond the budget number (`medium` 2.8×
`coarse`) — chose the narrowest reading: `fidelity_check_every`/`fidelity_check_cfg`
forward-evaluate the just-accepted iterate at a second config every N steps, t3 tier
only, discard the gradient, and attach the raw result to that step's row under
`"fidelity_check"`. Pure observation — the discarded gradient never reaches
`m`/`v`/`delta`/`z`, proven by a test that runs the same seed with the feature on and
off and asserts `z`/`grad`/`loss` are bit-identical at every step. No disagreement
threshold is invented (matches the M8b-i.6 lesson about `stress_scale`'s old rescale:
don't calibrate a threshold with no measurement behind it); a second-fidelity solve
failure is a `fidelity_check_failed` event, not an abort. CLI:
`--fidelity-check-every`/`--fidelity-check-config`, off by default. `descend_lbfgsb` is
untouched (not the default path, and its `fun` is called on every line-search
evaluation, not just accepted steps — "every N steps" would mean something else there).
`make test` **391 passed** (384 before).

**The hub fillet landed too — §3 below, full record in `HUB_PLAN.md`.** `_embed`'s inward
step now plunges radially instead of running 4.5 mm sideways, so the hub circle exists again
and **all 24 of its corners are filleted, from 0 of 12**; `fillet_junctions` no longer
abandons the family that refused the first rung, which also takes the rim from 12 of 24 to
24 of 24; and a junction is now priced by its **worst** corner, which is why the rim's
`kt_error_pct` moved off a +0.0% that had twelve square corners hiding behind it. `make test`
**396 passed** (395 before, one of which was the contract test pinning the old broken hub).

---

## THE VERDICT REVERSED: the problem is FEASIBLE, and always was

S9 called this design space infeasible. That verdict was read off `c * pnorm`, which has no
mesh-independent value. Re-scored on a constraint that does:

| elite | util | defl err | corner distance |
|---|---|---|---|
| **10** | **0.4548** | **+1.65%** | **0.000  ← FEASIBLE** |
| **9** | **0.4468** | **+2.29%** | **0.000  ← FEASIBLE** |
| 7 | 0.4667 | +7.93% | 0.586 |
| 12 | 0.4710 | +13.78% | 1.757 |
| 0 (`best_solution`) | 0.4063 | −25.43% | 4.086 |

All 16 scored, none failed. **Every elite is stress-feasible** (util 0.23–0.53 against 1.0);
the binding constraint is now deflection alone. Two elites are inside the feasible box on
both. `corner_distance` is zero only when `util <= 1.0` **and** `|defl_err| <= 5%`.

Spread across the 16: utilisation 0.2272–0.5272, deflection error −77.27% to +28.38%. **That
is a wide spread — these elites are not one basin**, which is what made S9's three descents
from a single start a statement about a basin rather than about the space.

### And the bound descents agree — `make m8bi5` complete, 9086.3 s, OVERALL: PASS

Both probed starts satisfy **both** constraints at a visited design, 20/20 steps accepted,
no reject events:

| start | probe | utilisation | deflection error |
|---|---|---|---|
| elite 9 | `stress_only` | 0.447 → 0.456 | 2.3% → −1.4% |
| elite 9 | `deflection_only` | 0.447 → 0.443 | 2.3% → **0.04%** |
| elite 10 | `stress_only` | 0.455 → 0.458 | 1.7% → 0.2% |
| elite 10 | `deflection_only` | 0.455 → 0.448 | 1.7% → 0.3% |

Restated over 2 probed starts and 16 scored designs, 100 (util, deflection) pairs measured:

```
lowest utilisation seen anywhere   0.2272   (feasible at <= 1.00)
smallest |deflection error| seen   0.04%    (feasible at <= 5%)
BOTH satisfied at any design       YES
```

**S9 said "each constraint is reachable alone, neither with the other." That is now false.**
Driving deflection to 0.04% error *lowered* utilisation (0.447 → 0.443). The two constraints
were never in tension; the tension was an artifact of measuring stress against a singularity.
The old bound "min reachable utilisation 0.932" is **invalid** — it is `c * pnorm` at p=30.

**So the genome does not need new genes to reach feasibility.** The question changes from
"can this wheel be built" to "how good a wheel can be built".

---

## The next changes, in order

### 0. WHAT TO DO NEXT

**READ §8, §9 AND §10 FIRST — THEY CLOSED ITEMS (1) AND (2) AND CHANGED WHAT (3) IS.**
One line each:

- **§8 — `MIN_WALL_MM` is priced, and the floor DOES stop binding — at the ends, near
  1.5 mm.** EIGHT 125-step arms, 2.2 mm down to 0.8 mm. `t3` lifts off the floor at
  1.4 mm and `t0` at 1.0 mm, both settling on floor-independent values (**`t0 ≈ 1.45`,
  `t3 ≈ 1.6`**); `t1`/`t2` stay pinned at every floor. **The first pass's headline
  (`29.5 g/mm`, "never stops binding") is RETRACTED** — it was fitted entirely inside the
  band where all four genes were on the bound, and the marginal cost actually falls 4x
  across the full band (~31 → ~20 → ~8 g/mm). 2.0 → 1.2 mm is worth 19.4 g; below 1.2 mm
  is worth 3.3 g. Item (2) is CLOSED.
- **§9 — M9 phase 3: the load factor is mesh-convergent and is NOT a safety factor.**
  `λ(f) > 1` at all 11 load levels out to **4× service**, zero solver refusals, so there
  is no fixed point and the "1.36" recedes as fast as the load advances. Phase 3 stays
  blocked, now for a MEASURED reason. Item (1)'s successor is CLOSED as a question.
- **§10 — item (3) is DEFERRED ON PURPOSE, and the golden test has been decoupled from
  it.** §8 makes elite 10 optimal *at the 2.0 floor only*, so which genome should ship
  now depends on a floor decision that has not been made. `best_solution.json` is
  untouched; `tests/test_golden.py` now reads `best_solution_ga_beam.json` so a future
  promotion is a one-file change that cannot re-baseline the regression net.
- **§11 — THE FLOOR DECISION LANDED (2026-08-05): the process can hold 1.2 mm.** So the
  candidate is `stage3_minwall_best_1.2.json`, **39.194 g**, and §10's two prerequisites
  are done: the `medium` re-score confirms the mass to five figures with every barrier
  still at 0.0, and the export is OCC-valid with **`kt_error_pct` = +0.0% at BOTH
  junctions** — the first genome whose built fillets match the ones its stress model
  priced. **It is still NOT promoted**: the exporter's `[WEAK JUNCTION]` check fired on
  the hub (18.12 mm³ against a 50 mm³ floor) and the smallest edge fell to 0.0087 mm.
  Deepening the embed is measured and does NOT fix it. Read §11 before touching
  `best_solution.json`.
- **§12 — THAT CHECK WAS MEASURING `t0`, AND IT IS FIXED (2026-08-05).** Junction overlap
  is quadratic in the root thickness, so the fixed 50 mm³ floor ranked genomes by how
  thick they were and called it a verdict on the weld — which is why **elite 10 failed it
  too, at 48.72 mm³**. Normalised, the three genomes measure **0.562 / 0.544 / 0.571**
  root thicknesses of penetration across a 2× range in `t0`: the same junction, three
  times over. `MIN_JUNCTION_OVERLAP_MM3` is gone, replaced by
  `wheel_geometry.junction_bite` and `MIN_JUNCTION_BITE = 0.25` — **a geometric floor, not
  a calibrated one**, since no junction in this repo has ever actually failed. Still a
  warning, never a raise. Three tests where there were none. Both genomes re-exported and
  the manifest diff proves it is reporting-only: **not one geometric number moved.**
  **`make test` 430 passed.** Promotion now waits on exactly one thing — the Inventor
  import of `export/stage3_minwall_best_1.2.step`.
- **§13 — PROMOTED (2026-08-06). That import passed, and `best_solution.json` is now
  `350f4c7`.** The shipped wheel is the 1.2 mm Stage-3 optimum: **74.12 g → 47.63 g as an
  OCC solid**, `kt_error_pct` **+0.0% at both junctions** (the first shipped part whose
  fillets match the ones its stress model priced), one fillet family per junction instead
  of two. The export is identical to §11's candidate in all 33 geometric keys. The price is
  stated plainly in §13: **stress utilisation 0.41 → 0.80** at the `medium` rung, still
  inside the barrier with every constraint term at exactly 0.0, but no longer comfortable.
  `best_solution_ga_beam.json` keeps the old genome and `tests/test_golden.py` keeps
  reading it, so nothing was re-baselined. **Every per-design number in §1–§12 now describes
  the old wheel** — see the banner at the top of this file.
  **`make test` came back 410 passed / 20 failed, and TEN OF THOSE WERE FINDINGS, NOT
  PINS.** The most important was that `MIN_WALL_MM` still defaulted to **2.0** while the
  shipped genome is a 1.2 mm design, putting all four thickness genes outside the default
  gene box — **the default is now 1.2**, which closed three failures on its own. The gate
  stands at **423 passed / 8 failed**. Read §13's gate section before running any driver on
  this genome.
- **§14 — those eight triaged (2026-08-06). Three were about the TESTS, and the
  "eight findings about a thinner wheel" reading was half wrong.** Every one was diagnosed
  by sweeping something — the mesh tier, the gene box, the reference — on the promoted
  genome AND on `best_solution_ga_beam.json`, so "the new design broke it" always had to
  survive a comparison against the design it replaced. Three times it did not: the area
  order is **exactly 2.000** on both genomes once measured self-referenced instead of
  against a beam integral that has its own error floor; the p99 ratio test is a divergence
  detector misfiring on an already-converged quantity, and the OLD genome fails it too one
  tier down; the contact failure is smoke-only and clears by 5x at `coarse`. Two more were
  tests pinning a **pathology** as an invariant. **Six thresholds touched, four of them
  now TIGHTER.** The exporter publishes `fillets.volume_mm3` (**6.18% of the solid**, not
  the 0.92% its docstring claimed), which turned the mass budget from a fitted band into a
  named one — and exposed a new open item: `EMBED_ALLOWANCE_PER_SPOKE_MM2 = 3.03` is a
  `t`-proxy and the real gusset is **0.98 mm²/spoke**. **Left deliberately red: the
  pre-registered GNL gate and the hub compliance share** (0.0321, sign not understood).
  Also still open: `make hubcap` at 1.2.
- **§14 item 4a — THE GNL GATE STANDS, and chasing it down found the biggest thing on this
  page.** The 1%-load gate is a tripwire for the service-load number, and the full ladder
  says the shipped wheel's axle drop at service load is **+23.3% under SVK against linear**
  (the GA/beam wheel: +3.95%, reproducing M5 exactly). Every headline for this genome was
  computed under `kinematics="linear"`. **Deflection is 2.409 mm, not the 1.953 the optimizer
  saw — 20% past the 2.0 mm target it was tuned to hit exactly**, and utilisation goes from
  0.799 to roughly **0.91** against an allowable of 1.0. Relaxing the gate would silence the
  only automatic warning that linear kinematics no longer describes this part; **the real
  question is whether Stage 3 should descend on a linear solve at a 1.2 mm wall at all**, and
  that is a scope decision. Also fixed en route: `study_wheel_fea.stress_report` was applying
  the LINEAR strain formula to SVK fields — a **+169.5% artefact against a real +14.3%** — and
  now dispatches on `res["meta"]["kinematics"]`.
- **§15 — STAGE 3 CAN NOW DESCEND UNDER SVK, AND THE WHEEL THAT DESCENT FINDS CANNOT BE BUILT.
  NOTHING PROMOTED (2026-08-10).** Working notes in `SVK_PLAN.md`. **Before this milestone
  every Stage-3 number in this repo was a linear-kinematics number** — read §15 before
  quoting a deflection or a utilisation out of §1–§14. The adjoint was proved correct under
  SVK first (**all ten M7 gates, thresholds unmodified**, including G1's unrolled-Newton
  check at 5.9e-11 against 1e-8); SVK costs **1.36×** time and 1.05× memory, and the penalty
  is in the gradient, not in Newton iterations. Re-scoring six designs at `medium` says the
  shipped wheel **IS still feasible** (util **0.875**, not §14's estimated ~0.91, every
  barrier 0.0) but that **the design ranking INVERTS** — which reverses §8's `minwall` sweep
  over the whole 1.2–2.0 mm range. Two 300-step SVK descents both cleared every
  pre-registered clause, and a `medium` re-convergence (`bc77614`) is **1.78 g lighter and
  12× closer to the deflection target** than the incumbent under the honest kinematics. **It
  is not promoted: it does not build.** `kt_error_pct` **+11.9%** at the hub, as-built
  utilisation **1.046** against a modelled 0.935, where the incumbent is +0.0% — a regression
  this arc introduced, caught by the control. The cause is **four measured defects in the
  objective**: `stress` has identically zero gradient below util 1.0, so `R_hub` and `R_rim`
  are **dead genes** (nonzero gradient on 2 of 602 steps), so nothing objected when the
  descent swung the hub arrival shallow — and the 1.2 mm floor, not the stress margin, is
  what stopped the run. **`best_solution.json` is UNCHANGED and still holds `350f4c7`.**


**The previous three items — (a) the hub-fillet cap, (b) the production descent, (c)
`make m9` in full — are ALL DONE. §5, §6 and §7 are their records.** A one-paragraph
summary of each is at the end of this section; read those three sections for anything real.
What follows is the new list, and it is ordered by *cost to decide*, not by importance.

**The new items are NUMBERED (1)(2)(3) and the closed ones keep their LETTERS (a)(b)(c)**,
because §5, §6 and §7 are titled after the letters and a fresh session reading "§0(b)" must
land on the production descent, not on the wall-thickness sweep. Letters are history;
numbers are open.

**~~(1) SETTLE WHY `lambda_min` IS MESH-DEPENDENT.~~ DONE. BOTH HYPOTHESES RUN. The answer
is that `study_m9`'s `K_t` is not a tangent at all — and a mesh-convergent replacement
exists and was measured.** Full record in H1 and H2 below; the two-line version:

- **`kinematics="linear"` is the default on `wheel_contact_problem`, so `prob.nonlinear` is
  False and the displacement passed to `assemble_stiffness` is IGNORED — measured at exactly
  0.000e+00 change.** §7's quantity is λ_min(K_linear + K_contact), which has no geometric
  stiffening in it. Every symptom §7 recorded follows from that one fact.
- **The generalised load factor converges: 1.378129 / 1.359846 / 1.356669 at smoke / coarse /
  medium**, error ~dof^−1.87, Richardson limit ≈1.3560, `medium` within 0.049% of it. It
  passes `GATE_MESH_REL` with 21× margin where λ_min fails it by twelve.

**What is now open is phase 3 itself**, plus four things H2 did NOT establish — see the end
of H2. **The load factor came out ≈1.36 on one design, which is alarmingly tight and must not
be quoted as a safety factor until it is independently checked.**

The original framing follows, because the reasoning is what made the cheap test worth running:

**~~H1 — THE CONTACT PENALTY.~~ RUN, AND REFUTED. It also corroborated H2.** `best_solution`,
`coarse`, phase 0, `eps_n` over two decades, ~2 min. Three independent discriminators, all
negative:

| eps_n | λ_min | vs default | λ_min, contact block REMOVED |
|---|---|---|---|
| 1e3 (0.1×) | 4.860825e-03 | 1.0069× | 4.703656e-03 |
| 3e3 | 4.894088e-03 | 1.0138× | 4.703656e-03 |
| **1e4 (default)** | **4.827619e-03** | 1.0000× | 4.703656e-03 |
| 3e4 | 4.851094e-03 | 1.0049× | 4.703656e-03 |
| 1e5 (10×) | 4.882208e-03 | 1.0113× | 4.703656e-03 |

1. **`eps_n` varied 100×; λ_min varied 1.0044×** — and *non-monotonically*, so even that
   0.4% is solve-to-solve variation rather than a trend. A penalty mode tracks its penalty
   roughly linearly. This does not track it at all.
2. **Ablating the contact block entirely moves λ_min by 2.6%**, and the ablated value is
   identical to 7 digits at every `eps_n`. The contact term is a couple of percent of the
   quantity; it is not the quantity.
3. **The eigenvector is a GLOBAL mode.** 5.05% of its energy sits on `rim_outer`, which is
   2.28% of the nodes — concentration factor 2.2, i.e. essentially none. The **top 10 nodes
   hold 0.11%**. Participation ratio **0.54**: the mode is spread over roughly half of every
   node in the mesh.

**Point 3 is why this was worth the two minutes even though it came back negative.** It is
independent corroboration of H2 from a different direction: λ ∝ h² is what the smallest
eigenvalue of an unnormalised discrete elliptic operator does, and "a global mode spread over
half the mesh" is what that mode looks like. The scaling exponent and the eigenvector now
agree. **H2 is the remaining explanation and it is no longer competing with anything.**

The driver is `eps_n_check.py`, written to a scratchpad rather than to `studies/` on purpose:
it is a one-question falsification with no gate and no artifact anyone should depend on, and
its answer is this table. Re-deriving it is ~2 min if it is ever doubted.

**~~H2 — THE FORMULATION.~~ RUN, AND CONFIRMED. A mesh-convergent buckling quantity EXISTS,
and it is cheap.** But the check that came with it found something bigger, so read that first.

##### H2(a) — `study_m9`'s "K_t" IS NOT A TANGENT. It has zero geometric stiffening in it.

`wheel_contact_problem` defaults to **`kinematics="linear"`**, so `prob.nonlinear` is
**False**, and `study_m9.measure()` never overrides it. `assemble_stiffness`'s own docstring
says the linear Hessian "is independent of `u`, so `u=None` (evaluate at zero) is exact
rather than an approximation." **Measured, not inferred** — assemble at the converged `u`
and at zero and diff:

| config | \|u\|max | K(linear) change with u | K(svk) change with u |
|---|---|---|---|
| smoke | 1.6153 mm | **0.000e+00** | 2.732e-02 |
| coarse | 1.6663 mm | **0.000e+00** | 8.769e-03 |

**Exactly zero.** The `res["u"]` that `study_m9` threads into `assemble_stiffness` is inert;
the kernel ignores it. So the quantity §7 measured is **λ_min(K_linear + K_contact)** — the
smallest eigenvalue of a fixed linear elliptic operator plus a penalty block. It was never a
tangent eigenvalue, and it contains **no buckling information by construction.**

That single fact explains every symptom §7 recorded, with nothing left over:
- **h², to within 3%, nine times** — that is what λ_min of a fixed linear operator does.
- **1.022× over a 4× load ladder** — the only load path into it is the contact block, and H1
  measured that block at 2.6% of the value.
- **A global mode, participation ratio 0.54** — a bulk discrete mode, not a structural one.

The three findings were never independent. They are one defect seen from three sides.

##### H2(b) — the generalised load factor, and it converges

Under **SVK kinematics** so a geometric term exists at all: `K_0 = K(u=0)`,
`K_t = K(u_service)`, `K_g = K_t − K_0`, then `(K_0 + λ K_g)x = 0` solved as the generalised
symmetric problem `K_g x = μ K_0 x` with `λ = −1/μ`. `λ` is a dimensionless **load factor** —
how many times the service load until the tangent goes singular.

| config | reduced dof | **load factor** | vs previous | `study_m9`'s λ_min, same mesh |
|---|---|---|---|---|
| smoke | 8,904 | 1.378129 | — | 2.275954e-02 |
| coarse | 41,064 | 1.359846 | **−1.327%** | 4.827619e-03 (−78.8%) |
| medium | 104,712 | 1.356669 | **−0.234%** | 1.921939e-03 (−60.2%) |

**Converging, and fast.** Successive changes are −0.018283 then −0.003176, ratio **5.76**, so
the error goes as **dof^−1.87**. Richardson-extrapolating puts the limit at **≈ 1.3560**, and
`medium` is **0.049%** from it. Against `GATE_MESH_REL = 0.05` the coarse→medium step passes
with **21× of margin**, on the same meshes where λ_min(K_t) fails it by twelve.

**So M9 phase 3 has a quantity.** It is dimensionless, mesh-convergent, physically meaningful,
constructible from existing functions with no new assembly, and costs 3.3 / 23.9 / 71.4 s at
the three rungs. That is the whole of what item (1) set out to establish.

##### What this does NOT establish — four things, before anything is built on it

1. **One design, one phase.** `best_solution` at phase 0.0 only. §7 measured λ_min's phase
   spread at 0.66–1.59%; the load factor's is **unmeasured**, and so is its spread across the
   16 elites. Both are needed before a threshold or a phase aggregation rule.
2. **THE LOAD FACTOR IS ≈ 1.36, AND THAT IS ALARMINGLY TIGHT.** It says this wheel buckles at
   36% above service load. That is either a real and important design finding — the Euler
   `buckling` proxy has been reporting ratios of 0.066–0.087 and would not have shown it — or
   the `K_g ≈ K_t − K_0` approximation is off. **Do not report 1.36 as a safety factor until
   it is checked against something independent.** It is the most consequential number this
   session produced and the least corroborated.
3. **`K_g` is linearised about the SERVICE state, not the reference state.** Textbook
   eigenvalue buckling prestresses `K_g` at a reference load and scales it linearly; this
   forms it at the converged service state instead. For a load factor near 1 the two nearly
   coincide, which is convenient here and would stop being true for a stiffer design.
4. **Contact is excluded from `K_0` and `K_g`.** Buckling is computed on the bulk only, while
   the load actually arrives through the contact patch. H1 puts the contact block at 2.6% of
   the old quantity; its effect on the *load factor* is unmeasured.

**Both hypotheses were worth running, and the cheap one paid twice.** H2 was always the more
likely — h^1.94 to h^2.03 across nine refinements is a bulk discrete mode, not a localised
penalty — but that was **inference from a scaling exponent**, and inference is what
`stress_scale` was. H1 died in two minutes and its eigenvector diagnostic turned out to be
positive evidence *for* H2 rather than mere absence of evidence against it; then H2's own
setup surfaced H2(a), which no amount of reasoning about eigenvalue scaling would have found.

Drivers: `eps_n_check.py` and `h2_check.py`, both in a scratchpad rather than `studies/` —
one-question falsifications with no gate and no artifact anything should depend on. Their
answers are these tables. Promoting `h2_check.py` into a real study driver is the FIRST
piece of phase-3 work, because items 1–4 above all need it.

**(2) PUT A NUMBER ON `MIN_WALL_MM`. The only item here that changes the WHEEL rather than
the model.** §6 measured that all four thickness genes pin to the 2.0 mm floor at both
production answers, so a manufacturing constant — not the FEA, not the deflection target, not
the stress constraint — sets **4 of the 14 genes**, and every gram below 58.660 is on the far
side of it. It sits on "The decision that is a human's" with **no quantification at all**,
which is not a decidable state.

  **The experiment: short descents at `MIN_WALL_MM` ∈ {1.6, 1.8, 2.0, 2.2}**, from the elite-10
  answer rather than from `rank:10`, and **well short of 300 steps.** Measured off elite 10's
  record, distance from its own final loss: **step 100 is +0.190%, step 125 +0.096%, step 150
  +0.048%, step 200 +0.011%.** So ~125 steps costs ~1.6 h and resolves to a tenth of a
  percent — far finer than a 0.2 mm floor change will move the mass. That turns "should the
  floor come down?" into "0.2 mm of floor buys N grams."

  **`MIN_WALL_MM` is NOT a parameter, and this is a code change before it is a run.** It is
  `src/wheel_fea.py:219`, and it is consumed at **import time** by the `GENE_BOUNDS` list
  literal at lines 259–262 (`t0`/`t1`/`t2`/`t3` low bounds). So it cannot be varied inside one
  interpreter without rebuilding the bounds; the sweep is either four separate processes with
  the constant overridden per process, or a small change making the floor an argument that
  `GENE_BOUNDS` is built from. **Prefer the latter and drive the sweep from one place** —
  editing a module constant four times by hand is how a run gets misattributed to the wrong
  floor. Four points, sequential and capped, is ~6.5 h; see the memory rules below.

**(3) DECIDE WHETHER THE PRODUCTION GENOME BECOMES THE SHIPPED GENOME. A human's call, and
it is the pending consequence of §6.** `best_solution.json` is still the GA optimum for the
BEAM surrogate — the genome this very file calls "a bad guide to the FEA," sitting at
**−25.43% deflection error** — and **both environments and every study driver read it**.
`stage3_prod_best_elite10.json` is 58.660 g, inside the feasible box on both constraints,
with every barrier at exactly 0.0.

  Promoting it is not a file copy. It needs `make export` (~4 min) and then a look at whether
  the hub fillets still build on the new geometry: §3's ladder and §5's cap are both
  genome-dependent, the new `R_hub` is **0.9666 against a cap of 1.0400** (under it, which is
  the good direction), and `kt_error_pct`'s +73.4% was set by a shallow near-cusp corner the
  cap does not model. Nobody has looked at 58.660 g of geometry in OCC. Expect the manifest
  to move, and re-measure rather than assume.

**HOUSEKEEPING, stated because it was skipped rather than passed.** `make test` has NOT been
run since §7's changes to `studies/study_m9.py`. Nothing outside the `Makefile` imports that
module — verified by grep over `tests/`, `src/` and the `Makefile` — and `--quick` exercised
every changed path before the full run, so the exposure is close to nil. It is still the
repo's gate and it is still unrun. ~22 min.

#### What just closed, in one line each

- **(a) `R_hub` vs the buildable ceiling — §5.** The constraint learned the cap rather than
  the bound coming down to 1.1: the void is a function of arrival angle and `t0`, spanning
  **0.9898–1.5265 mm** across the 16 elites, so a fixed bound would have been right for
  exactly one genome. The 4.0 box bound is untouched.
- **(b) The production multi-start descent — §6.** `make prod9` / `make prod10`. **73.689 →
  58.715 g (elite 9, −20.3%)** and **70.937 → 58.660 g (elite 10, −17.3%)**, every barrier at
  exactly 0.0, deflection error inside ±0.3% on both. The two land **0.09% apart in mass with
  1.8 mm of daylight between their spoke centerlines** — the optimum is a **valley, not a
  well**, so more starts are pointless and the leftover geometric freedom is the interesting
  part.
- **(c) `make m9` in full — §7.** **OVERALL: FAIL, and the failure is the result.** λ_min(K_t)
  has no mesh-independent value; phase 3 cannot be built as specified. `buckling` stays inert,
  and that is now a deliberate hold rather than an oversight — item (1) above is how it gets
  unblocked.

**THE MEMORY RULES BELOW DESCRIBE THE OLD 16-CORE / 31 GB BOX — see §8 for this one.**
The current machine is **24 cores / 61 GB**, one descent measured at a **13.56 GB peak**,
and three arms ran concurrently under 15 GB caps with no oomd event. What did NOT travel
is the speedup: three concurrent descents run at **~95 s/step each against ~38-47 s alone**,
because the binding resource is **memory bandwidth** (twelve concurrent `spsolve` LUs),
not cores or capacity. Keep the `systemd-run` caps regardless — they make a failure local
instead of taking the desktop's whole `user@1000.service` slice with it.

**THE ORIGINAL RULE, AS MEASURED ON THE OLD BOX:** A descent at `coarse` with 4 workers
sits flat at **~12.7 GB anonymous** with the fidelity check off; **two do not fit in 31 GB**,
so starts run **SEQUENTIALLY** and **capped** (`systemd-run --user -p MemoryMax=20G`). §1's
"two starts × 4 workers beats one start × 8" is CPU arithmetic that ignores memory, and
acting on it is what took the desktop down. The 17 GB quoted there was measured with the
fidelity check ON and before the `uniform` finding; `PROD_FIDELITY` now defaults to **0**.
`--phase-scheme uniform` is a **correctness** setting for the run, not a preference — see §1.

**The STEP consumer is Autodesk Inventor now, not Onshape.** Onshape was temporary. The
`wheel.step` interop history in `wheel_step_export.py` is Parasolid's, and Inventor is a
different kernel, so that record is history rather than a live constraint and re-testing the
Onshape import is NOT on this list. What still travels: `despecialize` exists because a
`SURFACE_OF_LINEAR_EXTRUSION` is not universally supported and the hub bore and OD are kept
analytic on purpose, and `wheel_nofillet.step` is still written before any fillet is
attempted. Whether Inventor is happy with the 48 fillet faces and the 0.162 mm shortest edge
the hub fillet added is unmeasured — worth one import, not a milestone.

### 1. M8b-ii — make the optimizer runnable at scale

Feasible points exist; the optimizer could not search for better ones in reasonable time.
The phase batch is parallel, `t1_vector` is jitted, and the fidelity check exists, all
measured; only the production run itself is open.

- **~~Process-parallel phase batch.~~ DONE — `--workers`, gated by S13 (`make m8bii1`).**

  `src/wheel_pool.py` (`PhasePool`) + `src/wheel_pool_worker.py`. Measured at `coarse`,
  8 phases, on a **16-core** box — the hours below do not travel to another machine, the
  efficiency column does:

  | workers | s / eval | speedup | efficiency | values vs serial | grad rel |
  |---|---|---|---|---|---|
  | serial | 139.4 | — | — | — | — |
  | 1 | 136.7 | 1.02× | 1.02 | **exact** | 0.0 |
  | 2 | 76.4 | 1.82× | 0.91 | **exact** | 0.0 |
  | 4 | 47.6 | 2.93× | 0.73 | **exact** | 2.1e-16 |
  | 8 | 35.3 | 3.95× | 0.49 | **exact** | 2.1e-16 |

  **The production projection is 46.46 h → 11.77 h** (300 steps × 4 starts).

  `0` is serial and stays the DEFAULT, so every gate and every committed artifact still
  runs the path they were measured on. `-1` sizes the pool to `min(n_phase, cpu_count)`;
  `N` is taken literally and is the only cap on memory, since the auto-size counts cores
  and knows nothing about RAM.

  Slot `i` is pinned to worker `i % n`, which is why this is not `multiprocessing.Pool`:
  `coord_fn` keys its jit cache on `float(phase)` and a miss is 0.774 s, so a phase that
  wanders between workers pays that forever. Replies come back in **slot order**, so every
  reduction sums the same floats in the same sequence as serial.

  **`XLA_FLAGS` is pinned along with the four thread counts, and it is a correctness
  setting.** Measured: two *plain serial* runs of one `coarse` adjoint, in two separate
  interpreters with **no pool anywhere**, agree on every forward value to the bit and
  disagree on the GRADIENT by 3.33e-16 — XLA's CPU thread pool does not associate its
  reductions the same way twice, and nothing in OMP/MKL/OPENBLAS/NUMEXPR reaches it.
  Pinned, that comparison is exactly zero, and it costs nothing: 19.84 s against 20.43 s
  for one `coarse` phase. Set in the `Makefile`, in `conftest.py`, and in
  `wheel_pool.PINNED_ENV` for a worker started by hand.

  **S13 gates values EXACTLY and gradients at 1e-14, and the split is a measurement.**
  Exact bit-identity of the gradient is not available to *anyone* here, pooled or not —
  what remains after the XLA pin is process-history dependent (a process that has already
  run other phases answers the next one differently in its last bit) and lives below this
  repo, in XLA's own codegen. Observed 2.1e-16 against a 1e-14 gate. Every value, every
  report leaf and the stress and ripple gradients are 0.0.

  **The thread pin moved no physics**: `make m8bi6` re-run and diffed against the previous
  `study_stage3_pnorm.json` — **2038 non-timing leaves, 0 differ**, gate verdict unchanged.
  The artifact in the tree is the earlier one, restored, because the two differed only in
  wall clock and the earlier run's timings were measured on a quiet machine.

  **End to end**, 2 steps from elite 9 at `coarse`, `--phase-scheme uniform`, serial vs
  `--workers -1`: **639.4 s → 201.6 s (3.17×)**, every step's loss equal to the last bit
  and the same final `genome_hash` (`99fea84`), no events either side, no orphaned
  workers. 3.17× rather than S13's 3.95× because the T1 precheck and step 0's tracing are
  a larger share of a 3-call run than of a steady-state one.

  **Do not read that exactness as a guarantee.** A trajectory can diverge in its last bits
  whenever a gradient difference survives the Adam update into the accepted iterate; it
  happened not to at this design. The claim S13 supports is per-evaluation — values exact,
  gradients to 1e-14 — not per-trajectory.

- **~~Jit `t1_vector`.~~ DONE — `_t1_cached_value_and_jacobian`, `wheel_objective.py`.**
  1.06 s → 0.0054 s at `coarse` (S10), ~195×. Details above, in "M8b-ii item 2 also
  landed."
- **~~Multi-fidelity checkpoints.~~ DONE — `fidelity_check_every`/`fidelity_check_cfg`
  on `wheel_stage3.descend()`, `--fidelity-check-every`/`--fidelity-check-config` on the
  CLI.** There was no spec for this beyond the budget number above (`medium` is 2.8×
  `coarse`, 243 s vs 87 s, not the 4× once assumed — still true, taken from the elite-1
  ladder rather than the shipped genome's for the reason already stated: the shipped
  genome runs first and its `coarse` rung carries the `coord_fn` jit trace). Chose the
  narrowest reading: a PURE OBSERVATION, off by default. Every N accepted steps (and at
  step 0), the just-accepted iterate is forward-evaluated a second time at a second
  config, t3 tier only, and the gradient that call returns is discarded — it can never
  reach `m`/`v`/`delta`/`z`, so it can only report on the trajectory, not redirect it.
  Result goes on that step's row under `"fidelity_check"` (raw `report` dict, not a
  diff/ratio — no disagreement threshold is invented here, on purpose: this repo already
  removed one derived, unrevisited quantity, `stress_scale`'s old rescale role, once its
  assumptions stopped holding, and a guessed threshold would repeat that with no
  measurement behind it yet). A second-fidelity solve failure is recorded as a
  `fidelity_check_failed` event and does not abort the run. A scheduled step that lands
  on an abandoned step is skipped, not rescheduled. `descend_lbfgsb` is untouched —
  `--optimizer lbfgsb` isn't the default and its `fun(z)` is called on every line-search
  evaluation rather than only accepted steps, so "every N steps" would mean something
  different there. `make test` **391 passed** (384 before). Recommended CLI value for
  the production run below: `--fidelity-check-every 25` or `50` against
  `--fidelity-check-config medium`, adding roughly 49 or 24 minutes to an 11.77 h run.

  **MEASURED, and that estimate was ~3× too low: one check costs 604.6 s (10.1 min).**
  `settings.fidelity_check_solve_s` from the first production attempt, one call, `medium`
  against a `coarse` run. So every-50 over 300 steps is 7 checks ≈ **1.2 h**, not 24 min,
  and every-25 is ≈ 2.4 h. The estimate above assumed the check rides the phase pool; it
  does not — `descend` builds `ev_fc` as a **wholly separate serial `Evaluator`**, on
  purpose, so the check pays 8 phases end to end with no parallelism at all.

  **It also costs ~3.4 GB, permanently.** That second `Evaluator` retains its own jit cache
  and mesh for the life of the run, so the parent goes 4.5 → 7.9 GB across the FIRST check
  and stays there. On a memory-bound box this is the first thing to turn off:
  `--fidelity-check-every 0`, which is what `make prod9`'s `PROD_FIDELITY` exposes. The
  check is a pure observation, so dropping it costs evidence, never trajectory.
- **Then the production multi-start run.** Start from elites 9 and 10, not
  `best_solution.json` — that is a GA optimum for the BEAM surrogate, which M8a measured as a
  bad guide to the FEA, and it sits at −25.43% deflection error. The 16-elite spread is wide
  (util range 0.30, deflection range 105.66 points), so a multi-start genuinely samples
  different basins rather than re-running one.

  **The objective it should descend is now mass**, not feasibility. Both constraints are
  satisfiable together and the barriers are flat at every feasible design, so `mass` is the
  only term with anything left to give — it was 19.6% of the loss at the shipped genome
  against deflection's 61.3%, and that ratio inverts once deflection is met.

  **Run it with `--workers`.** At 11.77 h for 300 × 4 this is now affordable. But 8 workers
  is not obviously the right setting: 4 gets 2.93× at 0.73 efficiency on a quarter of a
  16-core box, so **two starts × 4 workers beats one start × 8** and finishes the same
  budget sooner. Decide that against the machine it actually runs on.

  **DO NOT DO THAT. The paragraph above is CPU arithmetic on a machine that runs out of
  MEMORY first, and it was tried.** Measured on the 16-core / 31 GB box: one descent at
  `coarse` with 4 workers holds **~17 GB anonymous** (parent ~4.5 GB with checks off, 7.9 GB
  with them on; four workers ~3.3 GB each) and worker RSS was **still climbing** at step 5
  (2.2 → 3.4 GB), so the steady-state ceiling is unknown. Two starts is ~34-42 GB against
  31 GB of RAM and a 2 GB swapfile.

  **That 17 GB is the WORST case and has since been beaten down to a flat ~12.7 GB** — it
  was measured with `rqmc` (64 retained jit traces, see below) and the fidelity check ON
  (+3.4 GB in the parent, permanently). With `--phase-scheme uniform` and
  `--fidelity-check-every 0`, which is what §6's completed run used and what the `Makefile`
  now defaults to, the footprint is flat rather than climbing and the ceiling is known.
  **Two starts still do not fit** — 25 GB against 31 GB and a 2 GB swapfile leaves nothing
  for the desktop — so the sequential rule stands on the smaller number too.

  What that costs is not a slow run, it is **the desktop**: `systemd-oomd` kills the entire
  `user@1000.service` slice when memory pressure holds above 50% for 20 s, which drops the
  user to the login screen AND kills every run and terminal in that slice. The kernel OOM
  killer is not involved and `dmesg` shows nothing — `journalctl -u systemd-oomd` is where
  the evidence is. So on a memory-bound host: **run the starts one at a time, capped**, e.g.
  `systemd-run --user --unit=wheel-prod9 -p MemoryMax=20G --collect make prod9`, which
  inverts the failure — the kernel kills the run at its own cap instead of oomd killing the
  session. `--workers` is then bounded by RAM/worker, not by cores.

  **RESOLVED: S13's 47.6 s/eval is right, and `rqmc` re-tracing was the entire gap.**
  Under `rqmc` the per-step cost wandered 132/130/128/**52** on one run and 180/135/132 on
  another, which read like noise. Under `uniform`, on the same box and the same 4 workers:
  step 0 is 177.8 s (tracing), then 55.3, 46.7, and a steady 44-47 s thereafter — median
  **48.7 s**, and the last five steps at 46.3/46.4/47.4/44.9/44.4. **300 steps is ~3.8 h.**

  So roughly **80 s of every `rqmc` step was JIT compilation, not solving.** The trace cache
  below explains the wall clock and the memory with one mechanism; there was never a second
  problem. Still read `settings.elapsed_s / n_objective_calls` off a run before projecting,
  but S13's ladder travels once the phase set is fixed.

- **THE PRODUCTION RUN MUST USE `--phase-scheme uniform`, AND THE REASON IS MEMORY.**
  This is the finding that killed the first two production attempts, and neither
  `--workers` nor the fidelity check is the lever on it.

  `rqmc` — the DEFAULT, and what S13 and every projection above assumed — redraws the
  stencil every step from the `n_sub`-point sub-lattice. Verified from a run record: step 0
  drew phases at 2.812°, 6.562°, ... and step 1 drew 2.344°, 6.094°, ..., all different. So
  a run visits `n_phase * n_sub` = **64 distinct phase values**, and `wheel_wheel.coord_fn`
  keys its jit cache on `float(phase)` (`_COORD_FN_CACHE_MAX = 128`, FIFO), so **all 64
  traces are retained and none is ever evicted**.

  Measured: ~0.4 GB per retained trace. Worker RSS climbed 2.2 → 3.9 GB over three steps
  while the parent sat flat at 4.5 GB, and the run peaked at **18.9 GB and was OOM-killed at
  step 3**. The pool holds all 64 traces however the slots are divided, so **`--workers 2`
  does not help** — it redistributes the same cache and saves only one process baseline.

  **`uniform` was then measured and it holds.** The cache saturates at 8 traces after step
  0, and the run sits FLAT at **12.73 GB anon through step 27** with `memory.events` all
  zero — the 20 GB cap is never approached, rather than approached slowly. That is the
  difference between a bounded phase set and an open one, and it is why this is a
  correctness constraint on the run configuration rather than a tuning knob.

  The jit cache was designed for WARMTH — M7 measured a miss at 0.774 s, which is why slots
  are pinned to workers at all (§ the phase pool above). Nothing sized it against a stencil
  that keeps producing new keys. `uniform` is not a workaround here: it is the scheme
  `make m8bii1`'s end-to-end pooled-vs-serial check already ran on.

  **What it costs is the randomised quadrature**, and that is a real change to how the
  phase average is estimated, not a free win. It is a deliberate trade of variance
  reduction for a run that can physically complete.

### 2. M9 — `lambda_min(K_t)` via LOBPCG, replacing the Euler `buckling` proxy

**This milestone promoted M9.** `buckling` has a gradient of exactly 0.0 and is asserted zero
by the inert-term census (`INERT_EXPECTED = ("buckling",)`). With stress no longer binding, it
is **the only constraint left in the objective that a gradient method cannot act on**. A
diverged tangent is the only real buckling signal the run has today.

**M9 phase 1: all three new tests pass, including contact penalty smoothing.**

**M9 PHASE 2 HAS RUN IN FULL (§7), AND §0(1) THEN FOUND WHY IT FAILED — THIS SECTION'S TITLE
IS NOW WRONG.** `lambda_min(K_t)` diverges under refinement at a clean h², 28.8× across the
ladder against 1.02× across a 4× load ladder, with an independent `eigsh` cross-check
(3.4e-10) ruling out the solver. The cause is that **there is no `K_t`**:
`wheel_contact_problem` defaults to `kinematics="linear"`, so the displacement threaded into
`assemble_stiffness` is ignored — measured at exactly 0.000e+00 — and the quantity is
`lambda_min(K_linear + K_contact)`.

So the paragraph above still describes the *problem* correctly — `buckling` is inert and a
diverged tangent is the only real signal today — but **"via LOBPCG" was never the hard part
and the standard eigenproblem was the wrong question.** The replacement is the generalised
`det(K_0 + λ·K_g) = 0` under SVK kinematics, whose load factor **is** mesh-convergent
(1.3560 in the limit, `medium` within 0.049%, 21× inside `GATE_MESH_REL`). §0(1) H2 is the
measurement and lists the four things it does not yet establish.

### 3. ~~The hub fillet~~ — DONE as geometry, and it turned up a bound the genome cannot honour

**The hub junction exists and all 24 of its corners are filleted** (it was 0 of 12). Full
record in **`HUB_PLAN.md`**; the short version:

`_embed`'s inward step took the *least rotation from the junction tangent* that reached the
hub — a 4.516 mm run that swung the root cap **22.3° out of a 30° sector**, so adjacent spokes
lapped over the hub circle before either reached it and the circle stopped existing. Measured:
at r = 12.71 and r = 12.80 the ring is 360° material; the first void is a 0.16° sliver at
r = 12.8748, which is the 354° notch OCC was refusing. There was no spoke↔hub junction to
fillet. It now **plunges radially** — 1.788 mm, 0.57° — and the 24 corners are back on
r = 12.700, worst wedge 332°.

`fillet_junctions` also stopped abandoning the leftovers: it re-selects the corners that
refused a rung and walks the ladder again for them, so both flanks get a radius. That was
never a hub-only bug — it is why the **rim** shipped 12 of 24.

| junction | found | filleted | families | R worst | Kt model | Kt built | error |
|---|---|---|---|---|---|---|---|
| hub, before | 12 (off-circle) | **0** | — | 0.000 | 1.861 | 3.500 | +88.1% |
| hub, after | **24** | **24** | 12 @ 1.127, 12 @ 0.361 | 0.361 | 1.861 | 3.228 | **+73.4%** |
| rim, before | 24 | 12 | 12 @ 3.000 | 3.000 *(as priced then)* | 1.490 | 1.490 | +0.0% |
| rim, after | 24 | **24** | 12 @ 3.000, 12 @ 0.308 | 0.308 | 1.490 | 3.149 | **+111.4%** |

**The pricing decision changed with it: a junction is priced by its WORST corner.** The old
rule priced the radius that *was* applied, which is how the rim reported +0.0% while twelve of
its corners shipped square. The rim therefore looks worse than it ever has while its geometry
strictly improved; `fillet_families` keeps every per-family radius, so nothing is lost.

**The new finding, and it is a human's decision.** The void between adjacent spokes at the hub
circle is **9.907°, 2.196 mm of arc**, narrowing outward. Two fillets growing into that slot
can each take about half of it — OCC accepted **1.127 mm**, against 1.098 = half the gap. So
**`R_hub` cannot be built above about 1.1 mm on this genome and its bound is 4.0 mm.** The
constraint can price a fillet the part can never build, which is the same class of discrepancy
this milestone existed to remove, one level up. Either the bound comes down, or the constraint
learns the cap, or the spoke count/width changes.

Knock-ons, both bookkeeping: `EMBED_ALLOWANCE_PER_SPOKE_MM2` re-measured 4.27 → **3.03** (the
radial plunge leaves less gusset in the annulus), narrowing the deliberate mesh-vs-STEP AREA
gap −1.93% → **−1.384%**; and the fillets stopped being negligible — 0.29 mm² of cross-section
when 12 of 48 corners were built, **24.28 mm² (0.92%)** now that all 48 are — so the MASS gap
is a different number from the area gap, **−2.277%**, and the difference between them is the
fillet material. Export cost went 14.6 s → 230.5 s; `wheel.step` carries 111 faces against 75
and a 0.162 mm shortest edge. OCC calls it valid, self-intersection clean, no degenerate
edges; no other kernel has looked at it since (Inventor is the consumer now — see §0).

### 4. Minor, known, pre-existing — DONE

**~~`study_stage3.py --quick` exits 1 on S8.~~ FIXED — scoped to `coarse`, not more reps
at `smoke`.** The old failure (cold 6.16 s vs warm 6.32 s, −2.6%) was a `smoke`-tier
noise-floor artifact — the 960-element solve is too small a share of an evaluation
dominated by meshing/dispatch to resolve the warm-start saving at all, so more reps
would have chased a signal too small to measure rather than fixed anything. `run_warm`'s
section-registry entry now always calls it at `DEFAULT_CONFIG` ("coarse"), regardless of
`--quick` — "the gate that counts," per the diagnosis already on file. Re-measured under
`--quick --sections warm`: cold 43.3 s, warm 39.1 s, **saving 9.8%, PASS** (up from the
`smoke`-tier +2.4% quoted before, both comfortably above `GATE_WARM_SAVING = 0.0`, left
untouched). The section's report and printed output now carry which config it ran at,
so this doesn't silently look like a `smoke` number in a `--quick` run. No test asserted
the old behavior, so `make test` is unaffected; **391 passed** afterward.

### 5. §0(a) — the constraint learned the buildable hub fillet cap. DONE

`R_hub`'s box bound is still 4.0 mm. What changed is that the loss now knows what the part
can build, in two places, and the number is computed from the genome rather than remembered
from one export.

**The geometry.** `wheel_wheel.hub_void_deg` measures the empty arc adjacent spoke roots
leave on the hub circle — `SECTOR_DEG` minus the arc one root occupies, from `ring_station`
and an `arctan2`, the same arithmetic as `weld_footprints_deg`. It lives in `wheel_wheel`
because that module is jax-free and therefore importable in `.venv-cad`, which is what would
make "let the exporter consume the cap too" a small change later.

**That part is validated.** `make hubcap`'s `void` section classifies 14400 points on a ring
just outside the hub circle in OCC and measures the empty runs. Analytic against measured,
on three designs: **9.977 / 9.825, 8.931 / 8.800, 13.774 / 13.575** — 0.13 to 0.20° apart,
exactly the residual `hub_void_deg`'s docstring predicts from `_embed` plunging radially
from the centerline endpoint rather than the flank endpoint. Twelve runs every time.

**THE CAP MODEL IS A `min` OF TWO LIMITS, AND THE GATE IS WHY.** It was written as
`0.5 × slot` alone, on the strength of the hub fillet milestone's one recorded export
(void 2.196 mm, OCC built 1.127). Bisecting what OCC will *actually* accept on each of the
24 hub corners falsified that:

| design | void | slot arc | `0.5×arc` | **OCC, bisected** | t0 |
|---|---|---|---|---|---|
| best_solution | 9.977° | 2.2115 | 1.1057 | **1.3000** | 2.4774 |
| elite14 | 8.931° | 1.9795 | 0.9898 | **1.3445** | 2.5536 |
| elite13 | 13.774° | 3.0531 | 1.5265 | **1.3340** | 2.5536 |

The 1.127 on file was a **ladder rung accepted for a whole twelve-edge family**, not the
limit. And the limit does not track the slot: the void spans a **54% range** while the
threshold moves **3.4%**. elite14 and elite13 have *identical* `t0` and thresholds 0.7%
apart across the widest void gap in the set. It tracks **`t0/2`**, to 0.7%.

So `hub_fillet_cap_mm` is now `min(0.5 × slot_arc, 0.52 × t0)`. The slot term is kept, and
kept at an unvalidated 0.5, because it is the only one that knows a *closed* slot admits no
fillet at any radius — as adjacent roots converge the void goes to zero and then negative,
and the thickness term cannot see that at all. Both facts are in the constants' docstrings,
labelled as measurement and as assumption respectively.

**`make hubcap` PASSES, and the split confirms the `min`:**

| design | binding limit | cap | OCC | cap/OCC |
|---|---|---|---|---|
| best_solution | slot | 1.1057 | 1.3030 | 0.849 |
| elite14 | slot | 0.9898 | 1.3384 | 0.740 |
| **elite13** | **thickness** | 1.3279 | 1.3274 | **1.000** |
| **t0 = 2.0** | **thickness** | 1.0400 | 1.0395 | **1.000** |

Where the thickness term binds it predicts what OCC accepts to **0.05%**. Where the slot
term binds — the deliberately unvalidated 0.5 — it is conservative by 15–26%, which is the
harmless direction. That asymmetry is the model's shape, stated honestly rather than tuned
away.

**The thickness law is calibrated on `t0 ∈ [2.0, 2.6]` and is not known outside it.** Five
points there (0.5197 / 0.5254 from the sweep, 0.5247 / 0.5263 / 0.5224 from the designs) sit
in a 1.3% band. The sweep's rows above `t0 ≈ 3` report 0.63–0.94, and they are not evidence
against it: the void has collapsed and gone negative by then (−0.92° at `t0` = 6, −9.51° at
10), adjacent roots have merged, and there is no spoke-to-hub corner left for the number to
be about. Those rows are marked `same_feature: false` and excluded from the fit. In that
regime the slot term is negative and takes the `min` anyway — the cap is right there for a
different reason: no fillet exists at any radius, which is what a negative cap says.
`MIN_WALL_MM` is 2.0 and every design on disk sits at 2.468–2.627, so the calibrated band
covers everything the GA has produced.

**The gate is ONE-SIDED, and that is the model's actual claim.** It began as a two-sided
"within one ladder rung of OCC", written when `0.5 × slot` was believed to *be* the limit. A
`min` of one measured limit and one unvalidated one does not claim to be tight, it claims to
be **safe** — and the way to pass a tightness gate that a safe model fails is to loosen the
model until it passes, which is backwards. So: `cap/OCC ∈ [0.5, 1.01]`. Never promise more
than the part gives; and do not collapse to something vacuously small, since a cap of zero
would sail through a pure one-sided test while destroying every hub fillet in the wheel.

**Both halves of the fix, because either alone leaves half the defect.**
`fillet_cap = soft_barrier(R_hub − cap, 500)` pushes the gene under the slot; and `_kt_hub`
prices `Kt` on `smooth_min(R_hub, cap)` so the stress constraint stops crediting a fillet
that will not exist. `Kt_hub` moved **1.8609 → 2.0766, +11.6%** — the part is sharper than
the gene was asking for, and now the loss says so.

**`junction_kt` is no longer pure 14-vector gene space, and that is the change.** The cap
depends on the eight centerline genes and `t0` through `ring_station`, so `Kt_hub` does too.
It is **differentiated, not frozen** — freezing it and applying the chain rule by hand is
exactly the `stress_scale` failure mode this module already paid for once. Still no mesh and
no solve: it is a sampled curve, jitted and cached on `(cfg.name, span_mm, flanks)` the same
way `t1_vector` is. The consequence worth knowing: **`dKt_hub/dR_hub` is now EXACTLY 0.0**
at the shipped genome, because it sits 2.7 ladder rungs above its cap and buying more `R_hub`
there buys nothing. Below the cap the original physics is back (−1.99).

**`make hubcap` does not read the ladder, and that is a measurement too.** The obvious
criterion — "the largest ladder rung below the cap is what gets built" — is **false**: the
rungs from 1.5598 are 1.5598 / 1.3258 / 1.1269 / 0.9579, the largest under the 1.1057 cap is
0.9579, and OCC takes 1.3000. The rungs straddle the cap and which side they land on is an
accident of where `R_hub` starts. So the driver bisects the threshold where it actually is.

Four sections: `void` and `occ_limit` gate; `t0_sweep` is the **calibration** and is
deliberately not gated (gating a section on the constant it exists to measure is circular);
`sweep` is the picture. The three gated designs are chosen for different reasons — the
shipped genome, elite14 (tightest slot, 0.9898), and elite13, the one design already under
its cap and therefore a negative control rather than a fourth confirmation.

**`t0_sweep` exists because the disk cannot calibrate this.** All 17 designs sit at
`t0` between 2.468 and 2.627 — **6% of a box that runs 2.0 to 10.0** — so measuring more of
them says nothing about whether the thickness law holds at any other thickness. Sweeping
`t0` on one fixed shape does, and it does better than that: a thicker root leaves a
*narrower* void, so the two limits move in opposite directions and their crossover is
observed rather than assumed. That crossover is the entire justification for the `min`.

**15 of the 16 elites are above their cap**, including elites 9 and 10, which are the two
starts §0(b)'s production run begins from. That is what makes this a prerequisite rather
than a tidy-up.

**Verified.** `make test` **406 passed** (396 before). `make hubcap` PASS. Gate 7
(`study_objective.py --quick`) **OVERALL: PASS** — and it is the independent check on the new
term's gradient, because G4 finite-differences every entry of `T1_NAMES` generically:
`fillet_cap` agrees to **3.098e-09** over 11 live genes, which is the assurance that
differentiating through `ring_station` and the `min` is right. G8's inert census is still
`[] ⊆ ("buckling",)`. `make export` was not re-run and must not need to be: nothing on the
CAD side changed, and the manifest is byte-identical.

**A 4-step `coarse` descent from the shipped genome does what it should, and exercises BOTH
branches of the `min` on the way:**

| step | R_hub | cap | R_eff | Kt_hub | `fillet_cap` | t0 | binding |
|---|---|---|---|---|---|---|---|
| 0 | 1.5598 | 1.1057 | 1.1057 | 2.0766 | 103.07 | 2.4774 | slot |
| 2 | 1.4949 | 1.1705 | 1.1705 | 1.9967 | 52.60 | 2.3290 | slot |
| 4 | 1.4722 | 1.1875 | 1.1875 | 1.9748 | 40.53 | 2.2836 | **thickness** |

`R_hub` walks down, the cap walks *up* as `t0` thins, the barrier decays 103 → 41, and the
binding limit crosses from slot to thickness with no discontinuity — `0.52 × 2.2836 = 1.1875`
is the step-4 cap exactly. The gap is closing rather than closed (0.454 → 0.285 mm in four
steps on a decaying `lr`), which is what four steps should look like.

**Two things this does NOT do, stated so nobody expects them.** It does not move
`kt_error_pct` off +73.4%: the manifest's `r_built = 0.361 mm` is set by the SHALLOW
near-cusp corner, an arrival-angle limit the cap does not model. And it does not touch
`stress_concentration_kt`, the exporter, or the GA's numpy `spoke_overlap_penalty` — the
cap is applied by the caller, so the Kt twin stays bit-identical to `wheel_fea`'s and
`test_golden.py` does not move.

**Open, and flagged as the natural next item: `hub_overlap` is now subsumed.** Its chord
proxy wants `t0 + 2·R_hub + 1.3 ≤ 6.574` and reports a **+0.323 mm violation** on a wheel
that has been printed, while the true void at the same design is **+2.20 mm of clearance** —
it assumes the root sits square across the sector, and the shipped root arrives 10.5° from
tangent. `void < 0` already is "adjacent spokes overlap", so the new barrier covers the
collision case too. It was left in place deliberately: it is 47.9% of the T1+T2 value and
98.7% of its gradient norm, it is named in `wheel_stage3`'s deadlock post-mortem, and five
study drivers plus `tests/test_mesh.py` read it as a feasibility filter. Retiring a
48%-of-the-loss term inside a bound fix is how the `stress_scale` problem happened. The
`sweep` section of `make hubcap` prints the head-to-head that would justify it.

### 6. §0(b) — the production descent. DONE, BOTH STARTS. It found a 20% lighter wheel.

`make prod9`, `--start rank:9 --steps 300 --workers 4 --phase-scheme uniform
--fidelity-check-every 0`, `coarse`, 8 phases, seed 0. Record: `stage3_prod_elite9.json`,
genome `stage3_prod_best_elite9.json`. **7051.5 s for 150 objective calls = 47.0 s per
evaluation**, which is S13's 47.6 s ladder entry to within 1.3% — the projection travelled.

**Stopped by the operator at step 149 of 300, on convergence.** Loss improved 240.96 over
steps 0–50, 0.274 over 50–100 and 0.031 over 100–149; the last nine steps moved it 3.8e-03.
Zero rejects, zero abandoned steps, `events` empty.

| | step 0 (elite 9 as scored) | step 149 | |
|---|---|---|---|
| loss | 291.036 | **49.771** | |
| mesh mass | 73.689 g | **58.715 g** | **−20.3%** |
| axle drop, mean | 2.0491 mm | 1.9942 mm | **+2.46% → −0.29%** |
| stress utilisation | 0.5444 | 0.6018 | feasible at ≤ 1.0 |
| `hub_overlap` | 107.736 | **0.0** | |
| `fillet_cap` | 118.184 | **0.0** | |
| `Kt_hub` | 2.0978 | 2.0627 | |
| hub fillet cap | 1.1061 mm | 1.0400 mm | thickness-binding |
| `R_hub` effective | 1.1061 (**at** cap) | 0.9106 (**under** cap) | |
| buckling ratio | 0.0871 | 0.0663 | |
| min scaled Jacobian | 0.8892 | 0.8892 | |

**Every barrier in the objective is exactly 0.0 at the answer.** `x_order`, `hub_overlap`,
`fold`, `arrival`, `fillet`, `fillet_cap`, `buckling`, `min_sj`, `stress`, `phase_ripple` —
all of them. Three terms are left: **mass 48.259 (96.96% of the value, 44.4% of the gradient),
deflection 0.0214 (0.04% of the value, 46.5% of the GRADIENT), smoothness 1.491 (3.0% / 9.2%)**.

That split is the whole result. **Deflection is worth four hundredths of a percent of the loss
and nearly half of its gradient**: it is not paying for anything, it is *holding mass up*. The
descent is riding the deflection constraint down to the lightest wheel that still meets it,
which is exactly the regime §1 said the run would enter — "19.6% of the loss at the shipped
genome against deflection's 61.3%, and that ratio inverts once deflection is met."

**THE FOUR THICKNESS GENES ALL SATURATE AT THE LOWER BOUND, and that is the finding to argue
about.** `final.bound_saturation` is `t0, t1, t2, t3` all `low` at **2.0**, which is
`MIN_WALL_MM`. Minimising mass under a deflection constraint makes the wheel **as thin as it
is allowed to be everywhere**, and buys its stiffness back entirely from the eight centerline
genes. So the reported 58.715 g is not "the lightest wheel in this genome" — it is **the
lightest wheel at the 2.0 mm manufacturing floor**, and the floor, not the physics, is what
set four of fourteen genes. Whether 2.0 mm is the right floor for the actual process is a
human's question, and it is now worth about 20% of the mass. It belongs on the list below.

**§5 earned its place here.** Elite 9 started **above** its buildable hub fillet cap —
`fillet_cap` was 118.18, the second-largest term in the starting loss at 40.6% — and the
descent drove `R_hub` from 1.1061 (pinned at the cap) to 0.9106, comfortably under a cap that
had itself moved to 1.0400 as `t0` thinned. Without §0(a) this run would have spent 300 steps
buying a fillet the part cannot build. `hub_overlap` went 107.74 → 0.0 the same way, which
also answers §5's open question in one direction: that term is **satisfiable**, not binding at
the optimum, so retiring it is still optional rather than urgent.

### Elite 10 ran the full 300 steps, and the answer is a FLAT VALLEY, not a point

`make prod10`, same flags, a clean 300 from `rank:10` — the first attempt had been stopped at
step 83, well short of converged. **13656.6 s / 301 calls = 45.4 s per evaluation**, zero
rejects, zero abandoned steps, `events` empty, `Result=success` at a **13.7 GB** unit peak
against the 20 GB cap and **0 B** swap. Written by `main()`, so unlike elite 9's this genome
needed no reconstruction (`genome_hash eddcfc2`).

| | step 0 (elite 10 as scored) | step 300 | |
|---|---|---|---|
| loss | 308.847 | **49.7376** | |
| mesh mass | 70.937 g | **58.660 g** | **−17.3%** |
| axle drop, mean | 2.0379 mm | 1.9941 mm | **+1.90% → −0.30%** |
| stress utilisation | 0.5444 | 0.5904 | feasible at ≤ 1.0 |
| `hub_overlap` / `fillet_cap` | 159.194 / 87.010 | **0.0 / 0.0** | |
| `R_hub` effective | 1.2251 (**at** cap) | 0.9666 (**under** cap 1.0400) | |

Same shape of answer as elite 9: every barrier exactly 0.0, all four thickness genes pinned
low at 2.0, and the surviving split **mass 96.94% of the value / 44.0% of the gradient,
deflection 0.04% / 46.7%, smoothness 3.0% / 9.3%**. Converged hard — the last 50 steps moved
the loss 7e-04 and `|grad|` sat at 94.4 from step 100 onward.

**Now the comparison, and it is the point of running a multi-start at all:**

| | elite 9 (step 149) | elite 10 (step 300) | gap |
|---|---|---|---|
| loss | 49.7706 | **49.7376** | −0.033, **0.066%** |
| mesh mass | 58.7145 g | **58.6604 g** | −0.054 g, **0.092%** |
| axle drop, mean | 1.99415 mm | 1.99408 mm | 0.004% |
| stress utilisation | 0.6018 | 0.5904 | |
| `Kt_hub` | 2.0627 | 2.0223 | |

**Two starts, same mass to 0.09% — and GENUINELY DIFFERENT GEOMETRY.** This is not one basin
reached twice. The centerline genes disagree by far more than the objective does:

```
cy4   17.131 vs 15.312   -1.819 mm   10.6%
cx3   24.328 vs 26.011   +1.683 mm    6.9%
R_rim  2.339 vs  2.749   +0.411 mm   17.6%
cy2   23.586 vs 24.580   +0.994 mm    4.2%
t0..t3   2.000 vs 2.000       0.000    0.0%   <- both on the floor
```

So the optimum is a **valley, not a well**: at the 2.0 mm wall floor, mass is essentially the
spoke path length, deflection is the binding constraint, and there is a **family** of
centerlines with the same length that meet it. The 14-gene space is *underdetermined* by
(mass, deflection) — which is exactly why two starts 105 deflection points apart in Stage 2
land 0.09% apart in mass with 1.8 mm of daylight between their spokes.

**What that means for the multi-start: it was worth running, and it says more starts are
not.** A third start would be expected to find a third point in the same valley at the same
mass. The remaining freedom is not something the mass objective can spend — it is free
capacity to satisfy something the loss is not currently asking for (buckling margin,
manufacturability, fatigue at the hub). That is a better argument for M9 than anything in §2.

**The honest caveat: this is not apples to apples, and it slightly favours elite 10.** Elite 9
stopped at 149 while elite 10 ran to 300. At the SAME step the ordering is already elite 10's
— its step-150 loss is 49.7613 against elite 9's 49.7706 at 149 — and it then improved a
further 0.024 over steps 150–300. Extrapolating elite 9's remaining 150 steps at elite 10's
rate puts it near 49.747, i.e. **~0.01 of loss and ~0.02 g behind**, not ahead. The
conclusion is unchanged either way, but the 0.09% is an upper bound on the gap rather than a
measurement of it. Closing it properly costs one more 3.8 h `make prod9`, and nothing on this
list depends on the answer.

**Both records carry `workers`, `cpu_count` and `phase_scheme` in `settings`**, so the
47.0 s can be read on another machine. Memory held flat under a 20 GB `systemd-run` cap with
no `memory.events` and no oomd kill, which is the `uniform` + fidelity-off configuration §1
argues for and not a coincidence.

**One artifact correction.** `stage3_prod_best_elite9.json` is reconstructed by hand rather
than written by `main()`, because the run was stopped rather than completed — that is stated
inside the file. An earlier reconstruction took **step 148** (loss 49.771005); the record's
argmin over all 150 rows is **step 149** (loss 49.770621) and the record's own `best` block
already carried it with the full metric set. The file now carries step 149. The 3.8e-04
difference moves no gene past the fourth decimal; it is corrected because a file named `best`
should be the argmin, not because the design changed.

### 7. §0(c) — `make m9` in full. RAN. **OVERALL: FAIL, and the failure is the result.**

2930.7 s (49 min), 3.1 GB peak against a 22 GB cap — `fine` is 261k dof but this is a 2D
problem and memory was never the constraint. All four sections completed; `study_m9.json` is
whole (`complete: true`). Every section reports FAIL, and **they do not fail for the same
reason** — one is physics and three are a placeholder constant. Separating those is the work.

> **SUPERSEDED IN ITS DIAGNOSIS, NOT IN ITS NUMBERS — read §0(1) H2(a) with this section.**
> Everything measured below stands. What changed is *why*: `wheel_contact_problem` defaults
> to `kinematics="linear"`, so `prob.nonlinear` is False and the displacement this driver
> threads into `assemble_stiffness` is **ignored — measured at exactly 0.000e+00 change**.
> The quantity below is therefore `λ_min(K_linear + K_contact)`, not a tangent eigenvalue at
> all, and it contains no geometric stiffening by construction. The h² scaling, the 1.022×
> load response and the global eigenvector are not three findings — they are one defect seen
> from three sides. A generalised load factor formed under SVK kinematics **does** converge
> (1.3560 in the limit, `medium` within 0.049%); §0(1) H2(b) is that measurement.

#### THE FINDING: `lambda_min(K_t)` has no mesh-independent value. It scales as h².

The `fine` rung had never been run. It is what makes this measurable:

| best_solution, phase 0 | dof | λ_min |
|---|---|---|
| smoke | 8,904 | 2.275954e-02 |
| coarse | 41,064 | 4.827619e-03 |
| medium | 104,712 | 1.921939e-03 |
| **fine** | **261,864** | **7.898831e-04** |

Nine refinement steps across three designs give **λ ~ dof^−0.970 to dof^−1.014**, i.e.
**h^1.94 to h^2.03**. That is h², to within 3%, every time, with no sign of settling.
`last_pair_rel` is 0.589 against `GATE_MESH_REL = 0.05` — off by twelve, and flat.

**It is not a solver artifact, and that is the load-bearing check.** `measure()` runs an
independent `spla.eigsh` shift-invert reference at the first and last rungs, and it agrees
with LOBPCG to **2.668e-12 at smoke and 3.377e-10 at fine**. The number is genuinely
`λ_min(Kr)`. What diverges is the *quantity*, not the computation of it.

**Three spreads, and they settle the question:**

```
mesh ladder, ONE design, smoke -> fine        28.8x
design space, 16 designs at fixed mesh         1.457x
load ladder, ONE design, 4x the force          1.022x
```

**λ_min varies 28.8× with element size, 1.46× across the entire Stage-2 design space, and
1.02× across a 4× load range.** A buckling indicator is supposed to be dominated by load and
approach zero as the structure approaches its critical point. This one is dominated by the
mesh and is nearly indifferent to force — and over that 4× ladder it *rises* (4.723957e-03 →
4.827619e-03), which is the wrong sign for compressive geometric softening. It also tracks
stiffness rather than stability: against `1/axle_drop` over the 16 designs the correlation is
+0.91, though that is carried by one outlier (elite_11, axle drop 0.4305 against ~2.5) and
falls to **+0.45** without it, so read it as suggestive rather than established.

**This is M4 repeating, and §2 is the thing it lands on.** M8b-i.6 rewrote the stress
constraint because `c = max/pnorm` was anchored to a crack-tip singularity that diverges
under refinement. **M9 phase 3 proposes promoting λ_min(K_t) to a constraint with a margin, a
threshold and a phase aggregation rule.** A threshold calibrated against this number would be
`stress_scale` a second time: right at the mesh it was fitted on and meaningless at any other.
**Phase 3 is BLOCKED on a reformulation, not on more measurement.**

The likely cause is the formulation rather than the code. λ_min(K_t) is solved as a
**standard** eigenproblem, so it carries the dimensions of stiffness and its spectrum shrinks
with element size — h² is exactly what an unnormalised discrete operator's smallest eigenvalue
does. Classical linear buckling is **generalised** — `det(K_0 + λ·K_g) = 0` — and returns a
dimensionless *load factor* that is mesh-convergent by construction. That is the shape of the
fix, and it is a change to what is being asked, not to how well it is answered. **This is a
hypothesis from the scaling, not a measured result**, and it should be checked at two rungs
before anything is built on it.

#### The other three FAILs are one unmeasured constant, and Phase 2 existed to measure it

`phase`, `load` and `design_space` have **zero error rows and every λ finite**. They fail
solely on `lobpcg_converged`, which is `residual_rel <= wheel_adjoint.LOBPCG_RESIDUAL_REL`,
and that constant is **1.0e-7** carrying this comment:

```
LOBPCG_MAXITER = 200      # unmeasured starting point — Phase 2 measures iteration counts
LOBPCG_TOL = 1.0e-8       # explicit Phase 2 residual target; measured below
```

So the flag is not broken — **it is reporting that a self-declared placeholder was
optimistic, which is one of the things this study was for.** Measured, achievable
`residual_rel` degrades with problem size:

| | dof | residual_rel |
|---|---|---|
| smoke | 8,904 | 7e-09 – 1.5e-07 |
| coarse | 41,064 | 7.7e-08 – 9.4e-07 |
| medium | 104,712 | 1.0e-07 – 2.7e-06 |
| fine | 261,864 | 2.5e-06 – 6.3e-06 |

1.0e-7 is met at `smoke` and essentially nowhere else; 5 of 24 mesh rows, 13 of 39 phase rows,
2 of 16 design rows. **Do not "fix" this by loosening the constant to whatever passes** — that
is tuning a gate until it agrees, which §5 already refused once. What the number should be
depends on what precision the *constraint* needs, and there is no constraint yet because of
the finding above. Record it, leave it, and set it when phase 3 has a formulation.

`LOBPCG_MAXITER = 200` is NOT binding — iteration counts came in at 13–22 everywhere,
including `fine`. That half of the placeholder is fine.

#### What the study measured that IS usable

- **Phase dependence of λ_min is mild**: std/mean over 13 reference phases is **0.66% /
  1.15% / 1.59%** on the three designs, against the stress ripple's 9.8%. The 4-phase
  production stencil catches the reference min to 0.5–1.9% and the max to 1.7–4.1%. If a
  reformulated eigenvalue behaves similarly, `PRODUCTION_PHASES = 4` will be enough.
- **All 16 elites are finite and well-conditioned** at `coarse`, spread 1.457×, no solver
  refusals anywhere in 91 measurements.
- **`fine` is affordable**: 143–210 s per solve, 3.1 GB peak. Nothing about the 261k-dof rung
  needs special handling, which was unknown before this run.

#### Driver changes this run required, and why they are not cosmetic

`run_mesh` had **no `try` around its rows** while `run_load` and `run_design_space` did, and
`run_mesh` is both the first section and the only one that drives `fine`. Worse, the report
was written **once, at the end of `main()`** — so any failure erased every completed section,
and a cgroup SIGKILL at the cap is not catchable by any `try`. Both are fixed: rows are
guarded and a refused row is recorded and **fails** the section (`n_error_rows`, plus an
explicit error check in `pass`) rather than discarding the rungs that did converge; and the
report is **checkpointed after every section**, with `complete` and `sections_complete` so a
partial file can never be misread as a verdict (`pass` is absent until the run is whole).
`_series` skips error rows, so a truncated ladder cannot pass vacuously on `len < 2`.
Validated on `--quick` before the full run. This is the same precedent §1 records for
study_stage3's ladder, applied to the driver that needed it more.

---

## The decision that is a human's

Unchanged in list — rim-band genes / revisit the targets / accept a Pareto point / change
material — but **the premise moved.** Every one of those was blocked on "we need a stress
number first". The number exists now, and it says the current 14-gene space already contains
designs meeting both targets.

Adding rim-band genes to *reach* feasibility is no longer justified. Adding them to reduce
mass, or to buy margin, is a different argument and needs to be made on its own terms.

**And §6 added one to the list, with a number attached: `MIN_WALL_MM = 2.0`.** The production
descent drives **all four** thickness genes onto that floor and leaves them there, so the
2.0 mm wall is what sets 4 of the 14 genes at the answer — not the FEA, not the deflection
target, not the stress constraint. Every gram below 58.660 is on the other side of it. This
is the first item on this list that is no longer a preference: it is a manufacturing
parameter that is now provably binding, and it is worth asking the process what it can
actually hold before asking the optimizer for anything else.

**§8 HAS NOW MEASURED WHAT THE FLOOR IS WORTH, and it narrows this item rather than
closing it.** The floor is only sovereign over `t1`/`t2`; below ~1.5 mm the optimizer picks
`t0` and `t3` for itself and holds them there. So the question to put to the process is no
longer "what wall can you hold?" in the abstract — it is **"can you hold 1.2 mm?"**, worth
19.4 g against today's 2.0 mm, with everything below that worth only 3.3 g more. That is a
single yes/no with a number attached, which is what makes it decidable now.

**[CLOSED — THE ANSWER WAS YES, AND IT SHIPPED. NOTED §80.]**  §11 took the floor decision
on 2026-08-05 (*"the process can hold 1.2 mm"*) and §13 promoted the descended 1.2 mm
genome on 2026-08-06, so `MIN_WALL_MM = 1.2` is what `best_solution.json` is built at and
the 19.4 g has been collected.  This paragraph stayed open for nineteen sections after its
own question had been answered; the four items above it — rim-band genes, revisit the
targets, accept a Pareto point, change material — are the ones that are still open.

---

## Where gate 7 no longer helps, and what replaced it

`QUICK_GENES` now includes 12 and 13 (`R_hub`, `R_rim`) so `dKt/dg` is finite-differenced.
**It does not currently test that.** With utilisation at 0.375/0.300 the `soft_barrier` is
flat, so `stress` and `d_stress` are exactly zero and neither gene reaches the loss through
`Kt`, and `R_rim`'s row is `0 == 0`.

**CORRECTION, measured: `R_hub`'s +645.8 adjoint is `hub_overlap`'s, not the `fillet`
barrier's.** This paragraph said `fillet` from M8b-i.6 step 2 until §0(a) checked it. At the
shipped genome the fillet margins are `[+4.647, +0.125]`, both feasible, so that barrier is
flat and `d(fillet)/dR_hub` is exactly 0.0. Every unit of the 645.8 comes from
`hub_overlap`, whose chord proxy is violated by +0.323 mm there. §0(a) added a second live
term, `fillet_cap` at +454.0.

The product rule is tested by **`test_the_stress_gradient_obeys_the_product_rule`**
(`tests/test_objective.py`), which monkeypatches `ALLOWABLE_STRESS_MPA` down to 2.0 to force
the barrier onto its quadratic branch, then FDs genes 8, 11 and 13. **That test, not gate 7,
is what says `dKt*agg + Kt*dagg` is right.** Gene 12 was dropped from that list by §0(a):
`R_hub` is now priced through the buildable cap and the shipped genome sits above its cap,
so `dKt_hub/dR_hub` is exactly zero and its row there asserted `0 == 0`. What it used to
check moved to two solve-free tests that can afford to be far tighter —
`test_R_eff_is_exactly_the_cap_more_than_one_rung_above_it` and
`test_the_cap_gradient_matches_a_finite_difference`. Gene 13 stays in `QUICK_GENES` because
it costs nothing and becomes a live check the moment a design is stress-binding.

---

## How to run any of this

```bash
.venv-opt/bin/python studies/study_objective.py --quick   # gate 7 fast path, ~9 min
.venv-opt/bin/python studies/study_objective.py           # the full M8a gate, > 50 min
.venv-opt/bin/python studies/study_stage3.py --quick      # wiring check, ~13 min, see S8 note
.venv-opt/bin/python studies/study_stage3.py              # the M8b-i gate, S1-S10, ~2 h 45 m
make m8bi5                                                # S11 + S12, ~2 h 31 m
make m8bi6                                                # the p sweep, ~14 min
make m8bii1                                               # S13, the phase pool, ~30 min
make hubcap                                               # the hub-fillet cap vs OCC, ~10 min
make m9                                                   # M9 phase 2 in full, ~49 min.
                                                          # 3 designs x smoke/coarse/medium/fine
                                                          # x 13 phases.  3.1 GB peak; FAILS by
                                                          # design now -- see §7, exit 1 is the
                                                          # verdict, not a crash
make prod9 / make prod10                                  # §0(b), one start each, SEQUENTIAL
make m9buck                                               # §9, M9 phase 3: the generalised
                                                          # load factor + the load ramp that
                                                          # shows it is NOT a safety factor.
                                                          # ~34 min at coarse ALONE (~81 min
                                                          # beside 3 descents), ~3 GB
make minwall-<floor>                                      # §8, what the wall floor costs.
                                                          # Ran at 2.2/2.0/1.8/1.6/1.4/1.2/
                                                          # 1.0/0.8.  125 steps each; 2.0 is
                                                          # the CONTROL and the others are
                                                          # unreadable without it.  ~2-4 h
                                                          # each; four abreast is fine on 24
                                                          # cores / 61 GB and costs ~111-120
                                                          # s/step vs ~40 alone.
                                                          # NOTE: floors are a pattern rule,
                                                          # so NEVER add them to .PHONY --
                                                          # make skips pattern search for
                                                          # phony targets and every arm
                                                          # silently no-ops with exit 0.
make fillet                                               # §44, ~38 s: at what radius does the
                                                          # filleted spoke block fold, under the
                                                          # three criteria that disagreed by 20x.
                                                          # geometry + Jacobians, no field solved
make filletblock                                          # §47/§48, ~85 s: can the fillet BE a
                                                          # block, and can the sector be blocked
                                                          # around it? two 0 deg corners in PART 3's
                                                          # region; the boundary-layer block that
                                                          # does mesh; and the whole filleted
                                                          # sector, 11 blocks and 14 whole-edge
                                                          # seams, with what it costs the ring
make test                                                 # 633 passed / 3 xfailed, ~32 min
make export                                               # rebuild wheel.step, ~4 min
make studies                                              # all gates; NOT m8bi5/m8bi6/m8bii1
```

**`--workers` runs the phase loop across processes.** `0` (the default) is serial, `-1`
sizes the pool to `min(n_phase, cpu_count)`, `N` is literal and is the only memory cap:

```bash
.venv-opt/bin/python src/wheel_stage3.py --start rank:9 --steps 300 --workers -1
```

The run record carries `workers` and `cpu_count` in its `settings`, because a wall-clock
number without them cannot be read on another machine.

**On a memory-bound host, launch a production run CAPPED and DETACHED.** `systemd-oomd`
kills the whole `user@1000.service` slice at 50% memory pressure for 20 s, which drops the
user to the login screen and takes every terminal and run in that slice with it — the
kernel OOM killer is not involved, so `dmesg` is empty and `journalctl -u systemd-oomd` is
where the evidence is. A cgroup cap inverts that: the kernel kills the run at its own
limit instead.

```bash
systemd-run --user --unit=wheel-prod9 -p MemoryMax=20G --collect \
    --working-directory=$PWD /usr/bin/make prod9
```

`systemctl --user show wheel-prod9.service -p MemoryCurrent -p ActiveState` reads its
state, and the unit's journal records the peak and the verdict
(`Failed with result 'oom-kill'`, `18.9G memory peak`). It also survives the terminal
closing, which a plain child process does not.

**The `prod` targets run the interpreter with `-u`, and that is not cosmetic.** Python
block-buffers stdout when it is not a TTY, and under `systemd-run` it is a journal socket —
so without `-u` a detached descent emits **nothing** to `journalctl` until the buffer fills
or the process exits. Measured on the elite-10 re-run, which predates the flag: 162 steps
in, **zero** `[step ...]` lines in the unit's journal. That makes `-f` useless for progress
and hides a traceback until exit. On a run that predates the flag, progress has to be read
off the clock instead — `ExecMainStartTimestamp` against ~180 s for step 0 (tracing) plus
~47 s per step thereafter — and `MemoryCurrent` is the only live health signal.

Run a study driver directly and it needs `src/` on the path:
`PYTHONPATH=src .venv-opt/bin/python studies/study_stage3.py`. The Makefile exports it, so
anything driven by `make` is already covered — including the CAD hand-off, which
`src/wheel_fea.py` spawns into `.venv-cad` itself.

`--sections` selects and orders sections. `--ladder-p` takes any comma-separated exponents,
dedupes them, and **costs no extra solve** — every exponent is read off the displacement field
the adjoint already converged. `--ladder-configs smoke,coarse,medium,fine` adds a fourth rung;
`fine` is 261k dof and has never been run, so each row is wrapped in its own `try`.

**`make m8bi6` overwrites `studies/study_stage3_pnorm.json`.** Back it up before re-running if you
need to diff against it — its `pnorm_by_p` block is the step-1 evidence and must stay
reproducible.

---

## Repo layout

```
best_solution.json  stage2_elites.json     the provenance chain, read by BOTH envs
poster_summary.jpg                         written beside the genome it describes
src/        the modules — imported flat (`import wheel_fea as W`)
            project_paths.py  ROOT/SRC/STUDIES/EXPORT, stdlib only so the CAD env
                              can import it through wheel_fea
            wheel_pool.py     the parent half of the phase pool.  stdlib+numpy, NO jax
                              (a test asserts it), so sizing a pool or reading
                              PINNED_ENV costs nothing
            wheel_pool_worker.py  the child.  pins threads BEFORE its first import,
                              which is the whole reason it is a separate process
studies/    the 10 study drivers AND their .json/.jpg output, together
export/     what the CadQuery env produces: wheel.step, wheel_nofillet.step,
            wheel_step_manifest.json
tests/      conftest.py at the ROOT puts src/ on sys.path and into PYTHONPATH
```

**Imports were not rewritten.** `src/` reaches the interpreter three ways —
`pyproject.toml`'s `pythonpath` for pytest, `export PYTHONPATH` in the Makefile, and the
root `conftest.py` (which also seeds `os.environ` so the three subprocess-spawning tests
behave the same under bare `pytest` as under `make test`). A package with `__init__.py`
was rejected: `tests/test_import_hygiene.py` imports `wheel_fea` in a **jax-free**
interpreter, and an `__init__` importing jax-dependent siblings would break the CAD env.

A study driver's own `HERE` is `studies/`, so `os.path.join(HERE, args.out)` still puts
output beside the driver and was left alone. Only the INPUTS moved to `PP.ROOT` / `PP.EXPORT`.

---

## Artifacts

`study_stage3.json` / `.jpg` are M8b-i's record and `study_stage3_m8bi5.*` are M8b-i.5's; both
describe those runs and are deliberately left unedited rather than corrected after the fact.
`study_stage3_pnorm.*` were regenerated by step 2 — the `pnorm_by_p` leaves are bit-identical
to step 1's, and the top-level rows now carry the new constraint plus a `util_kt` column.

`study_stage3_pool.*` are S13's, and **half of that file describes this machine rather than
this commit** — the seconds and the projected hours are 16-core numbers. What travels is
`identical_values`, `worst_grad_rel`, and the efficiency column. Re-run `make m8bii1` on a
different host and expect different hours and a different ladder; expect the same verdict.

`study_hub_cap.*` are §5's, and unlike the above they describe **OCC's behaviour on this
shape** rather than this machine or this commit. They are the calibration behind
`HUB_CAP_THICKNESS_SHARE` and the falsification of the slot-only model, so the
`occ_limit` and `t0_sweep` blocks are the evidence and should stay reproducible. The wall
clock (583 s for three designs plus the eight-point sweep) is dominated by OCC fillet probes
— roughly 220 of them per design — and does travel between machines only loosely.

`stage3_prod_elite9.json` / `stage3_prod_elite10.json` are §6's, and they are the only
artifacts here that are a **SEARCH RESULT** rather than a gate, a calibration or a machine
description — nothing in them passes or fails. They sit at the repo root beside
`best_solution.json` and `stage2_elites.json` because that is where the provenance chain
lives and they are the next link in it. `stage3_prod_best_elite9.json` is the genome, and
it is the one file in the tree written by hand rather than by the code that names it — §6
says why and the file says so itself. `stage3_prod_best_elite10.json` (`eddcfc2`) was
written by `main()` at the end of a completed 300-step run and needs no such note; **it is
the better of the two answers and the one to carry forward.**

`studies/study_m9.json` is §7's, and it is the one artifact here whose **FAIL is the
deliverable**. `make m9` exits 1 by design: the `mesh` section fails on physics (h²
divergence) and the other three on `LOBPCG_RESIDUAL_REL = 1e-7`, a constant that says in its
own comment that Phase 2 was meant to measure it. Do not make this file pass by loosening
either gate. It also carries `complete`, `sections_complete` and per-section checkpoints, so
a killed run leaves a readable partial with no `pass` key rather than nothing — the `fine`
rung is 261k dof and had never been run when the guard was written. It turned out to cost
3.1 GB and 143–210 s per solve, so the guard was not needed this time; it is kept because
that was not knowable in advance and is now the only record of it.

The elite-9 record stops at step 149 and the elite-10 record runs to 300, so **the two are
not directly comparable step for step** — §6 states the size of that asymmetry and which way
it leans. Neither file is a gate: re-running either produces a different wall clock and,
because §1's S13 note applies, a trajectory that may differ in its last bits. What should
reproduce is the *shape* of the answer — every barrier zero, four thicknesses on the floor,
mass ~97% of the loss against deflection ~46% of the gradient.

---

### 8. §0(2) — what `MIN_WALL_MM` COSTS. DONE, IN TWO PASSES. The floor DOES stop binding — at the ENDS, near 1.5 mm — and the first pass's headline was wrong.

`make minwall-<floor>` for **2.2 / 2.0 / 1.8 / 1.6** (pass 1) and **1.4 / 1.2 / 1.0 / 0.8**
(pass 2), 125 steps each from `stage3_prod_best_elite10.json` (not `rank:10`), `coarse`,
8 phases, `uniform`, 4 workers, seed 0. **All EIGHT arms ran the full 125 steps with ZERO
events and zero rejects.** Records: `stage3_minwall_<floor>.json`, genomes
`stage3_minwall_best_<floor>.json`.

| floor | loss | mesh mass | axle drop | util | `t0` | `t1` | `t2` | `t3` | on floor |
|---|---|---|---|---|---|---|---|---|---|
| 2.2 | 56.1389 | 65.734 g | 1.9924 | 0.5731 | 2.2000 | 2.2000 | 2.2000 | 2.2000 | **4/4** |
| **2.0** | 49.7254 | 58.551 g | 1.9936 | 0.5910 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | **4/4** |
| 1.8 | 43.9892 | 52.458 g | 1.9955 | 0.6416 | 1.8000 | 1.8000 | 1.8000 | 1.8000 | **4/4** |
| 1.6 | 39.0107 | 47.026 g | 1.9970 | 0.6655 | 1.6000 | 1.6000 | 1.6000 | 1.6000 | **4/4** |
| 1.4 | 35.3760 | 42.821 g | 1.9990 | 0.7345 | 1.4000 | 1.4000 | 1.4000 | **1.5102** | 3/4 |
| 1.2 | 32.5051 | 39.194 g | 1.9992 | 0.7830 | 1.2000 | 1.2000 | 1.2000 | **1.4614** | 3/4 |
| 1.0 | 31.5398 | 37.899 g | 2.0004 | 0.7752 | **1.3826** | 1.0000 | 1.0000 | **1.5894** | 2/4 |
| 0.8 | 30.0993 | 35.911 g | 1.9991 | 0.8094 | **1.4638** | 0.8000 | 0.8000 | **1.6074** | 2/4 |

#### THE FINDING: the two ENDS pick a thickness and hold it. The mid-span never does.

**`t3` lifts off the floor at 1.4 mm and `t0` at 1.0 mm, and neither tracks the floor
afterwards.** Across a floor moving 1.4 → 0.8 mm, `t3` sits at **1.5102 / 1.4614 / 1.5894
/ 1.6074** and `t0` at **1.3826 / 1.4638**. Those are floor-INDEPENDENT values: the
optimizer is choosing them, which is the whole thing the sweep existed to detect.

**`t1` and `t2` are pinned `low` at every one of the eight floors, down to 0.8 mm.** They
never decide. The split is physical: `t0` and `t3` price the hub and rim junctions through
`Kt(R, t)`, so thinning them costs stress at a concentration and something pushes back.
Mid-span carries no local riser — it is pure mass — so it thins until the floor stops it.

The practical form: **the wheel wants ~1.45 mm at the hub, ~1.6 mm at the rim, and as
thin as the process can print in between.**

#### THE RETRACTION: `29.5 g/mm` was an artifact of only sampling floors where everything was pinned

The first pass measured four floors, found `mass ∝ floor^1.05`, and reported **29.5 g/mm
with no diminishing returns**. That fit was excellent — ~1% on all four points — and it
was excellent *because all sixteen thickness genes in it were sitting on their bound.* A
sweep in which every gene is pinned is measuring the floor, not the design, and it will
look linear no matter what the design would have done given room.

Extending the band breaks it:

```
2.2 -> 2.0 : -7.183 g  =  35.9 g/mm
2.0 -> 1.8 : -6.093 g  =  30.5 g/mm
1.8 -> 1.6 : -5.432 g  =  27.2 g/mm     <- last all-pinned interval
1.6 -> 1.4 : -4.205 g  =  21.0 g/mm     <- t3 lifts off
1.4 -> 1.2 : -3.627 g  =  18.1 g/mm
1.2 -> 1.0 : -1.296 g  =   6.5 g/mm     <- t0 lifts off
1.0 -> 0.8 : -1.988 g  =   9.9 g/mm
```

**The marginal cost falls by 4x across the band and the slope was never constant** — even
inside pass 1 it ran 35.9 → 27.2, which a `floor^1.05` fit absorbed into an exponent. Use
the interval table, not a single number. The bands worth quoting:

| band | marginal | total |
|---|---|---|
| 2.2 → 1.6 mm | ~31 g/mm | −18.7 g |
| 1.6 → 1.2 mm | ~20 g/mm | −7.8 g |
| 1.2 → 0.8 mm | ~8 g/mm | −3.3 g |

**Going 2.0 → 1.2 mm is worth 19.4 g, a third of the wheel. Going 1.2 → 0.8 mm is worth
3.3 g.** So the process conversation is worth having down to about 1.2 mm and is
approximately pointless below it.

**THIS IS THE FOURTH TIME THIS REPO HAS SHIPPED A CONVERGENT-LOOKING NUMBER THAT MEASURED
THE WRONG THING**, after `stress_scale`, `λ_min(K_t)` and the load factor (§9). The
pattern holds exactly: the quantity fit beautifully over the range sampled, and the
disproof came from **varying something other than what had been varied so far** — here,
extending the floor past the point where the answer stops touching the bound. The lesson
generalises to sweeps: *a fit taken entirely inside a region where a constraint is active
describes the constraint, not the system.*

#### NOTHING PHYSICAL BINDS ANYWHERE IN THE BAND — including at 0.8 mm

The `stress` loss term is **exactly 0.00000 at all eight floors**, as are `buckling`,
`min_sj`, `fillet_cap`, `fold`, `arrival`, `hub_overlap` and `x_order`. Only `mass`,
`smoothness` and `deflection` are ever nonzero. Utilisation reaches only **0.8094 at
0.8 mm** against an allowable of 1.0, and hub utilisation is the max at every floor
(`u_hub` 0.573 → 0.809, `u_rim` 0.428 → 0.572).

So the flattening is **not** stress switching on. It is the deflection target being bought
back with geometry: bending stiffness goes as `t³`, so past ~1.2 mm the centerline has to
distort to hold 2.0 mm of drop, and `smoothness` climbs **0.18 → 0.58** paying for it.
Axle drop stays within ±0.05% of target at every floor — the spec is met everywhere, it
just costs more geometry to meet.

**Two cautions that are NOT in the loss.** `min_scaled_jacobian` degrades monotonically
**0.8929 → 0.6366** as the floor drops; the `min_sj` barrier never fires, but the mesh is
measurably worse at 0.8 mm and any conclusion there rests on a poorer discretisation than
the ones above it. And `R_hub` moves non-monotonically (0.982 → 0.579 → 0.682) while
`kt_hub` stays pinned near 2.0-2.08 at every floor — the optimizer is trading fillet
radius against `t0` to hold the concentration factor constant, which is why `t0` lifting
off at 1.0 mm coincides with `R_hub` reversing direction. **1.2 → 1.0 is a branch change,
not a smooth continuation**, and it is why that one interval's marginal cost (6.5 g/mm) is
lower than its neighbour's on both sides.

#### What this does and does not settle

**Settled:** the optimizer picks the end thicknesses, the floor sets the mid-span, and the
floor is worth having a process conversation about down to ~1.2 mm.

**Not settled:** whether 0.8-1.0 mm is manufacturable at all, and whether the `coarse` mesh
is trustworthy at `min_sj = 0.64`. **A promotion candidate below 1.2 mm should be re-scored
at `medium` before anyone believes its mass.** No arm below 1.6 mm has been mesh-converged.

**Provenance gap in these artifacts.** The eight `stage3_minwall_best_*.json` were written
before `search_block()` existed, so their `search` blocks carry the optimizer settings but
**not** `min_wall_mm`. The floors are recoverable from `stage3_minwall_<floor>.json`
(`settings.min_wall_mm`) and from the pinned `t1`/`t2` values. Future `--best-out` records
carry the floor; these eight do not.

#### The control arm is what makes the other three readable, and it earned its place twice

`minwall-2.0` restarts from its own converged answer at its own floor, so it measures the
protocol rather than the floor. **It found a real defect in the experiment design before
the other arms had produced anything.**

§0(2) justified 125 steps from a measurement off elite 10's record — "step 125 is +0.096%
from its final loss". **That measurement does not transfer to a restart.** It was taken
INSIDE a running descent, where Adam's `m`/`v` were warm and the cosine `lr` had already
decayed. Restarting cold at the optimum with `lr = 0.01` and zeroed moments kicks the
iterate straight back out: the control's loss went 49.7376 → 49.95 → **51.09** over three
steps, a 2.7% excursion, before settling.

It recovered — final **49.7256**, best **49.7254 at step 61**, mass **58.551 g** against
elite 10's 58.660 g, i.e. **−0.19%**. So the transient costs nothing and the protocol is
sound. But that is a MEASURED conclusion, and without the control arm the 1.6 result would
have been reported against a start value from a different run under a different schedule.

**The control drift is the sweep's noise floor: ±0.19% in mass.** Every difference in the
table above is 50-100x that, which is what makes them signal.

#### What this run measured about the MACHINE, and it contradicts §1

This box is **24 cores / 61 GB / 7 GB swap**, not the 16-core / 31 GB box every number in
§1 was measured on. The memory rules relax — one descent measured a **13.56 GB peak**
(§1's 12.7 GB estimate travelled), so three fit where one used to.

**But the parallel speedup does NOT track cores or RAM, because the binding resource is
MEMORY BANDWIDTH.** Measured across both passes: one arm alone runs at **~38-47 s/step**;
three arms concurrently at **~95 s/step each**; **four arms concurrently at ~111-120
s/step each** (pass 2, 16 pool workers on 24 cores). Going 3 → 4 arms bought **~17% more
aggregate throughput for 33% more concurrency**, and against a single arm the aggregate
speedup is **~1.5x, not the ~4x** core count and capacity predict.

The same effect distorts any wall-clock read off a shared box, in both directions: §9's
buckling study took **4851 s** beside three descents and **2045 s** alone, and a
single-step `smoke` descent — normally a minute or two — took **6m41s wall / 11m13s CPU**
while four arms were running. **Never quote a wall clock from this repo without saying
what else was on the machine.** The inner loop is `spla.spsolve` (`wheel_fem.py:1183`), a sparse direct LU called
once per Newton iteration, and twelve of those saturate the memory controller.

Two consequences. **The GPU is irrelevant** — `requirements-opt.txt` pins CPU
`jax==0.11.0`/`jaxlib==0.11.0`, and even with a CUDA build the expensive half is SciPy's
LU on the CPU, while the `XLA_FLAGS` pin that makes the adjoint bit-reproducible is a
CPU-thread-pool setting. **And S13's efficiency ladder does not travel across concurrent
RUNS** — it measured workers inside ONE evaluation, which is a different contention
regime. If a single descent's wall clock matters, the lever is the solver (reusing
factorisations, or a preconditioned iterative solve), not the hardware.

**Pass 1** ran three-at-a-time with the fourth held back, each under
`systemd-run --user -p MemoryMax=15G --collect`, on the reasoning that four concurrently
is ~54 GB of 61 and slice-level memory pressure is what took the desktop down twice before.

**Pass 2 ran all four concurrently with no caps, and it was fine** — the box stayed at
~4 GB used with 40 GB free throughout, and all four arms completed with zero events. The
15 GB-per-arm figure that motivated the caps came from a `medium`-rung measurement; a
`coarse` arm with 4 workers is nowhere near it. Caps are still the right default for
`medium` or for anything left unattended, but they are not needed to run this sweep.

---

### 9. M9 PHASE 3 — THE LOAD FACTOR CONVERGES AND IS NOT A SAFETY FACTOR. `make m9buck`.

`studies/study_m9_buckling.py`, **2045 s** at `coarse`, **OVERALL: PASS** — and unlike §7
the PASS is not the interesting part. Artifact: `studies/study_m9_buckling.json`.

Budget 2045 s only on an otherwise-idle box. An earlier identical run took **4851 s**
because three `minwall-*` descents were sharing the machine — 2.4x, from contention alone,
with no setting changed. The inner loop is a sparse direct LU, so it is memory-bandwidth
bound and co-scheduled runs steal from each other far more than core count suggests (same
effect as §8's note on the sweep). Do not read a wall-clock off a shared run and project
from it.

**The formulation is now reproducible from the repo, which it was not.** §0(1) H2's
numbers lived only in a scratchpad `h2_check.py` that is no longer on disk, so this driver
had to re-derive them. It does, exactly:

| config | reduced dof | load factor | PLAN.md H2(b) |
|---|---|---|---|
| smoke | 8,904 | **1.378129** | 1.378129 |
| coarse | 41,064 | **1.359846** | 1.359846 |
| medium | 104,712 | **1.356669** | 1.356669 |

`last_pair_rel` **2.34e-03** against `GATE_MESH_REL = 0.05` — inside by **21x**, on the
same meshes where `λ_min(K_t)` misses by twelve.

**THE STATE MUST BE SOLVED UNDER SVK, NOT ONLY ASSEMBLED UNDER IT — AND PLAN.md DID NOT
SAY SO.** H2's prose describes `K_0 = K(u=0)`, `K_t = K(u_service)` without stating which
kinematics produced `u_service`, and the obvious reading — reuse the existing
linear-kinematics contact solve — is **wrong by +31%**:

| state solved under | smoke | coarse |
|---|---|---|
| `kinematics="linear"` | 1.800046 | 1.785253 |
| **`kinematics="svk"`** | **1.378129** | **1.359846** |

That is the same class of error `study_m9` made (a linear state threaded into a nonlinear
operator), one level up, and it was found by prototyping the formulation before writing a
driver on it. It is now documented in the driver header and the `m9buck` recipe.

#### THE FINDING: `lambda(f) > 1` AT EVERY LOAD LEVEL. There is no critical point.

A real critical factor is a **fixed point**: at `f = λ_cr` the remaining factor is 1.0.
Measured on `best_solution` at `coarse`, eleven load levels, **zero solver refusals**:

| f (× service) | 0.5 | 1.0 | 1.36 | 2.0 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|
| **λ(f)** | 1.9891 | 1.3598 | 1.2162 | 1.1070 | 1.0481 | **1.0263** |
| f·λ(f) | 0.9946 | 1.3598 | 1.6540 | 2.2140 | 3.1443 | **4.1051** |

**λ approaches 1 from ABOVE as roughly `1 + 0.43/f²` and never crosses it.** The implied
critical load recedes exactly as fast as the load advances — a treadmill, not a limit
point. `crosses_unity: false` is now a field in the artifact.

**Why that is decisive rather than suggestive.** `λ = 1` is precisely the condition
`det(K_0 + 1·K_g) = det(K_t) = 0`. So `λ(f) − 1` measures how close the CONVERGED TANGENT
is to singular, and it never reaches zero. The quantity is real and it is telling the
truth; what it is not is a load factor.

**So: 1.36 must not be reported as a safety factor** — PLAN.md's warning was right, and
this is the measurement that settles it. **What can be said instead is stronger and more
useful: the wheel is stable to at least 4× service load**, on eleven converged nonlinear
SVK contact solves with a non-singular tangent throughout. The Euler `buckling` proxy
(ratios 0.066-0.087) would never have shown that.

#### What else the study established

- **Phase spread 1.25-2.35%** over 13 reference phases on three designs, against
  `λ_min`'s 0.66-1.59%. `PRODUCTION_PHASES = 4` would be adequate *if* the quantity were
  ever promoted. Open item 1, half one: CLOSED.
- **Design space: 16/16 converged, 1.0912-2.4032, a 2.20x spread, none below 1.0.** So
  1.36 is *this design*; elites 1 and 2 sit near 1.10. Open item 1, half two: CLOSED.
- **Contact in the operator is +1.46%** at `coarse` (+3.38% at `smoke`, so it shrinks
  under refinement). Open item 4: CLOSED, and it is small.

#### Phase 3 stays blocked, and the reason is now measured rather than suspected

The quantity is mesh-convergent, phase-stable, design-discriminating and cheap. It fails
exactly one requirement — **load-independence** — and that is the one a constraint needs.
A threshold calibrated at service load would be `stress_scale` a third time.
`LOBPCG_RESIDUAL_REL` stays at 1e-7 and `buckling` stays inert; §7's rule ("record it,
leave it, set it when phase 3 has a formulation") is still not satisfied.

**THIS IS THE THIRD TIME THIS REPO HAS HIT THE SAME PATTERN**, and it is the most
transferable thing on this page: `stress_scale`'s `c = max/pnorm`, `λ_min(K_t)`, and now
the load factor. Each was well-posed, each converged or looked convergent, and each
measured the wrong thing. **In all three cases mesh convergence was treated as evidence of
meaning, and in all three the disproof came from varying something OTHER than the mesh** —
the exponent, the kinematics, the load. A quantity that has only ever been refined has not
been tested.

---

### 10. §0(3) — DEFERRED, DELIBERATELY. And the golden test no longer blocks it.

**Not done, and that is a decision rather than a slip.** §8 measured that
`stage3_prod_best_elite10.json` (58.660 g) is optimal **at the 2.0 mm floor only**, and
there are now **eight** candidates on disk spanning 65.734 g down to 35.911 g. Whichever
floor the process can actually hold determines which one should ship; promoting 58.660 g
now would ship a genome that one process conversation could supersede.
`best_solution.json` is **untouched**.

**The shortlist, given §8.** 1.2 mm (**39.194 g**) is the value-for-money point — it
captures 19.4 g of the available 22.6 g, and everything below it is worth 3.3 g total.
1.6 mm (**47.026 g**) is the conservative pick and the thinnest floor at which all four
genes are still pinned, i.e. the last one whose behaviour is fully understood. **Nothing
below 1.6 mm should be promoted without a `medium`-rung re-score first** — `min_sj` falls
to 0.71 at 1.2 mm and 0.64 at 0.8 mm, and no sub-1.6 arm has been mesh-converged.

**What WAS done is the part that made promotion risky, and it is now permanent.**
`tests/test_golden.py` — the repo's stated "safety net for every later milestone" — read
`geometry`, `loss_terms` and `metrics` out of `best_solution.json` and recomputed them
through `wheel_fea.evaluate_design`. That coupling meant any promotion forced a choice
between two bad outcomes:

1. **Break the golden test.** The Stage-3 record's `loss_terms` are the FEA objective's 13
   terms (`fillet_cap`, `min_sj`, `phase_ripple`, ...); `evaluate_design` produces the
   beam surrogate's 9. `test_loss_terms_reproduce` fails outright on the mismatch.
2. **Re-baseline it.** Regenerate the blocks with `evaluate_design` so it passes — at
   which point the test checks that function against numbers the function just produced,
   and **detects nothing**.

So the GA/beam reference is preserved as **`best_solution_ga_beam.json`**, carrying a
`note` block saying what it is, and `test_golden.py` reads that file. Its job is
regression detection on `evaluate_design`, which has nothing to do with which genome
ships. **11 tests pass against the pinned file, and `best_solution.json` is now free to
move whenever the floor decision lands** — a one-file change with no test churn.

The two files are byte-identical in every block today. That is the starting condition, not
an invariant.

**What promotion will still need when it happens**, unchanged from §0(3): `make export`
(~4 min, expect nearer 230 s now all 48 corners fillet), and then a real look at the
manifest rather than an assumption — `r_built`, `kt_error_pct`, face count, shortest edge,
and the AREA and MASS gaps (−1.384% / −2.277% today). Nobody has yet looked at any
Stage-3 geometry in OCC. `R_hub` on the elite-10 genome is **0.9820** in the file against
a cap of 1.0400, i.e. under it, which is the good direction — but the sub-1.6 mm arms run
`R_hub` down to **0.579**, and nothing has checked that OCC can actually build a fillet
that small next to a 1.2 mm wall. **Export the candidate before trusting its mass.**

**Two latent breaks on this path were found and fixed while the sweep ran**, both of which
would have failed SILENTLY rather than loudly, which is why they were worth catching first:

- **`--best-out` records did not carry the box they were descended in.** The GA writer
  records `min_wall_mm`/`cy_bound_mm` (`wheel_fea.py:1393`) and the Stage-3 writer did
  not — so the eight sweep genomes, every one of them a boundary optimum, were
  distinguishable only by reading their own pinned `t` values back out. Now
  `wheel_stage3.search_block()`, split out of `main()` so it is testable without a solve
  (the cheapest run that reaches that line still costs minutes).
- **The exporter's mass cross-check went `nan` on a Stage-3 genome.** `report()` read
  `metrics['total_mass_g']` through a `.get(..., nan)`; the GA writes that key and a
  descent writes `mesh_mass_g`. The export succeeded and printed `nan g` — losing one of
  the only checks that compares the FEA and CadQuery pipelines against each other, on
  exactly the genome that ships. Now `wheel_step_export.optimizer_spoke_mass()`, which
  returns the value **and which key it came from**: the two are the same role measured two
  different ways (analytic beam area vs integration over the FEA mesh) and normalising
  them into one number would hide that from the reader.

---

### 11. THE FLOOR DECISION LANDED: **the process can hold 1.2 mm** (2026-08-05). The 1.2 mm arm is re-scored, exported, and it is NOT promoted yet — one check fired.

§10's shortlist asked a single yes/no with 19.4 g attached. The answer is **yes**, so the
promotion candidate is `stage3_minwall_best_1.2.json` (genome `350f4c7`, **39.194 g**),
and §10's two prerequisites have now both been run against it.

**`best_solution.json` is still untouched.** Two things came back from the export that a
human should see before it moves — see "What fired" below.

#### The `medium` re-score: the mass is real and both constraints still hold

§8 and §10 both said nothing below 1.6 mm may be believed until it is re-scored on a
finer mesh, because no sub-1.6 arm had ever been mesh-converged. One forward evaluation
of the converged genome at `medium`, 8 phases, `uniform`, 4 workers, **271.8 s**:
`stage3_minwall_1.2_medium.json`, genome echoed to `stage3_minwall_best_1.2_medium.json`.

| quantity | `coarse` | `medium` | change |
|---|---|---|---|
| **mesh mass** | 39.19436 g | **39.19433 g** | **−0.00008%** |
| axle drop (target 2.0) | 1.99923 mm | **2.01931 mm** | deflection error **0.0% → +1.00%** |
| **utilisation** | 0.7830 | **0.7986** | +1.98%, against an allowable of 1.0 |
| `min_scaled_jacobian` | 0.71086 | **0.71114** | **+0.04%** |
| `phase_ripple` | 0.09570 | 0.09421 | −1.56% |
| loss | 32.5051 | 32.7376 | +0.72% |
| `max_stress_mpa` | 140.89 | 182.42 | **+29.5%** — the singularity, as always |

**Every barrier term is still exactly 0.00000 at `medium`** — `stress`, `min_sj`,
`fillet_cap`, `fold`, `arrival`, `hub_overlap`, `x_order`, `buckling`, `phase_ripple`.
Only `mass`, `smoothness` and `deflection` are nonzero, exactly as at `coarse`.

Three things worth reading off that table:

- **The mass survives refinement to five figures**, which is what the re-score existed to
  establish. 39.194 g is a real number.
- **`min_sj` does NOT degrade under refinement** — 0.7109 → 0.7111. §8's caution ("the
  mesh is measurably worse at 0.8 mm and any conclusion rests on a poorer
  discretisation") was about mesh quality falling as the *floor* falls; it does not
  compound as the mesh refines. The thin wall is what sets it, not the resolution.
- **`max_stress_mpa` moving +29.5% while `utilisation` moves +1.98% is the M4 result
  restated**, and it is the reason the constraint was rewritten in M8b-i.6. Anyone
  alarmed by 182 MPa against a 25 MPa allowable is reading the divergent field max.

**What this is NOT: a re-optimisation at `medium`.** It is `--steps 0` — the `coarse`
answer scored once on a finer mesh. It says the design is what `coarse` said it was; it
does not say a `medium` descent would stop here.

#### The export: for the FIRST time the Kt the optimizer priced is the Kt the part has

`export/stage3_minwall_best_1.2.step` (+ `_nofillet.step`, `_step_manifest.json`),
**67.3 s**, OCC valid, one solid, not self-intersecting, bounding box exact.

| junction | R requested | R built | edges | Kt model | Kt built | **error** |
|---|---|---|---|---|---|---|
| hub | 0.5790 | **0.5790** | **24/24**, ONE family | 2.0235 | 2.0235 | **+0.0%** |
| rim | 2.7495 | **2.7495** | **24/24**, ONE family | 1.4226 | 1.4226 | **+0.0%** |

Against the shipped genome's **+73.4%** and **+111.4%**. §3's whole finding — that a
junction priced by its worst corner reports what twelve square corners were hiding — has
no bite here because **there is no second family**: every corner takes the requested
radius at the first ladder rung. That is also why the export costs 67 s against 230 s.

**The mechanism is that small fillets build.** §5's cap model predicted this and the
descent obeyed it: `R_hub` = 0.579 sits under `hub_fillet_cap_mm` = 0.624, and 2.749 is
under the rim's slot. The genome stopped asking for fillets the part cannot build, so the
stress model and the solid finally agree. **`kt_error_pct` is now +0.0% rather than +73.4%
— the discrepancy this milestone existed to remove is gone on this genome.**

Solid mass **47.63 g** (whole wheel; the optimizer's 39.194 g is spoke material only, and
hub + rim add the rest) against the shipped genome's 74.12 g, **−35.7%**.

#### WHAT FIRED, and it is why this is not promoted in the same breath

1. **`[WEAK JUNCTION]` — hub overlap 18.12 mm³ against a 50.0 mm³ floor.** The shipped
   genome is 78.53 mm³ and the rim here is 68.13 mm³, so it is the **hub** end alone. The
   exporter's own words: "this spoke grazes the ring rather than meeting it. Deepen
   `HUB_EMBED_RADIUS_MM`, or constrain the optimizer's junction angle." Cause is not
   mysterious — the root is `t0` = 1.2 mm against 2.477 mm on the shipped genome, and the
   embed is only 0.5 mm deep (`HUB_EMBED_RADIUS_MM = HUB_RADIUS_MM − 0.5`), so there is
   simply less material buried inside the hub disk. **The union still produced a valid
   single solid with all 24 hub corners on the circle**, so this is a warning about weld
   robustness, not a build failure.

   **MEASURED, AND IT KILLS THE SUGGESTED FIX: deepening `HUB_EMBED_RADIUS_MM` does not
   get there.** Profile-only sweep, no fillets, three genomes, embed depth 0.5 → 2.0 mm
   (`embed_depth_probe.py`, a scratchpad one-question probe like `eps_n_check.py`):

   | embed depth | 1.2 mm genome | elite 10 (`t0` 2.00) | `best_solution` (`t0` 2.48) |
   |---|---|---|---|
   | **0.50 (shipped)** | **18.12** | **48.72  ← also under** | 78.53 |
   | 1.00 | 22.64 | 52.54 | 80.68 |
   | 1.50 | 27.20 | 56.30 | 82.86 |
   | 2.00 | 31.73 | 60.09 | 85.01 |

   Four times the embed depth buys the 1.2 mm genome **13.6 mm³**, and it needs 31.9.
   Extrapolating the 9.1 mm³/mm slope puts the crossing near **4 mm of embed** — a third
   of the 12.7 mm hub radius, with twelve roots converging inside it. That is not a
   constant to nudge; it is a different construction. (Self-intersections stayed **0** and
   min curvature never moved at any depth on any genome, so *"keep these SHALLOW"* was not
   what bit here — the sweep is safe, it just does not reach.)

   **What the sweep actually shows is that the 50 mm³ floor tracks root thickness, and it
   was set when roots were ~2.5 mm.** The overlap is superlinear in `t0` — 1.2 → 2.0 →
   2.48 mm of root gives 18.1 → 48.7 → 78.5 mm³ at the shipped depth — because a
   near-tangent arrival makes the buried region a wedge. **Elite 10, the candidate this
   repo was going to promote before today, is at 48.72 against the same 50 floor**, i.e.
   the check was already about to fire at the 2.0 mm floor and nobody had run the export
   to see it. So the question is not "how do we rescue this genome's weld" but **"what is
   `MIN_JUNCTION_OVERLAP_MM3` actually asserting, and is a fixed volume the right form of
   it for a thin-walled part?"** That is a modelling decision, and it is the one blocking
   promotion. **ANSWERED IN §12: no, it is not — the fixed volume was a `t0` proxy, and
   normalising it out shows all three genomes at the same 0.56 root thicknesses of
   engagement. The constant is gone and the candidate passes.**
2. **The smallest features got ~19x smaller.** `min_edge` **0.0087 mm** (shipped 0.162)
   and `min_face` **0.195 mm²** (shipped 3.63). OCC is happy — `BRepCheck` valid, 0
   degenerate edges, tolerance 1e-07 — and `min curvature R` is **0.579 mm**, still well
   over the 0.25 mm floor and over a 0.4 mm nozzle. But an 8.7 µm edge is exactly the kind
   of feature another kernel drops silently, and **Inventor has never imported anything
   from this repo** (§0). This is the one import that is now worth doing.

Neither is a reason the 1.2 mm floor was the wrong call. Both are reasons the *export* of
that genome is not yet a shippable part.

#### The exporter can build a candidate without touching the shipped STEP

`wheel_step_export.py` read `best_solution.json` unconditionally, so "export the candidate
before trusting its mass" (§10) was impossible to do in the right ORDER — the only way to
export a genome was to promote it first. It now takes **`--genome <record.json>`**, and
`output_paths()` names the three artifacts after that file's stem unless it IS the shipped
genome, which keeps `wheel.step` / `wheel_nofillet.step` / `wheel_step_manifest.json`
exactly as they were. **A candidate cannot overwrite the shipped STEP**, which would have
recreated on purpose the failure this file was audited for. `--out-prefix` overrides.
`make export EXPORT_GENOME=stage3_minwall_best_1.2.json` drives it. No arguments is
byte-identical behaviour to before, which is what the GA hand-off (`wheel_fea.py:1658`)
still calls.

`warn_if_stale` takes the step path it is actually about and prints the real source name,
rather than saying `best_solution.json` whatever it just read.

**`make test` 427 passed** (1279.97 s), unchanged from §8 — the flag is additive and the
no-argument path is the one every test and the hand-off already exercised.

#### What promotion still needs

`best_solution.json` is unchanged; `tests/test_golden.py` is already decoupled (§10), so
the promotion itself remains a one-file change. Before it:

1. ~~A decision on `MIN_JUNCTION_OVERLAP_MM3`.~~ **RESOLVED — see §12. The constant was
   measuring `t0`.** It is gone; the check now gates on `wheel_geometry.junction_bite`,
   the 1.2 mm candidate passes at 0.562 root thicknesses against a 0.25 floor, and
   nothing about the candidate's geometry changed.
2. ~~**One Inventor import** of `export/stage3_minwall_best_1.2.step`, against a 0.0087 mm
   edge and a 0.195 mm² face.~~ **DONE 2026-08-06 — it imported clean. THE PROMOTION
   HAPPENED; see §13.** Both items on this list are closed and this section is history.
3. Note that promoting also moves the WHEEL the studies describe: every study driver reads
   `best_solution.json`, so §9's load factor (1.36 on the *shipped* genome) and every other
   per-design number on this page is about a genome that would no longer be the shipped one.
   Nothing is invalidated — they are labelled by design — but they stop describing "the
   wheel" and start describing "the old wheel".

---

### 12. THE WEAK-JUNCTION CHECK WAS MEASURING `t0`. `MIN_JUNCTION_OVERLAP_MM3` IS GONE (2026-08-05).

§11 left one thing blocking promotion: the exporter's `[WEAK JUNCTION]` check fired on the
1.2 mm candidate at **18.12 mm³** against `MIN_JUNCTION_OVERLAP_MM3 = 50.0`. §11 had already
ruled out the exporter's own suggested fix (4× the hub embed depth buys 13.6 mm³ of the 31.9
needed) and noted that **elite 10 fails the same check at 48.72 mm³**. That second fact was
the tell, and it is now the whole answer.

#### The measurement

`check_junction_overlap` intersects ONE spoke's 2D outline with the hub disk and multiplies
by the 22.4 mm face width. That volume is **quadratic in the root thickness**: once through
the band width, and again because a near-tangent band of that width crosses the circle over a
proportionally longer arc. Divide it out — `overlap_mm3 / (t² · W)`, the weld's penetration
measured in ROOT THICKNESSES:

| genome | `t0` | hub mm³ | **bite** |
|---|---|---|---|
| `stage3_minwall_best_1.2.json` | 1.20 | 18.12 | **0.562** |
| `stage3_prod_best_elite10.json` | 2.00 | 48.72 | **0.544** |
| `best_solution.json` (shipped) | 2.48 | 78.53 | **0.571** |

**The raw volumes span 4.3×. The bites agree to 3%.** These are the same junction three
times over, and the old constant "failed" two of them purely for being thin. It was never a
verdict on the weld; it was `t0` with a threshold on it. That is why the thinnest design
failed it, why elite 10 failed it, and why no amount of embed depth was ever going to help.

#### What replaced it

`wheel_geometry.junction_bite(overlap_mm3, t_mm, width_mm)` and `MIN_JUNCTION_BITE = 0.25`.
They live in `wheel_geometry` for the same reason `MAX_ARRIVAL_DEG` does — it is the one
module both interpreters can import, so the exporter (CAD env) and the test that audits its
manifest (jax env) share one definition rather than two that can drift.

The geometric argument: a spoke arriving radially and burying depth `d` contributes
`overlap = t·d·W`, so the ratio is exactly `d/t`, and a grazing spoke drives it to 0 — which
is the failure the check exists for.

**0.25 IS A GEOMETRIC FLOOR, NOT A CALIBRATED ONE, AND THE CODE SAYS SO.** Three samples
that agree to 3% cannot calibrate a threshold, and this repo has never produced a junction
that actually failed, so there is no negative example to fit against. 0.25 is half of what
every measured design achieves. Replace it the moment a real failure turns up.

`check_junction_overlap` **still only warns, and must keep only warning.** It runs inside the
GA's export hand-off (`wheel_fea.py:1658`), which checks nothing but the return code — so
raising there would throw away a finished optimization run over a heuristic. The number goes
to the manifest instead, where a test can see it. The warning text also stopped advising
"deepen `HUB_EMBED_RADIUS_MM`", which §11 measured and ruled out.

#### It still catches what it exists for

Negative control: recede the hub ring toward the spoke root so the spoke genuinely only
grazes it, and the metric falls monotonically and fires.

| hub ring r | hub mm³ | bite | |
|---|---|---|---|
| 12.70 (real) | 18.12 | 0.562 | |
| 12.55 | 13.20 | 0.409 | |
| 12.40 | 9.27 | 0.287 | |
| 12.30 | 7.25 | **0.225** | **[WEAK JUNCTION]** |
| 12.25 | 6.43 | **0.199** | **[WEAK JUNCTION]** |

Read honestly: this shows the new metric detects grazing, not that it beats the old one on
this axis — the old floor would have fired here too. The evidence that the *form* was wrong
is the invariance table above, not this sweep.

#### The manifest, and the tests it now has

`junction_overlap_mm3` keeps `hub` and `rim` in mm³ under the same names, and gains `bite`,
`t_mm`, `pass` and `bite_floor`. `floor` is gone rather than repurposed: it was a mm³ number
and the new one is in root thicknesses, so a reader who missed the change gets a `KeyError`
instead of a plausible wrong comparison.

Three tests in `tests/test_export_contract.py`, where there were **none** — nothing under
`tests/` referenced this check before today, which is how it spent the entire `MIN_WALL_MM`
sweep reporting a `t0` proxy without anything noticing:

- the block carries the normalised fields at all;
- **the bite reconstructs from the genes** — recomputed in the jax env, from a manifest
  written by the CAD env, which is what pins the hub to `t0` and the rim to `t3`. On the
  shipped genome a swap moves the hub bite 54%, and the raw volumes cannot reveal it;
- the shipped part clears its own floor, and `pass` IS the floor comparison rather than an
  independently written opinion.

#### The re-exports prove the change is reporting-only

`make export` on the unchanged shipped genome: `junction_overlap_mm3` is **the only
non-volatile key in the manifest that moved**. `genome_hash` 36aed36, `solid.volume_mm3`
59777.4, `mass_g_pla` 74.12, every `fillets.detail` row (hub +73.4%, rim +111.4%),
`profile_health` and `step_health` are all identical. Shipped bite: hub **0.571**, rim
**4.379**, both pass.

`make export EXPORT_GENOME=stage3_minwall_best_1.2.json`: hub bite **0.562**, rim **1.424**,
both pass, **no warning**. Solid 38415.2 mm³ / 47.63 g, `kt_error_pct` +0.0% at both
junctions, min edge 0.0087 mm — every number identical to §11. Nothing about the candidate's
geometry changed; only what the exporter says about it.

**`make test` 427 + 3 = 430 passed.**

#### What is left

~~Promotion now waits on **one thing**: the Inventor import of
`export/stage3_minwall_best_1.2.step`, against that 0.0087 mm edge and 0.195 mm² face.~~
**That import passed on 2026-08-06 and the genome is promoted — §13.** `best_solution.json`
is no longer untouched; it is `350f4c7`.

---

### 13. PROMOTED. `best_solution.json` IS THE 1.2 mm STAGE-3 GENOME (2026-08-06).

The Inventor import of `export/stage3_minwall_best_1.2.step` passed — that was the last
gate §11 and §12 left standing, and it was the one nobody in this repo could measure from
a script: whether a 0.0087 mm edge and a 0.195 mm² face survive a real MCAD translator.
They do. So the promotion happened.

`best_solution.json` now holds **`350f4c7`**, verbatim from `stage3_minwall_best_1.2.json`
apart from a `note` block recording where it came from. The old shipped genome, **`36aed36`**,
is intact in `best_solution_ga_beam.json` — which is not a courtesy copy, it is the file
`tests/test_golden.py` reads (§10), and that decoupling is exactly why this promotion did
**not** re-baseline the regression net onto numbers regenerated by the function the net
exists to check.

#### What shipped

| gene | old `36aed36` | new `350f4c7` |
|---|---|---|
| `t0` | 2.4774 | **1.2000** |
| `t1` | 2.0000 | **1.2000** |
| `t2` | 2.0000 | **1.2000** |
| `t3` | 2.0000 | **1.4614** |
| `R_hub` | 1.5598 | **0.5790** |
| `R_rim` | 3.0000 | **2.7495** |

Three of the four thicknesses sit exactly on the 1.2 mm wall floor. `R_hub` 0.579 is well
under §5's cap of 0.624 for this genome, which is why every corner filleted at the radius
the stress model priced.

#### The export, old → new

`make export` on the promoted file. **The result is identical to the candidate export in
§11 and §12 — all 33 geometric keys, byte for byte; the only manifest key that differs
between them is `source`.** That is the check that matters here: promotion is a rename of
which genome is "the" genome, and it must not perturb geometry. It did not.

Stronger, and worth recording because it retires the obvious doubt: `export/wheel.step` and
`export/stage3_minwall_best_1.2.step` are **byte-identical apart from the `FILE_NAME`
timestamp** — same length, `diff` over everything past the header returns zero lines. The
file now shipping as `wheel.step` is *literally* the file that went into Inventor, not a
rebuild that ought to match it. Same for the `_nofillet` pair.

| manifest key | old | new |
|---|---|---|
| `solid.volume_mm3` | 59777.4 | **38415.2** |
| `solid.mass_g_pla` | 74.12 g | **47.63 g** |
| `fillets` hub `kt_error_pct` | +73.4% | **+0.0%** |
| `fillets` rim `kt_error_pct` | +111.4% | **+0.0%** |
| hub fillet families | 1.127 mm ×12, 0.361 ×12 | **0.579 mm ×24** |
| rim fillet families | 3.000 mm ×12, 0.308 ×12 | **2.749 mm ×24** |
| `junction_overlap_mm3.bite` hub / rim | 0.571 / 4.379 | **0.562 / 1.424** |
| `step_health.min_edge_mm` | 0.162221 | **0.008711** |
| `step_health.min_face_mm2` | 3.6338 | **0.1951** |
| `step_health.min_curvature_radius_mm` | 0.308309 | **0.578934** |
| surface census | 26 Plane / 48 BSpline | **14 Plane / 60 BSpline** |

**26.5 g of PLA, 35.8% of the solid, for a part that is now the first one this repo has
built whose fillets match the ones its stress model priced.** The `kt_error_pct` collapse
is not a fillet improvement — it is the consequence of `R_hub` dropping under the cap, so
OCC never has to fall back to a second, smaller family. The single-family rows are the
evidence.

The two `step_health` numbers that got WORSE are the ones the Inventor import was for.
The minimum edge fell 19× and the minimum face 19×; both are geometry the exporter has
always produced and neither is a defect, but 0.0087 mm is small enough that a translator
was entitled to drop it. It didn't.

#### The design this buys, stated honestly

Both rows below are the `medium` rung under the M8b-i.6 Kt formulation, so they *are*
comparable — unlike the two records' raw metric blocks, which use different key names for
different measurements (`total_mass_g` is the beam surrogate's analytic area, `mesh_mass_g`
is integrated over the FEA mesh; **do not subtract them**).

| | old `36aed36` | new `350f4c7` |
|---|---|---|
| stress utilisation (hub) | 0.4099 | **0.7986** |
| stress utilisation (rim) | — | 0.5594 |
| `Kt_hub` / `Kt_rim` | 1.861 / 1.490 | **2.031 / 1.423** |
| field max (mesh-divergent, **not** the constraint) | 48.47 MPa | 182.4 MPa |
| `min_scaled_jacobian` | — | 0.7111 |

**Utilisation roughly doubled.** That is what 26.5 g costs, and it is the honest headline of
this promotion. The barrier is still satisfied — every term in `loss_terms` that is a
constraint reads exactly `0.0`, `stress` included, and 0.799 is inside 1.0 — but the margin
is now thin where it used to be comfortable. The field max moving 48.5 → 182.4 MPa is **not**
a 3.8× stress increase and must not be read as one: §0 and M4 both record that the field max
diverges under refinement and is not a number the project constrains. `min_sj` 0.711 clears
the 0.64 threshold §8 flagged as the trustworthy floor for a Stage-3 re-score.

#### What is now stale, and it is a lot

Every driver in `studies/` defaults to `best_solution.json`. Nothing in those recorded
studies is *wrong*, but from today they describe **the old wheel**:

- **§9's M9 phase-3 load factor of 1.36** is on `36aed36`. It was never a safety factor
  (§9 is emphatic) and it is now also not this wheel's number. Re-running `make m9` measures
  `350f4c7`.
- **§0's utilisation table** (`best_solution` at 0.4099, elite 1 at 0.5063) is `36aed36`
  and `stage3_prod_best_elite*`.
- Every per-design figure in §1–§12 that names `best_solution`.

The banner at the top of this file says the same thing, because a fresh session reads that
before it reads this.

#### The gate: **20 failed, 410 passed** — and that is the real story of this promotion

`make test` on the promoted genome. Every failure is a test that had quietly encoded a
property of `36aed36` as if it were a property of the PROJECT. Nothing in `src/` broke.
But they are not all the same kind of stale, and lumping them together would be the
mistake — **ten were pins, ten are findings.**

**THE ONE THAT MATTERED MOST, and it was not a test problem — RESOLVED, see below.**
`wheel_fea.MIN_WALL_MM` still defaulted to **2.0** while the promoted genome has
`t0 = t1 = t2 = 1.2`, `t3 = 1.4614`. **All four thickness genes were OUTSIDE the repo's
default gene box**, at `z` = −0.100, −0.133, −0.133, −0.135. `wheel_stage3.descend`
projects its start into the box before stepping, so any driver that loaded
`best_solution.json` without `--min-wall 1.2` would have **silently lifted all four
thicknesses to 2.0 mm and optimised a different, heavier wheel without saying a word.**
That is what `test_a_failed_solve_is_a_step_reject_and_the_run_recovers` was actually
reporting when it said `iterate_unchanged: False` — a fault-injection test catching a box
problem, which is not what it was written for and is not something reading would have
found. §8 anticipated this exact shape of bug and
`test_a_start_below_a_raised_floor_is_projected_up_onto_it` describes it — for the 2.2 mm
arm. It became true of the shipped genome. **`MIN_WALL_MM` now defaults to 1.2.**

**Fixed here — ten pins, all verified passing.** In each case the test's INTENT survives
and the assertion was the thing that was wrong:

| test | what it had pinned | what it pins now |
|---|---|---|
| `test_the_hub_junction_..._filleted` | the hub splits into **2** fillet families | families account for every filleted edge; `r_built_mm` is the worst family's radius. Two families was the old genome's FALLBACK — asserting it made the fix look like a regression |
| `test_the_normalized_gradient_follows_the_moved_floor` ×2 | that the shipped genome violates `hub_overlap` | reads `best_solution_ga_beam.json`. Its own vacuity guard caught this, which is the guard working |
| `test_the_embed_difference_...` | −1.4% of the wheel | the absolute **36.36 mm²** gusset. The gap moved 36.595 → 36.501 mm²; only the denominator changed |
| `test_genome_hash_matches_manifest` | the GA/beam genome against the SHIPPED manifest | the shipped genome against the shipped manifest. **§10's decoupling missed this one assertion** and it was invisible while the two files were identical |
| `test_R_hub_is_dead_at_the_mesh_...` | `d(fillet_cap)/dR_hub > 0` at whatever ships | over-cap genome for the barrier, shipped genome for M7's mesh claim, **plus** a new assertion that the barrier is exactly 0.0 on a feasible design |
| `test_the_hub_cap_reproduces_the_measured_void` | 9.907 deg, measured on the OLD solid | the genome it was measured on. The void is 22.8 deg on the new design — a property of a design, not a constant |
| `test_R_eff_is_exactly_the_cap_more_than_one_rung_above_it` | shipped genome is above its cap | the over-cap fixture; the name was always the precondition |
| `test_the_fillet_cap_barrier_is_live_at_the_shipped_genome` | shipped genome is 0.45 mm over its slot | renamed `..._on_a_design_over_its_cap`; asserts live where it must bite AND zero where it must not |
| `test_kt_hub_is_priced_on_the_buildable_radius_...` | the hard-`min` branch | the over-cap fixture, since every hub assertion in it is the over-cap branch |

A new `genes_over_cap` fixture reads `best_solution_ga_beam.json` for the five cap tests.
That file was pinned by §10 so the golden test could not be re-baselined; it turns out to
be load-bearing for a second reason, which is that **it is the only genome in the repo
that is over its hub fillet cap.**

**A new test was added, for a regime nothing had ever landed in.**
`test_the_shipped_genome_is_inside_the_blend_and_is_priced_conservatively`. `R_hub` =
0.5790 against a cap of 0.6240 — under it, but by less than the smooth-min's blend width,
so `hub_fillet_r_effective` returns **0.5727**, about 1.1% below the radius OCC actually
builds. **This qualifies the `kt_error_pct` = +0.0% headline above and should be read next
to it**: that +0.0% is the EXPORTER comparing its own modelled Kt against its own built Kt,
and it is correct. The OBJECTIVE prices the same junction at Kt **2.0308** against the
exporter's **2.0235**. The blend can only pull `R_eff` down, so the constraint is
CONSERVATIVE, not optimistic — which is why this is a footnote and not a defect. The new
test pins the direction so a future change to the blend cannot quietly reverse it.

#### The ten that are findings, NOT pins — none of these has been touched

Each is the new design genuinely behaving differently. Re-tuning these thresholds would
convert a measurement into a green checkmark, so none of them has been.

> **§14 REVISED THIS TABLE AND THE HEADING OVER IT IS TOO STRONG.** Read §14 before
> trusting any row. Of the eight that were still red, only two survive as "the new design
> genuinely behaving differently" — the GNL gate and the hub compliance share. Three were
> the TESTS being wrong in ways the old genome also exposed once measured, and two were
> tests pinning a pathology. The rows below record what was measured on the day; the
> `what it means` column is the reading that §14 went on to check, and it does not hold
> for the arrival angle, the p99, the area order, the contact patch, or the mass budget.

| test | measured | gate | what it means |
|---|---|---|---|
| `test_a_failed_solve_is_a_step_reject_and_the_run_recovers` | `iterate_unchanged: False` | must be `True` | **the box problem above.** Fix the default, not the test |
| `test_the_arrival_angle_makes_the_junction_a_near_crack` | material wedge **315.4 deg** | `> 340` | the new junction is **materially less crack-like**. Good news — and it means `study_wheel_fea.py`'s convergence-rate explanation is now about a wedge the wheel no longer has |
| `test_peak_stress_diverges_but_the_field_converges` | p99 changes 0.028 → **0.016** | `d2 < 0.3·d1` | the plain-spoke p99 settles more slowly on a 1.2 mm wall. Probably wants a finer rung, not a looser gate |
| `test_the_correction_enters_at_first_order_in_the_load` | GNL at 1% load **0.205%** | `< 0.1%` | **the thin wheel is measurably more geometrically nonlinear.** The fitted exponent is 1.034, so the physics is right — it is the magnitude that moved |
| `test_the_correction_is_not_a_constant_over_the_design_space` | iso ratio **2.69** | `> 3.0` | M5's "the correction is not a constant" is weaker around this design. M5's conclusion is not overturned, but its margin is |
| `test_the_rim_band_holds_a_large_minority_of_the_compliance` | hub share **0.0321** | `< 0.03` | rim 0.306 and spoke 0.662 are both fine; the hub takes marginally more of a floppier wheel |
| `test_total_mass_matches_the_step_manifest_within_the_embed_difference` | mesh 44.36 g vs solid **47.63 g**, −6.9% | −3.0% to −1.2% | the gap is gusset (≈1.01 g, unchanged) **plus fillet material, which grew from ~0.68 g to ~2.26 g** because all 48 corners now build at full radius on a 36% lighter wheel. Explainable, but it is a budget that has to be restated, not a bound to widen |
| `test_area_converges_second_order` | orders 1.986 / 2.061 / **2.355** | `1.7 < r < 2.3` | the finest rung overshoots second order on the thin section |
| `test_random_directions_agree_with_the_adjoint` | worst rel **1.49e-5** | tighter | directional FD agreement degraded on the thinner geometry |
| `test_the_sampled_patch_extent_is_biased_not_merely_noisy` | axle drop moves **0.32%** between `n_quad` 6 and 20 | `< 0.1%` | the contact solve is more quadrature-sensitive on a softer wheel |

**Read the middle four together.** More geometric nonlinearity, a slower-settling stress
field, a less singular corner and a softer contact response are all the same wheel being
thinner and floppier. None of them says the design is unsafe — the constraint that governs
is `stress_utilisation`, which is 0.799 against 1.0. They say the MODELS were calibrated
on a stiffer part, and several of the conclusions written on this page were measured on
one.

#### THE FLOOR NOW DEFAULTS TO 1.2 (same day, decided after the gate)

`wheel_fea.MIN_WALL_MM` **2.0 → 1.2**. Three perimeters at a 0.4 mm nozzle rather than
five. §8 measured what the floor costs, §11 took the decision that the process can hold
1.2, §13 promoted a 1.2 mm genome — and until this change the box still said that genome
was infeasible. It is still settable per run (`set_min_wall`, `--min-wall`); what moved is
the default, and **a default's job is to describe the wheel that ships.**

**What this changes for anyone running anything.** The GA and every driver now search down
to 1.2 mm by default. A run whose numbers assume a 2.0 mm floor predates this — including
every Stage-2 elite and the GA/beam genome, both of which were produced inside the old box.
Nothing on disk was regenerated.

**Two stale claims fell out of it, both corrected in place:**

- `wheel_objective.HUB_CAP_SHARE`'s comment said "`MIN_WALL_MM` is 2.0 and every design on
  disk sits at 2.468–2.627, so the calibrated band covers the reachable design space's
  lower half". Both halves were false the moment the genome was promoted, and the law is
  explicitly "CALIBRATED ON [2.0, 2.6] AND NOT KNOWN OUTSIDE IT" — the shipped genome sits
  at `t0` = 1.2, **below the band**. What saves it is which branch binds: the cap at
  `t0` = 1.2 is 0.624 mm = `HUB_CAP_THICKNESS_SHARE` × 1.2 exactly, so the THICKNESS term
  takes the `min` and the slot share is not being extrapolated on the shipped part. **A
  design that was thin AND had a tight slot would extrapolate it for real, and nothing
  would say so.** Re-run `make hubcap` at the new floor before trusting the slot branch
  under 2.0 mm.
- Two tests hardcoded `2.0` as "the default floor" (`test_gene_space.py`,
  `test_stage3.py`). Both now read the module. In each the claim was "unchanged" or "what
  the module says", and the literal quietly turned it into "still 2.0".

**The gate after the floor change: 8 failed, 423 passed** (was 20/410). The floor change
closed three by itself — `test_a_failed_solve_is_a_step_reject_and_the_run_recovers`, which
was the one reporting the box problem in the first place, plus
`test_random_directions_agree_with_the_adjoint` and
`test_the_correction_is_not_a_constant_over_the_design_space`. **It also broke one, and that
one is worth reading rather than re-tuning.**

#### `test_the_beam_to_wheel_ratio_is_not_a_constant`: the box moved, not the wheel

It failed at **2.686** against a `> 3.0` gate. The obvious reading — that the promoted
genome made Gate 1's headline weaker — is wrong, and one measurement settles it:

| genome | floor | `fea_over_beam_ratio` |
|---|---|---|
| `best_solution_ga_beam.json` | 2.0 | **4.943** |
| `best_solution.json` (shipped) | 2.0 | **4.943** |
| `best_solution_ga_beam.json` | 1.2 | **2.686** |
| `best_solution.json` (shipped) | 1.2 | **2.686** |

**Identical down the genome column, to every digit.** `run_beam_blindness` draws a Latin
hypercube from the GENE BOX and computes the statistic over the drawn rows only — the
`genes` argument is deliberately excluded from the statistics, and its own comment says why
("it was optimised and the others were drawn, so pooling them would understate the
spread"). The number is a property of the box; the test's fixture never touched it.

The mechanism is written in that same function: "feasible random spokes are typically
10–100x stiffer than the 2.0 mm target, **because the wall-thickness floor is what binds
there**." Dropping the floor to 1.2 lets the hypercube draw thinner, floppier spokes, which
moves the drawn population toward the beam target and COMPRESSES the max/min ratio.
4.943 → 2.686 is that compression, and it is the correct answer for the box the project now
searches.

**Gate 1's conclusion is not overturned.** `correction_factor_is_defensible` is still
`False` — that assertion passed — and a 2.7x spread is still not a constant. What breached
is the `> 3.0` margin, a number picked in a box with a 2.0 mm floor. The mirror image
happened on the gnl side: `test_the_correction_is_not_a_constant_over_the_design_space`
measured 2.694 in the old box and now passes. **Two tests of the same shape swapped sides
of the same threshold when the box moved.** Whether 3.0 is still the right margin at a
1.2 mm floor is a judgement about Gate 1's claim, not about this promotion, so it has been
left alone rather than tuned down to fit.

Everything else in the ten findings above is unaffected by the floor change — they are
properties of the geometry, not of the box.


### 14. THE EIGHT THAT WERE LEFT. Five open items, three closed by measurement (2026-08-06).

§13 promoted the wheel and left **8 failed / 423 passed**, deliberately untouched. This
section is the triage. It is not eight problems — it is five, and the ones that turned out
to be about the tests rather than the wheel have been fixed.

**The rule that governed all of it: measure before touching a threshold.** Every item below
was diagnosed by sweeping something (the mesh tier, the gene box, the reference) on BOTH the
promoted genome and `best_solution_ga_beam.json`, so that "the new design broke it" always
had to survive a comparison against the design it replaced. Three times it did not.

| # | item | status |
|---|---|---|
| 1 | two tests pinned a PATHOLOGY as an invariant | **CLOSED** — both inverted |
| 2 | the mass test asserted a budget it could not measure | **CLOSED** — exporter now publishes the fillet term |
| 3 | four convergence failures, one suspected common mechanism | **CLOSED** — hypothesis wrong; three separate causes found, two fixed |
| 4a | the pre-registered GNL gate | **DECIDED — it stands, and linear kinematics is the real question** |
| 4b | the hub compliance share | **OPEN — a human's call** |
| 5 | `make hubcap` at the 1.2 mm floor | **OPEN** |
| 6 | `EMBED_ALLOWANCE_PER_SPOKE_MM2` is stale (new, found by item 2) | **OPEN** |

---

#### Item 1 — two tests pinned a pathology as an invariant. CLOSED.

Same mistake §13 already fixed once, when `fillet_families == 2` turned out to be pinning an
OCC radius fallback as though it were a requirement.

**`test_the_arrival_angle_makes_the_junction_a_near_crack`** asserted `material_wedge > 340`
and measured **315.4**. Read the name: it pins the junction being nearly a crack. A design
whose spokes arrive less tangentially has a *less* crack-like junction, which is an
improvement, and it broke the test. Renamed to
**`test_the_junction_is_re_entrant_enough_to_be_singular`** and rewritten against a bound
that is a property of the BOX rather than of one design: `MAX_ARRIVAL_DEG` caps the arrival
angle for every genome the optimizer can reach, so the wedge is at least
`360 - MAX_ARRIVAL_DEG` = **295 deg everywhere**. That is all the sibling
`test_peak_stress_diverges_but_the_field_converges` needs, and it never has to be re-fitted
when a genome ships.

**`test_the_beam_to_wheel_ratio_is_not_a_constant`** — the 2x2 table in §13 proved the
statistic is a property of the gene box and does not depend on `genes` at all. It now calls
`set_min_wall(2.0)` for the duration of the draw, with a `finally` that restores
unconditionally (`tests/test_stage3.py` caches bounds in a module-scoped fixture, so a leaked
floor would surface somewhere else entirely). Gate 1's conclusion was never in question:
`correction_factor_is_defensible` is `False` in both boxes. What is pinned is the margin, in
the box the margin was calibrated in. **Re-deriving Gate 1 at a 1.2 mm floor is real work and
has not been done.**

Note `test_the_free_arc_fraction_is_not_constant_over_the_design_space` is the same shape —
a box statistic from the same `run_beam_blindness` call — and currently passes at 0.05. It
has been left alone rather than churned, but it is the next one to move if the floor does.

---

#### Item 2 — the mass budget was two unmeasured terms. CLOSED, and it exposed a stale constant.

`test_total_mass_matches_the_step_manifest_within_the_embed_difference` asserted
`-3.0% < m/manifest - 1 < -1.2%` and measured **-6.9%**. Its docstring decomposed the gap
into "~1.4% gusset plus 0.92% fillets" — and **the manifest published neither number**. Both
were fitted to one wheel. When the band broke there was nothing to look at.

The fillet term did not need estimating. `wheel_step_export` already builds
`wheel_nofillet.step` as its guaranteed-valid fallback, so the material the fillets add is a
subtraction OCC does exactly. The exporter now measures that solid's volume before
despecializing (same rule as `vol_true`, because the two are subtracted) and publishes:

```
solid.volume_nofillet_mm3   36042.6
fillets.volume_mm3           2372.53      = 6.176% of the solid
```

**6.18%, against the 0.92% the old docstring claimed.** That single correction is most of the
missing 6.9 points. Re-exporting was verified reporting-only: a key-by-key diff of the
manifest before and after shows **exactly two keys added and not one other value changed** —
`genome_hash`, `volume_mm3`, `mass_g_pla`, every `fillets.detail` row and `step_health` all
byte-identical.

The test is now a budget rather than a band: subtract the published fillet mass, and what is
left must be the gusset alone — positive, and under 1.5% of the solid. It measures **0.70%**.
It also pins the new field against a sign or unit slip (`fillets.volume_mm3` must equal the
difference of the two published volumes, and must be positive, because filleting a re-entrant
corner adds material).

**And that is how the stale constant surfaced.** Closing the budget with named terms left a
residual that would not go away under refinement:

| tier | mesh g | + fillet + gusset | residual |
|---|---|---|---|
| smoke | 44.2841 | 48.2360 | **-0.6060 g** |
| coarse | 44.3562 | 48.3081 | **-0.6781 g** |
| medium | 44.3641 | 48.3160 | **-0.6860 g** |
| fine | 44.3668 | 48.3187 | **-0.6887 g** |

Converging, not shrinking — so it is not discretization. It is
`EMBED_ALLOWANCE_PER_SPOKE_MM2 = 3.03`, and the promoted genome's actual gusset is
**0.98 mm² per spoke**. See item 6. The rewritten test does not use the constant at all.

---

#### Item 3 — four convergence failures. The common-mechanism hypothesis was WRONG.

The hypothesis was element aspect ratio: `MeshConfig` fixes `n_thick` and `n_span` as
*counts*, so root element thickness went 2.4774/6 = 0.413 mm → 1.2/6 = 0.200 mm while
spanwise length did not move, doubling the slenderness of every root element. Plausible, and
it would have explained all four at once. **It explains none of them.** Three separate causes,
found by three separate sweeps:

**`test_area_converges_second_order` — the reference, not the mesh.** `n_thick` has *literally
zero* effect on this statistic (8, 16, 32 give bit-identical orders — the area of a quad block
does not care how it is subdivided through the thickness). Extending the sweep two levels past
where the test stopped shows what is actually happening:

| genome | vs the beam reference | self-referenced |
|---|---|---|
| `350f4c7` | 1.986 2.061 **2.355** 5.103 **-3.300** | 1.962 1.979 **2.000** 1.998 |
| `36aed36` | 1.993 2.033 2.187 3.158 0.029 | 1.979 1.986 **2.001** 2.001 |

An order of 5.1 and then minus 3.3 is not a convergence rate — it is a difference of two
discretizations passing through zero. `reference_area` is `wheel_fea`'s beam-style line
integral: independent code, which is what makes it a good CROSS-CHECK, and an approximation
carrying its own error. The mesh converges at **exactly 2.000 on both genomes**. Nothing broke;
the promoted wheel has a smaller section (52.9 vs 145.7 mm²) so its absolute error reaches the
reference's floor one refinement sooner. **The GA/beam genome was already one level from
failing this.** Split into two tests — a self-referenced order, and a Richardson-limit
agreement with the beam integral to 1e-5 — and both got *tighter*, not looser.

**`test_peak_stress_diverges_but_the_field_converges` — the test's own documented false alarm,
one tier up.** It asserted `d2 < 0.3*d1` on the p99's successive differences.

| genome | smoke | coarse | medium | fine | d2/d1 | d2/p99 |
|---|---|---|---|---|---|---|
| `350f4c7` | 18.327 | 17.274 | 17.246 | 17.230 | **0.573** | 0.094% |
| `36aed36` | 8.842 | 8.782 | 8.612 | 8.605 | 0.042 | 0.082% |

`d2/d1` fails; `d2/p99` says the p99 has settled to under a tenth of a percent. A
successive-difference ratio is a divergence detector that only means anything while `d1` is
still real discretization error — once converged, both are tail and the ratio is arbitrary.
The old comment already named this failure mode at `smoke`, and the GA/beam genome **fails the
identical test at 2.816 if the window starts one tier lower**. The window had to be hand-picked
per design, which is the tell. Now pinned as the CONTRAST the docstring is actually about —
the max grows 38.9% over coarse..fine while the p99 moves 0.26%, a ratio of relative drifts
that is dimensionless and does not care which tier a design converges on.

**`test_the_sampled_patch_extent_is_biased_not_merely_noisy` — genuinely resolution, and it
clears at `coarse`.**

| genome | smoke | coarse | medium |
|---|---|---|---|
| `350f4c7` | **-0.3233%** | -0.0170% | +0.0195% |
| `36aed36` | -0.0275% | +0.0001% | +0.0030% |

On `smoke` the promoted genome's contact patch is `patch_half_deg` = 0.53 deg against a rim
element several times that: the whole contact set lives inside one element, so going from 6 to
20 Gauss points changes which elements are loaded at all. The failure is smoke-only and clears
by **5x** at `coarse`. That one test now builds its own `coarse` mesh; everything else in
`test_contact.py` stays on `smoke`, because everything else in it is a claim about the
*direction* of a bias, which holds at every tier.

**`test_the_correction_enters_at_first_order_in_the_load` — not resolution. Real.** See item 4.

---

#### Item 4a — THE GNL GATE. DECIDED: IT STANDS, AND IT IS THE MOST IMPORTANT THING §14 FOUND.

Swept for mesh sensitivity first, and there is none:

| genome | smoke | coarse | medium |
|---|---|---|---|
| `350f4c7` | 0.2050% | 0.2081% | **0.2089%** |
| `36aed36` | 0.0373% | 0.0382% | **0.0384%** |

Converged by `coarse`, mesh-independent to three digits, fitted exponent 1.037 against 1.011.
The correction still enters at first order — **what moved is the coefficient, not the
exponent**, and the assertion the test is named for passes.

**But 0.1% vs 0.2% at 1% of load is not what this is about.** `rel_diff = c·f^1.03`, so the
1%-load number and the service-load number are the same fact stated twice, and the gate is a
tripwire for the second one. The full ladder at `coarse`:

| load | linear mm | SVK mm | rel_diff | | GA/beam rel_diff |
|---|---|---|---|---|---|
| 0.01x | 0.019530 | 0.019570 | +0.208% | | +0.038% |
| 0.25x | 0.488241 | 0.514371 | +5.352% | | +0.963% |
| 0.50x | 0.976483 | 1.084109 | +11.022% | | +1.943% |
| **1.00x** | **1.952966** | **2.408898** | **+23.346%** | | **+3.953%** |
| 2.00x | 3.905931 | 5.939932 | +52.075% | | +8.193% |
| 3.00x | 5.858897 | 10.909429 | +86.203% | | +12.758% |

The GA/beam column reproduces M5's recorded numbers exactly (0.038% / 3.95% / 12.8%), which is
what makes the other column trustworthy. **The shipped wheel's axle drop at service load is
23.3% larger under SVK than the linear model says**, against 3.95% for the wheel it replaced.

**What that costs, in the two numbers the design is judged on.** Every headline for this
genome was computed under `kinematics="linear"` — `wheel_contact_problem`'s default:

- **Deflection.** `TARGET_DEFLECTION_MM` is 2.0 and the objective wants to hit it *exactly*.
  Linear says 1.953 mm — 2.4% under target, essentially on it. SVK says **2.409 mm, 20% over**.
  The design was tuned to a target it does not actually hit.
- **Stress utilisation.** The 0.799 headline is a linear-field number. Plain-spoke p99 goes
  17.274 → 19.746 MPa under SVK, **+14.3%**, which puts utilisation on the order of **0.91**
  against an allowable of 1.0. *(An estimate: it scales the reported figure by the p99 ratio,
  while the constraint itself aggregates a p-norm at p=30 on the `medium` rung. The margin
  falls from ~20% to ~9%; it does not vanish.)*

**So the gate has not been moved, and the case for moving it is now weaker, not stronger.**
`study_gnl.py` records it as *"written down BEFORE the study was run, per the plan's rule"* —
and the thing it was written to catch is exactly the thing that happened. Relaxing 1e-3 to
accommodate 0.209% would silence the only automatic warning that **linear kinematics no longer
describes this part.** The real open question is not the threshold; it is whether Stage 3
should be descending on a linear solve at all for a wall this thin. That is a scope decision
and it is a human's.

##### The footgun this turned up — FIXED

The first attempt at the stress half of that measurement returned **+169.5%**, and it was
wrong. `wheel_fem.gauss_stresses` takes `nonlinear=False` by default, and
`study_wheel_fea.stress_report` was calling it without the argument — correct for the linear
solves it was written for, and **silently wrong for an SVK one**, applying the
engineering-strain formula to a large displacement field. No warning, no NaN, and 46.56 MPa is
plausible enough to quote against a 25 MPa allowable. The true figure is 19.75.

`stress_report` now reads `res["meta"]["kinematics"]`, which `wheel_problem` sets down both
paths, and passes `nonlinear=` accordingly. Pinned by
`tests/test_gnl.py::test_stress_recovery_follows_the_solves_kinematics`, which asserts the
correct recovery AND that the wrong one is still visibly different — a guard that stops being
a guard if the two ever converge.

#### Item 4b — OPEN, still a human's call. The hub compliance share.

`test_the_rim_band_holds_a_large_minority_of_the_compliance`:
hub share **0.0321** against `< 0.03`, 7% over. Rim 0.306 and spoke 0.662 are both comfortably
inside. This is the one where the *direction* is surprising: thinner, floppier spokes should
push compliance toward the spokes and the hub share DOWN. It went up. The plausible cause is
`R_hub` dropping 1.5598 → 0.5790 — much less material at the hub junction — but that is a
hypothesis and it has not been measured. **Least urgent of the eight and the only one whose
sign is not understood.**

---

#### Item 5 — OPEN. `make hubcap` at the 1.2 mm floor.

Carried forward unchanged from §13. `HUB_CAP_SHARE` is calibrated on `t0` in [2.0, 2.6] and
the shipped genome sits at 1.2. What saves it today is *which branch binds* — the thickness
branch, at 0.52 × 1.2 = 0.624 mm exactly — so the slot share is not actually being
extrapolated on this part. A design that were both thin AND tight-slotted would extrapolate it
for real and nothing would say so.

---

#### Item 6 — OPEN, and NEW. `EMBED_ALLOWANCE_PER_SPOKE_MM2` is a `t`-proxy, exactly like `MIN_JUNCTION_OVERLAP_MM3` was.

Fell out of item 2. The constant is **3.03 mm²**; the promoted genome's measured gusset is
**0.98 mm² per spoke** (settling from 1.01 at `coarse` to 0.978 at `fine`). Root thickness
fell by 2.06x and the gusset fell by 3.1x, so it is not a constant and it is not linear in `t`
either — the same shape of error §12 removed from the junction check, where a fixed mm³ floor
turned out to be a proxy for `t0` and nothing else.

**Do not guess a new number.** Replacing 3.03 with 0.98 would only re-stale it on the next
genome; what is needed is the scaling law, derived from `wheel_step_export._embed` the way
`wheel_geometry.junction_bite` was derived. The exporter now publishes
`solid.volume_nofillet_mm3`, which makes the true gusset measurable per genome for the first
time — subtract the mesh's own modelled volume from it — so the calibration data is one export
away for any design on disk.

**One thing to know while it stands.**
`test_the_embed_difference_from_the_shipped_step_is_the_known_amount` compares
`reference_shipped_step_mm2 - total_modelled_mm2` against `n_spokes * 3.03`, and
`reference_shipped_step_mm2` is *defined* as `reference_modelled_mm2 + n_spokes * 3.03`. Since
the mesh agrees with `reference_modelled_mm2` to 4e-5, that test is very nearly tautological:
it checks the mesh matches its own geometric reference, which is worth checking, but it cannot
see the constant being wrong and did not.

---

#### The gate after all of it

Item 1 closed two failures, item 2 one, item 3 three. **`make test`: 2 failed, 430 passed in
1452 s (24:12)** — nothing else meaningful on the box, and the export and the probe sweeps had
already finished. (423 → 430 rather than 429: splitting `test_area_converges_second_order` into
a convergence claim and a cross-code claim added a test.) The two reds are the GNL gate and the
hub compliance share, both item 4, both deliberately left.

Item 4a then added `test_stress_recovery_follows_the_solves_kinematics`, so the count should
be **2 failed / 431 passed**; that arithmetic has NOT been confirmed by a full run. What was
re-run after item 4a is `tests/test_gnl.py` and `tests/test_wheel_fea.py` — the two files it
touched — and both come back with only the two known reds. Nothing was
re-tuned to fit: of the six thresholds touched, four got tighter (the area order is now
measured against 2.000 rather than a contaminated reference, the beam ratio is pinned in its
own box, the p99 gained a 10x separation requirement, the mass budget went from a fitted 1.8
point band to a named 1.5% bound) and two moved a test to a mesh that resolves what it is
measuring.

---

### 15. STAGE 3 WAS DESCENDING ON THE WRONG PHYSICS. It can now descend on the right physics, and the wheel that descent finds cannot be built. **NOTHING PROMOTED** (2026-08-10).

**BEFORE THIS MILESTONE, EVERY STAGE-3 NUMBER IN THIS REPO WAS A LINEAR-KINEMATICS NUMBER.**
`wheel_contact_problem` defaults to `kinematics="linear"` (`src/wheel_fem.py:1693`) and
nothing in the Stage-3 path had ever overridden it. That is the one sentence to carry out of
this section. Those numbers are not *wrong* — they are answers to a different question, and
§14 item 4a is where the question got asked.

**`best_solution.json` IS UNCHANGED and still holds `350f4c7`.** Nothing was promoted,
nothing was re-baselined, and the banner at the top of this file still describes the shipped
wheel. `export/wheel.step` was regenerated from that same unchanged genome (manifest
`genome_hash` `350f4c7`) purely to obtain a control for the export check below.

The working notes are **`SVK_PLAN.md`**, seven steps, each with its own pre-registered gate
and its own Record block. This is the summary; that file is the evidence.

#### What landed in the code, and it is small

The plumbing already existed — `kinematics` rides `**problem_kw` from `Evaluator` all the way
to `wheel_contact_problem`, the adjoint kernels already dispatch on `prob.nonlinear`
(`src/wheel_adjoint.py:161, 190, 400`), and `wheel_pool_worker.py:66` already splats it. What
was missing was a CLI flag.

- **`src/wheel_stage3.py`** — `--kinematics {linear,svk}`, default `linear`, forwarded to
  **both** optimizers, recorded in `search_block` and in the run record's settings, and
  **printed in the console banner** (`:954`). The record reads `ev.problem_kw` — the very
  dict the `Evaluator` splats into the solver — so the record cannot disagree with what was
  solved. `search_block` has **no `getattr(args, "kinematics", "linear")` fallback** on
  purpose: a default there would report "linear" for an SVK run whose caller forgot the
  field, which is the exact misattribution the key exists to prevent.
- **`studies/study_gradient.py`** — `--kinematics`, threaded as a plain keyword argument
  through all 15 call sites that build a problem or a solve, **never as module state**, and
  recorded in `rep["settings"]`.
- **`studies/study_svk_rescore.py`** + `make svk` — scores any set of genomes under **both**
  kinematics with no optimizer involved. Deliberately **out of `make studies`**, for the
  reason `m8bi5`/`m9buck`/`hubcap` are: it measures the wheel, not the commit.
- **`make svk-shipped` / `svk-elite10` / `svk-medium`** — the descents, modelled on
  `prod9`/`prod10`, with distinct `--out` AND `--best-out` so two runs cannot clobber each
  other's genome.
- **Tests** — the S13 pooled-equals-serial contract under SVK (sharing one helper with the
  linear test so the two kinematics cannot drift into two standards, plus a sentinel
  asserting the two kinematics return *different* answers, without which the equivalence
  would hold no matter what the pool did with the key), and
  `test_the_run_record_carries_the_kinematics_it_actually_descended`.

**`--kinematics linear` is inert**, proved where it is checkable: `linear` is exactly what
`wheel_contact_problem` already defaults to. Built at `smoke` on the shipped genome, 15
problem fields compared — the control (default vs default) differs on `contact` and `dofmap`
by object identity alone, the test (default vs explicit `linear`) matches the control
exactly, and a **sentinel** (default vs explicit `svk`) also moves `meta` and `nonlinear`,
which is what gives the comparison the power to see a real change.

#### THE PREREQUISITE: the adjoint is correct under SVK. All ten gates, thresholds unmodified

M7's gate had only ever been run under linear kinematics. Two full `coarse` runs, this tree,
this genome:

| gate | SVK | linear | threshold |
|---|---|---|---|
| **G1 unrolled** | **5.893e-11** | **4.555e-11** | **1e-8** |
| G2 force identity | 2.970e-11 | 2.462e-11 | 1e-12 (residual exactly 0.0 in both) |
| G3 mesh coords mm | 3.553e-14 | 3.553e-14 | 1e-9 |
| G4 plateau rel / decades | 5.616e-06 / 2 | 2.910e-06 / 3 | 1e-4 / ≥1 |
| G5 directional | 3.379e-06 | 8.056e-06 | 1e-5 |
| G6 sweep median | 6.626e-06 | 6.298e-06 | 1e-3 |
| G9 secant | 1.079e-06 | 3.576e-07 | 1e-5 |

**G1 is the one that decided the arc.** It unrolls the Newton loop and differentiates it with
`jax.grad`, so it contains no finite difference anywhere and its tolerance is set by linear
algebra rather than by step size. The SVK adjoint reproduces brute-force differentiation of
its own SVK solve to 5.9e-11 against a 1e-8 gate. **SVK is not uniformly the
worse-conditioned side** — it is 2.4× better on G5 — and no gate was given any slack.

Two things this step found on the way, both of them the "always measure the control" rule
catching a live error:

- **`--quick` IS NOT A GATE.** The quick SVK run came back `OVERALL: FAIL` on G5/G6/G7/G9.
  The same run under **linear** also comes back `OVERALL: FAIL`, on the committed default:
  it is a reduced-fidelity smoke mode whose step ladder drops the rungs where the FD plateau
  lives. G5 reads 1.588e-05 linear against 1.579e-05 SVK — SVK marginally the *better* of
  two failures. At full fidelity both G7 and G9 are clean and SVK is again the better side
  on G9.
- **`studies/study_gradient.json` IS STALE, and is NOT refreshed by this arc.** 3755 of 4239
  non-timing leaves differ from a fresh run at the same settings, including the physics
  (linear axle drop **1.8746 mm** fresh against **1.6546 mm** committed). The dates say why:
  the artifact is 2026-08-03, `best_solution.json` was replaced 2026-08-06. **The committed
  report describes a wheel that is no longer in the file it names.** This arc used it as a
  control for about an hour and reported a +39.6% SVK difference on the strength of it; the
  real figure, against a control measured in the same session, is **+23.26%** — which
  independently reproduces §14's +23.346%.

**And one repair to `study_gradient.py` itself.** `_phase_sweep` carries a converged
displacement field across a 0.5 deg phase jump, and under SVK that can land where `K(u)` is
genuinely indefinite (`NewtonDivergedError`, "the tangent is not positive definite") at a
phase where a **cold solve converges without complaint**. Under linear kinematics it cannot —
§0's H2(a) measured `K`'s dependence on `u` at exactly 0.000e+00. Repaired by falling back to
a cold solve and **returning the phases where that fired** (`cold_retry_phases_deg`) rather
than swallowing them: it changes which starting guess is used, never which equilibrium is
reported. It fires 8 times at `smoke` and **0 times at `coarse`** — a smoke-mesh phenomenon.
Stage 3 is not exposed to it: its cross-step warm start is *scalar indentations* seeding a
secant, not displacement fields carried across phases.

#### WHAT SVK COSTS: 1.36× time, 1.05× memory — and the penalty is in the GRADIENT

`coarse`, 8 phases, `uniform`, fidelity off, from `best_solution.json`, s/eval read off
`elapsed_s / n_objective_calls` in the run record rather than hand-timed.

| | s/eval serial (all/steady) | s/eval, 4 workers | Newton iters/solve | peak anon RSS, 4 workers |
|---|---|---|---|---|
| linear | 193.7 / 129.2 | 72.5 / **45.9** | 26.00 | 12.56 GiB |
| svk | 234.2 / 165.7 | 91.1 / **62.3** | 26.75 | 13.16 GiB |

"steady" drops call 0, which is JIT warm-up and is paid once per run. **SVK adds almost no
Newton iterations** — contact already spends 26 of them under linear kinematics, because
`wheel_fem.solve` routes on `prob.nonlinear or prob.contact is not None`
(`src/wheel_fem.py:1274`), so the Newton loop was always there. Backtracks actually *fell*,
8 → 4. The forward solve is only **1.13×** while the full evaluation is **1.36×**, so most of
the penalty is the SVK tangent assembly and the nonlinear vJP kernels. **There is no
iteration count to trade away**, which is worth knowing before anyone tries to buy the cost
back with a solver tolerance.

Two unplanned corroborations that the rig measures what the Makefile's own numbers were
measured on: 12.56 GiB for a 4-worker linear descent reproduces `make prod10`'s help text
("~12.7 GB anon"), and 3.9 h projected for a 300-step linear descent reproduces `prod9`'s
"~4 h". **The 62.3 s/step projection then held twice**, on two 300-step runs, at 58.2 and
59.1 s/step — 6.6% conservative.

Pooled-equals-serial holds under SVK: values **bit-identical**, gradients within 1e-14.

#### THE RE-SCORE: §14's "~0.91" was pessimistic, and the design ranking INVERTS

`make svk`, `medium`, 8 uniform phases, 47.8 min, six distinct designs under both kinematics.
The driver is gated on a **pre-registered control** (`GATE_CONTROL_REL = 0.02`) that
reproduces §14's ladder to five significant figures — 23.346% and 3.953%, errors 1.6e-5 and
2.6e-5 — so the table is comparable to §14 rather than merely internally consistent. A
*second*, unplanned control landed with the first row: `350f4c7 linear` came back
`loss 32.73762364435313` — every digit of the committed `stage3_minwall_best_1.2_medium.json`,
written five days earlier by a code path this driver shares nothing with.

| genome | kin | drop mm | err % | util | mass g | loss |
|---|---|---|---|---|---|---|
| `350f4c7` shipped | lin | 2.0193 | +0.97% | 0.799 | 39.19 | 32.7376 |
| `350f4c7` shipped | **svk** | **2.3947** | **+19.74%** | **0.875** | 39.19 | **129.8963** |
| elite10 | lin | 2.0041 | +0.21% | 0.594 | 58.66 | 49.7265 |
| elite10 | svk | 2.1418 | +7.09% | 0.624 | 58.66 | 62.2812 |
| minwall 1.4 | svk | 2.2847 | +14.24% | 0.799 | 42.82 | 86.0519 |
| minwall 1.6 | svk | 2.2146 | +10.73% | 0.712 | 47.03 | 67.7961 |
| minwall 2.0 | svk | 2.1416 | +7.08% | 0.624 | 58.55 | 62.2346 |

*(`minwall 1.2` **is** `350f4c7` bit-for-bit, `max|Δgenes| = 0.0`; the two rows agree on every
float but `elapsed_s`, which is a free determinism check on the rest of the table. `36aed36`
is infeasible under **both** kinematics — pre-registered as expected, it predates the
fillet-cap work — and is a control on the correction, not a candidate.)*

**Three findings, and the third is the one with the blast radius.**

1. **SVK moves exactly ONE term.** `mass`, `smoothness`, `phase_ripple` and all nine barriers
   are **bit-identical** under both kinematics; `deflection` carries the entire difference —
   0.2330 → **97.3916** on the shipped genome, 418×, and 75% of its total loss. Nothing else
   in the objective is a function of the strain measure, which is why the `stress` *barrier*
   stays at 0.0 even as utilisation climbs 0.799 → 0.875.
2. **The correction is a function of the DESIGN, not a constant.** It falls monotonically with
   stiffness — +18.6% → +13.4% → +10.2% → +6.9% across the four `minwall` arms, reproducing
   on a fifth design that reached similar stiffness by a different route (elite10, +6.87%).
   §14 measured it on two genomes and quoted +23.346%; **it is not one number, it is a curve.**
3. **The ranking INVERTS on every comparison in the table.** The linear loss column is
   monotone *increasing* in wall thickness; the SVK column is monotone *decreasing*.
   `350f4c7` was promoted over elite10 because it won by 17 points under linear; under SVK it
   loses by 68. **The `minwall` sweep of §8 that chose the 1.2 mm floor ran entirely under
   linear kinematics, and this table reverses its ordering over the whole 1.2–2.0 mm range.**

**And the gate's own question, answered: the shipped genome IS FEASIBLE under SVK.** Every
barrier exactly 0.0, utilisation **0.8754** against an allowable of 1.0. §14's "on the order
of 0.91" was an estimate built by scaling a reported p99 by a ratio; the number the
constraint actually computes is 0.875. **§14 was pessimistic by ~4 points of margin.**

Read the `p30 util` column in `studies/study_svk_rescore.json` as a diagnostic and nothing
else — it sits at 10.747 for the shipped genome, and if the constraint used p=30 the promoted
wheel would be infeasible by 10×. It is not mesh-convergent (GCI 63%, M8b-i.5), which is
exactly why `STRESS_NOMINAL_P = 4.0` is the constraint. The driver asserts its p=4 probe
reproduces `stress_utilisation` to 1e-12, so the two columns are the same construction
differing only in the exponent — the line that stops the diagnostic and the verdict drifting
apart, which is precisely how "~0.91" came to stand in for a number nobody had computed.

#### THE DECISION, taken on those numbers and written down before anything launched

**Descend under SVK.** Three things had to be true and all three were measured, not assumed:
the SVK adjoint is correct (G1, unmodified); it is affordable (1.36×, 5.3 h per start, 16G);
and **there is something to descend to** — three Adam steps under SVK took the shipped genome
117.766 → 33.436 and pulled the drop 2.369 → 1.982 mm for +0.66 g, while the same three steps
under linear could not improve on the start at all. **The shipped genome is a local optimum of
the LINEAR objective and is demonstrably not one under SVK.**

Re-targeting `TARGET_DEFLECTION_MM` under linear was rejected not on the anticipated ground
(the correction is load-dependent, `c·f^1.03`) but on finding 2: **the correction is a
function of the design**, so a re-targeted constant is calibrated to the design that existed
when you calibrated it, and the optimizer's whole job is to move the design. It is not merely
approximate, it is unstable under the thing it is meant to enable. Accepting 2.409 mm was
*available* — that is the feasibility answer above, and it was not available before this arc —
and was rejected on value: a 19.74% spec miss while the same table holds feasible designs at
+7% and a 3-step probe moves the shipped genome most of the way for 0.66 g.

**The descents held the 1.2 mm floor**, deliberately, even though the ladder says the SVK
optimum over the *measured* designs is at the thick end. A floor is a **constraint, not a
target** — if SVK wants thicker walls it can walk there from inside the same box — and holding
the box identical across both starts is what makes them comparable. The outcome is
informative either way. It came back binding; see the successors.

#### THE DESCENTS: two starts, both pass, and the answer is a SHELF not a point

Two 300-step `coarse` descents, sequential, each under `systemd-run --user --scope -p
MemoryMax=16G -p MemorySwapMax=0 --collect`, from starts 19 g apart.

| run | start | steps | loss | mass g | defl err (svk) | util | worst barrier | wall |
|---|---|---|---|---|---|---|---|---|
| 1 `ae7092c` | `350f4c7` | 300/300 | **30.8207** | **37.451** | **−0.043%** | 0.8989 | **0.0** | 4.90 h |
| 2 `c4f207c` | elite10 | 300/300 | 30.8245 | 37.449 | −0.044% | 0.9085 | **0.0** | 4.94 h |

**Both clear all three pre-registered clauses, unmodified** — every one of the nine barriers
exactly 0.0, deflection 7× inside ±0.3%, and mass below the shipped 39.194 g. Run 1 took the
loss 117.766 → 30.8207 with **0 rejected steps and 0 events**; the descent never fought the
line search. Term by term, step 0 → 300: `deflection` 85.2617 → **0.00047**, `mass` 32.2145 →
30.7820, `smoothness` 0.2902 → 0.0382, every barrier 0.0 throughout.

> **The gate pre-registered its own likely breach and it did not fire.** Written before the
> re-score, the mass clause was expected to fail — if SVK's answer to a 19.74% deflection miss
> is more material, "mass below 39.194 g" asks the descent to beat the shipped wheel on the
> one axis it was over-optimized on. Instead **SVK did not cost this design grams; it saved
> 1.74 g** while moving deflection from 2.369 mm to 2.000 mm. The clause "SVK costs this
> design N grams" does not fire, and it is recorded here because it was written down first.

**The two runs agree to 0.012% on loss and 0.008% on mass, and that is NOT a shared point
optimum.** The genomes differ: `cx1` by 7.28%, `cy1` by 6.93%, `cy4` by 4.27%, while
`t0..t3` sit on the 1.2 mm floor in both and `R_rim` is identical to six decimals. **The
objective is flat along a manifold** — the spline control points move up to 7.3% for a
fourth-decimal change in loss. Report it as "both starts reach the same basin and the same
headline numbers", never as "the optimizer found THE answer".

Run 1 was **reproduced through an independent driver** while run 2 was in flight
(`study_svk_rescore.py --extra`, additive so the driver at its defaults still reproduces the
Step 3 artifact unchanged): drop 1.9991, util 0.8989, loss 30.8207, every digit. So the
Stage-3 run record is not reporting an internal state the saved genome does not encode.

**Run 2 hit one `solve_reject` at step 128, handled, and it is worth a line.**
`solve_wheel_contact` (`src/wheel_fem.py:1841`) is a secant on indentation with
`tol_rel=1e-8`; it stalled at 66.723265 N against a 66.7233 N target — a residual of
**5.2e-7 relative, 52× above the tolerance it is asked to hit**. The load is physically
reached to within a part in two million. What fails is the *outer* secant's ability to resolve
a force difference smaller than the noise floor of the inner Newton solve that produces it,
and SVK raises that floor. The function raises rather than returning the state, which is
correct and documented. `wheel_pool_worker.py:88-98` reports it to the parent as the
`solve_reject` that `descend` already knows how to handle — it prints a traceback and is not
a crash. **Run 1 is unaffected: 301 calls, `n_reject_cumulative` 0.** The tolerance was **not**
loosened; see the successors.

#### THE FIDELITY TRAP: the ±0.3% deflection gate is satisfiable at exactly ONE rung

The `coarse` candidate `ae7092c` was re-scored at `medium` before promotion, with the control
on — and it reads **+1.65%**, 5.5× the gate. Not promoted. That check was pre-registered in
the descent's own record before the descent finished: *"a deflection converged to −0.043% at
coarse is NOT thereby inside ±0.3% at medium ... that is a finding about the rung the descent
was run on, not licence to promote anyway."* The rule holds even though the check was one
this arc added rather than one the plan pre-registered — especially then.

**The control is what makes it readable, and it says the gate was never a `medium` gate.** The
INCUMBENT fails ±0.3% at `medium` too — **+0.97%**, under the very kinematics it was descended
on. **No design in this repo has ever met ±0.3% at `medium`.** So the response was to
re-converge at `medium`, not to move the number: `make svk-medium`, 100 steps warm-started
from `ae7092c`, 6.29 h, 224 s/step against a projected 273.

`bc77614` **passes every clause at `medium`**: all nine barriers 0.0, deflection **−0.041%**
(1.99919 mm), mass **37.414 g** (−1.781 g against the shipped 39.194). It is also lighter and
lower-loss than the coarse candidate it replaces.

**And the fidelity check, pointed back at `coarse`, gives the mirror image — which is the real
result here.** The medium answer reads **−1.71%** at coarse. The coarse answer reads +1.65% at
medium. **The two rungs disagree by ~1.7% on this wheel and no design can satisfy ±0.3% at
both.** Which rung the gate is stated against is a **choice, not a property of the design**;
this arc chose `medium` because it is the finer and because §14's control ladder is stated
there. The honest sentence is that the wheel is now specialised to a rung as well as to a
kinematics, and a third rung would move it again.

**A cheap process lesson.** Both descents ran with `--fidelity-check-every 0`. `descend` has
had the machinery for exactly this since §1 item 3 (`src/wheel_stage3.py:384`,
`_fidelity_check` at `:272`) — a pure observation that cannot redirect the descent, but with
`--fidelity-check-every 25 --fidelity-check-config medium` the coarse/medium gap would have
been on the record at step 0 instead of after 9.8 h of descending. Turning it off saved
perhaps 4% of wall clock and cost the arc a run.

#### WHY NOTHING IS PROMOTED: `bc77614` clears every FEA gate and is not buildable at the stress concentration it was priced at

`make export EXPORT_GENOME=stage3_svk_best_medium.json`, with the same export run on the
incumbent as the control this file's rules require:

| genome | worst wedge | hub fillets built | `kt_error_pct` |
|---|---|---|---|
| `350f4c7` shipped | 328.0 deg | 24/24 @ 0.579 mm | **0.0%** |
| `bc77614` svk-medium | 308.0 deg | 12 @ 0.579, 12 @ 0.418 mm | **+11.9%** |

**The incumbent builds exactly as modelled. The candidate does not** — and the control is the
only reason that sentence can be said with confidence. This is a regression this arc
introduced, not one it inherited. §13's +0.0% at both junctions was the first shipped part
whose built fillets matched the ones its stress model priced, and it is worth exactly this
much.

What it costs, in the only units that matter:

```
Kt at the hub    modelled 2.0235      as built 2.2643      +11.9%
peak stress      modelled 294.02 MPa  as built ~329.01 MPa
UTILISATION      modelled   0.9347    AS BUILT   1.0461    <- INFEASIBLE AS BUILT
```

Everything else in the export is clean, which is what makes the failure legible rather than
ambiguous: OCC valid, 1 solid, bbox 100.00 × 100.00 × 22.40 mm, BRepCheck valid, no
self-intersection, 0 degenerate, min curvature R 0.4184 mm against the 0.25 floor, junction
bite floor satisfied. **Only the fillet feasibility is red.**

**Lowering `R_hub` by hand until modelled == built was rejected.** It is fitting the geometry
to the check after seeing the check fail; it would have to be done by hand *precisely
because* the optimizer cannot do it; and it leaves the same blind spot in place for the next
design. The defect is that the objective cannot see buildability, and the fix belongs in the
objective.

#### THE FOUR DEFECTS IN THE OBJECTIVE, in the order they collected their debt

This is the part of §15 that outlives the numbers. All four were **measured** in this arc, and
all four were deliberately left alone, because acting on any of them mid-arc would have been
re-fitting a gate to the run that breached it.

1. **`stress` HAS ZERO GRADIENT BELOW `util = 1.0`.** It is `soft_barrier(util - 1.0, 4000)`
   (`src/wheel_objective.py:1027`) and `soft_barrier` is `scale * max(0, v)**2` (`:290`), so
   it is identically zero *and identically flat* for every `util <= 1.0`. Below the knee the
   optimizer cannot see stress at all; it sees mass, and it thins the wall. **The barrier is a
   wall to stop at, never a cost to trade against.** Measured: the `stress` term was > 0 on
   **0 of 602 descent steps**.
2. **THEREFORE `R_hub` AND `R_rim` ARE DEAD GENES.** The only paths from a fillet radius into
   the loss are `stress` and the fillet barriers, and both are identically flat unless
   breached. Over all 602 steps of both descents, **`R_rim` had a nonzero gradient on 0 steps
   and `R_hub` on 2** — and those two steps are *exactly* the two where the `fillet_cap`
   barrier was live. They stayed frozen to six decimals through another 100 steps at
   `medium`, which rules out the one benign explanation: it is not a coarse-mesh artefact.
   **Run 1's `R_hub` 0.5790 and `R_rim` 2.7495 are not optimisation results.** They are
   constants inherited from `best_solution.json` and carried untouched, and the same blind
   path was in place for every Stage-3 run behind §6, §8 and §14. The search is nominally
   14-dimensional; with all four thicknesses on the floor the live subspace is the **8 spline
   coordinates** — precisely where the two runs still disagree by 7.3%.
3. **THEREFORE THE DESCENT SWUNG THE HUB ARRIVAL SHALLOW WITH NOTHING OBJECTING** (wedge
   328 → 308 deg), and it could not have compensated by asking for a smaller radius either,
   because `R_hub` is exactly the gene it cannot move. The exporter's own diagnostic reaches
   the same place unprompted: *"what is left is the shallow corner of a near-tangent arrival,
   which is the arrival angle, i.e. the genome."*
4. **THE 1.2 mm WALL FLOOR IS DOING THE STOPPING, AND ITS JUSTIFICATION NO LONGER HOLDS.**
   Utilisation climbed monotonically through the descent and **plateaued at 0.899 for the last
   180 steps** — not because anything valued the remaining margin, but because at step 300 all
   four wall genes sit **on the 1.2 mm floor** (`final.bound_saturation`: t0..t3 all "low").
   The run ran out of wall to thin before it ran out of stress margin; set the floor lower and
   the same blind gradient would keep going. And the floor itself was chosen by §8's `minwall`
   sweep **under linear kinematics**, whose ranking finding 3 above inverts.

*(An in-flight extrapolation in the working notes projected utilisation reaching ~0.99 by step
300 from the 50→100 slope. It did not; it stopped 0.10 short of the knee. The mechanism above
is unchanged — what was wrong was the projection, and the run was saved by a different
constraint than the one that was worrying.)*

**Utilisation is the one number that got worse, and it should be read as the cost of this
arc:** 0.799 (shipped, linear, medium) → 0.875 (shipped, **svk**, medium) → 0.935 (`bc77614`,
svk, medium). Roughly **half of that is not a design change at all** — it is the correction
from measuring the same wheel honestly. The rest was spent by an optimizer that cannot see
stress below 1.0. Both halves are real.

One more caveat on the descent's own trace: `max_stress_mpa` grew **+37%** (160.6 → 220.2 MPa)
while the utilisation the constraint sees grew **+4.7%**. The two decouple because the
constraint is `Kt * pnorm(p=4) / 25.0` and a p=4 aggregate is deliberately insensitive to a
singular corner peak — that is what `Kt` exists to bridge, and p=30 is not mesh-convergent. The
p=4 number is the right one to gate on. But **the aggregate is tracking the field and the
corner moved more**, and that is the same corner the export then refused to fillet.

#### What is now stale, and what to read as a linear-kinematics number

- **Every Stage-3 deflection and utilisation number in §1–§14 is a linear-kinematics number.**
  §13's headline 0.80 utilisation and §14's 0.799 are `medium`-rung linear figures; the same
  wheel reads **0.875** under SVK. §13's implicit "2.0 mm deflection" is 1.953 linear and
  **2.409 SVK**. Nothing in those sections needs correcting — they need reading with the
  kinematics named.
- **`studies/study_gradient.json` is stale** for a second and unrelated reason (above): it
  describes the pre-`350f4c7` genome. Not refreshed here.
- **Every study driver in `studies/` still defaults to `kinematics="linear"`**, which remains
  the repo-wide default and was held so on purpose: every committed artifact must still
  reproduce bit-for-bit with no flag passed. A new default is a re-baselining and this repo
  does not re-baseline silently. `study_gradient.py` and `study_svk_rescore.py` are the two
  that can now be told otherwise.
- **`36aed36` is barely affected and that is not luck** — +3.95% against +23.3%. The
  correction tracks compliance, and the old wheel is 74 g of it. Any comparison between the
  two genomes under linear kinematics is comparing one number that is nearly right against one
  that is 20% off.

#### The gate

`make test` had not been run in full since the arc's baseline (**431 passed / 2 failed**,
1374.52 s, reproducing §14's predicted arithmetic exactly, both reds the deliberate ones —
the GNL gate at `small_load_rel_diff` 0.0020499 and the hub compliance share at
0.032076694850181206, each matching its recorded value to the digit).

**Re-run at the close of this arc, and it was measured rather than inferred: `make test` —
433 passed / 2 failed in 1468.57 s (24:28).** The arithmetic predicted 433/2 (Step 0's 431/2
plus exactly the two tests this arc added — `test_a_pooled_SVK_evaluation_matches_the_serial
_one` and `test_the_run_record_carries_the_kinematics_it_actually_descended`) and the run
confirms it. **The two reds are the same two, and the hub compliance share reproduces
0.032076694850181206 — every digit of §14's recorded value, and of Step 0's.** No new red,
nothing accommodated, and the +2 is fully accounted for. §14 had to close on an unconfirmed
sum; this one did not.

#### The successors, in priority order

All five were found by this arc and all five are named rather than acted on.

1. **MAKE THE OBJECTIVE SEE BUILDABILITY.** This is the one that has to be fixed before **any**
   SVK descent can ship, and it subsumes 2 and 3 as far as promotion is concerned. Nothing in
   the loss prices the hub arrival angle or the fillet the exporter can actually cut. The
   exporter already computes `kt_built` — the missing piece is a term that charges the
   difference, which would also give `R_hub` its first real gradient.
2. **GIVE THE OBJECTIVE A STRESS-MARGIN TERM.** Defect 1. It does not merely stop the optimizer
   spending margin it is not charged for — it **unfreezes two design variables that no run in
   this repo's history has been able to move**.
3. **RE-DERIVE THE MINIMUM-WALL FLOOR UNDER SVK.** Defect 4. The floor is load-bearing in every
   answer this arc produced and §8's justification for it is reversed by the re-score table.
4. **PUT A MESH-CONVERGENCE STUDY ON `axle_drop_mean_mm`.** Give the deflection QoI the GCI
   treatment M8b-i.5 gave the stress QoI, then state the ±0.3% gate against an extrapolated
   value instead of against whichever rung the descent happened to run on. Today the gate is
   satisfiable at exactly one rung and the choice of rung is undeclared.
5. **SET THE LOAD-CONTROL TOLERANCE FROM THE INNER SOLVE'S NOISE FLOOR.** `tol_rel=1e-8` in
   `solve_wheel_contact` is not universally achievable under SVK. Measure the floor, then set
   the outer tolerance from the measurement — **do not pick a looser round number**, and do
   not touch it on the evidence of the run that breached it.

#### Artifacts

Search results and measurements, none of them gates except where noted. At the repo root:
`stage3_svk_best_shipped.json` (`ae7092c`, run 1) and `stage3_svk_shipped.json` (its run
record), `stage3_svk_best_elite10.json` (`c4f207c`) / `stage3_svk_elite10.json`,
`stage3_svk_best_medium.json` (`bc77614`, the `medium` re-convergence) /
`stage3_svk_medium.json`. In `studies/`: `study_svk_rescore.json` (the Step 3 table, and it
**is** gated — `GATE_CONTROL_REL`), `study_svk_step6.json` (the same driver's `medium` check
on the coarse candidate, the one that stopped the promotion), `study_gradient_svk.json` and
`study_gradient_lin_check.json` (the ten adjoint gates under each kinematics, `"pass": true`
in both), plus their `--quick` counterparts, which are **not** gates and fail under both
kinematics. In `export/`: `stage3_svk_best_medium.step`, its `_nofillet` companion and its
manifest — **the artifact that refused**, kept because the +11.9% `kt_error_pct` is the arc's
terminal finding and re-deriving it costs an export.

**`export/wheel.step` was regenerated from the unchanged incumbent `350f4c7`** to obtain the
control for that comparison; its manifest `genome_hash` says so.

#### What this arc established, stated without the hedging

SVK is the honest kinematics for this part. The shipped wheel deflects **2.39 mm, not 1.95**,
and carries **0.875 of allowable, not 0.799** — and it is still feasible, which was not known
before. A design exists that is **12× closer to the deflection target and 1.78 g lighter**
under that kinematics. And the objective has **four named, measured defects** that together
explain why that design cannot ship. The deliverable of this milestone is a measurement, not
a promotion, and the plan pre-registered that outcome: *"a run that does not clear these is a
result, not a failure — record it and go back."*

---

### 16. THE OBJECTIVE CAN NOW SEE BUILDABILITY. **PROMOTED** — `best_solution.json` is `e4219f3` (2026-08-11).

§15 ended with a wheel that was 12× closer to the deflection target, 1.78 g lighter, and
**impossible to build**: `kt_error_pct` +11.9% at the hub, as-built utilisation 1.046, OCC
filleting 12 of 24 corners at the radius the optimizer had asked for. The deliverable was a
measurement and the instruction was to go back and fix the objective. This is that arc.

**The wheel that ships now builds 24/24 at both junctions at the full requested radius,
`kt_error_pct` +0.0% / +0.0%.** It weighs 37.568 g against 39.194, deflects 1.99996 mm against
a 2.0 mm target (**−0.002%**, where the incumbent was **+20.5%** once re-priced), and carries
0.996 of allowable priced / 0.987 as built. Every barrier is exactly 0.0.

#### The defect was not that the cap was missing. It was that the cap was WRONG, in both branches

`hub_fillet_cap_mm` existed since §5 and returned `min(by_slot, by_thickness)`. The `min`
structure was right. **Both of its arguments were the wrong shape.**

- `by_thickness` was `0.52 * t0` — a function of a gene that says nothing about the corner.
  §15's two candidate genomes have **identical `t0` = 1.2 and identical `R_hub` = 0.578951**,
  and OCC fillets all 24 hub corners on one and only 12 on the other. The old cap returned
  **0.6240 for both, to sixteen digits.** What they do not share is the **hub arrival angle**:
  19.68° against 48.89°. That was the missing variable, and it was already differentiable —
  `control_points` locks P0 at the origin, so arrival is `asin(|cx1| / hypot(cx1, cy1))`, a
  function of two genes and nothing else.
- `by_slot` was `0.5 * R_hub_ring * radians(void)`. It had **never been the binding branch
  before**, so its constant had never been tested. When the arrival fix made it bind, it was
  found to over-promise by up to **1.62×**. Re-fitted under the smallest of eight measured
  ratios, 0.3096.

The replacement, fitted on OCC ground truth and calibrated over [5°, 60°]:

```
by_thickness = t0 * (0.505 - 0.48 * (1 - cos(arrival_hub)))
by_slot      = 0.30 * R_hub_ring * radians(hub_void_deg)
cap          = min(by_slot, by_thickness)
```

`(1 - cos)` rather than a quadratic because it is bounded, monotone on [0°, 90°], and flat at
0° where a tangential arrival should be insensitive. The fit sits **under all 14 sweep
stations** (1.45–3.29% under thin, 4.00–11.61% under thick) and 3.3% / 3.1% under two
out-of-sample designs. **The conservatism is deliberate and it is the design margin** — a point
that matters again below. Cap ÷ OCC-worst across five designs went 1.067 / 1.615 / 1.199 /
1.463 / 1.479 → 0.979 / 0.969 / 0.827 / 0.967 / 0.969.

#### Two hub corner families, named by wedge, and the one that binds today

OCC's hub corners fall in two bands with a measured 24° gap: **SQUARE-ON** at 266–270°, limited
by root thickness and arrival, and **NEAR-CUSP** at 294–332°, limited by the slot. Which binds
is a property of the design, not of the wheel: square-on at `t0` = 1.2, near-cusp on the
`t0` = 2.55 elites. `study_hub_cap.CUSP_WEDGE_DEG = 285.0` splits them.

**The promoted wheel's worst hub wedge is 314.0° — NEAR-CUSP.** The arc was built around the
arrival branch and the arrival branch is what got the descent here, but **the slot branch is
what binds on the design that ships.** Its own arrival law is unfitted and parked (0.31 at
3.4° rising to 0.70 at 50°, ~2.2× on the table); that is now the live successor.

#### What the descent did, and the two headline results

Re-descending under SVK at `medium` with the corrected cap produced the arc's first real
finding immediately: **the incumbent `350f4c7`, re-priced on a fillet radius that can actually
be built, is INFEASIBLE at utilisation 1.051.** Its shipped metrics read 0.783 because they
were computed at `coarse`, under **linear** kinematics, against the superseded 0.624 mm cap.
Every number in that file came from a model this arc showed to be wrong in a specific way.
**Shipping it was the risky option; that is why the promotion happened.**

The first descent then failed its own pre-registered gate 3-of-4, with `R_hub` pinned at a box
floor of **0.5** that was a bare literal in `GENE_SPACE` with no comment behind it — twice the
exporter's `MIN_CURVATURE_RADIUS_MM`. The cap cleared at arrival ≤ 39.893° and the run
converged at 40.542°: **0.650° short, with no legal move left in that gene.**

The floor moved to **0.4 mm — one extrusion width** at the 0.4 mm nozzle `MIN_WALL_MM` is three
perimeters of. Deliberately **not** 0.25: `MIN_CURVATURE_RADIUS_MM` is a *fault detector*
("well under any fillet we ask for, so a violation always means a construction fault") and
`MIN_BUILDABLE_R_MM` exists only to keep a blend width positive. Adopting either as a design
floor would have destroyed the detector by putting legal designs on it. `R_rim` unchanged.

The re-descent, warm-started from the same control so it differed in **exactly one thing**,
put `fillet_cap` at **exactly 0.000000 by step 3** and `R_hub` roamed a 0.135 mm interior band.

Second headline, and a correction to one issued mid-arc: the first descent reported `t0` coming
**off the `MIN_WALL_MM` floor for the first time in the project** (1.2714). With `R_hub` free it
is **pinned low again**. The observation was real; the stated cause was too general. A thicker
wall buys buildable fillet **only when the radius cannot move.**

#### THE THREE DEFECTS THIS ARC ADDED TO §15's FOUR

§15 named four defects in the objective. **This arc fixed none of them** — it fixed the
*buildability model*, which is a different thing — and found three more. All three are about
the penalty formulation rather than the physics, and all three cost real time before they were
understood.

5. **A QUADRATIC SOFT BARRIER CANNOT CONVERGE TO EXACTLY ZERO UNDER OPPOSITION.**
   `soft_barrier` is `scale * max(0, v)**2`, so **its gradient vanishes at its own knee.** A
   larger `R_hub` lowers `kt_hub` and relieves `stress`; a smaller one relieves `fillet_cap`.
   The two push against each other and settle where their quadratic gradients cancel — which is
   necessarily **just inside both**, because neither can generate force at `v = 0`. The
   re-descent's selected best sits at `fillet_cap` 0.000546 and `stress` 0.000751, converged to
   six figures over its last seven steps, with `stress_utilisation` 1.00031. **This is a
   property of the formulation, not of the design**, and it is the same shape of finding as the
   Step 3 gate clause that had to be retired as unsatisfiable. It is the mirror image of
   §15's defect 1: below the knee the barrier is invisible, and *at* the knee it is powerless.
6. **`wheel_stage3.py` SELECTS `--best-out` BY LOSS AND IGNORES FEASIBILITY ENTIRELY.**
   55 of 101 iterates in the re-descent had every one-sided barrier at exactly 0.0. The one the
   code picked was not among them. Selecting the lowest-loss **feasible** iterate is the
   standard rule for a penalised constrained descent and arguably the only correct one; the
   difference here costs **0.030% of a loss that is 99.8% mass.** Recorded, not patched — a
   change to the selection rule is a change to every future run and belongs in its own arc.
   **Until it is fixed, never promote `--best-out` without re-checking feasibility by hand.**
7. **FEASIBILITY MUST BE CHECKED WITH SLACK AT EVERY FIDELITY, NOT AT ONE.** Step 82 — the
   lowest-loss strictly feasible iterate, and the first thing promoted — clears the cap at
   `medium` by **53 nm** and violates it at `coarse` by **93 nm**. Its feasibility depends on
   which mesh you ask. Worse, sitting on the knee of `max(0, v)**2`, whose second derivative is
   discontinuous there, makes a central difference straddle the kink: it fails
   `study_objective.run_closed_form` at **2.705e-06** against a 1e-6 tolerance. **G4 was not
   reporting a code bug. It was reporting where the design sat.** Step 71 was promoted instead:
   **2951 nm** of slack, 0.000e+00 on that gate, a better deflection (−0.002% vs +0.051%) and
   **more** stress margin (0.99639 vs 0.99998), for +0.13% of loss and +0.048 g.

   The reasoning that produced step 82 was that the cap is already fitted 1.45–11.61% under OCC
   truth, so padding the numerical slack would be inventing a second margin. **That is correct
   about buildability and irrelevant to numerical robustness** — OCC built step 82 24/24 at
   +0.0%. Two distinct failure modes; the argument for one was applied to the other.

#### The gene box changed, and that reinterprets history in exactly one gene

`GENE_SPACE[12]['low']` went **0.5 → 0.4**. Raw genomes on disk are stored **by name** and are
unaffected. **The normalized `z` traces inside historical Stage-3 run JSONs are not:** gene 12
now decodes to a different physical radius. Any pre-2026-08-11 `steps[i]["z"][12]` read with
today's `GENE_SPACE` is wrong by `0.1 * (1 - z)` mm. Decode with the box that was in force.

It also has a live consequence: **`test_the_beam_to_wheel_ratio_is_not_a_constant` is a
casualty of the box change, not of the promotion.** `run_beam_blindness` draws a Latin
hypercube from the box, so its statistic is genome-independent — measured at **4.943223** with
the 0.5 floor (matching §14's documented 4.943) and **2.412764** with 0.4, *identically for both
genomes*. Gate 1's actual conclusion, `correction_factor_is_defensible == False`, holds in both
boxes; only the `> 3.0` margin, calibrated in the old box, moved. **Not re-tuned** — the test's
own docstring is explicit that re-deriving it is a judgement about Gate 1, not a test edit.

#### The gate

`make test` closes at **6 failed / 430 passed**, against §15's 2 / 433. 436 collected both
times. Nothing was deleted, skipped, xfailed, or re-thresholded. The one test edit in the arc
was a bug fix in a test the arc itself had added four steps earlier: the new cap test
constructed its arrival angle but **inherited `t0` from `best_solution.json`** — the exact fuse
its own docstring warns against, and `by_thickness` is linear in `t0`. Completing the
construction (`g[8] = 1.2`) made its guard the assertion it was meant to be.

Of the six red, one is the gene-box casualty above and **five are characterisation tests
genuinely invalidated by shipping a materially different wheel.** They pin findings about the
design space and are written to fail loudly when the premise moves:

| test | `350f4c7` | `e4219f3` | gate |
|---|---|---|---|
| `self_intersection_margin_detects_a_fold` | −7.064 | **+0.282** | < 0 |
| `correction_is_not_a_constant_over_the_design_space` | 3.383 | **1.542** | > 3.0 |
| `correction_enters_at_first_order_in_the_load` | 0.00205 | **0.00258** | < 0.001 |
| `rim_band_holds_a_large_minority_of_the_compliance` | 0.0321 | **0.0508** | < 0.03 |
| `a_thicker_rim_monotonically_stiffens_the_wheel` | 2.2496 | **1.637** | straddle 2.0 |

The last two were already red and both moved further out. Three causes worth knowing:

- The **fold detector is not broken.** It builds its positive case by inflating the *shipped*
  spine to a 40 mm band against an assumed ~11 mm curvature radius. The new spine is straighter
  — healthy margin 11.94 → 19.28, min curvature 12.26 mm — so 40 mm no longer folds it, by
  0.28 mm. It needs a thicker band or a fold constructed independently of the shipped genome.
- The **hub compliance share is the one real physical cost.** A 0.457 mm hub fillet is more
  compliant than a 0.579 mm one, so the hub's share of strain energy rose 0.0321 → 0.0508.
  **This is the price of a buildable radius and no iterate choice recovers it.**
- The **rim straddle says the new design is markedly more mesh-sensitive.** The two genomes
  agree at `medium` (1.99996 vs 1.99923 mm) and differ by **27%** at `smoke` (1.637 vs 2.2496
  at rim_outer 49.7), because every thickness gene sits on the 1.2 mm floor and a coarse mesh
  resolves a thin wall badly. **Do not trust a `smoke`-rung number on this genome.**

#### The successors, in priority order

1. **The slot branch's own arrival law.** It is what binds on the wheel that ships (wedge
   314.0°, NEAR-CUSP) and it is unfitted: 0.31 at 3.4° to 0.70 at 50°, ~2.2× unclaimed. The
   experiment is the mirror of `study_hub_cap.run_t0_sweep` — hold arrival fixed, walk the void.
2. **§15's successor 2, the stress-margin term**, now with a second argument behind it.
   `R_hub` still goes inert the moment it is feasible: no gradient from `fillet_cap` when
   satisfied, none from `stress` below its knee. Defects 1, 2 and 5 are one defect seen from
   three sides, and a term that prices margin instead of walling it fixes all three.
3. **The `--best-out` selection rule** (defect 6). Cheap, mechanical, and it removes a
   promotion hazard that this arc walked into twice.
4. **Re-derive the five characterisation gates** against a design whose walls are all on the
   floor. Real work and a judgement about what each gate should now say — not a threshold edit.

#### Artifacts

`stage3_buildcap2_medium.json` (the re-descent trace, 101 steps), `stage3_buildcap2_slack_medium.json`
(step 71, promoted) and `stage3_buildcap2_feasible_medium.json` (step 82, kept because defect 7
is only legible with both). `stage3_promote2_best.json` is the canonical `--steps 0` re-score
that became `best_solution.json`. The predecessor `350f4c7` is preserved byte-identical as
`stage3_minwall_best_1.2.json` and `stage3_minwall_best_1.2_medium.json`;
`best_solution_ga_beam.json` is untouched, so `tests/test_golden.py` is **not** re-baselined.
`export/wheel.step` and its manifest are rebuilt from `e4219f3`. Full step-by-step record in
`BUILD_PLAN.md`, steps 3 through 6c.

#### What this arc established, stated without the hedging

The objective has priced a buildable hub fillet since §5, and from 2026-08-06 to 2026-08-10 it
priced it with **a constant fitted an octave away from the floor every shipped design sits on**,
returning the same number for two designs OCC disagrees about. That is fixed, and the wheel that
ships is the first in this repo that the optimizer priced and the kernel built **at the same
radius**. The cost is honest and it is in the table above: a thinner, stiffer, more
mesh-sensitive wheel with 5.1% of its compliance in the hub, and five findings about the design
space that were measured on a design that no longer ships. The three new defects are all in the
penalty formulation, all found by gates rather than by reasoning — **two hypotheses about the
G4 failure were plausible, confidently held, and wrong before a measurement settled it** — and
the ordering lesson is procedural and cheap: **promote, export, then test**, and run the suite
after a gene-box change before anything else moves.

### 17. DEFECT 6 IS FIXED, AND THE SUCCESSOR RANKED #1 IN §16 IS WORTH NOTHING. Measured, not argued (2026-08-12).

Two things, one small and one that corrects §16's own conclusion.

#### `--best-out` no longer selects an infeasible genome

Defect 6 was that `wheel_stage3.py` reported the lowest-loss iterate of a run, and the loss
is a weighted sum in which the barriers are terms like any other. So the reported iterate is
whichever bought the most objective for the least constraint — and a barrier is not a thing
that can be bought. That is a category error, not a tuning failure, and no amount of descent
fixes it.

`wheel_objective.BARRIER_TERMS` / `OBJECTIVE_TERMS` now split the weight table by what a term
answers: *may this ship* or *how good is it*. The split is asserted complete against `TERMS`
at import, so a term added without a classification fails loudly rather than defaulting to
"can never make a design unshippable". `wheel_stage3.selection_key` ranks in three tiers —
feasible with slack, feasible on the knife edge, in violation — and the same key is used
within a run and across multi-start runs. Tier 2 is still ranked, so a run with nothing
feasible reports its least-violating iterate rather than nothing at all, and the banner says
which tier it is returning.

The band, `MIN_CAP_SLACK_MM = 1e-3`, is defect 7 made operational: **feasibility is
fidelity-dependent**, so a barrier reading exactly 0.0 certifies only that *this mesh* saw no
violation. Step 82 cleared the cap at `medium` by 53 nm and violated it at `coarse` by 93 nm.

Replayed against the real 101-step trace (`stage3_buildcap2_medium.json`), which is now a
regression test rather than a paragraph:

| iterate | loss | cap slack | old rule | new rule |
|---|---|---|---|---|
| step 93 | **30.8914** (the minimum) | **−1.045 µm** | **reported** | tier 2 |
| step 82 | 30.9008 | +0.052 µm | promoted off it | tier 1 |
| step 75 | 30.9406 | +10.936 µm | — | **tier 0, selected** |
| step 71 | 30.9421 | +3.096 µm | — | tier 0 (shipped) |

53 of 101 iterates are tier 0. The old rule reported a violating one out of that.

#### The new rule picks step 75, and step 75 should not ship — which is the finding

| | step 71 (shipped) | step 75 (rule's pick) |
|---|---|---|
| loss | 30.9421 | **30.9406** |
| mass | 37.5678 g | **37.5556 g** |
| deflection error | **−0.002%** | +0.110% |
| stress utilisation | **0.9964** | 0.9988 |
| R_hub | **0.4571 mm** | 0.4510 mm |

Step 75 is lower-loss and worse at every margin: it buys 12 mg with deflection error, stress
headroom, and 6 µm of hub fillet. `best_solution.json` stays at `e4219f3`; nothing was
re-promoted, re-exported, or re-scored.

This is the sharpest available argument for the stress-margin term. Fixing the selection rule
removed the barrier-versus-objective confusion and left the *objective's own* indifference to
margin fully exposed: among tier-0 iterates the rank is still loss, and loss prefers 12 mg to
every margin the design has. **Defects 1, 2 and 5 and this are one defect seen from four
sides.**

#### §16 ranked the slot arrival law #1. It cannot pay, and here is the measurement

§16's argument was that the slot branch "is what binds on the wheel that ships (wedge 314.0°,
NEAR-CUSP)". That conflated two different things, and they disagree on this wheel:

- the **wedge family of the worst OCC corner** — near-cusp, 314.0°, true; and
- the **analytic branch that sets the cap** — thickness, 0.4601 mm against the slot branch's
  1.6221 mm, **253% away from binding**.

The corner OCC finds hardest and the branch the optimizer feels are not the same object. Even
granting the whole re-fit — the measured near-cusp share at the shipped arrival is 0.60
against the modelled 0.30 — the slot branch would move to 3.2476 mm, **7× above the binding
branch**. It changes the cap by exactly zero, on this wheel and on any wheel near it.

Meanwhile the branch that *does* bind is already tight. Against the square-on family at
`t0` = 1.2, the floor the shipped wheel sits on:

| arrival | OCC / t0 | model / t0 | model is |
|---|---|---|---|
| 5.14° | 0.5105 | 0.5031 | 1.5% conservative |
| 20.10° | 0.4872 | 0.4758 | 2.4% conservative |
| 30.06° | 0.4524 | 0.4404 | 2.7% conservative |
| 40.03° | 0.4059 | 0.3925 | 3.4% conservative |
| 50.00° | 0.3439 | 0.3336 | 3.1% conservative |
| 59.96° | 0.2722 | 0.2653 | 2.6% conservative |

Conservative everywhere, never over-promising, and never by more than 3.4%. At the shipped
arrival of 41.748° it is 3.2% conservative. **There is no fillet radius left on the table on
the branch that binds** — which is what §16 set out to achieve and is worth stating as an
achieved result rather than leaving implied.

One more reason the §16 successor was not ready to run, and it is a datum already in the tree
being read for a second purpose. BUILD_PLAN.md step 4 records `350f4c7_t0_1.2`'s near-cusp
threshold as `≥1.44` at all seven arrival stations, censored at the bisection bracket, and
dismisses it correctly — "never binding in the measured range, which is all this arc needs
from it." That is true for computing a **cap**, where a censored non-binding branch costs
nothing. It is fatal for **fitting the branch itself**, which is what the successor proposes:
the share then reads 0.323 → 0.264 across arrival, and that decline is a constant numerator
over a growing arc, an artifact of the ceiling rather than a law. So the slot law rests on
`elite13_t0_2.55` alone, n = 1, and fitting on one uncensored design is the same mistake §16
exists to correct. The sweep would have to be re-run with a raised bracket first.

#### The successors, re-ranked by what they are now measured to be worth

1. **The stress-margin term** (§15's successor 2). Promoted to #1 on the evidence above:
   `R_hub` goes inert the moment it is feasible, and the step-75-versus-71 comparison shows
   the objective will spend every margin the design has for 12 mg. This is the one with a
   payoff.
2. **Re-derive the five characterisation gates** against a design whose walls are all on the
   floor. Unchanged in rank, still real work and still a judgement about what each gate should
   say, not a threshold edit.
3. **The slot branch's arrival law.** Demoted from #1 to #3 and re-scoped: it is a
   correctness fix to a branch that is 253% from binding, not a source of fillet radius. If it
   is done, it needs the arrival sweep re-run with a raised bisection ceiling first, because
   the existing data is censored on one of its two designs.

### 18. THE OBJECTIVE CAN NOW SEE STRESS MARGIN. Two dead genes are alive and the wall came off the floor. **NOTHING PROMOTED** (2026-08-12).

§15's defect 1, named on 2026-08-10 and left alone twice since because acting on it mid-arc
would have been re-fitting a gate to the run that breached it. §17 gave it a second argument
that was not available in §15 — with selection fixed, the objective's *own* indifference to
margin was the only thing left explaining step 75 — so it is now the ranked successor and this
is it.

#### The defect, re-measured before it was touched

`stress` is `soft_barrier(util - 1)`, identically zero **and identically flat** for every
`util <= 1`. Below the knee the optimizer cannot see stress at all: it sees mass, and it thins
the wall. The consequence is not bad weights, it is **dead genes** — the only routes from a
fillet radius into the loss are `stress` and the fillet barriers, all flat unless breached.

Measured at the shipped genome on 2026-08-12, before any change: **`dL/dR_hub` and `dL/dR_rim`
are both exactly `+0.000000e+00`.** Not small. Zero. A nominally 14-dimensional search was
running in 8, which is what §15 measured a different way (over 602 steps `R_rim` moved on 0 and
`R_hub` on 2, both times only because `fillet_cap` was live).

#### The term, and the weight as a stated policy

`stress_margin = w * util^2`, summed over the same two junctions as the barrier and for the
same reason — a `max` would zero the gradient of whichever junction is not currently worst. It
is in `OBJECTIVE_TERMS`, never `BARRIER_TERMS`: `stress` is untouched, the wall still decides
shippability, and this only stops the approach to it being free.

The weight is an exchange rate, so it is derived rather than picked. The mass term is 30.88 at
37.57 g, so 1% of mass is 0.309 of loss; 1% of hub utilisation at `util` = 0.855 costs
`w * (1.01^2 - 1) * 0.855^2` = 0.0147 `w`. Indifference is `w` = 21.0. Shipped at **20.0** —
rounded *down*, toward buying less margin, which is the conservative direction for a term whose
purpose is to move the optimum. Quadratic rather than linear is the second half of the policy:
the exchange rate steepens as margin disappears, so the last 10% of utilisation costs far more
than the first. At the shipped genome the term lands at 20.23, 13.6% of the loss against mass's
20.7%.

The hand-written product rule was FD-checked, and this was not a formality: `dkt_hub`'s `R_hub`
path was previously multiplied by `max(0, util - 1)` = 0 at every feasible design, so an error
in it could not have been observed.

| gene | analytic | central FD | rel err |
|---|---|---|---|
| 12 `R_hub` | −1.234936e+01 | −1.234936e+01 | 1.79e-07 |
| 13 `R_rim` | −7.237940e-01 | −7.237940e-01 | 1.10e-09 |
| 8 `t0` | +3.336761e+02 | +3.336762e+02 | 3.67e-07 |

#### What a 40-step SVK probe did with it

`stage3_margin_probe.json`, `coarse`, 40 steps, uniform 8-phase, SVK, from the shipped genome.
1 h 57 m, 7.4 GB peak, clean exit, converged (last three losses 51.7487 / 51.7478 / 51.7474).

| | step 0 | step 40 |
|---|---|---|
| loss | 57.0300 | 51.7474 |
| `R_hub` | 0.45711 | **0.62435** (cap 0.62700) |
| `R_rim` | 2.74947 | **3.00000 — box maximum** |
| `t0` | 1.20084 | **1.36183 — off the 1.2 floor** |
| utilisation | 0.95921 | 0.80982 |
| axle drop | 1.96822 mm (−1.6%) | 1.99780 mm (−0.11%) |
| mass | 37.568 g | 39.410 g (**+4.9%**) |

Four things, one of them unplanned.

**`R_hub` converged to 0.4% under its own cap.** It rose until the geometry stopped it and
settled just below, `fillet_cap` exactly 0.0 for the last dozen steps. The design now asks for
the largest fillet it can actually build — which is what §16's cap was for and what no descent
before this could do.

**The cap itself rose, 0.46006 to 0.62700.** Nothing pushes the cap; it is a function of `t0`
and the hub arrival. The optimizer reshaped the hub to make room for a fillet that was, for the
first time, worth having. §16's cap model and this term composed into a behaviour neither was
designed to produce.

**Two of the four wall genes came off the floor** — `t0` to 1.362 and `t3` to 1.326. §15's
defect 4 said the descent "ran out of wall to thin before it ran out of stress margin, and set
the floor lower and the same blind gradient would keep going." That is now false for `t0` and
`t3`: with margin priced, the design chooses to be thicker. `t1` and `t2` are still pinned.

**`R_rim` hit its box ceiling of 3.0 and stayed there for 26 steps.** That ceiling was set while
`R_rim` was a dead gene, so no descent has ever tested it. It is now the binding constraint on
the rim fillet, and it is an untested number rather than a physical limit.

#### Defect 5 did not bite, and an intermediate reading of this run said it would

Mid-descent `fillet_cap` oscillated — 0.0, 0.44, 0.0, 0.24 — and at step 11 `R_hub` sat 0.022 mm
**above** its cap. Read at step 11 that is a soft-barrier equilibrium, defect 5 becoming
operational the moment the barrier acquired live opposition, and it was recorded here as such
before the run finished. **The full run says otherwise.** Those excursions are transients from
the cap moving faster than `R_hub` could track it; once the cap stabilised, `R_hub` settled
under it and stayed. 26 of 41 iterates are feasible and the converged point is one of them.
Defect 5 remains real and remains unfixed — it just is not what limits this term.

#### One measure-zero bug, found because the term made it reachable

`smooth_min`'s derivative at exactly `a == b` was **1.0 against a true two-sided 0.5**. Two
primitives are non-differentiable at the tie and autodiff picks a subgradient for each: `jnp.abs`
returns 0 at 0, which drops the blend term entirely, and `jnp.minimum` hands the full 1.0 to its
first argument. The value was never wrong, which is why the existing value-only exactness test
could not see it. Fixed by writing the min symmetrically as `(a + b - |a - b|)/2`, fenced inside
the blend by a `where` because that form is not bit-exact and exactness outside the blend is the
property the function exists for. Never observed to bite — recorded and fixed because this term
drives `R_hub` at its cap deliberately, which turns the tie from an accident into an attractor.

#### The gate

`6 failed, 438 passed` — **the same six reds as §17, no new ones**, and +4 from this section's
own tests. That is worth stating plainly because it was not the expected outcome: changing a
term in the objective changes the loss at every design, and a suite with loss numbers pinned in
it would have gone red in bulk. It did not, which says the gates are pinned to physics and
provenance rather than to the objective's arithmetic. The six are §16's, unchanged and
unrelated: five characterisation gates invalidated by the shipped wheel's geometry and one
gene-box casualty.

#### NOTHING IS PROMOTED, and the reason is not caution

`stage3_margin_probe_best.json` (`3ca40c1`) is a 40-step **coarse** probe under a **changed
objective**. Its loss is not comparable to any number in §16 or §17 — the same discontinuity
§14 recorded for linear-versus-SVK, for the same reason. And it costs **+4.9% mass**, which is
a real design change that deserves a medium-fidelity descent and an export check rather than an
assertion. `best_solution.json` is untouched at `e4219f3`.

#### The successors, re-ranked again

1. **A production descent under the new objective** — `medium`, SVK, from the shipped genome,
   with an export check. This is what turns §18 from a demonstration into a candidate, and the
   +4.9% mass is the thing it has to justify.
2. **`R_rim`'s box ceiling.** Newly binding, never tested, and cheap: the question is whether
   3.0 mm is a real limit or a number typed in when the gene was dead.
3. **Re-derive the five characterisation gates.** Unchanged, and now with more to re-derive.
4. **Defect 5**, the quadratic barrier that cannot hold a boundary against live opposition. Not
   what limited this run, but the opposition is new and it will be back.
5. **The slot branch's arrival law.** Still 253% from binding; still correctness, not payoff.

---

### 19. THE PRODUCTION DESCENT UNDER THE NEW OBJECTIVE. **PROMOTED — `e4219f3` → `e126cc3`** — and defect 5 bit, exactly where §18 said it would not (2026-08-13).

§18's ranked successor #1, run as specified: `medium`, SVK, 100 steps, uniform 8-phase, seed 0,
4 workers, `--fidelity-check-every 25 --fidelity-check-config coarse`, from the shipped genome,
`--min-wall 1.2`. Every knob identical to `stage3_svk_medium` and `stage3_buildcap2_medium`, so
the only difference between this run and those is the objective. 6 h 20 m (22 834 s), 101
objective calls, exit 0. `stage3_margin_medium.json`.

Its job was to justify or refute §18's +4.9% mass. **It refuted the number and confirmed the
trade**: at `medium` the mass goes up more, and the case for paying it gets stronger, not weaker.

#### Shipped against the candidate, same objective, same mesh, same kinematics

Step 0 **is** the shipped genome `e4219f3` evaluated under the new objective, so this table is
apples to apples in a way no comparison in §14–§18 could be.

| | step 0 = `e4219f3` | step 55 = `e126cc3` | |
|---|---|---|---|
| loss | 58.4115 | **51.3892** | −12.0% |
| `stress_margin` term | 27.4694 | 17.7156 | −35.5% |
| `mass` term | 30.8776 | 33.2954 | +7.8% |
| hub utilisation | **0.99639** | **0.77952** | −21.8% |
| rim utilisation | 0.61700 | 0.52738 | |
| spoke mesh mass | 37.568 g | 40.509 g | **+7.8%** |
| solid mass @PLA (OCC) | 45.34 g | 50.25 g | **+10.8%** |
| axle drop | 1.99996 mm (−0.002%) | 2.00806 mm (**+0.40%**) | |
| `R_hub` | 0.45711 (cap 0.46021) | 0.72795 (cap 0.73523) | +59% |
| `R_rim` | 2.74947 | **3.00000 — box maximum** | |
| `t0` / `t3` | 1.20084 / 1.21900 | **1.55944 / 1.46300** | off the floor |
| `t1` / `t2` | 1.2 / 1.2 | 1.20473 / 1.25241 | still pinned |
| `Kt` hub | 2.22145 | 2.06875 | |
| min scaled Jacobian | 0.65980 | 0.82664 | mesh quality up |

The mechanism is the one §18 predicted and is worth naming once more: the optimizer does not
merely enlarge the fillet, it **reshapes the hub so the cap will allow a larger one**. The cap
rose 0.46021 → 0.76697 monotonically over the run. Nothing pushes on the cap directly; it is a
function of `t0` and the hub arrival, and `t0` grew 30% because thicker wall now buys buildable
fillet, which buys `Kt`, which buys utilisation.

#### Defect 5 bit. §18 said it would not, and §18 was reading a run too short and too coarse to show it

**74 of 101 iterates are tier 2 — in violation. Every single step from 56 to 100 is one of
them**, each sitting 1.4 to 11.0 µm *over* its hub fillet cap, and the converged point at step
100 is 4.1 µm over. The run's own trajectory kept improving loss (51.389 → 51.077, −0.6%) while
being unshippable the entire way.

That is precisely defect 5: `soft_barrier(v) = scale * max(0, v)**2` has zero gradient at its
own knee, so against a term that pushes steadily outward it cannot hold the boundary — it
settles at whatever overshoot makes the quadratic's slope match the opposing pull. The
opposition did not exist until §18 created it. §18's "defect 5 did not bite" was a true
statement about a 40-step `coarse` probe and a false forecast about everything after it:

| | §18 probe (coarse, 40) | §19 (medium, 100) |
|---|---|---|
| converged `R_hub` vs cap | 0.4% **under** | 4.1 µm **over** |
| `fillet_cap` at convergence | 0.0 | 8.45e-03 |
| feasible iterates | 26 / 41 | 27 / 101 |
| last feasible iterate | step 40 (the last) | **step 55 of 100** |

**The defect-6 fix is the only reason this run produced a shippable genome at all.** The old
`--best-out` rule reported the lowest-loss iterate; that is step 100, which is over its cap.
`selection_key` returned step 55 — the last tier-0 iterate — and the banner said so. Second run
in a row where the rule caught a real over-cap promotion, and the first where the whole
converged tail was bad rather than a single iterate.

45 steps of a 6 h 20 m run were spent descending into the unbuildable. That is the cost of
defect 5 stated as wall-clock.

#### Fidelity: the candidate clears at both meshes, and the cap barely moves between them

The §16 trap — feasible at one mesh, breached at another, 93 nm on the wrong side — was checked
directly rather than assumed. `--steps 0` re-evaluations of `e126cc3`:

| | cap | `R_hub` | slack | `fillet_cap` | util | drop |
|---|---|---|---|---|---|---|
| `medium` | 0.735233 | 0.727953 | **+7.279 µm** | 0.000e+00 | 0.7795 | 2.00806 |
| `coarse` | 0.735015 | 0.727953 | **+7.062 µm** | 0.000e+00 | 0.7707 | 1.99079 |

**The cap differs by 0.22 µm across a mesh change — 33× less than the slack.** The five
scheduled fidelity checks agree the same way along the whole trajectory (`failed: false` at all
five; at step 100, coarse cap 0.76674 against medium 0.76697, utilisation 0.7612 against
0.7696). The new optimum is mesh-stable, which is the first positive evidence anywhere in this
tree on defect 7. The `medium` loss reproduced to 51.3892 from the genome file alone.

#### The export builds it exactly as modelled

`make export EXPORT_GENOME=stage3_margin_best_medium.json`, then again as `wheel.step` after
promotion:

| | requested | built | edges | worst wedge | `Kt` error |
|---|---|---|---|---|---|
| hub | 0.7280 | **0.7280** | 24/24 | 326.0° | **+0.0%** |
| rim | 3.0000 | **3.0000** | 24/24 | 326.0° | **+0.0%** |

OCC valid, single solid, 100.00 × 100.00 × 22.40 mm, BRepCheck valid, no self-intersections,
degenerate edges 0, min curvature radius 0.7279 mm against a 0.25 floor, junction bite
0.5126 / 1.9116 mm against a 0.25 floor — both pass. The worst hub wedge went 314.0° → 326.0°,
deeper into NEAR-CUSP territory, **and it still built at the full radius**, which is the cap
model from §16 doing its job at a corner harsher than the one it was fitted against.

#### PROMOTED, and the price is a rate I chose

`best_solution.json` is now `e126cc3`; `export/wheel.step` and its manifest rebuilt from it. The
predecessor is preserved. The call, stated so it can be argued with:

**For.** It beats the shipped genome by 12.0% on the objective the project now uses, at the same
mesh, kinematics and stencil, with step 0 of the same run as the control. It is tier 0 at both
fidelities. It builds with zero `Kt` error. And the condition it fixes is not cosmetic: **the
shipped wheel sits at 0.4% of its allowable stress.** For a printed PLA part, where layer
adhesion, print orientation and batch scatter are all ±10–20% effects, 99.6% utilisation is not
a design with thin margin, it is a design with none. 22 points of headroom for 4.9 g is a trade
I would make on any printed part.

**Against, and recorded rather than argued away.** The full-solid mass goes 45.34 → 50.25 g,
**+10.8%** — a bigger number than the +7.8% the optimizer's own mesh mass shows, because the
mesh mass covers only the spoke region. The axle drop moves from −0.002% to **+0.40%** off the
2.0 mm target: 8 µm, physically irrelevant on this part, but the direction is real and the
objective bought margin partly by detuning the deflection target. And the +10.8% is the price of
an exchange rate **I set** in §18 — 1% mass against 1% utilisation, `w` = 20.0. The optimizer did
not discover that mass is worth trading; it was told, and it obeyed to the point where the rate
balanced.

#### DEFECT 8: `util**2` does not taper, so the term keeps buying margin past where it is worth anything

The quadratic was chosen in §18 so "the exchange rate steepens as margin disappears". It does —
but far too weakly to encode the policy actually intended. Marginal price is `2*w*util`, so
going from `util` 0.9964 to 0.7795 drops the price of another point of margin by only **28%**
(39.9 → 31.2 per unit). The real preference is nothing like that: margin below roughly 0.8 is
close to worthless on this part, and margin above 0.95 is close to priceless. A quadratic cannot
express a knee. This did not produce a bad design here — 0.78 is a defensible place to stop —
but the design stopped there because mass finally caught up, **not because the term stopped
wanting margin**, and that is the wrong reason. Named, not fixed.

#### The gate: `11 failed, 433 passed` — up from 6, and the six new ones split three ways

The promotion moved the geometry every characterisation gate in this tree was written against,
so the diff is the interesting object, not the total. **One went green** —
`test_self_intersection_margin_detects_a_fold`, red since §16. **Five are §16's, unchanged.**
**Six are new**, and they are not one thing. Each was re-measured on both genomes before being
characterised, because §17's lesson was that the plausible cause is often not the cause.

**Group A — the contact patch halved, and three gates are downstream of that one fact.** `t3`
went 1.219 → 1.463 and `R_rim` to 3.0, so the rim wall is 20% thicker and conforms less:

| measured on the `smoke` mesh | `e4219f3` | `e126cc3` |
|---|---|---|
| real patch half-angle | 0.4963° | **0.2965°** (Hertz: 0.3082°) |
| patch / Hertz | 1.610 | **0.962** |
| worst rim-node gap | +1.115 µm | **−0.247 µm** (penetrates) |
| assumed-patch drop | 1.39796 mm | 1.62013 mm |
| real-contact drop | 1.35494 mm | 1.53472 mm |
| **assumed vs real** | **3.08%** | **5.27%** |

> **THE PARAGRAPH BELOW IS WRONG AND §20 RETRACTS IT. THE NUMBERS IN THE TABLE ABOVE ARE NOT
> — they all reproduce to the digit.** The objective does **not** compute deflection against an
> assumed 3.0° patch: its whole path runs `fem.solve_wheel_contact`, and `patch_half_deg`
> cannot even be passed to it (`TypeError`). So 2.008 mm **is** the real-contact number, the
> 5.27% is the *legacy* model's error, and what this red gate indicts is M4, M5,
> `studies/study_gnl.py` and `studies/study_wheel_fea.py` — not the wheel that ships. The
> objective's own quantity is mesh-convergent at **0.897%** coarse→medium, and `e126cc3`
> converges **2.2–2.5× better** than the `e4219f3` it replaced. Read §20 before quoting any
> sentence in this subsection.

The third row is the one that matters and it is **a red gate on the design that now ships**, not
on retired geometry. The objective computes deflection against an *assumed* 3.0° patch. On the
predecessor that assumption cost 3.08% of axle drop; on the promoted wheel it costs 5.27%, over
`test_but_the_assumed_patch_got_the_axle_drop_nearly_right`'s pre-registered 5% bound. **The
promoted wheel's true deflection is therefore near 1.90 mm against a 2.0 mm target, not the
2.008 mm the objective reports** — and the error grows as the optimizer moves, because nothing
in the loss knows the patch model is drifting out from under it.

This does not reverse the promotion. Utilisation is computed under the same assumption for both
designs, the binding junction is the hub (0.7795) rather than the rim (0.5274), and hub stress
is spoke-bending dominated and far less sensitive to patch width than rim stress is. The
margin result survives. The deflection number does not, and is corrected here rather than in a
footnote.

**Group B — one gate is defect 8, measured.** `test_the_margin_weight_is_the_exchange_rate_it_
claims_to_be`, written yesterday in §18, asserts the weight still buys 1% of utilisation for 1%
of mass at the shipped genome, within [0.5, 2.0]. At `e126cc3` the ratio is **0.379**: 1% of
utilisation costs 0.1261 against 1% of mass at 0.3330. The stated calibration no longer
describes the design that ships. That is not a broken test — it is the test doing its job,
reporting that a policy written at `e4219f3` did not survive being optimised against, which is
exactly defect 8 in one number.

**Group C — two latent test defects that the promotion exposed rather than caused.** Both were
verified as pre-existing, and both passed on `e4219f3` by luck rather than by margin:

- `test_the_bite_is_the_volume_divided_by_the_right_thickness` recomputes the manifest's bite
  from the manifest's own numbers to `abs=1e-4`, but the manifest stores volume at 2 dp and
  bite at 4 dp. The composite rounding band is **±1.4e-4 — wider than the tolerance**. On the
  hub the error was 7.66e-5 at `e4219f3` (pass) and is 1.25e-4 at `e126cc3` (fail), with a
  rounding band of ±1.55e-4 at the *old* genome, i.e. already unsatisfiable in the worst case.
  The failure this test exists for — `t0` and `t3` crossed between the two rings — passes on
  both genomes.
- `test_only_the_rim_od_near_the_bottom_is_loaded` allows loaded nodes "one element" beyond the
  patch edge and computes that element as `SECTOR_DEG / (n_weld + n_rim_free)` = **1.5°**. The
  rim-OD elements are quadratic: the one straddling the patch edge runs
  271.6824° / 273.0983° / 274.5142°, spanning **2.8318°, two node pitches**. Its far node sits
  4.5142° from the bottom against a 4.5° bound and carries **1.11% of the total load** — real
  spill, not numerical dust. The bound understates a real element by 1.9×. At `e4219f3` the
  patch edge fell elsewhere on the node grid and the worst node was 3.864°.

**Neither Group C test is fixed here.** Both diagnoses are solid and both fixes are small, but
loosening a tolerance in the same session as the promotion that reddened it is the pattern §5
and §16 exist to warn about. They stay red, with the cause recorded, and the fixes are ranked
below.

#### The successors, re-ranked again — a red gate on the shipped design goes first

1. ~~**The contact patch model.** Group A. This is the only red gate in the tree that indicts
   the wheel currently in `best_solution.json`, it biases the quantity the objective is
   steering by 5.27%, and it gets worse the harder the optimizer works. Everything below is
   cheaper and less important.~~ **RETIRED BY §20 (2026-08-13) — all three clauses refuted.**
   It indicts no wheel (the assumed patch cannot reach the objective), it biases nothing (the
   objective's own quantity converges to 0.897%), and it gets *better*, not worse, the harder
   the optimizer works (2.2–2.5× better than the predecessor). **Defect 5 is now #1.**
2. **Defect 5.** Cost 45 steps of a 6 h 20 m run and stands between this project and any
   *converged* buildable design.
3. **`R_rim`'s box ceiling of 3.0.** Pinned throughout and at convergence, still never tested,
   still cheap, and now binding on the promoted wheel.
4. **Defect 8**, and with it the §18 rate gate that is now red. A margin price with an actual
   knee, re-derived at the genome that ships.
5. **The two Group C test defects.** Diagnosed, unfixed, deliberately not bundled with the run
   that exposed them.
6. **Re-derive the five characterisation gates**, and **the slot arrival law** — unchanged.

### 20. THE CONTACT PATCH MODEL: §19's SUCCESSOR #1 IS RETIRED, NOT DEFERRED. The objective never touched the assumed patch (2026-08-13).

Working notes in `CONTACT_PLAN.md`. §19 ranked the contact patch model first, on the sentence
*"The objective computes deflection against an assumed 3.0° patch."* **It does not, and it
cannot** — so the harm that ranking claimed does not exist. Nothing promoted; nothing about
`e126cc3` changes; four red gates fixed and the gate closes at **11 → 7 failed, 433 → 438 passed**.

#### The premise, falsified by refusal rather than by comparison

Stage 3's deflection is `axle_drop` from `wheel_adjoint.service_qoi_value_and_grad`, and every
solve on that path is `fem.solve_wheel_contact` / `wheel_contact_problem` — **real penalty
contact**. `CONTACT_PATCH_HALF_DEG = 3.0` is consumed by `wheel_problem` / `solve_wheel` (the
M4 pressure model) and by nothing else in `src/`. Measured at `e126cc3`, `coarse`:

| | linear | svk |
|---|---|---|
| objective `axle_drop_mean_mm` | 1.5892237160122678 | 1.9132197613988573 |
| `solve_wheel_contact` | 1.5892237160122678 | 1.9132197613988573 |
| **bit-identical** | **True** | **True** |
| `solve_wheel` (assumed 3.0°) | 1.7025307168359758 | 2.0519394047655464 |
| objective vs assumed | 6.66% | 6.76% |

**The sentinel is the result.** Threading `patch_half_deg` into the objective raises
`TypeError: wheel_contact_problem() got an unexpected keyword argument 'patch_half_deg'`. The
constant is not merely uninfluential — it is **structurally unreachable** from the loss, which
is a stronger statement than the bit-identity the check was written to accept.

#### What is withdrawn from §19, and what is not

**Withdrawn.** *"The promoted wheel's true deflection is therefore near 1.90 mm against a
2.0 mm target, not the 2.008 mm the objective reports."* The comparison is the other way round:
**2.008 mm IS the real-contact number.** And *"it gets worse the harder the optimizer works"* —
measured against the control, `e126cc3` converges **2.2–2.5× better** than the `e4219f3` it
replaced, and its rim-OD element-size step is 16.8× against 30.3×.

**Not withdrawn: §19's measurements, all of which reproduce to the digit** — patch 0.2965°,
patch/Hertz 0.962, node gap −0.247 µm on `e126cc3`; 1.610 and +1.115 µm on `e4219f3`. What
moves is which model they indict: **M4, M5, `studies/study_gnl.py` and
`studies/study_wheel_fea.py`**, every one of which still solves `fem.solve_wheel`. The
divergence is real and refinement-stable (5.27% → 6.66% → 6.25% up the ladder) and nearly
doubled across one promotion. It says the assumed patch has stopped standing in for contact on
this geometry. It says nothing about the wheel that ships.

**Also not withdrawn: the promotion.** Every argument §19 made *for* `e126cc3` — 12.0% on the
objective against step 0 of its own run, tier 0 at both fidelities, 24/24 at both junctions at
+0.0% `Kt` error, utilisation 0.9964 → 0.7795 — is untouched here.

#### The objective's own quantity IS mesh-convergent, and that was the question worth asking

`make contact`, four cells (both genomes × both kinematics), `patch` section, smoke/coarse/
medium, forward solves only. Pre-registered gate on `e126cc3` under **svk**:

```
coarse -> medium relative change   0.897%   [< GATE_MESH_REL = 0.05]   PASS, 5.6x
n_quad_points_in_contact @ medium  3        [>= 2]                     PASS
worst rim-node gap @ coarse/med    +2.148e-04 / +1.567e-04 mm  [> 0]   PASS
```

| genome | kin | coarse→medium | Richardson limit | finest off it |
|---|---|---|---|---|
| **`e126cc3`** | **svk** | **+0.897%** | 1.936326 mm | **0.307%** |
| `e4219f3` | svk | +1.669% | 1.901637 mm | 0.670% |
| `e126cc3` | lin | +0.905% | 1.608769 mm | **0.321%** |
| `e4219f3` | lin | +1.609% | 1.456732 mm | 0.814% |

#### THE FINDING: a 16.8–30.3× element-size step on the rim OD, and the patch sits on it

The rim OD is not uniformly divided. Per 30° sector it gets `n_weld` segments over the weld
and `n_rim_free` over the free arc, and they are **not the same size** — at `coarse` on the
shipped genome, `10 × 0.1682° + 10 × 2.8318° = 30.000°` exactly. The ratio is **constant up
the mesh ladder and a function of the design**: 16.8 on `e126cc3`, 30.3 on `e4219f3`, tracking
the weld footprint. **Refining buys a smaller element and the same step.**

At `medium` the weld/free boundary sits at +1.6824° and the patch centre at +1.885° to
+2.082°. The patch is **smaller than the element containing it in all twelve cells**
(`patch/seg` 0.39–0.46 at `medium`), so which nodes it sees is decided by whether its edge
reaches back across the step:

| kin | patch interval | reaches back past 1.6824°? | nodes | quad |
|---|---|---|---|---|
| linear | [1.5429, 2.3623] | **yes, by 0.1395°** | **3** | 11 |
| svk | [1.7210, 2.4430] | no, misses by 0.0386° | **0** | 3 |

Same mesh, same genome, same load — a 3.7× difference in how well the patch is sampled,
decided by 0.04° of where an edge fell. That is `wheel_objective.py:107-114`'s aliasing given
a mechanism instead of a symptom, and it explains all three Group A reds with nothing left
over. It also retires `deg_per_segment` as a resolution measure: `360 / len(rim_outer)` is the
mean of a bimodal distribution, 0.938° at `medium` against a local truth of 0.105° or 1.770°.

#### Four reds fixed, four different kinds of wrong

| red | what it actually indicts | change |
|---|---|---|
| `test_the_centre_node_rise_is_not_the_axle_drop` | a **premise**; its own message blamed refinement, which is the wrong direction | premise recorded, bound moved to penetration depth against the rim band (3.6e-4, 3× inside `GATE_PENETRATION_FRAC`) |
| `test_the_real_patch_is_far_smaller_than_the_assumed_one` | a **`smoke`-tier artefact** — 0.962 in one cell of twelve, 1.082–1.720 in the rest | lower-bound clause scoped to `coarse`, on §4's and `test_the_sampled_patch_extent`'s precedent |
| `test_but_the_assumed_patch_got_the_axle_drop_nearly_right` | the **legacy assumed-patch model**; real and refinement-stable | renamed `test_the_assumed_patch_no_longer_stands_in_for_contact`, banded two-sided at 2–8% |
| `test_only_the_rim_od_near_the_bottom_is_loaded` | a **constant standing in for a mesh fact** | bound read off the mesh's widest rim-OD element |

**§19's diagnosis of that last one was wrong in its cause** and right in its ratio. It blamed
quadratic elements spanning two node pitches and proposed a factor of 2. The element *order*
has nothing to do with it: `SECTOR_DEG / (n_weld + n_rim_free)` = 1.5° is the mean of two sizes
16.8× apart, and the element straddling the patch edge is the free-arc one at **2.8318°, which
is 1.888×** the bound — §19's 1.9×, reached by a different route. A factor of 2 would have been
a second constant standing in for a number the mesh already knows.

**`make test` closes at 7 failed / 438 passed** — exactly 11 − 4 reds with no new red, and one
test added (the containment pin below). The seven left are the two §19 deliberately parked (the
manifest-bite tolerance, the §18 rate gate) and the five inherited.

#### Method notes, because two of them caught live errors

- **`studies/study_contact.py` gained `--kinematics` and `--sections`**, plus a `make contact`
  target and the resolution columns above. Default `linear`, and the inertness check is a
  clean in-session control rather than a stale artifact: the pre-edit driver and the edited one
  both run `--quick` with no flags, **243 non-timing leaves compared, 0 differ**, the only new
  leaves being this arc's own columns.
- **Two errors made and corrected mid-arc, recorded because the reasoning was wrong twice.**
  A `NameError` in the full-report printer surfaced only *after* 129 s of solving — the same
  shape as SVK_PLAN Step 2's `_record` bug — and a new column picked the rim segment with the
  nearest *centre* rather than the one *containing* the patch, which inverted the mechanism at
  `medium` until the node counts refused to add up.

#### The successors, re-ranked

1. ~~**Defect 5** — the `max(0,v)**2` barrier's dead knee. Now #1, and the only item here with a
   measured price: **45 of 100 steps** of a 6 h 20 m run spent descending into the unbuildable.~~
   **DEMOTED BY §21 (2026-08-13): the tail is NOT unbuildable.** Step 100 exports 24/24 at
   +0.0% `Kt` error; the overshoot is 4–7× inside the cap's deliberate conservatism, and the
   discarded design differs by under 1% on every metric. `R_rim`'s ceiling is now #1.
2. **`R_rim`'s box ceiling of 3.0** — pinned throughout §19's descent and at convergence.
3. **Defect 8** and §18's rate gate, still red at 0.379 against [0.5, 2.0].
4. **The rim OD element-size step** — new. Ranked here and NOT first, on §17's rule: its cost
   is unmeasured, and the adjacent measurement (M7's facet ratio, 3.8% at `coarse` and
   refining away) says small. The first piece of work is the cheap measurement — **G7
   phase-smoothness at `medium` on `e126cc3` under SVK** — not a mesh change. This arc
   measured the *value*; a facet artefact would show in the *gradient* first.
5. **The Group C manifest-bite tolerance**, diagnosed in §19, out of scope here.
6. **Re-derive the five inherited characterisation gates.**

### 21. DEFECT 5 IS REAL, ITS LAW IS NOW EXACT, AND IT DOES NOT JUSTIFY ITS FIX. The "unbuildable" converged iterate BUILDS (2026-08-13).

Working notes in `DEFECT5_PLAN.md`. §20 ranked defect 5 first — the `max(0,v)**2` barrier that
cannot hold a boundary — on the strength of §19's *"45 steps of a 6 h 20 m run descending into
the unbuildable."* **The tail is not unbuildable.** Two steps of measurement, no optimizer, no
descent: nothing promoted, no barrier changed, `best_solution.json` untouched.

#### The equilibrium law, and the half of it that is exact

```
    d(soft_barrier)/dv = 2 * w * max(0, v)        ->  zero at the knee
    fixed point         v* = P / (2 * w)          against an outward pull P
```

Measured at step 100 of `stage3_margin_medium.json`, `medium`, SVK, terms isolated by zeroing
every other weight:

| | measured | predicted |
|---|---|---|
| `dL_fillet_cap/dR_hub` | **+4.112098** | `2·w·v` = **+4.1121** |
| `dL_stress_margin/dR_hub` | −5.162679 | — |
| sum | **−1.050581** | 0 if stationary |

**The barrier's derivative is exact to six significant figures.** What is refuted is the
registration, not the law: **step 100 is not a stationary point** — 1.05 of net outward
gradient remained when the budget ended. The fixed point is `P/(2w)` = **5.163 µm**, and the
trajectory tail (steps ≥ 80: 3.843–5.525 µm, mean 4.500) brackets it. §19's "converged 4.1 µm
over" is a snapshot of a trajectory heading for 5.2 µm.

**Sweeping all fourteen terms found exactly two with a nonzero `dL/dR_hub`** — `stress` is
identically 0.0 because utilisation is under 1.0 and §15's defect 1 makes that barrier flat
there. A two-body problem with no hidden third force.

**Defect 5 and defect 8 are one problem.** The outward pull *is* §18's `stress_margin` term,
and defect 8 is exactly that `util**2` never stops wanting margin. `P` therefore does not
decay as the design improves, so a barrier with restoring force `2·w·v` must sit permanently
outside its own boundary. §20 ranked these #1 and #3 as separate items.

#### THE MEASUREMENT THAT STOPPED THE ARC: the converged iterate exports clean

Before choosing a barrier shape, the check §17 and §20 both taught: does the target bind?
One export, four minutes, against a 6 h 20 m descent.

```
genome 2674f42 = step 100, INFEASIBLE by 4.112 um on the modelled hub fillet cap
junction    R_req  R_worst   edges   wedge  Kt_model  Kt_built   error
hub         0.771    0.771   24/24   326.0     2.033     2.033   +0.0%
rim         3.000    3.000   24/24   326.0     1.405     1.405   +0.0%
OCC valid | single solid | BRepCheck valid | no self-intersections | 0 degenerate
min curvature R 0.7711 mm (floor 0.25) | 50.44 g
```

**Why it builds, and it is not luck.** BUILD_PLAN Step 3 fitted the cap **2.4–3.9%
pessimistic on purpose**. The fixed point sits at **0.673%** of the cap and step 100 at
**0.536%** — four to seven times inside that conservatism. BUILD_PLAN Step 6 recorded the same
thing from the other side: *"the Step 5 barrier was firing on a design that builds — a 0.90%
overshoot of a cap deliberately fitted 2.4–3.9% pessimistic."* **This arc measured a smaller
overshoot than that one.**

**And the discarded design is the same wheel.** Step 55 (shipped) against step 100: loss
−0.61%, axle drop +0.40% → **−0.26%** off target, hub utilisation −1.28%, mass **+0.38%**.
Every metric under 1%.

So defect 5's entire realised cost in §19 was **0.61% of loss on a design that builds at +0.0%
`Kt` error** — not a lost design and not an unbuildable one.

#### The decision, with each option's reason for not being taken

- **Shifted knee / augmented Lagrangian / a larger weight** all drive `v*` to 0. **`v* = 0` is
  the wrong target**: the barrier's zero sits at a cap already 4–7× more conservative than the
  violation, so reaching it buys nothing the part cares about and gives up real fillet radius.
- **Fixing defect 8 to shrink `P`** would change which design is optimal and re-open §19's
  exchange rate — a far larger decision than a barrier shape, and it must not ride in on one.
- **Promoting step 100** — considered, declined. 0.61% of loss for +0.38% mass is not worth a
  re-promotion, a fresh `wheel.step`, and the regression-net churn §16 and §19 both paid.

**Defect 5 restated, so nobody re-ranks it on §19's framing.** It is **not a buildability
defect. It is a classification defect**: the tier system marks iterates infeasible against a
modelled cap conservative by several times the violation, so it discards designs that build.
Anyone fixing it should move the *boundary*, not the *barrier* — and should note that a
tolerance band is another calibrated constant, which is what BUILD_PLAN Step 3 spent a whole
arc removing.

#### The successors, re-ranked — and this time #1 provably binds

Three arcs running, the #1 successor has now failed the "does it bind" test three times
(§17 the slot arrival law, §20 the contact patch model, §21 defect 5). The item promoted here
is the one that cannot fail it:

1. ~~**`R_rim`'s box ceiling of 3.0.**~~ **BLOCKED BY §22 (2026-08-13).** The bound does bind
   — pinned at 3.0 for 80 of 101 steps — but raising it harvests −0.84 of loss the part never
   pays for: mass is bit-identical across `R_rim`, the FEA cannot see it, it buys margin at the
   junction that needs it least, and the rim has no cap model. **Defect 8 is now #1.**
2. **Defect 8** and §18's rate gate, still red at 0.379 against [0.5, 2.0] — and now known to
   be the thing feeding defect 5.
3. **The rim OD element-size step** (§20), still needing its cheap G7 measurement first.
4. **The Group C manifest-bite tolerance**, diagnosed in §19.
5. **Re-derive the five inherited characterisation gates.**
6. **Defect 5's boundary placement**, demoted here — worth 0.61% of loss on the evidence.

### 22. `R_rim`'s CEILING IS A TRAP, NOT A PRIZE. The bound binds; raising it would harvest loss the part does not pay for (2026-08-13).

§21 promoted `R_rim`'s untested box ceiling of 3.0 to #1 on the strongest evidence available:
it is **pinned exactly at 3.0 for 80 of 101 steps** of §19's run, from step 21 onward, in both
the shipped and the converged iterate. The optimizer is sitting on the bound. Unlike the three
successors before it, this one provably *binds*.

**Binding is not the same as being worth raising**, and one probe — forward evaluations at the
shipped genome, `coarse`, SVK, everything but `R_rim` held — says so.

| `R_rim` | loss | Δloss | `stress_margin` | util rim | util hub | `Kt` rim | drop mm |
|---|---|---|---|---|---|---|---|
| **3.00** | 51.0043 | — | 17.3184 | 0.52141 | 0.77074 | 1.39959 | 1.99079 |
| 3.50 | 50.7123 | −0.2920 | 17.0264 | 0.50722 | 0.77074 | 1.36149 | 1.99079 |
| 4.00 | 50.4876 | **−0.5166** | 16.8018 | 0.49602 | 0.77074 | 1.33144 | 1.99079 |
| 5.00 | 50.1624 | **−0.8419** | 16.4765 | 0.47935 | 0.77074 | 1.28669 | 1.99079 |

**The payoff looks real.** −0.52 at `R_rim` = 4.0 is **1.7× the entire tail defect 5 discards**
(−0.31) and 7.4% of §19's whole 100-step descent (−7.02). And it is a *lower* bound: nothing
was re-optimised around the new radius.

#### Every one of the objective's checks on that radius is blind, and there are four

1. **The mass term cannot see it.** `mesh_mass_g` is **bit-identical at 3.0, 4.0 and 5.0 —
   40.509301 g**, because the mesh models no fillets (the M2b finding
   `test_the_fillet_genes_have_no_fea_gradient_at_all` pins). §14 measured the built fillets at
   **6.18% of the solid volume**. So the optimizer collects the whole −0.84 **for free** while
   the real part gains material.
2. **The FEA cannot see it.** `axle_drop_mean_mm` is 1.99079 at every radius and `util_hub` is
   0.77074 at every radius, to the digit. The entire effect travels through the analytic
   `Kt_rim` and nothing else.
3. **It buys margin where there is already most.** The rim sits at **0.521** against the hub's
   **0.771**. This is **defect 8 in one number**: `util**2` summed over both junctions has no
   knee, so it keeps paying for margin at the comfortable junction at the same rate as at the
   tight one.
4. **The rim has no buildability cap at all.** BUILD_PLAN parked exactly this — *"`R_rim` is
   equally dead and the rim has no cap model at all."* Nothing would stop the optimizer
   requesting a radius OCC will not cut, which is the §3 failure at the hub, one junction over.

**So the −0.84 is not a design improvement, it is an accounting artefact.** Raising the ceiling
today would let the optimizer trade a mass it cannot measure for a margin it does not need, at
a junction with no buildability model, priced by a term with no knee.

#### The successors, re-ranked — and #1 is now the thing that makes the others readable

Four arcs, four #1 successors that did not survive their own check (§17, §20, §21, §22). The
difference here is that the check found a **trap** rather than a null, and it named the cause.

1. **Defect 8 — `util**2` has no knee.** It is what makes the rim payoff illusory (§22), what
   keeps defect 5 fed (§21), and it is already measured red by §18's own rate gate at 0.379
   against [0.5, 2.0]. It is the only item that three separate arcs have now landed on from
   three directions.
2. **Price the fillets' mass in the objective.** Currently `R_hub` and `R_rim` are free in
   mass, against 6.18% of the solid measured. Cheap: the exporter already publishes
   `fillets.volume_mm3`.
3. **A rim cap model**, which §22 makes a prerequisite for (4) rather than an optional parallel.
4. **`R_rim`'s ceiling** — blocked behind 1–3, not refuted. Once margin has a knee, fillets
   have mass and the rim has a cap, the same probe is worth re-running.
5. The rim OD element-size step (§20), the Group C manifest-bite tolerance, the five inherited
   characterisation gates, and defect 5's boundary placement (§21).

### 23. DEFECT 8 IS FIXED. The margin term has a knee, and it agrees with §22 from the other side (2026-08-13).

Working notes in `DEFECT8_PLAN.md`. §22's ranked successor #1, and the first item in five arcs
whose target survived its own check. **The objective changed; no design did.** `make test` goes
**7 failed / 438 passed → 6 failed / 440 passed**, §18's own rate gate green, no new red.

#### The shape, measured before it was chosen

`stress_margin` was `w * util**2`, marginal price `2*w*util` — proportional to `util`, so a
stated exchange rate can be right at exactly one design. Measured across the range:

| `util` | **`util**2` (was)** | `util**6` | knee@0.80, n=2 | knee@0.80, n=4 |
|---|---|---|---|---|
| 0.50 | **0.585** | 0.068 | 0.000 | 0.000 |
| 0.78 | **0.912** | 0.632 | 0.000 | 0.000 |
| 0.95 | **1.111** | 1.694 | 2.727 | 20.285 |
| 0.99 | **1.158** | 2.081 | 3.455 | 41.226 |

(marginal price, normalised at the 0.855 where `w` was calibrated.) **The old term's price
varied 2.0× across the whole meaningful range** — it paid nearly as much for margin at a
junction loafing at half its allowable as at one about to yield. §19 quoted the drift as 28%
between two designs; across the range it is worse.

**The term is now `soft_barrier(util_j − MARGIN_KNEE_UTIL)` per junction, `MARGIN_KNEE_UTIL =
0.80`** — literally the `stress` wall's own function with the knee moved from 1.0 in to 0.80.
One line says the policy: *the wall is at 1.0, the price starts at 0.80.* The quartic was
rejected for duplicating the wall: "priceless above 0.95" is already the barrier's job, per
§18's own comment. `DEFAULT_WEIGHTS["stress_margin"]` 20.0 → **325.0**, derived (328.49 exact
at §18's reference util 0.855) and rounded **down** for §18's stated reason, so this is a shape
change at constant rate.

#### §18's red rate gate: the design AND the fidelity, and only the control separates them

| genome | `n_phase` | cfg | `util` hub | ratio | gate |
|---|---|---|---|---|---|
| `e126cc3` | **2** | smoke | 0.56008 | **0.379** | **FAIL ← the live red** |
| `e126cc3` | 8 | coarse | 0.70898 | 0.607 | PASS |
| `e126cc3` | 8 | medium | 0.77952 | 0.734 | PASS |
| `e4219f3` | 2 | smoke | 0.65569 | 0.560 | PASS |
| `e4219f3` | 8 | coarse | 0.85506 | 0.952 | PASS |

On the shipped genome alone the verdict flips on `n_phase` and nothing else, which reads as a
pure test artefact. **It is not** — `e4219f3` passes at `n_phase = 2` too, so the gate
discriminates designs at all four settings. The promotion moved the design to where it fires;
`n_phase = 2` carries it over the line rather than inventing the drift. **The most useful row
is a PASS**: `e4219f3` → `e126cc3` at `coarse`/8 takes the stated rate **0.952 → 0.607**, a 36%
decay at a fidelity where the gate never fires. `util**2` degrades the policy quadratically
*exactly as the design improves*. That is better evidence for defect 8 than the red is, and
§19's "one is defect 8 measured" pointed at the wrong number.

#### Two consequences the tests forced

**The fillet genes go flat below 0.80, and that is not §15 defect 2 returning.** Defect 2 was
flat below util **1.0** — dead at every shippable design, a 14-gene search running in 8. This
is flat below **0.80**, where the project has decided fillet is worth nothing. The regression
test now asserts both halves.

**`R_rim` is a dead gene under the knee at every design in this repo, and §22 says that is
right.** On `e4219f3` the hub is at 0.85506 and the rim at **0.47963**, so `dL/dR_rim` is
exactly 0.0 — my first version of the above-knee test conflated the two junctions and caught
it. §22 measured the same thing from the opposite direction: raising `R_rim`'s ceiling looked
worth −0.84 of loss under `util**2` while every check the objective has on that radius is
blind. **The old term made `R_rim` look valuable; the knee prices it at nothing.** Two arcs,
opposite directions, one answer — which is the strongest evidence this change is right.

#### What is not done, and the successors

**The objective moved, so `e126cc3` is no longer its optimum.** It sits at util 0.780 against a
knee at 0.800, so the margin term is inert and `mass` has nothing opposing it until utilisation
climbs back. A descent would find a slightly lighter wheel sitting *at* the knee. **That run
has not been made; nothing is promoted and `best_solution.json` is untouched.**

1. ~~**The production descent under the knee'd objective**~~ **DONE — §26.** Candidate
   `09e8188`, 48.64 g against 50.25 g, promotion pending the Inventor import.
2. ~~**Price the fillets' mass.**~~ **DONE — §24.** `R_hub`/`R_rim` are free in mass against
   **8.77%** of the solid at the shipped genome, not §14's 6.18%.
3. **A rim cap model**, still parked, still a prerequisite for `R_rim`'s ceiling (§22).
4. The rim OD element-size step (§20), the Group C manifest-bite tolerance, the five inherited
   characterisation gates, and defect 5's boundary placement (§21).

---

### 24. THE FILLETS ARE PRICED: 4.406 g, 8.77% of the shipped part, and `R_rim`'s ceiling costs 4× what it harvests. Buildability is NOT what holds it at 3.0 (2026-08-13).

§23's ranked successor #2, and it needed no new machinery: the exporter has published
`fillets.volume_mm3` since §14, so this is **seventeen OCC exports off the shipped genome
`e126cc3`, one gene moved at a time**, and the manifests read. No FEA, no descent, nothing
promoted. Probe manifests are in `export/filletprice_*` and `export/rimcap_*`; their STEP
geometry was deleted after reading, being 40 MB of throwaway.

#### What a millimetre of fillet radius weighs

`R_hub` held at 0.728, `R_rim` swept. Mass is the OCC solid at `DENSITY_PLA`, i.e. what gets
printed — **the shipped wheel's fillets are 3553.19 mm³ = 4.406 g, 8.77% of its 50.25 g**.
§14's 6.18% is a *different genome's* number and this arc had been quoting it as if it were
the shipped one; `R_rim` has since gone to its ceiling, which is most of the difference.

| `R_rim` | fillet mm³ | fillet g | solid g | Δ solid vs 3.0 | OCC |
|---|---|---|---|---|---|
| 2.00 | 1819.45 | 2.256 | 48.10 | **−2.15** | 24/24, `Kt` +0.0% |
| 2.50 | 2634.84 | 3.267 | 49.11 | −1.14 | 24/24, +0.0% |
| 2.749 | 3078.25 | 3.817 | 49.66 | −0.59 | 24/24, +0.0% |
| **3.00 = shipped** | **3553.19** | **4.406** | **50.25** | — | 24/24, +0.0% |
| 3.50 | 4606.33 | 5.712 | 51.55 | +1.30 | 24/24, +0.0% |
| 4.00 | 5750.80 | 7.131 | 52.97 | +2.72 | **FAILS — see below** |

`R_rim` held at 3.0, `R_hub` swept: 0.4 → 50.10 g, 0.6 → 50.18 g, **0.728 → 50.25 g**,
1.2 → 50.52 g (and `kt_error_pct` **+16.0%** at 1.2, which is §16's analytic cap of 0.735
firing exactly where it should — the built fillet stops being the one the stress model priced).

**So: `R_rim` costs ~2.6 g/mm at the ceiling, `R_hub` ~0.53 g/mm — a factor of five.** Both
junctions carry 24 edges, and the rim's radius is four times the hub's, so this is the shape
of `∂V/∂R` rather than anything about the design.

#### The arithmetic §22 could not do, and it reverses the sign

§22 measured what raising the ceiling *harvests* and could only say the mass term is blind to
it. Now the blindness has a number. Pricing a built gram at the objective's own rate for a
gram — `MASS_WEIGHT / MASS_REFERENCE_G` = 30.0/36.5 = **0.821918 of loss per gram**, which
reproduces the shipped `mass` term of 33.2954 at 40.50941 g:

| `R_rim` | §22 harvest (`util**2`) | mass it actually adds | that mass, in loss | **net** |
|---|---|---|---|---|
| 3.50 | −0.2920 | +1.30 g | +1.068 | **+0.776 WORSE** |
| 4.00 | −0.5166 | +2.72 g | +2.236 | **+1.719 WORSE** |

**The prize was 4.3× smaller than the bill at 4.0, and 3.7× smaller at 3.5.** §22 blocked the
ceiling raise on four blindnesses and called it a trap; that verdict was right, and this is the
number it was missing.

**The assumption, stated rather than buried:** 0.821918/g is what the objective charges for a
gram of *mesh* mass, applied here to a gram of *fillet*. That a fillet gram is as unwanted as a
spoke gram is what "if `mass` could see the fillets" means — it is the premise of the fix, not
a measurement. Nothing here re-optimises around the new radius, exactly as §22's probe did not.

**And under the knee the harvest column is now exactly zero.** §23 made `stress_margin`
`soft_barrier(util_j − 0.80)`, and the rim sits at util **0.527**. `soft_barrier` is
identically flat below its knee, so `dL/dR_rim` is **0.0 analytically, not approximately** —
`tests/test_objective.py::test_but_above_the_knee_the_fillet_radii_are_live` pins it and
passes. Under the objective that ships today, raising `R_rim`'s ceiling harvests **nothing**
and costs 2.6 g/mm of real part. Three arcs, three directions, one answer.

#### `R_rim` = 4.0 does not build — and the boundary is 0.016 mm wide, which is NOT a cap

The 4.0 export is `valid: true` and its solid mass is real, but the rim fillet **fragmented**:

```
  n_edges_found 24  ->  n_edges_filleted 25
  fillet_families   [ {4.0 mm: 12 edges}, {3.4 mm: 13 edges} ]
  r_built_mm 0.0 (the no-single-family sentinel)
  kt_modeled 1.3314   kt_built 3.5 (= the Kt clamp, not a measurement)   kt_error_pct +162.9%
```

OCC split an edge and fell back to 3.4 mm on over half the rim, so the stress model would have
priced a 4.0 mm fillet the part does not have — the §3/§13 failure mode, at the junction with
**no cap model at all**. Reproduced identically on two independent exports.

**So I bisected it, on the whole-part predicate** (24/24 edges, one family, `r_built` =
`r_requested`, `Kt` error 0.0), and the answer is not the smooth limit the fragmentation
suggested:

| `R_rim` | 3.50 | 3.75 | 3.8750 | 3.8844 | 3.9344 | 3.9375 | 3.9688 | **3.9844** | **4.0000** |
|---|---|---|---|---|---|---|---|---|---|
| | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **✅** | **❌** |

All builds are 24/24 at a single family and **+0.0%** `Kt` error. **The boundary is between
3.9844 and 4.0000 — 0.016 mm wide**, with two spot-checks below it against the bisection's
monotonicity assumption, which `study_hub_cap.py` warns is not free on OCC.

**This corrects the reading the fragmentation first invited, and the correction matters.** A
cliff that sharp, sitting exactly on a round number, is a geometric coincidence on **one
genome**, not a manufacturing limit — and OCC accepts the rim fillet to within 0.4% of 4.0,
which is **33% above the ceiling of 3.0**. So buildability does **not** justify that ceiling.
The mass does, on its own, by the table above. §22's successor #3 — a rim cap model — is
therefore still wanted, but for the reason §16 wanted the hub's: to know the boundary as a
function of `t3` and the rim arrival angle rather than at one design. What it is *not* is the
thing standing between `R_rim` and 4.0.

#### What this does not say

It does not say the shipped wheel is 4.4 g heavier than it should be. The fillets are load
paths, `Kt` is computed from them, and §13 promoted the first part whose built fillets match
the ones its stress model priced. It says those grams are **unpriced**: the objective spends
them without seeing them, so a gene that is free to it is 2.6 g/mm to the printer. Whether
`mass` should read the OCC solid instead of the mesh is a real question and it is not asked
here — it would put an OCC export inside the descent loop, which is the reason it reads the
mesh in the first place.

---

### 25. THE SVK RE-SCORE GATE HAD BEEN DEAD SINCE §18, AND §19'S PROMOTION SILENTLY TURNED ITS CONTROL RED (2026-08-14).

Found by trying to use it. `make svk` is the §16-trap check — feasibility at **both** fidelities
before a promotion — and DEFECT8's candidate is the first thing to ask it for since §18. It
could not answer. Two independent breaks, neither of which anything would have reported,
because a driver that exits 2 on line one never reaches the line that would have shown the
second.

#### Break 1: the term-set guard, red since 2026-08-13

```
RuntimeError: the objective's term set moved: missing [], unclassified ['stress_margin'].
Classify it in BARRIER_NAMES or HEADLINE_NAMES — an unclassified term would be reported as
feasible
```

§18 added `stress_margin` to the objective; this driver predates it and classifies every term
explicitly, **by design** — the guard exists so a new term cannot be silently treated as
feasible. It did its job and nobody ran it. So `make svk` has been failing on line one since
§18 landed, and §19 promoted `e126cc3` without this gate having run.

**A tripwire now exists, because the driver's own guard only fires when someone runs it** —
and this one was run once a promotion, which is the worst possible moment to discover it.
`tests/test_objective.py::test_every_objective_term_is_classified_by_the_svk_rescore_gate`
asserts that `wheel_objective.TERMS` and the driver's `BARRIER_NAMES + HEADLINE_NAMES` are the
same set, in both directions. It checks that the judgement has been *made* for every term, not
that it is correct — which one a term belongs in is a judgement, and `smoothness` is the
cautionary tale. Falsified before being trusted: with `stress_margin` removed from
`HEADLINE_NAMES`, i.e. the file as it stood this morning, it fails with exactly the message the
driver would have given. **`make test` goes 6 failed / 440 passed → 6 failed / 441 passed.**

**`stress_margin` is a HEADLINE, not a BARRIER**, and this file's own history says why: the
first version of it listed `smoothness` — a curvature-rate integral, positive at every real
genome — as a barrier and *reported the promoted design as infeasible*. Under §23's knee,
`stress_margin` has the same shape: exactly 0.0 below util 0.80, positive above. The wall is
`stress`, at util 1.0, and it is already a barrier. A design sitting just above the knee is
**priced, not infeasible** — and that design is the equilibrium §23's entire policy aims at.

#### Break 2: the control was wired to "whatever is shipped"

The control reproduces §14's force-controlled service point and is the check that says the rest
of the table means anything. It compared `best_solution.json` against
`PLAN14_SHIPPED_SERVICE_REL` — a **constant §14 measured on `350f4c7`**. Those were the same
wheel at Step 3. §19 made them different wheels and the constant did not move:

| genome | file | linear | svk | rel | §14 | err | |
|---|---|---|---|---|---|---|---|
| `350f4c7` | `stage3_minwall_best_1.2.json` | 1.952966 | 2.408898 | **23.346%** | 23.346% | **0.00%** | PASS |
| `e126cc3` | `best_solution.json` (shipped) | 1.702531 | 2.051939 | 20.523% | 23.346% | 12.09% | **FAIL** |

**The solver is fine.** It reproduces §14 to five significant figures on the genome §14
measured. The control was failing because the wheel under it had been promoted out from
underneath it — a gate that goes red on a *correct* change is worse than no gate, and it would
have gone red at the next promotion too.

**Fixed by pinning the control to the file, not to the shipped pointer**:
`run_control` now reads `stage3_minwall_best_1.2.json` explicitly. That **restores** Step 3's
measurement rather than changing it — 350f4c7 is what `best_solution.json` held when Step 3 ran
— and makes the row what it was always for: a check on the SOLVER, which no future promotion
can turn red. Re-run after the fix: **both control rows 0.00%, gate PASS, exit 0.**

#### And a documented invariant that stopped being true at the same moment

The file states `minwall 1.2` **is** the shipped genome bit-for-bit, and used the pair as a
free cross-check on the pool reduction and mesh cache — two rows that must agree to the last
digit. §19 ended that: `minwall 1.2` is 350f4c7, `best_solution.json` is `e126cc3`, and the
rows now differ at every column. **That cross-check is gone and its loss was never recorded.**
The comment is corrected in place; what the row is *now* for is the control above.

#### The lesson this arc keeps re-learning, in its sharpest form yet

Four arcs have now found that a red gate is about the test rather than the design (§14 ×3,
§20 ×2, §22, and DEFECT8 step 1). **This is the first one found to be about the PROMOTION** —
the artifact a promotion updates is the same artifact three gates take their reference from,
and nothing in the repo checks that a promotion leaves them consistent. §13's banner discipline
exists for exactly this in `PLAN.md`; the study drivers have no equivalent. Worth a successor:
a promotion check that greps the drivers for `best_solution.json` and asks, for each, whether
that reference means "the design we ship" or "the design this constant was measured on".

---

### 26. THE PRODUCTION DESCENT UNDER THE KNEE. **PROMOTED — `e126cc3` → `09e8188`** — lighter on both mass measures, with deflection back inside a gate the incumbent had fallen outside (2026-08-14).

Working notes in `DEFECT8_PLAN.md` step 4. §23's ranked successor #1, and the first item in six
arcs that was a *run* rather than a measurement. 6 h 19 m, 101 objective calls, exit 0, every
knob §19's so §19's own run is an exact control and the objective is the only difference.

**Step 0 was predicted before the run and read exactly.** `e126cc3` under the knee must lose
its entire `stress_margin` of 17.7156 — it sits at util 0.7795, below the 0.80 knee — so step 0
had to be 51.3892 − 17.7156 = **33.6736**. It was. That is the cheap check that a six-hour run
is measuring the change it claims, and it is the one the skipped probe would have given.

| | shipped `e126cc3` | candidate `09e8188` | |
|---|---|---|---|
| loss, same objective | 33.6736 | **32.7446** | −0.9290 |
| **OCC solid mass** | 50.25 g | **48.64 g** | **−3.20%** |
| mesh mass | 40.509 g | 39.470 g | −2.57% |
| fillet mass (§24) | 4.406 g | 3.818 g | −0.588 g |
| axle drop | +0.403% | **−0.129%** | |
| util hub | 0.7795 | 0.8201 | |
| `R_rim` | 3.0000 | 3.0000 | still pinned |
| `t1` / `t2` | 1.2047 / 1.2524 | 1.2000 / 1.2000 | onto the floor |

**It is lighter by MORE on the built part than on the modelled one**, because it also gave back
0.588 g of the fillet mass §24 measured the objective cannot see. First movement in this repo
where the unpriced mass moved *with* the priced mass instead of against it — and it happened
without anyone teaching `mass` to see fillets, because `R_hub` came down for stress reasons.

**What it paid: four of §19's twenty-two points of utilisation headroom.** That is the knee's
stated policy — margin below 0.80 is worthless, so the design settles just above it — and it is
what a promotion accepts rather than a surprise. The `stress` wall at 1.0 is untouched.

**Feasible at BOTH fidelities under BOTH kinematics, every barrier 0.0** (§16's trap), export
clean at **19 checks / 0 failures** with `Kt` error **+0.0%** at both junctions — the §13
property the current part was promoted for, preserved. `make test` **6 failed / 441 passed**:
the same six documented reds, plus the one test §25 added.

**Defect 5 still bites, and less: 46 of 101 iterates violate `fillet_cap` against §19's 74**,
and the selected iterate moves from step 55 to **74**. §21's diagnosis is unchanged and the
defect-6 selection rule is again the only reason a shippable iterate came out.

**`max_stress_mpa` +29% and `stress_scale_measured` +35% are NOT the constraint** — they are the
mesh-divergent field max and its diagnostic, which M8b-i.6 step 2 removed from the constraint
for exactly that reason. The constraint is `util` = 0.8201, under the wall.
`min_scaled_jacobian` 0.8266 → 0.7828 is a real mesh-quality loss, worth watching, not gating.

#### PROMOTED (2026-08-14). The import was clean.

`best_solution.json` is **`09e8188`**, copied from `stage3_knee_best_medium.json` verbatim
apart from a `note` — a provenance field the file used to carry and lost at §19, restored here.
`e126cc3` is preserved as `stage3_margin_best_medium.json`. `export/wheel.step` and its
manifest rebuilt from the new file: **48.64 g**, 111 faces / 327 edges, 24/24 at both junctions,
`Kt` error **+0.0%** at both, bite 0.5344 / 1.6416. `tests/test_golden.py` still reads
`best_solution_ga_beam.json` — §10's decoupling is what makes a promotion a one-file change
that cannot re-baseline the regression net, and it was checked, not assumed.

**The top-of-file banner was amended, and it needed more than this promotion's line.** It still
declared `350f4c7` shipped and stated that the shipped genome had not changed — false since
**§16 on 2026-08-11**, i.e. through two promotions, in the one place `SVK_PLAN.md` step 7
specifically requires to be updated when the genome moves. The full chain is now at the top and
the stale sentences are scoped to their dates rather than deleted. **Same defect as §25, in the
mechanism designed to prevent it** — which is why successor #1 below is what it is.

**`make svk` did NOT need re-running after the promotion, and that is §25's fix working.** The
warning I wrote before promoting — that promoting moves the reference the gate takes its
constants from — no longer applies: §25 pinned the §14 control to `stage3_minwall_best_1.2.json`
by file, so it is immune to promotions by construction. The `shipped` row simply follows
`best_solution.json`, which is what that row is for. The canonical `study_svk_rescore.json`
artifact is still Step-3 era and refreshing it is optional, not a gate.

#### The successors, re-ranked

1. ~~**A promotion-consistency check.**~~ **DONE — `tests/test_promotion.py`, five tests, and
   the first thing it caught was this file's own banner.** See below.
2. **The ±0.3% deflection gate's standing.** It is a plan-level number the incumbent violates
   at +0.403%, applied as binding in SVK_PLAN step 5 and BUILD_PLAN steps 8 and 10 and quietly
   not applied in §19. This candidate passes it either way so the question did not have to be
   settled here; the next one may not be so obliging.
3. **A rim cap model** (§22, §24) — still parked, and §24 narrowed what it is for: the boundary
   as a function of `t3` and the rim arrival angle, not as the thing blocking `R_rim`.
4. The rim OD element-size step (§20), the Group C manifest-bite tolerance, the five inherited
   characterisation gates, and defect 5's boundary placement (§21).

---

### 27. THE PROMOTION CONTRACT IS NOW A TEST. `tests/test_promotion.py` (2026-08-14).

§26's ranked successor #1, built immediately after §26's promotion because that promotion had
just produced the second instance of the defect in one session — §25 in the study drivers, and
PLAN.md's own banner, stale since §16 through two promotions, in the one place `SVK_PLAN.md`
step 7 requires to be amended when the genome moves.

**What it deliberately does NOT do.** There are ~100 references to `best_solution.json` across
`src/`, `studies/` and `tests/`, and **almost all of them are correct** — they mean "the design
we ship" and following a promotion is exactly right. The defect is the narrow case: a
genome-SPECIFIC constant sitting next to a read of a file that MOVES. No grep separates those,
and a scanner that flagged all 100 would be turned off within a week. So it does not scan.

**What it does instead: put the tripwire on the promotion.** `SHIPPED_GENOME_HASH = "09e8188"`
is recorded in the test file, and moving `best_solution.json` turns it red with a **six-item
checklist in the failure message** — amend the banner, rebuild `wheel.step`, re-check the
drivers that pair the shipped pointer with a measured constant, run `make svk` on the
assumption it has rotted, preserve the outgoing genome, leave the golden net alone. The
assertion is not the deliverable; the checklist arriving *at the moment someone is promoting*
is. Nothing in this tree did that before, which is why §16 and §19 both left leftovers.

Five tests: the shipped hash against what the tree documents; `wheel.step` not older than the
genome it claims (staleness — the hash half is already covered by
`test_golden.py::test_genome_hash_matches_manifest`, which compares two files to each other and
therefore agrees with itself when both are stale); the §14 control genome `350f4c7` unmoved,
since §25's fix pins the control to a file and a file can be overwritten too; the golden
reference `36aed36` unmoved; and the shipped file carrying a `note`, a provenance field that
existed at §13 and was lost at §19.

**Each was falsified before being trusted** — hash moved, control moved, golden re-baselined,
STEP stale, note dropped — and each failed with its own message. `best_solution.json` was
restored byte-identically afterwards, hash and mtime verified.

#### And the draft was wrong in a way worth recording

The first version asserted that `test_golden.py` must not reference `best_solution.json` at
all. It does, in exactly one place, and its docstring argues at length why that is right: a
manifest hash is a statement about *whichever genome the exporter last ran on*, so reading the
pinned fixture there would turn a traceability check into a second copy of the fixture. **§10
decoupled the FIXTURE, not the file.** The test was measuring the documentation as the
violation — the same class of error as reading a red gate as a design defect, which this tree
has now made six times (§14 ×3, §20 ×2, §22, DEFECT8 step 1, and here). The file said so
itself; scanning source for intent is the wrong instrument, and that is now written into the
test that replaced it.

---

### 28. A RED THAT THREE PLANS DEFERRED AS "EXPORT PRECISION" WAS A TOLERANCE THE DESIGN OUTGREW — and §26's promotion turned it green by luck (2026-08-14).

`test_export_contract.py::test_the_bite_is_the_volume_divided_by_the_right_thickness` has been
red since before CONTACT_PLAN Step 0, and was left out of scope three times under the same
label: §19 filed it Group C, CONTACT_PLAN Step 0 tabulated it as Group C, and Step 3 said in
terms **"out of scope, said explicitly: the Group C manifest-bite tolerance — an export-precision
defect, not a contact one."** The post-promotion suite came back **5 failed / 442 passed**
against the 6 / 441 the promotion predicted, and the test that had flipped was this one.

**A red that goes green on a change that had no business touching it is not good news.** The
manifest rounds the overlap to 2 dp and the bite to 4 dp, and `junction_bite` is exactly
`overlap / (t² · W)` — so the volume's half-ulp of 0.005 arrives at the assertion *divided by
t²·W*. The tolerance was a hard-coded `1e-4`. That is not a constant budget; it is achievable
only for **t > 2.113 mm**:

| genome | t0 | t3 | worst-case residual, hub / rim | inside 1e-4? |
|---|---|---|---|---|
| `36aed36` GA/beam — *when the docstring was written* | 2.4774 | 2.0000 | 8.6e-5 / 1.06e-4 | hub yes, rim no |
| `350f4c7` §14 min-wall | 1.2000 | 1.4614 | 2.05e-4 / 1.55e-4 | no |
| `e126cc3` §19 | 1.5594 | 1.4630 | 1.42e-4 / 1.54e-4 | no |
| `09e8188` **shipped** | 1.4738 | 1.4313 | 1.53e-4 / 1.59e-4 | no |

The old docstring read *"on the shipped genome (t0=2.48, t3=2.00)"* — true of `36aed36`, and
false through four promotions since. **Nothing was ever imprecise about the export.** A fixed
tolerance was compared against a quantity whose rounding scales as 1/t², and the wheel got
thinner. The label three plans inherited from each other was wrong, and re-reading it was
cheaper than the four sweeps it survived.

It went green at §26 because the volume happened to round favourably: residual **4.5e-5 hub,
2.1e-5 rim** against a 1.53e-4 budget. **That is worse than the red** — a test that passes on
where the rounding landed hides until it does not, and it would have flapped on the next
export of the same genome.

**The fix derives the tolerance rather than widening it**, which is what CONTACT_PLAN's own
rule demands ("do not widen 0.05 without stating what the number now is"):
`tol = 0.005/(t²·W) + 5e-5` — the volume's half-ulp pushed through the division, plus the
bite's own. It is 1.5e-4 on the shipped genome and 8.6e-5 on `36aed36`, and it cannot go stale
when the walls move again. Falsified twice: the t0/t3 swap it exists to catch (gap **3.22e-2**,
210× the budget) and a perturbation at 2× the derived tolerance. The manifest was restored
byte-identically, md5 verified.

#### The finding underneath, which is about the design and not the test

The swap this test guards has lost most of its signal, because **t0 and t3 have converged**:

| genome | t0 | t3 | a t0/t3 swap moves the hub bite by |
|---|---|---|---|
| `36aed36` | 2.4774 | 2.0000 | **53.4%** |
| `09e8188` | 1.4738 | 1.4313 | **6.0%** |

Still 0.0322 absolute, still 210× the tolerance, so the check holds today. But the spoke is
becoming prismatic — 0.04 mm of taper over its length — and if a descent drives t0 = t3 the
swap becomes undetectable *here* by construction, no matter what the tolerance is. That needs
catching at the exporter, where the two thicknesses are selected, rather than in a manifest
check that can only see their ratio. **Filed as a successor, not fixed:** it is a real gap, but
it is not yet open, and §17 is the standing lesson about ranking work on a bound that does not
bind.

#### Why this is not the re-fitting the standing rule forbids

A prior session diagnosed this same defect ("a 1e-4 bite tolerance against a manifest storing
2 dp") and deliberately **left it red**, on the rule that *loosening a tolerance in the same
session as the promotion that reddened it is indistinguishable from re-fitting the gate to the
run that breached it.* That rule is right and it is why this needs saying out loud rather than
glossing:

- **The promotion did not redden this test — it turned it green.** The rule guards against
  relieving a gate that a promotion breached. Here §26 relieved it by accident, and the change
  puts a principled bound back where luck was holding it.
- **The bound is derived, not fitted.** `0.005/(t²·W) + 5e-5` comes only from the manifest's
  rounding decimals and the definition of `junction_bite`. No measured residual enters it. The
  residual it permits today (4.5e-5) is a third of it.
- **It makes the test STRICTER where it used to be lax.** On `36aed36` the derived bound is
  8.6e-5 against the old 1e-4. It is not a widening; it is a slope where there was a constant.

**The alternative fix, considered and rejected for now:** make the *exporter* write the overlap
to 4 dp instead of 2. That attacks the root cause — the artifact discards precision in a value
that gates buildability — and would let the tolerance stay a tight constant (the budget falls
to 5.1e-5 for any plausible t). It is the better long-term answer. It is **not** being done in
this session because it changes a shipped artifact's contents immediately after a promotion,
and the 2-dp volumes are quoted in §24's fillet-price tables and the DEFECT8 records. Filed as
a successor. Until then the derived bound retains ample power: it is exceeded by any t0/t3
ratio error above ~0.1%, against the 6.0% a full swap produces.

#### SUITE RECORD — 2026-08-14, after §26's promotion, §27's contract tests and §28's fix

```
make test:   5 failed / 447 passed in 1685.73 s (28:05)   [452 collected]
  452 = 447 + 5, and the 5 NEW tests are §27's tests/test_promotion.py — an earlier run
  read 5 / 442 because it collected four minutes before that file was written.

THE ELEVEN REDS CONTACT_PLAN STEP 0 TABULATED ARE NOW THE FIVE IT CALLED INHERITED, and
nothing else.  Every red that plan classified into a group is green:
  Group A  x3   test_contact.py                          fixed, CONTACT_PLAN Step 3
  Group B  x1   test_objective.py::test_the_margin_...   fixed, PLAN §23 (defect 8)
  Group C  x1   test_wheel_fea.py::test_only_the_rim_... fixed, CONTACT_PLAN Step 3
  Group C  x1   test_export_contract.py::test_the_bite_. fixed, PLAN §28  <- this section
the five that remain are SVK_PLAN Step 0's deliberate pair plus §14's three, red by
intent and unchanged in count since 2026-08-13:
  test_gnl.py::test_the_correction_is_not_a_constant_over_the_design_space
  test_gnl.py::test_the_correction_enters_at_first_order_in_the_load
  test_wheel_fea.py::test_the_rim_band_holds_a_large_minority_of_the_compliance
  test_wheel_fea.py::test_the_beam_to_wheel_ratio_is_not_a_constant
  test_wheel_fea.py::test_a_thicker_rim_monotonically_stiffens_the_wheel

NO NEW RED from the promotion, from the five tests §27 added, or from §28's fix.
```

---

### 29. THE ±0.3% DEFLECTION GATE CANNOT BE ADJUDICATED, AND THE EVIDENCE WAS ALREADY IN THE TREE. Successor #2, closed by measurement (2026-08-14).

SVK_PLAN's closing item asked for the GCI treatment on the deflection QoI so the gate could be
stated against an extrapolated value instead of a rung. `studies/study_deflection_gci.py`
(`make gci`, 1 h 35 m) runs the ladder on the gate's OWN quantity — `axle_drop_mean_mm`,
8-phase uniform stencil, both kinematics, `flank_orientation` pinned at the finest rung and
verified identical at every rung, so it is a pure mesh refinement.

| rung | elem | h | linear mm | err % | **svk mm** | **err %** |
|---|---|---|---|---|---|---|
| smoke | 960 | 0.03227 | 1.59689 | −20.156 | 1.88090 | −5.955 |
| coarse | 4704 | 0.01458 | 1.67772 | −16.114 | 1.97608 | −1.196 |
| medium | 12288 | 0.00902 | 1.69502 | −15.249 | **1.99742** | **−0.129** |
| fine | 31200 | 0.00566 | 1.70683 | −14.659 | 2.01274 | +0.637 |

*(Element counts and `h` corrected 2026-08-15 — the first version of this table reported
`wheel_mesh`'s spoke-block ladder, 72/384/1280/4096, for a QoI solved on `wheel_wheel`'s
12-sector wheel. See "The h that was measured on the wrong mesh" below. The four deflections
are the measured ones and did not move.)*

The `medium`/SVK row reproduces DEFECT8 gate 4's −0.129% for the shipped genome exactly, which
is what makes the rest of the ladder comparable to the promotion record.

**THE OBSERVED ORDER IS p = 0.638.** On quadratic (Q9) elements, against a smooth solution,
that should be near 2. Sub-first-order is the signature of a singularity in the solution, and
it gives a **GCI on the finest rung of 2.749%** — against a **±0.3%** gate. *The numerical
uncertainty on the number the gate is stated in is nine times the width of the band.*

**Extrapolated: 2.05700 mm, +2.850%.** Every rung flatters the design, monotonically, and the
rung the gate is evaluated at is the second-most flattering of the four. The wheel deflects
*more* than the shipped record says, not less.

#### The three objections, answered by measurement rather than by argument

1. **"It depends how you define h"** — the ladder is not uniformly refined (span ×2.0,
   thickness ×1.500 then ×1.333), so `h` is genuinely ambiguous. All four defensible
   definitions are carried in the artifact. Under SVK the **smallest GCI any of them
   produces is 2.416%**, already 8× the band, and none of them puts the extrapolated value
   inside it. The verdict is identical under every one, which is why it is stated at all.
   What is NOT stable across them is `p` itself — **p ∈ [0.049, 0.685]** — and the spread is
   not noise: `1/n_thick` is a degenerate choice on this ladder, because thickness refines
   4 → 6 → 8 while span refines 48 → 96 → 192, so calling the thickness cell "the" cell size
   fits an order to a direction that is barely being refined. The three isotropic-ish
   definitions agree at **p ∈ [0.479, 0.685]**. Quote `p` from the primary definition
   (`1/sqrt(n_elements)`) or not at all.
2. **"Discretisation error cancels between similar designs, so the gate still ranks them"** —
   it does not cancel enough, and §19's and §26's own scheduled fidelity checks measure by how
   much. The coarse-minus-medium gap is **−1.59 pp at `e4219f3`, −0.863 pp at `e126cc3`,
   −1.077 pp at `09e8188`**: a **0.73 pp spread across three consecutive shipped genomes**,
   still more than twice the ±0.3% band. The part of the error that survives design-to-design
   is on its own larger than the gate.
3. **"The contact patch is under-resolved on the coarse meshes"** — already ruled out, by
   `study_wheel_fea.run_refinement`'s own control: re-running the ladder with a deliberately
   over-wide 12° patch (32/51/89 nodes in the patch instead of 8/13/22) leaves the rate
   sub-second-order at 1.613. That control was built for exactly this alternative and it
   fails, which is what leaves the **unfilleted junction corner** as the explanation.

#### The part that should sting: this was recorded, and filed under "does not matter here"

`studies/study_wheel_fea.json` has carried **`criterion_met: false`** on the axle drop, with a
**GCI of 0.633%** — already 2× the ±0.3% band — since before any of this. It was not missed;
it was correctly set aside, by a docstring that says so in terms:

> "The gate's DECISION does not rest on the axle drop being converged to 0.5%; it rests on the
>  compliance split being stable. Tracked separately so a failed convergence criterion cannot
>  be mistaken for a failed conclusion."

That is right *for that study*, whose conclusion is about the compliance split. What nobody
did was ask whether anything ELSE in the tree rested on the number being set aside — and a
plan-level promotion gate in three other files did. **A convergence failure quarantined inside
one study's scope was load-bearing outside it.** The `run_refinement` machinery, the control
that rules out the patch, and the failing criterion were all present; only the connection was
missing, and it cost 95 minutes to make.

#### THE CALL — the absolute band is retired, the relative clause is what survives

**±0.3% is withdrawn as an absolute promotion gate.** A band cannot adjudicate a quantity whose
numerical uncertainty is 8–12× its width, and the tree has been reading mesh error as design
quality: SVK_PLAN step 6 *blocked* a candidate at +1.65% on a number carrying ±2.8%.

**What replaces it, and it is what §26 already did.** DEFECT8's gate 4 required deflection
measured *both* against the band and against the incumbent, and pre-registered that "if they
disagree the disagreement is the finding". They agreed, so **§26's promotion is not unsettled
by this** — but it is now clear which of the two clauses was carrying it. The surviving gate is:

> Deflection is measured at a NAMED rung and reported with it (`medium`, SVK, 8-phase uniform),
> and the binding clause is **no worse than the incumbent measured identically**. Absolute
> distance from 2.0 mm is reported as an observation, not as a pass/fail.

This is decidable, it is what the last promotion turned on in practice, and it does not pretend
to a precision the model does not have.

**What would earn the absolute band back:** fillet the junction in the FEA model. The exported
solid has filleted junctions — §16 through §24 exist to build and price them — while the FEA
mesh still meets the rings at a sharp re-entrant corner and recovers `Kt` by a correction
factor instead. Removing the singularity is what would restore the convergence rate and make an
absolute deflection gate meaningful. That is now the highest-ranked open item, and unlike the
GCI study it is not cheap. **Named, not started**, and the hypothesis is explicitly the corner —
the patch control rules out the other candidate, but nothing here has yet re-run the ladder on
a filleted model, which is the test that would confirm it.

> **THE CORNER NEVER NEEDED CONFIRMING — see §30 (2026-08-15).** M4 established that the peak
> stress diverges, `test_peak_stress_diverges_but_the_field_converges` has pinned it since §14,
> and M8b-i.6 step 2 rebuilt the whole stress constraint around it. This paragraph asking for a
> filleted-model ladder "to confirm it", and §29 spending 95 minutes inferring it from a
> deflection order, were both looking past evidence already in the tree. §30 adds the rate
> (−0.44 on log h), the per-corner localisation and the wedge angles; it does not supply a
> confirmation that was missing. The paragraph is left standing to record the detour.

#### THE h WAS MEASURED ON THE WRONG MESH — and the Williams agreement is RETRACTED (2026-08-15)

**`study_deflection_gci.py` drew every `h` from `wheel_mesh` while `wheel_objective` solved on
`wheel_wheel`.** Both modules export configs called `smoke`/`coarse`/`medium`/`fine` and they
are different meshes: at `medium`, `wheel_mesh` is one 128×10 spoke block (1280 elements) and
`wheel_wheel` is the 12-sector wheel (12288). The study passed the config NAME to the objective
— which resolved it correctly — and the same name to `wheel_mesh.get_config` for its cell size,
which did not. Refinement ratios were reported as 1.826/1.789 against a true **1.616/1.593**.

**What this cost, and what it did not.** Fixed by `--reanalyse` on the saved report, no FEA:

| | reported | corrected |
|---|---|---|
| r21 / r32 | 1.8257 / 1.7889 | **1.6162 / 1.5934** |
| observed order p | 0.5023 | **0.6379** |
| extrapolated | 2.05789 mm (+2.894%) | **2.05700 mm (+2.850%)** |
| GCI(fine) | 2.804% | **2.749%** |

The diagnosis is confirmed by the one h definition that did NOT move: `1/n_span` reads
**p = 0.4789 both before and after**, because span happens to refine ×2.0 per rung in *both*
modules (24→64→128→256 and 16→48→96→192 both give 2.0 across the extrapolated three). Every
definition that touches a count where the two ladders differ moved; the one where they agree
did not.

**The extrapolated value and the GCI barely moved — and that is the trap, not the reassurance.**
From three points, `p` and `r` reach Richardson only through `r^p`, which the measured φ very
nearly fix on their own; rescaling `h` slides `p` and `r` in opposite directions and leaves the
product alone. So a convergence study can be **wrong about the one quantity it exists to
produce while every headline it reports stays right**. Every conclusion §29 draws about the
GATE stands unchanged — the band is still 9× too narrow, still undecidable under every h.

**What does not stand is the exponent, and with it the confirmation of the corner.** The
retracted claim: Williams' mode-I eigenvalue for a traction-free re-entrant wedge,
`sin(λω) + λ sin(ω) = 0`, gives λ = 0.5030 at the shipped hub's 322° wedge and 0.5035 at the
rim's 320°, and the measured p = 0.5023 was read as the geometry predicting the convergence
exponent. **The true p is 0.638. λ is 0.503. They do not agree**, and the agreement that was
reported existed only because `h` was 25% wrong in the log — `ln(1.826)/ln(1.616) = 1.25`, and
`0.5023 × 1.25 = 0.628`. The eigenvalue arithmetic was right and is still right; what it was
being compared against was not.

**The corner hypothesis is now neither confirmed nor refuted.** p = 0.638 sits between λ ≈ 0.50
and the 2λ ≈ 1.0 a smooth functional would give — closer to neither, and the honest reading is
that a single fitted exponent from three rungs of a ladder that refines five different block
directions at four different rates does not identify a mechanism. The one thing p = 0.638 does
still establish is the thing the gate turned on: **this is far from the p ≈ 2 that Q9 elements
give on a smooth solution**, so something singular or near-singular is in the field. The
unfilleted junction remains the leading candidate — the patch control still rules out the other
one — but it is a hypothesis again, and **the filleted-model ladder is back to being the test
that would settle it.**

The cross-genome discriminator stays retired, and for a reason the correction does not touch:
λ moves by less than 0.01 across every wedge this design family produces (322°, 320°, 326°), so
`p` could not track it either way.

**What this cost in ranking.** The strategic claim built on the old number — "λ is a property of
the wedge angle, the optimizer cannot descend out of it, no future promotion will converge
better than p ≈ 0.5" — is withdrawn along with the agreement it rested on. The FEA model does
still have a sharp corner **the physical part does not have**: the exported solid fillets both
junctions at the full requested radius, 24/24 edges, `kt_error_pct` +0.0%. Filleting the FEA
junction is still the highest-ranked open item. It is now ranked on the size of the convergence
deficit rather than on a matched exponent, which is a weaker case than §29 first made for it.

**The regression net that did not exist.** This study had no test file at all — an
analysis-only module that had already shipped one silent indexing bug. `tests/test_deflection_gci.py`
now pins: the counts come from an assembled `WheelMesh` and not from any formula or from
`wheel_mesh`; the refinement ratios the docstring quotes; `observed_order` against synthetic
constant AND non-constant ladders (only the second can catch the index swap); the rescaling
property above, stated as the reason `p` needs its own pin; and the retraction itself, as an
inequality, so `p ≈ λ` cannot quietly come back.

### 30. THE CORNER SINGULARITY WAS ALREADY KNOWN — WHAT WAS MISSING IS ITS RATE, ITS LOCATION AND ITS EXPONENT (2026-08-15).

**Read this correction first, because the first draft of this section got it wrong.** §30 was
written up as "the peak stress does not converge — measured for the first time". It is not the
first time. The tree has known it since M4, and says so in three places that were not consulted
before the section was written:

- **The banner at the top of this file**: *"the max is not a number. It diverges 31.02 → 41.54
  → 48.47 under refinement"*, and M8b-i.6 step 2 changed the entire stress constraint *because*
  of it — `c = max/pnorm` was anchored to a singularity, so `c * pnorm` converged at no
  exponent at either design.
- **`tests/test_wheel_fea.py::test_peak_stress_diverges_but_the_field_converges`**, which
  already asserts the rim's `max_singular_mpa` grows monotonically over coarse/medium/fine and
  by more than 20%, with a docstring explaining that quoting the max as a stress is the error
  it exists to prevent.
- **§14**, which renamed a test to `test_the_junction_is_re_entrant_enough_to_be_singular` and
  bounded the wedge at `360 − MAX_ARRIVAL_DEG` = 295° for every genome in the box, and **§15's**
  note that *"the aggregate is tracking the field and the corner moved more, and that is the
  same corner the export then refused to fillet."*

So the mechanism was established, tested and acted on. Writing it up as new was the failure
mode this file has recorded twice already — §28's "an export-precision defect" inherited
through three plans unchecked, and §29's quarantined `criterion_met: false` that three other
files leaned on. **The lesson generalises in both directions: check what the tree already
concluded before claiming a finding, not only before trusting one.** Nine words of `grep` would
have caught it.

#### What `studies/study_corner_singularity.py` (`make corner`, 8.5 s, 1.56 GB) actually adds

The existing test asserts *that* the peak grows. Four things nobody had measured:

**1. THE RATE, which is the part that connects to an exponent.** The old test's bar is "> 20%
over coarse..fine". The measured rate is `d log(peak) / d log(h)` = **−0.442** on the global
maximum (61.92 → 99.13 → 126.25 → **150.59 MPa** across smoke/coarse/medium/fine, 2.43×). A
convergent peak gives zero. A growth threshold cannot be compared to a wedge angle; a rate can.

**2. WHICH corner, out of four.** The junction has two re-entrant corners per ring, not one, and
they behave differently:

| corner | peak MPa: smoke → fine | slope | growth |
|---|---|---|---|
| rim `P_c` | 61.92 → **150.59** | −0.442 | 2.43× |
| hub `P_c` | 47.31 → 120.92 | −0.476 | 2.56× |
| hub `P_t` | 18.71 → 96.22 | −0.616 | 5.14× |
| rim `P_t` | 13.51 → 75.40 | −0.597 | 5.58× |

**The wheel's global maximum stress is the rim's `P_c` corner at every rung** — the two `P_c`
corners are the loaded ones, while the two `P_t` corners start four times lower and diverge
faster. `P_t` is where the straddling flank crosses the ring circle; `P_c` is the centerline
endpoint. Only `P_t` was ever named in this file before.

**3. THE WEDGE ANGLES, MEASURED ON THE MESH.** Summing the incident elements' interior angles at
each corner node — 360° for an interior node, verified as a control — gives **hub `P_t` 321.10°,
hub `P_c` 296.75°, rim `P_t` 321.33°, rim `P_c` 307.94°**. All four re-entrant, and the two
`P_c` values are new: the tree only ever had the `360 − arrival` figure, which describes `P_t`.

This is deliberately **not** the export manifest's 322°. The manifest measures the exported
solid, which is *filleted* at these corners and has no wedge to measure. Reading a prediction
off the wrong body is what §29 did with the cell size.

**4. WILLIAMS' EXPONENT PER CORNER, and an honest account of how well it agrees.** Solving
`sin(λω) + λ sin(ω) = 0` at the four measured wedges gives **λ = 0.5032 / 0.5144 / 0.5031 /
0.5079** (solver verified against a crack at exactly 0.500000 and the textbook 0.5445 at 270°).
Against the divergence rates: **0.384 / 0.524 / 0.403 / 0.558**. Agreement to about ±0.13 on a
ladder whose coarsest rung is 960 elements — **agreement on the mechanism, not on the number.**

The radial-decay fit (bin by log r, take the max over θ to divide out the angular shape, fit the
inner decade) is sharper where it works: **hub `P_c` λ = 0.573 at correlation −0.997, rim `P_c`
λ = 0.473 at −0.996**. It does *not* work everywhere — `rim:P_t` returns a slope of +0.008 at
correlation +0.03, no detectable decay in that window at all, while its peak diverges faster
than any other corner's. **That disagreement is unexplained.** The likely cause is a fit window
contaminated by the neighbouring corner, but nothing here has measured that, and it is in the
artifact rather than smoothed away.

`tests/test_corner_singularity.py` pins the divergence hard (strict monotone growth, slope
< −0.15) and λ loosely (abs=0.15). Those are two different strengths of claim, and §29 went
wrong precisely by reading a three-decimal coincidence as confirmation.

#### The qualifier, which the tree also already knew and which must travel with the headline

**The optimizer does not see the raw peak.** The constraint is `Kt * sigma_nominal(p=4)`, and a
p=4 aggregate is *deliberately* insensitive to a singular corner — that is what `Kt` exists to
bridge, and it is why M8b-i.6 step 2 changed the constraint in the first place. Measured up the
same ladder:

| | smoke | coarse | medium | fine | slope |
|---|---|---|---|---|---|
| p-norm `stress_utilisation` (SVK) | 0.7694 | 0.8057 | 0.8201 | 0.8301 | **−0.044** |
| raw peak von Mises | 61.92 | 99.13 | 126.25 | 150.59 | −0.516 |

+7.9% across the whole ladder, +1.2% from `medium` to `fine`, a **12× slower** divergence. Two
things follow and they pull in opposite directions, so state both: the p-norm is doing its job
and the constraint is not the singularity; **and** a −0.044 slope is still not zero, so the
stress constraint is not a converged quantity either — §13's "utilisation 0.80" is a reading at
a rung, exactly like the deflection is. "The peak diverges" does not license "the constraint is
meaningless".

#### WHAT THIS DOES TO THE RANKING — less than the first draft claimed

Filleting the junction in the FEA model stays the top open item, but §30 does **not** promote it
on new evidence; the evidence was already there and M8b-i.6 already acted on the part of it that
could be acted on cheaply. What §30 changes is smaller and worth having anyway: the fillet work
now has **four named corners with measured wedge angles and per-corner divergence rates to
validate against**, which is what a filleted-model run would have to move, and a **8.5-second**
harness to measure it with.

The one methodological finding that is new, and it is about process rather than about the wheel:
**`make corner` costs 8.5 s and `make gci` costs 95 minutes, and the cheap one answered the
question the expensive one was run to answer.** §29 spent the 95 minutes extrapolating a global
functional and then mis-identified the mechanism from its convergence order. A local field
measured locally was three orders of magnitude cheaper and did not need a duality argument. Ask
which quantity carries the mechanism before building a ladder for it.

**Still not started, and still not cheap:** the filleted FEA mesh itself.
---

### 31. THE FIVE INHERITED REDS, CLEARED — `make test` reads 0 failed for the first time since §14. Two were a seed lottery, one was an absolute claim at the wrong rung, one is a real design finding now measured in full, and one stays red on purpose (2026-08-15).

`make test` had read **`5 failed`** since §19. All five were documented and deliberate:
`CONTACT_PLAN.md` registered them before that arc began and every arc since re-declared
them. That was the right call at the time and it had become a cost. §28 found that one of
the "known reds" had been mislabelled "an export-precision defect" and carried through
three plan files unchecked — it was **hiding inside the known-red count**. §29 and §30 both
turned on the claim "no new red", and establishing that claim meant opening
`CONTACT_PLAN.md` to look up which five were expected, because `5 failed` on its own says
nothing about *which* five.

A suite that reads `5 failed` on a good day cannot cheaply tell you about a sixth. This arc
closed all five.

#### STEP 0 — the baseline, and every quoted value reproduced independently

The five node ids and their values were re-measured before anything was touched, each
through its own driver rather than through pytest, so the numbers below do not depend on
the test files this arc then edited:

| red | statistic | measured | gate |
|---|---|---|---|
| `..._beam_to_wheel_ratio...` | `fea_over_beam_ratio` | 2.41276 | `> 3.0` |
| `..._correction_is_not_a_constant...` | `iso_rel_diff_ratio` | 2.16721 | `> 3.0` |
| `..._thicker_rim...` | `drops[0]` | 1.8758 | `< 2.0 < drops[0]` |
| `..._rim_band_holds...` | hub share | 0.04165644522132511 | `< 0.03` |
| `..._correction_enters_at_first_order...` | `small_load_rel_diff` | 0.0020070 | `< 1e-3` |

All five match the values this arc was scoped against, to every digit quoted. Nothing had
moved.

#### STEP 0 RECORD — 2026-08-15, the baseline this arc was measured against

```
make test:   5 failed / 469 passed in 1962.89 s (32:42)   [474 collected]
  box: 24 cores / 61 GB.  THE WALL CLOCK IS NOT CLEAN — the measurement drivers for
  steps 1-3 ran alongside it on other cores.  §28's 28:05 is the number to compare
  against, not this one.  The COUNT and the node ids are what Step 0 gates on.

GATE: PASS.  Exactly five failures, exactly the five expected node ids:
  test_gnl.py::test_the_correction_is_not_a_constant_over_the_design_space
  test_gnl.py::test_the_correction_enters_at_first_order_in_the_load
  test_wheel_fea.py::test_the_rim_band_holds_a_large_minority_of_the_compliance
  test_wheel_fea.py::test_the_beam_to_wheel_ratio_is_not_a_constant
  test_wheel_fea.py::test_a_thicker_rim_monotonically_stiffens_the_wheel

and all five values reproduced to every digit quoted — each re-measured through its own
driver rather than through pytest, so the check does not depend on the test files this
arc then edited.
```

**The one discrepancy, run down rather than waved through.** This arc was scoped against
"461 passed, 5 failed, 466 collected" and the baseline read **474** collected, 8 more. It
is fully accounted for and it is not a sixth red: §28's record is 452 collected;
`tests/test_deflection_gci.py` adds 14 and `tests/test_corner_singularity.py` adds 8, and
452 + 14 + 8 = 474. The corner test file was written at 12:39 on 2026-08-15, **after** the
466 figure was taken and before this arc began, so the scoping line counted the GCI tests
but not the corner ones. All 8 pass. Nothing moved; a count was quoted from a run that was
already stale by lunchtime.

#### ITEMS 1 AND 2 — both `ratio > 3.0` gates were a seed lottery on a statistic that cannot carry a gate

`study_wheel_fea.run_beam_blindness` and `study_gnl.run_design_space` both compute
`float(r.max() / r.min())` over the drawn rows, and both tests asserted `> 3.0` at a
hard-coded `n` and `seed=7`.

**A max/min is an estimator of the sample RANGE, which grows without bound with the number
of draws.** It is therefore not a property of the design space at all — it is a property of
the design space *and* the sample size *and* the seed. `studies/study_reds_ratio_stability.py`
measured 109 cells (20 seeds at each test's own `n`, an `n` sweep at seed 7, and the beam
study again at the default 1.2 mm wall floor as well as the 2.0 mm floor its test pins).
Every cell returned exactly the rows it requested, so none of this is draw exhaustion:

| | over 20 seeds at the test's own `n` | passes `> 3.0` |
|---|---|---|
| beam @ 2.0 floor | 1.570 – 30.129 | **7/20** |
| beam @ 1.2 floor | 1.767 – 32.655 | **11/20** |
| gnl | 2.167 – 85.501 | **11/20** |

and against `n`, at seed 7:

```
beam  n =  6, 12, 24, 48, 96  ->   2.413,  9.995, 34.968, 34.968, 48.123
gnl   n =  4,  8, 12, 16, 24  ->   2.167,  4.203,  9.026,  9.026, 51.790
```

**Precisely: non-decreasing, with plateaus — not strictly monotone**, and the distinction is
worth keeping because this arc's scoping note asked for "grows monotonically with `n`". A
plateau is what a range estimator does when the extreme rows are already in the smaller
sample and the next batch adds nothing outside them; it is the same behaviour, not a
counter-example. Every cell returned exactly the rows it requested, so no plateau here is a
draw-exhaustion artefact. **The gate the ratio was asked to carry sits inside this range**:
beam crosses 3.0 between `n=6` and `n=12`, gnl between `n=4` and `n=8`. The verdict is
chosen by the sample size.

**Seed 7 — the seed both tests hard-coded — is the low outlier in both.** Nothing about the
wheel changed, and nothing about the wheel had to.

**A second, independent demonstration that the statistic drifts on its own**, found while
checking the above: §14 measured `fea_over_beam_ratio` = **4.943** at the same 2.0 mm floor
and the same seed 7 where it now reads **2.413**. This statistic *explicitly excludes the
shipped genome*, so no promotion can account for it — a property of the gene box moves when
the **box** moves, and the box did: `R_hub`'s floor went 0.5 → 0.4 on 2026-08-11
(`wheel_fea.py:282`, BUILD_PLAN steps 5 and 6), after §14's figure was taken. Not chased
further, because it does not need to be: **a quantity that a barrier-bound edit can halve is
not one to hang a threshold on**, which is the same conclusion the seed and `n` sweeps reach.

**The conclusion the two tests exist to protect is untouched and was never in doubt.** Both
studies compute `correction_factor_is_defensible = (r.std() / r.mean() < 0.10)`, and it is
`False` in **109 of 109 cells**, with the CV never below 0.145 against a 0.10 bar. The
Stage-2.5 off-ramp — "correct the beam model with one factor and skip Stages 2 and 3" — is
closed exactly as both docstrings claim. Both tests already asserted this **first**, and it
passed on every run; only the secondary `ratio > 3.0` line failed.

**Why replacing it is §28's move and not a loosening.** `correction_factor_is_defensible`
is *defined* as `cv < 0.10`. The CV is the claim's own arithmetic; `max/min` never was. So
this is §28's pattern in its exact form — *replacing a stale constant with the arithmetic it
should always have been* — and `cv > 0.14` strictly implies `cv > 0.10`, which is the
assertion that was already in both tests and already passing. The replacement is strictly
stronger than the surviving half of the old test, at a bound 40% inside the claim's own
pre-registered bar.

**The bound is derived, not picked**: 0.14 is the CV's floor over all 109 cells (0.1450,
which occurs in the *beam* study), floored to two decimals. **One constant serves both
tests, set by the worse of the two**, so neither is fitted to its own run — the GNL test
inherits a bound it clears by 1.8× and did not choose. Both now evaluate **five seeds and
take the ensemble's worst**, seeds 0–4 taken as the first five rather than selected for
difficulty, because a single hard-coded draw is the precise defect being fixed.

**One honest limit on the claim.** Both statistics move with the sample; the difference is
which side of its bound the noise can carry it across. The ratio's verdict flips with both
seed and `n` — it crosses 3.0 from the failing side as `n` grows. The CV's verdict against
0.10 never flips, and every move with `n` is *away* from the bar. A relative-margin
comparison between the two is not available and is not claimed: on the 7 beam seeds where
`> 3.0` passed, it passed by 1.03×–10.04× while `cv > 0.14` passes those same seeds by
2.11×–4.11×. The argument made here is §28's, not a margin argument.

**The retirement is pinned executably rather than only argued.** Both report dicts still
publish `..._ratio` — it is a useful diagnostic, and §29's `criterion_met: false` is the
cautionary tale about *deleting* a number rather than scoping it — so a new test in each
file, `test_the_retired_max_min_gate_is_decided_by_the_sample_size`, asserts that the
retired gate's **verdict flips with `n` at a fixed seed** (`small < 3.0 < large`). The
reason for the retirement therefore stays measured, and anyone tempted to put a threshold
back on that number is told why not by a failing test rather than by a docstring.

#### ITEM 3 — the rim sweep asserted an absolute deflection at the least converged mesh in the tree

`test_a_thicker_rim_monotonically_stiffens_the_wheel` sweeps `rim_outer` at `smoke` and
asserted two things. **The monotonicity — the one the test is named for — passes at every
rung and always did:**

| rung | drops (mm) at 49.7 / 50.0 / 50.6 / 51.2 | monotone? | brackets 2.0? | sweep cost |
|---|---|---|---|---|
| smoke | 1.8758 1.5453 1.1823 0.9813 | **yes** | no | 0.7 s |
| coarse | 1.9798 1.6208 1.2320 1.0193 | **yes** | no | 1.9 s |
| medium | 2.0034 1.6399 1.2462 1.0308 | **yes** | **yes** | 4.7 s |

What failed was `drops[-1] < TARGET_DEFLECTION_MM < drops[0]`, an **absolute** claim
evaluated at `smoke`, which §29's ladder measured at −5.955% under SVK — more than enough
to lose a bracket whose upper edge only reaches 2.0 by `medium`.

**Moving it to `medium` was measured and rejected, and the reason is not cost** — the
medium sweep is only +4.0 s, which the plan for this arc explicitly allowed. It is that
`medium` cannot support the claim either:

| rung | drop at 49.7 | margin over 2.0 | drift from previous rung |
|---|---|---|---|
| medium | 2.0034 | +0.169% | |
| fine | 2.0134 | +0.672% | **+0.50%** |

**The margin at `medium` is smaller than the quantity's own remaining discretisation
drift.** A bound a number clears by less than its own convergence error is not a gate. So
the bracket is **recorded in the docstring with its rung attached** — the target *is*
bracketed, at `medium` and more comfortably at `fine` — and not asserted. §29 retired the
plan-level version of exactly this claim.

**It was not simply dropped.** What replaces it is the **span**, `drops[0] / drops[-1]`,
which is a ratio and therefore survives the mesh: 1.912 (smoke), 1.942 (coarse), 1.944
(medium) — 1.7% total drift against 6% on the absolute drops. Gated at 1.5, which the
measured floor clears by 27%. The effect keeps a magnitude on it without an absolute claim
at a rung that cannot carry one.

#### ITEM 4 — THE HUB COMPLIANCE SHARE. §14's unmeasured hypothesis is dead, and the answer is a design finding

This was the one live question about the wheel among the five, and §14 named the
measurement nobody had done:

> "the *direction* is surprising: thinner, floppier spokes should push compliance toward
> the spokes and the hub share DOWN. It went up. The plausible cause is `R_hub` dropping
> 1.5598 → 0.5790 — much less material at the hub junction — but that is a hypothesis and
> it has not been measured."

**`studies/study_reds_hub_share.py` measured it, and the hypothesis is not merely
falsified — it is structurally impossible.** Sweeping `R_hub` across its entire gene box
(0.4 → 4.0) on the shipped genome leaves the solved wheel **bit-identical**: hub share
`0.04165644522132511` and axle drop `1.6207901051335216` at every point, including values
the buildability barriers forbid.

The reason is three words already in this tree, at `wheel_wheel.py:44` — **"FILLETS ARE NOT
MODELLED"**. `R_hub` and `R_rim` reach the beam model, the objective and the buildability
barriers, but they never reach the mesh. **The fact was in the tree; the inference was
not**, and §14 spent its one hypothesis on a gene that cannot move the quantity. This is
§30's lesson again — the evidence was already there — and it is the third time this file
has recorded that pattern.

**What actually moves the hub share is the spoke's CURVATURE.** One-at-a-time gene swaps
from the shipped genome toward `best_solution_ga_beam.json`, the design the 3% bound was
calibrated on, at `coarse` (0.0417 → 0.0138):

| gene | shipped | ga_beam | hub | closes the gap by |
|---|---|---|---|---|
| `cy4` | 6.4375 | 29.2919 | **0.0132** | **102.4%** |
| `cy3` | 9.4191 | 24.3248 | 0.0219 | 70.9% |
| `cy1` | 8.7212 | 27.9529 | 0.0250 | 59.7% |
| `cy2` | 11.8088 | 31.7187 | 0.0255 | 57.9% |
| `t0` | 1.4738 | 2.4774 | 0.0561 | **−51.9%** |

`cy4` alone takes it under the bound. The shipped spoke is far flatter (cy 6.4–11.8 against
24–32), and a flatter spoke feeds moment into the hub junction instead of storing it in its
own bending. **§14's instinct about thickness was right in sign** — the `t0` row shows a
*thicker* root raises the hub share, so thinning lowers it — it was simply swamped by a
curvature change pulling twice as hard the other way. That is why the direction looked
surprising.

**And the bound is not a mesh artefact**, which was the other live possibility (§29/§30).
Design × mesh, hub share, five rungs (the fifth built for this measurement only, at roughly
2× `fine`, and deliberately *not* added to `wheel_wheel.CONFIGS`):

| genome | smoke | coarse | medium | fine | ultra | drift |
|---|---|---|---|---|---|---|
| shipped (`R_hub` 0.6636) | 0.0392 | 0.0417 | 0.0433 | 0.0453 | 0.0463 | **+18.3%** |
| ga_beam (`R_hub` 1.5598) | 0.0139 | 0.0138 | 0.0139 | 0.0141 | 0.0143 | +2.4% |

The ga_beam design is converged on the same ladder and passes with ~53% to spare, so the
ladder is not the problem. The shipped design is **30.5% over at the coarsest rung and
54.3% over at the finest**, and refinement makes it worse. **Discretisation cannot rescue
this bound: it is measuring the wheel, not the mesh.** Note also that the shipped genome's
share is the one that will not settle — consistent with an unfilleted re-entrant corner at
a junction that now has much less material around it, which is the same mechanism §30
measured on the rim corner.

**THE DECISION, PUT TO THE USER WITH THESE NUMBERS AND HANDED BACK — SO IT IS MADE HERE:
THE BOUND STAYS AT `0.03` AND THIS STAYS RED, AS AN ACCEPTED DEFICIT.**

The measurement rules out both readings that would justify moving the bound. `< 0.03` is
**not unreachable** — `best_solution_ga_beam.json` meets it *converged*, 0.0139–0.0143
across the whole ladder, with 53% to spare. And it is **not a mesh artefact** — the shipped
genome is 30.5% over at the *coarsest* rung, before any refinement argument begins. What
remains is a real hub-stiffness deficit in the 1.2 mm wheel, and moving the bound to
accommodate it is re-fitting a gate to the design that breached it: the move §14's rule
exists to prevent, and the one this tree has refused three times for `GATE_SMALL_LOAD_REL`.
`xfail_strict` means the question reopens by itself if the wheel ever passes.

**The successor is a design change, not a threshold one, and it is filed rather than done:**
`cy4` alone moves the share by 102% of the gap, so if hub compliance is worth constraining
it belongs in Stage 2/3's objective. Out of scope for an arc restricted to tests and their
supporting measurements.

The three options as they were put, with the numbers, for the record:

1. **Accept a real hub-stiffness regression.** The shipped wheel holds 3.2× the hub
   compliance share of the design it replaced (0.0463 vs 0.0143 at the finest rung). The
   bound stays at 3%, the test stays xfail, and this becomes a known deficit of the 1.2 mm
   wheel.
2. **Restate the bound with a rung attached.** §29's discipline applied here: the share is
   not converged for the shipped genome, so a single fixed number is the wrong shape of
   claim regardless of its value. This is the only option that is a *method* fix, but note
   it does not make the wheel pass — even `smoke` is 30.5% over.
3. **Treat the flat spoke as a design finding for the objective.** `cy4` alone moves the
   share by 102% of the gap. If hub compliance is worth constraining, it is reachable
   through the curvature genes and belongs in Stage 2/3's objective rather than in a test
   threshold.

**Resolved as 1, with 3 filed as the successor.** Option 2 was rejected as *insufficient
and slightly misleading*: restating the bound against a rung is good discipline in general,
but here it would dress a 30–54% miss as a convergence question when the ga_beam control
shows the ladder is fine and the bound is met on it. The ladder is recorded above, which is
all the rung-attachment this claim actually needs; the number itself does not move.

#### ITEM 5 — the GNL small-load gate stays red, and that is correct

`small_load_rel_diff` = **0.0020070** against `GATE_SMALL_LOAD_REL = 1e-3`, over by 2.01×.
Re-measured this arc, unchanged. The exponent assertion — the one the test is named for —
passes at **1.0393** inside (0.7, 1.4), so the SVK path is behaving; the coefficient moved,
not the exponent.

**`GATE_SMALL_LOAD_REL` was not moved, and this is the third arc to refuse.** `study_gnl.py`
records it as "written down BEFORE the study was run, per the plan's rule"; §14 refused
twice; SVK_PLAN Step 0 re-declared it. It is red because the shipped 1.2 mm wheel is ~5.5×
more geometrically nonlinear than the GA/beam design it replaced. **That is a true statement
about the wheel**, and re-fitting a pre-registered gate to the design that breached it is
the exact move the rule exists to prevent.

**The successor this actually points at, flagged and deliberately not acted on:** whether
**linear kinematics is still an acceptable default for a 1.2 mm wall at all**. §14 called
this "the most important thing §14 found". If the answer is no, the fix belongs in the
physics defaults, not in this number. Out of scope for this arc.

#### THE MECHANISM THAT MAKES `0 failed` HONEST — `xfail_strict = true`

There was no such mechanism before: `xfail` appeared nowhere in `tests/`, and
`pyproject.toml` configured neither `markers` nor `xfail_strict`. Introducing one was part
of the work, not a detail.

- **`xfail_strict = true` in `pyproject.toml`**, so the default cannot be forgotten at the
  next site. This **changes the meaning of any future bare `xfail` in this tree** and the
  setting says so in place.
- **An xfail that starts passing is a FAILURE.** That is what makes this safe rather than a
  way of hiding things: the day the wheel changes enough for one of these to pass, the suite
  reopens the question by itself. Without it the accepted finding would go stale silently —
  the precise failure mode §28 caught after it had travelled through three plan files.
- **Every xfail carries a `reason=` naming the PLAN section that decided it**, so `N failed`
  never again has to be resolved by opening another plan file.
- **No test was deleted.** §29's lesson is that a correctly-quarantined number was
  load-bearing somewhere else, and a deleted assertion cannot be found by grep.

**Both xfailed assertions were SPLIT OUT of tests that also contained passing assertions**,
which was a real defect in its own right: while the hub bound lived inside
`test_the_rim_band_holds_a_large_minority_of_the_compliance`, the rim and spoke shares —
what that test is named for — were **not being checked on any run**, and the same was true
of the exponent assertion inside the GNL test. Splitting them recovered two passing
assertions that had been dark since §14.

#### WHAT THIS ARC DID NOT DO

- **Nothing in `best_solution.json` or any exported artifact was touched.** This arc changed
  tests and their supporting measurements only.
- **No threshold was moved to fit a run that breached it.** The one new bound (`cv > 0.14`)
  is derived from a 109-cell grid and is stricter than the claim's own pre-registered bar.
- **`GATE_SMALL_LOAD_REL` stays at `1e-3`.**
- **The monotonicity and `correction_factor_is_defensible` assertions were not weakened** —
  they are the findings.
- **The hub bound `< 0.03` was not moved.** It is escalated with numbers, above.
- **Left as a loose end, deliberately:** `test_the_beam_to_wheel_ratio_is_not_a_constant`
  still pins the wall floor at 2.0 mm, and the only stated reason for that pin was that the
  now-retired `> 3.0` margin had been calibrated there. §14 called re-deriving Gate 1 at the
  1.2 floor "a real piece of work and a judgement about Gate 1". **The measurement is now
  done** — the 1.2 box's CV floor is 0.1948 over the same 20 seeds, comfortably above the
  same 0.14 gate, so dropping the pin would change no verdict — but the judgement is still a
  human's and the pin was left alone.
- **Not started:** the filleted FEA mesh, still the top open item, and now with one more
  reason behind it — `R_hub` and `R_rim` are two of fourteen genes that the FEA cannot see
  at all.

#### SUITE RECORD — 2026-08-15/16. THE ARC'S GOAL, MET.

```
make test:   476 passed, 2 xfailed in 1726.40 s (28:46)   [478 collected]   exit 0
  box: 24 cores / 61 GB, uncontended.

  0 FAILED.  First time since §14.
```

**The arithmetic closes against Step 0 exactly, which is the check that the count means what
it says:**

| | collected | passed | failed | xfailed |
|---|---|---|---|---|
| Step 0 baseline | 474 | 469 | **5** | 0 |
| after the arc | 478 | 476 | **0** | 2 |

`478 − 474 = 4`, the four tests this arc added: the two `test_the_retired_max_min_gate_is_
decided_by_the_sample_size` pins, and the two assertions split out of tests that also held
passing ones. Of the five reds, **three now pass** and **two are strict `xfail`**. So
`469 + 3 + 4 = 476`. Nothing is unaccounted for and nothing was skipped or deselected.

**Cost: about +41 s**, against §28's 28:05 on a comparable uncontended box. That is what the
five-seed ensembles and the two instability pins buy, on a suite that runs ~29 minutes — and
the Step-1 replacement was checked against `medium`-rung alternatives before being chosen, so
this is the priced option rather than the first one that worked.

**What `0 failed` now costs to break.** `xfail_strict = true` means an xfail that starts
passing is a FAILURE, verified by probe (`XPASS(strict)`). So neither of the two accepted
findings can go stale silently, and any red from here is genuinely new — which was the whole
point of the arc.

---

### 32. LINEAR KINEMATICS IS NOT AN ACCEPTABLE DEFAULT FOR SEARCH. §14's "most important thing" is decided, seventeen sections after it was raised — and the shipped wheel was never at risk (2026-08-16).

Working notes in `KINEMATICS_PLAN.md`, three steps, each with its own record. This is the
summary; that file and `studies/study_kinematics_rank.json` are the evidence.

#### THE FEAR THIS ARC INHERITED WAS ALREADY OUT OF DATE, AND CHECKING THAT WAS STEP 0

`KINEMATICS_PLAN.md` opened by asserting, twice, that "nothing in the Stage-3 path ever
overrode" the linear default. **False since 2026-08-11.** `best_solution.json`'s own `search`
block reads `"kinematics": "svk"`; `svk-medium`, `buildcap` and `knee` all pass the flag
explicitly; §16, §19 and §26 were all SVK descents. §14's sentence was correct **when §14 wrote
it** and the SVK arc fixed the descent path without the successor's framing ever being updated.

**So the expensive fear is answered and the answer is good news: the shipped wheel was not
optimised on the wrong physics.** What was left is narrower — is the DEFAULT safe for anything
that does not pass the flag — and that is what this arc measured.

#### THE CRITERION WAS REGISTERED BEFORE THE MEASUREMENT, AND IT WAS REGISTERED KNOWING IT WOULD PROBABLY FAIL

The plan's original off-ramp — "under 2% at service load" — is failed by **11.4×** (22.75%
against 2.00%, reproduced this arc to every digit: `0.22748609045833867 /
0.002006986301629654 / 1.0393142889173537`). It was replaced in advance by the criterion the
plan itself proposed, made precise: **linear is acceptable for search iff it SELECTS what SVK
selects.** R1 argmin identity (binary, primary), R2 Spearman ρ ≥ 0.90 on the feasible subset,
R3 gradient cosine ≥ 0.90 in normalized gene space.

**Three artifacts already in the tree were read before the criterion was written, two of them
showing the inversion R1 tests for, and `KINEMATICS_PLAN.md` Step 0c says so in place.** A
pre-registration that hides what its author already knew is worth nothing; R1 was made binary
precisely so there would be nothing in it to loosen afterwards.

#### THE MEASUREMENT — `make kinrank`, 36 genomes, both kinematics, 3549 s

| | R1 argmin | R2 Spearman ρ | R3 cosine |
|---|---|---|---|
| bar | identical | ≥ +0.90 | ≥ +0.90 |
| **measured** | `minwall 0.8` **vs** `margin probe` | **−0.8303** feasible / +0.6914 full | **−0.5437** at `350f4c7` |
| | **FAIL** | **FAIL** | **FAIL** |

**The driver reproduces three independently-recorded numbers first.** §8's `minwall 1.4` and
`1.8` linear losses to four decimals (35.3760, 43.9892) and all five of its axle drops to every
digit; `study_svk_knee_coarse.json`'s linear AND SVK columns for both `e126cc3` (84.2905 /
33.6859) and `09e8188` (97.5232 / 32.9769). §8's thinner arms differ in the fourth digit and
that is diagnostic, not a miss: those sit above §23's 0.80 knee and carry a `stress_margin`
term §8's objective did not have, while the two that sit below it reproduce exactly.

**The cleanest cell is free of the selection effect that makes ρ = −0.83 look worse than it
should.** The feasible pool is a set of optima found by the two models, and it splits by
family — each model rates its own optimizer's output best — so ρ = −0.83 is a statement about
the designs this project produced, not about random designs. §8's minwall ladder has no such
problem: eight arms, one optimizer, one model, one start, only the wall floor differing.

| floor | 0.8 | 1.0 | 1.2 | 1.4 | 1.6 | 1.8 | 2.0 | 2.2 |
|---|---|---|---|---|---|---|---|---|
| linear loss | **30.19** | 31.92 | 32.53 | 35.38 | 39.47 | 43.99 | 49.73 | 70.57 |
| SVK loss | **274.59** | 155.50 | 119.32 | 79.50 | 64.39 | **59.01** | 60.30 | 78.91 |
| linear rank | **1** | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| SVK rank | **8** | 7 | 6 | 5 | 3 | **1** | 2 | 4 |

**ρ = −0.8333. Linear's best arm is SVK's worst of eight.** Linear's loss is monotone in the
floor — thinner always better, which is where §8's "as thin as the process can print" came
from — and SVK's has an interior optimum at 1.8 mm.

**One term does all of it.** On `minwall 0.8`, same genome and same mesh: `deflection` goes
**0.0005 → 237.9506** while `mass` (29.5160) and `smoothness` (0.5828) are bit-identical. The
linear descent hit the 2.0 mm target essentially exactly at every floor (1.9924–2.0004); under
SVK those same designs land at 2.1158–2.6170, worst at the thin end, because a thinner wall is
more geometrically nonlinear — and `deflection` is quadratic in the error.

**§8's loss column is therefore a linear-only result.** Its *mass* interval table stands —
mass does not touch the FEA — but the conclusion "the wheel wants as thin as the process can
print in between" is not an SVK statement. **What is NOT claimed:** that an SVK descent at
0.8 mm would fail. These are linear optima re-scored, not SVK optima. Re-deriving the floor
economics under SVK is a successor, filed below.

#### R3, READ HONESTLY: the norm carries it, not the cosine

Three of four probes clear +0.90 (`09e8188` +0.9885, `e126cc3` +0.9735, `36aed36` +0.9997 at
1.3°). The failure is at `350f4c7`, and **the cosine there is ill-conditioned rather than
damning**: the linear gradient norm is **52.42** — it is a linear stationary point, being
exactly the genome a linear descent returned and §13 promoted — so its direction comes from a
nearly-zero vector. **What carries the finding is the ratio: ‖g‖ SVK is 9014.53, 172× larger,
with 11 of 14 genes wanting the opposite sign.** Linear reports "converged" where the real
physics is steeply downhill. The mirror image holds at `09e8188`, found under SVK, whose SVK
gradient is the small one (471 against 4843). The `36aed36` control — correction 3.95% —
agrees to 1.3°, so the disagreement tracks the nonlinearity and is not a solver artefact.

#### THE ROOT CAUSE HAD BEEN MEASURED SINCE M5 AND NOBODY DREW THE RANKING CONCLUSION

A constant 22.75% offset would cancel in a ranking and this arc would have closed cheaply.
It is not constant. Re-measured this arc on the shipped genome: at MATCHED deflection the
correction spans **5.70%–48.54%, a factor of 8.5, cv 0.51** against `study_gnl`'s own 0.10 bar,
and `correction_factor_is_defensible` is **NO**. The mesh ladder is converged (22.749 / 23.160
/ 23.253% at smoke / coarse / medium) and Newton is healthy (continuation spread 1.71e-14,
observed order 4.46), so this is the wheel and not the solve.

`study_gnl.run_design_space`'s docstring has called this "*** THE M5 HEADLINE ***" since M5,
and the off-ramp it gates has always needed BOTH "small enough to ignore" AND "constant enough
to correct once". The second half was recorded as failing and never connected to whether a
linear descent could rank. **Third time this file has recorded that pattern** — §30 and §31
item 4 are the other two.

#### THE DECISION, MADE HERE RATHER THAN ESCALATED

§14 closed item 4a with "*That is a scope decision and it is a human's*", and it has been
carried unacted-on through seventeen sections. **`wheel_stage3.py`'s `--kinematics` default is
changed `linear` → `svk`.** It costs **1.49×** (median of the per-genome paired ratio over the
same 36 genomes on shared meshes; linear 34.2 s median against SVK 51.9 s) — not the multiple
Step 2 budgeted for, because the Newton loop warm-starts from the linear solution and
`study_gnl`'s G3 measures 8 → 5 iterations.

**Four live recipes took the old default and now behave differently: `stage3`, `prod9`,
`prod10`, `minwall-%`.** Nothing that ships changes — every recipe that produced a promoted
genome already passed the flag.

**What was deliberately NOT changed is the load-bearing half:**

- **`wheel_fem`'s kernel defaults stay `linear`.** A REPORTING question this arc did not
  measure, reaching ~470 tests and **11 study drivers that never mention `kinematics` at all**.
  Filed, not done on the strength of a ranking result.
- **`descend()`'s library default stays `linear`**, so a record written without the kwarg keeps
  meaning what every pre-§15 record meant. The CLI and the library now differ **on purpose**
  and the code says so in place.
- **`GATE_SMALL_LOAD_REL` is not moved. Fourth arc to refuse.** It stays red — and it is a
  KERNEL-default measurement, so no search-side change could have turned it green. Its
  `xfail` `reason=` and docstring are rewritten to say the question it was waiting on is now
  answered, which `xfail_strict` could not have caught (it would still fail, for a stale
  reason).
- **Nothing promoted, nothing re-descended, no exported artifact touched.**
- **Stage 2 cannot be affected**: `wheel_fea.evaluate_design` scores through
  `generalized_spoke_mechanics`, a Castigliano beam model with no FEA and no strain measure.

#### TWO DEFECTS FOUND ON THE WAY. RECORDED, NOT FIXED — §15's precedent

1. **`studies/study_gnl.json` DESCRIBES THE WRONG WHEEL.** Dated 2026-08-03, `settings.genome`
   says `best_solution.json`, and its service row is `linear 1.666633 → svk 1.732517,
   +3.953%` — `36aed36`'s numbers to six decimals. **The M5 artifact publishes the GA/beam
   wheel's 3.95% under the shipped wheel's name: the exact number this arc turns on, low by
   5.9×.** Identical to §15's `study_gradient.json` finding, in the sibling artifact, now three
   promotions old.
2. **`make studies` CANNOT COMPLETE and has not been run since 2026-08-03.** Verified: running
   `study_gnl.py` to a scratch path exits **1**, because `run_load_ladder`'s `pass` needs
   `small_load_rel_diff < 1e-3` and the shipped wheel reads 0.0020070. `make` stops at line 5
   of 9, so **`study_contact`, `study_gradient`, `study_objective` and `study_stage3` are
   unreachable by that recipe** — which is also why §15's stale `study_gradient.json` was never
   refreshed. Every artifact the recipe writes carries the same 2026-08-03 date, *including the
   four drivers before `study_gnl`*, which is what says it has not run at all rather than
   aborting partway.

#### The successors, ranked

1. **`make studies` is a dead gate — decide what M5's `pass` means.** It has been unrunnable
   since the 1.2 mm wheel was promoted, and it takes four other drivers down with it. The gate
   is correct (the wheel really does breach it) and the recipe is wrong to treat a
   characterisation finding as a hard stop. This is §25's "the gate had been dead since §18"
   in a second place, and it is cheap.
2. **Refresh `study_gnl.json` and `study_gradient.json`.** Both name `best_solution.json` and
   describe a wheel promoted out of that file three times ago. Blocked behind 1 for `make
   studies`, but each driver runs standalone (`study_gnl.py` is 303.5 s at `coarse`).
3. **Re-derive §8's wall-floor economics under SVK.** §8's mass table stands; its loss column
   and "as thin as the process can print" do not. Eight `minwall-%` arms now default to SVK,
   so this is `make minwall-0.8 … minwall-2.2` and nothing else — at 1.49×.
4. **The kernel default** (`wheel_fem.solve_wheel` / `wheel_problem` /
   `wheel_contact_problem`), and the 11 drivers that take it silently. Needs its own arc and
   its own criterion: this one measured search, not reporting.
5. `FILLET_PLAN.md` is now unblocked — see the ranking note below.

### 33. `make studies` WAS DEAD AT LINE 4, NOT LINE 5. Two gates were conflating "the solver is broken" with "the wheel is interesting", and §32's successors 1 AND 2 are both closed (2026-08-16).

§32 filed successor 1 as *"decide what M5's `pass` means"*. Executing it settled that question —
and found the premise it rested on was **half wrong in a way that mattered**.

#### THE PREMISE WAS WRONG: THE RECIPE NEVER REACHED `study_gnl` AT ALL

§32 recorded: *"`make` stops at line 5 of 9, so `study_contact`, `study_gradient`,
`study_objective` and `study_stage3` are unreachable by that recipe."* It verified
`study_gnl.py` **standalone** — that part is true, it did exit 1 — and **inferred** the
recipe's stopping point from it. Run for real, `make studies` died at **line 4 of 9**,
`study_wheel_fea` (M4), and `study_gnl` was never reached by the recipe at all.

**Fixing M5 alone would have unblocked nothing.** This is the fourth time this file has
recorded the same shape — §30, §31 item 4, §32 itself — and it is the standing lesson:
*a driver verified standalone does not tell you where the recipe stops.*

#### BOTH GATES WERE RED FOR THE SAME CATEGORY ERROR, IN TWO DIFFERENT FLAVOURS

A `make` recipe sees only an exit status, so it cannot tell these apart:

| | the claim | if red |
|---|---|---|
| **SOLVER** | an identity is violated, a mesh is degenerate | every number downstream is void — stop |
| **WHEEL** | a true, reproduced, deliberately-held fact about the design | report it — do not stop |

**M5 (`study_gnl`), measured.** The sole cause of exit 1 was `small_load_rel_diff`
**0.0020369** against the 1e-3 pre-registered gate. Everything else was green and always
had been: `all_softer` True, `fitted_exponent` **1.0413** inside (0.7, 1.4), G1 frame
indifference PASS, G3 Newton health PASS. The one red is a statement about a 1.2 mm wall.

**M4 (`study_wheel_fea`), measured.** The sole cause was `decision_robust`, computed as
`max(rim_share) - min(rim_share) < 0.01` over **every rung of the ladder**:

```
    smoke 0.301560  ->  coarse 0.316117  ->  medium 0.320262  ->  fine 0.320514
    deltas          +0.014558        +0.004144        +0.000252
```

That is textbook convergence, settled by `medium` to 2.5e-4. The all-rung span is
**0.018954** (FAIL); over the converged rungs it is **0.004397** (PASS). **The gate was
reading "`smoke` is coarse" — which nobody disputes — as "the conclusion is not robust".**
A max−min span is not a convergence test: it is dominated by the coarsest rung, prepending a
coarser one can only make it worse, and no amount of refinement can ever fix it.

#### THE TWO DECISIONS, MADE HERE

1. **`study_gnl`'s exit code is built from SOLVER CORRECTNESS ONLY.** New `solver_is_correct`
   (G1 ∧ G3 ∧ G2's solver half) drives the exit; `wheel_is_in_the_small_strain_regime`
   carries the breach and is reported, loudly, in the verdict block. The precedent is exact
   and one layer down: **§31 moved this very gate out of
   `test_the_correction_enters_at_first_order_in_the_load` into its own xfail** because,
   while it lived there, *it took a passing assertion down with it on every run.* The tests
   made the split; the exit code never followed.
2. **`decision_robust` is computed on the converged tail, not the whole span.** Its sibling
   `criterion_met` **in the same function** already Richardson-extrapolates the FINEST PAIR;
   the two now agree about where a convergence claim may be read from.

**NEITHER THRESHOLD WAS MOVED.** `GATE_SMALL_LOAD_REL` stays at 1e-3 — **fifth arc to
refuse** — and M4's 0.01 stays at 0.01, with the raw all-rung span still in the report. What
changed is *which half of a report a process exits on* and *which rungs a convergence
statistic may be computed over*. Neither is a gate re-fitted to a design that breached it.

#### A THIRD DEFECT, FOUND BY ACTUALLY RUNNING THE RECIPE — AND IT HAD HIDDEN BEHIND THE OTHER TWO

**Seven of the nine drivers wrote artifacts to the CURRENT DIRECTORY instead of `studies/`.**
`study_beam_agreement.py` was the only one of the nine with no `HERE` at all, so both its
`.json` and `.jpg` went to the repo root; six more (`study_wheel_fea`, `study_wheel_mesh`,
`study_gnl`, `study_contact`, `study_objective`, `study_gradient`) resolved their `.json`
against `HERE` correctly but passed a **bare basename** to `_plot`, so every committed `.jpg`
had been stale since 2026-08-03 while fresh copies piled up at the repo root. All seven now
resolve against `HERE`; an absolute `--out` still wins. **This was invisible for exactly the
reason it was worth finding: the recipe never ran twice.**

#### SUCCESSOR 2 IS CLOSED FOR FREE

`studies/study_gnl.json` now describes the wheel that ships. §32's defect 1 — *"the M5
artifact publishes the GA/beam wheel's 3.95% under the shipped wheel's name, low by 5.9×"* —
is gone: `service_rel_diff` reads **0.23160** on `09e8188`, and the independently-derived
`iso_rel_diff_cv` **0.5062** reproduces §32's 0.51 to two digits, which is the check that
none of this arc's edits perturbed the physics. The driver's own docstring headline is
annotated in place rather than overwritten, per this file's rule about records.

#### WHAT MUST NOT BE READ INTO THIS

- **Nothing promoted, nothing re-descended, no exported artifact touched.**
- **`wheel_fem`'s kernel defaults are still `linear`** — untouched, still §32 successor 4.
- **`GATE_SMALL_LOAD_REL` is still breached and still red**, pinned by a strict xfail that
  reopens itself if the wheel ever passes it. §33 did not make the wheel less nonlinear.
- **M4's 0.5% mesh criterion is still NOT MET** and is still reported as such.

#### THE RECIPE IS GREEN. ALL NINE DRIVERS, ALL NINE ARTIFACTS, FOR THE FIRST TIME SINCE 2026-08-03

```
  driver                  verdict   elapsed        artifact
  study_mesh_quality      PASS         ~20 s       refreshed
  study_wheel_mesh        PASS          4.9 s      refreshed
  study_beam_agreement    PASS          4.5 s      refreshed
  study_wheel_fea         PASS         82.8 s      refreshed   <- was the line-4 blocker
  study_gnl               PASS        148.5 s      refreshed   <- was believed the blocker
  study_contact           PASS       1154.4 s      refreshed   <- never reached before
  study_gradient          PASS        978.5 s      refreshed   <- never reached before
  study_objective         PASS       4384.2 s      refreshed   <- never reached before
  study_stage3            PASS       9976.2 s      refreshed   <- never reached before
```

**The four drivers that had never been reached cost 4 h 15 m between them, and every one of
them PASSES.** Nothing was wrong with them; they were simply behind a gate that exited on a
characterisation finding. `study_stage3` alone is 2 h 46 m, which is the practical reason this
recipe is not a CI gate and why a decade of staleness went unnoticed.

**All four name `best_solution.json` and describe `09e8188`**, so §32's defect 1 is closed in
both artifacts it named, not just `study_gnl.json`.

**`make test` READS 476 passed, 2 xfailed, 0 failed** (28 m 32 s), and the two xfails are the
two deliberately-held reds — `test_the_gnl_correction_is_small_at_one_percent_of_service_load`
(`GATE_SMALL_LOAD_REL`, fifth arc to refuse) and
`test_the_hub_junction_holds_under_three_percent_of_the_compliance` (§31 item 4's accepted
deficit, now `HUBSHARE_PLAN.md`). **Both still xfail rather than XPASS, which is this arc's
own control**: `xfail_strict = true`, so if changing `study_gnl`'s EXIT CODE had leaked into
the GATE, that test would have gone green and been reported as a FAILURE. It did not. The
exit-code semantics moved; the gate did not.

*Scope note, stated because it matters:* drivers 1–6 ran inside a single `make studies`
invocation. Drivers 7–9 were re-run in the same order and environment after a session teardown
killed the first attempt mid-`study_gradient` — not after any gate failure. The `make` glue
itself is sequential invocation and nothing more, but a single nine-driver invocation has not
been performed end to end.

#### The successors, ranked

1. **The `.jpg` half of every committed study artifact is one run old at best.** The JSONs
   for drivers 1–6 are current; their plots are current only from this arc forward.
2. **§32's successors 3 and 4 are untouched** — §8's wall-floor economics under SVK, and the
   kernel default with its 11 silent drivers. Both still stand as filed.
3. **`FILLET_PLAN.md` (arc 2) — its prerequisite is now CHECKED, and the answer is a
   caveat, not a blocker.** PLAN's ranking note asked whether the study drivers silently take
   `linear`. `studies/study_corner_singularity.py` — the driver Steps 0 and 2 both rest on —
   **mentions `kinematics` zero times** and calls `fem.solve_wheel(mesh)` bare at line 167, so
   it takes the kernel default. The corner **exponents** survive this (Williams' wedge
   solution is itself linear-elastic, and the driver reproduces the 360° crack at exactly
   0.5), but any claim it makes about stress **magnitudes** or about `R_hub`/`R_rim`
   sensitivity inherits the linear kernel and must say so.

### 34. THE FILLET ARC'S RISK WAS MIS-RANKED IN BOTH DIRECTIONS: the seam merge survives untouched, HALF THE CORNERS ARE NOT THE PART'S, and what actually blocks it is that the spoke block is ruled (2026-08-17).

`FILLET_PLAN.md` (arc 2) has been "the top open item" since §29 and was deferred five
times on one stated risk: *"a fillet at the hub and rim junctions changes the block
topology at both seams, which is the part that is not cheap."* **Step 1 is now answered,
and that sentence is wrong in the direction that mattered.** Working record in
`FILLET_PLAN.md`, STEP 1 RECORD PARTS 2 and 3.

#### FINDING 1 — `P_c` IS A MESH ARTEFACT. THE PART DOES NOT HAVE IT.

The FEA mesh and the exported solid are **different geometries at the junctions**.
`wheel_wheel.sector_blocks` stops the spoke ON the ring
circle and closes it with a half end cap. `wheel_step_export._embed` drives both flanks
straight THROUGH the ring — into the hub disk at r = 12.20, past the OD at r = 50.25 —
so the shipped solid has **no end cap at either ring**, and its two corners per junction
are both flank-crossings.

Reimplementing `_embed` in numpy (20 lines, no OCC) and walking the outline gives two
crossings per spoke per ring — **24 and 24, exactly the manifest's `hub_edges` and
`rim_edges`**, which is what says the reconstruction is the part:

```
  ring  corner              theta (deg)   wedge (deg)   spoke-side leg
  hub   MESH P_t             +6.117810      321.29      41.1182 mm  flank
  hub   PART top_flank       +6.117480      322.02      41.1182 mm  flank   <- 0.073 um apart
  hub   MESH P_c             +0.000000      297.33       0.7369 mm  END CAP
  hub   PART bot_flank       -1.526730      268.47      41.1182 mm  flank   <- 28.7 deg apart
  rim   MESH P_t             +1.360780      321.29      41.1182 mm  flank
  rim   PART top_flank       +1.360760      321.07      41.1182 mm  flank   <- 17 nm apart
  rim   MESH P_c             +0.000000      307.39       0.7156 mm  END CAP
  rim   PART bot_flank       -1.315860      219.90      41.1182 mm  flank   <- 87.5 deg apart
```

These wedges reproduce §30's independently measured mesh wedges to within 0.6 deg, and
the manifest's `worst_wedge_deg` (hub 322.0, rim 320.0) belongs to the `P_t` family.

**So `P_t` is real and correctly placed; `P_c` is not a corner of the shipped part.**
Filleting `P_c` would move the model AWAY from the part, and cannot be done at the
shipped radii anyway — its cap leg is `t/2` (0.737 / 0.716 mm) against tangent lengths
of 1.090 and 6.069 mm.

**CORRECTION, SAME DAY, AND IT MATTERS.** The first draft of this finding said "nothing
in the tree said so". **That was wrong** — `wheel_wheel.py`'s module docstring, under
*WHAT IS AND IS NOT MODELLED*, says it outright: *"`wheel_step_export._embed` IS NOT
REPRODUCED, and it does matter: it adds 3.03 mm2 per spoke inside the annulus, so this
mesh models ~1.4% less material than the shipped part, all of it at the junctions where
it acts as a gusset. **That is a real modelling difference and it is deliberate.**"* The
difference is documented, quantified and justified, and I should have read that docstring
before writing the sentence. Recorded rather than quietly deleted, per §15.

**What the recorded justification does NOT cover is Finding 4.** It weighs AREA and
STIFFNESS — *"its stiffness consequence is an M4 sensitivity run, not a guess made here"*
— and the consequence nobody wrote down is that the idealisation manufactures the corner
that carries the wheel's global peak stress. That half is new and it survives the
correction.

#### FINDING 2 — THE SEAM MERGE WAS NEVER THE PROBLEM

`build_wheel` and `sector_blocks` now take an opt-in `fillet=`. The plan's option 2 —
move the four-block corner from `P_t` to the tangent point `B` on the ring circle —
needs **no code at all in six of the seven blocks**, because the junctions and both ring
blocks already derive `theta_t` from the spoke's own end row. All eight seam entries
keep their counts and their pairings, `_seam_table` is untouched, and the six non-spoke
blocks come out with **zero mixed-sign cells**.

#### FINDING 3 — THE SPOKE BLOCK IS WHAT FAILS, AND IT IS A CONSTRUCTION LIMIT

A fillet in a **~39 degree notch** has `T = R/tan(void/2) = 2.847 R`, so the corner must
travel `T`. `T/t` against the wall it attaches to is **1.28 at the hub and 5.84 at the
rim**, and stays above 0.77 everywhere in the `R` box. The spoke's end cross-section then
runs from the moved corner to an unmoved far-flank point: 1.452 -> 2.759 mm at the hub,
1.417 -> **8.596 mm** at the rim, on a 1.43 mm wall. The corner's interior angle collapses
from ~89 deg to 3.60 / 8.52 and the cross product flips sign; `_orient_elements` stops
the build at 12 folded elements, one per sector.

**Swept, the construction survives 0.20 mm at `coarse` and 0.10 at `medium`** — 3x to
30x below what ships. **It TIGHTENS under refinement, which is what says the limit is the
ruled interior of the spoke block and not the geometry**; a fillet that genuinely did not
fit would fail mesh-independently. Do not quote 0.2 mm as a physical bound.

#### CONTROLS

`fillet=None` short-circuits: **bit-identical** at `smoke`/`coarse`/`medium` (`max|dx| =
0`, seam error unchanged at 3.20e-14 mm). `fillet=(0,0)` goes through the full Coons
rebuild and agrees to **2.842e-14 mm**, which independently confirms that `sample` is
affine in eta and therefore that the unfilleted spoke already *is* the Coons patch of its
own boundary curves.

**`make test` after the change reads 476 passed, 2 xfailed, 0 failed, exit 0 (28 m 36 s)**
— the same counts and the same two deliberately-held xfails as §33's baseline taken
before it (476/2/0, 28 m 32 s). `build_wheel` is what most of the suite runs through, so
this is the check that the new parameter is genuinely inert.

#### WHAT THIS COST, AND WHAT IT BOUGHT

One afternoon, against an arc budgeted in weeks. It retired a risk that had been steering
the ranking since §29 — and the lesson generalises: **check the cheap half of an
"expensive" claim first, because it may be the half that is wrong.**

#### FINDING 4 — THE WHEEL'S GLOBAL PEAK STRESS LIVES ON THE ARTEFACT CORNER

Checked before ranking a fillet block, because §17's lesson is to ask whether the target
binds. **It does not.** `study_corner_singularity.json` reports `global_max_vm_mpa`
equal to `rim:P_c`'s `peak_vm_mpa` at **every rung** — 61.92, 99.13, 126.25, 150.59 —
which is two different statistics agreeing four times and therefore worth checking
directly rather than reading as coincidence.

Located directly, by taking `argmax` over the whole Gauss-point field and measuring to
all twelve rotational copies of each corner:

```
  cfg      kinematics   global max vM     distance from the peak to...
                                          rim:P_c    rim:P_t    hub:P_c    hub:P_t
  coarse   kernel(lin)   99.1302 MPa     0.016190   1.151194   35.783821  35.881663
  coarse   svk           99.4487 MPa     0.016190   1.151194   35.783821  35.881663
  medium   kernel(lin)  126.2484 MPa     0.010745   1.151901   35.789256  35.887123
  medium   svk          126.3069 MPa     0.010745   1.151901   35.789256  35.887123
```

**The peak is 11-16 um from `rim:P_c` — it is ON that corner — and SVK does not move it**
(same location to the digit, 0.3% in magnitude). At `fine` the corner ranking is
`rim:P_c` 150.59 > `hub:P_c` 120.92 > `hub:P_t` 96.22 > `rim:P_t` 75.40: **both artefact
corners outrank both real ones.**

**So Step 2's stated success condition cannot be met by filleting `P_t`.** The plan says
*"the peak stress stops diverging — this is the one that unlocks quoting a max"*, and the
peak is not at `P_t`. A fillet block would round the corners that are real and leave the
corner that carries the maximum untouched.

*Stated because it is the honest qualifier:* `P_t` diverges FASTER (5.14x / 5.58x against
`P_c`'s 2.56x / 2.43x), because its wedge is larger. At `fine` the two rim corners are a
factor 2 apart and closing, so `P_t` would overtake somewhere beyond the current ladder.
**On every mesh anyone actually runs, `P_c` dominates**; asymptotically `P_t` does. Both
have to go, and the artefact goes first because it should never have been there.

#### The successors, RE-RANKED after Finding 4

1. **Stop capping the spoke at the ring** (`UNCAP_PLAN.md`). It was #3 on the first draft
   and Finding 4 moved it to #1: it removes both `P_c` corners outright, including the one
   carrying the measured global maximum, and it is the only item here that makes the mesh
   agree with the part. It also plausibly makes the fillet *easier* — with both corners
   becoming flank/ring crossings with 41 mm legs, the end cross-section that folded the
   spoke block no longer exists. **Three things measured about it before the plan was
   written, because the docstring's justification had to be taken seriously:**

   - *The cap is FORCED by the "cut at the ring circle" scheme.* The bottom flank never
     crosses either ring circle along the spline — closest approach **0.6589 mm** at the
     hub and **0.5656 mm** at the rim. There is no second corner to move `P_c` onto, so
     the cap is not a choice made at the junction; it is the only way to close the block.
   - *The docstring's stated reason for "no smooth alternative" is FALSE on the shipped
     genome.* It says the bottom flank's backward tangent *"MISSES the hub circle
     entirely (its closest approach exceeds 12.7)"*. Measured, that closest approach is
     **12.0771 mm** — inside 12.700, so it reaches. A straight tangent extension crosses
     r=12.700 after 1.7815 mm and r=48.500 after 0.9190 mm, in closed form: **smooth,
     differentiable, no argmax search**, so the M7 objection that rules out reproducing
     `_embed` does not rule this out.
   - *But a tangent extension is not obviously BETTER.* It puts the hub's second corner at
     theta = **-8.73684 deg** where the part's is at **-1.52673** — 7.21 deg off, against
     the end cap's 1.53 deg. **Smooth-alternative-exists is not the same as
     smooth-alternative-is-an-improvement**, and that is the first thing `UNCAP_PLAN`
     Step 1 has to measure rather than assume.

   **Not free either way:** it moves welded volume, and therefore mass, hub share, and
   every constant calibrated against them.
2. **A dedicated fillet block** (`FILLET_PLAN` option 1) over the curvilinear triangle
   `A - P_t - B`, with its own seam entries — for `P_t`, which is real. Ranked below 1
   now: it is worth building, but on its own it does not deliver the arc's headline.
3. **A generated spoke block** — transfinite smoothing with a boundary correction that
   decays away from the junction. Fixes the ruled block's other weakness: any end
   correction currently propagates linearly down all 41 mm. May be subsumed by 1.
4. **§32's successors 3 and 4** are still untouched — §8's wall-floor economics under
   SVK, and the kernel default with its 11 silent drivers.

**Step 2 is not reachable** until 1 exists, and Finding 4 is why: before Finding 4 the
blocker was "no filleted mesh assembles"; after it, the blocker is that filleting the
corners we *can* fillet would not move the number Step 2 is defined by.

`study_mesh_quality.py` at the fillet and `test_axle_drop_is_exactly_12_fold_periodic` on
a filleted mesh were deliberately NOT run: both need a mesh that assembles, and running
them on the 0.2 mm mesh that does assemble would measure a fillet 3-15x smaller than the
one that ships.

### 35. UNCAP STEP 1 IS A GO, AND BY TWO ORDERS OF MAGNITUDE: `_embed`'s DIRECTION IS SMOOTH — ONLY ITS ARGMAX IS NOT (2026-08-17).

§34 promoted "stop capping the spoke at the ring" to successor #1 and `UNCAP_PLAN.md`
planned it with an explicit expectation that **Step 1 would say no**, on the strength of
one number: the bottom flank's own tangent lands 7.21 deg from the part's hub corner
where the end cap lands 1.53. **That number was right and the conclusion drawn from it
was wrong** — it tested the wrong extension direction.

#### THE INSTRUMENT

`studies/study_junction_agreement.py` (`make junction`, ~4 s, **geometry only, no field
solved**). It reproduces `_embed` in numpy — so it runs in the OPT env with no OCC — walks
the part's outline for every ring crossing, and tabulates them beside the mesh's corners
with wedge angles and Williams lambda. **Self-checked two ways:** the crossing count comes
out 24 hub and 24 rim per wheel against the shipped manifest's `hub_edges`/`rim_edges`,
and the mesh's four wedges reproduce §30's independently measured ones (summed from
incident element angles on `fine`) to within 0.8 deg — 321.14/297.18/320.55/307.43 against
321.10/296.75/321.33/307.94.

#### THE INSIGHT

`_embed`'s non-differentiability is **entirely in its argmax**: a 20001-point length scan
and, at the rim, a 21-point blend scan. **Neither is needed.** The length is unnecessary
because the ring crossing is a closed-form line/circle intersection, and each of `_embed`'s
two extreme *directions* is a smooth function of the genes. And `_embed` says which to
expect: its hub branch is hard-coded to `(1.0,)` — *"Radial-inward always reaches, so the
search below is a single step at the hub"* — while its rim branch searches upward from 0.0.

```
  ring   candidate          wedge err (deg)   theta err (deg)   lambda err
  hub    end_cap (today)         28.71            1.5267         0.0329
  hub    own_tangent             70.45            7.2101         0.2866
  hub    shared_tangent          67.29            6.3115         0.2631
  hub    radial  (blend 1.0)      0.00            0.0000         0.0000   <- EXACT
  rim    end_cap (today)         87.53            1.3159         0.1896
  rim    own_tangent              1.49            0.0658         0.0075
  rim    shared_tangent (0.0)     0.02            0.0201         0.0001   <- best
  rim    radial                  50.61            0.8010         0.1540
```

**Across eight genomes spanning the whole design history** — shipped, the GA/beam
ancestor, both ends of the min-wall sweep, the knee, the SVK run, a buildcap run and
`defect5_step100` — `radial` is **exact (0.00 deg) at the hub on every one**, and a tangent
wins at the rim on every one, within 1 deg. The end cap's error over the same set is
4.81-50.90 deg at the hub and **84.00-88.37 deg at the rim.**

#### WHY THIS EXPLAINS §34 FINDING 4

The rim column is the headline: **the end cap is wrong by ~87 deg of wedge at the rim on
every genome tested.** In Williams terms the mesh models lambda = 0.5081 where the part has
0.6977 — **stress ~ r^-0.492 against the part's r^-0.302.** The wheel's global peak sits on
`rim:P_c` not because the rim junction is the most loaded place on the wheel, but because
**the mesh puts a far sharper corner there than the part has.** The artefact is not a small
distortion; it is nearly ninety degrees of wedge.

#### TWO CORRECTIONS TO §34's OWN NUMBERS

- §34 gave the part's second corners as 271.53 and 320.10 deg. Those came from a probe
  that took the ACUTE branch of the void angle **by assumption** rather than choosing the
  free-arc branch by construction. Measured properly: **268.47 and 219.90.** The hub number
  barely moves; **the rim number moves 100 deg**, and in the direction that matters. §34's
  tables are corrected in place and this is the record of why.
- §34 said the mesh/part difference was undocumented. It is not — `wheel_wheel.py`'s module
  docstring documents and justifies it. Corrected in §34 itself.

#### THE RESIDUAL RISK, NAMED RATHER THAN DISCOVERED LATER

**The rim's best direction is genome-dependent** — `shared_tangent` on seven of eight,
`own_tangent` on `best_solution_ga_beam` — which is `_embed`'s blend search reappearing,
and a search is exactly what must not enter the coordinate map. **The mitigation is that it
need not:** shared and own tangent differ by under 1 deg of wedge against the end cap's 87,
so **fixing** the rim to `shared_tangent` unconditionally stays smooth, differentiable and
search-free while remaining two orders of magnitude better than today. Step 2 fixes it and
records the residual; it does not reproduce the search.

#### The successors, ranked

1. **`UNCAP_PLAN.md` Step 2** — build it: blend 1.0 (radial) at the hub, blend 0.0 (shared
   tangent) at the rim, ring crossing in closed form. **The cost section of that plan
   still applies in full** — it moves welded volume, mass, hub share and every constant
   calibrated against them, so it is priced one number at a time with no threshold touched
   and nothing promoted inside the arc.
2. **`FILLET_PLAN.md`**, still blocked behind 1, and §34 Finding 4 is still why.
3. **§32's successors 3 and 4** — §8's wall-floor economics under SVK, and the kernel
   default with its 11 silent drivers.

### 36. UNCAP STEP 2: THE HUB IS FREE, THE RIM NEEDS A TOPOLOGY CHANGE — AND THE MODEL HAS BEEN OVERSTATING ITS OWN PEAK STRESS BY 2x (2026-08-18).

§35's Step 1 said go. Built, opt-in (`build_wheel(..., uncap=)`), with `uncap=False`
**bit-identical** at `smoke`/`coarse`/`medium` (`max|dx| = 0`) and the uncapped mesh
assembling with identical node and element counts at a 3.15e-14 mm seam error. Working
record in `UNCAP_PLAN.md`.

#### THE HEADLINE

```
  uncap            axle drop   comp hub  comp rim   area mm2   vs STEP   max vM   peak at
  False (today)     1.620790   0.041656  0.316117  1420.6161   -2.236%   99.130   rim:P_c
  (True, False)     1.594888   0.034111  0.315402  1421.9600   -2.154%   98.418   rim:P_c
  (False, True)     1.542107   0.041466  0.308861  1424.5844   -1.995%   74.833   hub:P_c
  (True, True)      1.517569   0.033985  0.308347  1425.9283   -1.913%   49.246   hub:P_t
```

**The peak CASCADES.** Fix the rim artefact and the maximum jumps to the *hub* artefact;
fix both and it lands on `hub:P_t`, a corner the part has to 0.073 um. **The model was
overstating its own peak by 2x — 99.13 -> 49.25 MPa — entirely on corners the shipped part
does not have.** That is §34 Finding 4 confirmed by construction rather than by
correlation, and it is the strongest result this arc has produced.

Corner fidelity, as built: hub **28.71 -> 0.01 deg** of wedge error, rim **87.53 -> 1.06**.
Area deficit against the shipped STEP narrows -2.236% -> -1.913% (the slivers are part of
`_embed`'s gusset); it does not close, and this arc never claimed it would.

**Flagged for `HUBSHARE_PLAN.md`, not acted on:** hub compliance falls 0.041656 -> 0.033985
(-18.4%). `test_the_hub_junction_holds_under_three_percent_of_the_compliance` gates on
`< 0.03` and **still fails** — but more than half of §31's accepted deficit turns out to be
a mesh artefact rather than a design shortfall. Re-read that plan before working it.

#### THE HUB IS FREE; THE RIM COSTS EVERYTHING, AND THE REASON IS EXACT

`min_scaled_jacobian` 0.782505 -> 0.007208, localised entirely to `rim_junction`
(`hub_junction` is *better*: 0.796181 -> 0.796983; `max_aspect_ratio` improves 20.27 ->
16.24; every other block stays above 0.999).

Measured, the angle between the far flank's own tangent at the ring and each candidate
direction is **0.587 deg (hub) and 0.489 deg (rim) for the shared tangent**, against
**116.669 and 53.017 for the radial**. A tangent continuation is by definition smooth, so
at `_embed`'s rim blend of 0.0 **the junction stops being a quadrilateral and becomes a
curvilinear TRIANGLE** — ring arc, cross-section, and one smooth [flank + extension]
curve. A four-sided structured block on a three-sided region always carries a ~180 deg
vertex; measured 179.35 deg, and sin(179.35 deg) = 0.011 is the scaled Jacobian. **No
interior blend or smoothing changes the angle between two boundary curves.** `_embed` uses
blend 1.0 at the hub, which is also the well-shaped choice — that is precisely why the hub
is free and the rim is not.

#### THE BLEND SWEEP — THE TWO CONDITIONS ARE DISJOINT

`MIN_SJ_TARGET = 0.2` (barrier weight 3000; `test_mesh.py` and `test_wheel.py` both assert
`> 0.2`). Sweeping the rim blend: **the peak leaves the artefact only at blend <= 0.15
(`min_sj` <= 0.133, failing the gate), and the gate is cleared only at blend >= ~0.23 (peak
back on `rim:P_c`).** No overlap.

**And the gate-clearing end is worse than it looks:** at blend 0.25 `min_sj` is 0.2218
against a floor of 0.2000, so the margin falls from 0.58 to 0.02 and `min_sj` becomes an
ACTIVE constraint against a 3000-weight barrier for every genome the optimiser tries.
**That would change what the search does — a far larger consequence than the corner it was
meant to fix. No blend compromise ships.**

#### The successors, ranked

1. **The rim junction as a THREE-QUAD block.** It buys blend-0 fidelity (1.06 deg) *and* a
   well-shaped block, because it removes the 180 deg vertex instead of trading against it.
   **This is the first thing in this arc that genuinely touches `_seam_table`** — new
   entries, new counts — and §34 Finding 2's "the seam table survives" does not cover it.
2. **Flip the hub default to uncapped.** Free on quality, better on fidelity and aspect
   ratio. It is a separate decision because it moves mass, hub share and axle drop, so it
   needs its own baseline refresh (`make studies`) and a full `make test`.
3. **`FILLET_PLAN.md`**, still blocked behind 1 — but note §34's fold analysis assumed an
   end cap, so re-run its radius sweep on an uncapped mesh before trusting the 0.2 mm
   ceiling.
4. **§32's successors 3 and 4** — §8's wall-floor economics under SVK, and the kernel
   default with its 11 silent drivers.

**`make test` after Step 2 reads 476 passed, 2 xfailed, 0 failed, exit 0 (28 m 39 s)** — the same counts, the same two deliberately-held xfails and the same runtime to within 3 s as the §33 and §34 baselines (476/2/0, 28 m 32 s and 28 m 36 s). `build_wheel` is what most of the suite runs through, so this is the check that `uncap=` is genuinely inert at its default rather than merely inert on the three configs probed directly.

**Nothing promoted, `best_solution.json` untouched, no threshold moved, default still
`uncap=False`.**

### 37. THE RIM TRI-BLOCK DOES NOT BIND — STOPPED BEFORE BUILDING IT, ON TWO CHECKS THAT NEEDED NOTHING BUILT (2026-08-18).

§36 ranked the rim three-quad block successor #1. **§36 wrote that ranking before knowing
what the divergence rate does, and the ranking was wrong.** Two cheap checks retired it.
Working record in `UNCAP_PLAN.md` STEP 3.

#### THE PRICE, DERIVED FIRST

A triangle admits no two-quad partition; the minimum is the three-quad Y-partition and it
**splits all three sides**, two of which are shared (the ring arc with `rim_band_weld`,
the cross-section with `spoke`). **So it needs PARTIAL-EDGE SEAMS**, and whole-edge single
ownership is what this module's docstring calls "the whole safety net". The element counts
are also constrained — matching opposite sides forces `a1 = b2`, `a2 = c1`, `c2 = b1`, so
an even split would need `n_weld == n_thick` (10 vs 4 at `coarse`), and the uneven
solutions give blocks of **7x1, 3x1, 7x3** with a forced 1-element strip.

#### CHECK 1 — UNCAPPING DOES NOT STOP THE PEAK DIVERGING

```
                              smoke   coarse   medium   growth
  CAPPED (today)   global    61.922   99.130  126.248    2.04x
  uncap rim 1.0    global    48.208   73.728   91.152    1.89x
  uncap rim 0.0    global    34.217   49.246   65.647    1.92x
```

Flat. It never could converge: every corner stays re-entrant (`P_t` 321-322 deg, lambda
~0.503; its own rate is 3.91x capped against 3.99x uncapped). **Only a fillet removes a
singularity.** M4's problem is that the peak has no limit — which is why M8b-i.6 rebuilt
the stress constraint around a p-norm x `Kt` — so buying a further 28% of an unquotable
magnitude buys nothing.

#### CHECK 2 — UNCAPPING DOES NOT UNBLOCK THE FILLET

§36's successor 3 warned §34's 0.2 mm ceiling assumed an end cap. Re-measured: **R_max is
0.200 mm at both rings under `False`, `(True,1.0)`, `(True,0.0)` and `True` — identical.**
§34's fold is in the SPOKE block; `uncap` changes the JUNCTION block; they do not
interact. **The caveat is discharged and §34's ceiling stands unqualified.**

#### THE RE-RANK

```
  option                     min_sj    rim wedge err   peak growth   machinery
  capped (today)            0.782505      87.53 deg       2.04x       -
  uncap (True, 1.0)         0.782226      50.61 deg       1.89x       none
  uncap (True, 0.0)         0.007208       1.06 deg       1.92x       tri-block
```

The tri-block buys **only rim corner fidelity** — not convergence, not the fillet, not a
quotable peak — against partial-edge seams and forced 1-element strips. **Filed, not
built.**

**`uncap=(True, 1.0)` is free and strictly better than today on every measured axis:**
`min_sj` 0.782226 against 0.782505 (0.03% apart, inside the 0.2 floor's 0.58 of margin),
hub corner EXACT at 0.01 deg, rim error more than halved, `max_aspect_ratio` 20.27 ->
16.24, and no new machinery.

#### The successors, ranked

1. **Adopt `uncap=(True, 1.0)` as the default.** Free on quality, exact at the hub. It is
   a model change, so it needs its own baseline: full `make test`, `make studies` refresh,
   and a record of the mass / hub-share / axle-drop deltas. **Not a promotion — nothing in
   `best_solution.json` moves.**
2. **`FILLET_PLAN.md`** is now the only route to a convergent peak, and §37 removes the
   one caveat that was hanging over its 0.2 mm ceiling. Its blocker is unchanged and
   unrelated to this arc: the spoke block is ruled.
3. **§32's successors 3 and 4** — §8's wall-floor economics under SVK, and the kernel
   default with its 11 silent drivers.
4. **The rim tri-block**, filed with its price stated, should anyone later need rim corner
   fidelity better than 50 deg.

#### The lesson, recorded because it is now three-for-three this session

§34 mis-ranked the seam merge as the fillet's blocker. §36 mis-ranked the tri-block as
worth building. Both were ranked from a plausible mechanism rather than from a
measurement, and both took under an hour to retire once measured. **The pattern: when a
successor's value rests on "and then X will improve", measure X on the CURRENT model
first — the improvement is often already absent or already present.**

### §38 — 2026-08-18. THE UNCAP DEFAULT IS ADOPTED, AND §37's WORD "FREE" WAS WRONG

`wheel_wheel.UNCAP_DEFAULT = (True, 1.0)` is the default of `sector_blocks`,
`_sector_coords` and `build_wheel`. `uncap=False` still reproduces the pre-flip geometry
bit-for-bit. **Nothing in `best_solution.json` moved.** Full record in UNCAP_PLAN.md's
STEP 4 RECORD.

#### THE CORRECTION THIS SECTION OWES §37

§37 closed with: "**`uncap=(True, 1.0)` is free and strictly better than today on every
measured axis.**" I wrote that, and it is wrong in a specific and instructive way. Every
axis in the list behind it — `min_sj`, `max_aspect_ratio`, corner error, seam error, node
and element counts — is a MESH-QUALITY axis. The claim was never measured on the area
bookkeeping, on the adjoint, or on the objective, **and it does not hold on any of the
three.** Flipping the default turned six tests red and needed two source changes beyond
the flip itself.

**"Every measured axis" is not "every axis", and the gap between them is exactly the set
of things nobody thought to measure.** That is a stronger and more useful lesson than
§37's own (which was about ranking successors from mechanisms instead of measurements),
because here the measurements WERE taken, were correct, and were still the wrong basis for
the word "free". Prefer naming the axes: "free on mesh quality" would have been true, and
would have invited the question that took the rest of the day to answer.

#### WHAT THE SIX REDS WERE

Four repairs, none of which moved a threshold; one finding; one false red in a place that
matters.

1. **Three of them were one cause**: `modelled_area_reference` describes a REGION whose
   ends are the very cross-sections `uncap` removes, so the mesh grew by 2.8187 mm² at `medium`
   (2.8328 at `coarse`) and its own reference did not. `error_vs_modelled` stopped being a discretisation residual
   and became a region mismatch that GROWS under refinement (+0.152% -> +0.167%), which
   is what `test_area_converges_under_refinement` exists to catch and did.
   `_uncap_reference_poly` extends the reference the same way and the residual returns to
   0.0088 mm². `reference_shipped_step_mm2` deliberately does NOT follow — the shipped
   solid is a fixed physical thing — so the newly-modelled 0.2342 mm²/spoke is reported
   as `gusset_modelled_per_spoke_mm2` instead of being netted into the constant.

2. **One was fragile, not wrong**: a whole ring of nodes sits at exactly r = 49.500000 and
   the corner-singularity test picked its "interior node" by `argsort`'s tie-break, which
   the flip reordered onto a midside.

3. **One was a false red on the ADJOINT, and it took three measurements to clear it.**
   G9 went 10x over its 1e-5 gate. Ring-by-ring: each ring alone is clean, only both
   together fails. Splitting the quotient: `dF/d delta` and `dF/dp` are each already as
   accurate as capped (2.79e-06, 2.03e-06). The residue was the SECANT's stopping rule in
   the FINITE-DIFFERENCE REFERENCE — one decade tighter and G9 reads 1.99e-06.
   `GATE_SECANT_REL` is untouched.

   **`run_axle_drop`'s docstring asserted the opposite, on a real measurement**: "THE
   SECANT'S TOLERANCE IS NOT WHAT LIMITS THIS... Tightening it to 1e-11 moves the
   difference by nothing at all." True on the mesh of the day, false one mesh later. **A
   measured justification is only as general as the configuration it was measured on, and
   nothing in a docstring records which configuration that was.** Same shape as §35's
   `_embed` correction, and that makes it a pattern rather than an incident.

4. **One is a finding.** See below.

#### THE FINDING: THE FLIP RE-PRICES THE STRESS TERM

`test_but_above_the_knee_the_fillet_radii_are_live` failed because its fixture stopped
being above the knee: **hub utilisation 0.85506 -> 0.77297 on the same genome at the same
settings.** The stress constraint is a p = 4 Gauss p-norm times `Kt`, and uncapping
deletes two of the four re-entrant corners per junction — the two the part does not have
— so the p-norm falls about 10%. (§36 measured the same effect on the raw maximum, where
it is a factor of two.)

**The shipped genome records `stress_utilisation` = 0.8201, ABOVE the 0.80 knee, with
`stress_margin` = 0.1317 of live loss.** Under the uncapped mesh that term goes to zero.
So part of what shaped the shipped design was a stress term reading corners that exist
only in the mesh — which is precisely what §36 predicted when it found the global peak
sitting 11-16 um from `rim:P_c`, now visible in the objective instead of in a diagnostic.

This does not make the flip wrong; the new number is the faithful one. It makes the flip a
**re-pricing of the objective**, which is a much larger claim than "free" and is the reason
nothing is promoted here: the shipped genome was optimised against a term it would no
longer feel.

**Measured across the archive, at `coarse` / 8 uniform phases:** `stage3_minwall_best_0.8`
— the thinnest wall and so the most stressed design in the tree — reads **0.78664**; the
fixture reads **0.77297** (was 0.85506); the shipped genome reads **0.71316** with
`stress_margin` measuring **exactly 0.000000** against the 0.13168 recorded at its
promotion. **No design on disk clears the knee any more.** That is not a broken term — a
knee'd penalty is meant to be flat below its knee — it is a statement about where the whole
archive sits once the stress field stops counting corners the part does not have.

`test_but_above_the_knee_the_fillet_radii_are_live` is SPLIT (§31's precedent, same repo):
the rim half passes and stays live; the hub half is a strict xfail carrying that table as
its reason. `MARGIN_KNEE_UTIL` is untouched at 0.80.

#### The suite

**476 passed, 3 xfailed, 0 failed, exit 0, twice, at 31m46s and 31m36s.** 479 tests where
there were 478 (one test split in two, one half now a strict xfail), so `passed` holds at
476. The new baseline is **31m40s** against the pre-flip 28m3x.

**The +3m07s is unattributed, and I got its cause wrong once already.** `area_report`
doubled (51 -> 101 ms) because `uncap` needs the area reference twice; caching it made that
call **190x faster** and moved the suite **10 seconds**. It was never on the critical path.
The cache is kept for sweeps and documented as buying throughput, not suite time. The one
contributor known by construction is this arc's `fd_tol_rel` 1e-8 -> 1e-9. **A fix that is
correct is not thereby the fix for the thing you attributed it to** — and that is the
fourth wrong cause in this arc, the only expensive one being the one acted on before
measuring.

#### `make studies` — five of nine, and the old blocker was hiding a new one

`study_gnl` **PASSES**, printing §33's own `pass_means`. The standing belief that this
recipe cannot complete because it exits 1 on the breached small-load gate was a full arc out
of date; §33 decoupled them and nobody re-ran it. Cost of checking: one grep.

It was masking `study_contact`'s **G1 at 1.021e-03 against a `< 1e-03` gate**, which no test
pins and which only `make studies` checks. Measured capped-vs-default rather than inferred:

```
                       drop @1e4   drop @1e5   penetration @1e4      rel     gate 1e-3
  capped (uncap=False)  1.532714    1.531242       5.9878e-04     9.610e-04   PASS
  DEFAULT (True, 1.0)   1.463794    1.462301       5.5167e-04     1.021e-03   FAIL
```

The flip caused it — and **the gate was already at 96% of its limit**, one small change from
red in any case. The mechanism deserves attention: G1's numerator is a penetration
difference while its denominator is the axle drop, so **making the wheel stiffer, i.e. the
model more faithful, makes this gate harder to pass with nothing about the contact model
changing.** The other half of the same criterion — the half anchored to the 1.5 mm band
rather than to the answer — moves the RIGHT way, 5.99e-04 -> 5.52e-04 mm.

**Nothing is done about it.** `GATE_EPS_PLATEAU_REL` is not moved (§19) and G1 is not
re-normalised, which would be the same act in better clothes. Its docstring records one
prior revision made on measurement; a second deserves the same standard and its own
decision. Artifacts are now MIXED: five drivers describe the uncapped mesh, `study_contact`
describes it with a red gate, and `study_gradient`/`study_objective`/`study_stage3` still
describe the capped one.

#### 2026-08-19 — THE FLIP BROKE THE DIFFERENTIABLE MESH PATH, AND I PUT THE BUG THERE

Found while auditing why `mesh_mass_g` came back **identical to four decimals** for a capped
and a faithful mesh whose utilisations plainly differed. It was not a coincidence.

`wheel_wheel.mesh_coords` and `wheel_wheel.coord_fn` both called `_sector_coords` **without
passing `uncap`**, so the traced half took the module default. While `UNCAP_DEFAULT` was
`False` that could not disagree with anything — the module default and every mesh's value
were the same object, so the omission was *vacuous*. §38 flipped the default to
`(True, 1.0)` and it became a wrong answer: a mesh built `uncap=False` was handed the
FAITHFUL geometry's coordinates.

**Measured: 0.448 mm, against the 1e-9 mm this path is documented to hold to.** Fixed, and
after the fix both settings reproduce their own eager mesh exactly — 0.0 mm through the
numpy path, 4.26e-14 mm through the jit.

The second half was the `_COORD_FN_CACHE` key, which omitted `uncap` too. Two meshes
differing ONLY in `uncap` share every other entry in that key, so whichever was built first
won and the other silently received its jitted function. **Passing `uncap` without also
keying on it fixes nothing**, and a single-build-order test passes with the cache still
wrong — which is why the new test is parametrized on build order.

**THE FIX IS VALIDATED AGAINST A PRE-FLIP RECORDED ARTEFACT, TO 1e-11.** The cleanest
available check: re-run the objective POST-FIX at `medium`/SVK/8 forced to `uncap=False`,
which is exactly the geometry and settings `best_solution.json` was descended under, and
compare against the `loss_terms` recorded in that file — a number written before the flip
existed, by code that never knew about it.

| term | post-fix capped | recorded in `best_solution.json` | agreement |
|---|---|---|---|
| `mass` | 32.440977356532755 | 32.440977356532755 | EXACT, all 17 digits |
| `smoothness` | 0.16782828102126746 | 0.16782828102126746 | EXACT |
| `total` / `loss` | 32.74463190311888 | 32.74463190340839 | 1e-11 rel |
| `stress_margin` | 0.13168057133155478 | 0.13168057521473925 | 3.9e-09 |
| `deflection` | 0.004145694233300536 | 0.004145690639631074 | 3.6e-09 |

**Pre-fix, that same cell returned `total` 32.80878 — off by 0.064**, and `mass_g` 39.5479
where the capped mesh's own mass is 39.46986. The gradient was wrong too, though less
dramatically: `stress_margin`'s `grad_share` reads 0.32099 post-fix against 0.32902 pre-fix,
so "about a third of the gradient" survives but the third digit did not.

**Nothing shipped is wrong.** No shipped path builds a non-default mesh, and the default
mesh reproduced to 4.26e-14 throughout. What it broke is precisely the capped-vs-faithful
comparison §38 exists to make, which is the worst possible thing for it to break quietly.

`tests/test_gradient.py::test_mesh_coords_reproduces_build_wheel` says of itself: *"if the
two paths ever diverge, every gradient in the project is a correct derivative of a mesh
nobody solved, and nothing else in the suite would notice."* That was accurate, including
about itself — it only ever built default meshes, so it passes on the broken code. The new
`..._at_a_NON_DEFAULT_uncap` sits beside it, is parametrized on build order, asserts through
BOTH the jit and the explicit-numpy call sites, and asserts the two meshes actually differ
so it cannot go vacuous. **Verified to fail on the unfixed code and pass on the fixed one**,
rather than merely believed in.

**What this invalidates in my own measurements**, stated rather than buried: any T2 quantity
(`mass`, `min_sj`) or gradient taken on an explicitly-capped mesh before this fix is wrong,
because those reach the genes through `mesh_coords`. Utilisations and `stress_margin` VALUES
are unaffected — they come from the solve on the eager mesh. The §38 delta table's
`mesh_mass_g` column was taken with `area_report`/`build_wheel`, not `mesh_coords`, and
stands.

#### 2026-08-19 — G1 IS UNREACHABLE, AND THE FLIP ONLY REVEALED IT

`study_contact`'s G1 read 1.021e-03 against `< 1e-03` after the flip, having read 9.610e-04
before it. I recorded that as "the gate was already at 96% of its limit". Measured properly,
it is worse than that: **there is no admissible setting in which this gate passes on the
faithful mesh at `coarse`.**

The full sweep, shipped genome, shipped default mesh, `coarse`:

| eps_n | axle_drop_mm | penetration_mm | pen/band | iters |
|---|---|---|---|---|
| 1e2 | 1.47914456 | 1.630377e-02 | 1.087e-02 | 1 |
| 1e3 | 1.46747589 | 3.455488e-03 | 2.304e-03 | 1 |
| 1e4 | 1.46379436 | 5.516664e-04 | 3.678e-04 | 1 |
| 1e5 | 1.46230081 | 6.231738e-05 | 4.154e-05 | 2 |
| 1e6 | **did not converge** | — | — | 60 |

| pair | abs resid mm | rel — THE GATE | vs 1e-3 |
|---|---|---|---|
| 1e2→1e3 | 1.166867e-02 | 7.8888e-03 | FAIL |
| 1e3→1e4 | 3.681526e-03 | 2.5087e-03 | FAIL |
| 1e4→1e5 | 1.493549e-03 | **1.0203e-03** | FAIL |

**Every measurable pair fails.** And `1e6` is a conditioning floor, not a tolerance choice:
retried at Newton `tol` 1e-10, 1e-8 and 1e-7, the relative residual **stalls at exactly
5.994e-07 in all three** (energy increment 5.887e-11 against `tol_energy` 1e-14). Identical
to seven digits at three tolerances is a stall, not slow convergence. `1e5` is the ceiling,
so G1's "next stiffer decade" beyond it does not exist and **1e4→1e5 is the only pair the
gate can ever read.**

**The docstring's model expires exactly where the gate reads.** It says a penalty method's
error in the axle drop simply IS the penetration, measured between 1e3 and 1e4. But the
ratios of successive drop differences are **3.17 then 2.47** — order `eps^-0.5` decaying to
`eps^-0.39` — while the penetration ratios are 4.72, 6.26, 8.85, i.e. approaching the
first-order 10. So `resid/pen` climbs **0.716 → 1.065 → 2.707** instead of holding near 1.
A 1e-3 relative gate on a sequence converging at order ~0.4 is not reachable at any eps_n a
penalty method can carry.

So the second version of this gate is unmeetable by construction too — for a different
reason than the first version was, and the flip did not cause it. Before the flip it read
9.610e-04, four percent under the line, which looked like a pass and was a coincidence.

**Nothing was changed.** `GATE_EPS_PLATEAU_REL` is not moved (§19), G1 is not re-normalised —
that is the same act in better clothes — and `DEFAULT_CONTACT_EPS_N` is not moved either:
1e5 is reachable and would halve the penetration, but it is one decade under a stall and it
would move every contact answer in the project, which is a promotion-scale act and not this
arc's. The first revision of this gate got its own record on measurement; the second one
deserves the same, and this is the measurement it would need. Untested: whether G1 reads
differently at `medium` or `fine` — the convergence ORDER is a property of the penalty
method rather than the mesh, so I do not expect the rung to rescue it.

#### 2026-08-19 — SUCCESSOR #1, MEASURED AT THE GENOME'S OWN SETTINGS

§38 said the shipped genome "was descended with `stress_margin` at 0.13168 and now measures
0.000000", and offered two answers: re-descend, or conclude the term was doing no useful
work. **Neither is right, and the framing hid the number that matters.**

First, the scope. `best_solution.json`'s `search` block is `medium` / SVK / 8 uniform
phases, and its `stress_margin` 0.13168057521473925 reproduces arithmetically as
`325 * (0.8201288598947069 - 0.80)**2`. Any capped-vs-faithful claim about that term has to
be taken THERE. Measured at exactly those settings:

| mesh | util_hub | util_rim | stress_margin | grad_share | axle drop mm |
|---|---|---|---|---|---|
| capped (`uncap=False`) | 0.82013 | 0.54537 | 0.131681 | 0.321 | 1.99742 |
| SHIPPED DEFAULT | **0.78519** | 0.52214 | **0.000000** | 0.000 | 1.91441 |

The capped cell reproduces the recorded numbers to **3.6e-10 relative** on the utilisation
and **3.9e-09 absolute** on the margin, which is what makes the pair trustworthy rather than
merely plausible. §38's headline comparison is therefore CORRECT — both of its numbers are
`medium`/SVK. What is not correct is any version of this taken at `coarse`/linear, where the
term reads 0.000000 on BOTH meshes (0.73539 capped, 0.71316 faithful) and the flip looks
like it changed nothing. It is the same trap as §37's "every measured axis".

**THE TERM IS NOT DEAD. IT IS UNLOADED BY 1.85%.** `soft_barrier` is
`scale * max(0, util - 0.80)**2`, a one-sided hinge: below the knee its value AND its
gradient are ALGEBRAICALLY zero, not small. So "does it measure 0.000000" was never a
question a descent was needed to answer. The question worth asking is how far below the knee
the genome sits, and the answer is **1.85%** — not the 11% that `coarse`/linear would have
suggested. On the capped mesh the term was live and carrying **a third of the gradient**; it
was a real constraint the descent pushed against, and the flip did not reveal it as useless,
it handed back margin that was being charged against phantom stress at two re-entrant
corners the part does not have.

**So the answer is "re-descend", but not for §38's reason.** The purpose is not to find out
whether the term matters. It is to collect what 1.85% of stress headroom buys, and that is
priceable in advance from §18's own stated exchange rate — the weight 328.49 was chosen so
1% of utilisation costs 1% of mass at the knee, making 1.85% of utilisation worth about
1.85% of mass, i.e. **~0.73 g of the 39.5 g mesh mass**. A descent will consume that headroom
and re-engage the barrier at 0.80; that is the barrier working, not the barrier being
pointless. The axle drop improves independently (1.99742 -> 1.91441 mm, -4.16%) because the
faithful mesh is stiffer.

**AND THE HEADROOM IS NOT A DESIGN FACT. MEASURED BEFORE SPENDING THE FIVE HOURS, NOT AFTER.**
The paragraph above priced ~0.73 g against 1.85% of headroom. That number is a `medium`-rung
artefact. The ladder at fixed kinematics, faithful mesh, shipped genome, 8 uniform phases:

| cfg | kinematics | util_hub | headroom to knee | axle drop mm |
|---|---|---|---|---|
| smoke | svk | 0.75490 | +5.64% | 1.82776 |
| coarse | svk | 0.77876 | +2.65% | 1.90110 |
| medium | svk | 0.78519 | +1.85% | 1.91441 |
| **fine** | **svk** | **0.78979** | **+1.28%** | 1.92328 |
| coarse | linear | 0.71316 | +10.85% | 1.62119 |
| medium | linear | 0.71959 | +10.05% | 1.63209 |

**The headroom shrinks at every refinement and the sequence has NOT converged.** Successive
increments are 0.02386, 0.00643, 0.00460 — the ratio jumps from 0.27 to 0.72, which is not
an asymptotic regime. Extrapolating geometrically from the first triple gives a limit near
**0.791**; from the last triple, near **0.801**. Those STRADDLE the 0.80 knee. So the honest
statement is: at every rung actually measured the term is exactly 0.000000, and the margin
of that conclusion is 1.28% at `fine` and falling. **There may be no headroom to collect at
all, and the genome may sit at or above the knee on a converged mesh with no descent
whatsoever.**

This is `HUBSHARE_PLAN.md`'s rule applied to this arc's own decision: do not build on a
non-converged quantity. It would have been easy to skip — the 1.85% figure came from the
genome's own settings, reproduced the recorded artefact to 3.6e-10, and looked authoritative.
It is still not converged.

**AND THE KERNEL INVERTS THE CHARACTER OF THE ANSWER, NOT JUST ITS MAGNITUDE.** Linear reads
~10.5% of headroom and looks SETTLED (10.85% -> 10.05%, barely moving across a rung); SVK
reads 1-5% and drifts hard (5.64% -> 1.28%). A reader who took the kernel default here would
conclude both that there is ten times more headroom than there is AND that the number is
converged when it is not. §33's rule — never take the kernel default for a sensitivity
claim — is doing real work at exactly this measurement.

**So the descent is NOT justified yet, and §38's item 1 does not close.** What it needs first
is the `ultra` rung, or an explicit decision to treat this utilisation the way §31 treats the
hub share: a quantity too discretisation-dependent to price a design change against. Five
hours and 32 GiB spent collecting 1.85% of a margin that reads 1.28% one rung finer, and
extrapolates to zero, is the expensive way to learn this.

### 2026-08-20 — THE `ultra` RUNG ANSWERED IT, AND THE ANSWER IS "DO NOT PRICE AGAINST IT"

Successor 1 asked for `ultra` or a ruling. It got `ultra` — 587208 dofs, 2.22x `fine`, 4878 s
at 14.8 GB, faithful mesh, SVK, 8 uniform phases, the shipped genome, same call as every
other rung. **`util_hub` = 0.79347, still below the 0.80 knee, `stress_margin` still exactly
0.000000.** Taken alone that reads as a clean answer. It is not, and the increments are why:

| rung | dofs | `util_hub` | increment | ratio to prior | headroom |
|---|---|---|---|---|---|
| smoke | 22 k | 0.75490 | — | — | +5.64% |
| coarse | 42 k | 0.77876 | 0.02386 | — | +2.65% |
| medium | 106 k | 0.78519 | 0.00643 | 0.269 | +1.85% |
| fine | 265 k | 0.78979 | 0.00460 | 0.716 | +1.28% |
| **ultra** | **587 k** | **0.79347** | **0.00367** | **0.799** | **+0.82%** |

**THE RATIO IS CLIMBING TOWARD 1, NOT DECAYING.** 0.269 -> 0.716 -> 0.799. A converging
sequence has a ratio bounded below 1 and typically falling; this one rises at every rung. In
`h` the apparent order falls from ~3.0 to ~0.5 (element counts give an `h` ratio of ~1.55 per
rung). Whatever this sequence is doing, it is not settling down.

**EVERY EXTRAPOLATION IS NOW ABOVE THE KNEE AND EVERY MEASUREMENT IS BELOW IT.** A constant
ratio at the last observed 0.7986 puts the limit at **0.8080**; at the previous 0.7154 it is
**0.8027**. On 2026-08-19 the two nearest extrapolations straddled 0.80 (0.791 against 0.801)
and one more rung looked decisive. It was decisive — it removed the straddle in the direction
that makes the quantity useless for pricing, not the direction that settles it.

**THE DRIFT IS IN THE FIELD, NOT IN THE GEOMETRY — MEASURED, NOT ASSUMED.** `util_hub` is
`kt * agg / ALLOWABLE`, and `kt` takes a config, so it could have been either. It is not:
`kt_hub` runs 2.09654624 / 2.09632456 / 2.09621446 / 2.09615960 / 2.09613221 across the same
five rungs — increments halving cleanly, first order, converged, and moving **-0.0197% in
total, in the OPPOSITE direction to the drift**. All +5.11% of the rise is `agg`, the p-norm
of the stress field.

That is the field this tree already knows is singular. M8b-i.6 step 1 demoted `c` to a
diagnostic precisely because the per-phase MAX diverges under refinement — 31.02 -> 41.54 ->
48.47 MPa, GCI 34.4% — at the same unfilleted re-entrant corner. A p-norm is a milder
functional of that field, so a slow, non-decaying drift is what the singularity predicts.
**Do not overstate this: a finite-p norm of an integrable singularity CAN converge, and the
ladder cannot separate "creeping to ~0.808" from "not converging at all".** It does not need
to. Both readings put the answer above the knee.

#### THE RULING — 2026-08-20

**`util_hub` on this design is too discretisation-dependent to price a design change against,
and the five-hour descent stays unfunded.** This is the same call §31 made for the hub share,
made on the same kind of evidence, and it is mine rather than deferred.

What it does NOT mean:

- **`MARGIN_KNEE_UTIL` does not move.** §19. A gate is not re-fitted to the design that
  approaches it, and it is certainly not re-fitted to an extrapolation.
- **The stress term is NOT confirmed inert.** `soft_barrier` is algebraically zero below the
  knee and it is zero at all five rungs, but the extrapolated limit is above the knee, so a
  converged mesh may well switch it on. "The term does no work" and "the term binds" are both
  unsupported. The honest statement is that the mesh cannot tell us which.
- **Nothing was promoted, re-descended, or exported.**

**WHAT UNBLOCKS IT IS THE FILLET, AND THAT IS NOW THE RANKING.** The drift lives in a field
p-norm at an unfilleted re-entrant corner; `FILLET_PLAN.md` is the arc that puts a modelled
radius there. It is the only open work that could make this quantity converge, which promotes
it from 4 to 1 on the strength of a measurement rather than on the standing argument that it
has been top of the arc list for five arcs.

### §39 — 2026-08-20. G1's THIRD REVISION: A BETTER ESTIMATOR, A WORSE NUMBER, AND §33's RULE APPLIED

§38 left G1 measured-unreachable and deliberately unchanged, with the note that a second
revision "deserves its own record on measurement". This is it. **No threshold moved. The gate
reads redder than before. `make studies` is unblocked anyway**, and those three facts are the
whole point: the bound was never the problem.

#### What was actually wrong — two things, neither of them the bound

**1. THE ESTIMATOR WAS OPTIMISTIC BY 68.6%.** G1 asked whether the shipped `eps_n` is
converged, and estimated the distance to the `eps_n → ∞` limit by the difference between two
decades. That estimator is only valid if the sequence is **first order**, and §38 had already
measured that it is not — the drop's successive differences fall by 3.17 then 2.47 per decade,
an order of **0.50 decaying to 0.39**. What the default still carries is therefore a geometric
**tail**, not a single step. Three estimators of the same quantity, all failing, disagreeing
badly:

| estimator | implied limit (mm) | error at the default | verdict |
|---|---|---|---|
| successive difference — *what the gate read* | 1.46230081 | 1.0203e-03 | FAIL |
| first-order Richardson | 1.46213486 | 1.1350e-03 | FAIL |
| **measured-order tail sum — reported now** | **1.46128129** | **1.7198e-03** | **FAIL** |

The gate now reports the **worst** of the three. A revision written to make a red gate go green
would have picked the first, which is the one that was already there. And because the ratios are
*rising* (0.316 → 0.406), convergence is slowing and even 1.72e-03 is a lower bound — the same
structural signature the `ultra` rung found in `util_hub` on the same day, in an unrelated
quantity.

**2. THE EXIT SEMANTICS CONFLATED TWO DIFFERENT CLAIMS.** G1 answers two questions with one
verdict — *is the solve trustworthy* and *is the penalty method's convergence rate at this
design good enough* — and only the first can invalidate a downstream number. That is exactly
§33's rule, the one this tree paid ten days for when `study_gnl` stopped the recipe over a
finding about the wheel. `study_gnl` already had the pattern; G1 now uses it:

| verdict | what it asks | reads |
|---|---|---|
| `solver_pass` | every admissible decade converged, the default is bracketed, the sequence is monotone and contracting, and the penetration at the shipped setting is negligible against the band it dents (3.678e-04 against 1e-03) | **PASS** |
| `regime_pass` | is the default within `GATE_EPS_PLATEAU_REL` of the extrapolated limit | **FAIL** at 1.7198e-03, and expected to keep failing |

`pass` still computes the same conjunction on the same old estimator, so that field means in
every `study_contact.json` ever written what it meant when it was written. It is simply no
longer what the exit code is built from — `solver_is_correct` is.

**A FAILED STIFFEST DECADE IS NOT A SOLVER FAILURE**, and the code says so explicitly.
`eps_n = 1e6` is the documented conditioning ceiling (§38: the residual stalls at 5.994e-07 at
Newton tol 1e-10, 1e-8 **and** 1e-7). Requiring it to solve would mean adding a decade to the
sweep could turn the gate red for a limit the tree already understands. What the solve must do
is *bracket* the default.

#### What is NOT changed, and one thing that is now open

`GATE_EPS_PLATEAU_REL`, `GATE_PENETRATION_FRAC`, the normalisation and
`wheel_fem.DEFAULT_CONTACT_EPS_N` are all untouched. §19 stands.

What §39 does open is narrower and honest: **the 2.5e-03 mm the default carries is 0.13% of the
2.0 mm axle-drop target the objective steers by, against a 5% feasibility band.** So `1e-3`
relative may be very much tighter than anything downstream needs. **Deriving** the bound from
that requirement is legitimate work and would be a fourth revision with its own record;
**widening** it because the gate is red is the move §19 forbids, and §39 deliberately does not
make it. The distinction is the entire difference between the two, and it is the reason this
revision left the number where it was and made the reading worse.

#### The successors, ranked — REVISED 2026-08-20 AFTER §39

§39 closes what was item 2 and unblocks what was item 3.

1. **`FILLET_PLAN.md`** — unchanged at the top, and on measured grounds: the `ultra` rung put
   `util_hub`'s non-convergence in `agg`, a p-norm of the field at the unfilleted re-entrant
   corner, with `kt` ruled out by direct measurement (−0.0197% across five rungs, and in the
   opposite direction). This is the only open arc that could make that quantity converge.
   Its Step 3 item 1 acceptance test, the `R_hub` sweep, is still bit-identical across the
   whole feasible box on the faithful mesh.
2. **Run `make studies` end to end.** §39 removed the blockage: `study_contact` now exits on
   `solver_is_correct`, which PASSES, and the other eight drivers were each run individually
   on the faithful mesh on 2026-08-19/20 and each exited 0. The recipe has not completed since
   **2026-08-06**. It is ~6 h of pure compute and nothing in it is a decision.
3. **G1's fourth revision — DERIVE the bound, or record that it should not be derived.**
   §39 deliberately did not touch `GATE_EPS_PLATEAU_REL`, and left the question stated: the
   default carries 0.13% of the axle-drop target against a 5% feasibility band. Anyone taking
   this must derive the number from the requirement and register the derivation BEFORE
   reading what the gate then says. Not urgent — the gate no longer blocks anything.
4. **§32's successors 3 and 4** — §8's wall-floor economics under SVK. Premise still holds
   (every `stage3_minwall_*.json` is linear/coarse/2026-08-03). Pure compute, and still behind
   item 1: it would run against an objective whose stress term sits 0.82% from a knee it may
   or may not cross.
5. **The rim tri-block**, still filed, still not binding.

**`HUBSHARE_PLAN.md` is NOT on this list and that is deliberate**: its Step 0 is blocked
behind `FILLET_PLAN.md` by its own rule — and item 1 is that arc, so this may become reachable
for the first time in five arcs.

---

### §40 — 2026-08-20. THE RECIPE COMPLETED. EXIT 0, 5:02:52, 30.8 GiB — AND IT REPRODUCED

§39's ranked item 2, run end to end. **`make studies` exited 0 for the first time since
2026-08-06.** There is no decision in this section: it is the compute §39 said was left once
the blockage was gone, plus one finding nobody was looking for.

#### The nine, and what they cost

| # | driver | verdict | s |
|---|---|---|---|
| 1 | `study_mesh_quality` | M2a **PASS** — with the fold constraint in the feasible set | ~40 |
| 2 | `study_wheel_mesh` | M2b **PASS** — area, design_space | 4.5 |
| 3 | `study_beam_agreement` | M3 **PASS** — A1_A2, A3, A4a, A4b | 4.5 |
| 4 | `study_wheel_fea` | M4 **PASS** | 77.8 |
| 5 | `study_gnl` | **PASS** on `solver_is_correct` | 143.1 |
| 6 | `study_contact` | **PASS** on `solver_is_correct` | 1161.4 |
| 7 | `study_gradient` | M7 **PASS** | 1029.1 |
| 8 | `study_objective` | M8a **PASS** | 4690.1 |
| 9 | `study_stage3` | **PASS** | 11021.2 |

**5:02:52 wall, 24971.9 s user + 6484.4 s system (173% CPU), peak RSS 30.8 GiB.** The ~6 h /
~32 GB estimate held. **`study_stage3` is 61% of the wall clock and `study_objective` another
26%** — the seven others together are 13%, and the two mesh drivers are 9 s of it. Anyone
costing a change to this recipe should cost it against those two drivers and ignore the rest.

#### §33's decoupling is now proven BY THE RECIPE, not by grep

Two drivers printed a red verdict and exited 0 anyway. That is the entire reason this run was
reachable, and §38 had established it by reading the code — this establishes it by running it.

- `study_gnl`: *"at 1% of service load the two agree to 0.1983% (gate 0.1%) → FAIL — the
  WHEEL, held red on purpose."* This is the driver that stopped the recipe at line 5 of 9 for
  ten days.
- `study_contact` prints both verdicts at equal volume, as §39 built it:
  `OVERALL (the SOLVE, and what the exit code is): PASS` beside
  `OVERALL (every verdict including characterisation findings): FAIL`. G1's `regime_pass`
  reads **1.7198e-03**, to the digit what §39 recorded.

#### THE UNPLANNED FINDING: THE RECIPE REPRODUCES TO FLOATING-POINT NOISE

Nine of the eighteen artifacts were **rewritten with no content change at all** — both mesh
drivers whole, four more `.jpg`. Of the nine files that did change, **four differ ONLY in
timing fields**: `study_beam_agreement`, `study_wheel_fea`, `study_gnl` and `study_contact`
have not one measured number altered from their individual 2026-08-19/20 runs. The other three
differ in last bits only — `study_gradient`'s adjoint `-1.5198858230529777` →
`-1.519885823052971` (~4e-15), `study_objective` the same order, `study_stage3`'s `R_hub`
~4e-13 and `loss_end` ~3e-10, the last larger because a descent accumulates last bits over
steps. S13 (`make m8bii1`) already documents that two plain serial interpreters disagree in the
adjoint's last bit with no pool involved; this is that, at recipe scale.

**Why it is worth a paragraph.** §38's parting observation was that the artifacts were MIXED —
five drivers describing the uncapped mesh, `study_contact` describing it with a red gate, and
`study_gradient`/`study_objective`/`study_stage3` still describing the capped one. §39 then
refreshed the stragglers individually. **That mixed state is now closed**: all nine artifacts
come from ONE invocation at ONE commit (`e12dfb6`) rather than from runs spread across
2026-08-18/19/20 at three tree states. And because the drivers reproduced, the individual runs
§39 leaned on were sound — the re-run could have contradicted them and did not.

#### What this section does NOT do

No threshold moved, no test deleted, nothing promoted. `best_solution.json` is untouched and
still 2026-08-14. `make test`: **478 passed, 3 xfailed, 1910.5 s.** `GATE_EPS_PLATEAU_REL`
remains where §19 and §39 left it — item 2 below is still the honest way to move it.

#### The successors, ranked — REVISED 2026-08-20 AFTER §40

§40 closes what §39 ranked item 2. Everything below simply moves up one, because running the
recipe was never going to reorder anything: it is a gate, and it reported what the tree already
believed.

1. **`FILLET_PLAN.md`** — unchanged at the top for the sixth arc, and still on §39's measured
   grounds: the `ultra` rung put `util_hub`'s non-convergence in `agg`, a p-norm of the field
   at the unfilleted re-entrant corner, with `kt` ruled out by direct measurement (−0.0197%
   across five rungs, in the opposite direction). It is the only open arc that could make that
   quantity converge. Its Step 3 item 1 acceptance test, the `R_hub` sweep, is still
   bit-identical across the whole feasible box on the faithful mesh. **Before starting it, the
   §32 check still applies**: the study drivers do NOT inherit `svk` by default — `wheel_fem`'s
   kernel defaults are `linear` on purpose and 11 drivers never mention `kinematics`, so a
   fillet ladder built on those takes linear silently.
2. **G1's fourth revision — DERIVE the bound, or record that it should not be derived.** The
   default carries 0.13% of the 2.0 mm axle-drop target against a 5% feasibility band. Whoever
   takes it must derive the number from the requirement and register the derivation BEFORE
   reading what the gate then says. Not urgent — §40 confirms the gate blocks nothing.
3. **§32's successors 3 and 4** — §8's wall-floor economics under SVK. Premise still holds
   (every `stage3_minwall_*.json` is linear/coarse/2026-08-03). Pure compute, still behind
   item 1: it would run against an objective whose stress term sits 0.82% from a knee it may or
   may not cross. **§40 prices it**: this is `study_stage3` territory, the 11021 s driver.
4. **The rim tri-block**, still filed, still not binding (§37).

**`HUBSHARE_PLAN.md` remains off this list**, still blocked behind `FILLET_PLAN.md` by its own
Step 0 rule — and item 1 is that arc, so it stays one arc from reachable.

**A standing note now that the recipe runs again.** It has completed exactly once. The cheapest
insurance against another ten-day gap is to run it after any change to `wheel_fem`, `wheel_wheel`
or the objective — but at 5 h and 30.8 GiB that is not a per-commit gate, and §40 deliberately
does not propose making it one. The two mesh drivers (9 s, both bit-reproducible) are the part
that could be.

---

### §41 — 2026-08-21. THE PER-COMMIT GATE IS REFUSED, AND THE AUDIT THAT REFUSED IT FOUND A FALSE GREEN

§40 closed with a question it deliberately did not answer: the recipe costs 5 h and 30.8 GiB,
so it cannot be a per-commit gate, but "the two mesh drivers — 9 s, both bit-reproducible —
are the part that could be." **This is the decision. The answer is no, at any tier**, and the
reason is not cost: the cheap tier was measured affordable and refused anyway.

#### What a fast tier would actually cost — measured, not estimated

Each driver run alone, full fidelity, `/usr/bin/time -v`:

| driver | wall | peak RSS | imported by a test? |
|---|---|---|---|
| `study_mesh_quality` | 19.93 s | 174 MB | **NO — the only one in the recipe** |
| `study_wheel_mesh` | 5.03 s | 150 MB | yes (`test_wheel`) |
| `study_beam_agreement` | 5.78 s | 1024 MB | yes (`test_fem`) |
| `study_wheel_fea` | 1:20.03 | 1901 MB | yes (`test_gnl`, `test_wheel_fea`) |
| `study_gnl` | 2:35.92 | 1816 MB | yes (`test_gnl`) |

**4 m 27 s and 1.9 GB for all five** — +14% on `make test`'s 31 m 50 s. Affordable. §40's "9 s"
was the three cheapest and undercounted the tier anyone would actually want, because the tier
worth having contains `study_gnl` — the driver that caused the first outage.

#### Why it is refused anyway — three measured reasons

**1. `make test` ALREADY COVERS EIGHT OF THE NINE.** Every recipe driver except
`study_mesh_quality` is imported by the suite and called into, at `--quick` fidelity, *by
explicit design*: `tests/test_fem.py:298-301` — the tests "call into it rather than
re-deriving it, so there is exactly one definition of each check and CI cannot drift away
from the published numbers." A fast tier re-runs those same checks at
higher sample counts. The liveness it would buy is already bought.

**2. RUNNING A DRIVER REWRITES ITS COMMITTED ARTIFACT.** Measured the hard way: the timing run
above dirtied five tracked artifacts and they had to be restored from §40's commit to keep the
recipe's provenance. A per-commit tier either churns `studies/*.json` on every commit — against
the header's "a study commit carries its artifacts" — or writes to a scratch `--out`, at which
point it has stopped being a gate and become a liveness check, which is reason 1.

**3. THE AFFORDABLE TIER MISSES WHERE THE SECOND OUTAGE LIVED.** It would have caught
`study_gnl` exiting 1 (§33's ten-day stall, and that is a real point in its favour). It would
NOT have caught §38's five-of-nine stall, which was `study_contact` at 1161 s — outside any
tier that could run per commit.

**So §40's sentence was true about cost and wrong about value.** The mesh drivers are the
cheapest part of the recipe and have never been the part that broke. It was ranked from a plausible
mechanism — "cheap and reproducible, therefore gateable" — rather than from a measurement of
what it would catch. That is exactly §37's three-for-three lesson, and §17's "the successor
ranked #1 is worth nothing", and it is now four-for-four: **§40 wrote that sentence and §41
retired it in one audit.**

#### THE AUDIT'S REAL FIND: `--quick` COULD FILE A FALSE GREEN AS THE M6 GATE

Every driver's `--out` defaults to its committed artifact name and **`--quick` does not change
it**. `study_contact` is the only one of the nine with any guard at all, and it refused a
partial `--sections` list and non-linear kinematics — but not `--quick`, which leaves `full`
True and `kinematics` "linear" and therefore walked straight through.

`studies/study_contact.py --quick` wrote **smoke-mesh** data into the committed
`studies/study_contact.json`. §39 had already measured what that reads: **G1 at 4.394e-04 with
BOTH halves passing**, against 1.7198e-03 and a red `regime_pass` at the real config. So this
was not a coarse gate standing in for a fine one — **it was a FALSE GREEN standing in for a
RED one**, in the single artifact that carries the M6 verdict.

Nothing had run that command, so nothing is known to be corrupt: `study_contact.json` is
§40's, from the full recipe. This is a live exposure closed before it fired, not a defect
found after the fact. It is the same shape as §33's third defect — seven of nine drivers
writing artifacts to the CWD — and the same shape as the exposure `test_smoke_does_not_touch_
real_artifacts` already pins for `wheel_fea.py`. **The main CLI has had this discipline for
arcs; the study drivers never got it.**

Fixed by extending the existing guard, which was the local choice and is argued for in its own
comment — "by name rather than by silently renaming the output". Seven regression tests pin it,
in both directions: four degraded invocations must refuse, and three (two redirected degraded
runs and the real gate itself) must pass. Verified against the pre-fix source: **exactly the
two `--quick` cases fail, the other five pass on both sides**, so the test pins the new
exposure and nothing else.

#### What is NOT done, and the one place a cheap addition still buys coverage

**`study_mesh_quality` is the only recipe driver no test imports.** M2a — the fold-margin
prediction, `MIN_FOLD_MARGIN_MM`, the 66.27%-vs-99.07% contrast that IS the gate — is exercised
nowhere outside the 5 h recipe.

> **THE SECOND SENTENCE ABOVE IS WRONG, AND §42 CORRECTS IT — 2026-08-21.** "Exercised
> nowhere outside the recipe" overstated what was checked, which was only that **no test
> IMPORTED the driver**. `tests/test_mesh.py::test_fold_margin_predicts_inversion` had tested
> the fold-margin property over 1500 samples all along. The real gap was **two definitions of
> the same check** — the test re-implemented `smq.fold_margin` inline — which is the drift
> `tests/test_fem.py:298-301` established the import idiom to prevent. Closed at `39dd96f`;
> the two agreed bit-identically on all 99 feasible genomes when substituted, so this was
> latent risk rather than a live defect. **Written as a caution about the sentence, not only
> about the driver**: "no test imports X" and "X is untested" are different claims, and this
> file stated the second having measured the first.

`tests/test_mesh.py:262` merely *cites* it in a comment. That
is the real gap this audit found, it is 19.93 s of compute, and **it belongs in `make test`,
not in a new `studies` target** — which is the whole decision restated in one case.

It is ranked below rather than done because it needs tests that assert something about the
fold margin, not an import line, and writing those is real work with its own record.

The other eight drivers are left alone. `GATE_EPS_PLATEAU_REL` is untouched, nothing is
promoted, `best_solution.json` is still 2026-08-14, and no threshold moved.

#### The successors, ranked — REVISED 2026-08-21 AFTER §41

1. **`FILLET_PLAN.md`** — unchanged at the top for the seventh arc, on §39's measured grounds.
   **PREMISE RE-CHECKED 2026-08-21 (FILLET_PLAN STEP 1 RECORD PART 5), AND ONE ROUTE INTO IT
   IS CLOSED.** PART 4 had re-ranked the arc on a prediction that removing the end cap would
   delete the corner that folds the spoke block, and §38 shipped that cap removal on
   2026-08-18. Measured A/B: **the fold is byte-identical capped and uncapped** — 12 of 4704
   elements at `coarse`, worst −3.0725e-02 mm², all 16 swept cells agreeing — because `uncap`
   is consumed in the **junction** block (`wheel_wheel.py:1067-1074`) and the fold is in the
   **spoke** block, which never receives it. Not the §38 plumbing bug; correct construction.
   So the arc still costs what it cost: a dedicated fillet block, or a generated spoke block.
   **The cheap way in does not exist**, and one hour of A/B is what says so.
   The §32 check still applies before starting: the study drivers do NOT inherit `svk` by
   default, so a fillet ladder built on them takes linear silently.
2. ~~**Cover `study_mesh_quality` in `make test`**~~ — **DONE 2026-08-21, `39dd96f`.** Three
   tests, each verified by mutating the driver and confirming the suite goes red: `meshable`
   aliased to `feasible_geom`, `MIN_SJ_ACCEPT` loosened 0.2→0.15, and `fold_margin`'s span
   wrong by 1%. The third exists because the first version of the work **failed its own
   mutation check** — importing the driver removes a duplicate definition but does not make
   its arithmetic observable, since there the margin only decides which genomes get meshed.
   All nine recipe drivers are now imported by the suite. +0.42 s; 488 passed, 3 xfailed.
3. **Extend the `--out` guard to the other eight drivers** — NEW, and deliberately ranked below
   item 2 rather than done with it. `study_contact` is fixed because that is where the false
   green was *measured*; the other eight have the same default-path exposure but no measured
   harm, no shared helper to hang a guard on (each driver has its own `argparse`), and
   different fidelity knobs. Doing it right is eight considered edits, not a sed.
4. **G1's fourth revision** — derive `GATE_EPS_PLATEAU_REL` from the requirement or record that
   it should not be derived. Unchanged; §40 confirmed the gate blocks nothing.
5. **§32's successors 3 and 4** — §8's wall-floor economics under SVK. `study_stage3`
   territory, the 11021 s driver (§40).
6. **The rim tri-block**, still filed, still not binding (§37).

**On cadence, since §40 left it open and this section closes it.** `make studies` stays a
deliberate full run, not a gate: at arc boundaries and before a promotion, where its artifacts
are supposed to describe the shipped genome. **The recurring failure it insures against is
artifacts describing a wheel that is no longer shipped** — §33's second defect, "two committed
study artifacts describe a wheel promoted out of their named file three times ago". That is a
promotion-checklist question, not a per-commit one, and it belongs to `tests/test_promotion.py`
and whoever runs the next promotion.

---

---

### §42 — 2026-08-21. THE M2a DRIVER IS UNDER TEST, AND §41's REASON FOR RANKING IT WAS WRONG

§41's ranked item 2, done — and the correction it forced is the more useful half.

#### The claim that ranked it, and what was actually measured

§41 wrote that `study_mesh_quality`'s gate is "exercised nowhere outside the 5 h recipe."
**That was too strong.** What had been measured is narrower: **no test IMPORTED the
driver.** `tests/test_mesh.py::test_fold_margin_predicts_inversion` had been testing the
fold-margin property over 1500 samples all along, so M2a's central claim was covered.

*"No test imports X"* and *"X is untested"* are different claims, and §41 stated the second
having checked the first. Recorded because the generalisation was invisible while writing
it and obvious the moment the file was opened — the same shape as §37's three-for-three
lesson and §41's own re-ranking error.

#### The real gap: TWO definitions of one check

The test re-implemented `smq.fold_margin` inline — the same `bezier_centerline` +
`self_intersection_margin` pair, at the same span — which is the drift
`tests/test_fem.py:298-301` established the import idiom to prevent: *"so there is exactly
one definition of each check and CI cannot drift away from the published numbers."*

They had **not** drifted (`SPAN = W.HUB_RIM_SPAN_MM` in both), so this closed latent risk
rather than fixing a live defect — and that was verified before substituting, not assumed:
over the same 1500-sample draw the two derivations agree **bit-identically on all 99
`feasible_geom` genomes, worst |old−new| exactly 0**.

#### Three tests, each verified by mutating the driver

| mutation | caught by |
|---|---|
| `meshable` aliased to `feasible_geom` | `test_meshable_is_feasible_geom_plus_a_positive_fold_margin` |
| `MIN_SJ_ACCEPT` loosened 0.2 → 0.15 | `test_the_m2a_acceptance_criterion_is_the_published_one` |
| `fold_margin`'s span wrong by 1% | `test_fold_margin_on_the_shipped_genome_is_the_recorded_value` |

**The third exists because the first version of the work failed its own mutation check.**
Importing the driver removes a duplicate definition but does **not** make its arithmetic
observable: in the fold test the margin only decides WHICH genomes get meshed, so a 1%
wrong span still selects genomes that mesh fine and every test stayed green. A golden value
closes it — `14.365501181531` at `coarse`, with `n_curve` named because the margin moves in
the 8th significant figure across the ladder (600 / 1200 / 2400 → `…181531` / `…553787` /
`…490528`), so the 1e-9 tolerance pins the arithmetic without pinning the ladder.

Pinning `MIN_SJ_ACCEPT` and `ACCEPT_FRACTION` earns its place separately: they are
pre-registered in the driver's docstring and in prose, §19 and §31 forbid moving a
threshold to make a gate green, and **a pre-registered number no test reads can be edited
in the same commit as the run that breached it.** These are what §40's recipe measured
66.27% and 99.07% against.

#### One more thing that had to be measured rather than guessed

The `meshable` test first used `rng.uniform` and tripped **its own "weak test" guard**:
only ~6.6% of a uniform draw is even `feasible_geom`, so 60 samples found five feasible
genomes and zero unmeshable ones — the sample could not see the gap M2a exists to report.
It now draws with the driver's own `latin_hypercube`, which is a third piece of the driver
under test: 60 stratified samples find 5 and 2, and 200 find 17 and 6 (seed 3: 15 and 4).
200 costs 0.1 s.

**All nine `make studies` drivers are now imported by the suite.** +0.42 s; 488 passed,
3 xfailed at `39dd96f`.

### §43 — 2026-08-22. THE GUARD REACHES ALL NINE, AND THE TEST FOR IT FIRST REPRODUCED THE DEFECT

§41's ranked item 3, and the other half of the exposure §41 closed for `study_contact`
alone. Every driver's `--out` defaults to the artifact `make studies` commits and no
fidelity flag changes that default, so the cheap invocation overwrites the gate's own
record with a weaker measurement — and the report still reads like the gate, because every
field is present and every verdict is computed.

#### One mechanism, nine judgements

`studies/_gate_guard.py` holds the part that is identical everywhere: compare `--out`
against the committed name, collect the reasons this run is not the gate, refuse by name.
The judgement is not shared — what degrades `study_mesh_quality` (fewer samples) has
nothing to do with what degrades `study_gradient` (a strain measure) — so each driver
passes its own `(condition, reason)` list. This is `tests/test_fem.py:298-301`'s argument
applied to a guard instead of a check, and it is the same reasoning §42 used on the M2a
duplicate: one definition, so a fix reaches all nine. **`study_contact` is retrofitted onto
the helper**, and its seven tests from `db4ab05` pass unchanged — which is what says the
retrofit preserved it rather than merely compiling.

| degraded by | drivers |
|---|---|
| `--quick` | all but `study_mesh_quality`, which has none |
| `--config` off the driver's own default | all but `study_beam_agreement` |
| `--genome` / `--elites` off the provenance chain | the six that read them |
| `--samples` BELOW the recipe's | `study_mesh_quality`, `study_wheel_mesh` |
| `--kinematics` non-linear | `study_gradient`, `study_contact` |
| partial `--sections`, non-default ladder | `study_stage3`, `study_contact` |
| `--no-plot` | all nine |

**`--no-plot` goes beyond the literal extension and is deliberate.** It refreshes the
`.json` and leaves the committed `.jpg` stale, breaking the header's "a study commit
carries its artifacts" the same way a degraded run breaks the gate. Only `make contact`
passes it, with an explicit `--out`. **`--seed` is deliberately NOT guarded**: a re-draw is
still the full gate, and requiring seed 0 would pin a particular random draw rather than a
fidelity.

#### The test that matters most is the PERMISSIVE one

A guard that fired on the recipe's own invocation would take `make studies` down — five
hours, nine drivers — and would do it at the END of each driver, since `make` sees only an
exit status. So the Makefile's exact `studies:` argv is asserted to pass for all nine.
Mutating one condition to the wrong default (`--samples != 2000` where the target passes
exactly 2000) fails that test, which is how it was checked rather than assumed.

Verified outside the suite too: both recipe invocations exit 0 and write their artifacts —
`study_mesh_quality.json` came back **bit-identical** for the third time — and
`study_gnl.py --quick` exits **2** with the guard's message. `make m8bi5`, `m8bi6`,
`m8bii1` and `contact` all pass explicit `--out` and are unaffected, pinned in both
directions.

#### THE LESSON, AND IT IS THE THIRD SELF-INFLICTED ONE THIS WEEK

**The first draft of the test reproduced the exact defect it was written to catch.** It
stopped execution only by having the guard wrapper raise — fine while every driver calls
the guard, and catastrophic the moment one does not, which is precisely the mutation the
tests exist to catch. Removing a driver's guard call to check the tests could fail meant
the wrapper never fired, `main()` ran the whole study, and it **overwrote
`studies/study_mesh_quality.json` and `.jpg`** and left two stray `_probe` files. Restored
from `39dd96f`.

The fixture now redirects every driver's module-level `HERE` to `tmp_path` — all nine write
through `os.path.join(HERE, args.out)` — and asserts the guard was reached. Re-run with a
guard call deleted outright: caught, and the artifact fingerprint is identical before and
after. **A test for an artifact-clobbering defect must not be able to clobber the
artifact**, and the general form is worth keeping: *a test that verifies a safety mechanism
must be safe when that mechanism is absent, because absent is the case it exists to
report.*

This is three in a row where the check found the flaw in my own work rather than in the
tree's — §42's mutation check caught a test that could not see a 1% error, §41's audit
caught the `--quick` exposure while arguing about something else, and this. The common
factor is that each was found by trying to make the new thing FAIL, not by running it.

#### The successors, ranked — REVISED 2026-08-22 AFTER §43

§41's list is now down to the items that were never cheap.

1. **`FILLET_PLAN.md`** — unchanged at the top for the eighth arc. Premise re-checked
   2026-08-21 (STEP 1 RECORD PART 5): the uncap flip did not move the fold, byte-identical
   capped and uncapped, because `uncap` is consumed in the junction block and the fold is
   in the spoke block. **The cheap way in does not exist.** Two routes, both real work: a
   dedicated fillet block with its own seam entries, or a generated spoke block. Its own
   PART 3 apparatus is what either gets checked against, and reconciling PART 3's
   surviving-radius table (0.20/0.10 recorded, 4.00/3.00/0.40/0.40 re-swept, criterion
   unrecorded) belongs there as its first act.
2. **G1's fourth revision** — derive `GATE_EPS_PLATEAU_REL` from the requirement or record
   that it should not be derived. §40 confirmed the gate blocks nothing.
3. **§32's successors 3 and 4** — §8's wall-floor economics under SVK. `study_stage3`
   territory, the 11021 s driver (§40).
4. **The rim tri-block**, still filed, still not binding (§37).

**`HUBSHARE_PLAN.md` remains off the list**, still blocked behind item 1 by its own Step 0
rule — now for the sixth arc.

**What this week's four sections did NOT do.** No threshold moved, nothing was promoted,
`best_solution.json` is untouched and still 2026-08-14, and every gate reads exactly what
it read on 2026-08-20. §40 through §43 bought a completed recipe, a closed false-green
exposure, one driver's first test coverage, and a retired premise on the top-ranked arc —
all of it plumbing around the measurements rather than new measurements.

### §44 — 2026-08-22. THE FILLET ARC'S TWO CONTESTED TABLES ARE RECONCILED: THEY WERE DIFFERENT CRITERIA, THE OLDER ONE IS RIGHT, AND THE NEWER ONE IS A GUARD THAT CANNOT SEE THE FOLD

§43's ranked item 1, taken at the point §43 said to take it — *"its own PART 3 apparatus is
what either route gets checked against, and reconciling PART 3's surviving-radius table
belongs there as its first act."*

`wheel_wheel.sector_blocks(..., fillet=)` has been the fillet arc's measuring instrument
since §34 and had **no test and no driver**. Its central number — the largest radius it
survives — was recorded twice, 0.20/0.10 mm (PART 3) against 4.00/3.00/0.40 (PART 5), and
filed open because neither criterion was written down and neither apparatus survived.
**Both rows now reproduce from one sweep, to the digit, and they were never measuring the
same thing.**

#### The criteria, and why the disagreement was 20x

| | criterion | coarse hub | medium hub |
|---|---|---|---|
| PART 3 | mixed-sign cells in the spoke block | 0.20 | 0.10 |
| PART 5 | does `build_wheel` raise | 4.00 | 0.40 |
| **new** | **`det J` at the Gauss points the assembly integrates** | **0.12-0.24** | **0.07-0.11** |

`_orient_elements` is a shoelace over each element's **four corners**. Every config is
`order=2`, so one Q9 element spans 2x2 cells and its five mid nodes take no part in that
sum: **a fold inside an element is invisible to the check that exists to catch folds.**
PART 5's row is that guard reading late, and it overstates what is usable by 10-20x.
PART 3's lands on the usable window's upper edge **exactly, at all four cells**. The
docstring is edited to say so; it had been carrying both rows as contested since §43.

#### The sharper answer neither row contained

**There is no usable interval `0 < R < R_max` — an arbitrarily small fillet folds too.**
The window has two edges and both are node allocation rather than geometry: above it
`k0 = clip(round((s_A - s0)/ds), 1, cap)` steps 1 -> 2 and the element straddling the arc's
end inverts (the geometry moves under 2% across that step); below it the same clamp's
lower bound drags the first element's mid-side node out of the middle half of its own
edge. Both scale with `ds`, which is the mechanism behind the one thing PART 3 and PART 5
already agreed on — the limit tightens under refinement.

#### AND THE GUARD'S BLIND SPOT IS NOT CONFINED TO THE OPT-IN PATH

Sampled over the gene box at `coarse`, 1000 geometrically feasible draws, in the committed
report: **177 of the 839 meshes `build_wheel` accepts (21.1%) have a non-positive `det J`
at a Gauss point**, and 4 of those also clear `study_mesh_quality`'s minSJ >= 0.2 floor —
which is corner-only for the same reason, so **both checks are blind in the same way at
once.** What actually covers the default path is `fold_margin > 0`, and it is not an
element check at all: it rejects the genome before the mesh exists, leaving 2 of 615.

**And it is the one constraint that cannot cover this arc.** `fold_margin` reads genes
0-11; `R_hub` and `R_rim` are 12 and 13. Pinned by moving both across their whole box and
finding the margin unchanged to the bit. **Strengthening the guard is deliberately NOT
done here** — it is a tree-wide change and this arc is not the place to make it — but it is
recorded in `_orient_elements`' docstring, because until then a `build_wheel` that returns
is not the same statement as a mesh that integrates.

#### What was built

`studies/study_fillet_fold.py` + `make fillet` (37.8 s, geometry and Jacobians, no field
solved, exits nonzero only on a self-check), its committed report, and
`tests/test_fillet_fold.py` — 20 tests on a construction that had none, including the two
zero-radius controls, the reconciliation of both tables, the criterion ORDERING, and the
blind spot pinned at a named radius.

**Nothing was promoted, `best_solution.json` is untouched and still 2026-08-14, the default
mesh is bit-identical, and no threshold moved.** Step 2 of `FILLET_PLAN.md` remains
unreachable until route 1 (a dedicated fillet block) or route 2 (a generated spoke block)
exists. What changed is that the instrument both will be judged against is re-runnable,
named and tested — and that "is this fillet mesh valid?" now has one criterion instead of
three that disagree by 20x.

#### The successors, ranked — REVISED 2026-08-22 AFTER §44

1. **`FILLET_PLAN.md` route 1 or route 2** — unchanged at the top for the ninth arc, and
   the errand that stood in front of it is done. A dedicated fillet block with its own
   seam entries (preferred) or a generated spoke block. **Both must be judged on `det J`
   at the Gauss points, not on `build_wheel` returning** — §44 is why.
2. **G1's fourth revision** — derive `GATE_EPS_PLATEAU_REL` from the requirement or record
   that it should not be derived. §40 confirmed the gate blocks nothing.
3. **§32's successors 3 and 4** — §8's wall-floor economics under SVK. `study_stage3`
   territory, the 11021 s driver (§40).
4. **The element-validity check itself** — NEW, and filed rather than ranked up: make the
   fold guard (and the optimizer's minSJ barrier) see sub-element folds. §44 measures the
   exposure at 21% of accepted meshes and shows `fold_margin` is what has been covering
   it. Cheap to state, not cheap to land: minSJ is differentiated by M8's barrier.
5. **The rim tri-block**, still filed, still not binding (§37).

**`HUBSHARE_PLAN.md` remains off the list**, still blocked behind item 1 by its own Step 0
rule — now for the seventh arc.

### §45 — 2026-08-22. THE FILLET ARC'S STEP 0 BASELINE DESCRIBED A MESH THE TREE STOPPED BUILDING FOUR DAYS LATER, AND THE COMMIT THAT BROKE IT EDITED THE TEST FILE NEXT TO IT

§44 built the fillet arc's apparatus. This is the check §44's own lesson implies: **PART 5
re-checked its prediction against §38's flip; nobody had asked the same question of Step
0.** `studies/study_corner_singularity.py` calls `build_wheel(genes, cfg)` bare, so it
takes `UNCAP_DEFAULT`. §38 flipped that on 2026-08-18. The committed artifact was from
2026-08-16.

`make corner` costs 8 s. It is re-run and the artifact is refreshed.

#### WHAT WAS WRONG IN THE COMMITTED RECORD

```
  corner       wedge deg            lambda_W           peak MPa at `fine`
  hub:P_t      321.10 -> 321.13     0.5032 -> 0.5032    96.22 ->  85.93
  hub:P_c      296.75 -> 268.08     0.5144 -> 0.5477   120.92 ->  66.77
  rim:P_t      321.33 -> 321.32     0.5031 -> 0.5031    75.40 ->  60.69
  rim:P_c      307.94 -> 271.02     0.5079 -> 0.5429   150.59 -> 108.57
```

The global peak was **27.9% too high** and two of four wedge angles were wrong by 28.7 and
36.9 degrees. §36 and §37 had already MEASURED the new global peaks (48.208 / 73.728 /
91.152 — reproduced here exactly), so the numbers were known; what was never done is
writing them into the artifact the tests and the plan file read.

**The `P_t` pair did not move** — 0.03 and 0.01 deg — because it is the PART's corner and
its wedge comes from the spoke geometry, not from a mesh option. That asymmetry is the
useful half and it is now pinned per family rather than as one band across all four.

#### THE RED IT SURFACED, AND WHY IT IS A FINDING RATHER THAN A TOLERANCE

Refreshing turned exactly one assertion red: `test_all_four_junction_corners_are_re_entrant`
required `0.50 <= lambda < 0.53` for every corner, and the two `P_c` corners now read
0.5477 and 0.5429. **That window was a capped-mesh property.** Uncapped, those corners are
barely re-entrant (268.1 and 271.0 deg against 296.8 and 307.9), so they are LESS singular
— the test was asserting "all four are nearly crack-like", which stopped being true at the
flip and kept passing only because the stale artifact still said so.

Updated with the measurement in the docstring: `P_t` pinned tight at 0.5031 ± 0.002 as a
property of the part, `P_c` pinned loosely as "re-entrant and singular" with a band that
holds both vintages, **because a number that moves when a model option moves must not be
asserted as though it were a property of the wheel.**

#### PART 4's FINDING SURVIVES; PART 4's RANKING DOES NOT

Re-measured by `argmax` over the whole Gauss field, to all twelve rotational copies of each
corner: **the peak is still on `rim:P_c`, 15-24 um away, at `coarse` and `medium`, under
linear and SVK solves and under both stress recoveries.** A fillet at `P_t` still cannot
deliver Step 2's headline.

But the capped ranking PART 4 drew its re-ranking from is gone. At `fine` it is now
`rim:P_c` 108.57 > **`hub:P_t` 85.93** > `hub:P_c` 66.77 > `rim:P_t` 60.69. §38's flip
fixed the hub artefact (§36: 28.71 deg of wedge error -> 0.01; stress down 45%), so **exactly one
artefact corner is left and it is the rim's** — the one §37 priced and filed as not
binding. The fillet's target is no longer the fourth-ranked corner; it is the second, and
the fastest-diverging of the four. That does not change the go/no-go, and it is the only
thing in four parts that argues up for routes 1 and 2.

*Also corrected:* PART 4 recorded that SVK moves the peak by 0.3%. On the uncapped mesh the
SOLVE moves it 4.3-4.5% and the SVK stress RECOVERY another 12-13%; PART 4 had the two
conflated in one column. The location is unmoved to the digit either way.

#### THE PROCESS DEFECT, WHICH IS THE PART WORTH KEEPING

**The uncap commit (`c416cb5`) edited `tests/test_corner_singularity.py`** — it repaired a
tie-break red in that file — **and left `studies/study_corner_singularity.json` beside it
untouched.** Nothing went red, because every test that reads the corner artifact reads the
same stale file. And `make studies` would not have caught it: this driver is one of the
cheap ones and is not on that recipe. The header's rule "a study commit carries its
artifacts" covers the driver changing; **nothing covered the DEFAULT the driver reads
changing**, which is the same class of defect as §25 (a promotion turning a control red)
one level further out.

`test_the_committed_report_describes_the_mesh_the_tree_BUILDS_TODAY` now rebuilds the
finest rung and compares wedge angles and element counts against the committed report.
Geometry only, well under a second, and it would have gone red on 2026-08-19.

#### THE AUDIT THIS IMPLIES, MEASURED BUT NOT DONE

Every committed `studies/*.json` was dated against the flip. **The nine `make studies` gate
artifacts are covered by §40's completed recipe run of 2026-08-20**, which is after the
flip — that RUN, not their commit dates, is the evidence: two of the nine still carry
2026-08-19 commit dates because §40 regenerated them byte-identically and there was nothing
to commit (§43 records `study_mesh_quality.json` coming back bit-identical "for the third
time"). **A commit date is not a run date, and dating artifacts by `git log` alone would
have mis-classified those two in both directions.** The pre-flip remainder is unaudited,
and one of them is read by a test in the same file as the artifact just refreshed: **`study_deflection_gci.json` (2026-08-16) describes the capped mesh**, and
`test_the_p_norm_the_optimizer_uses_diverges_far_more_slowly` now compares its slope
against a post-flip peak slope. The inequality is an order of magnitude wide so it survives
the mismatch; the docstring now says so rather than relying on it quietly. **`make gci` is
95 minutes and FILLET_PLAN.md forbids starting this arc with it**, so it is ranked, not run.

#### The successors, ranked — REVISED 2026-08-22 AFTER §45

1. **`FILLET_PLAN.md` route 1 or route 2** — unchanged at the top. Judge both on `det J` at
   the Gauss points (§44), and note §45: the corner the fillet reaches is now the
   second-ranked and fastest-diverging, while the peak still sits on the rim artefact.
2. **Re-run the pre-flip study artifacts, `study_deflection_gci.json` first.** NEW. The
   list is in this section; the nine gate artifacts are already clean. 95 minutes for the
   GCI ladder, minutes for the rest.
3. **G1's fourth revision** — derive `GATE_EPS_PLATEAU_REL` from the requirement or record
   that it should not be derived. §40 confirmed the gate blocks nothing.
4. **§32's successors 3 and 4** — §8's wall-floor economics under SVK.
5. **The element-validity check itself** (§44) — make the fold guard and the minSJ barrier
   see sub-element folds.
6. **The rim tri-block**, still filed (§37) — but §45 is the first section in which it is
   the *only* thing standing between the model and a peak that sits on a real corner.

**Nothing promoted, `best_solution.json` untouched, no threshold moved.** The refreshed
artifact changes no gate: every `make studies` verdict is computed from the nine, and this
driver is not one of them.

### §46 — 2026-08-22. THE SECOND CORNER'S "NEVER" WAS PRICED ON THE END CAP: IT FLIPS AT THE HUB, AND §37's REASON FOR SHELVING THE RIM TRI-BLOCK IS WRONG IN ONE MEASURABLE CLAUSE

§45 re-asked the uncap question of Step 0's baseline. This asks it of the other judgement
the fillet arc rests on. **FILLET_PLAN.md PART 2 ruled the second junction corner `P_c`
un-filletable — permanently, "and filleting it would be a mistake rather than merely hard"
— on a spoke-side leg of `t/2`. That leg was the END CAP, and §38 deleted it two days
later.** `make junction` costs 4 s. It now computes the pricing instead of leaving it to
arithmetic in a plan file, and `tests/test_junction_fit.py` pins it on a driver that had no
test.

#### THE RE-PRICED TABLE

```
  corner                    void    leg mm    T mm   T/leg   R_max    fits at shipped R
  hub  P_c  capped         62.82    0.7369   1.0867   1.47   0.450          NO   <- PART 2
  hub  P_c  AS BUILT       91.52    0.6600   0.6462   0.98   0.678          YES  <- flipped
  rim  P_c  capped         52.57    0.7156   6.0738   8.49   0.354          NO   <- PART 2
  rim  P_c  AS BUILT       89.49    0.5664   3.0270   5.34   0.561          NO
  rim  P_c  uncap=True    141.16    0.9114   1.0577   1.16   2.585          NO
```

`T = R/tan(void/2)`; `R_max = leg * tan(void/2)` is what the corner would accept as built.
**Both legs got SHORTER and both verdicts improved**, which is what identifies the void
angle as the term that moved: uncapping opens `P_c` from 63/53 deg to 92/89.

At the hub the shipped `R_hub` now fits — **by 2%, on a 0.66 mm stub that ends in a 117 deg
kink**, which is admissible on paper and not a fillet anyone should mesh. At the rim it
does not, by a factor of five. **The rim is the one that matters**: `rim:P_c` carries the
wheel's global peak (§34 Finding 4, re-measured in §45).

#### THE CLAUSE §37 GOT WRONG, AND WHY IT WAS NOT SLOPPY

§37 shelved the rim tri-block because it buys *"only rim corner fidelity — not convergence,
not the fillet, not a quotable peak"*. **On the faithful rim the corner's admissible radius
goes 0.561 mm -> 2.585, a factor of 4.6.** The tri-block does buy the fillet, at the one
corner that carries the peak.

§37 asked whether it unblocks the fillet arc's KNOWN blocker — the ruled spoke block at
`P_t` — and correctly answered no. The question nobody asked is whether it unblocks the
fillet at `P_c`, because PART 2 had already ruled that corner out forever, on numbers that
stopped being true two days after they were written. **The same shape as §45: a judgement
made on a mesh, quoted after the mesh moved.** Two in one day is a pattern about how this
tree's records age, not about either judgement.

#### THE HONEST OTHER HALF

**2.585 mm is still short of the shipped 3.0, by 14%.** `R_rim` is a gene and §22/§24
measured its 3.0 ceiling as a trap, so a design at `R_rim <= 2.58` is inside the box — but
**choosing a radius to make the mesh work is choosing the design to fit the model, and that
is backwards.** It is recorded as a finding, not proposed as a plan. And geometric
admissibility is not meshability: every radius in that table is far outside the
construction's usable window of 0.12-0.24 mm at `coarse` (§44).

#### THE CHAIN, NOW CLOSED AND EVERY LINK MEASURED

Step 2 wants a non-divergent peak; the peak is on `rim:P_c`; only a fillet removes a
singularity (§37 CHECK 1); `rim:P_c` refuses a fillet as built at 5.34x its leg; it accepts
one at 4.6x the radius on the faithful rim, which needs the tri-block; and `P_t` — the
corner routes 1 and 2 do reach — is now the second-ranked corner rather than the fourth
(§45). **The tri-block is no longer a fidelity nicety filed behind this arc; it is on the
only measured path to Step 2's success condition, and the fillet arc's two routes are
necessary but not sufficient.**

#### The successors, ranked — REVISED 2026-08-22 AFTER §46

1. **`FILLET_PLAN.md` route 1 or route 2** — still first, still the real work, and §45/§46
   sharpen what it is for: it addresses the second-ranked corner and it is necessary but
   not sufficient. Judge on `det J` at the Gauss points (§44).
2. **The rim tri-block — PROMOTED from filed.** §37's price stands (partial-edge seams,
   forced 1-element strips) and its stated payoff was wrong by the factor above. It is now
   the only measured route to the corner that carries the peak. **Re-read §37 before
   starting: the price has not changed, only the payoff.**
3. **Re-run the pre-flip study artifacts**, `study_deflection_gci.json` first (§45).
4. **G1's fourth revision** — §40 confirmed the gate blocks nothing.
5. **§32's successors 3 and 4** — §8's wall-floor economics under SVK.
6. **The element-validity check** (§44) — make the fold guard and the minSJ barrier see
   sub-element folds.

**Nothing promoted, `best_solution.json` untouched, no threshold moved.** `make junction`'s
change is additive: every pre-existing field in its artifact is byte-identical, checked
before the refresh.

### §47 — 2026-08-22. THE ARC'S RANKED ITEM 1 HAD A PREMISE AND IT IS FALSE: BOTH OF PART 3's ROUTES NAME THINGS THAT CANNOT BE BUILT, AND THE SPOKE BLOCK WAS NEVER THE BLOCKER

§44 and §46 ranked `FILLET_PLAN.md` route 1 or route 2 first, for the ninth arc. Taken at
last, and taken the way §45 and §46 were taken — **re-check the step's premise before
spending on the step.** Route 1's premise is that the region it names can be a block. It
cannot, and the measurement costs 42 s.

`studies/study_fillet_block.py` + `make filletblock` (geometry and Jacobians only, no
field solved), its committed report, and `tests/test_fillet_block.py` — 44 tests, every
one of which RE-MEASURES rather than reading the artifact, because §45's lesson is that a
committed artifact rots silently when every test that reads it reads the same stale file.

#### ROUTE 2 IS DEAD, AND NOT BY A TOLERANCE

What fails in the shipped `fillet=` construction is the angle at the moved corner —
3.601 deg (hub) / 8.524 (rim), reproduced on the current default. **Both curves that make
it are BOUNDARY curves of the spoke block**: the fillet arc sits on its flank edge and the
end cross-section is its end edge, so all three nodes carrying the angle are boundary
nodes. Route 2 is "a generated spoke block"; every generating scheme holds the boundary.
Measured, 2000 Winslow sweeps move the boundary by **exactly 0.0 mm** and return the angle
**bit-identical**. Pinned as an equality, because the claim is not that smoothing barely
helps.

*A separate correction falls out of it:* 3.601 is a SAMPLED angle and reads 10.543 at
`medium`. The angle between the two boundary curves is 19.800 / 12.432 and is
config-independent to 0.001 deg. Both are now reported.

#### ROUTE 1 AS WRITTEN NAMES A REGION WITH TWO ZERO-DEGREE CORNERS

PART 3 called `A - P_t - B` "the curvilinear triangle". A fillet is tangent to both legs,
so it meets each at zero angle and the region it adds is a **cusp sliver** — measured
interior angles **0.0000 deg at `B`** (exactly: both curves are circles and the tangency
is constructed, not solved) and **0.42-0.60 at `A`** (the spline flank's own curvature —
it stays inside a 0.2 deg band across a 30x span of radius, which a convergence residue
would not), against 38.06 / 38.89 at `P_t`. **No quad covers a 0 deg corner and a
tri-block invents none** — a tri-block subdivides a region's corners rather than adding to
them. Two ways of closing the region into a quad were built and both fold, both get WORSE
under refinement (3→9 and 6→15 mixed cells at the coarse hub), and an elliptic interior
rescues neither.

#### AND THE SPOKE BLOCK WAS NEVER THE BLOCKER

PART 3's standing headline — "the spoke block is ruled, and a fillet 1.3-5.8x the wall
cannot be absorbed by ruling" — is a statement about **where the fillet was put**. Take
the arc off the flank edge, end the spoke at the tangent station `s_A`, change nothing
else, and the same ruled block is clean — zero mixed cells, zero non-positive Gauss points
— **at every radius from 0.05 to 4.00 mm at both configs**, where the shipped construction
has a usable window of 0.12-0.24. At `R = 0` it is the default block to the bit. The fold
does not vanish; it moves, whole, into whichever block then carries the fillet.

#### WHAT DOES MESH, AND WHAT IT COSTS

A **boundary-layer block whose four corners are all OFF the tangencies**: the fillet arc
as its free edge, that arc offset into the material as its inner edge, the spoke's end
cross-section at `s_A` at one end, a radial cut at `B` at the other. **Min scaled Jacobian
0.91-1.00 with zero non-positive Gauss points at every radius in the gene box**, both
junctions, `coarse` and `medium`, 1x/2x/4x — against `MIN_SJ_TARGET` of 0.2. The first
filleted block in this arc that meshes at the radii that ship.

Its price is the cut at `B`, and the price is not optional: at `B` the material on the
free-surface side has zero thickness, so a block that stops at the ring circle degenerates
and one that crosses it does not. **The ring circle stops being the junction/collar
interface over the fillet's footprint** — a notch of 7.00 deg at the hub (cut 0.711 mm,
14% of the collar's depth) and 2.94 at the rim (0.651 mm, 43% of the band's), and the
spoke gives up 3.7 / 16.6 stations at `coarse`. So what remains is a **re-cut of the
neighbours**, not an eighth block: notch the ring block, split the spoke at `s_A`, extend
`_seam_table`.

#### A FINDING ABOUT `make junction`, FOUND ON THE WAY

Its `void_deg` at `P_t` — what §46's re-pricing table is built on — is the angle to the
spoke block's **second flank node**, not to the flank's tangent. Reproduced to the digit
and now reported next to the tangent: 38.8606 vs 38.0556 at the hub, 39.4546 vs 38.8886 at
the rim. The chord is **0.8 deg optimistic** because the flank is a spline. **No §46
verdict moves:** both `P_t` rows clear by 5-20x, and the `P_c` rows are unaffected because
under `uncap` that corner's leg is a straight continuation whose chord and tangent are the
same direction exactly.

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, the default mesh
bit-identical, no threshold moved.** No candidate block is wired into `build_wheel`;
`sector_blocks` changed only in its comment block, which had been carrying PART 3's two
routes as the way forward.

#### The successors, ranked — REVISED 2026-08-22 AFTER §47

1. **The boundary-layer fillet block, wired into `sector_blocks`** — route 1, corrected.
   The block itself is measured and meshes across the box; what is left is the re-cut it
   forces: notch the collar/band over the fillet's footprint, split the spoke at `s_A`,
   and extend `_seam_table` (the fillet block adds one whole-edge seam per junction; the
   notch is what may not stay whole-edge). **Judge it on `det J` at the Gauss points
   (§44), and on `test_axle_drop_is_exactly_12_fold_periodic`.** This is the first time
   this item has had a construction behind it rather than two names.
2. **The rim tri-block** — unchanged from §46. §37's price stands, its stated payoff was
   wrong by 4.6x, and it is still the only measured route to `rim:P_c`, which still
   carries the peak. Nothing in §47 touches that chain: `P_c`'s refusal is geometric
   (5.34x its leg), not a meshing question.
3. **Re-run the pre-flip study artifacts**, `study_deflection_gci.json` first (§45).
4. **G1's fourth revision** — §40 confirmed the gate blocks nothing.
5. **§32's successors 3 and 4** — §8's wall-floor economics under SVK.
6. **The element-validity check** (§44) — make the fold guard and the minSJ barrier see
   sub-element folds.

**`HUBSHARE_PLAN.md` remains off the list**, still blocked behind item 1 by its own Step 0
rule — now for the eighth arc.

### §48 — 2026-08-23. THE FILLETED SECTOR CLOSES: ELEVEN BLOCKS, FOURTEEN WHOLE-EDGE SEAMS, ACROSS THE BOX. AND PART 9's OWN CUT IS PROVED UNCLOSEABLE — THE OFFSET LANDS TANGENT

§47 promoted a boundary-layer block on the strength of one block meshing. This is the
half that a single block cannot answer, and it is ranked item **1a**: build EVERY block
and EVERY seam of the filleted sector, geometry only, and judge on `det J` at the Gauss
points AND on the seams closing. `make filletblock` (~85 s, was ~42), its regenerated
report, and `tests/test_fillet_block.py` at **66 tests, was 44** — every one of which
RE-MEASURES.

#### IT CLOSES

Eleven blocks — the trimmed spoke, two fillet blocks and one junction per junction, and
each ring's weld and free re-cut — and fourteen seams, **every one whole-edge**. At the
shipped radii: worst min scaled Jacobian **0.3594** (`coarse`) / **0.3623** (`medium`),
worst seam gap **1.4e-14 mm**. Across the admissible gene box, 48 cells at each config:
**48/48 valid AND closed**, worst 0.3569 against `MIN_SJ_TARGET`'s 0.2, and the same
worst seam gap. The unfilleted control, measured by the same instrument in the same run,
is **0.7827** — so the blocking costs a factor of **2.2** in the worst block and still
clears the barrier by 1.8x.

The fillet block is split at `N`, the point where its inner edge crosses the ring circle,
because above `N` that edge's partner is the junction block and below it the ring: one
edge with two partners is the partial-edge seam the arc had already refused. **Four extra
blocks; the seam table gains eight entries and loses two — each ring's `weld.i1 ~ free.i0`
goes, because the fillet block now separates them and they meet at a single POINT — and
it is written in `_seam_table`'s own shape** — only two of the
fourteen `reverse` flags depend on the genome, the same two the shipped table has.

#### AND PART 9's SHALLOW CUT IS PROVED UNCLOSEABLE, WHICH IS THE FINDING

PART 9 priced the re-cut as *"a notch of 7.003 deg at the hub and 2.935 at the rim"*.
**That notch has no whole-edge blocking at any depth short of the ring's far side.** A cut
stopping at depth `d` gives the free block's left edge two partners; splitting the free
block to fix it gives its own right edge two partners, so the split propagates round the
ring; and the block it propagates into is a **triangle**, because the fillet block's inner
edge is a concentric offset of an arc TANGENT to the ring circle and is therefore tangent
to every circle concentric with it. **Measured: that landing closes at 12.864 deg at the
hub and 4.381 at the rim — scaled Jacobians 0.2226 and 0.0764, the second below the
barrier outright.** Taking the cut to the bore (hub) / the tyre surface (rim) splits the
ring into exactly two quads and terminates.

#### TWO PRICES, NEITHER OF THEM THE ONE THAT WAS EXPECTED

**The ring's radial node count is forced to `n_thick`.** The cut carries `n_thick` nodes
and it is the free block's left edge, whose opposite edge is the next sector's weld
block's — so `n_collar_r` and `n_rim_r` are **not used by the filleted blocking at all**
(7 -> 9 at `coarse`, 9 -> 13 at `medium`). §47 anticipated a node-count coupling; this is
it, and it is why `fillet=None` needs its own path rather than a flag.

**And `R_hub` is now bounded by the SECTOR, not by the block.** §47 measured the block
clean from 0.05 to 4.00 mm and **it still is** — what runs out first is the ring's free
block, at **`R_hub` = 3.1297 mm**, where the fillet's tangent point sweeps past the next
sector's corner. Bisected, because it is the number a gene bound would be written against,
and `R_hub`'s box runs to 4.0. The rim never binds.

#### THE RADIUS BOX IS NOT THE GENE BOX, AND MEASURING THE DIFFERENCE FOUND A BUG AND A LIMIT

All of the above sweeps radii **at one genome**. `flank_orientation` is a property of the
centreline and its own docstring records that **only 16 of 60 feasible genomes share the
shipped one's `(+1, +1)`** — so a blocking measured there has been measured on a quarter of
the design space, and §47's "the whole gene box" meant the radii. Sixteen freshly drawn
feasible genomes, four per orientation, say two things.

**A bug the radius sweep could not reach.** `sector_blocks` lays both rings out in
increasing theta whatever the genome does, so that "the next sector" is always `k + 1`;
this blocking lays each ring from `theta_Q` toward the fillet, so the sector-closing seam
runs to `k + dirn`. Written `dk = +1` it closes for the shipped genome and **misses by a
whole sector for a flipped one — 12.6 mm at the hub, 50.0 at the rim.** Fixed, and pinned
both ways: the seam closes at `dk = dirn` and forcing `+1` re-opens it.

**And the honest scope of everything above.** With that fixed, **every built cell closes at
all four orientations, to 1.4e-14 mm** — but **6 of the 16 genomes refuse the fillet at
their own shipped radii** (always the same refusal: the tangent point has passed the next
sector's corner), and of the 10 that build, **only 4 clear `MIN_SJ_TARGET`** and one folds
outright, in the TRIMMED SPOKE. **The blocking is fit for STEP 2 and is not yet fit for the
optimizer**, which are different requirements rather than degrees of one: Step 2 needs one
filleted mesh at the shipped genome; the optimizer sweeps genomes. Nothing above is
retracted, and there is now a test whose whole job is to stop "48/48 across the box" being
quoted as "works everywhere".

#### THE ONE FREE PARAMETER, AND THE ZERO THAT WOULD HAVE BEEN TAKEN BY DEFAULT

The fillet block's inner edge is the arc offset by a cubic-Hermite width, then a radial
dive. Two constants, re-derived every run against the worst block over the whole box, on a
grid the report prints in full: `LAYER_ENTRY_SLOPE` **-0.45**, `LAYER_END_OFFSET`
**1.60**. The surface is a ridge — entry -0.35..-0.60 with end 1.4-1.8 all sit within 0.02
of the maximum — so the choice is on a plateau rather than tuned.

**The default anyone would have taken is the one that fails.** `entry = 0` is the plain
boundary layer, and it leaves the end cross-section TANGENT to the far flank — which is
the junction block's own top edge. Three blocks meet at that node and 180 degrees has to
be shared; a layer that takes the full wall takes all of it, and the junction block
becomes a cusp at **min scaled Jacobian 0.0400, against the chosen slope's 0.4272**. And the dive is radial because an offset
carried to the ring's full depth is a spiral of radius `R + w` about the arc's centre: at
the gene box's own floor, `R_hub = 0.4`, it folds the weld block. Both measured before the
construction was chosen, both under test.

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, the default mesh
bit-identical, no threshold moved, and `wheel_wheel.py` not edited at all** — every block
is built inside the study from its primitives.

#### The successors, ranked — REVISED 2026-08-23 AFTER §48

1. **STEP 1b: wire it into `sector_blocks`.** For the first time this item has no unknown
   geometry in it — the blocking is measured, the seam table is written in
   `_seam_table`'s shape and under test, and the node-count coupling is named.
   `sector_blocks` gains four blocks and a re-cut ring under `fillet=`; `_seam_table`
   gains eight entries and loses two; `BLOCK_ORDER`, `BLOCK_REGION`, `_edge_sets`,
   `_node_sets` and `modelled_area_reference` follow the eleven names. **The ring blocks
   keep the SHIPPED radial order per ring** — `_edge_sets` names `hub_tie`/`rim_outer`/
   `rim_inner_free` by side, and laying both rings out the same way round would move
   three boundary sets with nothing going red; §48 pins that on the coordinates. **`fillet=None`
   must stay the default and stay bit-identical.** DONE when
   `build_wheel(genes, cfg, fillet=True)` has zero non-positive Gauss points at `coarse`
   and `medium`, `check_seams` passes, and
   `test_axle_drop_is_exactly_12_fold_periodic` holds to 1e-10 on the FILLETED mesh.
   **FILLET_PLAN Step 2 becomes reachable the moment it lands** — and Step 2 deserves its
   own session, because it re-runs `make corner` on a filleted mesh and re-ranks the arc.
   **Scope, from §48's genome sweep: `fillet=True` lands as a MEASUREMENT INSTRUMENT for
   one genome. It must NOT be wired into `wheel_objective` or the GA** — 6 of 16 feasible
   genomes refuse it outright and 6 of the 10 that build sit under the barrier.
2. **Make the filleted blocking genome-robust** — NEW, and it is what stands between
   `fillet=True` and the optimizer. Two named causes, both measured in §48: the sector-fit
   refusal (the hub fillet's tangent point passing the next sector's corner, which is a
   bound on `R_hub` that depends on the centreline) and the barrier (the trimmed spoke and
   `<j>_ring_free` are the two blocks that go under). Ranked below 1b because Step 2 does
   not need it and above the tri-block because nothing else can use the fillet until it is
   done.
3. **The rim tri-block, RE-PRICED against §48 rather than against §37.** §37 priced it as
   partial-edge seams and §46 raised its payoff 4.6x. §48 changes the third term: a block
   CAN cross a ring circle with whole-edge seams, but only by carrying the cut to the
   ring's far boundary, and the price of that is a forced radial node count and a factor
   of 2.2 in the worst block. Re-price before spending; the re-pricing is itself a unit.
4. **Re-run the pre-flip study artifacts**, `study_deflection_gci.json` first — §45's
   audit list is the rest, and §45 measured it without doing it.
5. **G1's fourth revision** — §40 confirmed the gate blocks nothing.
6. **§32's successors 3 and 4** — §8's wall-floor economics under SVK.
7. **The element-validity check** (§44) — make the fold guard and the minSJ barrier see
   sub-element folds.

**`HUBSHARE_PLAN.md` remains off the list**, blocked behind item 1 by its own Step 0 rule
— now for the ninth arc.

### §49 — 2026-08-23. THE GCI LADDER IS RE-RUN POST-FLIP. §29's CALL SURVIVES AND EVERY NUMBER UNDER IT MOVED: THE WHEEL IS 4.5% STIFFER AND ITS DEFLECTION ERROR CHANGED SIGN

§45 ranked this and did not run it. The premise was checked before the 95 minutes were
spent rather than after, and it held three ways. `studies/study_deflection_gci.py` calls
`build_wheel(genes, cfg)` bare at line 117, so it takes `UNCAP_DEFAULT`, which §38 flipped
on **2026-08-18**. The artifact's **run** is 2026-08-14 — the Makefile's own recipe note,
not its 2026-08-16 commit date, and §45's rule that a commit date is not a run date cuts
both ways here. And the flip moves the shipped mesh's coordinates by **0.3366 mm** at
`smoke` and `coarse` alike while leaving the element and node counts **identical**, which
is exactly the shape of staleness that a count-based freshness check cannot see —
`test_mesh_counts_come_from_the_wheel_that_was_actually_solved` passes on the stale file.

`make gci` re-run in full — 8 phases, both kinematics, `smoke,coarse,medium,fine`,
`--workers 0`, **5843 s**.

#### THE VERDICT IS UNCHANGED, AND IT IS THE ONLY THING THAT IS

```
                          2026-08-14 (capped)      2026-08-23 (uncapped)
  svk  axle drop @fine       2.01274 mm               1.92328 mm      -4.45%
  svk  deflection error      +0.637 %                 -3.836 %
  svk  extrapolated          2.05700 mm  (+2.850%)    1.94306 mm  (-2.847%)
  svk  observed order p      0.6379                   0.7947
  svk  GCI(fine)             2.749 %                  1.286 %
  lin  axle drop @fine       1.70683 mm               1.63929 mm      -3.96%
  lin  extrapolated          1.73532 mm  (-13.234%)   1.65500 mm  (-17.250%)
  lin  observed order p      0.7440                   0.8100
  lin  GCI(fine)             2.087 %                  1.198 %
```

**§29's call stands**: GCI(fine) is still far wider than the ±0.3% band — 4.3x rather than
9.2x — the ladder is still monotone, and the gate is still undecidable under every
definition of `h` the study tries (p in [0.182, 0.847], GCI in [1.150%, 10.731%]).

**But the wheel it describes is a different wheel.** Removing the end cap stiffened it by
**4.5% under SVK and 4.0% under linear**, and the central estimate of the deflection error
**changed sign**: the design used to extrapolate to +2.85% of the 2.0 mm target and now
extrapolates to −2.85%. It was over-deflecting; it is now under-deflecting, by the same
margin to three figures. Nothing in this project had that number post-flip until today.

**And the convergence got better, not worse.** GCI halves and `p` rises at both
kinematics. §36 attributed 28.7 degrees of hub wedge error to the cap; a mesh whose
corners are the part's converges more like the part.

#### THE MISMATCH §45 FLAGGED AND DECLINED TO RELY ON IS GONE

`test_the_p_norm_the_optimizer_uses_diverges_far_more_slowly` compares a slope taken from
this artifact against one taken from `study_corner_singularity.json`, and §45 recorded in
that test's own docstring that the two came from **different mesh vintages** and that the
inequality survived only because it is an order of magnitude wide. Both sides are now
post-flip. The measured slope moves **−0.0441 → −0.0262** against a peak slope of −0.4695,
so the margin widened from 10.6x to 17.9x, and the docstring no longer has to warn about
its own inputs.

#### WHAT THIS DOES NOT DO

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, no threshold
moved, no gate verdict changed.** This driver is not on `make studies`, so no `make
studies` verdict is computed from it. And it is **one** artifact off §45's list: the nine
`make studies` gate artifacts were already covered by §40's post-flip run, and the rest of
the pre-flip remainder is still unaudited and still ranked.

#### The successors, ranked — REVISED 2026-08-23 AFTER §49

1. **STEP 1b: wire the filleted blocking into `sector_blocks`** — unchanged from §48, and
   §48's scope note travels with it: `fillet=True` lands as a measurement instrument for
   one genome, not as an optimizer path.
2. **Make the filleted blocking genome-robust** (§48).
3. **The rim tri-block, re-priced against §48** (§46's payoff, §37's price, §48's new
   third term).
4. **The REST of §45's audit list.** `study_deflection_gci.json` is done. What is left is
   the audit itself — which committed `studies/*.json` predate the flip AND come from a
   driver that builds a wheel on the bare default — and §45's rule that a commit date is
   not a run date means it cannot be answered from `git log`. That makes it a driver with
   an artifact and a test, not a table in a plan file.
5. **G1's fourth revision** — §40 confirmed the gate blocks nothing.
6. **§32's successors 3 and 4** — §8's wall-floor economics under SVK.
7. **The element-validity check** (§44).

### §50 — 2026-08-23. STEP 1b LANDS: `build_wheel(fillet=True)` IS AN ELEVEN-BLOCK MESH THAT INTEGRATES, `fillet=None` IS BIT-IDENTICAL, AND THE PORT'S ONE BUG WAS CAUGHT BY §48's OWN NUMBERS FAILING TO REPRODUCE

§48 measured the blocking and called what was left a wiring job. This is it, and it is
ranked item 1. `wheel_wheel` gained the construction, a filleted seam table, and per-mesh
block order, block regions and boundary sets; `tests/test_filleted_mesh.py` is new at 19
tests, and `tests/test_fillet_block.py` and `tests/test_fillet_fold.py` moved onto the
module's copy.

#### EVERY ACCEPTANCE CRITERION, MEASURED

```
  config   nodes            elements        min scaled J      seam error    non-positive
                                            (assembled)                     Gauss points
  coarse   21012 -> 26196   4704 -> 5952    0.7822 -> 0.3517   3.06e-14 mm       0
  medium   53124 -> 66468  12288 -> 15552   0.7826 -> 0.3575   3.24e-14 mm       0
```

and `test_axle_drop_is_exactly_12_fold_periodic` — the check FILLET_PLAN Step 1 names,
and the one that has already caught a real bug in this tree — **holds on the FILLETED mesh
through a real solve: 1.016e-11 at phase 0, 7.492e-12 at phase 7, against 1e-10.** The worst block is
`rim_ring_free` at both configs and it clears `MIN_SJ_TARGET` by 1.8x.

**And `fillet=None` is bit-identical** — coordinates, connectivity, all three node sets,
all three edge sets and the seam error, hashed at `smoke`, `coarse` and `medium` against
the previous commit's build. That is the half that could have gone wrong silently: the
block order and the boundary sets are now chosen per mesh rather than being module
constants.

#### THE PORT HAD ONE BUG AND §48's OWN NUMBERS ARE WHAT FOUND IT

`Q`, the ring blocks' other corner, follows `uncap` — it is where the FAR FLANK crosses
the ring circle, not the centreline endpoint, and `sector_blocks` has read it that way
since §38. The first port took the centreline endpoint unconditionally. **Nothing raised.**
What caught it is that the study's committed numbers stopped reproducing: the sector-fit
limit moved **3.1297 -> 3.4836 mm** and the worst block **0.3594 -> 0.3592**. Both
reproduce exactly with the fix.

*It has a second witness, and it is a cross-check three arcs old.* The filleted mesh
models **+8.7625% (coarse) / +8.6965% (medium)** more area than the unfilleted one,
against **§24's 8.77%** for the fillets' share of the part — measured on the CAD solid, by
mass, by a completely different computation. With the bug in place it read 8.06%. Two
independent paths landing within 0.01 of a point is not proof — a 2-D area fraction equals
a mass fraction only for a uniform extrusion, which this part is — but it is worth more
than either alone, and it is the first time this arc has had an external check on the
geometry it adds.

#### THE TWO THINGS THAT NOW REFUSE RATHER THAN ANSWER

**`mesh_coords` and `coord_fn` refuse a filleted mesh.** They rebuild the sector without
`fillet` and index it with `mesh.owners`; a filleted mesh has 26196 owners against 21012,
so the silent answer was another mesh's coordinates gathered through this one's index — a
*plausible* wrong number, which is the worst kind. This is also the mechanical guarantee
behind §48's scope note: the filleted mesh cannot reach the optimizer through the
differentiable path even by accident.

**`area_report` withholds its reference for a filleted mesh**, because
`modelled_area_reference` models the unfilleted region and `error_vs_modelled` would book
the fillets' 8.76% as a discretisation residual against a reference that is otherwise good
to 2e-4. Making that reference fillet-aware is not a closed form — the fillet's legs are a
spline and a circle, not two straight lines — and it is ranked below.
[DONE — §86 (2026-08-27). The wedge formula indeed does not apply and nothing needed it:
the added region is one curvilinear triangle per junction, integrated exactly by Green's
theorem. Only the STEP half is withheld now.]

#### AND PART 3's CONSTRUCTION IS KEPT ON PURPOSE

`fillet_blocking="spoke"` still builds the arc-on-the-flank-edge geometry §47 retired,
**because `make fillet` measures it**: PART 6's usable window of 0.12-0.24 mm is a
statement about that geometry, and deleting it would leave the table in the plan file with
nothing behind it. Its artifact came back identical apart from its wall-clock field and is
not re-committed.

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, no threshold
moved.** `fillet=` is opt-in and nothing in the tree passes it except this arc's own
drivers and tests.

#### The successors, ranked — REVISED 2026-08-23 AFTER §50

1. **FILLET_PLAN Step 2 — REACHABLE FOR THE FIRST TIME IN THE ARC.** Re-run `make corner`
   on a filleted mesh and ask whether the junction corner's peak still diverges under
   refinement once the corner is rounded. **It deserves its own session**: it re-ranks the
   arc, and PART 8's chain says the peak sits on `rim:P_c`, which this fillet does not
   reach — so the honest prior is that Step 2 answers "the corner the fillet reaches stops
   diverging and the peak does not move", and that has to be measured rather than assumed.
2. **Make the filleted blocking genome-robust** (§48) — 6 of 16 feasible genomes refuse it
   and 6 of the 10 that build sit under the barrier. Until this is done `fillet=True` is a
   measurement instrument and `mesh_coords` refuses it on purpose.
3. **The rim tri-block, re-priced against §48** — §46's payoff (4.6x on `rim:P_c`), §37's
   price, and §48's third term (a ring circle CAN be crossed whole-edge, at the cost of a
   forced radial node count and a factor of 2.2 in the worst block).
4. **Make `modelled_area_reference` fillet-aware** — NEW, small, and named by §50. The
   region is the unfilleted one, so a filleted mesh has no area cross-check at all today.
   Not a closed form; the legs are a spline and a circle.
5. **The REST of §45's audit list** (§49) — which committed `studies/*.json` predate the
   flip AND come from a driver that builds a wheel on the bare default. A driver with an
   artifact and a test, not a table in a plan file.
6. **G1's fourth revision** — §40 confirmed the gate blocks nothing.
7. **§32's successors 3 and 4** — §8's wall-floor economics under SVK.
8. **The element-validity check** (§44) — and it is worth more now than it was: the
   filleted mesh is the first construction in this tree whose validity the corner-only
   checks were never going to be able to judge.

### §51 — 2026-08-23. THE RE-CUT DOES NOT RESCUE THE FAITHFUL RIM AND MAKES IT WORSE, SO THE TRI-BLOCK KEEPS ITS PLACE — AND A PROBE SAYS §37 PRICED IT WRONG IN BOTH CLAUSES

§50 ranked the rim tri-block third and said to re-price it against §48 rather than
against §37. The first half of that re-pricing is cheap, is a question about the
construction that now ships, and is measured here. The second half is a probe and is
recorded AS a probe.

#### THE CHEAP HALF, MEASURED AND UNDER TEST

§46 promoted the tri-block because on the FAITHFUL rim — `uncap` blend 0.0, the geometry
it exists to make buildable — `rim:P_c`'s admissible fillet radius goes 0.561 -> 2.585 mm,
and that corner carries the wheel's global peak. The obvious hope after §50 is that the
re-cut which fixed `P_t` also fixes blend 0.0 and retires the tri-block with it.

**It does not, and it makes it worse.**

```
  config   rim blend   unfilleted worst   filleted worst   worst block
  coarse       1.00        0.782735          0.359414      rim_ring_free
  coarse       0.50        0.449821          0.359548      rim_ring_free
  coarse       0.25        0.222789          0.215146      rim_junction
  coarse       0.00        0.008176          0.000343      rim_junction
  medium       0.00        0.008251          0.003334      rim_junction
```

The unfilleted column is the control and it lands where §37 did: 0.008176 by the block
instrument here, 0.007208 by `quality_report` on the ASSEMBLED mesh, which is §37's
committed number to the digit. **What collapses at blend 0.0 is a
corner opening to 180 degrees** — the junction block's corner at `far_end`, where the
uncap edge becomes the flank's own straight continuation — measured at **179.35 deg** at
`coarse` against 128.12 at the shipped blend. The filleted junction block is SHORTER, so
the straight corner dominates more of it. Pinned as the ORDERING rather than as values, so
a construction that ever did fix it would go red and reopen the ranking.

#### THE PROBE, WHICH IS NOT A MEASUREMENT AND IS FILED AS ONE

§37 priced the tri-block on two clauses and a scratch probe says both are wrong. **It is
recorded here as a probe — no driver, no artifact, no test — precisely so that nobody
quotes it as this project's other numbers may be quoted.** What it would take to make it a
measurement is named at the end.

> **SUPERSEDED 2026-08-23 BY §53.** The unit named at the end of this subsection was built.
> Both clauses are retired and the numbers below are replaced by measured ones — the
> partition meshes at **0.626 / 0.582**, not ~0.25, and its factor over the block it
> replaces is **77x**, not 30x. **Quote §53, not this.** And what stopped the tri-block was
> neither of the two clauses this probe argues about.

**Clause 1, "it needs PARTIAL-EDGE SEAMS".** True only if the neighbours may not be split,
and §48 is the whole argument that they may: the Y-partition halves the ring arc and the
end cross-section, so split `rim_band_weld` in theta and split the `spoke` along a j-line,
which cascades once into the hub junction block and stops. The probe's arithmetic says
7 blocks -> 12, whole-edge throughout. **Not built.**

**Clause 2, "forced 1-element strips".** §37's algebra is right and its inputs are not.
Writing the three sides as A (arc, `n_weld`), B (free) and C (cross-section, `n_thick`),
the Y-partition forces `a1 = b2`, `a2 = c1`, `c2 = b1`, so `a1 = (A + B − C)/2`. §37 took
**B = 8** order-2 elements and got 7/3/1 — the strip. **B is the FREE side and its count is
free**; at B = 10, the same count the arc has, the splits are 8/2, 2/8, 2/2 and there is no
strip at all. The same at `medium`: A = 16, C = 6, B = 16 gives 13/3, 3/13, 3/3.

**And the three quads mesh.** Built in a scratch probe at both configs, swept over B and
over the interior point: worst min scaled Jacobian **~0.25**, zero non-positive Gauss
points, against the un-partitioned block's **0.0082** by the same instrument — a factor of
**30** in the number that shelved it. The interior point was a weighted centroid and nothing was smoothed, so
0.25 is a floor rather than an estimate.

**What would make this a measurement**: the driver §48 got — build all twelve blocks and
all their seams, sweep the free count and the interior point, and report validity and
seam closure at both configs. That is the unit, and it is ranked 2 below.

#### WHAT IS UNCHANGED

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, no threshold
moved, and the default mesh bit-identical.** `uncap`'s default is unchanged at
`(True, 1.0)`; blend 0.0 is swept, never adopted.

#### The successors, ranked — REVISED 2026-08-23 AFTER §51

1. **FILLET_PLAN Step 2** — reachable since §50 and unchanged at the top. Its own session.
2. **The rim tri-block, BUILT** — promoted from "re-price" to "build", because the
   re-pricing came back in its favour on both of §37's clauses and the probe meshes at 30x
   the number that shelved it. §48's driver is the template. **Do not quote §51's probe
   numbers until this exists.**
3. **Make the filleted blocking genome-robust** (§48) — 6 of 16 feasible genomes refuse it.
4. **Make `modelled_area_reference` fillet-aware** (§50).
5. **The REST of §45's audit list** (§49).
6. **G1's fourth revision** — §40 confirmed the gate blocks nothing.
7. **§32's successors 3 and 4** — §8's wall-floor economics under SVK.
8. **The element-validity check** (§44).

---

### §52 — 2026-08-23. STEP 2 IS RUN. THE FILLET DELETES THE SINGULARITY AT BOTH CORNERS IT REACHES, ITS HEADLINE IS STILL BLOCKED ON `rim:P_c` EXACTLY AS §46's CHAIN PREDICTED — AND THE UNFILLETED WHEEL'S AXLE DROP IS 61% HIGHER THAN THE FILLETED ONE'S

Ranked item 1 since §50. `studies/study_corner_singularity.py` grew a `--fillet` flag —
one driver, so the before and the after are one instrument — and `make corner-fillet`
(22 s) writes `studies/study_corner_singularity_fillet.json`. FILLET_PLAN.md PART 12 has
all of it.

#### THE HEADLINE IS NOT DELIVERED AND THAT IS THE PREDICTION ARRIVING

FILLET_PLAN Step 2's claim to test is *"the peak stress stops diverging — this is the one
that unlocks quoting a max"*. **It does not.** The wheel's global maximum is on `rim:P_c`
at every rung from `coarse` up — located by `argmax` over the whole Gauss field, 25.0 um
away at `fine`, by the driver now rather than by hand in a plan file — and its successive
differences hold a ratio of **+1.264**.

`rim:P_c` is the END CAP's corner. This fillet is tangent to `P_t`. That is §46's chain,
end to end, and **its wedge is unmoved across the re-cut — 271.02 deg unfilleted against
270.85 filleted** — which is what licenses comparing the two ladders' `P_c` columns at all.
Its magnitude fell 45% (108.57 -> 59.31 MPa at `fine`) because the whole wheel got
stiffer; **the claim is "still diverges", not "diverges faster"**, because the filleted
ring's radial count comes from `n_thick` and the near-corner mesh is a different mesh.

#### AND AT `P_t` THE SINGULARITY IS GONE, NOT REDUCED

`P_t` is not a corner on the filleted body — it is a point in the material's INTERIOR
(360.00 deg, four incident elements, 104-135 um from the nearest node):

```
  probe                     peak MPa up the ladder             d(N)/d(N-1)   verdict
  hub:P_t  unfilleted    16.71   50.03   66.59   85.93            +1.168     diverges
  hub:P_t  FILLETED       5.37    5.49    5.49    5.51            (noise)    settled
  rim:P_t  unfilleted    12.11   37.46   48.45   60.69            +1.113     diverges
  rim:P_t  FILLETED       3.81    3.59    3.96    3.98            +0.048     settling
```

**The successive differences are the instrument, not the log-log slope**, which was
written for a ladder where every probe was singular. On the unfilleted ladder all four
corners hold their ratio at **0.999 or above**; the filleted `P_t` pair's last increments
are 0.014 and 0.018 MPa on values of 5.5 and 4.0.

And the fillet's peak is a NUMBER, taken over the whole arc rather than at sampled points —
a tube of fixed radius about the analytic arc, so the region does not move with the mesh:
**hub 28.85 / 33.11 / 34.94 / 35.86 MPa** (ratios 0.430, 0.507, geometric tail -> **36.8**)
and **rim 12.90 / 15.10 / 15.87 / 16.05** (0.347, 0.242 -> **16.1**), against a sharp corner
running to 85.93 / 60.69 and still climbing. Nothing on the arc is re-entrant — both tangent
points measure 183-185 deg and `N`, where four blocks meet, is interior — so **the
construction introduces no corner of its own.**

#### THE RESULT NOBODY WENT LOOKING FOR

```
  axle drop mm      smoke     coarse    medium     fine     spread, coarse..fine
  unfilleted      1.499486  1.551645  1.562981  1.570505         1.216%
  FILLETED        0.902237  0.962456  0.963816  0.962579         0.141%
```

**-37.97% at the shipped radii — equivalently the unfilleted mesh reports a drop 61.2%
HIGHER — and it converges.** Two of FILLET_PLAN's four stated reasons for existing,
answered in one table.

*The magnitude* is reason 4, §31's: sweeping `R_hub` across its box used to leave the
solved wheel **bit-identical**, so two of the fourteen genes steered on a `Kt` correlation
with no mechanical feedback at all. On the filleted mesh the radius ladder moves the axle
drop from 1.567 to 0.765 mm, monotonically. **They are still invisible to the OPTIMIZER**,
which may not build this mesh — what changed is that the feedback exists and is measured.
§24 priced the fillets at 4.406 g and 8.77% of the part by mass; this is what that mass is
worth structurally, and `R_rim = 3.0` mm puts 8.5 mm of tangent length on a 41 mm flank at
the root, where the moment is.

*The convergence* is reason 2: the unfilleted drop is still climbing at `fine` and spans
1.216% over `coarse..fine`, the filleted one spans 0.141% and is flat from `coarse` up.
That is the singular field polluting a global functional — the mechanism §29 spent 95
minutes failing to identify from a convergence order.

*And the fillet is the PART's.* `export/wheel_step_manifest.json` reports both junctions
BUILT at exactly what was requested — 0.663606 and 3.000000 mm, 24 of 24 edges each,
`kt_error_pct` 0.0 — on corners whose `worst_wedge_deg` is 322.0 / 320.0, which is `P_t`'s.
`P_c`'s 268 / 271 appears nowhere in it. **The exporter and the mesh round the same corner
family at the same radius, and neither rounds `P_c`** — which is why the mesh's last
artefact corner is an artefact. Under test, and the check is not a formality: `kt_report`'s
own docstring records the rim once shipping twelve of twenty-four corners square while
reporting `kt_error_pct = +0.0%`.

**THE SCOPE TRAVELS WITH THE NUMBER, IN TWO DIRECTIONS.** *Kinematics*: this is
`solve_wheel`'s `axle_drop_mm`, ONE phase, LINEAR, one genome — §33's item 3 registered
that caveat for this driver in advance, and a 38% deflection change is exactly the
*"magnitude / `R_hub`-`R_rim` sensitivity"* class it names. The EXPONENT half of §52 does
not inherit it: Williams is linear-elastic and the driver reproduces the 360 deg crack at
0.5. *Which QoI*: the ±0.3% band §29 retired belongs to the GATE's quantity,
`axle_drop_mean_mm` — eight phases, both kinematics, `make gci`, re-run post-flip at §49 —
so **§52 does not earn the absolute band back.** It says the mechanism §29 named as the
obstacle is measurably the obstacle, and is gone on the filleted mesh for the cheap
single-phase surrogate. Whether the gate's own QoI follows is item 4 below.

**THE CONTROL IS WHAT MAKES THAT READABLE AND IT IS IN THE ARTIFACT.** A 38% shift between
two meshes is equally well explained by the fillet's stiffness and by a different model.
The blocking takes an explicit radius pair, so drive it toward zero: **-0.17% at R = 0.05
mm** against the unfilleted wheel, monotone in `R` everywhere above the floor. It is a
limit and not an identity — `sector_blocks` refuses `R = 0`, because the re-cut moves four
blocks — so the smallest rung's residual is asserted rather than hidden.

#### THE BUG THE MEASUREMENT FOUND

`measured_wedge_deg` took the nearest node of any kind. A Q9 midside is skipped by the
angle sum, so landing on one returns **0.00 deg and zero incident elements** — a number
shaped like a measurement. It could not fire while every probe was an exact vertex, which
the four unfilleted corners are; on the filleted mesh `rim:P_t` reported 0.00 against
`hub:P_t`'s correct 360.00, on which of two equally-near nodes came first. Restricted to
Q9 vertices.

#### WHAT IS UNCHANGED

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, no threshold
moved, and the default mesh bit-identical.** The refreshed unfilleted artifact was diffed
field by field against the committed one: every measured value identical, the additions
being `fillet`, `global_peak`, `kind`, `node_gap_mm` and the difference columns.

**`test_peak_stress_diverges_but_the_field_converges` still passes and was NOT touched.**
Step 2 anticipated it failing; it calls `build_wheel(genes, cfg)` bare, so it measures the
unfilleted default. Updating it would assert a filleted result against an unfilleted
measurement.

**§48's scope stands and is now under test.** `fillet=` is a measurement instrument for one
genome — 6 of 16 feasible genomes refuse it — and
`test_nothing_wires_the_fillet_into_the_objective` PARSES five `src/` modules and refuses
any non-`None` `fillet=` / `fillet_blocking=` keyword. It parses rather than greps because
both greps that would do the job are wrong: `wheel_objective` has a LOCAL `fillet =
jnp.sum(...)` — the fillet-MARGIN barrier, nothing to do with the mesh — and `wheel_fea`
passes `R_hub_fillet=` to the EXPORTER, where fillets have always been geometry.

#### The successors, ranked — REVISED 2026-08-23 AFTER §52

1. **The rim tri-block, BUILT** — was 2, now 1. §52 removed the fillet's half of the chain
   and left the tri-block as the *whole* remaining path to a quotable peak: `rim:P_c` is
   the last artefact corner, it carries the global maximum, and only the faithful rim
   admits a fillet on it. §51's probe says §37 priced it wrong on both clauses. §48's
   driver is the template. **Do not quote §51's probe numbers until this exists.**
   **DONE 2026-08-23 — §53. It is built, both clauses are retired, it meshes at 77x — and
   it folds on a quarter of the gene box, which is what it left behind.**
2. **FILLET_PLAN Step 3, item 1 — `R_hub` and `R_rim` as live FEA genes.** Promoted from
   "after Step 2 confirms the singularity is gone" and re-scoped by what §52 measured:
   §31's finding was that sweeping `R_hub` leaves the solved wheel BIT-IDENTICAL, and the
   filleted mesh moves the axle drop by 38% and monotonically in `R`. The two fillet genes
   have mechanical feedback for the first time. **This is blocked behind item 3**, because
   feedback the optimizer may not use is not feedback.
3. **Make the filleted blocking genome-robust** (§48) — 6 of 16 feasible genomes refuse it,
   and 6 of the 10 that build sit under `MIN_SJ_TARGET`. Was 3.
4. **Step 2's part C — the p-norm on a filleted mesh.** `make gci`, 95 minutes and 20.6 GB.
   What the optimizer actually sees is not the raw peak, and §52 measured the peak only.
5. **Make `modelled_area_reference` fillet-aware** (§50).
6. **The REST of §45's audit list** (§49).
7. **G1's fourth revision** — §40 confirmed the gate blocks nothing.
8. **§32's successors 3 and 4** — §8's wall-floor economics under SVK.
9. **The element-validity check** (§44).

---

### §53 — 2026-08-23. THE RIM TRI-BLOCK IS BUILT. BOTH OF §37's CLAUSES ARE RETIRED AND IT MESHES AT 77x THE BLOCK IT REPLACES — BUT IT FOLDS ON A QUARTER OF THE GENE BOX, AND THAT IS A THIRD OBSTACLE NEITHER §37 NOR §51 NAMED

> **SCOPE — THE BOX HERE IS THE UNIFORM DRAW, NOT THE DESIGN SPACE (§72, §76).**
> Every genome-box figure in this section comes from the uniform Latin hypercube in
> `study_mesh_quality.latin_hypercube` over the full gene box. That sampler puts about
> **one genome above 35 degrees of arc span in sixty-four**, and a draw conditioned on
> arc span reaches bows to 1.25 against its maximum of 0.54. The numbers below are
> correct about what was drawn; read "the box" as "this draw" throughout.

§52 promoted the rim tri-block to ranked item 1 and called it *"the whole remaining path to
a quotable peak"*. §51 had already re-priced it, with a scratch probe it filed **as** a
probe — *"no driver, no artifact, no test — precisely so that nobody quotes it as this
project's other numbers may be quoted"* — and named the unit that would make it a
measurement. This is that unit, and **§51's probe numbers are superseded by this section.**

`make triblock`, 14 s, geometry and Jacobians only. Driver `studies/study_tri_block.py`,
artifact `studies/study_tri_block.json`, 23 tests in `tests/test_tri_block.py`. Full
working record in `UNCAP_PLAN.md`'s **STEP 3 RECORD, PART 2**.

#### BOTH OF §37's CLAUSES ARE RETIRED, MEASURED

**Clause 2, "forced 1-element strips".** §37's six-way constraint is correct and its input
was not. `a1 = (A + B - C) / 2`, where A is the ring arc (`n_weld`), C the end
cross-section (`n_thick`) and **B the free side, whose count was never inherited**. §37
read B off the block it was replacing — 8 at `coarse` — and got 7x1, 3x1, 7x3. The
admissible set is enumerable and is enumerated:

```
  config  A   C     admissible B          strip-free B
  coarse  10  4     8, 10, 12             10
  medium  16  6     12, 14, 16, 18, 20    14, 16, 18
```

§37 missed the strip-free choice at `coarse` by one grid point. The driver reproduces
§37's own 7/3/1 and **gates its exit on doing so**, because a re-pricing that cannot
reproduce the number it re-prices is measuring a different partition.

**Clause 1, "it needs PARTIAL-EDGE SEAMS".** True only if the neighbours may not be split.
Split `rim_band_weld` in theta and the `spoke` along a j-line; that cascades once into
`hub_junction` — the spoke's hub row IS its `left` edge — and **stops**, because the
junction's cut runs across the collar arc rather than along it. **Seven blocks become
TWELVE and SEVENTEEN seams, every one a whole edge of both blocks it names, all closing at
7.1e-15 mm** at both configs. The Y's own three internal edges are exactly 0.0, because
each is passed as the same array to both of its blocks.

#### AND IT MESHES — 77x, AT 80% OF THE SHIPPED MESH

```
  config   shipped blend 1.0   faithful QUAD   faithful TRI       x   clears 0.2
  coarse       0.782735          0.008176        0.626233      76.6x     YES
  medium       0.782926          0.008251        0.581582      70.5x     YES
```

Against `MIN_SJ_TARGET` = 0.2 (barrier weight 3000, imported from `wheel_objective` rather
than written down) that is **3.1x of margin**, while delivering the **1.06 degree** corner
fidelity that was the whole point. §51 said ~0.25 and called it a floor. It was one.

The three cut neighbours are **sliced, not rebuilt** — bit-for-bit, pinned at `== 0.0` —
so the 77x is a measurement of a partition and not a comparison of two constructions. The
three quads' areas sum to the quad block's to 1e-5 relative.

#### THE OBSTACLE THAT ACTUALLY STOPS IT, AND WHY §48's PRECEDENT DOES NOT APPLY

**The faithful rim is not opt-in.** §48 could measure the filleted blocking at one genome,
name six refusals out of sixteen and still hand Step 2 a usable instrument, because
`fillet=` is passed by a study and never by the optimizer. Adopting blend 0.0 changes
`sector_blocks` for **every genome the search touches**. So the gene box is the
measurement, and one genome is not.

Sixteen freshly drawn feasible genomes, four per flank orientation, `B` held per config
because element counts may not depend on the design:

```
  config   fixed rule valid   best-per-genome valid   clears 0.2 (fixed)   seams
  coarse       12/16                15/16                  11/12          all close
  medium       10/16                12/16                  10/10          all close
```

The fixed rule applies the shipped genome's own barycentric triple — scale-free, so a
construction with no free parameter left, which is what would actually ship. The second
column re-sweeps the interior point per genome and is **the upper bound any adaptive rule
could reach**; it is not 16/16 either.

**The mechanism is named rather than counted:** it folds on the WIDE weld arcs — 15.9-41.2
degrees against 3.7-11.9 on the ones it does not, at `coarse`, where the two ranges
separate cleanly. **The shipped genome's arc is 2.73 degrees, at the very bottom of the
box.** A point tuned on a 2.7-degree sliver is in the wrong place on a 40-degree triangle.

**So the construction is proved and the rule that places its interior point is not.** That
is a different sentence from §37's and §51's, and neither of them was in a position to
write it: it is not visible until the thing is built.

#### AND A GENERATED INTERIOR CANNOT MOVE IT — WHICH NAMES THE SUCCESSOR

A Winslow solve on each quad's interior, boundaries held, changes the number by
**0.000000**. The worst corner is on a held boundary, and the Y's three spokes are
boundaries of two blocks each, so per-block smoothing holds them by definition. **The
number is set by where the Y's spokes GO.** They are straight lines today. Same shape as
§47's route-2 invariance, same conclusion: **a curved Y is the successor; a better smoother
is not.**

#### THE CELL IS A TUNED POINT AND THE REPORT SAYS SO

The rule is the argmax over the **published** grid, and the grid is published in full for
§48's reason. Here it is a tuned point rather than a plateau, and that is reported: **6.9%**
of valid cells at `coarse` and **8.3%** at `medium` sit within 10% of the maximum, and only
29/173 and 24/173 cells are valid at all. A finer local re-sweep is reported and
**deliberately not adopted** — at `medium` it gains 0.045 by walking one weight down to the
search box's clamp, and that cell generalises across the gene box *worse* than the grid
point it beats.

#### THE BUG THE INSTRUMENT FOUND, AND HOW IT HID

The first seam table paired the wrong halves of the cut spoke and the cut hub junction: the
junction's `left` edge is the spoke's hub row REVERSED when the straddling flank is at
eta = +1. At `coarse`, `n_thick` splits 2/2, so **the node counts still agreed** and only
the coordinates disagreed, by 0.62 mm. A seam check reporting closure as one boolean would
have read "17/17 counts agree" and said nothing. It reports the count and the gap
separately, which is why one run found it. A test reproduces the mismatch and asserts that
the counts still agree under it. The second was the ring weld block's low-theta end, which
is `P_t` or `Q` depending on the genome — exactly §48's `dk` trap.

#### WHAT IS UNCHANGED

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, no threshold
moved, `UNCAP_DEFAULT` still `(True, 1.0)`, `sector_blocks` still returns seven blocks, and
the default mesh bit-identical.** Blend 0.0 is measured, never adopted. Pinned by
`test_nothing_here_is_wired_into_the_mesh_the_tree_BUILDS`.

#### The successors, ranked — REVISED 2026-08-23 AFTER §53

1. **A rule for the tri-block's interior point that holds across the gene box, and the
   CURVED Y it probably needs alongside.** Was "the rim tri-block, BUILT"; the build is
   done and this is what it left. It is now a well-posed problem rather than an open one:
   the sixteen drawn genomes are in the artifact with their triangles' arc spans, side
   lengths and three wedge angles, the failure mode is the wide arc, and the ceiling any
   rule can reach at the current `B` is 15/16 and 12/16 — so a rule alone may not suffice.
   The Winslow column says the lever is where the Y's spokes go.
2. **Make the filleted blocking genome-robust** (§48) — was 3, unmoved in substance and now
   the same shape of problem as item 1: 6 of 16 feasible genomes refuse it, 6 of the 10 that
   build sit under `MIN_SJ_TARGET`. **These two are one arc if anyone wants them to be** —
   the question in both is a construction that was measured at one genome and has to hold
   across a search.
3. **FILLET_PLAN Step 3, item 1 — `R_hub` and `R_rim` as live FEA genes.** Was 2. Blocked
   behind item 2, unchanged: feedback the optimizer may not use is not feedback.
4. **Step 2's part C — the p-norm on a filleted mesh.** `make gci`, 95 minutes and 20.6 GB.
   What the optimizer sees is not the raw peak, and §52 measured the peak only.
5. **Make `modelled_area_reference` fillet-aware** (§50).
6. **The REST of §45's audit list** (§49).
7. **G1's fourth revision** — §40 confirmed the gate blocks nothing.
8. **§32's successors 3 and 4** — §8's wall-floor economics under SVK.
9. **The element-validity check** (§44).

**AND ONE THING §52's RANKING GOT WRONG, RECORDED BECAUSE IT IS THE THIRD TIME.** §52 put
the tri-block at 1 on the argument that it was *"the whole remaining path to a quotable
peak"*. It is still the only path, and it is now built and valid — but a path that folds on
a quarter of the gene box does not reach a quotable peak either, and **nothing in §52 or
§51 could have priced that, because neither had built it.** §37's own lesson said "when a
successor's value rests on *and then X will improve*, measure X on the CURRENT model
first". This is the complement: **when a successor's value rests on a construction that
does not exist, the thing that stops it is often not any of the reasons anyone listed** —
§37 named seams and strips, §51 retired both, and what stopped it was neither.

---

### §54 — 2026-08-23. THE FILLETED SECTOR'S QUALITY FAILURE IS FIXABLE ACROSS GENOMES — NINE OF NINE CLEAR INSTEAD OF FOUR OF NINE — BUT IT IS NOT ADOPTED, BECAUSE IT SPENDS §52's OWN RESULT FOR A BENEFIT NOTHING YET COLLECTS. AND THE SWEEP FOUND A GENOME BUG THAT HAS NOTHING TO DO WITH THE FILLET

§48 ranked item 2 as "make the filleted blocking genome-robust" — 6 of 16 feasible
genomes refuse it outright, 6 of the 10 that build sit under `MIN_SJ_TARGET`. This is
that item, worked on the QUALITY half only. Full record: FILLET_PLAN.md STEP 1 RECORD
PART 13.

**The quality half is fixable, and by a wide margin.** §48's `LAYER_ENTRY_SLOPE` /
`LAYER_END_OFFSET` were an argmax over ONE genome's radius box — the same scope limit
§48 FINDING 6 already diagnosed for the blocking itself. Re-derived against the ten
genomes §48 already drew (`sweep_layer_profile_genomes`, new in
`studies/study_fillet_block.py`), a different pair — `entry = -0.75, end = 0.70` against
the shipped `-0.45, 1.60` — clears `MIN_SJ_TARGET` for **nine of nine** non-pathological
built genomes, against **four of nine** at the shipped pair.

**It is measured and reported, not shipped, and the reason is a priced trade rather than
caution.** The same shipped genome §52 measured PART 12's headline result on pays for the
other nine: at the genome-robust pair, the filleted axle-drop spread over `coarse..fine`
widens from §52's **0.141%** to **0.513%**, crossing back over the +-0.3% band that
result was measured against. And the REFUSAL half of item 2 — the hub sector-fit
limit — is untouched by this pair or any other choice of `entry`/`end`; it is a pure
function of the genome's own geometry and the fixed 30-degree sector. So adopting the
new pair today would spend a working, published result (§52's) to partially fix one of
two problems that still leaves the blocking un-robust either way. `blend 0.0` in §53 is
the standing precedent for this exact call.

**The sweep also found a genome defect that is not about the fillet at all.** One drawn
genome's UNFILLETED flank has a near self-intersection around one arc-length station
that its own shipped 97-station grid happens to straddle without ever landing a folded
element — so §48's own "unfilleted sector is clean" feasibility gate passed it. The
fillet's trim re-parametrises the spoke's station grid onto `[s_A(hub), s_A(rim)]`, which
for this genome puts a station almost exactly on the defect, and the trimmed spoke folds
outright (`min scaled Jacobian` -0.05, sign-flipped). Confirmed independent of `entry`/
`end` — bit-identical across three widely separated profiles. **The fillet is the
instrument that found it, not the cause.** Filed rather than fixed: strengthening the
feasibility gate to catch this class of flank defect is separate work from either item
this section touches.

#### The successors, ranked — REVISED 2026-08-23 AFTER §54

1. **A rule for the tri-block's interior point that holds across the gene box, and the
   CURVED Y it probably needs alongside.** Unchanged from §53 — still open, still the
   same shape of problem this section just solved half of for the fillet.
2. **The filleted blocking's REFUSAL half** — the hub sector-fit limit, 6 of 16 feasible
   genomes, untouched by §54. Narrowed from "make the filleted blocking genome-robust"
   now that the quality half has a measured (if unshipped) fix; this is what is actually
   left of §48's item.
3. **A feasibility gate that catches the flank near-self-intersection §54 found.** New,
   ranked on the strength of being a genome-validity bug independent of anything this arc
   is about — `sweep_genomes`' own "unfilleted sector is clean" check passed a genome
   whose flank already had it.
4. **FILLET_PLAN Step 3, item 1 — `R_hub` and `R_rim` as live FEA genes.** Unchanged.
   Blocked behind item 2: feedback the optimizer may not use is not feedback, and item 2
   is now item 2 above rather than fully closed.
5. **Step 2's part C — the p-norm on a filleted mesh.** `make gci`, 95 minutes and 20.6 GB.
6. **Make `modelled_area_reference` fillet-aware** (§50).
7. **The REST of §45's audit list** (§49).
8. **G1's fourth revision** — §40 confirmed the gate blocks nothing.
9. **§32's successors 3 and 4** — §8's wall-floor economics under SVK.
10. **The element-validity check** (§44).

---

### §55 — 2026-08-23. THE TRI-BLOCK'S INTERIOR-POINT RULE HOLDS ACROSS THE GENE BOX AT ONE CONFIG FOR FREE, AND AT THE OTHER FOR A PRICE SMALLER THAN THE ONE THE FILLET PAID

> **SCOPE — THE BOX HERE IS THE UNIFORM DRAW, NOT THE DESIGN SPACE (§72, §76).**
> Every genome-box figure in this section comes from the uniform Latin hypercube in
> `study_mesh_quality.latin_hypercube` over the full gene box. That sampler puts about
> **one genome above 35 degrees of arc span in sixty-four**, and a draw conditioned on
> arc span reaches bows to 1.25 against its maximum of 0.54. The numbers below are
> correct about what was drawn; read "the box" as "this draw" throughout.

§54's ranking put "a rule for the tri-block's interior point that holds across the gene
box" at 1, framed as the same shape of problem the fillet's layer profile was. It is: a
joint argmax of the worst genome's worst block, over the sixteen genomes UNCAP_PLAN Step 3
PART 2 already drew plus the shipped genome, against a dedicated 25x25 barycentric grid
(`sweep_w_genomes`, new in `studies/study_tri_block.py`) reaches the SAME validity ceiling
`best_w` — a free per-genome parameter — reaches at `medium` (13/13 of the reachable
genomes) and within one of it at `coarse` (15/16). Full record: UNCAP_PLAN.md STEP 3
RECORD PART 3.

**The price is asymmetric, and at one config there isn't one.** At `coarse` the
genome-robust cell leaves the shipped genome's own number UNCHANGED — 0.6262 either way —
while fixing two more drawn genomes. At `medium` it costs the shipped genome's quoted
multiplier (0.5816 -> 0.4336, roughly 70x -> 52x over the collapse) while fixing two more
drawn genomes and gaining one more over the barrier; 0.4336 still clears `MIN_SJ_TARGET` by
more than double. Measured and not adopted — but for a reason distinct from every prior use
of that phrase: nothing reads the tri-block's `chosen` cell except this file's own printed
table, since the construction is not wired into `sector_blocks` at all. Adopting the
genome-robust cell would change a quoted number and nothing else, which is why the choice
is left to whoever next picks up this arc rather than decided here.

**What is still open.** The genome no `w` reaches at either config (41.2°, plus three more
unreachable at `medium` alone) is untouched — that is the curved Y's question, and the
ceiling itself says a rule alone cannot close it. Item 1 below is narrowed accordingly.

#### The successors, ranked — REVISED 2026-08-23 AFTER §55

1. **The curved Y.** Narrowed from "a rule for the interior point that holds across the
   gene box, and the curved Y it probably needs alongside" — the rule half now has a
   measured (if unshipped) answer at the ceiling `best_w` sets, and what is left is the one
   genome per config no placement of the straight Y's interior point can rescue. The
   Winslow column said the number is set by where the spokes go, not by how the interiors
   are filled, and they are straight lines today.
2. **The filleted blocking's REFUSAL half** — the hub sector-fit limit, 6 of 16 feasible
   genomes, unchanged from §54.
3. **A feasibility gate that catches the flank near-self-intersection §54 found.**
   Unchanged from §54.
4. **FILLET_PLAN Step 3, item 1 — `R_hub` and `R_rim` as live FEA genes.** Unchanged.
5. **Step 2's part C — the p-norm on a filleted mesh.** `make gci`, 95 minutes and 20.6 GB.
6. **Make `modelled_area_reference` fillet-aware** (§50).
7. **The REST of §45's audit list** (§49).
8. **G1's fourth revision** — §40 confirmed the gate blocks nothing.
9. **§32's successors 3 and 4** — §8's wall-floor economics under SVK.
10. **The element-validity check** (§44).

---

## §56 — 2026-08-23. [READ "THE WHOLE REACHABLE BOX" AS "THE UNIFORM DRAW" — SEE §72 AND §76.] THE CURVED Y IS BUILT, AND IT MAKES A FIXED RULE VALID ON THE WHOLE REACHABLE BOX AT ONE CONFIG — BUT THE GENOME THAT REFUSES IT IS NOT THE ONE THE MECHANISM PREDICTED

> **SCOPE — THE BOX HERE IS THE UNIFORM DRAW, NOT THE DESIGN SPACE (§72, §76).**
> Every genome-box figure in this section comes from the uniform Latin hypercube in
> `study_mesh_quality.latin_hypercube` over the full gene box. That sampler puts about
> **one genome above 35 degrees of arc span in sixty-four**, and a draw conditioned on
> arc span reaches bows to 1.25 against its maximum of 0.54. The numbers below are
> correct about what was drawn; read "the box" as "this draw" throughout.

§55's revised successors ranked the curved Y first, narrowed to "the one genome per config
no placement of the straight Y's interior point can rescue." That is now built, measured at
both configs, and it retires more than the narrowing expected and less than the mechanism
promised. UNCAP_PLAN.md's STEP 3 RECORD PART 4 has the full record; this is the section
number and the ranking.

**What the curve is.** §53's Winslow column said the tri-block's number is set by where the
Y's three spokes GO, not by how the interiors are filled, because each spoke is a BOUNDARY
of two blocks and per-block smoothing holds it. `_bent_spoke` moves them. Each spoke is
blended toward the two pieces of the region's own boundary it faces across its two quads,
at the fraction where its own foot sits between them, with a Coons correction that pins its
endpoints exactly — so **the three internal seams stay exact for every bend**, and no
resampling enters, because `splits` already forces the spoke and the two curves it blends
to carry the same node count. One parameter, `bend`, and `bend = 0.0` returns the straight
spoke untouched rather than an array equal to it.

```
                        per-genome ceiling            one FIXED (w, bend) rule
config   straight    curved   rescued      straight valid/clear/worst   curved valid/clear/worst
coarse     16/17     16/17       0            15/16  13  -0.0204          16/16  13  +0.0478
medium     14/17     16/17       2            13/16  12  -1.0000          13/16  13  -0.4875
```

**At `coarse` a single fixed rule becomes valid on every genome the plane reaches**, at
bend 0.20, with the joint floor moving from -0.0204 to +0.0478 — and it costs the shipped
genome nothing, 0.6262 either way. **At `medium` the curve raises the ceiling instead of
the count**: two genomes no interior point rescues become valid (35.3° at 0.3464, 15.9° at
0.1698) and a third crosses `MIN_SJ_TARGET` (22.7°, 0.0780 -> 0.2366), while one fixed rule
still reaches 13 of 16 and the shipped genome would pay 0.5816 -> 0.4384 to sit at it.

**The bend is inert where the region is fat.** One genome of seventeen wants a non-zero
bend at `coarse` and four do at `medium`; at the published cell the shipped genome's number
moves across the whole bend range by 0.000000 at `coarse` and 0.001 at `medium`. That is
what says the curve is a correction to cutting chords and not a knob being tuned.

**The mechanism is the BOW, not the arc span — and it does not explain the survivor.**
§53's gene box named the straight Y's failure mode as the wide weld arc. The quantity is
really the arc's greatest departure from its own chord over the region's cross-section
length, now `bow_over_width`: at `coarse` the fixed straight rule folds on 0.264-0.498 and
holds on 0.009-0.129, cleanly separated, and it sorts 18.5° (bow 0.149, fine) from 15.9°
(bow 0.264, folded) the way the arc span cannot. But **the largest bow in the box, 0.498, is
a genome the curve reaches**, and the one that refuses everything has a bow of 0.491.
**[FALSIFIED AS A STATEMENT ABOUT THE DESIGN SPACE — §72.** "The largest bow in the box" is
the largest bow the UNIFORM sampler drew. Conditioning the draw on arc span reaches bows to
**1.25**, against that sampler's maximum of 0.54. The non-separation below is real and the
sentence around it is a fact about a draw.**]**
`test_what_the_curve_does_NOT_reach_stays_reported` asserts the NON-separation, so that a
future run which does separate them registers as a finding.

**The refusal is priced at every free count, not just the one that ships.** 41.2° folds at
every interior point, every bend, and every admissible `B` at both configs — ceiling
-0.0134, valid at none — and a hand probe pushing the interior point outside the published
search box only reaches -0.0064, so the box is not the constraint either.

**Measured, not adopted, for the third time in this arc and for §53's reason.** `bend`
defaults to 0.0, `chosen` is still the straight Y's single-genome argmax, `per["sector"]`
still reports 0.626233 / 0.581582 and 76.6x / 70.5x, `UNCAP_DEFAULT` is still `(True, 1.0)`
and `sector_blocks` still returns seven blocks. The regenerated artifact was diffed against
the committed one: the only non-additive change is the wall clock. `make triblock` is now
~290 s and the Makefile's help says so.

#### The successors, ranked — REVISED 2026-08-23 AFTER §56

1. **The filleted blocking's REFUSAL half** — the hub sector-fit limit, 6 of 16 feasible
   genomes, unchanged from §54 and §55. Now the top item: the tri-block arc has had two
   sessions and both of its named levers are measured, while this one has been ranked
   second through two revisions without being touched.
2. **A feasibility gate that catches the flank near-self-intersection §54 found.**
   Unchanged from §54 and §55.
3. **What makes a region impossible.** Demoted from §55's item 1 in substance as well as
   rank: the curved Y is built, so what is left is not a construction but a QUANTITY — the
   one that separates the 0.491-bow genome that refuses everything from the 0.498-bow one
   the curve reaches. Everything needed to look for it is in the artifact.
4. **A bend that is a FUNCTION of the genome rather than a constant.** The curve enlarged
   what is reachable without making a constant rule better at reaching it, so at `medium`
   the gap between the per-genome ceiling (16/17) and one fixed rule (13/16) is now wider
   than the straight Y's was. `bow_over_width` is the obvious argument to fit against.
5. **FILLET_PLAN Step 3, item 1 — `R_hub` and `R_rim` as live FEA genes.** Unchanged.
6. **Step 2's part C — the p-norm on a filleted mesh.** `make gci`, 95 minutes and 20.6 GB.
7. **Make `modelled_area_reference` fillet-aware** (§50).
8. **The REST of §45's audit list** (§49).
9. **G1's fourth revision** — §40 confirmed the gate blocks nothing.
10. **§32's successors 3 and 4** — §8's wall-floor economics under SVK.
11. **The element-validity check** (§44).

---

## §57 — 2026-08-23. THE FILLETED BLOCKING'S REFUSAL HALF CLOSES, THE NUMBER THAT CLOSES IT WAS ALREADY IN THE FILE, AND IT DOUBLES WHAT §54'S SHELVED PROFILE IS WORTH

§56 ranked "the filleted blocking's REFUSAL half — the hub sector-fit limit, 6 of 16
feasible genomes" first, noting it had been ranked second through two revisions without
being touched. It is now measured. FILLET_PLAN.md's STEP 1 RECORD PART 14 has the full
record; this is the section number, the two findings that reach outside the fillet arc,
and the ranking.

**The refusal was already predictable.** §48's FINDING 4 bisected `sector_fit_limit` — the
radius at which the fillet's tangent point reaches the next sector's corner — and FINDING 6
named it as the mechanism behind all six refusals. What nobody had done is compute it **per
drawn genome** and compare it to that genome's own `R_hub`. `sector_fit_margin` does, and
**it classifies 16 of 16**: every genome whose radius exceeds its own limit refuses, no
other does. The margins run -0.4833 to +5.8980 mm across genomes that all pass the same
feasibility filter, so this is a wide feature of the design space rather than a boundary
effect. That turns the refusal from an exception caught after a build attempt into a
feasibility number of the same kind as `evaluate_design`'s `x_order`/`hub_overlap` pair —
which is the shape the flank near-self-intersection gate (item 2 below) also wants.

**And the fix is the same number used the other way.**

```
  profile         clamp    built   clears 0.2   min scaled J range    median
  shipped          none    10/16       4/16      -0.0508...+0.2898    0.1780
  shipped          0.95    16/16       8/16      -0.0508...+0.2898    0.2081
  genome_robust    none    10/16       9/16      -0.0508...+0.4592    0.2872
  genome_robust    0.95    16/16      15/16      -0.0508...+0.5156    0.2915
```

Clamping each radius inside its own sector's room makes **every drawn genome build**, and
it is inert on the shipped genome (hub 0.6636 against a limit of 3.1297; the rim has no
limit at all), so it costs the arc's published numbers nothing. It is insensitive to its
own factor across 0.75-0.99, and the test asserts the insensitivity rather than the value.

**A gate and a fix are different and the record keeps them apart.** `binds` is the gate:
exact, free, and it loses the genome. The clamp keeps the genome and models a smaller
fillet than the genes asked for — fine for an instrument sweeping the box, fine for an
optimizer only if the objective is told the clamped radius. If `R_hub`/`R_rim` become live
FEA genes the clamp is a bound projection, which is standard, provided the projected value
is what is reported back.

**THE FINDING THAT REACHES OUTSIDE THE FILLET ARC: §54's decision was taken against a
premise that is now false.** §54/FILLET_PLAN PART 13 declined the genome-robust layer
profile for two reasons. The first stands: it costs the shipped genome's filleted
axle-drop spread 0.141% -> 0.513%, back across the ±0.3% band, and the clamp does not
touch that. The second was that the refusals were untouched by any profile, so *"six of
sixteen feasible genomes still refuse outright regardless"* and nothing would collect what
the profile bought. With the clamp, that profile takes the draw from 8/16 clearing the
barrier to **15/16** — the exception being the trimmed-spoke genome §54 already isolated.
**The call does not change, and it now rests on one reason instead of two.** That is worth
recording as a general lesson and not only as a fillet fact: a "measured, not adopted"
decision is a decision against a *state of the world*, and this arc has now produced three
of them in four sections. They need re-checking when the world moves, not just citing.

**Measured, not adopted.** `sector_blocks` and `build_wheel` are untouched and still take
the radii they are given; `clamped_radii` is called by nothing outside the study;
`FILLET_LAYER_ENTRY_SLOPE`/`FILLET_LAYER_END_OFFSET` are still §48's. The regenerated
artifact was diffed field-by-field against the committed one and is purely additive.
`make filletblock` is now ~180 s and the Makefile's help says so.

#### The successors, ranked — REVISED 2026-08-23 AFTER §57

1. **A feasibility gate that catches the flank near-self-intersection §54 found.** Now the
   top item, and §57 makes it a better-posed one: `sector_fit_margin` is a worked example
   of the exact shape wanted — a quantity computed from the geometry alone, before any
   build, that classifies the draw without a false positive. The flank defect needs the
   same treatment and does not have it.
2. **The QUALITY half of the filleted blocking, which is now the whole of the old item 2.**
   Eight of sixteen sit under the barrier at the shipped profile with the clamp in place.
   The open question §57 leaves is whether §54's profile gain can be had without §54's
   cost — a profile derived against genomes AND constrained to hold the shipped genome's
   convergence spread, which is a two-objective version of §54's argmax and has not been
   attempted. Re-deriving that argmax against the now-buildable sixteen (rather than ten)
   is the cheap first half of it.
3. **What makes a region impossible** (§56) — the quantity separating the 0.491-bow genome
   that refuses the curved Y from the 0.498-bow one it reaches.
4. **A bend that is a FUNCTION of the genome rather than a constant** (§56).
5. **FILLET_PLAN Step 3, item 1 — `R_hub` and `R_rim` as live FEA genes.** §57 raises this
   one's stakes: the clamp is exactly the bound projection such genes would need, and it is
   now measured.
6. **Step 2's part C — the p-norm on a filleted mesh.** `make gci`, 95 minutes and 20.6 GB.
7. **Make `modelled_area_reference` fillet-aware** (§50).
8. **The REST of §45's audit list** (§49).
9. **G1's fourth revision** — §40 confirmed the gate blocks nothing.
10. **§32's successors 3 and 4** — §8's wall-floor economics under SVK.
11. **The element-validity check** (§44).

---

## §58 — 2026-08-23. THE FLANK DEFECT'S GATE IS THE OPTIMIZER'S OWN FOLD BARRIER, WHICH FIVE STUDIES ALREADY USE AND THE TWO BLOCKING STUDIES NEVER ASKED

> **SCOPE — THE BOX HERE IS THE UNIFORM DRAW, NOT THE DESIGN SPACE (§72, §76).**
> Every genome-box figure in this section comes from the uniform Latin hypercube in
> `study_mesh_quality.latin_hypercube` over the full gene box. That sampler puts about
> **one genome above 35 degrees of arc span in sixty-four**, and a draw conditioned on
> arc span reaches bows to 1.25 against its maximum of 0.54. The numbers below are
> correct about what was drawn; read "the box" as "this draw" throughout.

§57's rank-1 successor was a feasibility gate for the flank near-self-intersection §54
found — the one drawn genome whose trimmed spoke is sign-flipped at `-0.0508`, traced to a
near self-intersection in the **unfilleted** flank at `s = 0.051`.  §57 posed it well: the
shape wanted is a quantity computed from the geometry alone, before any build, that
classifies the draw without a false positive.

**It already existed.**  `wheel_geometry.self_intersection_margin` — `min_s(|1/kappa| -
t/2)`, closed form off the Bezier hodograph — is exactly that, its threshold
`MIN_FOLD_MARGIN_MM = 0.1` was calibrated over 2001 genomes, `wheel_objective` has carried
it as a live barrier since the objective was rewritten, and `study_gnl`, `study_contact`
and `study_wheel_fea` all gate their draws on it.  `study_fillet_block.sweep_genomes` and
`study_tri_block.sweep_genomes` use a two-term filter plus a mesh-based clause and never
ask.  That is the whole of why §54's genome was in the box.

**It classifies.**  Two of the sixteen drawn genomes describe a part that does not exist,
at margins `-0.3436` and `-0.0131` against a next-smallest of `+0.1244`; one of sixteen
inverts a block, and it is in that pair; no fold-clean genome inverts anything.  The
converse is deliberately NOT claimed — whether a folded flank shows as an inverted element
depends on where the trim puts a station, which is a property of the grid — so the gate
promises the direction that is a property of the geometry.

**The filter it replaces leaks, and by how much is measured.**  Over 20480 draws on the
same stream: 1454 pass the geometric pair and 514 of those fold (35.4%); 925 also mesh a
clean unfilleted sector and **25 of those still fold (2.7%)**.  The mesh clause removes 489
of 514 — it is a good proxy — but the box of sixteen got two, which is that rate.

**And the reason the answer is a closed form and not a finer grid is measurable.**  Audited
against the sampled flank on all 1454 at 2000 points: one disagreement, at |margin| =
2.09e-04 mm.  Recomputed at the config's own 1200: the closed form moves ≤1.59e-03 mm and
flips zero verdicts; the sampled flank misses two folds outright.  At the 97 stations §54's
shipped grid uses, **the sampled test calls both folded genomes healthy and the closed form
has already rejected both** — `-0.343621` constant to six decimals across a 40x refinement,
against a sampled value that changes sign.  §54's anecdote is a mechanism: not "the grid
happened to step over it" but "any grid can, and this quantity has no grid in it."

**What it does to the arc's own table.**  Over parts that exist, the genome-robust profile
under §57's clamp clears the barrier on **14 of 14** rather than 15 of 16 — §54's named
exception disappears, because it was never this blocking's defect.  §54 already excluded
that genome from its argmax by hand; the gate is that exclusion made principled and taken
before the build.  The adoption call on the profile is unchanged: §54's surviving reason,
the shipped genome's convergence spread, is measured at the shipped genome and untouched.

**And what it is not.**  `study_tri_block` draws the same sixteen genomes from the same
seed — verified — and the margin is anti-informative there: the fold-negative genomes sit
at fixed-rule `+0.5337` and `+0.2104`, and the worst cell in the box, `-0.9597`, is
fold-clean.  §56's "what makes a region impossible" loses a candidate rather than gaining
an answer, and the negative is gated so it cannot rot into an assumption.

**Measured, not adopted, and specifically:** the DRAW is unchanged.  Applying the gate
would swap two genomes and move every genome-box number §54 through §57 published, at the
same time as the gate was being judged.  Instead the margin rides on every row and the box
is re-tallied over the survivors with the same function on the same genomes, so the
difference between the two tables is the gate's price and nothing is confounded with it.
Both artifacts diffed field-by-field: purely additive.  `make filletblock` ~200 s.

#### The successors, ranked — REVISED 2026-08-23 AFTER §58

1. **The QUALITY half of the filleted blocking, which is now the only half.**  Eight of
   sixteen sit under the barrier at the shipped profile with §57's clamp; seven of fourteen
   over parts that exist.  Two named pieces, cheap one first: re-derive §54's `(entry, end)`
   argmax against the now-buildable, fold-clean fourteen rather than the ten it was fitted
   to; then the two-objective version — genome-robust AND holding the shipped genome's
   deflection-convergence spread inside the ±0.3% band, which is the one reason §54's call
   still rests on and has never been attacked directly.
2. **Apply the fold gate to the draw and re-derive the box** (§58).  Priced and not taken:
   one word in a filter tuple, two genomes out, two in, and every genome-box number in §54
   through §57 moves at once.  Worth its own unit, where that movement is the subject.
3. **What makes a region impossible** (§56) — the quantity separating the 0.491-bow genome
   that refuses the curved Y from the 0.498-bow one it reaches.  Two candidates now ruled
   out with numbers: the bow (§56) and the fold margin (§58).
4. **A bend that is a FUNCTION of the genome rather than a constant** (§56).
5. **FILLET_PLAN Step 3, item 1 — `R_hub` and `R_rim` as live FEA genes.**  §57's clamp is
   the bound projection such genes need; §58 adds the other half of the feasibility pair.
6. **Step 2's part C — the p-norm on a filleted mesh.**  `make gci`, 95 minutes, 20.6 GB.
7. **Make `modelled_area_reference` fillet-aware** (§50).
8. **The REST of §45's audit list** (§49).
9. **G1's fourth revision** — §40 confirmed the gate blocks nothing.
10. **§32's successors 3 and 4** — §8's wall-floor economics under SVK.
11. **The element-validity check** (§44).

---

## §59 — 2026-08-23. THE TWO-OBJECTIVE LAYER PROFILE EXISTS, IT BEATS THE SHIPPED PAIR ON BOTH OBJECTIVES AT ONCE, AND IT IS THE CELL EVERY SHORTLIST THROWS AWAY

§58's rank-1 item was the quality half of the filleted blocking, in two named pieces: the
cheap one, re-deriving §54's `(entry, end)` argmax against the now-buildable fold-clean
cells; and the open one, whether §54's genome-robustness can be had without §54's cost.

**The cheap piece is a negative and a useful one.**  Re-derived over fifteen cells rather
than ten, the argmax appears to move to `(-0.90, 0.80)` — and that cell refuses one of the
fifteen.  Ranking on "the worst over the genomes that BUILT" pays a cell for refusing a
hard genome; §54's own argmax happened to sit where nothing refused, so its answer was
never biased, but the clamped cells reach the corner where the profile itself starts
refusing and there the bias bites.  On the corrected rule the re-derivation **reproduces
§54's pair exactly**.  The argmax is not stale, so everything rested on the convergence
cost — which is one number, measured at one alternative pair, on a broad ridge.

**Pricing it needed the profile threaded to a full `build_wheel`.**  `sector_blocks`,
`_sector_coords` and `build_wheel` now take `layer_profile=(entry, end)`; `None` is the
shipped pair and the default path is asserted bit-identical three ways, with a test on the
other side so that bit-identity is not vacuous.  Three linear solves per pair, nine
seconds a ladder.

**Then two wrong answers, which is the part worth keeping.**  The top eight of the ridge
all failed the band and all had a steep entry — a clean negative, nearly written down.  The
entry ladder falsified it: at end 1.60 every entry from -0.45 to -0.90 holds the band.  So
the cost is carried by `end` — also wrong.  Enumerating the whole candidate set shows every
entry straddling the band and all but two ends straddling it: **neither variable alone
predicts the cost**, the failing set is the middle of the space, and it covers almost all
of the barrier-clearing region.

```
  profile                      genome-box floor (15 cells)     convergence, coarse..fine
  shipped       (-0.45, 1.60)   0.1194  under MIN_SJ_TARGET      0.141%  inside the band
  §54's pair    (-0.75, 0.70)   0.2430  CLEARS                   0.512%  OUTSIDE
  (-0.80, 1.00)                 0.2061  CLEARS                   0.110%  inside, and BETTER
```

`entry = -0.80, end = 1.00` clears the barrier on all fifteen, refuses none, and holds the
deflection band more tightly than the pair that ships — floor +73%, spread 0.141% ->
0.110%.  **It dominates the shipped profile on both measured objectives at once**, and it
is the one cell of fourteen that every shortlist drops, because it has the lowest
genome-box floor of the fourteen that clear the barrier.  A top-k rule ranks it fourteenth
of fourteen.  The candidate set is now the criterion that matters — clears
`MIN_SJ_TARGET` on the whole box, refuses none of it — and a self-check re-derives it.

**Not adopted here, and for a different reason than the last three times.**  Those were
declines: adoption would have traded away a published result.  This one would not — nothing
measured gets worse.  It is deferred because `FILLET_LAYER_ENTRY_SLOPE`/`END_OFFSET` are
the geometry underneath every filleted number this arc has published, and moving them
re-dates all of it at once.  The pair is named, measured, tested and unwired; the promotion
is its own unit with its own baseline.  Both artifacts diffed field-by-field: purely
additive.  `make corner-fillet` ~180 s.

#### The successors, ranked — REVISED 2026-08-23 AFTER §59

1. **Adopt `(-0.80, 1.00)` as the layer profile.**  The audit is the work, not the
   decision: every filleted artifact re-derived and re-dated, §54's convergence and 38%
   figure re-measured, the two constants moved, and `test_promotion.py`'s checklist
   extended to cover a layer-profile change the way it covers a genome change.
2. **A finer grid around `(-0.80, 1.00)`** (§59).  It is a grid point of a sweep laid out
   for a different question — ends jump 0.80 -> 1.00 -> 1.30 — so the band-holding,
   barrier-clearing region is located but not resolved, and its best point is not known to
   be this one.  Cheap, and it should precede item 1 rather than follow it.
3. **Why the middle of the profile space fails** (§59).  Short end plus steep entry spreads
   ~0.5%; unnamed mechanism, and §54's reading — that the fillet's convergence comes from
   removing the corner singularity — does not obviously predict a filleted region that is
   WORSE converged than the shipped pair.
4. **Apply the fold gate to the draw and re-derive the box** (§58).  One word in a filter
   tuple, priced, and it moves every genome-box number in §54 through §59 at once.
5. **What makes a region impossible** (§56) — three candidates now ruled out with numbers:
   the bow (§56), the fold margin (§58), and the interior point (§55).
6. **A bend that is a FUNCTION of the genome rather than a constant** (§56).
7. **FILLET_PLAN Step 3, item 1 — `R_hub` and `R_rim` as live FEA genes.**
8. **Step 2's part C — the p-norm on a filleted mesh.**  `make gci`, 95 minutes, 20.6 GB.
9. **Make `modelled_area_reference` fillet-aware** (§50).
10. **The REST of §45's audit list** (§49).
11. **G1's fourth revision**; **§32's successors 3 and 4**; **the element-validity check**
    (§44).

---

## §60 — 2026-08-23. THE REFINEMENT FINDS A BETTER PROFILE AND THEN FINDS THAT THE CRITERION RANKING IT IS PARTLY MEASURING THE CONTACT PATCH — WHICH ALSO MAKES §54's 0.141% A SINGLE-NODE READING

§59 named `(-0.80, 1.00)` and said in its own successor list that it is a grid point of a
sweep laid out for a different question, so the admissible region was located but not
resolved.  This resolves it at half the step, and produces two findings.

**A better cell, and §59's was third.**  Eleven cells of the refined grid clear
`MIN_SJ_TARGET` on all fifteen clamped fold-clean cells and refuse none; four are
admissible on every criterion at once:

```
  pair               genome-box floor    1-node spread    patch-mean spread   patch n, fine
  (-0.85, 1.00)              0.2125           0.109%            0.195%             31
  (-0.90, 1.10)              0.2062           0.091%            0.186%             31
  (-0.80, 1.00)  <- §59      0.2061           0.110%            0.170%             31
  (-0.85, 1.10)              0.2002           0.116%            0.161%             31
  shipped (-0.45, 1.60)      0.1194           0.141%            0.672%             35
```

**And then the criterion turned out to be reading something else.**  The 1-node spread
jumps fourfold between adjacent cells with coarse and medium agreeing to 0.0002 mm and the
whole difference in `fine`.  Chased rather than reported: not mesh quality (min scaled
Jacobian 0.269-0.283 across the cliff, and identical at all three configs), not aspect
ratio (a holding cell runs AR 103, a failing one 33), not topology (same element and node
counts everywhere).  **It is the contact patch.**  Every cell that holds both bands reaches
31 patch nodes at `fine`; every cell that fails either reaches 29 or 30.  The layer profile
moves the patch count because the fillet re-cuts the rim blocks, and a single-node axle
drop moves ~0.005 mm — 0.5% — when it changes.

**Which re-prices §54's headline.**  *"0.141% and flat from `coarse` up"* is
`axle_drop_mm`, one node.  The same three rungs read over the whole patch give the shipped
pair **0.672%**, outside the band by more than a factor of two — and the patch-mean had
never been recorded, at any rung, in any committed artifact.  Checked, not assumed.  §54's
OTHER finding, the 38% deflection reduction, is a magnitude at one rung rather than a
convergence claim and stands.  What falls is the reading that the filleted deflection is
settled at 0.141%, and with it the premise that the shipped layer profile is the
best-converged one available.

**Not adopted, and now blocked on a different thing.**  §59 deferred the promotion on the
size of the audit; that stands, and a second reason is in front of it.  `(-0.85, 1.00)` is
better than the shipped pair on every number measured here, and every number measured here
is a single-phase three-rung statistic just shown to carry a 0.5% contact-patch
discontinuity.  The gate's own QoI is `axle_drop_mean_mm` — eight phases, both kinematics,
`make gci`.  Promoting on a statistic this arc has just discredited would be §54's mistake
made knowingly.  Both artifacts diffed field-by-field and are purely additive.
`make filletblock` ~305 s, `make corner-fillet` ~230 s.

#### The successors, ranked — REVISED 2026-08-23 AFTER §60

1. **`make gci` on the filleted mesh at two of the four admissible profiles.**  The only
   instrument that settles the promotion, and now the thing everything else waits on.  95
   minutes a pass and 20.6 GB, so cut the four to two first: `(-0.85, 1.00)` on the floor
   and `(-0.85, 1.10)` on the patch spread bracket the set.  This subsumes what was item 8
   — Step 2's part C is the same run.
2. **Why the patch count moves with the layer profile at all** (§60).  The fillet re-cuts
   `rim_ring_free` and `rim_ring_weld`, so the rim's circumferential node distribution near
   the contact is profile-dependent — plausible, unmeasured, and if that is the mechanism
   then the effect belongs to the re-cut rather than to the profile, which would make it
   avoidable rather than a trade.  Cheap, geometry only, and it may remove item 1's
   ambiguity without the 95 minutes.
3. **Then adopt the winner** — every filleted artifact re-derived and re-dated, §54's
   convergence and 38% figure re-measured, and `test_promotion.py`'s checklist extended to
   cover a layer-profile change the way it covers a genome change.
4. **`(-0.95, 0.85)` scores 0.2448, the best floor anywhere on either grid, and refuses one
   genome** (§60).  Nothing has asked which genome or why.
5. **Apply the fold gate to the draw and re-derive the box** (§58).
6. **What makes a region impossible** (§56) — the bow, the fold margin and the interior
   point are all ruled out with numbers.
7. **A bend that is a FUNCTION of the genome rather than a constant** (§56).
8. **FILLET_PLAN Step 3, item 1 — `R_hub` and `R_rim` as live FEA genes.**
9. **Make `modelled_area_reference` fillet-aware** (§50); **the REST of §45's audit list**
   (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the element-validity
   check** (§44).

---

## §61 — 2026-08-23. THE DEFLECTION WAS BEING READ AT WHICHEVER NODE HAPPENED TO BE NEAREST THE BOTTOM, §54's 0.141% IS THAT ARTEFACT, AND CORRECTING IT REINSTATES §54's DECLINE ON A BETTER REASON

§60's item 2 was "why does the patch count move with the layer profile at all", filed as
cheap and possibly able to remove item 1's ambiguity without the 95 minutes.  It does, and
the answer is larger than the question.

**The mechanism is in `solve_wheel`, not in the fillet.**  `axle_drop_mm` is `uy` at
whichever `rim_outer` node is closest to `theta = -90`.  That would be harmless if `uy`
peaked at the bottom — but the spokes are a spiral, the wheel has no mirror symmetry about
the vertical, and `uy` runs MONOTONICALLY through the bottom at about **-0.024 mm/deg**.
So the reading carries a first-order error set by where the nearest node happens to sit,
and where it sits is a property of the blocking: the unfilleted mesh puts a node within
0.05° of the bottom at every rung, while the FILLETED mesh's re-cut rim shifts the phase
and its offset runs -0.163 / -0.076 / -0.013° up the ladder — a shrinking, h-dependent term
injected straight into the drop.

```
  FILLETED, shipped profile      coarse    medium      fine    spread     increments      ratio
  axle_drop_mm  (§54)          0.962456  0.963816  0.962579    0.141%   +0.001359 -0.001236  -0.909
  interpolated at the bottom   0.956652  0.961086  0.962100    0.568%   +0.004435 +0.001014  +0.229
```

**§54 read "0.141% and flat from `coarse` up" off a sequence whose increments flip sign.**
Read at the bottom it is monotone, spans 0.568%, and settles at ratio 0.229.  §54's OTHER
finding — the 38% deflection reduction — is a magnitude at one rung, reads -38.7% on the
corrected numbers, and stands.

**And the instrument this repo already owns separates the profiles where the band cannot.**
Across all 33 priced pairs the interpolated spread runs 0.464-0.665% — every profile,
including the one that ships — so §59's and §60's "holds the band / fails the band" split
was reading the artefact.  The ratio of successive increments does separate them, against
`SETTLING_RATIO = 0.75`, this module's own threshold:

```
  pair                    genome-box floor    interp spread    ratio    settles?   settled est
  shipped  (-0.45, 1.60)          0.1194          0.568%       0.229      yes       0.962401
  (-0.85, 1.00)                   0.2125          0.565%       0.406      yes       0.963117
  (-0.80, 1.00)  §59              0.2061          0.548%       0.458      yes       0.963508
  (-0.75, 0.70)  §54's argmax     0.2430          0.499%       0.851      NO        0.974754
```

**The ratio orders the profiles the other way up from the genome-box floor**, and it puts
§54's argmax above the settling threshold.  So §54's decline SURVIVES on a reason it did
not have: the pair does not settle, rather than its spread tripling.  §59's and §60's
candidates do settle and remain admissible.  The promotion trade is now correctly signed
and small: **+0.07% on the extrapolated deflection for +78% on the genome-box floor**, at
`(-0.85, 1.00)`.

**Nothing promoted, every mesh bit-identical, `axle_drop_mm` computed exactly as before** —
`axle_drop_interp_mm` and `patch_centre_offset_deg` are additional fields on `solve_wheel`'s
result and nothing consumes them yet.

#### The successors, ranked — REVISED 2026-08-23 AFTER §61

1. **Audit the nearest-node reading across the tree.**  Every deflection number this
   project has quoted came from `axle_drop_mm`.  On the unfilleted mesh the offsets are
   small and the error looks like ~0.1%, but that has been checked at ONE genome.  The
   ±0.3% band is one of this project's load-bearing gates and it has been evaluated with an
   instrument that snaps to nodes; `make gci`'s eight phases rotate the mesh under the
   ground, which SAMPLES the offset rather than averaging it away.  This now outranks the
   fillet work: it is cheap at one genome, it is the same class of defect as §29's
   wrong-mesh cell size, and everything downstream of it is currently unpriced.
2. **Re-derive the ±0.3% band on the interpolated reading**, wherever it is quoted.
3. **Adopt `(-0.85, 1.00)`** — trade quantified and small, audit unchanged in size.
4. **`(-0.95, 0.85)` scores 0.2448, the best floor on either grid, and refuses one genome**
   (§60).  Nothing has asked which genome or why.
5. **Apply the fold gate to the draw and re-derive the box** (§58).
6. **What makes a region impossible** (§56); **a bend that is a FUNCTION of the genome**
   (§56); **`R_hub`/`R_rim` as live FEA genes**; **`modelled_area_reference` fillet-aware**
   (§50); **the REST of §45's audit list** (§49); **G1's fourth revision**; **§32's
   successors 3 and 4**; **the element-validity check** (§44).

---

## §62 — 2026-08-23. [PARTLY WRONG — SEE §65, WHICH CORRECTS THE TWO HEADLINE CLAIMS.]  EVERY DEFLECTION NUMBER THIS PROJECT HAS QUOTED WAS READ AT WHICHEVER NODE HAPPENED TO BE NEAREST THE GROUND, AND AWAY FROM THE SHIPPED GENOME THAT COSTS UP TO 1.1% — FOUR TIMES THE BAND IT IS GATED ON

§61's item 1.  §61 found the nearest-node reading inside the fillet arc and said the
optimizer's history was *probably* safe, with "probably" doing the work.  This prices it,
and the answer is that the shipped genome is a lucky draw.

**The defect.**  `solve_wheel`'s `axle_drop_mm` is `uy` at whichever `rim_outer` node is
closest to `theta = -90`.  That is a first-order error rather than a rounding one, because
`uy` does not peak at the bottom — the spokes are a spiral, the wheel has no mirror
symmetry about the vertical, and `uy` runs monotonically through the bottom.  Pinned by
`test_the_vertical_displacement_runs_MONOTONICALLY_through_the_bottom` rather than argued.

**What it costs, over 7 genomes at `coarse`, unfilleted, on the gate's own 8-phase uniform
stencil:**

```
  genome      8-phase mean bias    worst single phase
  shipped            -0.103%             0.726%
  drawn 0            +0.077%             0.270%
  drawn 1            +0.122%             2.139%
  drawn 2            -1.133%             2.061%
  drawn 3            +0.165%             1.886%
  drawn 4            +1.019%             2.668%
  drawn 5            +0.250%             1.781%
                 |bias| max 1.133%, mean 0.410%
```

**The shipped genome is at the low end of that range.**  The tree's own headline deflection
numbers are about 0.1% off, which is why nothing has ever looked wrong — but that is luck,
not design, and the ±0.3% band this project gates on is exceeded by a factor of four
elsewhere in the box.

**It reaches the optimizer.**  `wheel_adjoint.py:644` takes `delta = float(sec
["axle_drop_mm"])` for both the objective value and the quantity whose gradient the
descent follows.  So the QoI carries a genome-dependent discretisation term of order 0.4%,
and a design can improve its reading by moving the rim's node phase rather than its
structure.  **NOT claimed: that any past optimization outcome was wrong.**  The adjoint
differentiates `uy` at a fixed node index, which is a legitimate gradient of a slightly
different functional; whether a 0.4% drift changes a descent path is unmeasured and is
filed rather than asserted.

**This was predicted in the tree and never chased.**  `phase_stencil`'s own docstring
describes `uniform` as *"the one that lets the rim's contact faceting alias into a
chaseable bias"* — and the aliasing is visible: across 8 uniform phases the unfilleted
centre-node offset takes only three distinct values.  Measured, `rqmc` does not fix it, it
RANDOMISES it: on the worst genome the systematic -1.133% becomes -0.600%, +0.614%,
+0.614%, +0.341% over four shifts, mean magnitude 0.542%.  A bias becomes noise of the same
size, which is worse for a gradient and no better for a gate.

**The fix is one `np.interp` and it is already in the tree.**  `axle_drop_interp_mm` and
`patch_centre_offset_deg` are additional fields on `solve_wheel`'s result (§61), consumed
by nothing.  Making the interpolated value THE axle drop is the right change and it is not
made here: it re-dates every deflection number in the repo at once, which is a mechanical
audit with its own baseline rather than a line in another arc's unit.

#### The successors, ranked — REVISED 2026-08-23 AFTER §62

1. **Make `axle_drop_interp_mm` the axle drop.**  The most load-bearing known defect in the
   tree, the fix is measured and free, and everything below it is priced against numbers
   that carry it.  The work is the audit: `test_promotion.py`'s checklist, every committed
   artifact re-derived and re-dated, and the ±0.3% band re-stated.
2. **Then re-derive the ±0.3% band itself** on the corrected reading, wherever quoted.
3. **Whether the 0.4% drift changes a descent path** (§62) — one short descent from the
   shipped genome under each reading, compared step by step.  Cheap relative to what it
   settles, and it decides whether §62 is a reporting defect or an optimizer one.
4. **Adopt `(-0.85, 1.00)`** as the layer profile (§61) — trade quantified and small, but it
   should follow items 1-2 rather than precede them, because its own trade was measured on
   the uncorrected reading.
5. **`(-0.95, 0.85)`, the best floor on either grid, refuses one genome** (§60).
6. **Apply the fold gate to the draw and re-derive the box** (§58); **what makes a region
   impossible** (§56); **a bend that is a FUNCTION of the genome** (§56); **`R_hub`/`R_rim`
   as live FEA genes**; **`modelled_area_reference` fillet-aware** (§50); **the REST of
   §45's audit list** (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the
   element-validity check** (§44).

---

## §63 — 2026-08-23. [MEASURES A PATH THE OPTIMIZER DOES NOT USE — SEE §65.]  THE NEAREST-NODE SNAP MOVES THE VALUE BY UP TO 1.1% AND THE GRADIENT BY 4-7%, AND THE GRADIENT ERROR GROWS WITH STEP LENGTH — SO §62 IS A GATING DEFECT FIRST AND A STEERING ONE SECOND

§62's item 3, which was filed as the measurement that decides whether §62 is a reporting
defect or an optimizer one.  Central differences of the 8-phase mean drop under both
readings, at `coarse`, same solves feeding both so the reading is the only difference.

**At the shipped genome the two gradients are the same vector:**

```
  |grad_node| 1.22142   |grad_interp| 1.22296   cosine 0.999994   |difference|/|grad| 0.0038
```

Twelve of the fourteen components agree to four or five digits; gene 7 differs by 14% of a
small component and gene 11 by 0.5% of a large one.  **The descent that produced the
shipped design was not being misled**, which is the reassuring half and it is worth having
explicitly rather than assuming.

**Away from it, it is not the same vector, and it gets worse with the step:**

```
  genome with the largest value bias (-1.133%)
    FD step 1e-3 of the gene range   cosine 0.999195   |difference|/|grad| 0.0414
    FD step 1e-2 of the gene range   cosine 0.997677   |difference|/|grad| 0.0727
    worst component (gene 11)        -0.03710 vs -0.04053   ->   -0.03522 vs -0.04240
```

Ten to twenty times the shipped genome's disagreement, and **it grows with step length**,
which is the signature of the mechanism: a longer step is more likely to cross a
node-switch, and the objective jumps by ~0.005 mm when it does.  Cosine 0.9977 is still
under 4 degrees, so the direction survives; individual components move by up to 17%.

**So the ranking of §62's consequences is now measured rather than guessed.**  The value and
the gate carry up to 1.1% — four times the ±0.3% band — everywhere except near the shipped
genome.  The gradient carries 4-7% away from the shipped genome and 0.4% at it, with the
direction intact.  A defect in what the project REPORTS and gates on, and a mild one in
what it steers by.  **Still not claimed: that any past optimization outcome was wrong.**
A 4-degree rotation with intact descent directions is not evidence of that, and nothing
here re-runs a descent.

#### The successors, ranked — REVISED 2026-08-23 AFTER §63

1. **Make `axle_drop_interp_mm` the axle drop.**  Unchanged as the top item and now
   correctly motivated: the case is the reported value and the ±0.3% gate, not a rescue of
   the optimizer.  The work is the audit — `test_promotion.py`'s checklist, every committed
   artifact re-derived and re-dated, and the band re-stated.
2. **Re-derive the ±0.3% band on the corrected reading**, wherever quoted.
3. **Adopt `(-0.85, 1.00)`** as the layer profile (§61), after items 1-2, because its own
   trade was measured on the uncorrected reading.
4. **`(-0.95, 0.85)`, the best floor on either grid, refuses one genome** (§60) — which
   genome, and why, is unasked.
5. **Apply the fold gate to the draw and re-derive the box** (§58); **what makes a region
   impossible** (§56); **a bend that is a FUNCTION of the genome** (§56); **`R_hub`/`R_rim`
   as live FEA genes**; **`modelled_area_reference` fillet-aware** (§50); **the REST of
   §45's audit list** (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the
   element-validity check** (§44).

---

## §64 — 2026-08-23. THE AXLE-DROP PROMOTION'S AUDIT, MEASURED AND SCOPED — AND HALF OF IT TURNS OUT TO BE EXACTLY IMMUNE

§63 left "make `axle_drop_interp_mm` the axle drop" as item 1 and called the work an audit.
An audit whose size nobody has measured is not a plan, and this tree's own discipline says a
promotion is never a one-file change.  So: the blast radius, enumerated, with the part that
does not need re-running separated from the part that does.

**The consumers, by what they do with the number.**  Both readings are LINEAR functionals of
the displacement field — `uy` at one node, or `uy` interpolated between two.  That is what
splits the list, and it is verified rather than assumed:

```
  same mesh, two loads, linear solve:   node ratio 1.300000   interp ratio 1.300000
  shipped genome, value error +0.118%   |   drawn genome, value error -0.195%
  ratio difference: +0.00000% at both
```

  * **CLASS A — the value is used absolutely.  Re-running required.**
    `wheel_adjoint` (objective value and gradient), `study_corner_singularity` and its
    fillet twin (the deflection ladders), `study_reds_hub_share`, `study_deflection_gci`
    (the ±0.3% band itself), `study_objective`, `study_gradient`, `study_stage3_m8bi5`.
  * **CLASS B — a RATIO of two readings on the SAME mesh.  Exactly immune where the fields
    are proportional, second-order otherwise, and NOT re-running required.**
    `study_wheel_fea`'s load ladder (exactly immune, measured above), `study_gnl` and
    `study_svk_rescore`'s linear-vs-SVK ratios (the offset is identical and both readings
    are linear functionals of their own field, so the residue is
    `(slope_svk - slope_lin) x offset`).  **This is the expensive half of the tree and it
    is the half that does not move** — which is the finding that makes item 1 affordable.
  * **CLASS C — the drop is fed BACK IN as an indentation.**  `study_m9`,
    `study_m9_buckling`, via `solve_wheel_contact_at(mesh, sec["axle_drop_mm"])`.  These
    carry the 0.1-1% error into a contact solve rather than merely reporting it, and they
    are the ones where the consequence is unpriced.
  * **CLASS D — ratios across DIFFERENT meshes, so not immune.  CORRECTED THE SAME DAY:
    THIS CLASS IS EMPTY.**  `study_contact` was put here on the strength of the phrase
    "patch-resolution matrix" without reading what its sweep varies.  It varies `eps_n`,
    the contact penalty, on a mesh built ONCE outside the loop — same mesh, same offset,
    so it is Class B.  Its other ratio compares `phase` to `phase + 30`, which is one
    whole sector of a twelve-fold wheel and therefore the same geometry against the
    ground: measured, the offsets are identical to six decimals and both ratios are
    exactly 1.000000000.  The only place ratios genuinely cross meshes is the mesh-ladder
    convergence in `study_deflection_gci`, which is Class A already.
    Recorded rather than quietly fixed because the error is the one this file keeps
    catching: a class assigned from a name instead of from the code.

**What is NOT yet known and should be settled before the promotion, not during it:**

  1. Class C's sensitivity — does a 0.1-1% shift in the indentation move `study_m9`'s
     buckling result at all?  One rerun at two indentations answers it.
  2. ~~Class D — `study_contact`'s plateau gate.~~  **Settled the same day: there is no
     per-mesh term in it to remove.**  See the correction above.
  3. Whether `axle_drop_mm` should keep its name.  The honest options are to redefine it
     (every historical number silently changes meaning) or to leave it and move the
     consumers (every consumer must be found, which is what this section is for).  The
     second is what this tree has done before and `test_promotion.py` is where the
     checklist for it belongs.

**Not started, deliberately.**  The re-runs include `make gci` at 95 minutes and 20.6 GB and
`make m8bi5` at roughly two hours, and a partial promotion leaves the tree inconsistent —
which is precisely the failure `test_promotion.py` exists to prevent.  Scoping it is the
completable piece; executing it needs a session that can finish it.

#### The successors, ranked — REVISED 2026-08-23 AFTER §64

1. **§64's remaining two unknowns**: Class C's sensitivity — does a 0.1-1% shift in the
   indentation move `study_m9`'s buckling result at all — and the naming decision.  Both
   are short and both are prerequisites for item 2.  (The third, Class D's gate, was
   settled by reading the code: the class is empty.)
2. **Make `axle_drop_interp_mm` the axle drop**, with Class B excluded from the re-run set
   and `test_promotion.py` extended to carry the checklist.
3. **Re-derive the ±0.3% band on the corrected reading.**
4. **Adopt `(-0.85, 1.00)`** as the layer profile (§61), after items 2-3.
5. **`(-0.95, 0.85)`, the best floor on either grid, refuses one genome** (§60).
6. **Apply the fold gate to the draw and re-derive the box** (§58); **what makes a region
   impossible** (§56); **a bend that is a FUNCTION of the genome** (§56); **`R_hub`/`R_rim`
   as live FEA genes**; **`modelled_area_reference` fillet-aware** (§50); **the REST of
   §45's audit list** (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the
   element-validity check** (§44).

---

## §65 — 2026-08-23. CORRECTION TO §62 AND §63: THE OPTIMIZER AND THE ±0.3% GATE DO NOT USE THE CONTAMINATED READING. I TRACED A VARIABLE NAME INSTEAD OF THE CALL THAT PRODUCED IT

§62 claimed the nearest-node reading "reaches the optimizer", citing
`wheel_adjoint.py:644`'s `delta = float(sec["axle_drop_mm"])`.  That line is real.  **What
`sec` is, I did not check**, and it decides the whole claim:

```
  wheel_adjoint:588,643   sec = fem.solve_wheel_contact(mesh, force=..., ...)
  wheel_fem:1816          _attach_contact_report:  res["axle_drop_mm"] = float(indentation_mm)
```

`solve_wheel_contact` secant-iterates the indentation until the reaction matches the target
force, and the `axle_drop_mm` it returns is **that converged prescribed indentation** — a
boundary condition, not a reading off a node.  `solve_wheel`'s nearest-node value appears in
that path exactly once, at `wheel_fem:1882`, as the secant's INITIAL GUESS, which a
converged iteration forgets.

**So the objective is immune, and so is its gradient.**  And §63's gradient comparison,
which I ran on `fem.solve_wheel` directly, measures a quantity the optimizer does not
consume.  Its numbers are correct about `solve_wheel` and irrelevant to the descent.

**The gate is immune too, and its own file says so in as many words.**
`study_deflection_gci`'s docstring: *"THE QoI HERE IS THE GATE'S ... The gate is stated on
`axle_drop_mean_mm` under SVK: the mean over the 8-phase uniform stencil, which is what the
`deflection` term is scored on and what every promotion has been judged by"* — and it
explicitly contrasts that with `run_refinement`'s use of `fem.solve_wheel(mesh)`, *"ONE
phase, LINEAR kinematics"*.  §62's headline — "four times the band it is gated on" — is
therefore wrong: the band is not gated on this quantity.  **The file that would have told me
this is the one whose docstring I quoted two sections earlier for something else.**

**What survives, and it is still worth having:**

  * `solve_wheel`'s `axle_drop_mm` genuinely carries the nearest-node error.  §61's
    measurement of it, and the mechanism — `uy` runs monotonically through the bottom at
    about -0.024 mm/deg because the spokes are a spiral — are unaffected.
  * **§61's correction to §54's filleted convergence stands in full.**  Both corner ladders
    call `fem.solve_wheel`, so "0.141% and flat from `coarse` up" really is an artefact and
    the interpolated ladder really does settle at ratio 0.229.  That was the finding this
    whole thread came from and it is untouched.
  * The affected consumers are `study_corner_singularity` (both ladders),
    `study_reds_hub_share`, `study_wheel_fea.run_refinement`, and `study_contact`'s
    assumed-drop comparisons.  **All single-phase** — and single-phase is where the error
    is LARGEST, up to 2.7% in §62's own table, so the number that matters for them is
    bigger than the 8-phase figure §62 led with.
  * §62's 8-phase table measures a mean of `solve_wheel` that nothing in the tree computes.
    It is not wrong, it is about nobody's quantity.
  * Class B and Class C of §64's manifest are both confirmed immune, Class D was already
    corrected to empty, and **Class A shrinks to the four consumers above.**

**The lesson, which is this file's own and I did not apply it:** §29 read a convergence
order off a cell size measured on the wrong mesh; §60 caught a band separating profiles by
their contact patch; and here I read a QoI's provenance off the name of the variable holding
it.  A `["axle_drop_mm"]` subscript does not tell you which solve produced it.

#### The successors, ranked — REVISED 2026-08-23 AFTER §65

1. **Move the four affected consumers to `axle_drop_interp_mm`.**  Much smaller than §64
   scoped: no optimizer change, no gate change, no SVK re-run, no `make gci`.  Two of the
   four are the corner ladders, which §61 has already re-measured; `study_reds_hub_share`
   and `study_wheel_fea.run_refinement` are short.  The naming decision from §64 still
   applies and is now the main open question.
2. **Adopt `(-0.85, 1.00)`** as the layer profile (§61), which item 1 no longer blocks in
   any large way.
3. **`(-0.95, 0.85)`, the best floor on either grid, refuses one genome** (§60).
4. **Apply the fold gate to the draw and re-derive the box** (§58); **what makes a region
   impossible** (§56); **a bend that is a FUNCTION of the genome** (§56); **`R_hub`/`R_rim`
   as live FEA genes**; **`modelled_area_reference` fillet-aware** (§50); **the REST of
   §45's audit list** (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the
   element-validity check** (§44).

---

## §66 — 2026-08-23. THE CORRECTED READING NEARLY DOUBLES A COMMITTED CONVERGENCE ORDER — 0.880 TO 1.607 — AND FLIPS THAT LADDER'S `criterion_met` FROM FALSE TO TRUE

§65's item 1 was "move the four affected consumers to `axle_drop_interp_mm`", filed as
mechanical.  Two of the four were mechanical.  The third was not.

`study_wheel_fea.run_refinement` does a Richardson extrapolation and a GCI on
`fem.solve_wheel`'s axle drop up the `smoke..fine` ladder.  That is a CONVERGENCE RATE
computed on a quantity that snaps to the nearest node, which is the same class of defect
as §61's filleted ladder — but here there is a threshold sitting on it.  At the shipped
genome, unfilleted:

```
  cfg      n_elem       node     interp   centre offset   patch n
  smoke       960   1.499486   1.497743      -0.04518          9
  coarse     4704   1.551645   1.549819      -0.04518         23
  medium    12288   1.562981   1.563534      +0.01341         36
  fine      31200   1.570505   1.570021      -0.01170         64

  reading   ratio   observed order   Richardson   finest err   GCI      criterion_met
  node     1.5065       0.880         1.585362      0.937%    1.183%       False
  interp   2.1144       1.607         1.575841      0.369%    0.463%       True
```

**The observed order goes 0.880 -> 1.607, a factor of 1.83, and `criterion_met` flips.**
The GCI more than halves.  The offsets are individually small — a few hundredths of a
degree — but they are not the same at each rung and they do not shrink monotonically, so
what they inject into the ladder is noise in the *increments*, which is exactly what an
observed order is computed from.

**This is the gate §64 went looking for and put in the wrong place.**  §64 guessed
`study_contact`'s plateau test and was corrected the same day; the threshold that actually
moves is here.  Note what saves it from being a changed verdict: this study already
decoupled its DECISION from this number — *"The gate's decision does not rest on the axle
drop being converged to 0.5%; it rests on the compliance split being stable"* — and
`decision_robust` is computed on the rim compliance share, which is a ratio and untouched.
So the gate's conclusion stands; what was wrong is a convergence rate the file reports and
a criterion it recorded as unmet.

**AND IT DOES NOT REACH §29.**  Checked before claiming, because I got exactly this wrong
once tonight: §29's p = 0.638 — the number matched against a Williams eigenvalue and the
reason the corner arc exists — comes from `study_deflection_gci`, which computes
`axle_drop_mean_mm` through `WO.objective` and therefore through the contact path, whose
`axle_drop_mm` is a prescribed indentation.  Immune.  **The sub-second-order rate §29
chased is a different QoI and survives.**  What §66 says is narrower and still worth
saying: the OTHER ladder in this tree, the one-phase linear one, reports an order that is
nearly half of what it should be.

**Nothing promoted, `axle_drop_mm` still the node reading, every top-level key in
`study_wheel_fea.json` unchanged** — the new `extrapolation` block carries both fits side
by side and is the one to quote, and the artifact diff is additive apart from a timing
field.  `study_reds_hub_share` now carries both readings for the same reason.

#### The successors, ranked — REVISED 2026-08-23 AFTER §66

1. **Finish §65's item 1: `study_contact`'s assumed-drop comparisons**, the last of the
   four, and then re-derive any convergence rate in the tree that is computed on
   `solve_wheel`.  §66 is the demonstration that these are not cosmetic.
2. **Adopt `(-0.85, 1.00)`** as the layer profile (§61).
3. **`(-0.95, 0.85)`, the best floor on either grid, refuses one genome** (§60) — which
   genome, and why, is unasked.
4. **Apply the fold gate to the draw and re-derive the box** (§58); **what makes a region
   impossible** (§56); **a bend that is a FUNCTION of the genome** (§56); **`R_hub`/`R_rim`
   as live FEA genes**; **`modelled_area_reference` fillet-aware** (§50); **the REST of
   §45's audit list** (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the
   element-validity check** (§44).

---

## §67 — 2026-08-23. THE LAST OF THE FOUR CONSUMERS, PRICED: `study_contact`'s COMPARISONS MOVE BY 0.1-0.3 POINTS AND NOTHING FLIPS, AND CLASS B IS CONFIRMED AT 0.016 POINTS

§66's item 1 left one consumer unpriced.  `study_contact` compares a contact solve's drop —
a PRESCRIBED indentation, exact — against `solve_wheel`'s assumed-patch drop, which is the
node reading.  A mixed comparison, so unlike the same-mesh ratios it does not cancel.

Measured at the shipped genome, `coarse`, across phase:

```
   phase    contact  assumed node  assumed interp   rel node   rel interp
     0.0   1.463794      1.551645        1.549819    -5.662%     -5.551%
    10.0   1.937837      1.931691        1.932351    +0.318%     +0.284%
    20.0   1.453034      1.481613        1.485833    -1.929%     -2.207%
    30.0   1.463794      1.551645        1.549819    -5.662%     -5.551%
   spread of rel_diff                                 5.980%      5.835%
```

**Every comparison moves by 0.03 to 0.28 percentage points, on a quantity that spans six
points across phase.  No sign changes and no conclusion moves.**  The phase-0 and phase-30
rows agreeing to the last digit is the twelve-fold periodicity, unaffected either way.

**And Class B is confirmed by measurement rather than by argument.**  §64 claimed that a
ratio of two readings on the SAME mesh is immune because both are linear functionals of
their own field, and verified it only for the exactly-proportional case (two loads, ratio
1.300000 either way).  The non-proportional case is SVK against linear on one mesh, where
the offset is identical but the two fields have different slopes through the bottom:

```
   SVK vs linear, same mesh:   node +22.52054%    interp +22.53687%
```

0.016 points on 22.5 — the residue is `(slope_svk - slope_lin) x offset`, and it is
second-order as claimed.

**Nothing changed in `study_contact.py`.**  Regenerating that artifact is 1161 s and the
correction it would carry is a 0.2-point annotation that flips nothing, so the code change
belongs with the next run of that study for its own reasons rather than as a stale-artifact
hazard tonight.  The measurement is the deliverable: **all four consumers §65 identified are
now priced, and only one of them — `run_refinement`'s convergence order, §66 — mattered.**

#### The successors, ranked — REVISED 2026-08-23 AFTER §67

1. **Adopt `(-0.85, 1.00)`** as the layer profile (§61): clears `MIN_SJ_TARGET` on the whole
   clamped fold-clean box at 0.2125 against the shipped pair's 0.1194, settles at ratio
   0.406, and costs +0.07% on the extrapolated deflection.  The audit is every filleted
   artifact re-derived and `test_promotion.py` extended.
2. **`(-0.95, 0.85)`, the best floor on either grid, refuses one genome** (§60) — which
   genome, and why, is unasked, and it is cheap.
3. **Carry `axle_drop_interp_mm` into `study_contact`** next time that study runs anyway.
4. **Apply the fold gate to the draw and re-derive the box** (§58); **what makes a region
   impossible** (§56); **a bend that is a FUNCTION of the genome** (§56); **`R_hub`/`R_rim`
   as live FEA genes**; **`modelled_area_reference` fillet-aware** (§50); **the REST of
   §45's audit list** (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the
   element-validity check** (§44).

---

## §68 — 2026-08-23. THE GENOME THE BEST-FLOOR PROFILE REFUSES IS THE SHIPPED ONE, AND EVERY CANDIDATE THIS ARC PRODUCED STANDS WITHIN 0.08 OF THAT CLIFF. THE PROFILE IS NOT ADOPTED, AND THE REASON IS NOW MEASURED

§60 left `(-0.95, 0.85)` — the best genome-box floor on either grid, 0.2448 — as a
curiosity that "refuses one genome", with *"nothing has asked which genome or why"*.

**It is the shipped genome**, and the refusal is `the layer's width profile reaches zero
thickness` at the rim: a hard geometric loss of the wheel every published number in this
project is measured at.  Bisected, identical at `coarse` and `medium`:

```
  end offset   entry at which the shipped genome loses its rim layer      candidate margins
      0.85                 -0.845458           (-0.85, 1.00)  §60 candidate   0.0311
      1.00                 -0.881143           (-0.80, 1.00)  §59 candidate   0.0811
      1.10                 -0.903400           (-0.85, 1.10)                  0.0534
      1.60                 -1.001967           (-0.90, 1.10)                  0.0034
                                               (-0.75, 0.70)  §54             0.0564
                                               (-0.45, 1.60)  SHIPPED         0.5520
```

**[SCOPE, ADDED §77: TRUE OF THE FIVE PROPOSALS NAMED ABOVE, NOT OF THE CANDIDATE SET THEY
WERE DRAWN FROM.**  With the margin computed for every cell rather than five, six of the
buildable grid's fourteen candidates exceed 0.08 and the roomiest stands at **0.2329** — 7.5x
`(-0.85, 1.00)`'s — while still clearing `MIN_SJ_TARGET`.  The sentence below is a fact about
what was proposed; the paragraph after it, about the search walking toward an edge, is a fact
about what was RANKED.  Both stand, and neither is a fact about the set.**]**

**Every profile this arc has proposed stands within 0.08 of a cliff; the pair that ships
stands 0.55 from it.**  `(-0.90, 1.10)`, listed as admissible in §60 and the best-converged
of the four, clears it by 0.0034 — it builds by luck.  The whole search maximised a floor
while walking toward an edge, and the distance to that edge was never a column in any table.

**The call, re-decided rather than repeated.**  §59 deferred on the audit's size; §61 added
the gci run; §65 removed that.  By those records the blocker was down to two study re-runs,
so the deferral had to be re-taken on its merits.  **`(-0.85, 1.00)` is not adopted: a
0.031 margin to a hard refusal of the shipped genome is not a defensible place to put a
default.**  The floor it buys (0.1194 -> 0.2125) is measured on a box built with §57's clamp
and §58's fold gate, neither adopted and neither consumed by anything; the margin it spends
is on the one genome every published number uses.  Wrong side of the trade for a benefit
nothing yet collects.

**[SCOPE, ADDED §80: THE "NEITHER ADOPTED" CLAUSE IS RETIRED, AND NOT BY MEASUREMENT.**
§57's clamp went into the mesh at §74 and held out at 32 of 32 (§78); §58's fold gate was
never an unadopted mechanism — it is `wheel_objective`'s own live barrier, and what §58
declined was a DRAW filter, not a mechanism.  The clause that survives is the second one,
*"neither consumed by anything"*, and §80 shows it cannot be discharged before the item it
blocks.  The paragraph below is unchanged as the record of what was decided on 2026-08-23.**]**

Two things would change it, both concrete: a candidate with a margin comparable to the
shipped pair's, or a consumer for the filleted blocking (Step 3's live `R_hub`/`R_rim`
genes) that makes the genome-box floor a number something reads.

**Nothing promoted, no artifact regenerated, no code changed** — §68 is a measurement and a
decision.

#### The successors, ranked — REVISED 2026-08-23 AFTER §68

1. **Search the profile surface with the margin as a CONSTRAINT**, not as an afterthought.
   The boundary is one bisection per `end` (§68 records four), so a candidate set defined as
   "clears `MIN_SJ_TARGET`, refuses nothing, AND stands at least as far from the cliff as
   some stated floor" is cheap to build.  Whether anything survives it is the open question,
   and it is the one that decides whether this arc has a profile to offer at all.
2. **A consumer for the filleted blocking** — FILLET_PLAN Step 3's live `R_hub`/`R_rim`
   genes.  Until one exists the genome-box floor is a number nothing reads, which is what
   makes every profile trade above hypothetical.
3. **Apply the fold gate to the draw and re-derive the box** (§58) — one word in a filter
   tuple, priced, and it moves every genome-box number in §54 through §68.
4. **What makes a region impossible** (§56) — the bow, the fold margin and the interior
   point are all ruled out with numbers.
5. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
6. **A bend that is a FUNCTION of the genome** (§56); **`modelled_area_reference`
   fillet-aware** (§50); **the REST of §45's audit list** (§49); **G1's fourth revision**;
   **§32's successors 3 and 4**; **the element-validity check** (§44).

---

## §69 — 2026-08-23. CONSTRAIN ON THE MARGIN AND THE CANDIDATE CHANGES: `(-0.70, 0.90)` HAS FIVE TIMES THE CLEARANCE FOR THREE PERCENT OF THE FLOOR — AND IT IS A CELL §60 EXCLUDED WITH THE CRITERION §61 SHOWED WAS AN ARTEFACT

> **SCOPE — THE BOX HERE IS THE UNIFORM DRAW, NOT THE DESIGN SPACE (§72, §76).**
> Every genome-box figure in this section comes from the uniform Latin hypercube in
> `study_mesh_quality.latin_hypercube` over the full gene box. That sampler puts about
> **one genome above 35 degrees of arc span in sixty-four**, and a draw conditioned on
> arc span reaches bows to 1.25 against its maximum of 0.54. The numbers below are
> correct about what was drawn; read "the box" as "this draw" throughout.

§68's item 1: build the candidate set with the cliff margin as a CONSTRAINT rather than
discovering it afterwards, and see whether anything survives.  The entry floor is one
bisection per `end`:

```
  end    0.50     0.60     0.70     0.80     0.85     0.90     1.00     1.10     1.20     1.30     1.60
  floor -0.7469  -0.7779  -0.8064  -0.8329  -0.8455  -0.8577  -0.8811  -0.9034  -0.9247  -0.9451  -1.0020
```

Twenty-four cells across both grids clear `MIN_SJ_TARGET` on all fifteen and refuse none.
Scored on all three criteria at once — box floor, margin to the shipped genome's cliff, and
the settling ratio on the interpolated deflection:

```
  pair                 box floor   refuses   margin   ratio   settled est   vs shipped
  (-0.45, 1.60) SHIPPED   0.1194       0     0.5520   0.229     0.962401      +0.000%
  (-0.60, 0.80)           0.2060       0     0.2329   0.560     0.964523      +0.221%
  (-0.70, 0.90)           0.2061       0     0.1577   0.437     0.963277      +0.091%
  (-0.85, 1.00) §60       0.2125       0     0.0311   0.406     0.963117      +0.074%
  (-0.60, 0.70)           0.2190       0     0.2064   0.680     0.966099      +0.384%
```

**`(-0.70, 0.90)` dominates §60's candidate on the criterion §68 added and gives up almost
nothing for it**: five times the margin (0.1577 against 0.0311) for 3% of the floor (0.2061
against 0.2125) and 0.017 points of extrapolated deflection (+0.091% against +0.074%).
`(-0.60, 0.80)` buys still more margin (0.2329) but pays for it in convergence — ratio
0.560 and +0.221%.  So the front is real and it is short, and the middle of it is the place
to stand.

**And `(-0.70, 0.90)` is a cell this arc already had and threw away.**  §60 listed it as
"node yes / patch NO" — it held the single-node band at 0.137% and failed the patch-mean
band at 0.471%, so it was dropped from the admissible four.  §61 then showed that BOTH of
those readings snap to nodes and that the patch band was separating profiles by their
contact-patch count.  On the reading that does not snap, it settles at 0.437, comfortably
inside `SETTLING_RATIO`.  **The criterion that excluded the best-conditioned candidate was
the artefact.**

**The adoption call does not change, and one of its two reasons does.**  §68 declined on a
0.031 margin AND on the floor being measured in a box built from two unadopted mechanisms
that nothing consumes.  The margin objection is largely answered — 0.1577 is a real
clearance, if still 3.5x less than the shipped pair's.  The second reason stands untouched
and is now the whole of it: **there is still no consumer for the filleted blocking, so the
genome-box floor these profiles compete on is a number nothing reads.**  `(-0.70, 0.90)`
replaces `(-0.85, 1.00)` as the candidate of record.

**Nothing promoted, no code changed, no artifact regenerated** — §69 is a measurement and a
re-ranking of the same shelf.

#### The successors, ranked — REVISED 2026-08-23 AFTER §69

1. **A consumer for the filleted blocking** — FILLET_PLAN Step 3's live `R_hub`/`R_rim`
   genes.  This is now the ONLY thing standing between the arc and its profile, and it is
   the item that makes every trade above stop being hypothetical.  Everything else in the
   fillet arc is measured and shelved behind it.
2. **Carry the margin into `profile_candidates` as a column** (§68) so a future search
   constrains on it.  Cheap; the boundary function is one bisection per `end` and §69
   records eleven.
3. **Apply the fold gate to the draw and re-derive the box** (§58) — one word in a filter
   tuple, priced, and it moves every genome-box number in §54 through §69.
4. **What makes a region impossible** (§56) — the bow, the fold margin and the interior
   point are all ruled out with numbers.
5. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
6. **A bend that is a FUNCTION of the genome** (§56); **`modelled_area_reference`
   fillet-aware** (§50); **the REST of §45's audit list** (§49); **G1's fourth revision**;
   **§32's successors 3 and 4**; **the element-validity check** (§44).

---

## §70 — 2026-08-23. WHAT MAKES A REGION IMPOSSIBLE: THE REFUSAL IS EXTREMAL ON THE INTERIOR-ANGLE SUM BY 57% OF THE OTHERS' SPREAD, AND WITH ONE NEGATIVE EXAMPLE THAT IS WHERE TO LOOK, NOT AN ANSWER

> **SCOPE — THE BOX HERE IS THE UNIFORM DRAW, NOT THE DESIGN SPACE (§72, §76).**
> Every genome-box figure in this section comes from the uniform Latin hypercube in
> `study_mesh_quality.latin_hypercube` over the full gene box. That sampler puts about
> **one genome above 35 degrees of arc span in sixty-four**, and a draw conditioned on
> arc span reaches bows to 1.25 against its maximum of 0.54. The numbers below are
> correct about what was drawn; read "the box" as "this draw" throughout.

§56's question, asked of the shape numbers the tri-block artifact already carries.  Three
quantities separate the single curved-Y refusal from all fifteen reached genomes at both
configs; the widest is the region's interior-angle sum, where the refusal sits **14 degrees
below the minimum of the others — 57% of their entire spread** (coarse 156.371 against
[170.330, 194.703]; medium 156.667 against [170.489, 194.906]).  `arc_span_deg` separates
by only 19% of its spread and is partly the straight-Y effect §56 already named.
`bow_over_width`, `turn_at_far_end_deg`, the smallest wedge and the A/C side ratio do not
separate — §56's negative, reconfirmed and extended.

**Not claimed as a mechanism, and the attempted derivation is reported as failing.**  Any
quantity on which a set of one is extremal will separate it from a set of fifteen.  In the
plane Gauss-Bonnet gives `interior angle sum = 180 + total boundary turning`, which would
have made this a statement about concavity — measured, the sides' turnings correlate with
`sum - 180` at **0.355**, because the three sides are not consistently oriented between
flank orientations.  The identity is presumably recoverable once they are reconciled; it is
not recovered here, so the interpretation is withheld.

**What settles it is a second refusal**, and the box has one only because sixteen genomes is
a small draw.  Drawing to thirty-two would turn all four candidates into testable claims at
once and costs about one more `make triblock` (~290 s for sixteen; the curved-Y sweep is the
expensive part).

**Nothing promoted, no code changed, no artifact regenerated** — every number is read from
the committed artifact or computed from the genes in it.

#### The successors, ranked — REVISED 2026-08-23 AFTER §70

1. **Draw the tri-block box to thirty-two genomes and find a second refusal** (§70).  It
   converts four candidate separators into tested ones in a single run, it is the cheapest
   remaining item with a real result attached, and unlike everything in the fillet arc it
   is not waiting on a consumer.
2. **A consumer for the filleted blocking** — FILLET_PLAN Step 3's live `R_hub`/`R_rim`
   genes.  The whole fillet arc (§54-§69) is measured and shelved behind this one item.
3. **Carry the cliff margin into `profile_candidates` as a column** (§68).
4. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced, and
   it moves every genome-box number in §54 through §69.  Worth bundling with item 1, which
   redraws that box anyway.
5. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
6. **A bend that is a FUNCTION of the genome** (§56); **`modelled_area_reference`
   fillet-aware** (§50); **the REST of §45's audit list** (§49); **G1's fourth revision**;
   **§32's successors 3 and 4**; **the element-validity check** (§44).

---

## §71 — 2026-08-24. THE EXPERIMENT §70 NAMED, RUN: NO SECOND REFUSAL AT SIXTY-FOUR GENOMES, AND THE CANDIDATE IT WAS BUILT TO CONFIRM DECAYED BY A FACTOR OF FOURTEEN

> **SCOPE — THE BOX HERE IS THE UNIFORM DRAW, NOT THE DESIGN SPACE (§72, §76).**
> Every genome-box figure in this section comes from the uniform Latin hypercube in
> `study_mesh_quality.latin_hypercube` over the full gene box. That sampler puts about
> **one genome above 35 degrees of arc span in sixty-four**, and a draw conditioned on
> arc span reaches bows to 1.25 against its maximum of 0.54. The numbers below are
> correct about what was drawn; read "the box" as "this draw" throughout.

§70 ranked the interior-angle sum first among three separators, said with one negative
example that was arithmetic rather than evidence, and named the fix: draw deeper until a
second region refuses.  Run at 16, 32 and 64 genomes, on a draw that is a SUPERSET of the
published box — same stream, same order, so the sixteen every other number is measured on
are the first four of each orientation and nothing above moves.

**No second refusal appeared; the curve reaches 63 of 64.**  The experiment did not do what
it was designed to do, and what it did instead is more useful:

```
  gap / reached-set spread          16 genomes   32 genomes   64 genomes
    interior-angle sum                  0.573        0.257        0.041
    |angle sum - 180|                   0.614        0.502        0.070
    arc_span_deg                        0.187        0.187        0.176
```

Each larger draw finds a reached genome closer to the refusal in angle sum — 170.3, 164.2,
157.9 against its 156.4 — so **§70's first-ranked quantity decays fourteenfold as the box
grows and its third-ranked one holds.**  `bow_over_width` is now decisively out rather than
merely unhelpful: a reached genome has bow 0.540 against the refusal's 0.491, so the refusal
is not extremal on it at all.

**A demotion, not a promotion.**  Arc span survived a fourfold box; it is still one refusal
against sixty-three.  What changed is that two of three candidates are now known to be
small-sample artefacts, and were caught being so — which is the only thing a bigger box can
do when the negative it was hunting does not appear.

**And why the negative is hard to find is itself the result:** the Latin-hypercube draw
produces almost nothing above 35 degrees of arc span — sixty-four genomes yielded exactly
one, and it is the refusal.  So the next experiment is not a bigger draw but a DIFFERENT
sampler: draw conditioned on large arc span, populate 35-45 degrees deliberately, and see
whether refusals cluster.  That is the first version of this question that could return a
mechanism instead of a candidate.

**Nothing promoted, every previously committed field reproduces exactly**; the artifact
gains `refusal_search`, a `num_points` key in each row's `fold` block, and two self-checks.
`make triblock` is ~445 s and the Makefile says so.

#### The successors, ranked — REVISED 2026-08-24 AFTER §71

1. **Draw conditioned on large arc span** (§71).  The uniform sampler cannot answer this
   question — it puts one genome above 35 degrees in sixty-four — so the next step is a
   sampler that targets the band, not a longer run of the same one.  It is the only route
   from candidate to mechanism, and it is cheap once the draw is conditioned.
2. **A consumer for the filleted blocking** — FILLET_PLAN Step 3's live `R_hub`/`R_rim`
   genes.  The whole fillet arc (§54-§69) is measured and shelved behind this one item.
3. **Carry the cliff margin into `profile_candidates` as a column** (§68).
4. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced.
5. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
6. **A bend that is a FUNCTION of the genome** (§56); **`modelled_area_reference`
   fillet-aware** (§50); **the REST of §45's audit list** (§49); **G1's fourth revision**;
   **§32's successors 3 and 4**; **the element-validity check** (§44).

---

## §72 — 2026-08-24. CONDITION THE DRAW ON ARC SPAN AND 22 OF 40 REGIONS REFUSE THE CURVE, AGAINST 1 OF 64 UNIFORM — A 35x ENRICHMENT. THE ARC SPAN IS A RISK FACTOR WITH A RATE, AND STILL NOT A GATE

§71's item 1.  §71 found no second refusal in a fourfold box and diagnosed why: the uniform
Latin hypercube puts about one genome above 35 degrees of arc span in sixty-four, so the
band where the refusal lives is unsampled and no amount of the same sampler reaches it.
Screening the stream on `arc_span_deg` before meshing costs nothing and reaches it directly:

```
  29582 drawn -> 67 above 30 degrees -> 40 of those mesh clean
  22 of the 40 REFUSE the curve at every bend and every admissible free count       55.0%
  the uniform box, for comparison                                          1 of 64   1.6%
```

**A 35x enrichment**, so the arc span is a real risk factor rather than a coincidence of one
draw — §70's third-ranked candidate, the one it discounted and the only one §71's larger box
did not decay.

**And it is still not a gate.**  Inside the band the classes overlap: refusals span
30.27-44.41 degrees, reached span 30.08-36.14.  Nothing else separates there either — the
refusals' interior-angle sums run 151.8-187.5 and their bows 0.25-1.25, straight across the
reached ranges.  The arc span predicts how often a region is impossible, not which one is.

**A methodological note worth more than the result.**  The conditioned draw reaches regions
the uniform one never produced — bows to 1.25 against a uniform maximum of 0.54 — so every
"the box spans X" statement in §51-§71 is a statement about what the UNIFORM sampler reaches,
not about the design space.  Three sections in a row were spent on statistics from a sampler
that could not visit the region under study, and the fix was two lines of screening.

**Nothing promoted; the published draw is untouched** — the band is its own stream (seed
offset +1000) and shares no genome with the box, which is what makes the two rates
comparable.  Artifact purely additive.  `make triblock` ~670 s.

#### The successors, ranked — REVISED 2026-08-24 AFTER §72

1. **What picks the refusal out of the band** (§72).  Forty genomes at a 55% failure rate,
   with every shape number overlapping, is the first well-conditioned version of §56's
   question — sixteen genomes with one refusal never was.  The band is now the testbed.
2. **A pass over "the box spans X" claims in §51-§71** (§72), which are all statements about
   the uniform sampler's reach rather than the design space's.
3. **A consumer for the filleted blocking** — FILLET_PLAN Step 3's live `R_hub`/`R_rim`
   genes.  The whole fillet arc (§54-§69) is measured and shelved behind this one item.
4. **Carry the cliff margin into `profile_candidates` as a column** (§68).
5. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced.
6. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
7. **A bend that is a FUNCTION of the genome** (§56); **`modelled_area_reference`
   fillet-aware** (§50); **the REST of §45's audit list** (§49); **G1's fourth revision**;
   **§32's successors 3 and 4**; **the element-validity check** (§44).

---

## §73 — 2026-08-24. WHAT MAKES A REGION IMPOSSIBLE, ANSWERED: THE SMALLEST CORNER ANGLE, AT AUC 0.043 INSIDE THE WIDE-ARC BAND — AND A HELD-OUT DRAW CUTS THE FITTED RULE FROM 1.000 TO 0.833 AND FALSIFIES HALF OF IT

§72 left forty band genomes at a 55% failure rate as the first well-conditioned version of
§56's question.  Twenty-two refusals against eighteen reached is a classification problem,
so it can be scored.

**The answer is the smallest interior angle of the curvilinear triangle**, and it is not
close.  Concordance between the two classes:

```
  min_wedge_deg 0.043 | wedge_sum 0.227 | |sum-180| 0.679 | arc_span 0.667
  turn_at_far_end 0.366 | A_over_C 0.429 | bow_over_width 0.472
```

AUC 0.043 means a random refusal has a smaller minimum corner angle than a random reached
region 95.7% of the time — and it is mechanically the right shape, since a transfinite blend
has to squeeze a structured grid into a sharp corner and that is what folds.

**It is a quantity §70 tested and dismissed**, and the reason is now visible: ten of the
sixty-three uniform reached genomes have a minimum wedge under 17 degrees and **none has an
arc span above 30**.  A sharp corner is harmless in a narrow region.  The two factors are
CONJUNCTIVE — which is why six sections of one-quantity-at-a-time searching found nothing,
and why §72's conditioned sampler was the precondition for seeing it.

**And then the hold-out, which is the part worth keeping.**  A conjunctive rule —
`arc > 36.16 OR (arc > 30 AND min_wedge < 17.12)` — fits all 104 genomes measured so far
perfectly: 23 of 23, no false positives, accuracy 1.000.  Both thresholds are fitted on the
data they are scored on and one lands 0.02 degrees from the nearest counterexample, so that
number is not evidence.  Frozen and scored on a fresh band from a disjoint stream:

```
  30 held-out genomes, 12 refuse (40% base rate)
    accuracy 0.833   precision 0.733   recall 0.917     (majority baseline 0.600)
```

**And it falsifies half the rule outright**: a region with arc span 39.97 and min wedge
20.27 was REACHED, so "a wide enough arc always refuses" — six for six in sample — is false.

**Established:** the minimum wedge dominates inside the difficult regime; the arc span sets
how often a region is in that regime; the two are conjunctive.  **Not established:** any
threshold.  §56 asked for a mechanism; this is a mechanism with an unvalidated threshold,
and the in-sample 1.000 would have been the wrong number to publish.

**Nothing promoted, no code changed, no artifact regenerated** — every number is computed
from the committed `arc_span_band` and `refusal_search` sections except the hold-out, whose
stream is named in UNCAP_PLAN PART 9 so it can be re-run.

#### The successors, ranked — REVISED 2026-08-24 AFTER §73

1. **Calibrate the two thresholds on a proper hold-out protocol** (§73) — fit on the band,
   score on a disjoint band, repeat over several streams.  The structure is settled; only
   the numbers are not, and the machinery to do it now exists and costs ~150 s a stream.
2. **A consumer for the filleted blocking** — FILLET_PLAN Step 3's live `R_hub`/`R_rim`
   genes.  The whole fillet arc (§54-§69) is measured and shelved behind this one item.
3. **A pass over "the box spans X" claims in §51-§71** (§72), which are statements about the
   uniform sampler's reach rather than the design space's.
4. **Carry the cliff margin into `profile_candidates` as a column** (§68).
5. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced.
6. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
7. **A bend that is a FUNCTION of the genome** (§56) — now with a target to fit against;
   **`modelled_area_reference` fillet-aware** (§50); **the REST of §45's audit list** (§49);
   **G1's fourth revision**; **§32's successors 3 and 4**; **the element-validity check**
   (§44).

---

## §74 — 2026-08-24. THE CLAMP §57 MEASURED IS IN THE MESH. `fillet=True` BUILDS ON 16 OF 16 DRAWN GENOMES WHERE IT BUILT ON 10, THE SHIPPED MESH IS BIT-IDENTICAL, AND §48's SCOPE NOTE LOSES ITS FIRST CLAUSE

§57 measured the fix for the filleted blocking's refusal half and then, correctly for that
section, left it in the study: *"`clamped_radii` is called by nothing outside the study."*
Three revisions of the ranking have carried "a consumer for the filleted blocking" as the
item everything else waits behind.  This is the half of it that is a mesh change.
FILLET_PLAN STEP 1 RECORD PART 21 has the full record.

**§48's scope note reads, in full**: *"`fillet=True` lands as a MEASUREMENT INSTRUMENT for
one genome.  It must NOT be wired into `wheel_objective` or the GA — 6 of 16 feasible
genomes refuse it outright and 6 of the 10 that build sit under the barrier."*  The first
clause is now retired and the second is untouched:

```
  clamp        genomes that BUILD
  None                  10/16      <- §48 FINDING 6, reproduced from the mesh
  0.95                  16/16
```

Six genomes clamp and they are the six refusals, one for one.  **The clamp is inert at the
shipped genome** — hub 0.6636 against a limit of 3.1297, rim unlimited — so
`build_wheel(fillet=True)` is bit-identical at `coarse` and `medium` with the clamp on, off,
and passed explicitly, and every number in §50-§69 stands.

**A CLAMP IS NOT A GATE AND THE MESH SAYS WHICH IT DID.**  §57's condition on adopting this
was that *"the projected value is what is reported back"*, and the pull-backs are not
cosmetic — `0.5533 -> 0.0664` on one drawn genome.  `mesh.fillet_radii_mm` and
`mesh.fillet_clamped` carry what was built; both are `None` on the unfilleted path, because
"there is no fillet" and "nothing was clamped" are different claims.

**IT APPLIES TO `fillet=True` ALONE.**  An explicit `fillet=(R_hub, R_rim)` is honoured or
refused exactly as before, which is what every existing caller needs — `make fillet`'s fold
table and the study's radius grid pass pairs precisely in order to measure the radius, and a
clamp there would silently retitle their x-axis.

**AND THE SECOND COPY OF THE CRITERION IS GONE.**  `study_fillet_block.sector_fit_limit`
delegates its root-find to the module's.  The module bisects 40 times against 80, moving the
shipped hub limit by **3.5e-12 mm** — eight orders under the 1e-4 these margins are quoted
to — and the artifact is regenerated for that digit.

**Nothing promoted, `best_solution.json` untouched.**  `mesh_coords` still refuses a filleted
mesh and `test_nothing_wires_the_fillet_into_the_objective` is still green: this changes what
the instrument can BUILD, not who may use it.

---

## §75 — 2026-08-24. FILLET_PLAN STEP 3's ACCEPTANCE TEST PASSES: THE `R_hub` SWEEP STOPS BEING BIT-IDENTICAL AFTER TWENTY-ODD SECTIONS OF FILLET WORK — AND THE TERM THE OBJECTIVE PRICES `R_hub` THROUGH IS EXACTLY FLAT OVER HALF THE FEASIBLE RANGE

The fillet arc has had a single named acceptance test since it opened, and §74 made it
runnable.  FILLET_PLAN Step 3 item 1: *"Re-run the `R_hub` sweep from
`studies/study_reds_hub_share.py --sweep`; it should stop being bit-identical.  That sweep
is the cleanest possible acceptance test for this whole arc — it currently returns the same
17 significant figures at every point in the box, and it should not."*  Both arms below are
SVK at `coarse`, because FILLET_PLAN's cost section forbids Step 3 taking the kernel default
and a linear control against an SVK test would be a comparison of kernels.

**IT PASSES.**

```
                            hub share                    axle drop (mm)
  UNFILLETED    0.02940959791 at EVERY point       1.901083305 at EVERY point
  FILLETED      0.002888 .. 0.007755               0.889754 .. 1.152108
```

**§14's HYPOTHESIS SURVIVES ITS FIRST TEST THAT COULD HAVE KILLED IT.**  Over the feasible
range the hub share runs 0.007755 -> 0.003703 as `R_hub` goes 0.400 -> 1.900 — it RISES as
`R_hub` falls, monotonically, which is what §14 predicted and called *"the one where the
direction is surprising"*.  **And the sweep that was supposed to have killed it never
tested it**: this driver printed *"§14's hypothesis IS KILLED"* off `first["hub"] >
last["hub"]`, which is False when the two are EXACTLY equal — and on a mesh with no fillets
every row is exactly equal.  A model that cannot express a hypothesis cannot falsify it.
The verdict has a third branch now and names the reason.

**THE FINDING THAT REACHES OUTSIDE THE FILLET ARC.**  `wheel_objective` prices `R_hub`
through `stress_concentration_kt(smooth_min(R_hub, hub_fillet_cap_mm(...)), t0)`, and at the
shipped genome that cap is 0.6657 mm:

```
  R_hub    Kt_hub    axle drop        R_hub    Kt_hub    axle drop
  0.400    2.4876     1.152108        1.300    2.0683     1.030303
  0.664    2.0961     1.110221        1.600    2.0683     0.999389
  1.000    2.0683     1.065081        1.900    2.0683     0.971531
```

`Kt_hub` is **identical from 1.0 mm upward** — `wheel_objective`'s own comment says
`dKt_hub/dR_hub` is EXACTLY zero above the cap — while over that same span the wheel gets
**8.8% stiffer**.  The surrogate agrees in sign where it is alive and is exactly zero where
the wheel is still moving, and the dead half is more than half the feasible range.

**THE SHIPPED GENOME IS PARKED AT THE CAP**: `R_hub` = 0.6636 against 0.6657, 99.7% of it,
and `wheel_objective` records that tie as an attractor on purpose.  The optimizer walked the
gene up until its only live term went flat and stopped.  Between there and the feasibility
edge near 1.9 mm the filleted FEA finds another **12.49%** of axle drop.

**AND IT IS NOT A FREE 12.49%.**  The fillet is material: over the same span the modelled
area rises **+1.67%**.  The objective is blind to that too, for the same reason — the mesh
models no fillets, so the mass term cannot see `R_hub` any more than the stiffness term can.
**Both sides of the trade are unpriced**, which is the sharper claim: 12.49% of deflection
for 1.67% of area, and the optimizer has never been shown the exchange rate.

**WHAT THIS IS NOT.**  [**FIRST TWO CLAUSES SUPERSEDED 2026-08-24 — SEE §79.  The filleted
mesh is differentiable and `mesh_coords` no longer refuses it; nothing here reaching the
optimizer, and the census tests, both still hold.**]  Sensitivity, not a gradient.
`mesh_coords` and `coord_fn` still refuse a filleted mesh, nothing here reaches the
optimizer, and every census test pinning `R_hub`/`R_rim` as the insensitive pair is still
correct and still green.  What changed is that the differentiable filleted path can now be
ranked with a number instead of an intuition — which is exactly what §48 and §50 said was
missing when they shelved it.

**Nothing promoted, no threshold moved, `best_solution.json` untouched.**  The unfilleted
sweep is kept under its own artifact key, because it is the control that makes "it stopped
being bit-identical" a finding.

#### The successors, ranked — REVISED 2026-08-24 AFTER §75

1. **The differentiable filleted mesh** — now the top item and now priced.  §48 and §50
   shelved it for having no consumer; §75 gives it one and a number: 12.49% of axle drop
   for 1.67% of area, over a span where the objective's gradient is exactly zero.  The
   blocker is unchanged and specific — `_fillet_tangency` and `_fillet_curves` are bracketed
   root-finds with data-dependent refusals — and `wheel_adjoint`'s header already says the
   pieces are separated so a `custom_vjp` is mechanical.
2. **The barrier half of the filleted blocking** (§48's surviving clause) — 6 of the 10 that
   built sat under `MIN_SJ_TARGET`, and §57's clamp does not touch it.  This is what stands
   between item 1 and an optimizer that may take the path.
3. **Calibrate §73's two thresholds on a proper hold-out protocol** — the structure is
   settled, no threshold is, and the machinery costs ~150 s a stream.
4. **A pass over "the box spans X" claims in §51-§71** (§72), which are statements about the
   uniform sampler's reach rather than the design space's.
5. **Carry the cliff margin into `profile_candidates` as a column** (§68).
6. **Re-run the hub-share ladder on a filleted mesh** (§75) — `HUBSHARE_PLAN`'s gate is
   `hub < 0.03`, the filleted mesh puts the shipped genome at 0.0062 against 0.0294, and one
   rung is not a ladder.
7. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced.
8. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
9. **A bend that is a FUNCTION of the genome** (§56); **`modelled_area_reference`
   fillet-aware** (§50) — §75 makes this one bite, since the filleted mesh is ~10% larger
   and has no area cross-check at all; **the REST of §45's audit list** (§49); **G1's fourth
   revision**; **§32's successors 3 and 4**; **the element-validity check** (§44).

---

## §76 — 2026-08-24. THE "BOX SPANS X" AUDIT: SIXTEEN SECTIONS MARKED IN PLACE, TWO CLAIMS FALSIFIED OUTRIGHT, AND THE WORDING'S SOURCE FIXED IN THE CODE THAT GENERATES IT

§72's methodological note ranked this: *"every 'the box spans X' statement in §51-§71 is a
statement about what the UNIFORM sampler reaches, not about the design space."*  This is that
pass.  It changes no measurement — every number in those sections is correct about what was
drawn — and it changes what they are claims ABOUT.

**THE SAMPLER IS ONE FUNCTION, WHICH MAKES THE SCOPE ONE SENTENCE.**
`study_mesh_quality.latin_hypercube` over the full gene box via
`wg.bounds_arrays(WFEA.GENE_SPACE)`, called from four places across the two blocking
studies, all at the same `GENOME_SWEEP_SEED = 20260823` — which is why both studies draw the
same sixteen genomes.  There is no second sampler and no sub-box anywhere, so one qualifier
covers every affected section.

**MARKED IN PLACE, NOT REWRITTEN**, because these are dated records of what was measured:
seven sections of `PLAN.md` (§53, §55, §56, §58, §69, §70, §71), six PARTs of
`UNCAP_PLAN.md` (2-7) and three of `FILLET_PLAN.md` (13, 14, 15) now carry a scope block at
the head naming the sampler and its reach.  **Per section rather than per claim** —
about forty-five individual claims are in scope, and a reader who jumps to §53 for its arc
ranges needs the caveat where they land, not forty-five times over.

**TWO CLAIMS ARE NOT SCOPE PROBLEMS, THEY ARE FALSE, AND THEY ARE MARKED AS SUCH.**

- **"The largest bow in the box, 0.498, is a genome the curve reaches"** (§56, and the same
  sentence in UNCAP PART 4).  §72's conditioned draw reaches bows to **1.25** against this
  sampler's maximum of 0.54, so 0.498 is not the largest bow the design space holds.  The
  comparison between the two genomes stands; the superlative does not.
- **§56's heading, "A FIXED RULE VALID ON THE WHOLE REACHABLE BOX"** — §72 and §73 put 22
  refusals of 40 in a band that draw never visited.  Marked in the heading, where §62 and
  §63's corrections also live, so the section cannot be read standalone.

**AND THE WORDING HAS A SOURCE IN THE CODE.**  `profile_candidates`' docstring read *"Every
cell that clears the barrier on all cells and refuses none of them"* — without saying whose
cells.  That sentence is where §59's and §69's "clears the barrier on the whole box" came
from, so it is fixed at the generator rather than only at the two places it printed.

**THE ONE THING THIS AUDIT DID NOT FIND** is a claim whose NUMBER is wrong.  Every count,
range and rate in the marked sections reproduces; what was wrong was the noun they attached
to.  That is worth stating, because "the box" appearing forty-five times looked at the outset
like a defect list and turned out to be one wording error propagated by copying.

**Nothing promoted, no code path changed, no artifact regenerated** — one docstring, sixteen
scope blocks and three corrections.

---

## §77 — 2026-08-24. §68's CLIFF IS A COLUMN NOW, IT REPRODUCES THAT SECTION'S FOUR HAND BISECTIONS TO 1e-7, AND THE FINDING IT CARRIES IS PINNED BY A CHECK INSTEAD OF A SENTENCE

§68 ended on a diagnosis of its own search: *"The whole search maximised a floor while
walking toward an edge, and the distance to that edge was never a column in any table."*
It is one now.

**WHAT THE CLIFF IS.**  For a fixed `end`, the `entry` at which the SHIPPED genome loses its
rim layer — `the layer's width profile reaches zero thickness`, a hard geometric loss of the
one design every published number in this project is measured at.  `cliff_entry` bisects it;
`profile_candidate_rows` carries `cliff_entry` and `cliff_margin` alongside the box floor,
once per distinct `end` rather than once per cell.

**IT REPRODUCES §68 EXACTLY**, which is the point of automating a number that a decision
already rests on:

```
  end     bisected        §68 published      delta
  0.85    -0.845458       -0.845458          1.7e-07
  1.00    -0.881143       -0.881143          3.7e-07
  1.10    -0.903437       -0.903400          3.7e-05   (§68 quoted 4 dp)
  1.60    -1.001967       -1.001967          3.2e-07
```

and so do all six of that section's margins — 0.0311, 0.0811, 0.0534, 0.0034, 0.0564, and
the shipped pair's 0.5520.

**THE REASON IS CHECKED, NOT ASSUMED.**  The blocking refuses for four distinct geometric
reasons and only one of them is the layer thinning.  A bisection that accepted any refusal
would happily report a sector-fit or tangency limit under this name — the exact class of
error PART 6 caught this file making once already — so `cliff_entry` returns `None` with the
reason when something else bounds it, and a test drives the verdict either side of the edge
to confirm which one it found.

**AND §68's FINDING IS NOW A CHECK RATHER THAN A SENTENCE.**  "Every candidate this arc
produced stands closer to the cliff than the pair that ships" is asserted over both grids,
in the study's self-checks and in the test file.  **A future grid that produced a roomier
candidate would go red** — which is the outcome that should reopen §68's call, and is a
better trigger than a line in a plan file nobody re-reads.

**AND IT FOUND SOMETHING ON ITS FIRST RUN, WHICH IS WHY IT WAS WORTH ADDING.**  §68's
sentence *"every profile this arc has proposed stands within 0.08 of a cliff"* is true of the
five proposals it listed and is not true of the candidate SET those were drawn from.  With
the margin computed for every cell:

```
  grid          roomiest candidate     margin   box floor      §60's (-0.85, 1.00)
  buildable     (-0.60, 0.80)          0.2329      0.2060      0.0311
  fine          (-0.70, 0.90)          0.1577      0.2061      (§69 found this one)
```

Six of the buildable grid's fourteen candidates clear 0.08, and the roomiest stands **7.5x**
farther from the cliff than the cell §60 published — while still clearing `MIN_SJ_TARGET`.
The trade §68 described as a search "maximising a floor while walking toward an edge" is real
but much less sharp than five cells suggested: the best floor is 0.2430 at a margin of 0.0564,
and giving up **15% of that floor** buys **4.1x the clearance**.  §69 found the fine grid's
version of this and reported 5x for 3% of the floor; the coarse grid has a bigger version
that nothing had looked at, because looking required the column.

**This is §59's failure mode again, in a section that was correcting §60 for it** — a claim
about a shortlist stated as a claim about the set.  Marked in place in both files rather than
rewritten.

**THE CALL DOES NOT CHANGE, AND NOW RESTS ON ONE REASON.**  §68 declined `(-0.85, 1.00)` on
two: a 0.031 margin is indefensible, and the floor it buys is measured on a box built from
two unadopted mechanisms.  The first no longer generalises — roomier candidates exist — but
the shipped pair still stands **2.4x** farther out than the best of them, and the second
reason is untouched.  This is the third time this arc a "measured, not adopted" decision has
had to be re-taken because the world moved under its premise, which is the lesson §57 already
recorded and this section is now the worked example of.

**Nothing promoted, no constant moved.**  `profile_candidates` still returns bare pairs and
`LAYER_PROFILE_CANDIDATES` is still what `study_corner_singularity` imports; the column is a
sibling, because widening a return type that a cross-study consumer iterates would make that
consumer pay for this file's convenience.  The artifact gains keys and loses none.

---

## §78 — 2026-08-24. §74's "16 OF 16" SURVIVES A HELD-OUT DRAW AT 32 OF 32, THE BARRIER HALF IS CONFIRMED OPEN AT EXACTLY HALF THE BOX ON BOTH — AND THE HELD-OUT GENOMES CONTAIN THE FIRST RIM REFUSAL, WHICH NARROWS §57's WORDING RATHER THAN ITS MECHANISM

§74 retired the first clause of §48's scope note on sixteen genomes **that the clamp was
designed against**, and UNCAP_PLAN PART 9 had just finished showing this project what an
in-sample rate is worth: a rule fitting 104 genomes at accuracy 1.000 scored **0.833** on a
disjoint stream and had half of itself falsified. §74's number deserved the same treatment
before anything was ranked behind it.

Same sampler, same feasibility filter, same config, a stream disjoint by construction and
checked from the genes: `GENOME_SWEEP_SEED + 7000`, eight per flank orientation, **32
genomes**.

```
                                    in-sample (16)      held-out (32)
  shipped profile, no clamp        10/16 built         26/32 built
                                    4/16 clear         13/32 clear
  shipped profile, clamp 0.95      16/16 built         32/32 built
                                    8/16 clear         16/32 clear
  genome-robust,   clamp 0.95      16/16 built         32/32 built
                                   15/16 clear         31/32 clear
```

**THE REFUSAL HALF IS GENUINELY CLOSED.** 32 of 32, on genomes the clamp never saw. This is
the outcome UNCAP_PLAN's rule did not get, and the reason is worth stating rather than
enjoying: §57's clamp is not a fitted threshold. It is a **projection onto a limit each
genome computes for itself**, so there was no parameter to overfit — the only free number is
the factor, and §74 already measured every value in 0.75-0.99 building all sixteen.

**AND THE BARRIER HALF IS CONFIRMED OPEN, AT EXACTLY HALF THE BOX BOTH TIMES.** 8/16 and
16/32 clear `MIN_SJ_TARGET` at the shipped profile. §48's surviving clause is not an artefact
of the first draw, and ranking item 2 keeps its place.

### The held-out box contains the first rim refusal

All six in-sample refusals bind at the **hub**, and §57 and §74 both wrote the predictor up
in those words — *"the hub margin classifies 16 of 16"*. One held-out genome refuses at the
**rim**:

```
  hub margin alone                 31/32
  sector-fit margin, either junction  32/32
```

**The mechanism is untouched** — a tangent point swept past the next sector's corner is the
same event at either ring, and the clamp already applied to both. What was in-sample was the
**junction in the sentence**, not the physics. Corrected in the self-check
(`the_sector_fit_margin_predicts_every_refusal_held_out`) and pinned by a test that goes red
if the rim refusal ever disappears from the draw, because the narrow form is the one that
would quietly come back.

### The per-genome layer profile, priced — and it does not rescue §68

§68 declined `GENOME_ROBUST_*` because it spends the shipped genome's cliff margin down to
~0.056 against the shipped pair's **0.5520**. Every candidate that section weighed was a
**global** pair, and a global pair has to be safe for the tightest genome in the box. §77's
`cliff_entry` makes the per-genome version measurable for the first time — the layer-width
cliff is a property of the genome, exactly as the sector-fit limit is — so each genome can
take a share of **its own** room. That is the shape §57's clamp works in, and it is the
obvious repair.

At `end` = 0.70, each genome built at `factor * cliff_entry(genome)` with its radii already
inside the sector-fit clamp (held-out box; the in-sample box agrees):

```
  factor   built    clears 0.2   shipped margin
   0.95    32/32      29/32          0.0403
   0.85    32/32      31/32          0.1210
   0.75    32/32      31/32          0.2016
   0.65    32/32      31/32          0.2822
   0.55    32/32      31/32          0.3629
  --------------------------------------------
  genome_robust (global)   31/32     0.0564
  shipped pair  (global)   16/32     0.5520
```

**IT DOMINATES THE GLOBAL PAIR §68 DECLINED.** At 0.75 and below the per-genome rule clears
**the same 31 of 32** as `GENOME_ROBUST_*` while leaving the shipped genome **0.2016 to
0.3629** of cliff margin against that pair's **0.0564** — between **3.6x and 6.4x the
clearance for identical barrier performance**. That is not a trade, it is the same purchase
at a lower price, and it is available because the limit is per-genome: a global pair has to
be safe for the tightest genome in the box and pays for that everywhere.

**AND THE FIRST VERSION OF THIS SECTION CONCLUDED THE OPPOSITE, OFF A MISCOUNT.**
`cliff_entry` returns `None` in two cases that read identically in a tally and mean opposite
things: *"bounded by something else"* is a measurement that failed, and *"builds across the
whole bracket"* is a genome with **no layer-width edge to stand back from at all** — the
safest case in the box. All three such genomes are the second kind. Tallied as losses they
made the rule look like it built 29 of 32 and cleared 28, and the section published
"dominated on both axes". Given the global entry instead — exactly as `_clamp_to_sector`
honours a radius when the junction has no limit — they build, and the rule clears 31. **The
conclusion inverted on the handling of three genomes.** Same class of error as reading a
bit-identical column as a falsification (§75): a `None` meaning "nothing to do here" tallied
as a `None` meaning "this did not work".

**WHAT THIS DOES AND DOES NOT REOPEN.** §68 declined `GENOME_ROBUST_*` on two reasons: a
~0.056 margin is indefensible, and the floor it buys is measured on a box built from
unadopted mechanisms. **The per-genome rule answers the first and not the second.** It also
does not beat the shipped pair, which still holds the most margin at 0.5520 — but the shipped
pair clears 16 of 32 against this rule's 31, so the comparison between those two is a real
two-objective choice and no longer a one-sided one. **Nothing is adopted here**: this is a
measurement, it is not wired into anything, and the module constants are untouched. What it
changes is that ranking item 2 now has a candidate with a number rather than an open
construction question.

### What is unchanged

**Nothing promoted, no module constant moved, `best_solution.json` untouched.**
`FILLET_LAYER_ENTRY_SLOPE`, `FILLET_LAYER_END_OFFSET` and `SECTOR_FIT_CLAMP` are as they
were. The committed box is **not re-drawn** — `sector.genomes`, `fit_clamp`, and both
configs' existing keys are unchanged field-for-field against the previous artifact, which was
checked rather than assumed; the artifact gains keys and loses none. `make filletblock` goes
301 s -> **462 s**.

#### The successors, ranked — REVISED 2026-08-24 AFTER §78

1. **The differentiable filleted mesh** — unchanged at the top, and §78 strengthens its
   premise rather than its price: the instrument it would differentiate now builds on a
   disjoint draw at 32 of 32.
2. **The barrier half of the filleted blocking** (§48's surviving clause) — confirmed open
   out of sample at 16 of 32, and §78 has priced a repair that **clears 31 of 32 at 3.6-6.4x
   the cliff margin of the global pair §68 declined**. It is measured and not adopted, and
   adopting it is a decision with §68's second reason still standing against it. That
   decision is now the item, and it is a smaller one than "make the blocking genome-robust"
   was.
3. **Calibrate §73's two thresholds on a proper hold-out protocol** — unchanged, and §78 is
   the second worked example of the protocol in this project.
4. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced.
5. **Re-run the hub-share ladder on a filleted mesh** (§75) — one rung is not a ladder.
6. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
7. **A bend that is a FUNCTION of the genome** (§56); **`modelled_area_reference`
   fillet-aware** (§50); **the REST of §45's audit list** (§49); **G1's fourth revision**;
   **§32's successors 3 and 4**; **the element-validity check** (§44).

**Two items left this list by being done rather than by being re-ranked**: §72's "box spans
X" audit and §68's cliff column landed as §76 and §77, and the ranking revised after §75 was
still carrying both.

---

## §79 — 2026-08-24. THE FILLETED MESH IS DIFFERENTIABLE. `R_hub` AND `R_rim` GO FROM AN IDENTICALLY ZERO GRADIENT TO THE TWO LARGEST OF FOURTEEN — AND THE BLOCKER WAS NOT THE ROOT-FINDS, IT WAS THE FOUR REFUSALS AROUND THEM

Ranking item 1.  §75 gave it a consumer and a price — 12.49% of axle drop for 1.67% of
area, over a span where the objective's own `Kt` surrogate is exactly flat — and §78 gave
it an instrument that builds on a disjoint draw at 32 of 32.  What it did not have was a
derivative.  `wheel_wheel.mesh_coords` raised `NotImplementedError` for any mesh built
with `fillet=`, and `wheel_adjoint`'s header recorded the consequence: genes 12 and 13
have an **identically zero** gradient, so *"a gradient-based Stage 3 would optimise 12 of
14 genes and never notice"*.

**THE PREMISE WAS RE-CHECKED FIRST, AND IT HELD.**  At `8277697` both `mesh_coords` and
`coord_fn` refuse a filleted mesh, and `value_and_grad(shipped, "coarse", "axle_drop")`
returns `-0.0` in slots 12 and 13 against a value of 1.4637943597 mm.  Two minutes, and
this project has re-taken more than one section for skipping them.

### IT IS DIFFERENTIABLE, AND THE CHECK IS THE ONE G3 AND G8 ALREADY MAKE

`studies/study_gradient.py` gains **G11**, which is G3's identity and G8's census taken on
`build_wheel(fillet=True)` instead of on the mesh Stage 3 builds:

```
  coarse                                  unfilleted            FILLETED
  mesh_coords vs build_wheel             4.26e-14 mm          4.97e-14 mm   [< 1e-9]
  genes with dcoords/dgene == 0           R_hub, R_rim          NONE
  largest |dcoords/dgene|, mm/mm          t3    45.15         R_rim  165.61
  second                                  t0    43.20         R_hub  142.30
  third                                   cy1   22.54         cy4     39.14
```

**THE TWO DEAD GENES ARE NOW THE TWO LARGEST MOVERS OF ALL FOURTEEN**, by a factor of 3.6
over the next one.  That is G8's census, run on the other mesh, inverted.

**And the reference is the EAGER BUILD, not the traced path's own arithmetic**, because
the risk that matters here is a derivative that is self-consistent and wrong.
Central-differencing `build_wheel(fillet=True)` re-scans, re-brackets and re-bisects at
every perturbed genome, and compares the whole coordinate array rather than a scalar that
could cancel:

```
  d(coords)/d(gene) vs a central difference of build_wheel(fillet=True), coarse
              max |dc/dg|   h/range = 1e-4     1e-5        1e-6
    t0           0.5212           2.2e-08    2.60e-10    2.2e-09
    R_hub        2.0051           3.3e-04    1.31e-10    1.6e-09
    R_rim        2.3808           1.7e-04    3.37e-10    4.2e-09     [gate 1e-6]
```

And at the end of the chain, the quantity Stage 3 optimises, at `coarse`:

```
                        drop (mm)       d/dR_hub        d/dR_rim     worst fd resid
  linear  FILLETED     0.8638515928   -0.1270583265   -0.1056336734    3.03e-07
  SVK     FILLETED     0.9942561792   -0.1428630400   -0.1502833900    2.33e-06
  either  unfilleted   1.4637943597    -0.0            -0.0             n/a
```

Every residual is against a central difference of the WHOLE load-controlled solve on
filleted meshes, at the tightened reference secant G9 uses and for G9's reason, and all
four clear G9's 1e-5 gate — three of them by two decades and the fourth, `R_rim` under
SVK, by four times.  The linear rows are the committed ladder's minimum over `h/range` =
1e-4..1e-6; the SVK row is a single rung at 1e-5 and is not in the artifact, because
`study_gradient.json` is a LINEAR artifact and re-baselining it was not on the table.
`R_hub`'s linear ladder reads 3.63e-5 -> **2.41e-07** -> 1.54e-6, with a rung wider at
1e-3 measured separately at 3.16e-3 — truncation on one side, the reference's own floor on
the other, and a clean minimum between them.

### THE BLOCKER WAS NAMED WRONG, AND THAT IS THE FINDING

Every ranking of this item since §48 has said the same thing: `_fillet_tangency` and
`_fillet_curves` *"are bracketed root-finds with data-dependent refusals, and are
numpy-only"*.  Those are two obstacles, not one, and **only the second was real**.

**THE ROOT-FINDS WERE NEVER THE OBSTACLE.**  A converged root needs no tracing.  Seed it
from the eager build and take **one Newton step**,

```
    s* = s_A - f(s_A, p) / (df/ds)(s_A, p)
```

and because `f(s_A, p0) = 0` the value is `s_A` to the bisection's own tolerance while
`ds*/dp = -(df/dp)/(df/ds)` is the implicit-function answer, exactly.  That is the same
argument `wheel_adjoint`'s header makes about `r(c, u) = 0`, one level lower down, and it
is why the traced path reproduces the eager mesh to 5e-14 mm rather than to a tolerance.
Differentiating the bisection instead puts 200 halvings and their `sign` comparisons on
the tape and returns zero.

**THE REFUSALS WERE THE OBSTACLE, AND THE ANSWER WAS TO STOP ASKING.**  `_fillet_curves`
can refuse in four ways and none of them has a derivative: they are `sign` tests, and a
genome that refuses at the linearisation point has no mesh whose nodes could move.  So
they are **frozen** — along with the void side and the three angle unwraps — in a `_roots`
record the eager build harvests and the traced path consumes.  **This is not a new kind of
decision**: it is exactly what `mesh_coords` has always done with the flank orientation
and the seam ownership, and its docstring already had the words for it — *"a step that
flips one is a genuine discontinuity of the design space, not a defect in this module"*.

**AND A `custom_vjp` WOULD HAVE BEEN THE WRONG INSTRUMENT.**  `wheel_adjoint`'s header
says the pieces are separated so that adding one later is mechanical, and testing that
claim was half the job.  **The separation half held completely** — `wheel_adjoint.py`
needed no code change at all, because `adjoint_grads` already takes `mesh=` and already
routes through `jax.vjp(mesh_coords)`, so a filleted mesh flows through the entire adjoint
untouched.  The `custom_vjp` half is answered differently: **there was nothing to wrap.**
A wrapper supplies a derivative for a function that has one and computes it badly; a
bracketed bisection does not have one at all.  What was needed was the root's own
equation, which the bisection had already solved and thrown away.

### THE ONE DESIGN DECISION THAT IS NOT OBVIOUS, AND WHAT IT COSTS

The frozen roots **depend on the genome** — a different design has a different tangency
station — so the obvious implementation closes over them, and that is wrong for the exact
reason `coord_fn`'s own docstring gives about keying its cache on the mesh object: every
finite difference and every optimizer step builds a NEW mesh, so a jaxpr with one genome's
roots baked in re-traces on every call while looking like it works.  Passed instead as a
**traced argument**, the recipe is genome-independent and the cache is a cache again:

```
  coarse, second genome onward     trace       per call     cache entries
    unfilleted                     2.29 s       0.0012 s        1
    FILLETED                       2.78 s       0.0095 s        1
```

Pinned on the cache SIZE and on the identity holding at a SECOND genome, because a
closed-over record passes every correctness test in the file and fails only in cost.

### WHAT IS STILL REFUSED, AND BECAUSE THE ANSWER WOULD BE WRONG

**A mesh whose radius the sector-fit clamp moved.**  There the built radius is
`factor * limit(genome)` and does not follow `R_hub` at all, so freezing the record would
report the gene as live where its true derivative is zero, and would miss the centreline
genes' path through the limit's own bisection — a gradient with the right direction and
the wrong length, which is `wheel_adjoint`'s named hardest-to-see failure.  It raises
instead.  **The scope that costs is measured and it is empty where this arc works**: at
the shipped genome the clamp does not bite until `R_hub` = **2.9732 mm** (0.95 x §48's
3.1297), and §75's feasibility edge is at ~1.9 mm, so the whole span §75 priced is
unclamped.

**`fillet_blocking="spoke"`**, PART 3's retired construction, which re-spreads its station
vector by ROUNDING a node count.  Nothing but `make fillet` builds it.

### WHAT IS UNCHANGED

**Nothing is wired into the objective**, and
`tests/test_corner_singularity.py::test_nothing_wires_the_fillet_into_the_objective` is
green without being touched — none of the five modules it parses passes `fillet=`, because
`mesh_coords` reads it off the mesh it is handed rather than taking a keyword.  `wheel_objective` still prices `R_hub` through the
`Kt` surrogate §75 measured exactly flat above the cap; §48's surviving clause — half of
each drawn genome box sits under `MIN_SJ_TARGET`, 8/16 and 16/32 (§78) — still stands
against letting the optimizer take this path, and it is ranking item 1 below.  **G8's
census on the unfilleted mesh is still correct and still green**, and so is every other
census test in the tree: `R_hub` and `R_rim` are still the dead pair of the mesh Stage 3
actually builds.

**Both meshes are BIT-IDENTICAL**: `fillet=None` and `fillet=True` alike reproduce their
pre-change coordinates exactly at `smoke` and `coarse`, checked against saved arrays rather
than assumed.  **`best_solution.json` untouched, no threshold moved, no module constant
moved** — `FILLET_LAYER_ENTRY_SLOPE`, `FILLET_LAYER_END_OFFSET` and `SECTOR_FIT_CLAMP` are
as they were.  `studies/study_gradient.py` goes 1029 s -> **1374 s**.

**AND `study_gradient.json` IS ADDITIVE TO THE LEAF**, which was checked rather than
assumed: against the committed artifact it gains `filleted` and changes twenty-four other
leaves, every one of them a G10 TIMING plus `elapsed_s`.  Not one measured gradient, ratio
or residual in G1-G9 moved a bit.

#### The successors, ranked — REVISED 2026-08-24 AFTER §79

1. **The barrier half of the filleted blocking** (§48's surviving clause) — up from 2, and
   it is now the ONLY thing between the filleted mesh and an optimizer that could take it.
   §78 priced a repair that clears 31 of 32 at 3.6-6.4x the cliff margin of the global pair
   §68 declined; it is measured and not adopted, and adopting it is a decision with §68's
   second reason still standing against it.
2. **Wire the fillet into `modelled_area_reference`** (§50) — promoted from item 7,
   because §79 changes what it is worth.  §75's price has TWO sides and the mass one is
   still unpriced and now conspicuously so: the stiffness term has a gradient in `R_hub`
   and the area term has neither a gradient nor a reference, so the exchange rate the
   optimizer would need is half-built rather than absent.  `area_report` still withholds
   the reference for a filleted mesh.
3. **Calibrate §73's two thresholds on a proper hold-out protocol** — unchanged.
4. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced.
5. **Re-run the hub-share ladder on a filleted mesh** (§75) — one rung is not a ladder.
6. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
7. **A bend that is a FUNCTION of the genome** (§56); **the REST of §45's audit list**
   (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the element-validity
   check** (§44).

**And one item leaves this list by being done**: the differentiable filleted mesh was item
1 in the §75 and §78 rankings and is §79.

---

## §80 — 2026-08-24. §68's SECOND REASON IS ANSWERED BY INVENTORY, NOT BY MEASUREMENT: ONE OF ITS TWO MECHANISMS IS IN THE MESH AND THE OTHER IS THE SHIPPED OBJECTIVE'S OWN BARRIER — AND WHAT SURVIVES OF IT IS A DEADLOCK RATHER THAN A REASON

Ranking item 1 is the barrier half, and §78 and §79 both ranked it with the same clause
attached: *"adopting it is a decision with §68's second reason still standing against it."*
Three sections have cited that reason and none has re-read it.  It is one grep, and it was
run before anything else.

**THE SENTENCE, IN FULL.**  §68: *"The floor it buys (0.1194 -> 0.2125) is measured on a box
built with §57's clamp and §58's fold gate, neither adopted and neither consumed by
anything."*  Two mechanisms are named and neither of them is §38's uncap default, which was
adopted on 2026-08-18 and was never part of this objection.  The clause has two halves and
they have different fates.

### THE "NEITHER ADOPTED" HALF IS DEAD, AND IT DIED TWO DIFFERENT DEATHS

**§57's clamp is in the mesh, and it has been since the day after §68 was written.**
`wheel_wheel.SECTOR_FIT_CLAMP = 0.95`; `_clamp_to_sector` acts on the `fillet=True` branch
of `sector_blocks`; the applied radii come back as `mesh.fillet_radii_mm` and
`mesh.fillet_clamped`, which is the condition §57 set for adopting it; and
`study_fillet_block.sector_fit_limit` no longer keeps a second copy of the criterion — it
delegates to `WW._sector_fit_limit`.  §74 landed it at `766d443`, and §78 held it out at 32
of 32 on a disjoint stream.  A mechanism in `wheel_wheel`, consumed by the mesh builder,
pinned by tests and validated out of sample is not an unadopted mechanism.

**§58's fold gate was never one.**  `wheel_geometry.MIN_FOLD_MARGIN_MM = 0.1` with
`self_intersection_margin` is a live `soft_barrier` in `wheel_objective` and in `wheel_fea`,
calibrated over 2001 genomes, and `study_gnl`, `study_contact` and `study_wheel_fea` all
gate their draws on it.  §58's own headline says exactly this — *"THE FLANK DEFECT'S GATE IS
THE OPTIMIZER'S OWN FOLD BARRIER, WHICH FIVE STUDIES ALREADY USE"*.  What §58 declined was
narrower than a mechanism: `sweep_genomes`'s DRAW filter is still `x_order` / `hub_overlap`
plus a clean unfilleted sector and does not ask for the margin, which is why "apply the fold
gate to the draw" is still a ranked item.  But the floor §68 quotes was not measured on that
unfiltered draw: `profile_genomes_buildable` and `profile_genomes_fine` are called with
`clamp=SECTOR_FIT_CLAMP, fold_gate=True`, so every cell in that table was tallied against the
shipped objective's own verdict on each genome.  **"An unadopted mechanism" and "a genome the
shipped barrier already rejects" are opposite claims about the same number.**

So the half of §68's second reason that reads as a chain — *nothing may be adopted until the
things it was measured against are adopted* — is not a rule this arc has to live under.  It
terminated on its own, because §74 adopted the clamp on the clamp's merits and §58's gate was
adopted long before either.

### THE HALF THAT SURVIVES IS "NEITHER CONSUMED BY ANYTHING", AND IT IS A CIRCLE

That clause is still literally true, and checkable: `build_wheel` defaults to `fillet=None`;
`FILLET_LAYER_ENTRY_SLOPE` / `FILLET_LAYER_END_OFFSET` are read only by `_fillet_curves` and
`_filleted_sector_blocks`, both on the filleted path; and
`test_nothing_wires_the_fillet_into_the_objective` parses five modules for the keyword and is
green (§79).  The genome-box floor is a number nothing reads.

**But it cannot be discharged before the item it is being used to block.**  Laid end to end,
from three sections that each state their link correctly:

  * §79, item 1: the barrier half *"is now the ONLY thing between the filleted mesh and an
    optimizer that could take it"*, and adopting §78's repair carries §68's second reason
    against it.
  * §68's own unblocker for that reason: *"a consumer for the filleted blocking (Step 3's
    live `R_hub`/`R_rim` genes) that makes the genome-box floor a number something reads."*
  * §79, WHAT IS UNCHANGED: §48's surviving clause — half of each drawn box under
    `MIN_SJ_TARGET`, 8/16 and 16/32 — *"still stands against letting the optimizer take this
    path."*

The repair waits on a consumer; the consumer waits on the barrier; the barrier is what the
repair clears — §78's per-genome rule takes 16/32 to 31/32.  **That does not terminate, and
it is the reason this item has been ranked first or second in four consecutive revisions
without moving.**  The exit is not another measurement against §68's second reason.  It is
noticing that the reason is now doing work its own text does not support.

### AND THE PRICE OF BREAKING IT IS ZERO ON EVERY PUBLISHED NUMBER

Because `fillet=None` is the default and the two layer constants are read only on the
filleted path, moving them moves nothing this project has shipped, gated or promoted.  §68
reads like a risk objection — *"the margin it spends is on the one genome every published
number uses"* — but the margin it names is a margin on the INSTRUMENT's own cliff, not on the
part.  It is a value objection, *"a benefit nothing yet collects"*, and a value objection
cannot outrank the item whose whole purpose is to create the collector.

### THE CANDIDATE §68 ASKED FOR IS ALREADY MEASURED, IN TWO COMMITTED ARTIFACTS

§68's other unblocker was *"a candidate with a margin comparable to the shipped pair's"*.
Read out of the tree rather than re-run:

```
  pair                  box floor   refuses   cliff margin   ratio   settles   settled est vs shipped
  (-0.45, 1.60) SHIPPED    0.1194      0         0.5520      0.229     yes            +0.000%
  (-0.85, 1.00) §60        0.2125      0         0.0311      0.406     yes            +0.074%
  (-0.70, 0.90) §69        0.2061      0         0.1577      0.437     yes            +0.091%
  (-0.75, 0.70) §54        0.2430      0         0.0564      0.851     NO             +1.284%
```

`profile_genomes_fine` in `study_fillet_block.json` puts `(-0.70, 0.90)` at **0.20608 over
fifteen genomes with none refused**, clear of `MIN_SJ_TARGET = 0.2`;
`study_corner_singularity_fillet.json` puts its increment ratio at **0.437 against
`SETTLING_RATIO = 0.75`**, so it settles where §54's argmax does not — which is §61's
corrected criterion, not the nearest-node spread §61 showed was an artefact.  **That is
§61's trade at a better price**: §61 offered *"+0.07% on the extrapolated deflection for +78%
on the genome-box floor"* at `(-0.85, 1.00)`; `(-0.70, 0.90)` is +0.091% for +72.6% **and
five times the cliff margin**.  Nothing here was run — both numbers were already in the tree,
in two studies, and no section had put them in one table.

### WHAT IS ACTUALLY MISSING, AND IT IS ONE TUPLE ENTRY

`(-0.70, 0.90)` is an **in-sample** argmax over the sixteen-genome draw.  §78's held-out
thirty-two has been swept at the shipped profile and at `GENOME_ROBUST_*` and at nothing
else, and this project has two worked examples of what in-sample rates are worth — UNCAP_PLAN
PART 9's 1.000 -> 0.833, and §78's own protocol.  Adopting a fitted pair without that check
would repeat the error both of them exist to prevent.  The check is `(-0.70, 0.90)` added to
the held-out sweep's pair list; the run is `make filletblock`.

### THE CALL

**§68's second reason is withdrawn as a blocker on the barrier half.**  Its first clause is
false by inventory, its second is a circle that the item it blocks is the only way out of,
and the cost of being wrong about it is zero on every number this project ships.  What stands
between `(-0.70, 0.90)` and adoption is a hold-out, not an adoption chain — and a hold-out is
a measurement with a price and a verdict, which is what this arc has been unable to name for
four rankings.

**Nothing promoted, no module constant moved, no artifact regenerated, no code changed.**
`FILLET_LAYER_ENTRY_SLOPE`, `FILLET_LAYER_END_OFFSET` and `SECTOR_FIT_CLAMP` are as they
were.  §80 is a read of the tree and a decision about a sentence.

#### The successors, ranked — REVISED 2026-08-24 AFTER §80

1. **Sweep `(-0.70, 0.90)` on §78's held-out thirty-two, then adopt or reject it on that
   number.**  One entry in the pair list, one `make filletblock`.  This is ranking item 2
   from §78 and §79 with its blocker removed and its candidate already chosen; if it clears
   out of sample it is the first layer-profile decision this arc can close.
2. **Wire the fillet into `modelled_area_reference`** (§50) — unchanged from §79, and §80
   does not touch it: the mass half of §75's price is still unpriced.
3. **Calibrate §73's two thresholds on a proper hold-out protocol** — unchanged.
4. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced, and
   §80 narrows what it is worth: the gate already rides on the tables that matter, so this
   buys consistency of the DRAW rather than correctness of the floor.
5. **Re-run the hub-share ladder on a filleted mesh** (§75) — one rung is not a ladder.
6. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
7. **A bend that is a FUNCTION of the genome** (§56); **the REST of §45's audit list**
   (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the element-validity
   check** (§44).

**And a doc item closed rather than ranked**: "The decision that is a human's" still framed
`MIN_WALL_MM` as the open question *"can you hold 1.2 mm?"*.  §11 answered yes on 2026-08-05
and §13 promoted the 1.2 mm genome on 2026-08-06; the section had never been updated.  Marked
closed in place.

---

## §81 — 2026-08-25. THE HOLD-OUT §80 ASKED FOR, RUN: `(-0.70, 0.90)`'s IN-SAMPLE PARITY IS IN-SAMPLE. IT CLEARS 28 OF 32 WHERE THE PAIR IT WAS MEANT TO REPLACE CLEARS 31, AND THE BARRIER HALF'S ANSWER IS THE PER-GENOME RULE INSTEAD

§80's ranking item 1, and it is a rejection.  `MARGIN_ROBUST_ENTRY/END = (-0.70, 0.90)` now
rides `sweep_sector_fit_clamp` beside the other two profiles, so the same function scores it
on the same genomes at the same factors, in-sample and on §78's held-out thirty-two.

```
  profile                        in-sample (16)   fold-clean (14)   held-out (32)
  shipped        (-0.45, 1.60)      8/16 clear        7/14 clear      16/32 clear
  genome_robust  (-0.75, 0.70)     15/16 clear       14/14 clear      31/32 clear
  margin_robust  (-0.70, 0.90)     15/16 clear       14/14 clear      28/32 clear
                                   ^^^^ tied         ^^^^ tied        ^^^^ not tied
```

All three build 32 of 32 under §57's clamp; the difference is entirely the barrier.  **The
two pairs are indistinguishable on the box `(-0.70, 0.90)` was selected on and separated by
three genomes on the box it was not.**  Worst min scaled Jacobian out of sample tells the
same story from the other end — 0.1576 against `genome_robust`'s 0.1721 and the shipped
pair's 0.0319.

**THE PART THAT DID NOT SURVIVE IS THE PART IT WAS CHOSEN FOR.**  §69 picked this cell
because it matched the barrier performance of the pairs before it while standing five times
farther from the shipped genome's cliff, and §80 carried that forward as "every stated
objection answered".  The margin is real and unchanged — 0.1577, and it is still the only
candidate this arc has produced that was selected with the cliff as a constraint.  What was
in-sample was the *parity*.  A fine-grid argmax fitted on fifteen genomes carries about a
tenth of its barrier claim as fit, and this is the third time this project has measured that
number: UNCAP_PLAN PART 9's 1.000 -> 0.833, §78's own inverted tally, and now 15/16 -> 28/32.

**THE CALL: `(-0.70, 0.90)` IS NOT ADOPTED.**  Not because 28 of 32 is bad — it is twelve
genomes better than the pair that ships, and it settles at ratio 0.437 where §54's argmax
does not settle at all.  It is not adopted because the argument for preferring it over
`GENOME_ROBUST_*` was the tie, the tie was an artefact of the box it was fitted on, and a
better repair is already measured on the same held-out genomes.

### AND THAT REPAIR IS §78's PER-GENOME RULE, WHICH THE SAME RUN RE-CONFIRMS

```
  factor    built   clears 0.2   shipped entry   shipped cliff margin
   0.95     32/32     29/32         -0.7661            0.0403
   0.85     32/32     31/32         -0.6854            0.1210
   0.75     32/32     31/32         -0.6048            0.2016
   0.65     32/32     31/32         -0.5242            0.2822
   0.55     32/32     31/32         -0.4435            0.3629
```

**It matches `GENOME_ROBUST_*`'s held-out 31 of 32 and it is flat across 0.55-0.85 of its
only free parameter** — a 1.5x range of the factor, four values, the same number.  That is
the argument §78 made and it is the argument §57's clamp was adopted on: a rule with one
insensitive free number projected onto a limit each genome computes for itself is not a
fitted threshold, which is exactly what `(-0.70, 0.90)` turned out to be.

### WHAT IT COSTS THE SHIPPED GENOME IS ONE PAIR ON A LADDER THAT ALREADY RUNS

The per-genome rule is not free of the convergence question, and the shape of the answer is
already visible.  At factor `f` the shipped genome is built at `f x cliff_entry = f x
-0.806403`, so **adopting the rule at 0.75 makes the shipped genome's pair `(-0.6048,
0.70)`** — and `study_corner_singularity_fillet.json` brackets it without containing it:

```
  pair              ratio    settles (< 0.75)   settled est vs shipped
  (-0.60, 0.70)     0.680          yes                 +0.379%
  (-0.6048, 0.70)     ?             ?                     ?          <- what the rule implies
  (-0.70, 0.70)     0.890          NO                  +1.763%
  --------------------------------------------------------------
  (-0.70, 0.90)     0.437          yes                 +0.091%      <- the pair just rejected
```

So the trade the arc actually faces is now stated: **the per-genome rule buys three more
held-out genomes than `(-0.70, 0.90)` and costs the shipped genome roughly four times as
much extrapolated deflection, at a settling ratio near the threshold rather than half of
it.**  That is one entry — `(-0.6048, 0.70)` — on `study_corner_singularity`'s pair list, and
it is the last number between this arc and a decision it can defend.  Note also that the
entry ladder at `end` = 0.70 degrades steeply and non-linearly (0.680 -> 0.890 over 0.10 of
entry), so the bracket is not tight enough to read off: it has to be measured.

### WHAT IS UNCHANGED

**Nothing promoted, no module constant moved, nothing wired into anything.**
`FILLET_LAYER_ENTRY_SLOPE`, `FILLET_LAYER_END_OFFSET` and `SECTOR_FIT_CLAMP` are as they
were, and `MARGIN_ROBUST_*` joins `GENOME_ROBUST_*` as a measured-not-adopted constant in the
study.  **The artifact is purely additive and it was diffed rather than assumed**: three
tables gain five rows each — the new profile crossed with the factors — one `seconds` field
moves, and not one existing measurement changed.  `make filletblock` goes 462 s -> **486 s**.
All twenty-one self-checks pass, and none of them gates what the new profile clears, because
that number was the question.

#### The successors, ranked — REVISED 2026-08-25 AFTER §81

1. **Price `(-0.6048, 0.70)` on the convergence ladder, then take the barrier half's
   decision.**  One entry in `study_corner_singularity`'s pair list.  Both candidates are
   measured on the held-out box; this is the only quantity left that separates them, and
   §80 established that nothing else stands in the way of the call.
2. **Wire the fillet into `modelled_area_reference`** (§50) — unchanged from §79 and §80.
3. **Calibrate §73's two thresholds on a proper hold-out protocol** — unchanged, and §81 is
   the third worked example of why.
4. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced.
5. **Re-run the hub-share ladder on a filleted mesh** (§75) — one rung is not a ladder.
6. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
7. **A bend that is a FUNCTION of the genome** (§56); **the REST of §45's audit list**
   (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the element-validity
   check** (§44).

## §82 — 2026-08-25. THE BARRIER HALF IS DECIDED. A PER-GENOME LAYER PROFILE IS ADOPTED AT THE BAND'S LOWER EDGE, THE CLIFF IT NEEDS TURNS OUT TO HAVE A CLOSED FORM — AND THE AXIS IT IS MEASURED ON HAD A THIRD SENTINEL IN IT

§80 retired §68's second reason by inventory and found the first one still standing:
`GENOME_ROBUST_*` buys 15 of 16 by spending the shipped genome's layer-width margin down
to ~0.06.  §81 killed the compromise pair `(-0.70, 0.90)` on a held-out draw — 28 of 32
against the 31 of the pair it was meant to replace — and named the per-genome rule as the
barrier half's answer, leaving one number between the arc and a decision.  This section
takes the decision, and finds two defects on the way to it.

**The rule is adopted.  `wheel_wheel.per_genome_layer_profile` and
`FILLET_LAYER_CLIFF_FACTOR = 0.45` are in the module.**

### THE OPERATING POINT IS THE BAND'S LOWER EDGE, BRACKETED ON BOTH SIDES

The admissible set is two conditions and neither is invented here: clear `MIN_SJ_TARGET`
on the held-out draw, and settle against `study_corner_singularity.SETTLING_RATIO` at the
shipped genome.

```
  factor   held-out clears   settling ratio   settles   cost vs shipped   margin left
   0.95        29 of 32          0.800          NO          +0.865%          0.040
   0.85        31 of 32          0.796          NO          +0.804%          0.121
   0.75        31 of 32          0.684          yes         +0.392%          0.202
   0.65        31 of 32          0.634          yes         +0.298%          0.282
   0.55        31 of 32          0.591          yes         +0.247%          0.363
   0.45        31 of 32          0.466          yes         +0.106%          0.444   <-- ADOPTED
   0.35        30 of 32          0.552          yes         +0.189%          0.524
   0.25        25 of 32          0.519          yes         +0.132%          0.605
   0.15        16 of 32          0.483          yes         +0.116%          0.685
```

0.95 and 0.85 fail the settling condition; 0.35 and below start losing genomes on the
barrier.  What is left is 0.75 / 0.65 / 0.55 / 0.45, all clearing the **same** 31 of 32,
across which both remaining axes improve monotonically as the factor falls.  Two axes
improving across a flat third puts the operating point at the band's lower **edge** — and
0.45 also happens to carry the lowest convergence cost of all nine factors.  §81's
correction was that §78 quoted 0.75 off a sweep that stopped at 0.55 without ever locating
an edge; this sweep runs to 0.15 and brackets the edge from below.

**What it buys, against what §68 declined.**  The rule clears 31 of 32 held-out genomes
where the shipped pair clears 16, and it leaves the shipped genome **0.4435** of
layer-width margin where `GENOME_ROBUST_*` leaves 0.056 — 7.9x — for the same barrier
count.  That is §68's first reason answered on its own terms: a global pair has to be safe
for the tightest genome in the box and pays for that at every other one, and a per-genome
share does not.

### THE BLOCKER WAS NEVER THE MEASUREMENT. THE CLIFF HAS A CLOSED FORM

A per-genome rule needs the genome's cliff *before* it can build, and `cliff_entry` finds
it by bisecting `sector_verdict` thirty times — thirty filleted sector builds, which
cannot sit inside a mesh default.  It does not have to.  The width profile is

```
  H(u) = _hermite(wall, end * wall, entry * (R_arc + wall) * sweep, 0, u)
```

a cubic in `u` whose only `entry`-dependence is one linear term, and `wall`, `R_arc` and
`sweep` all come out of the **tangency** solve, which does not depend on the layer profile
at all.  So the cliff is a root-find over arithmetic on two scalars the build already
computes.  `_fillet_curves` now returns them — **on the layer refusal as well as on the
build**, because the cliff has to be computable at an entry that refuses.

**Measured against the instrument it replaces: 9.1e-10 over the held-out draw**, which is
the 30-halving bisection's own resolution, at 27x the speed with no mesh built. At the
shipped genome, closed form -0.806402516565 against the bisection's -0.806402516551.

### AND THE AXIS HAD A THIRD SENTINEL IN IT

Two separate defects, found by auditing the axis rather than by measuring anything new.
Both are the same shape as the one §78 corrected — one name, two meanings, tallied as one.

**One: `CLIFF_BRACKET` is too narrow, and its sentinel reads as the opposite of the
truth.**  `cliff_entry` returns `None` with *"builds across the whole bracket"* for three
of the thirty-two held-out genomes, and `sweep_cliff_clamped_profile` reads that as *"no
layer-width edge to project onto — the SAFEST case"* and falls back to a global constant.
All three have edges, at **-2.51, -2.30 and -2.12**, just past the bracket's -2.0.  The
finding survives it: at the adopted factor those three clear the barrier on the rule's own
answer exactly as they do on the fallback — 31 of 32 either way, worst J 0.1721 and median
0.3466 on both — so this is a defect in what the sentinel *means*, not in a published
number.  The module's own bracket runs to -8.0 and holds all three.

**Two, and this one is load-bearing: `_sector_fit_span` counts a layer-width refusal as
"no room", so `_sector_fit_limit` reports the LAYER's cliff under the sector fit's name.**
Its docstring says "ANY refusal counts as no room" and names two reasons, both of which
are about the radius.  The layer-width refusal is not — it is about the profile.  Measured
at the shipped genome, `end` = 0.70:

```
  entry    R      builds?   free_span_deg / why
  -0.45   1.60      yes     8.1465
  -0.45   1.65      yes     7.8263
  -1.40   1.60      yes     8.1465      <- bit-identical to the shallow entry
  -1.40   1.65      NO      the layer's width profile reaches zero thickness
```

The reported "hub limit" at entry -1.40 is 1.6285 — a radius at which the free ring span
is **8.15 deg**, not the zero the limit is *defined* as.  The sector-fit limit is not
profile-dependent; its **instrument** is.

The consequences are real and were found as a stale artifact.
`study_corner_singularity_fillet.json` was last regenerated at `0b1cb04`, before §74 put
the clamp in the mesh; regenerating it moved **6 of 34 priced pairs**, flipped `(-0.90,
0.70)` from refused to built, and flipped two band booleans.  Bisecting across
`wheel_wheel.py` versions isolates the cause exactly — pre-§79 and HEAD agree bit for bit,
and HEAD with `clamp=None` reproduces pre-§74 bit for bit — so §74's clamp is the whole of
it, biting at steep entries for the wrong reason.

**So §74's scope note is wrong where it is widest.**  It wrote "the rim has no limit at
all", which is true at the shipped profile and false at a steep one, and concluded that
"every number in §50-§69 stands".  That does not hold for rows measured at steep,
non-shipped profiles.  `tests/test_corner_singularity.py::test_the_band_is_separating_the_CONTACT_PATCH_and_not_the_fillet`
is now an **xfail carrying this reason**, which is what its own docstring asked for — it
says it should fail rather than pass if the finding breaks.  The breaking row is `(-0.90,
1.10)`, which now holds the spread band without the patch band at the highest patch count
in the table, giving `ok={31}` against `bad={29,30,32}`.

### WHY THE ADOPTION DOES NOT DEPEND ON EITHER DEFECT

This matters, because a rule adopted on a broken axis would be the exact error the section
is recording.  **The operating point is shallower than the shipped entry** — `0.45 *
-0.806403 = -0.363` against `FILLET_LAYER_ENTRY_SLOPE = -0.45` — and the
`_sector_fit_span` defect only fires at a *steep* entry, where a layer refusal can happen
inside the radius bracket.  Checked directly rather than argued: across the held-out draw,
**the clamp bites at 0 of 29 genomes** that have a cliff, at the prescribed entry, and at
the shipped genome the hub sits at 0.664 against a limit of 2.973 and the rim at 3.000
against 7.584.  The hub limit is flat at 3.1297 from -0.45 through -1.00 and only collapses
past -1.40, which is well outside anything this rule asks for.

The margin the rule is quoted at is therefore **conservative rather than wrong**: 0.4435
is the distance to the fixed-radius cliff, and the mesh's own refusal boundary — re-clamping
at every entry — is at -2.4223, so the true clearance at the operating point is 2.06.

### WHAT IS UNCHANGED

**The shipped mesh does not move, and neither does the `fillet=True` default.**
`build_wheel`'s default is `fillet=None`, which is bit-identical to the pre-fillet
construction and is what the tree ships; `FILLET_LAYER_ENTRY_SLOPE` and
`FILLET_LAYER_END_OFFSET` are untouched and are still what `fillet=True` takes when nobody
asks for the rule.  What is adopted is the **mechanism and its operating point**, in the
shape §57 -> §74 already used: measure in the study, move the root-find into the module,
keep the study's name bound to the module's value so the two cannot drift.
`CLIFF_PROFILE_FACTOR` is now `WW.FILLET_LAYER_CLIFF_FACTOR`.

**Flipping the `fillet=True` default to the rule is deliberately NOT in this step.**  It
changes every filleted mesh at every non-shipped profile and re-dates every committed
artifact that carries one — and this section has just finished demonstrating that an
un-audited default change is how `study_corner_singularity_fillet.json` went stale for two
days without anyone noticing.  It is priced as successor 1.

#### The successors, ranked — REVISED 2026-08-25 AFTER §82

1. **Fix `_sector_fit_span` to distinguish a layer refusal from a sector-fit refusal, and
   re-derive the artifacts that carry a clamp.**  This is now the top item and it was not
   on the list before today.  The fix is inert at the shipped profile — no layer refusal
   fires anywhere in the radius bracket there — so §57's and §74's published margins
   should be unchanged, and that is an acceptance test rather than an assumption.  It
   reopens the xfail above by itself.
2. **Flip the `fillet=True` layer profile to `per_genome_layer_profile`**, with the
   artifact audit item 1 makes possible.  The mechanism is adopted and tested; this is the
   default change.
3. **Wire the fillet into `modelled_area_reference`** (§50) — unchanged from §79-§81.
4. **Widen `CLIFF_BRACKET` and retire the `CLIFF_NO_EDGE` fallback branch** — the rule now
   answers for all thirty-two, so the fallback is reachable only through a bracket that is
   known to be too narrow.  Small, and it removes a constant from the rule.
5. **Calibrate §73's two thresholds on a proper hold-out protocol** — unchanged; §81 and
   §82 are now the third and fourth worked examples of why.
6. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced.
7. **Re-run the hub-share ladder on a filleted mesh** (§75) — one rung is not a ladder.
8. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
9. **A bend that is a FUNCTION of the genome** (§56); **the REST of §45's audit list**
   (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the element-validity
   check** (§44).

## §83 — 2026-08-26. THE SECTOR-FIT LIMIT IS NOT PROFILE-DEPENDENT AND NEVER WAS: `_sector_fit_span` WAS READING A LAYER REFUSAL AS "NO ROOM". SEPARATING THEM PUTS ALL 34 PRICED CORNER PAIRS BACK TO THEIR PRE-§74 VALUES, FIELD FOR FIELD

§82 recorded this defect and did not fix it, because fixing it moves the clamp at every
steep profile and re-dates every artifact carrying one.  This is that step.

### THE FIX IS ONE HOIST, AND IT IS AVAILABLE BECAUSE THE SPAN NEVER NEEDED THE PROFILE

`free_span_deg = (th_end - th_B) * dirn` is a function of `Q` — the ring corner `uncap`
chooses — and of `B`, the tangency point.  **Neither `entry` nor `end` reaches it.**  But
it was computed ninety lines below two refusals that do depend on the profile — the
layer-width one and the ring-crossing one — so at exactly the radii where those fire, the
span was never reached and `_sector_fit_span` saw only `built: False`.

So the change is:

- `th_q` / `th_B` / `dirn` / `th_end` / `free_span_deg` move up to just after the tangency
  solve.  `th_N` stays where it was: it needs `N`, the offset root-find, which *does* move
  with the profile.  Only the half that does not is hoisted.
- Every refusal below now carries `free_span_deg` alongside §82's `layer_wall`/`layer_k` —
  the same pattern, for the same reason.
- `_sector_fit_span` returns the span whenever it is present, and `-1.0` only when the
  **tangency** failed, which is the one refusal that genuinely means *this radius* is too
  big.  The other three are statements about the layer profile, and their remedy is a
  shallower entry, not a smaller radius.

`study_fillet_block.sector_fit_limit` delegates its root-find to the module (PART 21), so
it follows automatically — which is exactly why that delegation was done.

### THE LIMIT STOPS COLLAPSING, WHICH IS THE WHOLE CLAIM

```
                    BEFORE (§82 measured this)          AFTER
   entry        hub_limit      rim_limit          hub_limit      rim_limit
   -0.450        3.129700       6.179707           3.129700         None
   -0.800        3.129700       3.033997           3.129700         None
   -1.400        1.628505       1.147810           3.129700         None
   -1.800        0.759410       0.571289           3.129700         None
   -2.200        0.250591       0.201077           3.129700         None
   -2.600        0.050000       0.050000           3.129700         None
```

The limit is now constant in the profile, which is what §57 and §74 both said it was.  The
left-hand column was never a sector-fit limit: at entry -1.40 the reported 1.6285 sits at
a radius with **8.15 deg of free ring left**, against the ZERO the limit is defined as.

### NOTHING PUBLISHED MOVED, AND THAT WAS PRE-REGISTERED RATHER THAN FOUND AFTERWARDS

Before touching anything, the claim was stated and measured: across the full
`SECTOR_FIT_BRACKET` at the shipped profile, no layer refusal fires — for the shipped
genome, or for any of the 32 held-out ones.  **0 of 64 junction-pairs.**  So the fix
cannot move a shipped-profile number, and it does not:

```
  §74's hub limit      3.1296998810584657  ->  3.129699881054943   (3.5e-12, the 40-vs-80
                                                bisection difference §57 already records)
  §74's rim limit      None                ->  None
  §74  clamp on, in sample                 16 of 16 build      (was 10 unclamped)
  §78  clamp on, held out                  32 of 32 build      (was 26 unclamped)
  §78  shipped pair, held out              16 of 32 clear
  §81  genome_robust / margin_robust       31 / 28 of 32 clear
  §82  shipped cliff                       -0.806402517
  §82  f = 0.45, held out                  32 built, 31 clear, margin 0.4435
```

All 23 of `study_fillet_block`'s self-checks pass.

### AND THE SIX MOVED CORNER ROWS GO BACK EXACTLY

`study_corner_singularity_fillet.json` regenerated against the fix, compared **field by
field** against the pre-§74 committed artifact: **34 shared rows, zero differences, zero
removed, nine added** — the nine being §82's per-genome factors.  The additivity was
checked by joining on `(entry, end)` and diffing every field, not by comparing list
lengths, which is how §82 got this wrong the first time.

So every one of the six rows that moved when the clamp reached this artifact was moved by
the defect and by nothing else.  **The clamp, correctly implemented, is inert on all 34
priced corner pairs** — the same thing it is at the shipped genome, and the reason §74
thought it was inert everywhere.

`test_the_band_is_separating_the_CONTACT_PATCH_and_not_the_fillet` passes again on its own
merits: `ok={31}` against `bad={29,30}`.  §82's strict xfail is **removed**, and the
episode moved into the test's docstring instead — it now records that this test has a
false-positive mode, and that the first thing to check when it reddens is whether the
artifact is stale against `wheel_wheel` rather than whether the band moved.

### WHAT THIS COSTS, STATED RATHER THAN BURIED

At steep profiles some genomes stop building, because the clamp no longer rescues them by
shrinking the fillet for the wrong reason.  At the shipped genome, `(-0.90, 0.70)` returns
to **refused** — and its reason is now the honest one, *"the layer's width profile reaches
zero thickness at the rim"*, rather than a silently smaller fillet.  That is a mesh the
construction genuinely cannot build at that profile, and §82's rule is how a genome that
wants a steep entry gets one it can actually hold.

**Nothing about §82's adoption depends on this.** Its operating point is shallower than
the shipped entry, and the clamp is inert there before and after.

#### The successors, ranked — REVISED 2026-08-26 AFTER §83

1. **Widen `CLIFF_BRACKET` and retire the `CLIFF_NO_EDGE` fallback** — unchanged from
   §82's item 4, now the cheapest thing on the list and a prerequisite for reading item 2's
   diff cleanly.  `cliff_entry` should delegate to `ww._layer_cliff_from_scalars`, which
   also takes the 30-build bisection out of `make filletblock`.
2. **Flip the `fillet=True` layer profile to `per_genome_layer_profile`** — the default
   change, with the artifact audit §83 has now made readable.
3. **Wire the fillet into `modelled_area_reference`** (§50) — unchanged from §79-§82.
4. **Make the layer cliff differentiable** — new.  §82's closed form makes the implicit
   function theorem applicable to `min_u H = 1e-6`; without it, item 2 has to refuse a
   gradient exactly as §79's clamp does.
5. **Calibrate §73's two thresholds on a proper hold-out protocol** — unchanged.
6. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced.
7. **Re-run the hub-share ladder on a filleted mesh** (§75) — one rung is not a ladder.
8. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
9. **A bend that is a FUNCTION of the genome** (§56); **the REST of §45's audit list**
   (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the element-validity
   check** (§44).

## §84 — 2026-08-26. THE OTHER SENTINEL: "BUILDS ACROSS THE WHOLE BRACKET" WAS DESCRIBING THE BRACKET. ALL 32 HELD-OUT GENOMES HAVE A CLIFF, THE FALLBACK BRANCH IS GONE, AND THE THIRTY-BUILD BISECTION GOES WITH IT

§82 found this and left it, because the finding survived it and widening a bracket
re-dates artifacts on its own.  §83 cleared the way; this is the cheap half.

### THE BRACKET IS THE MODULE'S NOW, AND THE FALLBACK IT FED IS RETIRED

`CLIFF_BRACKET` stopped at -2.0, so three of the thirty-two held-out genomes came back as
*"builds across the whole bracket"*.  §78 read that as **the safest case** — a genome with
no layer-width edge to project onto — and had those genomes fall back to
`GENOME_ROBUST_ENTRY`, tallying them as genomes the rule does not harm.  All three have
edges, at **-2.51, -2.30 and -2.12**.  The sentinel was reporting the bracket's width.

`CLIFF_BRACKET` and `CLIFF_BISECTIONS` are now bound to `wheel_wheel.LAYER_CLIFF_*` rather
than re-stated, so the two cannot drift — which is the property that made this possible,
and the same fix PART 21 applied to the clamp factor.  With the module's bracket **32 of 32
have a cliff**, the rule answers for all of them, and the `CLIFF_NO_EDGE` branch is dead
code and deleted.  `n_without_cliff` is **kept in the row schema, pinned at zero**: a
counter that silently stops existing is how the next regression hides.

### AND `cliff_entry` DELEGATES TO THE CLOSED FORM, SO THE THIRTY BUILDS ARE GONE

§82 showed the layer cliff is a root-find over arithmetic on two scalars the tangency
solve already produces.  The study kept its own thirty-build bisection anyway, because the
adoption commit was not the place to change the instrument every number in it was measured
with.  It is now `wheel_wheel.layer_cliff_entry`, and the four hand bisections PART 20
published still land:

```
   end      published      delegated      |difference|
  0.85     -0.845458      -0.845458          1.65e-07
  1.00     -0.881143      -0.881143          3.70e-07
  1.10     -0.903400      -0.903437          3.70e-05
  1.60     -1.001967      -1.001967          3.25e-07
```

all inside the 1e-4 the record was published to.  **`make filletblock` goes 508 s -> 401 s.**

**THE REASON IS STILL CHECKED, AT TWO BUILDS INSTEAD OF THIRTY.**  The closed form cannot
be wrong about the LAYER, but it can be right about the layer and wrong about the BUILD if
a sector-fit or tangency refusal binds at a shallower entry — which is PART 6's error, and
dropping the check to buy speed would re-introduce it.  So `cliff_entry` still takes the
two verdicts either side of the closed-form cliff: just inside it the sector must build,
just outside it must refuse for exactly this reason, or it returns `None` saying so.

**A bonus correction fell out of that.**  The old bisection reported `why` from the
bracket's LOW end, where both junctions have already refused and the hub is simply checked
first — so every published cliff said *"no filleted blocking exists at the hub"*.  The
binding junction at the shipped genome is the **rim** (-0.806 against the hub's -1.862),
and the two verdicts now name it correctly.  No number moves; the label was wrong.

### WHAT MOVED AND WHAT DID NOT

The three ex-fallback genomes now take `factor * their own cliff` instead of a constant, so
the tails of the factor sweep move — and the admissible band does not:

```
                  held out, clears MIN_SJ_TARGET of 32
   factor     before (fallback)     after (the rule's own answer)
    0.95            29                      29
    0.85            31                      31
    0.75            31                      31
    0.65            31                      31
    0.55            31                      31
    0.45            31                      31      <-- ADOPTED
    0.35            30                      30
    0.25            25                      24
    0.15            16                      13
```

**At the adopted factor nothing moves at all**: 31 of 32, worst J 0.1721, median 0.3466,
shipped margin 0.4435, shipped cliff -0.806402517 — every one identical to §82.  The band
edge is still bracketed on both sides, 0.85-0.45 flat at 31 and 0.35 dropping to 30.  This
was measured before the branch was deleted, not discovered afterwards.

`study_corner_singularity_fillet.json` regenerated: **43 shared rows, zero field
differences**, the only change being the corrected `shipped_cliff.why`.  All 23 of
`study_fillet_block`'s self-checks pass.

#### The successors, ranked — REVISED 2026-08-26 AFTER §84

1. **Flip the `fillet=True` layer profile to `per_genome_layer_profile`** — the default
   change.  The mechanism is adopted, the axis it is measured on is fixed, and the rule now
   answers for every genome in both draws with no fallback in it.
2. **Wire the fillet into `modelled_area_reference`** (§50) — unchanged from §79-§83.
3. **Make the layer cliff differentiable** — §82's closed form makes the implicit function
   theorem applicable to `min_u H = 1e-6`; without it, item 1 has to refuse a gradient
   exactly as §79's clamp does.
4. **Calibrate §73's two thresholds on a proper hold-out protocol** — unchanged.
5. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced.
6. **Re-run the hub-share ladder on a filleted mesh** (§75) — one rung is not a ladder.
7. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
8. **A bend that is a FUNCTION of the genome** (§56); **the REST of §45's audit list**
   (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the element-validity
   check** (§44).

## §85 — 2026-08-26. THE DEFAULT IS FLIPPED. `fillet=True` NOW TAKES THE PER-GENOME LAYER PROFILE, THE PROFILE IS RESOLVED WHERE THE GENES ARE SO THE JAX CACHE KEY STAYS CORRECT, AND §79's GRADIENT IS REFUSED RATHER THAN QUIETLY WRONG

§82 adopted the mechanism and left the default alone; §83 and §84 fixed the two defects in
the axis it was measured on.  This makes it what `fillet=True` takes.

### WHAT CHANGED, AND WHERE IT DELIBERATELY DID NOT

`layer_profile=None` on `fillet=True` with the eleven-block sector blocking is now
`per_genome_layer_profile`.  `FILLET_LAYER_SHIPPED` is the pair the module took before, kept
reachable **under a name**, because `study_fillet_block` re-derives against it and a default
nobody can ask for by name is a default nobody can measure.

Three paths keep the old constants, and each for a reason that already existed:

- **An explicit `(R_hub, R_rim)` pair does not get the rule**, exactly as `SECTOR_FIT_CLAMP`
  does not touch one: a caller passing radii is measuring *those* radii.  `study_fillet_fold`
  sweeps `fillet=(R, 0.0)` to a zero rim, where a per-genome cliff cannot be computed at all.
- **`fillet_blocking="spoke"` does not get it.**  §47's construction has no layer to profile,
  and resolving one would build an eleven-block sector purely to answer a question that
  construction does not pose.
- **The unfilleted path never reaches a layer.**  `build_wheel`'s default is still
  `fillet=None`, still bit-identical, and still what the tree ships.

Both of the first two were found by the suite rather than by reading: six `test_fillet_fold`
failures, from a first cut that resolved the rule for any `fillet` at all.

### THE CACHE KEY IS WHY THE PROFILE IS RESOLVED EAGERLY [HALF-SUPERSEDED BY §88]

**The eager resolution stands; the KEY does not.** The resolved pair is a function of the
genome, so putting it in the key made the key genome-dependent and the 2.78 s jaxpr would
re-trace on every finite difference — `coord_fn`'s own named failure, arriving by a second
route. Nothing caught it because `mesh_coords` refused these meshes and the key was never
reached. §88 resolves the entry inside the trace, the record holds `None` again, and the key
reads it raw. What follows is the reason the resolution is eager, and that is unchanged.

`coord_fn`'s key is built from `repr(_layer_profile(rec["layer_profile"]))`.  Left as `None`
in the record and resolved per genome further down, **two genomes with different profiles
would share one key** and the second would be handed the first's traced geometry — the exact
failure the `repr(uncap)` comment three lines below that key warns about.  So
`_resolve_layer_profile` settles it in `build_wheel`, before the record is written, and
`_layer_profile` stays a pure pass-through with no access to the genes *by design*.

Measured, two genomes, same call: `(-0.362881, 0.7)` and `(-0.421871, 0.7)`.  Under the old
arrangement both would have keyed as `(-0.45, 1.6)`.

### AND THE GRADIENT IS REFUSED, WHICH IS §79's OWN PRECEDENT [SUPERSEDED BY §88]

**It is not refused any more, and the tripwire this paragraph ends with is what fired.** The
cliff is a closed form — `max_u (Z - a(u))/b(u)` over the sampled width profile, the same root
to 2 ulp — so it is differentiated rather than held constant. The gradient refused here is
wrong by 3.0% to 24.4% at the shipped genome and every one of the 36 unclamped genomes
of the 48-genome box. G11e counts two
refusals now; G11f measures this case. Read §88 before quoting anything below.

The rule's entry is `FILLET_LAYER_CLIFF_FACTOR * cliff(genes)`.  It **does** depend on the
genes and the frozen path holds it constant, so a derivative taken through it is wrong in
the way nothing downstream could see — which is word for word why §79 refuses a mesh whose
radius the clamp moved.  `mesh_coords` now raises the analogous `NotImplementedError`, and
`study_gradient`'s G11e census carries it as a **third** refused case, measured rather than
asserted, so the day it stops refusing is the day someone made the cliff differentiable.

**§79's own numbers are unmoved**, because `study_gradient` now names
`FILLET_LAYER_SHIPPED` rather than inheriting a default: silently re-measuring §79's gate on
§85's geometry would leave that file reporting one section's threshold against another
section's mesh.  G11 passes.

### THE ARTIFACT AUDIT

```
  artifact                              moved?   why
  study_corner_singularity_fillet.json  YES      `--fillet genome` is `fillet=True`
  study_reds_hub_share.json             YES      `--fillet` is `fillet=True`
  study_gradient.json                   keys     pinned to the shipped pair; gains the
                                                 third refusal and its pair
  study_fillet_block.json               no       every sweep passes an explicit profile
  study_fillet_fold.json                no       explicit pairs, never the rule
  everything else                       no       unfilleted
```

**The corner artifact moved in exactly four numbers**, and they are the four that can:

```
   corner        before      after      moves with the layer profile?
   hub:A        183.012     183.314     yes — the fillet's tangency
   hub:B        183.972     183.670     yes
   rim:A        184.544     185.221     yes
   rim:B        183.439     182.762     yes
   hub/rim:P_t  360.000     360.000     no  — bit-identical
   hub:P_c      268.209     268.209     no
   rim:P_c      270.853     270.853     no
   hub/rim:arc  188.100 / 188.305       no
   hub/rim:N    360.000     360.000     no
```

Element and node counts are unchanged (37632 at the finest rung) — the profile moves where
the boundary layer's stations sit, not how many there are — and the 43-row `--profiles`
sweep is untouched, because it passes explicit pairs.

**§75's finding survives the flip.**  The hub-share ladder re-run on the rule still goes
*exactly* flat above the cap — four rows at 0.888507 against the old four at 0.889754 — with
11 distinct values of 14 as before, every row shifted about 0.1% by the gentler profile.
Worth stating because the opposite was plausible: `R_hub` is gene 12 and enters the cliff, so
under the rule it moves the layer profile as well as the radius.  It does not break the
plateau, and the reason is that above the cap the clamp pins the built radius, so the cliff —
computed at the radii the mesh is built at — stops moving too.

### ONE TEST NEEDED A NEW LINE, AND IT DID NOT GET A LOOSENED ONE

`test_the_interpolated_drop_is_the_same_number_when_a_node_IS_at_the_bottom` asserts §65's
correction is not inert on a filleted mesh, at 3x the unfilleted offset and gap.  The rule's
profile is gentler than the shipped pair — `(-0.3629, 0.70)` against `(-0.45, 1.60)`, a
shallower entry and a much shorter layer — so it perturbs the rim less: offset -0.1029 deg
against -0.1635, gap 3.76e-03 against 5.81e-03.  The claim holds at **2.3x and 2.1x**.

The 3x line is **left exactly where §65 put it**, now naming `FILLET_LAYER_SHIPPED`, and the
new default is asserted on its own line at its own level.  A threshold moved in the same
change that reddened it cannot be told apart from one fitted to the run that breached it.

### AND THREE MORE PLACES WHERE A TEST PINNED A SYMPTOM RATHER THAN THE FINDING

All three went red, none of them because the finding failed.  They are worth recording
together, because the same mistake made all three: the property was real and what got
written down was one of its shapes on one mesh.

**§65's node-reading artefact changed shape and got WORSE.**  The test asserted
`node["monotone"] is False` — the artefact's signature on the shipped profile, an
oscillation with increments +0.001359 then -0.001236 and ratio -0.909.  Under the rule the
ladder is **monotone and diverging**: -0.000651 then -0.001175, ratio 1.804, each rung
moving it further than the last.  Asserting non-monotonicity would have gone GREEN here
while the reading it guards got worse.  What §65 actually found is that the nearest-node
reading does not settle, and `node["settling"] is False` says so on both meshes; that is
what is pinned now, with `interp["settling"] is True` beside it.

**§50's `N` / `P_t` separation is a function of the layer's `end`.**  `N` is where the
layer's inner edge crosses the ring circle, so a shorter layer brings it in: the hub's goes
0.4719 mm -> 0.0997 under the rule's `end` of 0.70 against the shipped 1.60.  The 0.4 floor
stays on the profile it was derived from and the current default is checked at 0.05 — two
orders above the seam tolerance the mesh closes to and an order under the measurement, so
it still fails on `N` and `P_t` becoming one point rather than on the profile moving them.

**§29's convergence contrast narrowed, exactly as §82 priced it.**  Filleted tail 0.031% ->
0.131%, contrast 11.9x -> 2.8x.  Both band claims are untouched and both still hold — the
filleted ladder inside +-0.3%, the unfilleted outside — and the contrast factor is now
anchored to §82's published increment ratio for factor 0.45 (0.466, measured; the ladder
reads 0.4661) rather than to this run, so the test and that table cannot drift apart
without one of them saying so.

#### The successors, ranked — REVISED 2026-08-26 AFTER §85

1. **Wire the fillet into `modelled_area_reference`** (§50).  `area_report` still refuses a
   filleted mesh outright, and it is now the last thing between the fillet and being usable
   for something other than measurement.
2. **Make the layer cliff differentiable.**  §82's closed form makes the implicit function
   theorem applicable to `min_u H = 1e-6`, and §85 has just made the refusal it removes a
   thing the default path hits rather than a corner case.  It reopens G11e's third row.
3. **Calibrate §73's two thresholds on a proper hold-out protocol** — unchanged, and now the
   oldest thing on the list.
4. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced.
5. **Re-run the hub-share ladder on a filleted mesh** (§75) — DONE as a side effect of §85's
   audit, and the finding held; what remains is the ladder at more than one rung.
6. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
7. **A bend that is a FUNCTION of the genome** (§56); **the REST of §45's audit list**
   (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the element-validity
   check** (§44).

---

## §86 — 2026-08-27. THE FILLET IS WIRED INTO `modelled_area_reference`. THE ADDED REGION IS ONE CURVILINEAR TRIANGLE PER JUNCTION, THE TWO PATHS AGREE TO 4.7e-7, AND THE HALF THAT STAYS WITHHELD IS THE STEP ONE

§85's ranking item 1, and the oldest thing on that list — named by §50 on 2026-08-23,
carried through eleven rankings, promoted to item 2 by §79 and to item 1 by §85.
`area_report` withheld its reference for every filleted mesh; it does not any more.

### PART 11 SAID "NOT A CLOSED FORM", AND THE HALF OF THAT WHICH WAS TRUE IS THE HALF ABOUT THE WEDGE FORMULA

The sentence that had stood since §50 read: *"Making the reference fillet-aware is real
work and is not a closed form — the fillet's legs are a spline and a circle, not two
straight lines, so the inscribed-wedge formula does not apply."*

Both clauses are correct and the conclusion drawn from them was not. The inscribed-wedge
formula does not apply, and nothing needed it to. **The region the fillet adds is one
curvilinear triangle per junction**, and it has three sides:

```
  A -> F   along the STRADDLING flank    A the tangency point, F the ring crossing
  F -> B   along the ring circle         B the arc's tangency on that circle
  B -> A   along the fillet arc
```

and Green's theorem integrates exactly that: polyline shoelace down the flank, one
`r^2 dtheta` for the ring circle, one `R^2 dphi + C x (A - B)` for the arc. Two exact arc
terms of different centres, which is the only reason `_clip_polygon_to_disk` could not be
handed the loop — it knows one circle, about the origin.

**Everything else the eleven-block sector builds is a CUT BETWEEN BLOCKS**, not a
boundary: `cut_N`, `cut_B`, the radial dive to the ring's far radius, the far flank the
junction block still ends on. Each lies strictly inside the region whether the fillet is
there or not. That is the whole finding, and it is why the answer is three lines of
arithmetic rather than a new construction.

### VERIFIED AGAINST THE MESH, NOT ASSERTED — AND THE DECISIVE RUN IS THE SECOND ONE

At the shipped genome, `error_vs_modelled` down the config ladder:

```
  config   n_thick   filleted     unfilleted
  smoke      2       +0.441218%   -0.171071%
  coarse     4       +0.130919%   -0.022841%
  medium     6       +0.060229%   -0.007948%
  fine       8       +0.036161%   -0.002478%
```

Converging, and an order slower than the unfilleted mesh's — which on its own does not
distinguish "the right region, coarsely resolved" from "a slightly wrong region". So hold
`fine` and sweep `n_thick` ALONE, which is the direction that resolves the two arcs:

```
  n_thick    8        16        32        64
  error   +0.036161  +0.006791  -0.000566  -0.002405
```

**-0.0024% against the unfilleted `fine` mesh's -0.0025%.** The residual does not
converge to a floor; it converges to the residual the unfilleted mesh already has, which
is what a region that is exactly right and merely under-resolved does.

### THE INDEPENDENCE IS THE POINT AND IT SURVIVED

`modelled_area_reference` is a cross-check only while it is a DIFFERENT computation. The
tangency is therefore re-solved here on `thicken_3taper_curve`'s own samples — the
exporter's finite-difference offset normals — and not read off the mesh's analytic
hodograph. `_uncap_reference_poly`'s precedent is followed exactly: the DECISION rule is
shared (which flank straddles, read off the radii; the offset direction is the
centreline's normal, copied from `_fillet_centre` because a not-quite-perpendicular circle
on one path and a perpendicular one on the other would measure two different fillets), the
GEOMETRY is not.

Measured, twelve spokes' fillets: **139.16025 mm2 on the reference's polygon against
139.16031 mm2 through the mesh's own sampler and `_fillet_tangency`** — 6.5e-5 mm2, a
relative 4.7e-7. Per spoke the hub wedge is 0.50551 and the rim wedge 11.09118 mm2; the
term is **8.643% of the region**, against the mesh's measured +8.7625% (coarse) /
+8.6965% (medium) from FILLET_PLAN PART 11.

### AND ACROSS THE GENOME BOX, WHICH IS WHERE §50-ERA FILLET WORK USUALLY BREAKS

The 48 genomes `study_fillet_block.json` carries — 16 in-sample, §78's 32 held out, all
four flank orientations — every one of them at `fillet=True`:

```
  config   mesh built   reference available   clamped   worst |error_vs_modelled|
  coarse     48/48            48/48             12            0.1571%
  medium     48/48            48/48             12            0.0409%
```

Zero refusals, and every genome's residual falls with refinement. The reference has the
same reach as the mesh it describes.

### `fillet=True` IS REFUSED, AND THAT IS THE CLAMP

`sector_blocks` reads `fillet=True` as *"this genome's radii, MOVED BY `SECTOR_FIT_CLAMP`
if they have no room"*. Resolving that needs a config, an `uncap` and a layer profile,
none of which a pure area reference has. Accepting the flag would make one spelling name
two regions — §84's sentinel wearing different clothes — so `modelled_area_reference`
takes `(R_hub, R_rim)` or `None` and raises on `True`; `area_report` passes
`mesh.fillet_radii_mm`, which is what was BUILT.

**Priced rather than argued.** On the §57 genome whose hub radius the clamp moves
(3.28618 requested, 2.97321 built), the reference at the requested radius is **8.008 mm2**
above the one at the built radius — thirty times the residual the comparison exists to
see, and it would have read as a mesh defect. A test pins it.

### WHAT STAYS WITHHELD, AND IT IS NOT CAUTION

**`reference_shipped_step_mm2` and `error_vs_shipped_step` are withheld for a filleted
mesh**, with a named reason, exactly as the whole report used to be. Both numbers behind
them were measured against the UNFILLETED cross-section — the 2644.3509 mm2 profile and
`EMBED_ALLOWANCE_PER_SPOKE_MM2 = 3.03` — so the STEP reference describes an unfilleted
region and there is no like-for-like comparison to report. Anchoring it on the mesh's own
fillet instead would be inventing the number, since the exporter's fillet is OCC's
edge-fillet on the embedded solid and this one is a tangent arc on the un-embedded band.

`gusset_modelled_mm2` is **exactly unchanged**, to the bit, and by construction rather than
by luck: the fillet rounds the STRADDLING flank and `uncap` continues the FAR one, so the
term rides on both of `area_report`'s calls and cancels in their difference.

**`fillet_blocking="spoke"` still withholds everything.** §47's retired construction rounds
the flank a different way and leaves no `_applied` record to read a built radius out of;
`make fillet` still measures it, so it stays reachable and it must not be compared against
the sector blocking's region. The branch that used to withhold for every filleted mesh
survives, narrowed to exactly that case.

### A STALE PARAGRAPH THIS NOTICED  [CORRECTED IN PLACE BY §87 — SEE BELOW]

`wheel_wheel.py`'s module docstring says the exporter's fillets are worth **24.28 mm2,
0.92%** of the cross-section ("the filleted solid's cross-section is 2668.63, 59777.4 mm3
/ 22.4 mm"). Every current measurement disagrees by an order of magnitude:

```
  source                                     fillets, as cross-section
  module docstring                            24.28 mm2     0.92%
  §24, on genome e126cc3, by mass             (3553.19 mm3)  8.77%
  export/defect5_step100 manifest             160.14 mm2    9.67%   (3587.19 / 22.4 mm)
  this section, the MESH's own fillet         139.16 mm2    8.64%
```

The docstring's pair is internally consistent — 2644.3509 x 22.4 = 59233.5, and
59777.4 - 59233.5 = 543.9 mm3 = 24.28 x 22.4 — and describes a genome and a corner count
that no longer exist. **It is a stale paragraph, not a contradiction between kernels.**

**AND §86 SHOULD NOT HAVE CALLED IT A FINDING.** §14 found it, said so in as many words
— *"6.18% of the solid, not the 0.92% the old docstring claimed"* — and corrected it in
`tests/test_wheel_fea.py` and in this file. What was true is only that the correction
never reached `wheel_wheel.py`. §86 also said re-measuring "needs a fresh CAD export off
the shipped genome"; `export/wheel_step_manifest.json` has been committed since
2026-08-15 and is that export. Both errors are the same one — a claim made without
grepping the tree for what it already knows — and §87 is the correction.

### WHAT IS UNCHANGED

**Nothing promoted, `best_solution.json` untouched and still 2026-08-14, no threshold
moved, no artifact regenerated.** The unfilleted path is **bit-identical**: coordinates
hashed at `smoke`, `coarse` and `medium`, and `area_report`'s full JSON compared key for
key AND in key ORDER against the previous commit — identical, so `studies/study_wheel_mesh.json`
does not move. The new breakdown keys appear only when `fillet=` is passed, which is why.

`tests/test_filleted_mesh.py`'s `test_the_area_reference_is_WITHHELD_for_a_filleted_mesh`
is gone and four tests stand where it did: the reference DESCRIBES the region and its
residual shrinks under refinement; it takes the radii that were BUILT and `fillet=True`
raises; the fillet term is ADDED and the band's own figures are untouched; and the spoke
blocking still withholds.

#### The successors, ranked — REVISED 2026-08-27 AFTER §86

1. **Make the layer cliff differentiable.** §82's closed form makes the implicit function
   theorem applicable to `min_u H = 1e-6`, and §85 made the refusal it removes a thing the
   default path hits rather than a corner case. It reopens G11e's third row. Up from 2 by
   item 1 being done.
2. **Price the mesh's fillet against the EXPORTER's**, and re-measure the module
   docstring's stale 0.92% while the CAD env is open — NEW, and §86 is what makes it worth
   doing: the mesh now has an exact fillet area per genome, so the comparison is one export
   away instead of being unbuildable. It is also what would let the STEP half stop being
   withheld.
3. **Calibrate §73's two thresholds on a proper hold-out protocol** — unchanged, and still
   the oldest thing on the list now that §50's item is closed.
4. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced.
5. **Per-REGION agreement on a filleted mesh** — NEW and small, and §86 names its obstacle:
   `FILLETED_BLOCK_REGION` tags both fillet blocks `spoke`, but `*_fillet_b` straddles the
   ring circle, so the mesh's `hub`/`rim` regions are no longer the full ring annuli.
   `test_region_areas_are_individually_right` is unfilleted-only for that reason.
6. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
7. **A bend that is a FUNCTION of the genome** (§56); **the REST of §45's audit list**
   (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the element-validity
   check** (§44).

---

## §87 — 2026-08-27. THE EXPORTER'S FILLET, RE-MEASURED: 137.4451 mm2 AGAINST THE MESH'S 139.1602, 1.25% APART ON TWO KERNELS — AND THE STALE DOCSTRING WAS §14's FINDING, NOT §86's, WITH THE EXPORT IT SAID IT NEEDED ALREADY COMMITTED

§86's ranking item 2, and it begins by retracting two things §86 said about it.

### THE TWO CORRECTIONS, FIRST

**§86 called the 0.92% "an inconsistency this found". §14 found it**, on 2026-08-15,
and wrote it down twice — in
`test_total_mass_matches_the_step_manifest_within_the_embed_difference` (*"It is 2372.53
mm^3 — 6.18% of the solid, not the 0.92% the old docstring claimed"*) and in this file's
§14 item 2. What was actually true is narrower and duller: **the correction never reached
`wheel_wheel.py`'s module docstring**, which went on claiming 0.92% for three arcs while
two other files said otherwise.

**And §86 said re-measuring "needs a fresh CAD export off the shipped genome".** It does
not. `export/wheel_step_manifest.json` has been committed since 2026-08-15, describes
`best_solution.json` (`09e8188`, 2026-08-14), and `test_golden.py::test_genome_hash_matches_manifest`
guarantees it still does. I nearly ran a 56 s export to produce a file already on disk —
an `ls | head -20` truncated before `wheel_*` and I read the truncation as the directory.

Both errors are one error: **a claim made without grepping the tree for what it already
knows.** No new measurement was needed for either half of this section.

### WHAT THE SHIPPED MANIFEST SAYS

```
  solid 39224.5 mm3    nofillet 36145.8    fillets 3078.77    mass 48.64 g
```

At `SPOKE_WIDTH_MM = 22.4`, the same conversion `test_wheel_fea.py` already uses on the
gusset:

```
  unfilleted profile      1613.6518 mm2
  filleted profile        1751.0938 mm2
  fillets                  137.4451 mm2     7.85% of the solid, 8.52% of the unfilleted
```

**7.85%, not 0.92%.** And the percentage has read 6.18 (§14's genome), 8.77 (§24's
`e126cc3`) and 7.85 (this one), which is the point: it moves with the genome, so the
docstring now points at the manifest instead of transcribing a number.

### THE COMPARISON §86 RANKED, AND IT IS THE STRONGEST EVIDENCE THE FILLET REGION HAS

```
  the exporter   OCC edge fillet on the EMBEDDED solid, CadQuery      137.4451 mm2
  the mesh       tangent arc on the un-embedded band, §86             139.1602 mm2
                                                       ratio 1.012479, +1.7152 mm2
```

**1.25% apart, on two kernels that share no code.** §86 verified the wedge against the
MESH — refine `n_thick` and the residual falls to the unfilleted mesh's own — which proves
the reference and the mesh agree about a region they were both derived from. It could not
answer whether that region is the PART's. This does: the difference is the size of
`_embed` moving the corner OCC rounds, which is the difference the two constructions are
known to have.

`tests/test_filleted_mesh.py::test_the_fillet_reference_agrees_with_the_STEP_MANIFEST`
pins it at a 5% band, checks both built radii against the genes first, and asserts the
term is first-order on both sides. The band is deliberately loose: tightening it to fail
on the `_embed` difference would pin that difference instead of the agreement, and what a
5% band catches is a fillet off by a FACTOR or by a count of corners — which is exactly
what 0.92% was.

### AND THE TABLE'S FIRST ROW NAMES THE WRONG ANCHOR

The module docstring's mesh-vs-solid table, measured 2026-08-18:

```
        area vs unfilleted cross-section     -2.2205%  capped   ->  -2.0490%  default
        mass vs the FULL solid               -8.2241%  capped   ->  -8.0632%  default
        mass vs the NOFILLET solid           -0.4039%  capped   ->  -0.2292%  default
```

**All six values reproduce exactly at today's commit.** But row 1 is `error_vs_shipped_step`,
which is measured against the DERIVED anchor `reference_capped_mm2 + 12 x
EMBED_ALLOWANCE_PER_SPOKE_MM2` — not against the STEP's own unfilleted cross-section, which
is what its label says. Against the STEP's actual profile the answer is **row 3**: identical
to the mass row, because mass and area are the same ratio for a uniform extrusion.

**So the mesh is -0.2292% from the shipped solid's unfilleted profile, not -2.05%**, and
the docstring's sentence *"the remaining ~2.05% is a real modelling difference and it is
deliberate"* was wrong by an order of magnitude. The real unmodelled remainder is 3.5701
mm2, **0.2975 mm2 per spoke**. The label is corrected in place and the values are left
alone.

### THE ~1.8% BETWEEN THE TWO ROWS IS §14's OPEN ITEM 6, RE-MEASURED AND NOT CLOSED

The identical computation that produced 3.03 — `(STEP cross-section - this region's
capped reference) / 12` — on the shipped genome:

```
  1613.6518 - 1607.2718 = 6.3800  ->  0.5317 mm2 per spoke      against the constant's 3.03
```

**5.7x smaller**, so `reference_shipped_step_mm2` is high by 29.6 mm2, 1.8% of the wheel.
That quantity has now read **4.356, 3.032, 0.98 and 0.5317** on four genomes.

**The constant is NOT changed**, and that is §14's instruction rather than my caution:
*"Do not guess a new number. Replacing 3.03 with 0.98 would only re-stale it on the next
genome; what is needed is the scaling law, derived from `wheel_step_export._embed` the way
`wheel_geometry.junction_bite` was derived."* 0.5317 is a fourth reading of a quantity that
is not a constant — evidence FOR that item, not a candidate to close it with. It is written
into the constant's comment where the next person to reach for it will see it.

**And this is why the STEP half stays withheld for a filleted mesh** (§86). The anchor is
1.8% high before the fillet is even considered; adding a filleted region to a reference
built on it would report a number whose error is dominated by a constant everyone already
knows is stale.

### WHAT IS UNCHANGED

**No code path changed. No constant moved. No artifact regenerated, no export run,
`best_solution.json` untouched and still 2026-08-14.** This section is comments, one new
test, and two retractions. `make test` at §86 was 728 passed / 3 xfailed; the added test
brings `tests/test_filleted_mesh.py` to 36.

#### The successors, ranked — REVISED 2026-08-27 AFTER §87

1. **Make the layer cliff differentiable.** Unchanged from §86, and now the only item on
   this list with code in it.
2. **`EMBED_ALLOWANCE_PER_SPOKE_MM2`'s scaling law** — §14's open item 6, promoted from
   the tail because §87 gives it a fourth datapoint and a measured consequence (1.8% of the
   wheel, on the number `error_vs_shipped_step` publishes). It is also the only thing
   between a filleted mesh and a STEP comparison.
3. **Calibrate §73's two thresholds on a proper hold-out protocol** — unchanged.
4. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced.
5. **Per-REGION agreement on a filleted mesh** (§86) — `FILLETED_BLOCK_REGION` tags both
   fillet blocks `spoke` while `*_fillet_b` straddles the ring circle, so the mesh's
   `hub`/`rim` regions are no longer the full ring annuli.
6. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
7. **A bend that is a FUNCTION of the genome** (§56); **the REST of §45's audit list**
   (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the element-validity
   check** (§44).

---

## §88 — 2026-08-28. THE LAYER CLIFF IS DIFFERENTIABLE. IT WAS NEVER A SEARCH — `max_u (Z - a(u))/b(u)` IS THE SAME ROOT TO 2 ulp — AND THE GRADIENT §85 REFUSED IS WRONG BY 3.0% TO 24.4% ON ALL 37 UNCLAMPED GENOMES

§87's ranking item 1, named by §85 and carried through three rankings. `fillet=True` with no
`layer_profile` is the per-genome rule, whose entry is
`FILLET_LAYER_CLIFF_FACTOR * cliff(genes)`; §85 made `mesh_coords` refuse those meshes because
the frozen path held that entry constant, and wrote *"the day it stops refusing is the day
someone made the cliff differentiable"*. This is that day.

### THE CLIFF WAS NEVER A SEARCH, AND THAT IS THE WHOLE SECTION

The layer's width profile is `H(u) = _hermite(wall, end*wall, entry*layer_k, 0, u)`, and §82
already found the half that matters: `wall` and `layer_k` come out of the TANGENCY solve and
do not depend on `entry` at all, so `H` is **affine in `entry`**:

```
    H(u) = a(u) + entry * b(u),    a(u) = h00(u) wall + h01(u) end wall
                                   b(u) = h10(u) layer_k = u (1-u)^2 layer_k
```

`b(u) > 0` strictly inside `(0, 1)`, so `H(u) <= Z` exactly when `entry <= (Z - a(u))/b(u)`,
and the sampled minimum is at or below `Z` exactly when one sample is. Therefore

```
    cliff = max_u (LAYER_CLIFF_ZERO - a(u)) / b(u)
```

over the same `LAYER_CLIFF_SAMPLES = 401` grid `_fillet_curves` takes its minimum on. The two
endpoints drop out because `b` vanishes there and `H` is `wall` and `end*wall`, both orders
above `Z` — neither can ever be the minimum this solves for.

§82 called the cliff "a root-find over arithmetic instead of over thirty sector builds", which
is what made a per-genome profile affordable at all. It is not a root-find at all.
`_layer_cliff_from_scalars`
bisected `LAYER_CLIFF_BRACKET` ninety times; it now evaluates two Hermite bases and takes a
max, and `LAYER_CLIFF_BISECTIONS = 90` is deleted along with `study_fillet_block`'s
`CLIFF_BISECTIONS` that bound to it.

**MEASURED AGAINST THE BISECTION IT REPLACES: 2 ulp, worst relative 3.999e-16**, over the 98
junction-pairs of the shipped genome and all 48 genomes of `studies/study_fillet_block.json`.
Every published cliff from §68 to §85 is the same number.

### WHY THAT MAKES IT DIFFERENTIABLE, AND WHY A `custom_vjp` WOULD NOT HAVE

A bisection's derivative is ninety `sign` comparisons and a zero. §79 met that once already —
the tangency and ring-crossing solves — and the instrument there was `_newton_from_root`: the
answer is a root of an equation the traced path can re-evaluate, so freeze the root and take
one Newton step for the implicit-function-theorem derivative. **That instrument does not
apply here.** The cliff is not the root of an equation the mesh evaluates; it is the OUTPUT
of the search, consumed as a profile parameter. There is nothing to re-evaluate and nothing
for a wrapper to fix.

What removed the refusal was noticing the search was unnecessary. `wheel_adjoint.py`'s header
now carries the ordering that generalises: **is the search unnecessary, is its answer a root
worth freezing, and only then is it a discrete decision to be classified.** Two of this
project's three searches ended at the second question and one at the first; none needed a
`custom_vjp`.

### THE BINDING JUNCTION IS NOT FROZEN, AND THE DISTINCTION IS NOT COSMETIC

The sector's cliff is `max(cliff_hub, cliff_rim)`. `max()` compares and needs a concrete
bool, so it is `xp.maximum`, which carries the subgradient of whichever binds. That is the
OPPOSITE of the treatment `_fillet_curves`' four refusals get, and for the opposite reason:
this is a kink in a continuous function, not a change of construction. Freezing it would have
worked at almost every genome and been quietly wrong at the crossing.

It is also not academic. Over the 37 genomes measured — the shipped one and the 36 unclamped
members of the 48-genome box — **the rim binds 28 and the
hub binds 9** — so a frozen choice would have had to be per-genome anyway, and a hard-coded
one would have been wrong at a quarter of them.

### WHAT THE REFUSAL WAS WORTH: THE FROZEN GRADIENT IS WRONG BY 3.0% TO 24.4%

The control is the same mesh built at the pair the rule produced, HELD FIXED — bit-identical
geometry, and exactly the gradient §85 would have returned. The reference is a central
difference of `build_wheel(fillet=True)` itself, which re-derives the cliff from scratch at
every perturbed genome. At `coarse`, over the shipped genome and all 36 unclamped members of
the 48-genome box, on
`R_hub` and `R_rim`:

```
                                   worst      median       min     above 1%
  frozen layer profile             24.41%      6.95%     2.96%     37 of 37
  the rule, differentiated       2.39e-06%       --        --       0 of 37
```

The rule's row is **2.392e-08 relative**, worst over the 74 (genome, gene) rows and over a
three-step ladder `h/range in {1e-4, 1e-5, 1e-6}` — the ladder matters, because a single fixed
`h = 1e-5` reads 3.656e-04 at one genome and that is the reference secant's noise, not the
jacobian's. G11's gate is 1e-6.

At the shipped genome the rim binds, so the term lands on `R_rim`: **5.500e-02 frozen against
1.05e-09 for the honest one**, seven orders apart. `R_hub` is unaffected there to 4.4e-10,
which is the check that the term is the cliff's and not a general disagreement — and at
`in:[-1.0, 1.0]:0`, where the HUB binds, the two swap over exactly.

Not one of the 37 is below 1%. The refusal was not conservatism about a small term.

### AND IT REPAIRS A CACHE DEFECT §85 INTRODUCED THAT NOTHING COULD SEE

§85 put the RESOLVED pair into `coord_fn`'s cache key, correctly at the time: leaving `None`
there meant two genomes with different profiles sharing one key and the second being handed
the first's traced geometry. But the resolved pair is a function of the genome, so **the key
became genome-dependent** — and `coord_fn`'s own docstring is about exactly that failure for
the frozen roots: *"a jaxpr with one genome's roots baked in re-traces on every call"*, 2.78 s
each, against a 128-entry cache sized for an 8x8 phase lattice.

Nothing caught it because `mesh_coords` refused these meshes, so the key was never reached.
Now the record holds `None` again — meaning THE RULE, resolved inside the trace — the key
reads it raw rather than through `_layer_profile`, and the jaxpr is genome-independent.
Measured: two genomes at profiles `-0.362881` and `-0.362776` share **one** cache entry, and
37 genomes across four flank orientations produce **four** traces, one per orientation, not
one per genome. A SHIPPED-pair mesh still keys apart, which is the other half and is asserted
beside it.

### THE BRACKET IS A VALIDITY CHECK NOW, AND ITS UPPER END WAS NEVER CHECKED

`LAYER_CLIFF_BRACKET = (-8.0, 0.0)` survives, because the two sentinels it separates are still
separate: a cliff outside it is a genome this rule has never seen and must not quietly serve.
What changed is that it is applied to the answer rather than searched.

**And the closed form found a latent defect at the other end.** The bisection checked
`gone(lo)` and returned `None` for a cliff below -8; it never checked `hi`. A junction with no
layer at ANY negative entry — the cliff is positive — converged to the bracket's own zero and
reported it as a cliff. It fires at none of the 98 pairs, so this is a guard and not a change
of answer, but it is the same shape as §84's sentinel-with-two-meanings: a value the search
could only produce by running off its own end.

The check is SKIPPED under a trace, deliberately. A mesh being differentiated has already been
built, and its eager pass is what made that check.

### WHAT IS STILL REFUSED, AND IT IS TWO RATHER THAN THREE

`fillet_blocking="spoke"` (§47's construction re-spreads its station vector by ROUNDING a node
count) and a mesh whose radius the sector-fit clamp MOVED. Both unchanged, both still measured
in G11e rather than asserted. `refusals["per_genome_layer_profile"]` is KEPT as a key, pinned
at `None`, so an old artifact and a new one cannot be read as saying the same thing, and the
case moves to a new **G11f** that measures it: identity, the jacobian against the rule's own
central difference, the frozen control beside it, and the cache-sharing count.

The 12 clamped genomes of the 48 are refused exactly as before — the clamp's refusal is about
the RADIUS, not the profile, and this changes nothing about it.

### WHAT MOVED, AND WHAT THE 2 ulp COST THE MESH

Only `studies/study_gradient.json`, and only because G11e's shape changed. The eager cliff
moves by at most 2 ulp, and what that costs a built mesh was measured directly rather than
assumed: eighteen builds over `smoke`/`coarse`/`medium`, each compared against a mesh built at
the OLD bisection's pair passed explicitly — **worst coordinate difference 3.553e-14 mm, and
11 of 18 bit-identical.** Twelve orders below the tightest tolerance in the tree, so
`study_corner_singularity_fillet.json`, `study_reds_hub_share.json` and
`study_fillet_block.json` are all unmoved at every digit they print, and none is regenerated.

`fillet=None` is untouched: the unfilleted path never reaches a layer.

### AND `study_gradient.json`'s DIFF IS LARGE IN LINES AND SMALL IN MEANING — IT DOES NOT REPRODUCE TO THE BIT

Worth stating, because a 1586-line diff on an artifact invites the reading that something
moved. **It does not reproduce run to run, and that is not this change.** Two runs of the SAME
tree in the SAME pinned environment minutes apart differ in **127 non-timing floats**, worst
4.910e-07 relative — on `directional.rows[5].rel = 3.53e-10`, i.e. 1.7e-16 absolute. Against
the committed 2026-08-26 artifact the same comparison is 1476 floats, worst 9.096e-03 relative
on `unrolled.warm[3].worst_rel = 6.32e-11`, i.e. 5.8e-13 absolute. Every one of them is the
last digits of a quantity that is already a machine-precision residual, plus wall-clock
timings. **Structurally, the only differences are the four this section intended.**

`wheel_pool.PINNED_ENV`'s comment is not contradicted: its measurement is *"two plain serial
runs of one `coarse` adjoint"* agreeing exactly, and that is a narrower claim than "this whole
artifact is reproducible". Not chased further, because every difference is orders below every
gate the file carries — but recorded here rather than left for the next reader to rediscover
from a diff.

### A STALE CLAIM CORRECTED IN PASSING

`studies/study_reds_hub_share.py`'s `sweep` docstring said *"`mesh_coords` still refuses a
filleted mesh outright, so nothing here reaches the optimizer"*. False since §79 and doubly so
now. What is still true is the part that carries the sentence's point: nothing in
`wheel_objective` builds a filleted mesh, so the optimizer sees neither — **a fact about the
objective, not about `mesh_coords`.** Corrected in place; no artifact depends on it.

#### The successors, ranked — REVISED 2026-08-28 AFTER §88

1. **`EMBED_ALLOWANCE_PER_SPOKE_MM2`'s scaling law** — §14's open item 6, up from 2 now that
   §88 is done, and the only thing between a filleted mesh and a STEP comparison. §87 gave it
   a fourth datapoint (4.356, 3.032, 0.98, 0.5317 mm^2/spoke) and a measured consequence:
   `reference_shipped_step_mm2` is 1.8% high. Derive the law from `_embed` the way
   `junction_bite` was derived.
2. **WIRE THE FILLET INTO THE OBJECTIVE — NEW, and §88 is what makes it askable.** Every
   piece now exists: a filleted mesh that builds across the box, an area reference that
   describes it (§86), and a gradient through the DEFAULT filleted path rather than through a
   named pair. What is missing is a decision, not a mechanism: `wheel_objective` still prices
   `R_hub` through a `Kt` surrogate that is exactly flat over half its feasible range (§75),
   and the filleted mesh is 2-3x the cost of the unfilleted one. Price it before adopting it.
3. **Calibrate §73's two thresholds on a proper hold-out protocol** — unchanged, and still the
   oldest thing on the list.
4. **Apply the fold gate to the draw and re-derive the box** (§58) — one word, priced.
5. **Per-REGION agreement on a filleted mesh** (§86) — `FILLETED_BLOCK_REGION` tags both
   fillet blocks `spoke` while `*_fillet_b` straddles the ring circle, so the mesh's
   `hub`/`rim` regions are no longer the full ring annuli.
6. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
7. **A bend that is a FUNCTION of the genome** (§56); **the REST of §45's audit list**
   (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the element-validity
   check** (§44).

## §89 — 2026-08-29. §48's SURVIVING CLAUSE IS RETIRED, AND NOT BY CONCESSION: THE BARRIER PUTS THE FILLETED MESH AND THE UNFILLETED ONE ON THE SAME SIDE OF ITS OWN TARGET, ON THE SAME GENOME, WITH THE UNFILLETED ONE FURTHER UNDER

`FILLET_PLAN.md`'s header has said since §74 that ONE thing keeps `fillet=` a measurement
instrument rather than a path the optimizer may take: *"half of each drawn genome box sits
under `MIN_SJ_TARGET` (8/16 and 16/32), which is §48's surviving clause and is now the ONLY
reason."* `wheel_wheel.py` carried the same sentence and
`test_nothing_wires_the_fillet_into_the_objective` carried it as a check.

**It is wrong twice, and the second way is the one that matters.** Its numbers describe a
profile the tree stopped building at §85, and its criterion does not separate the filleted
mesh from the mesh the optimizer builds today.

### THE NUMBERS WERE THE SHIPPED GLOBAL PAIR'S

8 of 16 and 16 of 32 are `fit_clamp` / `fit_clamp_held_out` at `profile="shipped"` — the
two constants `FILLET_LAYER_ENTRY_SLOPE` / `FILLET_LAYER_END_OFFSET`. §82 adopted
`per_genome_layer_profile` and §85 made it what `fillet=True` takes when nobody asks. The
same tables at the adopted rule read **15 of 16 and 31 of 32** — quoted in §82's own text,
and in the committed artifact since §84 extended the factor grid past 0.55. Nothing restated
the clause, so a 16x claim about the barrier stood for four sections describing a mesh
nobody was building.

### AND THE COUNT WAS TAKEN WITH AN INSTRUMENT THE BARRIER DOES NOT USE

Every barrier count this arc has published came from `study_fillet_block.block_quality`,
which scores every **1x1 sub-cell of a block's node grid**. The mesh's elements are Q9 and
span **2x2** of those, and `wheel_objective.t2_vector` reads `wheel_mesh.scaled_jacobian`,
which slices `conn[:, :4]` — the element's four CORNER nodes. Two different measurements of
two different cells.

**Proved rather than inferred.** `block_quality` on the subsampled grid `[::2, ::2]`, which
is the Q9 element's corners and nothing else, reproduces the assembled mesh's number to
every digit:

```
                       sub-cells                     Q9 corners                  assembled
  in-sample  0     rim_ring_free   0.259141038   rim_ring_free  0.253191857     0.253191857
  in-sample  3     hub_junction    0.365802399   rim_ring_free  0.371207040     0.371207040
  in-sample 10     spoke          -0.050823172   rim_ring_free  0.410527770     0.410527770
```

The two disagree on **11 of the 16** in-sample genomes at the adopted factor, and on genome
10 they disagree in SIGN.

### THE CONTROL THE CLAUSE NEVER HAD, WHICH IS WHAT RETIRES IT

The barrier is a property of A MESH, and `fillet=None` is a mesh. Built assembled, both
ways, on the same genomes, at `coarse` — the config `wheel_objective.objective` defaults to
— through the barrier's own instrument:

```
                   clears MIN_SJ_TARGET        held-out    held-out    meshes that
                   in sample    held out       median J     worst      fold inside
                                                            barrier    an element
  fillet=True      16 of 16     31 of 32        0.3382       18.2047    1 of 48
  fillet=None      16 of 16     31 of 32        0.7447      327.1699    1 of 48
```

(the in-sample worst barrier is 0.0000 on both rows — nothing in that draw is marginal at
all; the folding mesh is the same genome on both rows and is the subject of the last
section below.)

**The one held-out genome under the target is under it either way, and further under it
unfilleted**: 0.129843 against the filleted 0.177513, 72 marginal elements of 4704 against
12 of 5952, and the barrier TERM `t2_vector` actually sums — a sum of soft barriers over
every element, not a min-crossing count — **327.17 against 18.20**. Its part does not
self-intersect (closed-form fold margin +0.5207 mm), so §58's gate does not dispose of it.

At `medium` the held-out rows read 31 of 32 and 31 of 32 and the same genome stays the one:
0.173321 filleted against 0.125099 unfilleted.

**AND ONE ROW GOES THE OTHER WAY, WHICH IS STATED HERE RATHER THAN LEFT IN THE ARTIFACT.**
At `medium` IN SAMPLE the filleted mesh is worse — **15 of 16 against 16 of 16**, barrier
term 2275.54 against 0.0000. It is worse on exactly one genome, and that genome is the
subject of the last section below: its part self-intersects, BOTH meshes fold an element on
it, and the corner-quad metric surfaces the filleted mesh's fold while missing the
unfilleted one's, which is DEEPER. So the row is the fold finding's worked example rather
than a counterexample to this one — but it is the reason the claim above is made at
`coarse`, where the objective runs, and not at every config in the tree.

**A criterion that puts both meshes on the same side of its own target cannot be the reason
to refuse one of them.** §48's surviving clause is retired — on `coarse`, on both draws, and
on the held-out draw at `medium` as well.

### THE HONEST OTHER HALF: WHAT THE FILLET COSTS IS HEADROOM, NOT CROSSINGS

Min scaled Jacobian is LOWER on **31 of the 32** held-out genomes, median 0.3382 against
0.7447 — the filleted mesh spends about half the room above the barrier. It simply does not
spend it in the place the clause was pointing at. Anyone reinstating a barrier-shaped
objection has to make it about the margin, and then say what margin, and then measure it.

### A FOLD ALL FOUR VALIDITY INSTRUMENTS MISS, AND IT IS IN THE SHIPPED MESH

Chasing the sign disagreement above found something that is not about the fillet and is
recorded because nothing in the tree records it. In-sample genome 10 at `medium`, the worst
spoke element:

```
  wheel_mesh.scaled_jacobian   corner quad          -0.051415    catches it
  wheel_mesh.element_areas     corner quad          +3.130e-02   MISSES it
  ff.mesh_gauss_verdict        3x3 Gauss, order 2   +4.035e-04   MISSES it
  block_quality                1x1 sub-cells        -0.051415    catches it (at `coarse`)
  det J over the reference square, 121x121          -6.393e-04   166 of 14641 points
```

**The quadrature the assembly integrates on does not sample the folded corner**, so criterion
C — PART 6's own criterion, and the one this arc has treated as the ground truth — reads the
element clean. Two things follow and both are worth having written down:

- **It is not the fillet's.** `fillet=None` on the same genome folds too, and DEEPER: 24
  elements at min `det J` -1.241e-02 at `coarse` against the filleted mesh's 12 at
  -2.893e-03. At `medium` both fold on 24 elements and the unfilleted one is again deeper
  (-2.061e-03 against -6.393e-04) — while the corner-quad metric flags only the FILLETED
  one. The proxy is not tracking the fold it is a proxy for.
- **It is one genome of 48, and that genome's PART self-intersects.** Its closed-form fold
  margin is -0.0131 mm against a 0.1 mm limit, i.e. it is exactly what §58's gate exists to
  reject and what this study's draw filter still does not ask. Two of the sixteen in-sample
  genomes self-intersect (8 and 10) and only 10 also folds an element; the held-out draw has
  no self-intersecting genome and folds nothing, filleted or not, at either config.

So this is filed, not acted on. Acting on it means changing a validity gate the whole tree
reads, which is its own unit of work and not this arc's.

### WHAT MOVED

`studies/study_fillet_block.py` gains `sweep_barrier_control` and the two instruments it
needs, wired into both configs and both draws, and the artifact is regenerated with it.
`wheel_wheel.py`'s scope comment and
`test_nothing_wires_the_fillet_into_the_objective` are corrected in place — **the gate
stays**. Nothing wires `fillet=` into the objective, because the reason not to has changed
rather than gone: it is a DECISION about cost and about the surrogate, and a decision is
taken in a record, not by a keyword reaching a call.

**And one number this section did NOT re-measure, flagged because it is about to carry the
decision.** §88's ranking item 2 says the filleted mesh is *"2-3x the cost of the unfilleted
one"*. Nothing in this tree measures that. The element counts at `coarse` are 5952 against
4704 — 1.27x — which is not the same quantity as solve time and does not refute it, but it
does mean the cost half of the remaining blocker is currently a number with no measurement
behind it.

#### The successors, ranked — REVISED 2026-08-29 AFTER §89

1. **PRICE THE FILLETED OBJECTIVE, which is now the whole of what is left.** §88's item 2
   said *"what is missing is a decision, not a mechanism"* and named two terms; §89 removes
   the third (mesh validity) and finds one of the two unmeasured. So: measure the cost —
   one `wheel_objective.objective` evaluation filleted against unfilleted, at `coarse`,
   forward and adjoint — and read `R_hub` through the filleted mesh against the flat `Kt`
   surrogate (§75). Those two numbers ARE the decision.
2. **`EMBED_ALLOWANCE_PER_SPOKE_MM2`'s scaling law** — §14's open item 6, unchanged from
   §88's item 1 and still the only thing between a filleted mesh and a STEP comparison.
3. **The sub-element fold, as its own unit** — NEW. Four instruments, one fold, and the one
   the tree calls ground truth misses it. Decide whether `scaled_jacobian` gets a Q9 path
   or whether `gauss_verdict` gets a denser rule, then apply §58's gate to the draw (which
   was item 4 and is now the same question) and re-derive the box.
4. **Calibrate §73's two thresholds on a proper hold-out protocol** — unchanged, and still
   the oldest thing on the list.
5. **Per-REGION agreement on a filleted mesh** (§86).
6. **Carry `axle_drop_interp_mm` into `study_contact`** next time it runs anyway (§67).
7. **A bend that is a FUNCTION of the genome** (§56); **the REST of §45's audit list**
   (§49); **G1's fourth revision**; **§32's successors 3 and 4**; **the element-validity
   check** (§44).
