# REDS_PLAN.md — clear the five inherited failures

**Working notes for one arc. Written 2026-08-15, after PLAN.md §30.**

**Ignore version control entirely. Do not commit, branch, stage, revert or otherwise touch
git — it is not part of this project's workflow and nothing here depends on it.**

---

## Why this arc exists

`make test` has read **`5 failed`** since §19. All five are documented and deliberate —
`CONTACT_PLAN.md` lines 140–151 registered them before that arc began, and every arc since has
re-declared them. That was the right call at the time. It has now become a cost:

- PLAN §28 found that one of the "known reds" (`test_the_bite_is_the_volume_divided_by_the_right_thickness`)
  had been mislabelled "an export-precision defect" and carried through three plan files
  unchecked. It was a tolerance the design had outgrown, and it was **hiding inside the known-red
  count**.
- PLAN §29 and §30 both turned on the claim "no new red". Establishing that claim required
  opening `CONTACT_PLAN.md` to look up which five were expected, because `5 failed` on its own
  carries no information about *which* five.

A suite that reads `5 failed` on a good day cannot cheaply distinguish a sixth. **The goal of
this arc is `make test` reading `0 failed`, with every one of the five either fixed, restated
against the statistic that actually carries its claim, or explicitly converted into a recorded
judgement that no longer runs as a gate.**

### THE RULE THAT GOVERNS THIS ARC, and it is not negotiable

**Measure before touching a threshold, and never re-fit a gate to the run that breached it.**
This tree has a standing rule (PLAN §14) that a pre-registered gate is not moved because a
design failed it. Two of the five below turn out to be gates on statistics that cannot support
them — that is a different finding and a legitimate fix, but the distinction has to be *earned
by measurement* and written down. PLAN §28 is the worked example of the acceptable form:

> Replacing a stale constant with the arithmetic it should always have been is not loosening.

If you cannot state the fix in that form, leave the test red and record why.

---

## How to run things

Two virtualenvs. `.venv-opt/bin/python` runs everything in this arc. The Makefile exports
`PYTHONPATH := $(CURDIR)/src`, and `pyproject.toml` puts `src`, `studies` and `.` on pytest's
path, so tests can `import study_wheel_fea` directly.

```bash
make test                                    # full suite, ~28 min, prints the summary line
.venv-opt/bin/python -m pytest -q            # DOUBLE quiet (addopts already has -q):
                                             #   pytest SUPPRESSES the final count line.
                                             #   Use `make test` when you want the totals.
.venv-opt/bin/python -m pytest tests/test_gnl.py tests/test_wheel_fea.py -q --tb=long
```

The five node ids, in full:

```
tests/test_gnl.py::test_the_correction_is_not_a_constant_over_the_design_space
tests/test_gnl.py::test_the_correction_enters_at_first_order_in_the_load
tests/test_wheel_fea.py::test_the_rim_band_holds_a_large_minority_of_the_compliance
tests/test_wheel_fea.py::test_the_beam_to_wheel_ratio_is_not_a_constant
tests/test_wheel_fea.py::test_a_thicker_rim_monotonically_stiffens_the_wheel
```

Baseline at the time of writing: **461 passed, 5 failed, 466 collected.**

---

## THE HEADLINE FINDING — three of the five are already diagnosed, and two share one cause

Measured 2026-08-15 while scoping this arc. **These numbers are in hand; do not re-derive them
before starting, but do re-run them to confirm the tree has not moved.**

### A. The two `ratio > 3.0` gates are a seed lottery on a sample-size-dependent statistic

`study_wheel_fea.run_beam_blindness` and `study_gnl.run_design_space` both compute

```python
"..._ratio": float(r.max() / r.min())        # max/min over the DRAWN rows
```

**A max/min ratio is not a fixed property of a distribution — it grows without bound with the
number of draws**, because it is an estimator of the range. Both tests hard-code a small `n` and
`seed=7`, and both assert `ratio > 3.0`. Measured:

`test_the_beam_to_wheel_ratio_is_not_a_constant` (`run_beam_blindness`, floor pinned at 2.0):

| seed (n=6) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | **7** | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| ratio | 1.63 | 3.10 | 2.12 | 2.53 | 2.93 | 2.26 | 2.04 | **2.41** | 6.89 | 3.86 |

Range **1.63 – 6.89** across seeds; **passes `> 3.0` in only 3 of 10**. And with the seed held
at 7: **n=6 → 2.41, n=12 → 10.00, n=24 → 34.97, n=48 → 34.97.**

`test_the_correction_is_not_a_constant_over_the_design_space` (`run_design_space`, n=4):

| seed | **7** | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| ratio | **2.17** | 3.21 | 3.83 | 3.91 | 5.29 |

Passes in 4 of 5 seeds. With seed 7 held: **n=4 → 2.17, n=8 → 4.20, n=12 → 9.03.**

**In both tests, seed 7 — the one hard-coded — happens to be the low outlier.** That is the
whole failure. Nothing about the wheel changed.

**AND THE CONCLUSION THE TESTS EXIST TO PROTECT IS ROCK SOLID.** Both studies compute a second,
stable statistic, and it is the one the argument actually rests on:

```python
"correction_factor_is_defensible": bool(r.std() / r.mean() < 0.10)
```

`correction_factor_is_defensible` is **False at every seed and every n tested**, with CV in
**0.145–0.516** (beam blindness) and **0.258–0.771** (GNL) against a 0.10 bar — never within 40%
of passing. The Stage-2.5 off-ramp ("correct the beam model with one factor and skip Stages 2
and 3") is closed, exactly as both docstrings claim. Both tests already assert this as their
FIRST assertion, and it passes. It is only the secondary `ratio > 3.0` line that fails.

> **This is PLAN §28's pattern exactly**: a gate whose bound the statistic cannot support,
> which looked green for a while by luck of the draw. §28's rule applies — replacing an
> unstable estimator with the stable one carrying the same claim is not loosening.

### B. `test_a_thicker_rim_monotonically_stiffens_the_wheel` fails only at the `smoke` rung

The test sweeps `rim_outer` over (49.7, 50.0, 50.6, 51.2) at **`smoke`** and asserts two things.
Measured across rungs:

| rung | drops (mm) | monotone? | brackets 2.0? |
|---|---|---|---|
| smoke | 1.8758, 1.5453, 1.1823, 0.9813 | **yes** | **no** |
| coarse | 1.9798, 1.6208, 1.2320, 1.0193 | **yes** | **no** |
| medium | 2.0034, 1.6399, 1.2462, 1.0308 | **yes** | **yes** |

**The monotonicity assertion — the one the test is named for — passes at every rung.** What
fails is `drops[-1] < TARGET_DEFLECTION_MM < drops[0]`, an **absolute** deflection claim
evaluated at **the least converged mesh in the tree**. The `smoke` rung reads ~6% low (PLAN §29's
ladder: −5.955% under SVK), which is more than enough to lose a bracket whose upper edge sits at
1.9798 by `coarse` and 2.0034 by `medium`.

The docstring justifies the reduced fidelity — *"the finding is a factor of ~30 and does not
need a converged mesh to be visible"* — and that justification is sound **for the monotonicity**
and unsound for the bracket. PLAN §29 retired exactly this kind of claim at the plan level: an
absolute distance from 2.0 mm quoted without naming its rung.

### C. `test_the_rim_band_holds_a_large_minority_of_the_compliance` — REAL, and still a human's call

Hub compliance share, `< 0.03`:

| rung | spoke | hub | rim |
|---|---|---|---|
| smoke | 0.6593 | **0.0392** | 0.3016 |
| coarse | 0.6422 | **0.0417** | 0.3161 |
| medium | 0.6364 | **0.0433** | 0.3203 |

§14 recorded **0.0321** and called it 7% over. It is now **0.0417** at the same rung — **+30%
since §14**, i.e. it moved again with §26's promotion — and it is *drifting upward with mesh
refinement*, so it is not a converged quantity either.

PLAN §14 item 4b left this open deliberately and said why:

> "This is the one where the *direction* is surprising: thinner, floppier spokes should push
> compliance toward the spokes and the hub share DOWN. It went up. The plausible cause is
> `R_hub` dropping 1.5598 → 0.5790 — much less material at the hub junction — but that is a
> hypothesis and it has not been measured. Least urgent of the eight and the only one whose
> sign is not understood."

`R_hub` on the shipped genome is now **0.6636**. **The hypothesis is still unmeasured.** This
one is not a test edit and must not be treated as one.

### D. `test_the_correction_enters_at_first_order_in_the_load` — REAL, and pre-registered

`small_load_rel_diff` = **0.002007** against `GATE_SMALL_LOAD_REL = 1e-3` (`studies/study_gnl.py:70`).
Over by 2.0×. The test's first assertion — `fitted_exponent` = 1.0393 inside (0.7, 1.4) — passes,
so the SVK path is behaving; the coefficient moved, not the exponent.

**This gate is pre-registered and has been deliberately held.** `study_gnl.py` records it as
*"written down BEFORE the study was run, per the plan's rule"*, PLAN §14 item 4a decided
explicitly that **it stands**, and SVK_PLAN Step 0 re-declared it. It is red because the shipped
1.2 mm wheel is ~5.5× more geometrically nonlinear than the GA/beam design it replaced. That is
a true statement about the wheel.

**Do not move `GATE_SMALL_LOAD_REL`.** §14 already refused to, twice, on the grounds that
re-fitting a pre-registered gate to the design that breached it is the exact move the rule
exists to prevent.

---

## THE PLAN

Five items. **1 and 2 are the same fix and should be done together. 3 is independent and
cheap. 4 is real work. 5 is a decision, not an edit.**

### Step 0 — reproduce the baseline

Run `make test`. Gate: **exactly 5 failures, exactly the five node ids above**, and each
reproducing the value quoted in this file:

| red | expected value | gate |
|---|---|---|
| `..._beam_to_wheel_ratio...` | `fea_over_beam_ratio` = 2.4128 | `> 3.0` |
| `..._correction_is_not_a_constant...` | `iso_rel_diff_ratio` = 2.1672 | `> 3.0` |
| `..._thicker_rim...` | `drops[0]` = 1.8758 | `< 2.0 < drops[0]` |
| `..._rim_band_holds...` | hub share = 0.041656 | `< 0.03` |
| `..._correction_enters_at_first_order...` | `small_load_rel_diff` = 0.0020070 | `< 1e-3` |

**A sixth failure, or any of these five reading differently, stops the arc.** It means something
moved that nobody recorded, and every step below would inherit it. Record the count, the wall
clock and the node ids, the way SVK_PLAN Step 0 does.

### Step 1 — the two ratio gates (items 1 and 2)

**First, prove the instability rather than assuming this file.** For each of the two studies,
sweep seed (≥10 values) and `n` (at least 3 values) and record the table. Confirm:

- the ratio spans a factor of ≥3 across seeds at the test's own `n`;
- the ratio grows monotonically with `n`;
- `correction_factor_is_defensible` is `False` in **every** cell, and the CV never approaches
  0.10.

**If any of that fails to reproduce, stop — the diagnosis in section A is wrong and the tests
stay red.**

**Then the fix.** The claim both tests exist to make is *"the correction factor is not
defensible, so the Stage-2.5 off-ramp does not exist."* That claim is carried by the CV, not by
max/min. Replace the `ratio > 3.0` assertion with one on a statistic that is stable under seed
and sample size — the CV against its 0.10 bar, with a margin derived from the measured spread
rather than picked.

Requirements on the replacement:

- **It must be strictly harder to pass than the old line was on the runs where the old line
  passed.** Demonstrate this, the way §28 demonstrated its derived tolerance was stricter than
  the constant it replaced on `36aed36`. If the new assertion is weaker, it is loosening and it
  is not allowed.
- **The docstring must record the retirement**: the measured seed/`n` table, that seed 7 was the
  low outlier in both, and that a max/min ratio is an estimator of range and therefore
  sample-size dependent. Someone will otherwise reintroduce it.
- Consider asserting the ratio's *instability* directly as a regression pin — i.e. that max/min
  moves with `n` — so the reason it was retired stays measured rather than merely asserted.

Both studies' report dicts still publish `..._ratio`; leave them publishing it. It is a
diagnostic, and PLAN §29's `criterion_met: false` is the cautionary tale about *deleting* a
number rather than scoping it — but see the warning in Step 5.

### Step 2 — the rim sweep (item 3)

Keep the monotonicity assertion exactly as it is: it passes at every rung and it is the finding.

For the bracket, pick one and record why:

- **(a)** evaluate the bracket at `medium`, where it holds (2.0034 > 2.0 > 1.0308), and accept
  the cost — the sweep becomes four `medium` solves instead of four `smoke` ones; measure the
  runtime before choosing this, since it lands in the default suite;
- **(b)** drop the bracket and state in the docstring that an absolute deflection claim cannot
  be made at `smoke`, citing PLAN §29 — the target is bracketed at `medium` and that is recorded
  in the docstring rather than asserted at a rung that cannot support it.

**(b) is the recommendation.** The test's name and its documented purpose are about
monotonicity; the bracket is a characterisation pin that was never what the test was for, and
§29 retired the plan-level version of the same claim. But (a) is defensible and cheap if the
`medium` sweep turns out to cost little — **measure it, do not assume.**

Either way: **do not weaken the monotonicity assertion**, and do not change `rim_outer` values
to make a bracket fit.

### Step 3 — the hub compliance share (item 4). REAL WORK.

This is the only one of the five that is a live question about the wheel, and PLAN §14 named the
measurement that has never been done. Do it:

1. **Measure the sign.** Sweep `R_hub` across its gene-box range (0.4 – 4.0) on the shipped
   genome, holding everything else fixed, and record the hub compliance share. §14's hypothesis
   is that the share rises as `R_hub` falls. Confirm or kill it. This is a handful of
   `solve_wheel` calls and needs no optimizer.
2. **Separate genome from mesh.** The share drifts 0.0392 → 0.0417 → 0.0433 up the rungs, so
   part of the gap to `< 0.03` is discretisation. Report the share at every rung for the shipped
   genome **and** for `best_solution_ga_beam.json`, the 2×2 that §14's own rule prescribes
   (design × mesh). Note the parallel to §30: the hub share may not be a converged quantity, in
   which case a fixed `< 0.03` bound is the §29 problem again — **check the drift against the
   bound before proposing any change to the bound.**
3. **Then decide, and the decision is a human's.** §14 said so and this file does not overrule
   it. Write the measurement up in `PLAN.md` and put the recommendation in front of the user
   with the numbers. Possible outcomes: the bound is fine and the wheel has a real hub-stiffness
   regression; the bound was calibrated on a mesh-dependent quantity and needs restating with a
   rung; or `R_hub` is genuinely under-sized and this is a design finding that belongs in the
   objective.

**Do not touch `< 0.03` before step 3.2 is measured and written down.**

### Step 4 — the GNL small-load gate (item 5). PROBABLY STAYS RED.

Expect this to remain red, and expect that to be correct.

The gate is pre-registered, §14 decided it stands, and the wheel really is more nonlinear than
the design the gate was written for. **The honest options are not "move the gate":**

- **(a)** Leave it red and stop calling it a failure — convert it into a recorded, non-executing
  characterisation (see Step 5), so the suite reads 0 failed and the finding is not lost.
- **(b)** Escalate the underlying question, which is the one §14 actually identified and which
  is bigger than this arc: *whether linear kinematics is still an acceptable default for a
  1.2 mm wall at all.* §14 called this "the most important thing §14 found". If the answer is
  no, the fix is in the physics defaults, not in the test.

**(a) is in scope for this arc; (b) is not — flag it and leave it.** Under no circumstances
raise `GATE_SMALL_LOAD_REL` to 3e-3 or similar; that is the forbidden move and it has already
been refused twice.

### Step 5 — make the remaining "deliberate" reds not read as failures

Whatever survives steps 1–4 as a genuine, accepted, non-actionable finding must stop consuming
the suite's signal.

**Checked while writing this file: there is no existing mechanism.** `xfail` appears nowhere in
`tests/`, and `pyproject.toml` configures neither `markers` nor `xfail_strict`. So this step
introduces one, and introducing it is part of the work rather than a detail — pick it
deliberately:

- `pytest.mark.xfail(strict=True)` is the strong option: it keeps the assertion *executing*, so
  the day the wheel changes and the test would pass, **`strict=True` turns that into a failure**
  and the finding is re-opened automatically. This is the right tool for item 5 and probably for
  the survivor of item 4.
- Deleting a test is the wrong tool. PLAN §29's whole lesson is that a correctly-quarantined
  number was load-bearing somewhere else; a deleted assertion cannot be found by grep.

**Every xfail must carry a `reason=` naming the PLAN section that decided it**, so `5 failed`
never again has to be resolved by opening another plan file.

Set `xfail_strict = true` in `pyproject.toml` rather than passing `strict=True` at each site, so
the default cannot be forgotten at the next one — and note in the same commit-free record that
this changes the meaning of any future bare `xfail` in this tree.

---

## The end state this arc is aiming at

```
make test  ->  0 failed
```

with:

- items 1 and 2 **fixed** — asserting the stable statistic that carries their claim, with the
  retirement of max/min documented and the instability itself pinned;
- item 3 **fixed** — monotonicity asserted, the absolute bracket either moved to a rung that
  supports it or retired with a citation;
- item 4 **measured and escalated** — §14's unmeasured hypothesis finally tested, written up in
  `PLAN.md`, and the threshold decision put to the user with numbers;
- item 5 **strict-xfail'd** with a reason pointing at PLAN §14 item 4a, and the linear-kinematics
  question flagged as a successor;
- **a new PLAN.md section** recording all of it, and `CONTACT_PLAN.md`'s inherited-red table
  updated to say the arc closed it, so the next session does not re-derive this list.

## What must NOT happen

- No threshold moved to fit a run that breached it. If a bound changes, the new bound is
  *derived* and demonstrably stricter where the old one passed.
- `GATE_SMALL_LOAD_REL` stays at `1e-3`.
- The monotonicity and `correction_factor_is_defensible` assertions are not weakened — they are
  the findings.
- No test deleted. Retired assertions become documented, executing xfails.
- Nothing in `best_solution.json` or any exported artifact is touched. **This arc changes tests
  and their supporting measurements only** — with the single exception that Step 3 may produce a
  design finding, which gets written up and handed over rather than acted on.

---

## EXECUTION RECORD — 2026-08-15. THE ARC RAN AND ALL FIVE ARE CLOSED.

**The full write-up is `PLAN.md` §31.** This is the plan file's own record, kept here so the
instructions above are never read without their outcome.

### Step 0 — GATE: PASS

```
make test:   5 failed / 469 passed in 1962.89 s (32:42)   [474 collected]
  exactly the five node ids listed above, and nothing else.
  wall clock inflated — the step 1-3 measurement drivers ran alongside on other cores.
```

All five values reproduced to every digit, each through its own driver rather than through
pytest, so the check does not depend on the test files this arc then edited:
`fea_over_beam_ratio` **2.41276**, `iso_rel_diff_ratio` **2.16721**, `drops[0]` **1.8758**,
hub share **0.04165644522132511**, `small_load_rel_diff` **0.0020070**.

**The count differs from this file's "466 collected" and it was run down rather than waved
through.** 474 = §28's 452 + 14 (`tests/test_deflection_gci.py`) + 8
(`tests/test_corner_singularity.py`, written 12:39 today, after the 466 figure was taken and
before this arc began). All 8 pass. Not a sixth red; a stale count in the scoping line.

### What each step did

| step | outcome |
|---|---|
| **1 — the two ratio gates** | **FIXED.** 109 cells measured (`make reds-ratio`, 3 m). Seed 7 is the low outlier in both studies; `> 3.0` passes at only 7/20 (beam) and 11/20 (gnl) seeds, and the ratio crosses its own gate as `n` grows. Replaced by `cv > 0.14` at five seeds — the arithmetic `correction_factor_is_defensible` is *defined* in, so §28's move, not a loosening. `correction_factor_is_defensible` is False in **109/109** cells |
| **2 — the rim bracket** | **FIXED, and option (a) was measured and rejected.** Not on cost — the `medium` sweep is only +4.0 s — but because at `medium` the bracket clears 2.0 by 0.169% while the medium→fine drift on the same quantity is 0.50%. A margin smaller than the quantity's own convergence error cannot carry a gate, so **(b)**: retired to the docstring with its rung, and replaced by the mesh-robust span ratio (1.912/1.942/1.944, gated at 1.5) |
| **3 — the hub share** | **MEASURED IN FULL, ESCALATED, BOUND UNTOUCHED.** §14's `R_hub` hypothesis is not just unmeasured, it is **impossible** — fillets are not meshed (`wheel_wheel.py:44`), and the solved wheel is bit-identical across the whole `R_hub` box. The real driver is spoke curvature: `cy4` alone closes 102.4% of the gap to the ga_beam design. The 2×2 says the bound is not a mesh artefact — ga_beam is converged at 0.0139–0.0143 and passes, while the shipped genome is 30.5% over at the coarsest rung and 54.3% over at the finest. **Strict `xfail` pending a human's call** |
| **4 — the GNL small-load gate** | **STAYS RED, option (a).** `GATE_SMALL_LOAD_REL` untouched at `1e-3`, refused for the third time. Converted to a strict `xfail` carrying §14 item 4a. The linear-kinematics successor question is **flagged and not acted on**, as scoped |
| **5 — the xfail mechanism** | **`xfail_strict = true` in `pyproject.toml`**, verified by probe: an xfail that passes is reported `XPASS(strict)` and **fails the suite**. Both xfails name the deciding PLAN section in `reason=`. No test deleted |

### The one thing this arc did that the plan above did not ask for, and why

**Both xfailed assertions were SPLIT OUT of tests that also contained passing assertions.**
While the hub bound lived inside `test_the_rim_band_holds_a_large_minority_of_the_compliance`,
the rim and spoke shares — what that test is *named* for — were **not being checked on any
run**, and the same was true of the exponent assertion inside the GNL test. Marking either
test `xfail` whole would have made that permanent. Splitting recovered two passing assertions
that had been dark since §14.
