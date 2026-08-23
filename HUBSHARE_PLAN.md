# HUBSHARE_PLAN.md — should hub compliance be an objective term?

**Open arc #3. Created 2026-08-16 from PLAN §31 item 4's filed successor. Nothing started.**

**VERSION CONTROL IS PART OF THIS PROJECT'S WORKFLOW — CHANGED 2026-08-19.** The rule that
stood here read *"Ignore version control entirely. Do not commit, branch, stage, revert or
otherwise touch git."* **It is superseded.** The rules live in `PLAN.md`'s header block and
**only** there, so they cannot drift across ten files: one commit per finished unit of work
on `feature`, `make test` green first, never while a study driver is mid-write, a study
commit carries its regenerated `.json` and `.jpg`, a promotion is one atomic commit and never
one file — and **commits carry no assistant or tool attribution, no `Co-Authored-By:`
trailer, no session link, no generated-with footer.**

---

## Why this arc exists

§31 measured the hub compliance share in full, decided the threshold question, and filed the
design question here. **The threshold is settled and is not what this arc is about.**

Settled in §31, do not reopen without new evidence:

- `compliance_split["hub"] < 0.03` **stays at 0.03**, and the assertion stays a strict `xfail`
  (`tests/test_wheel_fea.py::test_the_hub_junction_holds_under_three_percent_of_the_compliance`).
- §14's hypothesis — that the share rose because `R_hub` fell — is **structurally impossible**.
  Fillets are not meshed, so the wheel is bit-identical across the whole `R_hub` box.
- The bound is neither unreachable nor a mesh artefact: `best_solution_ga_beam.json` meets it
  **converged**, 0.0139–0.0143 across five rungs, with 53% to spare.

**What is open is the design question:** the shipped wheel holds **3.2× the hub compliance
share of the design it replaced**, and §31 identified the lever but did not pull it.

## The measurement already in hand — `make reds-hub`, ~40 s

One-at-a-time gene swaps from the shipped genome toward `best_solution_ga_beam.json` at
`coarse` (hub share 0.0417 → 0.0138):

| gene | shipped | ga_beam | hub | closes the gap by |
|---|---|---|---|---|
| `cy4` | 6.4375 | 29.2919 | **0.0132** | **102.4%** |
| `cy3` | 9.4191 | 24.3248 | 0.0219 | 70.9% |
| `cy1` | 8.7212 | 27.9529 | 0.0250 | 59.7% |
| `cy2` | 11.8088 | 31.7187 | 0.0255 | 57.9% |
| `t0` | 1.4738 | 2.4774 | 0.0561 | **−51.9%** |

**`cy4` alone takes it under the bound.** The shipped spoke is much flatter (cy 6.4–11.8
against 24–32), and a flatter spoke feeds moment into the hub junction instead of storing it
in its own bending. The thickness genes push the *other* way — a thicker root raises the hub
share — which is why §14's instinct about thickness looked wrong when it was right in sign and
simply swamped.

Design × mesh, five rungs:

| genome | smoke | coarse | medium | fine | ultra | drift |
|---|---|---|---|---|---|---|
| shipped | 0.0392 | 0.0417 | 0.0433 | 0.0453 | 0.0463 | **+18.3%** |
| ga_beam | 0.0139 | 0.0138 | 0.0139 | 0.0141 | 0.0143 | +2.4% |

## THE FIRST QUESTION, AND IT MAY CLOSE THE ARC FOR FREE

**Is a high hub compliance share actually bad?** Nothing in this project has established that
it is. It is a share of a partition that sums to 1 — the hub holding 4.6% instead of 1.4%
means the spokes and rim hold correspondingly less, and the wheel's *deflection* and *stress*
targets are both already constrained directly. A bound on an energy share is a proxy, and the
tree has been sharp elsewhere about proxies that outlive their justification (§28, §29, §31).

**So step 0 is not "add the term". It is "find out whether the term would buy anything".**

If the answer is that the share is a diagnostic rather than a constraint, this arc closes by
recording that, and §31's `xfail` becomes a characterisation pin rather than a deficit — which
is a *better* outcome than adding a term.

**And check `FILLET_PLAN.md` first.** The shipped genome's share does not converge (0.0392 →
0.0463, still climbing at the finest rung) while the ga_beam design's does. The unfilleted
re-entrant corner is the prime suspect, and if the fillet arc lands first this number may move
on its own. Do not build an objective term on a quantity that is not converged.

## THE PLAN

### Step 0 — does the share predict anything the objective does not already see?

Cheap, and it decides whether the rest of the arc is worth running. Over the elite pool
(`stage3_prod_elite9.json`, `stage3_prod_elite10.json`, `stage2_elites.json`):

- correlate hub share against **axle drop** and against **utilisation**. If it is strongly
  correlated with quantities the objective already constrains, a term on it is redundant.
- correlate it against anything the objective *cannot* see. That is the only case where it
  earns a place.

**Register what would justify a term BEFORE measuring**, per §14's standing rule.

### Step 1 — only if Step 0 justifies it: is the term reachable and cheap?

`cy4` is gene 7 and already in the box, so no gene-space change is needed. The term would go
in `wheel_objective`. Two things to check before writing it:

1. **Is it differentiable through the existing adjoint?** `wheel_adjoint` computes the
   gradient; a compliance-split term is an energy ratio and needs to survive `jax.grad`.
   `study_gradient.py`'s finite-difference check is the gate.
2. **What does it cost the other targets?** `cy4` at the ga_beam value takes the hub share
   under the bound, but it is a *large* move in a gene the deflection target also depends on.
   Price the trade before proposing it — a term that buys 3% of compliance share for grams or
   for deflection margin is not obviously worth having.

### Step 2 — write it up and put the trade in front of the user with numbers

Same discipline as §31: measure, then hand over. If the term costs mass or deflection margin,
that is a Pareto choice and it belongs on PLAN.md's "The decision that is a human's" list.

## What must NOT happen

- **The `< 0.03` bound is not moved.** Settled in §31 with the ga_beam control as the
  evidence; this arc is about the design, not the threshold.
- **No objective term is added before Step 0 shows it buys something.** Adding a term to make
  a red test green is exactly the move this tree keeps refusing.
- **Do not build on a non-converged quantity.** Check `FILLET_PLAN.md`'s status first.
- **`best_solution.json` is not re-descended and promoted inside this arc.**

---

## RE-MEASURED ON THE FAITHFUL MESH — 2026-08-19. THE PREMISE SURVIVES.

PLAN §38 flipped `wheel_wheel`'s `uncap` default to `(True, 1.0)`, which replaced the half
end cap at the spoke/ring junctions with the far flank's own continuation. That moved hub
compliance **−17.9% at `coarse` and −18.7% at `medium`** — the same junction this arc's
suspect lives at, and the same order as the drift the arc calls disqualifying. So the
obvious hypothesis was that the end cap WAS the non-convergence.

**It is not.** `make reds-hub` re-run on the shipped default (45.3 s,
`studies/study_reds_hub_share_UNCAPPED.json`), against the capped table above:

| rung | capped | faithful | change |
|---|---|---|---|
| smoke | 0.0392 | 0.0325 | −17.1% |
| coarse | 0.0417 | 0.0342 | −18.0% |
| medium | 0.0433 | 0.0352 | −18.7% |
| fine | 0.0453 | 0.0365 | −19.4% |
| ultra | 0.0463 | **0.0371** | −19.9% |
| **drift smoke→ultra** | **+18.3%** | **+14.05%** | |

The artefact was a **near-constant multiplicative offset of ~18–20%, not the drift**. The
share still climbs monotonically across all five rungs and is *still climbing at `ultra`*.
Removing ~19% of the level removed only 4.3 points of the 18.3% drift, which is what a
constant offset does to a ratio and no more.

**So "do not build on a non-converged quantity" stands, and `FILLET_PLAN.md` is still the
gate on this arc.** The unfilleted re-entrant corner remains the prime suspect precisely
because the one OTHER junction artefact has now been removed and the drift barely moved.
That is a stronger reason to wait for the fillet arc than the arc had before, not a weaker
one: the field of suspects narrowed by one and the symptom did not.

### What DID change, and what it does not change

- **The gate deficit shrank without closing.** `hub < 0.03`: over by 39% (capped `coarse`)
  → **14.0%** (faithful `coarse`); but **23.5% at `ultra`**, because the drift is intact.
  `tests/test_wheel_fea.py::test_the_hub_junction_holds_under_three_percent_of_the_compliance`
  stays an `xfail` and still fails — which, under `xfail_strict = true`, is the outcome that
  keeps the suite green. **Nothing here reopens the `< 0.03` bound.**
- **§14's hypothesis is killed harder than before.** The `R_hub` sweep is now *bit-identical*
  — 0.0342 at every one of the 14 sample points across the whole 0.4–4.0 box, feasible and
  infeasible alike. The plan above says "structurally impossible, fillets are not meshed";
  the faithful mesh shows it to the last digit. (The driver still prints the canned line
  "the hub share FALLS as `R_hub` falls", which is false when the column is constant. Cosmetic,
  in `studies/study_reds_hub_share.py`, and deliberately not touched here.)
- **The attribution re-ranked, and `cy1`/`cy2` swapped.** One-at-a-time swaps at `coarse`:

  | gene | capped closes | faithful closes |
  |---|---|---|
  | `cy4` | 102.4% | **114.9%** |
  | `cy3` | 70.9% | 81.1% |
  | `cy2` | 57.9% | **65.5%** |
  | `cy1` | **59.7%** | 46.7% |
  | `t0` | −51.9% | −46.7% |

  `cy4` alone still overshoots the bound, so Step 1's "no gene-space change is needed"
  holds. But **the capped table's ordering was not safe to quote**: `cy1` lost 13 points and
  fell below `cy2`. Any future statement of the form "the second/third strongest lever" has
  to be re-read off the faithful table.

**Step 0 is still not started, and this does not start it.** This measurement only refreshes
the inputs Step 0 would use and re-checks the prohibition that gates the whole arc. The
design question — whether a high hub share is actually bad — is untouched and still open.
