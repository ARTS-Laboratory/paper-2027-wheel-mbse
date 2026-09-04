# EXPORTPREC_PLAN.md — make the exporter write the overlap at 4 dp

**Open arc #7. Created 2026-08-16, carried forward from PLAN §28. Small, but it touches a
shipped artifact. Nothing started.**

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

§28 fixed `test_the_bite_is_the_volume_divided_by_the_right_thickness` by deriving its
tolerance instead of hard-coding a stale constant. It recorded, in the same breath, that this
was **not** the better fix:

> **The alternative fix, considered and rejected for now:** make the *exporter* write the
> overlap to 4 dp instead of 2. That attacks the root cause — **the artifact discards precision
> in a value that gates buildability** — and would let the tolerance stay a tight constant (the
> budget falls to 5.1e-5 for any plausible t). It is the better long-term answer.

It was deferred for a specific, still-valid reason:

> It is **not** being done in this session because it changes a shipped artifact's contents
> immediately after a promotion, and the 2-dp volumes are quoted in §24's fillet-price tables
> and the DEFECT8 records. Filed as a successor.

**Both halves of that still hold**, and the second half is the whole difficulty: this is a
one-line format change with a documentation blast radius.

## What the deferral bought, and its limit

§28's derived bound is not weak in the meantime:

> "the derived bound retains ample power: it is exceeded by any t0/t3 ratio error above ~0.1%,
> against the 6.0% a full swap produces."

So there is **no urgency**. This arc is about removing a latent precision loss in a
buildability-gating value, not about fixing something currently broken. Rank it accordingly.

§28 also filed a related gap in the same section — a check "that can only see their ratio",
**filed as a successor, not fixed**. Read §28 in full before starting; these two may be one
piece of work.

## THE COMPLICATION — the deferral reason is still live

The 2-dp volumes are **quoted in prose and tables** that this arc cannot silently invalidate:

- **§24's fillet-price tables**, and
- the **DEFECT8 records** (now in PLAN §23 and §26 — `DEFECT8_PLAN.md` was deleted 2026-08-16,
  see PLAN.md's header table).

If the exporter starts writing 4 dp, every one of those quoted numbers is still *correct for
the artifact it described*, but a reader re-running the export will get a different string and
have no way to tell whether the number moved or the format did. **That is the §26/§27 stale
banner failure mode**, which this tree has now recorded three times.

## THE PLAN

### Step 0 — establish that nothing depends on the 2-dp string

Grep both ways, per this project's standing habit:

1. What **reads** the exported overlap value? `tests/test_export_contract.py`,
   `tests/test_promotion.py`, and anything in `studies/` that parses the manifest.
2. What **quotes** it in prose? PLAN.md §23, §24, §26, §28.

A parser that assumes 2 dp is a functional break; a table that quotes 2 dp is a documentation
break. They need different treatment and must not be conflated.

### Step 1 — change the format, and change the tolerance with it

The point of the arc is that the two move together:

- exporter writes the overlap at 4 dp,
- `test_the_bite_is_the_volume_divided_by_the_right_thickness`'s tolerance becomes the tight
  constant §28 costed at **5.1e-5**, replacing §28's derived-but-looser bound.

**If only the first half is done, the arc has achieved nothing** — the precision is recovered
and nothing uses it.

### Step 2 — re-export and reconcile the quoted tables

Re-run `make export`. For every number found in Step 0's second list, either confirm it is
unchanged at 4 dp or annotate it with the format change and the date. **Do not edit historical
tables to the new precision** — the tree's convention (§13, §26, the deleted-files header) is
that historical numbers are kept as written and scoped to their date.

### Step 3 — the promotion checklist

`tests/test_promotion.py` exists because a promotion is never a one-file change, and this arc
changes what a promotion writes. Check whether the contract tests need a clause for the format
itself, so a future revert to 2 dp is caught.

## What must NOT happen

- **Do not change the format without tightening the tolerance.** That is the half-fix.
- **Do not rewrite historical quoted numbers to the new precision.**
- **Do not do this immediately after a promotion.** §28 deferred it for that reason and the
  reason is structural, not circumstantial — the shipped genome is `09e8188` (§26) and if a
  new promotion is in flight, this waits.

---

## PREMISE CHECKED AGAINST THE FILLET SWITCH — 2026-09-03. **INTACT, AND THIS ARC IS NOW BLOCKED BY ITS OWN RULE.**

PLAN.md §106.  The exporter is OCC-side and does not touch the FEA mesh, so §103 leaves
this arc's measurements alone: §28's derived tolerance, the 5.1e-5 budget, and the 2-dp
volumes in `export/wheel_step_manifest.json` (`solid 39224.5`, `nofillet 36145.8`,
`fillets 3078.77`, last exported 2026-08-15) all stand.

**WHAT CHANGED IS THE BLOCKING CONDITION.**  This file's own "What must NOT happen" reads
*"Do not do this immediately after a promotion ... if a new promotion is in flight, this
waits."*  A promotion **is** in flight: §103 and §104 both rank *"RE-RUN STAGE 3 AND
RE-PROMOTE"* as successor 1, and §103's finding is that the shipped genome now reads over
the stress wall, so that re-promotion is not optional.  This arc moves from *"no urgency,
rank it accordingly"* to **explicitly blocked until successor 1 lands** — a status its
ranking does not currently carry.
