# How to work in this repository

Two things live here and nothing else: where the project's own rules are written down, and
the behavioural guidelines that apply to every turn.

## Where the rules are

**`PLAN.md`'s header block is the single source for version control** — anonymous authorship,
one commit per finished unit of work, green before commit, branches, message shape. It says so
itself, and the eleven sibling plan files carry a pointer rather than a second copy. Do not
restate those rules anywhere, including here.

The numbered sections of `PLAN.md` are the project record. Six closed arc files were deleted on
2026-08-16 and about seventy comments still cite them by name; a dangling `see SVK_PLAN.md
step N` is a pointer to a numbered section, not a missing file to restore.

---

## Karpathy guidelines

Behavioural guidelines to reduce common LLM coding mistakes, from
[Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM
coding pitfalls (MIT-licensed, originally the `karpathy-guidelines` skill — moved here on
2026-08-29 so they apply without being invoked).

**Tradeoff: these bias toward caution over speed. For trivial tasks, use judgment.**

### 1. Think before coding

*Don't assume. Don't hide confusion. Surface tradeoffs.*

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what is confusing. Ask.

### 2. Simplicity first

*Minimum code that solves the problem. Nothing speculative.*

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or configurability that was not requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask: would a senior engineer call this overcomplicated? If yes, simplify.

**THIS ONE CUTS AGAINST THE HOUSE STYLE AND IS MEANT TO.** This tree documents heavily on
purpose — a docstring here carries the measurement that justifies a constant, and deleting it
loses evidence, not verbosity. The rule applies to CODE: control flow, abstractions,
parameters, branches. It is not a licence to thin the prose that records why a number is what
it is.

### 3. Surgical changes

*Touch only what you must. Clean up only your own mess.*

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor what is not broken.
- Match existing style even where you would do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports, variables and functions that YOUR change orphaned; leave pre-existing dead
  code alone unless asked.

The test: every changed line traces directly to the request.

### 4. Goal-driven execution

*Define success criteria. Loop until verified.*

Turn tasks into verifiable goals — "add validation" becomes "write tests for invalid inputs,
then make them pass"; "fix the bug" becomes "write a test that reproduces it, then make it
pass"; "refactor X" becomes "tests pass before and after". For multi-step work, state the plan
as steps each with its own check.

Strong criteria let the work loop independently. Weak criteria ("make it work") force constant
clarification.

---

## How a section states a rule

Added 2026-09-16 (PLAN.md §182 §8), and earned: §180 and §181 were each amended after audit to
scope a headline their own bodies had already scoped, and the audit that prompted the second
amendment then turned out to have made the same mistake in the other direction. Four instances
in two days, all of one shape — **a generalisation stated from runs that differ in more than
one way.**

> **A section may assert a rule about code it changed. A rule about behaviour it merely
> observed must name the design that could have refuted it — and where the runs behind it
> differ in more than one way, it is a hypothesis and must be written as one.**

§167's headline rule is right because the section wrote the refusal and its mutants prove it
fires. "The creep stops", "bounded in practice" and "`medium` is set once, `coarse` creeps" are
wrong because nothing in those sections could have come back and said otherwise.

Three checks, each a minute's work:

1. **Confound check.** Before writing any "X does / does not Y", list the runs it rests on and
   the ways they differ. More than one plausible cause makes it a hypothesis; write it as one.
2. **Instrument check.** A cross-run comparison names the instrument behind each figure, and
   **an instrument that reports only exceedances is silent about everything below** — its
   silence is not data. A high-water mark says "nothing beat the previous high", never "the
   quantity is steady"; a threshold alarm says "this operation passed 120 s", never how many
   did not. So a running maximum is never compared against a series, and a threshold reporter
   is never read as a count.
3. **Headline test.** A headline may state a rule only if the section registered a falsifier
   the run could have tripped and did not. Check it against the section's own falsifier list.

This applies to audits and findings files too, which is where it was learned: **an audit's own
generalisations get a falsifier beside them.**
