# How to work in this repository

Two things live here and nothing else: where the project's own rules are written down, and
the behavioural guidelines that apply to every turn.

## Where the rules are

**`PLAN.md`'s header block is the single source for version control** — anonymous authorship,
one commit per finished unit of work, green before commit, branches, message shape. It says so
itself, and the nine sibling plan files carry a pointer rather than a second copy. Do not
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
