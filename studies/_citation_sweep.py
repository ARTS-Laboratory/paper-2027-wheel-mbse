#!/usr/bin/env python3
"""Report every line citation in the tree's `.md` and `.py`, and whether it still holds.

PLAN.md §138's successor 2 asked for this and gave the reason: the bare-`:N` ownership
rule "should be a script, not a habit".  §136 §6 stated the rule, §138 §4 extended it
across line breaks, and both times it was applied by hand under time pressure — which is
exactly when the per-line version silently returns a third of the answer.  §155's
successor 1 named the half §138 did not describe (the symbol pass) and its successor 2
named the check that would have caught three too-narrow patterns in a row (compare the
pattern's hit count against the module's mention count).

**NOT A TEST, AND FILING IT AS ONE IS THE TRAP** (§138 successor 2).  It is red by
construction for anything a section has chosen to leave — and §138's own example holds:
all six of the deliberate dangles §135 records at `PLAN.md:21239-21240` are in the report.
Its output is a report a human reads.  It writes nothing: `tests/test_cli.py:81`'s rule
about not writing into the repo applies to audits too.

WHAT COUNTS AS A CITATION, MEASURED RATHER THAN GUESSED.  Every widening of this pattern
so far was discovered by the sweep it was too narrow for — §136 lost one to a missing
`(\\.py)?`, §154 wrote that optional group into the successor and lost two to a missing
`[A-Za-z0-9_.]*`, and §140's symbol rule, already established, was not carried into
either.  So the numeric forms here come from counting the tree instead: of 1810 `:N`
tokens, 1660 are single lines, 126 are `-` ranges, 14 are `--` ranges and 8 are comma
lists (`wheel_adjoint:588,643`, the citation §138 §4 found the `.py` optional hiding).
The first integer is the anchor; the rest are reported but not resolved.

HOW AN OWNER IS FOUND — §140's RULE, MECHANISED.  The owner of a `:N` is whatever
IDENTIFIES a file, never the nearest path: a tracked path, a basename, a module name, the
head of a dotted symbol path (`wheel_fem.wheel_contact_problem:1717`), or a symbol with
exactly ONE definition in the tree (`RigidGroundContact:652`).  Both of those last two are
citations into `wheel_fem.py` that a `wheel_fem`-shaped pattern cannot match, and they are
2 of that file's 13.

THE SYMBOL PASS COSTS ONE LOOP AND YIELDS FOUR CITATIONS IN THE WHOLE TREE.  §155's
successor 1 estimated the shape from one file — "55 symbols, one hit ... exactly the shape
that gets skipped by hand and should not be" — and over all 104 tracked files the ratio is
starker: 1347 top-level symbols have exactly one definition, and sweeping every one of them
finds 4 citations, all of the SAME symbol (`RigidGroundContact`), all into `wheel_fem.py`,
all four already known to §155.  Building the whole owner index — 104 basenames, 104 module
stems and the 1347-symbol table — takes 12 ms, and enumerating all 730 citations 0.238 s; the
2.3 s is almost entirely the `git` calls that resolve them.  A yield that low is worth having
for 12 ms, and is also why no hand sweep will ever run it.

A CARRIED OWNER IS WEAKER THAN AN ATTACHED ONE, AND `smoothness` IS THE EVIDENCE.  An
identifier written onto the token itself may be a symbol; an owner CARRIED across a
paragraph to a bare `:N` may not.  `wheel_fea.py:663` reads "``smoothness`` was only ever
an indirect proxy for this (``:596-598`` said so)" — and `smoothness` has exactly one
top-level definition in this tree, in `studies/study_fillet_pnorm.py`, which is not the
file that sentence is about.  Admitting symbols as carried owners would resolve that
citation against the wrong file and report a confident dangle.  So carrying accepts only
file-identifying mentions, and in a source file a bare `:N` with no such mention in its
paragraph falls back to the citing file itself.  That fallback buys the right KIND of owner
and nothing more: this very row comes back MOVED against `wheel_fea.py` — at f0a9e83, the
commit that wrote the sentence, `:596` was a `# ---` divider — which is a question for a
human either way, but now it is the right question.

IN CODE, ONLY A TOKEN THAT NAMES A FILE STANDS WITHOUT BACKTICKS, AND THAT IS ALSO A
MEASUREMENT.  `studies/study_fillet_block.py` contains 102 bare `:N` tokens and not one is
a citation — they are slices, format specs and dict literals.  Backticks separate the two
perfectly here: 8 backticked bare tokens exist across all `.py` files and all 8 are real
citations, including `wheel_objective.py:669-670`, where one owner serves four line numbers
across a line break.  The same rule has to cover symbols, for the same reason one level up:
`wheel_fea.py:1365` prints `f"{max_stress:7.2f} MPa"`, and `max_stress` is a top-level `def`
in `wheel_adjoint.py:573` — a format spec that the symbol rule reads as a citation into
another module.  A path may stand bare (`stage3_resume_genome.py:15` writes
`(wheel_stage3.py:980-982)`); a symbol or a carried owner may not.

RESOLUTION USES `git blame`, NOT `git log -S`, AND THE TWO DISAGREE ON THREE OF NINE.
§138 §1's method — `git log -S'<the citing sentence>' -- <citing file>` — is one call per
citation and needs a hand-chosen needle.  Blaming the citing line is one call per citing
FILE (0.55 s for all 24188 lines of `PLAN.md`) and needs nothing.  Replaying §155 §2's
nine unrepaired citations, blame agrees on six and differs on three, in both directions
and for three distinct reasons:

  citing site               log -S    blame     why they differ
  wheel_stage3.py:72        506acfe   4ec1d91   `log -S -- <path>` cannot see past the
                                                `src/` reorg; blame follows the rename
  MBSE_PLAN.md:208          0b8890a   de67144   the file was untracked until de67144
  PLAN.md:1904              43da58f   4e4a672   `log -S` finds first appearance, blame
                                                finds last touch: §8 was rewritten a day
                                                after the citation was written

All nine resolve identically either way, and blame's extra reach is worth having: at
4ec1d91 — two days EARLIER than the commit §155 §3 could see — `wheel_fem.py:1841` is still
`def solve_wheel_contact(`, which is the anchor §155 §3 restored.

**LAST TOUCH IS NOT A COMPROMISE, IT IS THE RIGHT SEMANTICS, AND THE BLIND SPOT IT LEAVES
IS 8 ROWS OF 730.**  A REPAIR re-dates an anchor: when `277a731` re-pointed
`MBSE_PLAN.md:66` from `wheel_objective.py:1129` to `:1151`, the claim that must hold is
the repaired one, and resolving at the line's first appearance would invent a MOVED.  What
blame cannot see is the other case — a citation carried unchanged through a re-wrap, where
the wrap is too new and real drift reads as OK.  `git log -L <n>,<n>:<file> -p` separates
them for 0.4 s a line, and over all 730:

  594  the citing line has ONE commit in its lineage -- blame IS the origin
  136  the line was edited after it first appeared
        32   both readings give the same verdict
        96   they differ AND the newest commit WROTE the anchor -- blame is right; 72 of
             the 104 flips are three commits whose job WAS re-pointing citations (57 at
             277a731, 12 at c51320e, 3 at 480016a, which is §155's repair this morning)
         8   they differ and the newest commit left the anchor alone -- the only rows
             where this instrument may be judging against a version too new

So the count below is short by at most 8, 1.1%, and the 8 are nameable rather than
estimated.  They are not noise either: two of them cite `tests/test_objective.py:1257`,
the dangle §138 §6 recorded as "4 name :1257, dangling and recorded so at §118" — of those
four sites blame already reports two as MOVED, and these are the two it hides.

THE SCOPE IS `.md` AND `.py`, AND `REPO_EXPLAINED.tex` IS DELIBERATELY OUTSIDE IT.  That
document writes a citation as `\\at{wheel\\_wheel.py}{173}`, and the macro
`\\newcommand{\\at}[2]{\\code{#1:#2}}` puts the colon in at render time — so its 83 `\\at{}`
citations contain no `:N` for any pattern to find, and its bare ones sit under a
`\\code{module.symbol}` owner rather than a backticked one.  §139 counted 145 line citations
there and §140 measured 55 of them moved; this sweep finds ZERO, which is why the file is
named in `NO_SCAN` rather than left to
report a silent nothing.  It stays in `SCOPE`, so it can still own a citation and still
counts as a mention.  730 is the count for `.md` and `.py`, not for the tree.

WHAT THE TWO SCANNED SUFFIXES SAY, THE FIRST TIME ANYONE HAS ASKED THEM ALL AT ONCE:
**730 citations, 527 still pointing at what they pointed at when they were written, 203 for
a human**, in 2.3 seconds.  455 of the 730 carry a path, 4 a symbol, 271 a carried owner —
and that split is the report's own error bar, because a path cannot name the wrong file
while a carried owner can.  The 203 divide along exactly that line:

  121   MOVED, owner is a path      drift, and the owner is not in question
   55   MOVED, owner was carried    drift, if the paragraph's owner is the right one
   27   anchor out of range         the OWNER is wrong: all 27 are a self-citation

**All 27 out-of-range rows are one failure, not twenty-seven.**  Each is a bare `:N` naming
a line of the file it is written in, inside a paragraph whose subject is a different file —
§135's six deliberate dangles are six of them, `PLAN.md:21239-21240` citing PLAN.md's own
§118 and §133 in a paragraph about `tests/test_objective.py`.  So §140's carried-owner rule
is wrong for 27 of the 271 citations that rely on it, 10.0%, in a single recognisable shape.
The obvious repair — prefer the citing file when the carried owner cannot hold the line — is
NOT applied here, because it would silently convert a genuine far-gone anchor into a
confident resolution against the wrong file, and this instrument has no way to tell those
apart.  The row says which two readings are available and stops.

TWO PROPERTIES OF "SINCE IT WAS WRITTEN" THAT ARE WORTH STATING BEFORE ANYONE ACTS ON A ROW.
A quotation of an already-broken citation — §155 §4 found one at `PLAN.md:21904`, and §136's
diagnosis tables are full of them — resolves as OK, because the line it names was already
holding something else when the quotation was typed and has not moved since.  That is the
right answer and it is free: no list of exemptions is needed to keep repair tables out of
the report.  And the re-wrap case above runs the same way — it can only turn drift into an
OK, never the reverse.  **Both errors are one-directional: 203 is a floor, and the measured
ceiling is 211.**

Usage:
    python studies/_citation_sweep.py                    # every scanned .md and .py
    python studies/_citation_sweep.py --into src/wheel_fem.py
"""

import argparse
import os
import re
import subprocess
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCOPE = ("*.md", "*.py", "*.tex")     # what can OWN a citation, and the mention counts
NO_SCAN = (".tex",)                   # what is not searched FOR citations -- see docstring
PROSE = (".md", ".tex")

# The identifier is optional: absent, the token is a bare `:N` and needs a carried owner.
TOKEN = re.compile(r"(?<![A-Za-z0-9_.])([A-Za-z0-9_./]*[A-Za-z0-9_])?"
                   r":([0-9]+)((?:(?:--|[,-])[0-9]+)*)")
BACKTICKED = re.compile(r"`([^`]+)`")
TOPLEVEL = re.compile(r"^(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
BLAME_HDR = re.compile(r"^([0-9a-f]{40}) [0-9]+ ([0-9]+)(?: ([0-9]+))?$")
# A blank line, a bare `"""` or a `# ---` rule can never have held a claim, so a
# pre-image that matches HEAD on one is a coincidence and not a resolution.
ALNUM = re.compile(r"[A-Za-z0-9]")


def git(*args):
    r = subprocess.run(("git",) + args, cwd=ROOT, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _identifying(pairs):
    """name -> path, keeping only names that identify exactly ONE file."""
    seen = defaultdict(set)
    for name, path in pairs:
        seen[name].add(path)
    return {n: p.pop() for n, p in seen.items() if len(p) == 1}


class Owners:
    """§140's rule: whatever identifies a file, never the nearest path."""

    def __init__(self, files):
        self.files = set(files)
        self.by_base = _identifying((os.path.basename(f), f) for f in files)
        self.by_stem = _identifying((os.path.splitext(os.path.basename(f))[0], f)
                                    for f in files)
        symbols = []
        for f in files:
            if f.endswith(".py"):
                for line in read(f).splitlines():
                    m = TOPLEVEL.match(line)
                    if m:
                        symbols.append((m.group(1), f))
        self.by_symbol = _identifying(symbols)

    def names_file(self, ident):
        """A path, a basename, a module name, or a dotted path's head."""
        if ident in self.files:
            return ident
        for table in (self.by_base, self.by_stem):
            if ident in table:
                return table[ident]
        head = ident.split(".")[0]
        if head != ident:
            return self.by_base.get(head) or self.by_stem.get(head)
        return None

    def owner(self, ident):
        """Everything `names_file` accepts, plus a symbol with one definition."""
        return (self.names_file(ident)
                or self.by_symbol.get(ident)
                or self.by_symbol.get(ident.split(".")[-1]))


def citations(path, owners):
    """Yield (line, owner, anchor, extra, how) for every citation in one file.

    `how` records what supplied the owner, because the three are not equally strong:
    a `path` cannot be wrong, a `symbol` can name the wrong file, and a `carried` owner
    can be the wrong file entirely -- see the docstring on `smoothness` and on
    `PLAN.md:21946`, whose bare `:21244` names `PLAN.md` and inherits `wheel_adjoint`.
    """
    carried = None if path.endswith(PROSE) else path
    for lineno, line in enumerate(read(path).splitlines(), 1):
        if not line.strip():
            carried = None if path.endswith(PROSE) else path
            continue
        events = [(m.start(), 0, m) for m in TOKEN.finditer(line)]
        for m in BACKTICKED.finditer(line):
            named = owners.names_file(m.group(1).strip())
            if named:
                events.append((m.start(), 1, named))
        for _, kind, payload in sorted(events, key=lambda e: (e[0], e[1])):
            if kind == 1:
                carried = payload
                continue
            ident, anchor, extra = payload.group(1), int(payload.group(2)), payload.group(3)
            named = owners.names_file(ident) if ident else None
            owner = named or (owners.owner(ident) if ident else carried)
            if owner is None:
                continue
            # In code, only a token that NAMES a file stands without backticks: a bare
            # `:N` is a slice or a format spec, and a symbol is a variable.
            quoted = payload.start() > 0 and line[payload.start() - 1] == "`"
            if named is None and not quoted and not path.endswith(PROSE):
                continue
            yield lineno, owner, anchor, extra, ("path" if named else
                                                 "symbol" if ident else "carried")
            if named:
                carried = named


def blame(path, cache={}):
    """line -> the commit that last touched it.  One call per citing file."""
    if path not in cache:
        at = {}
        for line in (git("blame", "--porcelain", "--", path) or "").splitlines():
            m = BLAME_HDR.match(line)
            if m:
                for i in range(int(m.group(2)), int(m.group(2)) + int(m.group(3) or 1)):
                    at[i] = m.group(1)[:7]
        cache[path] = at
    return cache[path]


def blob(commit, path, cache={}):
    """The cited file's lines at `commit`, falling back through the `src/` reorg."""
    if (commit, path) not in cache:
        text = git("show", "%s:%s" % (commit, path))
        if text is None:
            text = git("show", "%s:%s" % (commit, os.path.basename(path)))
        cache[(commit, path)] = text.splitlines() if text is not None else None
    return cache[(commit, path)]


def line_at(commit, path, n):
    lines = blob(commit, path)
    return lines[n - 1] if lines and 1 <= n <= len(lines) else None


def resolve(citing, citing_line, cited, anchor):
    """(status, commit, detail) -- MOVED is a candidate for a human, not a verdict.

    The claim is exactly this and no more: the cited line holds something different at HEAD
    than it held at the citing line's own commit.  A citation that DRIFTED and one that was
    WRONG THE DAY IT WAS TYPED both report MOVED, which is why `then` is printed.  14 of the
    176 name a blank line or a bare `# ---` rule and so can never have held the claim at all
    -- `wheel_fea.py:596-598`, cited from four separate sites, is the example.  A delta
    applied to an anchor that never held moves a wrong citation to a different wrong line.

    AND `then` IS READ ONE TREE TOO LATE WHENEVER THE CITING COMMIT ALSO EDITED THE CITED
    FILE -- 36 of 159 MOVED rows, 22.6%, measured.  An author writes a sentence against the
    tree they are READING, which is the citing commit's pre-image; this resolves the cited
    file at its post-image.  The two agree unless that same commit moved the anchor, and
    where they disagree the pre-image is the better `then`: `PLAN.md:3196` cites
    `wheel_stage3.py:384` for `fidelity_check_every` and `:272` for `_fidelity_check`, and at
    `b5c22c9^` those two lines are exactly that prose and exactly that `def`, while at
    `b5c22c9` they are a pooling comment and an unrelated docstring.  So `(pre: ...)` is
    appended wherever the readings differ, and a repairer reads THAT column.

    **A PRE-IMAGE THAT EQUALS HEAD SETTLES THE ROW, AND THAT IS SOUND RATHER THAN HEURISTIC**
    -- the citation then names at HEAD exactly the text its author was looking at, so it
    holds and the row is not for a human at all (`pre-ok`).  It is printed rather than
    dropped, because a count that silently absorbed it could not be audited.  THREE SUCH ROWS
    EXISTED THIS MORNING and were repaired by hand at §157 before this check was written;
    today the tree has **0**, plus the blank-on-blank coincidences the content guard rejects
    -- `ALNUM`, and without it §156 §3's 14 never-held rows come
    back as rescues.  The OPPOSITE reading is what must not be taken: 67 rows the report
    calls ok would turn MOVED under a blanket pre-image, 57 of them at `277a731` alone, whose
    job was re-pointing citations onto lines the SAME commit moved.  A repair re-dates an
    anchor (§156 §2), so the post-image is right for a repair commit and wrong only for a
    commit that moved the anchor incidentally.  Never downgrade an ok row; only upgrade a
    MOVED one.

    The price is one extra `git show <commit>^:<cited>` per MOVED row and nothing for the
    other 660: **0.17 s** on the resolve pass, 1.90 s -> 2.07 s.  §156 successor 2's
    `git log -L` pass buys a 1.1% correction for 2 min 28 s, 65x; this buys a 22.6% one for
    9%, which is why it is folded in here and that one is not.
    """
    commit = blame(citing).get(citing_line)
    if commit is None:
        return "unknown", "-", "no blame for the citing line"
    then, now = line_at(commit, cited, anchor), line_at("HEAD", cited, anchor)
    if then is None:
        # Out of range falsifies the OWNER as readily as the anchor, and §135's six
        # deliberate dangles are all of the first kind: bare `:N` naming PLAN.md's own
        # lines, in a paragraph about a test file.  Say which, and reassign nothing.
        here = blob("HEAD", citing)
        also = (" -- but %s:%d exists" % (citing, anchor)
                if citing != cited and here and anchor <= len(here) else "")
        return "unknown", commit, "no line %d in %s at %s%s" % (anchor, cited, commit, also)
    if then == now:
        return "ok", commit, then.strip()[:60]
    was = line_at(commit + "^", cited, anchor)
    if was == now and ALNUM.search(now):
        return "pre-ok", commit, "%-28s written against %s^" % (now.strip()[:28], commit)
    hits = [i for i, l in enumerate(blob("HEAD", cited) or [], 1) if l == then]
    where = "-> :%d" % hits[0] if len(hits) == 1 else "%d matches at HEAD" % len(hits)
    reading = "%-28s %s" % (then.strip()[:28], where)
    if was is not None and was != then and ALNUM.search(was):
        reading += "  (pre: %s)" % was.strip()[:40]
    return "MOVED", commit, reading


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--into", help="report only citations into this file")
    args = ap.parse_args()

    files = sorted((git("ls-files", *SCOPE) or "").split())
    owners = Owners(files)
    if args.into:
        args.into = owners.names_file(args.into)
        if args.into is None:
            raise SystemExit("--into names no tracked file in %r" % (SCOPE,))
    found = defaultdict(list)
    for f in files:
        if f.endswith(NO_SCAN):
            continue
        for lineno, cited, anchor, extra, how in citations(f, owners):
            if args.into in (None, cited):
                found[cited].append((f, lineno, anchor, extra, how))

    # The mention count is §155 successor 2's check: a pattern that finds 7 citations
    # where the module is named in 29 places is a pattern worth re-reading once.
    print("CITED FILE                          cites   path symbol carried   mentions")
    for cited in sorted(found, key=lambda c: -len(found[c])):
        rows = [r[4] for r in found[cited]]
        counts = (git("grep", "-c", os.path.splitext(os.path.basename(cited))[0],
                      "--", *SCOPE) or "").splitlines()
        mentions = sum(int(c.rsplit(":", 1)[1]) for c in counts)
        print("%-36s %5d %6d %6d %7d %10d"
              % (cited, len(rows), rows.count("path"), rows.count("symbol"),
                 rows.count("carried"), mentions))

    print("\nCITING SITE                    CITED                            OWNER    THEN "
          "(at the citing line's commit)")
    n_ok = n_pre = 0
    for cited in sorted(found):
        for f, lineno, anchor, extra, how in sorted(found[cited]):
            status, commit, detail = resolve(f, lineno, cited, anchor)
            if status == "ok":
                n_ok += 1
                continue
            n_pre += status == "pre-ok"
            print("%-30s %-32s %-8s %-7s %s %s"
                  % ("%s:%d" % (f, lineno), "%s:%d%s" % (cited, anchor, extra),
                     how, status, commit, detail))
    total = sum(len(v) for v in found.values())
    print("\n%d citations, %d resolve at their citing line's commit, %d more against its "
          "pre-image, %d for a human." % (total, n_ok, n_pre, total - n_ok - n_pre))


if __name__ == "__main__":
    main()
