"""The citing line's LINEAGE, which `_citation_sweep.py` does not read -- §160.

The sweep resolves a citation at `git blame`'s answer for the citing line, and §156 §2
measured the blind spot that leaves at 8 rows of 730 and named them.  This reads the
other instrument -- `git log -L <n>,<n>:<file> -p`, the citing line's whole history --
and answers two questions the sweep structurally cannot.

**1. WHICH COMMIT IS A CITATION'S ORIGIN?**  Not the one that created the LINE.  A line
can be repaired twice and carry more than one citation, and `PLAN.md` line 16795 is both:
created at `96a0ac5` naming `study_tri_block.py` line 228, re-pointed to line 242 by
`58b311a`, and the OTHER citation on it re-pointed by `277a731`.  §156 compared today's
line 242 against `96a0ac5`, where line 242 did not exist yet -- a tree predating the
claim, which manufactures a MOVED.  The origin is the newest commit whose diff put THIS
`:N` on the citing line: `anchor_origin` below.  Re-measured at `7a28091`, the blind spot
is **7 rows, not 8**, and nothing joined it on a tree 153 citations larger.

  702  the citing line has ONE commit in its lineage -- blame IS the anchor-origin
  163  the newest commit WROTE this `:N` -- blame is the anchor-origin anyway
   11  blame is NOT the anchor-origin, and both readings agree
    7  blame is NOT the anchor-origin AND the verdict flips  <- the blind spot

All 7 flip one way -- blame resolves, the anchor-origin reads MOVED -- so the sweep's
"203 is a floor" survives.  Blame equals the newest `-L` commit on all 883 rows.

**AND IT CONVICTS `df1168b`.**  That change read `then` at the citing commit's post-image
where an author writes against the pre-image, and folded `(pre: ...)` in.  But both columns
are read at BLAME, and blame is the right commit only when it is the anchor-origin -- true
for 865 of 883, false for **18, 2.0%**.  `tests/test_fem.py` line 324 is what that costs a
repairer: its anchor was written at `8b347a0`, blame at `7a60002` was `f0a9e83`, so the
report handed out `then`@`f0a9e83` and `(pre:)`@`f0a9e83^` and the repair followed them
faithfully onto the wrong lines.

**2. DID A REPAIR LAND ON THE RIGHT LINE?**  A repair re-dates an anchor (§156 §2), so a
repair that lands WRONG reads `ok` from that day on and no re-run of the sweep can ever
see it -- the row is not in the 203 and never will be.  That is not drift the sweep
measures; it is an error the sweep issues a clean bill of health for.  `audit` asks of
every repair step whether the line it now names holds what the anchor it replaced named.

FOUR THINGS THAT LOOK LIKE A WRONG REPAIR AND ARE NOT, ALL MEASURED THE HARD WAY -- an
earlier draft of this file convicted §157's own `641ce0f` and `cdfc545` of 14 wrong
anchors and every one was this instrument's bug:

  * the chain ORIGIN is the wrong baseline -- a creation commit's pre-image is unrelated
    text, and `wheel_fea.py` line 750 at a commit predating the `src/` reorg "found" its claim
    113 lines away.  The baseline is the PREVIOUS STEP.
  * ONE reading is the wrong question -- §157 repaired 22 anchors from the pre-image.
  * an EDITED line is still the same line -- `wheel_fea.py` line 455 reads
    `np.clip(thicknesses, thickness_clip[0], ...)` where the claim read
    `np.clip(thicknesses, 0.5, 20.0)`.  `similar` sees that; `==` cannot.
  * a repair may RE-AIM a citation that was wrong the day it was typed --
    `wheel_geometry.py` line 264 said lines 750-752 was "exactly what `thicken_3taper_curve` does"
    while that `def` sat at line 781.

AND THE RESULT IS A FLOOR, FOR TWO REASONS IT REPORTS RATHER THAN HIDES.  Accepting either
image is its own blind spot: where the two differ and a repair matched one, the repairer
made a CHOICE only a reader of the citing sentence can check, and `tests/test_fem.py` line 324
is the row that choice was made wrongly on -- counted and listed, not absorbed.  And the
`CLAIM IS AT` column is a DETECTION, not a repair instruction: `PLAN.md` line 13773 is a dated
quotation the record marks "deliberately left alone", so the repair there was to restore
line 1153, not to follow the content to line 1225.

Cost: 2 min 35 s for 883 citations against the sweep's 2.3 s, which is why §156 declined to
fold it in and why this is a separate command rather than a flag on that one.  Underscore
first, so `studies/study_*.py` never collects it.

Usage:
    python studies/_citation_lineage.py              # both reports
    python studies/_citation_lineage.py --census     # the blind spot only
    python studies/_citation_lineage.py --repairs    # the repair audit only
"""

import argparse
import difflib
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _citation_sweep as cs                                      # noqa: E402

HEX40 = re.compile(r"^@@@([0-9a-f]{40})$")
SIMILAR_ENOUGH = 0.6          # an edited line vs a different line -- see the docstring
LONG_ENOUGH = 12              # a claim shorter than this matches too much to mean anything


def lineage(path, lineno, cache={}):
    """[(commit, pre-image lines, post-image lines)] for ONE line, newest first."""
    key = (path, lineno)
    if key in cache:
        return cache[key]
    out = cs.git("log", "-L", "%d,%d:%s" % (lineno, lineno, path), "--format=@@@%H") or ""
    blocks, cur = [], None
    for line in out.splitlines():
        m = HEX40.match(line)
        if m:
            cur = [m.group(1)[:7], [], []]
            blocks.append(cur)
        elif cur is None or line.startswith(("diff --git", "@@ ")):
            continue
        elif line in ("--- a/%s" % path, "--- /dev/null", "+++ b/%s" % path):
            continue
        elif line.startswith("-"):
            cur[1].append(line[1:])
        elif line.startswith("+"):
            cur[2].append(line[1:])
    cache[key] = blocks
    return blocks


def has_anchor(lines, cited, anchor, extra, how, owners):
    """Is THIS citation present in this version of the citing line?"""
    for text in lines:
        for m in cs.TOKEN.finditer(text):
            if int(m.group(2)) != anchor or m.group(3) != extra:
                continue
            if how != "path":
                return True
            if m.group(1) and owners.names_file(m.group(1)) == cited:
                return True
    return False


def anchor_origin(path, lineno, cited, anchor, extra, how, owners):
    """The newest commit whose diff PUT this `:N` on the line -- what blame owes us.

    LIMIT, measured on `tests/test_fem.py` line 324: this is read off `-L`'s tracked region, so
    a commit that RE-WRAPS the paragraph around a citation can be over-attributed as having
    written the anchor -- here it answers `f0a9e83` where the true write is `8b347a0`.
    """
    for commit, pre, post in lineage(path, lineno):
        if has_anchor(post, cited, anchor, extra, how, owners) and \
           not has_anchor(pre, cited, anchor, extra, how, owners):
            return commit
    return None


def verdict(cited, anchor, commit):
    """`resolve()`'s three-way answer at an ARBITRARY commit, pre-image and all."""
    then, now = cs.line_at(commit, cited, anchor), cs.line_at("HEAD", cited, anchor)
    if then is None:
        return "unknown"
    if then == now:
        return "ok"
    was = cs.line_at(commit + "^", cited, anchor)
    if was == now and cs.ALNUM.search(now):
        return "pre-ok"
    return "MOVED"


def resolves(v):
    return v in ("ok", "pre-ok")


def similar(a, b):
    """Is this the SAME line, edited, rather than a different line?"""
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio() >= SIMILAR_ENOUGH


def scan(owners=None):
    """Every citation in the tree, as the sweep finds them."""
    files = sorted((cs.git("ls-files", *cs.SCOPE) or "").split())
    owners = owners or cs.Owners(files)
    rows = []
    for f in files:
        if f.endswith(cs.NO_SCAN):
            continue
        for lineno, cited, anchor, extra, how in cs.citations(f, owners):
            rows.append((f, lineno, cited, anchor, extra, how))
    return owners, rows


def census(owners, rows):
    """§157 successor 1: where blame is not the anchor-origin, and where that matters."""
    buckets, blind, misplaced = defaultdict(int), [], []
    for f, lineno, cited, anchor, extra, how in rows:
        blamed = cs.blame(f).get(lineno)
        if blamed is None:
            buckets["no blame for the citing line"] += 1
            continue
        origin = anchor_origin(f, lineno, cited, anchor, extra, how, owners)
        site, target = "%s:%d" % (f, lineno), "%s:%d%s" % (cited, anchor, extra)
        if origin is None:
            buckets["the lineage never shows this `:N` arriving"] += 1
        elif len(lineage(f, lineno)) <= 1:
            buckets["ONE commit in the lineage -- blame IS the anchor-origin"] += 1
        elif origin == blamed:
            buckets["the newest commit WROTE this `:N` -- blame is it anyway"] += 1
        else:
            misplaced.append((site, target, blamed, origin))
            v_new, v_org = verdict(cited, anchor, blamed), verdict(cited, anchor, origin)
            if v_new == v_org:
                buckets["blame is NOT the anchor-origin, both readings agree"] += 1
            else:
                buckets["blame is NOT the anchor-origin AND the verdict flips"] += 1
                blind.append((site, target, how, "%s@%s" % (v_new, blamed),
                              "%s@%s" % (v_org, origin)))

    print("=== THE CITING LINE'S LINEAGE: %d citations ===\n" % len(rows))
    for k, n in sorted(buckets.items(), key=lambda e: -e[1]):
        print("  %5d  %s" % (n, k))
    print("\n  %d of %d rows have blame != anchor-origin, so the sweep reads BOTH `then`"
          % (len(misplaced), len(rows)))
    print("  and `(pre:)` at the wrong commit for them:")
    for site, target, blamed, origin in sorted(misplaced):
        print("    %-30s %-32s blame=%s  written at %s" % (site, target, blamed, origin))

    print("\n=== THE BLIND SPOT: %d ===" % len(blind))
    print("%-32s %-34s %-8s %-22s %s"
          % ("CITING SITE", "CITED", "OWNER", "AT BLAME", "AT anchor_origin"))
    for row in sorted(blind):
        print("%-32s %-34s %-8s %-22s %s" % row)


def _anchors_on(lines, cited, owners):
    out = []
    for text in lines:
        for m in cs.TOKEN.finditer(text):
            if m.group(1) and owners.names_file(m.group(1)) == cited:
                out.append((int(m.group(2)), m.group(3)))
    return out


def history(path, lineno, cited, owners):
    """[(commit, anchor, extra)] oldest first -- what this citation NAMED over time."""
    steps = []
    for commit, _pre, post in reversed(lineage(path, lineno)):
        found = _anchors_on(post, cited, owners)
        if len(found) > 1:
            return None                      # two anchors into one file: cannot tell apart
        if not found:
            continue                         # `-L`'s region drifted off the citation on a
            # re-wrap -- skip the version, keep the chain.  Bailing here cost 28 of 554.
        if not steps or steps[-1][1:] != found[0]:
            steps.append((commit,) + found[0])
    return steps


def claim_text(commit, cited, anchor):
    """BOTH readings of what the anchor named at the step that WROTE it."""
    out = []
    for t in (cs.line_at(commit, cited, anchor), cs.line_at(commit + "^", cited, anchor)):
        if t is not None and cs.ALNUM.search(t) and len(t.strip()) >= LONG_ENOUGH \
                and t not in out:
            out.append(t)
    return out


def symbol_span(commit, cited, name):
    """(first, last) line of a top-level `def`/`class` in the cited file at `commit`."""
    lines = cs.blob(commit, cited)
    if not lines:
        return None
    start = None
    for i, line in enumerate(lines, 1):
        m = cs.TOPLEVEL.match(line)
        if m and m.group(1) == name:
            start = i
        elif m and start is not None:
            return (start, i - 1)
    return (start, len(lines)) if start else None


def reaimed(citing, lineno, cited, a_old, a_new, commit, c_prev):
    """Did the repair move the anchor INTO a symbol the citing line names?"""
    text = (cs.blob("HEAD", citing) or [""] * lineno)[lineno - 1]
    for m in cs.BACKTICKED.finditer(text):
        name = m.group(1).strip().rstrip("()").split(".")[-1]
        span = symbol_span(commit, cited, name)
        if not span:
            continue
        was_in = symbol_span(c_prev, cited, name)
        if span[0] <= a_new <= span[1] and not (was_in and was_in[0] <= a_old <= was_in[1]):
            return name
    return None


def audit(owners, rows):
    """Every REPAIR step: does the line it now names hold what the anchor it replaced did?"""
    tally, wrong, choice, reaim = defaultdict(int), [], [], []
    seen = set()
    for f, lineno, cited, anchor, extra, how in rows:
        if how != "path" or (f, lineno, cited, anchor, extra) in seen:
            continue
        seen.add((f, lineno, cited, anchor, extra))
        steps = history(f, lineno, cited, owners)
        if steps is None:
            tally["untrackable"] += 1
            continue
        if len(steps) == 1:
            tally["never repaired"] += 1
            continue
        tally["repaired"] += 1
        broken = False
        for i in range(1, len(steps)):
            c_prev, a_old = steps[i - 1][0], steps[i - 1][1]
            commit, a_new = steps[i][0], steps[i][1]
            site, target = "%s:%d" % (f, lineno), "%s:%d" % (cited, a_new)
            live = i == len(steps) - 1
            # ONCE A CHAIN IS KNOWN BROKEN, NOTHING DOWNSTREAM OF IT IS SCOREABLE, AND
            # THE COUNT IS NOT A HEALTH METRIC WITHOUT THIS.  Every step is scored against
            # what its PREDECESSOR named, which equals "was this repair wrong" only if the
            # predecessor was right.  When a bad anchor is later fixed, the fix moves AWAY
            # from the bad anchor's content and scores as a wrong delta -- so repairing the
            # tree RAISES the count: `13 steps / 8 live` at 7a28091 became `22 / 10` at
            # eea048c once §160's own eight rows were repaired, and six of those ten WERE
            # the repairs.  A flagged step therefore poisons its whole chain, and so does an
            # unverified image choice (`tests/test_fem.py` line 323, where `8b84fe2` deliberately
            # re-aimed what `641ce0f` had chosen).  Both arm this, and it never clears: the
            # live anchor of a poisoned chain still needs a reader, it just cannot be
            # scored by following content the audit already knows went the wrong way.
            if broken:
                tally["  downstream of a flagged step -- needs a reader, not a score"] += 1
                continue
            claim = claim_text(c_prev, cited, a_old)
            if not claim:
                tally["  the previous step's line was blank or too short"] += 1
                continue
            landed = cs.line_at(commit, cited, a_new)
            if landed is not None and any(landed == c or similar(landed, c) for c in claim):
                if len(claim) > 1:
                    pre = claim[1]
                    which = "pre" if landed == pre or similar(landed, pre) else "post"
                    other = claim[0] if which == "pre" else pre
                    if not (landed == other or similar(landed, other)):
                        tally["  preserved -- but CHOSE an image"] += 1
                        choice.append((site, target, commit, which, live))
                        broken = True
                        continue
                tally["  the repair preserved the claim"] += 1
                continue
            hits = [j for j, l in enumerate(cs.blob(commit, cited) or [], 1) if l in claim]
            if len(hits) == 1:
                why = reaimed(f, lineno, cited, a_old, a_new, commit, c_prev)
                tally["  RE-AIMED at a named symbol" if why else "  WRONG DELTA"] += 1
                broken = not why
                (reaim if why else wrong).append(
                    (site, target, commit, a_old, a_new, hits[0],
                     why or claim[0].strip()[:44], live))
            elif not hits:
                tally["  the claim content is gone from the file"] += 1
            else:
                tally["  the claim content is ambiguous at the repair commit"] += 1

    print("\n=== THE REPAIR AUDIT: %d path citations ===\n" % len(seen))
    for k in ("never repaired", "repaired", "untrackable"):
        print("  %5d  %s" % (tally[k], k))
    print()
    for k in sorted(k for k in tally if k.startswith("  ")):
        print("  %5d  %s" % (tally[k], k.strip()))

    live = [w for w in wrong if w[7]]
    print("\n=== WRONG DELTA: %d steps, %d of them the citation's CURRENT anchor ===" %
          (len(wrong), len(live)))
    print("  the `CLAIM IS AT` column is a DETECTION, not a repair instruction -- for a")
    print("  dated quotation the repair is to RESTORE the old anchor.  Read the sentence.")
    for site, target, commit, a_old, a_new, hit, text, is_live in sorted(wrong):
        print("  %-29s %-30s %-8s %-16s :%-6d %+5d %s %s"
              % (site, target, commit, "%d->%d (%+d)" % (a_old, a_new, a_new - a_old),
                 hit, hit - a_new, "LIVE " if is_live else "     ", text))

    for name, group, tail in (
            ("PRESERVED BUT CHOSE AN IMAGE -- this audit's own blind spot, named", choice,
             "the two images differ; only the citing SENTENCE says which is right"),
            ("RE-AIMED at a symbol the citing line names -- excluded", reaim, "")):
        print("\n=== %s: %d ===" % (name, len(group)))
        if tail:
            print("  %s" % tail)
        for row in sorted(group):
            if len(row) == 5:
                print("  %-29s %-30s %-8s followed the %-4s image%s"
                      % (row[0], row[1], row[2], row[3], "  LIVE" if row[4] else ""))
            else:
                print("  %-29s %-30s %-8s %d->%d  now inside `%s`%s"
                      % (row[0], row[1], row[2], row[3], row[4], row[6],
                         "  LIVE" if row[7] else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--census", action="store_true", help="the blind spot only")
    ap.add_argument("--repairs", action="store_true", help="the repair audit only")
    args = ap.parse_args()
    owners, rows = scan()
    if not args.repairs:
        census(owners, rows)
    if not args.census:
        audit(owners, rows)


if __name__ == "__main__":
    main()
